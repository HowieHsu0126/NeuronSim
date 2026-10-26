"""Read angular displacement and paired recovery from the complete PFC cohort.

The saved 1 ms vectors are measured spike moments. Numerical direction
availability is distinct from scientific direction eligibility. No memory
threshold, recovery time, confidence interval or qualification is inferred.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .benchmark_tasks import validate_protocol
from .pfc_observation_resolution import directions
from .pfc_parameter_domain_quality import BLOCKS, EPISODES, SPLITS, intended_split, validate_plan
from .pfc_reference_quality import phase_edges
from .pfc_reference_quality_summary import CONDITIONS, paired_contrasts, require
from .pfc_temporal_readouts import lagged_angular_displacement
from .provenance import cpu_execution_provenance, freeze_json, sha256_file

SCHEMA = "pfc_parameter_task_dynamics_v1"
BIN_TICKS = 80
BIN_SECONDS = .001
DT_MS = .0125
BINS = 5000
N_E = 80000
N_I = 20000
LAGS_TICKS = (400, 800, 8000)  # Retain the established 5/10/100 ms diagnostics.
CONTROL_COMMANDS = "cue_and_distractor_assignment_amplitudes_nA_with_active_flags"


def validate_summary(report: dict) -> None:
    require(report.get("schema") == "pfc_parameter_domain_summary_v1"
        and report.get("status") == "COMPLETE3000_DESCRIPTIVE_PARAMETER_DOMAIN_COHORT"
        and report.get("complete_development_episodes") == EPISODES
        and report.get("parameter_circuits") == 100
        and report.get("blocks") == list(BLOCKS)
        and report.get("conditions") == list(CONDITIONS)
        and report.get("circuit_split_counts") == SPLITS
        and report.get("core_units") == N_E + N_I
        and report.get("dt_ms") == DT_MS and report.get("duration_ticks") == BINS * BIN_TICKS
        and report.get("observation_bin_ticks") == BIN_TICKS
        and report.get("observation_bin_ms") == 1.
        and all(report.get(key) is False for key in (
            "source_qualification_granted", "scientific_parameter_domain_qualified",
            "scientific_claim_qualified", "uncertainty_coverage_qualified", "scaling_qualified"))
        and report.get("production_episodes") == 0,
        "PFC_TASK_COMPLETE3000_DESCRIPTIVE_SUMMARY_REQUIRED")
    expected = {f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_{k:02d}"
                for c in range(100) for b in BLOCKS for k in range(10)}
    bindings = report.get("bindings", [])
    require(len(bindings) == EPISODES and {b.get("episode_id") for b in bindings} == expected,
        "PFC_TASK_COMPLETE_EPISODE_IDENTITIES_REQUIRED")
    observation = report.get("observation_map", {})
    require(observation.get("frozen") is True
        and observation.get("populations") == ["E", "I"]
        and observation.get("unit_order") == "all_E_then_all_I_ascending_source_indices"
        and observation.get("controls", {}).get("commands") == CONTROL_COMMANDS
        and observation.get("temporal_resolution", {}).get("scientific_bin_ticks") == BIN_TICKS,
        "PFC_TASK_FROZEN_OBSERVATION_MAP_REQUIRED")
    names = {f"circuit_{c:03d}_joint_activity.npz" for c in range(100)}
    require(names <= set(report.get("outputs", {})), "PFC_TASK_ALL100_JOINT_ARRAYS_REQUIRED")


def load_circuit(summary: Path, report: dict, circuit: int) -> dict[str, np.ndarray]:
    name = f"circuit_{circuit:03d}_joint_activity.npz"
    path = summary / name
    require(sha256_file(path) == report["outputs"][name]["sha256"],
        "PFC_TASK_JOINT_ARRAY_CHANGED")
    with np.load(path, allow_pickle=False) as source:
        for key, expected in (
            ("time_start_ticks", np.arange(BINS, dtype=np.int64) * BIN_TICKS),
            ("time_stop_ticks", np.arange(1, BINS + 1, dtype=np.int64) * BIN_TICKS),
            ("source_unit_ids", np.arange(N_E + N_I, dtype=np.int64)),
            ("field_unit_counts", np.r_[np.full(16, N_E // 16), N_I]),
            ("block_indices", np.asarray(BLOCKS, dtype=np.int64)),
        ):
            require(np.array_equal(source[key], expected), "PFC_TASK_CLOCK_OR_UNIT_IDENTITY_CHANGED")
        result = {}
        for k in range(10):
            prefix = f"condition_{k:02d}"
            for suffix in ("population_field_counts1ms", "vector_x1ms", "vector_y1ms"):
                key = prefix + "_" + suffix
                result[key] = source[key]
    result.update(load_control_transitions(report, circuit))
    return result


def load_control_transitions(report: dict, circuit: int) -> dict[str, np.ndarray]:
    """Read only the small, bound control reports for all three histories."""
    result = {}
    bindings = {item["episode_id"]: item for item in report["bindings"]}
    # These transitions include changes in source activation flags, even when
    # the assignment amplitude is zero. They are not per-neuron current changes.
    for k in range(10):
        transitions = []
        for block in BLOCKS:
            episode = f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{k:02d}"
            entry = bindings[episode]
            path = Path(entry["path"]) / "observation1ms/report.json"
            require(sha256_file(path) == entry["observation_report_sha256"],
                "PFC_TASK_CONTROL_REPORT_CHANGED")
            observed = json.loads(path.read_text())
            require(observed.get("identity", {}).get("episode_id") == episode
                and observed.get("condition", {}).get("condition_id") == CONDITIONS[k],
                "PFC_TASK_CONTROL_IDENTITY_CHANGED")
            require(observed.get("observation_map", {}).get("controls", {}).get("commands")
                == CONTROL_COMMANDS, "PFC_TASK_CONTROL_FEATURE_SEMANTICS_CHANGED")
            rows = observed["source_control_transitions"]
            require(all(item["boundary_aligned"] is True
                and type(item["transition_tick"]) is int and 0 < item["transition_tick"] < BINS * BIN_TICKS
                and item["transition_tick"] % BIN_TICKS == 0 for item in rows),
                "PFC_TASK_ALIGNED_COMMAND_TRANSITIONS_REQUIRED")
            transitions.append([item["transition_tick"] for item in rows])
        require(all(ticks == transitions[0] for ticks in transitions),
            "PFC_TASK_CONTROL_SCHEDULE_DIFFERS_BETWEEN_HISTORIES")
        result[f"condition_{k:02d}_source_control_transition_ticks"] = np.asarray(transitions[0], dtype=np.int64)
    return result


def expected_control_transition_ticks(protocol: dict, condition: dict) -> np.ndarray:
    """Reproduce transitions of the four exported command features.

    The export stores cue/distractor assignment amplitudes AND activation flags.
    A zero-amplitude active cue therefore retains both schedule boundaries.
    Source time uses seconds and start + duration in that order: .1 + .2 is
    later than .3, so cue deactivation occurs at operation301. This function
    does not infer transitions of the separately recorded per-neuron current.
    """
    schedules = (
        (protocol["stimulus_start_ms"], protocol["stimulus_duration_ms"]),
        (condition["distractor_start_ms"], condition["distractor_duration_ms"]),
    )
    operation = protocol["stimulus_operation_ms"]
    operation_seconds = operation * .001
    transitions = set()
    for onset, duration in schedules:
        if duration == 0:
            continue
        start, stop = onset * .001, onset * .001 + duration * .001
        previous = False
        for index in range(round(protocol["duration_ms"] / operation)):
            active = start <= index * operation_seconds < stop
            if active != previous:
                transitions.add(round(index * operation / protocol["dt_ms"]))
            previous = active
    return np.asarray(sorted(transitions), dtype=np.int64)


def validate_control_transitions(joint: dict[str, np.ndarray], protocol: dict) -> None:
    """Reject wrong condition schedules before any activity-derived outcome."""
    validate_protocol(protocol, "pfc_working_memory")
    require(protocol["dt_ms"] == DT_MS, "PFC_TASK_REFERENCE_CLOCK_REQUIRED")
    for k, condition in enumerate(protocol["conditions"]):
        observed = joint[f"condition_{k:02d}_source_control_transition_ticks"]
        expected = expected_control_transition_ticks(protocol, condition)
        require(observed.ndim == 1 and observed.dtype.kind in "iu"
            and np.array_equal(observed, expected),
            "PFC_TASK_CONDITION_CONTROL_SCHEDULE_MISMATCH:" + condition["condition_id"])


def measured_trajectories(fields: np.ndarray, x: np.ndarray, y: np.ndarray,
                          cue_degrees: float) -> dict[str, np.ndarray]:
    """Project additive vectors before pairing; retain silence and signed effects."""
    fields, x, y = map(np.asarray, (fields, x, y))
    require(fields.ndim == 3 and fields.shape[1:] == (BINS, 17)
        and fields.shape[0] > 0 and fields.dtype.kind in "iu" and np.all(fields >= 0)
        and x.shape == y.shape == fields.shape[:2]
        and np.isfinite(x).all() and np.isfinite(y).all() and math.isfinite(cue_degrees),
        "PFC_TASK_COMPLETE_FINITE_TRAJECTORIES_REQUIRED")
    e = fields[:, :, :16].sum(axis=2)
    i = fields[:, :, 16]
    length = np.hypot(x, y)
    require(np.all(length <= e + 64 * math.ulp(1.) * np.maximum(1, e)),
        "PFC_TASK_VECTOR_OUTSIDE_EVENT_SUPPORT")
    defined, angle = directions(e, x, y)
    resultant = np.full(e.shape, np.nan)
    np.divide(length, e, out=resultant, where=e > 0)
    cue = math.radians(cue_degrees)
    return dict(
        excitatory_rate_Hz=e / (N_E * BIN_SECONDS),
        inhibitory_rate_Hz=i / (N_I * BIN_SECONDS),
        target_directed_activity_Hz=(x * math.cos(cue) + y * math.sin(cue)) / (N_E * BIN_SECONDS),
        orthogonal_activity_Hz=(-x * math.sin(cue) + y * math.cos(cue)) / (N_E * BIN_SECONDS),
        resultant_length=resultant,
        numerical_direction_defined=defined,
        decoded_angle_deg=angle,
        circular_cue_error_deg=(angle - cue_degrees + 180) % 360 - 180,
    )


def angular_rows(fields: np.ndarray, x: np.ndarray, y: np.ndarray, measured: dict,
                 block_position: int, edges: list[int]) -> list[dict]:
    b = block_position
    counts = fields[b, :, :16].sum(axis=1)
    rows = [dict(start_tick=t * BIN_TICKS, stop_tick=(t + 1) * BIN_TICKS,
        excitatory_spikes=int(counts[t]), vector_x_spikes=float(x[b, t]), vector_y_spikes=float(y[b, t]),
        direction_defined=bool(measured["numerical_direction_defined"][b, t]),
        decoded_angle_deg=float(measured["decoded_angle_deg"][b, t])
            if measured["numerical_direction_defined"][b, t] else None) for t in range(BINS)]
    return lagged_angular_displacement(rows, edges, bin_ticks=BIN_TICKS, lag_ticks=list(LAGS_TICKS))


def describe_circuit(circuit: int, joint: dict[str, np.ndarray], protocol: dict
                     ) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    """Retain each history; conditions and time bins do not add replicates."""
    validate_control_transitions(joint, protocol)
    edges = phase_edges(protocol)
    trajectories, angular, phases = {}, [], []
    condition_values = []
    for k, condition in enumerate(CONDITIONS):
        prefix = f"condition_{k:02d}"
        fields, x, y = (joint[prefix + "_" + suffix] for suffix in (
            "population_field_counts1ms", "vector_x1ms", "vector_y1ms"))
        require(fields.shape[0] == len(BLOCKS), "PFC_TASK_THREE_DISTINCT_HISTORY_ROWS_REQUIRED")
        values = measured_trajectories(fields, x, y, protocol["stimulus_center_deg"])
        for name, array in values.items():
            trajectories[prefix + "_" + name] = array
        condition_values.append(np.stack([values[name] for name in (
            "target_directed_activity_Hz", "orthogonal_activity_Hz",
            "excitatory_rate_Hz", "inhibitory_rate_Hz")], axis=-1))
        for b, block in enumerate(BLOCKS):
            identity = dict(circuit_index=circuit, split=intended_split(circuit), block_index=block,
                condition_id=condition,
                episode_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{k:02d}")
            angular.extend(dict(identity, lag_ms=row["lag_ticks"] * DT_MS, **row)
                           for row in angular_rows(fields, x, y, values, b, edges))
            for p, (first, stop) in enumerate(zip(edges[:-1], edges[1:])):
                part = slice(first // BIN_TICKS, stop // BIN_TICKS)
                counts = fields[b, part, :16].sum(axis=1)
                defined = values["numerical_direction_defined"][b, part]
                phases.append(dict(identity, phase_index=p, start_tick=first, stop_tick=stop,
                    total_bins=len(counts), silent_E_bins=int(np.count_nonzero(counts == 0)),
                    numerical_direction_defined_bins=int(defined.sum()),
                    cancelled_non_silent_bins=int(np.count_nonzero((counts > 0) & ~defined)),
                    mean_target_directed_activity_Hz=float(values["target_directed_activity_Hz"][b, part].mean()),
                    mean_orthogonal_activity_Hz=float(values["orthogonal_activity_Hz"][b, part].mean()),
                    mean_excitatory_rate_Hz=float(values["excitatory_rate_Hz"][b, part].mean()),
                    mean_inhibitory_rate_Hz=float(values["inhibitory_rate_Hz"][b, part].mean()),
                    direction_scientific_eligibility="UNQUALIFIED_NO_FROZEN_ACTIVITY_OR_RESULTANT_GATE"))
    removal_ticks = []
    for k in range(10):
        key = f"condition_{k:02d}_source_control_transition_ticks"
        require(all(int(tick) in edges for tick in joint[key]),
            "PFC_TASK_COMMAND_TRANSITION_OUTSIDE_FROZEN_PHASES")
        trajectories[key] = joint[key]
    for k in range(1, 5):
        cue_ticks = joint[f"condition_{k:02d}_source_control_transition_ticks"]
        omission_ticks = joint[f"condition_{k + 5:02d}_source_control_transition_ticks"]
        require(len(cue_ticks) == len(omission_ticks) == 4
            and np.array_equal(cue_ticks, omission_ticks)
            and np.all(np.diff(cue_ticks) > 0), "PFC_TASK_PAIRED_DISTRACTOR_TRANSITIONS_REQUIRED")
        removal_ticks.append(int(cue_ticks[-1]))
    require(np.array_equal(joint["condition_00_source_control_transition_ticks"],
                           np.asarray(protocol["cue_active_operation_indices"]) * BIN_TICKS)
        and np.array_equal(joint["condition_05_source_control_transition_ticks"],
                           joint["condition_00_source_control_transition_ticks"]),
        "PFC_TASK_CUE_OMISSION_CONTROL_TRANSITIONS_REQUIRED")
    relative_ticks = np.arange(BINS)[None, :] * BIN_TICKS - np.asarray(removal_ticks)[:, None]
    trajectories.update(distractor_removal_ticks=np.asarray(removal_ticks, dtype=np.int64),
        time_since_distractor_removal_ms=relative_ticks * DT_MS,
        post_distractor_removal=relative_ticks >= 0)
    # Last axis is condition; use the established cue/omission pairing exactly.
    values = np.stack(condition_values, axis=-1)
    cue, interaction = paired_contrasts(values)
    trajectories.update(
        cue_minus_omission=cue,
        distractor_difference_of_differences=interaction,
        contrast_metric_names=np.array(["target_directed_activity_Hz", "orthogonal_activity_Hz",
                                       "excitatory_rate_Hz", "inhibitory_rate_Hz"]),
        block_indices=np.asarray(BLOCKS, dtype=np.int64),
        time_start_ticks=np.arange(BINS, dtype=np.int64) * BIN_TICKS,
        time_stop_ticks=np.arange(1, BINS + 1, dtype=np.int64) * BIN_TICKS,
        phase_edges_ticks=np.asarray(edges, dtype=np.int64),
    )
    return trajectories, angular, phases


def append_rows(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "PFC_TASK_EMPTY_OUTCOME_TABLE")
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_inputs(summary: Path, prepared: Path) -> tuple[dict, dict]:
    report = json.loads((summary / "report.json").read_text())
    validate_summary(report)
    cpu_execution_provenance()
    preparation_path = prepared / "execution.json"
    require(sha256_file(preparation_path) == report["preparation_sha256"],
        "PFC_TASK_FROZEN_PREPARATION_CHANGED")
    plan = json.loads(preparation_path.read_text())
    validate_plan(plan)
    protocol = plan["task_protocol"]
    require(protocol.get("frozen") is True and protocol.get("dt_ms") == DT_MS
        and protocol.get("duration_ms") == 5000. and protocol.get("stimulus_operation_ms") == 1.
        and [item["condition_id"] for item in protocol["conditions"]] == list(CONDITIONS),
        "PFC_TASK_FROZEN_TASK_PROTOCOL_REQUIRED")
    validate_protocol(protocol, "pfc_working_memory")
    return report, protocol


def analyze(summary: Path, prepared: Path, output: Path) -> dict:
    report_path = summary / "report.json"
    preparation_path = prepared / "execution.json"
    report, protocol = load_inputs(summary, prepared)
    execution = cpu_execution_provenance()
    require(not output.exists(), "PFC_TASK_OUTCOME_OUTPUT_EXISTS")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pfc-task-dynamics-", dir="/tmp") as temp:
        work = Path(temp)
        outputs = {}
        for circuit in range(100):
            joint = load_circuit(summary, report, circuit)
            trajectories, angular, phases = describe_circuit(circuit, joint, protocol)
            name = f"circuit_{circuit:03d}_task_trajectories.npz"
            np.savez_compressed(work / name, **trajectories)
            outputs[name] = dict(sha256=sha256_file(work / name), circuit_index=circuit,
                                 split=intended_split(circuit))
            append_rows(work / "lagged_angular_displacement.csv", angular)
            append_rows(work / "phase_trajectory_summary.csv", phases)
        for name in ("lagged_angular_displacement.csv", "phase_trajectory_summary.csv"):
            outputs[name] = dict(sha256=sha256_file(work / name))
        result = dict(schema=SCHEMA, status="COMPLETE3000_DESCRIPTIVE_TASK_DYNAMICS",
            execution=execution,
            source_summary=str(report_path.resolve()), source_summary_sha256=sha256_file(report_path),
            preparation_sha256=sha256_file(preparation_path),
            complete_development_episodes=EPISODES, parameter_circuits=100, blocks=list(BLOCKS),
            conditions=list(CONDITIONS), circuit_split_counts=SPLITS, observation_bin_ms=1.,
            outputs=outputs, protocol=protocol,
            control_transition_semantics=CONTROL_COMMANDS,
            recovery_origin="observed_distractor_command_deactivation",
            contrast_axes=["history", "source_time_bin", "metric", "condition"],
            cue_contrast_conditions=list(CONDITIONS[:5]),
            interaction_conditions=list(CONDITIONS[1:5]),
            recovery_axis="absolute source time and time since recorded distractor removal; complete pre/post trajectory retained; no recovery-time threshold",
            angular_scope="5/10/100ms principal increments of1ms measured vectors within source phases; dependent temporal pairs",
            undefined_rule="silent/cancelled directions stay undefined; any missing endpoint withholds its angular summary",
            weak_direction_rule="retain resultant lengths; numerical direction does not qualify scientific direction or memory",
            inference_unit="joint initial/external history nested in parameter circuit; ten paired conditions",
            calibration_splits=["train", "calibration"], calibration_performed=False,
            protected_circuits_used_for_calibration=False,
            uncertainty="descriptive trajectories only; no confidence intervals,pvalues,source/scaling or memory PASS",
            source_qualification_granted=False, scaling_qualified=False,
            scientific_parameter_domain_qualified=False, scientific_claim_qualified=False,
            memory_success_qualified=False, production_episodes=0, source_simulations=0,
            elapsed_seconds=time.perf_counter() - started,
            implementation_sha256=sha256_file(Path(__file__)))
        freeze_json(work / "report.json", result)
        shutil.copytree(work, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.summary, args.prepared, args.output)
    print(json.dumps({key: report[key] for key in (
        "status", "complete_development_episodes", "source_simulations", "production_episodes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

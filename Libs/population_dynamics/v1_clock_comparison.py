"""Describe one V1 clock contrast after admitting two independent native replays.

This consumes archived outputs. It does not run NEST, change the same-clock
comparator, estimate uncertainty from neurons, or qualify convergence/fidelity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file

DURATION_MS = 3000.0
REFERENCE_DT_MS = 0.25
FINE_DT_MS = 0.125
SOURCE_TYPE_CSV = "recurrent/primary/glif/v1_node_types.csv"
NATIVE_ARRAYS = ("native/cortical_state.npz", "native/cortical_spikes.npz", "native/input_events.npz")
CONNECTION_ARRAYS = ("native/connection_source_batches.npz", "native/connection_readback.npz")
REFERENCE_STATUSES = {
    "diagnostic_pulses": ("POINTNET_NATIVE_DIAGNOSTIC_COMPLETE", "PASS_POINTNET_NATIVE_DIAGNOSTIC"),
    "constructed_visual_inputs": ("POINTNET_NATIVE_VISUAL_INPUTS_COMPLETE_DEVELOPMENT",
                                  "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT"),
}
FINE_STATUS = "POINTNET_NATIVE_TEMPORAL_SENSITIVITY_COMPLETE_DEVELOPMENT"


def required_files(report: dict) -> tuple[str, ...]:
    """Explicit small staging set; raw graph files are bound, never recopied."""
    arrays = report["array_artifacts"]
    expected = set(NATIVE_ARRAYS) | (set(CONNECTION_ARRAYS) if report["connections"]["full_native_tuple_readback"] else set())
    if len(arrays) != len(expected) or set(arrays) != expected:
        raise ValueError("CLOCK_NATIVE_ARTIFACT_SET_DIFFERS")
    files = ["delivery.json", "primary/report.json", "replay/report.json", "comparison/report.json", SOURCE_TYPE_CSV]
    files += [prefix + "/" + name for prefix in ("primary", "replay") for name in (*arrays, "native/parameters.json")]
    if report.get("clock_protocol") is not None:
        files.append("clock_protocol.resolved.yaml")
    return tuple(files)


def admit_archive(archive: Path, *, fine: bool) -> dict:
    delivery = json.loads((archive / "delivery.json").read_text())
    if delivery.get("exit_code") != 0 or not isinstance(delivery.get("job_id"), str):
        raise ValueError("CLOCK_COMPLETED_DELIVERY_REQUIRED")
    records = {}
    for row in delivery["files"]:
        if row["path"] in records:
            raise ValueError("CLOCK_DUPLICATE_DELIVERY_PATH")
        records[row["path"]] = row
    consumed = {"delivery.json": dict(bytes=(archive / "delivery.json").stat().st_size,
                                      sha256=sha256_file(archive / "delivery.json"))}

    def bind(name):
        if name in consumed:
            return consumed[name]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name not in records:
            raise ValueError("CLOCK_DELIVERED_FILE_MISSING:" + name)
        record = records[name]
        actual = dict(bytes=(archive / name).stat().st_size, sha256=sha256_file(archive / name))
        if actual != {key: record[key] for key in ("bytes", "sha256")}:
            raise ValueError("CLOCK_DELIVERED_FILE_CHANGED:" + name)
        consumed[name] = actual
        return actual

    reports = []
    for name in ("primary/report.json", "replay/report.json", "comparison/report.json"):
        bind(name)
        reports.append(json.loads((archive / name).read_text()))
    primary, replay, comparison = reports
    try:
        expected_status, expected_delivery = REFERENCE_STATUSES[primary["input_mode"]]
    except KeyError as error:
        raise ValueError("CLOCK_INPUT_MODE_NOT_SUPPORTED") from error
    if fine:
        expected_status, expected_delivery = FINE_STATUS, "PASS_POINTNET_NATIVE_TEMPORAL_SENSITIVITY_DEVELOPMENT"
    if delivery.get("status") != expected_delivery:
        raise ValueError("CLOCK_NATIVE_DELIVERY_STATUS_DIFFERS")
    for row in (primary, replay):
        if (row.get("status") != expected_status or row.get("exit_code") != 0
                or row.get("runtime", {}).get("slurm", {}).get("SLURM_JOB_ID") != delivery["job_id"]
                or row.get("mode") != delivery.get("mode") or row.get("mode") not in ("canary", "full")
                or row.get("simulated_ms") != DURATION_MS or row.get("native_nest_simulation") is not True
                or row.get("native_parameters_and_initial_state_exact") is not True
                or row.get("input_readback", {}).get("tics_per_ms") != 1000.0
                or row.get("scientific_qualification") is not False or row.get("production_episodes") != 0):
            raise ValueError("CLOCK_COMPLETE_NATIVE_EXECUTION_REQUIRED")
        if row["bindings"]["point_config"]["source_clock"] != dict(duration_ms=DURATION_MS, resolution_ms=REFERENCE_DT_MS):
            raise ValueError("CLOCK_AUTHOR_CONFIGURATION_DIFFERS")
        if fine:
            if row.get("execution_clock") != dict(duration_ms=DURATION_MS, resolution_ms=FINE_DT_MS,
                                                   tics_per_ms=1000.0, tics_per_step=125):
                raise ValueError("CLOCK_FINE_EXECUTION_CLOCK_DIFFERS")
        elif row.get("clock_protocol") is not None:
            raise ValueError("CLOCK_SOURCE_REFERENCE_REQUIRED")
    if (comparison.get("status") != "POINTNET_NATIVE_REPLAY_EQUAL" or comparison.get("exit_code") != 0
            or comparison.get("all_native_outputs_equal") is not True
            or comparison.get("independent_processes") is not True
            or comparison.get("binding_or_runtime_differences") != []
            or comparison.get("left_report_sha256") != consumed["primary/report.json"]["sha256"]
            or comparison.get("right_report_sha256") != consumed["replay/report.json"]["sha256"]
            or (primary["hostname"], primary["process_id"]) == (replay["hostname"], replay["process_id"])):
        raise ValueError("CLOCK_BOUND_INDEPENDENT_REPLAY_REQUIRED")
    for field in ("bindings", "array_artifacts", "connections", "input_mode", "input_event_schema", "source_denominators"):
        if primary[field] != replay[field]:
            raise ValueError("CLOCK_REFERENCE_REPLAY_CONTEXT_DIFFERS:" + field)
    compared = comparison.get("files", [])
    expected_compared = set(primary["array_artifacts"]) | {"native/parameters.json"}
    if (len(compared) != len(expected_compared) or {row["path"] for row in compared} != expected_compared
            or any(row.get("equal") is not True for row in compared)):
        raise ValueError("CLOCK_COMPLETE_NATIVE_REPLAY_COMPARISON_REQUIRED")
    for name in required_files(primary):
        bind(name)
    source_record = primary["bindings"]["recurrent"]["files"]["primary/glif/v1_node_types.csv"]
    if consumed[SOURCE_TYPE_CSV] != source_record:
        raise ValueError("CLOCK_POPULATION_METADATA_BINDING_DIFFERS")
    if fine and consumed["clock_protocol.resolved.yaml"]["sha256"] != primary["clock_protocol"]["sha256"]:
        raise ValueError("CLOCK_PROTOCOL_BINDING_DIFFERS")
    return dict(archive=str(archive), delivery=delivery, report=primary, replay_report=replay, consumed=consumed)


def require_pair(reference: dict, fine: dict) -> None:
    a, b = reference["report"], fine["report"]
    for field in ("bindings", "original_config_sha256", "source_denominators", "mode", "input_mode", "input_event_schema",
                  "cortical_units", "realized_glif_types", "kernel_rng_seed", "nest_version", "local_num_threads"):
        if a[field] != b[field]:
            raise ValueError("CLOCK_SOURCE_CONTEXT_DIFFERS:" + field)
    if a["wheel"]["sha256"] != b["wheel"]["sha256"]:
        raise ValueError("CLOCK_NATIVE_WHEEL_DIFFERS")
    context, connection = b["temporal_sensitivity"], b["connections"]
    digest = a["connections"]["source_batch_sha256"]
    if (context.get("reference_primary_report_sha256") != reference["consumed"]["primary/report.json"]["sha256"]
            or context.get("reference_delivery_sha256") != reference["consumed"]["delivery.json"]["sha256"]
            or context.get("reference_job_id") != reference["delivery"]["job_id"]
            or context.get("unchanged_upstream_bindings") is not True
            or context.get("reference_clock_source_batch_sha256") != digest
            or connection.get("reference_clock_source_batch_sha256") != digest
            or connection.get("reference_clock_source_batch_exact") is not True
            or connection.get("reference_clock_resolution_ms") != REFERENCE_DT_MS
            or connection.get("source_batch_resolution_ms") != FINE_DT_MS):
        raise ValueError("CLOCK_CONSUMED_REFERENCE_IDENTITY_DIFFERS")


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def integer_vector(values, name: str, *, unique=False) -> np.ndarray:
    values = np.asarray(values)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0)
            or (len(values) and int(values.max()) > np.iinfo(np.int64).max)):
        raise ValueError("CLOCK_INTEGER_DOMAIN_INVALID:" + name)
    result = values.astype(np.int64, copy=False)
    if unique and len(np.unique(result)) != len(result):
        raise ValueError("CLOCK_DUPLICATE_IDENTITY:" + name)
    return result


def align_state(values: dict) -> dict:
    ids = integer_vector(values["source_id"], "cortical source_id", unique=True)
    if not len(ids):
        raise ValueError("CLOCK_EMPTY_CORTICAL_STATE")
    order = np.argsort(ids)
    result = {}
    for key in ("source_id", "node_type_id", "native_id", "inverse_source_id", "spike_count"):
        vector = integer_vector(values[key], key, unique=key == "native_id")
        if len(vector) != len(ids):
            raise ValueError("CLOCK_CORTICAL_DENOMINATOR_DIFFERS")
        result[key] = vector[order]
    if np.any(result["native_id"] == 0) or not np.array_equal(result["source_id"], result["inverse_source_id"]):
        raise ValueError("CLOCK_CORTICAL_INVERSE_IDENTITY_DIFFERS")
    for key in ("initial_vm_mV", "terminal_vm_mV"):
        vector = np.asarray(values[key])
        if vector.shape != ids.shape or vector.dtype.kind != "f" or not np.isfinite(vector).all():
            raise ValueError("CLOCK_MEMBRANE_STATE_INVALID:" + key)
        result[key] = vector[order]
    return result


def align_spikes(values: dict, state: dict, dt_ms: float) -> dict:
    ids = integer_vector(values["source_id"], "spike source_id")
    native = integer_vector(values["native_id"], "spike native_id")
    times = np.asarray(values["time_ms"])
    if (times.shape != ids.shape or native.shape != ids.shape or times.dtype.kind != "f"
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > DURATION_MS)
            or not np.array_equal(times, np.rint(times / dt_ms) * dt_ms)):
        raise ValueError("CLOCK_SPIKE_TIME_DOMAIN_DIFFERS")
    positions = np.searchsorted(state["source_id"], ids)
    if (np.any(positions >= len(state["source_id"])) or not np.array_equal(state["source_id"][positions], ids)
            or not np.array_equal(state["native_id"][positions], native)):
        raise ValueError("CLOCK_SPIKE_SOURCE_IDENTITY_DIFFERS")
    counts = np.bincount(positions, minlength=len(state["source_id"]))
    if not np.array_equal(counts, state["spike_count"]):
        raise ValueError("CLOCK_SPIKE_COUNT_READBACK_DIFFERS")
    return dict(unit_index=positions, time_ms=times)


def benchmark_window_counts(events: dict, units: int) -> dict[str, np.ndarray]:
    """Retain the benchmark's source and delivered-command interval semantics."""
    times, indices = events["time_ms"], events["unit_index"]
    masks = dict(full=np.ones(len(times), dtype=bool), source_baseline=times <= 500.0,
                 source_response=times > 500.0, pre_command=times < 501.0,
                 command_response=(times >= 501.0) & (times < DURATION_MS),
                 terminal=times == DURATION_MS)
    return {name: np.bincount(indices[keep], minlength=units) for name, keep in masks.items()}


def canonical_inputs(values: dict, schema: str, dt_ms: float) -> dict:
    ids = integer_vector(values["source_id"], "input source_id")
    populations = np.asarray(values["population"])
    native = integer_vector(values["native_id"], "input native_id", unique=True)
    inverse = integer_vector(values["inverse_source_id"], "input inverse_source_id")
    counts = integer_vector(values["source_event_counts"], "input source_event_counts")
    precise = np.asarray(values["precise_times"])
    if (populations.shape != ids.shape or populations.dtype.kind not in "US" or len(native) != len(ids)
            or counts.shape != ids.shape or not np.array_equal(inverse, ids) or np.any(native == 0)
            or precise.shape != ids.shape or precise.dtype.kind != "b" or not precise.all()
            or not np.isin(populations, ["lgn", "bkg"]).all()):
        raise ValueError("CLOCK_INPUT_UNIT_SCHEMA_DIFFERS")
    order = np.lexsort((ids, populations))
    if len(ids) > 1 and np.any((ids[order][1:] == ids[order][:-1]) & (populations[order][1:] == populations[order][:-1])):
        raise ValueError("CLOCK_DUPLICATE_INPUT_IDENTITY")
    rank = np.empty(len(ids), dtype=np.int64); rank[order] = np.arange(len(ids))
    requested = np.asarray(values["requested_times_ms"])
    if schema == "diagnostic_pulse_matrix_v1":
        if requested.ndim != 2 or requested.shape[0] != len(ids) or not np.all(counts == requested.shape[1]):
            raise ValueError("CLOCK_DIAGNOSTIC_INPUT_SHAPE_DIFFERS")
        units = np.repeat(np.arange(len(ids)), requested.shape[1])
    elif schema == "packed_source_events_v1":
        units = integer_vector(values["source_event_unit"], "input event unit")
        if (requested.ndim != 1 or units.shape != requested.shape or np.any(units >= len(ids))
                or not np.array_equal(np.bincount(units, minlength=len(ids)), counts)):
            raise ValueError("CLOCK_PACKED_INPUT_SHAPE_DIFFERS")
    else:
        raise ValueError("CLOCK_INPUT_EVENT_SCHEMA_NOT_SUPPORTED")
    native_times = np.asarray(values["native_times_ms"])
    steps, offsets = np.asarray(values["encoded_steps"]), np.asarray(values["encoded_offsets_ms"])
    if (requested.dtype.kind != "f" or not np.isfinite(requested).all()
            or np.any(requested <= 0) or np.any(requested > DURATION_MS)
            or any(array.shape != requested.shape for array in (native_times, steps, offsets))
            or native_times.dtype.kind != "f" or not np.isfinite(native_times).all()
            or np.any(native_times <= 0) or np.any(native_times > DURATION_MS)
            or steps.dtype.kind not in "iu" or np.any(steps <= 0)
            or offsets.dtype.kind != "f" or not np.isfinite(offsets).all()
            or np.any(offsets < 0) or np.any(offsets >= dt_ms)):
        raise ValueError("CLOCK_NATIVE_INPUT_TIME_DOMAIN_DIFFERS")
    # Reuse the admitted generator encoding; do not demand equality of encoded
    # steps/offsets across clocks. NEST recorder decoding uses integer tics.
    decoded = .001 * (steps * int(1000 * dt_ms)) - offsets
    if not np.array_equal(decoded, native_times):
        raise ValueError("CLOCK_NATIVE_INPUT_DECODING_DIFFERS")
    event_order = np.lexsort((requested.reshape(-1), rank[units]))
    return dict(population=populations[order], source_id=ids[order], counts=counts[order],
                event_unit=rank[units][event_order], requested_times_ms=requested.reshape(-1)[event_order])


def source_populations(path: Path, type_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter=" ", skipinitialspace=True))
    mapping = {}
    for row in rows:
        identity = int(row["node_type_id"])
        if identity in mapping or not row["pop_name"] or row["population"] != "v1":
            raise ValueError("CLOCK_SOURCE_POPULATION_TABLE_INVALID")
        mapping[identity] = row["pop_name"]
    if not set(map(int, type_ids)).issubset(mapping):
        raise ValueError("CLOCK_SOURCE_POPULATION_TYPE_MISSING")
    names = np.asarray(list(dict.fromkeys(mapping.values())))
    index = {name: i for i, name in enumerate(names)}
    codes = np.asarray([index[mapping[int(identity)]] for identity in type_ids], dtype=np.int64)
    return names, codes


def require_same_requested_inputs(reference: dict, fine: dict) -> None:
    for name in reference:
        if reference[name].dtype != fine[name].dtype or not np.array_equal(reference[name], fine[name]):
            raise ValueError("CLOCK_REQUESTED_INPUTS_DIFFER:" + name)


def cumulative_counts(spikes: dict, population_codes: np.ndarray, population_count: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(int(DURATION_MS / REFERENCE_DT_MS) + 1, dtype=np.float64) * REFERENCE_DT_MS
    codes = population_codes[spikes["unit_index"]]
    curves = np.zeros((population_count, len(grid)), dtype=np.int64)
    for code in range(population_count):
        curves[code] = np.searchsorted(np.sort(spikes["time_ms"][codes == code]), grid, side="right")
    return grid, curves


def describe(values: np.ndarray) -> dict:
    if not len(values):
        return dict(n=0, minimum=None, maximum=None, mean=None, median=None, mean_absolute=None,
                    maximum_absolute=None, root_mean_square=None)
    values = np.asarray(values, dtype=np.float64)
    return dict(n=len(values), minimum=float(values.min()), maximum=float(values.max()), mean=float(values.mean()),
                median=float(np.median(values)), mean_absolute=float(np.abs(values).mean()),
                maximum_absolute=float(np.abs(values).max()), root_mean_square=float(np.sqrt(np.mean(values ** 2))))


def summarize_units(a: dict, b: dict, selected: np.ndarray) -> dict:
    counts_a, counts_b = a["spike_count"][selected], b["spike_count"][selected]
    volts_a, volts_b = a["terminal_vm_mV"][selected], b["terminal_vm_mV"][selected]
    return dict(units=len(counts_a), reference_spikes=int(counts_a.sum()), fine_spikes=int(counts_b.sum()),
        spike_count_difference=int(counts_b.sum() - counts_a.sum()),
        reference_silent_units=int(np.count_nonzero(counts_a == 0)), fine_silent_units=int(np.count_nonzero(counts_b == 0)),
        silent_to_active_units=int(np.count_nonzero((counts_a == 0) & (counts_b > 0))),
        active_to_silent_units=int(np.count_nonzero((counts_a > 0) & (counts_b == 0))),
        changed_spike_count_units=int(np.count_nonzero(counts_b != counts_a)),
        spike_counts=dict(reference=describe(counts_a), fine=describe(counts_b), difference=describe(counts_b - counts_a)),
        terminal_voltage_mV=dict(reference=describe(volts_a), fine=describe(volts_b), difference=describe(volts_b - volts_a)))


def analyze(reference_archive: Path, fine_archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="FAILED_V1_CLOCK_COMPARISON", exit_code=1, started_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=sha256_file(Path(__file__)), scientific_qualification=False, production_episodes=0,
        scope="One paired whole-model temporal-sensitivity diagnostic; no convergence, equivalence, CI, or significance claim.",
        independent_circuit_replicates=1, independent_same_clock_replays_per_condition=1,
        voltage_observation="Initial and terminal V_m only; no voltage trajectories.")
    try:
        reference = admit_archive(reference_archive, fine=False)
        fine = admit_archive(fine_archive, fine=True)
        report["inputs"] = {key: dict(archive=value["archive"], job_id=value["delivery"]["job_id"], consumed=value["consumed"])
                            for key, value in (("reference", reference), ("fine", fine))}
        require_pair(reference, fine)
        states, spikes, inputs, parameters = [], [], [], []
        for admitted, dt in ((reference, REFERENCE_DT_MS), (fine, FINE_DT_MS)):
            path = Path(admitted["archive"]) / "primary"
            state = align_state(load_arrays(path / "native/cortical_state.npz"))
            events = align_spikes(load_arrays(path / "native/cortical_spikes.npz"), state, dt)
            source_inputs = canonical_inputs(load_arrays(path / "native/input_events.npz"), admitted["report"]["input_event_schema"], dt)
            params = json.loads((path / "native/parameters.json").read_text())
            r = admitted["report"]
            if (len(state["source_id"]) != r["cortical_units"] or len(np.unique(state["node_type_id"])) != r["realized_glif_types"]
                    or len(events["time_ms"]) != r["cortical_spike_events"]
                    or int(np.count_nonzero(state["spike_count"] == 0)) != r["silent_cortical_units"]
                    or len(source_inputs["source_id"]) != r["input_readback"]["generators"]
                    or len(source_inputs["requested_times_ms"]) != r["input_readback"]["events"]):
                raise ValueError("CLOCK_REPORT_DENOMINATOR_DIFFERS")
            for identity in np.unique(state["node_type_id"]):
                row = params[str(identity)]
                selected = state["node_type_id"] == identity
                if (row.get("all_units_exact") is not True or row["units"] != int(selected.sum())
                        or not np.all(state["initial_vm_mV"][selected] == row["parameters"]["V_m"])):
                    raise ValueError("CLOCK_INITIAL_PHYSICAL_PARAMETER_DIFFERS")
            states.append(state); spikes.append(events); inputs.append(source_inputs); parameters.append(params)
        a, b = states
        for name in ("source_id", "node_type_id", "initial_vm_mV"):
            if not np.array_equal(a[name], b[name]):
                raise ValueError("CLOCK_CORTICAL_INITIAL_IDENTITY_DIFFERS:" + name)
        if parameters[0] != parameters[1]:
            raise ValueError("CLOCK_PHYSICAL_PARAMETERS_DIFFER")
        require_same_requested_inputs(*inputs)
        names, codes = source_populations(reference_archive / SOURCE_TYPE_CSV, a["node_type_id"])
        if len(names) != 21:
            raise ValueError("CLOCK_ALL_21_SOURCE_POPULATIONS_REQUIRED")
        grid, reference_curves = cumulative_counts(spikes[0], codes, len(names))
        _, fine_curves = cumulative_counts(spikes[1], codes, len(names))
        differences = fine_curves - reference_curves
        per_population = []
        for code, name in enumerate(names):
            selected = codes == code
            summary = summarize_units(a, b, selected)
            if (reference_curves[code, -1] != summary["reference_spikes"]
                    or fine_curves[code, -1] != summary["fine_spikes"]):
                raise ValueError("CLOCK_CUMULATIVE_ENDPOINT_COUNT_DIFFERS")
            per_population.append(dict(population=str(name), **summary,
                maximum_absolute_cumulative_count_difference=int(np.abs(differences[code]).max())))
        window_arrays = {}
        for label, events, state in zip(("reference", "fine"), spikes, states):
            windows = benchmark_window_counts(events, len(state["source_id"]))
            if (not np.array_equal(windows["full"], state["spike_count"])
                    or not np.array_equal(windows["full"], windows["source_baseline"] + windows["source_response"])
                    or not np.array_equal(windows["full"], windows["pre_command"] + windows["command_response"] + windows["terminal"])):
                raise ValueError("CLOCK_BENCHMARK_WINDOW_COUNTS_DIFFER")
            window_arrays.update({label + "_" + name + "_counts": value for name, value in windows.items()})
        np.savez_compressed(output / "per_unit.npz", **window_arrays,
            source_id=a["source_id"], node_type_id=a["node_type_id"],
            population=names[codes], initial_vm_mV=a["initial_vm_mV"],
            reference_spike_count=a["spike_count"], fine_spike_count=b["spike_count"],
            spike_count_difference=b["spike_count"]-a["spike_count"],
            reference_terminal_vm_mV=a["terminal_vm_mV"], fine_terminal_vm_mV=b["terminal_vm_mV"],
            terminal_vm_difference_mV=b["terminal_vm_mV"]-a["terminal_vm_mV"])
        np.savez_compressed(output / "population_cumulative_counts.npz", population=names, time_ms=grid,
            units=np.bincount(codes, minlength=len(names)), reference_counts=reference_curves,
            fine_counts=fine_curves, difference=differences)
        with (output / "population_cumulative_counts.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["population", "time_ms", "reference_cumulative_spikes", "fine_cumulative_spikes", "difference"])
            for code, name in enumerate(names):
                writer.writerows(zip(np.repeat(name, len(grid)), grid, reference_curves[code], fine_curves[code], differences[code]))
        report.update(status="V1_CLOCK_COMPARISON_COMPLETE_DEVELOPMENT", exit_code=0,
            mode=reference["report"]["mode"], reference_resolution_ms=REFERENCE_DT_MS, fine_resolution_ms=FINE_DT_MS,
            source_bindings_and_requested_inputs_exact=True, physical_parameters_and_initial_states_exact=True,
            all_units=summarize_units(a, b, np.ones(len(codes), dtype=bool)), populations=per_population,
            benchmark_windows=dict(source_baseline="[0,500] ms", source_response="(500,3000] ms",
                pre_command="[0,501) ms", command_response="[501,3000) ms", terminal="3000 ms"),
            population_authority=SOURCE_TYPE_CSV, common_grid=dict(start_ms=0.0, stop_ms=DURATION_MS,
                step_ms=REFERENCE_DT_MS, samples=len(grid), cumulative_rule="Number of events at times <= t; terminal events included."),
            execution_timing={label: {phase: {key: admitted[phase][key] for key in
                ("hostname", "setup_seconds", "simulation_and_output_seconds", "elapsed_seconds")}
                for phase in ("report", "replay_report")} for label, admitted in (("reference", reference), ("fine", fine))},
            timing_interpretation="Archived simulation_and_output_seconds includes module initialization and connection verification; it is not pure integration time. Different hosts may have been used; no speedup is inferred.")
        report["artifacts"] = {name: dict(bytes=(output / name).stat().st_size, sha256=sha256_file(output / name))
                               for name in ("per_unit.npz", "population_cumulative_counts.npz", "population_cumulative_counts.csv")}
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["analysis_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime"] = runtime_manifest({}, list(sys.argv), report["exit_code"])
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-archive", "fine-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    report = analyze(**vars(parser.parse_args(argv)))
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "all_units")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

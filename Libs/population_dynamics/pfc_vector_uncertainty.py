"""Joint PFC vector inference from complete, bound calibration histories.

Finite-sample bands cover ideal independent source histories. They use the
source refractory count bound and retain every condition, including silence.
The development analysis grants no source, memory or scaling qualification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np

CIRCUITS = (23, 63, 52, 86)
BLOCKS = (3, 4, 5)
SIZES = (2560, 100000)
CONDITIONS = (
    "cue_none", "cue_aligned", "cue_off_target", "cue_antipodal", "cue_late_off_target",
    "omission_none", "omission_aligned", "omission_off_target", "omission_antipodal",
    "omission_late_off_target",
)
PHASE_EDGES = (0, 8000, 24080, 32000, 36000, 192000, 196000, 320000, 400000)
DT_SECONDS = 0.0000125
REFRACTORY_TICKS = 160  # Source E absolute refractory period: 2 ms.
SUMMARY_SHA256 = "438a5d34479ef4f51b3d91726f14c4df1fa83b26c46f08d4b481c0efe3980256"
SCHEMA = "pfc_joint_vector_uncertainty_v1"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def phase_rate_bound(first, stop):
    """At most ceil(phase_ticks / refractory_ticks) events per E unit."""
    require(type(first) is int and type(stop) is int and 0 <= first < stop,
            "PFC_VECTOR_PHASE_INVALID")
    return ((stop - first - 1) // REFRACTORY_TICKS + 1) / ((stop - first) * DT_SECONDS)


def mean_band(values, lower, upper, alpha, endpoint_count):
    """Union-bound two-sided Hoeffding intervals for independent histories.

    Endpoint dependence within a history is unrestricted. A constant observed
    endpoint does not become a structural zero or receive a zero-width band.
    """
    x = np.asarray(values, dtype=float)
    require(x.ndim == 1 and len(x) >= 2 and np.isfinite(x).all(),
            "PFC_VECTOR_INDEPENDENT_HISTORY_VALUES_REQUIRED")
    require(math.isfinite(lower) and math.isfinite(upper) and lower < upper
            and 0 < alpha < 1 and type(endpoint_count) is int and endpoint_count > 0,
            "PFC_VECTOR_BAND_DOMAIN_INVALID")
    require(np.all((x >= lower) & (x <= upper)), "PFC_VECTOR_PHYSICAL_RANGE_VIOLATION")
    radius = (upper - lower) * math.sqrt(math.log(2 * endpoint_count / alpha) / (2 * len(x)))
    point = float(x.mean())
    return point, max(lower, point - radius), min(upper, point + radius)


def rectangle_direction(target_lower, target_upper, orthogonal_lower, orthogonal_upper,
                        cue_halfwidth_degrees):
    """Assess the entire confidence rectangle, rather than its mean angle."""
    require(all(math.isfinite(x) for x in (target_lower, target_upper,
            orthogonal_lower, orthogonal_upper, cue_halfwidth_degrees))
            and target_lower <= target_upper and orthogonal_lower <= orthogonal_upper
            and 0 < cue_halfwidth_degrees < 90, "PFC_VECTOR_RECTANGLE_INVALID")
    excludes_zero = not (target_lower <= 0 <= target_upper
                         and orthogonal_lower <= 0 <= orthogonal_upper)
    cue_support = (target_lower > 0 and max(abs(orthogonal_lower), abs(orthogonal_upper))
                   <= target_lower * math.tan(math.radians(cue_halfwidth_degrees)))
    return dict(confidence_region_excludes_zero=excludes_zero,
                confidence_region_inside_cue_support=cue_support)


def episode_vector(row, n_e, cue_degrees):
    """Normalize additive spike moments before computing within-history effects."""
    first, stop = int(row["start_tick"]), int(row["stop_tick"])
    total = int(row["excitatory_spikes"])
    x, y = float(row["vector_x_spikes"]), float(row["vector_y_spikes"])
    require(type(n_e) is int and n_e > 0 and total >= 0 and math.isfinite(cue_degrees)
            and all(math.isfinite(v) for v in (x, y)), "PFC_VECTOR_MOMENTS_INVALID")
    duration = (stop - first) * DT_SECONDS
    bound = phase_rate_bound(first, stop)
    length = math.hypot(x, y)
    require(total <= n_e * ((stop - first - 1) // REFRACTORY_TICKS + 1)
            and length <= total + 64 * math.ulp(1.) * max(1, total),
            "PFC_VECTOR_EVENT_SUPPORT_INVALID")
    if total == 0:
        require(x == y == 0., "PFC_VECTOR_SILENT_NONZERO_MOMENT")
    angle = math.radians(cue_degrees)
    target = (x * math.cos(angle) + y * math.sin(angle)) / (n_e * duration)
    orthogonal = (-x * math.sin(angle) + y * math.cos(angle)) / (n_e * duration)
    # Floating-point rotations can exceed the analytical bound by roundoff.
    require(max(abs(target), abs(orthogonal)) <= bound * (1 + 64 * math.ulp(1.)),
            "PFC_VECTOR_PROJECTED_SUPPORT_INVALID")
    defined = total > 0 and length > 64 * math.ulp(1.) * total
    return dict(target_activity_Hz=target, orthogonal_activity_Hz=orthogonal,
        excitatory_rate_Hz=total / (n_e * duration), excitatory_spikes=total,
        numerical_direction_defined=defined, silent=total == 0,
        resultant_length=length / total if total else None,
        circular_cue_error_deg=((math.degrees(math.atan2(y, x)) - cue_degrees + 180) % 360 - 180)
            if defined else None,
        physical_component_bound_Hz=bound)


def load_calibration(inputs):
    """Accept only the complete existing train/calibration denominator."""
    manifest_path = inputs / "inputs.json"
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("schema") == "pfc_joint_vector_inputs_v1"
            and manifest.get("summary_sha256") == SUMMARY_SHA256
            and sha256(inputs / "size_summary.json") == SUMMARY_SHA256,
            "PFC_VECTOR_BOUND_SIZE_SUMMARY_REQUIRED")
    summary = json.loads((inputs / "size_summary.json").read_bytes())
    require(summary["status"] == "COMPLETE120_DESCRIPTIVE_SIZE_CALIBRATION"
            and summary["circuits"] == list(CIRCUITS) and summary["blocks"] == list(BLOCKS)
            and summary["conditions"] == list(CONDITIONS)
            and summary["original_size_episodes"] == summary["expanded_size_episodes_reused"] == 120
            and summary["cross_size_noise_pairing"] is False
            and len(summary["bindings"]) == 120, "PFC_VECTOR_CALIBRATION_SCOPE_CHANGED")
    task_path = inputs / "task_definition.json"
    require(sha256(task_path) == manifest["task_definition_sha256"], "PFC_VECTOR_TASK_CHANGED")
    task = json.loads(task_path.read_bytes())["task_protocol"]
    require(task["frozen"] is True and task["dt_ms"] == .0125
            and task["duration_ms"] == 5000. and task["cue_active_operation_indices"] == [100, 301]
            and task["stimulus_center_deg"] == 120. and task["stimulus_width_deg"] == 30.
            and [x["condition_id"] for x in task["conditions"]] == list(CONDITIONS),
            "PFC_VECTOR_FROZEN_TASK_REQUIRED")
    bindings = {b["episode_id"]: b for b in summary["bindings"]}
    expected = {(c, b, k, size) for c in CIRCUITS for b in BLOCKS for k in range(10) for size in SIZES}
    cells, data, source_bindings = set(), {}, []
    for row in manifest["episodes"]:
        key = tuple(row[n] for n in ("circuit_index", "block_index", "condition_index", "source_size"))
        require(key in expected and key not in cells, "PFC_VECTOR_DUPLICATE_OR_PROTECTED_CELL")
        c, b, k, size = key
        episode = f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_{k:02d}"
        role = "original" if size == 2560 else "expanded"
        stem = episode + "_" + role
        report_path, phase_path = inputs / "episodes" / (stem + "_report.json"), inputs / "episodes" / (stem + "_phases.csv")
        require(sha256(report_path) == bindings[episode][role + "_report_sha256"],
                "PFC_VECTOR_REPORT_BINDING_CHANGED")
        report = json.loads(report_path.read_bytes())
        require(report["status"] == "PASS_REFERENCE_SOURCE_QUALITY_READOUT"
                and report["identity"]["episode_id"] == episode
                and tuple(report["identity"][n] for n in ("circuit_index", "block_index", "condition_index")) == (c, b, k)
                and report["identity"]["split"] == ("calibration" if c == 86 else "train")
                and report["condition"] == task["conditions"][k]
                and report["core_units"] == size and report["dt_ms"] == .0125
                and report["duration_ticks"] == 400000
                and report["source_qualification_granted"] is False
                and report["scientific_claim_qualified"] is False and report["production_episodes"] == 0,
                "PFC_VECTOR_SOURCE_IDENTITY_CHANGED")
        require(report["outputs"]["phase_responses.csv"]["sha256"] == sha256(phase_path),
                "PFC_VECTOR_PHASE_HASH_CHANGED")
        with phase_path.open(newline="", encoding="utf-8") as stream:
            phases = list(csv.DictReader(stream))
        require(len(phases) == 8, "PFC_VECTOR_INCOMPLETE_PHASES")
        vectors = []
        for phase, record in enumerate(phases):
            require(int(record["phase_index"]) == phase
                    and (int(record["start_tick"]), int(record["stop_tick"])) == PHASE_EDGES[phase:phase + 2]
                    and (int(record["circuit_index"]), int(record["block_index"])) == (c, b)
                    and record["episode_id"] == episode and record["condition_id"] == CONDITIONS[k],
                    "PFC_VECTOR_PHASE_IDENTITY_CHANGED")
            vectors.append(episode_vector(record, size * 4 // 5, task["stimulus_center_deg"]))
        data[key] = vectors
        source_bindings.append(dict(episode_id=episode, source_size=size,
            initial_state_stream=report["identity"]["initial_state_seed"],
            external_noise_stream=report["identity"]["background_seed"],
            initial_voltage_sha256=report["initial_voltage"]["file_sha256"],
            external_final_rng_sha256=report["external_input"]["final_rng_sha256"],
            report_sha256=sha256(report_path), phase_sha256=sha256(phase_path)))
        cells.add(key)
    require(cells == expected and len(source_bindings) == 240, "PFC_VECTOR_COMPLETE240_REQUIRED")
    # Verify conditions are paired, but sizes and histories do not share streams.
    source_by_id = {(r["episode_id"], r["source_size"]): r for r in source_bindings}
    for c in CIRCUITS:
        for size in SIZES:
            histories = []
            for b in BLOCKS:
                items = [source_by_id[(f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_{k:02d}", size)] for k in range(10)]
                fields = ("initial_state_stream", "external_noise_stream", "initial_voltage_sha256", "external_final_rng_sha256")
                require(all(tuple(r[f] for f in fields) == tuple(items[0][f] for f in fields) for r in items),
                        "PFC_VECTOR_CONDITIONS_NOT_PAIRED")
                histories.append(items[0])
            require(all(len({r[field] for r in histories}) == 3 for field in fields),
                    "PFC_VECTOR_DUPLICATED_HISTORY")
        for b in BLOCKS:
            small = source_by_id[(f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_00", 2560)]
            large = source_by_id[(f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_00", 100000)]
            require(all(small[f] != large[f] for f in ("initial_state_stream", "external_noise_stream")),
                    "PFC_VECTOR_CROSS_SIZE_STREAMS_NOT_DISTINCT")
    return manifest, data, source_bindings, task


def write_rows(path, rows):
    require(bool(rows), "PFC_VECTOR_EMPTY_OUTPUT")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(inputs, output, alpha):
    started = time.perf_counter()
    manifest, data, bindings, task = load_calibration(inputs)
    require(not output.exists(), "PFC_VECTOR_OUTPUT_EXISTS")
    endpoint_count = len(CIRCUITS) * len(SIZES) * len(CONDITIONS) * 8 * 2
    bands, episodes, contrasts, size_effects, lookup = [], [], [], [], {}
    halfwidth = task["stimulus_width_deg"] / 2
    for c in CIRCUITS:
        for size in SIZES:
            for k, condition in enumerate(CONDITIONS):
                for phase in range(8):
                    items = [data[c, b, k, size][phase] for b in BLOCKS]
                    bound = items[0]["physical_component_bound_Hz"]
                    base = dict(circuit_index=c, split="calibration" if c == 86 else "train",
                                source_size=size, condition_id=condition, phase_index=phase)
                    components = {}
                    for metric in ("target_activity_Hz", "orthogonal_activity_Hz"):
                        values = [r[metric] for r in items]
                        components[metric] = mean_band(values, -bound, bound, alpha, endpoint_count)
                    a, lo, hi = components["target_activity_Hz"]
                    o, olo, ohi = components["orthogonal_activity_Hz"]
                    region = rectangle_direction(lo, hi, olo, ohi, halfwidth)
                    entry = dict(base, independent_histories=3, mean_target_activity_Hz=a,
                        mean_orthogonal_activity_Hz=o, target_lower_Hz=lo, target_upper_Hz=hi,
                        orthogonal_lower_Hz=olo, orthogonal_upper_Hz=ohi,
                        between_history_vector_covariance_Hz2=float(np.cov(
                            [[r["target_activity_Hz"] for r in items],
                             [r["orthogonal_activity_Hz"] for r in items]], ddof=1)[0, 1]),
                        silent_histories=sum(r["silent"] for r in items),
                        numerical_direction_undefined_histories=sum(not r["numerical_direction_defined"] for r in items),
                        cue_halfwidth_degrees=halfwidth, **region,
                        scientific_direction_qualified=False, memory_success_qualified=False)
                    bands.append(entry)
                    lookup[c, size, k, phase] = components
                    for b, item in zip(BLOCKS, items):
                        episodes.append(dict(base, block_index=b, **item,
                            numerical_angle_inside_cue_support=(abs(item["circular_cue_error_deg"]) <= halfwidth)
                                if item["numerical_direction_defined"] else None,
                            scientific_direction_qualified=False))
                # Pair all ten conditions within their original histories.
            for phase in range(8):
                for k in range(5):
                    for metric in ("target_activity_Hz", "orthogonal_activity_Hz"):
                        left, right = lookup[c, size, k, phase][metric], lookup[c, size, k + 5, phase][metric]
                        values = [data[c, b, k, size][phase][metric] - data[c, b, k + 5, size][phase][metric] for b in BLOCKS]
                        contrasts.append(dict(circuit_index=c, source_size=size, phase_index=phase,
                            condition_id=CONDITIONS[k], contrast="cue_minus_omission", metric=metric,
                            mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)),
                            lower=left[1] - right[2], upper=left[2] - right[1], independent_histories=3,
                            interval_method="propagated_complete_primitive_simultaneous_bands"))
                        if k:
                            base_left, base_right = lookup[c, size, 0, phase][metric], lookup[c, size, 5, phase][metric]
                            interaction = [v - data[c, b, 0, size][phase][metric] + data[c, b, 5, size][phase][metric] for b, v in zip(BLOCKS, values)]
                            contrasts.append(dict(circuit_index=c, source_size=size, phase_index=phase,
                                condition_id=CONDITIONS[k], contrast="distractor_difference_of_differences", metric=metric,
                                mean=float(np.mean(interaction)), sample_sd=float(np.std(interaction, ddof=1)),
                                lower=left[1] - right[2] - base_left[2] + base_right[1],
                                upper=left[2] - right[1] - base_left[1] + base_right[2], independent_histories=3,
                                interval_method="propagated_complete_primitive_simultaneous_bands"))
        for k, condition in enumerate(CONDITIONS):
            for phase in range(8):
                for metric in ("target_activity_Hz", "orthogonal_activity_Hz"):
                    small, large = lookup[c, 2560, k, phase][metric], lookup[c, 100000, k, phase][metric]
                    size_effects.append(dict(circuit_index=c, condition_id=condition, phase_index=phase,
                        metric=metric, expanded_minus_original=large[0] - small[0],
                        lower=large[1] - small[2], upper=large[2] - small[1],
                        cross_size_noise_pairing=False, scaling_qualified=False))
    output.mkdir(parents=True)
    names = ("episode_vector_moments.csv", "joint_vector_mean_bands.csv", "paired_vector_effects.csv", "vector_size_effects.csv")
    for name, rows in zip(names, (episodes, bands, contrasts, size_effects)):
        write_rows(output / name, rows)
    report = dict(schema=SCHEMA, status="COMPLETE_CALIBRATION_VECTOR_UNCERTAINTY",
        recorded_utc=datetime.now(timezone.utc).isoformat(), source_bindings=bindings,
        input_manifest_sha256=sha256(inputs / "inputs.json"), summary_sha256=SUMMARY_SHA256,
        task_definition_sha256=manifest["task_definition_sha256"],
        execution=dict(backend="merlin", run_id=os.environ.get("RH2_JOB_RUN_ID"),
            trial_id=os.environ.get("ARNOLD_TRIAL_ID"), hostname=socket.gethostname()),
        circuits=list(CIRCUITS), blocks=list(BLOCKS), source_sizes=list(SIZES), conditions=list(CONDITIONS),
        complete_source_episodes=240, independent_histories_per_circuit_and_size=3,
        family_alpha=alpha, primitive_component_endpoints=endpoint_count,
        uncertainty_method="two_sided_Hoeffding_with_union_bound_over_all1280_primitive_component_means",
        coverage_basis="ideal independent joint source histories; bounded refractory spike moments; arbitrary endpoint dependence within history; no Gaussian/bootstrap or neuron independence assumption",
        uncertainty_scope="development calibration family only; no allocation of the formal source/coarse-engine error budget",
        derived_contrast_intervals="signed linear propagation of the same simultaneous primitive bands; no additional error allocation",
        direction_mean_scope="joint mean-vector confidence region excludes zero and lies within source cue support; not episode memory reliability",
        cue_support_role="source stimulus halfwidth15degrees is a descriptive geometric target; no automatic adoption as a formal memory criterion",
        refractory_count_bound="ceil(phase_ticks/160) events per E unit in each half-open phase; E refractory2ms",
        mean_regions_excluding_zero=sum(r["confidence_region_excludes_zero"] for r in bands),
        mean_regions_inside_cue_support=sum(r["confidence_region_inside_cue_support"] for r in bands),
        all_conditions_retained=True, protected_circuits_used=False,
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scientific_parameter_domain_qualified=False, scaling_qualified=False,
        scientific_direction_qualified=False, memory_success_qualified=False,
        formal_confirmation_released=False,
        elapsed_seconds=time.perf_counter() - started, implementation_sha256=sha256(Path(__file__)),
        outputs={name: dict(sha256=sha256(output / name), rows=len(rows))
                 for name, rows in zip(names, (episodes, bands, contrasts, size_effects))})
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get("RH2_JOB_RUN_ID")) and bool(os.environ.get("ARNOLD_TRIAL_ID"))
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0", "PFC_VECTOR_MERLIN_CPU_REQUIRED")
    result = analyze(args.inputs, args.output, args.alpha)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_bindings", "outputs")}))


if __name__ == "__main__":
    main()

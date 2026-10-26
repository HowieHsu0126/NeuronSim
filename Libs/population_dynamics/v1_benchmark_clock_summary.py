"""Summarize complete V1 benchmark clock pairs without a fidelity verdict."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_benchmark_readout import selectivity


def summarize_block(reference: np.ndarray, fine: np.ndarray, low_direction: int, duration_s: float = 2.5):
    """Preserve undefined selectivity and all-unit contrast denominators."""
    if (not np.isfinite(duration_s) or duration_s <= 0
            or type(low_direction) is not int or not 0 <= low_direction < 8
            or reference.ndim != 2 or reference.shape[0] != 10
            or reference.shape != fine.shape or not reference.shape[1]
            or any(x.dtype.kind not in "iu" or np.any(x < 0) for x in (reference, fine))):
        raise ValueError("CLOCK_COMPLETE_TEN_CONDITION_COUNTS_REQUIRED")
    units = reference.shape[1]
    a, b = selectivity(reference[:8]), selectivity(fine[:8])
    arrays, rows = {}, []
    for metric in ("direction_selectivity", "orientation_selectivity"):
        left, right = a[metric], b[metric]
        left_defined, right_defined = np.isfinite(left), np.isfinite(right)
        paired = left_defined & right_defined
        delta = right - left
        arrays.update({"reference_" + metric: left, "fine_" + metric: right, metric + "_difference": delta})
        rows.append(dict(metric=metric, units=units,
            reference_defined_units=int(left_defined.sum()), fine_defined_units=int(right_defined.sum()),
            paired_defined_units=int(paired.sum()),
            reference_mean=float(left[left_defined].mean()) if left_defined.any() else None,
            fine_mean=float(right[right_defined].mean()) if right_defined.any() else None,
            paired_mean_difference=float(delta[paired].mean()) if paired.any() else None,
            paired_mean_absolute_difference=float(np.abs(delta[paired]).mean()) if paired.any() else None))
    for metric, treated, control in (("low_minus_high_rate_hz", 9, low_direction),
                                      ("low_minus_gray_rate_hz", 9, 8),
                                      ("high_minus_gray_rate_hz", low_direction, 8)):
        left = (reference[treated].astype(np.float64) - reference[control].astype(np.float64)) / duration_s
        right = (fine[treated].astype(np.float64) - fine[control].astype(np.float64)) / duration_s
        arrays.update({"reference_" + metric: left, "fine_" + metric: right, metric + "_difference": right-left})
        rows.append(dict(metric=metric, units=units, reference_defined_units=units,
            fine_defined_units=units, paired_defined_units=units, reference_mean=float(left.mean()),
            fine_mean=float(right.mean()), paired_mean_difference=float((right-left).mean()),
            paired_mean_absolute_difference=float(np.abs(right-left).mean())))
    return arrays, rows


def summarize(cases_path: Path, submissions_path: Path, output: Path) -> dict:
    plan, submitted = json.loads(cases_path.read_text()), json.loads(submissions_path.read_text())
    cases = {row["identity"]["episode_id"]: row for row in plan["cases"]}
    jobs = {row["identity"]["episode_id"]: row for row in submitted["cases"]}
    if (len(cases) != len(plan["cases"]) or len(jobs) != len(submitted["cases"])
            or cases.keys() != jobs.keys() or not cases or plan["full_units"] != 230924):
        raise ValueError("CLOCK_PLANNED_CASE_IDENTITIES_DIFFER")
    groups = {}
    for key, case in cases.items():
        identity = case["identity"]
        if identity != jobs[key]["identity"]:
            raise ValueError("CLOCK_SUBMITTED_CASE_IDENTITY_DIFFERS")
        groups.setdefault(identity["block_id"], []).append(key)
    for keys in groups.values():
        if sorted(cases[key]["identity"]["condition_index"] for key in keys) != list(range(10)):
            raise ValueError("CLOCK_COMPLETE_TEN_CONDITION_BLOCK_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="FAILED_V1_BENCHMARK_CLOCK_SUMMARY", exit_code=1,
        scientific_qualification=False, production_episodes=0,
        cases_sha256=sha256_file(cases_path), submissions_sha256=sha256_file(submissions_path),
        interpretation="Descriptive whole-model clock effects. Neurons and blocks are not independent circuit replicates. No CI, significance, equivalence or convergence verdict.")
    case_rows, block_rows, evidence, artifacts = [], [], [], {}
    try:
        for block_id, keys in sorted(groups.items()):
            keys.sort(key=lambda key: cases[key]["identity"]["condition_index"])
            matrices = {window: {side: [] for side in ("reference", "fine")}
                        for window in ("source_response", "command_response")}
            source_ids = None
            for key in keys:
                case, job = cases[key], jobs[key]
                directory = Path(job["comparison_output"])
                delivery = json.loads((directory/"delivery.json").read_text())
                report_path = directory/"analysis/report.json"
                if (delivery.get("exit_code") != 0 or str(delivery.get("job_id")) != str(job["comparison_job"])
                        or delivery.get("reference_archive") != case["reference"]
                        or delivery.get("fine_archive") != job["native_archive"]
                        or delivery.get("report_sha256") != sha256_file(report_path)):
                    raise ValueError("CLOCK_COMPLETE_BOUND_CASE_COMPARISON_REQUIRED:" + key)
                report = json.loads(report_path.read_text())
                if (report.get("status") != "V1_CLOCK_COMPARISON_COMPLETE_DEVELOPMENT"
                        or report.get("exit_code") != 0 or report.get("mode") != "full"
                        or report.get("source_bindings_and_requested_inputs_exact") is not True
                        or report.get("physical_parameters_and_initial_states_exact") is not True
                        or report["inputs"]["reference"]["consumed"]["delivery.json"]["sha256"] != case["delivery_sha256"]
                        or str(report["inputs"]["fine"]["job_id"]) != str(job["native_job"])
                        or report.get("reference_resolution_ms") != .25 or report.get("fine_resolution_ms") != .125
                        or report["all_units"]["units"] != 230924):
                    raise ValueError("CLOCK_VALIDATED_FULL_CASE_REQUIRED:" + key)
                arrays_path = directory/"analysis/per_unit.npz"
                record = report["artifacts"]["per_unit.npz"]
                if arrays_path.stat().st_size != record["bytes"] or sha256_file(arrays_path) != record["sha256"]:
                    raise ValueError("CLOCK_CASE_ARRAY_CHANGED:" + key)
                with np.load(arrays_path, allow_pickle=False) as data:
                    ids = data["source_id"]
                    if len(ids) != 230924 or (source_ids is not None and not np.array_equal(ids, source_ids)):
                        raise ValueError("CLOCK_BLOCK_UNIT_IDENTITIES_DIFFER:" + key)
                    source_ids = ids.copy()
                    for window in matrices:
                        for side in matrices[window]:
                            matrices[window][side].append(data[side + "_" + window + "_counts"])
                value = report["all_units"]
                case_rows.append(dict(episode_id=key, circuit_id=case["identity"]["circuit_id"], block_id=block_id,
                    condition_index=case["identity"]["condition_index"], units=value["units"],
                    reference_spikes=value["reference_spikes"], fine_spikes=value["fine_spikes"],
                    spike_count_difference=value["spike_count_difference"],
                    changed_spike_count_units=value["changed_spike_count_units"],
                    reference_silent_units=value["reference_silent_units"], fine_silent_units=value["fine_silent_units"]))
                evidence.append(dict(episode_id=key, comparison_report=str(report_path), report_sha256=delivery["report_sha256"]))
            low_direction = cases[keys[0]]["identity"]["block_index"] % 8
            for window, matrix in matrices.items():
                duration_s = 2.5 if window == "source_response" else 2.499
                arrays, rows = summarize_block(np.stack(matrix["reference"]), np.stack(matrix["fine"]), low_direction, duration_s)
                name = block_id + "_" + window + ".npz"
                np.savez_compressed(output/name, source_id=source_ids, **arrays)
                artifacts[name] = dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name))
                block_rows.extend(dict(block_id=block_id, circuit_id=cases[keys[0]]["identity"]["circuit_id"],
                    window=window, **row) for row in rows)
        for name, rows in (("case_summary.csv", case_rows), ("task_summary.csv", block_rows)):
            with (output/name).open("x", newline="") as stream:
                writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            artifacts[name]=dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name))
        result.update(status="COMPLETE_V1_BENCHMARK_CLOCK_SUMMARY", exit_code=0,
            complete_pairs=len(case_rows), complete_blocks=len(groups),
            circuit_identities=len({row["circuit_id"] for row in case_rows}), evidence=evidence, artifacts=artifacts,
            windows=dict(source_response="(500,3000] ms", command_response="[501,3000) ms"))
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output/"report.json", result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", dest="cases_path", type=Path, required=True)
    parser.add_argument("--submissions", dest="submissions_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result=summarize(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("status", "complete_pairs", "complete_blocks", "circuit_identities")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

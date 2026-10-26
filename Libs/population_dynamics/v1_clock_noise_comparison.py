"""Compare accepted clock effects with one matched V1 noise-block contrast."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from .provenance import freeze_json, sha256_file

WINDOWS = ("full", "pre_command", "command_response")
LAGS = (0, 1, 10, 100)
TASK_METRICS = (
    "direction_selectivity", "orientation_selectivity",
    "low_minus_high_rate_hz", "low_minus_gray_rate_hz", "high_minus_gray_rate_hz",
)


def change_summary(reference_a, fine_a, reference_b, fine_b, *, allow_undefined=False):
    """Use the same defined elements for all four arms; retain all denominators."""
    raw = [np.asarray(x) for x in (reference_a, fine_a, reference_b, fine_b)]
    if (not raw[0].size or any(x.shape != raw[0].shape or x.dtype.kind not in "iuf" for x in raw)
            or any(np.isinf(x).any() for x in raw)):
        raise ValueError("CLOCK_NOISE_FINITE_MATCHED_ARRAYS_REQUIRED")
    finite = [np.isfinite(x) for x in raw]
    if not allow_undefined and not all(x.all() for x in finite):
        raise ValueError("CLOCK_NOISE_UNEXPECTED_UNDEFINED_VALUE")
    common = np.logical_and.reduce(finite)
    result = dict(elements=raw[0].size, common_defined_elements=int(common.sum()),
                  excluded_undefined_elements=int((~common).sum()))
    for name, mask in zip(("reference_a", "fine_a", "reference_b", "fine_b"), finite):
        result[name + "_defined_elements"] = int(mask.sum())
    for name in ("clock_a", "clock_b", "reference_noise", "fine_noise"):
        for moment in ("mean", "mean_absolute", "rms"):
            result[name + "_" + moment] = None
    result.update(mean_clock_to_reference_noise_l1_ratio=None,
                  mean_clock_to_reference_noise_rms_ratio=None,
                  reference_noise_l1_zero=None)
    if not common.any():
        return result
    a, fa, b, fb = [x[common].astype(np.longdouble) for x in raw]
    for name, delta in (("clock_a", fa-a), ("clock_b", fb-b),
                        ("reference_noise", b-a), ("fine_noise", fb-fa)):
        result[name + "_mean"] = float(delta.mean())
        result[name + "_mean_absolute"] = float(np.abs(delta).mean())
        result[name + "_rms"] = float(np.sqrt(np.mean(delta*delta)))
    for moment, label in (("mean_absolute", "l1"), ("rms", "rms")):
        denominator = result["reference_noise_" + moment]
        if denominator != 0:
            result["mean_clock_to_reference_noise_" + label + "_ratio"] = (
                (result["clock_a_" + moment] + result["clock_b_" + moment]) / (2*denominator))
    result["reference_noise_l1_zero"] = result["reference_noise_mean_absolute"] == 0
    return result


def select_blocks(cases, circuit_id: str, first_block: int, second_block: int):
    """Compare identical ten-condition controls, including rotated low contrast."""
    if (first_block == second_block or min(first_block, second_block) < 0
            or first_block % 8 != second_block % 8):
        raise ValueError("CLOCK_NOISE_DISTINCT_BLOCKS_WITH_MATCHED_LOW_CONTRAST_REQUIRED")
    selected = []
    for block in (first_block, second_block):
        rows = [r for r in cases if r["identity"]["circuit_id"] == circuit_id
                and r["identity"]["block_index"] == block]
        if (len(rows) != 10 or sorted(r["identity"]["condition_index"] for r in rows) != list(range(10))
                or any(r["identity"]["scenario_id"] != "v1_visual"
                       or r["identity"]["split"] != "train" for r in rows)):
            raise ValueError("CLOCK_NOISE_COMPLETE_TRAIN_DEVELOPMENT_BLOCK_REQUIRED")
        selected.append(sorted(rows, key=lambda r: r["identity"]["condition_index"]))
    for a, b in zip(*selected):
        x, y = a["identity"], b["identity"]
        if (x["connectivity_seed"] != y["connectivity_seed"]
                or x["background_seed"] == y["background_seed"]
                or x["stimulus_seed"] == y["stimulus_seed"]):
            raise ValueError("CLOCK_NOISE_FIXED_CIRCUIT_DISTINCT_DRIVER_BLOCKS_REQUIRED")
    return selected


def verified_arrays(path: Path, record: dict):
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("CLOCK_NOISE_ACCEPTED_ARRAY_CHANGED:" + str(path))
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def load_accepted_panel(panel: Path):
    """Require every declared case in both authenticated panel summaries."""
    acceptance = json.loads((panel / "panel_acceptance.json").read_text())
    plan_path, jobs_path = panel / "cases.json", panel / "envelope_submission.json"
    plan = json.loads(plan_path.read_text())
    cases = plan["cases"]
    ids = [case["identity"]["episode_id"] for case in cases]
    count = len(cases)
    if (acceptance.get("status") != "PASS_COMPLETE_CLOCK_PANEL_DELIVERY"
            or count == 0 or len(set(ids)) != count or plan.get("full_units") != 230924
            or acceptance.get("verified_pairs") != count):
        raise ValueError("CLOCK_NOISE_COMPLETE_ACCEPTED_PANEL_REQUIRED")
    reports = []
    for relative, key in (("task_summary/analysis/report.json", "task_summary_report_sha256"),
                          ("joint_activity/report.json", "joint_activity_report_sha256")):
        path = panel / relative
        if sha256_file(path) != acceptance[key]:
            raise ValueError("CLOCK_NOISE_ACCEPTED_PANEL_REPORT_CHANGED")
        report = json.loads(path.read_text())
        if (report["cases_sha256"] != sha256_file(plan_path)
                or report["submissions_sha256"] != sha256_file(jobs_path)):
            raise ValueError("CLOCK_NOISE_PANEL_IDENTITIES_CHANGED")
        reports.append(report)
    task, joint = reports
    evidence = task.get("evidence", [])
    joint_cases = joint.get("reports", [])
    if (task.get("status") != "COMPLETE_V1_BENCHMARK_CLOCK_SUMMARY" or task.get("exit_code") != 0
            or joint.get("status") != "COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PANEL"
            or task.get("complete_pairs") != count
            or joint.get("pairs") != count or joint.get("planned_pairs") != count
            or len(evidence) != count or len(joint_cases) != count
            or {row["episode_id"] for row in evidence} != set(ids)
            or {row["episode_id"] for row in joint_cases} != set(ids)):
        raise ValueError("CLOCK_NOISE_DECLARED_PANEL_COVERAGE_DIFFERS")
    return acceptance, plan, task, joint


def compare(panel: Path, circuit_id: str, first_block: int, second_block: int, output: Path):
    acceptance_path = panel / "panel_acceptance.json"
    acceptance, plan, task, joint = load_accepted_panel(panel)
    blocks = select_blocks(plan["cases"], circuit_id, first_block, second_block)
    task_dir = panel / "task_summary/analysis"
    joint_dir = panel / "joint_activity"
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="FAILED_V1_CLOCK_NOISE_COMPARISON", exit_code=1,
                  acceptance=str(acceptance_path.resolve()), acceptance_sha256=sha256_file(acceptance_path),
                  implementation_sha256=sha256_file(Path(__file__)), command=sys.argv,
                  circuit_id=circuit_id, block_indices=[first_block, second_block],
                  conditions=10, units=230924, populations=21, diagnostic_bin_ms=1, lag_ms=list(LAGS),
                  reference_dt_ms=0.25, fine_dt_ms=0.125, noise_block_pairs=1,
                  scientific_qualification=False, new_source_simulations=0, new_unique_episodes=0,
                  production_episodes=0, tolerances_frozen=False,
                  interpretation="One matched noise-block contrast within one circuit. Descriptive scale comparison only; no sampling uncertainty, equivalence, significance, independence or convergence verdict.")
    rows = []
    try:
        for a_case, b_case in zip(*blocks):
            cases = (a_case, b_case)
            reports = []
            for case in cases:
                key = case["identity"]["episode_id"]
                path = joint_dir / "cases" / key / "report.json"
                if sha256_file(path) != acceptance["joint_report_sha256_by_episode"][key]:
                    raise ValueError("CLOCK_NOISE_ACCEPTED_CASE_REPORT_CHANGED:" + key)
                report = json.loads(path.read_text())
                if report["identity"] != case["identity"] or report["units"] != 230924:
                    raise ValueError("CLOCK_NOISE_CASE_IDENTITY_DIFFERS")
                reports.append(report)
            for window in WINDOWS:
                data = [verified_arrays(joint_dir / "cases" / case["identity"]["episode_id"] / (window + ".npz"),
                                        report["artifacts"][window + ".npz"])
                        for case, report in zip(cases, reports)]
                for key in ("source_id", "population", "population_names", "population_index",
                            "population_unit_counts", "lag_ms"):
                    if not np.array_equal(data[0][key], data[1][key]):
                        raise ValueError("CLOCK_NOISE_STATIC_UNIT_MAP_DIFFERS:" + key)
                if data[0]["source_id"].shape != (230924,):
                    raise ValueError("CLOCK_NOISE_COMPLETE_UNIT_MAP_REQUIRED")
                metrics = [("unit_totals", None, "spikes_in_window"),
                           ("population_counts", None, "spikes_per_1ms_bin")]
                metrics.extend((name, i, "spike_count_squared")
                               for name in ("unit_count_autocovariance", "population_count_covariance")
                               for i in range(len(LAGS)))
                for metric, index, unit in metrics:
                    values = [d[side + "_" + metric] for d in data for side in ("reference", "fine")]
                    if index is not None:
                        values = [value[index] for value in values]
                    rows.append(dict(condition_index=cases[0]["identity"]["condition_index"],
                        window=window, metric=metric, lag_ms=None if index is None else LAGS[index],
                        value_unit=unit, **change_summary(*values)))
        for window in ("source_response", "command_response"):
            names = [block[0]["identity"]["block_id"] + "_" + window + ".npz" for block in blocks]
            data = [verified_arrays(task_dir / name, task["artifacts"][name]) for name in names]
            if not np.array_equal(data[0]["source_id"], data[1]["source_id"]):
                raise ValueError("CLOCK_NOISE_TASK_UNIT_MAP_DIFFERS")
            for metric in TASK_METRICS:
                values = [d[side + "_" + metric] for d in data for side in ("reference", "fine")]
                rows.append(dict(condition_index=None, window=window, metric=metric, lag_ms=None,
                    value_unit="Hz" if metric.endswith("_rate_hz") else "dimensionless",
                    **change_summary(*values, allow_undefined=metric.endswith("_selectivity"))))
        path = output / "comparison.csv"
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.update(status="COMPLETE_V1_CLOCK_NOISE_COMPARISON", exit_code=0,
                      comparison_rows=len(rows),
                      artifacts={path.name: dict(bytes=path.stat().st_size, sha256=sha256_file(path))},
                      unit_selection="All elements for counts/covariance. Selectivity uses the common mathematically defined set; each arm and excluded denominator are reported.")
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--first-block", type=int, required=True)
    parser.add_argument("--second-block", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = compare(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("status", "comparison_rows", "noise_block_pairs", "scientific_qualification")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

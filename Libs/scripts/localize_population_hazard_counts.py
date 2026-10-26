"""Localize retained hazard/count differences without fitting or sampling a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform


KEYS = ("model", "seed", "case_id", "origin_ms", "horizon_ms")
MEASURES = ("integrated_draw_hazard_per_unit", "integrated_mean_path_hazard_per_unit",
            "integrated_posterior_mean_hazard_per_unit", "native_sampled_expected_spikes_per_unit",
            "native_observed_spikes_per_unit")
READOUT_MEASURES = ("native_energy_score", "native_unit_bin_mean_mse_corrected")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(row):
    return (row["model"], int(row["seed"]), row["case_id"],
            float(row["origin_ms"]), float(row["horizon_ms"]))


def load_table(analysis_path, report, name):
    path = analysis_path.parent / name
    if sha256(path) != report["artifacts"][name]["sha256"]:
        raise ValueError("Source table hash differs: " + name)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != report["artifacts"][name]["rows"] or len(rows) != 1296:
        raise ValueError("The complete 1296-row source table is required")
    indexed = {identity(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate source origin/horizon")
    return indexed


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(prefix_path, readout_path, output):
    environment = {key: os.environ.get(key, "") for key in
                   ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")}
    if not all(environment.values()) or platform.node().split(".")[0] == "mgmt2":
        raise RuntimeError("This analysis must execute on Merlin")
    prefix = json.loads(prefix_path.read_text())
    readout = json.loads(readout_path.read_text())
    if (prefix["status"] != "COMPLETE_RETAINED_HAZARD_COUNT_COMPARISON"
            or readout["status"] != "COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC"
            or prefix["original_analysis_sha256"] != sha256(readout_path)
            or prefix["training_seeds"] != [41, 42, 43]
            or prefix["source_units"] != 100000 or prefix["origins_per_model_seed"] != 48):
        raise ValueError("Bound complete three-seed diagnostic inputs required")
    rates = load_table(prefix_path, prefix, "origin_horizon_hazard_counts.csv")
    native = load_table(readout_path, readout, "origin_horizon_moments.csv")
    if rates.keys() != native.keys():
        raise ValueError("The two complete origin sets differ")
    rows, cells = [], defaultdict(list)
    for key, rate in sorted(rates.items()):
        original = native[key]
        for field in ("block", "template_id", "prefix_bins"):
            if rate[field] != original[field]:
                raise ValueError("Source identity differs: " + field)
        row = dict(zip(KEYS, key), block=int(rate["block"]), template_id=rate["template_id"])
        row.update({name: float(rate[name]) for name in MEASURES})
        row.update({name: float(original[name]) for name in READOUT_MEASURES})
        for name in MEASURES[-2:]:
            if row[name] != float(original[name]):
                raise ValueError("Recorded native count changed")
        if not all(math.isfinite(row[name]) for name in MEASURES + READOUT_MEASURES):
            raise ValueError("Finite paired metrics required")
        parent_key = ("original",) + key[1:]
        parent = native[parent_key]
        row["native_count_error"] = row[MEASURES[3]] - row[MEASURES[4]]
        parent_error = float(parent[MEASURES[3]]) - float(parent[MEASURES[4]])
        row["absolute_count_error_change_from_original"] = abs(row["native_count_error"]) - abs(parent_error)
        for name in READOUT_MEASURES:
            row[name + "_change_from_original"] = row[name] - float(parent[name])
        rows.append(row)
        cells[key[0], key[1], key[-1]].append(row)
    expected = {(model, seed, horizon) for model in ("original", "current", "history")
                for seed in (41, 42, 43) for horizon in (20., 80., 500.)}
    if set(cells) != expected or any(len(value) != 48 for value in cells.values()):
        raise ValueError("All 27 cells and all 48 origins per cell are required")
    prior = {(row["model"], row["seed"], row["horizon_ms"]): row
             for row in prefix["seed_horizon_values"]}
    maximum_mean_difference = 0.
    summaries, focal = [], []
    for key, selected in sorted(cells.items()):
        summary = dict(model=key[0], seed=key[1], horizon_ms=key[2], total_origins=len(selected))
        for name in MEASURES:
            summary[name] = math.fsum(row[name] for row in selected) / len(selected)
            difference = abs(summary[name] - prior[key][name])
            maximum_mean_difference = max(maximum_mean_difference, difference)
            if not math.isclose(summary[name], prior[key][name], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Independent source-summary reproduction differs")
        for name in READOUT_MEASURES + ("absolute_count_error_change_from_original",):
            summary[name] = math.fsum(row[name] for row in selected) / len(selected)
        ordered = sorted(selected, key=lambda row: row[MEASURES[0]], reverse=True)
        exposure = math.fsum(row[MEASURES[0]] for row in selected)
        summary["top_one_exposure_fraction"] = ordered[0][MEASURES[0]] / exposure
        summary["top_three_exposure_fraction"] = math.fsum(row[MEASURES[0]] for row in ordered[:3]) / exposure
        summary["top_five_exposure_fraction"] = math.fsum(row[MEASURES[0]] for row in ordered[:5]) / exposure
        summaries.append(summary)
        if key[-1] == 500.:
            focal.append(dict(summary, top_three_origins=ordered[:3]))
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, values in (("paired_origin_metrics.csv", rows), ("exposure_concentration.csv", summaries)):
        write_csv(output / name, values)
        artifacts[name] = dict(rows=len(values), sha256=sha256(output / name))
    result = dict(status="COMPLETE_NATIVE_EXPOSURE_LOCALIZATION", runtime=dict(execution_backend="merlin",
        environment=environment, hostname=platform.node()), prefix_analysis_sha256=sha256(prefix_path),
        readout_analysis_sha256=sha256(readout_path), source_units=100000, training_seeds=[41, 42, 43],
        complete_cells=27, origins_per_cell=48, complete_paired_rows=1296,
        source_summary_max_absolute_difference=maximum_mean_difference,
        focal_horizon_ms=500., focal_cells=focal, artifacts=artifacts,
        interpretation="All source rows and all horizons remain in the paired table. The three highest-exposure origins are displayed only to localize contributions, not to qualify a selected subset. No causal or inferential statistical conclusion is computed.",
        model_fitting_performed=False, trajectory_sampling_performed=False, source_simulation_performed=False,
        scientific_claim_qualified=False, software_sha256=sha256(Path(__file__)))
    (output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-analysis", type=Path, required=True)
    parser.add_argument("--readout-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.prefix_analysis, args.readout_analysis, args.output)
    print(json.dumps({key: result[key] for key in ("status", "complete_cells", "complete_paired_rows", "artifacts")}))


if __name__ == "__main__":
    main()

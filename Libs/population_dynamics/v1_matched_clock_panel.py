"""Join completed V1 clock blocks and retain the accepted source panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_benchmark_clock_summary import summarize
from .v1_clock_joint_activity import accepted_comparison, main as joint_main
from .v1_clock_noise_comparison import load_accepted_panel


def indexed(rows):
    result = {row["identity"]["episode_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("MATCHED_PANEL_DUPLICATE_EPISODE")
    return result


def combine_tables(base_cases: list, base_jobs: list, extension: dict, reference_digests: dict):
    """Keep the coarse execution as reference and the fine execution as native_job."""
    cases, jobs = indexed(base_cases), indexed(base_jobs)
    additions = indexed(extension["cases"])
    if (cases.keys() != jobs.keys() or cases.keys() & additions.keys()
            or len(additions) != extension["planned_new_development_episodes"]
            or additions.keys() != reference_digests.keys()):
        raise ValueError("MATCHED_PANEL_DECLARED_EXTENSION_COVERAGE_DIFFERS")
    new_cases, new_jobs = list(base_cases), list(base_jobs)
    for key, row in additions.items():
        matched_key = row["matched_identity"]["episode_id"]
        if matched_key not in cases:
            raise ValueError("MATCHED_PANEL_REFERENCE_CASE_MISSING")
        original, original_job = cases[matched_key], jobs[matched_key]
        identity, matched = row["identity"], row["matched_identity"]
        if (matched != original["identity"] or row["matched_reference"] != original["reference"]
                or row["matched_fine"] != original_job["native_archive"]
                or identity["circuit_id"] != matched["circuit_id"]
                or identity["condition_index"] != matched["condition_index"]
                or identity["block_index"] == matched["block_index"]
                or identity["block_index"] % 8 != matched["block_index"] % 8
                or identity["connectivity_seed"] != matched["connectivity_seed"]
                or identity["background_seed"] == matched["background_seed"]
                or identity["stimulus_seed"] == matched["stimulus_seed"]
                or identity["split"] != "train"):
            raise ValueError("MATCHED_PANEL_FIXED_CIRCUIT_OR_CONTROLS_DIFFER")
        if (row["native_archive"] == original["reference"]
                or row["fine_archive"] == original_job["native_archive"]
                or row["native_archive"] == row["fine_archive"]
                or row["native_job"] == row["fine_job"]):
            raise ValueError("MATCHED_PANEL_FRESH_NATIVE_EXECUTIONS_REQUIRED")
        new_cases.append(dict(identity=identity, reference=row["native_archive"],
                              delivery_sha256=reference_digests[key]))
        new_jobs.append(dict(identity=identity, native_job=row["fine_job"],
            comparison_job=row["comparison_job"], native_archive=row["fine_archive"],
            comparison_output=row["comparison_output"]))
    groups = {}
    for case in new_cases:
        identity = case["identity"]
        groups.setdefault(identity["block_id"], []).append(identity["condition_index"])
    if any(sorted(conditions) != list(range(10)) for conditions in groups.values()):
        raise ValueError("MATCHED_PANEL_COMPLETE_TEN_CONDITION_BLOCK_REQUIRED")
    return new_cases, new_jobs


def check_artifacts(directory: Path, artifacts: dict):
    total = 0
    for name, record in artifacts.items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("MATCHED_PANEL_ARTIFACT_CHANGED:" + str(path))
        total += record["bytes"]
    return total


def verify_reference_export(row: dict):
    """Authenticate the delivered complete episode without relabeling its split."""
    directory = Path(row["export_output"]) / "episode"
    benchmark = json.loads((directory / "report.json").read_text())
    export = json.loads((directory / "export/report.json").read_text())
    catalog_path = directory / "export/dataset/dataset.json"
    catalog = json.loads(catalog_path.read_text())
    if (benchmark["status"] != "COMPLETE_V1_BENCHMARK_DEVELOPMENT_EPISODE"
            or benchmark["candidate_identity"] != row["identity"]
            or benchmark["actual_split"] != "development" or benchmark["complete_episodes"] != 1
            or benchmark["units"] != 230924 or not benchmark["independent_native_replay_verified"]
            or benchmark["source_archives"]["native"] != row["native_archive"]
            or export["exit_code"] != 0 or export["source_units"] != 230924
            or str(export["native_job_id"]) != str(row["native_job"])
            or benchmark["dataset_catalog_sha256"] != sha256_file(catalog_path)
            or export["dataset_catalog_sha256"] != sha256_file(catalog_path)
            or catalog["complete_episodes"] != 1):
        raise ValueError("MATCHED_PANEL_COMPLETE_REFERENCE_EXPORT_REQUIRED")
    artifacts = {value["path"]: dict(bytes=(catalog_path.parent/value["path"]).stat().st_size,
                                      sha256=value["sha256"])
                 for value in [*catalog["circuits"].values(), *catalog["shards"]]}
    check_artifacts(catalog_path.parent, artifacts)
    episodes = [episode for shard in catalog["shards"] for episode in shard["episodes"]]
    if (len(episodes) != 1 or episodes[0]["dt_ms"] != .25
            or episodes[0]["duration_ticks"] != 12000 or episodes[0]["split"] != "development"
            or episodes[0]["episode_id"] != benchmark["episode_id"]):
        raise ValueError("MATCHED_PANEL_REFERENCE_DURATION_OR_SPLIT_DIFFERS")
    return dict(identity=row["identity"], native_job=row["native_job"], export_job=row["export_job"],
                exported_episode_id=benchmark["episode_id"], catalog_sha256=sha256_file(catalog_path),
                event_count=episodes[0]["event_count"], duration_ms=3000, units=230924)


def verify_panel(output: Path):
    """Verify delivered arrays and their declared physical time partitions."""
    plan = json.loads((output / "cases.json").read_text())
    submitted = json.loads((output / "envelope_submission.json").read_text())
    cases, jobs = indexed(plan["cases"]), indexed(submitted["cases"])
    task_dir, joint_dir = output / "task_summary/analysis", output / "joint_activity"
    task = json.loads((task_dir / "report.json").read_text())
    joint = json.loads((joint_dir / "report.json").read_text())
    count = len(cases)
    if (cases.keys() != jobs.keys() or task["complete_pairs"] != count
            or joint["pairs"] != count or joint["planned_pairs"] != count
            or task["exit_code"] != 0
            or {row["episode_id"] for row in joint["reports"]} != cases.keys()):
        raise ValueError("MATCHED_PANEL_COMPLETE_SUMMARIES_REQUIRED")
    task_bytes = check_artifacts(task_dir, task["artifacts"])
    report_hashes, joint_bytes, unit_checks, covariance_checks = {}, 0, 0, 0
    for key, case in cases.items():
        comparison, comparison_sha = accepted_comparison(case, jobs[key])
        path = joint_dir / "cases" / key / "report.json"
        report = json.loads(path.read_text())
        if (report["status"] != "COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PAIR"
                or report["identity"] != case["identity"] or report["comparison_sha256"] != comparison_sha
                or report["units"] != 230924 or report["populations"] != 21
                or report["diagnostic_bin_ms"] != 1 or report["lag_ms"] != [0, 1, 10, 100]):
            raise ValueError("MATCHED_PANEL_JOINT_REPORT_BINDING_DIFFERS")
        joint_bytes += check_artifacts(path.parent, report["artifacts"])
        report_hashes[key] = sha256_file(path)
        per_unit = Path(jobs[key]["comparison_output"]) / "analysis/per_unit.npz"
        check_artifacts(per_unit.parent, {"per_unit.npz": comparison["artifacts"]["per_unit.npz"]})
        totals = {}
        with np.load(per_unit, allow_pickle=False) as paired:
            for window, bins in (("full", 3000), ("pre_command", 501), ("command_response", 2499)):
                with np.load(path.parent / (window + ".npz"), allow_pickle=False) as data:
                    if (not np.array_equal(data["source_id"], paired["source_id"])
                            or data["source_id"].shape != (230924,)
                            or int(data["population_unit_counts"].sum()) != 230924):
                        raise ValueError("MATCHED_PANEL_UNIT_MAP_DIFFERS")
                    for side in ("reference", "fine"):
                        unit = data[side + "_unit_totals"]
                        total = data[side + "_total_counts"]
                        population = data[side + "_population_counts"]
                        covariance = data[side + "_population_count_covariance"]
                        if (unit.shape != (230924,) or total.shape != (bins,)
                                or population.shape != (bins, 21)
                                or int(unit.sum()) != int(total.sum())
                                or not np.array_equal(population.sum(axis=1), total)
                                or covariance.shape != (4, 21, 21) or not np.isfinite(covariance).all()):
                            raise ValueError("MATCHED_PANEL_COUNTS_OR_COVARIANCE_INVALID")
                        covariance_checks += covariance.size
                        if window == "command_response":
                            if not np.array_equal(unit, paired[side + "_command_response_counts"]):
                                raise ValueError("MATCHED_PANEL_COMMAND_COUNTS_DIFFER")
                            unit_checks += len(unit)
                        if window == "full":
                            expected = comparison["all_units"][side + "_spikes"] - report["terminal_events"][side]
                            if int(unit.sum()) != expected:
                                raise ValueError("MATCHED_PANEL_TERMINAL_EVENT_ACCOUNTING_DIFFERS")
                        totals[(window, side)] = unit.copy()
            for side in ("reference", "fine"):
                if not np.array_equal(totals[("full", side)],
                                      totals[("pre_command", side)] + totals[("command_response", side)]):
                    raise ValueError("MATCHED_PANEL_TEMPORAL_PARTITION_DIFFERS")
    acceptance = dict(status="PASS_COMPLETE_CLOCK_PANEL_DELIVERY", planned_pairs=count, verified_pairs=count,
        complete_blocks=task["complete_blocks"], circuit_identities=task["circuit_identities"],
        task_summary_report_sha256=sha256_file(task_dir / "report.json"),
        joint_activity_report_sha256=sha256_file(joint_dir / "report.json"),
        joint_report_sha256_by_episode=report_hashes,
        task_artifacts=len(task["artifacts"]), task_artifact_bytes=task_bytes,
        joint_artifacts=3*count, joint_artifact_bytes=joint_bytes,
        command_response_unit_counts_matched_to_accepted_comparisons=unit_checks,
        population_covariance_values_finite_and_shape_checked=covariance_checks,
        temporal_partition_counts_exact=True, scientific_qualification=False,
        scientific_bins_frozen=False, production_episodes=0, new_unique_episodes=0)
    freeze_json(output / "panel_acceptance.json", acceptance)
    load_accepted_panel(output)
    return acceptance


def build(base_panel: Path, extension: Path, output: Path):
    base_acceptance, base_plan, _, _ = load_accepted_panel(base_panel)
    base_jobs = json.loads((base_panel / "envelope_submission.json").read_text())["cases"]
    submitted = json.loads((extension / "submission.json").read_text())
    references, exports = {}, []
    for row in submitted["cases"]:
        exports.append(verify_reference_export(row))
        references[row["identity"]["episode_id"]] = sha256_file(Path(row["native_archive"]) / "delivery.json")
    cases, jobs = combine_tables(base_plan["cases"], base_jobs, submitted, references)
    for case, job in zip(cases, jobs):
        accepted_comparison(case, job)
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(status="PREPARED_COMBINED_V1_CLOCK_PANEL", full_units=230924, cases=cases,
        base_panel=str(base_panel.resolve()), base_acceptance_sha256=sha256_file(base_panel / "panel_acceptance.json"),
        extension=str(extension.resolve()), reused_pairs=base_acceptance["verified_pairs"],
        added_pairs=len(submitted["cases"]), scientific_qualification=False, production_episodes=0)
    freeze_json(output / "cases.json", provenance)
    freeze_json(output / "envelope_submission.json",
                dict(status="DELIVERED_SOURCE_JOBS_JOINED", cases=jobs, new_submissions=0))
    freeze_json(output / "reference_exports.json",
                dict(status="VERIFIED_COMPLETE_REFERENCE_EXPORTS", episodes=exports, complete_episodes=len(exports),
                     production_episodes=0, scientific_qualification=False))
    joint_cases = output / "joint_activity/cases"
    joint_cases.mkdir(parents=True)
    for case in base_plan["cases"]:
        key = case["identity"]["episode_id"]
        (joint_cases / key).symlink_to((base_panel / "joint_activity/cases" / key).resolve(), target_is_directory=True)
    summarize(output / "cases.json", output / "envelope_submission.json", output / "task_summary/analysis")
    joint_main(["--cases", str(output / "cases.json"), "--submissions", str(output / "envelope_submission.json"),
                "--output", str(output / "joint_activity")])
    result = verify_panel(output)
    print(json.dumps({k: result[k] for k in ("status", "verified_pairs", "complete_blocks", "circuit_identities")}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-panel", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble PFC development inputs from accepted complete reader deliveries.

Only metadata is read. Existing source/export authentication is reused; source
events, circuit arrays and model outputs are not read or generated here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .pfc_parameter_block_execution import CONDITIONS, digest, identity, load_plan
from .pfc_scale_analysis import SOURCE_ENTRY_SHA256


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def accepted_reader(path: Path) -> tuple[dict, Path, dict, dict]:
    receipt = read_json(path)
    require(receipt.get("status") == "ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK"
            and receipt.get("complete_development_episodes") == 10,
            "PFC_INPUT_COMPLETE_ACCEPTED_READER_REQUIRED")
    archive = Path(receipt["archive"])
    require(digest(archive / "delivery.json") == receipt["delivery_sha256"]
            and digest(archive / "report.json") == receipt["report_sha256"],
            "PFC_INPUT_READER_RECEIPT_CHANGED")
    delivery, report = (read_json(archive / name) for name in ("delivery.json", "report.json"))
    require(delivery.get("verified") is True
            and delivery.get("status") == report.get("status") == "PASS_PFC_TEN_CONDITION_SOURCE_READBACK"
            and delivery.get("job_id") == report.get("job_id") == receipt["job_id"]
            and report.get("exit_code") == 0 and len(report["circuits"]) == 1,
            "PFC_INPUT_READER_DELIVERY_MISMATCH")
    files = {row["path"]: row for row in delivery["files"]}
    require(len(files) == len(delivery["files"]), "PFC_INPUT_DUPLICATE_DELIVERY_PATH")
    require(files["report.json"]["sha256"] == receipt["report_sha256"],
            "PFC_INPUT_REPORT_MANIFEST_MISMATCH")
    return receipt, archive, report, files


def assemble(baseline: Path, validation: Path, base: Path, plan: Path,
             reader_jobs: list[str]) -> tuple[dict, dict]:
    """Keep the validated block0 cohort and add accepted block1/2 groups."""
    require(len(reader_jobs) == len(set(reader_jobs)) == 10
            and all(job.replace("_", "").isdecimal() for job in reader_jobs),
            "PFC_INPUT_TEN_UNIQUE_READER_JOBS_REQUIRED")
    prior_validation = read_json(validation)
    binding = next(row for row in prior_validation["files"] if row["path"] == baseline.name)
    require(baseline.stat().st_size == binding["bytes"] and digest(baseline) == binding["sha256"],
            "PFC_INPUT_VALIDATED_BASELINE_CHANGED")
    spec = read_json(baseline)
    require(set(spec) == {"schema", "source_entry", "episodes"}
            and spec["schema"] == "pfc_population_development_inputs_v1"
            and digest(Path(spec["source_entry"])) == SOURCE_ENTRY_SHA256,
            "PFC_INPUT_BASELINE_SCHEMA_OR_SOURCE_MISMATCH")
    expected_baseline = {
        f"pfc_working_memory_circuit_{circuit:03d}_block_000_condition_{offset:02d}": role
        for circuit, role in ((23, "development_fit"), (63, "development_fit"),
                              (74, "development_validation")) for offset in range(10)}
    require(len(spec["episodes"]) == 30
            and {row["episode_id"]: row["partition"] for row in spec["episodes"]} == expected_baseline,
            "PFC_INPUT_ORIGINAL_COMPLETE_BLOCK0_COHORT_REQUIRED")
    receipt_bindings = baseline.parent / "input_receipt_bindings.json"
    prior_binding = next(row for row in prior_validation["files"]
                         if row["path"] == receipt_bindings.name)
    require(digest(receipt_bindings) == prior_binding["sha256"],
            "PFC_INPUT_BASELINE_RECEIPT_BINDINGS_CHANGED")
    receipts = []
    for binding in read_json(receipt_bindings):
        path = Path(binding["path"])
        require(digest(path) == binding["sha256"], "PFC_INPUT_BASELINE_ACCEPTANCE_CHANGED")
        receipt, _, _, _ = accepted_reader(path)
        require(receipt["delivery_sha256"] == binding["authenticated_delivery_sha256"]
                and receipt["report_sha256"] == binding["authenticated_report_sha256"],
                "PFC_INPUT_BASELINE_ACCEPTANCE_BINDING_MISMATCH")
        receipts.append(dict(path=str(path), sha256=binding["sha256"], block_index=0,
                             circuit_index=receipt["circuit_index"], bulk_validation_reused=True))
    tasks = load_plan(plan, require_release=True)
    groups, seen = [], set(expected_baseline)
    for group, job in enumerate(reader_jobs):
        selected = tasks[group * 10:(group + 1) * 10]
        circuit, block = (int(selected[0][name]) for name in ("circuit_index", "block_index"))
        role = selected[0]["split"]
        require(len(selected) == 10 and [row["condition_id"] for row in selected] == list(CONDITIONS)
                and len({(row["circuit_index"], row["block_index"], row["split"])
                         for row in selected}) == 1,
                "PFC_INPUT_COMPLETE_CANONICAL_CONDITION_GROUP_REQUIRED")
        path = base / f"parameter_block_readback_acceptance_{job}.json"
        cell = dict(reader_job=job, circuit_index=circuit, block_index=block,
                    intended_split=role, complete_episodes=0, status="PENDING_READER_ACCEPTANCE")
        groups.append(cell)
        if not path.exists():
            continue
        receipt, archive, report, files = accepted_reader(path)
        require(receipt["job_id"] == job and receipt["circuit_index"] == circuit
                and receipt["block_index"] == block,
                "PFC_INPUT_READER_CIRCUIT_BLOCK_MISMATCH")
        require([row["condition_id"] for row in receipt["sources"]] == list(CONDITIONS),
                "PFC_INPUT_ALL_TEN_SOURCE_CONDITIONS_REQUIRED")
        source_bindings = report["circuits"][0]["source_bindings"]
        require(len(source_bindings) == 10, "PFC_INPUT_TEN_REPORT_SOURCE_BINDINGS_REQUIRED")
        for task, source, reported in zip(selected, receipt["sources"], source_bindings):
            name, episode = task["condition_id"], task["episode_id"]
            require(source["episode_id"] == reported["episode_id"] == episode
                    and source["condition_id"] == reported["condition_id"] == name
                    and source["source_archive"] == reported["source_archive"]
                    and source["delivery_sha256"] == reported["delivery_sha256"],
                    "PFC_INPUT_SOURCE_IDENTITY_MISMATCH")
            relative = f"circuit_{circuit:03d}/block_{block:03d}/{name}_dataset/dataset/dataset.json"
            catalog_path = archive / relative
            require(catalog_path.stat().st_size == files[relative]["bytes"]
                    and digest(catalog_path) == files[relative]["sha256"],
                    "PFC_INPUT_EXPORTED_CATALOG_CHANGED")
            catalog = read_json(catalog_path)
            binding = catalog["binding"]
            candidate = binding["resolved_source_config"]["benchmark_episode_candidate"]
            require(candidate["identity"] == identity(task)
                    and candidate["intended_split"] == role
                    and binding["config_sha256"] == task["config_sha256"]
                    and binding["scope"] == "source_reproduction"
                    and binding["source_entry_sha256"] == SOURCE_ENTRY_SHA256,
                    "PFC_INPUT_EXPORTED_PROSPECTIVE_IDENTITY_MISMATCH")
            require(catalog["complete_episodes"] == 1 and len(catalog["circuits"]) == 1
                    and next(iter(catalog["circuits"].values()))["units"] == 100000
                    and len(catalog["shards"]) == 1 and len(catalog["shards"][0]["episodes"]) == 1,
                    "PFC_INPUT_COMPLETE_FULL_UNIT_EPISODE_REQUIRED")
            exported = catalog["shards"][0]["episodes"][0]
            require(exported["episode_id"] == episode and exported["duration_ticks"] == 100000
                    and exported["dt_ms"] == .05 and exported["split"] == "development",
                    "PFC_INPUT_EXPORTED_EPISODE_CLOCK_MISMATCH")
            origin = Path(source["source_archive"])
            require(digest(origin / "delivery.json") == source["delivery_sha256"],
                    "PFC_INPUT_SOURCE_DELIVERY_CHANGED")
            require(episode not in seen, "PFC_INPUT_DUPLICATE_EPISODE")
            seen.add(episode)
            if role in ("train", "validation"):
                spec["episodes"].append(dict(dataset=str(catalog_path.parent), episode_id=episode,
                    source_run=str(origin / "source_run"),
                    partition="development_fit" if role == "train" else "development_validation"))
            else:
                require(role == "calibration", "PFC_INPUT_UNREGISTERED_PROSPECTIVE_ROLE")
        cell.update(status="ACCEPTED_EXCLUDED_CALIBRATION" if role == "calibration"
                    else "ACCEPTED_TRAINING_INPUT", complete_episodes=10)
        receipts.append(dict(path=str(path), sha256=digest(path), circuit_index=circuit,
                             block_index=block, bulk_validation_reused=True))
    counts = Counter(row["partition"] for row in spec["episodes"])
    ready = all(cell["complete_episodes"] == 10 for cell in groups
                if cell["intended_split"] in ("train", "validation"))
    coverage = dict(schema="pfc_population_development_input_coverage_v1",
        status="COMPLETE_FIT_VALIDATION_INPUT_COVERAGE" if ready else "PARTIAL_FIT_VALIDATION_INPUT_COVERAGE",
        fitting_episodes=counts["development_fit"], validation_episodes=counts["development_validation"],
        fitting_target_episodes=80, validation_target_episodes=30, declared_complete_episode_envelope=140,
        accepted_export_episodes=40 + sum(cell["complete_episodes"] for cell in groups),
        reserved_calibration_episodes=10 + sum(cell["complete_episodes"] for cell in groups
                                             if cell["intended_split"] == "calibration"),
        fit_validation_coverage_ready=ready, reader_groups=groups, receipt_bindings=receipts,
        baseline_inputs=str(baseline), baseline_inputs_sha256=digest(baseline),
        baseline_validation=str(validation), baseline_validation_sha256=digest(validation),
        plan=str(plan), reader_jobs=reader_jobs, bulk_authentication_repeated=False,
        raw_source_read_repeated=False, source_simulations=0, optimizer_updates=0,
        training_submitted=False, production_episodes=0, scientific_fidelity_qualified=False)
    return spec, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "baseline-validation", "base", "plan", "output", "coverage"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reader-jobs", nargs=10, required=True)
    args = parser.parse_args()
    spec, coverage = assemble(args.baseline, args.baseline_validation, args.base, args.plan, args.reader_jobs)
    require(args.output != args.baseline, "PFC_INPUT_PRESERVE_VALIDATED_BASELINE")
    content = json.dumps(spec, indent=2) + "\n"
    if not args.output.exists() or args.output.read_text() != content:
        args.output.write_text(content)
    coverage.update(inputs=str(args.output), inputs_sha256=digest(args.output))
    args.coverage.write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps({key: coverage[key] for key in
                     ("status", "fitting_episodes", "validation_episodes", "fit_validation_coverage_ready")}))


if __name__ == "__main__":
    main()

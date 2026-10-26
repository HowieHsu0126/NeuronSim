"""Synchronize PFC development progress from already authenticated receipts.

This command reads small metadata only. It does not run source simulations,
repeat full-unit readback, qualify scientific gates, or modify Slurm jobs.
Run it under the shared management I/O lock when used on mgmt2.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from .pfc_parameter_block_execution import (
    complete_source_batch, group_rows, load_plan, source_path,
)
from .provenance import canonical_json, sha256_file

BRANCH_PATH = ("pfc_clock_refinement", "refinement_repeat", "first_passage_history",
               "shared_profile_history", "task_boundary_execution_20261011")
EXECUTION_KEY = "parameter_block_execution_20261012"
STATE_PATH = "Output/neural_population_dynamics/operations/continuation_resources_581147.json"
BASE_PATH = "Output/neural_population_dynamics/benchmark/pfc_parameter_design"


def branch_of(state: dict) -> dict:
    for key in BRANCH_PATH:
        state = state[key]
    return state


def metadata(path: Path, expected_hash: str | None = None) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("PFC_PROGRESS_METADATA_TOO_LARGE")
    if expected_hash is not None and sha256_file(path) != expected_hash:
        raise ValueError("PFC_PROGRESS_METADATA_BINDING_CHANGED:" + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, error: str) -> None:
    if not condition:
        raise ValueError(error)


def source_record(root: Path, base: Path, row: dict, array: str) -> tuple[dict, dict]:
    index = int(row["task_index"])
    path = base / f"parameter_block_source_acceptance_{array}_{index}.json"
    receipt = metadata(path)
    archive = source_path(base, row) / ("run_" + receipt["native_job_id"])
    require(receipt["status"] == "ACCEPTED_COMPLETE_FULL_SCALE_TASK_SOURCE_ARCHIVE"
            and receipt["slurm_array_job"] == f"{array}_{index}"
            and receipt["task_index"] == index
            and receipt["episode_id"] == row["episode_id"]
            and receipt["circuit_index"] == int(row["circuit_index"])
            and receipt["block_index"] == int(row["block_index"])
            and receipt["condition"] == row["condition_id"]
            and Path(receipt["archive"]) == archive
            and receipt["config_sha256"] == row["config_sha256"]
            and receipt["core_units"] == 100000 and receipt["duration_ms"] == 5000
            and receipt["complete_source_episodes"] == 1
            and receipt["scientific_fidelity_qualified"] is False
            and receipt["parameter_domain_qualified"] is False
            and receipt["production_episodes"] == 0,
            "PFC_PROGRESS_SOURCE_IDENTITY_MISMATCH")
    batch, _ = complete_source_batch(base, row)
    require(batch.name == archive.name, "PFC_PROGRESS_SOURCE_SUCCESS_CHANGED")
    metadata(batch / "delivery.json", receipt["batch_delivery_sha256"])
    delivery = metadata(archive / "delivery.json", receipt["source_delivery_sha256"])
    report = metadata(archive / "source_run/report.json", receipt["source_report_sha256"])
    require(delivery["source_execution_complete"] is True
            and delivery["job_id"] == receipt["native_job_id"]
            and delivery["case_id"] == row["episode_id"]
            and report["case_id"] == row["episode_id"]
            and report["source_execution_complete"] is True
            and report["config_sha256"] == row["config_sha256"]
            and report["resolved_config"]["benchmark_episode_candidate"]["identity"]["episode_id"]
            == row["episode_id"], "PFC_PROGRESS_SOURCE_METADATA_CHANGED")
    require(report["simulator_cost"] == receipt["simulator_cost"],
            "PFC_PROGRESS_SOURCE_COST_CHANGED")
    record = dict(index=index, receipt=str(path.relative_to(root)),
                  receipt_sha256=sha256_file(path), native_job_id=receipt["native_job_id"],
                  scheduler_seconds=receipt["scheduler_elapsed_seconds"],
                  verified_files=receipt["verified_files"], verified_bytes=receipt["verified_bytes"],
                  source_report_sha256=receipt["source_report_sha256"],
                  simulator_wall_seconds=receipt["simulator_cost"]["wall_seconds"],
                  acceptance_job=receipt["acceptance_job"])
    record["acceptance_step"] = receipt.get("acceptance_step")
    if receipt["bulk_source_authentication_reused"]:
        record["bulk_authentication_reused"] = True
    return record, receipt


def reader_record(root: Path, base: Path, selected: list[dict], group: int,
                  job: str, sources: dict[int, dict]) -> dict:
    path = base / f"parameter_block_readback_acceptance_{job}.json"
    receipt = metadata(path)
    archive = base / ("parameter_block_readback_" + job)
    require(receipt["status"] == "ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK"
            and receipt["job_id"] == job and Path(receipt["archive"]) == archive
            and receipt["circuit_index"] == int(selected[0]["circuit_index"])
            and receipt["block_index"] == int(selected[0]["block_index"])
            and receipt["complete_development_episodes"] == 10
            and receipt["complete_parameter_points"] == 1
            and receipt["production_episodes"] == 0
            and receipt["new_source_simulations"] == 0
            and all(receipt[key] is False for key in (
                "scientific_fidelity_qualified", "source_probability_law_qualified",
                "parameter_domain_qualified", "task_success_assessed")),
            "PFC_PROGRESS_READER_IDENTITY_MISMATCH")
    delivery = metadata(archive / "delivery.json", receipt["delivery_sha256"])
    report = metadata(archive / "report.json", receipt["report_sha256"])
    require(delivery["job_id"] == report["job_id"] == job
            and delivery["verified"] is True
            and report["status"] == "PASS_PFC_TEN_CONDITION_SOURCE_READBACK"
            and report["exit_code"] == 0
            and len(report["circuits"]) == 1, "PFC_PROGRESS_READER_METADATA_CHANGED")
    circuit = report["circuits"][0]
    bindings = circuit["source_bindings"]
    require(len(bindings) == len(receipt["sources"]) == 10,
            "PFC_PROGRESS_COMPLETE_TEN_SOURCE_GROUP_REQUIRED")
    attempts = []
    for row, binding, accepted in zip(selected, bindings, receipt["sources"]):
        index = int(row["task_index"])
        require(index in sources, "PFC_PROGRESS_READER_SOURCE_NOT_ACCEPTED")
        source = sources[index]
        require(binding["episode_id"] == accepted["episode_id"] == row["episode_id"]
                and binding["condition_id"] == accepted["condition_id"] == row["condition_id"]
                and binding["source_archive"] == accepted["source_archive"] == source["archive"]
                and binding["delivery_sha256"] == accepted["delivery_sha256"]
                == source["source_delivery_sha256"]
                and binding["complete_units"] == accepted["complete_units"] == 100000,
                "PFC_PROGRESS_READER_SOURCE_IDENTITY_MISMATCH")
        source_attempts = binding.get("source_attempts", [])
        if len(source_attempts) > 1:
            attempts.extend(source_attempts)
    record = dict(group=group, receipt=str(path.relative_to(root)), receipt_sha256=sha256_file(path),
                  job_id=job, scheduler_seconds=receipt["scheduler_elapsed_seconds"],
                  circuit_index=receipt["circuit_index"], block_index=receipt["block_index"],
                  complete_development_episodes=10, verified_files=receipt["verified_files"],
                  verified_bytes=receipt["verified_bytes"], temporal_bins=receipt["temporal_bins"],
                  paired_contrast_rows=receipt["paired_contrast_rows"],
                  all_unit_bin_values=receipt["all_unit_bin_values_checked_by_existing_export"],
                  events=receipt["events_verified_by_existing_export"],
                  delivery_sha256=receipt["delivery_sha256"], report_sha256=receipt["report_sha256"],
                  bulk_validation_reused=True, production_episodes=0)
    if attempts:
        record["source_attempts"] = attempts
    return record


def merge_records(previous: list[dict], current: list[dict], key: str) -> list[dict]:
    def indexed(records: list[dict]) -> dict[int, dict]:
        result = {}
        for record in records:
            index = record[key]
            require(type(index) is int and 0 <= index < (100 if key == "index" else 10),
                    "PFC_PROGRESS_INVALID_LOGICAL_INDEX")
            require(index not in result, "PFC_PROGRESS_DUPLICATE_LOGICAL_RECEIPT")
            result[index] = copy.deepcopy(record)
        return result
    old, new = indexed(previous), indexed(current)
    require(old.keys() <= new.keys(), "PFC_PROGRESS_ACCEPTED_RECEIPT_MISSING")
    for index, prior in old.items():
        item = new[index]
        for field, value in prior.items():
            # Recovery handles and failed-attempt accounting are management metadata.
            # Preserve them; receipt-owned identities and metrics still must agree.
            recovery_fields = {"actual_slurm_array_task", "logical_task",
                               "original_failed_raw_job", "original_failed_scheduler_seconds"}
            if key == "index" and field in recovery_fields and field not in item:
                item[field] = value
                continue
            # Management accounting can add a raw job/time absent from compute-node receipts.
            if field == "native_job_id" and key == "group" and field not in item:
                item[field] = value
            elif field == "scheduler_seconds" and item.get(field) is None and field in item:
                item[field] = value
            else:
                require(field in item and item[field] == value,
                        "PFC_PROGRESS_ACCEPTED_RECEIPT_CHANGED:" + field)
    return [new[index] for index in sorted(new)]


def synchronize_branch(branch: dict, source_records: list[dict], reader_records: list[dict]) -> dict:
    """Return a deep copy of the execution branch, preserving scheduler and scientific fields."""
    sources = merge_records(branch.get("accepted_new_source_receipts", []), source_records, "index")
    readers = merge_records(branch.get("accepted_new_readback_receipts", []), reader_records, "group")
    indices = {row["index"] for row in sources}
    for reader in readers:
        require(set(range(reader["group"] * 10, reader["group"] * 10 + 10)) <= indices
                and reader["complete_development_episodes"] == 10,
                "PFC_PROGRESS_COMPLETE_TEN_SOURCE_GROUP_REQUIRED")
    require(branch.get("complete_new_source_episodes", 0)
            == len(branch.get("accepted_new_source_receipts", []))
            and branch.get("complete_new_readback_episodes", 0)
            == 10 * len(branch.get("accepted_new_readback_receipts", [])),
            "PFC_PROGRESS_PREVIOUS_COUNT_DISAGREEMENT")
    result = copy.deepcopy(branch)
    summary = branch.get("descriptive_summary_acceptance")
    if summary is not None:
        require(len(sources) == 100 and len(readers) == 10
                and summary["complete_development_episodes"] == 100
                and summary["parameter_comparison_rows"] == 1800,
                "PFC_PROGRESS_COMPLETE_SUMMARY_REQUIRES_ALL_CELLS")
    status = ("COMPLETE_BOUNDED_PARAMETER_BLOCK_DEVELOPMENT" if summary is not None else
              f"ACTIVE_BOUNDED100_DEVELOPMENT_{len(sources)}_SOURCES_{10 * len(readers)}_READBACKS_ACCEPTED")
    result.update(accepted_new_source_receipts=sources, accepted_new_readback_receipts=readers,
                  complete_new_source_episodes=len(sources), complete_new_readback_episodes=10 * len(readers),
                  status=status)
    return result


def replace_state_pair(state_path: Path, mirror_path: Path, original_state_bytes: bytes,
                       original_mirror_bytes: bytes, updated_state: dict) -> None:
    """Reject conflicting snapshots; atomic file replacements leave partial failure detectable."""
    require(branch_of(json.loads(original_state_bytes)) == json.loads(original_mirror_bytes),
            "PFC_PROGRESS_MIRROR_DISAGREEMENT")
    state_bytes = canonical_json(updated_state).encode("utf-8")
    mirror_bytes = canonical_json(branch_of(updated_state)).encode("utf-8")
    staged = []
    try:
        for path, payload in ((state_path, state_bytes), (mirror_path, mirror_bytes)):
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pfc_progress_", delete=False) as stream:
                staged.append(Path(stream.name))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        require(state_path.read_bytes() == original_state_bytes
                and mirror_path.read_bytes() == original_mirror_bytes,
                "PFC_PROGRESS_CONCURRENT_STATE_CHANGE")
        os.replace(staged[0], state_path)
        require(state_path.read_bytes() == state_bytes
                and mirror_path.read_bytes() == original_mirror_bytes,
                "PFC_PROGRESS_CONCURRENT_STATE_CHANGE")
        os.replace(staged[1], mirror_path)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def synchronize(root: Path, *, write: bool = False) -> dict:
    root = root.resolve()
    base, state_path = root / BASE_PATH, root / STATE_PATH
    mirror_path = base / "task_boundary_submission.json"
    original_state, original_mirror = state_path.read_bytes(), mirror_path.read_bytes()
    state = json.loads(original_state)
    branch = branch_of(state)
    require(branch == json.loads(original_mirror), "PFC_PROGRESS_MIRROR_DISAGREEMENT")
    execution = branch[EXECUTION_KEY]
    metadata(root / execution["plan"] / "delivery.json", execution["plan_delivery_sha256"])
    rows = load_plan(root / execution["plan"], require_release=True)
    sources, receipts = [], {}
    array = str(execution["source_array_job"])
    for row in rows:
        index = int(row["task_index"])
        if (base / f"parameter_block_source_acceptance_{array}_{index}.json").exists():
            record, receipt = source_record(root, base, row, array)
            sources.append(record)
            receipts[index] = receipt
    readers = []
    for group in range(10):
        override = execution.get("reader_group_job_overrides", {}).get(str(group))
        job = str(override) if override is not None else f"{execution['reader_array_job']}_{group}"
        if (base / f"parameter_block_readback_acceptance_{job}.json").exists():
            readers.append(reader_record(root, base, group_rows(rows, group), group, job, receipts))
    current = copy.deepcopy(execution)
    summary_path = base / f"parameter_block_response_summary_acceptance_{execution['descriptive_summary_job']}.json"
    if summary_path.exists():
        acceptance = metadata(summary_path)
        require(acceptance["status"] == "ACCEPTED_COMPLETE_PFC_PARAMETER_BLOCK_RESPONSE_SUMMARY"
                and acceptance["job_id"] == str(execution["descriptive_summary_job"])
                and acceptance["complete_development_episodes"] == 100
                and acceptance["parameter_comparison_rows"] == 1800
                and acceptance["comparison_values_exactly_recomputed"] == 1800
                and acceptance["max_absolute_comparison_difference"] == 0
                and acceptance["production_episodes"] == 0,
                "PFC_PROGRESS_SUMMARY_ACCEPTANCE_MISMATCH")
        archive = base / f"parameter_block_response_summary_{execution['descriptive_summary_job']}"
        require(Path(acceptance["archive"]) == archive, "PFC_PROGRESS_SUMMARY_ARCHIVE_MISMATCH")
        metadata(archive / "delivery.json", acceptance["delivery_sha256"])
        report = metadata(archive / "report.json", acceptance["report_sha256"])
        require(report["source_readers"] == [row["job_id"] for row in readers]
                and report["stop_rule_reached"] is True and report["exit_code"] == 0,
                "PFC_PROGRESS_SUMMARY_READER_BINDINGS_CHANGED")
        summary = dict(receipt=str(summary_path.relative_to(root)), receipt_sha256=sha256_file(summary_path),
                       report_sha256=acceptance["report_sha256"], delivery_sha256=acceptance["delivery_sha256"],
                       complete_development_episodes=100, parameter_comparison_rows=1800)
        require(current.get("descriptive_summary_acceptance", summary) == summary,
                "PFC_PROGRESS_ACCEPTED_SUMMARY_CHANGED")
        current["descriptive_summary_acceptance"] = summary
    else:
        require("descriptive_summary_acceptance" not in current, "PFC_PROGRESS_ACCEPTED_SUMMARY_MISSING")
    updated = synchronize_branch(current, sources, readers)
    changed = updated != execution
    if changed and write:
        branch[EXECUTION_KEY] = updated
        n, r = updated["complete_new_source_episodes"], updated["complete_new_readback_episodes"]
        branch["status"] = (f"ACTIVE_{branch['source_task_episodes_complete']}_TASKS_"
                            f"{branch['complete_task_readback_episodes']}_READBACKS_"
                            f"{branch['accepted_history_tasks']}_HISTORIES_"
                            f"{branch['receipt_qualified_pairs']}_PAIRS_{n}_NEW_SOURCES_{r}_NEW_READBACKS_ACCEPTED")
        branch["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        branch["consecutive_dependency_wait_turns"] = 0
        branch["current_goal_turn_classification"] = "PROGRESS_ACCEPTED_PARAMETER_BLOCK_RECEIPTS"
        branch["next_action"] = ("Bounded100 development and descriptive comparison are complete. "
                                 "Continue the existing PFC training/evaluation chain and unresolved scientific qualification; "
                                 "production0/100000." if "descriptive_summary_acceptance" in updated else
                                 f"Continue existing bounded100 allocation and exact ten-condition readers; "
                                 f"accepted sources{n}/100, readbacks{r}/100. Consume only terminal deliveries. "
                                 "Complete original100cell descriptive comparison. Numerical/source-law/parameter/"
                                 "task/observation gates remain unqualified; production0/100000.")
        replace_state_pair(state_path, mirror_path, original_state, original_mirror, state)
    return dict(status="PFC_PARAMETER_BLOCK_PROGRESS_SYNC", changed=changed, written=changed and write,
                accepted_source_episodes=updated["complete_new_source_episodes"],
                accepted_readback_episodes=updated["complete_new_readback_episodes"],
                production_episodes=updated["production_episodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true", help="Update the existing state and mirror after checks")
    args = parser.parse_args()
    print(canonical_json(synchronize(args.root, write=args.write)), end="")


if __name__ == "__main__":
    main()

"""Synchronize PFC history progress from accepted receipts and small metadata.

The baseline retains its previous bulk authentication. Only explicit incremental
receipt indices are inspected. This command does not qualify scientific gates,
repeat integration or bulk hashing, or modify scheduler state. Use the shared
management I/O lock on mgmt2. The default mode is read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import tempfile

import yaml

from .pfc_parameter_block_progress import (
    BRANCH_PATH, STATE_PATH, branch_of, metadata, require, replace_state_pair,
)
from .pfc_profile_history import source_requests
from .pfc_profile_history_readback import readback_requests
from .pfc_history_profile_census import SOURCE_CLOCKS
from .provenance import canonical_json, sha256_file

BASE_PATH = "Output/neural_population_dynamics/benchmark/pfc_first_passage_performance"
RECOVERY_PATH = BASE_PATH + "/history_reader_dependency_recovery_20261013"
BASELINE_PATH = RECOVERY_PATH + "/completed_intake_acceptance.json"
MIRROR_PATH = "Output/neural_population_dynamics/benchmark/pfc_parameter_design/task_boundary_submission.json"
SOURCE_PREFIX = "Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/profile_history/complete/run_610289"
READER_PREFIX = "Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/profile_history_readback/complete/run_610296"
SOURCE_INDICES = range(81, 160)
READER_INDICES = range(60, 124)
SOURCE_CONFIG = "Libs/configs/neural_population_pfc_profile_history.yaml"
READER_CONFIG = "Libs/configs/neural_population_pfc_profile_history_readback.yaml"
COVERAGE_CONFIG = "Libs/configs/neural_population_pfc_profile_history_coverage.yaml"


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _metadata(path: Path, snapshots: dict[Path, str], expected: str | None = None) -> dict:
    require(path.stat().st_size <= 16 * 1024 * 1024, "PFC_HISTORY_PROGRESS_METADATA_TOO_LARGE")
    digest = sha256_file(path)
    require(expected is None or digest == expected,
            "PFC_HISTORY_PROGRESS_METADATA_BINDING_CHANGED:" + str(path))
    value = metadata(path, digest)
    require(sha256_file(path) == digest, "PFC_HISTORY_PROGRESS_METADATA_CHANGED:" + str(path))
    snapshots[path] = digest
    return value


def _yaml(path: Path, snapshots: dict[Path, str], expected: str) -> dict:
    require(path.stat().st_size <= 16 * 1024 * 1024, "PFC_HISTORY_PROGRESS_METADATA_TOO_LARGE")
    digest = sha256_file(path)
    require(digest == expected, "PFC_HISTORY_PROGRESS_CONFIG_BINDING_CHANGED:" + str(path))
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and sha256_file(path) == digest,
            "PFC_HISTORY_PROGRESS_CONFIG_CHANGED:" + str(path))
    snapshots[path] = digest
    return value


def _entries(delivery: dict) -> dict[str, dict]:
    entries = {row["path"]: row for row in delivery["files"]}
    require(len(entries) == len(delivery["files"]), "PFC_HISTORY_PROGRESS_DUPLICATE_DELIVERY_PATH")
    return entries


def _unqualified(report: dict) -> None:
    require(report["production_episodes"] == 0
            and all(report[key] is False for key in (
                "scientific_fidelity_qualified", "confirmation_execution_released"))
            and all(report[key] is False for key in (
                "source_history_population_uncertainty_established", "population_confidence_interval_computed",
                "third_spike_probability_or_bound_qualified", "source_probability_law_qualified") if key in report),
            "PFC_HISTORY_PROGRESS_SCIENTIFIC_GATE_CHANGED")
    require(all(report[key] == 0 for key in (
        "new_source_law_episodes", "new_source_simulations", "new_independent_source_blocks",
        "new_neural_trajectories", "new_full_network_episodes") if key in report),
        "PFC_HISTORY_PROGRESS_PRODUCTION_SCOPE_CHANGED")


def _profile_keys(keys: list, expected: list[str]) -> None:
    require(isinstance(keys, list) and len(keys) == len(set(keys)),
            "PFC_HISTORY_PROGRESS_DUPLICATE_PROFILE")
    require(keys == expected, "PFC_HISTORY_PROGRESS_FROZEN_PROFILE_SET_CHANGED")


def frozen_requests(root: Path, branch: dict, snapshots: dict[Path, str]) -> tuple[dict, dict]:
    """Use the frozen source and reader request builders, with their original bindings."""
    recovery = branch["history_reader_dependency_recovery_20261013"]
    coverage = _path(root, recovery["coverage_preparation"])
    delivery = _metadata(coverage / "delivery.json", snapshots, recovery["coverage_preparation_sha256"])
    require(delivery["verified"] is True, "PFC_HISTORY_PROGRESS_PREPARATION_NOT_VERIFIED")
    entries = _entries(delivery)
    settings = _yaml(coverage / COVERAGE_CONFIG, snapshots, entries[COVERAGE_CONFIG]["sha256"])
    _unqualified(settings)
    require(entries[COVERAGE_CONFIG]["sha256"] == recovery["coverage_config_sha256"],
            "PFC_HISTORY_PROGRESS_COVERAGE_CONFIG_CHANGED")
    history = _yaml(coverage / "coverage_inputs/foundation/source_config.yaml", snapshots,
                    entries["coverage_inputs/foundation/source_config.yaml"]["sha256"])
    require(snapshots[coverage / "coverage_inputs/foundation/source_config.yaml"]
            == settings["source_config_sha256"], "PFC_HISTORY_PROGRESS_SOURCE_CONFIG_CHANGED")
    _unqualified(history)
    reader_prepare = _path(root, recovery["recovery_preparation"])
    reader_delivery = _metadata(reader_prepare / "delivery.json", snapshots,
                                recovery["recovery_preparation_delivery_sha256"])
    reader_settings = _yaml(reader_prepare / READER_CONFIG, snapshots,
                            _entries(reader_delivery)[READER_CONFIG]["sha256"])
    require(snapshots[reader_prepare / READER_CONFIG] == recovery["recovery_config_sha256"]
            and reader_settings["source_config_sha256"] == settings["source_config_sha256"],
            "PFC_HISTORY_PROGRESS_READER_CONFIG_CHANGED")
    _unqualified(reader_settings)
    require(history["required_source_seeds"] == list(range(95001, 95033))
            and history["expected_units"] == 100000,
            "PFC_HISTORY_PROGRESS_ORIGINAL_SOURCE_DOMAIN_CHANGED")
    registry_path = _path(root, history["source_preparation"]["archive"])
    registry_delivery = _metadata(registry_path / "delivery.json", snapshots,
                                  history["source_preparation"]["delivery_sha256"])
    registry_entries = _entries(registry_delivery)
    registry = _metadata(registry_path / "source_index.json", snapshots,
                         registry_entries["source_index.json"]["sha256"])
    require(registry["profiles"] == history["profiles"]
            and sorted(map(int, registry["source_blocks"])) == history["required_source_seeds"],
            "PFC_HISTORY_PROGRESS_SOURCE_REGISTRY_CHANGED")
    sources = {}
    labels = list(SOURCE_CLOCKS)
    for index in range(80, 160):
        seed, label = history["required_source_seeds"][index // len(labels)], labels[index % len(labels)]
        original = registry["source_blocks"][str(seed)]["sources"][label]
        original_metadata = original["metadata"]
        binding = dict(archive=original["archive"], delivery_sha256=original["delivery_sha256"],
                       case_path=str(Path(original["raw"]["path"]).parent),
                       report_sha256=registry_entries[original_metadata + "/report.json"]["sha256"],
                       refinement_seeds=original["refinement_seeds"])
        sources[index] = dict(seed=seed, label=label, binding=binding,
                              profiles=[row["key"] for row in source_requests(history, label, pilot=False, seed=seed)],
                              config_sha256=settings["source_config_sha256"],
                              source_inputs={"source_inputs/source/" + name:
                                             registry_entries[original_metadata + "/" + name]["sha256"]
                                             for name in ("config.yaml", "report.json", "delivery.json")})
        sources[index]["source_inputs"]["source_inputs/source/raw.npz"] = original["raw"]["sha256"]
    requests = {row["group_index"]: row for row in readback_requests(reader_settings, history)}
    manifest = _metadata(root / RECOVERY_PATH / "requests.json", snapshots)
    require([row["index"] for row in manifest] == list(range(43, 124)),
            "PFC_HISTORY_PROGRESS_REQUEST_INDEX_CHANGED")
    for row in manifest:
        request = requests[row["index"]]
        require(row["seed"] == request["source_rng_seed"] and row["group"] == request["group_name"]
                and row["paired_profiles"] == len(request["profile_keys"])
                and row["sources"] == {clock: Path(run["archive"]).name.removeprefix("run_")
                                       for clock, run in request["source_runs"].items()},
                "PFC_HISTORY_PROGRESS_EXACT_REQUEST_BINDING_CHANGED")
    for request in requests.values():
        request["config_sha256"] = recovery["recovery_config_sha256"]
    return sources, requests


def _execution_metadata(root: Path, archive: Path, receipt: dict, job: str, task: str,
                        snapshots: dict[Path, str]) -> tuple[dict, dict]:
    delivery = _metadata(archive / "delivery.json", snapshots, receipt["delivery_sha256"])
    report = _metadata(archive / "report.json", snapshots, receipt["report_sha256"])
    analysis = _metadata(archive / "analysis/report.json", snapshots, receipt["analysis_sha256"])
    status = "COMPLETE_PFC_PROFILE_HISTORY" + ("_READBACK" if task == "profile_readback" else "") + "_COMPLETE"
    require(delivery["job_id"] == report["job_id"] == job
            and delivery["status"] == report["status"] == status
            and delivery["verified"] is True and report["exit_code"] == 0
            and report["task"] == task and report["mode"] == "complete",
            "PFC_HISTORY_PROGRESS_EXECUTION_IDENTITY_MISMATCH")
    _unqualified(report)
    _unqualified(analysis)
    return report, analysis


def source_record(root: Path, index: int, expected: dict,
                  snapshots: dict[Path, str]) -> tuple[dict, dict]:
    path = root / BASE_PATH / f"source_history_acceptance_610289_{index}.json"
    receipt = _metadata(path, snapshots)
    source = receipt["source"]
    job = f"610289_{index}"
    archive = root / (SOURCE_PREFIX + f"_{index}")
    require(receipt["status"] == "ACCEPTED_PFC_ORIGINAL_SOURCE_HISTORY"
            and source["job_id"] == job and _path(root, source["archive"]) == archive
            and source["status"] == "COMPLETE_PFC_PROFILE_HISTORY_COMPLETE"
            and isinstance(source["native_job_id"], str) and bool(source["native_job_id"])
            and source["source_rng_seed"] == expected["seed"]
            and source["source_label"] == expected["label"]
            and source["source_binding"] == expected["binding"]
            and receipt["new_accepted_history_tasks"] == 1
            and receipt["new_accepted_paired_profiles"] == 0,
            "PFC_HISTORY_PROGRESS_SOURCE_IDENTITY_MISMATCH")
    _unqualified(receipt)
    report, analysis = _execution_metadata(root, archive, source, job, "profile_history", snapshots)
    inputs = {row["path"]: row["sha256"] for row in report["inputs"]}
    require(inputs[SOURCE_CONFIG] == expected["config_sha256"]
            and all(inputs.get(key) == value for key, value in expected["source_inputs"].items()),
            "PFC_HISTORY_PROGRESS_SOURCE_INPUT_BINDING_CHANGED")
    require(analysis["status"] == source["analysis_status"] == "COMPLETE_PFC_PROFILE_SOURCE_HISTORY"
            and analysis["source_rng_seed"] == expected["seed"]
            and analysis["source_label"] == expected["label"] and analysis["pilot"] is False
            and source["complete_profile_mappings"] == len(expected["profiles"]),
            "PFC_HISTORY_PROGRESS_SOURCE_ANALYSIS_IDENTITY_MISMATCH")
    _profile_keys(analysis["requested_profile_keys"], expected["profiles"])
    _profile_keys(analysis["completed_profile_keys"], expected["profiles"])
    _profile_keys([row["key"] for row in analysis["frame_reports"]], expected["profiles"])
    require(all(analysis[key] == source[key] for key in (
        "reconstruction", "reconstruction_seconds", "reconstruction_drive", "total_inner_trials",
        "distinct_selected_physical_windows", "distinct_integrated_references", "peak_rss_kib")),
        "PFC_HISTORY_PROGRESS_SOURCE_ANALYSIS_BINDING_CHANGED")
    record = dict(index=index, receipt=str(path.relative_to(root)), receipt_sha256=snapshots[path],
                  job_id=job, native_job_id=source["native_job_id"], source_rng_seed=expected["seed"],
                  source_label=expected["label"], profile_keys=expected["profiles"],
                  delivery_sha256=source["delivery_sha256"], report_sha256=source["report_sha256"],
                  analysis_sha256=source["analysis_sha256"])
    return record, analysis


def reader_record(root: Path, index: int, expected: dict, source_requests_by_index: dict,
                  accepted_sources: set[int], snapshots: dict[Path, str]) -> dict:
    path = root / BASE_PATH / f"paired_history_acceptance_610296_{index}.json"
    receipt = _metadata(path, snapshots)
    job = f"610296_{index}"
    archive = root / (READER_PREFIX + f"_{index}")
    require(receipt["status"] == "ACCEPTED_PFC_PAIRED_PROFILE_HISTORY_READBACK"
            and receipt["job_id"] == job and receipt["readback_group_index"] == index
            and _path(root, receipt["archive"]) == archive
            and receipt["source_rng_seed"] == expected["source_rng_seed"]
            and receipt["readback_group"] == expected["group_name"]
            and receipt["paired_profiles"] == receipt["new_accepted_paired_profiles"] == len(expected["profile_keys"]),
            "PFC_HISTORY_PROGRESS_READER_IDENTITY_MISMATCH")
    _unqualified(receipt)
    _profile_keys(receipt["paired_profile_keys"], expected["profile_keys"])
    report, analysis = _execution_metadata(root, archive, receipt, job, "profile_readback", snapshots)
    inputs = {row["path"]: row["sha256"] for row in report["inputs"]}
    require(inputs[READER_CONFIG] == expected["config_sha256"],
            "PFC_HISTORY_PROGRESS_READER_INPUT_BINDING_CHANGED")
    require(analysis["status"] == "COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK"
            and analysis["source_rng_seed"] == expected["source_rng_seed"]
            and analysis["readback_group_index"] == index
            and analysis["readback_group"] == expected["group_name"]
            and analysis["paired_profiles"] == len(expected["profile_keys"])
            and analysis["full_source_requests"] is True
            and analysis["source_costs"] == receipt["source_costs"],
            "PFC_HISTORY_PROGRESS_READER_ANALYSIS_BINDING_CHANGED")
    _profile_keys([row["key"] for row in analysis["pairs"]], expected["profile_keys"])
    require(set(receipt["source_costs"]) == {"coarse", "fine"},
            "PFC_HISTORY_PROGRESS_TWO_SOURCE_BINDINGS_REQUIRED")
    for clock, run in expected["source_runs"].items():
        source_index = int(Path(run["archive"]).name.rsplit("_", 1)[1])
        require(source_index in accepted_sources, "PFC_HISTORY_PROGRESS_READER_SOURCE_NOT_ACCEPTED")
        cost = receipt["source_costs"][clock]
        source = source_requests_by_index[source_index]
        require(_path(root, cost["history_execution_archive"]) == root / run["archive"]
                and cost["history_execution_job_id"] == f"610289_{source_index}"
                and cost["source_binding"] == source["binding"]
                and source["seed"] == expected["source_rng_seed"]
                and source["label"] == run["source_label"],
                "PFC_HISTORY_PROGRESS_READER_SOURCE_BINDING_CHANGED")
    return dict(index=index, receipt=str(path.relative_to(root)), receipt_sha256=snapshots[path],
                job_id=job, source_rng_seed=expected["source_rng_seed"], readback_group=expected["group_name"],
                paired_profile_keys=expected["profile_keys"], delivery_sha256=receipt["delivery_sha256"],
                report_sha256=receipt["report_sha256"], analysis_sha256=receipt["analysis_sha256"])


def _preserve_records(previous: list[dict], current: list[dict]) -> None:
    old = {row["index"]: row for row in previous}
    new = {row["index"]: row for row in current}
    require(len(old) == len(previous) and len(new) == len(current),
            "PFC_HISTORY_PROGRESS_DUPLICATE_LOGICAL_RECEIPT")
    require(old.keys() <= new.keys(), "PFC_HISTORY_PROGRESS_ACCEPTED_RECEIPT_MISSING")
    for index, record in old.items():
        require(record == new[index], "PFC_HISTORY_PROGRESS_ACCEPTED_RECEIPT_CHANGED")


def _ledger_counts(original: bytes, previous_sources: int, previous_pairs: int,
                 sources: int, pairs: int) -> bytes:
    text = original.decode("utf-8")
    for field, previous, current in (("pfc_history_accepted_original_sources", previous_sources, sources),
                                     ("pfc_history_accepted_pairs", previous_pairs, pairs)):
        pattern = re.compile(r"^( *" + field + r": *)([0-9]+)( *\r?)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        require(len(matches) == 1 and int(matches[0].group(2)) == previous,
                "PFC_HISTORY_PROGRESS_STATE_LEDGER_STALE:" + field)
        text = pattern.sub(lambda match: match.group(1) + str(current) + match.group(3), text)
    return text.encode("utf-8")


def synchronize(root: Path, *, write: bool = False) -> dict:
    root = root.resolve()
    state_path, mirror_path, ledger_path = root / STATE_PATH, root / MIRROR_PATH, root / "state_ledger.yaml"
    original_state, original_mirror, original_ledger = (
        state_path.read_bytes(), mirror_path.read_bytes(), ledger_path.read_bytes())
    state = json.loads(original_state)
    branch = branch_of(state)
    require(branch == json.loads(original_mirror), "PFC_PROGRESS_MIRROR_DISAGREEMENT")
    snapshots: dict[Path, str] = {}
    baseline = _metadata(root / BASELINE_PATH, snapshots)
    require(baseline["status"] == "ACCEPTED_COMPLETED_ORIGINAL_HISTORIES_AND_RECOVERED_READERS"
            and baseline["accepted_sources"] == 76 and baseline["accepted_pairs"] == 1168
            and baseline["required_original_sources"] == 155 and baseline["required_pairs"] == 2304
            and baseline["production_episodes"] == 0 and baseline["scientific_fidelity_qualified"] is False
            and [row["index"] for row in baseline["sources"]] == list(range(59, 81))
            and {row["index"] for row in baseline["readers"]} == set(range(43, 60)),
            "PFC_HISTORY_PROGRESS_BASELINE_CHANGED")
    source_expected, reader_expected = frozen_requests(root, branch, snapshots)
    source_records, reader_records = [], []
    candidate_paths = {root / BASE_PATH / f"source_history_acceptance_610289_{index}.json": False
                       for index in SOURCE_INDICES}
    candidate_paths.update({root / BASE_PATH / f"paired_history_acceptance_610296_{index}.json": False
                            for index in READER_INDICES})
    for path in candidate_paths:
        candidate_paths[path] = path.exists()
    for index in SOURCE_INDICES:
        if candidate_paths[root / BASE_PATH / f"source_history_acceptance_610289_{index}.json"]:
            record, _ = source_record(root, index, source_expected[index], snapshots)
            source_records.append(record)
    accepted_sources = {row["index"] for row in baseline["sources"]} | {row["index"] for row in source_records}
    for index in READER_INDICES:
        if candidate_paths[root / BASE_PATH / f"paired_history_acceptance_610296_{index}.json"]:
            reader_records.append(reader_record(root, index, reader_expected[index], source_expected,
                                                accepted_sources, snapshots))
    pairs: set[tuple[int, str]] = set()
    for reader in reader_records:
        for key in reader["paired_profile_keys"]:
            identity = (reader["source_rng_seed"], key)
            require(identity not in pairs, "PFC_HISTORY_PROGRESS_DUPLICATE_PROFILE")
            pairs.add(identity)
    accepted_source_count, accepted_pair_count = 76 + len(source_records), 1168 + len(pairs)
    previous = branch.get("history_receipt_progress", {})
    if previous:
        require(previous["accepted_sources"] == branch["accepted_history_tasks"]
                == 76 + len(previous["source_receipts"])
                and previous["accepted_pairs"] == branch["receipt_qualified_pairs"]
                == 1168 + sum(len(row["paired_profile_keys"]) for row in previous["reader_receipts"]),
                "PFC_HISTORY_PROGRESS_PREVIOUS_COUNT_DISAGREEMENT")
    _preserve_records(previous.get("source_receipts", []), source_records)
    _preserve_records(previous.get("reader_receipts", []), reader_records)
    bindings = {str(path.relative_to(root)): digest for path, digest in snapshots.items()}
    for path, digest in previous.get("metadata_sha256", {}).items():
        require(bindings.get(path) == digest, "PFC_HISTORY_PROGRESS_ACCEPTED_METADATA_CHANGED:" + path)
    require(76 <= branch["accepted_history_tasks"] <= accepted_source_count <= 155
            and 1168 <= branch["receipt_qualified_pairs"] <= accepted_pair_count <= 2304
            and branch["production_episodes"] == 0,
            "PFC_HISTORY_PROGRESS_PREVIOUS_COUNT_DISAGREEMENT")
    progress = dict(status="PFC_HISTORY_RECEIPT_PROGRESS", baseline_receipt=BASELINE_PATH,
                    baseline_sources=76, baseline_pairs=1168, required_original_sources=155,
                    required_pairs=2304, source_receipts=source_records, reader_receipts=reader_records,
                    metadata_sha256=bindings, accepted_sources=accepted_source_count,
                    accepted_pairs=accepted_pair_count, production_episodes=0,
                    bulk_authentication_reused=True)
    updated = copy.deepcopy(branch)
    updated.update(accepted_history_tasks=accepted_source_count, receipt_qualified_pairs=accepted_pair_count,
                   history_receipt_progress=progress)
    status, replacements = re.subn(r"_[0-9]+_HISTORIES_[0-9]+_PAIRS_",
                                   f"_{accepted_source_count}_HISTORIES_{accepted_pair_count}_PAIRS_", branch["status"])
    require(replacements == 1, "PFC_HISTORY_PROGRESS_STATUS_FORMAT_CHANGED")
    updated["status"] = status
    updated_ledger = _ledger_counts(original_ledger, branch["accepted_history_tasks"], branch["receipt_qualified_pairs"],
                                accepted_source_count, accepted_pair_count)
    changed = updated != branch or updated_ledger != original_ledger
    if write and changed:
        updated_state = json.loads(original_state)
        cursor = updated_state
        for key in BRANCH_PATH[:-1]:
            cursor = cursor[key]
        cursor[BRANCH_PATH[-1]] = updated
        staged = None
        try:
            with tempfile.NamedTemporaryFile(dir=ledger_path.parent, prefix=".pfc_history_progress_", delete=False) as stream:
                staged = Path(stream.name)
                stream.write(updated_ledger)
                stream.flush()
                os.fsync(stream.fileno())
            require(state_path.read_bytes() == original_state and mirror_path.read_bytes() == original_mirror
                    and ledger_path.read_bytes() == original_ledger,
                    "PFC_HISTORY_PROGRESS_CONCURRENT_STATE_CHANGE")
            require(all(sha256_file(path) == digest for path, digest in snapshots.items())
                    and all(path.exists() == exists for path, exists in candidate_paths.items()),
                    "PFC_HISTORY_PROGRESS_CONCURRENT_RECEIPT_CHANGE")
            replace_state_pair(state_path, mirror_path, original_state, original_mirror, updated_state)
            require(ledger_path.read_bytes() == original_ledger, "PFC_HISTORY_PROGRESS_STATE_LEDGER_STALE")
            os.replace(staged, ledger_path)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
    return dict(status="PFC_HISTORY_PROGRESS_SYNC", changed=changed, written=write and changed,
                accepted_original_sources=accepted_source_count, required_original_sources=155,
                accepted_pairs=accepted_pair_count, required_pairs=2304, production_episodes=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true", help="Update existing state, mirror, and two ledger counts")
    args = parser.parse_args()
    print(canonical_json(synchronize(args.root, write=args.write)), end="")


if __name__ == "__main__":
    main()

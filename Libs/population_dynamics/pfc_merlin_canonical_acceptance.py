"""Merge immutable original intake batches and verified independent recoveries.

All inputs are local metadata. The source progress and prior artifacts remain
unchanged. This merger grants no source, scaling or production qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "pfc_merlin_scientific_intake_aggregation_v1"
BATCH_SCHEMA = "pfc_merlin_scientific_intake_v1"
FLAGS = ("source_qualification_granted", "scaling_qualified", "scientific_parameter_domain_qualified")
RECOVERY = {1906}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    data = Path(path).read_bytes()
    value = json.loads(data)
    require(isinstance(value, dict), "METADATA_OBJECT_REQUIRED")
    return value, hashlib.sha256(data).hexdigest()


def hash_valid(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def counter(value, field):
    result = value.get(field)
    require(type(result) is int and result >= 0, "COUNTER_INVALID:" + field)
    return result


def indices(value):
    require(isinstance(value, list) and all(type(index) is int and 0 <= index < 3000 for index in value),
            "TASK_INDICES_INVALID")
    require(len(value) == len(set(value)), "TASK_INDICES_DUPLICATE")
    return set(value)


def scope(value):
    require(counter(value, "production_episodes") == counter(value, "source_simulations") == 0,
            "SCIENTIFIC_SCOPE_CHANGED")
    require(all(value.get(field) is False for field in FLAGS), "QUALIFICATION_SCOPE_CHANGED")
    require(value.get("failures") == [], "FAILED_INTAKE")


def execution_key(value):
    require(isinstance(value, dict) and value.get("backend") == "merlin"
            and isinstance(value.get("run_id"), str) and value["run_id"]
            and isinstance(value.get("trial_id"), str) and value["trial_id"], "EXECUTION_IDENTITY_INVALID")
    return {key: item for key, item in value.items() if key != "recorded_utc"}


def completed_indices(items, checkpoint):
    require(isinstance(items, list), "COMPLETED_RECORDS_INVALID")
    for item in items:
        require(isinstance(item, dict) and type(item.get("index")) is int
                and hash_valid(item.get("acceptance_sha256"))
                and item.get("acceptance_uri") == checkpoint + f"/task_{item['index']:04d}_acceptance.json",
                "COMPLETED_IDENTITY_HASH_PATH_CHANGED")
    return indices([item["index"] for item in items])


def row_binding(row, base, number, execution):
    checkpoint = base + f"/batch_{number:04d}/pfc_scientific_intake"
    require(isinstance(row, dict) and row.get("acceptance_uri") == checkpoint + "/acceptance.json"
            and hash_valid(row.get("acceptance_sha256"))
            and row.get("run_id") == execution["run_id"] and row.get("trial_id") == execution["trial_id"],
            "BATCH_SOURCE_HASH_PATH_CHANGED")
    if "batch" in row:
        require(type(row["batch"]) is int and row["batch"] == number, "BATCH_SEQUENCE_GAP")
    if "batch_checkpoint_path" in row:
        require(row["batch_checkpoint_path"] == base + f"/batch_{number:04d}", "BATCH_SEQUENCE_GAP")
    if "failures" in row:
        require(row["failures"] == [], "FAILED_BATCH_ROW")
    new = completed_indices(row.get("completed"), checkpoint)
    require(len(new) == counter(row, "new_accepted_episode_count") and new, "BATCH_COUNT_MISMATCH")
    return new


def check_batch(row, path, reference, checkpoint):
    value, digest = read(path)
    require(digest == row["acceptance_sha256"], "IMMUTABLE_BATCH_HASH_CHANGED")
    require(value.get("schema") == BATCH_SCHEMA and value.get("status") == "PASS", "BATCH_NOT_PASS")
    scope(value)
    require(execution_key(value.get("execution")) == execution_key(reference.get("execution")),
            "SOURCE_EXECUTION_VERSION_CHANGED")
    require(hash_valid(value.get("config_sha256")) and hash_valid(value.get("control_validation_sha256"))
            and value["control_validation_sha256"] == reference["control_validation_sha256"]
            and value.get("control_validation") == reference.get("control_validation"), "CONTROL_BINDING_CHANGED")
    require(value.get("checkpoint_path") == checkpoint and value.get("completed") == row["completed"],
            "BATCH_COMPLETED_IDENTITY_CHANGED")
    new = completed_indices(value["completed"], checkpoint)
    require(new == indices(value.get("requested_task_indices"))
            and len(new) == counter(value, "new_accepted_episode_count") == row["new_accepted_episode_count"],
            "PARTIAL_OR_MISMATCHED_BATCH")
    cumulative = indices(value.get("accepted_task_indices"))
    require(len(cumulative) == counter(value, "accepted_episode_count") and new <= cumulative,
            "BATCH_CUMULATIVE_COUNT_CHANGED")
    return value, new, cumulative



def task_source_binding(record, directory):
    """Read only the task JSON named by its report-bound SHA; never load bundles."""
    require(directory is not None, "TASK_ACCEPTANCE_DIRECTORY_REQUIRED")
    accepted, digest = read(Path(directory) / (record["acceptance_sha256"] + ".json"))
    require(digest == record["acceptance_sha256"], "IMMUTABLE_TASK_ACCEPTANCE_HASH_CHANGED")
    index = record["index"]
    require(accepted.get("status") == "ACCEPTED_MERLIN_DEVELOPMENT_EPISODE"
            and type(accepted.get("index")) is int and accepted["index"] == index,
            "TASK_ACCEPTANCE_IDENTITY_CHANGED")
    require(counter(accepted, "source_simulations") == counter(accepted, "production_episodes") == 0,
            "SCIENTIFIC_SCOPE_CHANGED")
    require(all(accepted.get(field) is False for field in FLAGS), "QUALIFICATION_SCOPE_CHANGED")
    receipt = accepted.get("remote_receipt")
    require(isinstance(receipt, dict) and receipt.get("status") == "COMPLETE"
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and receipt.get("persisted") is True and type(receipt.get("index")) is int
            and receipt["index"] == index, "TASK_SOURCE_RECEIPT_INVALID")
    require(accepted.get("receipt_uri") == record["source_receipt_uri"]
            if "source_receipt_uri" in record else isinstance(accepted.get("receipt_uri"), str),
            "TASK_SOURCE_RECEIPT_URI_CHANGED")
    for field in ("run_id", "trial_id"):
        require(isinstance(accepted.get(field), str) and accepted[field]
                and accepted[field] == receipt.get(field), "TASK_SOURCE_EXECUTION_CHANGED")
    xie_job = receipt.get("xie_job")
    require(type(xie_job) in (str, int) and str(xie_job), "TASK_SOURCE_JOB_REQUIRED")
    if "xie_job" in accepted:
        require(accepted["xie_job"] == xie_job, "TASK_SOURCE_JOB_CHANGED")
    checkpoint = receipt.get("checkpoint_path")
    recovery = receipt.get("publication_recovery", {})
    require(isinstance(checkpoint, str) and checkpoint.startswith("hdfs://")
            and isinstance(recovery, dict), "TASK_SOURCE_BUNDLE_URI_INVALID")
    uri = recovery.get("source_bundle_uri", checkpoint.rstrip("/") + "/result.tar.gz")
    require(isinstance(uri, str) and uri.startswith("hdfs://")
            and accepted.get("source_bundle_uri") == uri
            and hash_valid(receipt.get("result_sha256"))
            and accepted.get("source_bundle_sha256") == receipt["result_sha256"]
            and type(receipt.get("result_bytes")) is int and receipt["result_bytes"] > 0,
            "TASK_SOURCE_BUNDLE_BINDING_CHANGED")
    identity, observation = accepted.get("identity"), accepted.get("observation_entry")
    require(isinstance(identity, dict) and isinstance(identity.get("episode_id"), str)
            and identity["episode_id"], "TASK_EPISODE_IDENTITY_REQUIRED")
    require(isinstance(observation, dict)
            and set(observation) == {"path", "development_report_sha256", "observation_report_sha256"}
            and isinstance(observation["path"], str) and observation["path"]
            and hash_valid(observation["development_report_sha256"])
            and hash_valid(observation["observation_report_sha256"]), "TASK_OBSERVATION_BINDING_REQUIRED")
    return dict(index=index, identity=identity, run_id=accepted["run_id"], trial_id=accepted["trial_id"],
                xie_job=xie_job, source_bundle_uri=uri, source_bundle_sha256=receipt["result_sha256"],
                source_bundle_bytes=receipt["result_bytes"], observation_entry=observation)


def prior_recovery_bindings(prior, recovered):
    records = prior.get("independent_recovery_bindings", [])
    require(isinstance(records, list), "RECOVERY_BINDINGS_INVALID")
    bindings = {}
    for record in records:
        require(isinstance(record, dict) and type(record.get("index")) is int
                and record["index"] in recovered and record["index"] not in bindings
                and hash_valid(record.get("acceptance_sha256"))
                and isinstance(record.get("acceptance_uri"), str)
                and isinstance(record.get("source_binding"), dict)
                and record["source_binding"].get("index") == record["index"], "RECOVERY_BINDINGS_INVALID")
        bindings[record["index"]] = record
    # Historical task1906 priors did not contain the per-task binding. Keep that
    # path valid; an actual overlap requires its verified recovery report.
    require(recovered - set(bindings) <= RECOVERY, "INDEPENDENT_RECOVERY_BINDING_REQUIRED")
    return bindings


def check_recovery(path, reference, known, directory):
    value, digest = read(path)
    require(value.get("schema") == BATCH_SCHEMA and value.get("status") == "PASS", "RECOVERY_NOT_PASS")
    scope(value)
    execution_key(value.get("execution"))
    require(hash_valid(value.get("config_sha256")) and hash_valid(value.get("prior_acceptance_sha256"))
            and hash_valid(value.get("control_validation_sha256"))
            and value["control_validation_sha256"] == reference["control_validation_sha256"]
            and value.get("control_validation") == reference.get("control_validation"), "CONTROL_BINDING_CHANGED")
    checkpoint = value.get("checkpoint_path")
    require(isinstance(checkpoint, str) and checkpoint.startswith("hdfs://"), "RECOVERY_CHECKPOINT_REQUIRED")
    new = completed_indices(value.get("completed"), checkpoint)
    require(new and new == indices(value.get("requested_task_indices"))
            and len(new) == counter(value, "new_accepted_episode_count"), "PARTIAL_OR_MISMATCHED_RECOVERY")
    cumulative = indices(value.get("accepted_task_indices"))
    require(len(cumulative) == counter(value, "accepted_episode_count") and new <= cumulative
            and cumulative - new <= known, "RECOVERY_CUMULATIVE_IDENTITY_CHANGED")
    bindings = []
    for record in value["completed"]:
        bindings.append(dict(record, source_binding=task_source_binding(record, directory)))
    descriptor = dict(acceptance=str(Path(path).resolve()), acceptance_sha256=digest,
                      acceptance_uri=checkpoint + "/acceptance.json", execution=value["execution"],
                      completed=value["completed"], requested_task_indices=sorted(new))
    return descriptor, bindings


def merge(prior_path, progress_path, reference_path, batch_directory, output,
          recovery_reports=(), task_acceptance_directory=None):
    prior_path, progress_path, reference_path, batch_directory, output = map(
        lambda value: Path(value).resolve(), (prior_path, progress_path, reference_path, batch_directory, output))
    recovery_reports = tuple(Path(path).resolve() for path in recovery_reports)
    require(not output.exists(), "FRESH_OUTPUT_REQUIRED")
    prior, prior_hash = read(prior_path)
    wrapper, progress_hash = read(progress_path)
    reference, reference_hash = read(reference_path)
    require(prior.get("schema") == SCHEMA and prior.get("status") == "PASS", "CANONICAL_PRIOR_INVALID")
    scope(prior)
    canonical = indices(prior.get("accepted_task_indices"))
    recovered = indices(prior.get("independent_recovery_indices"))
    if "original_continuous_task_indices" in prior:
        original = indices(prior["original_continuous_task_indices"])
        require(original | recovered == canonical, "INDEPENDENT_RECOVERY_BINDING_CHANGED")
    else:
        require(recovered == RECOVERY and recovered <= canonical, "INDEPENDENT_RECOVERY_BINDING_CHANGED")
        original = canonical - recovered
    bindings = prior_recovery_bindings(prior, recovered)
    old_original_count = len(original)
    count = counter(prior, "accepted_episode_count")
    offset = counter(prior, "continuous_intake_verified_batches")
    require(count == len(canonical) and counter(prior, "original_continuous_accepted_episode_count") == len(original)
            and count == counter(prior, "prior_accepted_episode_count") + counter(prior, "new_accepted_episode_count"),
            "INCONSISTENT_PRIOR_COUNTERS")
    require(isinstance(wrapper.get("progress"), dict), "FULL_WRAPPED_PROGRESS_REQUIRED")
    progress = wrapper["progress"]
    require(progress.get("schema") == SCHEMA and type(progress.get("target")) is int and progress["target"] == 3000,
            "ORIGINAL_PROGRESS_SCHEMA_CHANGED")
    scope(progress)
    remote = indices(progress.get("accepted_task_indices"))
    require(len(remote) == counter(progress, "accepted_episode_count"),
            "ORIGINAL_RECOVERY_OVERLAP_OR_COUNT_CHANGED")
    base = progress.get("checkpoint_path")
    require(isinstance(base, str) and base.startswith("hdfs://")
            and wrapper.get("uri") == base + "/progress.json", "ORIGINAL_PROGRESS_URI_CHANGED")
    expected_execution = execution_key(reference.get("execution"))
    observed_execution = execution_key(progress.get("execution"))
    require(all(key in expected_execution and item == expected_execution[key]
                for key,item in observed_execution.items()), "ORIGINAL_PROGRESS_EXECUTION_CHANGED")
    rows = progress.get("batches")
    require(isinstance(rows, list) and 1 <= offset <= len(rows)
            and (offset < len(rows) or recovery_reports), "NO_LATER_COMPLETE_BATCHES")
    all_new, prefix_new = set(), set()
    original_records = {}
    for number, row in enumerate(rows, 1):
        new = row_binding(row, base, number, expected_execution)
        require(not all_new & new, "OVERLAPPING_PROGRESS_TASKS")
        original_records.update((record["index"], record) for record in row["completed"])
        all_new |= new
        if number <= offset:
            prefix_new |= new
    require(all_new <= remote and len(all_new) == counter(progress, "new_accepted_episode_count"),
            "PROGRESS_BATCH_COUNT_CHANGED")
    initial = remote - all_new
    require(len(initial) == counter(progress, "prior_accepted_episode_count")
            and len(remote) == len(initial) + len(all_new)
            and original == initial | prefix_new, "PROGRESS_PREFIX_IDENTITY_CHANGED")
    prefix_rows = prior.get("batches")
    prior_original_new = counter(prior, "new_original_continuous_episode_count") if (
        "new_original_continuous_episode_count" in prior) else counter(prior, "new_accepted_episode_count")
    require(isinstance(prefix_rows, list) and len(prefix_rows) <= offset
            and (prefix_rows or ("new_original_continuous_episode_count" in prior and prior_original_new == 0)),
            "PRIOR_BATCH_PREFIX_REQUIRED")
    prior_new = set()
    for number, row in enumerate(prefix_rows, offset - len(prefix_rows) + 1):
        new = row_binding(row, base, number, expected_execution)
        actual = rows[number - 1]
        require(all(row.get(field) == actual.get(field) for field in
                    ("acceptance_sha256", "acceptance_uri", "completed", "new_accepted_episode_count", "run_id", "trial_id")),
                "PRIOR_PROGRESS_BATCH_BINDING_CHANGED")
        require(not prior_new & new and new <= original, "PRIOR_BATCH_IDENTITIES_CHANGED")
        prior_new |= new
    require(len(prior_new) == prior_original_new, "PRIOR_BATCH_COUNTER_CHANGED")
    require(reference_hash == rows[offset - 1]["acceptance_sha256"], "REFERENCE_HASH_CHANGED")
    _, _, reference_ids = check_batch(rows[offset - 1], reference_path, reference,
                                      base + f"/batch_{offset:04d}/pfc_scientific_intake")
    require(reference_ids == original, "REFERENCE_CANONICAL_PREFIX_CHANGED")
    integrated = []
    for number in range(offset + 1, len(rows) + 1):
        row = rows[number - 1]
        path = batch_directory / f"batch_{number:04d}_acceptance_package_bound.json"
        _, new, cumulative = check_batch(row, path, reference, base + f"/batch_{number:04d}/pfc_scientific_intake")
        require(not original & new, "OVERLAPPING_CANONICAL_TASKS")
        original |= new
        require(original == cumulative, "CUMULATIVE_IDENTITY_UNION_CHANGED")
        integrated.append(dict(batch=number, acceptance=str(path), acceptance_sha256=row["acceptance_sha256"],
            acceptance_uri=row["acceptance_uri"], completed=row["completed"],
            new_accepted_episode_count=row["new_accepted_episode_count"], run_id=row["run_id"], trial_id=row["trial_id"]))
    require(original == remote, "COMPLETE_PROGRESS_UNION_REQUIRED")
    recovery_descriptors = prior.get("independent_recovery_reports", [])
    require(isinstance(recovery_descriptors, list), "RECOVERY_REPORTS_INVALID")
    recovery_descriptors = list(recovery_descriptors)
    for path in recovery_reports:
        descriptor, task_bindings = check_recovery(path, reference, canonical | original, task_acceptance_directory)
        for binding in task_bindings:
            index = binding["index"]
            if index in bindings:
                require(bindings[index]["source_binding"] == binding["source_binding"],
                        "RECOVERY_SOURCE_CONFLICT:" + str(index))
            else:
                bindings[index] = binding
            recovered.add(index)
        if not any(row.get("acceptance_sha256") == descriptor["acceptance_sha256"] for row in recovery_descriptors):
            recovery_descriptors.append(descriptor)
    for index in sorted(original & recovered):
        require(index in bindings and index in original_records, "OVERLAP_SOURCE_BINDING_REQUIRED:" + str(index))
        require(task_source_binding(original_records[index], task_acceptance_directory) == bindings[index]["source_binding"],
                "RECOVERY_SOURCE_CONFLICT:" + str(index))
    canonical = original | recovered
    result = dict(prior)
    result.update(accepted_episode_count=len(canonical), accepted_task_indices=sorted(canonical),
        original_continuous_accepted_episode_count=len(original), continuous_intake_verified_batches=len(rows),
        original_continuous_task_indices=sorted(original),
        independent_recovery_indices=sorted(recovered),
        independent_recovery_bindings=[bindings[index] for index in sorted(bindings)],
        independent_recovery_reports=recovery_descriptors, batches=integrated,
        new_original_continuous_episode_count=len(original) - old_original_count,
        new_accepted_episode_count=len(canonical) - count, prior_accepted_episode_count=count,
        prior_acceptance=str(prior_path), prior_acceptance_sha256=prior_hash,
        progress_snapshot=str(progress_path), progress_snapshot_sha256=progress_hash,
        recorded_utc=datetime.now(timezone.utc).isoformat(), validator_source=str(Path(__file__).resolve()),
        validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(result["new_original_continuous_episode_count"] == sum(row["new_accepted_episode_count"] for row in integrated),
            "CANONICAL_NEW_COUNTER_CHANGED")
    scope(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--batch-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path, action="append", default=[])
    parser.add_argument("--task-acceptance-directory", type=Path,
                        help="Task metadata files named <acceptance_sha256>.json")
    args = parser.parse_args(argv)
    result = merge(args.prior, args.progress, args.reference, args.batch_directory, args.output,
                   args.recovery_report, args.task_acceptance_directory)
    print(json.dumps({field:result[field] for field in ("status", "accepted_episode_count",
        "original_continuous_accepted_episode_count", "continuous_intake_verified_batches", "new_accepted_episode_count")}))


if __name__ == "__main__":
    main()

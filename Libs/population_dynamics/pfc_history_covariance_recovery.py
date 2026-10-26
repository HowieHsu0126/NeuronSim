"""Recover exactly 218 missing covariance groups and reuse collector delta caches.

CLI request schema pfc_history_covariance_recovery_v1 binds analysis_request and
completed_snapshot with local path/bytes/sha256 descriptors. collector has
checkpoint_uri (ending /pfc_history_covariance_collection), run_id and trial_id.
The original 766-group snapshot and 16 supplemental groups exclude all previous
computation. Only HDFS -cat nonzero reads retry, with at most three attempts.
Optional source_acceptances are indexed path/bytes/sha256 task JSON bindings.
Optional resume_execution binds run_id/trial_id/checkpoint_uri; resume_checkpoints
is an ordered list of receipt and archive path/bytes/sha256 bindings from batch1.
Resumed compute also requires resume_source_terminal, a bound original final
receipt/failure JSON or a bound operations execution record confirming failure.
Cache preparation alone does not release ownership; complete1000 intake is reused.
Neither option changes the frozen analysis, numerical core or1000-group target.
source_replacements is limited to tasks2641/2717. Each record binds index,
original_origin and source_readout_request (local path/bytes/sha256). The latter
is validated as source input; its existence does not prove snapshot publication.
Its accepted_source and PASS scientific_intake prove the real native replacement.
The frozen recipe, streams, numerical source and original origin remain bound.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import tarfile
import time

from . import pfc_history_coincidences as core
from . import pfc_history_covariance_intake as intake

SCHEMA = "pfc_history_covariance_recovery_v1"
COLLECTOR_SCHEMA = "pfc_history_covariance_collection_v1"
CORE_SHA256 = "291681f95d82d8865fb589abe91b1f00b929fb676082706bccd67331f203f6ab"
ANALYSIS_SHA256 = "bef33710ece37c4846f7b0d36592b932275502a49aba845da6d6e0c2515e8c37"
SNAPSHOT_SHA256 = "7df523634c654c74c44d50d4b96a7f0a0d0bc02087e2590d174753bb4df1d383"
SUPPLEMENTAL_KEYS = set(range(10)) | set(range(30, 35)) | {1896}
SOURCE_REPLACEMENT_RECIPES = {
    2641: ("643653_641", "9a4de0cf18ec27f3a28a7dfd09abaed58b8652bc174299834101af2493887479"),
    2717: ("643653_717", "5e0a6fe8a97c224febf80c0cba6debf40f4a92abdeebccac0dd2063167bdcb7b"),
}
FROZEN_SOURCE_NUMERICAL_SHA256 = {
    "pfc_native_drive.py": "8b32fb865c71c89ce6c0b75f99ab3acf1da628ed3ee976035c3e00a5865ab30e",
    "pfc_deterministic_replay.py": "aa1d2ccca0dab4621bf0b2e99f0ea75e2e037e69c8a31d52877ae92a5c794171",
    "pfc_kernel_parameters.py": "6cd894ec851896dd591497d89dbbc696d40fd8e6d366c701124210cf91558907",
    "pfc_input_pairing.py": "a7daa1953591bc6971a6a9b45cdfebe71362034f192a9bc883f05278f8bdc093",
}
TABLES = (intake.VARIANCE_NAME, intake.PAIR_NAME)
CONTROLS = {"config.json", "input_manifest.json", "collection_progress.json", "source_progress.json", "failure.json"}
require = intake.require


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path, base=None):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(base)) if base is not None else str(path),
                bytes=len(data), sha256=digest(data))


def write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bound_json(binding, base):
    data, _ = intake.bound_file(binding, base)
    return json.loads(data)


def failure_details(error):
    stderr = getattr(error, "stderr", None)
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return dict(error_type=type(error).__name__, message=str(error),
                exit_code=getattr(error, "returncode", None), stderr=stderr,
                read_attempts=getattr(error, "attempts", None))


class HDFSReadError(RuntimeError):
    def __init__(self, attempts):
        self.attempts = attempts
        self.returncode = attempts[-1]["exit_code"]
        self.stderr = attempts[-1]["stderr"]
        super().__init__("HDFS_CAT_EXHAUSTED:" + json.dumps(attempts))


def command(*arguments):
    result = subprocess.run(["hdfs", "dfs", *map(str, arguments)], capture_output=True, timeout=120)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    return result.stdout


def read_cat(uri, read_failures):
    """Retry only completed read commands with nonzero exit; parsing is outside."""
    require(isinstance(uri, str) and uri.startswith("hdfs://"), "HDFS_READ_URI_INVALID")
    attempts = []
    for attempt in range(1, 4):
        result = subprocess.run(["hdfs", "dfs", "-cat", uri], capture_output=True, timeout=120)
        if result.returncode == 0:
            return result.stdout
        record = dict(uri=uri, attempt=attempt, exit_code=result.returncode,
                      stderr=result.stderr.decode("utf-8", errors="replace"),
                      recorded_utc=datetime.now(timezone.utc).isoformat())
        attempts.append(record)
        read_failures.append(record)
        print(json.dumps(dict(status="RETRYING_HDFS_CAT" if attempt < 3 else "FAILED_HDFS_CAT", **record)), flush=True)
        if attempt < 3:
            time.sleep(5)
    raise HDFSReadError(attempts)


def read_json_cat(uri, read_failures):
    return json.loads(read_cat(uri, read_failures))


def recovery_targets(analysis, snapshot):
    require(isinstance(analysis, dict) and analysis.get("schema") == "pfc_complete_domain_history_covariance_v1"
            and analysis.get("required_groups") == 1000
            and analysis.get("missing_native_event_indices") == list(range(35)), "ORIGINAL_ANALYSIS_CONTRACT_CHANGED")
    require(isinstance(snapshot, dict) and snapshot.get("schema") == analysis["schema"]
            and snapshot.get("execution", {}).get("run_id") == "413de649f13ddf8f"
            and snapshot.get("execution", {}).get("trial_id") == "437784903"
            and snapshot.get("request_sha256") == ANALYSIS_SHA256
            and snapshot.get("required_groups") == 1000 and snapshot.get("completed_groups") == 766
            and snapshot.get("failures") == [] and snapshot.get("source_simulations") == 0
            and snapshot.get("production_episodes") == 0
            and all(snapshot.get(field) is False for field in
                    ("source_qualification_granted", "scaling_qualified", "uncertainty_coverage_qualified")),
            "ORIGINAL_COMPLETE766_SNAPSHOT_CHANGED")
    keys, rows = snapshot.get("completed_group_keys"), snapshot.get("completed")
    require(isinstance(keys, list) and len(keys) == 766 and len(set(keys)) == 766
            and all(type(key) is int and key in intake.REQUIRED_GROUPS for key in keys)
            and isinstance(rows, list) and len(rows) == 766, "ORIGINAL_GROUP_KEYS_CHANGED")
    records = {}
    for row in rows:
        require(isinstance(row, dict) and type(row.get("circuit_index")) is int
                and row["circuit_index"] in range(100) and type(row.get("condition_index")) is int
                and row["condition_index"] in range(10) and intake.valid_hash(row.get("report_sha256")),
                "ORIGINAL_GROUP_RECORD_CHANGED")
        key = 30 * row["circuit_index"] + row["condition_index"]
        require(key not in records, "DUPLICATE_ORIGINAL_GROUP")
        records[key] = row
    require(set(records) == set(keys) and not set(keys) & SUPPLEMENTAL_KEYS,
            "ORIGINAL_SUPPLEMENTAL_GROUP_OVERLAP")
    old_keys = set(keys) | SUPPLEMENTAL_KEYS
    targets = intake.REQUIRED_GROUPS - old_keys
    require(len(old_keys) == 782 and len(targets) == 218, "EXACT218_RECOVERY_TARGETS_REQUIRED")
    return targets, old_keys, records


def accepted_indices(progress):
    require(isinstance(progress, dict) and progress.get("execution", {}).get("run_id") == "81caa23939cca1e3"
            and progress.get("execution", {}).get("trial_id") == "436946603"
            and progress.get("production_episodes") == progress.get("source_simulations") == 0
            and progress.get("failures") == [], "ORIGINAL_ACCEPTANCE_CONTRACT_CHANGED")
    indices = progress.get("accepted_task_indices")
    require(isinstance(indices, list) and all(type(index) is int and 0 <= index < 3000 for index in indices)
            and len(indices) == len(set(indices)) and type(progress.get("accepted_episode_count")) is int
            and len(indices) == progress["accepted_episode_count"], "ORIGINAL_ACCEPTANCE_COUNT_CHANGED")
    return indices



def validated_source_override(index, accepted, normalized_binding, origin, identity):
    require(isinstance(accepted, dict) and accepted.get("status") == "ACCEPTED_MERLIN_DEVELOPMENT_EPISODE"
            and type(accepted.get("index")) is int and accepted["index"] == index
            and accepted.get("identity") == identity
            and accepted.get("run_id") == origin["source_run"]
            and accepted.get("trial_id") == origin["source_trial"]
            and all(type(accepted.get(field)) is int and accepted[field] == 0
                for field in ("source_simulations", "production_episodes"))
            and all(accepted.get(field) is False for field in
                ("source_qualification_granted", "scientific_parameter_domain_qualified", "scaling_qualified")),
            "SOURCE_ACCEPTANCE_IDENTITY_OR_SCOPE_CHANGED")
    receipt = accepted.get("remote_receipt")
    require(isinstance(receipt, dict) and receipt.get("index") == index
            and receipt.get("run_id") == origin["source_run"] and receipt.get("trial_id") == origin["source_trial"]
            and receipt.get("status") == "COMPLETE" and receipt.get("persisted") is True
            and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
            and type(receipt.get("xie_job")) in (str, int) and str(receipt["xie_job"])
            and accepted.get("xie_job") == receipt["xie_job"], "SOURCE_ACCEPTANCE_RECEIPT_CHANGED")
    if "xie_job" in origin:
        require(origin["xie_job"] == receipt["xie_job"], "SOURCE_ACCEPTANCE_JOB_CHANGED")
    uri = origin.get("bundle_uri", origin["checkpoint_path"].rstrip("/") + "/result.tar.gz")
    recovery = receipt.get("publication_recovery", {})
    require(isinstance(recovery, dict) and isinstance(receipt.get("checkpoint_path"), str),
            "SOURCE_ACCEPTANCE_BUNDLE_CHANGED")
    receipt_uri = recovery.get("source_bundle_uri", receipt["checkpoint_path"].rstrip("/") + "/result.tar.gz")
    require(isinstance(uri, str) and uri.startswith("hdfs://") and accepted.get("source_bundle_uri") == uri
            and receipt_uri == uri and intake.valid_hash(receipt.get("result_sha256"))
            and accepted.get("source_bundle_sha256") == receipt["result_sha256"]
            and type(receipt.get("result_bytes")) is int and receipt["result_bytes"] > 0
            and isinstance(accepted.get("receipt_uri"), str) and accepted["receipt_uri"].startswith("hdfs://"),
            "SOURCE_ACCEPTANCE_BUNDLE_CHANGED")
    validation = accepted.get("validation_execution")
    require(isinstance(validation, dict) and validation.get("backend") == "merlin"
            and validation.get("allocated_gpus") == 0 and validation.get("numpy_version") == "2.4.2"
            and isinstance(validation.get("run_id"), str) and validation["run_id"]
            and isinstance(validation.get("trial_id"), str) and validation["trial_id"],
            "SOURCE_ACCEPTANCE_VALIDATION_CHANGED")
    return dict(index=index, identity=identity, receipt=receipt, receipt_uri=accepted["receipt_uri"],
        bundle_uri=uri, bundle_sha256=receipt["result_sha256"], bundle_bytes=receipt["result_bytes"],
        source_run=origin["source_run"], source_trial=origin["source_trial"], xie_job=receipt["xie_job"],
        acceptance_binding=normalized_binding, validation_execution=validation)


def load_source_overrides(bindings, base, analysis):
    require(isinstance(bindings, list), "SOURCE_ACCEPTANCE_BINDINGS_INVALID")
    overrides = {}
    for binding in bindings:
        require(isinstance(binding, dict) and type(binding.get("index")) is int
                and 0 <= binding["index"] < 3000, "SOURCE_ACCEPTANCE_INDEX_INVALID")
        index = binding["index"]
        require(index not in overrides, "SOURCE_OVERRIDE_DUPLICATE")
        raw, normalized_binding = intake.bound_file(binding, base)
        origin, identity = analysis["source_origins"][str(index)], analysis["identities"][str(index)]
        overrides[index] = validated_source_override(index, json.loads(raw), normalized_binding, origin, identity)
    return overrides


def load_source_replacements(bindings, base, analysis):
    require(isinstance(bindings, list), "SOURCE_REPLACEMENT_BINDINGS_INVALID")
    if not bindings:
        return {}
    # The legacy covariance capsule does not stage the optional source-proof
    # runtime. Load it only when an explicit replacement requires its validator.
    from . import pfc_merlin_observation_completion as observations

    replacements = {}
    for binding in bindings:
        require(isinstance(binding, dict) and type(binding.get("index")) is int
                and binding["index"] in SOURCE_REPLACEMENT_RECIPES, "SOURCE_REPLACEMENT_INDEX_INVALID")
        index = binding["index"]
        require(index not in replacements, "SOURCE_REPLACEMENT_DUPLICATE")
        origin, identity = analysis["source_origins"][str(index)], analysis["identities"][str(index)]
        require(binding.get("original_origin") == origin
                and origin.get("source_run") == "6de397acc866aadd"
                and origin.get("source_trial") == "435906314", "SOURCE_REPLACEMENT_ORIGINAL_ORIGIN_CHANGED")
        raw, request_binding = intake.bound_file(binding.get("source_readout_request"), base)
        request = json.loads(raw)
        require(isinstance(request, dict) and request.get("schema") == observations.SOURCE_READOUT_SCHEMA
                and type(request.get("index")) is int and request["index"] == index
                and request.get("publication_kind") == "native_source", "SOURCE_REPLACEMENT_REQUEST_CHANGED")
        request_base = Path(request_binding["path"]).parent
        runtime = observations.load_runtime(request.get("runtime"), request_base)
        plan, _ = observations.qualified_readout_plan(request, runtime)
        job, recipe_sha256 = SOURCE_REPLACEMENT_RECIPES[index]
        require(plan.get("raw_dt_ms") == .0125 and plan.get("duration_ms") == 5000.
                and type(plan.get("coupled_core_units")) is int and plan["coupled_core_units"] == 100000
                and plan["rows"][index].get("identity") == identity
                and plan["rows"][index].get("config_sha256") == recipe_sha256,
                "SOURCE_REPLACEMENT_FROZEN_DATASET_CHANGED")
        source, intake_binding, publisher = observations.single_source_publication_inputs(request, request_base, plan, runtime)
        accepted = bound_json(source["acceptance_binding"], request_base)
        require(source["source_execution"]["run_id"] != origin["source_run"]
                and re.fullmatch(r"[0-9a-f]{16}", source["source_execution"]["run_id"]) is not None
                and re.fullmatch(r"[0-9]+", source["source_execution"]["trial_id"]) is not None
                and int(source["source_execution"]["trial_id"]) > 0
                and source["source_execution"]["xie_job"] == job
                and ("xie_job" not in origin or origin["xie_job"] == job)
                and accepted.get("config_sha256") == recipe_sha256, "SOURCE_REPLACEMENT_EXECUTION_CHANGED")
        numerical = accepted.get("numerical_source")
        require(isinstance(numerical, dict) and set(numerical) == set(FROZEN_SOURCE_NUMERICAL_SHA256)
                and all(isinstance(numerical[name], dict)
                        and numerical[name].get("returned_sha256") == expected
                        and numerical[name].get("frozen_sha256") == expected
                        for name,expected in FROZEN_SOURCE_NUMERICAL_SHA256.items()),
                "SOURCE_REPLACEMENT_NUMERICAL_SOURCE_CHANGED")
        effective = dict(source_run=source["source_execution"]["run_id"], source_trial=source["source_execution"]["trial_id"],
            xie_job=job, checkpoint_path=accepted["remote_receipt"]["checkpoint_path"],
            receipt_uri=source["source_receipt_uri"], bundle_uri=source["source_bundle_snapshot"]["bundle_uri"],
            transport="merlin_bundle", config_sha256=recipe_sha256)
        override = validated_source_override(index, accepted, source["acceptance_binding"], effective, identity)
        override.update(original_origin=dict(origin), effective_origin=effective, replacement_binding=request_binding,
                        scientific_intake_binding=intake_binding, publication_execution=publisher,
                        config_sha256=recipe_sha256, numerical_source_sha256=dict(FROZEN_SOURCE_NUMERICAL_SHA256))
        replacements[index] = override
    return replacements


def effective_source_origin(analysis, index, overrides):
    origin = analysis["source_origins"][str(index)]
    override = (overrides or {}).get(index)
    if override is not None and "replacement_binding" in override:
        require(override.get("original_origin") == origin and isinstance(override.get("effective_origin"), dict),
                "SOURCE_REPLACEMENT_ORIGINAL_ORIGIN_CHANGED")
        return override["effective_origin"]
    return origin


def same_binding(actual, expected):
    return isinstance(actual, dict) and all(actual.get(field) == expected[field] for field in ("bytes", "sha256"))


def ready_recovery_groups(accepted, override_indices, excluded, targets):
    available = sorted(set(accepted) | set(override_indices))
    ready = core.ready_groups(available, sorted(excluded), set(range(35)))
    require(set(ready) <= targets, "RECOVERY_ATTEMPTED_COMPLETED_OR_FOREIGN_GROUP")
    return ready


def prepare_group(analysis, key, metadata, read_failures, source_overrides=None):
    circuit, condition = divmod(key, 30)
    tasks = []
    for position, index in enumerate((key, key + 10, key + 20)):
        original_origin = analysis["source_origins"][str(index)]
        origin = effective_source_origin(analysis, index, source_overrides)
        identity = analysis["identities"][str(index)]
        require(type(identity.get("circuit_index")) is int and identity["circuit_index"] == circuit
                and type(identity.get("condition_index")) is int and identity["condition_index"] == condition
                and type(identity.get("block_index")) is int and identity["block_index"] == 6 + position,
                "ORIGINAL_TARGET_SOURCE_IDENTITY_CHANGED")
        task = dict(origin, index=index, identity=identity)
        override = (source_overrides or {}).get(index)
        if override is not None:
            require(override["identity"] == identity and override["source_run"] == origin["source_run"]
                    and override["source_trial"] == origin["source_trial"], "SOURCE_OVERRIDE_IDENTITY_CHANGED")
            task.update(receipt_uri=override["receipt_uri"], bundle_uri=override["bundle_uri"],
                        override_acceptance_binding=override["acceptance_binding"])
            if "replacement_binding" in override:
                task.update(original_source_origin=dict(original_origin),
                            source_replacement_binding=override["replacement_binding"])
        manifest = None
        for name in ("receipt", "manifest"):
            uri = task["receipt_uri"] if name == "receipt" else origin["checkpoint_path"] + "/manifest.json"
            raw = read_cat(uri, read_failures)
            parsed = json.loads(raw)
            path = metadata / f"task_{index:04d}_{name}.json"
            path.write_bytes(raw)
            task[name + "_sha256"] = digest(raw)
            if name == "receipt":
                require(isinstance(parsed, dict) and parsed.get("index") == index
                        and parsed.get("run_id") == origin["source_run"]
                        and parsed.get("trial_id") == origin["source_trial"]
                        and parsed.get("persisted") is True and parsed.get("exit_code") == 0
                        and parsed.get("status") == "COMPLETE", "ORIGINAL_TARGET_SOURCE_NOT_COMPLETE")
                if override is not None:
                    require(parsed == override["receipt"] and parsed["result_sha256"] == override["bundle_sha256"]
                            and parsed["result_bytes"] == override["bundle_bytes"], "SOURCE_OVERRIDE_RECEIPT_CONFLICT")
            else:
                manifest = parsed
        require(isinstance(manifest, dict), "ORIGINAL_TARGET_MANIFEST_INVALID")
        prefixes = [name[:-len("/raw.npz")] for name in manifest if name.endswith("/source_run/raw.npz")]
        require(len(prefixes) == 1, "ORIGINAL_TARGET_ONE_RAW_MEMBER_REQUIRED")
        task["source_member_prefix"] = prefixes[0]
        tasks.append(task)
    return dict(schema=intake.SOURCE_SCHEMA, circuit_index=circuit, condition_index=condition,
                task_indices=[key, key + 10, key + 20], tasks=tasks)


def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name, "CHECKPOINT_MEMBER_PATH_INVALID")
    path = PurePosixPath(name)
    require(not path.is_absolute() and str(path) == name and all(part not in ("", ".", "..") for part in path.parts),
            "CHECKPOINT_MEMBER_PATH_INVALID")
    return Path(*path.parts)


def entry_files(entry):
    require(isinstance(entry, dict) and isinstance(entry.get("tables"), dict)
            and set(entry["tables"]) == set(TABLES), "CHECKPOINT_ORIGINAL_TABLES_INVALID")
    files = {}
    for row in (entry.get("report"), *entry["tables"].values()):
        require(isinstance(row, dict) and type(row.get("bytes")) is int and row["bytes"] >= 0
                and intake.valid_hash(row.get("sha256")), "CHECKPOINT_FILE_BINDING_INVALID")
        name = str(safe_relative(row.get("path")))
        require(name not in files, "CHECKPOINT_DUPLICATE_FILE_PATH")
        files[name] = row
    return files


def restore_delta(archive_path, receipt, destination, previous, previous_sequence, old_keys, original_records, collector):
    require(isinstance(receipt, dict) and receipt.get("schema") == COLLECTOR_SCHEMA
            and receipt.get("execution", {}).get("backend") == "merlin"
            and receipt["execution"].get("run_id") == collector["run_id"]
            and receipt["execution"].get("trial_id") == collector["trial_id"]
            and receipt.get("persisted") is True and type(receipt.get("checkpoint_sequence")) is int
            and receipt["checkpoint_sequence"] == previous_sequence + 1
            and receipt.get("source_simulations") == receipt.get("production_episodes") == 0
            and receipt.get("source_qualification_granted") is False,
            "COLLECTOR_RECEIPT_CHAIN_CHANGED")
    data = archive_path.read_bytes()
    require(type(receipt.get("bundle_bytes")) is int and len(data) == receipt["bundle_bytes"]
            and digest(data) == receipt.get("bundle_sha256"), "COLLECTOR_BUNDLE_CHANGED")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive:
            safe_relative(member.name)
            require(member.isfile() and member.name not in members, "CHECKPOINT_REGULAR_UNIQUE_MEMBERS_REQUIRED")
            members[member.name] = member
        require({"config.json", "input_manifest.json", "collection_progress.json", "source_progress.json"} <= set(members),
                "CHECKPOINT_CONTROL_FILES_MISSING")
        def read(name):
            stream = archive.extractfile(members[name])
            require(stream is not None, "CHECKPOINT_MEMBER_UNREADABLE")
            return stream.read()
        manifest, progress = json.loads(read("input_manifest.json")), json.loads(read("collection_progress.json"))
        config = json.loads(read("config.json"))
        require(isinstance(config, dict) and config.get("schema") == COLLECTOR_SCHEMA
                and config.get("main_run_id") == "413de649f13ddf8f"
                and config.get("main_trial_id") == "437784903"
                and config.get("main_request_sha256") == ANALYSIS_SHA256,
                "COLLECTOR_ORIGINAL_CONFIG_CHANGED")
        require(manifest.get("schema") == intake.SCHEMA and isinstance(manifest.get("groups"), list),
                "COLLECTOR_MANIFEST_CHANGED")
        entries = {}
        for entry in manifest["groups"]:
            key = entry.get("group_key")
            require(type(key) is int and key in old_keys and key not in entries, "COLLECTOR_GROUP_IDENTITY_CHANGED")
            entry_files(entry)
            if key in original_records:
                require(entry["report"]["sha256"] == original_records[key]["report_sha256"],
                        "COLLECTOR_ORIGINAL_REPORT_HASH_CHANGED")
            if key in previous:
                require(entry == previous[key], "COLLECTOR_PREVIOUS_BINDING_CHANGED")
            entries[key] = entry
        added = receipt.get("added_group_keys")
        require(isinstance(added, list) and all(type(key) is int for key in added) and len(added) == len(set(added))
                and set(entries) == set(previous) | set(added) and not set(previous) & set(added)
                and receipt.get("returned_groups") == len(entries), "COLLECTOR_DELTA_DENOMINATOR_CHANGED")
        require(progress.get("execution") == receipt["execution"]
                and progress.get("checkpoint_sequence") == receipt["checkpoint_sequence"]
                and progress.get("returned_groups") == len(entries)
                and set(progress.get("returned_group_keys", [])) == set(entries)
                and progress.get("failures") == [] and progress.get("source_failures") == []
                and progress.get("source_simulations") == progress.get("production_episodes") == 0
                and all(progress.get(field) is False for field in
                        ("source_qualification_granted", "scientific_parameter_domain_qualified",
                         "scaling_qualified", "uncertainty_coverage_qualified", "source_ci_qualified")),
                "COLLECTOR_PROGRESS_SCOPE_CHANGED")
        files = {}
        for key in added:
            for name, row in entry_files(entries[key]).items():
                require(name not in files, "COLLECTOR_CROSS_GROUP_FILE_ALIAS")
                files[name] = row
        require(set(members) == (set(members) & CONTROLS) | set(files), "CHECKPOINT_UNDECLARED_MEMBER")
        # Validate every byte and existing destination before writing any input.
        verified = {}
        for name, row in files.items():
            raw = read(name)
            require(len(raw) == row["bytes"] and digest(raw) == row["sha256"], "CHECKPOINT_FILE_CHANGED")
            path = destination / safe_relative(name)
            require(path.resolve().is_relative_to(destination.resolve()) and not path.is_symlink(),
                    "CHECKPOINT_DESTINATION_ESCAPE")
            if path.exists():
                require(path.is_file() and path.read_bytes() == raw, "CHECKPOINT_EXISTING_INPUT_CHANGED")
            verified[name] = raw
        for name, raw in verified.items():
            path = destination / safe_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(raw)
    return entries



def restore_recovery_delta(archive_path, receipt, destination, previous, previous_sequence, targets, source,
                           analysis, source_overrides=None):
    """Restore only receipt-bound completed outputs from a recovery delta chain."""
    execution = receipt.get("execution", {})
    require(receipt.get("schema") == SCHEMA and execution.get("backend") == "merlin"
            and execution.get("run_id") == source["run_id"] and execution.get("trial_id") == source["trial_id"]
            and receipt.get("persisted") is True and type(receipt.get("checkpoint_sequence")) is int
            and receipt["checkpoint_sequence"] == previous_sequence + 1
            and all(type(receipt.get(field)) is int and receipt[field] == 0
                for field in ("source_simulations", "production_episodes"))
            and receipt.get("source_qualification_granted") is False, "RECOVERY_RECEIPT_CHAIN_CHANGED")
    prefix = source["checkpoint_uri"].rstrip("/")
    require(prefix.startswith("hdfs://") and prefix.endswith("/" + source["run_id"] + "/pfc_history_covariance_recovery")
            and receipt.get("bundle_uri") == prefix + f"/batch_{receipt['checkpoint_sequence']:04d}/cache.tar.gz",
            "RECOVERY_CHECKPOINT_URI_CHANGED")
    raw = archive_path.read_bytes()
    require(type(receipt.get("bundle_bytes")) is int and len(raw) == receipt["bundle_bytes"]
            and digest(raw) == receipt.get("bundle_sha256"), "RECOVERY_CHECKPOINT_BUNDLE_CHANGED")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive:
            safe_relative(member.name)
            require(member.isfile() and member.name not in members, "CHECKPOINT_REGULAR_UNIQUE_MEMBERS_REQUIRED")
            members[member.name] = member
        require({"progress.json", "recovery_inputs.json"} <= set(members), "RECOVERY_CONTROL_FILES_MISSING")
        def read(name):
            stream = archive.extractfile(members[name])
            require(stream is not None, "CHECKPOINT_MEMBER_UNREADABLE")
            return stream.read()
        progress = json.loads(read("progress.json"))
        manifest = json.loads(read("recovery_inputs.json"))
        require(isinstance(progress, dict) and progress.get("schema") == SCHEMA
                and progress.get("execution") == execution and progress.get("checkpoint_sequence") == receipt["checkpoint_sequence"]
                and progress.get("core_sha256") == CORE_SHA256 and progress.get("original_analysis_sha256") == ANALYSIS_SHA256
                and progress.get("original_snapshot_sha256") == SNAPSHOT_SHA256
                and progress.get("target_group_keys") == sorted(targets)
                and progress.get("required_recovery_groups") == 218 and progress.get("required_full_groups") == 1000
                and progress.get("original_completed_groups") == 766 and progress.get("completed_supplemental_groups") == 16
                and progress.get("failures") == []
                and all(type(progress.get(field)) is int and progress[field] == 0
                    for field in ("source_simulations", "production_episodes"))
                and all(progress.get(field) is False for field in
                    ("source_qualification_granted", "scientific_parameter_domain_qualified", "scaling_qualified",
                     "uncertainty_coverage_qualified", "source_ci_qualified")), "RECOVERY_PROGRESS_CONTRACT_CHANGED")
        require(isinstance(manifest, dict) and manifest.get("schema") == intake.SCHEMA
                and isinstance(manifest.get("groups"), list), "RECOVERY_MANIFEST_CHANGED")
        entries, all_paths = {}, set()
        for entry in manifest["groups"]:
            key = entry.get("group_key")
            require(type(key) is int and key in targets and key not in entries, "RECOVERY_GROUP_IDENTITY_CHANGED")
            paths = entry_files(entry)
            require(not all_paths & set(paths), "RECOVERY_GROUP_FILE_ALIAS")
            all_paths |= set(paths)
            if key in previous:
                require(entry == previous[key], "RECOVERY_PREVIOUS_BINDING_CHANGED")
            entries[key] = entry
        added = receipt.get("added_group_keys")
        require(isinstance(added, list) and all(type(key) is int and key in targets for key in added)
                and len(added) == len(set(added)) and not set(previous) & set(added)
                and set(entries) == set(previous) | set(added)
                and type(receipt.get("completed_recovery_groups")) is int
                and receipt["completed_recovery_groups"] == len(entries)
                and progress.get("completed_recovery_groups") == len(entries)
                and progress.get("completed_group_keys") == sorted(entries), "RECOVERY_DELTA_DENOMINATOR_CHANGED")
        completed = progress.get("completed")
        require(isinstance(completed, list) and len(completed) == len(entries), "RECOVERY_COMPLETED_RECORDS_CHANGED")
        recorded = set()
        for row in completed:
            key = row.get("group_key")
            require(type(key) is int and key in entries and key not in recorded
                    and row.get("circuit_index") == key // 30 and row.get("condition_index") == key % 30
                    and row.get("report_sha256") == entries[key]["report"]["sha256"]
                    and row.get("report_uri") == entries[key].get("report_uri"), "RECOVERY_COMPLETED_RECORDS_CHANGED")
            recorded.add(key)
        files, requests = {}, {}
        for key in added:
            files.update(entry_files(entries[key]))
            name = f"group_requests/group_{key:04d}.json"
            require(name in members, "RECOVERY_GROUP_REQUEST_MISSING")
            data = read(name)
            request = json.loads(data)
            require(request.get("schema") == intake.SOURCE_SCHEMA and request.get("circuit_index") == key // 30
                    and request.get("condition_index") == key % 30 and request.get("task_indices") == [key,key+10,key+20]
                    and isinstance(request.get("tasks"), list) and len(request["tasks"]) == 3,
                    "RECOVERY_GROUP_REQUEST_CHANGED")
            for task, index in zip(request["tasks"], [key,key+10,key+20]):
                original_origin = analysis["source_origins"][str(index)]
                override = (source_overrides or {}).get(index)
                replaced = "source_replacement_binding" in task
                origin = effective_source_origin(analysis, index, source_overrides) if replaced else original_origin
                require(task.get("index") == index and task.get("identity") == analysis["identities"][str(index)]
                        and task.get("checkpoint_path") == origin["checkpoint_path"]
                        and task.get("source_run") == origin["source_run"] and task.get("source_trial") == origin["source_trial"]
                        and intake.valid_hash(task.get("receipt_sha256")) and intake.valid_hash(task.get("manifest_sha256")),
                        "RECOVERY_GROUP_SOURCE_CHANGED")
                if replaced:
                    require(override is not None and "replacement_binding" in override
                            and task.get("original_source_origin") == original_origin
                            and task.get("config_sha256") == override["config_sha256"]
                            and task.get("transport") == override["effective_origin"]["transport"]
                            and same_binding(task.get("source_replacement_binding"), override["replacement_binding"]),
                            "RESUME_SOURCE_REPLACEMENT_PROOF_REQUIRED")
                if replaced or task.get("receipt_uri") != original_origin["receipt_uri"]:
                    recorded = task.get("override_acceptance_binding")
                    # The same accepted bytes can be staged at a new absolute path.
                    require(override is not None and same_binding(recorded, override["acceptance_binding"])
                            and task["receipt_uri"] == override["receipt_uri"] and task.get("bundle_uri") == override["bundle_uri"],
                            "RESUME_OVERRIDE_ACCEPTANCE_REQUIRED")
            requests[name] = data
        controls = {"progress.json", "recovery_inputs.json", "failure.json"}
        require(set(members) == (set(members) & controls) | set(files) | set(requests), "CHECKPOINT_UNDECLARED_MEMBER")
        verified = dict(requests)
        for name, row in files.items():
            data = read(name)
            require(len(data) == row["bytes"] and digest(data) == row["sha256"], "CHECKPOINT_FILE_CHANGED")
            verified[name] = data
        for name, data in verified.items():
            path = destination / safe_relative(name)
            require(path.resolve().is_relative_to(destination.resolve()) and not path.is_symlink(), "CHECKPOINT_DESTINATION_ESCAPE")
            if path.exists():
                require(path.is_file() and path.read_bytes() == data, "CHECKPOINT_EXISTING_INPUT_CHANGED")
        for name, data in verified.items():
            path = destination / safe_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
    return entries



def validate_resume_terminal(request, base, execution):
    """Read real failed ownership evidence; successful1000 results are reused."""
    if not request.get("resume_checkpoints"):
        return None
    source = request.get("resume_execution")
    require(isinstance(source, dict) and source.get("run_id") != execution["run_id"],
            "SEPARATE_RESUME_EXECUTION_REQUIRED")
    binding = request.get("resume_source_terminal")
    require(isinstance(binding, dict), "RESUME_SOURCE_TERMINAL_REQUIRED")
    data, normalized_binding = intake.bound_file(binding, base)
    terminal = json.loads(data)
    require(isinstance(terminal, dict), "RESUME_SOURCE_TERMINAL_IDENTITY_CHANGED")
    if "platform_status" in terminal:
        require(terminal.get("run_id") == source.get("run_id") and terminal.get("trial_id") == source.get("trial_id"),
                "RESUME_SOURCE_TERMINAL_IDENTITY_CHANGED")
        status = terminal.get("platform_status")
        if (status in ("succeeded", "success", "completed") and terminal.get("terminal_confirmed") is True
                and type(terminal.get("final_exit_code")) is int and terminal["final_exit_code"] == 0):
            raise ValueError("SOURCE_COMPLETE1000_REUSE_FINAL_INTAKE")
        stop_proof = {}
        if status == "stopped":
            # The optional stop-proof validator is staged for authorized takeovers;
            # historical failed operations retain their existing standalone path.
            from .pfc_merlin_observation_completion import validate_authorized_stop
            stop_proof = validate_authorized_stop(terminal, source["run_id"], source["trial_id"])
            if terminal.get("complete_groups") == 1000:
                raise ValueError("SOURCE_COMPLETE1000_REUSE_FINAL_INTAKE")
        else:
            require(status == "failed" and terminal.get("terminal_confirmed") is True
                    and type(terminal.get("final_exit_code")) is int and terminal["final_exit_code"] != 0,
                    "RESUME_SOURCE_STILL_RUNNING_OR_UNRESOLVED")
        owner = dict(run_id=terminal["run_id"], trial_id=terminal["trial_id"], platform_status=status,
                     terminal_confirmed=terminal["terminal_confirmed"], final_exit_code=terminal["final_exit_code"])
        return dict(binding=normalized_binding, status=status, original_execution=owner,
                    evidence_kind="authorized_operational_stop" if stop_proof else "operations_execution", **stop_proof)
    owner = terminal.get("execution", {})
    require(terminal.get("schema") == SCHEMA and owner.get("backend") == "merlin"
            and owner.get("run_id") == source.get("run_id") and owner.get("trial_id") == source.get("trial_id")
            and terminal.get("original_snapshot_sha256") == SNAPSHOT_SHA256
            and all(type(terminal.get(field)) is int and terminal[field] == 0
                for field in ("source_simulations", "production_episodes"))
            and terminal.get("source_qualification_granted") is False,
            "RESUME_SOURCE_TERMINAL_IDENTITY_CHANGED")
    status = terminal.get("status")
    complete = (status == "COMPLETE1000_RECOVERED_COVARIANCE_INTAKE" and terminal.get("persisted") is True
                and type(terminal.get("exit_code")) is int and terminal["exit_code"] == 0
                and terminal.get("complete_groups") == 1000 and terminal.get("failures") == [])
    if complete:
        raise ValueError("SOURCE_COMPLETE1000_REUSE_FINAL_INTAKE")
    failed = (status == "FAILED_COVARIANCE_RECOVERY" and isinstance(terminal.get("failures"), list)
              and bool(terminal["failures"]) and terminal.get("core_sha256") == CORE_SHA256
              and terminal.get("original_analysis_sha256") == ANALYSIS_SHA256)
    require(failed, "RESUME_SOURCE_STILL_RUNNING_OR_UNRESOLVED")
    return dict(binding=normalized_binding, status=status, original_execution=owner, evidence_kind="native_terminal")


def resume_recovery(request, base, work, targets, analysis, source_overrides):
    checkpoints = request.get("resume_checkpoints", [])
    require(isinstance(checkpoints, list), "RESUME_CHECKPOINTS_INVALID")
    if not checkpoints:
        require("resume_execution" not in request, "RESUME_CHECKPOINTS_REQUIRED")
        return {}, []
    source = request.get("resume_execution")
    require(isinstance(source, dict) and all(isinstance(source.get(key), str) and source[key]
            for key in ("run_id", "trial_id", "checkpoint_uri")), "RESUME_EXECUTION_BINDING_REQUIRED")
    entries, provenance, execution = {}, [], None
    for sequence, binding in enumerate(checkpoints, 1):
        require(isinstance(binding, dict) and set(binding) == {"receipt", "archive"}, "RESUME_CHECKPOINT_BINDINGS_INVALID")
        receipt_data, receipt_binding = intake.bound_file(binding["receipt"], base)
        receipt = json.loads(receipt_data)
        _, archive_binding = intake.bound_file(binding["archive"], base)
        archive = Path(archive_binding["path"])
        require(execution is None or receipt.get("execution") == execution, "RECOVERY_EXECUTION_CHAIN_CHANGED")
        execution = receipt.get("execution")
        entries = restore_recovery_delta(archive, receipt, work, entries, sequence-1, targets, source, analysis, source_overrides)
        provenance.append(dict(sequence=sequence, receipt=receipt_binding, archive=archive_binding,
            original_execution=execution, added_group_keys=receipt["added_group_keys"], completed_groups=len(entries),
            bundle_uri=receipt["bundle_uri"]))
    return entries, provenance

def merge_manifest(old_entries, new_entries, targets):
    require(set(new_entries) == set(targets) and len(new_entries) == 218
            and set(old_entries) == intake.REQUIRED_GROUPS - set(targets) and len(old_entries) == 782
            and not set(old_entries) & set(new_entries), "EXACT782_PLUS218_MERGE_REQUIRED")
    merged = dict(old_entries)
    merged.update(new_entries)
    require(set(merged) == intake.REQUIRED_GROUPS, "EXACT1000_MERGE_REQUIRED")
    return dict(schema=intake.SCHEMA, groups=[merged[key] for key in sorted(merged)])


def run(request_path, work):
    work = Path(work).resolve()
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not work.exists(), "FRESH_MERLIN_CPU_RECOVERY_REQUIRED")
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_bytes())
    require(request.get("schema") == SCHEMA and request.get("analysis_request", {}).get("sha256") == ANALYSIS_SHA256
            and request.get("completed_snapshot", {}).get("sha256") == SNAPSHOT_SHA256
            and core.sha256(Path(core.__file__)) == CORE_SHA256, "FROZEN_RECOVERY_INPUTS_REQUIRED")
    analysis = bound_json(request["analysis_request"], request_path.parent)
    snapshot = bound_json(request["completed_snapshot"], request_path.parent)
    targets, old_keys, original_records = recovery_targets(analysis, snapshot)
    source_overrides = load_source_overrides(request.get("source_acceptances", []), request_path.parent, analysis)
    source_replacements = load_source_replacements(request.get("source_replacements", []), request_path.parent, analysis)
    require(not set(source_overrides) & set(source_replacements), "SOURCE_REPLACEMENT_OVERRIDE_CONFLICT")
    source_overrides.update(source_replacements)
    collector = request.get("collector")
    require(isinstance(collector, dict) and isinstance(collector.get("checkpoint_uri"), str)
            and collector["checkpoint_uri"].startswith("hdfs://")
            and collector["checkpoint_uri"].endswith("/pfc_history_covariance_collection")
            and isinstance(collector.get("run_id"), str) and collector["run_id"]
            and isinstance(collector.get("trial_id"), str) and collector["trial_id"], "ORIGINAL_COLLECTOR_BINDING_REQUIRED")
    work.mkdir()
    (work / "source_metadata").mkdir()
    (work / "group_requests").mkdir()
    old_cache = work / "collector_cache"
    old_cache.mkdir()
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/pfc_history_covariance_recovery"
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                     hostname=socket.gethostname(), allocated_gpus=0)
    state = dict(schema=SCHEMA, status="RECOVERING_REMAINING_GROUPS", execution=execution,
        original_main_execution=dict(run_id="413de649f13ddf8f", trial_id="437784903", status="FAILED", exit_code=1),
        original_snapshot_sha256=SNAPSHOT_SHA256, original_analysis_sha256=ANALYSIS_SHA256,
        core_sha256=CORE_SHA256, target_group_keys=sorted(targets), required_recovery_groups=218,
        original_completed_groups=766, completed_supplemental_groups=16, required_full_groups=1000,
        failures=[], read_failures=[], source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scientific_parameter_domain_qualified=False,
        scaling_qualified=False, uncertainty_coverage_qualified=False, source_ci_qualified=False)
    completed, old_entries, pending = {}, {}, set()
    resumed_count = 0
    sequence, restored_sequence, last_checkpoint = 0, 0, time.monotonic()

    def metadata():
        state.update(recorded_utc=datetime.now(timezone.utc).isoformat(), completed_recovery_groups=len(completed),
            completed_group_keys=sorted(completed), completed=[dict(group_key=key,
                circuit_index=key // 30, condition_index=key % 30, report_uri=completed[key]["report_uri"],
                report_sha256=completed[key]["report"]["sha256"]) for key in sorted(completed)],
            collector_restored_groups=len(old_entries), collector_restored_sequence=restored_sequence,
            checkpoint_sequence=sequence, newly_computed_recovery_groups=len(completed)-resumed_count)
        write_json(work / "progress.json", state)
        write_json(work / "recovery_inputs.json", dict(schema=intake.SCHEMA, groups=[completed[key] for key in sorted(completed)]))

    def persist():
        nonlocal sequence, last_checkpoint
        sequence += 1
        metadata()
        archive_path = work / f"recovery_batch_{sequence:04d}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in ("progress.json", "recovery_inputs.json", "failure.json"):
                path = work / name
                if path.exists():
                    archive.add(path, arcname=name, recursive=False)
            for key in sorted(pending):
                for row in (completed[key]["report"], *completed[key]["tables"].values()):
                    archive.add(work / row["path"], arcname=row["path"], recursive=False)
                path = work / "group_requests" / f"group_{key:04d}.json"
                archive.add(path, arcname=str(path.relative_to(work)), recursive=False)
            for row in state.get("failed_group_outputs", {}).values():
                archive.add(work / row["path"], arcname=row["path"], recursive=False)
        destination = checkpoint + f"/batch_{sequence:04d}"
        command("-mkdir", "-p", destination)
        command("-put", archive_path, destination + "/cache.tar.gz")
        receipt = work / f"recovery_batch_{sequence:04d}_receipt.json"
        info = descriptor(archive_path)
        write_json(receipt, dict(schema=SCHEMA, status=state["status"], execution=execution, persisted=True,
            checkpoint_sequence=sequence, added_group_keys=sorted(pending), completed_recovery_groups=len(completed),
            bundle_uri=destination + "/cache.tar.gz", bundle_bytes=info["bytes"], bundle_sha256=info["sha256"],
            source_simulations=0, production_episodes=0, source_qualification_granted=False))
        command("-put", receipt, destination + "/receipt.json")
        command("-put", "-f", work / "progress.json", checkpoint + "/progress.json")
        pending.clear()
        last_checkpoint = time.monotonic()

    try:
        state["resume_terminal_provenance"] = validate_resume_terminal(request, request_path.parent, execution)
        completed, resume_provenance = resume_recovery(request, request_path.parent, work, targets, analysis, source_overrides)
        resumed_count = len(completed)
        # Seed the first new namespace checkpoint with the restored outputs. This
        # keeps the new delta chain independently recoverable from batch1.
        pending.update(completed)
        state.update(resumed_recovery_groups=resumed_count, resume_checkpoint_chain=resume_provenance,
            source_override_provenance=[{key:value for key,value in source_overrides[index].items() if key != "receipt"}
                for index in sorted(source_overrides)])
        while set(completed) != targets:
            state["active_operation"] = "read_original_acceptance"
            accepted = accepted_indices(read_json_cat(analysis["accepted_progress_uri"], state["read_failures"]))
            ready = ready_recovery_groups(accepted, source_overrides, old_keys | set(completed), targets)
            for key in ready:
                state.update(active_operation="prepare_original_group", active_group_key=key)
                group_request = prepare_group(analysis, key, work / "source_metadata", state["read_failures"], source_overrides)
                path = work / "group_requests" / f"group_{key:04d}.json"
                write_json(path, group_request)
                directory = work / f"group_{key:04d}"
                state["active_operation"] = "compute_remaining_group"
                core.run(path, directory, f"circuit_{key // 30:03d}/condition_{key % 30:02d}")
                completed[key] = dict(group_key=key, report=descriptor(directory / "report.json", work),
                    tables={name:descriptor(directory / name, work) for name in TABLES},
                    report_uri=os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
                        + f"/circuit_{key // 30:03d}/condition_{key % 30:02d}/pfc_history_coincidences/report.json",
                    transport_origin="remaining218_recovery")
                pending.add(key)
                if len(pending) >= 20 or time.monotonic() - last_checkpoint >= 1800:
                    state["active_operation"] = "publish_recovery_checkpoint"
                    persist()
            metadata()
            if set(completed) != targets:
                if pending or time.monotonic() - last_checkpoint >= 1800:
                    state["active_operation"] = "publish_recovery_checkpoint"
                    persist()
                time.sleep(300)
        state["status"] = "COMPLETE218_WAITING_COLLECTOR782_CACHE"
        state["active_operation"] = "publish_recovery_checkpoint"
        persist()
        while set(old_entries) != old_keys:
            state["active_operation"] = "list_collector_checkpoints"
            listing = command("-ls", collector["checkpoint_uri"]).decode("utf-8")
            sequences = set()
            for line in listing.splitlines():
                uri = line.split()[-1] if line.split() else ""
                match = re.fullmatch(re.escape(collector["checkpoint_uri"]) + r"/batch_([0-9]{4})", uri)
                if match:
                    sequences.add(int(match.group(1)))
            require(not sequences or min(sequences) == 1, "COLLECTOR_CHECKPOINT_CHAIN_START_CHANGED")
            for number in sorted(sequence for sequence in sequences if sequence > restored_sequence):
                require(number == restored_sequence + 1, "COLLECTOR_CHECKPOINT_CHAIN_GAP")
                uri = collector["checkpoint_uri"] + f"/batch_{number:04d}"
                exists = subprocess.run(["hdfs", "dfs", "-test", "-e", uri + "/receipt.json"], capture_output=True, timeout=120)
                if exists.returncode == 1 and not exists.stderr:
                    break
                if exists.returncode:
                    raise subprocess.CalledProcessError(exists.returncode, exists.args, exists.stdout, exists.stderr)
                state["active_operation"] = "read_collector_receipt"
                receipt = read_json_cat(uri + "/receipt.json", state["read_failures"])
                require(receipt.get("bundle_uri") == uri + "/cache.tar.gz", "COLLECTOR_RECEIPT_BUNDLE_URI_CHANGED")
                directory = old_cache / "batches" / f"batch_{number:04d}"
                directory.mkdir(parents=True, exist_ok=False)
                write_json(directory / "receipt.json", receipt)
                state["active_operation"] = "receive_collector_delta_cache"
                command("-get", receipt["bundle_uri"], directory / "cache.tar.gz")
                old_entries = restore_delta(directory / "cache.tar.gz", receipt, old_cache, old_entries,
                    restored_sequence, old_keys, original_records, collector)
                restored_sequence = number
                metadata()
                if set(old_entries) == old_keys:
                    break
            if set(old_entries) != old_keys:
                if time.monotonic() - last_checkpoint >= 1800:
                    state["active_operation"] = "publish_recovery_checkpoint"
                    persist()
                time.sleep(300)
        normalized = {}
        for key, entry in old_entries.items():
            result = dict(entry, report=dict(entry["report"]), tables={name:dict(row) for name,row in entry["tables"].items()})
            for row in (result["report"], *result["tables"].values()):
                row["path"] = str(Path("collector_cache") / safe_relative(row["path"]))
            normalized[key] = result
        manifest = work / "input_manifest.json"
        write_json(manifest, merge_manifest(normalized, completed, targets))
        state["active_operation"] = "validate_exact1000_intake"
        output = work / "intake_output"
        full_report = intake.run(manifest, output)
        state["status"] = "COMPLETE1000_RECOVERED_COVARIANCE_INTAKE"
        metadata()
        write_json(output / "recovery_report.json", state)
        state["active_operation"] = "publish_final_outputs"
        destination = checkpoint + "/result"
        command("-mkdir", "-p", destination)
        outputs = {}
        for name in (*TABLES, "report.json", "recovery_report.json"):
            command("-put", output / name, destination + "/" + name)
            row = descriptor(output / name)
            outputs[name] = dict(uri=destination + "/" + name, bytes=row["bytes"], sha256=row["sha256"])
        state["active_operation"] = "publish_recovery_checkpoint"
        persist()
        receipt = work / "receipt.json"
        write_json(receipt, dict(schema=SCHEMA, status=state["status"], execution=execution, persisted=True, exit_code=0,
            computed_recovery_groups=218-resumed_count, resumed_recovery_groups=resumed_count,
            required_recovery_groups=218, resume_checkpoint_chain=state["resume_checkpoint_chain"],
            source_override_provenance=state["source_override_provenance"],
            reused_collector_groups=782, complete_groups=1000,
            original_main_execution=state["original_main_execution"], original_snapshot_sha256=SNAPSHOT_SHA256,
            collector=collector, intake_status=full_report["status"], outputs=outputs,
            source_simulations=0, production_episodes=0, source_qualification_granted=False,
            scientific_parameter_domain_qualified=False, scaling_qualified=False, source_ci_qualified=False,
            uncertainty_coverage_qualified=False, failures=[], read_failures=state["read_failures"]))
        state["active_operation"] = "publish_final_receipt"
        command("-put", receipt, checkpoint + "/receipt.json")
        print(json.dumps(dict(status=state["status"], completed_groups=1000, receipt_uri=checkpoint + "/receipt.json")), flush=True)
        return state
    except Exception as error:
        failed_operation = state.get("active_operation")
        state.update(status="FAILED_COVARIANCE_RECOVERY", failures=[dict(operation=failed_operation, **failure_details(error))])
        key = state.get("active_group_key")
        if type(key) is int and key not in completed:
            directory = work / f"group_{key:04d}"
            files = {name: directory / name for name in ("report.json", *TABLES)}
            if all(path.is_file() for path in files.values()):
                # Keep completed numerical output when publication failed. It
                # remains a failed attempt until a separate transport recovery.
                state["failed_group_outputs"] = {name:descriptor(path, work) for name,path in files.items()}
                path = work / "group_requests" / f"group_{key:04d}.json"
                state["failed_group_outputs"]["request"] = descriptor(path, work)
        metadata()
        write_json(work / "failure.json", state)
        try:
            if failed_operation in ("publish_recovery_checkpoint", "publish_final_outputs", "publish_final_receipt"):
                command("-mkdir", "-p", checkpoint)
                command("-put", work / "failure.json", checkpoint + "/failure.json")
            else:
                persist()
        except Exception as publication_error:
            print("RECOVERY_FAILURE_DIAGNOSIS_NOT_PERSISTED:" + str(publication_error), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    run(args.request, args.work)


if __name__ == "__main__":
    main()

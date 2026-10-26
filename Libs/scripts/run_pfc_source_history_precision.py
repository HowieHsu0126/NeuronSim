"""Collect admitted fine-source phase counts and publish precision planning snapshots.

Source simulation and scientific qualification are never performed by this runner.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time

from population_dynamics import pfc_source_history_precision as precision

INTAKE_RUN = "81caa23939cca1e3"
INTAKE_TRIAL = "436946603"
TOP_STREAM_FIELDS = {"benchmark_episode_candidate", "benchmark_circuit_candidate", "cases",
                     "initial_rng_seed", "external_noise_seed"}
CASE_STREAM_FIELDS = {"case_id", "initial_rng_seed", "external_noise_seed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value, mutable=False):
    data = precision.canonical_json(value)
    if mutable:
        temporary = path.with_name("." + path.name + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def safe_error(error):
    return re.sub(r"(?:https?|hdfs)://[^\s\"']+", "[remote_path]", str(error))[-2000:]


def hdfs(*arguments, read_only=False):
    """Retry only bounded, recognizably transient read-only transport failures."""
    for attempt in range(3 if read_only else 1):
        result = subprocess.run(["hdfs", "dfs", *map(str, arguments)], capture_output=True)
        if result.returncode == 0:
            return result.stdout
        stderr = result.stderr.decode("utf-8", errors="replace")
        transient = any(marker in stderr for marker in (
            "StandbyException", "RetriableException", "ConnectException", "Connection reset",
            "connection reset", "timed out", "TimeoutException", "temporarily unavailable"))
        if read_only and transient and attempt < 2:
            if arguments[0] == "-get":
                target = Path(arguments[-1])
                require(not target.is_symlink(), "READ_RETRY_DESTINATION_SYMLINK")
                target.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
            continue
        operation = "READ" if read_only else "PUBLICATION"
        raise RuntimeError(f"HDFS_{operation}_FAILED:exit={result.returncode}:" +
                           re.sub(r"(?:https?|hdfs)://[^\s\"']+", "[remote_path]", stderr)[-1200:])
    raise RuntimeError("HDFS_READ_RETRY_EXHAUSTED")


def identity_key(row):
    identity = row.get("identity")
    require(isinstance(identity, dict), "INPUT_IDENTITY_REQUIRED")
    for field, allowed in (("circuit_index", precision.CIRCUITS),
                           ("block_index", precision.BLOCKS), ("condition_index", precision.CONDITIONS)):
        require(type(identity.get(field)) is int and identity[field] in allowed,
                "INPUT_IDENTITY_OUTSIDE_DESIGN:" + field)
    circuit, block, condition = (identity[field] for field in
                                 ("circuit_index", "block_index", "condition_index"))
    circuit_id = f"pfc_working_memory_circuit_{circuit:03d}"
    require(identity.get("episode_id") == precision.episode_id(circuit, block, condition)
            and identity.get("circuit_id") == circuit_id
            and identity.get("block_id") == circuit_id + f"_block_{block:03d}",
            "CANONICAL_INPUT_IDENTITY_CHANGED")
    require(isinstance(row.get("condition_id"), str) and bool(row["condition_id"])
            and precision.valid_hash(row.get("config_sha256"))
            and isinstance(row.get("source_signature"), dict) and bool(row["source_signature"]),
            "INPUT_SOURCE_BINDING_INVALID")
    require(identity.get("condition_id", row["condition_id"]) == row["condition_id"],
            "INPUT_CONDITION_METADATA_CHANGED")
    for field in ("initial_state_seed", "background_seed"):
        require(type(identity.get(field)) is int and identity[field] >= 0,
                "INPUT_STREAM_IDENTITY_INVALID:" + field)
    precision.canonical_json(row["source_signature"])
    return circuit, block, condition


def manifest_header():
    return dict(schema=precision.SCHEMA, design_circuits=list(precision.CIRCUITS),
                block_indices=list(precision.BLOCKS), condition_indices=list(precision.CONDITIONS),
                dt_ms=.0125, duration_ticks=400000, core_units=100000,
                population_sizes={"E": 80000, "I": 20000})


def load_inputs(pilot_path, domain_path):
    pilot = read_json(pilot_path)
    header = manifest_header()
    for field, expected in header.items():
        value = pilot.get(field)
        if field in ("design_circuits", "block_indices", "condition_indices"):
            require(isinstance(value, list) and all(type(item) is int for item in value),
                    "PILOT_SCOPE_TYPE_INVALID:" + field)
        elif field in ("duration_ticks", "core_units"):
            require(type(value) is int, "PILOT_SCOPE_TYPE_INVALID:" + field)
        elif field == "dt_ms":
            require(type(value) in (int, float), "PILOT_SCOPE_TYPE_INVALID:" + field)
        elif field == "population_sizes":
            require(isinstance(value, dict) and all(type(item) is int for item in value.values()),
                    "PILOT_SCOPE_TYPE_INVALID:" + field)
        if field == "block_indices":
            require(value in (list(precision.BLOCKS), [3, 4, 5]), "PILOT_SCOPE_CHANGED:" + field)
        else:
            require(value == expected, "PILOT_SCOPE_CHANGED:" + field)
    old = pilot.get("episodes")
    require(isinstance(old, list), "PILOT_EPISODES_REQUIRED")
    old_keys, old_rows = set(), []
    for row in old:
        key = identity_key(row)
        require(key[1] in (3, 4, 5) and key not in old_keys, "PILOT_EPISODE_DUPLICATE_OR_OUTSIDE_SCOPE")
        old_keys.add(key)
        value = copy.deepcopy(row)
        binding = value.get("phase_csv")
        require(isinstance(binding, dict) and isinstance(binding.get("path"), str), "PILOT_PHASE_BINDING_REQUIRED")
        phase_path = Path(binding["path"])
        if not phase_path.is_absolute():
            phase_path = Path(pilot_path).parent / phase_path
        binding["path"] = str(phase_path.resolve())
        old_rows.append(value)
    expected_old = {(c, b, k) for c in precision.CIRCUITS for b in (3, 4, 5) for k in precision.CONDITIONS}
    require(old_keys == expected_old, "FULL120_ADMITTED_PILOT_REQUIRED")
    raw = read_json(domain_path)
    require(isinstance(raw, dict), "DOMAIN_INPUT_DICTIONARY_REQUIRED")
    domain = {}
    for text, row in raw.items():
        require(isinstance(text, str) and re.fullmatch(r"[0-9]+", text) is not None and str(int(text)) == text,
                "DOMAIN_TASK_KEY_INVALID")
        index = int(text)
        key = identity_key(row)
        require(key[1] in (6, 7, 8) and index == key[0] * 30 + (key[1] - 6) * 10 + key[2]
                and index not in domain, "DOMAIN_TASK_IDENTITY_CHANGED")
        domain[index] = copy.deepcopy(row)
    expected = {c * 30 + (b - 6) * 10 + k for c in precision.CIRCUITS for b in (6, 7, 8)
                for k in precision.CONDITIONS}
    require(set(domain) == expected, "FULL120_DOMAIN_INPUTS_REQUIRED")
    signatures, histories, owners, condition_names = {}, {}, {}, {}
    for row in old_rows + list(domain.values()):
        circuit, block, condition = identity_key(row)
        name = row["condition_id"]
        require(condition not in condition_names or condition_names[condition] == name,
                "SOURCE_CONDITION_ID_CHANGED")
        require(name not in condition_names.values() or condition_names.get(condition) == name,
                "SOURCE_CONDITION_ID_NOT_UNIQUE")
        condition_names[condition] = name
        signature = precision.canonical_json(row["source_signature"])
        require((circuit, condition) not in signatures or signatures[circuit, condition] == signature,
                "PILOT_DOMAIN_SOURCE_SIGNATURE_MISMATCH")
        signatures[circuit, condition] = signature
        history = tuple(row["identity"][field] for field in ("initial_state_seed", "background_seed"))
        require((circuit, block) not in histories or histories[circuit, block] == history,
                "PAIRED_HISTORY_STREAMS_CHANGED")
        histories[circuit, block] = history
        for field, value in zip(("initial_state_seed", "background_seed"), history):
            require((field, value) not in owners or owners[field, value] == (circuit, block),
                    "SOURCE_STREAM_REUSED_ACROSS_HISTORIES")
            owners[field, value] = circuit, block
    return old_rows, domain


def accepted_progress(uri, required):
    value = json.loads(hdfs("-cat", uri, read_only=True))
    execution = value.get("execution")
    require(isinstance(execution, dict) and execution.get("run_id") == INTAKE_RUN
            and execution.get("trial_id") == INTAKE_TRIAL, "CANONICAL_INTAKE_IDENTITY_CHANGED")
    require(type(value.get("production_episodes")) is int and value["production_episodes"] == 0
            and type(value.get("source_simulations")) is int and value["source_simulations"] == 0
            and value.get("failures") == [], "CANONICAL_INTAKE_SCOPE_OR_FAILURE_CHANGED")
    require(all(value.get(field) is False for field in (
        "source_qualification_granted", "scaling_qualified", "scientific_parameter_domain_qualified")),
        "CANONICAL_INTAKE_QUALIFICATION_CHANGED")
    accepted = value.get("accepted_task_indices")
    require(isinstance(accepted, list) and all(type(index) is int and 0 <= index < 3000 for index in accepted)
            and len(set(accepted)) == len(accepted)
            and type(value.get("accepted_episode_count")) is int
            and value["accepted_episode_count"] == len(accepted), "CANONICAL_ACCEPTED_INDICES_INVALID")
    status = value.get("status")
    require(isinstance(status, str) and bool(status) and not status.startswith("FAILED"),
            "CANONICAL_INTAKE_STATUS_INVALID")
    available = set(accepted)
    if status in ("COMPLETE", "PASS"):
        require(required <= available, "TERMINAL_INTAKE_MISSING_EXPECTED_DOMAIN_TASKS")
    return available, status, len(accepted)


def source_signature(config, identity):
    cases = config.get("cases")
    require(isinstance(cases, list), "SOURCE_CASES_REQUIRED")
    selected = [case for case in cases if isinstance(case, dict) and case.get("case_id") == identity["episode_id"]]
    require(len(selected) == 1, "UNIQUE_SOURCE_EPISODE_CASE_REQUIRED")
    case = selected[0]
    for settings in (config, case):
        require(settings.get("initial_rng_seed") == identity["initial_state_seed"]
                and type(settings.get("initial_rng_seed")) is int
                and settings.get("external_noise_seed") == identity["background_seed"]
                and type(settings.get("external_noise_seed")) is int, "SOURCE_STREAM_SETTINGS_CHANGED")
    return {"source_config": {key: value for key, value in config.items() if key not in TOP_STREAM_FIELDS},
            "case": {key: value for key, value in case.items() if key not in CASE_STREAM_FIELDS}}


def bound_members(bundle, manifest):
    require(isinstance(manifest, dict) and bool(manifest), "SOURCE_MANIFEST_REQUIRED")
    selected = {}
    for suffix, role in (("/source_run/report.json", "report"),
                         ("/development5ms/phase_responses.csv", "phase")):
        names = [name for name in manifest if isinstance(name, str) and name.endswith(suffix)]
        require(len(names) == 1, "UNIQUE_SOURCE_MEMBER_REQUIRED:" + role)
        selected[names[0]] = role
    contents, found = {}, set()
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive:
            if member.name not in selected:
                continue
            name = member.name
            path = PurePosixPath(name)
            row = manifest[name]
            require(name not in found and member.isfile() and not path.is_absolute()
                    and ".." not in path.parts and str(path) == name and isinstance(row, dict)
                    and type(row.get("bytes")) is int and row["bytes"] >= 0
                    and member.size == row["bytes"] and precision.valid_hash(row.get("sha256")),
                    "SOURCE_BOUND_MEMBER_INVALID:" + selected[name])
            stream = archive.extractfile(member)
            require(stream is not None, "SOURCE_BOUND_MEMBER_UNREADABLE")
            with stream:
                data = stream.read()
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                    "SOURCE_BOUND_MEMBER_CHANGED:" + selected[name])
            found.add(name)
            contents[selected[name]] = data
    require(found == set(selected), "SOURCE_BOUND_MEMBERS_MISSING")
    return contents, {role: manifest[name] for name, role in selected.items()}


def receive(index, row, args, scratch, domain_directory):
    checkpoint = args.source_root.rstrip("/") + f"/task_{index:04d}"
    print(json.dumps(dict(status="RECEIVING_ADMITTED_SOURCE", task_index=index, recorded_utc=now())), flush=True)
    with tempfile.TemporaryDirectory(prefix=f"task_{index:04d}_", dir=scratch) as temporary:
        directory = Path(temporary)
        hdfs("-get", checkpoint + "/receipt.json", directory / "receipt.json", read_only=True)
        receipt = read_json(directory / "receipt.json")
        expected = dict(index=index, run_id=args.source_run, trial_id=args.source_trial,
                        execution_backend="merlin", status="COMPLETE", exit_code=0, checkpoint_path=checkpoint)
        require(all(type(receipt.get(key)) is type(value) and receipt[key] == value for key, value in expected.items()),
                "ORIGINAL_SOURCE_RECEIPT_IDENTITY_CHANGED")
        if "persisted" in receipt:
            require(receipt["persisted"] is True, "ORIGINAL_SOURCE_NOT_PERSISTED")
        require(type(receipt.get("result_bytes")) is int and receipt["result_bytes"] >= 0
                and precision.valid_hash(receipt.get("result_sha256")), "SOURCE_BUNDLE_BINDING_REQUIRED")
        for name in ("manifest.json", "result.tar.gz"):
            hdfs("-get", checkpoint + "/" + name, directory / name, read_only=True)
        bundle = directory / "result.tar.gz"
        require(bundle.stat().st_size == receipt["result_bytes"]
                and precision.file_sha256(bundle) == receipt["result_sha256"], "ORIGINAL_SOURCE_BUNDLE_CHANGED")
        manifest = read_json(directory / "manifest.json")
        contents, bindings = bound_members(bundle, manifest)
        report = json.loads(contents["report"])
        config = report.get("resolved_config")
        require(isinstance(config, dict) and report.get("source_execution_complete") is True
                and report.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE"
                and report.get("production_episodes") == 0, "ORIGINAL_SOURCE_REPORT_NOT_COMPLETE")
        candidate = config.get("benchmark_episode_candidate")
        require(isinstance(candidate, dict) and candidate.get("identity") == row["identity"],
                "RESOLVED_SOURCE_EPISODE_IDENTITY_CHANGED")
        require(config.get("excitatory_neurons") == 80000 and config.get("inhibitory_neurons") == 20000
                and config.get("dt_ms") == .0125 and config.get("duration_ms") == 5000.,
                "RESOLVED_FINE_SOURCE_SCALE_CHANGED")
        require(report.get("config_sha256") == row["config_sha256"], "SOURCE_CONFIG_HASH_CHANGED")
        require(precision.canonical_json(source_signature(config, row["identity"])) ==
                precision.canonical_json(row["source_signature"]), "RESOLVED_SOURCE_SETTINGS_CHANGED")
        target = domain_directory / f"task_{index:04d}"
        require(not target.exists(), "RETURNED_TASK_ALREADY_MATERIALIZED")
        target.mkdir()
        phase = target / "phase_responses.csv"
        with phase.open("xb") as stream:
            stream.write(contents["phase"])
        value = copy.deepcopy(row)
        value["phase_csv"] = dict(path=str(phase), bytes=bindings["phase"]["bytes"],
                                  sha256=bindings["phase"]["sha256"])
        precision.load_phases(value["phase_csv"], value["identity"], value["condition_id"], target)
        source_binding = dict(index=index, identity=row["identity"], config_sha256=row["config_sha256"],
            source_execution_backend="merlin", source_run_id=receipt["run_id"], source_trial_id=receipt["trial_id"],
            source_receipt_sha256=precision.file_sha256(directory / "receipt.json"),
            source_manifest_sha256=precision.file_sha256(directory / "manifest.json"),
            source_bundle_sha256=receipt["result_sha256"], source_bundle_bytes=receipt["result_bytes"],
            source_report_sha256=bindings["report"]["sha256"], phase_csv_sha256=bindings["phase"]["sha256"],
            source_signature_sha256=hashlib.sha256(precision.canonical_json(row["source_signature"])).hexdigest())
        for field in ("hostname", "xie_job", "compatibility_execution_id"):
            if field in receipt:
                source_binding[field] = receipt[field]
        write_json(target / "source_binding.json", source_binding)
    return value, source_binding


def snapshot(args, checkpoint, pilot, collected, source_bindings, execution):
    count = len(pilot) + len(collected)
    directory = args.work / f"snapshot_{count:03d}"
    directory.mkdir()
    manifest = manifest_header()
    manifest["episodes"] = pilot + [collected[index] for index in sorted(collected)]
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    output = directory / "analysis"
    report = precision.run(manifest_path, output)
    require(report["actual_episodes"] == count and report["required_episodes"] == 240,
            "PRECISION_SNAPSHOT_DENOMINATOR_CHANGED")
    files = {name: output / name for name in
             ("report.json", "paired_history_phase_rates.csv", "history_precision_summary.csv")}
    files["manifest.json"] = manifest_path
    remote = checkpoint + f"/snapshot_{count:03d}"
    hdfs("-mkdir", "-p", remote)
    for name, path in files.items():
        hdfs("-put", path, remote + "/" + name)
    receipt = dict(schema="pfc_source_history_precision_snapshot_v1", status=report["status"],
        execution=execution, source_run_id=args.source_run, source_trial_id=args.source_trial,
        source_intake_run_id=INTAKE_RUN, source_intake_trial_id=INTAKE_TRIAL,
        actual_episodes=count, required_episodes=240, pilot_episodes=120, domain_episodes=len(collected),
        pilot_manifest_sha256=precision.file_sha256(args.pilot_manifest),
        domain_inputs_sha256=precision.file_sha256(args.domain_inputs),
        precision_implementation_sha256=precision.file_sha256(Path(precision.__file__)),
        runner_implementation_sha256=precision.file_sha256(Path(__file__)),
        source_bindings=[source_bindings[index] for index in sorted(source_bindings)],
        outputs={name: dict(bytes=path.stat().st_size, sha256=precision.file_sha256(path))
                 for name, path in files.items()},
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scaling_qualified=False, scientific_parameter_domain_qualified=False,
        uncertainty_coverage_qualified=False, recorded_utc=now())
    receipt_path = directory / "snapshot_receipt.json"
    write_json(receipt_path, receipt)
    hdfs("-put", receipt_path, remote + "/snapshot_receipt.json")
    return count, receipt


def run(args):
    run_id, trial_id, checkpoint_root = (os.environ.get(field) for field in
                                       ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS"))
    require(bool(run_id) and bool(trial_id) and isinstance(checkpoint_root, str)
            and checkpoint_root.startswith("hdfs://") and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0"
            and not os.environ.get("CUDA_VISIBLE_DEVICES"), "MERLIN_CPU_EXECUTION_REQUIRED")
    require(type(args.poll_seconds) is int and args.poll_seconds > 0, "POLL_SECONDS_INVALID")
    relative = PurePosixPath(args.checkpoint_subdirectory)
    require(bool(args.checkpoint_subdirectory) and not relative.is_absolute() and ".." not in relative.parts
            and str(relative) == args.checkpoint_subdirectory
            and all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in relative.parts),
            "CHECKPOINT_SUBDIRECTORY_INVALID")
    require(args.source_root.startswith("hdfs://") and args.source_root.rstrip("/").endswith(
            "/" + args.source_run + "/pfc") and bool(args.source_run) and bool(args.source_trial),
            "ORIGINAL_PRODUCER_ROOT_IDENTITY_INVALID")
    require(args.accepted_progress_hdfs.startswith("hdfs://"), "INTAKE_PROGRESS_URI_REQUIRED")
    args.work = args.work.resolve()
    require(not args.work.exists(), "FRESH_RUNNER_WORK_REQUIRED")
    args.work.mkdir(parents=True)
    scratch, domain_directory = args.work / "scratch", args.work / "domain"
    scratch.mkdir()
    domain_directory.mkdir()
    checkpoint = checkpoint_root.rstrip("/") + "/" + args.checkpoint_subdirectory
    require(not args.source_root.rstrip("/").startswith(checkpoint + "/")
            and not checkpoint.startswith(args.source_root.rstrip("/") + "/"), "SOURCE_PUBLICATION_ROOTS_OVERLAP")
    execution = dict(backend="merlin", run_id=run_id, trial_id=trial_id, hostname=socket.gethostname())
    collected, source_bindings, last_published = {}, {}, None
    current_task = None
    try:
        pilot, domain = load_inputs(args.pilot_manifest.resolve(), args.domain_inputs.resolve())
        hdfs("-mkdir", "-p", checkpoint)
        while True:
            accepted, intake_status, accepted_count = accepted_progress(args.accepted_progress_hdfs, set(domain))
            for index in sorted(set(domain).intersection(accepted) - set(collected)):
                current_task = index
                value, binding = receive(index, domain[index], args, scratch, domain_directory)
                collected[index], source_bindings[index] = value, binding
            current_task = None
            count = 120 + len(collected)
            if count != last_published:
                last_published, receipt = snapshot(args, checkpoint, pilot, collected, source_bindings, execution)
            else:
                receipt = None
            unfinished = sorted(set(domain) - set(collected))
            progress = dict(status="COMPLETE" if not unfinished else "WAITING_FOR_ADMITTED_SOURCE",
                execution=execution, actual_episodes=count, required_episodes=240,
                domain_episodes=len(collected), last_published_snapshot_episodes=last_published,
                accepted_source_episodes=accepted_count, intake_status=intake_status,
                unfinished_task_indices=unfinished,
                unfinished_episode_ids=[domain[index]["identity"]["episode_id"] for index in unfinished],
                source_simulations=0, production_episodes=0, source_qualification_granted=False,
                scaling_qualified=False, scientific_parameter_domain_qualified=False,
                uncertainty_coverage_qualified=False, recorded_utc=now())
            write_json(args.work / "progress.json", progress, mutable=True)
            hdfs("-put", "-f", args.work / "progress.json", checkpoint + "/progress.json")
            print(json.dumps({field: progress[field] for field in
                             ("status", "actual_episodes", "required_episodes", "accepted_source_episodes", "recorded_utc")} ), flush=True)
            if not unfinished:
                require(last_published == 240 and receipt is not None and receipt["status"] == "COMPLETE",
                        "COMPLETE_PRECISION_SNAPSHOT_REQUIRED")
                shutil.rmtree(scratch)
                return receipt
            time.sleep(args.poll_seconds)
    except Exception as error:
        diagnostics = dict(status="FAILED", execution=execution, task_index=current_task,
            actual_episodes=120 + len(collected), required_episodes=240,
            last_published_snapshot_episodes=last_published, error_type=type(error).__name__, error=safe_error(error),
            source_simulations=0, production_episodes=0, source_qualification_granted=False,
            scaling_qualified=False, scientific_parameter_domain_qualified=False,
            uncertainty_coverage_qualified=False, recorded_utc=now())
        write_json(args.work / "failure.json", diagnostics)
        print(json.dumps(diagnostics), flush=True)
        try:
            hdfs("-put", args.work / "failure.json", checkpoint + "/failure.json")
        except Exception as publication_error:
            print(json.dumps(dict(status="FAILURE_DIAGNOSTIC_PUBLICATION_FAILED",
                                  error_type=type(publication_error).__name__, error=safe_error(publication_error))), flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--domain-inputs", type=Path, required=True)
    parser.add_argument("--accepted-progress-hdfs", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-trial", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint-subdirectory", required=True)
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

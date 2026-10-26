#!/usr/bin/env python3
"""Execute the frozen CA3 1210 ms prefix replay inside an actual Merlin trial."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time

GIB = 1024 ** 3


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            value.update(data)
    return value.hexdigest()


def hdfs_put(path: Path, target: str) -> None:
    subprocess.run(["hdfs", "dfs", "-put", str(path), target], check=True)


def memory_evidence(resource_shape: dict, required_available_gib: int) -> dict:
    """Admit separate Merlin allocation and host-memory observations.

    Some verified Merlin images expose no cgroup mount. Their role allocation
    and host MemAvailable are separate admission checks; neither is a measured
    amount of available container memory. Visible memory controllers add a
    limit-minus-usage check and must never be silently skipped.
    """
    role = resource_shape.get("name")
    requested_mib = resource_shape.get("memory_mb")
    if (not isinstance(role, str) or not role or not role.replace("_", "").isalnum()
            or type(requested_mib) is not int or requested_mib <= 0
            or type(required_available_gib) is not int or required_available_gib <= 0):
        raise ValueError("MERLIN_MEMORY_RESOURCE_SHAPE_INVALID")
    allocation_key = "ARNOLD_" + role.upper() + "_MEM"
    allocated_text = os.environ.get(allocation_key)
    if allocated_text is None:
        raise ValueError("MERLIN_PLATFORM_MEMORY_ALLOCATION_REQUIRED:" + allocation_key)
    if not allocated_text.isdecimal() or int(allocated_text) <= 0:
        raise ValueError("MERLIN_PLATFORM_MEMORY_ALLOCATION_INVALID:" + allocation_key)
    allocated_mib = int(allocated_text)
    if allocated_mib < requested_mib:
        raise ValueError("MERLIN_PLATFORM_MEMORY_ALLOCATION_BELOW_REQUEST")

    required_bytes = required_available_gib * GIB
    host_memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, separator, value = line.partition(":")
        if key not in {"MemTotal", "MemAvailable"}:
            continue
        fields = value.split()
        if not separator or len(fields) != 2 or fields[1] != "kB" or not fields[0].isdecimal():
            raise ValueError("MERLIN_HOST_MEMORY_EVIDENCE_INVALID:" + key)
        host_memory[key] = int(fields[0]) * 1024
    if set(host_memory) != {"MemTotal", "MemAvailable"} or host_memory["MemTotal"] <= 0:
        raise ValueError("MERLIN_HOST_MEMORY_EVIDENCE_INCOMPLETE")
    if host_memory["MemAvailable"] < required_bytes:
        raise ValueError("MERLIN_HOST_AVAILABLE_MEMORY_FLOOR_NOT_MET")

    mount_lines = Path("/proc/self/mountinfo").read_text().splitlines()
    if not mount_lines:
        raise ValueError("MERLIN_CGROUP_MOUNT_EVIDENCE_REQUIRED")
    cgroup_mounts = []
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        fields = after.split()
        if not separator or len(before.split()) < 6 or len(fields) < 3:
            raise ValueError("MERLIN_CGROUP_MOUNT_EVIDENCE_INVALID")
        if fields[0] in {"cgroup", "cgroup2"}:
            cgroup_mounts.append(line)

    controllers = []
    observed = {}
    for version, limit_name, usage_name in (
            ("v2", "memory.max", "memory.current"),
            ("v1", "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        for name in (limit_name, usage_name):
            path = Path("/sys/fs/cgroup") / name
            if path.is_file():
                observed[name] = path.read_text().strip()
        limit_text, usage_text = observed.get(limit_name), observed.get(usage_name)
        if limit_text is None and usage_text is None:
            continue
        if limit_text is None or usage_text is None:
            raise ValueError("MERLIN_CGROUP_MEMORY_EVIDENCE_INCOMPLETE:" + version)
        if not usage_text.isdecimal():
            raise ValueError("MERLIN_CGROUP_MEMORY_USAGE_INVALID:" + version)
        usage_bytes = int(usage_text)
        if version == "v2" and limit_text == "max":
            limit_bytes = None
            headroom_bytes = None
        else:
            if not limit_text.isdecimal():
                raise ValueError("MERLIN_CGROUP_MEMORY_LIMIT_INVALID:" + version)
            limit_bytes = int(limit_text)
            headroom_bytes = max(0, limit_bytes - usage_bytes)
            if headroom_bytes < required_bytes:
                raise ValueError("MERLIN_CGROUP_MEMORY_HEADROOM_FLOOR_NOT_MET:" + version)
        controllers.append(dict(
            version=version, scope="visible_cgroup", limit_path="/sys/fs/cgroup/" + limit_name,
            usage_path="/sys/fs/cgroup/" + usage_name, limit_bytes=limit_bytes,
            usage_bytes=usage_bytes, limit_minus_usage_bytes=headroom_bytes))
    if not controllers and cgroup_mounts:
        raise ValueError("MERLIN_CGROUP_MEMORY_LAYOUT_UNSUPPORTED")

    checks = ["platform_allocation_meets_request", "host_available_memory_meets_floor"]
    if controllers:
        checks.append("visible_cgroup_memory_headroom_meets_floor")
    return dict(
        admission=dict(status="PASS", scope="merlin_role_allocation_and_host_available_memory",
                       checks=checks, required_available_memory_gib=required_available_gib,
                       container_available_memory_verified=False),
        platform_allocation=dict(scope="merlin_role_allocation", role=role,
                                 environment_variable=allocation_key, allocated_memory_mib=allocated_mib,
                                 requested_memory_mib=requested_mib),
        host_memory=dict(scope="host", source="/proc/meminfo", memory_bytes=host_memory,
                         required_available_bytes=required_bytes),
        cgroup=dict(status="memory_controller_observed" if controllers else "cgroup_not_exposed",
                    mountinfo_source="/proc/self/mountinfo", mounts=cgroup_mounts,
                    observed_values=observed, controllers=controllers))


def resources(job: dict, work: Path) -> dict:
    if job["resource_shape"]["memory_mb"] < 160 * 1024:
        raise ValueError("CA3_ORIGINAL_160_GIB_MEMORY_REQUEST_REQUIRED")
    memory = memory_evidence(job["resource_shape"], required_available_gib=128)
    filesystem = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(work)], text=True).strip()
    if filesystem not in {"ext2/ext3", "ext4", "xfs", "btrfs", "tmpfs", "overlayfs", "zfs"}:
        raise ValueError("CA3_LOCAL_SCRATCH_REQUIRED:" + filesystem)
    report = dict(memory_evidence=memory, required_memory_gib=128, original_requested_memory_gib=160,
                  requested_resource_shape=job["resource_shape"], scratch_filesystem=filesystem,
                  scratch_free_bytes=shutil.disk_usage(work).free,
                  cpu_affinity_count=len(os.sched_getaffinity(0)))
    if not os.sched_getaffinity(0):
        raise ValueError("CA3_CPU_ALLOCATION_REQUIRED")
    return report


def download_inputs(job: dict, work: Path) -> Path:
    manifest = job["upload"]
    if manifest.get("complete") is not True:
        raise ValueError("CA3_COMPLETE_UPLOAD_MANIFEST_REQUIRED")
    needed = manifest["bytes"] + sum(row["bytes"] for row in job["input_files"].values()) + 20 * GIB
    if shutil.disk_usage(work).free < needed:
        raise ValueError("CA3_INPUT_STAGING_STORAGE_INSUFFICIENT")
    chunks = manifest["chunks"]
    if [row["index"] for row in chunks] != list(range(len(chunks))):
        raise ValueError("CA3_UPLOAD_CHUNK_ORDER_DIFFERS")
    archive = work / "ca3_inputs.tar.gz"
    # The deployed native HDFS client aborted on a multi-path cat. Read one
    # path at a time; a CPU-assembled archive reduces this to one download.
    full = hashlib.sha256()
    total = 0
    with archive.open("xb") as stream, (work / "input_download.log").open("xb") as errors:
        for chunk in chunks:
            part = work / "input_chunk.part"
            subprocess.run(["hdfs", "dfs", "-get", chunk["hdfs_path"], str(part)],
                           stderr=errors, check=True)
            chunk_hash = hashlib.sha256()
            chunk_bytes = 0
            with part.open("rb") as source:
                while data := source.read(8 * 1024 * 1024):
                    chunk_hash.update(data)
                    chunk_bytes += len(data)
                    full.update(data)
                    stream.write(data)
            if chunk_bytes != chunk["bytes"] or chunk_hash.hexdigest() != chunk["sha256"]:
                raise ValueError("CA3_UPLOAD_CHUNK_IDENTITY_DIFFERS")
            total += chunk_bytes
            part.unlink()
    if total != manifest["bytes"] or full.hexdigest() != manifest["sha256"]:
        raise ValueError("CA3_UPLOAD_ARCHIVE_IDENTITY_DIFFERS")
    return archive


def stage_inputs(archive: Path, expected: dict) -> None:
    """Restore only declared regular files at their provenance-bound paths."""
    # The verified Merlin image runs as an unprivileged container user. Restore
    # the source-bound project paths in this trial's local container filesystem.
    source_root = Path("/workspace")
    if not source_root.exists():
        subprocess.run(["sudo", "-n", "mkdir", "-p", str(source_root)], check=True)
        subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(source_root)], check=True)
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError("CA3_UNSAFE_INPUT_ARCHIVE_MEMBER")
            target = Path("/") / member.name
            key = str(target)
            if key not in expected or key in seen or member.name != key.lstrip("/"):
                raise ValueError("CA3_UNDECLARED_OR_DUPLICATE_INPUT_ARCHIVE_MEMBER")
            row = expected[key]
            if member.size != row["bytes"]:
                raise ValueError("CA3_INPUT_ARCHIVE_MEMBER_SIZE_DIFFERS")
            seen.add(key)
            if target.is_symlink() or target.resolve() != target:
                raise ValueError("CA3_INPUT_PATH_SYMLINK_CONFLICT")
            if target.exists():
                if not target.is_file() or target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                    raise ValueError("CA3_EXISTING_INPUT_CONFLICT:" + key)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = bundle.extractfile(member)
            if incoming is None:
                raise ValueError("CA3_INPUT_ARCHIVE_MEMBER_UNREADABLE")
            value = hashlib.sha256()
            temporary = target.with_name(target.name + ".merlin-partial")
            with incoming, temporary.open("xb") as stream:
                while data := incoming.read(8 * 1024 * 1024):
                    stream.write(data)
                    value.update(data)
            if temporary.stat().st_size != row["bytes"] or value.hexdigest() != row["sha256"]:
                raise ValueError("CA3_EXTRACTED_INPUT_IDENTITY_DIFFERS")
            temporary.chmod(member.mode & 0o777)
            os.link(temporary, target)
            temporary.unlink()
    if seen != set(expected):
        raise ValueError("CA3_INPUT_ARCHIVE_MEMBERS_MISSING")


def verify_scientific_inputs(job: dict) -> None:
    identity = job["execution_inputs"]
    runtime = Path(identity["prepared"]) / "ca3_runtime"
    built = json.loads((runtime / "build_report.json").read_text())
    if (digest(runtime / "build_report.json") != identity["runtime_build_sha256"]
            or built["control_program"]["duration_ms"] != 1210
            or built["binary_sha256"] != identity["runtime_binary_sha256"]
            or built["backend_libraries"] != identity["runtime_libraries"]):
        raise ValueError("CA3_FROZEN_1210MS_RUNTIME_DIFFERS")
    import yaml
    protocol = yaml.safe_load(Path(identity["config"]).read_text())["simulation_campaign"]["scenarios"]["ca3_pattern_completion"]["task_protocol"]
    if protocol["retrieval_duration_ms"] != 1210 or protocol["condition_duration_ticks"] != [1210] * 10:
        raise ValueError("CA3_FROZEN_1210MS_PROTOCOL_DIFFERS")
    old = json.loads((Path(identity["reference"]) / "report.json").read_text())
    if (old["duration_ms"] != 1001 or old["units"] != 168106
            or old["status"] != "COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE"
            or old["candidate_identity"]["circuit_index"] != 0
            or old["candidate_identity"]["block_index"] != 0
            or old["candidate_identity"]["condition_index"] != 3):
        raise ValueError("CA3_FROZEN_COMPLETION50_PREFIX_REFERENCE_DIFFERS")


def compose_candidate(job: dict, work: Path) -> dict:
    """Compose a private runtime and software tree from verified explicit inputs."""
    identity = job["execution_inputs"]
    candidate = job["candidate_runtime"]
    for name, binding in candidate["input_files"].items():
        path = Path(name)
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != binding["bytes"]
                or digest(path) != binding["sha256"]):
            raise ValueError("CA3_CANDIDATE_INPUT_CHANGED:" + name)
    parent = Path(identity["prepared"]) / "ca3_runtime"
    runtime = work / "candidate_runtime"
    source_parent = Path(identity["source_repo"])
    source = work / "candidate_source"
    # Use manifest members only. No unbound current modules enter the source tree.
    for name, binding in job["input_files"].items():
        path = Path(name)
        if path.is_relative_to(parent):
            if path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
                raise ValueError("CA3_PARENT_RUNTIME_CHANGED:" + name)
            target = runtime / path.relative_to(parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name, expected in identity["source_file_bindings"].items():
        path = Path(name)
        if not path.is_relative_to(source_parent) or digest(path) != expected:
            raise ValueError("CA3_FROZEN_SOFTWARE_CHANGED:" + name)
        target = source / path.relative_to(source_parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(candidate["consumer"]["path"], source / "Libs/population_dynamics/ca3_benchmark.py")
    library = "libcarlsim.so.6.0.0"
    shutil.copy2(candidate["library"]["path"], runtime / "lib" / library)
    descriptor = work / "output/runtime_composition.json"
    composition = dict(runtime_root=str(runtime),
        parent_runtime=dict(root=str(parent), build_report_sha256=identity["runtime_build_sha256"]),
        candidate_library=dict(name=library, **{key: candidate["library"][key] for key in ("bytes", "sha256")}),
        candidate_build_report=candidate["build_report"], candidate_receipt=candidate["receipt"])
    write_json(descriptor, composition)
    sys.path.insert(0, str(source / "Libs"))
    from population_dynamics.ca3_benchmark import validate_runtime_composition
    built = json.loads((parent / "build_report.json").read_text())
    libraries, verified = validate_runtime_composition(runtime, built, descriptor)
    if digest(runtime / "bin/ca3_benchmark") != identity["runtime_binary_sha256"]:
        raise ValueError("CA3_CANDIDATE_ORIGINAL_DRIVER_CHANGED")
    execution = dict(source_repo=str(source), packaged=str(runtime),
                     runtime_composition=str(descriptor), effective_runtime_libraries=libraries,
                     verified_composition=verified, parent_source_repo=str(source_parent),
                     source_file_bindings=identity["source_file_bindings"],
                     consumer=candidate["consumer"])
    write_json(work / "output/software_composition.json", execution)
    return execution


def execution_paths(job: dict) -> dict:
    if "candidate_runtime" in job:
        return job["candidate_execution"]
    identity = job["execution_inputs"]
    return dict(source_repo=identity["source_repo"],
                packaged=str(Path(identity["prepared"]) / "ca3_runtime"))



PERSISTENCE_CONDITION_INDICES = (0, 1, 2, 5, 6, 7, 8, 9)


def persistence_batch_fingerprint(batch: dict) -> str:
    return hashlib.sha256(json.dumps(batch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bound_document(binding: dict, *, json_document: bool = True):
    """Read embedded authoritative bytes without creating a legacy source path."""
    raw = binding["content"].encode("utf-8")
    path = Path(binding["path"])
    if (not path.is_absolute() or ".." in path.parts or len(raw) != binding["bytes"]
            or hashlib.sha256(raw).hexdigest() != binding["sha256"]):
        raise ValueError("CA3_PERSISTENCE_DOCUMENT_BINDING_DIFFERS")
    return json.loads(raw) if json_document else raw.decode("utf-8")


def validate_condition_native(native: dict, identity: dict, condition: dict) -> None:
    full = 168106 * 1210
    replay = native.get("independent_current_replay", {})
    expected = dict(intervals_compared=1210, current_values_compared=full,
                    exact_float32_bits_equal=True, mismatched_values=0, tolerance=0,
                    maximum_absolute_difference_pa=0.0)
    if (native.get("status") != "COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE"
            or native.get("units") != 168106 or native.get("duration_ms") != 1210
            or native.get("candidate_identity") != identity or native.get("condition") != condition
            or native.get("unit_millisecond_counts_compared") != full
            or native.get("native_intervals_compared") != 1210
            or native.get("native_terminal_footer_verified") is not True
            or native.get("all_unit_event_counter_equality") is not True
            or native.get("production_episodes") != 0
            or native.get("scientific_fidelity_qualified") is not False
            or any(replay.get(key) != value for key, value in expected.items())):
        raise ValueError("CA3_PERSISTENCE_FULL_EPISODE_GATE_FAILED")


def validate_persistence_batch(job: dict) -> dict:
    """Bind the real Merlin prefix and exact saved condition tasks."""
    batch = job["persistence_batch"]
    if (batch.get("schema") != "ca3_persistence_condition_batch_v1"
            or job.get("execution_mode") != "persistence_conditions"
            or "candidate_runtime" not in job or batch.get("production_episodes") != 0
            or batch.get("scientific_fidelity_qualified") is not False):
        raise ValueError("CA3_PERSISTENCE_BATCH_SCOPE_DIFFERS")
    bound_document(batch["source_runner"], json_document=False)
    submission = bound_document(batch["source_submission"])
    rows = batch["conditions"]
    indices = [row["condition_index"] for row in rows]
    if (not 1 <= len(rows) <= len(PERSISTENCE_CONDITION_INDICES)
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index not in PERSISTENCE_CONDITION_INDICES for index in indices)):
        raise ValueError("CA3_PERSISTENCE_CONDITION_SET_DIFFERS")
    saved = {row["condition_index"]: row for row in submission["jobs"]}
    if len({row["source_job_id"] for row in rows}) != len(rows):
        raise ValueError("CA3_PERSISTENCE_SOURCE_JOB_MAPPING_DIFFERS")
    for row in rows:
        original = saved[row["condition_index"]]
        request = bound_document(row["source_request"])
        response = bound_document(row["source_response"])
        if (str(original["job_id"]) != row["source_job_id"]
                or response.get("returncode") != 0 or str(response.get("job_id")) != row["source_job_id"]
                or original["expected_report"] != row["expected_legacy_report"]
                or not request["command"][-1].endswith(
                    batch["source_runner"]["path"] + " " + str(row["condition_index"]))):
            raise ValueError("CA3_PERSISTENCE_SOURCE_JOB_MAPPING_DIFFERS")
        if row.get("prior_report") is not None:
            prior = bound_document(row["prior_report"])
            if prior.get("status") == "PASS_COMPLETE_CA3_PERSISTENCE_DEVELOPMENT_CONDITION":
                raise ValueError("CA3_PERSISTENCE_CONDITION_ALREADY_COMPLETE")
    origin = validate_prefix_prerequisite(job, batch["prerequisite"])
    return dict(origin=origin, condition_indices=indices,
                source_jobs=[row["source_job_id"] for row in rows],
                binding_sha256=persistence_batch_fingerprint(batch))


def validate_prefix_prerequisite(job: dict, prerequisite: dict) -> dict:
    """Require the exact completed condition3 prefix for either continuation."""
    receipt = bound_document(prerequisite["receipt"])
    binding = bound_document(prerequisite["episode_binding"])
    native = bound_document(prerequisite["episode_report"])
    origin = prerequisite["origin"]
    if (receipt.get("status") != "COMPLETE_CA3_SM80_CANDIDATE_PREFIX_REPLAY"
            or receipt.get("execution_backend") != "merlin"
            or receipt.get("exit_code") != 0 or receipt.get("source_exit_code") != 0
            or receipt.get("persisted") is not True
            or any(receipt.get(key) != origin.get(key) for key in
                   ("merlin_job_run_id", "merlin_trial_id", "result_sha256"))
            or not all(origin.get(key) for key in
                       ("merlin_job_run_id", "merlin_trial_id", "result_sha256"))
            or receipt.get("episode_binding") != binding
            or binding.get("status") != "PASS_CA3_PREFIX_REPLAY_AND_EPISODE_BINDING"
            or receipt.get("production_episodes") != 0
            or receipt.get("scientific_fidelity_qualified") is not False):
        raise ValueError("CA3_REAL_MERLIN_PREFIX_PREREQUISITE_REQUIRED")
    identity = native["candidate_identity"]
    if (identity["circuit_index"], identity["block_index"], identity["condition_index"]) != (0, 0, 3):
        raise ValueError("CA3_COMPLETION50_PREREQUISITE_IDENTITY_DIFFERS")
    validate_condition_native(native, identity, native["task_protocol"]["conditions"][3])
    prefix = native.get("original_protocol_prefix", {})
    expected = dict(prefix_intervals_compared=1001, extended_intervals_verified=1210,
                    unit_millisecond_counts_compared=168106 * 1001,
                    current_values_compared=168106 * 1001, exact_float32_bits_equal=True,
                    counts_exact=True, original_terminal_footer_verified=True,
                    extended_terminal_footer_verified=True,
                    reference=job["execution_inputs"]["reference"])
    if (binding.get("original_protocol_prefix") != prefix
            or any(prefix.get(key) != value for key, value in expected.items())
            or native["runtime_composition"]["candidate_library"]["sha256"]
               != job["candidate_runtime"]["library"]["sha256"]
            or native["runtime_composition"]["parent_runtime"]["build_report_sha256"]
               != job["execution_inputs"]["runtime_build_sha256"]):
        raise ValueError("CA3_FROZEN_PREFIX_PREREQUISITE_GATE_FAILED")
    return origin


def validate_completion75_recovery(job: dict) -> dict:
    """Admit the changed-cluster branch while keeping the old cause unknown."""
    recovery = job["completion75_recovery"]
    if (job.get("execution_mode") != "completion75_recovery"
            or type(job.get("condition_index")) is not int or job["condition_index"] != 4
            or "persistence_batch" in job or "candidate_runtime" not in job
            or recovery.get("schema") != "ca3_completion75_recovery_v1"
            or recovery.get("condition") != dict(condition_id="completion_75", true_cue_units=413, false_cue_units=0)
            or recovery.get("retry_branch") != "relevant_cluster_condition_changed"
            or recovery.get("administrative_cancellation_cause_resolved") is not False
            or recovery.get("production_episodes") != 0
            or recovery.get("scientific_fidelity_qualified") is not False):
        raise ValueError("CA3_COMPLETION75_RECOVERY_SCOPE_DIFFERS")
    cancelled = [bound_document(item) for item in recovery["cancelled_attempts"]]
    if ([item.get("job_id") for item in cancelled] != [609703, 609907]
            or cancelled[0].get("condition_index") != 4
            or any(item.get("status") != "CANCELLED_INCOMPLETE_CA3_DEVELOPMENT"
                   or item.get("terminal_report_exists") is not False
                   or item.get("complete_episodes") != 0 or item.get("production_episodes") != 0
                   or "CANCELLED by 0" not in item.get("cancel_authority", "") for item in cancelled)
            or cancelled[1].get("retry_status") !=
               "WITHHELD_UNTIL_CANCELLATION_CAUSE_OR_RELEVANT_CLUSTER_CONDITION_CHANGES"):
        raise ValueError("CA3_COMPLETION75_CANCELLED_ATTEMPT_BINDING_DIFFERS")
    previous = bound_document(recovery["previous_resource_recheck"])
    changed = bound_document(recovery["merlin_batch_receipt"])
    results = bound_document(recovery["merlin_condition_results"])["conditions"]
    origin = recovery["changed_execution"]
    if (previous.get("hostname") != "gpu03" or previous.get("slurm_job_id") != "609898"
            or previous.get("status") != "PASS"
            or changed.get("status") != "COMPLETE_CA3_MERLIN_PERSISTENCE_BATCH"
            or changed.get("execution_backend") != "merlin"
            or changed.get("hostname") == previous["hostname"]
            or changed.get("gpu_name") != "NVIDIA A800-SXM4-80GB"
            or changed.get("compute_capability") != [8, 0]
            or changed.get("exit_code") != 0 or changed.get("persisted") is not True
            or changed.get("completed_development_executions") != 8
            or changed.get("production_episodes") != 0
            or changed.get("scientific_fidelity_qualified") is not False
            or any(not origin.get(key) or changed.get(key) != origin[key] for key in
                   ("merlin_job_run_id", "merlin_trial_id", "hostname", "gpu_name", "result_sha256"))
            or changed.get("condition_results") != results
            or len(results) != 8
            or {item.get("condition_index") for item in results} != set(PERSISTENCE_CONDITION_INDICES)
            or any(item.get("status") != "COMPLETE_CA3_MERLIN_PERSISTENCE_CONDITION"
                   or item.get("exit_code") != 0 or item.get("source_exit_code") != 0
                   or item.get("persisted") is not True for item in results)):
        raise ValueError("CA3_COMPLETION75_CHANGED_CLUSTER_EXECUTION_REQUIRED")
    resources = changed["resource_preflight"]["memory_evidence"]
    candidate = changed["candidate_preflight"]
    if (resources["admission"].get("status") != "PASS"
            or resources["admission"].get("required_available_memory_gib") != 128
            or resources["platform_allocation"]["allocated_memory_mib"] < 160 * 1024
            or resources["host_memory"]["memory_bytes"]["MemAvailable"] < 128 * GIB
            or candidate.get("status") != "PASS_CA3_PERSISTENCE_CONDITION_PREFLIGHT"
            or candidate.get("exit_code") != 0 or candidate.get("tests_exit_code") != 0
            or candidate.get("tests_run") != 32 or candidate.get("persisted") is not True
            or candidate.get("candidate_input_files") != job["candidate_runtime"]["input_files"]
            or candidate.get("source_file_bindings") != job["execution_inputs"]["source_file_bindings"]
            or candidate.get("runtime_build_sha256") != job["execution_inputs"]["runtime_build_sha256"]):
        raise ValueError("CA3_COMPLETION75_CHANGED_CLUSTER_SOURCE_BINDING_DIFFERS")
    prefix_origin = validate_prefix_prerequisite(job, recovery["prerequisite"])
    if changed["persistence_prerequisite"].get("origin") != prefix_origin:
        raise ValueError("CA3_COMPLETION75_BATCH_PREFIX_ORIGIN_DIFFERS")
    return dict(retry_branch="relevant_cluster_condition_changed",
                administrative_cancellation_cause_resolved=False,
                historical_cancelled_jobs=[609703, 609907], previous_hostname=previous["hostname"],
                changed_execution=origin, prefix_origin=prefix_origin,
                binding_sha256=persistence_batch_fingerprint(recovery))


def persistence_condition_inputs(job: dict, condition_index: int) -> tuple[dict, dict]:
    allowed = (4,) if job.get("execution_mode") == "completion75_recovery" else PERSISTENCE_CONDITION_INDICES
    if type(condition_index) is not int or condition_index not in allowed:
        raise ValueError("CA3_PERSISTENCE_CONDITION_SET_DIFFERS")
    sys.path.insert(0, str(Path(execution_paths(job)["source_repo"]) / "Libs"))
    from population_dynamics.ca3_benchmark import campaign_config, episode_identities
    campaign = campaign_config(Path(job["execution_inputs"]["config"]))
    identity = next(row for row in episode_identities(campaign, "ca3_pattern_completion")
                    if (row["circuit_index"], row["block_index"], row["condition_index"]) == (0, 0, condition_index))
    condition = campaign["scenarios"]["ca3_pattern_completion"]["task_protocol"]["conditions"][condition_index]
    if job.get("execution_mode") == "completion75_recovery" and condition != job["completion75_recovery"]["condition"]:
        raise ValueError("CA3_COMPLETION75_FROZEN_CONDITION_DIFFERS")
    return identity, condition


def execute_condition_batch(job: dict, work: Path, output: Path, target: str,
                            receipt: dict, environment: dict) -> list[dict]:
    """Persist each independent condition before advancing to the next task."""
    results = []
    for row in job["persistence_batch"]["conditions"]:
        index = row["condition_index"]
        condition_work = work / f"condition_{index:02d}"
        condition_output = condition_work / "output"
        condition_output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(work / "pack_results.py", condition_work / "pack_results.py")
        for name in ("runtime_composition.json", "software_composition.json", "execution_inputs.json"):
            path = output / name
            if path.is_file():
                shutil.copy2(path, condition_output / name)
        condition_target = target + f"/condition_{index:02d}"
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", condition_target], check=True)
        item = {key: receipt[key] for key in
                ("execution_backend", "merlin_job_run_id", "merlin_trial_id", "hostname")}
        item.update(status="FAILED_CA3_MERLIN_PERSISTENCE_CONDITION", exit_code=1,
                    condition_index=index, source_pending_job=row["source_job_id"],
                    prerequisite=receipt["persistence_prerequisite"], production_episodes=0,
                    scientific_fidelity_qualified=False, persisted=False, started_utc=datetime.now(timezone.utc).isoformat())
        interrupted = None
        try:
            item["resource_preflight"] = resources(job, condition_work)
            condition_job = dict(job, condition_index=index)
            command = source_command(condition_job, condition_output / "episode")
            item.update(status="RUNNING_CA3_MERLIN_PERSISTENCE_CONDITION", command=command)
            write_json(condition_output / "merlin_execution.json", item)
            execute_source(command, environment, condition_output, condition_target, item)
            item["source_exit_code"] = 0
            bound = bind_result(condition_job, condition_output / "episode")
            write_json(condition_output / "episode_binding.json", bound)
            item.update(status="COMPLETE_CA3_MERLIN_PERSISTENCE_CONDITION", exit_code=0,
                        episode_binding=bound, completed_development_executions=1)
        except BaseException as error:
            item.update(status="FAILED_CA3_MERLIN_PERSISTENCE_CONDITION",
                        error_type=type(error).__name__, error=str(error),
                        source_exit_code=getattr(error, "returncode", item.get("source_exit_code")))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                interrupted = error
        finally:
            item["finished_utc"] = datetime.now(timezone.utc).isoformat()
            # A failed publication stops this batch. Keep already published task
            # receipts and do not launch further unpersisted source executions.
            try:
                persist(condition_output, condition_work, condition_target, item, chunk_bytes=256 * 1024)
            except BaseException as error:
                item.update(persisted=False, persistence_error_type=type(error).__name__,
                            persistence_error=str(error))
                receipt["condition_results"] = [*results, {key: item.get(key) for key in
                    ("status", "condition_index", "source_pending_job", "exit_code", "persisted",
                     "checkpoint_path", "persistence_error")}]
                write_json(output / "condition_results.json", dict(conditions=receipt["condition_results"]))
                raise
            summary = {key: item.get(key) for key in
                       ("status", "condition_index", "source_pending_job", "exit_code", "source_exit_code",
                        "persisted", "checkpoint_path", "result_sha256", "result_bytes", "error")}
            results.append(summary)
            receipt["condition_results"] = results
            write_json(output / "condition_results.json", dict(conditions=results))
            if item.get("persisted") is not True:
                raise ValueError("CA3_CONDITION_RESULT_NOT_PERSISTED")
        if interrupted is not None:
            raise interrupted
    return results


def source_command(job: dict, episode: Path) -> list[str]:
    identity = job["execution_inputs"]
    paths = execution_paths(job)
    mode = job.get("execution_mode", "prefix")
    index = job.get("condition_index") if mode != "prefix" else 3
    allowed = (4,) if mode == "completion75_recovery" else PERSISTENCE_CONDITION_INDICES
    if mode not in {"prefix", "persistence_conditions", "completion75_recovery"} or (mode != "prefix" and
            (type(index) is not int or index not in allowed)):
        raise ValueError("CA3_EXECUTION_MODE_OR_CONDITION_DIFFERS")
    command = [sys.executable, "-m", "population_dynamics.ca3_benchmark", "run",
            "--packaged", paths["packaged"],
            "--formation", identity["formation"], "--analysis", identity["analysis"],
            "--config", identity["config"], "--output", str(episode),
            "--circuit-index", "0", "--block-index", "0", "--condition-index", str(index),
            "--current-replay-engine", identity["current_replay_engine"]]
    if mode == "prefix":
        command.extend(["--prefix-reference", identity["reference"]])
    if "candidate_runtime" in job:
        command.extend(["--runtime-composition", paths["runtime_composition"]])
    return command


def execute_source(command: list[str], environment: dict, output: Path, target: str, receipt: dict) -> None:
    child = None
    last_progress = 0.0
    sequence = 0
    try:
        with (output / "source_cli.log").open("xb") as log:
            child = subprocess.Popen(command, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if time.monotonic() - last_progress >= 900:
                    progress_started = time.monotonic()
                    native = output / "episode/native_observation.bin"
                    progress = dict(
                        merlin_job_run_id=receipt["merlin_job_run_id"],
                        merlin_trial_id=receipt["merlin_trial_id"], hostname=receipt["hostname"],
                        observed_utc=datetime.now(timezone.utc).isoformat(), source_pid=child.pid,
                        source_exit_code=child.poll(), native_observation_bytes=native.stat().st_size if native.exists() else 0,
                        snapshot_kind="progress_only", resumable_scientific_checkpoint=False,
                        production_episodes=0)
                    try:
                        telemetry = subprocess.run(
                            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                        progress["gpu_telemetry_exit_code"] = telemetry.returncode
                        progress["gpu_telemetry"] = telemetry.stdout.strip()
                    except (OSError, subprocess.TimeoutExpired) as error:
                        progress["gpu_telemetry_error_type"] = type(error).__name__
                    path = output / f"progress_{sequence:05d}.json"
                    write_json(path, progress)
                    try:
                        subprocess.run(["hdfs", "dfs", "-put", str(path), target + "/" + path.name],
                                       check=True, timeout=120)
                    except (OSError, subprocess.SubprocessError) as error:
                        progress["progress_persistence_error_type"] = type(error).__name__
                        write_json(path, progress)
                    sequence += 1
                    last_progress = progress_started
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    continue
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()


def bind_result(job: dict, episode: Path) -> dict:
    sys.path.insert(0, str(Path(execution_paths(job)["source_repo"]) / "Libs"))
    from population_dynamics.ca3_benchmark_readout import bind_episode
    bound = bind_episode(episode)
    native = bound["native"]
    if job.get("execution_mode") in {"persistence_conditions", "completion75_recovery"}:
        identity, condition = persistence_condition_inputs(job, job["condition_index"])
        validate_condition_native(native, identity, condition)
        return dict(status="PASS_COMPLETE_CA3_PERSISTENCE_DEVELOPMENT_CONDITION",
                    context=bound["context"], condition=condition, candidate_identity=identity,
                    units=168106, duration_ms=1210, events=native["events"],
                    unit_millisecond_counts_verified=168106 * 1210,
                    native_terminal_footer_verified=True, independent_current_replay_exact=True,
                    production_episodes=0, scientific_fidelity_qualified=False,
                    completed_development_executions=1)
    prefix = native.get("original_protocol_prefix", {})
    expected = dict(prefix_intervals_compared=1001, extended_intervals_verified=1210,
                    unit_millisecond_counts_compared=168106 * 1001,
                    current_values_compared=168106 * 1001, exact_float32_bits_equal=True,
                    counts_exact=True, original_terminal_footer_verified=True,
                    extended_terminal_footer_verified=True)
    if (native["duration_ms"] != 1210 or native["production_episodes"] != 0
            or native["independent_current_replay"].get("exact_float32_bits_equal") is not True
            or prefix.get("reference") != job["execution_inputs"]["reference"]
            or any(prefix.get(key) != value for key, value in expected.items())):
        raise ValueError("CA3_FROZEN_PREFIX_OR_CURRENT_REPLAY_GATE_FAILED")
    return dict(status="PASS_CA3_PREFIX_REPLAY_AND_EPISODE_BINDING",
                context=bound["context"], condition=native["condition"], units=native["units"],
                events=native["events"], duration_ms=native["duration_ms"],
                original_protocol_prefix=prefix, production_episodes=0,
                scientific_fidelity_qualified=False, completed_development_executions=1)


def persist(output: Path, work: Path, target: str, receipt: dict,
            *, chunk_bytes: int = 1024 * 1024, additional_roots: tuple[Path, ...] = ()) -> None:
    # The native run contains an input-checkpoint symlink. Return regular output
    # files and retain that input reference without duplicating its large bytes.
    roots = (output, *additional_roots)
    links = {str(path): os.readlink(path) for root in roots for path in root.rglob("*") if path.is_symlink()}
    write_json(output / "input_symlink_references.json", dict(references=links))
    write_json(output / "merlin_terminal.json", receipt)
    files = sorted({path for root in roots for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()})
    bundle = work / "result.tar.gz"
    subprocess.run([sys.executable, str(work / "pack_results.py"), "--output", str(bundle),
                    *map(str, files)], check=True)
    hdfs_put(bundle, target + "/result.tar.gz")
    chunks = []
    # Reuse the directory publication path from rechunk_merlin_results. Starting
    # one native HDFS client per small file kept a completed GPU trial occupied
    # for hours. The receipt still follows successful publication of every chunk.
    with tempfile.TemporaryDirectory(prefix="return-chunks-", dir=work) as directory:
        chunk_root = Path(directory)
        with bundle.open("rb") as stream:
            while data := stream.read(chunk_bytes):
                name = f"chunk_{len(chunks):05d}.json"
                write_json(chunk_root / name, dict(content_base64=base64.b64encode(data).decode("ascii")))
                chunks.append(dict(name="chunks/" + name, bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest()))
        hdfs_put(chunk_root, target + "/chunks")
    receipt.update(persisted=True, checkpoint_path=target, result_sha256=digest(bundle),
                   result_bytes=bundle.stat().st_size, chunks=chunks)
    write_json(work / "receipt.json", receipt)
    hdfs_put(work / "receipt.json", target + "/receipt.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    run_id = os.environ.get("RH2_JOB_RUN_ID")
    trial = os.environ.get("ARNOLD_TRIAL_ID")
    checkpoint = os.environ.get("RH2_CHECKPOINT_HDFS")
    if not run_id or not trial or not checkpoint or socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("ACTUAL_MERLIN_RUN_TRIAL_AND_CHECKPOINT_REQUIRED")
    job = json.loads(args.job_json.read_text())
    work = args.job_json.resolve().parent
    output = work / "output"
    output.mkdir(exist_ok=False)
    target = checkpoint.rstrip("/") + "/ca3/completion50_prefix/" + trial
    compatibility = "merlin_" + trial + "_ca3_completion50"
    candidate_mode = "candidate_runtime" in job
    condition_mode = job.get("execution_mode") == "persistence_conditions"
    recovery_mode = job.get("execution_mode") == "completion75_recovery"
    if candidate_mode:
        target = checkpoint.rstrip("/") + "/ca3/sm80_candidate_prefix/" + trial
        compatibility += "_sm80_candidate"
    if condition_mode:
        target = checkpoint.rstrip("/") + "/ca3/persistence_conditions/" + trial
        compatibility = "merlin_" + trial + "_ca3_persistence_conditions"
    if recovery_mode:
        target = checkpoint.rstrip("/") + "/ca3/completion75_recovery/" + trial
        compatibility = "merlin_" + trial + "_ca3_completion75_recovery"
    receipt = dict(
        status="PREPARING", execution_backend="merlin", merlin_job_run_id=run_id,
        merlin_trial_id=trial, hostname=socket.gethostname(), legacy_compatibility_id=compatibility,
        compatibility_id_semantics="SLURM_JOB_ID interface compatibility only; not Slurm execution",
        source_pending_job=job["execution_inputs"]["original_pending_head_job"],
        started_utc=datetime.now(timezone.utc).isoformat(), output=str(output),
        scope="ca3_1210ms_completion50_prefix_replay", production_episodes=0,
        scientific_fidelity_qualified=False, persisted=False,
        original_requested_memory_gib=160, required_available_memory_gib=128)
    if candidate_mode:
        receipt.update(scope="ca3_sm80_candidate_1210ms_completion50_prefix_replay",
                       candidate_qualified=False, candidate_preflight=job["candidate_preflight"])
    if condition_mode:
        receipt.pop("source_pending_job")
        receipt.update(scope="ca3_1210ms_persistence_condition_batch",
                       source_pending_jobs=[row["source_job_id"] for row in job["persistence_batch"]["conditions"]])
    if recovery_mode:
        receipt.pop("source_pending_job")
        receipt.update(scope="ca3_1210ms_completion75_recovery", condition_index=4,
                       historical_cancelled_jobs=[609703, 609907],
                       administrative_cancellation_cause_resolved=False,
                       retry_branch="relevant_cluster_condition_changed")
    write_json(output / "merlin_execution.json", receipt)
    write_json(output / "execution_inputs.json", job["execution_inputs"])
    exit_code = 1
    target_ready = False

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        target_ready = True
        if condition_mode:
            receipt["persistence_prerequisite"] = validate_persistence_batch(job)
        elif recovery_mode:
            receipt["recovery_prerequisite"] = validate_completion75_recovery(job)
        elif job.get("execution_mode", "prefix") != "prefix":
            raise ValueError("CA3_EXECUTION_MODE_OR_CONDITION_DIFFERS")
        receipt["resource_preflight"] = resources(job, work)
        if receipt["resource_preflight"]["memory_evidence"]["admission"]["status"] != "PASS":
            raise ValueError("CA3_MEMORY_ADMISSION_REQUIRED")
        allocated = int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0"))
        if allocated != job["resource_shape"]["gpu"] or allocated < 1:
            raise ValueError("CA3_ACTUAL_MERLIN_GPU_ALLOCATION_DIFFERS")
        # Use only the first allocated GPU. Extra GPUs from a fixed memory flavor
        # remain visible in the resource receipt and require controller approval.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        if not visible or not visible[0]:
            raise ValueError("CA3_VISIBLE_GPU_REQUIRED")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[0]
        import torch
        expected_capability = [8, 0] if candidate_mode else [8, 9]
        if (job["expected_compute_capability"] != expected_capability
                or not torch.cuda.is_available()
                or list(torch.cuda.get_device_capability(0)) != expected_capability):
            raise ValueError("CA3_RUNTIME_REQUIRES_MATCHING_ACTUAL_DEVICE:" + str(expected_capability))
        receipt.update(gpu_name=torch.cuda.get_device_name(0), compute_capability=expected_capability,
                       allocated_gpus=allocated, source_execution_gpus=1)
        receipt["environment_versions"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "h5py", "PyYAML", "psutil", "torch")}
        receipt["python_version"] = sys.version
        write_json(output / "merlin_resource_preflight.json", receipt)
        hdfs_put(output / "merlin_resource_preflight.json", target + "/merlin_resource_preflight.json")
        archive = download_inputs(job, work)
        stage_inputs(archive, job["input_files"])
        verify_scientific_inputs(job)
        if candidate_mode:
            layer_work = work / "candidate_inputs"
            layer_work.mkdir(exist_ok=False)
            layer = job["candidate_runtime"]
            stage_inputs(download_inputs(layer, layer_work), layer["input_files"])
            job["candidate_execution"] = compose_candidate(job, work)
        os.environ.update(SLURM_JOB_ID=compatibility,
                          SLURM_CPUS_PER_TASK=str(job["resource_shape"]["cpu"]),
                          OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                          PYTHONDONTWRITEBYTECODE="1", NRNSIM_EXECUTION_BACKEND="merlin")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(execution_paths(job)["source_repo"]) / "Libs")
        if condition_mode:
            results = execute_condition_batch(job, work, output, target, receipt, environment)
            completed = sum(row["exit_code"] == 0 for row in results)
            receipt.update(status=("COMPLETE_CA3_MERLIN_PERSISTENCE_BATCH" if completed == len(results)
                                   else "FAILED_CA3_MERLIN_PERSISTENCE_BATCH"),
                           completed_development_executions=completed,
                           failed_development_executions=len(results) - completed)
            exit_code = 0 if completed == len(results) else 1
        else:
            command = source_command(job, output / "episode")
            receipt.update(status="RUNNING", command=command)
            write_json(output / "merlin_execution.json", receipt)
            execute_source(command, environment, output, target, receipt)
            receipt["source_exit_code"] = 0
            result = bind_result(job, output / "episode")
            write_json(output / "episode_binding.json", result)
            receipt.update(status="COMPLETE_CA3_MERLIN_PREFIX_REPLAY", source_exit_code=0,
                           episode_binding=result, completed_development_executions=1)
            if candidate_mode:
                receipt["status"] = "COMPLETE_CA3_SM80_CANDIDATE_PREFIX_REPLAY"
            if recovery_mode:
                receipt["status"] = "COMPLETE_CA3_MERLIN_COMPLETION75_RECOVERY"
            exit_code = 0
    except BaseException as error:
        receipt.update(status="FAILED_CA3_MERLIN_PREFIX_REPLAY", error_type=type(error).__name__,
                       error=str(error), source_exit_code=getattr(error, "returncode", receipt.get("source_exit_code")))
        if candidate_mode:
            receipt["status"] = "FAILED_CA3_SM80_CANDIDATE_PREFIX_REPLAY"
        if condition_mode:
            receipt["status"] = "FAILED_CA3_MERLIN_PERSISTENCE_BATCH"
        if recovery_mode:
            receipt["status"] = "FAILED_CA3_MERLIN_COMPLETION75_RECOVERY"
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        receipt.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "merlin_terminal.json", receipt)
        if target_ready:
            try:
                persist(output, work, target, receipt, chunk_bytes=256 * 1024)
            except BaseException as error:
                receipt.update(persisted=False, persistence_error_type=type(error).__name__)
                write_json(output / "merlin_terminal.json", receipt)
                exit_code = exit_code or 1
        print(json.dumps(dict(status=receipt["status"], merlin_job_run_id=run_id,
                              merlin_trial_id=trial, persisted=receipt["persisted"],
                              exit_code=exit_code, production_episodes=0)), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

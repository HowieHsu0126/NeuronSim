#!/usr/bin/env python3
"""Run declared AppWorld development attempts with an admitted Merlin GPU model.

This entry point restores admitted inputs. It does not submit jobs, train a
model, or execute formal splits. Input records carry bytes and SHA-256 values;
the original model metadata remains the authority for the complete snapshot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import sys
import subprocess
import tarfile
import tempfile
import threading
import time
import traceback
from typing import Any

GPU_SAMPLE_SECONDS = 600
EXPERIMENT_SEEDS = (41, 42, 43)

CUGA_METHOD = "cuga_full_graph"
CUGA_PRIOR_LAUNCHER_BYTES = 811636
CUGA_PRIOR_LAUNCHER_SHA256 = "489ea4b9a5e99738741d3839611efdb0a6f9a05d663f7da8001f8e1678446a26"
CUGA_PUBLIC_MODULES = (
    "__init__.py", "provenance.py", "resource_probe.py", "appworld_reference.py",
    "appworld_collaboration.py", "appworld_action_decoder.py", "appworld_local_actor.py",
    "appworld_benchmark_protocol.py", "appworld_benchmark_execution.py",
    "appworld_benchmark_runner.py", "appworld_benchmark_transport.py",
    "appworld_benchmark_development.py", "appworld_benchmark_scoring.py",
    "appworld_benchmark_model_probe.py", "appworld_benchmark_model_runtime.py",
    "appworld_benchmark_collaboration.py", "appworld_benchmark_host.py",
    "appworld_benchmark_collaboration_development.py", "appworld_benchmark_pipeline.py",
    "appworld_benchmark_native_tools.py", "appworld_benchmark_cuga_channel.py",
    "appworld_benchmark_cuga_process.py", "appworld_benchmark_cuga_runtime.py",
    "appworld_benchmark_cuga_tools.py", "appworld_benchmark_cuga_receipt.py",
    "appworld_benchmark_cuga_development.py",
)
CUGA_PUBLIC_FILES = (
    *(f"Libs/population_dynamics/{name}" for name in CUGA_PUBLIC_MODULES),
    "Libs/configs/neural_population_appworld_benchmark.yaml",
    "Libs/configs/neural_population_appworld_local_actor.yaml",
)
CUGA_SOURCE_FILES = (
    *CUGA_PUBLIC_FILES,
    "Libs/population_dynamics/appworld_benchmark_native_binding.py",
    "Libs/scripts/prepare_appworld_cuga_merlin.py",
    "Libs/scripts/run_appworld_benchmark_merlin.py",
)
MINIMUM_SCRATCH_BYTES = 80 * 1024**3
NATIVE_PREFIX = Path("/tmp/neuronsim-appworld-native-runtime")
MODEL_PREFIX = Path("/tmp/neuronsim-appworld-model-runtime")
DATA_PREFIX = Path("/tmp/neuronsim-appworld-data-admission")
BENCHMARK_CONFIG = "Libs/configs/neural_population_appworld_benchmark.yaml"
ACTOR_CONFIG = "Libs/configs/neural_population_appworld_local_actor.yaml"
MAX_HDFS_ATTEMPTS = 3
DECISIVE_ARTIFACTS = (
    "qualification/qualification.json",
    "development/pipeline.json",
    "development/attempt/execution_receipt.json",
    "development/attempt/development_score.json",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def verify_bytes(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    actual = file_identity(path)
    if any(actual[key] != record[key] for key in ("bytes", "sha256")):
        raise ValueError("DECLARED_FILE_IDENTITY_DIFFERS: " + str(path))
    return actual


def declared_content(record: dict[str, Any]) -> bytes:
    content = base64.b64decode(record["content_base64"], validate=True)
    if (len(content) != record["bytes"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]):
        raise ValueError("DECLARED_EMBEDDED_BYTES_DIFFER")
    return content


def require_merlin_gpu(environment: dict[str, str]) -> dict[str, Any]:
    run_id = environment.get("RH2_JOB_RUN_ID", "")
    trial_id = environment.get("ARNOLD_TRIAL_ID", "")
    checkpoint = environment.get("RH2_CHECKPOINT_HDFS", "").rstrip("/")
    hostname = socket.gethostname()
    gpu_count = int(environment.get("ARNOLD_APPLIED_GPU_NUM", "0"))
    if (not run_id or not trial_id or gpu_count <= 0
            or not checkpoint.startswith("hdfs://")
            or checkpoint.rsplit("/", 1)[-1] != run_id
            or "SLURM_JOB_ID" in environment
            or hostname.split(".")[0].lower() == "mgmt2"
            or "/" in run_id or "/" in trial_id):
        raise ValueError("APPWORLD_DEVELOPMENT_REQUIRES_REAL_MERLIN_GPU_ALLOCATION")
    return {
        "execution_backend": "merlin",
        "run_id": run_id,
        "trial_id": trial_id,
        "hostname": hostname,
        "allocated_gpus": gpu_count,
        "checkpoint_path": checkpoint + "/appworld_gpu_development",
    }


class HdfsStore:
    """Use the observed HDFS retry and settled-write reconciliation contract."""

    def __init__(self, prefix: str, work: Path, artifacts: Path,
                 timeout: int = 1800,
                 cancel_event: threading.Event | None = None) -> None:
        self.prefix = prefix
        self.work = work
        self.artifacts = artifacts
        self.timeout = timeout
        self.cancel_event = cancel_event
        self.event_lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        with self.event_lock:
            with (self.artifacts / "publication_events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": time.time(), **event}) + "\n")
        print(json.dumps(event), flush=True)

    def hdfs(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ["hdfs", "dfs", *arguments]
        if self.cancel_event is None:
            result = subprocess.run(
                command, text=True, capture_output=True, timeout=self.timeout,
            )
        else:
            if self.cancel_event.is_set():
                raise InterruptedError("DIAGNOSTIC_PUBLICATION_STOPPED")
            process = subprocess.Popen(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.timeout
            try:
                while True:
                    if self.cancel_event.is_set():
                        raise InterruptedError("DIAGNOSTIC_PUBLICATION_STOPPED")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, self.timeout)
                    try:
                        stdout, stderr = process.communicate(timeout=min(1.0, remaining))
                        result = subprocess.CompletedProcess(
                            command, process.returncode, stdout, stderr,
                        )
                        break
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                cleanup = stop_child(process)
                self.record({
                    "event": "diagnostic_hdfs_interrupted",
                    "arguments": list(arguments), "cleanup": cleanup,
                })
                if not cleanup["owner_terminated"]:
                    raise RuntimeError("DIAGNOSTIC_HDFS_OWNER_TERMINATION_UNCONFIRMED")
                raise
        if check and result.returncode:
            raise RuntimeError(
                "HDFS_COMMAND_FAILED: " + repr(arguments) + "\n" + result.stderr[-8000:]
            )
        return result

    def retry_pause(self, seconds: float) -> None:
        if self.cancel_event is None:
            time.sleep(seconds)
        elif self.cancel_event.wait(seconds):
            raise InterruptedError("DIAGNOSTIC_PUBLICATION_STOPPED")

    def remote_size(self, uri: str) -> int | None:
        for attempt in range(1, MAX_HDFS_ATTEMPTS + 1):
            result = self.hdfs("-stat", "%b", uri, check=False)
            if result.returncode == 0:
                value = result.stdout.strip()
                if not value.isdecimal():
                    raise ValueError("HDFS_SIZE_INVALID: " + uri)
                return int(value)
            message = result.stdout + result.stderr
            denied = any(marker in message.lower() for marker in (
                "permission denied", "accesscontrolexception",
                "authentication", "authorization",
            ))
            crashed = "signal: quit (core dumped)" in message
            if crashed and not denied and attempt < MAX_HDFS_ATTEMPTS:
                self.record({
                    "event": "read_only_stat_crash_retry", "target": uri,
                    "attempt": attempt, "exit_code": result.returncode,
                })
                self.retry_pause(5 * attempt)
                continue
            if (not denied and not crashed and "DT_PROTO_ERROR" not in message
                    and "java.io.EOFException" not in message
                    and any(marker in message for marker in (
                        "No such file", "does not exist", "FileNotFoundException",
                    ))):
                return None
            raise RuntimeError("HDFS_STATE_UNRESOLVED: " + uri + "\n" + message[-8000:])
        raise RuntimeError("HDFS_STAT_RETRIES_EXHAUSTED: " + uri)

    def remote_identity(self, uri: str, size: int) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="publication-reconcile-", dir=self.work) as tmp:
            target = Path(tmp) / "payload"
            self.hdfs("-get", uri, str(target))
            identity = file_identity(target)
            if identity["bytes"] != size:
                raise ValueError("HDFS_RECONCILIATION_SIZE_CHANGED: " + uri)
            return identity

    def publish(self, path: Path, relative: str, *, replace: bool = False) -> dict[str, Any]:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("PUBLICATION_OUTSIDE_CURRENT_RUN")
        uri = self.prefix + "/" + relative_path.as_posix()
        expected = file_identity(path)
        self.hdfs("-mkdir", "-p", uri.rsplit("/", 1)[0])
        size = self.remote_size(uri)
        previous = None if size is None else self.remote_identity(uri, size)
        if previous == expected:
            return {"uri": uri, **expected}
        if previous is not None and not replace:
            raise ValueError("EXISTING_PUBLISHED_IDENTITY_DIFFERS: " + uri)
        for attempt in range(1, MAX_HDFS_ATTEMPTS + 1):
            result = self.hdfs("-put", "-f", str(path), uri, check=False)
            size = self.remote_size(uri)  # The writer has exited before reconciliation.
            if result.returncode == 0:
                if size != expected["bytes"]:
                    raise ValueError("PUBLISHED_SIZE_DIFFERS: " + uri)
                return {"uri": uri, **expected}
            message = result.stdout + result.stderr
            self.record({
                "event": "publication_failed", "target": uri, "attempt": attempt,
                "exit_code": result.returncode, "remote_bytes": size,
                "error": message[-8000:],
            })
            denied = any(marker in message.lower() for marker in (
                "permission denied", "accesscontrolexception",
                "authentication", "authorization",
            ))
            eof_write_failure = (
                "java.io.EOFException" in message
                and any(marker in message.lower() for marker in (
                    "datanode", "failed to write", "write failed", "write failure",
                ))
            )
            if denied or not ("DT_PROTO_ERROR" in message or eof_write_failure):
                raise RuntimeError("NON_RETRYABLE_HDFS_PUBLICATION: " + message[-8000:])
            after = None if size is None else self.remote_identity(uri, size)
            if after == expected:
                self.record({"event": "publication_ack_reconciled", "target": uri})
                return {"uri": uri, **expected}
            if after is not None and (not replace or after != previous):
                raise ValueError("PUBLISHED_IDENTITY_DIFFERS_AFTER_FAILURE: " + uri)
            partial = uri + "._COPYING_"
            partial_size = self.remote_size(partial)
            self.record({
                "event": "publication_state_reconciled", "target": uri,
                "attempt": attempt, "partial_bytes": partial_size,
            })
            if partial_size is not None and partial_size > expected["bytes"]:
                raise ValueError("PUBLICATION_PARTIAL_EXCEEDS_EXPECTED_SIZE: " + partial)
            if attempt == MAX_HDFS_ATTEMPTS:
                raise RuntimeError("HDFS_PUBLICATION_RETRIES_EXHAUSTED: " + uri)
            if partial_size is not None:
                # Only this settled failed write's temporary object can be removed.
                self.hdfs("-rm", partial)
            self.retry_pause(5 * attempt)
        raise RuntimeError("HDFS_PUBLICATION_RETRIES_EXHAUSTED: " + uri)



def _verify_existing_runtime_member(handle: tarfile.TarFile, member: tarfile.TarInfo,
                                    destination: Path) -> None:
    """Verify an image-provided entry without modifying it."""
    matches = False
    if member.isdir():
        matches = destination.is_dir() and not destination.is_symlink()
    elif member.issym():
        matches = destination.is_symlink() and os.readlink(destination) == member.linkname
    elif (member.isfile() or member.islnk()) and destination.is_file() and not destination.is_symlink():
        digest, size = hashlib.sha256(), 0
        with handle.extractfile(member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        matches = file_identity(destination) == {"bytes": size, "sha256": digest.hexdigest()}
        matches = matches and (destination.stat().st_mode & 0o111) == (member.mode & 0o111)
    if not matches:
        raise ValueError("EXISTING_RUNTIME_ENTRY_DIFFERS: " + str(destination))

def restore_archive(store: HdfsStore, work: Path, name: str, uri: str,
                    identity: dict[str, Any], roots: tuple[Path, ...],
                    link_roots: tuple[Path, ...], *,
                    existing_identical_roots: tuple[Path, ...] = (),
                    existing_link_aliases: dict[Path, Path] | None = None) -> None:
    if not set(existing_identical_roots).issubset(roots):
        raise ValueError("EXISTING_RUNTIME_ROOT_OUTSIDE_RESTORE_ROOTS")
    archive = work / (name + ".tar.gz")
    store.hdfs("-get", uri, str(archive))
    verify_bytes(archive, identity)
    existing = tuple(root for root in roots if root.exists() or root.is_symlink())
    if any(root not in existing_identical_roots or root.is_symlink()
           or not root.is_dir() or root.resolve() != root for root in existing):
        raise ValueError("RESTORE_PREFIX_ALREADY_EXISTS: " + repr(roots))
    aliases = dict(existing_link_aliases or {})
    for alias, target in aliases.items():
        try:
            resolved = alias.resolve(strict=True)
            expected = target.resolve(strict=True)
        except OSError as error:
            raise ValueError("EXISTING_RUNTIME_LINK_ALIAS_DIFFERS: " + str(alias)) from error
        if (not alias.is_absolute() or resolved != expected
                or not expected.is_file()
                or not any(expected.is_relative_to(root) for root in existing)):
            raise ValueError("EXISTING_RUNTIME_LINK_ALIAS_DIFFERS: " + str(alias))
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if len({member.name for member in members}) != len(members):
            raise ValueError("DUPLICATE_ARCHIVE_MEMBER")
        for member in members:
            path = Path(member.name)
            destination = Path("/") / path
            if (path.is_absolute() or ".." in path.parts
                    or not any(destination.is_relative_to(root) for root in roots)):
                raise ValueError("UNSAFE_ARCHIVE_MEMBER: " + member.name)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError("UNSUPPORTED_ARCHIVE_MEMBER: " + member.name)
            if member.issym() or member.islnk():
                link = Path(member.linkname)
                if link.is_absolute():
                    target = link
                elif member.islnk():
                    target = Path("/") / link
                else:
                    target = destination.parent / link
                normalized = Path(os.path.normpath(str(target)))
                if (not any(normalized.is_relative_to(root) for root in link_roots)
                        and normalized not in aliases):
                    raise ValueError("UNSAFE_ARCHIVE_LINK: " + member.name)
        selected = []
        for member in members:
            destination = Path("/") / member.name
            if any(destination.is_relative_to(root) for root in existing):
                _verify_existing_runtime_member(handle, member, destination)
            else:
                selected.append(member)
        handle.extractall("/", members=selected, filter="fully_trusted")
    archive.unlink()


class GpuMonitor:
    """Sample independently of HDFS; retain immutable diagnostics on shutdown."""

    def __init__(self, store: HdfsStore, output: Path, report: dict[str, Any]) -> None:
        self.output = output
        self.report = report
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.store = HdfsStore(
            store.prefix, store.work, output, timeout=60, cancel_event=self.stop_event,
        )
        self.store.event_lock = store.event_lock
        self.pending: queue.Queue[tuple[Path, dict[str, Any]]] = queue.Queue()
        self.sampling_thread = threading.Thread(
            target=self._sampling_loop, name="appworld-gpu-sampler",
        )
        self.publication_thread = threading.Thread(
            target=self._publication_loop, name="appworld-diagnostic-publisher",
        )
        self.snapshots: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.started = False

    def _error(self, kind: str, error: BaseException | str, **details: Any) -> None:
        record = {"time": time.time(), "kind": kind, "error": str(error), **details}
        if isinstance(error, BaseException):
            record["error_type"] = type(error).__name__
        with self.state_lock:
            self.errors.append(record)

    def start(self) -> None:
        if self.started or self.stop_event.is_set():
            raise RuntimeError("GPU_MONITOR_ALREADY_STARTED_OR_STOPPED")
        self.started = True
        try:
            self.publication_thread.start()
            self.sampling_thread.start()
        except Exception as error:
            self._error("startup", error)
            self.stop()
            raise

    def stop(self) -> None:
        self.stop_event.set()
        # Do not join the queue. Queued immutable files enter the final bundle.
        # The publication store cancels and reaps its in-flight HDFS client.
        for thread in (self.sampling_thread, self.publication_thread):
            if thread.ident is not None:
                thread.join()
        with self.state_lock:
            for record in self.snapshots:
                if record["publication_status"] == "QUEUED":
                    record["publication_status"] = "LOCAL_ONLY_AT_SHUTDOWN"

    def _sample(self) -> None:
        sampled_at = time.time()
        with self.state_lock:
            sequence = len(self.snapshots)
        snapshot: dict[str, Any] = {
            "schema": "appworld_gpu_development_diagnostic_v1",
            "sampled_at": sampled_at,
            "run_id": self.report["run_id"],
            "trial_id": self.report["trial_id"],
            "stage": self.report.get("stage"),
            "status": self.report["status"],
            "resumable_checkpoint": False,
            "active_task_databases_included": False,
            "log_tails": {},
            "log_errors": {},
        }
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,"
                 "utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30,
            )
            snapshot["gpu"] = {
                "exit_code": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if result.returncode:
                self._error("gpu_sample", "NVIDIA_SMI_FAILED", sampled_at=sampled_at)
        except Exception as error:
            snapshot["gpu"] = {"error_type": type(error).__name__, "error": str(error)}
            self._error("gpu_sample", error, sampled_at=sampled_at)
        for path in sorted(self.output.rglob("*.log")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(self.output).as_posix()
            try:
                with path.open("rb") as stream:
                    stream.seek(max(0, os.fstat(stream.fileno()).st_size - 65536))
                    snapshot["log_tails"][relative] = (
                        stream.read(65536).decode("utf-8", errors="replace")
                    )
            except OSError as error:
                snapshot["log_errors"][relative] = str(error)
                self._error("log_sample", error, path=relative)
        path = self.output / "diagnostics" / f"sample_{sequence:05d}.json"
        write_json(path, snapshot)
        record: dict[str, Any] = {
            "path": path.relative_to(self.output).as_posix(),
            "sampled_at": sampled_at,
            "persisted_at": None,
            "publication_status": "QUEUED",
        }
        with self.state_lock:
            self.snapshots.append(record)
        self.pending.put((path, record))

    def _sampling_loop(self) -> None:
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self._sample()
            except Exception as error:
                self._error("sampling", error, traceback=traceback.format_exc())
            deadline += GPU_SAMPLE_SECONDS
            if self.stop_event.wait(max(0, deadline - time.monotonic())):
                break

    def _publication_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                path, record = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.stop_event.is_set():
                break
            with self.state_lock:
                record["publication_status"] = "PUBLISHING"
            try:
                artifact = self.store.publish(path, record["path"])
                with self.state_lock:
                    record.update(
                        artifact=artifact, persisted_at=time.time(),
                        publication_status="PERSISTED",
                    )
            except InterruptedError as error:
                # No retry after shutdown: a cancelled write has unresolved remote
                # state. Its local immutable snapshot remains in the final bundle.
                with self.state_lock:
                    record.update(
                        publication_status="CANCELLED_AT_SHUTDOWN",
                        publication_error=str(error), remote_state_unverified=True,
                    )
                if not self.stop_event.is_set():
                    self._error("publication", error, snapshot=record["path"])
            except Exception as error:
                with self.state_lock:
                    record.update(
                        publication_status="FAILED", publication_error=str(error),
                    )
                self._error("publication", error, snapshot=record["path"])


PREPARE_CONFIG = """
import json, pathlib, sys, yaml
from population_dynamics.appworld_local_actor import snapshot_entries
work = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else work / 'artifacts'
inputs = json.loads((work / 'inputs.json').read_text())
config = yaml.safe_load((work / 'Libs/configs/neural_population_appworld_benchmark.yaml').read_text())
config['source']['archive_manifest'] = {
    'path': str(work / 'source_archive_manifest.json'),
    'sha256': inputs['archive_manifest']['sha256'],
}
config['development_execution']['source_root'] = inputs['native_runtime']['environment']['source_checkout']
(output / 'bound_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
actor = yaml.safe_load((work / 'Libs/configs/neural_population_appworld_local_actor.yaml').read_text())
entries = snapshot_entries(inputs['source_metadata'], actor)
receipt = inputs['model_inputs']
if set(entries) != set(receipt['files']):
    raise ValueError('ADMITTED_MODEL_FILE_SET_DIFFERS_FROM_PINNED_METADATA')
if sum(row['size'] for row in entries.values()) != receipt['model_bytes']:
    raise ValueError('ADMITTED_MODEL_TOTAL_BYTES_DIFFER')
for name, row in entries.items():
    path = pathlib.Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('UNSAFE_MODEL_PATH:' + name)
    if row['size'] != receipt['files'][name]['bytes']:
        raise ValueError('ADMITTED_MODEL_FILE_SIZE_DIFFERS:' + name)
(work / 'model_entries.json').write_text(json.dumps(entries, indent=2) + '\\n')
"""

VERIFY_SNAPSHOT = """
import json, pathlib, sys, yaml
from population_dynamics.appworld_local_actor import verify_snapshot
work = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else work / 'artifacts'
inputs = json.loads((work / 'inputs.json').read_text())
actor = yaml.safe_load((work / 'Libs/configs/neural_population_appworld_local_actor.yaml').read_text())
verified = verify_snapshot(work / 'model', inputs['source_metadata'], actor)
if verified.get('all_published_file_identities_verified') is not True:
    raise ValueError('COMPLETE_PINNED_SNAPSHOT_REQUIRED')
if verified['files'] != inputs['model_inputs']['files']:
    raise ValueError('RESTORED_MODEL_SNAPSHOT_DIFFERS_FROM_ADMITTED_SNAPSHOT')
(output / 'model_snapshot.json').write_text(json.dumps(verified, indent=2) + '\\n')
"""

SELECT_TASK = """
import json, pathlib, sys
from population_dynamics.appworld_benchmark_development import load_benchmark_config
from population_dynamics.appworld_benchmark_pipeline import select_development_task
work = pathlib.Path(sys.argv[1])
artifact_output = pathlib.Path(sys.argv[6]) if len(sys.argv) > 6 else work / 'artifacts'
selection = select_development_task(
    load_benchmark_config(artifact_output / 'bound_config.yaml'),
    json.loads((work / 'split_manifest.json').read_text()),
    split=sys.argv[2], index=int(sys.argv[3]),
)
selection['seed'] = int(sys.argv[4])
directory = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else work / 'artifacts'
directory.mkdir(parents=True, exist_ok=True)
(directory / 'selection.json').write_text(json.dumps(selection, indent=2) + '\\n')
(directory / 'selected_task_id.txt').write_text(selection['task_id'] + '\\n')
"""


ADMIT_QUALIFICATION = """
import json, pathlib, sys
from population_dynamics.appworld_benchmark_development import load_benchmark_config, validate_model_qualification
config = load_benchmark_config(pathlib.Path(sys.argv[1]))
qualification = json.loads(pathlib.Path(sys.argv[2]).read_text())
validate_model_qualification(config, qualification)
"""


def admitted_native_data_link(
    store: HdfsStore, output: Path, path: Path,
) -> dict[str, Any] | None:
    """Identify registered runner-owned input links without traversing their targets."""
    relative = path.relative_to(output).as_posix()
    roots = getattr(store, "native_attempt_roots", ("development/attempt",))
    attempt_links = {root + "/data" for root in roots}
    session_links = {root + "/data" for root in getattr(store, "native_session_roots", ())}
    # Generic registrations retain their execution receipt even if both lists contain a root.
    native_session = relative not in attempt_links and relative in session_links
    if relative not in attempt_links and not native_session:
        return None
    metadata_paths = (
        store.work / "native_preparation.receipt.json",
        output / "input_admission.json",
        path.parent / ("native_session.json" if native_session else "execution_receipt.json"),
    )
    records = []
    for metadata in metadata_paths:
        if metadata.is_symlink() or not metadata.is_file():
            raise ValueError("NATIVE_DATA_LINK_ADMISSION_METADATA_REQUIRED: " + str(metadata))
        records.append(json.loads(metadata.read_text()))
    prepared, restored, owner = records
    environment = prepared.get("environment", {})
    data_root = environment.get("data_root")
    archive = {
        "uri": prepared.get("data_hdfs_uri"),
        "sha256": prepared.get("data_bundle_sha256"),
    }
    if (prepared.get("status") != "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION"
            or prepared.get("exit_code") != 0 or prepared.get("persisted") is not True
            or environment.get("status") != "PREPARED_RUNTIME_AND_SPLIT_METADATA"
            or not isinstance(data_root, str) or not Path(data_root).is_absolute()
            or prepared.get("data_root") != data_root
            or restored.get("status") != "RESTORED_INPUT_IDENTITIES_VERIFIED"
            or restored.get("data_archive") != archive):
        raise ValueError("NATIVE_DATA_LINK_ADMISSION_IDENTITY_DIFFERS")
    target = str(Path(data_root) / "data")
    if native_session:
        valid_owner = (
            owner.get("schema") == "appworld_native_tool_session_v1"
            and owner.get("scope") == "development_tools_only"
            and owner.get("formal_eligible") is False
        )
    else:
        valid_owner = (
            owner.get("schema") == "appworld_development_execution_v1"
            and owner.get("native_data_directory") == target
        )
    if (not valid_owner
            or owner.get("run_root") != str(path.parent)
            or owner.get("data_root") != data_root
            or os.readlink(path) != target):
        raise ValueError("NATIVE_DATA_LINK_TARGET_OR_OWNER_DIFFERS")
    return {"target": target, "data_archive": archive}


def decisive_artifact_names(store: HdfsStore) -> tuple[str, ...]:
    roots = getattr(store, "native_attempt_roots", None)
    if roots is None:
        generic = DECISIVE_ARTIFACTS
    else:
        generic = ("qualification/qualification.json", "development_batch.json", *(
            name for root in roots for name in (
                str(Path(root).parent / "pipeline.json"),
                root + "/execution_receipt.json", root + "/development_score.json",
            )
        ))
    return tuple(dict.fromkeys((*generic, *cuga_decisive_artifact_names(store))))


def publish_closed(store: HdfsStore, output: Path, paths: list[Path]) -> dict[str, Any]:
    records = {}
    for root in paths:
        candidates = [root] if root.is_symlink() or not root.is_dir() else sorted(root.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                if admitted_native_data_link(store, output, path) is not None:
                    continue
                raise ValueError("OUTPUT_SYMLINK_CANNOT_BE_PUBLISHED: " + str(path))
            if path.is_file():
                relative = path.relative_to(output).as_posix()
                records[relative] = store.publish(path, "stages/" + relative)
    return records


def stop_child(process: subprocess.Popen[str]) -> dict[str, Any]:
    """Reap the owner and report its exit; server cleanup remains the owner's job."""
    cleanup: dict[str, Any] = {
        "scope": "owner_process", "pid": process.pid, "signals": [],
        "owner_terminated": False, "reaped": False, "returncode": None,
    }

    def reap(timeout: float) -> bool:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        cleanup.update(
            returncode=returncode, reaped=True,
            owner_terminated=returncode is not None,
        )
        return cleanup["owner_terminated"]

    if process.poll() is not None and reap(0):
        return cleanup
    # SIGINT lets Python owners execute their existing model/scorer cleanup.
    for sig, wait_seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        if process.poll() is not None and reap(0):
            return cleanup
        event: dict[str, Any] = {"signal": sig.name}
        try:
            os.killpg(process.pid, sig)
            event["status"] = "SENT"
        except ProcessLookupError:
            # The owner can exit between poll() and killpg(); wait() decides.
            event["status"] = "PROCESS_GROUP_ALREADY_EXITED"
        except OSError as error:
            event.update(status="SIGNAL_FAILED", error=str(error))
        cleanup["signals"].append(event)
        if reap(wait_seconds):
            return cleanup
    reap(0)
    return cleanup


def run_command(stage: str, command: list[str], environment: dict[str, str],
                work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
                closed_outputs: list[Path]) -> None:
    report["stage"] = stage
    started = time.time()
    stage_record: dict[str, Any] = {
        "stage": stage, "command": command, "started_at": started,
        "exit_code": None,
    }
    stage_path = output / "stage_receipts" / (stage + ".json")
    log_path = output / (stage + ".log")
    process = None
    outputs_closed = True
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, cwd=work, env=environment, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            stage_record["pid"] = process.pid
            stage_record["exit_code"] = process.wait()
            stage_record["cleanup"] = stop_child(process)
            outputs_closed = stage_record["cleanup"]["owner_terminated"]
            if not outputs_closed:
                raise RuntimeError("OWNER_TERMINATION_UNCONFIRMED: " + stage)
    except BaseException as error:
        stage_record["error"] = traceback.format_exc()
        if process is not None:
            stage_record["cleanup"] = stop_child(process)
            stage_record["exit_code"] = stage_record["cleanup"]["returncode"]
            outputs_closed = stage_record["cleanup"]["owner_terminated"]
            if not outputs_closed:
                raise RuntimeError("OWNER_TERMINATION_UNCONFIRMED: " + stage) from error
        raise
    finally:
        stage_record["outputs_closed"] = outputs_closed
        stage_record["finished_at"] = time.time()
        stage_record["elapsed_seconds"] = stage_record["finished_at"] - started
        write_json(stage_path, stage_record)
        report.setdefault("stages", {})[stage] = stage_record
        paths = [stage_path]
        if outputs_closed:
            paths.extend([log_path, *closed_outputs])
        else:
            unclosed = report.setdefault("unclosed_output_paths", [])
            unclosed.extend(
                path.relative_to(output).as_posix() for path in [log_path, *closed_outputs]
            )
        records = publish_closed(store, output, paths)
        report.setdefault("decisive_artifacts", {}).update({
            name: row for name, row in records.items() if name in decisive_artifact_names(store)
        })
        stage_record["publication_complete"] = True
    if stage_record["exit_code"] != 0:
        raise RuntimeError("CHILD_STAGE_FAILED: " + stage + "; see " + str(log_path))


def model_download_uris(intake: dict[str, Any]) -> dict[str, str]:
    """Resolve locations separately from the snapshot's immutable identity rows."""
    if "model_file_uris" in intake:
        locations = intake["model_file_uris"]
        if (not isinstance(locations, dict)
                or set(locations) != set(intake["files"])
                or any(not isinstance(uri, str) or not uri.startswith("hdfs://")
                       for uri in locations.values())):
            raise ValueError("COMPLETE_HDFS_MODEL_FILE_URIS_REQUIRED")
        return dict(locations)
    prefix = intake.get("model_hdfs_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("hdfs://"):
        raise ValueError("LEGACY_HDFS_MODEL_PREFIX_REQUIRED")
    return {name: prefix.rstrip("/") + "/" + name for name in intake["files"]}


def validate_qualification_reference(reference: Any) -> None:
    if (not isinstance(reference, dict)
            or set(reference) != {"uri", "bytes", "sha256"}
            or not isinstance(reference["uri"], str)
            or not reference["uri"].startswith("hdfs://")
            or len(reference["uri"]) <= len("hdfs://")
            or type(reference["bytes"]) is not int or reference["bytes"] <= 0
            or not isinstance(reference["sha256"], str)
            or len(reference["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in reference["sha256"])):
        raise ValueError("MODEL_QUALIFICATION_REQUIRES_HDFS_URI_BYTES_SHA256")


def development_output_root(work: Path, attempts: list[dict[str, Any]]) -> Path:
    """Keep CUGA sandboxes outside the protected private repository."""
    if any(attempt["agent"] == "cuga" for attempt in attempts):
        return work.with_name(work.name + "-artifacts")
    return work / "artifacts"


def validate_cuga_inputs(inputs: dict[str, Any]) -> None:
    """Admit the persisted runtime and pinned staging recipe before restoration."""
    origin = inputs.get("cuga_runtime")
    if (not isinstance(origin, dict)
            or origin.get("schema") != "appworld_cuga_merlin_preparation_v1"
            or origin.get("persisted") is not True
            or origin.get("runtime_prepared") is not True
            or origin.get("runtime_persisted") is not True):
        raise ValueError("PERSISTED_CUGA_RUNTIME_RECEIPT_REQUIRED")
    launcher = inputs.get("cuga_prior_launcher")
    if (not isinstance(launcher, dict)
            or launcher.get("bytes") != CUGA_PRIOR_LAUNCHER_BYTES
            or launcher.get("sha256") != CUGA_PRIOR_LAUNCHER_SHA256):
        raise ValueError("PINNED_CUGA_PRIOR_LAUNCHER_REQUIRED")
    files = inputs.get("files")
    if not isinstance(files, dict) or not set(CUGA_SOURCE_FILES).issubset(files):
        raise ValueError("CUGA_DEVELOPMENT_SOURCE_SET_REQUIRED")


def validate_cuga_binding_qualification(inputs: dict[str, Any]) -> dict[str, Any]:
    """Read the successful CPU sandbox receipt used to pin the GPU binary."""
    receipt = inputs.get("cuga_binding_qualification")
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "appworld_cuga_merlin_binding_check_v1"
            or receipt.get("status") != "PASS" or receipt.get("exit_code") != 0
            or receipt.get("persisted") is not True
            or receipt.get("execution_backend") != "merlin"
            or type(receipt.get("allocated_gpus")) is not int
            or receipt["allocated_gpus"] != 0
            or not receipt.get("run_id") or not receipt.get("trial_id")
            or not receipt.get("hostname")
            or receipt["hostname"].split(".")[0].lower() == "mgmt2"
            or receipt.get("native_tasks_executed") != 0
            or receipt.get("model_generations") != 0
            or receipt.get("graph_loop_executed") is not False
            or receipt.get("formal_eligible") is not False):
        raise ValueError("SUCCESSFUL_MERLIN_CPU_CUGA_BINDING_QUALIFICATION_REQUIRED")
    sandbox = receipt.get("sandbox_probe", {})
    public_import = receipt.get("public_import", {})
    required_checks = (
        "private_repository_hidden", "procfs_absent", "work_directory", "public_modules_only",
    )
    if (not isinstance(sandbox, dict) or sandbox.get("exit_code") != 0
            or sandbox.get("sandbox_executed") is not True
            or not isinstance(public_import, dict) or public_import.get("status") != "PASS"
            or not isinstance(sandbox.get("isolation_checks"), dict)
            or any(sandbox["isolation_checks"].get(key) is not True for key in required_checks)):
        raise ValueError("CUGA_BINDING_REAL_SANDBOX_QUALIFICATION_REQUIRED")
    binary = receipt.get("bubblewrap", {})
    package = binary.get("package", {}) if isinstance(binary, dict) else {}
    if (not isinstance(binary, dict)
            or binary.get("status") != "VERIFIED"
            or not binary.get("executable") or not binary.get("resolved_executable")
            or type(binary.get("bytes")) is not int or binary["bytes"] <= 0
            or not isinstance(binary.get("sha256"), str)
            or len(binary["sha256"]) != 64
            or not binary.get("cli_version")
            or not isinstance(package, dict) or package.get("manager") != "dpkg"
            or package.get("name", "").split(":")[0] != "bubblewrap"
            or not package.get("version") or not package.get("architecture")
            or package.get("status") != "install ok installed"):
        raise ValueError("CUGA_BINDING_BUBBLEWRAP_IDENTITY_REQUIRED")
    command = sandbox.get("command", [])
    if (not isinstance(command, list) or not command
            or command[0] != binary["executable"]
            or not {"--unshare-all", "--share-net", "--cap-drop", "ALL"}.issubset(command)
            or sandbox.get("bubblewrap_sha256") != binary["sha256"]):
        raise ValueError("CUGA_BINDING_SANDBOX_BINARY_DIFFERS")
    return receipt


def prepare_cuga_bubblewrap(
    inputs: dict[str, Any], identity: dict[str, Any], environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
) -> dict[str, Any]:
    """Provision a missing system binary and compare GPU identity with CPU evidence."""
    run_id = environment.get("RH2_JOB_RUN_ID", "")
    trial_id = environment.get("ARNOLD_TRIAL_ID", "")
    checkpoint = environment.get("RH2_CHECKPOINT_HDFS", "").rstrip("/")
    gpu_text = environment.get("ARNOLD_APPLIED_GPU_NUM", "")
    hostname = socket.gethostname()
    if (not run_id or not trial_id or not gpu_text.isdecimal()
            or not checkpoint.startswith("hdfs://")
            or checkpoint.rsplit("/", 1)[-1] != run_id
            or "/" in run_id or "/" in trial_id or "SLURM_JOB_ID" in environment
            or hostname.split(".")[0].lower() == "mgmt2"
            or identity.get("execution_backend") != "merlin"
            or identity.get("run_id") != run_id or identity.get("trial_id") != trial_id
            or identity.get("hostname") != hostname):
        raise ValueError("CUGA_BUBBLEWRAP_REQUIRES_REAL_MERLIN_ALLOCATION")
    qualification = None
    if int(gpu_text) > 0 or inputs.get("cuga_binding_qualification") is not None:
        qualification = validate_cuga_binding_qualification(inputs)
    expected = qualification["bubblewrap"] if qualification is not None else None
    observed: dict[str, Any] = {
        "status": "PREPARING", "installed_by_this_run": False,
        "qualified_cpu_run_id": qualification["run_id"] if qualification else None,
        "qualified_cpu_trial_id": qualification["trial_id"] if qualification else None,
    }
    report["bubblewrap"] = observed
    executable = shutil.which("bwrap", path="/usr/bin:/bin")
    query = shutil.which("dpkg-query", path="/usr/bin:/bin")
    if query is None:
        raise FileNotFoundError("CUGA_BUBBLEWRAP_DPKG_QUERY_REQUIRED")
    if executable is None:
        apt = shutil.which("apt-get", path="/usr/bin:/bin")
        if apt is None:
            raise FileNotFoundError("CUGA_BUBBLEWRAP_APT_GET_REQUIRED")
        privilege: list[str] = []
        if os.geteuid() != 0:
            sudo = shutil.which("sudo", path="/usr/bin:/bin")
            if sudo is None:
                raise PermissionError("CUGA_BUBBLEWRAP_ROOT_OR_NONINTERACTIVE_SUDO_REQUIRED")
            privilege = [sudo, "-n"]
        package_spec = "bubblewrap"
        if expected is not None:
            package_spec = "bubblewrap=" + expected["package"]["version"]
        install_environment = dict(environment, DEBIAN_FRONTEND="noninteractive")
        run_command(
            "cuga_bubblewrap_package_index", [*privilege, apt, "update"], install_environment,
            work, output, store, report, [],
        )
        run_command(
            "cuga_bubblewrap_install",
            [*privilege, apt, "install", "--yes", "--no-install-recommends", package_spec],
            install_environment, work, output, store, report, [],
        )
        observed["installed_by_this_run"] = True
        observed["requested_package"] = package_spec
        executable = shutil.which("bwrap", path="/usr/bin:/bin")
        if executable is None:
            raise FileNotFoundError("CUGA_BUBBLEWRAP_MISSING_AFTER_INSTALL")
    binary = Path(executable)
    observed.update(executable=executable, resolved_executable=str(binary.resolve(strict=True)),
                    **file_identity(binary))
    run_command(
        "cuga_bubblewrap_version", [executable, "--version"], environment,
        work, output, store, report, [],
    )
    observed["cli_version"] = (output / "cuga_bubblewrap_version.log").read_text().strip()
    if not observed["cli_version"]:
        raise ValueError("CUGA_BUBBLEWRAP_VERSION_OUTPUT_EMPTY")
    run_command(
        "cuga_bubblewrap_package_identity",
        [query, "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\t${Status}\n",
         "bubblewrap"],
        environment, work, output, store, report, [],
    )
    package_output = (output / "cuga_bubblewrap_package_identity.log").read_text().strip()
    fields = package_output.split("\t")
    if (len(fields) != 4 or fields[0].split(":")[0] != "bubblewrap"
            or not fields[1] or not fields[2] or fields[3] != "install ok installed"):
        raise ValueError("CUGA_BUBBLEWRAP_INSTALLED_PACKAGE_IDENTITY_INVALID")
    observed["package"] = dict(
        manager="dpkg", name=fields[0], version=fields[1],
        architecture=fields[2], status=fields[3],
    )
    run_command(
        "cuga_bubblewrap_package_files", [query, "-L", "bubblewrap"], environment,
        work, output, store, report, [],
    )
    package_paths = (output / "cuga_bubblewrap_package_files.log").read_text().splitlines()
    if executable not in package_paths and observed["resolved_executable"] not in package_paths:
        raise ValueError("CUGA_BUBBLEWRAP_EXECUTABLE_NOT_IN_PACKAGE")
    keys = ("executable", "resolved_executable", "bytes", "sha256", "cli_version", "package")
    differences = [key for key in keys if expected is not None and observed[key] != expected[key]]
    observed.update(status="IDENTITY_MISMATCH" if differences else "VERIFIED",
                    identity_mismatches=differences)
    path = output / "bubblewrap_identity.json"
    write_json(path, observed)
    publish_closed(store, output, [path])
    if differences:
        raise ValueError("CUGA_BUBBLEWRAP_DIFFERS_FROM_CPU_QUALIFICATION: " + ", ".join(differences))
    return observed


def prepare_cuga_context(
    inputs: dict[str, Any], identity: dict[str, Any], environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
) -> dict[str, Path]:
    """Reuse the pinned ownership bootstrap, then restore and bind CUGA."""
    prepare_cuga_bubblewrap(inputs, identity, environment, work, output, store, report)
    source_path = str(work / "Libs")
    # The verified source tree is only available after prepare_inputs().
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, source_path)
        preparation = importlib.import_module("scripts.prepare_appworld_cuga_merlin")
    finally:
        sys.path[:] = previous_path
    if preparation.LEGACY_REPOSITORY.exists() or preparation.LEGACY_WORK.exists():
        raise FileExistsError("ORIGINAL_CUGA_RECIPE_PATH_ALREADY_EXISTS")
    launcher = work / "cuga_prior_launcher.sh"
    launcher.write_bytes(declared_content(inputs["cuga_prior_launcher"]))
    staging = work / "cuga_runtime_staging.sh"
    staging.write_bytes(preparation.staging_prefix(launcher))
    run_command(
        "cuga_runtime_bootstrap", ["bash", str(staging)], environment,
        work, output, store, report, [],
    )
    runtime_work = work / "cuga_restore"
    runtime_output = output / "cuga_runtime"
    runtime_work.mkdir()
    runtime_output.mkdir()
    runtime_store = HdfsStore(store.prefix + "/cuga_runtime", runtime_work, runtime_output)
    runtime_report: dict[str, Any] = {}
    report["cuga_runtime"] = runtime_report
    report["stage"] = "restore_cuga_runtime"
    environment_path = runtime_output / "environment.json"
    preparation.restore_prepared_runtime(
        inputs["cuga_runtime"],
        {**identity, "compatibility_execution_id": f"merlin_{identity['trial_id']}_cuga"},
        work=runtime_work, output=runtime_output, store=runtime_store,
        report=runtime_report, environment_path=environment_path,
    )
    public_root = work.with_name(work.name + "-public")
    public_root.mkdir()
    public_lib = public_root / "Libs"
    public_records = {}
    for relative in CUGA_PUBLIC_FILES:
        source = work / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError("CUGA_PUBLIC_SOURCE_MUST_BE_REGULAR_FILE: " + relative)
        record = inputs["files"][relative]
        verify_bytes(source, record)
        destination = public_lib / Path(relative).relative_to("Libs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        public_records[relative] = verify_bytes(destination, record)
    runtime_report.update(
        prior_launcher={key: inputs["cuga_prior_launcher"][key] for key in ("bytes", "sha256")},
        public_application_lib=str(public_lib),
        public_source_files=public_records,
    )
    # The helper's stage uploads precede the newly rebound environment file.
    publish_closed(store, output, [runtime_output])
    return {
        "cuga_environment": environment_path,
        "python_base_prefix": Path(runtime_report["runtime_identity"]["base_prefix"]),
        "public_application_lib": public_lib,
        "repository_root": work,
    }


def cuga_batch_attempt_score(
    row: dict[str, Any], output: Path, report: dict[str, Any],
) -> dict[str, Any]:
    """Read the existing CUGA pipeline contract without promoting cost coverage."""
    directory = output / row["output_directory"]
    records = {}
    for name in ("pipeline", "execution_receipt", "development_score"):
        value = json.loads((directory / (name + ".json")).read_text())
        if not isinstance(value, dict):
            raise ValueError("CUGA_ATTEMPT_RECEIPT_OBJECT_REQUIRED")
        records[name] = value
    pipeline = records["pipeline"]
    execution = records["execution_receipt"]
    score = records["development_score"]
    expected = {key: row[key] for key in ("task_id", "split", "seed")}
    expected["method"] = CUGA_METHOD
    if (
        pipeline.get("schema") != "appworld_cuga_development_pipeline_v1"
        or execution.get("schema") != "appworld_development_execution_v1"
        or score.get("schema") != "appworld_development_score_v1"
        or any(record.get(key) != value
               for record in (pipeline, execution, score) for key, value in expected.items())
        or any(record.get("formal_eligible") is not False
               for record in (pipeline, execution, score))
        or execution.get("run_root") != str(directory / "native_run")
    ):
        raise ValueError("DEVELOPMENT_ATTEMPT_IDENTITY_DIFFERS")
    if (
        pipeline.get("status") != "SCORED_DEVELOPMENT_ATTEMPT"
        or score.get("scoring_status") != "completed"
        or type(score.get("native_tgc")) is not bool
        or pipeline.get("native_tgc") is not score.get("native_tgc")
        or execution.get("terminal") is not True
        or execution.get("rpc_outcome_unresolved") is not False
        or execution.get("status") not in {"completed", "agent_failure", "infrastructure_failure"}
    ):
        raise ValueError("COMPLETED_SCORE_REQUIRED")
    stage = report.get("stages", {}).get(row["stage"], {})
    server = pipeline.get("model_server", {})
    scorer = pipeline.get("processes", {}).get("scorer", {})
    owner = pipeline.get("cuga_owner", {})
    worker = owner.get("worker", {})
    relay = owner.get("relay", {})
    cleanup = relay.get("native_cleanup", {})
    if (
        stage.get("exit_code") != 0 or stage.get("outputs_closed") is not True
        or stage.get("cleanup", {}).get("owner_terminated") is not True
        or server.get("status") != "STOPPED"
        or server.get("server_cleanup", {}).get("server_parent_reaped") is not True
        or scorer.get("exit_code") != 0
        or scorer.get("cleanup", {}).get("parent_reaped") is not True
        or pipeline.get("native_client", {}).get("status") != "closed"
        or owner.get("status") not in {"completed", "worker_failed"}
        or worker.get("reaped") is not True or type(worker.get("exit_code")) is not int
        or worker.get("termination_signals") != []
        or relay.get("status") != "closed"
        or owner.get("rpc_outcome_unresolved") is not False
        or relay.get("rpc_outcome_unresolved") is not False
        or relay.get("cleanup_error") is not None
        or cleanup.get("child_reaped") is not True
        or type(cleanup.get("exit_code")) is not int or cleanup["exit_code"] != 0
        or cleanup.get("termination_signals") != []
        or cleanup.get("request_outcome_unresolved") is not False
        or execution.get("native_cleanup") != cleanup
    ):
        raise ValueError("DEVELOPMENT_PROCESS_CLOSURE_UNCONFIRMED")
    names = [str((directory / (name + ".json")).relative_to(output)) for name in records]
    persisted = report.get("decisive_artifacts", {})
    if any(name not in persisted for name in names):
        raise ValueError("DEVELOPMENT_CLOSED_OUTPUTS_NOT_PERSISTED")
    return {
        "native_tgc": score["native_tgc"],
        "execution_status": execution["status"],
        "usage_complete": execution.get("usage_complete"),
        "lm_call_count": execution.get("lm_call_count"),
        "cuga_usage_transport_coverage_qualified": execution.get("cuga_usage_transport_coverage_qualified"),
        "cuga_model_usage": execution.get("cuga_model_usage"),
        "development_score": str((directory / "development_score.json").relative_to(output)),
        "closed_outputs_persisted": True,
    }


def cuga_decisive_artifact_names(store: HdfsStore) -> tuple[str, ...]:
    names = []
    for root in getattr(store, "native_session_roots", ()):
        directory = Path(root).parent.as_posix()
        names.extend((
            directory + "/pipeline.json",
            directory + "/execution_receipt.json",
            directory + "/development_score.json",
            root + "/native_session.json",
            directory + "/cuga_process/process_observation.json",
        ))
    return tuple(names)


def resolve_development_attempt(inputs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a default ReAct, explicit official, or uniform development attempt."""
    if "development_attempt" not in inputs:
        attempt = {"agent": "official_react_code", "split": "train",
                   "eligible_index": 0, "seed": 41}
    else:
        declared = inputs["development_attempt"]
        if not isinstance(declared, dict):
            raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_DECLARATION_OBJECT")
        agent = declared.get("agent")
        if agent in ("official_react_code", "official_full_code", "official_function_calling"):
            fields = {"agent", "split", "eligible_index", "seed"}
            if set(declared) != fields:
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_EXACT_OFFICIAL_FIELDS")
            if declared["split"] not in ("train", "dev"):
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_OFFICIAL_TRAIN_OR_DEV")
        elif agent == "cuga":
            if set(declared) != {"agent", "split", "eligible_index", "seed"}:
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_EXACT_CUGA_FIELDS")
            if declared["split"] not in ("train", "dev"):
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_CUGA_TRAIN_OR_DEV")
        elif agent == "collaboration_uniform":
            fields = {"agent", "split", "eligible_index", "seed",
                      "verification_budget", "max_model_calls"}
            if set(declared) != fields:
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_EXACT_UNIFORM_FIELDS")
            if declared["split"] != "train":
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_UNIFORM_TRAIN")
        else:
            raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_DECLARED_AGENT")
        if type(declared["seed"]) is not int or declared["seed"] not in EXPERIMENT_SEEDS:
            raise ValueError("DEVELOPMENT_ATTEMPT_SEED_MUST_BE_41_42_OR_43")
        integer_fields = (
            ("eligible_index", "verification_budget")
            if agent == "collaboration_uniform" else ("eligible_index",)
        )
        for name in integer_fields:
            if type(declared[name]) is not int or declared[name] < 0:
                raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_NONNEGATIVE_INTEGER: " + name)
        if agent == "collaboration_uniform" and (
            type(declared["max_model_calls"]) is not int or declared["max_model_calls"] <= 0
        ):
            raise ValueError("DEVELOPMENT_ATTEMPT_REQUIRES_POSITIVE_INTEGER: max_model_calls")
        attempt = dict(declared)
    attempt["method"] = (
        f"development_full_history_uniform_verification_budget_{attempt['verification_budget']}"
        if attempt["agent"] == "collaboration_uniform"
        else CUGA_METHOD if attempt["agent"] == "cuga" else attempt["agent"]
    )
    attempt["neuronsim_method"] = False
    return attempt


def resolve_development_attempts(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate an ordered declaration before any restoration or task execution."""
    if "development_attempts" not in inputs:
        return [resolve_development_attempt(inputs)]
    if "development_attempt" in inputs:
        raise ValueError("DEVELOPMENT_ATTEMPTS_SINGLE_AND_BATCH_ARE_EXCLUSIVE")
    declared = inputs["development_attempts"]
    if not isinstance(declared, list) or not declared:
        raise ValueError("DEVELOPMENT_ATTEMPTS_REQUIRES_NONEMPTY_LIST")
    attempts = [resolve_development_attempt({"development_attempt": item}) for item in declared]
    identities = [(item["agent"], item["split"], item["eligible_index"], item["seed"])
                  for item in attempts]
    if len(set(identities)) != len(identities):
        raise ValueError("DEVELOPMENT_ATTEMPTS_DUPLICATE_LOGICAL_ATTEMPT")
    return attempts


def validate_inputs(inputs: dict[str, Any]) -> None:
    attempts = resolve_development_attempts(inputs)
    if any(attempt["agent"] == "cuga" for attempt in attempts):
        validate_cuga_inputs(inputs)
    for name in ("native_runtime", "data_admission", "model_runtime"):
        receipt = inputs[name]
        if receipt.get("exit_code") != 0 or receipt.get("persisted") is not True:
            raise ValueError("COMPLETED_PERSISTED_INPUT_REQUIRED: " + name)
    native = inputs["native_runtime"]["environment"]
    data = inputs["data_admission"]["environment"]
    model = inputs["model_runtime"]["environment"]
    if (native["native_python"] != str(NATIVE_PREFIX / "native-env/bin/python")
            or native["source_checkout"] != str(NATIVE_PREFIX / "source")
            or data["native_python"] != native["native_python"]
            or data["source_checkout"] != native["source_checkout"]
            or data["data_root"] != str(DATA_PREFIX / "benchmark")
            or data["appworld_cache"] != str(DATA_PREFIX / "cache")
            or model["python"] != str(MODEL_PREFIX / "model-env/bin/python")
            or model["status"] != "MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED"):
        raise ValueError("ORIGINAL_RUNTIME_PREFIX_OR_STATUS_DIFFERS")
    intake = inputs["model_inputs"]
    if (intake.get("status") != "MODEL_SNAPSHOT_VERIFIED_GPU_ADMISSION_REQUIRED"
            or intake.get("exit_code") != 0 or intake.get("persisted") is not True
            or intake.get("full_snapshot_verified") is not True
            or intake.get("model_revision") != inputs["source_metadata"]["sha"]
            or intake.get("runtime_receipt") != inputs["model_runtime"]
            or not isinstance(intake.get("files"), dict) or not intake["files"]):
        raise ValueError("COMPLETE_ADMITTED_MODEL_INPUTS_REQUIRED")
    model_download_uris(intake)
    if "model_qualification" in inputs:
        validate_qualification_reference(inputs["model_qualification"])


def require_scratch_space(work: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(work)
    if usage.free < MINIMUM_SCRATCH_BYTES:
        raise ValueError(
            "INSUFFICIENT_SCRATCH_SPACE: required_bytes="
            + str(MINIMUM_SCRATCH_BYTES) + "; free_bytes=" + str(usage.free)
        )
    return {
        "status": "PASS", "path": str(work),
        "required_free_bytes": MINIMUM_SCRATCH_BYTES,
        "free_bytes": usage.free, "total_bytes": usage.total,
    }


def prepare_inputs(inputs: dict[str, Any], work: Path, output: Path, store: HdfsStore,
                   report: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    validate_inputs(inputs)
    report["stage"] = "scratch_admission"
    write_json(output / "scratch_admission.json", require_scratch_space(work))
    publish_closed(store, output, [output / "scratch_admission.json"])
    for name, row in inputs["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "Libs":
            raise ValueError("UNSAFE_DECLARED_SOURCE_PATH: " + name)
        target = work / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(declared_content(row))
    (work / "source_archive_manifest.json").write_bytes(declared_content(inputs["archive_manifest"]))
    write_json(work / "source_metadata.json", inputs["source_metadata"])
    write_json(work / "native_preparation.receipt.json", inputs["data_admission"])
    write_json(work / "model_package_environment.json", inputs["model_runtime"]["environment"])
    native = inputs["native_runtime"]
    model = inputs["model_runtime"]
    data = inputs["data_admission"]
    specifications = (
        ("restore_native_runtime", native["runtime_bundle_uri"],
         {"bytes": native["runtime_bundle_bytes"], "sha256": native["runtime_bundle_sha256"]},
         (NATIVE_PREFIX,), (NATIVE_PREFIX, Path("/opt/python"))),
        ("restore_data", data["data_hdfs_uri"],
         {"bytes": data["data_bundle_bytes"], "sha256": data["data_bundle_sha256"]},
         (DATA_PREFIX / "benchmark/data", DATA_PREFIX / "cache"),
         (DATA_PREFIX / "benchmark/data", DATA_PREFIX / "cache")),
        ("restore_model_runtime", model["runtime_bundle_uri"],
         {"bytes": model["runtime_bundle_bytes"], "sha256": model["runtime_bundle_sha256"]},
         (MODEL_PREFIX,), (MODEL_PREFIX, Path("/opt/python"),
                           Path("/opt/environments/neuronsim-mechanism-learning"))),
    )
    for stage, uri, identity, roots, link_roots in specifications:
        report["stage"] = stage
        started = time.time()
        restore_archive(store, work, stage, uri, identity, roots, link_roots)
        path = output / "stage_receipts" / (stage + ".json")
        write_json(path, {
            "stage": stage, "status": "VERIFIED_AND_RESTORED", "uri": uri, **identity,
            "started_at": started, "finished_at": time.time(),
            "restored_roots": [str(root) for root in roots],
        })
        publish_closed(store, output, [path])
    split = inputs["split_manifest"]
    store.hdfs("-get", split["uri"], str(work / "split_manifest.json"))
    verify_bytes(work / "split_manifest.json", split)
    environment = dict(
        os.environ, PYTHONPATH=str(work / "Libs"), PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(work / "tmp"), APPWORLD_CACHE=data["environment"]["appworld_cache"],
    )
    Path(environment["TMPDIR"]).mkdir()
    native_python = native["environment"]["native_python"]
    model_python = model["environment"]["python"]
    run_command(
        "bind_configuration", [model_python, "-c", PREPARE_CONFIG, str(work), str(output)],
        environment, work, output, store, report, [output / "bound_config.yaml"],
    )
    model_directory = work / "model"
    model_directory.mkdir()
    entries = json.loads((work / "model_entries.json").read_text())
    report["stage"] = "restore_model_snapshot"
    locations = model_download_uris(inputs["model_inputs"])
    for name in sorted(entries):
        target = model_directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        row = inputs["model_inputs"]["files"][name]
        uri = locations[name]
        store.hdfs("-get", uri, str(target))
        verify_bytes(target, row)
        report["restored_model_files"] = report.get("restored_model_files", 0) + 1
    run_command(
        "verify_model_snapshot", [model_python, "-c", VERIFY_SNAPSHOT, str(work), str(output)],
        environment, work, output, store, report, [output / "model_snapshot.json"],
    )
    write_json(output / "input_admission.json", {
        "status": "RESTORED_INPUT_IDENTITIES_VERIFIED",
        "source_sha256": {name: row["sha256"] for name, row in inputs["files"].items()},
        "source_archive_manifest_sha256": inputs["archive_manifest"]["sha256"],
        "split_manifest": {key: split[key] for key in ("uri", "bytes", "sha256")},
        "native_runtime": {"uri": native["runtime_bundle_uri"],
                           "sha256": native["runtime_bundle_sha256"]},
        "model_runtime": {"uri": model["runtime_bundle_uri"],
                          "sha256": model["runtime_bundle_sha256"]},
        "data_archive": {"uri": data["data_hdfs_uri"], "sha256": data["data_bundle_sha256"]},
        "model_input_run_id": inputs["model_inputs"]["run_id"],
        "model_input_trial_id": inputs["model_inputs"]["trial_id"],
        "model_revision": inputs["source_metadata"]["sha"],
        "formal_task_contents_inspected": False,
    })
    publish_closed(store, output, [output / "input_admission.json"])
    return native_python, model_python, environment


def reuse_model_qualification(
    reference: dict[str, Any], *, model_python: str, environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
) -> None:
    """Admit the original neutral receipt without changing its bytes or provenance."""
    validate_qualification_reference(reference)
    report["stage"] = "qualification_reuse_download"
    qualification = output / "qualification"
    qualification.mkdir()
    downloaded = work / "model_qualification.download.json"
    store.hdfs("-get", reference["uri"], str(downloaded))
    verify_bytes(downloaded, reference)
    receipt_path = qualification / "qualification.json"
    downloaded.replace(receipt_path)
    run_command(
        "qualification_reuse_admission",
        [model_python, "-c", ADMIT_QUALIFICATION,
         str(output / "bound_config.yaml"), str(receipt_path)],
        environment, work, output, store, report, [qualification],
    )
    report["qualification_reuse"] = {
        "reference": dict(reference), "neutral_probe_repeated": False,
    }


def prepare_development_context(
    native_python: str, model_python: str, environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any], *,
    qualification_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = output / "bound_config.yaml"
    native_environment = output / "bound_native_environment.json"
    split_manifest = work / "split_manifest.json"
    native_command = [
        native_python, "-m", "population_dynamics.appworld_benchmark_native_binding",
        "--config", str(config), "--native-environment", str(work / "native_preparation.receipt.json"),
        "--split-manifest", str(split_manifest), "--output", str(native_environment),
    ]
    run_command(
        "native_binding", native_command, dict(environment, CUDA_VISIBLE_DEVICES=""),
        work, output, store, report, [native_environment],
    )
    bound = json.loads(native_environment.read_text())
    if bound["status"] != "PREPARED_RUNTIME_AND_SPLIT_METADATA":
        raise ValueError("NATIVE_BINDING_NOT_PREPARED")
    environment = dict(environment, APPWORLD_CACHE=bound["appworld_cache"])
    common = [
        "--config", str(config), "--model-directory", str(work / "model"),
        "--source-metadata", str(work / "source_metadata.json"),
        "--snapshot-config", str(work / ACTOR_CONFIG),
        "--runtime-receipt", str(work / "model_package_environment.json"),
    ]
    qualification = output / "qualification"
    if qualification_reference is None:
        run_command(
            "model_qualification",
            [model_python, "-m", "population_dynamics.appworld_benchmark_model_runtime",
             *common, "--output", str(qualification)],
            environment, work, output, store, report, [qualification],
        )
    else:
        reuse_model_qualification(
            qualification_reference, model_python=model_python, environment=environment,
            work=work, output=output, store=store, report=report,
        )
    qualified = json.loads((qualification / "qualification.json").read_text())
    if qualified["status"] != "PASS":
        raise ValueError("MODEL_QUALIFICATION_NOT_PASS")
    report["model_qualification_status"] = qualified["status"]
    return {
        "common": common, "environment": environment, "native_environment": native_environment,
        "split_manifest": split_manifest, "qualification": qualification,
    }


def select_attempt(
    attempt: dict[str, Any], *, model_python: str, environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any],
    directory: Path, stage: str,
) -> dict[str, Any]:
    command = [model_python, "-c", SELECT_TASK, str(work),
               attempt["split"], str(attempt["eligible_index"]), str(attempt["seed"])]
    if output != work / "artifacts":
        command.extend([str(directory), str(output)])
    elif directory != output:
        command.append(str(directory))
    run_command(stage, command, environment, work, output, store, report,
                [directory / "selection.json", directory / "selected_task_id.txt"])
    selection = json.loads((directory / "selection.json").read_text())
    if selection["task_id"] != (directory / "selected_task_id.txt").read_text().strip():
        raise ValueError("DEVELOPMENT_SELECTION_TASK_ID_DIFFERS")
    return selection


def run_selected_attempt(
    attempt: dict[str, Any], task_id: str, *, context: dict[str, Any],
    model_python: str, work: Path, output: Path, store: HdfsStore,
    report: dict[str, Any], directory: Path, stage: str,
) -> dict[str, Any]:
    is_cuga = attempt["agent"] == "cuga"
    module = (
        "population_dynamics.appworld_benchmark_cuga_development"
        if is_cuga else "population_dynamics.appworld_benchmark_pipeline"
    )
    pipeline_command = [
        model_python, "-m", module, *context["common"],
        "--native-environment", str(context["native_environment"]),
        "--split-manifest", str(context["split_manifest"]),
        "--model-qualification", str(context["qualification"] / "qualification.json"),
        "--output", str(directory), "--split", attempt["split"],
        "--task-id", task_id, "--seed", str(attempt["seed"]),
    ]
    if is_cuga:
        binding = context.get("cuga")
        if not isinstance(binding, dict):
            raise ValueError("CUGA_RUNTIME_BINDING_REQUIRED")
        for name in ("cuga_environment", "python_base_prefix", "public_application_lib", "repository_root"):
            pipeline_command += ["--" + name.replace("_", "-"), str(binding[name])]
    else:
        pipeline_command += ["--agent", attempt["agent"]]
        if attempt["agent"] == "collaboration_uniform":
            pipeline_command += [
                "--verification-budget", str(attempt["verification_budget"]),
                "--max-model-calls", str(attempt["max_model_calls"]),
            ]
    directory.parent.mkdir(parents=True, exist_ok=True)
    run_command(stage, pipeline_command, context["environment"],
                work, output, store, report, [directory])
    pipeline = json.loads((directory / "pipeline.json").read_text())
    if pipeline["status"] != "SCORED_DEVELOPMENT_ATTEMPT":
        raise ValueError("DEVELOPMENT_PIPELINE_NOT_SCORED")
    score_directory = directory if is_cuga else directory / "attempt"
    if not (score_directory / "development_score.json").is_file():
        raise ValueError("INDEPENDENT_DEVELOPMENT_SCORE_MISSING")
    return pipeline


def run_development(native_python: str, model_python: str, environment: dict[str, str],
                    work: Path, output: Path, store: HdfsStore,
                    report: dict[str, Any], *, qualification_reference: dict[str, Any] | None = None,
                    development_attempt: dict[str, Any] | None = None,
                    cuga_context: dict[str, Path] | None = None) -> None:
    attempt = resolve_development_attempt(
        {} if development_attempt is None else {"development_attempt": development_attempt},
    )
    if attempt["agent"] == "cuga":
        store.native_attempt_roots = ()
        store.native_session_roots = ("development/native_run",)
    report.update(attempt)
    context = prepare_development_context(
        native_python, model_python, environment, work, output, store, report,
        qualification_reference=qualification_reference,
    )
    if cuga_context is not None:
        context["cuga"] = cuga_context
    selection = select_attempt(
        attempt, model_python=model_python, environment=context["environment"],
        work=work, output=output, store=store, report=report,
        directory=output, stage="select_training_task",
    )
    report["task_id"] = selection["task_id"]
    run_selected_attempt(
        attempt, selection["task_id"], context=context, model_python=model_python,
        work=work, output=output, store=store, report=report,
        directory=output / "development", stage="development_pipeline",
    )
    report.update(
        status="SCORED_DEVELOPMENT_ATTEMPT", exit_code=0, development_attempts=1,
        development_score=("development/development_score.json" if attempt["agent"] == "cuga"
                           else "development/attempt/development_score.json"),
    )


def batch_attempt_score(
    row: dict[str, Any], output: Path, report: dict[str, Any],
) -> dict[str, Any]:
    """Admit a score only after its isolated processes and publication have closed."""
    if row["agent"] == "cuga":
        return cuga_batch_attempt_score(row, output, report)
    directory = output / row["output_directory"]
    pipeline = json.loads((directory / "pipeline.json").read_text())
    execution = json.loads((directory / "attempt/execution_receipt.json").read_text())
    score = json.loads((directory / "attempt/development_score.json").read_text())
    for record in (execution, score):
        if any(record.get(key) != row[key] for key in ("task_id", "split", "seed", "method")):
            raise ValueError("DEVELOPMENT_BATCH_ATTEMPT_IDENTITY_DIFFERS")
    if (score.get("schema") != "appworld_development_score_v1"
            or score.get("scoring_status") != "completed"
            or type(score.get("native_tgc")) is not bool
            or pipeline.get("status") != "SCORED_DEVELOPMENT_ATTEMPT"
            or pipeline.get("native_tgc") is not score["native_tgc"]):
        raise ValueError("DEVELOPMENT_BATCH_COMPLETED_SCORE_REQUIRED")
    processes = pipeline.get("processes", {})
    server = pipeline.get("model_server", {})
    if (processes.get("scorer", {}).get("exit_code") != 0
            or any(processes.get(name, {}).get("cleanup", {}).get("parent_reaped") is not True
                   for name in ("actor", "scorer"))
            or server.get("status") != "STOPPED"
            or server.get("server_cleanup", {}).get("server_parent_reaped") is not True):
        raise ValueError("DEVELOPMENT_BATCH_PROCESS_CLOSURE_UNCONFIRMED")
    stage = report.get("stages", {}).get(row["stage"], {})
    if (stage.get("outputs_closed") is not True
            or stage.get("cleanup", {}).get("owner_terminated") is not True
            or stage.get("cleanup", {}).get("reaped") is not True
            or stage.get("publication_complete") is not True):
        raise ValueError("DEVELOPMENT_BATCH_PUBLICATION_OR_OWNER_UNCONFIRMED")
    for name in ("pipeline.json", "attempt/execution_receipt.json", "attempt/development_score.json"):
        relative = row["output_directory"] + "/" + name
        reference = report.get("decisive_artifacts", {}).get(relative)
        if reference is None:
            raise ValueError("DEVELOPMENT_BATCH_DECISIVE_ARTIFACT_NOT_PERSISTED: " + relative)
        verify_bytes(output / relative, reference)
    return {
        "native_tgc": score["native_tgc"],
        "lm_call_count": execution.get("lm_call_count"),
        "usage_complete": execution.get("usage_complete"),
        "closed_outputs_persisted": True,
    }


def run_development_batch(
    native_python: str, model_python: str, environment: dict[str, str],
    work: Path, output: Path, store: HdfsStore, report: dict[str, Any], *,
    development_attempts: list[dict[str, Any]],
    qualification_reference: dict[str, Any] | None = None,
    cuga_context: dict[str, Path] | None = None,
) -> None:
    """Restore once, then own and persist each declared native attempt in order.

    Each pipeline call owns its model server and fresh native world. Any unresolved
    attempt stops the batch. Cross-allocation recovery must reconcile this ledger;
    this entry point never retries or drops a declared attempt.
    """
    attempts = resolve_development_attempts({"development_attempts": development_attempts})
    rows = []
    for index, attempt in enumerate(attempts):
        relative = (f"development/{attempt['agent']}/{attempt['split']}/"
                    f"seed_{attempt['seed']}/eligible_{attempt['eligible_index']}")
        rows.append({
            **attempt, "output_directory": relative, "stage": f"development_attempt_{index:05d}",
            "state": "NOT_STARTED", "task_id": None, "native_tgc": None,
            "closed_outputs_persisted": False,
        })
    store.native_attempt_roots = tuple(
        row["output_directory"] + "/attempt" for row in rows if row["agent"] != "cuga"
    )
    store.native_session_roots = tuple(
        row["output_directory"] + "/native_run" for row in rows if row["agent"] == "cuga"
    )
    report.update(
        development_scope="declared_development_batch_and_diagnostics",
        development_attempts_planned=len(rows), development_attempts=0,
        development_plan=rows, model_server_reuse=False, automatic_attempt_retries=False,
    )
    ledger = output / "development_batch.json"

    def persist_ledger() -> None:
        report["development_attempts"] = sum(row["state"] in ("SCORED_SUCCESS", "SCORED_NEGATIVE")
                                             for row in rows)
        report["development_coverage"] = {
            "planned": len(rows),
            **{state.lower(): sum(row["state"] == state for row in rows)
               for state in ("NOT_STARTED", "RUNNING", "SCORED_SUCCESS", "SCORED_NEGATIVE", "UNKNOWN")},
        }
        write_json(ledger, {
            "schema": "appworld_development_batch_v1", "run_id": report["run_id"],
            "trial_id": report["trial_id"], "formal_eligible": False,
            "coverage": report["development_coverage"], "attempts": rows,
        })
        # This file is the mutable progress view. Attempt receipts remain immutable.
        report.setdefault("decisive_artifacts", {})["development_batch.json"] = store.publish(
            ledger, "stages/development_batch.json", replace=True,
        )

    persist_ledger()
    context = prepare_development_context(
        native_python, model_python, environment, work, output, store, report,
        qualification_reference=qualification_reference,
    )
    if cuga_context is not None:
        context["cuga"] = cuga_context
    # Resolve the full roster using native membership and configured demo exclusions.
    # No actor starts until all selections have passed and the roster is durable.
    for index, (attempt, row) in enumerate(zip(attempts, rows)):
        selection = select_attempt(
            attempt, model_python=model_python, environment=context["environment"],
            work=work, output=output, store=store, report=report,
            directory=output / "selections" / str(index), stage=f"select_development_{index:05d}",
        )
        row.update(task_id=selection["task_id"], selection=selection)
    persist_ledger()
    for attempt, row in zip(attempts, rows):
        row["state"] = "RUNNING"
        persist_ledger()
        try:
            run_selected_attempt(
                attempt, row["task_id"], context=context, model_python=model_python,
                work=work, output=output, store=store, report=report,
                directory=output / row["output_directory"], stage=row["stage"],
            )
            row.update(batch_attempt_score(row, output, report))
            row["state"] = "SCORED_SUCCESS" if row["native_tgc"] else "SCORED_NEGATIVE"
            persist_ledger()
        except BaseException as error:
            row.update(state="UNKNOWN", error_type=type(error).__name__, error=str(error))
            # Keep the original error if the publication service is also unavailable.
            try:
                persist_ledger()
            except Exception as publication_error:
                row["ledger_publication_error"] = str(publication_error)
            raise
    report.update(status="SCORED_DEVELOPMENT_BATCH", exit_code=0)


def publish_final(store: HdfsStore, work: Path, output: Path,
                  report: dict[str, Any]) -> None:
    excluded = tuple(Path(name) for name in report.get("unclosed_output_paths", []))
    decisive = report.setdefault("decisive_artifacts", {})
    for name in decisive_artifact_names(store):
        path = output / name
        if any(Path(name).is_relative_to(parent) for parent in excluded):
            continue
        if path.is_file() and name not in decisive:
            decisive[name] = store.publish(path, "stages/" + name)
    files = {}
    input_links = {}
    for path in sorted(output.rglob("*")):
        if any(path.relative_to(output).is_relative_to(parent) for parent in excluded):
            continue
        if path.is_symlink():
            binding = admitted_native_data_link(store, output, path)
            if binding is not None:
                input_links[path.relative_to(output).as_posix()] = binding
                continue
            raise ValueError("OUTPUT_BUNDLE_CONTAINS_SYMLINK: " + str(path))
        if path.is_file():
            files[path.relative_to(output).as_posix()] = file_identity(path)
    manifest = {
        "schema": "appworld_gpu_development_artifacts_v1",
        "run_id": report["run_id"], "trial_id": report["trial_id"],
        "scope": report.get("development_scope", "one_train_development_attempt_and_diagnostics"),
        "input_bundles_included": False,
        "input_links_excluded": input_links,
        "unclosed_outputs_excluded": [path.as_posix() for path in excluded],
        "files": files,
    }
    manifest_path = work / "output_manifest.json"
    write_json(manifest_path, manifest)
    bundle = work / "outputs.tar.gz"
    with tarfile.open(bundle, "w:gz") as handle:
        for name in files:
            handle.add(output / name, arcname="artifacts/" + name, recursive=False)
        handle.add(manifest_path, arcname="output_manifest.json", recursive=False)
    report["output_bundle"] = store.publish(bundle, "outputs.tar.gz")
    report["artifact_manifest"] = store.publish(manifest_path, "output_manifest.json")
    report["persisted"] = True
    receipt = work / "receipt.json"
    write_json(receipt, report)
    # The receipt is the final remote write and binds the complete output bundle.
    store.publish(receipt, "receipt.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = require_merlin_gpu(dict(os.environ))
    work = Path("/tmp") / ("neuronsim-appworld-gpu-development-" + identity["run_id"])
    work.mkdir(exist_ok=False)
    inputs = json.loads(args.inputs.read_text())
    attempts = resolve_development_attempts(inputs)
    output = development_output_root(work, attempts)
    output.mkdir()
    write_json(work / "inputs.json", inputs)
    report: dict[str, Any] = {
        "schema": "appworld_gpu_development_merlin_v1", **identity,
        "status": "RUNNING", "stage": "validate_inputs", "exit_code": None,
        "started_at": time.time(), "persisted": False,
        "authorized_training_seeds": list(EXPERIMENT_SEEDS), "neuronsim_method": False,
        "training_performed": False, "formal_evaluation_released": False,
        "formal_eligible": False, "development_attempts": 0,
        "resumable_task_checkpoint_created": False,
    }
    store = HdfsStore(identity["checkpoint_path"], work, output)
    monitor = GpuMonitor(store, output, report)

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError("MERLIN_JOB_SIGNAL: " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        attempts = resolve_development_attempts(inputs)
        if "development_attempts" not in inputs:
            report.update(attempts[0])
        report["development_attempts_planned"] = len(attempts)
        monitor.start()
        native_python, model_python, environment = prepare_inputs(
            inputs, work, output, store, report,
        )
        cuga_context = None
        if any(attempt["agent"] == "cuga" for attempt in attempts):
            cuga_context = prepare_cuga_context(
                inputs, identity, environment, work, output, store, report,
            )
        if "development_attempts" in inputs:
            run_development_batch(
                native_python, model_python, environment, work, output, store, report,
                qualification_reference=inputs.get("model_qualification"),
                development_attempts=inputs["development_attempts"],
                cuga_context=cuga_context,
            )
        else:
            run_development(
                native_python, model_python, environment, work, output, store, report,
                qualification_reference=inputs.get("model_qualification"),
                development_attempt=inputs.get("development_attempt"),
                cuga_context=cuga_context,
            )
    except (Exception, KeyboardInterrupt) as error:
        report.update(
            status="FAILED", exit_code=1, error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(),
        )
    finally:
        monitor.stop()
        report["finished_at"] = time.time()
        report["elapsed_seconds"] = report["finished_at"] - report["started_at"]
        sample_times = [row["sampled_at"] for row in monitor.snapshots]
        persisted_times = [row["persisted_at"] for row in monitor.snapshots
                           if row["persisted_at"] is not None]
        sample_boundaries = [report["started_at"], *sample_times, report["finished_at"]]
        persistence_boundaries = [
            report["started_at"], *persisted_times, report["finished_at"],
        ]
        report["monitoring"] = {
            "sample_interval_seconds": GPU_SAMPLE_SECONDS,
            "snapshot_kind": "gpu_progress_and_log_diagnostics_only",
            "snapshots": monitor.snapshots, "errors": monitor.errors,
            "maximum_sample_gap_seconds": max(
                (b - a for a, b in zip(sample_boundaries, sample_boundaries[1:])), default=0),
            "maximum_persistence_gap_seconds": max(
                (b - a for a, b in zip(persistence_boundaries, persistence_boundaries[1:])), default=0),
        }
        monitoring = report["monitoring"]
        monitoring["status"] = (
            "PASS" if not monitor.errors and sample_times and persisted_times
            and monitoring["maximum_sample_gap_seconds"] <= 900
            and monitoring["maximum_persistence_gap_seconds"] <= 1800 else "DEGRADED"
        )
        write_json(output / "execution_summary.json", report)
        try:
            publish_final(store, work, output, report)
        except Exception as error:
            report.update(
                persisted=False, publication_error_type=type(error).__name__,
                publication_error=str(error), exit_code=1,
            )
            write_json(work / "receipt.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

"""Checkpoint a completed CUGA package runtime in its current Merlin CPU job.

The original preparation parent is already held by the authorized recovery.
This command verifies the completed child and the real runtime, persists the
runtime plus its actual Python base, and resumes that exact parent in finally.
It does not qualify CUGA native execution or change the source environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts.prepare_appworld_cuga_merlin import (
    CUGA_REVISION, LEGACY_REPOSITORY, LEGACY_SCRATCH, BENCHMARK_OUTPUT,
    PREPARATION_STATUS, RUNTIME_IDENTITY, archive_runtime, publish_evidence,
    require_merlin_cpu,
)
from scripts.run_appworld_benchmark_merlin import HdfsStore, run_command, write_json


def process_record(pid: int) -> dict:
    # proc_pid_stat(5): after comm, fields 3, 4, 22 and 52 are state,
    # parent PID, start ticks and waitpid-formatted exit code.
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text.rsplit(") ", 1)[1].split()
    return {"pid": pid, "state": fields[0], "parent_pid": int(fields[1]),
            "start_ticks": int(fields[19]), "wait_status": int(fields[49])}


def check_parent(pid: int, run_id: str, trial_id: str) -> dict:
    proc = Path(f"/proc/{pid}")
    args = (proc / "cmdline").read_bytes().split(bytes([0]))
    environment = dict(item.split(b"=", 1) for item in
                       (proc / "environ").read_bytes().split(bytes([0])) if b"=" in item)
    if (b"scripts.prepare_appworld_cuga_merlin" not in args
            or environment.get(b"RH2_JOB_RUN_ID") != run_id.encode()
            or environment.get(b"ARNOLD_TRIAL_ID") != trial_id.encode()):
        raise ValueError("PREPARATION_PARENT_OWNERSHIP_DIFFERS")
    return process_record(pid)


def verify_process_reader() -> None:
    child = subprocess.Popen(["python3", "-c", "raise SystemExit(7)"])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            row = process_record(child.pid)
            if row["state"] == "Z":
                if os.waitstatus_to_exitcode(row["wait_status"]) != 7:
                    raise ValueError("PROCESS_EXIT_STATUS_CHECK_FAILED")
                return
            time.sleep(0.05)
        raise TimeoutError("PROCESS_EXIT_STATUS_CHECK_TIMEOUT")
    finally:
        child.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--source-child-pid", type=int, required=True)
    parser.add_argument("--expected-base-prefix", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    identity = require_merlin_cpu(dict(os.environ))
    parent = check_parent(args.parent_pid, identity["run_id"], identity["trial_id"])
    if parent["state"] != "T":
        raise ValueError("ORIGINAL_PARENT_MUST_ALREADY_BE_HELD")
    work = Path("/tmp") / ("appworld-cuga-runtime-checkpoint-" + identity["run_id"])
    work.mkdir(exist_ok=False)
    output = work / "artifacts"
    output.mkdir()
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld/cuga_runtime_checkpoint"
    store = HdfsStore(prefix, work, output)
    report = {**identity, "schema": "appworld_cuga_runtime_checkpoint_v1",
              "checkpoint_path": prefix, "status": "WAITING_FOR_SOURCE_CHILD",
              "exit_code": 1, "persisted": False, "runtime_persisted": False,
              "runtime_prepared": False, "structural_checks_passed": False,
              "native_integration_qualified": False, "formal_eligible": False,
              "model_generations": 0, "appworld_tasks_executed": 0,
              "parent": parent, "source_child_pid": args.source_child_pid,
              "started_at": time.time(), "stages": {}}
    archive_path = LEGACY_REPOSITORY / BENCHMARK_OUTPUT / "cuga_runtime" / (
        "run_" + identity["compatibility_execution_id"])
    environment_path = archive_path / "environment.json"

    def interrupted(signum, frame):
        raise InterruptedError("CHECKPOINT_SIGNAL_" + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        verify_process_reader()
        report["process_exit_status_check"] = "PASS_REAL_CHILD_EXIT7"
        write_json(output / "progress.json", report)
        store.publish(output / "progress.json", "progress.json")
        deadline = time.monotonic() + args.wait_seconds
        while True:
            current = check_parent(args.parent_pid, identity["run_id"], identity["trial_id"])
            if current["start_ticks"] != parent["start_ticks"] or current["state"] != "T":
                raise ValueError("HELD_PREPARATION_PARENT_CHANGED")
            child = process_record(args.source_child_pid)
            if child["parent_pid"] != args.parent_pid:
                raise ValueError("SOURCE_CHILD_OWNER_DIFFERS")
            if child["state"] == "Z":
                report["source_child"] = child
                report["source_child_exit_code"] = os.waitstatus_to_exitcode(child["wait_status"])
                if report["source_child_exit_code"] != 0:
                    raise ValueError("SOURCE_PACKAGE_PREPARATION_FAILED")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("SOURCE_PREPARATION_STILL_RUNNING_RESUME_PARENT")
            time.sleep(2)
        environment = json.loads(environment_path.read_text())
        if (environment.get("status") != PREPARATION_STATUS
                or environment.get("source_revision") != CUGA_REVISION
                or environment.get("job_id") != identity["compatibility_execution_id"]):
            raise ValueError("SOURCE_PREPARATION_RECEIPT_DIFFERS")
        shutil.copy2(environment_path, output / "environment.json")
        run_command("verify_checkpoint_identity",
                    [environment["python"], "-c", RUNTIME_IDENTITY, str(environment_path),
                     str(output / "runtime_identity.json")],
                    dict(os.environ, CUDA_VISIBLE_DEVICES=""), work, output, store, report,
                    [output / "environment.json", output / "runtime_identity.json"])
        runtime = json.loads((output / "runtime_identity.json").read_text())
        source = Path(environment["source_checkout"]).resolve(strict=True)
        base = Path(runtime["base_prefix"]).resolve(strict=True)
        if (not source.is_relative_to(LEGACY_SCRATCH)
                or base != args.expected_base_prefix.resolve(strict=True)
                or not base.is_relative_to(Path("/opt/python"))
                or runtime["source_checkout"] != str(source)
                or runtime["prefix"] != str(source / ".venv")):
            raise ValueError("ACTUAL_RUNTIME_ROOT_BINDING_DIFFERS")
        report.update(runtime_prepared=True, environment=environment, runtime_identity=runtime)
        report["runtime_archive"] = archive_runtime((source, base), work / "runtime.tar.gz")
        report["runtime_bundle"] = store.publish(work / "runtime.tar.gz", "runtime.tar.gz")
        report.update(runtime_persisted=True, status="RUNTIME_CHECKPOINT_CAPTURED", exit_code=0)
        shutil.copytree(archive_path, output / "package_preparation")
        # Publish the usable checkpoint before allowing the original wrapper
        # to terminate. Its original result and the capture result stay separate.
        write_json(work / "checkpoint.json", report)
        store.publish(work / "checkpoint.json", "checkpoint.json")
    except BaseException as error:
        report.update(status="CHECKPOINT_FAILED", error_type=type(error).__name__,
                      error=str(error), exit_code=1)
    finally:
        try:
            current = check_parent(args.parent_pid, identity["run_id"], identity["trial_id"])
            if current["start_ticks"] != parent["start_ticks"]:
                raise ValueError("PARENT_START_IDENTITY_CHANGED")
            if current["state"] == "T":
                os.kill(args.parent_pid, signal.SIGCONT)
            report["source_parent_resume"] = {"signal": "SIGCONT",
                                             "recorded_at": time.time(), "status": "SENT"}
        except FileNotFoundError:
            report["source_parent_resume"] = {"status": "PARENT_ALREADY_EXITED"}
        except BaseException as error:
            report.update(exit_code=1, source_parent_resume={
                "status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        report["finished_at"] = time.time()
        write_json(output / "checkpoint_result.json", report)
        try:
            publish_evidence(store, work, output, report)
        except BaseException as error:
            report.update(publication_error=str(error), exit_code=1)
            write_json(work / "receipt.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

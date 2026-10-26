#!/usr/bin/env python3
"""Dispatch independent V1 source jobs through copies of the frozen runtime."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, BinaryIO, Sequence


RUNTIME_FILES = ("v1_remote.py", "pack_results.py")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _preflight(
    runtime_dir: Path, job_files: Sequence[Path], workspace_root: Path, workers: int
) -> tuple[Path, dict[str, bytes], list[dict[str, Any]]]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an explicit positive integer")
    if not job_files:
        raise ValueError("at least one original job file is required")
    if _exists(workspace_root):
        raise ValueError(f"workspace already exists: {workspace_root}")
    root = workspace_root.resolve()
    run = os.environ.get("RH2_JOB_RUN_ID", "")
    trial = os.environ.get("ARNOLD_TRIAL_ID", "")
    if not run or not SAFE_ID.fullmatch(trial):
        raise ValueError("a real Merlin run and path-safe trial ID are required")
    if int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0")) != 1:
        raise ValueError("the frozen runtime requires one allocated GPU")
    if not os.environ.get("RH2_CHECKPOINT_HDFS"):
        raise ValueError("RH2_CHECKPOINT_HDFS is required by the frozen runtime")
    scripts = {name: (runtime_dir / name).read_bytes() for name in RUNTIME_FILES}
    jobs = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for original in job_files:
        job_file = Path(original).resolve(strict=True)
        job = json.loads(job_file.read_text(encoding="utf-8"))
        if not isinstance(job, dict) or job.get("mode") != "source_batch":
            raise ValueError(f"source_batch job object required: {job_file}")
        job_id = job.get("job")
        if not isinstance(job_id, str) or not SAFE_ID.fullmatch(job_id):
            raise ValueError(f"path-safe string job ID required: {job_file}")
        if job_id in seen_ids:
            raise ValueError(f"duplicate job ID: {job_id}")
        output_base = job.get("output")
        if not isinstance(output_base, str) or not Path(output_base).is_absolute():
            raise ValueError(f"absolute output path required: {job_file}")
        execution_id = f"merlin_{trial}_v1_{job_id}"
        # Source jobs share an output base; the frozen runtime adds this leaf.
        output = Path(output_base) / ("run_" + execution_id)
        effective_output = output.resolve()
        if effective_output in seen_outputs:
            raise ValueError(f"duplicate effective output: {effective_output}")
        if _exists(output):
            raise ValueError(f"job output already exists: {output}")
        workspace = root / ("job_" + job_id)
        jobs.append({
            "job_id": job_id,
            "job_file": str(job_file),
            "execution_id": execution_id,
            "effective_output": str(effective_output),
            "workspace": str(workspace),
            "log": str(workspace / "stdout_stderr.log"),
            "status": "NOT_STARTED",
            "exit_code": None,
        })
        seen_ids.add(job_id)
        seen_outputs.add(effective_output)
    return root, scripts, jobs


def run_batch(
    *, runtime_dir: Path, job_files: Sequence[Path], workspace_root: Path, workers: int
) -> dict[str, Any]:
    """Run at most workers children; stop dispatch after the first observed failure.

    Child results describe process exits. Scientific admission remains with the
    frozen runtime and its existing consumers. No child receives a modified job.
    """
    runtime_dir = Path(runtime_dir).resolve()
    root, scripts, jobs = _preflight(
        runtime_dir, job_files, Path(workspace_root), workers
    )
    state: dict[str, Any] = {
        "result_kind": "process_dispatch",
        "status": "PREPARING",
        "workers": workers,
        "runtime_dir": str(runtime_dir),
        "runtime_file_bytes": {name: len(data) for name, data in scripts.items()},
        "merlin_job_run_id": os.environ["RH2_JOB_RUN_ID"],
        "merlin_trial_id": os.environ["ARNOLD_TRIAL_ID"],
        "launch_cwd": os.getcwd(),
        "started_utc": _utc_now(),
        "exit_code": None,
        "first_failure": None,
        "jobs": jobs,
    }
    active: dict[int, tuple[subprocess.Popen, BinaryIO]] = {}
    root.mkdir(parents=True, exist_ok=False)

    def save() -> None:
        temporary = root / "batch_results.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "batch_results.json")

    def fail(job: dict[str, Any], code: int, error: str | None = None) -> None:
        if state["first_failure"] is None:
            state["first_failure"] = {
                "job_id": job["job_id"], "exit_code": code, "error": error
            }

    def finish(index: int, code: int) -> None:
        job = jobs[index]
        job.update(
            status="SUCCEEDED" if code == 0 else "FAILED",
            exit_code=code,
            finished_utc=_utc_now(),
        )
        if code != 0:
            fail(job, code)

    def reap() -> None:
        for index, (process, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del active[index]
                finish(index, code)
                save()

    try:
        save()
        # Complete every workspace before any child can start writing results.
        for job in jobs:
            workspace = Path(job["workspace"])
            workspace.mkdir()
            for name, data in scripts.items():
                (workspace / name).write_bytes(data)
        state["status"] = "RUNNING"
        save()
        next_index = 0
        while active or next_index < len(jobs):
            reap()
            while (
                state["first_failure"] is None
                and len(active) < workers
                and next_index < len(jobs)
            ):
                # Observe completed failures before using a newly free slot.
                reap()
                if state["first_failure"] is not None:
                    break
                index = next_index
                next_index += 1
                job = jobs[index]
                command = [
                    sys.executable, str(Path(job["workspace"]) / "v1_remote.py"),
                    "--job-file", job["job_file"], "--gpu", "0",
                ]
                job["command"] = command
                log = None
                try:
                    log = Path(job["log"]).open("xb")
                    # Inherit cwd and environment to preserve original inputs.
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                except OSError as error:
                    if log is not None:
                        log.close()
                    job.update(status="START_FAILED", error=str(error),
                               finished_utc=_utc_now())
                    fail(job, 1, str(error))
                    save()
                    break
                active[index] = (process, log)
                job.update(status="RUNNING", pid=process.pid, started_utc=_utc_now())
                save()
            if not active and state["first_failure"] is not None:
                break
            if active:
                time.sleep(0.05)
    except BaseException as error:
        state["dispatcher_error"] = f"{type(error).__name__}: {error}"
        if state["first_failure"] is None:
            state["first_failure"] = {
                "job_id": None,
                "exit_code": 130 if isinstance(error, KeyboardInterrupt) else 1,
                "error": state["dispatcher_error"],
            }
        raise
    finally:
        # Preserve active producers and their results after a sibling failure.
        for index, (process, log) in list(active.items()):
            code = process.wait()
            log.close()
            finish(index, code)
            del active[index]
            save()
        failure = state["first_failure"]
        if failure is not None:
            for job in jobs:
                if job["status"] == "NOT_STARTED":
                    job["not_started_reason"] = "dispatch stopped after failure"
            raw_code = failure["exit_code"]
            state["exit_code"] = raw_code if raw_code > 0 else 128 - raw_code
            state["status"] = "FAILED"
        else:
            state.update(status="SUCCEEDED", exit_code=0)
        state["finished_utc"] = _utc_now()
        save()
    return state


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--job-file", type=Path, action="append", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--workers", type=_positive_integer, required=True)
    args = parser.parse_args(argv)
    try:
        state = run_batch(
            runtime_dir=args.runtime_dir, job_files=args.job_file,
            workspace_root=args.workspace_root, workers=args.workers,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    print(json.dumps(state), flush=True)
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

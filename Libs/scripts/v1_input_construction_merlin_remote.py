#!/usr/bin/env python3
"""Validate and supervise C98/C99 input construction in a Merlin CPU job."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

import ca3_merlin_remote as transport


def main() -> int:
    work = Path(__file__).resolve().parent
    job = json.loads((work / "job.json").read_text())
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/v1/input_construction/" + trial
    output = work / "validation"
    output.mkdir(exist_ok=False)
    receipt = dict(status="FAILED_V1_MERLIN_INPUT_CONSTRUCTION", exit_code=1, persisted=False,
                   merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(),
                   started_utc=datetime.now(timezone.utc).isoformat(), circuits=[],
                   source_simulations=0, production_episodes=0, training_seeds_used=False)
    ready = False
    try:
        if (os.environ.get("ARNOLD_MONITOR_ROLE") != "head"
                or os.environ.get("ARNOLD_HEAD_GPU") != "0"
                or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"):
            raise ValueError("V1_INPUT_ACTUAL_MERLIN_CPU_ALLOCATION_REQUIRED")
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target + "/validation"], check=True)
        ready = True
        receipt["memory_evidence"] = transport.memory_evidence(job["resource_shape"], required_available_gib=64)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                           PYTHONHASHSEED="0", UV_CACHE_DIR=str(work / "uv_cache"),
                           UV_PYTHON_INSTALL_DIR=str(work / "python"))
        with (output / "environment_setup.log").open("x") as log:
            # Reuse the image interpreter only when it matches the source canary.
            interpreter = sys.executable
            if platform.python_version() != job["python_version"]:
                subprocess.run([sys.executable, "-m", "pip", "install", "uv==" + job["uv_version"]],
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                uv = [sys.executable, "-m", "uv"]
                subprocess.run([*uv, "python", "install", job["python_version"]], env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                interpreter = subprocess.check_output([*uv, "python", "find", job["python_version"]],
                                                      env=environment, text=True).strip()
            python = work / "venv/bin/python"
            subprocess.run([interpreter, "-m", "venv", str(work / "venv")], env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([str(python), "-m", "pip", "install", *job["python_packages"]],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt["python"] = str(python)
        archive = transport.download_inputs(job, work)
        transport.stage_inputs(archive, job["input_files"])
        receipt["runtime_overrides"] = job["runtime_overrides"]
        for name, encoded in job["supplemental_contents"].items():
            path = Path(name)
            data = base64.b64decode(encoded, validate=True)
            identity = job["supplemental_files"][name]
            if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError("V1_INPUT_SUPPLEMENTAL_IDENTITY_DIFFERS:" + name)
            if path.exists():
                before = transport.digest(path)
                expected = job["runtime_overrides"].get(name, {}).get("base_sha256", identity["sha256"])
                if before != expected:
                    raise ValueError("V1_INPUT_BASE_SNAPSHOT_DIFFERS:" + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        repo = Path(job["repo"])
        environment["PYTHONPATH"] = str(repo / "Libs")
        # The unchanged source caching tests already have frozen canary evidence.
        # These tests exercise the changed caller, config and resource boundary.
        xml = output / "tests.xml"
        command = [str(python), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--junitxml=" + str(xml), *job["tests"]]
        receipt["tests_command"] = command
        with (output / "tests.log").open("x") as log:
            code = subprocess.run(command, env=environment, cwd=repo, stdout=log,
                                  stderr=subprocess.STDOUT, check=False).returncode
        root = ET.parse(xml).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        receipt["tests"] = dict(counts, exit_code=code)
        if code or counts != dict(tests=16, failures=0, errors=0, skipped=0):
            raise ValueError("V1_INPUT_FOCUSED_TESTS_NOT_PASSED")
        for task in job["tasks"]:
            circuit = task["circuit_index"]
            task_work = work / f"circuit_{circuit:03d}"
            task_work.mkdir(exist_ok=False)
            shutil.copy2(work / "pack_results.py", task_work / "pack_results.py")
            delivery = task_work / "source"
            command = [str(python), "-B", str(repo / "Libs/scripts/v1_input_construction_merlin.py"),
                       "run", "--repo", str(repo), "--config", task["config"],
                       "--recurrent-record", task["recurrent_record"],
                       "--qualification", job["qualification"], "--output", str(delivery)]
            with (output / f"circuit_{circuit:03d}.log").open("x") as log:
                code = subprocess.run(command, env=environment, cwd=repo, stdout=log,
                                      stderr=subprocess.STDOUT, check=False).returncode
            record = dict(circuit_index=circuit, source_slurm_job=task["source_slurm_job"],
                          command=command, exit_code=code, merlin_job_run_id=run, merlin_trial_id=trial,
                          hostname=socket.gethostname(), persisted=False, production_episodes=0)
            if delivery.exists():
                source_receipt = json.loads((delivery / "delivery.json").read_text())
                record["status"] = source_receipt["status"]
                record["delivery_sha256"] = transport.digest(delivery / "delivery.json")
                circuit_target = target + f"/circuit_{circuit:03d}"
                subprocess.run(["hdfs", "dfs", "-mkdir", "-p", circuit_target], check=True)
                transport.persist(delivery, task_work, circuit_target, record, chunk_bytes=256 * 1024)
            receipt["circuits"].append(record)
            transport.write_json(output / "progress.json", receipt)
            if code or record.get("status") != "PASS_SOURCE_INPUT_BUILD_REPEATABILITY":
                raise ValueError("V1_INPUT_CIRCUIT_BUILD_FAILED:" + str(circuit))
        receipt.update(status="PASS_V1_MERLIN_TWO_INPUT_CONSTRUCTIONS", exit_code=0)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        transport.write_json(output / "merlin_terminal.json", receipt)
        if ready:
            transport.persist(output, work, target + "/validation", receipt, chunk_bytes=256 * 1024)
        print(json.dumps({key: receipt.get(key) for key in
                          ("status", "exit_code", "persisted", "merlin_job_run_id", "merlin_trial_id", "error")}), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

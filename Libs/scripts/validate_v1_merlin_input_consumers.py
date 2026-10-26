#!/usr/bin/env python3
"""Test input consumers in an isolated copy of a running Merlin CPU job's code."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    work = Path(request["workspace"])
    job = json.loads((work / "job.json").read_text())
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    if (run != request["run_sid"] or trial != request["trial_sid"]
            or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"):
        raise ValueError("V1_CONSUMER_VALIDATION_CPU_JOB_IDENTITY_DIFFERS")
    output = args.output.resolve() if args.output else work / "input_consumer_validation"
    if not output.is_relative_to(work):
        raise ValueError("V1_CONSUMER_VALIDATION_OUTPUT_OUTSIDE_WORKSPACE")
    output.mkdir(exist_ok=False)
    repo = Path(job["repo"])
    overlay = output / "code"
    identities = dict(job["input_files"], **job["supplemental_files"])
    for name, identity in identities.items():
        path = Path(name)
        if not path.is_relative_to(repo / "Libs") or path.suffix != ".py":
            continue
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError("V1_CONSUMER_BASE_CODE_CHANGED:" + name)
        target = overlay / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    deployed = {}
    for relative, item in request["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "Libs":
            raise ValueError("V1_CONSUMER_CODE_PATH_INVALID")
        data = base64.b64decode(item["content"], validate=True)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("V1_CONSUMER_DEPLOYMENT_IDENTITY_DIFFERS")
        target = overlay / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        deployed[relative] = item["sha256"]
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "tests.xml"), *[
                   str(overlay / "Libs/population_dynamics" / name) for name in (
                       "test_v1_construction_intake_merlin.py", "test_v1_reference_preparation_merlin.py")]]
    pythonpaths = [str(overlay / "Libs"), str(overlay / "Libs/scripts")]
    dependency_versions = {}
    if request.get("validation_dependencies"):
        dependencies = Path(request["validation_dependencies"]).resolve()
        if not dependencies.is_relative_to(work) or not dependencies.is_dir():
            raise ValueError("V1_CONSUMER_DEPENDENCIES_OUTSIDE_WORKSPACE_OR_ABSENT")
        pythonpaths.append(str(dependencies))
        dependency_versions = {item.metadata["Name"]: item.version
                               for item in metadata.distributions(path=[str(dependencies)])}
        if dependency_versions.get("requests") != "2.32.4":
            raise ValueError("V1_CONSUMER_REQUESTS_VERSION_DIFFERS")
    pythonpath = os.pathsep.join(pythonpaths)
    environment = dict(os.environ, PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (output / "tests.log").open("x") as log:
        code = subprocess.run(command, env=environment, cwd=overlay, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode
    root = ET.parse(output / "tests.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "errors", "failures", "skipped")}
    passed = code == 0 and counts == dict(tests=23, errors=0, failures=0, skipped=0)
    report = dict(status="PASS_INPUT_CONSUMER_TESTS" if passed else "FAILED_INPUT_CONSUMER_TESTS",
                  exit_code=0 if passed else (code or 1), tests=counts, tested_code_sha256=deployed,
                  command=command, pythonpath=pythonpath, validation_dependency_versions=dependency_versions,
                  merlin_job_run_id=run, merlin_trial_id=trial,
                  hostname=socket.gethostname(), observed_utc=datetime.now(timezone.utc).isoformat(),
                  source_simulations=0, production_episodes=0, canonical_files_written=0)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    target = (os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
              + "/v1/input_consumer_validation/" + trial + "/" + output.name)
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
    for name in ("tests.xml", "tests.log", "report.json"):
        subprocess.run(["hdfs", "dfs", "-put", str(output / name), target + "/" + name], check=True)
    print(json.dumps(report), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

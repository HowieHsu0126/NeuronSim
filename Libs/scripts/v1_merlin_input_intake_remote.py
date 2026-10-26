#!/usr/bin/env python3
"""Consume completed V1 input constructions in their existing Merlin CPU job."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    work = Path(request["workspace"])
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    if (run != request["run_sid"] or trial != request["trial_sid"]
            or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"):
        raise ValueError("V1_CONSUMER_ACTUAL_CPU_JOB_IDENTITY_REQUIRED")
    tests = json.loads((work / request["test_report"]).read_text())
    if (tests["status"] != "PASS_INPUT_CONSUMER_TESTS"
            or tests["tests"] != dict(tests=23, errors=0, failures=0, skipped=0)
            or tests["merlin_job_run_id"] != run or tests["merlin_trial_id"] != trial):
        raise ValueError("V1_CONSUMER_FOCUSED_TESTS_NOT_PASSED")
    overlay = (work / request["test_report"]).parent / "code"
    for name, expected in tests["tested_code_sha256"].items():
        if digest(overlay / name) != expected:
            raise ValueError("V1_CONSUMER_TESTED_CODE_CHANGED:" + name)
    sys.path.insert(0, str(work))
    import ca3_merlin_remote as transport
    circuits = request["circuits"]
    if not circuits or any(c not in (98, 99) for c in circuits) or len(set(circuits)) != len(circuits):
        raise ValueError("V1_CONSUMER_CIRCUITS_INVALID")
    label = "_".join(f"c{c:03d}" for c in circuits)
    task = work / ("input_intake_" + label)
    task.mkdir(exist_ok=False)
    shutil.copy2(work / "pack_results.py", task / "pack_results.py")
    output = task / "results"
    output.mkdir()
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/v1/input_intake/" + trial + "/" + label
    receipt = dict(status="FAILED_V1_INPUT_INTAKE_AND_REFERENCE", exit_code=1, persisted=False,
                   merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(),
                   circuits=circuits, source_simulations=0, production_episodes=0,
                   canonical_files_written=0, tested_code_sha256=tests["tested_code_sha256"],
                   started_utc=datetime.now(timezone.utc).isoformat())
    try:
        for name, identity in request["metadata"].items():
            path = Path(name)
            data = base64.b64decode(identity["content"], validate=True)
            if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise ValueError("V1_CONSUMER_METADATA_IDENTITY_DIFFERS:" + name)
            if path.exists():
                if digest(path) != identity["sha256"]:
                    raise ValueError("V1_CONSUMER_STAGED_METADATA_CONFLICT:" + name)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        rows = []
        for circuit in circuits:
            source = work / f"circuit_{circuit:03d}/source"
            sha = digest(source / "delivery.json")
            if sha != request["expected_deliveries"][str(circuit)]:
                raise ValueError("V1_CONSUMER_EXPECTED_SOURCE_DELIVERY_DIFFERS")
            delivery = json.loads((source / "delivery.json").read_text())
            canonical = (Path(request["source_root"]) / "input_construction" /
                         ("merlin_" + run + "_" + trial) / f"circuit_{circuit:03d}")
            shutil.copytree(source, canonical)
            rows.append(dict(circuit_index=circuit, archive=str(canonical), delivery_sha256=sha,
                             execution_context=delivery["execution_context"]))
        manifest = dict(schema="v1_merlin_input_deliveries_v1", constructions=rows)
        manifest_path = Path(request["manifest_target"])
        save(manifest_path, manifest)
        shutil.copy2(manifest_path, output / "source_manifest.json")
        q = Path(request["qualification_root"])
        shape = task / "resource_shape.json"
        save(shape, json.loads((work / "job.json").read_text())["resource_shape"])
        environment = dict(os.environ, PYTHONPATH=tests["pythonpath"], PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        commands = [
            ("intake", [str(overlay / "Libs/scripts/v1_construction_intake_merlin.py"), "run-merlin-inputs",
                        "--qualification-root", str(q), "--construction-root", str(q / "construction_expansion_20261017"),
                        "--source-root", request["source_root"], "--cohort-manifest", str(q / "identities.json"),
                        "--cohort-manifest-sha256", request["cohort_sha256"], "--construction-manifest", str(manifest_path),
                        "--construction-manifest-sha256", digest(manifest_path), "--resource-shape", str(shape),
                        "--output", str(output / "intake")]),
            ("reference", [str(overlay / "Libs/scripts/v1_reference_preparation_merlin.py"), "run",
                           "--qualification-root", str(q), "--source-control", str(q / "circuit_000/reference_execution/prepared_control.json"),
                           "--execution-resources", str(q / "construction_expansion_20261017/cpu04_execution_resources.yaml"),
                           "--intake-output", str(output / "intake"), "--circuits", *map(str, circuits),
                           "--output", str(output / "reference")])]
        receipt["commands"] = []
        for stage, arguments in commands:
            command = [sys.executable, "-B", *arguments]
            with (output / (stage + ".log")).open("x") as log:
                rc = subprocess.run(command, env=environment, cwd=overlay, stdout=log,
                                    stderr=subprocess.STDOUT, check=False).returncode
            receipt["commands"].append(dict(stage=stage, command=command, exit_code=rc))
            if rc:
                raise ValueError("V1_CONSUMER_STAGE_FAILED:" + stage)
        intake = json.loads((output / "intake/report.json").read_text())
        reference = json.loads((output / "reference/report.json").read_text())
        if (intake["validated_input_acceptances"] != len(circuits)
                or reference["prepared_conditions"] != 10 * len(circuits)):
            raise ValueError("V1_CONSUMER_COMPLETE_CIRCUITS_REQUIRED")
        receipt.update(status="PASS_V1_INPUT_INTAKE_AND_REFERENCE", exit_code=0,
                       validated_input_acceptances=intake["validated_input_acceptances"],
                       prepared_conditions=reference["prepared_conditions"],
                       source_manifest_target=str(manifest_path))
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "merlin_terminal.json", receipt)
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        transport.persist(output, task, target, receipt, chunk_bytes=256 * 1024)
        print(json.dumps({k: receipt.get(k) for k in ("status", "exit_code", "persisted", "validated_input_acceptances", "prepared_conditions", "error")}), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

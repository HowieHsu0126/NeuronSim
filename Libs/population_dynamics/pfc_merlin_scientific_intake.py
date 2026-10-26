"""Validate persisted PFC episodes with the existing consumer on Merlin CPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

import numpy as np

from return_results import integrate
from . import pfc_parameter_task_dynamics as dynamics
from .pfc_merlin_intake import (
    check_expected, read_json, require, validate_episode, write_json,
)
from .provenance import sha256_file


def hdfs(*arguments):
    return subprocess.run(["hdfs", "dfs", *map(str, arguments)],
                          check=True, text=True, capture_output=True)


def download_json(uri, path):
    hdfs("-get", uri, path)
    return read_json(path)


def receive_episode(config, task, work, execution):
    receipt = download_json(task["receipt_uri"], work / "remote_receipt.json")
    expected = {key: task[key] for key in ("index", "run_id", "trial_id", "xie_job")}
    expected.update(status="COMPLETE", exit_code=0, persisted=True)
    check_expected(receipt, expected)
    source_uri = receipt.get("publication_recovery", {}).get(
        "source_bundle_uri", receipt["checkpoint_path"].rstrip("/") + "/result.tar.gz")
    require(source_uri == task["bundle_uri"], "BOUND_SOURCE_BUNDLE_URI_CHANGED")
    bundle = work / "result.tar.gz"
    hdfs("-get", source_uri, bundle)
    require(bundle.stat().st_size == receipt["result_bytes"]
            and sha256_file(bundle) == receipt["result_sha256"],
            "PERSISTED_SOURCE_BUNDLE_CHANGED")
    manifest = None
    if receipt.get("external_manifest_uri"):
        manifest = work / "manifest.json"
        download_json(receipt["external_manifest_uri"], manifest)
    integration = integrate(bundle, manifest)
    accepted = validate_episode(config, task, receipt)
    accepted.update(validation_execution=execution, source_bundle_uri=source_uri,
        source_bundle_sha256=receipt["result_sha256"], receipt_uri=task["receipt_uri"],
        remote_receipt=receipt, verified_files=integration["files"],
        materialization="Merlin container-local canonical paths restored from HDFS",
        full_source_bundle_returned_to_xie=False)
    return accepted


def run(config_path, work):
    run_id, trial_id = os.environ.get("RH2_JOB_RUN_ID"), os.environ.get("ARNOLD_TRIAL_ID")
    require(bool(run_id) and bool(trial_id)
            and os.environ.get("ARNOLD_MONITOR_ROLE") == "head"
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0"
            and not os.environ.get("CUDA_VISIBLE_DEVICES"), "MERLIN_CPU_ALLOCATION_REQUIRED")
    require(np.__version__ == "2.4.2", "FROZEN_NUMPY_VERSION_REQUIRED")
    filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(work)], text=True).strip()
    require(filesystem not in {"nfs", "nfs4", "cifs", "smb2"}, "LOCAL_SCRATCH_REQUIRED")
    require(shutil.disk_usage(work).free >= 12 * 1024**3, "PFC_INTAKE_SCRATCH_SPACE_REQUIRED")
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/pfc_scientific_intake"
    config = read_json(config_path)
    control = read_json(config["control_validation"])
    require(sha256_file(Path(dynamics.__file__)) == control["implementation_sha256"],
            "VALIDATED_CONTROL_IMPLEMENTATION_CHANGED")
    prior = read_json(config["prior_acceptance"])
    accepted = set(prior["accepted_task_indices"])
    require(len(accepted) == prior["accepted_episode_count"], "PRIOR_ACCEPTANCE_COUNT_MISMATCH")
    tasks = config["return_tasks"]
    require(bool(tasks) and len({t["index"] for t in tasks}) == len(tasks)
            and not accepted.intersection(t["index"] for t in tasks), "DUPLICATE_OR_ACCEPTED_TASK")
    execution = dict(run_id=run_id, trial_id=trial_id, hostname=socket.gethostname(),
        backend="merlin", allocated_cpus=os.environ.get("ARNOLD_HEAD_CPU"),
        allocated_gpus=0, scratch_filesystem=filesystem, numpy_version=np.__version__,
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=sha256_file(Path(__file__)),
        existing_consumer_sha256=sha256_file(Path(__file__).with_name("pfc_merlin_intake.py")))
    report = dict(schema="pfc_merlin_scientific_intake_v1", status="RUNNING",
        execution=execution, checkpoint_path=checkpoint,
        prior_acceptance=config["prior_acceptance"],
        prior_acceptance_sha256=sha256_file(Path(config["prior_acceptance"])),
        control_validation=config["control_validation"],
        control_validation_sha256=sha256_file(Path(config["control_validation"])),
        config_sha256=sha256_file(config_path), requested_task_indices=[t["index"] for t in tasks],
        completed=[], failures=[], accepted_task_indices=sorted(accepted),
        accepted_episode_count=len(accepted), new_accepted_episode_count=0,
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scaling_qualified=False, scientific_parameter_domain_qualified=False,
        full_source_bundles_returned_to_xie=False)
    hdfs("-mkdir", "-p", checkpoint)
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"task_{task['index']:04d}_", dir=work) as directory:
            try:
                value = receive_episode(config, task, Path(directory), execution)
                path = work / f"task_{task['index']:04d}_acceptance.json"
                write_json(path, value)
                uri = checkpoint + "/" + path.name
                hdfs("-put", path, uri)
                accepted.add(task["index"])
                report["completed"].append(dict(index=task["index"], acceptance_uri=uri,
                                                acceptance_sha256=sha256_file(path)))
            except Exception as error:
                failure = dict(index=task["index"], error_type=type(error).__name__, error=str(error))
                if isinstance(error, subprocess.CalledProcessError):
                    failure["stderr"] = error.stderr[-4000:]
                report["failures"].append(failure)
        report.update(accepted_task_indices=sorted(accepted), accepted_episode_count=len(accepted),
                      new_accepted_episode_count=len(report["completed"]))
        path = work / "progress.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        hdfs("-put", "-f", path, checkpoint + "/progress.json")
        print(json.dumps(dict(index=task["index"], accepted=len(accepted),
                              failures=len(report["failures"]))), flush=True)
    report["status"] = "PASS" if not report["failures"] else "FAILED"
    write_json(work / "acceptance.json", report)
    hdfs("-put", work / "acceptance.json", checkpoint + "/acceptance.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report = run(args.config, args.work)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

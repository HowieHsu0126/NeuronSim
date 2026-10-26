#!/usr/bin/env python3
"""Run the frozen CA3 joint readout on ten existing Merlin source bundles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import time

from ca3_merlin_remote import digest, persist, write_json
from return_results import integrate
from v1_merlin_readout import cpu_preflight


def execute(job: dict, work: Path) -> int:
    output = work / "output"
    output.mkdir(exist_ok=False)
    trial = os.environ["ARNOLD_TRIAL_ID"]
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/ca3/ten_condition_readout/" + trial
    started = time.monotonic()
    receipt = dict(
        status="FAILED_CA3_MERLIN_TEN_CONDITION_READOUT", exit_code=1, persisted=False,
        execution_backend="merlin", merlin_job_run_id=os.environ["RH2_JOB_RUN_ID"],
        merlin_trial_id=trial, hostname=socket.gethostname(), output=str(output),
        started_utc=datetime.now(timezone.utc).isoformat(), source_simulations=0,
        production_episodes=0, scientific_fidelity_qualified=False, source_intakes=[])
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        if [row["condition_index"] for row in job["sources"]] != list(range(10)):
            raise ValueError("CA3_COMPLETE_ORDERED_TEN_CONDITION_SOURCES_REQUIRED")
        source = work / "source" / "Libs"
        for relative, expected in job["software_files"].items():
            path = work / relative
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError("CA3_FROZEN_READOUT_SOFTWARE_CHANGED:" + relative)
        environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
        tests = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                 str(source / "population_dynamics/test_ca3_benchmark_readout.py"),
                 "-k", "not actual_readout_launcher and not complete_block_producer",
                 "--junitxml=" + str(output / "tests.xml")]
        # The two excluded fixtures target the historical Slurm launchers. The
        # actual Merlin entry point below consumes all ten complete source outputs.
        with (output / "tests.log").open("w") as stream:
            tested = subprocess.run(tests, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        receipt.update(tests_exit_code=tested.returncode, tests_command=tests)
        if tested.returncode:
            raise RuntimeError("CA3_FROZEN_READOUT_TESTS_FAILED")
        episodes = []
        for row in job["sources"]:
            index = row["condition_index"]
            bundle = work / f"condition_{index:02d}.tar.gz"
            subprocess.run(["hdfs", "dfs", "-get", row["bundle_uri"], str(bundle)], check=True)
            if bundle.stat().st_size != row["bytes"] or digest(bundle) != row["sha256"]:
                raise ValueError("CA3_SOURCE_BUNDLE_IDENTITY_DIFFERS:" + str(index))
            original = Path(row["episode_path"])
            prefix = str(original.parent).lstrip("/") + "/"
            if (not original.is_absolute() or ".." in original.parts
                    or not str(original).startswith("/tmp/neuronsim-ca3-")):
                raise ValueError("CA3_SOURCE_EPISODE_PATH_INVALID")
            with tarfile.open(bundle) as archive:
                manifest = json.load(archive.extractfile("merlin_return_manifest.json"))
            if not manifest or any(not name.startswith(prefix) for name in manifest):
                raise ValueError("CA3_SOURCE_BUNDLE_OUTPUT_SCOPE_DIFFERS:" + str(index))
            needed = sum(value["bytes"] for value in manifest.values())
            if shutil.disk_usage(work).free <= needed:
                raise ValueError("CA3_SOURCE_INTAKE_SCRATCH_INSUFFICIENT")
            verified = integrate(bundle)
            stage = bundle.with_suffix(".intake")
            episode = stage / str(original).lstrip("/")
            native = json.loads((episode / "report.json").read_text())
            if (native["candidate_identity"]["condition_index"] != index
                    or native["duration_ms"] != 1210
                    or native["condition"]["condition_id"] != row["condition_id"]):
                raise ValueError("CA3_SOURCE_CONDITION_IDENTITY_DIFFERS:" + str(index))
            metadata = output / "source_metadata" / f"condition_{index:02d}"
            metadata.mkdir(parents=True)
            for name in manifest:
                path = Path(name)
                if path.suffix in {".json", ".log"}:
                    destination = metadata / name[len(prefix):]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stage / name, destination)
            receipt["source_intakes"].append(dict(
                condition_index=index, condition_id=row["condition_id"],
                original_run_sid=row["run_sid"], original_trial_sid=row["trial_sid"],
                source_bundle_uri=row["bundle_uri"], source_bundle_bytes=row["bytes"],
                source_bundle_sha256=row["sha256"], files_verified=verified["files"],
                episode_path=str(episode)))
            write_json(output / "source_intakes.json", receipt["source_intakes"])
            print(json.dumps(receipt["source_intakes"][-1]), flush=True)
            episodes.append(episode)
            bundle.unlink()  # The verified original remains in HDFS.
        sys.path.insert(0, str(source))
        readout = importlib.import_module("population_dynamics.ca3_benchmark_readout")
        report = readout.readout(episodes, output / "readout")
        if (report["status"] != "COMPLETE_CA3_BENCHMARK_TEN_CONDITION_DEVELOPMENT"
                or report["complete_episodes"] != 10 or report["formed_circuit_realizations"] != 1
                or report["production_episodes"] != 0
                or report["scientific_fidelity_qualified"] is not False):
            raise ValueError("CA3_JOINT_DEVELOPMENT_READOUT_GATE_FAILED")
        receipt.update(status="COMPLETE_CA3_MERLIN_TEN_CONDITION_READOUT", exit_code=0,
                       readout=report, original_bundles_verified=10,
                       original_raw_bundles_retained_in_hdfs=True)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic() - started)
        write_json(output / "job.json", job)
        write_json(output / "merlin_terminal.json", receipt)
        try:
            persist(output, work, target, receipt, chunk_bytes=256 * 1024)
        except BaseException as error:
            receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                           persistence_error_type=type(error).__name__, persistence_error=str(error))
        print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.job_json.read_text()), args.job_json.resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())

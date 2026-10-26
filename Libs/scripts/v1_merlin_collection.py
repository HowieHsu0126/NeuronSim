#!/usr/bin/env python3
"""Extend the V1 collection through its public CLI using accepted block readbacks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import ca3_merlin_remote as transport
from v1_merlin_readout import cpu_preflight


def execute(job: dict, work: Path) -> int:
    run = os.environ["RH2_JOB_RUN_ID"]
    trial = os.environ["ARNOLD_TRIAL_ID"]
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/v1/collection/" + trial
    output = work / "collection_preparation"
    output.mkdir(exist_ok=False)
    receipt = dict(status="FAILED_V1_COLLECTION_REFRESH", exit_code=1, persisted=False,
                   merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(),
                   source_simulations=0, episode_payloads_revalidated=False,
                   started_utc=datetime.now(timezone.utc).isoformat())
    ready = False
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        ready = True
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        archive = transport.download_inputs(job, work)
        transport.stage_inputs(archive, job["input_files"])
        plan = job["plan"]
        root = Path(plan["collection"])
        if transport.digest(root / "dataset.json") != plan["prior_catalog_sha256"]:
            raise ValueError("COLLECTION_INPUT_CATALOG_CHANGED")
        canonical = Path(job["output_root"]) / ("run_" + run)
        canonical.mkdir(parents=True, exist_ok=False)
        output.rmdir()
        output = canonical
        command = [sys.executable, "-B", "-m", "population_dynamics.benchmark_collection",
                   "--config", str(root / "frozen_campaign.yaml"), "--scenario", "v1_visual",
                   "--qualification", str(root / "qualification.json"),
                   "--identity-table", str(root / "identities.csv"),
                   "--output-root", str(root), "--resume"]
        for report in plan["readback_reports"]:
            command.extend(("--readback-report", report))
        environment = dict(os.environ, PYTHONPATH=plan["public_root"], PYTHONDONTWRITEBYTECODE="1",
                           NRNSIM_EXECUTION_BACKEND="merlin", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        (output / "collection.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"COLLECTION_CLI_FAILED:{result.returncode}")
        report = json.loads(result.stdout)
        if (report["complete_episodes"] != plan["prior_episodes"] + 10 * plan["added_blocks"]
                or report["complete_blocks"] != plan["prior_blocks"] + plan["added_blocks"]
                or report["expected_complete_episodes"] != 100000):
            raise ValueError("COLLECTION_COMPLETE_BLOCK_COUNT_DIFFERS")
        shutil.copyfile(root / "dataset.json", output / "dataset.json")
        transport.write_json(output / "report.json", report)
        transport.write_json(output / "execution_inputs.json", job)
        receipt.update(status="PASS_V1_COLLECTION_REFRESH", exit_code=0,
                       complete_episodes=report["complete_episodes"], complete_blocks=report["complete_blocks"],
                       prior_catalog_sha256=plan["prior_catalog_sha256"],
                       catalog_sha256=report["catalog_sha256"], canonical_dataset=str(root),
                       output=str(output), command=command)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        transport.write_json(output / "merlin_terminal.json", receipt)
        if ready:
            try:
                transport.persist(output, work, target, receipt, chunk_bytes=256 * 1024)
            except BaseException as error:
                receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                               persistence_error_type=type(error).__name__, persistence_error=str(error))
                transport.write_json(output / "merlin_terminal.json", receipt)
        print(json.dumps({key: receipt.get(key) for key in (
            "status", "exit_code", "persisted", "complete_episodes", "error")}), flush=True)
    return receipt["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.job_json.read_text()), args.job_json.resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())

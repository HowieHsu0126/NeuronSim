#!/usr/bin/env python3
"""Admit one saved V1 block on Merlin CPU through the public production CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import ca3_merlin_remote as transport
from v1_merlin_readout import cpu_preflight, restore_result, validate_receipt

CHUNK_BYTES = 256 * 1024


def run_cli(command: list[str], environment: dict, log: Path) -> str:
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"PRODUCTION_CLI_FAILED:{result.returncode}:{log}")
    return result.stdout


def execute(job: dict, work: Path) -> int:
    run = os.environ["RH2_JOB_RUN_ID"]
    trial = os.environ["ARNOLD_TRIAL_ID"]
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"]
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("MERLIN_CPU_EXECUTION_REQUIRED")
    output = work / "admission_preparation"
    output.mkdir(exist_ok=False)
    target = checkpoint.rstrip("/") + "/v1/production_admission/" + trial
    receipt = dict(status="PREPARING_V1_PRODUCTION_ADMISSION", exit_code=1,
                   execution_backend="merlin", merlin_job_run_id=run,
                   merlin_trial_id=trial, hostname=socket.gethostname(),
                   output=str(output), source_simulations=0, canonical_integration=False,
                   admitted_complete_episodes=0, persisted=False,
                   started_utc=datetime.now(timezone.utc).isoformat())
    ready = False
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        ready = True
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        for index, layer in enumerate(job["input_layers"]):
            stage = work / f"input_layer_{index}"
            stage.mkdir()
            archive = transport.download_inputs(layer, stage)
            transport.stage_inputs(archive, layer["input_files"])
        canonical_output = Path(job["output_root"]) / ("run_" + run)
        canonical_output.mkdir(parents=True, exist_ok=False)
        output.rmdir()
        output = canonical_output
        receipt["output"] = str(output)
        for name, expected in job["software_files"].items():
            if transport.digest(Path(name)) != expected["sha256"]:
                raise ValueError("ADMISSION_PUBLIC_SOFTWARE_CHANGED:" + name)
        sources = json.loads(Path(job["sources"]).read_text())
        cases = json.loads(Path(job["cases"]).read_text())
        if (sources["circuit_index"], sources["block_index"]) != (job["circuit_index"], job["block_index"]):
            raise ValueError("SOURCE_MAP_BLOCK_CHANGED")
        if [row["identity"] for row in cases["cases"]] != job["identities"]:
            raise ValueError("FROZEN_CASE_IDENTITIES_CHANGED")
        restored = {}
        if job["primary"]["merlin_job_run_id"] == job["replay"]["merlin_job_run_id"] or job["primary"]["merlin_trial_id"] == job["replay"]["merlin_trial_id"]:
            raise ValueError("INDEPENDENT_MERLIN_EXECUTIONS_REQUIRED")
        for label in ("primary", "replay"):
            validate_receipt(job[label], job["expected_library_sha256"])
            restored[label], receipt[label] = restore_result(job[label], work, label)
            header = json.loads((restored[label] / "coupled/report.json").read_text())
            if (header["circuit_index"], header["block_index"]) != (job["circuit_index"], job["block_index"]):
                raise ValueError("RESTORED_SOURCE_BLOCK_CHANGED:" + label)
            if header["cases_sha256"] != transport.digest(Path(job["cases"])):
                raise ValueError("RESTORED_SOURCE_CASES_CHANGED:" + label)
        for row in sources["conditions"]:
            row["sources"].update(gpu_primary=str(restored["primary"]),
                                  gpu_replay=str(restored["replay"]),
                                  circuit_validation=job["native_validation"])
        source_map = output / "sources.json"
        transport.write_json(source_map, sources)
        transport.write_json(output / "execution_inputs.json", job)
        block = Path(job["block_root"]) / "block"
        command = [sys.executable, "-B", "-m", "population_dynamics.benchmark_production_block",
                   "--config", job["config"], "--qualification", job["qualification"],
                   "--identity-table", job["identity_table"],
                   "--runtime-qualification", job["runtime_qualification"],
                   "--scenario", "v1_visual", "--circuit-index", str(job["circuit_index"]),
                   "--block-index", str(job["block_index"]),
                   "--inputs", str(source_map), "--output", str(block)]
        environment = dict(os.environ, PYTHONPATH=job["public_root"], PYTHONDONTWRITEBYTECODE="1",
                           CUDA_VISIBLE_DEVICES="", NRNSIM_EXECUTION_BACKEND="merlin",
                           SLURM_JOB_ID="merlin-trial-" + trial,
                           SLURM_CPUS_PER_TASK=str(job["resource_shape"]["cpu"]))
        receipt["command"] = command
        run_cli(command, environment, output / "admission.log")
        report = json.loads((block / "report.json").read_text())
        if (report["status"] != "COMPLETE_QUALIFIED_PRODUCTION_BLOCK"
                or report["complete_episodes"] != 10
                or [row["identity"] for row in report["results"]] != job["identities"]):
            raise ValueError("COMPLETE_FROZEN_PRODUCTION_BLOCK_REQUIRED")
        catalog_path = block / "dataset/dataset.json"
        before = transport.digest(catalog_path)
        reused = json.loads(run_cli(command + ["--resume"], environment, output / "readback.log"))
        catalog = json.loads(catalog_path.read_text())
        rows = [row for shard in catalog["shards"] for row in shard["episodes"]]
        if (reused["status"] != "REUSED_COMMITTED_PRODUCTION_BLOCK"
                or reused["reused_episodes"] != 10 or transport.digest(catalog_path) != before
                or {row["episode_id"] for row in rows} != {row["episode_id"] for row in job["identities"]}
                or len(rows) != 10 or catalog["complete_episodes"] != 10):
            raise ValueError("PRODUCTION_PUBLIC_READBACK_FAILED")
        readback = dict(status="PASS_COMPLETE_PRODUCTION_BLOCK_PUBLIC_REUSE",
                        job_id="merlin-trial-" + trial, consumer="benchmark_production_block CLI --resume",
                        dataset=str(block / "dataset"), catalog_sha256=before,
                        complete_episodes=10, reused_episodes=10, dataset_catalog_unchanged=True,
                        circuit_index=job["circuit_index"], block_index=job["block_index"],
                        source_qualification_sha256=report["source_qualification_sha256"],
                        identity_table_sha256=report["identity_table_sha256"],
                        units=next(iter(catalog["circuits"].values()))["units"],
                        dt_ms=rows[0]["dt_ms"], duration_ms=rows[0]["duration_ticks"] * rows[0]["dt_ms"],
                        total_spike_pairs=sum(row["event_count"] for row in rows),
                        source_simulations=0, gpu_executions_repeated=False,
                        recorded_utc=datetime.now(timezone.utc).isoformat())
        (output / "production_readback").mkdir()
        transport.write_json(output / "production_readback/report.json", readback)
        receipt.update(status="PASS_V1_PRODUCTION_ADMISSION_AND_PUBLIC_REUSE", exit_code=0,
                       admitted_complete_episodes=10, block_report=str(block / "report.json"),
                       readback_report=str(output / "production_readback/report.json"))
    except BaseException as error:
        receipt.update(status="FAILED_V1_PRODUCTION_ADMISSION", exit_code=1,
                       error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        transport.write_json(output / "merlin_terminal.json", receipt)
        if ready:
            try:
                transport.persist(output, work, target, receipt, chunk_bytes=CHUNK_BYTES,
                                  additional_roots=(Path(job["block_root"]) / "block",))
            except BaseException as error:
                receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                               persistence_error_type=type(error).__name__, persistence_error=str(error))
                transport.write_json(output / "merlin_terminal.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "exit_code", "persisted",
                                                "admitted_complete_episodes")}), flush=True)
    return receipt["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.job_json.read_text()), args.job_json.resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())

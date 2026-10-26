"""Freeze and deploy the saved-field MemNO comparison on a CPU Slurm node.

The comparison consumer remains unchanged. Preparation binds existing receipts;
runtime binds the three completed MemNO deliveries to their frozen preparation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from run_population_trajectory_campaign import copy_bound, digest, manifest_rows, software_closure, write_json

SEEDS = (41, 42, 43)
ENTRY = "Libs/scripts/run_memory_field_comparison.py"
BATCH = "Libs/scripts/run_memory_field_comparison.sbatch"
HELPER = "Libs/scripts/run_population_trajectory_campaign.py"
PREPARED = "PREPARED_SOURCE_BOUND_MEMORY_FIELD_COMPARISON"
RESULTS = ("comparison/comparison.json", "comparison/field_comparison_all_seeds.csv",
           "comparison/field_comparison_by_block.csv")


def cpu_placement():
    """Require an allowed CPU allocation, including CPU03 without a GPU."""
    node = os.environ.get("SLURMD_NODENAME")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    allowed = ((node == "cpu04" and partition == "cpu")
               or (node == "cpu03" and partition == "gpu"))
    if (not os.environ.get("SLURM_JOB_ID") or not allowed
            or os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise ValueError("memory field comparison requires CPU04/cpu or CPU03/GPU-partition without GPU")
    return dict(node=node, partition=partition, job_id=os.environ["SLURM_JOB_ID"])


def read_json(path):
    return json.loads(Path(path).read_text())


def bound_row(source, path, declared=None):
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("comparison staging path must be relative")
    source = Path(source)
    if declared is None:
        declared = dict(bytes=source.stat().st_size, sha256=digest(source))
    return dict(source=str(source), path=str(target), bytes=declared["bytes"], sha256=declared["sha256"])


def successful_seed(receipt, seed):
    if (receipt.get("status") != "PASS" or receipt.get("exit_code") != 0
            or receipt.get("training_seed") != seed
            or receipt.get("scientific_claim_qualified") is not False):
        raise ValueError("successful development-only seed delivery required: " + str(seed))
    return manifest_rows(receipt)


def consumed_names(kind):
    """List only the files read by the frozen comparison consumer."""
    if kind == "memory":
        names = ["training/run.json", "evaluation/evaluation.json", "replay/evaluation.json"]
        names += [f"evaluation/train_block_{block:03d}_familiar_{template}_origin_{int(origin / 5):04d}.npz"
                  for block in range(14, 18) for template in range(4) for origin in (1000., 3000., 5000.)]
    elif kind == "population":
        names = ["training/run.json", "forecasts/fitting_field_scale.npz"]
        names += [f"forecasts/train_block_{block:03d}_familiar_{template}_origin_{origin:g}.npz"
                  for block in range(14, 18) for template in range(4) for origin in (1000., 3000., 5000.)]
    else:
        raise ValueError("unknown comparison source: " + kind)
    return names


def seed_inputs(root, seed, kind, memory_preparation_sha256=None, frozen_receipt=None):
    root = Path(root)
    receipt_path = root / "delivery.json"
    receipt = read_json(receipt_path) if frozen_receipt is None else frozen_receipt
    index = successful_seed(receipt, seed)
    if kind == "memory" and (receipt.get("complete_training") is not True
            or receipt.get("replay_performed") is not True
            or receipt.get("preparation_sha256") != memory_preparation_sha256):
        raise ValueError("complete MemNO fit and replay must bind the frozen preparation: " + str(seed))
    names = consumed_names(kind)
    if any(name not in index for name in names):
        raise ValueError("seed delivery lacks a consumed comparison input: " + str(seed))
    rows = [bound_row(root / name, f"{kind}/seed_{seed}/{name}", index[name]) for name in names]
    binding = dict(training_seed=seed, job_id=receipt.get("job_id"),
                   delivery_sha256=digest(receipt_path), consumed_files=len(rows))
    if kind == "memory":
        rows.insert(0, bound_row(receipt_path, f"memory/seed_{seed}/delivery.json"))
    return rows, binding


def prepare(repo, output, memory_campaign, braid_comparison, population_campaign, memory_preparation):
    placement = cpu_placement()
    repo, output, memory_campaign, braid_comparison, population_campaign, memory_preparation = [
        Path(path).resolve() for path in (repo, output, memory_campaign, braid_comparison,
                                         population_campaign, memory_preparation)]
    if output.exists():
        raise FileExistsError("comparison preparation needs a new directory")
    memory_proof = read_json(memory_preparation / "preparation.json")
    if (memory_proof.get("status") != "PREPARED_SOURCE_BOUND_MEMNO"
            or memory_proof.get("training_seeds") != list(SEEDS)):
        raise ValueError("registered source-bound MemNO preparation required")
    fixed = [bound_row(memory_preparation / "preparation.json", "bindings/memory_preparation.json")]
    population_receipts = {}
    for seed in SEEDS:
        source = population_campaign / f"seed_{seed}/delivery.json"
        receipt = read_json(source)
        successful_seed(receipt, seed)
        row = bound_row(source, f"population/seed_{seed}/delivery.json")
        fixed.append(row)
        population_receipts[str(seed)] = row["sha256"]
    prior = read_json(braid_comparison / "delivery.json")
    if prior.get("status") != "PASS" or prior.get("exit_code") != 0:
        raise ValueError("accepted complete BRAID comparison required")
    prior_files = manifest_rows(prior)
    fixed.append(bound_row(braid_comparison / "delivery.json", "braid/delivery.json"))
    for name in ("comparison/comparison.json", "comparison/mean_comparison_all_seeds.csv"):
        if name not in prior_files:
            raise ValueError("BRAID delivery lacks source comparison: " + name)
        fixed.append(bound_row(braid_comparison / name, "braid/" + name, prior_files[name]))
    prior_report = read_json(braid_comparison / "comparison/comparison.json")
    for seed in SEEDS:
        if prior_report["consumed_deliveries"][str(seed)]["population"]["delivery_sha256"] != population_receipts[str(seed)]:
            raise ValueError("prior comparison population delivery differs: " + str(seed))
    names = software_closure(repo, ["memory_field_comparison", "resource_probe"])
    names += [ENTRY, BATCH, HELPER]
    output.mkdir(parents=True)
    code = []
    for name in dict.fromkeys(names):
        row = bound_row(repo / name, name)
        copy_bound(Path(row["source"]), output / name, row)
        code.append({key: row[key] for key in ("path", "bytes", "sha256")})
    for row in fixed:
        copy_bound(Path(row["source"]), output / "inputs" / row["path"], row)
    proof = dict(status=PREPARED, files=code, fixed_inputs=fixed, training_seeds=list(SEEDS),
                 memory_preparation_sha256=fixed[0]["sha256"],
                 sources=dict(memory_campaign=str(memory_campaign), braid_comparison=str(braid_comparison),
                              population_campaign=str(population_campaign)),
                 population_delivery_sha256=population_receipts, placement=placement,
                 scientific_claim_qualified=False, inference_performed=False,
                 recorded_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "preparation.json", proof)
    return dict(status=PREPARED, software_files=len(code), fixed_inputs=len(fixed), training_seeds=list(SEEDS))


def node_probe(work, software, input_bytes):
    spec = importlib.util.spec_from_file_location("memory_comparison_resource_probe",
                                                 software / "Libs/population_dynamics/resource_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    health = module.probe(work / "resources.json", required_memory_gib=4)
    if health["status"] != "PASS":
        raise RuntimeError("comparison resource admission failed: " + str(health["reasons"]))
    if shutil.disk_usage(work).free < input_bytes + 21 * 1024**3:
        raise RuntimeError("comparison scratch lacks declared inputs and1GiB writes above20GiB floor")


def run(preparation, work, destination):
    placement = cpu_placement()
    preparation, work, destination = map(Path, (preparation, work, destination))
    if work.exists() or destination.exists():
        raise FileExistsError("comparison scratch and destination must be new; establish prior outcome before reuse")
    work.mkdir(parents=True)
    record = dict(status="RUNNING", exit_code=2, placement=placement, job_id=placement["job_id"],
                  training_seeds=list(SEEDS), preparation=str(preparation),
                  started_utc=datetime.now(timezone.utc).isoformat(), scientific_claim_qualified=False,
                  inference_performed=False, retraining_performed=False)
    try:
        proof = read_json(preparation / "preparation.json")
        record["preparation_sha256"] = digest(preparation / "preparation.json")
        if proof.get("status") != PREPARED or proof.get("training_seeds") != list(SEEDS):
            raise ValueError("source-bound three-seed comparison preparation required")
        software = work / "software"
        for row in proof["files"]:
            copy_bound(preparation / row["path"], software / row["path"], row)
        rows, seeds = list(proof["fixed_inputs"]), {}
        for row in proof["fixed_inputs"]:
            # The frozen copy and the declared live receipt must both match.
            source = Path(row["source"])
            if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
                raise ValueError("frozen source receipt changed: " + str(source))
        for seed in SEEDS:
            memory_root = Path(proof["sources"]["memory_campaign"]) / f"seed_{seed}"
            memory_rows, memory_binding = seed_inputs(memory_root, seed, "memory", proof["memory_preparation_sha256"])
            population_root = Path(proof["sources"]["population_campaign"]) / f"seed_{seed}"
            receipt = read_json(preparation / f"inputs/population/seed_{seed}/delivery.json")
            population_rows, population_binding = seed_inputs(population_root, seed, "population", frozen_receipt=receipt)
            if population_binding["delivery_sha256"] != proof["population_delivery_sha256"][str(seed)]:
                raise ValueError("frozen population seed receipt changed: " + str(seed))
            rows += memory_rows + population_rows
            seeds[str(seed)] = dict(memory=memory_binding, population=population_binding)
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError("duplicate comparison staging path")
        input_bytes = sum(row["bytes"] for row in rows)
        write_json(work / "input_delivery.json", dict(status="PLANNED", files=rows, input_bytes=input_bytes))
        write_json(work / "runtime_seed_bindings.json", seeds)
        record.update(consumed_deliveries=seeds, input_bytes=input_bytes)
        node_probe(work, software, input_bytes)
        fixed_names = {row["path"] for row in proof["fixed_inputs"]}
        for row in rows:
            source = preparation / "inputs" / row["path"] if row["path"] in fixed_names else Path(row["source"])
            copy_bound(source, work / "inputs" / row["path"], row)
        write_json(work / "input_delivery.json", dict(status="PASS", files=rows, input_bytes=input_bytes))
        node_probe(work, software, 0)
        command = [sys.executable, "-B", "-m", "population_dynamics.memory_field_comparison",
                   "--memory-campaign", str(work / "inputs/memory"),
                   "--braid-comparison", str(work / "inputs/braid"),
                   "--population-campaign", str(work / "inputs/population"),
                   "--output", str(work / "comparison")]
        env = dict(os.environ, PYTHONPATH=str(software / "Libs"), PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TMPDIR=str(work))
        with (work / "comparison.log").open("w") as stream:
            child = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(work / "command.json", dict(command=command, exit_code=child.returncode))
        record.update(command=command, exit_code=child.returncode)
        if child.returncode:
            raise RuntimeError("saved-field comparison failed; inspect comparison.log")
        result = read_json(work / "comparison/comparison.json")
        if (result.get("status") != "COMPLETE_DESCRIPTIVE_FIELD_MEAN_COMPARISON"
                or result.get("score_rows") != 432 or result.get("block_rows") != 36
                or result.get("scientific_claim_qualified") is not False
                or result.get("inference_performed") is not False):
            raise ValueError("complete descriptive comparison result required")
        record.update(status="PASS", score_rows=432, block_rows=36)
    except Exception as error:
        record.update(status="FAILED", exit_code=record["exit_code"] or 2,
                      error=str(error), traceback=traceback.format_exc())
    destination.mkdir(parents=True)
    delivered = []
    for name in (*RESULTS, "comparison.log", "resources.json", "input_delivery.json",
                 "runtime_seed_bindings.json", "command.json"):
        source = work / name
        if source.is_file():
            row = bound_row(source, name)
            copy_bound(source, destination / name, row)
            delivered.append({key: row[key] for key in ("path", "bytes", "sha256")})
    record.update(files=delivered, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(destination / "delivery.json", record)
    (work / "archive_complete").touch()
    return dict(status=record["status"], exit_code=record["exit_code"], delivered_files=len(delivered),
                score_rows=record.get("score_rows"), block_rows=record.get("block_rows"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--preparation", type=Path, required=True)
    for name in ("repo", "memory-campaign", "braid-comparison", "population-campaign", "memory-preparation",
                 "work", "destination"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    required = ("repo", "memory_campaign", "braid_comparison", "population_campaign", "memory_preparation") if args.prepare else ("work", "destination")
    if any(getattr(args, name) is None for name in required):
        parser.error("selected mode requires " + ", ".join("--" + name.replace("_", "-") for name in required))
    result = (prepare(args.repo, args.preparation, args.memory_campaign, args.braid_comparison,
                      args.population_campaign, args.memory_preparation) if args.prepare
              else run(args.preparation, args.work, args.destination))
    print(json.dumps(result))
    raise SystemExit(result.get("exit_code", 0))


if __name__ == "__main__":
    main()

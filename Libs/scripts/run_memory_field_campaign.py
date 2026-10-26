"""Prepare and run the source-bound MemNO complete field development campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from run_population_trajectory_campaign import copy_bound, digest, software_closure, write_json

import yaml

COHORT = "Output/neural_population_dynamics/mechanistic_reference/spatial_connectivity/stimulus_trajectory_cohort/philox_development"
CONFIG = "Libs/configs/neural_population_memory_field.yaml"
EXTERNAL = ("LICENSE", "source.json", "ldns/__init__.py", "ldns/networks/__init__.py",
            "ldns/networks/s4.py", "ldns/networks/blocks.py", "ldns/networks/count_wrapper.py")


def prepare(repo, output, device=None):
    if output.exists():
        raise FileExistsError("MemNO preparation needs a new directory")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("MemNO preparation requires compute-node Slurm")
    files = software_closure(repo, ["memory_field_training", "memory_field_evaluation", "resource_probe"])
    files += [CONFIG, "Libs/scripts/run_memory_field_campaign.py",
              "Libs/scripts/run_memory_field_campaign.sbatch",
              "Libs/scripts/run_population_trajectory_campaign.py"]
    files += ["Libs/external/LDNS/" + name for name in EXTERNAL]
    output.mkdir(parents=True)
    code = []
    for name in files:
        source = repo / name
        row = dict(path=name, bytes=source.stat().st_size, sha256=digest(source))
        copy_bound(source, output / name, row)
        if name == CONFIG and device is not None:
            config = yaml.safe_load((output / name).read_text())
            config["device"] = device
            (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
            row.update(bytes=(output / name).stat().st_size, sha256=digest(output / name))
        code.append(row)
    acquisition = repo / COHORT / "train"
    fields = repo / COHORT / "train_fields"
    design = json.loads((acquisition / "design.json").read_text())
    inventory = []
    entries = [(acquisition / "design.json", "train/design.json"),
               (acquisition / "acquisition.json", "train/acquisition.json"),
               (fields / "dataset.json", "train_fields/dataset.json"),
               (fields / "episodes.csv", "train_fields/episodes.csv")]
    for case in design["cases"]:
        case_id = case["case_id"]
        if Path(case_id).name != case_id:
            raise ValueError("source case identity must be a single path component")
        entries += [(acquisition / case_id / name, f"train/{case_id}/{name}")
                    for name in ("report.json", "spikes.npz")]
        entries += [(fields / case_id / name, f"train_fields/{case_id}/{name}")
                    for name in ("summary.json", "spatial_fields.npz")]
    for source, target in entries:
        inventory.append(dict(source=str(source), path=target,
                              bytes=source.stat().st_size, sha256=digest(source)))
    write_json(output / "preparation.json", dict(status="PREPARED_SOURCE_BOUND_MEMNO",
        files=code, inputs=inventory, training_seeds=[41, 42, 43],
        input_bytes=sum(row["bytes"] for row in inventory),
        device_override=device, scientific_claim_qualified=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))
    return dict(files=len(code), inputs=len(inventory), input_bytes=sum(row["bytes"] for row in inventory))


def node_probe(work, software, memory):
    path = software / "Libs/population_dynamics/resource_probe.py"
    spec = importlib.util.spec_from_file_location("memory_resource_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    report = probe.probe(work / "resources.json", memory)
    if report["status"] != "PASS":
        raise RuntimeError("MemNO node resource admission failed: " + str(report["reasons"]))
    reserve = 20 * 1024**3
    if shutil.disk_usage(work).free < reserve + 4 * 1024**3:
        raise RuntimeError("MemNO work disk lacks 4GiB writes above20GiB floor")


def supervised_command(command, work, env, software, log_path):
    """Stop the child process group when node storage/memory/I/O gates fail."""
    with log_path.open("w") as stream:
        child = subprocess.Popen(command, env=env, cwd=work, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    node_probe(work, software, 8)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    write_json(work / "child_stop.json", dict(status="STOP_REQUESTED_CHILD_UNRESOLVED",
                                                              pid=child.pid, command=command))
            raise
    return child.returncode


def run(preparation, work, destination, seed, stop_after_epochs=None, resume=False,
        evaluate_partial=False, replay=False):
    if seed not in (41, 42, 43) or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("MemNO requires registered training seed in Slurm")
    proof = json.loads((preparation / "preparation.json").read_text())
    if proof["status"] != "PREPARED_SOURCE_BOUND_MEMNO":
        raise ValueError("MemNO source-bound preparation required")
    if work.exists() and not resume:
        raise FileExistsError("MemNO scratch exists; establish its outcome before reuse")
    work.mkdir(parents=True, exist_ok=resume)
    software = work / "software"
    for row in proof["files"]:
        copy_bound(preparation / row["path"], software / row["path"], row)
    node_probe(work, software, 8)
    staging = []
    for row in proof["inputs"]:
        target = work / "inputs" / row["path"]
        if resume:
            if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise ValueError("MemNO retained input changed")
        else:
            copy_bound(Path(row["source"]), target, row)
        staging.append({key: row[key] for key in ("path", "bytes", "sha256")})
    write_json(work / "input_delivery.json", dict(status="PASS", files=staging))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(software / "Libs") + ":" + str(software / "Libs/external/LDNS"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               NVIDIA_TF32_OVERRIDE="0", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    scratch = work / "scratch"
    scratch.mkdir(exist_ok=True)
    env["TMPDIR"] = str(scratch)
    base = ["--acquisition", str(work / "inputs/train"), "--fields", str(work / "inputs/train_fields")]
    train = [sys.executable, "-B", "-m", "population_dynamics.memory_field_training", *base,
             "--config", str(software / CONFIG), "--output", str(work / "training"), "--seed", str(seed)]
    if resume:
        train.append("--resume")
    if stop_after_epochs:
        train.extend(["--stop-after-epochs", str(stop_after_epochs)])
    commands = [train]
    if stop_after_epochs is None or evaluate_partial:
        evaluate = [sys.executable, "-B", "-m", "population_dynamics.memory_field_evaluation", *base,
                    "--training", str(work / "training"), "--output", str(work / "evaluation")]
        if evaluate_partial:
            evaluate.append("--allow-partial")
        commands.append(evaluate)
        if replay:
            replay_command = evaluate.copy()
            replay_command[replay_command.index("--output") + 1] = str(work / "replay")
            replay_command.extend(["--replay-from", str(work / "evaluation")])
            commands.append(replay_command)
    for index, command in enumerate(commands):
        node_probe(work, software, 8)
        code = supervised_command(command, work, env, software, work / f"command_{index}.log")
        write_json(work / f"command_{index}.json", dict(command=command, exit_code=code))
        if code:
            raise RuntimeError(f"MemNO command{index} failed; scratch retained: {work}")
    # Copy only declared products. Transient staged inputs and code stay local.
    if destination.exists():
        raise FileExistsError("MemNO delivery must preserve existing artifacts")
    destination.mkdir(parents=True)
    products = [work / name for name in ("training", "evaluation", "replay") if (work / name).exists()]
    products += [path for path in work.iterdir() if path.is_file()]
    delivery = []
    for product in products:
        paths = product.rglob("*") if product.is_dir() else [product]
        for path in paths:
            if path.is_file():
                name = str(path.relative_to(work))
                row = dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
                copy_bound(path, destination / name, row)
                delivery.append(row)
    write_json(destination / "delivery.json", dict(status="PASS", exit_code=0,
        training_seed=seed, job_id=os.environ["SLURM_JOB_ID"], files=delivery,
        preparation_sha256=digest(preparation / "preparation.json"),
        complete_training=stop_after_epochs is None, replay_performed=replay,
        scientific_claim_qualified=False))
    (work / "archive_complete").touch()
    return dict(status="PASS", training_seed=seed, delivered_files=len(delivery))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), help="prospective preparation device override")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-partial", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.repo, args.preparation, args.device)
    else:
        result = run(args.preparation, args.work, args.destination, args.seed,
                     args.stop_after_epochs, args.resume, args.evaluate_partial, args.replay)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

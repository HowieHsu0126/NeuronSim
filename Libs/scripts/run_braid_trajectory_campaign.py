"""Run one fixed BRAID development seed on prepared CPU04 inputs.

The source snapshot, runtime and complete native inputs are prepared and verified
before submission. This caller trains, evaluates and independently replays using
fresh processes, then publishes only the declared deliverable files.
"""
from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


WORK = Path("/tmp/neuronsim/neuronsim-braid-matched-development-20261011")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stage_recovery(parent, cell, seed, preparation_digest):
    """Restore the last committed fit without changing the prepared experiment."""
    proof = json.loads((parent / "delivery.json").read_text())
    if (proof.get("status") != "RESOURCE_BLOCKED" or proof.get("training_seed") != seed
            or proof.get("preparation_sha256") != preparation_digest
            or proof.get("scientific_claim_qualified") is not False):
        raise ValueError("same-seed source-bound BRAID resource interruption required")
    rows = {row["path"]: row for row in proof["files"]}
    if len(rows) != len(proof["files"]):
        raise ValueError("duplicate BRAID recovery artifact")
    for name in ("training/committed.json", "training/run.json", "fit_receipt.json"):
        if name not in rows:
            raise ValueError("BRAID recovery lacks committed fit evidence: " + name)
    for name, row in rows.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("BRAID recovery artifact outside delivery")
        if not name.startswith("training/") and name != "fit_receipt.json":
            continue
        source = parent / name
        if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
            raise ValueError("BRAID recovery artifact changed: " + name)
        target = cell / name if name.startswith("training/") else cell / "parent_fit_receipt.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != row["sha256"]:
            raise ValueError("BRAID recovery staging differs: " + name)
    run = json.loads((cell / "training/run.json").read_text())
    receipt = json.loads((cell / "parent_fit_receipt.json").read_text())
    if (run["training_seed"] != seed or run["status"] != "RUNNING"
            or not 0 < run["completed_epochs"] < 20 or run["stop_reason"] is not None
            or receipt["status"] != "RESOURCE_BLOCKED" or receipt["exit_code"] != -15):
        raise ValueError("incomplete committed BRAID resource-stopped fit required")
    shutil.copy2(parent / "delivery.json", cell / "recovery_delivery.json")
    return dict(parent=str(parent), parent_delivery_sha256=digest(parent / "delivery.json"),
                committed_epochs=run["completed_epochs"])


def run(seed, recovery_from=None):
    if seed not in [41, 42, 43]: raise ValueError("registered training seed required")
    if os.environ.get("SLURMD_NODENAME") != "cpu04" or os.environ.get("SLURM_JOB_PARTITION") != "cpu":
        raise ValueError("BRAID prepared runtime requires CPU04/cpu")
    preparation = json.loads((WORK / "preparation.json").read_text())
    for name, expected in preparation["files"].items():
        path = WORK / name
        if digest(path) != expected: raise ValueError("prepared input changed:" + name)
    cell = WORK / (f"seed_{seed}" if recovery_from is None else
                   f"seed_{seed}_recovery_{os.environ['SLURM_JOB_ID']}")
    if cell.exists(): raise FileExistsError("new campaign cell required")
    cell.mkdir()
    recovery = None
    if recovery_from is not None:
        recovery = stage_recovery(Path(recovery_from), cell, seed, digest(WORK / "preparation.json"))
    python = str(WORK / "runtime/bin/python")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(WORK / "source"), str(WORK / "source/Libs/external")]),
               TMPDIR=str(WORK / "scratch"), MPLCONFIGDIR=str(WORK / "scratch/mpl"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    common = ["--acquisition", str(WORK / "inputs/train"), "--fields", str(WORK / "inputs/train_fields"),
              "--config", str(WORK / "source/Libs/configs/neural_population_braid_trajectory.yaml"), "--seed", str(seed)]
    commands = [
        ("fit", [python, "-B", "-m", "Libs.population_dynamics.braid_trajectory_training", *common,
                 "--output", str(cell / "training")]),
        ("evaluate", [python, "-B", "-m", "Libs.population_dynamics.braid_trajectory_evaluation", *common,
                      "--training", str(cell / "training"), "--fitting-scale", str(WORK / "fitting_field_scale.npz"),
                      "--output", str(cell / "forecasts")]),
        ("replay", [python, "-B", "-m", "Libs.population_dynamics.braid_trajectory_evaluation", *common,
                    "--training", str(cell / "training"), "--fitting-scale", str(WORK / "fitting_field_scale.npz"),
                    "--output", str(cell / "forecasts"), "--replay"]),
    ]
    if recovery is not None:
        commands[0][1].append("--resume")
    for name, command in commands:
        spec = importlib.util.spec_from_file_location("braid_resource_probe", WORK / "source/Libs/population_dynamics/resource_probe.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        health = module.probe(cell / f"{name}_preflight.json", required_memory_gib=16)
        if health["status"] != "PASS": raise RuntimeError("BRAID resource admission failed")
        start = time.monotonic()
        with (cell / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=WORK / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
            def terminate_child(signum, frame):
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                write(cell / f"{name}_receipt.json", dict(status="INTERRUPTED", signal=signum,
                      exit_code=child.returncode, command=command, seconds=time.monotonic() - start))
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, terminate_child)
            signal.signal(signal.SIGINT, terminate_child)
            while child.poll() is None:
                time.sleep(30)
                health = module.probe(cell / f"{name}_watchdog.json", required_memory_gib=4)
                if health["status"] != "PASS":
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    write(cell / f"{name}_receipt.json", dict(status="RESOURCE_BLOCKED", exit_code=child.returncode,
                          command=command, seconds=time.monotonic() - start, resource=health))
                    raise RuntimeError("BRAID resource watchdog stopped child")
        write(cell / f"{name}_receipt.json", dict(status="PASS" if child.returncode == 0 else "FAILED",
              exit_code=child.returncode, command=command, seconds=time.monotonic() - start))
        if child.returncode: raise RuntimeError("BRAID phase failed:" + name)
    root = Path(preparation["output_root"]) / f"seed_{seed}"
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    files = []
    for path in sorted(cell.rglob("*")):
        if not path.is_file(): continue
        relative = path.relative_to(cell); destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
        source_digest = digest(path)
        if digest(destination) != source_digest: raise RuntimeError("BRAID delivery changed:" + str(relative))
        files.append(dict(path=str(relative), bytes=destination.stat().st_size, sha256=source_digest))
    write(root / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
          preparation_sha256=digest(WORK / "preparation.json"), files=files,
          resource_recovery=recovery,
          source_unit_count=100000, complete_fit_episodes=56, complete_validation_episodes=16,
          scientific_claim_qualified=False, delivered_utc=datetime.now(timezone.utc).isoformat()))
    write(cell / "delivery_complete.json", dict(output=str(root)))
    print(json.dumps(dict(status="PASS", seed=seed, output=str(root))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    parser.add_argument("--recovery-from", type=Path)
    args = parser.parse_args()
    run(args.seed, args.recovery_from)


if __name__ == "__main__": main()

"""Stage delivered mean-comparison inputs on CPU04 and archive the result."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


def _stage_comparison_cell(source, kind, destination, origins, sha256_file):
    """Copy the saved inputs consumed by comparison and training-cost readers."""
    names = ["forecasts/evaluation.json", "forecasts/scores.csv", "training/run.json"]
    if kind == "braid":
        names += ["forecasts/replay.json", "forecasts/origin_costs.csv", "training/committed.json"]
    else:
        names += ["replay.json", "forecasts/fitting_field_scale.npz"]
        # run() makes the authenticated consumer tree importable before staging
        # data. Use that consumer's profile identity for all prior input paths.
        from Libs.population_dynamics.population_forecast_analysis import _is_prior_refit

        training_run = source.read("training/run.json")
        first_attempt = 0
        if _is_prior_refit(training_run):
            names += ["training/best.pt", "parent_training/run.json", "parent_training/best.pt",
                      "prerequisites/prior_parent.json"]
            current = training_run["binding"].get("observation_implementation_sha256")
            parent = training_run["refit_initialization"]["parent_binding"].get("observation_implementation_sha256")
            if current != parent:
                names.append("Libs/population_dynamics/refractory_observation.py")
        if training_run.get("continuation") is not None:
            names += ["prerequisites/budget_parent/delivery.json",
                      "prerequisites/budget_parent/training/run.json"]
            parent_run = source.read("prerequisites/budget_parent/training/run.json")
            # The cost reader authenticates prior parent attempts at the original
            # parent path. Only subsequent child attempts belong in this stage.
            first_attempt = parent_run.get("attempt", 0) + 1
        names += [f"training/attempts/attempt_{i:03d}.json"
                  for i in range(first_attempt, training_run.get("attempt", 0))]
    names += [f"forecasts/train_block_{block:03d}_familiar_{template}_origin_{origin:g}.npz"
              for block in range(14, 18) for template in range(4) for origin in origins]
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.path(name), target)
        if sha256_file(target) != source.files[name]["sha256"]:
            raise ValueError("staged comparison input changed: " + name)
    shutil.copy2(source.root / "delivery.json", destination / "delivery.json")


def cpu_placement():
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    allowed = (node == "cpu04" and partition == "cpu") or (node == "cpu03" and partition == "gpu")
    if (not os.environ.get("SLURM_JOB_ID") or not allowed
            or os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")):
        raise ValueError("BRAID comparison requires CPU04/cpu or CPU03/gpu without a GPU")


def run(args):
    cpu_placement()
    output = args.output / ("run_" + os.environ["SLURM_JOB_ID"])
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="neuronsim-braid-comparison-", dir="/tmp/neuronsim") as folder:
        work = Path(folder)
        record = dict(status="RUNNING", job_id=os.environ["SLURM_JOB_ID"],
                      started_utc=datetime.now(timezone.utc).isoformat(), scientific_claim_qualified=False)
        try:
            pending, seen = ["braid_population_comparison", "resource_probe"], set()
            while pending:
                name = pending.pop()
                if name in seen:
                    continue
                seen.add(name)
                source = args.repo / "Libs/population_dynamics" / (name + ".py")
                target = work / "Libs/population_dynamics" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                for node in ast.walk(ast.parse(source.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level == 1:
                        pending.extend([node.module] if node.module else [a.name for a in node.names])
            for name in ("Libs/__init__.py", "Libs/population_dynamics/__init__.py"):
                shutil.copy2(args.repo / name, work / name)
            sys.path.insert(0, str(work))
            from Libs.population_dynamics.braid_population_comparison import DeliveredCell, ORIGINS
            from Libs.population_dynamics.resource_probe import probe
            from Libs.population_dynamics.provenance import sha256_file
            health = probe(work / "preflight.json", required_memory_gib=4)
            if health["status"] != "PASS":
                raise RuntimeError("mean comparison resource admission failed")
            for kind, campaign in (("braid", args.braid_campaign), ("population", args.population_campaign)):
                for seed in (41, 42, 43):
                    source = DeliveredCell(campaign / f"seed_{seed}", seed)
                    destination = work / kind / f"seed_{seed}"
                    _stage_comparison_cell(source, kind, destination, ORIGINS, sha256_file)
            command = [sys.executable, "-B", "-m", "Libs.population_dynamics.braid_population_comparison",
                       "--braid-campaign", "braid", "--population-campaign", "population", "--output", "comparison"]
            env = dict(os.environ, PYTHONPATH=str(work), PYTHONDONTWRITEBYTECODE="1",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TMPDIR=str(work))
            with (work / "comparison.log").open("w") as stream:
                child = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record.update(command=command, exit_code=child.returncode)
            if child.returncode:
                raise RuntimeError("mean comparison failed; inspect comparison.log")
            record["status"] = "PASS"
        except Exception as error:
            record.update(status="FAILED", exit_code=2, error=str(error), traceback=traceback.format_exc())
        output.mkdir(parents=True)
        files = []
        for name in ("Libs", "comparison", "comparison.log", "preflight.json"):
            root = work / name
            sources = root.rglob("*") if root.is_dir() else [root]
            for source in sources:
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                target = output / source.relative_to(work)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                # Use the staged utility for the archive that this caller consumes.
                from Libs.population_dynamics.provenance import sha256_file
                digest = sha256_file(source)
                if sha256_file(target) != digest:
                    raise ValueError("mean comparison archive changed")
                files.append(dict(path=str(source.relative_to(work)), bytes=source.stat().st_size, sha256=digest))
        record.update(files=files, finished_utc=datetime.now(timezone.utc).isoformat(),
                      braid_campaign=str(args.braid_campaign), population_campaign=str(args.population_campaign))
        (output / "delivery.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "files"}))
        return record["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "braid-campaign", "population-campaign", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()

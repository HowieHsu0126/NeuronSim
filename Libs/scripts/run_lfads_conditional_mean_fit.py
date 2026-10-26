"""Fit the declared LFADS template mean from an already verified fitting cache.

Run inside a CPU04 Slurm allocation. Native resource admission and supervision
reuse the campaign guard; source data and neural checkpoints are never changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "paired-analysis", "software", "guard", "config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--network-budget-epochs", type=int)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    if (os.environ.get("SLURMD_NODENAME") != "cpu04" or not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("CPU04_CPU_ONLY_SLURM_ALLOCATION_REQUIRED")
    if args.output.exists():
        raise FileExistsError(args.output)
    record = dict(status="RUNNING", job_id=os.environ["SLURM_JOB_ID"],
        started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        cache=str(args.cache), software=str(args.software),
        software_delivery_sha256=digest(args.software / "delivery.json"),
        new_posterior_draws=0, neural_training_updates=0, scientific_claim_qualified=False,
        formal_test_accessed=False, production_episodes=0)
    code = 1
    with tempfile.TemporaryDirectory(prefix="neuronsim-controller-mean-" + record["job_id"] + ".", dir="/tmp") as runtime:
        work = Path(runtime)
        (work / "scratch").mkdir()
        (work / "phases").mkdir()
        try:
            software = json.loads((args.software / "delivery.json").read_text())
            accepted = software["status"] == "PASS" or (
                software["status"] == "ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS"
                and software.get("validation_mode") == "conditional_mean_matrix"
                and software.get("requested_test_device") == "cpu")
            if not accepted or software["exit_code"] != 0:
                raise ValueError("PASSED_CURRENT_SOFTWARE_RECEIPT_REQUIRED")
            rows = {row["path"]: row for row in software["files"]}
            def copy_software(name):
                row = rows[name]
                target = work / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.software / name, target)
                if digest(target) != row["sha256"]:
                    raise ValueError("VALIDATED_SOURCE_CHANGED:" + name)
            copy_software("Libs/population_dynamics/__init__.py")
            copy_software("Libs/population_dynamics/resource_probe.py")
            sys.path.insert(0, str(work / "Libs"))
            from population_dynamics.resource_probe import probe
            admission = probe(work / "admission.json", required_memory_gib=8)
            if admission["status"] != "PASS":
                record.update(status="RESOURCE_BLOCKED", reasons=admission["reasons"])
                raise RuntimeError("MEAN_FIT_RESOURCE_ADMISSION_FAILED")
            for name in rows:
                if name.startswith("Libs/"):
                    copy_software(name)
            config = work / "Libs/configs" / args.config.name
            shutil.copy2(args.config, config)
            record["prospective_protocol_sha256"] = digest(config)
            guard = work / "run_guarded.py"
            guard.write_text(args.guard.read_text().replace("required_memory_gib=1", "required_memory_gib=8"))
            shutil.copy2(Path(__file__), work / "launcher.py")
            cache = work / "inputs/cache"
            cache.mkdir(parents=True)
            names = ["posterior_cache.json", "prior_parameters.npz"]
            names += [f"train_block_{b:03d}_familiar_{t}.npz" for b in range(14) for t in range(4)]
            for name in names:
                shutil.copy2(args.cache / name, cache / name)
            paired = work / "inputs/paired"
            (paired / "analysis").mkdir(parents=True)
            for name in ("delivery.json", "analysis/analysis.json"):
                shutil.copy2(args.paired_analysis / name, paired / name)
            command = [sys.executable, "-B", "-m", "population_dynamics.lfads_conditional_prior",
                "--cache", str(cache), "--paired-analysis", str(paired),
                "--config", str(config), "--output", str(work / "mean")]
            if args.training_seed is not None:
                command += ["--training-seed", str(args.training_seed)]
            if args.network_budget_epochs is not None:
                command += ["--network-budget-epochs", str(args.network_budget_epochs)]
            if args.training_run is not None:
                training_run = work / "inputs/training/run.json"
                training_run.parent.mkdir(parents=True)
                shutil.copy2(args.training_run, training_run)
                record["training_run"] = dict(path=str(args.training_run), sha256=digest(training_run))
                command += ["--training-run", str(training_run)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.pathsep.join([str(work), str(work / "Libs")]),
                TMPDIR=str(work / "scratch"), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([sys.executable, "-B", str(guard), "mean_fit", *command], cwd=work, env=env)
            code = result.returncode
            record.update(status="FITTED_TEMPLATE_TIME_CONTROLLER_MEAN" if code == 0 else "FAILED", exit_code=code)
            if (work / "mean/mean_prior.json").exists():
                record["mean_fit"] = json.loads((work / "mean/mean_prior.json").read_text())
        except Exception as error:
            record.update(error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
            if record["status"] == "RUNNING":
                record["status"] = "FAILED"
        finally:
            record["completed_utc"] = datetime.now(timezone.utc).isoformat()
            args.output.mkdir(parents=True, exist_ok=False)
            files = []
            selected = [work / name for name in ("admission.json", "launcher.py", "run_guarded.py")]
            selected += [work / "Libs/configs/neural_population_lfads_conditional_prior.yaml"]
            if args.config.name != "neural_population_lfads_conditional_prior.yaml":
                selected.append(work / "Libs/configs" / args.config.name)
            selected.append(work / "inputs/training/run.json")
            for name in ("mean", "phases"):
                selected += [p for p in (work / name).rglob("*") if p.is_file()]
            for source in selected:
                if not source.is_file():
                    continue
                relative = source.relative_to(work)
                target = args.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                checksum = digest(source)
                if digest(target) != checksum:
                    raise ValueError("MEAN_DELIVERY_CHANGED:" + str(relative))
                files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
            (args.output / "delivery.json").write_text(json.dumps(dict(record, files=files), indent=2) + "\n")
            print(json.dumps({k:v for k,v in record.items() if k != "mean_fit"}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

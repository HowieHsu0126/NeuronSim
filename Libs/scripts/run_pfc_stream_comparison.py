"""Compare completed PFC source archives through a validated readout environment."""
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
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--reference-job", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (os.environ.get("SLURMD_NODENAME") not in {"cpu04", "cpu05"}
            or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("AUTHORIZED_CPU_ONLY_SLURM_ALLOCATION_REQUIRED")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.reference_job <= 0:
        raise ValueError("POSITIVE_REFERENCE_JOB_REQUIRED")
    receipt = json.loads((args.readout / "delivery.json").read_text())
    if receipt.get("status") != "ARCHIVED_PFC_SOURCE_READOUTS" or receipt.get("exit_code") != 0:
        raise ValueError("COMPLETED_SOURCE_READOUT_REQUIRED")
    entries = {row["path"]: row for row in receipt["files"]}
    if len(entries) != len(receipt["files"]):
        raise ValueError("DUPLICATE_READOUT_DELIVERY_PATH")
    work = Path(tempfile.mkdtemp(prefix="neuronsim-pfc-comparison-" + os.environ["SLURM_JOB_ID"] + ".", dir="/tmp"))
    (work / "scratch").mkdir()
    (work / "phases").mkdir()
    record = dict(status="PREPARING", job_id=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"],
        runtime=str(work), readout_archive=str(args.readout),
        readout_delivery_sha256=digest(args.readout / "delivery.json"),
        started_utc=datetime.now(timezone.utc).isoformat(), commands=[], python=sys.version,
        new_source_simulations=0, production_episodes=0, scientific_fidelity_qualified=False)
    code = 1
    try:
        names = [name for name in entries if name.startswith("Libs/")]
        names += ["config.yaml", "stage_sources.py", "run_guarded.py", "input_origins.json"]
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("READOUT_PATH_INVALID")
            row = entries[name]
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.readout / relative, target)
            if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise ValueError("VALIDATED_READOUT_INPUT_CHANGED:" + name)
        selected = json.loads((work / "input_origins.json").read_text())
        if len(selected) != 1 or selected[0]["job_id"] == args.reference_job:
            raise ValueError("ONE_DISTINCT_CANDIDATE_AND_REFERENCE_REQUIRED")
        candidate = selected[0]
        shutil.copy2(work / "input_origins.json", work / "readout_origins.json")
        source_root = Path(__file__).resolve().parents[1] / "population_dynamics"
        record["implementation_sha256"] = {}
        for name in ["pfc_stream_comparison.py", "test_pfc_stream_comparison.py"]:
            target = work / "Libs/population_dynamics" / name
            shutil.copy2(source_root / name, target)
            record["implementation_sha256"][name] = digest(target)
        shutil.copy2(Path(__file__), work / "launcher.py")
        env = dict(os.environ, PYTHONPATH=str(work / "Libs"), TMPDIR=str(work / "scratch"),
            PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NRNSIM_PFC_READOUT_MODE="phases",
            NRNSIM_PFC_SOURCE_JOBS=f"{candidate['job_id']},{args.reference_job}")

        def run(phase, command):
            row = dict(phase=phase, command=command)
            record["commands"].append(row)
            result = subprocess.run([sys.executable, "-B", str(work / "run_guarded.py"), phase, *command],
                                    cwd=work, env=env)
            row["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(phase + " failed:" + str(result.returncode))

        run("tests", [sys.executable, "-B", "-m", "unittest", "-v",
                      "population_dynamics.test_pfc_stream_comparison"])
        run("staging", [sys.executable, "-B", str(work / "stage_sources.py")])
        origins = json.loads((work / "input_origins.json").read_text())
        signature = lambda row: (row["job_id"], row["case_id"], row["delivery_sha256"])
        if len(origins) != 2 or signature(origins[0]) != signature(candidate):
            raise ValueError("CANDIDATE_CHANGED_AFTER_VERIFIED_READOUT")
        if origins[1]["job_id"] != args.reference_job:
            raise ValueError("REFERENCE_SELECTION_CHANGED")
        run("comparison", [sys.executable, "-B", "-m", "population_dynamics.pfc_stream_comparison",
            "--candidate", origins[0]["staged"], "--reference", origins[1]["staged"],
            "--config", str(work / "config.yaml"), "--output", str(work / "analysis"),
            "--input-origins", str(work / "input_origins.json")])
        report = json.loads((work / "analysis/report.json").read_text())
        if (report.get("status") not in {"COMPLETE_RECORDED_ARRAYS_IDENTICAL", "COMPLETE_RECORDED_ARRAYS_DIFFERENT"}
                or report.get("comparison_completed") is not True):
            raise ValueError("RECORDED_ARRAY_COMPARISON_NOT_COMPLETE")
        record.update(status="ARCHIVED_PFC_STREAM_COMPARISON", comparison=report, source_origins=origins)
        code = 0
    except Exception as error:
        record.update(status="FAILED", error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    record.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
    (work / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
    # Keep compute-local evidence if durable archival fails.
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for source in sorted(work.rglob("*")):
        relative = source.relative_to(work)
        if not source.is_file() or set(relative.parts) & {"inputs", "scratch", "__pycache__"}:
            continue
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksum = digest(source)
        if digest(target) != checksum:
            raise ValueError("COMPARISON_ARCHIVE_COPY_MISMATCH:" + str(relative))
        files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
    (args.output / "delivery.json").write_text(json.dumps(dict(record, files=files), indent=2) + "\n")
    shutil.rmtree(work)
    print(json.dumps({key: record.get(key) for key in ["status", "job_id", "exit_code", "error"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

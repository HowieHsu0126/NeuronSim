#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_CountBoundary
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-count-boundary-%j.out
#SBATCH --error=/tmp/neuronsim-count-boundary-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_BOUNDARY_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_BOUNDARY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
import sys

repo = Path(os.environ["NEURONSIM_PROJECT_ROOT"]).resolve()
job = os.environ["SLURM_JOB_ID"]
assert socket.gethostname().split(".")[0] == "cpu04" and not os.environ.get("SLURM_JOB_GPUS")
family = Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement")
probe_relative = family / "threshold_branch_probe/execution/run_602022"
continuation_relative = family / "spike_branch_continuation/execution/run_602171"
probe_digest = "0028178795cca3f0255fcacc45adf6a21baf9f295e22aadc0d79fa8e3839ef1b"
continuation_digest = "70577629a08efc031d6576be2b612a2464f99ded124b65f9407520f71d1f7ab0"
archive = repo / family / "count_boundary_decomposition" / ("run_" + job)
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix="neuronsim-count-boundary-" + job + ".", dir="/tmp"))
(work / "scratch").mkdir()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(work / "Libs"))), TMPDIR=str(work / "scratch"))
inputs, commands = [], []
started, status, exit_code, error, delivered = time.perf_counter(), "FAILED_PFC_COUNT_BOUNDARY_DECOMPOSITION", 1, None, False

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def copy(source, relative, expected=None):
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    actual = digest(target)
    if expected is not None and actual != expected:
        raise RuntimeError("BOUNDARY_STAGED_INPUT_MISMATCH:" + str(relative))
    inputs.append(dict(source=str(source), path=str(relative), sha256=actual))

def stage_archive(relative, expected, names):
    copy(repo / relative / "delivery.json", relative / "delivery.json", expected)
    delivery = json.loads((work / relative / "delivery.json").read_text())
    assert delivery["verified"]
    manifest = {row["path"]: row for row in delivery["files"]}
    for name in names:
        copy(repo / relative / name, relative / name, manifest[name]["sha256"])
    return manifest

def run_phase(name, command):
    begin = time.perf_counter()
    with (work / (name + ".stdout")).open("w") as out, (work / (name + ".stderr")).open("w") as err:
        result = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err)
    commands.append(dict(phase=name, command=command, exit_code=result.returncode, wall_seconds=time.perf_counter() - begin))
    print(json.dumps(commands[-1]), flush=True)
    if result.returncode:
        raise RuntimeError("BOUNDARY_" + name.upper() + "_FAILED:" + str(result.returncode))

try:
    manifest = stage_archive(continuation_relative, continuation_digest,
        ("execution/report.json", "prepared/plan.json", "prepared/original_events.npz", "execution/counterfactual_raw.npz"))
    for name, entry in manifest.items():
        if Path(name).parts[0] == "Libs" and Path(name).suffix == ".py":
            copy(repo / continuation_relative / name, Path(name), entry["sha256"])
    module = Path("Libs/population_dynamics/pfc_count_boundary_decomposition.py")
    test = Path("Libs/population_dynamics/test_pfc_count_boundary_decomposition.py")
    for name in (module, test):
        copy(repo / name, name)
    copy(Path(os.environ["NEURONSIM_BOUNDARY_BATCH"]), Path("launcher.sbatch"))
    sys.path.insert(0, str(work))
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work / "resource_preflight.json", 4)
    if health["status"] != "PASS":
        raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(health["reasons"]))
    stage_archive(probe_relative, probe_digest,
        ("execution/report.json", "prepared/plan.json", "prepared/reversed_events.npz"))
    plan = json.loads((work / probe_relative / "prepared/plan.json").read_text())
    reverse = plan["source_bindings"]["reversed"]
    case = reverse["case_relative_path"]
    stage_archive(Path(reverse["archive"]), reverse["delivery_sha256"], (case + "/raw.npz", case + "/report.json"))
    run_phase("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test),
        "--basetemp=" + str(work / "test_tmp"), "--junitxml=" + str(work / "tests.xml")])
    run_phase("analysis", [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_count_boundary_decomposition",
        "--project-root", str(work), "--probe-archive", str(work / probe_relative), "--probe-delivery-sha256", probe_digest,
        "--continuation-archive", str(work / continuation_relative), "--continuation-delivery-sha256", continuation_digest,
        "--output", str(work / "analysis")])
    assert json.loads((work / "analysis/report.json").read_text())["status"] == "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION"
    status, exit_code = "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION", 0
except Exception as exc:
    error = dict(type=type(exc).__name__, message=str(exc))
    traceback.print_exc()
finally:
    (work / "report.json").write_text(json.dumps(dict(status=status, exit_code=exit_code, error=error,
        job_id=job, node=socket.gethostname(), inputs=inputs, commands=commands, source_simulations=0,
        new_full_episodes=0, new_independent_blocks=0, source_probability_bound_established=False,
        confirmation_execution_released=False, production_episodes=0,
        wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
    try:
        archive.mkdir(parents=True, exist_ok=False)
        files = []
        for source in sorted(work.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(work)
            if relative.parts[0] in ("Output", "scratch", "test_tmp") or "__pycache__" in relative.parts:
                continue
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(source)
            if digest(target) != checksum:
                raise RuntimeError("BOUNDARY_COPYBACK_MISMATCH:" + str(relative))
            files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
        (archive / "delivery.json").write_text(json.dumps(dict(status=status, job_id=job, verified=True, files=files), indent=2) + "\n")
        delivered = True
        print(json.dumps(dict(status=status, archive=str(archive), files=len(files), exit_code=exit_code)), flush=True)
    finally:
        if delivered:
            shutil.rmtree(work)
        else:
            print(json.dumps(dict(status="COPYBACK_FAILED_SCRATCH_PRESERVED", work=str(work))), file=sys.stderr, flush=True)
raise SystemExit(exit_code)
PY_BOUNDARY

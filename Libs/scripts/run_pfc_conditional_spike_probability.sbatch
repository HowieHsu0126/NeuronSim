#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_ConditionalSpike
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-conditional-spike-%j.out
#SBATCH --error=/tmp/neuronsim-conditional-spike-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_CONDITIONAL_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_CONDITIONAL'
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
probe_digest = "0028178795cca3f0255fcacc45adf6a21baf9f295e22aadc0d79fa8e3839ef1b"
stage = os.environ.get("NEURONSIM_CONDITIONAL_STAGE", "kernel")
if stage not in ("kernel", "pair", "tree"):
    raise ValueError("UNKNOWN_CONDITIONAL_STAGE:" + stage)
if stage == "tree":
    source_relative = family / "innovation_response/preparation/run_602489"
    source_digest = "6384d3faa4a237d165f676b9b182c95f4888cc5a638a458339d7653169fad7a5"
    source_names = ("prepared/plan.json", "prepared/initial_voltage.npy")
    kernel_relative = family / "conditional_spike_kernel/run_602381"
    kernel_digest = "b2aeacffdd7d39cc12662556fbec63717eb0112f8e852a7ebff4fbfb951ff0a1"
    response_relative = family / "innovation_response/integration/run_602520"
    response_digest = "b215ee50a690a1ea166bcd27c94af75686f9657c47e227f6fd97e1885a8ea659"
    archive = repo / family / "clock_tree_conditioning" / ("run_" + job)
    expected_status = "COMPLETE_PFC_CONDITIONAL_CLOCK_TREE"
elif stage == "pair":
    source_relative = Path(os.environ["NEURONSIM_KERNEL_ARCHIVE"])
    if source_relative.is_absolute():
        source_relative = source_relative.relative_to(repo)
    if ".." in source_relative.parts:
        raise ValueError("KERNEL_ARCHIVE_MUST_BE_WITHIN_PROJECT")
    source_digest = os.environ["NEURONSIM_KERNEL_DELIVERY_SHA256"]
    source_names = ("analysis/report.json", "analysis/conditional_thresholds.npz",
                    "analysis/binomial_tail_bounds.csv")
    archive = repo / family / "conditional_spike_kernel/paired" / ("run_" + job)
    expected_status = "COMPLETE_PFC_PAIRED_SPIKE_KERNEL"
    archive_flag = "--kernel-archive"
else:
    source_relative, source_digest = probe_relative, probe_digest
    source_names = ("execution/report.json", "prepared/plan.json", "prepared/initial_voltage.npy",
                    "execution/original_pre_rk2.npz", "execution/original_pre_threshold.npz",
                    "execution/reversed_pre_rk2.npz", "execution/reversed_pre_threshold.npz")
    archive = repo / family / "conditional_spike_kernel" / ("run_" + job)
    expected_status = "COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL"
    archive_flag = "--archive"
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix="neuronsim-conditional-spike-" + job + ".", dir="/tmp"))
(work / "scratch").mkdir()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(work / "Libs"))), TMPDIR=str(work / "scratch"))
inputs, commands = [], []
started, status, exit_code, error, delivered = time.perf_counter(), expected_status.replace("COMPLETE_", "FAILED_"), 1, None, False

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
        raise RuntimeError("CONDITIONAL_STAGED_INPUT_MISMATCH:" + str(relative))
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
        raise RuntimeError("CONDITIONAL_" + name.upper() + "_FAILED:" + str(result.returncode))

try:
    manifest = stage_archive(source_relative, source_digest, source_names)
    for name, entry in manifest.items():
        if Path(name).parts[0] == "Libs" and Path(name).suffix == ".py":
            copy(repo / source_relative / name, Path(name), entry["sha256"])
    if stage == "tree":
        stage_archive(kernel_relative, kernel_digest, ("analysis/report.json", "analysis/conditional_thresholds.npz"))
        stage_archive(response_relative, response_digest, ("report.json", "analysis/report.json", "analysis/integer_response_table.npz"))
        module = Path("Libs/population_dynamics/pfc_clock_tree_conditioning.py")
        tests = [Path("Libs/population_dynamics/test_pfc_clock_tree_conditioning.py")]
    else:
        module = Path("Libs/population_dynamics/pfc_conditional_spike_probability.py")
        tests = [Path("Libs/population_dynamics/test_pfc_conditional_spike_probability.py")]
        if stage == "pair":
            tests.append(Path("Libs/population_dynamics/test_pfc_paired_spike_kernel.py"))
    for name in (module, *tests):
        copy(repo / name, name)
    copy(Path(os.environ["NEURONSIM_CONDITIONAL_BATCH"]), Path("launcher.sbatch"))
    sys.path.insert(0, str(work))
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work / "resource_preflight.json", 4)
    if health["status"] != "PASS":
        raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(health["reasons"]))
    run_phase("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, tests),
        "--basetemp=" + str(work / "test_tmp"), "--junitxml=" + str(work / "tests.xml")])
    if stage == "tree":
        command = [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_clock_tree_conditioning",
            "--preparation-archive", str(work / source_relative), "--preparation-sha256", source_digest,
            "--kernel-archive", str(work / kernel_relative), "--kernel-sha256", kernel_digest,
            "--response-archive", str(work / response_relative), "--response-sha256", response_digest,
            "--output", str(work / "analysis")]
    else:
        command = [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_conditional_spike_probability",
            archive_flag, str(work / source_relative), "--delivery-sha256", source_digest,
            "--output", str(work / "analysis")]
    run_phase("analysis", command)
    assert json.loads((work / "analysis/report.json").read_text())["status"] == expected_status
    status, exit_code = expected_status, 0
except Exception as exc:
    error = dict(type=type(exc).__name__, message=str(exc))
    traceback.print_exc()
finally:
    (work / "report.json").write_text(json.dumps(dict(status=status, exit_code=exit_code, error=error,
        job_id=job, node=socket.gethostname(), stage=stage, inputs=inputs, commands=commands, source_simulations=0,
        new_full_episodes=0, new_independent_blocks=0, full_trajectory_bound_established=False, selected_event_probability_established=False,
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
                raise RuntimeError("CONDITIONAL_COPYBACK_MISMATCH:" + str(relative))
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
PY_CONDITIONAL

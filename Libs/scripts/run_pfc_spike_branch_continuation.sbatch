#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_SpikePrepare
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-spike-branch-%j.out
#SBATCH --error=/tmp/neuronsim-spike-branch-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_SPIKE_SUBMITTED_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_SPIKE'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import yaml

repo = Path(os.environ["NEURONSIM_PROJECT_ROOT"]).resolve()
stage = os.environ.get("NEURONSIM_SPIKE_STAGE", "prepare")
if stage not in ("prepare", "execute"):
    raise ValueError("SPIKE_STAGE_REQUIRED")
job = os.environ["SLURM_JOB_ID"]
expected_node = "cpu04" if stage == "prepare" else "cpu03"
assert socket.gethostname().split(".")[0] == expected_node
assert not os.environ.get("SLURM_JOB_GPUS")
family = Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/spike_branch_continuation")
archive = repo / family / ("preparation" if stage == "prepare" else "execution") / ("run_" + job)
assert not archive.exists()
config_name = Path("Libs/configs/neural_population_pfc_spike_branch_continuation.yaml")
module_name = Path("Libs/population_dynamics/pfc_spike_branch_continuation.py")
test_name = Path("Libs/population_dynamics/test_pfc_spike_branch_continuation.py")
work = Path(tempfile.mkdtemp(prefix="neuronsim-spike-branch-" + job + ".", dir="/tmp"))
(work / "scratch").mkdir()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(work / "Libs"))),
           TMPDIR=str(work / "scratch"))
inputs, commands = [], []
started = time.perf_counter()
status, exit_code, error, delivered = "FAILED_PFC_SPIKE_BRANCH_" + stage.upper(), 1, None, False

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def copy(source, relative, expected=None):
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    checksum = digest(target)
    if expected is not None and checksum != expected:
        raise RuntimeError("STAGED_INPUT_MISMATCH:" + str(relative))
    inputs.append(dict(source=str(source), path=str(relative), sha256=checksum))

def run_phase(name, command):
    begin = time.perf_counter()
    with (work / (name + ".stdout")).open("w") as out, (work / (name + ".stderr")).open("w") as err:
        result = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err)
    commands.append(dict(phase=name, command=command, exit_code=result.returncode,
                         wall_seconds=time.perf_counter() - begin))
    print(json.dumps(commands[-1]), flush=True)
    if result.returncode:
        raise RuntimeError("SPIKE_" + name.upper() + "_FAILED:" + str(result.returncode))

print(json.dumps(dict(status="STAGING", stage=stage, job_id=job, work=str(work))), flush=True)
try:
    if stage == "prepare":
        settings = yaml.safe_load((repo / config_name).read_text())
        snapshot = repo / settings["accepted_probe"]["archive"]
        assert digest(snapshot / "delivery.json") == settings["accepted_probe"]["delivery_sha256"]
        for entry in json.loads((snapshot / "delivery.json").read_text())["files"]:
            relative = Path(entry["path"])
            if relative.parts[0] == "Libs" and relative.suffix == ".py":
                copy(snapshot / relative, relative, entry["sha256"])
        for relative in (module_name, test_name, config_name):
            copy(repo / relative, relative)
    else:
        parent = Path(os.environ["NEURONSIM_SPIKE_PREPARED_ARCHIVE"]).resolve()
        delivery = json.loads((parent / "delivery.json").read_text())
        assert delivery["verified"] and delivery["status"] == "COMPLETE_PFC_SPIKE_BRANCH_PREPARATION"
        manifest = {row["path"]: row for row in delivery["files"]}
        assert digest(parent / "report.json") == manifest["report.json"]["sha256"]
        receipt = json.loads((parent / "report.json").read_text())
        assert receipt["exit_code"] == 0 and any(x["phase"] == "tests" and x["exit_code"] == 0 for x in receipt["commands"])
        assert any(x["phase"] == "prepare" and x["exit_code"] == 0 for x in receipt["commands"])
        for name, entry in manifest.items():
            relative = Path(name)
            if ((relative.parts[0] == "Libs" and relative.suffix == ".py")
                    or name == str(config_name) or relative.parts[0] == "prepared"):
                copy(parent / relative, relative, entry["sha256"])
        commands.append(dict(phase="tests_reused", archive=str(parent), exit_code=0,
                             report_sha256=digest(parent / "report.json")))
    copy(Path(os.environ["NEURONSIM_SPIKE_SUBMITTED_BATCH"]), Path("launcher.sbatch"))
    if stage == "execute":
        assert digest(work / "launcher.sbatch") == manifest["launcher.sbatch"]["sha256"]
    assert digest(work / config_name) == os.environ["NEURONSIM_SPIKE_CONFIG_SHA256"]
    sys.path.insert(0, str(work))
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work / "resource_preflight.json", 4)
    if health["status"] != "PASS":
        raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(health["reasons"]))
    if stage == "prepare":
        reference = settings["accepted_probe"]
        relative = Path(reference["archive"])
        copy(repo / relative / "delivery.json", relative / "delivery.json", reference["delivery_sha256"])
        accepted_manifest = {x["path"]: x for x in json.loads((work / relative / "delivery.json").read_text())["files"]}
        for name in ("report.json", "execution/report.json", "prepared/plan.json",
                     "prepared/initial_voltage.npy", "execution/original_pre_threshold.npz"):
            copy(repo / relative / name, relative / name, accepted_manifest[name]["sha256"])
        reference = settings["original_source"]
        relative = Path(reference["archive"])
        copy(repo / relative / "delivery.json", relative / "delivery.json", reference["delivery_sha256"])
        source_manifest = {x["path"]: x for x in json.loads((work / relative / "delivery.json").read_text())["files"]}
        for name in ("report.json", "raw.npz"):
            path = Path(settings["case_relative_path"]) / name
            copy(repo / relative / path, relative / path, source_manifest[str(path)]["sha256"])
        run_phase("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_name),
                           "--basetemp=" + str(work / "test_tmp"), "--junitxml=" + str(work / "tests.xml")])
        run_phase("prepare", [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_spike_branch_continuation",
                             "--stage", "prepare", "--config", str(work / config_name),
                             "--project-root", str(work), "--output", str(work / "prepared")])
        assert json.loads((work / "prepared/report.json").read_text())["status"] == "PREPARED_PFC_SPIKE_BRANCH_CONTINUATION"
        status = "COMPLETE_PFC_SPIKE_BRANCH_PREPARATION"
    else:
        run_phase("execute", [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_spike_branch_continuation",
                             "--stage", "execute", "--prepared", str(work / "prepared"),
                             "--output", str(work / "execution")])
        assert json.loads((work / "execution/report.json").read_text())["status"] == "COMPLETE_PFC_SPIKE_BRANCH_CONTINUATION"
        status = "COMPLETE_PFC_SPIKE_BRANCH_CONTINUATION"
    exit_code = 0
except Exception as exc:
    error = dict(type=type(exc).__name__, message=str(exc))
    traceback.print_exc()
finally:
    report = dict(status=status, exit_code=exit_code, error=error, stage=stage, job_id=job,
                  node=socket.gethostname(), inputs=inputs, commands=commands,
                  intended_counterfactual_prefixes=1 if stage == "execute" else 0,
                  counterfactual_event_sensitivity_diagnostic=True, legal_native_auxiliary_realization=False,
                  statistical_source_draw=False, admissible_inverse_stimulus=False,
                  auxiliary_enclosure_qualified=False, cdf_qualified=False,
                  new_full_episodes=0, new_independent_blocks=0, confirmation_execution_released=False,
                  production_episodes=0, wall_seconds=time.perf_counter() - started,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
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
                raise RuntimeError("ARCHIVE_COPY_MISMATCH:" + str(relative))
            files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
        (archive / "delivery.json").write_text(json.dumps(dict(status=status, job_id=job, verified=True, files=files), indent=2) + "\n")
        delivered = True
        print(json.dumps(dict(status=status, archive=str(archive), exit_code=exit_code, files=len(files))), flush=True)
    finally:
        if delivered:
            shutil.rmtree(work)
        else:
            print(json.dumps(dict(status="COPYBACK_FAILED_SCRATCH_PRESERVED", work=str(work))), file=sys.stderr, flush=True)
raise SystemExit(exit_code)
PY_SPIKE

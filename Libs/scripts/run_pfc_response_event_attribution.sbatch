#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_ResponseEvents
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-response-attribution-%j.out
#SBATCH --error=/tmp/neuronsim-response-attribution-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_ATTRIBUTION_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_ATTRIBUTION'
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
import yaml

repo = Path(os.environ["NEURONSIM_PROJECT_ROOT"]).resolve()
job = os.environ["SLURM_JOB_ID"]
assert socket.gethostname().split(".")[0] == "cpu04" and not os.environ.get("SLURM_JOB_GPUS")
assert (repo / "PROJECT_PREFIX").read_text().strip() == os.environ["PROJECT_PREFIX"]
family = Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement")
archive = repo / family / "response_event_attribution" / ("run_" + job)
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix="neuronsim-response-attribution-" + job + ".", dir="/tmp"))
(work / "scratch").mkdir()
env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(work), str(work / "Libs"))), TMPDIR=str(work / "scratch"))
inputs, commands = [], []
started, status, exit_code, error, delivered = time.perf_counter(), "FAILED_PFC_RESPONSE_EVENT_ATTRIBUTION", 1, None, False
config = Path("Libs/configs/neural_population_pfc_response_event_attribution.yaml")


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
        raise RuntimeError("ATTRIBUTION_STAGED_INPUT_MISMATCH:" + str(relative))
    inputs.append(dict(source=str(source), path=str(relative), sha256=actual))


def run_phase(name, command):
    begin = time.perf_counter()
    with (work / (name + ".stdout")).open("w") as out, (work / (name + ".stderr")).open("w") as err:
        result = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err)
    commands.append(dict(phase=name, command=command, exit_code=result.returncode, wall_seconds=time.perf_counter() - begin))
    print(json.dumps(commands[-1]), flush=True)
    if result.returncode:
        raise RuntimeError("ATTRIBUTION_" + name.upper() + "_FAILED:" + str(result.returncode))

try:
    # Reuse the exact proven dependency snapshot; replace only this task's owned modules.
    snapshot = family / "count_boundary_decomposition/run_602343"
    copy(repo / snapshot / "delivery.json", Path("dependency_delivery.json"), "648b874eb79480b5d3bffb0589883fa1794f4b26a59c65c30500d9a28cef23ff")
    prior = json.loads((work / "dependency_delivery.json").read_text())
    assert prior["verified"] and prior["status"] == "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION"
    owned = [Path("Libs/population_dynamics/" + name) for name in (
        "pfc_count_boundary_decomposition.py", "pfc_response_event_attribution.py",
        "test_pfc_count_boundary_decomposition.py", "test_pfc_response_event_attribution.py")]
    for entry in prior["files"]:
        name = Path(entry["path"])
        if name.parts[0] == "Libs" and name.suffix == ".py" and name not in owned and not name.name.startswith("test_"):
            copy(repo / snapshot / name, name, entry["sha256"])
    for name in [*owned, config]:
        copy(repo / name, name)
    copy(Path(os.environ["NEURONSIM_ATTRIBUTION_BATCH"]), Path("launcher.sbatch"))
    sys.path.insert(0, str(work))
    from Libs.population_dynamics.resource_probe import probe
    health = probe(work / "resource_preflight.json", 4)
    if health["status"] != "PASS":
        raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(health["reasons"]))
    tests = [str(p) for p in owned if p.name.startswith("test_")]
    run_phase("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests,
        "--basetemp=" + str(work / "test_tmp"), "--junitxml=" + str(work / "tests.xml")])
    settings = yaml.safe_load((work/config).read_text())
    manifests = {}
    for key, spec in settings["archives"].items():
        relative = Path(spec["path"])
        copy(repo / relative / "delivery.json", relative / "delivery.json", spec["delivery_sha256"])
        delivery = json.loads((work / relative / "delivery.json").read_text())
        assert delivery["status"] == spec["status"]
        manifests[key] = {entry["path"]: entry for entry in delivery["files"]}
    for case in settings["cases"]:
        root = Path(settings["archives"][case["archive"]]["path"])
        for leaf in ("raw.npz", "report.json", "config.yaml", "unit_metadata.json"):
            name = case["path"] + "/" + leaf
            copy(repo / root / name, root / name, manifests[case["archive"]][name]["sha256"])
    for spec in settings["score_tables"].values():
        root = Path(settings["archives"][spec["archive"]]["path"])
        assert manifests[spec["archive"]][spec["path"]]["sha256"] == spec["sha256"]
        copy(repo / root / spec["path"], root / spec["path"], spec["sha256"])
    run_phase("analysis", [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_response_event_attribution",
        "--project-root", str(work), "--config", str(work/config), "--output", str(work/"analysis")])
    assert json.loads((work / "analysis/report.json").read_text())["status"] == "COMPLETE_PFC_RESPONSE_EVENT_ATTRIBUTION"
    status, exit_code = "COMPLETE_PFC_RESPONSE_EVENT_ATTRIBUTION", 0
except Exception as exc:
    error = dict(type=type(exc).__name__, message=str(exc))
    traceback.print_exc()
finally:
    (work / "report.json").write_text(json.dumps(dict(status=status, exit_code=exit_code, error=error,
        job_id=job, node=socket.gethostname(), inputs=inputs, commands=commands, source_simulations=0,
        new_full_episodes=0, new_independent_blocks=0, confirmation_execution_released=False, production_episodes=0,
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
                raise RuntimeError("ATTRIBUTION_COPYBACK_MISMATCH:" + str(relative))
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
PY_ATTRIBUTION

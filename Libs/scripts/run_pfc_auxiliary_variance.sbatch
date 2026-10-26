#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AuxVariance
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=2G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-auxiliary-variance-%j.out
#SBATCH --error=/tmp/neuronsim-auxiliary-variance-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_SUBMITTED_BATCH="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_AUXILIARY'
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

import yaml

repo = Path(os.environ["NEURONSIM_PROJECT_ROOT"]).resolve()
job = os.environ["SLURM_JOB_ID"]
family = Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/refinement_repeat")
snapshot = repo / family / "comparison/run_599321"
mode = os.environ.get("NEURONSIM_PFC_DIAGNOSTIC", "auxiliary_variance")
if mode not in ("auxiliary_variance", "clock_scaling", "temporal_control_comparison", "cdf_moment_envelope", "cdf_moment_extrema"):
    raise ValueError("SUPPORTED_PFC_EXISTING_DATA_DIAGNOSTIC_REQUIRED")
tests_only_value = os.environ.get("NEURONSIM_EXISTING_DATA_TESTS_ONLY", "0")
if tests_only_value not in ("0", "1") or (mode != "temporal_control_comparison" and tests_only_value != "0"):
    raise ValueError("EXPLICIT_EXISTING_DATA_TESTS_MODE_REQUIRED")
tests_only = tests_only_value == "1"
if mode == "temporal_control_comparison":
    family = Path("Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/temporal_order_control")
    snapshot = repo / family / "validation/run_601642"
    archive = repo / family / ("comparison_validation" if tests_only else "comparison") / ("run_" + job)
else:
    archive = repo / family / mode / ("run_" + job)
config_name = Path("Libs/configs/neural_population_pfc_" + mode + ".yaml")
module_name = Path("Libs/population_dynamics/pfc_" + mode + ".py")
test_name = Path("Libs/population_dynamics/test_pfc_" + mode + ".py")
assert socket.gethostname().split(".")[0] == "cpu04" and not os.environ.get("SLURM_JOB_GPUS")
assert not archive.exists()
work = Path(tempfile.mkdtemp(prefix="neuronsim-auxiliary-variance-" + job + ".", dir="/tmp"))
inputs = []
commands = []
started = time.perf_counter()
status = "FAILED_PFC_" + mode.upper() + "_DIAGNOSTIC"
exit_code = 1
delivered = False

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
        raise RuntimeError(name.upper() + "_FAILED:" + str(result.returncode))

try:
    # Reuse the producer's accepted dependency snapshot; no original arrays or source simulations.
    for entry in json.loads((snapshot / "delivery.json").read_text())["files"]:
        relative = Path(entry["path"])
        if relative.parts[0] == "Libs" and relative.suffix == ".py":
            copy(snapshot / relative, relative, entry["sha256"])
    if mode == "temporal_control_comparison":
        for name in ("pfc_coupled_clock_comparison", "pfc_count_histograms", "pfc_phase_responses",
                     "pfc_fluctuation_analysis", "pfc_scale_analysis", "pfc_response_analysis",
                     "pfc_temporal_readouts"):
            relative = Path("Libs/population_dynamics") / (name + ".py")
            copy(repo / relative, relative, digest(repo / relative))
    for relative in (module_name, test_name, config_name):
        copy(repo / relative, relative, digest(repo / relative))
    if mode == "clock_scaling":
        auxiliary = repo / family / "auxiliary_variance/run_601413"
        auxiliary_report = json.loads((auxiliary / "analysis/report.json").read_text())
        helper = Path("Libs/population_dynamics/pfc_auxiliary_variance.py")
        copy(auxiliary / helper, helper, auxiliary_report["implementation_sha256"])
    copy(Path(os.environ["NEURONSIM_SUBMITTED_BATCH"]), Path("launcher.sbatch"))
    assert digest(work / config_name) == os.environ["NEURONSIM_AUXILIARY_CONFIG_SHA256"]
    sys.path.insert(0, str(work / "Libs"))
    from population_dynamics.resource_probe import probe
    health = probe(work / "resource_preflight.json", 4 if mode == "temporal_control_comparison" else 2)
    if health["status"] != "PASS":
        raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(health["reasons"]))
    settings = yaml.safe_load((work / config_name).read_text())
    if mode == "temporal_control_comparison":
        protocol = settings["protocol"]
        copy(repo / protocol["path"], Path(protocol["path"]), protocol["sha256"])
        if not tests_only:
            # These explicit manifests supply all required names; no source-tree discovery.
            def stage_archive(reference):
                relative = Path(reference["archive"])
                copy(repo / relative / "delivery.json", relative / "delivery.json",
                     reference.get("delivery_sha256"))
                entries = json.loads((work / relative / "delivery.json").read_text())["files"]
                return relative, {entry["path"]: entry for entry in entries}
            def stage_entry(relative, manifest, name):
                entry = manifest[name]
                copy(repo / relative / name, relative / name, entry["sha256"])
            previous, previous_files = stage_archive(settings["original_comparison"])
            for name in ("report.json", "phase_responses.npz", "pilot_count_cdfs.npz",
                         "numpy_dt_0p025_count_histograms.npz", "numpy_dt_0p0125_count_histograms.npz"):
                stage_entry(previous, previous_files, "analysis/" + name)
            original, original_files = stage_archive(settings["original_source"])
            source, source_files = stage_archive({"archive": settings["source_archive"]})
            stage_entry(source, source_files, "report.json")
            stage_entry(source, source_files, protocol["path"])
            for clock in ("0p025", "0p0125"):
                for condition in ("cue", "omission", "strong_cue"):
                    prefix = "source_cases/dt_" + clock + "/pfc_95001_" + condition + "/"
                    for name in ("report.json", "field_observations.npz"):
                        stage_entry(original, original_files, prefix + name)
                    stage_entry(source, source_files, prefix + "report.json")
                    case_report = json.loads((work / source / (prefix + "report.json")).read_text())
                    for name in case_report["outputs"]:
                        stage_entry(source, source_files, prefix + name)
    elif mode in ("cdf_moment_envelope", "cdf_moment_extrema"):
        for reference in settings["inputs"].values():
            source = Path(reference["path"])
            copy(repo / source / "report.json", source / "report.json", reference["report_sha256"])
            report = json.loads((work / source / "report.json").read_text())
            for name in reference["files"]:
                copy(repo / source / name, source / name, report["outputs"][name]["sha256"])
            if mode == "cdf_moment_extrema":
                helper = Path("Libs/population_dynamics/pfc_cdf_moment_envelope.py")
                copy(repo / source.parent / helper, helper, report["implementation_sha256"])
    else:
        reference = settings["comparison"]
        source = Path(reference["path"])
        copy(repo / source / "report.json", source / "report.json", reference["report_sha256"])
        report = json.loads((work / source / "report.json").read_text())
        input_names = (("auxiliary_repeat_comparison.csv", "primitive_nested_scores.npz")
                       if mode == "auxiliary_variance" else ("primitive_nested_scores.npz",))
        for name in input_names:
            copy(repo / source / name, source / name, report["outputs"][name]["sha256"])
        if mode == "clock_scaling":
            auxiliary_reference = settings["auxiliary_variance"]
            auxiliary_source = Path(auxiliary_reference["path"])
            copy(repo / auxiliary_source / "report.json", auxiliary_source / "report.json",
                 auxiliary_reference["report_sha256"])
            auxiliary_report = json.loads((work / auxiliary_source / "report.json").read_text())
            name = "auxiliary_variance_endpoints.csv"
            copy(repo / auxiliary_source / name, auxiliary_source / name,
                 auxiliary_report["outputs"][name]["sha256"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work), str(work / "Libs")]), TMPDIR=str(work))
    test_reference = os.environ.get("NEURONSIM_EXISTING_DATA_TEST_REFERENCE")
    if test_reference:
        tested = Path(test_reference)
        tested_report = json.loads((tested / "report.json").read_text())
        assert tested_report["exit_code"] == 0 and any(
            phase["phase"] == "tests" and phase["exit_code"] == 0 for phase in tested_report["commands"])
        tested_files = {entry["path"]: entry for entry in json.loads((tested / "delivery.json").read_text())["files"]}
        for relative in {entry["path"] for entry in inputs if entry["path"].startswith("Libs/") and entry["path"].endswith(".py")}:
            assert digest(work / relative) == tested_files[relative]["sha256"], "TESTED_DIAGNOSTIC_CODE_CHANGED:" + relative
        commands.append(dict(phase="tests_reused", archive=str(tested), exit_code=0,
                             report_sha256=digest(tested / "report.json")))
    else:
        run_phase("tests", [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           str(test_name), "--basetemp=" + str(work / "test_tmp"),
                           "--junitxml=" + str(work / "tests.xml")])
    if not tests_only:
        run_phase("analysis", [sys.executable, "-B", "-m", "Libs.population_dynamics.pfc_" + mode,
                              "--config", str(work / config_name), "--project-root", str(work),
                              "--output", str(work / "analysis")])
    status = "COMPLETE_PFC_" + mode.upper() + ("_SOFTWARE_VALIDATION" if tests_only else "_DIAGNOSTIC")
    exit_code = 0
except Exception as exc:
    (work / "failure.json").write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc)), indent=2) + "\n")
finally:
    (work / "report.json").write_text(json.dumps(dict(
        status=status, diagnostic=mode, tests_only=tests_only, analysis_executed=any(phase["phase"] == "analysis" for phase in commands), exit_code=exit_code, job_id=job, inputs=inputs, commands=commands,
        source_simulations=0, new_rng_draws=0, production_episodes=0, confirmation_execution_released=False,
        wall_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat()
    ), indent=2) + "\n")
    try:
        archive.mkdir(parents=True, exist_ok=False)
        files = []
        for source in sorted(work.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(work)
            if relative.parts[0] in ("Output", "test_tmp"):
                continue
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(source)
            if digest(target) != checksum:
                raise RuntimeError("DELIVERY_MISMATCH:" + str(relative))
            files.append(dict(path=str(relative), bytes=source.stat().st_size, sha256=checksum))
        (archive / "delivery.json").write_text(json.dumps(dict(
            status=status, job_id=job, verified=True, files=files
        ), indent=2) + "\n")
        delivered = True
        print(json.dumps(dict(status=status, exit_code=exit_code, archive=str(archive))), flush=True)
    finally:
        if delivered:
            shutil.rmtree(work)
        else:
            print(json.dumps(dict(status="COPYBACK_FAILED_SCRATCH_PRESERVED", work=str(work))),
                  file=sys.stderr, flush=True)
raise SystemExit(exit_code)
PY_AUXILIARY

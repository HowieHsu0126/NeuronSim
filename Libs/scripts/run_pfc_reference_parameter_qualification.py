#!/usr/bin/env python3
"""Prepare exact PFC kernel checks; execute one tuple after fixed150 assessment.

Preparation runs no simulator and submits no job. The generated batch entry point
runs the three archived launchers unchanged inside its Slurm allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
OPERATIONS = REPO / "Output/neural_population_dynamics/operations/pfc_reference_source_admission_20261018"
TEMPLATE_TUPLE = OPERATIONS / "tuple_completion/sigma24p444_jpos1p988"
PHASES = {
    "native": ("native_validation_634189", "tests", "PASS_NATIVE_INPUT_OBSERVATION_PARITY", "native_validation_"),
    "pairing": ("pairing_634190", "tests", "PASS_NATIVE_INPUT_PAIRING", "pairing_"),
    "stream": ("stream_numpy_634191", "invariants", "PASS_NATIVE_STREAM_SOURCE_EXECUTION", "stream_numpy_"),
}
LAUNCHER_PATHS = {
    "native": "Output/neural_population_dynamics/operations/pfc_native_input_validation.sbatch",
    "pairing": "Output/neural_population_dynamics/operations/pfc_input_pairing.sbatch",
    "stream": "Output/neural_population_dynamics/operations/pfc_stream_validation.sbatch",
}
PREPARED_STATUS = "PREPARED_ONLY_NATIVE_PAIRING_STREAM_REQUIRED"
REUSE_STATUS = "REUSE_COMPLETED_EXACT_TUPLE"
SCHEMA = "pfc_reference_parameter_qualification_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid relative path: {name}")
    return path


def checked(root: Path, row: dict[str, Any]) -> Path:
    path = root / relative(row["path"])
    if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError(f"Changed archived binding: {path}")
    return path


def entries(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = receipt["files"]
    result = {str(relative(row["path"])): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate delivery paths")
    return result


def file_row(root: Path, path: Path) -> dict[str, Any]:
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest(path))


def copy_file(src: Path, dst: Path, root: Path, files: dict[str, dict[str, Any]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and digest(dst) != digest(src):
        raise ValueError(f"Conflicting frozen files: {dst}")
    shutil.copy2(src, dst)
    if digest(src) != digest(dst):
        raise ValueError(f"Stage mismatch: {dst}")
    row = file_row(root, dst)
    files[row["path"]] = row


def validate_inputs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = load_json(path)
    report = load_json(path.parent / "report.json")
    if inputs.get("dependent_on_fixed150_quality_assessment") is not True:
        raise ValueError("Fixed150 quality assessment dependency must be retained")
    rows = inputs["rows"]
    if len(rows) != 100 or {row["circuit_index"] for row in rows} != set(range(100)):
        raise ValueError("Complete 100-circuit candidate assignment required")
    if report.get("candidate_circuits") != 100 or report.get("distinct_effective_kernels") != 100:
        raise ValueError("Report must describe 100 distinct candidate kernels")
    tuples = {tuple(row["parameters"][name] for name in ("sigma_weight_profile", "Jpos_excit2excit")) for row in rows}
    if len(tuples) != 100:
        raise ValueError("Distinct exact parameter tuples required")
    pending = [row for row in rows if row["status"] == PREPARED_STATUS]
    reused = [row for row in rows if row["status"] == REUSE_STATUS]
    if len(pending) != 95 or len(reused) != 5:
        raise ValueError("Exactly 95 prepared and five completed tuples required")
    for row in pending:
        configs = {}
        for name in ("independent", "global_config"):
            config = yaml.safe_load((path.parent / relative(row[name])).read_text())
            for parameter, value in row["parameters"].items():
                if config.get(parameter) != value:
                    raise ValueError(f"Circuit {row['circuit_index']} exact parameter mismatch: {parameter}")
            if config.get("dt_ms") != 0.0125 or config.get("synapse_execution") != "factorized_uniform_projections":
                raise ValueError("Frozen reference clock and source execution required")
            configs[name] = config
        if configs["independent"].get("random_stream_policy") != "independent_initial_external":
            raise ValueError("Independent initial/external stream semantics required")
        if len(configs["independent"].get("cases", [])) != 8 or len(configs["global_config"].get("cases", [])) != 2:
            raise ValueError("Frozen eight independent and two global source cases required")
        if "source_rng_seed" not in configs["global_config"]:
            raise ValueError("Global source stream identity required")
    return inputs, report


def prepare(inputs_path: Path, software_delivery: Path, template_root: Path, output: Path) -> dict[str, Any]:
    inputs, report = validate_inputs(inputs_path)
    software = entries(load_json(software_delivery))
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    copy_file(inputs_path, output / "inputs/qualification_inputs.json", output, files)
    copy_file(inputs_path.parent / "report.json", output / "inputs/report.json", output, files)
    copy_file(software_delivery, output / "software_delivery.json", output, files)
    for name, row in software.items():
        copy_file(checked(software_delivery.parent, row), output / "project" / name, output, files)
    for key, name in (("kernel_implementation_sha256", "Libs/population_dynamics/pfc_kernel_parameters.py"),
                      ("source_entry_sha256", "source/neurodynex3/working_memory_network/wm_model.py")):
        if key in report.get("inputs", {}) and report["inputs"][key] != digest(output / "project" / name):
            raise ValueError(f"Parameter preparation software differs: {key}")
    templates = {}
    for phase, (directory, label, expected_status, _) in PHASES.items():
        archive = template_root / directory
        receipt_path = archive / "delivery.json"
        archive_files = entries(load_json(receipt_path))
        execution = load_json(checked(archive, archive_files["execution.json"]))
        if execution.get("status") != expected_status:
            raise ValueError(f"Completed {phase} execution required")
        launcher = checked(archive, archive_files["launcher.sbatch"])
        copy_file(launcher, output / "templates" / (phase + ".sbatch"), output, files)
        # The templates also archive their source. Their scientific bodies stay exact.
        copy_file(launcher, output / "project" / LAUNCHER_PATHS[phase], output, files)
        tests = [row for row in execution["commands"] if row["label"] == label and row.get("exit_code") == 0]
        if len(tests) != 1 or tests[0]["status"] not in ("PASS", "REUSED_PASS"):
            raise ValueError(f"Completed {phase} test command required")
        test = tests[0]
        test_archive = archive
        if test["status"] == "REUSED_PASS":
            test_archive = Path(test["source_archive"])
            if digest(test_archive / "delivery.json") != test["source_delivery_sha256"]:
                raise ValueError(f"Changed {phase} test archive")
        test_receipt = load_json(test_archive / "delivery.json")
        test_files = entries(test_receipt)
        previous = load_json(checked(test_archive, test_files["execution.json"]))
        matches = [row for row in previous["commands"] if row["label"] == label and row["command"] == test["command"]
                   and row["status"] == "PASS" and row.get("exit_code") == 0]
        if len(matches) != 1:
            raise ValueError(f"Exact completed {phase} test command required")
        compact = output / "test_archives" / phase
        copy_file(test_archive / "delivery.json", compact / "delivery.json", output, files)
        names = ["execution.json", label + ".log"]
        if "tests_result.json" in test_files:
            names.append("tests_result.json")
        names.extend(name for name in test_files if name.startswith("Libs/") and name.endswith(".py"))
        for name in names:
            src = checked(test_archive, test_files[name])
            copy_file(src, compact / name, output, files)
            if name.startswith("Libs/"):
                copy_file(src, output / "project" / name, output, files)
        templates[phase] = dict(path="templates/" + phase + ".sbatch", archive=str(archive.resolve()),
                                delivery_sha256=digest(receipt_path), python=test["command"][0],
                                test_archive="test_archives/" + phase, test_label=label, test_command=test["command"])
    for row in inputs["rows"]:
        if row["status"] == PREPARED_STATUS:
            for key in ("independent", "global_config"):
                name = relative(row[key])
                copy_file(inputs_path.parent / name, output / "inputs" / name, output, files)
    copy_file(Path(__file__).resolve(), output / "runner/run_pfc_reference_parameter_qualification.py", output, files)
    execution = dict(schema=SCHEMA, status="PREPARED95_REFERENCE_KERNEL_CHECKS", rows=inputs["rows"],
                     dependent_on_fixed150_quality_assessment=True, qualification_inputs_sha256=digest(inputs_path),
                     inputs_origin=str(inputs_path.resolve()), software_delivery_origin=str(software_delivery.resolve()),
                     templates=templates, staged_files=list(files.values()), new_source_simulations=0,
                     production_episodes=0, source_qualification_granted=False, scientific_parameter_domain_qualified=False)
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    launcher = f'''#!/bin/bash
#SBATCH --job-name=NrnSim_pfc_parameter_qualification
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec {shlex.quote(templates["stream"]["python"])} -B {shlex.quote(str((output / "runner/run_pfc_reference_parameter_qualification.py").resolve()))} --execute --prepared {shlex.quote(str(output.resolve()))} --circuit-index "${{NEURONSIM_PFC_PARAMETER_CIRCUIT_INDEX:-${{SLURM_ARRAY_TASK_ID:?explicit circuit index required}}}}" --assessment "${{NEURONSIM_PFC_PARAMETER_ASSESSMENT:?fixed150 assessment artifact required}}"
'''
    (output / "launcher.sbatch").write_text(launcher)
    return dict(status=execution["status"], prepared_circuits=95, reused_circuits=5, new_source_simulations=0,
                source_qualification_granted=False, production_episodes=0, execution=str(output / "execution.json"),
                launcher=str(output / "launcher.sbatch"))


def admission(assessment_path: Path | None, prepared: dict[str, Any]) -> dict[str, Any]:
    if assessment_path is None:
        raise ValueError("Explicit fixed150 quality assessment required before remaining95 execution")
    assessment = load_json(assessment_path)
    if assessment.get("approved_remaining_reference_kernel_checks") is not True:
        raise ValueError("Explicit remaining reference-kernel admission required; summary status does not admit execution")
    if assessment.get("scope") != "remaining95_reference_kernel_checks" or not str(assessment.get("rationale", "")).strip():
        raise ValueError("Remaining95 scope and scientific assessment rationale required")
    if assessment.get("qualification_inputs_sha256") != prepared["qualification_inputs_sha256"]:
        raise ValueError("Assessment qualification input binding differs")
    binding = assessment["fixed150_quality_summary"]
    path = Path(binding["path"])
    if not path.is_absolute():
        path = assessment_path.parent / path
    if digest(path) != binding["sha256"]:
        raise ValueError("Fixed150 quality summary binding differs")
    summary = load_json(path)
    if summary.get("status") != "COMPLETE_REFERENCE_QUALITY_DESCRIPTIVE_COHORT" or summary.get("complete_episodes") != 150:
        raise ValueError("Complete fixed150 quality summary required")
    reused = {row["circuit_index"] for row in prepared["rows"] if row["status"] == REUSE_STATUS}
    if set(summary.get("circuits", [])) != reused or set(summary.get("blocks", [])) != {3, 4, 5} or len(set(summary.get("conditions", []))) != 10:
        raise ValueError("Complete fixed five-circuit, three-block, ten-condition cohort required")
    if summary.get("core_units") != 100000 or summary.get("duration_ticks") != 400000 or summary.get("dt_ms") != 0.0125:
        raise ValueError("Frozen full-scale reference quality cohort required")
    bindings = summary.get("bindings", [])
    if len(bindings) != 150 or len({row["episode_id"] for row in bindings}) != 150:
        raise ValueError("150 distinct complete episode bindings required")
    return dict(path=str(assessment_path.resolve()), sha256=digest(assessment_path),
                fixed150_quality_summary=str(path.resolve()), fixed150_quality_summary_sha256=binding["sha256"],
                scope=assessment["scope"], rationale=assessment["rationale"])


def validated_preparation(root: Path) -> dict[str, Any]:
    prepared = load_json(root / "execution.json")
    if prepared.get("schema") != SCHEMA or prepared.get("dependent_on_fixed150_quality_assessment") is not True:
        raise ValueError("Prepared remaining95 execution metadata required")
    for row in prepared["staged_files"]:
        checked(root, row)
    inputs, _ = validate_inputs(root / "inputs/qualification_inputs.json")
    if digest(root / "inputs/qualification_inputs.json") != prepared["qualification_inputs_sha256"] or inputs["rows"] != prepared["rows"]:
        raise ValueError("Prepared execution input binding differs")
    return prepared


def phase_environment(base: dict[str, str], local: Path, archive: Path, prepared: dict[str, Any], row: dict[str, Any], phase: str) -> dict[str, str]:
    template = prepared["templates"][phase]
    job_id = base["SLURM_JOB_ID"]
    scratch = local / "scratch"
    env = dict(base, NEURONSIM_PROJECT_ROOT=str(local / "project"), NEURONSIM_PFC_SCRATCH=str(scratch),
               NEURONSIM_PFC_OUTPUT_BASE=str(archive), NEURONSIM_PFC_SOURCE=str(local / "project/source"),
               NEURONSIM_PFC_CONFIG=str(local / "inputs" / row["independent"]),
               NEURONSIM_PFC_GLOBAL_CONFIG=str(local / "inputs" / row["global_config"]),
               NEURONSIM_PFC_PLAIN_REFERENCE="execute", NEURONSIM_PFC_REPLAY_BACKEND="numpy",
               NEURONSIM_PFC_PYTHON=template["python"], NEURONSIM_PFC_TEST_ARCHIVE=str(local / template["test_archive"]),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch))
    if phase in ("pairing", "stream"):
        env["NEURONSIM_PFC_VALIDATION"] = str(archive / (PHASES["native"][3] + job_id))
    if phase == "stream":
        env["NEURONSIM_PFC_PAIRING"] = str(archive / (PHASES["pairing"][3] + job_id))
        env["NEURONSIM_PFC_GLOBAL_REFERENCE"] = env["NEURONSIM_PFC_VALIDATION"] + "/global_reference"
    return env


def execute(root: Path, circuit_index: int, assessment_path: Path | None, *,
            resume_native_from: Path | None = None, attempt_output: Path | None = None) -> dict[str, Any]:
    prepared = validated_preparation(root)
    row = next((row for row in prepared["rows"] if row["circuit_index"] == circuit_index), None)
    if row is None:
        raise ValueError("Circuit is outside the prepared candidate assignment")
    if row["status"] == REUSE_STATUS:
        return dict(status=REUSE_STATUS, circuit_index=circuit_index, qualification=row["qualification"], new_source_simulations=0)
    assessment = admission(assessment_path, prepared)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") not in ("cpu03", "cpu04", "cpu05", "cpu06"):
        raise ValueError("An authorized PFC CPU Slurm batch allocation is required")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("Reference-kernel checks use the CPU-only source path")
    for phase, template in prepared["templates"].items():
        if not os.access(template["python"], os.X_OK):
            raise ValueError(f"Qualified {phase} interpreter unavailable: {template['python']}")
    if resume_native_from is not None and attempt_output is None:
        raise ValueError("Native resume requires a new explicit attempt output")
    archive = attempt_output if attempt_output is not None else root / "results" / f"circuit_{circuit_index:03d}"
    # Reconcile an existing attempt before retrying; never duplicate an unknown outcome.
    archive.mkdir(parents=True, exist_ok=False)
    job_id = os.environ["SLURM_JOB_ID"]
    result = dict(status="PREPARING", circuit_index=circuit_index, parameters=row["parameters"],
                  qualification_inputs_sha256=prepared["qualification_inputs_sha256"], assessment=assessment,
                  producer_job=job_id, node=os.environ["SLURMD_NODENAME"],
                  runner_implementation_sha256=digest(Path(__file__).resolve()),
                  phases={}, source_qualification_granted=False, scientific_parameter_domain_qualified=False, production_episodes=0)
    def save() -> None:
        (archive / "runner_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    local = Path(tempfile.mkdtemp(prefix="neuronsim-pfc-parameter-", dir="/tmp"))
    result.update(retained_runtime=str(local), runtime_state="ALLOCATED", active_phase=None)
    save()
    try:
        for item in prepared["staged_files"]:
            name = relative(item["path"])
            if name.parts[0] not in ("project", "templates", "test_archives") and str(name) not in (
                "inputs/" + row["independent"], "inputs/" + row["global_config"]):
                continue
            dst = local / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checked(root, item), dst)
            checked(local, item)
        (local / "scratch").mkdir()
        if resume_native_from is not None:
            parent = load_json(resume_native_from / "execution.json")
            if not isinstance(parent.get("job_id"), str) or not parent["job_id"].isdecimal():
                raise ValueError("Retained native producer identity required")
            for item in prepared["staged_files"]:
                name = relative(item["path"])
                if str(name).startswith("project/Libs/population_dynamics/"):
                    old = resume_native_from / Path(*name.parts[1:])
                    if old.is_file() and digest(old) != item["sha256"]:
                        raise ValueError("Retained native implementation differs:" + str(old))
                elif str(name).startswith("project/source/"):
                    old = resume_native_from / Path(*name.parts[1:])
                    if old.is_file() and digest(old) != item["sha256"]:
                        raise ValueError("Retained native source differs:" + str(old))
            helper = Path(__file__).with_name("resume_pfc_native_validation.py")
            shutil.copy2(helper, local / helper.name)
            result["native_resume"] = dict(parent_runtime=str(resume_native_from.resolve()),
                parent_job_id=parent["job_id"], parent_execution_sha256=digest(resume_native_from / "execution.json"),
                helper_sha256=digest(helper), parent_status=parent.get("status"))
            save()
        for phase, (_, _, expected_status, prefix) in PHASES.items():
            template = prepared["templates"][phase]
            env = phase_environment(dict(os.environ), local, archive, prepared, row, phase)
            result.update(status="RUNNING_" + phase.upper(), active_phase=phase, runtime_state="RUNNING")
            save()
            with (archive / (phase + "_launcher.log")).open("x") as log:
                command = ["bash", str(local / template["path"])]
                if phase == "native" and resume_native_from is not None:
                    env["PYTHONPATH"] = str(resume_native_from / "Libs") + os.pathsep + str(resume_native_from / "source")
                    command = [template["python"], "-B", str(local / "resume_pfc_native_validation.py"),
                        "--retained", str(resume_native_from), "--independent-config", str(local / "inputs" / row["independent"]),
                        "--global-config", str(local / "inputs" / row["global_config"]),
                        "--output", str(archive / (prefix + job_id))]
                completed = subprocess.run(command, env=env, cwd="/tmp", stdout=log, stderr=subprocess.STDOUT)
            phase_root = archive / (prefix + job_id)
            result["phases"][phase] = dict(status="LAUNCHER_RETURNED", archive=str(phase_root),
                                            launcher_exit_code=completed.returncode)
            if completed.returncode:
                local_name = dict(native="neuronsim-pfc-input-validation-",
                                  pairing="neuronsim-pfc-input-pairing-", stream="neuronsim-pfc-stream-validation-")[phase]
                local_phase = local / "scratch" / (local_name + job_id)
                failure = result["phases"][phase]
                failure.update(status="FAILED_LAUNCHER", retained_phase_runtime=str(local_phase))
                for name, path in (("local_execution", local_phase / "execution.json"),
                                   ("partial_archive_execution", phase_root / "execution.json")):
                    if path.is_file():
                        failure[name] = load_json(path)
                save()
                raise RuntimeError(f"{phase} launcher failed with exit code {completed.returncode}; see {log.name}")
            receipt = load_json(phase_root / "delivery.json")
            phase_result = load_json(checked(phase_root, entries(receipt)["execution.json"]))
            if phase_result["status"] != expected_status:
                raise ValueError(f"{phase} complete source-check receipt required")
            result["phases"][phase] = dict(status=expected_status, archive=str(phase_root),
                launcher_exit_code=completed.returncode, delivery_sha256=digest(phase_root / "delivery.json"))
            save()
        shutil.rmtree(local)
        result.update(status="PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS", active_phase=None,
                      retained_runtime=None, runtime_state="REMOVED_AFTER_SUCCESS")
        save()
    except BaseException as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__,
                      failed_phase=result.get("active_phase"), runtime_state="RETAINED_AFTER_FAILURE",
                      retained_runtime=str(local))
        save()
        raise
    return result


def collected_qualification(directory: Path, parameters: dict[str, Any], *,
                            allow_compact_reuse: bool = False) -> dict[str, Any]:
    """Consume the existing exact-clock qualification and its terminal metadata."""
    # Preparation/execution do not require the collector's source verifier.
    from population_dynamics import pfc_source_execution

    bound = pfc_source_execution._qualification(directory, "numpy", 0.0125, parameters)
    delivery = load_json(directory / "delivery.json")
    files = entries(delivery)
    if "execution.json" not in files:
        if not allow_compact_reuse or "comparison/run.json" not in files:
            raise ValueError("Complete bounded stream execution metadata required")
        # Preserved compact proofs were admitted through the source executor's
        # delivery-bound qualification contract. Their earlier comparison run
        # can be FAILED; it is historical evidence, not the successful receipt.
        comparison_path = checked(directory, files["comparison/run.json"])
        comparison = load_json(comparison_path)
        receipt = bound["qualification"]
        job = delivery.get("job_id")
        if (not isinstance(job, str) or not job.isdecimal()
                or delivery.get("scientific_fidelity_qualified") is not False
                or delivery.get("production_episodes") != 0
                or comparison.get("status") not in (
                    "PASS_NATIVE_STREAM_SOURCE_EXECUTION", "FAILED_NATIVE_STREAM_SOURCE_EXECUTION")
                or comparison.get("scientific_fidelity_qualified") is not False
                or comparison.get("full_scale_qualified") is not False
                or comparison.get("production_episodes") != 0
                or any(comparison.get(key) != receipt.get(key) for key in (
                    "backend", "numpy_version", "source_entry_sha256", "executor_sha256",
                    "drive_sha256", "kernel_parameterization_sha256"))
                or receipt.get("scientific_fidelity_qualified") is not False
                or receipt.get("full_scale_qualified") is not False):
            raise ValueError("Compact reused qualification metadata differs")
        return dict(bound, terminal_metadata=dict(
            kind="compact_reused_qualification_delivery", producer_job=job,
            qualification_status=receipt["status"], comparison_run=str(comparison_path.resolve()),
            comparison_run_sha256=files["comparison/run.json"]["sha256"],
            archived_comparison=comparison))
    execution_path = checked(directory, files["execution.json"])
    execution = load_json(execution_path)
    if (execution.get("status") != PHASES["stream"][2]
            or not isinstance(execution.get("job_id"), str) or not execution["job_id"].isdecimal()
            or execution.get("scientific_fidelity_qualified") is not False
            or execution.get("production_episodes") != 0):
        raise ValueError("Complete bounded stream execution metadata required")
    if delivery.get("job_id") is not None and delivery["job_id"] != execution["job_id"]:
        raise ValueError("Stream delivery producer differs from terminal execution")
    if bound["qualification"].get("scientific_fidelity_qualified") is not False or bound["qualification"].get("full_scale_qualified") is not False:
        raise ValueError("Bounded exact-tuple qualification must retain scientific scope")
    return dict(bound, terminal_metadata=dict(kind="stream_execution",
        producer_job=execution["job_id"], status=execution["status"],
        execution=str(execution_path.resolve()), execution_sha256=files["execution.json"]["sha256"]))


def same_reference(value: Any, target: Path) -> bool:
    """Compare existing artifact identities across retained recovery aliases."""
    if not isinstance(value, str) or not value:
        return False
    try:
        return Path(value).resolve(strict=True) == target.resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def collect(root: Path, output: Path) -> dict[str, Any]:
    """Bind all100 tuples; missing or failed results never produce a subset."""
    prepared = validated_preparation(root)
    pending = [row for row in prepared["rows"] if row["status"] == PREPARED_STATUS]
    outcomes = {}
    for row in pending:
        path = root / "results" / f"circuit_{row['circuit_index']:03d}" / "runner_execution.json"
        if not path.is_file():
            raise ValueError(f"Complete100 intake withheld: missing result for circuit {row['circuit_index']}")
        result = load_json(path)
        if result.get("status") != "PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS":
            raise ValueError(f"Complete100 intake withheld: circuit {row['circuit_index']} status {result.get('status')}")
        outcomes[row["circuit_index"]] = result
    qualifications, bindings, assessment_chain = {}, [], None
    for row in sorted(prepared["rows"], key=lambda item: item["circuit_index"]):
        index = row["circuit_index"]
        if row["status"] == REUSE_STATUS:
            directory = Path(row["qualification"])
            bound = collected_qualification(directory, row["parameters"], allow_compact_reuse=True)
            origin = dict(status=REUSE_STATUS)
        else:
            result = outcomes[index]
            if (result.get("circuit_index") != index or result.get("parameters") != row["parameters"]
                    or result.get("qualification_inputs_sha256") != prepared["qualification_inputs_sha256"]
                    or result.get("source_qualification_granted") is not False
                    or result.get("scientific_parameter_domain_qualified") is not False
                    or result.get("production_episodes") != 0 or set(result.get("phases", {})) != set(PHASES)):
                raise ValueError(f"Circuit {index} runner identity/input/scope binding differs")
            assessment = admission(Path(result["assessment"]["path"]), prepared)
            if result["assessment"] != assessment or (assessment_chain is not None and assessment != assessment_chain):
                raise ValueError("Result assessment chain differs")
            assessment_chain = assessment
            phase_roots, deliveries, metadata = {}, {}, {}
            jobs = set()
            for phase, (_, _, status, prefix) in PHASES.items():
                phase_result = result["phases"][phase]
                archive = Path(phase_result["archive"])
                if (phase_result.get("status") != status or archive.parent.resolve() != (root / "results" / f"circuit_{index:03d}").resolve()
                        or not archive.name.startswith(prefix) or not archive.name[len(prefix):].isdecimal()
                        or digest(archive / "delivery.json") != phase_result["delivery_sha256"]):
                    raise ValueError(f"Circuit {index} {phase} archive binding differs")
                job = archive.name[len(prefix):]
                delivery = load_json(archive / "delivery.json")
                execution = load_json(checked(archive, entries(delivery)["execution.json"]))
                delivery_status = dict(native="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY",
                    pairing="ARCHIVED_NATIVE_INPUT_PAIRING", stream="ARCHIVED_PFC_STREAM_VALIDATION")[phase]
                if (delivery.get("status") != delivery_status
                        or execution.get("status") != status or execution.get("job_id") != job
                        or execution.get("scientific_fidelity_qualified") is not False
                        or execution.get("production_episodes") != 0):
                    raise ValueError(f"Circuit {index} {phase} terminal execution differs")
                jobs.add(job)
                phase_roots[phase], deliveries[phase], metadata[phase] = archive, delivery, execution
            if len(jobs) != 1:
                raise ValueError("Three exact-tuple phases must retain one producer job")
            native, pairing, directory = (phase_roots[name] for name in PHASES)
            native_files = entries(deliveries["native"])
            for key, archived_name in (("independent", "Libs/configs/independent.yaml"), ("global_config", "Libs/configs/global.yaml")):
                archived = checked(native, native_files[archived_name])
                if digest(archived) != digest(root / "inputs" / relative(row[key])):
                    raise ValueError(f"Circuit {index} archived native {key} differs from prepared config")
            if (not same_reference(deliveries["pairing"].get("source_validation"), native)
                    or not same_reference(metadata["pairing"].get("source_validation"), native)
                    or deliveries["pairing"].get("source_delivery_sha256") != digest(native / "delivery.json")
                    or not same_reference(deliveries["stream"].get("source_validation"), native)
                    or not same_reference(deliveries["stream"].get("pairing_source"), pairing)
                    or not same_reference(deliveries["stream"].get("global_reference"), native / "global_reference")):
                raise ValueError(f"Circuit {index} native/pairing/stream reference chain differs")
            bound = collected_qualification(directory, row["parameters"])
            global_delivery_path = checked(native, native_files["global_reference/delivery.json"])
            global_delivery = load_json(global_delivery_path)
            if global_delivery.get("status") != "ARCHIVED_NATIVE_GLOBAL_REFERENCE":
                raise ValueError(f"Circuit {index} completed global reference required")
            global_report = checked(native / "global_reference", entries(global_delivery)["plain/report.json"])
            receipt = bound["qualification"]
            if (receipt.get("global_reference_delivery_sha256") != digest(global_delivery_path)
                    or receipt.get("global_reference_report_sha256") != digest(global_report)):
                raise ValueError(f"Circuit {index} global reference qualification binding differs")
            origin = dict(status=result["status"], runner_execution=str((root / "results" / f"circuit_{index:03d}" / "runner_execution.json").resolve()),
                          runner_execution_sha256=digest(root / "results" / f"circuit_{index:03d}" / "runner_execution.json"), phases=result["phases"])
        qualifications[str(index)] = str(directory.resolve())
        bindings.append(dict(circuit_index=index, parameters=row["parameters"], qualification=str(directory.resolve()),
                             receipt_sha256=bound["receipt_sha256"], delivery_sha256=bound["delivery_sha256"],
                             terminal_metadata=bound["terminal_metadata"], **origin))
    report = dict(schema=SCHEMA, status="COMPLETE100_EXACT_TUPLE_REFERENCE_KERNEL_BINDINGS",
        prepared_execution=str((root / "execution.json").resolve()), prepared_execution_sha256=digest(root / "execution.json"),
        qualification_inputs_sha256=prepared["qualification_inputs_sha256"], assessment=assessment_chain,
        qualifications=qualifications, bindings=bindings, complete_circuits=100, reused_circuits=5, collected_circuits=95,
        scope="bounded native/input-stream exact-clock and exact-tuple qualification; not full-scale source/task/domain qualification",
        source_quality_qualified=False, source_qualification_granted=False, scientific_parameter_domain_qualified=False,
        scientific_claim_qualified=False, production_episodes=0, new_source_simulations=0)
    output.mkdir(parents=True, exist_ok=False)
    (output / "qualifications.json").write_text(json.dumps(qualifications, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=OPERATIONS / "parameter_qualification_preparation_634280/parameters/qualification_inputs.json")
    parser.add_argument("--software-delivery", type=Path, default=OPERATIONS / "source_validation_preparation/delivery.json")
    parser.add_argument("--template-root", type=Path, default=TEMPLATE_TUPLE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run one admitted exact tuple inside a Slurm batch allocation")
    parser.add_argument("--collect", action="store_true", help="Collect all100 completed bounded qualifications without simulation")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--circuit-index", type=int)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--resume-native-from", type=Path, help="Retained native runtime with validated complete plain cases")
    parser.add_argument("--attempt-output", type=Path, help="Fresh recovery attempt root; preserves original failed result")
    args = parser.parse_args()
    if args.execute and args.collect:
        parser.error("Select --execute or --collect")
    if args.collect:
        if args.prepared is None or args.output is None:
            parser.error("--collect requires --prepared and --output")
        result = collect(args.prepared.resolve(), args.output.resolve())
    elif args.execute or args.resume_native_from is not None:
        if args.prepared is None or args.circuit_index is None:
            parser.error("--execute requires --prepared and --circuit-index")
        result = execute(args.prepared.resolve(), args.circuit_index, args.assessment,
                         resume_native_from=args.resume_native_from.resolve() if args.resume_native_from else None,
                         attempt_output=args.attempt_output.resolve() if args.attempt_output else None)
    else:
        if args.output is None:
            parser.error("Preparation requires --output")
        result = prepare(args.inputs.resolve(), args.software_delivery.resolve(), args.template_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify two persisted V1 sm80 executions and their public CPU archive readers.

This entry point does not simulate, train, admit a production cohort, or modify
canonical production outputs. All source inputs retain their frozen paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import subprocess
import sys
import tarfile

import ca3_merlin_remote as transport

GIB = 1024 ** 3


RUNTIME_TEST_RUNNER = r"""import hashlib,json,os,pathlib,sys,traceback,unittest
work=pathlib.Path(sys.argv[1])
report_path=pathlib.Path(sys.argv[2])
sys.path.insert(0,str(work))
report=json.loads(report_path.read_text())
report['child_pid']=os.getpid()
report['python_version']=sys.version
try:
    for name,expected in report['expected_files'].items():
        data=(work/name).read_bytes()
        identity=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        report['tested_files'][name]=identity
        if identity != expected:
            raise ValueError('V1_RUNTIME_TEST_FILE_IDENTITY_DIFFERS:'+name)
    suite=unittest.defaultTestLoader.loadTestsFromName(report['test_module'])
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    report.update(testsRun=result.testsRun,failures=len(result.failures),
                  errors=len(result.errors),skips=len(result.skipped),
                  success=result.wasSuccessful(),status='TESTS_FINISHED')
except BaseException as error:
    report.update(success=False,status='FAILED',runner_error_type=type(error).__name__,
                  runner_error=str(error))
    traceback.print_exc()
finally:
    temporary=report_path.with_name(report_path.name+'.partial')
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temporary.replace(report_path)
raise SystemExit(0 if report['success'] else 1)
"""


def validate_receipt(receipt: dict, expected_library: str) -> None:
    if (receipt.get("execution_backend") != "merlin" or receipt.get("exit_code") != 0
            or receipt.get("persisted") is not True
            or receipt.get("compute_capability") != [8, 0]
            or receipt.get("library_sha256") != expected_library
            or type(receipt.get("result_bytes")) is not int or receipt["result_bytes"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("result_sha256"))) is None
            or not str(receipt.get("checkpoint_path", "")).startswith("hdfs://")):
        raise ValueError("V1_COMPLETED_PERSISTED_SM80_RECEIPT_REQUIRED")
    for name in ("merlin_job_run_id", "merlin_trial_id", "hostname"):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            raise ValueError("V1_REAL_MERLIN_RECEIPT_IDENTITY_REQUIRED:" + name)
    output = PurePosixPath(receipt.get("output", ""))
    if not output.is_absolute() or ".." in output.parts or len(output.parts) < 3:
        raise ValueError("V1_SOURCE_RECEIPT_OUTPUT_PATH_INVALID")


def platform_identity(document: dict, expected: str, kind: str, expected_run: str) -> dict:
    """Bind the observed run/trial schema without searching unrelated nested IDs."""
    response = document.get("ResponseBase", {})
    data = document.get("data")
    if response.get("code") != 0 or not isinstance(data, dict) or data.get("sid") != expected:
        raise ValueError("V1_PLATFORM_RESPONSE_IDENTITY_NOT_VERIFIED:" + kind)
    if kind == "run":
        if data.get("status") not in {"done", "failed", "stopped"}:
            raise ValueError("V1_SOURCE_RUN_TERMINAL_OBSERVATION_REQUIRED")
    elif kind == "trial":
        if data.get("job_run_sid") != expected_run or not data.get("completed_at"):
            raise ValueError("V1_COMPLETED_TRIAL_AND_RUN_ASSOCIATION_REQUIRED")
    else:
        raise ValueError("V1_PLATFORM_RECORD_KIND_INVALID")
    # Scientific success comes from the verified zero-exit receipt and arrays.
    # Preserve a failed platform terminal rather than relabeling it as success.
    return dict(kind=kind, identity=expected, response_code=response["code"],
                platform_status=data.get("status"), job_run_sid=expected_run,
                completed_at=data.get("completed_at"))


def cpu_preflight(work: Path, shape: dict) -> dict:
    if int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0")) != 0 or shape["gpu"] != 0:
        raise ValueError("V1_READOUT_REQUIRES_CPU_ONLY_ALLOCATION")
    if shape["memory_mb"] < 16 * 1024:
        raise ValueError("V1_READOUT_ORIGINAL_16_GIB_MEMORY_REQUEST_REQUIRED")
    return dict(memory_evidence=transport.memory_evidence(shape, required_available_gib=8),
                required_available_memory_gib=8, resource_shape=shape,
                cpu_affinity_count=len(os.sched_getaffinity(0)))


def run_runtime_tests(work: Path, output: Path, specification: dict) -> dict:
    """Run staged memory tests in a child before importing scientific readers."""
    report_path = output / "runtime_memory_tests.json"
    log_path = output / "runtime_memory_tests.log"
    report = dict(
        status="NOT_RUN", scope="isolated_child_process", testsRun=0,
        failures=0, errors=0, skips=0, success=False,
        test_module=specification["module"], expected_tests=specification["expected_tests"],
        expected_files=specification["files"], tested_files={}, log_path=str(log_path))
    transport.write_json(report_path, report)
    completed = None
    try:
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", "-c", RUNTIME_TEST_RUNNER,
                 str(work), str(report_path)], cwd=work, stdout=log,
                stderr=subprocess.STDOUT, check=False)
    finally:
        report = json.loads(report_path.read_text())
        success = (
            completed is not None and completed.returncode == 0 and report.get("success") is True
            and report.get("testsRun") == specification["expected_tests"]
            and all(report.get(key) == 0 for key in ("failures", "errors", "skips"))
            and report.get("tested_files") == specification["files"])
        report.update(status="PASS" if success else "FAILED", success=success,
                      child_returncode=completed.returncode if completed is not None else None)
        transport.write_json(report_path, report)
    return report


def restore_result(receipt: dict, work: Path, label: str) -> tuple[Path, dict]:
    """Verify an existing pack_results bundle before using its source execution."""
    destination = work / (label + ".tar.gz")
    subprocess.run(["hdfs", "dfs", "-get", receipt["checkpoint_path"].rstrip("/") + "/result.tar.gz",
                    str(destination)], check=True)
    if (destination.stat().st_size != receipt["result_bytes"]
            or transport.digest(destination) != receipt["result_sha256"]):
        raise ValueError("V1_RETURNED_RESULT_ARCHIVE_IDENTITY_DIFFERS:" + label)
    root = Path(receipt["output"])
    with tarfile.open(destination, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or names.count("merlin_return_manifest.json") != 1:
            raise ValueError("V1_RETURNED_RESULT_MANIFEST_OR_MEMBER_NAMES_INVALID")
        manifest_member = archive.getmember("merlin_return_manifest.json")
        if not manifest_member.isfile() or manifest_member.size > 16 * 1024 * 1024:
            raise ValueError("V1_RETURN_MANIFEST_INVALID")
        incoming = archive.extractfile(manifest_member)
        if incoming is None:
            raise ValueError("V1_RETURN_MANIFEST_UNREADABLE")
        with incoming:
            manifest = json.load(incoming)
        if not isinstance(manifest, dict) or set(names) != set(manifest) | {"merlin_return_manifest.json"}:
            raise ValueError("V1_RETURN_MANIFEST_MEMBER_SET_DIFFERS")
        expected = {}
        for name, row in manifest.items():
            relative = PurePosixPath(name)
            target = Path("/") / name
            if (relative.is_absolute() or ".." in relative.parts or root not in target.parents
                    or name != str(target).lstrip("/") or type(row.get("bytes")) is not int
                    or row["bytes"] < 0 or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None):
                raise ValueError("V1_RETURN_MANIFEST_SCOPE_OR_IDENTITY_INVALID")
            expected[name] = row
        if shutil.disk_usage(work).free < sum(row["bytes"] for row in expected.values()) + 20 * GIB:
            raise ValueError("V1_RETURN_EXTRACTION_STORAGE_INSUFFICIENT")
        for member in members:
            if member.name == "merlin_return_manifest.json":
                continue
            row = expected[member.name]
            target = Path("/") / member.name
            if not member.isfile() or member.size != row["bytes"] or target.resolve() != target:
                raise ValueError("V1_RETURN_MEMBER_TYPE_SIZE_OR_PATH_INVALID")
            if target.exists():
                if (not target.is_file() or target.stat().st_size != row["bytes"]
                        or transport.digest(target) != row["sha256"]):
                    raise ValueError("V1_EXISTING_RETURNED_FILE_CONFLICT")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            incoming = archive.extractfile(member)
            if incoming is None:
                raise ValueError("V1_RETURN_MEMBER_UNREADABLE")
            value = hashlib.sha256()
            temporary = target.with_name(target.name + ".merlin-readout-partial")
            with incoming, temporary.open("xb") as outgoing:
                while data := incoming.read(8 * 1024 * 1024):
                    outgoing.write(data)
                    value.update(data)
            if temporary.stat().st_size != row["bytes"] or value.hexdigest() != row["sha256"]:
                raise ValueError("V1_RETURN_MEMBER_CHECKSUM_DIFFERS")
            temporary.chmod(member.mode & 0o777)
            os.link(temporary, target)
            temporary.unlink()
    terminal = json.loads((root / "merlin_terminal.json").read_text())
    for name in ("merlin_job_run_id", "merlin_trial_id", "hostname", "output",
                 "compute_capability", "library_sha256", "exit_code"):
        if terminal.get(name) != receipt.get(name):
            raise ValueError("V1_ARCHIVED_TERMINAL_AND_RECEIPT_DIFFER:" + name)
    return root, dict(status="VERIFIED", files=len(manifest), archive_sha256=receipt["result_sha256"],
                      merlin_job_run_id=receipt["merlin_job_run_id"], merlin_trial_id=receipt["merlin_trial_id"],
                      hostname=receipt["hostname"], output=str(root))


def verify_public_modules(input_files: dict) -> dict:
    bindings = {}
    for name, module in tuple(sys.modules.items()):
        if name != "population_dynamics" and not name.startswith("population_dynamics."):
            continue
        location = getattr(module, "__file__", None)
        if location is None:
            raise ValueError("V1_UNBOUND_PUBLIC_MODULE:" + name)
        path = str(Path(location).resolve())
        expected = input_files.get(path)
        if expected is None or transport.digest(Path(path)) != expected["sha256"]:
            raise ValueError("V1_PUBLIC_READER_MODULE_BINDING_DIFFERS:" + name)
        bindings[name] = dict(path=path, sha256=expected["sha256"])
    return bindings


def validate_source_block(root: Path, job: dict) -> None:
    header = json.loads((root / "coupled/report.json").read_text())
    delivery = json.loads((root / "merlin_terminal.json").read_text())
    if (header["circuit_index"] != 0 or header["block_index"] != 0
            or header["cases_sha256"] != job["input_files"][job["source_job"]["cases"]]["sha256"]
            or str(header["job_id"]) != str(delivery["execution_id"])):
        raise ValueError("V1_RETURNED_BLOCK_FROZEN_CASE_BINDING_DIFFERS")
    for index, row in enumerate(job["sources"]["conditions"]):
        case = json.loads((root / f"coupled/condition_{index:02d}/report.json").read_text())
        if (case["input_archive"] != row["sources"]["spike_source"]
                or case["circuit_reference"] != job["source_job"]["reference"]):
            raise ValueError("V1_RETURNED_CASE_SOURCE_INPUT_DIFFERS")


def native_reference_match(source_job: dict, input_files: dict) -> dict:
    """Select the native-matched case using bound JSON and event-file hashes only."""
    metadata = {}

    def read_bound(path: Path) -> dict:
        name = str(path)
        expected = input_files.get(name)
        if expected is None:
            raise ValueError("V1_NATIVE_MATCH_METADATA_NOT_STAGED:" + name)
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("V1_NATIVE_MATCH_METADATA_BINDING_DIFFERS:" + name)
        metadata[name] = expected
        return json.loads(raw)

    reference = Path(source_job["reference"])
    native = read_bound(reference / "primary/report.json")
    cases = read_bound(Path(source_job["cases"]))
    if (cases["circuit_index"] != 0 or cases["block_index"] != 0
            or [row["identity"]["condition_index"] for row in cases["cases"]] != list(range(10))):
        raise ValueError("V1_NATIVE_MATCH_COMPLETE_C0_B0_CASES_REQUIRED")
    native_inputs = native["bindings"]["spike_inputs"]
    native_clock = {key: native["execution_clock"][key] for key in ("duration_ms", "resolution_ms")}
    if native_inputs["source_clock"] != native_clock:
        raise ValueError("V1_NATIVE_MATCH_REFERENCE_CLOCK_DIFFERS")
    event_names = tuple(f"{population}/trial_0000.{extension}"
                        for population in ("bkg", "lgn") for extension in ("npz", "h5"))
    native_arrays = {}
    for name in ("delivery.json", "primary/native/input_events.npz",
                 "primary/native/cortical_state.npz", "primary/native/cortical_spikes.npz"):
        key = str(reference / name)
        if key not in input_files:
            raise ValueError("V1_NATIVE_REFERENCE_ARRAY_OR_DELIVERY_NOT_STAGED:" + key)
        native_arrays[key] = input_files[key]
    matches = []
    for case in cases["cases"]:
        if case["input_config"] != native_inputs["config"]:
            continue
        archive = Path(case["input_archive"])
        report_path = archive / "visual_inputs/report.json"
        report = read_bound(report_path)
        if metadata[str(report_path)]["sha256"] != case["input_report_sha256"]:
            raise ValueError("V1_NATIVE_MATCH_CASE_REPORT_BINDING_DIFFERS")
        if (report["config"] != native_inputs["config"]
                or report["source_clock"] != native_clock
                or report["input_construction_binding"] != native_inputs["input_construction_binding"]):
            continue
        event_bindings = {}
        for name in event_names:
            expected = native_inputs["files"]["visual_inputs/" + name]
            candidate = input_files.get(str(archive / "visual_inputs" / name))
            if (report["artifacts"].get(name) != expected["sha256"]
                    or candidate != expected):
                break
            event_bindings[name] = expected
        else:
            matches.append(dict(
                condition_index=case["identity"]["condition_index"],
                input_archive=str(archive), input_event_file_bindings=event_bindings,
                candidate_input_report_sha256=case["input_report_sha256"]))
    if len(matches) != 1:
        raise ValueError("V1_UNIQUE_NATIVE_INPUT_METADATA_MATCH_REQUIRED:" + str(len(matches)))
    return dict(
        status="MATCHED_NATIVE_REFERENCE_INPUT_METADATA", reference=str(reference),
        **matches[0], execution_clock=native["execution_clock"],
        native_input_report_sha256=native_inputs["bundle_report_sha256"],
        native_report_sha256=metadata[str(reference / "primary/report.json")]["sha256"],
        metadata_bindings=metadata, native_array_bindings=native_arrays,
        numerical_comparison="NOT_RUN", array_contents_read=False)


def public_readout(job: dict, primary: Path, replay: Path, output: Path) -> dict:
    public = Path(job["public_modules"])
    os.environ["PYTHONPATH"] = str(public.parent)
    sys.path.insert(0, str(public.parent))
    from population_dynamics.v1_glif_production import compare_executions, local_delivery, validate_saved_reference
    from population_dynamics.v1_benchmark import inspect_source_movie, export_source_controls
    from population_dynamics.v1_glif_dataset import export_block
    verify_public_modules(job["input_files"])
    for source in (primary, replay):
        validate_source_block(source, job)
    comparison = compare_executions(primary, replay, output / "source_replay")
    if (comparison["status"] != "PASS_GLIF_SOURCE_BLOCK_REPLAY"
            or comparison["passed_conditions"] != 10 or comparison["failed_conditions"] != 0
            or any(row["library_sha256"] != job["expected_library_sha256"]
                   for row in comparison["conditions"])):
        raise ValueError("V1_FULL_TEN_CONDITION_REPLAY_FAILED")
    native_match = native_reference_match(job["source_job"], job["input_files"])
    if native_match != job["native_reference_match"]:
        raise ValueError("V1_NATIVE_REFERENCE_MATCH_CHANGED_AFTER_PREPARATION")
    native_validation = validate_saved_reference(
        Path(native_match["reference"]), primary, native_match["condition_index"],
        output / "native_reference")
    if native_validation["status"] != "PASS_SAVED_NATIVE_GLIF_CUDA_REFERENCE":
        raise ValueError("V1_MATCHED_NATIVE_REFERENCE_COMPARISON_FAILED")
    controls = []
    for row in job["sources"]["conditions"]:
        index, sources = row["condition_index"], row["sources"]
        movie = output / f"movie_{index:02d}"
        commands = output / f"controls_{index:02d}"
        result = inspect_source_movie(Path(sources["rate_source"]), movie / "movie")
        if result["status"] != "SOURCE_MOVIE_CONTROL_WITNESS_PASS":
            raise ValueError("V1_SOURCE_MOVIE_CONTROL_WITNESS_FAILED")
        local_delivery(movie, "movie")
        export_source_controls(Path(sources["rate_source"]), movie,
                               Path(sources["spike_source"]), commands / "controls")
        local_delivery(commands, "controls")
        controls.append(dict(condition_index=index, control_archive=str(commands)))
    controls_file = output / "controls.json"
    transport.write_json(controls_file, dict(cases=controls))
    authority = Path(job["sources"]["conditions"][0]["sources"]["authority_source"])
    exports = {}
    for label, source in (("primary", primary), ("replay", replay)):
        result = export_block(source, controls_file, authority, output / ("export_" + label))
        if (result["status"] != "EXPORTED_VERIFIED_GLIF_SOURCE_BLOCK" or result["exit_code"] != 0
                or result["complete_episodes"] != 10 or result["production_episodes"] != 0
                or len(result["verification"]) != 10):
            raise ValueError("V1_COMPLETE_PUBLIC_EXPORT_READBACK_FAILED:" + label)
        exports[label] = result
    bindings = verify_public_modules(job["input_files"])
    return dict(
        status="PASS_V1_SM80_REPLAY_AND_PUBLIC_READOUT",
        backend_qualification=dict(status="PASS_FULL_BLOCK_REPLAY", comparison=comparison,
                                   scope="two distinct sm80 source executions",
                                   native_equivalence_comparison=native_validation,
                                   native_reference_match=native_match,
                                   native_equivalence_scope="matched C0 native condition on the primary execution"),
        public_readout=dict(status="PASS", exports=exports, module_bindings=bindings),
        production_gate="NOT_MODIFIED_OR_GRANTED", production_episodes=0,
        new_unique_episodes=0, source_simulations=0, source_condition_identities=10,
        verified_execution_exports=2, training_seeds_used=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    run = os.environ.get("RH2_JOB_RUN_ID")
    trial = os.environ.get("ARNOLD_TRIAL_ID")
    checkpoint = os.environ.get("RH2_CHECKPOINT_HDFS")
    if not run or not trial or not checkpoint or socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("ACTUAL_MERLIN_CPU_RUN_TRIAL_AND_CHECKPOINT_REQUIRED")
    job = json.loads(args.job_json.read_text())
    for source in ("primary", "replay"):
        validate_receipt(job[source], job["expected_library_sha256"])
    if (job["primary"]["merlin_job_run_id"] == job["replay"]["merlin_job_run_id"]
            or job["primary"]["merlin_trial_id"] == job["replay"]["merlin_trial_id"]):
        raise ValueError("TWO_DISTINCT_MERLIN_SOURCE_RUNS_AND_TRIALS_REQUIRED")
    work = args.job_json.resolve().parent
    output = work / "output"
    output.mkdir(exist_ok=False)
    compatibility = "merlin_" + trial + "_v1_cpu_readout"
    os.environ.update(CUDA_VISIBLE_DEVICES="", SLURM_JOB_ID=compatibility,
                      SLURM_CPUS_PER_TASK=str(job["resource_shape"]["cpu"]),
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      PYTHONDONTWRITEBYTECODE="1", NRNSIM_EXECUTION_BACKEND="merlin")
    receipt = dict(
        status="PREPARING_V1_CPU_READOUT", execution_backend="merlin", merlin_job_run_id=run,
        merlin_trial_id=trial, hostname=socket.gethostname(), output=str(output),
        legacy_compatibility_id=compatibility,
        compatibility_id_semantics="SLURM_JOB_ID interface compatibility only; not a Slurm execution",
        started_utc=datetime.now(timezone.utc).isoformat(), production_episodes=0,
        source_simulations=0, production_gate="NOT_MODIFIED_OR_GRANTED", persisted=False,
        platform_source_evidence=job["platform_evidence"])
    transport.write_json(output / "merlin_execution.json", receipt)
    target = checkpoint.rstrip("/") + "/v1/cpu_readout/" + trial
    target_ready = False
    exit_code = 1
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        target_ready = True
        receipt["runtime_tests"] = run_runtime_tests(work, output, job["runtime_tests"])
        for name in ("runtime_memory_tests.log", "runtime_memory_tests.json"):
            transport.hdfs_put(output / name, target + "/" + name)
        if receipt["runtime_tests"]["success"] is not True:
            raise ValueError("V1_RUNTIME_MEMORY_TESTS_FAILED")
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        if receipt["resource_preflight"]["memory_evidence"]["admission"]["status"] != "PASS":
            raise ValueError("V1_PUBLIC_EXPORT_MEMORY_ADMISSION_REQUIRED")
        receipt["environment_versions"] = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "h5py", "PyYAML", "psutil", "matplotlib", "scikit-image", "sympy")}
        stage = work / "source_inputs"
        stage.mkdir()
        # Shared transport needs only upload and input_files. It does not apply
        # CA3 resource admission or execute any CA3 scientific function.
        archive = transport.download_inputs(
            {"upload": job["upload"], "input_files": job["input_files"]}, stage)
        transport.stage_inputs(archive, job["input_files"])
        primary, primary_verified = restore_result(job["primary"], work, "primary")
        replay, replay_verified = restore_result(job["replay"], work, "replay")
        receipt["verified_source_returns"] = dict(primary=primary_verified, replay=replay_verified)
        transport.write_json(output / "verified_source_receipts.json",
                             dict(primary=primary_verified, replay=replay_verified))
        result = public_readout(job, primary, replay, output)
        transport.write_json(output / "readout_report.json", result)
        receipt.update(status=result["status"], readout_report=str(output / "readout_report.json"),
                       validation_exit_code=0)
        exit_code = 0
    except BaseException as error:
        receipt.update(status="FAILED_V1_CPU_READOUT", error_type=type(error).__name__, error=str(error))
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        receipt.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
        transport.write_json(output / "merlin_terminal.json", receipt)
        if target_ready:
            try:
                transport.persist(output, work, target, receipt)
            except BaseException as error:
                receipt.update(persisted=False, persistence_error_type=type(error).__name__)
                transport.write_json(output / "merlin_terminal.json", receipt)
                exit_code = exit_code or 1
        print(json.dumps(dict(status=receipt["status"], persisted=receipt["persisted"],
                              merlin_job_run_id=run, merlin_trial_id=trial,
                              exit_code=exit_code, production_episodes=0)), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

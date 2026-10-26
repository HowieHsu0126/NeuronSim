#!/usr/bin/env python3
"""Check the CA3 composition consumer and real library identities on Merlin CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import unittest

from ca3_merlin_remote import (compose_candidate, digest, download_inputs, persist,
                               source_command, stage_inputs, verify_scientific_inputs, write_json,
                               persistence_batch_fingerprint, persistence_condition_inputs,
                               validate_persistence_batch, validate_completion75_recovery)
from prepare_ca3_merlin_replay import validate_candidate, validate_inputs
from v1_merlin_readout import cpu_preflight


def execute(job: dict, work: Path) -> int:
    if "candidate_runtime" in job:
        return execute_frozen_candidate(job, work)
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/ca3/runtime_composition/" + trial
    output = work / "output"
    output.mkdir(exist_ok=False)
    receipt = dict(status="FAILED_CA3_RUNTIME_COMPOSITION_VALIDATION", exit_code=1,
        merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(),
        execution_backend="merlin", output=str(output), scope="cpu_runtime_composition_validation",
        started_utc=datetime.now(timezone.utc).isoformat(), persisted=False,
        simulations_run=0, production_episodes=0, candidate_qualified=False,
        scientific_fidelity_qualified=False)
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        archive = download_inputs(job, work)
        stage_inputs(archive, job["input_files"])
        sys.path.insert(0, job["public_root"])
        # The bound module closure is available only after input verification.
        from population_dynamics.ca3_benchmark import validate_runtime_composition

        suite = unittest.defaultTestLoader.loadTestsFromName(
            "population_dynamics.test_ca3_runtime_composition")
        expected = suite.countTestCases()
        with (output / "tests.log").open("x") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        tests = dict(tests_run=result.testsRun, expected_tests=expected,
                     failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
        write_json(output / "tests.json", tests)
        if not result.wasSuccessful() or result.skipped or result.testsRun != 15:
            raise ValueError("CA3_RUNTIME_COMPOSITION_TESTS_FAILED")
        parent = Path(job["parent_runtime"])
        runtime = work / "candidate_runtime"
        shutil.copytree(parent, runtime)
        candidate = Path(job["candidate_root"])
        library = "libcarlsim.so.6.0.0"
        shutil.copy2(candidate / "lib" / library, runtime / "lib" / library)

        def identity(path: Path) -> dict:
            return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))

        composition = dict(runtime_root=str(runtime),
            parent_runtime=dict(root=str(parent), build_report_sha256=digest(parent / "build_report.json")),
            candidate_library=dict(name=library, **{key: value for key, value in
                identity(runtime / "lib" / library).items() if key != "path"}),
            candidate_build_report=identity(candidate / "backend_build_report.json"),
            candidate_receipt=identity(candidate / "receipt.json"))
        descriptor = output / "runtime_composition.json"
        write_json(descriptor, composition)
        built = json.loads((parent / "build_report.json").read_text())
        libraries, verified = validate_runtime_composition(runtime, built, descriptor)
        write_json(output / "real_composition.json", dict(
            status="PASS_REAL_RUNTIME_LIBRARY_COMPOSITION", effective_runtime_libraries=libraries,
            runtime_composition=verified, driver_executed=False, prefix_comparison_run=False,
            scientific_fidelity_qualified=False, production_episodes=0))
        write_json(output / "execution_inputs.json", job)
        receipt.update(status="PASS_CA3_RUNTIME_COMPOSITION_CONSUMER", exit_code=0,
                       tests=tests, real_runtime_library_composition="PASS")
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "merlin_terminal.json", receipt)
        try:
            persist(output, work, target, receipt, chunk_bytes=256 * 1024)
        except BaseException as error:
            receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                           persistence_error_type=type(error).__name__, persistence_error=str(error))
            write_json(output / "merlin_terminal.json", receipt)
        print(json.dumps({key: receipt.get(key) for key in
                          ("status", "exit_code", "persisted", "tests", "error")}), flush=True)
    return receipt["exit_code"]


def execute_frozen_candidate(job: dict, work: Path) -> int:
    """Exercise the exact frozen dependency closure and native ELF loading on CPU."""
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    condition_mode = job.get("execution_mode") == "persistence_conditions"
    recovery_mode = job.get("execution_mode") == "completion75_recovery"
    suffix = ("/ca3/completion75_recovery_preflight/" if recovery_mode else
              "/ca3/persistence_condition_preflight/" if condition_mode else "/ca3/frozen_candidate_preflight/")
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + suffix + trial
    output = work / "output"
    output.mkdir(exist_ok=False)
    receipt = dict(status="FAILED_CA3_FROZEN_CANDIDATE_PREFLIGHT", exit_code=1,
        merlin_job_run_id=run, merlin_trial_id=trial, hostname=socket.gethostname(),
        execution_backend="merlin", output=str(output), scope="cpu_frozen_candidate_preflight",
        started_utc=datetime.now(timezone.utc).isoformat(), persisted=False,
        simulations_run=0, production_episodes=0, candidate_qualified=False,
        scientific_fidelity_qualified=False)
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        validate_inputs(job["execution_inputs"], job["input_files"])
        validate_candidate(job["candidate_runtime"], job["input_files"])
        stage_inputs(download_inputs(job, work), job["input_files"])
        verify_scientific_inputs(job)
        if condition_mode:
            receipt["persistence_prerequisite"] = validate_persistence_batch(job)
        if recovery_mode:
            receipt["recovery_prerequisite"] = validate_completion75_recovery(job)
        layer_work = work / "candidate_inputs"
        layer_work.mkdir(exist_ok=False)
        candidate = job["candidate_runtime"]
        stage_inputs(download_inputs(candidate, layer_work), candidate["input_files"])
        execution = compose_candidate(job, work)
        job["candidate_execution"] = execution
        from population_dynamics import ca3_benchmark as benchmark
        frozen = job["execution_inputs"]
        campaign = benchmark.campaign_config(Path(frozen["config"]))
        identity = next(row for row in benchmark.episode_identities(campaign, "ca3_pattern_completion")
                        if (row["circuit_index"], row["block_index"], row["condition_index"]) == (0, 0, 3))
        formation, analysis = Path(frozen["formation"]), Path(frozen["analysis"])
        realization = benchmark.retrieval_realization(
            json.loads((formation / "native_report.json").read_text()), identity)
        input_binding = benchmark.qualify_input(
            formation, analysis / "formation_replay.json", analysis / "network65p_assem550_divnorm.dat",
            analysis / "normalization.json", analysis / "normalization_replay.json",
            coupled_extension=True, source_realization=realization)
        write_json(output / "source_input_binding.json", input_binding)
        source = Path(execution["source_repo"])
        tests_path = source / "Libs/population_dynamics/test_ca3_runtime_composition.py"
        shutil.copy2(work / tests_path.name, tests_path)
        environment = dict(os.environ, PYTHONPATH=str(source / "Libs"),
            CA3_PREFLIGHT_JOB=str(work / "job.json"),
            CA3_FROZEN_CONSUMER=str(Path(job["execution_inputs"]["source_repo"]) /
                                    "Libs/population_dynamics/ca3_benchmark.py"),
            CA3_CURRENT_CONSUMER=str(work / "current_ca3_benchmark.py"))
        command = [sys.executable, "-m", "unittest", "-v",
                   "population_dynamics.test_ca3_runtime_composition",
                   "test_ca3_merlin_replay.TestCA3MerlinReplay"]
        expected_tests = 22
        if condition_mode or recovery_mode:
            batch_job = work / ("persistence_regression_job.json" if recovery_mode else "job.json")
            environment.update(CA3_PERSISTENCE_JOB=str(batch_job),
                               CA3_COMPOSED_SOURCE=str(source))
            command.append("test_ca3_merlin_replay.TestCA3PersistenceConditions")
            expected_tests += 10
            planned = []
            rows = [{"condition_index": 4}] if recovery_mode else job["persistence_batch"]["conditions"]
            for row in rows:
                selected, condition = persistence_condition_inputs(job, row["condition_index"])
                item = dict(identity=selected, condition=condition,
                    command=source_command(dict(job, condition_index=row["condition_index"]),
                                           output / ("condition_" + str(row["condition_index"])) / "episode"))
                if not recovery_mode:
                    item["source_job_id"] = row["source_job_id"]
                planned.append(item)
            write_json(output / "planned_conditions.json", dict(conditions=planned, simulations_run=0))
        if recovery_mode:
            environment["CA3_COMPLETION75_JOB"] = str(work / "job.json")
            command.append("test_ca3_merlin_replay.TestCA3Completion75Recovery")
            expected_tests += 10
        with (output / "tests.log").open("x") as log:
            result = subprocess.run(command, cwd=work, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError("CA3_FROZEN_CANDIDATE_TESTS_FAILED")
        count = re.search(r"^Ran (\d+) tests? in ", (output / "tests.log").read_text(), re.MULTILINE)
        if count is None or int(count[1]) != expected_tests or "skipped=" in (output / "tests.log").read_text():
            raise ValueError("CA3_FROZEN_CANDIDATE_TEST_COUNT_DIFFERS")
        write_json(output / "tests.json", dict(tests_run=int(count[1]), exit_code=0,
                                               dependency_scope="frozen_software_composition"))
        # --list resolves ELF dependencies without entering the simulator main.
        from population_dynamics.ca3_formation_execution import native_launch_command
        runtime = Path(execution["packaged"])
        launch = native_launch_command(runtime, runtime / "bin/ca3_benchmark")
        loader_command = [launch[0], "--list", *launch[1:]]
        loaded = subprocess.run(loader_command, capture_output=True, text=True,
                                env=dict(os.environ, LD_LIBRARY_PATH=str(runtime / "lib")))
        (output / "elf_loader.log").write_text(loaded.stdout + loaded.stderr)
        if loaded.returncode or "not found" in loaded.stdout + loaded.stderr:
            raise ValueError("CA3_CANDIDATE_ELF_DEPENDENCY_RESOLUTION_FAILED")
        for name in execution["effective_runtime_libraries"]:
            if str(runtime / "lib" / name) not in loaded.stdout:
                raise ValueError("CA3_CANDIDATE_ELF_LIBRARY_NOT_SELECTED:" + name)
        with (output / "consumer_help.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "population_dynamics.ca3_benchmark", "run", "--help"],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if "--runtime-composition" not in (output / "consumer_help.log").read_text():
            raise ValueError("CA3_CANDIDATE_CLI_COMPOSITION_OPTION_MISSING")
        write_json(output / "execution_inputs.json", job)
        receipt.update(status="PASS_CA3_FROZEN_CANDIDATE_PREFLIGHT", exit_code=0,
            candidate_input_files=candidate["input_files"],
            source_file_bindings=job["execution_inputs"]["source_file_bindings"],
            runtime_build_sha256=job["execution_inputs"]["runtime_build_sha256"],
            remote_sha256=digest(work / "ca3_merlin_remote.py"), image_meta=job["image_meta"],
            tests_exit_code=result.returncode, tests_run=int(count[1]), elf_loader_exit_code=loaded.returncode,
            elf_loader_command=loader_command, driver_main_executed=False,
            candidate_command=(source_command(dict(job, condition_index=job["persistence_batch"]["conditions"][0]["condition_index"]),
                                               output / "episode") if condition_mode
                               else source_command(job, output / "episode")))
        if condition_mode:
            receipt.update(status="PASS_CA3_PERSISTENCE_CONDITION_PREFLIGHT",
                           persistence_batch_sha256=persistence_batch_fingerprint(job["persistence_batch"]),
                           condition_indices=[row["condition_index"] for row in job["persistence_batch"]["conditions"]])
        if recovery_mode:
            receipt.update(status="PASS_CA3_COMPLETION75_RECOVERY_PREFLIGHT", condition_index=4,
                           completion75_recovery_sha256=persistence_batch_fingerprint(job["completion75_recovery"]),
                           administrative_cancellation_cause_resolved=False,
                           retry_branch="relevant_cluster_condition_changed")
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "merlin_terminal.json", receipt)
        try:
            persist(output, work, target, receipt, chunk_bytes=256 * 1024)
        except BaseException as error:
            receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                           persistence_error_type=type(error).__name__, persistence_error=str(error))
            write_json(output / "merlin_terminal.json", receipt)
        print(json.dumps({key: receipt.get(key) for key in
                          ("status", "exit_code", "persisted", "error")}), flush=True)
    return receipt["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.job_json.read_text()), args.job_json.resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())

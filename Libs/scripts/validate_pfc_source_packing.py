"""Validate PFC source packing and deliberate failure accounting through Slurm."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualified-stream", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-batch", type=Path)
    parser.add_argument("--fault-batch", type=Path)
    args = parser.parse_args()
    tasks = int(os.environ.get("SLURM_NTASKS", "0"))
    if (os.environ.get("SLURMD_NODENAME") not in {"cpu03", "cpu04", "cpu05", "cpu06"} or not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
            or tasks not in (4, 8)
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) != 1):
        raise ValueError("Authorized four- or eight-task one-CPU authorized Slurm allocation required")
    repo = Path(__file__).resolve().parents[2]
    job = os.environ["SLURM_JOB_ID"]
    args.output.mkdir(parents=True, exist_ok=False)
    work = Path(tempfile.mkdtemp(prefix="neuronsim-pfc-packing-check-" + job + ".", dir="/tmp"))
    report = dict(status="RUNNING", job_id=job, runtime=str(work), tasks=tasks, commands=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_fidelity_qualified=False, production_episodes=0)
    code = 1
    sys.path.insert(0, str(repo / "Libs"))
    from population_dynamics.resource_probe import probe
    from population_dynamics.provenance import sha256_file

    def run(label, command, env, expected=0):
        log = work / (label + ".log")
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report["commands"].append(dict(label=label, command=command, exit_code=result.returncode))
        if result.returncode != expected:
            raise RuntimeError(label + " unexpected exit code " + str(result.returncode))
        return result

    def verified_reference(archive, name, label):
        receipt = json.loads((archive / "delivery.json").read_text())
        rows = [row for row in receipt["files"] if row["path"] == name]
        if len(rows) != 1:
            raise ValueError("Unique reference binding required: " + name)
        destination = work / "references" / label / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive / name, destination)
        row = rows[0]
        if destination.stat().st_size != row["bytes"] or sha256_file(destination) != row["sha256"]:
            raise ValueError("Changed qualified reference: " + name)
        return destination

    try:
        admission = probe(work / "admission.json", required_memory_gib=2 * tasks)
        if admission["status"] != "PASS":
            raise RuntimeError("RESOURCE_BLOCKED:" + ",".join(admission["reasons"]))
        package = work / "Libs/population_dynamics"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"")
        for name in ("pfc_source_execution", "pfc_deterministic_replay", "pfc_native_drive",
                     "pfc_input_pairing", "provenance", "resource_probe", "test_pfc_source_launcher", "pfc_kernel_parameters"):
            shutil.copy2(repo / "Libs/population_dynamics" / (name + ".py"), package / (name + ".py"))
        for name in ("run_pfc_source_cases.py", "run_pfc_source_cases.sbatch",
                     "run_pfc_source_dataset.py", "validate_pfc_source_packing.py"):
            target = work / "Libs/scripts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / "Libs/scripts" / name, target)
        for name in ("pfc_source_readouts.sbatch",):
            target = work / "Output/neural_population_dynamics/operations" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / "Output/neural_population_dynamics/operations" / name, target)
        shutil.copy2(args.config, work / "config.yaml")
        shutil.copy2(Path(__file__), work / "validate_pfc_source_packing.py")
        config = yaml.safe_load((work / "config.yaml").read_text())
        if (config["excitatory_neurons"] != 1024 or config["inhibitory_neurons"] != 256
                or config["duration_ms"] != 800 or config["dt_ms"] != .025 or len(config["cases"]) != 8):
            raise ValueError("Registered bounded eight-case .025-ms canary required")
        env = dict(os.environ, PYTHONPATH=str(work / "Libs"), PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        env.pop("NEURONSIM_PFC_SOURCE_CASE_ID", None)
        env.pop("SLURM_ARRAY_TASK_ID", None)
        test_script = work / "run_tests.py"
        test_script.write_text(
            "import json,unittest\nfrom pathlib import Path\n"
            "suite=unittest.defaultTestLoader.loadTestsFromName('population_dynamics.test_pfc_source_launcher')\n"
            "r=unittest.TextTestRunner(verbosity=2).run(suite)\n"
            "Path('tests.json').write_text(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),"
            "errors=len(r.errors),skipped=len(r.skipped),success=r.wasSuccessful())))\n"
            "raise SystemExit(not r.wasSuccessful())\n")
        run("tests", ["srun", "--ntasks=1", "--cpus-per-task=1", sys.executable, "-B", str(test_script)], env)
        report["tests"] = json.loads((work / "tests.json").read_text())
        case_ids = [row["case_id"] for row in config["cases"]]
        (work / "cases.json").write_text(json.dumps(case_ids))
        launcher_path = work / "Libs/scripts/run_pfc_source_cases.py"
        spec = importlib.util.spec_from_file_location("staged_pfc_source_launcher", launcher_path)
        source_launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source_launcher)
        code_files = {name: sha256_file(work / name) for name in source_launcher._CODE_PATHS}
        expected_bindings = dict(config_sha256=sha256_file(work / "config.yaml"),
                                 cases_sha256=sha256_file(work / "cases.json"),
                                 code_sha256=source_launcher.code_fingerprint(code_files))
        report["input_bindings_expected"] = dict(expected=expected_bindings, code_files=code_files)
        source_env = dict(env, NEURONSIM_PROJECT_ROOT=str(work),
                          NEURONSIM_PFC_PYTHON=sys.executable, NEURONSIM_PFC_SCRATCH="/tmp",
                          NEURONSIM_PFC_SOURCE_BACKEND="numpy",
                          NEURONSIM_PFC_SOURCE_CONFIG=str(work / "config.yaml"),
                          NEURONSIM_PFC_SOURCE_QUALIFICATION=str(args.qualified_stream.resolve()),
                          NEURONSIM_PFC_SOURCE_CASES_JSON=str(work / "cases.json"),
                          NEURONSIM_PFC_OUTPUT_BASE=str(args.output / "positive"))
        source_env.update({source_launcher._BINDING_ENV[key]: value for key, value in expected_bindings.items()})
        launcher = work / "Libs/scripts/run_pfc_source_cases.sbatch"
        positive_path = args.positive_batch
        if positive_path is None:
            run("positive", ["bash", str(launcher)], source_env)
            positive_path = args.output / "positive/batches" / ("run_" + job)
        positive = json.loads((positive_path / "delivery.json").read_text())
        if positive["exit_code"] != 0 or [row["case_id"] for row in positive["cases"]] != case_ids:
            raise ValueError("Complete packed case denominator required")
        binding = positive.get("input_bindings", {})
        if (binding.get("status") != "VERIFIED" or binding.get("expected") != expected_bindings
                or binding.get("actual") != expected_bindings or binding.get("code_files") != code_files
                or any(row.get("archive_status") != "COMPLETE" for row in positive["cases"])):
            raise ValueError("Packed success requires verified frozen input bindings and complete archives")
        comparisons = []
        intervals = []
        for row in positive["cases"]:
            case_id = row["case_id"]
            archive = Path(row["archive"])
            receipt = json.loads((archive / "delivery.json").read_text())
            bound = {item["path"]: item for item in receipt["files"]}
            for name, item in bound.items():
                path = archive / name
                if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                    raise ValueError("Packed archive changed: " + str(path))
            folder = archive / "source_run"
            current = json.loads((folder / "report.json").read_text())
            observation = json.loads((folder / "engine_observation.json").read_text())
            if current["source_execution_complete"] is not True or current["resolved_config"] != config:
                raise ValueError("Packed source configuration or completion mismatch")
            compared_arrays = 0
            for name in ("raw.npz", "terminal_state.npz"):
                reference = verified_reference(args.qualified_stream,
                    "comparison/recorded_regression/" + case_id + "/" + name, "stream")
                with np.load(reference, allow_pickle=False) as expected, np.load(folder / name, allow_pickle=False) as actual:
                    if set(expected.files) != set(actual.files):
                        raise ValueError("Packed output schema mismatch")
                    for key in expected.files:
                        if actual[key].dtype != expected[key].dtype or not np.array_equal(actual[key], expected[key]):
                            raise ValueError("Packed output mismatch: " + case_id + "/" + name + "/" + key)
                        compared_arrays += 1
            reference = verified_reference(args.native_validation,
                "capture/native/" + case_id + "_initial_voltage.npy", "native")
            if not np.array_equal(np.load(reference, allow_pickle=False),
                                  np.load(folder / "initial_voltage.npy", allow_pickle=False)):
                raise ValueError("Packed initial state mismatch")
            reference = verified_reference(args.native_validation,
                "capture/inputs/" + case_id + "/numpy_random_state.npz", "native")
            with np.load(reference, allow_pickle=False) as expected, np.load(folder / "final_rng.npz", allow_pickle=False) as actual:
                if set(expected.files) != set(actual.files) or any(
                        not np.array_equal(expected[key], actual[key]) for key in expected.files):
                    raise ValueError("Packed terminal source RNG mismatch")
            reference = verified_reference(args.qualified_stream,
                "comparison/recorded_regression/" + case_id + "/comparison.json", "stream")
            expected_inputs = json.loads(reference.read_text())["inputs_used"]
            if any(current["inputs_used"][key] != expected_inputs[key] for key in ("E", "I", "ticks_consumed")):
                raise ValueError("Packed input consumption mismatch")
            if set(observation["numerical_threads"].values()) != {"1"}:
                raise ValueError("Numerical library thread count differs from one")
            intervals.append(observation)
            comparisons.append(dict(case_id=case_id, status="PASS_EXACT", output_arrays=compared_arrays,
                                    initial_voltage="EXACT", final_rng="EXACT", input_consumption="EXACT",
                                    engine_observation=observation, source_archive=str(archive),
                                    direct_applied_current_history_reference=None))
        boundaries = [(row["simulator_start_monotonic_ns"], 1) for row in intervals]
        boundaries += [(row["simulator_stop_monotonic_ns"], -1) for row in intervals]
        concurrency = maximum = 0
        for _, change in sorted(boundaries):
            concurrency += change
            maximum = max(maximum, concurrency)
        if maximum != tasks or len({row["engine_pid"] for row in intervals}) != len(case_ids):
            raise ValueError("Declared overlapping engine count and distinct case invocations required")
        report["positive"] = dict(status="PASS_EXACT", batch=str(positive_path), cases=comparisons,
            maximum_simultaneous_engine_calls=maximum, core_neurons=1280, duration_ms=800, dt_ms=.025,
            direct_full_input_sequence_compared=False,
            input_sequence_evidence="Retained native stream qualification; this canary compares consumption and final RNG",
            direct_full_current_history_compared=False)
        # An invalid source seed fails one rank; its next case must remain NOT_RUN.
        siblings = ["sibling_" + chr(ord("a") + i) for i in range(tasks - 1)]
        fault_ids = ["invalid_seed", *siblings, "same_rank_not_run",
                     *(name + "_again" for name in siblings)]
        fault = dict(config, excitatory_neurons=8, inhibitory_neurons=2, weight_scaling_factor=256.,
                     duration_ms=.05, random_stream_policy="source_global",
                     cases=[dict(case_id=name, source_rng_seed=-1 if i == 0 else 123+i,
                                 stimulus_strength_nA=0., distractor_strength_nA=0.)
                            for i, name in enumerate(fault_ids)],
                     purpose="Disposable two-tick software failure fixture; not scientific data")
        (work / "fault_config.yaml").write_text(yaml.safe_dump(fault))
        (work / "fault_cases.json").write_text(json.dumps(fault_ids))
        fault_env = dict(source_env, NEURONSIM_PFC_SOURCE_CONFIG=str(work / "fault_config.yaml"),
                         NEURONSIM_PFC_SOURCE_CASES_JSON=str(work / "fault_cases.json"),
                         NEURONSIM_PFC_OUTPUT_BASE=str(args.output / "fault"))
        fault_bindings = dict(expected_bindings, config_sha256=sha256_file(work / "fault_config.yaml"),
                              cases_sha256=sha256_file(work / "fault_cases.json"))
        fault_env.update({source_launcher._BINDING_ENV[key]: value for key, value in fault_bindings.items()})
        fault_path = args.fault_batch
        if fault_path is None:
            run("fault", ["bash", str(launcher)], fault_env, expected=1)
            fault_path = args.output / "fault/batches" / ("run_" + job)
        failed = json.loads((fault_path / "delivery.json").read_text())
        if (failed.get("input_bindings", {}).get("status") != "VERIFIED"
                or failed["input_bindings"].get("actual") != fault_bindings):
            raise ValueError("Fault fixture must reach source execution with verified input bindings")
        statuses = {row["case_id"]: row for row in failed["cases"]}
        if (failed["exit_code"] == 0 or failed["srun_exit_code"] == 0
                or statuses["invalid_seed"]["exit_code"] != 1
                or statuses["same_rank_not_run"]["status"] != "NOT_RUN"
                or statuses["same_rank_not_run"]["exit_code"] is not None
                or any(statuses[name]["status"] != "PFC_SOURCE_EXECUTION_COMPLETE"
                       for name in fault_ids if name not in {"invalid_seed", "same_rank_not_run"})):
            raise ValueError("Deliberate failure propagation or sibling accounting mismatch")
        failure_report = json.loads((Path(statuses["invalid_seed"]["archive"]) / "source_run/report.json").read_text())
        report.update(fault=dict(status="PASS_EXPECTED_FAILURE", batch=str(fault_path),
                      srun_exit_code=failed["srun_exit_code"], failed_cases=1, not_run_cases=1,
                      completed_software_fixture_cases=2 * tasks - 2, source_failure=failure_report.get("failure")),
                      status="PASS_PACKED_SOURCE_EXECUTION_CANARY",
                      full_scale_packing_qualified=False, full_scale_speedup_established=False)
        code = 0
    except Exception as error:
        report.update(status="FAILED", error=str(error), traceback=traceback.format_exc())
    report.update(exit_code=code, completed_utc=datetime.now(timezone.utc).isoformat())
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    files = []
    for source in sorted(work.rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        relative = source.relative_to(work)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise ValueError("Packing verification archive mismatch")
        files.append(dict(path=str(relative), sha256=digest, bytes=target.stat().st_size))
    (args.output / "delivery.json").write_text(json.dumps(dict(report, files=files), indent=2) + "\n")
    if code == 0:
        shutil.rmtree(work)
    print(json.dumps({key: report.get(key) for key in ("status", "job_id", "exit_code", "error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

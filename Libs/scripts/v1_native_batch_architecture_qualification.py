#!/usr/bin/env python3
"""Qualify sm80 against the complete frozen C0/B2 native family twice."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import torch

from v1_cuda_architecture_qualification import compare_replays, digest

CASES_SHA256 = "a6fad79c884c66fe3e04d9da5a5ffb383c5419cab04ab6b846a89f41d7a5959f"
SOURCE_BATCH_SHA256 = "09b8dd29bd6c30117c2ed22fdfd0b56d0b45444e405334bf4ca178911f25db17"
RUNTIME_CODE = {
    "v1_glif_component_emulator.py": "f5dce9f6c2ef223a832350c3e360f00ebdb942fc9cfcd245dbde0fe4794c842e",
    "v1_glif_cuda.py": "15eb17f8b2100696be82a4fe9cfe6a9fb0492ba957fe91e3a7bc19bfe59e3b59",
    "v1_glif_network.py": "325dcf4ebccaf4103858c6b4b5e849081597267437d237022da2763501a81be3",
    "v1_nest_time_semantics.py": "3f24949ddd61ea920b32fe072a4b0e5e3ca3f64ead02680d5c2899d6e4c5ae69",
}
PACKAGES = {"numpy": "2.4.2", "pandas": "2.3.3", "scipy": "1.16.2",
            "h5py": "3.14.0", "PyYAML": "6.0.3", "psutil": "7.1.0"}
DATASET_STREAMS = {"connectivity_seed": 100000000, "initial_state_seed": 110000002,
                   "background_seed": 120000002, "stimulus_seed": 130000002}
NATIVE_FILES = ("delivery.json", "original_config.yaml", "primary/report.json",
                "primary/native/parameters.json", "primary/native/cortical_state.npz",
                "primary/native/cortical_spikes.npz", "primary/native/input_events.npz")
NATIVE_PASS = "PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC"
BLOCK_PASS = "PASS_COMPLETE_TEN_CONDITION_NATIVE_GLIF_CUDA_BLOCK"
RTOL = ATOL = 1e-12
UNITS = 230924
CONNECTION_ROWS = 82246780
TRANSPORT_CHUNK_BYTES = 256 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def retain(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"FRESH_RETAINED_FILE_REQUIRED: {target}")
    shutil.copy2(source, target)
    return {"source": str(source), "retained": str(target), "sha256": digest(target)}


def validate_and_retain_sources(job: dict, output: Path) -> tuple[dict, dict]:
    cases_path = Path(job["cases"])
    require(job["reference"] == str(cases_path) and digest(cases_path) == CASES_SHA256,
            "FROZEN_C0_B2_CASES_IDENTITY_DIFFERS")
    cases = read_json(cases_path)
    require(cases["schema"] == "v1_native_block_comparison_v1"
            and cases["scenario_id"] == "v1_visual" and cases["circuit_index"] == 0
            and cases["block_index"] == 2 and len(cases["cases"]) == 10,
            "COMPLETE_FROZEN_C0_B2_FAMILY_REQUIRED")
    snapshot_path = Path(job["snapshot"]) / "report.json"
    snapshot = read_json(snapshot_path)
    require(snapshot["status"] == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
            and snapshot["expected_source_batch_sha256"] == SOURCE_BATCH_SHA256
            and snapshot["snapshot"]["source_batch_sha256"] == SOURCE_BATCH_SHA256
            and snapshot["snapshot"]["rows"] == CONNECTION_ROWS
            and snapshot["units"] == UNITS, "FROZEN_SNAPSHOT_IDENTITY_DIFFERS")
    identities = {"cases": retain(cases_path, output / "frozen_cases.json"),
                  "snapshot_report": retain(snapshot_path, output / "snapshot_report.json"),
                  "runtime_code": {}, "native_family": []}
    for name, expected in RUNTIME_CODE.items():
        source = Path(job["code"]) / name
        require(digest(source) == expected, "FROZEN_NUMERICAL_MODULE_DIFFERS:" + name)
        identities["runtime_code"][name] = retain(source, output / "software" / name)
    native_archives = set()
    for index, case in enumerate(cases["cases"]):
        identity = case["identity"]
        require(identity["scenario_id"] == "v1_visual"
                and identity["circuit_index"] == 0 and identity["block_index"] == 2
                and identity["condition_index"] == index and identity["split"] == "train"
                and identity["sample_unit"] == "complete_mechanistic_episode"
                and identity["episode_id"] == f"v1_visual_circuit_000_block_002_condition_{index:02d}"
                and all(identity[key] == value for key, value in DATASET_STREAMS.items()),
                "FROZEN_CONDITION_IDENTITY_DIFFERS")
        archive = Path(case["native_archive"])
        require(archive not in native_archives, "DUPLICATE_NATIVE_REFERENCE")
        native_archives.add(archive)
        native_path = archive / "primary/report.json"
        native = read_json(native_path)
        delivery = read_json(archive / "delivery.json")
        require(digest(native_path) == case["native_report_sha256"]
                and native["exit_code"] == 0 and native["cortical_units"] == UNITS
                and native["native_nest_simulation"] is True
                and native["native_parameters_and_initial_state_exact"] is True
                and native["training_seeds_used"] == []
                and native["connections"]["source_batch_sha256"] == SOURCE_BATCH_SHA256
                and native["connections"]["native_connections_to_cortex"] == CONNECTION_ROWS
                and native["execution_clock"] == snapshot["execution_clock"]
                and native["execution_clock"]["duration_ms"] == 3000.0
                and native["execution_clock"]["resolution_ms"] == 0.25
                and delivery["exit_code"] == 0 and delivery["mode"] == "full"
                and str(delivery["job_id"]) == str(case["native_job_id"])
                and delivery["spike_inputs_reference"] == case["spike_inputs_reference"],
                "NATIVE_REFERENCE_IDENTITY_DIFFERS")
        declarations = {row["path"]: row for row in delivery["files"]}
        retained = {}
        for relative in NATIVE_FILES:
            source = archive / relative
            if source.suffix == ".npz":
                # The verified input archive already preserves these raw arrays.
                # Replay comparison consumes only the generated output arrays.
                record = {"source": str(source), "bytes": source.stat().st_size,
                          "sha256": digest(source), "copied_to_result": False}
            else:
                record = retain(source, output / "source_references" / f"condition_{index:02d}" / relative)
            if relative != "delivery.json":
                require(source.stat().st_size == declarations[relative]["bytes"]
                        and record["sha256"] == declarations[relative]["sha256"],
                        "NATIVE_DECLARED_FILE_DIFFERS:" + str(source))
            retained[relative] = record
        require(retained["original_config.yaml"]["sha256"] == native["original_config_sha256"],
                "NATIVE_ORIGINAL_CONTROL_DIFFERS")
        identities["native_family"].append({"identity": identity, "files": retained,
                                            "native_job_id": case["native_job_id"]})
    return cases, identities


def inspect_execution(destination: Path, cases: dict, library_sha: str) -> dict:
    checks, conditions = {}, []
    try:
        aggregate = read_json(destination / "report.json")
        rows = aggregate.get("cases", [])
        observed_pass = sum(row.get("status") == NATIVE_PASS for row in rows)
        checks.update(
            aggregate_status=aggregate.get("status") == BLOCK_PASS,
            aggregate_cases_identity=aggregate.get("cases_sha256") == CASES_SHA256,
            complete_ordered_conditions=[row.get("condition_index") for row in rows] == list(range(10)),
            aggregate_counts_consistent=aggregate.get("planned_conditions") == 10
                and aggregate.get("passed_conditions") == observed_pass
                and aggregate.get("failed_conditions") == 10 - observed_pass and len(rows) == 10,
            all_conditions_passed=aggregate.get("passed_conditions") == 10
                and aggregate.get("failed_conditions") == 0,
            coupled_scale=aggregate.get("coupled_units") == UNITS
                and aggregate.get("source_connection_rows") == CONNECTION_ROWS,
            frozen_runtime=aggregate.get("runtime_code") == RUNTIME_CODE,
            matrix_reuse=aggregate.get("circuit_matrix_loads") == 1
                and aggregate.get("circuit_cache_hits") == 9,
            actual_candidate=aggregate.get("gpu_device_preflight", {}).get("library_sha256") == library_sha
                and aggregate.get("gpu_device_preflight", {}).get("compute_capability") == [8, 0],
        )
    except Exception as error:
        aggregate, rows = {}, []
        checks["aggregate_report_readable"] = False
        aggregate["read_error"] = str(error)
    for index, case in enumerate(cases["cases"]):
        condition = {"condition_index": index, "checks": {}}
        expected_path = destination / f"condition_{index:02d}" / "report.json"
        try:
            result = read_json(expected_path)
            row = rows[index] if index < len(rows) else {}
            binding = result.get("source_bindings", {})
            linked = Path(row.get("report", ""))
            if not linked.is_absolute():
                linked = destination / linked
            condition["checks"] = {
                "report_path": linked.resolve() == expected_path.resolve(),
                "aggregate_identity": row.get("identity") == case["identity"]
                    and row.get("native_job_id") == case["native_job_id"],
                "native_pass": result.get("status") == row.get("status") == NATIVE_PASS,
                "full_units": result.get("units") == UNITS,
                "full_duration": result.get("duration_ms") == 3000.0,
                "frozen_tolerances": result.get("component_rtol") == RTOL
                    and result.get("component_atol") == ATOL,
                "frozen_runtime": result.get("runtime_code") == RUNTIME_CODE,
                "native_binding": str(binding.get("native_job_id")) == str(case["native_job_id"])
                    and binding.get("spike_inputs_reference") == case["spike_inputs_reference"]
                    and binding.get("files", {}).get("primary/report.json", {}).get("sha256")
                    == case["native_report_sha256"],
                "native_comparison_metrics": all(result.get(key) is True and row.get(key) is True
                    for key in ("all_spike_pairs_exact", "all_unit_counts_exact",
                                "terminal_voltage_within_component_tolerance")),
                "arrays_present": all((expected_path.parent / name).is_file()
                    for name in ("cortical_state.npz", "cortical_spikes.npz")),
            }
            condition["report"] = str(expected_path)
        except Exception as error:
            condition["checks"]["condition_report_readable"] = False
            condition["error"] = str(error)
        condition["passed"] = bool(condition["checks"]) and all(condition["checks"].values())
        conditions.append(condition)
    return {"passed": all(checks.values()) and all(row["passed"] for row in conditions),
            "checks": checks, "aggregate": aggregate, "conditions": conditions}


def persist(output: Path, report: dict, target: str, run: str) -> None:
    root = Path(__file__).resolve().parent
    bundle = root / ("native_batch_sm80_result_" + run + ".tar.gz")
    require(not bundle.exists(), "FRESH_RESULT_BUNDLE_REQUIRED")
    subprocess.run([sys.executable, str(root / "pack_results.py"), "--output",
                    str(bundle), str(output)], check=True)
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
    subprocess.run(["hdfs", "dfs", "-put", str(bundle), target + "/result.tar.gz"], check=True)
    chunks = []
    with bundle.open("rb") as stream:
        for index, block in enumerate(iter(lambda: stream.read(TRANSPORT_CHUNK_BYTES), b"")):
            name = f"chunk_{index:05d}.json"
            path = root / (run + "_" + name)
            with path.open("x") as chunk:
                chunk.write(json.dumps({"content_base64": base64.b64encode(block).decode()}) + "\n")
            subprocess.run(["hdfs", "dfs", "-put", str(path), target + "/" + name], check=True)
            chunks.append({"name": name, "bytes": len(block),
                           "sha256": hashlib.sha256(block).hexdigest()})
            path.unlink()
    receipt = dict(report, persisted=True, checkpoint_path=target, chunks=chunks,
                   result_sha256=digest(bundle), result_bytes=bundle.stat().st_size,
                   transport_chunk_bytes=TRANSPORT_CHUNK_BYTES)
    path = root / ("native_batch_sm80_receipt_" + run + ".json")
    with path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    subprocess.run(["hdfs", "dfs", "-put", str(path), target + "/receipt.json"], check=True)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True)
    args = parser.parse_args()
    job = read_json(args.job_file)
    run, trial = os.environ["RH2_JOB_RUN_ID"], os.environ["ARNOLD_TRIAL_ID"]
    require(run and trial and "/" not in run and run not in (".", ".."),
            "REAL_MERLIN_IDENTITY_REQUIRED")
    require(job["scope"] == "cuda_architecture_qualification" and job["mode"] == "batch_check",
            "FROZEN_NATIVE_BATCH_QUALIFICATION_SCOPE_REQUIRED")
    output = Path(job["output"]) / ("run_" + run)
    output.mkdir(parents=True, exist_ok=False)
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/v1_sm80/frozen_native_family"
    report = {
        "status": "RUNNING_FROZEN_NATIVE_BATCH_ARCHITECTURE_QUALIFICATION",
        "run_sid": run, "trial_sid": trial, "hostname": socket.gethostname(),
        "execution_backend": "merlin", "output": str(output), "scope": job["scope"],
        "original_reference_job": job["job"], "original_command": job["command"],
        "new_benchmark_episodes": 0, "production_qualified": False,
        "scientific_qualification": False, "native_reference_rerun": False,
        "training_seeds_used": [], "dataset_streams": DATASET_STREAMS,
        "planned_executions": 2, "planned_conditions_per_execution": 10,
        "component_rtol": RTOL, "component_atol": ATOL, "executions": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    started, code = time.monotonic(), 1
    try:
        retain(Path(__file__), output / Path(__file__).name)
        helper = Path(sys.modules[compare_replays.__module__].__file__)
        report["replay_helper"] = retain(helper, output / helper.name)
        retain(args.job_file, output / "job.json")
        require(int(os.environ.get("ARNOLD_APPLIED_GPU_NUM", "0")) == 1
                and torch.cuda.is_available() and torch.cuda.device_count() == 1
                and list(torch.cuda.get_device_capability(0)) == [8, 0],
                "ONE_ACTUAL_SM80_GPU_REQUIRED")
        properties = torch.cuda.get_device_properties(0)
        report.update(gpu_name=properties.name, compute_capability=[8, 0],
                      gpu_total_memory_bytes=properties.total_memory,
                      python_version=sys.version, torch_cuda_version=torch.version.cuda)
        report["package_versions"] = {name: importlib.metadata.version(name)
                                      for name in (*PACKAGES, "torch")}
        require(all(report["package_versions"][name] == version for name, version in PACKAGES.items()),
                "QUALIFIED_PACKAGE_VERSION_DIFFERS")
        library = Path(job["library"])
        require(digest(library) == job["expected_library_sha256"], "CANDIDATE_LIBRARY_DIFFERS")
        build = read_json(Path(job["build_report"]))
        require(build["publication_status"] == "PASS" and build["exit_code"] == 0
                and build["library_sha256"] == job["expected_library_sha256"]
                and build["source_sha256"] == job["expected_cuda_source_sha256"]
                and "V12.8.61" in build["nvcc_version"], "PINNED_COMPILER_RECEIPT_DIFFERS")
        report.update(library_sha256=digest(library), source_sha256=build["source_sha256"],
                      compiler_run_sid=build["run_sid"])
        retain(library, output / library.name)
        retain(Path(job["build_report"]), output / "compiler_report.json")
        preflight = Path(__file__).parent / "merlin_resource_preflight.json"
        if preflight.is_file():
            retain(preflight, output / preflight.name)
        detector = retain(Path(job["detector"]), output / "detect_resources.py")
        with (output / "resources.log").open("x") as log:
            subprocess.run([sys.executable, "-B", detector["retained"],
                            "-o", str(output / "resources.json")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        cases, identities = validate_and_retain_sources(job, output)
        report["source_identities"] = identities
        print(json.dumps(report), flush=True)
        for label in ("primary", "independent_replay"):
            compatibility = "merlin_" + trial + "_v1_sm80_frozen_native_" + label
            destination = output / label
            env = dict(os.environ, SLURM_JOB_ID=compatibility, SLURM_CPUS_PER_TASK="1",
                       NRNSIM_EXECUTION_BACKEND="merlin",
                       PYTHONPATH=str(Path(job["code"]).parent), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-B", "-m", "population_dynamics.v1_glif_network",
                       "--cases", job["cases"], "--snapshot", job["snapshot"],
                       "--cuda-library", str(library), "--output", str(destination)]
            execution = {"label": label, "command": command, "output": str(destination),
                         "legacy_compatibility_id": compatibility,
                         "compatibility_id_semantics": "interface compatibility only; not a Slurm execution"}
            try:
                require(not destination.exists(), "FRESH_SUBPROCESS_OUTPUT_REQUIRED")
                with (output / (label + ".log")).open("x") as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    execution.update(process_id=child.pid, exit_code=child.wait())
            except Exception as error:
                execution.update(exit_code=1, error=str(error), traceback=traceback.format_exc())
            execution["inspection"] = inspect_execution(destination, cases, report["library_sha256"])
            execution["passed"] = execution["exit_code"] == 0 and execution["inspection"]["passed"]
            report["executions"].append(execution)
            write_json(output / "progress.json", report)
        comparisons = []
        for index in range(10):
            try:
                result = compare_replays(output / "primary" / f"condition_{index:02d}",
                                         output / "independent_replay" / f"condition_{index:02d}",
                                         RTOL, ATOL)
            except Exception as error:
                result = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
            comparisons.append({"condition_index": index, **result})
        replay = {"planned_conditions": 10,
                  "passed_conditions": sum(row["passed"] for row in comparisons),
                  "failed_conditions": sum(not row["passed"] for row in comparisons),
                  "conditions": comparisons}
        write_json(output / "independent_replay_comparison.json", replay)
        report["independent_replay"] = replay
        require(all(item["passed"] for item in report["executions"])
                and replay["passed_conditions"] == 10 and replay["failed_conditions"] == 0,
                "FROZEN_NATIVE_BATCH_OR_INDEPENDENT_REPLAY_FAILED")
        report.update(status="PASS_FROZEN_C0_B2_NATIVE_BATCH_AND_INDEPENDENT_REPLAY",
                      units=UNITS, duration_ms=3000.0,
                      passed_executions=2, passed_native_conditions=20)
        code = 0
    except BaseException as error:
        report.update(status="FAILED_FROZEN_NATIVE_BATCH_ARCHITECTURE_QUALIFICATION",
                      error=str(error), traceback=traceback.format_exc())
    finally:
        report.update(exit_code=code, elapsed_seconds=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "report.json", report)
        print(json.dumps(report), flush=True)
        try:
            persist(output, report, target, run)
        except BaseException as error:
            report.update(persistence_error=str(error), persistence_traceback=traceback.format_exc(),
                          persisted=False, exit_code=1)
            write_json(output / "report.json", report)
            print(json.dumps(report), flush=True)
            code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()


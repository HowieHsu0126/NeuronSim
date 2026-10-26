"""Compare existing PFC campaign outputs with independent pinned Brian execution.

The full source clock, coupled units and condition remain unchanged. Candidate
science is never repeated. Completed native outputs are persisted before array
comparison. Correspondence is a component check, not source/scaling admission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time

import brian2 as b2
import numpy as np
import yaml

from . import pfc_reference as reference
from .pfc_input_observation import NativeInputObserver
from .provenance import freeze_json, sha256_file

SOURCE_RUN = "6de397acc866aadd"
SOURCE_TRIAL = "435906314"
RAW_KEYS = ("native_spike_counts", "spike_seconds", "spike_ticks", "spike_unit_indices", "terminal_voltage_mV")
SOURCE_SHA256 = "416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720"
VOLTAGE_ATOL_MV = 1e-10


def require(value, reason):
    if not value:
        raise ValueError(reason)


def hdfs(*arguments):
    return subprocess.run(["hdfs", "dfs", *map(str, arguments)], check=True,
                          text=True, capture_output=True)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def validate_config(config, identity):
    require(config.get("scope") == "source_reproduction"
            and config.get("excitatory_neurons") == 80000 and config.get("inhibitory_neurons") == 20000
            and config.get("weight_scaling_factor") == .0256 and config.get("dt_ms") == .0125
            and config.get("duration_ms") == 5000. and config.get("readout_bin_ms") == 5.
            and config.get("synapse_execution") == "factorized_uniform_projections"
            and config.get("random_stream_policy") == "independent_initial_external"
            and config.get("expected_source_revision") == "d7c4f07fa83a052829e40790fd1594955fe1c067",
            "PFC_NATIVE_COMPLETE_SOURCE_CONFIG_REQUIRED")
    require(identity.get("split") in ("train", "calibration")
            and config.get("benchmark_episode_candidate", {}).get("identity") == identity
            and config["benchmark_episode_candidate"].get("production_eligible") is False,
            "PFC_NATIVE_TRAINING_CALIBRATION_IDENTITY_REQUIRED")
    require(len(config.get("cases", [])) == 1
            and config["cases"][0]["case_id"] == identity["episode_id"]
            and config["initial_rng_seed"] == identity["initial_state_seed"]
            and config["external_noise_seed"] == identity["background_seed"],
            "PFC_NATIVE_CASE_STREAM_IDENTITY_CHANGED")


def candidate_files(task, work):
    """Extract only the original source files from a receipt-bound bundle."""
    checkpoint = task["checkpoint_path"]
    require(checkpoint.endswith(f"/{SOURCE_RUN}/pfc/task_{task['index']:04d}"),
            "PFC_NATIVE_SOURCE_CHECKPOINT_CHANGED")
    receipt_path = work / "candidate_receipt.json"
    hdfs("-get", checkpoint + "/receipt.json", receipt_path)
    receipt = read_json(receipt_path)
    require(all(receipt.get(k) == v for k, v in dict(index=task["index"], run_id=SOURCE_RUN,
            trial_id=SOURCE_TRIAL, status="COMPLETE", exit_code=0, persisted=True).items()),
            "PFC_NATIVE_CANDIDATE_NOT_PERSISTED_SUCCESS")
    manifest_path, archive_path = work / "candidate_manifest.json", work / "candidate.tar.gz"
    hdfs("-get", checkpoint + "/manifest.json", manifest_path)
    manifest = read_json(manifest_path)
    require(sha256_file(manifest_path) == task["manifest_sha256"], "PFC_NATIVE_CANDIDATE_MANIFEST_CHANGED")
    hdfs("-get", checkpoint + "/result.tar.gz", archive_path)
    require(archive_path.stat().st_size == receipt["result_bytes"]
            and sha256_file(archive_path) == receipt["result_sha256"], "PFC_NATIVE_CANDIDATE_BUNDLE_CHANGED")
    names = ("report.json", "config.yaml", "raw.npz", "initial_voltage.npy", "final_rng.npz",
             "applied_current_observation.npz", "terminal_state.npz")
    prefix = task["source_member_prefix"].rstrip("/") + "/"
    selected = {prefix + name: name for name in names}
    require(all(name in manifest for name in selected), "PFC_NATIVE_CANDIDATE_FILES_MISSING")
    destination = work / "candidate"
    destination.mkdir()
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in selected:
                continue
            require(member.name not in seen and member.isfile() and member.size == manifest[member.name]["bytes"],
                    "PFC_NATIVE_CANDIDATE_MEMBER_INVALID")
            target = destination / selected[member.name]
            stream = archive.extractfile(member)
            require(stream is not None, "PFC_NATIVE_CANDIDATE_MEMBER_UNREADABLE")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            require(sha256_file(target) == manifest[member.name]["sha256"], "PFC_NATIVE_CANDIDATE_MEMBER_CHANGED")
            seen.add(member.name)
    require(seen == set(selected), "PFC_NATIVE_CANDIDATE_EXTRACTION_INCOMPLETE")
    config = yaml.safe_load((destination / "config.yaml").read_text())
    validate_config(config, task["identity"])
    source = read_json(destination / "report.json")
    require(source["source_entry_sha256"] == SOURCE_SHA256 and source["resolved_config"] == config
            and source["case_id"] == task["identity"]["episode_id"]
            and source["source_execution_complete"] is True
            and source["dimensions"]["coupled_core_units"] == 100000,
            "PFC_NATIVE_CANDIDATE_REPORT_CHANGED")
    archive_path.unlink()
    return destination, config, receipt


class CompleteSourceObservation:
    """Reuse full-unit monitoring and passive source-current observation."""

    def __init__(self, duration_ticks):
        self.duration_ticks = duration_ticks
        self.snapshots, self.flags, self.ticks = [], [], []
        self.monitor_calls = 0

    def install(self, module):
        original_spike, original_operation = module.SpikeMonitor, module.network_operation

        def full_spikes(population, record=None, **kwargs):
            self.monitor_calls += 1
            return original_spike(population, record=True, **kwargs)

        def operation(*args, **kwargs):
            def decorate(function):
                selected = function
                if function.__name__ == "stimulate_network":
                    closure = dict(zip(function.__code__.co_freevars,
                        (cell.cell_contents for cell in function.__closure__)))

                    @functools.wraps(function)
                    def observed(t):
                        function(t)
                        current = np.asarray(closure["excit_pop"].I_stim / b2.amp)
                        flags = (bool(closure["t_stimulus_start"] <= t < closure["t_stimulus_end"]),
                                 bool(closure["t_distractor_start"] <= t < closure["t_distractor_end"]))
                        tick = int(round(float(t / b2.ms) / .0125))
                        if not self.snapshots or flags != self.flags[-1] or not np.array_equal(current, self.snapshots[-1]):
                            self.snapshots.append(current.copy())
                            self.flags.append(flags)
                            self.ticks.append(tick)
                    selected = observed
                return original_operation(*args, **kwargs)(selected)
            return decorate

        module.SpikeMonitor, module.network_operation = full_spikes, operation

    def save(self, output):
        require(self.monitor_calls == 2 and self.ticks and self.ticks[0] == 0,
                "PFC_NATIVE_FULL_UNIT_OBSERVATION_INCOMPLETE")
        ticks = np.asarray(self.ticks, dtype=np.int64)
        np.savez(output / "applied_current_observation.npz", segment_start_ticks=ticks,
            segment_stop_ticks=np.r_[ticks[1:], self.duration_ticks],
            current_A=np.asarray(self.snapshots, dtype=np.float64),
            cue_active=np.asarray([x[0] for x in self.flags], dtype=np.bool_),
            distractor_active=np.asarray([x[1] for x in self.flags], dtype=np.bool_))


def execute_native(source, config, output):
    """Call the pinned Brian source with the qualified uniform projection helper."""
    observer = NativeInputObserver(output / "inputs", mode="audit")
    observation = CompleteSourceObservation(400000)
    original = reference.load_source

    def observed_source(path):
        module, manifest, entry = original(path)
        observation.install(module)
        return module, manifest, entry

    reference.load_source = observed_source
    try:
        report = reference.execute(source, config, output / "native", native_input_observer=observer)
        observer.finalize()
        observation.save(output)
        return report
    finally:
        reference.load_source = original
        observer.close()


def array_result(name, actual, expected, atol=0.):
    same_shape = actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = (float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
             if same_shape and actual.size else 0. if same_shape else None)
    return dict(array=name, shape=list(actual.shape), values_compared=int(actual.size),
        exact=atol == 0., atol=atol, rtol=0., max_absolute_difference=delta,
        passed=bool(same_shape and finite and (np.array_equal(actual, expected) if atol == 0.
                    else np.allclose(actual, expected, rtol=0., atol=atol))))


def compare(candidate, native, identity):
    report = read_json(native / "native/report.json")
    require(report["status"] == "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
            and report["synapse_execution"] == "factorized_uniform_projections"
            and len(report["cases"]) == 1 and report["cases"][0]["case_id"] == identity["episode_id"]
            and report["cases"][0]["core_units"] == 100000, "PFC_NATIVE_COMPLETE_COMPARISON_REQUIRED")
    case = report["cases"][0]
    rows = []
    with np.load(native / "native" / case["raw_file"], allow_pickle=False) as actual, np.load(candidate / "raw.npz", allow_pickle=False) as expected:
        require(set(actual.files) == set(expected.files) == set(RAW_KEYS), "PFC_NATIVE_RAW_SCHEMA_CHANGED")
        for name in RAW_KEYS:
            rows.append(array_result(name, actual[name], expected[name], VOLTAGE_ATOL_MV if name == "terminal_voltage_mV" else 0.))
    rows.append(array_result("initial_voltage_mV", np.load(native / "native" / case["initial_voltage_file"], allow_pickle=False),
        np.load(candidate / "initial_voltage.npy", allow_pickle=False)))
    for family, first, second in (("external_rng", native / "inputs" / identity["episode_id"] / "numpy_random_state.npz", candidate / "final_rng.npz"),
                                 ("applied_current", native / "applied_current_observation.npz", candidate / "applied_current_observation.npz")):
        with np.load(first, allow_pickle=False) as actual, np.load(second, allow_pickle=False) as expected:
            require(set(actual.files) <= set(expected.files), "PFC_NATIVE_INPUT_SCHEMA_CHANGED")
            for name in actual.files:
                a, e = actual[name], expected[name]
                rows.append(dict(array=family + ":" + name, passed=bool(a.shape == e.shape and a.dtype == e.dtype and np.array_equal(a, e)),
                                 values_compared=int(a.size), exact=True, atol=0., rtol=0.))
    return dict(status="PASS_NATIVE_CAMPAIGN_CORRESPONDENCE" if all(r["passed"] for r in rows) else "FAILED_NATIVE_CAMPAIGN_CORRESPONDENCE",
        identity=identity, arrays=rows, array_comparisons=len(rows), passed_arrays=sum(r["passed"] for r in rows),
        native_source_report_sha256=sha256_file(native / "native/report.json"),
        candidate_source_report_sha256=sha256_file(candidate / "report.json"),
        full_scale_source_case_complete=True, candidate_source_simulations=0, new_native_source_simulations=1,
        source_qualification_granted=False, scientific_fidelity_qualified=False, scaling_qualified=False,
        working_memory_qualified=False, production_episodes=0,
        scope="same-clock full100k/5s independent Brian correspondence; one exact task and parameter tuple")


def persist_native(native, checkpoint, work):
    """Persist complete native science before its downstream comparison."""
    archive = work / "native_result.tar.gz"
    files = []
    with tarfile.open(archive, "w:gz", compresslevel=1) as target:
        for path in sorted(native.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(native))
                files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path)))
                target.add(path, arcname=relative, recursive=False)
    hdfs("-put", archive, checkpoint + "/native_result.tar.gz")
    receipt = dict(status="PERSISTED_COMPLETE_NATIVE_EXECUTION", exit_code=0,
        run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
        files=files, bundle_bytes=archive.stat().st_size, bundle_sha256=sha256_file(archive),
        bundle_uri=checkpoint + "/native_result.tar.gz", source_qualification_granted=False, production_episodes=0)
    freeze_json(work / "native_receipt.json", receipt)
    hdfs("-put", work / "native_receipt.json", checkpoint + "/native_receipt.json")
    return receipt


def run(request_path, source, work, checkpoint_subdirectory=None):
    require(b2.__version__ == "2.5.4" and np.__version__ == "1.26.4",
            "PFC_NATIVE_PINNED_RUNTIME_REQUIRED")
    import scipy
    require(scipy.__version__ == "1.12.0", "PFC_NATIVE_PINNED_SCIPY_REQUIRED")
    require(bool(os.environ.get("RH2_JOB_RUN_ID")) and bool(os.environ.get("ARNOLD_TRIAL_ID"))
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not work.exists(),
            "PFC_NATIVE_FRESH_MERLIN_CPU_REQUIRED")
    work.mkdir()
    require(shutil.disk_usage(work).free >= 12 * 1024**3, "PFC_NATIVE_LOCAL_STORAGE_REQUIRED")
    request = read_json(request_path)
    require(request["schema"] == "pfc_merlin_native_correspondence_request_v1" and len(request["tasks"]) == 1,
            "PFC_NATIVE_SINGLE_REGISTERED_CANARY_REQUIRED")
    task = request["tasks"][0]
    checkpoint_root = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
    if checkpoint_subdirectory is not None:
        require(checkpoint_subdirectory == f"native_tasks/task_{task['index']:04d}",
                "PFC_NATIVE_TASK_CHECKPOINT_INVALID")
        checkpoint_root += "/" + checkpoint_subdirectory
    checkpoint = checkpoint_root + "/pfc_native_correspondence"
    hdfs("-mkdir", "-p", checkpoint)
    started = time.perf_counter()
    state = dict(status="PREPARING", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                 task_index=task["index"], identity=task["identity"], source_simulations_rerun=0,
                 native_source_simulations=0, production_episodes=0, hostname=socket.gethostname())

    def publish_state():
        path = work / "progress.json"
        path.write_text(json.dumps(state, indent=2) + "\n")
        hdfs("-put", "-f", path, checkpoint + "/progress.json")

    try:
        candidate, config, receipt = candidate_files(task, work)
        state.update(status="RUNNING_NATIVE", native_source_simulations=1,
            candidate_bundle_sha256=receipt["result_sha256"], started_utc=datetime.now(timezone.utc).isoformat())
        publish_state()
        native = work / "observed"
        native.mkdir()
        execute_native(source, candidate / "config.yaml", native)
        state.update(status="NATIVE_COMPLETE_PERSISTING", native_execution_complete=True)
        publish_state()
        persisted = persist_native(native, checkpoint, work)
        state.update(native_persisted=True, native_receipt=persisted)
        result = compare(candidate, native, task["identity"])
        result.update(execution=dict(run_id=state["run_id"], trial_id=state["trial_id"], hostname=state["hostname"]),
                      native_receipt_sha256=sha256_file(work / "native_receipt.json"),
                      native_bundle_uri=persisted["bundle_uri"], elapsed_seconds=time.perf_counter() - started)
        freeze_json(work / "comparison.json", result)
        hdfs("-put", work / "comparison.json", checkpoint + "/comparison.json")
        state.update(status=result["status"], comparison_passed=result["status"] == "PASS_NATIVE_CAMPAIGN_CORRESPONDENCE",
                     completed_utc=datetime.now(timezone.utc).isoformat())
        publish_state()
        return result
    except BaseException as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                     elapsed_seconds=time.perf_counter() - started)
        publish_state()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint-subdirectory", help="Task prefix within the actual Merlin checkpoint root")
    args = parser.parse_args()
    result = run(args.request, args.source, args.work, args.checkpoint_subdirectory)
    print(json.dumps(result))
    raise SystemExit(0 if result["status"] == "PASS_NATIVE_CAMPAIGN_CORRESPONDENCE" else 1)


if __name__ == "__main__":
    main()

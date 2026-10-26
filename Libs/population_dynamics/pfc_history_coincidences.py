"""Measure matched1ms unit covariance from complete independent PFC histories.

Source refractory checks establish binary counts. Sparse intersections preserve
time and unit identities. This is a descriptive measurement, not uncertainty
coverage or source/scaling admission.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time

import numpy as np

BIN_TICKS = 80
BIN_SECONDS = .001
INTERSECTION_CHUNK = 262144


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def binary_event_keys(ticks: np.ndarray, ids: np.ndarray, units: int,
                      duration_ticks: int) -> tuple[np.ndarray, int]:
    """Reject repeated unit/bin events; retain terminal events separately."""
    ticks, ids = np.asarray(ticks), np.asarray(ids)
    require(type(units) is int and units > 0 and type(duration_ticks) is int
        and duration_ticks > 0 and duration_ticks % BIN_TICKS == 0,
        "PFC_COINCIDENCE_EXACT_CLOCK_AND_UNITS_REQUIRED")
    require(ticks.ndim == ids.ndim == 1 and ticks.shape == ids.shape
        and ticks.dtype.kind in "iu" and ids.dtype.kind in "iu"
        and np.all((ticks >= 0) & (ticks <= duration_ticks))
        and np.all((ids >= 0) & (ids < units)), "PFC_COINCIDENCE_EVENT_IDENTITY_INVALID")
    keep = ticks < duration_ticks
    terminal = int(np.count_nonzero(~keep))
    keys = np.sort((ticks[keep].astype(np.int64) // BIN_TICKS) * units + ids[keep])
    require(not np.any(keys[1:] == keys[:-1]), "PFC_COINCIDENCE_NONBINARY1MS_COUNTS")
    return keys, terminal


def coincidence_counts(left: np.ndarray, right: np.ndarray, units: int, n_e: int,
                       phase_bins: np.ndarray) -> np.ndarray:
    """Count each same-unit/same-bin intersection in its half-open phase."""
    if len(left) > len(right):
        left, right = right, left
    result = np.zeros((len(phase_bins) - 1, 2), dtype=np.int64)
    for start in range(0, len(left), INTERSECTION_CHUNK):
        chunk = left[start:start + INTERSECTION_CHUNK]
        positions = np.searchsorted(right, chunk)
        valid = positions < len(right)
        selected = chunk[valid]
        selected = selected[right[positions[valid]] == selected]
        phase = np.searchsorted(phase_bins, selected // units, side="right") - 1
        population = (selected % units >= n_e).astype(np.int64)
        result += np.bincount(phase * 2 + population, minlength=result.size).reshape(result.shape)
    return result


def matched_history_variance(keys: list[np.ndarray], units: int, n_e: int,
                             phase_edges: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Separate the full unit diagonal and signed offdiagonal across histories.

    For n histories, A is their summed binary counts and C sums all unordered
    history-pair intersections. Sum_i sampleVar_h(X_i), averaged over T bins,
    is A/(n*T) - 2*C/(n*(n-1)*T). Divide by N² for the population diagonal.
    Time bins and units remain dependent observations within each history.
    """
    require(len(keys) >= 2 and type(units) is int and type(n_e) is int and 0 < n_e < units,
        "PFC_COINCIDENCE_COMPLETE_HISTORY_POPULATIONS_REQUIRED")
    edges = np.asarray(phase_edges)
    require(edges.ndim == 1 and edges.dtype.kind in "iu" and len(edges) > 1
        and edges[0] == 0 and np.all(np.diff(edges) > 0) and np.all(edges % BIN_TICKS == 0),
        "PFC_COINCIDENCE_PHASE_GRID_INVALID")
    phase_bins = edges // BIN_TICKS
    bins = int(phase_bins[-1])
    fields = []
    for values in keys:
        require(values.ndim == 1 and values.dtype.kind in "iu"
            and np.all((values >= 0) & (values < bins * units))
            and np.all(values[1:] > values[:-1]), "PFC_COINCIDENCE_SORTED_UNIQUE_KEYS_REQUIRED")
        population = (values % units >= n_e).astype(np.int64)
        fields.append(np.bincount((values // units) * 2 + population,
            minlength=bins * 2).reshape(bins, 2))
    fields = np.stack(fields)
    pair_rows, all_coincidences = [], np.zeros((len(edges) - 1, 2), dtype=np.int64)
    for left, right in combinations(range(len(keys)), 2):
        counts = coincidence_counts(keys[left], keys[right], units, n_e, phase_bins)
        all_coincidences += counts
        for phase in range(len(edges) - 1):
            pair_rows.append(dict(left_history_index=left, right_history_index=right, phase_index=phase,
                excitatory_unit_bin_coincidences=int(counts[phase, 0]),
                inhibitory_unit_bin_coincidences=int(counts[phase, 1])))
    rows = []
    n = len(keys)
    for phase, (first, stop) in enumerate(zip(phase_bins[:-1], phase_bins[1:])):
        t = int(stop - first)
        for name, columns, population_units in (("excitatory", [0], n_e),
            ("inhibitory", [1], units - n_e), ("all_units", [0, 1], units)):
            y = fields[:, int(first):int(stop), columns].sum(axis=2)
            a = int(y.sum())
            c = int(all_coincidences[phase, columns].sum())
            numerator = (n - 1) * a - 2 * c
            require(numerator >= 0, "PFC_COINCIDENCE_NEGATIVE_UNIT_VARIANCE")
            # Both variances share this exact integer denominator. Subtract
            # numerators before rate conversion to preserve structural zeros.
            population_numerator = n * sum(int(value)**2 for value in y.ravel())
            population_numerator -= sum(int(value)**2 for value in y.sum(axis=0))
            require(population_numerator >= 0, "PFC_COINCIDENCE_NEGATIVE_POPULATION_VARIANCE")
            denominator = n * (n - 1) * t * population_units**2 * BIN_SECONDS**2
            rows.append(dict(phase_index=phase, start_tick=int(edges[phase]), stop_tick=int(edges[phase + 1]),
                population=name, units=population_units, independent_histories=n, time_bins=t,
                bin_ticks=BIN_TICKS, bin_ms=1., complete_history_events=a,
                unordered_history_pair_unit_bin_coincidences=c,
                population_mean_variance_rate_Hz2=population_numerator / denominator,
                diagonal_contribution_rate_Hz2=numerator / denominator,
                ordered_offdiagonal_contribution_rate_Hz2=(population_numerator - numerator) / denominator,
                sample_axis="matched_time_across_independent_histories_within_circuit_and_condition"))
    return rows, pair_rows


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hdfs(*arguments: str) -> None:
    subprocess.run(["hdfs", "dfs", *map(str, arguments)], check=True, capture_output=True)


def receive_merlin_source(task: dict, directory: Path) -> dict:
    """Restore source-report/raw members from the original Merlin bundle."""
    index = task["index"]
    checkpoint = task["checkpoint_path"]
    source_run = task.get("source_run", "6de397acc866aadd")
    source_trial = task.get("source_trial", "435906314")
    require(task.get("receipt_uri", checkpoint + "/receipt.json").startswith("hdfs://")
        and task.get("bundle_uri", checkpoint + "/result.tar.gz").startswith("hdfs://")
        and checkpoint.endswith(f"/pfc/task_{index:04d}"),
        "PFC_COINCIDENCE_SOURCE_PATH_CHANGED")
    for name in ("receipt.json", "manifest.json"):
        uri = task.get("receipt_uri", checkpoint + "/receipt.json") if name == "receipt.json" else checkpoint + "/manifest.json"
        hdfs("-get", uri, directory / name)
        require(sha256(directory / name) == task[name[:-5] + "_sha256"],
            "PFC_COINCIDENCE_SOURCE_METADATA_CHANGED")
    receipt = json.loads((directory / "receipt.json").read_bytes())
    require(all(receipt.get(k) == v for k, v in dict(index=index, run_id=source_run,
        trial_id=source_trial, persisted=True, exit_code=0, status="COMPLETE").items()),
        "PFC_COINCIDENCE_COMPLETED_ORIGINAL_SOURCE_REQUIRED")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    bundle = directory / "result.tar.gz"
    hdfs("-get", task.get("bundle_uri", checkpoint + "/result.tar.gz"), bundle)
    require(bundle.stat().st_size == receipt["result_bytes"] and sha256(bundle) == receipt["result_sha256"],
        "PFC_COINCIDENCE_BUNDLE_CHANGED")
    prefix = task["source_member_prefix"].rstrip("/") + "/"
    selected = {prefix + name: name for name in ("raw.npz", "report.json")}
    require(set(selected) <= set(manifest), "PFC_COINCIDENCE_SOURCE_FILES_MISSING")
    seen = set()
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive:
            if member.name not in selected:
                continue
            row = manifest[member.name]
            require(member.name not in seen and member.isfile() and member.size == row["bytes"],
                "PFC_COINCIDENCE_BOUND_MEMBER_INVALID")
            target = directory / selected[member.name]
            stream = archive.extractfile(member)
            require(stream is not None, "PFC_COINCIDENCE_UNREADABLE_MEMBER")
            with target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            require(sha256(target) == row["sha256"], "PFC_COINCIDENCE_BOUND_MEMBER_CHANGED")
            seen.add(member.name)
    require(seen == set(selected), "PFC_COINCIDENCE_SOURCE_FILES_INCOMPLETE")
    bundle.unlink()
    return dict(source_execution_backend="merlin", checkpoint_path=checkpoint,
        source_run=source_run, source_trial=source_trial,
        source_bundle_sha256=receipt["result_sha256"], manifest_sha256=task["manifest_sha256"],
        raw_sha256=manifest[prefix + "raw.npz"]["sha256"])


def receive_slurm_source(task: dict, directory: Path) -> dict:
    """Restore verified Slurm events with separate transfer provenance."""
    require(0 <= task["index"] < 35 and task.get("transport") == "original_slurm_events"
        and isinstance(task.get("config_sha256"), str) and len(task["config_sha256"]) == 64,
        "PFC_COINCIDENCE_ORIGINAL_SLURM_TASK_REQUIRED")
    receipt_path = directory / "receipt.json"
    hdfs("-get", task["receipt_uri"], receipt_path)
    require(sha256(receipt_path) == task["receipt_sha256"], "PFC_COINCIDENCE_TRANSFER_RECEIPT_CHANGED")
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt.get("status") == "VERIFIED_ORIGINAL_XIE_NATIVE_EVENTS_TRANSFERRED"
        and receipt.get("source_execution_backend") == "slurm"
        and receipt.get("source_job_id") == task["source_job_id"]
        and str(receipt["source_job_id"]).isdecimal()
        and receipt.get("index") == task["index"] and receipt.get("identity") == task["identity"]
        and receipt.get("source_simulations") == receipt.get("production_episodes") == 0,
        "PFC_COINCIDENCE_ORIGINAL_EXECUTION_IDENTITY_CHANGED")
    for local, uri_field, sha_field in (("raw.npz", "raw_uri", "raw_sha256"),
        ("report.json", "source_report_uri", "source_report_sha256"),
        ("delivery.json", "source_delivery_uri", "source_delivery_sha256")):
        require(str(receipt[uri_field]).startswith("hdfs://"), "PFC_COINCIDENCE_TRANSFER_URI_INVALID")
        hdfs("-get", receipt[uri_field], directory / local)
        require(sha256(directory / local) == receipt[sha_field], "PFC_COINCIDENCE_ORIGINAL_TRANSFER_FILE_CHANGED")
    require((directory / "raw.npz").stat().st_size == receipt["raw_bytes"],
        "PFC_COINCIDENCE_ORIGINAL_RAW_SIZE_CHANGED")
    delivery = json.loads((directory / "delivery.json").read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    require(delivery.get("status") == "ARCHIVED_PFC_SOURCE_EXECUTION"
        and delivery.get("job_id") == receipt["source_job_id"] and delivery.get("exit_code") == 0
        and delivery.get("source_execution_complete") is True
        and delivery.get("case_id") == task["identity"]["episode_id"]
        and report.get("config_sha256") == task["config_sha256"]
        and report["outputs"]["raw.npz"]["sha256"] == receipt["raw_sha256"]
        and report["outputs"]["raw.npz"]["bytes"] == receipt["raw_bytes"],
        "PFC_COINCIDENCE_ORIGINAL_SOURCE_BINDING_CHANGED")
    return dict(source_execution_backend="slurm", source_run=receipt["source_job_id"], source_trial=None,
        checkpoint_path=task["receipt_uri"].rsplit("/", 1)[0],
        source_bundle_sha256=None, manifest_sha256=None, raw_sha256=receipt["raw_sha256"],
        source_delivery_sha256=receipt["source_delivery_sha256"],
        transfer_run=receipt["transfer_run_id"], transfer_trial=receipt["transfer_trial_id"],
        transfer_receipt_sha256=task["receipt_sha256"])


def run(request_path: Path, work: Path, checkpoint_subdirectory: str | None = None) -> dict:
    """Read original bundles once; publish compact covariance without simulation."""
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
        and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not work.exists(),
        "PFC_COINCIDENCE_FRESH_MERLIN_CPU_REQUIRED")
    request = json.loads(request_path.read_bytes())
    circuit = request.get("circuit_index", 23)
    condition = request.get("condition_index", 0)
    require(type(condition) is int and 0 <= condition < 10
        and type(circuit) is int and 0 <= circuit < 100
        and request.get("schema") == "pfc_matched1ms_history_coincidences_v1"
        and request.get("task_indices") == [30*circuit + condition + 10*h for h in range(3)]
        and len(request.get("tasks", [])) == 3,
        "PFC_COINCIDENCE_DECLARED_COMPLETE3_HISTORY_SAMPLE_REQUIRED")
    work.mkdir()
    keys, bindings, initial, external = [], [], set(), set()
    started = time.perf_counter()
    for index, task in zip(request["task_indices"], request["tasks"]):
        identity = task["identity"]
        require(task["index"] == index and identity["circuit_index"] == circuit
            and identity["block_index"] == 6 + len(keys) and identity["condition_index"] == condition
            and identity["split"] == ("train" if circuit < 70 else "validation" if circuit < 80
                else "calibration" if circuit < 90 else "test"),
            "PFC_COINCIDENCE_HISTORY_IDENTITY_CHANGED")
        directory = work / f"task_{index:04d}"
        directory.mkdir()
        transport = task.get("transport", "merlin_bundle")
        require(transport in ("merlin_bundle", "original_slurm_events"), "PFC_COINCIDENCE_SOURCE_TRANSPORT_INVALID")
        source_binding = (receive_slurm_source(task, directory) if transport == "original_slurm_events"
            else receive_merlin_source(task, directory))
        report = json.loads((directory / "report.json").read_bytes())
        config = report["resolved_config"]
        require(report["status"] == "PFC_SOURCE_EXECUTION_COMPLETE"
            and report["source_execution_complete"] is True
            and config["benchmark_episode_candidate"]["identity"] == identity
            and config["excitatory_neurons"] == 80000 and config["inhibitory_neurons"] == 20000
            and config["dt_ms"] == .0125 and config["duration_ms"] == 5000.
            and config["random_stream_policy"] == "independent_initial_external"
            and report["production_episodes"] == 0, "PFC_COINCIDENCE_FROZEN_SOURCE_CHANGED")
        for population, ticks in (("E", 160), ("I", 80)):
            check = report["validations"]["refractory_spacing_" + population]
            require(check["passed"] is True and check["violation_count"] == 0
                and check["minimum_required_ticks"] == ticks, "PFC_COINCIDENCE_REFRACTORY_CHANGED")
        initial.add(config["initial_rng_seed"])
        external.add(config["external_noise_seed"])
        with np.load(directory / "raw.npz", allow_pickle=False) as raw:
            current, terminal = binary_event_keys(raw["spike_ticks"], raw["spike_unit_indices"], 100000, 400000)
            require(len(current) + terminal == int(raw["native_spike_counts"].sum()),
                "PFC_COINCIDENCE_FULL_EVENT_COUNT_CHANGED")
        keys.append(current)
        bindings.append(dict(task_index=index, identity=identity, **source_binding,
            source_report_sha256=sha256(directory / "report.json"),
            retained_binary_unit_bins=len(current), terminal_events_excluded=terminal))
        (directory / "raw.npz").unlink()
    require(len(initial) == len(external) == 3, "PFC_COINCIDENCE_INDEPENDENT_HISTORY_STREAMS_REQUIRED")
    rows, pairs = matched_history_variance(keys, 100000, 80000,
        np.array([0, 8000, 24080, 32000, 36000, 192000, 196000, 320000, 400000], dtype=np.int64))
    for name, data in (("matched_history_variance1ms.csv", rows),
                       ("history_pair_unit_bin_coincidences.csv", pairs)):
        with (work / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = dict(schema=request["schema"], status="COMPLETE_REAL3_HISTORY_MATCHED1MS_UNIT_COVARIANCE",
        execution=dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"], hostname=socket.gethostname()),
        bindings=bindings, circuit_index=circuit, condition_index=condition,
        complete_source_episodes_reused=3, independent_histories=3, history_pairs=3,
        history_pairs_are_independent_replicates=False, core_units=100000, observation_bin_ticks=80,
        duration_ticks=400000, phase_count=8, source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scaling_qualified=False, uncertainty_coverage_qualified=False,
        request_sha256=sha256(request_path), implementation_sha256=sha256(Path(__file__)),
        elapsed_seconds=time.perf_counter() - started, recorded_utc=datetime.now(timezone.utc).isoformat(),
        outputs={name: dict(bytes=(work / name).stat().st_size, sha256=sha256(work / name)) for name in
            ("matched_history_variance1ms.csv", "history_pair_unit_bin_coincidences.csv")})
    (work / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
    if checkpoint_subdirectory is not None:
        require(checkpoint_subdirectory in (f"condition_{condition:02d}",
            f"circuit_{circuit:03d}/condition_{condition:02d}"),
            "PFC_COINCIDENCE_CONDITION_CHECKPOINT_INVALID")
        checkpoint += "/" + checkpoint_subdirectory
    checkpoint += "/pfc_history_coincidences"
    hdfs("-mkdir", "-p", checkpoint)
    for name in (*result["outputs"], "report.json"):
        hdfs("-put", work / name, checkpoint + "/" + name)
    return result


def ready_groups(accepted: list[int], completed: list[int], missing_events: set[int]) -> list[int]:
    """A group key is its history6task index; incomplete histories stay pending."""
    require(len(accepted) == len(set(accepted)) and all(type(i) is int and 0 <= i < 3000 for i in accepted),
        "PFC_COINCIDENCE_ACCEPTED_INDICES_INVALID")
    require(len(completed) == len(set(completed)) and all(type(i) is int and 0 <= i < 3000
        and i % 30 < 10 for i in completed), "PFC_COINCIDENCE_COMPLETE_GROUP_KEYS_INVALID")
    available = set(accepted) - missing_events
    return [30*c + k for c in range(100) for k in range(10) if 30*c+k not in completed
        and {30*c+k, 30*c+k+10, 30*c+k+20} <= available]


def run_cohort(request_path: Path, work: Path) -> dict:
    """Consume ready groups serially; never dispatch source simulation or intake."""
    require(os.environ.get("RH2_JOB_RUN_ID") and os.environ.get("ARNOLD_TRIAL_ID")
        and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not work.exists(),
        "PFC_COINCIDENCE_FRESH_MERLIN_CPU_REQUIRED")
    request = json.loads(request_path.read_bytes())
    require(request["schema"] == "pfc_complete_domain_history_covariance_v1"
        and request["missing_native_event_indices"] == list(range(35))
        and request["required_groups"] == 1000 and request["reuse_circuit"] == 23,
        "PFC_COINCIDENCE_DECLARED_FULL_DOMAIN_REQUIRED")
    work.mkdir()
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/pfc_domain_history_covariance"
    hdfs("-mkdir", "-p", checkpoint)
    completed, failures = {}, []
    tables = {"matched_history_variance1ms.csv": [], "history_pair_unit_bin_coincidences.csv": []}
    reuse = work / "reuse"
    reuse.mkdir()
    for name, entry in request["reuse_outputs"].items():
        hdfs("-get", request["reuse_checkpoint"] + "/" + name, reuse / name)
        require((reuse / name).stat().st_size == entry["bytes"] and sha256(reuse / name) == entry["sha256"],
            "PFC_COINCIDENCE_REUSED_COMPLETE_CIRCUIT_CHANGED")
    prior = json.loads((reuse / "report.json").read_bytes())
    require(prior["status"] == "COMPLETE10_CONDITION3_HISTORY_MATCHED1MS_UNIT_COVARIANCE"
        and prior["complete_conditions"] == 10 and prior["source_simulations"] == prior["production_episodes"] == 0,
        "PFC_COINCIDENCE_COMPLETE_REUSE_REQUIRED")
    for row in prior["completed"]:
        completed[690 + row["condition_index"]] = dict(row, circuit_index=23, reused=True)
    for name in tables:
        with (reuse / name).open() as stream:
            tables[name].extend(csv.DictReader(stream))

    def publish(status: str, accepted_count: int) -> dict:
        result = dict(schema=request["schema"], status=status,
            execution=dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"],
                trial_id=os.environ["ARNOLD_TRIAL_ID"], hostname=socket.gethostname()),
            completed_groups=len(completed), required_groups=1000, completed=list(completed.values()),
            completed_group_keys=sorted(completed), accepted_source_episodes=accepted_count,
            failures=failures, missing_native_event_indices=list(range(35)),
            missing_native_event_groups=[30*c+k for c in range(2) for k in range(10)
                if {30*c+k,30*c+k+10,30*c+k+20} & set(range(35))],
            source_simulations=0, production_episodes=0, source_qualification_granted=False,
            scaling_qualified=False, uncertainty_coverage_qualified=False,
            implementation_sha256=sha256(Path(__file__)), request_sha256=sha256(request_path),
            recorded_utc=datetime.now(timezone.utc).isoformat())
        path = work / "progress.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        hdfs("-put", "-f", path, checkpoint + "/progress.json")
        return result

    while True:
        response = subprocess.run(["hdfs", "dfs", "-cat", request["accepted_progress_uri"]],
            check=True, capture_output=True)
        acceptance = json.loads(response.stdout)
        require(acceptance["execution"]["run_id"] == "81caa23939cca1e3"
            and acceptance["execution"]["trial_id"] == "436946603"
            and acceptance["production_episodes"] == acceptance["source_simulations"] == 0
            and acceptance["failures"] == [], "PFC_COINCIDENCE_CANONICAL_ACCEPTANCE_CHANGED")
        accepted = acceptance["accepted_task_indices"]
        require(len(accepted) == acceptance["accepted_episode_count"], "PFC_COINCIDENCE_ACCEPTED_COUNT_CHANGED")
        selected = ready_groups(accepted, list(completed), set(range(35)))
        for group in selected:
            circuit, condition = divmod(group, 30)
            group_work = work / f"circuit_{circuit:03d}_condition_{condition:02d}"
            metadata = work / "metadata"
            metadata.mkdir(exist_ok=True)
            tasks = []
            for index in (group, group + 10, group + 20):
                origin = request["source_origins"][str(index)]
                task = dict(origin, index=index, identity=request["identities"][str(index)])
                for name in ("receipt", "manifest"):
                    path = metadata / f"task{index}_{name}.json"
                    uri = origin["receipt_uri"] if name == "receipt" else origin["checkpoint_path"] + "/manifest.json"
                    hdfs("-get", uri, path)
                    task[name + "_sha256"] = sha256(path)
                manifest = json.loads((metadata / f"task{index}_manifest.json").read_bytes())
                prefixes = [name[:-len("/raw.npz")] for name in manifest if name.endswith("/source_run/raw.npz")]
                require(len(prefixes) == 1, "PFC_COINCIDENCE_ONE_ORIGINAL_RAW_REQUIRED")
                task["source_member_prefix"] = prefixes[0]
                tasks.append(task)
            group_request = work / "group_request.json"
            group_request.write_text(json.dumps(dict(schema="pfc_matched1ms_history_coincidences_v1",
                circuit_index=circuit, condition_index=condition,
                task_indices=[group, group+10, group+20], tasks=tasks), indent=2) + "\n")
            try:
                report = run(group_request, group_work, f"circuit_{circuit:03d}/condition_{condition:02d}")
            except Exception as error:
                failures.append(dict(group=group, circuit_index=circuit, condition_index=condition,
                    error_type=type(error).__name__, error=str(error)))
                result = publish("FAILED_COMPLETE_GROUP", len(accepted))
                raise RuntimeError("PFC_COINCIDENCE_COMPLETE_GROUP_FAILED:" + str(group)) from error
            for name in tables:
                with (group_work / name).open() as stream:
                    tables[name].extend(dict(circuit_index=circuit, condition_index=condition, **row)
                        for row in csv.DictReader(stream))
            completed[group] = dict(circuit_index=circuit, condition_index=condition,
                report_uri=os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
                    + f"/circuit_{circuit:03d}/condition_{condition:02d}/pfc_history_coincidences/report.json",
                report_sha256=sha256(group_work / "report.json"), reused=False)
            publish("RUNNING_COMPLETE_GROUPS", len(accepted))
            shutil.rmtree(group_work)
            for path in metadata.iterdir():
                path.unlink()
        if len(accepted) == 3000:
            result = publish("PARTIAL985_REQUIRES_ORIGINAL35_NATIVE_EVENT_RETURN", len(accepted))
            require(len(completed) == 985, "PFC_COINCIDENCE_REMOTE_GROUP_DENOMINATOR_MISMATCH")
            for name, rows in tables.items():
                with (work / name).open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                hdfs("-put", work / name, checkpoint + "/" + name)
            result["outputs"] = {name: dict(bytes=(work / name).stat().st_size,
                sha256=sha256(work / name), rows=len(rows)) for name, rows in tables.items()}
            (work / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            hdfs("-put", work / "report.json", checkpoint + "/report.json")
            return result
        publish("WAITING_COMPLETE_SOURCE_HISTORY_GROUPS", len(accepted))
        time.sleep(request["poll_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint-subdirectory")
    parser.add_argument("--cohort", action="store_true")
    args = parser.parse_args()
    report = run_cohort(args.request, args.work) if args.cohort else run(args.request, args.work, args.checkpoint_subdirectory)
    print(json.dumps({k: v for k, v in report.items() if k not in ("bindings", "outputs")}))


if __name__ == "__main__":
    main()

"""Execute an authenticated native coupled GLIF snapshot on CUDA.

Saved native tuples retain zero weights, receptor types and actual rounded
delays. Source execution and comparison use the same discrete clock.
These development paths do not grant production qualification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import h5py
import numpy as np
from .provenance import freeze_json, sha256_file
from .v1_pointnet_native import bound_files, canonical_connections, input_trains_by_population, load_spike_inputs
from .v1_nest_time_semantics import encode_precise_times
from .v1_glif_component_reference import require
from .v1_glif_component_emulator import GlifCellPlan, prepare_cells, COMPONENT_RTOL, COMPONENT_ATOL
from .v1_glif_cuda import device_library, emulate_network_cuda
from .v1_connection_snapshot import verify_snapshot

NATIVE_FILES = ("report.json", "native/parameters.json", "native/cortical_state.npz",
                "native/cortical_spikes.npz", "native/input_events.npz",
                "native/connection_readback.npz", "native/connection_source_batches.npz")



def source_runtime_code() -> dict[str, str]:
    """Identify the code that maps source inputs and advances the GLIF state."""
    return {
        "v1_glif_network.py": sha256_file(Path(__file__)),
        "v1_glif_component_emulator.py": sha256_file(Path(prepare_cells.__code__.co_filename)),
        "v1_glif_cuda.py": sha256_file(Path(emulate_network_cuda.__code__.co_filename)),
        "v1_nest_time_semantics.py": sha256_file(Path(encode_precise_times.__code__.co_filename)),
    }


def positions(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and values.dtype.kind in "iu"
            and len(np.unique(ids)) == len(ids), "GLIF_NETWORK_IDENTITY_DOMAIN")
    order = np.argsort(ids)
    index = np.searchsorted(ids[order], values)
    require(np.all(index < len(ids)), "GLIF_NETWORK_UNKNOWN_IDENTITY")
    require(np.array_equal(ids[order[index]], values), "GLIF_NETWORK_UNKNOWN_IDENTITY")
    return order[index]


def connection_csr(raw: dict, cells: np.ndarray, inputs: np.ndarray, dt: float, receptors: int) -> dict:
    inventory = np.concatenate([cells, inputs])
    source = positions(inventory, raw["source"])
    target = positions(cells, raw["target"])
    weight = np.asarray(raw["weight"])
    delay = np.asarray(raw["delay"]) / dt
    receptor = np.asarray(raw["receptor_type"])
    count = len(source)
    require(all(np.asarray(raw[key]).shape == (count,) for key in raw)
            and weight.dtype == np.float64 and np.isfinite(weight).all()
            and np.isfinite(delay).all() and np.all(delay >= 1) and np.array_equal(delay, np.rint(delay))
            and receptor.dtype.kind in "iu" and np.all((receptor >= 1) & (receptor <= receptors)),
            "GLIF_NETWORK_CONNECTION_DOMAIN")
    order = np.argsort(source, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(source, minlength=len(inventory)))])
    return dict(offsets=offsets.astype(np.int64), target=target[order].astype(np.int64),
                receptor=(receptor[order]-1).astype(np.int64),
                delay=delay[order].astype(np.int64), weight=weight[order])


def external_events(raw: dict, cells: int, steps: int, dt: float) -> dict:
    counts = raw["source_event_counts"]
    ticks = raw["encoded_steps"]
    require(counts.ndim == 1 and counts.dtype.kind in "iu" and np.all(counts >= 0)
            and ticks.dtype.kind in "iu", "GLIF_NETWORK_EXTERNAL_DOMAIN")
    if ticks.ndim == 2:
        require(ticks.shape[0] == len(counts) and np.all(counts == ticks.shape[1]),
                "GLIF_NETWORK_PULSE_SHAPE")
        source = np.repeat(np.arange(len(counts)), counts)
    else:
        require(ticks.ndim == 1 and "source_event_unit" in raw,
                "GLIF_NETWORK_PACKED_EVENTS_REQUIRED")
        source = np.asarray(raw["source_event_unit"])
        require(source.dtype.kind in "iu" and np.all((source >= 0) & (source < len(counts)))
                and np.array_equal(np.bincount(source, minlength=len(counts)), counts),
                "GLIF_NETWORK_PACKED_EVENT_COUNTS_DIFFER")
    times, offsets = raw["native_times_ms"].reshape(-1), raw["encoded_offsets_ms"].reshape(-1)
    ticks = ticks.reshape(-1)
    require(len(source) == len(ticks) == counts.sum() and np.all((ticks >= 1) & (ticks <= steps))
            and np.array_equal(ticks*dt-offsets, times) and np.asarray(raw["precise_times"]).all(),
            "GLIF_NETWORK_NATIVE_EVENT_CLOCK_DIFFERS")
    order = np.argsort(ticks, kind="stable")
    row_offsets = np.concatenate([[0], np.cumsum(np.bincount(ticks-1, minlength=steps))])
    return dict(offsets=row_offsets.astype(np.int64), source=(source[order]+cells).astype(np.int64))


def snapshot_connections(directory: Path, original: dict, native_report_sha: str) -> tuple[dict, dict]:
    """Read source-call tuples only after matching the completed native digest."""
    report = json.loads((directory/"report.json").read_text())
    expected = original["connections"]["source_batch_sha256"]
    require(report["status"] == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
            and report["native_report_sha256"] == native_report_sha
            and report["expected_source_batch_sha256"] == expected
            and report["parameters_and_initial_state_exact"] and report["input_events_exact"]
            and report["native_biological_time_ms"] == 0, "GLIF_NETWORK_SNAPSHOT_REFERENCE_DIFFERS")
    path = directory/report["connections_file"]
    verified = verify_snapshot(path, expected, original["connections"]["native_connections_to_cortex"])
    with h5py.File(path,"r") as file:
        values = {key:file[key][:] for key in ("source","target","receptor_type","weight","delay")}
    return values, dict(directory=str(directory), report_sha256=sha256_file(directory/"report.json"),
                        verification=verified, provenance="source_call_digest_snapshot",
                        full_native_tuple_readback=original["connections"]["full_native_tuple_readback"])



CELL_IDENTITY_FIELDS = ("source_id", "native_id", "node_type_id",
                        "inverse_source_id", "population", "initial_vm_mV")
INPUT_IDENTITY_FIELDS = ("population", "source_id", "native_id",
                         "inverse_source_id", "precise_times")


@dataclass
class CircuitIdentity:
    matrix_digest: str
    connection_rows: int
    parameter_sha256: str
    clock: dict
    cells: dict
    inputs: dict

    @classmethod
    def capture(cls, report: dict, data: dict, binding: dict):
        def retained(values, names):
            result = {name: values[name].copy() for name in names}
            for array in result.values():
                array.setflags(write=False)
            return result
        return cls(report["connections"]["source_batch_sha256"],
            report["connections"]["native_connections_to_cortex"],
            binding["files"]["primary/native/parameters.json"]["sha256"],
            dict(report["execution_clock"]),
            retained(data["cortical_state"], CELL_IDENTITY_FIELDS),
            retained(data["input_events"], INPUT_IDENTITY_FIELDS))

    def validate(self, report: dict, data: dict, binding: dict) -> None:
        require(report["connections"]["source_batch_sha256"] == self.matrix_digest
                and report["connections"]["native_connections_to_cortex"] == self.connection_rows,
                "GLIF_CACHE_SOURCE_MATRIX_DIFFERS")
        require(report["execution_clock"] == self.clock, "GLIF_CACHE_EXECUTION_CLOCK_DIFFERS")
        require(binding["files"]["primary/native/parameters.json"]["sha256"] == self.parameter_sha256,
                "GLIF_CACHE_PARAMETER_BINDING_DIFFERS")
        for reference, current, label in (
            (self.cells, data["cortical_state"], "CELL"),
            (self.inputs, data["input_events"], "INPUT")):
            for name, expected in reference.items():
                value = current[name]
                require(value.dtype == expected.dtype and np.array_equal(value, expected),
                        "GLIF_CACHE_"+label+"_IDENTITY_DIFFERS:"+name)


@dataclass
class NativeCircuitCache:
    """Process-owned immutable circuit data; every call gets fresh evolving state."""
    snapshot: Path
    source_archive: Path
    cells: GlifCellPlan
    edges: dict
    identity: CircuitIdentity
    evidence: dict
    first_reference: tuple | None
    source_binding: dict
    source_report: dict
    includes_outcomes: bool
    cache_hits: int = 0
    source_inputs_prepared: int = 0

    @classmethod
    def load(cls, archive: Path, snapshot: Path, *, include_outcomes: bool = True):
        prepared = read_native(archive, snapshot, include_outcomes=include_outcomes)
        cells, edges, _, data, binding, report = prepared
        for array in edges.values():
            array.setflags(write=False)
        for value in vars(cells).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return cls(snapshot.resolve(), archive.resolve(), cells, edges,
                   CircuitIdentity.capture(report, data, binding),
                   deepcopy(binding["connection_snapshot"]), prepared if include_outcomes else None,
                   deepcopy(binding), deepcopy(report), include_outcomes)

    def read(self, archive: Path):
        require(self.includes_outcomes, "GLIF_CACHE_REFERENCE_OUTCOMES_DISABLED")
        if archive.resolve() == self.source_archive and self.first_reference is not None:
            prepared, self.first_reference = self.first_reference, None
            prepared[4]["circuit_cache"] = {"matrix_reused": False, "matrix_loads": 1}
            return prepared
        data, binding, report = read_reference_data(archive, require_tuples=False)
        self.identity.validate(report, data, binding)
        external = external_events(data["input_events"], self.cells.count, self.cells.steps, self.cells.dt)
        require(len(external["source"]) == report["input_readback"]["events"],
                "GLIF_NETWORK_EXTERNAL_DENOMINATOR_DIFFERS")
        self.cache_hits += 1
        binding.update(connection_snapshot=deepcopy(self.evidence),
            connection_provenance="verified_in_process_circuit_reuse",
            circuit_cache={"matrix_reused": True, "matrix_loads": 1,
                           "static_identity_and_parameters_exact": True,
                           "source_archive": str(self.source_archive)})
        return self.cells, self.edges, external, data, binding, report


    def source_inputs(self, inputs: Path):
        require(not self.includes_outcomes, "GLIF_SOURCE_CACHE_MUST_EXCLUDE_OUTCOMES")
        expected = self.source_report["bindings"]["spike_inputs"]["input_construction_binding"]
        inventories = {name: {"sorted_ids": self.identity.inputs["source_id"][
            self.identity.inputs["population"] == name]} for name in ("lgn", "bkg")}
        plan, binding, _ = load_spike_inputs(inputs, expected, inventories)
        raw = source_input_events(plan, self.identity)
        events = external_events(raw, self.cells.count, self.cells.steps, self.cells.dt)
        self.source_inputs_prepared += 1
        return raw, events, binding


def read_reference_data(archive: Path, *, require_tuples: bool, include_outcomes: bool = True):
    required = NATIVE_FILES if require_tuples else NATIVE_FILES[:-2]
    if not include_outcomes:
        required = tuple(name for name in required if name != "native/cortical_spikes.npz")
    delivery, binding = bound_files(archive, tuple("primary/"+name for name in required))
    report = json.loads((archive/"primary/report.json").read_text())
    require(delivery["exit_code"] == report["exit_code"] == 0
            and report["native_nest_simulation"] and report["native_parameters_and_initial_state_exact"]
            and (not require_tuples or report["connections"]["full_native_tuple_readback"]),
            "GLIF_NETWORK_COMPLETE_NATIVE_TUPLES_REQUIRED")
    binding.update(native_job_id=str(delivery["job_id"]),
                   spike_inputs_reference=delivery.get("spike_inputs_reference"))
    names = ["cortical_state", "input_events"]
    if include_outcomes:
        names.append("cortical_spikes")
    if require_tuples:
        names.extend(["connection_readback", "connection_source_batches"])
    data = {}
    for name in names:
        with np.load(archive/("primary/native/"+name+".npz"), allow_pickle=False) as file:
            fields = CELL_IDENTITY_FIELDS if name == "cortical_state" and not include_outcomes else file.files
            data[name] = {key: file[key] for key in fields}
    return data, binding, report


def validate_expected_reference(binding: dict, expected: dict) -> None:
    require(binding["native_job_id"] == str(expected["native_job_id"])
            and binding["spike_inputs_reference"] == expected["spike_inputs_reference"]
            and binding["files"]["primary/report.json"]["sha256"] == expected["native_report_sha256"],
            "GLIF_BATCH_NATIVE_REFERENCE_BINDING_DIFFERS")


def read_native(archive: Path, snapshot: Path | None = None, *, include_outcomes: bool = True):
    data, binding, report = read_reference_data(archive, require_tuples=snapshot is None,
                                               include_outcomes=include_outcomes)
    state = data["cortical_state"]
    if snapshot is None:
        raw = data["connection_readback"]
        supplied = canonical_connections(data["connection_source_batches"])
        observed = canonical_connections(raw)
        require(all(np.array_equal(supplied[key], observed[key]) for key in observed),
                "GLIF_NETWORK_SOURCE_AND_NATIVE_TUPLES_DIFFER")
        binding["connection_provenance"] = "native_tuple_readback"
    else:
        raw, evidence = snapshot_connections(snapshot, report, binding["files"]["primary/report.json"]["sha256"])
        binding["connection_snapshot"] = evidence
        binding["connection_provenance"] = evidence["provenance"]
    params = json.loads((archive/"primary/native/parameters.json").read_text())
    ids, types = state["native_id"], state["node_type_id"]
    require(len(ids) == report["cortical_units"] and len(raw["source"]) ==
            report["connections"]["native_connections_to_cortex"], "GLIF_NETWORK_DENOMINATOR_DIFFERS")
    parameters = [params[str(kind)]["parameters"] for kind in types]
    for kind in np.unique(types):
        row = params[str(kind)]
        require(row["model"] == "glif_psc" and row["all_units_exact"]
                and row["units"] == int(np.count_nonzero(types == kind)),
                "GLIF_NETWORK_PARAMETER_COVERAGE_DIFFERS")
    clock = report["execution_clock"]
    cells = prepare_cells(parameters, ids, dt=clock["resolution_ms"], duration=clock["duration_ms"],
                          tics_per_ms=clock["tics_per_ms"])
    require(np.array_equal(cells.initial_voltage+cells.e_l, state["initial_vm_mV"]),
            "GLIF_NETWORK_INITIAL_VOLTAGE_DIFFERS")
    edges = connection_csr(raw, ids, data["input_events"]["native_id"], cells.dt, cells.receptors)
    external = external_events(data["input_events"], cells.count, cells.steps, cells.dt)
    require(len(external["source"]) == report["input_readback"]["events"],
            "GLIF_NETWORK_EXTERNAL_DENOMINATOR_DIFFERS")
    return cells, edges, external, data, binding, report



def source_input_events(plan: dict, identity: CircuitIdentity) -> dict:
    """Encode the source producer's events using the pinned native clock."""
    inventory = identity.inputs
    populations = inventory["population"]
    data = {name: {"sorted_ids": inventory["source_id"][populations == name]}
            for name in ("lgn", "bkg")}
    require(np.array_equal(populations, np.concatenate([
        np.full(len(data[name]["sorted_ids"]), name) for name in ("lgn", "bkg")])),
        "GLIF_SOURCE_INPUT_POPULATION_ORDER_DIFFERS")
    trains_by_population = input_trains_by_population(plan, data)
    trains = [train for name in ("lgn", "bkg") for train in trains_by_population[name]]
    counts = np.asarray([len(train) for train in trains], dtype=np.int64)
    requested = np.concatenate(trains)
    encoded = encode_precise_times(requested, identity.clock["tics_per_ms"],
                                   identity.clock["resolution_ms"])
    return dict(inventory, source_event_counts=counts,
        source_event_unit=np.repeat(np.arange(len(counts), dtype=np.int64), counts),
        requested_times_ms=requested, native_times_ms=encoded["decoded_times_ms"],
        encoded_steps=encoded["steps"], encoded_offsets_ms=encoded["offsets_ms"])




# Keep the established NPZ format while reducing lossless recording CPU cost.
SOURCE_NPZ_COMPRESSION_LEVEL = 1


def save_source_arrays(path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=SOURCE_NPZ_COMPRESSION_LEVEL, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name+".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)


def validate_source_case(inputs: Path, binding: dict, expected: dict) -> None:
    identity, config = expected["identity"], binding["config"]
    require(inputs.resolve() == Path(expected["input_archive"]).resolve()
            and binding["bundle_report_sha256"] == expected["input_report_sha256"]
            and config == expected["input_config"],
            "GLIF_SOURCE_CASE_BINDING_DIFFERS")
    require(config["background"]["starting_noise_seed"] == identity["background_seed"]
            and config["spike_generation"]["starting_noise_seed"] == identity["stimulus_seed"],
            "GLIF_SOURCE_CASE_NOISE_IDENTITY_DIFFERS")


def execute_source(archive: Path, inputs: Path, library: Path, output: Path, snapshot: Path, *,
                   cache: NativeCircuitCache | None = None, expected_source: dict | None = None) -> dict:
    """Run source inputs from the circuit reset without reading cortical outcomes."""
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_SOURCE_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_SOURCE_INPUT_GLIF_CUDA_EXECUTION",
        job_id=os.environ["SLURM_JOB_ID"], scope="development_source_backend_execution",
        actual_split="development", production_episodes=0, benchmark_episodes=0,
        scientific_qualification=False, cortical_outcome_arrays_loaded=False, cortical_outcomes_used_for_execution=False,
        source_reset_policy="deterministic_source_reset_then_supplied_stochastic_warmup",
        code_sha256=sha256_file(Path(__file__)), runtime_code=source_runtime_code())
    started = time.monotonic()
    try:
        _, report["gpu_device_preflight"] = device_library(library)
        if cache is None:
            cache = NativeCircuitCache.load(archive, snapshot, include_outcomes=False)
        require(cache.source_archive == archive.resolve() and cache.snapshot == snapshot.resolve(),
                "GLIF_SOURCE_CACHE_CIRCUIT_DIFFERS")
        cells, edges, identity = cache.cells, cache.edges, cache.identity
        reused = cache.source_inputs_prepared > 0
        raw, external, input_binding = cache.source_inputs(inputs)
        if expected_source is not None:
            validate_source_case(inputs, input_binding, expected_source)
        binding = deepcopy(cache.source_binding)
        binding["circuit_cache"] = dict(matrix_loads=1, matrix_reused=reused,
            source_inputs_prepared=cache.source_inputs_prepared)
        report.update(circuit_reference=str(archive), circuit_bindings=binding,
            input_archive=str(inputs), input_bindings=input_binding,
            units=cells.count, connection_rows=len(edges["target"]),
            execution_clock=identity.clock, external_generators=len(raw["native_id"]),
            external_events=len(external["source"]),
            input_preparation_seconds=time.monotonic()-started)
        candidate, execution = emulate_network_cuda(cells, edges, external, library)
        recording_started = time.monotonic()
        save_source_arrays(output/"input_events.npz", **raw)
        save_source_arrays(output/"cortical_state.npz", **identity.cells,
            terminal_vm_mV=candidate["terminal_voltage_mV"], spike_count=candidate["spike_count"])
        save_source_arrays(output/"cortical_spikes.npz",
            native_id=candidate["spike_native_ids"], time_ms=candidate["spike_times_ms"])
        report.update(recording_seconds=time.monotonic()-recording_started,
            output_npz_compression=dict(method="deflate", level=SOURCE_NPZ_COMPRESSION_LEVEL))
        report.update(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",
            candidate_spikes=len(candidate["spike_times_ms"]), cuda_execution=execution)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
        shutil.copy2(__file__, output/Path(__file__).name)
    return report



def source_block_cases(path: Path) -> tuple[dict, list[dict]]:
    config = json.loads(path.read_text())
    cases = config["cases"]
    require(config["schema"] == "v1_source_input_block_v1" and config["scenario_id"] == "v1_visual"
            and len(cases) == 10 and [x["identity"]["condition_index"] for x in cases] == list(range(10))
            and len({x["identity"]["episode_id"] for x in cases}) == 10
            and len({str(Path(x["input_archive"]).resolve()) for x in cases}) == 10,
            "GLIF_SOURCE_COMPLETE_ORDERED_BLOCK_REQUIRED")
    for row in cases:
        identity = row["identity"]
        require(identity["scenario_id"] == config["scenario_id"]
                and identity["circuit_index"] == config["circuit_index"]
                and identity["block_index"] == config["block_index"]
                and identity["sample_unit"] == "complete_mechanistic_episode",
                "GLIF_SOURCE_BLOCK_IDENTITY_DIFFERS")
    for name in ("circuit_id", "block_id", "connectivity_seed", "initial_state_seed",
                 "background_seed", "stimulus_seed", "split"):
        require(len({row["identity"][name] for row in cases}) == 1,
                "GLIF_SOURCE_BLOCK_SHARED_IDENTITY_DIFFERS:" + name)
    return config, cases


def execute_source_block(archive: Path, cases_file: Path, library: Path,
                         output: Path, snapshot: Path) -> dict:
    """Execute every declared source condition with one immutable circuit load."""
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_SOURCE_BLOCK_FRESH_COMPUTE_OUTPUT_REQUIRED")
    config, cases = source_block_cases(cases_file)
    output.mkdir(parents=True)
    report = dict(status="FAILED_SOURCE_INPUT_GLIF_CUDA_BLOCK",
        job_id=os.environ["SLURM_JOB_ID"], cases_file=str(cases_file),
        cases_sha256=sha256_file(cases_file), circuit_index=config["circuit_index"],
        block_index=config["block_index"], planned_conditions=10, cases=[],
        scope="development_source_backend_execution", actual_split="development",
        identity_split_semantics="prospective_source_sampling_identity_not_production_admission",
        benchmark_episodes=0, production_episodes=0, scientific_qualification=False,
        cortical_outcome_arrays_loaded=False, cortical_outcomes_used_for_execution=False,
        code_sha256=sha256_file(Path(__file__)), runtime_code=source_runtime_code())
    started = time.monotonic()
    try:
        _, report["gpu_device_preflight"] = device_library(library)
        cache = NativeCircuitCache.load(archive, snapshot, include_outcomes=False)
        report.update(matrix_preparation_seconds=time.monotonic()-started,
            coupled_units=cache.cells.count, source_connection_rows=cache.identity.connection_rows)
        for row in cases:
            index = row["identity"]["condition_index"]
            destination = output/("condition_"+str(index).zfill(2))
            try:
                result = execute_source(archive, Path(row["input_archive"]), library, destination,
                                        snapshot, cache=cache, expected_source=row)
            except Exception as error:
                result = (json.loads((destination/"report.json").read_text())
                          if (destination/"report.json").exists() else
                          dict(status="FAILED_BEFORE_CASE_REPORT", error_type=type(error).__name__, error=str(error)))
            report["cases"].append(dict(condition_index=index, identity=row["identity"],
                input_archive=row["input_archive"], report=str((destination/"report.json").relative_to(output)),
                status=result["status"], candidate_spikes=result.get("candidate_spikes"),
                elapsed_seconds=result.get("elapsed_seconds")))
        passed = sum(row["status"] == "PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION" for row in report["cases"])
        report.update(passed_conditions=passed, failed_conditions=10-passed,
            circuit_matrix_loads=1, source_inputs_prepared=cache.source_inputs_prepared,
            circuit_cache_hits=max(0,cache.source_inputs_prepared-1))
        require(passed == 10, "GLIF_SOURCE_BLOCK_EXECUTION_FAILED")
        report["status"] = "PASS_COMPLETE_TEN_CONDITION_SOURCE_INPUT_GLIF_CUDA_BLOCK"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
        shutil.copy2(cases_file, output/"cases.json")
        shutil.copy2(__file__, output/Path(__file__).name)
    return report


def compare(archive: Path, library: Path, output: Path, *, snapshot: Path | None = None,
            cache: NativeCircuitCache | None = None, expected_reference: dict | None = None) -> dict:
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_NETWORK_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC", job_id=os.environ["SLURM_JOB_ID"],
        source_execution_repeated=False, benchmark_episodes=0, production_episodes=0,
        scientific_qualification=False, gpu_backend_qualified=False,
        component_rtol=COMPONENT_RTOL, component_atol=COMPONENT_ATOL,
        code_sha256=sha256_file(Path(__file__)), runtime_code=source_runtime_code())
    started = time.monotonic()
    try:
        if cache is None:
            prepared = read_native(archive, snapshot)
        else:
            require(snapshot is not None and snapshot.resolve() == cache.snapshot,
                    "GLIF_CACHE_SNAPSHOT_DIFFERS")
            prepared = cache.read(archive)
        cells, edges, external, native, binding, original = prepared
        if expected_reference is not None:
            validate_expected_reference(binding, expected_reference)
        candidate, execution = emulate_network_cuda(cells, edges, external, library)
        state, spikes = native["cortical_state"], native["cortical_spikes"]
        np.savez_compressed(output/"cortical_state.npz", native_id=cells.native_ids,
                            source_id=state["source_id"], initial_vm_mV=candidate["initial_voltage_mV"],
                            terminal_vm_mV=candidate["terminal_voltage_mV"], spike_count=candidate["spike_count"])
        np.savez_compressed(output/"cortical_spikes.npz", native_id=candidate["spike_native_ids"],
                            time_ms=candidate["spike_times_ms"])
        order = np.lexsort((spikes["native_id"], spikes["time_ms"]))
        other = np.lexsort((candidate["spike_native_ids"], candidate["spike_times_ms"]))
        pairs = (np.array_equal(spikes["native_id"][order], candidate["spike_native_ids"][other])
                 and np.array_equal(spikes["time_ms"][order], candidate["spike_times_ms"][other]))
        voltage = candidate["terminal_voltage_mV"]
        close = np.isclose(voltage, state["terminal_vm_mV"], rtol=COMPONENT_RTOL, atol=COMPONENT_ATOL)
        counts = candidate["spike_count"]-state["spike_count"]
        report.update(native_archive=str(archive), source_bindings=binding,
            native_reference_status=original["status"], units=cells.count, duration_ms=cells.steps*cells.dt,
            connection_rows=len(edges["target"]), external_generators=len(native["input_events"]["native_id"]),
            external_events=len(external["source"]), native_spikes=len(spikes["time_ms"]),
            candidate_spikes=len(candidate["spike_times_ms"]), all_spike_pairs_exact=pairs,
            all_unit_counts_exact=bool(np.all(counts==0)), count_mismatch_units=int(np.count_nonzero(counts)),
            maximum_unit_count_difference=int(np.abs(counts).max(initial=0)),
            terminal_voltage_within_component_tolerance=bool(close.all()),
            terminal_voltage_mismatch_units=int(np.count_nonzero(~close)),
            terminal_voltage_maximum_absolute_error_mV=float(np.max(np.abs(voltage-state["terminal_vm_mV"]))),
            cuda_execution=execution)
        require(pairs and np.all(counts==0) and close.all(), "GLIF_NETWORK_NATIVE_COMPARISON_DIFFERS")
        report["status"] = "PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
        shutil.copy2(__file__, output/Path(__file__).name)
    return report


def compare_block(cases_file: Path, library: Path, output: Path, snapshot: Path) -> dict:
    """Compare every predeclared condition without rebuilding the circuit matrix."""
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_BLOCK_FRESH_COMPUTE_OUTPUT_REQUIRED")
    config = json.loads(cases_file.read_text())
    cases = config["cases"]
    require(config["schema"] == "v1_native_block_comparison_v1"
            and config["scenario_id"] == "v1_visual" and len(cases) == 10
            and [row["identity"]["condition_index"] for row in cases] == list(range(10))
            and len({row["identity"]["episode_id"] for row in cases}) == 10
            and len({row["native_archive"] for row in cases}) == 10
            and all(row["identity"]["circuit_index"] == config["circuit_index"]
                    and row["identity"]["block_index"] == config["block_index"]
                    and row["identity"]["scenario_id"] == "v1_visual" for row in cases),
            "GLIF_BLOCK_COMPLETE_ORDERED_CONDITIONS_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_NATIVE_GLIF_CUDA_BLOCK_COMPARISON",
        job_id=os.environ["SLURM_JOB_ID"], cases_file=str(cases_file),
        cases_sha256=sha256_file(cases_file), circuit_index=config["circuit_index"],
        block_index=config["block_index"], planned_conditions=10, cases=[],
        scope="development_source_backend_comparison", actual_split="development",
        identity_split_semantics="prospective_source_sampling_identity_not_production_admission",
        benchmark_episodes=0, production_episodes=0, scientific_qualification=False,
        code_sha256=sha256_file(Path(__file__)), runtime_code=source_runtime_code())
    started = time.monotonic()
    try:
        _, report["gpu_device_preflight"] = device_library(library)
        cache = NativeCircuitCache.load(Path(cases[0]["native_archive"]), snapshot)
        report["matrix_preparation_seconds"] = time.monotonic()-started
        report["coupled_units"] = cache.cells.count
        report["source_connection_rows"] = cache.identity.connection_rows
        for row in cases:
            index = row["identity"]["condition_index"]
            destination = output/("condition_"+str(index).zfill(2))
            try:
                result = compare(Path(row["native_archive"]), library, destination,
                                 snapshot=snapshot, cache=cache, expected_reference=row)
            except Exception as error:
                # Retain each failed condition; continue the declared block.
                result = (json.loads((destination/"report.json").read_text())
                          if (destination/"report.json").exists() else
                          dict(status="FAILED_BEFORE_CASE_REPORT", error_type=type(error).__name__, error=str(error)))
            report["cases"].append(dict(condition_index=index, identity=row["identity"],
                native_job_id=row["native_job_id"], report=str((destination/"report.json").relative_to(output)),
                status=result["status"], native_spikes=result.get("native_spikes"),
                candidate_spikes=result.get("candidate_spikes"),
                all_spike_pairs_exact=result.get("all_spike_pairs_exact", False),
                all_unit_counts_exact=result.get("all_unit_counts_exact", False),
                terminal_voltage_within_component_tolerance=result.get("terminal_voltage_within_component_tolerance", False),
                elapsed_seconds=result.get("elapsed_seconds")))
        passed = sum(row["status"] == "PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC" for row in report["cases"])
        report.update(passed_conditions=passed, failed_conditions=10-passed,
                      circuit_matrix_loads=1, circuit_cache_hits=cache.cache_hits)
        require(passed == 10, "GLIF_BLOCK_NATIVE_COMPARISON_FAILED")
        report["status"] = "PASS_COMPLETE_TEN_CONDITION_NATIVE_GLIF_CUDA_BLOCK"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
        shutil.copy2(cases_file, output/"cases.json")
        shutil.copy2(__file__, output/Path(__file__).name)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive",type=Path)
    source.add_argument("--cases",type=Path)
    parser.add_argument("--cuda-library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path)
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument("--source-inputs",type=Path)
    inputs.add_argument("--source-cases",type=Path)
    args=parser.parse_args()
    if args.source_cases is not None:
        require(args.archive is not None and args.snapshot is not None and args.cases is None,
                "GLIF_SOURCE_ARCHIVE_AND_SNAPSHOT_REQUIRED")
        result=execute_source_block(args.archive,args.source_cases,args.cuda_library,args.output,args.snapshot)
    elif args.source_inputs is not None:
        require(args.archive is not None and args.snapshot is not None and args.cases is None,
                "GLIF_SOURCE_ARCHIVE_AND_SNAPSHOT_REQUIRED")
        result=execute_source(args.archive,args.source_inputs,args.cuda_library,args.output,args.snapshot)
    elif args.cases is not None:
        require(args.snapshot is not None, "GLIF_BLOCK_SNAPSHOT_REQUIRED")
        result=compare_block(args.cases,args.cuda_library,args.output,args.snapshot)
    else:
        result=compare(args.archive,args.cuda_library,args.output,snapshot=args.snapshot)
    print(json.dumps(result))


if __name__=="__main__":
    main()

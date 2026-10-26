"""Execute one registered complete PFC source case with a qualified backend.

The full configured network, duration and controls are retained. This runner
archives source-backend outputs and checks execution consistency; it does not
qualify working memory, large-scale source fidelity or production data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as executor
from . import pfc_native_drive as native_drive
from . import pfc_kernel_parameters as kernel_parameters
from .provenance import freeze_json, sha256_file

_SOURCE_REVISION = "d7c4f07fa83a052829e40790fd1594955fe1c067"
_NORMALIZATION_ATOL = 1e-10
_RAW_KEYS = {"spike_unit_indices", "spike_ticks", "spike_seconds", "native_spike_counts", "terminal_voltage_mV"}
_STATE_FIELDS = {
    "E": ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "s_NMDA", "x", "I_stim", "lastspike", "not_refractory"),
    "I": ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "lastspike", "not_refractory"),
}
_IDENTITY_FIELDS = ("source_indices", "archive_indices", "spike_counts")
_CURRENT_ARRAY_DTYPES = {
    "source_unit_indices": np.int64, "archive_unit_indices": np.int64,
    "segment_start_ticks": np.int64, "segment_stop_ticks": np.int64,
    "cue_active": np.bool_, "distractor_active": np.bool_, "current_A": np.float64,
}


def _qualification(directory, backend, dt_ms=0.05, config=None):
    delivery_path = directory / "delivery.json"
    path = directory / "comparison/qualification.json"
    delivery = json.loads(delivery_path.read_text())
    rows = [row for row in delivery["files"] if row["path"] == "comparison/qualification.json"]
    if (delivery.get("status") != "ARCHIVED_PFC_STREAM_VALIDATION"
            or delivery.get("result_status") != "PASS_NATIVE_STREAM_SOURCE_EXECUTION"
            or len(rows) != 1 or path.stat().st_size != rows[0]["bytes"]
            or sha256_file(path) != rows[0]["sha256"]):
        raise ValueError("PFC_SOURCE_EXECUTION_QUALIFICATION_DELIVERY_INVALID")
    receipt = json.loads(path.read_text())
    expected = dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",
                    source_entry_sha256=native_drive.SOURCE_ENTRY_SHA256, backend=backend,
                    numpy_version=np.__version__, executor_sha256=sha256_file(Path(executor.__file__)),
                    drive_sha256=sha256_file(Path(native_drive.__file__)),
                    input_replay_status="PASS_NATIVE_NUMPY_DRIVE_REPLAY", input_replay_passed_comparisons=24,
                    global_cases_passed=2, global_arrays_passed=10)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError("PFC_SOURCE_EXECUTION_QUALIFICATION_MISMATCH:" + key)
    clocks = receipt.get("validated_dt_ms")
    if clocks is None and receipt.get("global_reference_report_sha256") == (
            "99f1f6e8f93499dc0fa6310ade1ecc6cb6364514050542d9e6034d8b69b9814a"):
        # This preserved native reference is explicitly bound to dt=0.05 ms.
        clocks = [0.05]
    if (not isinstance(clocks, list) or len(clocks) != 1
            or type(clocks[0]) not in (int, float) or not np.isfinite(clocks[0]) or clocks[0] <= 0
            or type(dt_ms) not in (int, float) or dt_ms != clocks[0]):
        raise ValueError("PFC_SOURCE_EXECUTION_CLOCK_NOT_QUALIFIED")
    kernel_parameters.require_kernel_coverage(receipt, {} if config is None else config,
                                             sha256_file(Path(kernel_parameters.__file__)))
    return dict(directory=str(directory.resolve()), receipt_file="comparison/qualification.json",
                receipt_sha256=rows[0]["sha256"], delivery_sha256=sha256_file(delivery_path),
                qualification=receipt,
                scope="input-stream and bounded native-output qualification; not fidelity at this requested scale")


def _case_config(config, case_id):
    if (not isinstance(config, dict) or config.get("scope") != "source_reproduction"
            or config.get("expected_source_revision") != _SOURCE_REVISION
            or config.get("synapse_execution") != "factorized_uniform_projections"):
        raise ValueError("PFC_SOURCE_EXECUTION_PINNED_SOURCE_CONFIG_REQUIRED")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("PFC_SOURCE_EXECUTION_REGISTERED_CASES_REQUIRED")
    identities = []
    for row in cases:
        name = row.get("case_id")
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("PFC_SOURCE_EXECUTION_CASE_ID_INVALID")
        identities.append(name)
    if len(set(identities)) != len(identities) or identities.count(case_id) != 1:
        raise ValueError("PFC_SOURCE_EXECUTION_UNIQUE_REGISTERED_CASE_REQUIRED")
    n_e, n_i, dt, ticks, _ = executor._dimensions(config)
    kernel_parameters.resolve_kernel_parameters(config)
    scale = config["weight_scaling_factor"]
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("PFC_SOURCE_EXECUTION_WEIGHT_SCALE_INVALID")
    products = {"E": n_e * scale, "I": n_i * scale}
    if (not np.isclose(products["E"], 2048.0, rtol=0, atol=_NORMALIZATION_ATOL)
            or not np.isclose(products["I"], 512.0, rtol=0, atol=_NORMALIZATION_ATOL)):
        raise ValueError("PFC_SOURCE_EXECUTION_ORIGINAL_NORMALIZATION_REQUIRED")
    return cases[identities.index(case_id)], identities, dict(
        n_e=n_e, n_i=n_i, coupled_core_units=n_e + n_i, dt_seconds=dt,
        duration_ticks=ticks, duration_seconds=ticks * dt,
        weight_scaling_factor=scale, N_E_times_scale=products["E"], N_I_times_scale=products["I"],
        normalization_targets={"E": 2048.0, "I": 512.0}, normalization_atol=_NORMALIZATION_ATOL)


def _validate_applied_current(config, result):
    """Validate observed E current and its declared source-clock coverage.

    Timing is checked against the control schedule; current values are never
    reconstructed from configured amplitudes or substituted for observations.
    """
    n_e, _, dt, ticks, stride = executor._dimensions(config)
    observation = result.get("applied_current_observation")
    checks = {}

    def check(name, passed, **details):
        checks["applied_current_" + name] = dict(passed=bool(passed), **details)

    schema = (isinstance(observation, dict) and set(observation) == {"arrays", "metadata"}
              and isinstance(observation.get("arrays"), dict)
              and set(observation["arrays"]) == set(_CURRENT_ARRAY_DTYPES))
    check("schema", schema, expected_arrays=sorted(_CURRENT_ARRAY_DTYPES))
    if not schema:
        return checks
    arrays, metadata = observation["arrays"], observation["metadata"]
    expected_metadata = executor._applied_current_metadata(n_e, dt, ticks, stride)
    metadata_valid = (isinstance(metadata, dict) and set(metadata) == set(expected_metadata)
                      and all(type(metadata[key]) is type(value) and metadata[key] == value
                              for key, value in expected_metadata.items()))
    check("metadata", metadata_valid, expected=expected_metadata)
    types = all(isinstance(arrays[key], np.ndarray) and arrays[key].dtype == dtype
                for key, dtype in _CURRENT_ARRAY_DTYPES.items())
    check("array_dtypes", types)
    if not types:
        return checks
    starts, stops, values = (arrays[key] for key in ("segment_start_ticks", "segment_stop_ticks", "current_A"))
    segments = starts.size
    shape = (starts.ndim == 1 and 1 <= segments <= 5 and stops.shape == (segments,)
             and arrays["cue_active"].shape == arrays["distractor_active"].shape == (segments,)
             and values.shape == (segments, n_e)
             and arrays["source_unit_indices"].shape == arrays["archive_unit_indices"].shape == (n_e,))
    check("array_shapes", shape, snapshot_count=int(segments), observed_E_units=n_e)
    if not shape:
        return checks
    check("source_unit_identity", np.array_equal(arrays["source_unit_indices"], np.arange(n_e))
          and np.array_equal(arrays["archive_unit_indices"], np.arange(n_e)))
    check("finite_nonnegative_values", np.isfinite(values).all() and np.all(values >= 0), units="A")
    # Compare timing only. Native held-current overlap/persistence is represented
    # by the captured backend vector, not a regenerated stimulus envelope.
    windows = [(config.get(prefix + "_start_ms", 0.0) * 1e-3,
                config.get(prefix + "_duration_ms", 0.0) * 1e-3)
               for prefix in ("stimulus", "distractor")]
    expected_starts, expected_states = [], []
    for tick in range(0, ticks, stride):
        operation_time = (tick // stride) * executor._STIMULUS_DT
        state = tuple(start <= operation_time < start + duration for start, duration in windows)
        if not expected_states or state != expected_states[-1]:
            expected_starts.append(tick)
            expected_states.append(state)
    check("source_clock_coverage", np.array_equal(starts, expected_starts)
          and np.array_equal(stops, expected_starts[1:] + [ticks])
          and np.array_equal(arrays["cue_active"], [state[0] for state in expected_states])
          and np.array_equal(arrays["distractor_active"], [state[1] for state in expected_states]),
          expected_start_ticks=expected_starts, expected_stop_ticks=expected_starts[1:] + [ticks],
          timing="after_source_stimulus_operation_before_rk2",
          interval_semantics=expected_metadata["segment_interval_semantics"])
    check("terminal_consistency", np.array_equal(values[-1], result["terminal_state"].get("E__I_stim")))
    capture_seconds = result.get("cost", {}).get("applied_current_capture_seconds")
    wall_seconds = result.get("cost", {}).get("wall_seconds")
    measured = (type(capture_seconds) in (int, float) and np.isfinite(capture_seconds)
                and type(wall_seconds) in (int, float) and np.isfinite(wall_seconds)
                and 0 <= capture_seconds <= wall_seconds)
    check("measured_capture_cost", measured, capture_seconds=capture_seconds,
          scope="copies and array assembly; included in simulator wall time")
    return checks


def validate_result(config, result, drive):
    """Check all units/events and source-clock refractory constraints."""
    n_e, n_i, dt, ticks, _ = executor._dimensions(config)
    total = n_e + n_i
    raw, terminal = result["raw"], result["terminal_state"]
    expected_state = {pop + "__" + field for pop in ("E", "I")
                      for field in _STATE_FIELDS[pop] + _IDENTITY_FIELDS}
    checks = _validate_applied_current(config, result)

    def check(name, passed, **details):
        checks[name] = dict(passed=None if passed is None else bool(passed), **details)

    check("full_raw_schema", set(raw) == _RAW_KEYS, expected_keys=sorted(_RAW_KEYS), actual_keys=sorted(raw))
    check("full_terminal_schema", set(terminal) == expected_state,
          expected_keys=sorted(expected_state), actual_keys=sorted(terminal))
    if not checks["full_raw_schema"]["passed"] or not checks["full_terminal_schema"]["passed"]:
        check("dependent_checks", None, reason="missing_or_extra_output_arrays")
        return checks
    numeric = all(isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
                  for value in list(raw.values()) + list(terminal.values()))
    check("numeric_array_types", numeric)
    if not numeric:
        check("dependent_checks", None, reason="non_numeric_output_array")
        return checks
    ids, times, seconds = raw["spike_unit_indices"], raw["spike_ticks"], raw["spike_seconds"]
    event_shape = (ids.ndim == times.ndim == seconds.ndim == 1
                   and ids.shape == times.shape == seconds.shape and ids.dtype == np.int64
                   and times.dtype == np.int64 and seconds.dtype == np.float64)
    check("canonical_event_schema", event_shape, event_count=int(ids.size))
    per_unit_shape = (raw["native_spike_counts"].shape == (total,)
                      and raw["native_spike_counts"].dtype == np.int64
                      and raw["terminal_voltage_mV"].shape == (total,)
                      and raw["terminal_voltage_mV"].dtype == np.float64)
    state_shape = all(value.shape == ((n_e if name.startswith("E__") else n_i),)
                      for name, value in terminal.items())
    check("full_unit_output_shapes", per_unit_shape and state_shape, expected_units=total)
    check("finite_terminal_state", all(np.isfinite(value).all() for value in terminal.values()),
          array_count=len(terminal), values_checked=sum(int(value.size) for value in terminal.values()))
    check("finite_raw_outputs", all(np.isfinite(value).all() for value in raw.values()))
    if not event_shape or not per_unit_shape or not state_shape:
        check("dependent_checks", None, reason="invalid_output_shape_or_event_dtype")
        return checks
    legal_ids = bool(np.all((ids >= 0) & (ids < total)))
    legal_times = bool(np.all((times >= 0) & (times < ticks)))
    check("legal_unit_ids", legal_ids, permitted_minimum=0, permitted_exclusive_maximum=total)
    check("legal_spike_ticks", legal_times, permitted_minimum=0, permitted_exclusive_maximum=ticks)
    check("exact_spike_seconds", np.array_equal(seconds, times * dt), dt_seconds=dt)
    ordered = not (np.any(times[1:] < times[:-1]) or np.any((times[1:] == times[:-1]) & (ids[1:] < ids[:-1])))
    check("canonical_tick_then_unit_order", ordered)
    identity_ok = True
    for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
        identity_ok = identity_ok and np.array_equal(terminal[population + "__source_indices"], np.arange(size))
        identity_ok = identity_ok and np.array_equal(terminal[population + "__archive_indices"], np.arange(size) + offset)
    check("full_unit_identity_mapping", identity_ok, archive_population_order=["E", "I"])
    check("terminal_voltage_matches_raw", np.array_equal(raw["terminal_voltage_mV"],
          np.concatenate((terminal["E__v"], terminal["I__v"])) / 0.001), raw_units="mV", state_units="V")
    used = result["inputs_used"]
    consumption = (drive.n_e == n_e and drive.n_i == n_i and drive.dt_seconds == dt
                   and drive.ticks == drive.total_ticks == ticks and drive.remaining == 0
                   and used["ticks_consumed"] == ticks
                   and all(used[population]["values"] == ticks * size
                           and used[population]["events"] == drive.event_totals[population]
                           for population, size in (("E", n_e), ("I", n_i))))
    check("complete_source_input_consumption", consumption, expected_ticks=ticks,
          generated_ticks=drive.ticks, event_totals=drive.event_totals,
          expected_values={"E": ticks * n_e, "I": ticks * n_i})
    if not legal_ids or not legal_times:
        check("event_counts_and_refractory", None, reason="invalid_event_identifiers_or_clock")
        return checks
    reconstructed = np.bincount(ids, minlength=total)
    check("full_event_count_equality", np.array_equal(reconstructed, raw["native_spike_counts"]),
          events=int(ids.size), reconstructed_events=int(reconstructed.sum()))
    check("terminal_spike_count_equality", np.array_equal(reconstructed,
          np.concatenate((terminal["E__spike_counts"], terminal["I__spike_counts"]))))
    order = np.lexsort((times, ids))
    grouped_ids, grouped_times = ids[order], times[order]
    same_unit = grouped_ids[1:] == grouped_ids[:-1]
    gaps = grouped_times[1:] - grouped_times[:-1]
    required = {population: int(np.asarray((duration + 1e-3 * dt) / dt, dtype=np.int64))
                for population, duration in (("E", 2.0 * 1e-3), ("I", 1.0 * 1e-3))}
    for population in ("E", "I"):
        population_mask = grouped_ids[1:] < n_e if population == "E" else grouped_ids[1:] >= n_e
        selected = gaps[same_unit & population_mask]
        violations = int(np.count_nonzero(selected < required[population]))
        check("refractory_spacing_" + population, violations == 0,
              within_unit_event_pairs=int(selected.size), violation_count=violations,
              minimum_required_ticks=required[population],
              minimum_observed_ticks=None if not selected.size else int(selected.min()))
    if np.isfinite(seconds).all():
        expected_last = np.full(total, -1e4, dtype=np.float64)
        np.maximum.at(expected_last, ids, seconds)
        actual_last = np.concatenate((terminal["E__lastspike"], terminal["I__lastspike"]))
        check("terminal_lastspike_matches_events", np.array_equal(actual_last, expected_last),
              never_spiked_sentinel_seconds=-1e4)
        elapsed_steps = np.asarray((((ticks - 1) * dt - expected_last) + 1e-3 * dt) / dt, dtype=np.int64)
        thresholds = np.concatenate((np.full(n_e, required["E"]), np.full(n_i, required["I"])))
        expected_available = elapsed_steps >= thresholds
        # The native threshold operation explicitly clears this flag on a spike.
        expected_available[expected_last == (ticks - 1) * dt] = False
        available = np.concatenate((terminal["E__not_refractory"], terminal["I__not_refractory"]))
        check("terminal_refractory_state_matches_events", available.dtype == np.bool_
              and np.array_equal(available, expected_available))
    else:
        check("terminal_event_state", None, reason="nonfinite_event_seconds")
    return checks


def _units(n_e, n_i):
    return dict(unit_ids=np.arange(n_e + n_i, dtype=np.int64),
                source_unit_indices=np.concatenate((np.arange(n_e, dtype=np.int64), np.arange(n_i, dtype=np.int64))),
                population_index=np.concatenate((np.zeros(n_e, dtype=np.uint8), np.ones(n_i, dtype=np.uint8))),
                excitatory_preference_unit_ids=np.arange(n_e, dtype=np.int64),
                excitatory_preferred_angle_deg=360.0 * np.arange(n_e, dtype=np.float64) / n_e)


def execute(config_path: Path, case_id: str, output: Path, qualification: Path, backend: str):
    started = time.perf_counter()
    config_path, output, qualification = Path(config_path), Path(output), Path(qualification)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="PFC_SOURCE_EXECUTION_FAILED", source_execution_complete=False, case_id=case_id,
                  backend=backend, created_utc=datetime.now(timezone.utc).isoformat(), outputs={},
                  source_entry_sha256=native_drive.SOURCE_ENTRY_SHA256,
                  execution_kind="deterministic source backend with native NumPy input stream",
                  learned_simulator=False, scientific_fidelity_qualified=False, working_memory_qualified=False,
                  fidelity_at_requested_scale_qualified=False, production_episodes=0,
                  code=dict(runner_sha256=sha256_file(Path(__file__)),
                            executor_sha256=sha256_file(Path(executor.__file__)),
                            drive_sha256=sha256_file(Path(native_drive.__file__)),
                             kernel_parameterization_sha256=sha256_file(Path(kernel_parameters.__file__))),
                  costs=dict(initial_generation_seconds=None, simulator_call_seconds=None,
                             native_input_generation_seconds=None, applied_current_capture_seconds=None,
                              serialization_seconds=0.0,
                             output_hashing_seconds=0.0, postrun_validation_seconds=None), validations={})

    def register(name):
        at = time.perf_counter()
        path = output / name
        report["outputs"][name] = dict(path=name, sha256=sha256_file(path), bytes=path.stat().st_size)
        report["costs"]["output_hashing_seconds"] += time.perf_counter() - at

    def save_arrays(name, values, *, single=False):
        at = time.perf_counter()
        arrays = {"value": values} if single else values
        if any(not isinstance(value, np.ndarray) or value.dtype.kind not in "biufU"
               for value in arrays.values()):
            raise ValueError("PFC_SOURCE_EXECUTION_NONARRAY_SERIALIZATION_REJECTED")
        with (output / name).open("xb") as stream:
            if single:
                np.save(stream, values, allow_pickle=False)
            else:
                np.savez(stream, **values)
        report["costs"]["serialization_seconds"] += time.perf_counter() - at
        register(name)

    try:
        if backend not in ("numpy", "cuda"):
            raise ValueError("PFC_SOURCE_EXECUTION_EXPLICIT_BACKEND_REQUIRED")
        config_bytes = config_path.read_bytes()
        config = yaml.safe_load(config_bytes)
        raw_case, identities, dimensions = _case_config(config, case_id)
        report.update(resolved_config=config, selected_case=raw_case, dimensions=dimensions,
                      resolved_kernel_parameters=kernel_parameters.resolve_kernel_parameters(config),
                      circuit_provenance=kernel_parameters.circuit_provenance(config, native_drive.SOURCE_ENTRY_SHA256),
                      registered_case_count=len(identities), registered_case_ids=identities,
                      selected_case_count=1, completed_case_count=0,
                      other_cases="not_executed_by_this_process", original_config_path=str(config_path.resolve()))
        report["backend_qualification"] = _qualification(qualification, backend, config["dt_ms"], config)
        at = time.perf_counter()
        with (output / "config.yaml").open("xb") as stream:
            stream.write(config_bytes)
        report["costs"]["serialization_seconds"] += time.perf_counter() - at
        register("config.yaml")
        report["config_sha256"] = report["outputs"]["config.yaml"]["sha256"]
        freeze_json(output / "request.json", dict(status="PFC_SOURCE_EXECUTION_REQUESTED", case_id=case_id,
            selected_case=raw_case, registered_case_ids=identities, registered_case_count=len(identities),
            config_sha256=report["config_sha256"], backend=backend, dimensions=dimensions,
            code=report["code"], backend_qualification=report["backend_qualification"],
            resolved_kernel_parameters=report["resolved_kernel_parameters"], circuit_provenance=report["circuit_provenance"],
            source_execution_complete=False, scientific_fidelity_qualified=False, production_episodes=0))
        register("request.json")
        at = time.perf_counter()
        initial, drive = native_drive.create_drive(config, raw_case)
        report["costs"]["initial_generation_seconds"] = time.perf_counter() - at
        report["random_stream_policy"] = drive.random_stream_policy
        report["seeds"] = dict(drive.seeds)
        save_arrays("initial_voltage.npy", initial, single=True)
        at = time.perf_counter()
        result = executor.simulate(config, raw_case, initial, backend=backend, drive=drive,
                                   observe_applied_current=True)
        report["costs"]["simulator_call_seconds"] = time.perf_counter() - at
        input_time = result["cost"]["native_input_generation_seconds"]
        if not isinstance(input_time, (int, float)) or not np.isfinite(input_time) or input_time < 0:
            raise ValueError("PFC_SOURCE_EXECUTION_INPUT_COST_INVALID")
        report["costs"]["native_input_generation_seconds"] = input_time
        capture_time = result["cost"].get("applied_current_capture_seconds")
        if type(capture_time) not in (int, float) or not np.isfinite(capture_time) or capture_time < 0:
            raise ValueError("PFC_SOURCE_EXECUTION_CURRENT_CAPTURE_COST_INVALID")
        report["costs"]["applied_current_capture_seconds"] = capture_time
        report["simulator_cost"] = result["cost"]
        report["inputs_used"] = result["inputs_used"]
        report["driver"] = drive.metadata()
        # Full arrays remain available for failure analysis if checks reject them.
        save_arrays("raw.npz", result["raw"])
        save_arrays("terminal_state.npz", result["terminal_state"])
        current_observation = result["applied_current_observation"]
        save_arrays("applied_current_observation.npz", current_observation["arrays"])
        at = time.perf_counter()
        freeze_json(output / "applied_current_metadata.json", current_observation["metadata"])
        report["costs"]["serialization_seconds"] += time.perf_counter() - at
        register("applied_current_metadata.json")
        save_arrays("final_rng.npz", native_drive._state_arrays(drive.get_state()))
        save_arrays("units.npz", _units(dimensions["n_e"], dimensions["n_i"]))
        metadata = dict(unit_file="units.npz", unit_count=dimensions["coupled_core_units"],
            population_index_labels={"0": "E", "1": "I"},
            archive_order="all ascending E source indices, then all ascending I source indices",
            excitatory_preference="360 * source_index / N_E degrees, associated only with E unit IDs",
            inhibitory_preference_coordinate=None, anatomical_soma_coordinates_provided=False,
            initial_voltage_file="initial_voltage.npy", initial_voltage_units="mV",
            applied_current_observation_file="applied_current_observation.npz",
            applied_current_metadata_file="applied_current_metadata.json",
            terminal_state_units={"v": "V", "I_stim": "A", "lastspike": "s",
                                  "synaptic_state_variables": "dimensionless", "not_refractory": "boolean"},
            coupling=dict(representation="implicit source rules; no independent network tiling",
                          E_to_E="full circular source Gaussian-profile FFT, including self term",
                          E_to_I="all E states contribute to each I summed NMDA input",
                          I_to_E="every I spike adds one to every E GABA state",
                          I_to_I="every I spike adds one to all other I GABA states; self excluded",
                          normalization=dimensions["weight_scaling_factor"],
                           resolved_kernel_parameters=report["resolved_kernel_parameters"],
                           circuit_provenance=report["circuit_provenance"]))
        at = time.perf_counter()
        freeze_json(output / "unit_metadata.json", metadata)
        report["costs"]["serialization_seconds"] += time.perf_counter() - at
        register("unit_metadata.json")
        at = time.perf_counter()
        report["validations"] = validate_result(config, result, drive)
        report["costs"]["postrun_validation_seconds"] = time.perf_counter() - at
        passed = all(row["passed"] is True for row in report["validations"].values())
        report["source_execution_complete"] = passed
        report["completed_case_count"] = int(passed)
        report["status"] = "PFC_SOURCE_EXECUTION_COMPLETE" if passed else "PFC_SOURCE_EXECUTION_VALIDATION_FAILED"
        report["neural_event_count"] = int(result["raw"]["spike_ticks"].size)
        report["completed_duration_ticks"] = drive.ticks
        report["source_external_event_totals"] = drive.event_totals
    except Exception as error:
        report["status"] = "PFC_SOURCE_EXECUTION_FAILED"
        report["source_execution_complete"] = False
        report["failure"] = dict(type=type(error).__name__, message=str(error))
    report["costs"]["total_execute_wall_seconds"] = time.perf_counter() - started
    report["costs"]["scope"] = "execute call; module imports, scheduler wait, staging and copyout excluded"
    report["costs"]["component_overlap"] = "native input generation and applied-current capture are included in simulator call time; component times are not additive"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cuda"), required=True)
    args = parser.parse_args()
    report = execute(args.config, args.case_id, args.output, args.qualification, args.backend)
    print(json.dumps({key: report.get(key) for key in
                     ("status", "source_execution_complete", "case_id", "registered_case_count", "backend", "failure")}, indent=2))
    return 0 if report["source_execution_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

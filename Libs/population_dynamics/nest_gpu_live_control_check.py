"""Check live folded TC controls in fresh serial native GPU processes.

This check verifies the native control interface and execution continuity.
It does not qualify an intervention target or a scientific success claim.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

from .mesocircuit_continuation import prepare_pulse_protocol
from .nest_gpu_live_control import (future_poisson_reset_tick, replace_masked_tc_future,
                                   reset_uniform_poisson_future)

DT_MS = 0.1
DELAY_MS = 1.0
BOUNDARY_MS = 10.0
DURATION_MS = 20.0
BOUNDARY_TICK = 100
DURATION_TICKS = 200
N_UNITS = 8
STATE_FIELDS = ("V_m_rel", "I_syn_ex", "I_syn_in", "refractory_step")
ORIGINAL_EMISSION_MS = (2.0, 9.0, 9.5, 10.0, 10.1, 11.0, 14.0)
ACTIVE_TC_MASK = (True, True, False)
NATIVE_SEED = 73011
POISSON_RATE_HZ = 2000.0
SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
MODULE_NAME = "Libs.population_dynamics.nest_gpu_live_control_check"
COMPARE_ARRAYS = (
    "state_records", "terminal_state", "tc_spike_count", "cortex_spike_count",
    "poisson_spike_count", "tc_event_ticks", "tc_event_offsets", "tc_event_times_ms",
    "cortex_event_ticks", "cortex_event_offsets", "cortex_event_times_ms",
    "poisson_event_ticks", "poisson_event_offsets", "poisson_event_times_ms",
    "recurrent_edges",
)


def canary_cases():
    return {
        "unchanged": dict(initial=ORIGINAL_EMISSION_MS, future=None),
        "clear": dict(initial=ORIGINAL_EMISSION_MS, future=()),
        "shorter_replacement": dict(initial=ORIGINAL_EMISSION_MS, future=(10.2, 12.0)),
        "initially_empty": dict(initial=(), future=(10.2, 12.0, 14.0)),
        "exhausted": dict(initial=(2.0, 3.0, 4.0), future=(10.2, 12.0, 14.0)),
        "earliest_next_tick": dict(initial=ORIGINAL_EMISSION_MS, future=(10.1, 12.0)),
    }


def _protocol(name):
    case = canary_cases()[name]
    settings = dict(initial_emission_ms=list(case["initial"]), decision_time_ms=BOUNDARY_MS)
    if case["future"] is not None:
        settings["future_emission_ms"] = list(case["future"])
    return prepare_pulse_protocol(
        dict(thalamic_input=True, thalamic_input_type="pulses", th_pulse_start=3.0,
             th_interval=7.0, th_delay_pulse_generator=DELAY_MS),
        dict(sim_resolution=DT_MS, t_presim=0.0, t_sim=DURATION_MS), settings)


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_error(ng, operation, result=None):
    error_code = int(ng.GetErrorCode())
    if (result is not None and int(result) != 0) or error_code != 0:
        raise RuntimeError(f"{operation} failed: return={result}, native_error={error_code}: "
                           + str(ng.GetErrorMessage()))


def _event_arrays(ng, nodes, prefix):
    events = ng.GetRecSpikeTimes(nodes)
    if len(events) != len(nodes):
        raise ValueError(f"{prefix}: one native event record per node required")
    offsets = [0]
    flattened = []
    for row in events:
        values = np.asarray(row, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) < 0):
            raise ValueError(f"{prefix}: finite ordered native event times required")
        flattened.extend(values.tolist())
        offsets.append(len(flattened))
    times = np.asarray(flattened, dtype=np.float64)
    ticks = np.rint(times / DT_MS).astype(np.int64)
    if not np.allclose(times, ticks * DT_MS, atol=1e-5, rtol=0):
        raise ValueError(f"{prefix}: native event times differ from the configured tick grid")
    counts = np.asarray(ng.GetNeuronStatus(nodes, "spike_count")).reshape(-1)
    _native_error(ng, f"{prefix} event and counter readback")
    if (counts.shape != (len(nodes),) or not np.isfinite(counts).all()
            or np.any(counts < 0) or not np.array_equal(counts, np.rint(counts))):
        raise ValueError(f"{prefix}: one finite nonnegative integer native counter per node required")
    return {
        f"{prefix}_event_times_ms": times,
        f"{prefix}_event_ticks": ticks,
        f"{prefix}_event_offsets": np.asarray(offsets, dtype=np.int64),
        f"{prefix}_spike_count": counts.astype(np.int64),
    }


def _run_native(name, mode, *, future_rng_seed=None, background_delay_ticks=1):
    protocol = _protocol(name)
    ng = importlib.import_module("nestgpu")
    ng.SetTimeResolution(DT_MS)
    ng.SetRandomSeed(NATIVE_SEED)
    core_sequence = ng.Create("iaf_psc_exp", N_UNITS)
    cortex = [int(core_sequence[i]) for i in range(N_UNITS)]
    ng.SetStatus(cortex, dict(tau_m=10.0, C_m=250.0, E_L=-65.0, I_e=500.0,
                             t_ref=2.0, Theta_rel=15.0, V_reset_rel=0.0,
                             tau_ex=2.0, tau_in=1.2, den_delay=0.0,
                             I_syn_ex=0.0, I_syn_in=0.0))
    for index, node in enumerate(cortex):
        ng.SetStatus([node], {"V_m_rel": 10.0 + 0.25 * index})
    # Every core neuron has recurrent incoming and outgoing connections.
    recurrent = []
    for index, source in enumerate(cortex):
        for offset, weight, receptor in ((1, 180.0, 0), (2, -70.0, 1)):
            target = cortex[(index + offset) % N_UNITS]
            ng.Connect([source], [target], {"rule": "all_to_all"},
                       {"weight": weight, "delay": 0.2, "receptor": receptor})
            recurrent.append([index, (index + offset) % N_UNITS, weight, 0.2, receptor])
    tc_sequence = ng.Create("spike_generator", len(ACTIVE_TC_MASK))
    tc_nodes = [int(tc_sequence[i]) for i in range(len(ACTIVE_TC_MASK))]
    schedule_ticks = protocol.initial_ticks if mode == "live" else protocol.effective_ticks
    arrivals = np.asarray(schedule_ticks, dtype=np.int64) + protocol.delay_ticks
    # Fold generator-to-TC delay into TC event times. Native generator Update
    # uses a zero-based index; the recorded event is at the step end.
    folded = ((arrivals - 1) * np.float32(DT_MS)).astype(np.float32).tolist()
    for index, node in enumerate(tc_nodes):
        ng.SetStatus([node], {"spike_times": folded if ACTIVE_TC_MASK[index] else []})
        targets = [cortex[index % 2 + 2 * j] for j in range(N_UNITS // 2)]
        ng.Connect([node], targets, {"rule": "all_to_all"},
                   {"weight": 300.0, "delay": 0.2, "receptor": 0})
    poisson_sequence = ng.Create("poisson_generator", 1)
    poisson = [int(poisson_sequence[0])]
    ng.SetStatus(poisson, {"rate": POISSON_RATE_HZ})
    ng.Connect(poisson, cortex, {"rule": "all_to_all"},
               {"weight": 50.0, "delay": background_delay_ticks * DT_MS, "receptor": 0})
    for nodes in (core_sequence, tc_sequence, poisson_sequence):
        ng.ActivateSpikeCount(nodes)
        ng.ActivateRecSpikeTimes(nodes, 10000)
    record = ng.CreateRecord("", [field for _ in cortex for field in STATE_FIELDS],
                             [node for node in cortex for _ in STATE_FIELDS],
                             [0] * (N_UNITS * len(STATE_FIELDS)))
    ng.SetSimTime(DURATION_MS)
    ng.Calibrate()
    _native_error(ng, "Calibrate")
    library_path = os.environ["NESTGPU_LIB"]
    library = ctypes.CDLL(library_path)
    start = library.NESTGPU_StartSimulation
    start.argtypes = []
    start.restype = ctypes.c_int
    step = library.NESTGPU_SimulationStep
    step.argtypes = []
    step.restype = ctypes.c_int
    _native_error(ng, "StartSimulation", start())
    mutation = dict(future_schedule_replaced=False)
    reset_tick = (future_poisson_reset_tick(BOUNDARY_TICK, background_delay_ticks)
                  if future_rng_seed is not None else None)
    rng_reset = None
    for tick in range(1, DURATION_TICKS + 1):
        _native_error(ng, f"SimulationStep {tick}", step())
        if tick == BOUNDARY_TICK and mode == "live":
            mutation = replace_masked_tc_future(ng, tc_nodes, ACTIVE_TC_MASK, protocol)
        if tick == reset_tick:
            rng_reset = reset_uniform_poisson_future(
                ng, int(poisson_sequence[0]), core_sequence, BOUNDARY_TICK,
                background_delay_ticks, tick, future_rng_seed, DT_MS)
    records = np.asarray(ng.GetRecordData(record), dtype=np.float64)
    terminal = np.column_stack([
        np.asarray(ng.GetNeuronStatus(cortex, field)).reshape(-1) for field in STATE_FIELDS])
    arrays = dict(state_records=records, terminal_state=terminal,
                  recurrent_edges=np.asarray(recurrent, dtype=np.float64))
    for nodes, prefix in ((tc_sequence, "tc"), (core_sequence, "cortex"), (poisson_sequence, "poisson")):
        arrays.update(_event_arrays(ng, nodes, prefix))
    _native_error(ng, "terminal readback")
    return arrays, dict(
        protocol=dict(protocol.record(), boundary_phase="after the continuous native decision-tick step; before device schedule mutation"), mutation=mutation, state_fields=list(STATE_FIELDS),
        unit_ids=cortex, tc_node_ids=tc_nodes,
        active_tc_mask=list(ACTIVE_TC_MASK), poisson_rate_hz=POISSON_RATE_HZ,
        native_seed=NATIVE_SEED, native_start_calls=1, native_step_calls=DURATION_TICKS,
        repeated_simulate_calls=0, library=library_path, library_sha256=_sha256(library_path),
        source_revision=SOURCE_REVISION,
        poisson_event_semantics="direct Poisson emits per outgoing edge; node event counts do not measure drive",
        future_poisson_rng_reset=rng_reset, background_delay_ticks=background_delay_ticks,
        microscopic_state_export_qualified=False, native_future_noise_pairing_verified=False,
        scientific_intervention_qualified=False, scientific_claim_qualified=False)


def _validate_arrays(name, arrays):
    protocol = _protocol(name)
    records = arrays["state_records"]
    if records.shape != (DURATION_TICKS + 1, 1 + N_UNITS * len(STATE_FIELDS)):
        raise ValueError("complete zero-plus-all-step full-neuron state records required")
    if not np.isfinite(records).all():
        raise ValueError("all native state records must be finite")
    record_ticks = np.rint(records[:, 0] / DT_MS).astype(np.int64)
    if (not np.array_equal(record_ticks, np.arange(DURATION_TICKS + 1))
            or not np.allclose(records[:, 0], record_ticks * DT_MS, atol=1e-5, rtol=0)):
        raise ValueError("native record clock must cover exactly ticks zero through 200")
    if arrays["terminal_state"].shape != (N_UNITS, len(STATE_FIELDS)):
        raise ValueError("complete terminal native unit states required")
    if not np.isfinite(arrays["terminal_state"]).all():
        raise ValueError("terminal native states must be finite")
    expected = np.asarray(protocol.effective_ticks, dtype=np.int64) + protocol.delay_ticks
    offsets = arrays["tc_event_offsets"]
    for index, active in enumerate(ACTIVE_TC_MASK):
        actual = arrays["tc_event_ticks"][offsets[index]:offsets[index + 1]]
        required = expected if active else np.empty(0, dtype=np.int64)
        if not np.array_equal(actual, required):
            raise ValueError(f"TC generator {index} differs from the exact effective arrival schedule")
    for prefix in ("tc", "cortex"):
        if not np.array_equal(arrays[f"{prefix}_spike_count"], np.diff(arrays[f"{prefix}_event_offsets"])):
            raise ValueError(f"{prefix}: native counter differs from recorded native events")
    if int(arrays["cortex_spike_count"].sum()) == 0:
        raise ValueError("coupled core neurons must actually spike in the native canary")
    # Before any pulse, initial synaptic currents are zero and the Poisson
    # input is the only source of excitatory current. An early recurrent spike
    # would itself require stochastic drive under these initial conditions.
    early_ex = records[1:20, 2::len(STATE_FIELDS)]
    if not np.any(early_ex > 0):
        raise ValueError("early excitatory current must show actual stochastic drive execution")


def _child(args):
    output = args.native_child_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(case=args.case, mode=args.mode, status="RUNNING",
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  scientific_intervention_qualified=False, scientific_claim_qualified=False)
    _dump(output / "report.json", report)
    try:
        arrays, native = _run_native(args.case, args.mode)
        np.savez_compressed(output / "arrays.npz", **arrays)
        report.update(native, arrays_path="arrays.npz")
        _validate_arrays(args.case, arrays)
        report.update(status="PASS", native_execution_verified=True,
                      native_control_interface_verified=args.mode == "live" and _protocol(args.case).future_ticks is not None)
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), native_control_interface_verified=False)
    report.update(seconds=time.perf_counter() - started,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    _dump(output / "report.json", report)
    return int(report["status"] != "PASS")


def _compare(left, right):
    with np.load(left / "arrays.npz", allow_pickle=False) as a, np.load(right / "arrays.npz", allow_pickle=False) as b:
        mismatches = []
        for name in COMPARE_ARRAYS:
            if name not in a or name not in b or not np.array_equal(a[name], b[name]):
                mismatches.append(name)
    return dict(passed=not mismatches, exact_array_names=list(COMPARE_ARRAYS), mismatched_arrays=mismatches)


def _suite(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    patched_library = Path(os.environ["NESTGPU_LIB"]).resolve()
    original_library = None if args.original_library is None else args.original_library.resolve()
    if not patched_library.is_file() or (original_library is not None and not original_library.is_file()):
        raise ValueError("existing native library paths required")
    names = [args.case] if args.case is not None else list(canary_cases())
    report = dict(status="RUNNING", cases=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  native_execution_serial=True, fresh_process_per_execution=True,
                  original_library_regression_requested=original_library is not None,
                  scientific_intervention_qualified=False, scientific_claim_qualified=False)
    _dump(output / "report.json", report)
    failed = False
    for name in names:
        row = dict(case=name, executions=[], comparisons=[])
        deliveries = {}
        modes = ["preloaded", "live"] + (["original"] if original_library is not None else [])
        for mode in modes:
            child_output = output / "cases" / f"{name}_{mode}"
            command = [sys.executable, "-m", MODULE_NAME, "--output", str(output),
                       "--case", name, "--mode", mode,
                       "--native-child-output", str(child_output)]
            environment = os.environ.copy()
            environment["NESTGPU_LIB"] = str(original_library if mode == "original" else patched_library)
            log_path = output / f"{name}_{mode}.log"
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                execution = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            child_report_path = child_output / "report.json"
            try:
                child_report = json.loads(child_report_path.read_text()) if child_report_path.is_file() else {}
            except (OSError, ValueError) as error:
                child_report = dict(status="INVALID_REPORT", error=str(error))
            passed = execution.returncode == 0 and child_report.get("status") == "PASS"
            deliveries[mode] = (child_output, passed)
            row["executions"].append(dict(mode=mode, command=command, exit_code=execution.returncode,
                                           status=child_report.get("status", "MISSING_REPORT"),
                                           seconds=time.perf_counter() - started,
                                           output_path=str(child_output.relative_to(output)),
                                           log_path=str(log_path.relative_to(output))))
            failed |= not passed
        pairs = [("live", "preloaded", "live_matches_preloaded_effective")]
        if original_library is not None:
            pairs.append(("original", "preloaded", "patched_preloaded_matches_original_library"))
        for left, right, purpose in pairs:
            if deliveries[left][1] and deliveries[right][1]:
                try:
                    comparison = _compare(deliveries[left][0], deliveries[right][0])
                except (OSError, ValueError, KeyError) as error:
                    comparison = dict(passed=False, error=str(error))
            else:
                comparison = dict(passed=False, error="both native executions must pass before exact comparison")
            comparison.update(purpose=purpose, left=left, right=right)
            row["comparisons"].append(comparison)
            failed |= not comparison["passed"]
        row["passed"] = all(e["status"] == "PASS" and e["exit_code"] == 0 for e in row["executions"]) and all(
            c["passed"] for c in row["comparisons"])
        report["cases"].append(row)
        _dump(output / "report.json", report)
    report.update(status="FAILED" if failed else "PASS",
                  all_six_cases_executed=set(names) == set(canary_cases()),
                  native_control_interface_verified=not failed and set(names) == set(canary_cases()),
                  no_mutation_original_library_regression_verified=original_library is not None and not failed,
                  ended_utc=datetime.now(timezone.utc).isoformat())
    _dump(output / "report.json", report)
    return int(failed)


def _rng_child(args):
    """Reuse the continuous coupled canary with one background-stream switch."""
    output = args.native_child_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", scientific_claim_qualified=False)
    started = time.perf_counter()
    try:
        arrays, native = _run_native("unchanged", "preloaded",
            future_rng_seed=args.future_rng_seed, background_delay_ticks=args.background_delay_ticks)
        _validate_arrays("unchanged", arrays)
        np.savez_compressed(output / "arrays.npz", **arrays)
        report.update(native, status="PASS")
    except Exception as error:
        report.update(status="FAILED", error=str(error), traceback=traceback.format_exc())
    report.update(seconds=time.perf_counter() - started,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    _dump(output / "report.json", report)
    return int(report["status"] != "PASS")


def _rng_suite(args):
    """Check original regression, committed arrivals and future-seed replay."""
    if args.original_library is None:
        raise ValueError("original accepted live-control library required for RNG regression")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    patched = Path(os.environ["NESTGPU_LIB"]).resolve()
    report = dict(status="RUNNING", cases=[], native_execution_serial=True,
                  scientific_claim_qualified=False, microscopic_state_export_qualified=False)
    _dump(output / "report.json", report)
    failed = False
    for delay in (2, 17):
        case = dict(native_delay_ticks=delay, boundary_tick=BOUNDARY_TICK,
                    reset_tick=future_poisson_reset_tick(BOUNDARY_TICK, delay), executions=[])
        paths = {}
        for name, seed, library in (("original", None, args.original_library.resolve()),
                                   ("unchanged", None, patched), ("future_a", 73012, patched),
                                   ("future_a_replay", 73012, patched), ("future_b", 73013, patched)):
            path = output / f"delay_{delay}" / name
            command = [sys.executable, "-m", MODULE_NAME, "--output", str(output),
                       "--future-rng-suite", "--native-child-output", str(path),
                       "--background-delay-ticks", str(delay)]
            if seed is not None:
                command.extend(["--future-rng-seed", str(seed)])
            environment = dict(os.environ, NESTGPU_LIB=str(library))
            with (output / f"delay_{delay}_{name}.log").open("w") as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            child = json.loads((path / "report.json").read_text()) if (path / "report.json").is_file() else {}
            passed = result.returncode == 0 and child.get("status") == "PASS"
            case["executions"].append(dict(name=name, seed=seed, command=command,
                                           exit_code=result.returncode, status=child.get("status")))
            failed |= not passed
            paths[name] = path
        if all(row["exit_code"] == 0 and row["status"] == "PASS" for row in case["executions"]):
            original = _compare(paths["original"], paths["unchanged"])
            replay = _compare(paths["future_a"], paths["future_a_replay"])
            with np.load(paths["unchanged"] / "arrays.npz") as base, \
                 np.load(paths["future_a"] / "arrays.npz") as a, \
                 np.load(paths["future_b"] / "arrays.npz") as b:
                stop = case["reset_tick"] + 1
                prefix = all(np.array_equal(base["state_records"][:stop], value["state_records"][:stop])
                             for value in (a, b))
                changed = not np.array_equal(a["state_records"][stop:], b["state_records"][stop:])
                tc_unchanged = all(np.array_equal(base[key], value[key]) for value in (a, b)
                                   for key in COMPARE_ARRAYS if key.startswith("tc_"))
            case.update(original_library_regression=original, same_future_seed_replay=replay,
                        full_unit_states_through_last_committed_arrival_exact=prefix,
                        distinct_future_seeds_change_native_state=changed,
                        requested_tc_program_unchanged=tc_unchanged)
            failed |= not (original["passed"] and replay["passed"] and prefix and changed and tc_unchanged)
        else:
            case["comparison_unavailable"] = "all native executions must pass"
        report["cases"].append(case)
        _dump(output / "report.json", report)
    report.update(status="FAILED" if failed else "PASS",
                  future_poisson_rng_interface_verified=not failed,
                  qualification_scope="eight coupled units; uniform native delays 2 and 17; no full-scale or distributional qualification")
    _dump(output / "report.json", report)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=list(canary_cases()))
    parser.add_argument("--mode", choices=("preloaded", "live", "original"))
    parser.add_argument("--native-child-output", type=Path)
    parser.add_argument("--original-library", type=Path)
    parser.add_argument("--future-rng-suite", action="store_true")
    parser.add_argument("--future-rng-seed", type=int)
    parser.add_argument("--background-delay-ticks", type=int, default=1)
    args = parser.parse_args(argv)
    if args.future_rng_suite:
        if args.case is not None or args.mode is not None:
            parser.error("future RNG suite is separate from TC schedule cases")
        return _rng_child(args) if args.native_child_output is not None else _rng_suite(args)
    if args.future_rng_seed is not None or args.background_delay_ticks != 1:
        parser.error("future RNG arguments require --future-rng-suite")
    if args.native_child_output is not None:
        if args.case is None or args.mode is None:
            parser.error("native child execution requires --case and --mode")
        return _child(args)
    if args.mode is not None:
        parser.error("--mode requires --native-child-output")
    return _suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic, step-resolved qualification of the optional NEST GPU backend.

This diagnostic preserves raw timestamps and failed continuation cases. It does
not qualify spatial connectivity, stochastic drives, or population-scale truth.
Each native case runs in a fresh process; no simulator is imported at import time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import yaml

SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
CPU_RECORDING_DEVICE_DELAY_MS = 1.0
TRACE_COLUMNS = ["time_ms", "voltage_mv", "excitatory_current_pa", "inhibitory_current_pa"]


def psc_voltage_factor(h_ms: float, tau_m_ms: float, tau_s_ms: float,
                       capacitance_pf: float) -> float:
    """Exact subthreshold LIF propagator, including the equal-time-constant limit."""
    if any(not math.isfinite(x) or x <= 0
           for x in (h_ms, tau_m_ms, tau_s_ms, capacitance_pf)):
        raise ValueError("positive finite propagator parameters required")
    delta = 1.0 / tau_m_ms - 1.0 / tau_s_ms
    if delta == 0.0:
        return h_ms * math.exp(-h_ms / tau_m_ms) / capacitance_pf
    return (math.exp(-h_ms / tau_m_ms) * math.expm1(h_ms * delta)
            / (capacitance_pf * delta))


def align_trace(trace: list[list[float]], dt_ms: float,
                time_atol_ms: float) -> dict[int, list[float]]:
    """Map the recorded clock to ticks without fitting a time shift."""
    if not math.isfinite(dt_ms) or dt_ms <= 0 or not 0 <= time_atol_ms < dt_ms / 2:
        raise ValueError("invalid trace clock or alignment tolerance")
    result = {}
    previous = -1
    for row in trace:
        if len(row) != 4 or not all(math.isfinite(x) for x in row):
            raise ValueError("finite time, voltage and two currents required")
        tick = round(row[0] / dt_ms)
        if tick <= previous or abs(row[0] - tick * dt_ms) > time_atol_ms:
            raise ValueError("trace times must increase uniquely on the native grid")
        result[tick] = row
        previous = tick
    return result


def trace_errors(reference: list[list[float]], candidate: list[list[float]],
                 dt_ms: float, time_atol_ms: float) -> dict[str, Any]:
    ref = align_trace(reference, dt_ms, time_atol_ms)
    other = align_trace(candidate, dt_ms, time_atol_ms)
    ticks = sorted(set(ref) & set(other))
    if not ticks:
        raise ValueError("no common recorded ticks")
    # The native CPU multimeter omits endpoints. Only this documented endpoint
    # difference is allowed; missing interior samples cannot disappear in a join.
    for records in (ref, other):
        if set(range(ticks[0], ticks[-1] + 1)) - records.keys():
            raise ValueError("missing interior recorded tick")
    errors = [[other[t][k] - ref[t][k] for k in (1, 2, 3)] for t in ticks]
    return {
        "sample_count": len(ticks), "first_tick": ticks[0], "last_tick": ticks[-1],
        "reference_only_ticks": sorted(set(ref) - set(other)),
        "candidate_only_ticks": sorted(set(other) - set(ref)),
        "max_abs_mv": max(abs(e[0]) for e in errors),
        "max_abs_current_pa": max(abs(e[k]) for e in errors for k in (1, 2)),
        "rmse_mv": math.sqrt(sum(e[0] ** 2 for e in errors) / len(errors)),
    }


def load_protocol(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("campaign") != "neural_population_dynamics":
        raise ValueError("wrong campaign")
    if cfg.get("run_class") != "engineering_diagnostic":
        raise ValueError("this entrypoint only supports engineering diagnostics")
    if cfg.get("source_revision") != SOURCE_REVISION:
        raise ValueError("unreviewed NEST GPU source revision")
    if cfg.get("time_steps_ms") != [0.1, 0.025]:
        raise ValueError("both predeclared time steps are required, once each")
    for h in cfg["time_steps_ms"]:
        if h not in (0.1, 0.025):
            raise ValueError("unregistered diagnostic clock")
    for name in ("voltage_atol_mv", "current_atol_pa", "time_atol_ms"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError("positive finite diagnostic tolerances required")
    return cfg


def make_cases(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Predeclared numerical probes; these are not data-selected neural conditions."""
    cases = []
    parameters = dict(tau_m=10.0, C_m=250.0, E_L=-65.0, V_m=-61.0,
                      V_reset=-60.0, V_th=1000.0, tau_syn_ex=0.5,
                      tau_syn_in=1.2, t_ref=2.0, I_e=0.0)
    for dt in cfg["time_steps_ms"]:
        def add(name: str, **changes: Any) -> None:
            p = deepcopy(parameters)
            p.update(changes.pop("parameters", {}))
            case = dict(name=f"{name}_dt{dt:g}", family=name, dt_ms=dt,
                        duration_ms=10.0, parameters=p, ex_times_ms=[],
                        in_times_ms=[], ex_weight_pa=300.0, in_weight_pa=-150.0,
                        delay_ms=0.2, initial_currents_pa=[0.0, 0.0],
                        comparison="cpu", continuation=None)
            case.update(changes)
            cases.append(case)
        add("constant_drive", parameters={"I_e": 300.0}, comparison="analytic_and_cpu")
        for name, currents in [("initial_ex", [800.0, 0.0]),
                               ("initial_in", [0.0, -300.0]),
                               ("initial_signed", [800.0, -300.0])]:
            add(name, initial_currents_pa=currents, comparison="analytic")
        add("excitatory_impulses", ex_times_ms=[1.0, 3.0])
        add("inhibitory_impulses", in_times_ms=[1.0, 3.0])
        add("coincident_impulses", ex_times_ms=[1.0, 3.0], in_times_ms=[1.0, 3.0])
        add("threshold_reset", parameters={"I_e": 1000.0, "V_th": -50.0},
            duration_ms=20.0)
        add("refractory_inputs", parameters={"V_m": -49.0, "V_th": -50.0},
            ex_times_ms=[0.2, 1.8, 2.0], in_times_ms=[0.3, 1.7], delay_ms=0.1)
        for tau in (10.0, 9.99999, 10.00001):
            add(f"tau_equal_limit_{tau:.8g}",
                parameters={"tau_syn_ex": tau, "tau_syn_in": tau},
                ex_times_ms=[1.0], in_times_ms=[1.0])
        for tref in (2.04, 2.05, 2.06):
            add(f"refractory_rounding_{tref:g}",
                parameters={"V_m": -49.0, "V_th": -50.0, "t_ref": tref},
                ex_times_ms=[0.2, 1.8, 2.0], delay_ms=0.1)
        for name, schedule, continuation in [
            ("original_schedule", [1.0, 3.0, 5.0], None),
            ("revised_schedule", [1.0, 4.0, 6.0], None),
            ("cleared_schedule", [1.0], None),
            ("split_unchanged", [1.0, 3.0, 5.0], {"boundary_ms": 2.0}),
            ("replace_future", [1.0, 3.0, 5.0],
             {"boundary_ms": 2.0, "future_times_ms": [4.0, 6.0]}),
            ("clear_future", [1.0, 3.0, 5.0],
             {"boundary_ms": 2.0, "future_times_ms": []}),
        ]:
            add(name, parameters={"V_m": -65.0}, ex_times_ms=schedule,
                delay_ms=2.0, continuation=continuation)
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("case names must be unique")
    return cases


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _gpu_case(case: dict[str, Any], input_clock: str) -> dict[str, Any]:
    ng = importlib.import_module("nestgpu")
    h = case["dt_ms"]
    p = case["parameters"]
    ng.SetTimeResolution(h)
    ng.SetRandomSeed(73011)
    node = ng.Create("iaf_psc_exp", 1)
    mapped = {k: p[k] for k in ("tau_m", "C_m", "E_L", "I_e", "t_ref")}
    mapped.update(V_m_rel=p["V_m"] - p["E_L"], Theta_rel=p["V_th"] - p["E_L"],
                  V_reset_rel=p["V_reset"] - p["E_L"],
                  tau_ex=p["tau_syn_ex"], tau_in=p["tau_syn_in"], den_delay=0.0,
                  I_syn_ex=case["initial_currents_pa"][0],
                  I_syn_in=case["initial_currents_pa"][1])
    ng.SetStatus(node, mapped)
    ng.ActivateRecSpikeTimes(node, 1000)
    generators = []
    transmitted = []
    def translate(times: list[float]) -> list[float]:
        # Source Update receives a zero-based step index and PushSpike records the
        # end of that step. Translate requested emission times at the input only.
        return [t - h if input_clock == "emission" else t for t in times]
    for port, prefix in enumerate(("ex", "in")):
        generator = ng.Create("spike_generator", 1)
        times = translate(case[f"{prefix}_times_ms"])
        ng.SetStatus(generator, {"spike_times": times})
        ng.ActivateRecSpikeTimes(generator, 1000)
        ng.Connect(generator, node, {"rule": "all_to_all"},
                   {"weight": case[f"{prefix}_weight_pa"],
                    "delay": case["delay_ms"], "receptor": port})
        generators.append(generator)
        transmitted.append(times)
    record = ng.CreateRecord("", ["V_m_rel", "I_syn_ex", "I_syn_in"],
                             [node[0]] * 3, [0] * 3)
    ref_record = ng.CreateRecord("", ["refractory_step"], [node[0]], [0])
    continuation = case["continuation"]
    if continuation is None:
        ng.Simulate(case["duration_ms"])
    else:
        b = continuation["boundary_ms"]
        ng.Simulate(b)
        if "future_times_ms" in continuation:
            # NEST3 accepts a future-only replacement. Preserve pending events.
            future = translate(continuation["future_times_ms"])
            ng.SetStatus(generators[0], {"spike_times": future})
        ng.Simulate(case["duration_ms"] - b)
    raw = ng.GetRecordData(record)
    return {
        "trace": [[r[0], r[1] + p["E_L"], r[2], r[3]] for r in raw],
        "raw_relative_trace": raw, "refractory_trace": ng.GetRecordData(ref_record),
        "spike_times_ms": ng.GetRecSpikeTimes(node)[0],
        "source_spike_times_ms": [ng.GetRecSpikeTimes(g)[0] for g in generators],
        "transmitted_initial_schedules_ms": transmitted,
        "source_status_after_run": [ng.GetStatus(g) for g in generators],
        "library": os.environ["NESTGPU_LIB"],
        "library_sha256": hashlib.sha256(Path(os.environ["NESTGPU_LIB"]).read_bytes()).hexdigest(),
    }


def _cpu_case(case: dict[str, Any]) -> dict[str, Any]:
    nest = importlib.import_module("nest")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": case["dt_ms"], "local_num_threads": 1,
                          "rng_seed": 73011})
    node = nest.Create("iaf_psc_exp", params=case["parameters"])
    meter = nest.Create("multimeter", params={"interval": case["dt_ms"],
                        "record_from": ["V_m", "I_syn_ex", "I_syn_in"]})
    recorder = nest.Create("spike_recorder")
    nest.Connect(meter, node, syn_spec={"delay": CPU_RECORDING_DEVICE_DELAY_MS})
    nest.Connect(node, recorder, syn_spec={"delay": CPU_RECORDING_DEVICE_DELAY_MS})
    generators, source_recorders = [], []
    for prefix in ("ex", "in"):
        generator = nest.Create("spike_generator",
                                params={"spike_times": case[f"{prefix}_times_ms"]})
        source_recorder = nest.Create("spike_recorder")
        nest.Connect(generator, node, syn_spec={"weight": case[f"{prefix}_weight_pa"],
                                               "delay": case["delay_ms"]})
        nest.Connect(generator, source_recorder, syn_spec={"delay": CPU_RECORDING_DEVICE_DELAY_MS})
        generators.append(generator)
        source_recorders.append(source_recorder)
    continuation = case["continuation"]
    if continuation is None:
        nest.Simulate(case["duration_ms"])
    else:
        b = continuation["boundary_ms"]
        nest.Simulate(b)
        if "future_times_ms" in continuation:
            generators[0].set({"spike_times": continuation["future_times_ms"]})
        nest.Simulate(case["duration_ms"] - b)
    events = meter.get("events")
    return {"trace": [[float(events[k][i]) for k in ("times", "V_m", "I_syn_ex", "I_syn_in")]
                      for i in range(len(events["times"]))],
            "spike_times_ms": recorder.get("events")["times"].tolist(),
            "source_spike_times_ms": [r.get("events")["times"].tolist() for r in source_recorders],
            "version": nest.__version__}


def analytic_trace(case: dict[str, Any], actual: list[list[float]]) -> list[list[float]]:
    p = case["parameters"]
    if case["ex_times_ms"] or case["in_times_ms"] or p["V_th"] < 100:
        raise ValueError("analytic propagation probe requires no jumps or thresholds")
    expected = []
    for row in actual:
        t = row[0]
        if t == 0:
            factor_ex = factor_in = 0.0
        else:
            factor_ex = psc_voltage_factor(t, p["tau_m"], p["tau_syn_ex"], p["C_m"])
            factor_in = psc_voltage_factor(t, p["tau_m"], p["tau_syn_in"], p["C_m"])
        decay = math.exp(-t / p["tau_m"])
        v = (p["E_L"] + (p["V_m"] - p["E_L"]) * decay
             + p["I_e"] * p["tau_m"] / p["C_m"] * (1 - decay)
             + sum(i * f for i, f in zip(case["initial_currents_pa"], [factor_ex, factor_in])))
        currents = [i * math.exp(-t / tau) for i, tau in
                    zip(case["initial_currents_pa"], [p["tau_syn_ex"], p["tau_syn_in"]])]
        expected.append([t, v, *currents])
    return expected


def run_one(cfg: dict[str, Any], name: str, backend: str, clock: str, path: Path) -> int:
    case = next(c for c in make_cases(cfg) if c["name"] == name)
    result = {"case": case, "backend": backend, "gpu_input_clock": clock,
              "trace_columns": TRACE_COLUMNS, "status": "FAILED",
              "scientific_reference_qualified": False}
    try:
        if backend == "cpu" and case["comparison"] == "analytic":
            result.update(status="NOT_APPLICABLE", reason="NEST3 does not expose writable PSC initial states")
        else:
            result.update(_gpu_case(case, clock) if backend == "gpu" else _cpu_case(case))
            result["status"] = "EXECUTED"
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    _dump(path, result)
    return int(result["status"] == "FAILED")


def run_suite(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    _dump(output / "resolved_protocol.json", cfg)
    report = {"status": "RUNNING", "backend": args.backend, "gpu_input_clock": args.gpu_input_clock,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_revision": SOURCE_REVISION, "python": sys.version,
              "platform": platform.platform(), "cases": [], "scientific_reference_qualified": False}
    failed = False
    for case in make_cases(cfg):
        command = [sys.executable, "-m", "Libs.population_dynamics.nest_gpu_reference",
                   "--config", str(args.config.resolve()), "one", "--backend", args.backend,
                   "--gpu-input-clock", args.gpu_input_clock, "--case", case["name"],
                   "--output", str((output / (case["name"] + ".json")).resolve())]
        started = time.monotonic()
        with (output / (case["name"] + ".log")).open("w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        failed |= run.returncode != 0
        report["cases"].append({"name": case["name"], "command": command,
                                "exit_code": run.returncode, "seconds": time.monotonic() - started})
        _dump(output / "execution.json", report)
    report.update(status="FAILED" if failed else "EXECUTED",
                  ended_utc=datetime.now(timezone.utc).isoformat())
    _dump(output / "execution.json", report)
    return int(failed)


def _spike_ticks(times: list[float], h: float, atol: float) -> list[int]:
    if any(not math.isfinite(t) or abs(t - round(t / h) * h) > atol for t in times):
        raise ValueError("spike time off the native clock")
    return [round(t / h) for t in times]


def cpu_prearrival_trace(result: dict[str, Any]) -> list[list[float]]:
    """Translate the controlled single-cell CPU recorder to GPU prearrival phase.

    NEST GPU writes records before GetInputSpikes/GetSpikes, whereas CPU records
    contain the newly arrived PSC jumps. Subtract only the *recorded actual*
    incoming impulses at the same tick. Voltage and timestamps are untouched.
    This map is specific to the two explicit source projections in this probe;
    it is not a reconstruction of unknown recurrent network currents.
    """
    case = result["case"]
    h = case["dt_ms"]
    if len(result["source_spike_times_ms"]) != 2:
        raise ValueError("two known input projections required")
    jumps: dict[int, list[float]] = {}
    for port, prefix in enumerate(("ex", "in")):
        for emission in result["source_spike_times_ms"][port]:
            tick = round((emission + case["delay_ms"]) / h)
            jumps.setdefault(tick, [0.0, 0.0])[port] += case[f"{prefix}_weight_pa"]
    return [[row[0], row[1], row[2] - jumps.get(round(row[0] / h), [0., 0.])[0],
             row[3] - jumps.get(round(row[0] / h), [0., 0.])[1]]
            for row in result["trace"]]


def verify_completed_suite(directory: Path, cfg: dict[str, Any], backend: str) -> None:
    """Bind comparison to the completed run's original protocol and full case set."""
    if json.loads((directory / "resolved_protocol.json").read_text()) != cfg:
        raise ValueError("comparison protocol differs from executed protocol")
    receipt = json.loads((directory / "execution.json").read_text())
    expected = [c["name"] for c in make_cases(cfg)]
    if (receipt["status"] != "EXECUTED" or receipt["backend"] != backend
            or [c["name"] for c in receipt["cases"]] != expected
            or any(c["exit_code"] != 0 for c in receipt["cases"])):
        raise ValueError("complete successful native execution records required")


def compare(cfg: dict[str, Any], cpu: Path, gpu: Path, output: Path,
            current_phase: str = "raw") -> int:
    verify_completed_suite(cpu, cfg, "cpu")
    verify_completed_suite(gpu, cfg, "gpu")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in make_cases(cfg):
        name, h = case["name"], case["dt_ms"]
        candidate = json.loads((gpu / (name + ".json")).read_text())
        if candidate["case"] != case or candidate["backend"] != "gpu":
            raise ValueError("GPU case differs from the requested protocol")
        row = {"case": name, "family": case["family"], "passed": False}
        if candidate["status"] != "EXECUTED":
            row["error"] = candidate.get("error", candidate["status"])
            rows.append(row)
            continue
        if case["comparison"] == "analytic":
            reference_trace = analytic_trace(case, candidate["trace"])
            ref = None
        else:
            ref = json.loads((cpu / (name + ".json")).read_text())
            if ref["case"] != case or ref["backend"] != "cpu":
                raise ValueError("CPU case differs from the requested protocol")
            if ref["status"] != "EXECUTED":
                row["error"] = ref.get("error", ref["status"])
                rows.append(row)
                continue
            reference_trace = cpu_prearrival_trace(ref) if current_phase == "prearrival" else ref["trace"]
        if ref is not None and any(len(r["source_spike_times_ms"]) != 2 for r in (ref, candidate)):
            raise ValueError("both actual source spike streams are required")
        errors = trace_errors(reference_trace, candidate["trace"], h, cfg["time_atol_ms"])
        row.update(errors)
        if ref is not None:
            # CPU multimeter packets from the final minimum-delay slice have
            # not yet been delivered. Keep and report the full available
            # interval; do not fit a lag or silently drop interior samples.
            delivery_lag = min(case["delay_ms"], CPU_RECORDING_DEVICE_DELAY_MS)
            last_recorded_tick = round((case["duration_ms"] - delivery_lag) / h)
            if errors["first_tick"] != 1 or errors["last_tick"] != last_recorded_tick:
                raise ValueError("native comparison lacks its complete delivered recording interval")
            row["cpu_record_delivery_lag_ms"] = delivery_lag
            if current_phase == "prearrival":
                row["raw_record_phase_errors"] = trace_errors(ref["trace"], candidate["trace"],
                                                              h, cfg["time_atol_ms"])
        row["state_passed"] = (errors["max_abs_mv"] <= cfg["voltage_atol_mv"]
                               and errors["max_abs_current_pa"] <= cfg["current_atol_pa"])
        row["spike_ticks_match"] = ref is None or (
            _spike_ticks(ref["spike_times_ms"], h, cfg["time_atol_ms"])
            == _spike_ticks(candidate["spike_times_ms"], h, cfg["time_atol_ms"]))
        row["source_spike_ticks_match"] = ref is None or all(
            _spike_ticks(a, h, cfg["time_atol_ms"]) == _spike_ticks(b, h, cfg["time_atol_ms"])
            for a, b in zip(ref["source_spike_times_ms"], candidate["source_spike_times_ms"]))
        if case["comparison"] == "analytic_and_cpu":
            row["analytic_errors"] = trace_errors(analytic_trace(case, candidate["trace"]),
                                                  candidate["trace"], h, cfg["time_atol_ms"])
            row["state_passed"] &= row["analytic_errors"]["max_abs_mv"] <= cfg["voltage_atol_mv"]
        row["passed"] = row["state_passed"] and row["spike_ticks_match"] and row["source_spike_ticks_match"]
        rows.append(row)
    continuations = []
    targets = {"split_unchanged": "original_schedule", "replace_future": "revised_schedule",
               "clear_future": "cleared_schedule"}
    for backend, directory in (("cpu", cpu), ("gpu", gpu)):
        for case in make_cases(cfg):
            if case["family"] not in targets:
                continue
            suffix = f"_dt{case['dt_ms']:g}"
            actual = json.loads((directory / (case["name"] + ".json")).read_text())
            expected = json.loads((directory / (targets[case["family"]] + suffix + ".json")).read_text())
            item = {"backend": backend, "case": case["name"], "passed": False}
            if actual["status"] == expected["status"] == "EXECUTED":
                item.update(trace_errors(expected["trace"], actual["trace"],
                                         case["dt_ms"], cfg["time_atol_ms"]))
                item["source_spike_ticks_match"] = all(
                    _spike_ticks(a, case["dt_ms"], cfg["time_atol_ms"]) ==
                    _spike_ticks(b, case["dt_ms"], cfg["time_atol_ms"])
                    for a, b in zip(expected["source_spike_times_ms"], actual["source_spike_times_ms"]))
                item["passed"] = (item["max_abs_mv"] <= cfg["voltage_atol_mv"]
                                  and item["max_abs_current_pa"] <= cfg["current_atol_pa"]
                                  and item["source_spike_ticks_match"])
            else:
                item["error"] = actual.get("error", expected.get("error", "native case not executed"))
            continuations.append(item)
    passed = all(r["passed"] for r in rows + continuations)
    report = {"status": "PASS" if passed else "FAILED", "cases": rows,
              "continuation_comparisons": continuations, "resolved_protocol": cfg,
              "current_comparison_phase": current_phase,
              "cpu_directory": str(cpu.resolve()), "gpu_directory": str(gpu.resolve()),
              "scientific_reference_qualified": False}
    _dump(output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "cases_passed": sum(r["passed"] for r in rows),
                      "case_count": len(rows),
                      "continuation_passed": sum(r["passed"] for r in continuations),
                      "continuation_count": len(continuations)}, indent=2))
    return int(not passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("run", "one"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=("cpu", "gpu"), required=True)
        p.add_argument("--gpu-input-clock", choices=("native", "emission"), default="native")
        p.add_argument("--output", type=Path, required=True)
        if name == "one":
            p.add_argument("--case", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--current-phase", choices=("raw", "prearrival"), default="raw")
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_protocol(args.config)
    if args.operation == "one":
        if args.output.exists():
            raise FileExistsError(args.output)
        rc = run_one(cfg, args.case, args.backend, args.gpu_input_clock, args.output)
    elif args.operation == "run":
        rc = run_suite(cfg, args)
    else:
        rc = compare(cfg, args.cpu, args.gpu, args.output, args.current_phase)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

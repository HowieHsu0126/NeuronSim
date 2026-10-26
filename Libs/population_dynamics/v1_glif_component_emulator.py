# SPDX-License-Identifier: GPL-2.0-or-later
# GLIF update and propagator translation: Copyright (C) 2004 The NEST Initiative.
# Adapted from pinned NEST glif_psc and iaf_propagator; license in
# Libs/external/nest_glif_reference/LICENSE.
"""Compare a vectorized GLIF3 component implementation with saved native cases.

This is a deterministic component diagnostic. It has no recurrent connectivity
and produces zero benchmark episodes. Its numerical comparison tolerances do
not qualify a network backend or replace scientific fidelity tolerances.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_glif_component_reference import SIGNALS, build_cases, require
from .v1_nest_time_semantics import NEST_REVISION, source_refractory_steps

SOURCE_HASHES = {
    "models/glif_psc.cpp": "41fec8f5f6f6bc4c123ec9392b17def60b38dfc8ac06490df01dfcff48a99e7c",
    "models/glif_psc.h": "4c63d84725375a4a991367cf35af43baeb05ffac6dc36ad692fd9dea4ed39611",
    "libnestutil/iaf_propagator.cpp": "3a5320a565b477bef93fcf3c4e8ec0c9786828d3e2ed9b3ef17441b57a4fc394",
    "libnestutil/iaf_propagator.h": "46a9f991dc2168f0920c257a5b570f5af225a5e784a5d5694a18131c34cde8d9",
}
# These are implementation comparison thresholds, declared before comparison.
COMPONENT_RTOL = 1e-12
COMPONENT_ATOL = 1e-12
PROPAGATOR_STABILITY_FACTOR = 1e-7  # NEST's intrinsic branch constant.


def alpha_propagator(tau_syn: float, tau_m: float, capacitance: float,
                     dt: float) -> tuple[float, float]:
    """Port NEST IAFPropagatorAlpha, including its singular-time-constant branch."""
    require(all(math.isfinite(x) and x > 0 for x in (tau_syn, tau_m, capacitance, dt)),
            "GLIF_COMPONENT_PROPAGATOR_DOMAIN")
    delta = tau_m - tau_syn
    inverse_capacitance = 1.0 / capacitance
    inverse_beta = delta / (tau_syn * tau_m)
    exp_syn = math.exp(-dt * (1.0 / tau_syn))
    expm1 = math.expm1(dt * inverse_beta)
    if delta == 0:
        beta = gamma = math.inf
        regular_min = math.inf
    else:
        beta = tau_syn * tau_m / delta
        gamma = beta / capacitance
        regular_min = PROPAGATOR_STABILITY_FACTOR * tau_m * tau_m / abs(delta)
    p32 = gamma * exp_syn * expm1
    if not (math.isfinite(p32) and p32 >= np.finfo(np.float64).tiny):
        p32 = dt * inverse_capacitance * math.exp(-dt * (1.0 / tau_m))
    if dt > regular_min:
        p31 = gamma * exp_syn * (beta * expm1 - dt)
    else:
        p31 = 0.5 * dt * dt * inverse_capacitance * math.exp(-dt * (1.0 / tau_m))
    return p31, p32


def arrival_counts(emission_ms: np.ndarray, delay_ms: float, dt: float,
                   steps: int) -> np.ndarray:
    """Map this reference's precise emissions to the GLIF ring-buffer stamp.

    glif_psc handles the event stamp and ignores the within-step offset. Multiple
    emissions in one execution step remain multiple arrivals.
    """
    times = np.asarray(emission_ms, dtype=np.float64)
    require(times.ndim == 1 and np.isfinite(times).all() and np.all(times > 0)
            and delay_ms >= dt and delay_ms / dt == int(delay_ms / dt),
            "GLIF_COMPONENT_ARRIVAL_DOMAIN")
    indices = np.ceil(times / dt).astype(np.int64) + int(delay_ms / dt) - 1
    return np.bincount(indices[indices < steps], minlength=steps)


@dataclass
class GlifCellPlan:
    """Source cell parameters shared by isolated and coupled execution."""
    count: int
    receptors: int
    steps: int
    dt: float
    native_ids: np.ndarray
    e_l: np.ndarray
    threshold: np.ndarray
    reset: np.ndarray
    initial_voltage: np.ndarray
    initial_asc: np.ndarray
    refractory_counts: np.ndarray
    asc_amplitudes: np.ndarray
    asc_step_decay: np.ndarray
    asc_mean_coefficient: np.ndarray
    asc_refractory_decay: np.ndarray
    p33: np.ndarray
    p30: np.ndarray
    p11: np.ndarray
    p21: np.ndarray
    p31: np.ndarray
    p32: np.ndarray
    psc_initial: np.ndarray


@dataclass
class ComponentPlan(GlifCellPlan):
    drive: np.ndarray
    input_index: np.ndarray
    input_receptor: np.ndarray
    weights: np.ndarray


def prepare_cells(parameters: list[dict], native_ids: np.ndarray, *,
                  dt: float, duration: float, tics_per_ms: float) -> GlifCellPlan:
    """Compute unchanged NEST GLIF3 coefficients for the declared cells."""
    require(bool(parameters) and native_ids.ndim == 1 and native_ids.dtype.kind in "iu"
            and len(native_ids) == len(parameters) and len(np.unique(native_ids)) == len(native_ids),
            "GLIF_CELL_IDENTITIES_INVALID")
    require(all(p["after_spike_currents"] is True and p["adapting_threshold"] is False
                and p["spike_dependent_threshold"] is False for p in parameters),
            "GLIF_CELL_SOURCE_GLIF3_REQUIRED")
    require(np.isfinite(dt) and dt > 0 and np.isfinite(duration) and duration > dt,
            "GLIF_CELL_CLOCK_INVALID")
    steps = int(duration / dt)
    require(steps * dt == duration, "GLIF_CELL_CLOCK_INVALID")
    selected = parameters
    value = lambda key: np.asarray([p[key] for p in selected], dtype=np.float64)
    count = len(parameters)
    tau_syn = value("tau_syn")
    require(tau_syn.ndim == 2, "GLIF_COMPONENT_RECEPTOR_SHAPE")
    receptors = tau_syn.shape[1]
    e_l, capacitance, conductance = value("E_L"), value("C_m"), value("g")
    tau_m = capacitance / conductance
    threshold, reset = value("V_th") - e_l, value("V_reset") - e_l
    voltage = value("V_m") - e_l
    asc = value("asc_init")
    require(asc.shape == (count, 2), "GLIF_COMPONENT_ASC_SHAPE")
    asc_amplitudes, asc_decay = value("asc_amps"), value("asc_decay")
    # The source GLIF3 bank omits asc_r; NEST declares the default [1, 1].
    asc_r = np.asarray([p.get("asc_r", [1.0, 1.0]) for p in selected])
    refractory_counts = source_refractory_steps(value("t_ref"), tics_per_ms, dt)
    p33 = np.array([math.exp(-dt / tau) for tau in tau_m])
    p30 = 1.0 / capacitance * (1.0 - p33) * tau_m
    p11 = np.array([[math.exp(-dt / tau) for tau in row] for row in tau_syn])
    p21 = dt * p11
    p31 = np.empty_like(p11)
    p32 = np.empty_like(p11)
    for row in range(count):
        for receptor in range(receptors):
            p31[row, receptor], p32[row, receptor] = alpha_propagator(
                tau_syn[row, receptor], tau_m[row], capacitance[row], dt)
    asc_step_decay = np.array([[math.exp(-rate * dt) for rate in row] for row in asc_decay])
    asc_mean_coefficient = ((1.0 / asc_decay) / dt) * (1.0 - asc_step_decay)
    asc_refractory_decay = np.array([
        [asc_r[i, j] * math.exp(-asc_decay[i, j] * p["t_ref"]) for j in range(2)]
        for i, p in enumerate(selected)])
    psc_initial = math.e / tau_syn
    return GlifCellPlan(
        count=count,
        receptors=receptors,
        steps=steps,
        dt=dt,
        native_ids=native_ids,
        e_l=e_l,
        threshold=threshold,
        reset=reset,
        initial_voltage=voltage,
        initial_asc=asc,
        refractory_counts=refractory_counts,
        asc_amplitudes=asc_amplitudes,
        asc_step_decay=asc_step_decay,
        asc_mean_coefficient=asc_mean_coefficient,
        asc_refractory_decay=asc_refractory_decay,
        p33=p33,
        p30=p30,
        p11=p11,
        p21=p21,
        p31=p31,
        p32=p32,
        psc_initial=psc_initial,
    )


def prepare_components(reference: dict) -> ComponentPlan:
    """Add the authenticated component input protocol to the shared cells."""
    references, cases = reference["references"], reference["cases"]
    expected = build_cases(references)
    require(len(cases) == len(expected) and all(
        all(case[key] == item[key] for key in item)
        for case, item in zip(cases, expected)), "GLIF_COMPONENT_CASE_PROTOCOL_DIFFERS")
    ids = np.asarray([case["native_id"] for case in cases], dtype=np.int64)
    parameters = [references[case["source_index"]]["converted_parameters"] for case in cases]
    cells = prepare_cells(parameters, ids, dt=float(reference["dt_ms"]),
                          duration=float(reference["duration_ms"]), tics_per_ms=1000.0)
    expected_refractory = np.asarray(reference["source_refractory_steps"])[
        [case["source_index"] for case in cases]]
    require(np.array_equal(cells.refractory_counts, expected_refractory),
            "GLIF_COMPONENT_REFRACTORY_CLOCK_DIFFERS")
    protocols = reference["input_protocols_ms"]
    names = [None] + list(protocols)
    drive = np.stack([np.zeros(cells.steps, dtype=np.int64)] + [
        arrival_counts(np.asarray(protocols[name]), reference["native_delay_ms"], cells.dt, cells.steps)
        for name in names[1:]])
    return ComponentPlan(**vars(cells), drive=drive,
        input_index=np.array([names.index(case["input_kind"]) for case in cases]),
        input_receptor=np.array([0 if case["receptor_type"] is None else case["receptor_type"]-1 for case in cases]),
        weights=np.array([case["weight_pA"] for case in cases]))


def emulate(reference: dict) -> dict[str, np.ndarray]:
    """Run all declared cases with the source GLIF3 update and output order."""
    plan = prepare_components(reference)
    count = plan.count
    receptors = plan.receptors
    steps = plan.steps
    dt = plan.dt
    native_ids = plan.native_ids
    e_l = plan.e_l
    threshold = plan.threshold
    reset = plan.reset
    refractory_counts = plan.refractory_counts
    asc_amplitudes = plan.asc_amplitudes
    asc_step_decay = plan.asc_step_decay
    asc_mean_coefficient = plan.asc_mean_coefficient
    asc_refractory_decay = plan.asc_refractory_decay
    p33 = plan.p33
    p30 = plan.p30
    p11 = plan.p11
    p21 = plan.p21
    p31 = plan.p31
    p32 = plan.p32
    psc_initial = plan.psc_initial
    drive = plan.drive
    input_index = plan.input_index
    input_receptor = plan.input_receptor
    weights = plan.weights
    voltage, asc = plan.initial_voltage.copy(), plan.initial_asc.copy()
    y1, y2 = np.zeros_like(p11), np.zeros_like(p11)
    refractory = np.zeros(count, dtype=np.int64)
    asc_sum, synaptic_current = asc.sum(axis=1), np.zeros(count)
    output = {"signal_" + name: np.empty((count, steps - 1), dtype=np.float64)
              for name in SIGNALS}
    output.update(initial_voltage_mV=voltage.copy() + e_l, initial_ASC_pA=asc.copy())
    spike_ids, spike_times = [], []
    all_rows = np.arange(count)
    for step in range(steps):
        active = refractory == 0
        # I_syn and ASCurrents_sum retain their previous recorded value while
        # refractory. Hidden synapse states continue to advance below.
        asc_sum[active] = (asc_mean_coefficient * asc).sum(axis=1)[active]
        asc[active] *= asc_step_decay[active]
        voltage[active] = (voltage * p33 + asc_sum * p30)[active]
        synaptic_current[active] = 0.0
        for receptor in range(receptors):
            voltage[active] += (p31[:, receptor] * y1[:, receptor]
                                + p32[:, receptor] * y2[:, receptor])[active]
            synaptic_current[active] += y2[active, receptor]
        fired = active & (voltage > threshold)
        refractory[~active] -= 1
        refractory[fired] = refractory_counts[fired]
        asc[fired] = asc_amplitudes[fired] + asc[fired] * asc_refractory_decay[fired]
        voltage[fired] = reset[fired]
        if fired.any():
            spike_ids.append(native_ids[fired])
            spike_times.append(np.full(int(fired.sum()), (step + 1) * dt))
        y2 = p21 * y1 + p11 * y2
        y1 *= p11
        y1[all_rows, input_receptor] += (psc_initial[all_rows, input_receptor]
                                        * weights * drive[input_index, step])
        if step < steps - 1:
            for name, values in (("V_m", voltage + e_l), ("I_syn", synaptic_current),
                                 ("I", np.zeros(count)), ("ASCurrents_sum", asc_sum),
                                 ("threshold", threshold + e_l),
                                 ("threshold_spike", np.zeros(count)),
                                 ("threshold_voltage", np.zeros(count))):
                output["signal_" + name][:, step] = values
    output.update(terminal_voltage_mV=voltage + e_l, terminal_ASC_pA=asc,
                  spike_native_ids=np.concatenate(spike_ids) if spike_ids else np.empty(0, dtype=np.int64),
                  spike_times_ms=np.concatenate(spike_times) if spike_times else np.empty(0))
    return output


def compare(reference_dir: Path, source_dir: Path, output: Path, *,
            backend: str = "numpy", cuda_library: Path | None = None) -> dict:
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_COMPONENT_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_NATIVE_GLIF_COMPONENT_MAPPING", benchmark_episodes=0,
                  production_episodes=0, gpu_backend_qualified=False,
                  source_network_qualified=False, scientific_qualification=False,
                  job_id=os.environ["SLURM_JOB_ID"],
                  backend=backend, component_rtol=COMPONENT_RTOL, component_atol=COMPONENT_ATOL,
                  implementation_sha256=sha256_file(Path(__file__)))
    started = time.monotonic()
    try:
        source = json.loads((source_dir / "source.json").read_text())
        require(source["revision"] == NEST_REVISION and source["repository"] == "nest/nest-simulator",
                "GLIF_COMPONENT_SOURCE_REVISION_DIFFERS")
        for relative, expected in SOURCE_HASHES.items():
            require(sha256_file(source_dir / relative) == expected,
                    "GLIF_COMPONENT_SOURCE_BYTES_DIFFER:" + relative)
        reference = json.loads((reference_dir / "report.json").read_text())
        require(reference["status"] == "COMPLETE_NATIVE_GLIF_COMPONENT_REFERENCE"
                and reference["source_models"] == 111 and reference["component_cases"] == 1110
                and reference["repeated_spike_coverage_complete"],
                "GLIF_COMPONENT_COMPLETE_NATIVE_BANK_REQUIRED")
        artifact = reference_dir / reference["artifact"]
        require(sha256_file(artifact) == reference["artifact_sha256"],
                "GLIF_COMPONENT_NATIVE_ARRAY_BYTES_DIFFER")
        if backend == "numpy":
            candidate = emulate(reference)
        elif backend == "cuda":
            require(cuda_library is not None, "GLIF_COMPONENT_CUDA_LIBRARY_REQUIRED")
            from .v1_glif_cuda import emulate_cuda  # Optional backend; no native library loaded for CPU use.
            candidate, cuda_execution = emulate_cuda(prepare_components(reference), cuda_library)
            report["cuda_execution"] = cuda_execution
        else:
            raise ValueError("GLIF_COMPONENT_BACKEND_UNKNOWN:" + backend)
        comparisons = {}
        with np.load(artifact, allow_pickle=False) as archive:
            for key, predicted in candidate.items():
                if key.startswith("spike_"):
                    continue
                observed = archive[key]
                require(predicted.shape == observed.shape and predicted.dtype == observed.dtype
                        and np.isfinite(predicted).all(), "GLIF_COMPONENT_OUTPUT_SHAPE:" + key)
                difference = np.abs(predicted - observed)
                close = np.isclose(predicted, observed, rtol=COMPONENT_RTOL, atol=COMPONENT_ATOL)
                comparisons[key] = dict(values=int(predicted.size),
                    exact=bool(np.array_equal(predicted, observed)),
                    maximum_absolute_error=float(difference.max(initial=0)),
                    mismatched_values=int(np.count_nonzero(~close)), within_component_tolerance=bool(close.all()))
            real_ids, real_times = archive["spike_native_ids"], archive["spike_times_ms"]
            order_real = np.lexsort((real_ids, real_times))
            order_candidate = np.lexsort((candidate["spike_native_ids"], candidate["spike_times_ms"]))
            spikes_equal = (np.array_equal(real_ids[order_real], candidate["spike_native_ids"][order_candidate])
                            and np.array_equal(real_times[order_real], candidate["spike_times_ms"][order_candidate]))
        report.update(source_revision=NEST_REVISION, source_files=SOURCE_HASHES,
                      native_reference=str(reference_dir), native_reference_sha256=sha256_file(reference_dir / "report.json"),
                      native_artifact_sha256=reference["artifact_sha256"], source_models=111, component_cases=1110,
                      comparisons=comparisons, spike_events=int(len(real_ids)),
                      candidate_spike_events=int(len(candidate["spike_native_ids"])),
                      all_spike_id_time_pairs_exact=spikes_equal,
                      all_values_within_component_tolerance=all(x["within_component_tolerance"] for x in comparisons.values()))
        require(spikes_equal and report["all_values_within_component_tolerance"],
                "GLIF_COMPONENT_MAPPING_DIFFERS_FROM_NATIVE")
        report["status"] = ("PASS_NATIVE_GLIF_CUDA_COMPONENT_MAPPING" if backend == "cuda"
                            else "PASS_NATIVE_GLIF_COMPONENT_MAPPING")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output / "report.json", report)
        shutil.copy2(__file__, output / Path(__file__).name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["numpy", "cuda"], default="numpy")
    parser.add_argument("--cuda-library", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.source, args.output,
                             backend=args.backend, cuda_library=args.cuda_library)))


if __name__ == "__main__":
    main()

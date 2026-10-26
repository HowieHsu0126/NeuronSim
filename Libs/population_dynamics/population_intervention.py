"""Rank admissible native pulse schedules with full-unit population samples.

The search uses caller-declared reference trajectories, fitting-only field
scales and candidate schedules. It preserves committed emissions and their
in-flight TC arrivals. Predicted rankings require independent native execution
before they can support an intervention claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .mesocircuit_continuation import PulseProtocol, prepare_pulse_protocol
from .poisson_state_space import LatentRandomStreams
from .trajectory_scores import energy_score


@dataclass(frozen=True)
class PulseCandidate:
    name: str
    future_emission_ms: tuple[float, ...]


@dataclass(frozen=True)
class ReferenceTarget:
    field_rate_hz: np.ndarray         # independent reference row, time bin, field
    field_names: tuple[str, ...]
    scale_hz: np.ndarray              # fitted/calibrated externally before search
    independent_block_ids: tuple[str, ...]
    bin_seconds: float


def target_arrays(target: ReferenceTarget, model):
    rates, scale = np.asarray(target.field_rate_hz), np.asarray(target.scale_hz)
    if (target.field_names != model.field_names or target.bin_seconds != model.bin_seconds
            or rates.ndim != 3 or min(rates.shape) < 1 or rates.shape[-1] != len(model.field_names)
            or not np.isfinite(rates).all() or np.any(rates < 0)
            or scale.shape != (len(model.field_names),) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or len(target.independent_block_ids) != len(rates)
            or any(not isinstance(v, str) or not v for v in target.independent_block_ids)
            or len(set(target.independent_block_ids)) != len(rates)):
        raise ValueError("identified finite Hz references, independent blocks and positive frozen field scales required")
    return rates.astype(np.float64), scale.astype(np.float64)


def validate_native_controls(native, net, sim, initial_emission_ms):
    """Bind generator-to-TC timing to actual control events, never future cortex."""
    protocol = prepare_pulse_protocol(net, sim, dict(initial_emission_ms=list(initial_emission_ms)))
    if (protocol is None or protocol.resolution_ms != native.dt_ms
            or not math.isclose(native.duration_ticks * native.dt_ms,
                                sim["t_presim"] + sim["t_sim"], abs_tol=1e-9, rel_tol=0)):
        raise ValueError("complete source-native pulse clock required")
    expected = np.asarray(protocol.initial_ticks, dtype=np.int64) + protocol.delay_ticks
    if not np.array_equal(expected, native.pulse_ticks):
        raise ValueError("declared generator emissions differ from actual source TC controls")
    return protocol


def future_controls(protocol: PulseProtocol, model, horizon_bins: int):
    """Project absolute TC arrivals to the exact half-open model count bins."""
    if (type(horizon_bins) is not int or horizon_bins < 1 or protocol.decision_tick is None
            or protocol.resolution_ms / 1000 != model.tick_seconds
            or protocol.decision_tick % model.bin_ticks or protocol.delay_ticks < 0
            or model.input_size != 1):
        raise ValueError("aligned source-native pulse boundary and one TC-count input required")
    start, stop = protocol.decision_tick, protocol.decision_tick + horizon_bins * model.bin_ticks
    arrivals = np.asarray(protocol.effective_ticks, dtype=np.int64) + protocol.delay_ticks
    if len(np.unique(arrivals)) != len(arrivals):
        raise ValueError("distinct source TC pulse arrivals required")
    kept = arrivals[(arrivals >= start) & (arrivals < stop)]
    return np.bincount((kept - start) // model.bin_ticks, minlength=horizon_bins).astype(np.float32)[None, :, None]


def reference_energy_score(sampled_hz, target: ReferenceTarget, model):
    """Expected finite-ensemble energy score over all declared reference blocks.

    This differs from squared distance between ensemble means. Reference-block
    weighting is equal. No target outcome is used to refit the field scale.
    """
    rates, scale = target_arrays(target, model)
    draws = np.asarray(sampled_hz, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != rates.shape[1:] or len(draws) < 2:
        raise ValueError("complete joint sampled and reference field trajectories required")
    standardized = draws / scale
    scores = [energy_score(standardized, row / scale) for row in rates]
    return float(np.mean(scores)), scores


def sample_candidate(model, state, controls, stream_seeds, draws):
    """Generate every unit and project each bin; keep complete terminal clocks."""
    if (type(draws) is not int or draws < 2 or len(stream_seeds) != 3
            or len(set(stream_seeds)) != 3 or any(type(v) is not int or v < 0 for v in stream_seeds)):
        raise ValueError("at least two draws and three distinct explicit random stream seeds required")
    if torch.any(model.field_unit_counts <= 0):
        raise ValueError("positive source unit denominator for every target field required")
    device = model.field_unit_counts.device
    rng = LatentRandomStreams.from_seeds(*stream_seeds, device=device)
    projected, unit_totals, terminal = [], None, None
    generated_count_values = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in model.iter_simulation(state, controls, rng, draws, record_events=False):
        if step.counts.shape != (draws, 1, len(model.unit_ids)):
            raise ValueError("single-boundary complete unit samples required")
        current = step.counts[:, 0]
        fields = torch.stack([current.index_select(-1, getattr(model, f"field_units_{f}")).sum(-1)
                              for f in range(len(model.field_names))], -1)
        projected.append(fields.cpu().numpy())
        unit_totals = current.clone() if unit_totals is None else unit_totals + current
        terminal = step.ready_in_ticks[:, 0].cpu().numpy()
        generated_count_values += current.numel()
    if len(projected) != controls.shape[1]:
        raise ValueError("complete forecast duration required before candidate ranking")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    field_counts = np.stack(projected, axis=1)
    hz = field_counts / (model.field_unit_counts.cpu().numpy() * model.bin_seconds)
    return dict(field_counts=field_counts, field_rate_hz=hz, unit_total_counts=unit_totals.cpu().numpy(),
                terminal_ready_in_ticks=terminal, generated_unit_count_values=generated_count_values,
                generated_bins=len(projected), generation_and_projection_seconds=time.perf_counter() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)


def design_intervention(model, history, target: ReferenceTarget, *, native, net, sim,
                        initial_emission_ms, decision_time_ms: float, candidates,
                        stream_seeds: tuple[int, int, int], draws: int, pulse_cost_weight: float):
    """Select a predeclared pulse schedule by predicted distribution loss + cost.

    Candidates change only strictly future emissions. Caller-declared target
    scale and cost weight define the objective; no success tolerance is inferred.
    Common model streams control search noise. Native common-noise coupling is
    a separate requirement and is not asserted by matching model stream seeds.
    """
    references, _ = target_arrays(target, model)
    candidates = tuple(candidates)
    if (not candidates or any(not isinstance(c, PulseCandidate) or not isinstance(c.name, str) or not c.name
                             or not isinstance(c.future_emission_ms, tuple) for c in candidates)
            or len({c.name for c in candidates}) != len(candidates)
            or len({c.future_emission_ms for c in candidates}) != len(candidates)
            or not math.isfinite(pulse_cost_weight) or pulse_cost_weight < 0):
        raise ValueError("distinct predeclared candidate schedules and finite nonnegative pulse cost weight required")
    original = validate_native_controls(native, net, sim, initial_emission_ms)
    if (tuple(map(int, native.unit_ids)) != model.unit_ids
            or tuple(map(str, native.field_ids)) != model.field_names
            or not np.array_equal(native.unit_field_index, model.unit_field_index.cpu().numpy())
            or not np.array_equal(native.minimum_interval_ticks, model.minimum_interval_ticks.cpu().numpy())
            or native.bin_ms / 1000 != model.bin_seconds):
        raise ValueError("intervention native observation map differs from fitted model")
    protocols = []
    for candidate in candidates:
        protocol = prepare_pulse_protocol(net, sim, dict(initial_emission_ms=list(initial_emission_ms),
            decision_time_ms=decision_time_ms, future_emission_ms=list(candidate.future_emission_ms)))
        if protocol.decision_tick + references.shape[1] * model.bin_ticks > native.duration_ticks:
            raise ValueError("complete candidate response support outside native episode")
        end = protocol.decision_tick + references.shape[1] * model.bin_ticks
        if any(t + protocol.delay_ticks >= end for t in protocol.future_ticks):
            raise ValueError("every candidate pulse arrival must precede the scored response endpoint")
        controls = future_controls(protocol, model, references.shape[1])
        protocols.append((candidate, protocol, controls))
    # Different sub-bin times indistinguishable to this model must not be
    # advertised as an identifiable control comparison.
    if len({u.tobytes() for _, _, u in protocols}) != len(protocols):
        raise ValueError("candidate schedules are indistinguishable at model temporal resolution")
    boundary = protocols[0][1].decision_tick
    if (np.asarray(history["counts"]).shape != (1, boundary // model.bin_ticks, len(model.unit_ids))
            or np.asarray(history["stimulus"]).shape != (1, boundary // model.bin_ticks, model.input_size)
            or not np.array_equal(np.asarray(history["ready_in_ticks"]), native.ready_at(boundary)[None])):
        raise ValueError("complete causal prefix and native boundary availability required")
    prefix_arrivals = np.asarray(original.initial_ticks, dtype=np.int64) + original.delay_ticks
    past = prefix_arrivals[prefix_arrivals < boundary]
    expected = np.bincount(past // model.bin_ticks, minlength=boundary // model.bin_ticks)
    if not np.array_equal(np.asarray(history["stimulus"])[0, :, 0], expected):
        raise ValueError("history TC counts differ from the known native control program")
    observed, delivered_controls = native.window(0, boundary)
    if (not np.array_equal(np.asarray(history["counts"])[0], observed)
            or not np.array_equal(np.asarray(history["stimulus"])[0], delivered_controls)):
        raise ValueError("intervention history differs from the source observed prefix")
    del observed, delivered_controls
    started = time.perf_counter()
    with torch.no_grad():
        state = model.infer_state(history)
    inference_seconds = time.perf_counter() - started
    rows, predictions = [], {}
    for candidate, protocol, controls in protocols:
        prediction = sample_candidate(model, state, controls, stream_seeds, draws)
        score, reference_scores = reference_energy_score(prediction["field_rate_hz"], target, model)
        cost = len(protocol.future_ticks)
        row = dict(candidate=candidate.name, reference_energy_score=score, future_pulse_count=cost,
                   pulse_cost_weight=pulse_cost_weight, objective=score + pulse_cost_weight * cost,
                   reference_block_scores=reference_scores, protocol=protocol.record(),
                   generated_unit_count_values=prediction["generated_unit_count_values"],
                   generation_and_projection_seconds=prediction["generation_and_projection_seconds"],
                   peak_gpu_allocated_bytes=prediction["peak_gpu_allocated_bytes"])
        rows.append(row)
        predictions[candidate.name] = prediction
    chosen = min(rows, key=lambda row: (row["objective"], row["future_pulse_count"], row["candidate"]))
    report = dict(status="RANKED_DEVELOPMENT_NATIVE_PULSE_CANDIDATES", selected_candidate=chosen["candidate"],
                  candidates=rows, inference_seconds=inference_seconds, sampling_stream_seeds=list(stream_seeds),
                  joint_draws=draws, reference_blocks=list(target.independent_block_ids),
                  target_scale_hz=np.asarray(target.scale_hz).tolist(), field_names=list(model.field_names),
                  source_unit_count=len(model.unit_ids), horizon_bins=references.shape[1],
                  full_unit_outputs_generated=True, all_candidate_outcomes_retained=True,
                  model_common_streams=True, native_future_noise_pairing_verified=False,
                  success_tolerance_qualified=False, inverse_intervention_qualified=False,
                  scientific_claim_qualified=False, native_execution_performed=False)
    return report, predictions

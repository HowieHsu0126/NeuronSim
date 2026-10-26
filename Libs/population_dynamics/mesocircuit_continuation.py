"""Future-only native pulse control; no microscopic state export or RNG reset.

Times are absolute generator emission times in ms. A pulse emitted at the
boundary is committed, including its arrival after the boundary. Configuring
an empty future list cancels only pulses that have not been emitted yet.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

GRID_RATIO_ATOL = 1e-9


def _tick(value, dt: float, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)):
        raise ValueError(f"FINITE_PULSE_TIME_REQUIRED:{name}")
    ratio = value / dt
    if (not math.isfinite(ratio) or abs(ratio) >= 2**53
            or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=GRID_RATIO_ATOL)):
        raise ValueError(f"PULSE_TIME_MUST_ALIGN_WITH_GRID:{name}")
    return round(ratio)


def _schedule(values, dt, delay_ticks, stop_ticks, name) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"PULSE_SCHEDULE_LIST_REQUIRED:{name}")
    ticks = tuple(_tick(value, dt, name) for value in values)
    if any(t <= 0 or t + delay_ticks >= stop_ticks for t in ticks):
        raise ValueError(f"PULSE_ARRIVAL_OUTSIDE_RECORDED_WINDOW:{name}")
    if any(a >= b for a, b in zip(ticks, ticks[1:])):
        raise ValueError(f"PULSE_TIMES_MUST_STRICTLY_INCREASE:{name}")
    return ticks


@dataclass(frozen=True)
class PulseProtocol:
    resolution_ms: float
    delay_ticks: int
    initial_ticks: tuple[int, ...]
    initial_override: bool
    decision_tick: int | None
    future_ticks: tuple[int, ...] | None

    def milliseconds(self, ticks) -> np.ndarray:
        return np.asarray(ticks, dtype=np.float64) * self.resolution_ms

    @property
    def decision_time_ms(self) -> float | None:
        return None if self.decision_tick is None else self.decision_tick * self.resolution_ms

    @property
    def committed_ticks(self) -> tuple[int, ...]:
        return () if self.decision_tick is None else tuple(t for t in self.initial_ticks if t <= self.decision_tick)

    @property
    def effective_ticks(self) -> tuple[int, ...]:
        return self.initial_ticks if self.future_ticks is None else self.committed_ticks + self.future_ticks

    def record(self) -> dict[str, Any]:
        def times(ticks):
            return self.milliseconds(ticks).tolist()
        inflight = () if self.decision_tick is None else tuple(
            t for t in self.committed_ticks if t + self.delay_ticks > self.decision_tick)
        return {
            "schema": "mesocircuit_future_pulse_protocol_v1",
            "time_units": "ms", "time_origin": "native biological time zero",
            "control_semantics": "generator emissions; fixed native spatial mask, weight and delay",
            "resolution_ms": self.resolution_ms,
            "generator_delay_ms": self.delay_ticks * self.resolution_ms,
            "decision_time_ms": self.decision_time_ms,
            "initial_schedule_overridden": self.initial_override,
            "initial_emission_ms": times(self.initial_ticks),
            "requested_future_emission_ms": None if self.future_ticks is None else times(self.future_ticks),
            "committed_emission_ms": times(self.committed_ticks),
            "inflight_tc_arrival_ms": times(tuple(t + self.delay_ticks for t in inflight)),
            "effective_emission_ms": times(self.effective_ticks),
            "effective_tc_arrival_ms": times(tuple(t + self.delay_ticks for t in self.effective_ticks)),
            "boundary_phase": "after native Simulate returns, before any future schedule mutation",
            "observed_history_interval": "[0, decision_time_ms)",
            "full_microscopic_state_qualified": False,
            "paired_stochastic_future_noise_verified": False,
            "inverse_intervention_qualified": False,
        }


def prepare_pulse_protocol(net: dict, sim: dict, settings) -> PulseProtocol | None:
    if settings is None:
        return None
    allowed = {"initial_emission_ms", "decision_time_ms", "future_emission_ms"}
    if not isinstance(settings, dict) or not settings or set(settings) - allowed:
        raise ValueError("PULSE_CONTROL_FIELDS_INVALID")
    if not net["thalamic_input"] or net["thalamic_input_type"] != "pulses":
        raise ValueError("PULSE_CONTROL_REQUIRES_NATIVE_THALAMIC_PULSES")
    dt = float(sim["sim_resolution"])
    delay = _tick(net["th_delay_pulse_generator"], dt, "delay")
    stop = _tick(sim["t_presim"] + sim["t_sim"], dt, "stop")
    default = (np.arange(net["th_pulse_start"], sim["t_presim"] + sim["t_sim"],
                         net["th_interval"]) - net["th_delay_pulse_generator"]).tolist()
    initial = _schedule(settings.get("initial_emission_ms", default), dt, delay, stop, "initial")
    decision = None
    if "decision_time_ms" in settings:
        decision = _tick(settings["decision_time_ms"], dt, "decision")
        if not _tick(sim["t_presim"], dt, "presimulation") <= decision < stop:
            raise ValueError("PULSE_DECISION_MUST_BE_AFTER_PRESIMULATION_BEFORE_STOP")
    future = None
    if "future_emission_ms" in settings:
        if decision is None:
            raise ValueError("PULSE_FUTURE_REQUIRES_DECISION_TIME")
        future = _schedule(settings["future_emission_ms"], dt, delay, stop, "future")
        if any(t <= decision for t in future):
            raise ValueError("PULSE_REPLACEMENT_MUST_BE_STRICTLY_FUTURE")
    return PulseProtocol(dt, delay, initial, "initial_emission_ms" in settings, decision, future)


def replace_future_pulses(generator, nest, protocol: PulseProtocol) -> dict:
    """Mutate only future emissions of a live, already advanced native generator."""
    actual_tick = _tick(float(nest.biological_time), protocol.resolution_ms, "native_time")
    if protocol.decision_tick is None or actual_tick != protocol.decision_tick:
        raise ValueError("PULSE_REPLACEMENT_AT_WRONG_NATIVE_TIME")
    initial = protocol.milliseconds(protocol.initial_ticks)
    actual = np.asarray(generator.spike_times, dtype=float)
    if actual.shape != initial.shape or not np.allclose(actual, initial, rtol=0, atol=1e-10):
        raise ValueError("PULSE_INITIAL_SCHEDULE_CHANGED_BEFORE_DECISION")
    result = {"captured_native_time_ms": float(nest.biological_time),
              "future_schedule_replaced": protocol.future_ticks is not None}
    if protocol.future_ticks is not None:
        future = protocol.milliseconds(protocol.future_ticks)
        generator.set(spike_times=future.tolist())
        actual = np.asarray(generator.spike_times, dtype=float)
        if actual.shape != future.shape or not np.allclose(actual, future, rtol=0, atol=1e-10):
            raise ValueError("PULSE_FUTURE_SCHEDULE_READBACK_MISMATCH")
        result["future_emission_readback_ms"] = actual.tolist()
    return result


def configure_update_interval(nest, net: dict, sim: dict, policy: str) -> dict:
    """Fix kernel scheduling before node creation/connection; never alter delays.

    NEST requires min AND max delay before connections exist. For the source's
    bounded linear spatial delays, the circular masks provide a finite maximum.
    Normal delays are unbounded and are not silently truncated to support this
    optional profile.
    """
    if policy not in {"native", "resolution"}:
        raise ValueError("KERNEL_UPDATE_INTERVAL_POLICY_INVALID")
    if float(nest.biological_time) != 0.:
        raise ValueError("KERNEL_UPDATE_PROFILE_REQUIRES_TIME_ZERO")
    dt = float(sim["sim_resolution"])
    record = {"policy": policy, "resolution_ms": dt,
              "configuration_phase": "after native kernel setup, before create/connect/Prepare",
              "requested_min_delay_ms": None, "requested_max_delay_ms": None,
              "physical_synaptic_delays_modified": False,
              "source_native_random_sequence_preserved": policy == "native"}
    if policy == "resolution":
        if net["delay_type"] != "linear":
            raise ValueError("RESOLUTION_UPDATE_REQUIRES_BOUNDED_LINEAR_DELAYS")
        radius = np.minimum(np.asarray(net["mask_radius"]), float(net["extent"]) / np.sqrt(2.))
        spatial_max = float(np.max(np.asarray(net["delay_offset_matrix"]) + radius / np.asarray(net["prop_speed_matrix"])))
        default_delay = float(nest.GetDefaults("static_synapse")["delay"])
        bound = max(dt, spatial_max, default_delay, float(net["delay_poisson"]),
                    float(net["th_delay_pulse_generator"]) if net["thalamic_input"] and net["thalamic_input_type"] == "pulses" else dt)
        maximum = math.ceil(bound / dt) * dt
        nest.SetKernelStatus({"min_delay": dt, "max_delay": maximum})
        record.update(requested_min_delay_ms=dt, requested_max_delay_ms=maximum,
                      unrounded_physical_delay_upper_bound_ms=bound,
                      max_delay_bound="source linear delay over circular mask and periodic square; includes external/default device delays")
    return record


def check_update_segments(nest, sim: dict, protocol: PulseProtocol | None, record: dict) -> None:
    """Reject segmentation that changes native Poisson RNG scheduling order."""
    dt = float(sim["sim_resolution"])
    minimum, maximum = float(nest.min_delay), float(nest.max_delay)
    minimum_tick = _tick(minimum, dt, "kernel_min_delay")
    if minimum_tick <= 0:
        raise ValueError("KERNEL_MIN_DELAY_MUST_BE_POSITIVE")
    record.update(effective_min_delay_ms=minimum, effective_max_delay_ms=maximum)
    if record["policy"] == "resolution":
        if (minimum_tick != 1 or _tick(maximum, dt, "kernel_max_delay")
                != _tick(record["requested_max_delay_ms"], dt, "requested_max_delay")):
            raise ValueError("KERNEL_UPDATE_INTERVAL_READBACK_MISMATCH")
    presim = _tick(sim["t_presim"], dt, "presimulation")
    total = presim + _tick(sim["t_sim"], dt, "simulation")
    boundary = None if protocol is None else protocol.decision_tick
    segments = [("presimulation", presim)]
    if boundary is None:
        segments.append(("simulation", total - presim))
    else:
        segments.extend([("predecision", boundary - presim), ("continuation", total - boundary)])
    record["segments"] = [{"name": name, "duration_ticks": ticks,
                            "aligned_with_kernel_min_delay": ticks % minimum_tick == 0}
                           for name, ticks in segments]
    record["segmentation_aligned"] = all(row["aligned_with_kernel_min_delay"] for row in record["segments"])
    # Native source runs without an explicit pulse protocol retain their original
    # behavior. All continuation comparisons require safe segmentation.
    if protocol is not None and not record["segmentation_aligned"]:
        raise ValueError("NATIVE_SEGMENTATION_CHANGES_RNG_USE_RESOLUTION_UPDATE_PROFILE")

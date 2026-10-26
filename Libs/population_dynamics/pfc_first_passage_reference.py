"""Reconstruct source synaptic state from a complete, bound PFC spike tape.

No membrane equation is integrated here.  An observed tape is a suppressed-spike
reference only when the selected unit has no further spike in its window.  This
module supplies numerical prerequisites, not a first-passage probability.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import pfc_deterministic_replay as source
from .pfc_threshold_branch_probe import canonical_events


FIELDS = ("s_AMPA", "s_GABA", "s_NMDA_total", "lastspike", "not_refractory")
E_FIELDS = ("I_stim", "s_NMDA", "x")
Snapshot = dict[str, np.ndarray]
SnapshotCallback = Callable[[str, int, Snapshot], None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_FIRST_PASSAGE_" + message)


@dataclass(frozen=True)
class SpikeTape:
    """Canonical complete events, checked against the native full-unit counts.

    The caller must additionally bind these raw arrays to the accepted source
    receipt.  Internal count agreement cannot authenticate a rewritten archive.
    """

    events: np.ndarray
    unit_counts: np.ndarray
    n_e: int
    n_i: int
    dt_seconds: float
    total_ticks: int


def validate_tape(config: Mapping[str, Any], raw: Mapping[str, np.ndarray]) -> SpikeTape:
    n_e, n_i, dt, ticks, _ = source._dimensions(config)
    required = {"spike_ticks", "spike_unit_indices", "native_spike_counts"}
    require(isinstance(raw, Mapping) and required <= raw.keys(), "COMPLETE_SPIKE_TAPE_REQUIRED")
    events = canonical_events(raw["spike_ticks"], raw["spike_unit_indices"], n_e + n_i, ticks)
    counts = np.asarray(raw["native_spike_counts"])
    require(counts.shape == (n_e + n_i,) and counts.dtype.kind in "iu"
            and np.all(counts >= 0), "FULL_UNIT_SPIKE_COUNTS_REQUIRED")
    observed = np.bincount(events[:, 1], minlength=n_e + n_i)
    require(np.array_equal(observed, counts), "MISSING_OR_EXTRA_SPIKE_EVENTS")
    if "spike_seconds" in raw:
        seconds = np.asarray(raw["spike_seconds"])
        require(seconds.dtype == np.float64 and seconds.shape == np.asarray(raw["spike_ticks"]).shape
                and np.array_equal(seconds, np.asarray(raw["spike_ticks"]) * dt), "SPIKE_TIME_UNITS")
    counts = counts.copy()
    events.setflags(write=False)
    counts.setflags(write=False)
    return SpikeTape(events, counts, n_e, n_i, dt, ticks)


@dataclass(frozen=True)
class FirstSpikeWindow:
    unit_index: int
    first_spike_tick: int
    window_start_tick: int
    window_stop_tick: int
    refractory_ticks: int


def _check_tape(config: Mapping[str, Any], tape: SpikeTape) -> tuple[int, int, float, int, int]:
    dimensions = source._dimensions(config)
    require(isinstance(tape, SpikeTape)
            and dimensions[:4] == (tape.n_e, tape.n_i, tape.dt_seconds, tape.total_ticks),
            "TAPE_CONFIG_MISMATCH")
    return dimensions


def select_first_spike_window(
    config: Mapping[str, Any], tape: SpikeTape, *, eligible_units: Sequence[int],
    phase_start_tick: int, phase_stop_tick: int, window_ms: float = 5.0,
) -> FirstSpikeWindow:
    """Select earliest first-spike time, then unit, without inspecting later spikes.

    Units and phase must be fixed before selection.  A complete threshold tick
    after the source refractory period must remain in the selected window.
    """
    n_e, n_i, dt, ticks, _ = _check_tape(config, tape)
    units = np.asarray(eligible_units)
    require(units.ndim == 1 and units.size > 0 and units.dtype.kind in "iu"
            and np.all((units >= 0) & (units < n_e + n_i))
            and np.unique(units).size == units.size, "ELIGIBLE_UNITS_INVALID")
    require(type(window_ms) in (int, float) and np.isfinite(window_ms) and window_ms > 0,
            "WINDOW_INVALID")
    window_ticks = round(window_ms / config["dt_ms"])
    require(window_ticks > 0 and np.isclose(window_ticks * config["dt_ms"], window_ms, rtol=0, atol=1e-12),
            "WINDOW_OFF_CLOCK")
    require(type(phase_start_tick) is int and type(phase_stop_tick) is int
            and 0 <= phase_start_tick < phase_stop_tick <= ticks
            and phase_start_tick % window_ticks == phase_stop_tick % window_ticks == 0,
            "COMPLETE_PHASE_WINDOWS_REQUIRED")
    allowed = np.zeros(n_e + n_i, dtype=bool)
    allowed[units] = True
    seen: set[tuple[int, int]] = set()
    for event_tick, unit in tape.events:
        tick, unit = int(event_tick), int(unit)
        if tick < phase_start_tick or tick >= phase_stop_tick or not allowed[unit]:
            continue
        start = (tick // window_ticks) * window_ticks
        key = (unit, start)
        if key in seen:
            continue
        seen.add(key)
        population = "E" if unit < n_e else "I"
        refractory = source._constants(config, population)["refractory"]
        refractory_ticks = int(np.asarray((refractory + 1e-3 * dt) / dt, dtype=np.int64))
        stop = start + window_ticks
        if tick + refractory_ticks < stop:
            return FirstSpikeWindow(unit, tick, start, stop, refractory_ticks)
    raise ValueError("PFC_FIRST_PASSAGE_NO_ELIGIBLE_FIRST_SPIKE_WINDOW")


def require_observed_reference(tape: SpikeTape, selection: FirstSpikeWindow) -> None:
    """Reject selection-conditioned reuse; a hit needs a suppressed-network replay."""
    require(isinstance(selection, FirstSpikeWindow)
            and 0 <= selection.unit_index < tape.n_e + tape.n_i
            and 0 <= selection.window_start_tick <= selection.first_spike_tick
            < selection.window_stop_tick <= tape.total_ticks, "SELECTED_WINDOW_INVALID")
    events = tape.events
    times = events[(events[:, 1] == selection.unit_index)
                   & (events[:, 0] >= selection.window_start_tick)
                   & (events[:, 0] < selection.window_stop_tick), 0]
    require(times.size > 0 and int(times[0]) == selection.first_spike_tick,
            "SELECTED_EVENT_IS_NOT_FIRST_SPIKE")
    require(times.size == 1, "SUPPRESSED_NETWORK_REPLAY_REQUIRED")


def _advance(state: dict[str, np.ndarray], constants: Mapping[str, float], dt: float,
             population: str) -> None:
    """The source's non-membrane RK2 expressions, preserving their operation order."""
    a, b = state["s_AMPA"], state["s_GABA"]
    la, lb = (-dt) / constants["tau_ampa"], (-dt) / constants["tau_gaba"]
    ka, kb = la * a, lb * b
    a_mid, b_mid = (0.5 * ka) + a, (0.5 * kb) + b
    next_a, next_b = (la * a_mid) + a, (lb * b_mid) + b
    if population == "E":
        s, x = state["s_NMDA"], state["x"]
        inverse_tau = 1.0 / constants["tau_nmda_s"]
        alpha, lx = constants["alpha_nmda"], (-dt) / constants["tau_nmda_x"]
        ks = dt * ((alpha * (x * (1.0 - s))) - (inverse_tau * s))
        kx = lx * x
        next_s = (dt * ((alpha * (((0.5 * kx) + x) * ((1.0 + (0.5 * (-ks))) - s)))
                        - (inverse_tau * ((0.5 * ks) + s)))) + s
        next_x = (lx * ((0.5 * kx) + x)) + x
        state.update(s_NMDA=next_s, x=next_x)
    state.update(s_AMPA=next_a, s_GABA=next_b)


def _snapshot(populations: Mapping[str, Mapping[str, np.ndarray]]) -> Snapshot:
    return {population + "__" + key: value.copy()
            for population, state in populations.items() for key, value in state.items()}


def _state_view(populations: Mapping[str, Mapping[str, np.ndarray]]) -> Snapshot:
    """Borrow read-only arrays for a synchronous callback; values can later change."""
    result = {}
    for population, state in populations.items():
        for key, value in state.items():
            view = value.view()
            view.flags.writeable = False
            result[population + "__" + key] = view
    return result


def reconstruct(
    config: Mapping[str, Any], case: Mapping[str, Any], tape: SpikeTape, *, stop_tick: int,
    counts_E: np.ndarray | None = None, counts_I: np.ndarray | None = None, drive: Any = None,
    selection: FirstSpikeWindow | None = None, on_snapshot: SnapshotCallback | None = None,
    capture_ticks: Sequence[int] | None = None,
    on_state_view: SnapshotCallback | None = None,
) -> dict[str, Any]:
    """Reconstruct a complete prefix; copy selected snapshots to a caller callback.

    Snapshots are pre_rk2, pre_threshold, and pre_input (after GABA projection).
    Input is either recorded uint16 prefix arrays or a fresh matched source drive.
    The conditional record deliberately excludes the selected tick's fresh input.
    on_state_view reads borrowed arrays synchronously. Copy values that must survive
    the callback. on_snapshot retains its independent, copied-array contract.
    Projections are evaluated at observed, conditional-selection and terminal
    ticks. The projections do not enter the synaptic state equations.
    No voltage trajectory or first-passage probability is produced.
    """
    n_e, n_i, dt, total_ticks, stimulus_stride = _check_tape(config, tape)
    require(on_snapshot is None or on_state_view is None, "ONE_STATE_CALLBACK_REQUIRED")
    require(type(stop_tick) is int and 0 < stop_tick <= total_ticks, "PREFIX_STOP_INVALID")
    if selection is not None:
        require_observed_reference(tape, selection)
        require(selection.window_stop_tick <= stop_tick, "REFERENCE_WINDOW_INCOMPLETE")
    requested = None if capture_ticks is None else frozenset(capture_ticks)
    if requested is not None:
        require(all(type(t) is int and 0 <= t < stop_tick for t in requested), "CAPTURE_TICK_INVALID")
    if drive is None:
        for values, size in ((counts_E, n_e), (counts_I, n_i)):
            require(isinstance(values, np.ndarray) and values.dtype == np.uint16 and values.ndim == 2
                    and values.shape[1] == size and stop_tick <= values.shape[0] <= total_ticks,
                    "COMPLETE_UINT16_INPUT_PREFIX_REQUIRED")
    else:
        require(counts_E is None and counts_I is None and drive.n_e == n_e and drive.n_i == n_i
                and drive.dt_seconds == dt and drive.ticks == 0 and drive.total_ticks == total_ticks,
                "FRESH_MATCHED_SOURCE_DRIVE_REQUIRED")
    arithmetic = source._Arithmetic("numpy")
    constants = {p: source._constants(config, p) for p in ("E", "I")}
    populations: dict[str, dict[str, np.ndarray]] = {}
    for population, size in (("E", n_e), ("I", n_i)):
        state = {key: np.zeros(size, dtype=np.float64) for key in ("s_AMPA", "s_GABA", "s_NMDA_total")}
        state.update(lastspike=np.full(size, -1e4, dtype=np.float64), not_refractory=np.ones(size, dtype=bool))
        if population == "E":
            state.update({key: np.zeros(size, dtype=np.float64) for key in E_FIELDS})
        populations[population] = state
    e, i = populations["E"], populations["I"]
    program = source._control(config, case, n_e, arithmetic)
    fft_kernel = arithmetic.rfft(arithmetic.array(source._kernel(n_e, config)))
    chunk_rows = max(1, min(256, (32 * 1024 * 1024) // (8 * (n_e + n_i))))
    event_cursor = 0
    conditional = None
    has_observer = on_snapshot is not None or on_state_view is not None

    def emit(stage: str, tick: int) -> None:
        if requested is None or tick in requested:
            if on_snapshot is not None:
                on_snapshot(stage, tick, _snapshot(populations))
            elif on_state_view is not None:
                on_state_view(stage, tick, _state_view(populations))

    for tick in range(stop_tick):
        if tick % chunk_rows == 0:
            stop = min(tick + chunk_rows, stop_tick)
            blocks = ({"E": counts_E[tick:stop], "I": counts_I[tick:stop]} if drive is None
                      else drive.take(stop - tick))
            inputs = {}
            for population, size in (("E", n_e), ("I", n_i)):
                block = np.asarray(blocks[population])
                require(block.dtype == np.uint16 and block.shape == (stop - tick, size)
                        and np.all(block <= 1000), "INPUT_PREFIX_SUPPORT_OR_SHAPE")
                inputs[population] = block.astype(np.float64)
        t = tick * dt
        # Neither projection enters _advance. Preserve its source pre-RK2
        # value when a callback, conditional record or terminal output reads it.
        project = ((has_observer and (requested is None or tick in requested))
                   or (selection is not None and tick == selection.first_spike_tick)
                   or tick == stop_tick - 1)
        if project:
            e["s_NMDA_total"] = arithmetic.irfft(fft_kernel * arithmetic.rfft(e["s_NMDA"]))
        if tick % stimulus_stride == 0:
            source._held_current(e["I_stim"], program, (tick // stimulus_stride) * source._STIMULUS_DT)
        if project:
            i["s_NMDA_total"][:] = arithmetic.cumulative_last(e["s_NMDA"])
        emit("pre_rk2", tick)
        for population, state in populations.items():
            refractory = constants[population]["refractory"]
            refractory_ticks = int(np.asarray((refractory + 1e-3 * dt) / dt, dtype=np.int64))
            available = arithmetic.integer_steps(t - state["lastspike"], dt) >= refractory_ticks
            _advance(state, constants[population], dt, population)
            state["not_refractory"] = available
        emit("pre_threshold", tick)
        end = int(np.searchsorted(tape.events[:, 0], tick + 1, side="left"))
        units = tape.events[event_cursor:end, 1]
        event_cursor = end
        spikes = {"E": units[units < n_e], "I": units[units >= n_e] - n_e}
        for population, selected in spikes.items():
            state = populations[population]
            require(np.all(state["not_refractory"][selected]), "TAPE_REFRACTORY_VIOLATION")
            state["not_refractory"][selected] = False
            state["lastspike"][selected] = t
        for index in spikes["I"]:
            i["s_GABA"][:int(index)] += 1.0
            i["s_GABA"][int(index) + 1:] += 1.0
        for _ in spikes["I"]:
            e["s_GABA"][:] += 1.0
        emit("pre_input", tick)
        if selection is not None and tick == selection.first_spike_tick:
            population = "E" if selection.unit_index < n_e else "I"
            index = selection.unit_index if population == "E" else selection.unit_index - n_e
            state = populations[population]
            record = {key: value[index].item() for key, value in state.items()}
            record["v"] = constants[population]["reset"]
            if population == "E":
                record["x"] = float(state["x"][index] + 1.0)
            conditional = dict(unit_index=selection.unit_index, population=population,
                               first_spike_tick=tick, window_stop_tick=selection.window_stop_tick,
                               state=record, timing="first_spike_reset_with_preinput_ampa",
                               fresh_input_included=False, voltage_units="V", time_units="s")
        i["s_AMPA"][:] += inputs["I"][tick % chunk_rows]
        e["s_AMPA"][:] += inputs["E"][tick % chunk_rows]
        e["x"][spikes["E"]] += 1.0
    require(selection is None or conditional is not None, "CONDITIONAL_INITIAL_STATE_MISSING")
    require(drive is None or drive.ticks == stop_tick, "SOURCE_DRIVE_PREFIX_CLOCK")
    return dict(status="RECONSTRUCTED_PFC_FIRST_PASSAGE_REFERENCE", ticks_reconstructed=stop_tick,
                source_events_consumed=event_cursor, terminal_state=_snapshot(populations),
                conditional_initial=conditional, membrane_integration=False,
                first_passage_probability_established=False, population_bound_established=False)

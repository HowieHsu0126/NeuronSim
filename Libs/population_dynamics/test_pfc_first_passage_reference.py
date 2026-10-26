"""Compare a spike-tape reconstruction against unchanged source execution."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import json
import sys
import textwrap

import numpy as np
import pytest

from . import pfc_deterministic_replay as source
from . import pfc_first_passage_reference as reference
from . import pfc_native_drive
from . import pfc_threshold_branch_probe as branch
from .test_pfc_multispike_compensator import config as source_fixture_config


@contextmanager
def source_snapshots():
    """Observe source statements without modifying its code or locals."""
    anchors = branch.observer_anchors()
    stages = {anchors[name]["line"]: name for name in ("pre_rk2", "pre_threshold")}
    lines, start = inspect.getsourcelines(source.simulate)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    for name, statements in (("pre_input", ('i["s_AMPA"][:] += drive["I"][tick % chunk_rows]',
                                               'i["s_AMPA"][:] += drive["I"][tick - block_start]')),
                             ("tick_start", ("t = tick * dt",))):
        targets = {ast.dump(ast.parse(statement).body[0], include_attributes=False)
                   for statement in statements}
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)
                   and ast.dump(node, include_attributes=False) in targets]
        assert len(matches) == 1
        stages[start + matches[0].lineno - 1] = name
    snapshots = {}
    previous = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code is not source.simulate.__code__:
            return None
        if event == "line" and frame.f_lineno in stages:
            values = frame.f_locals
            snapshots[(stages[frame.f_lineno], values["tick"])] = {
                population + "__" + key: value.copy()
                for population, local in (("E", "e"), ("I", "i"))
                for key, value in values[local].items()}
        return trace

    try:
        sys.settrace(trace)
        yield snapshots
    finally:
        sys.settrace(previous)


@pytest.fixture(params=[.05, .025, .0125], ids=["coarse", "middle", "fine"])
def observed_source(request):
    config = source_fixture_config()
    config["dt_ms"] = request.param
    case = dict(case_id="synaptic_reference_fixture", stimulus_strength_nA=.5, distractor_strength_nA=.25)
    initial = np.linspace(-59., -51., 40, dtype=np.float64)
    ticks = round(config["duration_ms"] / config["dt_ms"])
    counts_e = np.zeros((ticks, 32), dtype=np.uint16)
    counts_e[:round(.5 / config["dt_ms"])] = 40
    counts_i = np.zeros((ticks, 8), dtype=np.uint16)
    counts_i[round(.5 / config["dt_ms"]):] = np.array([40, 5, 10, 15, 20, 25, 30, 35], dtype=np.uint16)
    plain = source.simulate(config, case, initial, counts_e, counts_i)
    with source_snapshots() as snapshots:
        traced = source.simulate(config, case, initial, counts_e, counts_i)
    for family in ("raw", "terminal_state"):
        for key in plain[family]:
            np.testing.assert_array_equal(traced[family][key], plain[family][key])
    ids = plain["raw"]["spike_unit_indices"]
    assert np.any(ids < 32) and np.any(ids >= 32)
    assert any(np.unique(row["I__s_GABA"]).size > 1
               for (stage, _), row in snapshots.items() if stage == "pre_input")
    return config, case, counts_e, counts_i, plain, snapshots


def test_every_source_tick_matches_nonmembrane_reconstruction(observed_source):
    config, case, counts_e, counts_i, plain, expected = observed_source
    tape = reference.validate_tape(config, plain["raw"])
    actual = {}
    reconstructed = reference.reconstruct(config, case, tape, stop_tick=tape.total_ticks,
        counts_E=counts_e, counts_I=counts_i,
        on_snapshot=lambda stage, tick, values: actual.__setitem__((stage, tick), values))
    compared = 0
    assert len(actual) == 3 * tape.total_ticks
    for key, values in actual.items():
        for field, value in values.items():
            assert value.dtype == expected[key][field].dtype
            assert value.tobytes() == expected[key][field].tobytes(), (key, field)
            compared += value.size
    for key, value in reconstructed["terminal_state"].items():
        assert value.tobytes() == plain["terminal_state"][key].tobytes(), key
    assert reconstructed["source_events_consumed"] == len(tape.events)
    assert not reconstructed["membrane_integration"]
    # Outside-cue distractor current persists after distractor offset while cue remains on.
    control = source._control(config, case, 32, source._Arithmetic("numpy"))
    outside = np.setdiff1d(control["distractor"]["indices"], control["stimulus"]["indices"])
    assert outside.size
    later = round(16. / config["dt_ms"])
    assert np.all(actual[("pre_rk2", later)]["E__I_stim"][outside] == .25 * 1e-9)
    print(json.dumps(dict(dt_ms=config["dt_ms"], source_ticks=tape.total_ticks,
                         bitwise_state_values_compared=compared, all_stages_equal=True)))


@pytest.mark.parametrize("dt_ms", [.05, .025], ids=["coarse", "middle"])
def test_first_spike_reset_record_excludes_fresh_input(dt_ms):
    # One focal source-supported pulse, no cue and no other external inputs.
    # The initial voltage remains strictly below the source firing threshold.
    config = source_fixture_config()
    config.update(dt_ms=dt_ms, duration_ms=5.)
    case = dict(case_id="single_spike_reset_fixture", stimulus_strength_nA=0., distractor_strength_nA=0.)
    initial = np.full(40, -60., dtype=np.float64)
    initial[0] = -50.000001
    ticks = round(config["duration_ms"] / dt_ms)
    counts_e = np.zeros((ticks, 32), dtype=np.uint16)
    counts_e[0, 0], counts_e[1, 0] = 20, 3
    counts_i = np.zeros((ticks, 8), dtype=np.uint16)
    with source_snapshots() as snapshots:
        plain = source.simulate(config, case, initial, counts_e, counts_i)
    tape = reference.validate_tape(config, plain["raw"])
    stop = round(5. / config["dt_ms"])
    selection = reference.select_first_spike_window(config, tape, eligible_units=[0],
                                                    phase_start_tick=0, phase_stop_tick=stop)
    assert selection.first_spike_tick == 1
    assert tape.unit_counts[0] == 1 and counts_e[selection.first_spike_tick, 0] > 0
    assert not snapshots[("pre_rk2", selection.first_spike_tick)]["E__I_stim"][0]
    assert selection.first_spike_tick + selection.refractory_ticks < selection.window_stop_tick
    reference.require_observed_reference(tape, selection)
    first = reference.reconstruct(config, case, tape, stop_tick=stop, counts_E=counts_e,
                                  counts_I=counts_i, selection=selection)["conditional_initial"]
    changed = counts_e.copy()
    changed[selection.first_spike_tick:, selection.unit_index] = 999
    second = reference.reconstruct(config, case, tape, stop_tick=stop, counts_E=changed,
                                   counts_I=counts_i, selection=selection)["conditional_initial"]
    assert first == second
    tick, unit = selection.first_spike_tick, selection.unit_index
    before = snapshots[("pre_input", tick)]
    after = snapshots[("tick_start", tick + 1)]
    state = first["state"]
    assert state["s_AMPA"] == before["E__s_AMPA"][unit]
    assert state["v"] == after["E__v"][unit] == source._constants(config, "E")["reset"]
    assert state["lastspike"] == after["E__lastspike"][unit] == tick * tape.dt_seconds
    assert state["not_refractory"] is False
    for key in ("s_GABA", "s_NMDA", "x", "I_stim"):
        assert state[key] == after["E__" + key][unit], key
    assert after["E__s_AMPA"][unit] == state["s_AMPA"] + float(counts_e[tick, unit])
    assert not first["fresh_input_included"]


def test_actual_multiple_spike_fixture_is_not_silently_reused(observed_source):
    config, _, _, _, plain, _ = observed_source
    tape = reference.validate_tape(config, plain["raw"])
    selected = reference.select_first_spike_window(config, tape, eligible_units=np.arange(32),
        phase_start_tick=0, phase_stop_tick=round(5. / config["dt_ms"]))
    if config["dt_ms"] < .05:
        with pytest.raises(ValueError, match="SUPPRESSED_NETWORK_REPLAY_REQUIRED"):
            reference.require_observed_reference(tape, selected)
    else:
        reference.require_observed_reference(tape, selected)


def test_native_drive_prefix_matches_recorded_inputs():
    config = source_fixture_config()
    config.update(random_stream_policy="source_global", source_rng_seed=101)
    case = dict(case_id="native_input_fixture", source_rng_seed=101)
    initial, recorded_drive = pfc_native_drive.create_drive(config, case)
    recorded = recorded_drive.take(recorded_drive.total_ticks)
    plain = source.simulate(config, case, initial, recorded["E"], recorded["I"])
    tape = reference.validate_tape(config, plain["raw"])
    _, fresh = pfc_native_drive.create_drive(config, case)
    stop = 137
    actual = reference.reconstruct(config, case, tape, stop_tick=stop, drive=fresh)
    expected = reference.reconstruct(config, case, tape, stop_tick=stop,
                                     counts_E=recorded["E"], counts_I=recorded["I"])
    assert fresh.ticks == stop
    for key in actual["terminal_state"]:
        np.testing.assert_array_equal(actual["terminal_state"][key], expected["terminal_state"][key])


def test_borrowed_state_view_matches_source_and_rejects_mutation(observed_source):
    config, case, counts_e, counts_i, plain, expected = observed_source
    tape = reference.validate_tape(config, plain["raw"])
    compared = 0

    def observe(stage, tick, values):
        nonlocal compared
        for field, value in values.items():
            assert not value.flags.writeable
            assert value.tobytes() == expected[(stage, tick)][field].tobytes()
            with pytest.raises(ValueError, match="read-only"):
                value[0] = 0
            compared += value.size

    actual = reference.reconstruct(config, case, tape, stop_tick=tape.total_ticks,
        counts_E=counts_e, counts_I=counts_i, on_state_view=observe)
    assert compared > 0
    for key, value in actual["terminal_state"].items():
        assert value.tobytes() == plain["terminal_state"][key].tobytes()


def test_state_callbacks_are_exclusive():
    config = source_fixture_config()
    tape = reference.validate_tape(config, make_raw([2, 45], [0, 1]))
    with pytest.raises(ValueError, match="ONE_STATE_CALLBACK_REQUIRED"):
        reference.reconstruct(config, {}, tape, stop_tick=100,
            on_snapshot=lambda *args: None, on_state_view=lambda *args: None)


def make_raw(ticks, units):
    return dict(spike_ticks=np.asarray(ticks, dtype=np.int64),
                spike_unit_indices=np.asarray(units, dtype=np.int64),
                native_spike_counts=np.bincount(units, minlength=40).astype(np.int64))


def test_complete_tape_required_and_event_integrity_checked():
    config = source_fixture_config()
    raw = make_raw([2, 45], [0, 1])
    for key in raw:
        missing = dict(raw)
        del missing[key]
        with pytest.raises(ValueError, match="COMPLETE_SPIKE_TAPE"):
            reference.validate_tape(config, missing)
    missing = dict(raw, spike_ticks=raw["spike_ticks"][:1], spike_unit_indices=raw["spike_unit_indices"][:1])
    with pytest.raises(ValueError, match="MISSING_OR_EXTRA"):
        reference.validate_tape(config, missing)
    with pytest.raises(ValueError, match="DUPLICATE_UNIT_TICK"):
        reference.validate_tape(config, make_raw([2, 2], [0, 0]))
    with pytest.raises(ValueError, match="EVENT_RANGE"):
        reference.validate_tape(config, make_raw([400], [0]))
    with pytest.raises(ValueError, match="INTEGER_EVENT"):
        reference.validate_tape(config, dict(raw, spike_ticks=np.array([2., 45.])))
    with pytest.raises(ValueError, match="SPIKE_TIME_UNITS"):
        reference.validate_tape(config, dict(raw, spike_seconds=np.array([2., 45.])))


def test_selection_does_not_skip_actual_multiple_spikes():
    config = source_fixture_config()
    # Unit 0 has the earliest first spike and a later hit; unit 1 is not a fallback.
    tape = reference.validate_tape(config, make_raw([2, 3, 45], [0, 1, 0]))
    selected = reference.select_first_spike_window(config, tape, eligible_units=[0, 1],
                                                  phase_start_tick=0, phase_stop_tick=100)
    assert selected.unit_index == 0 and selected.first_spike_tick == 2
    with pytest.raises(ValueError, match="SUPPRESSED_NETWORK_REPLAY_REQUIRED"):
        reference.require_observed_reference(tape, selected)
    with pytest.raises(ValueError, match="SUPPRESSED_NETWORK_REPLAY_REQUIRED"):
        reference.reconstruct(config, {}, tape, stop_tick=100, selection=selected)
    late = reference.validate_tape(config, make_raw([60], [0]))
    with pytest.raises(ValueError, match="NO_ELIGIBLE"):
        reference.select_first_spike_window(config, late, eligible_units=[0],
                                           phase_start_tick=0, phase_stop_tick=100)


@pytest.mark.parametrize("callback", ["copy", "view", "none"])
@pytest.mark.parametrize("capture", ["sparse", "empty"])
def test_sparse_projection_preserves_source_observations_terminal_and_inputs(
        observed_source, monkeypatch, callback, capture):
    config, case, counts_e, counts_i, plain, expected = observed_source
    tape = reference.validate_tape(config, plain["raw"])
    ticks = [0, 1, 137, tape.total_ticks-2] if capture == "sparse" else []
    observed = {}
    def observe(stage, tick, values):
        observed[(stage, tick)] = {key: value.copy() for key, value in values.items()}
    projections = []
    cumulative = source._Arithmetic.cumulative_last
    def count_projection(self, values):
        projections.append(True)
        return cumulative(self, values)
    monkeypatch.setattr(source._Arithmetic, "cumulative_last", count_projection)
    arguments = {"on_snapshot": observe} if callback == "copy" else (
        {"on_state_view": observe} if callback == "view" else {})
    result = reference.reconstruct(config, case, tape, stop_tick=tape.total_ticks,
        counts_E=counts_e, counts_I=counts_i, capture_ticks=ticks, **arguments)
    assert len(projections) == (len(ticks)+1 if callback != "none" else 1)
    assert set(observed) == ({(stage, tick) for tick in ticks
                             for stage in ("pre_rk2", "pre_threshold", "pre_input")}
                            if callback != "none" else set())
    for position, values in observed.items():
        for key, value in values.items():
            assert value.dtype == expected[position][key].dtype
            assert value.tobytes() == expected[position][key].tobytes(), (position, key)
    for key, value in result["terminal_state"].items():
        assert value.dtype == plain["terminal_state"][key].dtype
        assert value.tobytes() == plain["terminal_state"][key].tobytes(), key
    assert result["source_events_consumed"] == len(tape.events)

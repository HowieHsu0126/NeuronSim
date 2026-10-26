"""Software tests only: tiny fixtures do not qualify source neuroscience."""
from __future__ import annotations

from itertools import cycle

import numpy as np
import pytest

from . import pfc_coupled_drive as adapter_module
from .pfc_clock_coupling import CoupledClockRows
from .pfc_coupled_drive import CoupledFineDrive, create_coupled_drive
from .pfc_deterministic_replay import simulate
from .pfc_native_drive import _state_arrays, create_drive
from .pfc_source_execution import validate_result

COARSE_DT_MS = 0.05
REFINEMENT_SEEDS = (101, 102)


def _config(factor=2, duration_ms=3.2):
    return dict(excitatory_neurons=32, inhibitory_neurons=8,
                dt_ms=COARSE_DT_MS / factor, duration_ms=duration_ms,
                random_stream_policy="source_global", source_rng_seed=71,
                synapse_execution="factorized_uniform_projections",
                weight_scaling_factor=64.0,
                stimulus_start_ms=1.0, stimulus_duration_ms=5.0,
                stimulus_center_deg=180.0, stimulus_width_deg=40.0,
                distractor_start_ms=2.0, distractor_duration_ms=2.0,
                distractor_center_deg=90.0, distractor_width_deg=40.0)


def _case():
    return dict(case_id="software_coupled_drive", stimulus_strength_nA=4.0,
                distractor_strength_nA=1.1)


def _equal_arrays(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        assert actual[name].tobytes() == expected[name].tobytes(), name


def _chunks(drive, sizes=(1, 7, 2, 11, 3, 5)):
    pieces = {"E": [], "I": []}
    for requested in cycle(sizes):
        if not drive.remaining:
            break
        block = drive.take(min(requested, drive.remaining))
        for population in pieces:
            pieces[population].append(block[population])
    return {population: np.concatenate(values) for population, values in pieces.items()}


def _row_reference(config, case):
    initial, coarse = create_drive(dict(config, dt_ms=COARSE_DT_MS), case)
    rows = CoupledClockRows(coarse, REFINEMENT_SEEDS)
    factor = int(round(COARSE_DT_MS / config["dt_ms"]))
    packed = np.concatenate([rows.take_row()[1 if factor == 2 else 2]
                             for _ in range(coarse.total_ticks)])
    return initial, {"I": packed[:, :coarse.n_i], "E": packed[:, coarse.n_i:]}, rows


@pytest.mark.parametrize("factor", [2, 4])
def test_full_and_arbitrary_chunks_match_materialized_coupling_rows(factor):
    config, case = _config(factor), _case()
    saved_config, saved_case = dict(config), dict(case)
    initial, expected, reference = _row_reference(config, case)
    for sizes in ((1,), (1, 7, 2, 11, 3, 5), (100000,)):
        actual_initial, drive = create_coupled_drive(config, case, COARSE_DT_MS, REFINEMENT_SEEDS)
        np.testing.assert_array_equal(actual_initial, initial)
        actual = _chunks(drive, sizes)
        _equal_arrays(actual, expected)
        _equal_arrays(drive.final_state_arrays(), reference.state_arrays())
        assert drive.ticks == drive.total_ticks
        assert drive.remaining == 0
        assert drive.event_totals == {p: int(values.sum(dtype=np.uint64)) for p, values in expected.items()}
        metadata = drive.metadata()
        assert metadata["fine_source_seed_sequence_preserved"] is False
        assert metadata["coarse_source_seed_sequence_preserved"] is True
        assert metadata["buffered_fine_rows"] == 0
    assert config == saved_config
    assert case == saved_case


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("policy", ["source_global", "independent_initial_external"])
def test_initial_values_and_final_coarse_rng_match_native(factor, policy):
    config, case = _config(factor), _case()
    if policy == "independent_initial_external":
        config.pop("source_rng_seed")
        config.update(random_stream_policy=policy, initial_rng_seed=71, external_noise_seed=72)
    native_initial, native = create_drive(dict(config, dt_ms=COARSE_DT_MS), case)
    initial, drive = create_coupled_drive(config, case, COARSE_DT_MS, REFINEMENT_SEEDS)
    np.testing.assert_array_equal(initial, native_initial)
    native.take(native.total_ticks)
    _chunks(drive)
    state = drive.final_state_arrays()
    coarse_state = {key.removeprefix("coarse__"): value
                    for key, value in state.items() if key.startswith("coarse__")}
    _equal_arrays(coarse_state, _state_arrays(native.get_state()))


@pytest.mark.parametrize("factor", [2, 4])
def test_returned_arrays_have_independent_lifetime_and_cannot_mutate_cache(factor):
    config, case = _config(factor), _case()
    _, expected, _ = _row_reference(config, case)
    _, drive = create_coupled_drive(config, case, COARSE_DT_MS, REFINEMENT_SEEDS)
    first = drive.take(1)
    saved = {key: values.copy() for key, values in first.items()}
    second = drive.take(1)
    _equal_arrays(first, saved)
    for population in first:
        assert not np.shares_memory(first[population], second[population])
        first[population][:] = 65535
    remaining = _chunks(drive)
    for population in expected:
        np.testing.assert_array_equal(second[population], expected[population][1:2])
        np.testing.assert_array_equal(remaining[population], expected[population][2:])


@pytest.mark.parametrize("prefetch", [False, True])
def test_invalid_take_requests_do_not_change_rng_counters_or_buffer(prefetch):
    # Long nominal duration tests overflow rejection without generating it.
    _, drive = create_coupled_drive(_config(4, 200000.0), _case(), COARSE_DT_MS, REFINEMENT_SEEDS)
    if prefetch:
        drive.take(1)
    state = drive._rows.state_arrays()
    metadata = drive.metadata()
    buffer = None if drive._buffer is None else drive._buffer.copy()
    invalid = [0, -1, True, np.bool_(False), 1.5, "1", None,
               drive.remaining + 1, drive.max_rows + 1, 2**100,
               np.uint64(np.iinfo(np.uint64).max)]
    assert drive.max_rows < drive.remaining
    for request in invalid:
        with pytest.raises(ValueError):
            drive.take(request)
        _equal_arrays(drive._rows.state_arrays(), state)
        assert drive.metadata() == metadata
        if buffer is None:
            assert drive._buffer is None
        else:
            np.testing.assert_array_equal(drive._buffer, buffer)


def test_source_cursor_reuse_is_rejected_before_new_draws():
    initial, coarse = create_drive(dict(_config(), dt_ms=COARSE_DT_MS), _case())
    rows = CoupledClockRows(coarse, REFINEMENT_SEEDS)
    first = CoupledFineDrive(rows, 0.025 * 1e-3, 2)
    second = CoupledFineDrive(rows, 0.025 * 1e-3, 2)
    first.take(1)
    state = rows.state_arrays()
    with pytest.raises(ValueError, match="CURSOR_CHANGED_OR_REUSED"):
        second.take(1)
    with pytest.raises(ValueError, match="FRESH_COARSE_STREAM"):
        CoupledFineDrive(rows, 0.025 * 1e-3, 2)
    _equal_arrays(rows.state_arrays(), state)
    assert initial.shape == (40,)


@pytest.mark.parametrize("factor", [2, 4])
def test_final_state_requires_all_children_and_rejects_exhausted_take(factor):
    _, drive = create_coupled_drive(_config(factor, 0.05), _case(), COARSE_DT_MS, REFINEMENT_SEEDS)
    with pytest.raises(ValueError, match="COMPLETE_CONSUMPTION"):
        drive.final_state_arrays()
    drive.take(factor - 1)
    assert drive._rows.coarse.remaining == 0
    with pytest.raises(ValueError, match="COMPLETE_CONSUMPTION"):
        drive.final_state_arrays()
    drive.take(1)
    final = drive.final_state_arrays()
    with pytest.raises(ValueError, match="REMAINING_TICKS"):
        drive.take(1)
    _equal_arrays(drive.final_state_arrays(), final)
    final["coarse__key"][:] = 0
    assert np.any(drive.final_state_arrays()["coarse__key"] != 0)


def test_invalid_factory_arguments_reject_before_source_rng_draws(monkeypatch):
    def forbidden_source_draw(*args, **kwargs):
        raise AssertionError("invalid request reached the initial source draws")
    monkeypatch.setattr(adapter_module, "create_drive", forbidden_source_draw)
    config, case = _config(), _case()
    requests = [
        (dict(config, dt_ms=0.02), COARSE_DT_MS, REFINEMENT_SEEDS),
        (dict(config, dt_ms=COARSE_DT_MS), COARSE_DT_MS, REFINEMENT_SEEDS),
        (dict(config, dt_ms=True), COARSE_DT_MS, REFINEMENT_SEEDS),
        (dict(config, duration_ms=0.075), COARSE_DT_MS, REFINEMENT_SEEDS),
        (config, float("nan"), REFINEMENT_SEEDS),
        (config, COARSE_DT_MS, (101, 101)),
        (config, COARSE_DT_MS, (71, 101)),
        (config, COARSE_DT_MS, (101,)),
        (config, COARSE_DT_MS, (101, True)),
        (dict(config, excitatory_neurons=10000000), COARSE_DT_MS, REFINEMENT_SEEDS),
    ]
    for candidate, coarse_dt, seeds in requests:
        with pytest.raises(ValueError):
            create_coupled_drive(candidate, case, coarse_dt, seeds)


@pytest.mark.parametrize("factor", [2, 4])
def test_streamed_simulation_matches_identical_materialized_inputs(factor):
    config, case = _config(factor, 8.0), _case()
    initial, materialized_drive = create_coupled_drive(config, case, COARSE_DT_MS, REFINEMENT_SEEDS)
    counts = materialized_drive.take(materialized_drive.total_ticks)
    recorded = simulate(config, case, initial, counts["E"], counts["I"],
                        backend="numpy", observe_applied_current=True)
    streamed_initial, streamed_drive = create_coupled_drive(config, case, COARSE_DT_MS, REFINEMENT_SEEDS)
    streamed = simulate(config, case, streamed_initial, backend="numpy",
                        drive=streamed_drive, observe_applied_current=True)
    for family in ("raw", "terminal_state"):
        _equal_arrays(streamed[family], recorded[family])
    _equal_arrays(streamed["applied_current_observation"]["arrays"],
                  recorded["applied_current_observation"]["arrays"])
    assert (streamed["applied_current_observation"]["metadata"]
            == recorded["applied_current_observation"]["metadata"])
    assert streamed["raw"]["spike_ticks"].size > 0
    assert streamed["applied_current_observation"]["arrays"]["current_A"].shape == (5, 32)
    for result, drive in ((recorded, materialized_drive), (streamed, streamed_drive)):
        checks = validate_result(config, result, drive)
        assert all(row["passed"] is True for row in checks.values()), checks
    _equal_arrays(streamed_drive.final_state_arrays(), materialized_drive.final_state_arrays())

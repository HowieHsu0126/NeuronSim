"""Small real-source checks and descriptive counterfactual bookkeeping tests."""
from __future__ import annotations

from copy import deepcopy
import sys

import numpy as np
import pytest

from . import pfc_spike_branch_continuation as continuation
from . import pfc_threshold_branch_probe as branch
from . import pfc_deterministic_replay as executor
from .pfc_coupled_drive import create_coupled_drive


def _config():
    return dict(excitatory_neurons=32, inhibitory_neurons=8, dt_ms=.0125,
                duration_ms=5000., random_stream_policy="source_global", source_rng_seed=71,
                synapse_execution="factorized_uniform_projections", weight_scaling_factor=64.,
                stimulus_start_ms=100., stimulus_duration_ms=200., stimulus_center_deg=180.,
                stimulus_width_deg=40., distractor_start_ms=400., distractor_duration_ms=50.,
                distractor_center_deg=90., distractor_width_deg=40., excitatory_preference_bins=16)


def _run(observer=None, duration=3.):
    full_config = _config()
    case = dict(case_id="software_imposed_event", stimulus_strength_nA=0., distractor_strength_nA=0.)
    initial, drive = create_coupled_drive(full_config, case, .05, [101, 102])
    config = dict(full_config, duration_ms=duration)
    if observer is None:
        result = executor.simulate(config, case, initial, backend="numpy", drive=drive)
    else:
        with observer.observing():
            result = executor.simulate(config, case, initial, backend="numpy", drive=drive)
    return result, drive


def _reference():
    observer = branch.ThresholdObserver(0, _config())
    _run(observer, .05)
    return observer.snapshots["pre_threshold"]


def _equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert continuation._bitwise_equal(left[key], right[key]), key


def _events(rows):
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def test_unchanged_real_source_trace_on_off_exact_when_intervention_disabled():
    before = continuation._bindings()
    plain, original_drive = _run()
    observer = continuation.SpikeBranchObserver(0, 0, _config(), intervene=False)
    saved = sys.gettrace()
    observed, drive = _run(observer)
    assert sys.gettrace() is saved
    for family in ("raw", "terminal_state"):
        _equal(plain[family], observed[family])
    _equal(original_drive._rows.state_arrays(), drive._rows.state_arrays())
    assert observer.applied == 0
    assert drive.total_ticks == 400000 and drive.ticks == 240
    assert continuation._bindings() == before


def test_one_imposed_event_uses_real_source_reset_refractory_and_delayed_nmda():
    reference = _reference()
    observer = continuation.SpikeBranchObserver(0, 0, _config(), reference)
    result, _ = _run(observer)
    raw = result["raw"]
    np.testing.assert_array_equal(raw["spike_unit_indices"][raw["spike_ticks"] == 0], [0])
    assert observer.applied == 1
    assert all(observer.receipt["preintervention_bitwise_match"].values())
    assert observer.receipt["native_threshold_events_at_trigger"] == []
    assert observer.receipt["changed_state_locations"] == ["E__v[0]"]
    state = observer.followup["1_pre_rk2"]
    assert state["v"] == -.060 and state["x"] == 1.
    assert state["lastspike"] == 0. and state["not_refractory"] is False
    assert state["s_NMDA"] == 0.
    assert all(np.count_nonzero(x) == 0 for x in observer.recurrence[1].values())
    assert all(np.any(x != 0) for x in observer.recurrence[2].values())
    assert observer.followup["160_pre_threshold"]["not_refractory"] is True
    assert not np.any((raw["spike_unit_indices"] == 0) & (raw["spike_ticks"] > 0) & (raw["spike_ticks"] < 160))
    assert continuation._bitwise_equal(reference["E__v"], observer.snapshots["pre_threshold"]["E__v"])


def test_any_reference_state_mismatch_refuses_before_mutation_and_restores_trace():
    reference = _reference()
    reference["I__s_GABA"][0] = 1.
    observer = continuation.SpikeBranchObserver(0, 0, _config(), reference)
    saved = sys.gettrace()
    with pytest.raises(ValueError, match="PREINTERVENTION_STATE_MISMATCH"):
        _run(observer)
    assert sys.gettrace() is saved and observer.applied == 0
    assert observer.receipt["preintervention_bitwise_match"]["I__s_GABA"] is False


def test_trace_restores_prior_callback_on_injected_failure(monkeypatch):
    observer = continuation.SpikeBranchObserver(0, 0, _config(), intervene=False)
    def fail(local):
        raise RuntimeError("observer injection")
    def previous(frame, event, arg):
        return None
    monkeypatch.setattr(observer, "_trigger", fail)
    saved = sys.gettrace()
    try:
        sys.settrace(previous)
        with pytest.raises(RuntimeError, match="observer injection"):
            _run(observer)
        assert sys.gettrace() is previous
    finally:
        sys.settrace(saved)


def test_absolute_stop_excludes_boundary_and_preserves_last_preceding_tick():
    config = _config()
    events = _events([(7999, 0), (8000, 1), (20000, 2)])
    actual = continuation.clip_events(dict(spike_ticks=events[:, 0], spike_unit_indices=events[:, 1]), config, 8000)
    np.testing.assert_array_equal(actual, [[7999, 0]])
    assert continuation.stop_tick(100., .0125, 5000.) == 8000
    for stop in (100.0125, 0., 3.14159):
        with pytest.raises(ValueError, match="CLOCK"):
            continuation.stop_tick(stop, .0125, 5000.)


def test_non_target_mismatch_before_nmda_delay_is_rejected():
    with pytest.raises(ValueError, match="PRECEDES_NMDA_CAUSAL_DELAY"):
        continuation.event_comparison(_events([]), _events([(89, 0), (90, 1)]), 3, 89, 0, 160)
    with pytest.raises(ValueError, match="REFRACTORY_VIOLATION"):
        continuation.event_comparison(_events([]), _events([(89, 0), (248, 0)]), 3, 89, 0, 160)


def test_event_locality_and_count_locality_are_different_contracts(tmp_path):
    original, counter = _events([(20, 1)]), _events([(10, 0), (21, 1)])
    event_report, _ = continuation.event_comparison(original, counter, 3, 10, 0, 160)
    assert event_report["event_schedule_locality_observed"] is False
    assert event_report["other_units_with_event_discrepancies"] == 1
    assert event_report["earliest_new_event_difference"] == dict(tick=20, archive_unit_index=1)
    config = dict(_config(), excitatory_neurons=2, inhibitory_neurons=1, excitatory_preference_bins=2)
    settings = dict(stop_ms=100., forced_unit_index=0, windows_ms=[5., 50., 100.])
    counts = continuation.count_comparison(original, counter, config, settings, tmp_path)
    assert counts["count_locality_observed"] is True
    assert counts["full_100ms_count_delta_signed"] == counts["full_100ms_count_delta_absolute"] == 1
    with np.load(tmp_path / "descriptive_count_cdfs.npz") as cdf:
        assert cdf["count_support"].tolist() == [0, 1]
        assert np.all(cdf["cdf"][..., -1] == 1.)
    with np.load(tmp_path / "unit_window_count_differences.npz") as values:
        assert values["source_field_indices"].tolist() == [0, 1, 2]
        assert np.all(values["delta_5ms"][:, 1:] == 0)


def test_zero_net_count_change_preserves_opposite_non_target_changes(tmp_path):
    original, counter = _events([(20, 1)]), _events([(10, 0), (20, 2)])
    config = dict(_config(), excitatory_neurons=2, inhibitory_neurons=1, excitatory_preference_bins=2)
    report = continuation.count_comparison(original, counter, config,
        dict(stop_ms=100., forced_unit_index=0, windows_ms=[5., 50., 100.]), tmp_path)
    first = report["windows"][0]
    assert first["non_target_signed_delta"] == 0
    assert first["non_target_absolute_count_delta"] == first["non_target_units_with_count_difference"] == 2
    assert report["count_locality_observed"] is False


def test_counts_above_source_physical_support_reject(tmp_path):
    config = dict(_config(), excitatory_neurons=2, inhibitory_neurons=1, excitatory_preference_bins=2)
    with pytest.raises(ValueError, match="COUNT_EXCEEDS_SOURCE_REFRACTORY_BOUND"):
        continuation.count_comparison(_events([]), _events([(1, 0), (2, 0), (3, 0), (4, 0)]), config,
            dict(stop_ms=100., forced_unit_index=0, windows_ms=[5., 50., 100.]), tmp_path)


def test_counterfactual_scope_does_not_release_scientific_gates():
    result = continuation.scope()
    assert result["counterfactual_event_sensitivity_diagnostic"] is True
    for key in ("legal_native_auxiliary_realization", "statistical_source_draw", "admissible_inverse_stimulus",
                "cdf_qualified", "auxiliary_enclosure_qualified", "all_scientific_gates_released"):
        assert result[key] is False
    assert result["auxiliary_event"] is result["conditional_failure_probability"] is None
    assert result["new_independent_blocks"] == result["new_full_episodes"] == result["production_episodes"] == 0


def test_frozen_config_and_source_binding_reject_changes():
    settings = dict(schema=continuation.SCHEMA, source_rng_seed=95001, refinement_seeds=[95411, 95412],
        coarse_dt_ms=.05, dt_ms=.0125, expected_units=100000, forced_unit_index=6423, forced_tick=89,
        stop_ms=100., windows_ms=[5., 50., 100.], case_relative_path="source_cases/dt_0p0125/pfc_95001_omission",
        confirmation_execution_released=False, production_episodes=0,
        accepted_probe=dict(archive="accepted", delivery_sha256=continuation.ACCEPTED_DELIVERY,
            execution_report_sha256=continuation.ACCEPTED_REPORT, snapshot_sha256=continuation.ACCEPTED_SNAPSHOT),
        original_source=dict(archive="original", delivery_sha256=continuation.ORIGINAL_DELIVERY))
    continuation.validate_settings(settings)
    for key, value in (("forced_tick", 90), ("forced_unit_index", 6424), ("stop_ms", 101.),
                       ("confirmation_execution_released", True)):
        invalid = deepcopy(settings)
        invalid[key] = value
        with pytest.raises(ValueError):
            continuation.validate_settings(invalid)
    invalid = deepcopy(settings)
    invalid["original_source"]["delivery_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ARCHIVE_BINDING"):
        continuation.validate_settings(invalid)

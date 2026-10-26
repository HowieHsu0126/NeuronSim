"""Conditional source-kernel checks; toy/source fixtures are not scientific claims."""
from fractions import Fraction

import numpy as np
import pytest

from . import pfc_conditional_spike_probability as kernel
from . import pfc_threshold_branch_probe as branch
from . import pfc_deterministic_replay as executor
from . import pfc_coupled_drive, pfc_clock_temporal_control


def test_exact_small_binomial_tails_include_always_and_never():
    rows = kernel.binomial_tail_bounds(2, 1, 4, 4)
    assert [(r["lower_scaled"], r["upper_scaled"]) for r in rows] == [
        ("10000", "10000"), ("4375", "4375"), ("625", "625"), ("0", "0")]


def test_directed_bounds_enclose_nonterminating_rational_tails():
    rows = kernel.binomial_tail_bounds(3, 1, 3, 4)
    truth = [Fraction(1), Fraction(19, 27), Fraction(7, 27), Fraction(1, 27), Fraction(0)]
    for row, expected in zip(rows, truth):
        lo, hi = int(row["lower_scaled"]), int(row["upper_scaled"])
        assert Fraction(lo, 10000) <= expected <= Fraction(hi, 10000)
        assert 0 <= hi - lo <= 1


def test_full_native_support_does_not_drop_underflow_tail():
    rows = kernel.binomial_tail_bounds(1000, *(7 / 400000).as_integer_ratio())
    assert len(rows) == 1002
    assert rows[1000]["lower_scaled"] == "0" and rows[1000]["upper_scaled"] == "1"
    assert rows[1001]["lower_scaled"] == rows[1001]["upper_scaled"] == "0"
    assert int(rows[10]["lower_scaled"]) > 0


def test_field_expectation_and_union_bounds_keep_always_and_never():
    bounds = kernel.binomial_tail_bounds(2, 1, 4, 4)
    rows, total = kernel.summarize_thresholds(np.array([0, 3, 1]), np.array([0, 0, 1]),
                                             np.array([2, 1]), bounds, digits=4)
    assert rows[0]["always_spikes"] == rows[0]["never_spikes"] == 1
    assert rows[0]["expected_spikes_lower_scaled"] == "10000"
    assert total["expected_spikes_lower_scaled"] == total["expected_spikes_upper_scaled"] == "14375"
    assert total["any_spike_union_upper_scaled"] == "10000"


@pytest.mark.parametrize("factory", [pfc_coupled_drive.create_coupled_drive,
                                      pfc_clock_temporal_control.create_temporal_reversal_drive])
def test_reconstructed_history_and_enumerated_actual_transition_match_source_bitwise(factory):
    config = dict(excitatory_neurons=8, inhibitory_neurons=4, dt_ms=.0125, duration_ms=5000.,
        source_rng_seed=71, random_stream_policy="source_global", weight_scaling_factor=256.,
        synapse_execution="factorized_uniform_projections", stimulus_start_ms=100.,
        stimulus_duration_ms=200., distractor_start_ms=400., distractor_duration_ms=50.)
    case = dict(case_id="kernel_fixture", stimulus_strength_nA=0., distractor_strength_nA=0.)
    tick, seeds = 89, [101, 102]
    initial, drive = factory(config, case, .05, seeds)
    observer = branch.ThresholdObserver(tick, config)
    with observer.observing():
        executor.simulate(dict(config, duration_ms=(tick + 1) * config["dt_ms"]), case, initial, drive=drive)
    again, base, actual, combined, p, metadata = kernel.reconstruct_ampa(config, case, factory, .05, seeds, tick)
    assert kernel.exact_arrays_equal(initial, again)
    assert metadata["ticks_generated"] == tick and metadata["total_ticks"] == 400000
    assert p == 7 / 400000
    pre, post = observer.snapshots["pre_rk2"], observer.snapshots["pre_threshold"]
    for population in ("E", "I"):
        assert kernel.exact_arrays_equal(combined[population], pre[population + "__s_AMPA"])
    mapping, agreements = kernel.threshold_map(pre, post, base, actual, config, tick)
    assert all(agreements.values())
    for population in ("E", "I"):
        decision = (post[population + "__v"] > -.05) & post[population + "__not_refractory"]
        np.testing.assert_array_equal(actual[population] >= mapping[population], decision)


@pytest.mark.parametrize("args", [(-1, 1, 2), (2, 0, 3), (2, 3, 3), (True, 1, 2)])
def test_invalid_probability_parameters_rejected(args):
    with pytest.raises(ValueError, match="BINOMIAL_PARAMETERS"):
        kernel.binomial_tail_bounds(*args)

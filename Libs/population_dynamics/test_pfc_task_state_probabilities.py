"""Exact state uncertainty retains complete paired histories and rare outcomes."""
import pytest
from scipy.stats import binom

from .pfc_task_state_probabilities import (
    STATES, exact_interval, paired_state_effect, state_for_episode,
)


def row(count=10, angle=0., defined=True):
    return dict(excitatory_spikes=str(count), silent=str(count == 0),
        numerical_direction_defined=str(defined), circular_cue_error_deg=str(angle) if defined else "",
        numerical_angle_inside_cue_support=str(abs(angle) <= 15) if defined else "")


def test_state_partition_retains_quiet_and_undefined_activity():
    assert state_for_episode(row(0, defined=False)) == STATES[0]
    assert state_for_episode(row(3, defined=False)) == STATES[1]
    assert state_for_episode(row(3, 15.)) == STATES[2]
    assert state_for_episode(row(3, -15.)) == STATES[2]
    assert state_for_episode(row(3, 15.0001)) == STATES[3]


def test_invalid_direction_and_classification_reject():
    with pytest.raises(ValueError, match="SILENT_DIRECTION_DEFINED"):
        state_for_episode(row(0))
    bad = row(); bad["numerical_angle_inside_cue_support"] = "False"
    with pytest.raises(ValueError, match="CUE_SUPPORT_CHANGED"):
        state_for_episode(bad)


def test_boundary_probability_intervals_are_not_structural_zeros():
    alpha = .05 / 9216
    lo, hi = exact_interval(0, 3, alpha)
    assert lo == 0 and hi == pytest.approx(1 - (alpha / 2) ** (1 / 3))
    lo, hi = exact_interval(3, 3, alpha)
    assert hi == 1 and lo == pytest.approx((alpha / 2) ** (1 / 3))


def test_exact_interval_has_nominal_coverage_by_enumeration():
    n, alpha = 12, .05
    for probability in (.01, .1, .5, .9, .99):
        missed = sum(binom.pmf(k, n, probability) for k in range(n + 1)
                     if not exact_interval(k, n, alpha)[0] <= probability <= exact_interval(k, n, alpha)[1])
        assert missed <= alpha + 1e-12


def test_pair_gain_loss_keeps_concordant_histories_in_denominator():
    before = [STATES[2], STATES[3], STATES[3]]
    after = [STATES[2], STATES[2], STATES[3]]
    value = paired_state_effect(before, after, STATES[2], .01)
    assert (value["n00"], value["n01"], value["n10"], value["n11"]) == (1, 1, 0, 1)
    assert value["independent_histories"] == 3
    assert value["mean_probability_change"] == pytest.approx(1 / 3)
    assert value["gain_probability_lower"] == pytest.approx(exact_interval(1, 3, .01)[0])
    assert value["probability_change_lower"] < 0 < value["probability_change_upper"]


def test_pair_reversal_changes_sign_and_interval_orientation():
    before, after = [STATES[0], STATES[2], STATES[2]], [STATES[2], STATES[0], STATES[0]]
    a = paired_state_effect(before, after, STATES[2], .05)
    b = paired_state_effect(after, before, STATES[2], .05)
    assert a["mean_probability_change"] == -b["mean_probability_change"]
    assert a["probability_change_lower"] == pytest.approx(-b["probability_change_upper"])
    assert a["probability_change_upper"] == pytest.approx(-b["probability_change_lower"])


def test_bad_binomial_design_and_incomplete_pair_fail():
    with pytest.raises(ValueError, match="BINOMIAL_DESIGN_INVALID"):
        exact_interval(4, 3, .05)
    with pytest.raises(ValueError, match="COMPLETE_PAIRED_HISTORIES_REQUIRED"):
        paired_state_effect([STATES[0]], [], STATES[0], .05)

"""Physical support, joint direction and independent-history inference tests."""
import itertools
import math

import numpy as np
import pytest

from .pfc_vector_uncertainty import (
    episode_vector, mean_band, phase_rate_bound, rectangle_direction,
)


def test_refractory_half_open_count_bound_keeps_initial_event():
    assert phase_rate_bound(0, 160) == 500.
    assert phase_rate_bound(0, 161) == pytest.approx(2 / (.0125e-3 * 161))
    with pytest.raises(ValueError, match="PHASE_INVALID"):
        phase_rate_bound(100, 100)


def test_silence_cancellation_and_small_resultant_are_distinct():
    row = dict(start_tick=0, stop_tick=8000, excitatory_spikes=0,
               vector_x_spikes=0., vector_y_spikes=0.)
    silent = episode_vector(row, 2048, 120.)
    assert silent["silent"] and not silent["numerical_direction_defined"]
    assert silent["resultant_length"] is None
    row["excitatory_spikes"] = 10
    cancelled = episode_vector(row, 2048, 120.)
    assert not cancelled["silent"] and not cancelled["numerical_direction_defined"]
    assert cancelled["resultant_length"] == 0
    row["vector_x_spikes"] = 1e-8
    weak = episode_vector(row, 2048, 120.)
    assert weak["numerical_direction_defined"] and weak["resultant_length"] == 1e-9


def test_vector_physical_support_rejects_impossible_moments():
    row = dict(start_tick=0, stop_tick=8000, excitatory_spikes=1,
               vector_x_spikes=2., vector_y_spikes=0.)
    with pytest.raises(ValueError, match="EVENT_SUPPORT_INVALID"):
        episode_vector(row, 2048, 120.)
    row.update(excitatory_spikes=0, vector_x_spikes=1e-16)
    with pytest.raises(ValueError, match="SILENT_NONZERO_MOMENT"):
        episode_vector(row, 2048, 120.)


def test_projection_matches_cue_aligned_count_and_keeps_opposite_sign():
    cue = math.radians(120.)
    row = dict(start_tick=0, stop_tick=8000, excitatory_spikes=100,
               vector_x_spikes=100 * math.cos(cue), vector_y_spikes=100 * math.sin(cue))
    value = episode_vector(row, 2048, 120.)
    assert value["target_activity_Hz"] == pytest.approx(100 / (2048 * .1))
    assert value["orthogonal_activity_Hz"] == pytest.approx(0., abs=1e-14)
    row.update(vector_x_spikes=-row["vector_x_spikes"], vector_y_spikes=-row["vector_y_spikes"])
    assert episode_vector(row, 2048, 120.)["target_activity_Hz"] < 0


def test_constant_samples_do_not_prove_structural_zero():
    point, lower, upper = mean_band(np.zeros(3), -500., 500., .05, 1280)
    assert point == 0 and lower == -500 and upper == 500


def test_direction_requires_whole_region_within_geometric_target():
    assert rectangle_direction(2, 3, -.1, .1, 15) == dict(
        confidence_region_excludes_zero=True, confidence_region_inside_cue_support=True)
    wide = rectangle_direction(2, 3, -10, 10, 15)
    assert wide["confidence_region_excludes_zero"] and not wide["confidence_region_inside_cue_support"]
    cancelled = rectangle_direction(-1, 1, -1, 1, 15)
    assert not any(cancelled.values())
    opposite = rectangle_direction(-3, -2, -.1, .1, 15)
    assert opposite["confidence_region_excludes_zero"] and not opposite["confidence_region_inside_cue_support"]


def test_mean_band_exact_enumeration_covers_independent_bernoulli_law():
    # Exhaustive small design: no random experiment seed and no bootstrap.
    alpha, n = .05, 12
    for probability in (.05, .5, .95):
        miss = 0.
        for successes in range(n + 1):
            samples = [1.] * successes + [0.] * (n - successes)
            _, lower, upper = mean_band(samples, 0., 1., alpha, 1)
            if not lower <= probability <= upper:
                miss += math.comb(n, successes) * probability ** successes * (1 - probability) ** (n - successes)
        assert miss <= alpha


def test_direction_rectangle_corners_respect_circular_sector():
    for lower, upper, orthogonal in ((1., 2., .1), (3., 4., .7)):
        result = rectangle_direction(lower, upper, -orthogonal, orthogonal, 15)
        assert result["confidence_region_inside_cue_support"]
        for target, cross in itertools.product((lower, upper), (-orthogonal, orthogonal)):
            assert abs(math.degrees(math.atan2(cross, target))) <= 15


def test_invalid_range_and_history_shape_fail():
    with pytest.raises(ValueError, match="PHYSICAL_RANGE_VIOLATION"):
        mean_band([0., 2.], -1., 1., .05, 1280)
    with pytest.raises(ValueError, match="HISTORY_VALUES_REQUIRED"):
        mean_band([[0., 1.], [0., 1.]], -1., 1., .05, 1280)

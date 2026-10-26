"""Check source-defined boundaries and diagnose, without weakening raw equality."""
import numpy as np
import pytest

from .v1_nest_time_semantics import clock_tics_per_step, encode_precise_times, source_refractory_steps


def test_two_observed_cancellation_values_are_predicted_exactly():
    source = np.array([.000021, .052912])
    before = source.copy()
    result = encode_precise_times(source, 1000.)
    assert result['decoded_times_ms'].tolist() == [2.0999999999993246e-05, .052912000000000015]
    assert result['steps'].tolist() == [1, 1]
    assert np.all(source != result['decoded_times_ms'])
    np.testing.assert_array_equal(source, before)


def test_native_near_grid_branch_is_not_replaced_by_generic_ceil():
    below, above = np.nextafter(.25, 0), np.nextafter(.25, np.inf)
    source = np.array([below, .25, above, 1.1, 1.1, 2999.991119])
    result = encode_precise_times(source, 1000.)
    assert result['steps'].tolist() == [1, 1, 2, 5, 5, 12000]
    assert result['offsets_ms'][0] == result['offsets_ms'][1] == 0
    assert result['decoded_times_ms'][0] == .25
    np.testing.assert_array_equal(result['decoded_times_ms'][1:], source[1:])


@pytest.mark.parametrize('invalid', [[0.], [-1.], [np.nan], [np.inf], [3000.1], [[1.]]])
def test_invalid_source_domain_is_rejected(invalid):
    with pytest.raises(ValueError, match='DOMAIN_INVALID'):
        encode_precise_times(np.asarray(invalid), 1000.)


def test_clock_cannot_silently_truncate_fractional_tics():
    with pytest.raises(ValueError, match='NOT_REPRESENTABLE'):
        encode_precise_times(np.array([1.]), 1001.)


def test_half_source_step_preserves_near_grid_duplicate_and_empty_inputs():
    source = np.array([np.nextafter(.125, 0), .125, np.nextafter(.125, np.inf), 1.1, 1.1, 3000.])
    before = source.copy()
    result = encode_precise_times(source, 1000., .125)
    assert clock_tics_per_step(1000., .125) == 125
    assert result['steps'].tolist() == [1, 1, 2, 9, 9, 24000]
    assert result['offsets_ms'][:2].tolist() == [0., 0.]
    assert result['decoded_times_ms'][0] == .125
    np.testing.assert_array_equal(result['decoded_times_ms'][1:], source[1:])
    np.testing.assert_array_equal(source, before)
    assert all(value.size == 0 for value in encode_precise_times(np.empty(0), 1000., .125).values())
    with pytest.raises(ValueError, match='NOT_REPRESENTABLE'):
        encode_precise_times(source, 1001., .125)


@pytest.mark.parametrize('resolution', [0., -.125, np.nan, np.inf])
def test_invalid_execution_resolution_is_rejected(resolution):
    with pytest.raises(ValueError, match='DOMAIN_INVALID'):
        encode_precise_times(np.array([1.]), 1000., resolution)


def test_refractory_counts_follow_tic_rounding_then_step_ceiling():
    source = np.array([0., .125, .12549, .1255, .25, 1.55, 3.5])
    before = source.copy()
    assert source_refractory_steps(source, 1000.).tolist() == [0, 1, 1, 1, 1, 7, 14]
    assert source_refractory_steps(source, 1000., .125).tolist() == [0, 1, 1, 2, 2, 13, 28]
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize('invalid', [[-.1], [np.nan], [np.inf], [[1.]]])
def test_refractory_derivation_rejects_invalid_physical_durations(invalid):
    with pytest.raises(ValueError, match='REFRACTORY_DOMAIN_INVALID'):
        source_refractory_steps(np.asarray(invalid), 1000., .125)

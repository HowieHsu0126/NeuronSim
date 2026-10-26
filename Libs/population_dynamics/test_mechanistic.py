import numpy as np
import pytest

from Libs.population_dynamics.mechanistic import current_schedule


def test_current_overlap_is_additive_and_ends_at_declared_boundaries():
    pulses = [
        dict(population='L4e', onset_ms=2., duration_ms=3., amplitude_pa=100.),
        dict(population='L4e', onset_ms=4., duration_ms=3., amplitude_pa=-20.),
        dict(population='L6e', onset_ms=1., duration_ms=8., amplitude_pa=1000.),
    ]
    times, values = current_schedule(pulses, 'L4e', 10., 10.)
    np.testing.assert_array_equal(times, [0, 2, 4, 5, 7, 10])
    np.testing.assert_array_equal(values, [10, 110, 90, -10, 10, 10])
    assert np.sum((values[:-1] - 10.) * np.diff(times)) == 240.


@pytest.mark.parametrize('override', [
    {'onset_ms': -1.}, {'duration_ms': 0.}, {'onset_ms': 9., 'duration_ms': 2.},
    {'amplitude_pa': float('nan')}, {'amplitude_pa': float('inf')}, {'population': 'unknown'},
])
def test_invalid_current_cannot_reach_the_simulator(override):
    pulse = dict(population='L4e', onset_ms=1., duration_ms=2., amplitude_pa=50.)
    pulse.update(override)
    with pytest.raises(ValueError):
        current_schedule([pulse], 'L4e', 0., 10.)

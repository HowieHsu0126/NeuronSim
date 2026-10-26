import math
from pathlib import Path

import pytest

from Libs.population_dynamics.mechanistic import run_reference
from Libs.population_dynamics.synaptic_reference import analytic_after_refractory


def test_release_preserves_decayed_input_from_refractory_period():
    baseline, voltage = analytic_after_refractory(2.0, 0.0)
    stimulated, _ = analytic_after_refractory(2.0, 200.0)
    assert baseline == pytest.approx(1000.0 * math.exp(-4))
    assert stimulated - baseline == pytest.approx(200.0 * math.exp(-2))
    assert voltage == -65.0


def test_analytic_solution_satisfies_independent_lif_differential_equation():
    time, epsilon = 2.5, 1e-5
    current, voltage = analytic_after_refractory(time, 200.0)
    before_i, before_v = analytic_after_refractory(time - epsilon, 200.0)
    after_i, after_v = analytic_after_refractory(time + epsilon, 200.0)
    assert (after_i - before_i) / (2 * epsilon) == pytest.approx(-current / .5, rel=1e-8)
    assert 250 * (after_v - before_v) / (2 * epsilon) == pytest.approx(-250 * (voltage + 65) / 10 + current, rel=1e-7)


@pytest.mark.parametrize('time', [1.99, math.nan, math.inf])
def test_invalid_post_release_time_is_rejected(time):
    with pytest.raises(ValueError):
        analytic_after_refractory(time, 200.0)


def test_known_mismatched_port_rejects_formal_use_before_building(tmp_path):
    with pytest.raises(ValueError, match='refractory PSC semantic mismatch'):
        run_reference({'run_class': 'reference_candidate'}, Path('missing_source'), tmp_path / 'build', tmp_path / 'output')
    assert not (tmp_path / 'build').exists()
    assert not (tmp_path / 'output').exists()

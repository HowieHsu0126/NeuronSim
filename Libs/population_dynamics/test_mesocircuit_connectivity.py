"""Independent probability, periodic geometry and multiplicity checks."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from .mesocircuit_connectivity import conditional_moments, uniform_kernel_integral


def test_periodic_wrap_mask_self_exclusion_and_multiplicity():
    mean, variance = conditional_moments(
        [1, 2, 3], [[-.45, 0.], [.45, 0.], [0., 0.]],
        [1], [[-.45, 0.]], extent_mm=1., beta_mm=.2, radius_mm=.2, p0=.4, repeats=3)
    p = .4 * math.exp(-.1/.2)
    np.testing.assert_allclose(mean, [3*p], rtol=1e-14)
    np.testing.assert_allclose(variance, [3*p*(1-p)], rtol=1e-14)


def test_tiled_conditional_moments_equal_scalar_pairs():
    ids = np.arange(4)
    xy = np.array([[-.49, -.4], [.49, -.4], [.15, .1], [-.2, -.1]])
    expected_m, expected_v = [], []
    for i in range(4):
        m = v = 0.
        for j in range(4):
            if i == j:
                continue
            d = [min(abs(float(xy[i,k]-xy[j,k])), 1-abs(float(xy[i,k]-xy[j,k]))) for k in range(2)]
            r = math.hypot(*d)
            p = .7 * math.exp(-r/.3) if r <= .5 else 0.
            m += 2*p; v += 2*p*(1-p)
        expected_m.append(m); expected_v.append(v)
    full = conditional_moments(ids, xy, ids, xy, extent_mm=1., beta_mm=.3,
                               radius_mm=.5, p0=.7, repeats=2)
    np.testing.assert_allclose(full[0], expected_m, rtol=1e-14)
    np.testing.assert_allclose(full[1], expected_v, rtol=1e-14)
    parts = [conditional_moments(ids, xy, ids[i:i+2], xy[i:i+2], extent_mm=1.,
             beta_mm=.3, radius_mm=.5, p0=.7, repeats=2) for i in (0, 2)]
    np.testing.assert_array_equal(np.concatenate([p[0] for p in parts]), full[0])
    np.testing.assert_array_equal(np.concatenate([p[1] for p in parts]), full[1])


@pytest.mark.parametrize("extent,beta,radius", [(1., .1, .5), (1., .2, .4), (4., .3, 1.5)])
def test_uniform_kernel_formula_matches_numerical_quadrature(extent, beta, radius):
    integrated = quad(lambda r: 2*math.pi*r*math.exp(-r/beta)/extent**2, 0, radius,
                      epsabs=1e-13)[0]
    assert uniform_kernel_integral(extent, beta, radius) == pytest.approx(integrated, rel=1e-13)


def test_source_expression_has_squared_truncation_factor_under_stated_assumptions():
    n, k, extent, beta, radius = 1000, 10., 1., .1, .4
    f = radius/beta
    a = 1-math.exp(-f)*(1+f)
    source_p0 = k/(n*math.pi*radius**2/extent**2)*.5*f*f*a
    expected_indegree = n*source_p0*uniform_kernel_integral(extent, beta, radius)
    assert expected_indegree == pytest.approx(k*a*a, rel=1e-13)
    assert expected_indegree < k


@pytest.mark.parametrize("change", [
    {"repeats": 0}, {"repeats": True}, {"p0": 1.1}, {"p0": -.1},
    {"beta_mm": 0.}, {"radius_mm": .6}, {"extent_mm": math.inf}
])
def test_invalid_probability_parameters_rejected(change):
    options = dict(extent_mm=1., beta_mm=.2, radius_mm=.5, p0=.5, repeats=1)
    options.update(change)
    with pytest.raises(ValueError):
        conditional_moments([1], [[0.,0.]], [2], [[.1,0.]], **options)

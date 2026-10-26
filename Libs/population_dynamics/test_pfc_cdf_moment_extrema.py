"""Check sharp finite-support primal/dual solutions against known distributions."""
import numpy as np
import pytest
from . import pfc_cdf_moment_extrema as extrema


@pytest.mark.parametrize("threshold,expected", [(0,.2),(1,.7)])
def test_three_point_support_moments_identify_distribution(threshold,expected):
    bounds,solved=extrema.sharp_interval(1.1,1.7,threshold,2)
    assert bounds[0]<=expected<=bounds[1]
    assert bounds[1]-bounds[0]<3e-8
    for x in solved:
        np.testing.assert_allclose(x["probability"],[.2,.5,.3],atol=1e-10)
        assert x["primal_residual"]<1e-10 and abs(x["primal_dual_gap"])<1e-10


def test_wider_support_retains_distinct_compatible_extrema():
    p=np.array([.2,.3,.1,.2,.2]);s=np.arange(5)
    bounds,solved=extrema.sharp_interval(float(p@s),float(p@(s*s)),2,4)
    assert bounds[0] <= p[:3].sum() <= bounds[1]
    assert solved[0]["attained_cdf"] < solved[1]["attained_cdf"]
    for x in solved:
        assert np.all(x["probability"]>=0)
        assert abs(x["probability"].sum()-1)<1e-10


def test_source_moment_infeasibility_is_fatal():
    with pytest.raises(ValueError,match="SECOND_RANGE"):
        extrema.sharp_interval(2.,3.,1,5)

"""Distribution inequalities and complete finite empirical support regressions."""
import numpy as np
import pytest

from . import pfc_cdf_moment_envelope as envelope


def test_nonconsecutive_support_recovers_raw_moments_and_zero_units():
    mean, second = envelope.count_moments(np.array([[.5,.75,1.],[1.,1.,1.]]), np.array([0,2,4]), 4)
    np.testing.assert_array_equal(mean,[1.5,0.])
    np.testing.assert_array_equal(second,[5.,0.])


@pytest.mark.parametrize("pmf", [[.7,.3], [.1,.2,.3,.4], [0.,0.,1.], [1.,0.,0.], [.0001,.4999,.5]])
def test_bounds_enclose_entire_cdf_on_both_sides_of_mean(pmf):
    p=np.array(pmf);support=np.arange(len(p));mean=float(p@support);second=float(p@(support**2));cdf=np.cumsum(p)
    for k in range(-1,len(p)+1):
        actual=0. if k<0 else float(cdf[min(k,len(p)-1)])
        first=envelope.moment_interval(mean,second,k,len(p)-1,use_second=False)
        two=envelope.moment_interval(mean,second,k,len(p)-1,use_second=True)
        assert first[0]-1e-14<=two[0]<=actual+1e-14
        assert actual-1e-14<=two[1]<=first[1]+1e-14


def test_bernoulli_first_moment_bounds_are_exact():
    assert envelope.moment_interval(.3,.3,0,1,use_second=False)==(.7,.7)


def test_cantelli_uses_k_plus_one_for_integer_upper_tail():
    lo,hi=envelope.moment_interval(1.,2.,2,10,use_second=True)
    assert lo==.8  # variance1 and upper-tail event X>=3, not X>=2
    assert hi==1.


def test_unequal_distributions_keep_negative_signed_difference():
    left=envelope.moment_interval(.7,.7,0,1,use_second=True)
    right=envelope.moment_interval(.2,.2,0,1,use_second=True)
    lo,hi=envelope.difference_interval(left,right)
    assert lo==pytest.approx(-.5) and hi==pytest.approx(-.5)


@pytest.mark.parametrize("cdf,support,maximum", [([[.5,.9]],[0,1],1), ([[.8,.7,1.]],[0,1,2],2), ([[.5,1.]],[0,2],1)])
def test_incomplete_nonmonotone_or_out_of_source_support_is_rejected(cdf,support,maximum):
    with pytest.raises(ValueError):
        envelope.count_moments(np.asarray(cdf),np.asarray(support),maximum)


def test_invalid_moments_are_not_silently_clipped():
    with pytest.raises(ValueError,match="SECOND_RANGE"):
        envelope.moment_interval(2.,3.,1,5,use_second=True)
    with pytest.raises(ValueError,match="SECOND_RANGE"):
        envelope.moment_interval(2.,11.,1,5,use_second=True)

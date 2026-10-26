"""Joint-event counting, analytic tails, and known rare-law recovery."""
import math
import numpy as np
import pytest
from .pfc_cdf_tail_envelope import upper_certificate


def test_joint_departures_count_blocks_not_groups():
    x=np.full((3,128),.01);x[:,0]=.1;x[0,1]=.2
    result=upper_certificate(x,np.zeros(3),np.full(3,.02))
    assert result['joint_departures']==2
    assert result['groups']==3 and result['blocks']==128
    assert result['joint_tail_probability_upper']>.0125


def test_zero_departure_tail_and_union_radius():
    x=np.tile([.0096,.0098],64)[None,:]
    result=upper_certificate(x,[.0096],[.0098])
    p=1-.0125**(1/128)
    assert result['joint_departures']==0
    assert result['joint_tail_probability_upper']==pytest.approx(p)
    assert result['core_radius'][0]==pytest.approx(.0002*math.sqrt(math.log(80)/256))
    assert result['upper'][0]==pytest.approx(.0097+result['core_radius'][0]+p*(1-.0098))
    assert result['upper'][0]>.010103>.01


def test_pointwise_clip_and_joint_indicator_dominate_all_bounded_values():
    x=np.array([[0,.2,.3,1],[.2,.4,.8,.1]])
    a=np.array([.1,.25]);b=np.array([.5,.75])
    event=np.any((x<a[:,None])|(x>b[:,None]),axis=0)
    assert np.all(x<=np.clip(x,a[:,None],b[:,None])+(1-b[:,None])*event)
    result=upper_certificate(x,a,b)
    assert np.all((result['upper']>=0)&(result['upper']<=1))


def test_constant_clip_range_does_not_imply_zero_tail_uncertainty():
    result=upper_certificate(np.zeros((1,128)),[0],[0])
    assert result['core_radius'][0]==0 and result['upper'][0]>0
    assert result['upper'][0]==pytest.approx(1-.0125**(1/128))


@pytest.mark.parametrize('x,a,b',[(np.array([[1.1,0]]),[0],[1]),(np.array([[.1,.2]]),[.4],[.3]),(np.array([[np.nan,.2]]),[0],[1]),(np.array([[.1,.2]]),[0,0],[1,1])])
def test_invalid_physical_values_and_domains_rejected(x,a,b):
    with pytest.raises(ValueError):upper_certificate(x,a,b)

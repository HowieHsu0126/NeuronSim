"""Independent arithmetic and frozen failure-position checks; no native execution."""
from fractions import Fraction

import numpy as np
import pytest

from .lfads_integer_range_diagnostic import (
    BOUND_ERROR, INT64_LIMIT, _pair_product_sum, conservative_bounds,
    next_failed_origin, ordered_pair_oracle,
)


def test_next_failed_origin_uses_origin_outer_case_inner_order():
    cases=[dict(case_id=f'case_{i}') for i in range(16)]
    failure=dict(status='FAILED',error=BOUND_ERROR,training_seed=41,verified_origins=5)
    assert next_failed_origin(failure,cases,[1000.,2000.,3000.]) == (0,5,cases[5],1000.)
    # Same retained ordinal with a shorter explicit cohort crosses the origin boundary.
    assert next_failed_origin(failure,cases[:4],[1000.,2000.]) == (1,1,cases[1],2000.)
    for key,value in [('status','PASS'),('error','different failure'),('training_seed',42),('verified_origins',4)]:
        with pytest.raises(ValueError):
            next_failed_origin(dict(failure,**{key:value}),cases,[1000.])


def test_pair_products_split_reduction_before_python_fallback():
    limit=INT64_LIMIT
    a=np.array([limit,1,-limit],dtype=np.int64)
    b=np.array([1,limit,1],dtype=np.int64)
    value,fixed,arbitrary=_pair_product_sum(a,b)
    assert value==limit and fixed==3 and arbitrary==0
    value,fixed,arbitrary=_pair_product_sum(np.array([limit,-limit],dtype=np.int64),
                                           np.array([limit,limit],dtype=np.int64))
    assert value==0 and fixed==0 and arbitrary==2


def test_oracle_retains_negative_extreme_pairs_and_all_cells():
    limit=INT64_LIMIT
    values=np.array([[0,limit,9],[limit,0,5],[0,limit,4]],dtype=np.int64).reshape(3,1,3)
    truth=np.array([[limit//2,limit,3]],dtype=np.int64)
    numerator=sum((int(values[i,t,u])-int(truth[t,u]))*(int(values[j,t,u])-int(truth[t,u]))
                  for t in range(1) for u in range(3) for i in range(3) for j in range(3) if i!=j)
    denominator=3*2*3
    for chunk in (1,2,4096):
        result=ordered_pair_oracle(values,truth,chunk_units=chunk)
        assert result['exact_numerator']==numerator and result['exact_denominator']==denominator
        assert result['float32_value']==float(np.float32(float(Fraction(numerator,denominator))))
        assert result['fixed_width_pair_cells']+result['python_integer_pair_cells']==9


def test_conservative_bound_can_reject_an_exactly_zero_correction():
    values=np.array([2**40,0],dtype=np.int64).reshape(2,1,1)
    truth=np.zeros((1,1),dtype=np.int64)
    bounds=conservative_bounds(values,truth)
    assert bounds['first_rejected_block']['conservative_bound']==4*2**80
    assert ordered_pair_oracle(values,truth)['exact_numerator']==0


def test_oracle_rejects_noninteger_counts():
    with pytest.raises(ValueError):
        ordered_pair_oracle(np.zeros((2,1,1)),np.zeros((1,1),dtype=np.int64))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_oracle_accepts_verified_integral_float_observations(dtype):
    values=np.array([0,2],dtype=np.int32).reshape(2,1,1)
    truth=np.array([[1]],dtype=dtype)
    result=ordered_pair_oracle(values,truth)
    assert result['exact_numerator']==-2
    assert result['exact_denominator']==2 and result['float32_value']==-1.0


@pytest.mark.parametrize('value',[.5,-1.,np.nan,np.inf,float(2**63)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_oracle_does_not_silently_coerce_invalid_float_observations(value,dtype):
    with pytest.raises(ValueError):
        ordered_pair_oracle(np.zeros((2,1,1),dtype=np.int64),np.array([[value]],dtype=dtype))

"""Exact finite-population and noisy-block variance enumeration."""
from fractions import Fraction as F
from itertools import combinations,product
import numpy as np
import pytest
from .pfc_first_passage_history_summary import variance_components,between_components


def cases(shift=F(0)):
    z=[F(i,4)+shift for i in range(4)];result=[]
    for subset in combinations(range(4),2):
        for bits in product((0,1),repeat=4):
            means=[z[u]*(bits[2*j]+bits[2*j+1]) for j,u in enumerate(subset)]
            variances=[z[u]**2*(bits[2*j]-bits[2*j+1])**2 for j,u in enumerate(subset)]
            stats=variance_components(list(map(float,means)),list(map(float,variances)),4)
            y=sum(means)/2
            sample_var=(means[0]-means[1])**2/2;mean_v=sum(variances)/2
            e=(sample_var+mean_v)/4
            assert stats["sample_mean"]==float(y) and stats["within_archive_mean_variance_diagnostic"]==float(e)
            result.append((y,e,stats))
    return result


def test_exact96_case_sampling_and_inner_variance_identity():
    rows=cases();assert len(rows)==96
    mean=sum(r[0] for r in rows)/96
    variance=sum((r[0]-mean)**2 for r in rows)/96
    e=sum(r[1] for r in rows)/96
    assert mean==F(3,8) and variance==e==F(31,384)
    assert any(r[2]["latent_unit_window_variance_diagnostic"]<0 for r in rows)


def test_exact9216_case_between_block_correction_preserves_negatives():
    a,b=cases(),cases(F(1,8));total=F(0);negative=0
    for x,y in product(a,b):
        expected=(x[0]-y[0])**2/2-(x[1]+y[1])/2
        actual=between_components([x[2],y[2]])
        assert actual["latent_archived_block_mean_variance_diagnostic"]==float(expected)
        total+=expected;negative+=int(expected<0)
    assert total/9216==F(1,128) and negative>0


def test_census_and_invalid_denominators():
    census=variance_components([0,.25,.5,.75],[0,0,0,0],4)
    assert census["within_archive_mean_variance_diagnostic"]==0
    assert census["sampled_estimate_variance"]>0
    with pytest.raises(ValueError,match="BLOCK_DENOMINATOR"):
        variance_components([0],[0],4)
    with pytest.raises(ValueError,match="BLOCK_DENOMINATOR"):
        variance_components([0,1],[0,-1],4)

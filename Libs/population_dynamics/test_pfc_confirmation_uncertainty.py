"""Joint-draw reuse, monotone outer bounds and covariance block influence."""
import json
import subprocess
import sys

import numpy as np
import pytest
from .pfc_confirmation_uncertainty import (
    bootstrap_primitives,monotone_cdf_envelope,signed_interval,intersect_intervals,
    covariance_jackknife,construct_bands,STRUCTURAL,UNDEFINED,EMPIRICAL,
    binomial_probability_upper,frozen_tail_departures,
)


def test_joint_maximum_uses_the_same_draw_not_separate_quantiles():
    x=np.array([[0.,1.,2.],[3.,1.,0.]])
    m=np.array([[3,0,0],[0,3,0],[0,0,3],[1,1,1],[2,1,0]])
    prior=np.array([0.,9.,0.,0.,0.])
    point,se,maximum,influence=bootstrap_primitives(x,m,prior,1)
    draws=(m/3)@(x-x.mean(1)[:,None]).T
    expected=np.maximum(prior,(np.abs(draws)/draws.std(0,ddof=1)).max(1))
    np.testing.assert_allclose(maximum,expected)
    np.testing.assert_allclose(point,x.mean(1))
    np.testing.assert_allclose(se,draws.std(0,ddof=1))
    assert maximum[1]==9 and np.all(influence>0)


def test_constant_nonboundary_primitive_has_no_invented_bootstrap_variance():
    x=np.full((1,3),.1);m=np.array([[3,0,0],[1,1,1],[0,2,1]])
    point,se,maximum,influence=bootstrap_primitives(x,m,np.array([1.,2.,3.]))
    assert point[0]==.1 and se[0]==0 and np.isnan(influence[0])
    np.testing.assert_array_equal(maximum,[1,2,3])


def test_invalid_resample_block_count_is_rejected():
    with pytest.raises(ValueError,match='SHARED_WHOLE_BLOCK'):
        bootstrap_primitives(np.ones((2,3)),np.array([[1,1,0],[0,0,3]]),np.zeros(2))


def test_observed_saturation_does_not_anchor_a_population_cdf_at_one():
    # Last primitive was empirically1 but has no structural support proof.
    lo,hi=monotone_cdf_envelope([0,2,5],[.4,.8,0.],[.6,.9,1.],[-1,0,1,2,4,5,8,10],10)
    np.testing.assert_allclose(lo,[0,.4,.4,.8,.8,.8,.8,1])
    np.testing.assert_allclose(hi,[0,.6,.9,.9,1,1,1,1])


def test_monotone_or_channel_crossings_remain_failures():
    with pytest.raises(ValueError,match='CROSSING_MONOTONE'):
        monotone_cdf_envelope([0,1],[.9,.1],[1.,.8],[0,1],3)
    with pytest.raises(ValueError,match='CROSSING_SIMULTANEOUS'):
        intersect_intervals((0,1),(2,3))


def test_four_mean_contrast_propagates_every_sign():
    assert signed_interval([10,3,4,2],[10,3,4,2],[1,-1,-1,1])==(5.,5.)
    assert signed_interval([9,2,3,1],[11,4,5,3],[1,-1,-1,1])==(1.,9.)


@pytest.mark.parametrize('blocks',[3,4,7])
def test_covariance_pseudovalues_equal_brute_force_delete_one(blocks):
    pairs=np.array([(i,j) for i in range(blocks) for j in range(i+1,blocks)])
    h=np.random.default_rng(1401).normal(size=(4,len(pairs)))
    pseudo,se,influence,residual=covariance_jackknife(h,pairs,blocks)
    expected=np.stack([blocks*h.mean(1)-(blocks-1)*h[:,np.all(pairs!=block,axis=1)].mean(1)
                       for block in range(blocks)],axis=1)
    np.testing.assert_allclose(pseudo,expected,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(se,expected.std(1,ddof=1)/np.sqrt(blocks))
    np.testing.assert_allclose(pseudo.mean(1),h.mean(1))
    assert np.all((influence>0)&(influence<=1)) and np.all((residual>=0)&(residual<=1+1e-14))


def test_empirical_zero_cdf_and_response_get_nonzero_outer_width():
    support=[0,1];physical={(5.,0):3};lookup={};point=[];se=[];pl=[];ph=[]
    for label in ['left','right']:
        for threshold in support:
            lookup['cdf',label,'cue','late_delay',5.,0,threshold]=len(point)
            point.append(.9 if threshold==0 else 1.);se.append(.01 if threshold==0 else 0.);pl.append(0.);ph.append(1.)
    for label in ['left','right']:
        for condition in ['cue','omission']:
            lookup['response',label,condition,'late_delay','all_unit_rate_hz']=len(point)
            point.append(1.);se.append(.02);pl.append(0.);ph.append(680.)
    rows=[]
    for family,ident,status in [
        ('count_cdf',dict(left_cohort='left',right_cohort='right',condition='cue',phase='late_delay',window_ms=5,field_index=0,count_threshold=1),EMPIRICAL),
        ('paired_response',dict(left_cohort='left',right_cohort='right',contrast='cue_minus_omission',phase='late_delay',metric='all_unit_rate_hz'),EMPIRICAL),
        ('paired_response',dict(left_cohort='left',right_cohort='right',contrast='cue_minus_omission',phase='baseline',metric='all_unit_rate_hz'),STRUCTURAL),
        ('field_lag_covariance',dict(left_cohort='left',right_cohort='right',phase='baseline'),UNDEFINED),
    ]:
        rows.append(dict(family=family,endpoint_identity=json.dumps(ident),definition_status=status,estimate='0' if status!=UNDEFINED else ''))
    bands,_,_=construct_bands(rows,{},tuple(map(np.asarray,[point,se,pl,ph])),lookup,support,physical,2.)
    np.testing.assert_allclose([bands[0]['lower'],bands[0]['upper']],[-.12,.12])
    np.testing.assert_allclose([bands[1]['lower'],bands[1]['upper']],[-.16,.16])
    assert bands[2]['lower']==bands[2]['upper']==0
    assert bands[3]['lower']==bands[3]['upper']==''


def test_real_cli_rejects_confirmation_scope_with_nonzero_exit(tmp_path):
    config=tmp_path/'wrong.yaml';config.write_text('schema: pfc_confirmation_uncertainty_diagnostic_v1\nscope: confirmation\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_confirmation_uncertainty',
                           '--config',str(config),'--project-root',str(tmp_path),'--output',str(tmp_path/'output')],
                          capture_output=True,text=True)
    assert result.returncode!=0 and 'CALIBRATION_DIAGNOSTIC_ONLY_REQUIRED' in result.stderr
    assert not (tmp_path/'output').exists()


def test_fixed_binomial_zero_all_and_known_interior_limits():
    upper=binomial_probability_upper(0,128,.025)
    assert upper==pytest.approx(1-.025**(1/128)) and upper>0
    assert binomial_probability_upper(128,128,.025)==1
    # P[Binomial(8,1/2)<=2]=(1+8+28)/256.
    assert binomial_probability_upper(2,8,37/256)==pytest.approx(.5,abs=1e-14)
    assert binomial_probability_upper(0,32,.025)>upper
    assert binomial_probability_upper(1,128,.025)>upper


def test_joint_departures_count_blocks_not_primitive_changes():
    centers=np.array([0.,1.,.5])
    values=np.repeat(centers[:,None],4,axis=1)
    values[:,1]=[.1,.9,.4];values[0,3]=.2
    np.testing.assert_array_equal(frozen_tail_departures(values,centers),[False,True,False,True])
    values[0,0]=np.nextafter(0.,1.)
    assert frozen_tail_departures(values,centers)[0]
    values[0,0]=np.nan
    with pytest.raises(ValueError,match='FROZEN_PRIMITIVE'):
        frozen_tail_departures(values,centers)


def test_joint_tail_bound_attains_range_mixture_and_stays_nonzero():
    # A constant CDF0 lacks a lower-count anchor; its rare departure bound is p.
    p=binomial_probability_upper(0,128,.025)
    primitive=(np.array([0.,0.]),np.zeros(2),np.zeros(2),np.ones(2))
    lookup={('cdf',label,'cue','late_delay',5.,0,0):i for i,label in enumerate(['left','right'])}
    ident=dict(left_cohort='left',right_cohort='right',condition='cue',phase='late_delay',window_ms=5,field_index=0,count_threshold=0)
    domains=[dict(family='count_cdf',endpoint_identity=json.dumps(ident),definition_status=EMPIRICAL,estimate='0')]
    rows,lo,hi=construct_bands(domains,{},primitive,lookup,[0],{(5.,0):3},5.,p)
    np.testing.assert_allclose(lo,[0,0]);np.testing.assert_allclose(hi,[p,p])
    np.testing.assert_allclose([rows[0]['lower'],rows[0]['upper']],[-p,p])
    # A mixture with mass p at1 and mass1-p at0 attains E[F]=p.
    assert (1-p)*0+p*1==pytest.approx(hi[0])
    rows,_,_=construct_bands(domains,{},primitive,lookup,[0],{(5.,0):3},5.,1.)
    assert (rows[0]['lower'],rows[0]['upper'])==(-1.,1.)

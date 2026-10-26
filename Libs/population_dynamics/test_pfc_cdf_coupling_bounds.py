"""Complete blockwise CDF envelopes and faithful legacy-band reconstruction."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from . import pfc_cdf_coupling_bounds as bounds
from .pfc_confirmation_uncertainty import construct_bands, identity_key


def cdf_fixture():
    # Both matrices are monotone, bounded CDFs from paired block distributions.
    left=np.array([[.3,.8,1.],[.1,.8,1.],[.4,.8,1.],[0.,.8,1.]])
    right=np.array([[.2,.8,1.]]*4)
    domains=[];saved=[]
    for index,threshold in enumerate(range(3)):
        identity=dict(left_cohort='left',right_cohort='right',condition='cue',phase='late_delay',window_ms=5,field_index=0,count_threshold=threshold)
        row=dict(family='count_cdf',endpoint_identity=json.dumps(identity),definition_status=bounds.STRUCTURAL if threshold==2 else bounds.EMPIRICAL,
                 physical_maximum_count='2',estimate=str((left-right)[:,index].mean()))
        domains.append(row);saved.append(dict(row,domain_row_index=index))
    return domains,(left-right).T,saved


def test_group_max_is_per_block_over_every_threshold_not_max_of_means():
    domains,scores,saved=cdf_fixture()
    k,identities,physical,row_group,denominators,numerators,roundtrips=bounds.complete_cdf_group_scores(domains,scores,saved,[1])
    np.testing.assert_allclose(k,[[.1,.1,.2,.2]])
    assert k.mean()==pytest.approx(.15) and np.max(np.abs(scores.mean(1)))==pytest.approx(0.)
    assert len(identities)==1 and physical.tolist()==[2] and row_group.tolist()==[0,0,0]


@pytest.mark.parametrize('change',['missing_threshold','duplicate_threshold','physical_saturation','mean','block_range'])
def test_invalid_or_incomplete_physical_threshold_groups_rejected(change):
    domains,scores,saved=cdf_fixture()
    if change=='missing_threshold':
        domains=domains[1:];scores=scores[1:];saved=saved[1:]
        for i,row in enumerate(saved):row['domain_row_index']=i
    elif change=='duplicate_threshold':
        domains[1]['endpoint_identity']=domains[0]['endpoint_identity'];saved[1]['endpoint_identity']=domains[0]['endpoint_identity']
    elif change=='physical_saturation':scores[2]=.1;saved[2]['estimate']='.1'
    elif change=='mean':saved[0]['estimate']='.1'
    else:scores[0,0]=1.1
    with pytest.raises(ValueError):bounds.complete_cdf_group_scores(domains,scores,saved,[1])


def test_shared_maximum_and_quantile_do_not_reuse_separate_draw_order():
    k=np.array([[.1,.1,.2,.2],[0.,0.,0.,0.]])
    m=np.array([[4,0,0,0],[0,0,4,0],[1,1,1,1],[3,0,1,0],[0,1,0,3]])
    old=np.array([0.,0.,9.,0.,0.])
    point,se,maximum,_,critical=bounds.combined_critical(k,m,old,.025,1)
    draws=(m/4)@(k[0]-k[0].mean())
    expected=np.maximum(old,np.abs(draws)/draws.std(ddof=1))
    np.testing.assert_allclose(maximum,expected)
    assert maximum[2]==9 and critical>=np.quantile(old,.975,method='higher')
    assert se[1]==0 and point[1]==0


def legacy_fixture():
    domains,scores,saved=cdf_fixture()
    # Ensure direct and primitive channels have a common empirical center.
    for row in domains:row['estimate']='0'
    lookup={};point=[];se=[];lo=[];hi=[]
    for label in ['left','right']:
        for threshold in range(3):
            lookup['cdf',label,'cue','late_delay',5.,0,threshold]=len(point)
            point.append([.2,.8,1.][threshold]);se.append(.01 if threshold<2 else 0.);lo.append(1. if threshold==2 else 0.);hi.append(1.)
    ident=dict(left_cohort='left',right_cohort='right',condition='cue',phase='late_delay',field_index=0)
    domains.append(dict(family='field_mean',endpoint_identity=json.dumps(ident),definition_status='ESTIMATED_EMPIRICAL_SPREAD',estimate='0'))
    precision={identity_key('field_mean',ident):dict(status='ESTIMATED_EMPIRICAL_SPREAD',estimate=0.,bootstrap_standard_error=.02)}
    return domains,precision,tuple(map(np.array,[point,se,lo,hi])),lookup,[0,1,2],{(5.,0):2}


def test_all_legacy_channels_receive_new_critical_before_cdf_intersection():
    args=legacy_fixture();domains=args[0]
    old,_,_=construct_bands(*args,2.,.1)
    widened,_,_=construct_bands(*args,4.,.1)
    augmented,upper=bounds.augment_cdf_bands(widened,np.array([0,0,0,-1]),np.array([.015]),np.array([.002]),4.)
    assert old[3]['upper']==pytest.approx(.04) and augmented[3]['upper']==pytest.approx(.08)
    assert augmented[0]['upper']==pytest.approx(.023) and augmented[0]['lower']==pytest.approx(-.023)
    assert augmented[2]['lower']==augmented[2]['upper']==0
    assert augmented[3]['cdf_coupling_channel_used'] is False
    assert all(abs(float(row['estimate']))<=upper[0] for row in augmented[:3])


def test_constant_ks_even_zero_adds_no_population_support_channel():
    args=legacy_fixture();legacy,_,_=construct_bands(*args,4.,.1)
    augmented,upper=bounds.augment_cdf_bands(legacy,np.array([0,0,0,-1]),np.array([0.]),np.array([0.]),4.)
    assert upper.tolist()==[1.]
    assert augmented[0]['lower']==legacy[0]['lower'] and augmented[0]['upper']==legacy[0]['upper']
    assert augmented[0]['cdf_coupling_channel_used'] is False
    assert 'physical_fallback_1' in augmented[0]['cdf_coupling_anchor_status']
    assert augmented[0]['lower']<augmented[0]['upper']


def test_saved_actual_and_projected_bands_roundtrip_before_augmentation():
    args=legacy_fixture();actual,_,_=construct_bands(*args,2.,.1);future,_,_=construct_bands(*args,1.,.03)
    saved=[dict(row,planning_lower_root_n_and_zero_tail_at_128=f['lower'],planning_upper_root_n_and_zero_tail_at_128=f['upper'])
           for row,f in zip(actual,future)]
    check=bounds.compare_reconstructed_bands(saved,actual,future)
    assert check['rows']==4 and check['maximum_absolute_error']==0
    saved[3]['planning_upper_root_n_and_zero_tail_at_128']+=.01
    with pytest.raises(ValueError,match='NUMERICAL_RECONSTRUCTION_FAILED'):
        bounds.compare_reconstructed_bands(saved,actual,future)


def test_current_means_must_lie_inside_empirical_coupling_envelope():
    args=legacy_fixture();legacy,_,_=construct_bands(*args,4.,.1)
    legacy[0]['estimate']=.5
    with pytest.raises(ValueError,match='EMPIRICAL_POINT_MUST_LIE'):
        bounds.augment_cdf_bands(legacy,np.array([0,0,0,-1]),np.array([.01]),np.array([.001]),4.)


def test_primitive_restoration_preserves_tail_selection_and_lookup():
    rows=[]
    for label in ['left','right']:
        for threshold in range(2):
            ident=dict(family='primitive_cdf',cohort=label,condition='cue',phase='late_delay',window_ms=5,field_index=0,count_threshold=threshold)
            rows.append(dict(primitive_identity=json.dumps(ident),estimate='1',bootstrap_standard_error='0',physical_lower='0',physical_upper='1',calibration_constant_tail_candidate='True'))
    primitive,lookup,support=bounds.restore_primitives(rows)
    assert support==[0,1] and lookup['cdf','right','cue','late_delay',5.,0,1]==3
    assert np.all(primitive[1]==0) and np.all(primitive[2]==0) and np.all(primitive[3]==1)
    rows[0]['calibration_constant_tail_candidate']='False'
    with pytest.raises(ValueError,match='CONSTANT_PRIMITIVE_ROUTE_CHANGED'):bounds.restore_primitives(rows)


def test_real_cli_rejects_adoption_flag(tmp_path):
    config=tmp_path/'wrong.yaml';config.write_text('schema: pfc_cdf_coupling_bounds_diagnostic_v1\nscope: development_calibration\ncoverage_qualified: false\nmethod_adopted_for_confirmation: true\nconfirmation_execution_released: false\n')
    r=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_cdf_coupling_bounds','--config',str(config),
                      '--project-root',str(tmp_path),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert r.returncode!=0 and 'UNQUALIFIED_CALIBRATION_CANDIDATE_ONLY_REQUIRED' in r.stderr
    assert not (tmp_path/'out').exists()


def test_constant_count_numerator_ignores_floating_cdf_offset_roundoff():
    differences=np.array([[.3-.2,.7-.6,.6-.5,.4-.3],[0.,0.,0.,0.]])
    assert np.any(differences[0] != differences[0,0])
    k,numerator,diagnostic=bounds.integer_cdf_coupling(differences,100)
    np.testing.assert_array_equal(numerator,[10,10,10,10])
    np.testing.assert_array_equal(k,[.1,.1,.1,.1])
    assert diagnostic['raw_float_ks_varied_but_integer_ks_constant'] is True
    m=np.array([[4,0,0,0],[0,4,0,0],[0,0,4,0],[1,1,1,1]])
    _,se,maximum,_,_=bounds.combined_critical(k[None,:],m,np.ones(4),.025,256)
    assert se[0]==0 and np.all(maximum==1)


def test_unjustified_count_lattice_projection_rejected():
    with pytest.raises(ValueError,match='NOT_ON_SOURCE_COUNT_LATTICE'):
        bounds.integer_cdf_coupling(np.array([[.105,.1]]),100)


def test_omitted_proven_saturated_terminal_is_explicit_zero_not_missing_data():
    domains,scores,saved=cdf_fixture()
    result=bounds.complete_cdf_group_scores(domains[:-1],scores[:-1],saved[:-1],[1],terminal_boundary_identity_proven=True)
    np.testing.assert_allclose(result[0],[[.1,.1,.2,.2]])
    assert result[-1][0]['source_identity_terminal_boundary_added'] is True
    with pytest.raises(ValueError,match='UNPROVEN_MISSING_TERMINAL'):
        bounds.complete_cdf_group_scores(domains[:-1],scores[:-1],saved[:-1],[1])


def test_terminal_identity_never_fills_missing_nontrivial_threshold():
    domains,scores,saved=cdf_fixture()
    domains=[domains[0],domains[2]];scores=scores[[0,2]];saved=[saved[0],saved[2]];saved[1]['domain_row_index']=1
    with pytest.raises(ValueError,match='MISSING_NONTRIVIAL'):
        bounds.complete_cdf_group_scores(domains,scores,saved,[1],terminal_boundary_identity_proven=True)

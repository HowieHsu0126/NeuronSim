"""Frozen budgets preserve source units, weak contrasts and full interval checks."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from .pfc_numerical_error_budget import (
    response_spike_sensitivities,response_budget,build_margins,assess_planning,
    summarize_counts,PHASES,METRICS,STRUCTURAL,UNDEFINED,source_variance_plan,auxiliary_variance_point_plan,
)

POLICY=dict(field_mean_scale_fraction=.1,covariance_entry_scale_fraction=.1,count_cdf_absolute_probability=.01,
            response_signal_fraction=.01,response_block_sd_fraction=.1,response_single_spike_allowance=1.)


@pytest.mark.parametrize('ne,ni,center,width',[(12,3,120.,30.),(10,2,0.,72.)])
def test_event_sensitivity_matches_exhaustive_source_single_spike_readouts(ne,ni,center,width):
    config=dict(excitatory_neurons=ne,inhibitory_neurons=ni,stimulus_center_deg=center,stimulus_width_deg=width)
    sensitivities,_=response_spike_sensitivities(config)
    for phase,(start,stop) in PHASES.items():
        values=[];seconds=(stop-start)/1000
        for unit in range(ne+ni):
            counts=np.zeros(ne+ni,dtype=np.int64);counts[unit]=1
            row=summarize_counts(counts,ne,center,width,stop-start)
            row['all_unit_rate_hz']=1/((ne+ni)*seconds)
            row['vector_x_hz']=row['vector_x_spikes']/(ne*seconds)
            row['vector_y_hz']=row['vector_y_spikes']/(ne*seconds)
            values.append([row[name] for name in METRICS])
        actual=np.max(np.abs(values),axis=0)
        np.testing.assert_allclose([sensitivities[phase,name] for name in METRICS],actual,rtol=1e-14,atol=1e-15)


def test_one_spike_allowance_is_outside_percentages_and_not_doubled_for_paired_arms():
    margin,parts,dominant=response_budget(0.,0.,.003,POLICY)
    assert margin==.003 and parts['single_spike']==.003 and dominant=='single_spike'
    assert response_budget(-10.,.1,.003,POLICY)[0]==.1
    assert response_budget(.001,.4,.003,POLICY)[0]==pytest.approx(.04)


def fixture_domains():
    rows=[]
    for family,extra,status in [
        ('field_mean',dict(condition='cue',field_index='0'),'ESTIMATED_EMPIRICAL_SPREAD'),
        ('field_lag_covariance',dict(condition='cue',lag_ms='5',leading_field='0',trailing_field='1'),'ESTIMATED_EMPIRICAL_SPREAD'),
        ('count_cdf',dict(condition='cue',window_ms='5',field_index='0',count_threshold='1'),'EMPIRICALLY_DEGENERATE_UNCERTAIN'),
        ('paired_response',dict(contrast='cue_minus_omission',metric='all_unit_rate_hz'),'EMPIRICALLY_DEGENERATE_UNCERTAIN'),
        ('paired_response',dict(contrast='strong_cue_minus_omission',metric='all_unit_rate_hz'),'ESTIMATED_EMPIRICAL_SPREAD'),
        ('paired_response',dict(contrast='cue_minus_omission',metric='all_unit_rate_hz'),STRUCTURAL),
        ('field_lag_covariance',dict(condition='cue',lag_ms='100',leading_field='0',trailing_field='1'),UNDEFINED),
    ]:
        ident=dict(left_cohort='coarse',right_cohort='fine',phase='baseline' if status in (STRUCTURAL,UNDEFINED) else 'cue',**extra)
        rows.append(dict(family=family,endpoint_identity=json.dumps(ident),definition_status=status,estimate='0'))
    return rows


def test_field_reference_product_and_source_specific_weak_contrast_budget():
    references=({('cue','cue',0):dict(scale=.04,floor=.04),('cue','cue',1):dict(scale=.2,floor=.01)}, {},
                {('cue_minus_omission','cue','all_unit_rate_hz'):dict(mean=0.,sd=0.),
                 ('strong_cue_minus_omission','cue','all_unit_rate_hz'):dict(mean=100.,sd=1.)})
    sensitivity={('cue','all_unit_rate_hz'):.0001}
    rows=build_margins(fixture_domains(),references,sensitivity,POLICY)
    assert rows[0]['margin']==pytest.approx(.004)
    assert rows[1]['margin']==pytest.approx(.0008) and rows[1]['units']=='Hz^2'
    assert rows[2]['margin']==.01
    assert rows[3]['margin']==.0001 and rows[4]['margin']==1.
    assert rows[5]['margin']==0 and rows[6]['margin']==''
    # Changing clock differences cannot alter these predeclared budget rules.
    domains=fixture_domains()
    for row in domains:row['estimate']='123456'
    assert build_margins(domains,references,sensitivity,POLICY)==rows
    references[2]['strong_cue_minus_omission','cue','all_unit_rate_hz']['mean']=10000.
    assert build_margins(fixture_domains(),references,sensitivity,POLICY)[3]==rows[3]


def test_planning_requires_the_entire_interval_not_just_its_center():
    ident=json.dumps(dict(left_cohort='coarse',right_cohort='fine',condition='cue',phase='cue',field_index='0'))
    margins=[];bands=[]
    for i,(estimate,lo,hi) in enumerate([(0.,-.6,.6),(.7,.6,.8),(.1,-.2,.3)]):
        common=dict(domain_row_index=i,family='field_mean',endpoint_identity=ident,definition_status='ESTIMATED_EMPIRICAL_SPREAD')
        margins.append(dict(common,margin='0.5'))
        bands.append(dict(common,estimate=str(estimate),lower=str(lo),upper=str(hi),planning_lower_root_n_and_zero_tail_at_128=str(lo),planning_upper_root_n_and_zero_tail_at_128=str(hi)))
    rows,_=assess_planning(margins,bands,128)
    assert [row['planning_status'] for row in rows]==['PLANNING_BAND_EXCEEDS_BUDGET','PLANNING_BAND_EXCEEDS_BUDGET','PLANNING_BAND_WITHIN_BUDGET']
    assert [row['calibration_point_outside_margin'] for row in rows]==[False,True,False]
    assert [row['calibration_interval_relation'] for row in rows]==['OVERLAPS_BUDGET','DISJOINT_FROM_BUDGET','WITHIN_BUDGET']
    bands[1]['endpoint_identity']='changed'
    with pytest.raises(ValueError,match='IDENTITIES_DIFFER'):assess_planning(margins,bands,128)


def test_invalid_reference_scatter_is_rejected():
    with pytest.raises(ValueError,match='FINITE_SOURCE_RESPONSE_REFERENCE_REQUIRED'):
        response_budget(1.,-1.,.01,POLICY)


def test_real_cli_rejects_unfrozen_scope_without_writing_margins(tmp_path):
    config=tmp_path/'bad.yaml';config.write_text('schema: wrong\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_numerical_error_budget',
                           '--config',str(config),'--project-root',str(tmp_path),'--output',str(tmp_path/'output')],capture_output=True,text=True)
    assert result.returncode!=0 and 'PROSPECTIVE_FROZEN_ERROR_BUDGET_REQUIRED' in result.stderr
    assert not (tmp_path/'output').exists()
def _source_plan_fixture(family, margin, units, index=0, status='ESTIMATED_EMPIRICAL_SPREAD'):
    row=dict(domain_row_index=str(index),family=family,endpoint_identity=json.dumps(dict(family=family)),
             definition_status=status,margin=str(margin),units=units)
    cov=family=='field_lag_covariance'
    stat=dict(row,coupled_original_independent_blocks='32',coupled_signed_estimate='0.0',
        coupled_sample_variance='0.0001',coupled_sample_variance_kind='jackknife_pseudovalue_variance' if cov else 'paired_block_difference_variance',
        coupled_score_kind='covariance_jackknife_pseudovalue' if cov else 'paired_block_cdf_difference' if family=='count_cdf' else 'paired_block_linear_difference',
        coupled_empirical_status=status)
    return row,stat


def test_source_plan_uses_each_endpoint_margin_and_units():
    first,a=_source_plan_fixture('count_cdf','.01','probability')
    second,b=_source_plan_fixture('source_response','1e-5','Hz',index=1)
    rows=list(source_variance_plan([first,second],[a,b],family_size=75978))
    assert rows[1]['scalar_source_variance_ceiling']==pytest.approx(rows[0]['scalar_source_variance_ceiling']*1e-6)
    assert rows[1]['scalar_increment_ceiling_if_zero_variance']==pytest.approx(rows[0]['scalar_increment_ceiling_if_zero_variance']*1e-3)
    assert rows[0]['units']=='probability' and rows[1]['units']=='Hz'
    assert rows[0]['variance_units']=='probability^2' and rows[1]['variance_units']=='Hz^2'
    assert all(row['population_variance_certified'] is False for row in rows)


def test_source_plan_covariance_pseudovalues_are_not_block_variances():
    row,stat=_source_plan_fixture('field_lag_covariance','.01','Hz^2')
    result=list(source_variance_plan([row],[stat],family_size=75978))[0]
    assert result['planning_status']=='U_STATISTIC_REQUIRES_ITS_SOURCE_CONCENTRATION_METHOD'
    assert result['scalar_source_variance_ceiling']==''
    assert result['sample_variance_over_optimistic_ceiling']==''


def test_source_plan_empirical_constant_is_not_zero_source_variance():
    row,stat=_source_plan_fixture('count_cdf','.01','probability',status='EMPIRICALLY_DEGENERATE_UNCERTAIN')
    stat['coupled_sample_variance']='0'
    result=list(source_variance_plan([row],[stat],family_size=75978))[0]
    assert result['scalar_source_variance_ceiling']>0
    assert result['planning_status']=='CALIBRATION_ONLY_NO_SOURCE_VARIANCE_CERTIFICATE'
    assert result['population_variance_certified'] is False


def test_source_plan_keeps_structural_and_undefined_rows():
    identity,a=_source_plan_fixture('paired_response','0','Hz',status='STRUCTURAL_ZERO_CONTRAST')
    undefined,b=_source_plan_fixture('field_lag_covariance','','Hz^2',index=1,status='UNDEFINED_NO_WITHIN_PHASE_PAIRS')
    b['coupled_signed_estimate']=''
    results=list(source_variance_plan([identity,undefined],[a,b],family_size=75978))
    assert [row['planning_status'] for row in results]==['SOURCE_IDENTITY','UNDEFINED_NO_WITHIN_PHASE_PAIRS']
    assert all(row['scalar_source_variance_ceiling']=='' for row in results)


@pytest.mark.parametrize('change',[
    lambda s:s.update(margin='1.0'),lambda s:s.update(coupled_original_independent_blocks='155'),
    lambda s:s.update(coupled_sample_variance='-1'),lambda s:s.update(coupled_score_kind='covariance_jackknife_pseudovalue')])
def test_source_plan_rejects_changed_contract(change):
    row,stat=_source_plan_fixture('source_response','1e-5','Hz')
    change(stat)
    with pytest.raises(ValueError):list(source_variance_plan([row],[stat],family_size=75978))


def test_source_plan_rejects_truncated_source_table():
    row,_=_source_plan_fixture('count_cdf','.01','probability')
    with pytest.raises(ValueError):list(source_variance_plan([row],[],family_size=75978))


def test_source_plan_does_not_expand_confirmation_ceiling():
    with pytest.raises(ValueError):list(source_variance_plan([],[],family_size=75978,blocks=129))


def _auxiliary_plan_fixture(between,within):
    margin,stat=_source_plan_fixture('paired_response','.01','Hz')
    row=list(source_variance_plan([margin],[stat],family_size=75978))[0]
    row['scalar_source_variance_ceiling']=.0001
    auxiliary=dict(margin,variance_units='Hz^2',original_independent_blocks='32',auxiliary_realizations='2',
        original_estimate='0',estimate_midpoint='0',midpoint_sample_variance=str(float(between)+float(within)/2),
        auxiliary_within_variance_estimate=within,between_block_variance_estimate=between)
    return row,auxiliary


def test_negative_auxiliary_between_estimate_is_not_clipped():
    row,auxiliary=_auxiliary_plan_fixture('-.0001','.0004')
    out=auxiliary_variance_point_plan(row,auxiliary,dict(rtol=1e-12,atol=1e-14))
    assert out['auxiliary_planning_status']=='NEGATIVE_BETWEEN_ESTIMATE_UNRESOLVED'
    assert out['between_root_variance_estimate']=='-.0001'
    assert out['point_auxiliary_realizations_for_variance_ceiling']==''


def test_auxiliary_point_scenario_retains_irreducible_between_component():
    row,auxiliary=_auxiliary_plan_fixture('.0002','.0004')
    out=auxiliary_variance_point_plan(row,auxiliary,dict(rtol=1e-12,atol=1e-14))
    assert out['auxiliary_planning_status']=='NO_FINITE_AUXILIARY_COUNT_IN_POINT_SCENARIO'
    assert out['point_auxiliary_realizations_for_variance_ceiling']==''


def test_auxiliary_point_count_uses_exact_decimal_ceiling():
    row,auxiliary=_auxiliary_plan_fixture('.00002','.0004')
    out=auxiliary_variance_point_plan(row,auxiliary,dict(rtol=1e-12,atol=1e-14))
    assert out['auxiliary_planning_status']=='FINITE_COUNT_IN_POINT_SCENARIO_ONLY'
    assert out['point_auxiliary_realizations_for_variance_ceiling']==5


def test_auxiliary_zero_components_do_not_certify_population_variance():
    row,auxiliary=_auxiliary_plan_fixture('0','0')
    out=auxiliary_variance_point_plan(row,auxiliary,dict(rtol=1e-12,atol=1e-14))
    assert out['auxiliary_planning_status']=='OBSERVED_ZERO_COMPONENTS_UNCERTAIN'


def test_auxiliary_planning_rejects_changed_decomposition():
    row,auxiliary=_auxiliary_plan_fixture('.00002','.0004')
    auxiliary['midpoint_sample_variance']='0'
    with pytest.raises(ValueError,match='DECOMPOSITION_CHANGED'):
        auxiliary_variance_point_plan(row,auxiliary,dict(rtol=1e-12,atol=1e-14))

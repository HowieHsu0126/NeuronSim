"""Saved-result identity joins, adverse retention and planning semantics."""
import csv
import json
import subprocess
import sys

import pytest

from . import pfc_coupled_budget_gaps as gaps


def record(index, family='field_mean', definition='ESTIMATED_EMPIRICAL_SPREAD', point=0., actual=(-2., 2.), future=(-2., 2.), current=None):
    identity = json.dumps(dict(left_cohort='numpy_dt_0p05', right_cohort='numpy_dt_0p025', phase='late_delay', fixture_index=index))
    base = dict(domain_row_index=str(index), family=family, endpoint_identity=identity, definition_status=definition)
    margin = dict(base, margin='1.0', units='Hz')
    if definition == gaps.STRUCTURAL:
        margin['margin']='0.0'; point=0.; actual=future=(0.,0.)
        status='SOURCE_IDENTITY'; actual_relation=future_relation=''; outside=False; ratio=''
    elif definition == gaps.UNDEFINED:
        margin['margin']=''; point=''; actual=future=('','')
        status='UNDEFINED'; actual_relation=future_relation=''; outside=''; ratio=''
    else:
        actual_relation=gaps.interval_relation(*actual,1.); future_relation=gaps.interval_relation(*future,1.)
        status=gaps.WITHIN if future_relation=='WITHIN_BUDGET' else gaps.EXCEEDS
        outside=abs(point)>1.; ratio=max(abs(x) for x in future)
    assessment=dict(base, margin=margin['margin'], calibration_estimate=str(point), planning_lower=str(future[0]), planning_upper=str(future[1]),
        planning_status=status, calibration_point_outside_margin=str(outside), planning_absolute_bound_over_margin=str(ratio),
        calibration_interval_relation=actual_relation, planning_interval_relation=future_relation)
    band=dict(base, estimate=str(point), lower=str(actual[0]), upper=str(actual[1]), channel='direct_contrast',
        planning_lower_root_n_and_zero_tail_at_128=str(future[0]), planning_upper_root_n_and_zero_tail_at_128=str(future[1]),
        current_empirical_status=current or definition, frozen_native_estimate=str(point))
    return margin,assessment,band


def joined(index, native_kwargs=None, coupled_kwargs=None, **common):
    margin,native,native_band=record(index,**common,**(native_kwargs or {}))
    _,coupled,coupled_band=record(index,**common,**(coupled_kwargs or {}))
    coupled_band['frozen_native_estimate']=native_band['estimate']
    return [margin,native,native_band,coupled,coupled_band]


@pytest.mark.parametrize('lo,hi,expected',[(-1,1,'WITHIN_BUDGET'),(1,2,'OVERLAPS_BUDGET'),(1.01,2,'DISJOINT_FROM_BUDGET'),(-2,-1.01,'DISJOINT_FROM_BUDGET'),(-2,2,'OVERLAPS_BUDGET')])
def test_interval_classification_retains_touching_boundaries(lo,hi,expected):
    assert gaps.interval_relation(lo,hi,1)==expected


def test_signed_gap_negative_is_slack_and_point_shift_is_separate():
    entries=joined(0,coupled_kwargs=dict(point=.2,actual=(-.7,1.1),future=(-.3,.7)))
    row=gaps.merge_row(0,*entries,.03)
    assert row['coupled_point_outside'] is False and row['coupled_projected_exceeds'] is False
    assert row['coupled_calibration_positive_side_gap']==pytest.approx(.1)
    assert row['coupled_projected_positive_side_gap']==pytest.approx(-.3)
    assert row['coupled_projected_negative_side_gap']==pytest.approx(-.7)
    assert row['coupled_projected_relation']=='WITHIN_BUDGET'
    assert 'scientific_fidelity_qualified' not in row


@pytest.mark.parametrize('change',['index','identity','family','definition','margin','native_estimate'])
def test_join_rejects_same_index_with_wrong_identity_or_definition(change):
    entries=joined(0)
    if change=='index':entries[3]['domain_row_index']='1'
    elif change=='native_estimate':entries[4]['frozen_native_estimate']='.1'
    elif change=='margin':entries[3]['margin']='2.0'
    else:entries[3][{'identity':'endpoint_identity','family':'family','definition':'definition_status'}[change]]='changed'
    with pytest.raises(ValueError):gaps.merge_row(0,*entries,.03)


def test_current_and_frozen_status_remain_distinct():
    entries=joined(0,definition='EMPIRICALLY_DEGENERATE_UNCERTAIN',coupled_kwargs=dict(current='ESTIMATED_EMPIRICAL_SPREAD'))
    row=gaps.merge_row(0,*entries,.03)
    assert row['definition_status']=='EMPIRICALLY_DEGENERATE_UNCERTAIN'
    assert row['coupled_current_empirical_status']=='ESTIMATED_EMPIRICAL_SPREAD'


def test_full_denominator_and_all_adverse_subsets_are_retained(tmp_path):
    rows=[joined(0,coupled_kwargs=dict(actual=(-.5,.5),future=(-.5,.5))),
          joined(1,family='source_response',native_kwargs=dict(actual=(-.5,.5),future=(-.5,.5)),
                 coupled_kwargs=dict(point=1.2,actual=(.7,1.7),future=(1.1,1.3))),
          joined(2,family='count_cdf'),joined(3,family='paired_response',definition=gaps.STRUCTURAL),
          joined(4,family='field_lag_covariance',definition=gaps.UNDEFINED)]
    tables=list(zip(*rows));report=gaps.summarize_tables(tables,5,.03,tmp_path)
    assert report['domain_rows']==5 and report['coupled_projected_exceedance_count']==2
    assert report['coupled_point_outside_count']==report['coupled_projected_disjoint_count']==1
    assert report['coupled_projected_disjoint_rows'][0]['domain_row_index']==1
    assert report['frozen_definition_status_counts'][gaps.UNDEFINED]==report['frozen_definition_status_counts'][gaps.STRUCTURAL]==1
    with (tmp_path/'coupled_projected_exceedances.csv').open() as f:
        assert [int(r['domain_row_index']) for r in csv.DictReader(f)]==[1,2]
    transitions={(r['native'],r['coupled']):r['count'] for r in report['endpoint_transitions']['planning_status']}
    assert transitions[gaps.WITHIN,gaps.EXCEEDS]==1 and transitions[gaps.EXCEEDS,gaps.WITHIN]==1
    assert sum(transitions.values())==5


def test_truncated_input_table_cannot_silently_reduce_denominator(tmp_path):
    tables=[list(v) for v in zip(joined(0),joined(1))];tables[-1].pop()
    with pytest.raises(ValueError,match='MISSING_OR_EXTRA'):
        gaps.summarize_tables(tables,2,.03,tmp_path)


def test_changed_saved_classification_rejected():
    entries=joined(0);entries[3]['planning_status']=gaps.WITHIN
    with pytest.raises(ValueError,match='CLASSIFICATION_CHANGED'):gaps.merge_row(0,*entries,.03)


def test_descriptive_tail_match_is_not_a_cause_label():
    entries=joined(0,family='count_cdf',coupled_kwargs=dict(actual=(-.1,.1),future=(-.03,.03)))
    entries[4]['channel']='monotone_primitive_cdf'
    row=gaps.merge_row(0,*entries,.03)
    assert row['coupled_cdf_projected_absolute_bound_matches_saved_tail_value'] is True
    assert row['coupled_cdf_uses_primitive_channel'] is True
    assert not any('caused' in key or 'tail_limited' in key for key in row)


def test_real_cli_cannot_promote_projection_to_qualified_result(tmp_path):
    config=tmp_path/'invalid.yaml'
    config.write_text('schema: pfc_coupled_budget_gap_diagnosis_v1\nscope: saved_calibration_result_classification\nconfirmation_execution_released: false\ncoverage_qualified: true\n')
    result=subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.pfc_coupled_budget_gaps','--config',str(config),
                           '--project-root',str(tmp_path),'--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode!=0 and 'SAVED_UNQUALIFIED_RESULTS_ONLY_REQUIRED' in result.stderr
    assert not (tmp_path/'out').exists()

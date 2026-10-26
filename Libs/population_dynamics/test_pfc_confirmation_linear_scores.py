"""Linear score orientation, block identity and structural-zero preservation."""
import json
import numpy as np
import pytest

from .pfc_confirmation_linear_scores import (
    source_metric_bounds, validate_responses, linear_block_score, recover_linears,
    CONDITIONS, CONTRASTS, METRICS, PHASES, STRUCTURAL, EMPIRICAL, run,
)


def data_fixture():
    seeds=[95001,95002,95003]
    labels=['coarse','fine']
    data=dict(condition_names=np.asarray(CONDITIONS),phase_names=np.asarray(list(PHASES)),
              metric_names=np.asarray(METRICS),source_rng_seeds=np.asarray(seeds))
    for label,base in zip(labels,[5.,7.]):
        data[label+'_values']=np.full((3,3,5,8),base)
    bounds=source_metric_bounds(np.array([600.]*16+[1000.]),np.array([5000]*16+[20000]))
    return data,labels,seeds,bounds


def test_source_rate_and_signed_component_physical_ranges():
    _,_,_,(lo,hi)=data_fixture()
    np.testing.assert_array_equal(lo,[0,0,0,0,0,-600,-600,-600])
    np.testing.assert_array_equal(hi,[680,600,1000,600,600,600,600,600])
    assert (hi-lo)[METRICS.index('cue_selectivity_hz')]*2==2400


def test_paired_difference_preserves_clock_condition_orientation_and_each_block():
    data,labels,seeds,bounds=data_fixture()
    mi=METRICS.index('all_unit_rate_hz');pi=list(PHASES).index('cue')
    data['coarse_values'][1,:,pi,mi]=[10,20,30]
    data['coarse_values'][0,:,pi,mi]=[3,5,7]
    data['fine_values'][1,:,pi,mi]=[4,8,12]
    data['fine_values'][0,:,pi,mi]=[2,3,4]
    responses=validate_responses(data,labels,seeds,bounds)
    ident=dict(left_cohort='coarse',right_cohort='fine',phase='cue',metric=METRICS[mi],contrast='cue_minus_omission')
    np.testing.assert_array_equal(linear_block_score('paired_response',ident,{},responses),[5,10,15])
    ident.update(left_cohort='fine',right_cohort='coarse')
    np.testing.assert_array_equal(linear_block_score('paired_response',ident,{},responses),[-5,-10,-15])


def test_pre_cue_identity_does_not_erase_clock_difference():
    data,labels,seeds,bounds=data_fixture();responses=validate_responses(data,labels,seeds,bounds)
    ident=dict(left_cohort='coarse',right_cohort='fine',phase='baseline',metric=METRICS[0],condition='cue',contrast='cue_minus_omission')
    np.testing.assert_array_equal(linear_block_score('source_response',ident,{},responses),[-2]*3)
    np.testing.assert_array_equal(linear_block_score('paired_response',ident,{},responses),[0]*3)


def test_true_unit_vector_components_are_used_without_angle_or_field_projection():
    data,labels,seeds,bounds=data_fixture();pi=list(PHASES).index('cue')
    data['coarse_values'][1,:,pi,6]=[1,0,-1]
    data['coarse_values'][1,:,pi,7]=[0,1,0]
    data['fine_values'][1,:,pi,6:8]=0
    responses=validate_responses(data,labels,seeds,bounds)
    ident=dict(left_cohort='coarse',right_cohort='fine',phase='cue',condition='cue',metric='vector_x_hz')
    np.testing.assert_array_equal(linear_block_score('source_response',ident,{},responses),[1,0,-1])
    ident['metric']='vector_y_hz'
    np.testing.assert_array_equal(linear_block_score('source_response',ident,{},responses),[0,1,0])


@pytest.mark.parametrize('axis',['source_rng_seeds','condition_names','metric_names'])
def test_reordered_axis_is_rejected_instead_of_silently_repairing_pairing(axis):
    data,labels,seeds,bounds=data_fixture();data[axis]=data[axis][::-1]
    with pytest.raises(ValueError,match='SOURCE_RESPONSE_AXIS_CHANGED'):
        validate_responses(data,labels,seeds,bounds)


def test_missing_block_and_contradicted_pre_cue_identity_fail():
    data,labels,seeds,bounds=data_fixture();data['coarse_values']=data['coarse_values'][:,:-1]
    with pytest.raises(ValueError,match='COMPLETE_FINITE_SOURCE_RESPONSES_REQUIRED'):
        validate_responses(data,labels,seeds,bounds)
    data,labels,seeds,bounds=data_fixture();data['coarse_values'][1,0,0,0]+=1
    with pytest.raises(ValueError,match='SOURCE_PRE_CUE_IDENTITY_FAILED'):
        validate_responses(data,labels,seeds,bounds)


def test_signed_metrics_are_allowed_but_physical_overflow_is_rejected():
    data,labels,seeds,bounds=data_fixture();data['coarse_values'][:,:,:,5:]=-600
    validate_responses(data,labels,seeds,bounds)
    data['coarse_values'][1,0,1,5]=-601
    with pytest.raises(ValueError,match='SOURCE_RESPONSE_OUTSIDE_PHYSICAL_RANGE'):
        validate_responses(data,labels,seeds,bounds)


def test_full_domain_keeps_empirical_zeros_uncertain_and_baseline_structural():
    data,labels,seeds,_=data_fixture()
    sizes=np.array([5000]*16+[20000]);cohorts={};rows=[]
    for label,value in zip(labels,[5.,7.]):
        cohorts[label]=dict(sizes=sizes,config=dict(excitatory_neurons=80000,inhibitory_neurons=20000),
                           rates={condition:np.full((3,1000,17),value) for condition in CONDITIONS})
    for field in range(17):
        rows.append(dict(family='count_cdf',endpoint_identity=json.dumps(dict(window_ms=5,field_index=field)),
                         physical_maximum_count=str(3 if field<16 else 5)))
    for family,names,metrics in [('field_mean',CONDITIONS,list(map(str,range(17)))),
                                 ('source_response',CONDITIONS,METRICS),
                                 ('paired_response',[x[0] for x in CONTRASTS],METRICS)]:
        for condition in names:
            for phase in PHASES:
                for metric in metrics:
                    identity=dict(left_cohort='coarse',right_cohort='fine',phase=phase)
                    identity['contrast' if family=='paired_response' else 'condition']=condition
                    identity['field_index' if family=='field_mean' else 'metric']=metric
                    structural=family=='paired_response' and phase=='baseline'
                    rows.append(dict(family=family,endpoint_identity=json.dumps(identity),
                                     estimate='0' if family=='paired_response' else '-2',
                                     definition_status=STRUCTURAL if structural else EMPIRICAL,
                                     reason='identical_pre_cue_inputs_and_initial_state' if structural else 'no_population_uncertainty_claim'))
    result,checks=recover_linears(rows,cohorts,data,labels,seeds,5.)
    assert checks['family_counts']==dict(field_mean=255,source_response=120,paired_response=120)
    assert checks['definition_counts'][STRUCTURAL]==24
    for j,index in enumerate(result['domain_row_indices']):
        row=rows[index]
        if row['definition_status']==STRUCTURAL:
            assert result['physical_lower_bound'][j]==result['physical_upper_bound'][j]==0
        else:
            assert result['physical_lower_bound'][j]<result['physical_upper_bound'][j]
    rows.append(dict(rows[-1]))
    with pytest.raises(ValueError,match='UNIQUE_FROZEN_LINEAR_ENDPOINT_REQUIRED'):
        recover_linears(rows,cohorts,data,labels,seeds,5.)


def test_invalid_scientific_scope_fails_before_reading_artifacts(tmp_path):
    config=tmp_path/'bad.yaml';config.write_text('schema: confirmation\n')
    with pytest.raises(ValueError,match='CALIBRATION_ONLY_LINEAR_RECOVERY_REQUIRED'):
        run(config,tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()

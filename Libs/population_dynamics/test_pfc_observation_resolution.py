"""Observation-bin identities, source control boundaries and terminal events."""
import numpy as np
import pytest
from .pfc_observation_resolution import event_summary,control_boundaries,directions,validate_identity


def test_exact_all_unit_mapping_and_terminal_events_remain_separate():
    ticks=np.array([0,79,80,399,400,799,800],dtype=np.int64)
    ids=np.array([0,1,2,3,4,0,4],dtype=np.int64)
    result=event_summary([(ticks[:3],ids[:3]),(ticks[3:],ids[3:])],5,duration_ticks=800)
    assert result['retained_events']==6 and result['terminal_events']==1
    np.testing.assert_array_equal(result['unit_counts'],[2,1,1,1,1])
    np.testing.assert_array_equal(result['terminal_unit_counts'],[0,0,0,0,1])
    np.testing.assert_array_equal(result['field_counts'][400],result['field_counts'][80].reshape(2,5,17).sum(axis=1))
    assert result['field_counts'][80][0,0]==1
    assert result['field_counts'][80][0,4]==1
    assert result['field_counts'][80][1,8]==1
    assert result['field_counts'][80][4,12]==1
    assert result['field_counts'][80][5,16]==1
    assert result['all_unit_integer_reaggregation_exact']


def test301ms_control_transition_is_exact1ms_edge_and_mixed5ms_bin():
    ticks=np.array([0,8000,24080],dtype=np.int64)
    values=np.array([[0.,0.,0.,0.],[.5,0.,1.,0.],[0.,0.,0.,0.]])
    fine=control_boundaries(ticks,values,80,400000)
    coarse=control_boundaries(ticks,values,400,400000)
    assert all(x['boundary_aligned'] for x in fine)
    assert coarse[1]==dict(transition_tick=24080,transition_ms=301.,boundary_aligned=False,
        mixed_bin=60,mixed_start_tick=24000,mixed_stop_tick=24400)


@pytest.mark.parametrize('ticks,ids',[
    (np.array([1.5]),np.array([0])),(np.array([-1]),np.array([0])),
    (np.array([801]),np.array([0])),(np.array([0]),np.array([5])),
    (np.array([0]),np.array([-1]))])
def test_noninteger_or_invalid_event_support_rejected(ticks,ids):
    with pytest.raises(ValueError,match='INTEGER_EVENT_SUPPORT'):
        event_summary([(ticks,ids)],5,duration_ticks=800)


def test_silent_and_cancelled_directions_remain_undefined():
    flags,angle=directions(np.array([0,2,3]),np.array([0.,0.,3.]),np.array([0.,0.,0.]))
    np.testing.assert_array_equal(flags,[False,False,True])
    assert np.isnan(angle[:2]).all() and angle[2]==0


def test_terminal_event_does_not_create_extra_observation_bin():
    result=event_summary([(np.array([800],dtype=np.int64),np.array([1],dtype=np.int64))],5,duration_ticks=800)
    assert result['field_counts'][80].shape==(10,17) and result['field_counts'][400].shape==(2,17)
    assert result['field_counts'][80].sum()==0 and result['terminal_events']==1


def test_control_clock_requires_initial_zero_and_increasing_native_ticks():
    with pytest.raises(ValueError,match='CONTROL_DOMAIN'):
        control_boundaries(np.array([1,80]),np.zeros((2,4)),80,800)
    with pytest.raises(ValueError,match='CONTROL_DOMAIN'):
        control_boundaries(np.array([0,80,80]),np.zeros((3,4)),80,800)


def test_training_and_calibration_identity_keeps_dataset_stream_namespaces():
    identity=dict(circuit_index=86,block_index=4,condition_index=3,split='calibration',
        episode_id='pfc_working_memory_circuit_086_block_004_condition_03',
        initial_state_seed=210086004,background_seed=220086004)
    validate_identity(identity,identity['episode_id'],'expanded')
    original=dict(identity,initial_state_seed=310086004,background_seed=320086004)
    validate_identity(original,original['episode_id'],'original')
    with pytest.raises(ValueError,match='STREAM_IDENTITY'):
        validate_identity(original,original['episode_id'],'expanded')
    with pytest.raises(ValueError,match='SOURCE_IDENTITY'):
        validate_identity(dict(identity,split='test'),identity['episode_id'],'expanded')


def test_validation_circuit_does_not_enter_bin_selection():
    identity=dict(circuit_index=74,block_index=4,condition_index=3,split='validation',
        episode_id='pfc_working_memory_circuit_074_block_004_condition_03',
        initial_state_seed=210074004,background_seed=220074004)
    with pytest.raises(ValueError,match='SOURCE_IDENTITY'):
        validate_identity(identity,identity['episode_id'],'expanded')

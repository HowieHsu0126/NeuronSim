"""Identity, multiplicity and exact-timestamp tests; no cortical simulation."""
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import v1_nest_input_replay as replay


def plan():
    return replay.prepare_inputs([
        ({'ids':np.array([1,0]),'spikes':[[.000021,.25,.25,2999.991119],[]]},[]),
        ({'ids':np.array([0]),'spikes':[[.25,2.]]},[])],3000.)


def events(p, ids):
    return dict(senders=ids[p['source_event_unit']].copy(),times=p['source_event_time_ms'].copy())


def test_shuffled_native_senders_overlap_and_silent_source_ids():
    p=plan();ids=np.array([81,3,22]);record=events(p,ids)
    record={k:v[::-1] for k,v in record.items()}
    result=replay.compare_events(p,ids,record)
    assert result['all_event_pairs_exact']
    assert result['mapped_generators']==3 and result['silent_source_generators']==1
    assert p['source_id'].tolist()==[1,0,0]
    assert p['population'].tolist()==[0,0,1]
    assert result['expected_events']==6 and result['observed_events']==6


def test_omitted_duplicate_and_cross_population_assignment_are_rejected():
    p=plan();ids=np.array([81,3,22]);record=events(p,ids)
    dropped={k:np.delete(v,2) for k,v in record.items()}
    assert not replay.compare_events(p,ids,dropped)['every_generator_count_exact']
    record['senders'][2]=22
    result=replay.compare_events(p,ids,record)
    assert not result['every_generator_count_exact'] and result['mismatched_generator_counts']==2


@pytest.mark.parametrize('change',['rounded_grid','truncated_grid','seconds','one_ulp'])
def test_time_differences_are_disclosed_without_fitted_tolerance(change):
    p=plan();ids=np.array([81,3,22]);record=events(p,ids)
    if change=='rounded_grid':record['times']=np.round(record['times']/.25)*.25
    elif change=='truncated_grid':record['times']=np.floor(record['times']/.25)*.25
    elif change=='seconds':record['times']/=1000
    else:record['times'][0]=np.nextafter(record['times'][0],np.inf)
    result=replay.compare_events(p,ids,record)
    assert result['every_generator_count_exact'] and not result['timestamps_exact']
    assert result['timestamp_mismatches']>0 and result['max_absolute_time_difference_ms']>0


def test_unknown_sender_and_duplicate_native_mapping_are_rejected():
    p=plan();ids=np.array([81,3,22]);record=events(p,ids);record['senders'][0]=999
    with pytest.raises(ValueError,match='UNKNOWN_SENDER'):replay.compare_events(p,ids,record)
    with pytest.raises(ValueError,match='MAPPING'):replay.compare_events(p,np.array([81,3,3]),record)


def test_stop_and_flush_are_separate_cumulative_observations():
    p=plan();ids=np.array([81,3,22]);complete=events(p,ids)
    early={
        k:v[complete['times']<2999.] for k,v in complete.items()}
    before=replay.compare_events(p,ids,early);after=replay.compare_events(p,ids,complete)
    assert before['observed_events']==5 and not before['all_event_pairs_exact']
    assert after['observed_events']==6 and after['all_event_pairs_exact']
    combined={k:np.concatenate([early[k],complete[k]]) for k in complete}
    assert not replay.compare_events(p,ids,combined)['every_generator_count_exact']


@pytest.mark.parametrize('bad',[[0.],[-1.],[3000.01],[np.nan]])
def test_invalid_source_times_never_reach_native_setter(bad):
    with pytest.raises(ValueError,match='TIME'):
        replay.prepare_inputs([({'ids':np.array([0]),'spikes':[bad]},[])],3000.)


def test_pinned_pointnet_function_executes_and_preserves_original_trains(tmp_path):
    source=Path(os.environ.get('NEURONSIM_POINTNET_SOURCE',str(Path(__file__).resolve().parents[2]/
        'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/pointnet_source/run_584798')))
    class Native:
        calls=[]
        @classmethod
        def SetStatus(cls,node,parameters):cls.calls.append((node,parameters))
    setter=replay.load_pointnet_setter(source,Native)
    train=np.array([1.1,.000021,1.1]);original=train.copy()
    setter(5,'native-node',replay.SpikeTrains(train),t_offset=0.)
    np.testing.assert_array_equal(train,original)
    np.testing.assert_array_equal(Native.calls[0][1]['spike_times'],[.000021,1.1,1.1])
    setter(5,'silent-node',replay.SpikeTrains(np.array([])))
    assert len(Native.calls)==1
    with pytest.raises(ValueError):setter(5,'bad-node',replay.SpikeTrains(np.array([0.])))
    staged=tmp_path/'source';staged.mkdir()
    for relative in ['delivery.json','source_inspection.json','source/bmtk/simulator/pointnet/pointnetwork.py',
                     'source/bmtk/simulator/pointnet/pointsimulator.py']:
        target=staged/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,target)
    with (staged/'source/bmtk/simulator/pointnet/pointnetwork.py').open('a') as stream:stream.write('\n# changed\n')
    with pytest.raises(ValueError,match='IMPLEMENTATION_CHANGED'):
        replay.load_pointnet_setter(staged,Native)

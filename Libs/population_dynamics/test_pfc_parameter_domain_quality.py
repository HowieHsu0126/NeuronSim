"""Full-domain recipes preserve identities and refuse missing exact proofs."""
from itertools import product
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import pfc_parameter_domain_quality as domain


def plan():
    rows=[]
    for c,b,k in product(range(100),domain.BLOCKS,range(10)):
        circuit=f"pfc_working_memory_circuit_{c:03d}";block=circuit+f"_block_{b:03d}"
        rows.append(dict(task_index=len(rows),identity=dict(circuit_index=c,block_index=b,condition_index=k,
            circuit_id=circuit,block_id=block,episode_id=block+f"_condition_{k:02d}",split=domain.intended_split(c),
            connectivity_seed=200000000+c, initial_state_seed=210000000+c*1000+b,
            background_seed=220000000+c*1000+b,stimulus_seed=230000000+c*1000+b)))
    return dict(schema=domain.SCHEMA,scope="development",block_indices=list(domain.BLOCKS),
        planned_complete_episodes=3000,complete_episodes=0,production_episodes=0,
        source_quality_qualified=False,task_sampling_qualified=False,scientific_claim_qualified=False,
        production_eligible=False,circuit_split_counts=domain.SPLITS,rows=rows)


def test_honest3000_recipes_are_not_completed_or_qualified():
    value=plan();domain.validate_plan(value)
    assert len(value['rows'])==3000
    assert len({r['identity']['episode_id'] for r in value['rows']})==3000
    assert value['complete_episodes']==value['production_episodes']==0
    assert {s:sum(r['identity']['split']==s for r in value['rows']) for s in domain.SPLITS}=={
        'train':2100,'validation':300,'calibration':300,'test':300}


@pytest.mark.parametrize('change', ['duplicate','split','count','qualified'])
def test_identity_denominator_or_qualification_change_rejected(change):
    value=plan()
    if change=='duplicate':value['rows'][1]['identity']=copy.deepcopy(value['rows'][0]['identity'])
    elif change=='split':value['rows'][2100]['identity']['split']='train'
    elif change=='count':value['rows'].pop()
    else:value['task_sampling_qualified']=True
    with pytest.raises(ValueError):domain.validate_plan(value)


def test_source_refuses_missing_completeproof_before_recipe_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(domain,'load_task',lambda *a:pytest.fail('missing mapping read recipe'))
    monkeypatch.setattr(domain.subprocess,'run',lambda *a,**k:pytest.fail('missing mapping started source'))
    output=tmp_path/'attempt'
    with pytest.raises(ValueError,match='COMPLETE100_PROOF_MAPPING_REQUIRED'):
        domain.source_task(tmp_path,tmp_path/'prepared',0,tmp_path/'missing.json',output)
    assert not output.exists()


@pytest.mark.parametrize('change',[{'excitatory_neurons':8000},{'dt_ms':.025},{'duration_ms':100.}])
def test_source_rejects_reduced_scale_clock_or_duration(change):
    config=dict(scope='source_reproduction',dt_ms=.0125,duration_ms=5000.,excitatory_neurons=80000,
        inhibitory_neurons=20000,readout_bin_ms=5.,random_stream_policy='independent_initial_external')
    config.update(change)
    with pytest.raises(ValueError,match='COMPLETE100K_REFERENCE_SOURCE_REQUIRED'):
        domain.validate_source_config(config,{})


def test_default5ms_readout_remains_supported(tmp_path,monkeypatch):
    config=tmp_path/'config.yaml';row={'identity':{'episode_id':'fixture'}}
    monkeypatch.setattr(domain,'load_task',lambda *a:({},row,config))
    calls=[]
    monkeypatch.setattr(domain,'readout_episode',lambda *a:(calls.append(a) or {'status':'fixture'}))
    result=domain.readout_task(tmp_path,0,tmp_path/'source',tmp_path/'output')
    assert result=={'status':'fixture'} and calls==[(config,tmp_path/'source',tmp_path/'output')]


def test1ms_option_requires_frozen_map_before_export(tmp_path,monkeypatch):
    monkeypatch.setattr(domain,'load_task',lambda *a:({}, {},tmp_path/'config.yaml'))
    monkeypatch.setattr(domain,'readout_episode',lambda *a:pytest.fail('invalid map started export'))
    with pytest.raises(ValueError,match='OBSERVATION_CAMPAIGN_REQUIRED'):
        domain.readout_task(tmp_path,0,tmp_path/'source',tmp_path/'output',observation_bin_ms=1.)
    def reject(*args):raise ValueError('PFC_DOMAIN_FROZEN1MS_OBSERVATION_MAP_REQUIRED')
    monkeypatch.setattr(domain,'frozen_observation_map',reject)
    with pytest.raises(ValueError,match='FROZEN1MS_OBSERVATION_MAP_REQUIRED'):
        domain.readout_task(tmp_path,0,tmp_path/'source',tmp_path/'output',observation_bin_ms=1.,campaign_path=tmp_path/'campaign.yaml')
    assert not (tmp_path/'output').exists()


def test1ms_view_reuses_existing_integer_kernel_and_retains_terminal_events():
    import numpy as np
    ticks=np.asarray([0,79,80,399,400,399999,400000],dtype=np.int64)
    ids=np.asarray([0,79999,80000,10,99999,123,500],dtype=np.int64)
    result=domain.event_summary([(ticks,ids)],100000)
    assert result['field_counts'][80].shape==(5000,17)
    np.testing.assert_array_equal(result['field_counts'][80].reshape(1000,5,17).sum(axis=1),result['field_counts'][400])
    assert result['retained_events']==6 and result['terminal_events']==1
    assert result['unit_counts'].sum()==6 and result['terminal_unit_counts'].sum()==1


def test1ms_newdomain_readout_dispatch_preserves_recipe_and_scope(tmp_path,monkeypatch):
    row=plan()['rows'][2999];row['condition']={'condition_id':'omission_late_off_target'}
    config=tmp_path/'recipe.yaml';config.write_text('immutable recipe')
    monkeypatch.setattr(domain,'load_task',lambda *a:({},row,config))
    monkeypatch.setattr(domain,'frozen_observation_map',lambda *a:{'frozen':True})
    calls=[]
    monkeypatch.setattr(domain,'readout_episode',lambda *a:(calls.append(a) or {}))
    def observation(directory,path,actual,campaign,output):
        assert actual['identity']['circuit_index']==99 and actual['identity']['block_index']==8
        assert actual['identity']['split']=='test' and path==config
        return {'status':'DESCRIBED_FROZEN1MS_SOURCE_OBSERVATION','production_episodes':0}
    monkeypatch.setattr(domain,'observation_readout',observation)
    result=domain.readout_task(tmp_path,2999,tmp_path/'source',tmp_path/'output',observation_bin_ms=1.,campaign_path=tmp_path/'campaign.yaml')
    assert result['production_episodes']==0 and config.read_text()=='immutable recipe'
    assert calls==[(config,tmp_path/'source',tmp_path/'output/development5ms')]
    (tmp_path/'output/development5ms').mkdir(parents=True)
    # Recovery consumes the previously completed export; it never starts a
    # second export after a later observation failure.
    result=domain.readout_task(tmp_path,2999,tmp_path/'source',tmp_path/'output',observation_bin_ms=1.,campaign_path=tmp_path/'campaign.yaml')
    assert result['production_episodes']==0 and len(calls)==1

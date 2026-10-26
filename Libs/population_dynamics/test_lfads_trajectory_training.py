"""CPU full-source architecture tests with an explicit tiny cohort boundary."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import lfads_trajectory_training as training
from .lfads import LFADSConditionalSimulator
from .provenance import sha256_file


CONFIG = Path(__file__).resolve().parents[1] / 'configs/neural_population_lfads_trajectory.yaml'


class TinyCohort:
    """Only the data boundary/size is substituted; production gates stay intact."""
    marker = 'tiny_source_fixture'
    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id='fit_a'),dict(case_id='fit_b')]
        self.validation = [dict(case_id='selection_a')]
        self.unit_ids = np.array([3,7,12])
        self.history_bins, self.horizon_bins, self.bin_seconds = 2,4,.005
        self.binding = dict(acquisition_design_sha256='a'*64,dataset_sha256='b'*64,
                            native_source_identity=dict(fixture=self.marker))
    def window(self, case):
        counts = np.array([[1,0,0],[0,1,0],[1,0,1],[0,1,0],[0,0,1],[1,1,0]],dtype=np.float32)
        if case['case_id'] == 'fit_b': counts = counts[::-1].copy()
        stimulus = np.array([[0.],[1.],[0.],[1.],[0.],[1.]],dtype=np.float32)
        return counts, stimulus
    def verify_model(self, model):
        parameters = sum(int(np.prod(v.shape.as_list())) for v in model._full.model.train_vars)
        # Qualified architecture: 401 parameters per observed unit + 30,752.
        assert parameters == 401 * 3 + 30752
        return parameters


@pytest.fixture
def setup(tmp_path,monkeypatch):
    config = training.yaml.safe_load(CONFIG.read_text()); config['device']='cpu'
    path=tmp_path/'config.yaml'; path.write_text(training.yaml.safe_dump(config))
    monkeypatch.setattr(training,'TrajectoryCohort',TinyCohort)
    return config,path


def same_values(a,b):
    assert a.keys()==b.keys()
    for key in a: np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def checkpoint(output):
    directory=output/'latest'
    state=json.loads((directory/'training.json').read_text())
    with np.load(directory/'weights.npz',allow_pickle=False) as archive:
        values={key:archive[key] for key in archive.files}
    return state,values


def snapshot(path):
    return {str(p.relative_to(path)):sha256_file(p) for p in path.rglob('*') if p.is_file() and not p.is_symlink()}


def test_validation_namespace_is_separate_and_next_training_update_unchanged(setup,tmp_path):
    config,_=setup; counts,stimulus=TinyCohort(None,None,None).window(dict(case_id='fit_a'))
    options=dict(unit_ids=[3,7,12],input_size=1,history_bins=2,horizon_bins=4,bin_seconds=.005,
                 model_config={**config['model'],'keep_prob':.65},initialization_seed=41)
    with LFADSConditionalSimulator(**options) as reference, LFADSConditionalSimulator(**options) as evaluated:
        same_values(reference._full.values(),evaluated._full.values())
        assert reference.fit_step(counts[None],stimulus[None]) == evaluated.fit_step(counts[None],stimulus[None])
        before=evaluated._full.values(); sites=list(evaluated._full.randomness.sites)
        first=evaluated.validation_loss(counts[None],stimulus[None],validation_seed=0)
        assert first==evaluated.validation_loss(counts[None],stimulus[None],validation_seed=0)
        other=evaluated.validation_loss(counts[None],stimulus[None],validation_seed=2**63-1)
        assert first!=other
        assert evaluated.training_updates==reference.training_updates==1
        assert evaluated._full.randomness.sites==sites
        same_values(before,evaluated._full.values())
        assert reference.fit_step(counts[None],stimulus[None]) == evaluated.fit_step(counts[None],stimulus[None])
        same_values(reference._full.values(),evaluated._full.values())
        saved=tmp_path/'adapter_checkpoint'; evaluated.save(saved)
        with LFADSConditionalSimulator.load(saved) as restored:
            assert not restored.loaded_for_inference
            assert evaluated.validation_loss(counts[None],stimulus[None],validation_seed=92)==restored.validation_loss(counts[None],stimulus[None],validation_seed=92)
            assert evaluated.fit_step(counts[None],stimulus[None])==restored.fit_step(counts[None],stimulus[None])
            same_values(evaluated._full.values(),restored._full.values())
        for seed in (-1,2**63,True,None):
            with pytest.raises(ValueError,match='INT63'):
                evaluated.validation_loss(counts[None],stimulus[None],validation_seed=seed)
        for counter in (-1,2**63-1):
            evaluated.training_updates=counter
            with pytest.raises(ValueError,match='TRAINING_COUNTER'):
                evaluated.fit_step(counts[None],stimulus[None])


def test_full_runner_split_resume_preserves_model_adam_rng_history_and_logs(tmp_path,setup):
    _,config=setup
    whole,split=tmp_path/'whole',tmp_path/'split'
    training.train(tmp_path,tmp_path,config,whole,41,stop_after_epochs=3)
    first=training.train(tmp_path,tmp_path,config,split,41,stop_after_epochs=1)
    assert first['status']=='CHECKPOINTED_DEVELOPMENT' and first['training_updates']==2
    training.train(tmp_path,tmp_path,config,split,41,resume=True,stop_after_epochs=2)
    a,av=checkpoint(whole); b,bv=checkpoint(split)
    same_values(av,bv)
    for name in ('next_epoch','history','best','best_epoch','stale','training_updates'):
        assert a[name]==b[name]
    assert b['next_epoch']==3 and b['training_updates']==6
    with (split/'validation.csv').open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==12
    assert [row['validation_seed'] for row in rows[:4]]==[row['validation_seed'] for row in rows[4:8]]
    assert all(int(row['validation_counter'])==-(int(row['validation_seed'])+1)<0 for row in rows)
    assert (split/'latest').is_symlink() and (split/'best').is_symlink()
    retained={str(p.relative_to(split)) for p in (split/'checkpoints').iterdir()}
    assert retained=={str((split/'latest').readlink()),str((split/'best').readlink())}
    assert json.loads((split/'epochs.json').read_text())==b['history']
    assert (split/'attempts/attempt_0000.json').exists()
    assert (split/'attempts/attempt_0001.json').exists()
    counts,stimulus=TinyCohort(None,None,None).window(dict(case_id='fit_a'))
    with LFADSConditionalSimulator.load(whole/'latest') as left, LFADSConditionalSimulator.load(split/'latest') as right:
        assert left.fit_step(counts[None],stimulus[None])==right.fit_step(counts[None],stimulus[None])
        same_values(left._full.values(),right._full.values())


@pytest.mark.parametrize('change',['config','data','weights'])
def test_resume_binding_or_checkpoint_change_rejected_without_output_mutation(change,tmp_path,setup,monkeypatch):
    values,config=setup; output=tmp_path/'run'
    training.train(tmp_path,tmp_path,config,output,41,stop_after_epochs=1)
    if change=='config':
        values['validation_seed_base']+=1; config.write_text(training.yaml.safe_dump(values))
    elif change=='data': monkeypatch.setattr(TinyCohort,'marker','other_source')
    else:
        with (output/'latest/weights.npz').open('ab') as stream: stream.write(b'changed')
    before=snapshot(output)
    with pytest.raises(ValueError,match='changed'):
        training.train(tmp_path,tmp_path,config,output,41,resume=True)
    assert snapshot(output)==before


def test_partial_epoch_rolls_back_to_committed_model_and_preserves_failure(tmp_path,setup,monkeypatch):
    _,config=setup; baseline,failed=tmp_path/'baseline',tmp_path/'failed'
    training.train(tmp_path,tmp_path,config,baseline,41,stop_after_epochs=2)
    original=LFADSConditionalSimulator.fit_step; calls=0
    def fail_second(self,*args):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('injected partial epoch failure')
        return original(self,*args)
    monkeypatch.setattr(LFADSConditionalSimulator,'fit_step',fail_second)
    with pytest.raises(RuntimeError,match='partial epoch'):
        training.train(tmp_path,tmp_path,config,failed,41,stop_after_epochs=2)
    assert checkpoint(failed)[0]['next_epoch']==0
    assert json.loads((failed/'run.json').read_text())['status']=='FAILED'
    monkeypatch.setattr(LFADSConditionalSimulator,'fit_step',original)
    training.train(tmp_path,tmp_path,config,failed,41,resume=True,stop_after_epochs=2)
    a,av=checkpoint(baseline); b,bv=checkpoint(failed)
    assert a['history']==b['history']; same_values(av,bv)
    assert json.loads((failed/'attempts/attempt_0000.json').read_text())['status']=='FAILED'


def test_interrupted_best_alias_recovers_committed_epoch(tmp_path,setup,monkeypatch):
    _,config=setup; output=tmp_path/'run'; original=training._alias
    def fail_best(root,name,target):
        if name=='best': raise OSError('injected alias interruption')
        return original(root,name,target)
    monkeypatch.setattr(training,'_alias',fail_best)
    with pytest.raises(OSError,match='alias interruption'):
        training.train(tmp_path,tmp_path,config,output,41,stop_after_epochs=1)
    assert checkpoint(output)[0]['next_epoch']==1
    monkeypatch.setattr(training,'_alias',original)
    result=training.train(tmp_path,tmp_path,config,output,41,resume=True,stop_after_epochs=1)
    assert result['completed_epochs']==2
    state,_=checkpoint(output)
    assert str((output/'best').readlink())==state['best_checkpoint']


def test_production_configuration_rejects_reduced_units_lengths_or_capacity():
    config=training.yaml.safe_load(CONFIG.read_text()); config['device']='cpu'
    for key,value in [('expected_units',3),('history_bins',2),('horizon_bins',4),('maximum_epochs',1)]:
        with pytest.raises(ValueError,match='full-episode'):
            training.validate_config({**config,key:value},41)
    wrong=copy.deepcopy(config); wrong['model']['gen_dim']=4
    with pytest.raises(ValueError,match='full-capacity'):
        training.validate_config(wrong,41)
    with pytest.raises(ValueError,match='registered training seed'):
        training.validate_config(config,44)


def test_production_cohort_does_not_accept_partial_partition(tmp_path):
    config=training.yaml.safe_load(CONFIG.read_text()); config['device']='cpu'
    design=dict(acquired_split='train',cases=[dict(case_id=f'c{b}',block=b,split='train',family='trajectory',template_id='familiar_0') for b in range(18)])
    (tmp_path/'design.json').write_text(json.dumps(design))
    (tmp_path/'dataset.json').write_text('{}'); (tmp_path/'acquisition.json').write_text('{}')
    with pytest.raises(ValueError,match='56-fit/16-selection'):
        training.TrajectoryCohort(tmp_path,tmp_path,config)


def test_cuda_flag_requirement_and_cli_forwarding(tmp_path,setup,monkeypatch,capsys):
    config,_=setup; monkeypatch.delenv('NVIDIA_TF32_OVERRIDE',raising=False)
    with pytest.raises(ValueError,match='TF32_OVERRIDE=0'):
        training.validate_config({**config,'device':'cuda'},41)
    captured={}
    def fake_train(*args,**kwargs):
        captured.update(args=args,kwargs=kwargs); return {'status':'CLI_ARGUMENTS_ONLY'}
    monkeypatch.setattr(training,'train',fake_train)
    monkeypatch.setattr(sys,'argv',['lfads','--acquisition','a','--fields','f','--config','c','--output','o','--seed','43','--resume','--stop-after-epochs','2'])
    training.main()
    assert captured['args'][-2:]==(43,True) and captured['kwargs']==dict(stop_after_epochs=2,resume_from=None,total_epochs=None,parent_checkpoint_sha256=None)
    assert json.loads(capsys.readouterr().out)['status']=='CLI_ARGUMENTS_ONLY'
    monkeypatch.setattr(sys,'argv',['lfads','--seed','44'])
    with pytest.raises(SystemExit): training.main()


def test_full_episode_gate_rejects_short_windows_or_unit_subsets():
    cohort=object.__new__(training.TrajectoryCohort)
    cohort.unit_ids=np.arange(100000)
    cohort.unit_field_index=np.zeros(100000,dtype=np.int64)
    cohort.field_ids=np.array(['field'])
    data=SimpleNamespace(unit_ids=cohort.unit_ids,unit_field_index=cohort.unit_field_index,
                         field_ids=cohort.field_ids,bin_ms=5.,dt_ms=.1,bin_ticks=50,duration_ticks=1000)
    with pytest.raises(ValueError,match='1200-bin/100000-unit'):
        cohort._verify_episode(data)
    data.duration_ticks=60000
    data.unit_ids=data.unit_ids[:3]; data.unit_field_index=data.unit_field_index[:3]
    with pytest.raises(ValueError,match='1200-bin/100000-unit'):
        cohort._verify_episode(data)


@pytest.mark.skipif(training.os.environ.get('CUDA_VISIBLE_DEVICES')!='',reason='requires explicit CPU-only helper')
def test_unavailable_cuda_is_a_failed_stage_receipt_not_cpu_fallback(tmp_path,setup,monkeypatch):
    config,path=setup
    config['device']='cuda'; path.write_text(training.yaml.safe_dump(config))
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE','0')
    output=tmp_path/'cuda_unavailable'
    with pytest.raises(RuntimeError,match='LFADS_CUDA_REQUESTED_BUT_NO_VISIBLE_GPU'):
        training.train(tmp_path,tmp_path,path,output,41,stop_after_epochs=1)
    report=json.loads((output/'run.json').read_text())
    assert report['status']=='FAILED' and report['phase']=='graph_construction'
    assert report['completed_epochs']==0 and not (output/'latest').exists()
    assert not report['generative_fidelity_qualified']


def test_budget_descendant_matches_uninterrupted_updates_and_preserves_parent(tmp_path, setup):
    _, config = setup
    whole, parent, child = [tmp_path / name for name in ('whole_budget', 'parent_budget', 'child_budget')]
    training.train(tmp_path, tmp_path, config, whole, 41, stop_after_epochs=3)
    prior = training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=1)
    before = snapshot(parent)
    declared = sha256_file(parent / 'latest/training.json')
    result = training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent,
                            total_epochs=21, parent_checkpoint_sha256=declared, stop_after_epochs=2)
    a, av = checkpoint(whole); b, bv = checkpoint(child)
    same_values(av, bv)
    for key in ('next_epoch', 'history', 'best', 'best_epoch', 'stale', 'training_updates'):
        assert a[key] == b[key]
    assert snapshot(parent) == before
    assert result['binding']['config']['maximum_epochs'] == 20
    assert result['effective_maximum_epochs'] == 21 and result['completed_epochs'] == 3
    assert result['continuation']['parent_checkpoint_sha256'] == declared
    assert result['continuation']['parent_completed_epochs'] == prior['completed_epochs'] == 1
    assert result['continuation']['treatment'] == 'optimization_budget_only'
    assert result['convergence_established'] is False and result['formal_training_completed'] is False
    assert b['continuation_sha256'] == sha256_file(child / 'continuation.json')
    assert json.loads((child / 'epochs.json').read_text()) == b['history']
    # Ordinary resume inherits the declared ceiling; no second continuation request.
    training.train(tmp_path, tmp_path, config, whole, 41, resume=True, stop_after_epochs=1)
    training.train(tmp_path, tmp_path, config, child, 41, resume=True, stop_after_epochs=1)
    a, av = checkpoint(whole); b, bv = checkpoint(child)
    assert a['history'] == b['history']
    same_values(av, bv)
    assert snapshot(parent) == before


def test_interrupted_descendant_restores_imported_adam_rng_and_history(tmp_path, setup, monkeypatch):
    _, config = setup
    parent, reference, failed = [tmp_path / name for name in ('resume_parent', 'resume_reference', 'resume_failed')]
    training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=1)
    training.train(tmp_path, tmp_path, config, reference, 41, resume_from=parent, total_epochs=21, stop_after_epochs=2)
    before = snapshot(parent)
    original = LFADSConditionalSimulator.fit_step
    calls = 0
    def fail_after_one_update(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected continuation partial epoch')
        return original(self, *args)
    monkeypatch.setattr(LFADSConditionalSimulator, 'fit_step', fail_after_one_update)
    with pytest.raises(RuntimeError, match='continuation partial epoch'):
        training.train(tmp_path, tmp_path, config, failed, 41, resume_from=parent, total_epochs=21, stop_after_epochs=2)
    state, _ = checkpoint(failed)
    assert state['next_epoch'] == 1 and state['training_updates'] == 2
    assert state['inherited_parent_checkpoint']['binding'] == json.loads((parent/'run.json').read_text())['binding']
    monkeypatch.setattr(LFADSConditionalSimulator, 'fit_step', original)
    training.train(tmp_path, tmp_path, config, failed, 41, resume=True, stop_after_epochs=2)
    a, av = checkpoint(reference); b, bv = checkpoint(failed)
    assert a['history'] == b['history']
    same_values(av, bv)
    assert snapshot(parent) == before
    assert json.loads((failed/'attempts/attempt_0000.json').read_text())['status'] == 'FAILED'


def test_continuation_rejects_changed_parent_budget_and_lineage_before_updates(tmp_path, setup, monkeypatch):
    values, config = setup
    parent = tmp_path / 'guarded_parent'
    training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=1)
    before = snapshot(parent)
    child = tmp_path / 'guarded_child'
    for options, match in [
            (dict(total_epochs=20), 'must exceed'),
            (dict(total_epochs=21, parent_checkpoint_sha256='0'*64), 'checksum differs')]:
        with pytest.raises(ValueError, match=match):
            training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent, **options)
        assert not child.exists()
    monkeypatch.setattr(TinyCohort, 'marker', 'changed_source_fixture')
    with pytest.raises(ValueError, match='binding changed'):
        training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent, total_epochs=21)
    monkeypatch.setattr(TinyCohort, 'marker', 'tiny_source_fixture')
    training.train(tmp_path, tmp_path, config, child, 41, resume_from=parent, total_epochs=21, stop_after_epochs=1)
    manifest = json.loads((child/'continuation.json').read_text())
    manifest['total_epochs'] = 22
    (child/'continuation.json').write_text(json.dumps(manifest))
    child_before = snapshot(child)
    with pytest.raises(ValueError, match='lineage'):
        training.train(tmp_path, tmp_path, config, child, 41, resume=True, stop_after_epochs=1)
    assert snapshot(child) == child_before and snapshot(parent) == before


def test_only_pinned_trainer_transition_is_admitted(setup):
    values, _ = setup
    current = training._binding(TinyCohort(None, None, None), values, 41)
    parent = copy.deepcopy(current)
    parent['sources']['lfads_trajectory_training.py'] = training.LEGACY_TRAINER_SHA256
    assert set(training._source_transition(parent, current)) == {'lfads_trajectory_training.py'}
    assert training._source_transition(current, current) == {}
    for key in ('lfads.py', 'external/lfads.py', 'poisson_trajectory_training.py', 'lfads_trajectory_training.py'):
        changed = copy.deepcopy(parent)
        changed['sources'][key] = '0'*64
        with pytest.raises(ValueError, match='binding changed'):
            training._source_transition(changed, current)


def test_execution_options_require_explicit_new_descendant(tmp_path):
    parent = tmp_path/'parent'
    for resume, source, budget, digest in [(True, parent, 21, None), (False, None, 21, None),
            (False, parent, None, None), (False, parent, True, None)]:
        with pytest.raises(ValueError):
            training._execution_options(tmp_path/'child', resume, source, budget, digest, None)
    for target in (parent, parent/'nested_child'):
        with pytest.raises(ValueError, match='must not modify'):
            training._execution_options(target, False, parent, 21, None, None)
    (tmp_path/'existing').mkdir()
    with pytest.raises(FileExistsError):
        training._execution_options(tmp_path/'existing', False, parent, 21, None, None)


def test_continuation_cli_forwards_explicit_ceiling_and_parent(tmp_path, monkeypatch, capsys):
    captured = {}
    def fake_train(*args, **kwargs):
        captured.update(kwargs); return {'status':'CLI_ARGUMENTS_ONLY'}
    monkeypatch.setattr(training, 'train', fake_train)
    monkeypatch.setattr(sys, 'argv', ['lfads','--acquisition','a','--fields','f','--config','c',
        '--output','o','--seed','41','--resume-from','parent','--total-epochs','80',
        '--parent-checkpoint-sha256','a'*64])
    training.main()
    assert captured == dict(resume_from=Path('parent'),total_epochs=80,parent_checkpoint_sha256='a'*64,stop_after_epochs=None)


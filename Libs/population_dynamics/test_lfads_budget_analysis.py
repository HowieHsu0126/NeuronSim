"""Synthetic paired metadata fixtures; these tests do not establish LFADS fidelity.

The existing three-unit evaluator/replay fixture supplies actual metric tables.
Only metadata are replicated across training seeds. Complete parent/child
checkpoint lineages exercise the real continuation verification helpers.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import lfads_budget_analysis as analysis
from . import lfads_trajectory_evaluation as evaluator
from . import lfads_trajectory_training as training
from .provenance import sha256_file
from .trajectory_forecast_analysis import replay_scores
from .test_lfads_forecast_analysis import replicated
from .test_lfads_trajectory_evaluation import cohort_fixture, dump


def rewrite_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bind_evaluation(folder, train_root, seed, migration):
    run, _, _, selected_dir, selected = training._verified_run(train_root, 56)
    ev = analysis.read(folder / 'evaluation.json')
    ev.update(training_seed=seed, training_binding=run['binding'],
        training_run_sha256=sha256_file(train_root / 'run.json'),
        selected_checkpoint=run['best_checkpoint'],
        selected_training_sha256=sha256_file(selected_dir / 'training.json'),
        selected_model_artifacts=selected['model_artifacts'],
        completed_training_epochs=run['completed_epochs'], selected_epoch=run['best_epoch'],
        training_stop_reason=run['stop_reason'], inference_migration=migration,
        native_source_identity=dict(run['binding']['native_source_identity'],
            gpu_source_revision='synthetic-source-revision', dt_ms=.1))
    if run.get('continuation'):
        ev.update(effective_training_epoch_budget=run['effective_maximum_epochs'],
                  training_continuation=run['continuation'])
    else:
        # The immutable original receipts predate these optional fields.
        ev.pop('effective_training_epoch_budget', None)
        ev.pop('training_continuation', None)
    dump(folder / 'evaluation.json', ev)


def bind_replay(root, seed, repair_seeds=False):
    folder = root / f'seed_{seed}_forecast'
    path = root / f'seed_{seed}_replay.json'
    csv_path = path.with_suffix('.csv')
    if repair_seeds:
        rows = analysis.csv_rows(csv_path)
        cases = {c['case_id']: i for i, c in enumerate(analysis.CASES)}
        for row in rows:
            origin = evaluator.MATCHED_PROTOCOL['origins_ms'].index(float(row['origin_ms']))
            base = 86000 + seed * 100000 + cases[row['case_id']] * 100 + origin * 3
            row.update(training_seed=seed, initial_seed=base, dynamics_seed=base+1,
                       observation_seed=base+2)
        rewrite_csv(csv_path, rows)
    ev = analysis.read(folder / 'evaluation.json')
    replay = analysis.read(path)
    replay.update(training_seed=seed, evaluation_sha256=sha256_file(folder / 'evaluation.json'),
                  scores_sha256=ev['scores_sha256'], csv_artifact=csv_path.name,
                  csv_sha256=sha256_file(csv_path))
    dump(path, replay)


def original_training(source, target, seed):
    shutil.copytree(source, target, symlinks=True)
    run = analysis.read(target / 'run.json')
    binding = run['binding']
    binding['seed'] = seed
    binding['sources'].update({
        'lfads_trajectory_training.py': training.LEGACY_TRAINER_SHA256,
        'lfads.py': evaluator.LEGACY_ADAPTER_SHA256,
    })
    for relative in (run['best_checkpoint'], run['latest_checkpoint']):
        folder = target / relative
        model = analysis.read(folder / 'model.json')
        model['initialization_seed'] = seed
        dump(folder / 'model.json', model)
        state = analysis.read(folder / 'training.json')
        for row in state['history']:
            row['training'] = dict(source_cost=float(25-row['epoch']), kl_ic=.1)
        state.update(binding=binding,
                     model_artifacts={n: sha256_file(folder/n) for n in ('model.json', 'weights.npz')})
        dump(folder / 'training.json', state)
    latest = analysis.read(target / run['latest_checkpoint'] / 'training.json')
    run.update(binding=binding, training_seed=seed, attempt=1, elapsed_seconds=30.,
        latest_training_sha256=sha256_file(target / run['latest_checkpoint'] / 'training.json'),
        formal_training_completed=False, generative_fidelity_qualified=False,
        convergence_established=False)
    dump(target / 'epochs.json', latest['history'])
    dump(target / 'run.json', run)
    (target / 'attempts').mkdir()
    dump(target / 'attempts/attempt_0000.json', dict(status='FAILED', elapsed_seconds=5., error='fixture failure'))
    dump(target / 'attempts/attempt_0001.json', run)
    return run


def continued_training(parent, target):
    old = analysis.read(parent / 'run.json')
    binding = copy.deepcopy(old['binding'])
    binding['sources']['lfads_trajectory_training.py'] = evaluator.BUDGET_TRAINER_SHA256
    latest, selected, lineage = training._prepare_continuation(parent, 80, binding, 56)
    _, state = training._seed_continuation(target, parent, latest, selected, binding, lineage)
    # Keep the copied best at epoch18. Three additional worse epochs exhaust
    # patience4, including the single stale epoch inherited from the parent.
    state = copy.deepcopy(state)
    state['history'].extend(dict(epoch=i, training=dict(source_cost=float(25-i), kl_ic=.1),
        validation=dict(source_cost=float(i))) for i in range(20, 23))
    relative = 'checkpoints/epoch_0023_attempt_0000'
    folder = target / relative
    shutil.copytree(target / state['checkpoint_directory'], folder)
    model = analysis.read(folder / 'model.json')
    model['training_updates'] = 23*56
    dump(folder / 'model.json', model)
    np.savez(folder / 'weights.npz', fake=np.array([7.]))
    state.update(checkpoint_directory=relative, next_epoch=23, training_updates=23*56, stale=4,
        model_artifacts={n: sha256_file(folder/n) for n in ('model.json', 'weights.npz')})
    state.pop('inherited_parent_checkpoint')
    dump(folder / 'training.json', state)
    (target / 'latest').unlink()
    (target / 'latest').symlink_to(relative)
    run = copy.deepcopy(old)
    run.update(binding=binding, attempt=0, elapsed_seconds=12., completed_epochs=23,
        training_updates=23*56, latest_checkpoint=relative, stale_epochs=4,
        stop_reason='validation_patience', effective_maximum_epochs=80,
        continuation=lineage, continuation_sha256=sha256_file(target / 'continuation.json'),
        latest_training_sha256=sha256_file(folder / 'training.json'))
    dump(target / 'epochs.json', state['history'])
    dump(target / 'run.json', run)
    (target / 'attempts').mkdir()
    dump(target / 'attempts/attempt_0000.json', run)
    training._verified_run(target, 56)
    return run


@pytest.fixture
def paired(replicated, cohort_fixture, monkeypatch):
    root = cohort_fixture.root
    original = replicated
    continued = root / 'continued_forecasts'
    shutil.copytree(original, continued)
    parent_root, child_root = root / 'original_training', root / 'continued_training'
    mapping = {'41': 'cuda_candidate_seed41', '42': 'seed_42', '43': 'seed_43'}
    for seed in analysis.SEEDS:
        parent = parent_root / f'seed_{seed}'
        child = child_root / mapping[str(seed)]
        old = original_training(cohort_fixture.training, parent, seed)
        new = continued_training(parent, child)
        receipt = dict(schema='lfads_generation_memory_inference_migration_v1',
            status='QUALIFIED_INFERENCE_ONLY',
            from_adapter_sha256=evaluator.LEGACY_ADAPTER_SHA256,
            to_adapter_sha256='synthetic-inference-adapter',
            unchanged_sources={k: v for k, v in old['binding']['sources'].items() if k!='lfads.py'})
        migration = dict(receipt=receipt, receipt_sha256='synthetic-identical-migration-receipt')
        bind_evaluation(original / f'seed_{seed}_forecast', parent, seed, migration)
        migration = copy.deepcopy(migration)
        migration['trainer_bridge'] = dict(checkpoint_scope='continued_child',
            legacy_trainer_sha256=training.LEGACY_TRAINER_SHA256,
            budget_trainer_sha256=evaluator.BUDGET_TRAINER_SHA256,
            saved_trainer_sha256=evaluator.BUDGET_TRAINER_SHA256,
            inference_trainer_sha256=evaluator.BUDGET_TRAINER_SHA256,
            verified_continuation_sha256=new['continuation_sha256'])
        bind_evaluation(continued / f'seed_{seed}_forecast', child, seed, migration)
        for forecasts in (original, continued):
            bind_replay(forecasts, seed, repair_seeds=True)
    campaign = root / 'budget_campaign.json'
    dump(campaign, dict(training_seeds=[41,42,43], total_epoch_ceiling=80,
                        base_epoch_ceiling=20, outputs=mapping))
    monkeypatch.setattr(analysis, 'EXPECTED_UNITS', 3)
    return SimpleNamespace(original=original, continued=continued, parents=parent_root,
        children=child_root, campaign=campaign, root=root, mapping=mapping)


def run_analysis(fixture, name='comparison', **kwargs):
    return analysis.analyze(fixture.original, fixture.continued, fixture.parents,
        fixture.children, fixture.campaign, fixture.root/name, **kwargs)


def load_records(f):
    locations = analysis.training_locations(f.parents, f.children, f.campaign)
    return {budget: {seed: analysis.load_forecast(root, locations[budget][seed], seed)
        for seed in analysis.SEEDS} for budget, root in zip(analysis.BUDGETS, (f.original, f.continued))}


def test_complete_comparison_reuses_original_analysis_and_discloses_inherited_best(paired, monkeypatch):
    cached = paired.original / 'analysis'
    analysis.forecast.analyze(paired.original, cached)
    original_analyze = analysis.forecast.analyze
    calls = []
    def count_calls(root, output):
        calls.append(Path(root))
        return original_analyze(root, output)
    monkeypatch.setattr(analysis.forecast, 'analyze', count_calls)
    report = run_analysis(paired)
    output = paired.root / 'comparison'
    assert calls == [paired.continued]
    assert report['aggregate_sources']['original_20']['reused'] is True
    assert report['status'] == 'DESCRIBED_REPLAYED_BUDGET_COMPARISON'
    assert (report['score_rows'], report['forecast_origins']) == (864, 288)
    assert report['complete_seed_budget_pairs'] == 6
    assert report['scientific_claim_qualified'] is report['convergence_established'] is False
    assert report['actual_training_budgets_matched'] is False
    assert len(analysis.csv_rows(output/'forecast_all_seeds.csv')) == 864
    assert len(analysis.csv_rows(output/'costs_by_origin.csv')) == 288
    assert all(sha256_file(output/name)==digest for name,digest in report['artifacts'].items())
    paired_rows = analysis.csv_rows(output/'paired_summary.csv')
    assert all(r['mean_continued_minus_original'] in ('', '0.0') for r in paired_rows)
    assert not {'p_value','q_value','ci_lower','ci_upper'} & set(paired_rows[0])
    budgets = analysis.csv_rows(output/'training_budget.csv')
    assert all(r['completed_epochs']=='23' and r['effective_epoch_ceiling']=='80'
        and r['selected_epoch_zero_based']=='18' and r['stop_reason']=='validation_patience'
        and r['selected_checkpoint_inherited']=='True' and r['selected_weights_same_as_original']=='True'
        and r['reported_parent_plus_stage_seconds']=='47.0'
        for r in budgets if r['budget']==analysis.BUDGETS[1])
    curves = analysis.csv_rows(output/'posterior_objective_curves.csv')
    assert {r['partition'] for r in curves} == {'training','validation'}
    assert all(r['inherited_parent_epoch']=='True' for r in curves
               if r['budget']==analysis.BUDGETS[1] and int(r['epoch'])<20)
    assert any(r['status']=='FAILED' for r in analysis.csv_rows(output/'training_attempts.csv'))


@pytest.mark.parametrize('kind', ['missing_replay','failed_replay','running_training'])
def test_incomplete_seed_blocks_all_summaries_before_checkpoint_access(paired, monkeypatch, kind):
    if kind=='missing_replay':
        (paired.continued/'seed_43_replay.json').unlink()
    elif kind=='failed_replay':
        path=paired.continued/'seed_43_replay.json'
        value=analysis.read(path);value.update(status='FAILED',error='fixture failure');dump(path,value)
    else:
        path=paired.children/'seed_43/run.json'
        value=analysis.read(path);value['status']='RUNNING';dump(path,value)
    def forbidden(*args, **kwargs):
        pytest.fail('incomplete six-way inventory must precede all checkpoint access')
    monkeypatch.setattr(analysis,'_verified_run',forbidden)
    with pytest.raises(ValueError, match='all three seeds'):
        run_analysis(paired)
    output=paired.root/'comparison'
    report=analysis.read(output/'analysis.json')
    assert report['status']=='BLOCKED_INCOMPLETE_INPUTS'
    assert report['ready_seed_budget_pairs']==5
    assert len(analysis.csv_rows(output/'input_status.csv'))==6
    assert not (output/'forecast_summary.csv').exists()
    assert not (output/'paired_summary.csv').exists()


@pytest.mark.parametrize('kind', ['wrong_seed','wrong_case','loosened_tolerance','wrong_allowed',
                                  'missing_comparison','changed_field','source_identity','unit_map'])
def test_forecast_replay_or_identity_changes_are_rejected(paired, kind):
    root=paired.continued
    replay_path=root/'seed_41_replay.json'
    folder=root/'seed_41_forecast'
    if kind in ('wrong_seed','wrong_case','wrong_allowed','missing_comparison'):
        path=replay_path.with_suffix('.csv')
        rows=analysis.csv_rows(path)
        if kind=='wrong_seed':rows[0]['initial_seed']=0
        elif kind=='wrong_case':rows[0]['block']=99
        elif kind=='wrong_allowed':rows[0]['allowed_difference']=999
        else:rows.pop()
        rewrite_csv(path,rows)
        bind_replay(root,41)
    elif kind=='loosened_tolerance':
        value=analysis.read(replay_path);value['metric_tolerance']['rtol']=.1;dump(replay_path,value)
    elif kind=='changed_field':
        path=folder/'train_block_014_familiar_0_origin_1000.npz'
        with path.open('ab') as stream:stream.write(b'changed')
    elif kind=='source_identity':
        path=folder/'evaluation.json'
        value=analysis.read(path);value['native_source_identity']['gpu_source_revision']='other';dump(path,value)
        bind_replay(root,41)
    else:
        path=folder/'full_unit_samples.npz'
        with np.load(path,allow_pickle=False) as data:payload={k:data[k] for k in data.files}
        payload['unit_ids']=payload['unit_ids']+1
        np.savez_compressed(path,**payload)
        value=analysis.read(folder/'evaluation.json');value['artifacts'][path.name]=sha256_file(path)
        dump(folder/'evaluation.json',value);bind_replay(root,41)
    with pytest.raises(ValueError):run_analysis(paired)
    assert analysis.read(paired.root/'comparison/analysis.json')['status']=='FAILED'
    assert not (paired.root/'comparison/paired_summary.csv').exists()


@pytest.mark.parametrize('kind', ['parent_digest','ceiling','patience','wrong_bridge'])
def test_budget_or_parent_migration_changes_are_rejected(paired, kind):
    if kind=='wrong_bridge':
        path=paired.continued/'seed_41_forecast/evaluation.json'
        value=analysis.read(path);value['inference_migration']['trainer_bridge']['checkpoint_scope']='legacy_parent'
        dump(path,value);bind_replay(paired.continued,41)
        with pytest.raises(ValueError,match='pinned trainer'):
            run_analysis(paired)
        return
    # Mutate the object after independently verifying the on-disk lineage to
    # exercise the comparison's strict parent pairing, not the trainer itself.
    records=load_records(paired)
    left=records[analysis.BUDGETS[0]][41];right=records[analysis.BUDGETS[1]][41]
    name={'parent_digest':'parent_checkpoint_sha256','ceiling':'total_epochs',
          'patience':'inherited_early_stopping_patience'}[kind]
    right['run']['continuation'][name]={'parent_digest':'other','ceiling':81,'patience':5}[kind]
    with pytest.raises(ValueError,match='exact parent'):
        analysis.pair_bindings(left,right)


@pytest.mark.parametrize('kind', ['table','evaluation_binding','source_binding'])
def test_corrupt_cached_analysis_is_rejected(paired, kind):
    cache=paired.original/'analysis'
    analysis.forecast.analyze(paired.original,cache)
    if kind=='table':
        path=cache/'forecast_summary.csv';path.write_text(path.read_text()+'\n')
    else:
        path=cache/'analysis.json';value=analysis.read(path)
        if kind=='evaluation_binding':value['model_seed_bindings']['41']['evaluation_sha256']='wrong'
        else:value['source_binding']['dataset_sha256']='wrong'
        dump(path,value)
    with pytest.raises(ValueError,match='cached|source-bound'):
        run_analysis(paired)
    assert not (paired.root/'comparison/forecast_summary.csv').exists()


def test_pair_difference_sign_optional_lags_and_missing_costs(paired):
    records=load_records(paired)
    key=next(iter(records[analysis.BUDGETS[1]][41]['rows']))
    records[analysis.BUDGETS[1]][41]['rows'][key]['energy_score']+=2
    cases,blocks,summary=analysis.differences(records)
    changed=[r for r in cases if r['metric']=='energy_score' and r['continued_minus_original']!=0]
    assert len(changed)==1 and changed[0]['continued_minus_original']==2
    assert any(r['horizon_ms']==20 and r['metric']=='variogram_temporal_lag_16'
               and r['continued_minus_original'] is None for r in cases)
    selected={s:analysis.pair_bindings(records[analysis.BUDGETS[0]][s],
                                     records[analysis.BUDGETS[1]][s]) for s in analysis.SEEDS}
    (paired.parents/'seed_42/attempts/attempt_0000.json').unlink()
    budgets,_,attempts=analysis.training_tables(records,selected)
    assert any(r['budget']==analysis.BUDGETS[0] and r['seed']==42
               and r['reported_stage_attempt_seconds'] is None for r in budgets)
    assert any(r['budget']==analysis.BUDGETS[1] and r['seed']==42
               and r['reported_parent_plus_stage_seconds'] is None for r in budgets)
    assert any(r['seed']==42 and r['status']=='MISSING' for r in attempts)
    costs=analysis.evaluation_costs(records)
    for row in costs:
        graphs=records[row['budget']][row['seed']]['evaluation']['graph_lifetimes']
        assert row['unique_graph_setup_seconds']==sum(g['setup_seconds'] for g in graphs)
        assert row['peak_tensorflow_gpu_allocated_bytes'] is None
    records[analysis.BUDGETS[0]][41]['evaluation']['graph_lifetimes'][0]['setup_seconds']=float('nan')
    with pytest.raises(ValueError,match='nonnegative'):analysis.evaluation_costs(records)


def test_production_unit_gate_is_not_relaxed_by_tiny_fixture(paired, monkeypatch):
    monkeypatch.setattr(analysis,'EXPECTED_UNITS',100000)
    with pytest.raises(ValueError,match='fitting-only field scale'):
        run_analysis(paired)


def test_campaign_mapping_rejects_different_seeds_or_path_escape(tmp_path):
    path=tmp_path/'campaign.json'
    value=dict(training_seeds=[41,42,43],total_epoch_ceiling=80,base_epoch_ceiling=20,
               outputs={'41':'candidate41','42':'seed_42','43':'seed_43'})
    dump(path,value)
    locations=analysis.training_locations(tmp_path/'old',tmp_path/'child',path)
    assert locations[analysis.BUDGETS[1]][41]==tmp_path/'child/candidate41'
    value['outputs']['41']='../escape';dump(path,value)
    with pytest.raises(ValueError,match='inside'):
        analysis.training_locations(tmp_path/'old',tmp_path/'child',path)
    value['training_seeds']=[41,42,44];dump(path,value)
    with pytest.raises(ValueError,match='three-seed'):
        analysis.training_locations(tmp_path/'old',tmp_path/'child',path)


def test_cli_help_and_incomplete_inputs_have_correct_exit_status(tmp_path):
    command=[sys.executable,'-m','population_dynamics.lfads_budget_analysis']
    help_run=subprocess.run([*command,'--help'],capture_output=True,text=True)
    assert help_run.returncode==0 and '--original-analysis' in help_run.stdout
    campaign=tmp_path/'campaign.json'
    dump(campaign,dict(training_seeds=[41,42,43],total_epoch_ceiling=80,base_epoch_ceiling=20,
                      outputs={str(s):f'seed_{s}' for s in analysis.SEEDS}))
    args=[]
    for name in ('original','continued','original-training','continued-training'):
        args.extend(['--'+name,str(tmp_path/name)])
    output=tmp_path/'failed_comparison'
    failed=subprocess.run([*command,*args,'--campaign',str(campaign),'--output',str(output)],
                          capture_output=True,text=True)
    assert failed.returncode!=0 and 'all three seeds' in failed.stderr
    assert analysis.read(output/'analysis.json')['status']=='BLOCKED_INCOMPLETE_INPUTS'



def test_finite_ensemble_corrected_variogram_estimates_remain_signed():
    # Independent U-statistic: sample absolute differences 0 and 2 around an
    # observed difference of 1 produce a valid corrected score of -1.
    scores=replay_scores(np.array([[[0.,0.]],[[0.,2.]]]),np.array([[0.,1.]]),
                         {'spatial':np.array([[0,1]])},[])
    assert scores['variogram_spatial']==-1.
    row={name:'0' for name in analysis.METRICS}
    row.update(horizon_ms='20',variogram_spatial=str(scores['variogram_spatial']),
               unit_bin_mean_mse_corrected='-0.5',variogram_temporal_lag_4='',variogram_temporal_lag_16='')
    values=analysis._row_metrics(row)
    assert values['variogram_spatial']==-1. and values['unit_bin_mean_mse_corrected']==-.5
    row['observed_spikes_per_unit']='-1'
    with pytest.raises(ValueError,match='negative'):analysis._row_metrics(row)

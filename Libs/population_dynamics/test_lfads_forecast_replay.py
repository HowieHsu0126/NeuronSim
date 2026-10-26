"""All-origin replay with the evaluator's explicit small cohort fixture."""
import csv
import json
import numpy as np
import pytest

from . import lfads_forecast_replay as replay
from . import lfads_trajectory_evaluation as evaluator
from .test_lfads_trajectory_evaluation import cohort_fixture, call, SmallModel
from .provenance import sha256_file


def test_complete_independent_field_and_unit_replay(cohort_fixture, monkeypatch):
    f=cohort_fixture
    call(f)
    monkeypatch.setattr(replay,'fitting_scale',evaluator.fitting_scale)
    # This fixture has one periodic cell per population: its spatial edges
    # are self edges. Production adjacency replay retains its >=3x3 contract.
    monkeypatch.setattr(replay,'pair_sets',lambda _: dict(
        spatial=np.array([[0,0],[1,1]]), cross_population=np.array([[0,1]])))
    result=replay.replay(f.acquisition,f.fields,f.training,f.root/'forecasts',f.root/'replay.json','cpu')
    assert result['status']=='PASS'
    assert result['verified_origins']==48 and result['verified_score_rows']==144
    assert result['field_scores_recomputed']==720 and result['metric_comparisons']==1152
    assert result['sampled_unit_count_values_consumed']==48*16*100*3
    assert result['saved_full_unit_origins_compared']==1
    assert result['regenerated_origins_without_saved_full_unit_arrays']==47
    assert SmallModel.live==0
    with pytest.raises(ValueError,match='new JSON'):
        replay.replay(f.acquisition,f.fields,f.training,f.root/'forecasts',f.root/'replay.json','cpu')


def test_refitted_forecast_replay_requires_the_same_prior(cohort_fixture, monkeypatch):
    from . import lfads_prior_refit as refit
    f = cohort_fixture
    proof = dict(fitting_episodes=56, training_seed=41, fitted_parameters=8)
    monkeypatch.setattr(refit, 'load_prior_parameters', lambda *args, **kwargs: ({}, proof))
    monkeypatch.setattr(refit, 'apply_prior_parameters', lambda *args: None)
    evaluator.evaluate(f.acquisition, f.fields, f.training, f.config, f.root/'forecasts',
                       'cpu', prior_refit=f.root/'prior')
    monkeypatch.setattr(replay, 'fitting_scale', evaluator.fitting_scale)
    monkeypatch.setattr(replay, 'pair_sets', lambda _: dict(
        spatial=np.array([[0, 0], [1, 1]]), cross_population=np.array([[0, 1]])))
    with pytest.raises(ValueError, match='requires its fitted AR prior'):
        replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts', f.root/'missing.json', 'cpu')
    result = replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts',
        f.root/'refitted_replay.json', 'cpu', prior_refit=f.root/'prior')
    assert result['status'] == 'PASS' and result['verified_origins'] == 48
    assert result['prior_refit'] == proof
    monkeypatch.setattr(refit, 'load_prior_parameters', lambda *args, **kwargs: ({}, dict(proof, training_seed=42)))
    with pytest.raises(ValueError, match='prior-refit identity changed'):
        replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts', f.root/'wrong.json',
                      'cpu', prior_refit=f.root/'prior')


@pytest.mark.parametrize('corruption',['unit_score','field_score','sample_path'])
def test_replay_rejects_resigned_numeric_corruption(cohort_fixture, monkeypatch, corruption):
    f=cohort_fixture;call(f)
    monkeypatch.setattr(replay,'fitting_scale',evaluator.fitting_scale)
    # This fixture has one periodic cell per population: its spatial edges
    # are self edges. Production adjacency replay retains its >=3x3 contract.
    monkeypatch.setattr(replay,'pair_sets',lambda _: dict(
        spatial=np.array([[0,0],[1,1]]), cross_population=np.array([[0,1]])))
    folder=f.root/'forecasts';record=json.loads((folder/'evaluation.json').read_text())
    if corruption=='sample_path':
        name=f"{f.cohort.validation[0]['case_id']}_origin_1000.npz"
        with np.load(folder/name) as z:values={k:z[k] for k in z.files}
        values['lfads_future_generator_states']=values['lfads_future_generator_states']+1
        np.savez_compressed(folder/name,**values)
        record['artifacts'][name]=sha256_file(folder/name)
    else:
        path=folder/'scores.csv'
        with path.open() as stream:rows=list(csv.DictReader(stream))
        key='unit_total_coverage_90' if corruption=='unit_score' else 'energy_score'
        rows[0][key]=float(rows[0][key])+1
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        record['scores_sha256']=sha256_file(path)
    (folder/'evaluation.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='mismatch|tolerance'):
        replay.replay(f.acquisition,f.fields,f.training,folder,f.root/'replay.json','cpu')
    assert json.loads((f.root/'replay.json').read_text())['status'] in ('FAILED','MISMATCH')
    assert SmallModel.live==0


def test_explicit_rescore_preserves_original_and_commits_bound_replayed_scores(cohort_fixture,monkeypatch):
    f=cohort_fixture;call(f)
    monkeypatch.setattr(replay,'fitting_scale',evaluator.fitting_scale)
    monkeypatch.setattr(replay,'pair_sets',lambda _:dict(
        spatial=np.array([[0,0],[1,1]]),cross_population=np.array([[0,1]])))
    original=f.root/'forecasts';source_hash=sha256_file(original/'scores.csv')
    ev_hash=sha256_file(original/'evaluation.json')
    # Exercise the full rescore/replay path with a three-unit fixture. Actual
    # 100k-case arithmetic qualification is a separate native-run prerequisite.
    monkeypatch.setattr(replay,'_precision_binding',lambda *_:dict(
        corrected_unit_metrics_sha256=sha256_file(replay.Path(replay.__file__).with_name('trajectory_sample_metrics.py')),
        unchanged_metric_tolerance=dict(rtol=replay.METRIC_RTOL,atol=replay.METRIC_ATOL)))
    target=f.root/'rescored'
    result=replay.replay(f.acquisition,f.fields,f.training,original,f.root/'rescore_replay.json','cpu',
                        rescore_output=target,precision_qualification=f.root/'fixture_qualification')
    assert result['status']=='PASS' and result['metric_comparisons']==1152
    assert sha256_file(original/'scores.csv')==source_hash
    assert sha256_file(original/'evaluation.json')==ev_hash
    revised=json.loads((target/'evaluation.json').read_text())
    assert result['evaluation_sha256']==sha256_file(target/'evaluation.json')
    assert result['scores_sha256']==sha256_file(target/'scores.csv')
    assert revised['numerical_rescore']['original_forecast_evaluation_sha256']==ev_hash
    assert revised['numerical_rescore']['original_scores_sha256']==source_hash
    assert revised['training_binding']==json.loads((original/'evaluation.json').read_text())['training_binding']
    for name,digest in revised['artifacts'].items():
        assert sha256_file(target/name)==digest
    assert result['saved_full_unit_origins_compared']==1
    assert result['regenerated_origins_without_saved_full_unit_arrays']==47


def test_rescore_requires_actual_matching_numerical_qualification(tmp_path):
    proof=tmp_path/'report.json';proof.write_text(json.dumps(dict(status='FAILED')))
    with pytest.raises(ValueError,match='verified original-case'):
        replay._precision_binding(dict(unit_metrics_implementation_sha256=replay.LEGACY_UNIT_METRIC_SHA256),proof)


def test_only_verified_legacy_parent_metadata_bridge_is_admitted_for_rescore():
    saved=dict(receipt={'numerical':'same'},receipt_sha256='same-proof')
    bridge=dict(schema='lfads_pinned_budget_trainer_metadata_bridge_v1',
        legacy_trainer_sha256=replay.LEGACY_TRAINER_SHA256,budget_trainer_sha256=replay.BUDGET_TRAINER_SHA256,
        saved_trainer_sha256=replay.LEGACY_TRAINER_SHA256,inference_trainer_sha256=replay.BUDGET_TRAINER_SHA256,
        receipt_trainer_sha256=replay.LEGACY_TRAINER_SHA256,checkpoint_scope='legacy_parent',
        verified_continuation_sha256=None,
        scope='trainer metadata only; numerical adapter migration evidence remains unchanged')
    current=dict(saved,trainer_bridge=bridge)
    run=dict(binding=dict(sources={'lfads_trajectory_training.py':replay.LEGACY_TRAINER_SHA256}))
    precision={}
    assert replay._same_execution_migration(saved,current,run,precision)
    assert precision['verified_legacy_parent_trainer_bridge']==bridge
    assert not replay._same_execution_migration(saved,current,run,None)
    assert not replay._same_execution_migration(saved,dict(current,receipt_sha256='different'),run,{})
    assert not replay._same_execution_migration(saved,current,dict(run,continuation={'child':True}),{})
    assert not replay._same_execution_migration(saved,dict(current,trainer_bridge=dict(bridge,checkpoint_scope='continued_child')),run,{})


def test_conditional_prior_replays_all_origins_and_rejects_missing_or_changed_mean(cohort_fixture, monkeypatch):
    from . import lfads_prior_refit as refit
    from . import lfads_conditional_prior as conditional
    f = cohort_fixture
    proof = dict(fitting_episodes=56, training_seed=41, fitted_parameters=8)
    mean_binding = dict(mean_prior_sha256='a' * 64, additional_fitted_scalars=19200)
    mean = conditional.ConditionalMeanPrior(np.zeros((4, 1200, 4), np.float32))
    monkeypatch.setattr(refit, 'load_prior_parameters', lambda *args, **kwargs: ({}, proof))
    monkeypatch.setattr(refit, 'apply_prior_parameters', lambda *args: None)
    mean_load_cells = []
    def load_mean(*args, **kwargs):
        mean_load_cells.append(kwargs)
        return mean, mean_binding.copy()
    monkeypatch.setattr(conditional, 'load_mean_prior', load_mean)
    calls = []
    def zero_mean(model, state, stimulus, streams, draws, curve):
        assert curve.shape == (101, 4) and not curve.any()
        calls.append(model.history_bins)
        sample = model.simulate(state, stimulus, streams, draws)
        return conditional.ConditionalPriorSamples(**vars(sample),
            base_future_controller_inputs=sample.future_controller_inputs.copy())
    monkeypatch.setattr(conditional, 'simulate_with_mean', zero_mean)
    evaluator.evaluate(f.acquisition, f.fields, f.training, f.config, f.root/'forecasts', 'cpu',
        prior_refit=f.root/'prior', conditional_prior=f.root/'mean')
    record = json.loads((f.root/'forecasts/evaluation.json').read_text())
    assert record['model_variant'] == 'template_time_controller_mean_prior'
    assert record['conditional_prior'] == mean_binding
    monkeypatch.setattr(replay, 'fitting_scale', evaluator.fitting_scale)
    monkeypatch.setattr(replay, 'pair_sets', lambda _: dict(
        spatial=np.array([[0, 0], [1, 1]]), cross_population=np.array([[0, 1]])))
    with pytest.raises(ValueError, match='requires its conditional mean prior'):
        replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts', f.root/'missing_mean.json',
            'cpu', prior_refit=f.root/'prior')
    result = replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts', f.root/'mean_replay.json',
        'cpu', prior_refit=f.root/'prior', conditional_prior=f.root/'mean')
    assert result['status'] == 'PASS' and result['verified_origins'] == 48
    assert result['conditional_prior'] == mean_binding and len(calls) == 96
    expected_cell = dict(training_seed=record['training_seed'],
        network_budget_epochs=record['effective_training_epoch_budget'],
        selected_model_artifacts=record['selected_model_artifacts'])
    assert mean_load_cells == [expected_cell, expected_cell]
    mean_binding['mean_prior_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='conditional-prior identity changed'):
        replay.replay(f.acquisition, f.fields, f.training, f.root/'forecasts', f.root/'wrong_mean.json',
            'cpu', prior_refit=f.root/'prior', conditional_prior=f.root/'mean')

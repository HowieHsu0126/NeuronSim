"""LFADS evaluator lineage, causal protocol and shared-score contract tests.

Small explicit arrays substitute for the acquired cohort and LFADS graph. They
exercise the full 48-origin/144-row protocol, not a scientific-scale claim.
"""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from . import lfads_trajectory_evaluation as evaluator
from .lfads import LFADSCountSamples
from .lfads_trajectory_training import _binding
from .provenance import sha256_file
from .trajectory_sample_metrics import unit_count_metrics


UNITS = np.array([11, 17, 29], dtype=np.int64)
MAPPING = np.array([0, 1, 0], dtype=np.int64)
FIELDS = np.array(['["L23E",0,0]', '["L23I",0,0]'])
SOURCE = dict(native_library_sha256='a' * 64, graph_manifest_sha256='b' * 64,
    native_poisson_rng_family='per_edge_philox4_32_10')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class SmallCohort:
    def __init__(self, acquisition, fields, config):
        self.acquisition, self.fields, self.config = acquisition, fields, config
        cases = json.loads((acquisition / 'design.json').read_text())['cases']
        self.fit = [case for case in cases if case['block'] < 14]
        self.validation = [case for case in cases if case['block'] >= 14]
        self.unit_ids, self.unit_field_index, self.field_ids = UNITS, MAPPING, FIELDS
        self.source_identity = SOURCE
        self.records = json.loads((fields / 'dataset.json').read_text())['records']
        for case in cases:
            self._verify_metadata(case)
        self.binding = dict(acquisition_design_sha256=sha256_file(acquisition / 'design.json'),
            acquisition_ledger_sha256=sha256_file(acquisition / 'acquisition.json'),
            dataset_sha256=sha256_file(fields / 'dataset.json'), native_source_identity=SOURCE)

    def _verify_metadata(self, case):
        row = self.records[case['case_id']]
        if (sha256_file(self.acquisition / case['case_id'] / 'report.json') != row['raw_report_sha256']
                or sha256_file(self.fields / case['case_id'] / 'summary.json') != row['field_summary_sha256']):
            raise ValueError('bound native episode/field identity changed')

    def _verify_episode(self, data):
        if (not np.array_equal(data.unit_ids, UNITS) or not np.array_equal(data.unit_field_index, MAPPING)
                or data.duration_ticks != 60000 or data.bin_ticks != 50):
            raise ValueError('test cohort source identity or clock changed')

    def verify_model(self, model):
        return self.config['expected_parameters']


class SmallNative:
    response_change_after = None
    identity_change = False
    interval_change_case = None

    def __init__(self, episode, fields, **kwargs):
        self.unit_ids = UNITS[::-1].copy() if self.identity_change else UNITS.copy()
        self.unit_field_index, self.field_ids = MAPPING.copy(), FIELDS.copy()
        self.minimum_interval_ticks = np.array([21, 21, 21], dtype=np.int64)
        if episode.name == self.interval_change_case:
            self.minimum_interval_ticks[0] += 1
        self.source_identity = SOURCE.copy()
        self.binding = dict(episode_report_sha256=sha256_file(episode / 'report.json'),
            field_summary_sha256=sha256_file(fields / 'summary.json'))
        self.bin_ms, self.dt_ms, self.bin_ticks, self.duration_ticks = 5., .1, 50, 60000
        if kwargs.get('expected_source') not in (None, self.source_identity):
            raise ValueError('test native source identity changed')

    def window(self, start, stop):
        counts = np.zeros((1200, 3), dtype=np.float32)
        counts[::2, 0] = 1
        counts[::3, 1] = 1
        counts[::7, 2] = 2
        if self.response_change_after is not None:
            counts[self.response_change_after:] += 50
        stimulus = (np.arange(1200, dtype=np.float32) % 13)[:, None]
        return counts, stimulus

    def ready_at(self, tick):
        return np.zeros(3, dtype=np.int64)


class SmallModel:
    live = 0
    peak_live = 0
    graphs = []
    histories = []
    samples = []
    model_config = None
    bad_samples = False

    def __init__(self, directory, history_bins, horizon_bins, device):
        assert self.live == 0, 'origin graphs must be closed sequentially'
        SmallModel.live += 1
        SmallModel.peak_live = max(SmallModel.peak_live, SmallModel.live)
        self.history_bins, self.horizon_bins = history_bins, horizon_bins
        self.loaded_for_inference = True
        self.unit_ids = tuple(map(int, UNITS))
        self.training_history_bins, self.training_horizon_bins = 200, 1000
        self.initialization_seed, self.input_size, self.bin_seconds = 41, 1, .005
        self.execution_device = dict(requested=device, device_type='CPU')
        SmallModel.graphs.append((history_bins, horizon_bins))

    @classmethod
    def load(cls, directory, **kwargs):
        return cls(directory, **kwargs)

    def infer_state(self, history, metadata):
        assert history['counts'].shape == (1, self.history_bins, 3)
        assert history['stimulus'].shape == (1, self.history_bins, 1)
        assert metadata['unit_ids'] == list(UNITS)
        SmallModel.histories.append({key: value.copy() for key, value in history.items()})
        return SimpleNamespace(mean=float(history['counts'].mean()))

    def simulate(self, state, future, streams, n_samples):
        assert future.shape == (1, 100, 1)
        assert n_samples == 16
        config = self.model_config
        initial = streams.initial.normal(size=(n_samples, 1, config['ic_dim'])).astype(np.float32)
        history = np.zeros((n_samples, 1, self.history_bins, config['co_dim']), dtype=np.float32)
        controller = streams.dynamics.normal(size=(n_samples, 1, 100, config['co_dim'])).astype(np.float32)
        boundary = initial[..., :1] * np.ones((n_samples, 1, config['gen_dim']), dtype=np.float32)
        generator = np.broadcast_to(boundary[:, :, None], (n_samples, 1, 100, config['gen_dim'])).copy()
        means = np.full((n_samples, 1, 100, 3), .1 + state.mean, dtype=np.float32)
        counts = streams.observation.poisson(means)
        if self.bad_samples:
            counts = counts[:15]
        SmallModel.samples.append(dict(initial=initial.copy(), future=future.copy(), counts=counts.copy()))
        return LFADSCountSamples(counts, means, initial, history, controller, boundary, generator,
            self.unit_ids, self.bin_seconds)

    def close(self):
        SmallModel.live -= 1


@pytest.fixture
def cohort_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    acquisition, fields, training = [tmp_path / name for name in ('acquisition', 'fields', 'training')]
    for path in (acquisition, fields, training):
        path.mkdir()
    cases = [dict(case_id=f'train_block_{block:03d}_familiar_{template}', block=block,
        template_id=f'familiar_{template}', split='train') for block in range(18) for template in range(4)]
    records = {}
    for case in cases:
        raw, field = acquisition / case['case_id'], fields / case['case_id']
        raw.mkdir(); field.mkdir()
        dump(raw / 'report.json', dict(case=case['case_id']))
        dump(field / 'summary.json', dict(case=case['case_id']))
        records[case['case_id']] = dict(raw_report_sha256=sha256_file(raw / 'report.json'),
            field_summary_sha256=sha256_file(field / 'summary.json'))
    dump(acquisition / 'design.json', dict(cases=cases))
    dump(acquisition / 'acquisition.json', dict(status='EXPLICIT_SMALL_FIXTURE'))
    dump(fields / 'dataset.json', dict(records=records))
    config = yaml.safe_load((Path(__file__).parents[1] / 'configs/neural_population_lfads_trajectory.yaml').read_text())
    config['device'] = 'cpu'
    cohort = SmallCohort(acquisition, fields, config)
    binding = _binding(cohort, config, 41)
    history = [dict(epoch=i, validation=dict(source_cost=float(20 - i))) for i in range(20)]
    history[-1]['validation']['source_cost'] = 3.
    best_epoch = 18
    best_relative, latest_relative = 'checkpoints/epoch_0019_attempt_0000', 'checkpoints/epoch_0020_attempt_0000'
    for relative, epochs in ((best_relative, 19), (latest_relative, 20)):
        directory = training / relative
        directory.mkdir(parents=True)
        model = dict(schema='neural_population_lfads_conditional_v2', source_revision=evaluator.SOURCE_REVISION,
            unit_ids=list(map(int, UNITS)), input_size=1, history_bins=200, horizon_bins=1000,
            training_history_bins=200, training_horizon_bins=1000, model_config=config['model'],
            initialization_seed=41, bin_seconds=.005, training_updates=epochs * 56)
        dump(directory / 'model.json', model)
        np.savez(directory / 'weights.npz', fake=np.zeros(1))
        state = dict(binding=binding, next_epoch=epochs, history=history[:epochs], best=2., best_epoch=best_epoch,
            best_checkpoint=best_relative, checkpoint_directory=relative, training_updates=epochs * 56,
            stale=epochs - best_epoch - 1,
            model_artifacts={name: sha256_file(directory / name) for name in ('model.json', 'weights.npz')})
        dump(directory / 'training.json', state)
    (training / 'latest').symlink_to(latest_relative)
    (training / 'best').symlink_to(best_relative)
    run = dict(status='FITTED_DEVELOPMENT', binding=binding, training_seed=41, completed_epochs=20,
        training_updates=20 * 56, latest_checkpoint=latest_relative, best_checkpoint=best_relative,
        latest_training_sha256=sha256_file(training / latest_relative / 'training.json'),
        best_epoch=best_epoch, best_validation_source_cost=2., stale_epochs=1, stop_reason='epoch_budget')
    dump(training / 'run.json', run)
    dump(training / 'epochs.json', history)
    config_path = tmp_path / 'forecast.yaml'
    config_path.write_text(yaml.safe_dump(evaluator.MATCHED_PROTOCOL))
    monkeypatch.setattr(evaluator, 'TrajectoryCohort', SmallCohort)
    monkeypatch.setattr(evaluator, 'NativeRenewalEpisode', SmallNative)
    monkeypatch.setattr(evaluator, 'LFADSConditionalSimulator', SmallModel)
    monkeypatch.setattr(evaluator, 'fitting_scale', lambda *_: (np.zeros(2), np.ones(2),
        dict(unit_count=np.array([2, 1]), field_ids=FIELDS)))
    SmallNative.response_change_after = None
    SmallNative.identity_change = False
    SmallNative.interval_change_case = None
    SmallModel.live, SmallModel.peak_live = 0, 0
    SmallModel.graphs, SmallModel.histories, SmallModel.samples = [], [], []
    SmallModel.model_config = config['model']
    SmallModel.bad_samples = False
    return SimpleNamespace(root=tmp_path, acquisition=acquisition, fields=fields, training=training,
        config=config_path, cohort=cohort, binding=binding, run=run)


def call(fixture, output='forecasts'):
    return evaluator.evaluate(fixture.acquisition, fixture.fields, fixture.training, fixture.config,
        fixture.root / output, 'cpu')


def test_fitted_prior_uses_existing_all_origin_forecast_and_records_binding(cohort_fixture, monkeypatch):
    from . import lfads_prior_refit as refit
    fixture = cohort_fixture
    applied = []
    parameters = dict(logatau=np.ones((1, 4), np.float32), logevars=np.zeros((1, 4), np.float32))
    proof = dict(fitting_episodes=56, training_seed=41, fitted_parameters=8)
    def verified(directory, **kwargs):
        assert directory == fixture.root / 'prior'
        assert kwargs['run']['binding'] == fixture.binding
        assert kwargs['cohort_binding'] == fixture.cohort.binding
        return parameters, proof
    monkeypatch.setattr(refit, 'load_prior_parameters', verified)
    monkeypatch.setattr(refit, 'apply_prior_parameters', lambda model, values:
                        applied.append((model.history_bins, values)))
    report = evaluator.evaluate(fixture.acquisition, fixture.fields, fixture.training, fixture.config,
        fixture.root / 'refitted_forecasts', 'cpu', prior_refit=fixture.root / 'prior')
    assert report['status'] == 'EVALUATED_DEVELOPMENT'
    assert report['prior_refit'] == proof
    assert report['completed_origins'] == 48
    assert [boundary for boundary, _ in applied] == [200, 600, 1000]
    assert all(values is parameters for _, values in applied)


def test_full_origin_sample_score_contract_and_shared_estimator(cohort_fixture):
    f = cohort_fixture
    result = call(f)
    assert result['status'] == 'EVALUATED_DEVELOPMENT'
    assert result['completed_origins'] == 48 and result['score_rows'] == 144
    assert result['episodes'] == 16 and result['independent_blocks'] == 4
    assert SmallModel.graphs == [(200, 100), (600, 100), (1000, 100)]
    assert SmallModel.live == 0 and SmallModel.peak_live == 1
    assert len(SmallModel.histories) == len(SmallModel.samples) == 48
    assert result['likelihood_scores_compared'] is False
    assert result['convergence_established'] is False
    assert result['peak_tensorflow_gpu_allocated_bytes'] is None
    assert 'conditional means' in result['sampling_cost_scope']
    with (f.root / 'forecasts/scores.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len({(row['case_id'], row['origin_ms'], row['horizon_ms']) for row in rows}) == 144
    for origin_index, boundary in enumerate((200, 600, 1000)):
        for case_index in range(16):
            index = origin_index * 16 + case_index
            expected_seed = 86000 + 41 * 100000 + case_index * 100 + origin_index * 3
            expected_initial = np.random.default_rng(expected_seed).normal(size=(16, 1, f.binding['config']['model']['ic_dim'])).astype(np.float32)
            np.testing.assert_array_equal(SmallModel.samples[index]['initial'], expected_initial)
            raw = SmallNative(f.acquisition / f.cohort.validation[case_index]['case_id'],
                f.fields / f.cohort.validation[case_index]['case_id'])
            counts, stimulus = raw.window(0, 60000)
            np.testing.assert_array_equal(SmallModel.histories[index]['counts'], counts[None, :boundary])
            np.testing.assert_array_equal(SmallModel.histories[index]['stimulus'], stimulus[None, :boundary])
            np.testing.assert_array_equal(SmallModel.samples[index]['future'], stimulus[None, boundary:boundary + 100])
    artifact = f.root / 'forecasts/full_unit_samples.npz'
    with np.load(artifact, allow_pickle=False) as saved:
        assert saved['sampled_unit_counts'].shape == (16, 100, 3)
        np.testing.assert_array_equal(saved['unit_ids'], UNITS)
        assert saved['lfads_historical_controller_inputs'].shape == (16, 200, f.binding['config']['model']['co_dim'])
        assert 'latent_states' not in saved.files
        generated = saved['sampled_unit_counts'].copy()
    raw = SmallNative(f.acquisition / f.cohort.validation[0]['case_id'], f.fields / f.cohort.validation[0]['case_id'])
    counts, _ = raw.window(0, 60000)
    expected = unit_count_metrics(torch.from_numpy(generated), torch.from_numpy(counts[200:300]),
        raw.minimum_interval_ticks, 50, [1, 2, 4])
    for key, value in expected.items():
        assert float(rows[2][key]) == value
    for name, digest in result['artifacts'].items():
        assert sha256_file(f.root / 'forecasts' / name) == digest


def test_response_after_origin_cannot_enter_state_or_sample(cohort_fixture):
    f = cohort_fixture
    call(f, 'original')
    reference_history = SmallModel.histories[0]['counts'].copy()
    reference_sample = SmallModel.samples[0]['counts'].copy()
    SmallModel.graphs, SmallModel.histories, SmallModel.samples = [], [], []
    SmallNative.response_change_after = 200
    call(f, 'changed')
    np.testing.assert_array_equal(SmallModel.histories[0]['counts'], reference_history)
    np.testing.assert_array_equal(SmallModel.samples[0]['counts'], reference_sample)
    with (f.root / 'original/scores.csv').open() as a, (f.root / 'changed/scores.csv').open() as b:
        original, changed = list(csv.DictReader(a)), list(csv.DictReader(b))
    assert original[0]['observed_spikes_per_unit'] != changed[0]['observed_spikes_per_unit']


@pytest.mark.parametrize('change,match', [
    ('status', 'terminal FITTED'), ('weights', 'checkpoint changed'), ('source', 'binding changed'),
    ('data', 'binding changed'), ('model', 'model configuration'), ('best_alias', 'alias or lineage'),
    ('epoch_history', 'epoch history'), ('selection', 'validation minimum'),
])
def test_selected_training_identity_failures(cohort_fixture, change, match):
    f = cohort_fixture
    run = copy.deepcopy(f.run)
    if change == 'status':
        run['status'] = 'RUNNING'; dump(f.training / 'run.json', run)
    elif change == 'weights':
        (f.training / run['best_checkpoint'] / 'weights.npz').write_bytes(b'changed')
    elif change == 'source':
        run['binding']['sources']['external/lfads.py'] = '0' * 64; dump(f.training / 'run.json', run)
    elif change == 'data':
        dump(f.acquisition / 'acquisition.json', dict(changed=True))
    elif change == 'model':
        directory = f.training / run['best_checkpoint']
        model = json.loads((directory / 'model.json').read_text()); model['unit_ids'].reverse()
        dump(directory / 'model.json', model)
        state = json.loads((directory / 'training.json').read_text())
        state['model_artifacts']['model.json'] = sha256_file(directory / 'model.json')
        dump(directory / 'training.json', state)
    elif change == 'best_alias':
        (f.training / 'best').unlink(); (f.training / 'best').symlink_to(run['latest_checkpoint'])
    elif change == 'epoch_history':
        dump(f.training / 'epochs.json', [])
    elif change == 'selection':
        run['best_epoch'] = 19; dump(f.training / 'run.json', run)
    with pytest.raises(ValueError, match=match):
        call(f)
    assert not (f.root / 'forecasts').exists()
    assert not SmallModel.graphs


@pytest.mark.parametrize('change', ['samples', 'unit_ids', 'refractory'])
def test_runtime_sample_and_native_failures_are_retained(cohort_fixture, change):
    f = cohort_fixture
    if change == 'samples':
        SmallModel.bad_samples = True
    elif change == 'unit_ids':
        SmallNative.identity_change = True
    else:
        SmallNative.interval_change_case = f.cohort.validation[1]['case_id']
    with pytest.raises(ValueError):
        call(f)
    assert SmallModel.live == 0
    if change != 'unit_ids':
        result = json.loads((f.root / 'forecasts/evaluation.json').read_text())
        assert result['status'] == 'FAILED'
        assert result['scientific_claim_qualified'] is False


def test_migration_is_exact_and_rejects_other_source_or_data_changes(tmp_path):
    current = dict(config={'id': 3}, seed=41, sources={'lfads.py': 'b' * 64,
        'external/lfads.py': 'c' * 64, 'lfads_trajectory_training.py': 'd' * 64}, dataset_sha256='e' * 64)
    old = copy.deepcopy(current); old['sources']['lfads.py'] = evaluator.LEGACY_ADAPTER_SHA256
    receipt = dict(schema='lfads_generation_memory_inference_migration_v1', status='QUALIFIED_INFERENCE_ONLY',
        from_adapter_sha256=evaluator.LEGACY_ADAPTER_SHA256, to_adapter_sha256='b' * 64,
        source_revision=evaluator.SOURCE_REVISION,
        unchanged_sources={key: value for key, value in old['sources'].items() if key != 'lfads.py'}, evidence=[])
    for kind in ('cpu_equivalence', 'full_population_cuda'):
        path = tmp_path / (kind + '.json')
        dump(path, dict(scope='unit_test_receipt_fixture', status='TEST_FIXTURE'))
        receipt['evidence'].append(dict(kind=kind, path=path.name, sha256=sha256_file(path)))
    path = tmp_path / 'migration.json'; dump(path, receipt)
    assert evaluator._verify_inference_binding(old, current, path)['receipt_sha256'] == sha256_file(path)
    assert evaluator._verify_inference_binding(current, current, None) is None
    with pytest.raises(ValueError, match='binding changed'):
        evaluator._verify_inference_binding(old, current, None)
    for target, key in (('sources', 'external/lfads.py'), ('sources', 'lfads_trajectory_training.py'), (None, 'dataset_sha256')):
        changed = copy.deepcopy(current)
        (changed if target is None else changed[target])[key] = 'f' * 64
        with pytest.raises(ValueError, match='binding changed'):
            evaluator._verify_inference_binding(old, changed, path)
    bad = copy.deepcopy(receipt); bad['to_adapter_sha256'] = '0' * 64; dump(path, bad)
    with pytest.raises(ValueError, match='not qualified'):
        evaluator._verify_inference_binding(old, current, path)
    dump(path, receipt)
    (tmp_path / 'cpu_equivalence.json').write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'):
        evaluator._verify_inference_binding(old, current, path)


def test_cli_failure_is_nonzero_and_preserves_existing_outputs(tmp_path):
    bad = tmp_path / 'bad.yaml'; bad.write_text('joint_samples: 1\n')
    out = tmp_path / 'cli'
    command = [sys.executable, '-m', 'Libs.population_dynamics.lfads_trajectory_evaluation',
        '--acquisition', str(tmp_path), '--fields', str(tmp_path), '--training', str(tmp_path),
        '--config', str(bad), '--output', str(out), '--device', 'cpu']
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode != 0
    failure = json.loads((out / 'failure.json').read_text())
    assert failure['status'] == 'FAILED' and 'exact common development' in failure['error']
    original = (out / 'failure.json').read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode != 0
    assert (out / 'failure.json').read_bytes() == original


def continuation_fixture(f):
    """Metadata-only descendant fixture; this does not represent fitted evidence."""
    from . import lfads_trajectory_training as trainer
    child = f.root/'continued_training'
    latest, best, lineage = trainer._prepare_continuation(f.training, 21, f.binding, 56)
    directory, imported = trainer._seed_continuation(child, f.training, latest, best, f.binding, lineage)
    # The parent best (epoch18) remains selected while one additional epoch worsens.
    state = copy.deepcopy(imported)
    relative = 'checkpoints/epoch_0021_attempt_0000'
    state.update(next_epoch=21, checkpoint_directory=relative, training_updates=21*56, stale=2)
    state['history'].append(dict(epoch=20, validation=dict(source_cost=4.)))
    new = child/relative
    new.mkdir()
    for name in ('model.json','weights.npz'):
        (new/name).write_bytes((directory/name).read_bytes())
    model = json.loads((new/'model.json').read_text()); model['training_updates'] = 21*56
    dump(new/'model.json',model)
    state['model_artifacts'] = {name:sha256_file(new/name) for name in ('model.json','weights.npz')}
    dump(new/'training.json',state)
    (child/'latest').unlink(); (child/'latest').symlink_to(relative)
    run = dict(f.run, completed_epochs=21, training_updates=21*56, stale_epochs=2, latest_checkpoint=relative,
        latest_training_sha256=sha256_file(new/'training.json'), effective_maximum_epochs=21,
        continuation=lineage, continuation_sha256=sha256_file(child/'continuation.json'))
    dump(child/'run.json',run); dump(child/'epochs.json',state['history'])
    return child, run, state


def test_evaluator_accepts_executed_explicit_budget_and_inherited_best(cohort_fixture):
    f = cohort_fixture
    original_best = sha256_file(f.training/'best/weights.npz')
    child, run, state = continuation_fixture(f)
    cohort, actual, selected_dir, selected, migration = evaluator._verified_training(f.acquisition,f.fields,child)
    assert actual['completed_epochs'] == actual['effective_maximum_epochs'] == 21
    assert actual['binding']['config']['maximum_epochs'] == 20
    assert selected['next_epoch'] == 19 and selected['best_epoch'] == 18
    assert sha256_file(selected_dir/'weights.npz') == original_best
    assert selected['inherited_parent_checkpoint']['training_sha256'] == run['continuation']['parent_best_training_sha256']
    assert migration is None


@pytest.mark.parametrize('change', ['silent_extension','unexecuted_budget','lineage','parent_history','checkpoint'])
def test_evaluator_rejects_silent_or_unverified_continuation(cohort_fixture, change):
    f = cohort_fixture
    child, run, state = continuation_fixture(f)
    latest = child/run['latest_checkpoint']
    if change == 'silent_extension':
        for key in ('continuation','continuation_sha256','effective_maximum_epochs'):
            state.pop(key,None); run.pop(key,None)
        dump(latest/'training.json',state)
        run['latest_training_sha256']=sha256_file(latest/'training.json')
        dump(child/'run.json',run)
    elif change == 'unexecuted_budget':
        run['effective_maximum_epochs']=22
        dump(child/'run.json',run)
    elif change == 'lineage':
        lineage=copy.deepcopy(run['continuation']); lineage['total_epochs']=22
        dump(child/'continuation.json',lineage)
    elif change == 'parent_history':
        state['history'][0]['validation']['source_cost']+=.5
        dump(latest/'training.json',state)
        dump(child/'epochs.json',state['history'])
        run['latest_training_sha256']=sha256_file(latest/'training.json')
        dump(child/'run.json',run)
    else:
        (child/'best/weights.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):
        evaluator._verified_training(f.acquisition,f.fields,child)


def test_budget_extension_cannot_reset_exhausted_parent_patience(cohort_fixture):
    from . import lfads_trajectory_training as trainer
    f = cohort_fixture
    # Test the explicit guard after independent parent-state validation.
    run, latest_dir, latest, best_dir, best = trainer._verified_run(f.training,56)
    latest['stale']=f.binding['config']['early_stopping_patience']
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trainer,'_verified_run',lambda *a,**kw:(run,latest_dir,latest,best_dir,best))
        with pytest.raises(ValueError,match='exhausted inherited validation patience'):
            trainer._prepare_continuation(f.training,21,f.binding,56)



def original_migration_fixture(root, source_binding):
    """Same schema/source-map shape as the existing qualified receipt; tiny evidence."""
    sources = dict(source_binding['sources'])
    sources['lfads_trajectory_training.py'] = evaluator.LEGACY_TRAINER_SHA256
    receipt = dict(schema='lfads_generation_memory_inference_migration_v1',status='QUALIFIED_INFERENCE_ONLY',
        from_adapter_sha256=evaluator.LEGACY_ADAPTER_SHA256,
        to_adapter_sha256='172932895e3589cb55310c728792e9e60bb6fafc7d20ede185db22e8ae542a65',
        source_revision=evaluator.SOURCE_REVISION,
        unchanged_sources={key:value for key,value in sources.items() if key!='lfads.py'},evidence=[])
    for kind in ('cpu_equivalence','full_population_cuda'):
        path = root/(kind+'_trainer_bridge_fixture.json')
        dump(path,dict(scope='tiny_receipt_fixture_only'))
        receipt['evidence'].append(dict(kind=kind,path=path.name,sha256=sha256_file(path)))
    path=root/'original_migration_fixture.json';dump(path,receipt)
    return path,receipt


def legacy_training_binding_fixture(f):
    # Both parent and budget child were trained before the inference adapter
    # migration; the fixture must not inherit today's adapter hash.
    f.binding['sources']['lfads.py'] = evaluator.LEGACY_ADAPTER_SHA256
    run=copy.deepcopy(f.run)
    run['binding']['sources']['lfads_trajectory_training.py']=evaluator.LEGACY_TRAINER_SHA256
    for relative in {run['latest_checkpoint'],run['best_checkpoint']}:
        path=f.training/relative/'training.json'
        state=json.loads(path.read_text());state['binding']=copy.deepcopy(run['binding']);dump(path,state)
    run['latest_training_sha256']=sha256_file(f.training/run['latest_checkpoint']/'training.json')
    dump(f.training/'run.json',run)
    return run


def test_canonical_legacy_parent_keeps_original_inference_receipt(cohort_fixture,monkeypatch):
    f=cohort_fixture
    run=legacy_training_binding_fixture(f)
    path,receipt=original_migration_fixture(f.root,run['binding'])
    current=copy.deepcopy(f.binding)
    assert current['sources']['lfads_trajectory_training.py']==evaluator.BUDGET_TRAINER_SHA256
    current['sources']['lfads.py']=receipt['to_adapter_sha256']
    monkeypatch.setattr(evaluator,'_binding',lambda *a:current)
    *_,migration=evaluator._verified_training(f.acquisition,f.fields,f.training,path)
    bridge=migration['trainer_bridge']
    assert bridge['checkpoint_scope']=='legacy_parent'
    assert bridge['saved_trainer_sha256']==evaluator.LEGACY_TRAINER_SHA256
    assert bridge['inference_trainer_sha256']==evaluator.BUDGET_TRAINER_SHA256
    assert migration['receipt']==receipt and migration['receipt_sha256']==sha256_file(path)
    assert bridge['verified_continuation_sha256'] is None


def test_continued_child_uses_original_receipt_only_with_checked_lineage(cohort_fixture,monkeypatch):
    f=cohort_fixture
    legacy_training_binding_fixture(f)
    child,run,_=continuation_fixture(f)
    path,receipt=original_migration_fixture(f.root,run['binding'])
    current=copy.deepcopy(f.binding);current['sources']['lfads.py']=receipt['to_adapter_sha256']
    monkeypatch.setattr(evaluator,'_binding',lambda *a:current)
    *_,migration=evaluator._verified_training(f.acquisition,f.fields,child,path)
    bridge=migration['trainer_bridge']
    assert bridge['checkpoint_scope']=='continued_child'
    assert bridge['saved_trainer_sha256']==evaluator.BUDGET_TRAINER_SHA256
    assert bridge['verified_continuation_sha256']==sha256_file(child/'continuation.json')
    with pytest.raises(ValueError,match='lineage'):
        evaluator._verify_inference_binding(run['binding'],current,path)
    altered=json.loads((child/'continuation.json').read_text())
    altered['total_epochs']=22;dump(child/'continuation.json',altered)
    with pytest.raises(ValueError,match='lineage'):
        evaluator._verified_training(f.acquisition,f.fields,child,path)


@pytest.mark.parametrize('change',['arbitrary_trainer','numeric_source','dataset','backend','evidence'])
def test_pinned_metadata_bridge_rejects_unrelated_changes(cohort_fixture,change):
    f=cohort_fixture
    old=copy.deepcopy(f.binding);old['sources']['lfads_trajectory_training.py']=evaluator.LEGACY_TRAINER_SHA256
    path,receipt=original_migration_fixture(f.root,old)
    current=copy.deepcopy(f.binding);current['sources']['lfads.py']=receipt['to_adapter_sha256']
    if change=='arbitrary_trainer':
        current['sources']['lfads_trajectory_training.py']='f'*64
    elif change=='numeric_source':
        current['sources']['external/lfads.py']='f'*64
    elif change=='dataset':
        current['dataset_sha256']='f'*64
    elif change=='backend':
        current['backend']['tensorflow']='changed'
    else:
        (f.root/receipt['evidence'][0]['path']).write_text('changed')
    with pytest.raises(ValueError,match='binding changed|evidence changed'):
        evaluator._verify_inference_binding(old,current,path)

"""Replay integrity, native raw-response endpoints, and explicit small fixtures."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from Libs.population_dynamics import lfads_boundary_replay as replay
from Libs.population_dynamics import lfads_boundary_diagnostic as diagnostic
from Libs.population_dynamics import lfads_trajectory_evaluation as evaluator
from Libs.population_dynamics import trajectory_sample_metrics as metrics_module
from Libs.population_dynamics.lfads import LFADSConditionalSimulator, LFADSCountSamples, LFADSRandomStreams
from Libs.population_dynamics.lfads_boundary_analysis import metric_names
from Libs.population_dynamics.provenance import sha256_file
from Libs.population_dynamics.test_lfads_trajectory_evaluation import (
    cohort_fixture, SmallNative, UNITS, MAPPING, FIELDS,
)
from Libs.population_dynamics.trajectory_scores import energy_score, variogram_score


def test_independent_conditional_means_consume_every_unit_and_horizon():
    values = np.random.default_rng(71).uniform(0, 8, (4, 1, 5, 7)).astype(np.float32)
    actual, consumed = replay.conditional_count_means(values, [1, 3, 5], chunk_units=3)
    assert consumed == values.size
    for h, value in actual.items():
        assert value == pytest.approx(values[:, :, :h].sum(dtype=np.float64) / (4 * 7), rel=1e-15)
    values[:, :, -1, -1] += 100
    changed, _ = replay.conditional_count_means(values, [1, 3, 5], chunk_units=3)
    assert changed[1] == actual[1] and changed[3] == actual[3] and changed[5] > actual[5]
    values[0, 0, -1, -1] = np.nan
    with pytest.raises(ValueError, match='FINITE_NONNEGATIVE'):
        replay.conditional_count_means(values, [1, 3, 5], chunk_units=3)


def test_exact_saved_schema_dtype_and_values_are_checked(tmp_path):
    path = tmp_path / 'payload.npz'
    expected = dict(unit_ids=np.array([9, 2], dtype=np.int64), controller=np.ones((2, 3), dtype=np.float32))
    np.savez(path, **expected)
    replay._saved_branch(path, expected)
    for corrupt in (dict(expected, unit_ids=expected['unit_ids'][::-1]),
                    dict(expected, controller=expected['controller'].astype(np.float64)),
                    dict(expected, controller=expected['controller'] + 1)):
        np.savez(path, **corrupt)
        with pytest.raises(ValueError, match='EXACT_BOUNDARY'):
            replay._saved_branch(path, expected)
    np.savez(path, unit_ids=expected['unit_ids'])
    with pytest.raises(ValueError, match='SCHEMA_CHANGED'):
        replay._saved_branch(path, expected)


def test_future_conditioned_roles_cannot_be_promoted():
    record = dict(configuration=copy.deepcopy(diagnostic.PROTOCOL),
        branch_roles={branch: diagnostic.branch_role(branch) for branch in diagnostic.BRANCHES},
        formal_test_accessed=False, calibration_outcomes_accessed=False, scientific_claim_qualified=False,
        inverse_intervention_evidence=False, mechanistic_microstate_oracle=False, weights_updated=False)
    row = dict(method='lfads', **{k: str(v) if isinstance(v, bool) else v
                               for k, v in diagnostic.branch_role(diagnostic.BRANCHES[3]).items()})
    replay._scope_contract(record, {0: row})
    with pytest.raises(ValueError, match='ROLE_OR_FUTURE_RESPONSE'):
        replay._scope_contract(record, {0: dict(row, role='forecast')})
    record['weights_updated'] = True
    with pytest.raises(ValueError, match='EVIDENCE_SCOPE'):
        replay._scope_contract(record, {0: row})


def test_raw_full_response_reference_executes_original_encoders_and_noisy_source():
    import yaml
    source_path = Path(diagnostic.__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml'
    spec = yaml.safe_load(source_path.read_text())['lfads_baseline_check']['model']
    with LFADSConditionalSimulator([12, 4], 1, 3, 3, .05,
            dict(spec, controller_input_lag=2), initialization_seed=41, device='cpu') as model:
        counts = np.array([[[0, 2], [1, 0], [2, 1], [1, 3], [0, 1], [1, 1]]], dtype=np.int64)
        stimulus = np.linspace(0, .8, 6, dtype=np.float32).reshape(1, 6, 1)
        engine = diagnostic.LFADSBoundaryDiagnostic(model)
        prepared = engine.prepare(counts, stimulus, horizon_bins=2, n_samples=4, sampling_seeds=(410, 411, 412))
        sample = engine.sample_arm(prepared, diagnostic.BRANCHES[3])
        result = replay.raw_response_endpoint(model, prepared, counts, stimulus, sample,
            rtol=diagnostic.PROTOCOL['endpoint_rtol'], atol=diagnostic.PROTOCOL['endpoint_atol'])
        assert result['response_batch_size'] == prepared.n_samples and not result['encoders_cached']
        assert all(value['bitwise_equal'] for value in result['arrays'].values())
        assert not result['controller_or_boundary_imposed'] and result['all_within_tolerance']
        changed = counts.copy(); changed[:, 3:] += 20
        with pytest.raises(ValueError, match='RAW_RESPONSE_SOURCE_ENDPOINT_MISMATCH'):
            replay.raw_response_endpoint(model, prepared, changed, stimulus, sample,
                rtol=diagnostic.PROTOCOL['endpoint_rtol'], atol=diagnostic.PROTOCOL['endpoint_atol'])
        assert model.training_updates == 0


class SmallBoundaryModel:
    """No TensorFlow graph: explicitly substituted only by the pytest fixture."""
    live = 0
    model_config = None

    @classmethod
    def load(cls, directory, device):
        return cls(device)

    def __init__(self, device):
        type(self).live += 1
        self.unit_ids, self.bin_seconds = tuple(map(int, UNITS)), .005
        self.execution_device = dict(requested=device, device_type='CPU')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        type(self).live -= 1


class SmallBoundaryDiagnostic:
    def __init__(self, model):
        self.model = model

    def prepare(self, counts, stimulus, *, horizon_bins, n_samples, sampling_seeds):
        assert counts.shape == (1, 1200, 3) and stimulus.shape == (1, 1200, 1)
        assert horizon_bins == 100 and n_samples == 16
        return SimpleNamespace(sampling_seeds=sampling_seeds, mean=float(counts[:, :200].mean()))

    def sample_arm(self, prepared, branch):
        config = self.model.model_config
        rng = LFADSRandomStreams.from_seeds(*prepared.sampling_seeds)
        initial = rng.initial.normal(size=(16, 1, config['ic_dim'])).astype(np.float32)
        history = np.zeros((16, 1, 200, config['co_dim']), dtype=np.float32)
        future = rng.dynamics.normal(size=(16, 1, 100, config['co_dim'])).astype(np.float32)
        boundary = initial[..., :1] * np.ones((16, 1, config['gen_dim']), dtype=np.float32)
        states = np.broadcast_to(boundary[:, :, None], (16, 1, 100, config['gen_dim'])).copy()
        means = np.full((16, 1, 100, 3), .1 + prepared.mean + diagnostic.BRANCHES.index(branch) * .05, dtype=np.float32)
        return LFADSCountSamples(rng.observation.poisson(means), means, initial, history,
            future, boundary, states, self.model.unit_ids, self.model.bin_seconds)

    def verify_endpoint(self, prepared, branch, sample, *, rtol, atol):
        return dict(all_within_tolerance=True, rtol=rtol, atol=atol,
            sampled_counts_bitwise_equal=True,
            arrays={name: dict(bitwise_equal=True, within_tolerance=True) for name in
                    ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states')})


def _write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def replay_fixture(cohort_fixture, monkeypatch):
    """All 16 x 4 x 3 control paths on three units, never production evidence."""
    f = cohort_fixture
    # The existing strict production row parser is separately tested by the
    # analyzer. Only this explicit fixture substitutes its 100000-unit contract.
    cfg = dict(diagnostic.PROTOCOL, expected_units=3)
    monkeypatch.setattr(replay, 'PROTOCOL', cfg)
    SmallBoundaryModel.model_config = f.binding['config']['model']
    SmallBoundaryModel.live = 0
    monkeypatch.setattr(replay, 'LFADSConditionalSimulator', SmallBoundaryModel)
    monkeypatch.setattr(replay, 'LFADSBoundaryDiagnostic', SmallBoundaryDiagnostic)
    monkeypatch.setattr(replay, 'fitting_scale', evaluator.fitting_scale)
    pairs = dict(spatial=np.array([[0, 0], [1, 1]]), cross_population=np.array([[0, 1]]))
    monkeypatch.setattr(replay, 'pair_sets', lambda _: pairs)
    raw_calls = []
    def raw_check(model, prepared, counts, stimulus, sample, **kwargs):
        raw_calls.append(counts.shape)
        return dict(all_within_tolerance=True, response_batch_size=16, **kwargs)
    monkeypatch.setattr(replay, 'raw_response_endpoint', raw_check)
    def parse_small(folder, report):
        with (folder / 'scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 192
        result = {}
        for row in rows:
            for name in metric_names(cfg):
                row[name] = None if row[name] == '' else float(row[name])
            result[(row['case_id'], row['branch'], float(row['horizon_ms']))] = row
        return result
    monkeypatch.setattr(replay, 'read_rows', parse_small)
    folder = f.root / 'diagnostics'; folder.mkdir()
    directory = f.training / f.run['best_checkpoint']
    selected = json.loads((directory / 'training.json').read_text())
    module_config = Path(diagnostic.__file__).resolve().parents[1] / 'configs/neural_population_lfads_boundary_diagnostic.yaml'
    record = dict(configuration=cfg, status='DIAGNOSED_DEVELOPMENT', model_type='lfads_boundary_diagnostic',
        training_binding=f.binding, training_seed=41, training_run_sha256=sha256_file(f.training / 'run.json'),
        selected_training_sha256=sha256_file(directory / 'training.json'), selected_checkpoint=f.run['best_checkpoint'],
        selected_epoch=f.run['best_epoch'], selected_model_artifacts=selected['model_artifacts'],
        source_revision=replay.SOURCE_REVISION, inference_migration=None,
        dataset_sha256=f.binding['dataset_sha256'], acquisition_design_sha256=f.binding['acquisition_design_sha256'],
        native_source_identity=f.cohort.source_identity, configuration_sha256=sha256_file(module_config),
        evaluator_implementation_sha256=sha256_file(Path(diagnostic.__file__)),
        unit_metrics_implementation_sha256=sha256_file(Path(metrics_module.__file__)),
        branch_roles={branch: diagnostic.branch_role(branch) for branch in diagnostic.BRANCHES},
        formal_test_accessed=False, calibration_outcomes_accessed=False, scientific_claim_qualified=False,
        inverse_intervention_evidence=False, mechanistic_microstate_oracle=False, weights_updated=False,
        completed_cases=16, completed_branches=64, score_rows=192, artifacts={}, case_bindings=[])
    mean, sd, metadata = evaluator.fitting_scale(None, None)
    np.savez(folder / 'fitting_field_scale.npz', mean_hz=mean, standard_deviation_hz=sd, **metadata)
    record['artifacts']['fitting_field_scale.npz'] = sha256_file(folder / 'fitting_field_scale.npz')
    rows = []
    with SmallBoundaryModel('cpu') as model:
        engine = SmallBoundaryDiagnostic(model)
        for case_index, case in enumerate(f.cohort.validation):
            data = SmallNative(f.acquisition / case['case_id'], f.fields / case['case_id'])
            counts, stimulus = data.window(0, data.duration_ticks)
            seed = cfg['sampling_seed_base'] + 41 * 100000 + case_index * 100
            seeds = (seed, seed + 1, seed + 2)
            prepared = engine.prepare(counts[None], stimulus[None], horizon_bins=100, n_samples=16, sampling_seeds=seeds)
            record['case_bindings'].append(dict(case_id=case['case_id'], binding=data.binding, sampling_seeds=list(seeds)))
            for branch in diagnostic.BRANCHES:
                sample = engine.sample_arm(prepared, branch)
                paths = evaluator._sample_payload(sample, UNITS, 16, 200, 100, .005, f.binding['config']['model'])
                generated, observed = sample.counts[:, 0], counts[200:300]
                projected, actual = evaluator._project_counts(generated, observed, MAPPING, 2)
                name = f"{case['case_id']}_{branch}.npz"
                np.savez(folder / name, sampled_field_counts=projected, observed_field_counts=actual,
                    unit_ids=UNITS, field_ids=FIELDS, unit_field_index=MAPPING,
                    sampling_seeds=np.asarray(seeds, dtype=np.int64), known_input_sequence=stimulus, **paths)
                record['artifacts'][name] = sha256_file(folder / name)
                if case_index == 0:
                    full = 'full_unit_' + branch + '.npz'
                    np.savez(folder / full, sampled_unit_counts=generated, unit_ids=UNITS)
                    record['artifacts'][full] = sha256_file(folder / full)
                scaled = projected / (metadata['unit_count'] * .005)
                truth = actual / (metadata['unit_count'] * .005)
                for ms in cfg['horizons_ms']:
                    h = int(ms / 5)
                    row = dict(seed=41, block=case['block'], case_id=case['case_id'], template_id=case['template_id'],
                        method='lfads', **diagnostic.branch_role(branch), origin_ms=1000., horizon_ms=ms,
                        ensemble_artifact=name,
                        conditional_expected_spikes_per_unit=float(sample.conditional_expected_counts[:, :, :h].sum(dtype=np.float64) / (16 * 3)),
                        energy_score=energy_score(scaled[:, :h], truth[:h]),
                        **metrics_module.unit_count_metrics(torch.from_numpy(generated[:, :h]),
                            torch.from_numpy(observed[:h]), data.minimum_interval_ticks, data.bin_ticks, [1, 2, 4]))
                    for label, pair in pairs.items():
                        row['variogram_' + label] = variogram_score(scaled[:, :h], truth[:h], pair[:, 0], pair[:, 1])
                    for lag in cfg['temporal_lags_bins']:
                        row[f'variogram_temporal_lag_{lag}'] = (variogram_score(scaled[:, :h], truth[:h],
                            np.arange(2), np.arange(2), lag) if lag < h else None)
                    rows.append(row)
    _write_rows(folder / 'scores.csv', rows)
    record['scores_sha256'] = sha256_file(folder / 'scores.csv')
    (folder / 'evaluation.json').write_text(json.dumps(record))
    return SimpleNamespace(**vars(f), diagnostics=folder, record=record, rows=rows, raw_calls=raw_calls)


def _call(f):
    return replay.replay(f.acquisition, f.fields, f.training, f.diagnostics, f.root / 'replay.json', 'cpu')


def test_complete_small_control_path_replays_all_branches_and_metrics(replay_fixture):
    f = replay_fixture
    result = _call(f)
    assert result['status'] == 'PASS'
    assert (result['verified_cases'], result['verified_branches'], result['verified_score_rows']) == (16, 64, 192)
    assert result['all_branch_unit_metrics_replayed'] and result['all_conditional_means_replayed']
    assert result['saved_full_unit_branches_compared'] == 4
    assert result['regenerated_branches_without_saved_full_unit_arrays'] == 60
    assert not result['all_branch_full_unit_arrays_compared']
    assert result['sampled_unit_count_values_consumed'] == 64 * 16 * 100 * 3
    assert result['conditional_mean_values_consumed'] == 64 * 16 * 100 * 3
    assert result['field_scores_recomputed'] == 960
    assert result['metric_comparisons'] == 2880
    assert len(result['endpoint_checks']) == 32 and f.raw_calls == [(1, 1200, 3)]
    assert SmallBoundaryModel.live == 0
    with pytest.raises(ValueError, match='NEW_JSON'):
        _call(f)


@pytest.mark.parametrize('corruption', ['unit_score', 'field_score', 'conditional_mean', 'latent_path', 'role'])
def test_resigned_numeric_or_role_corruption_fails_closed(replay_fixture, corruption):
    f = replay_fixture
    if corruption == 'latent_path':
        name = f.rows[0]['ensemble_artifact']
        with np.load(f.diagnostics / name) as saved:
            values = {name: saved[name] for name in saved.files}
        values['lfads_future_generator_states'] += 1
        np.savez(f.diagnostics / name, **values)
        f.record['artifacts'][name] = sha256_file(f.diagnostics / name)
    else:
        name = {'unit_score': 'unit_total_coverage_90', 'field_score': 'energy_score',
                'conditional_mean': 'conditional_expected_spikes_per_unit', 'role': 'role'}[corruption]
        row_index = 3 if corruption == 'role' else 0
        f.rows[row_index][name] = 'forecast' if corruption == 'role' else float(f.rows[row_index][name]) + 1
        _write_rows(f.diagnostics / 'scores.csv', f.rows)
        f.record['scores_sha256'] = sha256_file(f.diagnostics / 'scores.csv')
    (f.diagnostics / 'evaluation.json').write_text(json.dumps(f.record))
    with pytest.raises(ValueError, match='TOLERANCE_FAILED|REPLAY_MISMATCH|ROLE_OR_FUTURE_RESPONSE'):
        _call(f)
    assert SmallBoundaryModel.live == 0
    if corruption != 'role':
        result = json.loads((f.root / 'replay.json').read_text())
        assert result['status'] in ('FAILED', 'MISMATCH')
        assert result['all_branch_unit_metrics_replayed'] is False


def test_cli_missing_input_has_non_success_receipt(tmp_path):
    output = tmp_path / 'out.json'
    argv = ['replay', '--acquisition', str(tmp_path / 'missing'), '--fields', str(tmp_path / 'missing'),
        '--training', str(tmp_path / 'missing'), '--diagnostics', str(tmp_path / 'missing'),
        '--output', str(output), '--device', 'cpu']
    with mock.patch('sys.argv', argv), pytest.raises(FileNotFoundError):
        replay.main()
    assert json.loads(output.read_text())['status'] == 'FAILED'

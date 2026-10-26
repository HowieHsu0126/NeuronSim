"""Small CPU execution of exhaustive replay and independent metric algebra."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import population_forecast_replay as replay_module
from . import population_trajectory_evaluation as evaluator
from . import test_population_trajectory_evaluation as existing
from . import trajectory_sample_metrics
from .population_forecast_replay import numpy_unit_metrics, replay
from .provenance import sha256_file
from .renewal_state_space import ControlledRenewalLDS


class CompleteOriginFixture(existing.ObservedEpisodeFixture):
    """Three identified units on a 600ms fixture, not mechanistic evidence."""
    def __init__(self, episode, fields, **kwargs):
        super().__init__(episode, fields, **kwargs)
        self.duration_ticks = 600


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def build_forecast(root, *, model_type='poisson', seed=41, validation_cases=16):
    helper = existing.MatchedEvaluationTests()
    acquisition, fields, training, config_path = helper.fixture(root)
    design = json.loads((acquisition / 'design.json').read_text())
    for block in range(2, validation_cases + 1):
        case = dict(case_id=f'train_block_{block:03d}_familiar_0', block=block, split='train',
                    family='trajectory', template_id='familiar_0')
        design['cases'].append(case)
        for folder, name in ((acquisition, 'report.json'), (fields, 'summary.json')):
            path = folder / case['case_id']; path.mkdir()
            (path / name).write_text(json.dumps({'fixture_case': case['case_id']}))
    (acquisition / 'design.json').write_text(json.dumps(design))
    rows = [dict(case_id=case['case_id'], split='train',
        raw_report_sha256=sha256_file(acquisition / case['case_id'] / 'report.json'),
        field_summary_sha256=sha256_file(fields / case['case_id'] / 'summary.json')) for case in design['cases']]
    write_rows(fields / 'episodes.csv', rows)
    dataset = json.loads((fields / 'dataset.json').read_text())
    dataset.update(completed_episodes=len(rows), episode_table_sha256=sha256_file(fields / 'episodes.csv'))
    (fields / 'dataset.json').write_text(json.dumps(dataset))
    run = json.loads((training / 'run.json').read_text())
    binding = run['binding']
    binding.update(seed=seed, acquisition_design_sha256=sha256_file(acquisition / 'design.json'),
                   dataset_sha256=sha256_file(fields / 'dataset.json'))
    binding['config']['validation_blocks'] = list(range(1, validation_cases + 1))
    checkpoint = torch.load(training / 'best.pt', weights_only=True)
    if model_type == 'renewal':
        model_configuration = dict(unit_ids=[11, 17, 29], input_size=1, latent_size=2, encoder_size=3,
            bin_seconds=.005, initialization_seed=seed, minimum_interval_ticks=[2, 3, 2], tick_seconds=.001)
        model = ControlledRenewalLDS(**model_configuration).double()
        with torch.no_grad():
            model.latent_model.loading.zero_()
            model.latent_model.log_rate_bias.copy_(torch.tensor([20., 40., 4000.]).double().log())
        checkpoint.update(model=model.state_dict(), model_configuration=model_configuration)
        for key, filename in [('model_implementation_sha256', 'renewal_state_space.py'),
                              ('observation_implementation_sha256', 'refractory_observation.py'),
                              ('loader_implementation_sha256', 'native_renewal_data.py')]:
            binding[key] = sha256_file(Path(__file__).with_name(filename))
    checkpoint['binding'] = binding
    torch.save(checkpoint, training / 'best.pt')
    run.update(binding=binding, best_checkpoint_sha256=sha256_file(training / 'best.pt'))
    (training / 'run.json').write_text(json.dumps(run))
    settings = yaml.safe_load(config_path.read_text())
    settings.update(origins_ms=[10., 25., 50.], horizons_ms=[20., 80., 500.])
    config_path.write_text(yaml.safe_dump(settings))
    forecasts = root / 'forecasts'
    with patch.object(evaluator, 'NativeRenewalEpisode', CompleteOriginFixture), \
         patch.object(evaluator, 'fitting_scale', helper.scale):
        report = evaluator.evaluate(acquisition, fields, training, config_path, forecasts, model_type, 'cpu')
    return acquisition, fields, training, forecasts, report


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    return build_forecast(tmp_path_factory.mktemp('all_origin_poisson'))


def copy_forecasts(cohort, tmp_path):
    destination = tmp_path / 'forecasts'
    shutil.copytree(cohort[3], destination)
    return destination, json.loads((destination / 'evaluation.json').read_text())


def refresh_scores(folder, report, rows):
    write_rows(folder / 'scores.csv', rows)
    report['scores_sha256'] = sha256_file(folder / 'scores.csv')
    (folder / 'evaluation.json').write_text(json.dumps(report))


def test_all_48_origins_through_cli_use_independent_algebra_and_immutable_outputs(cohort, tmp_path, monkeypatch):
    acquisition, fields, training, forecasts, evaluation = cohort
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', CompleteOriginFixture)
    def prohibited(*args, **kwargs):
        raise AssertionError('replay called the production unit scorer')
    monkeypatch.setattr(evaluator, 'unit_count_metrics', prohibited)
    monkeypatch.setattr(trajectory_sample_metrics, 'unit_count_metrics', prohibited)
    original_draw = replay_module._draw
    called_seeds = []
    def recorded_draw(model, model_type, state, future, seed, draws, device):
        called_seeds.append(seed)
        return original_draw(model, model_type, state, future, seed, draws, device)
    monkeypatch.setattr(replay_module, '_draw', recorded_draw)
    output = tmp_path / 'all_origins.json'
    argv = ['population_forecast_replay', '--device', 'cpu', '--all-origins']
    for name, path in [('acquisition', acquisition), ('fields', fields), ('training', training),
                       ('forecasts', forecasts), ('output', output)]:
        argv += ['--' + name, str(path)]
    monkeypatch.setattr(sys, 'argv', argv)
    replay_module.main()
    report = json.loads(output.read_text())
    assert report['status'] == 'PASS'
    assert report['verified_origins'] == report['expected_origins'] == 48
    assert report['verified_score_rows'] == 144
    assert report['metric_comparisons'] == 48 * 3 * 8
    assert report['sampled_unit_count_values_consumed'] == 48 * 4 * 100 * 3
    assert report['all_forecast_unit_metrics_replayed']
    assert report['all_origin_fields_and_latents_exact']
    assert not report['all_origin_unit_counts_compared_to_saved_full_arrays']
    assert report['saved_full_unit_origins_compared'] == 1
    assert report['regenerated_origins_without_saved_full_unit_arrays'] == 47
    assert report['first_origin_exact_replay']['status'] == 'EXACT_FULL_UNIT_AND_INDEPENDENT_FIELD_REPLAY'
    assert report['metric_tolerance'] == {'rtol': 5e-6, 'atol': 1e-7}
    base = evaluation['configuration']['sampling_seed_base'] + 41 * 100000
    assert called_seeds == [base + case * 100 + origin * 3 for case in range(16) for origin in range(3)]
    with output.with_suffix('.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report['metric_comparisons']
    assert sum(row['unit_count_reference'] == 'EXACT_SAVED_UNIT_COUNTS' for row in rows) == 24
    assert all(row['passed'] == 'True' for row in rows)
    assert report['csv_sha256'] == sha256_file(output.with_suffix('.csv'))
    before = output.read_bytes(), output.with_suffix('.csv').read_bytes()
    with pytest.raises(FileExistsError):
        replay(acquisition, fields, training, forecasts, output, 'cpu', all_origins=True)
    assert before == (output.read_bytes(), output.with_suffix('.csv').read_bytes())
    reserved = tmp_path / 'reserved.csv'; reserved.write_text('existing evidence')
    with pytest.raises(FileExistsError):
        replay(acquisition, fields, training, forecasts, reserved.with_suffix('.json'), 'cpu', all_origins=True)
    assert reserved.read_text() == 'existing evidence'
    assert len(called_seeds) == 48


def test_renewal_replay_preserves_exact_first_origin_events_and_other_seed(tmp_path, monkeypatch):
    acquisition, fields, training, forecasts, _ = build_forecast(tmp_path, model_type='renewal', seed=43, validation_cases=1)
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', CompleteOriginFixture)
    report = replay(acquisition, fields, training, forecasts, tmp_path / 'renewal.json', 'cpu', all_origins=True)
    assert report['status'] == 'PASS' and report['verified_origins'] == 3
    assert report['training_seed'] == 43
    assert report['first_origin_exact_replay']['event_support']['refractory_violations'] == 0
    assert report['first_origin_exact_replay']['event_support']['boundary_violations'] == 0
    assert report['origins'][2]['initial_seed'] == report['sampling_seed_base'] + 43 * 100000 + 6


@pytest.mark.parametrize('field', ['sampled_field_counts', 'observed_field_counts', 'latent_states', 'initial_latent_states'])
def test_changed_saved_field_or_latent_is_rejected_even_with_updated_digest(field, cohort, tmp_path, monkeypatch):
    forecasts, report = copy_forecasts(cohort, tmp_path)
    # The last latent failure explicitly exercises the end of the 48-origin loop.
    case = 'train_block_016_familiar_0' if field == 'latent_states' else 'train_block_001_familiar_0'
    name = f'{case}_origin_50.npz'
    with np.load(forecasts / name, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays[field].flat[0] += 1
    np.savez_compressed(forecasts / name, **arrays)
    report['artifacts'][name] = sha256_file(forecasts / name)
    (forecasts / 'evaluation.json').write_text(json.dumps(report))
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', CompleteOriginFixture)
    with pytest.raises(ValueError, match='independent field/latent replay differs: ' + field):
        replay(*cohort[:3], forecasts, tmp_path / 'rejected.json', 'cpu', all_origins=True)
    assert not (tmp_path / 'rejected.json').exists()


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'identity', 'digest', 'implementation'])
def test_score_identity_and_digest_failures_are_rejected_before_generation(failure, cohort, tmp_path, monkeypatch):
    forecasts, report = copy_forecasts(cohort, tmp_path)
    with (forecasts / 'scores.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if failure == 'missing':
        rows.pop()
    elif failure == 'duplicate':
        rows[-1] = rows[0].copy()
    elif failure == 'identity':
        rows[-1]['seed'] = '42'
    elif failure == 'implementation':
        report['unit_metrics_implementation_sha256'] = '0' * 64
    refresh_scores(forecasts, report, rows)
    if failure == 'digest':
        with (forecasts / 'scores.csv').open('a') as stream:
            stream.write('\n')
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', CompleteOriginFixture)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid identities reached generation')
    monkeypatch.setattr(replay_module, '_draw', forbidden)
    with pytest.raises(ValueError):
        replay(*cohort[:3], forecasts, tmp_path / 'rejected.json', 'cpu', all_origins=True)
    assert not (tmp_path / 'rejected.json').exists()


def test_later_unit_observation_map_change_is_rejected(cohort, tmp_path, monkeypatch):
    class ChangedMap(CompleteOriginFixture):
        def __init__(self, episode, fields, **kwargs):
            super().__init__(episode, fields, **kwargs)
            if episode.name == 'train_block_002_familiar_0':
                self.unit_field_index[0] = 1
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', ChangedMap)
    with pytest.raises(ValueError, match='observation map'):
        replay(*cohort[:4], tmp_path / 'rejected.json', 'cpu', all_origins=True)


def test_metric_mismatch_retains_all_origin_deltas_and_does_not_relax_tolerance(cohort, tmp_path, monkeypatch):
    forecasts, report = copy_forecasts(cohort, tmp_path)
    with (forecasts / 'scores.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]['unit_bin_mean_mse_corrected'] = str(float(rows[-1]['unit_bin_mean_mse_corrected']) + 1.)
    refresh_scores(forecasts, report, rows)
    monkeypatch.setattr(replay_module, 'NativeRenewalEpisode', CompleteOriginFixture)
    output = tmp_path / 'mismatch.json'
    with pytest.raises(ValueError, match='predeclared unit-metric comparison failed'):
        replay(*cohort[:3], forecasts, output, 'cpu', all_origins=True)
    result = json.loads(output.read_text())
    assert result['status'] == 'MISMATCH' and result['verified_origins'] == 48
    assert not result['all_forecast_unit_metrics_replayed']
    assert result['metric_tolerance'] == {'rtol': 5e-6, 'atol': 1e-7}
    with output.with_suffix('.csv').open(newline='') as stream:
        deltas = list(csv.DictReader(stream))
    assert len(deltas) == 1152
    assert sum(row['passed'] == 'False' for row in deltas) == 1


def test_independent_corrected_mse_is_unbiased_and_negative_estimates_are_retained():
    probability, expectation = .2, 0.
    for values in itertools.product((0, 1), repeat=3):
        sampled = np.array(values, dtype=np.int64).reshape(3, 1, 1)
        metrics, _, consumed = numpy_unit_metrics(sampled, np.zeros((1, 1)), np.array([1]), 1, [1], [1])
        expectation += probability**sum(values) * (1 - probability)**(3 - sum(values)) * metrics[1]['unit_bin_mean_mse_corrected']
        assert consumed == 3
    assert expectation == pytest.approx(probability**2, abs=1e-14)
    metrics, exact, _ = numpy_unit_metrics(np.array([0, 2]).reshape(2, 1, 1), np.ones((1, 1)), np.array([1]), 3, [1], [1, 4])
    assert metrics[1]['unit_bin_mean_mse_corrected'] == -1.
    assert metrics[1]['refractory_count_violation_4bin'] is None
    assert exact[1]['refractory_count_violation_4bin'] == (None, None)


def test_exact_support_denominators_quantiles_and_chunking_consume_all_units():
    sampled = np.array([[[2, 2, 2], [2, 2, 2]], [[1, 1, 1], [1, 1, 1]]], dtype=np.int64)
    first = numpy_unit_metrics(sampled, np.ones((2, 3)), np.array([4, 4, 4]), 5, [1, 2], [1, 2, 4], chunk_units=1)
    second = numpy_unit_metrics(sampled, np.ones((2, 3)), np.array([4, 4, 4]), 5, [1, 2], [1, 2, 4], chunk_units=2)
    assert first == second
    metrics, exact, consumed = first
    assert consumed == sampled.size == 12
    assert metrics[2]['unit_total_interval_width_90'] == pytest.approx(1.8)
    assert metrics[2]['refractory_count_violation_2bin'] == .5
    assert exact[2]['refractory_count_violation_2bin'] == (3, 6)
    assert exact[2]['unit_total_coverage_90'] == (0, 3)


@pytest.mark.parametrize('bad', ['fractional', 'negative', 'interval', 'overflow', 'incomplete_horizon'])
def test_invalid_metric_counts_and_denominators_fail(bad):
    samples, observed, intervals, horizons = np.zeros((2, 2, 1), dtype=np.int64), np.zeros((2, 1)), np.array([1]), [1, 2]
    if bad == 'fractional': observed[0, 0] = .5
    if bad == 'negative': samples[0, 0, 0] = -1
    if bad == 'interval': intervals[0] = 0
    if bad == 'overflow': samples[0, 0, 0] = 2**53
    if bad == 'incomplete_horizon': horizons = [1]
    with pytest.raises(ValueError):
        numpy_unit_metrics(samples, observed, intervals, 1, horizons, [1])

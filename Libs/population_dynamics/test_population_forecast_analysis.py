"""Synthetic count fixtures for the complete six-run development analysis CLI."""
from __future__ import annotations

import csv
import copy
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from . import population_forecast_analysis as analysis
from .poisson_trajectory_evaluation import fitting_scale
from .poisson_trajectory_training import _trainer_sources
from .provenance import sha256_file
from .population_state import ControlledPopulationState
from .trajectory_scores import energy_score, field_pairs, variogram_score


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _code():
    names = ('population_trajectory_evaluation.py', 'trajectory_sample_metrics.py', 'poisson_state_space.py',
             'renewal_state_space.py', 'refractory_observation.py', 'native_renewal_data.py')
    return {name: sha256_file(Path(analysis.__file__).with_name(name)) for name in names}


@pytest.fixture(scope='module')
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp('matched_count_fixture')
    acquisition, fields, forecasts = [root / name for name in ('acquisition', 'fields', 'forecasts')]
    acquisition.mkdir(); fields.mkdir(); forecasts.mkdir()
    unit_ids = np.arange(1, 100001, dtype=np.int64)
    mapping = (unit_ids - 1) % 128
    units = np.bincount(mapping, minlength=128)
    field_ids = np.array([json.dumps([f'P{p}', y, x], separators=(',', ':')) for p in range(8) for y in range(4) for x in range(4)])
    source_identity = dict(gpu_source_revision='74fc021cad2b4bd5c011c390b3f95a302b663483',
                           native_library_sha256='a' * 64, graph_manifest_sha256='b' * 64,
                           native_poisson_rng_family='per_edge_philox4_32_10', dt_ms=.1)
    cases, source_rows, truth = [], [], {}
    for block, template in itertools.product(range(18), range(4)):
        identity = f'train_block_{block:03d}_familiar_{template}'
        case = dict(case_id=identity, block=block, template_id=f'familiar_{template}', split='train',
                    family='trajectory', settings=dict(block=block, template=template))
        cases.append(case)
        native, field = acquisition / identity, fields / identity
        native.mkdir(); field.mkdir()
        _json(native / 'report.json', dict(status='EXECUTED', duration_ms=6000., config=case['settings'], **source_identity))
        counts = ((np.arange(1200)[:, None] // 20 + np.arange(128)[None, :] % 3 + block % 3 + template) % 7 + 1).astype(np.int64)
        np.savez_compressed(field / 'spatial_fields.npz', source_unit_ids=unit_ids, unit_field_index=mapping,
                            field_ids=field_ids, total_unit_count=units, count_sum=counts)
        _json(field / 'summary.json', dict(status='DESCRIBED', window_start_tick=0, window_stop_tick=60000,
              settings=dict(bin_ms=5.), artifacts={'spatial_fields.npz': sha256_file(field / 'spatial_fields.npz')}))
        source_rows.append(dict(case_id=identity, split='train', raw_report_sha256=sha256_file(native / 'report.json'),
                                field_summary_sha256=sha256_file(field / 'summary.json')))
        if block in analysis.BLOCKS:
            truth[identity] = counts
    _json(acquisition / 'design.json', dict(acquired_split='train', cases=cases, graph_manifest_sha256='b' * 64,
          native_library_sha256='a' * 64))
    _csv(fields / 'episodes.csv', source_rows)
    _json(fields / 'dataset.json', dict(status='BUILT', split='train', completed_episodes=72, cortical_units=[100000],
          design_sha256=sha256_file(acquisition / 'design.json'), episode_table_sha256=sha256_file(fields / 'episodes.csv'),
          settings=dict(bin_ms=5., cells_per_axis=4, start_ms=0.)))
    fitting = [case for case in cases if case['block'] < 14]
    validation = [case for case in cases if case['block'] in analysis.BLOCKS]
    mean, sd, metadata = fitting_scale(fields, fitting)
    config = yaml.safe_load(analysis.CONFIG_PATH.read_text())
    definitions = [json.loads(value) for value in field_ids]
    pairs = field_pairs([v[0] for v in definitions], [v[2] for v in definitions], [v[1] for v in definitions])
    code = _code()
    budget = dict(training_seeds=[41,42,43], fit_blocks=list(range(14)), validation_blocks=list(analysis.BLOCKS),
                  latent_size=16, encoder_size=32, bin_ms=5., learning_rate=.001, maximum_epochs=20,
                  early_stopping_patience=4, initialization_pseudocount=.5, gradient_norm_limit=100., initial_transition_diagonal=.98)
    # Finite-unit metrics below are controlled input values, not generated model
    # results. This tests their descriptive aggregation, not their replay math.
    for model, seed in itertools.product(analysis.MODELS, analysis.SEEDS):
        training, folder = forecasts / model / f'seed_{seed}', forecasts / model / f'seed_{seed}_forecast'
        training.mkdir(parents=True); folder.mkdir()
        binding = dict(config=budget, seed=seed, acquisition_design_sha256=sha256_file(acquisition / 'design.json'),
                       dataset_sha256=sha256_file(fields / 'dataset.json'),
                       model_implementation_sha256=code['poisson_state_space.py' if model == 'poisson' else 'renewal_state_space.py'])
        if model == 'renewal':
            binding.update(observation_implementation_sha256=code['refractory_observation.py'], loader_implementation_sha256=code['native_renewal_data.py'])
        torch.save(dict(binding=binding), training / 'best.pt')
        checkpoint_digest = sha256_file(training / 'best.pt')
        _json(training / 'run.json', dict(status='FITTED_DEVELOPMENT', binding=binding, best_checkpoint_sha256=checkpoint_digest, completed_epochs=20, best_epoch=19, stop_reason='epoch_budget'))
        np.savez(folder / 'fitting_field_scale.npz', mean_hz=mean, standard_deviation_hz=sd, **metadata)
        # Aggregation consumes only retained IDs. Actual unit count replay is
        # separately owned by population_forecast_replay, not this fixture.
        np.savez_compressed(folder / 'full_unit_samples.npz', unit_ids=unit_ids)
        artifacts = {'full_unit_samples.npz': sha256_file(folder / 'full_unit_samples.npz')}
        rows = []
        for case, origin in itertools.product(validation, config['origins_ms']):
            boundary = int(origin / 5)
            observed = truth[case['case_id']][boundary:boundary+100]
            offset = (2 if model == 'poisson' else 0) + (seed - 41)
            sampled = observed[None] + np.arange(16)[:, None, None] % 2 + offset
            name = f"{case['case_id']}_origin_{origin:g}.npz"
            np.savez_compressed(folder / name, sampled_field_counts=sampled, observed_field_counts=observed,
                                latent_states=np.zeros((16,100,16)), initial_latent_states=np.zeros((16,1,16)))
            artifacts[name] = sha256_file(folder / name)
            standardized = (sampled / (units * .005) - mean) / sd
            actual = (observed / (units * .005) - mean) / sd
            for horizon in config['horizons_ms']:
                bins = int(horizon / 5)
                row = dict(seed=seed, block=case['block'], case_id=case['case_id'], template_id=case['template_id'],
                           origin_ms=origin, horizon_ms=horizon, method=model,
                           energy_score=energy_score(standardized[:,:bins], actual[:bins]),
                           unit_bin_mean_mse_corrected=-.5 if model == 'poisson' else -.75,
                           sampled_expected_spikes_per_unit=1., observed_spikes_per_unit=1.,
                           unit_total_coverage_90=.75, unit_total_interval_width_90=2.,
                           inference_seconds=.01, full_unit_generation_seconds=.2 if model == 'poisson' else .3,
                           field_projection_seconds=.003, peak_gpu_allocated_bytes=None, ensemble_artifact=name)
                for width in config['refractory_support_windows_bins']:
                    row[f'refractory_count_violation_{width}bin'] = 0.
                for label, pair in pairs.items():
                    row['variogram_' + label] = variogram_score(standardized[:,:bins], actual[:bins], pair[:,0], pair[:,1])
                for lag in config['temporal_lags_bins']:
                    row[f'variogram_temporal_lag_{lag}'] = variogram_score(standardized[:,:bins], actual[:bins], np.arange(128), np.arange(128), lag) if lag < bins else None
                rows.append(row)
        _csv(folder / 'scores.csv', rows)
        _json(folder / 'evaluation.json', dict(status='EVALUATED_DEVELOPMENT', configuration=config, model_type=model, training_seed=seed,
              evaluation_blocks=list(analysis.BLOCKS), independent_blocks=4, episodes=16, score_rows=144,
              scientific_claim_qualified=False, formal_test_accessed=False, likelihood_scores_compared=False,
              all_unit_outputs_generated=True, field_outputs_from_unit_counts=True, native_source_identity=source_identity,
              acquisition_design_sha256=binding['acquisition_design_sha256'], dataset_sha256=binding['dataset_sha256'],
              evaluator_implementation_sha256=code['population_trajectory_evaluation.py'], unit_metrics_implementation_sha256=code['trajectory_sample_metrics.py'],
              checkpoint_sha256=checkpoint_digest, artifacts=artifacts, scores_sha256=sha256_file(folder / 'scores.csv')))
    return dict(root=root, forecasts=forecasts, acquisition=acquisition, fields=fields, config=config, code=code)


def _read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_complete_six_run_real_cli_and_descriptive_outputs(cohort, tmp_path):
    output = tmp_path / 'analysis'
    command = [sys.executable, '-m', 'Libs.population_dynamics.population_forecast_analysis',
               '--forecasts', str(cohort['forecasts']), '--acquisition', str(cohort['acquisition']),
               '--fields', str(cohort['fields']), '--output', str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'analysis.json').read_text())
    assert report['status'] == 'REPLAYED_DEVELOPMENT'
    assert report['score_rows'] == 864 and report['field_scores_recomputed'] == 4320
    assert report['cost_origin_rows'] == 288 and report['covariance_rows'] == 864
    assert report['scientific_claim_qualified'] is False and report['all_unit_metrics_independently_replayed'] is False
    rows = _read_rows(output / 'forecast_all_seeds.csv')
    assert all(row['variogram_temporal_lag_4'] == '' for row in rows if float(row['horizon_ms']) == 20.)
    assert all(row['variogram_temporal_lag_16'] == '' for row in rows if float(row['horizon_ms']) <= 80.)
    paired = _read_rows(output / 'forecast_paired_block_differences.csv')
    selected = [row for row in paired if row['metric'] == 'unit_bin_mean_mse_corrected']
    assert len(selected) == 12
    assert all(float(row['renewal_minus_poisson']) == -.25 for row in selected)
    assert all(row['training_seeds'] == '3' for row in paired)
    summaries = _read_rows(output / 'forecast_summary.csv')
    assert all(row['independent_blocks'] == '4' for row in summaries)
    assert not any(key in summaries[0] for key in ('p_value', 'q_value', 'ci_lower', 'ci_upper'))
    costs = _read_rows(output / 'costs_by_origin.csv')
    assert len(costs) == 288 and all(float(row['generated_horizon_ms']) == 500. for row in costs)
    assert all(row[name] == '' for row in costs for name in analysis.EVALUATOR_COST_COLUMNS)
    with np.load(output / 'covariance_arrays.npz') as arrays:
        assert len(arrays.files) == 864
        assert all(arrays[name].shape[0] == 4 and np.isfinite(arrays[name]).all() for name in arrays.files)
    assert all(sha256_file(output / name) == value for name, value in report['artifacts'].items())


@pytest.fixture
def changed_run(cohort, tmp_path):
    root = tmp_path / 'forecasts'
    model = root / 'poisson'
    model.mkdir(parents=True)
    for name in ('seed_41', 'seed_41_forecast'):
        shutil.copytree(cohort['forecasts'] / 'poisson' / name, model / name)
    return root, model / 'seed_41_forecast'


@pytest.fixture(scope='module')
def canonical(cohort):
    return analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])


def _resign(folder):
    path = folder / 'evaluation.json'
    report = json.loads(path.read_text())
    report['scores_sha256'] = sha256_file(folder / 'scores.csv')
    for name in report['artifacts']:
        report['artifacts'][name] = sha256_file(folder / name)
    _json(path, report)


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'unmatched_case', 'nan', 'blank_eligible', 'nonblank_ineligible', 'legacy_likelihood'])
def test_bad_score_rows_rejected(cohort, canonical, changed_run, failure):
    root, folder = changed_run
    rows = _read_rows(folder / 'scores.csv')
    if failure == 'duplicate': rows[-1] = rows[0].copy()
    elif failure == 'missing': rows.pop()
    elif failure == 'unmatched_case': rows[0]['case_id'] = 'renamed_case'
    elif failure == 'nan': rows[0]['unit_total_coverage_90'] = 'nan'
    elif failure == 'blank_eligible': rows[0]['variogram_temporal_lag_1'] = ''
    elif failure == 'nonblank_ineligible': rows[0]['variogram_temporal_lag_4'] = '0.0'
    elif failure == 'legacy_likelihood':
        for row in rows: row['poisson_lognormal_nll'] = '1.0'
    _csv(folder / 'scores.csv', rows); _resign(folder)
    with pytest.raises(ValueError):
        analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])


@pytest.mark.parametrize('failure', ['evaluation_status', 'config', 'source', 'dataset', 'checkpoint', 'artifact', 'scale', 'unit_ids'])
def test_changed_binding_rejected(cohort, canonical, changed_run, failure):
    root, folder = changed_run
    report_path = folder / 'evaluation.json'
    report = json.loads(report_path.read_text())
    if failure == 'evaluation_status': report['status'] = 'RUNNING'
    elif failure == 'config': report['configuration']['joint_samples'] = 8
    elif failure == 'source': report['native_source_identity']['native_library_sha256'] = 'f' * 64
    elif failure == 'dataset': report['dataset_sha256'] = 'f' * 64
    elif failure == 'checkpoint':
        with (root / 'poisson/seed_41/best.pt').open('ab') as stream: stream.write(b'changed')
    elif failure == 'artifact':
        name = next(name for name in report['artifacts'] if name != 'full_unit_samples.npz')
        with (folder / name).open('ab') as stream: stream.write(b'changed')
    elif failure == 'scale':
        p = folder / 'fitting_field_scale.npz'
        with np.load(p) as data: values = {key: data[key] for key in data.files}
        values['mean_hz'] = values['mean_hz'] + 1
        np.savez(p, **values)
    elif failure == 'unit_ids':
        np.savez_compressed(folder / 'full_unit_samples.npz', unit_ids=canonical.unit_ids[::-1])
        report['artifacts']['full_unit_samples.npz'] = sha256_file(folder / 'full_unit_samples.npz')
    _json(report_path, report)
    with pytest.raises(ValueError):
        analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])


def test_count_schema_and_unmatched_missing_aggregate_rejected():
    with pytest.raises(ValueError, match='integer count schema'):
        analysis._counts(np.ones((2,3), dtype=float), (2,3), 'fixture')
    with pytest.raises(ValueError, match='integer count schema'):
        analysis._counts(np.full((2,3), -1, dtype=np.int64), (2,3), 'fixture')
    with pytest.raises(ValueError, match='mixed missing'):
        analysis._mean([None, 1.])
    assert analysis._mean([-.5, -.75]) == -.625
    with pytest.raises(ValueError, match='nonfinite aggregate'):
        analysis._mean([1e308, 1e308])


def _measured_report(source, config, rows, original=None):
    """Measured-cost schema fixture; no performance or simulator evidence."""
    report = dict(original or {})
    longest, draws, units = int(max(config['horizons_ms']) / source.bin_ms), config['joint_samples'], len(source.unit_ids)
    hardware = dict(python='fixture', numpy='fixture', torch='fixture', host='fixture', device='cpu',
                    cpu_threads=1, cpu_interop_threads=1, model_parameter_dtypes=['torch.float64'],
                    model_parameter_devices=['cpu'])
    scope = dict(function_total_seconds='function entry through score hash; final report write excluded',
                 elapsed_seconds='legacy evaluation loop', setup_seconds='evaluation setup',
                 episode_costs='one preparation shared by all episode origins',
                 origin_total_seconds='one longest ensemble including scores and outputs',
                 horizon_reuse='one ensemble per origin shared by all horizons',
                 peak_process_rss_bytes='cumulative process-lifetime high-water mark',
                 excluded_costs=['imports', 'training', 'launcher staging/archive', 'final report write'],
                 speedup_qualified=False)
    origins, episodes = [], []
    for case in source.validation:
        for origin in config['origins_ms']:
            sibling = next(row for row in rows if row['case_id'] == case['case_id'] and row['origin_ms'] == origin)
            origins.append(dict(case_id=case['case_id'], block=case['block'], origin_ms=origin,
                horizons_ms=list(config['horizons_ms']), generated_horizon_ms=max(config['horizons_ms']),
                generated_bins=longest, draws=draws, unit_count=units, field_count=len(source.field_ids),
                score_rows=len(config['horizons_ms']), generated_unit_count_values=draws * longest * units,
                generated_unit_count_dtype='torch.int64', generated_unit_count_device='cpu',
                latent_dtype='torch.float64', latent_device='cpu', field_count_dtype='int64', field_metric_dtype='float64',
                stored_full_unit_counts=case == source.validation[0] and origin == config['origins_ms'][0],
                **{name: sibling[name] for name in analysis.COST_METRICS},
                scoring_seconds=.04, serialization_seconds=.05, hash_seconds=.01, csv_seconds=.01,
                origin_total_seconds=1., peak_process_rss_bytes=1000))
        bins = len(source.truth[case['case_id']])
        episodes.append(dict(case_id=case['case_id'], block=case['block'],
            native_duration_ticks=int(bins * source.bin_ms / source.source_identity['dt_ms']), native_bins=bins,
            unit_count=units, origins_ms=list(config['origins_ms']), loading_seconds=.1, window_seconds=.2,
            transfer_seconds=.01, episode_preparation_seconds=.4,
            count_dtype='float32', stimulus_dtype='float32', count_tensor_dtype='torch.float32',
            stimulus_tensor_dtype='torch.float32', count_tensor_device='cpu', stimulus_tensor_device='cpu'))
    report.update(hardware=hardware, cost_scope=scope, origin_costs=origins, episode_costs=episodes,
                  origins=len(origins), episodes=len(episodes), score_rows=len(rows),
                  generated_unit_count_values=len(origins) * draws * longest * units,
                  setup_seconds=1., elapsed_seconds=100., score_hash_seconds=.01,
                  function_total_seconds=102., peak_process_rss_bytes=2000)
    return report


@pytest.fixture
def measured_evaluation():
    config = dict(origins_ms=[10., 30.], horizons_ms=[5., 20.], joint_samples=4, field_metric_dtype='float64')
    cases = [dict(case_id=f'case_{block}', block=block, template_id='familiar') for block in (14, 15)]
    source = SimpleNamespace(validation=cases, bin_ms=5., unit_ids=np.array([11, 17, 29]),
                             field_ids=np.array(['field0', 'field1']),
                             source_identity=dict(dt_ms=1.), truth={case['case_id']: np.zeros((12, 2)) for case in cases})
    rows = [dict(seed=41, method='poisson', case_id=case['case_id'], block=case['block'], template_id='familiar',
                 origin_ms=origin, horizon_ms=horizon, inference_seconds=.01, full_unit_generation_seconds=.2,
                 field_projection_seconds=.003, peak_gpu_allocated_bytes=None)
            for case, origin, horizon in itertools.product(cases, config['origins_ms'], config['horizons_ms'])]
    return _measured_report(source, config, rows), rows, source, config


def test_measured_evaluator_cost_schema_shares_episode_and_keeps_legacy_nulls(measured_evaluation, tmp_path):
    report, rows, source, config = measured_evaluation
    parsed = analysis._evaluation_costs(report, rows, source, config)
    assert len(parsed['origin_costs']) == 4 and len(parsed['episode_costs']) == 2
    receipt = dict(measured_evaluation_costs=parsed)
    costs = [dict(case_id=case['case_id'], origin_ms=origin,
                  **analysis._measured_origin_columns(receipt, case['case_id'], origin))
             for case, origin in itertools.product(source.validation, config['origins_ms'])]
    assert sum(row['episode_preparation_seconds'] or 0 for row in costs) == .8
    assert sum(row['episode_loading_seconds'] is not None for row in costs) == 2
    assert all(row['episode_cost_origin_ms'] == 10. for row in costs)
    assert all(json.loads(row['evaluation_hardware']) == report['hardware'] for row in costs)
    assert all(row['generated_unit_count_values'] == 48 for row in costs)
    assert sum(row['stored_full_unit_counts'] for row in costs) == 1
    path = tmp_path / 'costs.csv'
    analysis._write_rows(path, costs)
    loaded = _read_rows(path)
    assert all(row['episode_loading_seconds'] == '' for row in loaded if float(row['origin_ms']) == 30.)
    assert all(row['scoring_seconds'] == '0.04' for row in loaded)
    assert analysis._evaluation_costs({}, rows, source, config) is None
    assert all(value is None for value in analysis._measured_origin_columns({}, 'case_14', 10.).values())


@pytest.mark.parametrize('failure', [
    'missing_origins', 'missing_episodes', 'duplicate_origin', 'duplicate_episode', 'foreign_case', 'wrong_block',
    'missing_identity', 'wrong_horizons', 'wrong_draws', 'wrong_units', 'wrong_denominator', 'wrong_total_denominator',
    'negative', 'nonfinite', 'missing_cost', 'fractional_rss', 'legacy_timer', 'sibling_timer', 'short_origin_total',
    'short_episode_total', 'wrong_episode_clock', 'wrong_episode_origins', 'wrong_retention', 'dtype', 'device',
    'hardware', 'qualified_scope', 'short_function_total', 'short_loop_total'])
def test_invalid_measured_evaluator_costs_rejected(measured_evaluation, failure):
    report, rows, source, config = measured_evaluation
    report, rows = copy.deepcopy(report), copy.deepcopy(rows)
    origin, episode = report['origin_costs'][0], report['episode_costs'][0]
    if failure == 'missing_origins': report['origin_costs'].pop()
    elif failure == 'missing_episodes': report.pop('episode_costs')
    elif failure == 'duplicate_origin': report['origin_costs'][-1] = copy.deepcopy(origin)
    elif failure == 'duplicate_episode': report['episode_costs'][-1] = copy.deepcopy(episode)
    elif failure == 'foreign_case': origin['case_id'] = 'foreign'
    elif failure == 'wrong_block': origin['block'] = 16
    elif failure == 'missing_identity': origin.pop('draws')
    elif failure == 'wrong_horizons': origin['horizons_ms'] = [20.]
    elif failure == 'wrong_draws': origin['draws'] = 8
    elif failure == 'wrong_units': origin['unit_count'] = 4
    elif failure == 'wrong_denominator': origin['generated_unit_count_values'] = 47
    elif failure == 'wrong_total_denominator': report['generated_unit_count_values'] += 1
    elif failure == 'negative': origin['hash_seconds'] = -1.
    elif failure == 'nonfinite': origin['csv_seconds'] = float('nan')
    elif failure == 'missing_cost': origin.pop('scoring_seconds')
    elif failure == 'fractional_rss': origin['peak_process_rss_bytes'] = 1.5
    elif failure == 'legacy_timer': origin['inference_seconds'] = .02
    elif failure == 'sibling_timer': rows[0]['inference_seconds'] = .02
    elif failure == 'short_origin_total': origin['origin_total_seconds'] = .01
    elif failure == 'short_episode_total': episode['episode_preparation_seconds'] = .01
    elif failure == 'wrong_episode_clock': episode['native_duration_ticks'] -= 1
    elif failure == 'wrong_episode_origins': episode['origins_ms'] = [10.]
    elif failure == 'wrong_retention': origin['stored_full_unit_counts'] = False
    elif failure == 'dtype': origin['generated_unit_count_dtype'] = 'torch.float64'
    elif failure == 'device': origin['latent_device'] = 'cuda:0'
    elif failure == 'hardware': report['hardware'].pop('host')
    elif failure == 'qualified_scope': report['cost_scope']['speedup_qualified'] = True
    elif failure == 'short_function_total': report['function_total_seconds'] = 1.
    elif failure == 'short_loop_total': report['elapsed_seconds'] = 1.
    with pytest.raises(ValueError, match='cost|measured|runtime|dtype|device|hardware|finite'):
        analysis._evaluation_costs(report, rows, source, config)


@pytest.fixture
def heterogeneous_case_costs(measured_evaluation):
    """Reuse the measured fixture for source-bound heterogeneous maps and clocks."""
    report, rows, source, config = copy.deepcopy(measured_evaluation)
    config.update(origins_ms=[.5, 1.5], horizons_ms=[.5, 1.5])
    contracts = {}
    for index, case in enumerate(source.validation):
        case["block"] = f"pfc_initial_{71 + index}_external_{91 + index}"
        contracts[case["case_id"]] = dict(block=case["block"], unit_count=3 if index == 0 else 7,
            field_count=2 if index == 0 else 4, bin_ms=.25 if index == 0 else .5,
            source_dt_ms=.05 if index == 0 else .125, episode_bins=12 if index == 0 else 10,
            full_unit_retention="all_origins", episode_inputs_materialized=False)
    origin_times, horizon_times = {10.: .5, 30.: 1.5}, {5.: .5, 20.: 1.5}
    for row in rows:
        row.update(block=contracts[row["case_id"]]["block"], origin_ms=origin_times[row["origin_ms"]],
                   horizon_ms=horizon_times[row["horizon_ms"]])
    for origin in report["origin_costs"]:
        contract = contracts[origin["case_id"]]
        bins = int(max(config["horizons_ms"]) / contract["bin_ms"])
        origin.update(block=contract["block"], origin_ms=origin_times[origin["origin_ms"]],
            horizons_ms=list(config["horizons_ms"]), generated_horizon_ms=max(config["horizons_ms"]),
            generated_bins=bins, unit_count=contract["unit_count"], field_count=contract["field_count"],
            generated_unit_count_values=config["joint_samples"] * bins * contract["unit_count"],
            stored_full_unit_counts=True)
    for episode in report["episode_costs"]:
        contract = contracts[episode["case_id"]]
        episode.update(block=contract["block"], unit_count=contract["unit_count"],
            origins_ms=list(config["origins_ms"]), native_bins=contract["episode_bins"],
            native_duration_ticks=int(contract["episode_bins"] * contract["bin_ms"] / contract["source_dt_ms"]),
            episode_inputs_materialized=False, window_seconds=0., transfer_seconds=0., episode_preparation_seconds=.2)
        for name in ("count_dtype", "stimulus_dtype", "count_tensor_dtype", "stimulus_tensor_dtype",
                     "count_tensor_device", "stimulus_tensor_device"):
            episode[name] = None
    report["generated_unit_count_values"] = sum(row["generated_unit_count_values"] for row in report["origin_costs"])
    report["cost_scope"]["episode_costs"] = "source adapter and map validation once; tensors are not materialized"
    return report, rows, source, config, contracts


def validate_case_costs(fixture):
    report, rows, source, config, contracts = fixture
    return analysis.validate_evaluation_costs(report, rows, config, validation=source.validation,
        unit_count=None, field_count=None, bin_ms=None, source_dt_ms=None, episode_bin_counts={},
        case_cost_contracts=contracts)


def test_heterogeneous_case_cost_contracts_preserve_source_maps_clocks_and_once_charged_loading(heterogeneous_case_costs):
    parsed = validate_case_costs(heterogeneous_case_costs)
    assert len(parsed["origin_costs"]) == 4 and len(parsed["episode_costs"]) == 2
    assert [row["generated_bins"] for row in parsed["origin_costs"]] == [6, 6, 3, 3]
    assert [row["unit_count"] for row in parsed["origin_costs"]] == [3, 3, 7, 7]
    assert [row["field_count"] for row in parsed["origin_costs"]] == [2, 2, 4, 4]
    assert all(row["generated_horizon_ms"] == 1.5 and row["stored_full_unit_counts"] for row in parsed["origin_costs"])
    assert all(row["count_dtype"] is None and row["count_tensor_device"] is None for row in parsed["episode_costs"])
    receipt = dict(measured_evaluation_costs=parsed)
    source, config = heterogeneous_case_costs[2:4]
    charged = [analysis._measured_origin_columns(receipt, case["case_id"], origin)
               for case, origin in itertools.product(source.validation, config["origins_ms"])]
    assert sum(row["episode_loading_seconds"] or 0 for row in charged) == .2
    assert sum(row["episode_preparation_seconds"] or 0 for row in charged) == .4
    assert sum(row["episode_loading_seconds"] is not None for row in charged) == 2


@pytest.mark.parametrize("failure", [
    "missing_contract", "foreign_contract", "unknown_contract_field", "bool_units", "bool_clock",
    "contract_block", "contract_retention", "contract_materialized", "fractional_episode_bins",
    "unaligned_case_bin", "wrong_origin_block", "wrong_units", "wrong_fields", "wrong_generated_bins",
    "wrong_denominator", "missing_origin", "duplicate_origin", "false_retention", "bool_retention",
    "wrong_episode_clock", "wrong_episode_bins", "wrong_episode_units", "missing_absent_dtype",
    "pretend_dtype", "pretend_device", "missing_materialized_flag", "repeated_window_cost", "repeated_transfer_cost",
    "bool_timer", "bool_denominator", "sibling_timer", "short_origin_total", "short_loop_total",
])
def test_invalid_heterogeneous_case_cost_contracts_fail_closed(heterogeneous_case_costs, failure):
    fixture = copy.deepcopy(heterogeneous_case_costs)
    report, rows, source, config, contracts = fixture
    origin, episode = report["origin_costs"][0], report["episode_costs"][0]
    contract = contracts[origin["case_id"]]
    if failure == "missing_contract": contracts.pop(origin["case_id"])
    elif failure == "foreign_contract": contracts["foreign"] = dict(contract)
    elif failure == "unknown_contract_field": contract["optional"] = True
    elif failure == "bool_units": contract["unit_count"] = True
    elif failure == "bool_clock": contract["bin_ms"] = True
    elif failure == "contract_block": contract["block"] = "another_source_block"
    elif failure == "contract_retention": contract["full_unit_retention"] = "first_origin"
    elif failure == "contract_materialized": contract["episode_inputs_materialized"] = True
    elif failure == "fractional_episode_bins": contract["episode_bins"] = 12.5
    elif failure == "unaligned_case_bin": contract["bin_ms"] = .3
    elif failure == "wrong_origin_block": origin["block"] = "another_source_block"
    elif failure == "wrong_units": origin["unit_count"] += 1
    elif failure == "wrong_fields": origin["field_count"] += 1
    elif failure == "wrong_generated_bins": origin["generated_bins"] += 1
    elif failure == "wrong_denominator": origin["generated_unit_count_values"] -= 1
    elif failure == "missing_origin": report["origin_costs"].pop()
    elif failure == "duplicate_origin": report["origin_costs"][-1] = dict(origin)
    elif failure == "false_retention": report["origin_costs"][-1]["stored_full_unit_counts"] = False
    elif failure == "bool_retention": origin["stored_full_unit_counts"] = 1
    elif failure == "wrong_episode_clock": episode["native_duration_ticks"] -= 1
    elif failure == "wrong_episode_bins": episode["native_bins"] -= 1
    elif failure == "wrong_episode_units": episode["unit_count"] += 1
    elif failure == "missing_absent_dtype": episode.pop("count_dtype")
    elif failure == "pretend_dtype": episode["count_tensor_dtype"] = "torch.float32"
    elif failure == "pretend_device": episode["count_tensor_device"] = "cpu"
    elif failure == "missing_materialized_flag": episode.pop("episode_inputs_materialized")
    elif failure == "repeated_window_cost": episode["window_seconds"] = .01
    elif failure == "repeated_transfer_cost": episode["transfer_seconds"] = .01
    elif failure == "bool_timer": origin["inference_seconds"] = True
    elif failure == "bool_denominator": origin["generated_unit_count_values"] = True
    elif failure == "sibling_timer": rows[0]["inference_seconds"] += .01
    elif failure == "short_origin_total": origin["origin_total_seconds"] = .01
    elif failure == "short_loop_total": report["elapsed_seconds"] = .01
    with pytest.raises(ValueError, match="cost|measured|identity|integer"):
        validate_case_costs(fixture)


def test_default_cost_contract_accepts_fractional_aligned_milliseconds(measured_evaluation):
    _, rows, source, config = copy.deepcopy(measured_evaluation)
    config.update(origins_ms=[.5, 1.5], horizons_ms=[.5, 1.5])
    source.bin_ms = .25
    source.source_identity["dt_ms"] = .05
    for row in rows:
        row["origin_ms"] = {10.: .5, 30.: 1.5}[row["origin_ms"]]
        row["horizon_ms"] = {5.: .5, 20.: 1.5}[row["horizon_ms"]]
    report = _measured_report(source, config, rows)
    parsed = analysis._evaluation_costs(report, rows, source, config)
    assert all(row["generated_horizon_ms"] == 1.5 for row in parsed["origin_costs"])
    assert [row["stored_full_unit_counts"] for row in parsed["origin_costs"]] == [True, False, False, False]
    assert all(row["count_dtype"] == "float32" for row in parsed["episode_costs"])


def test_bound_run_carries_measured_costs_without_replacing_evaluator_hash(cohort, canonical, changed_run):
    root, folder = changed_run
    _, rows, old_receipt = analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])
    path = folder / 'evaluation.json'
    old = json.loads(path.read_text())
    report = _measured_report(canonical, cohort['config'], rows, old)
    _json(path, report)
    _, _, receipt = analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])
    assert 'measured_evaluation_costs' not in old_receipt
    assert receipt['evaluation_sha256'] == sha256_file(path)
    assert len(receipt['measured_evaluation_costs']['origin_costs']) == 48
    assert receipt['measured_evaluation_costs']['hardware'] == report['hardware']
    assert report['evaluator_implementation_sha256'] == old['evaluator_implementation_sha256']
    report['evaluator_implementation_sha256'] = 'f' * 64
    _json(path, report)
    with pytest.raises(ValueError, match='evaluated implementation changed'):
        analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])


def test_aggregate_mixed_measured_and_legacy_costs_preserves_shared_episode_charge(cohort, canonical, changed_run, tmp_path):
    root, folder = changed_run
    for model, seed in itertools.product(analysis.MODELS, analysis.SEEDS):
        if (model, seed) == ('poisson', 41):
            continue
        (root / model).mkdir(exist_ok=True)
        for name in (f'seed_{seed}', f'seed_{seed}_forecast'):
            (root / model / name).symlink_to(cohort['forecasts'] / model / name, target_is_directory=True)
    _, rows, _ = analysis._load_run(root, 'poisson', 41, canonical, cohort['config'], cohort['code'])
    path = folder / 'evaluation.json'
    _json(path, _measured_report(canonical, cohort['config'], rows, json.loads(path.read_text())))
    output = tmp_path / 'measured_analysis'
    report = analysis.analyze(root, cohort['acquisition'], cohort['fields'], output)
    costs = _read_rows(output / 'costs_by_origin.csv')
    assert len(costs) == 288 and report['measured_evaluation_runs'] == 1
    measured = [row for row in costs if row['evaluation_hardware']]
    legacy = [row for row in costs if not row['evaluation_hardware']]
    assert len(measured) == 48 and len(legacy) == 240
    assert all(row['seed'] == '41' and row['method'] == 'poisson' for row in measured)
    assert sum(float(row['episode_loading_seconds']) for row in measured if row['episode_loading_seconds']) == pytest.approx(1.6)
    assert sum(bool(row['episode_preparation_seconds']) for row in measured) == 16
    assert all(row['episode_preparation_seconds'] == '' for row in measured if float(row['origin_ms']) != 1000.)
    assert all(row[name] == '' for row in legacy for name in analysis.EVALUATOR_COST_COLUMNS)
    assert all(json.loads(row['evaluation_hardware'])['host'] == 'fixture' for row in measured)
    assert all(float(row['origin_total_seconds']) == 1. for row in measured)
    assert report['end_to_end_cost_measured'] is False and report['same_hardware_cost_comparison'] is False
    assert report['speedup_qualified'] is False


def test_cli_incomplete_inputs_fail_nonzero_and_preserve_failure(tmp_path):
    output = tmp_path / 'failed'
    command = [sys.executable, '-m', 'Libs.population_dynamics.population_forecast_analysis', '--forecasts', str(tmp_path),
               '--acquisition', str(tmp_path), '--fields', str(tmp_path), '--output', str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert json.loads((output / 'analysis.json').read_text())['status'] == 'FAILED'


@pytest.mark.parametrize('failure', ['score_value', 'count_dtype', 'observed_truth', 'cost_horizon_reuse'])
def test_resigned_but_incorrect_numeric_evidence_rejected(cohort, changed_run, tmp_path, failure):
    root, folder = changed_run
    for model, seed in itertools.product(analysis.MODELS, analysis.SEEDS):
        if (model, seed) == ('poisson', 41):
            continue
        (root / model).mkdir(exist_ok=True)
        for name in (f'seed_{seed}', f'seed_{seed}_forecast'):
            (root / model / name).symlink_to(cohort['forecasts'] / model / name, target_is_directory=True)
    rows = _read_rows(folder / 'scores.csv')
    if failure == 'score_value':
        rows[0]['energy_score'] = str(float(rows[0]['energy_score']) + 1.)
        _csv(folder / 'scores.csv', rows)
    elif failure == 'cost_horizon_reuse':
        rows[0]['full_unit_generation_seconds'] = '.999'
        _csv(folder / 'scores.csv', rows)
    else:
        path = folder / rows[0]['ensemble_artifact']
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        if failure == 'count_dtype':
            payload['sampled_field_counts'] = payload['sampled_field_counts'].astype(np.float64)
        else:
            payload['observed_field_counts'] = payload['observed_field_counts'].copy()
            payload['observed_field_counts'][0,0] += 1
        np.savez_compressed(path, **payload)
    _resign(folder)
    expected = {'score_value': 'field score replay mismatch', 'count_dtype': 'integer count schema',
                'observed_truth': 'truth differs', 'cost_horizon_reuse': 'cost differs'}[failure]
    output = tmp_path / 'rejected_analysis'
    with pytest.raises(ValueError, match=expected):
        analysis.analyze(root, cohort['acquisition'], cohort['fields'], output)
    assert json.loads((output / 'analysis.json').read_text())['status'] == 'FAILED'


def test_continuation_budget_uses_executed_ceiling_and_selected_lineage():
    config = dict(maximum_epochs=20, early_stopping_patience=4)
    continuation = dict(schema='neural_population_trajectory_budget_continuation_v1',
        base_maximum_epochs=20, total_epochs=80, parent_completed_epochs=20,
        inherited_early_stopping_patience=4, current_source_hashes={'trainer': 'new'},
        parent_checkpoint_sha256='latest', parent_artifact_sha256={'best.pt': 'parent-best'},
        source_migration={'trainer': {'parent': 'old', 'current': 'new'}})
    run = dict(binding=dict(config=config), effective_maximum_epochs=80,
        completed_epochs=36, best_epoch=31, stop_reason='validation_patience',
        continuation=continuation, trainer_sources={'trainer': 'new'})
    receipt = analysis._training_budget(run, {'continuation': continuation}, 'continued-best')
    assert receipt['effective_maximum_epochs'] == 80 and receipt['base_maximum_epochs'] == 20
    assert receipt['completed_epochs'] == 36 and receipt['best_epoch'] == 31
    with pytest.raises(ValueError, match='lineage'):
        analysis._training_budget(run, {}, 'unrelated-best')
    inherited = dict(run, completed_epochs=23, best_epoch=18)
    assert analysis._training_budget(inherited, {}, 'parent-best')['best_epoch'] == 18
    with pytest.raises(ValueError, match='continuation'):
        analysis._training_budget(dict(run, continuation=None), {}, 'continued-best')
    with pytest.raises(ValueError, match='ceiling'):
        analysis._training_budget(dict(run, effective_maximum_epochs=60), {'continuation': continuation}, 'continued-best')


@pytest.fixture
def population_campaign(cohort, tmp_path, monkeypatch):
    """Reuse synthetic complete coverage; real constructor/load and score replay."""
    root = tmp_path / 'population_campaign'
    canonical = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    native = SimpleNamespace(unit_ids=canonical.unit_ids, field_ids=canonical.field_ids,
        unit_field_index=canonical.unit_field_index, minimum_interval_ticks=np.full(100000, 21),
        bin_ms=5., dt_ms=.1, source_identity=canonical.source_identity)
    monkeypatch.setattr(analysis, 'NativeRenewalEpisode', lambda *a, **kw: native)
    for seed in analysis.SEEDS:
        cell = root / f'seed_{seed}'
        shutil.copytree(cohort['forecasts'] / 'renewal' / f'seed_{seed}', cell / 'training')
        shutil.copytree(cohort['forecasts'] / 'renewal' / f'seed_{seed}_forecast', cell / 'forecasts')
        model = ControlledPopulationState(unit_ids=list(map(int, native.unit_ids)),
            unit_field_index=native.unit_field_index, field_names=list(map(str, native.field_ids)),
            minimum_interval_ticks=native.minimum_interval_ticks, tick_seconds=.0001,
            bin_seconds=.005, input_size=1, initialization_seed=seed)
        training, forecasts = cell / 'training', cell / 'forecasts'
        run = json.loads((training / 'run.json').read_text())
        binding = run['binding']
        binding['config'].update(model_family='population', latent_channels=4,
                                 encoder_size=16, representation='recurrent', unit_chunk_size=4096)
        binding['native_source_identity'] = native.source_identity
        for key, name in (('model_implementation_sha256', 'population_state.py'),
                          ('latent_model_implementation_sha256', 'poisson_state_space.py'),
                          ('native_counts_implementation_sha256', 'native_count_windows.py')):
            binding[key] = sha256_file(Path(analysis.__file__).with_name(name))
        torch.save(dict(binding=binding, model=model.state_dict(), model_configuration=model.configuration()),
                   training / 'best.pt')
        run.update(model_family='population', representation='recurrent',
                   best_checkpoint_sha256=sha256_file(training / 'best.pt'))
        _json(training / 'run.json', run)
        rows = _read_rows(forecasts / 'scores.csv')
        for row in rows:
            row['method'] = 'population'
        _csv(forecasts / 'scores.csv', rows)
        evaluation = json.loads((forecasts / 'evaluation.json').read_text())
        evaluation.update(model_type='population', checkpoint_sha256=run['best_checkpoint_sha256'],
                          scores_sha256=sha256_file(forecasts / 'scores.csv'))
        _json(forecasts / 'evaluation.json', evaluation)
        _json(cell / 'replay.json', dict(status='PASS', model_type='population', training_seed=seed,
            verified_origins=48, expected_origins=48, verified_score_rows=144,
            all_forecast_unit_metrics_replayed=True, all_origin_fields_and_latents_exact=True,
            evaluation_sha256=sha256_file(forecasts / 'evaluation.json'),
            scores_sha256=evaluation['scores_sha256'], source_checkpoint_sha256=run['best_checkpoint_sha256']))
        names = ('training/run.json', 'forecasts/evaluation.json', 'replay.json')
        _json(cell / 'delivery.json', dict(status='PASS', exit_code=0, training_seed=seed,
            core_neurons=100000, complete_episodes=72, episode_ms=6000, fit_episodes=56,
            validation_episodes=16, verified_forecast_origins=48, score_rows=144,
            scientific_claim_qualified=False, calibration_or_test_accessed=False,
            files=[dict(path=name, bytes=(cell / name).stat().st_size,
                        sha256=sha256_file(cell / name)) for name in names]))
    return root, native


def test_population_complete_campaign_reuses_full_score_and_model_checks(cohort, population_campaign, tmp_path):
    root, _ = population_campaign
    output = tmp_path / 'population_analysis'
    report = analysis.analyze(root, cohort['acquisition'], cohort['fields'], output,
                              population_campaign=True)
    assert report['status'] == 'REPLAYED_DEVELOPMENT'
    assert report['models'] == ['population'] and report['score_rows'] == 432
    assert report['field_scores_recomputed'] == 2160 and report['cost_origin_rows'] == 144
    assert report['all_unit_metrics_independently_replayed'] is True
    assert report['scientific_claim_qualified'] is False
    assert report['paired_model_comparison_available'] is False
    assert not (output / 'forecast_paired_block_differences.csv').exists()
    rows = _read_rows(output / 'forecast_by_block.csv')
    assert len(rows) == 36 and {int(r['block']) for r in rows} == {14, 15, 16, 17}
    assert {int(r['seed']) for r in rows} == {41, 42, 43}
    assert all(float(r['unit_bin_mean_mse_corrected']) == -.75 for r in rows)
    assert report['common_training_budget']['latent_channels'] == 4


@pytest.mark.parametrize('failure', ['incomplete', 'foreign_replay', 'delivery_changed', 'missing_seed'])
def test_population_campaign_rejects_incomplete_or_unbound_replay(population_campaign, failure):
    root, _ = population_campaign
    cell = root / 'seed_41'
    if failure == 'missing_seed':
        with pytest.raises(FileNotFoundError):
            analysis._campaign_cell(root, 44)
        return
    if failure == 'delivery_changed':
        with (cell / 'replay.json').open('a') as stream:
            stream.write(' ')
    else:
        replay = json.loads((cell / 'replay.json').read_text())
        replay.update({'verified_origins': 47} if failure == 'incomplete'
                      else {'source_checkpoint_sha256': 'foreign'})
        _json(cell / 'replay.json', replay)
        delivery = json.loads((cell / 'delivery.json').read_text())
        row = next(r for r in delivery['files'] if r['path'] == 'replay.json')
        row.update(bytes=(cell / 'replay.json').stat().st_size, sha256=sha256_file(cell / 'replay.json'))
        _json(cell / 'delivery.json', delivery)
    with pytest.raises(ValueError, match='binding changed|all-origin replay'):
        analysis._campaign_cell(root, 41)


@pytest.mark.parametrize('failure', ['model_code', 'observation_buffer'])
def test_population_aggregation_reuses_checkpoint_source_rejection(cohort, population_campaign, failure):
    root, native = population_campaign
    canonical = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    canonical.acquisition, canonical.fields, canonical.population_native = cohort['acquisition'], cohort['fields'], native
    cell = root / 'seed_41'
    training, forecasts = cell / 'training', cell / 'forecasts'
    checkpoint = torch.load(training / 'best.pt', map_location='cpu', weights_only=True)
    if failure == 'model_code':
        checkpoint['binding']['model_implementation_sha256'] = 'changed'
    else:
        checkpoint['model']['unit_field_index'][0] = 1
    torch.save(checkpoint, training / 'best.pt')
    run = json.loads((training / 'run.json').read_text())
    run.update(binding=checkpoint['binding'], best_checkpoint_sha256=sha256_file(training / 'best.pt'))
    _json(training / 'run.json', run)
    evaluation = json.loads((forecasts / 'evaluation.json').read_text())
    evaluation['checkpoint_sha256'] = run['best_checkpoint_sha256']
    _json(forecasts / 'evaluation.json', evaluation)
    replay = json.loads((cell / 'replay.json').read_text())
    replay.update(source_checkpoint_sha256=run['best_checkpoint_sha256'],
                  evaluation_sha256=sha256_file(forecasts / 'evaluation.json'))
    _json(cell / 'replay.json', replay)
    delivery = json.loads((cell / 'delivery.json').read_text())
    for row in delivery['files']:
        row.update(bytes=(cell / row['path']).stat().st_size, sha256=sha256_file(cell / row['path']))
    _json(cell / 'delivery.json', delivery)
    code = _code()
    for name in ('population_state.py', 'native_count_windows.py'):
        code[name] = sha256_file(Path(analysis.__file__).with_name(name))
    with pytest.raises(ValueError, match='model implementation|state observation map'):
        analysis._load_run(root, 'population', 41, canonical, cohort['config'], code, campaign_layout=True)


def test_population_cli_incomplete_campaign_exits_nonzero(tmp_path):
    output = tmp_path / 'incomplete'
    command = [sys.executable, '-m', 'Libs.population_dynamics.population_forecast_analysis',
               '--forecasts', str(tmp_path), '--acquisition', str(tmp_path), '--fields', str(tmp_path),
               '--output', str(output), '--population-campaign']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert json.loads((output / 'analysis.json').read_text())['status'] == 'FAILED'


def baseline_aggregate_fixture(root, cohort, source, *, lfads=False):
    folder = root / 'analysis'
    folder.mkdir(parents=True)
    rows = []
    methods = ('renewal',) if lfads else analysis.MODELS
    for model, seed in itertools.product(methods, analysis.SEEDS):
        rows.extend(_read_rows(cohort['forecasts'] / model / f'seed_{seed}_forecast/scores.csv'))
    if lfads:
        for row in rows:
            row['method'] = 'lfads'
            for name, value in dict(input_loading_seconds=.02, sample_serialization_seconds=.04,
                                   graph_setup_seconds=3. + float(row['origin_ms']) / 1000.,
                                   scoring_seconds=.05, origin_total_seconds=.42,
                                   process_peak_rss_bytes=2000000., peak_tensorflow_gpu_allocated_bytes='').items():
                row[name] = value
            del row['peak_gpu_allocated_bytes']
    _csv(folder / 'forecast_all_seeds.csv', rows)
    costs = []
    for row in rows:
        if float(row['horizon_ms']) != max(cohort['config']['horizons_ms']):
            continue
        if lfads:
            cost = {name: row[name] for name in ('seed','case_id','origin_ms')}
            cost.update({name: value for name, value in row.items() if name.endswith(('_seconds','_bytes'))})
        else:
            cost = {name: row[name] for name in ('seed','block','case_id','template_id','origin_ms','method')}
            cost['generated_horizon_ms'] = 500.
            cost.update({name: row[name] for name in analysis.COST_METRICS})
        costs.append(cost)
    _csv(folder / 'costs_by_origin.csv', costs)
    config = dict(cohort['config'], scope='historical_development_label')
    _json(folder / 'analysis.json', dict(status='DESCRIBED_REPLAYED_DEVELOPMENT' if lfads else 'REPLAYED_DEVELOPMENT',
        score_rows=len(rows), scientific_claim_qualified=False, formal_test_accessed=False, configuration=config,
        source_binding=source.binding, model='lfads' if lfads else None,
        model_seed_bindings={'fixture': 'independently tested above'},
        costs='LFADS simulate includes conditional means and controller ancestry; setup repeated per graph lifetime' if lfads
              else analysis.CANDIDATE_COST_SCOPE,
        artifacts={'costs_by_origin.csv': sha256_file(folder / 'costs_by_origin.csv')}))
    names = ('analysis/analysis.json', 'analysis/forecast_all_seeds.csv', 'analysis/costs_by_origin.csv')
    _json(root / 'delivery.json', dict(status='ARCHIVED_VERIFIED_DEVELOPMENT', training_seeds=[41,42,43],
        calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        verification=dict(all_unit_metrics_independently_replayed=True, seeds=[41,42,43], score_rows=len(rows)),
        files=[dict(path=name, bytes=(root / name).stat().st_size, sha256=sha256_file(root / name)) for name in names]))
    return root


def test_population_comparison_reuses_complete_baselines_without_generation(cohort, population_campaign, tmp_path):
    root, _ = population_campaign
    source = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    baseline = baseline_aggregate_fixture(tmp_path / 'baseline', cohort, source)
    output = tmp_path / 'paired'
    report = analysis.analyze(root, cohort['acquisition'], cohort['fields'], output,
        population_campaign=True, baseline_analyses=[baseline])
    rows = _read_rows(output / 'forecast_paired_block_differences.csv')
    assert report['paired_model_comparison_available'] is True
    assert {row['baseline'] for row in rows} == {'poisson', 'renewal'}
    selected = [row for row in rows if row['metric'] == 'unit_bin_mean_mse_corrected']
    assert len(selected) == 24
    assert all(float(row['candidate_minus_baseline']) == (-.25 if row['baseline'] == 'poisson' else 0.) for row in selected)
    assert {int(row['block']) for row in selected} == {14,15,16,17}
    assert report['scientific_claim_qualified'] is False
    costs = _read_rows(output / 'cost_fidelity_by_origin.csv')
    assert len(costs) == 432 and {row['method'] for row in costs} == {'population','poisson','renewal'}
    assert all(row['horizon_ms'] == '500.0' and row['joint_samples'] == '16' for row in costs)
    assert all(row['source_unit_count'] == '100000' and row['field_count'] == '128' for row in costs)
    assert all(row['input_loading_seconds'] == '' and row['origin_total_seconds'] == '' for row in costs)
    assert report['end_to_end_cost_measured'] is False and report['speedup_qualified'] is False


@pytest.mark.parametrize('failure', ['source', 'protocol', 'missing_condition'])
def test_aggregate_comparison_rejects_unmatched_evidence(cohort, tmp_path, failure):
    source = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    baseline = baseline_aggregate_fixture(tmp_path / 'baseline', cohort, source)
    rows = []
    for seed in analysis.SEEDS:
        rows.extend(_read_rows(cohort['forecasts'] / 'renewal' / f'seed_{seed}_forecast/scores.csv'))
    candidate = []
    for raw in rows:
        row = dict(raw)
        for key in ('seed','block'): row[key] = int(row[key])
        for key in ('origin_ms','horizon_ms'): row[key] = float(row[key])
        for key in analysis._metric_names(cohort['config'])[0] + analysis._metric_names(cohort['config'])[1]:
            row[key] = None if raw[key] == '' else float(raw[key])
        candidate.append(row)
    path = baseline / 'analysis/analysis.json'
    record = json.loads(path.read_text())
    if failure == 'source':
        record['source_binding']['dataset_sha256'] = 'foreign'
    elif failure == 'protocol':
        record['configuration']['joint_samples'] = 8
    else:
        table = _read_rows(baseline / 'analysis/forecast_all_seeds.csv')
        table[-1] = dict(table[-2])
        _csv(baseline / 'analysis/forecast_all_seeds.csv', table)
    _json(path, record)
    delivery = json.loads((baseline / 'delivery.json').read_text())
    for row in delivery['files']:
        row.update(bytes=(baseline / row['path']).stat().st_size, sha256=sha256_file(baseline / row['path']))
    _json(baseline / 'delivery.json', delivery)
    with pytest.raises(ValueError, match='source cohort|measurement protocol|condition identities'):
        analysis._baseline_comparisons([baseline], dict(source_binding=source.binding), candidate, cohort['config'])



def _refresh_baseline_cost_binding(root):
    record = json.loads((root / 'analysis/analysis.json').read_text())
    record['artifacts']['costs_by_origin.csv'] = sha256_file(root / 'analysis/costs_by_origin.csv')
    _json(root / 'analysis/analysis.json', record)
    delivery = json.loads((root / 'delivery.json').read_text())
    for row in delivery['files']:
        row.update(bytes=(root / row['path']).stat().st_size, sha256=sha256_file(root / row['path']))
    _json(root / 'delivery.json', delivery)


def _baseline_candidate_rows(cohort):
    rows = []
    for seed in analysis.SEEDS:
        for raw in _read_rows(cohort['forecasts'] / 'renewal' / f'seed_{seed}_forecast/scores.csv'):
            row = dict(raw)
            for key in ('seed','block'):
                row[key] = int(row[key])
            for key in ('origin_ms','horizon_ms'):
                row[key] = float(row[key])
            for key in analysis._metric_names(cohort['config'])[0] + analysis._metric_names(cohort['config'])[1]:
                row[key] = None if raw[key] == '' else float(raw[key])
            rows.append(row)
    return rows


def test_lfads_cost_schema_uses_matched_field_identity_and_retains_unmeasured_costs(cohort, tmp_path):
    source = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    baseline = baseline_aggregate_fixture(tmp_path / 'lfads', cohort, source, lfads=True)
    joined = []
    comparisons, receipts = analysis._baseline_comparisons([baseline], dict(source_binding=source.binding),
        _baseline_candidate_rows(cohort), cohort['config'], cost_fidelity_rows=joined)
    assert comparisons and len(joined) == 144
    assert {row['method'] for row in joined} == {'lfads'}
    assert {row['block'] for row in joined} == {14,15,16,17}
    assert {row['template_id'] for row in joined} == {'familiar_0','familiar_1','familiar_2','familiar_3'}
    assert all(row['peak_gpu_allocated_bytes'] is None and row['peak_tensorflow_gpu_allocated_bytes'] is None for row in joined)
    assert all(row['sample_serialization_seconds'] == .04 and row['process_peak_rss_bytes'] == 2000000. for row in joined)
    assert sum(row['graph_setup_seconds'] is not None for row in joined) == 144
    assert all(row['scientific_claim_qualified'] is False for row in joined)
    assert receipts[0]['cost_artifact_sha256'] == sha256_file(baseline / 'analysis/costs_by_origin.csv')
    assert 'controller ancestry' in receipts[0]['cost_scope']


@pytest.mark.parametrize('lfads,failure', [(lfads,failure) for lfads,failure in itertools.product(
    [False,True], ['missing','duplicate','nonfinite','negative','fractional_bytes','wrong_horizon','sibling','graph_lifetime','corrupt_binding'])
    if lfads or failure != 'graph_lifetime'])
def test_baseline_cost_join_rejects_invalid_measured_evidence(cohort, tmp_path, lfads, failure):
    source = analysis.CanonicalFields(cohort['acquisition'], cohort['fields'])
    baseline = baseline_aggregate_fixture(tmp_path / 'baseline', cohort, source, lfads=lfads)
    path = baseline / 'analysis/costs_by_origin.csv'
    rows = _read_rows(path)
    if failure == 'missing':
        rows.pop()
    elif failure == 'duplicate':
        rows[-1] = dict(rows[0])
    elif failure == 'nonfinite':
        rows[0]['full_unit_generation_seconds'] = 'nan'
    elif failure == 'negative':
        rows[0]['inference_seconds'] = '-1'
    elif failure == 'fractional_bytes':
        rows[0]['process_peak_rss_bytes' if lfads else 'peak_gpu_allocated_bytes'] = '1.5'
    elif failure == 'wrong_horizon':
        for row in rows:
            row['generated_horizon_ms'] = '80'
    elif failure == 'graph_lifetime':
        rows[0]['graph_setup_seconds'] = '999'
        table = _read_rows(baseline / 'analysis/forecast_all_seeds.csv')
        for row in table:
            if (row['seed'],row['case_id'],row['origin_ms']) == (rows[0]['seed'],rows[0]['case_id'],rows[0]['origin_ms']):
                row['graph_setup_seconds'] = '999'
        _csv(baseline / 'analysis/forecast_all_seeds.csv', table)
    elif failure == 'sibling':
        table = _read_rows(baseline / 'analysis/forecast_all_seeds.csv')
        table[0]['inference_seconds'] = '999'
        _csv(baseline / 'analysis/forecast_all_seeds.csv', table)
    else:
        rows[0]['inference_seconds'] = '999'
    _csv(path, rows)
    if failure != 'corrupt_binding':
        _refresh_baseline_cost_binding(baseline)
    with pytest.raises(ValueError, match='cost|allocation|horizon|setup|finite|delivery'):
        analysis._baseline_comparisons([baseline], dict(source_binding=source.binding),
            _baseline_candidate_rows(cohort), cohort['config'])


@pytest.fixture
def delivered_representation_pair(tmp_path):
    """Small complete aggregate fixtures; no model fitting or source simulation."""
    config = yaml.safe_load(analysis.CONFIG_PATH.read_text())
    field_metrics, other_metrics = analysis._metric_names(config)
    roots = []
    for representation, offset, parameters in (("fixed", 1., 100), ("recurrent", 3., 200)):
        root = tmp_path / representation
        folder = root / "analysis"
        folder.mkdir(parents=True)
        campaign = tmp_path / (representation + "_campaign")
        budgets, deliveries = {}, {}
        budget = dict(representation=representation, maximum_epochs=20, encoder_size=16, latent_channels=4)
        for seed in analysis.SEEDS:
            cell = campaign / f"seed_{seed}"
            (cell / "training").mkdir(parents=True)
            run = dict(status="FITTED_DEVELOPMENT", model_family="population", representation=representation,
                binding=dict(seed=seed, config=budget), best_checkpoint_sha256="checkpoint" + str(seed),
                parameter_count=parameters, completed_epochs=20, best_epoch=19, stop_reason="epoch_budget",
                elapsed_seconds=offset, peak_process_rss_bytes=1000, hardware=dict(host="fixture", device="cpu"))
            _json(cell / "training/run.json", run)
            _json(cell / "delivery.json", dict(status="PASS", training_seed=seed))
            budgets[f"population_{seed}"] = dict(training_config=budget,
                training_report_sha256=sha256_file(cell / "training/run.json"),
                checkpoint_sha256=run["best_checkpoint_sha256"], executed_training_budget=dict(
                    completed_epochs=20, best_epoch=19, stop_reason="epoch_budget", effective_maximum_epochs=20))
            deliveries[str(seed)] = sha256_file(cell / "delivery.json")
        rows, costs = [], []
        for seed, block, template, origin in itertools.product(analysis.SEEDS, analysis.BLOCKS, range(4), config['origins_ms']):
            case = f"train_block_{block:03d}_familiar_{template}"
            cost = dict(seed=seed, block=block, case_id=case, template_id=f"familiar_{template}",
                        origin_ms=origin, method="population", generated_horizon_ms=500.,
                        inference_seconds=offset, full_unit_generation_seconds=offset+1,
                        field_projection_seconds=offset+2, peak_gpu_allocated_bytes=None)
            costs.append(cost)
            for horizon in config["horizons_ms"]:
                row = {k: cost[k] for k in ("seed", "block", "case_id", "template_id", "origin_ms", "method")}
                row.update(horizon_ms=horizon, ensemble_artifact=f"{case}_origin_{origin:g}.npz")
                bins = horizon / 5.
                for name in field_metrics + other_metrics:
                    if name in analysis.COST_METRICS:
                        row[name] = cost[name]
                    elif name.startswith("variogram_temporal_lag_") and int(name.rsplit("_", 1)[1]) >= bins:
                        row[name] = None
                    elif name == "unit_bin_mean_mse_corrected":
                        row[name] = -offset
                    elif name == "unit_total_coverage_90":
                        row[name] = .9
                    elif name.startswith("refractory_count_violation_"):
                        row[name] = 0.
                    else:
                        row[name] = offset
                rows.append(row)
        _csv(folder / "forecast_all_seeds.csv", rows)
        _csv(folder / "costs_by_origin.csv", costs)
        report = dict(status="REPLAYED_DEVELOPMENT", models=["population"], seeds=list(analysis.SEEDS),
            score_rows=432, cost_origin_rows=144, independent_blocks=4, scientific_claim_qualified=False,
            source_fidelity_qualified=False, formal_test_accessed=False, all_unit_metrics_independently_replayed=True,
            all_observed_field_counts_checked_against_source=True, configuration=config, common_training_budget=budget,
            inputs=dict(forecasts=str(campaign)), model_seed_bindings=budgets, campaign_deliveries=deliveries,
            source_binding=dict(dataset_sha256="shared_dataset", field_bindings=dict(fixture="shared_observation_map")),
            code_sha256=dict(population_state="shared_model"),
            artifacts={name: sha256_file(folder / name) for name in ("forecast_all_seeds.csv", "costs_by_origin.csv")})
        _json(folder / "analysis.json", report)
        _rebind_population_delivery(root)
        roots.append(root)
    return roots


def test_covariance_decomposition_preserves_signed_components_and_rate_sensitivity():
    observed = np.full((2, 3), 2.)
    within = np.full((2, 3), .8)
    between = np.full((2, 3), -.1)
    total = between + (1 - 1 / 16) * within
    result = analysis._covariance_decomposition(np.stack((observed, total, within, between)), 16, 2.)
    assert result['corrected_between_covariance_mean'] == pytest.approx(-.15)
    assert result['predicted_covariance_mean'] == pytest.approx(.65)
    assert result['scaled_corrected_between_covariance_mean'] == pytest.approx(-.6)
    assert result['scaled_predicted_covariance_mean'] == pytest.approx(2.6)
    assert result['maximum_reconstruction_residual'] < 1e-15


@pytest.mark.parametrize('failure', ['changed_total', 'nonfinite', 'bad_draws', 'bad_ratio'])
def test_covariance_decomposition_rejects_invalid_components(failure):
    payload = np.ones((4, 2, 3))
    payload[1] = payload[3] + (1 - 1 / 16) * payload[2]
    draws, ratio = 16, 1.
    if failure == 'changed_total': payload[1, 0, 0] += .01
    if failure == 'nonfinite': payload[0, 0, 0] = np.nan
    if failure == 'bad_draws': draws = 1
    if failure == 'bad_ratio': ratio = np.inf
    with pytest.raises(ValueError): analysis._covariance_decomposition(payload, draws, ratio)


def _saved_covariance_fixture(root):
    report = json.loads((root / 'analysis/analysis.json').read_text())
    rows, arrays = [], {}
    config = report['configuration']
    for seed, template, origin, horizon in itertools.product(analysis.SEEDS, range(4), config['origins_ms'], config['horizons_ms']):
        requests = [('spatial', 0), ('cross_population', 0)] + [
            (f'temporal_lag_{lag}', lag) for lag in config['temporal_lags_bins'] if lag < horizon / 5]
        for statistic, lag in requests:
            shape = (int(horizon / 5) - lag, 2)
            within, between, observed = np.full(shape, .8), np.full(shape, -.1), np.full(shape, 2.)
            total = between + 15 / 16 * within
            key = f'g{len(rows)}'
            arrays[key] = np.stack((observed, total, within, between))
            rows.append(dict(seed=seed, method='population', template_id=f'familiar_{template}', origin_ms=origin,
                horizon_ms=horizon, statistic=statistic, independent_blocks=4, pairs=2, phases=shape[0],
                observed_covariance_mean=2., predicted_covariance_mean=.65, covariance_mae=1.35, array_key=key))
    _csv(root / 'analysis/covariance_diagnostic.csv', rows)
    np.savez(root / 'analysis/covariance_arrays.npz', **arrays)
    report.update(covariance_rows=len(rows), covariance_scope='four blocks at matched phase',
        covariance_array_axis=['observed', 'predicted_total', 'mean_within', 'between_sample_means'])
    names = ['covariance_diagnostic.csv', 'covariance_arrays.npz']
    report['artifacts'].update({name: sha256_file(root / 'analysis' / name) for name in names})
    _json(root / 'analysis/analysis.json', report)
    _rebind_population_delivery(root)
    receipt = json.loads((root / 'delivery.json').read_text())
    receipt['files'] += [dict(path='analysis/' + name, bytes=(root / 'analysis' / name).stat().st_size,
                             sha256=sha256_file(root / 'analysis' / name)) for name in names]
    _json(root / 'delivery.json', receipt)


def test_saved_covariance_real_cli_and_changed_archive_rejection(delivered_representation_pair, tmp_path):
    root = delivered_representation_pair[1]
    _saved_covariance_fixture(root)
    output = tmp_path / 'decomposition'
    command = [sys.executable, '-m', 'population_dynamics.population_forecast_analysis',
               '--covariance-analysis', str(root), '--output', str(output)]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads((output / 'analysis.json').read_text())
    assert result['covariance_rows'] == 432 and result['score_rows'] == 432
    assert result['scientific_claim_qualified'] is False
    with (output / 'covariance_component_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(float(row['corrected_between_covariance_mean']) < 0 for row in rows)
    assert all(float(row['rate_ratio']) == 1 for row in rows)
    path = root / 'analysis/covariance_arrays.npz'
    with path.open('ab') as stream: stream.write(b'changed')
    rejected = subprocess.run(command[:-1] + [str(tmp_path / 'rejected')], capture_output=True, text=True)
    assert rejected.returncode != 0
    assert 'saved covariance binding changed' in rejected.stderr
    assert not (tmp_path / 'rejected').exists()


def test_latent_projection_matches_direct_unit_log_hazards_and_process_variance():
    rng = np.random.default_rng(73)
    latents = rng.normal(size=(2, 4, 3, 3, 2))
    weights = rng.normal(size=(5, 2))
    fields = np.asarray([0, 0, 0, 1, 1])
    process = np.asarray([[.2, .3], [.4, .5], [.6, .7]])
    result = analysis._latent_projection(latents, weights, fields, process)
    direct = np.einsum('bdtuk,uk->bdtu', latents[:, :, :, fields], weights)
    for field in range(2):
        selected = fields == field
        assert np.allclose(result['unit_log_hazard_variance'][:, :, field],
                           direct[:, :, :, selected].var(1, ddof=1).mean(-1))
        mean_hazards = direct[:, :, :, selected].mean(-1)
        assert np.allclose(result['field_mean_log_hazards'][:, :, :, field], mean_hazards)
        assert np.allclose(result['field_mean_log_hazard_variance'][:, :, field],
                           mean_hazards.var(1, ddof=1))
        assert result['projected_process_unit_variance'][field] == pytest.approx(
            (weights[selected] ** 2 * process[field]).sum(-1).mean())
    assert np.array_equal(result['unit_counts'], [3, 2, 0])
    assert np.count_nonzero(result['field_mean_log_hazards'][:, :, :, 2]) == 0
    assert result['projected_process_unit_variance'][2] == 0


@pytest.mark.parametrize('failure', ['nonfinite_latent', 'foreign_unit_field', 'zero_process', 'loading_channels'])
def test_latent_projection_rejects_invalid_measurement_maps(failure):
    z, weights = np.ones((2, 3, 2, 2, 2)), np.ones((4, 2))
    fields, process = np.asarray([0, 0, 1, 1]), np.ones((2, 2))
    if failure == 'nonfinite_latent': z[0, 0, 0, 0, 0] = np.inf
    if failure == 'foreign_unit_field': fields[0] = 2
    if failure == 'zero_process': process[0, 0] = 0
    if failure == 'loading_channels': weights = weights[:, :1]
    with pytest.raises(ValueError): analysis._latent_projection(z, weights, fields, process)


def test_latent_analysis_rejects_checkpoint_change_before_output(delivered_representation_pair, tmp_path):
    root = delivered_representation_pair[1]
    report = json.loads((root / 'analysis/analysis.json').read_text())
    checkpoint = Path(report['inputs']['forecasts']) / 'seed_41/training/best.pt'
    checkpoint.write_bytes(b'foreign checkpoint')
    output = tmp_path / 'latent_rejected'
    with pytest.raises(ValueError, match='latent projection checkpoint changed'):
        analysis.describe_saved_latents(root, output)
    assert not output.exists()


def _rebind_population_delivery(root):
    names = ("analysis/analysis.json", "analysis/forecast_all_seeds.csv", "analysis/costs_by_origin.csv")
    _json(root / 'delivery.json', dict(status='PASS', exit_code=0, mode='complete',
        training_seeds=[41,42,43], scientific_claim_qualified=False, calibration_or_test_accessed=False,
        files=[dict(path=name, bytes=(root/name).stat().st_size, sha256=sha256_file(root/name)) for name in names]))


def _check_saved_baseline_comparison(population, tmp_path, baseline_method):
    path = population/'analysis/analysis.json'
    report = json.loads(path.read_text())
    report['source_binding'].update(acquisition_design_sha256='shared_design', native_source_identity={'graph':'shared'})
    _json(path, report)
    _rebind_population_delivery(population)
    baseline = tmp_path/'saved_baseline'
    (baseline/'analysis').mkdir(parents=True)
    rows = _read_rows(population/'analysis/forecast_all_seeds.csv')
    for row in rows:
        row['method']=baseline_method
        row['graph_setup_seconds']=3. if int(row['block'])<16 else 5.
    _csv(baseline/'analysis/forecast_all_seeds.csv',rows)
    costs=[]
    for row in rows:
        if float(row['horizon_ms'])!=500.:
            continue
        group='parent' if int(row['block'])<16 else 'current'
        charge=float(row['graph_setup_seconds']) if row['case_id']==('train_block_014_familiar_0' if group=='parent' else 'train_block_016_familiar_0') else 0.
        cost={key:row[key] for key in ('seed','case_id','origin_ms','method',*analysis.COST_METRICS)}
        cost.update(graph_setup_seconds=row['graph_setup_seconds'],graph_setup_id=group+row['origin_ms'],
            graph_setup_cases=8,graph_setup_source=group,graph_setup_charged_seconds=charge)
        costs.append(cost)
    _csv(baseline/'analysis/costs_by_origin.csv',costs)
    names=('analysis/analysis.json','analysis/forecast_all_seeds.csv','analysis/costs_by_origin.csv')
    _json(baseline/names[0],dict(status='DESCRIBED_REPLAYED_DEVELOPMENT',score_rows=432,
        scientific_claim_qualified=False,formal_test_accessed=False,configuration=report['configuration'],
        source_binding=report['source_binding'],model_seed_bindings=report['model_seed_bindings'],
        costs='raw and once-charged setup by measured lifetime',
        artifacts={'costs_by_origin.csv':sha256_file(baseline/names[2])}))
    _json(baseline/'delivery.json',dict(status='ARCHIVED_VERIFIED_DEVELOPMENT',training_seeds=[41,42,43],
        calibration_outcomes_accessed=False,test_outcomes_accessed=False,
        verification=dict(all_unit_metrics_independently_replayed=True,seeds=[41,42,43],score_rows=432),
        files=[dict(path=name,bytes=(baseline/name).stat().st_size,sha256=sha256_file(baseline/name)) for name in names]))
    output=tmp_path/'saved_comparison'
    result=subprocess.run([sys.executable,'-m','population_dynamics.population_forecast_analysis',
        '--population-analysis',str(population),'--baseline-analysis',str(baseline),'--output',str(output)],
        text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    result_report=json.loads((output/'analysis.json').read_text())
    assert result_report['new_forecasts']==0 and result_report['cost_fidelity_origins']==288
    joined=_read_rows(output/'cost_fidelity_by_origin.csv')
    assert sum(float(row['graph_setup_charged_seconds']) for row in joined if row['graph_setup_id'])==72.
    assert result_report['scientific_claim_qualified'] is False
    costs[0]['graph_setup_charged_seconds']=0.
    with pytest.raises(ValueError,match='charged exactly once'):
        analysis._origin_cost_fidelity(rows,costs,report['configuration'],'scoped')


def test_saved_baselines_compare_through_real_cli_with_restart_costs(delivered_representation_pair, tmp_path):
    _check_saved_baseline_comparison(delivered_representation_pair[1], tmp_path,
                                    "lfads_template_time_controller_mean_prior_budget20")


def test_delivered_representation_comparison_through_real_cli(delivered_representation_pair, tmp_path):
    roots = delivered_representation_pair
    output = tmp_path / 'comparison'
    result = subprocess.run([sys.executable, '-m', 'population_dynamics.population_forecast_analysis',
        '--population-analysis', str(roots[1]), '--population-analysis', str(roots[0]), '--output', str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/'analysis.json').read_text())
    assert report['status'] == 'DESCRIBED_REPLAYED_DEVELOPMENT'
    assert report['paired_forecast_conditions'] == 432 and report['score_rows'] == 864
    assert report['cost_fidelity_origin_rows'] == 288
    assert report['state_selection_enabled'] is False and report['scientific_claim_qualified'] is False
    blocks = _read_rows(output/'representation_paired_blocks.csv')
    assert all(float(row['recurrent_minus_fixed']) == (-2. if row['metric'] == 'unit_bin_mean_mse_corrected'
        else 0. if row['metric'] == 'unit_total_coverage_90' or row['metric'].startswith('refractory_count_') else 2.) for row in blocks)
    training = _read_rows(output/'representation_training.csv')
    assert len(training) == 6 and {r['parameter_count'] for r in training} == {'100','200'}
    costs = _read_rows(output/'representation_cost_differences.csv')
    assert all(r['recurrent_minus_fixed'] == '' for r in costs if r['metric'] == 'origin_total_seconds')
    assert all(float(r['recurrent_minus_fixed']) == 2. for r in costs if r['metric'] == 'inference_seconds')
    assert not any('p_value' in r or 'q_value' in r for r in blocks)
    for name, digest in report['artifacts'].items():
        assert sha256_file(output/name) == digest


@pytest.mark.parametrize('failure', ['delivery_changed', 'source', 'code', 'config', 'budget',
    'training_metadata', 'same_representation', 'missing_cost', 'duplicate_condition', 'incomplete', 'test_access'])
def test_representation_comparison_rejects_unmatched_evidence(delivered_representation_pair, tmp_path, failure):
    roots = delivered_representation_pair
    root = roots[1]
    path = root/'analysis/analysis.json'
    report = json.loads(path.read_text())
    if failure == 'delivery_changed':
        path.write_text(path.read_text()+' ')
    elif failure == 'training_metadata':
        p=Path(report['inputs']['forecasts'])/'seed_41/training/run.json'
        p.write_text(p.read_text()+' ')
    elif failure in ('missing_cost','duplicate_condition'):
        table=root/('analysis/costs_by_origin.csv' if failure=='missing_cost' else 'analysis/forecast_all_seeds.csv')
        rows=_read_rows(table)
        rows.pop() if failure=='missing_cost' else rows.__setitem__(1,dict(rows[0]))
        _csv(table,rows)
        report['artifacts'][table.name]=sha256_file(table)
        _json(path,report);_rebind_population_delivery(root)
    elif failure == 'test_access':
        proof=json.loads((root/'delivery.json').read_text());proof['calibration_or_test_accessed']=True
        _json(root/'delivery.json',proof)
    else:
        if failure=='source': report['source_binding']['field_bindings']['fixture']='foreign_map'
        if failure=='code': report['code_sha256']['population_state']='foreign_model'
        if failure=='config': report['configuration']['sampling_seed_base']+=1
        if failure=='budget': report['common_training_budget']['encoder_size']+=1
        if failure=='same_representation': roots=[roots[0],roots[0]]
        if failure=='incomplete': report['score_rows']=431
        _json(path,report);_rebind_population_delivery(root)
    output=tmp_path/failure
    with pytest.raises(ValueError): analysis.compare_population_analyses(roots,output)
    assert json.loads((output/'analysis.json').read_text())['status']=='FAILED'
    assert not (output/'forecast_all_seeds.csv').exists()


def test_resumed_training_cost_includes_each_bound_attempt_once(tmp_path):
    cell=tmp_path/'seed_42';(cell/'training/attempts').mkdir(parents=True)
    binding=dict(config=dict(maximum_epochs=20),seed=42)
    run=dict(attempt=2,training_seed=42,binding=binding,elapsed_seconds=5.)
    rows=[]
    for attempt,seconds in [(0,100.),(1,20.)]:
        name=f'training/attempts/attempt_{attempt:03d}.json'
        _json(cell/name,dict(attempt=attempt,training_seed=42,binding=binding,elapsed_seconds=seconds))
        rows.append(dict(path=name,bytes=(cell/name).stat().st_size,sha256=sha256_file(cell/name)))
    _json(cell/'delivery.json',dict(files=rows))
    result=analysis._training_attempt_cost(cell,run)
    assert result['training_elapsed_seconds']==125.
    assert result['training_attempts']==3
    assert 'unmeasured' in result['training_cost_scope']
    (cell/rows[0]['path']).write_text('{}')
    with pytest.raises(ValueError,match='binding changed'):analysis._training_attempt_cost(cell,run)


def test_resumed_training_cost_requires_complete_prior_attempts(tmp_path):
    cell=tmp_path/'seed_42';cell.mkdir();_json(cell/'delivery.json',dict(files=[]))
    with pytest.raises(ValueError,match='unavailable'):
        analysis._training_attempt_cost(cell,dict(attempt=1,elapsed_seconds=5.))
    fresh=analysis._training_attempt_cost(cell,dict(elapsed_seconds=5.))
    assert fresh['training_elapsed_seconds']==5. and fresh['training_attempts']==1


def _prior_lineage_fixture(root, seed=41):
    """A bound 20-to-80 parent and a new two-attempt prior objective."""
    base, parent, child = [root / kind / f"seed_{seed}" for kind in ("base", "parent", "prior")]
    config = dict(schema="neural_population_state_trajectory_training_v1", model_family="population",
        representation="recurrent", maximum_epochs=20, early_stopping_patience=4)
    binding = dict(config=config, seed=seed, observation_implementation_sha256="native-observation",
                   **{key: "same-" + key for key in analysis.PRIOR_SOURCE_BINDINGS})
    constructor = dict(unit_ids=[1, 2], bin_seconds=.005, tick_seconds=.0001)
    model = {"encoder.weight": torch.tensor([1.]), "unit_loading": torch.tensor([2.]),
             "local_weight": torch.tensor([3.])}
    sources = {"trainer": "original"}

    def row(folder, name):
        return dict(path=name, bytes=(folder / name).stat().st_size, sha256=sha256_file(folder / name))

    def write_training(folder, run, payload):
        (folder / "training").mkdir(parents=True, exist_ok=True)
        torch.save(payload, folder / "training/best.pt")
        shutil.copyfile(folder / "training/best.pt", folder / "training/latest.pt")
        run.update(best_checkpoint_sha256=sha256_file(folder / "training/best.pt"),
                   latest_checkpoint_sha256=sha256_file(folder / "training/latest.pt"))
        _json(folder / "training/run.json", run)
        return [row(folder, name) for name in ("training/run.json", "training/best.pt", "training/latest.pt")]

    proof = dict(status="PASS", exit_code=0, training_seed=seed, core_neurons=100000,
        complete_episodes=72, episode_ms=6000, fit_episodes=56, validation_episodes=16,
        scientific_claim_qualified=False, calibration_or_test_accessed=False)
    original = dict(status="FITTED_DEVELOPMENT", training_seed=seed, binding=binding,
        model_family="population", representation="recurrent", model_configuration=constructor,
        parameter_count=3, hardware=dict(host="fixture", device="cpu"), peak_process_rss_bytes=1000,
        completed_epochs=20, best_epoch=19, effective_maximum_epochs=20, stop_reason="epoch_budget",
        continuation=None, attempt=0, elapsed_seconds=100., trainer_sources=sources)
    original_checkpoint = dict(binding=binding, model=model, model_configuration=constructor,
                               next_epoch=20, continuation=None)
    base_rows = write_training(base, original, original_checkpoint)
    _json(base / "delivery.json", dict(proof, files=base_rows))
    continuation = dict(schema="neural_population_trajectory_budget_continuation_v1",
        base_maximum_epochs=20, total_epochs=80, parent_effective_maximum_epochs=20,
        inherited_early_stopping_patience=4, parent_completed_epochs=20,
        parent_checkpoint_sha256=original["latest_checkpoint_sha256"], parent_binding=binding,
        parent_artifact_sha256={"run.json": sha256_file(base / "training/run.json"),
                               "best.pt": original["best_checkpoint_sha256"]},
        parent_source_hashes=sources, current_source_hashes=sources, source_migration={})
    parent_run = dict(original, completed_epochs=80, best_epoch=79, effective_maximum_epochs=80,
                      continuation=continuation, attempt=2, elapsed_seconds=20.)
    parent_checkpoint = dict(original_checkpoint, next_epoch=80, continuation=continuation)
    parent_rows = write_training(parent, parent_run, parent_checkpoint)
    for name in ("delivery.json", "training/run.json"):
        target = parent / "prerequisites/budget_parent" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
        parent_rows.append(row(parent, str(target.relative_to(parent))))
    (parent / "training/attempts").mkdir()
    _json(parent / "training/attempts/attempt_001.json", dict(parent_run, attempt=1, elapsed_seconds=60.))
    parent_rows.append(row(parent, "training/attempts/attempt_001.json"))
    _json(parent / "delivery.json", dict(proof, files=parent_rows, optimization_budget_continuation=dict(
        parent=str(base), parent_delivery_sha256=sha256_file(base / "delivery.json"),
        parent_checkpoint_sha256=original["latest_checkpoint_sha256"], parent_completed_epochs=20, total_epochs=80)))
    initialization = dict(mode="selected_parent_prior_refit", parent_training="/removed/old-scratch/parent_training",
        parent_checkpoint_sha256=parent_run["best_checkpoint_sha256"], parent_report_sha256=sha256_file(parent / "training/run.json"),
        parent_selected_epoch=79, parent_completed_epochs=80, parent_binding=binding,
        trainable_parameter_names=["local_weight", "field_mix_logits", "control_weight", "transition_bias", "decay_logit", "process_log_variance"],
        optimizer_and_random_streams_reinitialized=True)
    prior_config = dict(config, schema=analysis.PRIOR_REFIT_SCHEMA, maximum_epochs=2, early_stopping_patience=12,
        observation_objective=analysis.PRIOR_REFIT_OBJECTIVE, training_sequence="complete_episode_causal_posterior_cache",
        prior_origins_ms=[1000, 3000, 5000], prior_horizons_ms=[5, 20, 80, 500],
        prior_selection="fixed_draw_validation_multistep_kl", prior_target_gradient="stopped",
        parent_checkpoint_sha256={str(value): parent_run["best_checkpoint_sha256"] for value in analysis.SEEDS})
    run = dict(original, binding=dict(binding, config=prior_config), completed_epochs=2, best_epoch=1,
        effective_maximum_epochs=2, attempt=1, elapsed_seconds=6., trainer_sources={"trainer": "prior"},
        training_objective=analysis.PRIOR_REFIT_OBJECTIVE, selection_metric="validation_multistep_prior_kl",
        complete_episode_updates=False, complete_source_episode_encoding=True, posterior_targets_fixed=True,
        observation_parameters_refitted=False, observation_likelihood_evaluated_during_refit=False,
        fitting_episodes=56, validation_episodes=16, fitting_origin_windows=168, validation_origin_windows=48,
        units=100000, refit_initialization=initialization, frozen_parameters_unchanged_in_attempt=True,
        calibration_outcomes_accessed=False, test_outcomes_accessed=False)
    checkpoint = dict(binding=run["binding"], model=dict(model, local_weight=torch.tensor([4.])),
        model_configuration=constructor, next_epoch=2, continuation=None, initialization=initialization)
    rows = write_training(child, run, checkpoint)
    (child / "training/attempts").mkdir()
    _json(child / "training/attempts/attempt_000.json", dict(run, attempt=0, elapsed_seconds=4.))
    rows.append(row(child, "training/attempts/attempt_000.json"))
    (child / "parent_training").mkdir()
    for name in ("run.json", "best.pt"):
        shutil.copyfile(parent / "training" / name, child / "parent_training" / name)
        rows.append(row(child, "parent_training/" + name))
    (child / "prerequisites").mkdir()
    _json(child / "prerequisites/prior_parent.json", dict(parent_cell=str(parent),
        delivery_sha256=sha256_file(parent / "delivery.json"),
        files=[entry for entry in parent_rows if entry["path"] in ("training/run.json", "training/best.pt")]))
    rows.append(row(child, "prerequisites/prior_parent.json"))
    (child / "forecasts").mkdir()
    _json(child / "forecasts/evaluation.json", dict(status="EVALUATED_DEVELOPMENT", scores_sha256="scores"))
    _json(child / "replay.json", dict(status="PASS", model_type="population", training_seed=seed,
        verified_origins=48, expected_origins=48, verified_score_rows=144,
        all_forecast_unit_metrics_replayed=True, all_origin_fields_and_latents_exact=True,
        evaluation_sha256=sha256_file(child / "forecasts/evaluation.json"), scores_sha256="scores",
        source_checkpoint_sha256=run["best_checkpoint_sha256"]))
    rows += [row(child, name) for name in ("forecasts/evaluation.json", "replay.json")]
    _json(child / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
        source_units=100000, fit_episodes=56, validation_episodes=16, scientific_claim_qualified=False,
        complete_origins=48, verified_origins=48, score_rows=144, frozen_parameters_unchanged=True,
        parent_cell=str(parent), parent_delivery_sha256=sha256_file(parent / "delivery.json"), files=rows))
    return base, parent, child, parent_run, parent_checkpoint, run, checkpoint


def test_prior_native_delivery_schema_preserves_full_source_and_replay(tmp_path):
    _, _, child, _, _, run, checkpoint = _prior_lineage_fixture(tmp_path)
    assert analysis._campaign_cell(child.parent, 41) == child
    lineage = analysis._prior_parent_evidence(child, run, checkpoint)[2]
    assert lineage["parent_completed_epochs"] == 80 and lineage["parent_selected_epoch"] == 79
    assert not Path(run["refit_initialization"]["parent_training"]).exists()
    delivery = json.loads((child / "delivery.json").read_text())
    assert "core_neurons" not in delivery and "calibration_or_test_accessed" not in delivery
    delivery["status"] = "COMMITTED_TRAINING_SLICE"
    _json(child / "delivery.json", delivery)
    with pytest.raises(ValueError, match="complete registered"):
        analysis._campaign_cell(child.parent, 41)


def test_prior_cost_includes_full_budget_parent_and_each_refit_attempt_once(tmp_path):
    _, _, child, _, _, run, _ = _prior_lineage_fixture(tmp_path)
    cost = analysis._training_attempt_cost(child, run)
    assert cost["parent_training_elapsed_seconds"] == 180.
    assert cost["incremental_training_elapsed_seconds"] == 10.
    assert cost["training_elapsed_seconds"] == 190. and cost["training_attempts"] == 5
    assert cost["parent_training_attempts"] == 3 and cost["incremental_training_attempts"] == 2
    assert cost["launcher_staging_seconds"] is None and cost["launcher_archive_seconds"] is None
    assert cost["full_pipeline_elapsed_seconds"] is None and cost["full_pipeline_cost_measured"] is False


@pytest.mark.parametrize("failure", ["parent_report", "parent_seed", "source", "frozen_parameter", "frozen_dtype", "staged_parent", "parent_attempt", "refit_attempt"])
def test_prior_lineage_rejects_unbound_sources_states_and_costs(tmp_path, failure):
    _, parent, child, _, _, run, checkpoint = _prior_lineage_fixture(tmp_path)
    if failure == "parent_report": run["refit_initialization"]["parent_report_sha256"] = "foreign"
    elif failure == "parent_seed": run["refit_initialization"]["parent_binding"]["seed"] = 42
    elif failure == "source": run["binding"]["dataset_sha256"] = "foreign"
    elif failure == "frozen_parameter": checkpoint["model"]["encoder.weight"] = torch.zeros(1)
    elif failure == "frozen_dtype": checkpoint["model"]["encoder.weight"] = checkpoint["model"]["encoder.weight"].double()
    elif failure == "staged_parent": (child / "parent_training/run.json").write_text("{}")
    elif failure == "parent_attempt": (parent / "training/attempts/attempt_001.json").unlink()
    else: (child / "training/attempts/attempt_000.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError), match="prior|parent|binding|frozen|No such file"):
        if failure in ("parent_attempt", "refit_attempt"):
            analysis._training_attempt_cost(child, run)
        else:
            analysis._prior_parent_evidence(child, run, checkpoint)


@pytest.fixture
def delivered_prior_pair(delivered_representation_pair, tmp_path):
    child_root, parent_root = delivered_representation_pair
    cells = [_prior_lineage_fixture(tmp_path / "prior_lineage", seed) for seed in analysis.SEEDS]
    for prior, root in ((False, parent_root), (True, child_root)):
        report = json.loads((root / "analysis/analysis.json").read_text())
        bindings, deliveries = {}, {}
        for seed, data in zip(analysis.SEEDS, cells):
            cell, run, checkpoint = (data[2], data[5], data[6]) if prior else (data[1], data[3], data[4])
            receipt = dict(training_config=run["binding"]["config"],
                training_report_sha256=sha256_file(cell / "training/run.json"), checkpoint_sha256=run["best_checkpoint_sha256"],
                executed_training_budget=analysis._training_budget(run, checkpoint, run["best_checkpoint_sha256"]))
            if prior:
                receipt.update(prior_refit_lineage=analysis._prior_parent_evidence(cell, run, checkpoint)[2],
                    training_objective=run["training_objective"], selection_metric=run["selection_metric"])
            bindings[f"population_{seed}"] = receipt
            deliveries[str(seed)] = sha256_file(cell / "delivery.json")
        budget = dict(representation="recurrent", maximum_epochs=2 if prior else 80, base_maximum_epochs=2 if prior else 20,
                      encoder_size=16, latent_channels=4)
        if prior: budget.update(training_objective=analysis.PRIOR_REFIT_OBJECTIVE, selection_metric="validation_multistep_prior_kl")
        report.update(inputs=dict(forecasts=str(cells[0][2 if prior else 1].parent)),
            model_seed_bindings=bindings, campaign_deliveries=deliveries, common_training_budget=budget)
        if prior: report["code_sha256"]["population_prior_objective.py"] = "declared-new-objective"
        _json(root / "analysis/analysis.json", report)
        _rebind_population_delivery(root)
    return parent_root, child_root


def test_prior_objective_comparison_real_cli_preserves_cost_and_labels(delivered_prior_pair, tmp_path):
    parent, child = delivered_prior_pair
    output = tmp_path / "prior_comparison"
    result = subprocess.run([sys.executable, "-m", "population_dynamics.population_forecast_analysis",
        "--prior-objective-comparison", "--population-analysis", str(child), "--population-analysis", str(parent),
        "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "analysis.json").read_text())
    assert report["comparison_mode"] == "prior_objective" and report["paired_forecast_conditions"] == 432
    assert report["models"] == ["population_recurrent_parent", "population_recurrent_prior_refit"]
    assert report["inherited_training_cost_included"] is True and report["launcher_staging_cost_measured"] is False
    assert "common_training_budget" not in report and report["scientific_claim_qualified"] is False
    assert report["implementation_differences"]["population_prior_objective.py"]["parent"] is None
    rows = _read_rows(output / "prior_objective_training.csv")
    own = [row for row in rows if row["method"] == "population_recurrent_prior_refit"]
    assert len(own) == 3 and all(float(row["training_elapsed_seconds"]) == 190. for row in own)
    assert all(row["training_objective"] == analysis.PRIOR_REFIT_OBJECTIVE for row in own)
    assert all(row["launcher_staging_seconds"] == "" for row in own)
    for mode in ({}, {"budget_comparison": True}):
        with pytest.raises(ValueError):
            analysis.compare_population_analyses([parent, child], tmp_path / ("budget" if mode else "representation"), **mode)


def test_prior_objective_comparison_rejects_foreign_source(delivered_prior_pair, tmp_path):
    parent, child = delivered_prior_pair
    report = json.loads((child / "analysis/analysis.json").read_text())
    report["source_binding"]["dataset_sha256"] = "foreign"
    _json(child / "analysis/analysis.json", report)
    _rebind_population_delivery(child)
    with pytest.raises(ValueError, match="source or measurement"):
        analysis.compare_population_analyses([parent, child], tmp_path / "comparison", prior_objective_comparison=True)


@pytest.mark.parametrize("baseline_method", ["ldns", "classic_state_space"])
def test_saved_prior_baseline_comparison_retains_method_objective_and_parent_cost(delivered_prior_pair, tmp_path, baseline_method):
    _check_saved_baseline_comparison(delivered_prior_pair[1], tmp_path, baseline_method)
    output = tmp_path / "saved_comparison"
    rows = _read_rows(output / "forecast_paired_block_differences.csv")
    assert {row["candidate"] for row in rows} == {"population_recurrent_prior_refit"}
    assert {row["baseline"] for row in rows} == {baseline_method}
    assert {row["candidate_training_objective"] for row in rows} == {analysis.PRIOR_REFIT_OBJECTIVE}
    resources = _read_rows(output / "population_training_resources.csv")
    assert all(float(row["training_elapsed_seconds"]) == 190. for row in resources)
    assert all(row["method"] == "population_recurrent_prior_refit" for row in resources)


@pytest.mark.parametrize("case", ["reviewed", "unknown", "tampered"])
def test_prior_observation_compatibility_requires_reviewed_delivered_source(tmp_path, monkeypatch, case):
    name = "Libs/population_dynamics/refractory_observation.py"
    parent, child = tmp_path / "parent", tmp_path / "child"
    for root, content in ((parent, "def event_count(x): return x\n"),
                          (child, "# annotation revision\ndef event_count(x): return x\n")):
        (root / name).parent.mkdir(parents=True)
        (root / name).write_text(content)
    old, current = [sha256_file(root / name) for root in (parent, child)]
    parent_rows = {name: dict(path=name, bytes=(parent / name).stat().st_size, sha256=old)}
    rows = {name: dict(path=name, bytes=(child / name).stat().st_size, sha256=current)}
    if case != "unknown":
        monkeypatch.setitem(analysis.PRIOR_OBSERVATION_COMPATIBILITY, (old, current), "reviewed fixture annotation change")
    if case == "tampered":
        (child / name).write_text("def event_count(x): return 0\n")
    if case == "reviewed":
        result = analysis._prior_observation_compatibility(child, rows, parent, parent_rows, old, current)
        assert result["mode"] == "verified_typing_only_change"
        assert result["parent_sha256"] == old and result["prior_refit_sha256"] == current
    else:
        with pytest.raises(ValueError, match="unreviewed|binding changed"):
            analysis._prior_observation_compatibility(child, rows, parent, parent_rows, old, current)


def _continued_prior_cost_fixture(root, seed=41):
    _, _, parent, _, _, parent_run, parent_checkpoint = _prior_lineage_fixture(root, seed)
    sources = _trainer_sources("population")
    for name in ("population_prior_refit.py", "population_prior_objective.py"):
        sources[name] = sha256_file(Path(__file__).with_name(name))
    parent_run["binding"].update(
        trainer_implementation_sha256=sources["renewal_trajectory_training.py"],
        training_utilities_implementation_sha256=sources["poisson_trajectory_training.py"])
    parent_run["trainer_sources"] = sources
    parent_checkpoint["binding"] = parent_run["binding"]
    for name in ("best.pt", "latest.pt"):
        torch.save(parent_checkpoint, parent / "training" / name)
    parent_run["best_checkpoint_sha256"] = sha256_file(parent / "training/best.pt")
    parent_run["latest_checkpoint_sha256"] = sha256_file(parent / "training/latest.pt")
    _json(parent / "training/run.json", parent_run)
    _json(parent / "training/attempts/attempt_000.json", dict(parent_run, attempt=0, elapsed_seconds=4.))
    proof = json.loads((parent / "delivery.json").read_text())
    for row in proof["files"]:
        path = parent / row["path"]
        row.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    _json(parent / "delivery.json", proof)
    child = root / "prior_budget" / f"seed_{seed}"
    shutil.copytree(parent, child)
    continuation = dict(schema="neural_population_trajectory_budget_continuation_v1",
        base_maximum_epochs=2, total_epochs=4, parent_effective_maximum_epochs=2,
        inherited_early_stopping_patience=12, parent_completed_epochs=2,
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"],
        parent_binding=copy.deepcopy(parent_run["binding"]),
        parent_artifact_sha256={"run.json": sha256_file(parent / "training/run.json"),
                               "best.pt": parent_run["best_checkpoint_sha256"]},
        parent_source_hashes=sources, current_source_hashes=sources, source_migration={})
    run = copy.deepcopy(parent_run)
    run.update(completed_epochs=4, best_epoch=3, effective_maximum_epochs=4,
               continuation=continuation, attempt=3, elapsed_seconds=8.)
    _json(child / "training/run.json", run)
    _json(child / "training/attempts/attempt_002.json", dict(run, attempt=2, elapsed_seconds=7.))
    for name in ("delivery.json", "training/run.json"):
        target = child / "prerequisites/budget_parent" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, target)
    rows = [dict(path=str(path.relative_to(child)), bytes=path.stat().st_size, sha256=sha256_file(path))
            for path in sorted(child.rglob("*")) if path.is_file() and path != child / "delivery.json"]
    proof.update(files=rows, optimization_budget_continuation=dict(parent=str(parent),
        parent_delivery_sha256=sha256_file(parent / "delivery.json"),
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"],
        parent_completed_epochs=2, total_epochs=4))
    _json(child / "delivery.json", proof)
    return child, run


def test_prior_budget_cost_composes_original_refit_and_new_attempts_once(tmp_path):
    child, run = _continued_prior_cost_fixture(tmp_path)
    cost = analysis._training_attempt_cost(child, run)
    assert cost["parent_training_elapsed_seconds"] == 190.
    assert cost["incremental_training_elapsed_seconds"] == 15.
    assert cost["training_elapsed_seconds"] == 205.
    assert cost["parent_training_attempts"] == 5
    assert cost["incremental_training_attempts"] == 2
    assert cost["training_attempts"] == 7
    assert cost["full_pipeline_cost_measured"] is False


@pytest.mark.parametrize("failure", ["source", "migration", "initialization", "parent_receipt"])
def test_prior_budget_cost_rejects_changed_lineage(tmp_path, failure):
    child, run = _continued_prior_cost_fixture(tmp_path)
    if failure == "source":
        run["trainer_sources"] = dict(run["trainer_sources"], **{"population_prior_objective.py": "foreign"})
    elif failure == "migration":
        run["continuation"]["source_migration"] = {"unexpected": "foreign"}
    elif failure == "initialization":
        run["refit_initialization"]["parent_checkpoint_sha256"] = "foreign"
    else:
        (child / "prerequisites/budget_parent/delivery.json").write_text("{}")
    with pytest.raises(ValueError):
        analysis._training_attempt_cost(child, run)


def test_prior_budget_pair_preserves_objective_and_three_seed_lineage(tmp_path):
    items = []
    for seed in analysis.SEEDS:
        _continued_prior_cost_fixture(tmp_path, seed)
    for kind, ceiling in (("prior", 2), ("prior_budget", 4)):
        campaign = tmp_path / kind
        bindings = {}
        for seed in analysis.SEEDS:
            run = json.loads((campaign / f"seed_{seed}/training/run.json").read_text())
            bindings[f"population_{seed}"] = dict(training_config=run["binding"]["config"],
                training_report_sha256=sha256_file(campaign / f"seed_{seed}/training/run.json"),
                executed_training_budget=dict(effective_maximum_epochs=ceiling))
        report = dict(common_training_budget=dict(representation="recurrent", maximum_epochs=ceiling,
            base_maximum_epochs=2, training_objective=analysis.PRIOR_REFIT_OBJECTIVE),
            model_seed_bindings=bindings, inputs=dict(forecasts=str(campaign)),
            source_binding="same", code_sha256="same", configuration="same")
        items.append((report, [], [], [], {}))
    assert analysis._population_budget_pair(list(reversed(items))) == tuple(items)
    child = tmp_path / "prior_budget/seed_42/training/run.json"
    run = json.loads(child.read_text())
    run["continuation"]["source_migration"] = {"unknown": "changed"}
    _json(child, run)
    with pytest.raises(ValueError, match="migration"):
        analysis._population_budget_pair(items)


def _refresh_prior_cell(cell, run, checkpoint):
    """Rebind mutable fixture payloads using the same delivered-file contract."""
    checkpoint["binding"] = run["binding"]
    checkpoint["initialization"] = run["refit_initialization"]
    for name in ("best.pt", "latest.pt"):
        torch.save(checkpoint, cell / "training" / name)
    run["best_checkpoint_sha256"] = sha256_file(cell / "training/best.pt")
    run["latest_checkpoint_sha256"] = sha256_file(cell / "training/latest.pt")
    _json(cell / "training/run.json", run)
    for path in (cell / "training/attempts").glob("attempt_*.json"):
        previous = json.loads(path.read_text())
        # A profile conversion applies to every fixture attempt. Keep each
        # attempt's work counters and elapsed time instead of copying the final run.
        previous.update({key: run[key] for key in
            ("binding", "refit_initialization", "training_objective", "selection_metric", "trainer_sources")})
        for key in ("prediction_particles", "posterior_target_draws", "mixture_scope"):
            if key in run:
                previous[key] = run[key]
            else:
                previous.pop(key, None)
        _json(path, previous)
    evaluation_path = cell / "forecasts/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["checkpoint_sha256"] = run["best_checkpoint_sha256"]
    _json(evaluation_path, evaluation)
    replay_path = cell / "replay.json"
    replay = json.loads(replay_path.read_text())
    replay.update(evaluation_sha256=sha256_file(evaluation_path),
                  source_checkpoint_sha256=run["best_checkpoint_sha256"])
    _json(replay_path, replay)
    proof = json.loads((cell / "delivery.json").read_text())
    for row in proof["files"]:
        path = cell / row["path"]
        row.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    _json(cell / "delivery.json", proof)


def _set_prior_fixture_profile(cell, *, mixture):
    """Use actual checkpointed generator states and the producer's fixed profile."""
    run = json.loads((cell / "training/run.json").read_text())
    checkpoint = torch.load(cell / "training/best.pt", map_location="cpu", weights_only=True)
    config = run["binding"]["config"]
    config["variational_rng"] = "explicit_separate_checkpointed_generator"
    if mixture:
        config.update(schema=analysis.PRIOR_MIXTURE_SCHEMA, observation_objective=analysis.PRIOR_MIXTURE_OBJECTIVE,
                      prior_selection="fixed_draw_validation_joint_predictive_mixture_kl",
                      prediction_particles=8, posterior_target_draws=4)
        run.update(training_objective=analysis.PRIOR_MIXTURE_OBJECTIVE,
                   selection_metric="validation_joint_predictive_mixture_kl",
                   prediction_particles=8, posterior_target_draws=4,
                   mixture_scope=analysis.PRIOR_MIXTURE_SCOPE)
        generator = torch.Generator().manual_seed(run["training_seed"])
        for name in ("variational_rng", "prior_dynamics_rng", "prior_target_rng"):
            checkpoint[name] = generator.get_state()
            torch.rand(1, generator=generator)
    _refresh_prior_cell(cell, run, checkpoint)
    return run, checkpoint


def _rebind_prior_analysis(root, campaign, *, mixture):
    report = json.loads((root / "analysis/analysis.json").read_text())
    bindings, deliveries = {}, {}
    for seed in analysis.SEEDS:
        cell = campaign / f"seed_{seed}"
        run = json.loads((cell / "training/run.json").read_text())
        checkpoint = torch.load(cell / "training/best.pt", map_location="cpu", weights_only=True)
        bindings[f"population_{seed}"] = dict(training_config=run["binding"]["config"],
            training_report_sha256=sha256_file(cell / "training/run.json"),
            checkpoint_sha256=run["best_checkpoint_sha256"],
            executed_training_budget=analysis._training_budget(run, checkpoint, run["best_checkpoint_sha256"]),
            prior_refit_lineage=analysis._prior_parent_evidence(cell, run, checkpoint)[2],
            training_objective=run["training_objective"], selection_metric=run["selection_metric"])
        deliveries[str(seed)] = sha256_file(cell / "delivery.json")
    report["common_training_budget"].update(analysis._prior_config_protocol(run["binding"]["config"]),
        training_objective=run["training_objective"], selection_metric=run["selection_metric"])
    if mixture:
        report["common_training_budget"]["mixture_scope"] = analysis.PRIOR_MIXTURE_SCOPE
        report["code_sha256"]["population_prior_predictive_mixture.py"] = "fixture-mixture-objective"
    report.update(inputs=dict(forecasts=str(campaign)), model_seed_bindings=bindings,
                  campaign_deliveries=deliveries)
    _json(root / "analysis/analysis.json", report)
    _rebind_population_delivery(root)


@pytest.fixture
def delivered_prior_objective_pair(delivered_prior_pair, tmp_path):
    _, analytic = delivered_prior_pair
    report = json.loads((analytic / "analysis/analysis.json").read_text())
    analytic_campaign = Path(report["inputs"]["forecasts"])
    mixture, mixture_campaign = tmp_path / "mixture_analysis", tmp_path / "mixture_campaign"
    shutil.copytree(analytic, mixture)
    shutil.copytree(analytic_campaign, mixture_campaign)
    for campaign, mixed in ((analytic_campaign, False), (mixture_campaign, True)):
        for seed in analysis.SEEDS:
            _set_prior_fixture_profile(campaign / f"seed_{seed}", mixture=mixed)
    _rebind_prior_analysis(analytic, analytic_campaign, mixture=False)
    _rebind_prior_analysis(mixture, mixture_campaign, mixture=True)
    return analytic, mixture


def test_predictive_mixture_native_intake_keeps_parent_frozen_tensors_and_cost(tmp_path):
    _, _, cell, _, _, _, _ = _prior_lineage_fixture(tmp_path)
    run, checkpoint = _set_prior_fixture_profile(cell, mixture=True)
    assert analysis._campaign_cell(cell.parent, 41) == cell
    lineage = analysis._prior_parent_evidence(cell, run, checkpoint)[2]
    assert lineage["parent_completed_epochs"] == 80
    assert analysis._training_attempt_cost(cell, run)["training_elapsed_seconds"] == 190.
    assert analysis._prior_config_protocol(run["binding"]["config"])["prediction_particles"] == 8


@pytest.mark.parametrize("failure", ["unknown_schema", "objective", "selection", "prior_selection",
    "particles", "target_draws", "report_particles", "mixture_scope", "variational_rng", "target_rng"])
def test_predictive_mixture_intake_rejects_mismatched_registered_profile(tmp_path, failure):
    _, _, cell, _, _, _, _ = _prior_lineage_fixture(tmp_path)
    run, checkpoint = _set_prior_fixture_profile(cell, mixture=True)
    config = run["binding"]["config"]
    if failure == "unknown_schema": config["schema"] = "unregistered"
    elif failure == "objective": run["training_objective"] = analysis.PRIOR_REFIT_OBJECTIVE
    elif failure == "selection": run["selection_metric"] = "validation_multistep_prior_kl"
    elif failure == "prior_selection": config["prior_selection"] = "fixed_draw_validation_multistep_kl"
    elif failure == "particles": config["prediction_particles"] = 4
    elif failure == "target_draws": config["posterior_target_draws"] = 8
    elif failure == "report_particles": run["prediction_particles"] = 1
    elif failure == "mixture_scope": run["mixture_scope"] = "marginal_field_mixtures"
    elif failure == "variational_rng": config["variational_rng"] = "global_generator"
    else: checkpoint.pop("prior_target_rng")
    with pytest.raises(ValueError):
        analysis._prior_protocol(run, checkpoint)


def test_two_prior_objectives_real_cli_keeps_different_particle_budget_and_cost(delivered_prior_objective_pair, tmp_path):
    analytic, mixture = delivered_prior_objective_pair
    output = tmp_path / "same_parent_objectives"
    child = subprocess.run([sys.executable, "-m", "population_dynamics.population_forecast_analysis",
        "--prior-objective-comparison", "--population-analysis", str(mixture),
        "--population-analysis", str(analytic), "--output", str(output)], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    report = json.loads((output / "analysis.json").read_text())
    assert report["prior_objective_comparison_kind"] == "same_parent_prior_objectives"
    assert report["models"] == ["population_recurrent_prior_refit", "population_recurrent_predictive_mixture"]
    assert report["paired_forecast_conditions"] == 432 and report["training_compute_matched"] is False
    assert report["scientific_claim_qualified"] is False and report["generation_repeated"] is False
    assert "common_training_budget" not in report
    particle = report["training_particle_budgets"]
    assert particle["analytic_prior_kl"]["prediction_particles"] == 1
    assert particle["analytic_prior_kl"]["posterior_target_draws"] is None
    assert particle["predictive_mixture"]["prediction_particles"] == 8
    assert particle["predictive_mixture"]["posterior_target_draws"] == 4
    rows = _read_rows(output / "prior_objective_training.csv")
    assert len(rows) == 6 and all(float(row["training_elapsed_seconds"]) == 190. for row in rows)
    assert {row["training_objective"] for row in rows} == {
        analysis.PRIOR_REFIT_OBJECTIVE, analysis.PRIOR_MIXTURE_OBJECTIVE}
    assert {int(row["prediction_particles"]) for row in rows} == {1, 8}
    assert len(_read_rows(output / "cost_fidelity_by_origin.csv")) == 288


@pytest.mark.parametrize("failure", ["same_objective", "parent_checkpoint", "parent_report", "frozen_lineage",
                                    "budget", "training_config", "forecast_protocol", "observation_map"])
def test_two_prior_objectives_reject_unmatched_pair(delivered_prior_objective_pair, failure):
    items = [analysis._delivered_population_analysis(root) for root in delivered_prior_objective_pair]
    if failure == "same_objective":
        items[1] = items[0]
    elif failure in ("parent_checkpoint", "parent_report", "frozen_lineage"):
        lineage = items[1][0]["model_seed_bindings"]["population_42"]["prior_refit_lineage"]
        key = {"parent_checkpoint": "parent_checkpoint_sha256",
               "parent_report": "parent_report_sha256", "frozen_lineage": "trainable_parameter_names"}[failure]
        lineage[key] = "foreign"
    elif failure == "budget":
        items[1][0]["common_training_budget"]["maximum_epochs"] += 1
    elif failure == "training_config":
        items[1][0]["model_seed_bindings"]["population_42"]["training_config"]["learning_rate"] = .01
    elif failure == "forecast_protocol":
        items[1][0]["configuration"]["sampling_seed_base"] += 1
    else:
        items[1][0]["source_binding"]["field_bindings"]["fixture"] = "foreign_map"
    with pytest.raises(ValueError):
        analysis._population_prior_pair(items)


def test_predictive_mixture_intake_rejects_changed_frozen_tensor(tmp_path):
    _, _, cell, _, _, _, _ = _prior_lineage_fixture(tmp_path)
    run, checkpoint = _set_prior_fixture_profile(cell, mixture=True)
    checkpoint["model"]["unit_loading"] = checkpoint["model"]["unit_loading"] + 1
    _refresh_prior_cell(cell, run, checkpoint)
    with pytest.raises(ValueError, match="frozen parameter"):
        analysis._prior_parent_evidence(cell, run, checkpoint)


def test_saved_mixture_baseline_comparison_retains_actual_objective(delivered_prior_objective_pair, tmp_path):
    _check_saved_baseline_comparison(delivered_prior_objective_pair[1], tmp_path, "lfads_template_time_controller_mean_prior_budget20")
    rows = _read_rows(tmp_path / "saved_comparison/forecast_paired_block_differences.csv")
    assert {row["candidate"] for row in rows} == {"population_recurrent_predictive_mixture"}
    assert {row["candidate_training_objective"] for row in rows} == {analysis.PRIOR_MIXTURE_OBJECTIVE}

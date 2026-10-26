"""Synthetic replicated receipt fixtures test aggregation, not training claims."""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import lfads_forecast_analysis as analysis
from . import lfads_evaluator_compatibility as compatibility
from . import lfads_forecast_replay as replay
from . import lfads_trajectory_evaluation as evaluator
from .test_lfads_evaluator_compatibility import SOURCE as EVALUATOR_SOURCE
from .test_lfads_trajectory_evaluation import cohort_fixture, call
from .provenance import sha256_file


@pytest.fixture
def replicated(cohort_fixture, monkeypatch):
    f=cohort_fixture;call(f)
    pairs=lambda _:dict(spatial=np.array([[0,0],[1,1]]),cross_population=np.array([[0,1]]))
    monkeypatch.setattr(replay,'fitting_scale',evaluator.fitting_scale)
    monkeypatch.setattr(replay,'pair_sets',pairs)
    monkeypatch.setattr(analysis,'pair_sets',pairs)
    replay.replay(f.acquisition,f.fields,f.training,f.root/'forecasts',f.root/'replay.json','cpu')
    root=f.root/'three_seed_fixture';root.mkdir()
    for seed in (41,42,43):
        folder=root/f'seed_{seed}_forecast';shutil.copytree(f.root/'forecasts',folder)
        with (folder/'scores.csv').open() as stream:rows=list(csv.DictReader(stream))
        for row in rows:row['seed']=seed
        with (folder/'scores.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        ev=json.loads((folder/'evaluation.json').read_text());ev['training_seed']=seed;ev['training_binding']['seed']=seed
        ev['scores_sha256']=sha256_file(folder/'scores.csv');(folder/'evaluation.json').write_text(json.dumps(ev))
        rp=root/f'seed_{seed}_replay.json';record=json.loads((f.root/'replay.json').read_text())
        shutil.copy2(f.root/'replay.csv',rp.with_suffix('.csv'))
        record.update(training_seed=seed,evaluation_sha256=sha256_file(folder/'evaluation.json'),
            scores_sha256=ev['scores_sha256'],csv_artifact=rp.with_suffix('.csv').name,csv_sha256=sha256_file(rp.with_suffix('.csv')))
        rp.write_text(json.dumps(record))
    return root


def test_three_seed_summary_keeps_replication_and_no_inference(replicated,tmp_path):
    output=tmp_path/'analysis';result=analysis.analyze(replicated,output)
    assert result['status']=='DESCRIBED_REPLAYED_DEVELOPMENT'
    assert result['score_rows']==432 and result['forecast_origins']==144
    assert result['independent_blocks']==4 and result['connectivity_instances']==1
    assert result['scientific_claim_qualified'] is False and result['convergence_established'] is False
    with (output/'forecast_summary.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert all(r['training_seeds']=='3' and r['independent_blocks']=='4' for r in rows)
    assert all(r['minimum_seed_mean']==r['maximum_seed_mean'] for r in rows)
    assert not {'p_value','q_value','ci_lower','ci_upper'} & set(rows[0])
    assert all(sha256_file(output/name)==value for name,value in result['artifacts'].items())


def test_explicit_forecasts_preserve_all_horizons_and_baseline_label(replicated, tmp_path):
    inputs = {seed:(replicated/f'seed_{seed}_forecast', replicated/f'seed_{seed}_replay.json')
              for seed in analysis.SEEDS}
    result = analysis.analyze(tmp_path/'unused', tmp_path/'explicit', forecast_inputs=inputs,
                              method='lfads_template_time_controller_mean_prior_budget20')
    with (tmp_path/'explicit/forecast_all_seeds.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 432
    assert {r['method'] for r in rows} == {result['model']}
    assert {float(r['horizon_ms']) for r in rows} == {20., 80., 500.}
    with pytest.raises(ValueError, match='three-seed'):
        analysis.analyze(replicated, tmp_path/'missing', forecast_inputs={41:inputs[41]})


def test_graph_setup_scopes_preserve_restart_and_charge_each_invocation_once():
    rows = {(f'case{i:02}', 1000.):dict(ensemble_artifact=f'case{i:02}.npz',
            graph_setup_seconds=3. if i<9 else 5.) for i in range(16)}
    evaluation = dict(graph_lifetimes=[dict(origin_ms=1000., setup_seconds=3., completed_cases=9),
        dict(origin_ms=1000., setup_seconds=5., completed_cases=7)],
        forecast_continuation=dict(inherited_artifacts={f'case{i:02}.npz':'bound' for i in range(9)}))
    scopes = analysis._graph_cost_scopes(evaluation, rows, 41)
    assert sum(s['graph_setup_charged_seconds'] for s in scopes.values()) == 8.
    assert sum(s['graph_setup_source']=='parent' for s in scopes.values()) == 9
    assert {s['graph_setup_cases'] for s in scopes.values()} == {7,9}
    rows[('case00',1000.)]['graph_setup_seconds']=7.
    with pytest.raises(ValueError, match='invocation membership'):
        analysis._graph_cost_scopes(evaluation, rows, 41)


@pytest.mark.parametrize('change',['missing_replay','wrong_dataset','changed_score'])
def test_unverified_input_is_rejected_without_summary(replicated,tmp_path,change):
    if change=='missing_replay':(replicated/'seed_43_replay.json').unlink()
    elif change=='wrong_dataset':
        p=replicated/'seed_43_replay.json';r=json.loads(p.read_text());r['dataset_sha256']='different';p.write_text(json.dumps(r))
    else:
        p=replicated/'seed_43_forecast/scores.csv';p.write_text(p.read_text().replace('0.1','0.2',1)+'\n')
    output=tmp_path/'analysis'
    with pytest.raises((FileNotFoundError,ValueError)):analysis.analyze(replicated,output)
    assert not output.exists()


# These two-arm receipts are synthetic aggregation fixtures, not new model runs.
def _bind_prior_arm(root, arm):
    folder = root / arm / 'forecasts'
    evaluation_path = folder / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    evaluation['scores_sha256'] = sha256_file(folder / 'scores.csv')
    for name in evaluation['artifacts']:
        evaluation['artifacts'][name] = sha256_file(folder / name)
    evaluation_path.write_text(json.dumps(evaluation))
    replay_path = root / arm / 'replay.json'
    record = json.loads(replay_path.read_text())
    record.update(evaluation_sha256=sha256_file(evaluation_path),
                  scores_sha256=evaluation['scores_sha256'], csv_artifact='replay.csv',
                  csv_sha256=sha256_file(root / arm / 'replay.csv'))
    if 'prior_refit' in evaluation:
        record['prior_refit'] = evaluation['prior_refit']
    replay_path.write_text(json.dumps(record))


def _seal_prior_comparison(root):
    prior = json.loads((root / 'fitted_prior/forecasts/evaluation.json').read_text())['prior_refit']
    record = dict(status='COMPARED_PRIORS_DEVELOPMENT', training_seed=41,
        comparison_arms=[dict(arm=arm, status='VERIFIED', forecast_status='EVALUATED_DEVELOPMENT',
                             replay_status='PASS', forecasts=arm + '/forecasts', replay=arm + '/replay.json')
                         for arm in ('original_prior', 'fitted_prior')],
        comparison_cache=dict(status='VERIFIED', verified_posterior_episodes=56, prior_binding=prior))
    (root / 'execution.json').write_text(json.dumps(record))
    names = ['execution.json']
    for arm in ('original_prior', 'fitted_prior'):
        evaluation = json.loads((root / arm / 'forecasts/evaluation.json').read_text())
        names.extend(arm + '/forecasts/' + name for name in evaluation['artifacts'])
        names.extend([arm + '/forecasts/evaluation.json', arm + '/forecasts/scores.csv',
                      arm + '/replay.json', arm + '/replay.csv'])
    files = [dict(path=name, sha256=sha256_file(root / name), bytes=(root / name).stat().st_size)
             for name in names]
    (root / 'delivery.json').write_text(json.dumps(dict(record, files=files)))


@pytest.fixture
def prior_comparison(replicated, tmp_path):
    root = tmp_path / 'paired_priors'
    for arm in ('original_prior', 'fitted_prior'):
        folder = root / arm / 'forecasts'
        shutil.copytree(replicated / 'seed_41_forecast', folder)
        shutil.copy2(replicated / 'seed_41_replay.json', root / arm / 'replay.json')
        shutil.copy2(replicated / 'seed_41_replay.csv', root / arm / 'replay.csv')
        if arm == 'fitted_prior':
            evaluation_path = folder / 'evaluation.json'
            evaluation = json.loads(evaluation_path.read_text())
            evaluation['prior_refit'] = dict(posterior_cache_sha256='a' * 64,
                prior_parameters_sha256='b' * 64, fitted_parameters=8, fitting_episodes=56,
                training_seed=41, selected_epoch=evaluation['selected_epoch'])
            for graph in evaluation['graph_lifetimes']:
                graph['prior_refit_state_check'] = dict(assigned_scalars=8,
                    nonprior_weights_and_optimizer_state_exact=True)
            evaluation_path.write_text(json.dumps(evaluation))
            with (folder / 'scores.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                delta = (int(row['block']) - 13 + int(row['template_id'].split('_')[-1]) / 10
                         + float(row['origin_ms']) / 10000)
                row['energy_score'] = float(row['energy_score']) + delta
            analysis._write_rows(folder / 'scores.csv', rows)
        _bind_prior_arm(root, arm)
    _seal_prior_comparison(root)
    return root


@pytest.mark.parametrize('decompose', [False, True])
def test_prior_comparison_cli_preserves_paired_blocks_and_unavailable_metrics(prior_comparison, tmp_path, monkeypatch, decompose):
    output = tmp_path / 'prior_analysis'
    monkeypatch.setattr('sys.argv', ['lfads_forecast_analysis', '--prior-comparison',
                                    str(prior_comparison), '--output', str(output)]
                                    + (['--decompose-field-errors'] if decompose else []))
    analysis.main()
    report = json.loads((output / 'analysis.json').read_text())
    assert report['status'] == 'DESCRIBED_REPLAYED_PRIOR_COMPARISON'
    assert report['training_seed'] == 41 and report['training_seeds'] == 1
    assert report['independent_blocks'] == 4 and report['connectivity_instances'] == 1
    assert report['forecast_origins_per_arm'] == 48 and report['matched_score_rows'] == 144
    assert report['episodes_per_arm'] == 16 and report['scientific_claim_qualified'] is False
    assert report['completion_contracts'] == {
        name: sha256_file(prior_comparison / name) for name in ('execution.json', 'delivery.json')}
    with (output / 'forecast_by_block.csv').open() as stream:
        blocks = list(csv.DictReader(stream))
    assert len(blocks) == 24 and all(row['matched_episode_origins'] == '12' for row in blocks)
    with (output / 'forecast_paired_block_differences.csv').open() as stream:
        differences = list(csv.DictReader(stream))
    assert len(differences) == 12 * len(report['metrics'])
    for row in differences:
        if row['metric'] == 'energy_score':
            assert float(row['fitted_minus_original']) == pytest.approx(int(row['block']) - 13 + .45)
    with (output / 'forecast_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 3 * len(report['metrics'])
    assert {row['metric'] for row in summary} == set(report['metrics'])
    for row in summary:
        assert row['training_seeds'] == '1' and row['independent_blocks'] == '4'
        if row['metric'] == 'energy_score':
            assert float(row['fitted_minus_original_mean']) == pytest.approx(2.95)
        if float(row['horizon_ms']) == 20 and row['metric'] == 'variogram_temporal_lag_4':
            assert row['available'] == 'False' and row['fitted_minus_original_mean'] == ''
    assert not {'p_value', 'q_value', 'ci_lower', 'ci_upper'} & set(summary[0])
    assert all(sha256_file(output / name) == value for name, value in report['artifacts'].items())
    if decompose:
        scope = report['field_error_decomposition']
        assert scope['origin_horizon_arm_rows'] == 288 and scope['field_rows'] == 576
        assert scope['new_forecast_origins'] == 0 and not scope['rate_calibration_fitted']
        assert not scope['latent_dynamics_vs_observation_process_identified']
        with (output / 'field_error_summary.csv').open() as stream:
            errors = list(csv.DictReader(stream))
        assert len(errors) == 33
        assert all(float(row['fitted_minus_original_mean']) == 0 for row in errors)
        assert {row['horizon_ms'] for row in errors} == {'20.0', '80.0', '500.0'}


@pytest.mark.parametrize('deviation,bias,temporal,bias_mc,total_mc', [
    ([[3, 3], [3, 3]], 9, 0, 0, 0),
    ([[-1, 1], [-1, 1]], 0, 1, 0, 0),
    ([[-1, -1], [1, 1]], 0, 0, 1, 1),
    ([[-1, 1], [1, -1]], 0, 0, 0, 1),
])
def test_field_error_partition_preserves_temporal_covariance_and_signed_correction(deviation, bias, temporal, bias_mc, total_mc):
    truth = np.full((2, 1), 3)
    samples = truth + np.asarray(deviation)[:, :, None]
    result = analysis.field_mean_error_components(samples, truth, np.array([2]), .5)
    assert result['time_mean_bias_mse_hz2'][0] == pytest.approx(bias)
    assert result['temporal_residual_mse_hz2'][0] == pytest.approx(temporal)
    assert result['time_mean_bias_mc_mse_hz2'][0] == pytest.approx(bias_mc)
    assert result['ensemble_mean_mc_mse_hz2'][0] == pytest.approx(total_mc)
    assert result['mean_mse_corrected_hz2'][0] == pytest.approx(bias + temporal - total_mc)
    assert result['time_mean_bias_mse_corrected_hz2'][0] == pytest.approx(bias - bias_mc)
    assert result['temporal_residual_mse_corrected_hz2'][0] == pytest.approx(temporal - total_mc + bias_mc)


@pytest.mark.parametrize('change', ['draws', 'shape', 'nonfinite', 'negative', 'units', 'clock'])
def test_field_error_partition_rejects_invalid_observations(change):
    sample, truth, units, seconds = np.ones((2, 3, 1)), np.ones((3, 1)), np.array([2]), .005
    if change == 'draws': sample = sample[:1]
    elif change == 'shape': truth = truth[:2]
    elif change == 'nonfinite': sample[0, 0, 0] = np.nan
    elif change == 'negative': truth[0, 0] = -1
    elif change == 'units': units[0] = 0
    else: seconds = 0
    with pytest.raises(ValueError):
        analysis.field_mean_error_components(sample, truth, units, seconds)


@pytest.mark.parametrize('shift,noise,cross', [(0., 0., 0.), (1., 1., 2.), (-1., 1., -2.)])
def test_observation_noise_partition_and_poisson_ensemble_variance(shift, noise, cross):
    truth = np.ones((2, 1))
    conditional = np.full((2, 2, 1), 2.)
    samples = np.array([[[1.], [1.]], [[3.], [3.]]]) + shift
    result = analysis.observation_noise_components(samples, conditional, truth, np.array([2]), .5)
    assert result['observation_noise_mean_mse_hz2'][0] == pytest.approx(noise)
    assert result['observation_error_cross_term_hz2'][0] == pytest.approx(cross)
    assert result['expected_poisson_mean_mse_hz2'][0] == pytest.approx(1.)
    assert (1 + shift) ** 2 == pytest.approx(1 + noise + cross)


def _split_prior_graphs(root, completed):
    path = root / 'fitted_prior/forecasts/evaluation.json'
    evaluation = json.loads(path.read_text())
    evaluation.setdefault('evaluator_implementation_sha256', 'e' * 64)
    original = evaluation['graph_lifetimes'][-1]
    segments = [dict(original, completed_cases=count, setup_seconds=500. + index)
                for index, count in enumerate(completed)]
    evaluation['graph_lifetimes'][-1:] = segments
    inherited = 32 + completed[0]
    evaluation['forecast_continuation'] = dict(inherited_origins=inherited,
        inherited_score_rows=inherited * len(evaluation['configuration']['horizons_ms']),
        numerical_program_without_continuation_equal=True, full_48_origin_replay_required=True,
        current_evaluator_sha256=evaluation['evaluator_implementation_sha256'])
    path.write_text(json.dumps(evaluation))
    _bind_prior_arm(root, 'fitted_prior')
    _seal_prior_comparison(root)
    return path


@pytest.mark.parametrize('completed', [(0, 16), (9, 7)])
def test_prior_comparison_reuses_split_graphs_without_changing_tables_or_costs(prior_comparison, tmp_path, completed):
    expected = analysis.compare_priors(prior_comparison, tmp_path / 'single_attempt')
    path = _split_prior_graphs(prior_comparison, completed)
    before = path.read_bytes()
    actual = analysis.compare_priors(prior_comparison, tmp_path / 'continued')
    _assert_same_tables(tmp_path / 'single_attempt', tmp_path / 'continued', expected)
    assert path.read_bytes() == before
    graphs = json.loads(before)['graph_lifetimes']
    assert len(graphs) == 4 and [g['completed_cases'] for g in graphs[-2:]] == list(completed)
    assert [g['setup_seconds'] for g in graphs[-2:]] == [500., 501.]
    assert actual['forecast_origins_per_arm'] == 48 and actual['independent_blocks'] == 4


@pytest.mark.parametrize('change', ['short_total', 'duplicate_cases', 'negative', 'noninteger',
    'unknown_origin', 'reordered_origins', 'zero_case_state_changed', 'assigned_scalars',
    'missing_continuation', 'unverified_program', 'wrong_inherited_rows', 'changed_evaluator'])
def test_prior_comparison_rejects_invalid_split_graph_evidence(prior_comparison, tmp_path, change):
    path = _split_prior_graphs(prior_comparison, (0, 16))
    evaluation = json.loads(path.read_text())
    graphs = evaluation['graph_lifetimes']
    if change == 'short_total': graphs[-1]['completed_cases'] = 15
    elif change == 'duplicate_cases': graphs[-2]['completed_cases'] = 1
    elif change == 'negative': graphs[-2]['completed_cases'] = -1
    elif change == 'noninteger': graphs[-2]['completed_cases'] = 0.0
    elif change == 'unknown_origin': graphs[-2]['origin_ms'] = 4000.
    elif change == 'reordered_origins': graphs[0], graphs[1] = graphs[1], graphs[0]
    elif change == 'zero_case_state_changed': graphs[-2]['prior_refit_state_check']['nonprior_weights_and_optimizer_state_exact'] = False
    elif change == 'assigned_scalars': graphs[-2]['prior_refit_state_check']['assigned_scalars'] = 7
    elif change == 'missing_continuation': evaluation.pop('forecast_continuation')
    elif change == 'unverified_program': evaluation['forecast_continuation']['numerical_program_without_continuation_equal'] = False
    elif change == 'wrong_inherited_rows': evaluation['forecast_continuation']['inherited_score_rows'] -= 1
    else: evaluation['forecast_continuation']['current_evaluator_sha256'] = '0' * 64
    path.write_text(json.dumps(evaluation))
    _bind_prior_arm(prior_comparison, 'fitted_prior')
    _seal_prior_comparison(prior_comparison)
    output = tmp_path / 'invalid_continuation'
    with pytest.raises(ValueError): analysis.compare_priors(prior_comparison, output)
    assert not output.exists()


@pytest.mark.parametrize('change', ['missing_pair', 'wrong_template', 'checkpoint', 'scale',
    'missing_value', 'nonfinite', 'prior_identity', 'nonprior_state', 'incomplete_replay', 'incomplete_contract', 'delivery_binding'])
def test_prior_comparison_rejects_unmatched_or_unqualified_inputs(prior_comparison, tmp_path, change):
    root, arm = prior_comparison, 'fitted_prior'
    folder = root / arm / 'forecasts'
    evaluation_path, replay_path = folder / 'evaluation.json', root / arm / 'replay.json'
    if change in ('missing_pair', 'wrong_template', 'missing_value', 'nonfinite'):
        with (folder / 'scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if change == 'missing_pair':
            rows.pop()
        elif change == 'wrong_template':
            rows[0]['template_id'] = 'familiar_9'
        else:
            rows[0]['energy_score'] = '' if change == 'missing_value' else 'nan'
        analysis._write_rows(folder / 'scores.csv', rows)
    elif change == 'scale':
        path = folder / 'fitting_field_scale.npz'
        with np.load(path, allow_pickle=False) as arrays:
            values = {name: arrays[name].copy() for name in arrays.files}
        values['mean_hz'] += 1
        np.savez(path, **values)
    elif change in ('checkpoint', 'prior_identity', 'nonprior_state'):
        evaluation = json.loads(evaluation_path.read_text())
        if change == 'checkpoint':
            evaluation['selected_training_sha256'] = 'c' * 64
        elif change == 'prior_identity':
            evaluation['prior_refit']['fitted_parameters'] = 7
        else:
            evaluation['graph_lifetimes'][0]['prior_refit_state_check']['nonprior_weights_and_optimizer_state_exact'] = False
        evaluation_path.write_text(json.dumps(evaluation))
    _bind_prior_arm(root, arm)
    if change == 'incomplete_replay':
        record = json.loads(replay_path.read_text())
        record['status'] = 'RUNNING'
        replay_path.write_text(json.dumps(record))
    _seal_prior_comparison(root)
    if change == 'incomplete_contract':
        path = root / 'execution.json'
        record = json.loads(path.read_text())
        record['status'] = 'COMPARING_PRIORS_DEVELOPMENT'
        path.write_text(json.dumps(record))
    if change == 'delivery_binding':
        path = root / 'delivery.json'
        record = json.loads(path.read_text())
        next(row for row in record['files'] if row['path'] == 'fitted_prior/forecasts/scores.csv')['sha256'] = 'd' * 64
        path.write_text(json.dumps(record))
    output = tmp_path / 'rejected_prior_analysis'
    with pytest.raises(ValueError):
        analysis.compare_priors(root, output)
    assert not output.exists()


def _seal_delivery(root, record, names, declared=()):
    files = [dict(path=name, sha256=sha256_file(root / name), bytes=(root / name).stat().st_size)
             for name in names]
    (root / 'delivery.json').write_text(json.dumps(dict(record, files=files + list(declared))))


def _original_archive(paired, archive, *, rescore=False, variant='original_20'):
    folder = archive / 'forecasts' / variant / 'seed_41_forecast'
    shutil.copytree(paired / 'original_prior/forecasts', folder)
    replay_path = folder.with_name('seed_41_replay.json')
    shutil.copy2(paired / 'original_prior/replay.csv', replay_path.with_suffix('.csv'))
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    record = json.loads((paired / 'original_prior/replay.json').read_text())
    if rescore:
        proof = dict(schema='lfads_exact_count_correction_rescore_v1', unchanged_estimand=True,
            unchanged_output_dtype=evaluation['configuration']['unit_metric_dtype'],
            unchanged_metric_tolerance=record['metric_tolerance'],
            verified_legacy_evaluator=dict(numerical_program_ast_equal=True))
        evaluation['numerical_rescore'] = record['numerical_rescore'] = proof
        (folder / 'evaluation.json').write_text(json.dumps(evaluation))
    record.update(csv_artifact=replay_path.with_suffix('.csv').name,
        csv_sha256=sha256_file(replay_path.with_suffix('.csv')),
        evaluation_sha256=sha256_file(folder / 'evaluation.json'))
    replay_path.write_text(json.dumps(record))
    names = [str(path.relative_to(archive)) for path in
             [*(folder / name for name in evaluation['artifacts']),
              folder / 'evaluation.json', folder / 'scores.csv', replay_path, replay_path.with_suffix('.csv')]]
    # This is the actual canonical archive schema: no top-level exit_code.
    _seal_delivery(archive, dict(status='ARCHIVED_REPLAYED_BUDGET_COMPARISON'), names)
    return folder, replay_path


def _assert_same_tables(left, right, report):
    assert report['artifacts']
    for name in report['artifacts']:
        assert (left / name).read_bytes() == (right / name).read_bytes(), name


@pytest.mark.parametrize('decompose', [False, True])
def test_external_original_route_preserves_all_numerical_tables(prior_comparison, tmp_path, decompose):
    folder, replay_path = _original_archive(prior_comparison, tmp_path / 'original_archive', rescore=True)
    original_files = [path for path in (tmp_path / 'original_archive').rglob('*') if path.is_file()]
    before = {path: sha256_file(path) for path in original_files}
    legacy, reused = tmp_path / 'legacy', tmp_path / 'reused'
    expected = analysis.compare_priors(prior_comparison, legacy, decompose_field_errors=decompose)
    actual = analysis.compare_priors(prior_comparison, reused, decompose_field_errors=decompose,
                                    original_forecast=folder, original_replay=replay_path)
    _assert_same_tables(legacy, reused, expected)
    assert actual['original_prior_reused'] is True and actual['new_forecast_origins'] == 0
    assert actual['completion_contracts']['original']['numerical_rescore']['unchanged_estimand'] is True
    assert {path: sha256_file(path) for path in original_files} == before


def _bind_archived_original(folder, replay_path, evaluation, record):
    (folder / 'evaluation.json').write_text(json.dumps(evaluation))
    record['evaluation_sha256'] = sha256_file(folder / 'evaluation.json')
    replay_path.write_text(json.dumps(record))
    archive = folder.parents[2]
    delivery = json.loads((archive / 'delivery.json').read_text())
    names = [row['path'] for row in delivery['files']]
    source_name = str(analysis.ARCHIVED_EVALUATOR)
    if source_name not in names:
        names.append(source_name)
    _seal_delivery(archive, delivery, names)


@pytest.fixture
def archived_original(prior_comparison, tmp_path, monkeypatch):
    folder, replay_path = _original_archive(prior_comparison, tmp_path / 'budget_archive',
        rescore=True, variant='continued_80_ceiling')
    source = folder.parents[2] / analysis.ARCHIVED_EVALUATOR
    source.parent.mkdir(parents=True)
    source.write_text(EVALUATOR_SOURCE)
    current = tmp_path / 'current_evaluator.py'
    current.write_text(EVALUATOR_SOURCE.replace("model_type='lfads')",
        "model_type='lfads', effective_training_epoch_budget=80, training_continuation=None)"))
    monkeypatch.setattr(compatibility, 'PRE_REFIT_EVALUATOR_SHA256', sha256_file(source))
    verifier = compatibility.verify_legacy_evaluator

    def verify(recorded, archived_path, current_path):
        assert archived_path == source
        assert current_path == Path(analysis.__file__).with_name('lfads_trajectory_evaluation.py')
        return verifier(recorded, archived_path, current)

    monkeypatch.setattr(analysis, 'verify_legacy_evaluator', verify)
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    record = json.loads(replay_path.read_text())
    for document in (evaluation, record):
        document['numerical_rescore'].pop('verified_legacy_evaluator')
    evaluation['evaluator_implementation_sha256'] = sha256_file(source)
    evaluation['training_continuation'] = dict(treatment='optimization_budget_only')
    _bind_archived_original(folder, replay_path, evaluation, record)
    return dict(folder=folder, replay=replay_path, source=source, current=current,
                evaluation=evaluation, record=record)


@pytest.mark.parametrize('decompose', [False, True])
def test_archived_evaluator_proof_preserves_tables_and_historical_rescore(
        archived_original, prior_comparison, tmp_path, decompose):
    f = archived_original
    archive = f['folder'].parents[2]
    before = {p: sha256_file(p) for p in archive.rglob('*') if p.is_file()}
    legacy, reused = tmp_path / 'legacy', tmp_path / 'reused'
    expected = analysis.compare_priors(prior_comparison, legacy, decompose_field_errors=decompose)
    result = analysis.compare_priors(prior_comparison, reused, decompose_field_errors=decompose,
        original_forecast=f['folder'], original_replay=f['replay'])
    _assert_same_tables(legacy, reused, expected)
    completion = result['completion_contracts']['original']
    proof = completion['verified_archived_evaluator']
    assert proof['numerical_program_ast_equal'] is True
    assert proof['original_evaluator_sha256'] == sha256_file(f['source'])
    assert proof['execution_evaluator_sha256'] == sha256_file(f['current'])
    assert 'verified_legacy_evaluator' not in completion['numerical_rescore']
    assert completion['numerical_rescore'] == f['evaluation']['numerical_rescore']
    assert {p: sha256_file(p) for p in before} == before
    accepted = analysis._forecast_files(f['folder'], f['replay'], f['evaluation'], f['record'])
    analysis._original_completion(f['folder'], f['replay'], f['evaluation'], f['record'], accepted)
    assert accepted[f['source']] == sha256_file(f['source'])


@pytest.mark.parametrize('change', ['missing_source', 'tampered_source', 'undeclared_source',
    'delivery_hash', 'delivery_bytes', 'evaluation_hash', 'numerical_program',
    'proof_null', 'proof_false', 'proof_empty', 'dtype', 'tolerance', 'estimand',
    'prior_refit', 'conditional_prior', 'forecast_continuation'])
def test_archived_evaluator_fallback_rejects_invalid_evidence(archived_original, tmp_path, change):
    f = archived_original
    evaluation, record = f['evaluation'], f['record']
    if change == 'evaluation_hash':
        evaluation['evaluator_implementation_sha256'] = '0' * 64
    elif change.startswith('proof_'):
        proof = {'proof_null': None, 'proof_false': dict(numerical_program_ast_equal=False), 'proof_empty': {}}[change]
        for document in (evaluation, record):
            document['numerical_rescore']['verified_legacy_evaluator'] = proof
    elif change in ('dtype', 'tolerance', 'estimand'):
        key, value = {'dtype': ('unchanged_output_dtype', 'float64'),
            'tolerance': ('unchanged_metric_tolerance', dict(rtol=1, atol=1)),
            'estimand': ('unchanged_estimand', False)}[change]
        for document in (evaluation, record):
            document['numerical_rescore'][key] = value
    elif change in ('prior_refit', 'conditional_prior', 'forecast_continuation'):
        evaluation[change] = dict(enabled=True)
    _bind_archived_original(f['folder'], f['replay'], evaluation, record)
    if change == 'missing_source':
        f['source'].unlink()
    elif change == 'tampered_source':
        f['source'].write_text(EVALUATOR_SOURCE.replace('x + 1', 'x + 2'))
    elif change == 'numerical_program':
        f['current'].write_text(EVALUATOR_SOURCE.replace('x + 1', 'x + 2'))
    elif change in ('undeclared_source', 'delivery_hash', 'delivery_bytes'):
        path = f['folder'].parents[2] / 'delivery.json'
        delivery = json.loads(path.read_text())
        entry = next(r for r in delivery['files'] if r['path'] == str(analysis.ARCHIVED_EVALUATOR))
        if change == 'undeclared_source': delivery['files'].remove(entry)
        elif change == 'delivery_hash': entry['sha256'] = '0' * 64
        else: entry['bytes'] += 1
        path.write_text(json.dumps(delivery))
    accepted = analysis._forecast_files(f['folder'], f['replay'], evaluation, record)
    with pytest.raises(ValueError):
        analysis._original_completion(f['folder'], f['replay'], evaluation, record, accepted)
    assert f['source'] not in accepted


@pytest.fixture
def stationary_comparison(prior_comparison, tmp_path):
    # Enrich the existing small forecast fixture with the source-native binding
    # fields consumed by stationary intake. No posterior payload is regenerated.
    cases = [f'train_block_{block:03d}_familiar_{template}' for block in range(14) for template in range(4)]
    for arm in analysis.PRIOR_ARMS:
        path = prior_comparison / arm / 'forecasts/evaluation.json'
        evaluation = json.loads(path.read_text())
        evaluation['training_binding'].update(episode_table_sha256='c' * 64,
            episode_bindings={case: dict(raw_report_sha256='d' * 64,
                                       field_summary_sha256='e' * 64) for case in cases})
        path.write_text(json.dumps(evaluation))
        _bind_prior_arm(prior_comparison, arm)
    evaluation_path = prior_comparison / 'fitted_prior/forecasts/evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    proof = dict(frozen_global_variables_verified=152, training_updates_before_and_after=1120)
    parent = tmp_path / 'completed_fit'
    (parent / 'cache').mkdir(parents=True)
    np.savez(parent / 'cache/prior_parameters.npz', logatau=np.ones((1, 4)), logevars=np.zeros((1, 4)))
    parameters_digest = sha256_file(parent / 'cache/prior_parameters.npz')
    prior_epoch = evaluation['selected_epoch'] + 1
    cohort_keys = ('acquisition_design_sha256', 'acquisition_ledger_sha256', 'dataset_sha256',
                   'episode_table_sha256', 'native_source_identity', 'episode_bindings')
    binding = dict(training_seed=41, training_binding=evaluation['training_binding'],
        training_run_sha256=evaluation['training_run_sha256'],
        selected_model_artifacts=evaluation['selected_model_artifacts'], selected_epoch=prior_epoch,
        inference_migration=evaluation['inference_migration'],
        cohort_binding={key: evaluation['training_binding'][key] for key in cohort_keys},
        case_ids=cases, fitting_blocks=list(range(14)), excluded_selection_blocks=list(range(14, 18)),
        joint_samples=16, complete_bins=1200, selected_training_updates=1120)
    episodes = [dict(case_id=case, path=case + '.npz', training_seed=41, sha256='f' * 64, **proof)
                for case in cases]
    ledger = dict(status='FITTED_FROZEN_NETWORK_AR_PRIOR', fitting_cohort_extracted=True,
        model_parameters_updated=False, formal_test_accessed=False, complete_episodes=56,
        posterior_paths=896, prior_parameters_sha256=parameters_digest, binding=binding,
        episodes=episodes, **proof)
    (parent / 'cache/posterior_cache.json').write_text(json.dumps(ledger))
    fit = dict(status='FITTED_PRIOR_ONLY', action='fit', training_seed=41, network_variant='budget_20',
        staged_inputs=dict(status='VERIFIED'),
        commands=[dict(label='posterior_fit', status='PASS', exit_code=0)])
    declared = [dict(path='cache/' + row['path'], sha256=row['sha256'], bytes=100) for row in episodes]
    _seal_delivery(parent, fit, ['cache/posterior_cache.json', 'cache/prior_parameters.npz'], declared)
    evaluation['prior_refit'].update(posterior_cache_sha256=sha256_file(parent / 'cache/posterior_cache.json'),
        prior_parameters_sha256=parameters_digest, selected_epoch=prior_epoch)
    evaluation_path.write_text(json.dumps(evaluation))
    _bind_prior_arm(prior_comparison, 'fitted_prior')
    _seal_prior_comparison(prior_comparison)
    folder, replay_path = _original_archive(prior_comparison, tmp_path / 'original_archive')
    candidate = tmp_path / 'stationary_candidate'
    shutil.copytree(prior_comparison / 'fitted_prior', candidate / 'candidate')
    record = dict(status='REPLAYED_STATIONARY_PRIOR_CANDIDATE', exit_code=0,
        training_seed=41, network_budget_epochs=20, prior_variant='stationary', completed_origins=48,
        original_reference_reexecuted=False, neural_training_updates=0,
        scientific_claim_qualified=False, formal_test_accessed=False, mean_run=None,
        reference_run=str(parent), reference_delivery_sha256=sha256_file(parent / 'delivery.json'),
        stationary_prior_intake=dict(status='VERIFIED', complete_episodes=56, **proof))
    (candidate / 'execution.json').write_text(json.dumps(record))
    names = ['execution.json', 'candidate/replay.json', 'candidate/replay.csv',
             'candidate/forecasts/evaluation.json', 'candidate/forecasts/scores.csv',
             *('candidate/forecasts/' + name for name in evaluation['artifacts'])]
    _seal_delivery(candidate, record, names)
    return dict(paired=prior_comparison, candidate=candidate, parent=parent,
                original_forecast=folder, original_replay=replay_path)


@pytest.mark.parametrize('relocate', [False, True])
def test_fitted_only_cli_reuses_original_and_three_parent_proof_files(stationary_comparison, tmp_path, monkeypatch, relocate):
    fixture = stationary_comparison
    legacy, reused = tmp_path / 'legacy', tmp_path / 'reused'
    expected = analysis.compare_priors(fixture['paired'], legacy, decompose_field_errors=True)
    parent = fixture['parent']
    extra = []
    if relocate:
        relocated = tmp_path / 'relocated_fit'
        shutil.move(parent, relocated)
        parent = relocated
        extra = ['--fit-run', str(parent)]
    assert sorted(str(path.relative_to(parent)) for path in parent.rglob('*') if path.is_file()) == [
        'cache/posterior_cache.json', 'cache/prior_parameters.npz', 'delivery.json']
    monkeypatch.setattr('sys.argv', ['lfads_forecast_analysis', '--prior-comparison', str(fixture['candidate']),
        '--original-forecast', str(fixture['original_forecast']), '--original-replay', str(fixture['original_replay']),
        '--decompose-field-errors', '--output', str(reused), *extra])
    analysis.main()
    actual = json.loads((reused / 'analysis.json').read_text())
    _assert_same_tables(legacy, reused, expected)
    assert actual['completion_contracts']['fitted']['completed_native_intake_reused'] is True
    assert actual['completion_contracts']['fitted']['fit_run'] == str(parent.resolve())
    assert actual['original_prior_reused'] is True and actual['new_forecast_origins'] == 0


@pytest.mark.parametrize('change', ['candidate_running', 'incomplete_intake', 'parent_delivery',
    'parent_running', 'cache_seed', 'cache_cohort', 'cache_checkpoint', 'cache_migration',
    'cache_proof', 'episode_proof', 'missing_episode_hash', 'replay_running', 'rescore_mismatch'])
def test_reused_comparison_rejects_incomplete_or_mismatched_proof(stationary_comparison, tmp_path, change):
    fixture = stationary_comparison
    candidate, parent = fixture['candidate'], fixture['parent']
    delivery = json.loads((candidate / 'delivery.json').read_text())
    if change in ('candidate_running', 'incomplete_intake', 'parent_delivery'):
        if change == 'candidate_running': delivery['status'] = 'RUNNING'
        elif change == 'incomplete_intake': delivery['stationary_prior_intake']['complete_episodes'] = 55
        else: delivery['reference_delivery_sha256'] = '0' * 64
        (candidate / 'delivery.json').write_text(json.dumps(delivery))
    elif change in ('replay_running', 'rescore_mismatch'):
        path = fixture['original_replay']
        record = json.loads(path.read_text())
        if change == 'replay_running': record['status'] = 'RUNNING'
        else: record['numerical_rescore'] = dict(unchanged_estimand=False)
        path.write_text(json.dumps(record))
        archive = fixture['original_forecast'].parents[2]
        archived = json.loads((archive / 'delivery.json').read_text())
        _seal_delivery(archive, archived, [row['path'] for row in archived['files']])
    else:
        fit = json.loads((parent / 'delivery.json').read_text())
        ledger_path = parent / 'cache/posterior_cache.json'
        ledger = json.loads(ledger_path.read_text())
        if change == 'parent_running': fit['status'] = 'FITTING_PRIOR'
        elif change == 'cache_seed': ledger['binding']['training_seed'] = 42
        elif change == 'cache_cohort': ledger['binding']['cohort_binding']['dataset_sha256'] = '0' * 64
        elif change == 'cache_checkpoint': ledger['binding']['selected_model_artifacts'] = {}
        elif change == 'cache_migration': ledger['binding']['inference_migration'] = dict(changed=True)
        elif change == 'cache_proof': ledger['frozen_global_variables_verified'] = 151
        elif change == 'episode_proof': ledger['episodes'][0]['training_updates_before_and_after'] = 1121
        else:
            ledger['episodes'][0].pop('sha256')
            next(row for row in fit['files'] if row['path'] == 'cache/' + ledger['episodes'][0]['path']).pop('sha256')
        ledger_path.write_text(json.dumps(ledger))
        declared = [row for row in fit['files'] if row['path'].startswith('cache/train_block_')]
        _seal_delivery(parent, fit, ['cache/posterior_cache.json', 'cache/prior_parameters.npz'], declared)
        evaluation_path = candidate / 'candidate/forecasts/evaluation.json'
        evaluation = json.loads(evaluation_path.read_text())
        evaluation['prior_refit']['posterior_cache_sha256'] = sha256_file(ledger_path)
        evaluation_path.write_text(json.dumps(evaluation))
        _bind_prior_arm(candidate, 'candidate')
        delivery['reference_delivery_sha256'] = sha256_file(parent / 'delivery.json')
        (candidate / 'execution.json').write_text(json.dumps({key: value for key, value in delivery.items() if key != 'files'}))
        _seal_delivery(candidate, delivery, [row['path'] for row in delivery['files']])
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        analysis.compare_priors(candidate, output, original_forecast=fixture['original_forecast'],
                                original_replay=fixture['original_replay'])
    assert not output.exists()


@pytest.mark.parametrize('arguments', [
    ['--prior-comparison', 'candidate', '--original-forecast', 'original'],
    ['--prior-comparison', 'candidate', '--original-replay', 'replay.json'],
    ['--forecasts', 'forecasts', '--original-forecast', 'original', '--original-replay', 'replay.json'],
    ['--forecasts', 'forecasts', '--fit-run', 'parent'],
    ['--prior-matrix', 'chain.json'],
    ['--forecasts', 'forecasts', '--reference-analysis', 'reference'],
    ['--forecasts', 'forecasts', '--matrix-inputs', 'inputs'],
])
def test_reuse_cli_requires_paired_explicit_inputs(arguments, tmp_path, monkeypatch):
    output = tmp_path / 'rejected'
    monkeypatch.setattr('sys.argv', ['lfads_forecast_analysis', *arguments, '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        analysis.main()
    assert error.value.code == 2 and not output.exists()


@pytest.fixture
def prior_matrix(tmp_path):
    """Tiny completed CSV archives; no neural or posterior arrays are needed."""
    inputs = tmp_path / 'matrix_archives'
    config = dict(horizons_ms=[20., 80., 500.], origins_ms=[1000., 3000., 5000.],
                  temporal_lags_bins=[4], refractory_support_windows_bins=[])
    metrics = analysis._forecast_metrics(config)
    common_binding = dict(config=dict(maximum_epochs=20, model=dict(ic_dim=64)),
        acquisition_design_sha256='a' * 64, acquisition_ledger_sha256='b' * 64,
        dataset_sha256='c' * 64, episode_table_sha256='d' * 64,
        native_source_identity=dict(solver='tiny-fixture'), episode_bindings=dict(case='immutable'),
        source_revision='fixture', backend='cpu')
    cells, archives = [], {}
    for budget in (20, 80):
        for seed in analysis.SEEDS:
            cell = f'budget_{budget}/seed_{seed}'
            archive = inputs / f'analysis_budget_{budget}_seed_{seed}'
            source = inputs / f'source_budget_{budget}_seed_{seed}'
            (archive / 'analysis').mkdir(parents=True)
            source.mkdir()
            source_record = dict(training_seed=seed, selected_epoch=3)
            if (budget, seed) == (20, 41):
                source_record.update(status='COMPARED_PRIORS_DEVELOPMENT', action='compare', network_variant='budget_20')
            else:
                source_record.update(status='REPLAYED_STATIONARY_PRIOR_CANDIDATE', exit_code=0,
                                     network_budget_epochs=budget)
            (source / 'delivery.json').write_text(json.dumps(source_record))
            source_digest = sha256_file(source / 'delivery.json')
            rows = []
            for block in range(14, 18):
                for horizon in config['horizons_ms']:
                    for metric in metrics:
                        left = 100. + block - 14
                        right = left + budget / 20 + 2 * (block - 14) + 10 * (seed - 42)
                        available = not (metric == 'variogram_temporal_lag_4' and horizon == 20.)
                        rows.append(dict(seed=seed, block=block, horizon_ms=horizon, metric=metric,
                            original_prior=left if available else None, fitted_prior=right if available else None,
                            fitted_minus_original=right - left if available else None, available=available,
                            matched_episode_origins=12, training_seeds=1))
            table = archive / 'analysis/forecast_paired_block_differences.csv'
            analysis._write_rows(table, rows)
            fitted = ({'delivery.json': source_digest} if (budget, seed) == (20, 41)
                      else {'delivery_sha256': source_digest})
            report = dict(status='DESCRIBED_REPLAYED_PRIOR_COMPARISON', model='lfads', training_seed=seed,
                training_seeds=1, independent_blocks=4, connectivity_instances=1, episodes_per_arm=16,
                forecast_origins_per_arm=48, score_rows_per_arm=144, matched_score_rows=144,
                configuration=config, metrics=metrics, training_binding=dict(common_binding, seed=seed,
                    sources={'lfads_trajectory_training.py': str(budget)}),
                completion_contracts=dict(original=dict(delivery_sha256='e' * 64), fitted=fitted),
                selected_epoch=3, all_unit_metrics_independently_replayed=True, scientific_claim_qualified=False,
                convergence_established=False, formal_test_accessed=False, calibration_outcomes_accessed=False,
                original_prior_reused=True, new_forecast_origins=0, difference='fitted_prior minus original_prior',
                artifacts={table.name: sha256_file(table)})
            (archive / 'analysis/analysis.json').write_text(json.dumps(report))
            delivery = dict(status=report['status'], exit_code=0, analysis_exit_code=0, training_seed=seed,
                            source_comparison=str(source), source_delivery_sha256=source_digest)
            _seal_delivery(archive, delivery, ['analysis/analysis.json', 'analysis/' + table.name])
            archives[cell] = archive
            if (budget, seed) != (20, 41):
                cells.append(dict(cell=cell, forecast_archive=str(source), analysis_archive=str(archive)))
    chain = tmp_path / 'completion_chain.json'
    chain.write_text(json.dumps(dict(cells=cells)))
    return dict(chain=chain, reference=archives['budget_20/seed_41'], archives=archives, inputs=inputs)


def test_prior_matrix_uses_four_shared_blocks_and_preserves_seed_range(prior_matrix, tmp_path):
    output = tmp_path / 'matrix'
    report = analysis.aggregate_prior_matrix(prior_matrix['chain'], prior_matrix['reference'], output)
    assert report['status'] == 'DESCRIBED_REPLAYED_PRIOR_MATRIX'
    assert report['completed_cells'] == 6 and report['independent_blocks'] == 4 and report['training_seeds'] == 3
    assert report['source_provenance_changes_between_budgets'] is True
    assert report['training_bindings_by_budget']['80']['sources'] != report['training_bindings_by_budget']['20']['sources']
    assert report['new_forecast_origins'] == 0 and report['scientific_claim_qualified'] is False
    with (output / 'forecast_matrix_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    with (output / 'forecast_seed_averaged_blocks.csv').open() as stream:
        blocks = list(csv.DictReader(stream))
    with (output / 'forecast_all_seed_blocks.csv').open() as stream:
        raw = list(csv.DictReader(stream))
    for budget, effect in ((20, 4.), (80, 7.)):
        row = next(row for row in summary if row['network_budget_epochs'] == str(budget)
                   and row['metric'] == 'energy_score' and float(row['horizon_ms']) == 20.)
        assert float(row['fitted_minus_original_mean']) == pytest.approx(effect)
        assert float(row['minimum_seed_mean_difference']) == pytest.approx(effect - 10)
        assert float(row['maximum_seed_mean_difference']) == pytest.approx(effect + 10)
        selected = [row for row in blocks if row['network_budget_epochs'] == str(budget)
                    and row['metric'] == 'energy_score' and float(row['horizon_ms']) == 20.]
        assert len(selected) == 4 and {row['block'] for row in selected} == {'14', '15', '16', '17'}
    assert all(row['independent_blocks'] == '4' and row['training_seeds'] == '3' for row in summary + blocks)
    unavailable = [row for row in summary if row['metric'] == 'variogram_temporal_lag_4'
                   and float(row['horizon_ms']) == 20.]
    assert len(unavailable) == 2 and all(row['available'] == 'False' and row['fitted_minus_original_mean'] == '' for row in unavailable)
    assert any(float(row['fitted_minus_original']) < 0 for row in raw if row['available'] == 'True')
    assert not {'p_value', 'q_value', 'ci_lower', 'ci_upper'} & set(summary[0])
    assert all(sha256_file(output / name) == digest for name, digest in report['artifacts'].items())


def test_prior_matrix_cli_reads_only_sparse_staged_receipts_and_tables(prior_matrix, tmp_path, monkeypatch):
    staging = tmp_path / 'staged'
    for archive in prior_matrix['archives'].values():
        target = staging / archive.name
        (target / 'analysis').mkdir(parents=True)
        for name in ('delivery.json', 'analysis/analysis.json', 'analysis/forecast_paired_block_differences.csv'):
            shutil.copy2(archive / name, target / name)
        delivery = json.loads((archive / 'delivery.json').read_text())
        shutil.copy2(Path(delivery['source_comparison']) / 'delivery.json', target / 'source_delivery.json')
    shutil.rmtree(prior_matrix['inputs'])
    before = prior_matrix['chain'].read_bytes()
    output = tmp_path / 'staged_matrix'
    monkeypatch.setattr('sys.argv', ['lfads_forecast_analysis', '--prior-matrix', str(prior_matrix['chain']),
        '--reference-analysis', str(prior_matrix['reference']), '--matrix-inputs', str(staging), '--output', str(output)])
    analysis.main()
    assert json.loads((output / 'analysis.json').read_text())['completed_cells'] == 6
    assert prior_matrix['chain'].read_bytes() == before


@pytest.mark.parametrize('change', ['missing_cell', 'duplicate_cell', 'corrupt_csv', 'missing_row',
    'incomplete_delivery', 'source_hash', 'wrong_budget', 'protocol', 'cohort', 'within_budget_source',
    'availability', 'scientific_scope'])
def test_prior_matrix_rejects_incomplete_or_mismatched_evidence(prior_matrix, tmp_path, change):
    archive = prior_matrix['archives']['budget_80/seed_43']
    delivery_path, report_path = archive / 'delivery.json', archive / 'analysis/analysis.json'
    delivery, report = json.loads(delivery_path.read_text()), json.loads(report_path.read_text())
    table = archive / 'analysis/forecast_paired_block_differences.csv'
    if change in ('missing_cell', 'duplicate_cell'):
        chain = json.loads(prior_matrix['chain'].read_text())
        if change == 'missing_cell': chain['cells'].pop()
        else: chain['cells'][-1] = chain['cells'][0]
        prior_matrix['chain'].write_text(json.dumps(chain))
    elif change == 'corrupt_csv':
        table.write_text(table.read_text() + '\n')
    elif change == 'incomplete_delivery':
        delivery['analysis_exit_code'] = 1
        delivery_path.write_text(json.dumps(delivery))
    elif change == 'source_hash':
        source = Path(delivery['source_comparison']) / 'delivery.json'
        source.write_text(source.read_text() + '\n')
    else:
        if change == 'wrong_budget':
            source = Path(delivery['source_comparison']) / 'delivery.json'
            record = json.loads(source.read_text()); record['network_budget_epochs'] = 20
            source.write_text(json.dumps(record))
            delivery['source_delivery_sha256'] = sha256_file(source)
            report['completion_contracts']['fitted']['delivery_sha256'] = sha256_file(source)
        elif change == 'protocol': report['configuration']['origins_ms'][0] += 1
        elif change == 'cohort': report['training_binding']['dataset_sha256'] = '0' * 64
        elif change == 'within_budget_source': report['training_binding']['sources']['lfads_trajectory_training.py'] = 'changed'
        elif change == 'scientific_scope': report['scientific_claim_qualified'] = True
        else:
            with table.open() as stream: rows = list(csv.DictReader(stream))
            if change == 'missing_row': rows.pop()
            else:
                row = next(row for row in rows if row['available'] == 'False')
                row.update(available='True', original_prior='0', fitted_prior='0', fitted_minus_original='0')
            analysis._write_rows(table, rows)
            report['artifacts'][table.name] = sha256_file(table)
        report_path.write_text(json.dumps(report))
        _seal_delivery(archive, delivery, ['analysis/analysis.json', 'analysis/' + table.name])
    output = tmp_path / 'rejected_matrix'
    with pytest.raises(ValueError):
        analysis.aggregate_prior_matrix(prior_matrix['chain'], prior_matrix['reference'], output)
    assert not output.exists()

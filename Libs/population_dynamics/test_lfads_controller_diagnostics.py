"""Controller innovation math, frozen bindings and replay-gated serialization."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from Libs.population_dynamics import lfads_controller_diagnostics as controllers
from Libs.population_dynamics import test_lfads_boundary_analysis as analysis_fixtures
from Libs.population_dynamics.provenance import sha256_file


def prior_record(seed=41):
    logv = np.log(np.array([.09, .16, .25, .36])).astype(np.float32).astype(np.float64)
    logtau = np.log(np.array([7., 9., 11., 13.])).astype(np.float32).astype(np.float64)
    return dict(training_seed=seed, selected_epoch=19,
        selected_model_artifacts={'weights.npz': f'{seed:064x}', 'model.json': f'{seed + 100:064x}'},
        prior_parameters=dict(tensor_names=controllers.PRIOR_TENSORS, tensor_shape=[1, 4], tensor_dtype='float32',
            logevars=logv.tolist(), logatau=logtau.tolist(), phi=np.exp(-np.exp(-logtau)).tolist(),
            innovation_std=np.exp(.5 * logv).tolist(), autocorrelation_tau_bins=np.exp(logtau).tolist(),
            phi_expression='exp(-exp(-logatau))', innovation_std_expression='exp(0.5*logevars)',
            variance_semantics='innovation variance, not stationary process variance'))


def ar_path(previous, noise, phi, sigma, dtype=np.float64):
    values = []
    last = previous
    for tick in range(noise.shape[1]):
        last = (last * phi + noise[:, tick] * sigma).astype(dtype)
        values.append(last)
    return np.stack(values, axis=1)


def test_standardization_uses_boundary_and_channel_specific_parameters():
    rng = np.random.default_rng(127)
    previous = rng.normal(size=(3, 4)) + 7
    noise = rng.normal(size=(3, 7, 4))
    phi, sigma = np.array([.2, .4, .6, .8]), np.array([.1, .3, .5, .7])
    future = ar_path(previous, noise, phi, sigma)
    actual = controllers.standardized_innovations(previous, future, phi, sigma)
    np.testing.assert_allclose(actual, noise, rtol=1e-13, atol=1e-13)
    changed = controllers.standardized_innovations(previous + 2, future, phi, sigma)
    np.testing.assert_allclose(changed[:, 0] - actual[:, 0], np.broadcast_to(-2 * phi / sigma, (3, 4)))
    np.testing.assert_array_equal(changed[:, 1:], actual[:, 1:])
    with pytest.raises(ValueError, match='ALIGNED_CONTROLLER'):
        controllers.standardized_innovations(previous, future, phi, np.array([0., .3, .5, .7]))


def test_moments_have_declared_denominators_and_scalar_rotation_invariance():
    value = np.arange(48, dtype=np.float64).reshape(2, 6, 4) / 10 - 2
    mean, std, second, scalar = controllers.innovation_moments(value)
    np.testing.assert_allclose(second, std**2 + mean**2, atol=1e-14)
    assert scalar['vector_innovations'] == 12 and scalar['scalar_innovations'] == 48
    assert scalar['mean_squared_norm'] == pytest.approx(4 * scalar['mean_square_per_channel'])
    orthogonal = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]]) / np.sqrt(2)
    other = controllers.innovation_moments(value @ orthogonal)[3]
    assert other['mean_squared_norm'] == pytest.approx(scalar['mean_squared_norm'])
    assert other['mean_square_per_channel'] == pytest.approx(scalar['mean_square_per_channel'])


def test_mean_basis_fits_exclude_the_entire_held_out_block():
    time = np.linspace(-2, 2, 32)[None, None, None, :, None]
    template = np.arange(2)[None, :, None, None, None] * 3
    channels = np.arange(4)[None, None, None, None, :] + 1
    values = np.broadcast_to(time + template + channels, (3, 2, 4, 32, 4)).copy()
    predictions = controllers.leave_block_out_mean_predictions(values, 1)
    np.testing.assert_allclose(values[1] - predictions['template_time_mean'][:, None], 0, atol=1e-14)
    assert np.mean((values[1]-predictions['shared_intercept'][:, None])**2) > 0
    changed = values.copy(); changed[1] += 100
    for key, prediction in predictions.items():
        np.testing.assert_array_equal(prediction, controllers.leave_block_out_mean_predictions(changed, 1)[key])
    with pytest.raises(ValueError): controllers.leave_block_out_mean_predictions(values[:1], 0)
    with pytest.raises(ValueError): controllers.leave_block_out_mean_predictions(values, -1)


def test_residual_correlations_do_not_join_draw_boundaries_or_impute_zero_variance():
    time = np.arange(32)[None, :] + np.array([[0], [1000]])
    values = np.stack((time, -time, 2*time, np.full_like(time, 5)), axis=-1)[None]
    cross, temporal = controllers.residual_structure(values)
    for row in cross:
        if row['right_channel'] == 3:
            assert row['correlation'] is None
        elif row['left_channel'] == 0:
            assert row['correlation'] == pytest.approx(-1 if row['right_channel'] == 1 else 1)
    for row in temporal:
        assert row['paired_vectors'] == 2 * (32-row['lag_bins'])
        if row['channel'] == 3:
            assert row['correlation'] is None
        else:
            assert row['correlation'] == pytest.approx(1)
    with pytest.raises(ValueError): controllers.residual_structure(values, lags=(32,))


@pytest.fixture
def fitting_cache_fixture(tmp_path):
    from .test_lfads_prior_refit import CacheExecutionTests
    helper = CacheExecutionTests()
    helper.setUp()
    try:
        helper.run_cache()
        record_path = helper.output / 'posterior_cache.json'
        cache = json.loads(record_path.read_text())
        paired = tmp_path / 'paired'; (paired / 'analysis').mkdir(parents=True)
        prior = dict(fitting_episodes=56, fitted_parameters=8,
            posterior_cache_sha256=sha256_file(record_path), prior_parameters_sha256=cache['prior_parameters_sha256'])
        analysis = dict(status='DESCRIBED_REPLAYED_PRIOR_COMPARISON',
            training_binding=cache['binding']['training_binding'],
            arm_bindings={'fitted_prior': dict(prior_refit=prior, selected_model_artifacts=cache['binding']['selected_model_artifacts'])})
        path = paired / 'analysis/analysis.json'; path.write_text(json.dumps(analysis))
        (paired / 'delivery.json').write_text(json.dumps(dict(status=analysis['status'],
            files=[dict(path='analysis/analysis.json', sha256=sha256_file(path))])))
        yield helper, paired
    finally:
        helper.doCleanups()


def test_fitting_cache_cli_keeps_all_blocks_and_does_not_resample(fitting_cache_fixture, tmp_path, monkeypatch):
    helper, paired = fitting_cache_fixture
    samples_before = helper.sampler.call_count
    output = tmp_path / 'mean_diagnostic'
    monkeypatch.setattr('sys.argv', ['lfads_controller_diagnostics', '--fitting-cache', str(helper.output),
        '--paired-analysis', str(paired), '--output', str(output)])
    controllers.main()
    result = json.loads((output / 'controller_diagnostics.json').read_text())
    assert result['fitting_blocks'] == list(range(14)) and result['excluded_selection_blocks'] == [14, 15, 16, 17]
    assert result['episodes'] == 56 and result['transition_vectors'] == 1074304
    assert not result['prior_parameters_modified'] and not result['scientific_claim_qualified']
    assert not result['posterior_mean_and_variance_arrays_available'] and helper.sampler.call_count == samples_before
    with (output / 'mean_models_by_case.csv').open() as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 504 and {int(row['held_out_block']) for row in rows} == set(range(14))
    assert {row['draw_split'] for row in rows} == {'all', 'first_half', 'second_half'}
    assert all(sha256_file(output / name) == digest for name, digest in result['artifacts'].items())


def test_fitting_cache_changed_payload_is_rejected_before_output(fitting_cache_fixture, tmp_path):
    helper, paired = fitting_cache_fixture
    path = helper.output / 'train_block_000_familiar_0.npz'
    with path.open('ab') as stream: stream.write(b'changed')
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='CACHE_IDENTITY_CHANGED'):
        controllers.analyze_fitting_cache(helper.output, paired, output)
    assert not output.exists()


def test_prior_logs_formula_and_checkpoint_binding_are_exact():
    saved = prior_record()
    evaluation = {key: saved[key] for key in ('training_seed', 'selected_epoch', 'selected_model_artifacts')}
    phi, sigma = controllers.prior_from_audit(saved, evaluation)
    np.testing.assert_array_equal(phi, saved['prior_parameters']['phi'])
    np.testing.assert_array_equal(sigma, saved['prior_parameters']['innovation_std'])
    for key in ('phi', 'innovation_std', 'autocorrelation_tau_bins'):
        changed = copy.deepcopy(saved)
        changed['prior_parameters'][key][0] += 1e-12
        with pytest.raises(ValueError, match='DERIVED_FORMULA'):
            controllers.prior_from_audit(changed, evaluation)
    changed = copy.deepcopy(saved)
    changed['prior_parameters']['logevars'][0] += 1e-12
    with pytest.raises(ValueError, match='EXACT_FINITE_SAVED_FLOAT32'):
        controllers.prior_from_audit(changed, evaluation)
    changed = copy.deepcopy(evaluation); changed['selected_model_artifacts']['weights.npz'] = '0' * 64
    with pytest.raises(ValueError, match='CHECKPOINT_BINDING'):
        controllers.prior_from_audit(saved, changed)


def _write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def controller_fixture(tmp_path):
    """Synthetic controller-only archives; no real diagnostic outputs are read."""
    root = tmp_path / 'diagnostics'; root.mkdir()
    audit_path = tmp_path / 'audit.json'
    source_file = Path(controllers.boundary_analysis.__file__).resolve().parents[1] / 'external/lfads/distributions.py'
    audit = dict(status='READ_ONLY_SOURCE_AUDIT_COMPLETE', scientific_claim_qualified=False,
        model_graph_built=False, training_executed=False, source_revision='fixture_source_revision',
        source_files_sha256={'external/lfads/distributions.py': sha256_file(source_file)},
        seeds=[prior_record(seed) for seed in controllers.SEEDS])
    helper = analysis_fixtures.BoundaryAnalysisTests()
    for seed, source in zip(controllers.SEEDS, audit['seeds']):
        folder = root / f'seed_{seed}'; folder.mkdir()
        rows, report = helper.fixture(folder)
        report.update(training_seed=seed, source_revision=audit['source_revision'],
            selected_epoch=source['selected_epoch'], selected_model_artifacts=source['selected_model_artifacts'],
            training_binding={'config': {'model': {'co_dim': 4}}}, dataset_sha256='d' * 64,
            acquisition_design_sha256='a' * 64, native_source_identity={'fixture': 'no raw unit data'})
        for row in rows:
            row['seed'] = seed
        phi = np.asarray(source['prior_parameters']['phi'])
        sigma = np.asarray(source['prior_parameters']['innovation_std'])
        rng = np.random.default_rng(seed)
        for block in range(14, 18):
            for template in range(4):
                case = f'train_block_{block:03d}_familiar_{template}'
                for branch in controllers.UNCROSSED_BRANCHES:
                    history = rng.normal(size=(16, 200, 4)).astype(np.float32)
                    noise = rng.normal(size=(16, 100, 4))
                    if branch == controllers.UNCROSSED_BRANCHES[1]:
                        noise = .5 + 2 * noise
                    future = ar_path(history[:, -1], noise, phi, sigma, np.float32)
                    name = f'{case}_{branch}.npz'
                    np.savez_compressed(folder / name,
                        lfads_historical_controller_inputs=history, lfads_future_controller_inputs=future)
                    report['artifacts'][name] = sha256_file(folder / name)
        _write_csv(folder / 'scores.csv', rows)
        report['scores_sha256'] = sha256_file(folder / 'scores.csv')
        (folder / 'evaluation.json').write_text(json.dumps(report))
        replay = dict(status='PASS', training_seed=seed, verified_cases=16, verified_branches=64,
            verified_score_rows=192, all_branch_unit_metrics_replayed=True,
            evaluation_sha256=sha256_file(folder / 'evaluation.json'), scores_sha256=report['scores_sha256'])
        (root / f'seed_{seed}_replay.json').write_text(json.dumps(replay))
    audit_path.write_text(json.dumps(audit))
    return root, audit_path, tmp_path / 'result'


def test_complete_fixture_writes_seed_specific_channels_and_scalar_summary(controller_fixture):
    root, audit, output = controller_fixture
    result = controllers.analyze_controllers(root, audit, output)
    assert result['status'] == 'DESCRIBED_REPLAYED_CONTROLLER_INNOVATIONS'
    assert result['channel_rows'] == 1152 and result['case_horizon_rows'] == 288
    assert result['controller_artifacts_verified'] == 96
    assert len(result['scalar_moments_by_seed']) == 18
    assert len(result['scalar_summary_across_seeds']) == 12
    assert all(not row['channel_coordinates_pooled_across_seeds'] for row in result['scalar_summary_across_seeds'])
    assert not result['scientific_claim_qualified'] and result['selection_only']
    with (output / 'controller_innovations.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1152 and {int(row['channel']) for row in rows} == set(range(4))
    assert {int(row['seed']) for row in rows} == {41, 42, 43}
    assert {int(row['vector_innovations']) for row in rows} == {64, 256, 1600}
    for seed in controllers.SEEDS:
        norms = {row['branch']: row['mean_square_per_channel'] for row in result['scalar_moments_by_seed']
                 if row['seed'] == seed and row['horizon_ms'] == 500}
        assert norms[controllers.UNCROSSED_BRANCHES[1]] > norms[controllers.UNCROSSED_BRANCHES[0]]
    written = json.loads((output / 'controller_diagnostics.json').read_text())
    assert written == result
    assert result['artifacts']['controller_innovations.csv'] == sha256_file(output / 'controller_innovations.csv')
    with pytest.raises(FileExistsError):
        controllers.analyze_controllers(root, audit, output)


@pytest.mark.parametrize('corruption', ['replay_status', 'artifact', 'prior_formula', 'checkpoint', 'path_shape', 'role'])
def test_invalid_evidence_creates_no_analysis_output(controller_fixture, corruption):
    root, audit_path, output = controller_fixture
    folder = root / 'seed_41'
    report = json.loads((folder / 'evaluation.json').read_text())
    replay_path = root / 'seed_41_replay.json'
    receipt = json.loads(replay_path.read_text())
    name = 'train_block_014_familiar_0_' + controllers.UNCROSSED_BRANCHES[0] + '.npz'
    if corruption == 'replay_status':
        receipt['status'] = 'MISMATCH'
        replay_path.write_text(json.dumps(receipt))
    elif corruption in ('artifact', 'path_shape'):
        with np.load(folder / name) as arrays:
            history, future = arrays['lfads_historical_controller_inputs'], arrays['lfads_future_controller_inputs']
        if corruption == 'artifact':
            future = future + 1
        else:
            future = future[:, :-1]
        np.savez_compressed(folder / name, lfads_historical_controller_inputs=history,
                            lfads_future_controller_inputs=future)
        if corruption == 'path_shape':
            report['artifacts'][name] = sha256_file(folder / name)
            (folder / 'evaluation.json').write_text(json.dumps(report))
            receipt['evaluation_sha256'] = sha256_file(folder / 'evaluation.json')
            replay_path.write_text(json.dumps(receipt))
    elif corruption in ('prior_formula', 'checkpoint'):
        audit = json.loads(audit_path.read_text())
        if corruption == 'prior_formula':
            audit['seeds'][0]['prior_parameters']['phi'][0] += 1e-12
        else:
            audit['seeds'][0]['selected_model_artifacts']['weights.npz'] = '0' * 64
        audit_path.write_text(json.dumps(audit))
    else:
        with (folder / 'scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        rows[9]['role'] = 'forecast'
        _write_csv(folder / 'scores.csv', rows)
        report['scores_sha256'] = sha256_file(folder / 'scores.csv')
        (folder / 'evaluation.json').write_text(json.dumps(report))
        receipt.update(evaluation_sha256=sha256_file(folder / 'evaluation.json'), scores_sha256=report['scores_sha256'])
        replay_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        controllers.analyze_controllers(root, audit_path, output)
    assert not output.exists()

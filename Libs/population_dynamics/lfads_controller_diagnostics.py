"""Source-bound AR innovation summaries from replayed LFADS controller paths.

Only the uncrossed prefix/prior and full-response/posterior paths are compared.
Channel coordinates remain seed specific. Across-seed summaries contain scalar
squared norms or mean squares only. These are selection-block model diagnostics,
not exact posterior, mechanistic-microstate or biological causal evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from Libs.population_dynamics import lfads_boundary_analysis as boundary_analysis
from Libs.population_dynamics.lfads_boundary_analysis import BRANCHES, SEEDS, read_rows
from Libs.population_dynamics.provenance import sha256_file


UNCROSSED_BRANCHES = (BRANCHES[0], BRANCHES[3])
PRIOR_TENSORS = dict(logevars='LFADS/z/u_prior_ar1/logevars:0',
                     logatau='LFADS/z/u_prior_ar1/logatau:0')
CHANNELS = 4
CACHE_LAGS = (1, 4, 16)
MEAN_MODELS = ('zero_mean', 'shared_intercept', 'template_time_mean')


def prior_from_audit(seed_audit, evaluation):
    """Check exact saved-log representation, derivations and checkpoint binding."""
    if (seed_audit['training_seed'] != evaluation['training_seed']
            or seed_audit['selected_model_artifacts'] != evaluation['selected_model_artifacts']
            or seed_audit['selected_epoch'] != evaluation['selected_epoch']):
        raise ValueError('SOURCE_PRIOR_CHECKPOINT_BINDING_CHANGED')
    params = seed_audit['prior_parameters']
    if (params['tensor_names'] != PRIOR_TENSORS or params['tensor_shape'] != [1, CHANNELS]
            or params['tensor_dtype'] != 'float32'
            or params['phi_expression'] != 'exp(-exp(-logatau))'
            or params['innovation_std_expression'] != 'exp(0.5*logevars)'
            or params['variance_semantics'] != 'innovation variance, not stationary process variance'):
        raise ValueError('SOURCE_PRIOR_PARAMETER_CONTRACT_CHANGED')
    logs = {}
    for key in ('logevars', 'logatau'):
        value = np.asarray(params[key], dtype=np.float64)
        if (value.shape != (CHANNELS,) or not np.isfinite(value).all()
                or not np.array_equal(value, value.astype(np.float32).astype(np.float64))):
            raise ValueError('EXACT_FINITE_SAVED_FLOAT32_PRIOR_LOGS_REQUIRED')
        logs[key] = value
    phi = np.exp(-np.exp(-logs['logatau']))
    sigma = np.exp(.5 * logs['logevars'])
    tau = np.exp(logs['logatau'])
    if (not np.isfinite(phi).all() or not np.isfinite(sigma).all() or not np.isfinite(tau).all()
            or (phi <= 0).any() or (phi >= 1).any() or (sigma <= 0).any()):
        raise ValueError('FINITE_ADMISSIBLE_SOURCE_AR_PARAMETERS_REQUIRED')
    for key, actual in (('phi', phi), ('innovation_std', sigma), ('autocorrelation_tau_bins', tau)):
        reported = np.asarray(params[key], dtype=np.float64)
        if reported.shape != actual.shape or not np.array_equal(reported, actual):
            raise ValueError('SOURCE_PRIOR_DERIVED_FORMULA_MISMATCH:' + key)
    return phi, sigma


def standardized_innovations(previous, future, phi, innovation_std):
    """Use the actual boundary input for t=0 and stored future inputs thereafter."""
    previous, future, phi, sigma = [np.asarray(value, dtype=np.float64)
                                   for value in (previous, future, phi, innovation_std)]
    if (future.ndim != 3 or min(future.shape) < 1 or future.shape[-1] != CHANNELS
            or previous.shape != (future.shape[0], CHANNELS)
            or phi.shape != (CHANNELS,) or sigma.shape != (CHANNELS,)
            or any(not np.isfinite(v).all() for v in (previous, future, phi, sigma))
            or (phi <= 0).any() or (phi >= 1).any() or (sigma <= 0).any()):
        raise ValueError('FINITE_ALIGNED_CONTROLLER_PATH_AND_AR_PARAMETERS_REQUIRED')
    predecessor = np.concatenate((previous[:, None], future[:, :-1]), axis=1)
    result = (future - predecessor * phi) / sigma
    if not np.isfinite(result).all():
        raise ValueError('NONFINITE_STANDARDIZED_CONTROLLER_INNOVATIONS')
    return result


def innovation_moments(innovations):
    values = np.asarray(innovations, dtype=np.float64)
    if (values.ndim != 3 or min(values.shape) < 1 or values.shape[-1] != CHANNELS
            or not np.isfinite(values).all()):
        raise ValueError('FINITE_DRAW_TIME_CHANNEL_INNOVATIONS_REQUIRED')
    mean = values.mean(axis=(0, 1))
    std = values.std(axis=(0, 1), ddof=0)
    second = np.square(values).mean(axis=(0, 1))
    scalar = dict(mean_squared_norm=float(np.square(values).sum(axis=-1).mean()),
                  mean_square_per_channel=float(second.mean()),
                  vector_innovations=values.shape[0] * values.shape[1],
                  scalar_innovations=int(values.size))
    return mean, std, second, scalar


def leave_block_out_mean_predictions(values, held_out):
    """Fit diagnostic mean bases without any draw from the omitted block.

    Axes are block, known stimulus template, draw, time, channel. The richest
    basis uses the existing 5-ms clock, not a window or basis selected by scores.
    It is an analysis of variational paths, not a fitted causal simulator.
    """
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 5 or min(values.shape) < 1 or values.shape[0] < 2
            or values.shape[-1] != CHANNELS or not np.isfinite(values).all()
            or type(held_out) is not int or not 0 <= held_out < len(values)):
        raise ValueError('finite block/template/draw/time/four-channel paths required')
    training = values[np.arange(len(values)) != held_out]
    shape = (values.shape[1], values.shape[3], CHANNELS)
    return dict(zero_mean=np.zeros(shape),
        shared_intercept=np.broadcast_to(training.mean(axis=(0, 1, 2, 3)), shape),
        template_time_mean=training.mean(axis=(0, 2)))


def residual_structure(values, lags=CACHE_LAGS):
    """Centered residual correlations, with no pair crossing case/draw boundaries."""
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 4 or min(values.shape) < 1 or values.shape[-1] != CHANNELS
            or not np.isfinite(values).all() or any(type(lag) is not int or not 0 < lag < values.shape[2] for lag in lags)):
        raise ValueError('finite template/draw/time/channel paths and supported lags required')
    flat = values.reshape(-1, CHANNELS)
    centered = flat - flat.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    variance = np.diag(covariance)
    correlation = np.full((CHANNELS, CHANNELS), np.nan)
    denominator = np.sqrt(variance[:, None] * variance[None])
    np.divide(covariance, denominator, out=correlation, where=denominator > 0)
    cross = []
    for left in range(CHANNELS):
        for right in range(left + 1, CHANNELS):
            cross.append(dict(left_channel=left, right_channel=right,
                covariance=float(covariance[left, right]),
                correlation=None if not np.isfinite(correlation[left, right]) else float(correlation[left, right])))
    temporal = []
    for lag in lags:
        left = values[:, :, :-lag].reshape(-1, CHANNELS)
        right = values[:, :, lag:].reshape(-1, CHANNELS)
        left, right = left-left.mean(axis=0), right-right.mean(axis=0)
        cov = np.mean(left * right, axis=0)
        denom = np.sqrt(np.mean(left**2, axis=0) * np.mean(right**2, axis=0))
        for channel in range(CHANNELS):
            temporal.append(dict(lag_bins=lag, channel=channel, paired_vectors=len(left),
                covariance=float(cov[channel]),
                correlation=None if denom[channel] == 0 else float(cov[channel] / denom[channel])))
    return cross, temporal


def analyze_fitting_cache(cache, paired_analysis, output):
    """Diagnose fixed-AR mean misspecification using fitting blocks only."""
    # The cache verifier and its TF dependencies are only needed for this mode.
    from .lfads_prior_refit import _cache_identity, _cached_paths

    cache, paired_analysis, output = map(Path, (cache, paired_analysis, output))
    if output.exists():
        raise FileExistsError(output)
    delivery = json.loads((paired_analysis / 'delivery.json').read_text())
    analysis_path = paired_analysis / 'analysis/analysis.json'
    row = next(r for r in delivery['files'] if r['path'] == 'analysis/analysis.json')
    if (delivery['status'] != 'DESCRIBED_REPLAYED_PRIOR_COMPARISON'
            or sha256_file(analysis_path) != row['sha256']):
        raise ValueError('verified complete paired prior analysis required')
    comparison = json.loads(analysis_path.read_text())
    prior = comparison['arm_bindings']['fitted_prior']['prior_refit']
    record_path = cache / 'posterior_cache.json'
    if (comparison['status'] != 'DESCRIBED_REPLAYED_PRIOR_COMPARISON'
            or prior['fitting_episodes'] != 56 or prior['fitted_parameters'] != 8
            or sha256_file(record_path) != prior['posterior_cache_sha256']):
        raise ValueError('paired forecast fitting cache identity changed')
    record = json.loads(record_path.read_text())
    binding = record['binding']
    ids = [f'train_block_{b:03d}_familiar_{t}' for b in range(14) for t in range(4)]
    if (record['status'] != 'FITTED_FROZEN_NETWORK_AR_PRIOR' or not record['fitting_cohort_extracted']
            or record['formal_test_accessed'] or record['complete_episodes'] != 56
            or record['posterior_paths'] != 896 or binding['fitting_blocks'] != list(range(14))
            or binding['excluded_selection_blocks'] != list(range(14, 18))
            or binding['selected_model_artifacts'] != comparison['arm_bindings']['fitted_prior']['selected_model_artifacts']
            or binding['training_binding'] != comparison['training_binding']
            or [r['case_id'] for r in record['episodes']] != ids):
        raise ValueError('complete source-bound fitting-only cache required')
    parameter_path = cache / 'prior_parameters.npz'
    if sha256_file(parameter_path) != prior['prior_parameters_sha256']:
        raise ValueError('paired fitted AR parameters changed')
    with np.load(parameter_path, allow_pickle=False) as parameters:
        logs = {k: parameters[k] for k in ('logatau', 'logevars')}
    if any(v.dtype != np.float32 or v.shape != (1, CHANNELS) or not np.isfinite(v).all() for v in logs.values()):
        raise ValueError('exact fitted float32 AR logs required')
    phi = np.exp(-np.exp(-logs['logatau'][0].astype(np.float64)))
    sigma = np.exp(.5 * logs['logevars'][0].astype(np.float64))
    innovations = []
    for row in record['episodes']:
        case = {k: row[k] for k in ('case_id', 'block', 'template_id')}
        paths = _cached_paths(cache, case, row, _cache_identity(binding, case))
        innovations.append(standardized_innovations(paths[:, 0], paths[:, 1:], phi, sigma))
    values = np.stack(innovations).reshape(14, 4, 16, 1199, CHANNELS)
    cases, blocks, cross_rows, temporal_rows = [], [], [], []
    for held_out in range(14):
        predictions = leave_block_out_mean_predictions(values, held_out)
        for model in MEAN_MODELS:
            error = values[held_out] - predictions[model][:, None]
            cross, temporal = residual_structure(error)
            identity = dict(seed=binding['training_seed'], held_out_block=held_out, mean_model=model)
            cross_rows.extend(dict(identity, **row) for row in cross)
            temporal_rows.extend(dict(identity, **row) for row in temporal)
            for split, selected in [('all', slice(None)), ('first_half', slice(0, 8)), ('second_half', slice(8, 16))]:
                scores = []
                for template in range(4):
                    mse = float(np.mean(error[template, selected] ** 2))
                    scores.append(mse)
                    cases.append(dict(identity, template_id=f'familiar_{template}', draw_split=split,
                        residual_mean_square=mse, vector_values=error[template, selected].shape[0] * 1199))
                blocks.append(dict(identity, draw_split=split, residual_mean_square=float(np.mean(scores))))
    summary = []
    for model in MEAN_MODELS:
        for split in ('all', 'first_half', 'second_half'):
            selected = [r for r in blocks if r['mean_model'] == model and r['draw_split'] == split]
            baseline = [r['residual_mean_square'] for r in blocks if r['mean_model'] == 'zero_mean' and r['draw_split'] == split]
            delta = [r['residual_mean_square']-base for r,base in zip(selected, baseline)]
            summary.append(dict(seed=binding['training_seed'], mean_model=model, draw_split=split,
                residual_mean_square=float(np.mean([r['residual_mean_square'] for r in selected])),
                mean_change_from_zero=float(np.mean(delta)), minimum_block_change=min(delta),
                maximum_block_change=max(delta), improved_blocks=sum(d < 0 for d in delta), blocks=14))
    output.mkdir(parents=True, exist_ok=False)
    tables = {'mean_models_by_case.csv': cases, 'mean_models_by_block.csv': blocks,
              'mean_models_summary.csv': summary, 'residual_cross_channel.csv': cross_rows,
              'residual_temporal.csv': temporal_rows}
    for name, rows in tables.items():
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='DESCRIBED_FITTING_CACHE_BLOCK_HELD_OUT_MEANS', training_seed=binding['training_seed'],
        fitting_blocks=list(range(14)), excluded_selection_blocks=list(range(14, 18)), episodes=56,
        posterior_paths=896, transition_vectors=14*4*16*1199, channels=4, lags_bins=list(CACHE_LAGS),
        mean_models=list(MEAN_MODELS), fit_unit='13 whole initial/background blocks; held-out block excluded in every fold',
        held_out_scope='additional diagnostic mean fit only; not end-to-end model validation',
        fixed_network_and_ar_fit_blocks=binding['fitting_blocks'],
        mean_basis='zero; shared channel intercept; known template and native elapsed-time bin',
        template_conditioning='four preplanned familiar stimulus schedules; no test-block response used in centering fits',
        posterior_mean_and_variance_arrays_available=False,
        residual_scope='AR-filtered full-response variational paths; not identified mechanistic or process innovations',
        half_draw_scope='Monte Carlo stability only; halves are not independent biological replicates',
        correlation_scope='centered pooled template/draw/time residuals; temporal pairs never cross paths',
        fitting_cache_sha256=prior['posterior_cache_sha256'], prior_parameters_sha256=prior['prior_parameters_sha256'],
        paired_analysis_sha256=sha256_file(analysis_path), phi=phi.tolist(), innovation_std=sigma.tolist(),
        ar_formula_dtype='float64 from exact saved float32 logs', prior_parameters_modified=False,
        new_forecasts=0, scientific_claim_qualified=False, formal_test_accessed=False, connectivity_instances=1,
        implementation_sha256=sha256_file(Path(__file__)), artifacts={name: sha256_file(output / name) for name in tables})
    (output / 'controller_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
def _replay_gate(root, folder, report, seed):
    path = root / f'seed_{seed}_replay.json'
    replay = json.loads(path.read_text())
    if (replay['status'] != 'PASS' or replay['training_seed'] != seed
            or replay['verified_cases'] != 16 or replay['verified_branches'] != 64
            or replay['verified_score_rows'] != 192
            or replay['evaluation_sha256'] != sha256_file(folder / 'evaluation.json')
            or replay['scores_sha256'] != report['scores_sha256']
            or replay['all_branch_unit_metrics_replayed'] is not True):
        raise ValueError('COMPLETE_INDEPENDENT_REPLAY_PASS_REQUIRED')
    return path


def summarize_scalar_moments(case_rows):
    """Pool scalar energies within each seed, then summarize seed-specific means."""
    groups = defaultdict(list)
    for row in case_rows:
        groups[(row['seed'], row['branch'], row['horizon_ms'])].append(row)
    by_seed = []
    for (seed, branch, horizon), members in sorted(groups.items()):
        if (len(members) != 16 or len({row['case_id'] for row in members}) != 16
                or {row['block'] for row in members} != {14, 15, 16, 17}):
            raise ValueError('ALL_SELECTION_CASES_REQUIRED_FOR_CONTROLLER_SUMMARY')
        total = sum(row['vector_innovations'] for row in members)
        by_seed.append(dict(seed=seed, branch=branch, horizon_ms=horizon, cases=16,
            vector_innovations=total, scalar_innovations=total * CHANNELS,
            **{name: sum(row[name] * row['vector_innovations'] for row in members) / total
               for name in ('mean_squared_norm', 'mean_square_per_channel')}))
    expected = {(seed, branch, float(ms)) for seed in SEEDS for branch in UNCROSSED_BRANCHES
                for ms in (20., 80., 500.)}
    if set(groups) != expected:
        raise ValueError('ALL_SEEDS_UNCROSSED_BRANCHES_AND_HORIZONS_REQUIRED')
    summary = []
    for branch in UNCROSSED_BRANCHES:
        for horizon in (20., 80., 500.):
            members = [row for row in by_seed if (row['branch'], row['horizon_ms']) == (branch, horizon)]
            for metric in ('mean_squared_norm', 'mean_square_per_channel'):
                values = np.asarray([row[metric] for row in members], dtype=np.float64)
                summary.append(dict(branch=branch, horizon_ms=horizon, metric=metric,
                    mean_of_seed_means=float(values.mean()), minimum_seed_mean=float(values.min()),
                    maximum_seed_mean=float(values.max()), training_seeds=len(SEEDS), selection_cases_per_seed=16,
                    channel_coordinates_pooled_across_seeds=False))
    return by_seed, summary


def analyze_controllers(diagnostics_root, source_audit_path, output):
    root, audit_path, output = map(Path, (diagnostics_root, source_audit_path, output))
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'READ_ONLY_SOURCE_AUDIT_COMPLETE'
            or audit['scientific_claim_qualified'] is not False
            or audit['model_graph_built'] is not False or audit['training_executed'] is not False
            or [row['training_seed'] for row in audit['seeds']] != list(SEEDS)):
        raise ValueError('COMPLETE_ORDERED_THREE_SEED_SOURCE_AUDIT_REQUIRED')
    source_file = Path(boundary_analysis.__file__).resolve().parents[1] / 'external/lfads/distributions.py'
    if audit['source_files_sha256']['external/lfads/distributions.py'] != sha256_file(source_file):
        raise ValueError('AUDITED_SOURCE_AR_DISTRIBUTION_IMPLEMENTATION_CHANGED')
    channel_rows, case_rows = [], []
    bindings, prior_records = {}, {}
    common_config = common_source = None
    for seed, seed_audit in zip(SEEDS, audit['seeds']):
        folder = root / f'seed_{seed}'
        report = json.loads((folder / 'evaluation.json').read_text())
        replay_path = _replay_gate(root, folder, report, seed)
        indexed = read_rows(folder, report)
        if (report['training_seed'] != seed or report['source_revision'] != audit['source_revision']
                or report['training_binding']['config']['model']['co_dim'] != CHANNELS):
            raise ValueError('DIAGNOSTIC_SEED_SOURCE_OR_CONTROLLER_DIMENSION_CHANGED')
        phi, sigma = prior_from_audit(seed_audit, report)
        config = report['configuration']
        source = {key: report[key] for key in ('dataset_sha256', 'acquisition_design_sha256', 'native_source_identity')}
        if common_config is None:
            common_config, common_source = config, source
        if config != common_config or source != common_source:
            raise ValueError('CONTROLLER_COMPARISON_COHORT_OR_PROTOCOL_CHANGED_ACROSS_SEEDS')
        prior_records[str(seed)] = seed_audit['prior_parameters']
        checked = {}
        for block in range(14, 18):
            for template in range(4):
                case = f'train_block_{block:03d}_familiar_{template}'
                for branch in UNCROSSED_BRANCHES:
                    row = indexed[(case, branch, 500.)]
                    name = row['ensemble_artifact']
                    path = folder / name
                    if Path(name).name != name or sha256_file(path) != report['artifacts'][name]:
                        raise ValueError('CONTROLLER_PATH_ARTIFACT_CHANGED:' + name)
                    checked[name] = report['artifacts'][name]
                    with np.load(path, allow_pickle=False) as saved:
                        history = saved['lfads_historical_controller_inputs']
                        future = saved['lfads_future_controller_inputs']
                    if (history.shape != (config['joint_samples'], config['original_history_bins'], CHANNELS)
                            or future.shape != (config['joint_samples'], config['forecast_horizon_bins'], CHANNELS)
                            or history.dtype != np.float32 or future.dtype != np.float32
                            or not np.isfinite(history).all() or not np.isfinite(future).all()):
                        raise ValueError('COMPLETE_FINITE_SOURCE_CONTROLLER_PATHS_REQUIRED')
                    innovations = standardized_innovations(history[:, -1], future, phi, sigma)
                    for horizon in config['horizons_ms']:
                        bins = int(horizon / config['bin_ms'])
                        mean, std, second, scalar = innovation_moments(innovations[:, :bins])
                        identifiers = dict(seed=seed, block=block, case_id=case, template_id=f'familiar_{template}',
                            branch=branch, role=row['role'], origin_ms=1000., horizon_ms=horizon,
                            ensemble_artifact=name, ensemble_sha256=checked[name])
                        case_rows.append(dict(identifiers, **scalar))
                        for channel in range(CHANNELS):
                            channel_rows.append(dict(identifiers, channel=channel,
                                vector_innovations=scalar['vector_innovations'], mean=float(mean[channel]),
                                standard_deviation_ddof0=float(std[channel]), second_moment=float(second[channel]),
                                phi=float(phi[channel]), innovation_std=float(sigma[channel])))
        bindings[str(seed)] = dict(evaluation_sha256=sha256_file(folder / 'evaluation.json'),
            replay_sha256=sha256_file(replay_path), scores_sha256=report['scores_sha256'],
            selected_model_artifacts=report['selected_model_artifacts'], controller_artifacts=checked)
    if len(channel_rows) != 3 * 16 * 2 * 3 * CHANNELS or len(case_rows) != 3 * 16 * 2 * 3:
        raise ValueError('COMPLETE_CONTROLLER_CHANNEL_DENOMINATOR_REQUIRED')
    by_seed, summary = summarize_scalar_moments(case_rows)
    result = dict(status='DESCRIBED_REPLAYED_CONTROLLER_INNOVATIONS',
        training_seeds=list(SEEDS), branches=list(UNCROSSED_BRANCHES), selection_cases_per_seed=16,
        channel_rows=len(channel_rows), case_horizon_rows=len(case_rows), controller_artifacts_verified=96,
        source_audit_sha256=sha256_file(audit_path), source_revision=audit['source_revision'],
        source_distribution_sha256=sha256_file(source_file), model_seed_bindings=bindings,
        source_parameters_by_seed=prior_records, configuration=common_config, source_binding=common_source,
        formula='(u_t - phi * u_previous) / innovation_std; u_previous is last historical input at t=0',
        calculation_dtype='float64 from saved float32 source paths and exact saved log parameters',
        per_channel_statistics='pooled draws and time within case; population standard deviation ddof=0',
        case_scalar_moments=case_rows, scalar_moments_by_seed=by_seed, scalar_summary_across_seeds=summary,
        channel_coordinates_pooled_across_seeds=False,
        scientific_claim_qualified=False, formal_test_accessed=False, inverse_intervention_evidence=False,
        microstate_oracle=False, selection_only=True,
        interpretation='descriptive model-conditioning diagnostics; no exact-posterior or biological causal claim',
        moment_scope='agreement of these moments does not establish equal distributions or an adequate controller prior; temporal and cross-channel distribution structure is not tested',
        uncertainty='draws and time bins are not independent biological units; four selection blocks recur across templates and training seeds; seed ranges are not confidence intervals',
        data_scope='only saved historical/future controller arrays read; no full-unit counts read',
        artifact_hash_scope='all 96 consumed controller archives; full-unit archives are not read by this analysis',
        implementation_sha256=sha256_file(Path(__file__)))
    # Gate and calculate all seeds before creating any analytical result.
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'controller_innovations.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(channel_rows[0]))
        writer.writeheader(); writer.writerows(channel_rows)
    result['artifacts'] = {csv_path.name: sha256_file(csv_path)}
    (output / 'controller_diagnostics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--diagnostics', type=Path)
    mode.add_argument('--fitting-cache', type=Path)
    parser.add_argument('--source-audit', type=Path)
    parser.add_argument('--paired-analysis', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.fitting_cache is not None:
        if args.paired_analysis is None: parser.error('--fitting-cache requires --paired-analysis')
        result = analyze_fitting_cache(args.fitting_cache, args.paired_analysis, args.output)
    else:
        if args.source_audit is None: parser.error('--diagnostics requires --source-audit')
        result = analyze_controllers(args.diagnostics, args.source_audit, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'training_seeds', 'training_seed', 'episodes',
        'channel_rows', 'controller_artifacts_verified') if key in result}))


if __name__ == '__main__':
    main()

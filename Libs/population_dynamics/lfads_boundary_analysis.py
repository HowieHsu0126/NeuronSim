"""Describe the registered LFADS boundary/input crossing after independent replay.

The four realization blocks selected epochs. Contrasts therefore describe model
diagnostics; they are neither independent hypothesis tests nor biological effects.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .population_forecast_replay import METRIC_ATOL, METRIC_RTOL, UNIT_METRICS
from .provenance import sha256_file
from .trajectory_forecast_analysis import _write_rows, covariance_components, pair_sets

SEEDS = (41, 42, 43)
BRANCHES = (
    'prefix_boundary_prior_future', 'full_response_boundary_prior_future',
    'prefix_boundary_posterior_future', 'full_response_boundary_posterior_future',
)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metric_names(config):
    return [
        'energy_score', 'variogram_spatial', 'variogram_cross_population',
        *(f'variogram_temporal_lag_{lag}' for lag in config['temporal_lags_bins']),
        *UNIT_METRICS,
        *(f'refractory_count_violation_{w}bin' for w in config['refractory_support_windows_bins']),
        'conditional_expected_spikes_per_unit',
    ]


def read_rows(folder, report):
    """Validate complete registered cases, arms and horizons before aggregation."""
    folder = Path(folder)
    cfg = report['configuration']
    if (cfg['training_seeds'] != list(SEEDS) or cfg['branches'] != list(BRANCHES)
            or cfg['selection_blocks'] != [14, 15, 16, 17]
            or cfg['expected_cases_per_seed'] != 16 or cfg['expected_units'] != 100000
            or cfg['joint_samples'] != 16 or cfg['horizons_ms'] != [20., 80., 500.]
            or cfg['full_response_encoder_batch'] != 'joint_samples'
            or cfg['original_history_bins'] != 200 or cfg['original_horizon_bins'] != 1000
            or cfg['forecast_horizon_bins'] != 100 or cfg['bin_ms'] != 5.
            or cfg['endpoint_rtol'] != METRIC_RTOL or cfg['endpoint_atol'] != METRIC_ATOL
            or cfg['formal_evaluation'] is not False or cfg['update_weights'] is not False):
        raise ValueError('registered original-boundary development diagnostic required')
    if (report['status'] != 'DIAGNOSED_DEVELOPMENT'
            or report['model_type'] != 'lfads_boundary_diagnostic'
            or report['completed_cases'] != 16 or report['completed_branches'] != 64
            or report['score_rows'] != 192 or report['training_seed'] not in SEEDS
            or report['formal_test_accessed'] is not False
            or report['calibration_outcomes_accessed'] is not False
            or report['weights_updated'] is not False
            or report['scientific_claim_qualified'] is not False):
        raise ValueError('complete diagnostic with unchanged evidence scope required')
    expected_endpoints = {(f'train_block_{b:03d}_familiar_{t}', branch)
                          for b in range(14, 18) for t in range(4)
                          for branch in (BRANCHES[0], BRANCHES[3])}
    endpoints = report['endpoint_checks']
    if (len(endpoints) != len(expected_endpoints)
            or {(r['case_id'], r['branch']) for r in endpoints} != expected_endpoints):
        raise ValueError('all registered endpoint checks required')
    for check in endpoints:
        if (check['all_within_tolerance'] is not True
                or check['rtol'] != cfg['endpoint_rtol'] or check['atol'] != cfg['endpoint_atol']
                or set(check['arrays']) != {'conditional_expected_counts', 'boundary_generator_states', 'future_generator_states'}
                or any(r['within_tolerance'] is not True for r in check['arrays'].values())):
            raise ValueError('endpoint equivalence failed or tolerance changed')
        if check['branch'] == BRANCHES[0] and (check['sampled_counts_bitwise_equal'] is not True
                or any(r['bitwise_equal'] is not True for r in check['arrays'].values())):
            raise ValueError('causal endpoint requires exact public replay')
    if sha256_file(folder / 'scores.csv') != report['scores_sha256']:
        raise ValueError('diagnostic score table changed')
    expected = {(f'train_block_{b:03d}_familiar_{t}', branch, float(h))
                for b in range(14, 18) for t in range(4)
                for branch in BRANCHES for h in cfg['horizons_ms']}
    indexed = {}
    with (folder / 'scores.csv').open(encoding='utf-8', newline='') as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            key = (row['case_id'], row['branch'], float(row['horizon_ms']))
            block = int(row['block'])
            if (key not in expected or key in indexed or int(row['seed']) != report['training_seed']
                    or row['case_id'] != f"train_block_{block:03d}_{row['template_id']}"
                    or float(row['origin_ms']) != 1000.):
                raise ValueError('duplicate, unexpected or mismatched diagnostic row')
            expected_role = 'forecast' if row['branch'] == BRANCHES[0] else 'response_conditioned_diagnostic'
            if row['role'] != expected_role:
                raise ValueError('diagnostic arm role changed')
            name = row['ensemble_artifact']
            if (Path(name).name != name or name not in report['artifacts']
                    or name != f"{row['case_id']}_{row['branch']}.npz"):
                raise ValueError('branch ensemble identity changed')
            for metric in metric_names(cfg):
                value = row[metric]
                allowed_missing = metric.startswith('variogram_temporal_lag_') and int(metric.rsplit('_', 1)[1]) >= int(key[2] / cfg['bin_ms'])
                if value == '' and allowed_missing:
                    row[metric] = None
                else:
                    row[metric] = float(value)
                    if not np.isfinite(row[metric]):
                        raise ValueError('nonfinite diagnostic metric: ' + metric)
            row.update(seed=int(row['seed']), block=block, origin_ms=1000., horizon_ms=key[2])
            row['conditional_mean_bias'] = row['conditional_expected_spikes_per_unit'] - row['observed_spikes_per_unit']
            row['absolute_conditional_mean_error'] = abs(row['conditional_mean_bias'])
            indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('missing registered diagnostic cases, branches or horizons')
    return indexed


def crossed_contrasts(values):
    """Within-case changes; negative error contrasts indicate improvement."""
    if set(values) != set(BRANCHES):
        raise ValueError('all four registered branches required for a contrast')
    a, b, c, d = (values[key] for key in BRANCHES)
    if all(x is None for x in (a, b, c, d)):
        return {}
    if any(x is None or not np.isfinite(x) for x in (a, b, c, d)):
        raise ValueError('matched finite contrast availability required')
    return dict(boundary_at_prior=b-a, boundary_at_posterior=d-c,
                future_at_prefix=c-a, future_at_full_response=d-b,
                boundary_future_interaction=d-b-c+a)


def _mean(values):
    if all(x is None for x in values):
        return None
    if any(x is None or not np.isfinite(x) for x in values):
        raise ValueError('matched finite metric availability required')
    return float(np.mean(values))


def source_binding(report):
    """Retain the evaluated native identity and check the training-era subset."""
    trained = report['training_binding']
    result = {key: report[key] for key in
              ('acquisition_design_sha256', 'dataset_sha256', 'native_source_identity')}
    for key in ('acquisition_design_sha256', 'dataset_sha256'):
        if result[key] != trained[key]:
            raise ValueError('evaluated cohort differs from training binding')
    native = result['native_source_identity']
    if (any(native.get(key) != value for key, value in trained['native_source_identity'].items())
            or native.get('dt_ms') != trained['config']['dt_ms']):
        raise ValueError('evaluated native identity differs from training binding')
    return result


def analyze(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, block_rows, contrasts, cov_rows, costs = [], [], [], [], []
    case_costs, graph_costs = [], []
    cov_arrays, bindings = {}, {}
    common_config = common_source = common_scale = None
    for seed in SEEDS:
        folder = root / f'seed_{seed}'
        report = _read(folder / 'evaluation.json')
        replay_path = root / f'seed_{seed}_replay.json'
        replay = _read(replay_path)
        if (replay['status'] != 'PASS' or replay['training_seed'] != seed
                or replay['verified_cases'] != 16 or replay['verified_branches'] != 64
                or replay['verified_score_rows'] != 192
                or replay['evaluation_sha256'] != sha256_file(folder / 'evaluation.json')
                or replay['scores_sha256'] != report['scores_sha256']
                or replay['all_branch_unit_metrics_replayed'] is not True):
            raise ValueError('complete independent branch replay required before analysis')
        indexed = read_rows(folder, report)
        if report['training_seed'] != seed:
            raise ValueError('training seed differs from output identity')
        for name, digest in report['artifacts'].items():
            if Path(name).name != name or sha256_file(folder / name) != digest:
                raise ValueError('diagnostic artifact changed: ' + name)
        cfg = report['configuration']
        source = source_binding(report)
        with np.load(folder / 'fitting_field_scale.npz', allow_pickle=False) as data:
            scale = {name: data[name] for name in ('mean_hz', 'standard_deviation_hz', 'unit_count', 'field_ids')}
        if common_config is None:
            common_config, common_source, common_scale = cfg, source, scale
        if cfg != common_config or source != common_source or any(not np.array_equal(scale[k], common_scale[k]) for k in scale):
            raise ValueError('protocol, cohort or field scale differs across training seeds')
        metrics = metric_names(cfg) + ['conditional_mean_bias', 'absolute_conditional_mean_error']
        seed_rows = list(indexed.values())
        rows.extend(seed_rows)
        graph_costs.append(dict(seed=seed, graph_setup_seconds=report['graph_setup_seconds'],
                               diagnostic_elapsed_seconds=report['elapsed_seconds']))
        shared_cost_names = {'input_loading_seconds', 'prefix_encoding_and_latent_sampling_seconds',
                             'full_response_encoder_seconds', 'full_response_latent_sampling_seconds', 'graph_setup_seconds'}
        for case in report['case_bindings']:
            member = indexed[(case['case_id'], BRANCHES[0], 500.)]
            case_costs.append(dict(seed=seed, case_id=case['case_id'], input_loading_seconds=float(member['input_loading_seconds']),
                                   **case['preparation_costs']))
        ensembles = {}
        for row in seed_rows:
            name = row['ensemble_artifact']
            if name not in ensembles:
                with np.load(folder / name, allow_pickle=False) as data:
                    factor = scale['unit_count'] * cfg['bin_ms'] / 1000
                    ensembles[name] = ((data['sampled_field_counts']/factor-scale['mean_hz'])/scale['standard_deviation_hz'],
                                       (data['observed_field_counts']/factor-scale['mean_hz'])/scale['standard_deviation_hz'])
                costs.append({key: row[key] for key in row if key not in shared_cost_names and
                              (key.endswith('_seconds') or key.endswith('_bytes') or key in ('seed', 'case_id', 'branch'))})
        for block in range(14, 18):
            for ms in cfg['horizons_ms']:
                for branch in BRANCHES:
                    members = [r for r in seed_rows if (r['block'], r['branch'], r['horizon_ms']) == (block, branch, ms)]
                    block_rows.append(dict(seed=seed, block=block, branch=branch, horizon_ms=ms,
                                           **{m: _mean([r[m] for r in members]) for m in metrics}))
                for template in range(4):
                    case = f'train_block_{block:03d}_familiar_{template}'
                    for metric in metrics:
                        values = {b: indexed[(case, b, float(ms))][metric] for b in BRANCHES}
                        for contrast, value in crossed_contrasts(values).items():
                            contrasts.append(dict(seed=seed, block=block, case_id=case, horizon_ms=ms,
                                                  metric=metric, contrast=contrast, value=value))
        pairs = pair_sets(scale['field_ids'])
        for branch in BRANCHES:
            for template in range(4):
                for ms in cfg['horizons_ms']:
                    h = int(ms/cfg['bin_ms'])
                    members = [indexed[(f'train_block_{b:03d}_familiar_{template}', branch, float(ms))] for b in range(14, 18)]
                    sample = np.stack([ensembles[r['ensemble_artifact']][0][:, :h] for r in members])
                    truth = np.stack([ensembles[r['ensemble_artifact']][1][:h] for r in members])
                    requests = [(name, ij[:, 0], ij[:, 1], 0) for name, ij in pairs.items()]
                    requests += [(f'temporal_lag_{lag}', np.arange(sample.shape[-1]), np.arange(sample.shape[-1]), lag)
                                 for lag in cfg['temporal_lags_bins'] if lag < h]
                    for name, left, right, lag in requests:
                        observed, predicted, within, between = covariance_components(sample, truth, left, right, lag)
                        key = f'g{len(cov_rows)}'
                        cov_arrays[key] = np.stack((observed, predicted, within, between))
                        if not np.isfinite(cov_arrays[key]).all():
                            raise ValueError('nonfinite covariance diagnostic')
                        cov_rows.append(dict(seed=seed, branch=branch, template_id=f'familiar_{template}', horizon_ms=ms,
                                             statistic=name, independent_blocks=4, pairs=len(left), phases=h-lag, array_key=key,
                                             observed_covariance_mean=float(observed.mean()), predicted_covariance_mean=float(predicted.mean()),
                                             covariance_mse=float(((predicted-observed)**2).mean()),
                                             covariance_mae=float(np.abs(predicted-observed).mean())))
        bindings[str(seed)] = dict(evaluation_sha256=sha256_file(folder / 'evaluation.json'),
                                  replay_sha256=sha256_file(replay_path), selected_model_artifacts=report['selected_model_artifacts'])
    summary = []
    for branch in BRANCHES:
        for ms in common_config['horizons_ms']:
            for metric in metrics:
                values = [_mean([r[metric] for r in block_rows if (r['seed'], r['branch'], r['horizon_ms']) == (s, branch, ms)]) for s in SEEDS]
                value = _mean(values)
                if value is not None:
                    summary.append(dict(branch=branch, horizon_ms=ms, metric=metric, mean=value,
                                        minimum_seed_mean=min(values), maximum_seed_mean=max(values), training_seeds=3, independent_blocks=4))
    grouped = defaultdict(list)
    for r in contrasts:
        grouped[(r['seed'], r['block'], r['horizon_ms'], r['metric'], r['contrast'])].append(r['value'])
    contrast_blocks = [dict(seed=s, block=b, horizon_ms=h, metric=m, contrast=c, value=_mean(v)) for (s,b,h,m,c),v in sorted(grouped.items())]
    grouped = defaultdict(list)
    for r in contrast_blocks:
        grouped[(r['horizon_ms'], r['metric'], r['contrast'])].append(r)
    contrast_summary = []
    for (h,m,c), members in sorted(grouped.items()):
        values = [_mean([r['value'] for r in members if r['seed'] == s]) for s in SEEDS]
        contrast_summary.append(dict(horizon_ms=h, metric=m, contrast=c, mean=_mean(values),
                                     minimum_seed_mean=min(values), maximum_seed_mean=max(values), training_seeds=3, independent_blocks=4))
    grouped = defaultdict(list)
    for r in cov_rows:
        grouped[(r['branch'], r['horizon_ms'], r['statistic'])].append(r)
    cov_summary = [dict(branch=b, horizon_ms=h, statistic=stat, training_seeds=3, independent_blocks=4,
                        **{k: _mean([r[k] for r in members]) for k in
                           ('observed_covariance_mean', 'predicted_covariance_mean', 'covariance_mse', 'covariance_mae')})
                   for (b,h,stat), members in sorted(grouped.items())]
    tables = {'diagnostic_all_seeds.csv': rows, 'diagnostic_by_block.csv': block_rows, 'diagnostic_summary.csv': summary,
              'contrasts_by_case.csv': contrasts, 'contrasts_by_block.csv': contrast_blocks, 'contrast_summary.csv': contrast_summary,
              'covariance_diagnostic.csv': cov_rows, 'covariance_summary.csv': cov_summary, 'costs_by_branch.csv': costs,
              'shared_costs_by_case.csv': case_costs, 'setup_costs_by_seed.csv': graph_costs}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in tables.items():
        _write_rows(output / name, data)
    np.savez_compressed(output / 'covariance_arrays.npz', **cov_arrays)
    result = dict(status='DESCRIBED_REPLAYED_DEVELOPMENT', training_seeds=list(SEEDS), seed_case_pairs=48, branches=192,
                  score_rows=len(rows), configuration=common_config, source_binding=common_source, model_seed_bindings=bindings,
                  independent_blocks=4, connectivity_instances=1, scientific_claim_qualified=False, formal_test_accessed=False,
                  interpretation='selection-block model diagnostics; no inferential CI/p/q or biological causal claim',
                  covariance_scope='total predictive covariance across four initial/background selection blocks at fixed template and phase',
                  costs='shared preparation and endpoint validation are recorded separately; branch costs are not a solver speedup comparison',
                  implementation_sha256=sha256_file(Path(__file__)),
                  artifacts={name: sha256_file(output/name) for name in [*tables, 'covariance_arrays.npz']})
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.diagnostics, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'seed_case_pairs', 'branches', 'score_rows')}))


if __name__ == '__main__':
    main()

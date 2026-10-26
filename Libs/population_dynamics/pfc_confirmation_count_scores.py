"""Recover PFC paired block CDF scores over the complete physical domain."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from .pfc_confirmation_domain import NEW_STATUS
from .provenance import freeze_json, sha256_file

# Float64 subtraction/averaging order roundtrip, not a scientific error margin.
ROUNDTRIP_ABSOLUTE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recover_scores(domains: list[dict], data: dict, labels: list[str], seeds: list[int]):
    support = np.asarray(data['count_support'])
    require(support.ndim == 1 and len(support) > 0 and support.dtype.kind in 'iu'
            and np.all(support >= 0) and np.all(np.diff(support) > 0), 'SORTED_INTEGER_OBSERVED_SUPPORT_REQUIRED')
    require(len(seeds) >= 2 and seeds == sorted(set(seeds))
            and np.array_equal(data['source_rng_seeds'], seeds), 'INDEPENDENT_BLOCK_IDENTITY_MISMATCH')
    condition_names = data['condition_names'].tolist()
    phase_names = data['phase_names'].tolist()
    windows = data['window_ms'].tolist()
    sizes = data['field_unit_counts']
    for axis in (condition_names, phase_names, windows):
        require(len(axis) > 0 and len(axis) == len(set(axis)), 'DUPLICATE_OR_EMPTY_CDF_AXIS')
    require(np.asarray(sizes).dtype.kind in 'iu' and np.all(sizes > 0), 'POSITIVE_FIELD_SIZES_REQUIRED')
    shape = (len(condition_names), len(seeds), len(phase_names), len(windows), len(sizes), len(support))
    cdfs = {}
    for label in labels:
        x = np.asarray(data[label+'_cdf'])
        require(x.shape == shape and x.dtype.kind == 'f' and np.isfinite(x).all()
                and np.all((x >= 0) & (x <= 1)) and np.all(np.diff(x, axis=-1) >= 0),
                'INVALID_COMPLETE_BLOCK_CDF:'+label)
        require(np.all(x[..., -1] == 1.), 'CDF_TERMINAL_MASS_NOT_COMPLETE:'+label)
        cdfs[label] = x
    ci = {name: i for i, name in enumerate(condition_names)}
    pi = {name: i for i, name in enumerate(phase_names)}
    wi = {float(width): i for i, width in enumerate(windows)}
    pairs = set(zip(labels[:-1], labels[1:]))
    output = [dict(row) for row in domains]
    indices, scores = [], []
    maximum_roundtrip_error = 0.
    recovered = 0
    identity_keys = set()
    for row_index, row in enumerate(output):
        if row['family'] != 'count_cdf':
            continue
        ident = json.loads(row['endpoint_identity'])
        canonical = json.dumps(ident, sort_keys=True)
        require(canonical not in identity_keys, 'DUPLICATE_CDF_ENDPOINT_IDENTITY')
        identity_keys.add(canonical)
        left, right = ident['left_cohort'], ident['right_cohort']
        require((left, right) in pairs, 'ADJACENT_FROZEN_CLOCK_PAIR_REQUIRED')
        field, threshold = int(ident['field_index']), int(ident['count_threshold'])
        require(0 <= field < len(sizes) and threshold >= 0, 'INVALID_PHYSICAL_CDF_IDENTITY')
        key = (ci[ident['condition']], slice(None), pi[ident['phase']], wi[float(ident['window_ms'])], field)
        support_index = int(np.searchsorted(support, threshold, side='right'))-1
        a = np.zeros(len(seeds)) if support_index < 0 else cdfs[left][key+(support_index,)]
        b = np.zeros(len(seeds)) if support_index < 0 else cdfs[right][key+(support_index,)]
        if threshold >= int(row['physical_maximum_count']):
            require(np.all(a == 1.) and np.all(b == 1.), 'SOURCE_REFRACTORY_CDF_BOUND_CONTRADICTED')
        delta = a-b
        estimate = float(delta.mean())
        if row['definition_status'] == NEW_STATUS:
            require(row['estimate'] == '', 'UNESTIMATED_DOMAIN_ROW_HAS_NUMERICAL_VALUE')
            constant = bool(np.all(delta == delta[0]))
            row.update(estimate=str(estimate), empirical_status='RECOVERED_FROM_COMPLETE_BLOCK_CDFS',
                       definition_status='EMPIRICALLY_DEGENERATE_UNCERTAIN' if constant else 'ESTIMATED_EMPIRICAL_SPREAD',
                       reason='right_continuous_empirical_cdf_recovery; no_population_uncertainty_claim')
            recovered += 1
        else:
            error = abs(estimate-float(row['estimate']))
            maximum_roundtrip_error = max(maximum_roundtrip_error, error)
            require(error <= ROUNDTRIP_ABSOLUTE_TOLERANCE, 'ARCHIVED_CDF_POINT_ESTIMATE_MISMATCH')
        indices.append(row_index);scores.append(delta)
    require(len(scores) > 0, 'CDF_ENDPOINT_DOMAIN_REQUIRED')
    matrix = np.stack(scores)
    require(matrix.shape == (len(indices), len(seeds)) and np.isfinite(matrix).all()
            and np.all(np.abs(matrix) <= 1), 'BOUNDED_PAIRED_BLOCK_SCORES_REQUIRED')
    checks = {'cdf_endpoint_rows': len(indices), 'independent_joint_rng_blocks': len(seeds),
              'recovered_previously_unestimated_thresholds': recovered,
              'source_count_support': support.tolist(),
              'terminal_cdf_cells_verified': len(labels)*int(np.prod(shape[:-1])),
              'maximum_archived_point_estimate_roundtrip_error': maximum_roundtrip_error,
              'definition_status_counts': dict(Counter(row['definition_status'] for row in output)),
              'newly_estimated_thresholds_are_population_structural_zeros': False}
    return output, matrix, np.asarray(indices, dtype=np.int64), checks


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def bound_report(root, expected_hash):
    path = root/'report.json'
    require(sha256_file(path) == expected_hash, 'SOURCE_REPORT_BINDING_CHANGED:'+str(path))
    return json.loads(path.read_text())


def run(config_path: Path, project_root: Path, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    require(config['schema'] == 'pfc_confirmation_count_scores_v1'
            and config['scope'] == 'development_calibration'
            and config['independent_unit'] == 'joint_initial_background_rng_block'
            and config['confirmation_execution_released'] is False, 'CALIBRATION_ONLY_SCORE_RECOVERY_REQUIRED')
    comparison_root, domain_root = (project_root/config[key] for key in ('comparison_archive','domain_archive'))
    comparison = bound_report(comparison_root, config['comparison_report_sha256'])
    domain = bound_report(domain_root, config['domain_report_sha256'])
    require(comparison['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
            and comparison['count_cdf_comparison']['status'] == 'COMPLETE_DESCRIPTIVE_COUNT_CDF_COMPARISON'
            and domain['status'] == 'COMPLETE_PHYSICAL_ENDPOINT_DOMAIN_ESTIMATES_PARTIAL', 'COMPLETE_SOURCE_COMPARISON_AND_DOMAIN_REQUIRED')
    cdf_path, domain_path = comparison_root/'block_count_cdfs.npz', domain_root/'endpoint_domains.csv'
    require(sha256_file(cdf_path) == comparison['outputs'][cdf_path.name]['sha256'], 'BLOCK_CDF_PAYLOAD_BINDING_CHANGED')
    require(sha256_file(domain_path) == domain['output']['sha256'], 'PHYSICAL_DOMAIN_BINDING_CHANGED')
    with np.load(cdf_path, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    labels, seeds = config['cohorts'], config['joint_rng_seeds']
    for label in labels:
        source = comparison['inputs'][label]
        require(source['coupled_units'] == 100000 and source['source_rng_blocks'] == seeds
                and np.array_equal(data['field_unit_counts'], source['field_unit_counts']),
                'CALIBRATION_COHORT_OR_FIELD_IDENTITY_CHANGED')
    source_checks = comparison['count_cdf_comparison']
    require(np.array_equal(data['count_support'], source_checks['count_support'])
            and seeds == source_checks['independent_joint_blocks'], 'SOURCE_CDF_SUPPORT_OR_BLOCKS_CHANGED')
    domains = read_rows(domain_path)
    require(len(domains) == domain['complete_endpoint_rows'], 'SOURCE_DOMAIN_ROW_COUNT_CHANGED')
    rows, matrix, indices, checks = recover_scores(domains, data, labels, seeds)
    require(checks['terminal_cdf_cells_verified'] == source_checks['unit_window_mass_checks']
            == source_checks['independent_field_spike_total_checks'], 'COMPLETE_PRIOR_MASS_VALIDATION_REQUIRED')
    require(checks['recovered_previously_unestimated_thresholds'] == domain['new_nontrivial_cdf_thresholds'],
            'PHYSICAL_THRESHOLD_RECOVERY_INCOMPLETE')
    output.mkdir(parents=True, exist_ok=False)
    table = output/'endpoint_domains.csv'
    with table.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scores_path = output/'paired_block_count_cdf_scores.npz'
    np.savez_compressed(scores_path, paired_cdf_differences=matrix, domain_row_indices=indices,
                        source_rng_seeds=np.asarray(seeds,dtype=np.int64))
    report = dict(status='COMPLETE_CALIBRATION_BLOCK_CDF_SCORES', python_version=platform.python_version(), numpy_version=np.__version__, **checks,
        domain_rows=len(rows), settings=config, settings_sha256=sha256_file(config_path),
        independent_unit=config['independent_unit'], matrix_axes=['cdf_endpoint','joint_initial_background_rng_block'],
        domain_row_indices_semantics='zero-based data row in endpoint_domains.csv, header excluded',
        reused_source_calibration_episodes=domain['source_calibration_episodes'],
        new_source_simulations=0, raw_spike_data_reread=False, confidence_intervals=False,
        numerical_margins_frozen=False, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, production_episodes=0,
        source_mass_validation_reused=dict(report_sha256=config['comparison_report_sha256'],
            unit_window_mass_checks=source_checks['unit_window_mass_checks'],
            independent_field_spike_total_checks=source_checks['independent_field_spike_total_checks']),
        interpretation='Observed complete block CDFs and paired scores only; zero observed tail mass or score spread is not zero population uncertainty.',
        outputs={p.name:{'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in (table,scores_path)},
        implementation_sha256=sha256_file(Path(__file__)))
    freeze_json(output/'report.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.config,args.project_root,args.output)
    print(json.dumps({key:report[key] for key in ('status','cdf_endpoint_rows','independent_joint_rng_blocks','recovered_previously_unestimated_thresholds')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

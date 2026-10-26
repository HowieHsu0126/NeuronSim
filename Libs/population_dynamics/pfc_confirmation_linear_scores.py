"""Recover paired block scores for PFC field means and source-native responses."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_confirmation_covariance_scores import field_cohort
from .pfc_delay_analysis import PHASES
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS
from .provenance import freeze_json, sha256_file

FAMILIES = ('field_mean', 'source_response', 'paired_response')
STRUCTURAL = 'STRUCTURAL_ZERO_CONTRAST'
EMPIRICAL = 'EMPIRICALLY_DEGENERATE_UNCERTAIN'


def source_metric_bounds(rate_bounds: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use per-unit refractory support, including conservative vector bounds.

    These are physical ranges, not tolerances. Source E fields precede one I
    field. Every phase consists of whole readout bins, so the same per-unit
    ceiling bounds any nonempty cue/noncue E subset and each vector component.
    """
    b, n = np.asarray(rate_bounds, dtype=float), np.asarray(sizes)
    require(b.shape == n.shape and b.ndim == 1 and len(b) >= 2
            and np.isfinite(b).all() and np.all(b > 0)
            and n.dtype.kind in 'iu' and np.all(n > 0)
            and np.all(b[:-1] == b[0]), 'SOURCE_E_THEN_I_PHYSICAL_BOUNDS_REQUIRED')
    e, inh = b[0], b[-1]
    ranges = {
        'all_unit_rate_hz': (0., float(np.dot(b, n)/n.sum())),
        'excitatory_rate_hz': (0., e), 'inhibitory_rate_hz': (0., inh),
        'cue_region_rate_hz': (0., e), 'noncue_region_rate_hz': (0., e),
        'cue_selectivity_hz': (-e, e), 'vector_x_hz': (-e, e), 'vector_y_hz': (-e, e),
    }
    return tuple(np.asarray([ranges[name][i] for name in METRICS]) for i in (0, 1))


def validate_responses(data: dict, labels: list[str], seeds: list[int], bounds: tuple) -> dict:
    for name, expected in [('condition_names', CONDITIONS), ('phase_names', tuple(PHASES)),
                           ('metric_names', METRICS), ('source_rng_seeds', tuple(seeds))]:
        require(np.array_equal(data[name], expected), 'SOURCE_RESPONSE_AXIS_CHANGED:'+name)
    require(len(seeds) >= 2 and seeds == sorted(set(seeds)), 'UNIQUE_ORDERED_BLOCK_IDENTITIES_REQUIRED')
    lo, hi = bounds
    result = {}
    for label in labels:
        values = np.asarray(data[label+'_values'])
        require(values.shape == (len(CONDITIONS), len(seeds), len(PHASES), len(METRICS))
                and values.dtype.kind == 'f' and np.isfinite(values).all(),
                'COMPLETE_FINITE_SOURCE_RESPONSES_REQUIRED:'+label)
        require(np.all(values >= lo) and np.all(values <= hi), 'SOURCE_RESPONSE_OUTSIDE_PHYSICAL_RANGE')
        require(np.array_equal(values[0, :, 0], values[1, :, 0])
                and np.array_equal(values[0, :, 0], values[2, :, 0]), 'SOURCE_PRE_CUE_IDENTITY_FAILED')
        # Preserve true-unit angle projections; never substitute field centers.
        result[label] = values
    return result


def linear_block_score(family: str, identity: dict, field_means: dict, responses: dict) -> np.ndarray:
    left, right, phase = identity['left_cohort'], identity['right_cohort'], identity['phase']
    pi = list(PHASES).index(phase)
    if family == 'field_mean':
        field = int(identity['field_index'])
        return field_means[left][identity['condition'], phase][:, field] - field_means[right][identity['condition'], phase][:, field]
    metric = METRICS.index(identity['metric'])
    a, b = responses[left], responses[right]
    if family == 'source_response':
        condition = CONDITIONS.index(identity['condition'])
        return a[condition, :, pi, metric] - b[condition, :, pi, metric]
    require(family == 'paired_response', 'KNOWN_LINEAR_FAMILY_REQUIRED')
    contrast = {name: (CONDITIONS.index(positive), CONDITIONS.index(negative))
                for name, positive, negative in CONTRASTS}
    pos, neg = contrast[identity['contrast']]
    return (a[pos, :, pi, metric]-a[neg, :, pi, metric]) - (b[pos, :, pi, metric]-b[neg, :, pi, metric])


def recover_linears(domains, cohorts, data, labels, seeds, bin_ms):
    sizes = cohorts[labels[0]]['sizes']
    fields = len(sizes)
    counts = {}
    for row in domains:
        if row['family'] != 'count_cdf':
            continue
        identity = json.loads(row['endpoint_identity'])
        if float(identity['window_ms']) != bin_ms:
            continue
        field, maximum = int(identity['field_index']), int(row['physical_maximum_count'])
        require(field not in counts or counts[field] == maximum, 'INCONSISTENT_PER_UNIT_REFRACTORY_SUPPORT')
        counts[field] = maximum
    require(set(counts) == set(range(fields)), 'COMPLETE_PER_UNIT_REFRACTORY_SUPPORT_REQUIRED')
    rate_bounds = np.asarray([counts[i]/(bin_ms/1000) for i in range(fields)])
    bounds = source_metric_bounds(rate_bounds, sizes)
    responses = validate_responses(data, labels, seeds, bounds)
    means = {}
    for label in labels:
        cohort = cohorts[label]
        require(np.array_equal(cohort['sizes'], sizes)
                and int(sizes[:-1].sum()) == cohort['config']['excitatory_neurons']
                and int(sizes[-1]) == cohort['config']['inhibitory_neurons'], 'SOURCE_POPULATION_FIELD_MAP_CHANGED')
        means[label] = {}
        for condition in CONDITIONS:
            rates = cohort['rates'][condition]
            require(rates.shape == (len(seeds), 1000, fields) and np.isfinite(rates).all()
                    and np.all((rates >= 0) & (rates <= rate_bounds)), 'INVALID_COMPLETE_FIELD_RATES')
            for phase, (start, stop) in PHASES.items():
                first, last = round(start/bin_ms), round(stop/bin_ms)
                require(first*bin_ms == start and last*bin_ms == stop, 'PHASE_MUST_CONTAIN_WHOLE_READOUT_BINS')
                means[label][condition, phase] = rates[:, first:last].mean(axis=1)
    pairs = list(zip(labels[:-1], labels[1:]))
    expected = set()
    for left, right in pairs:
        for condition, phase, field in itertools.product(CONDITIONS, PHASES, range(fields)):
            expected.add(('field_mean', left, right, condition, phase, str(field)))
        for family, names in [('source_response', CONDITIONS), ('paired_response', [x[0] for x in CONTRASTS])]:
            for condition, phase, metric in itertools.product(names, PHASES, METRICS):
                expected.add((family, left, right, condition, phase, metric))
    seen, indices, scores, lower, upper, statuses, families = set(), [], [], [], [], [], []
    maximum_error, maximum_allowance = 0., 0.
    for index, row in enumerate(domains):
        family = row['family']
        if family not in FAMILIES:
            continue
        ident = json.loads(row['endpoint_identity'])
        key = (family, ident['left_cohort'], ident['right_cohort'],
               ident['contrast' if family == 'paired_response' else 'condition'], ident['phase'],
               ident['field_index' if family == 'field_mean' else 'metric'])
        require(key in expected and key not in seen, 'UNIQUE_FROZEN_LINEAR_ENDPOINT_REQUIRED')
        seen.add(key)
        delta = linear_block_score(family, ident, means, responses)
        if family == 'field_mean':
            radius = rate_bounds[int(ident['field_index'])]
        else:
            mi = METRICS.index(ident['metric'])
            radius = (bounds[1][mi]-bounds[0][mi]) * (2 if family == 'paired_response' else 1)
        lo, hi = -radius, radius
        structural = family == 'paired_response' and ident['phase'] == 'baseline'
        require((row['definition_status'] == STRUCTURAL) == structural, 'STRUCTURAL_STATUS_MUST_FOLLOW_SOURCE_IDENTITY')
        if structural:
            require(row['reason'] == 'identical_pre_cue_inputs_and_initial_state' and np.all(delta == 0),
                    'PRE_CUE_STRUCTURAL_ZERO_CONTRADICTED')
            lo = hi = 0.
        elif row['definition_status'] == EMPIRICAL:
            require(np.all(delta == delta[0]), 'EMPIRICAL_DEGENERACY_CHANGED')
        else:
            require(row['definition_status'] == 'ESTIMATED_EMPIRICAL_SPREAD', 'KNOWN_LINEAR_DEFINITION_STATUS_REQUIRED')
        require(delta.shape == (len(seeds),) and np.isfinite(delta).all()
                and np.all(delta >= lo) and np.all(delta <= hi), 'LINEAR_SCORE_OUTSIDE_PHYSICAL_RANGE')
        error = abs(float(delta.mean())-float(row['estimate']))
        # Operation-count/range allowance only; never used as a scientific margin.
        allowance = 64*np.finfo(float).eps*1000*max(1., radius)
        require(error <= allowance, 'LINEAR_POINT_ROUNDTRIP_FAILED:'+str(key))
        maximum_error, maximum_allowance = max(maximum_error, error), max(maximum_allowance, allowance)
        indices.append(index); scores.append(delta); lower.append(lo); upper.append(hi)
        statuses.append(row['definition_status']); families.append(family)
    require(seen == expected, 'COMPLETE_LINEAR_ENDPOINT_DOMAIN_REQUIRED')
    matrix = np.asarray(scores)
    return dict(paired_linear_differences=matrix, domain_row_indices=np.asarray(indices, dtype=np.int64),
                source_rng_seeds=np.asarray(seeds, dtype=np.int64), physical_lower_bound=np.asarray(lower),
                physical_upper_bound=np.asarray(upper), point_estimates=matrix.mean(axis=1)), dict(
        linear_endpoints=len(indices), family_counts=dict(Counter(families)), definition_counts=dict(Counter(statuses)),
        original_independent_blocks=len(seeds), field_rate_upper_bounds_hz=rate_bounds.tolist(),
        per_clock_response_physical_ranges_hz={name: [float(bounds[0][i]), float(bounds[1][i])] for i, name in enumerate(METRICS)},
        maximum_archived_point_roundtrip_error=maximum_error, maximum_arithmetic_roundtrip_allowance=maximum_allowance,
        roundtrip_allowance_is_scientific_margin=False)


def run(config_path: Path, project_root: Path, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    require(config['schema'] == 'pfc_confirmation_linear_scores_v1' and config['scope'] == 'development_calibration'
            and config['independent_unit'] == 'joint_initial_background_rng_block'
            and config['confirmation_execution_released'] is False, 'CALIBRATION_ONLY_LINEAR_RECOVERY_REQUIRED')
    comparison_root, domain_root = (project_root/config[key] for key in ('comparison_archive', 'domain_archive'))
    comparison = bound_report(comparison_root, config['comparison_report_sha256'])
    domain = bound_report(domain_root, config['domain_report_sha256'])
    require(comparison['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
            and domain['status'] == 'COMPLETE_CALIBRATION_BLOCK_CDF_SCORES', 'COMPLETE_BOUND_INPUTS_REQUIRED')
    table, response_path = domain_root/'endpoint_domains.csv', comparison_root/'block_source_responses.npz'
    for path, report in [(table, domain), (response_path, comparison)]:
        require(sha256_file(path) == report['outputs'][path.name]['sha256'], 'BOUND_SCORE_INPUT_CHANGED:'+path.name)
    domains = read_rows(table)
    require(len(domains) == domain['domain_rows'], 'DOMAIN_ROW_COUNT_CHANGED')
    seeds, labels = config['joint_rng_seeds'], list(config['field_archives'])
    info = comparison['source_phase_responses']
    require(info['status'] == 'COMPLETE_SOURCE_PHASE_RESPONSE_COMPARISON' and info['joint_rng_blocks'] == seeds
            and info['metrics'] == list(METRICS) and info['contrasts'] == [x[0] for x in CONTRASTS]
            and all(info['input_checks'][label]['source_prefix_and_rng_pairs_verified'] == len(CONDITIONS)*len(seeds)
                    for label in labels), 'VERIFIED_SOURCE_RESPONSE_AND_PRE_CUE_PAIRING_REQUIRED')
    require(all(comparison['inputs'][label]['source_rng_blocks'] == seeds for label in labels)
            and all(comparison['inputs'][a]['dt_ms'] > comparison['inputs'][b]['dt_ms'] for a,b in zip(labels[:-1],labels[1:])),
            'ORDERED_CALIBRATION_CLOCKS_AND_BLOCKS_REQUIRED')
    bin_ms = comparison['settings']['bin_ms']
    cohorts = {label: field_cohort(project_root/path, comparison['inputs'][label], seeds, bin_ms)
               for label,path in config['field_archives'].items()}
    with np.load(response_path, allow_pickle=False) as stored:
        data = {name: stored[name] for name in ['condition_names','phase_names','metric_names','source_rng_seeds']+[label+'_values' for label in labels]}
    payload, checks = recover_linears(domains, cohorts, data, labels, seeds, bin_ms)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output/'paired_block_linear_scores.npz'
    np.savez_compressed(artifact, **payload)
    report = dict(status='COMPLETE_CALIBRATION_LINEAR_BLOCK_SCORES', **checks,
        settings=config, settings_sha256=sha256_file(config_path), python_version=platform.python_version(), numpy_version=np.__version__,
        domain_rows=len(domains), domain_table_sha256=sha256_file(table), source_response_sha256=sha256_file(response_path),
        source_field_sha256={label: cohort['source_field_sha256'] for label,cohort in cohorts.items()},
        domain_row_indices_semantics='zero-based row of bound source endpoint_domains.csv; header excluded',
        matrix_axes=['linear_endpoint','joint_initial_background_rng_block'], units='Hz',
        independent_unit=config['independent_unit'], source_episodes_reused=3*len(labels)*len(seeds),
        interpretation='Source-clock contrasts and differences of paired source-control responses. Same block labels are not a common physical noise path across clocks. Source-angle vector components retain all blocks, including undefined directions.',
        bounds_interpretation='Source per-unit refractory support; physical ranges are not scientific margins. Empirically constant scores retain nonzero physical uncertainty ranges.',
        new_source_simulations=0, raw_spike_data_reread=False, confidence_intervals=False, numerical_margins_frozen=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False, production_episodes=0,
        outputs={artifact.name: dict(bytes=artifact.stat().st_size, sha256=sha256_file(artifact))},
        implementation_sha256=sha256_file(Path(__file__)))
    freeze_json(output/'report.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.project_root, args.output)
    print(json.dumps({key:result[key] for key in ('status','linear_endpoints','family_counts','original_independent_blocks')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

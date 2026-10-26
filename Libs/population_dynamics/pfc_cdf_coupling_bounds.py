"""Evaluate whole-block CDF coupling envelopes as an approximate diagnostic.

For each complete CDF group, K_b=max_k|D_b(k)| gives
|E D_b(k)| <= E|D_b(k)| <= E K_b simultaneously over every threshold.
Only the expectation bound is structural; its bootstrap estimate remains an
unqualified calibration approximation. Empirically constant K adds no channel.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_confirmation_uncertainty import (
    EMPIRICAL, IDENTITY, STRUCTURAL, UNDEFINED, bootstrap_primitives,
    construct_bands, identity_key, load_npz, write_rows,
)
from .pfc_coupled_uncertainty import LABELS, SEEDS, precision_lookup
from .pfc_delay_analysis import PHASES
from .pfc_numerical_error_budget import assess_planning
from .pfc_phase_responses import CONDITIONS, METRICS
from .provenance import freeze_json, sha256_file

SCHEMA = 'pfc_cdf_coupling_bounds_diagnostic_v1'
ROUNDING_FACTOR = 64
GROUP_KEYS = ('left_cohort', 'right_cohort', 'condition', 'phase', 'window_ms', 'field_index')
INPUT_FILES = {
    'coupled_scores': ['coupled_block_scores.npz'], 'domain': ['endpoint_domains.csv'],
    'uncertainty': ['endpoint_sampling_precision.csv', 'primitive_bands.csv', 'endpoint_bands.csv', 'joint_bootstrap_maximum.npz'],
    'budget': ['frozen_numerical_margins.csv'], 'denominator_source': [], 'domain_definition': [],
}


def same_number(a, b):
    if a == '' or b == '': return a == b
    a, b = float(a), float(b)
    return np.isfinite(a) and np.isfinite(b) and abs(a-b) <= ROUNDING_FACTOR*np.finfo(float).eps*max(1., abs(a), abs(b))


def cdf_group(identity):
    return tuple(float(identity[key]) if key == 'window_ms' else int(identity[key]) if key == 'field_index'
                 else identity[key] for key in GROUP_KEYS)


def integer_cdf_coupling(matrix, denominator):
    """Recover unique integer count differences under known float64 operations.

    Two bounded integer/denominator divisions, one subtraction and one scaling
    need at most8*epsilon*denominator absolute error. Requiring this below1/2
    makes nearest-integer recovery unique; this is not an empirical SD cutoff.
    """
    require(type(denominator) is int and 0 < denominator <= 2**52, 'EXACT_CDF_DENOMINATOR_REQUIRED')
    values = np.asarray(matrix, dtype=float)
    require(values.ndim == 2 and np.isfinite(values).all() and np.all(np.abs(values) <= 1), 'BOUNDED_CDF_DIFFERENCES_REQUIRED')
    scaled = values*denominator; nearest = np.rint(scaled)
    allowance = 8*np.finfo(float).eps*denominator
    error = float(np.max(np.abs(scaled-nearest)))
    require(allowance < .5 and error <= allowance, 'CDF_DIFFERENCE_NOT_ON_SOURCE_COUNT_LATTICE')
    numerator = np.max(np.abs(nearest.astype(np.int64)), axis=0)
    k = numerator/denominator
    return k, numerator, dict(maximum_integer_roundtrip_error=error, integer_roundtrip_allowance=allowance,
                             maximum_probability_roundtrip_error=float(np.max(np.abs(values-nearest/denominator))),
                             raw_float_ks_varied_but_integer_ks_constant=bool(np.any(np.max(np.abs(values),axis=0) != np.max(np.abs(values),axis=0)[0])
                                                                           and np.all(numerator == numerator[0])))


def complete_cdf_group_scores(domains, scores, current_bands, field_sizes, terminal_boundary_identity_proven=False):
    """Use every physical integer threshold and each original block exactly once."""
    require(len(domains) == len(scores) == len(current_bands) and scores.ndim == 2 and scores.shape[1] >= 2,
            'COMPLETE_ENDPOINT_BY_ORIGINAL_BLOCK_MATRIX_REQUIRED')
    groups = {}; row_to_group = np.full(len(domains), -1, dtype=np.int64)
    for index, (row, band) in enumerate(zip(domains, current_bands)):
        require(int(band['domain_row_index']) == index and all(row[key] == band[key] for key in ('family', 'endpoint_identity', 'definition_status')),
                'ORDERED_FROZEN_CDF_ENDPOINT_BINDING_CHANGED')
        if row['family'] != 'count_cdf': continue
        ident = json.loads(row['endpoint_identity']); key = cdf_group(ident)
        threshold = int(ident['count_threshold']); physical = int(row['physical_maximum_count'])
        require(float(ident['count_threshold']) == threshold and threshold >= 0 and physical >= 0,
                'INTEGER_PHYSICAL_CDF_DOMAIN_REQUIRED')
        group = groups.setdefault(key, dict(physical=physical, rows={}))
        require(group['physical'] == physical and threshold not in group['rows'], 'UNIQUE_CDF_THRESHOLD_AND_PHYSICAL_SUPPORT_REQUIRED')
        x = scores[index]
        require(np.isfinite(x).all() and np.all((x >= -1) & (x <= 1)), 'FINITE_PAIRED_CDF_DIFFERENCES_IN_UNIT_RANGE_REQUIRED')
        require(same_number(x.mean(), band['estimate']), 'CURRENT_CDF_MEAN_AND_BOUND_ENDPOINT_DIFFER')
        require((row['definition_status'] == STRUCTURAL) == (threshold >= physical), 'CDF_STRUCTURAL_PHYSICAL_SUPPORT_CHANGED')
        if threshold >= physical: require(np.all(x == 0), 'PHYSICAL_SATURATION_DIFFERENCE_MUST_BE_ZERO')
        group['rows'][threshold] = index
    identities, values, physicals, denominators, numerators, roundtrips = [], [], [], [], [], []
    for group_index, (key, group) in enumerate(groups.items()):
        require(set(range(group['physical'])).issubset(group['rows']), 'MISSING_NONTRIVIAL_PHYSICAL_INTEGER_CDF_THRESHOLD')
        terminal_missing = group['physical'] not in group['rows']
        require(not terminal_missing or terminal_boundary_identity_proven, 'UNPROVEN_MISSING_TERMINAL_CDF_BOUNDARY')
        indices = [index for threshold, index in sorted(group['rows'].items())]
        matrix = scores[indices]
        if terminal_missing:
            # The bound domain rule proves both CDFs equal1 atM; no stored endpoint is changed.
            matrix = np.vstack((matrix, np.zeros((1, scores.shape[1]))))
        start, stop = PHASES[key[3]]; width, field = key[4], key[5]
        windows = round((stop-start)/width)
        require(width > 0 and windows > 0 and windows*width == stop-start and 0 <= field < len(field_sizes), 'WHOLE_SOURCE_PHASE_WINDOWS_REQUIRED')
        denominator = int(field_sizes[field])*windows
        k, numerator, roundtrip = integer_cdf_coupling(matrix, denominator)
        roundtrip['source_identity_terminal_boundary_added'] = terminal_missing
        denominators.append(denominator); numerators.append(numerator); roundtrips.append(roundtrip)
        mean_k = float(k.mean())
        require(np.all(np.abs(matrix.mean(1)) <= mean_k+ROUNDING_FACTOR*np.finfo(float).eps),
                'EMPIRICAL_CDF_COUPLING_INEQUALITY_CONTRADICTED')
        identities.append(dict(zip(GROUP_KEYS, key))); values.append(k); physicals.append(group['physical'])
        row_to_group[indices] = group_index
    require(bool(values), 'NONEMPTY_COMPLETE_CDF_GROUP_DOMAIN_REQUIRED')
    return (np.asarray(values), identities, np.asarray(physicals, dtype=np.int64), row_to_group,
            np.asarray(denominators,dtype=np.int64), np.asarray(numerators,dtype=np.int64), roundtrips)


def restore_primitives(rows):
    lookup, support = {}, set()
    for index, row in enumerate(rows):
        ident = json.loads(row['primitive_identity'])
        if ident['family'] == 'primitive_cdf':
            threshold = int(ident['count_threshold']); support.add(threshold)
            key = ('cdf', ident['cohort'], ident['condition'], ident['phase'], float(ident['window_ms']), int(ident['field_index']), threshold)
        else:
            require(ident['family'] == 'primitive_response', 'KNOWN_SAVED_PRIMITIVE_FAMILY_REQUIRED')
            key = ('response', ident['cohort'], ident['condition'], ident['phase'], ident['metric'])
        require(key not in lookup, 'DUPLICATE_SAVED_PRIMITIVE_IDENTITY'); lookup[key] = index
    arrays = tuple(np.asarray([float(row[key]) for row in rows])
                   for key in ('estimate', 'bootstrap_standard_error', 'physical_lower', 'physical_upper'))
    point, se, lo, hi = arrays
    require(all(np.isfinite(x).all() for x in arrays) and np.all(se >= 0) and np.all(lo <= point) and np.all(point <= hi),
            'FINITE_SAVED_PRIMITIVE_ESTIMATES_AND_PHYSICAL_SUPPORT_REQUIRED')
    for index, row in enumerate(rows):
        require((row['calibration_constant_tail_candidate'] == 'True') == bool(se[index] == 0 and lo[index] < hi[index]),
                'SAVED_CONSTANT_PRIMITIVE_ROUTE_CHANGED')
    return arrays, lookup, sorted(support)


def compare_reconstructed_bands(saved, actual, projected):
    require(len(saved) == len(actual) == len(projected), 'COMPLETE_LEGACY_BAND_RECONSTRUCTION_REQUIRED')
    maximum_error = 0.; compared = 0
    for index, (reference, row, future) in enumerate(zip(saved, actual, projected)):
        require(int(reference['domain_row_index']) == row['domain_row_index'] == future['domain_row_index'] == index
                and all(reference[key] == row[key] == future[key] for key in ('family', 'endpoint_identity', 'definition_status', 'channel')),
                'LEGACY_BAND_IDENTITY_OR_CHANNEL_RECONSTRUCTION_FAILED')
        for key, observed in [('estimate', row['estimate']), ('lower', row['lower']), ('upper', row['upper']),
                              ('direct_bootstrap_standard_error', row['direct_bootstrap_standard_error']),
                              ('planning_lower_root_n_and_zero_tail_at_128', future['lower']),
                              ('planning_upper_root_n_and_zero_tail_at_128', future['upper'])]:
            expected = reference[key]
            require(same_number(expected, observed), 'LEGACY_BAND_NUMERICAL_RECONSTRUCTION_FAILED:'+str(index)+':'+key)
            if expected != '': maximum_error = max(maximum_error, abs(float(expected)-float(observed))); compared += 1
    return dict(rows=len(saved), numeric_cells=compared, maximum_absolute_error=maximum_error,
                tolerance='64*float64_epsilon*max(1,abs(expected),abs(observed)); arithmetic roundoff only')


def combined_critical(values, multiplicities, prior_maximum, bootstrap_alpha, chunk_size):
    point, se, maximum, influence = bootstrap_primitives(values, multiplicities, prior_maximum, chunk_size)
    old = float(np.quantile(prior_maximum, 1-bootstrap_alpha, method='higher'))
    critical = float(np.quantile(maximum, 1-bootstrap_alpha, method='higher'))
    require(np.all(maximum >= prior_maximum) and critical >= old, 'COMPLETE_SHARED_MAXIMUM_CANNOT_DECREASE')
    return point, se, maximum, influence, critical


def augment_cdf_bands(legacy_bands, row_to_group, point, se, critical, planning_scale=1.):
    """Intersect nonconstant K bounds after every legacy channel was widened."""
    upper = np.ones(len(point))
    variable = se > 0
    upper[variable] = np.minimum(1., point[variable]+critical*se[variable]*planning_scale)
    require(np.all(upper >= point) and np.all((upper >= 0) & (upper <= 1)), 'VALID_NONNEGATIVE_COUPLING_UPPER_BOUNDS_REQUIRED')
    result = []
    for index, source in enumerate(legacy_bands):
        row = dict(source); group = int(row_to_group[index])
        row.update(cdf_coupling_group_index=group if group >= 0 else '', cdf_coupling_anchor_status='', cdf_coupling_upper_bound='',
                   legacy_lower_before_coupling=row['lower'], legacy_upper_before_coupling=row['upper'], cdf_coupling_channel_used=False)
        if group >= 0:
            row['cdf_coupling_anchor_status'] = 'nonconstant_approximate_mean_bound' if variable[group] else 'empirical_constant_physical_fallback_1_no_channel'
            row['cdf_coupling_upper_bound'] = float(upper[group])
            require(abs(float(row['estimate'])) <= point[group]+ROUNDING_FACTOR*np.finfo(float).eps,
                    'EMPIRICAL_POINT_MUST_LIE_WITHIN_MEAN_COUPLING_ENVELOPE')
            if variable[group] and row['definition_status'] != STRUCTURAL:
                require(row['family'] == 'count_cdf' and row['definition_status'] != UNDEFINED, 'COUPLING_CHANNEL_REQUIRES_DEFINED_CDF')
                lower, high = max(float(row['lower']), -float(upper[group])), min(float(row['upper']), float(upper[group]))
                require(lower <= high, 'CROSSING_LEGACY_AND_COUPLING_BANDS')
                row.update(lower=lower, upper=high, channel=row['channel']+'+whole_block_cdf_coupling', cdf_coupling_channel_used=True)
        result.append(row)
    return result, upper


def _bound_rows(root, report, name):
    path = root/name; binding = report['outputs'][name]
    require(path.stat().st_size == binding['bytes'] and sha256_file(path) == binding['sha256'], 'BOUND_INPUT_CHANGED:'+name)
    return read_rows(path)


def _outputs(output):
    return {path.name: dict(bytes=path.stat().st_size, sha256=sha256_file(path))
            for path in sorted(output.iterdir()) if path.is_file() and path.name != 'report.json'}


def load_inputs(config, root):
    roots = {key: root/ref['path'] for key, ref in config['inputs'].items()}
    reports = {key: bound_report(roots[key], ref['report_sha256']) for key, ref in config['inputs'].items()}
    prior = reports['uncertainty']; coupled = reports['coupled_scores']; domain_report = reports['domain']; budget = reports['budget']
    require(prior['status'] == 'COMPLETE_COUPLED_CALIBRATION_JOINT_UNCERTAINTY_DIAGNOSTIC'
            and coupled['status'] == 'COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON'
            and domain_report['status'] == 'COMPLETE_CALIBRATION_BLOCK_CDF_SCORES'
            and budget['numerical_margins_frozen'] is True, 'COMPLETE_FROZEN_COUPLED_DIAGNOSTICS_REQUIRED')
    require(prior['settings']['inputs']['coupled'] == config['inputs']['coupled_scores']
            and prior['settings']['inputs']['domain'] == config['inputs']['domain']
            and prior['settings']['inputs']['budget'] == config['inputs']['budget']
            and prior['settings']['joint_rng_seeds'] == list(SEEDS)
            and prior['settings']['bootstrap_seed'] == 97001 and prior['settings']['bootstrap_replicates'] == 5000
            and prior['bootstrap_alpha'] == config['bootstrap_alpha'] == .025
            and prior['joint_tail_alpha'] == config['joint_tail_alpha'] == .025,
            'UNCHANGED_SHARED_CALIBRATION_AND_ERROR_BUDGET_BINDINGS_REQUIRED')
    require(all(prior[key] is False for key in ('coverage_qualified', 'method_adopted_for_confirmation',
                'confirmation_execution_released', 'scientific_fidelity_qualified')), 'UNQUALIFIED_PRIOR_DIAGNOSTIC_REQUIRED')
    native = reports['denominator_source']
    require(coupled['settings']['native_comparison'] == config['inputs']['denominator_source']
            and native['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
            and coupled['phase_boundaries_ms'] == PHASES, 'SOURCE_DENOMINATOR_PROVENANCE_REQUIRED')
    field_sizes = native['inputs'][LABELS[0]]['field_unit_counts']
    require(field_sizes == [5000]*16+[20000]
            and all(native['inputs'][label]['field_unit_counts'] == field_sizes
                    and native['inputs'][label]['source_rng_blocks'] == list(SEEDS)
                    and native['inputs'][label]['coupled_units'] == 100000 for label in LABELS),
            'SOURCE_NATIVE_FIELD_SIZES_AND_ORIGINAL_BLOCKS_REQUIRED')
    definition = reports['domain_definition']
    require(domain_report['settings']['domain_archive'] == config['inputs']['domain_definition']['path']
            and domain_report['settings']['domain_report_sha256'] == config['inputs']['domain_definition']['report_sha256']
            and definition['status'] == 'COMPLETE_PHYSICAL_ENDPOINT_DOMAIN_ESTIMATES_PARTIAL'
            and definition['complete_endpoint_rows'] == 84330 and definition['cdf_support_groups'] == 1530
            and definition['outside_domain_rule'] == 'count threshold <0 gives CDF0; threshold>=source physical maximum gives CDF1; adjacent-clock contrast0',
            'BOUND_SOURCE_SATURATION_IDENTITY_REQUIRED')
    domains = _bound_rows(roots['domain'], domain_report, 'endpoint_domains.csv')
    margins = _bound_rows(roots['budget'], budget, 'frozen_numerical_margins.csv')
    saved = _bound_rows(roots['uncertainty'], prior, 'endpoint_bands.csv')
    direct = _bound_rows(roots['uncertainty'], prior, 'endpoint_sampling_precision.csv')
    primitives = _bound_rows(roots['uncertainty'], prior, 'primitive_bands.csv')
    require(len(domains) == len(margins) == len(saved) == len(direct) == prior['domain_rows'] == 84330,
            'ALL_FROZEN_ENDPOINTS_REQUIRED')
    physical = {}; current = []
    for index, (domain, margin, band, precision) in enumerate(zip(domains, margins, saved, direct)):
        require(all(int(row['domain_row_index']) == index for row in (margin, band, precision))
                and all(all(row[key] == domain[key] for key in ('family', 'endpoint_identity', 'definition_status'))
                        for row in (margin, band, precision)), 'EXACT_FULL_FROZEN_ENDPOINT_JOIN_REQUIRED')
        require(precision['current_empirical_status'] == band['current_empirical_status'], 'CURRENT_EMPIRICAL_STATUS_CHANGED')
        current.append(dict(domain, estimate=band['estimate']))
        if domain['family'] == 'count_cdf':
            ident = json.loads(domain['endpoint_identity']); key = float(ident['window_ms']), int(ident['field_index'])
            maximum = int(domain['physical_maximum_count'])
            require(key not in physical or physical[key] == maximum, 'CONSISTENT_FROZEN_PHYSICAL_SUPPORT_REQUIRED'); physical[key] = maximum
    scores = load_npz(roots['coupled_scores']/'coupled_block_scores.npz', coupled)
    maxima = load_npz(roots['uncertainty']/'joint_bootstrap_maximum.npz', prior)
    require(np.array_equal(scores['source_rng_seeds'], SEEDS) and np.array_equal(maxima['source_rng_seeds'], SEEDS)
            and np.array_equal(scores['domain_row_indices'], np.arange(len(domains)))
            and scores['scores'].shape == (len(domains), len(SEEDS))
            and maxima['bootstrap_multiplicities'].shape == (5000, len(SEEDS))
            and maxima['joint_maximum'].shape == (5000,) and np.isfinite(maxima['joint_maximum']).all(),
            'EXACT_ORDERED_ORIGINAL_BLOCK_AND_DRAW_AXES_REQUIRED')
    return current, margins, saved, direct, primitives, physical, scores['scores'], maxima, reports, field_sizes


def run(config_path, project_root, output):
    config = yaml.safe_load(config_path.read_text())
    require(config.get('schema') == SCHEMA and config.get('scope') == 'development_calibration'
            and all(config.get(key) is False for key in ('coverage_qualified', 'method_adopted_for_confirmation', 'confirmation_execution_released')),
            'UNQUALIFIED_CALIBRATION_CANDIDATE_ONLY_REQUIRED')
    require(config['expected_endpoint_rows'] == 84330 and config['expected_cdf_groups'] == 1530
            and config['planning_blocks'] == 128 and config['nominal_alpha'] == .05
            and config['endpoint_chunk_size'] == 256 and set(config['inputs']) == set(INPUT_FILES), 'UNCHANGED_FULL_DOMAIN_AND_RESERVATION_REQUIRED')
    data = load_inputs(config, project_root)
    domains, margins, saved, direct, primitives, physical, scores, maxima, reports, field_sizes = data
    prior = reports['uncertainty']; output.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    report = dict(settings=config, settings_sha256=sha256_file(config_path), domain_rows=len(domains), original_independent_blocks=32,
        independent_circuit_instances=1, source_simulations=0, original_bootstrap_recomputed=False, frozen_margins_recomputed=False,
        frozen_margin_sha256=reports['budget']['outputs']['frozen_numerical_margins.csv']['sha256'],
        coverage_qualified=False, uncertainty_coverage_qualified=False, method_adopted_for_confirmation=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False, confirmation_n_selected=None,
        production_episodes=0, planning_blocks=128, bootstrap_seed=97001, bootstrap_replicates=5000,
        nominal_alpha=.05, bootstrap_alpha=.025, joint_tail_alpha=.025, implementation_sha256=sha256_file(Path(__file__)),
        inequality='For every threshold in one complete group, abs(E D_b(k)) <= E abs(D_b(k)) <= E max_k abs(D_b(k)).',
        independent_unit='one original paired initial/background-noise block; no independence of neurons, thresholds or within-block windows is assumed',
        constant_anchor_policy='An empirically constant K uses physical upper bound1 and adds no intersection channel.',
        interpretation='Approximate joint bootstrap augmented by whole-block CDF mean envelopes. Reused calibration data and diagnostic planning intervals do not establish coverage, simulator fidelity or confirmation eligibility.')
    try:
        primitive, lookup, support = restore_primitives(primitives); precision = precision_lookup(domains, direct)
        old_critical = float(np.quantile(maxima['joint_maximum'], .975, method='higher'))
        require(old_critical == prior['joint_critical_value'], 'SAVED_MAXIMUM_CRITICAL_VALUE_CHANGED')
        tail, future_tail = prior['joint_tail_probability_upper_calibration'], prior['joint_tail_probability_upper_zero_departure_planning']
        old_actual, _, _ = construct_bands(domains, precision, primitive, lookup, support, physical, old_critical, tail)
        old_projected, _, _ = construct_bands(domains, precision, primitive, lookup, support, physical, old_critical*.5, future_tail)
        reconstruction = compare_reconstructed_bands(saved, old_actual, old_projected)
        report['legacy_reconstruction'] = reconstruction
        old_assessment, _ = assess_planning(margins, saved, 128)
        require(dict(Counter(row['planning_status'] for row in old_assessment)) == prior['planning_status_counts'], 'RECONSTRUCTED_PRIOR_PLANNING_COUNTS_CHANGED')
        del old_actual, old_projected
        k, identities, maximum_counts, row_group, denominators, numerators, roundtrips = complete_cdf_group_scores(domains, scores, saved, field_sizes, terminal_boundary_identity_proven=True)
        require(len(k) == config['expected_cdf_groups'], 'ALL_CLOCK_CONDITION_PHASE_WINDOW_FIELD_GROUPS_REQUIRED')
        expected_keys = {(left,right,condition,phase,float(window),field) for left,right in zip(LABELS[:-1],LABELS[1:])
                         for condition in CONDITIONS for phase in PHASES for window in [5,50,100] for field in range(17)}
        require({cdf_group(identity) for identity in identities} == expected_keys, 'FULL_CDF_GROUP_IDENTITY_COVERAGE_REQUIRED')
        np.savez_compressed(output/'cdf_group_block_coupling_scores.npz', scores=k, source_rng_seeds=np.asarray(SEEDS),
                            group_identities=np.asarray([json.dumps(identity,sort_keys=True) for identity in identities]),
                            physical_maximum_counts=maximum_counts, count_denominators=denominators, exact_maximum_count_difference_numerators=numerators,
                            domain_row_indices=np.arange(len(domains)),
                            group_indices_by_domain_row=row_group, units=np.asarray('probability'))
        point, se, maximum, influence, critical = combined_critical(k, maxima['bootstrap_multiplicities'], maxima['joint_maximum'], .025, 256)
        legacy_actual, _, _ = construct_bands(domains, precision, primitive, lookup, support, physical, critical, tail)
        legacy_future, _, _ = construct_bands(domains, precision, primitive, lookup, support, physical, critical*.5, future_tail)
        bands, upper = augment_cdf_bands(legacy_actual, row_group, point, se, critical)
        projected, future_upper = augment_cdf_bands(legacy_future, row_group, point, se, critical, .5)
        for row, future, previous in zip(bands, projected, saved):
            row['current_empirical_status'] = previous['current_empirical_status']; row['frozen_native_estimate'] = previous['frozen_native_estimate']
            row['planning_lower_root_n_and_zero_tail_at_128'] = future['lower']; row['planning_upper_root_n_and_zero_tail_at_128'] = future['upper']
            row['planning_legacy_lower_before_coupling'] = future['legacy_lower_before_coupling']; row['planning_legacy_upper_before_coupling'] = future['legacy_upper_before_coupling']
            row['planning_cdf_coupling_upper_bound'] = future['cdf_coupling_upper_bound']
            if row['definition_status'] not in (STRUCTURAL, UNDEFINED):require(row['lower'] < row['upper'], 'NO_ZERO_WIDTH_NONSTRUCTURAL_UNCERTAINTY_ALLOWED')
        assessment, grouped = assess_planning(margins, bands, 128)
        anchors = [dict(group_index=i, group_identity=json.dumps(identity,sort_keys=True), physical_maximum_count=int(maximum_counts[i]),
            mean_whole_block_cdf_maximum=float(point[i]), bootstrap_standard_error=float(se[i]), count_denominator=int(denominators[i]),
            **roundtrips[i],
            upper_bound=float(upper[i]), planning_upper_bound_at_128=float(future_upper[i]),
            largest_block_squared_influence_fraction=float(influence[i]) if np.isfinite(influence[i]) else '',
            influence_flag=bool(influence[i] >= .25),
            status='nonconstant_approximate_mean_bound' if se[i] > 0 else 'empirical_constant_physical_fallback_1_no_channel')
            for i,identity in enumerate(identities)]
        write_rows(output/'cdf_coupling_anchors.csv', anchors); write_rows(output/'endpoint_bands.csv', bands)
        write_rows(output/'calibration_budget_assessment.csv', assessment)
        transitions = [dict(domain_row_index=i, family=row['family'], endpoint_identity=row['endpoint_identity'], definition_status=row['definition_status'],
                            prior_planning_status=old['planning_status'], augmented_planning_status=row['planning_status'],
                            prior_point_outside=old['calibration_point_outside_margin'], augmented_point_outside=row['calibration_point_outside_margin'],
                            prior_projected_relation=old['planning_interval_relation'], augmented_projected_relation=row['planning_interval_relation'])
                       for i,(old,row) in enumerate(zip(old_assessment,assessment))]
        write_rows(output/'all_endpoint_transitions.csv', transitions)
        adverse = [dict(row, **{key:bands[i][key] for key in ['lower','upper','channel','current_empirical_status','cdf_coupling_group_index',
                   'cdf_coupling_anchor_status','cdf_coupling_upper_bound','planning_cdf_coupling_upper_bound']})
                   for i,row in enumerate(assessment) if row['planning_status'] == 'PLANNING_BAND_EXCEEDS_BUDGET']
        if adverse:write_rows(output/'projected_exceedances.csv', adverse)
        np.savez_compressed(output/'joint_bootstrap_maximum.npz', prior_joint_maximum=maxima['joint_maximum'], joint_maximum=maximum,
                            bootstrap_multiplicities=maxima['bootstrap_multiplicities'], source_rng_seeds=np.asarray(SEEDS))
        counts = Counter(row['planning_status'] for row in assessment)
        transition_counts = Counter((row['prior_planning_status'],row['augmented_planning_status']) for row in transitions)
        report.update(status='COMPLETE_CALIBRATION_CDF_COUPLING_BOUND_DIAGNOSTIC', cdf_groups=len(k),
            source_identity_terminal_boundaries_added=sum(row['source_identity_terminal_boundary_added'] for row in roundtrips),
            terminal_boundary_completion_rule=reports['domain_definition']['outside_domain_rule'],
            count_denominator_rule='source field units times complete nonoverlapping windows in frozen phase',
            source_field_unit_counts=field_sizes, count_lattice_rounding_factor=8,
            maximum_count_numerator_roundtrip_error=max(row['maximum_integer_roundtrip_error'] for row in roundtrips),
            maximum_count_probability_roundtrip_error=max(row['maximum_probability_roundtrip_error'] for row in roundtrips),
            raw_float_varied_integer_constant_ks_groups=sum(row['raw_float_ks_varied_but_integer_ks_constant'] for row in roundtrips),
            anchor_status_counts=dict(Counter(row['status'] for row in anchors)), prior_critical_value=old_critical, joint_critical_value=critical,
            maximum_draws_increased=int(np.count_nonzero(maximum > maxima['joint_maximum'])),
            cdf_anchor_single_block_influence_flags=sum(row['influence_flag'] for row in anchors),
            anchors_reusing_identical_prior_multiplicities=True, all_legacy_channels_recomputed_at_augmented_critical=True,
            joint_tail_probability_upper_calibration=tail, joint_tail_probability_upper_zero_departure_planning=future_tail,
            planning_status_counts=dict(counts), planning_by_clock_pair_and_family=grouped,
            planning_transitions=[dict(prior=a,augmented=b,count=n) for (a,b),n in sorted(transition_counts.items())],
            projected_exceedances_by_family=dict(Counter(row['family'] for row in adverse)),
            calibration_interval_relations=dict(Counter(row['calibration_interval_relation'] for row in assessment if row['calibration_interval_relation'])),
            planning_interval_relations=dict(Counter(row['planning_interval_relation'] for row in assessment if row['planning_interval_relation'])),
            calibration_points_outside_budget=sum(row['calibration_point_outside_margin'] is True for row in assessment),
            point_outside_rows=[row for row in assessment if row['calibration_point_outside_margin'] is True],
            projected_disjoint_rows=[row for row in assessment if row['planning_interval_relation'] == 'DISJOINT_FROM_BUDGET'])
    except (ValueError, FloatingPointError) as error:
        report.update(status='FAILED_CALIBRATION_CDF_COUPLING_BOUND_DIAGNOSTIC', error=str(error), outputs=_outputs(output), wall_seconds=time.monotonic()-start)
        freeze_json(output/'report.json',report); raise
    report.update(outputs=_outputs(output), wall_seconds=time.monotonic()-start)
    freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.config,args.project_root,args.output)
    print(json.dumps({key:report[key] for key in ['status','cdf_groups','joint_critical_value','planning_status_counts']}))
    return 0


if __name__ == '__main__':raise SystemExit(main())

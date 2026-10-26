"""Recompute the existing approximate joint uncertainty diagnostic on coupled blocks.

All endpoint identities and numerical margins remain frozen. This consumes only
completed readouts, never source trajectories or confirmation blocks. Whole-block
covariance resampling uses the order-two estimator, not resampled pseudovalues.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from . import pfc_numerical_precision, pfc_profile_history_coverage
from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_confirmation_covariance_scores import lag_pair_kernels, round_robin_pairings
from .pfc_confirmation_linear_scores import linear_block_score
from .pfc_confirmation_uncertainty import (
    EMPIRICAL, IDENTITY, STRUCTURAL, UNDEFINED, binomial_probability_upper,
    bootstrap_primitives, construct_bands, covariance_jackknife,
    frozen_tail_departures, identity_key, load_npz, prepare_primitives, write_rows,
)
from .pfc_delay_analysis import PHASES
from .pfc_numerical_error_budget import assess_planning
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS
from .pfc_profile_history import profile_key
from .provenance import freeze_json, sha256_file
from .source_variant_analysis import paired_bootstrap_multiplicities

SCHEMA = 'pfc_coupled_uncertainty_diagnostic_v1'
LABELS = ('numpy_dt_0p05', 'numpy_dt_0p025', 'numpy_dt_0p0125')
SEEDS = tuple(range(95001, 95033))
FIELD_SIZES = (5000,)*16 + (20000,)
BIN_MS = 5.
POSITIVE = 'ESTIMATED_EMPIRICAL_SPREAD'


def validate_settings(settings):
    require(settings.get('schema') == SCHEMA
            and settings.get('scope') == 'development_calibration'
            and settings.get('method_adopted_for_confirmation') is False
            and settings.get('confirmation_execution_released') is False
            and settings.get('coverage_qualified') is False,
            'CALIBRATION_DIAGNOSTIC_ONLY_REQUIRED')
    require(settings['cohorts'] == list(LABELS) and settings['joint_rng_seeds'] == list(SEEDS)
            and settings['bootstrap_replicates'] == 5000 and settings['bootstrap_seed'] == 97001
            and settings['planning_blocks'] == 128 and settings['nominal_alpha'] == .05
            and settings['joint_tail_alpha'] == .025 and settings['endpoint_chunk_size'] == 256
            and settings['single_block_influence_fraction_flag'] == .25,
            'UNCHANGED_CALIBRATION_RESAMPLING_AND_PLANNING_DESIGN_REQUIRED')
    require(set(settings['inputs']) == {'coupled', 'domain', 'budget', 'linear', 'resampling_design'},
            'EXPLICIT_COUPLED_DOMAIN_MARGIN_AND_DESIGN_BINDINGS_REQUIRED')


def bound_table(root, report, name):
    path = root/name
    require(path.stat().st_size == report['outputs'][name]['bytes']
            and sha256_file(path) == report['outputs'][name]['sha256'],
            'BOUND_INPUT_CHANGED:'+name)
    return path


def validate_axes(fields, responses, cdf, scores, seeds, rows):
    """Require source axes before consuming values or taking any differences."""
    required = {
        'fields': {'source_rng_seeds', 'field_unit_counts', 'field_indices', 'condition_names', 'time_start_ms', 'bin_ms', 'units'}
                  | {label+'_'+condition for label in LABELS for condition in CONDITIONS},
        'responses': {'source_rng_seeds', 'condition_names', 'phase_names', 'metric_names', 'phase_start_ms', 'phase_stop_ms', 'units'}
                     | {label+'_values' for label in LABELS},
        'cdf': {'source_rng_seeds', 'count_support', 'condition_names', 'phase_names', 'window_ms', 'field_unit_counts', 'field_indices', 'units'}
               | {label+'_cdf' for label in LABELS},
        'scores': {'source_rng_seeds', 'domain_row_indices', 'scores'},
    }
    for name, data in [('fields', fields), ('responses', responses), ('cdf', cdf), ('scores', scores)]:
        require(set(data) == required[name], 'COMPLETE_NAMED_ARRAY_SCHEMA_REQUIRED:'+name)
        require('source_rng_seeds' in data and np.array_equal(data['source_rng_seeds'], seeds),
                'SHARED_ORIGINAL_BLOCK_AXIS_CHANGED:'+name)
    require(scores['domain_row_indices'].dtype.kind in 'iu'
            and np.array_equal(scores['domain_row_indices'], np.arange(rows))
            and scores['scores'].shape == (rows, len(seeds)), 'FULL_ORDERED_SCORE_DOMAIN_REQUIRED')
    for name, data in [('fields', fields), ('responses', responses), ('cdf', cdf)]:
        require(np.array_equal(data['condition_names'], CONDITIONS), 'SOURCE_CONDITION_AXIS_CHANGED:'+name)
    for name, data in [('responses', responses), ('cdf', cdf)]:
        require(np.array_equal(data['phase_names'], list(PHASES)), 'SOURCE_PHASE_AXIS_CHANGED:'+name)
    for name, data in [('fields', fields), ('cdf', cdf)]:
        require(np.array_equal(data['field_unit_counts'], FIELD_SIZES)
                and np.array_equal(data['field_indices'], np.arange(len(FIELD_SIZES))),
                'SOURCE_FIELD_AXIS_CHANGED:'+name)
    require(str(fields['units']) == str(responses['units']) == 'Hz'
            and str(cdf['units']) == 'probability', 'SOURCE_OBSERVATION_UNITS_CHANGED')
    require(float(fields['bin_ms']) == BIN_MS
            and np.array_equal(fields['time_start_ms'], np.arange(1000)*BIN_MS)
            and np.array_equal(responses['phase_start_ms'], [v[0] for v in PHASES.values()])
            and np.array_equal(responses['phase_stop_ms'], [v[1] for v in PHASES.values()])
            and np.array_equal(responses['metric_names'], METRICS)
            and np.array_equal(cdf['window_ms'], [5, 50, 100]), 'SOURCE_TIME_OR_METRIC_AXIS_CHANGED')
    support = cdf['count_support']
    require(support.ndim == 1 and support.dtype.kind in 'iu' and len(support) > 0
            and support[0] == 0 and np.all(np.diff(support) > 0), 'ORDERED_COUNT_SUPPORT_REQUIRED')
    for label in LABELS:
        for condition in CONDITIONS:
            x = fields[label+'_'+condition]
            require(x.shape == (len(seeds), 1000, len(FIELD_SIZES)) and np.isfinite(x).all()
                    and np.all(x >= 0), 'COMPLETE_FINITE_FIELD_RATES_REQUIRED')
        x = responses[label+'_values']
        require(x.shape == (3, len(seeds), len(PHASES), len(METRICS)) and np.isfinite(x).all(),
                'COMPLETE_FINITE_RESPONSE_ARRAY_REQUIRED')
        x = cdf[label+'_cdf']
        require(x.shape == (3, len(seeds), len(PHASES), 3, len(FIELD_SIZES), len(support))
                and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
                and np.all(np.diff(x, axis=-1) >= 0) and np.all(x[..., -1] == 1),
                'COMPLETE_MONOTONE_EMPIRICAL_CDF_REQUIRED')


def bind_endpoints(domains, margins, statistics, scores, blocks):
    """Bind current estimates without replacing any frozen definition or margin."""
    require(len(domains) == len(margins) == len(scores), 'FULL_FROZEN_DOMAIN_REQUIRED')
    current, seen = [], set()
    for index, stat in enumerate(statistics):
        require(index < len(domains), 'EXTRA_COUPLED_ENDPOINT')
        domain, margin = domains[index], margins[index]
        require(int(stat['domain_row_index']) == int(margin['domain_row_index']) == index
                and all(stat[k] == margin[k] == domain[k]
                        for k in ('family', 'endpoint_identity', 'definition_status')),
                'FROZEN_ENDPOINT_ORDER_OR_IDENTITY_CHANGED')
        family = domain['family']; ident = json.loads(domain['endpoint_identity'])
        require(family in IDENTITY and set(ident) == set(IDENTITY[family]), 'FULL_ENDPOINT_IDENTITY_REQUIRED')
        key = identity_key(family, ident)
        require(key not in seen, 'DUPLICATE_ENDPOINT_IDENTITY'); seen.add(key)
        require((ident['left_cohort'], ident['right_cohort']) in set(zip(LABELS[:-1], LABELS[1:]))
                and ident['phase'] in PHASES, 'CANONICAL_CLOCK_PAIR_AND_PHASE_REQUIRED')
        require(stat['margin'] == margin['margin'] and stat['units'] == margin['units']
                and stat['physical_maximum_count'] == domain['physical_maximum_count']
                and int(stat['coupled_original_independent_blocks']) == blocks,
                'FROZEN_MARGIN_UNITS_OR_BLOCK_COUNT_CHANGED')
        status = domain['definition_status']
        empirical = stat['coupled_empirical_status']
        require(status in (POSITIVE, EMPIRICAL, STRUCTURAL, UNDEFINED)
                and empirical in (POSITIVE, EMPIRICAL, STRUCTURAL, UNDEFINED), 'KNOWN_ENDPOINT_STATUS_REQUIRED')
        require((status == UNDEFINED) == (empirical == UNDEFINED)
                and (status == STRUCTURAL) == (empirical == STRUCTURAL), 'SOURCE_IDENTITY_OR_LAG_DOMAIN_CHANGED')
        if status == UNDEFINED:
            require(stat['coupled_signed_estimate'] == '' and np.isnan(scores[index]).all(),
                    'ONLY_UNDEFINED_LAGS_MAY_LACK_ESTIMATES')
        else:
            require(np.isfinite(scores[index]).all() and np.isfinite(float(stat['coupled_signed_estimate'])),
                    'FINITE_DEFINED_BLOCK_SCORES_REQUIRED')
            if status == STRUCTURAL:
                require(float(stat['coupled_signed_estimate']) == 0 and np.all(scores[index] == 0),
                        'STRUCTURAL_ZERO_CONTRADICTED')
        expected_kind = (UNDEFINED if status == UNDEFINED else 'covariance_jackknife_pseudovalue'
                         if family == 'field_lag_covariance' else 'paired_block_cdf_difference'
                         if family == 'count_cdf' else 'paired_block_linear_difference')
        require(stat['coupled_score_kind'] == expected_kind, 'ORIGINAL_BLOCK_SCORE_KIND_CHANGED')
        current.append(dict(domain, estimate=stat['coupled_signed_estimate'],
                            frozen_native_estimate=domain['estimate'], current_empirical_status=empirical))
    require(len(current) == len(domains), 'MISSING_COUPLED_ENDPOINTS')
    return current


def shared_multiplicities(settings, native_report):
    design = native_report['calibration_precision']
    require(design['status'] == 'COMPLETE_CALIBRATION_SAMPLING_PRECISION'
            and design['settings']['schema'] == 'pfc_numerical_precision_v1'
            and design['all_clocks_conditions_and_endpoints_share_resample_indices'] is True
            and all(design['settings'][key] == settings[key]
                    for key in ('joint_rng_seeds', 'bootstrap_replicates', 'bootstrap_seed')),
            'BOUND_NATIVE_RESAMPLING_DESIGN_CHANGED')
    require(design['implementation_sha256'] == sha256_file(Path(pfc_numerical_precision.__file__)),
            'VERIFIED_PRECISION_IMPLEMENTATION_CHANGED')
    return paired_bootstrap_multiplicities(len(settings['joint_rng_seeds']),
                                          settings['bootstrap_replicates'], settings['bootstrap_seed'])


def add_linear_precision(summary, scores, point, identities, multiplicities):
    """Use the same complete block draw for every linear/CDF endpoint."""
    x = np.asarray(scores, dtype=float)
    require(x.shape == (len(point), multiplicities.shape[1]) and np.isfinite(x).all(),
            'COMPLETE_LINEAR_BLOCK_DIFFERENCES_REQUIRED')
    draws = (multiplicities/x.shape[1])@x.T
    constant = np.all(x == x[:, :1], axis=1)
    # An exactly constant block statistic must not acquire numerical spread.
    draws[:, constant] = x[constant, 0]
    return summary.add(point, draws, point, identities)


def conditional_history_scores(history_settings, records, domains, source_rng_seeds):
    """Prepare 108 diagnostics from every registered original-block A/B pair.

    The paired second-spike score is fine P(N>=2) minus coarse P(N>=2).
    This has the frozen left-minus-right CDF-at-one sign. A/B estimates and
    their fixed equal midpoint retain the same 32-block source axis. Signed
    variance diagnostics remain descriptive; they do not modify these scores.
    """
    seeds = np.asarray(source_rng_seeds)
    require(seeds.shape == (len(SEEDS),) and seeds.dtype.kind in 'iu'
            and np.array_equal(seeds, SEEDS), 'CONDITIONAL_HISTORY_ORIGINAL_BLOCK_AXIS_REQUIRED')
    require(len(domains) == 84330
            and sum(row['definition_status'] == UNDEFINED for row in domains) == 1734,
            'CONDITIONAL_HISTORY_FULL_FROZEN_DOMAIN_REQUIRED')
    coverage = pfc_profile_history_coverage.aggregate(history_settings, list(SEEDS), records)
    require(coverage['full_registered_scope_covered'] is True
            and coverage['completed_pairs'] == 2304
            and coverage['available_source_seeds'] == list(SEEDS),
            'CONDITIONAL_HISTORY_COMPLETE_REGISTERED_COVERAGE_REQUIRED')
    profiles = history_settings['profiles']
    mapped, unresolved, registered_keys, registered_indices = [], [], set(), set()
    for profile_index, profile in enumerate(profiles):
        endpoints = profile['endpoints']
        require(isinstance(endpoints, list)
                and all(type(endpoint['count_threshold']) is int
                        and endpoint['count_threshold'] in (1, 2) for endpoint in endpoints),
                'CONDITIONAL_HISTORY_REGISTERED_THRESHOLD_DESCRIPTORS_REQUIRED')
        threshold_one = [endpoint for endpoint in endpoints if endpoint['count_threshold'] == 1]
        require(len(threshold_one) == 1, 'CONDITIONAL_HISTORY_ONE_REGISTERED_THRESHOLD1_REQUIRED')
        bound_one = None
        for endpoint in endpoints:
            index, threshold = endpoint['row_index'], endpoint['count_threshold']
            require(type(index) is int and 0 <= index < len(domains)
                    and index not in registered_indices,
                    'CONDITIONAL_HISTORY_UNIQUE_REGISTERED_DOMAIN_ROW_REQUIRED')
            registered_indices.add(index)
            row = domains[index]
            expected = {name: profile[name] for name in IDENTITY['count_cdf'] if name != 'count_threshold'}
            expected['count_threshold'] = threshold
            require(row['family'] == 'count_cdf' and row.get('units') == 'probability',
                    'CONDITIONAL_HISTORY_COUNT_CDF_PROBABILITY_ENDPOINT_REQUIRED')
            identity = json.loads(row['endpoint_identity'])
            require(set(identity) == set(IDENTITY['count_cdf'])
                    and identity_key('count_cdf', identity) == identity_key('count_cdf', expected),
                    'CONDITIONAL_HISTORY_REGISTERED_ENDPOINT_IDENTITY_CHANGED')
            key = identity_key('count_cdf', identity)
            require(key not in registered_keys, 'CONDITIONAL_HISTORY_DUPLICATE_REGISTERED_ENDPOINT')
            registered_keys.add(key)
            binding = dict(domain_row_index=index, family='count_cdf',
                           endpoint_identity=row['endpoint_identity'],
                           definition_status=row['definition_status'], units=row['units'],
                           profile_index=profile_index, count_threshold=threshold)
            if threshold == 1:
                require(row['definition_status'] in (POSITIVE, EMPIRICAL),
                        'CONDITIONAL_HISTORY_DEFINED_NONSTRUCTURAL_THRESHOLD1_REQUIRED')
                bound_one = binding
            else:
                unresolved.append(dict(binding, registered_descriptor=dict(endpoint),
                    status='UNRESOLVED_THRESHOLD2_NO_CONDITIONAL_HISTORY_SCORE',
                    conditional_history_score_computed=False))
        mapped.append(bound_one)
    # Check the complete domain for another row with a registered identity.
    occurrences = Counter()
    for row in domains:
        if row['family'] == 'count_cdf':
            key = identity_key('count_cdf', json.loads(row['endpoint_identity']))
            if key in registered_keys:
                occurrences[key] += 1
    require(all(occurrences[key] == 1 for key in registered_keys),
            'CONDITIONAL_HISTORY_UNIQUE_FROZEN_ENDPOINT_IDENTITIES_REQUIRED')
    indexed = {(record['source_rng_seed'], record['key']): record for record in coverage['records']}
    scores, identities = [], []
    for repeat in ('A', 'B'):
        for profile, binding in zip(profiles, mapped):
            key = profile_key(profile, repeat)
            scores.append([indexed[seed, key]['paired']['sample_mean'] for seed in SEEDS])
            identities.append(dict(binding, coordinate_kind='conditional_history_empirical_diagnostic',
                auxiliary_repeat=repeat, profile_key=key, source_profile_keys=[key], merger_weights=[1.]))
    scores = np.asarray(scores, dtype=float)
    midpoint = .5*(scores[:36] + scores[36:])
    for profile, binding in zip(profiles, mapped):
        keys = [profile_key(profile, repeat) for repeat in ('A', 'B')]
        identities.append(dict(binding, coordinate_kind='conditional_history_empirical_diagnostic',
            auxiliary_repeat='A_B_midpoint', profile_key='A_B_midpoint/'+keys[0].split('/', 1)[1],
            source_profile_keys=keys, merger_weights=[.5, .5]))
    scores = np.concatenate((scores, midpoint))
    point = scores.mean(1)
    constant = np.all(scores == scores[:, :1], axis=1)
    point[constant] = scores[constant, 0]
    return dict(scores=scores, point=point, identities=identities, coverage=coverage,
        source_rng_seeds=seeds.copy(), domain_rows=len(domains), undefined_domain_rows=1734,
        unresolved_threshold2=unresolved, diagnostic_coordinates=108,
        coordinate_order='A_profiles_then_B_profiles_then_equal_AB_midpoints',
        score_definition='fine_P_N_ge2_minus_coarse_P_N_ge2_equals_left_minus_right_CDF_le1',
        frozen_endpoint_scores_replaced=False, negative_variance_diagnostics_preserved=True,
        variance_noise_subtraction_applied_to_bootstrap_scores=False,
        threshold2_score_computed=False, population_confidence_interval_computed=False,
        source_history_population_uncertainty_established=False)


def augment_conditional_history_precision(original_joint_maximum, original_multiplicities, prepared_scores):
    """Extend the bound whole-domain maximum with the 108 shared-draw diagnostics.

    The returned maximum is one maximum per original draw. No diagnostic subset
    maximum, bands or confidence intervals are returned. The new-coordinate
    counters are separate from the unchanged original-domain counters.
    """
    m = np.asarray(original_multiplicities)
    require(m.shape == (5000, len(SEEDS)) and m.dtype.kind in 'iu'
            and np.all(m >= 0) and np.all(m.sum(1) == len(SEEDS))
            and np.array_equal(m, paired_bootstrap_multiplicities(len(SEEDS), 5000, 97001)),
            'CONDITIONAL_HISTORY_IDENTICAL_ORIGINAL_BOOTSTRAP_DRAWS_REQUIRED')
    original = np.asarray(original_joint_maximum, dtype=float)
    require(original.shape == (5000,) and np.isfinite(original).all() and np.all(original >= 0),
            'CONDITIONAL_HISTORY_BOUND_WHOLE_DOMAIN_MAXIMUM_REQUIRED')
    require(prepared_scores['domain_rows'] == 84330
            and prepared_scores['undefined_domain_rows'] == 1734
            and prepared_scores['coverage']['full_registered_scope_covered'] is True
            and prepared_scores['coverage']['completed_pairs'] == 2304
            and np.array_equal(prepared_scores['source_rng_seeds'], SEEDS),
            'CONDITIONAL_HISTORY_PREPARED_FULL_ORIGINAL_SCOPE_REQUIRED')
    x = np.asarray(prepared_scores['scores'], dtype=float)
    point = np.asarray(prepared_scores['point'], dtype=float)
    identities = prepared_scores['identities']
    require(x.shape == (108, len(SEEDS)) and np.isfinite(x).all()
            and point.shape == (108,) and np.isfinite(point).all() and len(identities) == 108
            and np.array_equal(x[72:], .5*(x[:36]+x[36:72])),
            'CONDITIONAL_HISTORY_FIXED_EQUAL_AB_DIAGNOSTICS_REQUIRED')
    expected_point = x.mean(1)
    constant = np.all(x == x[:, :1], axis=1)
    expected_point[constant] = x[constant, 0]
    require(np.array_equal(point, expected_point), 'CONDITIONAL_HISTORY_BLOCK_MEAN_POINT_REQUIRED')
    summary = pfc_numerical_precision.PrecisionSummary(len(m))
    rows = add_linear_precision(summary, x, point, identities, m)
    joint_maximum = np.maximum(original, summary.maximum)
    _, new_diagnostics = summary.finish()
    diagnostics = dict(original_domain_rows=84330, original_undefined_domain_rows=1734,
        conditional_history_coordinates=108, conditional_history_precision=new_diagnostics,
        bootstrap_replicates=5000, bootstrap_seed=97001, original_independent_blocks=32,
        all_coordinates_share_original_block_multiplicities=True,
        original_whole_domain_joint_maximum_reused=True, frozen_endpoint_scores_replaced=False,
        original_domain_rows_removed=0, diagnostic_subset_maximum_returned=False,
        maximum_is_simultaneous_confidence_statement=False, population_confidence_interval_computed=False,
        bands_computed=False, coverage_qualified=False, confirmation_execution_released=False,
        source_history_population_uncertainty_established=False,
        negative_variance_diagnostics_preserved=True,
        variance_noise_subtraction_applied_to_bootstrap_scores=False,
        threshold2_score_computed=False, unresolved_threshold2_rows=len(prepared_scores['unresolved_threshold2']))
    return dict(joint_maximum=joint_maximum, conditional_history_precision=rows, diagnostics=diagnostics)



def _history_artifact(archive, delivery, name):
    entries = [row for row in delivery['files'] if row['path'] == name]
    require(len(entries) == 1, 'CONDITIONAL_HISTORY_UNIQUE_BOUND_ARTIFACT:'+name)
    path = archive/name
    require(path.stat().st_size == entries[0]['bytes']
            and sha256_file(path) == entries[0]['sha256'],
            'CONDITIONAL_HISTORY_BOUND_ARTIFACT_CHANGED:'+name)
    return path


def run_conditional_history_precision(settings, project_root, output):
    """Consume complete coverage and the existing full-domain joint maximum.

    This extends empirical precision diagnostics. It does not rerun the original
    covariance calculation, replace frozen scores or generate confidence bands.
    Incomplete coverage is rejected before creating an analysis directory.
    """
    validate_settings(settings)
    references = settings['conditional_history']
    require(set(references) == {'coverage_archive', 'history_config_sha256',
                               'joint_precision', 'midpoint_weights'}
            and references['midpoint_weights'] == [.5, .5],
            'CONDITIONAL_HISTORY_EXPLICIT_INPUTS_AND_FIXED_MIDPOINT_REQUIRED')
    root, output = Path(project_root), Path(output)
    archive = root/references['coverage_archive']
    delivery = json.loads((archive/'delivery.json').read_text())
    require(delivery['verified'] is True
            and delivery['status'] == 'COMPLETE_PFC_PROFILE_HISTORY_COVERAGE_COMPLETE',
            'CONDITIONAL_HISTORY_COMPLETED_COVERAGE_DELIVERY_REQUIRED')
    coverage = json.loads(_history_artifact(archive, delivery, 'analysis/report.json').read_text())
    history_path = _history_artifact(archive, delivery, 'coverage_inputs/foundation/source_config.yaml')
    require(sha256_file(history_path) == references['history_config_sha256'],
            'CONDITIONAL_HISTORY_FROZEN_PROFILE_CONFIG_CHANGED')
    require(coverage['status'] == 'COMPLETE_REQUESTED_PFC_PROFILE_HISTORY_COVERAGE'
            and coverage['full_registered_scope_covered'] is True
            and coverage['completed_pairs'] == 2304
            and coverage['requested_source_seeds'] == list(SEEDS)
            and coverage['source_history_population_uncertainty_established'] is False,
            'CONDITIONAL_HISTORY_COMPLETE2304_DELIVERY_REQUIRED')
    history = yaml.safe_load(history_path.read_text())
    donor_ref = references['joint_precision']
    donor_root = root/donor_ref['path']
    donor = bound_report(donor_root, donor_ref['report_sha256'])
    require(donor['status'] == 'COMPLETE_COUPLED_CALIBRATION_JOINT_UNCERTAINTY_DIAGNOSTIC'
            and donor['domain_rows'] == 84330 and donor['undefined_endpoints'] == 1734
            and donor['original_independent_blocks'] == 32
            and donor['all_endpoints_share_identical_original_block_multiplicities'] is True
            and donor['coverage_qualified'] is False
            and donor['settings'] == {key: value for key, value in settings.items()
                                      if key != 'conditional_history'},
            'CONDITIONAL_HISTORY_MATCHING_FULL_DOMAIN_JOINT_PRECISION_REQUIRED')
    domain_ref = settings['inputs']['domain']
    domain_root = root/domain_ref['path']
    domain_report = bound_report(domain_root, domain_ref['report_sha256'])
    domains = read_rows(bound_table(domain_root, domain_report, 'endpoint_domains.csv'))
    margin_ref = settings['inputs']['budget']
    margin_root = root/margin_ref['path']
    margin_report = bound_report(margin_root, margin_ref['report_sha256'])
    margins = read_rows(bound_table(margin_root, margin_report, 'frozen_numerical_margins.csv'))
    require(len(domains) == len(margins) == 84330
            and donor['frozen_margin_sha256'] == margin_report['outputs']['frozen_numerical_margins.csv']['sha256'],
            'CONDITIONAL_HISTORY_MATCHING_FROZEN_DOMAIN_AND_MARGINS_REQUIRED')
    for index, (domain, margin) in enumerate(zip(domains, margins)):
        require(int(margin['domain_row_index']) == index
                and all(domain[key] == margin[key]
                        for key in ('family', 'endpoint_identity', 'definition_status')),
                'CONDITIONAL_HISTORY_FROZEN_MARGIN_IDENTITY_CHANGED')
        domain['units'] = margin['units']
    prepared = conditional_history_scores(history, coverage['records'], domains, SEEDS)
    with np.load(bound_table(donor_root, donor, 'joint_bootstrap_maximum.npz'), allow_pickle=False) as bank:
        require(set(bank.files) == {'direct_maximum', 'joint_maximum', 'bootstrap_multiplicities', 'source_rng_seeds'}
                and np.array_equal(bank['source_rng_seeds'], SEEDS),
                'CONDITIONAL_HISTORY_BOUND_JOINT_DRAW_SCHEMA_REQUIRED')
        result = augment_conditional_history_precision(bank['joint_maximum'], bank['bootstrap_multiplicities'], prepared)
        multiplicities = bank['bootstrap_multiplicities'].copy()
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/'conditional_history_sampling_precision.csv', result['conditional_history_precision'])
    write_rows(output/'unresolved_threshold2.csv', prepared['unresolved_threshold2'])
    np.savez_compressed(output/'conditional_history_joint_precision.npz',
        source_rng_seeds=prepared['source_rng_seeds'], conditional_history_scores=prepared['scores'],
        bootstrap_multiplicities=multiplicities, joint_maximum=result['joint_maximum'])
    report = dict(status='COMPLETE_CONDITIONAL_HISTORY_JOINT_EMPIRICAL_PRECISION',
        settings=settings, **result['diagnostics'],
        registered_pairs=2304, midpoint_weights=[.5, .5],
        conditional_history_block_mean_variance_decomposition=prepared['coverage']['statistics'],
        joint_critical_value_diagnostic=float(np.quantile(result['joint_maximum'], .975, method='higher')),
        original_joint_critical_value_diagnostic=donor['joint_critical_value'],
        score_definition=prepared['score_definition'], new_source_simulations=0,
        new_auxiliary_trials=0, raw_history_payloads_reread=False, production_episodes=0,
        scientific_fidelity_qualified=False, uncertainty_method_adopted=False,
        interpretation='Joint empirical precision on complete archived conditional histories. Nominal source expectation identity does not establish finite PRNG or simultaneous population coverage.',
        implementation_sha256=sha256_file(Path(__file__)), outputs=_outputs(output))
    freeze_json(output/'report.json', report)
    return report


def covariance_precision(left, right, lag, multiplicities, pairs, identities, summary):
    """Recompute order-two draws and delete-one diagnostics on identical blocks."""
    a = pfc_numerical_precision.resampled_lag_covariance(left, lag, multiplicities)
    b = pfc_numerical_precision.resampled_lag_covariance(right, lag, multiplicities)
    require(a is not None and b is not None, 'DEFINED_WITHIN_PHASE_COVARIANCE_REQUIRED')
    h = (lag_pair_kernels(left, lag, pairs)-lag_pair_kernels(right, lag, pairs)).reshape(len(pairs), -1).T
    point = h.mean(1)
    rows = summary.add(point, (a-b).reshape(len(multiplicities), -1),
                       (len(left)-1)/len(left)*point, identities)
    pseudo, se, influence, residual = covariance_jackknife(h, pairs, len(left))
    return rows, pseudo, se, influence, residual


def recompute_precision(domains, fields, responses, cdf, scores, multiplicities, chunk_size, influence_limit):
    """Stream one covariance group or linear endpoint chunk into a joint maximum."""
    n = multiplicities.shape[1]; summary = pfc_numerical_precision.PrecisionSummary(len(multiplicities))
    direct = [None]*len(domains); covariance_groups = defaultdict(dict)
    means = {label: {(condition, phase): fields[label+'_'+condition][:, round(start/BIN_MS):round(stop/BIN_MS)].mean(1)
                     for condition in CONDITIONS for phase, (start, stop) in PHASES.items()} for label in LABELS}
    response_values = {label: responses[label+'_values'] for label in LABELS}
    pending = []
    max_roundtrip = 0.
    for index, row in enumerate(domains):
        family = row['family']; ident = json.loads(row['endpoint_identity'])
        identity = dict(domain_row_index=index, family=family, endpoint_identity=row['endpoint_identity'],
                        definition_status=row['definition_status'], current_empirical_status=row['current_empirical_status'])
        if row['definition_status'] == UNDEFINED:
            direct[index] = summary.add_undefined([identity])[0]; continue
        if family == 'field_lag_covariance':
            lag_ms = float(ident['lag_ms']); lag = round(lag_ms/BIN_MS)
            require(lag >= 0 and lag*BIN_MS == lag_ms, 'INTEGER_READOUT_LAG_REQUIRED')
            group = covariance_groups[ident['condition'], ident['phase'], lag, ident['left_cohort'], ident['right_cohort']]
            ij = int(ident['leading_field']), int(ident['trailing_field'])
            require(ij not in group, 'DUPLICATE_COVARIANCE_FIELD_PAIR'); group[ij] = (index, identity)
            continue
        if family == 'count_cdf':
            ci, pi = CONDITIONS.index(ident['condition']), list(PHASES).index(ident['phase'])
            wi = list(cdf['window_ms']).index(float(ident['window_ms']))
            ki = int(np.searchsorted(cdf['count_support'], int(ident['count_threshold']), side='right')-1)
            require(ki >= 0, 'CDF_THRESHOLD_BELOW_REGISTERED_SUPPORT')
            fi = int(ident['field_index'])
            expected = (cdf[ident['left_cohort']+'_cdf'][ci, :, pi, wi, fi, ki]
                        -cdf[ident['right_cohort']+'_cdf'][ci, :, pi, wi, fi, ki])
        else:
            expected = linear_block_score(family, ident, means, response_values)
        require(np.array_equal(expected, scores[index]), 'EXPORTED_LINEAR_BLOCK_SCORE_MISMATCH:'+str(index))
        point = float(row['estimate']); error = abs(point-float(expected.mean()))
        max_roundtrip = max(max_roundtrip, error)
        require(error <= 1e-12*max(1., abs(point)), 'SIGNED_LINEAR_POINT_MISMATCH:'+str(index))
        pending.append((index, identity))
        if len(pending) == chunk_size:
            indices, identities = zip(*pending)
            rows = add_linear_precision(summary, scores[list(indices)], [float(domains[i]['estimate']) for i in indices], identities, multiplicities)
            for i, item in zip(indices, rows): direct[i] = item
            pending.clear()
    if pending:
        indices, identities = zip(*pending)
        rows = add_linear_precision(summary, scores[list(indices)], [float(domains[i]['estimate']) for i in indices], identities, multiplicities)
        for i, item in zip(indices, rows): direct[i] = item
    pairs, _ = round_robin_pairings(n); influence_rows = []
    for (condition, phase, lag, left, right), group in covariance_groups.items():
        ordered = [(i, j) for i in range(len(FIELD_SIZES)) for j in range(len(FIELD_SIZES))]
        require(set(group) == set(ordered), 'FULL_DIRECTED_COVARIANCE_DOMAIN_REQUIRED')
        slots, identities = zip(*(group[ij] for ij in ordered))
        start, stop = PHASES[phase]
        a, b = (fields[label+'_'+condition][:, round(start/BIN_MS):round(stop/BIN_MS)] for label in (left, right))
        rows, pseudo, jackknife_se, influence, residual = covariance_precision(a, b, lag, multiplicities, pairs, identities, summary)
        require(np.allclose(pseudo, scores[list(slots)], rtol=1e-11, atol=1e-12),
                'EXPORTED_COVARIANCE_PSEUDOVALUE_MISMATCH')
        for j, (slot, item) in enumerate(zip(slots, rows)):
            expected = float(domains[slot]['estimate']); error = abs(item['estimate']-expected)
            max_roundtrip = max(max_roundtrip, error)
            require(error <= 1e-11*max(1., abs(expected)), 'SIGNED_COVARIANCE_POINT_MISMATCH:'+str(slot))
            direct[slot] = item; bootstrap_se = item['bootstrap_standard_error']
            influence_rows.append(dict(domain_row_index=slot, endpoint_identity=domains[slot]['endpoint_identity'],
                bootstrap_standard_error=bootstrap_se, jackknife_standard_error=float(jackknife_se[j]),
                jackknife_to_bootstrap_se_ratio=float(jackknife_se[j]/bootstrap_se) if bootstrap_se > 0 else '',
                largest_block_squared_influence_fraction=float(influence[j]) if np.isfinite(influence[j]) else '',
                pair_residual_energy_fraction=float(residual[j]) if np.isfinite(residual[j]) else '',
                influence_flag=bool(influence[j] >= influence_limit), first_order_degenerate=bool(jackknife_se[j] == 0),
                bootstrap_degenerate=bool(bootstrap_se == 0)))
    require(all(row is not None for row in direct), 'COMPLETE_DIRECT_BOOTSTRAP_DOMAIN_REQUIRED')
    maximum, diagnostic = summary.finish()
    require(np.isfinite(maximum).all(), 'NONEMPTY_POSITIVE_SPREAD_BOOTSTRAP_REQUIRED')
    return direct, maximum, influence_rows, diagnostic, max_roundtrip


def precision_lookup(domains, direct):
    require(len(domains) == len(direct), 'FULL_DIRECT_PRECISION_DOMAIN_REQUIRED')
    precision = {}
    for index, (domain, row) in enumerate(zip(domains, direct)):
        require(int(row['domain_row_index']) == index and row['endpoint_identity'] == domain['endpoint_identity'],
                'ORDERED_DIRECT_PRECISION_IDENTITIES_REQUIRED')
        key = identity_key(domain['family'], json.loads(domain['endpoint_identity']))
        require(key not in precision, 'UNIQUE_DIRECT_PRECISION_REQUIRED'); precision[key] = row
    return precision


def frozen_budget_assessment(margins, bands, planning_blocks):
    """Consume the frozen table directly; no reference scales are recomputed."""
    return assess_planning(margins, bands, planning_blocks)


def load_inputs(settings, project_root):
    roots = {key: project_root/ref['path'] for key, ref in settings['inputs'].items()}
    reports = {key: bound_report(roots[key], ref['report_sha256']) for key, ref in settings['inputs'].items()}
    coupled, native, domain = reports['coupled'], reports['resampling_design'], reports['domain']
    require(coupled['status'] == 'COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON'
            and coupled['original_independent_blocks'] == 32 and coupled['independent_circuit_instances'] == 1
            and coupled['source_case_count'] == 192 and coupled['row_count'] == 84330
            and coupled['phase_boundaries_ms'] == PHASES
            and coupled['settings']['source_rng_seeds'] == settings['joint_rng_seeds']
            and coupled['confirmation_execution_released'] is False and coupled['scientific_fidelity_qualified'] is False,
            'COMPLETE_UNQUALIFIED_FIXED_COUPLED_BANK_REQUIRED')
    require(domain['status'] == 'COMPLETE_CALIBRATION_BLOCK_CDF_SCORES'
            and domain['domain_rows'] == 84330
            and domain['settings']['joint_rng_seeds'] == settings['joint_rng_seeds']
            and reports['linear']['status'] == 'COMPLETE_CALIBRATION_LINEAR_BLOCK_SCORES'
            and native['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
            and reports['budget']['numerical_margins_frozen'] is True
            and reports['budget']['planning_blocks'] == settings['planning_blocks'],
            'BOUND_FROZEN_DOMAIN_AND_CALIBRATION_DESIGN_REQUIRED')
    for source_key, input_key in [('domain', 'domain'), ('budget', 'budget'), ('native_comparison', 'resampling_design')]:
        require(coupled['settings'][source_key] == settings['inputs'][input_key], 'COUPLED_INPUT_PROVENANCE_CHANGED:'+source_key)
    require(reports['budget']['settings']['inputs']['domain'] == settings['inputs']['domain'], 'FROZEN_BUDGET_DOMAIN_CHANGED')
    domain_path = bound_table(roots['domain'], domain, 'endpoint_domains.csv')
    require(reports['linear']['domain_table_sha256'] == domain['outputs']['endpoint_domains.csv']['sha256'],
            'PHYSICAL_RESPONSE_RANGE_DOMAIN_CHANGED')
    domains = read_rows(domain_path)
    margin_path = bound_table(roots['budget'], reports['budget'], 'frozen_numerical_margins.csv')
    margins = read_rows(margin_path)
    arrays = {key: load_npz(roots['coupled']/name, coupled) for key, name in {
        'fields': 'coupled_field_rates.npz', 'responses': 'coupled_responses.npz',
        'cdf': 'coupled_count_cdfs.npz', 'scores': 'coupled_block_scores.npz'}.items()}
    require(len(domains) == len(margins) == 84330, 'ALL_FROZEN_ENDPOINTS_REQUIRED')
    validate_axes(**arrays, seeds=settings['joint_rng_seeds'], rows=len(domains))
    statistics_path = bound_table(roots['coupled'], coupled, 'endpoint_precision_comparison.csv')
    with statistics_path.open(newline='', encoding='utf-8') as stream:
        current = bind_endpoints(domains, margins, csv.DictReader(stream), arrays['scores']['scores'], len(SEEDS))
    physical = {}
    for row in domains:
        if row['family'] == 'count_cdf':
            ident = json.loads(row['endpoint_identity']); key = float(ident['window_ms']), int(ident['field_index'])
            maximum = int(row['physical_maximum_count'])
            require(key not in physical or physical[key] == maximum, 'FROZEN_PHYSICAL_SUPPORT_CHANGED')
            physical[key] = maximum
    require(set(physical) == {(float(width), field) for width in [5, 50, 100] for field in range(17)},
            'FULL_PHYSICAL_COUNT_DOMAIN_REQUIRED')
    multiplicities = shared_multiplicities(settings, native)
    return current, margins, arrays, physical, multiplicities, reports


def _outputs(output):
    return {path.name: dict(bytes=path.stat().st_size, sha256=sha256_file(path))
            for path in sorted(output.iterdir()) if path.is_file() and path.name != 'report.json'}


def run(config_path, project_root, output):
    settings = yaml.safe_load(config_path.read_text()); validate_settings(settings)
    if 'conditional_history' in settings:
        return run_conditional_history_precision(settings, project_root, output)
    domains, margins, arrays, physical, multiplicities, reports = load_inputs(settings, project_root)
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    report = dict(settings=settings, settings_sha256=sha256_file(config_path), domain_rows=len(domains),
        original_independent_blocks=len(SEEDS), independent_circuit_instances=1, bootstrap_replicates=len(multiplicities),
        covariance_overlapping_pairs=len(SEEDS)*(len(SEEDS)-1)//2,
        all_endpoints_share_identical_original_block_multiplicities=True,
        native_resampling_design_reused=True, native_maximum_or_bootstrap_statistics_reused=False,
        frozen_definition_status_counts=dict(Counter(row['definition_status'] for row in domains)),
        current_empirical_status_counts=dict(Counter(row['current_empirical_status'] for row in domains)),
        numerical_margins_frozen=True, frozen_margins_recomputed=False,
        frozen_margin_sha256=reports['budget']['outputs']['frozen_numerical_margins.csv']['sha256'],
        nominal_alpha=settings['nominal_alpha'], joint_tail_alpha=settings['joint_tail_alpha'],
        bootstrap_alpha=settings['nominal_alpha']-settings['joint_tail_alpha'], planning_blocks=settings['planning_blocks'],
        coverage_qualified=False, uncertainty_coverage_qualified=False, finite_sample_distribution_free_guarantee=False,
        uncertainty_method_adopted=False, method_adopted_for_confirmation=False, confirmation_n_selected=None,
        confirmation_execution_released=False, scientific_fidelity_qualified=False, numerical_equivalence_qualified=False,
        new_source_simulations=0, raw_spike_data_reread=False, production_episodes=0,
        matched_end_to_end_speedup_qualified=False, observed_saturation_is_source_identity=False,
        empirical_degeneracy_is_zero_population_uncertainty=False, influences_are_exclusions=False,
        inference='approximate_joint_bootstrap_plus_fixed_joint_departure_event',
        assumptions='IID whole blocks and adequate joint bootstrap approximation. Positive spread or influence diagnostics do not establish coverage.',
        interpretation='Calibration diagnostic against unchanged margins. Neither projected bands nor their budget relations qualify fidelity or confirmation. Inadequate precision is not proof of simulator inaccuracy or infeasibility.',
        planning_projection='Root-n nondegenerate widths and zero joint departures at128, with calibration centers/critical value fixed. Not future evidence, selected n or power assurance.',
        delay_boundary_semantics=reports['coupled']['delay_boundary_semantics'],
        python_version=platform.python_version(), numpy_version=np.__version__,
        implementation_sha256=sha256_file(Path(__file__)), precision_implementation_sha256=sha256_file(Path(pfc_numerical_precision.__file__)))
    try:
        direct, previous, influences, diagnostic, roundtrip = recompute_precision(
            domains, arrays['fields'], arrays['responses'], arrays['cdf'], arrays['scores']['scores'], multiplicities,
            settings['endpoint_chunk_size'], settings['single_block_influence_fraction_flag'])
        write_rows(output/'endpoint_sampling_precision.csv', direct)
        write_rows(output/'covariance_influence.csv', influences)
        report.update(direct_bootstrap=diagnostic, maximum_exported_point_roundtrip_error=roundtrip,
                      covariance_influence_recomputed=True, bootstrap_draws_recomputed=True)
        precision = precision_lookup(domains, direct)
        unsupported = [row for row in direct if row['family'] in ('field_mean', 'field_lag_covariance')
                       and row['definition_status'] not in (UNDEFINED, STRUCTURAL) and row['status'] != POSITIVE]
        if unsupported:
            write_rows(output/'unsupported_defined_endpoints.csv', unsupported)
            report['unsupported_defined_endpoints'] = len(unsupported)
        require(not unsupported, 'UNRESOLVED_NONSTRUCTURAL_FIELD_OR_COVARIANCE_DEGENERACY')
        values, identities, lookup, physical_lo, physical_hi = prepare_primitives(
            arrays['cdf'], arrays['responses'], LABELS, physical, reports['linear']['per_clock_response_physical_ranges_hz'])
        require(np.all(values >= physical_lo[:, None]) and np.all(values <= physical_hi[:, None]),
                'PRIMITIVE_PHYSICAL_SUPPORT_CONTRADICTION')
        point, spread, maximum, influence = bootstrap_primitives(values, multiplicities, previous, settings['endpoint_chunk_size'])
        selected = (spread == 0) & (physical_lo < physical_hi)
        require(not np.any(frozen_tail_departures(values[selected], point[selected])), 'CALIBRATION_CONSTANT_SELECTION_CONTRADICTED')
        critical = float(np.quantile(maximum, 1-report['bootstrap_alpha'], method='higher'))
        tail = binomial_probability_upper(0, len(SEEDS), settings['joint_tail_alpha'])
        future_tail = binomial_probability_upper(0, settings['planning_blocks'], settings['joint_tail_alpha'])
        primitive = point, spread, physical_lo, physical_hi
        bands, lo, hi = construct_bands(domains, precision, primitive, lookup, arrays['cdf']['count_support'].tolist(), physical, critical, tail)
        projected, _, _ = construct_bands(domains, precision, primitive, lookup, arrays['cdf']['count_support'].tolist(), physical,
                                          critical*np.sqrt(len(SEEDS)/settings['planning_blocks']), future_tail)
        for row, future, domain in zip(bands, projected, domains):
            row['current_empirical_status'] = domain['current_empirical_status']
            row['frozen_native_estimate'] = domain['frozen_native_estimate']
            row['planning_lower_root_n_and_zero_tail_at_128'] = future['lower']
            row['planning_upper_root_n_and_zero_tail_at_128'] = future['upper']
            if row['definition_status'] not in (STRUCTURAL, UNDEFINED):
                require(row['lower'] < row['upper'], 'NONSTRUCTURAL_ENDPOINT_CANNOT_HAVE_ZERO_UNCERTAINTY')
        write_rows(output/'endpoint_bands.csv', bands)
        primitive_rows = [dict(primitive_identity=json.dumps(ident, sort_keys=True), estimate=float(point[i]),
            bootstrap_standard_error=float(spread[i]), physical_lower=float(physical_lo[i]), physical_upper=float(physical_hi[i]),
            lower=float(lo[i]), upper=float(hi[i]), largest_block_squared_influence_fraction=float(influence[i]) if np.isfinite(influence[i]) else '',
            calibration_constant_tail_candidate=bool(selected[i]), joint_tail_probability_upper=tail if selected[i] else '',
            status='source_identity' if physical_lo[i] == physical_hi[i] else 'joint_event_tail_bound_candidate' if selected[i] else 'nondegenerate_bootstrap_anchor')
            for i, ident in enumerate(identities)]
        write_rows(output/'primitive_bands.csv', primitive_rows)
        np.savez_compressed(output/'joint_bootstrap_maximum.npz', direct_maximum=previous, joint_maximum=maximum,
                            bootstrap_multiplicities=multiplicities, source_rng_seeds=np.asarray(SEEDS))
        planning, groups = frozen_budget_assessment(margins, bands, settings['planning_blocks'])
        write_rows(output/'calibration_budget_assessment.csv', planning)
        report.update(status='COMPLETE_COUPLED_CALIBRATION_JOINT_UNCERTAINTY_DIAGNOSTIC',
            defined_endpoints=sum(row['definition_status'] != UNDEFINED for row in bands),
            undefined_endpoints=sum(row['definition_status'] == UNDEFINED for row in bands),
            channels=dict(Counter(row['channel'] for row in bands)),
            current_empirical_degenerate_endpoints_with_nonzero_width=sum(row['current_empirical_status'] == EMPIRICAL and row['lower'] < row['upper'] for row in bands),
            frozen_empirical_degenerate_endpoints_with_nonzero_width=sum(row['definition_status'] == EMPIRICAL and row['lower'] < row['upper'] for row in bands),
            joint_critical_value=critical, direct_critical_value=float(np.quantile(previous, 1-report['bootstrap_alpha'], method='higher')),
            primitive_endpoints=len(values), primitive_status_counts=dict(Counter(row['status'] for row in primitive_rows)),
            primitive_single_block_influence_flags=int(np.count_nonzero(influence >= settings['single_block_influence_fraction_flag'])),
            covariance_single_block_influence_flags=sum(row['influence_flag'] for row in influences),
            covariance_first_order_degenerate=sum(row['first_order_degenerate'] for row in influences),
            covariance_bootstrap_degenerate=sum(row['bootstrap_degenerate'] for row in influences),
            joint_tail_candidate_primitives=int(selected.sum()), joint_tail_calibration_departures_by_selection=0,
            joint_tail_probability_upper_calibration=tail, joint_tail_probability_upper_zero_departure_planning=future_tail,
            joint_tail_is_one_event_per_independent_block=True, joint_tail_primitives_are_independent_replicates=False,
            joint_tail_requires_route_frozen_before_confirmation=True, calibration_tail_bound_has_confirmatory_coverage=False,
            planning_status_counts=dict(Counter(row['planning_status'] for row in planning)), planning_by_clock_pair_and_family=groups,
            calibration_interval_relations=dict(Counter(row['calibration_interval_relation'] for row in planning if row['calibration_interval_relation'])),
            planning_interval_relations=dict(Counter(row['planning_interval_relation'] for row in planning if row['planning_interval_relation'])),
            calibration_points_outside_budget=sum(row['calibration_point_outside_margin'] is True for row in planning))
    except (ValueError, FloatingPointError) as error:
        report.update(status='FAILED_COUPLED_CALIBRATION_UNCERTAINTY_DIAGNOSTIC', error=str(error),
                      outputs=_outputs(output), wall_seconds=time.monotonic()-started)
        freeze_json(output/'report.json', report)
        raise
    report.update(outputs=_outputs(output), wall_seconds=time.monotonic()-started)
    freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = run(args.config, args.project_root, args.output)
    print(json.dumps({key: report[key] for key in ('status', 'domain_rows', 'joint_critical_value', 'planning_status_counts', 'registered_pairs', 'joint_critical_value_diagnostic') if key in report}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

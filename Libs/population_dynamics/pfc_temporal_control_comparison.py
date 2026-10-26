"""Describe a same-clock temporal-order intervention on one reused PFC block.

The source job replays the original coarse and auxiliary RNG streams and reverses
child rows within each coarse tick. This reader reuses the accepted original
readouts and reconstructs only the reversed source's readouts. Neither the two
clocks nor the three stimulus conditions are additional independent blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_count_histograms
from . import pfc_coupled_clock_comparison as comparison
from . import pfc_coupled_clock_execution as execution
from .pfc_delay_analysis import PHASES, write_rows
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS
from .pfc_scale_analysis import _source_bound_file, _source_manifest, _source_pairing
from .provenance import freeze_json, sha256_file

SCHEMA = 'pfc_temporal_control_comparison_v1'
SCOPE = 'single_reused_block_descriptive'
STATUS = 'COMPLETE_DESCRIPTIVE_PFC_TEMPORAL_CONTROL_COMPARISON'
SOURCE_STATUS = 'COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL'
ORIGINAL_STATUS = 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT'
ORIGINAL_COMPARISON_STATUS = 'COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_COMPARISON'
SEED = 95001
CLOCKS = (('numpy_dt_0p025', .025), ('numpy_dt_0p0125', .0125))
PAIRS = tuple(('reversed_'+label, 'original_'+label) for label, _ in CLOCKS)
SCHEME = 'within_coarse_tick_tree_temporal_reversal'
FIELD_SIZES = np.asarray([5000]*16+[20000], dtype=np.int64)
WINDOWS_MS = (5, 50, 100)
INPUT_CHECKS = (
    'initial_voltage_equal', 'all_coarse_and_auxiliary_final_rng_equal',
    'full_fine_clock_applied_current_equal', 'population_external_event_totals_equal',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_settings(settings):
    require(isinstance(settings, dict) and settings.get('schema') == SCHEMA
            and settings.get('scope') == SCOPE
            and settings.get('confirmation_execution_released') is False
            and settings.get('scientific_fidelity_qualified') is False
            and type(settings.get('production_episodes')) is int
            and settings['production_episodes'] == 0,
            'REGISTERED_SINGLE_BLOCK_DESCRIPTIVE_SCOPE_REQUIRED')
    require(all(settings.get(key, False) is False for key in
                ('confidence_intervals', 'covariance_estimates', 'fdr_tests', 'frozen_margin_tests')),
            'SINGLE_REUSED_BLOCK_HAS_NO_INFERENTIAL_QUALIFICATION')
    require(type(settings.get('source_job_id')) is int and settings['source_job_id'] > 0
            and isinstance(settings.get('source_archive'), str) and settings['source_archive'],
            'REGISTERED_TEMPORAL_SOURCE_ARCHIVE_REQUIRED')
    for name, path_key, hash_keys in (
        ('protocol', 'path', ('sha256',)),
        ('original_source', 'archive', ('delivery_sha256',)),
        ('original_comparison', 'archive', ('delivery_sha256', 'report_sha256')),
    ):
        row = settings.get(name, {})
        require(isinstance(row, dict) and isinstance(row.get(path_key), str) and row[path_key]
                and all(_sha(row.get(key)) for key in hash_keys),
                'HASH_BOUND_TEMPORAL_INPUT_REQUIRED:'+name)


def read_archive(root, status, *, expected_sha256=None, job_id=None):
    delivery_path = root/'delivery.json'
    actual_sha256 = sha256_file(delivery_path)
    require(expected_sha256 is None or actual_sha256 == expected_sha256,
            'TEMPORAL_INPUT_DELIVERY_CHANGED')
    delivery = json.loads(delivery_path.read_text())
    require(delivery.get('status') == status and delivery.get('verified') is True
            and (job_id is None or str(delivery.get('job_id')) == str(job_id)),
            'COMPLETE_VERIFIED_TEMPORAL_INPUT_REQUIRED')
    return _source_manifest(delivery), actual_sha256


def bound_output(root, manifest, relative, report, name):
    require(Path(name).name == name and name in report.get('outputs', {}),
            'REGISTERED_READOUT_OUTPUT_REQUIRED:'+name)
    path = _source_bound_file(root, manifest, str(relative/name))
    record = report['outputs'][name]
    require(path.stat().st_size == record.get('bytes') and sha256_file(path) == record.get('sha256'),
            'READOUT_REPORT_OUTPUT_BINDING_CHANGED:'+name)
    return path


def accepted_original_cases(report, original_reference):
    require(report.get('status') == ORIGINAL_COMPARISON_STATUS
            and report.get('selected_source_rng_seed') == SEED
            and report.get('original_independent_blocks') == 1
            and report.get('pilot_delivery_sha256') == original_reference['delivery_sha256']
            and report.get('settings', {}).get('pilot_archive') == original_reference['archive'],
            'ACCEPTED_ORIGINAL_COMPARISON_BINDING_REQUIRED')
    cases = report.get('pilot_cases', [])
    expected = {(dt, f'pfc_{SEED}_{condition}') for _, dt in CLOCKS for condition in CONDITIONS}
    indexed = {(row.get('dt_ms'), row.get('case_id')): row for row in cases}
    require(len(cases) == len(indexed) == 6 and set(indexed) == expected
            and all(_sha(row.get('report_sha256')) for row in cases),
            'ALL_SIX_ACCEPTED_ORIGINAL_CASES_REQUIRED')
    return indexed


def validate_original_case(report, dt, condition):
    require(report.get('status') == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE'
            and report.get('source_execution_complete') is True
            and report.get('source_rng_seed') == SEED
            and report.get('refinement_seeds') == [95411, 95412]
            and report.get('case_id') == f'pfc_{SEED}_{condition}'
            and report.get('condition') == condition and report.get('dt_ms') == dt
            and report.get('coarse_dt_ms') == .05 and report.get('coupled_core_units') == 100000
            and report.get('duration_ms') == 5000.
            and not report.get('drive', {}).get('coupling', {}).get('input_order_control'),
            'EXACT_ACCEPTED_ORIGINAL_CASE_REQUIRED')
    config = report.get('resolved_config', {})
    require(config.get('dt_ms') == dt and config.get('duration_ms') == 5000.
            and config.get('readout_bin_ms') == 5. and config.get('excitatory_neurons') == 80000
            and config.get('inhibitory_neurons') == 20000 and config.get('excitatory_preference_bins') == 16,
            'ORIGINAL_OBSERVATION_MAP_CHANGED')


def validate_temporal_case(report, original, protocol, protocol_sha256, original_report_sha256, dt, condition):
    validate_original_case(original, dt, condition)
    require(report.get('status') == 'COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE'
            and report.get('source_execution_complete') is True
            and report.get('scope') == execution.TEMPORAL_CONTROL_SCOPE
            and report.get('source_rng_seed') == SEED
            and report.get('refinement_seeds') == [95411, 95412]
            and report.get('dt_ms') == dt and report.get('coarse_dt_ms') == .05
            and report.get('condition') == condition and report.get('case_id') == f'pfc_{SEED}_{condition}'
            and report.get('coupled_core_units') == 100000 and report.get('duration_ms') == 5000.
            and report.get('settings_sha256') == protocol_sha256
            and report.get('original_fine_reference') == protocol['original_fine_reference']
            and report.get('original_fine_report_sha256') == original_report_sha256
            and report.get('temporal_transform') == SCHEME,
            'EXACT_SAME_CLOCK_TEMPORAL_CASE_REQUIRED')
    require(all(report.get(key) == original.get(key) for key in
                ('resolved_config', 'selected_case', 'coarse_reference'))
            and report.get('coarse_reference') == protocol['coarse_references'][condition]
            and report.get('new_original_independent_blocks') == 0
            and report.get('reused_original_independent_blocks') == 1
            and report.get('new_auxiliary_streams') == 0
            and report.get('rng_streams_replayed') is True
            and report.get('independent_auxiliary_repeat') is False
            and report.get('confirmation_execution_released') is False
            and report.get('scientific_fidelity_qualified') is False
            and report.get('production_episodes') == 0,
            'REPLAYED_TEMPORAL_CASE_SCOPE_OR_CONFIG_CHANGED')
    checks = report.get('original_fine_input_agreement', {})
    require(set(checks) == set(INPUT_CHECKS) and all(checks[key] is True for key in INPUT_CHECKS)
            and report.get('coarse_final_rng_matches_archived_episode') is True
            and bool(report.get('coarse_current_agreement'))
            and all(value is True for value in report['coarse_current_agreement'].values())
            and bool(report.get('validations'))
            and all(row.get('passed') is True for row in report['validations'].values()),
            'COMPLETE_TEMPORAL_INPUT_AND_SOURCE_VALIDATION_REQUIRED')
    inputs = dict(report.get('inputs_used', {}))
    control = inputs.pop('input_order_control', None)
    require(bool(control) and control == report.get('drive', {}).get('coupling', {}).get('input_order_control')
            and inputs == original.get('inputs_used')
            and report.get('drive', {}).get('external_event_totals')
                == original.get('drive', {}).get('external_event_totals'),
            'SAME_CLOCK_INPUT_CONSUMPTION_CHANGED')
    # The source adapter changes temporal row order; the numerical engine and
    # original binomial refinement kernel retain the accepted implementation.
    original_code, changed_code = original.get('code', {}), report.get('code', {})
    for module in (execution.executor, execution.native_drive, execution.native_execution,
                   execution.pfc_clock_coupling):
        name = module.__name__.split('.')[-1]
        require(_sha(original_code.get(name)) and changed_code.get(name) == original_code[name],
                'ORIGINAL_NUMERICAL_KERNEL_CHANGED:'+name)
    require(_sha(changed_code.get('pfc_clock_temporal_control')), 'TEMPORAL_ADAPTER_IMPLEMENTATION_UNBOUND')


def validate_fields(fields):
    counts, rates = fields['field_counts'], fields['rates_hz']
    require(np.array_equal(fields['field_unit_counts'], FIELD_SIZES)
            and np.array_equal(fields['field_indices'], np.arange(17))
            and np.array_equal(fields['time_start_ms'], np.arange(1000)*5.),
            'TEMPORAL_FIELD_AXIS_CHANGED')
    require(counts.shape == rates.shape == (1000, 17)
            and counts.dtype.kind in 'iu' and np.all(counts >= 0)
            and np.all(counts <= np.iinfo(np.int64).max)
            and np.isfinite(rates).all()
            and np.array_equal(rates, counts/(FIELD_SIZES[None, :]*.005)),
            'EXACT_FIELD_COUNT_RATE_RECONSTRUCTION_REQUIRED')
    return counts.astype(np.int64, copy=False), rates


def read_histogram(path):
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    validate_histogram(arrays)
    return arrays


def validate_histogram(arrays):
    require(arrays['coordinate_axes'].tolist() == list(pfc_count_histograms.AXES)
            and arrays['condition_names'].tolist() == list(CONDITIONS)
            and arrays['source_rng_seeds'].tolist() == [SEED]
            and arrays['phase_names'].tolist() == list(PHASES)
            and np.array_equal(arrays['phase_bounds_ms'], list(PHASES.values()))
            and arrays['window_ms'].tolist() == list(WINDOWS_MS)
            and np.array_equal(arrays['field_unit_counts'], FIELD_SIZES),
            'EXACT_COUNT_HISTOGRAM_AXES_REQUIRED')
    coordinates, observations = arrays['coordinates'], arrays['observations']
    shape = (3, 1, 5, 3, 17)
    require(coordinates.ndim == 2 and coordinates.shape[1] == 6 and len(coordinates) > 0
            and coordinates.dtype.kind in 'iu' and np.all(coordinates >= 0)
            and np.all(coordinates[:, :5] < np.asarray(shape))
            and observations.shape == (len(coordinates),) and observations.dtype.kind in 'iu'
            and np.all(observations > 0)
            and len({tuple(row) for row in coordinates.tolist()}) == len(coordinates),
            'UNIQUE_INTEGER_HISTOGRAM_MASS_REQUIRED')
    expected = np.empty(shape, dtype=np.int64)
    for pi, (start, stop) in enumerate(PHASES.values()):
        for wi, width in enumerate(WINDOWS_MS):
            expected[:, :, pi, wi] = round((stop-start)/width)*FIELD_SIZES
    require(arrays['expected_observations'].dtype.kind in 'iu'
            and np.all(observations <= expected[tuple(coordinates[:, :5].T)]),
            'BOUNDED_INTEGER_HISTOGRAM_DENOMINATORS_REQUIRED')
    observed = np.zeros(shape, dtype=np.int64)
    np.add.at(observed, tuple(coordinates[:, :5].T), observations)
    require(np.array_equal(arrays['expected_observations'], expected)
            and np.array_equal(observed, expected),
            'COMPLETE_UNIT_WINDOW_DENOMINATORS_REQUIRED')


def histogram_masses(histogram, support):
    """Place integer masses on complete common support, retaining zero entries."""
    coordinates, observations = histogram['coordinates'], histogram['observations']
    require(support.dtype.kind in 'iu' and support.ndim == 1 and len(support) > 0
            and np.array_equal(support, np.arange(len(support)))
            and int(coordinates[:, 5].max()) <= support[-1], 'COMPLETE_INTEGER_MASS_SUPPORT_REQUIRED')
    result = np.zeros((*histogram['expected_observations'].shape, len(support)), dtype=np.int64)
    np.add.at(result, tuple(coordinates.T), observations)
    return result[:, 0]


def joint_support(old_support, histograms):
    require(old_support.ndim == 1 and old_support.dtype.kind in 'iu' and len(old_support) > 0
            and np.array_equal(old_support, np.arange(len(old_support))),
            'COMPLETE_ORIGINAL_INTEGER_SUPPORT_REQUIRED')
    maximum = max(int(old_support[-1]), *(int(row['coordinates'][:, 5].max()) for row in histograms))
    return np.arange(maximum+1, dtype=np.int64)


def bind_rows(rows, case_inputs, protocol_sha256):
    """Every scalar points to the accepted pair of source cases producing it."""
    identities = {('reversed_'+label, 'original_'+label): dt for label, dt in CLOCKS}
    contrasts = {name: (positive, negative) for name, positive, negative in CONTRASTS}
    result = []
    for row in rows:
        pair = row['left_cohort'], row['right_cohort']
        require(pair in identities, 'SAME_CLOCK_PAIRING_REQUIRED')
        dt = identities[pair]
        conditions = (row['condition'],) if 'condition' in row else contrasts[row['contrast']]
        refs = [case_inputs[(dt, condition)] for condition in conditions]
        result.append(dict(row, dt_ms=dt, source_rng_seed=SEED, protocol_sha256=protocol_sha256,
            original_case_report_sha256=';'.join(ref['original_report_sha256'] for ref in refs),
            reversed_case_report_sha256=';'.join(ref['reversed_report_sha256'] for ref in refs),
            case_ids=';'.join(ref['case_id'] for ref in refs)))
    return result


def run(config_path, project_root, output):
    config_path, project_root, output = map(Path, (config_path, project_root, output))
    started = time.perf_counter()
    settings = yaml.safe_load(config_path.read_text())
    validate_settings(settings)
    protocol_path = project_root/settings['protocol']['path']
    protocol_sha256 = sha256_file(protocol_path)
    require(protocol_sha256 == settings['protocol']['sha256'], 'REGISTERED_TEMPORAL_PROTOCOL_CHANGED')
    protocol = yaml.safe_load(protocol_path.read_text())
    require(protocol.get('schema') == execution.TEMPORAL_CONTROL_SCHEMA,
            'TEMPORAL_SOURCE_PROTOCOL_REQUIRED')
    execution.case_grid(protocol)
    require(protocol['original_fine_reference'] == settings['original_source'],
            'ORIGINAL_SOURCE_PROTOCOL_BINDING_CHANGED')

    source_root = project_root/settings['source_archive']
    source_manifest, source_delivery_sha256 = read_archive(source_root, SOURCE_STATUS, job_id=settings['source_job_id'])
    source_report = json.loads(_source_bound_file(source_root, source_manifest, 'report.json').read_text())
    require(source_report.get('status') == SOURCE_STATUS and source_report.get('exit_code') == 0
            and source_report.get('tests_only') is False
            and str(source_report.get('job_id')) == str(settings['source_job_id']),
            'COMPLETE_TEMPORAL_SOURCE_WRAPPER_REQUIRED')
    delivered_protocol = _source_bound_file(source_root, source_manifest, settings['protocol']['path'])
    require(sha256_file(delivered_protocol) == protocol_sha256, 'DELIVERED_TEMPORAL_PROTOCOL_CHANGED')

    original_root = project_root/settings['original_source']['archive']
    original_manifest, original_delivery_sha256 = read_archive(original_root, ORIGINAL_STATUS,
        expected_sha256=settings['original_source']['delivery_sha256'])
    previous_root = project_root/settings['original_comparison']['archive']
    previous_manifest, previous_delivery_sha256 = read_archive(previous_root, ORIGINAL_COMPARISON_STATUS,
        expected_sha256=settings['original_comparison']['delivery_sha256'])
    previous_report_path = _source_bound_file(previous_root, previous_manifest, 'analysis/report.json')
    require(sha256_file(previous_report_path) == settings['original_comparison']['report_sha256'],
            'ACCEPTED_ORIGINAL_COMPARISON_REPORT_CHANGED')
    previous_report = json.loads(previous_report_path.read_text())
    accepted = accepted_original_cases(previous_report, settings['original_source'])

    def previous_output(name):
        return bound_output(previous_root, previous_manifest, Path('analysis'), previous_report, name)

    with np.load(previous_output('phase_responses.npz'), allow_pickle=False) as stored:
        old_values = {label: stored['coupled_'+label] for label, _ in CLOCKS}
    with np.load(previous_output('pilot_count_cdfs.npz'), allow_pickle=False) as stored:
        old_support = stored['count_support']
        old_cdfs = {label: stored['coupled_'+label] for label, _ in CLOCKS}
    require(old_support.tolist() == previous_report.get('empirical_count_support'),
            'ORIGINAL_COUNT_SUPPORT_REPORT_CHANGED')
    old_histograms = {label: read_histogram(previous_output(label+'_count_histograms.npz'))
                      for label, _ in CLOCKS}
    fields, responses, case_inputs, paired_checks = {}, {}, {}, []
    new_histograms = {label: {} for label, _ in CLOCKS}
    counts_original, counts_reversed = [], []
    input_refs = []
    for label, dt in CLOCKS:
        original_label, reversed_label = 'original_'+label, 'reversed_'+label
        fields[original_label], fields[reversed_label] = {}, {}
        responses[original_label] = old_values[label]
        require(responses[original_label].shape == (3, 5, 8)
                and np.isfinite(responses[original_label]).all(), 'COMPLETE_ACCEPTED_ORIGINAL_RESPONSES_REQUIRED')
        responses[reversed_label] = np.empty((3, 5, 8))
        old_counts, new_counts, paired = [], [], {}
        for ci, condition in enumerate(CONDITIONS):
            relative = Path('source_cases')/execution.case_key(dt, condition, SEED)
            old_report_path = _source_bound_file(original_root, original_manifest, str(relative/'report.json'))
            old_report_sha256 = sha256_file(old_report_path)
            require(old_report_sha256 == accepted[(dt, f'pfc_{SEED}_{condition}')]['report_sha256'],
                    'ORIGINAL_CASE_NOT_BOUND_TO_ACCEPTED_READOUT')
            original = json.loads(old_report_path.read_text())
            changed_report_path = _source_bound_file(source_root, source_manifest, str(relative/'report.json'))
            changed = json.loads(changed_report_path.read_text())
            validate_temporal_case(changed, original, protocol, protocol_sha256, old_report_sha256, dt, condition)
            original_fields_path = bound_output(original_root, original_manifest, relative, original, 'field_observations.npz')
            with np.load(original_fields_path, allow_pickle=False) as stored:
                old_field = {key: stored[key] for key in stored.files}
            original_counts, original_rates = validate_fields(old_field)
            # Only these NEW source raw events are consumed by the reconstruction.
            paths = comparison.bound_case_outputs(source_root, source_manifest, relative, changed)
            readout = comparison.inspect_case_readouts(paths, changed['resolved_config'],
                                                      changed['selected_case'], delivered_readouts=True)
            reversed_counts, reversed_rates = validate_fields(readout['fields'])
            reconstructed = comparison.accumulate_case_histograms(
                readout['raw'], changed['resolved_config'], ci, new_histograms[label])
            require(np.array_equal(reconstructed, reversed_counts), 'NEW_HISTOGRAM_AND_FIELD_COUNTS_DISAGREE')
            fields[original_label][condition], fields[reversed_label][condition] = original_rates, reversed_rates
            responses[reversed_label][ci] = readout['values']
            old_counts.append(original_counts)
            new_counts.append(reversed_counts)
            readout['paired']['inputs_used'] = changed['inputs_used']
            paired[changed['case_id']] = readout['paired']
            info = dict(case_id=changed['case_id'], condition=condition, dt_ms=dt, source_rng_seed=SEED,
                original_report_sha256=old_report_sha256,
                reversed_report_sha256=sha256_file(changed_report_path),
                original_field_sha256=sha256_file(original_fields_path),
                reversed_field_sha256=changed['outputs']['field_observations.npz']['sha256'],
                reversed_raw_sha256=changed['outputs']['raw.npz']['sha256'],
                original_input_invariant_receipt=changed['original_fine_input_agreement'],
                semantic_values_verified=readout['semantic_values_verified'])
            case_inputs[(dt, condition)] = info
            input_refs.append(info)
        counts_original.append(np.stack(old_counts))
        counts_reversed.append(np.stack(new_counts))
        checks = _source_pairing(paired, {key: (SEED, key.removeprefix(f'pfc_{SEED}_')) for key in paired}, [SEED])
        require(all(all(row[key] is True for key in ('exact_pre_cue_events', 'exact_initial_voltage',
                    'exact_final_source_rng', 'same_source_input_consumption')) for row in checks),
                'REVERSED_CONDITION_INPUT_PAIRING_FAILED')
        paired_checks.extend(dict(dt_ms=dt, **row) for row in checks)

    output.mkdir(parents=True, exist_ok=False)
    histogram_metadata, reversed_histograms = {}, {}
    for label, _ in CLOCKS:
        path = output/('reversed_'+label+'_count_histograms.npz')
        histogram_metadata[label] = pfc_count_histograms.write(path, new_histograms[label],
            conditions=CONDITIONS, seeds=[SEED], phases=PHASES, windows_ms=WINDOWS_MS, field_sizes=FIELD_SIZES)
        reversed_histograms[label] = read_histogram(path)
    support = joint_support(old_support, [*old_histograms.values(), *reversed_histograms.values()])
    cdfs, mass_original, mass_reversed = {}, [], []
    for label, _ in CLOCKS:
        old_mass, new_mass = (histogram_masses(rows[label], support)
                              for rows in (old_histograms, reversed_histograms))
        denominator = old_histograms[label]['expected_observations'][:, 0]
        require(np.array_equal(denominator, reversed_histograms[label]['expected_observations'][:, 0]),
                'SAME_CLOCK_COUNT_DENOMINATORS_CHANGED')
        require(old_cdfs[label].shape == (3, 5, 3, 17, len(old_support))
                and np.isfinite(old_cdfs[label]).all(), 'COMPLETE_ACCEPTED_ORIGINAL_CDF_REQUIRED')
        original_cdf = comparison.cdf_on_support(old_cdfs[label], old_support, support)
        require(np.array_equal(original_cdf, np.cumsum(old_mass, axis=-1)/denominator[..., None]),
                'ACCEPTED_CDF_AND_INTEGER_MASS_DISAGREE')
        cdfs['original_'+label] = original_cdf
        cdfs['reversed_'+label] = pfc_count_histograms.cdf(
            reversed_histograms[label]['coordinates'], reversed_histograms[label]['observations'],
            support, (3, 1, 5, 3, 17))[:, 0]
        mass_original.append(old_mass)
        mass_reversed.append(new_mass)

    source_rows, contrast_rows = comparison.response_rows({SCHEME: responses}, PAIRS)
    tables = {
        'field_temporal_discrepancies.csv': comparison.discrepancy_rows({SCHEME: fields}, PAIRS),
        'source_response_comparison.csv': source_rows,
        'paired_response_comparison.csv': contrast_rows,
        'count_cdf_comparison.csv': comparison.count_cdf_rows({SCHEME: cdfs}, support, PAIRS),
    }
    for name, rows in tables.items():
        tables[name] = bind_rows(rows, case_inputs, protocol_sha256)
        write_rows(output/name, tables[name])
    require(len(tables['field_temporal_discrepancies.csv']) == 510
            and len(source_rows) == len(contrast_rows) == 240
            and len(tables['count_cdf_comparison.csv']) == 1530*len(support),
            'COMPLETE_TEMPORAL_COMPARISON_DENOMINATOR_REQUIRED')
    write_rows(output/'shared_input_checks.csv', paired_checks)
    axes = dict(clock_labels=np.asarray([label for label, _ in CLOCKS]),
        dt_ms=np.asarray([dt for _, dt in CLOCKS]), condition_names=np.asarray(CONDITIONS),
        source_rng_seeds=np.asarray([SEED], dtype=np.int64),
        phase_names=np.asarray(list(PHASES)), phase_bounds_ms=np.asarray(list(PHASES.values())),
        field_indices=np.arange(17), field_unit_counts=FIELD_SIZES,
        protocol_sha256=np.asarray(protocol_sha256),
        original_case_report_sha256=np.asarray([[case_inputs[(dt, c)]['original_report_sha256']
                                                 for c in CONDITIONS] for _, dt in CLOCKS]),
        reversed_case_report_sha256=np.asarray([[case_inputs[(dt, c)]['reversed_report_sha256']
                                                 for c in CONDITIONS] for _, dt in CLOCKS]))
    original_counts, reversed_counts = np.stack(counts_original), np.stack(counts_reversed)
    original_rates = original_counts/(FIELD_SIZES[None, None, None, :]*.005)
    reversed_rates = reversed_counts/(FIELD_SIZES[None, None, None, :]*.005)
    np.savez_compressed(output/'field_temporal_differences.npz', **axes,
        data_axes=np.asarray(['clock', 'condition', 'time_bin', 'field']), time_start_ms=np.arange(1000)*5.,
        original_counts=original_counts, reversed_counts=reversed_counts,
        reversed_minus_original_counts=reversed_counts-original_counts,
        original_rates_hz=original_rates, reversed_rates_hz=reversed_rates,
        reversed_minus_original_rates_hz=reversed_rates-original_rates)
    original_mass, reversed_mass = np.stack(mass_original), np.stack(mass_reversed)
    original_cdf = np.stack([cdfs['original_'+label] for label, _ in CLOCKS])
    reversed_cdf = np.stack([cdfs['reversed_'+label] for label, _ in CLOCKS])
    np.savez_compressed(output/'count_temporal_differences.npz', **axes,
        data_axes=np.asarray(['clock', 'condition', 'phase', 'window', 'field', 'spike_count']),
        count_support=support, window_ms=np.asarray(WINDOWS_MS),
        expected_observations=np.stack([old_histograms[label]['expected_observations'][:, 0] for label, _ in CLOCKS]),
        original_observations=original_mass, reversed_observations=reversed_mass,
        reversed_minus_original_observations=reversed_mass-original_mass,
        original_cdf=original_cdf, reversed_cdf=reversed_cdf,
        reversed_minus_original_cdf=reversed_cdf-original_cdf)
    original_responses = np.stack([responses['original_'+label] for label, _ in CLOCKS])
    reversed_responses = np.stack([responses['reversed_'+label] for label, _ in CLOCKS])
    np.savez_compressed(output/'phase_temporal_differences.npz', **axes,
        data_axes=np.asarray(['clock', 'condition', 'phase', 'metric']), metric_names=np.asarray(METRICS),
        original_values_hz=original_responses, reversed_values_hz=reversed_responses,
        reversed_minus_original_hz=reversed_responses-original_responses)

    summaries = []
    for li, (_, dt) in enumerate(CLOCKS):
        for ci, condition in enumerate(CONDITIONS):
            delta = reversed_counts[li, ci]-original_counts[li, ci]
            cdf_delta = reversed_cdf[li, ci]-original_cdf[li, ci]
            maximum = np.unravel_index(np.abs(cdf_delta).argmax(), cdf_delta.shape)
            pi, wi, fi, ki = map(int, maximum)
            summaries.append(dict(dt_ms=dt, condition=condition,
                full_field_counts_exactly_equal=bool(np.array_equal(reversed_counts[li, ci], original_counts[li, ci])),
                changed_field_time_bins=int(np.count_nonzero(delta)),
                total_original_spikes=int(original_counts[li, ci].sum()),
                total_reversed_spikes=int(reversed_counts[li, ci].sum()),
                maximum_absolute_field_count_difference=int(np.abs(delta).max()),
                maximum_absolute_field_rate_difference_hz=float(np.abs(reversed_rates[li, ci]-original_rates[li, ci]).max()),
                changed_integer_histogram_cells=int(np.count_nonzero(reversed_mass[li, ci]-original_mass[li, ci])),
                maximum_absolute_cdf_difference=float(np.abs(cdf_delta).max()),
                cdf_maximum_identity=dict(phase=list(PHASES)[pi], window_ms=WINDOWS_MS[wi],
                                         field_index=fi, count_threshold=int(support[ki]),
                                         reversed_minus_original=float(cdf_delta[maximum])),
                all_unit_phase_rate_difference_hz={phase: float(reversed_responses[li, ci, pi, 0]
                    -original_responses[li, ci, pi, 0]) for pi, phase in enumerate(PHASES)}))
    report = dict(schema=SCHEMA, status=STATUS, scope=SCOPE, settings=settings,
        settings_sha256=sha256_file(config_path), protocol_sha256=protocol_sha256,
        source_job_id=settings['source_job_id'], source_delivery_sha256=source_delivery_sha256,
        original_source_delivery_sha256=original_delivery_sha256,
        original_comparison_delivery_sha256=previous_delivery_sha256,
        original_comparison_report_sha256=sha256_file(previous_report_path),
        cases=input_refs, case_summaries=summaries, paired_input_checks=paired_checks,
        complete_reversed_cases_consumed=6, complete_original_cases_reused=6,
        source_simulations=0, original_raw_episodes_read=0, new_original_independent_blocks=0,
        original_independent_blocks=1, reused_original_independent_blocks=1, new_auxiliary_streams=0,
        independent_auxiliary_repeat=False, full_field_cell_count=int(original_counts.size),
        observations_are_independent_replicates=False, temporal_transform=SCHEME,
        same_clock_pairs=[list(pair) for pair in PAIRS],
        empirical_original_count_support=old_support.tolist(), empirical_common_count_support=support.tolist(),
        original_cdf_support_extended=bool(support[-1] > old_support[-1]),
        row_counts={**{name: len(rows) for name, rows in tables.items()},
                    'shared_input_checks.csv': len(paired_checks)},
        histogram_metadata=histogram_metadata, covariance_estimates=False, confidence_intervals=False,
        fdr_tests=False, frozen_margin_tests=False, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, production_episodes=0,
        interpretation='All three conditions, five phases, 17 source fields and 5/50/100 ms windows are retained. '
            'Every difference is reversed minus original at the same clock. The source execution receipt binds '
            'equal initial state, final coarse and auxiliary RNG states, full applied current and external event totals. '
            'Original accepted readouts are reused; only reversed raw events are reconstructed. This is one reused '
            'initial/background block with replayed auxiliary streams, not an independent repeat, covariance estimate, '
            'confidence interval, convergence test, frozen-margin qualification or population tail bound.',
        implementation_sha256=sha256_file(Path(__file__)),
        dependencies={module.__name__.split('.')[-1]: sha256_file(Path(module.__file__))
                      for module in (comparison, execution, pfc_count_histograms)},
        wall_seconds=time.perf_counter()-started,
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha256_file(p))
                 for p in output.iterdir() if p.is_file()})
    freeze_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.project_root, args.output)
    print(json.dumps({key: report[key] for key in ('status', 'row_counts', 'source_simulations')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

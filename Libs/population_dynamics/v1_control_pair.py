"""Compare recorded sampler initialization and pre-control source spike multisets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_configured_inputs import compare_population, validate_plan
from .v1_lgn_spikes import load_spike_trial
from .v1_nest_input_replay import bound_json

SOURCE_UNITS = 17400
RATE_SAMPLES = 3001
CHANGED_FRAME = 501
DURATION_MS = 3000.0
STARTING_NOISE_SEED = 12345
SEED_FIELDS = ('source_primary_seed', 'source_poisson_seed', 'source_sampler_code')


def validate_rate_pair(pair: dict) -> float:
    """Require the complete exact rate-prefix result and return its event boundary."""
    comparison = pair.get('comparison', {})
    prefix = comparison.get('unchanged_prefix', {})
    later = comparison.get('at_and_after_changed_frame', {})
    ids = np.asarray(pair.get('source_unit_ids', []))
    rows = comparison.get('per_unit', [])
    if (pair.get('status') != 'FULL_SOURCE_CONTROL_PREFIX_EXACT_DEVELOPMENT'
            or pair.get('exit_code') != 0 or pair.get('source_units') != SOURCE_UNITS
            or pair.get('every_unit_clock_receipts_verified') is not True
            or comparison.get('source_units') != SOURCE_UNITS or comparison.get('clock_exact') is not True
            or ids.shape != (SOURCE_UNITS,) or ids.dtype.kind not in 'iu'
            or (ids < 0).any() or len(np.unique(ids)) != SOURCE_UNITS
            or len(rows) != SOURCE_UNITS or [row.get('source_id') for row in rows] != ids.tolist()
            or not all(row.get('prefix_exact') is True and row.get('prefix_mismatch_count') == 0
                       and row.get('prefix_max_absolute_difference_hz') == 0.0 for row in rows)
            or prefix.get('exact') is not True or prefix.get('mismatch_count') != 0
            or prefix.get('max_absolute_difference_hz') != 0.0
            or prefix.get('affected_source_ids') != []
            or prefix.get('frame_start') != 0 or prefix.get('frame_stop_exclusive') != CHANGED_FRAME
            or pair.get('changed_input_frame') != CHANGED_FRAME
            or later.get('frame_start') != CHANGED_FRAME
            or later.get('first_time_s') != CHANGED_FRAME/1000.0):
        raise ValueError('V1_SPIKE_PAIR_COMPLETE_EXACT_RATE_PREFIX_REQUIRED')
    references = pair['source_array_references']
    if any(references[role].get('shape') != [SOURCE_UNITS, RATE_SAMPLES]
           for role in ('baseline', 'control')):
        raise ValueError('V1_SPIKE_PAIR_FULL_RATE_SHAPES_REQUIRED')
    hashes = pair['prepared_runtime_hashes']
    if pair.get('prepared_runtime_hashes_exact') is not True or hashes['baseline'] != hashes['control']:
        raise ValueError('V1_SPIKE_PAIR_MATCHED_PREPARED_RUNTIME_REQUIRED')
    boundary_ms = later['first_time_s']*1000.0
    if boundary_ms != float(CHANGED_FRAME):
        raise ValueError('V1_SPIKE_PAIR_CHANGED_FRAME_TIME_DIFFERS')
    return boundary_ms


def validate_spike_reports(baseline: dict, control: dict, pair: dict) -> None:
    first, second = baseline['config'], control['config']
    expected = dict(first, stimulus=dict(first['stimulus']))
    expected['stimulus']['contrast_percent'] = 40.0
    if (first['stimulus'].get('contrast_percent') != 80.0 or second != expected):
        raise ValueError('V1_SPIKE_PAIR_CONFIG_MAY_ONLY_CHANGE_CONTRAST_80_TO_40')
    for role, report, rate_config, report_hash_key in (
            ('baseline', baseline, pair['baseline_config'], 'baseline_report_sha256'),
            ('control', control, pair['config'], 'control_report_sha256')):
        config = report['config']
        trials = report.get('trials', [])
        if (report.get('status') != 'SOURCE_SPIKES_EXPORTED_EXACT'
                or report.get('source_units') != SOURCE_UNITS
                or report.get('rate_shape') != [SOURCE_UNITS, RATE_SAMPLES]
                or config.get('scope') != 'development_source_runtime_qualification'
                or config.get('unit_coverage') != 'all_source_units'
                or config.get('spike_generation') != {'starting_noise_seed': STARTING_NOISE_SEED, 'trials': 1}
                or report.get('starting_noise_seed') != STARTING_NOISE_SEED
                or len(trials) != 1 or trials[0].get('trial') != 0
                or trials[0].get('source_units') != SOURCE_UNITS
                or {key: value for key, value in config.items() if key != 'spike_generation'}
                   != {key: value for key, value in rate_config.items() if key != 'spike_generation'}):
            raise ValueError('V1_SPIKE_PAIR_FULL_DECLARED_INVOCATION_DIFFERS:' + role)
        expected_hashes = dict(rate_report_sha256=pair[report_hash_key],
            rate_file_sha256=pair['source_array_references'][role]['sha256'],
            prepared_runtime_sha256=pair['prepared_runtime_hashes'][role])
        if any(report.get(key) != value for key, value in expected_hashes.items()):
            raise ValueError('V1_SPIKE_PAIR_RATE_SOURCE_IDENTITY_DIFFERS:' + role)


def validate_source_plan(plan: dict, expected_ids) -> None:
    """Validate every raw event before taking a temporal prefix."""
    validate_plan(plan)
    if (len(plan['source_id']) != SOURCE_UNITS or not np.all(plan['population'] == 0)
            or not np.array_equal(plan['source_id'], np.asarray(expected_ids))
            or np.any(plan['source_event_time_ms'] > DURATION_MS)):
        raise ValueError('V1_SPIKE_PAIR_FULL_INVENTORY_OR_EVENT_DOMAIN_DIFFERS')
    for key, dtype in (('source_primary_seed', np.dtype('int64')),
                       ('source_poisson_seed', np.dtype('int64')),
                       ('source_sampler_code', np.dtype('int8'))):
        array = np.asarray(plan[key])
        if array.shape != (SOURCE_UNITS,) or array.dtype != dtype:
            raise ValueError('V1_SPIKE_PAIR_SAMPLER_ARRAY_SCHEMA_DIFFERS:' + key)
    primary, poisson, code = [plan[key] for key in SEED_FIELDS]
    if (np.any(primary < 0) or np.any(primary >= 10000)
            or not np.isin(code, [0, 1]).all()
            or np.any(poisson[code == 0] != -1)
            or np.any(poisson[code == 1] < 0) or np.any(poisson[code == 1] >= 10000)):
        raise ValueError('V1_SPIKE_PAIR_RECORDED_SAMPLER_DOMAIN_INVALID')


def _segment(plan, selected):
    units = plan['source_event_unit'][selected]
    return dict(population=plan['population'], source_id=plan['source_id'],
        source_counts=np.bincount(units.astype(np.int64), minlength=len(plan['source_id'])),
        source_event_unit=units, source_event_time_ms=plan['source_event_time_ms'][selected])


def _train_difference(first, second):
    left, right = np.sort(first), np.sort(second)
    same_count = len(left) == len(right)
    mismatch = int(np.count_nonzero(left != right)) if same_count else None
    maximum = (float(np.max(np.abs(left-right))) if len(left) else 0.0) if same_count else None
    return dict(baseline_count=len(left), control_count=len(right), counts_exact=same_count,
        event_multiset_exact=bool(same_count and mismatch == 0),
        timestamp_mismatches=mismatch, max_absolute_time_difference_ms=maximum)


def _segment_difference(baseline, control):
    event_ids = control['source_id'][control['source_event_unit']]
    global_check = compare_population(baseline, 0, event_ids, control['source_event_time_ms'])
    left = np.split(baseline['source_event_time_ms'], np.cumsum(baseline['source_counts'])[:-1])
    right = np.split(control['source_event_time_ms'], np.cumsum(control['source_counts'])[:-1])
    rows = [_train_difference(first, second) for first, second in zip(left, right)]
    if global_check['all_event_pairs_exact'] != all(row['event_multiset_exact'] for row in rows):
        raise RuntimeError('V1_SPIKE_PAIR_GLOBAL_AND_UNIT_COMPARISONS_DISAGREE')
    return global_check, rows


def compare_plans(baseline: dict, control: dict, expected_ids, boundary_ms: float) -> dict:
    validate_source_plan(baseline, expected_ids)
    validate_source_plan(control, expected_ids)
    if boundary_ms != float(CHANGED_FRAME):
        raise ValueError('V1_SPIKE_PAIR_BOUNDARY_NOT_RATE_BOUND')
    prefix_left = baseline['source_event_time_ms'] < boundary_ms
    prefix_right = control['source_event_time_ms'] < boundary_ms
    prefix_global, prefix_units = _segment_difference(_segment(baseline, prefix_left),
                                                       _segment(control, prefix_right))
    later_global, later_units = _segment_difference(_segment(baseline, ~prefix_left),
                                                    _segment(control, ~prefix_right))
    per_unit = []
    changed_initialization, changed_prefix, changed_later = [], [], []
    field_changes = {key: [] for key in SEED_FIELDS}
    for index, source_id in enumerate(baseline['source_id']):
        identity = int(source_id)
        first = {key: int(baseline[key][index]) for key in SEED_FIELDS}
        second = {key: int(control[key][index]) for key in SEED_FIELDS}
        changes = [key for key in SEED_FIELDS if first[key] != second[key]]
        if changes:
            changed_initialization.append(identity)
        for key in changes:
            field_changes[key].append(identity)
        if not prefix_units[index]['event_multiset_exact']:
            changed_prefix.append(identity)
        if not later_units[index]['event_multiset_exact']:
            changed_later.append(identity)
        per_unit.append(dict(source_id=identity, baseline_initialization=first,
            control_initialization=second, initialization_exact=not changes,
            initialization_changed_fields=changes, prefix=prefix_units[index], later=later_units[index]))
    initialization_exact = not changed_initialization
    prefix_exact = prefix_global['all_event_pairs_exact']
    return dict(source_units=SOURCE_UNITS, source_unit_ids=np.asarray(expected_ids).tolist(),
        recorded_sampler_initialization=dict(exact=initialization_exact,
            changed_source_ids=changed_initialization, changed_source_ids_by_field=field_changes),
        prefix=dict(boundary_ms=boundary_ms, boundary_rule='0 < time_ms < boundary_ms',
            exact=prefix_exact, changed_source_ids=changed_prefix, global_comparison=prefix_global),
        later=dict(boundary_rule='boundary_ms <= time_ms <= 3000',
            changed_source_ids=changed_later, global_comparison=later_global,
            response_difference_required_for_each_unit=False),
        per_unit=per_unit, pair_exact=bool(initialization_exact and prefix_exact))


def compare_archives(baseline_spike_archive: Path, control_spike_archive: Path,
                     rate_pair_archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_SOURCE_SPIKE_CONTROL_PAIR', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        baseline_spike_archive=str(baseline_spike_archive), control_spike_archive=str(control_spike_archive),
        rate_pair_archive=str(rate_pair_archive), source_fidelity_qualified=False,
        scientific_qualification=False, visual_action_mapping_qualified=False,
        production_domain_qualified=False, production_episodes=0, cortical_execution=False,
        coupled_neural_episodes=0, new_sampling=False, new_native_replay=False,
        tolerance_applied=False, hidden_random_draws_equal_claimed=False,
        per_unit_noise_independence_claimed=False,
        interpretation=('This verifies recorded sampler initialization and the observed event prefix. '
            'It does not establish equality of all hidden random draws, independent per-unit noise, '
            'or biological acausality from a differing pair.'))
    try:
        relative = 'full_causality/report.json'
        pair, _ = bound_json(rate_pair_archive, relative)
        report.update(rate_pair_report_path=relative,
                      rate_pair_report_sha256=sha256_file(rate_pair_archive/relative))
        boundary_ms = validate_rate_pair(pair)
        baseline, first_report = load_spike_trial(baseline_spike_archive, trial=0)
        control, second_report = load_spike_trial(control_spike_archive, trial=0)
        report.update(baseline_config=first_report['config'], control_config=second_report['config'])
        validate_spike_reports(first_report, second_report, pair)
        report['source_array_references'] = {}
        for role, archive, source_report in (('baseline', baseline_spike_archive, first_report),
                                             ('control', control_spike_archive, second_report)):
            trial = source_report['trials'][0]
            report['source_array_references'][role] = dict(archive=str(archive), trial=0,
                plan_path='spikes/'+trial['plan'], plan_sha256=source_report['artifacts'][trial['plan']],
                spike_report_path='spikes/report.json', spike_report_sha256=sha256_file(archive/'spikes/report.json'),
                input_identity_sha256=trial['input_identity_sha256'],
                rate_report_sha256=source_report['rate_report_sha256'],
                rate_file_sha256=source_report['rate_file_sha256'],
                prepared_runtime_sha256=source_report['prepared_runtime_sha256'],
                raw_values_preserved_in_original_archive=True)
        result = compare_plans(baseline, control, pair['source_unit_ids'], boundary_ms)
        report.update(comparison=result,
            status=('SOURCE_SPIKE_CONTROL_PAIR_EXACT_DEVELOPMENT' if result['pair_exact']
                    else 'SOURCE_SPIKE_CONTROL_PAIR_DIFFERS_DIAGNOSTIC'),
            exit_code=0 if result['pair_exact'] else 2)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest(report.get('control_config', {}), list(sys.argv), report['exit_code'])
        freeze_json(output/'report.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-spike-archive', 'control-spike-archive', 'rate-pair-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    return compare_archives(args.baseline_spike_archive, args.control_spike_archive,
                            args.rate_pair_archive, args.output)['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

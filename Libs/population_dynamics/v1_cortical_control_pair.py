"""Archive-only cortical response comparison for a bound 80/40 visual-control pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_configured_inputs import compare_population, delivered_file, validate_plan
from .v1_control_pair import _train_difference
from .v1_nest_input_replay import bound_json
from .v1_nest_repeatability import compare_activity, load_outputs
from .v1_source_dataset import exact_ticks

CORTICAL_UNITS = 230924
LGN_UNITS = 17400
DURATION_MS = 3000.0
NATIVE_DT_MS = 0.25
PREFIX_END_MS = 501.0
PREFIX_END_TICK = 2004
SOURCE_RECEIPTS = ('graph/graph_identity.json', 'reference/reference/intake.json',
    'parameters/comparison.json', 'inputs/input_identity.json',
    'pointnet/source_inspection.json', 'timing/diagnosis/report.json')
NATIVE_SOURCE_FILES = tuple('Libs/population_dynamics/'+name+'.py' for name in (
    'v1_nest_network', 'v1_nest_time_semantics', 'v1_nest_input_replay', 'provenance',
    'v1_lgn_spikes', 'v1_lgn_runtime', 'v1_lgn_source', 'v1_configured_inputs'))
MATCHED_NATIVE_FIELDS = ('implementation_sha256', 'nest_version', 'threads',
    'connection_batch_size', 'kernel_rng_seed', 'coupled_cortical_neurons',
    'native_duration_ms', 'input_generators', 'recurrent_connections', 'input_connections',
    'native_total_connections', 'observation_scope', 'kernel_delays_ms',
    'parameter_initialization_source')


def validate_input_pair(pair):
    comparison = pair.get('comparison', {})
    ids = np.asarray(comparison.get('source_unit_ids', []))
    rows = comparison.get('per_unit', [])
    if (pair.get('status') != 'SOURCE_SPIKE_CONTROL_PAIR_EXACT_DEVELOPMENT'
            or pair.get('exit_code') != 0 or comparison.get('source_units') != LGN_UNITS
            or comparison.get('pair_exact') is not True
            or comparison.get('recorded_sampler_initialization', {}).get('exact') is not True
            or comparison.get('prefix', {}).get('exact') is not True
            or comparison['prefix'].get('boundary_ms') != PREFIX_END_MS
            or comparison['prefix'].get('global_comparison', {}).get('all_event_pairs_exact') is not True
            or ids.shape != (LGN_UNITS,) or ids.dtype.kind not in 'iu' or (ids < 0).any()
            or len(np.unique(ids)) != LGN_UNITS or len(rows) != LGN_UNITS
            or [row.get('source_id') for row in rows] != ids.tolist()
            or not all(row.get('initialization_exact') is True
                       and row.get('prefix', {}).get('event_multiset_exact') is True for row in rows)):
        raise ValueError('CORTICAL_PAIR_COMPLETE_MATCHED_SOURCE_INPUT_REQUIRED')
    return ids


def validate_native_reports(first, second, first_delivery, second_delivery, pair):
    for role, native, delivery, config_key in (
            ('baseline', first, first_delivery, 'baseline_config'),
            ('control', second, second_delivery, 'control_config')):
        if (delivery.get('status') != 'PASS' or delivery.get('exit_code') != 0
                or native.get('status') != 'NATIVE_SOURCE_GRAPH_DIAGNOSTIC_COMPLETE'
                or native.get('full_cortical_network_executed') is not True
                or native.get('coupled_cortical_neurons') != CORTICAL_UNITS
                or native.get('native_duration_ms') != DURATION_MS
                or native.get('input_generators') != LGN_UNITS+1):
            raise ValueError('CORTICAL_PAIR_COMPLETE_NATIVE_EXECUTION_REQUIRED:' + role)
        generated = native.get('generated_lgn', {})
        source = pair['source_array_references'][role]
        if (generated.get('config') != pair[config_key]
                or any(generated.get(key) != source.get(key) or key not in source
                       for key in ('spike_report_sha256', 'input_identity_sha256', 'trial'))
                or generated.get('generated_units') != LGN_UNITS
                or generated.get('background_events_unchanged') is not True
                or generated.get('input_edge_objects_unchanged') is not True):
            raise ValueError('CORTICAL_PAIR_GENERATED_SOURCE_BINDING_DIFFERS:' + role)
    for key in MATCHED_NATIVE_FIELDS:
        if key not in first or key not in second or first[key] != second[key]:
            raise ValueError('CORTICAL_PAIR_NATIVE_EXECUTION_CONTRACT_DIFFERS:' + key)
    for key in ('original_plan_sha256', 'input_origin_sha256'):
        if (key not in first['generated_lgn'] or key not in second['generated_lgn']
                or first['generated_lgn'][key] != second['generated_lgn'][key]):
            raise ValueError('CORTICAL_PAIR_BACKGROUND_OR_INPUT_ORIGIN_DIFFERS:' + key)
    matched_sources = {}
    for relative in NATIVE_SOURCE_FILES:
        hashes = []
        for role, native, delivery in (('baseline', first, first_delivery),
                                      ('control', second, second_delivery)):
            rows = [row for row in delivery.get('files', []) if row.get('path') == relative]
            digest = rows[0].get('sha256') if len(rows) == 1 else None
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(character not in '0123456789abcdef' for character in digest)):
                raise ValueError('CORTICAL_PAIR_NATIVE_SOURCE_HASH_MISSING_OR_INVALID:' + role + ':' + relative)
            if relative == NATIVE_SOURCE_FILES[0] and digest != native['implementation_sha256']:
                raise ValueError('CORTICAL_PAIR_NATIVE_IMPLEMENTATION_HASH_UNBOUND:' + role)
            hashes.append(digest)
        if hashes[0] != hashes[1]:
            raise ValueError('CORTICAL_PAIR_NATIVE_SOURCE_HASH_DIFFERS:' + relative)
        matched_sources[relative] = hashes[0]
    return matched_sources


def validate_cortical_identities(first, second):
    for identity in (first, second):
        for key in ('source_id', 'native_id', 'node_type_id'):
            values = identity[key]
            if values.shape != (CORTICAL_UNITS,) or values.dtype.kind not in 'iu' or (values < 0).any():
                raise ValueError('CORTICAL_PAIR_IDENTITY_SCHEMA_INVALID:' + key)
        if (not np.array_equal(identity['source_id'], np.arange(CORTICAL_UNITS))
                or (identity['native_id'] <= 0).any()
                or len(np.unique(identity['native_id'])) != CORTICAL_UNITS
                or identity['initial_vm_mV'].shape != (CORTICAL_UNITS,)
                or identity['initial_vm_mV'].dtype.kind != 'f'
                or not np.isfinite(identity['initial_vm_mV']).all()):
            raise ValueError('CORTICAL_PAIR_INITIAL_IDENTITY_OR_VOLTAGE_INVALID')
    for key in first:
        if first[key].dtype != second[key].dtype:
            raise ValueError('CORTICAL_PAIR_IDENTITY_DTYPE_DIFFERS:' + key)
        if key != 'initial_vm_mV' and not np.array_equal(first[key], second[key]):
            raise ValueError('CORTICAL_PAIR_ORDERED_IDENTITY_DIFFERS:' + key)


def validate_merged_input(plan, identity, native, expected_lgn_ids):
    validate_plan(plan)
    n = LGN_UNITS+1
    if (len(plan['source_id']) != n or (plan['source_event_time_ms'] > DURATION_MS).any()
            or set(np.unique(plan['population']).tolist()) != {0, 1}
            or np.count_nonzero(plan['population'] == 1) != 1
            or not np.array_equal(plan['source_id'][plan['population'] == 0], expected_lgn_ids)
            or int(plan['source_counts'].sum()) != native['input_events']):
        raise ValueError('CORTICAL_PAIR_MERGED_INPUT_DOMAIN_OR_INVENTORY_DIFFERS')
    for key in ('population', 'source_id', 'native_id', 'source_event_counts'):
        if identity[key].shape != (n,) or identity[key].dtype.kind not in 'iu':
            raise ValueError('CORTICAL_PAIR_INPUT_IDENTITY_SCHEMA_INVALID:' + key)
    if ((identity['native_id'] <= 0).any() or len(np.unique(identity['native_id'])) != n
            or not np.array_equal(identity['population'], plan['population'])
            or not np.array_equal(identity['source_id'], plan['source_id'])
            or not np.array_equal(identity['source_event_counts'], plan['source_counts'])):
        raise ValueError('CORTICAL_PAIR_INPUT_IDENTITY_JOIN_DIFFERS')


def compare_actual_inputs(first, second):
    for key in ('population', 'source_id'):
        if first[key].dtype != second[key].dtype or not np.array_equal(first[key], second[key]):
            raise ValueError('CORTICAL_PAIR_ORDERED_INPUT_IDENTITIES_DIFFER')
    checks = {}
    for label, population, cutoff in (('background', 1, None), ('lgn_prefix', 0, PREFIX_END_MS)):
        left_mask = np.ones(len(first['source_event_time_ms']), dtype=bool)
        if cutoff is not None:
            left_mask &= first['source_event_time_ms'] < cutoff
        event_units = first['source_event_unit'][left_mask]
        left = dict(population=first['population'], source_id=first['source_id'],
            source_counts=np.bincount(event_units.astype(np.int64), minlength=len(first['source_id'])),
            source_event_unit=event_units, source_event_time_ms=first['source_event_time_ms'][left_mask])
        right_mask = second['population'][second['source_event_unit']] == population
        if cutoff is not None:
            right_mask &= second['source_event_time_ms'] < cutoff
        event_ids = second['source_id'][second['source_event_unit'][right_mask]]
        checks[label] = compare_population(left, population, event_ids, second['source_event_time_ms'][right_mask])
    return checks


def _group_activity(activity):
    # One global grouping; all subsequent masks operate only on one unit's slice.
    order = np.argsort(activity['spike_source_id'], kind='stable')
    offsets = np.concatenate((np.array([0], dtype=np.int64),
                              np.cumsum(activity['spike_counts'], dtype=np.int64)))
    return activity['spike_time_ms'][order], offsets


def cortical_differences(first, second, first_identity, second_identity):
    arrays = dict(source_id=first_identity['source_id'].copy(), node_type_id=first_identity['node_type_id'].copy(),
        initial_vm_exact=first_identity['initial_vm_mV'] == second_identity['initial_vm_mV'],
        initial_vm_difference_mV=second_identity['initial_vm_mV']-first_identity['initial_vm_mV'],
        terminal_vm_exact=first['final_vm_mV'] == second['final_vm_mV'],
        terminal_vm_difference_mV=second['final_vm_mV']-first['final_vm_mV'])
    for label in ('prefix', 'later'):
        for name in ('baseline_count', 'control_count', 'count_difference'):
            arrays[label+'_'+name] = np.zeros(CORTICAL_UNITS, dtype=np.int64)
        for name in ('event_multiset_exact', 'timing_metrics_valid'):
            arrays[label+'_'+name] = np.zeros(CORTICAL_UNITS, dtype=bool)
        for name in ('timestamp_mismatches', 'max_absolute_time_difference_ms'):
            arrays[label+'_'+name] = np.full(CORTICAL_UNITS, np.nan)
    left_times, left_offsets = _group_activity(first)
    right_times, right_offsets = _group_activity(second)
    for unit in range(CORTICAL_UNITS):
        left = left_times[left_offsets[unit]:left_offsets[unit+1]]
        right = right_times[right_offsets[unit]:right_offsets[unit+1]]
        left_prefix, right_prefix = left < PREFIX_END_MS, right < PREFIX_END_MS
        for label, a, b in (('prefix', left[left_prefix], right[right_prefix]),
                            ('later', left[~left_prefix], right[~right_prefix])):
            difference = _train_difference(a, b)
            for key in ('baseline_count', 'control_count', 'event_multiset_exact'):
                arrays[label+'_'+key][unit] = difference[key]
            arrays[label+'_count_difference'][unit] = len(b)-len(a)
            arrays[label+'_timing_metrics_valid'][unit] = difference['counts_exact']
            if difference['counts_exact']:
                for key in ('timestamp_mismatches', 'max_absolute_time_difference_ms'):
                    arrays[label+'_'+key][unit] = difference[key]
    summary = {}
    for label in ('prefix', 'later'):
        valid = arrays[label+'_timing_metrics_valid']
        summary[label] = dict(compared_units=CORTICAL_UNITS,
            baseline_events=int(arrays[label+'_baseline_count'].sum()),
            control_events=int(arrays[label+'_control_count'].sum()),
            baseline_silent_units=int(np.count_nonzero(arrays[label+'_baseline_count'] == 0)),
            control_silent_units=int(np.count_nonzero(arrays[label+'_control_count'] == 0)),
            event_multisets_exact=bool(arrays[label+'_event_multiset_exact'].all()),
            differing_units=int(np.count_nonzero(~arrays[label+'_event_multiset_exact'])),
            unequal_count_units=int(np.count_nonzero(~valid)),
            timestamp_mismatches=int(arrays[label+'_timestamp_mismatches'].sum()) if valid.all() else None,
            max_absolute_time_difference_ms=float(arrays[label+'_max_absolute_time_difference_ms'].max()) if valid.all() else None)
    for label in ('initial', 'terminal'):
        summary[label+'_vm'] = dict(exact=bool(arrays[label+'_vm_exact'].all()),
            differing_units=int(np.count_nonzero(~arrays[label+'_vm_exact'])),
            max_absolute_difference_mV=float(np.abs(arrays[label+'_vm_difference_mV']).max()))
    return arrays, summary


def compare_archives(baseline_native_archive: Path, control_native_archive: Path,
                     input_pair_archive: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAILED_CORTICAL_CONTROL_PAIR', exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(),
        baseline_native_archive=str(baseline_native_archive), control_native_archive=str(control_native_archive),
        input_pair_archive=str(input_pair_archive), scientific_qualification=False,
        source_fidelity_qualified=False, visual_action_mapping_qualified=False,
        production_domain_qualified=False, production_episodes=0, new_cortical_executions=0,
        intermediate_voltage_recorded=False, hidden_random_draws_equality_claimed=False,
        tolerance_applied=False, prefix_end_ms=PREFIX_END_MS, prefix_end_tick=PREFIX_END_TICK,
        native_dt_ms=NATIVE_DT_MS,
        prefix_rule='0 < time_ms < 501; native ticks < 2004; no synaptic-delay extension.',
        later_rule='501 <= time_ms <= 3000, including the endpoint; changes are descriptive.',
        response_difference_required_for_every_unit=False,
        initial_state_scope='Observed initial V_m plus identical constructor/source receipts; no full hidden-state claim.')
    try:
        pair, _ = bound_json(input_pair_archive, 'spike_pair/report.json')
        lgn_ids = validate_input_pair(pair)
        report['input_pair_report_sha256'] = sha256_file(input_pair_archive/'spike_pair/report.json')
        first, first_delivery = bound_json(baseline_native_archive, 'network/report.json')
        second, second_delivery = bound_json(control_native_archive, 'network/report.json')
        report['native_report_sha256'] = dict(baseline=sha256_file(baseline_native_archive/'network/report.json'),
                                             control=sha256_file(control_native_archive/'network/report.json'))
        report['matched_native_source_sha256'] = validate_native_reports(
            first, second, first_delivery, second_delivery, pair)
        report['matched_source_receipts'] = {}
        for relative in SOURCE_RECEIPTS:
            paths = [delivered_file(archive, relative) for archive in (baseline_native_archive, control_native_archive)]
            hashes = [sha256_file(path) for path in paths]
            if hashes[0] != hashes[1]:
                raise ValueError('CORTICAL_PAIR_SOURCE_RECEIPT_DIFFERS:' + relative)
            report['matched_source_receipts'][relative] = hashes[0]
            if relative == 'inputs/input_identity.json':
                clock = json.loads(paths[0].read_text())['source_config_clock']
                if clock != {'duration': DURATION_MS, 'dt': NATIVE_DT_MS}:
                    raise ValueError('CORTICAL_PAIR_NATIVE_SOURCE_CLOCK_DIFFERS')
        outputs = [load_outputs(archive/'network', native) for archive, native in
                   ((baseline_native_archive, first), (control_native_archive, second))]
        identities = [values['cortical_identity.npz'] for values in outputs]
        validate_cortical_identities(*identities)
        activity = [values['cortical_activity.npz'] for values in outputs]
        report['whole_response_descriptive_comparison'] = compare_activity(*activity, CORTICAL_UNITS, DURATION_MS)
        for values in activity:
            exact_ticks(values['spike_time_ms'], NATIVE_DT_MS, int(DURATION_MS/NATIVE_DT_MS))
        plans = []
        report['source_array_references'] = {}
        for role, archive, native, values in (('baseline', baseline_native_archive, first, outputs[0]),
                                               ('control', control_native_archive, second, outputs[1])):
            path = delivered_file(archive, 'network/source_inputs.npz')
            if sha256_file(path) != native['artifacts']['source_inputs.npz']:
                raise ValueError('CORTICAL_PAIR_RAW_INPUT_ARTIFACT_CHANGED')
            with np.load(path, allow_pickle=False) as stored:
                plan = {key: stored[key] for key in stored.files}
            validate_merged_input(plan, values['input_identity.npz'], native, lgn_ids)
            plans.append(plan)
            report['source_array_references'][role] = dict(archive=str(archive),
                arrays={name: dict(path='network/'+name, sha256=native['artifacts'][name]) for name in
                    ('cortical_identity.npz', 'cortical_activity.npz', 'input_identity.npz', 'source_inputs.npz')},
                generated_source=pair['source_array_references'][role])
        for key in ('population', 'source_id', 'native_id'):
            a, b = (values['input_identity.npz'][key] for values in outputs)
            if a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError('CORTICAL_PAIR_NATIVE_INPUT_MAPPING_DIFFERS:' + key)
        report['actual_input_comparisons'] = compare_actual_inputs(*plans)
        if not all(value['all_event_pairs_exact'] for value in report['actual_input_comparisons'].values()):
            raise ValueError('CORTICAL_PAIR_ACTUAL_BACKGROUND_OR_LGN_PREFIX_DIFFERS')
        arrays, summary = cortical_differences(*activity, *identities)
        with (output/'per_unit.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        exact = summary['prefix']['event_multisets_exact'] and summary['initial_vm']['exact']
        report.update(status='CORTICAL_CONTROL_PAIR_EXACT_DEVELOPMENT' if exact else 'CORTICAL_CONTROL_PAIR_DIFFERS_DIAGNOSTIC',
            exit_code=0 if exact else 2, compared_cortical_units=CORTICAL_UNITS, comparison=summary,
            per_unit_artifact=dict(path='per_unit.npz', sha256=sha256_file(output/'per_unit.npz'),
                columns=list(arrays), units=CORTICAL_UNITS,
                undefined_timing='NaN exactly where the matching prefix/later timing_metrics_valid mask is false.'))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['runtime'] = runtime_manifest({}, list(sys.argv), report['exit_code'])
        freeze_json(output/'report.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-native-archive', 'control-native-archive', 'input-pair-archive', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    return compare_archives(args.baseline_native_archive, args.control_native_archive,
                            args.input_pair_archive, args.output)['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

"""Export one reconciled coupled CA3 retrieval through the shared circuit archive.

The primary retrieval is the episode; an observed replay supplies delivery
verification, never an additional episode. One-ms controls describe source set
assignments and background laws, not future realized currents. Coarser reader
windows return time means and do not retain the within-bin assignment order.
An explicitly selected cue omission reuses the source circuit/block identity
and a completed paired measurement; it does not add an independent noise block.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from . import ca3_control_trace as control
from . import ca3_native_observation_replay as reconciliation
from .ca3_retrieval_readouts import _bound_inputs, _source_proof, source_identity, SOURCE_REVISION, PYRAMIDAL
from .ca3_realization import from_record, verify_retrieval_program, validate_source_reset_contract
from .carlsim_records import spike_blocks
from .circuit_data import SCHEMA, CircuitUnits, CircuitEpisode, CircuitArchiveWriter, CircuitDataset
from .provenance import canonical_json, sha256_file

DURATION_MS = 1001
VERIFY_WINDOW_BINS = 100


class BoundArchive:
    def __init__(self, root, status):
        self.root = Path(root)
        self.delivery_path = self.root/'delivery.json'
        self.delivery_sha256 = sha256_file(self.delivery_path)
        document = json.loads(self.delivery_path.read_text())
        if (document.get('status') != status or document.get('exit_code') != 0
                or document.get('copied_files_verified') is not True):
            raise ValueError('CA3_DATASET_COMPLETED_DELIVERY_REQUIRED')
        self.rows = {row['path']: row for row in document['files']}
        if len(self.rows) != len(document['files']):
            raise ValueError('CA3_DATASET_DUPLICATE_DELIVERY_PATH')

    def file(self, relative):
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('CA3_DATASET_INPUT_PATH_INVALID')
        path = self.root/name
        row = self.rows[relative]
        if (path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve())
                or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']):
            raise ValueError('CA3_DATASET_BOUND_INPUT_CHANGED:'+relative)
        return path

    def json(self, relative):
        return json.loads(self.file(relative).read_text())


def _source_parameters(native, *, base_main_sha256=None, include_config_path=True):
    parameters = dict(framework_revision=native['framework_revision'], author_revision=native['author_revision'],
        source_file_byte_checks=deepcopy(native['source_file_byte_checks']),
        coupled_extension_config=deepcopy(native['coupled_extension_config']),
        conductance_memory_overlay=native.get('conductance_memory_overlay'), integration=native['integration'])
    if base_main_sha256 is not None:
        row = next(row for row in parameters['source_file_byte_checks'] if row['path'].startswith('src/main_ca3_'))
        if (row.pop('pre_realization_staged_sha256') != base_main_sha256
                or row.pop('source_realization_transformation_verified') is not True):
            raise ValueError('CA3_DATASET_VERIFIED_SOURCE_NORMALIZATION_REQUIRED')
        row['staged_sha256'] = base_main_sha256
    if not include_config_path:
        parameters['coupled_extension_config'].pop('path')
    return parameters


def source_units(native, groups, *, realization_source=None, reference_native=None):
    validate_source_reset_contract(native)
    identity = source_identity(coupled_extension=True)
    n = identity['core_units']
    populations = np.empty(n, dtype='U32')
    fields = np.full(n, -1, dtype=np.int64)
    names = ['CA3_assembly_1', 'CA3_assembly_2', 'CA3_assembly_3', 'CA3_Pyramidal_other']
    for group in groups:
        populations[group.start:group.stop] = group.name
        if group.name == PYRAMIDAL:
            fields[group.start:group.stop] = 3
            for i, start in enumerate(identity['assembly_starts']):
                fields[group.start+start:group.start+start+identity['assembly_units']] = i
        else:
            fields[group.start:group.stop] = len(names)
            names.append(group.name)
    if np.any(fields < 0) or len(names) != 11:
        raise ValueError('CA3_DATASET_COMPLETE_SOURCE_FIELD_MAP_REQUIRED')
    if 'source_realization' in native:
        spec = from_record(native['source_realization']['realization'])
        if realization_source is None:
            raise ValueError('CA3_DATASET_REALIZATION_SOURCE_REQUIRED')
        verified = verify_retrieval_program(Path(realization_source).read_bytes(), spec)
        parameters = _source_parameters(native, base_main_sha256=verified['base_main_sha256'],
                                        include_config_path=False)
        if reference_native is not None:
            # The old public identity included a scratch path. Compare semantic
            # content first, then retain that already published identity exactly.
            _source_proof(reference_native, coupled_extension=True)
            if (native['normalized_input_sha256'] != reference_native['normalized_input_sha256']
                    or parameters != _source_parameters(reference_native, include_config_path=False)):
                raise ValueError('CA3_DATASET_CIRCUIT_REFERENCE_PARAMETERS_DIFFER')
            parameters = _source_parameters(reference_native)
    else:
        if reference_native is not None:
            raise ValueError('CA3_DATASET_CIRCUIT_REFERENCE_REQUIRES_REALIZATION')
        parameters = _source_parameters(native)
    parameter_id = hashlib.sha256(canonical_json(parameters).encode()).hexdigest()
    provenance = dict(connectivity_identity=native['normalized_input_sha256'], parameter_identity=parameter_id,
        connectivity_identity_kind='normalized checkpoint: topology and learned initial weights; not pure topology or microscopic restart')
    return CircuitUnits('ca3_'+native['normalized_input_sha256'][:16]+'_'+parameter_id[:8],
        np.arange(n, dtype=np.int64), populations, fields, tuple(names),
        'source cell types and Pyramidal-local contiguous 550-cell memory assemblies; no anatomical coordinates',
        SOURCE_REVISION, circuit_provenance=provenance)


def _validate_commands(commands, expected):
    if set(commands) != set(expected) or any(not np.array_equal(commands[k], expected[k]) for k in expected):
        raise ValueError('CA3_DATASET_ORDERED_SOURCE_COMMANDS_CHANGED')


def source_controls(groups, pattern, commands, *, cue_omission=False):
    """Project identical repeated set operations to presence, never impulse sums."""
    pyr = next(group for group in groups if group.name == PYRAMIDAL)
    expected = control.source_commands(pattern, pyr.start, coupled_extension=True)
    if cue_omission:
        expected = {key: value[:0] for key, value in expected.items()}
    _validate_commands(commands, expected)
    identity = source_identity(coupled_extension=True)
    targets = np.concatenate([np.arange(pyr.start+start, pyr.start+start+identity['eligible_units'])
                              for start in identity['assembly_starts']])
    values = np.zeros((DURATION_MS, len(targets)+len(groups)+2), dtype=np.float64)
    columns = np.searchsorted(targets, commands['unit_ids'])
    if np.any(columns >= len(targets)) or not np.array_equal(targets[columns], commands['unit_ids']):
        raise ValueError('CA3_DATASET_COMMAND_TARGET_OUTSIDE_SOURCE_ELIGIBILITY')
    values[commands['time_ms'], columns] = 1.0
    features = [dict(name=f'set_current_global_unit_{int(unit)}', unit='dimensionless', operation='set',
                     global_unit_id=str(int(unit)), assignment_value='10000', assignment_unit='pA',
                     semantics='presence of at least one source assignment during this native millisecond') for unit in targets]
    for group in groups:
        features.append(dict(name=group.name+'_refresh_background', unit='dimensionless',
                             semantics='1: draw each native millisecond; 0: hold previous realized current'))
    features.extend([dict(name='CA3_Pyramidal_cpp_lognormal_s', unit='dimensionless'),
                     dict(name='CA3_Pyramidal_background_cap_enabled', unit='dimensionless')])
    for t in range(DURATION_MS):
        _, cue = control.source_phase(t)
        for i, group in enumerate(groups):
            values[t, len(targets)+i] = int(not cue or group.name == PYRAMIDAL)
        values[t, -2:] = (1.0 if cue else control.BACKGROUND_SIGMA[PYRAMIDAL], int(not cue))
    return np.arange(DURATION_MS, dtype=np.int64), values, tuple(features)


def _events(primary, recording, groups, saved):
    rows = {row['group']: row for row in recording['files']}
    if len(rows) != 8 or len(recording['files']) != 8:
        raise ValueError('CA3_DATASET_EIGHT_EVENT_FILES_REQUIRED')
    ticks, units = [], []
    for group in groups:
        row = rows[group.name]
        relative = 'native_run/results/spk_'+group.name+'.dat'
        path = primary.file(relative)
        if row['relative_path'] != relative or row['sha256'] != primary.rows[relative]['sha256']:
            raise ValueError('CA3_DATASET_EVENT_BINDING_CHANGED')
        local, per_ms, count = np.zeros(group.units, dtype=np.int64), np.zeros(DURATION_MS, dtype=np.int64), 0
        for block in spike_blocks(path, units=group.units, stop_ms=DURATION_MS):
            np.add.at(local, block[:, 1], 1); np.add.at(per_ms, block[:, 0], 1)
            ticks.append(block[:, 0].astype(np.int64)); units.append(block[:, 1].astype(np.int64)+group.start)
            count += len(block)
        if (count != row['records'] or not np.array_equal(local, saved['unit_counts'][group.start:group.stop])
                or not np.array_equal(per_ms, saved['population_counts_per_ms'][:, group.group_id])):
            raise ValueError('CA3_DATASET_RAW_COUNTS_CHANGED')
    raw_ticks = np.concatenate(ticks) if ticks else np.empty(0, dtype=np.int64)
    raw_units = np.concatenate(units) if units else np.empty(0, dtype=np.int64)
    if len(raw_ticks) != recording['records']:
        raise ValueError('CA3_DATASET_EVENT_DENOMINATOR_CHANGED')
    order = np.argsort(raw_ticks, kind='stable')
    return raw_ticks[order], raw_units[order], order


def verify_export(dataset, episode, units):
    row = dataset.episodes[episode.episode_id]
    shard = dataset.shards[row['shard']]
    with h5py.File(dataset._verified(shard['path'], shard['sha256']), 'r') as source:
        stored = source[episode.episode_id]
        for name in ('spike_ticks', 'spike_unit_indices', 'control_ticks', 'control_values'):
            if not np.array_equal(stored[name][:], getattr(episode, name)):
                raise ValueError('CA3_DATASET_RAW_ROUNDTRIP_CHANGED:'+name)
        for name, value in episode.privileged.items():
            if not np.array_equal(stored['privileged'][name][:], value):
                raise ValueError('CA3_DATASET_PROVENANCE_ROUNDTRIP_CHANGED:'+name)
        metadata = json.loads(stored.attrs['metadata_json'])
        if metadata['seeds'] != episode.seeds or metadata['source_record'] != episode.source_record:
            raise ValueError('CA3_DATASET_METADATA_ROUNDTRIP_CHANGED')
    start, checked, events = 0, 0, 0
    n = len(units.unit_ids)
    while start < DURATION_MS:
        bins = min(VERIFY_WINDOW_BINS, DURATION_MS-start)
        if DURATION_MS-start-bins == 1:
            bins -= 1
        past = bins//2
        view = dataset.window(episode.episode_id, boundary_tick=start+past, history_ticks=past,
                              horizon_ticks=bins-past, bin_ticks=1)
        left, right = np.searchsorted(episode.spike_ticks, [start, start+bins])
        ticks = episode.spike_ticks[left:right]-start
        indices = episode.spike_unit_indices[left:right]
        expected = np.bincount(ticks*n+indices, minlength=bins*n).reshape(bins, n)
        actual = np.concatenate([view['history'], view['targets']])
        if not np.array_equal(actual, expected):
            raise ValueError('CA3_DATASET_ALL_UNIT_COUNTS_CHANGED')
        field_count = np.bincount(ticks*len(units.field_names)+units.field_index[indices],
                                 minlength=bins*len(units.field_names)).reshape(bins, -1)
        if (not np.array_equal(np.concatenate([view['history_fields'], view['target_fields']]), field_count)
                or not np.array_equal(view['unit_ids'], units.unit_ids)
                or not np.array_equal(view['known_controls'], episode.control_values[start:start+bins])):
            raise ValueError('CA3_DATASET_FIELDS_OR_NATIVE_COMMANDS_CHANGED')
        if 'privileged' in view or any(name in view for name in episode.privileged):
            raise ValueError('CA3_DATASET_PRIVATE_PROVENANCE_EXPOSED')
        checked += actual.size; events += int(actual.sum()); start += bins
    if events != len(episode.spike_ticks):
        raise ValueError('CA3_DATASET_RECOUNT_DENOMINATOR_CHANGED')
    return dict(status='PASS', event_records=events, all_unit_bin_values_checked=checked,
                lossless_control_bin_ticks=1, fields_checked=len(units.field_names),
                duplicated_assignments_preserved_in_provenance=True, future_realized_currents_in_model_inputs=False)


def bound_cue_omission(omission, contrast, source_native, source_proof, analysis, groups):
    """Consume the measured pair without rerunning counters or current replay."""
    measured = contrast.json('analysis/report.json')
    n = source_identity(coupled_extension=True)['core_units']
    required = dict(status='MEASURED_CA3_CUE_OMISSION_CONTRAST', units=n, duration_ms=DURATION_MS,
        scope='development_whole_schedule_contrast', pre_cue_events_equal=True,
        pre_cue_interval_ms=[0, 520], configured_initialization_identity_equal=True,
        backend_library_bytes_equal=True, source_counter_and_current_qualification_reused=True)
    current = dict(intervals_compared=DURATION_MS, current_values_compared=n*DURATION_MS,
        mismatched_values=0, exact_float32_bits_equal=True, tolerance=0)
    counters = dict(unit_millisecond_counts_compared=n*DURATION_MS, native_intervals_compared=DURATION_MS)
    if (any(measured.get(k) != v for k, v in required.items())
            or any(measured.get('omission_current_checks', {}).get(k) != v for k, v in current.items())
            or any(measured.get('omission_counter_checks', {}).get(k) != v for k, v in counters.items())):
        raise ValueError('CA3_DATASET_COMPLETE_CUE_CONTRAST_REQUIRED')
    bindings = measured['input_sha256']
    if (bindings['source_reconciliation'] != analysis.rows['reconciliation.json']['sha256']
            or bindings['source']['native_report.json'] != source_proof['observed_report_sha256']
            or bindings['source']['native_observation.bin'] != source_proof['native_observation_sha256']):
        raise ValueError('CA3_DATASET_CONTRAST_SOURCE_BINDING_CHANGED')
    native = omission.json('native_report.json')
    recording = omission.json('observations/recording.json')
    path = omission.file('observations/population_counts.npz')
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    names = ('native_report.json', 'observations/recording.json', 'observations/population_counts.npz')
    hashes = {name: omission.rows[name]['sha256'] for name in names}
    if any(bindings['omission'][name] != digest for name, digest in hashes.items()):
        raise ValueError('CA3_DATASET_CONTRAST_OMISSION_BINDING_CHANGED')
    if (native.get('retrieval_condition') != 'cue_omission'
            or native.get('status') != 'COUPLED_EXTENSION_RETRIEVAL_EXIT_ZERO'
            or native.get('observed_native_duration_ms') != DURATION_MS
            or native.get('exit_code') != 0
            or native.get('host_request_runtime_current_equal') is not True
            or native.get('native_observation_stream_validated') is not True
            or recording['native_report_sha256'] != hashes['native_report.json']
            or recording['observation_sha256'] != hashes['observations/population_counts.npz']
            or native['native_observation_sha256'] != bindings['omission']['native_observation.bin']
            or any(native[k] != source_native[k] for k in ('normalized_input_sha256', 'pattern_sha256',
                'formation_report_sha256', 'formation_replay_sha256', 'normalization_report_sha256',
                'normalization_replay_sha256', 'raw65p_sha256', 'coupled_extension'))
            or any(native['coupled_extension_config'][k] != source_native['coupled_extension_config'][k]
                   for k in ('sha256', 'resolved'))):
        raise ValueError('CA3_DATASET_CUE_OMISSION_IDENTITY_CHANGED')
    fields = ('group_id', 'name', 'start', 'stop')
    if ([tuple(row[k] for k in fields) for row in recording['groups']]
            != [tuple(getattr(g, k) for k in fields) for g in groups]
            or not np.array_equal(saved['unit_ids'], np.arange(n))
            or saved['unit_counts'].shape != (n,)
            or saved['population_counts_per_ms'].shape != (DURATION_MS, len(groups))
            or not np.array_equal(saved['group_names'], [g.name for g in groups])):
        raise ValueError('CA3_DATASET_CUE_OMISSION_UNIT_MAP_CHANGED')
    return native, recording, saved, hashes


def export(primary_archive, observation_analysis_archive, output, *, input_origins=None,
           omission_archive=None, cue_contrast_archive=None, source_realization_config=None,
           circuit_reference_primary=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (omission_archive is None) != (cue_contrast_archive is None):
        raise ValueError('CA3_DATASET_OMISSION_AND_CONTRAST_REQUIRED_TOGETHER')
    primary = BoundArchive(primary_archive, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    analysis = BoundArchive(observation_analysis_archive, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    archives = dict(primary=primary, observation_analysis=analysis)
    cue_omission = omission_archive is not None
    if cue_omission and source_realization_config is not None:
        raise ValueError('CA3_DATASET_REALIZATION_CUE_OMISSION_NOT_QUALIFIED')
    if circuit_reference_primary is not None:
        reference = BoundArchive(circuit_reference_primary, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
        archives['circuit_reference_primary'] = reference
    if cue_omission:
        omission = BoundArchive(omission_archive, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
        contrast = BoundArchive(cue_contrast_archive, 'MEASURED_CA3_CUE_OMISSION_CONTRAST')
        archives.update(omission=omission, cue_contrast=contrast)
    origins = {role: str(archive.root.resolve()) for role, archive in archives.items()}
    if input_origins is not None:
        declared = json.loads(Path(input_origins).read_text())
        if set(declared) != set(origins):
            raise ValueError('CA3_DATASET_ORIGIN_ROLES_CHANGED')
        for role, archive in archives.items():
            origin = declared[role]
            durable = Path(origin['archive'])/'delivery.json'
            if (origin['delivery_sha256'] != archive.delivery_sha256 or not durable.is_file()
                    or sha256_file(durable) != archive.delivery_sha256):
                raise ValueError('CA3_DATASET_DURABLE_ORIGIN_BINDING_CHANGED')
            origins[role] = str(durable.parent.resolve())
    build, native, recording, groups, saved, pattern, hashes = _bound_inputs(
        primary.root, coupled_extension=True, source_realization_config=source_realization_config)
    reset_contract = validate_source_reset_contract(native)
    if build.get('source_initial_state_contract') != native.get('source_initial_state_contract'):
        raise ValueError('CA3_DATASET_INITIAL_STATE_BUILD_BINDING_CHANGED')
    validate_source_reset_contract(build)
    for name, digest in hashes.items():
        if primary.rows[name]['sha256'] != digest or (primary.root/name).stat().st_size != primary.rows[name]['bytes']:
            raise ValueError('CA3_DATASET_PRIMARY_DELIVERY_CHANGED:'+name)
    result = analysis.json('analysis.json')
    proof = analysis.json('reconciliation.json')
    delivered = analysis.json('control/report.json')
    flags = ('event_multisets_equal', 'actual_current_trace_verified', 'native_internal_spike_counter_verified',
             'terminal_native_clock_independently_verified', 'observation_source_execution_equivalence_qualified')
    if (result.get('status') != 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL' or result.get('exit_code') != 0
            or proof.get('status') != 'PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY'
            or proof.get('units') != 168106 or proof.get('duration_ms') != DURATION_MS
            or any(proof.get(key) is not True for key in flags)
            or proof['original_report_sha256'] != hashes['native_report.json']
            or proof['original_recording_sha256'] != hashes['observations/recording.json']
            or proof['implementation_sha256'] != sha256_file(Path(reconciliation.__file__))
            or delivered.get('status') != 'VERIFIED_COUPLED_EXTENSION_CA3_CONTROL_DELIVERY'
            or delivered['implementation_sha256'] != sha256_file(Path(control.__file__))
            or delivered.get('current_units', {}).get('model_specific_physical_unit') != 'pA'
            or delivered.get('numeric_conversion_applied') is not False
            or delivered['input_sha256']['reconciliation'] != analysis.rows['reconciliation.json']['sha256']
            or delivered['input_sha256']['native_report.json'] != proof['observed_report_sha256']
            or delivered['input_sha256']['native_observation.bin'] != proof['native_observation_sha256']
            or any(obj.get('coupled_extension') != native['coupled_extension']
                   or not reconciliation.same_coupled_configuration(
                       obj['coupled_extension_config'], native['coupled_extension_config'])
                   for obj in (proof, delivered))):
        raise ValueError('CA3_DATASET_BOUND_OBSERVATION_AND_CONTROL_REQUIRED')
    for obj in (proof, delivered):
        if any(obj.get(key) != native.get(key) for key in ('source_realization', 'realization_circuit_binding')):
            raise ValueError('CA3_DATASET_OBSERVATION_REALIZATION_BINDING_CHANGED')
    realization = None
    if source_realization_config is not None:
        realization = from_record(native['source_realization']['realization'])
        identity = native.get('realization_circuit_binding', {})
        required = dict(circuit_id=realization.circuit_id, circuit_seed=realization.circuit_seed,
            block_id=realization.block_id, normalized_input_sha256=native['normalized_input_sha256'],
            formation_report_sha256=native['formation_report_sha256'],
            new_circuit_claimed=False, microscopic_state_restarted_from_checkpoint=False)
        if any(identity.get(k) != value for k, value in required.items()):
            raise ValueError('CA3_DATASET_REALIZATION_CIRCUIT_BINDING_CHANGED')
        if realization.split != 'development':
            raise ValueError('CA3_DATASET_DEVELOPMENT_REALIZATION_REQUIRED')
    reference_native = None
    if circuit_reference_primary is not None:
        reference_native = reference.json('native_report.json')
        reference_build = reference.json('build_report.json')
        reference_recording = reference.json('observations/recording.json')
        if (reference_native.get('status') != 'COUPLED_EXTENSION_RETRIEVAL_EXIT_ZERO'
                or reference_native.get('exit_code') != 0
                or reference_native.get('expected_duration_ms') != DURATION_MS
                or reference_native.get('normalized_input_qualified') is not True
                or reference_build.get('source_file_byte_checks') != reference_native['source_file_byte_checks']
                or reference_recording.get('native_report_sha256') != reference.rows['native_report.json']['sha256']
                or any(reference_recording.get(k) != recording[k]
                       for k in ('groups', 'units', 'duration_ms', 'time_resolution_ms'))):
            raise ValueError('CA3_DATASET_CIRCUIT_REFERENCE_IDENTITY_CHANGED')
    with np.load(analysis.file('control/cue_command_events.npz'), allow_pickle=False) as source:
        commands = {key: source[key] for key in source.files}
    for name in ('cue_command_events.npz', 'background_schedule.csv'):
        path = analysis.file('control/'+name)
        if delivered['artifacts'][name] != sha256_file(path):
            raise ValueError('CA3_DATASET_CONTROL_ARTIFACT_CHANGED')
    with analysis.file('control/background_schedule.csv').open() as stream:
        schedule = list(csv.DictReader(stream))
    expected = [{key: '' if value is None else str(value) for key, value in row.items()}
                for row in control.background_schedule(groups)]
    if schedule != expected:
        raise ValueError('CA3_DATASET_SOURCE_BACKGROUND_SCHEDULE_CHANGED')
    # Derive the existing circuit and configured block from the original source
    # primary, never from the condition-specific main program or scratch path.
    units = source_units(native, groups,
        realization_source=primary.file('realization_source.cpp') if realization is not None else None,
        reference_native=reference_native)
    seeds = dict(source_carlsim_seed=10, source_host_srand_seed=3)
    condition = dict(circuit_id=units.circuit_id, pattern_sha256=native['pattern_sha256'],
                     overlay_source_sha256=native['overlay_source_sha256'], seeds=seeds)
    if realization is not None:
        seeds.update(source_carlsim_seed=realization.circuit_seed,
                     source_uniform_seed=realization.uniform_seed, source_engine_seed=realization.engine_seed)
        # Jobs, observer replays and staging paths do not define another block.
        condition = dict(circuit_id=units.circuit_id, pattern_sha256=native['pattern_sha256'],
                         seeds=seeds, source_realization=native['source_realization'])
    key = hashlib.sha256(canonical_json(condition).encode()).hexdigest()[:20]
    source_reference_hash = hashes['native_report.json']
    if cue_omission:
        pyr = next(group for group in groups if group.name == PYRAMIDAL)
        _validate_commands(commands, control.source_commands(pattern, pyr.start, coupled_extension=True))
        native, recording, saved, hashes = bound_cue_omission(
            omission, contrast, native, proof, analysis, groups)
        primary = omission
        commands = {name: value[:0] for name, value in commands.items()}
    ticks, values, features = source_controls(groups, pattern, commands, cue_omission=cue_omission)
    spike_ticks, spike_units, ordinal = _events(primary, recording, groups, saved)
    record = dict(complete_unit_recording=True, simulation_completed=True,
        recorded_until_tick=DURATION_MS, expected_duration_ticks=DURATION_MS, core_unit_count=len(units.unit_ids),
        source_revision=SOURCE_REVISION, circuit_provenance=units.circuit_provenance,
        primary_delivery_sha256=primary.delivery_sha256, primary_native_report_sha256=hashes['native_report.json'],
        observation_analysis_delivery_sha256=analysis.delivery_sha256,
        reconciliation_sha256=analysis.rows['reconciliation.json']['sha256'],
        control_report_sha256=analysis.rows['control/report.json']['sha256'],
        source_event_file_order=[group.name for group in groups], native_monitor_tick_ms=1,
        numerical_integration=native['integration'], normalized_payload_rehashed_by_exporter=False,
        source_random_streams='source seeds10/3; initialization and external noise not independently separated',
        microscopic_restart_available=False, observed_replay_counted_as_episode=False,
        provenance_array_roles=dict(source_event_ordinal='position in native-group-order concatenated raw records before stable time sort',
            raw_commands='ordered known source assignments; repeated identical sets project to presence in model controls'),
        background_control_constants=dict(cpp_lognormal_m=4, source_sigma=control.BACKGROUND_SIGMA,
            uniform_multiplier='float32(drand48())', background_cap_pA=10000,
            inhibitory_cap='applies on refresh; held during every cue',
            refresh_unit='each native millisecond, not each control row transition'))
    if realization is not None:
        record.update(source_realization=native['source_realization'],
            realization_circuit_binding=native['realization_circuit_binding'],
            source_realization_config=dict(sha256=hashes['source_realization.yaml'], resolved=realization.record()),
            source_random_streams='explicit post-setup host noise; shared uniform stream and common seed for nine persistent engines',
            independent_noise_qualified=False, independent_microscopic_states_qualified=False)
    if reset_contract is not None:
        record['source_initial_state_contract'] = reset_contract
        record['source_noise_block_scope'] = 'post-setup external noise; deterministic reset is not initial-state diversity'
    if reference_native is not None:
        record.update(circuit_reference_archive=origins['circuit_reference_primary'],
            circuit_reference_delivery_sha256=reference.delivery_sha256,
            circuit_reference_native_report_sha256=reference.rows['native_report.json']['sha256'])
    if cue_omission:
        record.update(retrieval_condition='cue_omission', source_reference_native_report_sha256=source_reference_hash,
            cue_contrast_delivery_sha256=contrast.delivery_sha256,
            cue_contrast_report_sha256=contrast.rows['analysis/report.json']['sha256'],
            shared_source_circuit_and_configured_block=True, independent_noise_realization_claimed=False)
        record['provenance_array_roles']['raw_commands'] = 'empty: no cue overwrite executed; background laws retained'
    private = dict(source_event_ordinal=ordinal)
    private.update({'raw_command_'+name: value for name, value in commands.items()})
    episode_prefix = 'ca3_cue_omission_' if cue_omission else 'ca3_source_'
    episode = CircuitEpisode(episode_prefix+key, units.circuit_id, 'ca3_source_block_'+key, 'development',
        DURATION_MS, 1.0, spike_ticks, spike_units, ticks, values, features, seeds=seeds,
        source_record=record, privileged=private, task_outcome='not_scientifically_assessed')
    binding = dict(schema=SCHEMA, scope='source_reproduction', scenario_id='ca3_pattern_completion',
        source_revision=SOURCE_REVISION, sample_unit='complete_mechanistic_episode',
        primary_archive=origins['omission' if cue_omission else 'primary'], observation_analysis_archive=origins['observation_analysis'],
        source_bindings=dict(primary_delivery_sha256=primary.delivery_sha256, observation_analysis_delivery_sha256=analysis.delivery_sha256),
        control_semantics='non-additive set-current presence and known background law; no future realized currents',
        control_window_reduction='time mean; one native tick retains exact assignment timing',
        exporter_sha256=sha256_file(Path(__file__)))
    if realization is not None:
        binding['source_realization'] = native['source_realization']
        binding['source_bindings']['source_realization_config_sha256'] = hashes['source_realization.yaml']
    if reset_contract is not None:
        binding['source_initial_state_contract'] = reset_contract
    if reference_native is not None:
        binding['circuit_reference_archive'] = origins['circuit_reference_primary']
        binding['source_bindings']['circuit_reference_delivery_sha256'] = reference.delivery_sha256
    if cue_omission:
        binding.update(retrieval_condition='cue_omission', source_reference_archive=origins['primary'],
            cue_contrast_archive=origins['cue_contrast'])
        binding['source_bindings'].update(source_reference_delivery_sha256=archives['primary'].delivery_sha256,
            cue_contrast_delivery_sha256=contrast.delivery_sha256)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status='EXPORTING', new_source_simulations=0, production_episodes=0,
        source_episodes_added_to_campaign=0, scientific_fidelity_qualified=False, source_fidelity_qualified=False)
    try:
        with CircuitArchiveWriter(output/'dataset', binding, target_shard_bytes=1) as writer:
            writer.register_circuit(units); writer.add(episode)
        verified = verify_export(CircuitDataset(output/'dataset', splits=('development',)), episode, units)
        report.update(status='EXPORTED_VERIFIED_CA3_DEVELOPMENT_DATASET', complete_episodes=1,
            episode_id=episode.episode_id, units=len(units.unit_ids), control_features=len(features),
            retrieval_condition='cue_omission' if cue_omission else 'source',
            circuit_id=episode.circuit_id, block_id=episode.block_id,
            verification=verified, dataset_catalog_sha256=sha256_file(output/'dataset/dataset.json'))
    except Exception as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'observation-analysis', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--input-origins', type=Path)
    parser.add_argument('--source-realization-config', type=Path, help='Explicit validated source-noise configuration')
    parser.add_argument('--circuit-reference-primary', type=Path, help='Completed legacy primary whose circuit identity is retained')
    parser.add_argument('--cue-omission', type=Path, help='Observed omission archive; requires --cue-contrast')
    parser.add_argument('--cue-contrast', type=Path, help='Successful paired contrast archive')
    args = parser.parse_args(argv)
    print(json.dumps(export(args.primary, args.observation_analysis, args.output,
        input_origins=args.input_origins, omission_archive=args.cue_omission,
        cue_contrast_archive=args.cue_contrast, source_realization_config=args.source_realization_config,
        circuit_reference_primary=args.circuit_reference_primary), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

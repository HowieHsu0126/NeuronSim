"""Measure the complete source cue schedule versus its observed omission.

This development comparison retains all source phases and unit identities. It
does not treat the three successive cues as independent replicates, perform
inference, or qualify biological fidelity. Source current/counter qualification
is reused; omission currents and native counters are checked independently.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import ca3_host_current_replay as host_replay
from . import ca3_native_observation as observer
from .ca3_cue_omission import validate_report as validate_omission
from .ca3_native_observation_replay import BoundRun, _events, _compare_counters
from .ca3_retrieval_readouts import (
    AUTHOR_MAIN_SHA256, CONFIG_SHA256, DURATION_MS, MAIN_PATH, PYRAMIDAL,
    SETUP_SHA256, _source_proof, source_identity, source_masks,
)
from .carlsim_records import validate_coupled_identity
from .provenance import sha256_file

LIBRARIES = ('lib/libcarlsim.so.6.0.0', 'lib/libcarlsim-stopwatch.so', 'lib/libcurand.so.10')
PHASE_EDGES_MS = (0, 520, 540, 750, 770, 980, 1000, 1001)
PHASE_NAMES = ('initial_background', 'cue_1', 'background_after_cue_1', 'cue_2',
               'background_after_cue_2', 'cue_3', 'background_after_cue_3')
INPUT_FIELDS = ('normalized_input_sha256', 'raw65p_sha256', 'formation_report_sha256',
                'formation_replay_sha256', 'normalization_report_sha256', 'normalization_replay_sha256')


def _json(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError('CA3_CONTRAST_OBJECT_REQUIRED')
    return result


def _delivery(root, status):
    document = _json(root/'delivery.json')
    if (document.get('status') != status or document.get('exit_code') != 0
            or document.get('copied_files_verified') is not True):
        raise ValueError('CA3_CONTRAST_COMPLETE_DELIVERY_REQUIRED:'+str(root))
    rows = {row['path']: row for row in document['files']}
    if len(rows) != len(document['files']):
        raise ValueError('CA3_CONTRAST_DUPLICATE_DELIVERY_PATH')
    return rows


def _file(root, rows, relative):
    path, row = root/relative, rows[relative]
    if path.is_symlink() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
        raise ValueError('CA3_CONTRAST_BOUND_FILE_CHANGED:'+relative)
    return path


def bind_run(root, *, omission, neuron_substeps=None):
    if omission and neuron_substeps is not None:
        raise ValueError('CA3_REFINEMENT_REQUIRES_SOURCE_CUES')
    root = Path(root)
    rows = _delivery(root, 'COUPLED_RETRIEVAL_AND_NATIVE_RECORDS_VALIDATED')
    names = ('build_report.json', 'native_report.json', 'manifest.json',
             'observations/recording.json', 'observations/population_counts.npz',
             'trainPat.csv', 'native_observation.bin', 'bin/ca3_retrieval', *LIBRARIES)
    paths = {name: _file(root, rows, name) for name in names}
    hashes = {name: rows[name]['sha256'] for name in names}
    build, native, manifest, recording = (_json(paths[name]) for name in names[:4])
    declared = {row['path']: row for row in manifest['files']}
    if len(declared) != len(manifest['files']):
        raise ValueError('CA3_CONTRAST_DUPLICATE_RUNTIME_PATH')
    if manifest.get('scope') != 'coupled_extension_retrieval_runtime' or manifest.get('native_observation_enabled') is not True:
        raise ValueError('CA3_CONTRAST_OBSERVED_RUNTIME_REQUIRED')
    for name in ('build_report.json', 'trainPat.csv', 'bin/ca3_retrieval', *LIBRARIES):
        if declared[name]['sha256'] != hashes[name] or declared[name]['bytes'] != rows[name]['bytes']:
            raise ValueError('CA3_CONTRAST_RUNTIME_DELIVERY_DIFFERS:'+name)
    identity = source_identity(coupled_extension=True, neuron_substeps=neuron_substeps)
    condition = 'cue_omission' if omission else 'source'
    common = dict(scope='coupled_extension_retrieval', expected_duration_ms=DURATION_MS,
        core_neurons=identity['core_units'], populations=identity['populations'],
        pattern_sha256=identity['pattern_sha256'], pattern_integer_count=identity['pattern_integer_count'],
        original_author_main_sha256=AUTHOR_MAIN_SHA256, integration='RUNGE_KUTTA4_' + str(neuron_substeps or 5), delay_ms=1,
        retrieval_program_changed_only_in_input_path=False)
    for record in (build, native):
        validate_coupled_identity(record, neuron_substeps=neuron_substeps)
        if (record.get('retrieval_condition', 'source') != condition
                or any(record.get(k) != v for k, v in common.items())):
            raise ValueError('CA3_CONTRAST_DECLARED_CONDITION_REQUIRED')
        contract = record['retrieval_input_contract']
        required = dict(effective_input_binding=identity['input_network'], source_path_binding_only=False,
            source_carlsim_seed=10, source_host_srand_seed=3, source_estdp_enabled=True,
            fresh_coupled_checkpoint_required=True, raw_network65p_substitution_allowed=False)
        if any(contract.get(k) != v for k, v in required.items()):
            raise ValueError('CA3_CONTRAST_SOURCE_INITIALIZATION_OR_PLASTICITY_CHANGED')
        if omission:
            validate_omission(record, population_multiplier=2)
            indexed = {row['path']: row for row in record['source_file_byte_checks']}
            if len(indexed) != len(record['source_file_byte_checks']):
                raise ValueError('CA3_CONTRAST_DUPLICATE_SOURCE_PATH')
            expected = {'generateCONFIGStateSTP.h': (CONFIG_SHA256, identity['config_sha256'], False),
                        'generateSETUPStateSTP.h': (SETUP_SHA256, SETUP_SHA256, True)}
            for name, (original, effective, preserved) in expected.items():
                row = indexed[name]
                if (row['source_sha256'] != original or row['staged_sha256'] != effective
                        or row['source_bytes_preserved'] is not preserved
                        or row['source_path_binding_only'] is not False
                        or row['coupled_extension_transformation_verified'] is not True):
                    raise ValueError('CA3_CONTRAST_EXTRA_CONFIGURATION_CHANGE:'+name)
            if indexed[MAIN_PATH]['source_sha256'] != AUTHOR_MAIN_SHA256:
                raise ValueError('CA3_CONTRAST_AUTHOR_MAIN_CHANGED')
        else:
            _source_proof(record, coupled_extension=True, neuron_substeps=neuron_substeps)
    if neuron_substeps is not None and (manifest.get('numerical_refinement') != native['numerical_refinement']
            or recording.get('numerical_refinement') != native['numerical_refinement']):
        raise ValueError('CA3_REFINEMENT_RUNTIME_DECODER_IDENTITY_DIFFERS')
    if (build['status'] != 'COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED'
            or native['status'] != 'COUPLED_EXTENSION_RETRIEVAL_EXIT_ZERO' or native['exit_code'] != 0
            or native.get('native_run_executed') is not True or native.get('retrieval_executed') is not True
            or native.get('normalized_input_qualified') is not True
            or native.get('native_observation_stream_validated') is not True
            or native.get('observed_native_duration_ms') != DURATION_MS
            or native.get('host_request_runtime_current_equal') is not True
            or native['runtime_manifest_sha256'] != hashes['manifest.json']
            or native['native_observation_sha256'] != hashes['native_observation.bin']
            or native['native_observation_bytes'] != rows['native_observation.bin']['bytes']
            or recording['native_report_sha256'] != hashes['native_report.json']
            or recording['observation_sha256'] != hashes['observations/population_counts.npz']
            or hashes['trainPat.csv'] != identity['pattern_sha256']):
        raise ValueError('CA3_CONTRAST_COMPLETED_NATIVE_OBSERVATION_REQUIRED')
    for record in (build, native):
        if record['binary_sha256'] != hashes['bin/ca3_retrieval']:
            raise ValueError('CA3_CONTRAST_EXECUTED_BINARY_CHANGED')
    units, groups = observer.read_header(paths['native_observation.bin'])
    if units != identity['core_units'] or {g.name: g.units for g in groups} != identity['populations']:
        raise ValueError('CA3_CONTRAST_SOURCE_POPULATIONS_REQUIRED')
    fields = ('group_id', 'name', 'start', 'stop')
    if [tuple(row[k] for k in fields) for row in recording['groups']] != [
            tuple(getattr(g, k) for k in fields) for g in groups]:
        raise ValueError('CA3_CONTRAST_TRACE_AND_EVENT_IDS_DIFFER')
    with np.load(paths['observations/population_counts.npz'], allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    if (not np.array_equal(saved['unit_ids'], np.arange(units))
            or saved['unit_ids'].dtype.kind not in 'iu'
            or saved['unit_counts'].shape != (units,)
            or saved['unit_counts'].dtype.kind not in 'iu' or np.any(saved['unit_counts'] < 0)
            or saved['population_counts_per_ms'].shape != (DURATION_MS, len(groups))
            or saved['population_counts_per_ms'].dtype.kind not in 'iu'
            or np.any(saved['population_counts_per_ms'] < 0)
            or not np.array_equal(saved['group_names'], [g.name for g in groups])):
        raise ValueError('CA3_CONTRAST_DECODER_UNIT_MAP_CHANGED')
    overlays = dict(build['native_observation_overlay']['overlay_sha256'])
    overlays.update(build['conductance_memory_overlay']['overlay_sha256'])
    return BoundRun(root, build, native, recording, groups, saved, hashes,
                    {name: row['sha256'] for name, row in declared.items()}, overlays,
                    dict(scope='explicit_condition_binding', condition=condition))


def bind_pair(source, omission, source_analysis, source_rng):
    left, right = bind_run(source, omission=False), bind_run(omission, omission=True)
    if left.groups != right.groups:
        raise ValueError('CA3_CONTRAST_PAIRED_UNIT_IDENTITIES_DIFFER')
    for key in (*INPUT_FIELDS, 'coupled_extension',
                'native_observation_overlay', 'observer_implementation_sha256',
                'conductance_memory_overlay', 'conductance_memory_implementation_sha256'):
        if left.native[key] != right.native[key]:
            raise ValueError('CA3_CONTRAST_PAIRED_INITIALIZATION_OR_BACKEND_DIFFERS:'+key)
    for key in ('sha256', 'resolved'):
        if left.native['coupled_extension_config'][key] != right.native['coupled_extension_config'][key]:
            raise ValueError('CA3_CONTRAST_PAIRED_CONFIGURATION_DIFFERS')
    paired = _json(right.root/'manifest.json')['paired_backend']
    if (paired.get('backend_library_bytes_equal') is not True
            or paired['manifest_sha256'] != left.hashes['manifest.json']
            or paired['source_build_sha256'] != left.hashes['build_report.json']):
        raise ValueError('CA3_CONTRAST_PAIRED_BACKEND_REFERENCE_DIFFERS')
    libraries = {row['path']: row['sha256'] for row in paired['libraries']}
    if set(libraries) != set(LIBRARIES) or any(
            left.hashes[name] != right.hashes[name] or libraries[name] != left.hashes[name] for name in LIBRARIES):
        raise ValueError('CA3_CONTRAST_BACKEND_LIBRARY_BYTES_DIFFER')
    engine, proofs = bind_source_evidence(left, source_analysis, source_rng)
    return left, right, engine, proofs


def bind_source_evidence(left, source_analysis, source_rng):
    """Reuse complete source counter/current qualification for a matched contrast."""
    source_analysis, source_rng = Path(source_analysis), Path(source_rng)
    rows = _delivery(source_analysis, 'PASS_COUPLED_OBSERVATION_RECONCILIATION_AND_CONTROL')
    reconciliation_path = _file(source_analysis, rows, 'reconciliation.json')
    reconciliation = _json(reconciliation_path)
    units = left.native['core_neurons']
    required = dict(status='PASS_COUPLED_EXTENSION_CA3_OBSERVATION_REPLAY', units=units,
        duration_ms=DURATION_MS, actual_current_trace_verified=True,
        native_internal_spike_counter_verified=True, event_multisets_equal=True,
        terminal_native_clock_independently_verified=True, observation_source_execution_equivalence_qualified=True,
        unit_millisecond_counts_compared=units*DURATION_MS, native_intervals_compared=DURATION_MS)
    if (any(reconciliation.get(k) != v for k, v in required.items())
            or reconciliation['observed_report_sha256'] != left.hashes['native_report.json']
            or reconciliation['native_observation_sha256'] != left.hashes['native_observation.bin']
            or reconciliation['normalized_input_sha256'] != left.native['normalized_input_sha256']
            or reconciliation['helper_implementation_sha256']['ca3_native_observation'] != sha256_file(Path(observer.__file__))):
        raise ValueError('CA3_CONTRAST_SOURCE_RECONCILIATION_REQUIRED')
    rows = _delivery(source_rng, 'VERIFIED_CA3_HOST_CURRENT_REPLAY')
    rng_path = _file(source_rng, rows, 'replay/report.json')
    engine = _file(source_rng, rows, 'bin/ca3_host_current_replay')
    rng = _json(rng_path)
    required = dict(status='VERIFIED_CA3_HOST_CURRENT_REPLAY', coupled_extension=True, units=units,
        intervals_compared=DURATION_MS, current_values_compared=DURATION_MS*units, mismatched_values=0,
        exact_float32_bits_equal=True, tolerance=0, source_host_current_rng_reconstructed=True,
        current_unit='pA', numeric_conversion_applied=False)
    contract = rng['rng_contract']
    if (any(rng.get(k) != v for k, v in required.items())
            or any(rng['input_sha256'][name] != left.hashes[name] for name in
                   ('native_report.json','native_observation.bin','trainPat.csv'))
            or rng['engine_sha256'] != rows['bin/ca3_host_current_replay']['sha256']
            or rng['cpp_source_sha256'] != sha256_file(host_replay.SOURCE_FILE)
            or contract['carlsim_srand48_seed'] != 10 or contract['pre_current_drand48_draws'] != 9*units
            or contract['background_intervals'] != 941 or contract['recall_intervals'] != 60
            or contract['fitted_state_or_parameters'] is not False):
        raise ValueError('CA3_CONTRAST_COMPLETE_SOURCE_RNG_REPLAY_REQUIRED')
    return engine, dict(source_reconciliation=sha256_file(reconciliation_path),
                                    source_rng_replay=sha256_file(rng_path), engine=rng['engine_sha256'])


def compare_prefix(source, omission):
    def ordered_prefix(events):
        selected = events[events[:, 0] < PHASE_EDGES_MS[1]]
        return selected[np.lexsort((selected[:, 1], selected[:, 0]))]
    left, right = ordered_prefix(source), ordered_prefix(omission)
    if not np.array_equal(left, right):
        raise ValueError('CA3_CONTRAST_PRE_CUE_EVENT_MULTISET_DIFFERS')
    return len(left)


def field_counts(events, groups):
    """Source populations plus six fixed eligibility subsets, including silence."""
    units = sum(g.units for g in groups)
    events = np.asarray(events)
    if (events.ndim != 2 or events.shape[1] != 2 or events.dtype.kind not in 'iu'
            or np.any(events < 0) or np.any(events[:, 0] >= DURATION_MS) or np.any(events[:, 1] >= units)):
        raise ValueError('CA3_CONTRAST_NATIVE_EVENT_BOUNDS_REQUIRED')
    names = [g.name for g in groups]
    members = [np.arange(g.start, g.stop) for g in groups]
    pyramidal = next(g for g in groups if g.name == PYRAMIDAL)
    masks = source_masks(pyramidal.units, coupled_extension=True)
    subsets = []
    for cue in range(3):
        for key, label in [('eligible_for_boost', 'source_eligible'), ('target_never_boosted', 'source_never_boosted')]:
            indices = np.flatnonzero(masks[key][cue])+pyramidal.start
            names.append(f'assembly_{cue+1}_{label}')
            members.append(indices)
            subsets.extend(indices.tolist())
    if len(subsets) != len(set(subsets)) or any(len(ids) == 0 for ids in members):
        raise ValueError('CA3_CONTRAST_FIXED_SOURCE_SUBSETS_INVALID')
    counts = np.zeros((DURATION_MS, len(names)), dtype=np.int64)
    for column, indices in enumerate(members):
        mask = np.zeros(units, dtype=bool)
        mask[indices] = True
        selected = events[mask[events[:, 1]], 0]
        counts[:, column] = np.bincount(selected, minlength=DURATION_MS)
    return names, np.asarray([len(ids) for ids in members]), counts


def rate_rows(names, sizes, source, omission, phase, start, stop):
    rows = []
    for column, (name, size) in enumerate(zip(names, sizes)):
        left = int(source[start:stop, column].sum())
        right = int(omission[start:stop, column].sum())
        scale = 1000.0/(int(size)*(stop-start))
        rows.append(dict(phase=phase, start_ms=start, stop_ms=stop, field=name, units=int(size),
            source_count=left, omission_count=right, source_minus_omission_count=left-right,
            source_hz=left*scale, omission_hz=right*scale, source_minus_omission_hz=(left-right)*scale))
    return rows


def phase_rows(names, sizes, source, omission):
    return [row for phase, start, stop in zip(PHASE_NAMES, PHASE_EDGES_MS[:-1], PHASE_EDGES_MS[1:])
            for row in rate_rows(names, sizes, source, omission, phase, start, stop)]


def analyze(source, omission, source_analysis, source_rng, output, *, input_origins=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='PREPARING', scope='development_whole_schedule_contrast',
        native_simulation_executed=False, production_episodes=0, scientific_fidelity_qualified=False,
        independent_rng_streams_qualified=False, inferential_statistics_performed=False,
        independent_cue_effects_claimed=False, future_realized_current_model_visible=False)
    try:
        left, right, engine, proofs = bind_pair(source, omission, source_analysis, source_rng)
        source_events, _ = _events(left, retain=True)
        omission_events, _ = _events(right, retain=True)
        counter_checks = {}
        report['omission_counter_checks'] = counter_checks
        _compare_counters(right.root/'native_observation.bin', omission_events, right.native['core_neurons'], counter_checks)
        empty_commands = dict(time_ms=np.empty(0, dtype=np.int32), unit_ids=np.empty(0, dtype=np.int64))
        current_checks = host_replay.compare_currents(engine, right.groups, empty_commands,
            observer.iter_intervals(right.root/'native_observation.bin', expected_stop_ms=DURATION_MS))
        report.update(omission_counter_checks=counter_checks, omission_current_checks=current_checks)
        if not current_checks['exact_float32_bits_equal']:
            raise ValueError('CA3_CONTRAST_OMISSION_SOURCE_RNG_REPLAY_DIFFERENT')
        prefix_events = compare_prefix(source_events, omission_events)
        names, sizes, source_counts = field_counts(source_events, left.groups)
        other_names, other_sizes, omission_counts = field_counts(omission_events, right.groups)
        if names != other_names or not np.array_equal(sizes, other_sizes):
            raise ValueError('CA3_CONTRAST_FIELD_IDENTITIES_DIFFER')
        rows = phase_rows(names, sizes, source_counts, omission_counts)
        with (output/'phase_counts.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(output/'counts.npz', time_ms=np.arange(DURATION_MS), field_names=np.asarray(names),
            field_unit_counts=sizes, source_field_counts=source_counts, omission_field_counts=omission_counts,
            unit_ids=left.saved['unit_ids'], source_unit_totals=left.saved['unit_counts'],
            omission_unit_totals=right.saved['unit_counts'])
        origins = input_origins or dict(source=str(source), omission=str(omission),
                                       source_analysis=str(source_analysis), source_rng=str(source_rng))
        report.update(status='MEASURED_CA3_CUE_OMISSION_CONTRAST', units=left.native['core_neurons'],
            duration_ms=DURATION_MS, source_events=len(source_events), omission_events=len(omission_events),
            pre_cue_events_equal=True, compared_pre_cue_events=prefix_events, pre_cue_interval_ms=[0,520],
            configured_initialization_identity_equal=True, complete_microstate_equality_independently_observed=False,
            backend_library_bytes_equal=True, fixed_source_fields=names, source_phase_edges_ms=list(PHASE_EDGES_MS),
            phase_rows=len(rows), difference_direction='source_minus_omission', input_origins=origins,
            whole_schedule_fields=rate_rows(names, sizes, source_counts, omission_counts,
                                             'whole_schedule', 0, DURATION_MS),
            input_sha256=dict(source=left.hashes, omission=right.hashes, **proofs),
            independent_units=dict(circuits=1, configured_initial_noise_blocks=1),
            later_phase_interpretation='Differences include all preceding omitted cues, not an isolated effect of each cue',
            raw_events='all AER records retained in referenced native archives; duplicates preserved',
            source_counter_and_current_qualification_reused=True, checkpoint_payload_rehashed=False,
            artifact_sha256={name:sha256_file(output/name) for name in ('counts.npz','phase_counts.csv')})
    except BaseException as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                      implementation_sha256=sha256_file(Path(__file__)))
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'omission', 'source-analysis', 'source-rng', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--input-origins', type=Path)
    args = parser.parse_args()
    origins = _json(args.input_origins) if args.input_origins else None
    print(json.dumps(analyze(args.source, args.omission, args.source_analysis, args.source_rng,
                             args.output, input_origins=origins), indent=2))


if __name__ == '__main__':
    main()

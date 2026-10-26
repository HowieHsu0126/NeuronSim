"""Verify fresh-process CA3 retrieval by exact event pairs and decoded counts.

File bytes and event ordering are reported separately from time/unit multisets.
This checks source execution and recording consistency, not pattern completion,
physical current units, or production. Explicit observed substep repeats also
compare recorded currents and native counters; they do not establish complete
microscopic state equality or numerical convergence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import zip_longest
import json
from pathlib import Path
import sys
import time
from zipfile import BadZipFile

import numpy as np

from .ca3_native_observation import iter_intervals, read_header
from .ca3_neuron_substeps import COUNTS, validate_report as validate_substeps
from .carlsim_event_order import compare_events
from .carlsim_prefix_replay import _independent_counts, _same_bytes
from .carlsim_records import SOURCE_REVISION, _sha, network_header, validate_coupled_identity

AUTHOR_REVISION = '1989764942516bf110dafc29ad08d815fd699f4c'
INPUT_NAME = 'network65p_assem275_divnorm.dat'
DURATION_MS = 1001
INPUT_SECONDS = 45.849998474121094
SOURCE_FIELDS = (
    'framework_revision', 'author_revision', 'binary_sha256', 'core_neurons',
    'populations', 'integration', 'delay_ms', 'source_rng', 'pattern_sha256',
    'pattern_integer_count', 'original_author_main_sha256', 'overlay_source_sha256',
    'source_file_byte_checks', 'retrieval_input_contract',
)
INPUT_FIELDS = (
    'normalized_input_sha256', 'raw65p_sha256', 'formation_report_sha256',
    'formation_replay_sha256', 'normalization_report_sha256', 'normalization_replay_sha256',
)


@dataclass(frozen=True)
class BoundRun:
    root: Path
    report: dict
    recording: dict
    header: dict
    hashes: dict[str, str]
    manifest_files: dict[str, str]
    files: dict
    started: datetime
    finished: datetime


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('object receipt required: '+str(path))
    return value


def _digest(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _clock(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError('native process timestamp required')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError('native process timestamp timezone required')
    return parsed


def _bind(root: Path, *, coupled_extension: bool = False,
          neuron_substeps: int | None = None, source_realization_config: Path | None = None) -> BoundRun:
    if neuron_substeps is not None and not coupled_extension:
        raise ValueError('CA3_REFINEMENT_REQUIRES_COUPLED_SOURCE_RETRIEVAL')
    hashes = {name:_sha(root/name) for name in (
        'native_report.json', 'observations/recording.json',
        'observations/population_counts.npz', 'manifest.json', 'build_report.json')}
    report = _read(root/'native_report.json')
    recording = _read(root/'observations/recording.json')
    build = _read(root/'build_report.json')
    if source_realization_config is not None or 'source_realization' in report or 'source_realization' in build:
        from .ca3_realization import bind_runtime

        if not coupled_extension or neuron_substeps is not None:
            raise ValueError('CA3_REALIZATION_COUPLED_SOURCE_REPLAY_REQUIRED')
        for item in (build, report):
            bind_runtime(root, item, source_realization_config)
        if (recording.get('source_realization') != report['source_realization']
                or recording.get('realization_circuit_binding') != report.get('realization_circuit_binding')):
            raise ValueError('CA3_REALIZATION_REPLAY_DECODER_IDENTITY_DIFFERS')
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    input_name = 'network65p_assem550_divnorm.dat' if coupled_extension else INPUT_NAME
    if coupled_extension:
        validate_coupled_identity(report, neuron_substeps=neuron_substeps)
        validate_coupled_identity(build, neuron_substeps=neuron_substeps)
        if (build['coupled_extension'] != report['coupled_extension']
                or build['coupled_extension_config'] != report['coupled_extension_config']
                or recording.get('coupled_extension') != report['coupled_extension']):
            raise ValueError('coupled retrieval identity differs across build/execution/decoder')
    if (report['status'] != family+'_RETRIEVAL_EXIT_ZERO' or report['exit_code'] != 0
            or report['native_run_executed'] is not True or report['retrieval_executed'] is not True
            or report['scope'] != ('coupled_extension_retrieval' if coupled_extension else 'native_source_retrieval')
            or report['expected_duration_ms'] != DURATION_MS
            or report['framework_revision'] != SOURCE_REVISION or report['author_revision'] != AUTHOR_REVISION
            or report['normalized_input_qualified'] is not True
            or report['retrieval_program_changed_only_in_input_path'] is not (not coupled_extension)):
        raise ValueError('completed pinned native retrieval required')
    contract = report['retrieval_input_contract']
    if (contract['effective_input_binding'] != input_name or contract['source_path_binding_only'] is not (not coupled_extension)
            or (coupled_extension and contract.get('fresh_coupled_checkpoint_required') is not True)
            or contract['source_carlsim_seed'] != 10 or contract['source_host_srand_seed'] != 3
            or contract['source_estdp_enabled'] is not True):
        raise ValueError('unchanged source retrieval input binding required')
    expected_build = ('CA3_REALIZATION_BUILD_PASSED_NATIVE_QUALIFICATION_PENDING'
                      if source_realization_config is not None else
                      'COUPLED_EXTENSION_BUILD_PASSED_NATIVE_EXECUTION_UNQUALIFIED' if coupled_extension
                      else 'NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING')
    if (build['status'] != expected_build
            or any(build[key] != report[key] for key in SOURCE_FIELDS)):
        raise ValueError('native retrieval differs from source build receipt')
    if not coupled_extension:
        validate_substeps(report, count=neuron_substeps)
        validate_substeps(build, count=neuron_substeps)
    if neuron_substeps is not None and any(recording.get(key) != report[key]
            for key in ('numerical_refinement', 'integration')):
        raise ValueError('refined retrieval decoder numerical declaration differs')
    for key in (*INPUT_FIELDS, 'original_author_main_sha256', 'overlay_source_sha256'):
        if not _digest(report[key]):
            raise ValueError('qualified input receipt hash required: '+key)
    if (type(report['native_pid']) is not int or report['native_pid'] <= 0
            or not isinstance(report['slurm_job_id'], str) or not report['slurm_job_id']):
        raise ValueError('native process identity required')
    started, finished = _clock(report['started_utc']), _clock(report['finished_utc'])
    if finished <= started:
        raise ValueError('completed native process interval required')
    if hashes['manifest.json'] != report['runtime_manifest_sha256']:
        raise ValueError('runtime manifest differs from native report')
    manifest = _read(root/'manifest.json')
    manifest_files = {}
    for row in manifest['files']:
        relative = Path(row['path'])
        name = str(relative)
        if (relative.is_absolute() or '..' in relative.parts or name in ('', '.')
                or name in manifest_files or not _digest(row['sha256'])):
            raise ValueError('runtime manifest path or digest invalid')
        if _sha(root/relative) != row['sha256']:
            raise ValueError('runtime input differs from manifest: '+name)
        manifest_files[name] = row['sha256']
    if not {'build_report.json', 'bin/ca3_retrieval', 'trainPat.csv'} <= manifest_files.keys():
        raise ValueError('runtime binary, build and pattern manifest coverage required')
    if (manifest_files['bin/ca3_retrieval'] != report['binary_sha256']
            or manifest_files['trainPat.csv'] != report['pattern_sha256']
            or _sha(root/'native_run/trainPat.csv') != report['pattern_sha256']):
        raise ValueError('native binary or pattern differs from source receipt')
    if source_realization_config is not None:
        if (manifest_files.get('realization_source.cpp') != report['overlay_source_sha256']
                or manifest_files.get('source_realization.yaml') != _sha(source_realization_config)):
            raise ValueError('CA3_REALIZATION_REPLAY_MANIFEST_COVERAGE_REQUIRED')
    checkpoint = root/'native_run'/input_name
    hashes['native_run/'+input_name] = _sha(checkpoint)
    if hashes['native_run/'+input_name] != report['normalized_input_sha256']:
        raise ValueError('normalized input differs from native receipt')
    header = network_header(checkpoint)
    if (header['simulation_seconds'] != INPUT_SECONDS or header['units'] != report['core_neurons']
            or {g.name:g.units for g in header['groups']} != report['populations']):
        raise ValueError('normalized input clock or population identity differs')
    if (recording['status'] != 'DECODED_'+family+'_RETRIEVAL_FILES'
            or recording['retrieval_executed'] is not True
            or recording['native_report_sha256'] != hashes['native_report.json']
            or recording['observation_sha256'] != hashes['observations/population_counts.npz']
            or recording['duration_ms'] != DURATION_MS or recording['time_resolution_ms'] != 1
            or recording['units'] != header['units']):
        raise ValueError('complete bound retrieval decoder receipt required')
    groups = [{**asdict(g), 'grid':list(g.grid)} for g in header['groups']]
    if recording['groups'] != groups:
        raise ValueError('decoder group identity differs')
    with np.load(root/'observations/population_counts.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['group_names'], [g.name for g in header['groups']]):
            raise ValueError('saved observation group identity differs')
        if any(not np.issubdtype(saved[key].dtype, np.integer) for key in (
                'unit_ids', 'unit_counts', 'population_counts_per_ms')):
            raise ValueError('decoded identities and counts must be integers')
    expected = {g.name:'native_run/results/spk_'+g.name+'.dat' for g in header['groups']}
    rows = recording['files']
    files = {row['group']:row for row in rows}
    if len(rows) != len(expected) or set(files) != set(expected):
        raise ValueError('decoder event file coverage differs')
    for name, relative in expected.items():
        if files[name]['relative_path'] != relative or _sha(root/relative) != files[name]['sha256']:
            raise ValueError('native spike file differs from decoder receipt: '+name)
        if type(files[name]['records']) is not int or files[name]['records'] < 0:
            raise ValueError('event file denominator invalid: '+name)
    if neuron_substeps is not None:
        relative = 'native_observation.bin'
        trace = root/relative
        if (manifest.get('native_observation_enabled') is not True
                or report.get('native_observation_stream_validated') is not True
                or report.get('observed_native_duration_ms') != DURATION_MS
                or report.get('host_request_runtime_current_equal') is not True
                or not isinstance(report['native_observation_overlay'], dict)
                or report['native_observation_overlay'].get('native_execution_verified') is not True
                or not _digest(report.get('native_observation_sha256'))
                or type(report.get('native_observation_bytes')) is not int
                or report['native_observation_bytes'] <= 0):
            raise ValueError('completed bound native observation required for refined repeat')
        hashes[relative] = _sha(trace)
        if (hashes[relative] != report['native_observation_sha256']
                or trace.stat().st_size != report['native_observation_bytes']):
            raise ValueError('native observation differs from native receipt')
        units, observed_groups = read_header(trace)
        if (units != header['units'] or
                [(g.group_id, g.name, g.start, g.stop) for g in observed_groups] !=
                [(g.group_id, g.name, g.start, g.stop) for g in header['groups']]):
            raise ValueError('native observation group identity differs from normalized input')
    return BoundRun(root, report, recording, header, hashes, manifest_files, files, started, finished)


def _unchanged(run: BoundRun) -> None:
    # The large normalized input was already hashed. Rehashing it here detects
    # changes while event comparison runs, including changes behind its symlink.
    for name, digest in {**run.hashes, **run.manifest_files}.items():
        if _sha(run.root/name) != digest:
            raise ValueError('bound retrieval input or receipt changed during replay: '+name)
    for row in run.files.values():
        if _sha(run.root/row['relative_path']) != row['sha256']:
            raise ValueError('native spike file changed during replay: '+row['group'])


def _compare_native_observations(left: BoundRun, right: BoundRun) -> dict:
    """Compare every recorded input/counter interval, exhausting both footers."""
    paths = [run.root/'native_observation.bin' for run in (left, right)]
    headers = [read_header(path) for path in paths]
    if headers[0] != headers[1]:
        raise ValueError('native observation runtime group mapping changed')
    units = headers[0][0]
    intervals = 0
    streams = [iter_intervals(path, expected_stop_ms=DURATION_MS) for path in paths]
    try:
        for a, b in zip_longest(*streams):
            if a is None or b is None:
                raise ValueError('native observation interval coverage changed')
            expected = (intervals, intervals+1)
            if ((a.start_ms, a.stop_ms) != expected or
                    (b.start_ms, b.stop_ms) != expected):
                raise ValueError('native observation interval clock changed')
            for field in ('requested_current', 'effective_current'):
                if not np.array_equal(getattr(a, field).view('<u4'), getattr(b, field).view('<u4')):
                    raise ValueError('native observation '+field+' bits changed at ms '+str(intervals))
            if not np.array_equal(a.native_counts, b.native_counts):
                raise ValueError('native observation counter array changed at ms '+str(intervals))
            intervals += 1
    finally:
        for stream in streams:
            stream.close()
    if intervals != DURATION_MS:
        raise ValueError('native observation terminal clock differs')
    return dict(intervals_compared=intervals, terminal_time_ms=intervals,
                units=units, values_compared_per_array=intervals*units,
                requested_current_bits_equal=True, effective_current_bits_equal=True,
                native_counter_arrays_equal=True, both_terminal_footers_verified=True,
                observation_sha256=[run.hashes['native_observation.bin'] for run in (left, right)],
                counter_equality_is_independent_aer_reconciliation=False,
                complete_microstate_equality_qualified=False,
                whole_simulator_convergence_qualified=False)


def compare(original: Path, repeat: Path, output: Path, *, coupled_extension: bool = False,
            neuron_substeps: int | None = None, source_realization_config: Path | None = None) -> dict:
    """Write failures as well as successful exact-multiset replay receipts."""
    original, repeat, output = map(Path, (original, repeat, output))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    result = dict(status='FAILED_'+family+'_RETRIEVAL_REPLAY', original=str(original), repeat=str(repeat),
        duration_ms=DURATION_MS, groups=[], independently_reconstructed_counts=[],
        implementation_sha256=_sha(Path(__file__)),
        helper_implementation_sha256={name:_sha(Path(__file__).with_name(name+'.py')) for name in
            ('carlsim_records', 'carlsim_prefix_replay', 'carlsim_event_order')},
        time_or_id_tolerance=0, multiplicity_preserved=True, independent_fresh_process_replay=False,
        native_internal_spike_counter_verified=False, actual_current_trace_verified=False,
        terminal_native_clock_independently_verified=False, full_state_restart_available=False,
        complete_formation_retrieval=False, pattern_completion_qualified=False,
        scientific_fidelity_qualified=False, full_scale_qualified=False, production_episodes=0,
        upstream_receipt_validation='matching execution-bound hashes; upstream receipt files are not replayed here')
    try:
        if original.resolve() == repeat.resolve() or (original/'native_run').resolve() == (repeat/'native_run').resolve():
            raise ValueError('distinct fresh retrieval runtime required')
        binding_options = dict(coupled_extension=coupled_extension, neuron_substeps=neuron_substeps)
        if source_realization_config is not None:
            binding_options['source_realization_config'] = source_realization_config
        left, right = (_bind(root, **binding_options) for root in (original, repeat))
        if coupled_extension:
            for key in ('coupled_extension', 'coupled_extension_config'):
                if left.report[key] != right.report[key]:
                    raise ValueError('coupled retrieval source changed: '+key)
            result['coupled_extension'] = left.report['coupled_extension']
        if source_realization_config is not None:
            result['source_realization'] = left.report['source_realization']
            result['realization_circuit_binding'] = left.report['realization_circuit_binding']
        result.update(original_report_sha256=left.hashes['native_report.json'],
            repeat_report_sha256=right.hashes['native_report.json'],
            original_recording_sha256=left.hashes['observations/recording.json'],
            repeat_recording_sha256=right.hashes['observations/recording.json'], units=left.header['units'])
        for key in (*SOURCE_FIELDS, *INPUT_FIELDS):
            if left.report[key] != right.report[key]:
                raise ValueError('retrieval source or qualified input changed: '+key)
        if left.manifest_files != right.manifest_files:
            raise ValueError('retrieval runtime file set changed')
        for key in ('units', 'groups', 'simulation_seconds', 'payload_offset'):
            if left.header[key] != right.header[key]:
                raise ValueError('retrieval input identity changed: '+key)
        identities = [{key:run.report[key] for key in
            ('slurm_job_id', 'native_pid', 'started_utc', 'finished_utc')} for run in (left, right)]
        if identities[0] == identities[1] or right.started < left.finished:
            raise ValueError('independent subsequent retrieval process required')
        explicit_original = right.report.get('independent_repeat_of_report_sha256')
        if explicit_original is not None and explicit_original != left.hashes['native_report.json']:
            raise ValueError('repeat identifies a different original retrieval')
        result.update(native_processes=identities, independent_fresh_process_replay=True,
            normalized_input_sha256=left.report['normalized_input_sha256'], actual_normalized_input_verified=True,
            qualified_input_receipt_hashes={key:left.report[key] for key in INPUT_FIELDS},
            binary_sha256=left.report['binary_sha256'])
        for run in (left, right):
            counts = _independent_counts(run.root, run.header, run.recording)
            result['independently_reconstructed_counts'].append(dict(duration_ms=DURATION_MS, **counts))
        for group in left.header['groups']:
            relative = left.files[group.name]['relative_path']
            a, b = original/relative, repeat/relative
            equal, compared, hashes = _same_bytes(a, b)
            if hashes != [run.files[group.name]['sha256'] for run in (left, right)]:
                raise ValueError('native spike file changed during comparison: '+group.name)
            comparison = compare_events(a, b, units=group.units, duration_ms=DURATION_MS)
            if (comparison['original_rows'] != left.files[group.name]['records']
                    or comparison['repeated_rows'] != right.files[group.name]['records']):
                raise ValueError('event file denominator differs: '+group.name)
            result['groups'].append(dict(group=group.name, relative_path=relative,
                file_bytes_equal=equal, bytes_compared=compared,
                original_sha256=hashes[0], repeat_sha256=hashes[1], **comparison))
        if neuron_substeps is not None:
            result.update(numerical_refinement=left.report['numerical_refinement'],
                          integration=left.report['integration'], neuron_substeps=neuron_substeps)
            result['helper_implementation_sha256'].update({
                name:_sha(Path(__file__).with_name(name+'.py')) for name in
                ('ca3_neuron_substeps', 'ca3_native_observation')})
            result['native_observation_replay'] = _compare_native_observations(left, right)
            result.update(actual_current_trace_verified=True,
                          native_counter_replay_equal=True,
                          terminal_native_clock_independently_verified=True)
        for run in (left, right):
            _unchanged(run)
        exact = all(g['header_bytes_equal'] and g['event_multiset_equal'] for g in result['groups'])
        strict = all(g['file_bytes_equal'] for g in result['groups'])
        ordered = all(g['event_order_equal'] for g in result['groups'])
        timestamps = all(g['timestamp_sequence_equal'] for g in result['groups'])
        result.update(events_compared=sum(g['original_rows'] for g in result['groups']),
            event_multisets_equal=exact, event_file_bytes_equal=strict, event_order_equal=ordered,
            timestamp_sequence_equal=timestamps,
            strict_file_replay_status=('PASS_' if strict else 'MISMATCH_')+family+'_RETRIEVAL_FILE_BYTES',
            same_time_file_order_differences=exact and timestamps and not ordered,
            timestamp_order_differences=exact and not timestamps,
            source_event_file_order_preserved=True,
            file_order_interpretation='descriptive difference only; does not establish a change in source dynamics')
        if not exact:
            result['status'] = 'MISMATCH_'+family+'_RETRIEVAL_REPLAY'
            raise ValueError('retrieval exact event multiset or event header differs')
        result['status'] = 'PASS_'+family+'_RETRIEVAL_REPLAY_AND_COUNTS'
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        result.update(error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter()-started)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
        raise
    result['elapsed_seconds'] = time.perf_counter()-started
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'repeat', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--coupled-extension', action='store_true')
    parser.add_argument('--neuron-substeps', type=int, choices=COUNTS)
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.original, args.repeat, args.output,
                         coupled_extension=args.coupled_extension, neuron_substeps=args.neuron_substeps,
                         source_realization_config=args.source_realization_config)
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

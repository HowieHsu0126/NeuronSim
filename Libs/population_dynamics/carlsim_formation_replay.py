"""Verify a completed70s formation against the archived4.45s source prefix.

The full event stream is read in chunks. Only the fixed prefix is materialized
for exact time/ID/multiplicity comparison; its file-order differences remain
visible. Full-duration counts are independently reconstructed from raw records.
This is an execution/recording check, not full-dynamics or retrieval fidelity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from .carlsim_event_order import compare_events
from .carlsim_prefix_replay import _independent_counts, _same_bytes
from .carlsim_records import (SOURCE_REVISION, SPIKE_HEADER, _sha, network_header,
                              spike_blocks, validate_coupled_identity)

PREFIX_MS = 4450
FORMATION_MS = 70000
SOURCE_FIELDS = ('framework_revision', 'author_revision', 'core_neurons', 'populations',
                 'integration', 'delay_ms', 'source_rng', 'pattern_sha256', 'pattern_integer_count')


@dataclass(frozen=True)
class BoundRun:
    root: Path
    report: dict
    recording: dict
    header: dict
    report_sha256: str
    recording_sha256: str
    files: dict


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _bind(root: Path, *, formation: bool, coupled_extension: bool = False,
          source_realization_config: Path | None = None) -> BoundRun:
    report_path = root/'native_report.json'
    recording_path = root/'observations/recording.json'
    report_sha, recording_sha = _sha(report_path), _sha(recording_path)
    report, recording = _read(report_path), _read(recording_path)
    if source_realization_config is not None or 'source_realization' in report:
        from .ca3_realization import bind_runtime

        if not coupled_extension:
            raise ValueError('CA3_REALIZATION_COUPLED_FORMATION_REPLAY_REQUIRED')
        bind_runtime(root, report, source_realization_config)
        if (recording.get('source_realization') != report['source_realization']
                or recording.get('realization_circuit_binding') != report.get('realization_circuit_binding')):
            raise ValueError('CA3_FORMATION_REALIZATION_DECODER_BINDING_DIFFERS')
    duration = FORMATION_MS if formation else PREFIX_MS
    kind = 'FORMATION' if formation else 'PREFIX'
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    scope = ('coupled_extension_' if coupled_extension else 'native_source_') + kind.lower()
    if coupled_extension:
        validate_coupled_identity(report)
    if (report['status'] != f'{family}_{kind}_EXIT_ZERO' or report['exit_code'] != 0
            or report['native_run_executed'] is not True
            or report['framework_revision'] != SOURCE_REVISION
            or report['scope'] != scope
            or report['expected_duration_ms' if formation else 'expected_prefix_ms'] != duration):
        raise ValueError('completed bound native execution required')
    if formation:
        original_main = report.get('original_author_main_sha256')
        executed_main = report.get('overlay_source_sha256')
        unchanged = report.get('complete_author_formation_program_unchanged')
        source_matches = (unchanged is False if coupled_extension else
                          unchanged is True and original_main == executed_main)
        if (report.get('complete_formation_executed') is not True
                or not original_main or not executed_main or not source_matches):
            raise ValueError('complete coupled extension formation required' if coupled_extension
                             else 'complete unchanged author formation required')
    if (recording['status'] != f'DECODED_{family}_{kind}_FILES'
            or recording['native_report_sha256'] != report_sha
            or recording['observation_sha256'] != _sha(root/'observations/population_counts.npz')
            or recording['duration_ms'] != duration or recording['time_resolution_ms'] != 1
            or (formation and recording.get('complete_formation_executed') is not True)
            or (coupled_extension and recording.get('coupled_extension') != report['coupled_extension'])):
        raise ValueError('bound decoder receipt required')
    checkpoint = root/'native_run'/('network100p.dat' if formation else 'network5p.dat')
    header = network_header(checkpoint)
    if (header['simulation_seconds'] != float(np.float32(duration/1000))
            or header['units'] != report['core_neurons'] or recording['units'] != header['units']
            or {g.name:g.units for g in header['groups']} != report['populations']):
        raise ValueError('native duration or population identity differs')
    groups = [{**asdict(g), 'grid':list(g.grid)} for g in header['groups']]
    if recording['groups'] != groups:
        raise ValueError('decoder group identity differs')
    with np.load(root/'observations/population_counts.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['group_names'], [g.name for g in header['groups']]):
            raise ValueError('saved observation group identity differs')
    # Prefix checkpoint hashing is shared with the byte comparison below.
    if formation and _sha(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('terminal checkpoint differs from native report')
    expected = {g.name:'native_run/results/spk_'+g.name+'.dat' for g in header['groups']}
    rows = recording['files']
    files = {row['group']:row for row in rows}
    if len(rows) != len(expected) or set(files) != set(expected):
        raise ValueError('decoder event file coverage differs')
    for name, relative in expected.items():
        if files[name]['relative_path'] != relative or _sha(root/relative) != files[name]['sha256']:
            raise ValueError('native spike file differs from decoder receipt: '+name)
    return BoundRun(root, report, recording, header, report_sha, recording_sha, files)


def _extract_prefix(source: Path, target: Path, *, units: int) -> tuple[int, int]:
    """Filter every block: CARLsim event-file timestamps need not be sorted."""
    total = prefix = 0
    with source.open('rb') as stream, target.open('xb') as out:
        out.write(stream.read(SPIKE_HEADER.size))
        for rows in spike_blocks(source, units=units, stop_ms=FORMATION_MS):
            selected = rows[rows[:, 0] < PREFIX_MS]
            out.write(selected.tobytes())
            total += len(rows)
            prefix += len(selected)
    return total, prefix


def _unchanged_receipts(run: BoundRun) -> None:
    if (_sha(run.root/'native_report.json') != run.report_sha256
            or _sha(run.root/'observations/recording.json') != run.recording_sha256
            or _sha(run.root/'observations/population_counts.npz') != run.recording['observation_sha256']):
        raise ValueError('source or decoder receipt changed during replay')


def compare(original: Path, formation: Path, output: Path, *, workdir: Path | None = None,
            coupled_extension: bool = False, source_realization_config: Path | None = None) -> dict:
    """Write a retained failure receipt and raise on any substantive mismatch."""
    original, formation, output = map(Path, (original, formation, output))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    family = 'COUPLED_EXTENSION' if coupled_extension else 'NATIVE'
    result = dict(status=f'FAILED_{family}_FORMATION_REPLAY', original=str(original), formation=str(formation),
        prefix_interval_ms=[0, PREFIX_MS], formation_duration_ms=FORMATION_MS,
        checks=[], groups=[], independently_reconstructed_counts=[],
        implementation_sha256=_sha(Path(__file__)),
        helper_implementation_sha256={name:_sha(Path(__file__).with_name(name+'.py')) for name in
            ('carlsim_records','carlsim_prefix_replay','carlsim_event_order')},
        time_or_id_tolerance=0, multiplicity_preserved=True,
        complete_formation_executed=False, independent_full_formation_execution_replay=False,
        complete_formation_retrieval=False, full_state_restart_available=False,
        native_internal_spike_counter_verified=False, synaptic_payload_semantics_validated=False,
        scientific_fidelity_qualified=False, full_scale_qualified=False, production_episodes=0)
    try:
        left = _bind(original, formation=False, coupled_extension=coupled_extension, source_realization_config=source_realization_config)
        right = _bind(formation, formation=True, coupled_extension=coupled_extension, source_realization_config=source_realization_config)
        result.update(original_report_sha256=left.report_sha256, formation_report_sha256=right.report_sha256,
            original_recording_sha256=left.recording_sha256, formation_recording_sha256=right.recording_sha256,
            units=left.header['units'], complete_formation_executed=True,
            executable_hashes={'prefix':left.report.get('reused_binary_sha256',left.report.get('binary_sha256')),
                               'formation':right.report.get('binary_sha256')})
        if left.report.get('source_realization') != right.report.get('source_realization'):
            raise ValueError('CA3_FORMATION_PREFIX_REALIZATION_DIFFERS')
        if source_realization_config is not None:
            result['source_realization'] = right.report['source_realization']
            result['realization_circuit_binding'] = right.report['realization_circuit_binding']
        identity_fields = SOURCE_FIELDS + (('coupled_extension',) if coupled_extension else ())
        for key in identity_fields:
            if left.report[key] != right.report[key]:
                label = 'coupled extension identity changed: ' if coupled_extension else 'native source changed: '
                raise ValueError(label + key)
        if left.report.get('conductance_memory_overlay') != right.report.get('conductance_memory_overlay'):
            raise ValueError('formation and independent prefix conductance memory policies differ')
        if coupled_extension:
            result['coupled_extension'] = left.report['coupled_extension']
        for key in ('units','groups','payload_offset'):
            if left.header[key] != right.header[key]:
                raise ValueError('full formation unit/group identity differs: '+key)
        checkpoint = formation/'native_run/network5p.dat'
        prefix_header = network_header(checkpoint)
        for key in ('units','groups','simulation_seconds','payload_offset'):
            if prefix_header[key] != left.header[key]:
                raise ValueError('formation prefix checkpoint identity differs: '+key)
        equal, compared, hashes = _same_bytes(original/'native_run/network5p.dat', checkpoint,
                                             ignore_execution_time=True)
        if hashes[0] != left.report['checkpoint_sha256']:
            raise ValueError('original checkpoint differs from native report')
        result['checks'].append(dict(name='network5p.dat', equal=equal, bytes_compared=compared,
            ignored_byte_range=[12,16], original_sha256=hashes[0], formation_sha256=hashes[1]))
        for run in (left, right):
            counts = _independent_counts(run.root, run.header, run.recording)
            result['independently_reconstructed_counts'].append(dict(duration_ms=run.recording['duration_ms'], **counts))
        # Temporary files contain only the fixed4.45s prefix, never the70s stream.
        # Use --workdir or TMPDIR to place them on compute-node-local storage.
        with tempfile.TemporaryDirectory(prefix='carlsim-formation-prefix-', dir=workdir) as temporary:
            for group in left.header['groups']:
                relative = left.files[group.name]['relative_path']
                filtered = Path(temporary)/(str(group.group_id)+'.dat')
                total, prefix = _extract_prefix(formation/relative, filtered, units=group.units)
                if total != right.files[group.name]['records']:
                    raise ValueError('full event file denominator differs: '+group.name)
                comparison = compare_events(original/relative, filtered, units=group.units, duration_ms=PREFIX_MS)
                if comparison['original_rows'] != left.files[group.name]['records'] or comparison['repeated_rows'] != prefix:
                    raise ValueError('prefix event file denominator differs: '+group.name)
                result['groups'].append(dict(group=group.name, full_records=total,
                    future_records=total-prefix, original_file_sha256=left.files[group.name]['sha256'],
                    full_file_sha256=right.files[group.name]['sha256'], **comparison))
        for run in (left, right):
            _unchanged_receipts(run)
        exact = all(g['event_multiset_equal'] and g['header_bytes_equal'] for g in result['groups'])
        ordered = all(g['event_order_equal'] for g in result['groups'])
        same_times = all(g['timestamp_sequence_equal'] for g in result['groups'])
        result.update(prefix_events_compared=sum(g['original_rows'] for g in result['groups']),
            prefix_event_multisets_equal=exact, prefix_event_order_equal=ordered,
            prefix_timestamp_sequence_equal=same_times,
            same_time_file_order_differences=exact and same_times and not ordered,
            timestamp_order_differences=exact and not same_times)
        if not equal or not exact:
            result['status'] = f'MISMATCH_{family}_FORMATION_PREFIX_REPLAY'
            raise ValueError('formation prefix checkpoint or exact event multiset differs')
        result['status'] = f'PASS_{family}_FORMATION_PREFIX_REPLAY_AND_COUNTS'
    except (OSError, ValueError, KeyError, TypeError) as error:
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
    for name in ('original','formation','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workdir', type=Path, help='Existing compute-node-local directory for temporary prefix files')
    parser.add_argument('--coupled-extension', action='store_true',
                        help='Compare two declared 2x coupled runs; never mix in the native reference')
    parser.add_argument('--source-realization-config', type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.original, args.formation, args.output, workdir=args.workdir,
                         coupled_extension=args.coupled_extension, source_realization_config=args.source_realization_config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Explain V1 native time representation without changing source equality gates.

The oracle follows pinned NEST v3.10 Time::fromstamp, spike_generator offset
storage, and memory-recorder decoding. Its epsilon branch is NEST's own encoding
rule, not an acceptance tolerance. Archived events are reused without replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_nest_input_replay import NEST_VERSION, bound_json, compare_events

NEST_REVISION = 'acca9704da248750219a027db99fec6cd1f9052a'
SOURCE_HASHES = {
    'models/spike_generator.cpp': 'cbca94d141b456a905068094017a399f5be47858233066c5cc8708cd22fd85e1',
    'nestkernel/nest_time.h': 'd2cd0b042334cb7e1161f6330a98d85632c64672a415a05fc5f80ca2bd6b9a7b',
    'nestkernel/nest_time.cpp': 'a834388c5ad48153540a64f2126e956f844cde5e3d1f65bf772095fe0c0968ea',
    'nestkernel/recording_backend_memory.cpp': '4a61d8666ed6259cc59896c9b52521e25500f80d6e7aa7cad7c1d4d76002c2f0',
}
V1_DURATION_MS = 3000.0
V1_RESOLUTION_MS = 0.25


def verify_sources(directory: Path) -> dict:
    source = json.loads((directory / 'source.json').read_text())
    if source['repository'] != 'nest/nest-simulator' or source['revision'] != NEST_REVISION:
        raise ValueError('V1_NEST_TIME_SOURCE_REVISION_CHANGED')
    rows = {row['path']: row for row in source['files']}
    for relative, expected in SOURCE_HASHES.items():
        path = directory / relative
        row = rows[relative]
        if (row['sha256'] != expected or path.stat().st_size != row['bytes']
                or sha256_file(path) != expected):
            raise ValueError('V1_NEST_TIME_SOURCE_CHANGED:' + relative)
    return source


def clock_tics_per_step(tics_per_ms: float, resolution_ms: float = V1_RESOLUTION_MS) -> int:
    """Require an exactly representable positive NEST execution step."""
    if (not np.isfinite(tics_per_ms) or tics_per_ms <= 0
            or not np.isfinite(resolution_ms) or resolution_ms <= 0):
        raise ValueError('V1_NEST_TIME_DOMAIN_INVALID')
    ticks_per_step = tics_per_ms * resolution_ms
    if (not np.isfinite(ticks_per_step) or ticks_per_step != int(ticks_per_step)
            or ticks_per_step < 1 or ticks_per_step >= np.iinfo(np.int64).max):
        raise ValueError('V1_NEST_TIME_RESOLUTION_NOT_REPRESENTABLE')
    return int(ticks_per_step)


def source_refractory_steps(durations_ms: np.ndarray, tics_per_ms: float,
                            resolution_ms: float = V1_RESOLUTION_MS) -> np.ndarray:
    """Derive Time::ms(t_ref).get_steps(); these are not private-state readbacks.

    NEST first rounds nonnegative milliseconds to integer tics, then rounds the
    tic count up to an execution step. This differs from rounding t_ref/dt.
    """
    durations = np.asarray(durations_ms)
    if (durations.ndim != 1 or durations.dtype.kind not in 'iuf'
            or not np.isfinite(durations).all() or np.any(durations < 0)):
        raise ValueError('V1_NEST_REFRACTORY_DOMAIN_INVALID')
    step_tics = clock_tics_per_step(tics_per_ms, resolution_ms)
    scaled = durations.astype(np.float64) * tics_per_ms + .5
    if np.any(scaled >= np.iinfo(np.int64).max - step_tics):
        raise ValueError('V1_NEST_TIME_INTEGER_RANGE_EXCEEDED')
    ticks = np.trunc(scaled).astype(np.int64)
    return np.trunc((ticks + step_tics - 1) * (1.0 / step_tics)).astype(np.int64)


def encode_precise_times(times_ms: np.ndarray, tics_per_ms: float,
                         resolution_ms: float = V1_RESOLUTION_MS) -> dict[str, np.ndarray]:
    """Translate the released C++ operations, retaining the source-clock default."""
    times = np.asarray(times_ms)
    if (times.ndim != 1 or times.dtype.kind not in 'iuf' or not np.isfinite(times).all()
            or np.any(times <= 0) or np.any(times > V1_DURATION_MS)
            or not np.isfinite(tics_per_ms) or tics_per_ms <= 0):
        raise ValueError('V1_NEST_TIME_DOMAIN_INVALID')
    times = times.astype(np.float64, copy=True)
    ticks_per_step = clock_tics_per_step(tics_per_ms, resolution_ms)
    # Restrict integer conversion to the documented clock's finite range.
    if tics_per_ms * V1_DURATION_MS >= np.iinfo(np.int64).max:
        raise ValueError('V1_NEST_TIME_INTEGER_RANGE_EXCEEDED')
    ticks = np.trunc(times * tics_per_ms).astype(np.int64)
    ticks -= ticks % ticks_per_step
    lower_ms = ticks * (1.0 / ticks_per_step) * resolution_ms
    ticks += (lower_ms < times) * ticks_per_step
    stamps = (1.0 / tics_per_ms) * ticks
    offsets = stamps - times
    # This is the source encoding branch, never a comparison tolerance.
    tiny_offset = ((np.abs(offsets) < np.finfo(np.float64).eps * np.abs(stamps + times) * 2.0)
                   | (np.abs(offsets) < np.finfo(np.float64).tiny))
    offsets[tiny_offset] = 0.0
    if np.any(offsets < 0):
        raise ValueError('V1_NEST_TIME_NEGATIVE_SOURCE_OFFSET')
    return {'steps': ticks // ticks_per_step, 'offsets_ms': offsets,
            'decoded_times_ms': stamps - offsets}


def native_witness(nest, requested_delays_ms: list[float]) -> dict:
    """Observe packed time fields and actual GLIF connection delay readbacks."""
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': V1_RESOLUTION_MS, 'local_num_threads': 1})
    tics = float(nest.GetKernelStatus('tics_per_ms'))
    raw = np.array([.000021, .052912, .125, np.nextafter(.25, 0), .25,
                    np.nextafter(.25, np.inf), 1.1, 1.1, np.nextafter(2., 0)])
    encoded = encode_precise_times(raw, tics)
    generator = nest.Create('spike_generator', len(raw), params={'precise_times': True})
    native_ids = np.asarray(generator.tolist(), dtype=np.int64)
    stored = []
    for source_id, value in zip(native_ids, raw):
        node = nest.NodeCollection([int(source_id)])
        nest.SetStatus(node, {'spike_times': [float(value)]})
        stored.append(nest.GetStatus(node, 'spike_times')[0][0])
    stored = np.asarray(stored)
    packed = nest.Create('spike_recorder', params={'time_in_steps': True})
    decoded = nest.Create('spike_recorder')
    for recorder in (packed, decoded):
        nest.Connect(generator, recorder, syn_spec={'delay': V1_RESOLUTION_MS})
    delay_rows = []
    for delay in requested_delays_ms:
        cell = nest.Create('glif_psc')
        delay_source = nest.NodeCollection([int(native_ids[0])])
        nest.Connect(delay_source, cell, syn_spec={'delay': delay, 'weight': 1.0, 'receptor_type': 1})
        actual = float(np.asarray(nest.GetConnections(source=delay_source, target=cell).get('delay')).item())
        expected = float(np.floor(delay / V1_RESOLUTION_MS + .5) * V1_RESOLUTION_MS)
        delay_rows.append({'requested_ms': delay, 'native_ms': actual,
                           'released_source_rounding_ms': expected, 'matches_source_rule': actual == expected})
    nest.Simulate(4.0)
    events = packed.get('events')
    steps, offsets = np.asarray(events['times']), np.asarray(events['offsets'])
    order = np.argsort(events['senders'])
    decoded_events = decoded.get('events')
    decoded_order = np.argsort(decoded_events['senders'])
    native_times = np.asarray(decoded_events['times'])[decoded_order]
    checks = {
        'source_getter_matches_encoding_exactly': np.array_equal(stored, encoded['decoded_times_ms']),
        'recorder_steps_match_encoding_exactly': np.array_equal(steps[order], encoded['steps']),
        'recorder_offsets_match_encoding_exactly': np.array_equal(offsets[order], encoded['offsets_ms']),
        'recorder_ms_matches_encoding_exactly': np.array_equal(native_times, encoded['decoded_times_ms']),
        'recorder_senders_match': (np.array_equal(np.asarray(events['senders'])[order], native_ids)
                                   and np.array_equal(np.asarray(decoded_events['senders'])[decoded_order], native_ids)),
        'source_delay_readbacks_match_released_rule': all(row['matches_source_rule'] for row in delay_rows),
    }
    return {'status': 'PASS' if all(checks.values()) else 'FAILED',
            'checks': {key: bool(value) for key, value in checks.items()},
            'tics_per_ms': tics, 'resolution_ms': V1_RESOLUTION_MS,
            'source_times_ms': raw.tolist(), 'stored_times_ms': stored.tolist(),
            'native_source_ids': native_ids.tolist(),
            'recorder_steps': steps[order].tolist(), 'recorder_offsets_ms': offsets[order].tolist(),
            'recorder_times_ms': native_times.tolist(), 'delay_readbacks': delay_rows,
            'scope': 'Nine input generators, two recorders and two default GLIF cells; no source cortical network.'}


def diagnose(replay_archive: Path, input_inspection_archive: Path, nest_source_archive: Path, output: Path) -> dict:
    import nest  # Optional native dependency; pure oracle tests do not import NEST.

    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'FAILED_TIME_REPRESENTATION_DIAGNOSIS',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'implementation_sha256': sha256_file(Path(__file__)),
              'raw_source_times_modified': False, 'strict_source_comparator_changed': False,
              'full_input_replay_repeated': False, 'full_cortical_network_executed': False,
              'source_scientific_fidelity_qualified': False, 'production_episodes': 0}
    try:
        if nest.__version__ != NEST_VERSION:
            raise ValueError('V1_NEST_TIME_NATIVE_VERSION_CHANGED')
        sources = verify_sources(nest_source_archive)
        original, delivery = bound_json(replay_archive, 'native_replay/report.json')
        if (original['native_input_simulation_executed'] is not True
                or original['source_config_clock'] != {'duration': V1_DURATION_MS, 'dt': V1_RESOLUTION_MS}
                or original['nest_version'] != NEST_VERSION):
            raise ValueError('V1_NEST_TIME_COMPLETE_NATIVE_REPLAY_REQUIRED')
        inspection, _ = bound_json(input_inspection_archive, 'input_identity.json')
        if (inspection['input_member_sha256'] != original['source_input_member_sha256']
                or sha256_file(input_inspection_archive / 'input_identity.json') != original['input_inspection_sha256']):
            raise ValueError('V1_NEST_TIME_INPUT_IDENTITY_CHANGED')
        data = {}
        for name in ['source_inputs.npz', 'events_at_source_stop.npz']:
            path = replay_archive / 'native_replay' / name
            if sha256_file(path) != original['artifacts'][name]:
                raise ValueError('V1_NEST_TIME_REPLAY_ARTIFACT_CHANGED:' + name)
            with np.load(path, allow_pickle=False) as archive:
                data[name] = {key: archive[key] for key in archive.files}
        plan = data['source_inputs.npz']
        events = data['events_at_source_stop.npz']
        raw_comparison = compare_events(plan, plan['native_id'], events)
        if raw_comparison != original['at_source_stop']:
            raise ValueError('V1_NEST_TIME_ORIGINAL_COMPARISON_CHANGED')
        delays = []
        for population in inspection['populations']:
            if population['delay_min'] != population['delay_max']:
                raise ValueError('V1_NEST_TIME_NONUNIFORM_INPUT_DELAY_REQUIRES_FULL_DOMAIN')
            delays.append(float(population['delay_min']))
        witness = native_witness(nest, delays)
        encoded = encode_precise_times(plan['source_event_time_ms'], witness['tics_per_ms'])
        encoded_plan = dict(plan, source_event_time_ms=encoded['decoded_times_ms'])
        encoded_comparison = compare_events(encoded_plan, plan['native_id'], events)
        report.update(nest_version=nest.__version__, nest_release_revision=sources['revision'],
                      released_source_files=sources['files'], native_replay_job=delivery['job_id'],
                      native_replay_archive=str(replay_archive),
                      original_replay_status=original['status'], strict_source_comparison=raw_comparison,
                      source_encoding_comparison=encoded_comparison, native_witness=witness,
                      interpretation='Exact source-derived native representation is a diagnosis; raw float equality and scientific qualification are not promoted.')
        if witness['status'] == 'PASS' and encoded_comparison['all_event_pairs_exact']:
            report['status'] = ('NATIVE_TIME_REPRESENTATION_EXPLAINED_RAW_EXACTNESS_UNSATISFIED'
                                if not raw_comparison['all_event_pairs_exact'] else 'NATIVE_TIME_REPRESENTATION_AND_RAW_EXACT')
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / 'report.json', report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['replay-archive', 'input-inspection-archive', 'nest-source-archive', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    report = diagnose(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2))
    return 0 if report['status'].startswith('NATIVE_TIME_REPRESENTATION_') else 2


if __name__ == '__main__':
    raise SystemExit(main())

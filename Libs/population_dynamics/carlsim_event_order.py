"""Diagnose event ordering separately from the immutable strict native replay.

At CARLsim c4955998b950b9c561b3373a71b611aedc705be9, GPU firing-table
positions use atomic increments (snn_gpu_module.cu:558-574), and the monitor
writes table order (snn_manager.cpp:8482-8508). There is no unit-ID sort.
This diagnostic compares exact time/ID multisets, including multiplicity;
it neither changes source files nor promotes a byte mismatch to a strict pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .carlsim_records import SPIKE_HEADER, _sha, network_header, spike_blocks


def compare_events(left: Path, right: Path, *, units: int, duration_ms: int) -> dict:
    """Compare bounded prefix files using the validated reader and exact sorting."""
    arrays = []
    headers = []
    for path in (left, right):
        with path.open('rb') as stream:
            headers.append(stream.read(SPIKE_HEADER.size))
        blocks = list(spike_blocks(path, units=units, stop_ms=duration_ms))
        arrays.append(np.concatenate(blocks) if blocks else np.empty((0, 2), dtype='<i4'))
    a, b = arrays
    same_shape = a.shape == b.shape
    same_order = bool(np.array_equal(a, b))
    timestamp_equal = bool(np.array_equal(a[:, 0], b[:, 0]))
    ordered = [x[np.lexsort((x[:, 1], x[:, 0]))] for x in arrays]
    multiset_equal = bool(np.array_equal(*ordered))
    differing_rows = int(np.count_nonzero(np.any(a != b, axis=1))) if same_shape else None
    first = None
    if same_shape and not same_order:
        index = int(np.flatnonzero(np.any(a != b, axis=1))[0])
        first = {'row': index, 'original': a[index].tolist(), 'repeated': b[index].tolist()}
    return {
        'header_bytes_equal': headers[0] == headers[1],
        'original_rows': int(len(a)), 'repeated_rows': int(len(b)),
        'timestamp_sequence_equal': timestamp_equal, 'event_order_equal': same_order,
        'event_multiset_equal': multiset_equal, 'differing_file_rows': differing_rows,
        'first_difference': first,
    }


def diagnose(original: Path, repeated: Path, strict_replay: Path, output: Path) -> dict:
    original, repeated, strict_replay, output = map(Path, (original, repeated, strict_replay, output))
    if output.exists():
        raise FileExistsError(output)
    strict_hash = _sha(strict_replay)
    strict = json.loads(strict_replay.read_text())
    if strict['status'] not in ('MISMATCH_NATIVE_PREFIX_REPLAY', 'PASS_NATIVE_PREFIX_REPLAY'):
        raise ValueError('completed strict comparison required')
    for root, label in ((original, 'original'), (repeated, 'repeated')):
        if _sha(root/'native_report.json') != strict[label + '_report_sha256']:
            raise ValueError('native report differs from strict comparison')
    headers = [network_header(root/'native_run/network5p.dat') for root in (original, repeated)]
    if headers[0]['groups'] != headers[1]['groups'] or headers[0]['units'] != strict['units']:
        raise ValueError('native group identity differs from strict comparison')
    checks = {row['name']: row for row in strict['checks']}
    groups = []
    for group in headers[0]['groups']:
        row = checks[group.name]
        relative = 'native_run/results/spk_' + group.name + '.dat'
        if row['relative_path'] != relative:
            raise ValueError('native spike path differs from strict comparison')
        for root, label in ((original, 'original'), (repeated, 'repeated')):
            if _sha(root/relative) != row[label + '_sha256']:
                raise ValueError('native spike file differs from strict comparison')
        result = compare_events(original/relative, repeated/relative,
                                units=group.units, duration_ms=strict['duration_ms'])
        if not result['header_bytes_equal']:
            raise ValueError('native spike header differs')
        groups.append({'group': group.name, **result})
    exact = all(g['event_multiset_equal'] for g in groups)
    ordered = all(g['event_order_equal'] for g in groups)
    timestamps = all(g['timestamp_sequence_equal'] for g in groups)
    if not exact:
        status = 'EVENT_TIME_ID_MULTISET_MISMATCH'
    elif ordered:
        status = 'EVENT_FILES_IDENTICAL'
    elif timestamps:
        status = 'EVENT_TIME_ID_MULTISETS_EQUAL_FILE_ORDER_DIFFERS'
    else:
        status = 'EVENT_TIME_ID_MULTISETS_EQUAL_TIMESTAMP_ORDER_DIFFERS'
    result = {
        'status': status, 'strict_replay_status': strict['status'],
        'strict_replay_sha256': strict_hash, 'groups': groups,
        'records_compared': sum(g['original_rows'] for g in groups),
        'multiplicity_preserved': True, 'time_or_id_tolerance': 0,
        'original_strict_result_retained': True,
        'implementation_sha256': _sha(Path(__file__)),
        'scientific_fidelity_qualified': False, 'full_scale_qualified': False,
        'complete_formation_retrieval': False,
    }
    if _sha(strict_replay) != strict_hash:
        raise ValueError('strict replay changed during diagnosis')
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'repeated', 'strict-replay', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.original, args.repeated, args.strict_replay, args.output)
    print(json.dumps(result, indent=2))
    return 2 if result['status'] == 'EVENT_TIME_ID_MULTISET_MISMATCH' else 0


if __name__ == '__main__':
    raise SystemExit(main())

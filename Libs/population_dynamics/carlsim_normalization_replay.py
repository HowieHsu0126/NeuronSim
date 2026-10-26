"""Independently check every byte of the author's CA3 weight transformation.

This reader uses raw eight-word records, separately from the structured writer.
The selected weights retain source serialization order for the author's NumPy
float64 mean; all non-weight bytes must be unchanged, including signed zeros.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np

from .carlsim_records import network_header
from .provenance import sha256_file

AUTHOR_REVISION = '8c6c185994f8d54b7a3e027badcfaa7d15f232d1'


def verify(source: Path, normalized: Path, normalization_report: Path,
           output: Path, scratch: Path, *, chunk_records: int = 131072) -> dict:
    source, normalized, normalization_report, output, scratch = map(
        Path, (source, normalized, normalization_report, output, scratch))
    if output.exists():
        raise FileExistsError(output)
    if type(chunk_records) is not int or chunk_records <= 0:
        raise ValueError('positive integer chunk size required')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='FAILED_NORMALIZATION_REPLAY', all_record_bytes_verified=False,
                  native_retrieval_executed=False, scientific_fidelity_qualified=False,
                  production_episodes=0, implementation_sha256=sha256_file(Path(__file__)))
    started = time.perf_counter()
    try:
        receipt = json.loads(normalization_report.read_text())
        source_sha, target_sha = sha256_file(source), sha256_file(normalized)
        if (receipt['status'] != 'APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION'
                or receipt['source_sha256'] != source_sha or receipt['output_sha256'] != target_sha
                or receipt['normalization_source_revision'] != AUTHOR_REVISION
                or receipt['selected_connection_id'] != 50 or receipt['target_global_mean'] != 0.625):
            raise ValueError('bound author normalization receipt required')
        header = network_header(source)
        groups = header['groups']
        pyramidal = [g.group_id for g in groups if g.name == 'CA3_Pyramidal']
        if len(pyramidal) != 1:
            raise ValueError('one source pyramidal group required')
        offset = header['payload_offset'] + 8
        with source.open('rb') as stream:
            original_header = stream.read(offset)
        with normalized.open('rb') as stream:
            if stream.read(offset) != original_header:
                raise ValueError('normalization changed network header or group identities')
        blocks, count = struct.unpack_from('<ii', original_header, offset-8)
        if blocks != 1 or count <= 0:
            raise ValueError('one nonempty serialized network required')
        expected_size = offset + count*32
        if source.stat().st_size != expected_size or normalized.stat().st_size != expected_size:
            raise ValueError('synapse payload length differs')
        sizes = np.array([g.units for g in groups])
        raw = np.memmap(source, mode='r', dtype='<i4', offset=offset, shape=(count, 8))
        actual = np.memmap(normalized, mode='r', dtype='<i4', offset=offset, shape=(count, 8))
        try:
            selected_count = 0
            for start in range(0, count, chunk_records):
                rows = raw[start:start+chunk_records]
                for group, unit in [(0, 2), (1, 3)]:
                    if ((rows[:, group] < 0).any() or (rows[:, group] >= len(groups)).any()
                            or (rows[:, unit] < 0).any()):
                        raise ValueError('invalid source group or unit identity')
                    if (rows[:, unit] >= sizes[rows[:, group]]).any():
                        raise ValueError('source group-local unit outside group')
                selected = rows[:, 4] == 50
                if (rows[selected, :2] != pyramidal[0]).any():
                    raise ValueError('connection50 differs from pyramidal-to-pyramidal')
                if not np.isfinite(rows[:, 5:7].view('<f4')).all():
                    raise ValueError('nonfinite source synapse weight')
                selected_count += int(selected.sum())
            if selected_count == 0:
                raise ValueError('no selected pyramidal synapses')
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='ca3-normalization-replay-', dir=scratch) as folder:
                values = np.memmap(Path(folder)/'author_weights.f64', mode='w+', dtype='<f8',
                                   shape=(selected_count,))
                try:
                    cursor = 0
                    for start in range(0, count, chunk_records):
                        rows = raw[start:start+chunk_records]
                        selected = rows[rows[:, 4] == 50, 5].view('<f4')
                        values[cursor:cursor+len(selected)] = selected
                        cursor += len(selected)
                    mean = float(np.mean(values))
                    if not np.isfinite(mean) or mean <= 0:
                        raise ValueError('positive finite author mean required')
                    for start in range(0, count, chunk_records):
                        rows = raw[start:start+chunk_records]
                        expected = np.array(rows, copy=True)
                        selected = rows[:, 4] == 50
                        # Same scalar expression as modifyMat.ipynb, in two operations.
                        weights = rows[selected, 5].view('<f4').astype(np.float64)
                        expected[selected, 5] = ((weights/mean)*0.625).astype('<f4').view('<i4')
                        if not np.array_equal(expected, actual[start:start+len(rows)]):
                            bad = np.argwhere(expected != actual[start:start+len(rows)])[0]
                            raise ValueError(f'normalized bytes differ at record{start+int(bad[0])}, word{int(bad[1])}')
                finally:
                    del values
            if (receipt['selected_synapses'] != selected_count or receipt['synapses'] != count
                    or receipt['original_global_mean'] != mean or receipt['source_units'] != header['units']
                    or receipt['output_bytes'] != expected_size):
                raise ValueError('normalization receipt counts or mean differ')
            if sha256_file(source) != source_sha or sha256_file(normalized) != target_sha:
                raise ValueError('normalization inputs changed during replay')
            result.update(status='PASS_NORMALIZATION_REPLAY', all_record_bytes_verified=True,
                          source_sha256=source_sha, normalized_sha256=target_sha,
                          normalization_report_sha256=sha256_file(normalization_report),
                          units=header['units'], simulation_seconds=header['simulation_seconds'],
                          synapses=count, selected_synapses=selected_count, bytes_compared=expected_size,
                          original_global_mean=mean, target_global_mean=0.625,
                          normalization_source_revision=AUTHOR_REVISION,
                          numpy_version=np.__version__, weight_tolerance=0,
                          unchanged_header_and_nonselected_fields=True)
        finally:
            del raw, actual
    except BaseException as error:
        result.update(error=str(error), error_type=type(error).__name__)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'normalized', 'normalization-report', 'output', 'scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('normalization replay requires compute allocation')
    result = verify(args.source, args.normalized, args.normalization_report, args.output, args.scratch)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Apply the author's global CA3 pyramidal-weight normalization with bounded RAM.

Source: jkopsick/cell_assembly_formation_retrieval at
8c6c185994f8d54b7a3e027badcfaa7d15f232d1, modifyMat.ipynb cells2,3,6.
A contiguous float64 mapped vector preserves the author's selected-weight
order and NumPy mean. Division precedes multiplication, then float32 storage.
This prepares a retrieval input; it does not establish retrieval success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time

import numpy as np

from .carlsim_records import network_header
from .provenance import sha256_file

CONNECTION_ID = 50
TARGET_MEAN = 0.625
SOURCE_REVISION = '8c6c185994f8d54b7a3e027badcfaa7d15f232d1'
SYNAPSE_DTYPE = np.dtype([
    ('pre_group', '<i4'), ('post_group', '<i4'),
    ('pre_unit', '<i4'), ('post_unit', '<i4'), ('connection', '<i4'),
    ('weight', '<f4'), ('max_weight', '<f4'), ('delay', '<i4')])


def _layout(source: Path) -> tuple[dict, int, int]:
    header = network_header(source)
    with source.open('rb') as stream:
        stream.seek(header['payload_offset'])
        counts = stream.read(8)
    if len(counts) != 8:
        raise ValueError('CA3_TRUNCATED_NETWORK_BLOCK_HEADER')
    networks, synapses = struct.unpack('<ii', counts)
    if networks != 1 or synapses <= 0:
        raise ValueError('CA3_NORMALIZATION_REQUIRES_ONE_NONEMPTY_NETWORK_BLOCK')
    offset = header['payload_offset'] + 8
    if source.stat().st_size != offset + synapses*SYNAPSE_DTYPE.itemsize:
        raise ValueError('CA3_SYNAPSE_PAYLOAD_LENGTH_MISMATCH')
    return header, offset, synapses


def _selected(block: np.ndarray, groups: list, pyramidal: int) -> np.ndarray:
    for group_key, unit_key in [('pre_group', 'pre_unit'), ('post_group', 'post_unit')]:
        ids = block[group_key]
        if (ids < 0).any() or (ids >= len(groups)).any() or (block[unit_key] < 0).any():
            raise ValueError('CA3_SYNAPSE_GROUP_OR_UNIT_INVALID')
        sizes = np.asarray([g.units for g in groups], dtype=np.int64)
        if (block[unit_key] >= sizes[ids]).any():
            raise ValueError('CA3_SYNAPSE_GROUP_LOCAL_UNIT_INVALID')
    if not np.isfinite(block['weight']).all() or not np.isfinite(block['max_weight']).all():
        raise ValueError('CA3_NONFINITE_SYNAPTIC_WEIGHT')
    selected = block['connection'] == CONNECTION_ID
    if ((block['pre_group'][selected] != pyramidal).any()
            or (block['post_group'][selected] != pyramidal).any()):
        raise ValueError('CA3_CONNECTION_50_IS_NOT_PYRAMIDAL_TO_PYRAMIDAL')
    return selected


def normalize(source: Path, output: Path, report_path: Path, scratch: Path,
              *, chunk_records: int = 65536) -> dict:
    source, output, report_path, scratch = map(Path, (source, output, report_path, scratch))
    if output.exists() or report_path.exists():
        raise FileExistsError('CA3_NORMALIZATION_OUTPUT_ALREADY_EXISTS')
    if type(chunk_records) is not int or chunk_records <= 0:
        raise ValueError('CA3_NORMALIZATION_CHUNK_INVALID')
    header, offset, count = _layout(source)
    matches = [g.group_id for g in header['groups'] if g.name == 'CA3_Pyramidal']
    if len(matches) != 1:
        raise ValueError('CA3_PYRAMIDAL_GROUP_REQUIRED')
    started = time.perf_counter()
    source_sha = sha256_file(source)
    report = dict(status='NORMALIZING', source_sha256=source_sha,
                  normalization_source_revision=SOURCE_REVISION,
                  source_units=header['units'], simulation_seconds=header['simulation_seconds'],
                  network_blocks=1, synapses=count, selected_connection_id=CONNECTION_ID,
                  target_global_mean=TARGET_MEAN, numpy_version=np.__version__,
                  expression='(float64(weight) / float64(global_mean)) * float64(0.625), stored float32',
                  source_retrieval_input_qualified=False, retrieval_executed=False,
                  scientific_fidelity_qualified=False, production_episodes=0)
    mapped = np.memmap(source, mode='r', dtype=SYNAPSE_DTYPE, offset=offset, shape=(count,))
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        selected_count = sum(int(_selected(mapped[i:i+chunk_records], header['groups'], matches[0]).sum())
                             for i in range(0, count, chunk_records))
        if selected_count == 0:
            raise ValueError('CA3_EMPTY_PYRAMIDAL_CONNECTION_SET')
        with tempfile.TemporaryDirectory(prefix='ca3_normalization_', dir=scratch) as temporary:
            folder = Path(temporary)
            weights = np.memmap(folder/'selected_weights.f64', mode='w+', dtype=np.float64,
                                shape=(selected_count,))
            cursor = 0
            for i in range(0, count, chunk_records):
                block = mapped[i:i+chunk_records]
                values = block['weight'][block['connection'] == CONNECTION_ID]
                weights[cursor:cursor+len(values)] = values
                cursor += len(values)
            if cursor != selected_count:
                raise ValueError('CA3_SELECTED_WEIGHT_COUNT_CHANGED')
            # One contiguous reduction, just like np.mean(author_python_float_list).
            mean = np.mean(weights)
            if not np.isfinite(mean) or mean <= 0:
                raise ValueError('CA3_POSITIVE_GLOBAL_PYRAMIDAL_MEAN_REQUIRED')
            temporary_output = folder/'normalized_network.dat'
            shutil.copyfile(source, temporary_output)
            target = np.memmap(temporary_output, mode='r+', dtype=SYNAPSE_DTYPE,
                               offset=offset, shape=(count,))
            for i in range(0, count, chunk_records):
                block = mapped[i:i+chunk_records]
                selected = block['connection'] == CONNECTION_ID
                values = block['weight'][selected].astype(np.float64)
                normalized = (values/mean)*TARGET_MEAN
                if not np.isfinite(normalized.astype(np.float32)).all():
                    raise ValueError('CA3_NORMALIZED_FLOAT32_WEIGHT_NOT_FINITE')
                target['weight'][i:i+chunk_records][selected] = normalized
            target.flush()
            del target, weights
            if sha256_file(source) != source_sha:
                raise ValueError('CA3_INPUT_CHANGED_DURING_NORMALIZATION')
            output_sha = sha256_file(temporary_output)
            # The output may be on another filesystem; never overwrite its name.
            with output.open('xb') as destination, temporary_output.open('rb') as source_stream:
                shutil.copyfileobj(source_stream, destination, length=4*1024**2)
            if sha256_file(output) != output_sha:
                raise ValueError('CA3_NORMALIZED_OUTPUT_COPY_DIFFERS')
        report.update(status='APPLIED_AUTHOR_GLOBAL_WEIGHT_NORMALIZATION',
                      selected_synapses=selected_count, original_global_mean=float(mean),
                      output_sha256=output_sha, output_bytes=output.stat().st_size,
                      elapsed_seconds=time.perf_counter()-started,
                      implementation_sha256=sha256_file(Path(__file__)),
                      temporary_weight_bytes=selected_count*8,
                      unchanged_fields='all bytes except selected weight fields at record offsets20..23')
    except BaseException as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        del mapped
        with report_path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'report', 'scratch'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--chunk-records', type=int, default=65536)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('CA3_NORMALIZATION_REQUIRES_COMPUTE_ALLOCATION')
    result = normalize(args.source, args.output, args.report, args.scratch, chunk_records=args.chunk_records)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Offline native Opto source reading with explicit partial-data boundaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .provenance import freeze_json, runtime_manifest, sha256_file
from .qualification import NwbSpikeTimes

TRIAL_COLUMNS = ('id', 'start_time', 'stop_time', 'duration', 'power', 'site',
                 'num_pulses', 'wavelength', 'emission_location', 'stimulation_type',
                 'stim_id', 'inter_pulse_interval', 'inter_trial_interval', 'rise_time')
UNIT_COLUMNS = ('id', 'unit_name', 'default_qc', 'spike_times_index', 'device_name', 'brain_region')


class _Bundle:
    """Consume the existing source copy-out receipt; never infer missing bytes."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        receipt = json.loads((self.root / 'copy_out.json').read_text())
        self.files = {}
        self.consumed = {}
        for row in receipt['files']:
            name = row['path']
            if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                    or name in self.files):
                raise ValueError('OPTO_SOURCE_RECEIPT_PATH_INVALID')
            self.files[name] = row
        self.metadata = self.read_json('nwb_zmetadata.json')['metadata']
        if self.metadata.get('.zattrs', {}).get('neurodata_type') != 'NWBFile':
            raise ValueError('OPTO_SOURCE_NOT_NWB')

    def verified_file(self, name: str) -> Path:
        if name not in self.files:
            raise ValueError(f'OPTO_SOURCE_FILE_NOT_VERIFIED:{name}')
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f'OPTO_SOURCE_FILE_MISSING:{name}')
        if name not in self.consumed:
            row = self.files[name]
            if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
                raise ValueError(f'OPTO_SOURCE_BYTES_CHANGED:{name}')
            self.consumed[name] = row['sha256']
        return path

    def read_json(self, name: str) -> Any:
        return json.loads(self.verified_file(name).read_text(encoding='utf-8'))

    def array(self, name: str) -> '_SourceArray':
        return _SourceArray(self, name)


class _SourceArray:
    """One-dimensional source arrays with verified chunks and native decoding."""
    def __init__(self, bundle: _Bundle, name: str):
        self.bundle, self.name = bundle, name
        self.prefix = f'selected_source_arrays/{name}'
        for suffix in ('.zarray', '.zattrs'):
            actual = bundle.read_json(f'{self.prefix}/{suffix}')
            if actual != bundle.metadata.get(f'{name}/{suffix}'):
                raise ValueError(f'OPTO_ARRAY_METADATA_MISMATCH:{name}:{suffix}')
        spec = bundle.metadata[f'{name}/.zarray']
        if spec.get('zarr_format') != 2 or len(spec.get('shape', [])) != 1:
            raise ValueError(f'OPTO_ONE_DIMENSIONAL_ZARR2_REQUIRED:{name}')
        self.shape = tuple(spec['shape'])
        self.ndim = 1
        self.chunk_size = spec['chunks'][0]
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError(f'OPTO_CHUNK_SIZE_INVALID:{name}')
        self.attrs = bundle.metadata[f'{name}/.zattrs']
        # Zarr3.2.1 open_array ignores config kwargs for existing arrays. The
        # public scoped setting constructs an array that retains strict reads;
        # leaving this context restores the process's prior configuration.
        with zarr.config.set({'array.read_missing_chunks': False}):
            self.native = zarr.open_array(str(bundle.root / self.prefix), mode='r')
        if getattr(self.native.config, 'read_missing_chunks', True):
            raise RuntimeError('OPTO_ZARR_STRICT_READ_UNAVAILABLE')

    def __getitem__(self, selection: slice) -> np.ndarray:
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError('OPTO_CONTIGUOUS_SOURCE_SLICE_REQUIRED')
        lo = 0 if selection.start is None else int(selection.start)
        hi = self.shape[0] if selection.stop is None else int(selection.stop)
        if lo < 0 or hi < lo or hi > self.shape[0]:
            raise ValueError('OPTO_SOURCE_SLICE_OUT_OF_BOUNDS')
        if hi > lo:
            for chunk in range(lo // self.chunk_size, (hi - 1) // self.chunk_size + 1):
                self.bundle.verified_file(f'{self.prefix}/{chunk}')
        value = np.asarray(self.native[lo:hi])
        if value.shape != (hi - lo,):
            raise ValueError('OPTO_SOURCE_SLICE_LENGTH_CHANGED')
        return value


def _column_values(value: np.ndarray, name: str) -> np.ndarray:
    # VLenUTF8 decodes into object arrays. Store strings without pickle while
    # recording the original source dtype separately; reject mixed objects.
    if value.dtype.kind in ('O', 'T'):
        if not all(isinstance(x, str) for x in value):
            raise ValueError(f'OPTO_NONSCALAR_STRING_COLUMN:{name}')
        return np.asarray(value.tolist(), dtype=str)
    return value


def check_opto_source(source_dir: Path, output: Path, config: dict[str, Any],
                      command: list[str], diagnostic_unit_ids: list[int]) -> tuple[dict[str, Any], int]:
    """Read a declared training anchor without granting dataset/model eligibility."""
    settings = config.get('opto_source')
    if not isinstance(settings, dict):
        raise ValueError('OPTO_SOURCE_CONFIG_REQUIRED')
    if (not diagnostic_unit_ids or any(type(x) is not int for x in diagnostic_unit_ids)
            or len(set(diagnostic_unit_ids)) != len(diagnostic_unit_ids)):
        raise ValueError('OPTO_EXPLICIT_UNIQUE_UNIT_IDS_REQUIRED')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'OPTO_OUTPUT_EXISTS:{output}')
    bundle = _Bundle(source_dir)
    context = bundle.read_json('source_context.json')
    subject = bundle.read_json('subject.json')
    expected = settings['engineering_anchor_subject_id']
    if (subject.get('subject_id') != expected
            or context.get('engineering_scope', {}).get('subject_id') != expected):
        raise ValueError('OPTO_TRAINING_ANCHOR_IDENTITY_MISMATCH')
    if (context.get('source', {}).get('origin') != settings['source_origin']
            or context.get('source', {}).get('bucket') != settings['source_bucket']
            or context.get('nwb_prefix') != settings['nwb_prefix']):
        raise ValueError('OPTO_SOURCE_LOCATION_MISMATCH')
    if settings['required_future_split'] != 'development_training_only':
        raise ValueError('OPTO_DEVELOPMENT_TRAINING_SCOPE_REQUIRED')

    for table, required in (('intervals/trials', TRIAL_COLUMNS), ('units', UNIT_COLUMNS)):
        table_attributes = bundle.metadata.get(f'{table}/.zattrs')
        names = table_attributes.get('colnames') if isinstance(table_attributes, dict) else None
        if (not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names)
                or not (set(required) - {'id', 'spike_times_index'}).issubset(names)):
            raise ValueError(f'OPTO_TABLE_COLUMN_DECLARATION_INVALID:{table}')
    columns, attributes, dtypes = {}, {}, {}
    for table, names in (('intervals/trials', TRIAL_COLUMNS), ('units', UNIT_COLUMNS)):
        for name in names:
            key = f'{table}/{name}'
            array = bundle.array(key)
            raw = array[:]
            columns[key] = _column_values(raw, key)
            dtypes[key] = str(raw.dtype)
            attributes[key] = array.attrs
        ids = columns[f'{table}/id']
        if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or not len(ids):
            raise ValueError(f'OPTO_TABLE_IDS_INVALID:{table}')
        if any(len(columns[f'{table}/{name}']) != len(ids) for name in names):
            raise ValueError(f'OPTO_TABLE_LENGTH_MISMATCH:{table}')
    starts, stops = columns['intervals/trials/start_time'], columns['intervals/trials/stop_time']
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(stops)) or np.any(stops < starts):
        raise ValueError('OPTO_TRIAL_TIME_INVALID')
    for name in ('start_time', 'stop_time'):
        if attributes[f'intervals/trials/{name}'].get('unit', 'seconds') not in ('s', 'seconds'):
            raise ValueError('OPTO_TRIAL_TIME_UNIT_INVALID')
    ids = columns['units/id']
    qc = columns['units/default_qc']
    if qc.dtype.kind != 'b':
        raise ValueError('OPTO_SOURCE_QC_FLAG_NOT_BOOLEAN')
    positions = {int(unit_id): i for i, unit_id in enumerate(ids)}
    for unit_id in diagnostic_unit_ids:
        if unit_id not in positions:
            raise ValueError(f'OPTO_UNIT_UNKNOWN:{unit_id}')
        if not qc[positions[unit_id]]:
            raise ValueError(f'OPTO_UNIT_NOT_SOURCE_QC_ELIGIBLE:{unit_id}')
    spike_array = bundle.array('units/spike_times')
    if spike_array.attrs.get('unit', 'seconds') not in ('s', 'seconds'):
        raise ValueError('OPTO_SPIKE_TIME_UNIT_INVALID')
    spikes = NwbSpikeTimes(spike_array, ids, columns['units/spike_times_index'], retain_values=True)
    selected, offsets = [], [0]
    for unit_id in diagnostic_unit_ids:
        values = spikes[unit_id]
        if values.dtype.kind != 'f' or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
            raise ValueError(f'OPTO_SPIKE_TIMES_INVALID:{unit_id}')
        selected.append(values)
        offsets.append(offsets[-1] + len(values))
    counts = np.asarray([spikes.count(int(unit_id)) for unit_id in ids], dtype=np.int64)
    omitted = {}
    for table, names in (('intervals/trials', TRIAL_COLUMNS), ('units', UNIT_COLUMNS)):
        declared = bundle.metadata.get(f'{table}/.zattrs', {}).get('colnames', [])
        omitted[table] = sorted(set(declared) - set(names) - {'spike_times'})
    summary = {
        'status': 'PARTIAL', 'scope': 'OFFLINE_NATIVE_SOURCE_READ_ENGINEERING_ONLY',
        'subject_id': expected, 'future_split_requirement': settings['required_future_split'],
        'source_unit_count': len(ids), 'source_default_qc_units': int(np.count_nonzero(qc)),
        'source_trial_count': len(starts), 'selected_unit_count': len(selected),
        'selected_unit_ids': diagnostic_unit_ids, 'selected_spike_count': offsets[-1],
        'total_source_spike_count': int(counts.sum()),
        'omitted_unit_count': len(ids) - len(selected), 'source_qc_not_read': int(np.count_nonzero(qc)) - len(selected),
        'source_columns_read': sorted(columns), 'source_columns_not_read': omitted,
        'spike_slices': spikes.consumed_units, 'dataset_ready': False,
        'physical_power_qualified': False, 'formal_training_eligible': False,
        'control_semantics': 'Original values and unit attributes preserved; duration/power label issues are unresolved by this reader. No rescaling or optical calibration is inferred.',
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'source_tables.npz', **columns)
    np.savez(output / 'selected_spikes.npz', unit_ids=np.asarray(diagnostic_unit_ids, dtype=ids.dtype),
             spike_offsets=np.asarray(offsets, dtype=np.int64), spike_times_s=np.concatenate(selected))
    np.savez(output / 'source_unit_counts.npz', unit_ids=ids, spike_counts=counts, source_default_qc=qc)
    freeze_json(output / 'source_attributes.json', {'columns': attributes, 'source_dtypes': dtypes,
                'spike_times': spike_array.attrs, 'context': context, 'subject': subject})
    freeze_json(output / 'summary.json', summary)
    names = ('source_tables.npz', 'selected_spikes.npz', 'source_unit_counts.npz', 'source_attributes.json', 'summary.json')
    manifest = {'schema': 'neural_population_opto_source_read_v1', **summary,
                'source_bundle': str(source_dir.resolve()), 'source_copy_out_sha256': sha256_file(source_dir / 'copy_out.json'),
                'consumed_source_files': bundle.consumed,
                'artifacts': {name: sha256_file(output / name) for name in names},
                'reader_sha256': sha256_file(Path(__file__)), 'zarr_version': zarr.__version__,
                **runtime_manifest(config, command, 2)}
    freeze_json(output / 'manifest.json', manifest)
    return summary, 2

"""Nominal Opto pulse timelines and complete-control engineering windows."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import freeze_json, runtime_manifest, sha256_file
from .spike_windows import bin_spikes
from .stimulus_windows import StimulusWindows, encode_stimulus_windows

WAVEFORM_FIELDS = ('emission_location', 'stimulation_type', 'wavelength', 'site',
                   'power_label', 'duration_s', 'rise_s', 'num_pulses', 'gap_s', 'stim_id')
FLOAT64_COMPOSITION_ULPS = 4


@dataclass(frozen=True)
class PulseTables:
    presentations: pd.DataFrame
    members: pd.DataFrame
    trials: pd.DataFrame
    catalog: pd.DataFrame


def build_pulse_tables(trials: pd.DataFrame) -> PulseTables:
    """Reconstruct nominal supports; caller must establish explicit second units.

    Inter-trial interval is context, not part of the repeated pulse waveform.
    Actual source start times anchor every train; never accumulate nominal gaps.
    """
    required = {'id', 'start_time', 'stop_time', *WAVEFORM_FIELDS}
    if not required.issubset(trials.columns) or not trials.columns.is_unique or trials.empty:
        raise ValueError('OPTO_PULSE_COLUMNS_REQUIRED')
    if trials.id.dtype.kind not in 'iu' or trials.id.duplicated().any():
        raise ValueError('OPTO_TRIAL_IDS_INVALID')
    numeric = ('start_time', 'stop_time', 'duration_s', 'rise_s', 'gap_s', 'num_pulses', 'power_label', 'site', 'wavelength')
    for name in numeric:
        values = trials[name].to_numpy()
        if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
            raise ValueError(f'OPTO_FINITE_NUMERIC_FIELD_REQUIRED:{name}')
    if (np.any(trials.duration_s <= 0) or np.any(trials.gap_s < 0) or np.any(trials.rise_s < 0)
            or np.any(trials.num_pulses < 1) or np.any(trials.num_pulses != np.rint(trials.num_pulses))):
        raise ValueError('OPTO_PULSE_TIMING_INVALID')
    span = trials.num_pulses * trials.duration_s + (trials.num_pulses - 1) * trials.gap_s
    atol = FLOAT64_COMPOSITION_ULPS * np.spacing(np.maximum(np.abs(trials.stop_time), 1.0))
    if np.any(np.abs(trials.stop_time - trials.start_time - span) > atol):
        raise ValueError('OPTO_TRAIN_SPAN_DISAGREES_WITH_SOURCE')
    pulse_rows, memberships, trial_rows, catalog = [], [], [], {}
    for row in trials.to_dict('records'):
        key = json.dumps([row[name] for name in WAVEFORM_FIELDS], separators=(',', ':'), allow_nan=False)
        catalog[key] = {'condition_id': key, 'stimulus_name': 'opto_nominal_pulse',
                        **{name: row[name] for name in WAVEFORM_FIELDS}}
        trial_rows.append({'trial_id': row['id'], 'condition_id': key, 'stimulus_name': 'opto_nominal_pulse'})
        for pulse in range(int(row['num_pulses'])):
            start = row['start_time'] + pulse * (row['duration_s'] + row['gap_s'])
            pulse_id = f"{row['id']}:{pulse}"
            pulse_rows.append({'presentation_id': pulse_id, 'source_trial_id': row['id'],
                               'pulse_index': pulse, 'start_time': start, 'stop_time': start + row['duration_s'],
                               'stimulus_name': 'opto_nominal_pulse', 'condition_id': key})
            memberships.append({'presentation_id': pulse_id, 'trial_id': row['id']})
    return PulseTables(pd.DataFrame(pulse_rows), pd.DataFrame(memberships), pd.DataFrame(trial_rows),
                       pd.DataFrame([catalog[key] for key in sorted(catalog)]))


def encode_fixed_control_windows(tables: PulseTables, absolute_edges_seconds: np.ndarray,
                                 catalog: pd.DataFrame) -> StimulusWindows:
    """Pad/reindex the existing interval encoder into a declared control vocabulary.

    Reuse the same catalog when evaluating different future control schedules.
    Exposure counts overlapping annotations; it does not add optical powers.
    """
    if (not {'condition_id', 'stimulus_name'}.issubset(catalog.columns)
            or catalog.condition_id.duplicated().any() or catalog.empty):
        raise ValueError('UNIQUE_CONTROL_CATALOG_REQUIRED')
    lookup = {key: i + 1 for i, key in enumerate(catalog.condition_id)}
    names = dict(zip(catalog.condition_id, catalog.stimulus_name))
    for row in tables.trials.itertuples():
        if row.condition_id not in lookup or names[row.condition_id] != row.stimulus_name:
            raise ValueError('CONTROL_OUTSIDE_DECLARED_CATALOG')
    dynamic = encode_stimulus_windows(tables.presentations, tables.members, tables.trials, absolute_edges_seconds)
    features = [dict(dynamic.features[0])]
    features.extend({'feature_index': i + 1, 'feature_kind': 'condition', 'stimulus_name': row.stimulus_name,
                     'condition_id': row.condition_id, 'frame': None} for i, row in enumerate(catalog.itertuples()))
    values = np.zeros((*dynamic.values.shape[:2], len(features)), dtype=np.float64)
    values[:, :, 0] = dynamic.values[:, :, 0]
    for feature in dynamic.features[1:]:
        if feature['frame'] is not None:
            raise ValueError('OPTO_CATALOG_CANNOT_ENCODE_MOVIE_FRAME')
        values[:, :, lookup[feature['condition_id']]] = dynamic.values[:, :, feature['feature_index']]
    return StimulusWindows(values, features, dynamic.overlap_fraction)


def check_opto_windows(source_dir: Path, control_dir: Path, output: Path, config: dict[str, Any],
                       command: list[str], trial_ids: list[int]) -> tuple[dict[str, Any], int]:
    """Numerical windows only; support and optical calibration remain unqualified."""
    if output.exists():
        raise FileExistsError(f'OPTO_WINDOW_OUTPUT_EXISTS:{output}')
    if not trial_ids or any(type(x) is not int for x in trial_ids) or len(set(trial_ids)) != len(trial_ids):
        raise ValueError('EXPLICIT_UNIQUE_TRIAL_IDS_REQUIRED')
    spec = config['opto_window_check']
    manifest = json.loads((source_dir / 'manifest.json').read_text())
    if (manifest.get('subject_id') != config['opto_source']['engineering_anchor_subject_id']
            or manifest.get('future_split_requirement') != 'development_training_only'):
        raise ValueError('OPTO_WINDOW_REQUIRES_DECLARED_TRAINING_ANCHOR')
    for name in ('source_tables.npz', 'selected_spikes.npz', 'summary.json', 'source_attributes.json'):
        if sha256_file(source_dir / name) != manifest['artifacts'][name]:
            raise ValueError(f'OPTO_WINDOW_SOURCE_CHANGED:{name}')
    control_receipt = json.loads((control_dir / 'copy_out.json').read_text())
    entries = {row['path']: row for row in control_receipt['files']}
    for name in ('opto_control.csv', 'assessment.json'):
        if sha256_file(control_dir / name) != entries[name]['sha256']:
            raise ValueError(f'OPTO_CONTROL_SOURCE_CHANGED:{name}')
    control_assessment = json.loads((control_dir / 'assessment.json').read_text())
    if control_assessment['subject_id'] != manifest['subject_id']:
        raise ValueError('OPTO_CONTROL_SUBJECT_MISMATCH')
    with np.load(source_dir / 'source_tables.npz', allow_pickle=False) as arrays:
        trials = pd.DataFrame({name.split('/')[-1]: arrays[name] for name in arrays.files if name.startswith('intervals/trials/')})
    with (control_dir / 'opto_control.csv').open(newline='') as stream:
        original = list(csv.DictReader(stream))
    mapping = {'': 'id', 'site': 'site', 'power': 'power', 'param_group': 'stim_id',
               'emission_location': 'emission_location', 'duration': 'duration', 'rise_time': 'rise_time',
               'num_pulses': 'num_pulses', 'pulse_interval': 'inter_pulse_interval', 'wavelength': 'wavelength',
               'type': 'stimulation_type', 'interval': 'inter_trial_interval'}
    numeric = {'', 'site', 'power', 'duration', 'rise_time', 'num_pulses', 'pulse_interval', 'wavelength', 'interval'}
    for raw, column in mapping.items():
        values = np.asarray([row[raw] for row in original], dtype=float if raw in numeric else str)
        if raw in {'duration', 'rise_time', 'pulse_interval'}:
            values = values / 1000
        if not np.array_equal(values, trials[column].to_numpy()):
            raise ValueError(f'OPTO_CONTROL_TABLE_MISMATCH:{column}')
    trials = trials.rename(columns={'duration': 'duration_s', 'rise_time': 'rise_s',
                                    'inter_pulse_interval': 'gap_s', 'power': 'power_label'})
    tables = build_pulse_tables(trials)
    indexed = trials.set_index('id', drop=False)
    if not set(trial_ids).issubset(indexed.index):
        raise ValueError('UNKNOWN_OPTO_WINDOW_TRIAL')
    selected = indexed.loc[trial_ids]
    with np.load(source_dir / 'selected_spikes.npz', allow_pickle=False) as data:
        ids, offsets, times = data['unit_ids'], data['spike_offsets'], data['spike_times_s']
        if (offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                or offsets[-1] != len(times)):
            raise ValueError('OPTO_SELECTED_SPIKE_OFFSETS_INVALID')
        spikes = {int(uid): times[offsets[i]:offsets[i + 1]] for i, uid in enumerate(ids)}
        windows = bin_spikes(spikes, ids, selected.start_time.to_numpy(), **spec)
    stimulus = encode_fixed_control_windows(tables, windows.absolute_edges_seconds, tables.catalog)
    edges = windows.absolute_edges_seconds
    pstart, pstop = tables.presentations.start_time.to_numpy(), tables.presentations.stop_time.to_numpy()
    pulse_trials = tables.presentations.source_trial_id.to_numpy()
    source_starts = trials.start_time.to_numpy()
    metadata_rows = []
    for i, trial in enumerate(trial_ids):
        boundary = float(selected.iloc[i].start_time)
        touched = np.flatnonzero((pstart < edges[i, -1]) & (pstop > edges[i, 0]))
        metadata_rows.append({'trial_id': trial, 'window_start': edges[i, 0], 'boundary': boundary,
                              'window_stop': edges[i, -1], 'intersecting_pulse_count': len(touched),
                              'distinct_stimulus_trials': len(np.unique(pulse_trials[touched])),
                              'later_stimulus_onsets': int(np.count_nonzero((source_starts > boundary) & (source_starts < edges[i, -1]))),
                              'scientific_window_eligible': False})
    output.mkdir(parents=True)
    np.savez_compressed(output / 'windows.npz', counts=windows.counts, unit_ids=windows.unit_ids,
                        relative_edges_seconds=windows.relative_edges_seconds,
                        absolute_edges_seconds=windows.absolute_edges_seconds,
                        history_bin_count=windows.history_bin_count, stimulus_features=stimulus.values,
                        concurrent_annotation_fraction=stimulus.overlap_fraction)
    tables.catalog.to_csv(output / 'control_catalog.csv', index=False)
    tables.presentations.to_csv(output / 'pulse_annotations.csv', index=False)
    pd.DataFrame(metadata_rows).to_csv(output / 'window_metadata.csv', index=False)
    freeze_json(output / 'stimulus_features.json', stimulus.features)
    summary = {'status': 'PARTIAL', 'subject_id': manifest['subject_id'], 'selected_trial_ids': trial_ids,
               'source_trial_count': len(trials), 'source_pulse_count': len(tables.presentations),
               'window_count': len(trial_ids), 'unique_animal_count': 1, 'unique_recording_count': 1,
               'unit_ids': windows.unit_ids.tolist(), 'count_shape': list(windows.counts.shape),
               'stimulus_shape': list(stimulus.values.shape), 'history_bin_count': windows.history_bin_count,
               'control_condition_count': len(tables.catalog), 'dataset_ready': False,
               'recording_coverage_qualified': False, 'physical_power_qualified': False,
               'pulse_timing': 'Nominal pulse support reconstructed from original CSV seconds and recorded electrical trigger times; not measured optical onset.',
               'future_control': 'Retrospectively supplied recorded schedule; not a claim of prospective availability.',
               'control_amplitude': 'Condition identity and interval exposure only; no optical power or within-pulse waveform inference.',
               'history_contract': 'Half-open history ends at boundary; control vocabulary fixed from this training anchor metadata before response analysis.',
               'independence': 'Overlapping windows and trials are not independent biological replicates; observation coverage remains unknown.'}
    freeze_json(output / 'summary.json', summary)
    names = ('windows.npz', 'control_catalog.csv', 'pulse_annotations.csv', 'window_metadata.csv', 'stimulus_features.json', 'summary.json')
    freeze_json(output / 'manifest.json', {'schema': 'opto_complete_control_windows_v1', **summary,
                'source_manifest_sha256': sha256_file(source_dir / 'manifest.json'),
                'control_csv_sha256': sha256_file(control_dir / 'opto_control.csv'),
                'artifacts': {name: sha256_file(output / name) for name in names},
                'implementation_sha256': sha256_file(Path(__file__)), **runtime_manifest(config, command, 2)})
    return summary, 2

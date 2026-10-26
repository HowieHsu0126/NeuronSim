"""Bounded real-input BRAID engineering check; no formal fidelity claim."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import tensorflow as tf

from .baseline_data import load_engineering_windows
from .braid import BRAIDMeanPredictor, SOURCE_REVISION
from .provenance import freeze_json, runtime_manifest, sha256_file


def check_braid(source_dir: Path, output_dir: Path, config: dict[str, Any],
                command: list[str], seed: int) -> tuple[dict[str, Any], int]:
    if output_dir.exists():
        raise FileExistsError(f'BRAID_CHECK_OUTPUT_EXISTS:{output_dir}')
    if seed not in config['training_seeds']:
        raise ValueError('UNREGISTERED_TRAINING_SEED')
    spec = config['braid_baseline_check']
    if type(spec['optimizer_steps']) is not int or spec['optimizer_steps'] < 1:
        raise ValueError('BRAID_POSITIVE_OPTIMIZER_STEPS_REQUIRED')
    data = load_engineering_windows(source_dir, config)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    output_dir.mkdir(parents=True)
    try:
        init_start = time.monotonic()
        model = BRAIDMeanPredictor(data.unit_ids, data.inputs.shape[-1], data.history_bins,
                    data.counts.shape[1] - data.history_bins, data.bin_seconds, spec['model'],
                    initialization_seed=seed)
        initialization_seconds = time.monotonic() - init_start
        fit_start = time.monotonic()
        with (output_dir / 'optimization.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = None
            for step in range(spec['optimizer_steps']):
                row = {'step': step, **model.fit_step(data.counts, data.inputs)}
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); stream.flush()
        fit_seconds = time.monotonic() - fit_start
        inference_start = time.monotonic()
        state = model.infer_state({'counts': data.counts[:, :data.history_bins],
                                   'stimulus': data.inputs[:, :data.history_bins]},
                                  {'unit_ids': data.unit_ids, 'bin_seconds': data.bin_seconds})
        inference_seconds = time.monotonic() - inference_start
        forecast_start = time.monotonic()
        means = model.forecast(state, data.inputs[:, data.history_bins:]).numpy()
        forecast_seconds = time.monotonic() - forecast_start
        gradient_start = time.monotonic()
        gradient = model.expected_count_gradient(state, data.inputs[:, data.history_bins:],
                                                 np.ones_like(means)).numpy()
        gradient_seconds = time.monotonic() - gradient_start
        model.save(output_dir / 'checkpoint')
        with h5py.File(output_dir / 'mean_forecast.h5', 'x') as h:
            h.create_dataset('expected_counts', data=means)
            h.create_dataset('boundary_state', data=state.boundary.numpy())
            h.create_dataset('total_expected_count_input_gradient', data=gradient)
            h.create_dataset('unit_ids', data=data.unit_ids)
            h.create_dataset('trial_ids', data=data.trial_ids, dtype=h5py.string_dtype())
            h.attrs['bin_seconds'] = data.bin_seconds
            h.attrs['history_bins'] = data.history_bins
            h.attrs['scope'] = 'engineering_only'
        if data.input_features is not None:
            freeze_json(output_dir / 'input_features.json', data.input_features)
        summary = {
            'status': 'PARTIAL', 'scope': 'engineering_only', 'model': 'braid_neural_only_poisson_mean',
            'source_revision': SOURCE_REVISION, 'training_seed': seed, 'formal_seed_complete': False,
            'trial_count': len(data.trial_ids), 'unit_ids': list(data.unit_ids),
            'known_input_channels': data.inputs.shape[-1], 'forecast_shape': list(means.shape),
            'training_origins_per_trial_per_head': model.block_samples,
            'training_targets_per_head': len(data.trial_ids) * model.block_samples,
            'parameter_count': int(model.native.model.count_params()),
            'last_preupdate_training_terms': row, 'checkpoint_contains_optimizer_state': True,
            'observation': 'native Poisson mean count per bin; no additional bin-width factor',
            'joint_spike_sampler': False, 'generative_fidelity_qualified': False,
            'inverse_intervention_qualified': False, 'scientific_claims_qualified': False,
            'timings_seconds': {'initialization': initialization_seconds, 'fit': fit_seconds,
                               'state_inference': inference_seconds, 'mean_forecast': forecast_seconds,
                               'input_gradient': gradient_seconds},
            'elapsed_seconds': time.monotonic() - start}
        freeze_json(output_dir / 'summary.json', summary)
        names = ['checkpoint/model.json', 'checkpoint/model.index', 'checkpoint/model.data-00000-of-00001',
                 'mean_forecast.h5', 'optimization.csv', 'summary.json']
        if data.input_features is not None:
            names.append('input_features.json')
        freeze_json(output_dir / 'manifest.json', {
            'schema': 'neural_population_braid_check_v1', 'scope': 'engineering_only',
            'source_revision': SOURCE_REVISION,
            'source_manifest': str((source_dir / 'manifest.json').resolve()),
            'source_manifest_sha256': sha256_file(source_dir / 'manifest.json'),
            'tensorflow_version': str(tf.__version__), 'dtype': 'float32', 'resumed': False,
            'started_at_utc': started, 'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            'artifacts': {name: sha256_file(output_dir / name) for name in names},
            **runtime_manifest(config, command, 2)})
    except Exception as exc:
        freeze_json(output_dir / 'failure.json', {
            'status': 'FAILED', 'scope': 'engineering_only', 'error_type': type(exc).__name__,
            'error': str(exc), 'seed': seed, 'started_at_utc': started,
            'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            **runtime_manifest(config, command, 1)})
        raise
    return {**summary, 'manifest': str(output_dir / 'manifest.json')}, 2

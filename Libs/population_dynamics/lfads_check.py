"""Bounded real-input engineering fit and conditional LFADS generation."""
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
from .lfads import LFADSConditionalSimulator, LFADSRandomStreams, SOURCE_REVISION
from .provenance import freeze_json, runtime_manifest, sha256_file


def check_lfads(source_dir: Path, output_dir: Path, config: dict[str, Any],
                command: list[str], seed: int) -> tuple[dict[str, Any], int]:
    if output_dir.exists():
        raise FileExistsError(f'LFADS_CHECK_OUTPUT_EXISTS:{output_dir}')
    if seed not in config['training_seeds']:
        raise ValueError('UNREGISTERED_TRAINING_SEED')
    spec = config['lfads_baseline_check']
    for name in ('optimizer_steps', 'n_samples'):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError('POSITIVE_INTEGER_REQUIRED:' + name)
    streams = LFADSRandomStreams.from_seeds(**spec['sampling_seeds'])
    data = load_engineering_windows(source_dir, config)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    output_dir.mkdir(parents=True)
    try:
        initialization_start = time.monotonic()
        with LFADSConditionalSimulator(data.unit_ids, data.inputs.shape[-1], data.history_bins,
                data.counts.shape[1] - data.history_bins, data.bin_seconds, spec['model'], initialization_seed=seed) as model:
            initialization_seconds = time.monotonic() - initialization_start
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
            sampling_start = time.monotonic()
            samples = model.simulate(state, data.inputs[:, data.history_bins:], streams, spec['n_samples'])
            sampling_seconds = time.monotonic() - sampling_start
            model.save(output_dir / 'checkpoint')
            parameters = sum(int(np.prod(v.shape.as_list())) for v in model._full.model.train_vars)
            with h5py.File(output_dir / 'count_samples.h5', 'x') as h:
                for name in ('counts', 'conditional_expected_counts', 'initial_latent_states',
                             'historical_controller_inputs', 'future_controller_inputs',
                             'boundary_generator_states', 'future_generator_states'):
                    h.create_dataset(name, data=getattr(samples, name))
                for name in ('initial_mean', 'initial_log_variance', 'controller_encoding_forward',
                             'controller_encoding_reverse', 'known_input_history'):
                    value = getattr(state, name)
                    if value is not None:
                        h.create_dataset('state/' + name, data=value)
                h.create_dataset('unit_ids', data=data.unit_ids)
                h.create_dataset('trial_ids', data=data.trial_ids, dtype=h5py.string_dtype())
                h.attrs['bin_seconds'] = data.bin_seconds
                h.attrs['history_bins'] = data.history_bins
                h.attrs['scope'] = 'engineering_only'
        if data.input_features is not None:
            freeze_json(output_dir / 'input_features.json', data.input_features)
        summary = {'status': 'PARTIAL', 'scope': 'engineering_only', 'model': 'original_lfads_prefix_adapter',
                   'training_seed': seed, 'formal_seed_complete': False, 'unit_ids': list(data.unit_ids),
                   'trial_count': len(data.trial_ids), 'known_input_channels': data.inputs.shape[-1],
                   'parameter_count': parameters, 'sample_shape': list(samples.counts.shape),
                   'posterior': 'observed-prefix variational distribution with ancestral generator feedback',
                   'future_latent_law': 'source AR(1) conditional on paired final historical controller input',
                   'observation_units': 'source expected counts per bin; no additional bin-width multiplier',
                   'sampling_seeds': spec['sampling_seeds'], 'last_preupdate_training_terms': row,
                   'timings_seconds': {'initialization': initialization_seconds, 'fit': fit_seconds,
                                       'state_inference': inference_seconds, 'trajectory_sampling': sampling_seconds},
                   'elapsed_seconds': time.monotonic() - start,
                   'checkpoint_training_rng_resume_supported': not model.loaded_for_inference,
                   'generative_fidelity_qualified': False, 'inverse_intervention_qualified': False,
                   'scientific_claims_qualified': False}
        freeze_json(output_dir / 'summary.json', summary)
        names = ['checkpoint/model.json', 'checkpoint/weights.npz', 'count_samples.h5', 'optimization.csv', 'summary.json']
        if data.input_features is not None:
            names.append('input_features.json')
        freeze_json(output_dir / 'manifest.json', {
            'schema': 'neural_population_lfads_check_v1', 'scope': 'engineering_only',
            'source_revision': SOURCE_REVISION, 'source_manifest': str((source_dir / 'manifest.json').resolve()),
            'source_manifest_sha256': sha256_file(source_dir / 'manifest.json'),
            'tensorflow_version': str(tf.__version__), 'dtype': 'float32', 'resumed': False,
            'started_at_utc': started, 'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            'artifacts': {name: sha256_file(output_dir / name) for name in names},
            **runtime_manifest(config, command, 2)})
    except Exception as exc:
        freeze_json(output_dir / 'failure.json', {'status': 'FAILED', 'scope': 'engineering_only',
            'error_type': type(exc).__name__, 'error': str(exc), 'seed': seed,
            'started_at_utc': started, 'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            **runtime_manifest(config, command, 1)})
        raise
    return {**summary, 'manifest': str(output_dir / 'manifest.json')}, 2

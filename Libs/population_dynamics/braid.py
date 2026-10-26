"""Neural-only BRAID mean forecasting with trial-local native training tensors.

This adapter uses the reference's Poisson readout option. It is deterministic:
neither repeated means nor an unvalidated observation draw are joint spike samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .provenance import freeze_json

SOURCE_REVISION = 'ShanechiLab/BRAID@f9735960fdacff8351829100ca67408dd7e0351f'


@dataclass(frozen=True)
class BRAIDState:
    boundary: tf.Tensor
    unit_ids: tuple[int, ...]
    bin_seconds: float


class BRAIDMeanPredictor:
    """Source A/K history inference, learned Afw/Kfw forecast and count decoder.

    Only the neural stage is used, with a regression RNN (not LSTM). Future known
    inputs enter both transition and readout with the source's one-bin offset.
    Training origins are local to each trial, with a common maximum-horizon
    truncation; the source's concatenated-trial target shifts are not used.
    """

    def __init__(self, unit_ids: tuple[int, ...], input_dim: int, history_bins: int,
                 horizon_bins: int, bin_seconds: float, settings: dict[str, Any],
                 *, initialization_seed: int):
        if not str(tf.__version__).startswith('2.5.'):
            raise RuntimeError('BRAID_REQUIRES_VERIFIED_TENSORFLOW_2_5_ENVIRONMENT')
        # Legacy private TensorFlow imports are only loaded in the verified runtime.
        from Libs.external.BRAID.RNNModel import RNNModel
        self.unit_ids = tuple(unit_ids)
        if (not self.unit_ids or len(set(self.unit_ids)) != len(self.unit_ids)
                or any(type(v) is not int or v < 0 for v in self.unit_ids)):
            raise ValueError('BRAID_UNIT_IDENTITIES_REQUIRED')
        for name, value, minimum in (('input_dim', input_dim, 0), ('history_bins', history_bins, 1),
                                     ('horizon_bins', horizon_bins, 1), ('seed', initialization_seed, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError('BRAID_INVALID_INTEGER:' + name)
        if type(bin_seconds) not in (float, int) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError('BRAID_POSITIVE_BIN_WIDTH_REQUIRED')
        required = {'latent_dim', 'hidden_units', 'steps_ahead', 'learning_rate', 'gradient_clip_norm'}
        if set(settings) != required:
            raise ValueError('BRAID_EXPLICIT_MODEL_SETTINGS_REQUIRED')
        self.settings = json.loads(json.dumps(settings))
        for name in ('latent_dim', 'hidden_units'):
            if type(settings[name]) is not int or settings[name] < 1:
                raise ValueError('BRAID_POSITIVE_MODEL_DIMENSION:' + name)
        for name in ('learning_rate', 'gradient_clip_norm'):
            if type(settings[name]) not in (float, int) or not math.isfinite(settings[name]) or settings[name] <= 0:
                raise ValueError('BRAID_POSITIVE_OPTIMIZER_SETTING:' + name)
        steps = settings['steps_ahead']
        if (not isinstance(steps, (list, tuple)) or not steps or any(type(v) is not int or v < 1 for v in steps)
                or list(steps) != sorted(set(steps)) or steps[0] != 1 or steps[-1] < 2):
            raise ValueError('BRAID_SORTED_UNIQUE_HORIZONS_REQUIRE_ONE_AND_FORWARD_TRAINING')
        self.steps_ahead = tuple(steps)
        self.block_samples = history_bins + horizon_bins - max(steps)
        if self.block_samples < 1:
            raise ValueError('BRAID_TRIAL_TOO_SHORT_FOR_TRAINING_HORIZONS')
        self.input_dim, self.history_bins, self.horizon_bins = input_dim, history_bins, horizon_bins
        self.bin_seconds, self.initialization_seed = float(bin_seconds), initialization_seed
        self.training_steps = 0
        tf.random.set_seed(initialization_seed)
        base = {'units': [settings['hidden_units']], 'activation': 'tanh',
                'output_activation': 'linear', 'use_bias': True}
        self.native = RNNModel(
            settings['latent_dim'], len(self.unit_ids) + input_dim, self.block_samples, None,
            ny_out=len(self.unit_ids), nft=input_dim, n1_in=input_dim,
            out_dist='poisson', has_prior_pred=False, name='BRAID_neural_', stateful=False,
            linear_cell=False, LSTM_cell=False, enable_forward_pred=True,
            multi_step_with_A_KC=False, observable_U_in_Cfw=True,
            steps_ahead=list(steps), steps_ahead_loss_weights=[1.0] * len(steps),
            optimizer_name='Adam', optimizer_args={'learning_rate': settings['learning_rate'],
                                                   'clipnorm': settings['gradient_clip_norm']},
            cell_args={'ASettings': dict(base), 'KSettings': dict(base),
                       'CSettings': {**base, 'output_activation': 'exponential'},
                       'learn_initial_state': True})

    @property
    def cell(self):
        return self.native.rnn.cell

    def _counts(self, value: Any, bins: int) -> np.ndarray:
        array = np.asarray(value)
        if (array.ndim != 3 or array.shape[0] < 1 or array.shape[1:] != (bins, len(self.unit_ids))
                or not np.isrealobj(array) or not np.isfinite(array).all() or (array < 0).any()
                or not np.equal(array, np.round(array)).all()):
            raise ValueError('BRAID_FINITE_INTEGER_COUNTS_WITH_EXPECTED_SHAPE_REQUIRED')
        return array.astype(np.float32)

    def _inputs(self, value: Any, batch: int, bins: int) -> tf.Tensor:
        tensor = tf.convert_to_tensor(value)
        if not (tensor.dtype.is_floating or tensor.dtype.is_integer):
            raise ValueError('BRAID_REAL_INPUT_REQUIRED')
        tensor = tf.cast(tensor, tf.float32)
        if tuple(tensor.shape) != (batch, bins, self.input_dim):
            raise ValueError('BRAID_INPUT_SHAPE_MISMATCH')
        tf.debugging.assert_all_finite(tensor, 'BRAID_NONFINITE_INPUT')
        return tensor

    def training_tensors(self, counts: Any, stimulus: Any):
        """All trials contribute; no target or known input crosses a trial boundary."""
        y = self._counts(counts, self.history_bins + self.horizon_bins)
        u = self._inputs(stimulus, len(y), y.shape[1])
        length = self.block_samples
        labels = [tf.convert_to_tensor(y[:, step:step + length]) for step in self.steps_ahead]
        inputs = [tf.concat([y[:, :length], u[:, :length]], axis=-1)]
        if self.input_dim:
            inputs.append(tf.stack([u[:, step:step + length] for step in self.steps_ahead], axis=-1))
            # The native cell allocates max_h entries but consumes max_h - 1.
            forward = [u[:, step:step + length] for step in range(1, max(self.steps_ahead))]
            forward.append(tf.zeros_like(u[:, :length]))
            inputs.append(tf.stack(forward, axis=-1))
        dummy = tf.zeros((len(y), length, self.settings['latent_dim']), tf.float32)
        return tuple(inputs), labels + [dummy] * (2 * len(self.steps_ahead))

    def fit_step(self, counts: Any, stimulus: Any) -> dict[str, float]:
        inputs, labels = self.training_tensors(counts, stimulus)
        values = self.native.model.train_on_batch(inputs, labels, return_dict=True)
        # Keras layer counters depend on process history; scientific log keys do not.
        result = {'loss': float(values['loss'])}
        for step in self.steps_ahead:
            result[f'L_pred_h{step}'] = float(values[f'{self.native.rnn.name}_{step}step_loss'])
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError('BRAID_NONFINITE_TRAINING_LOSS')
        for value in self.native.model.weights:
            tf.debugging.assert_all_finite(value, 'BRAID_NONFINITE_FITTED_WEIGHT')
        self.training_steps += 1
        return result

    def infer_state(self, history: dict[str, Any], metadata: dict[str, Any]) -> BRAIDState:
        if (tuple(metadata['unit_ids']) != self.unit_ids
                or metadata['bin_seconds'] != self.bin_seconds):
            raise ValueError('BRAID_HISTORY_IDENTITY_OR_UNITS_MISMATCH')
        if np.shape(history['counts'])[1:2] != (self.history_bins,):
            raise ValueError('BRAID_FINITE_INTEGER_COUNTS_WITH_EXPECTED_SHAPE_REQUIRED')
        return self.infer_prefix(history, metadata)

    def infer_prefix(self, history: dict[str, Any], metadata: dict[str, Any]) -> BRAIDState:
        """Apply source history recursion to all observed bins before an origin.

        Training tensors keep their configured complete-trial length. Forecast
        evaluation can use different causal origin times without changing the
        trained model or discarding the earlier observed prefix.
        """
        if (tuple(metadata['unit_ids']) != self.unit_ids
                or metadata['bin_seconds'] != self.bin_seconds):
            raise ValueError('BRAID_HISTORY_IDENTITY_OR_UNITS_MISMATCH')
        shape = np.shape(history['counts'])
        if len(shape) != 3 or shape[1] < 1:
            raise ValueError('BRAID_NONEMPTY_OBSERVED_PREFIX_REQUIRED')
        bins = int(shape[1])
        y = self._counts(history['counts'], bins)
        u = self._inputs(history['stimulus'], len(y), bins)
        state = self.cell.get_initial_state(batch_size=len(y), dtype=tf.float32)
        for index in range(bins):
            state, _ = self.cell.applyRecursion(state, tf.concat([y[:, index], u[:, index]], axis=-1))
        tf.debugging.assert_all_finite(state, 'BRAID_NONFINITE_BOUNDARY_STATE')
        return BRAIDState(tf.stop_gradient(state), self.unit_ids, self.bin_seconds)

    def forecast(self, state: BRAIDState, stimulus: Any) -> tf.Tensor:
        if state.unit_ids != self.unit_ids or state.bin_seconds != self.bin_seconds:
            raise ValueError('BRAID_STATE_IDENTITY_OR_UNITS_MISMATCH')
        boundary = tf.convert_to_tensor(state.boundary, dtype=tf.float32)
        if boundary.shape.rank != 2 or boundary.shape[1] != self.settings['latent_dim'] or boundary.shape[0] < 1:
            raise ValueError('BRAID_STATE_SHAPE_MISMATCH')
        shape = np.shape(stimulus)
        if len(shape) != 3 or shape[1] < 1:
            raise ValueError('BRAID_FUTURE_INPUT_SEQUENCE_REQUIRED')
        u = self._inputs(stimulus, int(boundary.shape[0]), int(shape[1]))
        tf.debugging.assert_all_finite(boundary, 'BRAID_NONFINITE_BOUNDARY_STATE')
        output = []
        for index in range(int(u.shape[1])):
            current_u = u[:, index] if self.input_dim else None
            output.append(self.cell.get_pred_from_states_and_feedthrough(boundary, current_u, None))
            if index + 1 < int(u.shape[1]):
                boundary, _ = self.cell.applyForwardRecursion(boundary, current_u)
        result = tf.stack(output, axis=1)
        tf.debugging.assert_all_finite(result, 'BRAID_NONFINITE_FORECAST')
        return result

    def expected_count_gradient(self, state: BRAIDState, stimulus: Any, weights: Any) -> tf.Tensor:
        u = tf.convert_to_tensor(stimulus, dtype=tf.float32)
        w = tf.convert_to_tensor(weights, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(u)
            means = self.forecast(state, u)
            if w.shape != means.shape:
                raise ValueError('BRAID_OBJECTIVE_WEIGHTS_SHAPE_MISMATCH')
            tf.debugging.assert_all_finite(w, 'BRAID_NONFINITE_OBJECTIVE_WEIGHTS')
            objective = tf.reduce_sum(means * w)
        gradient = tape.gradient(objective, u)
        if gradient is None:
            raise RuntimeError('BRAID_MISSING_INPUT_GRADIENT')
        tf.debugging.assert_all_finite(gradient, 'BRAID_NONFINITE_INPUT_GRADIENT')
        return gradient

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        self.native.model.optimizer._create_all_weights(self.native.model.trainable_variables)
        checkpoint = tf.train.Checkpoint(model=self.native.model, optimizer=self.native.model.optimizer)
        checkpoint.write(str(path / 'model'))
        freeze_json(path / 'model.json', {
            'source_revision': SOURCE_REVISION, 'unit_ids': list(self.unit_ids), 'input_dim': self.input_dim,
            'history_bins': self.history_bins, 'horizon_bins': self.horizon_bins, 'bin_seconds': self.bin_seconds,
            'settings': self.settings, 'initialization_seed': self.initialization_seed,
            'training_steps': self.training_steps, 'tensorflow_version': tf.__version__,
            'observation': 'native Poisson mean count per bin; no joint sampler',
            'training': 'trial-local common-origin native multi-horizon loss'})

    @classmethod
    def load(cls, path: Path) -> BRAIDMeanPredictor:
        metadata = json.loads((path / 'model.json').read_text(encoding='utf-8'))
        if metadata['source_revision'] != SOURCE_REVISION:
            raise ValueError('BRAID_CHECKPOINT_SOURCE_MISMATCH')
        model = cls(tuple(metadata['unit_ids']), metadata['input_dim'], metadata['history_bins'],
                    metadata['horizon_bins'], metadata['bin_seconds'], metadata['settings'],
                    initialization_seed=metadata['initialization_seed'])
        # Adam slots must exist before a strict restore; no update is performed.
        model.native.model.optimizer._create_all_weights(model.native.model.trainable_variables)
        checkpoint = tf.train.Checkpoint(model=model.native.model, optimizer=model.native.model.optimizer)
        checkpoint.read(str(path / 'model')).assert_consumed()
        model.training_steps = metadata['training_steps']
        return model

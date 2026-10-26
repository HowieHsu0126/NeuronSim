"""History-conditioned LFADS sampling using the pinned original model equations.

Two graph lengths separate posterior encoding of the observed prefix from future
prior generation. Historical controller inputs are sampled ancestrally, preserving
source generator feedback. No future neural response is used for state inference
or sampling. Source Poisson outputs are expected counts per bin, not Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
import contextlib
import sys
import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import tensorflow.compat.v1 as tf

from Libs.external.lfads.lfads import LFADS as SourceLFADS
from .provenance import freeze_json


SOURCE_REVISION = 'fd700f0cb2e104544c445d9fbf3991d8388ff18a'
TRAINING_RANDOM_SCHEMA = 'tensorflow_stateless_update_and_site_v1'
VALIDATION_RANDOM_SCHEMA = 'tensorflow_stateless_negative_validation_counter_and_site_v1'


class _TrainingRandomness:
    """Independent stateless keys per update/site, without execution-order races."""

    def __init__(self, seed: int):
        self.seed = seed
        self.update = tf.placeholder_with_default(tf.constant(0, tf.int64), shape=(),
                                                  name='training_random_update')
        self.sites: list[str] = []

    def _key(self, kind: str):
        site = len(self.sites)
        self.sites.append(kind)
        key = tf.stack([tf.constant(self.seed, tf.int64), self.update])
        return tf.random.experimental.stateless_fold_in(key, tf.constant(site, tf.int64))

    def normal(self, shape, mean=0., stddev=1., dtype=tf.float32, seed=None, name=None):
        if seed is not None:
            raise ValueError('LFADS_TRAINING_SITE_SEED_IS_MANAGED_BY_ADAPTER')
        return tf.random.stateless_normal(shape, self._key('normal'), mean=mean,
                                          stddev=stddev, dtype=dtype, name=name)

    def dropout(self, x, keep_prob, noise_shape=None, seed=None, name=None):
        if seed is not None:
            raise ValueError('LFADS_TRAINING_SITE_SEED_IS_MANAGED_BY_ADAPTER')
        # Same keep-probability conversion as compat.v1.nn.dropout; native TF
        # implements elementwise Bernoulli masking and inverse-keep scaling.
        return tf.nn.experimental.stateless_dropout(x, rate=1 - keep_prob,
            seed=self._key('dropout'), noise_shape=noise_shape, name=name)


@dataclass(frozen=True)
class LFADSHistoryState:
    initial_mean: np.ndarray
    initial_log_variance: np.ndarray
    controller_encoding_forward: np.ndarray  # batch, observed bin, encoder unit
    controller_encoding_reverse: np.ndarray | None
    known_input_history: np.ndarray
    unit_ids: tuple[int, ...]
    bin_seconds: float
    uncertainty: str = 'observed_prefix_variational_path_distribution'


@dataclass(frozen=True)
class LFADSRandomStreams:
    initial: np.random.Generator
    dynamics: np.random.Generator
    observation: np.random.Generator

    @classmethod
    def from_seeds(cls, initial: int, dynamics: int, observation: int) -> LFADSRandomStreams:
        seeds = (initial, dynamics, observation)
        if any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != 3:
            raise ValueError('THREE_DISTINCT_NONNEGATIVE_RANDOM_STREAM_SEEDS_REQUIRED')
        return cls(*(np.random.default_rng(s) for s in seeds))


@dataclass(frozen=True)
class LFADSCountSamples:
    counts: np.ndarray  # draw, batch, future bin, observed unit
    conditional_expected_counts: np.ndarray
    initial_latent_states: np.ndarray
    historical_controller_inputs: np.ndarray
    future_controller_inputs: np.ndarray
    boundary_generator_states: np.ndarray
    future_generator_states: np.ndarray
    unit_ids: tuple[int, ...]
    bin_seconds: float

    @property
    def population_counts(self) -> np.ndarray:
        return self.counts.sum(axis=-1)


class _GraphView:
    """One explicit graph/session; does not alter global TF eager mode or seeds."""

    def __init__(self, hparams: dict[str, Any], unit_count: int, seed: int, kind: str,
                 device: str):
        grappler_policy = hparams.get('grappler_optimization', 'default')
        if grappler_policy not in ('default', 'off'):
            raise ValueError('LFADS_GRAPPLER_OPTIMIZATION_MUST_BE_DEFAULT_OR_OFF')
        self.graph = tf.Graph()
        self.tensorflow_device = '/cpu:0' if device == 'cpu' else '/gpu:0'
        config = tf.ConfigProto(intra_op_parallelism_threads=2,
                                inter_op_parallelism_threads=2,
                                device_count={'GPU': 0 if device == 'cpu' else 1},
                                allow_soft_placement=False)
        if device == 'cuda':
            config.gpu_options.allow_growth = True
        if grappler_policy == 'off':
            # Full unrolled graphs spend their first run in dependency reduction
            # and memory-planning passes. Execute the original graph while
            # preserving its control edges, equations and optimization target.
            config.graph_options.rewrite_options.disable_meta_optimizer = True
        self.session = tf.Session(graph=self.graph, config=config)
        self.model = None
        try:
            expected_type = 'CPU' if device == 'cpu' else 'GPU'
            available = [entry for entry in self.session.list_devices()
                         if entry.device_type == expected_type
                         and entry.name.endswith(f'/device:{expected_type}:0')]
            if not available:
                if device == 'cuda':
                    raise RuntimeError('LFADS_CUDA_REQUESTED_BUT_NO_VISIBLE_GPU')
                raise RuntimeError('LFADS_CPU_SESSION_DEVICE_UNAVAILABLE')
            self.execution_device = {'requested': device, 'device_type': expected_type,
                                     'tensorflow_device': available[0].name,
                                     'tensorflow_version': tf.__version__,
                                     'grappler_optimization': grappler_policy}
            # Strict placement also rejects unsupported CUDA kernels; it never
            # silently runs a requested GPU graph on the CPU.
            with self.graph.as_default(), self.session.as_default(), tf.device(self.tensorflow_device):
                tf.set_random_seed(seed)
                self.randomness = _TrainingRandomness(seed)
                with tf.variable_scope('LFADS'), contextlib.redirect_stdout(sys.stderr):
                    self.model = SourceLFADS(SimpleNamespace(**hparams), kind=kind,
                        datasets={'recording': {'train_data': np.zeros((1, hparams['num_steps'], unit_count), dtype=np.int64)}},
                        random_ops=self.randomness)
                self.variables = {v.name: v for v in tf.global_variables()}
                self.assign_inputs = {name: tf.placeholder(v.dtype.base_dtype, shape=v.shape,
                                                          name=f'restore_{i}')
                                      for i, (name, v) in enumerate(self.variables.items())}
                self.assign_ops = [tf.assign(v, self.assign_inputs[name]) for name, v in self.variables.items()]
                self.session.run(tf.global_variables_initializer())
        except Exception:
            self.close()
            raise

    def values(self) -> dict[str, np.ndarray]:
        return self.session.run(self.variables)

    def restore(self, values: dict[str, np.ndarray]) -> None:
        if set(values) != set(self.variables):
            raise ValueError('LFADS_CHECKPOINT_VARIABLE_NAMES_MISMATCH')
        feed = {}
        for name, variable in self.variables.items():
            value = np.asarray(values[name])
            if (tuple(variable.shape.as_list()) != value.shape or value.dtype != variable.dtype.as_numpy_dtype
                    or not np.isfinite(value).all()):
                raise ValueError('LFADS_CHECKPOINT_SHAPE_DTYPE_OR_VALUES_INVALID:' + name)
            feed[self.assign_inputs[name]] = value
        self.session.run(self.assign_ops, feed_dict=feed)

    def close(self) -> None:
        if self.model is not None and hasattr(self.model, 'writer'):
            self.model.writer.close()
        self.session.close()


class LFADSConditionalSimulator:
    """Original nonlinear LFADS core with a prefix-posterior / future-prior adapter.

    The encoder may be bidirectional within history. Each historical controller
    posterior is conditioned on previously imposed latent samples, retaining the
    source feedback mechanism. The resulting boundary is approximate, not a known
    physiological state. Capacity and all source model settings are explicit.
    """

    def __init__(self, unit_ids: Sequence[int], input_size: int, history_bins: int,
                 horizon_bins: int, bin_seconds: float, model_config: dict[str, Any],
                 *, initialization_seed: int, device: str = 'cpu'):
        if type(device) is not str or device not in ('cpu', 'cuda'):
            raise ValueError('LFADS_DEVICE_MUST_BE_CPU_OR_CUDA')
        self.device = device
        self.checkpoint_execution_device: dict[str, Any] | None = None
        ids = tuple(unit_ids)
        if not ids or any(type(u) is not int or u < 0 for u in ids) or len(set(ids)) != len(ids):
            raise ValueError('UNIQUE_NONNEGATIVE_INTEGER_UNIT_IDS_REQUIRED')
        if (type(input_size) is not int or input_size < 0
                or any(type(x) is not int or x < 1 for x in (history_bins, horizon_bins))):
            raise ValueError('INVALID_INPUT_HISTORY_OR_HORIZON_SIZE')
        if type(bin_seconds) not in (float, int) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError('POSITIVE_BIN_SECONDS_REQUIRED')
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**31:
            raise ValueError('NONNEGATIVE_TENSORFLOW_SEED_REQUIRED')
        for name in ('ic_dim', 'co_dim', 'ic_enc_dim', 'ci_enc_dim', 'con_dim', 'gen_dim', 'factors_dim'):
            if type(model_config[name]) is not int or model_config[name] < 1:
                raise ValueError('POSITIVE_LFADS_DIMENSION_REQUIRED:' + name)
        if type(model_config['controller_input_lag']) is not int or model_config['controller_input_lag'] < 0:
            raise ValueError('NONNEGATIVE_CONTROLLER_LAG_REQUIRED')
        if model_config['inject_ext_input_to_gen'] is not True:
            raise ValueError('KNOWN_INPUT_MUST_DRIVE_GENERATOR')
        keep_prob = model_config.get('keep_prob', 1.)
        if type(keep_prob) not in (int, float) or not math.isfinite(keep_prob) or not 0 < keep_prob <= 1:
            raise ValueError('LFADS_TRAINING_KEEP_PROBABILITY_MUST_BE_IN_0_1')
        self.unit_ids, self.input_size = ids, input_size
        self.history_bins, self.horizon_bins = history_bins, horizon_bins
        self.training_history_bins, self.training_horizon_bins = history_bins, horizon_bins
        self.bin_seconds, self.initialization_seed = float(bin_seconds), initialization_seed
        self.model_config = dict(model_config)
        self._temporary = tempfile.TemporaryDirectory(prefix='neuronsim_lfads_')
        self._full = self._prefix = None
        self._prefix_dirty = True
        self.loaded_for_inference = False
        self.training_updates = 0
        try:
            common = {**model_config, 'dataset_names': ['recording'], 'dataset_dims': {'recording': len(ids)},
                      'ndatasets': 1, 'batch_size': 1, 'ext_input_dim': input_size, 'output_dist': 'poisson',
                      'num_steps_for_gen_ic': history_bins, 'max_ckpt_to_keep': 1, 'max_ckpt_to_keep_lve': 1,
                      'lfads_save_dir': self._temporary.name}
            self._full = _GraphView({**common, 'num_steps': history_bins + horizon_bins}, len(ids), initialization_seed, 'train', device)
            # The source prior_sample graph still constructs posterior encoders,
            # while omitting an unused sequence KL that rejects H=1. We fetch
            # posterior parameters, never its prior initial-state draw.
            self._prefix = _GraphView({**common, 'num_steps': history_bins}, len(ids), initialization_seed, 'prior_sample', device)
            self.execution_device = dict(self._full.execution_device)
            self._gradient_views = {}
            self._generation_fetches = {}
            self._sync_prefix()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for view in (self._prefix, self._full):
            if view is not None:
                view.close()
        self._prefix = self._full = None
        self._temporary.cleanup()

    def __enter__(self) -> LFADSConditionalSimulator:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _float_array(values: Any) -> np.ndarray:
        raw = np.asarray(values)
        if not np.issubdtype(raw.dtype, np.number) or not np.isrealobj(raw):
            raise ValueError('REAL_NUMERIC_ARRAY_REQUIRED')
        result = raw.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('FINITE_ARRAY_REQUIRED')
        return result

    def _counts(self, values: Any, bins: int) -> np.ndarray:
        y = np.asarray(values)
        if (not np.issubdtype(y.dtype, np.number) or not np.isrealobj(y)
                or y.ndim != 3 or y.shape[0] < 1 or y.shape[1:] != (bins, len(self.unit_ids))
                or not np.isfinite(y).all() or (y < 0).any() or (y != np.round(y)).any()
                or (y > np.iinfo(np.int32).max).any()):
            raise ValueError('FINITE_INT32_COUNT_SEQUENCE_REQUIRED')
        return y.astype(np.int32)

    def _stimulus(self, values: Any, batch: int, bins: int) -> np.ndarray:
        u = self._float_array(values)
        if u.shape != (batch, bins, self.input_size):
            raise ValueError('KNOWN_INPUT_SHAPE_MISMATCH')
        return u

    def _sync_prefix(self) -> None:
        if self._prefix_dirty:
            values = self._full.values()
            self._prefix.restore({name: values[name] for name in self._prefix.variables})
            self._prefix_dirty = False

    def fit_step(self, counts: Any, stimulus: Any) -> dict[str, float]:
        """Source training update; gradient_norm is measured before source clipping."""
        if self.loaded_for_inference:
            raise ValueError('CHECKPOINT_IS_INFERENCE_ONLY_TRAINING_RNG_RESUME_UNSUPPORTED')
        if type(self.training_updates) is not int or not 0 <= self.training_updates < 2**63 - 1:
            raise ValueError('LFADS_NONNEGATIVE_TRAINING_COUNTER_REQUIRED')
        y = self._counts(counts, self.history_bins + self.horizon_bins)
        u = self._stimulus(stimulus, len(y), y.shape[1])
        m = self._full.model
        feed = m.build_feed_dict('recording', y, u, keep_prob=self.model_config.get('keep_prob', 1.))
        feed[self._full.randomness.update] = self.training_updates
        fetch = {'source_cost': m.cost, 'reconstruction_nll': m.recon_cost, 'kl_cost': m.kl_cost,
                 'l2_cost': m.l2_cost, 'kl_weight': m.kl_weight, 'l2_weight': m.l2_weight,
                 'gradient_norm': m.grad_global_norm, 'update': m.train_op}
        values = self._full.session.run(fetch, feed_dict=feed)
        self.training_updates += 1
        values.pop('update')
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError('NONFINITE_LFADS_OPTIMIZATION')
        self._prefix_dirty = True
        return {key: float(value) for key, value in values.items()}

    def validation_loss(self, counts: Any, stimulus: Any, *, validation_seed: int) -> dict[str, float]:
        """Full posterior selection loss; does not update parameters or training RNG.

        Training uses nonnegative update counters. Validation occupies the
        disjoint signed-int64 range through counter = -(validation_seed + 1).
        The existing graph and all historical training keys remain unchanged.
        Dropout is disabled; posterior variational draws remain stochastic.
        This objective conditions on full responses and is not a forecast score.
        """
        if type(validation_seed) is not int or not 0 <= validation_seed < 2**63:
            raise ValueError('LFADS_VALIDATION_SEED_MUST_BE_NONNEGATIVE_INT63')
        y = self._counts(counts, self.history_bins + self.horizon_bins)
        u = self._stimulus(stimulus, len(y), y.shape[1])
        source = self._full.model
        feed = source.build_feed_dict('recording', y, u, keep_prob=1.)
        feed[self._full.randomness.update] = -(validation_seed + 1)
        values = self._full.session.run(dict(source_cost=source.cost,
            reconstruction_nll=source.recon_cost, kl_cost=source.kl_cost,
            l2_cost=source.l2_cost, kl_weight=source.kl_weight, l2_weight=source.l2_weight), feed_dict=feed)
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError('NONFINITE_LFADS_VALIDATION_LOSS')
        return {key: float(value) for key, value in values.items()}

    def infer_state(self, history: dict[str, Any], metadata: dict[str, Any] | None = None) -> LFADSHistoryState:
        if not isinstance(history, dict) or not {'counts', 'stimulus'} <= history.keys():
            raise ValueError('COUNT_AND_KNOWN_STIMULUS_HISTORY_REQUIRED')
        if metadata is not None and (tuple(metadata.get('unit_ids', self.unit_ids)) != self.unit_ids
                or metadata.get('bin_seconds', self.bin_seconds) != self.bin_seconds):
            raise ValueError('HISTORY_IDENTITY_OR_BIN_WIDTH_MISMATCH')
        y = self._counts(history['counts'], self.history_bins)
        u = self._stimulus(history['stimulus'], len(y), self.history_bins)
        self._sync_prefix()
        m = self._prefix.model
        fetch = [m.posterior_zs_g0.mean, m.posterior_zs_g0.logvar, m.ci_enc_fwd]
        if not self.model_config['do_causal_controller']:
            fetch.append(m.ci_enc_rev)
        values = self._prefix.session.run(fetch, feed_dict=m.build_feed_dict('recording', y, u, keep_prob=1.))
        forward = np.stack(values[2], axis=1)
        reverse = np.stack(values[3], axis=1) if len(values) == 4 else None
        for value in (*values[:2], forward, *((reverse,) if reverse is not None else ())):
            if not np.isfinite(value).all():
                raise FloatingPointError('NONFINITE_LFADS_PREFIX_ENCODING')
        return LFADSHistoryState(values[0], values[1], forward, reverse, u, self.unit_ids, self.bin_seconds)

    def _state_and_future(self, state: LFADSHistoryState, stimulus: Any) -> tuple[int, np.ndarray]:
        if state.unit_ids != self.unit_ids or state.bin_seconds != self.bin_seconds:
            raise ValueError('STATE_IDENTITY_OR_BIN_WIDTH_MISMATCH')
        mean = np.asarray(state.initial_mean)
        if mean.ndim != 2 or mean.shape[0] < 1 or mean.shape[1] != self.model_config['ic_dim']:
            raise ValueError('INITIAL_POSTERIOR_SHAPE_MISMATCH')
        batch = len(mean)
        for value, shape in ((state.initial_mean, mean.shape), (state.initial_log_variance, mean.shape),
                (state.controller_encoding_forward, (batch, self.history_bins, self.model_config['ci_enc_dim'])),
                (state.known_input_history, (batch, self.history_bins, self.input_size))):
            if np.shape(value) != shape or not np.isfinite(value).all():
                raise ValueError('STATE_SHAPE_OR_VALUES_INVALID')
        if self.model_config['do_causal_controller']:
            if state.controller_encoding_reverse is not None:
                raise ValueError('UNEXPECTED_REVERSE_ENCODING')
        elif (np.shape(state.controller_encoding_reverse) != np.shape(state.controller_encoding_forward)
              or not np.isfinite(state.controller_encoding_reverse).all()):
            raise ValueError('REVERSE_ENCODING_SHAPE_OR_VALUES_INVALID')
        u = self._float_array(stimulus)
        if u.ndim == 2:
            u = np.broadcast_to(u, (batch, *u.shape)).copy()
        if (u.ndim != 3 or u.shape[0] != batch or u.shape[2] != self.input_size
                or not 1 <= u.shape[1] <= self.horizon_bins):
            raise ValueError('FUTURE_INPUT_SHAPE_OR_HORIZON_MISMATCH')
        return batch, u

    @staticmethod
    def _streams(rng: LFADSRandomStreams, n_samples: int) -> None:
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError('POSITIVE_SAMPLE_COUNT_REQUIRED')
        if not isinstance(rng, LFADSRandomStreams):
            raise TypeError('LFADS_RANDOM_STREAMS_REQUIRED')
        generators = (rng.initial, rng.dynamics, rng.observation)
        if any(not isinstance(g, np.random.Generator) for g in generators) or len({id(g) for g in generators}) != 3:
            raise ValueError('DISTINCT_NUMPY_GENERATORS_REQUIRED')

    def _history_feed(self, state: LFADSHistoryState, initial: np.ndarray,
                      stimulus: np.ndarray) -> dict[Any, Any]:
        samples, batch, _ = initial.shape
        total = self.history_bins + self.horizon_bins
        flat = samples * batch
        m = self._full.model
        padded = np.zeros((batch, self.horizon_bins, self.input_size), dtype=np.float32)
        padded[:, :stimulus.shape[1]] = stimulus
        known = np.concatenate((state.known_input_history, padded), axis=1)
        known = np.broadcast_to(known, (samples, *known.shape)).reshape(flat, total, self.input_size)
        # Counts feed only the source encoders, likelihood and derived batch size.
        # All encoder/g0 outputs below are imposed from the observed prefix; no
        # likelihood is fetched. Feed the remaining scalar dependency directly.
        # Leaving dataset_ph unfed also rejects any new response dependency.
        feed = {m.dataName: 'recording', m.batch_size: flat, m.keep_prob: 1.,
                m.g0s_val: initial.reshape(flat, -1)}
        if m.ext_input is not None:
            feed[m.ext_input] = known
        # Feed encoder outputs, not zero response observations, at the H boundary.
        for encoded, graph_encoded in ((state.controller_encoding_forward, m.ci_enc_fwd),
                                      (state.controller_encoding_reverse, m.ci_enc_rev)):
            if encoded is None:
                continue
            values = np.broadcast_to(encoded, (samples, *encoded.shape)).reshape(flat, self.history_bins, -1)
            zero = np.zeros((flat, values.shape[-1]), dtype=np.float32)
            for t in range(total):
                feed[graph_encoded[t]] = values[:, t] if t < self.history_bins else zero
        return feed

    def _draw_feed(self, state: LFADSHistoryState, stimulus: Any,
                   rng: LFADSRandomStreams, n_samples: int) -> tuple[dict[Any, Any], dict[str, np.ndarray], int]:
        batch, u = self._state_and_future(state, stimulus)
        self._streams(rng, n_samples)
        standard_deviation = np.exp(0.5 * state.initial_log_variance)
        if not np.isfinite(standard_deviation).all() or (standard_deviation <= 0).any():
            raise FloatingPointError('INVALID_INITIAL_POSTERIOR_VARIANCE')
        initial = (state.initial_mean + standard_deviation * rng.initial.standard_normal(
                    (n_samples, batch, self.model_config['ic_dim'])).astype(np.float32)).astype(np.float32)
        feed = self._history_feed(state, initial, u)
        m = self._full.model
        posteriors = m.posterior_zs_co[:self.history_bins]
        # Inject independent standardized noises, then let the original graph
        # execute its feedback-dependent posterior ancestry in one session call.
        # Imposing one sequence of posterior means for all particles would be wrong.
        noise_shape = (n_samples * batch, self.model_config['co_dim'])
        for posterior in posteriors:
            feed[posterior.noise_bxn] = rng.initial.standard_normal(noise_shape).astype(np.float32)
        history, posterior_means, posterior_log_variances = self._full.session.run(
            [[p.sample for p in posteriors], [p.mean for p in posteriors],
             [p.logvar for p in posteriors]], feed_dict=feed)
        posterior_scales = np.exp(0.5 * np.asarray(posterior_log_variances))
        if (any(not np.isfinite(value).all() for value in (*history, *posterior_means, *posterior_log_variances))
                or not np.isfinite(posterior_scales).all() or (posterior_scales <= 0).any()):
            raise FloatingPointError('INVALID_HISTORICAL_CONTROLLER_POSTERIOR')
        for t, value in enumerate(history):
            feed[m.controller_outputs[t]] = value
        phi, log_innovation_variance = self._full.session.run(
            [m.prior_zs_ar_con.phis_bxu, m.prior_zs_ar_con.logevars_1xu], feed_dict=feed)
        innovation_std = np.exp(0.5 * log_innovation_variance)
        if (not np.isfinite(phi).all() or not np.isfinite(innovation_std).all()
                or (innovation_std <= 0).any() or (phi < 0).any() or (phi > 1).any()):
            raise FloatingPointError('INVALID_FUTURE_AR_PRIOR')
        previous = history[-1]
        future = []
        for t in range(u.shape[1]):
            previous = (phi * previous + innovation_std * rng.dynamics.standard_normal(previous.shape).astype(np.float32)).astype(np.float32)
            feed[m.controller_outputs[self.history_bins + t]] = previous
            future.append(previous)
        latent = {'initial_latent_states': initial,
                  'historical_controller_inputs': np.stack(history, axis=1).reshape(n_samples, batch, self.history_bins, -1),
                  'future_controller_inputs': np.stack(future, axis=1).reshape(n_samples, batch, u.shape[1], -1)}
        return feed, latent, u.shape[1]

    def _read_future(self, feed: dict[Any, Any], n_samples: int, bins: int) -> dict[str, np.ndarray]:
        m = self._full.model
        h, stop = self.history_bins, self.history_bins + bins
        if bins not in self._generation_fetches:
            # Add only deterministic scalar reductions after the source graphs
            # are complete, preserving all variable names and random-site keys.
            # Inspect the whole requested path without copying its unit readouts
            # to the host. The returned arrays contain no historical backing.
            with self._full.graph.as_default(), tf.device(self._full.tensorflow_device):
                finite = tf.reduce_all(tf.stack([
                    tf.reduce_all(tf.math.is_finite(value))
                    for sequence in (m.output_dist_params, m.unclipped_log_rates, m.gen_states)
                    for value in sequence[:stop]]))
                unclipped = tf.reduce_all(tf.stack([
                    tf.reduce_all(tf.abs(value) < self.model_config['_clip_value'])
                    for value in m.unclipped_log_rates[:stop]]))
            self._generation_fetches[bins] = dict(finite=finite, unclipped=unclipped,
                means=m.output_dist_params[h:stop], boundary=m.gen_states[h - 1],
                states=m.gen_states[h:stop])
        result = self._full.session.run(self._generation_fetches[bins], feed_dict=feed)
        if not result['finite']:
            raise FloatingPointError('NONFINITE_LFADS_GENERATION')
        if not result['unclipped']:
            raise FloatingPointError('LFADS_SOURCE_LOG_RATE_CLIP_ACTIVE')
        mean = np.stack(result['means'], axis=1)
        generator = np.stack(result['states'], axis=1)
        batch = mean.shape[0] // n_samples
        return {'conditional_expected_counts': mean.reshape(n_samples, batch, bins, len(self.unit_ids)),
                'boundary_generator_states': result['boundary'].reshape(n_samples, batch, -1),
                'future_generator_states': generator.reshape(n_samples, batch, bins, -1)}

    def simulate(self, state: LFADSHistoryState, stimulus: Any, rng: LFADSRandomStreams,
                 n_samples: int) -> LFADSCountSamples:
        feed, latent, bins = self._draw_feed(state, stimulus, rng, n_samples)
        future = self._read_future(feed, n_samples, bins)
        latent['future_controller_inputs'] = latent['future_controller_inputs'][:, :, :bins]
        # Source logp is k*logr-exp(logr)-lgamma(k+1): exp(logr) is counts/bin.
        counts = rng.observation.poisson(future['conditional_expected_counts'])
        return LFADSCountSamples(counts=counts, **future, **latent,
                                 unit_ids=self.unit_ids, bin_seconds=self.bin_seconds)

    def expected_count_gradient(self, state: LFADSHistoryState, stimulus: Any,
                                readout_weights: Any, rng: LFADSRandomStreams, n_samples: int) -> np.ndarray:
        """Pathwise derivative of a weighted Monte Carlo count mean; no action claim."""
        if self.input_size == 0:
            raise ValueError('CONTROL_GRADIENT_REQUIRES_KNOWN_INPUTS')
        feed, _, bins = self._draw_feed(state, stimulus, rng, n_samples)
        future = self._read_future(feed, n_samples, bins)
        batch = future['conditional_expected_counts'].shape[1]
        weights = self._float_array(readout_weights)
        if weights.shape != (batch, bins, len(self.unit_ids)):
            raise ValueError('READOUT_WEIGHT_SHAPE_MISMATCH')
        if bins not in self._gradient_views:
            m = self._full.model
            with self._full.graph.as_default(), tf.device(self._full.tensorflow_device):
                weight_input = tf.placeholder(tf.float32, [None, bins, len(self.unit_ids)])
                rates = tf.stack(m.output_dist_params[self.history_bins:self.history_bins + bins], axis=1)
                # Feeding a tensor changes its value, not its symbolic derivative.
                # All imposed latent draws/encodings must be explicit gradient stops.
                imposed = [m.g0s_val, *m.controller_outputs, *m.ci_enc_fwd,
                           *(x for x in (m.ci_enc_rev or ()) if x is not None)]
                gradient_op = tf.gradients(tf.reduce_sum(rates * weight_input), m.ext_input,
                                          stop_gradients=imposed)[0]
                self._gradient_views[bins] = weight_input, gradient_op
        weight_input, gradient_op = self._gradient_views[bins]
        feed[weight_input] = np.broadcast_to(weights, (n_samples, *weights.shape)).reshape(-1, bins, len(self.unit_ids))
        gradient = self._full.session.run(gradient_op, feed_dict=feed)
        if not np.isfinite(gradient).all():
            raise FloatingPointError('NONFINITE_LFADS_CONTROL_GRADIENT')
        return gradient.reshape(n_samples, batch, self.history_bins + self.horizon_bins, self.input_size).mean(0)[:, self.history_bins:self.history_bins + bins]

    def save(self, directory: Path) -> None:
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        np.savez(directory / 'weights.npz', **self._full.values())
        freeze_json(directory / 'model.json', {'schema': 'neural_population_lfads_conditional_v2',
            'source_revision': SOURCE_REVISION, 'unit_ids': list(self.unit_ids), 'input_size': self.input_size,
            'history_bins': self.history_bins, 'horizon_bins': self.horizon_bins,
            'training_history_bins': self.training_history_bins, 'training_horizon_bins': self.training_horizon_bins,
            'bin_seconds': self.bin_seconds,
            'model_config': self.model_config, 'initialization_seed': self.initialization_seed,
            'execution_device': self.execution_device,
            'training_rng_resume_supported': not self.loaded_for_inference,
            'training_rng_schema': TRAINING_RANDOM_SCHEMA, 'training_updates': self.training_updates,
            'training_random_sites': self._full.randomness.sites,
            'checkpoint_purpose': ('parameter_and_inference_replay' if self.loaded_for_inference
                                   else 'training_and_inference_replay')})

    @classmethod
    def load(cls, directory: Path, *, history_bins: int | None = None,
             horizon_bins: int | None = None, device: str = 'cpu') -> LFADSConditionalSimulator:
        """Restore weights at explicit inference lengths, preserving training lengths.

        Omitting both overrides retains checkpoint behavior. Either changed
        length makes the loaded view inference-only, including reallocations
        between history and horizon that leave the total graph length equal.
        Execution placement is selected by device, independently of saved device
        provenance. Moving a checkpoint preserves state, not bitwise cross-device
        training trajectories. Legacy checkpoints need no device metadata.
        """
        record = json.loads((directory / 'model.json').read_text())
        if (record['schema'] not in ('neural_population_lfads_conditional_v1', 'neural_population_lfads_conditional_v2')
                or record['source_revision'] != SOURCE_REVISION):
            raise ValueError('LFADS_CHECKPOINT_SCHEMA_OR_SOURCE_MISMATCH')
        keys = ('unit_ids', 'input_size', 'history_bins', 'horizon_bins', 'bin_seconds', 'model_config', 'initialization_seed')
        configuration = {key: record[key] for key in keys}
        training_history = record.get('training_history_bins', record['history_bins'])
        training_horizon = record.get('training_horizon_bins', record['horizon_bins'])
        if any(type(value) is not int or value < 1 for value in (training_history, training_horizon)):
            raise ValueError('LFADS_CHECKPOINT_TRAINING_LENGTHS_INVALID')
        if history_bins is not None:
            configuration['history_bins'] = history_bins
        if horizon_bins is not None:
            configuration['horizon_bins'] = horizon_bins
        model = cls(**configuration, device=device)
        model.checkpoint_execution_device = record.get('execution_device')
        model.training_history_bins, model.training_horizon_bins = training_history, training_horizon
        try:
            with np.load(directory / 'weights.npz', allow_pickle=False) as archive:
                model._full.restore({key: archive[key] for key in archive.files})
            model._prefix_dirty = True
            model.loaded_for_inference = True
            model.training_updates = None
            if record.get('training_rng_resume_supported'):
                if (record['schema'] != 'neural_population_lfads_conditional_v2'
                        or record.get('training_rng_schema') != TRAINING_RANDOM_SCHEMA
                        or type(record.get('training_updates')) is not int
                        or not 0 <= record['training_updates'] < 2**63):
                    raise ValueError('LFADS_CHECKPOINT_TRAINING_RANDOM_STATE_INVALID')
                unchanged_lengths = (model.history_bins == record['history_bins'] == training_history
                                     and model.horizon_bins == record['horizon_bins'] == training_horizon)
                if unchanged_lengths:
                    if record.get('training_random_sites') != model._full.randomness.sites:
                        raise ValueError('LFADS_CHECKPOINT_TRAINING_RANDOM_SITES_CHANGED')
                    model.training_updates = record['training_updates']
                    model.loaded_for_inference = False
            return model
        except Exception:
            model.close()
            raise

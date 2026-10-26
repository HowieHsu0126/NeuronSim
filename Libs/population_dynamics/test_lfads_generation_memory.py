"""Bounded generation against the pre-repair LFADS response/feed reference.

Reference adapter SHA256: 083ae71a344a1cf8c931c2ae1479d88c36c9bb5fddf64b6b05218f6451f5d9f7.
The old methods are retained only here as independent inference migration oracles.
"""
from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow.compat.v1 as tf
import yaml

from .lfads import LFADSConditionalSimulator, LFADSCountSamples, LFADSRandomStreams


class LegacyGenerationLFADS(LFADSConditionalSimulator):
    """Exact previous feed and host checks, with the same pinned source graph."""

    def _history_feed(self, state, initial, stimulus):
        samples, batch, _ = initial.shape
        total = self.history_bins + self.horizon_bins
        flat = samples * batch
        m = self._full.model
        padded = np.zeros((batch, self.horizon_bins, self.input_size), dtype=np.float32)
        padded[:, :stimulus.shape[1]] = stimulus
        known = np.concatenate((state.known_input_history, padded), axis=1)
        known = np.broadcast_to(known, (samples, *known.shape)).reshape(flat, total, self.input_size)
        feed = m.build_feed_dict('recording', np.zeros((flat, total, len(self.unit_ids)), dtype=np.int32), known, keep_prob=1.)
        feed[m.g0s_val] = initial.reshape(flat, -1)
        for encoded, graph_encoded in ((state.controller_encoding_forward, m.ci_enc_fwd),
                                      (state.controller_encoding_reverse, m.ci_enc_rev)):
            if encoded is None:
                continue
            values = np.broadcast_to(encoded, (samples, *encoded.shape)).reshape(flat, self.history_bins, -1)
            zero = np.zeros((flat, values.shape[-1]), dtype=np.float32)
            for t in range(total):
                feed[graph_encoded[t]] = values[:, t] if t < self.history_bins else zero
        return feed

    def _read_future(self, feed, n_samples, bins):
        m = self._full.model
        means, log_rates, states = self._full.session.run(
            [m.output_dist_params[:self.history_bins + bins], m.unclipped_log_rates[:self.history_bins + bins],
             m.gen_states[:self.history_bins + bins]], feed_dict=feed)
        mean = np.stack(means, axis=1)
        log_rate = np.stack(log_rates, axis=1)
        generator = np.stack(states, axis=1)
        if any(not np.isfinite(v).all() for v in (mean, log_rate, generator)):
            raise FloatingPointError('NONFINITE_LFADS_GENERATION')
        if (np.abs(log_rate) >= self.model_config['_clip_value']).any():
            raise FloatingPointError('LFADS_SOURCE_LOG_RATE_CLIP_ACTIVE')
        batch = mean.shape[0] // n_samples
        h = self.history_bins
        return {'conditional_expected_counts': mean[:, h:h + bins].reshape(n_samples, batch, bins, len(self.unit_ids)),
                'boundary_generator_states': generator[:, h - 1].reshape(n_samples, batch, -1),
                'future_generator_states': generator[:, h:h + bins].reshape(n_samples, batch, bins, -1)}


def streams():
    return LFADSRandomStreams.from_seeds(51001, 51002, 51003)


def assert_streams_equal(left, right):
    for name in ('initial', 'dynamics', 'observation'):
        assert getattr(left, name).bit_generator.state == getattr(right, name).bit_generator.state


def assert_values_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def backing_bytes(value):
    while isinstance(value.base, np.ndarray):
        value = value.base
    return value.nbytes


@pytest.fixture(scope='module', params=['bidirectional_factors', 'bidirectional_rates', 'causal'])
def pair(request):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'configs/neural_population_dynamics.yaml').read_text())['lfads_baseline_check']['model']
    config = {**config, 'controller_input_lag': 2, 'keep_prob': .65,
              'implicit_identity_readin': True, 'grappler_optimization': 'off',
              'do_causal_controller': request.param == 'causal',
              'do_feed_factors_to_controller': request.param != 'causal',
              'feedback_factors_or_rates': 'rates' if request.param == 'bidirectional_rates' else 'factors'}
    counts = (np.arange(2 * 8 * 3).reshape(2, 8, 3) % 4).astype(np.int64)
    stimulus = np.linspace(0., .8, 16, dtype=np.float32).reshape(2, 8, 1)
    with LFADSConditionalSimulator([109, 7, 43], 1, 5, 3, .005, config, initialization_seed=41, device=os.environ.get('LFADS_TEST_DEVICE', 'cpu')) as current, \
         LegacyGenerationLFADS([109, 7, 43], 1, 5, 3, .005, config, initialization_seed=41, device=os.environ.get('LFADS_TEST_DEVICE', 'cpu')) as legacy:
        assert current._full.randomness.sites == legacy._full.randomness.sites
        assert_values_equal(current._full.values(), legacy._full.values())
        assert_values_equal(current.fit_step(counts, stimulus), legacy.fit_step(counts, stimulus))
        history = dict(counts=counts[:, :5], stimulus=stimulus[:, :5])
        yield current, legacy, counts, stimulus, current.infer_state(history), legacy.infer_state(history)


def test_all_sample_arrays_streams_cache_and_backing_match_reference(pair):
    current, legacy, _, stimulus, state, old_state = pair
    before = current._full.values()
    sites = list(current._full.randomness.sites)
    graph_sizes = {}
    for bins in (3, 1, 2, 3, 1):
        rng, old_rng = streams(), streams()
        actual = current.simulate(state, stimulus[:, 5:5 + bins], rng, 16)
        expected = legacy.simulate(old_state, stimulus[:, 5:5 + bins], old_rng, 16)
        for field in fields(LFADSCountSamples):
            left, right = getattr(actual, field.name), getattr(expected, field.name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right, err_msg=field.name)
            else:
                assert left == right
        assert_streams_equal(rng, old_rng)
        for name in ('conditional_expected_counts', 'boundary_generator_states', 'future_generator_states'):
            value = getattr(actual, name)
            assert backing_bytes(value) == value.nbytes
            assert backing_bytes(getattr(expected, name)) > value.nbytes
        assert len(current._generation_fetches) == len(set(graph_sizes) | {bins})
        graph_size = len(current._full.graph.get_operations())
        if bins in graph_sizes:
            assert graph_size == max(graph_sizes.values())
        graph_sizes[bins] = graph_size
    assert current._full.randomness.sites == sites
    assert_values_equal(before, current._full.values())


def test_no_response_buffer_and_missing_response_ancestry_fails(pair):
    current, _, _, stimulus, state, _ = pair
    feed, _, _ = current._draw_feed(state, stimulus[:, 5:], streams(), 16)
    source = current._full.model
    assert source.dataset_ph not in feed
    assert feed[source.batch_size] == 32
    assert not any(isinstance(value, np.ndarray) and value.shape == (32, 8, 3)
                   for value in feed.values())
    incomplete = dict(feed)
    incomplete.pop(source.ci_enc_fwd[0])
    with pytest.raises(tf.errors.InvalidArgumentError, match='placeholder'):
        current._full.session.run(source.posterior_zs_co[2].mean, feed_dict=incomplete)


def fixed_path_feed(current, state, stimulus):
    feed, _, bins = current._draw_feed(state, stimulus[:, 5:], streams(), 2)
    source = current._full.model
    sequences = (source.output_dist_params, source.unclipped_log_rates, source.gen_states)
    values = current._full.session.run(sequences, feed_dict=feed)
    for sequence, arrays in zip(sequences, values):
        feed.update(zip(sequence, arrays))
    return feed, bins


@pytest.mark.parametrize('sequence', ['output_dist_params', 'unclipped_log_rates', 'gen_states'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_history_only_nonfinite_is_rejected_with_finite_future(pair, sequence, bad):
    current, _, _, stimulus, state, _ = pair
    feed, bins = fixed_path_feed(current, state, stimulus)
    tensor = getattr(current._full.model, sequence)[0]
    feed[tensor] = np.full_like(feed[tensor], bad)
    for reader in (current._read_future, lambda *args: LegacyGenerationLFADS._read_future(current, *args)):
        with pytest.raises(FloatingPointError, match='NONFINITE_LFADS_GENERATION'):
            reader(feed, 2, bins)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_history_only_clip_boundary_is_rejected(pair, sign):
    current, _, _, stimulus, state, _ = pair
    feed, bins = fixed_path_feed(current, state, stimulus)
    tensor = current._full.model.unclipped_log_rates[0]
    bound = np.float32(sign * current.model_config['_clip_value'])
    feed[tensor] = np.full_like(feed[tensor], np.nextafter(bound, np.float32(0.)))
    expected = LegacyGenerationLFADS._read_future(current, feed, 2, bins)
    assert_values_equal(current._read_future(feed, 2, bins), expected)
    feed[tensor].fill(bound)
    for reader in (current._read_future, lambda *args: LegacyGenerationLFADS._read_future(current, *args)):
        with pytest.raises(FloatingPointError, match='LOG_RATE_CLIP_ACTIVE'):
            reader(feed, 2, bins)


def test_only_requested_future_is_checked(pair):
    current, _, _, stimulus, state, _ = pair
    feed, _ = fixed_path_feed(current, state, stimulus)
    tensor = current._full.model.unclipped_log_rates[-1]
    feed[tensor] = np.full_like(feed[tensor], np.nan)
    expected = LegacyGenerationLFADS._read_future(current, feed, 2, 1)
    assert_values_equal(current._read_future(feed, 2, 1), expected)
    with pytest.raises(FloatingPointError, match='NONFINITE_LFADS_GENERATION'):
        current._read_future(feed, 2, 3)


def test_gradients_next_training_updates_and_checkpoint_match_reference(pair, tmp_path):
    current, legacy, counts, stimulus, state, old_state = pair
    for bins in (1, 3, 2, 1):
        future = stimulus[:, 5:5 + bins]
        weights = np.linspace(.2, 1., 2 * bins * 3, dtype=np.float32).reshape(2, bins, 3)
        rng, old_rng = streams(), streams()
        actual = current.expected_count_gradient(state, future, weights, rng, 16)
        expected = legacy.expected_count_gradient(old_state, future, weights, old_rng, 16)
        np.testing.assert_array_equal(actual, expected)
        assert np.any(actual != 0)
        assert_streams_equal(rng, old_rng)
    for _ in range(3):
        assert_values_equal(current.fit_step(counts, stimulus), legacy.fit_step(counts, stimulus))
        assert_values_equal(current._full.values(), legacy._full.values())
    current.save(tmp_path / 'current')
    legacy.save(tmp_path / 'legacy')
    assert json.loads((tmp_path / 'current/model.json').read_text()) == json.loads((tmp_path / 'legacy/model.json').read_text())
    with LFADSConditionalSimulator.load(tmp_path / 'current', device=os.environ.get('LFADS_TEST_DEVICE', 'cpu')) as restored:
        assert restored._full.randomness.sites == legacy._full.randomness.sites
        assert_values_equal(restored._full.values(), legacy._full.values())
        assert_values_equal(restored.fit_step(counts, stimulus), legacy.fit_step(counts, stimulus))
        assert_values_equal(restored._full.values(), legacy._full.values())

"""Source-graph equivalence, information boundaries and LFADS sampling semantics."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import yaml

from .__main__ import main, parser
from .lfads import LFADSConditionalSimulator, LFADSRandomStreams
from .provenance import sha256_file
from .test_reduced_rank import write_windows

CONFIG = Path(__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml'


def streams(initial=51, dynamics=52, observation=53):
    return LFADSRandomStreams.from_seeds(initial, dynamics, observation)


class LFADSAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = yaml.safe_load(CONFIG.read_text())['lfads_baseline_check']['model']
        cls.spec = {**cls.spec, 'controller_input_lag': 2}
        cls.eager_before = tf.executing_eagerly()
        cls.model = LFADSConditionalSimulator([12, 4], 1, 3, 3, .05, cls.spec, initialization_seed=41)
        cls.y = np.array([[[0, 2], [1, 0], [2, 1], [1, 3], [0, 1], [1, 1]],
                          [[2, 1], [1, 0], [0, 1], [0, 2], [1, 0], [0, 1]]], dtype=np.int64)
        cls.u = np.linspace(0, .8, 12, dtype=np.float32).reshape(2, 6, 1)
        cls.first_terms = cls.model.fit_step(cls.y, cls.u)
        cls.state = cls.model.infer_state({'counts': cls.y[:, :3], 'stimulus': cls.u[:, :3]})

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_optimization_and_global_execution_mode(self):
        self.assertTrue(np.isfinite(list(self.first_terms.values())).all())
        self.assertGreater(self.first_terms['gradient_norm'], 0)
        self.assertEqual(tf.executing_eagerly(), self.eager_before)
        self.assertFalse(self.model.model_config['do_causal_controller'])
        self.assertTrue(self.model.model_config['do_feed_factors_to_controller'])

    def test_extended_graph_matches_native_prefix_with_feedback(self):
        model, state = self.model, self.state
        model._sync_prefix()
        native, extended = model._prefix.model, model._full.model
        g0 = state.initial_mean + np.array([[.25, -.3], [-.4, .2]], dtype=np.float32)
        prefix_feed = native.build_feed_dict('recording', self.y[:, :3], self.u[:, :3], keep_prob=1.)
        prefix_feed[native.g0s_val] = g0
        extended_feed = model._history_feed(state, g0[None], self.u[:, 3:])
        latent = np.array([[[.2], [-.1], [.4]], [[-.3], [.5], [.1]]], dtype=np.float32)
        for t in range(3):
            left = model._prefix.session.run([native.posterior_zs_co[t].mean,
                                             native.posterior_zs_co[t].logvar], feed_dict=prefix_feed)
            right = model._full.session.run([extended.posterior_zs_co[t].mean,
                                             extended.posterior_zs_co[t].logvar], feed_dict=extended_feed)
            for a, b in zip(left, right):
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
            prefix_feed[native.controller_outputs[t]] = latent[:, t]
            extended_feed[extended.controller_outputs[t]] = latent[:, t]
            for field in ('gen_states', 'con_states', 'output_dist_params'):
                a = model._prefix.session.run(getattr(native, field)[t], feed_dict=prefix_feed)
                b = model._full.session.run(getattr(extended, field)[t], feed_dict=extended_feed)
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        # A changed sampled historical input must affect subsequent feedback in
        # exactly the same way as the original H-bin source graph.
        prefix_feed.pop(native.controller_outputs[2])
        extended_feed.pop(extended.controller_outputs[2])
        before = model._full.session.run(extended.posterior_zs_co[2].mean, feed_dict=extended_feed)
        prefix_feed[native.controller_outputs[0]] = latent[:, 0] + 2.
        extended_feed[extended.controller_outputs[0]] = latent[:, 0] + 2.
        a = model._prefix.session.run(native.posterior_zs_co[2].mean, feed_dict=prefix_feed)
        b = model._full.session.run(extended.posterior_zs_co[2].mean, feed_dict=extended_feed)
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        self.assertGreater(float(np.max(np.abs(b - before))), 1e-5)

    def test_future_responses_do_not_affect_generation(self):
        feed, _, bins = self.model._draw_feed(self.state, self.u[:, 3:], streams(), 4)
        first = self.model._read_future(feed, 4, bins)
        m = self.model._full.model
        changed = dict(feed)
        changed[m.dataset_ph] = np.full(
            (4 * self.state.initial_mean.shape[0], self.model.history_bins + self.model.horizon_bins,
             len(self.model.unit_ids)), 999, dtype=np.int32)
        second = self.model._read_future(changed, 4, bins)
        for name in first: np.testing.assert_array_equal(first[name], second[name])
        # The supplied state contains prefix encodings, not a padded response.
        state2 = self.model.infer_state({'counts': self.y[:, :3], 'stimulus': self.u[:, :3]})
        for name in ('initial_mean', 'initial_log_variance', 'controller_encoding_forward', 'controller_encoding_reverse'):
            np.testing.assert_array_equal(getattr(self.state, name), getattr(state2, name))

    def test_extension_length_and_short_requested_horizon(self):
        before = self.model.simulate(self.state, self.u[:, 3:5], streams(), 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint'; self.model.save(checkpoint)
            with LFADSConditionalSimulator.load(checkpoint, horizon_bins=5) as longer:
                self.assertEqual(longer.training_horizon_bins, 3)
                self.assertEqual(longer.horizon_bins, 5)
                state = longer.infer_state({'counts': self.y[:, :3], 'stimulus': self.u[:, :3]})
                after = longer.simulate(state, self.u[:, 3:5], streams(), 4)
                for name in ('initial_mean', 'initial_log_variance', 'controller_encoding_forward', 'controller_encoding_reverse'):
                    np.testing.assert_array_equal(getattr(self.state, name), getattr(state, name))
                for name in ('counts', 'conditional_expected_counts', 'historical_controller_inputs', 'future_controller_inputs', 'boundary_generator_states'):
                    np.testing.assert_array_equal(getattr(before, name), getattr(after, name))


    def test_changed_history_matches_source_prefix_and_samples_without_future_response(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint'
            self.model.save(checkpoint)
            with LFADSConditionalSimulator.load(checkpoint, history_bins=5, horizon_bins=2) as model:
                history = {'counts': self.y[:, :5], 'stimulus': self.u[:, :5]}
                state = model.infer_state(history)
                self.assertEqual(state.controller_encoding_forward.shape[1], 5)
                np.testing.assert_array_equal(state.known_input_history, self.u[:, :5])
                native, extended = model._prefix.model, model._full.model
                initial = state.initial_mean + np.full_like(state.initial_mean, .2)
                prefix_feed = native.build_feed_dict('recording', self.y[:, :5], self.u[:, :5], keep_prob=1.)
                prefix_feed[native.g0s_val] = initial
                extended_feed = model._history_feed(state, initial[None], self.u[:, 5:6])
                imposed = np.linspace(-.3, .4, 2 * 5 * model.model_config['co_dim'], dtype=np.float32).reshape(2, 5, -1)
                for tick in range(5):
                    source_posterior = model._prefix.session.run([native.posterior_zs_co[tick].mean,
                        native.posterior_zs_co[tick].logvar], feed_dict=prefix_feed)
                    adapted_posterior = model._full.session.run([extended.posterior_zs_co[tick].mean,
                        extended.posterior_zs_co[tick].logvar], feed_dict=extended_feed)
                    for left, right in zip(source_posterior, adapted_posterior):
                        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
                    prefix_feed[native.controller_outputs[tick]] = imposed[:, tick]
                    extended_feed[extended.controller_outputs[tick]] = imposed[:, tick]
                    for field in ('gen_states', 'con_states', 'output_dist_params'):
                        left = model._prefix.session.run(getattr(native, field)[tick], feed_dict=prefix_feed)
                        right = model._full.session.run(getattr(extended, field)[tick], feed_dict=extended_feed)
                        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
                altered = self.y[:, :5].copy()
                altered[:, -1] += 7
                other = model.infer_state({'counts': altered, 'stimulus': self.u[:, :5]})
                self.assertFalse(np.array_equal(state.initial_mean, other.initial_mean))
                first = model.simulate(state, self.u[:, 5:6], streams(), 4)
                second = model.simulate(state, self.u[:, 5:6], streams(), 4)
                self.assertEqual(first.counts.shape, (4, 2, 1, 2))
                self.assertEqual(first.historical_controller_inputs.shape[2], 5)
                for name in ('counts', 'conditional_expected_counts', 'historical_controller_inputs',
                             'future_controller_inputs', 'boundary_generator_states', 'future_generator_states'):
                    np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
                feed, _, bins = model._draw_feed(state, self.u[:, 5:6], streams(), 4)
                expected = model._read_future(feed, 4, bins)
                changed = dict(feed)
                # Generation no longer supplies a dummy response. Inject an
                # independent adversarial response to retain the causality test.
                changed[extended.dataset_ph] = np.full(
                    (4 * state.initial_mean.shape[0], model.history_bins + model.horizon_bins,
                     len(model.unit_ids)), 999, dtype=np.int32)
                actual = model._read_future(changed, 4, bins)
                for name in expected:
                    np.testing.assert_array_equal(expected[name], actual[name])
                with self.assertRaisesRegex(ValueError, 'COUNT_SEQUENCE'):
                    model.infer_state({'counts': self.y[:, :3], 'stimulus': self.u[:, :3]})

    def test_random_streams_and_no_global_numpy_consumption(self):
        before = np.random.get_state()
        first = self.model.simulate(self.state, self.u[:, 3:], streams(), 8)
        observation = self.model.simulate(self.state, self.u[:, 3:], streams(observation=54), 8)
        dynamics = self.model.simulate(self.state, self.u[:, 3:], streams(dynamics=55), 8)
        initial = self.model.simulate(self.state, self.u[:, 3:], streams(initial=56), 8)
        np.testing.assert_array_equal(first.conditional_expected_counts, observation.conditional_expected_counts)
        np.testing.assert_array_equal(first.future_generator_states, observation.future_generator_states)
        self.assertFalse(np.array_equal(first.counts, observation.counts))
        np.testing.assert_array_equal(first.boundary_generator_states, dynamics.boundary_generator_states)
        np.testing.assert_array_equal(first.historical_controller_inputs, dynamics.historical_controller_inputs)
        self.assertFalse(np.array_equal(first.future_controller_inputs, dynamics.future_controller_inputs))
        self.assertFalse(np.array_equal(first.boundary_generator_states, initial.boundary_generator_states))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_conditional_ar_innovations_and_observation_units(self):
        state = replace(self.state, **{name: getattr(self.state, name)[:1] for name in
            ('initial_mean', 'initial_log_variance', 'controller_encoding_forward',
             'controller_encoding_reverse', 'known_input_history')})
        draw = self.model.simulate(state, self.u[:1, 3:5], streams(), 12000)
        m = self.model._full.model
        tau, log_variance = self.model._full.session.run(
            [m.prior_zs_ar_con.logataus_1xu, m.prior_zs_ar_con.logevars_1xu])
        phi = np.exp(-np.exp(-tau))[0]
        variance = np.exp(log_variance)[0]
        e0 = draw.future_controller_inputs[:, 0, 0] - phi * draw.historical_controller_inputs[:, 0, -1]
        e1 = draw.future_controller_inputs[:, 0, 1] - phi * draw.future_controller_inputs[:, 0, 0]
        for innovation in (e0, e1):
            self.assertTrue(np.all(np.abs(innovation.mean(0)) < 6 * np.sqrt(variance / len(innovation))))
            self.assertTrue(np.all(np.abs(innovation.var(0) - variance) < 6 * variance * np.sqrt(2 / (len(innovation) - 1))))
        self.assertLess(abs(np.corrcoef(e0[:, 0], e1[:, 0])[0, 1]), 6 / np.sqrt(len(e0)))
        # Poisson residuals have conditional variance equal to the source mean.
        residual = draw.counts - draw.conditional_expected_counts
        z = residual.sum() / np.sqrt(draw.conditional_expected_counts.sum())
        self.assertLess(abs(z), 6)
        # Multiplying the source mean by bin_seconds would fail this check.
        self.assertGreater(draw.counts.mean(), 5 * self.model.bin_seconds * draw.conditional_expected_counts.mean())

    def test_control_gradient_with_fixed_random_samples(self):
        future = self.u[:, 3:5].copy()
        weights = np.ones((2, 2, 2), dtype=np.float32)
        gradient = self.model.expected_count_gradient(self.state, future, weights, streams(), 16)
        epsilon = .002
        numerical = np.empty_like(future)
        for index in np.ndindex(future.shape):
            up, down = future.copy(), future.copy()
            up[index] += epsilon; down[index] -= epsilon
            a = self.model.simulate(self.state, up, streams(), 16).conditional_expected_counts
            b = self.model.simulate(self.state, down, streams(), 16).conditional_expected_counts
            numerical[index] = ((a.astype(np.float64) * weights).sum() - (b.astype(np.float64) * weights).sum()) / (2 * epsilon * 16)
        np.testing.assert_allclose(gradient, numerical, atol=2e-4, rtol=.005)
        self.assertGreater(np.max(np.abs(gradient)), 1e-4)

    def test_checkpoint_replay_and_explicit_resume_boundary(self):
        first = self.model.simulate(self.state, self.u[:, 3:], streams(), 4)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'checkpoint'; self.model.save(p)
            # A temporary v1-format fixture retains the historical inference-only
            # contract. Newly saved v2 checkpoints have separate training tests.
            record = json.loads((p / 'model.json').read_text())
            record['schema'] = 'neural_population_lfads_conditional_v1'
            record['training_rng_resume_supported'] = False
            record['checkpoint_purpose'] = 'parameter_and_inference_replay'
            for field in ('training_rng_schema', 'training_updates', 'training_random_sites', 'training_history_bins'):
                record.pop(field)
            (p / 'model.json').write_text(json.dumps(record))
            with LFADSConditionalSimulator.load(p) as restored:
                self.assertEqual((restored.training_history_bins, restored.training_horizon_bins), (3, 3))
                state = restored.infer_state({'counts': self.y[:, :3], 'stimulus': self.u[:, :3]})
                second = restored.simulate(state, self.u[:, 3:], streams(), 4)
                for name in ('counts', 'conditional_expected_counts', 'historical_controller_inputs',
                             'future_controller_inputs', 'boundary_generator_states', 'future_generator_states'):
                    np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
                with self.assertRaisesRegex(ValueError, 'TRAINING_RNG_RESUME'):
                    restored.fit_step(self.y, self.u)
            with self.assertRaises(FileExistsError): self.model.save(p)

    def test_short_history_causal_controller_and_invalid_inputs(self):
        config = {**self.spec, 'do_causal_controller': True, 'do_feed_factors_to_controller': False}
        with LFADSConditionalSimulator([12, 4], 1, 1, 1, .05, config, initialization_seed=42) as model:
            state = model.infer_state({'counts': self.y[:, :1], 'stimulus': self.u[:, :1]})
            self.assertIsNone(state.controller_encoding_reverse)
            self.assertEqual(model.simulate(state, self.u[:, 1:2], streams(), 2).counts.shape, (2, 2, 1, 2))
        with self.assertRaisesRegex(ValueError, 'COUNT_SEQUENCE'):
            self.model.infer_state({'counts': self.y[:, :4], 'stimulus': self.u[:, :4]})
        with self.assertRaisesRegex(ValueError, 'BIN_WIDTH'):
            self.model.simulate(replace(self.state, bin_seconds=.1), self.u[:, 3:], streams(), 2)
        with self.assertRaisesRegex(ValueError, 'REAL_NUMERIC'):
            self.model.simulate(self.state, self.u[:, 3:].astype(complex), streams(), 2)
        with self.assertRaisesRegex(ValueError, 'DISTINCT'): LFADSRandomStreams.from_seeds(1, 1, 2)
        saved = self.model._full.values()
        degenerate = {key: value.copy() for key, value in saved.items()}
        degenerate['LFADS/con/con_to_post_co/logvar/b:0'].fill(-1000.)
        try:
            self.model._full.restore(degenerate)
            with self.assertRaisesRegex(FloatingPointError, 'HISTORICAL_CONTROLLER_POSTERIOR'):
                self.model.simulate(self.state, self.u[:, 3:], streams(), 2)
        finally:
            self.model._full.restore(saved)
        feed, _, bins = self.model._draw_feed(self.state, self.u[:, 3:], streams(), 2)
        feed[self.model._full.model.unclipped_log_rates[3]] = np.full((4, 2), 81., dtype=np.float32)
        with self.assertRaisesRegex(FloatingPointError, 'LOG_RATE_CLIP_ACTIVE'):
            self.model._read_future(feed, 2, bins)


class LFADSCommandTests(unittest.TestCase):
    def test_canonical_cli_manifest_and_nonzero_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); write_windows(root / 'input')
            config = yaml.safe_load(CONFIG.read_text()); config['lfads_baseline_check']['optimizer_steps'] = 2
            (root / 'config.yaml').write_text(yaml.safe_dump(config))
            args = ['--config', str(root / 'config.yaml'), 'baseline-check', '--recording-dir', str(root / 'input'),
                    '--output-dir', str(root / 'result'), '--seed', '41', '--model', 'lfads']
            self.assertEqual(parser().parse_args(args).model, 'lfads')
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
            result = json.loads(output.getvalue())
            self.assertFalse(result['formal_seed_complete'])
            self.assertFalse(result['generative_fidelity_qualified'])
            self.assertEqual(result['sample_shape'], [8, 2, 3, 2])
            manifest = json.loads((root / 'result/manifest.json').read_text())
            for name, digest in manifest['artifacts'].items(): self.assertEqual(sha256_file(root / 'result' / name), digest)
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(args), 1)
            self.assertEqual(json.loads(errors.getvalue())['error_type'], 'FileExistsError')


if __name__ == '__main__':
    unittest.main()

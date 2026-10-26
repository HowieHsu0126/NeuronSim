"""Engineering checks for native BRAID timing, gradients and resumable mean prediction.

These small fixtures qualify adapter behavior only, not generative fidelity or a
neuroscientific finding. BRAID here predicts deterministic mean counts per bin.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from .braid import BRAIDMeanPredictor, SOURCE_REVISION


SETTINGS = {
    'latent_dim': 3,
    'hidden_units': 5,
    'steps_ahead': [1, 2, 4],
    'learning_rate': .001,
    'gradient_clip_norm': 1.,
}


def make_model(seed=41):
    return BRAIDMeanPredictor((1, 2), 2, 5, 6, .05, SETTINGS,
                              initialization_seed=seed)


def numpy_regression(regression, values):
    """Independent float64 Dense/tanh/exponential arithmetic for this configuration.

    This neural-only configuration has no stage-one prior; C evaluates its Dense
    path directly, equivalent to multiplying its mean by a neutral prior of one.
    No native TensorFlow forward call contributes to this reference calculation.
    """
    result = np.asarray(values, dtype=np.float64)
    dense_layers = [layer for layer in regression.model.layers
                    if isinstance(layer, tf.keras.layers.Dense)]
    if len(dense_layers) != 2:
        raise AssertionError('The configured one-hidden-layer source model changed')
    for layer in dense_layers:
        weights = layer.get_weights()
        if len(weights) != 2:
            raise AssertionError('Reference expects a Dense kernel and bias')
        kernel, bias = (np.asarray(value, dtype=np.float64) for value in weights)
        result = result @ kernel + bias
        activation = layer.activation.__name__
        if activation == 'tanh':
            result = np.tanh(result)
        elif activation == 'exponential':
            result = np.exp(result)
        elif activation != 'linear':
            raise AssertionError('Unaccounted source activation: ' + activation)
    return result


class BRAIDAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model()
        trial = np.arange(3)[:, None, None]
        time = np.arange(11)[None, :, None]
        unit = np.arange(2)[None, None, :]
        cls.y = ((2 * trial + time + unit) % 5).astype(np.int64)
        cls.u = np.linspace(-.4, .6, 66, dtype=np.float32).reshape(3, 11, 2)
        cls.metadata = {'unit_ids': (1, 2), 'bin_seconds': .05}
        cls.state = cls.model.infer_state(
            {'counts': cls.y[:, :5], 'stimulus': cls.u[:, :5]}, cls.metadata)

    def test_training_tensors_keep_every_trial_and_exact_native_time_offsets(self):
        model = self.model
        inputs, labels = model.training_tensors(self.y, self.u)
        origins = 11 - 4
        self.assertEqual(model.block_samples, origins)
        self.assertEqual(len(inputs), 3)
        self.assertEqual(len(labels), 9)
        np.testing.assert_array_equal(inputs[0][..., :2], self.y[:, :origins])
        np.testing.assert_array_equal(inputs[0][..., 2:], self.u[:, :origins])
        for head, ahead in enumerate((1, 2, 4)):
            # Distinct trial rows are retained even though they share time indices.
            np.testing.assert_array_equal(labels[head], self.y[:, ahead:ahead + origins])
            np.testing.assert_array_equal(inputs[1][..., head],
                                          self.u[:, ahead:ahead + origins])
            self.assertEqual(tuple(labels[head].shape), (3, origins, 2))
        for index, ahead in enumerate((1, 2, 3)):
            np.testing.assert_array_equal(inputs[2][..., index],
                                          self.u[:, ahead:ahead + origins])
        np.testing.assert_array_equal(inputs[2][..., -1], np.zeros((3, origins, 2)))
        for target in labels[3:]:
            np.testing.assert_array_equal(target, np.zeros((3, origins, 3)))

    def test_variable_observed_prefix_matches_source_recursion_and_future_is_excluded(self):
        cell = self.model.cell
        for bins in (1, 5, 9):
            with self.subTest(prefix_bins=bins):
                state = self.model.infer_prefix(
                    {'counts': self.y[:, :bins], 'stimulus': self.u[:, :bins]}, self.metadata)
                expected = np.broadcast_to(np.asarray(cell.initial_state.numpy(), dtype=np.float64),
                                           state.boundary.shape).copy()
                for index in range(bins):
                    observed = np.concatenate((self.y[:, index], self.u[:, index]), axis=-1)
                    expected = numpy_regression(cell.A, expected) + numpy_regression(cell.K, observed)
                np.testing.assert_allclose(state.boundary, expected, atol=2e-7, rtol=2e-6)
        with self.assertRaisesRegex(ValueError, 'NONEMPTY_OBSERVED_PREFIX'):
            self.model.infer_prefix({'counts': self.y[:, :0], 'stimulus': self.u[:, :0]}, self.metadata)
        with self.assertRaisesRegex(ValueError, 'EXPECTED_SHAPE'):
            self.model.infer_state({'counts': self.y[:, :9], 'stimulus': self.u[:, :9]}, self.metadata)

    def test_unused_forward_input_padding_cannot_change_native_outputs(self):
        inputs, _ = self.model.training_tensors(self.y, self.u)
        expected = self.model.native.model(inputs, training=False)
        changed = list(inputs)
        padded = changed[2].numpy().copy()
        padded[..., -1] = 12345.
        changed[2] = tf.convert_to_tensor(padded)
        actual = self.model.native.model(tuple(changed), training=False)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_forecast_agrees_with_native_multihorizon_heads_at_history_boundary(self):
        inputs, _ = self.model.training_tensors(self.y, self.u)
        native = self.model.native.model(inputs, training=False)
        means = self.model.forecast(self.state, self.u[:, 5:]).numpy()
        self.assertEqual(means.shape, (3, 6, 2))
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue((means > 0).all())
        for head, ahead in enumerate((1, 2, 4)):
            # Origin 4 has seen bins 0..4, so ahead=1 predicts response bin 5.
            np.testing.assert_allclose(native[head][:, 4], means[:, ahead - 1],
                                       atol=2e-7, rtol=2e-6)
        np.testing.assert_array_equal(
            self.model.forecast(self.state, self.u[:, 5:7]), means[:, :2])

    def test_history_and_free_future_match_independent_numpy_recursions(self):
        cell = self.model.cell
        state = np.broadcast_to(np.asarray(cell.initial_state.numpy(), dtype=np.float64),
                                (3, SETTINGS['latent_dim'])).copy()
        for index in range(5):
            observed = np.concatenate((self.y[:, index], self.u[:, index]), axis=-1)
            state = numpy_regression(cell.A, state) + numpy_regression(cell.K, observed)
        np.testing.assert_allclose(self.state.boundary, state, atol=2e-7, rtol=2e-6)
        expected = []
        for index in range(5, 11):
            expected.append(numpy_regression(cell.C, np.concatenate((state, self.u[:, index]), axis=-1)))
            if index < 10:
                state = numpy_regression(cell.Afw, state) + numpy_regression(cell.Kfw, self.u[:, index])
        expected = np.stack(expected, axis=1)
        # C's exponential outputs counts/bin. No bin_seconds multiplier belongs here.
        np.testing.assert_allclose(self.model.forecast(self.state, self.u[:, 5:]), expected,
                                   atol=2e-7, rtol=2e-6)

    def test_future_neural_responses_do_not_leak_into_native_boundary_or_forecast(self):
        original_inputs, _ = self.model.training_tensors(self.y, self.u)
        native_before = self.model.native.model(original_inputs, training=False)
        expected = self.model.forecast(self.state, self.u[:, 5:]).numpy()
        changed = self.y.copy()
        changed[:, 5:] += 1000
        changed_inputs, _ = self.model.training_tensors(changed, self.u)
        native_after = self.model.native.model(changed_inputs, training=False)
        state = self.model.infer_state(
            {'counts': changed[:, :5], 'stimulus': self.u[:, :5]}, self.metadata)
        np.testing.assert_array_equal(state.boundary, self.state.boundary)
        np.testing.assert_array_equal(self.model.forecast(state, self.u[:, 5:]), expected)
        for head in range(3):
            np.testing.assert_array_equal(native_before[head][:, :5], native_after[head][:, :5])

    def test_trial_permutation_and_unrelated_trial_history_are_independent(self):
        order = np.array([2, 0, 1])
        state = self.model.infer_state(
            {'counts': self.y[order, :5], 'stimulus': self.u[order, :5]}, self.metadata)
        expected = self.model.forecast(self.state, self.u[:, 5:]).numpy()
        np.testing.assert_allclose(state.boundary, self.state.boundary.numpy()[order],
                                   atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(self.model.forecast(state, self.u[order, 5:]),
                                   expected[order], atol=1e-7, rtol=1e-6)
        changed = self.y[:, :5].copy()
        changed[2] += 30
        altered = self.model.infer_state(
            {'counts': changed, 'stimulus': self.u[:, :5]}, self.metadata)
        actual = self.model.forecast(altered, self.u[:, 5:]).numpy()
        np.testing.assert_array_equal(actual[:2], expected[:2])
        self.assertGreater(float(np.max(np.abs(altered.boundary[2] - self.state.boundary[2]))), 1e-7)

    def test_nonlinear_stimulus_gradient_matches_finite_differences(self):
        # A weighted objective exercises both instantaneous readout and later Afw/Kfw propagation.
        future = self.u[:1, 5:].copy()
        state = replace(self.state, boundary=self.state.boundary[:1])
        weights = np.linspace(.2, 1.1, 12, dtype=np.float32).reshape(1, 6, 2)
        gradient = self.model.expected_count_gradient(state, future, weights).numpy()
        epsilon = .002
        numerical = np.empty_like(future)
        for index in np.ndindex(future.shape):
            upper, lower = future.copy(), future.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            a = self.model.forecast(state, upper).numpy().astype(np.float64)
            b = self.model.forecast(state, lower).numpy().astype(np.float64)
            numerical[index] = np.sum((a - b) * weights) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, atol=1e-4, rtol=.01)
        self.assertGreater(float(np.max(np.abs(gradient))), 1e-6)
        # A target in the first future bin cannot depend on later stimulus values.
        first_bin = np.zeros_like(weights)
        first_bin[:, 0] = 1.
        causal = self.model.expected_count_gradient(state, future, first_bin).numpy()
        np.testing.assert_array_equal(causal[:, 1:], np.zeros_like(causal[:, 1:]))
        with self.assertRaisesRegex(ValueError, 'OBJECTIVE_WEIGHTS_SHAPE'):
            self.model.expected_count_gradient(state, future, weights[:, :-1])

    def test_native_training_loss_reaches_observed_forward_and_initial_state_parameters(self):
        model = self.model
        inputs, labels = model.training_tensors(self.y, self.u)
        variables = model.native.model.trainable_variables
        with tf.GradientTape() as tape:
            outputs = model.native.model(inputs, training=True)
            # Same mean-count Poisson negative log likelihood, excluding target-only
            # log-factorial constants and non-loss native state output heads.
            terms = [tf.reduce_mean(mean - target * tf.math.log(mean))
                     for mean, target in zip(outputs[:3], labels[:3])]
            loss = tf.add_n(terms)
        gradients = tape.gradient(loss, variables)
        by_identity = {id(variable): gradient for variable, gradient in zip(variables, gradients)}
        families = {name: list(getattr(model.cell, name).model.trainable_variables)
                    for name in ('A', 'K', 'C', 'Afw', 'Kfw')}
        families['initial_state'] = [model.cell.initial_state]
        covered = set()
        for family, members in families.items():
            with self.subTest(family=family):
                self.assertTrue(members)
                values = []
                for variable in members:
                    self.assertIn(id(variable), by_identity, variable.name)
                    gradient = by_identity[id(variable)]
                    self.assertIsNotNone(gradient, variable.name)
                    value = np.asarray(gradient)
                    self.assertTrue(np.isfinite(value).all(), variable.name)
                    values.append(value)
                    covered.add(id(variable))
                self.assertGreater(sum(float(np.sum(value ** 2)) for value in values), 0.)
        self.assertEqual(covered, {id(variable) for variable in variables})

    def test_finite_optimizer_update_and_checkpoint_exact_training_replay(self):
        model = make_model(seed=43)
        before = [variable.numpy().copy() for variable in model.native.model.trainable_variables]
        loss = model.fit_step(self.y, self.u)
        self.assertTrue(loss)
        self.assertTrue(np.isfinite(list(loss.values())).all())
        self.assertEqual(model.training_steps, 1)
        self.assertTrue(any(not np.array_equal(left, right.numpy()) for left, right in
                            zip(before, model.native.model.trainable_variables)))
        self.assertEqual(int(model.native.model.optimizer.iterations.numpy()), 1)
        model.fit_step(self.y, self.u)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint'
            model.save(path)
            metadata = json.loads((path / 'model.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['source_revision'], SOURCE_REVISION)
            self.assertEqual(metadata['training_steps'], 2)
            self.assertIn('no joint sampler', metadata['observation'])
            restored = BRAIDMeanPredictor.load(path)
            self.assertEqual(restored.training_steps, 2)
            left_state = model.infer_state(
                {'counts': self.y[:, :5], 'stimulus': self.u[:, :5]}, self.metadata)
            right_state = restored.infer_state(
                {'counts': self.y[:, :5], 'stimulus': self.u[:, :5]}, self.metadata)
            np.testing.assert_array_equal(left_state.boundary, right_state.boundary)
            np.testing.assert_array_equal(model.forecast(left_state, self.u[:, 5:]),
                                          restored.forecast(right_state, self.u[:, 5:]))
            left_optimizer = model.native.model.optimizer.variables()
            right_optimizer = restored.native.model.optimizer.variables()
            self.assertGreater(len(left_optimizer), 1)
            self.assertEqual(len(left_optimizer), len(right_optimizer))
            for left, right in zip(left_optimizer, right_optimizer):
                np.testing.assert_array_equal(left, right)
            left_loss = model.fit_step(self.y, self.u)
            right_loss = restored.fit_step(self.y, self.u)
            self.assertEqual(left_loss.keys(), right_loss.keys())
            for key in left_loss:
                self.assertEqual(left_loss[key], right_loss[key], key)
            self.assertEqual(model.training_steps, restored.training_steps)
            for left, right in zip(model.native.model.weights, restored.native.model.weights):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(model.native.model.optimizer.variables(),
                                   restored.native.model.optimizer.variables()):
                np.testing.assert_array_equal(left, right)
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_identity_bin_width_shape_and_integer_count_validation(self):
        history = {'counts': self.y[:, :5], 'stimulus': self.u[:, :5]}
        for metadata in ({'unit_ids': (2, 1), 'bin_seconds': .05},
                         {'unit_ids': (1, 2), 'bin_seconds': .1}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'IDENTITY_OR_UNITS'):
                self.model.infer_state(history, metadata)
        for state in (replace(self.state, unit_ids=(2, 1)),
                      replace(self.state, bin_seconds=.1)):
            with self.assertRaisesRegex(ValueError, 'IDENTITY_OR_UNITS'):
                self.model.forecast(state, self.u[:, 5:])
        for bad_value in (-1., .5, np.nan, np.inf):
            counts = self.y.astype(np.float64)
            counts[0, 0, 0] = bad_value
            with self.subTest(count=bad_value), self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
                self.model.training_tensors(counts, self.u)
        with self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
            self.model.training_tensors(self.y.astype(np.complex128), self.u)
        with self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
            self.model.training_tensors(self.y[:, :-1], self.u)
        with self.assertRaisesRegex(ValueError, 'INPUT_SHAPE'):
            self.model.infer_state({**history, 'stimulus': self.u[:, :4]}, self.metadata)
        with self.assertRaisesRegex(ValueError, 'REAL_INPUT'):
            self.model.forecast(self.state, self.u[:, 5:].astype(complex))
        with self.assertRaisesRegex(ValueError, 'FUTURE_INPUT_SEQUENCE'):
            self.model.forecast(self.state, self.u[:, :0])
        with self.assertRaisesRegex(ValueError, 'STATE_SHAPE'):
            self.model.forecast(replace(self.state, boundary=tf.zeros((3, 4))), self.u[:, 5:])
        with self.assertRaisesRegex(ValueError, 'UNIT_IDENTITIES'):
            BRAIDMeanPredictor((1, 1), 2, 5, 6, .05, SETTINGS, initialization_seed=41)


if __name__ == '__main__':
    unittest.main()

"""Independent algebra, causal forecasting and source-boundary checks."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from .__main__ import main, parser
from .baseline_data import load_engineering_windows
from .config import load_config
from .count_autoregression import check_count_baseline
from .provenance import sha256_file
from .reduced_rank import ReducedRankAutoregression, check_reduced_rank


CONFIG = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


def fixture_data():
    rng = np.random.default_rng(172)
    y = rng.integers(0, 8, size=(3, 8, 3)).astype(np.float64)
    u = rng.normal(size=(3, 8, 2)) * np.array([.02, 40.])
    return y, u


def coefficient(model):
    return model.input_factor.cpu().numpy() @ model.output_factor.cpu().numpy().T


def write_windows(directory, *, inputs=True, split="train"):
    directory.mkdir()
    rng = np.random.default_rng(69)
    counts = rng.integers(0, 5, size=(2, 7, 2))
    summary = {"scope": "engineering_only", "split": split, "session_id": 847657808}
    (directory / "summary.json").write_text(json.dumps(summary))
    with h5py.File(directory / "spike_windows.h5", "w") as handle:
        handle["counts"] = counts
        handle["unit_ids"] = [12, 4]
        handle.attrs["history_bin_count"] = 4
        handle["relative_edges_seconds"] = np.arange(8) * .1 - .4
        handle.create_dataset("trial_ids", data=["trial_a", "trial_b"], dtype=h5py.string_dtype())
        if inputs:
            handle["stimulus_features"] = np.arange(14).reshape(2, 7, 1) / 14.
    names = ["summary.json", "spike_windows.h5"]
    if inputs:
        (directory / "stimulus_features.json").write_text(json.dumps({"features": [{"feature_index": 0}]}))
        names.append("stimulus_features.json")
    manifest = {"window_specification": {"bin_seconds": .1},
                "artifacts": {name: sha256_file(directory / name) for name in names}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return counts


def refresh_digest(directory, name):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][name] = sha256_file(directory / name)
    path.write_text(json.dumps(manifest))


class ReducedRankTests(unittest.TestCase):
    def test_augmented_solver_and_objective_certificate(self):
        y, u = fixture_data()
        for rank in (0, 1, None):
            for ridge in (0., .7):
                for standardize in (False, True):
                    with self.subTest(rank=rank, ridge=ridge, standardize=standardize):
                        model = ReducedRankAutoregression([1, 7, 20], 2, 2, .05, rank)
                        fit = model.fit_sequence(y, u, ridge_weight=ridge, standardize=standardize)
                        xt, yt = model.causal_rows(y, u)
                        x, target = xt.numpy(), yt.numpy()
                        xc, yc = x - x.mean(0), target - target.mean(0)
                        scale = xc.std(0) if standardize else np.ones(x.shape[1])
                        scale[scale == 0] = 1.
                        xs = xc / scale
                        a = np.vstack((xs, np.sqrt(ridge) * np.eye(x.shape[1])))
                        b = np.vstack((yc, np.zeros((x.shape[1], target.shape[1]))))
                        w0 = np.linalg.lstsq(a, b, rcond=fit['rcond'])[0]
                        _, _, vt = np.linalg.svd(a @ w0, full_matrices=False)
                        width = target.shape[1] if rank is None else rank
                        projection = vt[:width].T @ vt[:width]
                        expected = (w0 @ projection) / scale[:, None]
                        np.testing.assert_allclose(coefficient(model), expected, atol=2e-8, rtol=2e-8)
                        np.testing.assert_allclose(model.bias.numpy(), target.mean(0) - x.mean(0) @ expected, atol=2e-8)
                        self.assertLessEqual(np.linalg.matrix_rank(coefficient(model), tol=1e-8), width)
                        self.assertAlmostEqual(fit['objective'], fit['spectral_optimum_objective'], delta=1e-8 * max(1., fit['objective']))
                        if model.output_factor.shape[1]:
                            q = model.output_factor.numpy()
                            np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12)

    def test_rank_deficient_minimum_norm_and_free_rollout(self):
        y = np.array([[[1., 1., 0.], [2., 2., 0.], [3., 3., 0.], [4., 4., 0.]]])
        u = np.zeros((1, 4, 2))
        model = ReducedRankAutoregression([8, 3, 9], 2, 1, .1)
        fit = model.fit_sequence(y, u, ridge_weight=0., standardize=False)
        x, target = model.causal_rows(y, u)
        xc, yc = x.numpy() - x.numpy().mean(0), target.numpy() - target.numpy().mean(0)
        expected = np.linalg.lstsq(xc, yc, rcond=fit['rcond'])[0]
        np.testing.assert_allclose(coefficient(model), expected, atol=1e-12)
        self.assertEqual(fit['design_rank_at_cutoff'], 1)
        self.assertEqual(fit['constant_feature_count'], 3)
        result = model.forecast(model.infer_state(y[:, -1:]), np.zeros((2, 2))).values
        np.testing.assert_allclose(result.numpy(), [[[5., 5., 0.], [6., 6., 0.]]], atol=1e-12)

    def test_intercept_translation_and_constant_responses(self):
        y, u = fixture_data()
        shift_y, shift_u = np.array([3., 7., 1.]), np.array([2., -4.])
        first = ReducedRankAutoregression([1, 2, 3], 2, 2, .1, 1)
        shifted = ReducedRankAutoregression([1, 2, 3], 2, 2, .1, 1)
        first.fit_sequence(y, u, ridge_weight=2.)
        shifted.fit_sequence(y + shift_y, u + shift_u, ridge_weight=2.)
        np.testing.assert_allclose(coefficient(first), coefficient(shifted), atol=1e-10)
        dx = np.concatenate((shift_y, shift_y, shift_u))
        np.testing.assert_allclose(shifted.bias.numpy(), first.bias.numpy() + shift_y - dx @ coefficient(first), atol=1e-10)
        constant = ReducedRankAutoregression([1, 2, 3], 2, 2, .1)
        fit = constant.fit_sequence(np.ones_like(y) * 4., u, ridge_weight=1.)
        self.assertEqual(fit['residual_sse'], 0.)
        np.testing.assert_array_equal(constant.bias.numpy(), [4., 4., 4.])

    def test_causal_rows_have_current_input_and_no_cross_trial_lags(self):
        model = ReducedRankAutoregression([12, 4], 1, 2, .5)
        y = np.stack((np.arange(8).reshape(4, 2), np.arange(8).reshape(4, 2) + 100))
        u = np.arange(8).reshape(2, 4, 1) + 20
        x, target = model.causal_rows(y, u)
        expected_x, expected_y = [], []
        for trial in range(2):
            for step in range(2, 4):
                expected_x.append(np.r_[y[trial, step-2:step].ravel(), u[trial, step]])
                expected_y.append(y[trial, step])
        np.testing.assert_array_equal(x.numpy(), expected_x)
        np.testing.assert_array_equal(target.numpy(), expected_y)

    def test_negative_predictions_are_unclipped_and_stimulus_gradient_is_causal(self):
        model = ReducedRankAutoregression([12, 4], 1, 1, .5)
        model.fit_sequence(np.zeros((1, 4, 2)), np.zeros((1, 4, 1)), ridge_weight=1.)
        model.input_factor = torch.tensor([[.5, 0.], [0., .25], [2., 1.]], dtype=torch.float64)
        model.output_factor = torch.eye(2, dtype=torch.float64)
        model.bias.copy_(torch.tensor([-3., -1.]))
        state = model.infer_state([[2., 4.]])
        stimulus = torch.tensor([[0.], [2.]], dtype=torch.float64, requires_grad=True)
        prediction = model.forecast(state, stimulus).values
        torch.testing.assert_close(prediction, torch.tensor([[[-2., 0.], [0., 1.]]], dtype=torch.float64))
        prediction.sum().backward()
        torch.testing.assert_close(stimulus.grad, torch.tensor([[4.25], [3.]], dtype=torch.float64))
        torch.testing.assert_close(state.counts, torch.tensor([[[2., 4.]]], dtype=torch.float64))

    def test_prefix_state_is_copied_and_future_responses_are_unused(self):
        y, u = fixture_data()
        model = ReducedRankAutoregression([1, 2, 3], 2, 2, .1, 1)
        model.fit_sequence(y, u, ridge_weight=1.)
        history = y[:, :3].copy()
        state = model.infer_state(history, {'unit_ids': [1, 2, 3], 'bin_seconds': .1})
        before = model.forecast(state, u[:, 3:]).values
        history[:] = 99
        y[:, 3:] = 10000
        torch.testing.assert_close(model.forecast(state, u[:, 3:]).values, before)
        self.assertFalse(torch.any(state.counts == 99))
        with self.assertRaisesRegex(ValueError, 'UNIT_ORDER'):
            model.infer_state(history, {'unit_ids': [3, 2, 1]})
        with self.assertRaisesRegex(ValueError, 'BIN_WIDTH'):
            model.infer_state(history, {'bin_seconds': .2})

    def test_fit_and_checkpoint_replay_preserve_rng_and_dtype(self):
        y, u = fixture_data()
        before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        model = ReducedRankAutoregression([1, 2, 3], 2, 2, .1, 1)
        model.fit_sequence(y, u, ridge_weight=1.)
        model.float()
        state = model.infer_state(y[:, :3])
        expected = model.forecast(state, u[:, 3:]).values
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            model.save(path)
            restored = ReducedRankAutoregression.load(path)
            self.assertEqual(restored.bias.dtype, torch.float32)
            self.assertTrue(torch.equal(restored.forecast(restored.infer_state(y[:, :3]), u[:, 3:]).values, expected))
            with self.assertRaises(FileExistsError):
                model.save(path)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_invalid_inputs_and_nonfinite_rollout_reject(self):
        y, u = fixture_data()
        model = ReducedRankAutoregression([1, 2, 3], 2, 2, .1, 1)
        with self.assertRaisesRegex(ValueError, 'NOT_FITTED'):
            model.forecast(model.infer_state(y[:, :3]), u[:, 3:])
        for ridge in (-1., float('nan'), True):
            with self.subTest(ridge=ridge), self.assertRaisesRegex(ValueError, 'RIDGE_WEIGHT'):
                model.fit_sequence(y, u, ridge_weight=ridge)
        for invalid in (-1., .5, float('nan')):
            bad = y.copy(); bad[0, 0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
                model.fit_sequence(bad, u, ridge_weight=1.)
        model.fit_sequence(y, u, ridge_weight=1.)
        with self.assertRaisesRegex(ValueError, 'STIMULUS_SEQUENCE'):
            model.forecast(model.infer_state(y[:, :3]), np.zeros((3, 2, 3)))
        model.input_factor.fill_(1e308)
        with self.assertRaisesRegex(FloatingPointError, 'NONFINITE_REGRESSION_FORECAST'):
            model.forecast(model.infer_state(y[:, :3]), u[:, 3:])


class BaselineWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_config(CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def test_input_and_no_input_loading_preserve_source_units_and_trials(self):
        for known in (False, True):
            directory = self.root / str(known)
            counts = write_windows(directory, inputs=known)
            data = load_engineering_windows(directory, self.config)
            np.testing.assert_array_equal(data.counts, counts)
            self.assertEqual(data.unit_ids, (12, 4))
            self.assertEqual(data.trial_ids, ('trial_a', 'trial_b'))
            self.assertEqual(data.inputs.shape, (2, 7, int(known)))
            self.assertEqual(data.history_bins, 4)
            self.assertEqual(data.bin_seconds, .1)

    def test_hash_and_protected_split_reject_before_output(self):
        directory = self.root / 'source'; write_windows(directory, split='test')
        with self.assertRaisesRegex(ValueError, 'ENGINEERING_TRAINING_ANCHOR'):
            check_reduced_rank(directory, self.root / 'output', self.config, ['test'], 41)
        self.assertFalse((self.root / 'output').exists())
        (directory / 'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'DIGEST_MISMATCH'):
            load_engineering_windows(directory, self.config)

    def test_source_alignment_errors_do_not_hide_behind_valid_hashes(self):
        for mutation in ('unit', 'trial', 'time', 'feature'):
            with self.subTest(mutation=mutation):
                directory = self.root / mutation; write_windows(directory)
                with h5py.File(directory / 'spike_windows.h5', 'r+') as handle:
                    if mutation == 'unit': handle['unit_ids'][:] = [12, 12]
                    if mutation == 'trial': handle['trial_ids'][:] = ['same', 'same']
                    if mutation == 'time': handle['relative_edges_seconds'][3] += .001
                    if mutation == 'feature': handle['stimulus_features'][0, 0, 0] = np.nan
                refresh_digest(directory, 'spike_windows.h5')
                with self.assertRaises(ValueError):
                    load_engineering_windows(directory, self.config)

    def test_engineering_checks_replay_both_baseline_consumers(self):
        directory = self.root / 'source'; write_windows(directory)
        self.config['count_baseline_check']['optimizer_steps'] = 2
        a, status_a = check_count_baseline(directory, self.root / 'count', self.config, ['test'], 41)
        b, status_b = check_reduced_rank(directory, self.root / 'rank', self.config, ['test'], 41)
        self.assertEqual((status_a, status_b), (2, 2))
        self.assertEqual(a['sample_shape'], [8, 2, 3, 2])
        self.assertEqual(b['forecast_shape'], [2, 3, 2])
        self.assertFalse(b['provides_joint_spike_samples'])
        self.assertFalse(b['formal_training_completed'])
        for folder in ('count', 'rank'):
            manifest = json.loads((self.root / folder / 'manifest.json').read_text())
            for name, digest in manifest['artifacts'].items():
                self.assertEqual(sha256_file(self.root / folder / name), digest)

    def test_cli_selector_and_failure_status(self):
        args = ['--config', str(CONFIG), 'baseline-check', '--recording-dir', str(self.root / 'absent'),
                '--output-dir', str(self.root / 'output'), '--seed', '41']
        self.assertEqual(parser().parse_args(args).model, 'poisson-autoregression')
        self.assertEqual(parser().parse_args(args + ['--model', 'reduced-rank']).model, 'reduced-rank')
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            status = main(args + ['--model', 'reduced-rank'])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(errors.getvalue())['status'], 'FAILED')
        self.assertFalse((self.root / 'output').exists())


if __name__ == '__main__':
    unittest.main()

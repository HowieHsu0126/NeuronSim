"""Causality, shared variability, control gradients and real-entrypoint contracts."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
import yaml

from .__main__ import main, parser
from .config import load_config
from .poisson_state_space import ControlledPoissonLDS, GaussianPopulationState, LatentRandomStreams
from .provenance import sha256_file
from .test_reduced_rank import write_windows


CONFIG = Path(__file__).resolve().parents[1] / 'configs/neural_population_dynamics.yaml'


def model_fixture(input_size=1):
    model = ControlledPoissonLDS([12, 4], input_size, 2, 4, .05, initialization_seed=41).double()
    with torch.no_grad():
        model.transition.copy_(torch.tensor([[.65, .25], [-.15, .5]]))
        model.loading.copy_(torch.tensor([[.5, .3], [.2, .65]]))
        model.log_rate_bias.fill_(np.log(45.))
        model.input_weight.fill_(.15)
        model.process_log_variance.copy_(torch.tensor([.08, .11]).log())
    state = GaussianPopulationState(torch.tensor([[.1, -.2]], dtype=torch.float64),
                                    torch.tensor([[.15, .3]], dtype=torch.float64), (12, 4), .05)
    return model, state


def rng(initial=201, dynamics=202, observation=203):
    return LatentRandomStreams.from_seeds(initial, dynamics, observation)


class StateSpaceTests(unittest.TestCase):
    def test_causal_encoder_history_inputs_and_forecast_alignment(self):
        model, _ = model_fixture()
        y = torch.arange(20).reshape(1, 10, 2).double() % 4
        u = torch.linspace(0, 1, 10).reshape(1, 10, 1).double()
        mean, variance = model.encode(y, u)
        changed = y.clone(); changed[:, 5:] += 11
        changed_u = u.clone(); changed_u[:, 5:] += 7
        m2, v2 = model.encode(changed, changed_u)
        self.assertTrue(torch.equal(mean[:, :5], m2[:, :5]))
        self.assertTrue(torch.equal(variance[:, :5], v2[:, :5]))
        state = model.infer_state({'counts': y[:, :5], 'stimulus': u[:, :5]})
        self.assertTrue(torch.equal(state.mean, mean[:, 4]))
        altered = model.infer_state({'counts': y[:, :5], 'stimulus': u[:, :5] + 4})
        self.assertFalse(torch.equal(state.mean, altered.mean))
        future = model.forecast_moments(state, u[:, 5:])
        expected_first = state.mean @ model.transition.T + u[:, 5] @ model.input_weight.T + model.transition_bias
        torch.testing.assert_close(future.mean[:, 0], expected_first, atol=0, rtol=0)

    def test_full_covariance_and_input_gradients(self):
        model, state = model_fixture()
        u = torch.tensor([[[.2], [-.3], [.1]]], dtype=torch.float64, requires_grad=True)
        result = model.forecast_moments(state, u)
        a, c = model.transition.detach().numpy(), model.loading.detach().numpy()
        m, p = state.mean.numpy()[0], np.diag(state.variance.numpy()[0])
        q = np.diag(model.process_log_variance.detach().exp().numpy())
        for t in range(3):
            m = a @ m + model.input_weight.detach().numpy() @ u.detach().numpy()[0, t] + model.transition_bias.detach().numpy()
            p = a @ p @ a.T + q
            expected = .05 * np.exp(c @ m + model.log_rate_bias.detach().numpy() + .5 * np.diag(c @ p @ c.T))
            np.testing.assert_allclose(result.covariance[0, t].detach().numpy(), p, atol=1e-14)
            np.testing.assert_allclose(result.expected_counts[0, t].detach().numpy(), expected, atol=1e-14)
        self.assertGreater(abs(p[0, 1]), 1e-3)
        result.expected_counts.sum().backward()
        epsilon = 1e-6
        for t in range(3):
            up = u.detach().clone(); down = up.clone()
            up[0, t, 0] += epsilon; down[0, t, 0] -= epsilon
            derivative = (model.forecast_mean(state, up).sum() - model.forecast_mean(state, down).sum()) / (2 * epsilon)
            self.assertAlmostEqual(u.grad[0, t, 0].item(), derivative.item(), places=7)

    def test_shared_count_moments_match_ancestral_samples(self):
        model, state = model_fixture()
        stimulus = torch.zeros(1, 2, 1, dtype=torch.float64)
        moments = model.forecast_moments(state, stimulus)
        draws = model.simulate(state, stimulus, rng(), 60000)
        y = draws.counts[:, 0, -1].double().numpy()
        expected = moments.expected_counts[0, -1].detach().numpy()
        p = moments.covariance[0, -1].detach().numpy()
        c = model.loading.detach().numpy()
        covariance = np.diag(expected) + expected[:, None] * expected[None, :] * np.expm1(c @ p @ c.T)
        # Monte Carlo tolerances tied to the standard error, not model performance.
        mean_error = np.abs(y.mean(0) - expected)
        self.assertTrue(np.all(mean_error < 6 * np.sqrt(np.diag(covariance) / len(y))))
        products = (y - expected)[:, :, None] * (y - expected)[:, None, :]
        covariance_se = products.std(0) / np.sqrt(len(y))
        self.assertTrue(np.all(np.abs(products.mean(0) - covariance) < 6 * covariance_se))
        self.assertGreater(covariance[0, 1], .1)
        np.testing.assert_allclose(draws.latent_states[:, 0, -1].numpy().mean(0),
                                   moments.mean[0, -1].detach().numpy(), atol=.01)
        np.testing.assert_allclose(np.cov(draws.latent_states[:, 0, -1].numpy(), rowvar=False), p, atol=.005)

    def test_random_stream_isolation_and_serialization(self):
        global_state = torch.get_rng_state().clone()
        model, state = model_fixture()
        u = torch.zeros(1, 3, 1)
        first = model.simulate(state, u, rng(), 64)
        observation = model.simulate(state, u, rng(observation=204), 64)
        self.assertTrue(torch.equal(first.latent_states, observation.latent_states))
        self.assertTrue(torch.equal(first.conditional_expected_counts, observation.conditional_expected_counts))
        self.assertFalse(torch.equal(first.counts, observation.counts))
        dynamics = model.simulate(state, u, rng(dynamics=205), 64)
        self.assertTrue(torch.equal(first.initial_latent_states, dynamics.initial_latent_states))
        self.assertFalse(torch.equal(first.latent_states, dynamics.latent_states))
        initial = model.simulate(state, u, rng(initial=206), 64)
        self.assertFalse(torch.equal(first.initial_latent_states, initial.initial_latent_states))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.pt'; model.save(path)
            restored = ControlledPoissonLDS.load(path)
            replay = restored.simulate(state, u, rng(), 64)
            for name in ('counts', 'conditional_expected_counts', 'latent_states', 'initial_latent_states'):
                self.assertTrue(torch.equal(getattr(first, name), getattr(replay, name)))
            self.assertEqual(restored.loading.dtype, torch.float64)
            with self.assertRaises(FileExistsError): model.save(path)
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))

    def test_training_loss_normalization_and_zero_input_special_case(self):
        model, _ = model_fixture(input_size=0)
        y = torch.tensor([[[0., 2.], [1., 3.], [2., 0.]]]).double()
        u = torch.empty(1, 3, 0, dtype=torch.float64)
        terms = model.sequence_loss(y, u)
        duplicate = model.sequence_loss(y.expand(3, -1, -1), u.expand(3, -1, -1))
        for key in terms: torch.testing.assert_close(terms[key], duplicate[key])
        torch.testing.assert_close(terms['loss'], sum(terms[k] for k in ('expected_observation_nll', 'initial_kl', 'transition_kl')))
        single = model.sequence_loss(y[:, :1], u[:, :1])
        self.assertEqual(single['transition_kl'].item(), 0.)
        terms['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_data_states_streams_and_explosions_reject(self):
        model, state = model_fixture()
        with self.assertRaisesRegex(ValueError, 'HISTORY_REQUIRED'): model.infer_state(torch.zeros(1, 2, 2))
        for value in (-1., .5, float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
                model.encode(torch.full((1, 2, 2), value), torch.zeros(1, 2, 1))
        with self.assertRaisesRegex(ValueError, 'REAL_VALUES'):
            model.encode(torch.ones(1, 2, 2, dtype=torch.complex64), torch.zeros(1, 2, 1))
        with self.assertRaisesRegex(ValueError, 'BIN_WIDTH'):
            model.forecast_mean(replace(state, bin_seconds=.1), torch.zeros(2, 1))
        with self.assertRaisesRegex(FloatingPointError, 'VARIANCE'):
            model.simulate(replace(state, variance=state.variance * 0), torch.zeros(2, 1), rng(), 1)
        with self.assertRaisesRegex(ValueError, 'DISTINCT'): LatentRandomStreams.from_seeds(1, 1, 2)
        shared = torch.Generator().manual_seed(1)
        with self.assertRaisesRegex(ValueError, 'DISTINCT'):
            model.simulate(state, torch.zeros(2, 1), LatentRandomStreams(shared, shared, shared), 1)
        with torch.no_grad(): model.log_rate_bias.fill_(1e4)
        with self.assertRaisesRegex(FloatingPointError, 'NONFINITE_GENERATION'): model.simulate(state, torch.zeros(2, 1), rng(), 1)
        with self.assertRaisesRegex(FloatingPointError, 'NONFINITE_FORECAST'): model.forecast_mean(state, torch.zeros(2, 1))

    def test_cli_fit_manifest_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); write_windows(root / 'input')
            config = load_config(CONFIG); config['poisson_state_space_check']['optimizer_steps'] = 2
            (root / 'config.yaml').write_text(yaml.safe_dump(config))
            args = ['--config', str(root / 'config.yaml'), 'baseline-check', '--recording-dir', str(root / 'input'),
                    '--output-dir', str(root / 'result'), '--seed', '41']
            self.assertEqual(parser().parse_args(args).model, 'poisson-autoregression')
            args += ['--model', 'poisson-state-space']
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(args), 2)
            summary = json.loads(stream.getvalue())
            self.assertFalse(summary['formal_seed_complete'])
            self.assertFalse(summary['scientific_claims_qualified'])
            manifest = json.loads((root / 'result/manifest.json').read_text())
            for name, digest in manifest['artifacts'].items():
                self.assertEqual(sha256_file(root / 'result' / name), digest)
            with h5py.File(root / 'result/count_samples.h5') as h:
                self.assertEqual(h['counts'].shape, (8, 2, 3, 2))
                self.assertEqual(h['marginal_expected_counts'].shape, (2, 3, 2))
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(args), 1)
            self.assertEqual(json.loads(errors.getvalue())['error_type'], 'FileExistsError')


if __name__ == '__main__':
    unittest.main()

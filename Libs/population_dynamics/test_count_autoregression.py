"""Distribution, causality and serialization checks for the count baseline."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import torch

from .count_autoregression import CountHistoryState, PoissonAutoregression


def model(input_size=1, history_bins=1):
    result = PoissonAutoregression([12, 4], input_size, history_bins, .5, initialization_seed=41).double()
    with torch.no_grad():
        result.history_weight.zero_(); result.input_weight.zero_()
        result.log_rate_bias.fill_(math.log(2.))
    return result


class CountBaselineTests(unittest.TestCase):
    def test_count_units_and_exact_poisson_score(self):
        m = model()
        y = torch.tensor([[[0., 0.], [0., 1.], [3., 2.]]], dtype=torch.float64)
        score = m.sequence_loss(y, torch.zeros(1, 3, 1))
        expected = (4. + math.lgamma(1.) + math.lgamma(2.) + math.lgamma(4.) + math.lgamma(3.)) / 4.
        self.assertAlmostEqual(score['poisson_nll'].item(), expected, places=12)
        self.assertAlmostEqual(m.conditional_log_expected_counts(y[:, :1], torch.zeros(1, 1, dtype=torch.float64)).exp()[0, 0].item(), 1., places=12)

    def test_history_unfold_matches_manual_causal_likelihood(self):
        m = model(history_bins=2)
        with torch.no_grad():
            m.history_weight.copy_(torch.tensor([[.1, -.2, .3, -.1], [-.2, .1, -.1, .2]]))
            m.input_weight.fill_(.25)
        y = torch.tensor([[[1., 0.], [0., 2.], [3., 1.], [2., 0.]]], dtype=torch.float64)
        u = torch.tensor([[[0.], [1.], [-1.], [2.]]], dtype=torch.float64)
        scores = []
        for t in range(2, y.shape[1]):
            log_mu = m.conditional_log_expected_counts(y[:, t-2:t], u[:, t])
            scores.append(log_mu.exp() - y[:, t] * log_mu + torch.lgamma(y[:, t] + 1))
        torch.testing.assert_close(m.sequence_loss(y, u)['poisson_nll'], torch.stack(scores).mean())

    def test_prefix_state_uses_last_lags_and_preserves_unit_order(self):
        m = model(history_bins=2)
        history = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        state = m.infer_state(history, {'unit_ids': [12, 4]})
        torch.testing.assert_close(state.counts[0], history[-2:].double())
        history[-1] = 99
        self.assertEqual(state.counts[0, -1, 0].item(), 5.)
        self.assertIn('point_mass', state.uncertainty)
        with self.assertRaisesRegex(ValueError, 'UNIT_ORDER'):
            m.infer_state(history, {'unit_ids': [4, 12]})

    def test_constructor_and_sampling_do_not_consume_global_rng(self):
        before = torch.random.get_rng_state().clone()
        m = model()
        m.simulate(m.infer_state([[0., 0.]]), torch.zeros(3, 1), torch.Generator().manual_seed(801), 4)
        torch.testing.assert_close(before, torch.random.get_rng_state())

    def test_known_stimulus_changes_conditional_rate_and_gradient(self):
        m = model()
        with torch.no_grad(): m.input_weight.fill_(math.log(2.))
        h = torch.zeros(1, 1, 2, dtype=torch.float64)
        u = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
        mu = m.conditional_log_expected_counts(h, u).exp()
        torch.testing.assert_close(mu, torch.full_like(mu, 2.))
        mu.sum().backward()
        self.assertAlmostEqual(u.grad.item(), 4. * math.log(2.), places=12)

    def test_generation_replays_and_uses_sampled_counts_in_future_history(self):
        m = model()
        with torch.no_grad(): m.history_weight[1, 0] = .5
        state = m.infer_state([[0., 0.]])
        first = m.simulate(state, torch.zeros(3, 1), torch.Generator().manual_seed(201), 32)
        replay = m.simulate(state, torch.zeros(3, 1), torch.Generator().manual_seed(201), 32)
        other = m.simulate(state, torch.zeros(3, 1), torch.Generator().manual_seed(202), 32)
        torch.testing.assert_close(first.counts, replay.counts)
        self.assertFalse(torch.equal(first.counts, other.counts))
        self.assertEqual(first.counts.dtype, torch.int64)
        self.assertEqual(first.counts.shape, (32, 1, 3, 2))
        torch.testing.assert_close(first.conditional_expected_counts[:, :, 1:, 1],
                                   torch.exp(.5 * first.counts[:, :, :-1, 0].double()))
        self.assertEqual(state.counts.sum().item(), 0.)
        torch.testing.assert_close(first.population_counts, first.counts.sum(-1))

    def test_no_known_input_channels_and_multiple_batches(self):
        m = model(input_size=0)
        counts = torch.zeros(2, 4, 2)
        score = m.sequence_loss(counts, torch.empty(2, 4, 0))
        self.assertAlmostEqual(score['poisson_nll'].item(), 1.)
        sample = m.simulate(m.infer_state(counts), torch.empty(2, 3, 0), torch.Generator().manual_seed(15), 2)
        self.assertEqual(sample.counts.shape, (2, 2, 3, 2))

    def test_invalid_counts_and_missing_input_reject(self):
        m = model()
        for value in (-1., .25, float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'INTEGER_COUNTS'):
                m.infer_state([[value, 0.]])
        with self.assertRaisesRegex(ValueError, 'STIMULUS_SEQUENCE'):
            m.simulate(m.infer_state([[0., 0.]]), torch.empty(3, 0), torch.Generator().manual_seed(1), 1)

    def test_exploding_generation_fails_without_clipping(self):
        m = model()
        with torch.no_grad(): m.log_rate_bias.fill_(1000.)
        with self.assertRaisesRegex(FloatingPointError, 'NONFINITE_GENERATION'):
            m.simulate(m.infer_state([[0., 0.]]), torch.zeros(1, 1), torch.Generator().manual_seed(1), 1)

    def test_poisson_intercept_mle_matches_analytic_count_mean(self):
        m = model(input_size=0)
        y = torch.tensor([[[0., 0.], [0., 1.], [2., 2.], [4., 0.], [2., 1.]]], dtype=torch.float64)
        u = torch.empty(1, 5, 0, dtype=torch.float64)
        optimizer = torch.optim.LBFGS([m.log_rate_bias], max_iter=40, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = m.sequence_loss(y, u)['loss']; loss.backward()
            return loss
        optimizer.step(closure)
        torch.testing.assert_close(m.log_rate_bias.exp() * m.bin_seconds, y[:, 1:].mean((0, 1)), atol=1e-6, rtol=1e-6)

    def test_checkpoint_restores_dtype_identity_and_stochastic_replay(self):
        m = model(history_bins=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            m.save(path); restored = PoissonAutoregression.load(path)
            self.assertEqual(restored.unit_ids, (12, 4))
            self.assertEqual(restored.log_rate_bias.dtype, torch.float64)
            h = [[0., 0.], [1., 0.]]; stimulus = torch.zeros(5, 1)
            a = m.simulate(m.infer_state(h), stimulus, torch.Generator().manual_seed(2025), 8)
            b = restored.simulate(restored.infer_state(h), stimulus, torch.Generator().manual_seed(2025), 8)
            torch.testing.assert_close(a.counts, b.counts)
            with self.assertRaises(FileExistsError): m.save(path)


if __name__ == '__main__':
    unittest.main()

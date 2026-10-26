"""Renewal likelihood normalization, source-clock exposure and event sampling."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .native_count_windows import NativeEpisodeCounts
from .poisson_state_space import LatentRandomStreams
from .refractory_observation import observation_nll, renewal_window, sample_bin
from .renewal_state_space import ControlledRenewalLDS


def enumerate_paths(length, interval, initial_ready, probability):
    result = []
    for path in itertools.product((0, 1), repeat=length):
        ready, likelihood, exposure = initial_ready, 1., 0
        valid = True
        for tick, spike in enumerate(path):
            if tick < ready:
                if spike:
                    valid = False
                    break
            else:
                exposure += 1
                likelihood *= probability if spike else 1 - probability
                if spike:
                    ready = tick + interval
        if valid:
            result.append((path, exposure, likelihood))
    return result


class RenewalObservationTests(unittest.TestCase):
    def test_exact_point_process_likelihood_normalizes_over_valid_event_paths(self):
        rate, dt = 270., .001
        probability = 1 - math.exp(-rate * dt)
        paths = enumerate_paths(7, 3, 1, probability)
        self.assertAlmostEqual(sum(p for _, _, p in paths), 1., places=14)
        for path, exposure, probability in paths:
            nll = observation_nll(torch.tensor([sum(path)], dtype=torch.float64),
                                  torch.tensor([exposure], dtype=torch.float64),
                                  torch.tensor([math.log(rate)], dtype=torch.float64), dt)
            self.assertAlmostEqual(float(nll), -math.log(probability), places=13)

    def test_log_rate_gradient_matches_finite_difference(self):
        rate = torch.tensor([[math.log(20.), math.log(1400.)]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[1., 2.]], dtype=torch.float64)
        exposure = torch.tensor([[31., 12.]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda x: observation_nll(y, exposure, x, .0001), (rate,)))

    def test_geometric_sampler_matches_enumerated_distribution(self):
        rate, dt, draws = 270., .001, 50000
        exact = np.zeros(4)
        for path, _, probability in enumerate_paths(7, 3, 1, 1 - math.exp(-rate * dt)):
            exact[sum(path)] += probability
        sampled = sample_bin(torch.full((draws, 1), math.log(rate), dtype=torch.float64),
                             torch.ones((draws, 1), dtype=torch.int64), torch.tensor([3]),
                             7, dt, torch.Generator().manual_seed(491))
        empirical = np.bincount(sampled.counts.numpy().ravel(), minlength=4) / draws
        np.testing.assert_allclose(empirical, exact, atol=.008, rtol=0)
        event_count = torch.bincount(sampled.events[:, 0], minlength=draws)
        torch.testing.assert_close(event_count, sampled.counts[:, 0])

    def fixture(self):
        data = NativeEpisodeCounts.__new__(NativeEpisodeCounts)
        data.unit_ids = np.array([12, 19])
        data.bin_ticks, data.duration_ticks = 5, 20
        pairs = sorted([(1, 0), (4, 0), (7, 0), (10, 0), (16, 0), (0, 1), (2, 1), (8, 1), (18, 1)])
        data.event_ticks = np.array([p[0] for p in pairs])
        data.event_units = np.array([p[1] for p in pairs])
        data.pulse_ticks = np.array([6, 11])
        return data

    def test_sparse_exposure_matches_tick_by_tick_with_boundary_history(self):
        data = self.fixture()
        interval = np.array([3, 2])
        result = renewal_window(data, 5, 15, interval)
        eligible = np.zeros((2, 2), dtype=int)
        counts = np.zeros((2, 2), dtype=int)
        last = [-100, -100]
        for tick in range(15):
            for unit in range(2):
                if tick >= 5:
                    eligible[(tick - 5) // 5, unit] += tick >= last[unit] + interval[unit]
                if np.any((data.event_ticks == tick) & (data.event_units == unit)):
                    last[unit] = tick
                    if tick >= 5:
                        counts[(tick - 5) // 5, unit] += 1
        np.testing.assert_array_equal(result.eligible_ticks, eligible)
        np.testing.assert_array_equal(result.counts, counts)
        np.testing.assert_array_equal(result.initial_ready_in_ticks, [2, 0])
        np.testing.assert_array_equal(result.terminal_ready_in_ticks, [0, 0])
        # A future event changes neither the current exposure nor boundary state.
        data.event_ticks[-1] += 1
        again = renewal_window(data, 5, 15, interval)
        np.testing.assert_array_equal(again.eligible_ticks, eligible)

    def test_incompatible_spikes_and_unobserved_availability_are_rejected(self):
        data = self.fixture()
        with self.assertRaisesRegex(ValueError, "violates"):
            renewal_window(data, 5, 15, np.array([4, 2]))
        with self.assertRaisesRegex(ValueError, "boundary availability"):
            sample_bin(torch.zeros(2), torch.tensor([3, 0]), torch.tensor([3, 2]),
                       5, .001, torch.Generator())


class RenewalStateSpaceTests(unittest.TestCase):
    def model(self):
        return ControlledRenewalLDS(unit_ids=[12, 19, 27], input_size=1, latent_size=2,
                                   encoder_size=3, bin_seconds=.005, initialization_seed=41,
                                   tick_seconds=.001, minimum_interval_ticks=[2, 3, 4]).double()

    def history(self):
        return dict(counts=torch.tensor([[[1., 0., 1.], [0., 1., 0.]]], dtype=torch.float64),
                    stimulus=torch.zeros(1, 2, 1, dtype=torch.float64),
                    ready_in_ticks=torch.tensor([[0, 2, 0]]))

    def test_loss_randomness_gradients_and_checkpoint_replay(self):
        model = self.model()
        history = self.history()
        exposure = torch.tensor([[[4., 5., 2.], [4., 3., 5.]]], dtype=torch.float64)
        generator = torch.Generator().manual_seed(421)
        rng_state = generator.get_state()
        terms = model.sequence_loss(history["counts"], history["stimulus"], exposure, generator)
        terms["loss"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        generator.set_state(rng_state)
        repeated = model.sequence_loss(history["counts"], history["stimulus"], exposure, generator)
        torch.testing.assert_close(terms["loss"], repeated["loss"], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "renewal.pt"
            model.save(path)
            loaded = ControlledRenewalLDS.load(path)
            future = torch.zeros(1, 12, 1, dtype=torch.float64)
            a = model.simulate(model.infer_state(history), future,
                               LatentRandomStreams.from_seeds(101, 102, 103), 7)
            b = loaded.simulate(loaded.infer_state(history), future,
                                LatentRandomStreams.from_seeds(101, 102, 103), 7)
            for name in ("counts", "events", "latent_states", "initial_latent_states", "terminal_ready_in_ticks"):
                torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)

    def test_events_reproduce_counts_deadtime_and_terminal_history_across_bins(self):
        model = self.model()
        with torch.no_grad():
            model.latent_model.log_rate_bias.fill_(math.log(1000.))
        history = self.history()
        sampled = model.simulate(model.infer_state(history), torch.zeros(1, 12, 1, dtype=torch.float64),
                                 LatentRandomStreams.from_seeds(101, 102, 103), 7)
        counts = np.zeros(sampled.counts.shape, dtype=np.int64)
        events = sampled.events.numpy()
        for draw, batch, unit, tick in events:
            counts[draw, batch, tick // 5, unit] += 1
        np.testing.assert_array_equal(counts, sampled.counts.numpy())
        for draw in range(7):
            for unit, interval in enumerate([2, 3, 4]):
                times = np.sort(events[(events[:, 0] == draw) & (events[:, 2] == unit), 3])
                self.assertTrue(np.all(np.diff(times) >= interval))
                self.assertGreaterEqual(times[0], int(history["ready_in_ticks"][0, unit]))
                expected = max(int(times[-1]) + interval - 60, 0)
                self.assertEqual(int(sampled.terminal_ready_in_ticks[draw, 0, unit]), expected)

    def test_history_and_clock_contract(self):
        history = self.history()
        del history["ready_in_ticks"]
        with self.assertRaisesRegex(ValueError, "last-spike"):
            self.model().infer_state(history)
        with self.assertRaisesRegex(ValueError, "align"):
            ControlledRenewalLDS(unit_ids=[1], input_size=1, latent_size=1, encoder_size=1,
                                 bin_seconds=.005, initialization_seed=41, tick_seconds=.0003,
                                 minimum_interval_ticks=[7])

    def test_count_only_output_preserves_draws_latents_history_and_rng_states(self):
        model = self.model()
        with torch.no_grad():
            model.latent_model.log_rate_bias.fill_(math.log(1000.))
        state = model.infer_state(self.history())
        future = torch.zeros(1, 12, 1, dtype=torch.float64)
        complete_rng = LatentRandomStreams.from_seeds(101, 102, 103)
        counts_rng = LatentRandomStreams.from_seeds(101, 102, 103)
        complete = model.simulate(state, future, complete_rng, 7)
        counts = model.simulate(state, future, counts_rng, 7, record_events=False)
        self.assertGreater(len(complete.events), 0)
        self.assertTrue(complete.events_recorded)
        self.assertFalse(counts.events_recorded)
        self.assertEqual(tuple(counts.events.shape), (0, 4))
        for name in ("counts", "latent_states", "initial_latent_states", "terminal_ready_in_ticks"):
            torch.testing.assert_close(getattr(complete, name), getattr(counts, name), rtol=0, atol=0)
        for name in ("initial", "dynamics", "observation"):
            torch.testing.assert_close(getattr(complete_rng, name).get_state(), getattr(counts_rng, name).get_state())


if __name__ == "__main__":
    unittest.main()

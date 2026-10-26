"""Trajectory, boundary, and native-clock tests for frozen residual correction."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from population_dynamics.poisson_state_space import LatentRandomStreams
from population_dynamics.population_state import ControlledPopulationState
from population_dynamics.population_residual_memory import ResidualMemoryPopulationState
from population_dynamics.residual_prediction import RidgeModel


def make_parent():
    parent = ControlledPopulationState(
        unit_ids=[2, 7, 11, 13], unit_field_index=[0, 0, 1, 1], field_names=["A", "B"],
        minimum_interval_ticks=[2, 3, 2, 3], tick_seconds=.001, bin_seconds=.005,
        input_size=1, latent_channels=2, encoder_size=3, representation="recurrent",
        initialization_seed=41, unit_chunk_size=2, metadata=None,
    ).eval()
    with torch.no_grad():
        parent.log_hazard_bias.fill_(5)
    return parent


def history():
    return {"counts": torch.tensor([[[0., 1., 0., 0.], [1., 0., 1., 0.],
                                     [0., 0., 0., 1.], [0., 1., 0., 0.]]]),
            "stimulus": torch.tensor([[[0.], [.1], [-.2], [.3]]]),
            "ready_in_ticks": torch.tensor([[0, 1, 0, 2]], dtype=torch.long)}


def streams():
    return LatentRandomStreams.from_seeds(41, 42, 43, device="cpu")


def predictor(mode, memory_gain=0., constant=0.):
    d, p = 4, 5 if mode == "current" else 9
    coefficients = np.zeros((p, d))
    if mode == "history":
        coefficients[-d:] = memory_gain * np.eye(d)
    return RidgeModel(.0001, np.zeros(p), np.ones(p), np.full(d, constant), np.ones(d), coefficients)


def assert_steps_equal(test, left, right):
    test.assertEqual(len(left), len(right))
    for a, b in zip(left, right):
        for name in ("counts", "events", "latent", "ready_in_ticks", "initial_latent"):
            test.assertTrue(torch.equal(getattr(a, name), getattr(b, name)), name)


class ResidualMemoryPopulationTests(unittest.TestCase):
    def test_zero_correction_preserves_parent_and_all_explicit_rng_states(self):
        for mode in ("current", "history"):
            parent = make_parent()
            before = {name: value.detach().cpu().numpy().tobytes() for name, value in parent.state_dict().items()}
            global_rng = torch.get_rng_state().clone()
            identity = {"source": {"seed": 41}}
            model = ResidualMemoryPopulationState(parent, predictor(mode), mode, identity)
            self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
            identity["source"]["seed"] = 42
            self.assertEqual(model.correction_identity["source"]["seed"], 41)
            original, corrected = parent.infer_state(history()), model.infer_state(history())
            u = torch.tensor([[[.1], [.2], [-.1], [0.]]])
            parent_rng, corrected_rng = streams(), streams()
            a = list(parent.iter_simulation(original, u, parent_rng, 3))
            b = list(model.iter_simulation(corrected, u, corrected_rng, 3))
            assert_steps_equal(self, a, b)
            for name in ("initial", "dynamics", "observation"):
                self.assertTrue(torch.equal(getattr(parent_rng, name).get_state(), getattr(corrected_rng, name).get_state()))
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()),
                             sum(parameter.numel() for parameter in parent.parameters())
                             + model.correction_weight.numel() + model.correction_bias.numel())
            self.assertEqual(before, {name: value.detach().cpu().numpy().tobytes() for name, value in parent.state_dict().items()})

    def test_causal_boundary_encodes_once_and_keeps_last_posterior(self):
        parent = make_parent()
        model = ResidualMemoryPopulationState(parent, predictor("history"), "history", {})
        observed = history()
        mean, variance = parent.encode(observed["counts"], observed["stimulus"])
        expected_memory = mean[:, -1] - parent.transition_mean(mean[:, -2], observed["stimulus"][:, -1])
        with patch.object(parent, "encode", wraps=parent.encode) as encode:
            boundary = model.infer_state(observed)
        self.assertEqual(encode.call_count, 1)
        self.assertTrue(torch.equal(boundary.mean, mean[:, -1]))
        self.assertTrue(torch.equal(boundary.variance, variance[:, -1]))
        self.assertTrue(torch.equal(boundary.memory, expected_memory))
        short = {**observed, "counts": observed["counts"][:, :1], "stimulus": observed["stimulus"][:, :1]}
        with self.assertRaisesRegex(ValueError, "two observed bins"):
            model.infer_state(short)
        with self.assertRaisesRegex(ValueError, "metadata"):
            model.infer_state(observed, {"unit_ids": [99], "field_names": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "availability"):
            model.infer_state({**observed, "ready_in_ticks": parent.minimum_interval_ticks[None]})

    def test_affine_conversion_matches_normalized_ridge_in_parent_precision(self):
        rng = np.random.default_rng(43)
        for dtype in (torch.float32, torch.float64):
            for mode in ("current", "history"):
                parent = make_parent().to(dtype=dtype)
                p = 5 if mode == "current" else 9
                fitted = RidgeModel(.0001, rng.normal(size=p), rng.uniform(.5, 2., size=p),
                                    rng.normal(size=4), rng.uniform(.5, 2., size=4),
                                    rng.normal(size=(p, 4)) * .2)
                model = ResidualMemoryPopulationState(parent, fitted, mode, {})
                z = torch.tensor(rng.normal(size=(3, 1, 2, 2)), dtype=dtype)
                h = torch.tensor(rng.normal(size=(3, 1, 2, 2)), dtype=dtype)
                u = torch.tensor([[.3]], dtype=dtype)
                parts = [z.numpy().reshape(3, 1, 4), np.broadcast_to(u.numpy(), (3, 1, 1))]
                if mode == "history":
                    parts.append(h.numpy().reshape(3, 1, 4))
                expected = fitted.predict(np.concatenate(parts, axis=-1)).reshape(3, 1, 2, 2)
                tolerance = 2e-6 if dtype == torch.float32 else 1e-12
                np.testing.assert_allclose(model.correction(z, u, h).numpy(), expected, atol=tolerance, rtol=tolerance)
                self.assertEqual(model.correction_weight.dtype, dtype)
                self.assertEqual(model.correction_weight.device, parent.posterior_mean.weight.device)

    def test_memory_recurrence_includes_the_original_process_draw(self):
        parent = make_parent()
        model = ResidualMemoryPopulationState(parent, predictor("history", memory_gain=.35), "history", {})
        state = model.infer_state(history())
        u = torch.tensor([[[.2], [-.1], [.4]]])
        reference_rng, actual_rng = streams(), streams()
        shape = (2, *state.mean.shape)
        x = state.mean + state.variance.sqrt() * torch.randn(shape, generator=reference_rng.initial)
        initial = x.clone()
        memory = state.memory.expand_as(x)
        scale = parent._field_parameters()[3].exp().sqrt()
        for time, step in enumerate(model.iter_simulation(state, u, actual_rng, 2)):
            base = parent.transition_mean(x, u[:, time])
            x = base + .35 * memory + scale * torch.randn(shape, generator=reference_rng.dynamics)
            memory = x - base
            torch.testing.assert_close(step.latent, x, atol=2e-7, rtol=2e-7)
            self.assertTrue(torch.equal(step.initial_latent, initial))
        self.assertTrue(torch.equal(actual_rng.initial.get_state(), reference_rng.initial.get_state()))
        self.assertTrue(torch.equal(actual_rng.dynamics.get_state(), reference_rng.dynamics.get_state()))

    def test_interleaved_iterators_do_not_share_trajectory_memory(self):
        parent = make_parent()
        model = ResidualMemoryPopulationState(parent, predictor("history", memory_gain=.2), "history", {})
        a = model.infer_state(history())
        b = replace(a, mean=a.mean + .4, memory=a.memory - .7)
        u = torch.tensor([[[.1], [.3], [-.1]]])
        reference_a = list(model.iter_simulation(a, u, streams(), 2))
        reference_b = list(model.iter_simulation(b, u, streams(), 2))
        iterator_a, iterator_b = model.iter_simulation(a, u, streams(), 2), model.iter_simulation(b, u, streams(), 2)
        actual_a, actual_b = [], []
        for _ in range(3):
            actual_a.append(next(iterator_a))
            actual_b.append(next(iterator_b))
        assert_steps_equal(self, reference_a, actual_a)
        assert_steps_equal(self, reference_b, actual_b)
        current = ResidualMemoryPopulationState(make_parent(), predictor("current", constant=.1), "current", {})
        boundary = current.infer_state(history())
        assert_steps_equal(
            self, list(current.iter_simulation(boundary, u, streams(), 2)),
            list(current.iter_simulation(replace(boundary, memory=boundary.memory + 100), u, streams(), 2)),
        )

    def test_events_match_counts_and_recording_does_not_change_rng_or_latents(self):
        model = ResidualMemoryPopulationState(make_parent(), predictor("history", memory_gain=.2), "history", {})
        boundary = model.infer_state(history())
        u = torch.zeros((1, 4, 1))
        recorded_rng, counts_rng = streams(), streams()
        recorded = list(model.iter_simulation(boundary, u, recorded_rng, 3, record_events=True))
        count_only = list(model.iter_simulation(boundary, u, counts_rng, 3, record_events=False))
        total_events = 0
        for time, (step, sparse) in enumerate(zip(recorded, count_only)):
            counts = torch.zeros_like(step.counts)
            events = step.events
            total_events += len(events)
            counts.index_put_((events[:, 0], events[:, 1], events[:, 2]), torch.ones(len(events), dtype=counts.dtype, device=counts.device), accumulate=True)
            self.assertTrue(torch.equal(counts, step.counts))
            self.assertTrue(torch.all((events[:, 3] >= time * model.bin_ticks) & (events[:, 3] < (time + 1) * model.bin_ticks)))
            self.assertEqual(len(sparse.events), 0)
            for name in ("counts", "latent", "initial_latent", "ready_in_ticks"):
                self.assertTrue(torch.equal(getattr(step, name), getattr(sparse, name)))
        self.assertGreater(total_events, 0)
        for name in ("initial", "dynamics", "observation"):
            self.assertTrue(torch.equal(getattr(recorded_rng, name).get_state(), getattr(counts_rng, name).get_state()))

    def test_boundary_and_stream_rejections_preserve_parent_contract(self):
        parent = make_parent()
        model = ResidualMemoryPopulationState(parent, predictor("history"), "history", {})
        state, u = model.infer_state(history()), torch.zeros((1, 1, 1))
        for boundary in (parent.infer_state(history()), replace(state, field_names=("wrong",)),
                         replace(state, memory=torch.zeros((1, 1))),
                         replace(state, memory=torch.full_like(state.memory, float("nan")))):
            with self.subTest(boundary=type(boundary).__name__), self.assertRaises(ValueError):
                next(model.iter_simulation(boundary, u, streams(), 1))
        rng = streams()
        rng = LatentRandomStreams(initial=rng.initial, dynamics=rng.initial, observation=rng.observation)
        with self.assertRaisesRegex(ValueError, "independent"):
            next(model.iter_simulation(state, u, rng, 1))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            ResidualMemoryPopulationState(parent, predictor("current"), "history", {})


if __name__ == "__main__":
    unittest.main()

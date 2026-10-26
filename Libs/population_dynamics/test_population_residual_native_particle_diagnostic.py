"""Nested draws and gradients for the unchanged native residual objective."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .population_residual_memory import ResidualMemoryPopulationState
from .population_residual_native_particle_diagnostic import probe, mean_probe, mean_conditions
from .population_residual_mean_objective import residual_mean_objective
from .population_residual_observation import configure_correction_parameters, residual_observation_mixture_loss
from .population_state import ControlledPopulationState
from .residual_prediction import RidgeModel


def fixture(mode="history"):
    parent = ControlledPopulationState(unit_ids=[0, 1, 2, 3], unit_field_index=[0, 0, 1, 1],
        field_names=["A", "B"], minimum_interval_ticks=[2] * 4, tick_seconds=.001, bin_seconds=.005,
        input_size=1, latent_channels=2, encoder_size=4, representation="recurrent", initialization_seed=41).double()
    width = 9 if mode == "history" else 5
    reference = RidgeModel(1., np.zeros(width), np.ones(width), np.zeros(4), np.ones(4),
                          np.full((width, 4), .01))
    model = ResidualMemoryPopulationState(parent, reference, mode, {})
    mean = torch.linspace(-.3, .7, 32, dtype=torch.float64).reshape(1, 8, 2, 2)
    variance = torch.full_like(mean, .2)
    controls = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1) / 8
    counts = torch.zeros(1, 2, 3, 4, dtype=torch.int64)
    counts[:, :, -1, ::2] = 1
    eligible = torch.full_like(counts, 5)
    return model, (mean, variance, controls, counts, eligible), (2, 4), (1, 3)


class NativeParticleTests(unittest.TestCase):
    def test_largest_budget_matches_direct_native_loss_and_gradient(self):
        model, inputs, origins, horizons = fixture()
        parameters = configure_correction_parameters(model)
        terms = residual_observation_mixture_loss(model, *inputs, origins, horizons,
            torch.Generator().manual_seed(42), torch.Generator().manual_seed(43),
            prediction_particles=4, unit_chunk_size=2, checkpoint_readout=True)
        expected = torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(terms["loss"], parameters)]).numpy()
        row, actual = probe(model, inputs, origins, horizons, 42, 43, 4, 4, 2)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(row["loss"], float(terms["loss"]))

    def test_nested_prefix_uses_same_native_trajectories_and_preserves_global_rng(self):
        model, inputs, origins, horizons = fixture()
        before = torch.get_rng_state().clone()
        paths = {}
        original = model.parent.log_hazard
        for particles in (2, 4):
            seen = []
            def observe(path, start=0, stop=None):
                if not seen:
                    seen.append(path.detach().clone())
                return original(path, start, stop)
            with patch.object(model.parent, "log_hazard", side_effect=observe):
                probe(model, inputs, origins, horizons, 42, 43, particles, 4, 2)
            paths[particles] = seen[0]
        torch.testing.assert_close(paths[2], paths[4][:2], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    def test_repeatability_and_no_parameter_update_for_both_modes(self):
        for mode in ("current", "history"):
            model, inputs, origins, horizons = fixture(mode)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            a, ga = probe(model, inputs, origins, horizons, 42, 43, 2, 4, 2)
            b, gb = probe(model, inputs, origins, horizons, 42, 43, 2, 4, 2)
            np.testing.assert_array_equal(ga, gb)
            self.assertEqual(a["loss"], b["loss"])
            self.assertTrue(a["gradient_finite"])
            self.assertTrue(a["gradient_nonzero"])
            for name, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_incompatible_labels_are_rejected(self):
        model, inputs, origins, horizons = fixture()
        inputs = (*inputs[:3], inputs[3][..., :-1], inputs[4])
        with self.assertRaisesRegex(ValueError, "aligned native event"):
            probe(model, inputs, origins, horizons, 42, 43, 2, 4, 2)



class MeanParticleTests(unittest.TestCase):
    def batch(self, mode="history"):
        model, inputs, origins, horizons = fixture(mode)
        batch = dict(zip(("mean", "variance", "controls", "target_counts", "target_eligible_ticks"), inputs))
        batch.update(origins=origins, horizons=horizons,
            initial_ready_in_ticks=torch.zeros(1, len(origins), len(model.unit_ids), dtype=torch.int64),
            field_standard_deviation_hz=torch.ones(len(model.field_names), dtype=torch.float64))
        return model, batch

    def test_mean_largest_budget_matches_direct_terms_and_component_gradients(self):
        model, batch = self.batch()
        parameters = configure_correction_parameters(model)
        terms = residual_mean_objective(model, **batch,
            initial_generator=torch.Generator().manual_seed(42),
            dynamics_generator=torch.Generator().manual_seed(43),
            prediction_particles=4, unit_chunk_size=2, mean_unit_chunk_size=2)
        expected = {}
        for i, key in enumerate(("loss", "native_loss", "field_mean_loss")):
            expected[key] = torch.cat([g.detach().reshape(-1) for g in
                torch.autograd.grad(terms[key], parameters, retain_graph=i < 2)]).numpy()
        row, vectors = mean_probe(model, batch, 42, 43, 4, 4, 2, 2)
        for key, value in terms.items():
            self.assertEqual(row[key], float(value))
        for key in expected:
            np.testing.assert_array_equal(vectors[key], expected[key])
        self.assertTrue(row["total_component_sum_verified"])
        np.testing.assert_allclose(vectors["loss"], vectors["native_loss"] + vectors["field_mean_loss"],
                                   rtol=1e-12, atol=1e-12)

    def test_mean_nested_paths_and_reconstruction_and_state_unchanged(self):
        for mode in ("current", "history"):
            model, batch = self.batch(mode)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            rng_before = torch.get_rng_state().clone()
            original = model.parent.log_hazard
            paths = {}
            for budget in (2, 4):
                seen = []
                def observe(path, start=0, stop=None):
                    if not seen:
                        seen.append(path.detach().clone())
                    return original(path, start, stop)
                with patch.object(model.parent, "log_hazard", side_effect=observe):
                    row, _ = mean_probe(model, batch, 42, 43, budget, 4, 2, 2)
                self.assertTrue(row["native_rng_reconstruction_verified"])
                paths[budget] = seen[0]
            torch.testing.assert_close(paths[2], paths[4][:2], rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(rng_before, torch.get_rng_state(), rtol=0, atol=0)
            for name, value in before.items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_mean_failure_restores_sampler_and_model(self):
        model, batch = self.batch()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        original = torch.randn
        batch["initial_ready_in_ticks"] = batch["initial_ready_in_ticks"][..., :-1]
        with self.assertRaises(ValueError):
            mean_probe(model, batch, 42, 43, 2, 4, 2, 2)
        self.assertIs(torch.randn, original)
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_mean_protocol_has_complete_four_template_denominator(self):
        rows = mean_conditions()
        keys = [(r["seed"], r["mode"], r["case_id"], r["replicate"], r["particles"]) for r in rows]
        self.assertEqual(len(keys), 144)
        self.assertEqual(len(set(keys)), 144)
        self.assertEqual({r["case_id"] for r in rows}, {f"train_block_000_familiar_{i}" for i in range(4)})
        self.assertEqual({r["seed"] for r in rows}, {41, 42, 43})


if __name__ == "__main__":
    unittest.main()

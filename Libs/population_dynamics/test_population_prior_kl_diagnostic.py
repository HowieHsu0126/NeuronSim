"""Finite-draw decomposition and unchanged stochastic-objective contracts."""
import unittest
from unittest.mock import patch

import torch

from .population_prior_kl_diagnostic import conditional_kl_components, recorded_rollout
from .population_prior_objective import prior_rollout_loss
from .population_state import ControlledPopulationState


def example():
    return ControlledPopulationState(unit_ids=list(range(7)),
        unit_field_index=[i % 2 for i in range(7)], field_names=["source_E", "source_I", "empty"],
        minimum_interval_ticks=[3]*7, tick_seconds=.001, bin_seconds=.005,
        input_size=1, latent_channels=2, encoder_size=5, representation="recurrent",
        initialization_seed=41, unit_chunk_size=3)


class PriorKLDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_finite_draw_identity_uses_population_variance(self):
        means = torch.tensor([[[0., 1.]], [[2., 5.]], [[4., 3.]]], dtype=torch.float64)
        target = torch.tensor([[3., 2.]], dtype=torch.float64)
        variance = torch.tensor([[.5, 2.]], dtype=torch.float64)
        process = torch.tensor([[2., .25]], dtype=torch.float64)
        parts = conditional_kl_components(means, target, variance, process)
        expected = .5 * (torch.log(process / variance) - 1 + (variance + (target-means).square()) / process).mean(0)
        torch.testing.assert_close(parts["direct_kl"], expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(parts["between_path_variance"], means.var(0, correction=0))
        torch.testing.assert_close(parts["variance_kl"]+parts["mean_bias_kl"]+parts["path_spread_kl"], expected)

    def test_exact_matching_gaussians_have_zero_components(self):
        mean = torch.tensor([[.5, -.5]])
        parts = conditional_kl_components(mean.repeat(4,1,1), mean, torch.ones_like(mean), torch.ones_like(mean))
        for name in ("direct_kl", "variance_kl", "mean_bias_kl", "path_spread_kl"):
            self.assertTrue(torch.equal(parts[name], torch.zeros_like(mean).double()))

    def test_recording_preserves_loss_and_both_random_streams(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = example().to(device)
                mean = torch.linspace(-.5, .5, 2*10*3*2).reshape(2,10,3,2).to(device)
                variance = torch.full_like(mean, .2)
                controls = torch.linspace(-1,1,20).reshape(2,10,1).to(device)
                def streams():
                    return (torch.Generator(device=model.unit_loading.device).manual_seed(41),
                            torch.Generator(device=model.unit_loading.device).manual_seed(42))
                a,b = streams(),streams()
                with torch.no_grad():
                    expected = prior_rollout_loss(model,mean,variance,controls,(2,5),(4,1,2),*a)
                actual,recorded = recorded_rollout(model,mean,variance,controls,(2,5),(4,1,2),*b)
                for name in expected:
                    self.assertTrue(torch.equal(expected[name],actual[name]))
                for first,second in zip(a,b):
                    self.assertTrue(torch.equal(first.get_state(),second.get_state()))
                self.assertEqual(set(recorded),{1,2,4})
                self.assertEqual(recorded[1].shape,(4,3,2))
                self.assertEqual(recorded[1].device.type,"cpu")
                self.assertNotIn("transition_mean",model.__dict__)

    def test_recording_restores_method_after_failure(self):
        model = example()
        with patch("Libs.population_dynamics.population_prior_kl_diagnostic.prior_rollout_loss",side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError,"failure"):
                recorded_rollout(model,None,None,None,(1,),(1,),None,None)
        self.assertNotIn("transition_mean",model.__dict__)

    def test_nonfinite_or_nonpositive_variance_and_wrong_axes_reject(self):
        mean=torch.zeros(4,3,2); target=torch.zeros(3,2); variance=torch.ones(3,2)
        for process in (torch.zeros(3,2),torch.full((3,2),float("nan")),torch.ones(3)):
            with self.subTest(shape=process.shape),self.assertRaises(ValueError):
                conditional_kl_components(mean,target,variance,process)
        with self.assertRaises(ValueError):
            conditional_kl_components(mean[:1],target,variance,variance)


if __name__ == "__main__":
    unittest.main()

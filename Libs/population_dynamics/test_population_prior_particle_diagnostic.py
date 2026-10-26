"""Nested particle draw layout and unchanged objective/gradient checks."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .population_observation_prior import prior_observation_mixture_loss
from .population_prior_objective import configure_prior_parameters
from .population_prior_particle_diagnostic import nested_objective, gradient_probe
from .population_state import ControlledPopulationState


def fixture():
    model = ControlledPopulationState(unit_ids=[0,1,2,3],unit_field_index=[0,0,1,1],field_names=["A","B"],
        minimum_interval_ticks=[2]*4,tick_seconds=.001,bin_seconds=.005,input_size=1,
        latent_channels=2,encoder_size=4,representation="recurrent",initialization_seed=41).double()
    configure_prior_parameters(model)
    mean = torch.linspace(-.3,.7,32,dtype=torch.float64).reshape(1,8,2,2)
    variance = torch.full_like(mean,.2)
    controls = torch.arange(8,dtype=torch.float64).reshape(1,8,1)/8
    counts = torch.zeros(1,2,3,4,dtype=torch.int64)
    counts[:,:,-1,::2] = 1
    eligible = torch.full_like(counts,5)
    return model,(mean,variance,controls,counts,eligible),(2,4),(1,3)


class ParticleDiagnosticTests(unittest.TestCase):
    def test_largest_budget_equals_original_objective_and_gradient(self):
        model,inputs,origins,horizons=fixture()
        direct=prior_observation_mixture_loss(model,*inputs,origins,horizons,
            torch.Generator().manual_seed(42),torch.Generator().manual_seed(43),prediction_particles=4)
        expected=torch.autograd.grad(direct["loss"],tuple(p for p in model.parameters() if p.requires_grad))
        nested=nested_objective(model,inputs,origins,horizons,
            torch.Generator().manual_seed(42),torch.Generator().manual_seed(43),particles=4,maximum_particles=4)
        actual=torch.autograd.grad(nested["loss"],tuple(p for p in model.parameters() if p.requires_grad))
        for key in direct:
            torch.testing.assert_close(direct[key],nested[key],rtol=0,atol=0)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_nested_paths_match_at_every_transition_and_global_rng_unchanged(self):
        model,inputs,origins,horizons=fixture()
        paths={}
        original=model.transition_mean
        rng_before=torch.get_rng_state().clone()
        for particles in (2,4):
            saved=[]
            def observe(state,control):
                saved.append(state.detach().clone())
                return original(state,control)
            with patch.object(model,"transition_mean",side_effect=observe):
                nested_objective(model,inputs,origins,horizons,
                    torch.Generator().manual_seed(42),torch.Generator().manual_seed(43),
                    particles=particles,maximum_particles=4)
            paths[particles]=saved
        for small,large in zip(paths[2],paths[4]):
            torch.testing.assert_close(small,large[:4],rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(rng_before,torch.get_rng_state(),rtol=0,atol=0)

    def test_gradient_probe_is_repeatable_and_does_not_update_parameters(self):
        model,inputs,origins,horizons=fixture()
        before={k:v.clone() for k,v in model.state_dict().items()}
        a,ga=gradient_probe(model,inputs,origins,horizons,42,43,2,4)
        b,gb=gradient_probe(model,inputs,origins,horizons,42,43,2,4)
        np.testing.assert_array_equal(ga,gb)
        self.assertEqual(a["loss"],b["loss"])
        self.assertGreater(a["gradient_norm"],0)
        for key,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)

    def test_invalid_nested_budget_rejected(self):
        model,inputs,origins,horizons=fixture()
        for particles,maximum in ((0,4),(5,4),(True,4)):
            with self.subTest(particles=particles),self.assertRaises(ValueError):
                nested_objective(model,inputs,origins,horizons,
                    torch.Generator().manual_seed(42),torch.Generator().manual_seed(43),
                    particles=particles,maximum_particles=maximum)

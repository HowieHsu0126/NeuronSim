"""Pinned BrainDyn author forecasting and optimization entry points."""
import pytest
import torch

from . import braindyn


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_author_field_forecast_gradient_and_rng_preservation(seed):
    before = torch.get_rng_state().clone()
    model = braindyn.initialize(3, 5, seed)
    assert torch.equal(before, torch.get_rng_state())
    history = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    predicted = braindyn.forecast(model, history, edges, 4, .005)
    assert predicted.shape == (4, 3) and torch.isfinite(predicted).all()
    loss = (predicted - history[-1]).square().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    old = model.dynamics.vector_field[0].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(old, model.dynamics.vector_field[0].weight)
    assert torch.equal(before, torch.get_rng_state())


def test_forecast_keeps_author_values_without_clipping():
    model = braindyn.initialize(2, 3, 41)
    values = torch.full((3, 2), -2.)
    edges = torch.tensor([[0, 1], [1, 0]])
    prediction = braindyn.forecast(model, values, edges, 1, .005)
    assert torch.any(prediction < 0)


@pytest.mark.parametrize("seed", [0, 40, 44, True])
def test_unregistered_random_seed_rejects(seed):
    with pytest.raises(ValueError, match="SEED_41_42_43"):
        braindyn.initialize(2, 3, seed)

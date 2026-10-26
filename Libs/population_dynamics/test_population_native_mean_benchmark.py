"""Tiny exact unit-to-field means and checkpointed readout gradients."""
import pytest
import torch
from torch import nn

from .native_renewal_expectation import expected_sequence_counts
from .population_native_mean_benchmark import expected_field_counts


class ToyReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit_ids = tuple(range(5))
        self.field_names = ("field_a", "field_b", "field_c")
        self.latent_channels = 2
        self.bin_ticks, self.tick_seconds = 2, .001
        self.register_buffer("unit_field_index", torch.tensor([2, 0, 2, 1, 0]))
        self.register_buffer("minimum_interval_ticks", torch.tensor([3, 1, 2, 2, 1]))
        self.register_buffer("unit_loading", torch.tensor([[.2, -.1], [.1, .4], [-.3, .2], [.4, .1], [.2, .3]], dtype=torch.float64))
        self.log_hazard_bias = nn.Parameter(torch.tensor([4., 4.5, 5., 4.2, 3.8], dtype=torch.float64))

    def log_hazard(self, path, start=0, stop=None):
        stop = len(self.unit_ids) if stop is None else stop
        selected = path.index_select(-2, self.unit_field_index[start:stop])
        return (selected * self.unit_loading[start:stop]).sum(-1) + self.log_hazard_bias[start:stop]


def inputs():
    parent = ToyReadout()
    path = torch.linspace(-.3, .4, 36, dtype=torch.float64).reshape(2, 3, 3, 2).requires_grad_()
    ready = torch.tensor([0, 0, 1, 1, 0])
    return parent, path, ready


def direct_fields(parent, path, ready):
    counts = expected_sequence_counts(parent.log_hazard(path), ready.expand(path.shape[0], -1),
        parent.minimum_interval_ticks, parent.bin_ticks, parent.tick_seconds)
    # Independent field aggregation explicitly visits each named unit.
    return torch.stack([counts[..., parent.unit_field_index == field].sum(-1)
                        for field in range(len(parent.field_names))], dim=-1)


def test_field_aggregation_covers_every_unit_and_preserves_total_expected_spikes():
    parent, path, ready = inputs()
    actual = expected_field_counts(parent, path, ready, unit_chunk_size=2, checkpoint_readout=False)
    expected = direct_fields(parent, path, ready)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    counts = expected_sequence_counts(parent.log_hazard(path), ready.expand(path.shape[0], -1),
        parent.minimum_interval_ticks, parent.bin_ticks, parent.tick_seconds)
    torch.testing.assert_close(actual.sum(-1), counts.sum(-1), rtol=1e-12, atol=1e-14)
    assert actual.shape == (2, 3, 3) and actual.dtype == path.dtype


@pytest.mark.parametrize("chunk", [1, 2, 5, 9])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_checkpoint_and_chunk_values_and_gradients_match_direct_readout(chunk, checkpointed):
    parent, path, ready = inputs()
    weights = torch.tensor([1., .3, 1.7], dtype=path.dtype)
    expected = direct_fields(parent, path, ready)
    expected_gradients = torch.autograd.grad((expected.square() * weights).sum(), (path, parent.log_hazard_bias))
    actual = expected_field_counts(parent, path, ready, unit_chunk_size=chunk, checkpoint_readout=checkpointed)
    actual_gradients = torch.autograd.grad((actual.square() * weights).sum(), (path, parent.log_hazard_bias))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-11, atol=1e-14)
        assert torch.isfinite(actual_gradient).all() and actual_gradient.norm() > 0
    assert path.grad is None and parent.log_hazard_bias.grad is None


def test_checkpointed_readout_preserves_inputs_parameters_and_rng():
    parent, path, ready = inputs()
    before = {name: value.detach().clone() for name, value in parent.state_dict().items()}
    path_before, ready_before, rng = path.detach().clone(), ready.clone(), torch.get_rng_state().clone()
    result = expected_field_counts(parent, path, ready, unit_chunk_size=2, checkpoint_readout=True)
    torch.autograd.grad(result.sum() / (len(path) * len(parent.unit_ids)), (path, parent.log_hazard_bias))
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.testing.assert_close(path, path_before, rtol=0, atol=0)
    torch.testing.assert_close(ready, ready_before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["chunk", "ready", "map", "unoccupied_field", "path"])
def test_incomplete_chunk_and_source_interfaces_fail(change):
    parent, path, ready = inputs()
    chunk = 2
    if change == "chunk":
        chunk = 0
    elif change == "ready":
        ready = ready[:-1]
    elif change == "map":
        parent.unit_field_index[0] = 3
    elif change == "unoccupied_field":
        parent.unit_field_index[3] = 0
    else:
        path = path[:, :, :2]
    with pytest.raises(ValueError, match="(particle/time|source units)"):
        expected_field_counts(parent, path, ready, unit_chunk_size=chunk)

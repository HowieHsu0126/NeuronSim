"""Causal, spectral and explicit-state checks for paper-reimplemented MemNO."""
import pytest
import torch

from .memory_neural_operator import MemoryNeuralOperator


@pytest.fixture(params=[41, 42, 43])
def model(request):
    torch.set_num_threads(1)
    torch.manual_seed(request.param)
    return MemoryNeuralOperator(2, (4, 4), 1, .005)


def inputs():
    return torch.rand(2, 6, 2, 4, 4), torch.randn(2, 6, 1)


def test_complete_grid_gradient_update(model):
    field, control = inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = model.readout.weight.detach().clone()
    prediction = model(field, control)
    assert prediction.shape == field.shape
    assert torch.isfinite(prediction).all() and (prediction >= 0).all()
    prediction.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.readout.weight.detach())


def test_complete_four_by_four_nyquist_mode(model):
    layer = model.spatial_layers[0]
    value = torch.zeros(1, 1, 4, 4, 64)
    value[..., 0] = torch.tensor([1., -1., 1., -1.])
    with torch.no_grad():
        layer.x_weight.zero_()
        layer.y_weight.zero_()
        layer.x_weight[0, 0, 2, 0] = 1.
        result = layer.kernel(value)
    torch.testing.assert_close(result, value, atol=1e-6, rtol=1e-6)


def test_future_fields_and_controls_do_not_change_prefix(model):
    model.eval()
    field, control = inputs()
    changed_field, changed_control = field.clone(), control.clone()
    changed_field[:, 3:] *= 3
    changed_control[:, 3:] += 7
    with torch.no_grad():
        original = model(field, control)
        changed = model(changed_field, changed_control)
    torch.testing.assert_close(original[:, :3], changed[:, :3], atol=1e-5, rtol=1e-5)


def test_stream_matches_batch_sequence(model):
    model.eval()
    field, control = inputs()
    with torch.no_grad():
        expected = model(field, control)
        state = model.reset_state(field.shape[0])
        outputs = []
        for index in range(field.shape[1]):
            output, state = model.one_step(field[:, index], control[:, index], state)
            outputs.append(output)
        actual = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_invalid_inputs_are_rejected(model):
    field, control = inputs()
    negative, nonfinite = field.clone(), field.clone()
    negative[0, 0, 0, 0, 0] = -1
    nonfinite[0, 0, 0, 0, 0] = float("nan")
    bad_control = control.clone()
    bad_control[0, 0, 0] = float("inf")
    for bad_field, bad_control in ((negative, control), (nonfinite, control), (field, bad_control),
                                    (field[..., :3], control), (field, control[:, :-1]),
                                    (field[:, :0], control[:, :0])):
        with pytest.raises(ValueError):
            model(bad_field, bad_control)
    with pytest.raises(ValueError, match="sequence state"):
        model.one_step(field[:, 0], control[:, 0], torch.zeros(1))


@pytest.mark.parametrize("bin_seconds", [0, -.01, float("nan"), float("inf"), True])
def test_invalid_transition_duration(bin_seconds):
    with pytest.raises(ValueError, match="bin_seconds"):
        MemoryNeuralOperator(2, (4, 4), 1, bin_seconds)


def test_checkpoint_rejects_changed_transition_contract(model):
    incompatible = MemoryNeuralOperator(2, (4, 4), 1, .02)
    with pytest.raises(ValueError, match="checkpoint model contract"):
        incompatible.load_state_dict(model.state_dict())

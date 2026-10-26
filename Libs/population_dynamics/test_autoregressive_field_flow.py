"""Source transform, flow gradients and explicit causal stochastic forecasts."""
import pytest
import torch

from .autoregressive_field_flow import AutoregressiveFieldFlow, SourceCountTransform


def test_zero_and_positive_count_transform_round_trip_and_folded_decoder():
    transform = SourceCountTransform([1, 50, 200], .005, [0., 1., 2.], [1., 2., 3.])
    count = torch.tensor([[0., 1., 100.], [2., 0., 8.]])
    rate = count / transform.exposure
    torch.testing.assert_close(transform.decode(transform.encode(rate)), rate, atol=1e-5, rtol=1e-6)
    z = torch.tensor([[-1., -1., -1.]])
    torch.testing.assert_close(transform.decode(z), (z * transform.scale + transform.center) ** 2 / transform.exposure)


def test_decode_samples_before_mean_and_reject_invalid_observations():
    transform = SourceCountTransform([1], 1., [0.], [1.])
    samples = torch.tensor([[-1.], [1.]])
    assert transform.decode(samples).mean().item() == 1
    assert transform.decode(samples.mean(0)).item() == 0
    for value in (torch.tensor([[-1.]]), torch.tensor([[float("nan")]])):
        with pytest.raises(ValueError):
            transform.encode(value)


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_gradient_update_and_exact_generator_replay(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = AutoregressiveFieldFlow(3, 1, 4, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history = torch.arange(18.).reshape(2, 3, 3) / 18
    past, target_control, target = torch.zeros(2, 3, 1), torch.ones(2, 1), torch.ones(2, 3)
    before = model.decoder.weight.detach().clone()
    generator = torch.Generator().manual_seed(seed)
    loss = model.matching_loss(history, past, target_control, target, generator=generator)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.decoder.weight)
    model.eval()
    controls = torch.zeros(2, 6, 1)
    first = model.forecast(history, controls, generator=torch.Generator().manual_seed(seed), bins=3, steps=4)
    second = model.forecast(history, controls, generator=torch.Generator().manual_seed(seed), bins=3, steps=4)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 3, 3)


def test_forecast_target_control_alignment_and_generated_history():
    model = AutoregressiveFieldFlow(1, 1, 3, hidden=4)
    calls = []
    def step(history, past, target_control, **kwargs):
        calls.append((history.clone(), past.clone(), target_control.clone()))
        return history[:, -1] + 100
    model.next_sample = step
    observed = torch.arange(4.).reshape(1, 4, 1)
    controls = torch.arange(7.).reshape(1, 7, 1)
    output = model.forecast(observed, controls, generator=torch.Generator().manual_seed(41), bins=3)
    assert [float(row[2].item()) for row in calls] == [4., 5., 6.]
    assert [row[0].flatten().tolist() for row in calls] == [[1., 2., 3.], [2., 3., 103.], [3., 103., 203.]]
    torch.testing.assert_close(output.flatten(), torch.tensor([103., 203., 303.]))


def test_checkpoint_contract_and_invalid_context_rejected():
    a = AutoregressiveFieldFlow(2, 1, 3, hidden=8)
    b = AutoregressiveFieldFlow(2, 1, 4, hidden=8)
    with pytest.raises(ValueError, match="architecture"):
        b.load_state_dict(a.state_dict())
    with pytest.raises(ValueError, match="context"):
        a.encode_context(torch.zeros(1, 4, 2), torch.zeros(1, 4, 1))


def test_verified_time_embedding_and_dense_dimensions():
    model = AutoregressiveFieldFlow(128, 1, 22)
    torch.testing.assert_close(model.frequencies, torch.pi * torch.arange(1, 9))
    assert model.context.bidirectional and model.context.num_layers == 4
    assert model.context.hidden_size == 128
    assert len(model.blocks) == 5
    assert model.blocks[0].in_features == 16 + 256 + 16
    assert model.blocks[0].out_features == 16
    assert model.decoder.in_features == 16 + 256

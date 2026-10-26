"""The device probe must advance warmup and change denoiser parameters."""
from __future__ import annotations

import copy

import pytest
import torch

from . import latent_spike_diffusion_device_check as probe
from .latent_spike_diffusion import PrefixCondition
from .latent_spike_diffusion_training import FIXED_PROTOCOL, _model


class ProbeModel:
    def __init__(self):
        self.denoiser = torch.nn.Linear(3, 2)

    def diffusion_loss(self, target, state, stimulus, *, noise, timestep):
        return (self.denoiser(stimulus) - target).square().mean()


def test_probe_uses_source_warmup_then_changes_parameters():
    torch.manual_seed(41)
    model = ProbeModel()
    before = [parameter.detach().clone() for parameter in model.denoiser.parameters()]
    result = probe.diffusion_updates(model, torch.zeros(1, 2), None, torch.ones(1, 3),
        copy.deepcopy(FIXED_PROTOCOL), torch.Generator().manual_seed(43000))
    assert result["optimizer_steps"] == result["scheduler_steps"] == 2
    assert result["updates"][0]["learning_rates"] == [0.]
    assert result["updates"][1]["learning_rates"] == [FIXED_PROTOCOL["learning_rate"] / (56 * 3 * 50)]
    assert result["initial_zero_rate_parameters_unchanged"]
    assert result["maximum_parameter_change"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.denoiser.parameters()))


def test_probe_rejects_scheduler_that_never_advances(monkeypatch):
    original = probe._optimizer
    def stopped(model, config, stage):
        optimizer, scheduler = original(model, config, stage)
        scheduler.step = lambda: None
        return optimizer, scheduler
    monkeypatch.setattr(probe, "_optimizer", stopped)
    with pytest.raises(ValueError, match="WARMUP_DID_NOT_ADVANCE"):
        probe.diffusion_updates(ProbeModel(), torch.zeros(1, 2), None, torch.ones(1, 3),
            copy.deepcopy(FIXED_PROTOCOL), torch.Generator().manual_seed(43000))


def test_probe_updates_source_s4_denoiser_with_unchanged_warmup():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    config = copy.deepcopy(FIXED_PROTOCOL)
    config.update(device="cpu", complete_episode_bins=32, forecast_bins=8,
        model=dict(latent_channels=8, hidden_channels=32, encoder_blocks=1,
                   denoiser_blocks=1, diffusion_steps=1000))
    cohort = type("Cohort", (), {"unit_ids": tuple(range(8))})()
    model = _model(cohort, config)
    target = torch.randn(1, 8, 8)
    state = PrefixCondition(torch.randn(1, 8, 1), model.unit_ids, 16)
    result = probe.diffusion_updates(model, target, state, torch.ones(1, 8, 1),
        config, torch.Generator().manual_seed(43000))
    assert result["changed_parameter_tensors"] > 0
    assert result["maximum_parameter_change"] > 0

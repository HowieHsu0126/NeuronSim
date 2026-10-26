"""Native count generation along a prescribed future-observation latent path.

This diagnostic changes the supplied latent path, not the observation law.
It does not establish predictive capability.
"""
import torch

from .refractory_observation import sample_bin


@torch.no_grad()
def sample_prescribed_latents(model, latents, initial_ready, observation_generator):
    """Return all unit counts and endogenous terminal clocks for a fixed path."""
    ref = model.posterior_mean.weight
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        raise ValueError("latents require [draw,batch,time,field,channel]")
    draws, batch, steps, fields, channels = latents.shape
    units = len(model.unit_ids)
    if min(draws, batch, steps) < 1 or (fields, channels) != (len(model.field_names), model.latent_channels):
        raise ValueError("latent dimensions differ from the model")
    if latents.device != ref.device or latents.dtype != ref.dtype or not torch.isfinite(latents).all():
        raise ValueError("finite latents on the model device and dtype required")
    if (not isinstance(initial_ready, torch.Tensor) or initial_ready.dtype not in (torch.int32, torch.int64)
            or initial_ready.device != ref.device):
        raise ValueError("initial_ready requires integer ticks on the model device")
    if (not isinstance(observation_generator, torch.Generator)
            or torch.device(observation_generator.device) != ref.device):
        raise ValueError("explicit observation generator on the model device required")
    if initial_ready.shape == (batch, units):
        ready = model._ready(initial_ready, batch).expand(draws, -1, -1).clone()
    elif initial_ready.shape == (draws, batch, units):
        ready = model._ready(initial_ready.reshape(draws * batch, units), draws * batch).reshape(draws, batch, units).clone()
    else:
        raise ValueError("initial_ready requires [batch,unit] or [draw,batch,unit]")
    counts = []
    for step in range(steps):
        sampled = sample_bin(model.log_hazard(latents[:, :, step]), ready, model.minimum_interval_ticks,
                             model.bin_ticks, model.tick_seconds, observation_generator, record_events=False)
        counts.append(sampled.counts)
        ready = sampled.ready_in_ticks
    return torch.stack(counts, dim=2), ready

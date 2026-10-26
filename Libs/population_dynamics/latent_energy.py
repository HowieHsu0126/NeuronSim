"""EAG count model with a separately encoded observed prefix.

The autoencoder uses the paper's deeper bidirectional S4 architecture through
the already vendored MIT LDNS primitives. Its weights require new training.
Future targets never enter ``infer_state`` or ``simulate``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ldns.networks.blocks import AutoEncoder
from ldns.networks.count_wrapper import CountWrapper

from .latent_energy_generator import LatentEnergyGenerator
from .latent_spike_diffusion import SOURCE_REVISION, verify_source


@dataclass(frozen=True)
class EnergyPrefixCondition:
    latent: torch.Tensor
    controls: torch.Tensor
    unit_ids: tuple[int, ...]


class ControlledLatentEnergy(nn.Module):
    """Full-unit Poisson observations of a conditional latent energy generator."""

    def __init__(self, unit_ids, *, input_size=1, bin_seconds=.005,
                 complete_episode_bins=1200, maximum_prefix_bins=1000, forecast_bins=100,
                 latent_channels=16, hidden_channels=256, encoder_blocks=4,
                 decoder_blocks=4, linear_layers_per_block=4, generator_options=None):
        super().__init__()
        verify_source()
        ids = tuple(unit_ids)
        dimensions = (input_size, complete_episode_bins, maximum_prefix_bins, forecast_bins,
                      latent_channels, hidden_channels, encoder_blocks, decoder_blocks,
                      linear_layers_per_block)
        if (not ids or len(set(ids)) != len(ids) or any(type(x) is not int for x in ids)
                or any(type(x) is not int or x < 1 for x in dimensions)
                or linear_layers_per_block < 2 or maximum_prefix_bins < 2 or forecast_bins < 2
                or maximum_prefix_bins + forecast_bins > complete_episode_bins
                or isinstance(bin_seconds, bool) or not math.isfinite(bin_seconds) or bin_seconds <= 0):
            raise ValueError("identified units, positive clock and complete source-window dimensions required")
        self.unit_ids, self.input_size, self.bin_seconds = ids, input_size, bin_seconds
        self.complete_episode_bins = complete_episode_bins
        self.maximum_prefix_bins, self.forecast_bins = maximum_prefix_bins, forecast_bins
        self.latent_channels = latent_channels
        self.autoencoder = CountWrapper(AutoEncoder(
            C_in=len(ids), C=hidden_channels, C_latent=latent_channels, L=1,
            num_blocks=encoder_blocks, num_blocks_decoder=decoder_blocks,
            num_lin_per_mlp=linear_layers_per_block, bidirectional=True))
        self.energy = LatentEnergyGenerator(latent_channels, input_size,
            maximum_prefix_bins + forecast_bins, **(generator_options or {}))
        self.register_buffer("latent_mean", torch.zeros(1, latent_channels, 1))
        self.register_buffer("latent_sd", torch.ones(1, latent_channels, 1))
        self.scale_fitted = False
        self.configuration = dict(input_size=input_size, bin_seconds=bin_seconds,
            complete_episode_bins=complete_episode_bins, maximum_prefix_bins=maximum_prefix_bins,
            forecast_bins=forecast_bins, latent_channels=latent_channels, hidden_channels=hidden_channels,
            encoder_blocks=encoder_blocks, decoder_blocks=decoder_blocks,
            linear_layers_per_block=linear_layers_per_block, bidirectional=True,
            generator=self.energy.configuration)
        # S4 changes its C representation when first encountering a longer
        # sequence. Prepare both networks before an optimizer or checkpoint exists.
        for block in (*self.autoencoder.ae_net.encoder, *self.autoencoder.ae_net.decoder):
            block.time_mixer.kernel._setup_C(complete_episode_bins)

    def get_extra_state(self):
        return dict(unit_ids=self.unit_ids, configuration=self.configuration,
                    primitive_source_revision=SOURCE_REVISION, scale_fitted=self.scale_fitted)

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if any(state.get(key) != expected[key] for key in
               ("unit_ids", "configuration", "primitive_source_revision")):
            raise ValueError("EAG checkpoint unit, clock, source or architecture changed")
        if type(state.get("scale_fitted")) is not bool:
            raise ValueError("explicit latent-scale fitted state required")
        self.scale_fitted = state["scale_fitted"]

    def _counts(self, values):
        if (not isinstance(values, torch.Tensor) or values.ndim != 3
                or values.shape[0] < 1 or values.shape[2] != len(self.unit_ids)
                or not 2 <= values.shape[1] <= self.complete_episode_bins
                or not torch.isfinite(values).all() or torch.any(values < 0)
                or torch.any(values != values.round())):
            raise ValueError("complete finite integer counts [batch,time,identified_units] required")
        return values.transpose(1, 2).to(self.latent_mean)

    def autoencoder_loss(self, counts, *, mask, mask_probability=.5,
                         latent_beta=.001, temporal_beta=.2):
        data = self._counts(counts)
        if (mask.shape != data.shape or mask.dtype != torch.bool or mask.device != data.device
                or not torch.any(mask) or not torch.any(~mask) or not 0 < mask_probability < 1
                or any(not math.isfinite(x) or x < 0 for x in (latent_beta, temporal_beta))):
            raise ValueError("matched coordinated-dropout mask and nonnegative regularizers required")
        rates, latent = self.autoencoder(data * mask / (1 - mask_probability))
        reconstruction = (F.poisson_nll_loss(rates, data, log_input=False, reduction="none") * ~mask).mean()
        latent_l2 = latent.square().sum() / data.numel()
        temporal = sum((latent[:, :, :-lag] - latent[:, :, lag:]).square().sum() / (1 + lag)
                       for lag in range(1, min(6, latent.shape[-1]))) / data.numel()
        loss = reconstruction + latent_beta * latent_l2 + temporal_beta * temporal
        return dict(loss=loss, reconstruction_nll=reconstruction,
                    latent_l2=latent_l2, latent_temporal_difference=temporal)

    @torch.no_grad()
    def set_latent_scale(self, mean, sd):
        mean, sd = torch.as_tensor(mean).to(self.latent_mean), torch.as_tensor(sd).to(self.latent_sd)
        if (mean.shape != self.latent_mean.shape or sd.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(sd).all() or torch.any(sd <= 0)):
            raise ValueError("finite training-only latent mean and positive standard deviation required")
        self.latent_mean.copy_(mean)
        self.latent_sd.copy_(sd)
        self.scale_fitted = True

    @torch.no_grad()
    def encode_target(self, future_counts):
        if not self.scale_fitted:
            raise ValueError("training-only latent scale required")
        if future_counts.shape[1] != self.forecast_bins:
            raise ValueError("registered complete forecast target required")
        latent = self.autoencoder.encode(self._counts(future_counts))
        return ((latent - self.latent_mean) / self.latent_sd).transpose(1, 2)

    @torch.no_grad()
    def infer_state(self, observed_counts, observed_controls):
        if not self.scale_fitted:
            raise ValueError("training-only latent scale required")
        data = self._counts(observed_counts)
        if (data.shape[2] > self.maximum_prefix_bins
                or observed_controls.shape != (data.shape[0], data.shape[2], self.input_size)
                or not torch.isfinite(observed_controls).all()):
            raise ValueError("observed-prefix controls and counts must share their clock")
        # Bidirectional S4 and temporal normalization receive only observations.
        latent = self.autoencoder.encode(data)
        return EnergyPrefixCondition(((latent - self.latent_mean) / self.latent_sd).transpose(1, 2),
                                     observed_controls.to(self.latent_mean).clone(), self.unit_ids)

    @torch.no_grad()
    def simulate(self, state, stimulus, *, latent_rng, observation_rng,
                 steps=50, guidance=4., temperature=.7):
        if not self.scale_fitted or state.unit_ids != self.unit_ids:
            raise ValueError("fitted same-unit prefix condition required")
        if (stimulus.shape != (state.latent.shape[0], self.forecast_bins, self.input_size)
                or not torch.isfinite(stimulus).all()):
            raise ValueError("complete known future stimulus on the registered clock required")
        controls = torch.cat((state.controls, stimulus.to(self.latent_mean)), dim=1)
        latent = self.energy.sample(state.latent, controls, self.forecast_bins,
                                   generator=latent_rng, steps=steps, guidance=guidance, temperature=temperature)
        rates = self.autoencoder.decode(latent.transpose(1, 2) * self.latent_sd + self.latent_mean).transpose(1, 2)
        if not torch.isfinite(rates).all() or torch.any(rates < 0):
            raise FloatingPointError("nonfinite or negative EAG Poisson observation intensity")
        counts = torch.poisson(rates, generator=observation_rng).to(torch.int64)
        return dict(counts=counts, expected_counts=rates, latent=latent.transpose(1, 2),
                    unit_ids=self.unit_ids, bin_seconds=self.bin_seconds,
                    native_refractory_support_qualified=False)

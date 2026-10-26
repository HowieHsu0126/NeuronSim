"""Source LDNS adapted to observed-prefix and future-stimulus conditioning.

The author autoencoder and joint S4 denoiser are retained. The prefix is encoded
separately because source temporal normalization reads the full supplied input.
Counts use the source Poisson law; native-clock renewal support is not implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Vendored blocks uses the author's absolute ldns import. The selected runner
# places Libs/external/LDNS on PYTHONPATH; package intake remains framework-free.
from ldns.networks.blocks import AutoEncoder, Denoiser, DenoiserBlock
from ldns.networks.count_wrapper import CountWrapper

SOURCE_REVISION = "mackelab/LDNS@24d4a0c1a55aa364b6dfcf8916b18cf7501d7ef9"


def verify_source():
    root = Path(__file__).parents[1] / "external/LDNS"
    record = json.loads((root / "source.json").read_text())
    if record["commit"] != SOURCE_REVISION.rsplit("@", 1)[1]:
        raise ValueError("pinned LDNS source revision required")
    for name, expected in record["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("pinned LDNS source changed: " + name)


@dataclass(frozen=True)
class PrefixCondition:
    latent: torch.Tensor
    unit_ids: tuple[int, ...]
    observed_bins: int


class ControlledLatentSpikeDiffusion(nn.Module):
    """One joint latent diffusion trajectory with all identified unit outputs."""

    def __init__(self, unit_ids, *, input_size, bin_seconds, latent_channels=16,
                 hidden_channels=256, encoder_blocks=4, denoiser_blocks=6,
                 diffusion_steps=1000):
        super().__init__()
        verify_source()
        ids = tuple(unit_ids)
        if (not ids or len(set(ids)) != len(ids) or any(type(value) is not int for value in ids)
                or type(input_size) is not int or input_size < 1
                or not math.isfinite(bin_seconds) or not 0 < bin_seconds
                or any(type(value) is not int or value < 1 for value in
                       (latent_channels, hidden_channels, encoder_blocks, denoiser_blocks, diffusion_steps))
                or diffusion_steps < 2 or hidden_channels % 4):
            raise ValueError("identified units, positive clock and supported model dimensions required")
        self.unit_ids, self.input_size, self.bin_seconds = ids, input_size, bin_seconds
        self.latent_channels, self.diffusion_steps = latent_channels, diffusion_steps
        self.autoencoder = CountWrapper(AutoEncoder(
            C_in=len(ids), C=hidden_channels, C_latent=latent_channels, L=1,
            num_blocks=encoder_blocks, num_blocks_decoder=0, bidirectional=False))
        channels = 2 * latent_channels + input_size
        self.denoiser = Denoiser(C_in=channels, C=hidden_channels, L=1,
                                 num_blocks=0, in_groups=1, bidirectional=True)
        # Correct the pinned wrapper's positional mismatch with explicit keywords.
        # Joint future denoising remains bidirectional, as in the author notebook.
        self.denoiser.blocks = nn.ModuleList([
            DenoiserBlock(C=hidden_channels, L=1, bidirectional=True, kernel_params=None)
            for _ in range(denoiser_blocks)])
        betas = torch.linspace(.0001, .02, diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, 0))
        self.register_buffer("latent_mean", torch.zeros(1, latent_channels, 1))
        self.register_buffer("latent_sd", torch.ones(1, latent_channels, 1))
        self.scale_fitted = False

    def get_extra_state(self):
        return dict(unit_ids=self.unit_ids, input_size=self.input_size, bin_seconds=self.bin_seconds,
                    source_revision=SOURCE_REVISION, scale_fitted=self.scale_fitted)

    def set_extra_state(self, state):
        if (tuple(state["unit_ids"]) != self.unit_ids or state["input_size"] != self.input_size
                or state["bin_seconds"] != self.bin_seconds or state["source_revision"] != SOURCE_REVISION):
            raise ValueError("LDNS checkpoint source/unit/control/clock identity differs")
        self.scale_fitted = state["scale_fitted"]

    def _counts(self, values):
        if (values.ndim != 3 or values.shape[-1] != len(self.unit_ids) or values.shape[1] < 2
                or not torch.isfinite(values).all() or torch.any(values < 0)
                or torch.any(values != values.round())):
            raise ValueError("complete finite integer observed counts [batch,time,units] required")
        return values.transpose(1, 2).to(self.betas)

    def autoencoder_loss(self, counts, *, mask, mask_probability=.5, latent_beta=.001, temporal_beta=.2):
        data = self._counts(counts)
        if (mask.shape != data.shape or mask.dtype != torch.bool
                or not torch.any(~mask) or not torch.any(mask)
                or not 0 < mask_probability < 1):
            raise ValueError("explicit nonempty observed and masked autoencoder cells required")
        rates, latent = self.autoencoder(data * mask / (1-mask_probability))
        reconstruction = (F.poisson_nll_loss(rates, data, log_input=False, reduction="none") * ~mask).mean()
        temporal = sum((latent[:, :, :-lag] - latent[:, :, lag:]).square().sum() / (1 + lag)
                       for lag in range(1, min(6, latent.shape[-1])))
        latent_l2 = latent.square().sum() / data.numel()
        temporal = temporal / data.numel()
        loss = reconstruction + latent_beta * latent_l2 + temporal_beta * temporal
        return dict(loss=loss, reconstruction_nll=reconstruction, latent_l2=latent_l2,
                    latent_temporal_difference=temporal)

    @torch.no_grad()
    def set_latent_scale(self, mean, sd):
        mean, sd = torch.as_tensor(mean).to(self.betas), torch.as_tensor(sd).to(self.betas)
        if (mean.shape != self.latent_mean.shape or sd.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(sd).all() or torch.any(sd <= 0)):
            raise ValueError("finite fitting-only latent mean and positive standard deviation required")
        self.latent_mean.copy_(mean); self.latent_sd.copy_(sd); self.scale_fitted = True

    @torch.no_grad()
    def encode_target(self, future_counts):
        """Encode training targets only; this tensor is never a prefix condition."""
        if not self.scale_fitted:
            raise ValueError("fitting-only latent scale required")
        return (self.autoencoder.encode(self._counts(future_counts)) - self.latent_mean) / self.latent_sd

    @torch.no_grad()
    def infer_state(self, observed_counts):
        if not self.scale_fitted:
            raise ValueError("fitting-only latent scale required")
        prefix = self.autoencoder.encode(self._counts(observed_counts))
        # Encoding receives only actual observations, including its normalization.
        summary = (prefix[:, :, -1:] - self.latent_mean) / self.latent_sd
        return PrefixCondition(summary, self.unit_ids, observed_counts.shape[1])

    def _noise_prediction(self, noisy_latent, timestep, state, stimulus):
        if (state.unit_ids != self.unit_ids or noisy_latent.ndim != 3
                or noisy_latent.shape[1] != self.latent_channels
                or state.latent.shape != (noisy_latent.shape[0], self.latent_channels, 1)
                or stimulus.shape != (noisy_latent.shape[0], noisy_latent.shape[2], self.input_size)
                or not torch.isfinite(stimulus).all() or not torch.isfinite(state.latent).all()):
            raise ValueError("matched observed prefix and finite future stimulus required")
        condition = state.latent.to(noisy_latent).expand(-1, -1, noisy_latent.shape[-1])
        values = torch.cat([noisy_latent, condition, stimulus.transpose(1,2).to(noisy_latent)],1)
        return self.denoiser(values,timestep.to(values.device))[:, :self.latent_channels]

    def diffusion_loss(self, future_latent, state, stimulus, *, noise, timestep):
        if (noise.shape != future_latent.shape or not torch.isfinite(noise).all()
                or timestep.shape != (future_latent.shape[0],) or timestep.dtype != torch.long
                or torch.any(timestep < 0) or torch.any(timestep >= self.diffusion_steps)):
            raise ValueError("explicit finite diffusion noise and valid timesteps required")
        alpha = self.alpha_bar[timestep].reshape(-1,1,1)
        noisy = alpha.sqrt() * future_latent + (1-alpha).sqrt() * noise
        prediction = self._noise_prediction(noisy,timestep,state,stimulus)
        return F.smooth_l1_loss(prediction,noise,beta=.05)

    @torch.no_grad()
    def simulate(self, state, stimulus, *, latent_rng, observation_rng):
        """Sample the joint future latent path, then source Poisson unit counts."""
        if not self.scale_fitted:
            raise ValueError("fitting-only latent scale required")
        shape = (stimulus.shape[0],self.latent_channels,stimulus.shape[1])
        latent = torch.randn(shape,device=self.betas.device,dtype=self.betas.dtype,generator=latent_rng)
        for step in range(self.diffusion_steps-1,-1,-1):
            timestep = torch.full((shape[0],),step,device=latent.device,dtype=torch.long)
            noise = self._noise_prediction(latent,timestep,state,stimulus)
            alpha, beta = self.alpha_bar[step], self.betas[step]
            previous = self.alpha_bar[step-1] if step else alpha.new_tensor(1.)
            clean = (latent-(1-alpha).sqrt()*noise)/alpha.sqrt()
            mean = previous.sqrt()*beta/(1-alpha)*clean + (1-beta).sqrt()*(1-previous)/(1-alpha)*latent
            if step:
                innovation = torch.randn(shape,device=latent.device,dtype=latent.dtype,generator=latent_rng)
                mean = mean + (beta*(1-previous)/(1-alpha)).sqrt()*innovation
            latent = mean
        rates = self.autoencoder.decode(latent*self.latent_sd+self.latent_mean).transpose(1,2)
        if not torch.isfinite(rates).all() or torch.any(rates < 0):
            raise FloatingPointError("nonfinite LDNS decoded count rates")
        counts = torch.poisson(rates,generator=observation_rng).to(torch.int64)
        return dict(counts=counts,expected_counts=rates,latent=latent,unit_ids=self.unit_ids,
                    bin_seconds=self.bin_seconds,observation_law="source_poisson_counts")

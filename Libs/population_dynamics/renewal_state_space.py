"""Controlled shared Gaussian latent dynamics with native-clock renewal outputs.

Reuses the existing Poisson LDS's latent dynamics and causal encoder. Only the
observation law changes. This is a physical-support control, not an established
novel closure representation. Poisson forecast moments are deliberately absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .poisson_state_space import (ControlledPoissonLDS, GaussianPopulationState,
                                  LatentRandomStreams, initial_kl, transition_kl)
from .refractory_observation import observation_nll, sample_bin


@dataclass(frozen=True)
class RenewalPopulationState:
    latent: GaussianPopulationState
    ready_in_ticks: Tensor  # batch, unit; relative to the next bin's first tick


@dataclass(frozen=True)
class RenewalSamples:
    counts: Tensor  # draw,batch,time,unit
    # Sparse rows: draw,batch,unit_index,relative native emission tick.
    events: Tensor
    latent_states: Tensor
    initial_latent_states: Tensor
    terminal_ready_in_ticks: Tensor
    unit_ids: tuple[int, ...]
    tick_seconds: float
    bin_ticks: int
    events_recorded: bool = True


class ControlledRenewalLDS(nn.Module):
    def __init__(self, *, minimum_interval_ticks, tick_seconds: float, **latent_config):
        super().__init__()
        self.latent_model = ControlledPoissonLDS(**latent_config)
        interval = torch.as_tensor(minimum_interval_ticks)
        if (interval.shape != (len(self.latent_model.unit_ids),)
                or interval.dtype not in (torch.int32, torch.int64) or torch.any(interval < 1)
                or not 0 < tick_seconds <= self.latent_model.bin_seconds):
            raise ValueError("positive native tick and identified unit minimum intervals required")
        ratio = self.latent_model.bin_seconds / tick_seconds
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("model bins must align to the native event clock")
        self.tick_seconds, self.bin_ticks = float(tick_seconds), round(ratio)
        self.register_buffer("minimum_interval_ticks", interval.to(torch.int64))

    def _ready(self, values: Any, batch: int) -> Tensor:
        result = torch.as_tensor(values, device=self.minimum_interval_ticks.device)
        if (result.shape != (batch, len(self.latent_model.unit_ids))
                or result.dtype not in (torch.int32, torch.int64)
                or torch.any(result < 0) or torch.any(result >= self.minimum_interval_ticks)):
            raise ValueError("observed boundary availability must match unit identity and interval")
        return result.to(torch.int64)

    def infer_state(self, history: dict, metadata: dict | None = None) -> RenewalPopulationState:
        latent = self.latent_model.infer_state(history, metadata)
        if "ready_in_ticks" not in history:
            raise ValueError("observed last-spike boundary availability required")
        return RenewalPopulationState(latent, self._ready(history["ready_in_ticks"], latent.mean.shape[0]))

    def sequence_loss(self, counts: Any, stimulus: Any, eligible_ticks: Any,
                      generator: torch.Generator) -> dict[str, Tensor]:
        """One reparameterized observation draw; latent KL terms remain analytic.

        The explicit generator belongs to optimization/validation and must be
        checkpointed by a training runner. This stochastic ELBO estimate is not
        a held-out forecast likelihood or an exact Gaussian expectation.
        """
        model = self.latent_model
        y, u = model._sequence(counts, stimulus)
        exposure = model._tensor(eligible_ticks)
        if exposure.shape != y.shape or torch.any(exposure > self.bin_ticks):
            raise ValueError("eligible observation exposure differs from model bins")
        if torch.device(generator.device) != y.device:
            raise ValueError("variational RNG must be on the model device")
        mean, variance = model._encode(y, u)
        z = mean + variance.sqrt() * torch.randn(mean.shape, dtype=mean.dtype,
                                                device=mean.device, generator=generator)
        p0, process = model.initial_log_variance.exp(), model.process_log_variance.exp()
        model._positive_finite(p0); model._positive_finite(process)
        normalizer = y.shape[0] * y.shape[1]
        observation = observation_nll(y, exposure, z @ model.loading.T + model.log_rate_bias,
                                      self.tick_seconds).sum() / normalizer
        initial = initial_kl(mean[:, 0], variance[:, 0],
                             model.initial_mean + u[:, 0] @ model.input_weight.T, p0).sum() / normalizer
        transition = transition_kl(mean[:, 1:], variance[:, 1:], mean[:, :-1], variance[:, :-1],
                                    model.transition, u[:, 1:] @ model.input_weight.T + model.transition_bias,
                                    process).sum() / normalizer
        loss = observation + initial + transition
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite renewal negative ELBO estimate")
        return dict(loss=loss, sampled_observation_nll=observation,
                    initial_kl=initial, transition_kl=transition)

    @torch.no_grad()
    def simulate(self, state: RenewalPopulationState, stimulus: Any,
                 rng: LatentRandomStreams, n_samples: int, *, record_events: bool = True) -> RenewalSamples:
        model = self.latent_model
        mean, variance, u = model._forecast_inputs(state.latent, stimulus)
        if type(n_samples) is not int or n_samples < 1 or not isinstance(rng, LatentRandomStreams):
            raise ValueError("positive sample count and distinct latent random streams required")
        streams = (rng.initial, rng.dynamics, rng.observation)
        if (any(torch.device(g.device) != mean.device for g in streams)
                or len({id(g) for g in streams}) != 3 or len({g.initial_seed() for g in streams}) != 3):
            raise ValueError("distinct random streams on the model device required")
        batch, units = mean.shape[0], len(model.unit_ids)
        ready = self._ready(state.ready_in_ticks, batch).unsqueeze(0).expand(n_samples, -1, -1).clone()
        shape = (n_samples, batch, model.latent_size)
        z = mean + variance.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.initial)
        initial = z.clone()
        process = model.process_log_variance.exp()
        model._positive_finite(process)
        latent, counts, events = [], [], []
        for t in range(u.shape[1]):
            z = (z @ model.transition.T + u[:, t] @ model.input_weight.T + model.transition_bias
                 + process.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.dynamics))
            sampled = sample_bin(z @ model.loading.T + model.log_rate_bias, ready,
                                 self.minimum_interval_ticks, self.bin_ticks, self.tick_seconds, rng.observation,
                                 record_events=record_events)
            flat, within = sampled.events[:, 0], sampled.events[:, 1]
            events.append(torch.stack((flat // (batch * units), (flat // units) % batch,
                                       flat % units, within + t * self.bin_ticks), dim=1))
            ready = sampled.ready_in_ticks
            latent.append(z); counts.append(sampled.counts)
        return RenewalSamples(torch.stack(counts, dim=2), torch.cat(events, dim=0),
                              torch.stack(latent, dim=2), initial, ready,
                              model.unit_ids, self.tick_seconds, self.bin_ticks, record_events)

    def save(self, path: Path) -> None:
        model = self.latent_model
        config = dict(unit_ids=model.unit_ids, input_size=model.input_size, latent_size=model.latent_size,
                      encoder_size=model.encoder_size, bin_seconds=model.bin_seconds,
                      initialization_seed=model.initialization_seed, tick_seconds=self.tick_seconds,
                      minimum_interval_ticks=self.minimum_interval_ticks.tolist())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(dict(schema="neural_population_controlled_renewal_lds_v1",
                            configuration=config, state_dict=self.state_dict()), stream)

    @classmethod
    def load(cls, path: Path, *, device="cpu") -> ControlledRenewalLDS:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema"] != "neural_population_controlled_renewal_lds_v1":
            raise ValueError("renewal LDS schema mismatch")
        model = cls(**payload["configuration"]).to(device=device,
                    dtype=payload["state_dict"]["latent_model.loading"].dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

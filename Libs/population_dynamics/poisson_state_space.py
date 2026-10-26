"""Controlled Poisson LDS baseline with causal, factorized variational inference.

The generative model is linear Gaussian in a shared latent state and Poisson in
counts, with rate in Hz and explicit bin width. The causal encoder is a campaign
adaptation, not a reproduction of a published inference implementation. Its joint
q factorizes over latent times even though each q's parameters use the observed
prefix. Latent coordinates have no assumed cell-type or biophysical identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn


def expected_observation_nll(y: Tensor, mean: Tensor, variance: Tensor,
                             loading: Tensor, bias: Tensor, bin_seconds: float) -> Tensor:
    """E_q[-log p(y|z)] summed over units, preserving batch/time dimensions."""
    log_rate = mean @ loading.T + bias
    log_rate_variance = variance @ loading.square().T
    return (bin_seconds * (log_rate + 0.5 * log_rate_variance).exp()
            - y * (math.log(bin_seconds) + log_rate) + torch.lgamma(y + 1)).sum(-1)


def initial_kl(mean: Tensor, variance: Tensor, prior_mean: Tensor,
               prior_variance: Tensor) -> Tensor:
    """KL between diagonal Gaussian initial q and initial prior."""
    return 0.5 * ((prior_variance.log() - variance.log() - 1)
                  + (variance + (mean - prior_mean).square()) / prior_variance).sum(-1)


def transition_kl(mean: Tensor, variance: Tensor, previous_mean: Tensor,
                  previous_variance: Tensor, transition: Tensor, drive: Tensor,
                  process_variance: Tensor) -> Tensor:
    """E_q(previous)[KL(q(current)||p(current|previous,u))], factorized q."""
    residual = mean - previous_mean @ transition.T - drive
    propagated_variance = previous_variance @ transition.square().T
    return 0.5 * ((process_variance.log() - variance.log() - 1)
                  + (variance + propagated_variance + residual.square()) / process_variance).sum(-1)


@dataclass(frozen=True)
class GaussianPopulationState:
    mean: Tensor  # batch, latent dimension; after observing the last history bin
    variance: Tensor  # diagonal boundary covariance
    unit_ids: tuple[int, ...]
    bin_seconds: float
    uncertainty: str = "causal_factorized_variational_boundary"


@dataclass(frozen=True)
class GaussianForecast:
    mean: Tensor  # batch, future bin, latent dimension
    covariance: Tensor  # batch, future bin, latent dimension, latent dimension
    expected_counts: Tensor  # exact marginal means under the approximate boundary


@dataclass(frozen=True)
class LatentRandomStreams:
    initial: torch.Generator
    dynamics: torch.Generator
    observation: torch.Generator

    @classmethod
    def from_seeds(cls, initial: int, dynamics: int, observation: int,
                   *, device: str | torch.device = "cpu") -> LatentRandomStreams:
        seeds = (initial, dynamics, observation)
        if any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != 3:
            raise ValueError("THREE_DISTINCT_NONNEGATIVE_RANDOM_STREAM_SEEDS_REQUIRED")
        resolved = torch.device(device)
        # CUDA's unindexed alias can remain unindexed on Generator.device while
        # tensors report cuda:<index>. Bind the actual device at construction.
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        return cls(*(torch.Generator(device=resolved).manual_seed(s) for s in seeds))


@dataclass(frozen=True)
class LatentCountSamples:
    counts: Tensor  # draw, batch, future bin, unit
    conditional_expected_counts: Tensor  # conditional on the sampled latent path
    latent_states: Tensor  # draw, batch, future bin, latent dimension
    initial_latent_states: Tensor  # draw, batch, latent dimension
    unit_ids: tuple[int, ...]
    bin_seconds: float

    @property
    def population_counts(self) -> Tensor:
        return self.counts.sum(-1)


class ControlledPoissonLDS(nn.Module):
    """Known-input shared-latent count baseline; no automatic stability guarantee."""

    def __init__(self, unit_ids: Sequence[int], input_size: int, latent_size: int,
                 encoder_size: int, bin_seconds: float, *, initialization_seed: int):
        super().__init__()
        ids = tuple(unit_ids)
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("UNIQUE_NONNEGATIVE_INTEGER_UNIT_IDS_REQUIRED")
        if (type(input_size) is not int or input_size < 0
                or any(type(s) is not int or s < 1 for s in (latent_size, encoder_size))):
            raise ValueError("INVALID_INPUT_LATENT_OR_ENCODER_SIZE")
        if type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("POSITIVE_BIN_SECONDS_REQUIRED")
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
            raise ValueError("NONNEGATIVE_INITIALIZATION_SEED_REQUIRED")
        self.unit_ids, self.input_size = ids, input_size
        self.latent_size, self.encoder_size = latent_size, encoder_size
        self.bin_seconds, self.initialization_seed = float(bin_seconds), initialization_seed
        d, h, n = latent_size, encoder_size, len(ids)
        self.transition = nn.Parameter(0.8 * torch.eye(d))
        self.input_weight = nn.Parameter(torch.empty(d, input_size))
        self.transition_bias = nn.Parameter(torch.zeros(d))
        self.initial_mean = nn.Parameter(torch.zeros(d))
        self.initial_log_variance = nn.Parameter(torch.zeros(d))
        self.process_log_variance = nn.Parameter(torch.full((d,), math.log(0.1)))
        self.loading = nn.Parameter(torch.empty(n, d))
        self.log_rate_bias = nn.Parameter(torch.zeros(n))
        self.encoder_observation = nn.Parameter(torch.empty(h, n))
        self.encoder_input = nn.Parameter(torch.empty(h, input_size))
        self.encoder_recurrent = nn.Parameter(0.5 * torch.eye(h))
        self.encoder_bias = nn.Parameter(torch.zeros(h))
        self.encoder_mean_weight = nn.Parameter(torch.empty(d, h))
        self.encoder_mean_bias = nn.Parameter(torch.zeros(d))
        self.encoder_variance_weight = nn.Parameter(torch.empty(d, h))
        self.encoder_log_variance_bias = nn.Parameter(torch.full((d,), math.log(0.2)))
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        for parameter in (self.input_weight, self.loading, self.encoder_observation,
                          self.encoder_input, self.encoder_mean_weight, self.encoder_variance_weight):
            nn.init.normal_(parameter, std=0.05, generator=generator)

    def _tensor(self, values: Any) -> Tensor:
        raw = torch.as_tensor(values, device=self.loading.device)
        if raw.is_complex():
            raise ValueError("REAL_VALUES_REQUIRED")
        return raw.to(dtype=self.loading.dtype)

    @staticmethod
    def _positive_finite(variance: Tensor) -> None:
        if not torch.isfinite(variance).all() or torch.any(variance <= 0):
            raise FloatingPointError("FINITE_POSITIVE_VARIANCE_REQUIRED")

    def _sequence(self, counts: Any, stimulus: Any) -> tuple[Tensor, Tensor]:
        y, u = self._tensor(counts), self._tensor(stimulus)
        if y.ndim != 3 or min(y.shape) < 1 or y.shape[-1] != len(self.unit_ids):
            raise ValueError("COUNT_SEQUENCE_SHAPE_MISMATCH")
        if not torch.isfinite(y).all() or torch.any(y < 0) or torch.any(y != y.round()):
            raise ValueError("FINITE_NONNEGATIVE_INTEGER_COUNTS_REQUIRED")
        if u.shape != (*y.shape[:2], self.input_size) or not torch.isfinite(u).all():
            raise ValueError("KNOWN_STIMULUS_SEQUENCE_SHAPE_OR_VALUES_INVALID")
        return y, u

    def _encode(self, y: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        hidden = y.new_zeros((y.shape[0], self.encoder_size))
        means, variances = [], []
        for t in range(y.shape[1]):
            hidden = torch.tanh(torch.log1p(y[:, t]) @ self.encoder_observation.T
                                + u[:, t] @ self.encoder_input.T
                                + hidden @ self.encoder_recurrent.T + self.encoder_bias)
            means.append(hidden @ self.encoder_mean_weight.T + self.encoder_mean_bias)
            variances.append((hidden @ self.encoder_variance_weight.T
                              + self.encoder_log_variance_bias).exp())
        mean, variance = torch.stack(means, 1), torch.stack(variances, 1)
        if not torch.isfinite(mean).all():
            raise FloatingPointError("NONFINITE_ENCODER_MEAN")
        self._positive_finite(variance)
        return mean, variance

    def encode(self, counts: Any, stimulus: Any) -> tuple[Tensor, Tensor]:
        """q_t sees observations and known input through t; never later bins."""
        return self._encode(*self._sequence(counts, stimulus))

    def infer_state(self, history: dict[str, Any], metadata: dict[str, Any] | None = None) -> GaussianPopulationState:
        """History supplies counts AND aligned known stimuli, each batch,time,feature."""
        if not isinstance(history, dict) or not {"counts", "stimulus"} <= history.keys():
            raise ValueError("COUNT_AND_KNOWN_STIMULUS_HISTORY_REQUIRED")
        if metadata is not None:
            if tuple(metadata.get("unit_ids", self.unit_ids)) != self.unit_ids:
                raise ValueError("HISTORY_UNIT_ORDER_MISMATCH")
            if metadata.get("bin_seconds", self.bin_seconds) != self.bin_seconds:
                raise ValueError("HISTORY_BIN_WIDTH_MISMATCH")
        mean, variance = self.encode(history["counts"], history["stimulus"])
        return GaussianPopulationState(mean[:, -1], variance[:, -1], self.unit_ids, self.bin_seconds)

    def sequence_loss(self, counts: Any, stimulus: Any) -> dict[str, Tensor]:
        """Negative ELBO, all terms summed then divided by batch*time.

        This is a training/smoothing-density bound, not a forecasting score.
        Reconstruction conditions on current counts; free forecasts do not.
        """
        y, u = self._sequence(counts, stimulus)
        mean, variance = self._encode(y, u)
        p0, q = self.initial_log_variance.exp(), self.process_log_variance.exp()
        self._positive_finite(p0); self._positive_finite(q)
        normalizer = y.shape[0] * y.shape[1]
        observation = expected_observation_nll(y, mean, variance, self.loading,
                                               self.log_rate_bias, self.bin_seconds).sum() / normalizer
        initial = initial_kl(mean[:, 0], variance[:, 0],
                             self.initial_mean + u[:, 0] @ self.input_weight.T, p0).sum() / normalizer
        transition = transition_kl(mean[:, 1:], variance[:, 1:], mean[:, :-1], variance[:, :-1],
                                    self.transition, u[:, 1:] @ self.input_weight.T + self.transition_bias,
                                    q).sum() / normalizer
        loss = observation + initial + transition
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_NEGATIVE_ELBO")
        return {"loss": loss, "expected_observation_nll": observation,
                "initial_kl": initial, "transition_kl": transition}

    def _forecast_inputs(self, state: GaussianPopulationState, stimulus: Any) -> tuple[Tensor, Tensor, Tensor]:
        if state.unit_ids != self.unit_ids or state.bin_seconds != self.bin_seconds:
            raise ValueError("STATE_UNIT_ORDER_OR_BIN_WIDTH_MISMATCH")
        mean, variance = self._tensor(state.mean), self._tensor(state.variance)
        if (mean.ndim != 2 or mean.shape[0] < 1 or mean.shape[1] != self.latent_size
                or variance.shape != mean.shape or not torch.isfinite(mean).all()):
            raise ValueError("STATE_SHAPE_OR_MEAN_INVALID")
        self._positive_finite(variance)
        u = self._tensor(stimulus)
        if u.ndim == 2:
            u = u.unsqueeze(0).expand(mean.shape[0], -1, -1)
        if (u.ndim != 3 or u.shape[0] != mean.shape[0] or u.shape[1] < 1
                or u.shape[2] != self.input_size or not torch.isfinite(u).all()):
            raise ValueError("KNOWN_STIMULUS_SEQUENCE_SHAPE_OR_VALUES_INVALID")
        return mean, variance, u

    def forecast_moments(self, state: GaussianPopulationState, stimulus: Any) -> GaussianForecast:
        """Exact future marginal moments for boundary q and linear Gaussian prior.

        Keep full covariance after the first transition; diagonal propagation
        would lose cross-latent terms in the lognormal count mean. Autograd through
        future input is retained. The unknown true posterior is not assumed exact.
        """
        mean, variance, u = self._forecast_inputs(state, stimulus)
        covariance = torch.diag_embed(variance)
        process = self.process_log_variance.exp()
        self._positive_finite(process)
        means, covariances, expected = [], [], []
        for t in range(u.shape[1]):
            mean = mean @ self.transition.T + u[:, t] @ self.input_weight.T + self.transition_bias
            covariance = self.transition @ covariance @ self.transition.T + torch.diag(process)
            log_variance = torch.einsum("nd,bde,ne->bn", self.loading, covariance, self.loading)
            count_mean = (mean @ self.loading.T + self.log_rate_bias
                          + 0.5 * log_variance + math.log(self.bin_seconds)).exp()
            if any(not torch.isfinite(x).all() for x in (mean, covariance, count_mean)):
                raise FloatingPointError(f"NONFINITE_FORECAST_AT_BIN:{t}")
            means.append(mean); covariances.append(covariance); expected.append(count_mean)
        return GaussianForecast(torch.stack(means, 1), torch.stack(covariances, 1), torch.stack(expected, 1))

    def forecast_mean(self, state: GaussianPopulationState, stimulus: Any) -> Tensor:
        return self.forecast_moments(state, stimulus).expected_counts

    @torch.no_grad()
    def simulate(self, state: GaussianPopulationState, stimulus: Any,
                 rng: LatentRandomStreams, n_samples: int) -> LatentCountSamples:
        mean, variance, u = self._forecast_inputs(state, stimulus)
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("POSITIVE_SAMPLE_COUNT_REQUIRED")
        if not isinstance(rng, LatentRandomStreams):
            raise TypeError("THREE_LATENT_RANDOM_STREAMS_REQUIRED")
        streams = (rng.initial, rng.dynamics, rng.observation)
        if (any(not isinstance(g, torch.Generator) or torch.device(g.device) != mean.device for g in streams)
                or len({id(g) for g in streams}) != 3
                or len({g.initial_seed() for g in streams}) != 3):
            raise ValueError("DISTINCT_RANDOM_STREAMS_ON_MODEL_DEVICE_REQUIRED")
        process = self.process_log_variance.exp()
        self._positive_finite(process)
        shape = (n_samples, mean.shape[0], self.latent_size)
        z = mean + variance.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.initial)
        initial = z.clone()
        latent, draws, means = [], [], []
        for t in range(u.shape[1]):
            z = (z @ self.transition.T + u[:, t] @ self.input_weight.T + self.transition_bias
                 + process.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.dynamics))
            count_mean = (z @ self.loading.T + self.log_rate_bias + math.log(self.bin_seconds)).exp()
            if not torch.isfinite(z).all() or not torch.isfinite(count_mean).all():
                raise FloatingPointError(f"NONFINITE_GENERATION_AT_BIN:{t}")
            sampled = torch.poisson(count_mean, generator=rng.observation)
            if (not torch.isfinite(sampled).all() or torch.any(sampled < 0)
                    or torch.any(sampled >= torch.iinfo(torch.int64).max)):
                raise FloatingPointError(f"COUNT_SAMPLE_OUTSIDE_INT64_AT_BIN:{t}")
            latent.append(z); means.append(count_mean); draws.append(sampled.to(torch.int64))
        return LatentCountSamples(torch.stack(draws, 2), torch.stack(means, 2), torch.stack(latent, 2),
                                   initial, self.unit_ids, self.bin_seconds)

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save({"schema": "neural_population_controlled_poisson_lds_v1",
                        "configuration": {"unit_ids": self.unit_ids, "input_size": self.input_size,
                                          "latent_size": self.latent_size, "encoder_size": self.encoder_size,
                                          "bin_seconds": self.bin_seconds, "initialization_seed": self.initialization_seed},
                        "state_dict": self.state_dict()}, stream)

    @classmethod
    def load(cls, path: Path, *, device: str | torch.device = "cpu") -> ControlledPoissonLDS:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema"] != "neural_population_controlled_poisson_lds_v1":
            raise ValueError("POISSON_LDS_SCHEMA_MISMATCH")
        model = cls(**payload["configuration"]).to(device=device, dtype=payload["state_dict"]["loading"].dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

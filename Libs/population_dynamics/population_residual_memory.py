"""Frozen affine residual correction of a source-indexed population simulator.

The boundary memory is a causal posterior-mean residual. Generated memory is
local to each trajectory and includes that trajectory's unchanged process draw.
The parent encoder, initial posterior, transition, and renewal observation model
retain their parameters and native-clock semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from .poisson_state_space import LatentRandomStreams
from .population_state import ControlledPopulationState, PopulationBoundary, PopulationStep
from .refractory_observation import sample_bin
from .residual_prediction import RidgeModel


@dataclass(frozen=True)
class ResidualMemoryBoundary(PopulationBoundary):
    memory: Tensor = field(kw_only=True)  # batch, source field, latent channel


class ResidualMemoryPopulationState(nn.Module):
    """Compose a frozen parent with a current-state or one-residual correction."""

    def __init__(self, parent: ControlledPopulationState, predictor: RidgeModel,
                 mode: str, correction_identity: dict):
        super().__init__()
        if mode not in ("current", "history"):
            raise ValueError("residual correction mode must be current or history")
        if not isinstance(correction_identity, dict):
            raise ValueError("correction_identity must be a JSON object")
        identity = json.loads(json.dumps(correction_identity, allow_nan=False))
        d = len(parent.field_names) * parent.latent_channels
        p = d + parent.input_size + (d if mode == "history" else 0)
        arrays = {name: np.asarray(getattr(predictor, name)) for name in
                  ("x_mean", "x_scale", "y_mean", "y_scale", "coefficients")}
        shapes = {"x_mean": (p,), "x_scale": (p,), "y_mean": (d,),
                  "y_scale": (d,), "coefficients": (p, d)}
        if any(value.shape != shapes[name] or value.dtype.kind not in "fiu"
               or not np.isfinite(value).all() for name, value in arrays.items()):
            raise ValueError("ridge correction dimensions or values do not match the parent")
        if np.any(arrays["x_scale"] <= 0) or np.any(arrays["y_scale"] <= 0):
            raise ValueError("ridge correction scales must be positive")
        arrays = {name: value.astype(np.float64, copy=False) for name, value in arrays.items()}
        affine = arrays["coefficients"] * arrays["y_scale"][None, :] / arrays["x_scale"][:, None]
        bias = arrays["y_mean"] - arrays["x_mean"] @ affine
        if not np.isfinite(affine).all() or not np.isfinite(bias).all():
            raise ValueError("affine ridge correction is not finite")
        reference = parent.posterior_mean.weight
        weight_tensor = torch.as_tensor(affine.T.copy(), dtype=reference.dtype, device=reference.device)
        bias_tensor = torch.as_tensor(bias.copy(), dtype=reference.dtype, device=reference.device)
        if not torch.isfinite(weight_tensor).all() or not torch.isfinite(bias_tensor).all():
            raise ValueError("affine ridge correction is not representable in parent dtype")
        # Explicit tensors avoid the RNG consumption of a freshly initialized Linear.
        self.parent = parent
        self.correction_weight = nn.Parameter(weight_tensor, requires_grad=False)
        self.correction_bias = nn.Parameter(bias_tensor, requires_grad=False)
        self.parent.requires_grad_(False)
        self.mode = mode
        self.correction_identity = identity
        for name in ("unit_ids", "field_names", "input_size", "latent_channels",
                     "tick_seconds", "bin_seconds", "bin_ticks", "source_metadata_identity",
                     "representation", "unit_chunk_size"):
            setattr(self, name, getattr(parent, name))

    @property
    def unit_field_index(self):
        return self.parent.unit_field_index

    @property
    def minimum_interval_ticks(self):
        return self.parent.minimum_interval_ticks

    def correction(self, latent: Tensor, controls: Tensor, memory: Tensor) -> Tensor:
        """Apply the fitted affine map to parent-dtype trajectory tensors."""
        control = controls.expand(*latent.shape[:-2], self.input_size)
        parts = [latent.flatten(-2), control]
        if self.mode == "history":
            parts.append(memory.flatten(-2))
        features = torch.cat(parts, dim=-1)
        return (features @ self.correction_weight.T + self.correction_bias).reshape_as(latent)

    def infer_state(self, history: dict, metadata: dict | None = None) -> ResidualMemoryBoundary:
        if metadata is not None and (tuple(metadata["unit_ids"]) != self.unit_ids
                                     or tuple(metadata["field_names"]) != self.field_names):
            raise ValueError("history metadata differs from the source observation map")
        mean, variance = self.parent.encode(history["counts"], history["stimulus"])
        if mean.shape[1] < 2:
            raise ValueError("at least two observed bins are required for residual memory")
        previous_control = self.parent._tensor(history["stimulus"])[:, -1]
        memory = mean[:, -1] - self.parent.transition_mean(mean[:, -2], previous_control)
        if not torch.isfinite(memory).all():
            raise FloatingPointError("invalid causal residual memory")
        return ResidualMemoryBoundary(
            mean[:, -1], variance[:, -1],
            self.parent._ready(history["ready_in_ticks"], mean.shape[0]),
            self.unit_ids, self.field_names, self.bin_seconds, self.source_metadata_identity,
            memory=memory,
        )

    @torch.no_grad()
    def iter_simulation(self, state: ResidualMemoryBoundary, stimulus,
                        rng: LatentRandomStreams, n_samples: int, *, record_events: bool = True
                        ) -> Iterator[PopulationStep]:
        if not isinstance(state, ResidualMemoryBoundary):
            raise ValueError("a causal residual-memory boundary is required")
        if (state.unit_ids != self.unit_ids or state.field_names != self.field_names
                or state.bin_seconds != self.bin_seconds
                or state.source_metadata_identity != self.source_metadata_identity):
            raise ValueError("boundary source/observation map mismatch")
        mean, variance = self.parent._tensor(state.mean), self.parent._tensor(state.variance)
        memory, u = self.parent._tensor(state.memory), self.parent._tensor(stimulus)
        b, f, k = mean.shape
        if (mean.shape != (b, len(self.field_names), self.latent_channels) or variance.shape != mean.shape
                or memory.shape != mean.shape or not torch.isfinite(memory).all()
                or not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or torch.any(variance <= 0)
                or u.ndim != 3 or u.shape[0] != b or u.shape[1] < 1 or u.shape[2] != self.input_size
                or not torch.isfinite(u).all() or type(n_samples) is not int or n_samples < 1):
            raise ValueError("finite boundary, residual memory, controls and positive draw count required")
        streams = (rng.initial, rng.dynamics, rng.observation)
        if (any(torch.device(g.device) != mean.device for g in streams)
                or len({g.initial_seed() for g in streams}) != 3 or len({id(g) for g in streams}) != 3):
            raise ValueError("three independent explicit device-local random streams required")
        shape = (n_samples, b, f, k)
        z = mean + variance.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=rng.initial)
        initial = z.clone()
        h = memory.expand(n_samples, -1, -1, -1)
        ready = self.parent._ready(state.ready_in_ticks, b).expand(n_samples, -1, -1).clone()
        process = self.parent._field_parameters()[3].exp()
        if not torch.isfinite(process).all() or torch.any(process <= 0):
            raise FloatingPointError("invalid process variance")
        units = len(self.unit_ids)
        for t in range(u.shape[1]):
            base_next = self.parent.transition_mean(z, u[:, t])
            z = base_next + self.correction(z, u[:, t], h) + process.sqrt() * torch.randn(
                shape, dtype=mean.dtype, device=mean.device, generator=rng.dynamics)
            h = z - base_next
            sampled = sample_bin(self.parent.log_hazard(z), ready, self.minimum_interval_ticks,
                                 self.bin_ticks, self.tick_seconds, rng.observation, record_events=record_events)
            flat, within = sampled.events[:, 0], sampled.events[:, 1]
            events = torch.stack((flat // (b * units), (flat // units) % b,
                                  flat % units, within + t * self.bin_ticks), -1)
            ready = sampled.ready_in_ticks
            yield PopulationStep(sampled.counts, events, z, ready, initial)

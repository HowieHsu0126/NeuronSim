"""Independent AFM mathematical reimplementation with a source-field decoder.

Rogalla et al., arXiv:2604.11178v1 (2026), Section2.5: conditional
Normal-base linear flow matching, GRU context,16 Fourier flow-time features,
time-conditioned dense velocity, and autoregressive Euler sampling. No
unlicensed author code is copied. The default dimensions follow the verified
optimized author configuration. Source-field/control semantics remain an
explicit adaptation of the paper's BOLD and movie-stimulus observations.

The neuroscience adaptation maps source counts to square-root counts. A
folded square decoder produces nonnegative continuous field rates. It preserves
the source transform's zeros but does not model a discrete zero atom, count
support, refractory support or identified-unit spike activity.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

IMPLEMENTATION = "AFM2026-independent-mathematical-field-reimplementation"
PAPER_URL = "https://arxiv.org/html/2604.11178v1"


class SourceCountTransform(nn.Module):
    """Fit-only normalization of sqrt(rate_Hz * neurons * bin_seconds)."""

    def __init__(self, occupancy, bin_seconds, center, scale):
        super().__init__()
        occupancy = torch.as_tensor(occupancy, dtype=torch.float32)
        center, scale = (torch.as_tensor(value, dtype=torch.float32) for value in (center, scale))
        if (occupancy.ndim != 1 or not len(occupancy) or center.shape != occupancy.shape
                or scale.shape != occupancy.shape or not torch.isfinite(occupancy).all()
                or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
                or torch.any(occupancy <= 0) or torch.any(occupancy != occupancy.round())
                or torch.any(scale <= 0) or type(bin_seconds) not in (int, float)
                or not math.isfinite(bin_seconds) or bin_seconds <= 0):
            raise ValueError("positive integer source occupancy, finite fit-only moments and clock required")
        self.bin_seconds = float(bin_seconds)
        self.register_buffer("exposure", occupancy * bin_seconds)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def encode(self, rate):
        if (rate.shape[-1] != len(self.exposure) or not torch.isfinite(rate).all()
                or not rate.is_floating_point() or torch.any(rate < 0)):
            raise ValueError("finite nonnegative complete source rate fields required")
        return (torch.sqrt(rate * self.exposure) - self.center) / self.scale

    def decode(self, normalized):
        if normalized.shape[-1] != len(self.exposure) or not torch.isfinite(normalized).all():
            raise ValueError("finite complete transformed fields required")
        result = (normalized * self.scale + self.center).square() / self.exposure
        if not torch.isfinite(result).all():
            raise FloatingPointError("AFM field decoder overflow")
        return result


class AutoregressiveFieldFlow(nn.Module):
    """Joint next-field flow conditioned on observed history and target control.

    Joint denotes simultaneous field coordinates. Neuron outputs are absent.
    Time in the velocity is flow time[0,1], rather than the5ms source clock.
    Context length and hidden dimensions are explicit configuration choices.
    """

    def __init__(self, fields, controls, context_bins, hidden=256, layers=4,
                 latent_hidden=16, dense_blocks=5, dropout=.05):
        super().__init__()
        for name, value in (("fields", fields), ("controls", controls),
                            ("context_bins", context_bins), ("hidden", hidden), ("layers", layers),
                            ("latent_hidden", latent_hidden), ("dense_blocks", dense_blocks)):
            if type(value) is not int or value < (0 if name == "controls" else 1):
                raise ValueError(f"invalid {name}")
        if hidden % 2 or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("even hidden dimension and valid dropout required")
        self.fields, self.controls, self.context_bins = fields, controls, context_bins
        self.hidden, self.layers = hidden, layers
        self.latent_hidden, self.dense_blocks, self.dropout = latent_hidden, dense_blocks, float(dropout)
        self.observation_lifting = nn.Linear(fields, hidden // 2)
        self.control_lifting = nn.Linear(controls, hidden // 2)
        # Bidirectionality operates entirely within the observed history.
        self.context = nn.GRU(hidden, hidden // 2, layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0.)
        self.state_lifting = nn.Linear(fields, latent_hidden)
        # Target-bin control is a declared exogenous neural-source adaptation.
        self.target_control_lifting = nn.Linear(controls, hidden, bias=False)
        self.blocks = nn.ModuleList(nn.Linear(latent_hidden + hidden + 16, latent_hidden)
                                    for _ in range(dense_blocks))
        self.decoder = nn.Linear(latent_hidden + hidden, fields)
        self.register_buffer("frequencies", math.pi * torch.arange(1, 9), persistent=False)

    def get_extra_state(self):
        return dict(implementation=IMPLEMENTATION, fields=self.fields, controls=self.controls,
                    context_bins=self.context_bins, hidden=self.hidden, layers=self.layers,
                    latent_hidden=self.latent_hidden, dense_blocks=self.dense_blocks, dropout=self.dropout)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("AFM checkpoint architecture differs")

    def encode_context(self, history, past_controls):
        if (history.ndim != 3 or history.shape[-1] != self.fields
                or not 1 <= history.shape[1] <= self.context_bins
                or past_controls.shape != (*history.shape[:2], self.controls)
                or not torch.isfinite(history).all() or not torch.isfinite(past_controls).all()):
            raise ValueError("complete admissible context and aligned past controls required")
        lifted = torch.cat((F.gelu(self.observation_lifting(history)),
                            F.gelu(self.control_lifting(past_controls))), dim=-1)
        _, state = self.context(lifted)
        # Both final directional states summarize only the supplied history.
        return torch.cat((state[-2], state[-1]), dim=-1)

    def velocity(self, sample, time, context, target_control):
        if (sample.ndim != 2 or sample.shape[1] != self.fields
                or context.shape != (len(sample), self.hidden)
                or target_control.shape != (len(sample), self.controls)
                or time.shape != (len(sample), 1) or torch.any(time < 0) or torch.any(time > 1)
                or any(not torch.isfinite(value).all() for value in (sample, time, context, target_control))):
            raise ValueError("complete finite flow state,time,context and target controls required")
        phase = time * self.frequencies
        temporal = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        condition = context + self.target_control_lifting(target_control)
        value = F.gelu(self.state_lifting(sample))
        for layer in self.blocks:
            value = layer(torch.cat((value, condition, temporal), dim=-1))
            value = F.gelu(value) + value
        return self.decoder(torch.cat((value, condition), dim=-1))

    def matching_loss(self, history, past_controls, target_control, target, *, generator):
        context = self.encode_context(history, past_controls)
        if target.shape != (len(history), self.fields) or not torch.isfinite(target).all():
            raise ValueError("finite complete next-field target required")
        noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        time = torch.rand((len(target), 1), dtype=target.dtype, device=target.device, generator=generator)
        interpolated = (1 - time) * noise + time * target
        velocity = self.velocity(interpolated, time, context, target_control)
        return (velocity - (target - noise)).square().mean()

    @torch.no_grad()
    def next_sample(self, history, past_controls, target_control, *, generator, steps=100):
        if type(steps) is not int or steps < 1:
            raise ValueError("positive explicit Euler step count required")
        context = self.encode_context(history, past_controls)
        sample = torch.randn((len(history), self.fields), dtype=history.dtype,
                             device=history.device, generator=generator)
        for index in range(steps):
            time = sample.new_full((len(sample), 1), index / steps)
            sample = sample + self.velocity(sample, time, context, target_control) / steps
        if not torch.isfinite(sample).all():
            raise FloatingPointError("nonfinite AFM next-field sample")
        return sample

    @torch.no_grad()
    def forecast(self, history, controls, *, generator, bins, steps=100):
        origin = history.shape[1]
        if (type(bins) is not int or bins < 1 or origin < 1
                or controls.shape != (history.shape[0], origin + bins, self.controls)):
            raise ValueError("observed-only prefix and complete known controls required")
        outputs = []
        for index in range(origin, origin + bins):
            start = max(0, index - self.context_bins)
            current = self.next_sample(history[:, start:index], controls[:, start:index],
                                       controls[:, index], generator=generator, steps=steps)
            outputs.append(current)
            history = torch.cat((history, current[:, None]), dim=1)
        return torch.stack(outputs, dim=1)

"""Exact differentiable count means for the native deterministic-dead-time law.

An eligible tick fires with probability 1-exp(-exp(log_rate)*tick_seconds).
An event at tick t next permits an event at t+minimum_interval_ticks.
Rates are constant within each model bin; no future exposure or events are read.
"""
from __future__ import annotations

from collections import deque
import math
from numbers import Real

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def expected_sequence_counts(log_rate: Tensor, initial_ready_in_ticks: Tensor, minimum_interval_ticks: Tensor,
                             bin_ticks: int, tick_seconds: float) -> Tensor:
    """Return expected counts with the same [..., time, unit] shape and dtype.

    Availability uses disjoint refractory exclusions: a_(t+1) = a_t-e_t
    + e_(t+1-R) + 1[initial_ready=t+1], where e_t=p_t*a_t. A delay queue
    retains the event tensors and their autograd links across model bins.
    Only units with the same source minimum interval share a recurrence.
    """
    if (not isinstance(log_rate, Tensor) or not log_rate.is_floating_point()
            or log_rate.ndim < 2 or any(size < 1 for size in log_rate.shape)
            or not torch.isfinite(log_rate).all()):
        raise ValueError("finite floating log rates with nonempty [..., time, unit] shape required")
    if (type(bin_ticks) is not int or bin_ticks < 1 or not isinstance(tick_seconds, Real)
            or isinstance(tick_seconds, bool) or not math.isfinite(tick_seconds) or tick_seconds <= 0):
        raise ValueError("positive integer native bin and finite positive tick duration required")
    interval = torch.as_tensor(minimum_interval_ticks, device=log_rate.device)
    ready = torch.as_tensor(initial_ready_in_ticks, device=log_rate.device)
    boundary_shape = log_rate.shape[:-2] + log_rate.shape[-1:]
    if (interval.shape != log_rate.shape[-1:] or ready.shape != boundary_shape
            or interval.dtype not in (torch.int32, torch.int64) or ready.dtype not in (torch.int32, torch.int64)
            or torch.any(interval < 1) or torch.any(ready < 0) or torch.any(ready >= interval)):
        raise ValueError("aligned integer boundary readiness and positive per-unit intervals required")
    hazard = log_rate.exp() * tick_seconds
    if not torch.isfinite(hazard).all() or torch.any(hazard <= 0):
        raise FloatingPointError("renewal hazard outside finite positive numerical support")
    probability = -torch.expm1(-hazard)
    if (not torch.isfinite(probability).all() or torch.any(probability <= 0) or torch.any(probability > 1)):
        raise FloatingPointError("renewal event probability outside positive numerical support")

    sorted_intervals, order = interval.sort()
    values, widths = torch.unique_consecutive(sorted_intervals, return_counts=True)
    groups, offset = [], 0
    for minimum_interval, width in zip(values.tolist(), widths.tolist()):
        indices = order.narrow(0, offset, width)
        offset += width
        group_probability = probability.index_select(-1, indices)
        group_ready = ready.index_select(-1, indices)
        if minimum_interval == 1:
            # Every tick is eligible and every admissible initial readiness is 0.
            groups.append(group_probability * bin_ticks)
            continue
        available = (group_ready == 0).to(dtype=log_rate.dtype)
        delayed: deque[Tensor] = deque()
        bin_counts, elapsed = [], 0
        for model_bin in range(log_rate.shape[-2]):
            p = group_probability.select(-2, model_bin)
            count = torch.zeros_like(available)
            for _ in range(bin_ticks):
                event = p * available
                count = count + event
                delayed.append(event)
                available = available - event
                if len(delayed) == minimum_interval:
                    available = available + delayed.popleft()
                elapsed += 1
                if elapsed < minimum_interval:
                    available = available + (group_ready == elapsed).to(dtype=log_rate.dtype)
            bin_counts.append(count)
        groups.append(torch.stack(bin_counts, dim=-2))
    result = torch.cat(groups, dim=-1).index_select(-1, order.argsort())
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise FloatingPointError("renewal expectation outside finite nonnegative numerical support")
    return result


def expected_field_counts(parent, trajectory, initial_ready, *, unit_chunk_size=4096, checkpoint_readout=True):
    """Reduce every unit's count mean into its source field.

    Readiness is either shared [unit] or distinct [path, unit]. The first
    trajectory axis can flatten particles and distinct batch/origin cases.
    """
    units, fields = len(parent.unit_ids), len(parent.field_names)
    if (trajectory.ndim != 4 or trajectory.shape[-2:] != (fields, parent.latent_channels)
            or min(trajectory.shape) < 1 or type(unit_chunk_size) is not int or unit_chunk_size < 1
            or type(checkpoint_readout) is not bool):
        raise ValueError("complete particle/time/field/channel path and positive unit chunk required")
    ready = torch.as_tensor(initial_ready, device=trajectory.device)
    mapping, intervals = parent.unit_field_index, parent.minimum_interval_ticks
    if (ready.shape not in ((units,), (trajectory.shape[0], units))
            or mapping.shape != (units,) or intervals.shape != (units,)
            or mapping.device != trajectory.device or intervals.device != trajectory.device
            or mapping.dtype != torch.int64 or torch.any(mapping < 0) or torch.any(mapping >= fields)
            or torch.any(torch.bincount(mapping, minlength=fields) == 0)):
        raise ValueError("complete source units, occupied fields and boundary readiness required")
    if ready.ndim == 1:
        ready = ready.expand(trajectory.shape[0], -1)
    result = trajectory.new_zeros((*trajectory.shape[:2], fields))
    for start in range(0, units, unit_chunk_size):
        stop = min(start + unit_chunk_size, units)
        # Bind indices now: checkpoint invokes this function again in backward.
        def readout_chunk(path, start=start, stop=stop):
            log_rate = parent.log_hazard(path, start, stop)
            if log_rate.shape != (*path.shape[:2], stop - start):
                raise ValueError("unit readout lost particle/time/unit identity")
            counts = expected_sequence_counts(log_rate, ready[:, start:stop],
                intervals[start:stop], parent.bin_ticks, parent.tick_seconds)
            return path.new_zeros((*path.shape[:2], fields)).index_add(-1, mapping[start:stop], counts)
        contribution = (checkpoint(readout_chunk, trajectory, use_reentrant=False, preserve_rng_state=False)
                        if checkpoint_readout else readout_chunk(trajectory))
        result = result + contribution
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise FloatingPointError("nonfinite or negative expected source fields")
    return result


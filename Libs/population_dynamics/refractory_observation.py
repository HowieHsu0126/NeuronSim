"""Native-clock renewal observations with an explicit deterministic dead time.

An eligible tick spikes with probability 1-exp(-rate_hz*tick_seconds).
For a minimum interval R, a spike at s excludes ticks s+1,...,s+R-1.
Rates are constant within each model bin. This is a point-process model,
not a clipped Poisson count distribution or a mechanistic neuron equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class RenewalEpisode(Protocol):
    """Observed events and aligned count/control windows on a source clock."""

    unit_ids: np.ndarray
    event_ticks: np.ndarray
    event_units: np.ndarray
    bin_ticks: int

    def window(self, start_tick: int, stop_tick: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class RenewalWindow:
    counts: np.ndarray
    eligible_ticks: np.ndarray
    stimulus: np.ndarray
    initial_ready_in_ticks: np.ndarray
    terminal_ready_in_ticks: np.ndarray


def renewal_window(data: RenewalEpisode, start_tick: int, stop_tick: int,
                   minimum_interval_ticks: np.ndarray) -> RenewalWindow:
    """Sufficient point-process likelihood statistics without a dense tick grid.

    Uses observed spikes before the boundary to recover availability. The data
    object contains the complete episode prefix; hidden membrane state is not
    read. Counts/exposures depend only on events before stop_tick.
    """
    interval = np.asarray(minimum_interval_ticks)
    if (interval.shape != (len(data.unit_ids),) or interval.dtype.kind not in "iu"
            or np.any(interval < 1)):
        raise ValueError("positive per-unit integer minimum intervals required")
    interval = interval.astype(np.int64, copy=False)
    counts, stimulus = data.window(start_tick, stop_tick)
    event_stop = np.searchsorted(data.event_ticks, stop_tick, side="left")
    event_ticks = data.event_ticks[:event_stop]
    event_units = data.event_units[:event_stop]
    order = np.lexsort((event_ticks, event_units))
    consecutive = np.diff(event_units[order]) == 0
    if np.any(np.diff(event_ticks[order])[consecutive] < interval[event_units[order][1:][consecutive]]):
        raise ValueError("observed spike train violates the declared minimum interval")
    last = np.full(len(interval), -int(interval.max()), dtype=np.int64)
    earlier = event_ticks < start_tick
    np.maximum.at(last, event_units[earlier], event_ticks[earlier])
    initial_ready = np.maximum(last + interval - start_tick, 0)
    np.maximum.at(last, event_units[~earlier], event_ticks[~earlier])
    terminal_ready = np.maximum(last + interval - stop_tick, 0)
    eligible = np.full(counts.shape, data.bin_ticks, dtype=np.int32)
    # Refractory intervals cannot overlap after the source-support check above.
    left = np.maximum(event_ticks + 1, start_tick)
    right = np.minimum(event_ticks + interval[event_units], stop_tick)
    retained = right > left
    left, right, units = left[retained], right[retained], event_units[retained]
    first_bin = (left - start_tick) // data.bin_ticks
    last_bin = (right - 1 - start_tick) // data.bin_ticks
    max_span = int(np.max(last_bin - first_bin, initial=0))
    for offset in range(max_span + 1):
        current = first_bin + offset
        active = current <= last_bin
        bin_left = start_tick + current[active] * data.bin_ticks
        blocked = np.minimum(right[active], bin_left + data.bin_ticks) - np.maximum(left[active], bin_left)
        np.add.at(eligible, (current[active], units[active]), -blocked.astype(np.int32))
    if np.any(eligible < counts) or np.any(eligible > data.bin_ticks):
        raise ValueError("invalid eligible exposure or spike support")
    return RenewalWindow(counts, eligible, stimulus, initial_ready, terminal_ready)


def observation_nll(counts: Tensor, eligible_ticks: Tensor, log_rate: Tensor,
                    tick_seconds: float) -> Tensor:
    """Exact event-time NLL given piecewise-constant log rates; sum over units.

    There is no binomial coefficient: the observations are ordered event times,
    summarized by success and eligible-failure counts, not unordered counts.
    """
    if (counts.shape != log_rate.shape or eligible_ticks.shape != counts.shape
            or not np.isfinite(tick_seconds) or tick_seconds <= 0
            or not torch.isfinite(log_rate).all()
            or not torch.isfinite(counts).all() or not torch.isfinite(eligible_ticks).all()
            or torch.any(counts < 0) or torch.any(counts != counts.round())
            or torch.any(eligible_ticks < counts) or torch.any(eligible_ticks != eligible_ticks.round())):
        raise ValueError("valid aligned event counts, eligible exposure and log rates required")
    hazard = log_rate.exp() * tick_seconds
    if not torch.isfinite(hazard).all() or torch.any(hazard <= 0):
        raise FloatingPointError("renewal hazard outside finite positive numerical support")
    log_event = torch.log(-torch.expm1(-hazard))
    return ((eligible_ticks - counts) * hazard - counts * log_event).sum(-1)


@dataclass(frozen=True)
class RenewalBinSamples:
    counts: Tensor
    ready_in_ticks: Tensor
    # Rows: flat leading-dimension/unit index, within-bin emission tick.
    events: Tensor
    events_recorded: bool = True


@torch.no_grad()
def sample_bin(log_rate: Tensor, ready_in_ticks: Tensor, minimum_interval_ticks: Tensor,
               bin_ticks: int, tick_seconds: float, generator: torch.Generator,
               *, record_events: bool = True) -> RenewalBinSamples:
    """Sample the renewal law by geometric waiting times, including bin boundaries."""
    if (type(bin_ticks) is not int or bin_ticks < 1 or not np.isfinite(tick_seconds)
            or tick_seconds <= 0 or type(record_events) is not bool):
        raise ValueError("positive native bin and tick duration required")
    interval = torch.as_tensor(minimum_interval_ticks, device=log_rate.device)
    ready = torch.as_tensor(ready_in_ticks, device=log_rate.device)
    if (interval.shape != (log_rate.shape[-1],) or ready.shape != log_rate.shape
            or interval.dtype not in (torch.int32, torch.int64)
            or ready.dtype not in (torch.int32, torch.int64)
            or torch.any(interval < 1) or torch.any(ready < 0) or torch.any(ready >= interval)
            or torch.device(generator.device) != log_rate.device):
        raise ValueError("per-unit integer interval, boundary availability and RNG device must match")
    hazard = log_rate.exp() * tick_seconds
    if not torch.isfinite(hazard).all() or torch.any(hazard <= 0):
        raise FloatingPointError("renewal hazard outside finite positive numerical support")
    ready = ready.clone().to(torch.int64)
    counts = torch.zeros_like(ready)
    active = ready < bin_ticks
    events = []
    maximum_spikes = (bin_ticks - 1) // int(interval.min()) + 1
    for _ in range(maximum_spikes):
        uniform = torch.rand(log_rate.shape, dtype=log_rate.dtype, device=log_rate.device, generator=generator)
        wait = -torch.log1p(-uniform) / hazard
        # Values >= bin_ticks are censored waits, not accepted or clipped spikes.
        proposed = ready + wait.clamp_max(bin_ticks).floor().to(torch.int64)
        fired = active & (proposed < bin_ticks)
        if record_events:
            flat = torch.nonzero(fired.reshape(-1), as_tuple=False).flatten()
            events.append(torch.stack((flat, proposed.reshape(-1)[flat]), dim=1))
        counts += fired
        ready = torch.where(fired, proposed + interval, ready)
        active = fired
    recorded = torch.cat(events, dim=0) if record_events else ready.new_empty((0, 2))
    return RenewalBinSamples(counts, (ready - bin_ticks).clamp_min(0), recorded, record_events)

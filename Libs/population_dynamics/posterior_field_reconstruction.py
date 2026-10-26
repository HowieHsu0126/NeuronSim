"""Observed-exposure posterior field reconstruction and response covariance.

These deterministic reconstructions condition on observed native exposure.
Probability integration bounds are not confidence or covariance bounds.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .population_activity_diagnostic import gaussian_tick_bounds


@torch.no_grad()
def posterior_field_reconstruction(
    model,
    mean: Tensor,
    variance: Tensor,
    eligible_ticks: np.ndarray | Tensor,
    unit_chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Return four CPU float64 arrays with shape [time, source_field].

    Posterior tensors have shape [1, time, source_field, latent_channel].
    Eligible exposure has shape [time, source_unit] and integer values from
    zero through model.bin_ticks. Zero variance is a valid degenerate input;
    admission of real cached posteriors remains the caller's responsibility.

    The frozen native readout runs on model.unit_loading.device. Every source
    unit contributes to its declared field, including a final partial chunk.
    Fields with no units remain zero. Inputs and model tensors are unchanged.
    """
    units, fields, channels = len(model.unit_ids), len(model.field_names), model.latent_channels
    if (not isinstance(mean, Tensor) or not isinstance(variance, Tensor)
            or mean.ndim != 4 or mean.shape[0] != 1 or mean.shape[1] < 1
            or mean.shape != variance.shape or mean.shape[2:] != (fields, channels)
            or min(units, fields, channels) < 1
            or not mean.is_floating_point() or not variance.is_floating_point()
            or not torch.isfinite(mean).all() or not torch.isfinite(variance).all()
            or torch.any(variance < 0)):
        raise ValueError("finite aligned posterior tensors and nonnegative variance required")
    if type(unit_chunk_size) is not int or unit_chunk_size < 1:
        raise ValueError("positive integer unit_chunk_size required")
    tick_seconds, bin_seconds, bin_ticks = model.tick_seconds, model.bin_seconds, model.bin_ticks
    if (type(bin_ticks) is not int or bin_ticks < 1
            or not math.isfinite(tick_seconds) or tick_seconds <= 0
            or not math.isfinite(bin_seconds) or bin_seconds <= 0):
        raise ValueError("positive finite native clocks required")
    loading, field_index = model.unit_loading, model.unit_field_index
    if (not isinstance(loading, Tensor) or not isinstance(field_index, Tensor)
            or loading.shape != (units, channels) or not loading.is_floating_point()
            or not torch.isfinite(loading).all() or field_index.shape != (units,)
            or field_index.dtype not in (torch.int32, torch.int64)
            or field_index.device != loading.device
            or torch.any(field_index < 0) or torch.any(field_index >= fields)):
        raise ValueError("complete finite unit loading and source field map required")
    exposure = (eligible_ticks.detach().cpu().numpy() if isinstance(eligible_ticks, Tensor)
                else np.asarray(eligible_ticks))
    if (exposure.shape != (mean.shape[1], units) or exposure.dtype.kind not in "iuf"
            or not np.isfinite(exposure).all() or np.any(exposure < 0)
            or np.any(exposure > bin_ticks) or np.any(exposure != np.rint(exposure))):
        raise ValueError("complete nonnegative integer native eligible exposure required")

    device = loading.device
    mean64 = mean.detach().to(device=device, dtype=torch.float64)
    variance64 = variance.detach().to(device=device, dtype=torch.float64)
    result = {name: np.zeros((mean.shape[1], fields), dtype=np.float64) for name in (
        "mean_state_count", "posterior_expected_count_lower",
        "posterior_expected_count_upper", "posterior_integrated_intensity")}
    for start in range(0, units, unit_chunk_size):
        stop = min(start + unit_chunk_size, units)
        field = field_index[start:stop]
        log_mean_tensor = model.log_hazard(mean64, start, stop)
        if log_mean_tensor.shape != (1, mean.shape[1], stop - start):
            raise ValueError("frozen readout output shape differs from source units")
        log_mean = log_mean_tensor.detach()[0].cpu().numpy()
        log_variance = (
            variance64.index_select(-2, field) * loading[start:stop].double().square()
        ).sum(-1).detach()[0].cpu().numpy()
        moments = gaussian_tick_bounds(log_mean, log_variance, tick_seconds)
        observed = exposure[:, start:stop]
        values = {
            "mean_state_count": -np.expm1(-np.exp(log_mean) * tick_seconds) * observed,
            "posterior_expected_count_lower": moments["probability_lower"] * observed,
            "posterior_expected_count_upper": moments["probability_upper"] * observed,
            "posterior_integrated_intensity": moments["expected_rate_hz"] * bin_seconds,
        }
        source_fields = field.detach().cpu().numpy()
        for name, value in values.items():
            # CPU unbuffered accumulation retains source-unit order across chunks.
            np.add.at(result[name].T, source_fields, value.T)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("finite complete posterior field totals required")
    return result


def response_covariance(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    lag: int = 0,
) -> np.ndarray:
    """Between-block covariance of deterministic matched responses.

    values: [block, time, field]. left/right: aligned one-dimensional integer
    field indices. Return [time-lag, pair], centered across blocks at each
    matched time/pair, with denominator blocks-1. This is the between-block
    formula in trajectory_forecast_analysis.covariance_components, without
    a draw axis or a within-ensemble term. It is not predictive covariance.
    """
    values = np.asarray(values)
    left, right = np.asarray(left), np.asarray(right)
    if (values.ndim != 3 or values.shape[0] < 2 or min(values.shape[1:]) < 1
            or values.dtype.kind not in "iuf" or not np.isfinite(values).all()):
        raise ValueError("finite deterministic responses with at least two blocks required")
    if (left.ndim != 1 or right.shape != left.shape or left.size < 1
            or left.dtype.kind not in "iu" or right.dtype.kind not in "iu"
            or np.any(left < 0) or np.any(right < 0)
            or np.any(left >= values.shape[2]) or np.any(right >= values.shape[2])):
        raise ValueError("aligned valid integer source field pairs required")
    if type(lag) is not int or not 0 <= lag < values.shape[1]:
        raise ValueError("integer lag within the response window required")
    responses = values.astype(np.float64, copy=False)
    x, y = responses[:, :values.shape[1] - lag, left], responses[:, lag:, right]
    covariance = ((x - x.mean(0)) * (y - y.mean(0))).sum(0) / (values.shape[0] - 1)
    if not np.isfinite(covariance).all():
        raise ValueError("finite matched-block response covariance required")
    return covariance

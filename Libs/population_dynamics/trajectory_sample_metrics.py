"""Common finite-ensemble unit-count metrics for population simulators."""
from __future__ import annotations

import torch


def _integer_pair_sum(residual):
    """Sum every cell's ordered-pair product without fixed-width overflow.

    Split only the arithmetic work when a block bound is too large. An extreme
    individual cell uses Python integers; no cell or count is dropped or clipped.
    """
    draws = residual.shape[0]
    pending = [residual.reshape(draws, -1)]
    limit = torch.iinfo(torch.int64).max
    numerator = 0
    while pending:
        values = pending.pop()
        cells = values.shape[1]
        maximum = int(values.abs().max().item())
        if draws*draws*maximum*maximum*cells <= limit:
            total = values.sum(0)
            numerator += int((total.square()-values.square().sum(0)).sum().item())
        elif cells > 1:
            middle = cells // 2
            pending.extend((values[:, :middle], values[:, middle:]))
        else:
            integers = [int(v) for v in values[:, 0].tolist()]
            numerator += sum(integers)**2 - sum(v*v for v in integers)
    return numerator


def corrected_count_mse(samples, observed, *, chunk_units=4096):
    """The same U-statistic, with an exact integer numerator and FP32 output.

    For residuals r_m = count_m - observed, the per-bin numerator is
    (sum_m r_m)^2 - sum_m r_m^2. Integer arithmetic avoids subtracting
    separately rounded FP32 sample-mean error and Monte Carlo variance.
    Python integers accumulate chunks; oversized blocks split and individual
    extreme cells use arbitrary-precision integers before the same FP32 output.
    """
    if type(chunk_units) is not int or chunk_units < 1:
        raise ValueError('positive integer metric chunk size required')
    draws, bins, units = samples.shape
    if min(bins, units) < 1 or draws < 2:
        raise ValueError('nonempty count trajectories and at least two draws required')
    integer_limit = torch.iinfo(torch.int64).max
    if int(observed.max().item()) > integer_limit:
        raise ValueError('observed counts exceed exact int64 correction support')
    numerator = 0
    for start in range(0, units, chunk_units):
        current = samples[:, :, start:start+chunk_units].to(torch.int64)
        actual = observed[:, start:start+chunk_units].to(torch.int64)
        residual = current - actual.unsqueeze(0)
        numerator += _integer_pair_sum(residual)
    estimate = numerator / (draws*(draws-1)*bins*units)
    return float(torch.tensor(estimate, dtype=torch.float32).item())


@torch.no_grad()
def unit_count_metrics(samples, observed, minimum_interval_ticks, bin_ticks, support_windows):
    """Use identical sampled-count estimands for different observation laws.

    The corrected squared-error estimate subtracts the Monte Carlo variance of
    the sample mean. Negative estimates are valid and are not clipped.
    """
    interval = torch.as_tensor(minimum_interval_ticks, device=samples.device)
    if (samples.ndim != 3 or len(samples) < 2 or observed.shape != samples.shape[1:]
            or interval.shape != (samples.shape[-1],) or torch.any(interval < 1)
            or interval.dtype not in (torch.int32, torch.int64)
            or type(bin_ticks) is not int or bin_ticks < 1
            or samples.dtype not in (torch.int32, torch.int64) or torch.any(samples < 0)
            or not torch.isfinite(observed).all() or torch.any(observed < 0)
            or torch.any(observed != observed.round())):
        raise ValueError("identified finite integer count trajectories and source spacing required")
    # The MSE numerator uses arbitrary precision, but total-count and rolling
    # support metrics below still require exact fixed-width trajectory sums.
    integer_limit = torch.iinfo(torch.int64).max
    if (samples.shape[1] * int(samples.max().item()) > integer_limit
            or observed.shape[0] * int(observed.max().item()) > integer_limit):
        raise ValueError('count trajectory totals may exceed exact int64 reduction support')
    mse = corrected_count_mse(samples, observed)
    totals = samples.sum(1).float()
    quantiles = torch.quantile(totals, totals.new_tensor([.05, .95]), dim=0)
    actual_total = observed.sum(0)
    result = dict(unit_bin_mean_mse_corrected=float(mse),
                  sampled_expected_spikes_per_unit=float(totals.mean()),
                  observed_spikes_per_unit=float(actual_total.float().mean()),
                  unit_total_coverage_90=float(((actual_total >= quantiles[0]) & (actual_total <= quantiles[1])).float().mean()),
                  unit_total_interval_width_90=float((quantiles[1] - quantiles[0]).mean()))
    cumulative = torch.cat((samples.new_zeros((len(samples), 1, samples.shape[-1])), samples.cumsum(1)), dim=1)
    for width in support_windows:
        if type(width) is not int or width < 1:
            raise ValueError("positive integer support windows required")
        name = f"refractory_count_violation_{width}bin"
        if width > samples.shape[1]:
            result[name] = None
        else:
            maximum = (width * bin_ticks - 1) // interval + 1
            rolling = cumulative[:, width:] - cumulative[:, :-width]
            result[name] = float((rolling > maximum).float().mean())
    return result

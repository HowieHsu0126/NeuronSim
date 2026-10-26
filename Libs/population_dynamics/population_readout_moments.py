"""Chunked latent and affine-readout moments for a frozen population simulator.

Cached q is a diagnostic distribution in the frozen encoder's coordinates, not
physical latent truth. Its diagonal Gaussian marginal gives an exact per-time
lognormal readout adjustment. The finite-draw empirical Jensen gap is a different
quantity. Their decomposition concerns hazards, not refractory count errors.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .population_state import ControlledPopulationState


def _real_double(value, name: str, device: torch.device) -> Tensor:
    value = torch.as_tensor(value, device=device)
    if value.is_complex() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real values")
    return value.to(dtype=torch.float64)


@torch.no_grad()
def trajectory_readout_moments(
    parent: ControlledPopulationState,
    samples: Tensor,
    initial_samples: Tensor,
    posterior_mean: Tensor,
    posterior_variance: Tensor,
    coordinate_scale: Tensor,
    unit_chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Return float64 arrays of shape [time] on the parent's device.

    samples has shape [draw, time, field, channel]; initial_samples has shape
    [draw, field, channel]. The cached q mean/variance have shape [time, field,
    channel]. coordinate_scale is a positive fitting-only vector [field*channel].

    Unit means weight every source unit equally. Ensemble variances and lag-one
    same-draw covariance use draw_count-1. Lag at the first time uses the supplied
    initial samples. Readout component mean squares are unit means of squared
    displacements, not squares of the unit-averaged displacement.

    For affine log-hazard h, A=h(E[z])-h(E_q[z]), B=empirical_Jensen-Jensen_q,
    and C=log(E_draw[exp(h(z))])-log(E_q[exp(h(z))]). The returned cross term is
    E_unit[2*A*B]. Pointwise and mean-square closure residuals remain unmodified.
    No full coordinate or unit covariance matrix is constructed.

    Arithmetic hazard summaries sum rates across all units before taking a
    logarithm. Their reference is the posterior-mean path used by the separate
    native count-closure diagnostic. These rates do not include refractory
    availability and therefore are not expected spike counts.
    """
    device = parent.unit_field_index.device
    samples = _real_double(samples, "samples", device)
    initial_samples = _real_double(initial_samples, "initial_samples", device)
    posterior_mean = _real_double(posterior_mean, "posterior_mean", device)
    posterior_variance = _real_double(posterior_variance, "posterior_variance", device)
    coordinate_scale = _real_double(coordinate_scale, "coordinate_scale", device)
    fields, channels = len(parent.field_names), parent.latent_channels
    if (samples.ndim != 4 or samples.shape[0] < 2 or samples.shape[1] < 1
            or samples.shape[2:] != (fields, channels)
            or initial_samples.shape != (samples.shape[0], fields, channels)
            or posterior_mean.shape != samples.shape[1:]
            or posterior_variance.shape != posterior_mean.shape
            or coordinate_scale.shape != (fields * channels,)):
        raise ValueError("aligned draw/time/field/channel arrays and at least two draws required")
    if torch.any(posterior_variance <= 0) or torch.any(coordinate_scale <= 0):
        raise ValueError("posterior variance and coordinate scale must be strictly positive")
    chunk = parent.unit_chunk_size if unit_chunk_size is None else unit_chunk_size
    if type(chunk) is not int or chunk < 1:
        raise ValueError("unit_chunk_size must be a positive integer")
    units = len(parent.unit_ids)
    if units < 1:
        raise ValueError("at least one source unit required")
    draws, times = samples.shape[:2]
    mean_state = samples.mean(dim=0)
    centered = samples - mean_state
    initial_centered = initial_samples - initial_samples.mean(dim=0)
    previous_centered = torch.cat((initial_centered[:, None], centered[:, :-1]), dim=1)
    scale = coordinate_scale.reshape(fields, channels)
    standardized = centered / scale
    results = dict(
        standardized_latent_mean_mse=((mean_state-posterior_mean) / scale).square().mean(dim=(-2, -1)),
        standardized_latent_ensemble_variance_mean=(standardized.square().sum(dim=0) / (draws-1)).mean(dim=(-2, -1)),
        standardized_latent_lag_one_covariance_mean=(
            (standardized * (previous_centered / scale)).sum(dim=0) / (draws-1)).mean(dim=(-2, -1)),
    )
    readout_names = (
        "predicted_mean_log_hazard_unit_mean", "posterior_mean_log_hazard_unit_mean",
        "empirical_log_expected_hazard_unit_mean", "posterior_log_expected_hazard_unit_mean",
        "empirical_log_hazard_variance_unit_mean", "empirical_jensen_gap_unit_mean",
        "posterior_lognormal_adjustment_unit_mean", "mean_displacement_unit_mean",
        "mean_displacement_unit_mean_square", "stochastic_gap_displacement_unit_mean",
        "stochastic_gap_displacement_unit_mean_square", "total_log_expected_hazard_gap_unit_mean",
        "total_log_expected_hazard_gap_unit_mean_square", "decomposition_cross_term_unit_mean",
    )
    totals = {name: torch.zeros(times, dtype=torch.float64, device=device) for name in readout_names}
    rate_log_sums = {name: torch.full((times,), -math.inf, dtype=torch.float64, device=device)
                     for name in ("draw", "mean_path", "posterior_mean")}
    closure_max = torch.zeros(times, dtype=torch.float64, device=device)
    for start in range(0, units, chunk):
        stop = min(start+chunk, units)
        sample_log_hazard = parent.log_hazard(samples, start, stop)
        mean_log_hazard = parent.log_hazard(mean_state, start, stop)
        q_mean_log_hazard = parent.log_hazard(posterior_mean, start, stop)
        if parent.metadata is None:
            loading = parent.unit_loading[start:stop].to(dtype=torch.float64)
        else:
            readout = parent.shared_unit_readout(parent._intrinsic_unit_features(start, stop))
            loading = readout[:, :-1].to(dtype=torch.float64)
        unit_variance = posterior_variance.index_select(1, parent.unit_field_index[start:stop])
        q_adjustment = .5 * (unit_variance * loading.square()).sum(dim=-1)
        empirical_log_expected = torch.logsumexp(sample_log_hazard, dim=0) - math.log(draws)
        for name, log_rates in (("draw", empirical_log_expected), ("mean_path", mean_log_hazard),
                                ("posterior_mean", q_mean_log_hazard)):
            rate_log_sums[name] = torch.logaddexp(rate_log_sums[name], torch.logsumexp(log_rates, dim=-1))
        q_log_expected = q_mean_log_hazard + q_adjustment
        empirical_jensen = empirical_log_expected - mean_log_hazard
        log_hazard_variance = (
            (sample_log_hazard-sample_log_hazard.mean(dim=0)).square().sum(dim=0) / (draws-1))
        mean_displacement = mean_log_hazard - q_mean_log_hazard
        stochastic_displacement = empirical_jensen - q_adjustment
        total_gap = empirical_log_expected - q_log_expected
        closure = total_gap - mean_displacement - stochastic_displacement
        values = dict(
            predicted_mean_log_hazard_unit_mean=mean_log_hazard,
            posterior_mean_log_hazard_unit_mean=q_mean_log_hazard,
            empirical_log_expected_hazard_unit_mean=empirical_log_expected,
            posterior_log_expected_hazard_unit_mean=q_log_expected,
            empirical_log_hazard_variance_unit_mean=log_hazard_variance,
            empirical_jensen_gap_unit_mean=empirical_jensen,
            posterior_lognormal_adjustment_unit_mean=q_adjustment,
            mean_displacement_unit_mean=mean_displacement,
            mean_displacement_unit_mean_square=mean_displacement.square(),
            stochastic_gap_displacement_unit_mean=stochastic_displacement,
            stochastic_gap_displacement_unit_mean_square=stochastic_displacement.square(),
            total_log_expected_hazard_gap_unit_mean=total_gap,
            total_log_expected_hazard_gap_unit_mean_square=total_gap.square(),
            decomposition_cross_term_unit_mean=2*mean_displacement*stochastic_displacement,
        )
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("affine readout moments are not finite")
        for name, value in values.items():
            totals[name] += value.sum(dim=-1)
        closure_max = torch.maximum(closure_max, closure.abs().amax(dim=-1))
    results.update({name: total / units for name, total in totals.items()})
    log_rates = {name: value - math.log(units) for name, value in rate_log_sums.items()}
    results.update({f"log_unit_mean_{name}_hazard": value for name, value in log_rates.items()})
    results.update(
        unit_mean_hazard_mean_path_log_ratio=log_rates["mean_path"] - log_rates["posterior_mean"],
        unit_mean_hazard_empirical_jensen_gap=log_rates["draw"] - log_rates["mean_path"],
        unit_mean_hazard_draw_to_posterior_mean_log_ratio=log_rates["draw"] - log_rates["posterior_mean"],
    )
    results["decomposition_closure_max_absolute"] = closure_max
    results["decomposition_mean_square_closure"] = (
        results["total_log_expected_hazard_gap_unit_mean_square"]
        - results["mean_displacement_unit_mean_square"]
        - results["stochastic_gap_displacement_unit_mean_square"]
        - results["decomposition_cross_term_unit_mean"])
    if not all(torch.isfinite(value).all() for value in results.values()):
        raise ValueError("trajectory readout moments are not finite")
    return results

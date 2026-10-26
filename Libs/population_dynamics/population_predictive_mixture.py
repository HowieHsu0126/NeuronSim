"""Score the joint latent predictive mixture against a frozen posterior.

The mixture components are uninterrupted native prior paths. This changes the
training loss, not the transition, observation law or model architecture.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .population_prior_objective import _validated_prior_inputs
from .population_state import ControlledPopulationState


_LOG_TWO_PI = math.log(2 * math.pi)


def joint_gaussian_mixture_log_density(
    samples: Tensor, component_means: Tensor, component_variance: Tensor,
) -> Tensor:
    """Return log density [draw, case] for a uniform diagonal Gaussian mixture.

    samples is [draw, case, source field, channel]; component_means is
    [particle, case, source field, channel]. component_variance is the native
    process variance [source field, channel], shared by the components.
    One mixture component covers the entire field/channel state. Summing its
    coordinate log densities before logsumexp preserves that joint meaning.
    """
    if (samples.ndim != 4 or component_means.ndim != 4
            or min(samples.shape) < 1 or min(component_means.shape) < 1
            or samples.shape[1:] != component_means.shape[1:]
            or component_variance.shape != samples.shape[-2:]):
        raise ValueError("aligned draw/particle, case, field and channel tensors required")
    difference = samples[:, None] - component_means[None]
    component_log_density = -.5 * (
        difference.square() / component_variance
        + component_variance.log() + _LOG_TWO_PI).sum((-1, -2))
    return torch.logsumexp(component_log_density, dim=1) - math.log(component_means.shape[0])


def prior_predictive_mixture_loss(
    model: ControlledPopulationState,
    mean: Tensor,
    variance: Tensor,
    controls: Tensor,
    origins: Sequence[int],
    horizons: Sequence[int],
    initial_generator: torch.Generator,
    dynamics_generator: torch.Generator,
    target_generator: torch.Generator,
    *,
    prediction_particles: int = 8,
    posterior_target_draws: int = 4,
) -> dict[str, Tensor]:
    """Estimate KL(q_target || finite joint predictive mixture) at each horizon.

    mean and variance are [batch, time, source field, channel]; controls is
    [batch, time, control]. An origin counts observed bins. Its boundary is
    q[origin - 1]; step h uses control and target at origin + h - 1.
    All supplied posterior states and controls are detached. Only the native
    transition and process parameters participate in the predictive paths.

    Draw prediction_particles independent boundary states, then propagate
    each path without posterior resets. At each declared horizon the final
    conditional transition Gaussians form a uniform mixture. Use independent
    posterior_target_draws samples to estimate its cross entropy under q;
    subtract the exact diagonal-Gaussian entropy of q. Each component covers
    ALL source fields and channels. Average cases, then weight horizons equally.
    A finite Monte Carlo KL estimate may be negative; it is not clamped.

    Prediction draws use [particle, batch * origins, field, channel], with
    batch-major cases. Only transition_mean flattens particle and case axes.
    Target draws use [draw, batch * origins, field, channel], in ascending
    endpoint order. Initial, dynamics and target streams must be distinct,
    have distinct seeds and match the tensor device. Their states advance in
    place; global RNG is untouched. Dynamics consumes max(horizons) - 1 draws
    of the prediction shape. The final transition noise is integrated in the
    Gaussian components. Target draws do not affect predictive path states.
    Replaying requires the same layouts, endpoints, counts and stream states.
    """
    if (type(prediction_particles) is not int or prediction_particles < 1
            or type(posterior_target_draws) is not int or posterior_target_draws < 1):
        raise ValueError("positive integer prediction_particles and posterior_target_draws required")
    mean, variance, controls, origins, horizons, process_variance = _validated_prior_inputs(
        model, mean, variance, controls, origins, horizons,
        (initial_generator, dynamics_generator, target_generator),
        stream_names="initial, dynamics and target")
    batch, _, fields, channels = mean.shape
    cases = batch * len(origins)
    case_shape = (cases, fields, channels)
    prediction_shape = (prediction_particles, *case_shape)
    target_shape = (posterior_target_draws, *case_shape)
    origin_indices = torch.tensor(origins, dtype=torch.long, device=mean.device)
    boundary_mean = mean.index_select(1, origin_indices - 1).reshape(case_shape)
    boundary_variance = variance.index_select(1, origin_indices - 1).reshape(case_shape)
    state = boundary_mean + boundary_variance.sqrt() * torch.randn(
        prediction_shape, dtype=mean.dtype, device=mean.device, generator=initial_generator)
    process_std = process_variance.sqrt()
    horizon_terms: dict[int, Tensor] = {}
    maximum_horizon = max(horizons)
    for step in range(1, maximum_horizon + 1):
        indices = origin_indices + step - 1
        step_controls = controls.index_select(1, indices).reshape(cases, model.input_size)
        prior_mean = model.transition_mean(
            state.reshape(prediction_particles * cases, fields, channels),
            step_controls.expand(prediction_particles, -1, -1).reshape(
                prediction_particles * cases, model.input_size)).reshape(prediction_shape)
        if step in horizons:
            target_mean = mean.index_select(1, indices).reshape(case_shape)
            target_variance = variance.index_select(1, indices).reshape(case_shape)
            targets = target_mean + target_variance.sqrt() * torch.randn(
                target_shape, dtype=mean.dtype, device=mean.device, generator=target_generator)
            log_density = joint_gaussian_mixture_log_density(targets, prior_mean, process_variance)
            target_entropy = .5 * (target_variance.log() + 1 + _LOG_TWO_PI).sum((-1, -2))
            horizon_terms[step] = (-target_entropy - log_density.mean(0)).mean()
        if step < maximum_horizon:
            state = prior_mean + process_std * torch.randn(
                prediction_shape, dtype=mean.dtype, device=mean.device, generator=dynamics_generator)
    loss = torch.stack([horizon_terms[horizon] for horizon in horizons]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite population predictive mixture KL")
    return {"loss": loss, **{
        f"prior_mixture_kl_h{horizon}": horizon_terms[horizon] for horizon in horizons
    }}

"""Refine a native population prior against fixed causal posterior states.

This is multi-step latent KL refinement in the latent overshooting family
(Hafner et al., "Learning Latent Dynamics for Planning from Pixels", ICML 2019).
It is a training objective, not a standalone algorithm or evidence of closure.
The existing transition and native observation law retain their definitions.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .poisson_state_space import initial_kl
from .population_state import ControlledPopulationState


PRIOR_PARAMETER_NAMES = (
    "local_weight",
    "field_mix_logits",
    "control_weight",
    "transition_bias",
    "decay_logit",
    "process_log_variance",
)


def _native_prior_parameters(model: ControlledPopulationState) -> dict[str, nn.Parameter]:
    if not isinstance(model, ControlledPopulationState) or model.metadata is not None:
        raise ValueError("prior refinement requires the native unshared population model")
    parameters = dict(model.named_parameters())
    missing = tuple(name for name in PRIOR_PARAMETER_NAMES if name not in parameters)
    if missing:
        raise ValueError(f"native prior parameters missing: {', '.join(missing)}")
    return parameters


def configure_prior_parameters(model: ControlledPopulationState) -> tuple[nn.Parameter, ...]:
    """Freeze the encoder, posterior, initial prior and observation parameters.

    Return the native transition/process parameters in PRIOR_PARAMETER_NAMES
    order for the refit optimizer. Validate the profile before changing flags.
    Frozen parameters also lose stale gradients from preceding training.
    """
    parameters = _native_prior_parameters(model)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in PRIOR_PARAMETER_NAMES)
        if not parameter.requires_grad:
            parameter.grad = None
    return tuple(parameters[name] for name in PRIOR_PARAMETER_NAMES)


def _positive_indices(values: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of positive integers") from error
    if (not result or any(type(value) is not int or value < 1 for value in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"{name} must contain unique positive integers")
    return result


def _validated_prior_inputs(
    model: ControlledPopulationState,
    mean: Tensor,
    variance: Tensor,
    controls: Tensor,
    origins: Sequence[int],
    horizons: Sequence[int],
    generators: tuple[torch.Generator, ...],
    *,
    stream_names: str,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...], tuple[int, ...], Tensor]:
    """Validate a native prior objective without consuming random streams."""
    parameters = _native_prior_parameters(model)
    origins = _positive_indices(origins, "origins")
    horizons = _positive_indices(horizons, "horizons")
    if (not all(isinstance(value, Tensor) for value in (mean, variance, controls))
            or mean.ndim != 4 or min(mean.shape) < 1
            or variance.shape != mean.shape):
        raise ValueError("aligned nonempty batch/time/field/channel posterior tensors required")
    batch, time, fields, channels = mean.shape
    if ((fields, channels) != (len(model.field_names), model.latent_channels)
            or controls.shape != (batch, time, model.input_size)):
        raise ValueError("posterior fields/channels and controls must match the native model")
    if (not mean.is_floating_point()
            or any(value.dtype != mean.dtype or value.device != mean.device
                   for value in (variance, controls))):
        raise ValueError("posterior and controls must share a floating dtype and device")
    if max(origins) + max(horizons) > time:
        raise ValueError("every origin and horizon must fit within the posterior sequence")
    if (any(not isinstance(generator, torch.Generator) for generator in generators)
            or len({id(generator) for generator in generators}) != len(generators)
            or len({generator.initial_seed() for generator in generators}) != len(generators)
            or any(torch.device(generator.device) != mean.device for generator in generators)):
        raise ValueError(f"distinct explicit device-local {stream_names} random streams required")
    if any(parameter.device != mean.device or parameter.dtype != mean.dtype
           for parameter in parameters.values()):
        raise ValueError("posterior and controls must use the model parameter dtype and device")
    mean, variance, controls = mean.detach(), variance.detach(), controls.detach()
    if (not torch.isfinite(mean).all() or not torch.isfinite(variance).all()
            or torch.any(variance <= 0) or not torch.isfinite(controls).all()):
        raise ValueError("finite posterior/controls and strictly positive posterior variance required")
    process_variance = model.process_log_variance.exp()
    if (process_variance.shape != (fields, channels)
            or not torch.isfinite(process_variance).all()
            or torch.any(process_variance <= 0)):
        raise FloatingPointError("finite positive native process variance required")

    return mean, variance, controls, origins, horizons, process_variance


def prior_rollout_loss(
    model: ControlledPopulationState,
    mean: Tensor,
    variance: Tensor,
    controls: Tensor,
    origins: Sequence[int],
    horizons: Sequence[int],
    initial_generator: torch.Generator,
    dynamics_generator: torch.Generator,
) -> dict[str, Tensor]:
    """Score differentiable free-prior paths against detached Gaussian posteriors.

    mean and variance have shape [batch, time, source field, latent channel].
    controls has shape [batch, time, control]. An origin is the number of
    observed bins: its sampled boundary is q[origin - 1]. Step h uses control
    at origin + h - 1 and targets q at that same index.

    Each batch/origin pair receives one posterior boundary draw. Intermediate
    states then follow the stochastic prior, without posterior resets. At each
    declared horizon, analytically integrate the final conditional Gaussian
    using initial_kl(q_target, transition_mean, process_variance). This
    Monte Carlo path objective is an expectation of conditional KLs; it is
    not an analytic KL to the marginalized multi-step predictive mixture.

    Sum each KL over latent channels and source fields. Average over batches
    and origins, then give every declared horizon equal weight in loss.
    The posterior and controls are detached, including the boundary sample.
    No count or observation samples are drawn.

    Generators must be distinct, have distinct seeds and match the tensor
    device. Their states continue in place and the global RNG is untouched.
    Draw layout is [batch * origins, fields, channels], in batch-major order.
    The dynamics stream advances max(horizons) - 1 times; the final scored
    transition needs no sample. Replaying requires the same layout, origin
    order, maximum horizon and generator states. Batch chunking invariance
    is not part of this interface.
    """
    mean, variance, controls, origins, horizons, process_variance = _validated_prior_inputs(
        model, mean, variance, controls, origins, horizons,
        (initial_generator, dynamics_generator), stream_names="initial and dynamics")
    batch, _, fields, channels = mean.shape

    origin_indices = torch.tensor(origins, dtype=torch.long, device=mean.device)
    shape = (batch * len(origins), fields, channels)
    boundary_mean = mean.index_select(1, origin_indices - 1).reshape(shape)
    boundary_variance = variance.index_select(1, origin_indices - 1).reshape(shape)
    state = boundary_mean + boundary_variance.sqrt() * torch.randn(
        shape, dtype=mean.dtype, device=mean.device, generator=initial_generator)
    process_std = process_variance.sqrt()
    horizon_terms: dict[int, Tensor] = {}
    maximum_horizon = max(horizons)
    for step in range(1, maximum_horizon + 1):
        indices = origin_indices + step - 1
        step_controls = controls.index_select(1, indices).reshape(
            batch * len(origins), model.input_size)
        prior_mean = model.transition_mean(state, step_controls)
        if step in horizons:
            target_mean = mean.index_select(1, indices).reshape(shape)
            target_variance = variance.index_select(1, indices).reshape(shape)
            # initial_kl already sums channels; only the field sum remains.
            horizon_terms[step] = initial_kl(
                target_mean, target_variance, prior_mean, process_variance).sum(-1).mean()
        if step < maximum_horizon:
            state = prior_mean + process_std * torch.randn(
                shape, dtype=mean.dtype, device=mean.device, generator=dynamics_generator)
    loss = torch.stack([horizon_terms[horizon] for horizon in horizons]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite multi-step population prior KL")
    return {"loss": loss, **{
        f"prior_kl_h{horizon}": horizon_terms[horizon] for horizon in horizons
    }}

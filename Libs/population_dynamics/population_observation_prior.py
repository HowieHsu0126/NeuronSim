"""Score observed native event paths under uninterrupted prior particles.

The conditional event-path likelihood is exact for the native observation law.
Marginalizing a finite set of latent particles is a Monte Carlo estimate, not an
exact marginal likelihood. Observed exposure is a scoring label, not an input to
free latent dynamics or a generated forecast.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .population_prior_objective import PRIOR_PARAMETER_NAMES, _validated_prior_inputs
from .population_state import ControlledPopulationState
from .refractory_observation import observation_nll


def prior_observation_mixture_loss(
    model: ControlledPopulationState,
    mean: Tensor,
    variance: Tensor,
    controls: Tensor,
    target_counts: Tensor,
    target_eligible_ticks: Tensor,
    origins: Sequence[int],
    horizons: Sequence[int],
    initial_generator: torch.Generator,
    dynamics_generator: torch.Generator,
    *,
    prediction_particles: int = 8,
    unit_chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Estimate joint native event-path mixture NLL in nats per unit/bin.

    Posterior tensors are [batch, time, field, channel]; controls are
    [batch, time, control]. Labels are [batch, origin, max(horizons), unit],
    in the supplied origin order. Labels are native observed event counts and
    eligible ticks. They are detached and never enter latent propagation.
    Source ingestion must establish that these statistics describe a valid
    native event path. This function checks their shape and scalar support.

    Each particle starts at q[origin - 1]. Every step applies transition_mean
    with control[origin + step - 1], then adds independent process noise,
    including the final step. No later posterior state resets a particle.
    Random draws have shape [particle, batch * origins, field, channel], with
    batch-major cases. Explicit initial and dynamics streams advance in place;
    the global RNG is untouched. Replay requires unchanged draw layout,
    origin order, maximum horizon and stream states.

    Form one trajectory [particle, case, time, field, channel]. Evaluate its
    frozen readout in unit chunks, retaining gradients to the latent path.
    Sum conditional NLL over ALL units and each horizon's complete time prefix
    before logsumexp over particles. Normalize by units * horizon, average
    cases, then weight declared horizons equally. Float64 accumulation reduces
    unit-chunk summation error; readout and native NLL use the model dtype.
    Returned particle ESS is the detached mean across cases at each horizon.
    ESS is diagnostic and does not contribute to loss.

    Call configure_prior_parameters before this objective. Exactly the six
    existing PRIOR_PARAMETER_NAMES remain trainable; no parameter is added.
    """
    if type(prediction_particles) is not int or prediction_particles < 1:
        raise ValueError("positive integer prediction_particles required")
    mean, variance, controls, origins, horizons, process_variance = _validated_prior_inputs(
        model, mean, variance, controls, origins, horizons,
        (initial_generator, dynamics_generator), stream_names="initial and dynamics")
    if any(parameter.requires_grad != (name in PRIOR_PARAMETER_NAMES)
           for name, parameter in model.named_parameters()):
        raise ValueError("configure_prior_parameters must select exactly the six native prior parameters")
    chunk = model.unit_chunk_size if unit_chunk_size is None else unit_chunk_size
    if type(chunk) is not int or chunk < 1:
        raise ValueError("positive integer unit_chunk_size required")
    batch, _, fields, channels = mean.shape
    units, maximum_horizon = len(model.unit_ids), max(horizons)
    label_shape = (batch, len(origins), maximum_horizon, units)
    labels = (target_counts, target_eligible_ticks)
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (any(not isinstance(value, Tensor) for value in labels)
            or any(value.shape != label_shape or value.device != mean.device for value in labels)
            or any(not value.is_floating_point() and value.dtype not in integer_dtypes for value in labels)):
        raise ValueError("aligned device-local batch/origin/horizon/unit integer event labels required")
    counts, eligible = (value.detach() for value in labels)
    if (any(not torch.isfinite(value).all() for value in (counts, eligible))
            or any(value.is_floating_point() and torch.any(value != value.round())
                   for value in (counts, eligible))
            or torch.any(counts < 0) or torch.any(eligible < counts)
            or torch.any(eligible > model.bin_ticks)):
        raise ValueError("finite integer event counts and eligible ticks within native bin support required")

    cases = batch * len(origins)
    case_shape = (cases, fields, channels)
    prediction_shape = (prediction_particles, *case_shape)
    origin_indices = torch.tensor(origins, dtype=torch.long, device=mean.device)
    boundary_mean = mean.index_select(1, origin_indices - 1).reshape(case_shape)
    boundary_variance = variance.index_select(1, origin_indices - 1).reshape(case_shape)
    state = boundary_mean + boundary_variance.sqrt() * torch.randn(
        prediction_shape, dtype=mean.dtype, device=mean.device, generator=initial_generator)
    process_std = process_variance.sqrt()
    states = []
    for step in range(1, maximum_horizon + 1):
        step_controls = controls.index_select(1, origin_indices + step - 1).reshape(cases, model.input_size)
        prior_mean = model.transition_mean(
            state.reshape(prediction_particles * cases, fields, channels),
            step_controls.expand(prediction_particles, -1, -1).reshape(
                prediction_particles * cases, model.input_size)).reshape(prediction_shape)
        state = prior_mean + process_std * torch.randn(
            prediction_shape, dtype=mean.dtype, device=mean.device, generator=dynamics_generator)
        states.append(state)
    trajectory = torch.stack(states, dim=2)
    counts = counts.to(dtype=mean.dtype).reshape(cases, maximum_horizon, units)
    eligible = eligible.to(dtype=mean.dtype).reshape(cases, maximum_horizon, units)
    per_bin_nll = torch.zeros(
        (prediction_particles, cases, maximum_horizon), dtype=torch.float64, device=mean.device)
    for start in range(0, units, chunk):
        stop = min(start + chunk, units)
        log_rate = model.log_hazard(trajectory, start, stop)
        native_nll = observation_nll(
            counts[..., start:stop].unsqueeze(0).expand(prediction_particles, -1, -1, -1),
            eligible[..., start:stop].unsqueeze(0).expand(prediction_particles, -1, -1, -1),
            log_rate, model.tick_seconds)
        per_bin_nll = per_bin_nll + native_nll.to(torch.float64)
    cumulative_nll = per_bin_nll.cumsum(-1)
    terms: dict[str, Tensor] = {}
    horizon_losses = []
    for horizon in horizons:
        joint_nll = cumulative_nll[..., horizon - 1]
        nll = (math.log(prediction_particles) - torch.logsumexp(-joint_nll, dim=0)).mean() / (units * horizon)
        weights = torch.softmax(-joint_nll.detach(), dim=0)
        ess = weights.square().sum(0).reciprocal().mean()
        terms[f"prior_observation_nll_h{horizon}"] = nll
        terms[f"prior_observation_particle_ess_h{horizon}"] = ess
        horizon_losses.append(nll)
    loss = torch.stack(horizon_losses).mean()
    if not torch.isfinite(loss) or any(not torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError("nonfinite native event-path predictive-mixture NLL or particle ESS")
    return {"loss": loss, **terms}

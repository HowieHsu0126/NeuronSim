"""Fit affine residual dynamics with the frozen native event-path likelihood.

Integrate shared stochastic trajectories before normalizing over units and time.
Counts and eligible ticks are labels. Only the causal boundary and scheduled
controls enter a rollout. This loss does not inherit the old ridge qualification.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .population_prior_objective import _validated_prior_inputs
from .population_residual_memory import ResidualMemoryPopulationState
from .refractory_observation import observation_nll
from .residual_prediction import RidgeModel


CORRECTION_PARAMETERS = ("correction_weight", "correction_bias")


def configure_correction_parameters(model: ResidualMemoryPopulationState) -> tuple[nn.Parameter, ...]:
    """Train only the affine correction; keep all parent parameters fixed."""
    if not isinstance(model, ResidualMemoryPopulationState):
        raise TypeError("a native residual-memory wrapper is required")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in CORRECTION_PARAMETERS)
        parameter.grad = None
    return tuple(getattr(model, name) for name in CORRECTION_PARAMETERS)


def _normalization(model, reference):
    d, p = model.correction_weight.shape
    arrays = {name: np.asarray(getattr(reference, name)) for name in
              ("x_mean", "x_scale", "y_mean", "y_scale", "coefficients")}
    shapes = dict(x_mean=(p,), x_scale=(p,), y_mean=(d,), y_scale=(d,), coefficients=(p, d))
    if (any(value.shape != shapes[name] or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all() for name, value in arrays.items())
            or np.any(arrays["x_scale"] <= 0) or np.any(arrays["y_scale"] <= 0)):
        raise ValueError("finite fitting-only normalization matching the correction is required")
    return arrays


def standardized_ridge_penalty(model: ResidualMemoryPopulationState, reference: RidgeModel) -> Tensor:
    """Mean-output squared standardized coefficients; intercept is unpenalized.

    The caller supplies the new objective weight. The old ridge alpha describes
    initialization and is not a selected regularization weight for this loss.
    """
    arrays = _normalization(model, reference)
    weight = model.correction_weight
    x_scale = torch.as_tensor(arrays["x_scale"], device=weight.device, dtype=weight.dtype)
    y_scale = torch.as_tensor(arrays["y_scale"], device=weight.device, dtype=weight.dtype)
    return (weight.T * x_scale[:, None] / y_scale[None, :]).square().sum() / weight.shape[0]


def export_affine_predictor(model: ResidualMemoryPopulationState, reference: RidgeModel) -> RidgeModel:
    """Export into the unchanged runtime container, with new provenance required.

    The container alpha records ridge initialization only. The caller must bind
    the native objective and selection evidence separately from this container.
    """
    arrays = _normalization(model, reference)
    weight = model.correction_weight.detach().double().cpu().numpy().T
    bias = model.correction_bias.detach().double().cpu().numpy()
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise FloatingPointError("cannot export nonfinite residual coefficients")
    return RidgeModel(reference.alpha, arrays["x_mean"].copy(), arrays["x_scale"].copy(),
                      bias + arrays["x_mean"] @ weight, arrays["y_scale"].copy(),
                      weight * arrays["x_scale"][:, None] / arrays["y_scale"][None, :])


def residual_observation_mixture_loss(
    model: ResidualMemoryPopulationState,
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
    checkpoint_readout: bool = True,
) -> dict[str, Tensor]:
    """Joint native mixture NLL in nats/unit/bin over free corrected rollouts.

    Posterior arrays are [batch,time,field,channel], controls [batch,time,control],
    and labels [batch,origin,max_horizon,unit]. An origin is the number of observed
    bins and must be at least two. Boundary memory is m[origin-1] minus the parent
    transition of m[origin-2]. Each subsequent memory is the new state minus the
    uncorrected parent transition, including the unchanged process draw.

    Sum all unit/time likelihoods before mixing particles. Checkpoint only the
    deterministic readout to bound retained activations. Backward does not replay
    random draws. Particle ESS is diagnostic. Marginal likelihood is approximate.
    """
    if not isinstance(model, ResidualMemoryPopulationState):
        raise TypeError("a native residual-memory wrapper is required")
    if (type(prediction_particles) is not int or prediction_particles < 1
            or type(checkpoint_readout) is not bool):
        raise ValueError("positive particle count and boolean checkpoint_readout required")
    if any(parameter.requires_grad != (name in CORRECTION_PARAMETERS)
           for name, parameter in model.named_parameters()):
        raise ValueError("configure_correction_parameters must select only affine correction parameters")
    parent = model.parent
    mean, variance, controls, origins, horizons, process_variance = _validated_prior_inputs(
        parent, mean, variance, controls, origins, horizons,
        (initial_generator, dynamics_generator), stream_names="initial and dynamics")
    if min(origins) < 2:
        raise ValueError("two observed bins are required for causal residual memory")
    chunk = model.unit_chunk_size if unit_chunk_size is None else unit_chunk_size
    if type(chunk) is not int or chunk < 1:
        raise ValueError("positive integer unit chunk required")
    batch, _, fields, channels = mean.shape
    units, maximum = len(model.unit_ids), max(horizons)
    label_shape = (batch, len(origins), maximum, units)
    labels = (target_counts, target_eligible_ticks)
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (any(not isinstance(value, Tensor) for value in labels)
            or any(value.shape != label_shape or value.device != mean.device for value in labels)
            or any(not value.is_floating_point() and value.dtype not in integer_types for value in labels)):
        raise ValueError("aligned native event counts and eligible ticks required")
    counts, eligible = (value.detach() for value in labels)
    if (any(not torch.isfinite(value).all() for value in (counts, eligible))
            or any(value.is_floating_point() and torch.any(value != value.round()) for value in (counts, eligible))
            or torch.any(counts < 0) or torch.any(eligible < counts) or torch.any(eligible > parent.bin_ticks)):
        raise ValueError("integer counts and eligible ticks within native bin support required")
    if any(not torch.isfinite(getattr(model, name)).all() for name in CORRECTION_PARAMETERS):
        raise FloatingPointError("nonfinite residual parameters")

    indices = torch.tensor(origins, dtype=torch.long, device=mean.device)
    cases = batch * len(origins)
    case_shape = (cases, fields, channels)
    shape = (prediction_particles, *case_shape)
    boundary = mean.index_select(1, indices - 1).reshape(case_shape)
    before = mean.index_select(1, indices - 2).reshape(case_shape)
    previous_control = controls.index_select(1, indices - 1).reshape(cases, model.input_size)
    memory = (boundary - parent.transition_mean(before, previous_control)).expand(shape)
    boundary_variance = variance.index_select(1, indices - 1).reshape(case_shape)
    state = boundary + boundary_variance.sqrt() * torch.randn(
        shape, dtype=mean.dtype, device=mean.device, generator=initial_generator)
    states = []
    for step in range(maximum):
        control = controls.index_select(1, indices + step).reshape(cases, model.input_size)
        base = parent.transition_mean(state, control)
        state = base + model.correction(state, control, memory) + process_variance.sqrt() * torch.randn(
            shape, dtype=mean.dtype, device=mean.device, generator=dynamics_generator)
        memory = state - base
        states.append(state)
    trajectory = torch.stack(states, dim=2)
    counts = counts.to(mean.dtype).reshape(cases, maximum, units)
    eligible = eligible.to(mean.dtype).reshape(cases, maximum, units)
    per_bin = torch.zeros((prediction_particles, cases, maximum), dtype=torch.float64, device=mean.device)
    for start in range(0, units, chunk):
        stop = min(start + chunk, units)
        # Bind indices because checkpoint recomputes the function during backward.
        def score_chunk(path, observed, exposure, start=start, stop=stop):
            rate = parent.log_hazard(path, start, stop)
            return observation_nll(observed.unsqueeze(0).expand_as(rate),
                                   exposure.unsqueeze(0).expand_as(rate), rate, parent.tick_seconds)
        observed, exposure = counts[..., start:stop], eligible[..., start:stop]
        if checkpoint_readout and torch.is_grad_enabled():
            score = checkpoint(score_chunk, trajectory, observed, exposure,
                               use_reentrant=False, preserve_rng_state=False)
        else:
            score = score_chunk(trajectory, observed, exposure)
        per_bin = per_bin + score.double()
    cumulative = per_bin.cumsum(-1)
    terms = {}
    for horizon in horizons:
        joint = cumulative[..., horizon - 1]
        terms[f"native_nll_h{horizon}"] = (
            math.log(prediction_particles) - torch.logsumexp(-joint, dim=0)).mean() / (units * horizon)
        weights = torch.softmax(-joint.detach(), dim=0)
        terms[f"particle_ess_h{horizon}"] = weights.square().sum(0).reciprocal().mean()
    terms["loss"] = torch.stack([terms[f"native_nll_h{h}"] for h in horizons]).mean()
    if any(not torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("nonfinite corrected native likelihood or particle ESS")
    return terms

"""Frozen native mixture loss plus one finite-particle-corrected field mean term.

The field weight is fixed at one. Ridge regularization remains the caller's
existing separate term. Native likelihood arithmetic and public RNG advancement
are preserved by reconstructing its paths with cloned pre-call generators.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .native_renewal_expectation import expected_field_counts
from .population_prior_objective import _validated_prior_inputs
from .population_residual_memory import ResidualMemoryPopulationState
from .population_residual_observation import residual_observation_mixture_loss

FIELD_MEAN_WEIGHT = 1.0


def field_mean_mse_terms(expected_counts: Tensor, observed_counts: Tensor, unit_count: Tensor,
                         standard_deviation_hz: Tensor, bin_seconds: float, horizons: Sequence[int]) -> dict[str, Tensor]:
    """Correct squared ensemble-mean error by between-particle variance / K."""
    if (expected_counts.ndim != 4 or expected_counts.shape[0] < 2 or min(expected_counts.shape) < 1
            or observed_counts.shape != expected_counts.shape[1:]
            or not expected_counts.is_floating_point() or not torch.isfinite(expected_counts).all()
            or torch.any(expected_counts < 0) or not torch.isfinite(observed_counts).all()
            or torch.any(observed_counts < 0)
            or (observed_counts.is_floating_point() and torch.any(observed_counts != observed_counts.round()))):
        raise ValueError("finite complete particle/case/time/field means and integer source observations required")
    fields = expected_counts.shape[-1]
    units = torch.as_tensor(unit_count, device=expected_counts.device)
    sd = torch.as_tensor(standard_deviation_hz, device=expected_counts.device)
    if (units.shape != (fields,) or units.dtype not in (torch.int32, torch.int64) or torch.any(units <= 0)
            or sd.shape != (fields,) or not sd.is_floating_point() or sd.requires_grad
            or not torch.isfinite(sd).all() or torch.any(sd <= 0)):
        raise ValueError("positive source field occupancies and fixed floating Hz standard deviations required")
    scale = units.double() * bin_seconds * sd.detach().double()
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("finite positive original field-count normalization required")
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or not 1 <= h <= expected_counts.shape[2] for h in horizons)):
        raise ValueError("distinct complete field-mean horizons required")
    residual = (expected_counts.double() - observed_counts.detach().double().unsqueeze(0)) / scale
    corrected = residual.mean(0).square() - residual.var(0, unbiased=True) / expected_counts.shape[0]
    terms = {f"field_mean_mse_h{h}": corrected[:, :h].mean() for h in horizons}
    terms["field_mean_loss"] = torch.stack(list(terms.values())).mean()
    if any(not torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError("nonfinite finite-particle-corrected field mean objective")
    return terms


def _free_residual_paths(model, mean, variance, controls, origins, horizons, initial, dynamics, particles):
    """Reconstruct the native loss's exact particle/batch-origin draw layout."""
    parent = model.parent
    mean, variance, controls, origins, horizons, process_variance = _validated_prior_inputs(
        parent, mean, variance, controls, origins, horizons, (initial, dynamics), stream_names="initial and dynamics")
    indices = torch.tensor(origins, dtype=torch.long, device=mean.device)
    cases = mean.shape[0] * len(origins)
    case_shape = (cases, mean.shape[2], mean.shape[3])
    shape = (particles, *case_shape)
    boundary = mean.index_select(1, indices - 1).reshape(case_shape)
    before = mean.index_select(1, indices - 2).reshape(case_shape)
    previous_control = controls.index_select(1, indices - 1).reshape(cases, model.input_size)
    memory = (boundary - parent.transition_mean(before, previous_control)).expand(shape)
    boundary_variance = variance.index_select(1, indices - 1).reshape(case_shape)
    state = boundary + boundary_variance.sqrt() * torch.randn(
        shape, dtype=mean.dtype, device=mean.device, generator=initial)
    states = []
    for step in range(max(horizons)):
        control = controls.index_select(1, indices + step).reshape(cases, model.input_size)
        base = parent.transition_mean(state, control)
        state = base + model.correction(state, control, memory) + process_variance.sqrt() * torch.randn(
            shape, dtype=mean.dtype, device=mean.device, generator=dynamics)
        memory = state - base
        states.append(state)
    return torch.stack(states, dim=2)


def residual_mean_objective(
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
    initial_ready_in_ticks: Tensor,
    field_standard_deviation_hz: Tensor,
    prediction_particles: int = 8,
    unit_chunk_size: int = 4096,
    mean_unit_chunk_size: int = 32768,
    checkpoint_readout: bool = True,
) -> dict[str, Tensor]:
    """Add weight-one standardized field mean MSE to the unchanged native loss.

    Readiness [batch, origin, unit] describes only the observed prefix boundary.
    Hz standard deviations must already follow the authenticated source field
    order. Neither future eligible exposure nor target counts drive mean paths.
    """
    if not isinstance(model, ResidualMemoryPopulationState):
        raise TypeError("a native residual-memory wrapper is required")
    if (not isinstance(mean, Tensor) or mean.ndim != 4
            or type(prediction_particles) is not int or prediction_particles < 2
            or type(mean_unit_chunk_size) is not int or mean_unit_chunk_size < 1):
        raise ValueError("aligned posterior coordinates, at least two particles and a positive mean unit chunk required")
    parent = model.parent
    units, fields = len(parent.unit_ids), len(parent.field_names)
    ready = torch.as_tensor(initial_ready_in_ticks, device=mean.device)
    sd = torch.as_tensor(field_standard_deviation_hz, device=mean.device)
    mapping, intervals = parent.unit_field_index, parent.minimum_interval_ticks
    if (mapping.shape != (units,) or mapping.dtype != torch.int64 or mapping.device != mean.device
            or intervals.shape != (units,) or intervals.device != mean.device
            or intervals.dtype not in (torch.int32, torch.int64) or torch.any(intervals < 1)
            or torch.any(mapping < 0) or torch.any(mapping >= fields)):
        raise ValueError("authenticated source field map and native intervals required")
    unit_count = torch.bincount(mapping, minlength=fields)
    if torch.any(unit_count <= 0):
        raise ValueError("all source fields must contain identified units")
    if (ready.shape != (mean.shape[0], len(origins), units) or ready.dtype not in (torch.int32, torch.int64)
            or torch.any(ready < 0) or torch.any(ready >= intervals)):
        raise ValueError("integer causal readiness for every batch/origin/unit required")
    if (sd.shape != (fields,) or not sd.is_floating_point() or sd.requires_grad
            or not torch.isfinite(sd).all() or torch.any(sd <= 0)):
        raise ValueError("fixed finite positive field standard deviations in source order required")

    initial_state, dynamics_state = initial_generator.get_state().clone(), dynamics_generator.get_state().clone()
    native = residual_observation_mixture_loss(model, mean, variance, controls, target_counts, target_eligible_ticks,
        origins, horizons, initial_generator, dynamics_generator, prediction_particles=prediction_particles,
        unit_chunk_size=unit_chunk_size, checkpoint_readout=checkpoint_readout)
    replay_initial = torch.Generator(device=initial_generator.device)
    replay_dynamics = torch.Generator(device=dynamics_generator.device)
    replay_initial.set_state(initial_state)
    replay_dynamics.set_state(dynamics_state)
    trajectory = _free_residual_paths(model, mean, variance, controls, origins, horizons,
                                     replay_initial, replay_dynamics, prediction_particles)
    if (not torch.equal(replay_initial.get_state(), initial_generator.get_state())
            or not torch.equal(replay_dynamics.get_state(), dynamics_generator.get_state())):
        raise RuntimeError("reconstructed native trajectory RNG advancement differs")
    particles, cases, maximum, _, channels = trajectory.shape
    path_ready = ready.reshape(cases, units).unsqueeze(0).expand(particles, -1, -1).reshape(particles * cases, units)
    expected = expected_field_counts(parent, trajectory.reshape(particles * cases, maximum, fields, channels), path_ready,
        unit_chunk_size=mean_unit_chunk_size, checkpoint_readout=checkpoint_readout).reshape(particles, cases, maximum, fields)
    counts = target_counts.detach().to(torch.int64).reshape(cases, maximum, units)
    observed = counts.new_zeros((cases, maximum, fields)).index_add(-1, mapping, counts)
    mean_terms = field_mean_mse_terms(expected, observed, unit_count, sd, parent.bin_seconds, horizons)
    terms = dict(native)
    terms.update(mean_terms)
    terms["native_loss"] = native["loss"]
    terms["loss"] = terms["native_loss"] + FIELD_MEAN_WEIGHT * terms["field_mean_loss"]
    if not torch.isfinite(terms["loss"]):
        raise FloatingPointError("nonfinite combined native and field mean objective")
    return terms

"""Causal responses across source-compiled PFC task conditions.

Compilation establishes registered command semantics, not scientific task
qualification. Candidate configuration is an input program; candidate outcomes
and microscopic states are never inputs to boundary inference.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from .circuit_data import _control_bins
from .pfc_deterministic_replay import _Arithmetic, _control, _held_current, _STIMULUS_DT
from .pfc_population_evaluation import (
    _registered_source_block, prepare_factual_response, source_controls,
)
from .pfc_renewal_data import population_control_features, PfcRenewalEpisode
from .pfc_source_dataset import _benchmark_registration, _controls
from .pfc_source_execution import _case_config
from .provenance import canonical_json


def candidate_population_metadata(data, config, control_features):
    """Apply the fitted source metadata method to a known candidate program.

    The facade exposes static configuration and admitted identities. It exposes
    no event arrays, windows, RNG state or microscopic source states.
    """
    static = SimpleNamespace(source_config=config, unit_ids=data.unit_ids,
        unit_field_index=data.unit_field_index, field_names=data.field_names,
        model_control_features=control_features, binding=data.binding, data=data.data)
    return PfcRenewalEpisode.population_metadata(static)


def _program(config, case, dimensions):
    stride = int(round(_STIMULUS_DT / dimensions["dt_seconds"]))
    ticks = np.arange(0, dimensions["duration_ticks"], stride, dtype=np.int64)
    native = _control(config, case, dimensions["n_e"], _Arithmetic("numpy"))
    times = np.arange(len(ticks)) * _STIMULUS_DT
    flags = np.column_stack([
        (times >= native[prefix]["start"]) & (times < native[prefix]["end"])
        for prefix in ("stimulus", "distractor")
    ])
    changes = np.r_[True, np.any(flags[1:] != flags[:-1], axis=1)]
    starts, values, features = _controls(config, case, dict(
        segment_start_ticks=ticks[changes], cue_active=flags[changes, 0],
        distractor_active=flags[changes, 1]))
    return dict(native=native, stride=stride, starts=starts,
                stops=np.r_[starts[1:], dimensions["duration_ticks"]],
                values=values, features=features)


def _unchanged_physics(config):
    # Only fields produced by the task compiler can differ. In particular,
    # kernel parameters, source execution, maps, clocks and extra fields remain.
    result = deepcopy(config)
    for name in ("benchmark_episode_candidate", "cases", "stimulus_strength_nA",
                 "distractor_strength_nA", "distractor_center_deg", "distractor_width_deg",
                 "distractor_start_ms", "distractor_duration_ms"):
        result.pop(name, None)
    return result


def _same_prefix(factual, candidate, origin, n_e):
    if factual["stride"] != candidate["stride"]:
        raise ValueError("PFC_COMPILED_RESPONSE_OPERATION_CLOCK_CHANGED")
    left, right = np.zeros(n_e), np.zeros(n_e)
    # Reuse the native assignment/reset/overwrite operation. Comparison of
    # scalar bin means would miss target geometry and held currents.
    for tick in np.union1d(factual["starts"], candidate["starts"]):
        if tick >= origin:
            break
        # Repeated assignment within a segment is idempotent. Apply each
        # source operation at a flag change, retaining previous held currents.
        time = int(tick // factual["stride"]) * _STIMULUS_DT
        a = _held_current(left, factual["native"], time)
        b = _held_current(right, candidate["native"], time)
        if a != b or not np.array_equal(left, right):
            raise ValueError("PFC_COMPILED_RESPONSE_SOURCE_UNIT_COMMAND_PREFIX_CHANGED")


@torch.no_grad()
def prepare_compiled_response(view, data, origin: int, horizon_bins: int, candidate_config: dict):
    """Return candidate view, transferred factual boundary and known commands.

    Both configurations must reproduce the existing task compiler. The candidate
    retains every non-control parameter and the same episode block and split.
    Boundary inference runs once on the factual view. Only an explicitly checked
    geometry binding changes on the returned boundary; its tensors are retained.
    """
    if (type(origin) is not int or type(horizon_bins) is not int or origin < 1 or horizon_bins < 1
            or origin % data.bin_ticks or origin + horizon_bins * data.bin_ticks > data.duration_ticks):
        raise ValueError("PFC_FORECAST_COMPLETE_ALIGNED_PREFIX_AND_FUTURE_REQUIRED")
    config = data.source_config
    binding = data.data.catalog["binding"]
    if (not isinstance(candidate_config, dict)
            or "benchmark_episode_candidate" not in config
            or "benchmark_episode_candidate" not in candidate_config
            or binding.get("resolved_source_config") != config
            or config.get("expected_source_revision") != data.binding["source_revision"]):
        raise ValueError("PFC_COMPILED_RESPONSE_AUTHENTICATED_COMPILED_SOURCE_REQUIRED")
    _benchmark_registration(config)
    _benchmark_registration(candidate_config)
    identity = config["benchmark_episode_candidate"]["identity"]
    target_identity = candidate_config["benchmark_episode_candidate"]["identity"]
    if (identity["episode_id"] != data.episode_id
            or {k: v for k, v in identity.items() if k not in ("episode_id", "condition_index")}
            != {k: v for k, v in target_identity.items() if k not in ("episode_id", "condition_index")}):
        raise ValueError("PFC_COMPILED_RESPONSE_SAME_CIRCUIT_BLOCK_AND_SPLIT_REQUIRED")
    if _unchanged_physics(config) != _unchanged_physics(candidate_config):
        raise ValueError("PFC_COMPILED_RESPONSE_NONCONTROL_SOURCE_PARAMETER_CHANGED")
    factual_case, _, dimensions = _case_config(config, data.episode_id)
    candidate_id = target_identity["episode_id"]
    candidate_case, _, candidate_dimensions = _case_config(candidate_config, candidate_id)
    factual_block = _registered_source_block(config, factual_case)
    candidate_block = _registered_source_block(candidate_config, candidate_case)
    if (candidate_dimensions != dimensions or factual_block != candidate_block
            or factual_block["block_id"] != data.binding["block_id"]
            or dimensions["duration_ticks"] != data.duration_ticks
            or dimensions["dt_seconds"] != data.dt_ms / 1000):
        raise ValueError("PFC_COMPILED_RESPONSE_SOURCE_DIMENSIONS_OR_RANDOM_BLOCK_CHANGED")
    factual, candidate = (_program(value, case, dimensions)
                          for value, case in ((config, factual_case), (candidate_config, candidate_case)))
    observed = source_controls(data, 0, data.duration_ticks)
    model_features = population_control_features(config, factual["features"])
    candidate_features = population_control_features(candidate_config, candidate["features"])
    if (view.metadata is None or view.metadata != data.population_metadata()
            or view.metadata["control_features"] != model_features
            or observed["control_features"] != model_features or candidate_features != model_features
            or not all(np.array_equal(observed[key], factual[name]) for key, name in (
                ("segment_start_ticks", "starts"), ("segment_stop_ticks", "stops"),
                ("segment_values", "values")))):
        raise ValueError("PFC_COMPILED_RESPONSE_FACTUAL_SOURCE_COMMAND_BINDING_CHANGED")
    _same_prefix(factual, candidate, origin, dimensions["n_e"])
    # Metadata is constructed from known source geometry, never candidate events.
    metadata = candidate_population_metadata(data, candidate_config, candidate_features)
    candidate_view = view.training_for_circuit(unit_ids=list(map(int, data.unit_ids)),
        unit_field_index=data.unit_field_index.tolist(), field_names=list(data.field_names),
        minimum_interval_ticks=data.minimum_interval_ticks.tolist(), metadata=metadata).eval()
    boundary, _ = prepare_factual_response(view, data, origin, horizon_bins)
    if (boundary.source_metadata_identity != view.source_metadata_identity
            or candidate_view.unit_ids != boundary.unit_ids
            or candidate_view.field_names != boundary.field_names
            or candidate_view.bin_seconds != boundary.bin_seconds
            or not torch.equal(candidate_view.unit_field_index, view.unit_field_index)
            or not torch.equal(candidate_view.minimum_interval_ticks, view.minimum_interval_ticks)
            or any(parameter is not dict(candidate_view.named_parameters())[name]
                   for name, parameter in view.named_parameters())):
        raise ValueError("PFC_COMPILED_RESPONSE_BOUNDARY_TRANSFER_BINDING_CHANGED")
    boundary = replace(boundary, source_metadata_identity=candidate_view.source_metadata_identity)
    stop = origin + horizon_bins * data.bin_ticks
    selected = (candidate["starts"] < stop) & (candidate["stops"] > origin)
    future = dict(known_controls=_control_bins(candidate["starts"], candidate["values"], "piecewise_constant",
                    origin, stop, data.bin_ticks),
        control_features=candidate_features, source_control_features=list(candidate["features"]),
        segment_start_ticks=candidate["starts"][selected], segment_stop_ticks=candidate["stops"][selected],
        segment_values=candidate["values"][selected], start_tick=origin, stop_tick_exclusive=stop,
        control_kind="piecewise_constant", source_binding=data.binding,
        registered_response_binding=dict(factual_case_id=data.episode_id, candidate_case_id=candidate_id,
            factual_config=canonical_json(config), candidate_config=canonical_json(candidate_config),
            source_rng_block=candidate_block, factual_source_metadata_identity=view.source_metadata_identity,
            candidate_source_metadata_identity=candidate_view.source_metadata_identity,
            source_dt_ms=data.dt_ms, stimulus_operation_stride_ticks=candidate["stride"],
            command_units="nA", control_domain="source_compiled_discrete_task_conditions",
            prefix_comparison="exact_source_E_assignment_reset_overwrite_current_and_activation_flags",
            boundary_transfer="unchanged_factual_mean_variance_and_readiness_with_checked_candidate_geometry",
            candidate_outcomes_read=False, scientific_intervention_qualified=False))
    return candidate_view, boundary, future

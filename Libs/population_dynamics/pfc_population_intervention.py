"""Rank predeclared compiled PFC programs against independent field references.

The selector uses the factual prefix and known candidate programs. It returns
an executable source configuration and every candidate prediction. Rankings
remain development predictions until independently executed and qualified.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

from .pfc_compiled_response import prepare_compiled_response
from .population_intervention import (
    ReferenceTarget, reference_energy_score, sample_candidate, target_arrays,
)
from .population_state import ControlledPopulationState


@dataclass(frozen=True)
class PfcInterventionCandidate:
    name: str
    source_config: dict


def _actuation_model(view):
    if (not isinstance(view, ControlledPopulationState)
            or view.control_geometry != "actuation_only"
            or view.metadata is None or not view.source_metadata_identity
            or not hasattr(view, "_unit_target_columns")
            or not hasattr(view, "_field_target_columns")):
        raise ValueError("PFC_INTERVENTION_SOURCE_BOUND_ACTUATION_ONLY_MODEL_REQUIRED")


def _effective_controls(view, known_controls, horizon):
    """Compare the model's field inputs before any fitted control weights."""
    controls = view._tensor(known_controls)
    if (controls.shape != (horizon, view.input_size)
            or not torch.isfinite(controls).all()):
        raise ValueError("PFC_INTERVENTION_FINITE_COMPLETE_CONTROL_INPUT_REQUIRED")
    fractions = view.field_metadata[:, view._field_target_columns]
    effective = controls[:, None, :] * fractions[None, :, (0, 1, 0, 1)]
    if not torch.isfinite(effective).all():
        raise ValueError("PFC_INTERVENTION_FINITE_COMPLETE_CONTROL_INPUT_REQUIRED")
    return effective


def _validate_prediction(prediction, view, horizon, draws):
    shapes = {
        "field_counts": (draws, horizon, len(view.field_names)),
        "field_rate_hz": (draws, horizon, len(view.field_names)),
        "unit_total_counts": (draws, len(view.unit_ids)),
        "terminal_ready_in_ticks": (draws, len(view.unit_ids)),
    }
    for name, shape in shapes.items():
        value = np.asarray(prediction[name])
        if (value.shape != shape or np.iscomplexobj(value)
                or not np.isfinite(value).all() or np.any(value < 0)
                or (name != "field_rate_hz" and np.any(value != np.floor(value)))):
            raise ValueError("PFC_INTERVENTION_FINITE_COMPLETE_PREDICTION_REQUIRED:" + name)
    if (prediction["generated_bins"] != horizon
            or prediction["generated_unit_count_values"] != draws * horizon * len(view.unit_ids)):
        raise ValueError("PFC_INTERVENTION_COMPLETE_UNIT_GENERATION_DENOMINATOR_REQUIRED")
    seconds = prediction["generation_and_projection_seconds"]
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("PFC_INTERVENTION_FINITE_GENERATION_COST_REQUIRED")


@torch.no_grad()
def design_pfc_intervention(
    view: ControlledPopulationState, factual_data, target: ReferenceTarget, *,
    origin: int, candidates, stream_seeds: tuple[int, int, int] = (41, 42, 43),
    draws: int,
):
    """Select the minimum reference energy score from a fixed candidate set.

    The target supplies the common horizon and externally frozen field scales.
    Every program must pass the source adapter before any candidate is sampled.
    Each candidate uses the same declared initial/process/observation streams.
    Equal scores select the lexically first candidate name. No physical cost,
    success tolerance, native execution or scientific qualification is inferred.

    Return (report, predictions_by_name). Predictions retain field trajectories,
    per-unit horizon totals and terminal clocks. All per-bin unit counts are
    generated; those per-bin unit arrays are not retained.
    """
    _actuation_model(view)
    if (not isinstance(target, ReferenceTarget)
            or type(draws) is not int or draws < 2
            or not isinstance(stream_seeds, (tuple, list)) or len(stream_seeds) != 3
            or any(type(seed) is not int for seed in stream_seeds)
            or set(stream_seeds) != {41, 42, 43}):
        raise ValueError("PFC_INTERVENTION_REFERENCE_MULTIPLE_DRAWS_AND_SEEDS_41_42_43_REQUIRED")
    rates, scales = target_arrays(target, view)
    target = ReferenceTarget(rates.copy(), tuple(target.field_names), scales.copy(),
                             tuple(target.independent_block_ids), target.bin_seconds)
    horizon = rates.shape[1]
    candidates = tuple(candidates)
    if (not candidates
            or any(not isinstance(candidate, PfcInterventionCandidate)
                   or not isinstance(candidate.name, str) or not candidate.name
                   or not isinstance(candidate.source_config, dict) for candidate in candidates)
            or len({candidate.name for candidate in candidates}) != len(candidates)):
        raise ValueError("PFC_INTERVENTION_DISTINCT_NAMED_COMPILED_CANDIDATES_REQUIRED")
    candidates = tuple(PfcInterventionCandidate(row.name, deepcopy(row.source_config))
                       for row in candidates)

    prepared, effective_inputs, case_ids = [], [], set()
    for candidate in candidates:
        candidate_view, boundary, future = prepare_compiled_response(
            view, factual_data, origin, horizon, candidate.source_config)
        _actuation_model(candidate_view)
        case_id = future["registered_response_binding"]["candidate_case_id"]
        if case_id in case_ids:
            raise ValueError("PFC_INTERVENTION_DISTINCT_SOURCE_CANDIDATE_IDENTITIES_REQUIRED")
        case_ids.add(case_id)
        effective = _effective_controls(candidate_view, future["known_controls"], horizon)
        if any(torch.equal(effective, previous) for previous in effective_inputs):
            raise ValueError("PFC_INTERVENTION_CANDIDATES_INDISTINGUISHABLE_AT_MODEL_INPUT_RESOLUTION")
        effective_inputs.append(effective)
        if prepared and any(not torch.equal(getattr(boundary, name), getattr(prepared[0][2], name))
                            for name in ("mean", "variance", "ready_in_ticks")):
            raise ValueError("PFC_INTERVENTION_COMMON_FACTUAL_BOUNDARY_REQUIRED")
        prepared.append((candidate, candidate_view, boundary, future))

    rows, predictions = [], {}
    for candidate, candidate_view, boundary, future in prepared:
        prediction = sample_candidate(candidate_view, boundary,
                                      future["known_controls"][None], stream_seeds, draws)
        _validate_prediction(prediction, candidate_view, horizon, draws)
        score, block_scores = reference_energy_score(prediction["field_rate_hz"], target, candidate_view)
        if (not math.isfinite(score) or len(block_scores) != len(target.independent_block_ids)
                or not np.isfinite(block_scores).all()):
            raise FloatingPointError("PFC_INTERVENTION_FINITE_COMPLETE_REFERENCE_SCORES_REQUIRED")
        rows.append(dict(
            candidate=candidate.name, candidate_case_id=future["registered_response_binding"]["candidate_case_id"],
            candidate_identity=deepcopy(candidate.source_config["benchmark_episode_candidate"]["identity"]),
            source_config=deepcopy(candidate.source_config),
            registered_response_binding=deepcopy(future["registered_response_binding"]),
            reference_energy_score=score, reference_block_scores=block_scores, objective=score,
            generated_unit_count_values=prediction["generated_unit_count_values"],
            generation_and_projection_seconds=prediction["generation_and_projection_seconds"],
            peak_gpu_allocated_bytes=prediction["peak_gpu_allocated_bytes"],
        ))
        predictions[candidate.name] = prediction

    ranking = sorted(rows, key=lambda row: (row["objective"], row["candidate"]))
    chosen = ranking[0]
    report = dict(
        status="RANKED_DEVELOPMENT_COMPILED_PFC_CANDIDATES",
        selected_candidate=chosen["candidate"],
        selected_action=dict(
            candidate_case_id=chosen["candidate_case_id"], source_config=deepcopy(chosen["source_config"]),
            registered_response_binding=deepcopy(chosen["registered_response_binding"]),
            source_binding=deepcopy(factual_data.binding), origin_tick=origin, horizon_bins=horizon,
        ),
        candidates=rows, candidate_ranking=[row["candidate"] for row in ranking],
        candidate_count=len(rows), objective="reference_energy_score",
        tie_break="lexical_candidate_name", sampling_stream_seeds=list(stream_seeds), joint_draws=draws,
        origin_tick=origin, horizon_bins=horizon, source_unit_count=len(view.unit_ids),
        field_names=list(view.field_names), model_source_metadata_identity=view.source_metadata_identity,
        model_control_geometry=view.control_geometry, source_binding=deepcopy(factual_data.binding),
        reference_target=dict(field_rate_hz=target.field_rate_hz.tolist(), scale_hz=target.scale_hz.tolist(),
                              independent_block_ids=list(target.independent_block_ids),
                              field_names=list(target.field_names), bin_seconds=target.bin_seconds),
        candidate_resolution="binned_controls_times_source_field_target_fractions",
        model_common_streams=True, full_unit_outputs_generated=True,
        retained_prediction_arrays=["field_counts", "field_rate_hz", "unit_total_counts", "terminal_ready_in_ticks"],
        per_bin_unit_counts_retained=False, all_candidate_predictions_retained=True,
        candidate_future_outcomes_read=False, native_future_noise_pairing_verified=False,
        native_execution_performed=False, native_execution_qualified=False,
        success_tolerance_qualified=False, inverse_intervention_qualified=False,
        scientific_claim_qualified=False,
    )
    return report, predictions

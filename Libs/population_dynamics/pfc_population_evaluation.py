"""PFC source admission and registered interventions for shared forecast/replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .pfc_population_training_data import PfcPopulationTrainingCohort
from .pfc_renewal_data import population_control_features
from .pfc_source_execution import _case_config as _registered_source_case
from .pfc_source_dataset import _controls as _source_command_representation
from .pfc_deterministic_replay import _STIMULUS_DT
from .circuit_data import _control_bins
from .provenance import sha256_file
from . import source_population_evaluation as shared
from .source_population_evaluation import (
    source_controls, prepare_factual_response, _forecast, _scores,
    _case_cost_contract, validate_forecast_costs,
)

EVALUATION_STATUS = "EVALUATED_PFC_DEVELOPMENT_CHECKPOINT"

def _load(inputs, training, device, *, resource_delivery=None, prepared_inputs=None):
    run = json.loads((training / "run.json").read_text())
    resource_diagnostic = resource_delivery is not None
    if resource_diagnostic:
        # A terminal resource attempt can retain a committed checkpoint while
        # its final training report still says RUNNING. Admit it only through
        # the archived delivery and exact checkpoint/report byte bindings.
        receipt = json.loads(Path(resource_delivery).read_text())
        if (receipt.get("status") != "RESOURCE_BLOCKED" or receipt.get("exit_code") != 3
                or receipt.get("training_seed") != run.get("training_seed")
                or receipt.get("input_family") != "pfc_source_reproduction"
                or receipt.get("scientific_claim_qualified") is not False):
            raise ValueError("PFC_DIAGNOSTIC_TERMINAL_RESOURCE_DELIVERY_REQUIRED")
        for name in ("training/run.json", "training/best.pt"):
            entries = [row for row in receipt["files"] if row["path"] == name]
            bound = training.parent / name
            if (len(entries) != 1 or bound.stat().st_size != entries[0]["bytes"]
                    or sha256_file(bound) != entries[0]["sha256"]):
                raise ValueError("PFC_DIAGNOSTIC_COMMITTED_RESOURCE_ARTIFACT_CHANGED:" + name)
        if type(run.get("completed_epochs")) is not int or run["completed_epochs"] < 1:
            raise ValueError("PFC_DIAGNOSTIC_COMMITTED_EPOCH_REQUIRED")
    if (run.get("input_family") != "pfc_source_reproduction"
            or (run["status"] not in ("CHECKPOINTED_DEVELOPMENT", "FITTED_DEVELOPMENT")
                and not (resource_diagnostic and run["status"] == "RUNNING"))):
        raise ValueError("COMMITTED_PFC_DEVELOPMENT_TRAINING_REQUIRED")
    path = training / "best.pt"
    if sha256_file(path) != run["best_checkpoint_sha256"]:
        raise ValueError("PFC_EVALUATION_CHECKPOINT_CHANGED")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["binding"] != run["binding"] or checkpoint["next_epoch"] != run["best_epoch"] + 1:
        raise ValueError("PFC_EVALUATION_CHECKPOINT_AND_REPORT_DIFFER")
    config = checkpoint["binding"]["config"]
    cohort = (PfcPopulationTrainingCohort(inputs, config) if prepared_inputs is None
              else prepared_inputs)
    model = shared.restore_source_model(checkpoint, run, cohort, device, error_prefix="PFC")
    return model, cohort, run

def _config(path):
    return shared.validate_config(yaml.safe_load(path.read_text()),
        schema="pfc_population_checkpoint_forecast_v1",
        evaluation_scope="declared_development_validation_checkpoint")

def _registered_source_block(config, case):
    """Resolve source RNG identity without creating or advancing random streams."""
    policy = config.get("random_stream_policy", "source_global")
    if case.get("random_stream_policy", policy) != policy:
        raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_RNG_POLICY_CHANGED")
    def seed(name):
        value = case.get(name, config.get(name))
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_SEED_INVALID:" + name)
        return value
    if policy == "source_global":
        if any(name in item for item in (config, case)
               for name in ("initial_rng_seed", "external_noise_seed")):
            raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_RNG_POLICY_CHANGED")
        values = dict(source_rng_seed=seed("source_rng_seed"))
        block = "pfc_source_block_" + str(values["source_rng_seed"])
    elif policy == "independent_initial_external":
        if "source_rng_seed" in config:
            raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_RNG_POLICY_CHANGED")
        values = dict(initial_rng_seed=seed("initial_rng_seed"), external_noise_seed=seed("external_noise_seed"))
        # Match the native driver's explicitly labelled, resolved external alias.
        if "source_rng_seed" in case and (case.get("random_stream_policy") != policy
                                         or case["source_rng_seed"] != values["external_noise_seed"]):
            raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_SEED_ALIAS_CHANGED")
        block = "pfc_initial_" + str(values["initial_rng_seed"]) + "_external_" + str(values["external_noise_seed"])
    else:
        raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_RNG_POLICY_CHANGED")
    return dict(random_stream_policy=policy, seeds=values, block_id=block)


def _registered_amplitudes(config, case):
    """Accept only the native case fields that can change the registered command."""
    allowed = {"case_id", "condition", "block_id", "random_stream_policy", "source_rng_seed",
               "initial_rng_seed", "external_noise_seed", "stimulus_strength_nA", "distractor_strength_nA"}
    for name, value in case.items():
        if name not in allowed and (name not in config or value != config[name]):
            raise ValueError("PFC_REGISTERED_RESPONSE_CASE_PROGRAM_OVERRIDE_UNSUPPORTED:" + name)
    result = dict(case)
    for prefix, default in (("stimulus", .07), ("distractor", 0.)):
        value = case.get(prefix + "_strength_nA", config.get(prefix + "_strength_nA", default))
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("PFC_REGISTERED_RESPONSE_FINITE_NONNEGATIVE_SCALAR_NA_REQUIRED")
        result[prefix + "_strength_nA"] = float(value)
    return result


@torch.no_grad()
def prepare_registered_response(view, data, origin: int, horizon_bins: int, candidate_case_id: str):
    """Bind a registered discrete command to an unchanged factual causal prefix.

    The candidate uses the same authenticated source configuration and source RNG
    block. Its whole-case command must agree with the factual command before the
    origin. This interface reads no candidate or factual future spike outcomes.
    Returned amplitudes are source commands in nA, not per-unit delivered current.
    """
    if (type(origin) is not int or type(horizon_bins) is not int or origin < 1 or horizon_bins < 1
            or origin % data.bin_ticks or origin + horizon_bins * data.bin_ticks > data.duration_ticks):
        raise ValueError("PFC_FORECAST_COMPLETE_ALIGNED_PREFIX_AND_FUTURE_REQUIRED")
    config = data.source_config
    catalog_binding = data.data.catalog["binding"]
    if (not isinstance(candidate_case_id, str)
            or catalog_binding.get("resolved_source_config") != config
            or config.get("expected_source_revision") != data.binding["source_revision"]):
        raise ValueError("PFC_REGISTERED_RESPONSE_AUTHENTICATED_SOURCE_CONFIG_REQUIRED")
    candidate, _, dimensions = _registered_source_case(config, candidate_case_id)
    factual, _, _ = _registered_source_case(config, data.episode_id)
    candidate = _registered_amplitudes(config, candidate)
    factual = _registered_amplitudes(config, factual)
    candidate_block = _registered_source_block(config, candidate)
    factual_block = _registered_source_block(config, factual)
    if candidate_block != factual_block or factual_block["block_id"] != data.binding["block_id"]:
        raise ValueError("PFC_REGISTERED_RESPONSE_SAME_INITIAL_EXTERNAL_BLOCK_REQUIRED")
    if (dimensions["duration_ticks"] != data.duration_ticks
            or dimensions["dt_seconds"] != data.dt_ms / 1000
            or view.metadata is None):
        raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_TIME_OR_FEATURE_SCHEMA_CHANGED")
    windows = []
    for prefix in ("stimulus", "distractor"):
        start = config.get(prefix + "_start_ms", 0.)
        duration = config.get(prefix + "_duration_ms", 0.)
        if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in (start, duration)):
            raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_TIME_OR_FEATURE_SCHEMA_CHANGED")
        windows.append((start * 1e-3, start * 1e-3 + duration * 1e-3))
    # Source admission has already authenticated these native segments. Recheck
    # their configuration binding without touching a spike or microscopic array.
    program = source_controls(data, 0, data.duration_ticks)
    stride = int(round(_STIMULUS_DT / dimensions["dt_seconds"]))
    operation_ticks = np.arange(0, data.duration_ticks, stride, dtype=np.int64)
    operation_times = np.arange(len(operation_ticks)) * _STIMULUS_DT
    flags = np.column_stack([((operation_times >= start) & (operation_times < stop)) for start, stop in windows])
    changes = np.r_[True, np.any(flags[1:] != flags[:-1], axis=1)]
    starts = operation_ticks[changes]
    stops = np.r_[starts[1:], data.duration_ticks]
    observed = dict(segment_start_ticks=starts, cue_active=flags[changes, 0], distractor_active=flags[changes, 1])
    _, factual_values, features = _source_command_representation(config, factual, observed)
    _, candidate_values, _ = _source_command_representation(config, candidate, observed)
    model_features = population_control_features(config, features)
    if (program["control_features"] != model_features
            or view.metadata["control_features"] != model_features
            or not np.array_equal(program["segment_start_ticks"], starts)
            or not np.array_equal(program["segment_stop_ticks"], stops)
            or not np.array_equal(program["segment_values"], factual_values)):
        raise ValueError("PFC_REGISTERED_RESPONSE_SOURCE_TIME_OR_FEATURE_SCHEMA_CHANGED")
    prefix_segments = starts < origin
    if not np.array_equal(candidate_values[prefix_segments], factual_values[prefix_segments]):
        raise ValueError("PFC_REGISTERED_RESPONSE_CANDIDATE_PREFIX_CHANGED")
    # All candidate rejection above precedes the factual prefix outcome read.
    boundary, future = prepare_factual_response(view, data, origin, horizon_bins)
    stop_tick = origin + horizon_bins * data.bin_ticks
    selected = (starts < stop_tick) & (stops > origin)
    future = dict(future, known_controls=_control_bins(starts, candidate_values, "piecewise_constant",
                                                       origin, stop_tick, data.bin_ticks),
                  segment_values=candidate_values[selected], registered_response_binding=dict(
                      factual_case_id=data.episode_id, candidate_case_id=candidate_case_id,
                      source_config_sha256=catalog_binding["config_sha256"], source_rng_block=candidate_block,
                      source_metadata_identity=view.source_metadata_identity, source_dt_ms=data.dt_ms,
                      stimulus_operation_stride_ticks=stride, control_features=model_features,
                      source_control_features=list(features),
                      command_units="nA", control_domain="registered_discrete_cases",
                      timing="after_source_stimulus_operation_before_rk2", segment_bounds="start_inclusive_stop_exclusive",
                      cue_assignment="active: assign cue amplitude to source cue indices; inactive: clear all E I_stim",
                      distractor_assignment="active: assign distractor amplitude after cue assignment/reset",
                      held_current="while cue active, other E currents persist unless overwritten",
                      scientific_intervention_qualified=False))
    return boundary, future


def evaluate(inputs: Path, training: Path, config_path: Path, output: Path, device_name="cpu"):
    return shared.evaluate(inputs, training, config_path, output, device_name,
        load=_load, read_config=_config, evaluator_path=Path(__file__), evaluation_status=EVALUATION_STATUS)


def replay(inputs: Path, training: Path, forecasts: Path, output: Path, device_name="cpu"):
    return shared.replay(inputs, training, forecasts, output, device_name,
        load=_load, evaluator_path=Path(__file__), evaluation_status=EVALUATION_STATUS)

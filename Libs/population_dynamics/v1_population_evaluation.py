"""Original-validation V1 checkpoint admission for shared all-unit forecast/replay."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from .v1_population_training_data import V1PopulationTrainingCohort, SCHEMA as TRAINING_SCHEMA
from .provenance import sha256_file
from . import source_population_evaluation as shared

SCHEMA = "v1_population_checkpoint_forecast_v1"
EVALUATION_STATUS = "EVALUATED_V1_DEVELOPMENT_CHECKPOINT"


def _config(path):
    value = shared.validate_config(yaml.safe_load(path.read_text()), schema=SCHEMA,
                                   evaluation_scope="original_validation_checkpoint")
    if value.get("random_seed_policy") != "exact_registered_stream_seeds" or value["sampling_seed_base"] != 41:
        raise ValueError("V1_FORECAST_STREAM_SEEDS_MUST_BE_41_42_43")
    return value


def _load(inputs, training, device):
    run = json.loads((training / "run.json").read_text())
    if (run.get("input_family") != "v1_qualified_source_episodes"
            or run.get("status") not in ("CHECKPOINTED_DEVELOPMENT", "FITTED_DEVELOPMENT")
            or run["binding"]["config"].get("schema") != TRAINING_SCHEMA):
        raise ValueError("COMMITTED_V1_DEVELOPMENT_TRAINING_REQUIRED")
    # Original validation membership is checked before source arrays or weights.
    cohort = V1PopulationTrainingCohort(inputs, run["binding"]["config"])
    path = training / "best.pt"
    if sha256_file(path) != run["best_checkpoint_sha256"]:
        raise ValueError("V1_EVALUATION_CHECKPOINT_CHANGED")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint["binding"] != run["binding"] or checkpoint["next_epoch"] != run["best_epoch"] + 1:
        raise ValueError("V1_EVALUATION_CHECKPOINT_AND_REPORT_DIFFER")
    model = shared.restore_source_model(checkpoint, run, cohort, device, error_prefix="V1")
    return model, cohort, run


def evaluate(inputs: Path, training: Path, config_path: Path, output: Path, device_name="cpu"):
    return shared.evaluate(inputs, training, config_path, output, device_name,
        load=_load, read_config=_config, evaluator_path=Path(__file__), evaluation_status=EVALUATION_STATUS)


def replay(inputs: Path, training: Path, forecasts: Path, output: Path, device_name="cpu"):
    return shared.replay(inputs, training, forecasts, output, device_name,
        load=_load, evaluator_path=Path(__file__), evaluation_status=EVALUATION_STATUS)

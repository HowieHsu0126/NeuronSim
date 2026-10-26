"""Small, explicit configuration boundary for real neural population data intake."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

TRAINING_SEEDS = [41, 42, 43]


def load_config(path: Path, *, validate_training_seeds: bool = True) -> dict[str, Any]:
    """Validate shared settings; source/BENCH readers do not use training seeds."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CONFIG_MAPPING_REQUIRED")
    if value.get("campaign") != "neural_population_dynamics":
        raise ValueError("CONFIG_CAMPAIGN_MISMATCH")
    if validate_training_seeds and (value.get("training_seeds") != TRAINING_SEEDS or any(
        type(seed) is not int for seed in value.get("training_seeds", [])
    )):
        raise ValueError("TRAINING_SEEDS_MUST_BE_41_42_43")
    for name in ("cohort", "network", "unit_qc", "eligibility"):
        if not isinstance(value.get(name), dict):
            raise ValueError(f"CONFIG_MAPPING_REQUIRED:{name}")
    if "opto_source" in value:
        opto = value["opto_source"]
        if not isinstance(opto, dict):
            raise ValueError("CONFIG_MAPPING_REQUIRED:opto_source")
        for name in ("engineering_anchor_subject_id", "required_future_split", "source_origin",
                     "source_bucket", "nwb_prefix"):
            if not isinstance(opto.get(name), str) or not opto[name].strip():
                raise ValueError(f"OPTO_SOURCE_NONEMPTY_STRING_REQUIRED:{name}")
        if opto["required_future_split"] != "development_training_only":
            raise ValueError("OPTO_SOURCE_SPLIT_MUST_BE_DEVELOPMENT_TRAINING_ONLY")
        prefix = opto["nwb_prefix"]
        if prefix.startswith("/") or "\\" in prefix or ".." in prefix.split("/"):
            raise ValueError("OPTO_SOURCE_NWB_PREFIX_INVALID")
    if "opto_window_check" in value:
        window = value["opto_window_check"]
        if not isinstance(window, dict):
            raise ValueError("CONFIG_MAPPING_REQUIRED:opto_window_check")
        for name in ("history_seconds", "horizon_seconds", "bin_seconds"):
            number = window.get(name)
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"OPTO_WINDOW_POSITIVE_FINITE_SECONDS_REQUIRED:{name}")
        for name in ("history_seconds", "horizon_seconds"):
            bins = window[name] / window["bin_seconds"]
            if (not math.isfinite(bins) or round(bins) < 1
                    or not math.isclose(bins, round(bins), rel_tol=0.0, abs_tol=1e-9)):
                raise ValueError(f"OPTO_WINDOW_INTEGER_BIN_COUNT_REQUIRED:{name}")
    if "mesocircuit_field_check" in value:
        field = value["mesocircuit_field_check"]
        if not isinstance(field, dict) or set(field) != {
            "onset_seconds", "history_seconds", "horizon_seconds", "bin_seconds",
            "cells_per_axis", "unit_chunk_size"
        }:
            raise ValueError("MESOCIRCUIT_FIELD_CONFIG_FIELDS_INVALID")
        for name in ("onset_seconds", "history_seconds", "horizon_seconds", "bin_seconds"):
            number = field[name]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (name != "onset_seconds" and number == 0):
                raise ValueError("MESOCIRCUIT_FIELD_SECONDS_INVALID:" + name)
        for name in ("cells_per_axis", "unit_chunk_size"):
            if type(field[name]) is not int or field[name] < 1:
                raise ValueError("MESOCIRCUIT_FIELD_POSITIVE_INTEGER_REQUIRED:" + name)
    if "population_field_check" in value:
        field = value["population_field_check"]
        if not isinstance(field, dict) or field.get("group_by") not in ("brain_region", "device_name"):
            raise ValueError("POPULATION_FIELD_SOURCE_GROUP_REQUIRED")
        if type(field.get("unit_chunk_size")) is not int or field["unit_chunk_size"] < 1:
            raise ValueError("POPULATION_FIELD_POSITIVE_CHUNK_REQUIRED")
    cohort = value["cohort"]
    for name in ("expected_metadata_session_count", "expected_protocol_session_count"):
        if type(cohort.get(name)) is not int or cohort[name] <= 0:
            raise ValueError(f"METADATA_COHORT_COUNT_REQUIRED:{name}")
    counts = cohort.get("split_counts_by_donor")
    if (not isinstance(counts, dict) or set(counts) != {"train", "validation", "test"}
            or any(type(count) is not int or count < 0 for count in counts.values())
            or counts["train"] < 1 or cohort.get("split_protocol") != "donor_quota_v1"):
        raise ValueError("EXPLICIT_DONOR_SPLIT_QUOTAS_REQUIRED")
    if cohort.get("exploration_train_session_ids") != [847657808]:
        raise ValueError("EXPLORATION_ANCHOR_MUST_BE_PREDECLARED")
    if not isinstance(cohort.get("split_salt"), str) or not cohort["split_salt"]:
        raise ValueError("COHORT_SPLIT_SALT_REQUIRED")
    for name in ("connect_timeout_seconds", "read_timeout_seconds"):
        if type(value["network"].get(name)) not in (int, float) or not math.isfinite(value["network"][name]) or value["network"][name] <= 0:
            raise ValueError(f"NETWORK_TIMEOUT_INVALID:{name}")
    if type(value["network"].get("trust_env")) is not bool:
        raise ValueError("NETWORK_TRUST_ENV_BOOLEAN_REQUIRED")
    qc = value["unit_qc"]
    if qc.get("threshold_comparison") != "strict":
        raise ValueError("QC_COMPARISON_MUST_BE_EXPLICIT_STRICT")
    if not isinstance(qc.get("accepted_quality"), list) or not qc["accepted_quality"]:
        raise ValueError("UNIT_QUALITY_POLICY_REQUIRED")
    for key in ("presence_ratio_minimum", "amplitude_cutoff_maximum", "isi_violations_maximum"):
        if type(qc.get(key)) not in (int, float) or not math.isfinite(qc[key]) or qc[key] < 0:
            raise ValueError(f"UNIT_QC_THRESHOLD_INVALID:{key}")
    required_criteria = {"minimum_eligible_units", "minimum_valid_repeats_per_condition",
                         "required_history_seconds", "required_horizon_seconds"}
    if set(value["eligibility"]) != required_criteria:
        raise ValueError("ELIGIBILITY_CRITERIA_MISSING_OR_UNKNOWN")
    if not isinstance(value.get("movie_frames_per_repeat"), dict) or any(
        type(count) is not int or count <= 0 for count in value["movie_frames_per_repeat"].values()
    ):
        raise ValueError("MOVIE_FRAME_COUNT_INVALID")
    for key, number in value["eligibility"].items():
        if number is not None and (type(number) not in (int, float) or not math.isfinite(number) or number <= 0):
            raise ValueError(f"ELIGIBILITY_POSITIVE_OR_UNDEFINED:{key}")
    return value


def project_path(project: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project / candidate

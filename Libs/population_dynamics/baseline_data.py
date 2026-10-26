"""Shared, source-checked engineering windows for neural population baselines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .provenance import sha256_file


@dataclass(frozen=True)
class BaselineWindows:
    counts: np.ndarray
    inputs: np.ndarray
    unit_ids: tuple[int, ...]
    history_bins: int
    bin_seconds: float
    trial_ids: tuple[str, ...]
    input_features: dict[str, Any] | None


def load_engineering_windows(source_dir: Path, config: dict[str, Any]) -> BaselineWindows:
    """Read the existing training-anchor fixture without qualifying formal data."""
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") == "opto_complete_control_windows_v1":
        return _load_opto_windows(source_dir, config, manifest)
    for name in ("summary.json", "spike_windows.h5"):
        if sha256_file(source_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"SOURCE_WINDOW_DIGEST_MISMATCH:{name}")
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if (source["scope"] != "engineering_only" or source["split"] != "train"
            or source["session_id"] not in config["cohort"]["exploration_train_session_ids"]):
        raise ValueError("COUNT_CHECK_REQUIRES_ENGINEERING_TRAINING_ANCHOR")
    with h5py.File(source_dir / "spike_windows.h5", "r") as recording:
        counts = recording["counts"][:]
        ids = tuple(recording["unit_ids"][:].tolist())
        history = recording.attrs["history_bin_count"]
        edges = recording["relative_edges_seconds"][:]
        trials = tuple(recording["trial_ids"].asstr()[:])
        inputs = recording["stimulus_features"][:] if "stimulus_features" in recording else None
    if (counts.ndim != 3 or min(counts.shape) < 1 or not np.isrealobj(counts)
            or not np.isfinite(counts).all() or (counts < 0).any()
            or not np.equal(counts, np.round(counts)).all()):
        raise ValueError("SOURCE_FINITE_NONNEGATIVE_COUNT_ARRAY_REQUIRED")
    if (len(ids) != counts.shape[2] or any(type(unit) is not int or unit < 0 for unit in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("SOURCE_UNIT_IDENTITY_OR_SHAPE_MISMATCH")
    if len(trials) != counts.shape[0] or len(set(trials)) != len(trials):
        raise ValueError("SOURCE_TRIAL_IDENTITY_OR_SHAPE_MISMATCH")
    if (isinstance(history, (bool, np.bool_)) or not isinstance(history, (int, np.integer))
            or not 0 < history < counts.shape[1]):
        raise ValueError("SOURCE_HISTORY_OR_RESPONSE_TOO_SHORT")
    bin_seconds = manifest["window_specification"]["bin_seconds"]
    if (type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0
            or edges.shape != (counts.shape[1] + 1,) or not np.isfinite(edges).all()
            or not np.allclose(np.diff(edges), bin_seconds, atol=1e-12, rtol=0)):
        raise ValueError("SOURCE_BIN_WIDTH_MISMATCH")
    feature_spec = None
    if inputs is not None:
        if sha256_file(source_dir / "stimulus_features.json") != manifest["artifacts"]["stimulus_features.json"]:
            raise ValueError("SOURCE_STIMULUS_FEATURE_METADATA_MISMATCH")
        feature_spec = json.loads((source_dir / "stimulus_features.json").read_text(encoding="utf-8"))
        if (inputs.ndim != 3 or inputs.shape[:2] != counts.shape[:2] or not np.isrealobj(inputs)
                or not np.isfinite(inputs).all()
                or [f["feature_index"] for f in feature_spec["features"]] != list(range(inputs.shape[2]))):
            raise ValueError("SOURCE_STIMULUS_FEATURE_ORDER_OR_SHAPE_MISMATCH")
    else:
        inputs = np.empty((*counts.shape[:2], 0), dtype=np.float64)
    return BaselineWindows(counts, inputs, ids, int(history), float(bin_seconds), trials, feature_spec)


def _load_opto_windows(source_dir: Path, config: dict[str, Any],
                       manifest: dict[str, Any]) -> BaselineWindows:
    """Load the fixed-vocabulary Opto format for engineering consumers."""
    for name in ("summary.json", "windows.npz", "stimulus_features.json"):
        if sha256_file(source_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"SOURCE_WINDOW_DIGEST_MISMATCH:{name}")
    source = json.loads((source_dir / "summary.json").read_text())
    if (manifest.get("subject_id") != config["opto_source"]["engineering_anchor_subject_id"]
            or source.get("subject_id") != manifest["subject_id"]
            or source.get("dataset_ready") is not False):
        raise ValueError("OPTO_CHECK_REQUIRES_ENGINEERING_TRAINING_ANCHOR")
    with np.load(source_dir / "windows.npz", allow_pickle=False) as arrays:
        counts, inputs = arrays["counts"], arrays["stimulus_features"]
        ids = arrays["unit_ids"]
        history = arrays["history_bin_count"]
        edges = arrays["relative_edges_seconds"]
    if (counts.ndim != 3 or min(counts.shape) < 1 or counts.dtype.kind not in "iu"
            or (counts < 0).any()):
        raise ValueError("SOURCE_FINITE_NONNEGATIVE_COUNT_ARRAY_REQUIRED")
    if (ids.shape != (counts.shape[2],) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("SOURCE_UNIT_IDENTITY_OR_SHAPE_MISMATCH")
    if history.shape != () or history.dtype.kind not in "iu" or not 0 < int(history) < counts.shape[1]:
        raise ValueError("SOURCE_HISTORY_OR_RESPONSE_TOO_SHORT")
    trials = source["selected_trial_ids"]
    if (len(trials) != counts.shape[0] or any(type(x) is not int for x in trials)
            or len(set(trials)) != len(trials)):
        raise ValueError("SOURCE_TRIAL_IDENTITY_OR_SHAPE_MISMATCH")
    bin_seconds = manifest["resolved_config"]["opto_window_check"]["bin_seconds"]
    if (type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0
            or edges.shape != (counts.shape[1] + 1,) or not np.isfinite(edges).all()
            or not np.allclose(np.diff(edges), bin_seconds, rtol=0, atol=1e-12)
            or edges[int(history)] != 0):
        raise ValueError("SOURCE_BIN_WIDTH_MISMATCH")
    features = json.loads((source_dir / "stimulus_features.json").read_text())
    if (inputs.shape[:2] != counts.shape[:2] or inputs.ndim != 3
            or inputs.dtype.kind not in "fiu" or not np.isfinite(inputs).all()
            or [f["feature_index"] for f in features] != list(range(inputs.shape[2]))):
        raise ValueError("SOURCE_STIMULUS_FEATURE_ORDER_OR_SHAPE_MISMATCH")
    return BaselineWindows(counts, inputs, tuple(map(int, ids)), int(history), float(bin_seconds),
                           tuple(map(str, trials)), {"features": features,
                           "vocabulary_scope": "fixed_training_anchor_control_catalog"})

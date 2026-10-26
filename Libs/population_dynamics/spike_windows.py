"""Exact count windows and a bounded real-input replay for the intake CLI."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .provenance import freeze_json, runtime_manifest, sha256_file
from .qualification import _intervals, _invalid_overlap
from .stimulus_windows import encode_stimulus_windows


@dataclass(frozen=True)
class SpikeWindows:
    counts: np.ndarray  # trial, time bin, unit; counts, not Hz
    unit_ids: np.ndarray
    relative_edges_seconds: np.ndarray
    absolute_edges_seconds: np.ndarray
    history_bin_count: int

    @property
    def history(self) -> np.ndarray:
        return self.counts[:, :self.history_bin_count, :]

    @property
    def response(self) -> np.ndarray:
        return self.counts[:, self.history_bin_count:, :]


def window_edges(history_seconds: float, horizon_seconds: float,
                 bin_seconds: float) -> tuple[np.ndarray, int]:
    values = (history_seconds, horizon_seconds, bin_seconds)
    if any(isinstance(v, bool) or not np.isscalar(v) or not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("WINDOW_DURATIONS_MUST_BE_FINITE_POSITIVE")
    bins = np.asarray(values[:2], dtype=float) / bin_seconds
    rounded = np.rint(bins).astype(np.int64)
    if np.any(rounded < 1) or not np.allclose(bins, rounded, rtol=0, atol=1e-9):
        raise ValueError("WINDOW_DURATIONS_MUST_BE_INTEGER_BINS")
    n_history, n_future = map(int, rounded)
    # Construct zero exactly, so onset belongs to response and never to history.
    edges = np.arange(-n_history, n_future + 1, dtype=float) * bin_seconds
    return edges, n_history


def bin_spikes(spike_times: Mapping[int, np.ndarray], unit_ids: Sequence[int],
               onsets_seconds: np.ndarray, *, history_seconds: float,
               horizon_seconds: float, bin_seconds: float) -> SpikeWindows:
    """Count in [left, right) bins; retain silent units and caller unit order.

    This numerical operation does not infer recording coverage from spike silence.
    The caller must establish the observation support of any scientific window.
    """
    onsets = np.asarray(onsets_seconds, dtype=float)
    if onsets.ndim != 1 or not np.isfinite(onsets).all():
        raise ValueError("FINITE_ONSET_VECTOR_REQUIRED")
    edges, n_history = window_edges(history_seconds, horizon_seconds, bin_seconds)
    absolute = onsets[:, None] + edges[None, :]
    counts = count_spikes_at_edges(spike_times, unit_ids, absolute)
    return SpikeWindows(counts, np.asarray(unit_ids).copy(), edges, absolute, n_history)


def count_spikes_at_edges(spike_times: Mapping[int, np.ndarray], unit_ids: Sequence[int],
                         absolute_edges_seconds: np.ndarray) -> np.ndarray:
    """Count exact half-open bins on supplied clocks; never infer recording support.

    Native simulators can supply edges aligned to their own clock instead of
    recomputing absolute endpoints by floating-point onset addition.
    """
    ids = np.asarray(unit_ids)
    if ids.ndim != 1 or len(ids) == 0 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("UNIQUE_INTEGER_UNIT_IDS_REQUIRED")
    absolute = np.asarray(absolute_edges_seconds, dtype=float)
    if absolute.ndim != 2 or absolute.shape[1] < 2:
        raise ValueError("ABSOLUTE_BIN_EDGE_MATRIX_REQUIRED")
    if not np.isfinite(absolute).all() or np.any(np.diff(absolute, axis=1) <= 0):
        raise ValueError("ABSOLUTE_TIME_PRECISION_CANNOT_RESOLVE_BINS")
    counts = np.empty((absolute.shape[0], absolute.shape[1] - 1, len(ids)), dtype=np.int64)
    for index, unit in enumerate(ids):
        spikes = np.asarray(spike_times[int(unit)])
        if spikes.ndim != 1 or spikes.dtype.kind not in "fiu" or not np.isfinite(spikes).all() or np.any(spikes[1:] < spikes[:-1]):
            raise ValueError(f"FINITE_SORTED_SPIKE_VECTOR_REQUIRED:{unit}")
        positions = np.searchsorted(spikes, absolute, side="left")
        counts[:, :, index] = np.diff(positions, axis=1)
    return counts


def check_recording_windows(source_dir: Path, output_dir: Path, config: dict[str, Any],
                            command: list[str]) -> tuple[dict[str, Any], int]:
    """Replay a training anchor's numerical input path, never qualify science."""
    if output_dir.exists():
        raise FileExistsError(f"WINDOW_CHECK_OUTPUT_EXISTS:{output_dir}")
    spec = config["window_check"]
    edges, _ = window_edges(spec["history_seconds"], spec["horizon_seconds"], spec["bin_seconds"])
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(source_dir / "summary.json") != manifest["summary_sha256"]:
        raise ValueError("SOURCE_SUMMARY_DIGEST_MISMATCH")
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if source["split"] != "train" or source["session_id"] not in config["cohort"]["exploration_train_session_ids"]:
        raise ValueError("WINDOW_CHECK_REQUIRES_TRAINING_ANCHOR")
    record_path = source_dir / manifest["selected_recording"]["path"]
    table_names = ["trials.csv", "invalid_times.csv"]
    if spec.get("encode_stimulus", False):
        table_names += ["stimulus_presentations.csv", "trial_members.csv"]
    for name in table_names:
        if sha256_file(source_dir / name) != manifest["artifacts"][name]:
            raise ValueError(f"SOURCE_TABLE_DIGEST_MISMATCH:{name}")
    if sha256_file(record_path) != manifest["selected_recording"]["sha256"]:
        raise ValueError("SOURCE_RECORDING_DIGEST_MISMATCH")
    all_trials = pd.read_csv(source_dir / "trials.csv")
    trials = all_trials.loc[all_trials.stimulus_name == spec["stimulus_name"]].copy()
    if trials.empty:
        raise ValueError("REQUESTED_STIMULUS_HAS_NO_TRIALS")
    if trials.trial_id.duplicated().any() or not (trials.session_id == source["session_id"]).all() or not (trials.split == "train").all():
        raise ValueError("TRIAL_IDENTITY_OR_SPLIT_MISMATCH")
    invalid = _intervals(pd.read_csv(source_dir / "invalid_times.csv"), "invalid_times")
    trials["window_start_time"] = trials.start_time + edges[0]
    trials["window_stop_time"] = trials.start_time + edges[-1]
    reasons = []
    for row in trials.itertuples():
        reason = []
        if not row.structurally_eligible:
            reason.append("source_trial_structurally_excluded")
        if row.window_stop_time > row.stop_time:
            reason.append("response_exceeds_trial")
        if _invalid_overlap(row.window_start_time, row.window_stop_time, invalid):
            reason.append("requested_window_overlaps_invalid_time")
        reasons.append(";".join(reason))
    trials["window_exclusion_reasons"] = reasons
    trials["engineering_window_usable"] = trials.window_exclusion_reasons == ""
    trials["scientific_window_eligible"] = False
    selected = trials.loc[trials.engineering_window_usable]
    if selected.empty:
        raise ValueError("NO_ENGINEERING_WINDOWS_REMAIN")
    with h5py.File(record_path, "r") as recording:
        if recording.attrs["spike_time_unit"] != "seconds" or recording.attrs["session_id"] != source["session_id"]:
            raise ValueError("RECORDING_IDENTITY_OR_TIME_UNIT_MISMATCH")
        ids = recording["unit_ids"][:]
        offsets = recording["spike_offsets"][:]
        if (len(offsets) != len(ids) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                or offsets[-1] != len(recording["spike_times_s"])):
            raise ValueError("RECORDING_SPIKE_OFFSETS_INVALID")
        spikes = {int(u): recording["spike_times_s"][offsets[i]:offsets[i + 1]] for i, u in enumerate(ids)}
    windows = bin_spikes(spikes, ids, selected.start_time.to_numpy(),
                         history_seconds=spec["history_seconds"], horizon_seconds=spec["horizon_seconds"],
                         bin_seconds=spec["bin_seconds"])
    encoded = None
    if spec.get("encode_stimulus", False):
        presentations = pd.read_csv(source_dir / "stimulus_presentations.csv",
                                   usecols=["presentation_id", "start_time", "stop_time", "stimulus_name", "frame"])
        members = pd.read_csv(source_dir / "trial_members.csv")
        encoded = encode_stimulus_windows(presentations, members, all_trials, windows.absolute_edges_seconds)
    # Independent boolean interval counts check history/response separately. This
    # detects onset/right-edge mistakes without duplicating searchsorted binning.
    for i, onset in enumerate(selected.start_time):
        for j, unit in enumerate(ids):
            times = spikes[int(unit)]
            for a, b, counts in ((onset + edges[0], onset, windows.history),
                                 (onset, onset + edges[-1], windows.response)):
                if np.count_nonzero((times >= a) & (times < b)) != counts[i, :, j].sum():
                    raise ValueError("DIRECT_INTERVAL_COUNT_REPLAY_MISMATCH")
    output_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(output_dir / "spike_windows.h5", "x") as output:
        output.attrs.update(schema="neural_population_spike_window_check_v1", scope="engineering_only",
                            session_id=source["session_id"], donor_id=source["animal_id"],
                            count_unit="spikes_per_bin", bin_convention="[left,right)",
                            history_bin_count=windows.history_bin_count, dataset_ready=False,
                            source_coverage_verified=source["recording_coverage_source_verified"])
        for name, values in (("counts", windows.counts), ("unit_ids", ids),
                             ("relative_edges_seconds", windows.relative_edges_seconds),
                             ("absolute_edges_seconds", windows.absolute_edges_seconds)):
            output.create_dataset(name, data=values)
        output.create_dataset("trial_ids", data=selected.trial_id.to_list(), dtype=h5py.string_dtype("utf-8"))
        if encoded is not None:
            output.create_dataset("stimulus_features", data=encoded.values)
            output.create_dataset("stimulus_overlap_fraction", data=encoded.overlap_fraction)
    with h5py.File(output_dir / "spike_windows.h5", "r") as output:
        if not np.array_equal(output["counts"][:], windows.counts) or list(output["trial_ids"].asstr()[:]) != selected.trial_id.to_list():
            raise ValueError("PERSISTED_WINDOW_REPLAY_MISMATCH")
        if encoded is not None and (not np.array_equal(output["stimulus_features"][:], encoded.values)
                                    or not np.array_equal(output["stimulus_overlap_fraction"][:], encoded.overlap_fraction)):
            raise ValueError("PERSISTED_STIMULUS_WINDOW_REPLAY_MISMATCH")
    trials.to_csv(output_dir / "trials.csv", index=False)
    artifacts = ["spike_windows.h5", "trials.csv", "summary.json"]
    if encoded is not None:
        freeze_json(output_dir / "stimulus_features.json", {
            "features": encoded.features, "unit": "fraction_of_bin_duration",
            "vocabulary_scope": "source_annotations_overlapping_requested_training_anchor_windows",
            "semantics": "recorded condition and played movie-frame occupancy; unannotated is not blank",
            "arbitrary_continuous_intervention_space": False})
        artifacts.append("stimulus_features.json")
    summary = {"status": "PARTIAL", "numerical_window_replay": "PASS", "scope": "engineering_only",
               "session_id": source["session_id"], "split": source["split"], "unit_count": len(ids),
               "raw_unit_count": source["raw_unit_count"], "qc_eligible_unit_count": source["qc_eligible_unit_count"],
               "candidate_trials": len(trials), "counted_trials": len(selected), "excluded_trials": len(trials) - len(selected),
               "history_spike_occurrences": int(windows.history.sum()),
               "response_spike_occurrences": int(windows.response.sum()),
               "occurrence_count_note": "Overlapping windows may reuse spikes; these are not independent replicates.",
               "state_covariates_included": [], "prior_stimulus_assumed_blank": False,
               "known_stimulus_feature_count": encoded.values.shape[2] if encoded is not None else 0,
               "stimulus_unannotated_fraction_max": float(encoded.values[:, :, 0].max()) if encoded is not None else None,
               "stimulus_overlap_fraction_max": float(encoded.overlap_fraction.max()) if encoded is not None else None,
               "recording_coverage_source_verified": source["recording_coverage_source_verified"],
               "dataset_ready": False, "model_ready": False, "scientific_claims_qualified": False}
    freeze_json(output_dir / "summary.json", summary)
    receipt = {"schema": "neural_population_window_check_manifest_v1", "status": "PARTIAL",
               "source_manifest": str((source_dir / "manifest.json").resolve()),
               "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
               "source_recording_sha256": manifest["selected_recording"]["sha256"],
               "window_specification": spec, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
               "artifacts": {name: sha256_file(output_dir / name) for name in artifacts},
               **runtime_manifest(config, command, 2)}
    freeze_json(output_dir / "manifest.json", receipt)
    return {**summary, "manifest": str(output_dir / "manifest.json")}, 2

"""Local NWB quality accounting and trial reconstruction without model imports."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class SessionData:
    session_id: int
    subject_id: str | None
    units: pd.DataFrame
    spike_times: Mapping[int, np.ndarray]
    presentations: pd.DataFrame
    invalid_times: pd.DataFrame
    optotagging: pd.DataFrame
    spike_time_unit: str = "seconds"
    recording_intervals: np.ndarray | None = None
    source_full_asset_downloaded: bool = False
    required_columns_present: bool = False
    source_etag: str | None = None
    source_version_check: Callable[[], None] | None = None
    column_evidence: dict[str, Any] | None = None
    covariates: dict[str, pd.DataFrame] | None = None
    electrodes: pd.DataFrame | None = None


class NwbSpikeTimes(Mapping):
    """Read one unit at a time while the parent NWB context remains open."""
    def __init__(self, dataset, unit_ids: np.ndarray, ends: np.ndarray, *, retain_values: bool = False):
        if dataset.ndim != 1 or len(ends) != len(unit_ids) or ends.ndim != 1:
            raise ValueError("NWB_SPIKE_INDEX_SHAPE_INVALID")
        if not np.issubdtype(ends.dtype, np.integer) or np.any(ends < 0) or np.any(ends[1:] < ends[:-1]):
            raise ValueError("NWB_SPIKE_INDEX_VALUES_INVALID")
        if not len(ends) or int(ends[-1]) != dataset.shape[0]:
            raise ValueError("NWB_SPIKE_INDEX_DENOMINATOR_MISMATCH")
        self.dataset = dataset
        self.ends = ends
        self.positions = {int(unit_id): index for index, unit_id in enumerate(unit_ids)}
        self.consumed_units = {}
        self.retain_values = retain_values
        self.selected_values = {}
        if len(self.positions) != len(unit_ids):
            raise ValueError("DUPLICATE_NWB_UNIT_ID")

    def __len__(self):
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def __getitem__(self, unit_id):
        if unit_id in self.selected_values:
            return self.selected_values[unit_id]
        index = self.positions[unit_id]
        start = int(self.ends[index - 1]) if index else 0
        values = np.asarray(self.dataset[start:int(self.ends[index])])
        self.consumed_units[str(unit_id)] = {"start_index": start, "stop_index": int(self.ends[index]),
            "shape": list(values.shape), "dtype": str(values.dtype), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}
        if self.retain_values:
            self.selected_values[unit_id] = values
        return values

    def count(self, unit_id):
        index = self.positions[unit_id]
        return int(self.ends[index]) - (int(self.ends[index-1]) if index else 0)


def _h5_value(value, handle):
    import h5py
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, h5py.Reference):
        return handle[value].name if value else None
    if isinstance(value, np.void) and value.dtype.names:
        return {name: _h5_value(value[name], handle) for name in value.dtype.names}
    if isinstance(value, (np.ndarray, tuple, list)):
        return [_h5_value(item, handle) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_column(dataset, handle, evidence):
    raw = dataset[()]
    value = _h5_value(raw, handle)
    digest = hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, allow_nan=False).encode()).hexdigest()
    evidence[dataset.name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype), "sha256": digest,
        "source_attributes": {name: _h5_value(dataset.attrs[name], handle)
                              for name in ("unit", "conversion", "offset", "description") if name in dataset.attrs}}
    return value


def _require_seconds(dataset, handle):
    # NWB interval coordinates and TimeSeries timestamps are seconds by schema.
    declared = _h5_value(dataset.attrs.get("unit", "seconds"), handle)
    if declared not in {"s", "seconds"}:
        raise ValueError(f"TIME_UNIT_MUST_BE_SECONDS:{dataset.name}:{declared}")


def _read_interval_table(group, handle, evidence):
    if "id" not in group:
        raise ValueError(f"NWB_INTERVAL_IDS_MISSING:{group.name}")
    ids = _read_column(group["id"], handle, evidence)
    if len(set(ids)) != len(ids):
        raise ValueError(f"NWB_INTERVAL_IDS_DUPLICATE:{group.name}")
    names = _h5_value(group.attrs.get("colnames", []), handle)
    if not names:
        raise ValueError(f"NWB_INTERVAL_COLUMNS_MISSING:{group.name}")
    columns = {}
    for name in names:
        if name not in group:
            raise ValueError(f"NWB_INTERVAL_COLUMN_MISSING:{group.name}:{name}")
        if name in {"start_time", "stop_time"}:
            _require_seconds(group[name], handle)
        values = _read_column(group[name], handle, evidence)
        if f"{name}_index" in group:
            ends = np.asarray(_read_column(group[f"{name}_index"], handle, evidence))
            if (ends.shape != (len(ids),) or not np.issubdtype(ends.dtype, np.integer)
                    or np.any(ends < 0) or np.any(ends[1:] < ends[:-1])
                    or (len(ends) and ends[-1] != len(values))):
                raise ValueError(f"NWB_VECTOR_INDEX_INVALID:{group.name}:{name}")
            starts = np.r_[0, ends[:-1]]
            values = [values[int(start):int(stop)] for start, stop in zip(starts, ends)]
        if len(values) != len(ids):
            raise ValueError(f"NWB_INTERVAL_COLUMN_LENGTH_MISMATCH:{group.name}:{name}")
        columns[name] = values
    return pd.DataFrame(columns, index=ids)


def _read_behavioral_covariates(handle, evidence):
    covariates = {}
    running = "processing/running/running_speed"
    running_end = "processing/running/running_speed_end_times/timestamps"
    pupil = "processing/filtered_gaze_mapping/pupil_area"
    for name, path, expected_unit in (("running_speed", running, "cm/s"), ("filtered_pupil_area", pupil, "Pixels ^ 2")):
        if f"{path}/data" not in handle or f"{path}/timestamps" not in handle:
            continue
        unit = _h5_value(handle[f"{path}/data"].attrs.get("unit"), handle)
        conversion = float(handle[f"{path}/data"].attrs.get("conversion", 1.0))
        if unit != expected_unit or conversion != 1.0:
            raise ValueError(f"STATE_SIGNAL_UNIT_OR_CONVERSION_UNSUPPORTED:{name}")
        offset = float(handle[f"{path}/data"].attrs.get("offset", 0.0))
        if not np.isfinite(offset) or offset != 0.0:
            raise ValueError(f"STATE_SIGNAL_OFFSET_UNSUPPORTED:{name}")
        values = np.asarray(_read_column(handle[f"{path}/data"], handle, evidence), dtype=float)
        _require_seconds(handle[f"{path}/timestamps"], handle)
        times = np.asarray(_read_column(handle[f"{path}/timestamps"], handle, evidence), dtype=float)
        if values.ndim != 1 or times.shape != values.shape or not np.isfinite(times).all() or np.any(times[1:] < times[:-1]):
            raise ValueError(f"STATE_SIGNAL_SHAPE_OR_TIMESTAMPS_INVALID:{name}")
        frame = pd.DataFrame({"start_time": times, "value": values, "value_is_finite": np.isfinite(values)})
        if name == "running_speed":
            if running_end not in handle:
                raise ValueError("RUNNING_INTERVAL_END_TIMESTAMPS_MISSING")
            _require_seconds(handle[running_end], handle)
            ends = np.asarray(_read_column(handle[running_end], handle, evidence), dtype=float)
            if ends.shape != times.shape or not np.isfinite(ends).all() or np.any(ends < times):
                raise ValueError("RUNNING_INTERVAL_END_TIMESTAMPS_INVALID")
            frame["stop_time"] = ends
        frame.attrs.update({"unit": unit, "conversion": conversion, "offset": offset,
                            "source_path": path, "causal_use_qualified": False,
                            "state_window_rule": "stop_time <= stimulus_start" if name == "running_speed" else "filter causality unverified"})
        covariates[name] = frame
    return covariates


def _read_electrodes(handle, evidence):
    root = "general/extracellular_ephys/electrodes"
    if f"{root}/id" not in handle:
        return pd.DataFrame()
    ids = _read_column(handle[f"{root}/id"], handle, evidence)
    if len(set(ids)) != len(ids):
        raise ValueError("ELECTRODE_IDS_DUPLICATE")
    columns = {"electrode_id":ids}
    for name in ("location", "probe_id", "probe_vertical_position", "probe_horizontal_position"):
        if f"{root}/{name}" in handle:
            values = _read_column(handle[f"{root}/{name}"], handle, evidence)
            if len(values) != len(ids):
                raise ValueError(f"ELECTRODE_COLUMN_LENGTH_MISMATCH:{name}")
            columns[name] = values
    return pd.DataFrame(columns)


def read_nwb_handle(handle, expected_session_id: int, *, source_etag: str | None = None,
                    source_full_asset_downloaded: bool = False,
                    etag_reader: Callable[[], str] | None = None, retain_spikes: bool = False,
                    progress: Callable[[dict[str, Any]], None] | None = None) -> SessionData:
    """Read the verified Allen schema from an open local or cloud-backed HDF5 file.

    The caller keeps the handle open through qualify_session. Cloud callers pass
    an ETag reader; it is checked before metadata and before/after spike reads.
    No waveform, amplitude or discarded-unit spike values are materialized.
    """
    evidence = {}
    def check_version():
        if not source_etag or etag_reader is None or etag_reader() != source_etag:
            raise ValueError("NWB_SOURCE_ETAG_MISMATCH_OR_UNVERIFIED")
    if etag_reader is not None:
        check_version()
    if "nwb_version" not in handle.attrs or "units/spike_times" not in handle:
        raise ValueError("NWB_SPIKE_UNITS_REQUIRED")
    identity = str(_read_column(handle["identifier"], handle, evidence))
    if identity != str(expected_session_id):
        raise ValueError(f"NWB_SESSION_IDENTITY_MISMATCH:{identity}")
    subject = str(_read_column(handle["general/subject/subject_id"], handle, evidence))
    unit = _h5_value(handle["units/spike_times"].attrs.get("unit"), handle)
    if unit is None:
        description = str(_h5_value(handle["units/spike_times"].attrs.get("description", ""), handle))
        if "(s)" not in description and "seconds" not in description:
            raise ValueError("SPIKE_TIME_UNIT_DECLARATION_UNVERIFIED")
        unit = "seconds"
    unit_ids = np.asarray(_read_column(handle["units/id"], handle, evidence))
    if unit_ids.ndim != 1 or not np.issubdtype(unit_ids.dtype, np.integer) or np.any(unit_ids < 0):
        raise ValueError("UNIT_IDENTITY_INVALID")
    columns = {}
    qc_columns = ("quality", "presence_ratio", "amplitude_cutoff", "isi_violations")
    for name in (*qc_columns, "peak_channel_id"):
        if f"units/{name}" in handle:
            dataset = handle[f"units/{name}"]
            if dataset.shape != unit_ids.shape:
                raise ValueError(f"UNIT_METADATA_SHAPE_INVALID:{name}")
            columns[name] = _read_column(dataset, handle, evidence)
    units = pd.DataFrame(columns, index=unit_ids)
    ends = np.asarray(_read_column(handle["units/spike_times_index"], handle, evidence))
    spikes = NwbSpikeTimes(handle["units/spike_times"], unit_ids, ends, retain_values=retain_spikes)
    if progress:
        progress({"event":"UNIT_METADATA_READY", "raw_units":len(unit_ids), "raw_spikes":int(ends[-1])})
    tables = []
    for name in handle["intervals"]:
        if not name.endswith("_presentations"):
            continue
        table = _read_interval_table(handle[f"intervals/{name}"], handle, evidence)
        table["source_table"] = name
        table["source_presentation_id"] = table.index
        table.index = pd.Index([f"{name}:{index}" for index in table.index], name="presentation_id")
        if "stimulus_name" not in table:
            table["stimulus_name"] = name.removesuffix("_presentations")
        tables.append(table)
        if progress:
            progress({"event":"STIMULUS_TABLE_READY", "table":name, "presentations":len(table)})
    if not tables:
        raise ValueError("NWB_STIMULUS_PRESENTATIONS_MISSING")
    presentations = pd.concat(tables).sort_values("start_time", kind="stable")
    invalid = (_read_interval_table(handle["intervals/invalid_times"], handle, evidence)
               if "intervals/invalid_times" in handle else pd.DataFrame())
    opto_path = "processing/optotagging/optogenetic_stimulation"
    optotagging = _read_interval_table(handle[opto_path], handle, evidence) if opto_path in handle else pd.DataFrame()
    covariates = _read_behavioral_covariates(handle, evidence)
    electrodes = _read_electrodes(handle, evidence)
    return SessionData(expected_session_id, subject, units, spikes, presentations, invalid,
        optotagging, str(unit), None, source_full_asset_downloaded,
        all(name in units for name in qc_columns) and len(covariates) == 2, source_etag,
        check_version if etag_reader is not None else None, evidence, covariates, electrodes)


@contextmanager
def read_local_nwb(path: Path, expected_session_id: int, *, source_full_asset_downloaded: bool = False, retain_spikes: bool = False, progress=None):
    import h5py
    with h5py.File(path, "r") as handle:
        yield read_nwb_handle(handle, expected_session_id, source_full_asset_downloaded=source_full_asset_downloaded, retain_spikes=retain_spikes, progress=progress)


def _intervals(frame: pd.DataFrame, label: str) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2))
    if not {"start_time", "stop_time"}.issubset(frame):
        raise ValueError(f"INTERVAL_COLUMNS_REQUIRED:{label}")
    values = frame[["start_time", "stop_time"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError(f"INVALID_TIME_INTERVAL:{label}")
    return values


def _invalid_overlap(start: float, stop: float, invalid: np.ndarray) -> bool:
    return bool(len(invalid) and np.any((invalid[:, 0] < stop) & (invalid[:, 1] > start)))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "object_id"):
        return {"nwb_object_id": str(value.object_id), "name": str(value.name)}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def condition_parameters(row: pd.Series, movie: bool) -> dict[str, Any]:
    # Retain all source-declared stimulus parameters, including less common dot
    # direction/speed fields; timestamps, row identities and NWB references are
    # not independent experimental conditions.
    excluded = {"start_time", "stop_time", "duration", "stimulus_block", "tags", "timeseries",
                "source_table", "source_presentation_id", "stimulus_condition_id", "stimulus_index"}
    if movie:
        excluded.add("frame")
    fields = {key: _json_value(value) for key, value in row.items() if key not in excluded}
    fields = {key: value for key, value in fields.items() if value is not None}
    return fields


def _condition(row: pd.Series, movie: bool) -> str:
    fields = condition_parameters(row, movie)
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def reconstruct_trials(presentations: pd.DataFrame, invalid_times: pd.DataFrame,
                       session_id: int, animal_id: int, split: str,
                       expected_movie_frames: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if presentations.empty:
        raise ValueError("STIMULUS_PRESENTATIONS_EMPTY")
    if not presentations.index.is_unique:
        raise ValueError("DUPLICATE_PRESENTATION_ID")
    if "stimulus_name" not in presentations:
        raise ValueError("STIMULUS_NAME_REQUIRED")
    _intervals(presentations, "presentations")
    invalid = _intervals(invalid_times, "invalid_times")
    ordered = presentations.sort_values("start_time", kind="stable")
    trials: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    movie_rows: list[tuple[Any, pd.Series]] = []
    movie_group: tuple[Any, Any] | None = None

    def emit(rows: list[tuple[Any, pd.Series]], movie: bool) -> None:
        first_id, first = rows[0]
        name = str(first["stimulus_name"])
        start, stop = float(first["start_time"]), float(rows[-1][1]["stop_time"])
        reasons = []
        expected = expected_movie_frames.get(name) if movie else 1
        if movie:
            frames = np.asarray([row.get("frame", np.nan) for _, row in rows], dtype=float)
            if not np.isfinite(frames).all() or np.any(frames != np.floor(frames)) or np.any(frames < 0):
                raise ValueError(f"MOVIE_FRAME_INDEX_INVALID:{name}")
            if expected is None:
                reasons.append("movie_repeat_length_not_declared")
            elif len(frames) != expected or not np.array_equal(frames, np.arange(expected)):
                reasons.append("incomplete_or_nonsequential_movie_repeat")
            starts = np.asarray([row["start_time"] for _, row in rows], dtype=float)
            stops = np.asarray([row["stop_time"] for _, row in rows], dtype=float)
            if len(rows) > 1 and not np.allclose(starts[1:], stops[:-1], atol=1e-6, rtol=0):
                reasons.append("movie_frame_time_gap_or_overlap")
        if _invalid_overlap(start, stop, invalid):
            reasons.append("invalid_time_overlap_any_scope")
        trial_id = f"{session_id}:{first_id}"
        trials.append({
            "trial_id": trial_id, "session_id": session_id, "animal_id": animal_id,
            "split": split, "stimulus_name": name, "condition_id": _condition(first, movie),
            "start_time": start, "stop_time": stop, "duration_seconds": stop-start,
            "trial_kind": "movie_repeat" if movie else "presentation",
            "presentation_count": len(rows), "expected_presentation_count": expected,
            "structurally_eligible": not reasons, "exclusion_reasons": ";".join(reasons),
        })
        members.extend({"presentation_id": item_id, "trial_id": trial_id, "session_id": session_id,
                        "animal_id": animal_id, "split": split} for item_id, _ in rows)

    for presentation_id, row in ordered.iterrows():
        name = str(row["stimulus_name"])
        movie = name.startswith("natural_movie")
        block = _json_value(row.get("stimulus_block", None))
        group = (name, block)
        frame = row.get("frame", None)
        if movie_rows and (not movie or group != movie_group or frame == 0):
            emit(movie_rows, True)
            movie_rows = []
        if movie:
            movie_group = group
            movie_rows.append((presentation_id, row))
        else:
            emit([(presentation_id, row)], False)
            movie_group = None
    if movie_rows:
        emit(movie_rows, True)
    trials_frame, members_frame = pd.DataFrame(trials), pd.DataFrame(members)
    if len(members_frame) != len(presentations) or not members_frame["presentation_id"].is_unique:
        raise ValueError("TRIAL_PRESENTATION_ACCOUNTING_MISMATCH")
    if not trials_frame["trial_id"].is_unique:
        raise ValueError("DUPLICATE_TRIAL_ID")
    return trials_frame, members_frame


def diagnostic_selection(unit_ids: list[int] | None, cohort_row: dict[str, Any],
                         config: dict[str, Any]) -> set[int] | None:
    """Explicit engineering selection is limited to predeclared training anchors."""
    if unit_ids is None:
        return None
    if (cohort_row["split"] != "train" or cohort_row["session_id"] not in
            config["cohort"]["exploration_train_session_ids"]):
        raise ValueError("DIAGNOSTIC_SELECTION_REQUIRES_TRAINING_ANCHOR")
    if (not unit_ids or any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                           or value < 0 for value in unit_ids) or len(set(unit_ids)) != len(unit_ids)):
        raise ValueError("DIAGNOSTIC_UNIT_IDS_MUST_BE_NONEMPTY_UNIQUE_INTEGERS")
    return set(map(int, unit_ids))


def qualify_session(data: SessionData, cohort_row: dict[str, Any], config: dict[str, Any], *,
                    progress=None, diagnostic_unit_ids: list[int] | None = None) -> dict[str, Any]:
    diagnostic_ids = diagnostic_selection(diagnostic_unit_ids, cohort_row, config)
    if data.session_id != cohort_row["session_id"] or not cohort_row["included"]:
        raise ValueError("SESSION_COHORT_IDENTITY_MISMATCH")
    if data.subject_id != str(cohort_row["animal_id"]):
        raise ValueError("NWB_SUBJECT_DONOR_IDENTITY_MISMATCH")
    if data.spike_time_unit not in {"s", "seconds"}:
        raise ValueError(f"SPIKE_TIME_UNIT_MUST_BE_SECONDS:{data.spike_time_unit}")
    if data.units.empty or not data.units.index.is_unique:
        raise ValueError("UNIT_TABLE_EMPTY_OR_DUPLICATE")
    if any(not isinstance(index, (int, np.integer)) or isinstance(index, bool) or index < 0 for index in data.units.index):
        raise ValueError("UNIT_IDENTITY_INVALID")
    if set(data.units.index) != set(data.spike_times):
        raise ValueError("UNIT_SPIKE_IDENTITY_MISMATCH")
    if data.source_version_check is not None:
        data.source_version_check()
    trials, members = reconstruct_trials(data.presentations, data.invalid_times, data.session_id,
        cohort_row["animal_id"], cohort_row["split"], config["movie_frames_per_repeat"])
    if progress:
        progress({"event":"TRIAL_METADATA_VALIDATED", "trials":len(trials), "presentations":len(data.presentations)})
    policy = config["unit_qc"]
    qc_annotations = {}
    for unit_id, row in data.units.iterrows():
        reasons = []
        quality = row.get("quality")
        if not isinstance(quality, str) or quality not in policy["accepted_quality"]:
            reasons.append("quality_missing_or_not_accepted")
        metrics = {}
        for name, threshold, lower in (
            ("presence_ratio", policy["presence_ratio_minimum"], True),
            ("amplitude_cutoff", policy["amplitude_cutoff_maximum"], False),
            ("isi_violations", policy["isi_violations_maximum"], False),
        ):
            value = row.get(name, np.nan)
            if not isinstance(value, (float, int, np.number)) or not np.isfinite(value):
                metrics[name] = None
                reasons.append(f"{name}_missing")
            else:
                metrics[name] = float(value)
                if (lower and value <= threshold) or (not lower and value >= threshold):
                    reasons.append(f"{name}_threshold")
        qc_annotations[int(unit_id)] = (quality, metrics, reasons)
    if diagnostic_ids is not None:
        if not diagnostic_ids.issubset(qc_annotations):
            raise ValueError("DIAGNOSTIC_UNIT_ID_NOT_IN_SOURCE")
        if any(qc_annotations[unit][2] for unit in diagnostic_ids):
            raise ValueError("DIAGNOSTIC_SELECTED_UNIT_NOT_QC_ELIGIBLE")
    unit_rows = []
    for processed, (unit_id, row) in enumerate(data.units.iterrows(), 1):
        quality, metrics, reasons = qc_annotations[int(unit_id)]
        omitted = diagnostic_ids is not None and int(unit_id) not in diagnostic_ids
        if reasons or omitted:
            spike_count = data.spike_times.count(unit_id) if isinstance(data.spike_times, NwbSpikeTimes) else len(data.spike_times[unit_id])
            spikes = None
            validation = "NOT_READ_QC_EXCLUDED" if reasons else "NOT_READ_DIAGNOSTIC_LIMIT"
        else:
            spikes = np.asarray(data.spike_times[unit_id])
            validation = "PASS"
            if spikes.ndim != 1 or not np.issubdtype(spikes.dtype, np.number) or np.issubdtype(spikes.dtype, np.complexfloating):
                raise ValueError(f"SPIKE_SHAPE_INVALID:{unit_id}")
            if not np.isfinite(spikes).all() or np.any(spikes[1:] < spikes[:-1]):
                raise ValueError(f"SPIKE_TIMES_INVALID:{unit_id}")
            spike_count = len(spikes)
        unit_rows.append({
            "unit_id": int(unit_id), "session_id": data.session_id,
            "animal_id": cohort_row["animal_id"], "split": cohort_row["split"],
            "quality": quality, **metrics, "spike_count": spike_count,
            "spike_time_validation": validation,
            "source_peak_channel_id": _json_value(row.get("peak_channel_id", None)),
            "first_observed_spike_seconds": float(spikes[0]) if spikes is not None and len(spikes) else None,
            "last_observed_spike_seconds": float(spikes[-1]) if spikes is not None and len(spikes) else None,
            "qc_eligible": not reasons, "exclusion_reasons": ";".join(reasons),
        })
        if progress and (processed % 100 == 0 or processed == len(data.units)):
            progress({"event":"UNIT_SPIKE_VALIDATION_PROGRESS", "units_processed":processed,
                "raw_units":len(data.units), "eligible_units_validated":sum(item["spike_time_validation"] == "PASS" for item in unit_rows),
                "eligible_spikes_validated":sum(item["spike_count"] for item in unit_rows if item["spike_time_validation"] == "PASS")})
    units = pd.DataFrame(unit_rows)
    if data.electrodes is not None and not data.electrodes.empty:
        channel_table = data.electrodes.rename(columns={name:f"recording_channel_{name}" for name in data.electrodes if name != "electrode_id"})
        units = units.merge(channel_table,how="left",left_on="source_peak_channel_id",right_on="electrode_id",validate="many_to_one",sort=False)
        if len(units) != len(data.units):
            raise ValueError("ELECTRODE_LEFT_JOIN_CHANGED_UNIT_DENOMINATOR")
    partial_reasons = []
    if diagnostic_ids is not None:
        partial_reasons.append("diagnostic_unit_selection_not_full_session_qualification")
    if not data.required_columns_present:
        partial_reasons.append("required_source_columns_not_verified")
    if not data.source_full_asset_downloaded and data.source_version_check is None:
        partial_reasons.append("remote_source_version_not_verified")
    criteria = config["eligibility"]
    undefined = sorted(name for name, value in criteria.items() if value is None)
    if undefined:
        partial_reasons.append("eligibility_thresholds_not_frozen:" + ",".join(undefined))
    coverage_verified = data.recording_intervals is not None
    if coverage_verified:
        intervals = np.asarray(data.recording_intervals, dtype=float)
        if intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all() or np.any(intervals[:,1] <= intervals[:,0]):
            raise ValueError("RECORDING_INTERVAL_SHAPE_OR_VALUES_INVALID")
    else:
        intervals = np.empty((0, 2))
        partial_reasons.append("recording_coverage_not_source_verified")
    history, horizon = criteria.get("required_history_seconds"), criteria.get("required_horizon_seconds")
    coverage = []
    for _, trial in trials.iterrows():
        requested_start = trial["start_time"] - history if history is not None else None
        requested_stop = trial["start_time"] + horizon if horizon is not None else None
        if requested_start is None or requested_stop is None or not coverage_verified:
            coverage.append("UNDEFINED")
        elif not np.any((intervals[:,0] <= requested_start) & (intervals[:,1] >= requested_stop)):
            coverage.append("INSUFFICIENT_RECORDING_COVERAGE")
        elif _invalid_overlap(requested_start, requested_stop, _intervals(data.invalid_times, "invalid_times")):
            coverage.append("INVALID_TIME_OVERLAP")
        else:
            coverage.append("PASS")
    trials["requested_window_coverage"] = coverage
    trials["eligible_for_requested_window"] = trials["structurally_eligible"] & (trials["requested_window_coverage"] == "PASS")
    condition_counts = trials.groupby(["stimulus_name", "condition_id"], dropna=False).agg(
        total_trials=("trial_id", "size"), structurally_eligible_trials=("structurally_eligible", "sum"),
        window_eligible_trials=("eligible_for_requested_window", "sum"),
    ).reset_index()
    minimum_units = criteria.get("minimum_eligible_units")
    if minimum_units is not None and int(units["qc_eligible"].sum()) < minimum_units:
        partial_reasons.append("eligible_unit_count_below_required")
    minimum_repeats = criteria.get("minimum_valid_repeats_per_condition")
    if minimum_repeats is not None and (condition_counts["window_eligible_trials"] < minimum_repeats).any():
        partial_reasons.append("condition_repeat_coverage_below_required")
    if not trials["structurally_eligible"].all():
        partial_reasons.append("excluded_or_unresolved_trial_boundaries_present")
    if not units["qc_eligible"].any():
        partial_reasons.append("no_qc_eligible_units")
    if data.source_version_check is not None:
        data.source_version_check()
    summary = {
        "schema": "neural_population_session_qualification_v1",
        "status": "PARTIAL" if partial_reasons else "SESSION_QUALIFIED",
        "qualification_scope": "training_anchor_engineering_sample" if diagnostic_ids is not None else "single_session_required_columns_only",
        "diagnostic_unit_ids": sorted(diagnostic_ids) if diagnostic_ids is not None else None,
        "source_full_asset_downloaded": data.source_full_asset_downloaded,
        "required_columns_complete": bool(data.required_columns_present and
            not (units["qc_eligible"] & (units["spike_time_validation"] != "PASS")).any()),
        "source_metadata_columns_complete": data.required_columns_present,
        "source_etag": data.source_etag,
        "source_version_rechecked": data.source_version_check is not None,
        "spike_verification_scope": "EXPLICIT_DIAGNOSTIC_UNITS_ONLY" if diagnostic_ids is not None else "QC_ELIGIBLE_UNITS_ONLY",
        "electrode_metadata_count":len(data.electrodes) if data.electrodes is not None else 0,
        "location_semantics":"peak recording-channel annotation, not soma position or cell type",
        "behavioral_covariates": {name: {"sample_count": len(frame),
            "finite_value_count": int(frame["value_is_finite"].sum()), **frame.attrs}
            for name, frame in (data.covariates or {}).items()},
        "session_id": data.session_id, "animal_id": cohort_row["animal_id"], "split": cohort_row["split"],
        "nwb_subject_id": data.subject_id, "animal_identity_namespace": "Allen donor_id",
        "specimen_id": cohort_row["specimen_id"], "spike_time_unit": "seconds",
        "spike_time_unit_source": "NWB Units.spike_times seconds specification; explicit unit attribute checked when present",
        "raw_unit_count": len(units), "qc_eligible_unit_count": int(units["qc_eligible"].sum()),
        "raw_spike_count": int(units["spike_count"].sum()),
        "validated_spike_unit_count": int((units["spike_time_validation"] == "PASS").sum()),
        "validated_spike_count": int(units.loc[units["spike_time_validation"] == "PASS", "spike_count"].sum()),
        "diagnostic_omitted_eligible_unit_count": int((units["spike_time_validation"] == "NOT_READ_DIAGNOSTIC_LIMIT").sum()),
        "stimulus_presentation_count": len(data.presentations), "reconstructed_trial_count": len(trials),
        "structurally_eligible_trial_count": int(trials["structurally_eligible"].sum()),
        "invalid_interval_count": len(data.invalid_times), "optotagging_presentation_count": len(data.optotagging),
        "recording_coverage_source_verified": coverage_verified,
        "unit_qc_policy": policy, "partial_reasons": partial_reasons,
        "dataset_ready": False, "model_ready": False, "scientific_claims_qualified": False,
    }
    return {"summary": summary, "units": units, "trials": trials, "trial_members": members,
            "condition_counts": condition_counts,
            "column_evidence": data.column_evidence or {},
            "spike_slice_evidence": getattr(data.spike_times, "consumed_units", {})}


def write_selected_recording(path: Path, data: SessionData, qualification: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Persist consumed numerical source columns and replay every stored slice."""
    import h5py
    if not isinstance(data.spike_times, NwbSpikeTimes) or not data.spike_times.retain_values:
        raise ValueError("SELECTED_SPIKE_VALUES_WERE_NOT_RETAINED")
    if path.exists():
        raise FileExistsError(f"SELECTED_RECORDING_EXISTS:{path}")
    validated = qualification["units"]["qc_eligible"] & (qualification["units"]["spike_time_validation"] == "PASS")
    ids = qualification["units"].loc[validated, "unit_id"].to_numpy(dtype=np.int64)
    selected = data.spike_times.selected_values
    if set(ids) != set(selected):
        raise ValueError("SELECTED_RECORDING_UNIT_COVERAGE_MISMATCH")
    counts = np.array([len(selected[unit]) for unit in ids], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    partial = path.with_suffix(".partial.h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(partial, "x") as output:
        output.attrs.update({"schema":"neural_population_qc_selected_recording_v1",
            "session_id":data.session_id, "donor_id":int(data.subject_id), "spike_time_unit":"seconds",
            "source_file_id":int(source["file_id"]), "source_etag":source.get("etag", ""),
            "source_full_asset_downloaded":data.source_full_asset_downloaded,
            "required_columns_complete":qualification["summary"]["required_columns_complete"],
            "qualification_scope":qualification["summary"]["qualification_scope"],
            "qc_eligible_unit_count":qualification["summary"]["qc_eligible_unit_count"],
            "validated_spike_unit_count":len(ids),
            "dataset_ready":False})
        output.create_dataset("unit_ids",data=ids)
        output.create_dataset("spike_offsets",data=offsets)
        peaks = qualification["units"].loc[validated,"source_peak_channel_id"].to_numpy(dtype=float)
        output.create_dataset("peak_channel_ids",data=peaks)
        if data.electrodes is not None and not data.electrodes.empty:
            channels = output.create_group("electrodes")
            channels.attrs["semantics"]="recording channel, not true soma coordinates or cell type"
            for column in data.electrodes:
                values = data.electrodes[column]
                if values.dtype == object:
                    known = values.notna().to_numpy()
                    strings = [str(value) if present else "" for value,present in zip(values,known)]
                    channels.create_dataset(column,data=strings,dtype=h5py.string_dtype("utf-8"))
                    channels.create_dataset(column+"_known",data=known)
                else:
                    channels.create_dataset(column,data=values.to_numpy())
        times = output.create_dataset("spike_times_s",shape=(int(offsets[-1]),),dtype="float64")
        for index, unit in enumerate(ids):
            values = selected[unit]
            if values.dtype != np.float64:
                raise ValueError("SELECTED_SPIKE_DTYPE_UNSUPPORTED")
            times[int(offsets[index]):int(offsets[index+1])] = values
        for name, frame in (data.covariates or {}).items():
            group = output.create_group(name)
            group.attrs.update(frame.attrs)
            for column in frame:
                group.create_dataset(column,data=frame[column].to_numpy())
        output.flush()
    with h5py.File(partial, "r") as output:
        if not np.array_equal(output["unit_ids"][:], ids) or not np.array_equal(output["spike_offsets"][:], offsets):
            raise ValueError("SELECTED_RECORDING_INDEX_REPLAY_MISMATCH")
        if not np.array_equal(output["peak_channel_ids"][:],peaks,equal_nan=True):
            raise ValueError("SELECTED_RECORDING_PEAK_IDS_REPLAY_MISMATCH")
        if data.electrodes is not None and not data.electrodes.empty:
            for column in data.electrodes:
                values = data.electrodes[column]
                if values.dtype == object:
                    known = values.notna().to_numpy()
                    strings = [str(value) if present else "" for value,present in zip(values,known)]
                    valid = (np.array_equal(output[f"electrodes/{column}_known"][:],known)
                             and list(output[f"electrodes/{column}"].asstr()[:]) == strings)
                else:
                    valid = np.array_equal(output[f"electrodes/{column}"][:],values.to_numpy(),equal_nan=True)
                if not valid:
                    raise ValueError(f"SELECTED_RECORDING_ELECTRODE_REPLAY_MISMATCH:{column}")
        for index, unit in enumerate(ids):
            recorded = output["spike_times_s"][int(offsets[index]):int(offsets[index+1])]
            expected = data.spike_times.consumed_units[str(unit)]["sha256"]
            if hashlib.sha256(recorded.tobytes()).hexdigest() != expected:
                raise ValueError(f"SELECTED_RECORDING_SPIKE_REPLAY_MISMATCH:{unit}")
        for name, frame in (data.covariates or {}).items():
            for column in frame:
                if not np.array_equal(output[f"{name}/{column}"][:], frame[column].to_numpy(), equal_nan=True):
                    raise ValueError(f"SELECTED_RECORDING_STATE_REPLAY_MISMATCH:{name}:{column}")
    partial.rename(path)
    return {"path":str(path), "unit_count":len(ids), "spike_count":int(offsets[-1]),
            "spike_time_unit":"seconds", "recorded_columns_replayed":True,
            "qualification_scope":qualification["summary"]["qualification_scope"],
            "source_full_asset_downloaded":False if not data.source_full_asset_downloaded else True}

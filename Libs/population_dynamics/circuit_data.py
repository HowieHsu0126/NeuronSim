"""Shared, streamed complete-episode archives for source neural circuits.

Unit metadata are stored once per circuit. Atomic shard/catalog publication
keeps interrupted shards out of completed counts. Counts use half-open windows;
events exactly at the source terminal tick remain in the raw archive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from . import v1_runtime_qualification as runtime
from .circuit_campaign import episode_identities, production_prerequisites
from .provenance import canonical_json, sha256_file

SCHEMA = "neural_circuit_episodes_v1"
EVENT_CHUNK = 65536


def _identity(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("INVALID_CIRCUIT_OR_EPISODE_ID")
    return value


def _integers(values, name, *, nonempty=False):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu" or (array < 0).any() or (nonempty and not len(array)):
        raise ValueError("NONNEGATIVE_INTEGER_VECTOR_REQUIRED:" + name)
    if len(array) and int(array.max()) >= 2**63:
        raise ValueError("INTEGER_VECTOR_OUT_OF_RANGE:" + name)
    return array.astype(np.int64, copy=False)


def _atomic_json(path: Path, value: dict):
    temporary = path.with_suffix(".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class CircuitUnits:
    circuit_id: str
    unit_ids: np.ndarray
    populations: np.ndarray
    field_index: np.ndarray
    field_names: tuple[str, ...]
    coordinate_semantics: str
    source_revision: str
    coordinates: np.ndarray | None = None
    coordinate_units: str | None = None
    circuit_provenance: dict[str, str] = field(default_factory=dict)
    node_type_ids: np.ndarray | None = None
    source_tuning_angle_deg: np.ndarray | None = None

    def validate(self):
        _identity(self.circuit_id)
        ids = _integers(self.unit_ids, "unit_ids", nonempty=True)
        groups = _integers(self.field_index, "field_index", nonempty=True)
        if len(np.unique(ids)) != len(ids) or groups.shape != ids.shape:
            raise ValueError("UNIT_IDENTITY_OR_FIELD_MAP_INVALID")
        if (not self.field_names or len(set(self.field_names)) != len(self.field_names)
                or groups.max() >= len(self.field_names)):
            raise ValueError("FIELD_LABELS_INVALID")
        populations = np.asarray(self.populations)
        if populations.shape != ids.shape or populations.dtype.kind not in "US":
            raise ValueError("SOURCE_POPULATION_LABELS_REQUIRED")
        if self.node_type_ids is not None:
            types = _integers(self.node_type_ids, "node_type_ids", nonempty=True)
            if types.shape != ids.shape:
                raise ValueError("SOURCE_NODE_TYPE_IDENTITY_SHAPE_INVALID")
        if self.source_tuning_angle_deg is not None:
            tuning = np.asarray(self.source_tuning_angle_deg)
            if (tuning.shape != ids.shape or tuning.dtype.kind not in "iuf"
                    or not np.isfinite(tuning).all() or np.any(tuning < 0) or np.any(tuning >= 360)):
                raise ValueError("SOURCE_TUNING_ANGLE_ONE_FINITE_DEGREE_VALUE_PER_UNIT_REQUIRED")
        if not self.source_revision or not self.coordinate_semantics:
            raise ValueError("SOURCE_AND_OBSERVATION_SEMANTICS_REQUIRED")
        if any(not self.circuit_provenance.get(k) for k in ("connectivity_identity", "parameter_identity")):
            raise ValueError("REALIZED_CIRCUIT_PROVENANCE_REQUIRED")
        if self.coordinates is not None:
            positions = np.asarray(self.coordinates)
            if (positions.ndim != 2 or positions.shape[0] != len(ids)
                    or not np.isfinite(positions).all() or not self.coordinate_units):
                raise ValueError("SOURCE_COORDINATES_OR_UNITS_INVALID")


@dataclass(frozen=True)
class CircuitEpisode:
    episode_id: str
    circuit_id: str
    block_id: str
    split: str
    duration_ticks: int
    dt_ms: float
    spike_ticks: np.ndarray
    spike_unit_indices: np.ndarray
    control_ticks: np.ndarray
    control_values: np.ndarray
    control_features: tuple[dict[str, str], ...]
    control_kind: str = "piecewise_constant"
    seeds: dict[str, int] = field(default_factory=dict)
    source_record: dict[str, Any] = field(default_factory=dict)
    privileged: dict[str, np.ndarray] = field(default_factory=dict)
    task_outcome: str = "not_assessed"

    def validate(self, units: int):
        for identifier in (self.episode_id, self.circuit_id, self.block_id):
            _identity(identifier)
        if self.split not in {"train", "validation", "calibration", "test", "development"}:
            raise ValueError("UNKNOWN_EPISODE_SPLIT")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("COMPLETE_EPISODE_DURATION_REQUIRED")
        if isinstance(self.dt_ms, bool) or not np.isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise ValueError("POSITIVE_NATIVE_TIME_STEP_REQUIRED")
        ticks = _integers(self.spike_ticks, "spike_ticks")
        indices = _integers(self.spike_unit_indices, "spike_unit_indices")
        if (ticks.shape != indices.shape or (np.diff(ticks) < 0).any()
                or (len(ticks) and (ticks[-1] > self.duration_ticks or indices.max() >= units))):
            raise ValueError("SOURCE_EVENTS_OUTSIDE_CLOCK_OR_UNIT_SUPPORT")
        controls = _integers(self.control_ticks, "control_ticks")
        values = np.asarray(self.control_values)
        if (values.shape != (len(controls), len(self.control_features)) or not np.isfinite(values).all()
                or (len(controls) and ((np.diff(controls) < 0).any() or controls[-1] > self.duration_ticks))):
            raise ValueError("DECLARED_CONTROL_PROGRAM_INVALID")
        if self.control_kind not in {"piecewise_constant", "impulse"}:
            raise ValueError("UNKNOWN_CONTROL_SEMANTICS")
        if self.control_kind == "piecewise_constant" and len(controls) and (np.diff(controls) == 0).any():
            raise ValueError("AMBIGUOUS_SIMULTANEOUS_CONTROL_UPDATES")
        if self.control_kind == "piecewise_constant" and self.control_features and (not len(controls) or controls[0] != 0):
            raise ValueError("INITIAL_CONTROL_AT_TICK_ZERO_REQUIRED")
        if any(not feature.get("name") or not feature.get("unit") for feature in self.control_features):
            raise ValueError("CONTROL_FEATURE_NAMES_AND_UNITS_REQUIRED")
        for name, value in self.privileged.items():
            _identity(name)
            if np.asarray(value).dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("PRIVILEGED_STATE_INVALID")
        if self.source_record.get("complete_unit_recording") is not True:
            raise ValueError("COMPLETE_UNIT_RECORDING_RECEIPT_REQUIRED")
        if (self.source_record.get("simulation_completed") is not True
                or self.source_record.get("recorded_until_tick") != self.duration_ticks
                or self.source_record.get("expected_duration_ticks") != self.duration_ticks
                or self.source_record.get("core_unit_count") != units):
            raise ValueError("SOURCE_EPISODE_COMPLETION_NOT_ESTABLISHED")


class CircuitArchiveWriter:
    def __init__(self, root: Path, binding: dict, *, target_shard_bytes=1 << 30,
                 resume=False, production_config=None, qualification_path=None, identity_table=None,
                 runtime_qualification=None):
        self.root = Path(root)
        self.binding = json.loads(canonical_json(binding))
        if (binding.get("schema") != SCHEMA or not binding.get("source_revision")
                or binding.get("sample_unit") != "complete_mechanistic_episode"):
            raise ValueError("COMPLETE_SOURCE_EPISODE_BINDING_REQUIRED")
        self.production = binding.get("scope") == "production"
        self.expected = None
        self.runtime_document = runtime_qualification
        if self.production:
            if production_config is None or qualification_path is None or identity_table is None:
                raise ValueError("PRODUCTION_QUALIFICATION_REQUIRED")
            qualification = json.loads(Path(qualification_path).read_text())
            missing = production_prerequisites(production_config, binding["scenario_id"], qualification)
            if missing or qualification.get("source_revision") != binding["source_revision"]:
                raise ValueError("PRODUCTION_SOURCE_NOT_QUALIFIED:" + ",".join(missing))
            identity_table = Path(identity_table)
            frozen = json.loads(identity_table.with_suffix(".json").read_text())
            if (frozen["resolved_campaign"] != production_config
                    or frozen["identity_table_sha256"] != sha256_file(identity_table)
                    or frozen["source_qualification_sha256"] != sha256_file(Path(qualification_path))):
                raise ValueError("FROZEN_PRODUCTION_IDENTITY_BINDING_CHANGED")
            self.expected = {}
            with identity_table.open(newline="") as stream:
                actual = csv.DictReader(stream)
                expected_rows = episode_identities(production_config, binding["scenario_id"], production=True)
                for reference in expected_rows:
                    row = next(actual, None)
                    if row != {k: str(v) for k, v in reference.items()}:
                        raise ValueError("FROZEN_PRODUCTION_IDENTITY_ROW_CHANGED")
                    self.expected[reference["episode_id"]] = reference
                if next(actual, None) is not None:
                    raise ValueError("EXTRA_PRODUCTION_IDENTITIES")
            self.binding["identity_table_sha256"] = frozen["identity_table_sha256"]
            self.binding["source_qualification_sha256"] = frozen["source_qualification_sha256"]
            self.task_protocol = production_config["scenarios"][binding["scenario_id"]]["task_protocol"]
        elif binding.get("scope") != "source_reproduction":
            raise ValueError("ARCHIVE_SCOPE_REQUIRED")
        if type(target_shard_bytes) is not int or target_shard_bytes < 1:
            raise ValueError("POSITIVE_SHARD_TARGET_REQUIRED")
        self.target = target_shard_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        catalog_path = self.root / "dataset.json"
        if catalog_path.exists():
            if not resume:
                raise FileExistsError("ARCHIVE_REQUIRES_EXPLICIT_RESUME")
            self.catalog = json.loads(catalog_path.read_text())
            if self.catalog["binding"] != self.binding:
                raise ValueError("ARCHIVE_RESUME_BINDING_CHANGED")
        else:
            self.catalog = dict(binding=self.binding, circuits={}, shards=[], complete_episodes=0,
                                source_fidelity_qualified=self.production)
        if self.runtime_document is not None:
            if not self.production or self.binding["scenario_id"] != "v1_visual":
                raise ValueError("GLIF_RUNTIME_REQUIRES_V1_PRODUCTION")
            runtime.registered_backend(self.runtime_document, self.binding["source_qualification_sha256"])
            registry = self.catalog.setdefault("runtime_qualifications", {})
            sha = self.runtime_document["sha256"]
            if sha in registry and registry[sha] != self.runtime_document:
                raise ValueError("GLIF_RUNTIME_REGISTRY_CHANGED")
            registry[sha] = self.runtime_document
        runtime.validate_registry(self.binding, self.catalog.get("runtime_qualifications", {}))
        self.ids = {row["episode_id"] for shard in self.catalog["shards"] for row in shard["episodes"]}
        self.circuit_splits = {row["circuit_id"]: row["split"] for shard in self.catalog["shards"] for row in shard["episodes"]}
        self.pending, self.handle, self.pending_bytes = [], None, 0
        self.temporary = None
        if resume:
            self._recover()

    def _recover(self):
        """Publish a closed shard, or retain an unfinished attempt before retry."""
        index = len(self.catalog["shards"])
        closed = self.root / f"episodes_{index:06d}.h5"
        partial = closed.with_suffix(".h5.writing")
        if closed.exists():
            with h5py.File(closed, "r") as source:
                rows = json.loads(source.attrs["completed_episode_index_json"])
                if set(source.keys()) != {row["episode_id"] for row in rows}:
                    raise ValueError("INTERRUPTED_SHARD_INDEX_MISMATCH")
                # Authenticate every new runtime reference before mutating the
                # recovered identity set or publishing the closed shard.
                for row in rows:
                    metadata = json.loads(source[row["episode_id"]].attrs["metadata_json"])
                    runtime.validate_episode(self.binding,
                        self.catalog.get("runtime_qualifications", {}), metadata, row)
                for row in rows:
                    if (row["episode_id"] in self.ids
                            or row["circuit_id"] not in self.catalog["circuits"]
                            or self.circuit_splits.get(row["circuit_id"], row["split"]) != row["split"]):
                        raise ValueError("INTERRUPTED_SHARD_IDENTITY_CONFLICT")
                    self.ids.add(row["episode_id"])
                    self.circuit_splits[row["circuit_id"]] = row["split"]
            self.catalog["shards"].append(dict(path=closed.name, sha256=sha256_file(closed), episodes=rows))
            self.catalog["complete_episodes"] += len(rows)
        if partial.exists():
            attempts = self.catalog.setdefault("incomplete_packaging_attempts", [])
            target = self.root / "incomplete" / f"episodes_{index:06d}_attempt_{len(attempts):04d}.h5"
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise FileExistsError("INCOMPLETE_ATTEMPT_DESTINATION_EXISTS")
            partial.replace(target)
            attempts.append(dict(path=str(target.relative_to(self.root)), counted_episodes=0))
        _atomic_json(self.root / "dataset.json", self.catalog)

    def register_circuit(self, units: CircuitUnits):
        units.validate()
        if units.source_revision != self.binding["source_revision"]:
            raise ValueError("CIRCUIT_SOURCE_REVISION_CHANGED")
        if self.production and len(units.unit_ids) < 100000:
            raise ValueError("PRODUCTION_CORE_SCALE_TOO_SMALL")
        if units.circuit_id in self.catalog["circuits"]:
            existing = self.catalog["circuits"][units.circuit_id]
            path = self.root / existing["path"]
            if sha256_file(path) != existing["sha256"]:
                raise ValueError("CIRCUIT_METADATA_CHANGED")
            with h5py.File(path, "r") as source:
                for name in ("unit_ids", "field_index", "coordinates", "node_type_ids", "source_tuning_angle_deg"):
                    value = getattr(units, name)
                    if (name in source) != (value is not None) or (value is not None and not np.array_equal(source[name][:], value)):
                        raise ValueError("CIRCUIT_ID_REBOUND:" + name)
                if (not np.array_equal(source["populations"].asstr()[:], units.populations)
                        or tuple(source["field_names"].asstr()[:]) != units.field_names
                        or source.attrs["coordinate_semantics"] != units.coordinate_semantics
                        or source.attrs["coordinate_units"] != (units.coordinate_units or "not_available")
                        or json.loads(source.attrs["circuit_provenance_json"]) != units.circuit_provenance):
                    raise ValueError("CIRCUIT_OBSERVATION_MAP_REBOUND")
            return
        relative = "circuits/" + units.circuit_id + ".h5"
        path = self.root / relative
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix(".writing")
        if path.exists() or temporary.exists():
            raise FileExistsError("UNREGISTERED_CIRCUIT_FILE_REQUIRES_INSPECTION")
        with h5py.File(temporary, "x") as out:
            out.create_dataset("unit_ids", data=units.unit_ids)
            out.create_dataset("field_index", data=units.field_index)
            if units.node_type_ids is not None:
                out.create_dataset("node_type_ids", data=units.node_type_ids)
            if units.source_tuning_angle_deg is not None:
                out.create_dataset("source_tuning_angle_deg", data=units.source_tuning_angle_deg)
            strings = h5py.string_dtype("utf-8")
            out.create_dataset("populations", data=np.asarray(units.populations, dtype=object), dtype=strings)
            out.create_dataset("field_names", data=np.asarray(units.field_names, dtype=object), dtype=strings)
            if units.coordinates is not None:
                out.create_dataset("coordinates", data=units.coordinates)
            out.attrs.update(coordinate_semantics=units.coordinate_semantics,
                             coordinate_units=units.coordinate_units or "not_available",
                             source_revision=units.source_revision,
                             circuit_provenance_json=canonical_json(units.circuit_provenance))
        temporary.replace(path)
        self.catalog["circuits"][units.circuit_id] = dict(path=relative, sha256=sha256_file(path),
                                                        units=len(units.unit_ids), provenance=units.circuit_provenance)
        _atomic_json(self.root / "dataset.json", self.catalog)

    def add(self, episode: CircuitEpisode):
        if episode.circuit_id not in self.catalog["circuits"]:
            raise ValueError("REGISTER_CIRCUIT_BEFORE_EPISODES")
        episode.validate(self.catalog["circuits"][episode.circuit_id]["units"])
        if (episode.source_record.get("source_revision") != self.binding["source_revision"]
                or episode.source_record.get("circuit_provenance") != self.catalog["circuits"][episode.circuit_id]["provenance"]):
            raise ValueError("EPISODE_SOURCE_OR_CIRCUIT_BINDING_CHANGED")
        if self.production:
            reference = self.expected.get(episode.episode_id)
            if reference is None or any(getattr(episode, k) != reference[k] for k in ("circuit_id", "block_id", "split")):
                raise ValueError("EPISODE_NOT_IN_FROZEN_PRODUCTION_COHORT")
            if episode.seeds != {k: reference[k] for k in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed")}:
                raise ValueError("PRODUCTION_RANDOM_STREAM_IDENTITY_CHANGED")
            expected_duration = self.task_protocol["condition_duration_ticks"][reference["condition_index"]]
            if episode.duration_ticks != expected_duration or episode.dt_ms != self.task_protocol["dt_ms"]:
                raise ValueError("EPISODE_CLOCK_DIFFERS_FROM_FROZEN_PROTOCOL")
        runtime_sha = episode.source_record.get("production_source_admission", {}).get(
            "runtime_qualification_sha256")
        selected_sha = self.runtime_document["sha256"] if self.runtime_document is not None else None
        if runtime_sha != selected_sha:
            raise ValueError("GLIF_RUNTIME_WRITER_SELECTION_DIFFERS")
        metadata = {name: getattr(episode, name) for name in
                    ("episode_id", "circuit_id", "block_id", "split", "seeds", "source_record")}
        runtime.validate_episode(self.binding, self.catalog.get("runtime_qualifications", {}),
                                 metadata, {"runtime_qualification_sha256": runtime_sha})
        if episode.episode_id in self.ids:
            raise ValueError("DUPLICATE_COMPLETE_EPISODE")
        old_split = self.circuit_splits.get(episode.circuit_id)
        if old_split is not None and old_split != episode.split:
            raise ValueError("WHOLE_CIRCUIT_SPLIT_LEAKAGE")
        if self.production and episode.split == "development":
            raise ValueError("DEVELOPMENT_EPISODE_IN_PRODUCTION")
        if self.handle is None:
            index = len(self.catalog["shards"])
            self.temporary = self.root / f"episodes_{index:06d}.h5.writing"
            if self.temporary.exists() or self.temporary.with_suffix("").exists():
                raise FileExistsError("INTERRUPTED_SHARD_REQUIRES_INSPECTION")
            self.handle = h5py.File(self.temporary, "x")
        group = self.handle.create_group(episode.episode_id)
        for name in ("spike_ticks", "spike_unit_indices", "control_ticks", "control_values"):
            value = np.asarray(getattr(episode, name))
            group.create_dataset(name, data=value, compression="gzip", shuffle=True)
            self.pending_bytes += value.nbytes
        if episode.privileged:
            private = group.create_group("privileged")
            for name, value in episode.privileged.items():
                private.create_dataset(name, data=value, compression="gzip", shuffle=True)
                self.pending_bytes += value.nbytes
        metadata = {name: getattr(episode, name) for name in (
            "episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms", "control_features",
            "control_kind", "seeds", "source_record", "task_outcome")}
        group.attrs["metadata_json"] = canonical_json(metadata)
        self.pending.append(dict(episode_id=episode.episode_id, circuit_id=episode.circuit_id,
                                 block_id=episode.block_id, split=episode.split,
                                 duration_ticks=episode.duration_ticks, dt_ms=episode.dt_ms,
                                 event_count=len(episode.spike_ticks)))
        if runtime_sha is not None:
            self.pending[-1]["runtime_qualification_sha256"] = runtime_sha
        self.ids.add(episode.episode_id)
        self.circuit_splits[episode.circuit_id] = episode.split
        if self.pending_bytes >= self.target:
            self.flush()

    def flush(self):
        if self.handle is None:
            return
        self.handle.attrs["completed_episode_index_json"] = canonical_json(self.pending)
        self.handle.flush()
        self.handle.close()
        self.handle = None
        path = self.temporary.with_suffix("")
        self.temporary.replace(path)
        self.catalog["shards"].append(dict(path=path.name, sha256=sha256_file(path), episodes=self.pending))
        self.catalog["complete_episodes"] += len(self.pending)
        _atomic_json(self.root / "dataset.json", self.catalog)
        self.pending, self.pending_bytes, self.temporary = [], 0, None

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        elif self.handle is not None:
            self.handle.close()
            self.handle = None
            # No catalog entry refers to this task-owned uncommitted shard.
            self.temporary.unlink()


class CircuitDataset:
    def __init__(self, root: Path, *, splits=("train", "validation", "calibration", "development"),
                 allow_test=False):
        self.root = Path(root)
        self.catalog = json.loads((self.root / "dataset.json").read_text())
        runtime.validate_registry(self.catalog["binding"], self.catalog.get("runtime_qualifications", {}))
        if self.catalog["binding"]["schema"] != SCHEMA:
            raise ValueError("UNKNOWN_CIRCUIT_ARCHIVE_SCHEMA")
        if "test" in splits and not allow_test:
            raise ValueError("FORMAL_TEST_ACCESS_REQUIRES_EXPLICIT_REQUEST")
        if (not splits or len(set(splits)) != len(splits)
                or not set(splits) <= {"train", "validation", "calibration", "test", "development"}):
            raise ValueError("EXPLICIT_UNIQUE_SUPPORTED_SPLITS_REQUIRED")
        self.episodes = {}
        self.shards = {}
        identities, circuit_splits = set(), {}
        for shard in self.catalog["shards"]:
            if shard["path"] in self.shards:
                raise ValueError("DUPLICATE_CIRCUIT_SHARD")
            self.shards[shard["path"]] = shard
            for row in shard["episodes"]:
                if row["split"] not in {"train", "validation", "calibration", "test", "development"}:
                    raise ValueError("UNKNOWN_CIRCUIT_EPISODE_SPLIT")
                if row["episode_id"] in identities:
                    raise ValueError("DUPLICATE_COMPLETE_EPISODE")
                identities.add(row["episode_id"])
                if row["circuit_id"] not in self.catalog["circuits"]:
                    raise ValueError("EPISODE_CIRCUIT_METADATA_MISSING")
                previous = circuit_splits.setdefault(row["circuit_id"], row["split"])
                if previous != row["split"]:
                    raise ValueError("WHOLE_CIRCUIT_SPLIT_LEAKAGE")
                if row["split"] in splits:
                    self.episodes[row["episode_id"]] = dict(row, shard=shard["path"])
        if len(identities) != self.catalog["complete_episodes"]:
            raise ValueError("COMPLETE_EPISODE_CATALOG_COUNT_DIFFERS")
        self.circuit_splits = circuit_splits
        self.verified = set()

    def validate_runtime_metadata(self, metadata, row):
        runtime.validate_episode(self.catalog["binding"],
                                 self.catalog.get("runtime_qualifications", {}), metadata, row)

    def _verified(self, relative, digest):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("ARCHIVE_PATH_ESCAPES_DATASET")
        if relative not in self.verified:
            if sha256_file(path) != digest:
                raise ValueError("ARCHIVE_CONTENT_CHANGED:" + relative)
            self.verified.add(relative)
        return path

    def iter_episodes(self):
        yield from self.episodes.values()

    def circuit_metadata(self, circuit_id):
        """Read source-declared static descriptors without episode outcomes."""
        if not any(row["circuit_id"] == circuit_id for row in self.episodes.values()):
            raise ValueError("CIRCUIT_OUTSIDE_SELECTED_SPLITS")
        circuit = self.catalog["circuits"][circuit_id]
        with h5py.File(self._verified(circuit["path"], circuit["sha256"]), "r") as source:
            result = dict(circuit_id=circuit_id, unit_ids=source["unit_ids"][:],
                unit_field_index=source["field_index"][:], populations=source["populations"].asstr()[:],
                field_names=tuple(source["field_names"].asstr()[:]),
                coordinate_semantics=source.attrs["coordinate_semantics"],
                coordinate_units=source.attrs["coordinate_units"],
                source_revision=source.attrs["source_revision"],
                circuit_provenance=json.loads(source.attrs["circuit_provenance_json"]))
            for name in ("coordinates", "node_type_ids", "source_tuning_angle_deg"):
                result[name] = source[name][:] if name in source else None
        return result

    def complete_episode(self, episode_id, *, bin_ticks):
        """Read every modeled tick in [0,duration); retain endpoint events in raw data.

        The caller selects an exact source-clock divisor. Scientific time
        resolution is never inferred by rounding or silently changing bins.
        Privileged states and task outcomes do not enter this training interface.
        """
        row = self.episodes[episode_id]
        if (type(bin_ticks) is not int or bin_ticks < 1
                or row["duration_ticks"] % bin_ticks):
            raise ValueError("EXACT_COMPLETE_EPISODE_BIN_DIVISOR_REQUIRED")
        values = self._read(episode_id, 0, row["duration_ticks"], bin_ticks,
                            0, self.catalog["circuits"][row["circuit_id"]]["units"])
        return dict(values, block_id=row["block_id"], split=row["split"],
                    duration_ticks=row["duration_ticks"], bin_ticks=bin_ticks,
                    consumed_start_tick=0, consumed_stop_tick_exclusive=row["duration_ticks"])

    def window(self, episode_id, *, boundary_tick, history_ticks, horizon_ticks, bin_ticks,
               unit_start=0, unit_stop=None):
        row = self.episodes[episode_id]
        values = (boundary_tick, history_ticks, horizon_ticks, bin_ticks, unit_start)
        if any(type(v) is not int for v in values) or min(history_ticks, horizon_ticks, bin_ticks) < 1:
            raise ValueError("POSITIVE_INTEGER_WINDOW_REQUIRED")
        start, stop = boundary_tick - history_ticks, boundary_tick + horizon_ticks
        if start < 0 or stop > row["duration_ticks"] or history_ticks % bin_ticks or horizon_ticks % bin_ticks:
            raise ValueError("CAUSAL_WINDOW_OUTSIDE_COMPLETE_EPISODE")
        circuit = self.catalog["circuits"][row["circuit_id"]]
        if unit_stop is None:
            unit_stop = circuit["units"]
        if type(unit_stop) is not int or not 0 <= unit_start < unit_stop <= circuit["units"]:
            raise ValueError("UNIT_CHUNK_OUTSIDE_SOURCE_MAP")
        values = self._read(episode_id, start, stop, bin_ticks, unit_start, unit_stop)
        past = history_ticks // bin_ticks
        counts, fields = values.pop("counts"), values.pop("field_counts")
        return dict(values, history=counts[:past], targets=counts[past:],
                    history_fields=fields[:past], target_fields=fields[past:],
                    observation_mask=np.ones_like(counts, dtype=bool))

    def _read(self, episode_id, start, stop, bin_ticks, unit_start, unit_stop):
        row = self.episodes[episode_id]
        circuit = self.catalog["circuits"][row["circuit_id"]]
        with h5py.File(self._verified(circuit["path"], circuit["sha256"]), "r") as units:
            if len(units["unit_ids"]) != circuit["units"]:
                raise ValueError("SOURCE_UNIT_COUNT_DIFFERS_FROM_CATALOG")
            ids = units["unit_ids"][unit_start:unit_stop]
            field_index = units["field_index"][:]
            fields = tuple(units["field_names"].asstr()[:])
        bins = (stop - start) // bin_ticks
        counts = np.zeros((bins, len(ids)), dtype=np.int64)
        field_counts = np.zeros((bins, len(fields)), dtype=np.int64)
        shard = self.shards[row["shard"]]
        with h5py.File(self._verified(shard["path"], shard["sha256"]), "r") as source:
            group = source[episode_id]
            metadata = json.loads(group.attrs["metadata_json"])
            self.validate_runtime_metadata(metadata, row)
            if any(metadata[name] != row[name] for name in
                   ("episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms")):
                raise ValueError("SOURCE_EPISODE_IDENTITY_DIFFERS_FROM_CATALOG")
            if (type(row["event_count"]) is not int or row["event_count"] < 0
                    or any(group[name].shape != (row["event_count"],)
                           for name in ("spike_ticks", "spike_unit_indices"))):
                raise ValueError("SOURCE_EVENT_COUNT_DIFFERS_FROM_CATALOG")
            for offset in range(0, row["event_count"], EVENT_CHUNK):
                ticks = group["spike_ticks"][offset:offset + EVENT_CHUNK]
                if len(ticks) and ticks[0] >= stop:
                    break
                selected = (ticks >= start) & (ticks < stop)
                indices = group["spike_unit_indices"][offset:offset + EVENT_CHUNK][selected].astype(np.int64)
                times = (ticks[selected].astype(np.int64) - start) // bin_ticks
                np.add.at(field_counts, (times, field_index[indices]), 1)
                keep = (indices >= unit_start) & (indices < unit_stop)
                np.add.at(counts, (times[keep], indices[keep] - unit_start), 1)
            control_ticks = group["control_ticks"][:]
            control_values = group["control_values"][:]
        controls = _control_bins(control_ticks, control_values, metadata["control_kind"], start, stop, bin_ticks)
        return dict(episode_id=episode_id, circuit_id=row["circuit_id"], unit_ids=ids,
                    unit_field_index=field_index[unit_start:unit_stop],
                    counts=counts, field_counts=field_counts,
                    known_controls=controls, field_names=fields,
                    bin_seconds=bin_ticks * row["dt_ms"] / 1000, source_dt_ms=row["dt_ms"],
                    control_features=metadata["control_features"], control_kind=metadata["control_kind"])


def _control_bins(ticks, values, kind, start, stop, width):
    result = np.zeros(((stop - start) // width, values.shape[1]), dtype=np.float64)
    if kind == "impulse":
        keep = (ticks >= start) & (ticks < stop)
        np.add.at(result, (ticks[keep].astype(np.int64) - start) // width, values[keep])
    else:
        for i, time in enumerate(ticks):
            end = ticks[i + 1] if i + 1 < len(ticks) else stop
            left, right = max(start, int(time)), min(stop, int(end))
            while left < right:
                index = (left - start) // width
                until = min(right, start + (index + 1) * width)
                result[index] += values[i] * ((until - left) / width)
                left = until
    return result

"""Source-bound V1 GLIF events for the shared population observation model.

The complete recorded episode includes gray warmup. Reset is tick zero;
emission starts at tick one. Model inputs contain observed cortical spikes,
source static descriptors and visual commands, never realized external events.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .circuit_data import CircuitDataset
from .provenance import canonical_json, sha256_file
from .refractory_observation import RenewalWindow, renewal_window
from .v1_nest_time_semantics import clock_tics_per_step, source_refractory_steps
from .v1_visual_controls import CONTROL_FEATURES

MINIMUM_INTERVAL_POLICY = "source_glif_refractory_counter_steps_plus_one"
RESET_POLICY = "deterministic_source_reset_then_supplied_stochastic_warmup"


def _document(record, name, expected):
    document = record["production_source_admission"]["source_documents"][name]
    payload = document["payload"]
    if (document["sha256"] != expected
            or hashlib.sha256(payload.encode("utf-8")).hexdigest() != expected):
        raise ValueError("V1_RENEWAL_SOURCE_DOCUMENT_CHANGED:" + name)
    return json.loads(payload)


class V1RenewalEpisode:
    """Read one admitted complete V1 episode and its source-native parameters."""

    def __init__(self, data: CircuitDataset, episode_id: str, *, bin_ticks: int,
                 native_parameters: Path):
        row = data.episodes[episode_id]  # Enforce the selected split before source access.
        binding = data.catalog["binding"]
        if (binding.get("scenario_id") != "v1_visual" or binding.get("scope") != "production"
                or binding.get("sample_unit") != "complete_mechanistic_episode"
                or not binding.get("source_qualification_sha256")):
            raise ValueError("V1_RENEWAL_QUALIFIED_PRODUCTION_SOURCE_REQUIRED")
        if type(bin_ticks) is not int or bin_ticks < 1 or row["duration_ticks"] % bin_ticks:
            raise ValueError("V1_RENEWAL_EXACT_COMPLETE_CLOCK_REQUIRED")
        source = data.circuit_metadata(row["circuit_id"])
        shard = data.shards[row["shard"]]
        with h5py.File(data._verified(shard["path"], shard["sha256"]), "r") as stored:
            group = stored[episode_id]
            metadata = json.loads(group.attrs["metadata_json"])
            if any(metadata[key] != row[key] for key in
                   ("episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms")):
                raise ValueError("V1_RENEWAL_EPISODE_IDENTITY_CHANGED")
            ticks, units = group["spike_ticks"][:], group["spike_unit_indices"][:]
        record = metadata["source_record"]
        admission = record["production_source_admission"]
        declaration = record["production_declaration"]
        if (record.get("complete_unit_recording") is not True or record.get("simulation_completed") is not True
                or record.get("recorded_until_tick") != row["duration_ticks"]
                or record.get("core_unit_count") != len(source["unit_ids"])
                or record.get("source_revision") != source["source_revision"]
                or source["source_revision"] != binding["source_revision"]
                or record.get("circuit_provenance") != source["circuit_provenance"]
                or declaration.get("production_eligible") is not True
                or declaration.get("intended_split") != row["split"]
                or declaration.get("circuit_id") != row["circuit_id"]
                or any(item.get("source_qualification_sha256") != binding["source_qualification_sha256"]
                       for item in (declaration, admission))
                or any(admission["cohort_identity"].get(k) != row[k] for k in
                       ("episode_id", "circuit_id", "block_id", "split"))):
            raise ValueError("V1_RENEWAL_SOURCE_ADMISSION_CHANGED")
        reference = _document(record, "circuit_reference", record["circuit_reference_report_sha256"])
        execution = _document(record, "execution", record["execution_report_sha256"])
        if (execution.get("status") != "PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION"
                or execution.get("source_reset_policy") != RESET_POLICY
                or execution.get("units") != len(source["unit_ids"])
                or reference.get("native_parameters_and_initial_state_exact") is not True):
            raise ValueError("V1_RENEWAL_SOURCE_RESET_OR_PARAMETERS_UNSUPPORTED")
        clock = reference["execution_clock"]
        if (clock["resolution_ms"] != row["dt_ms"]
                or clock["duration_ms"] != row["duration_ticks"] * row["dt_ms"]
                or clock["tics_per_step"] != clock_tics_per_step(clock["tics_per_ms"], row["dt_ms"])):
            raise ValueError("V1_RENEWAL_SOURCE_CLOCK_CHANGED")
        parameters = json.loads(Path(native_parameters).read_text())
        identity = hashlib.sha256(canonical_json({key: dict(model=item["model"], parameters=item["parameters"])
                                    for key, item in parameters.items()}).encode("utf-8")).hexdigest()
        if identity != source["circuit_provenance"]["parameter_identity"]:
            raise ValueError("V1_RENEWAL_NATIVE_PARAMETER_IDENTITY_CHANGED")
        types = source["node_type_ids"]
        if types is None or types.shape != source["unit_ids"].shape:
            raise ValueError("V1_RENEWAL_SOURCE_NODE_TYPES_REQUIRED")
        unique, inverse, sizes = np.unique(types, return_inverse=True, return_counts=True)
        if set(map(str, unique)) != set(parameters):
            raise ValueError("V1_RENEWAL_SOURCE_PARAMETER_TYPE_COVERAGE_CHANGED")
        rows = [parameters[str(key)] for key in unique]
        if any(item.get("model") != "glif_psc" or item.get("all_units_exact") is not True
               or item.get("units") != int(size) for item, size in zip(rows, sizes)):
            raise ValueError("V1_RENEWAL_SOURCE_PARAMETER_UNIT_MAP_CHANGED")
        self.parameters = [item["parameters"] for item in rows]
        refractory_ms = np.asarray([item["t_ref"] for item in self.parameters])
        counters = source_refractory_steps(refractory_ms, clock["tics_per_ms"], row["dt_ms"])
        # Current qualified GLIF sources have positive dead time. The shared
        # sampler requires ready < interval, including the tick-one reset delay.
        if np.any(counters < 1):
            raise ValueError("V1_RENEWAL_ZERO_REFRACTORY_RESET_NOT_SUPPORTED")
        self.minimum_interval_ticks = (counters + 1)[inverse]
        n = len(source["unit_ids"])
        if (ticks.shape != (row["event_count"],) or units.shape != ticks.shape
                or ticks.dtype.kind not in "iu" or units.dtype.kind not in "iu"
                or np.any(ticks < 1) or np.any(ticks > row["duration_ticks"])
                or np.any(ticks[1:] < ticks[:-1]) or np.any(units < 0) or np.any(units >= n)):
            raise ValueError("V1_RENEWAL_NATIVE_EVENT_SUPPORT_INVALID")
        if metadata["control_features"] != list(CONTROL_FEATURES) or metadata["control_kind"] != "piecewise_constant":
            raise ValueError("V1_RENEWAL_SOURCE_VISUAL_COMMANDS_REQUIRED")
        self.data, self.episode_id, self.source = data, episode_id, source
        self.unit_ids, self.unit_field_index = source["unit_ids"], source["unit_field_index"]
        self.field_names, self.model_control_features = source["field_names"], list(CONTROL_FEATURES)
        self.event_ticks, self.event_units, self.type_indices = ticks, units, inverse
        self.bin_ticks, self.duration_ticks, self.dt_ms = bin_ticks, row["duration_ticks"], row["dt_ms"]
        self.binding = dict(episode_id=episode_id, circuit_id=row["circuit_id"], block_id=row["block_id"],
            split=row["split"], source_revision=source["source_revision"],
            source_qualification_sha256=binding["source_qualification_sha256"],
            circuit_sha256=data.catalog["circuits"][row["circuit_id"]]["sha256"], shard_sha256=shard["sha256"],
            native_parameters_sha256=sha256_file(Path(native_parameters)), parameter_identity=identity,
            source_reference_sha256=record["circuit_reference_report_sha256"],
            source_execution_sha256=record["execution_report_sha256"], minimum_interval_policy=MINIMUM_INTERVAL_POLICY,
            initial_availability="reset_boundary_tick_0; first_emission_tick_1; gray_warmup_retained",
            bin_ticks=bin_ticks, source_dt_ms=self.dt_ms)

    def window(self, start_tick: int, stop_tick: int):
        if (type(start_tick) is not int or type(stop_tick) is not int
                or not 0 <= start_tick < stop_tick <= self.duration_ticks
                or start_tick % self.bin_ticks or stop_tick % self.bin_ticks):
            raise ValueError("V1_RENEWAL_COMPLETE_ALIGNED_WINDOW_REQUIRED")
        value = self.data._read(self.episode_id, start_tick, stop_tick, self.bin_ticks, 0, len(self.unit_ids))
        return value["counts"], value["known_controls"]

    def renewal_window(self, start_tick: int, stop_tick: int) -> RenewalWindow:
        value = renewal_window(self, start_tick, stop_tick, self.minimum_interval_ticks)
        if start_tick != 0:
            return value
        # The shared source default allows tick-zero emission. V1 timestamps
        # follow integration, so reset consumes exactly one unavailable tick.
        eligible = value.eligible_ticks.copy()
        eligible[0] -= 1
        if np.any(eligible < value.counts):
            raise ValueError("V1_RENEWAL_RESET_EXPOSURE_INVALID")
        return RenewalWindow(value.counts, eligible, value.stimulus,
                             np.ones(len(self.unit_ids), dtype=np.int64), value.terminal_ready_in_ticks)

    def ready_at(self, tick: int) -> np.ndarray:
        """Use only observed events strictly before the requested boundary."""
        if type(tick) is not int or not 0 <= tick <= self.duration_ticks:
            raise ValueError("V1_RENEWAL_CAUSAL_BOUNDARY_REQUIRED")
        stop = np.searchsorted(self.event_ticks, tick, side="left")
        ticks, units = self.event_ticks[:stop], self.event_units[:stop]
        order = np.lexsort((ticks, units))
        same = np.diff(units[order]) == 0
        if np.any(np.diff(ticks[order])[same] < self.minimum_interval_ticks[units[order][1:][same]]):
            raise ValueError("observed spike train violates the declared minimum interval")
        last = 1 - self.minimum_interval_ticks
        np.maximum.at(last, units, ticks)
        return np.maximum(last + self.minimum_interval_ticks - tick, 0)

    def population_metadata(self):
        """Source coordinates and GLIF parameters; no outcome-fitted descriptors."""
        source = self.source
        if source["coordinate_units"] != "um" or source["coordinates"].shape != (len(self.unit_ids), 3):
            raise ValueError("V1_RENEWAL_SOURCE_POINT_COORDINATES_REQUIRED")
        tuning = source["source_tuning_angle_deg"]
        if tuning is None or tuning.shape != self.unit_ids.shape or not np.isfinite(tuning).all():
            raise ValueError("V1_RENEWAL_SOURCE_TUNING_REQUIRED")
        intrinsic = np.asarray([[p["C_m"] / 100, p["g"] / 10,
                                 (p["V_th"] - p["V_reset"]) / 100, p["t_ref"]]
                                for p in self.parameters])[self.type_indices]
        angle = np.deg2rad(tuning)
        unit = np.column_stack((source["coordinates"] / 1000, np.sin(angle), np.cos(angle), intrinsic))
        if not np.isfinite(unit).all():
            raise ValueError("V1_RENEWAL_STATIC_DESCRIPTORS_NONFINITE")
        def descriptor(name, unit, meaning, transform):
            return dict(name=name, unit=unit, meaning=meaning, transform=transform)
        schema = [descriptor("source_" + axis, "um", "source-sampled point-neuron coordinate", "divide_by_1000")
                  for axis in ("x", "y", "z")]
        schema += [descriptor("tuning_sin", "degree", "source tuning angle", "sin_degrees"),
                   descriptor("tuning_cos", "degree", "source tuning angle", "cos_degrees"),
                   descriptor("capacitance", "pF", "source GLIF membrane capacitance", "divide_by_100"),
                   descriptor("conductance", "nS", "source GLIF leak conductance", "divide_by_10"),
                   descriptor("threshold_reset_gap", "mV", "source threshold minus reset voltage", "divide_by_100"),
                   descriptor("refractory_duration", "ms", "source GLIF t_ref before clock conversion", "identity")]
        sizes = np.bincount(self.unit_field_index, minlength=len(self.field_names))
        field = np.zeros((len(self.field_names), unit.shape[1]))
        for index in range(unit.shape[1]):
            np.add.at(field[:, index], self.unit_field_index, unit[:, index])
        field /= sizes[:, None]
        field = np.column_stack((field, np.log1p(sizes)))
        fields = [dict(item, name=item["name"] + "_mean", meaning="source field mean: " + item["meaning"])
                  for item in schema]
        fields.append(descriptor("field_size", "neurons", "complete source field unit count", "log1p"))
        return dict(source_identity=dict(circuit_id=self.binding["circuit_id"], source_revision=source["source_revision"],
                    **{key: source["circuit_provenance"][key] for key in ("connectivity_identity", "parameter_identity")}),
            unit_feature_schema=schema, field_feature_schema=fields, control_features=self.model_control_features,
            unit_features=unit.tolist(), field_features=field.tolist())

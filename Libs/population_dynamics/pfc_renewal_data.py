"""Source-bound PFC event support for the existing renewal observation model.

The pinned EPFL/Brian source has 2 ms E and 1 ms I refractory periods.
This adapter verifies archived source evidence before exposing observed events,
controls and eligible ticks. It does not qualify a trained population model.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np

from .circuit_data import CircuitDataset
from .pfc_scale_analysis import (
    SOURCE_ENTRY_SHA256, SOURCE_REVISION, _source_header, observed_source_registration, _source_values,
)
from .pfc_source_dataset import _benchmark_registration, _controls, _units
from .provenance import canonical_json, sha256_file
from .pfc_kernel_parameters import resolve_kernel_parameters
from .refractory_observation import RenewalWindow, renewal_window

MINIMUM_INTERVAL_POLICY = "pinned_epfl_brian_timestep_refractory_E_2ms_I_1ms"


def _local_admission_paths(data, row, source_run, source_entry):
    """Identify already declared files eligible for process-local admission reuse."""
    roots = (data.root, source_run, source_entry)
    if any(not Path(path).resolve().is_relative_to(Path("/tmp/neuronsim")) for path in roots):
        return None
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        mounts.append((Path(before.split()[4].replace("\\040", " ")), after.split()[0]))
    for root in roots:
        matching = [(len(mount.parts), filesystem) for mount, filesystem in mounts
                    if Path(root).resolve().is_relative_to(mount)]
        if not matching or max(matching)[1] not in {"ext4", "xfs", "btrfs", "overlay", "zfs"}:
            return None
    report = json.loads((source_run / "report.json").read_text())
    names = ["source_run/report.json", *["source_run/" + name for name in report["outputs"]],
             "qualification/delivery.json", "qualification/comparison/qualification.json"]
    names += ["Libs/population_dynamics/" + name for name in (
        "pfc_source_execution.py", "pfc_deterministic_replay.py", "pfc_native_drive.py")]
    if "kernel_parameterization_sha256" in report["code"]:
        names.append("Libs/population_dynamics/pfc_kernel_parameters.py")
    circuit = data.catalog["circuits"][row["circuit_id"]]
    shard = data.shards[row["shard"]]
    paths = [source_entry, source_run.parent / "delivery.json", data.root / "dataset.json",
             data.root / circuit["path"], data.root / shard["path"]]
    paths += [source_run.parent / name for name in names]
    if any(not Path(path).resolve().is_relative_to(Path("/tmp/neuronsim")) for path in paths):
        return None
    return tuple(dict.fromkeys(Path(path).resolve() for path in paths))


def _file_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_local_admission(data, row, record):
    if canonical_json(dict(catalog=data.catalog, episode=row)) != record["identity"]:
        raise ValueError("PFC_RENEWAL_ADMITTED_IDENTITY_CHANGED")
    for path, signature in record["files"].items():
        try:
            current = _file_signature(Path(path))
        except OSError as error:
            raise ValueError("PFC_RENEWAL_ADMITTED_FILE_CHANGED:" + path) from error
        if current != signature:
            raise ValueError("PFC_RENEWAL_ADMITTED_FILE_CHANGED:" + path)


def population_control_features(config, features):
    """Keep task geometry in source-indexed features instead of channel names.

    Raw archive control metadata remains unchanged. Historical models keep
    their original contract. Compiled tasks have two explicit target masks
    in the unit and field descriptors, so the four command channels can share
    one measurement contract across distractor locations.
    """
    if "benchmark_episode_candidate" not in config:
        return list(features)
    return [dict({key: value for key, value in feature.items()
                  if key not in ("center_degree", "width_degree")},
                 geometry="source_index_target_masks_in_population_metadata")
            if "center_degree" in feature else dict(feature) for feature in features]


def _source_target_mask(config, prefix, total):
    n_e = config["excitatory_neurons"]
    center = int(round(n_e / 360. * config[prefix + "_center_deg"]))
    half_width = int(round(n_e / 360. * config[prefix + "_width_deg"] / 2.))
    indices = (center + np.arange(-half_width, half_width + 1)) % n_e
    mask = np.zeros(total, dtype=np.float64)
    mask[indices] = 1
    return mask


class PfcRenewalEpisode:
    """Observed-event adapter for selected archived PFC source episodes.

    Source admission verifies initial and terminal states. Those states are
    verification evidence and never enter the returned model observations.
    Initial eligibility follows the source's all-ready initialization.
    """

    def __init__(self, data: CircuitDataset, episode_id: str, *, bin_ticks: int,
                 source_run: Path, source_entry: Path):
        row = data.episodes[episode_id]  # Reject an unselected split before source access.
        binding = data.catalog["binding"]
        if (binding.get("scenario_id") != "pfc_working_memory"
                or binding.get("scope") != "source_reproduction"
                or binding.get("source_revision") != SOURCE_REVISION
                or binding.get("source_entry_sha256") != SOURCE_ENTRY_SHA256):
            raise ValueError("PFC_RENEWAL_PINNED_SOURCE_ARCHIVE_REQUIRED")
        if (type(bin_ticks) is not int or bin_ticks < 1 or row["duration_ticks"] % bin_ticks):
            raise ValueError("PFC_RENEWAL_EXACT_COMPLETE_CLOCK_REQUIRED")
        source_run = Path(source_run)
        source_run = source_run if source_run.name == "source_run" else source_run / "source_run"
        source_run, source_entry = source_run.resolve(), Path(source_entry).resolve()
        key = (episode_id, str(source_run), str(source_entry), bin_ticks)
        admissions = getattr(data, "pfc_source_admissions", {})
        if key in admissions:
            self._load_admitted(data, row, admissions[key], bin_ticks)
            return
        local_paths = _local_admission_paths(data, row, source_run, source_entry)
        signatures = ({str(path): _file_signature(path) for path in local_paths}
                      if local_paths is not None else None)
        if sha256_file(Path(source_entry)) != SOURCE_ENTRY_SHA256:
            raise ValueError("PFC_RENEWAL_SOURCE_ENTRY_CHANGED")
        source_config = data.catalog["binding"]["resolved_source_config"]
        if "benchmark_episode_candidate" in source_config:
            config, registered, _, dimensions = _benchmark_registration(source_config)
        else:
            config, registered, _, dimensions = observed_source_registration(source_run / "config.yaml")
        config_sha = sha256_file(source_run / "config.yaml")
        status, header = _source_header(source_run, config, registered, config_sha)
        if header is None or status["status"] != "COMPLETE" or status["case_id"] != episode_id:
            raise ValueError("PFC_RENEWAL_COMPLETE_MATCHING_SOURCE_REQUIRED")
        if (binding.get("resolved_source_config") != config or binding.get("config_sha256") != config_sha
                or not any(reference.get("case_id") == episode_id
                    and reference.get("report_sha256") == status["report_sha256"]
                    and reference.get("delivery_sha256") == status["delivery_sha256"]
                    for reference in binding.get("source_bindings", []))):
            raise ValueError("PFC_RENEWAL_DATASET_SOURCE_BINDING_CHANGED")
        raw, _, _ = _source_values(header, config, registered, dimensions, config_sha)
        expected_units = _units(config, header, dimensions)
        source_units = data.circuit_metadata(row["circuit_id"])
        for name, expected in (("unit_ids", expected_units.unit_ids),
                               ("unit_field_index", expected_units.field_index),
                               ("populations", expected_units.populations)):
            if not np.array_equal(source_units[name], expected):
                raise ValueError("PFC_RENEWAL_SOURCE_UNIT_MAP_CHANGED:" + name)
        if (row["circuit_id"] != expected_units.circuit_id
                or source_units["field_names"] != expected_units.field_names
                or source_units["circuit_provenance"] != expected_units.circuit_provenance
                or source_units["source_revision"] != SOURCE_REVISION
                or source_units["coordinate_semantics"] != expected_units.coordinate_semantics
                or source_units["coordinates"] is not None
                or row["dt_ms"] != config["dt_ms"]
                or row["duration_ticks"] != dimensions["duration_ticks"]):
            raise ValueError("PFC_RENEWAL_SOURCE_MAP_OR_CLOCK_CHANGED")
        report = header["report"]
        seeds = report["seeds"]
        expected_block = ("pfc_source_block_" + str(seeds["source_rng_seed"])
                          if report["random_stream_policy"] == "source_global" else
                          "pfc_initial_" + str(seeds["initial_rng_seed"]) + "_external_" + str(seeds["external_noise_seed"]))
        if row["block_id"] != expected_block:
            raise ValueError("PFC_RENEWAL_SOURCE_BLOCK_IDENTITY_CHANGED")
        observed_path = header["source_run"] / "applied_current_observation.npz"
        with np.load(observed_path, allow_pickle=False) as observed:
            control_ticks, control_values, features = _controls(config, report["selected_case"], observed)
        shard = data.shards[row["shard"]]
        with h5py.File(data._verified(shard["path"], shard["sha256"]), "r") as stored:
            group = stored[episode_id]
            for name, expected in (("spike_ticks", raw["spike_ticks"]),
                                   ("spike_unit_indices", raw["spike_unit_indices"]),
                                   ("control_ticks", control_ticks), ("control_values", control_values)):
                if not np.array_equal(group[name][:], expected):
                    raise ValueError("PFC_RENEWAL_SOURCE_EVENTS_OR_CONTROLS_CHANGED:" + name)
        # The shared reader checks episode/catalog identities and the all-unit count.
        first = data._read(episode_id, 0, bin_ticks, bin_ticks, 0, len(expected_units.unit_ids))
        if first["control_features"] != list(features) or first["control_kind"] != "piecewise_constant":
            raise ValueError("PFC_RENEWAL_SOURCE_CONTROL_SEMANTICS_CHANGED")
        self.data, self.episode_id = data, episode_id
        self.source_config = config
        self.model_control_features = population_control_features(config, features)
        self.unit_ids = expected_units.unit_ids
        self.unit_field_index, self.field_names = expected_units.field_index, expected_units.field_names
        self.event_ticks, self.event_units = raw["spike_ticks"], raw["spike_unit_indices"]
        self.bin_ticks, self.duration_ticks, self.dt_ms = bin_ticks, row["duration_ticks"], row["dt_ms"]
        dt = self.dt_ms / 1000
        # Reuse the source integer-step expression. No Mesocircuit +1 correction.
        refractory_seconds = np.where(expected_units.populations == "E", .002, .001)
        self.minimum_interval_ticks = np.asarray((refractory_seconds + 1e-3 * dt) / dt, dtype=np.int64)
        if np.any(self.minimum_interval_ticks < 1):
            raise ValueError("PFC_RENEWAL_REFRACTORY_CLOCK_UNSUPPORTED")
        self.binding = dict(dataset_sha256=sha256_file(data.root / "dataset.json"),
                            episode_id=episode_id, circuit_id=row["circuit_id"], block_id=row["block_id"],
                            split=row["split"], source_revision=SOURCE_REVISION,
                            source_entry_sha256=SOURCE_ENTRY_SHA256, source_report_sha256=status["report_sha256"],
                            source_delivery_sha256=status["delivery_sha256"],
                            minimum_interval_policy=MINIMUM_INTERVAL_POLICY,
                            initial_availability="source_all_ready; no unobserved prior spikes",
                            bin_ticks=bin_ticks, source_dt_ms=self.dt_ms)
        self._admission_record = None
        if signatures is not None:
            record = dict(identity=canonical_json(dict(catalog=data.catalog, episode=row)),
                files=signatures, source_config=deepcopy(config),
                model_control_features=deepcopy(self.model_control_features), binding=deepcopy(self.binding))
            _check_local_admission(data, row, record)
            # Records retain identities and file signatures, never episode arrays.
            if not hasattr(data, "pfc_source_admissions"):
                data.pfc_source_admissions = {}
            data.pfc_source_admissions[key] = record
            self._admission_record = record

    def _load_admitted(self, data, row, record, bin_ticks):
        """Reload only active events after rejecting changed local evidence."""
        _check_local_admission(data, row, record)
        source = data.circuit_metadata(row["circuit_id"])
        shard = data.shards[row["shard"]]
        with h5py.File(data._verified(shard["path"], shard["sha256"]), "r") as stored:
            group = stored[row["episode_id"]]
            self.event_ticks = group["spike_ticks"][:]
            self.event_units = group["spike_unit_indices"][:]
        _check_local_admission(data, row, record)
        self.data, self.episode_id = data, row["episode_id"]
        self.source_config = deepcopy(record["source_config"])
        self.model_control_features = deepcopy(record["model_control_features"])
        self.unit_ids, self.unit_field_index = source["unit_ids"], source["unit_field_index"]
        self.field_names = source["field_names"]
        self.bin_ticks, self.duration_ticks, self.dt_ms = bin_ticks, row["duration_ticks"], row["dt_ms"]
        refractory_seconds = np.where(source["populations"] == "E", .002, .001)
        dt = self.dt_ms / 1000
        self.minimum_interval_ticks = np.asarray((refractory_seconds + 1e-3 * dt) / dt, dtype=np.int64)
        self.binding = deepcopy(record["binding"])
        self._admission_record = record

    def population_metadata(self):
        """Static PFC descriptors for shared parameters across source maps.

        Preferred angles come from the verified source E index. The I angle
        slots have explicit zero availability. No anatomical position is inferred.
        Intrinsic transforms are declared; no episode outcome fits these features.
        """
        config = self.source_config
        n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
        excitatory = np.arange(len(self.unit_ids)) < n_e
        angle = np.zeros(len(self.unit_ids))
        angle[:n_e] = 2 * np.pi * np.arange(n_e) / n_e
        sine, cosine = np.where(excitatory, np.sin(angle), 0), np.where(excitatory, np.cos(angle), 0)
        kernel = resolve_kernel_parameters(config)
        circuit = np.asarray([np.log1p(n_e), np.log1p(n_i), np.log1p(config["weight_scaling_factor"]),
                              kernel["sigma_weight_profile"] / 180, kernel["Jpos_excit2excit"]])
        values = np.column_stack((excitatory, ~excitatory, sine, cosine, excitatory))
        task = "benchmark_episode_candidate" in config
        if task:
            targets = np.column_stack([_source_target_mask(config, prefix, len(self.unit_ids))
                                       for prefix in ("stimulus", "distractor")])
            values = np.column_stack((values, targets))
        unit = np.column_stack((values, np.broadcast_to(circuit, (len(self.unit_ids), len(circuit)))))
        sizes = np.bincount(self.unit_field_index, minlength=len(self.field_names))
        field = np.zeros((len(self.field_names), values.shape[1] + 1))
        for feature in range(values.shape[1]):
            np.add.at(field[:, feature], self.unit_field_index, values[:, feature])
        field[:, :-1] /= np.maximum(sizes[:, None], 1)
        field[:, -1] = np.log1p(sizes)
        field = np.column_stack((field, np.broadcast_to(circuit, (len(field), len(circuit)))))
        def descriptor(name, unit, meaning, transform="identity"):
            return dict(name=name, unit=unit, meaning=meaning, transform=transform)
        common = [descriptor("source_E_size", "neurons", "verified E population size", "log1p"),
                  descriptor("source_I_size", "neurons", "verified I population size", "log1p"),
                  descriptor("synaptic_weight_scale", "dimensionless", "source synaptic scaling factor", "log1p"),
                  descriptor("kernel_sigma", "degree", "source E-to-E Gaussian width", "divide_by_180"),
                  descriptor("kernel_Jpos", "dimensionless", "source E-to-E peak kernel weight")]
        unit_targets = ([descriptor("cue_target", "dimensionless", "membership in source cue assignment indices"),
                         descriptor("distractor_target", "dimensionless", "membership in source distractor overwrite indices")]
                        if task else [])
        field_targets = ([descriptor("cue_target_fraction", "dimensionless", "fraction in source cue assignment indices"),
                          descriptor("distractor_target_fraction", "dimensionless", "fraction in source distractor overwrite indices")]
                         if task else [])
        source = self.data.circuit_metadata(self.binding["circuit_id"])
        return dict(source_identity=dict(circuit_id=self.binding["circuit_id"],
            source_revision=source["source_revision"], **source["circuit_provenance"]), unit_feature_schema=[
            descriptor("is_E", "dimensionless", "source E population membership"),
            descriptor("is_I", "dimensionless", "source I population membership"),
            descriptor("preference_sin", "dimensionless", "source E preferred angle sine; zero when unavailable"),
            descriptor("preference_cos", "dimensionless", "source E preferred angle cosine; zero when unavailable"),
            descriptor("preference_available", "dimensionless", "E source preference available; I unavailable"),
            *unit_targets, *common],
            field_feature_schema=[
            descriptor("E_fraction", "dimensionless", "source field E fraction"),
            descriptor("I_fraction", "dimensionless", "source field I fraction"),
            descriptor("preference_sin_mean", "dimensionless", "all-unit mean of masked source preference sine"),
            descriptor("preference_cos_mean", "dimensionless", "all-unit mean of masked source preference cosine"),
            descriptor("preference_available_fraction", "dimensionless", "fraction with source preference"),
            *field_targets,
            descriptor("field_size", "neurons", "actual source field unit count", "log1p"), *common],
            control_features=self.model_control_features,
            unit_features=unit.tolist(), field_features=field.tolist())

    def window(self, start_tick: int, stop_tick: int):
        if (type(start_tick) is not int or type(stop_tick) is not int
                or not 0 <= start_tick < stop_tick <= self.duration_ticks
                or start_tick % self.bin_ticks or stop_tick % self.bin_ticks):
            raise ValueError("PFC_RENEWAL_COMPLETE_ALIGNED_WINDOW_REQUIRED")
        if self._admission_record is not None:
            _check_local_admission(self.data, self.data.episodes[self.episode_id], self._admission_record)
        value = self.data._read(self.episode_id, start_tick, stop_tick, self.bin_ticks,
                                0, len(self.unit_ids))
        return value["counts"], value["known_controls"]

    def renewal_window(self, start_tick: int, stop_tick: int) -> RenewalWindow:
        return renewal_window(self, start_tick, stop_tick, self.minimum_interval_ticks)

    def ready_at(self, tick: int) -> np.ndarray:
        """Compute the next emission availability from the observed prefix only."""
        if type(tick) is not int or not 0 <= tick <= self.duration_ticks:
            raise ValueError("PFC_RENEWAL_CAUSAL_BOUNDARY_REQUIRED")
        stop = np.searchsorted(self.event_ticks, tick, side="left")
        ticks, units = self.event_ticks[:stop], self.event_units[:stop]
        order = np.lexsort((ticks, units))
        same = np.diff(units[order]) == 0
        if np.any(np.diff(ticks[order])[same] < self.minimum_interval_ticks[units[order][1:][same]]):
            raise ValueError("observed spike train violates the declared minimum interval")
        last = np.full(len(self.unit_ids), -int(self.minimum_interval_ticks.max()), dtype=np.int64)
        np.maximum.at(last, units, ticks)
        return np.maximum(last + self.minimum_interval_ticks - tick, 0)

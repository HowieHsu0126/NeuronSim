"""Complete source-bound PFC episodes for shared population development fitting.

This consumer leaves archive splits unchanged. A declared whole-circuit
development fit/validation partition is separate from formal campaign splits.
Source-reproduction archives cannot establish production or formal qualification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .circuit_data import CircuitDataset
from .pfc_renewal_data import PfcRenewalEpisode
from .pfc_scale_analysis import SOURCE_ENTRY_SHA256
from .provenance import sha256_file
from .source_population_training import SourcePopulationTrainingCohort


def validate_config(config, seed):
    frozen = dict(schema="pfc_population_development_training_v1", model_family="population",
        training_seeds=[41, 42, 43], input_family="pfc_source_reproduction",
        training_sequence="complete_source_episode", observation_objective="native_event_time_mc_negative_elbo",
        partition_scope="whole_circuit_development_only", formal_training_completed=False,
        validation_randomness="fixed_per_training_seed_case_and_draw",
        variational_rng="explicit_separate_checkpointed_generator")
    mode=config.get("control_geometry","static")
    if mode not in ("static","actuation_only"):
        raise ValueError("PFC_TRAINING_REGISTERED_CONTROL_GEOMETRY_REQUIRED")
    if mode=="actuation_only":
        frozen["validation_randomness"]="fixed_per_training_seed_complete_validation_sequence"
        if (config.get("random_seed_policy")!="exact_registered_stream_seeds"
                or config.get("training_variational_seed_base")!=41
                or config.get("validation_seed_base")!=42):
            raise ValueError("PFC_TRAINING_ACTUATION_SEEDS_MUST_BE_41_42_43")
    if any(config.get(key) != value for key, value in frozen.items()):
        raise ValueError("PFC_SHARED_DEVELOPMENT_TRAINING_PROTOCOL_REQUIRED")
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("REGISTERED_TRAINING_SEED_REQUIRED")
    for name in ("fit_circuit_ids", "validation_circuit_ids"):
        values = config.get(name)
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)):
            raise ValueError("EXPLICIT_NONEMPTY_WHOLE_CIRCUIT_PARTITION_REQUIRED")
    if set(config["fit_circuit_ids"]) & set(config["validation_circuit_ids"]):
        raise ValueError("PFC_TRAINING_WHOLE_CIRCUIT_SPLIT_LEAKAGE")
    for name in ("latent_channels", "encoder_size", "unit_chunk_size", "bin_ticks",
                 "maximum_epochs", "early_stopping_patience", "validation_draws", "cpu_threads"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError("PFC_TRAINING_POSITIVE_DIMENSION_OR_BUDGET_REQUIRED:" + name)
    if config["cpu_threads"] > 2 or config.get("representation") not in ("fixed", "recurrent"):
        raise ValueError("PFC_TRAINING_REGISTERED_REPRESENTATION_AND_CPU_THREADS_REQUIRED")
    for name in ("learning_rate", "initialization_pseudocount", "gradient_norm_limit"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("PFC_TRAINING_FINITE_POSITIVE_SETTING_REQUIRED:" + name)
    for name in ("training_variational_seed_base", "validation_seed_base"):
        if type(config.get(name)) is not int or not 0 <= config[name] < 2**63:
            raise ValueError("PFC_TRAINING_EXPLICIT_RANDOM_SEED_BASE_REQUIRED")
    if torch.device(config["device"]).type not in ("cpu", "cuda"):
        raise ValueError("PFC_TRAINING_SUPPORTED_DEVICE_REQUIRED")


def _declared_path(value, inputs_path, *, source_run=False):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (inputs_path.parent / path).resolve()
    return path / "source_run" if source_run and path.name != "source_run" else path


def _unique_map_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("PFC_EXECUTION_PATH_MAP_DUPLICATE_KEY")
        result[key] = value
    return result


def _execution_paths(inputs_path, spec, config):
    """Resolve bound local copies without changing scientific input identities.

    Map entries bind dataset.json, archive delivery.json, or the pinned source
    file. The existing adapters still authenticate every consumed source file.
    """
    map_file = os.environ.get("NEURONSIM_EXECUTION_PATH_MAP")
    if map_file is None:
        return {}
    try:
        value = json.loads(Path(map_file).read_text(), object_pairs_hook=_unique_map_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("PFC_EXECUTION_PATH_MAP_UNREADABLE") from error
    if (not isinstance(value, dict)
            or set(value) != {"schema", "inputs_sha256", "paths"}
            or value["schema"] != "pfc_population_execution_path_map_v1"
            or not isinstance(value["paths"], dict)):
        raise ValueError("PFC_EXECUTION_PATH_MAP_SCHEMA_REQUIRED")
    if value["inputs_sha256"] != sha256_file(inputs_path):
        raise ValueError("PFC_EXECUTION_PATH_MAP_INPUT_CHANGED")
    source_entry = _declared_path(spec["source_entry"], inputs_path)
    kinds = {source_entry: "source_entry"}
    for row in spec["episodes"]:
        for key, kind in (("dataset", "dataset"), ("source_run", "source_run")):
            original = _declared_path(row[key], inputs_path, source_run=key == "source_run")
            if original in kinds and kinds[original] != kind:
                raise ValueError("PFC_EXECUTION_PATH_MAP_DECLARED_IDENTITIES_REQUIRED")
            kinds[original] = kind
    declared = {str(path) for path in kinds}
    if set(value["paths"]) != declared:
        raise ValueError("PFC_EXECUTION_PATH_MAP_DECLARED_IDENTITIES_REQUIRED")
    # Longest containing mount identifies the actual target filesystem.
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        mount = before.split()[4].replace("\\040", " ")
        mounts.append((Path(mount), after.split()[0]))
    allowed_filesystems = {"ext4", "xfs", "btrfs", "tmpfs", "overlay", "ramfs", "zfs"}
    mapped, digests, targets = {}, {}, set()
    for original, kind in kinds.items():
        row = value["paths"][str(original)]
        if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                or not isinstance(row["path"], str) or not row["path"]
                or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in row["sha256"])):
            raise ValueError("PFC_EXECUTION_PATH_MAP_SCHEMA_REQUIRED")
        target = Path(row["path"])
        if not target.is_absolute() or ".." in target.parts:
            raise ValueError("PFC_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
        try:
            target = target.resolve(strict=True)
        except OSError as error:
            raise ValueError("PFC_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED") from error
        matching = [(len(mount.parts), filesystem) for mount, filesystem in mounts
                    if target.is_relative_to(mount)]
        if (not target.is_relative_to(Path("/tmp/neuronsim"))
                or target == Path("/tmp/neuronsim") or target in targets or not matching
                or max(matching)[1] not in allowed_filesystems):
            raise ValueError("PFC_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
        if ((kind == "source_entry" and not target.is_file())
                or (kind != "source_entry" and not target.is_dir())
                or (kind == "source_run" and target.name != "source_run")):
            raise ValueError("PFC_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
        metadata = (target / "dataset.json" if kind == "dataset" else
                    target.parent / "delivery.json" if kind == "source_run" else target)
        if not metadata.is_file() or sha256_file(metadata) != row["sha256"]:
            raise ValueError("PFC_EXECUTION_PATH_MAP_METADATA_CHANGED")
        mapped[original], digests[original] = target, row["sha256"]
        targets.add(target)
    if digests[source_entry] != SOURCE_ENTRY_SHA256:
        raise ValueError("PFC_TRAINING_PINNED_SOURCE_ENTRY_CHANGED")
    # Inspect all declared identities before the first adapter can read arrays.
    catalogs = {}
    for row in spec["episodes"]:
        original_dataset = _declared_path(row["dataset"], inputs_path)
        original_source = _declared_path(row["source_run"], inputs_path, source_run=True)
        if original_dataset not in catalogs:
            catalogs[original_dataset] = json.loads((mapped[original_dataset] / "dataset.json").read_text())
        catalog = catalogs[original_dataset]
        binding = catalog["binding"]
        episodes = [episode for shard in catalog["shards"] for episode in shard["episodes"]
                    if episode["episode_id"] == row["episode_id"]]
        candidate = binding["resolved_source_config"].get("benchmark_episode_candidate")
        intended = "train" if row["partition"] == "development_fit" else "validation"
        selected = config["fit_circuit_ids"] if intended == "train" else config["validation_circuit_ids"]
        if (len(episodes) != 1 or episodes[0]["split"] != "development"
                or episodes[0]["circuit_id"] not in selected
                or (candidate is not None and candidate.get("intended_split") != intended)):
            raise ValueError("PFC_EXECUTION_PATH_MAP_PROSPECTIVE_IDENTITY_CHANGED")
        source = mapped[original_source]
        delivery = json.loads((source.parent / "delivery.json").read_text())
        references = [reference for reference in binding.get("source_bindings", [])
                      if reference.get("case_id") == row["episode_id"]]
        if (len(references) != 1
                or _declared_path(references[0]["path"], inputs_path, source_run=True) != original_source
                or references[0]["delivery_sha256"] != digests[original_source]
                or delivery.get("case_id") != row["episode_id"]
                or delivery.get("status") != "ARCHIVED_PFC_SOURCE_EXECUTION"
                or delivery.get("result_status") != "PFC_SOURCE_EXECUTION_COMPLETE"
                or delivery.get("source_execution_complete") is not True):
            raise ValueError("PFC_EXECUTION_PATH_MAP_SOURCE_IDENTITY_CHANGED")
        files = [item for item in delivery["files"] if item["path"] == "source_run/report.json"]
        report_path = source / "report.json"
        if (len(files) != 1 or report_path.stat().st_size != files[0]["bytes"]
                or sha256_file(report_path) != files[0]["sha256"]
                or files[0]["sha256"] != references[0]["report_sha256"]):
            raise ValueError("PFC_EXECUTION_PATH_MAP_METADATA_CHANGED")
        report = json.loads(report_path.read_text())
        if (report.get("case_id") != row["episode_id"]
                or report.get("resolved_config") != binding["resolved_source_config"]
                or report.get("config_sha256") != binding["config_sha256"]
                or sha256_file(source / "config.yaml") != binding["config_sha256"]):
            raise ValueError("PFC_EXECUTION_PATH_MAP_SOURCE_IDENTITY_CHANGED")
    return mapped


class PfcPopulationTrainingCohort(SourcePopulationTrainingCohort):
    """Read only declared development episodes, one complete episode at a time."""

    def __init__(self, inputs_path: Path, config: dict):
        validate_config(config, 41)
        path = Path(inputs_path)
        spec = json.loads(path.read_text())
        if (set(spec) != {"schema", "source_entry", "episodes"}
                or spec["schema"] != "pfc_population_development_inputs_v1"
                or not isinstance(spec["episodes"], list) or not spec["episodes"]):
            raise ValueError("DECLARED_PFC_DEVELOPMENT_INPUTS_REQUIRED")
        for row in spec["episodes"]:
            if (not isinstance(row, dict)
                    or set(row) != {"dataset", "episode_id", "source_run", "partition"}
                    or row["partition"] not in ("development_fit", "development_validation")):
                raise ValueError("PFC_TRAINING_EXPLICIT_DEVELOPMENT_PARTITION_REQUIRED")
        execution_paths = _execution_paths(path, spec, config)
        def resolve(value, *, source_run=False):
            original = _declared_path(value, path, source_run=source_run)
            return execution_paths.get(original, original)
        self.source_entry = resolve(spec["source_entry"])
        if sha256_file(self.source_entry) != SOURCE_ENTRY_SHA256:
            raise ValueError("PFC_TRAINING_PINNED_SOURCE_ENTRY_CHANGED")
        self.bin_ticks = config["bin_ticks"]
        self.data, self.fit, self.validation = {}, [], []
        records, identities = [], set()
        fit_ids, validation_ids = set(config["fit_circuit_ids"]), set(config["validation_circuit_ids"])
        selected = {"development_fit": fit_ids, "development_validation": validation_ids}
        self.config = config
        contract = None
        for row in spec["episodes"]:
            if (set(row) != {"dataset", "episode_id", "source_run", "partition"}
                    or row["partition"] not in selected):
                raise ValueError("PFC_TRAINING_EXPLICIT_DEVELOPMENT_PARTITION_REQUIRED")
            root = resolve(row["dataset"])
            if root not in self.data:
                self.data[root] = CircuitDataset(root, splits=("development",))
                if (self.data[root].catalog["binding"].get("scope") != "source_reproduction"
                        or self.data[root].catalog["binding"].get("scenario_id") != "pfc_working_memory"):
                    raise ValueError("PFC_TRAINING_SOURCE_REPRODUCTION_ARCHIVE_REQUIRED")
            data = self.data[root]
            episode = data.episodes[row["episode_id"]]  # Protected splits never expose source files.
            candidate = data.catalog["binding"]["resolved_source_config"].get("benchmark_episode_candidate")
            if candidate is not None:
                intended = "train" if row["partition"] == "development_fit" else "validation"
                if candidate.get("intended_split") != intended:
                    raise ValueError("PFC_TRAINING_PROSPECTIVE_CIRCUIT_SPLIT_CHANGED")
            circuit = episode["circuit_id"]
            if circuit not in selected[row["partition"]]:
                raise ValueError("PFC_TRAINING_WHOLE_CIRCUIT_PARTITION_CHANGED")
            identity = (circuit, row["episode_id"])
            if identity in identities:
                raise ValueError("PFC_TRAINING_DUPLICATE_COMPLETE_EPISODE")
            identities.add(identity)
            if episode["duration_ticks"] % self.bin_ticks:
                raise ValueError("PFC_TRAINING_COMPLETE_EPISODE_CLOCK_REQUIRED")
            source = resolve(row["source_run"], source_run=True)
            adapter = PfcRenewalEpisode(data, row["episode_id"], bin_ticks=self.bin_ticks,
                                       source_run=source, source_entry=self.source_entry)
            metadata = adapter.population_metadata()
            current_contract = (adapter.dt_ms, metadata["unit_feature_schema"],
                                metadata["field_feature_schema"], metadata["control_features"])
            if contract is not None and contract != current_contract:
                raise ValueError("PFC_TRAINING_COMMON_FEATURE_CONTROL_CLOCK_CONTRACT_REQUIRED")
            contract = current_contract
            case = dict(case_id=circuit + "__" + row["episode_id"], episode_id=row["episode_id"],
                        circuit_id=circuit, dataset=root, source_run=source,
                        source_binding=adapter.binding)
            if candidate is not None:
                case["source_target_geometry"] = tuple(adapter.source_config[name]
                    for name in ("stimulus_center_deg", "stimulus_width_deg",
                                 "distractor_center_deg", "distractor_width_deg"))
            (self.fit if row["partition"] == "development_fit" else self.validation).append(case)
            records.append(dict(partition=row["partition"], **adapter.binding))
        if ({c["circuit_id"] for c in self.fit} != fit_ids
                or {c["circuit_id"] for c in self.validation} != validation_ids):
            raise ValueError("PFC_TRAINING_DECLARED_CIRCUIT_DENOMINATOR_INCOMPLETE")
        self.binding = dict(circuit_inputs_sha256=sha256_file(path), pfc_source_entry_sha256=SOURCE_ENTRY_SHA256,
                            complete_episode_bindings=records, partition_unit="whole_source_circuit",
                            partition_scope="declared_development_only; original_archive_splits_preserved")
        self.unit_ids = self.data[self.fit[0]["dataset"]].circuit_metadata(self.fit[0]["circuit_id"])["unit_ids"]
        self.active_case, self.active_episode = None, None

    def _episode(self, case):
        episode = PfcRenewalEpisode(self.data[case["dataset"]], case["episode_id"], bin_ticks=self.bin_ticks,
            source_run=case["source_run"], source_entry=self.source_entry)
        if episode.binding != case["source_binding"]:
            raise ValueError("PFC_TRAINING_SOURCE_EPISODE_BINDING_CHANGED")
        return episode

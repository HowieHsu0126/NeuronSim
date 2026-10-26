"""Complete V1 episodes with original whole-circuit train/validation membership.

The catalog freezes the selected episode set. Partial collection is reported as
such; this development trainer does not establish the 100000-episode benchmark.
Missing validation rejects before parameter or source-array access.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .circuit_data import CircuitDataset
from .provenance import canonical_json, sha256_file
from .source_population_training import SourcePopulationTrainingCohort
from .v1_renewal_data import V1RenewalEpisode

SCHEMA = "v1_population_development_training_v1"
INPUT_SCHEMA = "v1_population_training_inputs_v1"


def validate_config(config: dict, seed: int):
    frozen = dict(schema=SCHEMA, model_family="population", training_seeds=[41,42,43],
        input_family="v1_qualified_source_episodes", training_sequence="complete_source_episode",
        observation_objective="native_event_time_mc_negative_elbo", partition_scope="original_whole_circuit_splits",
        formal_training_completed=False, validation_randomness="fixed_per_training_seed_complete_validation_sequence",
        variational_rng="explicit_separate_checkpointed_generator", random_seed_policy="exact_registered_stream_seeds",
        training_variational_seed_base=41, validation_seed_base=42)
    if any(config.get(key) != value for key,value in frozen.items()):
        raise ValueError("V1_POPULATION_REGISTERED_DEVELOPMENT_PROTOCOL_REQUIRED")
    if type(seed) is not int or seed not in (41,42,43):
        raise ValueError("REGISTERED_TRAINING_SEED_REQUIRED")
    for name in ("fit_circuit_ids","validation_circuit_ids"):
        ids=config.get(name)
        if (not isinstance(ids,list) or not ids or any(not isinstance(x,str) or not x for x in ids)
                or len(ids)!=len(set(ids))):
            raise ValueError("EXPLICIT_NONEMPTY_WHOLE_CIRCUIT_PARTITION_REQUIRED")
    if set(config["fit_circuit_ids"]) & set(config["validation_circuit_ids"]):
        raise ValueError("V1_TRAINING_WHOLE_CIRCUIT_SPLIT_LEAKAGE")
    for name in ("latent_channels","encoder_size","unit_chunk_size","bin_ticks","maximum_epochs",
                 "early_stopping_patience","validation_draws","cpu_threads"):
        if type(config.get(name)) is not int or config[name]<1:
            raise ValueError("V1_TRAINING_POSITIVE_DIMENSION_OR_BUDGET_REQUIRED:"+name)
    if (config["cpu_threads"]>2 or config.get("representation") not in ("fixed","recurrent")
            or config.get("control_geometry","static")!="static"):
        raise ValueError("V1_TRAINING_SOURCE_REPRESENTATION_REQUIRED")
    for name in ("learning_rate","initialization_pseudocount","gradient_norm_limit"):
        value=config.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError("V1_TRAINING_FINITE_POSITIVE_SETTING_REQUIRED:"+name)
    if torch.device(config["device"]).type not in ("cpu","cuda"):
        raise ValueError("V1_TRAINING_SUPPORTED_DEVICE_REQUIRED")


@dataclass(frozen=True)
class V1InputInspection:
    raw: bytes
    spec: dict
    data: CircuitDataset
    parameters: dict[str, Path]
    expected_parameter_sha256: dict[str, str]


def _unique_execution_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("V1_EXECUTION_PATH_MAP_DUPLICATE_KEY")
        result[key] = value
    return result


def _execution_target(row, *, directory):
    if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str) or not row["path"]
            or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])):
        raise ValueError("V1_EXECUTION_PATH_MAP_SCHEMA_REQUIRED")
    target = Path(row["path"])
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("V1_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
    try:
        target = target.resolve(strict=True)
    except OSError as error:
        raise ValueError("V1_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED") from error
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        mount = Path(before.split()[4].replace("\\040", " "))
        if target.is_relative_to(mount):
            mounts.append((len(mount.parts), after.split()[0]))
    allowed = {"ext4", "xfs", "btrfs", "tmpfs", "overlay", "ramfs", "zfs"}
    if (not target.is_relative_to(Path("/tmp/neuronsim")) or target == Path("/tmp/neuronsim")
            or not mounts or max(mounts)[1] not in allowed
            or (not target.is_dir() if directory else not target.is_file())):
        raise ValueError("V1_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
    return target


def inspect_inputs(inputs_path: Path, config: dict, *,
                   config_validator: Callable[[dict, int], None] | None = None) -> V1InputInspection:
    """Admit original splits using metadata, before parameter bytes or source arrays.

    Execution maps relocate declared roles, not the unchanged input strings.
    Preparation supplies their digests from its frozen inventory; the cohort
    still verifies every parameter and source payload when consumed.
    """
    (validate_config if config_validator is None else config_validator)(config, 41)
    path = Path(inputs_path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    if (not isinstance(spec, dict) or set(spec) != {"schema", "dataset", "native_parameters"}
            or spec["schema"] != INPUT_SCHEMA
            or not isinstance(spec["dataset"], str) or not spec["dataset"]
            or not isinstance(spec["native_parameters"], dict)):
        raise ValueError("DECLARED_V1_POPULATION_INPUTS_REQUIRED")

    def resolve(value):
        item = Path(value)
        return item.resolve() if item.is_absolute() else (path.parent / item).resolve()

    execution = None
    map_file = os.environ.get("NEURONSIM_EXECUTION_PATH_MAP")
    if map_file is not None:
        try:
            execution = json.loads(Path(map_file).read_text(), object_pairs_hook=_unique_execution_object)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V1_EXECUTION_PATH_MAP_UNREADABLE") from error
        if (not isinstance(execution, dict)
                or set(execution) != {"schema", "inputs_sha256", "dataset", "native_parameters"}
                or execution["schema"] != "v1_population_execution_path_map_v1"
                or not isinstance(execution["native_parameters"], dict)):
            raise ValueError("V1_EXECUTION_PATH_MAP_SCHEMA_REQUIRED")
        if execution["inputs_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("V1_EXECUTION_PATH_MAP_INPUT_CHANGED")
        dataset = _execution_target(execution["dataset"], directory=True)
        metadata = dataset / "dataset.json"
        if not metadata.is_file() or sha256_file(metadata) != execution["dataset"]["sha256"]:
            raise ValueError("V1_EXECUTION_PATH_MAP_DATASET_METADATA_CHANGED")
    else:
        dataset = resolve(spec["dataset"])

    data = CircuitDataset(dataset, splits=("train", "validation"))
    binding = data.catalog["binding"]
    if binding.get("scope") != "production" or binding.get("scenario_id") != "v1_visual":
        raise ValueError("V1_TRAINING_QUALIFIED_SOURCE_ARCHIVE_REQUIRED")
    fit_ids, validation_ids = set(config["fit_circuit_ids"]), set(config["validation_circuit_ids"])
    available = {split: {row["circuit_id"] for row in data.episodes.values() if row["split"] == split}
                 for split in ("train", "validation")}
    for split, declared in (("validation", validation_ids), ("train", fit_ids)):
        if not declared <= available[split]:
            raise ValueError("V1_TRAINING_ORIGINAL_" + split.upper() + "_CIRCUITS_MISSING_OR_MISASSIGNED")
    selected = fit_ids | validation_ids
    if (set(spec["native_parameters"]) != selected
            or any(not isinstance(value, str) or not value for value in spec["native_parameters"].values())):
        raise ValueError("V1_TRAINING_EXACT_SELECTED_PARAMETER_MAP_REQUIRED")
    expected = {}
    if execution is None:
        parameters = {cid: resolve(value) for cid, value in spec["native_parameters"].items()}
    else:
        if set(execution["native_parameters"]) != selected:
            raise ValueError("V1_EXECUTION_PATH_MAP_EXACT_SELECTED_PARAMETERS_REQUIRED")
        parameters = {cid: _execution_target(row, directory=False)
                      for cid, row in execution["native_parameters"].items()}
        expected = {cid: row["sha256"] for cid, row in execution["native_parameters"].items()}
    return V1InputInspection(raw, spec, data, parameters, expected)


class V1PopulationTrainingCohort(SourcePopulationTrainingCohort):
    """Freeze metadata first; admit each full source episode when consumed."""

    def __init__(self, inputs_path: Path, config: dict, *,
                 config_validator: Callable[[dict, int], None] | None = None):
        inspected = inspect_inputs(inputs_path, config, config_validator=config_validator)
        raw, self.data, self.parameters = inspected.raw, inspected.data, inspected.parameters
        binding = self.data.catalog["binding"]
        fit_ids, validation_ids = set(config["fit_circuit_ids"]), set(config["validation_circuit_ids"])
        selected = fit_ids | validation_ids
        self.parameter_digests = {cid: sha256_file(value) for cid, value in self.parameters.items()}
        if (inspected.expected_parameter_sha256
                and self.parameter_digests != inspected.expected_parameter_sha256):
            raise ValueError("V1_EXECUTION_PATH_MAP_PARAMETER_CHANGED")
        self.config,self.bin_ticks=config,config["bin_ticks"]
        self.fit,self.validation=[],[]
        representatives={}
        for episode in sorted(self.data.episodes.values(),key=lambda row:row["episode_id"]):
            circuit=episode["circuit_id"]
            if circuit not in selected:continue
            if episode["duration_ticks"] % self.bin_ticks:
                raise ValueError("V1_TRAINING_COMPLETE_EPISODE_CLOCK_REQUIRED")
            case=dict(case_id=episode["episode_id"],episode_id=episode["episode_id"],circuit_id=circuit,
                      split=episode["split"],block_id=episode["block_id"])
            (self.fit if circuit in fit_ids else self.validation).append(case)
            representatives.setdefault(circuit,case)
        self.binding=dict(circuit_inputs_sha256=hashlib.sha256(raw).hexdigest(),
            catalog_content_identity=hashlib.sha256(canonical_json(self.data.catalog).encode()).hexdigest(),
            native_parameter_sha256=self.parameter_digests,partition_unit="whole_source_circuit",
            partition_scope="original_archive_train_validation_splits; all_selected_catalog_episodes",
            fitting_episodes=len(self.fit),validation_episodes=len(self.validation),
            source_binding=binding)
        self.active_case,self.active_episode=None,None
        contract=None
        self.source_dimensions = {}
        for circuit in sorted(selected):
            case=representatives[circuit]
            episode=self._episode(case)
            metadata=episode.population_metadata()
            self.source_dimensions[circuit] = dict(units=len(episode.unit_ids), fields=len(episode.field_names),
                                                    source_dt_ms=episode.dt_ms)
            current=(episode.dt_ms,metadata["unit_feature_schema"],metadata["field_feature_schema"],metadata["control_features"])
            if contract is not None and current!=contract:
                raise ValueError("V1_TRAINING_COMMON_FEATURE_CONTROL_CLOCK_CONTRACT_REQUIRED")
            contract=current
            if case==self.fit[0]:self.unit_ids=episode.unit_ids.copy()
            del episode,metadata

    def _episode(self, case):
        episode=V1RenewalEpisode(self.data,case["episode_id"],bin_ticks=self.bin_ticks,
                                native_parameters=self.parameters[case["circuit_id"]])
        if episode.binding["native_parameters_sha256"]!=self.parameter_digests[case["circuit_id"]]:
            raise ValueError("V1_TRAINING_SOURCE_PARAMETER_FILE_CHANGED")
        return episode

"""Fitting-only empirical PFC references for development stimulus selection.

Condition00 is the undisturbed cue program. These references do not establish
memory recovery, intervention success or independent circuit-level inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import getpass
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import yaml

from .benchmark_tasks import task_protocol
from .circuit_data import CircuitDataset
from .pfc_kernel_parameters import circuit_provenance
from .pfc_population_training_data import _declared_path, _execution_paths, _unique_map_object
from .pfc_scale_analysis import SOURCE_ENTRY_SHA256, SOURCE_REVISION
from .pfc_source_dataset import _benchmark_registration
from .population_intervention import ReferenceTarget, target_arrays
from .provenance import canonical_json, freeze_json, sha256_file


def validate_config(config):
    expected = dict(
        schema="pfc_intervention_reference_v1", source_scope="pfc_source_reproduction_development",
        training_seeds=[41, 42, 43], fitting_circuit_blocks={23: [0, 1, 2], 52: [1, 2], 63: [0, 1, 2]},
        condition_indices=list(range(10)), reference_condition_index=0, origin_tick=2000,
        stop_tick_exclusive=100000, bin_ticks=100, expected_units=100000, expected_fields=17,
        scale_rule="fit_all_conditions_rms_with_single_spike_quantum_floor",
        reference_weighting="equal_circuit_initial_noise_block",
        field_semantics="source_preference_bins_plus_inhibitory_population", scientific_claim_qualified=False,
    )
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError("PFC_REFERENCE_EXPLICIT_CONFIG_KEYS_REQUIRED")
    for key, value in expected.items():
        # JSON object keys become strings when the bound report is loaded.
        if canonical_json(config[key]) != canonical_json(value):
            raise ValueError("PFC_REFERENCE_REGISTERED_CONFIG_REQUIRED:" + key)
    return expected


def _identity(name):
    match = re.fullmatch(r"pfc_working_memory_circuit_(\d{3})_block_(\d{3})_condition_(\d{2})", name)
    if match is None:
        raise ValueError("PFC_REFERENCE_PROSPECTIVE_EPISODE_ID_REQUIRED")
    return tuple(map(int, match.groups()))


def _manifest(spec, config):
    if (not isinstance(spec, dict) or set(spec) != {"schema", "source_entry", "episodes"}
            or spec["schema"] != "pfc_population_development_inputs_v1"
            or not isinstance(spec["source_entry"], str) or not spec["source_entry"]
            or not isinstance(spec["episodes"], list)):
        raise ValueError("PFC_REFERENCE_DECLARED_INPUT_MANIFEST_REQUIRED")
    expected = {(circuit, block, condition) for circuit, blocks in config["fitting_circuit_blocks"].items()
                for block in blocks for condition in config["condition_indices"]}
    fit, validation, seen = [], [], set()
    # Finish this pass before resolving any source path, including map entries.
    for row in spec["episodes"]:
        if (not isinstance(row, dict) or set(row) != {"dataset", "source_run", "episode_id", "partition"}
                or any(not isinstance(value, str) or not value for value in row.values())
                or row["partition"] not in ("development_fit", "development_validation")):
            raise ValueError("PFC_REFERENCE_PROTECTED_OR_UNDECLARED_PARTITION_REJECTED")
        identity = _identity(row["episode_id"])
        if row["episode_id"] in seen:
            raise ValueError("PFC_REFERENCE_DUPLICATE_EPISODE")
        seen.add(row["episode_id"])
        if row["partition"] == "development_fit":
            fit.append((identity, row))
        else:
            if identity[0] in config["fitting_circuit_blocks"]:
                raise ValueError("PFC_REFERENCE_WHOLE_CIRCUIT_PARTITION_CHANGED")
            validation.append(row)
    if len(fit) != len(expected) or {identity for identity, _ in fit} != expected:
        raise ValueError("PFC_REFERENCE_EXACT_80_FITTING_EPISODES_REQUIRED")
    return sorted(fit), validation


def _mapped_paths(inputs, spec):
    """Reuse the existing resolver; bootstrap only its circuit-ID whitelist."""
    map_name = os.environ.get("NEURONSIM_EXECUTION_PATH_MAP")
    if map_name is None:
        return _execution_paths(inputs, spec, {}), 0
    value = json.loads(Path(map_name).read_text(), object_pairs_hook=_unique_map_object)
    if (not isinstance(value, dict) or value.get("schema") != "pfc_population_execution_path_map_v1"
            or not isinstance(value.get("paths"), dict)):
        raise ValueError("PFC_EXECUTION_PATH_MAP_SCHEMA_REQUIRED")
    selected = {"fit_circuit_ids": set(), "validation_circuit_ids": set()}
    catalogs, validation_reads = {}, 0
    for row in spec["episodes"]:
        original = _declared_path(row["dataset"], inputs)
        entry = value["paths"].get(str(original))
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("PFC_EXECUTION_PATH_MAP_DECLARED_IDENTITIES_REQUIRED")
        target = Path(entry["path"])
        if not target.is_absolute() or ".." in target.parts or not target.is_relative_to(Path("/tmp") / getpass.getuser()):
            raise ValueError("PFC_EXECUTION_PATH_MAP_LOCAL_TARGET_REQUIRED")
        if target not in catalogs:
            catalogs[target] = json.loads((target / "dataset.json").read_text())
        episodes = [episode for shard in catalogs[target]["shards"] for episode in shard["episodes"]
                    if episode["episode_id"] == row["episode_id"]]
        if len(episodes) != 1:
            raise ValueError("PFC_EXECUTION_PATH_MAP_PROSPECTIVE_IDENTITY_CHANGED")
        key = "fit_circuit_ids" if row["partition"] == "development_fit" else "validation_circuit_ids"
        selected[key].add(episodes[0]["circuit_id"])
        validation_reads += int(row["partition"] == "development_validation")
    return _execution_paths(inputs, spec, {key: sorted(value) for key, value in selected.items()}), validation_reads


def _source_metadata(catalog, episode, manifest_row, identity, declared_source):
    binding = catalog["binding"]
    config = binding.get("resolved_source_config", {})
    if (binding.get("scope") != "source_reproduction" or binding.get("scenario_id") != "pfc_working_memory"
            or binding.get("sample_unit") != "complete_mechanistic_episode"
            or binding.get("source_revision") != SOURCE_REVISION
            or binding.get("source_entry_sha256") != SOURCE_ENTRY_SHA256
            or config.get("expected_source_revision") != SOURCE_REVISION
            or config.get("benchmark_task_protocol", task_protocol("pfc_working_memory"))
               != task_protocol("pfc_working_memory")):
        raise ValueError("PFC_REFERENCE_BOUND_SOURCE_REPRODUCTION_REQUIRED")
    candidate = config.get("benchmark_episode_candidate", {})
    declared = candidate.get("identity", {})
    circuit, block, condition = identity
    alias = f"pfc_working_memory_circuit_{circuit:03d}"
    if (declared.get("episode_id") != manifest_row["episode_id"]
            or tuple(declared.get(name) for name in ("circuit_index", "block_index", "condition_index")) != identity
            or declared.get("circuit_id") != alias or declared.get("block_id") != f"{alias}_block_{block:03d}"
            or declared.get("scenario_id") != "pfc_working_memory"
            or declared.get("sample_unit") != "complete_mechanistic_episode"
            or declared.get("split") != "train" or candidate.get("intended_split") != "train"
            or candidate.get("execution_scope") != "development"
            or candidate.get("production_eligible") is not False):
        raise ValueError("PFC_REFERENCE_FITTING_CANDIDATE_IDENTITY_CHANGED")
    _benchmark_registration(config)
    expected_source = dict(excitatory_neurons=80000, inhibitory_neurons=20000, excitatory_preference_bins=16,
                           dt_ms=.05, duration_ms=5000., readout_bin_ms=5.)
    if any(config.get(key) != value for key, value in expected_source.items()):
        raise ValueError("PFC_REFERENCE_COMMON_SOURCE_MAP_AND_CLOCK_REQUIRED")
    if condition == 0 and (candidate.get("condition_id") != "cue_none"
                          or config["stimulus_strength_nA"] != .5
                          or config["distractor_duration_ms"] != 0
                          or config["distractor_strength_nA"] != 0):
        raise ValueError("PFC_REFERENCE_UNDISTURBED_CUE_NONE_REQUIRED")
    provenance = circuit_provenance(config, SOURCE_ENTRY_SHA256)
    native_circuit = "pfc_" + provenance["parameter_identity"][:16]
    native_block = f"pfc_initial_{declared['initial_state_seed']}_external_{declared['background_seed']}"
    found = [dict(item, shard=shard["path"]) for shard in catalog["shards"] for item in shard["episodes"]
             if item["episode_id"] == manifest_row["episode_id"]]
    if (found != [episode] or episode["split"] != "development"
            or episode["circuit_id"] != native_circuit or episode["block_id"] != native_block
            or episode["dt_ms"] != .05 or episode["duration_ticks"] != 100000
            or type(episode["event_count"]) is not int or episode["event_count"] < 0):
        raise ValueError("PFC_REFERENCE_NATIVE_EPISODE_BINDING_CHANGED")
    circuit_row = catalog["circuits"][native_circuit]
    if circuit_row["units"] != 100000 or circuit_row["provenance"] != provenance:
        raise ValueError("PFC_REFERENCE_NATIVE_CIRCUIT_BINDING_CHANGED")
    sources = [source for source in binding.get("source_bindings", [])
               if source.get("case_id") == manifest_row["episode_id"]]
    if len(sources) != 1 or Path(sources[0]["path"]) != Path(declared_source):
        raise ValueError("PFC_REFERENCE_DECLARED_SOURCE_BINDING_CHANGED")
    return candidate, provenance, sources[0]


def _source_files(source, binding, source_binding):
    delivery_path, report_path = source.parent / "delivery.json", source / "report.json"
    if (sha256_file(delivery_path) != source_binding["delivery_sha256"]
            or sha256_file(report_path) != source_binding["report_sha256"]
            or sha256_file(source / "config.yaml") != binding["config_sha256"]):
        raise ValueError("PFC_REFERENCE_SOURCE_METADATA_BYTES_CHANGED")
    delivery, report = json.loads(delivery_path.read_text()), json.loads(report_path.read_text())
    files = [row for row in delivery["files"] if row["path"] == "source_run/report.json"]
    if (delivery.get("case_id") != source_binding["case_id"]
            or delivery.get("status") != "ARCHIVED_PFC_SOURCE_EXECUTION"
            or delivery.get("result_status") != "PFC_SOURCE_EXECUTION_COMPLETE"
            or delivery.get("source_execution_complete") is not True
            or len(files) != 1 or files[0]["sha256"] != source_binding["report_sha256"]
            or files[0]["bytes"] != report_path.stat().st_size
            or report.get("case_id") != source_binding["case_id"]
            or report.get("resolved_config") != binding["resolved_source_config"]
            or report.get("config_sha256") != binding["config_sha256"]):
        raise ValueError("PFC_REFERENCE_SOURCE_METADATA_IDENTITY_CHANGED")


def _field_names():
    return tuple([f"E_preference_{index * 360 / 16:g}_{(index + 1) * 360 / 16:g}_deg"
                  for index in range(16)] + ["I_population"])


def _static_map(metadata, provenance, common_ids):
    ids, fields = np.asarray(metadata["unit_ids"]), np.asarray(metadata["unit_field_index"])
    expected_fields = np.r_[np.arange(80000) * 16 // 80000, np.full(20000, 16)]
    if (ids.shape != (100000,) or not np.issubdtype(ids.dtype, np.integer)
            or fields.shape != ids.shape or not np.array_equal(fields, expected_fields)
            or metadata["field_names"] != _field_names()
            or not np.array_equal(metadata["populations"], np.where(expected_fields == 16, "I", "E"))
            or metadata["source_revision"] != SOURCE_REVISION or metadata["circuit_provenance"] != provenance
            or metadata["coordinate_semantics"] !=
               "E angle=360*source_index/N_E degrees; I has no preference coordinate"
            or (common_ids is None and len(np.unique(ids)) != len(ids))
            or (common_ids is not None and not np.array_equal(ids, common_ids))):
        raise ValueError("PFC_REFERENCE_COMMON_SOURCE_UNIT_ORDER_FIELDS_AND_UNITS_REQUIRED")
    return ids.copy(), np.bincount(fields, minlength=17)


def prepare_references(inputs_path: Path, config_path: Path, output: Path):
    """Write one bound reference.json without training or admitting validation outcomes."""
    inputs, config_path, output = Path(inputs_path), Path(config_path), Path(output)
    config = validate_config(yaml.safe_load(config_path.read_text()))
    spec = json.loads(inputs.read_text())
    fitting, validation = _manifest(spec, config)
    mapped, validation_metadata_reads = _mapped_paths(inputs, spec)
    def resolve(value, *, source_run=False):
        declared = _declared_path(value, inputs, source_run=source_run)
        return declared, mapped.get(declared, declared)
    declared_entry, source_entry = resolve(spec["source_entry"])
    if sha256_file(source_entry) != SOURCE_ENTRY_SHA256:
        raise ValueError("PFC_REFERENCE_PINNED_SOURCE_ENTRY_CHANGED")
    prepared, datasets, native_circuits = [], {}, {}
    common_ids, field_units = None, None
    # Every identity, source metadata binding and static map passes before outcomes.
    for identity, row in fitting:
        declared_dataset, root = resolve(row["dataset"])
        declared_source, source = resolve(row["source_run"], source_run=True)
        if root not in datasets:
            datasets[root] = CircuitDataset(root, splits=("development",))
        data = datasets[root]
        episode = data.episodes.get(row["episode_id"])
        if episode is None:
            raise ValueError("PFC_REFERENCE_FITTING_EPISODE_OUTSIDE_DEVELOPMENT_ARCHIVE")
        candidate, provenance, source_binding = _source_metadata(
            data.catalog, episode, row, identity, declared_source)
        _source_files(source, data.catalog["binding"], source_binding)
        previous = native_circuits.setdefault(identity[0], episode["circuit_id"])
        if previous != episode["circuit_id"]:
            raise ValueError("PFC_REFERENCE_PROSPECTIVE_CIRCUIT_HAS_MULTIPLE_NATIVE_IDENTITIES")
        common_ids, field_units = _static_map(data.circuit_metadata(episode["circuit_id"]), provenance, common_ids)
        record = dict(manifest_row=deepcopy(row), candidate_identity=deepcopy(candidate["identity"]),
                      declared_dataset=str(declared_dataset), execution_dataset=str(root),
                      declared_source_run=str(declared_source), execution_source_run=str(source),
                      catalog_sha256=sha256_file(root / "dataset.json"), catalog=deepcopy(data.catalog),
                      episode=deepcopy(episode), source_metadata_verified=True)
        prepared.append((identity, data, record))
    if len(set(native_circuits.values())) != 3:
        raise ValueError("PFC_REFERENCE_THREE_DISTINCT_NATIVE_FITTING_CIRCUITS_REQUIRED")
    horizon = (config["stop_tick_exclusive"] - config["origin_tick"]) // config["bin_ticks"]
    bin_seconds = config["bin_ticks"] * .05 / 1000
    sum_squares = np.zeros(17, dtype=np.float64)
    references, reference_ids, reference_blocks = [], [], []
    for identity, data, record in prepared:
        result = data._read(record["episode"]["episode_id"], config["origin_tick"],
                            config["stop_tick_exclusive"], config["bin_ticks"], unit_start=0, unit_stop=0)
        counts = np.asarray(result["field_counts"])
        if (counts.shape != (horizon, 17) or not np.isfinite(counts).all()
                or np.any(counts < 0) or np.any(counts != np.floor(counts))
                or result["counts"].shape != (horizon, 0) or len(result["unit_ids"]) != 0
                or result["field_names"] != _field_names() or result["bin_seconds"] != bin_seconds
                or result["episode_id"] != record["episode"]["episode_id"]
                or result["circuit_id"] != record["episode"]["circuit_id"]):
            raise ValueError("PFC_REFERENCE_COMPLETE_STREAMED_FIELD_COUNTS_REQUIRED")
        rates = counts.astype(np.float64) / (field_units * bin_seconds)
        sum_squares += np.einsum("tf,tf->f", rates, rates)
        if identity[2] == config["reference_condition_index"]:
            references.append(rates)
            reference_ids.append(record["episode"]["episode_id"])
            reference_blocks.append(record["candidate_identity"]["block_id"])
    rms = np.sqrt(sum_squares / (len(prepared) * horizon))
    quantum = 1. / (field_units * bin_seconds)
    report = dict(
        schema="pfc_intervention_reference_preparation_v1", status="PREPARED_FITTING_ONLY_PFC_REFERENCES",
        config=config, config_path=str(config_path.resolve()), config_sha256=sha256_file(config_path),
        inputs_path=str(inputs.resolve()), inputs_sha256=sha256_file(inputs), input_manifest=spec,
        source_entry=dict(declared_path=str(declared_entry), execution_path=str(source_entry),
                          sha256=SOURCE_ENTRY_SHA256),
        source_bindings=[record for _, _, record in prepared],
        scale_episode_count=len(prepared), scale_bins_per_episode=horizon, reference_episode_ids=reference_ids,
        reference_block_count=len(references), fitting_circuit_count=len(native_circuits),
        native_fitting_circuits={str(key): value for key, value in native_circuits.items()},
        field_unit_counts=field_units.tolist(), rms_rate_hz=rms.tolist(), single_spike_quantum_hz=quantum.tolist(),
        reference_target=dict(field_rate_hz=np.stack(references).tolist(), scale_hz=np.maximum(rms, quantum).tolist(),
                              field_names=list(_field_names()), independent_block_ids=reference_blocks,
                              bin_seconds=bin_seconds),
        reference_meaning="undisturbed_cue_program_empirical_mixture",
        reference_weighting="equal_circuit_initial_noise_block",
        inference_scope="eight_circuit_initial_noise_blocks_nested_in_three_fitting_circuits",
        validation_rows_skipped=len(validation), validation_catalog_metadata_reads=validation_metadata_reads,
        validation_outcome_reads=0, protected_outcome_reads=0, all_silent_references_retained=True,
        complete_source_unit_denominator=True, dense_unit_time_arrays_created=False,
        training_performed=False, generation_performed=False, native_execution_performed=False,
        success_tolerance_qualified=False, memory_recovery_qualified=False, scientific_claim_qualified=False,
    )
    _target_from_report(report)
    freeze_json(output / "reference.json", report)
    return report


def _target_from_report(report):
    if (report.get("schema") != "pfc_intervention_reference_preparation_v1"
            or report.get("status") != "PREPARED_FITTING_ONLY_PFC_REFERENCES"):
        raise ValueError("PFC_REFERENCE_PREPARATION_REPORT_REQUIRED")
    config = validate_config(report["config"])
    fitting, validation = _manifest(report["input_manifest"], config)
    if (report["scale_episode_count"] != 80 or report["scale_bins_per_episode"] != 980
            or report["reference_block_count"] != 8 or report["fitting_circuit_count"] != 3
            or report["validation_rows_skipped"] != len(validation)
            or report["reference_meaning"] != "undisturbed_cue_program_empirical_mixture"
            or report["reference_weighting"] != config["reference_weighting"]
            or report["inference_scope"] != "eight_circuit_initial_noise_blocks_nested_in_three_fitting_circuits"
            or any(report[key] is not False for key in (
                "dense_unit_time_arrays_created", "training_performed", "generation_performed",
                "native_execution_performed", "success_tolerance_qualified", "memory_recovery_qualified",
                "scientific_claim_qualified"))
            or report["validation_outcome_reads"] != 0 or report["protected_outcome_reads"] != 0
            or report["all_silent_references_retained"] is not True
            or report["complete_source_unit_denominator"] is not True):
        raise ValueError("PFC_REFERENCE_COMPLETE_FITTING_ONLY_SCOPE_REQUIRED")
    sources = report["source_bindings"]
    if len(sources) != len(fitting):
        raise ValueError("PFC_REFERENCE_COMPLETE_SOURCE_BINDINGS_REQUIRED")
    references, blocks, circuits = [], [], {}
    for (identity, row), record in zip(fitting, sources):
        if record["manifest_row"] != row or record["source_metadata_verified"] is not True:
            raise ValueError("PFC_REFERENCE_SOURCE_BINDING_ORDER_CHANGED")
        candidate, _, _ = _source_metadata(record["catalog"], record["episode"], row, identity,
                                           record["declared_source_run"])
        if record["candidate_identity"] != candidate["identity"]:
            raise ValueError("PFC_REFERENCE_CANDIDATE_BINDING_CHANGED")
        if circuits.setdefault(str(identity[0]), record["episode"]["circuit_id"]) != record["episode"]["circuit_id"]:
            raise ValueError("PFC_REFERENCE_NATIVE_CIRCUIT_BINDING_CHANGED")
        if identity[2] == 0:
            references.append(row["episode_id"])
            blocks.append(candidate["identity"]["block_id"])
    if (references != report["reference_episode_ids"] or circuits != report["native_fitting_circuits"]
            or len(set(circuits.values())) != 3):
        raise ValueError("PFC_REFERENCE_COMPLETE_REFERENCE_IDENTITIES_REQUIRED")
    values = report["reference_target"]
    target = ReferenceTarget(np.asarray(values["field_rate_hz"], dtype=np.float64), tuple(values["field_names"]),
                             np.asarray(values["scale_hz"], dtype=np.float64),
                             tuple(values["independent_block_ids"]), values["bin_seconds"])
    target_arrays(target, SimpleNamespace(field_names=_field_names(), bin_seconds=.005))
    rms, quantum = np.asarray(report["rms_rate_hz"]), np.asarray(report["single_spike_quantum_hz"])
    expected_units = np.array([5000] * 16 + [20000])
    if (target.field_rate_hz.shape != (8, 980, 17) or list(target.independent_block_ids) != blocks
            or not np.array_equal(report["field_unit_counts"], expected_units)
            or rms.shape != (17,) or not np.isfinite(rms).all() or np.any(rms < 0)
            or not np.array_equal(quantum, 1. / (expected_units * .005))
            or not np.array_equal(target.scale_hz, np.maximum(rms, quantum))):
        raise ValueError("PFC_REFERENCE_BOUND_RMS_QUANTUM_AND_TARGET_REQUIRED")
    return target


def load_reference(path: Path):
    """Load the existing ReferenceTarget and its complete preparation binding."""
    report = json.loads(Path(path).read_text())
    return _target_from_report(report), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_references(args.inputs, args.config, args.output)
    print(json.dumps(dict(status=report["status"], scale_episodes=report["scale_episode_count"],
                          reference_blocks=report["reference_block_count"],
                          output=str(args.output / "reference.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

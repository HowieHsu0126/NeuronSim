"""Run the pinned native PointNet pipeline on a bound source realization.

The default synchronous pulses are a software diagnostic. An optional bound
visual-input bundle replays the constructed LGN and declared background. Neither
route qualifies source fidelity or a production episode. SONATA ingestion, raw
Allen GLIF conversion, source weights, and native construction remain in BMTK.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import h5py
import numpy as np
import yaml

from . import v1_circuit_build as core
from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_cell_parameters import BMTK_CONVERTER_SHA256
from .v1_circuit_inputs import INPUT_SOURCES, REFERENCE_FILES, node_data
from .v1_circuit_source import read_source
from .v1_circuit_stimulus import load_inputs
from .v1_lgn_source import read_input_construction_cells
from .v1_nest_input_replay import NEST_VERSION, POINTNETWORK_SHA256, POINTSIMULATOR_SHA256
from .v1_nest_network import native_connection_admission
from .v1_nest_time_semantics import (V1_DURATION_MS, V1_RESOLUTION_MS, clock_tics_per_step,
                                    encode_precise_times, source_refractory_steps)
from .v1_synaptic_weights import RULE_NAMES, extract_functions


V1_FILES = ("v1_nodes.h5", "v1_node_types.csv", "v1_v1_edges.h5", "v1_v1_edge_types.csv")
INPUT_FILES = tuple(f"{pop}_{suffix}" for pop in ("lgn", "bkg")
                    for suffix in ("nodes.h5", "node_types.csv", "v1_edges.h5", "v1_edge_types.csv"))
PULSES_MS = (0.125, 0.5)
# This bounds optional tuple materialization; all sizes retain native row counts
# and the six-connection array witness. It is not a scientific acceptance gate.
MAX_MATERIALIZED_CONNECTION_ROWS = 2_000_000
ARRAY_ARTIFACTS = ("native/cortical_state.npz", "native/cortical_spikes.npz", "native/input_events.npz")
PASS_STATUS = "POINTNET_NATIVE_DIAGNOSTIC_COMPLETE"
VISUAL_PASS_STATUS = "POINTNET_NATIVE_VISUAL_INPUTS_COMPLETE_DEVELOPMENT"
TEMPORAL_PASS_STATUS = "POINTNET_NATIVE_TEMPORAL_SENSITIVITY_COMPLETE_DEVELOPMENT"
DIAGNOSTIC_MODE = "diagnostic_pulses"
VISUAL_MODE = "constructed_visual_inputs"
DIAGNOSTIC_SCHEMA = "diagnostic_pulse_matrix_v1"
VISUAL_SCHEMA = "packed_source_events_v1"
CLOCK_PROTOCOL_FIXED = dict(schema_version=1, scope="native_temporal_sensitivity_development",
    reference_resolution_ms=V1_RESOLUTION_MS, execution_resolution_ms=0.125, duration_ms=V1_DURATION_MS,
    independent_replay_required=True, maximum_new_full_conditions=1, allow_further_resolution_sweep=False,
    production_eligible=False, scientific_qualification=False)


BENCHMARK_CLOCK_SCOPE = "benchmark_temporal_sensitivity_development"

def validate_clock_protocol(protocol: dict) -> None:
    paths = {"reference_full_archive", "reference_canary_archive"}
    if not isinstance(protocol, dict) or set(protocol) != set(CLOCK_PROTOCOL_FIXED) | paths:
        raise ValueError("POINTNET_CLOCK_PROTOCOL_FIELDS_NOT_AUTHORIZED")
    fixed = dict(CLOCK_PROTOCOL_FIXED)
    if protocol.get("scope") == BENCHMARK_CLOCK_SCOPE:
        fixed["scope"] = BENCHMARK_CLOCK_SCOPE
    for key, expected in fixed.items():
        actual = protocol[key]
        valid_type = (type(actual) in (int, float) if type(expected) is float else type(actual) is type(expected))
        if not valid_type or actual != expected:
            raise ValueError("POINTNET_CLOCK_PROTOCOL_NOT_AUTHORIZED:" + key)
    if any(not isinstance(protocol[key], str) or not protocol[key].strip() for key in paths):
        raise ValueError("POINTNET_CLOCK_REFERENCE_PATH_REQUIRED")


def require_clock_canary_protocol(current: dict, canary: dict) -> None:
    """Reuse a numerical canary across independently bound benchmark cases.

    Each invocation still executes one full reference condition. The legacy
    single-condition protocol remains exact. Benchmark cases may change only
    the full reference archive; native admission binds that archive to the
    actual circuit, condition inputs, parameters, and source-clock replay.
    """
    validate_clock_protocol(current)
    validate_clock_protocol(canary)
    left, right = dict(current), dict(canary)
    if left["scope"] == right["scope"] == BENCHMARK_CLOCK_SCOPE:
        left.pop("reference_full_archive")
        right.pop("reference_full_archive")
    if left != right:
        raise ValueError("POINTNET_CLOCK_CANARY_PROTOCOL_DIFFERS")


def load_clock_protocol(path: Path) -> dict:
    protocol = yaml.safe_load(path.read_text())
    validate_clock_protocol(protocol)
    return protocol


def execution_clock(nest, resolution_ms: float) -> dict:
    resolution = float(nest.GetKernelStatus("resolution"))
    tics = float(nest.GetKernelStatus("tics_per_ms"))
    if resolution != resolution_ms or tics != 1000.0:
        raise ValueError("POINTNET_NATIVE_EXECUTION_CLOCK_DIFFERS")
    return dict(duration_ms=V1_DURATION_MS, resolution_ms=resolution, tics_per_ms=tics,
                tics_per_step=clock_tics_per_step(tics, resolution))


def refractory_clock_semantics(references: dict, clock: dict) -> dict:
    ids = list(references)
    physical = np.asarray([references[key]["converted_parameters"]["t_ref"] for key in ids], dtype=np.float64)
    source_steps = source_refractory_steps(physical, clock["tics_per_ms"])
    selected_steps = source_refractory_steps(physical, clock["tics_per_ms"], clock["resolution_ms"])
    return dict(source_rule="NEST Time::ms(t_ref).get_steps(): rounded integer tics, then ceiling to steps.",
        private_native_state_instrumented=False, physical_t_ref_unchanged=True,
        reference_resolution_ms=V1_RESOLUTION_MS, execution_resolution_ms=clock["resolution_ms"],
        tics_per_ms=clock["tics_per_ms"],
        models={str(key): dict(t_ref_ms=float(value), reference_steps=int(old), execution_steps=int(new),
                              reference_duration_ms=float(old * V1_RESOLUTION_MS),
                              execution_duration_ms=float(new * clock["resolution_ms"]))
                for key, value, old, new in zip(ids, physical, source_steps, selected_steps)})


def bound_files(archive: Path, relatives: tuple[str, ...]) -> tuple[dict, dict]:
    """Verify only the passed archive's explicitly needed delivered files."""
    delivery = json.loads((archive / "delivery.json").read_text())
    if delivery.get("exit_code") != 0:
        raise ValueError("POINTNET_PASSED_DELIVERY_REQUIRED")
    records = {}
    for relative in relatives:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("POINTNET_ARCHIVE_RELATIVE_PATH_REQUIRED")
        rows = [row for row in delivery["files"] if row["path"] == relative]
        source = archive / path
        if (len(rows) != 1 or source.stat().st_size != rows[0]["bytes"]
                or sha256_file(source) != rows[0]["sha256"]):
            raise ValueError("POINTNET_DELIVERED_FILE_CHANGED:" + relative)
        records[relative] = {key: rows[0][key] for key in ("bytes", "sha256")}
    return delivery, dict(delivery_sha256=sha256_file(archive / "delivery.json"), files=records)


def bind_clock_reference(protocol: dict, current: dict) -> tuple[dict, dict]:
    """Admit the frozen source-clock reference before native execution starts."""
    mode = current["mode"]
    if mode not in ("canary", "full"):
        raise ValueError("POINTNET_CLOCK_REFERENCE_MODE_NOT_AUTHORIZED")
    archive = Path(protocol["reference_" + mode + "_archive"])
    relatives = ("primary/report.json", "replay/report.json", "comparison/report.json")
    delivery, binding = bound_files(archive, relatives)
    primary, replay, comparison = [json.loads((archive / name).read_text()) for name in relatives]
    reference_job_id = delivery.get("job_id")
    visual = mode == "full"
    expected_delivery = "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT" if visual else "PASS_POINTNET_NATIVE_DIAGNOSTIC"
    expected_status = VISUAL_PASS_STATUS if visual else PASS_STATUS
    if (not isinstance(reference_job_id, str) or not reference_job_id
            or delivery.get("mode") != mode or delivery.get("status") != expected_delivery
            or any(row.get("status") != expected_status or row.get("exit_code") != 0
                   or row.get("runtime", {}).get("slurm", {}).get("SLURM_JOB_ID") != reference_job_id
                   or row.get("mode") != mode or row.get("simulated_ms") != V1_DURATION_MS
                   or row.get("native_nest_simulation") is not True
                   or row.get("native_parameters_and_initial_state_exact") is not True
                   or row.get("nest_version") != NEST_VERSION
                   or row.get("input_readback", {}).get("tics_per_ms") != 1000.0
                   or row.get("clock_protocol") is not None for row in (primary, replay))):
        raise ValueError("POINTNET_FROZEN_SOURCE_CLOCK_REFERENCE_REQUIRED")
    if (comparison.get("status") != "POINTNET_NATIVE_REPLAY_EQUAL" or comparison.get("exit_code") != 0
            or comparison.get("all_native_outputs_equal") is not True
            or comparison.get("independent_processes") is not True
            or comparison.get("binding_or_runtime_differences") != []
            or comparison.get("left_report_sha256") != binding["files"]["primary/report.json"]["sha256"]
            or comparison.get("right_report_sha256") != binding["files"]["replay/report.json"]["sha256"]
            or (primary["hostname"], primary["process_id"]) == (replay["hostname"], replay["process_id"])):
        raise ValueError("POINTNET_BOUND_INDEPENDENT_REFERENCE_REPLAY_REQUIRED")
    required_arrays = set(primary["array_artifacts"]) | {"native/parameters.json"}
    compared = comparison.get("files", [])
    if (len(compared) != len(required_arrays) or {row["path"] for row in compared} != required_arrays
            or any(row.get("equal") is not True for row in compared)):
        raise ValueError("POINTNET_COMPLETE_REFERENCE_OUTPUT_COMPARISON_REQUIRED")
    fields = ("bindings", "original_config_sha256", "mode", "source_denominators", "input_mode", "stimulus_kind")
    differences = [key for key in fields if current[key] != primary[key]]
    if current["wheel"]["sha256"] != primary["wheel"]["sha256"]:
        differences.append("wheel")
    if differences:
        raise ValueError("POINTNET_TEMPORAL_SOURCE_IDENTITY_DIFFERS:" + ",".join(differences))
    source_clock = primary["bindings"]["point_config"]["source_clock"]
    if source_clock != dict(duration_ms=V1_DURATION_MS, resolution_ms=V1_RESOLUTION_MS):
        raise ValueError("POINTNET_REFERENCE_SOURCE_CLOCK_DIFFERS")
    digest = primary["connections"]["source_batch_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("POINTNET_REFERENCE_SOURCE_BATCH_DIGEST_REQUIRED")
    return primary, dict(scope=protocol["scope"], reference_archive=str(archive),
        reference_job_id=reference_job_id, reference_receipts=binding,
        reference_primary_report_sha256=binding["files"]["primary/report.json"]["sha256"],
        reference_delivery_sha256=binding["delivery_sha256"],
        unchanged_upstream_bindings=True, reference_clock_source_batch_sha256=digest,
        interpretation="Whole-model temporal sensitivity; not pure integration error, convergence, or scientific qualification.")


def require_input_reference(recurrent_binding: dict, input_report: dict) -> None:
    reference = input_report["reference"]
    if reference["delivery_sha256"] != recurrent_binding["delivery_sha256"]:
        raise ValueError("POINTNET_INPUT_BOUND_TO_DIFFERENT_RECURRENT_DELIVERY")
    for relative in ("report.json", *REFERENCE_FILES):
        if reference["files"][relative] != recurrent_binding["files"]["primary/" + relative]:
            raise ValueError("POINTNET_INPUT_REFERENCE_CHANGED:" + relative)


def stage_network(recurrent: Path, inputs: Path, output: Path, config_sha: str) -> tuple[dict, dict, dict]:
    rec_files = ("primary/report.json", *("primary/glif/" + name for name in V1_FILES),
                 "primary/biophysical/v1_nodes.h5", "primary/biophysical/v1_node_types.csv")
    inp_files = ("primary/report.json", *("primary/glif/" + name for name in (*V1_FILES[:2], *INPUT_FILES)))
    rec_delivery, rec_binding = bound_files(recurrent, rec_files)
    inp_delivery, inp_binding = bound_files(inputs, inp_files)
    rec = json.loads((recurrent / "primary/report.json").read_text())
    inp = json.loads((inputs / "primary/report.json").read_text())
    mode = rec.get("mode")
    expected = {"canary": "ORIGINAL_RECURRENT_CANARY_COMPLETE", "full": "ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE"}
    if (mode not in expected or rec.get("status") != expected[mode]
            or inp.get("status") != "ORIGINAL_CIRCUIT_INPUTS_COMPLETE"
            or not rec_delivery.get("status", "").startswith("PASS_SOURCE_RECURRENT_")
            or not inp_delivery.get("status", "").startswith("PASS_SOURCE_INPUT_")
            or any(row.get("mode") != mode for row in (inp, rec_delivery, inp_delivery))
            or any(row.get("exit_code") != 0 for row in (rec, inp))):
        raise ValueError("POINTNET_COMPLETED_MATCHING_REALIZATIONS_REQUIRED")
    if any(row.get("original_config_sha256") != config_sha for row in (rec, inp)):
        raise ValueError("POINTNET_ORIGINAL_CIRCUIT_CONFIG_CHANGED")
    require_input_reference(rec_binding, inp)
    for name in V1_FILES[:2]:
        key = "primary/glif/" + name
        if rec_binding["files"][key] != inp_binding["files"][key]:
            raise ValueError("POINTNET_INPUT_CORTICAL_IDENTITY_CHANGED:" + name)
    network = output / "network"
    network.mkdir()
    for archive, names in ((recurrent, V1_FILES), (inputs, INPUT_FILES)):
        for name in names:
            shutil.copy2(archive / "primary/glif" / name, network / name)
    return rec, inp, dict(recurrent=rec_binding, inputs=inp_binding)


def bind_sources(source_root: Path, reports: tuple[dict, ...]) -> tuple[dict, dict]:
    roles = set(core.SOURCE_FILES) | set(INPUT_SOURCES)
    roles.update(("original_glif_recurrent_parameters", "weight_semantics/legacy_pointnet_runner"))
    roles.update("weight_semantics/legacy_" + name for name in ("e2e", "e2i", "i2e", "i2i"))
    bodies, bindings = {}, {}
    for role in sorted(roles):
        body, receipt = read_source(source_root / role)
        record = dict(sha256=receipt["sha256"], bytes=len(body),
                      receipt_sha256=sha256_file(source_root / role / "source.json"))
        for report in reports:
            if role in report["sources"]:
                previous = report["sources"][role]
                if (record["sha256"] != previous["original_sha256"]
                        or record["bytes"] != previous["original_bytes"]
                        or record["receipt_sha256"] != previous["receipt_sha256"]):
                    raise ValueError("POINTNET_REALIZATION_SOURCE_CHANGED:" + role)
        bodies[role], bindings[role] = body, record
    return bodies, bindings


def parameter_join(types: dict, references: list[dict]) -> dict:
    by_type = {}
    for row in references:
        identity = row["specimen_id"]
        if type(identity) is not int or identity in by_type:
            raise ValueError("POINTNET_UNIQUE_REFERENCE_SPECIMENS_REQUIRED")
        by_type[identity] = row
    if not set(types).issubset(by_type):
        raise ValueError("POINTNET_REALIZED_GLIF_REFERENCE_MISSING")
    for identity, row in types.items():
        if (row["model_type"] != "point_process" or row["model_template"] != "nest:glif_lif_asc_psc"
                or by_type[identity]["converted_template"] != "nest:glif_psc"):
            raise ValueError("POINTNET_SOURCE_GLIF_TEMPLATE_DIFFERS")
        name = Path(row["dynamics_params"])
        if name.is_absolute() or ".." in name.parts or name.suffix != ".json":
            raise ValueError("POINTNET_SOURCE_PARAMETER_FILENAME_INVALID")
    return {identity: by_type[identity] for identity in types}


def stage_parameters(archive: Path, data: dict, output: Path) -> tuple[dict, dict]:
    delivery, binding = bound_files(archive, ("reference/intake.json",))
    reference = json.loads((archive / "reference/intake.json").read_text())
    if (delivery.get("status") != "REFERENCE_PARAMETERS_ACQUIRED"
            or reference.get("status") != "REFERENCE_PARAMETERS_ACQUIRED"
            or reference.get("expected_specimens") != 111 or reference.get("acquired_references") != 111
            or len(reference["references"]) != 111
            or reference["bmtk_converter_sha256"] != BMTK_CONVERTER_SHA256):
        raise ValueError("POINTNET_COMPLETE_BOUND_GLIF_REFERENCE_REQUIRED")
    joined = parameter_join(data["types"], reference["references"])
    relative = tuple("reference/" + row["source"]["path"] for row in reference["references"])
    _, raw_binding = bound_files(archive, relative)
    binding["files"].update(raw_binding["files"])
    destination = output / "components/cell_models/nest_models"
    destination.mkdir(parents=True)
    written = {}
    for identity, row in joined.items():
        source = archive / "reference" / row["source"]["path"]
        bound = binding["files"]["reference/" + row["source"]["path"]]
        if bound != {key: row["source"][key] for key in ("bytes", "sha256")}:
            raise ValueError("POINTNET_RAW_PARAMETER_RECEIPT_DIFFERS")
        raw = json.loads(source.read_text())
        if raw.get("type") != "GLIF":
            raise ValueError("POINTNET_AUTOMATIC_GLIF_CONVERSION_MARKER_MISSING")
        name = data["types"][identity]["dynamics_params"]
        if name in written and written[name] != bound["sha256"]:
            raise ValueError("POINTNET_CONFLICTING_SOURCE_PARAMETER_FILENAME")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name not in written:
            shutil.copy2(source, target)
        written[name] = bound["sha256"]
    binding.update(realized_specimens=sorted(joined), raw_parameter_filenames=written,
                   purpose="Verify PointNet automatic conversion against existing reference metadata; not source-fidelity evidence.")
    return joined, binding


def edge_inventory(network: Path, population: str, expected: dict, chunk_size: int) -> dict:
    """Audit denominators only; BMTK remains the SONATA connection consumer."""
    types = core.table_records(network / f"{population}_v1_edge_types.csv", "edge_type_id")
    if population in ("lgn", "bkg") and any(row.get("weight_function") is not None for row in types.values()):
        raise ValueError("POINTNET_SOURCE_INPUT_DEFAULT_WEIGHT_PATH_REQUIRED")
    known = np.asarray(sorted(types), dtype=np.uint64)
    zeros = np.asarray([types[int(key)]["syn_weight"] == 0 for key in known])
    physical, zero_rows = 0, 0
    with h5py.File(network / f"{population}_v1_edges.h5", "r") as handle:
        group = handle[f"edges/{population}_to_v1"]
        count = len(group["source_node_id"])
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            nsyns = core.nsyns_for_rows(group, group["edge_group_id"][start:stop], group["edge_group_index"][start:stop])
            physical += int(nsyns.sum(dtype=np.uint64))
            positions = core.positions_for_ids(known, group["edge_type_id"][start:stop])
            zero_rows += int(np.count_nonzero(zeros[positions]))
    if count != expected["edge_rows"] or physical != expected["physical_synapses"]:
        raise ValueError("POINTNET_SOURCE_EDGE_DENOMINATOR_DIFFERS:" + population)
    return dict(edge_rows=count, physical_synapses=physical, source_table_zero_weight_rows=zero_rows)


def write_diagnostic_spikes(path: Path, population: str, source_ids: np.ndarray) -> None:
    source_ids = np.asarray(source_ids)
    if (source_ids.ndim != 1 or source_ids.dtype.kind not in "iu" or not len(source_ids)
            or np.any(source_ids < 0) or len(np.unique(source_ids)) != len(source_ids)):
        raise ValueError("POINTNET_DIAGNOSTIC_SOURCE_IDENTITIES_INVALID")
    # One population per file makes PointNet's node-ID-only spike setter unambiguous.
    with h5py.File(path, "x") as handle:
        group = handle.create_group("spikes/" + population)
        group.attrs["sorting"] = "by_id"
        group.create_dataset("node_ids", data=np.repeat(source_ids, len(PULSES_MS)))
        group.create_dataset("timestamps", data=np.tile(PULSES_MS, len(source_ids))).attrs["units"] = "ms"


def input_trains_by_population(plan: dict, data: dict) -> dict:
    """Join the loaded plan to source IDs, retaining silent generators."""
    if set(np.unique(plan['population'])) != {0, 1} or len(plan['trains']) != len(plan['source_id']):
        raise ValueError('POINTNET_EXTERNAL_INPUT_POPULATION_INVENTORY_DIFFERS')
    trains = {}
    for code, name in enumerate(('lgn', 'bkg')):
        units = np.flatnonzero(plan['population'] == code)
        order = np.argsort(plan['source_id'][units])
        units = units[order]
        if not np.array_equal(plan['source_id'][units], data[name]['sorted_ids']):
            raise ValueError('POINTNET_EXTERNAL_INPUT_SOURCE_IDS_DIFFER:' + name)
        # The pinned PointNet setter sorts each unit's timestamps before SetStatus.
        trains[name] = [np.sort(plan['trains'][int(unit)]) for unit in units]
        if any(len(train) != plan['source_counts'][unit] for train, unit in zip(trains[name], units)):
            raise ValueError('POINTNET_EXTERNAL_INPUT_TRAIN_COUNTS_DIFFER:' + name)
    return trains


def load_spike_inputs(archive: Path, expected_binding: dict, data: dict) -> tuple[dict, dict, dict]:
    """Admit only a complete bound producer bundle; never regenerate its events."""
    plan, bundle = load_inputs(archive, expected_binding, require_complete=True)
    input_trains_by_population(plan, data)
    relatives = ('visual_inputs/report.json', *(f"visual_inputs/{bundle['populations'][name][key]}"
                 for name in ('lgn', 'bkg') for key in ('plan', 'sonata')))
    _, binding = bound_files(archive, relatives)
    binding.update(bundle_report_sha256=bundle['bundle_report_sha256'],
        input_identity_sha256=bundle['input_identity_sha256'], bundle_status=bundle['status'],
        input_construction_binding=bundle['input_construction_binding'], config=bundle['config'],
        source_clock=bundle['source_clock'], complete_lgn_population=bundle['complete_lgn_population'])
    paths = {name: archive/'visual_inputs'/bundle['populations'][name]['sonata'] for name in ('lgn', 'bkg')}
    return plan, binding, paths


def stage_point_config(source: Path, output: Path, data: dict, bodies: dict,
                       *, spike_files: dict | None = None,
                       resolution_ms: float = V1_RESOLUTION_MS) -> tuple[Path, dict]:
    raw, receipt = read_source(source)
    config = json.loads(raw)
    if config["run"]["duration"] != V1_DURATION_MS or config["run"]["dt"] != V1_RESOLUTION_MS:
        raise ValueError("POINTNET_SOURCE_CLOCK_CHANGED")
    if config.get("target_simulator") != "NEST":
        raise ValueError("POINTNET_SOURCE_SIMULATOR_CHANGED")
    if resolution_ms not in (V1_RESOLUTION_MS, CLOCK_PROTOCOL_FIXED["execution_resolution_ms"]):
        raise ValueError("POINTNET_EXECUTION_RESOLUTION_NOT_AUTHORIZED")
    (output / "source_point_config.json").write_bytes(raw)
    destination = output / "components/synaptic_models"
    destination.mkdir(parents=True)
    for name in ("e2e", "e2i", "i2e", "i2i"):
        body = bodies["weight_semantics/legacy_" + name]
        params = json.loads(body)
        if type(params.get("receptor_type")) is not int or not 1 <= params["receptor_type"] <= 4:
            raise ValueError("POINTNET_SOURCE_RECEPTOR_INVALID")
        (destination / (name + ".json")).write_bytes(body)
    inputs = output / "inputs"
    inputs.mkdir()
    if spike_files is not None and set(spike_files) != {"lgn", "bkg"}:
        raise ValueError("POINTNET_EXTERNAL_SONATA_POPULATIONS_REQUIRED")
    input_paths = {}
    for population in ("lgn", "bkg"):
        if spike_files is None:
            path = inputs / (population + "_diagnostic_spikes.h5")
            write_diagnostic_spikes(path, population, data[population]["sorted_ids"])
        else:
            path = inputs / (population + "_spikes.h5")
            shutil.copy2(spike_files[population], path)
        input_paths[population] = path
    resolved = deepcopy(config)
    resolved["manifest"].update({"$BASE_DIR": str(output), "$NETWORK_DIR": str(output / "network"),
        "$COMPONENTS_DIR": str(output / "components"), "$OUTPUT_DIR": str(output / "output"),
        "$LGN_DIR": str(inputs), "$BKG_DIR": str(inputs)})
    seen = []
    for value in resolved["inputs"].values():
        population = value["node_set"]
        if population not in ("lgn", "bkg") or value["input_type"] != "spikes":
            raise ValueError("POINTNET_SOURCE_SPIKE_INPUT_DECLARATION_DIFFERS")
        seen.append(population)
        value.update(module="sonata", input_file=str(input_paths[population]))
    if sorted(seen) != ["bkg", "lgn"]:
        raise ValueError("POINTNET_TWO_SOURCE_INPUT_POPULATIONS_REQUIRED")
    resolved["run"].update(n_thread=1, precise_times=True, dt=resolution_ms)
    config_path = output / "point_config.json"
    freeze_json(config_path, resolved)
    binding = dict(sha256=receipt["sha256"], bytes=len(raw),
        receipt_sha256=sha256_file(source / "source.json"),
        source_clock=dict(duration_ms=V1_DURATION_MS, resolution_ms=V1_RESOLUTION_MS),
        source_inputs=config["inputs"], diagnostic_replacement=spike_files is None,
        population_files_separate=True)
    if spike_files is None:
        binding["pulse_times_ms"] = list(PULSES_MS)
    else:
        binding["external_source_spike_replay"] = True
    return config_path, binding


def native_vector(values, count: int, field: str, integer=False) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim == 0:
        result = np.full(count, result.item())
    if (result.shape != (count,) or result.dtype.kind not in ("iu" if integer else "iuf")
            or not np.isfinite(result).all()):
        raise ValueError("POINTNET_NATIVE_BATCH_FIELD_INVALID:" + field)
    return result.astype(np.int64 if integer else np.float64, copy=True)


def canonical_connections(values: dict) -> dict:
    fields = ("source", "target", "receptor_type", "weight", "delay")
    count = len(values["source"])
    result = {field: native_vector(values[field], count, field, field in fields[:3]) for field in fields}
    order = np.lexsort(tuple(result[field] for field in reversed(fields)))
    return {field: value[order] for field, value in result.items()}


def cortical_connection_count(nest, cortical_ids: np.ndarray) -> tuple[int, dict]:
    """Count cortical targets using the complete complementary node partition.

    Kernel counts and GetConnections have matching scope in this serial runner.
    Querying other targets includes actual output-device connections and any
    other noncortical connections, without allocating descriptors for the much
    larger cortical graph. The pinned spike_generator model cannot receive
    connections. Exclude those targets: NEST otherwise scans outgoing device
    connections for each generator even though the result must be empty.
    Uninstrumented cortical additions remain detectable.
    """
    if nest.NumProcesses() != 1:
        raise ValueError("POINTNET_CONNECTION_COUNT_REQUIRES_SERIAL_KERNEL")
    ids = np.asarray(cortical_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("POINTNET_CORTICAL_COUNT_IDENTITIES_INVALID")
    all_ids = np.asarray(nest.GetNodes().tolist(), dtype=np.int64)
    if not np.isin(ids, all_ids).all():
        raise ValueError("POINTNET_CORTICAL_COUNT_IDENTITIES_MISSING")
    outside = np.setdiff1d(all_ids, ids)
    total = int(nest.GetKernelStatus("num_connections"))
    if len(outside):
        outside_nodes = nest.NodeCollection(outside.tolist())
        models = np.asarray(nest.GetStatus(outside_nodes, "model"))
        if models.shape != outside.shape:
            raise ValueError("POINTNET_NATIVE_NONCORTICAL_MODEL_COVERAGE_DIFFERS")
        receiving_ids = outside[models != "spike_generator"]
        excluded = (len(nest.GetConnections(target=nest.NodeCollection(receiving_ids.tolist())))
                    if len(receiving_ids) else 0)
    else:
        excluded = 0
    if excluded > total:
        raise ValueError("POINTNET_NATIVE_CONNECTION_COUNT_SCOPE_DIFFERS")
    return total - excluded, dict(native_count_method="kernel_total_minus_complementary_targets",
        kernel_node_count=len(all_ids), noncortical_target_nodes=len(outside),
        native_noncortical_target_connections=excluded)


class ConnectionEvidence:
    """Transparent instrumentation of PointNet's already computed native batches."""
    def __init__(self, nest, graph, expected: dict, output: Path, *,
                 resolution_ms: float = V1_RESOLUTION_MS, reference_source_batch_sha256: str | None = None):
        self.nest, self.graph, self.expected, self.output = nest, graph, expected, output
        self.resolution_ms = resolution_ms
        self.expected_reference_digest = reference_source_batch_sha256
        self.clock_checked = False
        self.rows = {name: 0 for name in expected}
        self.zeros = {name: 0 for name in expected}
        self.requested_changes = 0
        self.max_delay_change = 0.0
        self.digest = hashlib.sha256()
        self.reference_digest = hashlib.sha256() if resolution_ms != V1_RESOLUTION_MS else self.digest
        self.materialize = sum(row["edge_rows"] for row in expected.values()) <= MAX_MATERIALIZED_CONNECTION_ROWS
        self.batches = []

    @contextmanager
    def capture(self):
        original = self.graph._nest_connect

        def traced(source, target, conn_spec="one_to_one", syn_spec=None):
            if not self.clock_checked:
                execution_clock(self.nest, self.resolution_ms)
                self.clock_checked = True
            count = len(source)
            if conn_spec != "one_to_one" or syn_spec is None or syn_spec["synapse_model"] != "static_synapse":
                raise ValueError("POINTNET_SOURCE_NATIVE_CONNECTION_MODE_DIFFERS")
            src = native_vector(source, count, "source", True)
            dst = native_vector(target, count, "target", True)
            pool = self.graph.gid_map if int(src[0]) in self.graph.gid_map._gid2pop_id else self.graph._virtual_gids
            _, source_pop = pool.get_node_ids(src)
            _, target_pop = self.graph.gid_map.get_node_ids(dst)
            names = set(source_pop)
            if len(names) != 1 or not np.all(target_pop == "v1") or next(iter(names)) not in self.expected:
                raise ValueError("POINTNET_NATIVE_ENDPOINT_POPULATIONS_DIFFER")
            population = next(iter(names))
            row = dict(source=src, target=dst,
                receptor_type=native_vector(syn_spec["receptor_type"], count, "receptor_type", True),
                weight=native_vector(syn_spec["weight"], count, "weight"),
                requested_delay=native_vector(syn_spec["delay"], count, "delay"))
            if np.any(row["requested_delay"] <= 0) or np.any((row["receptor_type"] < 1) | (row["receptor_type"] > 4)):
                raise ValueError("POINTNET_NATIVE_DELAY_OR_RECEPTOR_INVALID")
            row["delay"] = np.floor(row["requested_delay"] / self.resolution_ms + .5) * self.resolution_ms
            before = int(self.nest.GetKernelStatus("num_connections"))
            # Pass through the original objects and parameters without coercion.
            result = original(source, target, conn_spec=conn_spec, syn_spec=syn_spec)
            if int(self.nest.GetKernelStatus("num_connections")) - before != count:
                raise ValueError("POINTNET_NATIVE_BATCH_CONNECTION_COUNT_DIFFERS")
            self.rows[population] += count
            self.zeros[population] += int(np.count_nonzero(row["weight"] == 0))
            differences = np.abs(row["delay"] - row["requested_delay"])
            self.requested_changes += int(np.count_nonzero(differences))
            self.max_delay_change = max(self.max_delay_change, float(differences.max(initial=0.0)))
            self.digest.update(population.encode())
            for key, value in row.items():
                self.digest.update(key.encode())
                self.digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                self.digest.update(value.tobytes())
            if self.reference_digest is not self.digest:
                # Preserve the historical serialization of the SAME input tuple,
                # including its requested delay, represented on the source clock.
                self.reference_digest.update(population.encode())
                for key, value in row.items():
                    if key == "delay":
                        value = np.floor(row["requested_delay"] / V1_RESOLUTION_MS + .5) * V1_RESOLUTION_MS
                    self.reference_digest.update(key.encode())
                    self.reference_digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                    self.reference_digest.update(value.tobytes())
            if self.materialize:
                self.batches.append(row)
            return result

        self.graph._nest_connect = traced
        try:
            yield
        finally:
            self.graph._nest_connect = original

    def verify(self, cortical_ids: np.ndarray) -> dict:
        if (self.expected_reference_digest is not None
                and self.reference_digest.hexdigest() != self.expected_reference_digest):
            raise ValueError("POINTNET_REFERENCE_CLOCK_SOURCE_BATCH_DIFFERS")
        for population, row in self.expected.items():
            if self.rows[population] != row["edge_rows"]:
                raise ValueError("POINTNET_NATIVE_POPULATION_ROW_COUNT_DIFFERS:" + population)
            if population in ("lgn", "bkg") and self.zeros[population] != row["source_table_zero_weight_rows"]:
                raise ValueError("POINTNET_NATIVE_ZERO_WEIGHT_INPUT_ROWS_DIFFER:" + population)
        total = sum(self.rows.values())
        if self.materialize:
            target = self.nest.NodeCollection(sorted(map(int, cortical_ids)))
            connections = self.nest.GetConnections(target=target)
            actual_count = len(connections)
            count_evidence = dict(native_count_method="full_cortical_target_query")
        else:
            actual_count, count_evidence = cortical_connection_count(self.nest, cortical_ids)
        if actual_count != total:
            raise ValueError("POINTNET_NATIVE_CORTICAL_TARGET_CONNECTION_COUNT_DIFFERS")
        result = dict(rows_by_source_population=self.rows, native_connections_to_cortex=total,
            physical_synapses=sum(row["physical_synapses"] for row in self.expected.values()),
            zero_weight_rows_by_population=self.zeros, source_batch_sha256=self.digest.hexdigest(),
            source_batch_resolution_ms=self.resolution_ms, reference_clock_resolution_ms=V1_RESOLUTION_MS,
            reference_clock_source_batch_sha256=self.reference_digest.hexdigest(),
            instrumentation="Transparent PointNetwork._nest_connect pass-through; original arguments unchanged and hook restored.",
            source_delays_rounded_by_nest=self.requested_changes, maximum_delay_rounding_ms=self.max_delay_change,
            full_native_tuple_readback=self.materialize, native_tuple_readback_limit_rows=MAX_MATERIALIZED_CONNECTION_ROWS)
        result.update(count_evidence)
        if self.expected_reference_digest is not None:
            result["reference_clock_source_batch_exact"] = True
        if self.materialize:
            expected = {key: np.concatenate([row[key] for row in self.batches]) for key in self.batches[0]}
            native = connections.get()
            actual = dict(source=native["source"], target=native["target"], receptor_type=native["receptor"],
                          weight=native["weight"], delay=native["delay"])
            a, b = canonical_connections(expected), canonical_connections(actual)
            for key in a:
                if not np.array_equal(a[key], b[key]):
                    raise ValueError("POINTNET_NATIVE_CONNECTION_TUPLES_DIFFER:" + key)
            np.savez_compressed(self.output / "native/connection_source_batches.npz", **expected)
            np.savez_compressed(self.output / "native/connection_readback.npz", **b)
            result["exact_native_tuples_verified"] = total
        return result


def status_rows(nest, native_ids: np.ndarray, keys: list[str]):
    """NEST collections are sorted; restore the caller's explicit source order."""
    order = np.argsort(native_ids)
    rows = nest.GetStatus(nest.NodeCollection(np.asarray(native_ids)[order].tolist()), keys)
    if len(rows) != len(native_ids):
        raise ValueError("POINTNET_NATIVE_STATUS_COVERAGE_DIFFERS")
    result = [None] * len(rows)
    for position, row in zip(order, rows):
        result[int(position)] = row
    return result


def verify_parameters(nest, data: dict, native_ids: np.ndarray, reference: dict, output: Path) -> np.ndarray:
    initial = np.empty(data["count"], dtype=np.float64)
    readback = {}
    for identity, ref in reference.items():
        positions = np.flatnonzero(data["sorted_types"] == identity)
        expected = ref["converted_parameters"]
        keys = list(expected)
        rows = status_rows(nest, native_ids[positions], keys)
        for position, row in zip(positions, rows):
            for key, value in zip(keys, row):
                observed, wanted = np.asarray(value), np.asarray(expected[key])
                if observed.shape != wanted.shape or not np.isfinite(observed).all() or not np.array_equal(observed, wanted):
                    raise ValueError(f"POINTNET_NATIVE_PARAMETER_DIFFERS:{identity}:{key}")
            initial[position] = row[keys.index("V_m")]
        readback[str(identity)] = dict(units=len(positions), model="glif_psc", all_units_exact=True,
            parameters={key: np.asarray(value).tolist() for key, value in zip(keys, rows[0])},
            raw_parameter_sha256=ref["source"]["sha256"])
    freeze_json(output / "native/parameters.json", readback)
    return initial


class InputReadback:
    """Observe generators after PointNet's two input modules initialize."""
    def __init__(self, nest, data: dict, evidence: ConnectionEvidence, cortical_ids, output: Path, report: dict,
                 *, input_plan: dict | None = None, resolution_ms: float = V1_RESOLUTION_MS):
        self.nest, self.data, self.evidence = nest, data, evidence
        self.cortical_ids, self.output, self.report = cortical_ids, output, report
        self.input_trains = None if input_plan is None else input_trains_by_population(input_plan, data)
        self.resolution_ms = resolution_ms

    def initialize(self, sim):
        records = {key: [] for key in ("population", "source_id", "native_id", "inverse_source_id", "source_event_counts",
            "requested_times_ms", "native_times_ms", "encoded_steps", "encoded_offsets_ms", "precise_times")}
        self.report["execution_clock"] = execution_clock(self.nest, self.resolution_ms)
        ticks = self.report["execution_clock"]["tics_per_ms"]
        external = self.input_trains is not None
        if external:
            records["source_event_unit"] = []
        encoded = None if external else encode_precise_times(np.asarray(PULSES_MS), ticks, self.resolution_ms)
        unit_offset = 0
        for population in ("lgn", "bkg"):
            ids = self.data[population]["sorted_ids"]
            pool = sim.net._virtual_gids
            native_ids = np.asarray(pool.get_nestids(population, ids), dtype=np.int64)
            inverse, populations = pool.get_node_ids(native_ids)
            if not np.array_equal(inverse, ids) or not np.all(populations == population):
                raise ValueError("POINTNET_NATIVE_INPUT_IDENTITY_DIFFERS")
            rows = status_rows(self.nest, native_ids, ["spike_times", "precise_times"])
            precise = np.asarray([row[1] for row in rows], dtype=bool)
            if external:
                trains = self.input_trains[population]
                encodings = [encode_precise_times(train, ticks, self.resolution_ms) for train in trains]
                observed = [np.asarray(row[0], dtype=np.float64) for row in rows]
                counts = np.asarray([len(train) for train in trains], dtype=np.int64)
                if not precise.all() or any(not np.array_equal(actual, item["decoded_times_ms"])
                                           for actual, item in zip(observed, encodings)):
                    raise ValueError("POINTNET_NATIVE_EXTERNAL_SPIKE_TIMES_DIFFER:" + population)
                values = dict(population=np.full(len(ids), population), source_id=ids.astype(np.int64, copy=False),
                    native_id=native_ids, inverse_source_id=np.asarray(inverse, dtype=np.int64),
                    source_event_counts=counts, precise_times=precise,
                    source_event_unit=np.repeat(np.arange(len(ids), dtype=np.int64) + unit_offset, counts),
                    requested_times_ms=np.concatenate(trains), native_times_ms=np.concatenate(observed),
                    encoded_steps=np.concatenate([item["steps"] for item in encodings]),
                    encoded_offsets_ms=np.concatenate([item["offsets_ms"] for item in encodings]))
                unit_offset += len(ids)
            else:
                observed = np.asarray([row[0] for row in rows], dtype=np.float64)
                wanted = np.tile(encoded["decoded_times_ms"], (len(ids), 1))
                if observed.shape != wanted.shape or not np.array_equal(observed, wanted) or not precise.all():
                    raise ValueError("POINTNET_NATIVE_DIAGNOSTIC_PULSES_DIFFER")
                values = dict(population=np.full(len(ids), population), source_id=ids, native_id=native_ids,
                    inverse_source_id=inverse, source_event_counts=np.full(len(ids), len(PULSES_MS), dtype=np.int64),
                    requested_times_ms=np.tile(PULSES_MS, (len(ids), 1)), native_times_ms=observed,
                    encoded_steps=np.tile(encoded["steps"], (len(ids), 1)),
                    encoded_offsets_ms=np.tile(encoded["offsets_ms"], (len(ids), 1)), precise_times=precise)
            for key in records:
                records[key].append(values[key])
        arrays = {key: np.concatenate(value) for key, value in records.items()}
        np.savez_compressed(self.output / "native/input_events.npz", **arrays)
        self.report["input_readback"] = dict(generators=len(arrays["source_id"]),
            events=int(arrays["source_event_counts"].sum()),
            all_population_and_source_ids_exact=True, tics_per_ms=ticks, resolution_ms=self.resolution_ms,
            timestamp_semantics="Released-NEST precise-time generator encoding at the execution clock; GLIF updates remain grid based.")
        self.report["input_event_schema"] = VISUAL_SCHEMA if external else DIAGNOSTIC_SCHEMA
        if external:
            self.report["input_readback"].update(all_generator_events_exact=True,
                silent_generators=int(np.count_nonzero(arrays["source_event_counts"] == 0)),
                event_unit_index="Index into complete source inventory in this artifact, including silent units.")
        else:
            self.report["input_readback"]["all_generator_pulses_exact"] = True
        self.report["connections"] = self.evidence.verify(self.cortical_ids)
        self.report["native_total_connections_including_output_devices"] = int(self.nest.GetKernelStatus("num_connections"))

    def finalize(self, sim):
        pass


def collect_spikes(path: Path, source_ids: np.ndarray, native_ids: np.ndarray) -> tuple[dict, np.ndarray]:
    with h5py.File(path, "r") as handle:
        root = handle["spikes"]
        if set(root) - {"v1"}:
            raise ValueError("POINTNET_UNKNOWN_CORTICAL_SPIKE_POPULATION")
        if "v1" in root:
            ids = core.integer_vector(root["v1/node_ids"], "spike node_ids")
            times = np.asarray(root["v1/timestamps"], dtype=np.float64)
            units = root["v1/timestamps"].attrs.get("units", "ms")
            if isinstance(units, bytes):
                units = units.decode()
            if units != "ms":
                raise ValueError("POINTNET_CORTICAL_SPIKE_UNITS_DIFFER")
        else:
            ids, times = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if (times.shape != ids.shape or not np.isfinite(times).all() or np.any(times <= 0)
            or np.any(times > V1_DURATION_MS)):
        raise ValueError("POINTNET_CORTICAL_SPIKE_VALUES_INVALID")
    positions = core.positions_for_ids(source_ids, ids)
    counts = np.bincount(positions, minlength=len(source_ids))
    order = np.lexsort((times, ids))
    return dict(source_id=ids[order], native_id=native_ids[positions][order], time_ms=times[order]), counts


def probe(config_path: Path, recurrent: Path, inputs: Path, parameters: Path,
          source_config: Path, output: Path, *, spike_inputs_archive: Path | None = None,
          clock_protocol: Path | None = None) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    external = spike_inputs_archive is not None
    failure_status = ("FAILED_POINTNET_NATIVE_TEMPORAL_SENSITIVITY" if clock_protocol is not None else
                      "FAILED_POINTNET_NATIVE_VISUAL_INPUTS" if external else "FAILED_POINTNET_NATIVE_DIAGNOSTIC")
    report = dict(status=failure_status, exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid(), hostname=socket.gethostname(),
        scientific_qualification=False, source_scientific_fidelity_qualified=False, production_episodes=0,
        training_seeds_used=[], input_mode=VISUAL_MODE if external else DIAGNOSTIC_MODE,
        stimulus_kind="constructed_visual_and_background_spike_replay" if external else "diagnostic_synchronous_precise_pulses",
        visual_input_replay=external, visual_episode=False,
        observation_scope="All cortical spikes and per-unit counts including silence; initial and terminal V_m, not voltage trajectories.",
        implementation_sha256=sha256_file(Path(__file__)))
    try:
        protocol = None if clock_protocol is None else load_clock_protocol(clock_protocol)
        resolution_ms = V1_RESOLUTION_MS if protocol is None else protocol["execution_resolution_ms"]
        if protocol is not None:
            report["clock_protocol"] = dict(sha256=sha256_file(clock_protocol), resolved=protocol)
        config, snapshots = core.load_config(config_path)
        config_sha = snapshots["original_config_sha256"] or snapshots["input_config_sha256"]
        report["original_config_sha256"] = config_sha
        report["wheel"] = core.verify_wheel(config)
        rec, inp, binding = stage_network(recurrent, inputs, output, config_sha)
        report.update(mode=rec["mode"], bindings=binding)
        if any(value["wheel"]["sha256"] != report["wheel"]["sha256"] for value in (rec, inp)):
            raise ValueError("POINTNET_REALIZATION_WHEEL_CHANGED")
        bodies, binding["sources"] = bind_sources(Path(config["source_root"]), (rec, inp))
        data = {pop: node_data(output / "network", pop) for pop in ("v1", "lgn", "bkg")}
        if (data["v1"]["count"] != rec["validation"]["nodes"]
                or data["lgn"]["count"] != 17400 or data["bkg"]["count"] != 1):
            raise ValueError("POINTNET_BOUND_POPULATION_COUNTS_DIFFER")
        input_plan, spike_files = None, None
        if external:
            _, _, expected_binding = read_input_construction_cells(inputs)
            input_plan, binding["spike_inputs"], spike_files = load_spike_inputs(spike_inputs_archive, expected_binding, data)
        refs, binding["parameters"] = stage_parameters(parameters, data["v1"], output)
        native_config, binding["point_config"] = stage_point_config(source_config, output, data, bodies,
            spike_files=spike_files, resolution_ms=resolution_ms)
        expected = {pop: edge_inventory(output / "network", pop,
            rec["validation"] if pop == "v1" else inp["validation"][pop], config["edge_chunk_size"])
            for pop in ("v1", "lgn", "bkg")}
        report["source_denominators"] = expected
        clock_reference = None
        if protocol is not None:
            clock_reference, report["temporal_sensitivity"] = bind_clock_reference(protocol, report)
        (output / "native").mkdir()
        with core.original_runtime(output, Path(config["bmtk_wheel"]["path"])):
            # Optional native dependencies are imported only inside the verified runtime.
            import nest
            from bmtk.simulator import pointnet
            from bmtk.simulator.pointnet import glif_utils, pointnetwork, pointsimulator

            if nest.__version__ != NEST_VERSION:
                raise ValueError("POINTNET_NATIVE_NEST_VERSION_DIFFERS")
            for module, expected_sha in ((glif_utils, BMTK_CONVERTER_SHA256), (pointnetwork, POINTNETWORK_SHA256),
                                         (pointsimulator, POINTSIMULATOR_SHA256)):
                if sha256_file(Path(module.__file__)) != expected_sha:
                    raise ValueError("POINTNET_IMPORTED_SOURCE_CHANGED:" + module.__name__)
            report["mpi"] = core.require_serial_runtime()
            report["native_connection_admission"] = native_connection_admission(nest, next(iter(refs.values()))["converted_parameters"])
            report["native_connection_admission"].update(fixture_resolution_ms=V1_RESOLUTION_MS,
                validates_selected_execution_clock=resolution_ms == V1_RESOLUTION_MS)
            rules, changes = extract_functions(bodies["weight_semantics/legacy_pointnet_runner"].decode(), RULE_NAMES)
            if changes:
                raise ValueError("POINTNET_ORIGINAL_DIRECTION_RULES_MUST_BE_UNCHANGED")
            for name in RULE_NAMES:
                pointnet.synaptic_weight(rules[name])
            cfg = pointnet.Config.from_json(str(native_config), validate=True)
            cfg.build_env()
            graph = pointnet.PointNetwork.from_config(cfg)
            evidence = ConnectionEvidence(nest, graph, expected, output, resolution_ms=resolution_ms,
                reference_source_batch_sha256=(None if clock_reference is None else
                                               clock_reference["connections"]["source_batch_sha256"]))
            with evidence.capture():
                sim = pointnet.PointSimulator.from_config(cfg, graph)
                report["execution_clock"] = execution_clock(nest, resolution_ms)
                ids = data["v1"]["sorted_ids"]
                native_ids = np.asarray(graph.gid_map.get_nestids("v1", ids), dtype=np.int64)
                inverse, populations = graph.gid_map.get_node_ids(native_ids)
                if not np.array_equal(inverse, ids) or not np.all(populations == "v1") or len(np.unique(native_ids)) != len(ids):
                    raise ValueError("POINTNET_CORTICAL_IDENTITY_READBACK_DIFFERS")
                initial = verify_parameters(nest, data["v1"], native_ids, refs, output)
                report["refractory_clock_semantics"] = refractory_clock_semantics(refs, report["execution_clock"])
                sim.add_mod(InputReadback(nest, data, evidence, native_ids, output, report,
                                         input_plan=input_plan, resolution_ms=resolution_ms))
                report.update(nest_version=nest.__version__, kernel_rng_seed=int(nest.GetKernelStatus("rng_seed")),
                              local_num_threads=int(nest.GetKernelStatus("local_num_threads")))
                if report["local_num_threads"] != 1:
                    raise ValueError("POINTNET_ONE_NATIVE_THREAD_REQUIRED")
                if clock_reference is not None and any(report[key] != clock_reference[key]
                        for key in ("nest_version", "kernel_rng_seed", "local_num_threads")):
                    raise ValueError("POINTNET_TEMPORAL_NATIVE_RUNTIME_DIFFERS")
                report["setup_seconds"] = time.monotonic() - started
                before = time.monotonic()
                sim.run()
                report["simulation_and_output_seconds"] = time.monotonic() - before
                final = np.asarray([row[0] for row in status_rows(nest, native_ids, ["V_m"])], dtype=np.float64)
                if final.shape != initial.shape or not np.isfinite(final).all():
                    raise ValueError("POINTNET_NONFINITE_TERMINAL_CORTICAL_STATE")
                biological_time = float(nest.GetKernelStatus("biological_time"))
                if biological_time != V1_DURATION_MS:
                    raise ValueError("POINTNET_NATIVE_DURATION_DIFFERS")
                spikes, counts = collect_spikes(output / "output/spikes.h5", ids, native_ids)
                np.savez_compressed(output / "native/cortical_spikes.npz", **spikes)
                np.savez_compressed(output / "native/cortical_state.npz", source_id=ids,
                    node_type_id=data["v1"]["sorted_types"], native_id=native_ids, inverse_source_id=inverse,
                    population=np.asarray(populations, dtype="U2"), initial_vm_mV=initial, terminal_vm_mV=final,
                    spike_count=counts)
                status = TEMPORAL_PASS_STATUS if protocol is not None else VISUAL_PASS_STATUS if external else PASS_STATUS
                report.update(status=status, exit_code=0, simulated_ms=biological_time,
                    cortical_units=len(ids), realized_glif_types=len(refs), cortical_spike_events=len(spikes["time_ms"]),
                    silent_cortical_units=int(np.count_nonzero(counts == 0)),
                    native_parameters_and_initial_state_exact=True, native_nest_simulation=True,
                    array_artifacts=list(ARRAY_ARTIFACTS) + (["native/connection_source_batches.npz", "native/connection_readback.npz"]
                                                          if evidence.materialize else []))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime"] = runtime_manifest({}, list(sys.argv), report["exit_code"])
        freeze_json(output / "report.json", report)
    return report


def compare(left: Path, right: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_POINTNET_NATIVE_COMPARISON", exit_code=1,
                  scientific_qualification=False, production_episodes=0, files=[])
    try:
        a, b = [json.loads((path / "report.json").read_text()) for path in (left, right)]
        statuses = {DIAGNOSTIC_MODE: PASS_STATUS, VISUAL_MODE: VISUAL_PASS_STATUS}
        for row in (a, b):
            temporal = row.get("clock_protocol") is not None
            expected_status = TEMPORAL_PASS_STATUS if temporal else statuses.get(row.get("input_mode", DIAGNOSTIC_MODE))
            if row.get("status") != expected_status or row.get("exit_code") != 0:
                raise ValueError("POINTNET_TWO_COMPLETED_NATIVE_PROBES_REQUIRED")
            if temporal:
                validate_clock_protocol(row["clock_protocol"]["resolved"])
                if row.get("execution_clock") != dict(duration_ms=V1_DURATION_MS,
                        resolution_ms=CLOCK_PROTOCOL_FIXED["execution_resolution_ms"], tics_per_ms=1000.0,
                        tics_per_step=125):
                    raise ValueError("POINTNET_TEMPORAL_REPLAY_CLOCK_DIFFERS")
        if (a["hostname"], a["process_id"]) == (b["hostname"], b["process_id"]):
            raise ValueError("POINTNET_INDEPENDENT_NATIVE_PROCESSES_REQUIRED")
        fields = ("bindings", "original_config_sha256", "implementation_sha256", "mode", "source_denominators",
                  "nest_version", "kernel_rng_seed", "local_num_threads", "stimulus_kind", "input_readback", "connections",
                  "cortical_units", "realized_glif_types", "array_artifacts", "simulated_ms")
        differences = [field for field in fields if a[field] != b[field]]
        for field, default in (("input_mode", DIAGNOSTIC_MODE), ("input_event_schema", DIAGNOSTIC_SCHEMA),
                               ("execution_clock", None), ("clock_protocol", None),
                               ("temporal_sensitivity", None), ("refractory_clock_semantics", None)):
            if a.get(field, default) != b.get(field, default):
                differences.append(field)
        if a["wheel"]["sha256"] != b["wheel"]["sha256"]:
            differences.append("wheel")
        report["binding_or_runtime_differences"] = differences
        for relative in a["array_artifacts"]:
            with np.load(left / relative, allow_pickle=False) as x, np.load(right / relative, allow_pickle=False) as y:
                keys_equal = set(x.files) == set(y.files)
                equal = keys_equal and all(x[key].dtype == y[key].dtype and np.array_equal(x[key], y[key]) for key in x.files)
            report["files"].append(dict(path=relative, equal=bool(equal)))
        relative = "native/parameters.json"
        equal = json.loads((left / relative).read_text()) == json.loads((right / relative).read_text())
        report["files"].append(dict(path=relative, equal=equal))
        passed = not differences and all(row["equal"] for row in report["files"])
        report.update(status="POINTNET_NATIVE_REPLAY_EQUAL" if passed else "POINTNET_NATIVE_REPLAY_DIFFERENT",
            exit_code=0 if passed else 1, all_native_outputs_equal=passed, independent_processes=True,
            left_report_sha256=sha256_file(left / "report.json"), right_report_sha256=sha256_file(right / "report.json"))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("probe")
    for key in ("config", "recurrent", "inputs", "parameters", "source-config", "output"):
        run.add_argument("--" + key, type=Path, required=True)
    run.add_argument("--spike-inputs", type=Path)
    run.add_argument("--clock-protocol", type=Path)
    replay = commands.add_parser("compare")
    for key in ("left", "right", "output"):
        replay.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        report = probe(args.config, args.recurrent, args.inputs, args.parameters, args.source_config, args.output,
                       spike_inputs_archive=args.spike_inputs, clock_protocol=args.clock_protocol)
    else:
        report = compare(args.left, args.right, args.output)
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

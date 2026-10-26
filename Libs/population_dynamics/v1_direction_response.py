"""Descriptive responses of all units in eight bound V1 development conditions.

This consumes completed native archives; it never runs a simulator or promotes
source fidelity, tuning, uncertainty, or production qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_clock_comparison import (
    DURATION_MS, REFERENCE_DT_MS, SOURCE_TYPE_CSV, admit_archive, align_spikes,
    align_state, canonical_inputs, load_arrays, source_populations,
)
from .v1_source_dataset import pointnet_input_evidence

DIRECTIONS = tuple(range(0, 360, 45))
REFERENCE_DIRECTION = 315
CORTICAL_UNITS = 230924
LGN_UNITS = 17400
GRAY_END_MS = 500.0
BINDING_FIELDS = {"inputs", "parameters", "point_config", "recurrent", "sources", "spike_inputs"}
CONTEXT_FIELDS = (
    "original_config_sha256", "source_denominators", "mode", "input_mode", "input_event_schema",
    "cortical_units", "realized_glif_types", "kernel_rng_seed", "nest_version", "local_num_threads",
    "connections", "native_total_connections_including_output_devices",
)
SPATIAL_REFERENCE_FILES = ("delivery.json", "export/report.json", "export/dataset/dataset.json")
INPUT_CONSTRUCTION_FILES = {
    "primary/report.json", "primary/glif/lgn_nodes.h5", "primary/glif/lgn_node_types.csv",
    "primary/glif/bkg_nodes.h5", "primary/glif/bkg_node_types.csv",
}


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DIRECTION_DUPLICATE_JSON_KEY:" + key)
        result[key] = value
    return result


def _json(path: Path) -> dict:
    return json.loads(path.read_text(), object_pairs_hook=unique_json_object)


def _protocol(path: Path, submission_path: Path) -> tuple[dict, dict, dict, dict[int, str]]:
    protocol = yaml.safe_load(path.read_text())
    submission = _json(submission_path)
    expected = dict(native_resolution_ms=REFERENCE_DT_MS, cortical_units=CORTICAL_UNITS,
        new_directions_deg=list(DIRECTIONS[:-1]), reused_direction_deg=REFERENCE_DIRECTION,
        maximum_new_full_conditions=7, trials_per_direction=1, vary_only="stimulus.direction_deg",
        noise_pairing_claim="none", source_fidelity_qualified=False, production_eligible=False,
        formal_protocol_frozen=False)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("DIRECTION_FROZEN_PROTOCOL_DIFFERS")
    readout = protocol["readout"]
    if (any(readout.get(key) is not True for key in ("retain_all_cortical_units", "retain_silent_units",
            "retain_source_population_and_spatial_indices", "retain_terminal_events_and_voltage_separately"))
            or readout.get("descriptive_source_analysis_window_ms") != [GRAY_END_MS, DURATION_MS]
            or readout.get("source_analysis_lower_endpoint") != "open"
            or readout.get("actual_control_transition_from") != "delivered_movie_and_controls"
            or readout.get("inferential_statistics") is not False
            or readout.get("source_qualification") is not False
            or readout.get("scientific_tolerances") is not None
            or protocol["execution"].get("full_lgn_units") != LGN_UNITS):
        raise ValueError("DIRECTION_READOUT_PROTOCOL_DIFFERS")
    if (submission.get("protocol_sha256") != sha256_file(path)
            or submission.get("maximum_new_full_conditions") != 7
            or submission.get("condition_concurrency") != 1
            or submission.get("source_qualification") is not False
            or submission.get("production_episodes") != 0):
        raise ValueError("DIRECTION_SUBMISSION_PROTOCOL_BINDING_DIFFERS")
    jobs = {}
    for row in submission["jobs"]:
        if row.get("phase") != "full_native_and_replay":
            continue
        direction, job = row.get("direction_deg"), row.get("job_id")
        if (type(direction) is not int or direction not in DIRECTIONS[:-1] or direction in jobs
                or type(job) is not int or job <= 0 or row.get("returncode") != 0
                or row.get("status") != "SUBMITTED"):
            raise ValueError("DIRECTION_SUBMISSION_CONDITION_INVALID")
        jobs[direction] = str(job)
    reference_name = Path(protocol["references"]["reused_native"]).name
    if not reference_name.startswith("run_") or not reference_name[4:].isdigit():
        raise ValueError("DIRECTION_REFERENCE_JOB_ID_MISSING")
    jobs[REFERENCE_DIRECTION] = reference_name[4:]
    if set(jobs) != set(DIRECTIONS) or len(set(jobs.values())) != len(DIRECTIONS):
        raise ValueError("DIRECTION_UNIQUE_EIGHT_SUBMITTED_CONDITIONS_REQUIRED")
    # Canonical protocol and visual base are staged at their repository-relative paths.
    root = path.resolve().parents[2]
    base = yaml.safe_load((root / protocol["visual_config_base"]).read_text())
    if (base["stimulus"].get("direction_deg") != REFERENCE_DIRECTION
            or base["stimulus"].get("duration_s") != DURATION_MS / 1000
            or base["stimulus"].get("gray_screen_s") != GRAY_END_MS / 1000
            or base["stimulus"].get("contrast_percent") != 80
            or base.get("unit_coverage") != "all_source_units"
            or base["spike_generation"].get("trials") != 1):
        raise ValueError("DIRECTION_REFERENCE_VISUAL_CONFIG_DIFFERS")
    return protocol, submission, base, jobs


def _archive_paths(paths: Mapping, *, intake_only: bool) -> dict[int, Path]:
    result = {}
    for key, path in paths.items():
        if isinstance(key, bool) or str(key) not in {str(value) for value in DIRECTIONS}:
            raise ValueError("DIRECTION_ARCHIVE_KEY_INVALID:" + str(key))
        direction = int(key)
        if direction in result:
            raise ValueError("DIRECTION_DUPLICATE_ARCHIVE_CONDITION")
        result[direction] = Path(path)
    expected = {REFERENCE_DIRECTION} if intake_only else set(DIRECTIONS)
    if set(result) != expected:
        raise ValueError("DIRECTION_COMPLETE_ARCHIVE_COVERAGE_REQUIRED")
    if len({str(path.resolve()) for path in result.values()}) != len(result):
        raise ValueError("DIRECTION_DUPLICATE_ARCHIVE_PATH")
    return result


def _connection_context(report: dict) -> dict:
    connections = report["connections"]
    # The clock-sensitivity implementation added these annotations after the
    # original 315-degree run. At the source clock they restate the same batch;
    # validate them before comparing the shared physical connection record.
    annotations = dict(source_batch_resolution_ms=REFERENCE_DT_MS,
        reference_clock_resolution_ms=REFERENCE_DT_MS,
        reference_clock_source_batch_sha256=connections["source_batch_sha256"])
    for key, expected in annotations.items():
        if key in connections and connections[key] != expected:
            raise ValueError("DIRECTION_SOURCE_CLOCK_CONNECTION_ANNOTATION_DIFFERS:" + key)
    return {key: value for key, value in connections.items() if key not in annotations}


def _require_full(admitted: dict, direction: int, job: str, base: dict) -> None:
    report = admitted["report"]
    if admitted["delivery"]["job_id"] != job:
        raise ValueError("DIRECTION_DELIVERED_JOB_DIFFERS")
    clock = dict(duration_ms=DURATION_MS, resolution_ms=REFERENCE_DT_MS)
    if (report.get("mode") != "full" or report.get("cortical_units") != CORTICAL_UNITS
            or report.get("simulated_ms") != DURATION_MS
            or report.get("input_mode") != "constructed_visual_inputs"
            or report.get("input_event_schema") != "packed_source_events_v1"
            or report.get("source_scientific_fidelity_qualified") is not False
            or report.get("scientific_qualification") is not False or report.get("production_episodes") != 0
            or set(report["bindings"]) != BINDING_FIELDS
            or report["bindings"]["point_config"]["source_clock"] != clock
            or report.get("execution_clock") not in (None, dict(**clock, tics_per_ms=1000.0, tics_per_step=250))):
        raise ValueError("DIRECTION_FULL_SOURCE_CLOCK_AND_DENOMINATOR_REQUIRED")
    for key in (*CONTEXT_FIELDS, "input_readback", "cortical_spike_events", "silent_cortical_units"):
        if report[key] != admitted["replay_report"][key]:
            raise ValueError("DIRECTION_REPLAY_CONTEXT_DIFFERS:" + key)
    for key in ("sha256", "bytes", "version"):
        if report["wheel"][key] != admitted["replay_report"]["wheel"][key]:
            raise ValueError("DIRECTION_REPLAY_NATIVE_WHEEL_DIFFERS")
    _connection_context(report)
    expected_config = deepcopy(base)
    expected_config["stimulus"]["direction_deg"] = direction
    spikes = report["bindings"]["spike_inputs"]
    if (spikes.get("config") != expected_config or spikes.get("complete_lgn_population") is not True
            or spikes.get("bundle_status") != "CONSTRUCTED_VISUAL_INPUTS_COMPLETE_DEVELOPMENT"
            or spikes.get("source_clock") != clock):
        raise ValueError("DIRECTION_ONLY_DECLARED_STIMULUS_MAY_VARY")
    inputs, construction = report["bindings"]["inputs"], spikes["input_construction_binding"]
    if (construction.get("mode") != "full" or construction.get("delivery_sha256") != inputs["delivery_sha256"]
            or set(construction["files"]) != INPUT_CONSTRUCTION_FILES
            or any(record != inputs["files"].get(name) for name, record in construction["files"].items())):
        raise ValueError("DIRECTION_INPUT_CONSTRUCTION_BINDING_DIFFERS")


def _v1_population(values: dict, length: int) -> None:
    labels = np.asarray(values["population"])
    if labels.shape != (length,) or labels.dtype.kind not in "US" or not np.all(labels.astype("U") == "v1"):
        raise ValueError("DIRECTION_CORTICAL_POPULATION_INVALID")


def read_condition(admitted: dict) -> dict:
    path, report = Path(admitted["archive"]) / "primary", admitted["report"]
    raw = load_arrays(path / "native/cortical_state.npz")
    _v1_population(raw, len(raw["source_id"]))
    state = align_state(raw)
    raw_spikes = load_arrays(path / "native/cortical_spikes.npz")
    # collect_spikes emits source_id/native_id/time_ms. Its events join the
    # complete, explicitly V1 state inventory; no event population column exists.
    spikes = align_spikes(raw_spikes, state, REFERENCE_DT_MS)
    incoming = load_arrays(path / "native/input_events.npz")
    input_identity, _ = pointnet_input_evidence(report, incoming)
    inputs = canonical_inputs(incoming, report["input_event_schema"], REFERENCE_DT_MS)
    if (len(state["source_id"]) != CORTICAL_UNITS or report["cortical_units"] != CORTICAL_UNITS
            or len(np.unique(state["node_type_id"])) != report["realized_glif_types"]
            or len(spikes["time_ms"]) != report["cortical_spike_events"]
            or int(np.count_nonzero(state["spike_count"] == 0)) != report["silent_cortical_units"]
            or np.count_nonzero(inputs["population"] == "lgn") != LGN_UNITS
            or np.count_nonzero(inputs["population"] == "bkg") != 1
            or len(inputs["source_id"]) != report["input_readback"]["generators"]
            or len(inputs["requested_times_ms"]) != report["input_readback"]["events"]
            or int(np.count_nonzero(inputs["counts"] == 0)) != report["input_readback"]["silent_generators"]):
        raise ValueError("DIRECTION_NATIVE_ARRAY_DENOMINATOR_DIFFERS")
    parameters = _json(path / "native/parameters.json")
    if set(parameters) != {str(int(value)) for value in np.unique(state["node_type_id"])}:
        raise ValueError("DIRECTION_PHYSICAL_PARAMETER_TYPE_INVENTORY_DIFFERS")
    for identity, row in parameters.items():
        selected = state["node_type_id"] == int(identity)
        if (row.get("all_units_exact") is not True or row.get("units") != int(selected.sum())
                or not np.all(state["initial_vm_mV"][selected] == row["parameters"]["V_m"])):
            raise ValueError("DIRECTION_INITIAL_PHYSICAL_PARAMETER_DIFFERS")
    names, codes = source_populations(Path(admitted["archive"]) / SOURCE_TYPE_CSV, state["node_type_id"])
    if len(names) != 21:
        raise ValueError("DIRECTION_ALL_21_SOURCE_POPULATIONS_REQUIRED")
    gray = np.bincount(spikes["unit_index"][spikes["time_ms"] <= GRAY_END_MS], minlength=CORTICAL_UNITS)
    grating = np.bincount(spikes["unit_index"][spikes["time_ms"] > GRAY_END_MS], minlength=CORTICAL_UNITS)
    terminal = np.bincount(spikes["unit_index"][spikes["time_ms"] == DURATION_MS], minlength=CORTICAL_UNITS)
    if not np.array_equal(gray + grating, state["spike_count"]):
        raise ValueError("DIRECTION_ANALYSIS_WINDOW_COUNT_DIFFERS")
    return dict(state=state, parameters=parameters, population_names=names, population_index=codes,
        input_inventory={key: inputs[key] for key in ("population", "source_id")}, input_identity_sha256=input_identity,
        full_spike_count=state["spike_count"], gray_spike_count=gray, grating_spike_count=grating,
        terminal_event_count=terminal, terminal_vm_mV=state["terminal_vm_mV"])


def _same_context(reference: dict, current: dict, first: dict, second: dict) -> None:
    a, b = reference["report"], current["report"]
    for key in CONTEXT_FIELDS:
        left, right = (_connection_context(a), _connection_context(b)) if key == "connections" else (a[key], b[key])
        if left != right:
            raise ValueError("DIRECTION_IMMUTABLE_NATIVE_CONTEXT_DIFFERS:" + key)
    for key in BINDING_FIELDS - {"spike_inputs"}:
        if a["bindings"][key] != b["bindings"][key]:
            raise ValueError("DIRECTION_IMMUTABLE_NATIVE_BINDING_DIFFERS:" + key)
    for key in ("sha256", "bytes", "version"):
        if a["wheel"][key] != b["wheel"][key]:
            raise ValueError("DIRECTION_NATIVE_WHEEL_DIFFERS")
    for key in ("source_id", "node_type_id", "native_id", "inverse_source_id", "initial_vm_mV"):
        if not np.array_equal(first["state"][key], second["state"][key]):
            raise ValueError("DIRECTION_ALL_UNIT_INITIAL_IDENTITY_DIFFERS:" + key)
    for key in ("population_names", "population_index"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError("DIRECTION_SOURCE_POPULATION_INVENTORY_DIFFERS")
    for key in ("population", "source_id"):
        if not np.array_equal(first["input_inventory"][key], second["input_inventory"][key]):
            raise ValueError("DIRECTION_INPUT_SOURCE_INVENTORY_DIFFERS")
    if first["parameters"] != second["parameters"]:
        raise ValueError("DIRECTION_PHYSICAL_PARAMETERS_DIFFER")


def _spatial_reference(protocol_path: Path, protocol: dict, reference: dict) -> dict:
    relative = Path(protocol["references"]["reused_observation"])
    directory = protocol_path.resolve().parents[2] / relative
    delivery = _json(directory / "delivery.json")
    records = {row["path"]: row for row in delivery["files"]}
    if delivery.get("exit_code") != 0 or len(records) != len(delivery["files"]):
        raise ValueError("DIRECTION_SPATIAL_REFERENCE_DELIVERY_INVALID")
    for name in SPATIAL_REFERENCE_FILES[1:]:
        record = records[name]
        if (record["bytes"] != (directory / name).stat().st_size
                or record["sha256"] != sha256_file(directory / name)):
            raise ValueError("DIRECTION_SPATIAL_REFERENCE_CHANGED:" + name)
    exported, catalog = [_json(directory / name) for name in SPATIAL_REFERENCE_FILES[1:]]
    if (exported.get("status") != "EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION"
            or exported.get("exit_code") != 0 or exported.get("source_execution_mode") != "full"
            or exported.get("native_job_id") != reference["delivery"]["job_id"]
            or exported.get("source_units") != CORTICAL_UNITS or exported.get("coordinate_units") != "um"
            or exported.get("dataset_catalog_sha256") != records["export/dataset/dataset.json"]["sha256"]
            or catalog["binding"]["source_bindings"] != [dict(job_id=reference["delivery"]["job_id"],
                delivery_sha256=reference["consumed"]["delivery.json"]["sha256"])]
            or set(catalog["circuits"]) != {exported["circuit_id"]}):
        raise ValueError("DIRECTION_SPATIAL_REFERENCE_NATIVE_BINDING_DIFFERS")
    circuit = catalog["circuits"][exported["circuit_id"]]
    graph = reference["report"]["bindings"]["recurrent"]["files"]
    if (circuit["units"] != CORTICAL_UNITS
            or circuit["provenance"]["unit_metadata_identity"] != graph["primary/glif/v1_nodes.h5"]["sha256"]
            or circuit["provenance"]["node_types_identity"] != graph["primary/glif/v1_node_types.csv"]["sha256"]
            or not circuit["provenance"].get("coordinate_unit_evidence")
            or Path(circuit["path"]).is_absolute() or ".." in Path(circuit["path"]).parts):
        raise ValueError("DIRECTION_SPATIAL_REFERENCE_GRAPH_DIFFERS")
    return dict(reference_archive=str(relative), reference_delivery_sha256=sha256_file(directory / "delivery.json"),
        dataset_catalog_sha256=records["export/dataset/dataset.json"]["sha256"], circuit_id=exported["circuit_id"],
        circuit_h5_path=str(relative / "export/dataset" / circuit["path"]), circuit_record=circuit,
        coordinate_units=exported["coordinate_units"], join="source_population=v1 and source_id equals static unit_ids",
        coordinate_array_rechecked=False,
        semantics="Reuse the delivered source coordinate map and its original coordinate meaning; no inferred soma or electrode positions.")


def _summary_rows(direction: int, data: dict) -> list[dict]:
    names, codes = data["population_names"], data["population_index"]
    rows = []
    for code, name in [(-1, "all")] + list(enumerate(names)):
        selected = np.ones(len(codes), dtype=bool) if code == -1 else codes == code
        n = int(selected.sum())
        row = dict(direction_deg=direction, population_index=code, population=str(name), units=n)
        for window, seconds in (("full", 3.0), ("gray", 0.5), ("grating", 2.5)):
            counts = data[window + "_spike_count"][selected]
            total = int(counts.sum())
            row.update({window + "_spikes": total, window + "_silent_units": int(np.count_nonzero(counts == 0)),
                        window + "_mean_rate_hz": total / (n * seconds) if n else None})
        volts = data["terminal_vm_mV"][selected]
        row.update(terminal_spikes=int(data["terminal_event_count"][selected].sum()),
            terminal_vm_mV_mean=float(volts.mean()) if n else None,
            terminal_vm_mV_minimum=float(volts.min()) if n else None,
            terminal_vm_mV_maximum=float(volts.max()) if n else None)
        rows.append(row)
    return rows


def summarize(protocol_path: Path, submission_path: Path, archive_paths: Mapping, output: Path,
              *, intake_only: bool = False) -> dict:
    """Read staged archives; intake-only validates 315 degrees without eight-condition completion."""
    protocol_path, submission_path, output = map(Path, (protocol_path, submission_path, output))
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {}
    report = dict(status="FAILED_V1_DIRECTION_RESPONSE", exit_code=1,
        started_utc=datetime.now(timezone.utc).isoformat(), implementation_sha256=sha256_file(Path(__file__)),
        complete_eight_direction_coverage=False, scientific_qualification=False, source_fidelity_qualified=False,
        production_eligible=False, formal_protocol_frozen=False, production_episodes=0, inferential_statistics=False,
        independent_circuit_replicates=1, trials_per_direction=1, noise_pairing_claim="none", training_seeds_used=[],
        conditions=[], failures=[])
    try:
        protocol, submission, base, jobs = _protocol(protocol_path, submission_path)
        paths = _archive_paths(archive_paths, intake_only=intake_only)
        report.update(protocol_sha256=sha256_file(protocol_path), submission_sha256=sha256_file(submission_path),
            expected_directions_deg=list(DIRECTIONS), requested_directions_deg=sorted(paths),
            previous_failure_evidence={key: submission[key] for key in
                ("supersedes_failed_submission", "failure_evidence") if key in submission})
        accepted, reference, first = {}, None, None
        for direction in [REFERENCE_DIRECTION] + sorted(set(paths) - {REFERENCE_DIRECTION}):
            record = dict(direction_deg=direction, expected_job_id=jobs[direction], archive=str(paths[direction]),
                          status="FAILED")
            report["conditions"].append(record)
            try:
                admitted = admit_archive(paths[direction], fine=False)
                record.update(job_id=admitted["delivery"]["job_id"], consumed=admitted["consumed"])
                _require_full(admitted, direction, jobs[direction], base)
                data = read_condition(admitted)
                if direction != REFERENCE_DIRECTION:
                    if reference is None:
                        raise ValueError("DIRECTION_REFERENCE_CONDITION_NOT_ADMITTED")
                    _same_context(reference, admitted, first, data)
                record.update(status="ADMITTED", input_identity_sha256=data["input_identity_sha256"],
                    execution_timing={phase: {key: admitted[phase][key] for key in
                        ("hostname", "setup_seconds", "simulation_and_output_seconds", "elapsed_seconds")}
                        for phase in ("report", "replay_report")})
                accepted[direction] = data
                if direction == REFERENCE_DIRECTION:
                    reference, first = admitted, data
            except Exception as error:
                record.update(error_type=type(error).__name__, error=str(error))
                report["failures"].append(dict(record))
        report["admitted_conditions"] = len(accepted)
        if report["failures"]:
            raise ValueError("DIRECTION_REQUIRED_CONDITIONS_FAILED:" + ",".join(str(row["direction_deg"]) for row in report["failures"]))
        report["spatial_mapping"] = _spatial_reference(protocol_path, protocol, reference)
        directions = sorted(accepted)
        arrays = {key: np.stack([accepted[direction][key] for direction in directions]) for key in
                  ("full_spike_count", "gray_spike_count", "grating_spike_count", "terminal_event_count", "terminal_vm_mV")}
        for window, seconds in (("full", 3.0), ("gray", 0.5), ("grating", 2.5)):
            arrays[window + "_rate_hz"] = arrays[window + "_spike_count"] / seconds
        np.savez_compressed(output / "per_unit.npz", direction_deg=np.asarray(directions, dtype=np.int64),
            source_id=first["state"]["source_id"], node_type_id=first["state"]["node_type_id"],
            native_id=first["state"]["native_id"], source_population=np.asarray("v1"),
            spatial_source_id=first["state"]["source_id"], population_names=first["population_names"],
            population_index=first["population_index"], initial_vm_mV=first["state"]["initial_vm_mV"],
            terminal_event_time_ms=np.asarray(DURATION_MS), **arrays)
        rows = [row for direction in directions for row in _summary_rows(direction, accepted[direction])]
        with (output / "population_summary.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        report.update(status="V1_DIRECTION_INTAKE_ONLY" if intake_only else "V1_DIRECTION_RESPONSE_COMPLETE_DEVELOPMENT",
            exit_code=0, complete_eight_direction_coverage=not intake_only, cortical_units=CORTICAL_UNITS,
            native_resolution_ms=REFERENCE_DT_MS, duration_ms=DURATION_MS, all_units=[row for row in rows if row["population"] == "all"],
            source_population_authority=SOURCE_TYPE_CSV, source_populations=first["population_names"].tolist(),
            windows=dict(gray="[0,500] ms", grating="(500,3000] ms", full="[0,3000] ms",
                terminal="Events at 3000 ms remain in full/grating counts and are also retained separately.",
                interpretation="Source-analysis windows; actual control transitions are determined by delivered movie and controls."),
            control_time_authority="delivered_movie_and_controls", reused_reference_controls=protocol["references"]["reused_controls"],
            rate_units="spikes per second per source unit", voltage_observation="Initial and terminal V_m in mV; no voltage trajectories.",
            inference_scope="One exposed constructed circuit, one trial per direction; no OSI/DSI, inference, tuning qualification, or independent-noise comparison.",
            timing_interpretation="Archived measured setup, simulation/output and elapsed times; simulation/output includes initialization and connection verification. No speedup is inferred.")
        report["artifacts"] = {name: dict(bytes=(output / name).stat().st_size, sha256=sha256_file(output / name))
                               for name in ("per_unit.npz", "population_summary.csv")}
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(analysis_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat(),
                      runtime=runtime_manifest(protocol, list(sys.argv), report["exit_code"]))
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "submission", "archives-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--intake-only", action="store_true", help="Validate only reused 315 degrees; never completes the eight-direction readout.")
    args = parser.parse_args(argv)
    report = summarize(args.protocol, args.submission, _json(args.archives_json), args.output, intake_only=args.intake_only)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "admitted_conditions", "complete_eight_direction_coverage")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Export source-input GLIF CUDA blocks through the shared circuit archive.

Circuit metadata and control witnesses retain their existing source identities.
GPU executions keep their own provenance and do not claim native NEST execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode
from .provenance import freeze_json, sha256_file
from .v1_glif_component_reference import require
from .v1_glif_network import CELL_IDENTITY_FIELDS, CircuitIdentity, positions, source_input_events
from .v1_pointnet_native import bound_files, load_spike_inputs
from .v1_source_dataset import (constructed_coordinate_authority, metadata_units,
    pointnet_circuit_identity, pointnet_input_evidence, pointnet_case_metadata,
    prepare_spikes, verify_coordinate_preservation, verify_export)
from .v1_visual_controls import load_visual_controls

BLOCK_PASS = "PASS_COMPLETE_TEN_CONDITION_SOURCE_INPUT_GLIF_CUDA_BLOCK"
CASE_PASS = "PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION"
SOURCE_ARRAYS = ("input_events.npz", "cortical_state.npz", "cortical_spikes.npz")


def deliver(execution: Path) -> dict:
    """Record completed producer files for the existing authenticated readers."""
    require(bool(os.environ.get("SLURM_JOB_ID")), "GLIF_DELIVERY_COMPUTE_REQUIRED")
    root = execution/"coupled"
    report = json.loads((root/"report.json").read_text())
    require(report["status"] in (BLOCK_PASS, CASE_PASS), "GLIF_DELIVERY_PASSED_EXECUTION_REQUIRED")
    directories = [root]
    names = ["coupled/report.json", "coupled/v1_glif_network.py"]
    if report["status"] == BLOCK_PASS:
        require(report["passed_conditions"] == 10 and report["failed_conditions"] == 0
                and len(report["cases"]) == 10, "GLIF_DELIVERY_COMPLETE_BLOCK_REQUIRED")
        names.append("coupled/cases.json")
        directories = []
        for index, row in enumerate(report["cases"]):
            require(row["condition_index"] == index and row["status"] == CASE_PASS
                    and row["report"] == f"condition_{index:02d}/report.json",
                    "GLIF_DELIVERY_CONDITION_PATH_DIFFERS")
            directories.append(root/f"condition_{index:02d}")
    for directory in directories:
        case = json.loads((directory/"report.json").read_text())
        require(case["status"] == CASE_PASS and case["job_id"] == report["job_id"],
                "GLIF_DELIVERY_CONDITION_EXECUTION_DIFFERS")
        for name in ("report.json", "v1_glif_network.py", *SOURCE_ARRAYS):
            relative = (directory/name).relative_to(execution).as_posix()
            if relative not in names:
                names.append(relative)
    result = dict(exit_code=0, job_id=report["job_id"],
        delivery_job_id=os.environ["SLURM_JOB_ID"], scope="development_source_backend_execution",
        files=[dict(path=name, bytes=(execution/name).stat().st_size,
                    sha256=sha256_file(execution/name)) for name in names])
    freeze_json(execution/"delivery.json", result)
    return result


def circuit_source(archive: Path, authority_path: Path):
    relatives = ("primary/report.json", "primary/native/parameters.json",
        "primary/native/cortical_state.npz", "primary/native/input_events.npz",
        "primary/network/v1_nodes.h5", "primary/network/v1_node_types.csv",
        "recurrent/primary/biophysical/v1_nodes.h5")
    _, binding = bound_files(archive, relatives)
    original = json.loads((archive/"primary/report.json").read_text())
    require(original["exit_code"] == 0 and original["native_parameters_and_initial_state_exact"],
            "GLIF_EXPORT_CIRCUIT_REFERENCE_REQUIRED")
    parameters = json.loads((archive/"primary/native/parameters.json").read_text())
    authority = constructed_coordinate_authority(authority_path, original)
    network = {}
    for population in ("v1", "lgn", "bkg"):
        role = "recurrent" if population == "v1" else "inputs"
        for suffix in ("nodes.h5", "node_types.csv", "v1_edges.h5", "v1_edge_types.csv"):
            name = population+"_"+suffix
            network[name] = original["bindings"][role]["files"]["primary/glif/"+name]
    for name in ("v1_nodes.h5", "v1_node_types.csv"):
        require(binding["files"]["primary/network/"+name] == network[name],
                "GLIF_EXPORT_SOURCE_GEOMETRY_DIFFERS")
    circuit_id, revision, provenance = pointnet_circuit_identity(original, network, parameters)
    provenance.update(unit_metadata_identity=network["v1_nodes.h5"]["sha256"],
        node_types_identity=network["v1_node_types.csv"]["sha256"],
        coordinate_unit_evidence=sha256_file(authority_path/"authority.json"))
    with np.load(archive/"primary/native/cortical_state.npz", allow_pickle=False) as source:
        state = {key: source[key] for key in CELL_IDENTITY_FIELDS}
    with np.load(archive/"primary/native/input_events.npz", allow_pickle=False) as source:
        inputs = {key: source[key] for key in ("population", "source_id", "native_id",
                                             "inverse_source_id", "precise_times")}
    units, _ = metadata_units(archive/"primary/network/v1_nodes.h5",
        archive/"primary/network/v1_node_types.csv", state, revision, provenance,
        authority["coordinate_semantics"], circuit_id=circuit_id,
        allowed_locations=tuple(authority["cortical_locations"]))
    bio = "recurrent/primary/biophysical/v1_nodes.h5"
    require(binding["files"][bio] == original["bindings"]["recurrent"]["files"]["primary/biophysical/v1_nodes.h5"],
            "GLIF_EXPORT_BIOPHYSICAL_GEOMETRY_DIFFERS")
    geometry = verify_coordinate_preservation(units, archive/bio)
    identity = CircuitIdentity.capture(original, dict(cortical_state=state, input_events=inputs), binding)
    return original, binding, identity, units, geometry


def episode_from_source(execution: dict, arrays: dict, original: dict, identity: CircuitIdentity,
                        units, controls_path: Path, source_binding: dict, geometry: dict):
    require(execution["status"] == CASE_PASS and execution["units"] == len(units.unit_ids)
            and execution["execution_clock"] == identity.clock
            and execution["connection_rows"] == identity.connection_rows,
            "GLIF_EXPORT_EXECUTION_SCALE_OR_CLOCK_DIFFERS")
    state = arrays["cortical_state.npz"]
    spikes = arrays["cortical_spikes.npz"]
    incoming = arrays["input_events.npz"]
    for key in CELL_IDENTITY_FIELDS:
        require(np.array_equal(state[key], identity.cells[key]), "GLIF_EXPORT_CELL_IDENTITY_DIFFERS:"+key)
    terminal = state["terminal_vm_mV"]
    require(terminal.shape == units.unit_ids.shape and terminal.dtype.kind == "f"
            and np.isfinite(terminal).all() and state["spike_count"].dtype.kind in "iu",
            "GLIF_EXPORT_INVALID_TERMINAL_STATE")
    inventories = {name: {"sorted_ids": identity.inputs["source_id"][
        identity.inputs["population"] == name]} for name in ("lgn", "bkg")}
    plan, supplied, _ = load_spike_inputs(Path(execution["input_archive"]),
        original["bindings"]["spike_inputs"]["input_construction_binding"], inventories)
    require(supplied == execution["input_bindings"], "GLIF_EXPORT_INPUT_BINDING_DIFFERS")
    reconstructed = source_input_events(plan, identity)
    require(set(incoming) == set(reconstructed)
            and all(np.array_equal(incoming[key], value) for key, value in reconstructed.items()),
            "GLIF_EXPORT_SOURCE_INPUT_ENCODING_DIFFERS")
    context = dict(input_mode="constructed_visual_inputs", input_event_schema="packed_source_events_v1",
        kernel_rng_seed=original["kernel_rng_seed"],
        bindings=dict(spike_inputs=supplied),
        input_readback=dict(generators=len(incoming["source_id"]),
            events=len(incoming["source_event_unit"]), all_generator_events_exact=True,
            all_population_and_source_ids_exact=True, tics_per_ms=identity.clock["tics_per_ms"]))
    # These shared helpers verify input encoding and control binding. The context
    # contains no NEST-execution claim and is not published as a native receipt.
    input_identity, private = pointnet_input_evidence(context, incoming)
    controls = load_visual_controls(controls_path, context, native_family="pointnet")
    clock = dict(duration_ms=identity.clock["duration_ms"], resolution_ms=identity.clock["resolution_ms"])
    duration = int(clock["duration_ms"]/clock["resolution_ms"])
    event_units = positions(state["native_id"], spikes["native_id"])
    activity = dict(spike_time_ms=spikes["time_ms"], spike_source_id=state["source_id"][event_units],
                    spike_counts=state["spike_count"])
    ticks, indices = prepare_spikes(activity, units, duration, clock["resolution_ms"])
    require(len(ticks) == execution["candidate_spikes"], "GLIF_EXPORT_EVENT_COUNT_DIFFERS")
    case_id, block_id, seeds, initial_identity = pointnet_case_metadata(
        context, units.circuit_id, state["initial_vm_mV"], input_identity, clock)
    endpoint = ticks == duration
    private.update(initial_vm_mV=state["initial_vm_mV"], terminal_vm_mV=state["terminal_vm_mV"],
        final_endpoint_ticks=ticks[endpoint], final_endpoint_unit_indices=indices[endpoint])
    execution_reports = [value["sha256"] for name,value in source_binding["files"].items() if name.endswith("/report.json")]
    require(len(execution_reports) == 1, "GLIF_EXPORT_ONE_EXECUTION_REPORT_REQUIRED")
    source_record = dict(complete_unit_recording=True, simulation_completed=True,
        recorded_until_tick=duration, expected_duration_ticks=duration, core_unit_count=len(units.unit_ids),
        source_revision=units.source_revision, circuit_provenance=units.circuit_provenance,
        execution_backend="source_derived_float64_cuda", native_nest_simulation=False,
        input_mode="constructed_visual_inputs", input_event_schema="packed_source_events_v1",
        execution_binding=source_binding, cuda_execution=execution["cuda_execution"],
        execution_report_sha256=execution_reports[0],
        circuit_reference_report_sha256=execution["circuit_bindings"]["files"]["primary/report.json"]["sha256"],
        upstream_bindings=dict(recurrent=original["bindings"]["recurrent"],
                               inputs=original["bindings"]["inputs"], spike_inputs=supplied),
        input_identity_sha256=input_identity, initial_vm_identity=initial_identity,
        coordinate_preservation=geometry, visual_commands=controls["source_control_record"],
        native_connections=original["connections"], source_denominators=original["source_denominators"],
        random_stream_semantics="LGN/background streams are realized inputs. Native kernel seed labels the source reset reference; CUDA dynamics use no random draws.",
        ordinary_events=int((~endpoint).sum()), final_endpoint_events=int(endpoint.sum()),
        endpoint_semantics="Raw terminal events retained; half-open model windows exclude the terminal tick.",
        scientific_qualification=False, production_episodes=0)
    return CircuitEpisode(case_id, units.circuit_id, block_id, "development", duration,
        clock["resolution_ms"], ticks, indices, controls["control_ticks"], controls["control_values"],
        controls["control_features"], seeds=seeds, source_record=source_record,
        privileged=private, task_outcome="not_scientifically_assessed")


def export_block(execution: Path, controls_file: Path, authority: Path, output: Path) -> dict:
    require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
            "GLIF_EXPORT_FRESH_COMPUTE_OUTPUT_REQUIRED")
    _, block_binding = bound_files(execution, ("coupled/report.json", "coupled/cases.json"))
    block = json.loads((execution/"coupled/report.json").read_text())
    require(block["status"] == BLOCK_PASS and block["passed_conditions"] == 10
            and block["failed_conditions"] == 0 and len(block["cases"]) == 10,
            "GLIF_EXPORT_COMPLETE_SOURCE_BLOCK_REQUIRED")
    controls = json.loads(controls_file.read_text())["cases"]
    require([row["condition_index"] for row in controls] == list(range(10)), "GLIF_EXPORT_COMPLETE_CONTROLS_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_GLIF_SOURCE_BLOCK_EXPORT", exit_code=1, complete_episodes=0,
        production_episodes=0, scientific_qualification=False, execution=str(execution),
        execution_binding=block_binding, controls_file=str(controls_file), controls_sha256=sha256_file(controls_file))
    started = time.monotonic()
    try:
        episodes = []
        original = identity = units = geometry = circuit_binding = None
        circuit_archive = None
        for index, row in enumerate(block["cases"]):
            require(row["condition_index"] == index and row["report"] == f"condition_{index:02d}/report.json",
                    "GLIF_EXPORT_CONDITION_PATH_DIFFERS")
            prefix = f"coupled/condition_{index:02d}/"
            _, binding = bound_files(execution, tuple(prefix+name for name in ("report.json", *SOURCE_ARRAYS)))
            case = json.loads((execution/(prefix+"report.json")).read_text())
            if circuit_archive is None:
                circuit_archive = Path(case["circuit_reference"])
                original, circuit_binding, identity, units, geometry = circuit_source(circuit_archive, authority)
            require(Path(case["circuit_reference"]) == circuit_archive
                    and case["circuit_bindings"]["delivery_sha256"] == circuit_binding["delivery_sha256"],
                    "GLIF_EXPORT_MIXED_CIRCUIT_BLOCK")
            arrays = {}
            for name in SOURCE_ARRAYS:
                with np.load(execution/(prefix+name), allow_pickle=False) as source:
                    arrays[name] = {key: source[key] for key in source.files}
            episodes.append(episode_from_source(case, arrays, original, identity, units,
                Path(controls[index]["control_archive"]), binding, geometry))
        writer_binding = dict(schema=SCHEMA, scenario_id="v1_visual", scope="source_reproduction",
            sample_unit="complete_mechanistic_episode", source_revision=units.source_revision,
            execution_backend="source_derived_float64_cuda", execution_binding=block_binding,
            exporter_sha256=sha256_file(Path(__file__)))
        with CircuitArchiveWriter(output/"dataset", writer_binding) as writer:
            writer.register_circuit(units)
            for episode in episodes:
                writer.add(episode)
        dataset = CircuitDataset(output/"dataset", splits=("development",))
        verification = [verify_export(dataset, episode, units) for episode in episodes]
        report.update(status="EXPORTED_VERIFIED_GLIF_SOURCE_BLOCK", exit_code=0,
            complete_episodes=len(episodes), circuit_id=units.circuit_id, units=len(units.unit_ids),
            total_events=sum(len(episode.spike_ticks) for episode in episodes),
            episode_ids=[episode.episode_id for episode in episodes], verification=verification,
            dataset_catalog_sha256=sha256_file(output/"dataset/dataset.json"))
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deliver", "export"))
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "deliver":
        result = deliver(args.execution)
    else:
        if any(value is None for value in (args.controls, args.authority, args.output)):
            parser.error("export requires --controls, --authority and --output")
        result = export_block(args.execution, args.controls, args.authority, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

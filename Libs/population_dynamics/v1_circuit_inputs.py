"""Build source LGN/background SONATA inputs against a frozen recurrent realization."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
from datetime import datetime, timezone

import h5py
import numpy as np

from . import v1_circuit_build as core
from .provenance import cpu_execution_provenance, freeze_json, runtime_manifest, sha256_file
from .v1_circuit_source import read_source
from .v1_source_conversion import query_value


INPUT_SOURCES = {
    "original_lgn_node_parameters": "build_files/biophys_props/lgn_node_models.json",
    "original_lgn_edge_parameters": "build_files/biophys_props/lgn_v1_edge_models.csv",
    "original_bkg_edge_parameters": "build_files/biophys_props/bkg_v1_edge_models.csv",
    "original_glif_lgn_parameters": "glif_props/lgn_v1_edge_types.csv",
    "original_glif_bkg_parameters": "glif_props/bkg_v1_edge_types.csv",
}
REFERENCE_FILES = tuple(f"{kind}/{name}" for kind in ("biophysical", "glif")
                        for name in ("v1_nodes.h5", "v1_node_types.csv"))
NATIVE_FILES = tuple(f"{kind}/{population}_{suffix}" for kind in ("biophysical", "glif")
                    for population in ("lgn", "bkg")
                    for suffix in ("nodes.h5", "node_types.csv", "v1_edges.h5", "v1_edge_types.csv"))
DUAL_MODELS = {"sONsOFF_001", "sONtOFF_001"}
DOM_FIELDS = ("kpeaks_dom_0", "kpeaks_dom_1", "weight_dom_0", "weight_dom_1", "delay_dom_0", "delay_dom_1")
OPTIONAL_FIELDS = ("kpeaks_non_dom_0", "kpeaks_non_dom_1", "weight_non_dom_0", "weight_non_dom_1",
                   "delay_non_dom_0", "delay_non_dom_1", "tuning_angle", "sf_sep")


def bind_reference(archive: Path, output: Path) -> tuple[dict, dict]:
    delivery_path = archive / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    if delivery.get("exit_code") != 0 or not delivery.get("status", "").startswith("PASS_SOURCE_RECURRENT_"):
        raise ValueError("V1_INPUT_PASSED_RECURRENT_DELIVERY_REQUIRED")
    files = {}
    for relative in ("report.json", *REFERENCE_FILES):
        key = "primary/" + relative
        matches = [row for row in delivery["files"] if row["path"] == key]
        if len(matches) != 1:
            raise ValueError("V1_INPUT_UNIQUE_REFERENCE_DELIVERY_ENTRY_REQUIRED:" + key)
        row, path = matches[0], archive / key
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("V1_INPUT_REFERENCE_DELIVERY_CHANGED:" + key)
        files[relative] = dict(bytes=row["bytes"], sha256=row["sha256"])
    primary = json.loads((archive / "primary/report.json").read_text())
    mode = primary.get("mode")
    expected = {"canary": "ORIGINAL_RECURRENT_CANARY_COMPLETE", "full": "ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE"}
    if mode not in expected or primary.get("status") != expected[mode] or primary.get("exit_code") != 0 or delivery.get("mode") != mode:
        raise ValueError("V1_INPUT_COMPLETED_RECURRENT_REALIZATION_REQUIRED")
    for relative in REFERENCE_FILES:
        target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive / "primary" / relative, target)
    return primary, dict(archive=str(archive.resolve()), delivery_sha256=sha256_file(delivery_path), files=files,
                         mode=mode, numpy_checkpoint="builder_after_build", recurrent_edges_read=False)


def restore_numpy_state(state: dict) -> None:
    keys = state.get("keys", [])
    if (state.get("algorithm") != "MT19937" or len(keys) != 624
            or any(type(value) is not int or not 0 <= value <= np.iinfo(np.uint32).max for value in keys)
            or type(state.get("position")) is not int or not 0 <= state["position"] <= 624
            or state.get("has_gauss") not in (0, 1) or not np.isfinite(state.get("cached_gaussian", np.nan))):
        raise ValueError("V1_INPUT_VALID_MT19937_CHECKPOINT_REQUIRED")
    np.random.set_state((state["algorithm"], np.asarray(keys, dtype=np.uint32), state["position"],
                         state["has_gauss"], state["cached_gaussian"]))
    if core.rng_state() != state:
        raise ValueError("V1_INPUT_MT19937_RESTORE_DIFFERS")


def fit_selection(source: str, models: dict, files: dict) -> dict:
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_filter_temporal_params")
    predicates = [node.test for node in ast.walk(function) if isinstance(node, ast.If)
                  and {"pkl_savename", "tf_str", "cell_type"}.issubset(
                      {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)})]
    if len(predicates) != 1:
        raise ValueError("V1_INPUT_UNIQUE_ORIGINAL_FIT_SELECTION_PREDICATE_REQUIRED")
    predicate = compile(ast.Expression(predicates[0]), "<original-fit-selection>", "eval")
    fixed = []
    for node in ast.walk(function):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open_pickle"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call)):
            literals = [arg.value for arg in node.args[0].args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            fixed.extend(name for name in literals if name.endswith(".pkl"))
    if len(fixed) != 3 or any(name not in files for name in fixed):
        raise ValueError("V1_INPUT_REQUIRED_FIXED_FIT_MISSING")
    selections = {}
    for model in models:
        if model in DUAL_MODELS:
            selections[model] = dict(fixed_files=fixed)
            continue
        cell_type, tf_str = model.split("_", 1)
        matches = [name for name in files if eval(predicate, {"__builtins__": {}},
                   dict(pkl_savename="build_files/biophys_props/lgn_fitted_models/" + name,
                        tf_str=tf_str, cell_type=cell_type))]
        if not matches:
            raise ValueError("V1_INPUT_MODEL_FIT_MISSING:" + model)
        if len({files[name]["sha256"] for name in matches}) != 1:
            raise ValueError("V1_INPUT_AMBIGUOUS_NONIDENTICAL_FITS:" + model)
        selections[model] = dict(matches=matches, duplicate_matches_byte_equal=True,
                                 selected_value_sha256=files[matches[0]]["sha256"])
    return dict(predicate_ast=ast.dump(predicates[0], include_attributes=False), fixed_files=fixed, models=selections)


def stage_input_sources(source_root: Path, output: Path, node_source: str) -> tuple[dict, dict, dict]:
    records = {}
    for role, relative in INPUT_SOURCES.items():
        body, receipt = read_source(source_root / role)
        path = output / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
        records[role] = dict(original_sha256=receipt["sha256"], original_bytes=len(body),
                            receipt_sha256=sha256_file(source_root / role / "source.json"), staged_path=relative)
    models = json.loads((output / INPUT_SOURCES["original_lgn_node_parameters"]).read_text())
    if len(models) != 14 or sum(row["N"] * 15 * 10 for row in models.values()) != 17400:
        raise ValueError("V1_INPUT_ORIGINAL_14_TYPE_17400_LGN_REQUIRED")
    directory = source_root / "lgn_fitted_models_archive"
    source = json.loads((directory / "source.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if report.get("status") != "LGN_FITTED_ARCHIVE_ACQUIRED" or any(report.get(key) != source.get(key) for key in ("bytes", "sha256")):
        raise ValueError("V1_INPUT_COMPLETE_FITTED_ARCHIVE_RECEIPT_REQUIRED")
    rows = report.get("fitted_files", [])
    if len(rows) != 13 or len({row["path"] for row in rows}) != 13:
        raise ValueError("V1_INPUT_THIRTEEN_EXPLICIT_FITTED_FILES_REQUIRED")
    members = {row["name"]: row for row in report["members"]}
    destination = output / "build_files/biophys_props/lgn_fitted_models"; destination.mkdir()
    files = {}
    for row in rows:
        relative = Path(row["path"])
        if relative.parts != ("fits", relative.name) or relative.suffix != ".pkl" or relative.name not in members:
            raise ValueError("V1_INPUT_FIT_MEMBER_IDENTITY_INVALID")
        path = directory / relative
        if path.stat().st_size != row["bytes"] or members[relative.name]["bytes"] != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("V1_INPUT_FITTED_FILE_CHANGED:" + str(relative))
        shutil.copy2(path, destination / relative.name)
        files[relative.name] = dict(bytes=row["bytes"], sha256=row["sha256"])
    selection = fit_selection(node_source, models, files)
    return records, models, dict(archive_sha256=report["sha256"], files=files, selection=selection,
        source_receipt_sha256=sha256_file(directory / "source.json"), intake_report_sha256=sha256_file(directory / "report.json"),
        whole_archive_reverified=False, per_file_hashes_verified_before_source_pickle_loading=True)


def node_data(directory: Path, population: str, fields=()) -> dict:
    with h5py.File(directory / f"{population}_nodes.h5", "r") as handle:
        group = handle["nodes"][population]
        values = {key: core.integer_vector(group[key], key) for key in ("node_id", "node_type_id", "node_group_id", "node_group_index")}
        count = len(values["node_id"])
        if any(len(value) != count for value in values.values()) or len(np.unique(values["node_id"])) != count:
            raise ValueError("V1_INPUT_NODE_IDENTITIES_INVALID")
        properties = {name: np.empty(count, dtype=np.float64) for name in fields}
        for identity in np.unique(values["node_group_id"]):
            selected = values["node_group_id"] == identity
            for name in fields:
                properties[name][selected] = core.values_at(group[str(int(identity))][name], values["node_group_index"][selected])
    types = core.table_records(directory / f"{population}_node_types.csv", "node_type_id")
    if set(map(int, np.unique(values["node_type_id"]))) != set(types):
        raise ValueError("V1_INPUT_NODE_TYPE_INVENTORY_DIFFERS")
    order = np.argsort(values["node_id"])
    return dict(values=values, properties=properties, types=types, count=count,
                sorted_ids=values["node_id"][order], sorted_types=values["node_type_id"][order])


def verify_imported_target(network, directory: Path) -> dict:
    expected = node_data(directory, "v1", ("x", "y", "z", "tuning_angle"))
    nodes = list(network.nodes())
    if [node.node_id for node in nodes] != expected["values"]["node_id"].tolist():
        raise ValueError("V1_INPUT_IMPORTED_CORTICAL_NODE_ORDER_CHANGED")
    if [node["node_type_id"] for node in nodes] != expected["values"]["node_type_id"].tolist():
        raise ValueError("V1_INPUT_IMPORTED_CORTICAL_TYPE_CHANGED")
    for field, values in expected["properties"].items():
        if not core.arrays_equal(np.asarray([node[field] for node in nodes]), values):
            raise ValueError("V1_INPUT_IMPORTED_CORTICAL_PROPERTY_CHANGED:" + field)
    if any(node["pop_name"] != expected["types"][int(kind)]["pop_name"] for node, kind in zip(nodes, expected["values"]["node_type_id"])):
        raise ValueError("V1_INPUT_IMPORTED_CORTICAL_POPULATION_CHANGED")
    return dict(nodes=len(nodes), native_id_order_preserved=True, node_types_populations_and_coordinates_preserved=True)


def validate_lgn_fields(data: dict, models: dict) -> dict:
    if data["count"] != 17400 or len(data["types"]) != 14:
        raise ValueError("V1_INPUT_COMPLETE_LGN_COUNT_DIFFERS")
    counts, intentional_nan = {}, 0
    for kind, row in data["types"].items():
        population = row["pop_name"]
        if population not in models:
            raise ValueError("V1_INPUT_UNKNOWN_LGN_MODEL")
        selected = data["values"]["node_type_id"] == kind
        counts[population] = int(selected.sum())
        if counts[population] != models[population]["N"] * 150:
            raise ValueError("V1_INPUT_LGN_MODEL_COUNT_DIFFERS")
        for field in ("x", "y", "spatial_size", *DOM_FIELDS):
            if not np.isfinite(data["properties"][field][selected]).all():
                raise ValueError("V1_INPUT_NONFINITE_ACTIVE_LGN_FIELD:" + field)
        for field in OPTIONAL_FIELDS:
            values = data["properties"][field][selected]
            if population in DUAL_MODELS:
                if not np.isfinite(values).all():
                    raise ValueError("V1_INPUT_NONFINITE_DUAL_LGN_FIELD:" + field)
            else:
                if not np.isnan(values).all():
                    raise ValueError("V1_INPUT_MONO_FILTER_NAN_SEMANTICS_CHANGED:" + field)
                intentional_nan += len(values)
    return dict(model_counts=counts, intentional_nan_values=intentional_nan, active_fields_finite=True)


def validate_inputs(output: Path, population: str, models: dict, chunk_size: int) -> dict:
    fields = ("x", "y", "spatial_size", *DOM_FIELDS, *OPTIONAL_FIELDS) if population == "lgn" else ("x", "y")
    source = node_data(output / "biophysical", population, fields)
    metadata = validate_lgn_fields(source, models) if population == "lgn" else {}
    if population == "bkg" and (source["count"] != 1 or any(not np.isfinite(value).all() for value in source["properties"].values())):
        raise ValueError("V1_INPUT_SINGLE_FINITE_BACKGROUND_NODE_REQUIRED")
    copied = core.compare_h5(output / f"biophysical/{population}_nodes.h5", output / f"glif/{population}_nodes.h5", chunk_size)
    if not copied["equal"] or core.table_records(output / f"biophysical/{population}_node_types.csv", "node_type_id") != core.table_records(output / f"glif/{population}_node_types.csv", "node_type_id"):
        raise ValueError("V1_INPUT_SOURCE_NODE_COPY_CHANGED")
    targets = {name: node_data(output / name, "v1") for name in ("biophysical", "glif")}
    tables = {name: core.table_records(output / f"{name}/{population}_v1_edge_types.csv", "edge_type_id") for name in targets}
    frozen = core.table_records(output / f"glif_props/{population}_v1_edge_types.csv", "edge_type_id")
    for identity, row in tables["glif"].items():
        if identity not in frozen or any(row[key] != frozen[identity][key] for key in ("source_query", "target_query")):
            raise ValueError("V1_INPUT_FROZEN_GLIF_QUERY_CHANGED")
    table_arrays = {name: (np.asarray(sorted(table), dtype=np.uint64),
                           np.asarray([query_value(table[k]["target_query"], "node_type_id") for k in sorted(table)], dtype=np.uint64))
                    for name, table in tables.items()}
    synapses, seen = 0, set()
    with h5py.File(output / f"biophysical/{population}_v1_edges.h5", "r") as bio, h5py.File(output / f"glif/{population}_v1_edges.h5", "r") as glif:
        groups = {"biophysical": bio[f"edges/{population}_to_v1"], "glif": glif[f"edges/{population}_to_v1"]}
        size = len(groups["biophysical"]["source_node_id"])
        if size == 0:
            raise ValueError("V1_INPUT_EMPTY_SOURCE_CONNECTIONS")
        for group in groups.values():
            for field in ("source_node_id", "target_node_id", "edge_type_id", "edge_group_id", "edge_group_index"):
                if group[field].shape != (size,) or group[field].dtype.kind not in "ui":
                    raise ValueError("V1_INPUT_EDGE_SCHEMA_DIFFERS")
        for start in range(0, size, chunk_size):
            end = min(size, start + chunk_size)
            data = {name: {field: group[field][start:end] for field in ("source_node_id", "target_node_id", "edge_type_id", "edge_group_id", "edge_group_index")}
                    for name, group in groups.items()}
            for field in ("source_node_id", "target_node_id", "edge_group_id", "edge_group_index"):
                if data["biophysical"][field].dtype != data["glif"][field].dtype or not np.array_equal(data["biophysical"][field], data["glif"][field]):
                    raise ValueError("V1_INPUT_CONVERTED_EDGE_IDENTITY_CHANGED:" + field)
            for name, row in data.items():
                core.positions_for_ids(source["sorted_ids"], row["source_node_id"])
                target = core.positions_for_ids(targets[name]["sorted_ids"], row["target_node_id"])
                edge = core.positions_for_ids(table_arrays[name][0], row["edge_type_id"])
                if not np.array_equal(table_arrays[name][1][edge], targets[name]["sorted_types"][target]):
                    raise ValueError("V1_INPUT_TARGET_TYPE_PREDICATE_DIFFERS:" + name)
            values = [core.nsyns_for_rows(groups[name], data[name]["edge_group_id"], data[name]["edge_group_index"]) for name in ("biophysical", "glif")]
            if not np.array_equal(*values):
                raise ValueError("V1_INPUT_CONVERTED_NSYNS_CHANGED")
            synapses += int(values[0].sum(dtype=np.uint64)); seen.update(map(int, np.unique(data["glif"]["edge_type_id"])))
    if seen != set(tables["glif"]):
        raise ValueError("V1_INPUT_CONVERTED_EDGE_TYPE_INVENTORY_DIFFERS")
    return dict(source_units=source["count"], edge_rows=size, physical_synapses=synapses,
                all_edge_rows_verified=size, identities_nsyns_and_target_predicates_verified=True, **metadata)


def build_inputs(config_path: Path, recurrent: Path, output: Path) -> dict:
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_ORIGINAL_CIRCUIT_INPUTS", exit_code=1, scientific_qualification=False,
        production_episodes=0, native_nest_simulation=False, recurrence_generated=False, glif_node_assignment_regenerated=False,
        training_seeds_used=[], numpy_version=np.__version__, started_utc=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    try:
        config, snapshots = core.load_config(config_path); report.update(snapshots)
        primary, reference = bind_reference(recurrent, output); report.update(reference=reference, mode=primary["mode"])
        if any(field in os.environ for field in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID")):
            report["execution_context"] = cpu_execution_provenance()
        else:
            node = os.environ.get("SLURMD_NODENAME", socket.gethostname().split(".")[0])
            allowed_nodes = ("cpu03", "cpu05", "cpu06") if primary["mode"] == "full" else ("cpu03", "cpu04", "cpu05", "cpu06")
            if node not in allowed_nodes:
                raise ValueError("V1_INPUT_COMPUTE_PLACEMENT_INVALID:" + node)
        report["wheel"] = core.verify_wheel(config)
        if report["wheel"]["sha256"] != primary["wheel"]["sha256"]:
            raise ValueError("V1_INPUT_RECURRENT_RUNTIME_WHEEL_DIFFERS")
        bodies, sources = core.stage_sources(Path(config["source_root"]), output)
        if any(sources[key]["original_sha256"] != primary["sources"][key]["original_sha256"] for key in sources):
            raise ValueError("V1_INPUT_RECURRENT_SOURCE_BINDING_CHANGED")
        additional, models, fits = stage_input_sources(Path(config["source_root"]), output, bodies["original_node_functions"].decode())
        report.update(sources={**sources, **additional}, fitted_models=fits)
        with core.original_runtime(output, Path(config["bmtk_wheel"]["path"])):
            report["mpi"] = core.require_serial_runtime()
            builder = core.load_author_module("original_v1_input_builder", output / "build_network.py")
            converter = core.load_author_module("original_v1_input_converter", output / "convert_bio2glif.py")
            target = builder.NetworkBuilder("v1")
            target.import_nodes(str(output / "biophysical/v1_nodes.h5"), str(output / "biophysical/v1_node_types.csv"))
            report["imported_v1"] = verify_imported_target(target, output / "biophysical")
            if report["imported_v1"]["nodes"] != primary["validation"]["nodes"]:
                raise ValueError("V1_INPUT_BOUND_CORTICAL_COUNT_DIFFERS")
            restore_numpy_state(primary["rng"]["builder_after_build"])
            random.seed(config["builder_seed"])
            report["rng"] = dict(numpy_start=core.rng_state(), lgn_selection_python_seed=config["builder_seed"],
                python_start=random.getstate(), historical_python_state_recovered=False,
                numpy_checkpoint_precedes_recurrent_save=True, historical_master_rng_equivalence_verified=False,
                initialization="NumPy restored from recurrent builder checkpoint; Python random independently initialized for this development construction.", saves={})
            for population in ("lgn", "bkg"):
                network = builder.add_nodes_lgn() if population == "lgn" else builder.add_nodes_bkg()
                network = builder.add_lgn_v1_edges(target, network) if population == "lgn" else builder.add_bkg_v1_edges(target, network)
                network.build()
                before_np, before_python = core.rng_state(), random.getstate()
                network.save("biophysical")
                report["rng"]["saves"][population] = dict(numpy_before=before_np, numpy_after=core.rng_state(),
                    numpy_state_unchanged=before_np == core.rng_state(), python_before=before_python,
                    python_after=random.getstate(), python_state_unchanged=before_python == random.getstate())
                del network; gc.collect()
            report["rng"]["numpy_after_construction"] = core.rng_state()
            report["rng"]["python_after_construction"] = random.getstate()
            for population in ("lgn", "bkg"):
                converter.copy_nodes(population, bio_network_dir="biophysical", glif_network_dir="glif")
                converter.convert_edges(population, bio_network_dir="biophysical", glif_network_dir="glif",
                                        block_size=config["edge_chunk_size"], join_on_src=False)
            report["rng"]["numpy_after_conversion"] = core.rng_state()
            report["rng"]["python_after_conversion"] = random.getstate()
        for relative in REFERENCE_FILES:
            if sha256_file(output / relative) != reference["files"][relative]["sha256"]:
                raise ValueError("V1_INPUT_BOUND_V1_FILE_CHANGED:" + relative)
        report["validation"] = {population: validate_inputs(output, population, models, config["edge_chunk_size"]) for population in ("lgn", "bkg")}
        report.update(status="ORIGINAL_CIRCUIT_INPUTS_COMPLETE", exit_code=0, stored_v1_files_preserved=True,
            native_files=[dict(path=name, bytes=(output / name).stat().st_size) for name in NATIVE_FILES])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime"] = runtime_manifest({}, list(sys.argv), report["exit_code"])
        freeze_json(output / "report.json", report)
    return report


def compare_inputs(left: Path, right: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_ORIGINAL_INPUT_COMPARISON", exit_code=1, scientific_qualification=False, production_episodes=0, files=[])
    try:
        a, b = [json.loads((path / "report.json").read_text()) for path in (left, right)]
        if any(row.get("status") != "ORIGINAL_CIRCUIT_INPUTS_COMPLETE" or row.get("exit_code") != 0 for row in (a, b)):
            raise ValueError("V1_INPUT_TWO_COMPLETE_BUILDS_REQUIRED")
        differences = [key for key in ("mode", "rng") if a[key] != b[key]]
        if (a["reference"]["files"] != b["reference"]["files"] or a["reference"]["delivery_sha256"] != b["reference"]["delivery_sha256"]
                or a["wheel"]["sha256"] != b["wheel"]["sha256"] or a["fitted_models"]["files"] != b["fitted_models"]["files"]
                or {key: row["original_sha256"] for key, row in a["sources"].items()} != {key: row["original_sha256"] for key, row in b["sources"].items()}):
            differences.append("provenance")
        report["binding_differences"] = differences
        for relative in (*NATIVE_FILES, *REFERENCE_FILES):
            if relative.endswith(".h5"):
                result = core.compare_h5(left / relative, right / relative, 1000000)
            else:
                identity = "node_type_id" if relative.endswith("node_types.csv") else "edge_type_id"
                result = dict(equal=core.table_records(left / relative, identity) == core.table_records(right / relative, identity))
            report["files"].append(dict(path=relative, **result))
        equal = not differences and all(row["equal"] for row in report["files"])
        report.update(status="ORIGINAL_CIRCUIT_INPUT_REPLAY_EQUAL" if equal else "ORIGINAL_CIRCUIT_INPUT_REPLAY_DIFFERS",
                      exit_code=0 if equal else 1, all_native_files_equal=all(row["equal"] for row in report["files"]), bound_reference_and_rng_states_equal=not differences)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build")
    for name in ("config", "recurrent", "output"): command.add_argument("--" + name, type=Path, required=True)
    command = sub.add_parser("compare")
    for name in ("left", "right", "output"): command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_inputs(args.config, args.recurrent, args.output) if args.command == "build" else compare_inputs(args.left, args.right, args.output)
    print(json.dumps({key: report[key] for key in ("status", "exit_code")})); return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

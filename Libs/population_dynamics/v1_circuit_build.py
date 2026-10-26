"""Execute the preserved recurrent builder and GLIF conversion with explicit evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_circuit_source import read_source, declared_models, check_mapping, source_directory
from .v1_source_conversion import query_value


WHEEL_SHA256 = "d46d7407e4830fe492b2274fd466d5bad1b12d8a547f71cd8ea8c753ae7fdaa2"
WHEEL_BYTES = 985750
FULL_NODE_COUNT = 230924
MINIMUM_FULL_SCC = 100000
SOURCE_FILES = {
    "original_build_network": "build_network.py",
    "original_node_functions": "build_files/node_funcs.py",
    "original_edge_functions": "build_files/edge_funcs.py",
    "original_node_parameters": "build_files/biophys_props/v1_node_models.json",
    "original_edge_parameters": "build_files/biophys_props/v1_v1_edge_models.csv",
    "original_connection_parameters": "build_files/biophys_props/v1_conn_props.json",
    "original_lgn_connection_parameters": "build_files/biophys_props/lgn_conn_props.json",
    "weight_semantics/legacy_glif_converter": "convert_bio2glif.py",
    "original_glif_node_parameters": "glif_props/v1_node_models.json",
    "original_bio_to_glif_map": "glif_props/bio2glif_type_map.json",
    "weight_semantics/legacy_recurrent_table": "glif_props/v1_v1_edge_types.csv",
}
NATIVE_FILES = tuple(f"{network}/{name}" for network in ("biophysical", "glif")
                     for name in ("v1_nodes.h5", "v1_node_types.csv", "v1_v1_edges.h5", "v1_v1_edge_types.csv"))


def load_config(path: Path) -> tuple[dict, dict]:
    original = yaml.safe_load(path.read_text())
    if not isinstance(original, dict):
        raise ValueError("CIRCUIT_BUILD_CONFIG_MAPPING_REQUIRED")
    config = dict(original)
    # Seeds vary across the approved independent circuit realizations. Keep
    # the source scale and construction fractions fixed for every realization.
    required = {"expected_full_nodes": FULL_NODE_COUNT, "canary_fraction": 0.01, "full_fraction": 1.0}
    for key, expected in required.items():
        if key not in config or isinstance(config[key], bool) or config[key] != expected:
            raise ValueError("CIRCUIT_BUILD_FROZEN_CONFIG_DIFFERS:" + key)
    if any(type(config.get(key)) is not int for key in ("builder_seed", "glif_assignment_seed", "expected_full_nodes")):
        raise ValueError("CIRCUIT_BUILD_INTEGER_CONFIG_REQUIRED")
    for key in ("builder_seed", "glif_assignment_seed"):
        if not 0 <= config[key] <= np.iinfo(np.uint32).max:
            raise ValueError("CIRCUIT_BUILD_UINT32_SEED_REQUIRED:" + key)
    if config["builder_seed"] == config["glif_assignment_seed"]:
        raise ValueError("CIRCUIT_BUILD_SEPARATE_RANDOM_STREAM_SEEDS_REQUIRED")
    chunk = config.get("edge_chunk_size", 1000000)
    if type(chunk) is not int or chunk < 1:
        raise ValueError("CIRCUIT_BUILD_POSITIVE_CHUNK_REQUIRED")
    config["edge_chunk_size"] = chunk
    config["source_root"] = str(Path(config["source_root"]).resolve())
    config["bmtk_wheel"] = dict(config["bmtk_wheel"])
    config["bmtk_wheel"]["path"] = str(Path(config["bmtk_wheel"]["path"]).resolve())
    if config["bmtk_wheel"].get("sha256") != WHEEL_SHA256:
        raise ValueError("CIRCUIT_BUILD_PINNED_WHEEL_REQUIRED")
    snapshots = dict(input_config=original, resolved_config=config,
                     original_config=original.get("original_config", original),
                     original_config_sha256=original.get("original_config_sha256"),
                     input_config_path=str(path.resolve()), input_config_sha256=sha256_file(path))
    return config, snapshots


def stage_sources(root: Path, work: Path) -> tuple[dict, dict]:
    bodies, records = {}, {}
    for role, relative in SOURCE_FILES.items():
        raw, receipt = read_source(root / role)
        bodies[role] = raw
        text = raw.decode()
        changes = []
        if role == "original_edge_functions":
            if text.count("np.NaN") != 2:
                raise ValueError("CIRCUIT_BUILD_EXPECTED_TWO_NAN_ALIASES")
            changes = [dict(line=i, before="np.NaN", after="np.nan")
                       for i, line in enumerate(text.splitlines(), 1) if "np.NaN" in line]
            text = text.replace("np.NaN", "np.nan")
        elif role == "weight_semantics/legacy_glif_converter":
            if len(re.findall(r"\bnp\.int\b", text)) != 1:
                raise ValueError("CIRCUIT_BUILD_EXPECTED_ONE_INT_ALIAS")
            changes = [dict(line=i, before="np.int", after="int")
                       for i, line in enumerate(text.splitlines(), 1) if re.search(r"\bnp\.int\b", line)]
            text = re.sub(r"\bnp\.int\b", "int", text)
            # The source's one-to-one branch assigns a length-one list to
            # multiple rows. Pandas 2 requires the equivalent scalar value.
            assignment = "nodes_df.loc[subgrp.index.values, 'glif_node_type_id'] = glif_ntids"
            if text.count(assignment) != 1:
                raise ValueError("CIRCUIT_BUILD_EXPECTED_SINGLE_GLIF_BROADCAST")
            line = next(i for i, value in enumerate(text.splitlines(), 1) if assignment in value)
            changes.append(dict(line=line, before=assignment, after=assignment + "[0]"))
            text = text.replace(assignment, assignment + "[0]")
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        records[role] = dict(original_sha256=receipt["sha256"], original_bytes=len(raw),
            receipt_sha256=sha256_file(root / role / "source.json"),
            published_url=receipt.get("published_url"), staged_path=relative,
            staged_sha256=sha256_file(target), compatibility=changes)
    (work / "build_files/__init__.py").write_text("")
    return bodies, records


def verify_wheel(config: dict) -> dict:
    path = Path(config["bmtk_wheel"]["path"])
    if path.stat().st_size != WHEEL_BYTES or sha256_file(path) != WHEEL_SHA256:
        raise ValueError("CIRCUIT_BUILD_WHEEL_CONTENT_DIFFERS")
    return dict(path=str(path), bytes=WHEEL_BYTES, sha256=WHEEL_SHA256, version="1.2.0")


@contextmanager
def original_runtime(work: Path, wheel: Path | None = None):
    """Scope source imports and the verified wheel to this child process call."""
    old_path = list(sys.path)
    old_modules = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="v1-build-runtime-") as temporary:
        runtime = Path(temporary)
        if wheel is not None:
            if any(name == "bmtk" or name.startswith("bmtk.") for name in sys.modules):
                raise ValueError("CIRCUIT_BUILD_REQUIRES_FRESH_BMTK_IMPORT")
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(runtime)
            sys.path.insert(0, str(runtime))
        sys.path.insert(0, str(work))
        try:
            with source_directory(work):
                if wheel is not None:
                    bmtk = importlib.import_module("bmtk")
                    if bmtk.__version__ != "1.2.0" or not Path(bmtk.__file__).resolve().is_relative_to(runtime):
                        raise ValueError("CIRCUIT_BUILD_RUNTIME_IMPORT_DIFFERS")
                yield
        finally:
            sys.path[:] = old_path
            for name in set(sys.modules) - old_modules:
                if name == "bmtk" or name.startswith("bmtk.") or name == "build_files" or name.startswith("build_files."):
                    del sys.modules[name]


def load_author_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def require_serial_runtime() -> dict:
    # Both modules are actual runtime dependencies of the original builder.
    from mpi4py import MPI
    from bmtk.builder import builder_utils
    comm = MPI.COMM_WORLD
    record = dict(mpi_size=comm.Get_size(), mpi_rank=comm.Get_rank(),
                  builder_mpi_size=builder_utils.mpi_size, builder_mpi_rank=builder_utils.mpi_rank)
    if record != dict(mpi_size=1, mpi_rank=0, builder_mpi_size=1, builder_mpi_rank=0):
        raise ValueError("CIRCUIT_BUILD_SINGLE_ACTUAL_MPI_PROCESS_REQUIRED:" + json.dumps(record))
    return record


def population_counts(document: dict, fraction: float) -> tuple[dict, dict, int]:
    targets, sources = {}, Counter()
    for groups in document["locations"].values():
        for population, group in groups.items():
            for model in group["models"]:
                identity = model["node_type_id"]
                if type(identity) is not int or identity in targets or type(model["N"]) is not int or model["N"] < 1:
                    raise ValueError("CIRCUIT_BUILD_INVALID_SOURCE_MODEL_COUNT")
                count = int(np.max((model["N"] * fraction, 1)))
                label = ("LIF" if model["model_type"] != "biophysical" else "") + population
                targets[identity] = count
                sources[label] += count
    return targets, dict(sources), sum(targets.values())


def require_node_count(mode: str, actual: int, expected_canary: int) -> None:
    expected = FULL_NODE_COUNT if mode == "full" else expected_canary
    if actual != expected:
        raise ValueError(f"CIRCUIT_BUILD_NODE_COUNT_DIFFERS:{actual}!={expected}")


def require_memory_lower_bound(mode: str, dense_bytes: int, environment=None) -> dict:
    environment = os.environ if environment is None else environment
    if mode != "full":
        return dict(applied=False, dense_nsyns_bytes=dense_bytes, peak_memory_qualified=False)
    value = environment.get("SLURM_MEM_PER_NODE")
    if value is None or not str(value).isdigit() or int(value) <= 0:
        raise ValueError("CIRCUIT_BUILD_FULL_SLURM_MEMORY_ALLOCATION_REQUIRED")
    allocated = int(value) * 1024 * 1024
    if allocated < dense_bytes:
        raise ValueError(f"CIRCUIT_BUILD_MEMORY_BELOW_DENSE_LOWER_BOUND:{allocated}<{dense_bytes}")
    return dict(applied=True, allocated_memory_bytes=allocated, dense_nsyns_bytes=dense_bytes,
                lower_bound_satisfied=True, peak_memory_qualified=False,
                interpretation="Passing the dense-array lower bound does not admit Cartesian/collation/sorting peak memory.")


def estimate_dense(bodies: dict, work: Path, fraction: float, pair_parameters) -> dict:
    document = json.loads(bodies["original_node_parameters"])
    targets, sources, count = population_counts(document, fraction)
    _, _, full_count = population_counts(document, 1.0)
    if full_count != FULL_NODE_COUNT:
        raise ValueError("CIRCUIT_BUILD_ORIGINAL_FULL_COUNT_DIFFERS")
    frame = pd.read_csv(work / SOURCE_FILES["original_edge_parameters"], sep=" ")
    maps, skipped = [], []
    for position, (_, row) in enumerate(frame.iterrows()):
        source, target = row["source_label"], int(row["target_model_id"])
        parameters = pair_parameters(source, row["target_label"])
        if not np.isfinite(parameters["A_new"]):
            raise ValueError("CIRCUIT_BUILD_NONFINITE_SOURCE_CONNECTION_PROBABILITY")
        if parameters["A_new"] <= 0:
            skipped.append(position)
            continue
        if source not in sources or target not in targets:
            raise ValueError("CIRCUIT_BUILD_ACTIVE_SOURCE_OR_TARGET_MISSING")
        pairs = sources[source] * targets[target]
        maps.append(dict(csv_row=position, csv_edge_type_id=int(row["edge_type_id"]),
            source_population=source, target_node_type_id=target,
            source_nodes=sources[source], target_nodes=targets[target], candidate_pairs=pairs,
            dense_nsyns_bytes=4 * pairs, A_new=float(parameters["A_new"]),
            csv_edge_type_id_explicitly_used=bool(row["target_label"].startswith("LIF"))))
    pairs = sum(row["candidate_pairs"] for row in maps)
    return dict(fraction=fraction, nodes=count, full_source_nodes=full_count,
                active_connection_maps=len(maps), skipped_csv_rows=skipped,
                candidate_pairs=pairs, dense_nsyns_bytes=4 * pairs,
                largest_map=max(maps, key=lambda row: row["candidate_pairs"]) if maps else None,
                maps=maps, source_population_counts=sources, target_type_counts=targets,
                interpretation="Dense uint32 candidate-count allocations only; not peak RSS. Cartesian, collation, sorting, indexes and Python objects require additional memory.")


def rng_state() -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(algorithm=algorithm, keys=keys.tolist(), position=int(position),
                has_gauss=int(has_gauss), cached_gaussian=float(cached_gaussian))


def seed_and_record(seed: int) -> dict:
    np.random.seed(seed)
    return rng_state()


def table_records(path: Path, identity: str) -> dict:
    frame = pd.read_csv(path, sep=" ")
    if identity not in frame or frame[identity].isna().any() or frame[identity].duplicated().any():
        raise ValueError("CIRCUIT_BUILD_UNIQUE_TABLE_IDENTITIES_REQUIRED:" + str(path))
    result = {}
    for row in frame.to_dict(orient="records"):
        key = row[identity]
        if isinstance(key, bool) or int(key) != key:
            raise ValueError("CIRCUIT_BUILD_INTEGER_TABLE_IDENTITY_REQUIRED")
        result[int(key)] = {name: None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
                            for name, value in row.items()}
    return result


def integer_vector(dataset, name: str) -> np.ndarray:
    if dataset.ndim != 1 or dataset.dtype.kind not in "ui":
        raise ValueError("CIRCUIT_BUILD_INTEGER_VECTOR_REQUIRED:" + name)
    values = dataset[...]
    if values.dtype.kind == "i" and np.any(values < 0):
        raise ValueError("CIRCUIT_BUILD_NONNEGATIVE_IDS_REQUIRED:" + name)
    return values


def node_metadata(directory: Path) -> dict:
    with h5py.File(directory / "v1_nodes.h5", "r") as handle:
        group = handle["nodes/v1"]
        values = {name: integer_vector(group[name], name) for name in
                  ("node_id", "node_type_id", "node_group_id", "node_group_index")}
    count = len(values["node_id"])
    if any(len(array) != count for array in values.values()) or len(np.unique(values["node_id"])) != count:
        raise ValueError("CIRCUIT_BUILD_NODE_IDENTITIES_INVALID")
    types = table_records(directory / "v1_node_types.csv", "node_type_id")
    if set(map(int, np.unique(values["node_type_id"]))) != set(types):
        raise ValueError("CIRCUIT_BUILD_NODE_TYPES_DO_NOT_MATCH_DECLARATIONS")
    order = np.argsort(values["node_id"])
    return dict(values=values, types=types, count=count, order=order,
                sorted_ids=values["node_id"][order], sorted_types=values["node_type_id"][order])


def positions_for_ids(sorted_ids: np.ndarray, query: np.ndarray) -> np.ndarray:
    if query.ndim != 1 or query.dtype.kind not in "ui" or (query.dtype.kind == "i" and np.any(query < 0)):
        raise ValueError("CIRCUIT_BUILD_INVALID_EDGE_NODE_IDS")
    positions = np.searchsorted(sorted_ids, query)
    if len(sorted_ids) == 0 or np.any(positions >= len(sorted_ids)) or not np.array_equal(sorted_ids[positions], query):
        raise ValueError("CIRCUIT_BUILD_EDGE_REFERENCES_UNKNOWN_NODE")
    return positions


def edge_type_arrays(records: dict, populations: dict, require_unique_pairs: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(records), dtype=np.uint64)
    targets, sources, pairs = [], [], set()
    for identity in ids:
        row = records[int(identity)]
        target = query_value(row["target_query"], "node_type_id")
        source = query_value(row["source_query"], "pop_name")
        if source not in populations or (require_unique_pairs and (target, source) in pairs):
            raise ValueError("CIRCUIT_BUILD_EDGE_TABLE_PAIR_INVALID")
        pairs.add((target, source)); targets.append(target); sources.append(populations[source])
    return ids, np.asarray(targets, dtype=np.uint64), np.asarray(sources, dtype=np.int64)


def values_at(dataset, indices: np.ndarray) -> np.ndarray:
    if indices.dtype.kind not in "ui" or np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise ValueError("CIRCUIT_BUILD_EDGE_GROUP_INDEX_OUT_OF_RANGE")
    if len(indices) == 0:
        return dataset[0:0]
    if len(indices) == 1 or np.all(np.diff(indices.astype(np.int64)) == 1):
        return dataset[int(indices[0]):int(indices[-1]) + 1]
    unique, inverse = np.unique(indices, return_inverse=True)
    if len(unique) == 1 or np.all(np.diff(unique.astype(np.int64)) == 1):
        values = dataset[int(unique[0]):int(unique[-1]) + 1]
    else:
        values = dataset[unique]
    return values[inverse]


def nsyns_for_rows(group, group_ids: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.uint64)
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        dataset = group[str(int(group_id))]["nsyns"]
        if dataset.ndim != 1 or dataset.dtype.kind not in "ui":
            raise ValueError("CIRCUIT_BUILD_NSYNS_INTEGER_VECTOR_REQUIRED")
        values = values_at(dataset, indices[mask])
        if np.any(values < 1):
            raise ValueError("CIRCUIT_BUILD_POSITIVE_NSYNS_REQUIRED")
        result[mask] = values
    return result


def connectivity_components(source: np.ndarray, target: np.ndarray, count: int) -> dict:
    adjacency = coo_matrix((np.ones(len(source), dtype=np.bool_), (source, target)), shape=(count, count)).tocsr()
    components, labels = connected_components(adjacency, directed=True, connection="strong", return_labels=True)
    sizes = np.bincount(labels, minlength=components)
    return dict(strong_components=int(components), largest_strong_component=int(sizes.max()) if len(sizes) else 0,
                nodes=count, source_edge_rows=len(source), adjacency_pairs=int(adjacency.nnz),
                adjacency_representation="Boolean adjacency for connectivity only; original source edge arrays are retained unchanged.")


def require_coupled_component(mode: str, components: dict) -> None:
    if mode == "full" and components["largest_strong_component"] < MINIMUM_FULL_SCC:
        raise ValueError("CIRCUIT_BUILD_FULL_COUPLED_COMPONENT_BELOW_100000:" + json.dumps(components))


def validate_native(work: Path, mapping: dict, mode: str, expected_canary: int, chunk_size: int,
                    expected_synapses: int | None = None) -> dict:
    bio = node_metadata(work / "biophysical")
    glif = node_metadata(work / "glif")
    require_node_count(mode, bio["count"], expected_canary)
    require_node_count(mode, glif["count"], expected_canary)
    for name in ("node_id", "node_group_id", "node_group_index"):
        if not np.array_equal(bio["values"][name], glif["values"][name]):
            raise ValueError("CIRCUIT_BUILD_CONVERTED_NODE_IDENTITY_CHANGED:" + name)
    for kind in np.unique(bio["values"]["node_type_id"]):
        definition = mapping[str(int(kind))]
        row = bio["types"][int(kind)]
        expected_population = ("LIF" if row["model_type"] != "biophysical" else "") + definition["pop_name"]
        if row["pop_name"] != expected_population:
            raise ValueError("CIRCUIT_BUILD_BIOLOGICAL_POPULATION_DIFFERS")
        selected = glif["values"]["node_type_id"][bio["values"]["node_type_id"] == kind]
        if not np.isin(selected, definition["node_type_id"]).all() or any(
                glif["types"][int(value)]["pop_name"] != definition["pop_name"] for value in np.unique(selected)):
            raise ValueError("CIRCUIT_BUILD_GLIF_ASSIGNMENT_OUTSIDE_SOURCE_MAPPING")
    labels = sorted({row["pop_name"] for data in (bio, glif) for row in data["types"].values()})
    population_codes = {label: i for i, label in enumerate(labels)}
    checks = {}
    for name, metadata in (("biophysical", bio), ("glif", glif)):
        records = table_records(work / name / "v1_v1_edge_types.csv", "edge_type_id")
        ids, target_types, source_populations = edge_type_arrays(records, population_codes, require_unique_pairs=name == "glif")
        checks[name] = dict(ids=ids, target_types=target_types, source_populations=source_populations,
            node_populations=np.asarray([population_codes[metadata["types"][int(kind)]["pop_name"]]
                                         for kind in metadata["sorted_types"]]), metadata=metadata, seen=set())
    # Only the declared target/source query is used; no weights are fitted or inferred.
    frozen = table_records(work / "glif_props/v1_v1_edge_types.csv", "edge_type_id")
    written = table_records(work / "glif/v1_v1_edge_types.csv", "edge_type_id")
    for identity, row in written.items():
        if identity not in frozen or any(row[key] != frozen[identity][key] for key in ("source_query", "target_query")):
            raise ValueError("CIRCUIT_BUILD_GLIF_EDGE_QUERY_CHANGED")
    total_synapses = 0
    with h5py.File(work / "biophysical/v1_v1_edges.h5", "r") as bio_file, h5py.File(work / "glif/v1_v1_edges.h5", "r") as glif_file:
        groups = {"biophysical": bio_file["edges/v1_to_v1"], "glif": glif_file["edges/v1_to_v1"]}
        count = len(groups["biophysical"]["source_node_id"])
        if count < 1:
            raise ValueError("CIRCUIT_BUILD_EMPTY_RECURRENT_GRAPH")
        for group in groups.values():
            for name in ("source_node_id", "target_node_id", "edge_type_id", "edge_group_id", "edge_group_index"):
                if group[name].shape != (count,) or group[name].dtype.kind not in "ui":
                    raise ValueError("CIRCUIT_BUILD_EDGE_SCHEMA_DIFFERS:" + name)
        index_dtype = np.int32 if glif["count"] <= np.iinfo(np.int32).max else np.int64
        source_indices, target_indices = np.empty(count, dtype=index_dtype), np.empty(count, dtype=index_dtype)
        for start in range(0, count, chunk_size):
            end = min(count, start + chunk_size)
            row_data = {key: {name: group[name][start:end] for name in
                        ("source_node_id", "target_node_id", "edge_type_id", "edge_group_id", "edge_group_index")}
                        for key, group in groups.items()}
            for name in ("source_node_id", "target_node_id", "edge_group_id", "edge_group_index"):
                if row_data["biophysical"][name].dtype != row_data["glif"][name].dtype or not np.array_equal(
                        row_data["biophysical"][name], row_data["glif"][name]):
                    raise ValueError("CIRCUIT_BUILD_CONVERTED_EDGE_IDENTITY_CHANGED:" + name)
            for key, data in row_data.items():
                check = checks[key]
                src = positions_for_ids(check["metadata"]["sorted_ids"], data["source_node_id"])
                trg = positions_for_ids(check["metadata"]["sorted_ids"], data["target_node_id"])
                et = positions_for_ids(check["ids"], data["edge_type_id"])
                if not np.array_equal(check["target_types"][et], check["metadata"]["sorted_types"][trg]) or not np.array_equal(
                        check["source_populations"][et], check["node_populations"][src]):
                    raise ValueError("CIRCUIT_BUILD_EDGE_TYPE_PREDICATE_DIFFERS:" + key)
                check["seen"].update(map(int, np.unique(data["edge_type_id"])))
                if key == "glif":
                    source_indices[start:end], target_indices[start:end] = src, trg
            nsyns = [nsyns_for_rows(groups[key], row_data[key]["edge_group_id"], row_data[key]["edge_group_index"])
                     for key in ("biophysical", "glif")]
            if not np.array_equal(*nsyns):
                raise ValueError("CIRCUIT_BUILD_CONVERTED_NSYNS_CHANGED")
            total_synapses += int(nsyns[0].sum(dtype=np.uint64))
    if checks["glif"]["seen"] != set(map(int, checks["glif"]["ids"])):
        raise ValueError("CIRCUIT_BUILD_GLIF_EDGE_TYPE_INVENTORY_DIFFERS")
    if expected_synapses is not None and total_synapses != expected_synapses:
        raise ValueError("CIRCUIT_BUILD_PHYSICAL_SYNAPSE_COUNT_DIFFERS")
    components = connectivity_components(source_indices, target_indices, glif["count"])
    require_coupled_component(mode, components)
    return dict(nodes=glif["count"], edge_rows=count, physical_synapses=total_synapses,
        all_edge_rows_verified=count, endpoints_and_nsyns_preserved=True,
        node_assignment_membership_verified=True, source_and_target_predicates_verified=True,
        glif_type_counts={str(key): value for key, value in sorted(Counter(map(int, glif["values"]["node_type_id"])).items())},
        connectivity=components, full_scale_gate_applied=mode == "full",
        minimum_full_strong_component=MINIMUM_FULL_SCC)


def execute(config_path: Path, mode: str, output: Path, estimate_only: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="FAILED_ORIGINAL_RECURRENT_BUILD", exit_code=1, mode=mode,
        scientific_qualification=False, production_episodes=0, training_seeds_used=[],
        native_nest_simulation=False, lgn_or_background_generated=False,
        representation="Source SONATA recurrent graph only", full_recurrent_realization=False,
        started_utc=datetime.now(timezone.utc).isoformat())
    try:
        config, snapshots = load_config(config_path)
        report.update(snapshots)
        report["wheel"] = verify_wheel(config)
        bodies, records = stage_sources(Path(config["source_root"]), output)
        report["sources"] = records
        bio, _ = declared_models(json.loads(bodies["original_node_parameters"]))
        glif, _ = declared_models(json.loads(bodies["original_glif_node_parameters"]))
        mapping = json.loads(bodies["original_bio_to_glif_map"])
        report["mapping"] = check_mapping(bio, glif, mapping)
        fraction = config["full_fraction" if mode == "full" else "canary_fraction"]
        with original_runtime(output):
            edge_functions = load_author_module("original_v1_edge_functions", output / "build_files/edge_funcs.py")
            report["dense_estimate"] = estimate_dense(bodies, output, fraction, edge_functions.compute_pair_type_parameters)
        if estimate_only:
            report.update(status="ORIGINAL_RECURRENT_DENSE_ESTIMATE_COMPLETE", exit_code=0)
        else:
            report["memory_lower_bound"] = require_memory_lower_bound(mode, report["dense_estimate"]["dense_nsyns_bytes"])
            with original_runtime(output, Path(config["bmtk_wheel"]["path"])):
                report["mpi"] = require_serial_runtime()
                builder = load_author_module("original_v1_builder", output / "build_network.py")
                converter = load_author_module("original_v1_converter", output / "convert_bio2glif.py")
                report["rng"] = dict(builder_seed=config["builder_seed"], glif_assignment_seed=config["glif_assignment_seed"],
                                     builder_start=seed_and_record(config["builder_seed"]))
                phase_started = time.monotonic()
                network = builder.add_nodes_v1(fraction=fraction)
                network = builder.add_v1_v1_edges(network)
                network.build()
                require_node_count(mode, int(network.nnodes), report["dense_estimate"]["nodes"])
                builder_synapses = int(network.nedges)
                report["builder_physical_synapses"] = builder_synapses
                report["rng"]["builder_after_build"] = rng_state()
                report["build_seconds"] = time.monotonic() - phase_started
                phase_started = time.monotonic()
                network.save("biophysical")
                report["save_seconds"] = time.monotonic() - phase_started
                del network
                gc.collect()
                report["builder_released_before_conversion"] = True
                report["rng"]["glif_start"] = seed_and_record(config["glif_assignment_seed"])
                phase_started = time.monotonic()
                converter.convert_v1_nodes(bio_network_dir="biophysical", glif_network_dir="glif")
                converter.convert_edges("v1", bio_network_dir="biophysical", glif_network_dir="glif", block_size=config["edge_chunk_size"])
                report["conversion_seconds"] = time.monotonic() - phase_started
                report["rng"]["glif_after_conversion"] = rng_state()
            report["validation"] = validate_native(output, mapping, mode, report["dense_estimate"]["nodes"],
                                                    config["edge_chunk_size"], builder_synapses)
            report["native_files"] = [dict(path=name, bytes=(output / name).stat().st_size) for name in NATIVE_FILES]
            report.update(status="ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE" if mode == "full" else "ORIGINAL_RECURRENT_CANARY_COMPLETE",
                          exit_code=0, full_recurrent_realization=mode == "full")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["runtime"] = runtime_manifest({}, list(sys.argv), report["exit_code"])
        report["runtime"]["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
        freeze_json(output / "report.json", report)
    return report


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" and right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def h5_inventory(handle) -> tuple[dict, dict]:
    datasets, attributes = {}, {"/": dict(handle.attrs)}
    def collect(name, item):
        if isinstance(item, h5py.Dataset):
            datasets[name] = item
        attributes[name] = dict(item.attrs)
    handle.visititems(collect)
    return datasets, attributes


def compare_h5(left: Path, right: Path, chunk_size: int) -> dict:
    differences, compared = [], 0
    with h5py.File(left, "r") as lhs, h5py.File(right, "r") as rhs:
        left_sets, left_attrs = h5_inventory(lhs)
        right_sets, right_attrs = h5_inventory(rhs)
        if set(left_sets) != set(right_sets):
            differences.append(dict(reason="dataset_inventory", left_only=sorted(set(left_sets)-set(right_sets)),
                                    right_only=sorted(set(right_sets)-set(left_sets))))
        for name in sorted(set(left_sets) & set(right_sets)):
            a, b = left_sets[name], right_sets[name]
            if a.shape != b.shape or a.dtype != b.dtype or a.dtype.metadata != b.dtype.metadata:
                differences.append(dict(dataset=name, reason="dtype_or_shape", left_shape=list(a.shape),
                    right_shape=list(b.shape), left_dtype=str(a.dtype), right_dtype=str(b.dtype)))
                continue
            compared += 1
            intervals = [(None, None)] if a.shape == () else [(start, min(start+chunk_size, a.shape[0])) for start in range(0, a.shape[0], chunk_size)]
            for start, end in intervals:
                av, bv = (np.asarray(a[()]), np.asarray(b[()])) if start is None else (a[start:end], b[start:end])
                if not arrays_equal(av, bv):
                    differences.append(dict(dataset=name, reason="values", first_axis_interval=[start, end]))
                    break
        for name in sorted(set(left_attrs) | set(right_attrs)):
            aa, bb = left_attrs.get(name, {}), right_attrs.get(name, {})
            if set(aa) != set(bb) or any(not arrays_equal(np.asarray(aa[key]), np.asarray(bb[key])) for key in set(aa) & set(bb)):
                differences.append(dict(object=name, reason="user_attributes"))
    return dict(equal=not differences, datasets_compared=compared, differences=differences,
                policy="Every dataset dtype, shape and decoded value; NaNs compare equal. User attributes included; HDF5 binary creation metadata is not compared.")


def comparison_specification(report: dict) -> dict:
    config = report["resolved_config"]
    return dict(mode=report["mode"], wheel=report["wheel"]["sha256"],
                source_hashes={key: value["original_sha256"] for key, value in report["sources"].items()},
                config={key: config[key] for key in ("builder_seed", "glif_assignment_seed", "expected_full_nodes",
                                                     "canary_fraction", "full_fraction", "edge_chunk_size")})


def compare_builds(left: Path, right: Path, output: Path, chunk_size: int = 1000000) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_ORIGINAL_RECURRENT_COMPARISON", exit_code=1, scientific_qualification=False,
                  production_episodes=0, left=str(left.resolve()), right=str(right.resolve()), files=[])
    try:
        reports = [json.loads((path / "report.json").read_text()) for path in (left, right)]
        valid = {"ORIGINAL_RECURRENT_CANARY_COMPLETE", "ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE"}
        if any(item.get("status") not in valid or item.get("exit_code") != 0 for item in reports):
            raise ValueError("CIRCUIT_BUILD_TWO_COMPLETED_NATIVE_BUILDS_REQUIRED")
        if comparison_specification(reports[0]) != comparison_specification(reports[1]):
            raise ValueError("CIRCUIT_BUILD_REPLAY_SPECIFICATION_DIFFERS")
        for name in NATIVE_FILES:
            if name.endswith(".h5"):
                result = compare_h5(left / name, right / name, chunk_size)
            else:
                key = "node_type_id" if name.endswith("node_types.csv") else "edge_type_id"
                a, b = table_records(left / name, key), table_records(right / name, key)
                unequal = [identity for identity in sorted(set(a) | set(b)) if a.get(identity) != b.get(identity)]
                result = dict(equal=not unequal, rows_left=len(a), rows_right=len(b), different_type_ids=unequal)
            report["files"].append(dict(path=name, **result))
        equal = all(item["equal"] for item in report["files"])
        report.update(status="ORIGINAL_RECURRENT_REPLAY_EQUAL" if equal else "ORIGINAL_RECURRENT_REPLAY_DIFFERS",
                      exit_code=0 if equal else 1, mode=reports[0]["mode"], all_native_datasets_and_tables_equal=equal,
                      input_report_sha256=[sha256_file(path / "report.json") for path in (left, right)])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("build", "estimate"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--mode", choices=("canary", "full"), required=True)
        subparser.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--left", type=Path, required=True)
    comparison.add_argument("--right", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--chunk-size", type=int, default=1000000)
    args = parser.parse_args(argv)
    if args.action == "compare":
        if args.chunk_size < 1:
            parser.error("chunk-size must be positive")
        report = compare_builds(args.left, args.right, args.output, args.chunk_size)
    else:
        report = execute(args.config, args.mode, args.output, estimate_only=args.action == "estimate")
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

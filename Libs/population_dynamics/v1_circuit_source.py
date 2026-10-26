"""Inspect the original circuit recipe and execute its recurrent sampler on source nodes.

This is a development probe, not a circuit realization or source-fidelity gate.
The preserved author files are verified and only copied into local scratch.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_graph_identity import _h5_metadata, _types

SOURCE_DIRECTORIES = (
    "original_build_network", "original_node_functions", "original_edge_functions",
    "original_connection_parameters", "original_lgn_connection_parameters",
    "original_node_parameters", "original_edge_parameters",
    "original_bio_to_glif_map", "original_glif_node_parameters",
    "original_glif_recurrent_parameters",
)
PROBE_SOURCES_PER_POPULATION = 8
CONSTRUCTION_SEEDS = (100, 101)


def read_source(directory):
    receipt = json.loads((directory / "source.json").read_text())
    path = directory / "source_asset.txt"
    raw = path.read_bytes()
    if (receipt["artifact"] != path.name or len(raw) != receipt["bytes"]
            or hashlib.sha256(raw).hexdigest() != receipt["sha256"]):
        raise ValueError("V1_CIRCUIT_SOURCE_RECEIPT_MISMATCH:" + directory.name)
    return raw, receipt


def declared_models(document):
    models = {}
    populations = {}
    for location, groups in document["locations"].items():
        for population, group in groups.items():
            if population in populations:
                raise ValueError("V1_CIRCUIT_DUPLICATE_POPULATION")
            populations[population] = group
            for model in group["models"]:
                identity = model["node_type_id"]
                if type(identity) is not int or identity in models:
                    raise ValueError("V1_CIRCUIT_DUPLICATE_OR_INVALID_MODEL")
                if type(model["N"]) is not int or model["N"] < 1:
                    raise ValueError("V1_CIRCUIT_INVALID_DECLARED_COUNT")
                models[identity] = dict(model, pop_name=population, location=location)
    return models, populations


def check_mapping(bio, glif, mapping):
    missing = sorted(set(map(str, bio)) - set(mapping))
    if missing:
        raise ValueError("V1_CIRCUIT_MAPPING_MISSING_SOURCE_TYPES:" + str(missing))
    covered = set()
    stochastic = []
    for identity, model in bio.items():
        row = mapping[str(identity)]
        targets = row["node_type_id"]
        if not targets or len(targets) != len(set(targets)) or any(i not in glif for i in targets):
            raise ValueError("V1_CIRCUIT_MAPPING_TARGET_INVALID")
        if row["pop_name"] != model["pop_name"] or any(glif[i]["pop_name"] != model["pop_name"] for i in targets):
            raise ValueError("V1_CIRCUIT_MAPPING_POPULATION_DIFFERS")
        if len(targets) > 1:
            weights = np.asarray(row["proportions"], dtype=float)
            if weights.shape != (len(targets),) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("V1_CIRCUIT_MAPPING_PROPORTIONS_INVALID")
            # NumPy's own categorical sampler validates normalization; no fitted tolerance.
            np.random.RandomState(100).choice(len(targets), size=1, p=weights)
            stochastic.append({"source_type": identity, "target_types": targets,
                               "declared_probability_sum": float(weights.sum())})
        covered.update(targets)
    if covered != set(glif):
        raise ValueError("V1_CIRCUIT_MAPPING_INCOMPLETE_GLIF_COVERAGE")
    return {"source_types": len(bio), "covered_glif_types": len(covered),
            "stochastic_mappings": stochastic, "extra_mapping_keys": sorted(set(mapping) - set(map(str, bio)))}


def edge_pairs(text, models, populations):
    pairs = set()
    for row in csv.DictReader(text.splitlines(), delimiter=" "):
        target = re.fullmatch(r"node_type_id=='([0-9]+)'", row["target_query"])
        source = re.fullmatch(r"pop_name=='([A-Za-z0-9]+)'", row["source_query"])
        if target is None or source is None:
            raise ValueError("V1_CIRCUIT_UNSUPPORTED_SOURCE_QUERY")
        pair = (int(target[1]), source[1])
        if pair[0] not in models or pair[1] not in populations or pair in pairs:
            raise ValueError("V1_CIRCUIT_EDGE_TYPE_BINDING_INVALID")
        if not np.isfinite(float(row["syn_weight"])) or not np.isfinite(float(row["delay"])) or float(row["delay"]) < 0:
            raise ValueError("V1_CIRCUIT_EDGE_PARAMETER_INVALID")
        pairs.add(pair)
    return pairs


class SourceNode(dict):
    def __init__(self, node_id, **values):
        super().__init__(values)
        self.node_id = node_id


@contextmanager
def source_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_module(path):
    spec = importlib.util.spec_from_file_location("original_v1_edge_functions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_counts(module, nodes, types, pairs, seed):
    labels = sorted({row["pop_name"] for row in types.values()})
    node_labels = np.asarray([types[int(i)]["pop_name"] for i in nodes["node_type_id"]])
    selected = {label: np.flatnonzero(node_labels == label)[:PROBE_SOURCES_PER_POPULATION] for label in labels}
    source_ids = np.concatenate(list(selected.values()))
    target_ids = np.asarray([np.flatnonzero(nodes["node_type_id"] == identity)[0] for identity in sorted(types)])
    def node(index):
        return SourceNode(int(nodes["node_id"][index]), **{k: nodes[k][index] for k in ("x", "y", "z", "tuning_angle")})
    output = np.zeros((len(target_ids), len(source_ids)), dtype=np.int64)
    np.random.seed(seed)
    for row_index, target_index in enumerate(target_ids):
        target = node(target_index)
        target_type = int(nodes["node_type_id"][target_index])
        offset = 0
        for label, indices in selected.items():
            if (target_type, label) in pairs:
                parameters = module.compute_pair_type_parameters(label, types[target_type]["pop_name"])
                if parameters["A_new"] > 0:
                    counts = module.connect_cells([node(i) for i in indices], target, parameters)
                    output[row_index, offset:offset + len(indices)] = [0 if n is None else n for n in counts]
            offset += len(indices)
    return source_ids, target_ids, output


def inspect(source_root, nodes_path, types_path, node_report, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scientific_qualification": False, "production_episodes": 0,
              "new_full_circuit_realizations": 0, "training_seeds_used": [],
              "construction_probe_seeds": list(CONSTRUCTION_SEEDS), "source_sampling_law_changed": False}
    try:
        sources = {name: read_source(source_root / name) for name in SOURCE_DIRECTORIES}
        report["sources"] = {name: {"sha256": receipt["sha256"], "published_url": receipt["published_url"]}
                             for name, (_, receipt) in sources.items()}
        original_nodes = json.loads(node_report.read_text())
        if sha256_file(nodes_path) != original_nodes["nodes_sha256"] or sha256_file(types_path) != original_nodes["node_types_sha256"]:
            raise ValueError("V1_CIRCUIT_NATIVE_NODE_BINDING_CHANGED")
        nodes, types = _h5_metadata(nodes_path), _types(types_path)
        bio, _ = declared_models(json.loads(sources["original_node_parameters"][0]))
        glif, populations = declared_models(json.loads(sources["original_glif_node_parameters"][0]))
        report["mapping"] = check_mapping(bio, glif, json.loads(sources["original_bio_to_glif_map"][0]))
        if set(glif) != set(types) or any(glif[i]["pop_name"] != types[i]["pop_name"] for i in types):
            raise ValueError("V1_CIRCUIT_NATIVE_GLIF_IDENTITY_DIFFERS")
        observed = Counter(map(int, nodes["node_type_id"]))
        report["model_counts"] = [{"node_type_id": identity, "declared_N": glif[identity]["N"], "observed_N": observed[identity]}
                                  for identity in sorted(glif)]
        report["count_interpretation"] = "Declared GLIF counts and one realized graph; differences are not silently remapped or treated as scientific rejection. Mapping law/consumer still needs qualification."
        pairs = edge_pairs(sources["original_glif_recurrent_parameters"][0].decode(), glif, populations)
        report.update(cortical_units=len(nodes["node_id"]), glif_model_types=len(types), source_populations=len(populations), declared_recurrent_type_pairs=len(pairs))
        with tempfile.TemporaryDirectory(prefix="v1-circuit-kernel-") as name:
            work = Path(name)
            props = work / "build_files/biophys_props"; props.mkdir(parents=True)
            (props / "v1_conn_props.json").write_bytes(sources["original_connection_parameters"][0])
            (props / "lgn_conn_props.json").write_bytes(sources["original_lgn_connection_parameters"][0])
            code = sources["original_edge_functions"][0].decode()
            module_path = work / "edge_funcs.py"; module_path.write_text(code)
            with source_directory(work):
                module = load_module(module_path)
                try:
                    first = sample_counts(module, nodes, types, pairs, CONSTRUCTION_SEEDS[0])
                    report["original_runtime_error"] = None
                except AttributeError as error:
                    if 'NaN' not in str(error) or code.count('np.NaN') != 2:
                        raise
                    report["original_runtime_error"] = {"type": type(error).__name__, "message": str(error)}
                    # NumPy renamed this constant alias; preserve the original file and law.
                    code = code.replace('np.NaN', 'np.nan')
                    module_path.write_text(code)
                    module = load_module(module_path)
                    first = sample_counts(module, nodes, types, pairs, CONSTRUCTION_SEEDS[0])
                    report["runtime_compatibility"] = {"replacement": "np.NaN -> np.nan", "occurrences": 2,
                                                        "source_original_preserved": True}
                repeated = sample_counts(module, nodes, types, pairs, CONSTRUCTION_SEEDS[0])
                second = sample_counts(module, nodes, types, pairs, CONSTRUCTION_SEEDS[1])
                if any(not np.array_equal(a, b) for a, b in zip(first, repeated)):
                    raise ValueError("V1_CIRCUIT_KERNEL_REPLAY_DIFFERS")
                np.savez_compressed(output / "kernel_probe.npz", source_ids=first[0], target_ids=first[1],
                                    first_counts=first[2], repeated_counts=repeated[2], second_seed_counts=second[2])
                report["kernel_probe"] = {"source_units": len(first[0]), "target_units": len(first[1]),
                    "same_seed_exact": True, "different_seed_count_entries": int(np.count_nonzero(first[2] != second[2])),
                    "selection": "First eight source IDs per population, first target ID per node type; all available units when fewer than eight.",
                    "artifact": "kernel_probe.npz", "numpy_version": np.__version__,
                    "scope": "Original recurrent sampler only; no LGN generation, weight application, full graph, or population-law validation."}
        report["status"] = "ORIGINAL_CIRCUIT_SOURCE_KERNEL_DEVELOPMENT_COMPLETE"
    except Exception as error:
        report.update(status="FAILED_ORIGINAL_CIRCUIT_SOURCE_PROBE", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "nodes", "types", "node-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.source_root, args.nodes, args.types, args.node_report, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()

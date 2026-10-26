"""Build the source recurrent graph without storing rejected candidate pairs.

The source sampler still evaluates every source probability for each target in
the original order. This adapter caches immutable source attributes, retains
only positive synapse counts, and restores source-major order before BMTK saves
the existing SONATA format. It does not change sampling or connection rules.
"""
from __future__ import annotations

import ast
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np


def array_sampler(source_sampler):
    """Compile the inspected source body with cached attributes and array output."""
    module = ast.parse(inspect.getsource(source_sampler))
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "connect_cells":
        raise ValueError("V1_RECURRENT_SOURCE_CONNECT_CELLS_REQUIRED")
    expected = {
        "sources_x": "np.array([s['x'] for s in sources])",
        "sources_z": "np.array([s['z'] for s in sources])",
        "sources_tuning_angle": "[s['tuning_angle'] for s in sources]",
    }
    found = set()
    for statement in function.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        name = ast.unparse(statement.targets[0])
        if name in expected:
            wanted = ast.parse(expected[name], mode="eval").body
            if name in found or ast.dump(statement.value) != ast.dump(wanted):
                raise ValueError("V1_RECURRENT_SOURCE_ATTRIBUTE_EXTRACTION_DIFFERS:" + name)
            found.add(name)
            statement.value = ast.Subscript(value=ast.Name(id="source_attributes", ctx=ast.Load()),
                                           slice=ast.Constant(name), ctx=ast.Load())
    tail = ast.parse("nsyns_ret = [Nsyn if Nsyn != 0 else None for Nsyn in p_connected]\nreturn nsyns_ret").body
    if found != set(expected) or [ast.dump(node) for node in function.body[-2:]] != [ast.dump(node) for node in tail]:
        raise ValueError("V1_RECURRENT_SOURCE_COUNT_RETURN_DIFFERS")
    function.body[-2:] = [ast.Return(value=ast.Name(id="p_connected", ctx=ast.Load()))]
    function.args.kwonlyargs.append(ast.arg(arg="source_attributes"))
    function.args.kw_defaults.append(None)
    ast.fix_missing_locations(module)
    namespace = dict(source_sampler.__globals__)
    exec(compile(module, inspect.getsourcefile(source_sampler), "exec"), namespace)
    return namespace["connect_cells"]


class RecurrentEdgeTable:
    """The BMTK saving interface for source-major positive recurrent counts."""

    def __init__(self, connection_map, pairs: np.ndarray, counts: np.ndarray):
        self._connection_map = connection_map
        self.source_network = connection_map.source_network_name
        self.target_network = connection_map.target_network_name
        self.edge_type_id = connection_map.edge_type_id
        self.edge_group_id = -1
        self._pairs = pairs
        self._counts = counts
        self.n_edges = len(counts)
        self.n_syns = int(counts.sum(dtype=np.uint64))

    @property
    def edge_type_properties(self):
        return self._connection_map.edge_type_properties

    @property
    def edge_type_node_ids(self):
        return self._pairs

    @property
    def hash_key(self):
        # Match BMTK's existing model-group key and uint32 count metadata.
        return hashlib.md5(b"nsyns(uint32)").hexdigest()[:9]

    def get_property_metatadata(self):
        return [dict(name="nsyns", dtype=np.dtype("uint32"))]

    def get_property_value(self, prop_name):
        if prop_name != "nsyns":
            raise ValueError("V1_RECURRENT_ONLY_SOURCE_SYNAPSE_COUNTS_SUPPORTED")
        return self._counts

    def free_data(self):
        self._counts = None
        self._pairs = None


def build_edge_table(connection_map, sampler, source_sampler) -> RecurrentEdgeTable:
    """Consume source draws in target order and sort only the accepted pairs."""
    if (connection_map.iterator != "all_to_one" or connection_map.connector is not source_sampler
            or connection_map.params or set(connection_map.connector_params) != {"params"}):
        raise ValueError("V1_RECURRENT_SOURCE_ALL_TO_ONE_WITHOUT_EDGE_PROPERTIES_REQUIRED")
    sources = list(connection_map.source_nodes)
    targets = list(connection_map.target_nodes)
    attributes = dict(sources_x=np.array([node["x"] for node in sources]),
                      sources_z=np.array([node["z"] for node in sources]),
                      sources_tuning_angle=[node["tuning_angle"] for node in sources])
    source_ids = np.asarray([node.node_id for node in sources], dtype=np.uint64)
    target_ids = np.asarray([node.node_id for node in targets], dtype=np.uint64)
    source_indices, target_indices, counts = [], [], []
    for target_index, target in enumerate(targets):
        values = sampler(sources, target, **connection_map.connector_params, source_attributes=attributes)
        if (values.shape != (len(sources),) or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max)):
            raise ValueError("V1_RECURRENT_SOURCE_SYNAPSE_COUNTS_INVALID")
        accepted = np.flatnonzero(values)
        if len(accepted):
            source_indices.append(accepted.astype(np.uint32))
            target_indices.append(np.full(len(accepted), target_index, dtype=np.uint32))
            counts.append(values[accepted].astype(np.uint32))
    if not counts:
        return RecurrentEdgeTable(connection_map, np.empty((0, 2), dtype=np.uint64), np.empty(0, dtype=np.uint32))
    rows = np.concatenate(source_indices)
    columns = np.concatenate(target_indices)
    values = np.concatenate(counts)
    # Legacy flattening enumerates sources first, then targets within sources.
    order = np.lexsort((columns, rows))
    pairs = np.column_stack((source_ids[rows[order]], target_ids[columns[order]]))
    return RecurrentEdgeTable(connection_map, pairs, values[order])


def install(network, source_sampler) -> dict:
    """Install on this recurrent network instance; leave other BMTK builds alone."""
    network = network.adaptor
    if network.edges_table() or network.edges_built:
        raise ValueError("V1_RECURRENT_UNBUILT_NETWORK_REQUIRED")
    sampler = array_sampler(source_sampler)
    report = dict(implementation="positive_source_counts_with_cached_attributes",
                  candidate_pairs=0, accepted_pairs=0, edge_types=0,
                  dense_candidate_tables=False, source_sampling_arithmetic_unchanged=True,
                  source_and_target_order_unchanged=True, source_major_output_order=True)

    def add_edges(self, connection_map, index):
        table = build_edge_table(connection_map, sampler, source_sampler)
        self._target_networks[connection_map.target_network_name] = connection_map.target_nodes.network
        self._nedges += table.n_syns
        self.edges_table().append(table)
        report["candidate_pairs"] += connection_map.max_connections()
        report["accepted_pairs"] += table.n_edges
        report["edge_types"] += 1

    network._add_edges = MethodType(add_edges, network)
    return report


@contextmanager
def recurrent_execution(core, output: Path):
    """Use existing construction, conversion and validation with this storage path."""
    original_loader = core.load_author_module
    original_memory_check = core.require_memory_lower_bound
    original_freeze = core.freeze_json
    execution = {}

    def load_author_module(name, path):
        nonlocal execution
        module = original_loader(name, path)
        if name == "original_v1_builder":
            add_edges = module.add_v1_v1_edges

            def add_recurrent_edges(network):
                network = add_edges(network)
                nonlocal execution
                execution = install(network, module.connect_cells)
                return network

            module.add_v1_v1_edges = add_recurrent_edges
        return module

    def memory_check(mode, dense_bytes, environment=None):
        if mode != "full":
            return original_memory_check(mode, dense_bytes, environment)
        environment = os.environ if environment is None else environment
        allocated = environment.get("SLURM_MEM_PER_NODE")
        if allocated is None or not str(allocated).isdigit() or int(allocated) <= 0:
            raise ValueError("V1_RECURRENT_FULL_SLURM_MEMORY_REQUIRED")
        return dict(applied=False, allocated_memory_bytes=int(allocated) * 1024**2,
                    legacy_dense_nsyns_bytes=dense_bytes, dense_candidate_tables=False,
                    peak_memory_qualified=False,
                    interpretation="Compact accepted-edge storage replaces dense candidate tables; peak RSS must be measured.")

    def freeze(path, report):
        if Path(path).resolve() == output.resolve() / "report.json":
            report["recurrent_execution"] = execution
            report["recurrent_execution_implementation_sha256"] = core.sha256_file(Path(__file__))
        return original_freeze(path, report)

    core.load_author_module = load_author_module
    core.require_memory_lower_bound = memory_check
    core.freeze_json = freeze
    try:
        yield
    finally:
        core.load_author_module = original_loader
        core.require_memory_lower_bound = original_memory_check
        core.freeze_json = original_freeze


def main():
    from . import v1_circuit_build as core

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with recurrent_execution(core, args.output):
        report = core.execute(args.config, args.mode, args.output)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "build_seconds", "recurrent_execution")}))


if __name__ == "__main__":
    main()

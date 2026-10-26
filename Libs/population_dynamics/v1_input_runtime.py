"""Build source LGN inputs with cached grouping and positive-pair storage.

The source selection arithmetic and random calls remain unchanged. Source
identities and model grouping are immutable during input construction. The
adapter evaluates targets in source order and preserves BMTK source-major output.
Background construction, source conversion and input validation are unchanged.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from types import MethodType

import numpy as np

from .v1_recurrent_runtime import RecurrentEdgeTable


def cached_sampler(source_sampler):
    """Replace only repeated source identity and model-group construction."""
    module = ast.parse(inspect.getsource(source_sampler))
    function = module.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != "select_lgn_sources":
        raise ValueError("V1_INPUT_SOURCE_SELECT_LGN_REQUIRED")
    expected = ast.parse(
        "source_ids = [s.node_id for s in sources]\n"
        "cell_type_dict = {}\n"
        "for lgn_model in lgn_models:\n"
        "    cell_type_dict[lgn_model] = [(src_id, src_dict) for src_id, src_dict in "
        "zip(source_ids, sources) if src_dict['pop_name'] == lgn_model]"
    ).body
    replacements = (
        ast.parse('source_ids = source_cache["source_ids"]').body,
        ast.parse('cell_type_dict = source_cache["cell_type_dict"]').body,
        [],
    )
    found = set()
    body = []
    for statement in function.body:
        match = next((i for i, node in enumerate(expected)
                      if ast.dump(statement) == ast.dump(node)), None)
        if match is None:
            body.append(statement)
        else:
            if match in found:
                raise ValueError("V1_INPUT_SOURCE_GROUP_CONSTRUCTION_DUPLICATED")
            found.add(match)
            body.extend(replacements[match])
    if found != {0, 1, 2}:
        raise ValueError("V1_INPUT_SOURCE_GROUP_CONSTRUCTION_DIFFERS")
    function.body = body
    function.args.kwonlyargs.append(ast.arg(arg="source_cache"))
    function.args.kw_defaults.append(None)
    ast.fix_missing_locations(module)
    namespace = dict(source_sampler.__globals__)
    exec(compile(module, inspect.getsourcefile(source_sampler), "exec"), namespace)
    return namespace["select_lgn_sources"]


def source_cache(sources, models):
    source_ids = [source.node_id for source in sources]
    return dict(source_ids=source_ids, cell_type_dict={
        model: [(identity, source) for identity, source in zip(source_ids, sources)
                if source["pop_name"] == model] for model in models})


def build_edge_table(connection_map, sampler, source_sampler):
    if (connection_map.iterator != "all_to_one" or connection_map.connector is not source_sampler
            or connection_map.params or set(connection_map.connector_params) != {
                "lgn_mean", "lgn_models", "nsyns", "edge_type_id"}):
        raise ValueError("V1_INPUT_SOURCE_LGN_MAP_WITHOUT_EDGE_PROPERTIES_REQUIRED")
    sources = list(connection_map.source_nodes)
    targets = list(connection_map.target_nodes)
    cache = source_cache(sources, connection_map.connector_params["lgn_models"])
    source_ids = np.asarray(cache["source_ids"], dtype=np.uint64)
    target_ids = np.asarray([target.node_id for target in targets], dtype=np.uint64)
    source_indices, target_indices, counts = [], [], []
    for target_index, target in enumerate(targets):
        values = sampler(sources, target, **connection_map.connector_params, source_cache=cache)
        # The source returns None for all pairs when target selection is rejected.
        values = np.asarray([0 if value is None else value for value in values])
        if (values.shape != (len(sources),) or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max)):
            raise ValueError("V1_INPUT_SOURCE_SYNAPSE_COUNTS_INVALID")
        accepted = np.flatnonzero(values)
        if len(accepted):
            source_indices.append(accepted.astype(np.uint32))
            target_indices.append(np.full(len(accepted), target_index, dtype=np.uint32))
            counts.append(values[accepted].astype(np.uint32))
    if not counts:
        return RecurrentEdgeTable(connection_map, np.empty((0, 2), dtype=np.uint64),
                                  np.empty(0, dtype=np.uint32))
    rows = np.concatenate(source_indices)
    columns = np.concatenate(target_indices)
    values = np.concatenate(counts)
    order = np.lexsort((columns, rows))
    pairs = np.column_stack((source_ids[rows[order]], target_ids[columns[order]]))
    # Reuse the existing count-table interface; it has no recurrent-rule arithmetic.
    return RecurrentEdgeTable(connection_map, pairs, values[order])


def install(network, source_sampler):
    adaptor = network.adaptor
    if adaptor.edges_table() or adaptor.edges_built:
        raise ValueError("V1_INPUT_UNBUILT_LGN_NETWORK_REQUIRED")
    sampler = cached_sampler(source_sampler)
    report = dict(implementation="cached_source_groups_positive_lgn_counts",
                  candidate_pairs=0, accepted_pairs=0, edge_types=0,
                  source_selection_arithmetic_unchanged=True, source_major_output_order=True,
                  dense_candidate_tables=False)

    def add_edges(self, connection_map, index):
        table = build_edge_table(connection_map, sampler, source_sampler)
        self._target_networks[connection_map.target_network_name] = connection_map.target_nodes.network
        self._nedges += table.n_syns
        self.edges_table().append(table)
        report["candidate_pairs"] += connection_map.max_connections()
        report["accepted_pairs"] += table.n_edges
        report["edge_types"] += 1

    adaptor._add_edges = MethodType(add_edges, adaptor)
    return report


@contextmanager
def input_execution(core, inputs, output):
    original_loader = core.load_author_module
    original_freeze = inputs.freeze_json
    execution = {}

    def load_author_module(name, path):
        module = original_loader(name, path)
        if name == "original_v1_input_builder":
            add_edges = module.add_lgn_v1_edges

            def add_lgn_edges(*args, **kwargs):
                nonlocal execution
                network = add_edges(*args, **kwargs)
                execution = install(network, module.select_lgn_sources)
                return network

            module.add_lgn_v1_edges = add_lgn_edges
        return module

    def freeze(path, report):
        if Path(path).resolve() == output.resolve() / "report.json":
            report["lgn_execution"] = execution
            report["lgn_execution_implementation_sha256"] = core.sha256_file(Path(__file__))
        return original_freeze(path, report)

    core.load_author_module = load_author_module
    inputs.freeze_json = freeze
    try:
        yield
    finally:
        core.load_author_module = original_loader
        inputs.freeze_json = original_freeze


def main():
    from . import v1_circuit_build as core
    from . import v1_circuit_inputs as inputs

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "recurrent", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with input_execution(core, inputs, args.output):
        report = inputs.build_inputs(args.config, args.recurrent, args.output)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "elapsed_seconds", "lgn_execution")}))


if __name__ == "__main__":
    main()

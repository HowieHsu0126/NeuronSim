"""Compare LGN caching and compact storage with the pinned source and BMTK."""
from pathlib import Path
import importlib.util
import os
import random

import numpy as np
import pytest

from . import v1_input_runtime as runtime

pytestmark = pytest.mark.skipif(not os.environ.get("NRNSIM_V1_STAGED_EDGE_FUNCTIONS"),
                                reason="Pinned V1 source and BMTK require compute-node staging")


class Node(dict):
    def __init__(self, identity, **properties):
        super().__init__(properties)
        self.node_id = identity


@pytest.fixture
def source_sampler():
    path = Path(os.environ["NRNSIM_V1_STAGED_EDGE_FUNCTIONS"])
    spec = importlib.util.spec_from_file_location("source_lgn_edge_functions", path)
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(path.parents[1])
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module.select_lgn_sources


def assert_rng_equal(expected):
    numpy_state, python_state = expected
    actual = np.random.get_state()
    assert actual[0] == numpy_state[0] and actual[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert random.getstate() == python_state


def test_cached_groups_preserve_selection_and_both_rng_states(source_sampler):
    models = {name: {} for name in ("sON_TF1", "sOFF_TF2", "tOFF_TF4", "sONsOFF_001", "sONtOFF_001")}
    sources = [Node(100 - i, pop_name=name, x=110.0 + i % 12, y=52.0 + i % 13,
                    tuning_angle=float((i * 15) % 360))
               for i, name in enumerate(list(models) * 30)]
    targets = [Node(i, pop_name=population, x=0.0, z=0.0, tuning_angle=tuning)
               for i, (population, tuning) in enumerate((
                   ("LIFe4Scnn1a", 0.0), ("LIFe4Scnn1a", 90.0),
                   ("LIFi4Pvalb", np.nan), ("LIFe6Ntsr1", 270.0)))]
    cached = runtime.cached_sampler(source_sampler)
    cache = runtime.source_cache(sources, models)
    # Multiple seeds exercise rejected targets and oriented/unoriented selection.
    for seed in (41, 42, 43):
        np.random.seed(seed); random.seed(seed + 100)
        expected = [source_sampler(sources, target, (120.0, 60.0), models, 15, 6580)
                    for target in targets]
        state = np.random.get_state(), random.getstate()
        np.random.seed(seed); random.seed(seed + 100)
        actual = [cached(sources, target, (120.0, 60.0), models, 15, 6580, source_cache=cache)
                  for target in targets]
        assert expected == actual
        assert_rng_equal(state)


def test_compact_input_table_matches_bmtk_ids_counts_and_metadata(source_sampler):
    from bmtk.builder import NetworkBuilder
    from bmtk.builder.network_adaptors.edge_props_table import EdgeTypesTableMemory

    sources = NetworkBuilder("lgn")
    sources.add_nodes(N=20, node_id=np.arange(20)[::-1] * 3, pop_name="sON_TF1",
                      x=np.linspace(112.0, 124.0, 20), y=np.linspace(54.0, 66.0, 20), tuning_angle=np.zeros(20))
    targets = NetworkBuilder("v1")
    targets.add_nodes(N=5, node_id=[20, 15, 10, 5, 0], pop_name="LIFe4Scnn1a",
                      x=np.zeros(5), z=np.zeros(5), tuning_angle=[0.0, 90.0, 180.0, 270.0, np.nan])
    connection_map = sources.add_edges(target=targets.nodes(), connection_rule=source_sampler,
        iterator="all_to_one", connection_params=dict(lgn_mean=(120.0, 60.0), lgn_models={"sON_TF1": {}},
                                                       nsyns=15, edge_type_id=6580))
    for network in (sources, targets):
        if not network.adaptor.nodes_built:
            network.adaptor._build_nodes()
    np.random.seed(991); random.seed(992)
    original = EdgeTypesTableMemory(connection_map, "lgn")
    for source, target, count in connection_map.connection_itr():
        if count:
            original.set_nsyns(source, target, count)
    state = np.random.get_state(), random.getstate()
    np.random.seed(991); random.seed(992)
    compact = runtime.build_edge_table(connection_map, runtime.cached_sampler(source_sampler), source_sampler)
    np.testing.assert_array_equal(compact.edge_type_node_ids, original.edge_type_node_ids)
    np.testing.assert_array_equal(compact.get_property_value("nsyns"), original.get_property_value("nsyns"))
    assert compact.n_syns == original.n_syns and compact.n_edges == original.n_edges
    assert compact.hash_key == original.hash_key
    assert compact.get_property_metatadata() == original.get_property_metatadata()
    assert_rng_equal(state)


def test_rejects_other_input_connection_rules(source_sampler):
    from bmtk.builder import NetworkBuilder

    source = NetworkBuilder("lgn")
    source.add_nodes(N=1, x=120.0, y=60.0, pop_name="sON_TF1")
    connection_map = source.add_edges(connection_rule=lambda a, b: 1)
    with pytest.raises(ValueError, match="SOURCE_LGN_MAP"):
        runtime.build_edge_table(connection_map, runtime.cached_sampler(source_sampler), source_sampler)

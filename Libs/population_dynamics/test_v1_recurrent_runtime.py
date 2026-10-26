"""Compare compact recurrent storage to the actual source and BMTK interfaces."""
from pathlib import Path
import os
from types import SimpleNamespace

import numpy as np
import pytest

from . import v1_recurrent_runtime as runtime

pytestmark = pytest.mark.skipif(not os.environ.get("NRNSIM_V1_STAGED_EDGE_FUNCTIONS"),
                                reason="Pinned V1 source and BMTK wheel require compute-node staging")

def test_array_sampler_preserves_all_counts_and_rng(source_sampler):
    sources = [SimpleNode(i, x=float(i * 3), z=float(i % 3), tuning_angle=float(i * 20)) for i in range(20)]
    attributes = dict(sources_x=np.array([s["x"] for s in sources]),
                      sources_z=np.array([s["z"] for s in sources]),
                      sources_tuning_angle=[s["tuning_angle"] for s in sources])
    sampler = runtime.array_sampler(source_sampler)
    for gradient, intercept in ((np.nan, np.nan), (-0.005, 1.25)):
        params = dict(A_new=0.5, sigma=80, gradient=gradient, intercept=intercept, nsyn_range=[1, 8])
        targets = sources[:3] + [SimpleNode(99, x=14.0, z=8.0, tuning_angle=15.0)]
        np.random.seed(901)
        expected = [[0 if n is None else n for n in source_sampler(sources, target, params)] for target in targets]
        expected_state = np.random.get_state()
        for angles in (attributes['sources_tuning_angle'], np.asarray(attributes['sources_tuning_angle'], dtype=float)):
            np.random.seed(901)
            actual = [sampler(sources, target, params, source_attributes=dict(attributes, sources_tuning_angle=angles)) for target in targets]
            actual_state = np.random.get_state()
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual_state[1], expected_state[1])
            assert actual_state[0] == expected_state[0] and actual_state[2:] == expected_state[2:]


class SimpleNode(dict):
    def __init__(self, identity, **properties):
        super().__init__(properties)
        self.node_id = identity


@pytest.fixture
def source_sampler():
    import importlib.util
    path = Path(os.environ["NRNSIM_V1_STAGED_EDGE_FUNCTIONS"])
    spec = importlib.util.spec_from_file_location("source_edge_functions", path)
    module = importlib.util.module_from_spec(spec)
    previous = Path.cwd()
    try:
        os.chdir(path.parents[1])
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module.connect_cells


def test_edge_table_matches_bmtk_source_major_order_and_uint32_metadata(source_sampler):
    from bmtk.builder import NetworkBuilder
    from bmtk.builder.network_adaptors.edge_props_table import EdgeTypesTableMemory

    network = NetworkBuilder("v1")
    network.add_nodes(N=12, node_id=np.arange(12)[::-1] * 3,
        x=np.arange(12, dtype=float), z=np.zeros(12), tuning_angle=np.arange(12, dtype=float) * 20)
    connection_map = network.add_edges(iterator="all_to_one", connection_rule=source_sampler,
        connection_params={"params": dict(A_new=0.5, sigma=80, gradient=-0.005, intercept=1.25, nsyn_range=[1, 8])})
    network.adaptor._build_nodes()
    np.random.seed(902)
    original = EdgeTypesTableMemory(connection_map, "v1")
    for source, target, count in connection_map.connection_itr():
        if count:
            original.set_nsyns(source, target, count)
    expected_state = np.random.get_state()
    np.random.seed(902)
    compact = runtime.build_edge_table(connection_map, runtime.array_sampler(source_sampler), source_sampler)
    actual_state = np.random.get_state()
    np.testing.assert_array_equal(compact.edge_type_node_ids, original.edge_type_node_ids)
    np.testing.assert_array_equal(compact.get_property_value("nsyns"), original.get_property_value("nsyns"))
    assert compact.n_syns == original.n_syns and compact.n_edges == original.n_edges
    assert compact.hash_key == original.hash_key
    assert compact.get_property_metatadata() == original.get_property_metatadata()
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]


def test_rejects_other_rules_and_properties(source_sampler):
    from bmtk.builder import NetworkBuilder

    network = NetworkBuilder("v1")
    network.add_nodes(N=2, x=[0.0, 1.0], z=[0.0, 0.0], tuning_angle=[0.0, 180.0])
    connection_map = network.add_edges(connection_rule=source_sampler, iterator="one_to_one")
    with pytest.raises(ValueError, match="SOURCE_ALL_TO_ONE"):
        runtime.build_edge_table(connection_map, runtime.array_sampler(source_sampler), source_sampler)

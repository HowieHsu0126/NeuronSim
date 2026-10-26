import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Libs.population_dynamics.network_snapshot import (
    export_initial_network, load_initial_network, NEURON_PARAMETERS, _plain,
)
from Libs.population_dynamics.nest_reference import EDGE_DTYPE


class Nodes:
    def __init__(self, ids, values=None):
        self.ids = ids
        self.values = values or {}
        self.spike_times = [1., 2.]
    def __len__(self): return len(self.ids)
    def tolist(self): return self.ids
    def get(self, names):
        return {name: self.values.get(name, np.full(len(self), 1.)) for name in names}


def source_fixture():
    records = np.array([(10, 30, 1., .2), (11, 10, .5, .3),
                        (11, 10, .75, .4), (20, 31, 1., .1)], dtype=EDGE_DTYPE)
    records.sort(order=["target", "source", "weight", "delay"])
    class Nest:
        biological_time = 0.
        @staticmethod
        def NodeCollection(ids): return ids
        @staticmethod
        def GetConnections(target, synapse_model):
            chosen = records[np.isin(records["target"], target)]
            return SimpleNamespace(get=lambda names: {n: chosen[n] for n in names})
    params = {name: np.full(2, 2.) for name in NEURON_PARAMETERS}
    network = SimpleNamespace(
        pops=[Nodes([10, 11], params), Nodes([20])],
        poisson_bg_input=Nodes([30], {"rate": [100.], "origin": [0.],
                                    "start": [0.], "stop": [np.inf]}),
        spike_pulse_input_th=Nodes([31]),
    )
    net = dict(poisson_input=True, thalamic_input=True, thalamic_input_type="pulses",
               dc_input=False, extent=1., neuron_model="iaf_psc_exp")
    fingerprint = dict(connection_count=4, target_batch_size=2,
                       sha256=hashlib.sha256(records.tobytes()).hexdigest())
    return network, Nest(), net, fingerprint, records


def write_snapshot(path):
    network, nest, net, fingerprint, records = source_fixture()
    paths = export_initial_network(network, nest, net,
        {"sim_resolution": .1, "t_presim": 0., "t_sim": 3.}, {},
        {"revision": "f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5"},
        np.array([10, 11, 20]), np.array(["L23E", "L23E", "TC"]),
        np.array([[0., 0.], [.1, .1], [-.1, .1]]),
        fingerprint, None, path)
    return paths, records


def test_roundtrip_preserves_native_ids_geometry_and_multiedges(tmp_path):
    path = tmp_path/"initial_network"
    paths, original = write_snapshot(path)
    metadata, arrays = load_initial_network(path)
    np.testing.assert_array_equal(arrays["edges"], original)
    np.testing.assert_array_equal(arrays["unit_ids"], [10, 11, 20])
    assert arrays["edges"].dtype == EDGE_DTYPE
    assert metadata["devices"][0]["parameters"]["stop"] == "positive_infinity"
    assert metadata["devices"][1]["parameters"]["effective_spike_times_ms"] == [1., 2.]
    assert len(paths) == 7


def test_corrupt_snapshot_is_rejected(tmp_path):
    path = tmp_path/"initial_network"
    write_snapshot(path)
    np.save(path/"unit_ids.npy", np.array([10, 12, 20]), allow_pickle=False)
    with pytest.raises(ValueError, match="snapshot changed"):
        load_initial_network(path)


def test_only_initial_time_can_be_exported(tmp_path):
    network, nest, net, fingerprint, _ = source_fixture()
    nest.biological_time = .1
    with pytest.raises(ValueError, match="before time advances"):
        export_initial_network(network, nest, net, {}, {}, {}, [], [], [],
                               fingerprint, None, tmp_path/"late")
    assert not (tmp_path/"late").exists()


def test_numpy_scalars_and_nonfinite_semantics_are_explicit():
    assert _plain(np.array(2.)) == 2.
    assert _plain(np.array([np.inf, -np.inf])) == ["positive_infinity", "negative_infinity"]
    with pytest.raises(ValueError, match="NaN"):
        _plain(float("nan"))

"""Identity and coupled-scale gates on explicit synthetic graph fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from unittest.mock import patch
import zipfile

import h5py
import numpy as np
import pytest

from . import v1_graph_identity
from .v1_graph_identity import ARCHIVE_MEMBERS, compare_type_metadata, inspect_archive, main, validate_raw_graph


def node_group(ids):
    return {"ids": np.asarray(ids, dtype=np.int64), "params": {
        "V_th": -50.0, "g": 10.0, "E_L": -65.0, "k": [0.1, 0.2],
        "C_m": 200.0, "V_reset": -65.0, "tau_syn": [1.0] * 4,
        "t_ref": 2.0, "asc_amps": [0.0, 0.0],
    }}


def edge(source, target, weight=None, delay=2.0, receptor=1):
    return {"source": np.asarray(source, dtype=np.int64), "target": np.asarray(target, dtype=np.int64),
            "params": {"weight": np.ones(len(source)) if weight is None else np.asarray(weight),
                       "delay": delay, "receptor_type": receptor}}


def graph(n=3):
    return {"nodes": [node_group(range(n))],
            "edges": [edge(np.arange(n), np.roll(np.arange(n), -1))]}


def test_identity_joins_by_id_and_rejects_missing_duplicate_or_mixed_groups():
    raw = graph()
    raw["nodes"] = [node_group([2]), node_group([1, 0])]
    report = validate_raw_graph(raw, np.array([10, 10, 20]))
    assert [row["native_node_type_id"] for row in report["node_groups"]] == [20, 10]
    for ids in ([1, 1], [0], [0, 1, 2], [0, 3], [-1, 0]):
        modified = deepcopy(raw)
        modified["nodes"][1]["ids"] = np.asarray(ids)
        with pytest.raises(ValueError, match="V1_RAW_"):
            validate_raw_graph(modified, np.array([10, 10, 20]))
    with pytest.raises(ValueError, match="CROSSES_NATIVE_CELL_TYPES"):
        validate_raw_graph(graph(), np.array([10, 10, 20]))


def test_recurrent_coupling_excludes_zero_weights_self_loops_and_receptor_rows():
    raw = graph(4)
    raw["edges"] = [edge([0] * 256 + [1], [1] * 256 + [0]),
                    edge([1, 2, 3], [2, 3, 3], weight=[0, 0, 1], receptor=4)]
    topology = validate_raw_graph(raw, np.ones(4, dtype=int))["topology"]
    assert topology["distinct_directed_interunit_pairs"] == 2
    assert topology["strong"]["largest_neurons"] == 2
    assert topology["weak"]["largest_neurons"] == 2
    assert topology["isolated_neurons"] == 2
    assert not topology["structural_coupled_scale_qualified"]


def test_two_fifty_thousand_cycles_do_not_qualify_one_hundred_thousand_coupling():
    raw = graph(100_000)
    native_types = np.ones(100_000, dtype=int)
    report = validate_raw_graph(raw, native_types)
    assert report["topology"]["structural_coupled_scale_qualified"]
    assert not report["native_dynamics_qualified"]
    assert report["production_episodes"] == 0
    raw["edges"] = [edge(np.arange(50_000), np.roll(np.arange(50_000), -1)),
                    edge(np.arange(50_000, 100_000), np.roll(np.arange(50_000, 100_000), -1))]
    topology = validate_raw_graph(raw, native_types)["topology"]
    assert topology["strong"] == {"components": 2, "largest_neurons": 50_000}
    assert not topology["structural_coupled_scale_qualified"]


def test_directed_feedforward_coupling_does_not_require_mutual_reachability():
    raw = graph(100_000)
    raw["edges"] = [edge(np.arange(99_999), np.arange(1, 100_000))]
    report = validate_raw_graph(raw, np.ones(100_000, dtype=int))
    assert report["topology"]["weak"]["largest_neurons"] == 100_000
    assert report["topology"]["strong"]["largest_neurons"] == 1
    assert report["topology"]["structural_coupled_scale_qualified"]
    assert not report["native_dynamics_qualified"]


def test_delay_alignment_requires_actual_association_change_not_just_variation():
    raw = graph()
    raw["edges"] = [edge([0, 2, 1], [2, 0, 1], delay=[9, 1, 5])]
    result = validate_raw_graph(raw, np.ones(3, dtype=int))["loader_delay_alignment"]
    assert result["edge_association_mismatches"] == 3
    assert result["examples"][0] == {"sorted_position": 0, "raw_edge_index": 1,
        "source": 2, "target": 0, "receptor_type": 1, "weight": 1.0,
        "loader_returned_delay": 9.0, "edge_associated_delay": 1.0}
    raw["edges"][0]["params"]["delay"] = 2.0
    assert validate_raw_graph(raw, np.ones(3, dtype=int))["loader_delay_alignment"]["status"] == "ALIGNED_FOR_THIS_PAYLOAD"
    raw["edges"] = [edge([2, 1, 0], [0, 1, 2], delay=[1, 5, 9])]
    assert validate_raw_graph(raw, np.ones(3, dtype=int))["loader_delay_alignment"]["edge_association_mismatches"] == 0


def test_duplicate_indices_and_scalar_delays_follow_author_sort():
    raw = graph()
    raw["edges"] = [edge([0, 0], [2, 2], delay=9), edge([1, 2], [0, 0], delay=1)]
    result = validate_raw_graph(raw, np.ones(3, dtype=int))["loader_delay_alignment"]
    assert result["edge_association_mismatches"] == 4
    assert len(result["examples"]) == 4


@pytest.mark.parametrize("mutation,error", [
    (lambda r: r["edges"][0].update(source=np.array([0.0, 1.0, 2.0])), "INTEGRAL_VECTOR"),
    (lambda r: r["edges"][0].update(target=np.array([0, 1, 3])), "EDGE_ENDPOINT"),
    (lambda r: r["edges"][0]["params"].update(weight=np.array([1e100, 1, 1])), "NONFINITE_NATIVE_FLOAT32"),
    (lambda r: r["edges"][0]["params"].update(delay=[1, 2]), "DELAY_SHAPE"),
    (lambda r: r["edges"][0]["params"].update(receptor_type=5), "RECEPTOR_TYPE"),
    (lambda r: r["nodes"][0]["params"].update(k=[0.1]), "NUMERIC_SHAPE"),
])
def test_source_data_errors_fail_explicitly(mutation, error):
    raw = graph()
    mutation(raw)
    with pytest.raises(ValueError, match=error):
        validate_raw_graph(raw, np.ones(3, dtype=int))


def archive_fixture(root, n=3, legacy_declarations=False):
    nodes, types, loader = root / "native.h5", root / "types.csv", root / "load_sparse.py"
    with h5py.File(nodes, "w") as handle:
        group = handle.create_group("nodes/v1")
        group["node_id"] = np.arange(n)
        group["node_type_id"] = np.ones(n, dtype=int)
        group["node_group_id"] = np.zeros(n, dtype=int)
        group["node_group_index"] = np.roll(np.arange(n), 1)
        data = group.create_group("0")
        for name in ("x", "y", "z", "tuning_angle"):
            data[name] = np.arange(n, dtype=float)
    types.write_text("node_type_id model_type model_template dynamics_params pop_name gaba_synapse ei location population layer\n"
                     "1 point_process nest:glif_psc 1_glif_psc.json fixture NULL e Cortex v1 L5\n")
    loader.write_text("# checksum-bound loader fixture\n")
    report, inspection = root / "metadata.json", root / "inspection.json"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report.write_text(json.dumps({"status": "NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS",
        "nodes_sha256": sha(nodes), "node_types_sha256": sha(types), "cortical_units": n}))
    inspection.write_text(json.dumps({"loader_source_inspection": {"files": [
        {"path": "load_sparse.py", "sha256": sha(loader)}]}}))
    archive = root / "graph.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
        target.write(nodes, ARCHIVE_MEMBERS["nodes"])
        if legacy_declarations:
            text = types.read_text()
            text = text.replace(' layer\n', '\n').replace(' L5\n', '\n')
            text = text.replace('glif_psc', 'glif_lif_asc_psc')
            target.writestr(ARCHIVE_MEMBERS["types"], text)
        else:
            target.write(types, ARCHIVE_MEMBERS["types"])
        target.writestr(ARCHIVE_MEMBERS["graph"], pickle.dumps(graph(n)))
    receipt = root / "download.json"
    receipt.write_text(json.dumps({"expected_bytes": archive.stat().st_size,
        "expected_md5": hashlib.md5(archive.read_bytes()).hexdigest()}))
    return {"archive": archive, "download_receipt": receipt, "reference_nodes": nodes,
            "reference_types": types, "reference_report": report, "loader": loader,
            "loader_inspection": inspection, "output": root / "result.json"}


def test_real_entry_point_inspects_fixture_without_promoting_small_scale(tmp_path):
    inputs = archive_fixture(tmp_path)
    argv = [value for name, path in inputs.items() for value in ("--" + name.replace("_", "-"), str(path))]
    assert main(argv) == 2
    result = json.loads(inputs["output"].read_text())
    assert result["native_metadata_identity_verified"]
    assert result["cortical_neurons"] == 3
    assert not result["topology"]["structural_coupled_scale_qualified"]
    assert not result["source_scientific_fidelity_qualified"]


def test_legacy_model_declarations_are_reported_without_hiding_identity_changes():
    reference = {1: {"node_type_id": "1", "model_type": "point_process", "pop_name": "e5Rbp4",
                     "gaba_synapse": "NULL", "ei": "e", "location": "Cortex", "population": "v1",
                     "layer": "L5", "model_template": "nest:glif_psc", "dynamics_params": "1_glif_psc.json"}}
    archived = deepcopy(reference)
    archived[1].pop("layer")
    archived[1].update(model_template="nest:glif_lif_asc_psc", dynamics_params="1_glif_lif_asc_psc.json")
    result = compare_type_metadata(reference, archived)
    assert result["identity_fields_match"] and not result["all_declarations_equal"]
    assert not result["cell_model_parameter_equivalence_verified"]
    assert set(result["differences"][0]["fields"]) == {"layer", "model_template", "dynamics_params"}
    archived[1]["ei"] = "i"
    with pytest.raises(ValueError, match="CELL_IDENTITY_DIFFERS"):
        compare_type_metadata(reference, archived)


def test_full_size_graph_cli_keeps_legacy_declarations_without_native_promotion(tmp_path):
    inputs = archive_fixture(tmp_path, n=100_000, legacy_declarations=True)
    argv = [value for name, path in inputs.items() for value in ("--" + name.replace("_", "-"), str(path))]
    assert main(argv) == 0
    report = json.loads(inputs["output"].read_text())
    assert report["native_metadata_identity_verified"]
    assert report["topology"]["structural_coupled_scale_qualified"]
    assert not report["cell_type_comparison"]["all_declarations_equal"]
    assert not report["cell_type_comparison"]["cell_model_parameter_equivalence_verified"]
    assert not report["native_dynamics_qualified"]
    assert not report["source_scientific_fidelity_qualified"]
    assert report["production_episodes"] == 0


@pytest.mark.parametrize("change", ["size", "digest", "reference", "loader"])
def test_untrusted_archive_or_reference_is_rejected_before_pickle(tmp_path, change):
    inputs = archive_fixture(tmp_path)
    if change == "size":
        with inputs["archive"].open("ab") as stream:
            stream.write(b'x')
    elif change == "digest":
        data = bytearray(inputs["archive"].read_bytes());data[50] ^= 1
        inputs["archive"].write_bytes(data)
    elif change == "reference":
        inputs["reference_types"].write_text("changed")
    else:
        inputs["loader"].write_text("changed")
    with patch.object(v1_graph_identity.pickle, "load") as load:
        with pytest.raises(ValueError):
            inspect_archive(**inputs)
        load.assert_not_called()

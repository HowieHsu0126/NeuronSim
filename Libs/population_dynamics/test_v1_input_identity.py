"""Behavior tests for raw input identity, timing and source-bound execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from population_dynamics.v1_input_identity import INPUT_MEMBER, inspect_archive, inspect_inputs


def population():
    return [
        {"ids": np.array([0, 1, 2]), "spikes": [
            np.array([0.0, 1000.0, 1000.125, 1000.5, 1999.999, 2000.0, 3000.0]),
            np.array([1001.0]), np.array([])]},
        [{"source": np.array([0, 1, 2]), "target": np.array([2, 1, 0]),
          "params": {"receptor_type": 1, "weight": np.array([1.0, 2.0, 3.0]),
                     "delay": np.array([1.0, 2.0, 3.0])}}],
    ]


def inspect(raw):
    return inspect_inputs(raw, 3, duration=3000.0, dt=0.25)


def test_clock_boundaries_multiplicity_and_raw_inputs_preserved():
    raw = [population()]
    before = pickle.dumps(raw)
    report = inspect(raw)
    spike = report["populations"][0]["spikes"]
    assert spike["raw_events"] == 8
    assert spike["author_interval_events"] == 4
    assert spike["author_within_unit_bin_collisions"] == 1
    assert spike["events_not_exactly_on_source_config_dt"] == 2
    assert spike["exact_boundary_event_counts"] == {"0.0": 1, "1000.0": 1, "2000.0": 1, "3000.0": 1}
    assert pickle.dumps(raw) == before
    assert report["source_config_clock"] == {"duration": 3000.0, "dt": 0.25}
    assert report["author_convenience_clock"] == {"start": 1000.0, "duration": 1000.0, "dt": 1.0}
    assert report["production_episodes"] == 0
    assert report["original_pointnet_input_equivalence_qualified"] is False


def test_delay_permutation_preserves_complete_edge_association():
    row = inspect([population()])["populations"][0]
    alignment = row["loader_delay_alignment"]
    assert alignment["mismatches"] == 2
    assert alignment["examples"][0] == {
        "sorted_position": 0, "raw_edge_index": 2, "source": 2, "target": 0,
        "receptor": 1, "weight": 3.0, "author_returned_delay": 1.0, "edge_associated_delay": 3.0}


def test_constant_scalar_delays_make_sorting_numerically_invisible():
    raw = [population()]
    raw[0][1][0]["params"]["delay"] = 2.5
    row = inspect(raw)["populations"][0]
    assert row["loader_delay_alignment"]["mismatches"] == 0
    assert row["delay_min"] == row["delay_max"] == 2.5


def test_unordered_input_ids_remain_population_local_without_reindexing():
    raw = [population(), population()]
    raw[1][0]["ids"] = np.array([2, 0, 1])
    report = inspect(raw)
    assert report["populations"][1]["ids_already_sorted"] is False
    assert [p["input_units"] for p in report["populations"]] == [3, 3]


@pytest.mark.parametrize("ids", [np.array([0, 0, 2]), np.array([0, 2, 3]),
                                  np.array([-1, 0, 1]), np.array([0.0, 1.0, 2.0])])
def test_duplicate_gapped_negative_and_nonintegral_ids_rejected(ids):
    raw = [population()]
    raw[0][0]["ids"] = ids
    with pytest.raises(ValueError):
        inspect(raw)


@pytest.mark.parametrize("key,value", [
    ("source", np.array([0, 1, 3])), ("target", np.array([2, 3, 0])),
    ("target", np.array([0, 1])), ("source", np.array([0, -1, 2]))])
def test_unknown_or_misaligned_edge_endpoints_rejected(key, value):
    raw = [population()]
    raw[0][1][0][key] = value
    with pytest.raises(ValueError, match="EDGE_ENDPOINT"):
        inspect(raw)


@pytest.mark.parametrize("change", ["missing_train", "nonfinite_time", "time_matrix",
                                   "weight_length", "nonfinite_weight", "negative_delay",
                                   "delay_length", "invalid_receptor"])
def test_invalid_spike_or_edge_data_cannot_be_silently_truncated(change):
    raw = [population()]
    nodes, edges = raw[0]
    params = edges[0]["params"]
    if change == "missing_train":
        nodes["spikes"].pop()
    elif change == "nonfinite_time":
        nodes["spikes"][0][0] = np.nan
    elif change == "time_matrix":
        nodes["spikes"][0] = np.zeros((2, 2))
    elif change == "weight_length":
        params["weight"] = np.array([1.0])
    elif change == "nonfinite_weight":
        params["weight"][0] = np.inf
    elif change == "negative_delay":
        params["delay"][0] = -1.0
    elif change == "delay_length":
        params["delay"] = np.array([1.0, 2.0])
    else:
        params["receptor_type"] = 5
    with pytest.raises(ValueError):
        inspect(raw)


def test_background_collision_matches_actual_pinned_array_assignment():
    bkg = population()
    bkg[1] = [{"source": np.array([1, 0]), "target": np.array([0, 0]),
               "params": {"receptor_type": 2, "weight": np.array([4.0, 2.0]),
                          "delay": np.array([2.0, 1.0])}}]
    report = inspect([population(), bkg])
    flat = report["populations"][1]["author_flat_background_assignment"]
    assert flat["overwritten_rows"] == 1
    assert flat["collision_slots"] == 1
    assert flat["examples"] == [{"target_receptor_slot": 1,
                                 "assigned_float32_weight": 4.0, "raw_rows": 2}]
    assert "background:FLAT_ASSIGNMENT_COLLAPSES_EDGE_ROWS" in report["reconciliation_concerns"]


@pytest.mark.parametrize("source_dtypes,target_dtypes", [
    ((np.uint64, np.uint64), (np.uint64, np.uint64)),
    ((np.uint64, np.int64), (np.int64, np.uint64)),
    ((np.int64, np.uint64), (np.uint64, np.int64)),
    ((np.uint32, np.int64), (np.uint32, np.int64)),
])
def test_unsigned_and_mixed_edge_groups_keep_integer_sort_and_background_assignment(
        source_dtypes, target_dtypes):
    bkg = population()
    bkg[0]["ids"] = bkg[0]["ids"].astype(np.uint64)
    bkg[1] = [
        {"source": np.array([2, 0], dtype=source_dtypes[0]),
         "target": np.array([0, 0], dtype=target_dtypes[0]),
         "params": {"receptor_type": np.int64(2), "weight": np.array([8.0, 2.0]),
                    "delay": np.array([8.0, 2.0])}},
        {"source": np.array([1], dtype=source_dtypes[1]),
         "target": np.array([0], dtype=target_dtypes[1]),
         "params": {"receptor_type": np.int64(2), "weight": np.array([4.0]),
                    "delay": np.array([4.0])}},
    ]
    raw = [population(), bkg]
    before = pickle.dumps(raw)
    report = inspect(raw)
    row = report["populations"][1]
    alignment = row["loader_delay_alignment"]
    assert alignment["mismatches"] == 3
    assert [example["raw_edge_index"] for example in alignment["examples"]] == [1, 2, 0]
    assert [example["source"] for example in alignment["examples"]] == [0, 1, 2]
    assert [example["edge_associated_delay"] for example in alignment["examples"]] == [2.0, 4.0, 8.0]
    assert row["author_flat_background_assignment"] == {
        "rows": 3, "unique_target_receptor_slots": 1, "overwritten_rows": 2,
        "collision_slots": 1,
        "examples": [{"target_receptor_slot": 1, "assigned_float32_weight": 8.0, "raw_rows": 3}],
        "source_spike_times_and_delays_used_by_this_assignment": False,
    }
    assert "population_1:AUTHOR_SORTED_DELAY_ASSOCIATION_DIFFERS" in report["reconciliation_concerns"]
    assert "background:FLAT_ASSIGNMENT_COLLAPSES_EDGE_ROWS" in report["reconciliation_concerns"]
    assert pickle.dumps(raw) == before


def test_empty_unsigned_edge_group_preserves_empty_background_assignment():
    bkg = population()
    bkg[1] = [{"source": np.array([], dtype=np.uint64), "target": np.array([], dtype=np.uint64),
               "params": {"receptor_type": np.int64(1), "weight": np.array([], dtype=np.float64),
                          "delay": np.array([], dtype=np.float64)}}]
    row = inspect([population(), bkg])["populations"][1]
    assert row["raw_edges"] == 0
    assert row["loader_delay_alignment"]["mismatches"] == 0
    assert row["author_flat_background_assignment"]["rows"] == 0
    assert row["author_flat_background_assignment"]["examples"] == []


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_integral_valued_float_edge_ids_are_rejected_before_cast(endpoint):
    raw = [population(), population()]
    edge = raw[1][1][0]
    edge[endpoint] = edge[endpoint].astype(np.float64)
    with pytest.raises(ValueError, match="V1_INTEGRAL_VECTOR_REQUIRED:input_edge_" + endpoint):
        inspect(raw)


@pytest.mark.parametrize("endpoint", ["source", "target"])
def test_out_of_range_uint64_edge_ids_are_rejected_before_cast(endpoint):
    raw = [population(), population()]
    raw[1][1][0][endpoint] = np.array([0, 1, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="V1_INPUT_EDGE_ENDPOINT_INVALID"):
        inspect(raw)


def _delivered(folder: Path, name: str, data: dict, status: str):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "delivery.json").write_text(json.dumps({
        "status": status, "exit_code": 0,
        "files": [{"path": name, "bytes": path.stat().st_size, "sha256": digest}]}))


@pytest.fixture
def archive_inputs(tmp_path):
    acquisition = tmp_path / "acquisition"
    (acquisition / "asset").mkdir(parents=True)
    archive = acquisition / "asset/GLIF_network.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(INPUT_MEMBER, pickle.dumps([population(), population()]))
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    source = {"status": "SOURCE_ASSET_ACQUIRED", "asset_body_acquired": True,
              "source": {"artifact": archive.name, "expected_bytes": archive.stat().st_size, "expected_md5": md5}}
    _delivered(acquisition, "asset/source.json", source, "ARCHIVED_V1_SOURCE_ASSET")
    graph = tmp_path / "graph"
    _delivered(graph, "graph_identity.json",
               {"status": "SOURCE_GRAPH_INSPECTED", "native_metadata_identity_verified": True,
                "cortical_neurons": 3, "source_archive_bytes": archive.stat().st_size,
                "source_archive_md5": md5}, "INSPECTED_REQUIRES_RECONCILIATION")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"target_simulator": "NEST", "run": {"duration": 3000.0, "dt": 0.25}}))
    config_source = tmp_path / "config_source.json"
    config_source.write_text(json.dumps({"sha256": hashlib.sha256(config.read_bytes()).hexdigest()}))
    return dict(source_acquisition_archive=acquisition, graph_inspection_archive=graph,
                point_config=config, point_config_source=config_source, output=tmp_path/"input_identity.json")


def test_source_bound_real_cli_writes_inspection_without_promoting_fixture(archive_inputs):
    command = [sys.executable, "-B", "-m", "population_dynamics.v1_input_identity"]
    for key, value in archive_inputs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    completed = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(archive_inputs["output"].read_text())
    assert report["status"] == "SOURCE_INPUT_INSPECTED"
    assert report["source_scientific_fidelity_qualified"] is False
    assert report["reconciliation_concerns"]
    assert report["input_member_sha256"]


def test_incomplete_receipt_rejected_before_unpickle(archive_inputs, monkeypatch):
    root = archive_inputs["source_acquisition_archive"]
    receipt = json.loads((root/"asset/source.json").read_text())
    receipt["status"] = "PARTIAL"
    receipt["asset_body_acquired"] = False
    _delivered(root, "asset/source.json", receipt, "ARCHIVED_INCOMPLETE_V1_SOURCE_ACQUISITION")
    monkeypatch.setattr(pickle, "load", lambda *_: pytest.fail("partial archive unpickled"))
    with pytest.raises(ValueError, match="COMPLETE_SOURCE"):
        inspect_archive(**archive_inputs)


@pytest.mark.parametrize("change", ["graph_identity", "archive_bytes", "config_bytes", "delivered_json"])
def test_changed_provenance_or_payload_rejected(archive_inputs, change, monkeypatch):
    acquisition = archive_inputs["source_acquisition_archive"]
    if change == "graph_identity":
        root = archive_inputs["graph_inspection_archive"]
        data = json.loads((root/"graph_identity.json").read_text())
        data["source_archive_md5"] = "0" * 32
        _delivered(root, "graph_identity.json", data, "INSPECTED")
    elif change == "archive_bytes":
        path = acquisition/"asset/GLIF_network.zip"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    elif change == "config_bytes":
        archive_inputs["point_config"].write_text("{}")
    else:
        (acquisition/"asset/source.json").write_text("{}")
    monkeypatch.setattr(pickle, "load", lambda *_: pytest.fail("changed archive unpickled"))
    with pytest.raises(ValueError):
        inspect_archive(**archive_inputs)


def test_cli_failure_is_nonzero_and_does_not_emit_authoritative_output(archive_inputs):
    archive_inputs["point_config"].write_text("{}")
    command = [sys.executable, "-B", "-m", "population_dynamics.v1_input_identity"]
    for key, value in archive_inputs.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    completed = subprocess.run(command, capture_output=True, text=True, env=os.environ)
    assert completed.returncode != 0
    assert "V1_ORIGINAL_POINTNET_CONFIG_CHANGED" in completed.stderr
    assert not archive_inputs["output"].exists()

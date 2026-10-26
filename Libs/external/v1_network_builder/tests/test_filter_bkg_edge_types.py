import importlib.util
from pathlib import Path
import subprocess
import sys

import h5py
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "filter_bkg_edge_types", ROOT / "filter_bkg_edge_types.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def bkg_files(tmp_path):
    tuned = pd.DataFrame(
        {
            "edge_type_id": [11, 33, 22],
            "target_query": [
                "node_type_id=='202'",
                "node_type_id=='303'",
                "node_type_id=='101'",
            ],
            "source_query": "*",
            "delay": 1.0,
            "weight_function": "weight_function_bkg",
            "dynamics_params": ["exc_to_pv.json", "exc_to_sst.json", "exc_to_e4.json"],
            "model_template": "static_synapse",
            "syn_weight": [13.25, 19.5, 9.8125],
        }
    )
    tuned_path = tmp_path / "tuned.csv"
    tuned.to_csv(tuned_path, sep=" ", index=False)
    edges_path = tmp_path / "bkg_v1_edges.h5"
    with h5py.File(edges_path, "w") as handle:
        edges = handle.create_group("edges/bkg_to_v1")
        edges["edge_type_id"] = [900, 400, 900]
        edges["target_node_id"] = [42, 9, 17]
        edges["0/nsyns"] = [1, 1, 1]
    with h5py.File(tmp_path / "v1_nodes.h5", "w") as handle:
        nodes = handle.create_group("nodes/v1")
        nodes["node_id"] = [9, 17, 42]
        nodes["node_type_id"] = [202, 101, 101]
    return tuned_path, edges_path, tmp_path / "bkg_v1_edge_types.csv"


def test_subset_preserves_tuned_values_and_actual_edge_ids(bkg_files):
    tuned_path, edges_path, output_path = bkg_files
    originals = [source.read_bytes() for source in (tuned_path, edges_path)]

    MODULE.filter_bkg_edge_types(tuned_path, edges_path, output_path)

    actual = pd.read_csv(output_path, sep=" ")
    expected = pd.read_csv(tuned_path, sep=" ").iloc[[0, 2]].reset_index(drop=True)
    expected["edge_type_id"] = [400, 900]
    pd.testing.assert_frame_equal(actual, expected)
    assert [source.read_bytes() for source in (tuned_path, edges_path)] == originals


def test_implicit_node_ids(bkg_files):
    tuned_path, edges_path, output_path = bkg_files
    with h5py.File(edges_path, "r+") as handle:
        handle["edges/bkg_to_v1/target_node_id"][:] = [2, 0, 1]
    with h5py.File(edges_path.with_name("v1_nodes.h5"), "r+") as handle:
        del handle["nodes/v1/node_id"]

    MODULE.filter_bkg_edge_types(tuned_path, edges_path, output_path)

    actual = pd.read_csv(output_path, sep=" ")
    assert actual["edge_type_id"].tolist() == [400, 900]
    assert actual["syn_weight"].tolist() == [13.25, 9.8125]


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing_model", "No tuned BKG weights"),
        ("duplicate_model", "duplicate target cell-model"),
        ("ambiguous_edge_type", "targets multiple cell-model"),
        ("unknown_node", "absent from V1 nodes"),
        ("duplicate_node", "duplicate node IDs"),
        ("invalid_query", "target_query must identify"),
        ("initialization_table", "Expected a tuned BKG"),
    ],
)
def test_invalid_mapping_preserves_existing_output(bkg_files, defect, message):
    tuned_path, edges_path, output_path = bkg_files
    tuned = pd.read_csv(tuned_path, sep=" ")
    if defect == "missing_model":
        tuned = tuned.iloc[:2]
    elif defect == "duplicate_model":
        tuned.loc[1, "target_query"] = tuned.loc[0, "target_query"]
    elif defect == "invalid_query":
        tuned.loc[0, "target_query"] = "pop_name=='e4'"
    elif defect == "initialization_table":
        tuned = tuned.rename(columns={"target_query": "model_id"})
    elif defect == "ambiguous_edge_type":
        with h5py.File(edges_path, "r+") as handle:
            handle["edges/bkg_to_v1/edge_type_id"][:] = [900, 900, 900]
    elif defect == "unknown_node":
        with h5py.File(edges_path, "r+") as handle:
            handle["edges/bkg_to_v1/target_node_id"][:] = [999, 9, 17]
    elif defect == "duplicate_node":
        with h5py.File(edges_path.with_name("v1_nodes.h5"), "r+") as handle:
            handle["nodes/v1/node_id"][:] = [9, 9, 42]
    tuned.to_csv(tuned_path, sep=" ", index=False)
    output_path.write_text("existing network CSV\n")

    with pytest.raises(ValueError, match=message):
        MODULE.filter_bkg_edge_types(tuned_path, edges_path, output_path)

    assert output_path.read_text() == "existing network CSV\n"


def test_cli_accepts_explicit_nodes_path(bkg_files):
    tuned_path, edges_path, output_path = bkg_files
    nodes_path = edges_path.with_name("v1_nodes.h5").rename(
        edges_path.with_name("custom_nodes.h5")
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "filter_bkg_edge_types.py"),
            str(tuned_path),
            str(edges_path),
            str(output_path),
            "--v1-nodes",
            str(nodes_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = pd.read_csv(output_path, sep=" ")
    assert actual["edge_type_id"].tolist() == [400, 900]
    assert actual["syn_weight"].tolist() == [13.25, 9.8125]


def test_snakefile_override_uses_tuned_table_and_target_nodes():
    rule = (ROOT / "Snakefile").read_text().split("rule bkg_edges_override:")[1]
    rule = rule.split("\nrule ")[0]
    assert 'bkg_weights="precomputed_props/bkg_v1_edge_types.csv"' in rule
    assert 'v1_nodes="{network_name}/network/v1_nodes.h5"' in rule
    assert "glif_props/bkg_weights_model.csv" not in rule
    assert "{input.bkg_weights}" in rule
    assert "--v1-nodes {input.v1_nodes}" in rule

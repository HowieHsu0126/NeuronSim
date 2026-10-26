import argparse
from pathlib import Path

import h5py
import pandas as pd


def filter_bkg_edge_types(
    bkg_weights_path, edges_path, output_path, v1_nodes_path=None
):
    bkg_weights = pd.read_csv(bkg_weights_path, sep=" ")
    if not {"target_query", "syn_weight"}.issubset(bkg_weights.columns):
        raise ValueError(
            "Expected a tuned BKG edge-type table with target_query and syn_weight."
        )
    model_ids = bkg_weights["target_query"].str.extract(
        r"""^\s*node_type_id\s*==\s*['"]?(\d+)['"]?\s*$""", expand=False
    )
    if model_ids.isna().any():
        raise ValueError("Tuned BKG target_query must identify a node_type_id.")
    model_ids = model_ids.astype("int64")
    if model_ids.duplicated().any():
        raise ValueError("Tuned BKG table has duplicate target cell-model IDs.")
    bkg_weights.index = model_ids

    with h5py.File(edges_path, "r") as h5:
        edges = h5["edges/bkg_to_v1"]
        edge_type_ids = edges["edge_type_id"][()]
        target_node_ids = edges["target_node_id"][()]

    if v1_nodes_path is None:
        v1_nodes_path = Path(edges_path).with_name("v1_nodes.h5")
    with h5py.File(v1_nodes_path, "r") as h5:
        nodes = h5["nodes/v1"]
        node_types = pd.Series(
            nodes["node_type_id"][()],
            index=nodes["node_id"][()] if "node_id" in nodes else None,
        )
    if not node_types.index.is_unique:
        raise ValueError("V1 nodes contain duplicate node IDs.")
    target_models = node_types.reindex(target_node_ids)
    if target_models.isna().any():
        raise ValueError("BKG edges reference target node IDs absent from V1 nodes.")

    edge_models = (
        pd.DataFrame(
            {"edge_type_id": edge_type_ids, "model_id": target_models.to_numpy()}
        )
        .drop_duplicates()
        .sort_values("edge_type_id")
    )
    if edge_models["edge_type_id"].duplicated().any():
        raise ValueError("A BKG edge_type_id targets multiple cell-model IDs.")
    missing_models = pd.Index(edge_models["model_id"]).difference(bkg_weights.index)
    if len(missing_models):
        raise ValueError(
            f"No tuned BKG weights for target cell-model IDs: {missing_models.tolist()}"
        )

    edge_types = bkg_weights.loc[edge_models["model_id"]].copy()
    edge_types["edge_type_id"] = edge_models["edge_type_id"].to_numpy()
    edge_types.to_csv(output_path, sep=" ", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply tuned BKG weights by target cell model, preserving edge IDs."
    )
    parser.add_argument("bkg_weights_path", help="Precomputed tuned BKG edge-type CSV")
    parser.add_argument("edges_path")
    parser.add_argument("output_path")
    parser.add_argument(
        "--v1-nodes", help="V1 nodes HDF5; defaults to v1_nodes.h5 beside the edges"
    )
    args = parser.parse_args()

    filter_bkg_edge_types(
        args.bkg_weights_path, args.edges_path, args.output_path, args.v1_nodes
    )

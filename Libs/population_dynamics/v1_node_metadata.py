"""Check the acquired native V1 unit table and its source cell-type join.

This inspects static metadata only. No graph, native trajectory or fidelity is
qualified by a successful metadata check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from .provenance import freeze_json, sha256_file


def check(nodes: Path, node_receipt: Path, types: Path, type_receipt: Path, output: Path):
    node_source, type_source = [json.loads(path.read_text()) for path in (node_receipt, type_receipt)]
    if sha256_file(nodes) != node_source["sha256"] or sha256_file(types) != type_source["sha256"]:
        raise ValueError("V1_ACQUIRED_METADATA_CHANGED")
    with types.open() as stream:
        rows = list(csv.DictReader(stream, delimiter=" "))
    by_type = {int(row["node_type_id"]): row for row in rows}
    if len(by_type) != len(rows):
        raise ValueError("V1_NODE_TYPE_DUPLICATE")
    with h5py.File(nodes) as handle:
        group = handle["nodes/v1"]
        identities = group["node_id"][...]
        type_ids = group["node_type_id"][...]
        groups = group["node_group_id"][...]
        indices = group["node_group_index"][...]
        n = len(identities)
        if (n != 230924 or len(np.unique(identities)) != n or not np.array_equal(np.sort(indices), np.arange(n))
                or not np.all(groups == 0) or type_ids.shape != identities.shape):
            raise ValueError("V1_SOURCE_UNIT_IDENTITY_OR_GROUP_LAYOUT_MISMATCH")
        if set(map(int, np.unique(type_ids))) != set(by_type):
            raise ValueError("V1_SOURCE_CELL_TYPE_JOIN_INCOMPLETE")
        positions = np.column_stack([group["0"][axis][...][indices] for axis in ("x", "y", "z")])
        tuning = group["0/tuning_angle"][...][indices]
        if not np.isfinite(positions).all() or not np.isfinite(tuning).all():
            raise ValueError("V1_NONFINITE_UNIT_METADATA")
    type_counts = Counter(map(int, type_ids))
    by_population, by_layer, by_ei = Counter(), Counter(), Counter()
    for type_id, count in type_counts.items():
        row = by_type[type_id]
        if row["population"] != "v1" or row["location"] != "Cortex" or row["model_type"] != "point_process":
            raise ValueError("V1_SOURCE_UNITS_ARE_NOT_ALL_CORTICAL_POINT_NEURONS")
        by_population[row["pop_name"]] += count
        by_layer[row["layer"]] += count
        by_ei[row["ei"]] += count
    radius = np.hypot(positions[:, 0], positions[:, 2])
    report = dict(status="NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS",
                  cortical_units=n, source_cell_models=len(by_type), source_population_count=len(by_population),
                  population_counts=dict(by_population), layer_counts=dict(by_layer), ei_counts=dict(by_ei),
                  model_templates=sorted({row["model_template"] for row in rows}),
                  coordinate_arrays=["x", "y", "z"], coordinate_row_mapping="node_group_index",
                  radial_axes=["x", "z"], radial_subset_threshold_source_units=400.0,
                  radial_subset_units=int(np.count_nonzero(radius < 400.0)),
                  maximum_radius_source_units=float(radius.max()),
                  count_semantics="all_native_cortical_units_excluding_LGN_and_background_generators",
                  radial_subset_is_separate_from_full_cortical_count=True,
                  nodes_sha256=node_source["sha256"], node_types_sha256=type_source["sha256"],
                  complete_model_acquired=False, connectivity_qualified=False,
                  native_episodes_completed=0, scientific_fidelity_qualified=False)
    freeze_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nodes", "node-receipt", "types", "type-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.nodes, args.node_receipt, args.types, args.type_receipt, args.output)))


if __name__ == "__main__":
    main()

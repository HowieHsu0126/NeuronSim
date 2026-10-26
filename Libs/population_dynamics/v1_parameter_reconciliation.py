"""Compare inspected V1 graph cell parameters with archived Allen GLIF3 references.

This is an identity/value comparison, not native trajectory qualification. The
published loader's nine parameters are compared exactly and after its float32
conversion; initialization and model flags absent from the raw graph stay visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
import zipfile

import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_cell_parameters import BMTK_CONVERTER_SHA256, GLIF3_TEMPLATE
from .v1_graph_identity import ARCHIVE_MEMBERS, PARAMETER_SHAPES, _finite_float32, _h5_metadata, _integers

# The author uses exp(-dt*k) for ASC decay. The pinned BMTK converter calls
# the same inverse-ms rate asc_decay. All other loader parameter names agree.
REFERENCE_NAMES = {name: "asc_decay" if name == "k" else name for name in PARAMETER_SHAPES}
PARAMETER_UNITS = {
    "V_th": "mV", "g": "nS", "E_L": "mV", "k": "1/ms", "C_m": "pF",
    "V_reset": "mV", "tau_syn": "ms", "t_ref": "ms", "asc_amps": "pA",
}


def compare_parameters(raw_nodes: list[dict], native_type_ids: np.ndarray,
                       references: list[dict]) -> dict:
    """Join by source node identity, never by group/table iteration order."""
    types = _integers(native_type_ids, "native_type_ids")
    if not len(types) or not references or any(type(row["specimen_id"]) is not int for row in references):
        raise ValueError("V1_NONEMPTY_INTEGRAL_REFERENCE_COHORT_REQUIRED")
    by_id = {row["specimen_id"]: row for row in references}
    if len(by_id) != len(references) or set(by_id) != set(map(int, types)):
        raise ValueError("V1_PARAMETER_REFERENCE_COHORT_DIFFERS")
    seen = np.zeros(len(types), dtype=bool)
    seen_types = set()
    comparisons = []
    for index, group in enumerate(raw_nodes):
        ids = _integers(group["ids"], "node_ids")
        if (not len(ids) or np.any(ids < 0) or np.any(ids >= len(types))
                or len(np.unique(ids)) != len(ids) or seen[ids].any()):
            raise ValueError("V1_PARAMETER_NODE_IDENTITY_INVALID")
        identities = np.unique(types[ids])
        if len(identities) != 1 or int(identities[0]) in seen_types:
            raise ValueError("V1_PARAMETER_GROUP_IDENTITY_INVALID")
        identity = int(identities[0])
        seen[ids] = True
        seen_types.add(identity)
        reference = by_id[identity]
        expected = reference["converted_parameters"]
        if (reference["neuronal_model_template_id"] != GLIF3_TEMPLATE
                or reference["converted_template"] != "nest:glif_psc"
                or expected.get("after_spike_currents") is not True
                or expected.get("spike_dependent_threshold") is not False
                or expected.get("adapting_threshold") is not False):
            raise ValueError("V1_PARAMETER_REFERENCE_FAMILY_DIFFERS")
        params = group["params"]
        if set(PARAMETER_SHAPES) - set(params):
            raise ValueError("V1_LOADER_PARAMETER_MISSING")
        values = {}
        for name, shape in PARAMETER_SHAPES.items():
            other = REFERENCE_NAMES[name]
            if other not in expected:
                raise ValueError("V1_REFERENCE_PARAMETER_MISSING:" + other)
            actual32 = _finite_float32(params[name], shape, name)
            expected32 = _finite_float32(expected[other], shape, other)
            actual = np.asarray(params[name], dtype=np.float64)
            target = np.asarray(expected[other], dtype=np.float64)
            values[name] = dict(reference_name=other, unit=PARAMETER_UNITS[name],
                raw_value=actual.tolist(), reference_value=target.tolist(),
                raw_dtype=str(np.asarray(params[name]).dtype),
                exact_equal=bool(np.array_equal(actual, target)),
                author_float32_equal=bool(np.array_equal(actual32, expected32)),
                max_absolute_difference=float(np.max(np.abs(actual - target))))
        unassessed = sorted(set(params) - set(PARAMETER_SHAPES))
        reference_only = sorted(set(expected) - set(REFERENCE_NAMES.values()))
        comparisons.append(dict(raw_group_index=index, specimen_id=identity,
            neuronal_model_id=reference["neuronal_model_id"], neurons=len(ids), parameters=values,
            unassessed_raw_parameters=unassessed,
            unassessed_raw_parameter_values={name: np.asarray(params[name]).tolist() for name in unassessed},
            reference_parameters_absent_from_loader=reference_only,
            reference_parameter_values_absent_from_loader={
                name: np.asarray(expected[name]).tolist() for name in reference_only}))
    if not seen.all() or seen_types != set(by_id):
        raise ValueError("V1_PARAMETER_NODE_COVERAGE_INCOMPLETE")
    exact = all(value["exact_equal"] for row in comparisons for value in row["parameters"].values())
    float32 = all(value["author_float32_equal"] for row in comparisons for value in row["parameters"].values())
    unknown = any(row["unassessed_raw_parameters"] for row in comparisons)
    return dict(status="LOADER_PARAMETERS_MATCH_REFERENCE" if float32 and not unknown
                else "LOADER_PARAMETERS_REQUIRE_RECONCILIATION",
        cortical_neurons=len(types), cell_types=len(comparisons),
        compared_parameter_arrays=len(comparisons) * len(PARAMETER_SHAPES),
        loader_parameters_exactly_equal=exact,
        loader_parameters_equal_at_author_float32=float32,
        all_raw_parameters_assessed=not unknown, cells=comparisons,
        comparison_rule="Exact values and exact equality after the pinned author float32 conversion; no fitted tolerance.",
        scope="Nine loader parameters only. Missing initialization/model flags and extra raw parameters are not inferred.",
        source_network_parameter_equality_verified=False,
        native_dynamics_qualified=False, source_scientific_fidelity_qualified=False,
        production_episodes=0)


def _delivered_json(archive: Path, name: str) -> tuple[dict, dict]:
    delivery = json.loads((archive / "delivery.json").read_text())
    rows = [row for row in delivery["files"] if row["path"] == name]
    path = archive / name
    if len(rows) != 1 or path.stat().st_size != rows[0]["bytes"] or sha256_file(path) != rows[0]["sha256"]:
        raise ValueError("V1_PARAMETER_INPUT_DELIVERY_DIFFERS:" + name)
    return json.loads(path.read_text()), delivery


def reconcile_archive(source_archive: Path, graph_inspection_archive: Path,
                      parameter_reference_archive: Path, output: Path) -> dict:
    """Reuse completed graph qualification; verify the two consumed ZIP members."""
    started = time.monotonic()
    graph, graph_delivery = _delivered_json(graph_inspection_archive, "graph_identity.json")
    if (graph_delivery["status"] not in ("INSPECTED", "INSPECTED_REQUIRES_RECONCILIATION")
            or graph["status"] != "SOURCE_GRAPH_INSPECTED"
            or graph["native_metadata_identity_verified"] is not True):
        raise ValueError("V1_COMPLETED_GRAPH_INSPECTION_REQUIRED")
    reference, _ = _delivered_json(parameter_reference_archive, "reference/intake.json")
    if (reference["status"] != "REFERENCE_PARAMETERS_ACQUIRED"
            or reference["bmtk_converter_sha256"] != BMTK_CONVERTER_SHA256
            or len(reference["references"]) != reference["acquired_references"]
            or reference["acquired_references"] != reference["expected_specimens"]):
        raise ValueError("V1_COMPLETE_PARAMETER_REFERENCE_REQUIRED")
    if source_archive.stat().st_size != graph["source_archive_bytes"]:
        raise ValueError("V1_INSPECTED_ARCHIVE_SIZE_DIFFERS")
    # The completed graph report already verified the whole publisher checksum.
    # Compare the exact member bytes it consumed before unpickling again; do not
    # repeat the topology calculation or reinterpret a changed source payload.
    with tempfile.TemporaryDirectory(prefix="v1-parameters-") as temporary:
        root = Path(temporary)
        paths = {}
        with zipfile.ZipFile(source_archive) as archive:
            for key in ("graph", "nodes"):
                member = ARCHIVE_MEMBERS[key]
                if archive.namelist().count(member) != 1:
                    raise ValueError("V1_PARAMETER_MEMBER_NOT_UNIQUE:" + member)
                path = root / Path(member).name
                digest = hashlib.sha256()
                with archive.open(member) as src, path.open("xb") as dst:
                    for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        dst.write(block)
                if digest.hexdigest() != graph["member_sha256"][member]:
                    raise ValueError("V1_INSPECTED_MEMBER_CHANGED:" + member)
                paths[key] = path
        native = _h5_metadata(paths["nodes"])
        with paths["graph"].open("rb") as stream:
            raw = pickle.load(stream)
        report = compare_parameters(raw["nodes"], native["node_type_id"], reference["references"])
        identities = [{"raw_group_index": row["raw_group_index"],
                       "native_node_type_id": row["specimen_id"], "neurons": row["neurons"]}
                      for row in report["cells"]]
        if identities != graph["node_groups"] or report["cortical_neurons"] != graph["cortical_neurons"]:
            raise ValueError("V1_INSPECTED_PARAMETER_GROUPS_DIFFER")
    report.update(graph_inspection_archive=str(graph_inspection_archive),
        graph_inspection_sha256=sha256_file(graph_inspection_archive / "graph_identity.json"),
        parameter_reference_archive=str(parameter_reference_archive),
        parameter_reference_sha256=sha256_file(parameter_reference_archive / "reference/intake.json"),
        source_archive=str(source_archive),
        source_archive_md5_reused_from_graph_inspection=graph["source_archive_md5"],
        consumed_member_sha256={ARCHIVE_MEMBERS[key]: graph["member_sha256"][ARCHIVE_MEMBERS[key]]
                                for key in ("graph", "nodes")},
        elapsed_seconds=time.monotonic() - started)
    freeze_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-archive", "graph-inspection-archive", "parameter-reference-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = reconcile_archive(**vars(args))
    print(json.dumps({key: value for key, value in report.items() if key != "cells"}))
    return 0 if report["status"] == "LOADER_PARAMETERS_MATCH_REFERENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

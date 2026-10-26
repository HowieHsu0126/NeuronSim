"""Inspect the complete author V1 graph before native dynamics qualification.

Only an archive bound to the acquired size/MD5 is unpickled. This checks cortical
identities, nonzero recurrent coupling and the pinned loader's delay association;
it does not qualify a simulator, a task protocol, or a production episode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import time
import zipfile

import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .provenance import freeze_json, sha256_file

MINIMUM_COUPLED_NEURONS = 100_000
PARAMETER_SHAPES = {
    "V_th": (), "g": (), "E_L": (), "k": (2,), "C_m": (),
    "V_reset": (), "tau_syn": (4,), "t_ref": (), "asc_amps": (2,),
}
ARCHIVE_MEMBERS = {
    "graph": "GLIF_network2/network_dat.pkl",
    "nodes": "GLIF_network2/network/v1_nodes.h5",
    "types": "GLIF_network2/network/v1_node_types.csv",
}
TYPE_IDENTITY_FIELDS = ("node_type_id", "model_type", "pop_name", "gaba_synapse",
                        "ei", "location", "population")


def _integers(value, label: str) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("V1_INTEGRAL_VECTOR_REQUIRED:" + label)
    return values


def _finite_float32(value, shape: tuple, label: str) -> np.ndarray:
    values = np.asarray(value)
    if values.shape != shape or values.dtype.kind not in "iuf":
        raise ValueError("V1_NUMERIC_SHAPE_INVALID:" + label)
    with np.errstate(over="ignore", invalid="ignore"):
        result = values.astype(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(result).all():
        raise ValueError("V1_NONFINITE_NATIVE_FLOAT32:" + label)
    return result


def validate_raw_graph(raw: dict, native_type_ids: np.ndarray) -> dict:
    """Use native node IDs as indices, as required by the pinned author loader."""
    n = len(native_type_ids)
    seen = np.zeros(n, dtype=bool)
    groups = []
    for group in raw["nodes"]:
        ids = _integers(group["ids"], "node_ids")
        if (not len(ids) or np.any(ids >= n) or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids) or seen[ids].any()):
            raise ValueError("V1_RAW_NODE_IDENTITY_INVALID")
        types = np.unique(native_type_ids[ids])
        if len(types) != 1:
            raise ValueError("V1_RAW_GROUP_CROSSES_NATIVE_CELL_TYPES")
        for name, shape in PARAMETER_SHAPES.items():
            _finite_float32(group["params"][name], shape, name)
        seen[ids] = True
        groups.append({"raw_group_index": len(groups), "native_node_type_id": int(types[0]),
                       "neurons": len(ids)})
    if (not seen.all() or len({row["native_node_type_id"] for row in groups}) != len(groups)
            or set(row["native_node_type_id"] for row in groups) != set(map(int, np.unique(native_type_ids)))):
        raise ValueError("V1_RAW_NODE_TYPE_COVERAGE_INVALID")

    count = sum(len(edge["source"]) for edge in raw["edges"])
    source = np.empty(count, dtype=np.int64)
    target = np.empty(count, dtype=np.int64)
    receptor = np.empty(count, dtype=np.int8)
    weight = np.empty(count, dtype=np.float32)
    delay = np.empty(count, dtype=np.float32)
    offset = 0
    for edge in raw["edges"]:
        src = _integers(edge["source"], "edge_source")
        dst = _integers(edge["target"], "edge_target")
        length = len(src)
        if (len(dst) != length or np.any(src < 0) or np.any(dst < 0)
                or np.any(src >= n) or np.any(dst >= n)):
            raise ValueError("V1_EDGE_ENDPOINT_INVALID")
        params = edge["params"]
        r = np.asarray(params["receptor_type"])
        if r.shape != () or r.dtype.kind not in "iu" or not 1 <= int(r) <= 4:
            raise ValueError("V1_RECEPTOR_TYPE_INVALID")
        w = _finite_float32(params["weight"], (length,), "edge_weight")
        d = np.asarray(params["delay"])
        if d.shape not in ((), (length,)):
            raise ValueError("V1_DELAY_SHAPE_INVALID")
        d = _finite_float32(d, d.shape, "edge_delay")
        if np.any(d < 0):
            raise ValueError("V1_NEGATIVE_EDGE_DELAY")
        end = offset + length
        source[offset:end], target[offset:end] = src, dst
        receptor[offset:end], weight[offset:end], delay[offset:end] = int(r), w, d
        offset = end

    # Boolean sparse data keeps duplicate nonzero synapses from cancelling or
    # wrapping to zero. Neither receptor rows nor self-loops add cortical cells.
    active = (weight != 0) & (source != target)
    adjacency = coo_matrix((np.ones(int(active.sum()), dtype=bool),
                            (source[active], target[active])), shape=(n, n)).tocsr()
    components = {}
    for connection in ("strong", "weak"):
        total, labels = connected_components(adjacency, directed=True, connection=connection)
        sizes = np.bincount(labels, minlength=total)
        components[connection] = {"components": int(total), "largest_neurons": int(sizes.max(initial=0))}
    outgoing = np.diff(adjacency.indptr) > 0
    incoming = np.bincount(adjacency.indices, minlength=n) > 0
    topology = {
        **components, "nonzero_interunit_synapses": int(active.sum()),
        "distinct_directed_interunit_pairs": int(adjacency.nnz),
        "no_interunit_input_neurons": int((~incoming).sum()),
        "no_interunit_output_neurons": int((~outgoing).sum()),
        "isolated_neurons": int((~incoming & ~outgoing).sum()),
        "minimum_required_neurons": MINIMUM_COUPLED_NEURONS,
        # A directed feed-forward connection still couples its endpoints. Strong
        # connectivity is reported separately, not imposed as an extra goal gate.
        "structural_coupled_scale_qualified": components["weak"]["largest_neurons"] >= MINIMUM_COUPLED_NEURONS,
    }
    del adjacency

    alignment = {"edge_delay_float32_min": float(delay.min()) if count else None,
                 "edge_delay_float32_max": float(delay.max()) if count else None,
                 "edge_association_mismatches": 0, "examples": [],
                 "comparison": "raw_delay[sorted_position] versus raw_delay[author_sort_order]"}
    if count:
        # Exact key and default argsort used by pinned load_sparse.sort_indices.
        packed_target = 4 * target + receptor - 1
        max_ind = max(int(packed_target.max()), int(source.max())) + 1
        if int(packed_target.max()) * max_ind + int(source.max()) > np.iinfo(np.int64).max:
            raise ValueError("V1_AUTHOR_SORT_KEY_OVERFLOW")
        order = np.argsort(packed_target * max_ind + source)
        mismatches = np.flatnonzero(delay != delay[order])
        alignment["edge_association_mismatches"] = len(mismatches)
        for position in mismatches[:8]:
            raw_index = order[position]
            alignment["examples"].append({
                "sorted_position": int(position), "raw_edge_index": int(raw_index),
                "source": int(source[raw_index]), "target": int(target[raw_index]),
                "receptor_type": int(receptor[raw_index]), "weight": float(weight[raw_index]),
                "loader_returned_delay": float(delay[position]),
                "edge_associated_delay": float(delay[raw_index]),
            })
    alignment["status"] = "MISMATCH" if alignment["edge_association_mismatches"] else "ALIGNED_FOR_THIS_PAYLOAD"
    return {"cortical_neurons": n, "raw_synapses": count, "node_groups": groups,
            "topology": topology, "loader_delay_alignment": alignment,
            "native_dynamics_qualified": False, "source_scientific_fidelity_qualified": False,
            "production_episodes": 0}


def _h5_metadata(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as handle:
        group = handle["nodes/v1"]
        data = {name: group[name][...] for name in ("node_id", "node_type_id", "node_group_id", "node_group_index")}
        n = len(data["node_id"])
        if (not np.array_equal(data["node_id"], np.arange(n))
                or not np.all(data["node_group_id"] == 0)
                or not np.array_equal(np.sort(data["node_group_index"]), np.arange(n))):
            raise ValueError("V1_AUTHOR_NODE_INDEX_ASSUMPTION_NOT_MET")
        for axis in ("x", "y", "z", "tuning_angle"):
            data[axis] = group["0"][axis][...][data["node_group_index"]]
    return data


def _types(path: Path) -> dict:
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter=" "))
    result = {int(row["node_type_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("V1_DUPLICATE_NODE_TYPE_ROW")
    return result


def compare_type_metadata(reference: dict, archived: dict) -> dict:
    if set(reference) != set(archived):
        raise ValueError("V1_ARCHIVE_NATIVE_NODE_TYPES_DIFFER")
    differences = []
    for identity, original in reference.items():
        candidate = archived[identity]
        if any(original[name] != candidate[name] for name in TYPE_IDENTITY_FIELDS):
            raise ValueError("V1_ARCHIVE_NATIVE_CELL_IDENTITY_DIFFERS:" + str(identity))
        changed = {name: {"reference": original.get(name), "archive": candidate.get(name)}
                   for name in set(original) | set(candidate)
                   if original.get(name) != candidate.get(name)}
        if changed:
            differences.append({"node_type_id": identity, "fields": changed})
    # Legacy template/parameter filenames and absent annotation columns are
    # visible differences. Identity agreement does not prove parameter equality.
    return {"identity_fields": list(TYPE_IDENTITY_FIELDS), "identity_fields_match": True,
            "all_declarations_equal": not differences, "differences": differences,
            "cell_model_parameter_equivalence_verified": False}


def inspect_archive(archive: Path, download_receipt: Path, reference_nodes: Path,
                    reference_types: Path, reference_report: Path,
                    loader: Path, loader_inspection: Path, output: Path) -> dict:
    started = time.monotonic()
    expected = json.loads(download_receipt.read_text())
    if archive.stat().st_size != expected["expected_bytes"]:
        raise ValueError("V1_COMPLETE_ARCHIVE_SIZE_REQUIRED")
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["expected_md5"]:
        raise ValueError("V1_COMPLETE_ARCHIVE_MD5_REQUIRED")
    reference = json.loads(reference_report.read_text())
    if reference["status"] != "NATIVE_NODE_IDENTITY_AND_CELL_TYPE_JOIN_PASS":
        raise ValueError("V1_NATIVE_METADATA_NOT_VERIFIED")
    if (sha256_file(reference_nodes) != reference["nodes_sha256"]
            or sha256_file(reference_types) != reference["node_types_sha256"]):
        raise ValueError("V1_REFERENCE_METADATA_CHANGED")
    inspection = json.loads(loader_inspection.read_text())["loader_source_inspection"]
    loader_row = next(row for row in inspection["files"] if Path(row["path"]).name == "load_sparse.py")
    if sha256_file(loader) != loader_row["sha256"]:
        raise ValueError("V1_INSPECTED_AUTHOR_LOADER_CHANGED")
    native = _h5_metadata(reference_nodes)
    member_hashes = {}
    with tempfile.TemporaryDirectory(prefix="v1-graph-") as temporary:
        root = Path(temporary)
        paths = {}
        with zipfile.ZipFile(archive) as source:
            for name, member in ARCHIVE_MEMBERS.items():
                if source.namelist().count(member) != 1:
                    raise ValueError("V1_REQUIRED_ARCHIVE_MEMBER_NOT_UNIQUE:" + member)
                target = root / Path(member).name
                with source.open(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                paths[name] = target
                member_hashes[member] = sha256_file(target)
        archived = _h5_metadata(paths["nodes"])
        if any(not np.array_equal(native[name], archived[name], equal_nan=True) for name in native):
            raise ValueError("V1_ARCHIVE_NATIVE_NODE_METADATA_DIFFERS")
        type_comparison = compare_type_metadata(_types(reference_types), _types(paths["types"]))
        if len(native["node_id"]) != reference["cortical_units"]:
            raise ValueError("V1_REFERENCE_CORTICAL_COUNT_DIFFERS")
        # The source's pickle format is read only after complete publisher
        # checksum verification and matching the independently acquired nodes.
        with paths["graph"].open("rb") as stream:
            raw = pickle.load(stream)
        report = validate_raw_graph(raw, native["node_type_id"])
    report.update(
        status="SOURCE_GRAPH_INSPECTED", source_archive_md5=digest.hexdigest(),
        source_archive_bytes=archive.stat().st_size, member_sha256=member_hashes,
        native_metadata_identity_verified=True, author_loader_sha256=loader_row["sha256"],
        cell_type_comparison=type_comparison,
        reference_metadata_sha256=sha256_file(reference_report),
        elapsed_seconds=time.monotonic() - started,
    )
    freeze_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "download-receipt", "reference-nodes", "reference-types",
                 "reference-report", "loader", "loader-inspection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect_archive(**vars(args))
    print(json.dumps({key: value for key, value in report.items() if key != "node_groups"}))
    # Declaration differences remain explicit evidence for native parameter
    # reconciliation. They do not invalidate otherwise matching graph identity.
    inspection_passed = (report["topology"]["structural_coupled_scale_qualified"]
                        and report["loader_delay_alignment"]["status"] == "ALIGNED_FOR_THIS_PAYLOAD")
    return 0 if inspection_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

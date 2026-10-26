"""Assemble verified development episode archives without changing identities.

The existing writer stores circuit metadata once and enforces episode uniqueness
and whole-circuit splits. This intake preserves raw events, known controls and
privileged records. Assembly creates no simulated episodes or production data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits
from .provenance import freeze_json, sha256_file


def read_circuit(dataset: CircuitDataset, circuit_id: str) -> CircuitUnits:
    row = dataset.catalog["circuits"][circuit_id]
    with h5py.File(dataset._verified(row["path"], row["sha256"]), "r") as source:
        circuit = CircuitUnits(
            circuit_id=circuit_id, unit_ids=source["unit_ids"][:],
            populations=source["populations"].asstr()[:].astype(str), field_index=source["field_index"][:],
            field_names=tuple(source["field_names"].asstr()[:]),
            coordinate_semantics=source.attrs["coordinate_semantics"],
            source_revision=source.attrs["source_revision"],
            circuit_provenance=json.loads(source.attrs["circuit_provenance_json"]),
            coordinates=source["coordinates"][:] if "coordinates" in source else None,
            coordinate_units=(source.attrs["coordinate_units"] if "coordinates" in source else None),
            node_type_ids=source["node_type_ids"][:] if "node_type_ids" in source else None,
            source_tuning_angle_deg=source["source_tuning_angle_deg"][:] if "source_tuning_angle_deg" in source else None)
    circuit.validate()
    if len(circuit.unit_ids) != row["units"] or circuit.circuit_provenance != row["provenance"]:
        raise ValueError("BENCHMARK_STATIC_CATALOG_DIFFERS")
    return circuit


def read_episode(dataset: CircuitDataset, row: dict) -> CircuitEpisode:
    shard = dataset.shards[row["shard"]]
    with h5py.File(dataset._verified(shard["path"], shard["sha256"]), "r") as source:
        group = source[row["episode_id"]]
        metadata = json.loads(group.attrs["metadata_json"])
        dataset.validate_runtime_metadata(metadata, row)
        if any(metadata.get(key) != row[key] for key in (
                "episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms")):
            raise ValueError("BENCHMARK_EPISODE_CATALOG_DIFFERS")
        arrays = {key: group[key][:] for key in (
            "spike_ticks", "spike_unit_indices", "control_ticks", "control_values")}
        private = {key: group["privileged"][key][:] for key in group.get("privileged", {})}
    if len(arrays["spike_ticks"]) != row["event_count"]:
        raise ValueError("BENCHMARK_EVENT_COUNT_DIFFERS")
    metadata["control_features"] = tuple(metadata["control_features"])
    return CircuitEpisode(**metadata, **arrays, privileged=private)


def bind_source_tuning(circuit: CircuitUnits, source_nodes: Path) -> CircuitUnits:
    """Attach source angles only after the existing geometry identity matches."""
    expected = circuit.circuit_provenance.get("unit_metadata_identity")
    if not expected or sha256_file(source_nodes) != expected:
        raise ValueError("BENCHMARK_SOURCE_TUNING_NODE_IDENTITY_DIFFERS")
    with h5py.File(source_nodes, "r") as source:
        group = source["nodes/v1"]
        ids, types, groups, indices = [
            group[name][:] for name in ("node_id", "node_type_id", "node_group_id", "node_group_index")]
        n = len(circuit.unit_ids)
        if (any(value.shape != (n,) or value.dtype.kind not in "iu" for value in (ids, types, groups, indices))
                or not np.all(groups == 0) or not np.array_equal(np.sort(indices), np.arange(n))):
            raise ValueError("BENCHMARK_SOURCE_TUNING_NODE_MAP_DIFFERS")
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.sort(circuit.unit_ids)) or circuit.node_type_ids is None:
            raise ValueError("BENCHMARK_SOURCE_TUNING_NODE_MAP_DIFFERS")
        # Archive unit order is authoritative, including supported non-sorted IDs.
        positions_for_ids = np.searchsorted(ids[order], circuit.unit_ids)
        source_order = order[positions_for_ids]
        if not np.array_equal(types[source_order], circuit.node_type_ids):
            raise ValueError("BENCHMARK_SOURCE_TUNING_NODE_MAP_DIFFERS")
        positions = np.column_stack([group["0/" + axis][:][indices[source_order]] for axis in ("x", "y", "z")])
        if circuit.coordinates is None or not np.array_equal(positions, circuit.coordinates):
            raise ValueError("BENCHMARK_SOURCE_TUNING_COORDINATES_DIFFER")
        angles = group["0/tuning_angle"][:][indices[source_order]]
    if circuit.source_tuning_angle_deg is not None and not np.array_equal(circuit.source_tuning_angle_deg, angles):
        raise ValueError("BENCHMARK_SOURCE_TUNING_EXISTING_ANGLES_DIFFER")
    enriched = replace(circuit, source_tuning_angle_deg=angles)
    enriched.validate()
    return enriched


def assemble(sources: list[Path], output: Path, *, scenario_id: str,
             source_node_files: list[Path] | None = None) -> dict:
    if not sources or output.exists() or scenario_id not in ("v1_visual", "ca3_pattern_completion"):
        raise ValueError("BENCHMARK_FRESH_V1_OR_CA3_ASSEMBLY_REQUIRED")
    if len({source.resolve() for source in sources}) != len(sources):
        raise ValueError("BENCHMARK_DUPLICATE_SOURCE_ARCHIVE")
    if source_node_files and scenario_id != "v1_visual":
        raise ValueError("BENCHMARK_SOURCE_TUNING_V1_ONLY")
    node_files = {}
    for path in source_node_files or ():
        digest = sha256_file(path)
        if digest in node_files:
            raise ValueError("BENCHMARK_DUPLICATE_SOURCE_NODE_IDENTITY")
        node_files[digest] = Path(path)
    enriched_circuits, used_nodes = {}, set()
    datasets, bound, ids = [], [], set()
    revision = None
    for root in sources:
        dataset = CircuitDataset(root, splits=("development",))
        catalog = dataset.catalog
        binding = catalog["binding"]
        rows = [row for shard in catalog["shards"] for row in shard["episodes"]]
        if (binding.get("schema") != SCHEMA or binding.get("scenario_id") != scenario_id
                or binding.get("scope") != "source_reproduction"
                or catalog.get("source_fidelity_qualified") is not False
                or catalog.get("complete_episodes") != len(rows) or not rows
                or any(row["split"] != "development" for row in rows)
                or len(dataset.episodes) != len(rows)):
            raise ValueError("BENCHMARK_COMPLETE_DEVELOPMENT_ARCHIVE_REQUIRED")
        if revision is not None and binding["source_revision"] != revision:
            raise ValueError("BENCHMARK_SOURCE_REVISION_DIFFERS")
        revision = binding["source_revision"]
        for row in rows:
            if row["episode_id"] in ids:
                raise ValueError("BENCHMARK_DUPLICATE_COMPLETE_EPISODE")
            ids.add(row["episode_id"])
        bound.append(dict(dataset=str(root.resolve()), catalog_sha256=sha256_file(root / "dataset.json"),
                          source_binding=binding))
        datasets.append(dataset)
        if node_files:
            for circuit_id in catalog["circuits"]:
                original = read_circuit(dataset, circuit_id)
                digest = original.circuit_provenance.get("unit_metadata_identity")
                if digest not in node_files:
                    raise ValueError("BENCHMARK_BOUND_SOURCE_NODES_MISSING:" + circuit_id)
                enriched = bind_source_tuning(original, node_files[digest])
                used_nodes.add(digest)
                if circuit_id in enriched_circuits:
                    previous = enriched_circuits[circuit_id]
                    for name in ("unit_ids", "node_type_ids", "coordinates", "source_tuning_angle_deg"):
                        if not np.array_equal(getattr(previous, name), getattr(enriched, name)):
                            raise ValueError("BENCHMARK_SOURCE_CIRCUIT_REBOUND")
                enriched_circuits[circuit_id] = enriched
    if node_files and set(node_files) != used_nodes:
        raise ValueError("BENCHMARK_UNUSED_SOURCE_NODE_FILES")
    output.mkdir(parents=True)
    report = dict(status="ASSEMBLING_BENCHMARK_DEVELOPMENT_DATASET", scenario_id=scenario_id,
                  complete_episodes=0, production_episodes=0, new_source_simulations=0,
                  scientific_qualification=False, sources=bound,
                  source_tuning_metadata=[dict(path=str(node_files[digest].resolve()), sha256=digest)
                                          for digest in sorted(used_nodes)])
    try:
        binding = dict(schema=SCHEMA, scenario_id=scenario_id, scope="source_reproduction",
                       sample_unit="complete_mechanistic_episode", source_revision=revision,
                       source_archives=bound)
        if node_files:
            binding["source_tuning_metadata"] = report["source_tuning_metadata"]
        with CircuitArchiveWriter(output / "dataset", binding) as writer:
            for dataset in datasets:
                for circuit_id in dataset.catalog["circuits"]:
                    original = read_circuit(dataset, circuit_id)
                    # Preserve every source's static descriptors. The writer
                    # must reject conflicting populations or field maps for
                    # the same circuit, including during angle enrichment.
                    selected = replace(original,
                        source_tuning_angle_deg=enriched_circuits[circuit_id].source_tuning_angle_deg
                    ) if node_files else original
                    writer.register_circuit(selected)
                for row in dataset.iter_episodes():
                    writer.add(read_episode(dataset, row))
        assembled = CircuitDataset(output / "dataset", splits=("development",))
        if set(assembled.episodes) != ids or assembled.catalog["complete_episodes"] != len(ids):
            raise ValueError("BENCHMARK_ASSEMBLED_EPISODE_IDENTITIES_DIFFER")
        for circuit_id, expected in enriched_circuits.items():
            actual = read_circuit(assembled, circuit_id)
            if not np.array_equal(actual.source_tuning_angle_deg, expected.source_tuning_angle_deg):
                raise ValueError("BENCHMARK_ASSEMBLED_SOURCE_TUNING_DIFFERS")
        for dataset in datasets:
            for row in dataset.iter_episodes():
                original = read_episode(dataset, row)
                actual = read_episode(assembled, assembled.episodes[row["episode_id"]])
                for name in ("spike_ticks", "spike_unit_indices", "control_ticks", "control_values"):
                    if not np.array_equal(getattr(original, name), getattr(actual, name)):
                        raise ValueError("BENCHMARK_ASSEMBLED_VALUES_DIFFER:" + name)
                if set(original.privileged) != set(actual.privileged) or any(
                        not np.array_equal(value, actual.privileged[name])
                        for name, value in original.privileged.items()):
                    raise ValueError("BENCHMARK_ASSEMBLED_PRIVILEGED_RECORD_DIFFERS")
                if any(getattr(original, name) != getattr(actual, name) for name in (
                        "episode_id", "circuit_id", "block_id", "split", "duration_ticks", "dt_ms",
                        "control_kind", "source_record", "seeds", "control_features", "task_outcome")):
                    raise ValueError("BENCHMARK_ASSEMBLED_METADATA_DIFFERS")
        report.update(status="COMPLETE_BENCHMARK_DEVELOPMENT_DATASET", complete_episodes=len(ids),
                      circuit_realizations=len(assembled.catalog["circuits"]),
                      source_episodes_repackaged=len(ids), all_episode_values_exact=True,
                      circuits={key: row["units"] for key, row in assembled.catalog["circuits"].items()},
                      dataset_catalog_sha256=sha256_file(output / "dataset/dataset.json"),
                      source_tuning_circuits=len(enriched_circuits),
                      source_tuning_values_verified=sum(len(c.unit_ids) for c in enriched_circuits.values()))
    except BaseException as error:
        report.update(status="FAILED_BENCHMARK_DEVELOPMENT_DATASET", error_type=type(error).__name__,
                      error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("v1_visual", "ca3_pattern_completion"), required=True)
    parser.add_argument("--source-node-file", type=Path, action="append",
                        help="Original V1 node file whose hash is already bound in each circuit's provenance.")
    args = parser.parse_args()
    print(json.dumps(assemble(args.source, args.output, scenario_id=args.scenario,
                              source_node_files=args.source_node_file), indent=2))


if __name__ == "__main__":
    main()

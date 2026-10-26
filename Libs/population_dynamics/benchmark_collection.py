"""Index accepted production blocks through the existing CircuitDataset reader.

The output root must contain the source datasets and their public readback
reports. Only dataset.json is written. Payloads and source catalogs stay in place.
A partial collection reports its actual size and the frozen target separately.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .benchmark_dataset import read_circuit
from .benchmark_production import ProductionCohort, require
from .circuit_data import CircuitDataset, SCHEMA, _atomic_json
from .provenance import sha256_file

SPLITS = ("train", "validation", "calibration", "test")


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(root), "COLLECTION_SOURCE_OUTSIDE_OUTPUT_ROOT")
    return str(resolved.relative_to(root))


def _same_units(left, right) -> bool:
    for item in fields(left):
        a, b = getattr(left, item.name), getattr(right, item.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return False
        elif a != b:
            return False
    return True


def assemble(cohort: ProductionCohort, reports: list[Path], output_root: Path,
             *, resume: bool = False, allow_test_packaging: bool = False) -> dict:
    """Publish a catalog from complete blocks that passed public resume readback."""
    cohort.require_current()
    require(cohort.scenario in ("v1_visual", "ca3_pattern_completion"),
            "COLLECTION_V1_OR_CA3_REQUIRED")
    root = output_root.resolve()
    require(root.is_dir() and bool(reports), "COLLECTION_EXISTING_ROOT_AND_READBACKS_REQUIRED")
    paths = sorted(Path(path).resolve() for path in reports)
    require(len(set(paths)) == len(paths), "COLLECTION_DUPLICATE_READBACK")
    target = root / "dataset.json"
    old = None
    if target.exists():
        require(resume, "COLLECTION_EXPLICIT_RESUME_REQUIRED")
        old = json.loads(target.read_text())
        require(old.get("collection", {}).get("kind") == "accepted_production_blocks",
                "COLLECTION_OUTPUT_IS_ANOTHER_DATASET")
    require(not target.with_suffix(".writing").exists(), "COLLECTION_UNRESOLVED_CATALOG_WRITE")
    binding = dict(schema=SCHEMA, scenario_id=cohort.scenario, scope="production",
                   sample_unit="complete_mechanistic_episode",
                   source_revision=cohort.qualification["source_revision"],
                   source_qualification_sha256=cohort.qualification_sha256,
                   identity_table_sha256=cohort.identity_table_sha256)
    catalog = dict(binding=binding, circuits={}, shards=[], complete_episodes=0,
                   source_fidelity_qualified=True)
    seen, block_keys, source_roots = set(), set(), set()
    static_sources, accepted = {}, []
    protocol = cohort.campaign["scenarios"][cohort.scenario]["task_protocol"]
    expected_units = cohort.qualification["coupled_core_neurons"]
    for path in paths:
        report_relative = _relative(path, root)
        report = json.loads(path.read_text())
        require(report.get("status") == "PASS_COMPLETE_PRODUCTION_BLOCK_PUBLIC_REUSE"
                and report.get("complete_episodes") == report.get("reused_episodes") == 10
                and report.get("dataset_catalog_unchanged") is True
                and report.get("source_qualification_sha256") == cohort.qualification_sha256
                and report.get("identity_table_sha256") == cohort.identity_table_sha256,
                "COLLECTION_ACCEPTED_PUBLIC_READBACK_REQUIRED")
        source_root = Path(report["dataset"]).resolve()
        relative = _relative(source_root / "dataset.json", root)
        require(source_root != root, "COLLECTION_DUPLICATE_SOURCE_DATASET")
        require(sha256_file(source_root / "dataset.json") == report["catalog_sha256"],
                "COLLECTION_SOURCE_CATALOG_CHANGED")
        dataset = CircuitDataset(source_root,
            splits=SPLITS if allow_test_packaging else SPLITS[:-1], allow_test=allow_test_packaging)
        source = dataset.catalog
        if source.get("runtime_qualifications"):
            registry = catalog.setdefault("runtime_qualifications", {})
            for sha, document in source["runtime_qualifications"].items():
                require(sha not in registry or registry[sha] == document,
                        "COLLECTION_RUNTIME_REGISTRY_CHANGED")
                registry[sha] = document
        require(source["binding"] == binding and source.get("source_fidelity_qualified") is True,
                "COLLECTION_PRODUCTION_BINDING_DIFFERS")
        rows = [row for shard in source["shards"] for row in shard["episodes"]]
        require(len(rows) == 10, "COLLECTION_COMPLETE_FROZEN_BLOCK_REQUIRED")
        # Earlier accepted readbacks omit redundant block indices. Resolve them
        # from the authenticated catalog and frozen cohort, then check any
        # indices that the report does declare.
        first = cohort.identities.get(rows[0]["episode_id"])
        require(first is not None, "COLLECTION_COMPLETE_FROZEN_BLOCK_REQUIRED")
        key = (first["circuit_index"], first["block_index"])
        for name, expected in zip(("circuit_index", "block_index"), key):
            require(name not in report or (type(report[name]) is int and report[name] == expected),
                    "COLLECTION_READBACK_IDENTITY_DIFFERS")
        identities = [cohort.identity(*key, k) for k in range(10)]
        require(key not in block_keys, "COLLECTION_DUPLICATE_COMPLETE_BLOCK")
        block_keys.add(key)
        require(source_root not in source_roots, "COLLECTION_DUPLICATE_SOURCE_DATASET")
        source_roots.add(source_root)
        split = identities[0]["split"]
        require(split != "test" or allow_test_packaging,
                "COLLECTION_TEST_PACKAGING_REQUIRES_EXPLICIT_REQUEST")
        require(source["complete_episodes"] == len(rows) == len(dataset.episodes) == 10
                and {row["episode_id"] for row in rows} == {item["episode_id"] for item in identities},
                "COLLECTION_COMPLETE_FROZEN_BLOCK_REQUIRED")
        by_id = {item["episode_id"]: item for item in identities}
        require(set(source["circuits"]) == {identities[0]["circuit_id"]},
                "COLLECTION_BLOCK_CIRCUIT_DIFFERS")
        for row in rows:
            identity = by_id[row["episode_id"]]
            require(all(row[name] == identity[name] for name in ("circuit_id", "block_id", "split"))
                    and row["duration_ticks"] == protocol["condition_duration_ticks"][identity["condition_index"]]
                    and row["dt_ms"] == protocol["dt_ms"],
                    "COLLECTION_IDENTITY_OR_CLOCK_DIFFERS")
            require(row["episode_id"] not in seen, "COLLECTION_DUPLICATE_COMPLETE_EPISODE")
            seen.add(row["episode_id"])
        require(sum(row["event_count"] for row in rows) == report["total_spike_pairs"]
                and report["units"] == expected_units and report["dt_ms"] == protocol["dt_ms"]
                and all(row["duration_ticks"] * row["dt_ms"] == report["duration_ms"] for row in rows),
                "COLLECTION_READBACK_MEASUREMENTS_DIFFER")
        for circuit_id, row in source["circuits"].items():
            require(row["units"] == expected_units, "COLLECTION_CORE_SCALE_DIFFERS")
            _relative(source_root / row["path"], source_root)
            rebased = dict(row, path=_relative(source_root / row["path"], root))
            previous = catalog["circuits"].get(circuit_id)
            if previous is None:
                # Validate the canonical observation map once per circuit.
                units = read_circuit(dataset, circuit_id)
                require(units.source_revision == binding["source_revision"],
                        "COLLECTION_CIRCUIT_SOURCE_REVISION_DIFFERS")
                catalog["circuits"][circuit_id] = rebased
                static_sources[circuit_id] = dataset
            elif {k: v for k, v in previous.items() if k != "path"} != {k: v for k, v in row.items() if k != "path"}:
                require(_same_units(read_circuit(static_sources[circuit_id], circuit_id),
                                    read_circuit(dataset, circuit_id)),
                        "COLLECTION_CIRCUIT_OBSERVATION_MAP_REBOUND")
        for shard in source["shards"]:
            _relative(source_root / shard["path"], source_root)
            catalog["shards"].append(dict(shard, path=_relative(source_root / shard["path"], root)))
        accepted.append(dict(circuit_index=key[0], block_index=key[1],
                             readback_report=report_relative, readback_sha256=sha256_file(path),
                             dataset_catalog=relative, catalog_sha256=report["catalog_sha256"]))
    accepted.sort(key=lambda row: (row["circuit_index"], row["block_index"]))
    catalog["shards"].sort(key=lambda shard: shard["episodes"][0]["episode_id"])
    require(len({row["path"] for row in catalog["shards"]}) == len(catalog["shards"]),
            "COLLECTION_DUPLICATE_SHARD_PATH")
    catalog["complete_episodes"] = len(seen)
    catalog["collection"] = dict(kind="accepted_production_blocks", sources=accepted,
        expected_complete_episodes=len(cohort.identities), complete=(seen == set(cohort.identities)))
    if old is not None:
        require(old["binding"] == binding, "COLLECTION_RESUME_BINDING_CHANGED")
        current = {(row["circuit_index"], row["block_index"]): row for row in accepted}
        require(all(current.get((row["circuit_index"], row["block_index"])) == row
                    for row in old["collection"]["sources"]), "COLLECTION_RESUME_REMOVED_OR_CHANGED_BLOCK")
        # Keep the previous canonical static path when adding an earlier block.
        for circuit_id, row in old["circuits"].items():
            chosen = catalog["circuits"][circuit_id]
            if {k: v for k, v in row.items() if k != "path"} != {k: v for k, v in chosen.items() if k != "path"}:
                prior = CircuitDataset(root, splits=SPLITS, allow_test=True)
                require(_same_units(read_circuit(prior, circuit_id),
                                    read_circuit(static_sources[circuit_id], circuit_id)),
                        "COLLECTION_RESUME_CIRCUIT_CHANGED")
            catalog["circuits"][circuit_id] = row
    cohort.require_current()
    if old != catalog:
        _atomic_json(target, catalog)
    # Normal readers retain their existing split and payload checksum boundaries.
    actual = CircuitDataset(root, splits=SPLITS if allow_test_packaging else SPLITS[:-1],
                            allow_test=allow_test_packaging)
    require(set(actual.episodes) == seen, "COLLECTION_PUBLIC_INDEX_DIFFERS")
    return dict(status="COMPLETE_PRODUCTION_COLLECTION" if catalog["collection"]["complete"]
                else "PARTIAL_PRODUCTION_COLLECTION",
                complete_episodes=len(seen), expected_complete_episodes=len(cohort.identities),
                complete_blocks=len(accepted), circuits=len(catalog["circuits"]),
                dataset=str(root), catalog_sha256=sha256_file(target),
                new_source_simulations=0, copied_payload_bytes=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", choices=("v1_visual", "ca3_pattern_completion"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--identity-table", type=Path, required=True)
    parser.add_argument("--readback-report", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-test-packaging", action="store_true")
    args = parser.parse_args(argv)
    cohort = ProductionCohort.load(args.config, args.scenario, args.qualification, args.identity_table)
    print(json.dumps(assemble(cohort, args.readback_report, args.output_root,
                              resume=args.resume, allow_test_packaging=args.allow_test_packaging)))


if __name__ == "__main__":
    main()

"""Collection behavior on disposable production-schema fixtures, not simulations."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from . import benchmark_collection as collection
from .benchmark_dataset import read_episode
from .benchmark_production import COHORT_SEEDS
from .circuit_data import CircuitDataset, CircuitEpisode, CircuitUnits
from .provenance import sha256_file
from .test_benchmark_production import cohort


def make_block(root, cohort, circuit, block):
    source = root / f"circuit_{circuit:03d}" / f"block_{block:03d}"
    dataset_root = source / "dataset"
    identity = cohort.identity(circuit, block, 0)
    n = cohort.qualification["coupled_core_neurons"]
    protocol = cohort.campaign["scenarios"][cohort.scenario]["task_protocol"]
    units = CircuitUnits(identity["circuit_id"], np.arange(n), np.full(n, "E"),
        np.zeros(n, dtype=np.int64), ("E",), "synthetic source map",
        cohort.qualification["source_revision"],
        circuit_provenance=dict(connectivity_identity=f"fixture-graph-{circuit}",
                                parameter_identity="fixture-parameters"))
    with cohort.writer(dataset_root, units.source_revision) as writer:
        writer.register_circuit(units)
        for k in range(10):
            item = cohort.identity(circuit, block, k)
            duration = protocol["condition_duration_ticks"][k]
            episode = CircuitEpisode(item["episode_id"], item["circuit_id"], item["block_id"],
                item["split"], duration, protocol["dt_ms"], np.array([0, duration]),
                np.array([0, n - 1]), np.array([0]), np.array([[1.]]),
                (dict(name="synthetic_command", unit="dimensionless"),),
                seeds={key: item[key] for key in COHORT_SEEDS},
                source_record=dict(complete_unit_recording=True, simulation_completed=True,
                    recorded_until_tick=duration, expected_duration_ticks=duration, core_unit_count=n,
                    source_revision=units.source_revision, circuit_provenance=units.circuit_provenance),
                privileged=dict(fixture_state=np.array([3.])), task_outcome="failed_synthetic_target")
            writer.add(episode)
    report = dict(status="PASS_COMPLETE_PRODUCTION_BLOCK_PUBLIC_REUSE",
        circuit_index=circuit, block_index=block, complete_episodes=10, reused_episodes=10,
        dataset_catalog_unchanged=True, source_qualification_sha256=cohort.qualification_sha256,
        identity_table_sha256=cohort.identity_table_sha256, dataset=str(dataset_root),
        catalog_sha256=sha256_file(dataset_root / "dataset.json"), total_spike_pairs=20, units=n,
        dt_ms=protocol["dt_ms"], duration_ms=duration * protocol["dt_ms"])
    path = source / "readback.json"
    path.write_text(json.dumps(report))
    return path


def test_union_resume_preserves_payloads_static_path_and_public_values(cohort, tmp_path):
    later = make_block(tmp_path, cohort, 0, 1)
    source = Path(json.loads(later.read_text())["dataset"])
    before = (source / "dataset.json").read_bytes()
    result = collection.assemble(cohort, [later], tmp_path)
    assert result["status"] == "PARTIAL_PRODUCTION_COLLECTION"
    assert result["complete_episodes"] == 10 and result["expected_complete_episodes"] == 100000
    old = (tmp_path / "dataset.json").read_bytes()
    assert collection.assemble(cohort, [later], tmp_path, resume=True) == result
    assert (tmp_path / "dataset.json").read_bytes() == old
    earlier = make_block(tmp_path, cohort, 0, 0)
    result = collection.assemble(cohort, [earlier, later], tmp_path, resume=True)
    assert result["complete_episodes"] == 20 and result["circuits"] == 1
    assert result["copied_payload_bytes"] == result["new_source_simulations"] == 0
    dataset = CircuitDataset(tmp_path, splits=("train",))
    old_catalog = json.loads(old)
    assert dataset.catalog["circuits"] == old_catalog["circuits"]
    identity = cohort.identity(0, 1, 0)
    actual = read_episode(dataset, dataset.episodes[identity["episode_id"]])
    native = CircuitDataset(source, splits=("train",))
    expected = read_episode(native, native.episodes[identity["episode_id"]])
    np.testing.assert_array_equal(actual.spike_ticks, expected.spike_ticks)
    np.testing.assert_array_equal(actual.control_values, expected.control_values)
    assert actual.seeds == expected.seeds and actual.task_outcome == "failed_synthetic_target"
    np.testing.assert_array_equal(actual.privileged["fixture_state"], expected.privileged["fixture_state"])
    assert (source / "dataset.json").read_bytes() == before
    retained = (tmp_path / "dataset.json").read_bytes()
    with pytest.raises(ValueError, match="REMOVED_OR_CHANGED_BLOCK"):
        collection.assemble(cohort, [earlier], tmp_path, resume=True)
    assert (tmp_path / "dataset.json").read_bytes() == retained


def test_test_split_requires_packaging_flag_and_remains_hidden_from_default_reader(cohort, tmp_path):
    path = make_block(tmp_path, cohort, 90, 0)
    with pytest.raises(ValueError, match="TEST_PACKAGING_REQUIRES_EXPLICIT"):
        collection.assemble(cohort, [path], tmp_path)
    assert not (tmp_path / "dataset.json").exists()
    result = collection.assemble(cohort, [path], tmp_path, allow_test_packaging=True)
    assert result["complete_episodes"] == 10
    assert not CircuitDataset(tmp_path).episodes
    with pytest.raises(ValueError, match="FORMAL_TEST_ACCESS"):
        CircuitDataset(tmp_path, splits=("test",))
    assert len(CircuitDataset(tmp_path, splits=("test",), allow_test=True).episodes) == 10


def test_rejects_catalog_changes_incomplete_blocks_wrong_clocks_and_path_escape(cohort, tmp_path):
    path = make_block(tmp_path, cohort, 0, 0)
    report = json.loads(path.read_text())
    target = Path(report["dataset"]) / "dataset.json"
    original = json.loads(target.read_text())
    cases = ("stale_hash", "development", "missing_condition", "wrong_clock", "wrong_split", "path_escape")
    patterns = ("SOURCE_CATALOG_CHANGED", "PRODUCTION_BINDING", "COMPLETE_FROZEN_BLOCK",
                "IDENTITY_OR_CLOCK", "WHOLE_CIRCUIT_SPLIT_LEAKAGE", "SOURCE_OUTSIDE")
    for case, pattern in zip(cases, patterns):
        changed = deepcopy(original)
        if case in ("stale_hash", "development"):
            changed["binding"]["scope"] = "source_reproduction"
        elif case == "missing_condition":
            changed["shards"][0]["episodes"].pop()
            changed["complete_episodes"] -= 1
        elif case == "wrong_clock":
            changed["shards"][0]["episodes"][0]["duration_ticks"] -= 1
        elif case == "wrong_split":
            changed["shards"][0]["episodes"][0]["split"] = "validation"
        else:
            changed["shards"][0]["path"] = "../outside.h5"
        target.write_text(json.dumps(changed))
        path.write_text(json.dumps(dict(report, catalog_sha256=report["catalog_sha256"]
                                       if case == "stale_hash" else sha256_file(target))))
        with pytest.raises(ValueError, match=pattern):
            collection.assemble(cohort, [path], tmp_path)
        assert not (tmp_path / "dataset.json").exists()


def test_rejects_duplicates_and_conflicting_static_observation_map(cohort, tmp_path):
    first = make_block(tmp_path, cohort, 0, 0)
    later = make_block(tmp_path, cohort, 0, 1)
    with pytest.raises(ValueError, match="DUPLICATE_READBACK"):
        collection.assemble(cohort, [first, first], tmp_path)
    duplicate = first.with_name("duplicate.json")
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="DUPLICATE_COMPLETE_BLOCK"):
        collection.assemble(cohort, [first, duplicate], tmp_path)
    report = json.loads(later.read_text())
    source = Path(report["dataset"])
    cat = json.loads((source / "dataset.json").read_text())
    row = next(iter(cat["circuits"].values()))
    with h5py.File(source / row["path"], "r+") as handle:
        handle.attrs["coordinate_semantics"] = "different source map"
    row["sha256"] = sha256_file(source / row["path"])
    (source / "dataset.json").write_text(json.dumps(cat))
    report["catalog_sha256"] = sha256_file(source / "dataset.json")
    later.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="OBSERVATION_MAP_REBOUND"):
        collection.assemble(cohort, [first, later], tmp_path)
    assert not (tmp_path / "dataset.json").exists()


def test_cli_publishes_partial_catalog_and_rejects_implicit_overwrite(cohort, tmp_path):
    path = make_block(tmp_path, cohort, 0, 0)
    command = [sys.executable, "-B", "-m", "population_dynamics.benchmark_collection",
        "--config", str(cohort.qualification_path.parent / "campaign.yaml"),
        "--scenario", cohort.scenario, "--qualification", str(cohort.qualification_path),
        "--identity-table", str(cohort.identity_table), "--readback-report", str(path),
        "--output-root", str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["complete_episodes"] == 10
    before = (tmp_path / "dataset.json").read_bytes()
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "EXPLICIT_RESUME_REQUIRED" in result.stderr
    assert (tmp_path / "dataset.json").read_bytes() == before


def test_legacy_readback_uses_frozen_catalog_identity_and_rejects_disagreement(cohort, tmp_path):
    path = make_block(tmp_path, cohort, 70, 7)
    report = json.loads(path.read_text())
    report.pop("circuit_index")
    report.pop("block_index")
    path.write_text(json.dumps(dict(report, circuit_index=0)))
    with pytest.raises(ValueError, match="READBACK_IDENTITY_DIFFERS"):
        collection.assemble(cohort, [path], tmp_path)
    assert not (tmp_path / "dataset.json").exists()
    path.write_text(json.dumps(report))
    original = path.read_bytes()
    result = collection.assemble(cohort, [path], tmp_path)
    assert result["complete_episodes"] == 10
    dataset = CircuitDataset(tmp_path, splits=("validation",))
    expected = {cohort.identity(70, 7, k)["episode_id"] for k in range(10)}
    assert set(dataset.episodes) == expected
    source = dataset.catalog["collection"]["sources"][0]
    assert (source["circuit_index"], source["block_index"]) == (70, 7)
    assert path.read_bytes() == original

"""Lossless assembly, retained failure outcomes and rejected source identities."""
from dataclasses import replace
import json
import h5py

import numpy as np
import pytest

from .benchmark_dataset import assemble, read_episode
from .circuit_data import CircuitArchiveWriter, CircuitDataset
from .test_circuit_data import binding, episode, units


def archive(root, name, *, circuit=None, value=None, scenario="v1_visual"):
    selected = circuit or units()
    value = replace(value or episode(name, "development"), episode_id=name, circuit_id=selected.circuit_id,
                    source_record=dict(episode().source_record, circuit_provenance=selected.circuit_provenance))
    with CircuitArchiveWriter(root, dict(binding(), scenario_id=scenario)) as writer:
        writer.register_circuit(selected)
        writer.add(value)
    return root


def test_assembly_keeps_one_static_map_raw_endpoint_controls_and_privileged_records(tmp_path):
    roots = [archive(tmp_path / name, name) for name in ("first", "second")]
    report = assemble(roots, tmp_path / "joined", scenario_id="v1_visual")
    assert report["complete_episodes"] == 2 and report["circuit_realizations"] == 1
    assert report["production_episodes"] == report["new_source_simulations"] == 0
    joined = CircuitDataset(tmp_path / "joined/dataset", splits=("development",))
    assert len(joined.catalog["circuits"]) == 1
    for row in joined.iter_episodes():
        actual = read_episode(joined, row)
        np.testing.assert_array_equal(actual.spike_ticks, episode().spike_ticks)
        assert actual.spike_ticks[-1] == actual.duration_ticks
        assert actual.task_outcome == "failed_memory"
        assert actual.privileged["hidden_future_state"].tolist() == [100.]
        window = joined.window(row["episode_id"], boundary_tick=4, history_ticks=4, horizon_ticks=6, bin_ticks=2)
        assert "privileged" not in window and window["unit_ids"].tolist() == [19, 3, 72]


def test_assembly_keeps_ordered_simultaneous_ca3_cue_assignments(tmp_path):
    value = replace(episode("cue", "development"), control_kind="impulse",
                    control_ticks=np.array([3, 3, 7]),
                    control_values=np.array([[0., 10000.], [2., 10000.], [0., 0.]]),
                    control_features=({"name": "unit_index", "unit": "index"},
                                      {"name": "current", "unit": "pA"}),
                    seeds={"background_seed": 11, "stimulus_seed": 12})
    source = archive(tmp_path / "source", "cue", value=value, scenario="ca3_pattern_completion")
    assemble([source], tmp_path / "joined", scenario_id="ca3_pattern_completion")
    joined = CircuitDataset(tmp_path / "joined/dataset", splits=("development",))
    actual = read_episode(joined, joined.episodes["cue"])
    assert actual.control_kind == "impulse" and actual.seeds == value.seeds
    assert actual.control_features == value.control_features
    np.testing.assert_array_equal(actual.control_ticks, value.control_ticks)
    np.testing.assert_array_equal(actual.control_values, value.control_values)


def test_assembly_rejects_duplicate_episodes_and_changed_circuit_metadata(tmp_path):
    first = archive(tmp_path / "first", "same")
    second = archive(tmp_path / "second", "same")
    with pytest.raises(ValueError, match="DUPLICATE_COMPLETE_EPISODE"):
        assemble([first, second], tmp_path / "duplicate", scenario_id="v1_visual")
    assert not (tmp_path / "duplicate").exists()
    with pytest.raises(ValueError, match="DUPLICATE_SOURCE"):
        assemble([first, first], tmp_path / "alias", scenario_id="v1_visual")
    changed = archive(tmp_path / "changed", "different", circuit=replace(units(), populations=np.array(["I", "E", "I"])))
    with pytest.raises(ValueError, match="REBOUND"):
        assemble([first, changed], tmp_path / "conflict", scenario_id="v1_visual")
    report = json.loads((tmp_path / "conflict/report.json").read_text())
    assert report["status"] == "FAILED_BENCHMARK_DEVELOPMENT_DATASET" and report["complete_episodes"] == 0


def test_assembly_rejects_corruption_and_production_relabeling(tmp_path):
    source = archive(tmp_path / "source", "episode")
    catalog_path = source / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    shard = source / catalog["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="ARCHIVE_CONTENT_CHANGED"):
        assemble([source], tmp_path / "corrupt", scenario_id="v1_visual")
    catalog["binding"]["scope"] = "production"
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="COMPLETE_DEVELOPMENT_ARCHIVE"):
        assemble([source], tmp_path / "production", scenario_id="v1_visual")


def test_assembly_preserves_source_tuning_without_inventing_legacy_angles(tmp_path):
    from .benchmark_dataset import read_circuit
    static = replace(units(), source_tuning_angle_deg=np.array([45., 90., 135.]))
    source = archive(tmp_path / "source", "episode", circuit=static)
    assemble([source], tmp_path / "joined", scenario_id="v1_visual")
    dataset = CircuitDataset(tmp_path / "joined/dataset", splits=("development",))
    np.testing.assert_array_equal(read_circuit(dataset, static.circuit_id).source_tuning_angle_deg, static.source_tuning_angle_deg)


def bound_node_fixture(tmp_path, *, tuning=(45., 90., 135.)):
    from .provenance import sha256_file
    static = units()
    path = tmp_path / "original_nodes.h5"
    # Source IDs and node-group rows use distinct permutations.
    with h5py.File(path, "w") as source:
        group = source.create_group("nodes/v1")
        group["node_id"] = [72, 19, 3]
        group["node_type_id"] = [202, 101, 101]
        group["node_group_id"] = [0, 0, 0]
        group["node_group_index"] = [0, 2, 1]
        original_positions = np.array([[0., 0., 0.], [180., 0., 0.], [0., 0., 0.]])
        source_positions = original_positions[[2, 1, 0]]
        group.create_group("0")
        for index, axis in enumerate(("x", "y", "z")):
            group["0/" + axis] = source_positions[:, index]
        group["0/tuning_angle"] = np.asarray(tuning)[[2, 1, 0]]
    static = replace(static, node_type_ids=np.array([101, 101, 202]),
        coordinates=original_positions, coordinate_units="um",
        circuit_provenance=dict(static.circuit_provenance, unit_metadata_identity=sha256_file(path)))
    return static, path


def test_bound_source_tuning_migration_keeps_episode_identity_and_values(tmp_path):
    from .benchmark_dataset import read_circuit
    static, path = bound_node_fixture(tmp_path)
    source = archive(tmp_path / "source", "episode", circuit=static)
    report = assemble([source], tmp_path / "joined", scenario_id="v1_visual", source_node_files=[path])
    dataset = CircuitDataset(tmp_path / "joined/dataset", splits=("development",))
    migrated = read_circuit(dataset, static.circuit_id)
    np.testing.assert_array_equal(migrated.source_tuning_angle_deg, [45., 90., 135.])
    np.testing.assert_array_equal(migrated.unit_ids, static.unit_ids)
    assert report["source_tuning_circuits"] == 1 and report["source_tuning_values_verified"] == 3
    assert report["new_source_simulations"] == report["production_episodes"] == 0
    original = read_episode(CircuitDataset(source, splits=("development",)), next(
        CircuitDataset(source, splits=("development",)).iter_episodes()))
    actual = read_episode(dataset, next(dataset.iter_episodes()))
    assert actual.episode_id == original.episode_id and actual.seeds == original.seeds
    np.testing.assert_array_equal(actual.spike_ticks, original.spike_ticks)


def test_bound_source_tuning_rejects_wrong_nodes_and_unrecorded_identity(tmp_path):
    static, path = bound_node_fixture(tmp_path)
    source = archive(tmp_path / "source", "episode", circuit=static)
    other = tmp_path / "other_nodes.h5"
    other.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="BOUND_SOURCE_NODES_MISSING"):
        assemble([source], tmp_path / "rejected", scenario_id="v1_visual", source_node_files=[other])
    assert not (tmp_path / "rejected").exists()
    unbound = archive(tmp_path / "unbound", "unbound")
    with pytest.raises(ValueError, match="BOUND_SOURCE_NODES_MISSING"):
        assemble([unbound], tmp_path / "unrecorded", scenario_id="v1_visual", source_node_files=[path])
    assert not (tmp_path / "unrecorded").exists()


def test_bound_source_tuning_rejects_inconsistent_geometry_and_invalid_angles(tmp_path):
    from .benchmark_dataset import bind_source_tuning
    static, path = bound_node_fixture(tmp_path)
    with pytest.raises(ValueError, match="COORDINATES_DIFFER"):
        bind_source_tuning(replace(static, coordinates=static.coordinates + 1.), path)
    with pytest.raises(ValueError, match="EXISTING_ANGLES_DIFFER"):
        bind_source_tuning(replace(static, source_tuning_angle_deg=np.array([1., 2., 3.])), path)
    ca3 = archive(tmp_path / "ca3_source", "ca3_episode", scenario="ca3_pattern_completion")
    with pytest.raises(ValueError, match="SOURCE_TUNING_V1_ONLY"):
        assemble([ca3], tmp_path / "ca3", scenario_id="ca3_pattern_completion", source_node_files=[path])

    
def test_source_tuning_migration_keeps_conflicting_population_rejection(tmp_path):
    static, nodes = bound_node_fixture(tmp_path)
    first = archive(tmp_path / "first", "first_episode", circuit=static)
    second = archive(tmp_path / "second", "second_episode",
        circuit=replace(static, populations=np.array(["I", "E", "I"])))
    with pytest.raises(ValueError, match="OBSERVATION_MAP_REBOUND"):
        assemble([first, second], tmp_path / "rejected", scenario_id="v1_visual",
                 source_node_files=[nodes])
    failed = json.loads((tmp_path / "rejected/report.json").read_text())
    assert failed["status"] == "FAILED_BENCHMARK_DEVELOPMENT_DATASET"
    assert failed["complete_episodes"] == 0

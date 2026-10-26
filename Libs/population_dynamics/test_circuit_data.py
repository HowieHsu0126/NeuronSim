"""Time/identity round trips and interrupted acquisition use disposable fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from .circuit_data import SCHEMA, CircuitUnits, CircuitEpisode, CircuitArchiveWriter, CircuitDataset
from .circuit_campaign import campaign_config, episode_identities, freeze_identities
from .provenance import sha256_file


def binding():
    return dict(schema=SCHEMA, sample_unit="complete_mechanistic_episode",
                scenario_id="pfc_working_memory", scope="source_reproduction", source_revision="test-source")


def units():
    return CircuitUnits("pfc_a", np.array([19, 3, 72]), np.array(["E", "E", "I"]),
                        np.array([0, 0, 1]), ("E", "I"), "stimulus_preference", "test-source",
                        np.array([[0.], [180.], [0.]]), "degree",
                        {"connectivity_identity": "test-graph", "parameter_identity": "test-parameters"})


def episode(name="episode_a", split="train"):
    return CircuitEpisode(name, "pfc_a", "pfc_a_block_0", split, 10, .1,
                          np.array([0, 3, 4, 5, 8, 10]), np.array([0, 1, 2, 0, 2, 1]),
                          np.array([0, 3, 7]), np.array([[0.], [2.], [0.]]),
                          ({"name": "current", "unit": "pA"},),
                          source_record={"complete_unit_recording": True, "simulation_completed": True,
                                         "recorded_until_tick": 10, "expected_duration_ticks": 10,
                                         "core_unit_count": 3, "source_revision": "test-source",
                                         "circuit_provenance": units().circuit_provenance},
                          privileged={"hidden_future_state": np.array([100.])}, task_outcome="failed_memory")


class CircuitArchiveTests(unittest.TestCase):
    def test_complete_episode_preserves_all_units_exact_clock_and_window_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(units())
                writer.add(episode())
            dataset = CircuitDataset(root, splits=("train",))
            sequence = dataset.complete_episode("episode_a", bin_ticks=2)
            window = dataset.window("episode_a", boundary_tick=4, history_ticks=4,
                                    horizon_ticks=6, bin_ticks=2)
            np.testing.assert_array_equal(sequence["counts"],
                                          np.concatenate([window["history"], window["targets"]]))
            np.testing.assert_array_equal(sequence["field_counts"],
                                          np.concatenate([window["history_fields"], window["target_fields"]]))
            for key in ("unit_ids", "unit_field_index", "known_controls"):
                np.testing.assert_array_equal(sequence[key], window[key])
            self.assertEqual(sequence["counts"].sum(), 5)
            self.assertEqual(sequence["counts"].shape, (5, 3))
            self.assertEqual((sequence["consumed_start_tick"], sequence["consumed_stop_tick_exclusive"]), (0, 10))
            self.assertEqual(sequence["duration_ticks"], sequence["counts"].shape[0] * sequence["bin_ticks"])
            self.assertEqual(sequence["block_id"], "pfc_a_block_0")
            self.assertEqual(sequence["split"], "train")
            self.assertEqual(sequence["source_dt_ms"], .1)
            self.assertEqual(sequence["bin_seconds"], .0002)
            for key in ("task_outcome", "privileged", "source_record"):
                self.assertNotIn(key, sequence)
            with h5py.File(root / "episodes_000000.h5", "r") as raw:
                np.testing.assert_array_equal(raw["episode_a/spike_ticks"][:], [0, 3, 4, 5, 8, 10])

    def test_complete_episode_requires_exact_integer_divisor_before_source_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(units())
                writer.add(episode())
            dataset = CircuitDataset(root, splits=("train",))
            for width in (0, -1, 3, 11, 2., True, np.int64(2)):
                with self.subTest(width=width), patch("population_dynamics.circuit_data.h5py.File",
                        side_effect=AssertionError("INVALID_CLOCK_OPENED_SOURCE")):
                    with self.assertRaisesRegex(ValueError, "EXACT_COMPLETE_EPISODE_BIN_DIVISOR"):
                        dataset.complete_episode("episode_a", bin_ticks=width)

    def test_source_metadata_retains_optional_descriptors_and_coordinate_semantics(self):
        for optional in (True, False):
            with self.subTest(optional=optional), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = replace(units(), node_type_ids=np.array([101, 101, 202])) if optional else replace(
                    units(), coordinates=None, coordinate_units=None)
                with CircuitArchiveWriter(root, binding()) as writer:
                    writer.register_circuit(source)
                    writer.add(episode())
                metadata = CircuitDataset(root, splits=("train",)).circuit_metadata("pfc_a")
                np.testing.assert_array_equal(metadata["unit_ids"], source.unit_ids)
                np.testing.assert_array_equal(metadata["unit_field_index"], source.field_index)
                np.testing.assert_array_equal(metadata["populations"], source.populations)
                self.assertEqual(metadata["field_names"], source.field_names)
                self.assertEqual(metadata["source_revision"], source.source_revision)
                self.assertEqual(metadata["circuit_provenance"], source.circuit_provenance)
                self.assertEqual(metadata["coordinate_semantics"], "stimulus_preference")
                self.assertEqual(metadata["coordinate_units"], "degree" if optional else "not_available")
                if optional:
                    np.testing.assert_array_equal(metadata["coordinates"], source.coordinates)
                    np.testing.assert_array_equal(metadata["node_type_ids"], source.node_type_ids)
                else:
                    self.assertIsNone(metadata["coordinates"])
                    self.assertIsNone(metadata["node_type_ids"])
                for key in ("task_outcome", "privileged", "source_record"):
                    self.assertNotIn(key, metadata)

    def test_unselected_circuits_and_episodes_never_open_source_payload(self):
        for split in ("validation", "calibration", "test"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with CircuitArchiveWriter(root, binding()) as writer:
                    writer.register_circuit(units())
                    writer.add(episode(split=split))
                with patch("population_dynamics.circuit_data.h5py.File",
                           side_effect=AssertionError("UNSELECTED_SOURCE_OPENED")):
                    dataset = CircuitDataset(root, splits=("train",))
                    with self.assertRaises(KeyError):
                        dataset.complete_episode("episode_a", bin_ticks=2)
                    with self.assertRaisesRegex(ValueError, "CIRCUIT_OUTSIDE_SELECTED_SPLITS"):
                        dataset.circuit_metadata("pfc_a")

    def test_catalog_rejects_duplicate_counts_missing_circuits_and_split_leakage(self):
        cases = ("duplicate_episode", "duplicate_shard", "missing_circuit", "split_leakage",
                 "count", "unknown_split")
        messages = ("DUPLICATE_COMPLETE_EPISODE", "DUPLICATE_CIRCUIT_SHARD",
                    "EPISODE_CIRCUIT_METADATA_MISSING", "WHOLE_CIRCUIT_SPLIT_LEAKAGE",
                    "COMPLETE_EPISODE_CATALOG_COUNT_DIFFERS", "UNKNOWN_CIRCUIT_EPISODE_SPLIT")
        for case, message in zip(cases, messages):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with CircuitArchiveWriter(root, binding()) as writer:
                    writer.register_circuit(units())
                    writer.add(episode())
                catalog = json.loads((root / "dataset.json").read_text())
                row = catalog["shards"][0]["episodes"][0]
                if case == "duplicate_episode":
                    catalog["shards"][0]["episodes"].append(dict(row))
                elif case == "duplicate_shard":
                    catalog["shards"].append(dict(catalog["shards"][0]))
                elif case == "missing_circuit":
                    row["circuit_id"] = "absent"
                elif case == "split_leakage":
                    catalog["shards"][0]["episodes"].append(dict(row, episode_id="hidden_test", split="test"))
                elif case == "count":
                    catalog["complete_episodes"] = 2
                else:
                    row["split"] = "unknown"
                (root / "dataset.json").write_text(json.dumps(catalog))
                with patch("population_dynamics.circuit_data.h5py.File",
                           side_effect=AssertionError("INVALID_CATALOG_OPENED_SOURCE")):
                    with self.assertRaisesRegex(ValueError, message):
                        CircuitDataset(root, splits=("train",))

    def test_catalog_payload_identity_event_count_and_unit_count_must_agree(self):
        for case in ("event_count", "event_array", "unit_count", "source_identity"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                with CircuitArchiveWriter(root, binding()) as writer:
                    writer.register_circuit(units())
                    writer.add(episode())
                catalog = json.loads((root / "dataset.json").read_text())
                row = catalog["shards"][0]["episodes"][0]
                if case == "event_count":
                    row["event_count"] -= 1
                elif case == "unit_count":
                    catalog["circuits"]["pfc_a"]["units"] -= 1
                else:
                    path = root / catalog["shards"][0]["path"]
                    with h5py.File(path, "r+") as raw:
                        group = raw["episode_a"]
                        if case == "event_array":
                            del group["spike_unit_indices"]
                            group.create_dataset("spike_unit_indices", data=[0, 1])
                        else:
                            metadata = json.loads(group.attrs["metadata_json"])
                            metadata["block_id"] = "foreign_block"
                            group.attrs["metadata_json"] = json.dumps(metadata)
                    catalog["shards"][0]["sha256"] = sha256_file(path)
                (root / "dataset.json").write_text(json.dumps(catalog))
                message = ("SOURCE_UNIT_COUNT" if case == "unit_count" else
                           "SOURCE_EPISODE_IDENTITY" if case == "source_identity" else "SOURCE_EVENT_COUNT")
                with self.assertRaisesRegex(ValueError, message):
                    CircuitDataset(root, splits=("train",)).complete_episode("episode_a", bin_ticks=2)

    def test_production_admission_is_bound_to_prospective_rows(self):
        config = campaign_config(Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml")
        scenario = "pfc_working_memory"
        config["scenarios"][scenario].update(parameter_domain={"scope":"synthetic_test_fixture"},
            observation_map={"scope":"synthetic_test_fixture"},
            task_protocol={"dt_ms":.1, "condition_duration_ticks":[10]*10})
        qualification = dict(status="PASS", scenario_id=scenario, coupled_core_neurons=100000,
                             source_revision="test-source", evidence=["synthetic_unit_test_only"],
                             checks={key:True for key in ("source_reproduction", "numerical_integrity", "driver_statistics",
                                                         "control_delivery", "full_unit_recording", "coupling", "scaling_fidelity")},
                             source_clock={"dt_ms": .1, "backend": "numpy"},
                             source_admission_route="coarse_engine",
                             cross_clock_acceptance={
                                 "status": "PASS", "complete_frozen_domain": True,
                                 "combined_error_budget_accepted": True,
                                 "reference_cohort": "numpy_dt_0p0125",
                                 "evidence": ["synthetic_unit_test_only"]})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "qualification.json"
            source.write_text(json.dumps(qualification))
            table = root / "identities.csv"
            freeze_identities(config, scenario, source, table)
            reference = next(episode_identities(config, scenario, production=True))
            large_units = replace(units(), circuit_id=reference["circuit_id"],
                                  unit_ids=np.arange(100000), populations=np.array(["E"]*100000),
                                  field_index=np.zeros(100000,dtype=np.int64), field_names=("E",),
                                  coordinates=None, coordinate_units=None)
            valid = replace(episode(), episode_id=reference["episode_id"], circuit_id=reference["circuit_id"],
                            block_id=reference["block_id"],
                            source_record=dict(episode().source_record,core_unit_count=100000),
                            seeds={key:reference[key] for key in ("connectivity_seed","initial_state_seed","background_seed","stimulus_seed")})
            with CircuitArchiveWriter(root / "archive", dict(binding(),scope="production"),
                                      production_config=config,qualification_path=source,identity_table=table) as writer:
                writer.register_circuit(large_units)
                with self.assertRaisesRegex(ValueError, "FROZEN_PRODUCTION"):
                    writer.add(replace(valid, episode_id="renamed_episode"))
                with self.assertRaisesRegex(ValueError, "FROZEN_PRODUCTION"):
                    writer.add(replace(valid, split="validation"))
                with self.assertRaisesRegex(ValueError, "STREAM_IDENTITY"):
                    writer.add(replace(valid, seeds={}))
                short = replace(valid,duration_ticks=8,spike_ticks=np.array([0,3,4,5,7]),
                                spike_unit_indices=np.array([0,1,2,0,2]),
                                source_record=dict(valid.source_record,recorded_until_tick=8,expected_duration_ticks=8))
                with self.assertRaisesRegex(ValueError, "FROZEN_PROTOCOL"):
                    writer.add(short)
                writer.add(valid)
            self.assertEqual(CircuitDataset(root / "archive").catalog["complete_episodes"],1)

    def test_optional_source_model_ids_roundtrip_and_cannot_rebind(self):
        enriched = replace(units(), node_type_ids=np.array([101, 101, 202]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(enriched)
                writer.register_circuit(enriched)
                with self.assertRaisesRegex(ValueError, "CIRCUIT_ID_REBOUND:node_type_ids"):
                    writer.register_circuit(replace(enriched, node_type_ids=np.array([101, 202, 202])))
                with self.assertRaisesRegex(ValueError, "CIRCUIT_ID_REBOUND:node_type_ids"):
                    writer.register_circuit(units())
            with h5py.File(root / "circuits/pfc_a.h5", "r") as stored:
                np.testing.assert_array_equal(stored["node_type_ids"][:], [101, 101, 202])

    def test_source_model_ids_require_one_integral_identity_per_unit(self):
        for ids in (np.array([1, 2]), np.array([1., 2., 3.]), np.array([-1, 2, 3])):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    replace(units(), node_type_ids=ids).validate()

    def test_causal_half_open_counts_and_known_program(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(units())
                writer.add(episode())
            dataset = CircuitDataset(root, splits=("train",))
            result = dataset.window("episode_a", boundary_tick=4, history_ticks=4, horizon_ticks=6, bin_ticks=2)
            expected = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 1]])
            np.testing.assert_array_equal(np.concatenate([result["history"], result["targets"]]), expected)
            np.testing.assert_array_equal(result["unit_ids"], [19, 3, 72])
            np.testing.assert_array_equal(result["unit_field_index"], [0, 0, 1])
            self.assertEqual(result["field_names"], ("E", "I"))
            np.testing.assert_array_equal(result["known_controls"][:, 0], [0, 1, 2, 1, 0])
            self.assertEqual(result["bin_seconds"], .0002)
            self.assertEqual(result["source_dt_ms"], .1)
            self.assertEqual(result["control_kind"], "piecewise_constant")
            self.assertNotIn("privileged", result)
            self.assertNotIn("source_record", result)
            self.assertEqual(dataset.catalog["complete_episodes"], 1)
            shard = dataset.catalog["shards"][0]
            with h5py.File(root / shard["path"], "r") as raw:
                self.assertEqual(raw["episode_a/spike_ticks"][-1], 10)
                self.assertEqual(json.loads(raw["episode_a"].attrs["metadata_json"])["task_outcome"], "failed_memory")
            subset = dataset.window("episode_a", boundary_tick=4, history_ticks=4, horizon_ticks=6,
                                    bin_ticks=2, unit_start=1, unit_stop=3)
            np.testing.assert_array_equal(subset["history"], expected[:2, 1:])
            np.testing.assert_array_equal(subset["unit_ids"], [3, 72])
            np.testing.assert_array_equal(subset["unit_field_index"], [0, 1])
            self.assertEqual(subset["field_names"], result["field_names"])
            self.assertEqual(subset["source_dt_ms"], .1)
            self.assertEqual(subset["control_kind"], "piecewise_constant")
            np.testing.assert_array_equal(subset["target_fields"], result["target_fields"])

    def test_duplicate_split_and_identity_rebinding_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with CircuitArchiveWriter(Path(folder), binding()) as writer:
                original = units()
                writer.register_circuit(original)
                writer.add(episode())
                with self.assertRaisesRegex(ValueError, "DUPLICATE"):
                    writer.add(episode())
                with self.assertRaisesRegex(ValueError, "SPLIT_LEAKAGE"):
                    writer.add(episode("another", "test"))
                with self.assertRaisesRegex(ValueError, "REBOUND"):
                    writer.register_circuit(replace(original, populations=np.array(["I", "E", "I"])))
                with self.assertRaisesRegex(ValueError, "REBOUND"):
                    writer.register_circuit(replace(original, circuit_provenance=dict(original.circuit_provenance, connectivity_identity="changed")))

    def test_control_impulses_and_invalid_source_events(self):
        with tempfile.TemporaryDirectory() as folder:
            with CircuitArchiveWriter(Path(folder), binding()) as writer:
                writer.register_circuit(units())
                writer.add(replace(episode(), control_kind="impulse", control_ticks=np.array([4, 4, 10]),
                                   control_values=np.array([[1.], [2.], [99.]])))
                with self.assertRaisesRegex(ValueError, "SOURCE_EVENTS"):
                    writer.add(replace(episode("bad"), spike_ticks=np.array([0, 3, 5, 4, 8, 10])))
            result = CircuitDataset(Path(folder)).window("episode_a", boundary_tick=4, history_ticks=4,
                                                       horizon_ticks=6, bin_ticks=2)
            np.testing.assert_array_equal(result["known_controls"][:, 0], [0, 0, 3, 0, 0])
            self.assertEqual(result["control_kind"], "impulse")
            self.assertEqual(result["source_dt_ms"], .1)
            np.testing.assert_array_equal(result["unit_field_index"], [0, 0, 1])

    def test_incomplete_source_and_unspecified_control_prefix_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with CircuitArchiveWriter(Path(folder), binding()) as writer:
                writer.register_circuit(units())
                original = episode()
                with self.assertRaisesRegex(ValueError, "COMPLETION_NOT_ESTABLISHED"):
                    writer.add(replace(original, source_record=dict(original.source_record, recorded_until_tick=8)))
                with self.assertRaisesRegex(ValueError, "INITIAL_CONTROL"):
                    writer.add(replace(original, control_ticks=np.array([1, 3, 7])))
                with self.assertRaisesRegex(ValueError, "BINDING_CHANGED"):
                    writer.add(replace(original, source_record=dict(original.source_record, source_revision="other")))
            self.assertEqual(CircuitDataset(Path(folder)).catalog["complete_episodes"], 0)

    def test_recovery_after_hard_interruption_and_after_shard_rename(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(units())
            (root / "episodes_000000.h5.writing").write_bytes(b"incomplete HDF5")
            with CircuitArchiveWriter(root, binding(), resume=True) as writer:
                writer.add(episode())
            catalog = json.loads((root / "dataset.json").read_text())
            self.assertEqual(len(catalog["incomplete_packaging_attempts"]), 1)
            self.assertEqual(catalog["complete_episodes"], 1)
            # Simulate termination after HDF5 rename and before catalog publish.
            catalog["shards"] = []
            catalog["complete_episodes"] = 0
            (root / "dataset.json").write_text(json.dumps(catalog))
            with CircuitArchiveWriter(root, binding(), resume=True) as writer:
                with self.assertRaisesRegex(ValueError, "DUPLICATE"):
                    writer.add(episode())
            self.assertEqual(CircuitDataset(root).catalog["complete_episodes"], 1)

    def test_committed_shards_survive_interruption_without_duplicate_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding(), target_shard_bytes=1) as writer:
                writer.register_circuit(units())
                writer.add(episode())
            try:
                with CircuitArchiveWriter(root, binding(), resume=True) as writer:
                    writer.add(episode("interrupted"))
                    raise RuntimeError("simulated consumer crash")
            except RuntimeError:
                pass
            self.assertEqual(CircuitDataset(root).catalog["complete_episodes"], 1)
            with CircuitArchiveWriter(root, binding(), resume=True) as writer:
                writer.add(episode("interrupted"))
            self.assertEqual(CircuitDataset(root).catalog["complete_episodes"], 2)

    def test_source_qualification_test_access_and_window_support(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "QUALIFICATION"):
                CircuitArchiveWriter(Path(folder), dict(binding(), scope="production"))
            with CircuitArchiveWriter(Path(folder), binding()) as writer:
                writer.register_circuit(units())
                writer.add(episode(split="test"))
            self.assertEqual(len(CircuitDataset(Path(folder)).episodes), 0)
            with self.assertRaisesRegex(ValueError, "TEST_ACCESS"):
                CircuitDataset(Path(folder), splits=("test",))
            dataset = CircuitDataset(Path(folder), splits=("test",), allow_test=True)
            with self.assertRaisesRegex(ValueError, "OUTSIDE_COMPLETE"):
                dataset.window("episode_a", boundary_tick=2, history_ticks=4, horizon_ticks=6, bin_ticks=2)


if __name__ == "__main__":
    unittest.main()


class SourceTuningMetadataTests(unittest.TestCase):
    def test_optional_tuning_roundtrip_and_rebind_rejection(self):
        from .benchmark_dataset import read_circuit
        source = replace(units(), source_tuning_angle_deg=np.array([45., 90., 315.]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with CircuitArchiveWriter(root, binding()) as writer:
                writer.register_circuit(source)
                writer.add(episode())
                with self.assertRaisesRegex(ValueError, "CIRCUIT_ID_REBOUND:source_tuning_angle_deg"):
                    writer.register_circuit(replace(source, source_tuning_angle_deg=np.array([45., 91., 315.])))
            dataset = CircuitDataset(root, splits=("train",))
            np.testing.assert_array_equal(dataset.circuit_metadata("pfc_a")["source_tuning_angle_deg"], source.source_tuning_angle_deg)
            np.testing.assert_array_equal(read_circuit(dataset, "pfc_a").source_tuning_angle_deg, source.source_tuning_angle_deg)

    def test_invalid_tuning_shape_units_and_values_reject(self):
        for value in (np.array([1., 2.]), np.array([0., np.nan, 1.]), np.array([-1., 1., 2.]), np.array([0., 360., 1.])):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "SOURCE_TUNING_ANGLE"):
                replace(units(), source_tuning_angle_deg=value).validate()

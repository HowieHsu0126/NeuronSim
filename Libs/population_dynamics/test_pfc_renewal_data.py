"""PFC source admission and tick-wise exposure checks on disposable archives."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .circuit_data import CircuitDataset
from .pfc_renewal_data import PfcRenewalEpisode, MINIMUM_INTERVAL_POLICY
from . import test_pfc_source_dataset as fixtures
from . import pfc_source_dataset as exporter


class PfcRenewalDataTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PfcSourceDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.source = fixture.source()
        exporter.export([self.source], fixture.config, self.root / "export")
        self.dataset = CircuitDataset(self.root / "export/dataset", splits=("development",))
        self.source_entry = Path(__file__).resolve().parents[1] / (
            "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py")

    def adapter(self, **changes):
        args = dict(bin_ticks=100, source_run=self.source, source_entry=self.source_entry)
        args.update(changes)
        return PfcRenewalEpisode(self.dataset, "pfc_94001_strong_cue", **args)

    def test_source_support_and_exposures_match_independent_tick_simulation(self):
        data = self.adapter()
        np.testing.assert_array_equal(data.minimum_interval_ticks, [40] * 16 + [20] * 4)
        self.assertEqual(data.binding["minimum_interval_policy"], MINIMUM_INTERVAL_POLICY)
        self.assertEqual(data.binding["split"], "development")
        for start, stop in ((0, 100000), (2000, 2500), (80000, 80500)):
            with self.subTest(start=start, stop=stop):
                result = data.renewal_window(start, stop)
                ready = np.zeros(20, dtype=np.int64)
                expected_counts = np.zeros_like(result.counts)
                expected_exposure = np.zeros_like(result.eligible_ticks)
                events = {}
                for tick, unit in zip(data.event_ticks, data.event_units):
                    events.setdefault(int(tick), []).append(int(unit))
                initial = None
                for tick in range(stop):
                    if tick == start:
                        initial = np.maximum(ready - tick, 0)
                    if tick >= start:
                        expected_exposure[(tick - start) // data.bin_ticks] += ready <= tick
                    for unit in events.get(tick, []):
                        self.assertLessEqual(ready[unit], tick)
                        ready[unit] = tick + data.minimum_interval_ticks[unit]
                        if tick >= start:
                            expected_counts[(tick - start) // data.bin_ticks, unit] += 1
                np.testing.assert_array_equal(result.counts, expected_counts)
                np.testing.assert_array_equal(result.eligible_ticks, expected_exposure)
                np.testing.assert_array_equal(result.initial_ready_in_ticks, initial)
                np.testing.assert_array_equal(result.terminal_ready_in_ticks, np.maximum(ready - stop, 0))
                np.testing.assert_array_equal(data.ready_at(start), initial)
                np.testing.assert_array_equal(data.ready_at(stop), result.terminal_ready_in_ticks)
                self.assertEqual(set(result.__dict__), {"counts", "eligible_ticks", "stimulus",
                    "initial_ready_in_ticks", "terminal_ready_in_ticks"})

    def test_prefix_availability_and_exposure_ignore_future_events(self):
        data = self.adapter()
        result = data.renewal_window(2000, 2500)
        before = data.ready_at(2500)
        future = data.event_ticks >= 2500
        data.event_ticks[future] += 1
        again = data.renewal_window(2000, 2500)
        np.testing.assert_array_equal(data.ready_at(2500), before)
        np.testing.assert_array_equal(again.eligible_ticks, result.eligible_ticks)
        np.testing.assert_array_equal(again.initial_ready_in_ticks, result.initial_ready_in_ticks)
        np.testing.assert_array_equal(again.terminal_ready_in_ticks, result.terminal_ready_in_ticks)

    def test_foreign_source_changed_config_and_unselected_episode_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            foreign = Path(folder) / "source.py"
            foreign.write_text("different source")
            with self.assertRaisesRegex(ValueError, "SOURCE_ENTRY_CHANGED"):
                self.adapter(source_entry=foreign)
        original = self.dataset.catalog["binding"]["config_sha256"]
        self.dataset.catalog["binding"]["config_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "DATASET_SOURCE_BINDING_CHANGED"):
            self.adapter()
        self.dataset.catalog["binding"]["config_sha256"] = original
        row = self.dataset.episodes["pfc_94001_strong_cue"]
        block = row["block_id"]
        row["block_id"] = "foreign_block"
        with self.assertRaisesRegex(ValueError, "SOURCE_BLOCK_IDENTITY_CHANGED"):
            self.adapter()
        row["block_id"] = block
        selected = self.dataset.episodes
        self.dataset.episodes = {}
        with patch("population_dynamics.pfc_renewal_data.sha256_file",
                   side_effect=AssertionError("UNSELECTED_EPISODE_READ_SOURCE")):
            with self.assertRaises(KeyError):
                self.adapter()
        self.dataset.episodes = selected

    def test_source_support_violation_and_partial_clocks_reject(self):
        for width in (0, 3, 100001, True):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "EXACT_COMPLETE_CLOCK"):
                self.adapter(bin_ticks=width)
        data = self.adapter()
        with self.assertRaisesRegex(ValueError, "COMPLETE_ALIGNED_WINDOW"):
            data.renewal_window(0, 101)
        unit = int(data.event_units[0])
        tick = int(data.event_ticks[0])
        data.event_ticks = np.r_[tick, tick + 1]
        data.event_units = np.array([unit, unit])
        with self.assertRaisesRegex(ValueError, "violates the declared minimum interval"):
            data.renewal_window(0, 1000)
        with self.assertRaisesRegex(ValueError, "violates the declared minimum interval"):
            data.ready_at(1000)

    def test_shared_population_descriptors_use_source_preference_and_explicit_missingness(self):
        data = self.adapter()
        metadata = data.population_metadata()
        units = np.asarray(metadata["unit_features"])
        fields = np.asarray(metadata["field_features"])
        np.testing.assert_array_equal(units[:, 0], [1] * 16 + [0] * 4)
        np.testing.assert_array_equal(units[:, 1], [0] * 16 + [1] * 4)
        np.testing.assert_allclose(units[:16, 2], np.sin(2 * np.pi * np.arange(16) / 16), rtol=0, atol=0)
        np.testing.assert_allclose(units[:16, 3], np.cos(2 * np.pi * np.arange(16) / 16), rtol=0, atol=0)
        np.testing.assert_array_equal(units[16:, 2:5], 0)
        np.testing.assert_array_equal(fields[-1, :5], [0, 1, 0, 0, 0])
        np.testing.assert_allclose(fields[:, 5], np.log1p(np.bincount(data.unit_field_index)), rtol=0, atol=0)
        self.assertEqual(metadata["control_features"][0]["unit"], "nA")
        self.assertNotIn("task_outcome", metadata)

    def test_complete_episode_population_consumer_uses_real_cli(self):
        from .circuit_training_check import main
        import contextlib
        import io
        import json
        output = self.root / "population_update.json"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--dataset", str(self.dataset.root), "--source-run", str(self.source),
                  "--source-entry", str(self.source_entry), "--episode-id", "pfc_94001_strong_cue",
                  "--output", str(output), "--seed", "41"])
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "ENGINEERING_UPDATE_PASS")
        self.assertEqual(report["complete_sequence_bins"], 1000)
        self.assertEqual(report["consumed_stop_tick_exclusive"], 100000)
        self.assertEqual(report["model_family"], "metadata_conditioned_population")
        self.assertEqual(report["generated_unit_count_values"], 20)
        self.assertTrue(report["replay_exact"])
        self.assertFalse(report["scientific_claim_qualified"])
        with self.assertRaisesRegex(ValueError, "REQUIRED_TOGETHER"):
            main(["--dataset", str(self.dataset.root), "--source-run", str(self.source),
                  "--output", str(self.root / "invalid.json")])

    def test_compiled_task_adapter_exposes_geometry_and_preserves_raw_commands(self):
        from .population_state import ControlledPopulationState
        from .pfc_population_evaluation import prepare_factual_response

        fixture = fixtures.PfcSourceDatasetTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source, identity = fixture.benchmark_source("_renewal")
        output = fixture.root / "task_export"
        exporter.export([source], fixture.config, output)
        dataset = CircuitDataset(output / "dataset", splits=("development",))
        adapter = PfcRenewalEpisode(dataset, identity["episode_id"], bin_ticks=100,
            source_run=source, source_entry=self.source_entry)
        metadata = adapter.population_metadata()
        unit = np.asarray(metadata["unit_features"])
        names = [item["name"] for item in metadata["unit_feature_schema"]]
        np.testing.assert_array_equal(unit[:, names.index("cue_target")], np.r_[np.isin(np.arange(16), [4,5,6]), np.zeros(4)])
        np.testing.assert_array_equal(unit[:, names.index("distractor_target")], np.r_[np.isin(np.arange(16), [3,4,5]), np.zeros(4)])
        field = np.asarray(metadata["field_features"])
        field_names = [item["name"] for item in metadata["field_feature_schema"]]
        for name, source_name in (("cue_target_fraction", "cue_target"), ("distractor_target_fraction", "distractor_target")):
            expected = [unit[adapter.unit_field_index == index, names.index(source_name)].mean()
                        for index in range(len(adapter.field_names))]
            np.testing.assert_array_equal(field[:, field_names.index(name)], expected)
        assert "center_degree" not in metadata["control_features"][1]
        model = ControlledPopulationState(unit_ids=list(map(int, adapter.unit_ids)),
            unit_field_index=adapter.unit_field_index.tolist(), field_names=adapter.field_names,
            minimum_interval_ticks=adapter.minimum_interval_ticks.tolist(), tick_seconds=.00005,
            bin_seconds=.005, input_size=4, latent_channels=2, encoder_size=5, metadata=metadata)
        boundary, future = prepare_factual_response(model, adapter, 48000, 10)
        assert boundary.mean.shape == (1,5,2)
        np.testing.assert_array_equal(future["known_controls"][:,1], .5)
        assert future["control_features"] == metadata["control_features"]
        assert future["source_control_features"][1]["center_degree"] == 90.
        # Geometry changes the model input without relabeling command channels.
        adapter.source_config = dict(adapter.source_config, distractor_center_deg=300.)
        changed = adapter.population_metadata()
        assert changed["control_features"] == metadata["control_features"]
        assert not np.array_equal(np.asarray(changed["unit_features"]), unit)
        transferred = model.training_for_circuit(unit_ids=list(map(int,adapter.unit_ids)),
            unit_field_index=adapter.unit_field_index.tolist(), field_names=adapter.field_names,
            minimum_interval_ticks=adapter.minimum_interval_ticks.tolist(), metadata=changed)
        assert all(a is b for a,b in zip(model.parameters(), transferred.parameters()))
        assert not torch.equal(model.unit_features, transferred.unit_features)

    def test_compiled_task_adapter_rejects_changed_registration_before_raw_reads(self):
        fixture = fixtures.PfcSourceDatasetTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        source, identity = fixture.benchmark_source("_rejected_renewal")
        output = fixture.root / "task_export"
        exporter.export([source], fixture.config, output)
        dataset = CircuitDataset(output / "dataset", splits=("development",))
        config = dataset.catalog["binding"]["resolved_source_config"]
        config["distractor_start_ms"] = 2200.
        with patch("population_dynamics.pfc_renewal_data._source_values", side_effect=AssertionError("RAW_READ")):
            with self.assertRaisesRegex(ValueError,"COMPILER_BINDING"):
                PfcRenewalEpisode(dataset, identity["episode_id"], bin_ticks=100,
                    source_run=source, source_entry=self.source_entry)


if __name__ == "__main__":
    unittest.main()

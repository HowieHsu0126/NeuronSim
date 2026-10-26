"""Complete source-clock coverage through the real archive reader and LDS update."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from .circuit_data import CircuitArchiveWriter
from .circuit_training_check import check
from .poisson_state_space import ControlledPoissonLDS
from .test_circuit_data import binding, episode, units


def write_episode(root, *, duration=1001, dt_ms=1., split="development"):
    original = episode(split=split)
    value = replace(original, duration_ticks=duration, dt_ms=dt_ms,
                    spike_ticks=np.array([0, duration - 1, duration]),
                    spike_unit_indices=np.array([0, 2, 1]),
                    control_ticks=np.array([0, duration - 1, duration]),
                    control_values=np.array([[.2], [.4], [99.]]),
                    source_record=dict(original.source_record, recorded_until_tick=duration,
                                       expected_duration_ticks=duration))
    with CircuitArchiveWriter(root, binding()) as writer:
        writer.register_circuit(units())
        writer.add(value)


class CircuitTrainingCheckTests(unittest.TestCase):
    def run_check(self, root):
        observed = {}
        original_loss = ControlledPoissonLDS.sequence_loss

        def capture_loss(model, counts, controls):
            observed.update(counts=counts.detach().clone(), controls=controls.detach().clone(),
                            unit_ids=model.unit_ids, bin_seconds=model.bin_seconds,
                            training_seed=model.initialization_seed)
            return original_loss(model, counts, controls)

        with patch.object(ControlledPoissonLDS, "sequence_loss", capture_loss), patch(
                "population_dynamics.circuit_training_check.CircuitDataset.window",
                side_effect=AssertionError("COMPLETE_EPISODE_CONSUMER_USED_WINDOW")):
            report = check(root / "data", root / "report.json")
        self.assertEqual(json.loads((root / "report.json").read_text()), report)
        self.assertEqual(report["status"], "ENGINEERING_UPDATE_PASS")
        self.assertEqual(report["training_seed"], 41)
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertTrue(report["gradient_finite"])
        self.assertTrue(report["parameters_changed"])
        self.assertFalse(report["formal_training_completed"])
        self.assertFalse(report["generative_fidelity_qualified"])
        self.assertEqual(observed["unit_ids"], (19, 3, 72))
        self.assertEqual(observed["training_seed"], 41)
        self.assertEqual(observed["bin_seconds"], report["bin_seconds"])
        self.assertEqual(report["history_ticks"] + report["horizon_ticks"],
                         report["source_duration_ticks"])
        self.assertEqual(report["complete_sequence_bins"] * report["bin_ticks"],
                         report["source_duration_ticks"])
        self.assertEqual(report["consumed_start_tick"], 0)
        self.assertEqual(report["consumed_stop_tick_exclusive"], report["source_duration_ticks"])
        return report, observed

    def test_odd_episode_includes_last_source_tick_and_retains_half_open_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root / "data")
            report, observed = self.run_check(root)
            self.assertEqual((report["history_ticks"], report["horizon_ticks"]), (500, 501))
            self.assertEqual(report["complete_sequence_bins"], 1001)
            self.assertEqual(report["bin_ticks"], 1)
            self.assertEqual(report["bin_seconds"], .001)
            expected = torch.zeros(1, 1001, 3)
            expected[0, 0, 0] = 1
            expected[0, 1000, 2] = 1
            torch.testing.assert_close(observed["counts"], expected, atol=0, rtol=0)
            torch.testing.assert_close(observed["controls"][0, -1], torch.tensor([.4]))
            with h5py.File(root / "data/episodes_000000.h5", "r") as raw:
                self.assertEqual(raw["episode_a/spike_ticks"][-1], 1001)

    def test_divisible_episode_preserves_five_ms_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root / "data", duration=1000, dt_ms=.1, split="train")
            report, observed = self.run_check(root)
            self.assertEqual(report["split"], "train")
            self.assertEqual((report["history_ticks"], report["horizon_ticks"]), (500, 500))
            self.assertEqual(report["complete_sequence_bins"], 20)
            self.assertEqual(report["bin_ticks"], 50)
            self.assertEqual(report["bin_seconds"], .005)
            expected = torch.zeros(1, 20, 3)
            expected[0, 0, 0] = 1
            expected[0, -1, 2] = 1
            torch.testing.assert_close(observed["counts"], expected, atol=0, rtol=0)
            self.assertAlmostEqual(observed["controls"][0, -1, 0].item(), .204, places=6)

    def test_odd_number_of_five_ms_bins_uses_unequal_aligned_halves(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root / "data", duration=1005)
            report, observed = self.run_check(root)
            self.assertEqual((report["history_ticks"], report["horizon_ticks"]), (500, 505))
            self.assertEqual(report["complete_sequence_bins"], 201)
            self.assertEqual(report["bin_ticks"], 5)
            self.assertEqual(observed["counts"][0, -1, 2].item(), 1)

    def test_nonintegral_five_ms_clock_uses_and_reports_native_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_episode(root / "data", duration=12, dt_ms=.3)
            report, observed = self.run_check(root)
            self.assertEqual(report["bin_ticks"], 1)
            self.assertEqual(report["bin_seconds"], .0003)
            self.assertEqual(report["complete_sequence_bins"], 12)
            self.assertEqual(observed["counts"][0, -1, 2].item(), 1)

    def test_protected_splits_never_open_source_files_or_train(self):
        for split in ("validation", "calibration", "test"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                write_episode(root / "data", duration=10, split=split)
                with patch("population_dynamics.circuit_data.h5py.File",
                           side_effect=AssertionError("PROTECTED_DATA_WAS_OPENED")):
                    with self.assertRaisesRegex(ValueError, "NO_TRAIN_OR_DEVELOPMENT_EPISODES"):
                        check(root / "data", root / "report.json")
                self.assertFalse((root / "report.json").exists())

    def test_single_tick_episode_rejects_without_a_partial_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = episode()
            value = replace(original, duration_ticks=1, dt_ms=1.,
                            spike_ticks=np.array([0, 1]), spike_unit_indices=np.array([0, 1]),
                            control_ticks=np.array([0]), control_values=np.array([[.2]]),
                            source_record=dict(original.source_record, recorded_until_tick=1,
                                               expected_duration_ticks=1))
            with CircuitArchiveWriter(root / "data", binding()) as writer:
                writer.register_circuit(units())
                writer.add(value)
            with self.assertRaisesRegex(ValueError, "REQUIRES_TWO_SOURCE_TICKS"):
                check(root / "data", root / "report.json")
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

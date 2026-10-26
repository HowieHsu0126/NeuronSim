"""Temporal boundaries, preserved identities and the canonical window-check CLI."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from .config import load_config
from .provenance import freeze_json, sha256_file
from .spike_windows import bin_spikes, check_recording_windows, window_edges

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


def source_fixture(path: Path, split: str = "train") -> None:
    path.mkdir()
    with h5py.File(path / "selected_recording.h5", "w") as recording:
        recording.attrs.update(spike_time_unit="seconds", session_id=847657808)
        recording["unit_ids"] = np.array([11, 3])
        recording["spike_offsets"] = np.array([0, 0, 5])
        recording["spike_times_s"] = np.array([0., 0.5, 1., 1.5, 2.])
    pd.DataFrame({"trial_id": ["a", "b", "c"], "session_id": [847657808] * 3,
                  "animal_id": [827809879] * 3, "condition_id": ["movie_condition"] * 3,
                  "split": [split] * 3, "stimulus_name": ["natural_movie_one_more_repeats"] * 3,
                  "start_time": [1., 4., 8.], "stop_time": [3., 6., 8.2],
                  "structurally_eligible": [True, True, True]}).to_csv(path / "trials.csv", index=False)
    pd.DataFrame({"start_time": [3.8], "stop_time": [3.9]}).to_csv(path / "invalid_times.csv", index=False)
    pd.DataFrame({"presentation_id": ["pa", "pb", "pc"], "start_time": [1., 4., 8.],
                  "stop_time": [3., 6., 8.2], "stimulus_name": ["natural_movie_one_more_repeats"] * 3,
                  "frame": [0, 0, 0]}).to_csv(path / "stimulus_presentations.csv", index=False)
    pd.DataFrame({"presentation_id": ["pa", "pb", "pc"], "trial_id": ["a", "b", "c"],
                  "session_id": [847657808] * 3, "animal_id": [827809879] * 3,
                  "split": [split] * 3}).to_csv(path / "trial_members.csv", index=False)
    freeze_json(path / "summary.json", {"session_id": 847657808, "animal_id": 827809879,
                "split": split, "recording_coverage_source_verified": False,
                "raw_unit_count": 7, "qc_eligible_unit_count": 4})
    freeze_json(path / "manifest.json", {
        "summary_sha256": sha256_file(path / "summary.json"),
        "selected_recording": {"path": "selected_recording.h5", "sha256": sha256_file(path / "selected_recording.h5")},
        "artifacts": {name: sha256_file(path / name) for name in ("trials.csv", "invalid_times.csv", "stimulus_presentations.csv", "trial_members.csv")}})


class BinningTests(unittest.TestCase):
    def test_half_open_bins_causal_boundary_and_silent_unit_order(self):
        result = bin_spikes({3: np.array([0., .5, 1., 1.5, 2.]), 11: np.array([])}, [11, 3],
                            np.array([1.]), history_seconds=1., horizon_seconds=1., bin_seconds=.5)
        np.testing.assert_array_equal(result.unit_ids, [11, 3])
        np.testing.assert_array_equal(result.counts[0, :, 0], [0, 0, 0, 0])
        np.testing.assert_array_equal(result.counts[0, :, 1], [1, 1, 1, 1])
        self.assertEqual(result.history.sum(), 2)
        self.assertEqual(result.response.sum(), 2)
        self.assertEqual(result.relative_edges_seconds[result.history_bin_count], 0.)

    def test_future_spikes_do_not_change_history(self):
        args = dict(unit_ids=[3], onsets_seconds=np.array([1.]), history_seconds=1., horizon_seconds=1., bin_seconds=.5)
        first = bin_spikes({3: np.array([.2, .7, 1.3])}, **args)
        second = bin_spikes({3: np.array([.2, .7, 1., 1.1, 1.3, 1.9])}, **args)
        np.testing.assert_array_equal(first.history, second.history)
        self.assertNotEqual(first.response.sum(), second.response.sum())

    def test_unsorted_and_nonfinite_source_values_reject(self):
        for values in (np.array([1., .5]), np.array([.5, np.nan]), np.array([2, 1], dtype=np.uint64)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "FINITE_SORTED"):
                bin_spikes({1: values}, [1], np.array([1.]), history_seconds=1., horizon_seconds=1., bin_seconds=.5)

    def test_ambiguous_window_durations_reject(self):
        for history, horizon, width in ((1., 1., .3), (0., 1., .5), (1., np.inf, .5), (1., 1., 0.)):
            with self.subTest(values=(history, horizon, width)), self.assertRaises(ValueError):
                window_edges(history, horizon, width)

    def test_unresolvable_absolute_time_and_duplicate_units_reject(self):
        for units, onsets in (([1, 1], [1.]), ([1], [1e20])):
            with self.subTest(units=units, onsets=onsets), self.assertRaises(ValueError):
                bin_spikes({1: np.array([])}, units, np.array(onsets), history_seconds=1., horizon_seconds=1., bin_seconds=.5)

    def test_overlapping_windows_preserve_each_trial_and_duplicate_events(self):
        result = bin_spikes({1: np.array([.5, .5, 1.])}, [1], np.array([1., 1.25]),
                            history_seconds=1., horizon_seconds=1., bin_seconds=.5)
        np.testing.assert_array_equal(result.counts.sum(axis=(1, 2)), [3, 3])


class WindowReplayTests(unittest.TestCase):
    def test_entrypoint_saves_counts_exclusions_and_never_promotes_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_fixture(root / "source")
            run = subprocess.run([sys.executable, "-B", "-m", "Libs.population_dynamics",
                                  "--config", str(CONFIG_PATH), "window-check", "--recording-dir", str(root / "source"),
                                  "--output-dir", str(root / "output")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stderr)
            summary = json.loads(run.stdout)
            self.assertEqual((summary["candidate_trials"], summary["counted_trials"], summary["excluded_trials"]), (3, 1, 2))
            self.assertEqual(summary["numerical_window_replay"], "PASS")
            self.assertFalse(summary["recording_coverage_source_verified"])
            self.assertFalse(summary["dataset_ready"])
            self.assertEqual(summary["raw_unit_count"], 7)
            trials = pd.read_csv(root / "output/trials.csv")
            self.assertIn("invalid_time", trials.iloc[1].window_exclusion_reasons)
            self.assertIn("exceeds_trial", trials.iloc[2].window_exclusion_reasons)
            self.assertFalse(trials.scientific_window_eligible.any())
            with h5py.File(root / "output/spike_windows.h5") as saved:
                self.assertEqual(saved["counts"][:].sum(), 4)
                self.assertEqual(list(saved["trial_ids"].asstr()[:]), ["a"])
                np.testing.assert_array_equal(saved["unit_ids"][:], [11, 3])
                self.assertEqual(saved["stimulus_features"].shape, (1, 40, 2))
                np.testing.assert_array_equal(saved["stimulus_features"][0, :20, 0], np.ones(20))
                np.testing.assert_array_equal(saved["stimulus_features"][0, 20:, 1], np.ones(20))
            with self.assertRaises(FileExistsError):
                check_recording_windows(root / "source", root / "output", load_config(CONFIG_PATH), ["test"])

    def test_protected_split_rejects_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_fixture(root / "source", split="test")
            with self.assertRaisesRegex(ValueError, "TRAINING_ANCHOR"):
                check_recording_windows(root / "source", root / "output", load_config(CONFIG_PATH), ["test"])
            self.assertFalse((root / "output").exists())

    def test_altered_source_is_a_cli_failure_not_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source_fixture(root / "source")
            with (root / "source/trials.csv").open("a") as stream:
                stream.write("\n")
            run = subprocess.run([sys.executable, "-B", "-m", "Libs.population_dynamics",
                                  "--config", str(CONFIG_PATH), "window-check", "--recording-dir", str(root / "source"),
                                  "--output-dir", str(root / "output")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn("SOURCE_TABLE_DIGEST_MISMATCH", run.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()

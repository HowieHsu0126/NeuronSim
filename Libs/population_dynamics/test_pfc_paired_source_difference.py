"""Meaningful event and source-binding regressions for the paired diagnostic."""
import json
import unittest

import numpy as np

from . import pfc_paired_source_difference as difference
from . import test_pfc_source_scale_analysis as fixtures
from .pfc_scale_analysis import analyze_source_runs


def raw(events):
    values = np.asarray(events, dtype=np.int64).reshape(-1, 2)
    return dict(spike_ticks=values[:, 0], spike_unit_indices=values[:, 1])


class PairedSourceDifferenceTests(unittest.TestCase):
    def test_equal_scalar_or_unit_totals_can_hide_event_differences(self):
        for left, right, same_counts, changed_units in [
            ([(2, 0), (3, 1)], [(2, 0), (3, 2)], False, 2),
            ([(1, 0), (4, 1)], [(2, 0), (5, 1)], True, 0),
        ]:
            with self.subTest(left=left, right=right):
                result = difference.compare_events(raw(left), raw(right), 3, 0, 10)
                self.assertFalse(result["late_event_sequence_equal"])
                self.assertEqual(result["late_unit_counts_equal"], same_counts)
                self.assertEqual(result["late_units_with_differing_counts"], changed_units)

    def test_within_tick_order_ignored_but_multiplicity_preserved(self):
        left = raw([(5, 0), (5, 1), (5, 1)])
        right = raw([(5, 1), (5, 0), (5, 1)])
        result = difference.compare_events(left, right, 2, 0, 10)
        self.assertTrue(result["late_event_sequence_equal"])
        self.assertIsNone(result["last_differing_tick"])
        result = difference.compare_events(left, raw([(5, 0), (5, 1)]), 2, 0, 10)
        self.assertEqual(result["whole_episode_left_only_events"], 1)
        self.assertEqual(result["whole_episode_right_only_events"], 0)
        self.assertEqual(result["last_differing_tick"], 5)
        self.assertFalse(result["late_event_sequence_equal"])

    def test_half_open_window_and_whole_episode_difference(self):
        result = difference.compare_events(raw([(9, 0), (10, 1), (20, 2)]),
                                           raw([(8, 2), (10, 1), (20, 0)]), 3, 10, 20)
        self.assertTrue(result["late_event_sequence_equal"])
        self.assertEqual(result["last_differing_tick"], 20)
        self.assertFalse(difference.compare_events(raw([(10, 1)]), raw([(10, 2)]),
                                                   3, 10, 20)["late_event_sequence_equal"])
        self.assertTrue(difference.compare_events(raw([]), raw([]), 3, 10, 20)["late_unit_counts_equal"])

    def test_malformed_events_rejected_even_outside_window(self):
        invalid = [([1, 2], [0]), ([[1]], [0]), ([1.0], [0]), ([1], [.5]),
                   ([-1], [0]), ([1], [-1]), ([1], [3]), ([2, 1], [0, 1]),
                   ([True], [0]), (np.array([2**63], dtype=np.uint64), [0])]
        for ticks, units in invalid:
            bad = dict(spike_ticks=np.asarray(ticks), spike_unit_indices=np.asarray(units))
            for left, right in ((bad, raw([(6, 0)])), (raw([(6, 0)]), bad)):
                with self.subTest(ticks=ticks, units=units), self.assertRaises(ValueError):
                    difference.compare_events(left, right, 3, 5, 10)

    def test_terminal_roles_and_exact_large_integer_differences(self):
        left = {"E__v": np.array([-.06]), "E__not_refractory": np.array([False]),
                "E__spike_counts": np.array([2**63], dtype=np.uint64),
                "E__signed": np.array([-2**63], dtype=np.int64),
                "E__source_indices": np.array([0]), "E__archive_indices": np.array([0])}
        right = {key: value.copy() for key, value in left.items()}
        for key, value in [("E__v", -.04), ("E__not_refractory", True),
                           ("E__spike_counts", 2**63 + 1), ("E__signed", 2**63 - 1)]:
            right[key][0] = value
        result = difference.compare_terminal(left, right)
        self.assertAlmostEqual(result["E__v"]["max_absolute_difference"], .02)
        self.assertEqual(result["E__v"]["role"], "dynamic_state")
        self.assertEqual(result["E__not_refractory"]["differing_values"], 1)
        self.assertIsNone(result["E__not_refractory"]["max_absolute_difference"])
        self.assertEqual(result["E__spike_counts"]["max_absolute_difference"], 1)
        self.assertEqual(result["E__spike_counts"]["role"], "cumulative_spike_counts")
        self.assertEqual(result["E__signed"]["max_absolute_difference"], 2**64-1)
        self.assertEqual(result["E__source_indices"]["role"], "source_indices")

    def test_terminal_rejects_mismatches_nonfinite_values_and_overflow(self):
        for a, b in [(np.array([0.]), np.zeros(2)),
                     (np.array([0.]), np.array([0.], dtype=np.float32)),
                     (np.array([np.nan]), np.array([np.nan])),
                     (np.array([1e308]), np.array([-1e308]))]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                difference.compare_terminal({"E__v": a}, {"E__v": b})
        with self.assertRaises(ValueError):
            difference.compare_terminal({"E__v": np.array([0.])}, {})


class PairedSourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceReadoutTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.sources = [self.fixture.add_current(self.fixture.archive(case_id))
                        for case_id in self.fixture.registered]
        self.reference = self.root / "reference"
        analyze_source_runs(self.sources, self.fixture.config_path, self.reference / "analysis")
        names = ["analysis/" + name for name in
                 ("analysis.json", "readout_replay.json", "responses.csv", "phase_unit_counts.npz")]
        self.fixture.write_json(self.reference / "delivery.json",
            dict(status="ARCHIVED_PFC_SOURCE_READOUTS", exit_code=0,
                 files=[self.fixture.file_row(self.reference, name) for name in names]))
        self.selected = [path for path in self.sources if "strong_cue" not in path.parent.name]

    def test_real_cli_preserves_six_cases_and_separate_state_roles(self):
        output = self.root / "diagnostic"
        args = ["--config", str(self.fixture.config_path), "--reference-readouts", str(self.reference),
                "--output", str(output)]
        for source in self.selected:
            args += ["--source-run", str(source)]
        self.assertEqual(difference.main(args), 0)
        result = json.loads((output / "report.json").read_text())
        self.assertEqual(result["diagnosed_source_cases"], 6)
        self.assertEqual(result["registered_source_cases"], 9)
        self.assertEqual(result["late_unit_count_values_verified"], 120)
        self.assertEqual(result["independently_recomputed_scalar_values"], 90)
        self.assertFalse(result["scientific_fidelity_qualified"])
        self.assertEqual(result["production_episodes"], 0)
        for pair in result["pairs"]:
            self.assertFalse(pair["events"]["late_event_sequence_equal"])
            self.assertEqual(pair["terminal"]["E__v"]["differing_values"], 0)
            self.assertGreater(pair["terminal"]["E__spike_counts"]["differing_values"], 0)
            self.assertTrue(all(pair["pairing"].values()))

    def test_missing_case_or_duplicate_is_rejected_with_failure_artifact(self):
        for index, sources in enumerate([self.selected[:-1], self.selected + self.selected[:1]]):
            output = self.root / str(index)
            with self.assertRaises(ValueError):
                difference.run(sources, self.fixture.config_path, self.reference, output)
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "FAILED")

    def test_changed_raw_source_binding_and_changed_reference_are_rejected(self):
        source = self.selected[0]
        with (source / "raw.npz").open("ab") as stream:
            stream.write(b"changed")
        self.fixture.rebind(source)
        with self.assertRaisesRegex(ValueError, "source differs"):
            difference.run(self.selected, self.fixture.config_path, self.reference, self.root / "source_changed")
        with (self.reference / "analysis/responses.csv").open("a") as stream:
            stream.write("changed")
        with self.assertRaisesRegex(ValueError, "size or hash mismatch"):
            difference.run(self.selected, self.fixture.config_path, self.reference, self.root / "readout_changed")

    def test_matching_delivery_cannot_hide_changed_initial_pair(self):
        base = dict(prefix_ticks=np.array([1]), prefix_ids=np.array([0]), initial=np.array([-.06]),
                    final_rng={"key": np.array([1])}, inputs_used={"ticks": 10})
        changed = dict(base, initial=np.array([-.05]))
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            difference.compare_input_pair(base, changed)

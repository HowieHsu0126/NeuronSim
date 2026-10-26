"""Numerical and archive regressions for all-unit fluctuation calibration."""
import json
import unittest

import numpy as np
import yaml

from . import pfc_fluctuation_analysis as analysis
from . import test_pfc_source_scale_analysis as fixtures
from .pfc_scale_analysis import analyze_source_runs


def summarize(counts, mapping, groups):
    means, variances = analysis.unit_moments(counts)
    field_count = int(mapping.max())+1
    fields = np.stack([counts[:, :, mapping == group].sum(axis=2)
                       for group in range(field_count)], axis=2)
    covariance = analysis.field_covariance(fields)
    return analysis.population_summary(means, variances, fields, covariance, mapping, groups)


class FluctuationMomentTests(unittest.TestCase):
    def test_identical_transient_blocks_have_zero_ensemble_variance(self):
        transient = np.arange(12, dtype=np.int64).reshape(6, 2)
        for offset in (0, 100):
            counts = np.repeat((transient+offset)[None], 3, axis=0)
            mean, variance = analysis.unit_moments(counts)
            np.testing.assert_array_equal(variance, np.zeros(2))
            self.assertGreater(np.var(counts[0, :, 0]), 0)
            np.testing.assert_array_equal(analysis.field_covariance(counts), np.zeros((2, 2)))
            row = summarize(counts, np.array([0, 1]), [0, 1])
            self.assertEqual(row["ordered_offdiagonal_contribution_to_population_variance"], 0)
            self.assertEqual(row["population_mean_count_variance"], 0)

    def test_positive_and_negative_shared_activity_preserve_sign_and_normalization(self):
        first = np.arange(3, dtype=np.int64)
        for second, covariance, total in ((first, .5, 1.), (first[::-1], -.5, 0.)):
            counts = np.stack((first, second), axis=1)[:, None, :]
            row = summarize(counts, np.array([0, 1]), [0, 1])
            self.assertEqual(row["diagonal_contribution_to_population_variance"], .5)
            self.assertEqual(row["ordered_offdiagonal_contribution_to_population_variance"], covariance)
            self.assertEqual(row["population_mean_count_variance"], total)

    def test_unequal_fields_silent_units_and_permutation_preserve_population_moment(self):
        counts = np.array([[[0, 1, 0, 0]], [[1, 2, 0, 2]], [[2, 3, 0, 4]]], dtype=np.int64)
        mapping = np.array([0, 0, 0, 1])
        row = summarize(counts, mapping, [0, 1])
        self.assertEqual(row["units"], 4)
        self.assertEqual(row["silent_units"], 1)
        self.assertEqual(row["population_mean_count_variance"], 1.)
        order = [3, 1, 2, 0]
        self.assertEqual(row, summarize(counts[:, :, order], mapping[order], [0, 1]))
        silent = summarize(np.zeros_like(counts), mapping, [0, 1])
        self.assertIsNone(silent["pooled_single_unit_ensemble_fano"])

    def test_chunked_event_counts_keep_boundaries_duplicates_and_silent_units(self):
        ticks = np.array([0, 4, 5, 5, 14], dtype=np.int64)
        ids = np.array([2, 0, 0, 0, 3], dtype=np.int64)
        events = analysis.prepared_events(dict(spike_ticks=ticks, spike_unit_indices=ids), 4, 5, 3)
        pieces = list(analysis.unit_chunks([events, events], 4, 3, chunk_units=3))
        got = np.concatenate([x for _, _, x in pieces], axis=2)
        expected = np.array([[1, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(got, np.stack([expected, expected]))
        bad = dict(spike_ticks=np.array([15]), spike_unit_indices=np.array([0]))
        with self.assertRaisesRegex(ValueError, "time support"):
            analysis.prepared_events(bad, 4, 5, 3)

    def test_fractional_counts_or_missing_block_rejected(self):
        for counts in (np.ones((1, 4, 2), dtype=int), np.ones((3, 4, 2)), -np.ones((3, 4, 2), dtype=int)):
            with self.assertRaises(ValueError):
                analysis.unit_moments(counts)


class FluctuationArchiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SourceReadoutTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.fixture.config.update(readout_bin_ms=5., excitatory_preference_bins=16)
        self.fixture.config_path.write_text(yaml.safe_dump(self.fixture.config))
        self.sources = [self.fixture.add_current(self.fixture.archive(case_id))
                        for case_id in self.fixture.registered]
        self.reference = self.root/"reference"
        analyze_source_runs(self.sources, self.fixture.config_path, self.reference/"analysis")
        names = ["analysis/"+name for name in
                 ("analysis.json", "readout_replay.json", "responses.csv", "phase_unit_counts.npz")]
        self.fixture.write_json(self.reference/"delivery.json",
            dict(status="ARCHIVED_PFC_SOURCE_READOUTS", exit_code=0,
                 files=[self.fixture.file_row(self.reference, name) for name in names]))

    def test_real_cli_preserves_every_case_unit_and_field(self):
        output = self.root/"fluctuations"
        args = ["--config", str(self.fixture.config_path), "--reference-readouts", str(self.reference),
                "--output", str(output)]
        for source in self.sources:
            args += ["--source-run", str(source)]
        protocol = self.root/'measurements.yaml'
        protocol.write_text(yaml.safe_dump(dict(scope='development_calibration', measurements=dict(
            observation_bin_ms=5., phases=list(analysis.PHASES), all_unit_count_windows_ms=[5,50,100]))))
        args += ['--measurement-protocol', str(protocol)]
        self.assertEqual(analysis.main(args), 0)
        result = json.loads((output/"report.json").read_text())
        self.assertEqual(result['unit_count_histograms']['unit_window_observations'], 103500)
        self.assertEqual(result['unit_count_histograms']['blocks'], 3)
        with np.load(output/'unit_count_histograms.npz') as stored:
            self.assertEqual(stored['expected_observations'].shape, (3,3,5,3,17))
            self.assertEqual(int(stored['observations'].sum()), 103500)
            np.testing.assert_array_equal(stored['source_rng_seeds'], self.fixture.seeds)
        self.assertEqual(result["completed_source_cases"], 9)
        self.assertEqual(result["unit_phase_count_values_verified"], 9*5*20)
        self.assertEqual(result["field_bin_count_values_verified"], 9*1000*17)
        self.assertEqual(len(result["population_summaries"]), 45)
        self.assertFalse(result["finite_size_scaling_qualified"])
        self.assertFalse(result["confidence_intervals"])
        with np.load(output/"unit_moments.npz") as values:
            np.testing.assert_array_equal(values["unit_ids"], np.arange(20))
            self.assertEqual(values["cue_mean_counts"].shape, (5, 20))
            np.testing.assert_array_equal(values["cue_variance_counts"], np.zeros((5, 20)))

    def test_missing_source_block_rejected_without_partial_success(self):
        output = self.root/"missing"
        with self.assertRaisesRegex(ValueError, "all nine"):
            analysis.run(self.sources[:-1], self.fixture.config_path, self.reference, output)
        result = json.loads((output/"report.json").read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["production_episodes"], 0)

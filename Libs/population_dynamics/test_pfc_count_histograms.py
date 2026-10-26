"""Exact count probabilities, silent-unit mass and temporal aggregation tests."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import pfc_count_histograms as counts


class CountHistogramTests(unittest.TestCase):
    def test_unequal_fields_silent_units_and_window_aggregation(self):
        x = np.array([[[0,0,0],[1,0,1],[0,0,0],[2,0,0]],
                      [[1,0,0],[0,0,0],[1,0,2],[0,0,0]]], dtype=np.int64)
        fields = np.array([0,0,1])
        histogram = {}
        counts.accumulate(histogram, x, fields, 2, condition=0, phase=0, window=0)
        expected = {(0,0,0,0,0,0):2, (0,0,0,0,0,1):1, (0,0,0,0,0,2):1,
                    (0,0,0,0,1,0):1, (0,0,0,0,1,1):1,
                    (0,1,0,0,0,0):2, (0,1,0,0,0,1):2,
                    (0,1,0,0,1,0):1, (0,1,0,0,1,2):1}
        self.assertEqual(histogram, expected)
        chunked = {}
        for indices in ([2], [1,0]):
            counts.accumulate(chunked, x[:,:,indices], fields[indices], 2,
                              condition=0, phase=0, window=0)
        self.assertEqual(chunked, expected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'histogram.npz'
            report = counts.write(path, histogram, conditions=['cue'], seeds=[1,2],
                phases={'delay':[0.,20.]}, windows_ms=[10.], field_sizes=np.array([2,1]))
            self.assertEqual(report['unit_window_observations'], 12)
            self.assertFalse(report['observations_are_independent_replicates'])
            with np.load(path, allow_pickle=False) as stored:
                cdf = counts.cdf(stored['coordinates'], stored['observations'], np.array([0,1,2]),
                                 stored['expected_observations'].shape)
                np.testing.assert_array_equal(cdf[0,0,0,0,0], [.5,.75,1.])
                np.testing.assert_array_equal(cdf[0,1,0,0,0], [.5,1.,1.])

    def test_equal_mean_counts_do_not_hide_distribution_or_block_differences(self):
        x = np.array([[[2],[0],[0],[0]], [[1],[1],[0],[0]]])
        self.assertEqual(x[0].mean(), x[1].mean())
        histogram = {}
        counts.accumulate(histogram, x, np.array([0]), 1, condition=0, phase=0, window=0)
        keys = np.asarray(sorted(histogram))
        mass = np.asarray([histogram[tuple(key)] for key in keys])
        cdf = counts.cdf(keys, mass, np.array([0,1,2]), (1,2,1,1,1))
        np.testing.assert_array_equal(cdf[0,0,0,0,0], [.75,.75,1.])
        np.testing.assert_array_equal(cdf[0,1,0,0,0], [.5,1.,1.])

    def test_large_integer_counts_are_retained_without_dense_support_or_clipping(self):
        histogram = {}
        counts.accumulate(histogram, np.array([[[2**40,0]]]), np.array([0,0]), 1,
                          condition=0, phase=0, window=0)
        self.assertEqual(histogram, {(0,0,0,0,0,0):1, (0,0,0,0,0,2**40):1})

    def test_missing_or_duplicated_unit_window_mass_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for amount in (1,3):
                with self.subTest(amount=amount), self.assertRaisesRegex(ValueError, 'histogram mass'):
                    counts.write(Path(temporary)/str(amount), {(0,0,0,0,0,0):amount},
                        conditions=['cue'], seeds=[1], phases={'phase':[0.,10.]},
                        windows_ms=[5.], field_sizes=np.array([1]))

    def test_invalid_counts_windows_or_identity_rejected(self):
        good = np.ones((2,4,3), dtype=np.int64); fields = np.array([0,0,1])
        for x, groups, width in ((good.astype(float),fields,2), (-good,fields,2),
            (good,fields,3), (good,fields,0), (good,fields.astype(float),2),
            (np.full_like(good,np.iinfo(np.int64).max),fields,2)):
            with self.subTest(width=width), self.assertRaises(ValueError):
                counts.accumulate({}, x, groups, width, condition=0, phase=0, window=0)


if __name__ == '__main__':
    unittest.main()

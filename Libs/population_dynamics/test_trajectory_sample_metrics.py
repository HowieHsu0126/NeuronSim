import itertools
from fractions import Fraction
import unittest

import numpy as np
import torch

from .trajectory_sample_metrics import corrected_count_mse, unit_count_metrics


class SampleMetricTests(unittest.TestCase):
    def test_integer_correction_matches_independent_ordered_pair_formula(self):
        rng = np.random.default_rng(904)
        values = rng.integers(0, 5000, size=(16, 5, 13), dtype=np.int64)
        truth = rng.integers(0, 5000, size=(5, 13), dtype=np.int64)
        numerator = sum((int(values[i,t,u])-int(truth[t,u]))*(int(values[j,t,u])-int(truth[t,u]))
                        for t in range(5) for u in range(13) for i in range(16) for j in range(16) if i != j)
        expected = float(np.float32(float(Fraction(numerator,16*15*5*13))))
        for chunk in (1, 3, 13):
            self.assertEqual(corrected_count_mse(torch.from_numpy(values),torch.from_numpy(truth),
                                                chunk_units=chunk),expected)

    def test_sparse_ensemble_cancellation_is_exact_at_zero(self):
        values = torch.zeros((16, 3, 5),dtype=torch.int64)
        values[0] = torch.tensor([[10001,99999,3,123456,17]]*3)
        # Only one nonzero residual: every ordered distinct-draw product is0.
        self.assertEqual(corrected_count_mse(values,torch.zeros((3,5))),0.0)

    def test_extreme_integer_cells_use_exact_arbitrary_precision(self):
        values = torch.tensor([2**40,0],dtype=torch.int64).reshape(2,1,1)
        self.assertEqual(corrected_count_mse(values,torch.zeros((1,1))),0.0)
        values.fill_(2**40)
        self.assertEqual(corrected_count_mse(values,torch.zeros((1,1))),float(2**80))
        with self.assertRaisesRegex(ValueError,'positive integer metric chunk'):
            corrected_count_mse(values,torch.zeros((1,1)),chunk_units=0)
        with self.assertRaisesRegex(ValueError,'observed counts exceed exact int64'):
            corrected_count_mse(torch.zeros((2,1,1),dtype=torch.int64),
                                torch.tensor([[float(2**63)]],dtype=torch.float64))

    def test_block_partition_and_extreme_cell_preserve_every_pair(self):
        values=np.array([[0,2**40,4000000,2],[2,3,4000001,1],[1,5,0,4]],dtype=np.int64).reshape(3,2,2)
        truth=np.array([[1,7],[4000000,0]],dtype=np.int64)
        numerator=sum((int(values[i,t,u])-int(truth[t,u]))*(int(values[j,t,u])-int(truth[t,u]))
                      for t in range(2) for u in range(2) for i in range(3) for j in range(3) if i != j)
        expected=float(np.float32(float(Fraction(numerator,3*2*2*2))))
        for chunk in (1,2,4096):
            self.assertEqual(corrected_count_mse(torch.from_numpy(values),torch.from_numpy(truth),chunk_units=chunk),expected)

    def test_large_common_offset_and_int32_products_keep_the_same_estimand(self):
        for offset in (0,2**40):
            values=torch.tensor([offset+50000,offset+2,offset],dtype=torch.int64).reshape(3,1,1)
            truth=torch.tensor([[offset+1]],dtype=torch.int64)
            expected=float(np.float32((49999*1+49999*(-1)+1*(-1))/3))
            self.assertEqual(corrected_count_mse(values,truth),expected)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for integer reduction parity')
    def test_cuda_exact_reduction_handles_full_nonnegative_int64_domain(self):
        limit = torch.iinfo(torch.int64).max
        values = torch.tensor([[0, limit, 4000000, 2], [limit, 0, 4000001, 1],
                               [1, limit-1, 0, 4]], dtype=torch.int64).reshape(3, 2, 2)
        truth = torch.tensor([[limit, 0], [4000000, 0]], dtype=torch.int64)
        numerator = sum((int(values[i,t,u])-int(truth[t,u])) *
                        (int(values[j,t,u])-int(truth[t,u]))
                        for t in range(2) for u in range(2)
                        for i in range(3) for j in range(3) if i != j)
        expected = float(np.float32(float(Fraction(numerator, 3*2*2*2))))
        for device in ('cpu', 'cuda'):
            for chunk in (1, 2):
                self.assertEqual(corrected_count_mse(values.to(device), truth.to(device),
                                                     chunk_units=chunk), expected)

    def test_mean_error_correction_is_unbiased_over_bernoulli_ensembles(self):
        probability = .2
        expectation = 0.
        for values in itertools.product((0, 1), repeat=3):
            samples = torch.tensor(values).reshape(3, 1, 1)
            result = unit_count_metrics(samples, torch.zeros(1, 1), [1], 1, [1])
            weight = probability**sum(values) * (1-probability)**(3-sum(values))
            expectation += weight * result["unit_bin_mean_mse_corrected"]
        self.assertAlmostEqual(expectation, probability**2, places=7)
        negative = unit_count_metrics(torch.tensor([0, 2]).reshape(2, 1, 1), torch.ones(1, 1), [1], 3, [1])
        self.assertLess(negative["unit_bin_mean_mse_corrected"], 0)

    def test_adjacent_bins_reveal_infeasible_counts_even_when_each_bin_is_legal(self):
        samples = torch.tensor([[[2], [2]], [[1], [1]]])
        result = unit_count_metrics(samples, torch.ones(2, 1), [4], 5, [1, 2, 4])
        self.assertEqual(result["refractory_count_violation_1bin"], 0.)
        self.assertEqual(result["refractory_count_violation_2bin"], .5)
        self.assertIsNone(result["refractory_count_violation_4bin"])

    def test_interval_and_mean_use_complete_identified_unit_totals(self):
        samples = torch.tensor([[[0, 2], [0, 1]], [[2, 0], [1, 0]], [[1, 1], [1, 1]], [[3, 1], [1, 1]]])
        actual = torch.tensor([[0., 1.], [0., 1.]])
        result = unit_count_metrics(samples, actual, [1, 1], 5, [1])
        totals = samples.numpy().sum(1)
        low, high = np.quantile(totals, [.05, .95], axis=0)
        expected_coverage = np.mean((actual.numpy().sum(0) >= low) & (actual.numpy().sum(0) <= high))
        self.assertEqual(result["unit_total_coverage_90"], expected_coverage)
        self.assertAlmostEqual(result["unit_total_interval_width_90"], np.mean(high-low), places=6)
        self.assertAlmostEqual(result["sampled_expected_spikes_per_unit"], totals.mean(), places=6)

    def test_fractional_observations_and_invalid_identity_are_rejected(self):
        for actual, interval in [(torch.tensor([[.5]]), [1]), (torch.zeros(1, 1), [0]), (torch.zeros(1, 1), [1, 2])]:
            with self.assertRaises(ValueError):
                unit_count_metrics(torch.zeros(2, 1, 1, dtype=torch.int64), actual, interval, 5, [1])

    def test_total_count_metrics_reject_potential_fixed_width_overflow(self):
        limit = torch.iinfo(torch.int64).max
        for samples, actual in [
                (torch.full((2, 2, 1), limit), torch.zeros(2, 1)),
                (torch.zeros(2, 2, 1, dtype=torch.int64), torch.full((2, 1), limit))]:
            with self.assertRaisesRegex(ValueError, 'trajectory totals may exceed exact int64'):
                unit_count_metrics(samples, actual, [1], 5, [1, 2])


if __name__ == "__main__":
    unittest.main()

"""Exact metric and all-seed eligibility checks for fitting rollout qualification."""
import copy
import unittest

import torch

from population_dynamics.population_residual_rollout_qualification import (
    CANDIDATES, HORIZONS_MS, ORIGIN_COUNT, SEEDS,
    aggregate_candidates, posterior_coordinate_energy_score,
)


class RolloutQualificationTests(unittest.TestCase):
    def test_unbiased_pair_term_and_trajectory_dimension_normalization(self):
        # Two draws 0 and4, target-1, scale2: target distances .5 and2.5;
        # unordered pair distance2. Thus1.5 - half(2) = .5 at either time point.
        samples = torch.tensor([[[0.], [0.]], [[4.], [4.]]], dtype=torch.float64)
        result = posterior_coordinate_energy_score(samples, torch.tensor([[-1.], [-1.]]), [2.])
        self.assertAlmostEqual(result["mean_target_distance"], 1.5, places=14)
        self.assertAlmostEqual(result["half_mean_offdiagonal_pair_distance"], 1., places=14)
        self.assertAlmostEqual(result["posterior_coordinate_energy_score"], .5, places=14)
        # Including diagonal pairs would instead produce1.0, so this checks the
        # finite-ensemble unbiased term independently of its implementation.

    def test_coordinate_unit_changes_cancel_and_invalid_scales_reject(self):
        samples = torch.tensor([[[0., 1.], [2., 3.]], [[3., 2.], [1., 0.]], [[1., 1.], [1., 1.]]], dtype=torch.float64)
        target, scale = torch.zeros((2, 2), dtype=torch.float64), torch.tensor([2., 4.], dtype=torch.float64)
        original = posterior_coordinate_energy_score(samples, target, scale)
        units = torch.tensor([7., .25], dtype=torch.float64)
        changed = posterior_coordinate_energy_score(samples*units, target*units, scale*units)
        for key in original:
            self.assertAlmostEqual(original[key], changed[key], places=14)
        with self.assertRaises(ValueError):
            posterior_coordinate_energy_score(samples, target, [0., 4.])
        with self.assertRaises(ValueError):
            posterior_coordinate_energy_score(samples[:1], target, scale)

    def _records(self):
        rows = []
        for seed in SEEDS:
            for candidate, mode, alpha in CANDIDATES:
                if mode == "parent":
                    score = 2.
                elif mode == "current":
                    score = {.0001: 3., .01: 3., 1.: 4., 100.: 5.}[alpha]
                else:
                    score = {.0001: 5., .01: 1., 1.: 2., 100.: 2.}[alpha]
                rows.append(dict(seed=seed, candidate=candidate, mode=mode, alpha=alpha,
                    status="COMPLETE", completed_origins=ORIGIN_COUNT,
                    scored_origin_counts={str(h): ORIGIN_COUNT for h in HORIZONS_MS},
                    mean_posterior_coordinate_energy_score={str(h): score for h in HORIZONS_MS}))
        return rows

    def test_failed_seed_excludes_candidate_and_exact_ties_choose_smaller_alpha(self):
        rows = self._records()
        failed = next(row for row in rows if row["seed"] == 43 and row["mode"] == "history" and row["alpha"] == .01)
        failed.update(status="FAILED", completed_origins=0)
        result = aggregate_candidates(rows)
        self.assertEqual(result["selected_alphas"]["current"]["alpha"], .0001)
        self.assertEqual(result["selected_alphas"]["history"]["alpha"], 1.)
        excluded = next(row for row in result["candidate_summaries"] if row["mode"] == "history" and row["alpha"] == .01)
        self.assertFalse(excluded["eligible"])
        self.assertIsNone(excluded["mean_posterior_coordinate_energy_score"]["500"])
        self.assertEqual(result["selected_alphas"]["history"]["delta_500ms_vs_parent"], 0.)

    def test_no_eligible_mode_and_missing_candidate_are_explicit(self):
        rows = self._records()
        for row in rows:
            if row["mode"] == "history":
                row.update(status="FAILED", completed_origins=0)
        result = aggregate_candidates(rows)
        self.assertEqual(result["selected_alphas"]["history"]["status"], "NO_ELIGIBLE_CANDIDATE")
        self.assertIsNone(result["selected_alphas"]["history"]["alpha"])
        with self.assertRaises(ValueError):
            aggregate_candidates(rows[:-1])
        duplicate = copy.deepcopy(rows)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaises(ValueError):
            aggregate_candidates(duplicate)


if __name__ == "__main__":
    unittest.main()

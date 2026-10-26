"""Focused causal/statistical contracts for the development field-history probe."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from . import field_history_probe as probe
from .conditional_regression import conditional_prediction
from .trajectory_scores import variogram_score


def configuration():
    path = Path(__file__).resolve().parents[1] / 'configs/neural_population_field_history_probe.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def regression_fixture():
    config = configuration()
    config.update(fit_blocks=[0, 1, 2, 3], selection_blocks=[4, 5],
                  regularization=[0.01, 0.1, 1.0])
    blocks = np.repeat(np.arange(6), 2)
    cells = np.tile(np.array([0, 1]), 6)
    recent = np.column_stack([blocks, cells, (blocks + 1) * (cells + 1) + 0.1 * blocks**2]).astype(float)
    history = np.column_stack([recent, np.sin(blocks + cells), blocks**2, cells * (blocks + 1)])
    targets = np.column_stack([
        1 + 0.4 * blocks + 0.7 * cells,
        2 + 0.2 * blocks**2 - 0.1 * cells,
        0.3 + np.abs(np.sin(blocks + cells)),
    ])
    arrays = dict(recent=recent, recent_and_history=history,
                  targets=targets, blocks=blocks, cells=cells)
    slices = [
        dict(family='mean_rate', horizon_ms=50, start=0, stop=2),
        dict(family='spatial_variogram', horizon_ms=50, start=2, stop=3),
    ]
    return arrays, config, slices


class FieldHistoryWindowTests(unittest.TestCase):
    def test_history_is_half_open_and_ignores_all_future_bins(self):
        config = dict(recent_ms=10, history_ms=20, history_bin_ms=10)
        rates = np.column_stack([2 * np.arange(8), 10 + 2 * np.arange(8)]).astype(float)
        result = probe.window_features(rates, origin_ms=20, bin_ms=5, config=config)
        np.testing.assert_array_equal(result['recent'], [5, 15])
        np.testing.assert_array_equal(result['recent_and_history'], [5, 15, 1, 11, 5, 15])
        changed = rates.copy()
        changed[4:] = 1e9
        repeated = probe.window_features(changed, origin_ms=20, bin_ms=5, config=config)
        for name in result:
            np.testing.assert_array_equal(result[name], repeated[name])

    def test_future_mean_and_p1_variograms_match_hand_calculation(self):
        # Future bins are exactly rows 2, 3, 4. Outside rows must not contribute.
        rates = np.array([
            [1000, 2000, 3000, 4000], [9000, 8000, 7000, 6000],
            [1, 4, 2, 8], [3, 3, 7, 4], [8, 2, 5, 1],
            [900, 800, 700, 600],
        ], dtype=float)
        pairs = dict(spatial=np.array([[0, 1], [2, 3]]),
                     cross_population=np.array([[0, 2], [1, 3]]))
        result = probe.future_statistics(rates, origin_ms=10, horizon_ms=15,
                                         bin_ms=5, pairs=pairs, config=dict(temporal_lag_ms=5))
        expected = {
            'mean_rate': [4, 3, 14 / 3, 13 / 3],
            'spatial_variogram': [3, 13 / 3],
            'cross_population_variogram': [8 / 3, 2],
            'temporal_variogram': [3.5, 1, 3.5, 3.5],
        }
        self.assertEqual(set(result), set(expected))
        for name, values in expected.items():
            np.testing.assert_allclose(result[name], values, rtol=0, atol=1e-14)
        changed = rates.copy()
        changed[:2] = -1e10
        changed[5:] = 1e10
        repeated = probe.future_statistics(changed, 10, 15, 5, pairs, dict(temporal_lag_ms=5))
        for name in result:
            np.testing.assert_array_equal(result[name], repeated[name])

    def test_incomplete_or_misaligned_support_is_rejected(self):
        rates = np.zeros((8, 2))
        config = dict(recent_ms=10, history_ms=20, history_bin_ms=10)
        with self.assertRaisesRegex(ValueError, 'complete aligned causal'):
            probe.window_features(rates, 15, 5, config)
        with self.assertRaisesRegex(ValueError, 'align with source bins'):
            probe.window_features(rates, 22, 5, config)
        with self.assertRaisesRegex(ValueError, 'complete aligned causal'):
            probe.window_features(rates, 45, 5, config)
        pairs = dict(spatial=np.array([[0, 1]]), cross_population=np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, 'complete future support'):
            probe.future_statistics(rates, 30, 15, 5, pairs, dict(temporal_lag_ms=5))
        with self.assertRaisesRegex(ValueError, 'shorter temporal lag'):
            probe.future_statistics(rates, 20, 10, 5, pairs, dict(temporal_lag_ms=10))

    def test_finite_ensemble_p1_score_preserves_signed_correction(self):
        # Observed |0-1|=1; draws have magnitudes 0 and 2. Their unbiased
        # variance is 2, so the corrected score is 0 - 2/2 = -1.
        samples = np.array([[[0.0, 0.0]], [[0.0, 2.0]]])
        observed = np.array([[0.0, 1.0]])
        self.assertEqual(variogram_score(samples, observed, np.array([0]), np.array([1])), -1.0)


class FieldHistorySelectionTests(unittest.TestCase):
    def test_changing_selection_targets_cannot_change_predictions_or_penalties(self):
        arrays, config, slices = regression_fixture()
        original = probe.fit_predictions(arrays, config, slices)
        changed = {key: value.copy() for key, value in arrays.items()}
        selection = np.isin(arrays['blocks'], config['selection_blocks'])
        changed['targets'][selection] = np.arange(int(selection.sum()) * 3).reshape(-1, 3) * 1e6 - 4e6
        repeated = probe.fit_predictions(changed, config, slices)
        np.testing.assert_array_equal(original[0], repeated[0])
        self.assertEqual(original[1:4], repeated[1:4])
        np.testing.assert_array_equal(original[4], repeated[4])

    def test_selection_features_do_not_enter_penalty_selection_or_scaling(self):
        arrays, config, slices = regression_fixture()
        original = probe.fit_predictions(arrays, config, slices)
        changed = {key: value.copy() for key, value in arrays.items()}
        selection = np.isin(arrays['blocks'], config['selection_blocks'])
        for name in config['representations']:
            changed[name][selection] = changed[name][selection] * 1000 + 9000
        repeated = probe.fit_predictions(changed, config, slices)
        self.assertEqual(original[1:4], repeated[1:4])
        np.testing.assert_array_equal(original[4], repeated[4])
        self.assertTrue(np.isfinite(repeated[0]).all())

    def test_every_inner_and_outer_fit_respects_whole_blocks(self):
        arrays, config, slices = regression_fixture()
        fitting = set(config['fit_blocks'])
        selection = set(config['selection_blocks'])
        calls = []

        def checked(train_x, train_y, train_cells, test_x, test_cells, kind, alpha):
            # First feature is a hand-declared block identifier in this fixture.
            train_blocks = set(train_x[:, 0].astype(int))
            test_blocks = set(test_x[:, 0].astype(int))
            self.assertFalse(train_blocks & test_blocks)
            self.assertTrue(train_blocks <= fitting)
            if test_blocks <= fitting:
                self.assertEqual(len(test_blocks), 1)
                self.assertEqual(train_blocks, fitting - test_blocks)
                role = 'inner'
            else:
                self.assertEqual(test_blocks, selection)
                self.assertEqual(train_blocks, fitting)
                role = 'outer'
            calls.append((role, tuple(sorted(test_blocks))))
            return conditional_prediction(train_x, train_y, train_cells, test_x, test_cells, kind, alpha)

        with patch.object(probe, 'conditional_prediction', side_effect=checked):
            result = probe.fit_predictions(arrays, config, slices)
        self.assertEqual({blocks[0] for role, blocks in calls if role == 'inner'}, fitting)
        self.assertTrue(any(role == 'outer' for role, _ in calls))
        np.testing.assert_array_equal(result[4], np.isin(arrays['blocks'], config['selection_blocks']))
        self.assertEqual(result[0].shape, (5, 4, 3))

    def test_unknown_or_overlapping_blocks_are_rejected(self):
        arrays, config, slices = regression_fixture()
        changed = deepcopy(config)
        changed['selection_blocks'] = [3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'distinct whole fitting and selection'):
            probe.validate_config(changed)
        changed = {key: value.copy() for key, value in arrays.items()}
        changed['blocks'][-1] = 99
        with self.assertRaisesRegex(ValueError, 'exactly one registered block set'):
            probe.fit_predictions(changed, config, slices)

    def test_cell_constant_features_use_only_fitting_cell_means(self):
        train_x = np.array([[1, 5], [1, 5], [8, 5], [8, 5]], dtype=float)
        train_y = np.array([[2, 4], [6, 8], [10, 14], [14, 18]], dtype=float)
        train_cells = np.array([0, 0, 1, 1])
        test_x = np.array([[1e9, -1e9], [-1e8, 1e8], [0, 0]], dtype=float)
        test_cells = np.array([0, 1, 0])
        for kind in ('cells', 'linear', 'rbf'):
            prediction, diagnostic = conditional_prediction(
                train_x, train_y, train_cells, test_x, test_cells, kind, 0.1)
            np.testing.assert_array_equal(prediction, [[4, 6], [12, 16], [4, 6]])
            self.assertEqual(diagnostic, dict(active_features=0, effective_df=2.0))

    def test_development_only_configuration_and_three_seed_authority(self):
        config = configuration()
        probe.validate_config(config)
        for key, value in (('source_split', 'test'), ('formal_training_seeds', [41, 42]),
                           ('training_rng', 41), ('variogram_power', 2.0)):
            changed = deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                probe.validate_config(changed)


if __name__ == '__main__':
    unittest.main()

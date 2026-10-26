"""Numerical and leakage checks for optional cell-conditioned witnesses."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from . import closure_selection as closure
from .conditional_regression import conditional_prediction


def cell_primal(train_x, train_y, train_cells, test_x, test_cells, alpha):
    """Independent primal ridge with unpenalized fitted cell intercepts."""
    unique = np.unique(train_cells)
    means_x = np.stack([train_x[train_cells == cell].mean(axis=0) for cell in unique])
    means_y = np.stack([train_y[train_cells == cell].mean(axis=0) for cell in unique])
    lookup = {cell: i for i, cell in enumerate(unique)}
    fit_index = np.asarray([lookup[cell] for cell in train_cells])
    test_index = np.asarray([lookup[cell] for cell in test_cells])
    centered = train_x - means_x[fit_index]
    active = np.any(np.stack([np.ptp(train_x[train_cells == cell], axis=0) > 0 for cell in unique]), axis=0)
    scale = centered.std(axis=0)[active]
    x = centered[:, active] / scale
    query = (test_x - means_x[test_index])[:, active] / scale
    residual = train_y - means_y[fit_index]
    weights = np.linalg.solve(x.T @ x + alpha * len(x) * np.eye(x.shape[1]), x.T @ residual)
    return means_y[test_index] + query @ weights


def witness_data(seed=187):
    rng = np.random.default_rng(seed)
    fit_rows, selection_rows = 24, 12
    state = rng.normal(size=(fit_rows + selection_rows, 3))
    history = rng.normal(size=(fit_rows + selection_rows, 2))
    cells = np.tile(np.arange(3), 12)
    controls = np.eye(3)[cells]
    effects = np.array([-4., 2., 8.])[cells, None]
    targets = effects + state[:, :1] + 2 * history[:, :1]
    return dict(
        fit_state=state[:fit_rows], selection_state=state[fit_rows:],
        fit_history=history[:fit_rows], selection_history=history[fit_rows:],
        fit_controls=controls[:fit_rows], selection_controls=controls[fit_rows:],
        fit_targets=targets[:fit_rows], selection_targets=targets[fit_rows:],
        fit_blocks=np.repeat(np.arange(4), 6), selection_blocks=np.repeat([4, 5], 6),
        fit_cells=cells[:fit_rows], selection_cells=cells[fit_rows:],
        penalty=.01, target_names=["controlled_response"],
    )


class ConditionalPredictionTests(unittest.TestCase):
    def test_dual_matches_independent_primal_for_small_and_wide_state(self):
        rng = np.random.default_rng(924)
        cells = np.tile(np.arange(3), 5)
        query_cells = np.tile(np.arange(3), 2)
        for width in (4, 31):
            with self.subTest(features=width):
                train_x = rng.normal(size=(15, width)) + cells[:, None] * 3
                test_x = rng.normal(size=(6, width)) + query_cells[:, None] * 3
                # This column must remain inactive despite arbitrary query values.
                train_x = np.column_stack((train_x, cells * 1.25))
                test_x = np.column_stack((test_x, np.full(6, 1000.)))
                train_y = rng.normal(size=(15, 2)) + cells[:, None] * 7
                predicted, diagnostic = conditional_prediction(
                    train_x, train_y, cells, test_x, query_cells, "linear", .01)
                expected = cell_primal(train_x, train_y, cells, test_x, query_cells, .01)
                np.testing.assert_allclose(predicted, expected, rtol=1e-11, atol=1e-11)
                self.assertEqual(diagnostic["active_features"], width)

    def test_query_rows_do_not_change_fitted_scaling_or_active_features(self):
        train_x = np.array([[0., 2.], [1., 2.], [4., 9.], [6., 9.]])
        train_cells = np.array([0, 0, 1, 1])
        train_y = np.array([[2.], [3.], [10.], [12.]])
        query = np.array([[.5, 999.], [5., -999.]])
        query_cells = np.array([0, 1])
        first, diagnostic = conditional_prediction(train_x, train_y, train_cells, query, query_cells, "linear", .01)
        extreme = np.concatenate((query, [[1.e8, 1.e10]]))
        extended, other = conditional_prediction(train_x, train_y, train_cells, extreme, np.array([0, 1, 0]), "linear", .01)
        np.testing.assert_array_equal(first, extended[:2])
        self.assertEqual(diagnostic, other)
        self.assertEqual(diagnostic["active_features"], 1)

    def test_cell_offsets_are_unpenalized_and_unseen_cells_rejected(self):
        train_cells = np.tile([0, 1], 4)
        train_x = np.column_stack((train_cells, 10 + train_cells))
        train_y = (100 + train_cells * 50)[:, None]
        predicted, diagnostic = conditional_prediction(train_x, train_y, train_cells,
            np.array([[999, -3], [-9, 12]]), np.array([0, 1]), "linear", .01)
        np.testing.assert_array_equal(predicted, [[100.], [150.]])
        self.assertEqual(diagnostic["active_features"], 0)
        with self.assertRaisesRegex(ValueError, "unseen factorial cell"):
            conditional_prediction(train_x, train_y, train_cells, train_x[:1], np.array([2]), "linear", .01)


class OptionalCellWitnessTests(unittest.TestCase):
    def test_original_interaction_witness_keeps_existing_behavior(self):
        rng = np.random.default_rng(410)
        state = rng.normal(size=(48, 1))
        history = rng.normal(size=(48, 1))
        controls = np.tile([0., 1.], 24)[:, None]
        target = state + 3 * history * controls
        arguments = dict(
            fit_state=state[:32], selection_state=state[32:],
            fit_history=history[:32], selection_history=history[32:],
            fit_controls=controls[:32], selection_controls=controls[32:],
            fit_targets=target[:32], selection_targets=target[32:],
            fit_blocks=np.repeat(np.arange(4), 8), selection_blocks=np.repeat([4, 5], 8),
            penalty=.01, target_names=["paired_control_response"],
        )
        result = closure.assess_state(**arguments)
        self.assertEqual(result, closure.assess_state(**arguments, witness_kind="linear"))
        self.assertEqual(result["control_conditioning"], "state_control_interactions")
        scale = target[:32].std(axis=0)
        state_x = closure._conditioned(state[:32], controls[:32])
        state_query = closure._conditioned(state[32:], controls[32:])
        full_x = closure._conditioned(np.concatenate((state[:32], history[:32]), axis=1), controls[:32])
        full_query = closure._conditioned(np.concatenate((state[32:], history[32:]), axis=1), controls[32:])
        state_prediction = closure._ridge_predictions(state_x, target[:32] / scale, state_query, .01)
        full_prediction = closure._ridge_predictions(full_x, target[:32] / scale, full_query, .01)
        truth = target[32:] / scale
        gains = ((state_prediction - truth) ** 2 - (full_prediction - truth) ** 2)
        np.testing.assert_allclose(result["mean_history_gain"], gains.mean(axis=0), rtol=1e-12, atol=1e-12)
        self.assertTrue(result["control_interactions"])
        self.assertFalse(result["full_joint_distribution_certified"])

    def test_optional_cells_use_shared_linear_predictions_and_all_blocks(self):
        arguments = witness_data()
        result = closure.assess_state(**arguments)
        self.assertEqual(result, closure.assess_state(**arguments, witness_kind="linear"))
        self.assertEqual(result["control_conditioning"], "fixed_factorial_cells_shared_linear_state")
        scale = arguments["fit_targets"].std(axis=0)
        state_prediction = cell_primal(arguments["fit_state"], arguments["fit_targets"] / scale,
            arguments["fit_cells"], arguments["selection_state"], arguments["selection_cells"], .01)
        full_prediction = cell_primal(
            np.concatenate((arguments["fit_state"], arguments["fit_history"]), axis=1),
            arguments["fit_targets"] / scale, arguments["fit_cells"],
            np.concatenate((arguments["selection_state"], arguments["selection_history"]), axis=1),
            arguments["selection_cells"], .01)
        truth = arguments["selection_targets"] / scale
        errors = (state_prediction - truth) ** 2
        full_errors = (full_prediction - truth) ** 2
        self.assertEqual([r["block"] for r in result["source_block_rows"]], ["4", "5"])
        for row in result["source_block_rows"]:
            mask = arguments["selection_blocks"] == int(row["block"])
            self.assertEqual(row["rows"], 6)
            np.testing.assert_allclose(row["state_mse"], errors[mask].mean(axis=0), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(row["augmented_mse"], full_errors[mask].mean(axis=0), rtol=1e-11, atol=1e-11)
        self.assertGreater(result["mean_history_gain"][0], 0.)
        self.assertFalse(result["control_interactions"])
        self.assertFalse(result["formal_tolerances_qualified"])
        self.assertFalse(result["confidence_intervals_computed"])

    def test_optional_cells_fit_target_scaling_only_on_fitting_rows(self):
        arguments = witness_data()
        before = closure.assess_state(**arguments)
        changed = arguments.copy()
        changed["selection_targets"] = arguments["selection_targets"] + 10000.
        after = closure.assess_state(**changed)
        np.testing.assert_array_equal(before["training_target_scale"], after["training_target_scale"])
        np.testing.assert_array_equal(before["defined_targets"], after["defined_targets"])
        self.assertNotEqual(before["source_block_rows"], after["source_block_rows"])

    def test_optional_cells_reject_inconsistent_controls_or_partial_cell_input(self):
        for field in ("fit_controls", "selection_controls"):
            with self.subTest(controls=field):
                arguments = witness_data()
                arguments[field] = arguments[field].copy()
                arguments[field][0] = [0., 0., 1.]
                with self.assertRaises(ValueError):
                    closure.assess_state(**arguments)
        arguments = witness_data()
        del arguments["selection_cells"]
        with self.assertRaises(ValueError):
            closure.assess_state(**arguments)

    def test_optional_cells_avoid_expanding_state_control_interactions(self):
        # The complete unit boundary must use cell fixed effects without its
        # 12-fold explicit state-control interaction matrix.
        with patch.object(closure, "_conditioned", side_effect=AssertionError("interaction path used")):
            result = closure.assess_state(**witness_data())
        self.assertEqual(result["selection_blocks"], 2)


def cell_gaussian_solution(train_x, train_y, train_cells, query_x, query_cells, alpha):
    """Independent Gaussian saddle system with unpenalized cell intercepts."""
    cells = np.unique(train_cells)
    fit_indicator = (train_cells[:, None] == cells).astype(float)
    query_indicator = (query_cells[:, None] == cells).astype(float)
    means = np.stack([train_x[train_cells == cell].mean(axis=0) for cell in cells])
    centered = train_x - fit_indicator @ means
    active = np.any(np.stack([
        np.ptp(train_x[train_cells == cell], axis=0) > 0 for cell in cells]), axis=0)
    scale = centered[:, active].std(axis=0)
    z = centered[:, active] / scale
    query = (query_x - query_indicator @ means)[:, active] / scale
    # Use direct coordinate differences, not the production Gram-distance code.
    kernel = np.exp(-np.square(z[:, None, :] - z[None, :, :]).sum(axis=-1) / (2 * z.shape[1]))
    cross = np.exp(-np.square(query[:, None, :] - z[None, :, :]).sum(axis=-1) / (2 * z.shape[1]))
    n, outputs = train_y.shape
    system = np.block([
        [kernel + alpha * n * np.eye(n), fit_indicator],
        [fit_indicator.T, np.zeros((len(cells), len(cells)))],
    ])
    rhs = np.vstack((np.column_stack((train_y, np.eye(n))),
                     np.zeros((len(cells), outputs + n))))
    solution = np.linalg.solve(system, rhs)
    prediction = np.column_stack((cross, query_indicator)) @ solution[:, :outputs]
    smoother = np.column_stack((kernel, fit_indicator)) @ solution[:, outputs:]
    return prediction, dict(active_features=int(active.sum()), effective_df=float(np.trace(smoother)))


class RbfCellWitnessTests(unittest.TestCase):
    def test_both_witness_arms_match_independent_gaussian_solution(self):
        for seed in (41, 42, 43):
            with self.subTest(seed=seed):
                arguments = witness_data(seed)
                for prefix in ("fit", "selection"):
                    state, history = arguments[prefix + "_state"], arguments[prefix + "_history"]
                    effects = np.array([-4., 2., 8.])[arguments[prefix + "_cells"], None]
                    arguments[prefix + "_targets"] = effects + state[:, :1] ** 2 + 2 * history[:, :1] ** 2
                result = closure.assess_state(**arguments, witness_kind="rbf")
                scale = arguments["fit_targets"].std(axis=0)
                predictions = []
                for arm, include_history in (("state", False), ("augmented", True)):
                    fit_x, query_x = arguments["fit_state"], arguments["selection_state"]
                    if include_history:
                        fit_x = np.column_stack((fit_x, arguments["fit_history"]))
                        query_x = np.column_stack((query_x, arguments["selection_history"]))
                    predicted, diagnostic = cell_gaussian_solution(
                        fit_x, arguments["fit_targets"] / scale, arguments["fit_cells"],
                        query_x, arguments["selection_cells"], arguments["penalty"])
                    predictions.append(predicted)
                    actual = result[arm + "_predictor_diagnostic"]
                    self.assertEqual(actual["active_features"], diagnostic["active_features"])
                    self.assertAlmostEqual(actual["effective_df"], diagnostic["effective_df"], places=10)
                truth = arguments["selection_targets"] / scale
                state_error, augmented_error = [(prediction - truth) ** 2 for prediction in predictions]
                gains = []
                for row in result["source_block_rows"]:
                    mask = arguments["selection_blocks"] == int(row["block"])
                    expected_gain = (state_error[mask] - augmented_error[mask]).mean(axis=0)
                    for key, expected in (("state_mse", state_error[mask].mean(axis=0)),
                                          ("augmented_mse", augmented_error[mask].mean(axis=0)),
                                          ("history_gain", expected_gain)):
                        np.testing.assert_allclose(row[key], expected, rtol=1e-10, atol=1e-11)
                    gains.append(expected_gain)
                np.testing.assert_allclose(result["mean_history_gain"], np.mean(gains, axis=0), rtol=1e-10, atol=1e-11)
                self.assertEqual(result["witness_kind"], "rbf")
                self.assertEqual(result["control_conditioning"], "fixed_factorial_cells_shared_rbf_state")
                self.assertEqual(result["ridge_diagonal_penalty"], arguments["penalty"] * len(arguments["fit_state"]))

    def test_selection_values_do_not_fit_scaling_features_or_diagnostics(self):
        arguments = witness_data(42)
        for prefix in ("fit", "selection"):
            arguments[prefix + "_state"] = np.column_stack((
                arguments[prefix + "_state"], arguments[prefix + "_cells"] * 2.))
        before = closure.assess_state(**arguments, witness_kind="rbf")
        changed = arguments.copy()
        changed["selection_state"] = arguments["selection_state"].copy()
        changed["selection_state"][:, -1] = np.arange(len(changed["selection_state"])) * 1.e6
        inactive_changed = closure.assess_state(**changed, witness_kind="rbf")
        self.assertEqual(before, inactive_changed)
        changed["selection_state"] = changed["selection_state"] * 1.e6
        changed["selection_history"] = arguments["selection_history"] * -1.e6
        changed["selection_targets"] = arguments["selection_targets"] + 10000.
        after = closure.assess_state(**changed, witness_kind="rbf")
        for key in ("training_target_scale", "defined_targets", "state_predictor_diagnostic",
                    "augmented_predictor_diagnostic"):
            self.assertEqual(before[key], after[key])
        np.testing.assert_allclose(before["training_target_scale"], arguments["fit_targets"].std(axis=0))
        self.assertEqual(before["normalization_fitted_on"], "fit_only")
        self.assertEqual(before["state_predictor_diagnostic"]["active_features"], 3)
        self.assertEqual(before["augmented_predictor_diagnostic"]["active_features"], 5)
        self.assertNotEqual(before["source_block_rows"], after["source_block_rows"])

    def test_rbf_requires_both_cell_arrays_and_rejects_unknown_kinds(self):
        for omitted in (("fit_cells",), ("selection_cells",), ("fit_cells", "selection_cells")):
            with self.subTest(omitted=omitted):
                arguments = witness_data(43)
                for key in omitted:
                    del arguments[key]
                with self.assertRaisesRegex(ValueError, "explicit fitting and selection factorial cells"):
                    closure.assess_state(**arguments, witness_kind="rbf")
        for kind in ("gaussian", "cells", None, 1):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "witness_kind must be linear or rbf"):
                    closure.assess_state(**witness_data(41), witness_kind=kind)


if __name__ == "__main__":
    unittest.main()

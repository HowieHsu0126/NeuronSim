"""Behavioral tests for whole-block residual prediction."""

import unittest

import numpy as np

from population_dynamics.residual_prediction import crossfit_residuals, fit_residual_predictor


ALPHAS = [0.0001, 0.01, 1.0, 100.0]


class ResidualPredictionTests(unittest.TestCase):
    def test_outer_block_cannot_change_fitted_models_or_alpha(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(5, 2, 13, 3))
        y = 3 + x[..., :2] + 0.2 * rng.normal(size=(5, 2, 13, 2))
        before = next(crossfit_residuals(x, y, 2, ALPHAS))
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[0] = changed_x[0] * 100 + 700
        changed_y[0] = -900
        after = next(crossfit_residuals(changed_x, changed_y, 2, ALPHAS))
        self.assertEqual(before["selected_alphas"], after["selected_alphas"])
        self.assertEqual((before["block_index"], before["inner_block_index"]), (0, 1))
        np.testing.assert_array_equal(before["training_target_scale"], after["training_target_scale"])
        for name in ("current", "history"):
            for field in ("x_mean", "x_scale", "y_mean", "y_scale", "coefficients"):
                np.testing.assert_array_equal(
                    getattr(before["models"][name], field), getattr(after["models"][name], field)
                )
            self.assertFalse(np.allclose(before["predictions"][name], after["predictions"][name]))

    def test_alpha_is_selected_on_inner_block_and_refit_uses_that_block(self):
        values = np.array([-3., -2., -1., 1., 2., 3.])
        x = np.zeros((4, 1, 6, 2))
        x[..., 0] = values
        y = np.broadcast_to(values[None, None, :, None], (4, 1, 6, 1)).copy()
        original = next(crossfit_residuals(x, y, 1, ALPHAS))
        self.assertEqual(original["selected_alphas"], {"current": 0.0001, "history": 0.0001})
        y[1] = 0
        changed = next(crossfit_residuals(x, y, 1, ALPHAS))
        self.assertEqual(changed["selected_alphas"], {"current": 100.0, "history": 100.0})
        model = changed["models"]["current"]
        # The final refit includes the zero-target inner block, unlike selection fitting.
        np.testing.assert_allclose(model.y_scale, y[1:].reshape(-1, 1).std(axis=0))
        np.testing.assert_allclose(changed["predictions"]["current"][0, :, 0], values * (2 / 3) / 101)

    def test_ridge_matches_augmented_least_squares_and_unpenalized_intercept(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5, 2, 11, 3)) * np.array([2., 0.3, 4.]) + np.array([9., -2., 7.])
        y = x @ np.array([[2., 0.5], [-3., 1.], [0.2, -0.4]]) + np.array([13., -8.])
        alpha = 0.7
        result = next(crossfit_residuals(x, y, 2, [alpha]))
        fitting_x, fitting_y = x[1:].reshape(-1, 3), y[1:].reshape(-1, 2)
        for name, width in (("current", 2), ("history", 3)):
            model = result["models"][name]
            xm, xs = fitting_x[:, :width].mean(axis=0), fitting_x[:, :width].std(axis=0)
            ym, ys = fitting_y.mean(axis=0), fitting_y.std(axis=0)
            design = (fitting_x[:, :width] - xm) / xs
            target = (fitting_y - ym) / ys
            augmented_x = np.vstack((design, np.sqrt(design.shape[0] * alpha) * np.eye(width)))
            augmented_y = np.vstack((target, np.zeros((width, 2))))
            expected, *_ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
            np.testing.assert_allclose(model.x_mean, xm)
            np.testing.assert_allclose(model.x_scale, xs)
            np.testing.assert_allclose(model.y_mean, ym)
            np.testing.assert_allclose(model.y_scale, ys)
            np.testing.assert_allclose(model.coefficients, expected, atol=1e-12)
            np.testing.assert_allclose(
                result["predictions"][name], (((x[0, ..., :width] - xm) / xs) @ expected) * ys + ym,
                atol=1e-12,
            )
        shifted = next(crossfit_residuals(x, y + 1000, 2, [alpha]))
        np.testing.assert_allclose(shifted["predictions"]["history"], result["predictions"]["history"] + 1000)
        np.testing.assert_allclose(
            shifted["models"]["history"].coefficients, result["models"]["history"].coefficients, atol=1e-12
        )

    def test_feature_and_target_units_do_not_change_selected_ridge(self):
        rng = np.random.default_rng(43)
        x = rng.normal(size=(5, 2, 15, 3))
        y = x[..., :2] + rng.normal(size=(5, 2, 15, 2))
        baseline = next(crossfit_residuals(x, y, 2, ALPHAS))
        scaled = next(crossfit_residuals(
            x * np.array([0.01, 20., 3.]) + np.array([2., 5., -8.]),
            y * np.array([7., 0.3]) + np.array([100., -10.]), 2, ALPHAS,
        ))
        self.assertEqual(baseline["selected_alphas"], scaled["selected_alphas"])
        for name in ("current", "history"):
            np.testing.assert_allclose(
                scaled["predictions"][name], baseline["predictions"][name] * np.array([7., 0.3]) + np.array([100., -10.]),
                atol=1e-11,
            )

    def test_constant_features_and_targets_have_explicit_support(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(5, 2, 9, 3))
        x[..., 1], x[..., 2] = 1 / 3, 7
        y = np.stack((2 * x[..., 0] + 5, np.full(x.shape[:3], 2 / 3)), axis=-1)
        y[0, ..., 1] = 19
        result = next(crossfit_residuals(x, y, 2, ALPHAS))
        for name, width in (("current", 2), ("history", 3)):
            model = result["models"][name]
            np.testing.assert_array_equal(model.x_scale[1:], np.ones(width - 1))
            np.testing.assert_array_equal(model.coefficients[1:], np.zeros((width - 1, 2)))
            self.assertEqual(model.y_scale[1], 1)
            np.testing.assert_array_equal(model.coefficients[:, 1], np.zeros(width))
            np.testing.assert_array_equal(result["predictions"][name][..., 1], np.full((2, 9), 2 / 3))

    def test_temporal_memory_predicts_heldout_blocks(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(6, 2, 60, 2))
        y = np.empty((6, 2, 60, 1))
        for block in range(6):
            for episode in range(2):
                previous = rng.normal()
                for time in range(60):
                    x[block, episode, time, 1] = previous
                    previous = 0.9 * previous + 0.1 * rng.normal()
                    y[block, episode, time, 0] = previous
        current_error, history_error = 0.0, 0.0
        seen = []
        for result in crossfit_residuals(x, y, 1, ALPHAS):
            block = result["block_index"]
            seen.append(block)
            self.assertEqual(result["inner_block_index"], (block + 1) % 6)
            current_error += np.mean((y[block] - result["predictions"]["current"])**2)
            history_error += np.mean((y[block] - result["predictions"]["history"])**2)
        self.assertEqual(seen, list(range(6)))
        self.assertLess(history_error, 0.2 * current_error)

    def test_invalid_inputs_are_rejected(self):
        x, y = np.ones((4, 1, 3, 2)), np.ones((4, 1, 3, 1))
        bad_cases = [
            (x[0], y, 1, ALPHAS), (x[:3], y[:3], 1, ALPHAS),
            (x, y[:, :, :2], 1, ALPHAS), (x[:, :, :0], y[:, :, :0], 1, ALPHAS),
            (x.astype(complex), y, 1, ALPHAS),
            (x, y, 0, ALPHAS), (x, y, 2, ALPHAS), (x, y, True, ALPHAS),
            (x, y, 1.0, ALPHAS), (x, y, 1, []), (x, y, 1, [1, 1]),
            (x, y, 1, [0]), (x, y, 1, [-1]), (x, y, 1, [np.inf]),
            (x, y, 1, [[1]]), (x, y, 1, ["1"]),
        ]
        for args in bad_cases:
            with self.subTest(shapes=(args[0].shape, args[1].shape), width=args[2], alphas=args[3]):
                with self.assertRaises(ValueError):
                    next(crossfit_residuals(*args))
        for which in ("x", "y"):
            bad_x, bad_y = x.copy(), y.copy()
            (bad_x if which == "x" else bad_y)[-1, 0, 0, 0] = np.nan
            with self.subTest(nonfinite=which), self.assertRaisesRegex(ValueError, "finite"):
                next(crossfit_residuals(bad_x, bad_y, 1, ALPHAS))


class AllBlockResidualFitTests(unittest.TestCase):
    def test_all_blocks_and_requested_width_match_augmented_least_squares(self):
        rng = np.random.default_rng(43)
        x = rng.normal(size=(3, 2, 7, 3)) + np.arange(3)[:, None, None, None]
        y = x[..., :2] * np.array([2., -3.]) + np.array([7., 4.])
        alpha = 0.01
        for width in (2, 3):
            model = fit_residual_predictor(x, y, width, alpha)
            rows_x, rows_y = x.reshape(-1, 3)[:, :width], y.reshape(-1, 2)
            xm, xs = rows_x.mean(0), rows_x.std(0)
            ym, ys = rows_y.mean(0), rows_y.std(0)
            design = np.vstack(((rows_x - xm) / xs, np.sqrt(len(rows_x) * alpha) * np.eye(width)))
            target = np.vstack(((rows_y - ym) / ys, np.zeros((width, 2))))
            expected, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.assertEqual(model.alpha, alpha)
            np.testing.assert_allclose(model.x_mean, xm)
            np.testing.assert_allclose(model.y_mean, ym)
            np.testing.assert_allclose(model.coefficients, expected, atol=1e-12)
        changed = y.copy()
        changed[0] += 30
        np.testing.assert_allclose(
            fit_residual_predictor(x, changed, 2, alpha).y_mean,
            fit_residual_predictor(x, y, 2, alpha).y_mean + 10,
        )

    def test_single_fitting_block_is_supported(self):
        x = np.array([[[[-1., 7.], [1., 7.]]]])
        y = np.array([[[[-2.], [2.]]]])
        model = fit_residual_predictor(x, y, 2, 0.01)
        np.testing.assert_allclose(model.predict(x), y / 1.01)
        np.testing.assert_array_equal(model.coefficients[1], [0.])

    def test_fixed_fit_rejects_invalid_inputs(self):
        x, y = np.ones((2, 1, 3, 2)), np.ones((2, 1, 3, 1))
        for width, alpha in ((0, .01), (3, .01), (True, .01), (1.5, .01),
                             (1, 0), (1, -1), (1, np.inf), (1, np.nan),
                             (1, True), (1, [1]), (1, "1")):
            with self.subTest(width=width, alpha=alpha), self.assertRaises(ValueError):
                fit_residual_predictor(x, y, width, alpha)
        invalid_x = x.copy()
        invalid_x[-1, 0, 0, -1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_residual_predictor(invalid_x, y, 1, .01)
        with self.assertRaises(ValueError):
            fit_residual_predictor(x[:, :, :0], y[:, :, :0], 1, .01)


if __name__ == "__main__":
    unittest.main()

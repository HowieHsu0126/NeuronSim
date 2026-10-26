"""Whole-block cross-fitting for causal residual prediction.

For each output coordinate j, ridge minimizes the mean over fitting rows of
normalized squared error plus ``alpha * sum(coefficients[:, j] ** 2)``.
Averaging this objective over coordinates leaves alpha independent of output
width. Feature and target means and population standard deviations come only
from the fitting rows; the intercept is unpenalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class RidgeModel:
    """A predictor whose coefficients act on standardized inputs and targets."""

    alpha: float
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if values.ndim < 1 or values.shape[-1] != self.x_mean.size:
            raise ValueError("prediction feature width does not match fitted model")
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("prediction features must be finite real numbers")
        standardized = (values - self.x_mean) / self.x_scale
        return (standardized @ self.coefficients) * self.y_scale + self.y_mean


@dataclass(frozen=True)
class _Moments:
    n: int
    x_mean: np.ndarray
    y_mean: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray


def _block_moments(features: np.ndarray, targets: np.ndarray) -> _Moments:
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    y = np.asarray(targets, dtype=np.float64).reshape(-1, targets.shape[-1])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must contain only finite values")
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)
    # Exact constant columns must not acquire a scale from mean-rounding noise.
    x_mean[x_min == x_max] = x_min[x_min == x_max]
    y_mean[y_min == y_max] = y_min[y_min == y_max]
    xc, yc = x - x_mean, y - y_mean
    return _Moments(
        x.shape[0], x_mean, y_mean, xc.T @ xc, xc.T @ yc,
        np.einsum("ij,ij->j", yc, yc), x_min, x_max, y_min, y_max,
    )


def _combine(blocks: list[_Moments], indices: Sequence[int]) -> _Moments:
    """Merge centered sufficient statistics without loading fitting rows again."""
    result = blocks[indices[0]]
    for index in indices[1:]:
        other = blocks[index]
        n = result.n + other.n
        dx, dy = other.x_mean - result.x_mean, other.y_mean - result.y_mean
        weight = result.n * (other.n / n)
        result = _Moments(
            n,
            result.x_mean + dx * (other.n / n),
            result.y_mean + dy * (other.n / n),
            result.xx + other.xx + weight * np.outer(dx, dx),
            result.xy + other.xy + weight * np.outer(dx, dy),
            result.yy + other.yy + weight * dy * dy,
            np.minimum(result.x_min, other.x_min),
            np.maximum(result.x_max, other.x_max),
            np.minimum(result.y_min, other.y_min),
            np.maximum(result.y_max, other.y_max),
        )
    return result


def _problem(stats: _Moments, width: int) -> tuple[RidgeModel, np.ndarray, np.ndarray]:
    x_mean, y_mean = stats.x_mean[:width].copy(), stats.y_mean.copy()
    x_constant = stats.x_min[:width] == stats.x_max[:width]
    y_constant = stats.y_min == stats.y_max
    x_mean[x_constant] = stats.x_min[:width][x_constant]
    y_mean[y_constant] = stats.y_min[y_constant]
    x_var, y_var = np.diag(stats.xx)[:width] / stats.n, stats.yy / stats.n
    if np.any(x_var[~x_constant] <= 0) or np.any(y_var[~y_constant] <= 0):
        raise ValueError("nonconstant fitting columns have unrepresentable variance")
    x_scale = np.sqrt(x_var)
    y_scale = np.sqrt(y_var)
    x_scale[x_constant], y_scale[y_constant] = 1.0, 1.0
    gram = stats.xx[:width, :width] / stats.n / x_scale[:, None] / x_scale[None, :]
    cross = stats.xy[:width] / stats.n / x_scale[:, None] / y_scale[None, :]
    gram[x_constant, :], gram[:, x_constant] = 0.0, 0.0
    cross[x_constant, :], cross[:, y_constant] = 0.0, 0.0
    if not all(np.isfinite(array).all() for array in (gram, cross, x_scale, y_scale)):
        raise ValueError("fitting sufficient statistics are not finite")
    model = RidgeModel(0.0, x_mean, x_scale, y_mean, y_scale, np.empty_like(cross))
    return model, (gram + gram.T) * 0.5, cross


def _with_coefficients(base: RidgeModel, alpha: float, coefficients: np.ndarray) -> RidgeModel:
    return RidgeModel(
        float(alpha), base.x_mean, base.x_scale, base.y_mean, base.y_scale,
        coefficients,
    )


def _validation_terms(base: RidgeModel, stats: _Moments) -> tuple:
    width = base.x_mean.size
    gram = stats.xx[:width, :width] / stats.n / base.x_scale[:, None] / base.x_scale[None, :]
    cross = stats.xy[:width] / stats.n / base.x_scale[:, None] / base.y_scale[None, :]
    target_variance = float(np.sum(stats.yy / stats.n / base.y_scale**2))
    x_shift = (stats.x_mean[:width] - base.x_mean) / base.x_scale
    y_shift = (stats.y_mean - base.y_mean) / base.y_scale
    return gram, cross, target_variance, x_shift, y_shift


def _validation_score(coefficients: np.ndarray, terms: tuple) -> float:
    gram, cross, target_variance, x_shift, y_shift = terms
    cross_term = 2.0 * float(np.sum(coefficients * cross))
    quadratic = float(np.sum(coefficients * (gram @ coefficients)))
    bias = float(np.sum((x_shift @ coefficients - y_shift)**2))
    error = target_variance - cross_term + quadratic + bias
    tolerance = 64 * np.finfo(np.float64).eps * (
        abs(target_variance) + abs(cross_term) + abs(quadratic) + abs(bias) + 1.0
    )
    if not np.isfinite(error) or error < -tolerance:
        raise ValueError("inner validation error is numerically invalid")
    return max(error, 0.0) / coefficients.shape[1]


def _select_alpha(fitting: _Moments, validation: _Moments, width: int, alphas: np.ndarray) -> float:
    base, gram, cross = _problem(fitting, width)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = 64 * np.finfo(np.float64).eps * width * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    if eigenvalues[0] < -tolerance:
        raise ValueError("fitting Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_cross = eigenvectors.T @ cross
    terms = _validation_terms(base, validation)
    best_alpha, best_score = float(alphas[0]), float("inf")
    for alpha in alphas:
        coefficients = eigenvectors @ (projected_cross / (eigenvalues[:, None] + alpha))
        score = _validation_score(coefficients, terms)
        if score < best_score:
            best_alpha, best_score = float(alpha), score
    return best_alpha


def _fit(stats: _Moments, width: int, alpha: float) -> RidgeModel:
    base, gram, cross = _problem(stats, width)
    gram.flat[:: width + 1] += alpha
    coefficients = np.linalg.solve(gram, cross)
    if not np.isfinite(coefficients).all():
        raise ValueError("fitted ridge coefficients are not finite")
    return _with_coefficients(base, alpha, coefficients)


def _validate_arrays(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 4 or Y.ndim != 4:
        raise ValueError("X and Y must have four dimensions: block, episode, time, coordinate")
    if X.shape[:3] != Y.shape[:3] or any(size == 0 for size in X.shape + Y.shape):
        raise ValueError("X and Y must have matching nonempty block, episode, and time dimensions")
    if X.dtype.kind not in "fiu" or Y.dtype.kind not in "fiu":
        raise ValueError("X and Y must contain real numeric values")
    return X, Y


def fit_residual_predictor(
    X: np.ndarray, Y: np.ndarray, width: int, alpha: float,
) -> RidgeModel:
    """Fit the first ``width`` features on every supplied fitting block.

    Alpha is supplied after fitting-only model selection. This function performs
    no selection or split assignment; callers must supply only admissible fitting
    blocks. It uses the same normalization and ridge objective as cross-fitting.
    """
    X, Y = _validate_arrays(X, Y)
    if (isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer))
            or not 0 < width <= X.shape[-1]):
        raise ValueError("width must be a positive integer no larger than the feature width")
    value = np.asarray(alpha)
    if value.ndim != 0 or value.dtype.kind not in "fiu":
        raise ValueError("alpha must be a finite positive real scalar")
    alpha = float(value)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive real scalar")
    blocks = [_block_moments(X[index], Y[index]) for index in range(X.shape[0])]
    return _fit(_combine(blocks, list(range(len(blocks)))), width, alpha)


def crossfit_residuals(
    X: np.ndarray,
    Y: np.ndarray,
    current_width: int,
    alphas: Sequence[float],
) -> Iterator[dict]:
    """Yield predictions for each outer held-out block.

    X and Y have shape [block, episode, time, coordinate]. The current predictor
    uses X[..., :current_width]; the history predictor uses all columns. Callers
    construct causal features without crossing episode boundaries.

    For outer block b, block (b + 1) % B is the inner validation block. Alpha
    selection fits B - 2 blocks; the selected model is refit on B - 1 blocks.
    Both fits compute their own normalization. Equal validation scores select
    the smallest alpha. The outer block never enters either fit or selection.

    Statistics require O(B * (p*p + p*d)) memory. Only each held-out block's
    predictions and two fitted models are yielded; callers need not retain them.
    Constant fitting features have scale one and zero coefficients. Constant
    fitting targets have scale one and are predicted by their fitting mean.
    """
    X, Y = _validate_arrays(X, Y)
    if X.shape[0] < 4:
        raise ValueError("at least four blocks are required")
    if isinstance(current_width, (bool, np.bool_)) or not isinstance(current_width, (int, np.integer)):
        raise ValueError("current_width must be an integer")
    if not 0 < current_width < X.shape[-1]:
        raise ValueError("current_width must be positive and smaller than the history width")
    alpha_values = np.asarray(alphas)
    if alpha_values.ndim != 1 or alpha_values.size == 0 or alpha_values.dtype.kind not in "fiu":
        raise ValueError("alphas must be a nonempty one-dimensional real numeric sequence")
    alpha_values = np.asarray(alpha_values, dtype=np.float64)
    if not np.isfinite(alpha_values).all() or np.any(alpha_values <= 0):
        raise ValueError("alphas must be finite and positive")
    if np.unique(alpha_values).size != alpha_values.size:
        raise ValueError("alphas must be unique")
    alpha_values = np.sort(alpha_values)
    blocks = [_block_moments(X[index], Y[index]) for index in range(X.shape[0])]
    for outer in range(len(blocks)):
        inner = (outer + 1) % len(blocks)
        inner_fit = _combine(blocks, [b for b in range(len(blocks)) if b not in (outer, inner)])
        selected = {
            name: _select_alpha(inner_fit, blocks[inner], width, alpha_values)
            for name, width in (("current", current_width), ("history", X.shape[-1]))
        }
        outer_fit = _combine(blocks, [b for b in range(len(blocks)) if b != outer])
        models = {
            name: _fit(outer_fit, width, selected[name])
            for name, width in (("current", current_width), ("history", X.shape[-1]))
        }
        yield {
            "block_index": outer,
            "inner_block_index": inner,
            "selected_alphas": selected,
            "training_target_scale": models["history"].y_scale.copy(),
            "predictions": {
                "current": models["current"].predict(X[outer, ..., :current_width]),
                "history": models["history"].predict(X[outer]),
            },
            "models": models,
        }

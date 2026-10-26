"""Finite-ensemble-corrected scores for declared population-field projections."""
from __future__ import annotations

import numpy as np


def energy_score_terms(samples, observed):
    """Expose the two normalized terms while preserving the score's arithmetic."""
    x = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    y = np.asarray(observed, dtype=np.float64).reshape(-1)
    if x.shape[0] < 2 or x.shape[1] != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("at least two finite matching field samples required")
    first = np.linalg.norm(x - y, axis=1).mean()
    pair_sum = sum(np.linalg.norm(x[i + 1:] - x[i], axis=1).sum() for i in range(len(x) - 1))
    second = pair_sum / (len(x) * (len(x) - 1))
    normalizer = np.sqrt(y.size)
    return dict(draw_to_observation=float(first / normalizer), subtracted_draw_to_draw=float(second / normalizer),
                energy_score=float((first - second) / normalizer))


def energy_score(samples, observed):
    return energy_score_terms(samples, observed)["energy_score"]


def variogram_score(samples, observed, left, right, lag=0):
    """Mean p=1 variogram score; remove finite-ensemble variance penalty.

    Pairs are declared field indices, with right indices observed `lag` bins
    later. Negative corrected estimates are valid and are never clipped.
    """
    x, y = np.asarray(samples, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    left, right = np.asarray(left), np.asarray(right)
    if x.ndim != 3 or y.shape != x.shape[1:] or len(x) < 2 or not 0 <= lag < len(y):
        raise ValueError("complete ensemble, field trajectory and valid lag required")
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("nonempty paired field indices required")
    stop = y.shape[0] - lag
    actual = np.abs(y[:stop, left] - y[lag:, right])
    draws = np.abs(x[:, :stop, left] - x[:, lag:, right])
    return float(((actual - draws.mean(0)) ** 2 - draws.var(0, ddof=1) / len(x)).mean())


def field_pairs(field_populations, ix, iy):
    populations, ix, iy = map(np.asarray, (field_populations, ix, iy))
    n = int(max(ix.max(), iy.max())) + 1
    lookup = {(str(p), int(x), int(y)): i for i, (p, x, y) in enumerate(zip(populations, ix, iy))}
    spatial, cross = [], []
    names = sorted(set(populations))
    for i, (population, x, y) in enumerate(zip(populations, ix, iy)):
        for key in ((str(population), (int(x) + 1) % n, int(y)), (str(population), int(x), (int(y) + 1) % n)):
            spatial.append((i, lookup[key]))
        for p in names:
            if str(p) > str(population):
                cross.append((i, lookup[(str(p), int(x), int(y))]))
    return {"spatial": np.asarray(spatial, dtype=int), "cross_population": np.asarray(cross, dtype=int)}

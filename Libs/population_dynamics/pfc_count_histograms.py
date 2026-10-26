"""Sparse all-unit count histograms with complete block/window denominators."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .pfc_delay_analysis import require

AXES = ('condition', 'block', 'phase', 'window', 'field', 'spike_count')


def accumulate(histograms, counts, unit_fields, window_bins, *, condition, phase, window):
    x, fields = np.asarray(counts), np.asarray(unit_fields)
    require(x.ndim == 3 and min(x.shape) > 0 and x.dtype.kind in 'iu' and np.all(x >= 0)
            and fields.shape == (x.shape[2],) and fields.dtype.kind in 'iu' and np.all(fields >= 0),
            'complete nonnegative integer block/time/unit counts and field identities required')
    require(type(window_bins) is int and window_bins > 0 and x.shape[1] % window_bins == 0
            and int(x.max()) <= np.iinfo(np.int64).max//window_bins,
            'complete count windows and safe int64 sums required')
    grouped = x.reshape(x.shape[0], x.shape[1]//window_bins, window_bins, x.shape[2]).sum(axis=2, dtype=np.int64)
    for field in np.unique(fields):
        selected = grouped[:, :, fields == field]
        for block in range(len(x)):
            values, numbers = np.unique(selected[block], return_counts=True)
            for value, number in zip(values, numbers):
                key = (condition, block, phase, window, int(field), int(value))
                histograms[key] = histograms.get(key, 0)+int(number)


def write(path, histograms, *, conditions, seeds, phases, windows_ms, field_sizes):
    sizes = np.asarray(field_sizes)
    require(sizes.ndim == 1 and sizes.dtype.kind in 'iu' and np.all(sizes > 0),
            'complete positive integer field sizes required')
    keys = sorted(histograms)
    require(keys, 'nonempty all-unit histogram required')
    coordinates = np.asarray(keys, dtype=np.int64)
    observations = np.asarray([histograms[key] for key in keys], dtype=np.int64)
    shape = (len(conditions), len(seeds), len(phases), len(windows_ms), len(sizes))
    require(coordinates.shape == (len(keys), 6) and np.all(coordinates >= 0)
            and np.all(coordinates[:, :5] < np.asarray(shape))
            and np.all(observations > 0), 'valid sparse count-histogram coordinates required')
    expected = np.zeros(shape, dtype=np.int64)
    for phase, (start, stop) in enumerate(phases.values()):
        for window, width in enumerate(windows_ms):
            windows = round((stop-start)/width)
            require(width > 0 and windows > 0 and windows*width == stop-start,
                    'whole registered phase windows required')
            expected[:, :, phase, window, :] = windows*sizes
    observed = np.zeros(shape, dtype=np.int64)
    np.add.at(observed, tuple(coordinates[:, :5].T), observations)
    require(np.array_equal(observed, expected), 'missing or duplicated unit/window histogram mass')
    np.savez_compressed(Path(path), coordinates=coordinates, observations=observations,
        coordinate_axes=np.asarray(AXES), condition_names=np.asarray(conditions),
        source_rng_seeds=np.asarray(seeds, dtype=np.int64), phase_names=np.asarray(list(phases)),
        phase_bounds_ms=np.asarray(list(phases.values()), dtype=float),
        window_ms=np.asarray(windows_ms, dtype=float), field_unit_counts=sizes,
        expected_observations=expected)
    return dict(sparse_rows=len(keys), unit_window_observations=int(observations.sum()),
        blocks=len(seeds), field_count=len(sizes), window_ms=list(windows_ms),
        observations_are_independent_replicates=False,
        count_support='all observed integer counts, including zero; no clipping or tail truncation',
        denominator='complete source units in each field times complete nonoverlapping phase windows')


def cdf(coordinates, observations, thresholds, shape):
    """Return block-level field CDFs; units/windows are averaged observations."""
    indices, mass, edges = map(np.asarray, (coordinates, observations, thresholds))
    require(indices.ndim == 2 and indices.shape[1] == 6 and indices.dtype.kind in 'iu'
            and mass.shape == (len(indices),) and mass.dtype.kind in 'iu' and np.all(mass > 0)
            and edges.ndim == 1 and len(edges) > 0 and edges.dtype.kind in 'iu'
            and np.all(edges >= 0) and np.all(edges[1:] > edges[:-1])
            and len(shape) == 5 and np.all(indices >= 0)
            and np.all(indices[:, :5] < np.asarray(shape)), 'aligned sparse histogram and sorted integer CDF thresholds required')
    total = np.zeros(shape, dtype=np.int64)
    np.add.at(total, tuple(indices[:, :5].T), mass)
    require(np.all(total > 0), 'every registered histogram denominator required')
    values = np.zeros((*shape, len(edges)), dtype=np.float64)
    for index, threshold in enumerate(edges):
        chosen = indices[:, 5] <= threshold
        numerator = np.zeros(shape, dtype=np.int64)
        np.add.at(numerator, tuple(indices[chosen, :5].T), mass[chosen])
        values[..., index] = numerator/total
    return values

"""Periodic cell boundaries, empty support, conservation and shared mapping."""
import numpy as np
import pytest

from .population_field import PopulationFieldMapping
from .spatial_field import PeriodicPopulationGrid


def grid(ids, populations, positions, n=2, extent=2.0):
    return PeriodicPopulationGrid.from_positions(ids, populations, positions,
        extent_mm=extent, cells_per_axis=n, definition="native source 2D periodic positions in mm")


def test_positive_boundary_wrap_and_internal_half_open_cells():
    inside = np.nextafter(1.0, -np.inf)
    positions = np.array([[-1., -1.], [1., 1.], [0., -.5], [-.5, 0.], [inside, inside], [1., 0.]])
    observer = grid([40, 7, 91, 3, 12, 100], ["P"] * 6, positions)
    np.testing.assert_array_equal(observer.mapping.group_index, [0, 0, 1, 2, 3, 2])
    result = observer.project([([40, 7, 91, 3, 12, 100], np.arange(1, 7), None)], bin_seconds=1.0)
    np.testing.assert_array_equal(result.field.count_sum, [3, 3, 10, 5])
    np.testing.assert_array_equal(result.field.total_unit_count, [2, 1, 2, 1])
    np.testing.assert_array_equal(observer.centers_mm, [[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]])
    np.testing.assert_array_equal(positions[1], [1., 1.])


def test_all_grid_cells_retained_and_empty_cells_differ_from_missing_units():
    observer = grid([8], ["P"], [[-1., -1.]])
    complete = observer.project([([8], np.array([2]), None)], bin_seconds=.5)
    absent = observer.project([([], np.empty(0), None)], bin_seconds=.5)
    np.testing.assert_array_equal(complete.field.total_unit_count, [1, 0, 0, 0])
    np.testing.assert_array_equal(complete.density_hz_per_mm2, [4., 0., 0., 0.])
    np.testing.assert_array_equal(complete.field.observed_rate_hz, [4., np.nan, np.nan, np.nan])
    assert complete.field.complete.all()
    np.testing.assert_array_equal(absent.field.complete, [False, True, True, True])
    np.testing.assert_array_equal(absent.density_hz_per_mm2, [np.nan, 0., 0., 0.])
    assert np.isnan(absent.field.observed_rate_hz).all()


def test_masked_unit_makes_density_unknown_without_losing_observed_counts():
    observer = grid([1, 2], ["P", "P"], [[-.5, -.5], [-.25, -.25]])
    counts, mask = np.array([[0., np.nan], [2., 4.]]), np.array([[True, False], [True, True]])
    result = observer.project([([1, 2], counts, mask)], bin_seconds=.25)
    np.testing.assert_array_equal(result.field.count_sum[:, 0], [0, 6])
    np.testing.assert_array_equal(result.field.observed_unit_count[:, 0], [1, 2])
    np.testing.assert_array_equal(result.density_hz_per_mm2[:, 0], [np.nan, 24.])
    np.testing.assert_array_equal(result.field.observed_rate_hz[:, 0], [0., 12.])


def test_population_unit_and_chunk_permutations_preserve_cells_and_density():
    ids, populations = [91, 7, 44], ["Z", "A", "Z"]
    positions = np.array([[-1., -1.], [0., 0.], [.5, -.5]])
    observer = grid(ids, populations, positions)
    permuted = grid([44, 91, 7], ["Z", "Z", "A"], positions[[2, 0, 1]])
    full = observer.project([(ids, np.array([[1, 2, 3], [4, 5, 6]]), None)], bin_seconds=.5)
    chunks = [([7], np.array([[2], [5]]), None), ([44, 91], np.array([[3, 1], [6, 4]]), None)]
    split = observer.project(chunks, bin_seconds=.5)
    reordered = permuted.project(list(reversed(chunks)), bin_seconds=.5)
    for candidate in (split, reordered):
        np.testing.assert_array_equal(candidate.field.field_ids, full.field.field_ids)
        np.testing.assert_array_equal(candidate.field.count_sum, full.field.count_sum)
        np.testing.assert_array_equal(candidate.density_hz_per_mm2, full.density_hz_per_mm2)
    np.testing.assert_array_equal(observer.population_labels, ["A"] * 4 + ["Z"] * 4)
    np.testing.assert_array_equal(observer.mapping.unit_ids, ids)
    assert observer.mapping.unit_ids.size == split.field.total_unit_count.sum()


def test_density_conservation_and_same_mapping_for_source_and_prediction():
    observer = grid([2, 8, 13], ["E", "I", "E"], [[-1., -1.], [0., 0.], [.5, -.5]], n=4)
    source = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    generated = source + 1
    observed = observer.project([([2, 8, 13], source, None)], bin_seconds=.25)
    simulated = observer.project([([13, 2], generated[..., [2, 0]], None),
                                  ([8], generated[..., [1]], None)], bin_seconds=.25)
    for original, output in ((source, observed), (generated, simulated)):
        np.testing.assert_array_equal(output.field.count_sum.sum(axis=-1), original.sum(axis=-1))
        np.testing.assert_allclose(output.density_hz_per_mm2 * observer.cell_area_mm2 * .25,
                                   output.field.count_sum, rtol=0, atol=0)
    np.testing.assert_array_equal(observed.field.field_ids, simulated.field.field_ids)
    np.testing.assert_array_equal(observed.field.total_unit_count, simulated.field.total_unit_count)
    np.testing.assert_array_equal((simulated.field.count_sum - observed.field.count_sum).sum(axis=-1), np.full((2, 3), 3))


def test_explicit_field_vocabulary_keeps_order_and_empty_fields():
    mapping = PopulationFieldMapping.from_groups([1, 2], ["A", "B"], definition="source labels",
                                                 field_ids=["empty", "B", "A"])
    result = mapping.project([([1, 2], np.array([2, 3]), None)], bin_seconds=1.)
    np.testing.assert_array_equal(result.field_ids, ["empty", "B", "A"])
    np.testing.assert_array_equal(result.count_sum, [0, 3, 2])
    np.testing.assert_array_equal(result.total_unit_count, [0, 1, 1])
    assert result.complete.all() and np.isnan(result.observed_rate_hz[0])


@pytest.mark.parametrize("vocabulary", [[], ["A"], ["A", "B", "A"], ["A", "B", " "], ["A", "B", 3], "AB"])
def test_invalid_explicit_vocabulary_rejected(vocabulary):
    with pytest.raises(ValueError):
        PopulationFieldMapping.from_groups([1, 2], ["A", "B"], definition="source labels", field_ids=vocabulary)


@pytest.mark.parametrize("position", [[np.nextafter(1., np.inf), 0.], [np.nextafter(-1., -np.inf), 0.],
                                      [np.nan, 0.], [0., np.inf], [True, False]])
def test_invalid_positions_are_not_silently_wrapped(position):
    with pytest.raises(ValueError):
        grid([1], ["P"], [position])


@pytest.mark.parametrize("ids,populations,positions", [
    ([1, 1], ["P", "P"], [[0., 0.], [.5, .5]]), ([1], ["P"], [[0., 0., 0.]]),
    ([1], ["P", "Q"], [[0., 0.]]), ([1], [2], [[0., 0.]]), ([1], [" "], [[0., 0.]]),
])
def test_invalid_source_identity_or_position_shape_rejected(ids, populations, positions):
    with pytest.raises(ValueError):
        grid(ids, populations, positions)


@pytest.mark.parametrize("extent,n", [(0., 2), (-1., 2), (np.inf, 2), (True, 2),
                                      (2., 0), (2., -1), (2., True), (2., 2.), (2., np.int64(2))])
def test_invalid_grid_geometry_rejected(extent, n):
    with pytest.raises(ValueError):
        grid([1], ["P"], [[0., 0.]], n=n, extent=extent)


def test_grid_projection_retains_shared_reducer_rejections():
    observer = grid([1], ["P"], [[0., 0.]])
    for chunks, width in (([([2], np.array([1]), None)], 1.),
                          ([([1], np.array([1]), None), ([1], np.array([1]), None)], 1.),
                          ([([1], np.array([-1]), None)], 1.),
                          ([([1], np.array([1]), None)], True)):
        with pytest.raises(ValueError):
            observer.project(chunks, bin_seconds=width)

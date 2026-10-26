"""Independent population-field conservation and missing-observation contracts."""
import numpy as np
import pytest

from .population_field import PopulationFieldMapping


def mapping(ids=(42, 7, 900, 3), labels=("B", "A", "B", "A")):
    return PopulationFieldMapping.from_groups(ids, labels, definition="source brain-region labels")


def assert_same_readout(first, second):
    for name in ("count_sum", "observed_unit_count", "total_unit_count", "observed_rate_hz",
                 "complete", "field_ids", "supplied_unit_ids"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_permuted_noncontiguous_ids_conserve_counts_and_chunk_order():
    observer = mapping()
    chunks = [([900, 7], np.array([[5, 2], [0, 8]]), None),
              ([3, 42], np.array([[4, 1], [6, 3]]), None)]
    first = observer.project(chunks, bin_seconds=0.5)
    second = observer.project(list(reversed(chunks)), bin_seconds=0.5)
    whole = observer.project([([3, 900, 42, 7], np.array([[4, 5, 1, 2], [6, 0, 3, 8]]), None)], bin_seconds=0.5)
    assert_same_readout(first, second)
    assert_same_readout(first, whole)
    np.testing.assert_array_equal(first.field_ids, ["A", "B"])
    np.testing.assert_array_equal(first.count_sum, [[6, 6], [14, 3]])
    np.testing.assert_array_equal(first.count_sum.sum(axis=-1), [12, 17])
    np.testing.assert_array_equal(first.observed_unit_count, [[2, 2], [2, 2]])
    np.testing.assert_array_equal(first.total_unit_count, [2, 2])
    np.testing.assert_array_equal(first.observed_rate_hz, [[6.0, 6.0], [14.0, 3.0]])
    assert first.complete.all()
    np.testing.assert_array_equal(first.supplied_unit_ids, [42, 7, 900, 3])


def test_missing_units_and_masks_are_distinct_from_silent_supplied_units():
    observer = mapping((10, 11, 12), ("A", "A", "B"))
    counts = np.array([[np.nan, 0.0], [4.0, -999.0]])
    mask = np.array([[False, True], [True, False]])
    result = observer.project([([12, 10], counts, mask)], bin_seconds=0.1)
    np.testing.assert_array_equal(result.total_unit_count, [2, 1])
    np.testing.assert_array_equal(result.count_sum, [[0, 0], [0, 4]])
    np.testing.assert_array_equal(result.observed_unit_count, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(result.complete, [[False, False], [False, True]])
    np.testing.assert_array_equal(result.observed_rate_hz, [[0.0, np.nan], [np.nan, 40.0]])
    np.testing.assert_array_equal(result.supplied_unit_ids, [10, 12])
    assert np.isnan(counts[0, 0]) and counts[1, 1] == -999.0


def test_same_mapping_projects_observed_and_generated_multidimensional_activity():
    observer = mapping((2, 8, 13), ("left", "right", "left"))
    source = np.arange(18, dtype=np.int64).reshape(2, 3, 3)
    generated = source + 2
    observed = observer.project([([2, 8, 13], source, None)], bin_seconds=0.25)
    simulated = observer.project([([13, 2], generated[..., [2, 0]], None),
                                  ([8], generated[..., [1]], None)], bin_seconds=0.25)
    assert observed.count_sum.shape == (2, 3, 2)
    np.testing.assert_array_equal(observed.count_sum[..., 0], source[..., 0] + source[..., 2])
    np.testing.assert_array_equal(observed.count_sum[..., 1], source[..., 1])
    np.testing.assert_array_equal(simulated.count_sum - observed.count_sum, np.broadcast_to([4, 2], (2, 3, 2)))
    np.testing.assert_array_equal(observed.total_unit_count, simulated.total_unit_count)
    np.testing.assert_array_equal(observed.field_ids, simulated.field_ids)


def test_explicit_absence_retains_all_denominators_and_nan_rates():
    result = mapping().project([([], np.empty((2, 0)), None)], bin_seconds=1.0)
    np.testing.assert_array_equal(result.count_sum, np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(result.observed_unit_count, 0)
    np.testing.assert_array_equal(result.total_unit_count, [2, 2])
    assert np.isnan(result.observed_rate_hz).all()
    assert not result.complete.any()
    assert result.supplied_unit_ids.size == 0


def test_scalar_prefix_and_zero_observed_mask():
    result = mapping((1, 2), ("group", "group")).project(
        [([1, 2], np.array([np.nan, np.inf]), np.array([False, False]))], bin_seconds=1.0)
    assert result.count_sum.shape == (1,)
    np.testing.assert_array_equal(result.observed_unit_count, [0])
    assert np.isnan(result.observed_rate_hz[0]) and not result.complete[0]


@pytest.mark.parametrize("ids", [[1, 1], [1.0, 2.0], [1, True], [[1, 2]], []])
def test_invalid_fixed_source_ids_rejected(ids):
    with pytest.raises(ValueError):
        PopulationFieldMapping.from_groups(ids, ["group"] * len(ids), definition="source label")


@pytest.mark.parametrize("labels,definition", [(["A"], "source"), (["A", 2], "source"),
                                               (["A", " "], "source"), (["A", "B"], " ")])
def test_source_labels_and_definition_are_explicit(labels, definition):
    with pytest.raises(ValueError):
        PopulationFieldMapping.from_groups([1, 2], labels, definition=definition)


def test_fixed_mapping_arrays_do_not_allow_accidental_mutation():
    observer = mapping()
    for name in ("unit_ids", "field_ids", "group_index"):
        with pytest.raises(ValueError):
            getattr(observer, name)[0] = getattr(observer, name)[0]


@pytest.mark.parametrize("chunks", [
    [([42, 42], np.array([1, 2]), None)],
    [([42], np.array([1]), None), ([42], np.array([2]), None)],
    [([999], np.array([1]), None)],
    [([42], np.array([[1]]), None), ([7], np.array([2]), None)],
    [([42], np.array([1]), np.array([1]))],
    [([42], np.array([1]), np.array([[True]]))],
    [([42, 7], np.array([1]), None)],
    [],
])
def test_ambiguous_identity_shape_and_masks_rejected(chunks):
    with pytest.raises(ValueError):
        mapping().project(chunks, bin_seconds=1.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, 0.5, True, "1"])
def test_corrupt_observed_counts_are_not_coerced(value):
    with pytest.raises(ValueError):
        mapping().project([([42], np.array([value]), None)], bin_seconds=1.0)


@pytest.mark.parametrize("width", [0.0, -1.0, np.nan, np.inf, True, "0.1"])
def test_invalid_bin_width_rejected(width):
    with pytest.raises(ValueError):
        mapping().project([([42], np.array([1]), None)], bin_seconds=width)


def test_rate_overflow_rejected_instead_of_infinite_readout():
    with pytest.raises(ValueError, match="RATE"):
        mapping().project([([42], np.array([1]), None)], bin_seconds=np.nextafter(0.0, 1.0))


def test_large_representable_sum_is_exact_and_overflow_is_rejected():
    observer = mapping((1, 2), ("group", "group"))
    limit = np.iinfo(np.int64).max
    result = observer.project([([1, 2], np.array([limit, 0], dtype=np.int64), None)], bin_seconds=1.0)
    assert result.count_sum[0] == limit
    for chunks in ([([1, 2], np.array([limit, 1], dtype=np.int64), None)],
                   [([1], np.array([limit], dtype=np.int64), None), ([2], np.array([1]), None)],
                   [([1], np.array([2**63], dtype=np.uint64), None)]):
        with pytest.raises(ValueError):
            observer.project(chunks, bin_seconds=1.0)

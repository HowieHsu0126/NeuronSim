"""Counts preserve original source-block units and reject incomplete summaries."""
from copy import deepcopy

import pytest

from .pfc_history_profile_summary import aggregate_counts


def records():
    return [dict(source_rng_seed=seed, field_index=5, row_index=53223, count_threshold=1,
                 auxiliary_repeat="A", physical_window_denominator=10,
                 coarse_cdf_count=left, fine_cdf_count=right,
                 coarse_tail_count=10-left, fine_tail_count=10-right,
                 paired_discordant_windows=discordant, fine_minus_coarse_cdf_count=right-left,
                 fine_minus_coarse_empirical_cdf=(right-left)/10)
            for seed, left, right, discordant in [(95001, 8, 7, 3), (95002, 10, 9, 1)]]


def test_counts_retain_source_blocks_and_dependent_physical_windows():
    rows = records()
    original = deepcopy(rows)
    result, = aggregate_counts(rows, [95001, 95002])
    assert rows == original
    assert result["original_blocks"] == 2
    assert result["physical_window_denominator"] == 20
    assert result["coarse_tail_count"] == 2 and result["fine_tail_count"] == 4
    assert result["paired_discordant_windows"] == 4
    assert result["fine_minus_coarse_cdf_count"] == -2
    assert result["fine_minus_coarse_empirical_cdf"] == -.1
    assert result["source_law_probability_or_interval_established"] is False


@pytest.mark.parametrize("failure", ["duplicate_block", "changed_frame", "count_mismatch", "pair_mismatch"])
def test_incomplete_or_inconsistent_counts_do_not_aggregate(failure):
    rows = records()
    if failure == "duplicate_block":
        rows[1]["source_rng_seed"] = 95001
    elif failure == "changed_frame":
        rows[1]["physical_window_denominator"] = 11
    elif failure == "count_mismatch":
        rows[1]["fine_tail_count"] = 8
    else:
        rows[1]["fine_minus_coarse_cdf_count"] = 1
    with pytest.raises(ValueError, match="PROFILE_SUMMARY"):
        aggregate_counts(rows, [95001, 95002])

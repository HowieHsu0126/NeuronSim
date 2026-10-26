"""Contract checks for shared source identity and archived CDF measurements."""
from copy import deepcopy

import numpy as np
import pytest

from .pfc_history_profile_census import (
    endpoint_counts, frame_requests, unpack_pair, validate_source,
)
from .pfc_history_timing_census import census, pair
from .test_pfc_history_timing_census import FRAME, config, raw


def profile(left="numpy_dt_0p05", right="numpy_dt_0p025", row=1):
    return dict(left_cohort=left, right_cohort=right, condition="strong_cue",
                field_index=5, phase="cue", window_ms=5.,
                endpoints=[dict(row_index=row, count_threshold=1),
                           dict(row_index=row + 1, count_threshold=2)])


def test_shared_frames_preserve_both_clock_pairs_and_each_endpoint():
    profiles = [profile(), profile("numpy_dt_0p025", "numpy_dt_0p0125", row=3)]
    settings = dict(profiles=profiles, time_edges_ms=[0., 1., 2., 3.])
    original = deepcopy(settings)
    assert frame_requests(settings) == [FRAME]
    assert settings == original
    settings["profiles"][1]["endpoints"][0]["row_index"] = 1
    with pytest.raises(ValueError, match="PROFILE_ENDPOINT_IDENTITY"):
        frame_requests(settings)


def test_quarter_source_requires_original_native_parent_and_auxiliary_identity():
    cfg = config(dt=.0125, n_e=80000, n_i=20000)
    parent = dict(archive="native_source", delivery_sha256="native_delivery")
    report = dict(resolved_config=cfg, source_execution_complete=True,
                  selected_case=dict(source_rng_seed=95001, case_id="pfc_95001_strong_cue"),
                  status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE", coarse_dt_ms=.05,
                  refinement_seeds=[95411, 95412], drive=dict(refinement_factor=4),
                  coarse_reference=parent, completed_duration_ticks=400000)
    args = dict(seed=95001, label="quarter_A", expected_units=100000,
                refinement_seeds=[95411, 95412], coarse_reference=parent)
    validate_source(cfg, report, **args)
    for key, value in (("coarse_dt_ms", .025), ("refinement_seeds", [99411, 99412]),
                       ("drive", dict(refinement_factor=2))):
        changed = dict(report, **{key: value})
        with pytest.raises(ValueError, match="ORIGINAL_PARENT_OR_AUXILIARY_CHANGED"):
            validate_source(cfg, changed, **args)
    with pytest.raises(ValueError, match="NATIVE_PARENT_CHANGED"):
        validate_source(cfg, dict(report, coarse_reference=dict(parent, archive="half_clock")), **args)
    with pytest.raises(ValueError, match="SOURCE_TRUNCATED"):
        validate_source(cfg, dict(report, completed_duration_ticks=24000), **args)


def test_both_cdf_thresholds_keep_multispikes_and_paired_disagreement():
    arrays = dict(coarse_observed_spike_count=np.array([0, 1, 2, 3, 0, 0]),
                  fine_observed_spike_count=np.array([3, 2, 1, 3, 0, 2]))
    rows = endpoint_counts(profile(), arrays)
    assert [row["count_threshold"] for row in rows] == [1, 2]
    assert [(row["coarse_tail_count"], row["fine_tail_count"]) for row in rows] == [(2, 4), (1, 2)]
    assert [row["paired_discordant_windows"] for row in rows] == [4, 1]
    assert [row["fine_minus_coarse_cdf_count"] for row in rows] == [-2, -1]
    assert [row["physical_window_denominator"] for row in rows] == [6, 6]
    np.testing.assert_allclose([row["fine_minus_coarse_empirical_cdf"] for row in rows], [-2/6, -1/6])


def test_existing_pair_can_supply_each_clock_without_losing_physical_ids():
    coarse_cfg, fine_cfg = config(), config(dt=.025)
    coarse = census(coarse_cfg, raw(coarse_cfg, [(2000, 10), (2040, 10)]), **FRAME)
    fine = census(fine_cfg, raw(fine_cfg, [(4000, 11)]), **FRAME)
    summary, arrays = pair(*coarse, *fine)
    for clock, expected in (("coarse", coarse), ("fine", fine)):
        metadata, unpacked = unpack_pair(summary, arrays, clock)
        assert metadata == expected[0]
        for key, value in unpacked.items():
            np.testing.assert_array_equal(value, expected[1][key])
    damaged = dict(arrays, coarse_multispike=np.zeros_like(arrays["coarse_multispike"]))
    with pytest.raises(ValueError, match="TIMING_ASSIGNMENT_CHANGED"):
        unpack_pair(summary, damaged, "coarse")

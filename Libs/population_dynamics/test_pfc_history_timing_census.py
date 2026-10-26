"""Exact physical-frame fixtures; no PFC simulation or neural model is run."""
from __future__ import annotations

from copy import deepcopy
import json
from unittest.mock import Mock
import weakref

import numpy as np
import pytest
import yaml

from . import pfc_history_timing_census as timing_census
from .pfc_history_timing_census import census, census_many, pair, run_pair


FRAME = dict(field_index=5, phase="cue", window_ms=5., time_edges_ms=[0., 1., 2., 3.])


def config(dt=.05, n_e=32, n_i=8):
    return dict(excitatory_neurons=n_e, inhibitory_neurons=n_i,
                excitatory_preference_bins=16, dt_ms=dt, duration_ms=5000.,
                weight_scaling_factor=.0256)


def raw(cfg, events=()):
    events = sorted(events)
    ticks = np.asarray([tick for tick, _ in events], dtype=np.int64)
    units = np.asarray([unit for _, unit in events], dtype=np.int64)
    return dict(spike_ticks=ticks, spike_unit_indices=units,
                spike_seconds=ticks * (cfg["dt_ms"] * 1e-3),
                native_spike_counts=np.bincount(
                    units, minlength=cfg["excitatory_neurons"] + cfg["inhibitory_neurons"]))


def test_source_phase_and_half_open_window_boundaries_keep_full_denominator():
    cfg = config()
    meta, arrays = census(cfg, raw(cfg, [(1999, 10), (2000, 10), (2099, 10),
                                         (2100, 10), (5999, 11), (6000, 11),
                                         (2000, 12)]), **FRAME)
    np.testing.assert_array_equal(arrays["unit_indices"], [10, 11])
    assert meta["windows"] == 40 and meta["frame_size"] == 80
    assert meta["observed_spikes"] == 4
    assert meta["no_first_spike_windows"] == 77
    assert meta["refractory_zero_windows"] == 1
    assert meta["eligible_windows"] == 2
    assert meta["multispike_windows"] == 1
    assert arrays["observed_spike_count"][0] == 2
    assert arrays["first_spike_tick"][0] == 2000
    assert arrays["first_spike_tick"][2] == 2100
    assert arrays["first_spike_tick"][-1] == 5999
    assert arrays["time_bin"][-1] == -1
    np.testing.assert_array_equal(arrays["window_start_tick"][:2], [2000, 2100])
    np.testing.assert_array_equal(arrays["window_stop_tick"][-1:], [6000])


def test_exact_time_edges_and_both_structural_zero_reasons():
    for dt in (.05, .025):
        cfg = config(dt=dt, n_e=96)
        event_ms = [100., 101., 102., 103., 103.05]
        events = [(round(ms / dt), 30 + index) for index, ms in enumerate(event_ms)]
        meta, arrays = census(cfg, raw(cfg, events), **FRAME)
        np.testing.assert_array_equal(arrays["time_bin"][:6], [2, 1, 0, -1, -1, -1])
        assert np.all(arrays["refractory_ticks"] == round(2 / dt))
        np.testing.assert_allclose(arrays["post_refractory_remaining_ticks"][:6] * dt,
                                   [3., 2., 1., 0., -.05, 0.], rtol=0, atol=1e-15)
        assert arrays["first_spike_tick"][5] == -1
        assert meta["eligible_windows"] == 3
        assert meta["refractory_zero_windows"] == 2
        assert meta["no_first_spike_windows"] == 235
        assert meta["time_bin_counts"] == {"-1": 237, "0": 1, "1": 1, "2": 1}
    with pytest.raises(ValueError, match="WINDOW_OFF_CLOCK"):
        census(cfg, raw(cfg), **dict(FRAME, window_ms=5.01))
    with pytest.raises(ValueError, match="SOURCE_EXCITATORY_FIELD_REQUIRED"):
        census(cfg, raw(cfg), **dict(FRAME, field_index=16))
    malformed = raw(cfg)
    malformed["native_spike_counts"][0] = 1
    with pytest.raises(ValueError, match="MISSING_OR_EXTRA_SPIKE_EVENTS"):
        census(cfg, malformed, **FRAME)


def test_pair_preserves_multispike_histories_and_different_clock_zeros_and_firsts():
    coarse_cfg, fine_cfg = config(n_e=48), config(dt=.025, n_e=48)
    coarse = census(coarse_cfg, raw(coarse_cfg, [(2000, 15), (2040, 15), (2020, 17)]), **FRAME)
    fine = census(fine_cfg, raw(fine_cfg, [(4000, 16), (4080, 17)]), **FRAME)
    summary, arrays = pair(*coarse, *fine)
    assert summary["frame_size"] == 120 and summary["discarded_physical_windows"] == 0
    np.testing.assert_array_equal(arrays["joint_time_bin"][:3], [12, 3, 9])
    assert arrays["coarse_observed_spike_count"][0] == 2
    assert arrays["coarse_multispike"][0] and arrays["coarse_time_bin"][0] == 2
    assert arrays["fine_first_spike_tick"][0] == -1
    assert arrays["coarse_first_spike_tick"][1] == -1
    assert arrays["fine_first_spike_tick"][1] == 4000
    assert arrays["coarse_first_spike_tick"][2] * .05 != arrays["fine_first_spike_tick"][2] * .025
    assert summary["coarse_zero_fine_eligible_windows"] == 1
    assert summary["fine_zero_coarse_eligible_windows"] == 1
    assert summary["clocks"]["coarse"]["multispike_windows"] == 1
    assert sum(summary["joint_time_bin_counts"].values()) == 120
    assert not summary["population_confidence_interval_computed"]
    assert not summary["scientific_fidelity_qualified"]
    assert summary["new_neural_trajectories"] == 0


def test_all_sixteen_joint_bins_retain_exact_archived_frame_weights():
    coarse_cfg, fine_cfg = config(n_e=256), config(dt=.025, n_e=256)
    coarse_events, fine_events = [], []
    coarse_first = {0: 2040, 1: 2020, 2: 2000}
    fine_first = {0: 4080, 1: 4040, 2: 4000}
    for coarse_bin in (-1, 0, 1, 2):
        for fine_bin in (-1, 0, 1, 2):
            code = (coarse_bin + 1) * 4 + fine_bin + 1
            unit = 80 + code
            if coarse_bin >= 0:
                coarse_events.append((coarse_first[coarse_bin], unit))
            if fine_bin >= 0:
                fine_events.append((fine_first[fine_bin], unit))
    coarse = census(coarse_cfg, raw(coarse_cfg, coarse_events), **FRAME)
    fine = census(fine_cfg, raw(fine_cfg, fine_events), **FRAME)
    summary, arrays = pair(*coarse, *fine)
    np.testing.assert_array_equal(arrays["joint_time_bin"][:16], np.arange(16))
    assert summary["frame_size"] == 640
    expected = {str(i): 625 if i == 0 else 1 for i in range(16)}
    assert summary["joint_time_bin_counts"] == expected
    assert summary["joint_time_bin_weights"] == {key: count / 640 for key, count in expected.items()}
    assert sum(summary["joint_time_bin_counts"].values()) == 640
    assert summary["coarse_zero_fine_eligible_windows"] == 3
    assert summary["fine_zero_coarse_eligible_windows"] == 3


def test_pair_rejects_changed_physical_frame_and_source_unit_mapping():
    coarse_cfg, fine_cfg = config(), config(dt=.025)
    coarse = census(coarse_cfg, raw(coarse_cfg), **FRAME)
    fine = census(fine_cfg, raw(fine_cfg), **FRAME)
    changed = deepcopy(fine)
    changed[0]["window_ms"] = 10.
    with pytest.raises(ValueError, match="FULL_FRAME_DENOMINATOR"):
        pair(*coarse, *changed)
    changed = deepcopy(fine)
    changed[1]["window_start_tick"][0] += 1
    with pytest.raises(ValueError, match="WINDOW_GRID_CHANGED"):
        pair(*coarse, *changed)
    changed = deepcopy(fine)
    changed[1]["unit_indices"] = changed[1]["unit_indices"][::-1]
    with pytest.raises(ValueError, match="SOURCE_UNIT_MAPPING"):
        pair(*coarse, *changed)
    changed = deepcopy(fine)
    changed[1]["unit_indices"][1] = changed[1]["unit_indices"][0]
    with pytest.raises(ValueError, match="SOURCE_UNIT_MAPPING"):
        pair(*coarse, *changed)


def test_staged_run_requires_full_source_identity_and_exports_all_physical_ids(tmp_path):
    settings = dict(FRAME, expected_units=100000, source_rng_seed=95001,
                    case_id="pfc_95001_strong_cue",
                    clocks={"coarse": {"dt_ms": .05}, "fine": {"dt_ms": .025}})
    selected = dict(case_id=settings["case_id"], source_rng_seed=95001,
                    stimulus_strength_nA=.5, distractor_strength_nA=0.)
    for clock, dt in (("coarse", .05), ("fine", .025)):
        directory = tmp_path / clock
        directory.mkdir()
        cfg = config(dt=dt, n_e=80000, n_i=20000)
        (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
        (directory / "report.json").write_text(json.dumps(dict(resolved_config=cfg, selected_case=selected)))
        np.savez_compressed(directory / "raw.npz", **raw(cfg))
    output = tmp_path / "census"
    report = run_pair(settings, tmp_path / "coarse", tmp_path / "fine", output)
    assert report["frame_size"] == 200000 and report["field_units"] == 5000
    assert report["joint_time_bin_counts"] == {str(i): 200000 if i == 0 else 0 for i in range(16)}
    assert report["production_episodes"] == 0 and not report["confirmation_execution_released"]
    with np.load(output / "frame_assignments.npz", allow_pickle=False) as assignments:
        assert assignments["unit_indices"].shape == (5000,)
        assert assignments["joint_time_bin"].shape == (200000,)
        assert assignments["coarse_observed_spike_count"].shape == (200000,)
    assert json.loads((output / "report.json").read_text()) == report
    path = tmp_path / "fine" / "report.json"
    changed = json.loads(path.read_text())
    changed["selected_case"]["source_rng_seed"] = 95002
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SOURCE_CASE_MISMATCH:fine"):
        run_pair(settings, tmp_path / "coarse", tmp_path / "fine", tmp_path / "bad_identity")
    assert not (tmp_path / "bad_identity").exists()
    changed["selected_case"]["source_rng_seed"] = 95001
    changed["resolved_config"]["duration_ms"] = 4000.
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="SOURCE_CONFIG_REPORT_MISMATCH:fine"):
        run_pair(settings, tmp_path / "coarse", tmp_path / "fine", tmp_path / "bad_config")


@pytest.mark.parametrize("dt", [.05, .025, .0125])
def test_census_many_matches_single_across_fields_phases_and_late_windows(dt):
    cfg = config(dt=dt, n_e=48)
    physical_events = [
        (0., 16), (2., 16), (99.95, 15),
        (100., 15), (102., 15), (103., 16), (299.95, 17), (300., 15),
        (100., 18), (102., 18), (105., 19), (299.95, 20),
        (300., 18), (302., 18), (499.95, 20),
        (3999.95, 18), (4000., 18), (4002., 18),
        (4995., 19), (4999.95, 20), (4000., 29), (4000., 48),
    ]
    source_raw = raw(cfg, [(round(ms / dt), unit) for ms, unit in physical_events])
    requests = [
        dict(FRAME, field_index=6, phase="late_delay"),
        dict(FRAME),
        dict(FRAME, field_index=6, phase="early_delay"),
        dict(FRAME, field_index=6, window_ms=10., time_edges_ms=[0., 2., 4., 8.]),
        dict(FRAME, phase="baseline"),
    ]
    results = census_many(cfg, source_raw, requests)
    for request in requests:
        actual_meta, actual_arrays = next(results)
        expected_meta, expected_arrays = census(cfg, source_raw, **request)
        assert actual_meta == expected_meta
        assert actual_arrays.keys() == expected_arrays.keys()
        for key in actual_arrays:
            assert actual_arrays[key].dtype == expected_arrays[key].dtype
            np.testing.assert_array_equal(actual_arrays[key], expected_arrays[key])
        if request["phase"] == "late_delay":
            assert actual_meta["windows"] == 200 and actual_meta["frame_size"] == 600
            assert actual_meta["observed_spikes"] == 4
            assert actual_meta["multispike_windows"] == 1
            assert actual_meta["no_first_spike_windows"] == 597
            assert actual_arrays["observed_spike_count"][0] == 2
            assert actual_arrays["first_spike_tick"][0] == round(4000. / dt)
            assert actual_arrays["first_spike_tick"][-1] == round(4999.95 / dt)
            assert actual_arrays["window_stop_tick"][-1] == round(5000. / dt)
    with pytest.raises(StopIteration):
        next(results)


def test_census_many_streams_requests_and_releases_arrays_with_one_source_validation(monkeypatch):
    cfg = config()
    source_raw = raw(cfg)
    validated = Mock(wraps=timing_census.validate_tape)
    mapped = Mock(wraps=timing_census.unit_mapping)
    monkeypatch.setattr(timing_census, "validate_tape", validated)
    monkeypatch.setattr(timing_census, "unit_mapping", mapped)
    visited = []

    def requests():
        for phase in ("cue", "late_delay"):
            visited.append(phase)
            yield dict(FRAME, phase=phase)

    results = census_many(cfg, source_raw, requests())
    assert iter(results) is results
    assert visited == [] and validated.call_count == mapped.call_count == 0
    first = next(results)
    assert first[0]["phase"] == "cue" and visited == ["cue"]
    assert validated.call_count == mapped.call_count == 1
    previous_arrays = [weakref.ref(value) for value in first[1].values()]
    del first
    second = next(results)
    assert second[0]["phase"] == "late_delay" and visited == ["cue", "late_delay"]
    assert all(reference() is None for reference in previous_arrays)
    with pytest.raises(StopIteration):
        next(results)
    assert validated.call_count == 1
    mapped.assert_called_once_with(32, 8, 16)


def test_census_many_empty_requests_validate_source_and_yield_nothing(monkeypatch):
    cfg = config()
    validated = Mock(wraps=timing_census.validate_tape)
    mapped = Mock(wraps=timing_census.unit_mapping)
    monkeypatch.setattr(timing_census, "validate_tape", validated)
    monkeypatch.setattr(timing_census, "unit_mapping", mapped)
    assert list(census_many(cfg, raw(cfg), ())) == []
    assert validated.call_count == mapped.call_count == 1


@pytest.mark.parametrize("requests", [(), (FRAME,)])
def test_census_many_rejects_malformed_raw_even_without_requests(requests):
    cfg = config()
    malformed = raw(cfg)
    malformed["native_spike_counts"][0] = 1
    with pytest.raises(ValueError, match="MISSING_OR_EXTRA_SPIKE_EVENTS"):
        list(census_many(cfg, malformed, requests))


def test_census_many_duplicate_requests_have_independent_equal_results():
    cfg = config()
    results = census_many(cfg, raw(cfg), (FRAME, FRAME))
    first_meta, first_arrays = next(results)
    second_meta, second_arrays = next(results)
    assert first_meta == second_meta
    assert first_arrays.keys() == second_arrays.keys()
    for key in first_arrays:
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        assert not np.shares_memory(first_arrays[key], second_arrays[key])
    first_arrays["observed_spike_count"][0] = 99
    assert second_arrays["observed_spike_count"][0] == 0
    with pytest.raises(StopIteration):
        next(results)

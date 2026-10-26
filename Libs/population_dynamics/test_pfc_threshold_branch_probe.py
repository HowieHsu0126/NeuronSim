"""Bounded software tests; synthetic archives are not scientific evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import yaml

from . import pfc_threshold_branch_probe as probe
from . import pfc_deterministic_replay as executor
from .pfc_clock_temporal_control import create_temporal_reversal_drive
from .pfc_coupled_drive import create_coupled_drive
from .provenance import freeze_json, sha256_file
from .test_pfc_coupled_drive import _case, _config, _equal_arrays


def _events(rows, units=40, ticks=1000):
    values = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    return probe.canonical_events(values[:, 0], values[:, 1], units, ticks)


@pytest.mark.parametrize("original,reversed_rows,tick,a_only,b_only", [
    ([(3, 2), (1, 1)], [(1, 1), (3, 2)], None, [], []),
    ([], [], None, [], []),
    ([], [(2, 3)], 2, [], [3]),
    ([(1, 1)], [], 1, [1], []),
    ([(1, 1)], [(1, 1), (8, 2)], 8, [], [2]),
    ([(1, 1), (8, 2)], [(1, 1)], 8, [2], []),
    ([(2, 7), (2, 3), (1, 1)], [(2, 8), (1, 1), (2, 3)], 2, [7], [8]),
    ([(3, 1)], [(4, 1)], 3, [1], []),
])
def test_first_set_difference_handles_order_empty_and_unequal_endings(original, reversed_rows, tick, a_only, b_only):
    result = probe.first_spike_difference(_events(original), _events(reversed_rows))
    if tick is None:
        assert result is None
    else:
        assert result["tick"] == tick
        assert result["original_only_unit_indices"] == a_only
        assert result["reversed_only_unit_indices"] == b_only


def test_duplicate_events_are_rejected_instead_of_hidden_by_set_conversion():
    with pytest.raises(ValueError, match="DUPLICATE_UNIT_TICK_EVENT"):
        _events([(3, 1), (1, 2), (3, 1)])


@pytest.mark.parametrize("ticks,ids", [
    (np.array([1.0]), np.array([1])),
    (np.array([1, 2]), np.array([1])),
    (np.array([[-1]]), np.array([[1]])),
    (np.array([-1]), np.array([1])),
    (np.array([1000]), np.array([1])),
    (np.array([1]), np.array([40])),
])
def test_invalid_event_shapes_types_and_ranges_reject(ticks, ids):
    with pytest.raises(ValueError):
        probe.canonical_events(ticks, ids, 40, 1000)


@pytest.mark.parametrize("tick,stop", [(0, 400), (399, 400), (400, 800), (7999, 8000)])
def test_prefix_ends_at_exclusive_end_of_containing_five_ms_bin(tick, stop):
    assert probe.prefix_stop_tick(tick, .0125, 5., 100.) == stop


def test_prefix_cap_does_not_silently_truncate_or_extend():
    with pytest.raises(ValueError, match="EXCEEDS_PREFIX_CAP"):
        probe.prefix_stop_tick(8000, .0125, 5., 100.)
    with pytest.raises(ValueError, match="OFF_CLOCK"):
        probe.prefix_stop_tick(0, .03, 5., 100.)


@pytest.mark.parametrize("factory", [create_coupled_drive, create_temporal_reversal_drive])
def test_actual_source_trace_on_and_off_have_exact_raw_and_terminal_arrays(factory):
    # Keep the source drive's full nominal duration while executing only 8 ms.
    full_config, case = _config(4, 5000.0), _case()
    config = dict(full_config, duration_ms=8.0)
    initial, plain_drive = factory(full_config, case, .05, [101, 102])
    plain = executor.simulate(config, case, initial, backend="numpy", drive=plain_drive)
    traced_initial, traced_drive = factory(full_config, case, .05, [101, 102])
    observer = probe.ThresholdObserver(100, config)
    previous = sys.gettrace()
    with observer.observing():
        traced = executor.simulate(config, case, traced_initial, backend="numpy", drive=traced_drive)
    assert sys.gettrace() is previous
    for family in ("raw", "terminal_state"):
        _equal_arrays(traced[family], plain[family])
    assert plain["raw"]["spike_ticks"].size > 0
    assert set(observer.snapshots) == {"pre_rk2", "pre_threshold"}
    assert traced_drive.total_ticks == 400000
    assert traced_drive.ticks == 640 and traced_drive.remaining > 0
    for stage in observer.snapshots.values():
        assert stage["E__v"].shape == (32,)
        assert stage["I__v"].shape == (8,)
        assert stage["E__not_refractory"].dtype == bool
        assert np.all(np.abs(stage["E__v"]) < 1)  # SI volts, not mV.
    _equal_arrays(traced_drive._rows.state_arrays(), plain_drive._rows.state_arrays())


def test_trace_restores_previous_trace_when_observer_raises(monkeypatch):
    config, case = _config(4, .05), _case()
    initial, drive = create_coupled_drive(config, case, .05, [101, 102])
    observer = probe.ThresholdObserver(0, config)

    def fail(stage, local):
        raise RuntimeError("injected snapshot failure")

    def previous_trace(frame, event, arg):
        return None

    monkeypatch.setattr(observer, "capture", fail)
    saved = sys.gettrace()
    try:
        sys.settrace(previous_trace)
        with pytest.raises(RuntimeError, match="injected snapshot failure"):
            with observer.observing():
                executor.simulate(config, case, initial, backend="numpy", drive=drive)
        assert sys.gettrace() is previous_trace
    finally:
        sys.settrace(saved)


def test_actual_source_current_tick_external_input_enters_next_rk2():
    config, case = _config(4, .05), _case()
    initial = np.full(40, -55.0, dtype=np.float64)
    counts_e, counts_i = np.zeros((4, 32), dtype=np.uint16), np.zeros((4, 8), dtype=np.uint16)
    counts_e[0, 0] = 20
    snapshots = []
    for tick in (0, 1):
        observer = probe.ThresholdObserver(tick, config)
        with observer.observing():
            executor.simulate(config, case, initial, counts_e, counts_i, backend="numpy")
        snapshots.append(observer.snapshots)
    assert snapshots[0]["pre_threshold"]["E__s_AMPA"][0] == 0.0
    assert snapshots[1]["pre_rk2"]["E__s_AMPA"][0] == 20.0
    assert snapshots[1]["pre_threshold"]["E__s_AMPA"][0] < 20.0


def test_observer_anchors_require_exact_source_statement(monkeypatch):
    anchors = probe.observer_anchors()
    assert anchors["pre_rk2"]["line"] < anchors["pre_threshold"]["line"]
    monkeypatch.setitem(probe.ANCHORS, "pre_rk2", "_rk2(e, wrong_constants, t, dt, arithmetic, 'E')")
    with pytest.raises(ValueError, match="ANCHOR_CHANGED"):
        probe.observer_anchors()


def _branch_snapshots():
    result = {}
    for variant in ("original", "reversed"):
        result[variant] = {}
        for stage in probe.ANCHORS:
            arrays = {}
            for population, size in (("E", 2), ("I", 1)):
                for field in probe.STATE_FIELDS + (probe.E_FIELDS if population == "E" else ()):
                    value = np.zeros(size, dtype=bool if field == "not_refractory" else np.float64)
                    if field == "v":
                        value[:] = -.060
                    if field == "not_refractory":
                        value[:] = True
                    arrays[population + "__" + field] = value
            result[variant][stage] = arrays
    result["original"]["pre_threshold"]["E__v"][0] = -.050
    result["reversed"]["pre_threshold"]["E__v"][0] = np.nextafter(-.050, np.inf)
    return result


def test_actual_strict_branch_requires_voltage_crossing_and_refractory_eligibility():
    a, b = _events([]), _events([(0, 0)])
    difference = probe.first_spike_difference(a, b)
    snapshots = _branch_snapshots()
    config = dict(excitatory_neurons=2, inhibitory_neurons=1)
    analysis = probe._branch_analysis(snapshots, a, b, difference, config)
    unit = analysis["differing_units"][0]
    assert unit["original"]["threshold_margin_V"] == 0
    assert unit["original"]["spike_branch"] is False
    assert unit["reversed"]["spike_branch"] is True
    assert unit["two_path_voltage_span_V"] > 0
    assert analysis["one_frozen_spike_schedule_encloses_both_observed_paths"] is False
    snapshots["reversed"]["pre_threshold"]["E__not_refractory"][0] = False
    with pytest.raises(ValueError, match="REFRACTORY_HISTORY_DIFFERS"):
        probe._branch_analysis(snapshots, a, b, difference, config)


def test_observer_copies_and_validates_full_state_fields():
    config = dict(excitatory_neurons=2, inhibitory_neurons=1)
    observer = probe.ThresholdObserver(0, config)
    arrays = _branch_snapshots()["original"]["pre_rk2"]
    local = {short: {key.split("__", 1)[1]: value for key, value in arrays.items()
                     if key.startswith(population + "__")}
             for short, population in (("e", "E"), ("i", "I"))}
    observer.capture("pre_rk2", local)
    local["e"]["v"][0] = 99.0
    assert observer.snapshots["pre_rk2"]["E__v"][0] == -.060
    local["i"]["v"] = np.zeros(2)
    with pytest.raises(ValueError, match="SHAPE_OR_DTYPE"):
        observer.capture("pre_threshold", local)


def _settings():
    return dict(schema=probe.SCHEMA, case_relative_path="source_cases/dt_0p0125/pfc_95001_omission",
                source_rng_seed=95001, refinement_seeds=[95411, 95412], coarse_dt_ms=.05,
                dt_ms=.0125, expected_units=100000, prefix_window_ms=5., maximum_prefix_ms=100.,
                confirmation_execution_released=False, production_episodes=0,
                original_source=dict(archive="original", delivery_sha256="a" * 64),
                reversed_source=dict(archive="reversed", delivery_sha256="b" * 64))


@pytest.mark.parametrize("key,value", [("source_rng_seed", 95002), ("refinement_seeds", [95411, 95413]),
                                      ("expected_units", 40), ("maximum_prefix_ms", 101.),
                                      ("confirmation_execution_released", True), ("production_episodes", 1),
                                      ("dt_ms", .025), ("schema", "other")])
def test_frozen_settings_do_not_drift(key, value):
    settings = _settings()
    settings[key] = value
    with pytest.raises(ValueError):
        probe.validate_settings(settings)


def _synthetic_archives(tmp_path):
    settings = _settings()
    config = dict(_config(4, 5000.0), excitatory_neurons=80000, inhibitory_neurons=20000,
                  readout_bin_ms=5., excitatory_preference_bins=16, source_rng_seed=95001)
    case = dict(case_id="pfc_95001_omission", source_rng_seed=95001,
                stimulus_strength_nA=0., distractor_strength_nA=0.)
    original_report_digest = None
    for variant, rows in (("original", [(399, 5), (402, 8)]),
                          ("reversed", [(400, 5), (402, 8)])):
        root = tmp_path / variant
        directory = root / settings["case_relative_path"]
        directory.mkdir(parents=True)
        initial = np.full(100000, -55.0, dtype=np.float64)
        np.save(directory / "initial_voltage.npy", initial)
        events = _events(rows, units=100000)
        np.savez(directory / "raw.npz", spike_ticks=events[:, 0], spike_unit_indices=events[:, 1])
        outputs = {name: probe._file_record(directory / name) for name in ("raw.npz", "initial_voltage.npy")}
        report = dict(status=("COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE" if variant == "original"
                              else "COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE"),
                      source_execution_complete=True, case_id=case["case_id"], condition="omission",
                      source_rng_seed=95001, refinement_seeds=[95411, 95412], dt_ms=.0125,
                      coarse_dt_ms=.05, duration_ms=5000., coupled_core_units=100000,
                      confirmation_execution_released=False, production_episodes=0,
                      validations=dict(software_fixture=dict(passed=True)), resolved_config=config,
                      selected_case=case, outputs=outputs, code=probe._code_bindings(), drive=dict(coupling={}))
        if variant == "reversed":
            report.update(temporal_transform=probe.TRANSFORM,
                          original_fine_input_agreement={key: True for key in probe.PAIRED_INVARIANTS},
                          original_fine_reference=settings["original_source"],
                          original_fine_report_sha256=original_report_digest)
            report["drive"]["coupling"]["input_order_control"] = dict(
                name=probe.TRANSFORM, middle_row_order=[1, 0], fine_row_order=[3, 2, 1, 0],
                unit_axis_permuted=False, rng_draw_order_changed=False)
        freeze_json(directory / "report.json", report)
        if variant == "original":
            original_report_digest = sha256_file(directory / "report.json")
        files = [dict(path=settings["case_relative_path"] + "/" + name,
                      **probe._file_record(directory / name))
                 for name in ("raw.npz", "initial_voltage.npy", "report.json")]
        freeze_json(root / "delivery.json", dict(verified=True, files=files,
                    status=("COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT" if variant == "original"
                            else "COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL")))
        settings[variant + "_source"]["delivery_sha256"] = sha256_file(root / "delivery.json")
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    return config_path, settings


def test_preparation_binds_sources_and_saves_only_self_contained_prefixes(tmp_path):
    config_path, _ = _synthetic_archives(tmp_path)
    output = tmp_path / "prepared"
    report = probe.prepare(config_path, tmp_path, output)
    assert report["status"] == probe.PREPARED, report
    assert report["prefix_duration_ms"] == 5.0
    plan, paths = probe._load_prepared(output)
    assert plan["full_config"]["duration_ms"] == 5000.0
    assert plan["first_mismatch"]["tick"] == 399
    assert set(report["outputs"]) == {"plan.json", "initial_voltage.npy", "original_events.npz", "reversed_events.npz"}
    with np.load(paths["original_events.npz"]) as data:
        assert data["spike_ticks"].tolist() == [399]
        assert set(data.files) == {"spike_ticks", "spike_unit_indices"}
    with np.load(paths["reversed_events.npz"]) as data:
        assert data["spike_ticks"].size == 0
    # Prepared execution never needs the source archive to resolve its inputs.
    for variant in ("original", "reversed"):
        (tmp_path / variant).rename(tmp_path / (variant + "_unavailable"))
    probe._load_prepared(output)
    with paths["original_events.npz"].open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="BOUND_FILE_CHANGED"):
        probe._load_prepared(output)


def test_changed_source_binding_is_failure_with_persisted_receipt(tmp_path):
    config_path, settings = _synthetic_archives(tmp_path)
    settings["original_source"]["delivery_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(settings))
    output = tmp_path / "failed"
    report = probe.prepare(config_path, tmp_path, output)
    assert report["status"] == probe.FAILED
    assert "DELIVERY_CHANGED" in report["error"]["message"]
    assert json.loads((output / "report.json").read_text()) == report
    assert report["auxiliary_event"] is None
    assert report["conditional_failure_probability"] is None
    assert report["new_independent_blocks"] == report["new_full_episodes"] == 0


def test_frozen_schedule_witness_does_not_require_different_window_counts():
    # These exact schedules differ, but both units have the same count in 5 ms.
    a, b = _events([(10, 0), (20, 1)]), _events([(11, 0), (20, 1)])
    difference = probe.first_spike_difference(a, b)
    snapshots = _branch_snapshots()
    snapshots["original"]["pre_threshold"]["E__v"][0] = np.nextafter(-.050, np.inf)
    snapshots["reversed"]["pre_threshold"]["E__v"][0] = -.050
    result = probe._branch_analysis(snapshots, a, b, difference,
                                   dict(excitatory_neurons=2, inhibitory_neurons=1))
    assert result["one_frozen_spike_schedule_encloses_both_observed_paths"] is False
    np.testing.assert_array_equal(np.bincount(a[:, 1]), np.bincount(b[:, 1]))

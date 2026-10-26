"""Pulse-support conservation, causal windows and Opto CLI boundaries."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from . import __main__ as population_cli
from . import opto_windows
from .config import load_config
from .opto_windows import build_pulse_tables, encode_fixed_control_windows
from .spike_windows import bin_spikes


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "neural_population_dynamics.yaml"


def trial(trial_id, start, duration=0.5, pulses=1, gap=0.0, power=0.1, stim_id="wave"):
    """Explicit seconds; stop is the independently declared train extent."""
    return {"id": trial_id, "start_time": start,
            "stop_time": start + pulses * duration + (pulses - 1) * gap,
            "emission_location": "tip", "stimulation_type": "pulse", "wavelength": 470.0,
            "site": 0, "power_label": power, "duration_s": duration, "rise_s": 0.0,
            "num_pulses": pulses, "gap_s": gap, "stim_id": stim_id}


def pulse_tables(*rows):
    return build_pulse_tables(pd.DataFrame(rows))


def condition_for(tables, trial_id):
    return tables.trials.set_index("trial_id").loc[trial_id, "condition_id"]


def feature_for(encoded, condition):
    return next(row["feature_index"] for row in encoded.features if row["condition_id"] == condition)


def test_crossing_pulse_conserves_exposure_across_history_boundary():
    tables = pulse_tables(trial(17, 0.75, duration=0.5), trial(21, 2.0, duration=0.25, power=0.2))
    edges = np.array([[0.0, 0.5, 1.0, 1.5, 2.0]])
    encoded = encode_fixed_control_windows(tables, edges, tables.catalog)
    crossing = feature_for(encoded, condition_for(tables, 17))
    outside = feature_for(encoded, condition_for(tables, 21))
    np.testing.assert_array_equal(encoded.values[0, :, crossing], [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(encoded.values[0, :, outside], 0.0)
    np.testing.assert_array_equal(encoded.values[0, :, 0], [1.0, 0.5, 0.5, 1.0])
    exposure_seconds = encoded.values[0, :, crossing] * np.diff(edges[0])
    assert exposure_seconds[:2].sum() == 0.25
    assert exposure_seconds[2:].sum() == 0.25
    assert exposure_seconds.sum() == 0.5
    np.testing.assert_array_equal(encoded.overlap_fraction, 0.0)


def test_neighboring_and_concurrent_trains_retain_source_ids_and_exposure():
    tables = pulse_tables(trial(70, 0.0, duration=0.75),
                          trial(90, 0.25, duration=0.5, pulses=2, gap=0.25, power=0.2),
                          trial(110, 0.5, duration=0.25, power=0.3))
    assert tables.presentations.presentation_id.tolist() == ["70:0", "90:0", "90:1", "110:0"]
    assert tables.presentations.source_trial_id.tolist() == [70, 90, 90, 110]
    assert tables.members.trial_id.tolist() == [70, 90, 90, 110]
    np.testing.assert_array_equal(tables.presentations.start_time, [0.0, 0.25, 1.0, 0.5])
    encoded = encode_fixed_control_windows(tables, np.array([[0.0, 1.0, 2.0]]), tables.catalog)
    for source_id, expected in ((70, [0.75, 0.0]), (90, [0.5, 0.5]), (110, [0.25, 0.0])):
        np.testing.assert_array_equal(encoded.values[0, :, feature_for(encoded, condition_for(tables, source_id))], expected)
    np.testing.assert_array_equal(encoded.values[0, :, 0], [0.25, 0.5])
    np.testing.assert_array_equal(encoded.overlap_fraction, [[0.5, 0.0]])
    # Simultaneous annotations accumulate separately; they are not summed powers.
    assert encoded.values[0, 0, 1:].sum() == 1.5


def test_fixed_catalog_keeps_history_coordinates_when_future_control_changes():
    historical = trial(10, 0.0, power=0.1)
    future_a = trial(20, 1.0, power=0.2)
    future_b = trial(20, 1.0, power=0.3)
    registered_b = {**future_b, "id": 30, "start_time": 2.0, "stop_time": 2.5}
    catalog = pulse_tables(historical, future_a, registered_b).catalog
    edges = np.array([[0.0, 0.5, 1.0, 1.5, 2.0]])
    first = encode_fixed_control_windows(pulse_tables(historical, future_a), edges, catalog)
    second = encode_fixed_control_windows(pulse_tables(historical, future_b), edges, catalog)
    no_future = encode_fixed_control_windows(pulse_tables(historical), edges, catalog)
    assert first.features == second.features == no_future.features
    np.testing.assert_array_equal(first.values[:, :2, :], second.values[:, :2, :])
    np.testing.assert_array_equal(first.values[:, :2, :], no_future.values[:, :2, :])
    assert not np.array_equal(first.values[:, 2:, :], second.values[:, 2:, :])
    assert first.values.shape[-1] == len(catalog) + 1


def test_catalog_and_control_encoding_ignore_source_row_order():
    frame = pd.DataFrame([trial(12, 0.0, power=0.3), trial(5, 1.0, power=0.1)])
    first = build_pulse_tables(frame)
    second = build_pulse_tables(frame.iloc[::-1].copy())
    pd.testing.assert_frame_equal(first.catalog, second.catalog)
    edges = np.array([[0.0, 0.5, 1.0, 1.5]])
    a = encode_fixed_control_windows(first, edges, first.catalog)
    b = encode_fixed_control_windows(second, edges, first.catalog)
    assert a.features == b.features
    np.testing.assert_array_equal(a.values, b.values)


def test_unregistered_future_control_is_rejected():
    registered = pulse_tables(trial(10, 0.0, power=0.1))
    changed = pulse_tables(trial(10, 0.0, power=0.1), trial(20, 1.0, power=0.4))
    with pytest.raises(ValueError, match="CATALOG"):
        encode_fixed_control_windows(changed, np.array([[0.0, 1.0, 2.0]]), registered.catalog)


@pytest.mark.parametrize("mutation", ["span", "milliseconds_as_seconds", "fractional_pulses", "negative_gap"])
def test_invalid_train_timing_is_rejected(mutation):
    row = trial(14, 1.0, duration=0.01, pulses=3, gap=0.005)
    if mutation == "span":
        row["stop_time"] += 0.001
    elif mutation == "milliseconds_as_seconds":
        row["duration_s"] *= 1000.0
        row["gap_s"] *= 1000.0
    elif mutation == "fractional_pulses":
        row["num_pulses"] = 2.5
    else:
        row["gap_s"] = -0.001
    with pytest.raises(ValueError):
        pulse_tables(row)


def test_half_open_spike_boundary_and_window_total_conservation():
    spikes = {7: np.array([0.0, 0.5, 1.0, 1.5, 2.0]), 4: np.array([], dtype=float)}
    windows = bin_spikes(spikes, [7, 4], np.array([1.0]), history_seconds=1.0,
                         horizon_seconds=1.0, bin_seconds=0.5)
    np.testing.assert_array_equal(windows.counts[0, :, 0], [1, 1, 1, 1])
    np.testing.assert_array_equal(windows.counts[0, :, 1], 0)
    np.testing.assert_array_equal(windows.unit_ids, [7, 4])
    assert windows.history.sum() == 2
    assert windows.response.sum() == 2
    # Boolean interval oracle does not depend on the implementation's searchsorted.
    for index, unit in enumerate(windows.unit_ids):
        values = spikes[int(unit)]
        assert windows.counts[0, :, index].sum() == np.count_nonzero((values >= 0.0) & (values < 2.0))


def test_changed_future_spikes_leave_historical_counts_unchanged():
    common_history = np.array([0.0, 0.25, 0.5, 0.75])
    first = {4: np.concatenate([common_history, [1.0, 1.5]])}
    altered = {4: np.concatenate([common_history, [1.0, 1.0, 1.1, 1.2, 1.9]])}
    settings = {"history_seconds": 1.0, "horizon_seconds": 1.0, "bin_seconds": 0.25}
    a = bin_spikes(first, [4], np.array([1.0]), **settings)
    b = bin_spikes(altered, [4], np.array([1.0]), **settings)
    np.testing.assert_array_equal(a.history, b.history)
    assert not np.array_equal(a.response, b.response)


@pytest.fixture
def campaign_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def write_config(tmp_path, config):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def cli_args(config_path, tmp_path):
    return ["--config", str(config_path), "--project-root", str(tmp_path), "opto-window-check",
            "--source-dir", "source", "--control-dir", "controls", "--trial-ids", "0", "2",
            "--output-dir", "windows"]


@pytest.mark.parametrize("flag", ["--source-dir", "--control-dir", "--trial-ids", "--output-dir"])
def test_cli_requires_explicit_inputs_and_trial_selection(flag, tmp_path):
    args = cli_args(CONFIG, tmp_path)
    start = args.index(flag)
    end = start + 1
    while end < len(args) and not args[end].startswith("--"):
        end += 1
    del args[start:end]
    with pytest.raises(SystemExit) as error:
        population_cli.parser().parse_args(args)
    assert error.value.code == 2


@pytest.mark.parametrize("field,value", [
    ("history_seconds", float("nan")), ("horizon_seconds", float("inf")),
    ("bin_seconds", float("-inf")), ("history_seconds", True), ("horizon_seconds", False),
    ("bin_seconds", True), ("history_seconds", 0.0), ("horizon_seconds", -1.0),
    ("bin_seconds", "0.001"), ("history_seconds", 1.0005),
    ("horizon_seconds", 0.0015), ("bin_seconds", 0.003),
])
def test_optional_window_config_rejects_invalid_numbers_and_fractional_bins(tmp_path, campaign_config, field, value):
    campaign_config["opto_window_check"][field] = value
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, campaign_config))


def test_missing_window_config_remains_valid_for_other_commands(tmp_path, campaign_config):
    del campaign_config["opto_window_check"]
    config_path = write_config(tmp_path, campaign_config)
    assert "opto_window_check" not in load_config(config_path)
    args = population_cli.parser().parse_args(cli_args(config_path, tmp_path))
    with pytest.raises(ValueError, match="OPTO_WINDOW_CHECK_CONFIG_REQUIRED"):
        population_cli.run(args, ["python", "-m", "population_dynamics"])


def test_cli_dispatches_before_visual_cohort_access(tmp_path, campaign_config, monkeypatch):
    config_path = write_config(tmp_path, campaign_config)
    recorded = []
    result = {"status": "PARTIAL", "dataset_ready": False, "physical_power_qualified": False}
    def fake_check(source, control, output, config, command, trial_ids):
        recorded.append((source, control, output, copy.deepcopy(config), command, trial_ids))
        return result, 2
    monkeypatch.setattr(opto_windows, "check_opto_windows", fake_check)
    args = population_cli.parser().parse_args(cli_args(config_path, tmp_path))
    command = ["python", "-m", "population_dynamics", "opto-window-check"]
    assert population_cli.run(args, command) == (result, 2)
    assert len(recorded) == 1
    source, control, output, config, received_command, selected = recorded[0]
    assert (source, control, output) == (tmp_path / "source", tmp_path / "controls", tmp_path / "windows")
    assert received_command == command
    assert selected == [0, 2]
    assert config["opto_window_check"] == campaign_config["opto_window_check"]

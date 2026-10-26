"""Readout windows, silent units, complete blocks and condition identity."""
from dataclasses import replace
import json

import numpy as np
import pytest

from .benchmark_tasks import task_protocol
from .circuit_data import CircuitArchiveWriter
from .test_circuit_data import binding, episode, units
from .v1_benchmark_readout import CONDITIONS, describe_episode, readout, selectivity
from .v1_visual_controls import CONTROL_FEATURES, make_commands


def visual_episode(name="direction_000", *, block="pfc_a_block_0", contrast=80, direction=0):
    stimulus = dict(duration_s=3., gray_screen_s=.5, spatial_frequency_cycles_per_degree=.04,
                    temporal_frequency_hz=2., direction_deg=direction, contrast_percent=contrast)
    config = dict(stimulus=stimulus, spike_generation=dict(starting_noise_seed=7),
                  background=dict(starting_noise_seed=8))
    commands = make_commands(stimulus)
    original = episode(name, "development")
    return replace(original, block_id=block, duration_ticks=12000, dt_ms=.25,
        spike_ticks=np.array([2000, 2001, 2003, 2004, 11999, 12000]),
        spike_unit_indices=np.array([0, 1, 1, 0, 1, 1]),
        control_ticks=commands["control_ticks"], control_values=commands["control_values"],
        control_features=CONTROL_FEATURES, seeds=dict(native_kernel_rng_seed=1,
            lgn_starting_noise_seed=7, background_starting_noise_seed=8),
        source_record=dict(original.source_record, recorded_until_tick=12000, expected_duration_ticks=12000,
            input_mode="constructed_visual_inputs", initial_vm_identity="fixture-initial-voltage",
            upstream_bindings=dict(spike_inputs=dict(config=config)),
            visual_commands=dict(source_command_representation_verified=True, config=config)))


def archive(path, values):
    with CircuitArchiveWriter(path, dict(binding(), scenario_id="v1_visual")) as writer:
        writer.register_circuit(units())
        for value in values:
            writer.add(value)
    return path


def conditions(block="pfc_a_block_0"):
    return [visual_episode(name + "_" + block, block=block, direction=index * 45) for index, name in enumerate(CONDITIONS[:8])] + [
        visual_episode("gray_" + block, block=block, contrast=0),
        visual_episode("low_" + block, block=block, contrast=20, direction=45)]


def test_source_and_delivered_control_windows_keep_terminal_and_silent_units():
    result = describe_episode(visual_episode(), units())
    assert result["source_baseline_counts"].tolist() == [1, 0, 0]
    assert result["source_response_counts"].tolist() == [1, 4, 0]
    assert result["pre_command_counts"].tolist() == [1, 2, 0]
    assert result["command_response_counts"].tolist() == [1, 1, 0]
    assert result["terminal_counts"].tolist() == [0, 1, 0]
    assert result["population_counts"].shape == (2, 12000)
    assert result["population_counts"].sum() == 5
    assert result["task_outcome"] == "failed_memory"


def test_circular_selectivity_has_defined_direction_orientation_and_silence():
    rates = np.zeros((8, 4)); rates[:, 0] = 1; rates[0, 1:3] = 1; rates[4, 2] = 1
    result = selectivity(rates)
    np.testing.assert_allclose(result["direction_selectivity"][:3], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(result["orientation_selectivity"][:3], [0, 1, 1], atol=1e-15)
    assert np.isnan(result["direction_selectivity"][3]) and np.isnan(result["orientation_selectivity"][3])


def test_readout_handles_two_complete_blocks_and_matches_low_contrast_direction(tmp_path):
    values = conditions() + conditions("pfc_a_block_1")
    # Different high-direction counts make selecting the actual low direction observable.
    values[1] = replace(values[1], spike_ticks=np.array([2004]), spike_unit_indices=np.array([0]))
    source = archive(tmp_path / "source", values)
    report = readout([source], tmp_path / "readout", task_protocol("v1_visual"))
    assert report["complete_source_episodes_read"] == 20
    assert report["source_circuit_realizations"] == 1 and report["complete_ten_condition_blocks"] == 2
    assert report["production_episodes"] == 0 and report["inferential_statistics"] is False
    path = tmp_path / "readout" / report["blocks"][0]["artifacts"]
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["unit_ids"].tolist() == [19, 3, 72]
        np.testing.assert_allclose(arrays["low_minus_high_rate_hz"], [0, 1.6, 0])
        assert np.isnan(arrays["direction_selectivity"][2])


def test_incomplete_envelope_fails_and_intake_does_not_create_selectivity(tmp_path):
    source = archive(tmp_path / "source", [visual_episode()])
    with pytest.raises(ValueError, match="TEN_CONDITION_ENVELOPE_INCOMPLETE"):
        readout([source], tmp_path / "failure", task_protocol("v1_visual"))
    failed = json.loads((tmp_path / "failure/report.json").read_text())
    assert failed["status"] == "FAILED_V1_BENCHMARK_READOUT"
    report = readout([source], tmp_path / "intake", task_protocol("v1_visual"), intake_only=True)
    assert report["complete_ten_condition_blocks"] == 0 and report["status"] == "V1_BENCHMARK_INTAKE_ONLY"
    with np.load(tmp_path / "intake" / report["blocks"][0]["artifacts"]) as arrays:
        assert "direction_selectivity" not in arrays


def test_condition_control_and_noise_conflicts_are_rejected(tmp_path):
    value = visual_episode()
    commands = value.control_values.copy(); commands[501, 0] = 0
    with pytest.raises(ValueError, match="COMMAND_PROGRAM_DIFFERS"):
        describe_episode(replace(value, control_values=commands), units())
    with pytest.raises(ValueError, match="NOISE_STREAMS_DIFFER"):
        describe_episode(replace(value, seeds=dict(value.seeds, lgn_starting_noise_seed=99)), units())
    duplicate = archive(tmp_path / "duplicate", [value, replace(value, episode_id="same_condition")])
    with pytest.raises(ValueError, match="DUPLICATE_CONDITION_IN_BLOCK"):
        readout([duplicate], tmp_path / "rejected", task_protocol("v1_visual"))


def test_different_initial_state_and_changed_archive_are_rejected(tmp_path):
    values = conditions()
    values[1] = replace(values[1], source_record=dict(values[1].source_record, initial_vm_identity="other"))
    source = archive(tmp_path / "source", values)
    with pytest.raises(ValueError, match="SAME_INITIAL_STATE_AND_NOISE_BLOCK"):
        readout([source], tmp_path / "initial", task_protocol("v1_visual"))
    catalog = json.loads((source / "dataset.json").read_text())
    with (source / catalog["shards"][0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="ARCHIVE_CONTENT_CHANGED"):
        readout([source], tmp_path / "corrupt", task_protocol("v1_visual"))


def test_archive_source_orientation_is_reflected_and_silent_units_remain(tmp_path):
    static = replace(units(), source_tuning_angle_deg=np.array([45., 135., 0.]))
    values = conditions()
    for index in range(8):
        # Preferred source movie direction315 for source tuning45.
        values[index] = replace(values[index], spike_ticks=np.array([2004]) if index == 7 else np.array([], dtype=np.int64),
            spike_unit_indices=np.array([0]) if index == 7 else np.array([], dtype=np.int64))
    path = tmp_path / "source"
    with CircuitArchiveWriter(path, dict(binding(), scenario_id="v1_visual")) as writer:
        writer.register_circuit(static)
        for value in values: writer.add(value)
    result = readout([path], tmp_path / "readout", task_protocol("v1_visual"))
    with np.load(tmp_path / "readout" / result["blocks"][0]["artifacts"]) as arrays:
        np.testing.assert_array_equal(arrays["source_tuning_angle_deg"], [45., 135., 0.])
        np.testing.assert_array_equal(arrays["source_grating_orientation_deg"], [135., 45., 0.])
        assert arrays["source_orientation_alignment"][0] == pytest.approx(1.)
        assert np.isnan(arrays["source_orientation_alignment"][1:]).all()
    assert result["blocks"][0]["source_tuning_angle_available"]

    
def test_source_orientation_serialization_uses_one_direct_modulo(tmp_path):
    # Noninteger source degrees expose rounding from a nested 360/180 reduction.
    angles = np.array([45.123456789, 135.987654321, 0.000000123])
    static = replace(units(), source_tuning_angle_deg=angles)
    path = tmp_path / "source"
    with CircuitArchiveWriter(path, dict(binding(), scenario_id="v1_visual")) as writer:
        writer.register_circuit(static)
        for value in conditions(): writer.add(value)
    result = readout([path], tmp_path / "readout", task_protocol("v1_visual"))
    with np.load(tmp_path / "readout" / result["blocks"][0]["artifacts"]) as arrays:
        np.testing.assert_array_equal(arrays["source_grating_orientation_deg"], np.mod(-angles, 180.))
        np.testing.assert_array_equal(arrays["source_tuning_angle_deg"], angles)

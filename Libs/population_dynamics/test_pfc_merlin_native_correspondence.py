"""Native campaign comparison preserves exact identities and frozen tolerances."""
import copy

import numpy as np
import pytest

from .pfc_merlin_native_correspondence import array_result, validate_config


def config():
    identity = dict(split="train", episode_id="pfc_working_memory_circuit_023_block_006_condition_00",
                    initial_state_seed=210023006, background_seed=220023006)
    value = dict(scope="source_reproduction", excitatory_neurons=80000, inhibitory_neurons=20000,
        weight_scaling_factor=.0256, dt_ms=.0125, duration_ms=5000., readout_bin_ms=5.,
        synapse_execution="factorized_uniform_projections", random_stream_policy="independent_initial_external",
        expected_source_revision="d7c4f07fa83a052829e40790fd1594955fe1c067",
        benchmark_episode_candidate=dict(identity=identity, production_eligible=False),
        cases=[dict(case_id=identity["episode_id"])], initial_rng_seed=identity["initial_state_seed"],
        external_noise_seed=identity["background_seed"])
    return value, identity


def test_full_task_binding_and_protected_scope_reject():
    value, identity = config()
    validate_config(value, identity)
    for key, replacement in [("dt_ms", .05), ("duration_ms", 800.), ("excitatory_neurons", 2048),
                             ("weight_scaling_factor", 1.)]:
        changed = copy.deepcopy(value); changed[key] = replacement
        with pytest.raises(ValueError, match="COMPLETE_SOURCE_CONFIG_REQUIRED"):
            validate_config(changed, identity)
    changed = copy.deepcopy(value); changed["benchmark_episode_candidate"]["identity"]["split"] = "test"
    with pytest.raises(ValueError, match="TRAINING_CALIBRATION_IDENTITY_REQUIRED"):
        validate_config(changed, changed["benchmark_episode_candidate"]["identity"])


def test_candidate_stream_and_condition_cannot_change():
    value, identity = config(); value["external_noise_seed"] += 1
    with pytest.raises(ValueError, match="CASE_STREAM_IDENTITY_CHANGED"):
        validate_config(value, identity)


def test_exact_events_reject_a_single_tick_change():
    a = np.array([1, 5, 9], dtype=np.int64)
    b = np.array([1, 6, 9], dtype=np.int64)
    assert array_result("spike_ticks", a, a)["passed"]
    assert not array_result("spike_ticks", a, b)["passed"]
    assert not array_result("spike_ticks", a, a.astype(np.int32))["passed"]


def test_voltage_uses_absolute_existing_tolerance_and_zero_relative_tolerance():
    a = np.array([-60.], dtype=np.float64)
    assert array_result("terminal_voltage_mV", a, a + 5e-11, 1e-10)["passed"]
    assert not array_result("terminal_voltage_mV", a, a + 2e-10, 1e-10)["passed"]
    assert not array_result("terminal_voltage_mV", a, np.array([np.nan]), 1e-10)["passed"]

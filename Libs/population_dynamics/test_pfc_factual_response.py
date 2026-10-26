"""Factual source commands and causal preparation on authenticated fixtures."""
from dataclasses import replace
from copy import deepcopy
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from .pfc_population_evaluation import (source_controls, prepare_factual_response,
                                         prepare_registered_response, _registered_source_block)
from .population_state import ControlledPopulationState
from . import test_pfc_renewal_data as renewal_fixture


@pytest.fixture
def admitted():
    fixture = renewal_fixture.PfcRenewalDataTests()
    fixture.setUp()
    try:
        yield fixture.adapter()
    finally:
        fixture.doCleanups()


def fitted_view(data):
    # Source admission is real fixture validation; this model has no fitted
    # scientific evidence. The test establishes only interface compatibility.
    return ControlledPopulationState(unit_ids=data.unit_ids.tolist(),
        unit_field_index=data.unit_field_index, field_names=data.field_names,
        minimum_interval_ticks=data.minimum_interval_ticks, tick_seconds=data.dt_ms/1000,
        bin_seconds=data.bin_ticks*data.dt_ms/1000, input_size=4,
        latent_channels=2, encoder_size=5, initialization_seed=41,
        metadata=data.population_metadata()).eval()


def test_controls_preserve_native_segments_units_and_subbin_active_flags(admitted):
    expected = admitted.window(6000, 6200)[1]
    original = h5py.Group.__getitem__
    def only_commands(group, name):
        if str(name).startswith(("spike", "privileged")):
            raise AssertionError("CONTROL_READER_ACCESSED_OUTCOMES")
        return original(group, name)
    with patch.object(h5py.Group, "__getitem__", only_commands):
        actual = source_controls(admitted, 6000, 6200)
    np.testing.assert_array_equal(actual["known_controls"], expected)
    np.testing.assert_array_equal(actual["segment_start_ticks"], [2000, 6020])
    np.testing.assert_array_equal(actual["segment_stop_ticks"], [6020, 100000])
    np.testing.assert_array_equal(actual["known_controls"][0], [.1, 0., .2, 0.])
    assert actual["control_features"][0]["unit"] == "nA"
    assert actual["control_features"][2]["role"] == "derived_source_schedule_indicator"
    assert set(actual) == {"known_controls", "control_features", "segment_start_ticks",
                          "segment_stop_ticks", "segment_values", "start_tick",
                          "stop_tick_exclusive", "control_kind", "source_binding", "source_control_features"}


def test_factual_preparation_uses_only_complete_prefix_and_known_future_commands(admitted):
    model = fitted_view(admitted)
    original = admitted.window
    calls = []
    def prefix_only(start, stop):
        calls.append((start, stop))
        assert (start, stop) == (0, 6000)
        return original(start, stop)
    random_before = torch.get_rng_state().clone()
    with patch.object(admitted, "window", prefix_only):
        boundary, future = prepare_factual_response(model, admitted, 6000, 2)
    assert calls == [(0, 6000)]
    assert torch.equal(random_before, torch.get_rng_state())
    np.testing.assert_array_equal(future["known_controls"], original(6000, 6200)[1])
    np.testing.assert_array_equal(boundary.ready_in_ticks.numpy(), admitted.ready_at(6000)[None])
    assert boundary.source_metadata_identity == model.source_metadata_identity
    # A changed future spike realization cannot affect causal preparation.
    admitted.event_ticks[admitted.event_ticks >= 6000] += 1
    again, commands = prepare_factual_response(model, admitted, 6000, 2)
    assert torch.equal(boundary.mean, again.mean) and torch.equal(boundary.variance, again.variance)
    np.testing.assert_array_equal(future["known_controls"], commands["known_controls"])


@pytest.mark.parametrize("origin,horizon", [(0, 2), (6001, 2), (6000, 0), (99900, 2), (True, 2)])
def test_invalid_response_clock_rejects_before_observations(admitted, origin, horizon):
    model = fitted_view(admitted)
    with patch.object(admitted, "window", side_effect=AssertionError("INVALID_CLOCK_READ_OUTCOME")):
        with pytest.raises(ValueError, match="COMPLETE_ALIGNED_PREFIX"):
            prepare_factual_response(model, admitted, origin, horizon)


def test_factual_preparation_rejects_foreign_map_and_control_schema(admitted):
    model = fitted_view(admitted)
    model.unit_ids = model.unit_ids[::-1]
    with patch.object(admitted, "window", side_effect=AssertionError("FOREIGN_MAP_READ_OUTCOME")):
        with pytest.raises(ValueError, match="MODEL_SOURCE_MAP"):
            prepare_factual_response(model, admitted, 6000, 2)
    model = fitted_view(admitted)
    model.metadata["control_features"][0]["unit"] = "pA"
    with pytest.raises(ValueError, match="SOURCE_CONTROL_SEMANTICS"):
        prepare_factual_response(model, admitted, 6000, 2)


def test_active_zero_current_retains_source_reset_semantics(admitted):
    # The archive reader must retain the active flag even when the amplitude
    # is zero. This disposable episode changes no authoritative source archive.
    from .circuit_data import CircuitArchiveWriter, CircuitDataset
    from .test_circuit_data import binding, units, episode
    from tempfile import TemporaryDirectory
    from pathlib import Path
    from types import SimpleNamespace
    with TemporaryDirectory() as directory:
        root = Path(directory)
        raw = replace(episode(), control_ticks=np.array([0, 2, 4], dtype=np.int64),
            control_values=np.array([[0., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 0.]]),
            control_features=tuple(admitted.population_metadata()["control_features"]))
        with CircuitArchiveWriter(root, binding()) as writer:
            writer.register_circuit(units())
            writer.add(raw)
        data = CircuitDataset(root, splits=("train",))
        adapter = SimpleNamespace(data=data, episode_id=raw.episode_id, duration_ticks=10,
                                 bin_ticks=2, binding={"fixture": True}, source_config=admitted.source_config,
                                 model_control_features=list(raw.control_features))
        commands = source_controls(adapter, 0, 6)
        np.testing.assert_array_equal(commands["known_controls"][:, 0], [0., 0., 0.])
        np.testing.assert_array_equal(commands["known_controls"][:, 2], [0., 1., 0.])



def reduced_case(data):
    factual = next(row for row in data.source_config["cases"] if row["case_id"] == data.episode_id)
    return next(row["case_id"] for row in data.source_config["cases"]
                if row["source_rng_seed"] == factual["source_rng_seed"] and row["stimulus_strength_nA"] == 0.)


def assert_same_boundary(left, right):
    assert torch.equal(left.mean, right.mean)
    assert torch.equal(left.variance, right.variance)
    assert torch.equal(left.ready_in_ticks, right.ready_in_ticks)
    assert left.source_metadata_identity == right.source_metadata_identity


def test_registered_same_case_matches_factual_and_keeps_global_rng(admitted):
    model = fitted_view(admitted)
    factual, program = prepare_factual_response(model, admitted, 2000, 2)
    rng = torch.get_rng_state().clone()
    boundary, candidate = prepare_registered_response(model, admitted, 2000, 2, admitted.episode_id)
    assert_same_boundary(factual, boundary)
    assert torch.equal(rng, torch.get_rng_state())
    for name, expected in program.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(candidate[name], expected)
        else:
            assert candidate[name] == expected
    binding = candidate["registered_response_binding"]
    assert binding["candidate_case_id"] == binding["factual_case_id"] == admitted.episode_id
    assert binding["source_config_sha256"] == admitted.data.catalog["binding"]["config_sha256"]
    assert binding["command_units"] == "nA" and not binding["scientific_intervention_qualified"]


def test_registered_reduction_preserves_native_flags_segments_and_zero_assignment(admitted):
    model = fitted_view(admitted)
    factual, program = prepare_factual_response(model, admitted, 2000, 2)
    boundary, candidate = prepare_registered_response(model, admitted, 2000, 2, reduced_case(admitted))
    assert_same_boundary(factual, boundary)
    assert np.any(program["known_controls"][:, 0] > 0)
    np.testing.assert_array_equal(candidate["known_controls"][:, 0], 0.)
    np.testing.assert_array_equal(candidate["known_controls"][:, 2:], program["known_controls"][:, 2:])
    np.testing.assert_array_equal(candidate["segment_start_ticks"], program["segment_start_ticks"])
    np.testing.assert_array_equal(candidate["segment_stop_ticks"], program["segment_stop_ticks"])
    np.testing.assert_array_equal(candidate["segment_values"][:, 2:], program["segment_values"][:, 2:])
    assert np.all(candidate["segment_values"][:, 2] == 1.)
    assert "inactive: clear all E" in candidate["registered_response_binding"]["cue_assignment"]


@pytest.mark.parametrize("candidate", ["unknown_case", "pfc_94002_strong_cue"])
def test_unregistered_or_foreign_block_rejects_before_prefix_outcomes(admitted, candidate):
    model = fitted_view(admitted)
    with patch.object(admitted, "window", side_effect=AssertionError("INVALID_CANDIDATE_READ_OUTCOME")):
        with pytest.raises(ValueError, match="UNIQUE_REGISTERED_CASE|SAME_INITIAL_EXTERNAL_BLOCK"):
            prepare_registered_response(model, admitted, 2000, 2, candidate)


def test_registered_reduction_rejects_changed_whole_case_prefix_before_outcomes(admitted):
    model = fitted_view(admitted)
    with patch.object(admitted, "window", side_effect=AssertionError("CHANGED_PREFIX_READ_OUTCOME")):
        with pytest.raises(ValueError, match="CANDIDATE_PREFIX_CHANGED"):
            prepare_registered_response(model, admitted, 6000, 2, reduced_case(admitted))


def test_registered_program_and_boundary_do_not_read_future_spikes(admitted):
    model = fitted_view(admitted)
    original = admitted.window
    calls = []
    def prefix_only(start, stop):
        calls.append((start, stop))
        assert (start, stop) == (0, 2000)
        return original(start, stop)
    with patch.object(admitted, "window", prefix_only):
        first, program = prepare_registered_response(model, admitted, 2000, 2, reduced_case(admitted))
        admitted.event_ticks[admitted.event_ticks >= 2000] += 1
        second, again = prepare_registered_response(model, admitted, 2000, 2, reduced_case(admitted))
    assert calls == [(0, 2000), (0, 2000)]
    assert_same_boundary(first, second)
    for name in ("known_controls", "segment_start_ticks", "segment_stop_ticks", "segment_values"):
        np.testing.assert_array_equal(program[name], again[name])
    assert program["registered_response_binding"] == again["registered_response_binding"]


@pytest.mark.parametrize("value", [-.1, float("nan"), float("inf"), True, [.1] * 20])
def test_registered_response_rejects_unsafe_amplitude_before_outcomes(admitted, value):
    config = deepcopy(admitted.source_config)
    candidate = reduced_case(admitted)
    next(row for row in config["cases"] if row["case_id"] == candidate)["stimulus_strength_nA"] = value
    model = fitted_view(admitted)
    # Only a disposable in-memory registry changes. No source archive is rewritten.
    with patch.object(admitted, "source_config", config), \
         patch.dict(admitted.data.catalog["binding"], resolved_source_config=config), \
         patch.object(admitted, "window", side_effect=AssertionError("UNSAFE_CURRENT_READ_OUTCOME")):
        with pytest.raises(ValueError, match="FINITE_NONNEGATIVE_SCALAR_NA"):
            prepare_registered_response(model, admitted, 2000, 2, candidate)


@pytest.mark.parametrize("change,reason", [
    ("duplicate", "UNIQUE_REGISTERED_CASE"),
    ("per_unit", "PROGRAM_OVERRIDE_UNSUPPORTED"),
    ("time", "TIME_OR_FEATURE_SCHEMA"),
    ("feature", "TIME_OR_FEATURE_SCHEMA"),
])
def test_registered_response_rejects_ambiguous_or_changed_program_before_outcomes(admitted, change, reason):
    config = deepcopy(admitted.source_config)
    candidate = reduced_case(admitted)
    selected = next(row for row in config["cases"] if row["case_id"] == candidate)
    if change == "duplicate":
        config["cases"].append(dict(selected))
    elif change == "per_unit":
        selected["current_A"] = [0.] * len(admitted.unit_ids)
    elif change == "time":
        config["stimulus_start_ms"] = 101.
    else:
        config["stimulus_center_deg"] = 90.
    model = fitted_view(admitted)
    with patch.object(admitted, "source_config", config), \
         patch.dict(admitted.data.catalog["binding"], resolved_source_config=config), \
         patch.object(admitted, "window", side_effect=AssertionError("INVALID_PROGRAM_READ_OUTCOME")):
        with pytest.raises(ValueError, match=reason):
            prepare_registered_response(model, admitted, 2000, 2, candidate)


def test_source_block_resolver_keeps_independent_initial_and_external_seeds_separate():
    config = dict(random_stream_policy="independent_initial_external", initial_rng_seed=94001, external_noise_seed=95001)
    block = _registered_source_block(config, {"case_id": "named_sources"})
    assert block == dict(random_stream_policy="independent_initial_external",
                         seeds=dict(initial_rng_seed=94001, external_noise_seed=95001),
                         block_id="pfc_initial_94001_external_95001")
    changed_initial = _registered_source_block(config, dict(initial_rng_seed=94002))
    changed_external = _registered_source_block(config, dict(external_noise_seed=95002))
    assert changed_initial != block and changed_external != block
    resolved = dict(source_rng_seed=95001, random_stream_policy="independent_initial_external")
    assert _registered_source_block(config, resolved) == block
    with pytest.raises(ValueError, match="SEED_ALIAS_CHANGED"):
        _registered_source_block(config, dict(resolved, source_rng_seed=94001))

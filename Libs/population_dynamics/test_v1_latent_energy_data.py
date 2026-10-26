"""V1 source-order adaptation without EAG training or a neuron-identity match."""
from __future__ import annotations

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from . import v1_population_training_data as source
from .test_v1_population_training_data import configuration as population_configuration
from .test_v1_population_training_data import inputs
from .v1_latent_energy_data import V1LatentEnergyCohort
from .v1_visual_controls import CONTROL_FEATURES


def configuration(root):
    config = population_configuration()
    catalog = json.loads((root / "dataset.json").read_text())
    episode = catalog["shards"][0]["episodes"][0]
    bins = episode["duration_ticks"] // config["bin_ticks"]
    bin_ms = episode["dt_ms"] * config["bin_ticks"]
    config.update(schema="v1_latent_energy_count_adapter_test", model_family="eag",
                  complete_episode_bins=bins, bin_ms=bin_ms,
                  model=dict(complete_episode_bins=bins, bin_seconds=bin_ms / 1000,
                             input_size=len(CONTROL_FEATURES)))
    return config


def count_config_validator(config, seed):
    # Architecture and budgets belong to the production EAG validator. This
    # adapter fixture verifies explicit validator routing and source admission.
    if config["schema"] != "v1_latent_energy_count_adapter_test" or seed not in (41, 42, 43):
        raise ValueError("count adapter test protocol required")
    if set(config["fit_circuit_ids"]) & set(config["validation_circuit_ids"]):
        raise ValueError("whole-circuit split leakage")


def test_explicit_validator_routes_through_inspection_and_cohort(tmp_path):
    path = inputs(tmp_path)
    config = configuration(tmp_path)
    validator = Mock(side_effect=count_config_validator)
    with patch.object(source, "validate_config", side_effect=AssertionError("population validator used")):
        inspected = source.inspect_inputs(path, config, config_validator=validator)
        cohort = V1LatentEnergyCohort(path, config, config_validator=validator)
    assert inspected.raw == path.read_bytes()
    assert validator.call_count == 2
    assert all(call.args == (config, 41) for call in validator.call_args_list)
    assert len(cohort.fit) == 2 and len(cohort.validation) == 1
    # Omitting the callback keeps the original population admission contract.
    with pytest.raises(ValueError, match="REGISTERED_DEVELOPMENT_PROTOCOL"):
        source.inspect_inputs(path, config)
    assert source.inspect_inputs(path, population_configuration()).raw == inspected.raw


def test_same_width_distinct_source_ids_preserve_all_channels_and_controls(tmp_path, monkeypatch):
    path = inputs(tmp_path)
    config = configuration(tmp_path)
    real_episode = source.V1RenewalEpisode

    def admitted_episode(data, episode_id, **kwargs):
        episode = real_episode(data, episode_id, **kwargs)
        # Isolate the adapter's identity contract with distinct case labels.
        # Raw source admission is exercised independently below and upstream.
        offset = 100000 if episode.binding["circuit_id"] == "v1_validation" else 200000
        episode.unit_ids = episode.unit_ids + offset
        return episode

    monkeypatch.setattr(source, "V1RenewalEpisode", admitted_episode)
    with patch.object(real_episode, "renewal_window", side_effect=AssertionError("unused eligibility computed")):
        cohort = V1LatentEnergyCohort(path, config, config_validator=count_config_validator)
        seen = {}
        for case in (*cohort.fit, *cohort.validation):
            counts, controls = cohort.window(case)
            native = cohort.native_episode(case)
            expected_counts, expected_controls = native.window(0, native.duration_ticks)
            np.testing.assert_array_equal(counts, expected_counts)
            np.testing.assert_array_equal(controls, expected_controls)
            np.testing.assert_array_equal(cohort.unit_ids, np.arange(len(native.unit_ids)))
            assert counts.shape == (config["complete_episode_bins"], len(native.unit_ids))
            assert controls.shape == (config["complete_episode_bins"], len(CONTROL_FEATURES))
            assert native.binding["episode_id"] == case["episode_id"]
            assert native.binding["circuit_id"] == case["circuit_id"]
            assert cohort.native_episode(case) is native
            seen[case["circuit_id"]] = native.unit_ids.copy()
    assert not np.array_equal(seen["v1_fit"], seen["v1_validation"])
    policy = cohort.binding["count_adapter"]
    assert policy["source_unit_order"] == "unchanged_per_case_circuit_archive_order"
    assert policy["cross_circuit_unit_correspondence"] == "not_assumed"
    assert policy["control_features"] == list(CONTROL_FEATURES)
    assert cohort.binding["partition_unit"] == "whole_source_circuit"


def test_variable_source_width_rejects_during_cohort_construction(tmp_path, monkeypatch):
    path = inputs(tmp_path)
    real_episode = source.V1RenewalEpisode

    def different_width(data, episode_id, **kwargs):
        episode = real_episode(data, episode_id, **kwargs)
        if episode.binding["circuit_id"] == "v1_validation":
            episode.unit_ids = episode.unit_ids[:-1]
        return episode

    monkeypatch.setattr(source, "V1RenewalEpisode", different_width)
    with pytest.raises(ValueError, match="EQUAL_SOURCE_CHANNEL_COUNTS"):
        V1LatentEnergyCohort(path, configuration(tmp_path), config_validator=count_config_validator)


@pytest.mark.parametrize("mismatch", ["episode_bins", "model_bins", "bin_ms", "model_seconds", "controls"])
def test_declared_clock_and_control_mismatches_reject(tmp_path, mismatch):
    path = inputs(tmp_path)
    config = configuration(tmp_path)
    if mismatch == "episode_bins": config["complete_episode_bins"] += 1
    elif mismatch == "model_bins": config["model"]["complete_episode_bins"] += 1
    elif mismatch == "bin_ms": config["bin_ms"] *= 2
    elif mismatch == "model_seconds": config["model"]["bin_seconds"] *= 2
    else: config["model"]["input_size"] += 1
    message = "SOURCE_CONTROL_CHANNELS" if mismatch == "controls" else "COMPLETE_SOURCE_CLOCK"
    with pytest.raises(ValueError, match=message):
        V1LatentEnergyCohort(path, config, config_validator=count_config_validator)


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_missing_original_validation_rejects_before_hashes_and_arrays(tmp_path, split):
    path = inputs(tmp_path, validation_split=split)
    config = configuration(tmp_path)
    with patch.object(source, "V1RenewalEpisode", side_effect=AssertionError("source arrays read")), \
         patch.object(source, "sha256_file", side_effect=AssertionError("parameter bytes read")), \
         patch("h5py.File", side_effect=AssertionError("HDF5 arrays read")):
        with pytest.raises(ValueError, match="ORIGINAL_VALIDATION_CIRCUITS_MISSING_OR_MISASSIGNED"):
            V1LatentEnergyCohort(path, config, config_validator=count_config_validator)

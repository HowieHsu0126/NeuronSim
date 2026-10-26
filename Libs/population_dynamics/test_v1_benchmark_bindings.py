"""Candidate exports retain the actual construction and nested noise identity."""
from copy import deepcopy
from pathlib import Path

import pytest

from .benchmark_tasks import condition_for_episode
from .circuit_campaign import campaign_config, episode_identities
from .v1_benchmark import bind_candidate


def fixture(circuit=70, block=3, condition=9):
    campaign = campaign_config(Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml")
    identity = next(row for row in episode_identities(campaign, "v1_visual")
                    if row["circuit_index"] == circuit and row["block_index"] == block
                    and row["condition_index"] == condition)
    candidate = dict(status="PREPARED_BENCHMARK_CIRCUIT_CONSTRUCTION_INPUT",
                     scenario_id="v1_visual", circuit_id=identity["circuit_id"], circuit_index=circuit,
                     intended_split=identity["split"], construction_scope="development",
                     connectivity_seed=identity["connectivity_seed"], complete_episodes=0,
                     production_eligible=False)
    config = dict(benchmark_circuit_candidate=candidate, builder_seed=identity["connectivity_seed"],
                  glif_assignment_seed=identity["connectivity_seed"] + 1000000)
    recurrent = dict(original_config=deepcopy(config), original_config_sha256="bound",
                     rng={key:config[key] for key in ("builder_seed", "glif_assignment_seed")})
    inputs = dict(original_config=deepcopy(config), original_config_sha256="bound",
                  rng=dict(lgn_selection_python_seed=identity["connectivity_seed"]))
    rates = dict(spike_generation=dict(starting_noise_seed=identity["stimulus_seed"]),
                 background=dict(starting_noise_seed=identity["background_seed"]))
    native = dict(original_config_sha256="bound", bindings=dict(spike_inputs=dict(config=deepcopy(rates))))
    return campaign, recurrent, inputs, native, rates, identity


def test_candidate_export_binds_whole_circuit_split_and_nested_noise_block():
    campaign, recurrent, inputs, native, rates, identity = fixture()
    actual = bind_candidate(campaign, recurrent, inputs, native, rates,
                            circuit_index=70, block_index=3, condition_index=9)
    assert actual == identity and actual["split"] == "validation"
    protocol = campaign["scenarios"]["v1_visual"]["task_protocol"]
    assert condition_for_episode(protocol, "v1_visual", actual)["direction_deg"] == 135
    assert recurrent["original_config"]["benchmark_circuit_candidate"]["construction_scope"] == "development"


@pytest.mark.parametrize("change", ["circuit_alias", "missing_candidate", "builder_seed", "actual_rng", "config_binding"])
def test_candidate_export_rejects_rebound_construction(change):
    campaign, recurrent, inputs, native, rates, _ = fixture()
    if change == "circuit_alias":
        inputs["original_config"]["benchmark_circuit_candidate"]["circuit_id"] = "another"
    elif change == "missing_candidate":
        del inputs["original_config"]["benchmark_circuit_candidate"]
    elif change == "builder_seed":
        recurrent["original_config"]["builder_seed"] += 1
    elif change == "actual_rng":
        recurrent["rng"]["builder_seed"] += 1
    else:
        inputs["original_config_sha256"] = "another"
    with pytest.raises(ValueError, match="CIRCUIT|SEEDS_DIFFER"):
        bind_candidate(campaign, recurrent, inputs, native, rates,
                       circuit_index=70, block_index=3, condition_index=9)


@pytest.mark.parametrize("change", ["lgn", "background", "native"])
def test_candidate_export_rejects_wrong_realized_noise_block(change):
    campaign, recurrent, inputs, native, rates, _ = fixture()
    if change == "native":
        native["bindings"]["spike_inputs"]["config"]["background"]["starting_noise_seed"] += 1
    else:
        rates["spike_generation" if change == "lgn" else "background"]["starting_noise_seed"] += 1
    with pytest.raises(ValueError, match="NOISE_BLOCK_DIFFERS"):
        bind_candidate(campaign, recurrent, inputs, native, rates,
                       circuit_index=70, block_index=3, condition_index=9)


def test_legacy_archive_cannot_become_an_independent_circuit_or_noise_block():
    campaign, *_ = fixture()
    assert bind_candidate(campaign, {}, {}, {}, {}, circuit_index=0, block_index=0, condition_index=0) is None
    for circuit, block in ((1, 0), (0, 1)):
        with pytest.raises(ValueError, match="REQUIRES_CANDIDATE_CIRCUIT"):
            bind_candidate(campaign, {}, {}, {}, {}, circuit_index=circuit, block_index=block, condition_index=0)


def test_candidate_export_reuses_rates_and_binds_actual_spike_noise():
    campaign, recurrent, inputs, native, rates, identity = fixture()
    reused_rates = deepcopy(rates)
    reused_rates["spike_generation"]["starting_noise_seed"] -= 1
    reused_rates["background"]["starting_noise_seed"] -= 1
    actual = bind_candidate(campaign, recurrent, inputs, native, reused_rates,
                           circuit_index=70, block_index=3, condition_index=9,
                           spike_config=rates)
    assert actual == identity
    assert native["bindings"]["spike_inputs"]["config"] == rates
    assert reused_rates != rates


@pytest.mark.parametrize("change", ["condition", "trials", "coverage", "lgn", "background", "native"])
def test_reused_rates_reject_changed_recipe_or_actual_noise(change):
    campaign, recurrent, inputs, native, rates, _ = fixture()
    spikes = deepcopy(rates)
    if change == "condition":
        rates["stimulus"] = dict(direction_deg=0)
        spikes["stimulus"] = dict(direction_deg=45)
    elif change == "trials":
        rates["spike_generation"]["trials"] = 1
        spikes["spike_generation"]["trials"] = 2
    elif change == "coverage":
        rates["unit_coverage"] = "all_source_units"
        spikes["unit_coverage"] = "model_representatives"
    elif change == "native":
        native["bindings"]["spike_inputs"]["config"]["background"]["starting_noise_seed"] += 1
    else:
        spikes["spike_generation" if change == "lgn" else "background"]["starting_noise_seed"] += 1
    with pytest.raises(ValueError, match="RATE_RECIPE_DIFFERS|NOISE_BLOCK_DIFFERS"):
        bind_candidate(campaign, recurrent, inputs, native, rates,
                       circuit_index=70, block_index=3, condition_index=9,
                       spike_config=spikes)

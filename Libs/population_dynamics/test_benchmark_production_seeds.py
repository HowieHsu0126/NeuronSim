"""Keep exposed development random inputs out of fresh V1/CA3 production."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from . import ca3_benchmark
from .ca3_realization import MAX_SEED as CA3_MAX_SEED
from .circuit_campaign import (MAX_CAMPAIGN_SEED, SEED_NAMESPACE_SPAN, campaign_config,
                               episode_identities, production_prerequisites, seed_namespace)
from .v1_execution_recipe import UINT32_MAX


CONFIG = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"
SEEDS = ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed")


@pytest.mark.parametrize("scenario,development_base,production_base", (
    ("v1_visual", 100000000, 400000000),
    ("ca3_pattern_completion", 300000000, 500000000),
))
def test_complete_envelopes_have_disjoint_source_seeds_and_identical_nonseed_identities(
        scenario, development_base, production_base):
    campaign = campaign_config(CONFIG)
    development_seeds, production_seeds = set(), set()
    splits = Counter()
    count = 0
    for old, fresh in zip(episode_identities(campaign, scenario),
                          episode_identities(campaign, scenario, production=True), strict=True):
        assert {k: v for k, v in old.items() if k not in SEEDS} == {k: v for k, v in fresh.items() if k not in SEEDS}
        assert old["connectivity_seed"] == development_base + old["circuit_index"]
        assert fresh["connectivity_seed"] == production_base + fresh["circuit_index"]
        development_seeds.update(old[key] for key in SEEDS)
        production_seeds.update(fresh[key] for key in SEEDS)
        if scenario == "v1_visual":
            development_seeds.add(old["connectivity_seed"] + 1000000)
            production_seeds.add(fresh["connectivity_seed"] + 1000000)
        splits[fresh["split"]] += 1
        count += 1
    assert count == 100000
    assert splits == dict(train=70000, validation=10000, calibration=10000, test=10000)
    assert not development_seeds & production_seeds
    assert max(production_seeds) <= CA3_MAX_SEED <= UINT32_MAX


def test_pfc_seed_allocation_is_unchanged():
    campaign = campaign_config(CONFIG)
    assert list(episode_identities(campaign, "pfc_working_memory")) == list(
        episode_identities(campaign, "pfc_working_memory", production=True))


@pytest.mark.parametrize("base", (100000000, 200000000, 300000000, 500000000,
                                  130999999, MAX_CAMPAIGN_SEED, False, 400000000.0))
def test_overlapping_or_invalid_production_namespace_rejects(base):
    campaign = campaign_config(CONFIG)
    campaign["scenarios"]["v1_visual"]["production_seed_namespace"] = base
    with pytest.raises(ValueError, match="PRODUCTION_SEED_NAMESPACE"):
        next(episode_identities(campaign, "v1_visual", production=True))


def test_source_seed_upper_bound_and_legacy_development_config():
    campaign = campaign_config(CONFIG)
    assert MAX_CAMPAIGN_SEED == CA3_MAX_SEED
    campaign["scenarios"]["v1_visual"]["production_seed_namespace"] = MAX_CAMPAIGN_SEED - SEED_NAMESPACE_SPAN + 1
    assert seed_namespace(campaign, "v1_visual", production=True) == MAX_CAMPAIGN_SEED - SEED_NAMESPACE_SPAN + 1
    legacy = deepcopy(campaign)
    for scenario in ("v1_visual", "ca3_pattern_completion"):
        legacy["scenarios"][scenario].pop("production_seed_namespace")
    assert next(episode_identities(legacy, "v1_visual"))["connectivity_seed"] == 100000000
    assert next(episode_identities(legacy, "ca3_pattern_completion"))["connectivity_seed"] == 300000000
    with pytest.raises(ValueError, match="FRESH_PRODUCTION_SEED_NAMESPACE_REQUIRED"):
        next(episode_identities(legacy, "v1_visual", production=True))
    assert "FRESH_PRODUCTION_SEED_NAMESPACE_REQUIRED:v1_visual" in production_prerequisites(legacy, "v1_visual", None)


@pytest.mark.parametrize("production", (False, True))
def test_ca3_source_entry_selects_the_requested_namespace_before_execution(tmp_path, monkeypatch, production):
    campaign = campaign_config(CONFIG)
    expected = next(episode_identities(campaign, "ca3_pattern_completion", production=production))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-preflight-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-no-native-execution")

    def condition(protocol, scenario, identity):
        assert identity == expected
        raise RuntimeError("stop before native inputs or execution")

    monkeypatch.setattr(ca3_benchmark, "condition_for_episode", condition)
    with pytest.raises(RuntimeError, match="stop before native"):
        ca3_benchmark.run(tmp_path / "absent", tmp_path / "absent", tmp_path / "absent", CONFIG,
                          tmp_path / "output", block_index=0, condition_index=0,
                          production_identity=expected if production else None)
    assert not (tmp_path / "output").exists()


def test_ca3_cannot_pass_a_development_identity_as_production(tmp_path, monkeypatch):
    campaign = campaign_config(CONFIG)
    development = next(episode_identities(campaign, "ca3_pattern_completion"))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-preflight-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-no-native-execution")
    with pytest.raises(ValueError, match="CA3_BENCHMARK_PRODUCTION_IDENTITY_DIFFERS"):
        ca3_benchmark.run(tmp_path / "absent", tmp_path / "absent", tmp_path / "absent", CONFIG,
                          tmp_path / "output", block_index=0, condition_index=0,
                          production_identity=development)
    assert not (tmp_path / "output").exists()

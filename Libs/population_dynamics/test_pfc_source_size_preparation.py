"""Original-size PFC comparison preparation; no simulator or science claims."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from . import benchmark_sampling as sampling
from .benchmark_tasks import task_protocol
from .circuit_campaign import campaign_config, episode_identities
from .pfc_kernel_parameters import circuit_provenance
from .pfc_native_drive import SOURCE_ENTRY_SHA256
from .pfc_source_dataset import _benchmark_registration
from .pfc_source_execution import _case_config
from .provenance import sha256_file


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROJECT = Path(__file__).resolve().parents[2]


def source_circuit(root: Path, circuit_index: int = 0) -> tuple[Path, Path]:
    """Build one candidate from existing source definitions, without100k kernels."""
    root.mkdir(parents=True)
    campaign_path = CONFIGS / "neural_population_dynamics.yaml"
    campaign = campaign_config(campaign_path)
    identity = next(item for item in episode_identities(campaign, "pfc_working_memory")
                    if item["circuit_index"] == circuit_index)
    base = yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source_dt_0p0125ms.yaml").read_text())
    base.update(sigma_weight_profile=22.22222222222222, Jpos_excit2excit=1.7555555555555555)
    base["benchmark_circuit_candidate"] = dict(
        status="PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE", scenario_id="pfc_working_memory",
        circuit_id=identity["circuit_id"], circuit_index=circuit_index,
        intended_split=identity["split"], construction_scope="development",
        parameters=dict(sigma_weight_profile=base["sigma_weight_profile"], Jpos_excit2excit=base["Jpos_excit2excit"]),
        effective_kernel_sha256="fixture100khashalreadyprepared",
        production_eligible=False, scientific_parameter_domain_qualified=False,
        **circuit_provenance(base, SOURCE_ENTRY_SHA256))
    path = root / "circuit.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return campaign_path, path


def test_reference_size_block_preserves_complete_task_namespace_and_source_binding(tmp_path):
    campaign_path, source_path = source_circuit(tmp_path / "source")
    source_before = source_path.read_bytes()
    source_config = yaml.safe_load(source_before)
    output = tmp_path / "comparison"

    report = sampling.prepare_pfc_reference_size_block(campaign_path, source_path, 0, 6, output)

    assert report["status"] == "PREPARED_PFC_REFERENCE_SIZE_DEVELOPMENT_BLOCK_RECIPES"
    assert report["coupled_core_units"] == 2560 and report["target_coupled_core_units"] == 100000
    assert report["complete_episodes"] == report["campaign_episodes"] == report["production_episodes"] == 0
    assert report["new_source_simulations"] == 0
    assert report["production_eligible"] is report["source_quality_qualified"] is False
    assert report["cross_size_noise_pairing"] is False
    assert report["original_circuit_config_sha256"] == sha256_file(source_path)
    assert report["original_circuit_candidate"] == source_config["benchmark_circuit_candidate"]
    assert source_path.read_bytes() == source_before
    assert len(report["conditions"]) == 10
    assert [item["identity"]["condition_index"] for item in report["conditions"]] == list(range(10))
    assert len({item["condition"]["condition_id"] for item in report["conditions"]}) == 10
    seeds = set()
    for row in report["conditions"]:
        config = yaml.safe_load((output / row["config_path"]).read_text())
        identity, canonical = row["identity"], row["canonical_campaign_identity"]
        assert config["excitatory_neurons"] == 2048 and config["inhibitory_neurons"] == 512
        assert config["weight_scaling_factor"] == 1.0
        assert config["dt_ms"] == .0125 and config["duration_ms"] == 5000.
        assert config["random_stream_policy"] == "independent_initial_external"
        for name in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"):
            assert identity[name] == canonical[name] + sampling.PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET
        assert identity["initial_state_seed"] == 310000006
        assert identity["background_seed"] == 320000006
        assert identity["episode_id"] == canonical["episode_id"]
        assert config["initial_rng_seed"] == config["cases"][0]["initial_rng_seed"] == identity["initial_state_seed"]
        assert config["external_noise_seed"] == config["cases"][0]["external_noise_seed"] == identity["background_seed"]
        seeds.add((config["initial_rng_seed"], config["external_noise_seed"]))
        assert config["benchmark_episode_candidate"]["execution_scope"] == "development"
        assert config["benchmark_episode_candidate"]["production_eligible"] is False
        assert config["pfc_source_size_comparison"]["comparator_role"] == "original_size2560"
        assert config["pfc_source_size_comparison"]["canonical_campaign_identity"] == canonical
        provenance = circuit_provenance(config, SOURCE_ENTRY_SHA256)
        assert provenance == report["circuit_provenance"]
        assert provenance["parameter_identity"] != source_config["benchmark_circuit_candidate"]["parameter_identity"]
        assert provenance["connectivity_identity"] != source_config["benchmark_circuit_candidate"]["connectivity_identity"]
        assert _benchmark_registration(config)[0] == config
        _, _, dimensions = _case_config(config, identity["episode_id"])
        assert dimensions["coupled_core_units"] == 2560 and dimensions["duration_ticks"] == 400000
    assert len(seeds) == 1
    assert report["conditions"][0]["condition"]["stimulus_strength_nA"] == .5
    assert report["conditions"][5]["condition"]["stimulus_strength_nA"] == 0.
    assert json.loads((output / "report.json").read_text()) == report


def test_reference_size_block_does_not_reduce_existing_fullscale_gate(tmp_path):
    campaign_path, source_path = source_circuit(tmp_path / "source")
    output = tmp_path / "comparison"
    report = sampling.prepare_pfc_reference_size_block(campaign_path, source_path, 0, 0, output)
    comparator = yaml.safe_load((output / report["conditions"][0]["config_path"]).read_text())
    comparator.pop("benchmark_episode_candidate")
    path = tmp_path / "comparator.yaml"
    path.write_text(yaml.safe_dump(comparator))

    with pytest.raises(ValueError, match="FULL_COUPLED_SOURCE_REQUIRED"):
        sampling.prepare_pfc_block(campaign_path, path, 0, 1, tmp_path / "wrong100k")

    assert not (tmp_path / "wrong100k").exists()


@pytest.mark.parametrize("damage", ["production", "source_size", "normalization", "clock", "tuple", "alias"])
def test_reference_size_block_rejects_changed_source_before_output(tmp_path, damage):
    campaign_path, source_path = source_circuit(tmp_path / "source")
    config = yaml.safe_load(source_path.read_text())
    if damage == "production":
        config["benchmark_circuit_candidate"].update(status="PREPARED_QUALIFIED_PRODUCTION_CIRCUIT", production_eligible=True)
    elif damage == "source_size":
        config["excitatory_neurons"] = 2048
    elif damage == "normalization":
        config["weight_scaling_factor"] = 1.
    elif damage == "clock":
        config["dt_ms"] = .05
    elif damage == "tuple":
        config["Jpos_excit2excit"] += .1
    else:
        config["benchmark_circuit_candidate"]["circuit_index"] = 1
    source_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "comparison"

    with pytest.raises(ValueError):
        sampling.prepare_pfc_reference_size_block(campaign_path, source_path, 0, 0, output)

    assert not output.exists()


@pytest.mark.parametrize("circuit,block", [(-1,0),(100,0),(0,-1),(0,100),(False,0),(0,True)])
def test_reference_size_block_rejects_invalid_indices(tmp_path,circuit,block):
    campaign_path, source_path = source_circuit(tmp_path / "source")
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="CIRCUIT_OR_BLOCK_INDEX_INVALID"):
        sampling.prepare_pfc_reference_size_block(campaign_path,source_path,circuit,block,output)
    assert not output.exists()


def test_reference_size_block_cli_and_missing_arguments_exit(tmp_path):
    campaign_path, source_path = source_circuit(tmp_path / "source")
    output = tmp_path / "comparison"
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "Libs"), PYTHONDONTWRITEBYTECODE="1")
    common = [sys.executable,"-B","-m","population_dynamics.benchmark_sampling","--config",str(campaign_path),
              "--reference-size-block","--base-config",str(source_path),"--circuit-index","0"]

    child = subprocess.run([*common,"--block-index","6","--output",str(output)],env=env,
                           capture_output=True,text=True,cwd=tmp_path)
    assert child.returncode == 0, child.stderr
    assert len(json.loads((output / "report.json").read_text())["conditions"]) == 10
    missing = tmp_path / "missing"
    rejected = subprocess.run([*common,"--output",str(missing)],env=env,capture_output=True,text=True,cwd=tmp_path)
    assert rejected.returncode == 2
    assert not missing.exists()
    conflict = tmp_path / "conflict"
    rejected = subprocess.run([*common,"--block-index","6","--scenario","pfc_working_memory",
                               "--output",str(conflict)],env=env,capture_output=True,text=True,cwd=tmp_path)
    assert rejected.returncode == 2
    assert not conflict.exists()

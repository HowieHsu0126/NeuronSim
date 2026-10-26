"""BENCH source identities do not depend on model-training seed metadata."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from .ca3_realization import SourceRealization, load_realization
from .ca3_reference import execute
from .circuit_campaign import campaign_config, episode_identities
from .config import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


def write_config(tmp_path, training_seeds):
    config = yaml.safe_load(CONFIG.read_text())
    if training_seeds is None:
        config.pop("training_seeds")
    else:
        config["training_seeds"] = training_seeds
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.parametrize("training_seeds", (None, [], [17, 29], [41, 42, 43, 44], ["41", "42", "43"]))
def test_benchmark_parser_ignores_training_metadata_and_default_loader_rejects_it(tmp_path, training_seeds):
    path = write_config(tmp_path, training_seeds)
    assert campaign_config(path) == campaign_config(CONFIG)
    with pytest.raises(ValueError, match="TRAINING_SEEDS_MUST_BE_41_42_43"):
        load_config(path)


def test_default_loader_preserves_exact_training_seeds():
    assert load_config(CONFIG)["training_seeds"] == [41, 42, 43]


def test_ca3_source_realization_retains_its_own_rng_limits(tmp_path):
    path = write_config(tmp_path, None)
    config = yaml.safe_load(path.read_text())
    realization = SourceRealization("ca3_source", "ca3_source_formation", "formation",
                                    "development", 300000007, 300000007, 1)
    config["ca3_source_realization"] = realization.record()
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    assert load_realization(path) == realization
    invalid = copy.deepcopy(config)
    invalid["ca3_source_realization"]["circuit_seed"] = 0
    path.write_text(yaml.safe_dump(invalid, sort_keys=False))
    with pytest.raises(ValueError, match="CA3_REALIZATION_SEED_INVALID"):
        load_realization(path)


def test_ca3_extension_preflight_is_independent_of_training_seeds(tmp_path, monkeypatch):
    path = write_config(tmp_path, None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    destination = tmp_path / "native"
    with pytest.raises(ValueError, match="CA3_BUILD_REQUIRES_COMPUTE_ALLOCATION"):
        execute(tmp_path / "sources", destination, "89", build_only=True,
                coupled_extension_config=path)
    assert not destination.exists()
    invalid = yaml.safe_load(path.read_text())
    invalid["ca3_coupled_extension"]["population_multiplier"] = 3
    path.write_text(yaml.safe_dump(invalid, sort_keys=False))
    with pytest.raises(ValueError, match="CA3_COUPLED_EXTENSION_CONFIG_REQUIRED"):
        execute(tmp_path / "sources", destination, "89", build_only=True,
                coupled_extension_config=path)
    assert not destination.exists()


@pytest.mark.parametrize("scenario", ("v1_visual", "ca3_pattern_completion"))
def test_real_benchmark_cli_writes_the_unchanged_complete_identity_envelope(tmp_path, scenario):
    path = write_config(tmp_path, [17, 29])
    output = tmp_path / scenario
    result = subprocess.run([sys.executable, "-B", "-m", "population_dynamics.benchmark_sampling",
                             "--config", str(path), "--output", str(output), "--scenario", scenario],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["candidate_rows"] == {scenario: 100000}
    assert report["complete_episodes"] == 0 and report["production_eligible"] is False
    expected = episode_identities(campaign_config(CONFIG), scenario)
    with (output / (scenario + ".csv")).open(newline="") as stream:
        actual = csv.DictReader(stream)
        count = 0
        for identity in expected:
            row = next(actual)
            assert {key: row[key] for key in identity} == {key: str(value) for key, value in identity.items()}
            count += 1
        assert next(actual, None) is None and count == 100000

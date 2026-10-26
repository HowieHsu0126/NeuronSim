"""Canonical V1 block recipes preserve source controls and distinct RNG meanings."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from .benchmark_sampling import prepare_visual_block
from .benchmark_tasks import v1_visual_config
from .circuit_campaign import campaign_config, episode_identities

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CAMPAIGN = CONFIGS / "neural_population_dynamics.yaml"
BASE = CONFIGS / "neural_population_v1_constructed_visual_development.yaml"


def test_visual_block_cli_matches_canonical_conditions_and_noise_identity(tmp_path):
    output = tmp_path / "block"
    subprocess.run([sys.executable, "-B", "-m", "Libs.population_dynamics.benchmark_sampling",
        "--config", str(CAMPAIGN), "--base-config", str(BASE), "--block-scenario", "v1_visual",
        "--circuit-index", "80", "--block-index", "9", "--output", str(output)], check=True)
    report = json.loads((output / "report.json").read_text())
    assert report["intended_split"] == "calibration" and report["actual_execution_scope"] == "development"
    assert report["complete_episodes"] == 0 and report["production_eligible"] is False
    campaign, base = campaign_config(CAMPAIGN), yaml.safe_load(BASE.read_text())
    identities = [row for row in episode_identities(campaign, "v1_visual")
                  if row["circuit_index"] == 80 and row["block_index"] == 9]
    assert len(report["conditions"]) == 10
    for entry, identity in zip(report["conditions"], identities, strict=True):
        assert entry["identity"] == identity
        config = yaml.safe_load((output / entry["config_path"]).read_text())
        assert config == v1_visual_config(base, campaign["scenarios"]["v1_visual"]["task_protocol"], identity)
        assert config["spike_generation"]["starting_noise_seed"] == 130080009
        assert config["background"]["starting_noise_seed"] == 120080009
        assert config["unit_coverage"] == "all_source_units"
    low = yaml.safe_load((output / "low_contrast.yaml").read_text())
    assert low["stimulus"]["direction_deg"] == 45 and low["stimulus"]["contrast_percent"] == 20
    gray = yaml.safe_load((output / "gray.yaml").read_text())
    assert gray["stimulus"]["direction_deg"] == 0 and gray["stimulus"]["contrast_percent"] == 0
    with pytest.raises(FileExistsError):
        prepare_visual_block(CAMPAIGN, BASE, 80, 9, output)


@pytest.mark.parametrize("circuit,block", [(-1, 0), (100, 0), (0, -1), (0, 100), (True, 0)])
def test_visual_block_invalid_indices_do_not_write(tmp_path, circuit, block):
    output = tmp_path / "block"
    with pytest.raises(ValueError, match="INDEX_INVALID"):
        prepare_visual_block(CAMPAIGN, BASE, circuit, block, output)
    assert not output.exists()

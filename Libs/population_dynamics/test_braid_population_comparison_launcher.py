"""Exercise prior checkpoint staging through the real BRAID comparison reader."""
import csv
import json

import pytest

from Libs.scripts.run_braid_population_comparison import _stage_comparison_cell
from .braid_population_comparison import DeliveredCell, ORIGINS, compare
from .provenance import sha256_file
from .test_braid_population_comparison import fixture, prior_population_fixture
from .test_population_forecast_analysis import _set_prior_fixture_profile


def _campaigns(root, profile):
    fixture(root)
    population = root / "population"
    if profile != "ordinary":
        population = prior_population_fixture(root)
    if profile == "predictive_mixture":
        for seed in (41, 42, 43):
            _set_prior_fixture_profile(population / f"seed_{seed}", mixture=True)
    return root / "braid", population


@pytest.mark.parametrize("profile", ["ordinary", "analytic_prior", "predictive_mixture"])
def test_staged_campaigns_compare_with_bound_parent_and_attempt_costs(tmp_path, profile):
    braid, population = _campaigns(tmp_path / "source", profile)
    staged = tmp_path / "staged"
    for kind, campaign in (("braid", braid), ("population", population)):
        for seed in (41, 42, 43):
            source = DeliveredCell(campaign / f"seed_{seed}", seed)
            destination = staged / kind / f"seed_{seed}"
            _stage_comparison_cell(source, kind, destination, ORIGINS, sha256_file)
            assert (destination / "delivery.json").read_bytes() == (source.root / "delivery.json").read_bytes()
            if kind == "population" and profile != "ordinary":
                for name in ("training/best.pt", "parent_training/best.pt", "parent_training/run.json",
                             "prerequisites/prior_parent.json", "training/attempts/attempt_000.json"):
                    assert sha256_file(destination / name) == source.files[name]["sha256"]
    output = tmp_path / "comparison"
    result = compare(staged / "braid", staged / "population", output)
    assert result["score_rows"] == 432 and result["descriptive_block_rows"] == 36
    assert result["scientific_claim_qualified"] is False and result["samples_regenerated"] is False
    if profile == "ordinary":
        assert "candidate_method" not in result
        assert not (staged / "population/seed_41/training/best.pt").exists()
        return
    method = ("population_recurrent_predictive_mixture" if profile == "predictive_mixture"
              else "population_recurrent_prior_refit")
    assert result["candidate_method"] == method
    with (output / "training_costs.csv").open() as stream:
        costs = [row for row in csv.DictReader(stream) if row["method"] == method]
    assert len(costs) == 3
    assert all(float(row["recorded_training_seconds"]) == 190. for row in costs)
    assert all(float(row["parent_training_elapsed_seconds"]) == 180. for row in costs)
    assert all(float(row["incremental_training_elapsed_seconds"]) == 10. for row in costs)
    for seed in (41, 42, 43):
        original = json.loads((population / f"seed_{seed}/delivery.json").read_text())
        lineage = result["population_training_bindings"][str(seed)]["prior_refit_lineage"]
        assert lineage["parent_cell"] == original["parent_cell"]
        assert lineage["parent_delivery_sha256"] == original["parent_delivery_sha256"]


@pytest.mark.parametrize("profile", ["analytic_prior", "predictive_mixture"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_staging_rejects_unavailable_or_changed_prior_checkpoint(tmp_path, profile, damage):
    _, population = _campaigns(tmp_path / "source", profile)
    source = DeliveredCell(population / "seed_41", 41)
    checkpoint = source.root / "training/best.pt"
    if damage == "missing":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    destination = tmp_path / "rejected_stage"
    with pytest.raises((FileNotFoundError, ValueError)):
        _stage_comparison_cell(source, "population", destination, ORIGINS, sha256_file)
    assert not (destination / "delivery.json").exists()

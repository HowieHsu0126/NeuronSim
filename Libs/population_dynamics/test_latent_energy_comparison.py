"""EAG delivery, complete-budget, and shared-comparison consumer contracts."""
import csv
import json
import subprocess
import sys

import pytest
import yaml

from . import latent_energy_comparison as comparison
from .latent_spike_diffusion_comparison import compare as compare_shared
from .provenance import sha256_file
from .test_braid_population_comparison import delivery, write_json, write_rows, prior_population_fixture
from .test_latent_spike_diffusion_comparison import inputs


def _rebind(cell, seed, *, config=True, pointer=True):
    evaluation_path = cell / "forecasts/evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["artifacts"]["scores.csv"] = sha256_file(cell / "forecasts/scores.csv")
    evaluation["artifacts"]["costs.csv"] = sha256_file(cell / "forecasts/costs.csv")
    if config:
        evaluation["binding"]["configuration_sha256"] = sha256_file(
            cell / "Libs/configs/neural_population_latent_energy.yaml")
    if pointer:
        evaluation["binding"]["training_committed_sha256"] = sha256_file(cell / "training/committed.json")
    write_json(evaluation_path, evaluation)
    replay = json.loads((cell / "forecasts/replay.json").read_text())
    replay["evaluation_sha256"] = sha256_file(evaluation_path)
    write_json(cell / "forecasts/replay.json", replay)
    delivery(cell, seed)


@pytest.fixture
def eag_inputs(inputs):
    source, population, output = inputs
    eag = source.with_name("eag")
    source.rename(eag)
    config = dict(schema="neural_population_latent_energy_training_v1",
        training_seeds=[41,42,43], fit_blocks=list(range(14)), validation_blocks=list(range(14,18)),
        bin_ms=5., complete_episode_bins=1200, forecast_origins_bins=[200,600,1000], forecast_bins=100,
        expected_poisson_rng_family="per_edge_philox4_32_10",
        model=dict(bin_seconds=.005,complete_episode_bins=1200,forecast_bins=100),
        autoencoder=dict(maximum_epochs=260), energy=dict(maximum_epochs=4000,source_batch_size=256),
        sampling=dict(joint_samples=16))
    denominators = dict(fit_complete_episodes=56,validation_complete_episodes=16,
                        fit_origin_windows=168,validation_origin_windows=48,future_bins=100,units=100000)
    for seed in (41,42,43):
        cell = eag / f"seed_{seed}"
        config_path = cell / "Libs/configs/neural_population_latent_energy.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(config))
        checkpoint = cell / "training/checkpoints/complete_000001.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic saved EAG checkpoint; no model inference\n")
        checkpoint_sha256 = sha256_file(checkpoint)
        write_json(cell / "training/committed.json", dict(checkpoint="checkpoints/complete_000001.pt",
                   sha256=checkpoint_sha256,stage="complete",next_epoch=4000,commits=1))
        run = json.loads((cell / "training/run.json").read_text())
        run.pop("units")
        run.update(source_denominators=denominators,autoencoder_completed_epochs=260,
                   energy_completed_passes=4000,energy_presentations=672000,energy_optimizer_steps=2625,
                   completed_stage_epochs=4000,energy_epoch_rows=4000,
                   latest_checkpoint_sha256=checkpoint_sha256)
        write_json(cell / "training/run.json", run)
        with (cell / "forecasts/scores.csv").open() as stream:
            scores = list(csv.DictReader(stream))
        for row in scores:
            row["method"] = "eag"
            row["energy_score"] = 2.
            row["refractory_count_violation_1bin"] = .25
        write_rows(cell / "forecasts/scores.csv", scores)
        evaluation = json.loads((cell / "forecasts/evaluation.json").read_text())
        evaluation.update(method="eag",seed=seed,source_unit_count=100000,
                          observation_law="source_softplus_poisson",native_refractory_qualified=False)
        evaluation["binding"].update(method="eag",origins_bins=[200,600,1000],horizons_ms=[20.,80.,500.])
        write_json(cell / "forecasts/evaluation.json", evaluation)
        replay = json.loads((cell / "forecasts/replay.json").read_text())
        replay.update(source_unit_count=100000,count_values_digest_replayed=48*160000000)
        write_json(cell / "forecasts/replay.json", replay)
        _rebind(cell, seed)
    return eag, population, output


def test_complete_eag_cli_preserves_labels_costs_prior_budget_and_violation_metrics(eag_inputs):
    prior_population_fixture(eag_inputs[1].parent)
    child = subprocess.run([sys.executable,"-B","-m","Libs.population_dynamics.latent_energy_comparison",
        "--eag-root",str(eag_inputs[0]),"--population-root",str(eag_inputs[1]),"--output",str(eag_inputs[2])],
        capture_output=True,text=True)
    assert child.returncode == 0, child.stderr
    report = json.loads((eag_inputs[2] / "comparison.json").read_text())
    assert report["matched_rows"] == 432 and report["block_rows"] == 36
    assert report["baseline_method"] == "eag"
    assert report["baseline_observation_law"] == "source_softplus_poisson"
    assert report["baseline_native_refractory_qualified"] is False
    assert not report["speedup_qualified"] and not report["scientific_claim_qualified"]
    assert report["candidate_method"] == "population_recurrent_prior_refit"
    assert (eag_inputs[2] / "eag_costs_by_origin.csv").is_file()
    assert not (eag_inputs[2] / "ldns_costs_by_origin.csv").exists()
    assert all(set(report["bindings"][str(seed)]) == {"eag","population"} for seed in (41,42,43))
    with (eag_inputs[2] / "joint_comparison_all_seeds.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    assert len(scores) == 432
    assert all(not any("ldns" in key for key in row) for row in scores)
    assert all(float(row["eag_unit_bin_mean_mse_corrected"]) == -.01 for row in scores)
    assert all(float(row["population_minus_eag_energy_score"]) == -1. for row in scores)
    assert all(float(row["eag_refractory_count_violation_1bin"]) == .25 for row in scores)
    assert all(float(row["population_minus_eag_refractory_count_violation_1bin"]) == .75 for row in scores)
    with (eag_inputs[2] / "training_costs.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    baseline = [row for row in costs if row["method"] == "eag"]
    assert len(baseline) == 3
    assert all(row["training_budget"] == "260AE+4000energypasses;672000presentations;2625updates" for row in baseline)
    assert all(float(row["recorded_training_seconds"]) == 200. for row in baseline)
    assert all(int(row["energy_presentations"]) == 672000 and int(row["energy_optimizer_steps"]) == 2625 for row in baseline)
    assert all(float(row["recorded_training_seconds"]) == 190. for row in costs if row["method"] != "eag")
    with (eag_inputs[2] / "costs_by_origin.csv").open() as stream:
        origins = list(csv.DictReader(stream))
    assert len(origins) == 144
    assert sum(float(row["eag_charged_episode_loading_seconds"] or 0) for row in origins) == 24.
    assert sum(float(row["eag_accounted_origin_total_seconds"]) for row in origins) == 168.
    assert all("eag_hardware" in row and "ldns_hardware" not in row for row in origins)


@pytest.mark.parametrize("counter", ["autoencoder_completed_epochs","energy_completed_passes",
    "energy_presentations","energy_optimizer_steps","completed_stage_epochs","energy_epoch_rows"])
def test_incomplete_eag_training_budget_rejected(eag_inputs, counter):
    cell = eag_inputs[0] / "seed_41"
    path = cell / "training/run.json"
    run = json.loads(path.read_text())
    run[counter] -= 1
    write_json(path, run)
    _rebind(cell, 41)
    with pytest.raises(ValueError, match="EAG training budget"):
        comparison.compare(*eag_inputs)
    assert not eag_inputs[2].exists()


@pytest.mark.parametrize("denominator", ["fit_complete_episodes","validation_complete_episodes",
    "fit_origin_windows","validation_origin_windows","future_bins","units"])
def test_incomplete_eag_source_coverage_rejected(eag_inputs, denominator):
    cell = eag_inputs[0] / "seed_41"
    path = cell / "training/run.json"
    run = json.loads(path.read_text())
    run["source_denominators"][denominator] -= 1
    write_json(path, run)
    _rebind(cell, 41)
    with pytest.raises(ValueError, match="EAG source coverage"):
        comparison.compare(*eag_inputs)
    assert not eag_inputs[2].exists()


@pytest.mark.parametrize("location", ["evaluation","binding","scores"])
def test_mislabelled_eag_method_rejected(eag_inputs, location):
    cell = eag_inputs[0] / "seed_41"
    if location == "scores":
        path = cell / "forecasts/scores.csv"
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["method"] = "ldns"
        write_rows(path, rows)
    else:
        path = cell / "forecasts/evaluation.json"
        evaluation = json.loads(path.read_text())
        (evaluation if location == "evaluation" else evaluation["binding"])["method"] = "ldns"
        write_json(path, evaluation)
    _rebind(cell, 41)
    with pytest.raises(ValueError):
        comparison.compare(*eag_inputs)
    assert not eag_inputs[2].exists()


@pytest.mark.parametrize("change", ["config_bytes","config_budget","pointer_bytes",
    "pointer_stage","pointer_digest","checkpoint_bytes","latest_digest"])
def test_changed_eag_config_pointer_or_checkpoint_binding_rejected(eag_inputs, change):
    cell = eag_inputs[0] / "seed_41"
    if change in ("config_bytes","config_budget"):
        path = cell / "Libs/configs/neural_population_latent_energy.yaml"
        config = yaml.safe_load(path.read_text())
        if change == "config_bytes":
            config["fixture_note"] = "changed after evaluation"
        else:
            config["energy"]["maximum_epochs"] = 3999
        path.write_text(yaml.safe_dump(config))
    elif change in ("pointer_bytes","pointer_stage","pointer_digest"):
        path = cell / "training/committed.json"
        pointer = json.loads(path.read_text())
        if change == "pointer_bytes":
            pointer["commits"] += 1
        elif change == "pointer_stage":
            pointer["stage"] = "energy"
        else:
            pointer["sha256"] = "0" * 64
        write_json(path, pointer)
    elif change == "checkpoint_bytes":
        (cell / "training/checkpoints/complete_000001.pt").write_bytes(b"changed declared checkpoint\n")
    else:
        path = cell / "training/run.json"
        run = json.loads(path.read_text())
        run["latest_checkpoint_sha256"] = "0" * 64
        write_json(path, run)
    _rebind(cell, 41, config=change != "config_bytes", pointer=change != "pointer_bytes")
    with pytest.raises(ValueError):
        comparison.compare(*eag_inputs)
    assert not eag_inputs[2].exists()


def test_unregistered_baseline_kind_rejects_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="baseline kind"):
        compare_shared(tmp_path/"missing", tmp_path/"population", tmp_path/"out", baseline_kind="other")
    assert not (tmp_path/"out").exists()

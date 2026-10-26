"""Frozen field arrays and accepted receipts exercise the complete diagnostic."""
import json
import os
import sys

import numpy as np
import pytest

from . import population_native_forecast_decomposition as decomposition
from .lif_neural_field import sampled_mean_field_mse
from .provenance import sha256_file
from .trajectory_scores import energy_score
from .test_population_residual_correction_analysis import (
    evidence, native_evidence, ldns_evidence, paths, read_rows, write_rows, write_json,
    refresh, refresh_ldns, refresh_ldns_field,
)


def bind_files(cell, *names):
    receipt = json.loads((cell / "delivery.json").read_text())
    index = {row["path"]: row for row in receipt["files"]}
    for name in names:
        path = cell / name
        index[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))
    receipt.update(files=list(index.values()), scientific_claim_qualified=False)
    write_json(cell / "delivery.json", receipt)


def reseal(data):
    """Bind intentional fixture changes before testing the semantic contract."""
    ldns_root, field_root = data["ldns_comparison"], data["field_comparison"]
    report_path, field_path = ldns_root / "analysis/comparison.json", field_root / "comparison.json"
    report, field = json.loads(report_path.read_text()), json.loads(field_path.read_text())
    scale_digest = sha256_file(data["field_scale"])
    for seed in (41, 42, 43):
        baseline = data["ldns_forecasts"] / f"seed_{seed}"
        proof = json.loads((baseline / "delivery.json").read_text())
        report["bindings"][str(seed)]["ldns"] = dict(delivery_sha256=sha256_file(baseline / "delivery.json"),
            consumed_files={row["path"]: row["sha256"] for row in proof["files"]})
        for binding in (report["bindings"][str(seed)]["population"], field["bindings"][str(seed)]["baseline_population"]):
            binding["consumed_files"]["forecasts/fitting_field_scale.npz"] = scale_digest
        field["bindings"][str(seed)]["parent"]["delivery_sha256"] = sha256_file(data["parent_campaign"] / f"seed_{seed}/delivery.json")
        field["bindings"][str(seed)]["corrections"]["delivery_sha256"] = sha256_file(data["corrections"] / f"seed_{seed}/delivery.json")
    write_json(report_path, report)
    write_json(field_path, field)
    refresh_ldns(ldns_root)
    refresh_ldns_field(field_root)


@pytest.fixture
def forecast_evidence(ldns_evidence, tmp_path):
    corrections, parents, ldns_root, field_root = ldns_evidence
    data = dict(corrections=corrections, parent_campaign=parents, ldns_comparison=ldns_root,
        field_comparison=field_root, ldns_forecasts=tmp_path / "ldns_forecasts", field_scale=tmp_path / "fit_scale.npz")
    units = np.full(128, 781, dtype=np.int64)
    units[0] += 100000 - units.sum()
    mean, sd = np.linspace(.2, .4, 128), np.linspace(1., 2., 128)
    ids = np.asarray([f"field_{index:03d}" for index in range(128)])
    np.savez(data["field_scale"], mean_hz=mean, standard_deviation_hz=sd, unit_count=units, field_ids=ids)
    truth = np.full((100, 128), 2, dtype=np.int64)
    samples = {"ldns": np.stack([truth + (-1 if index % 2 else 1) for index in range(16)]),
               "current": np.stack([truth + (-2 if index % 2 else 2) for index in range(16)]),
               "history": np.repeat(truth[None], 16, axis=0)}
    templates = tmp_path / "templates"
    templates.mkdir()
    expected_energy = {}
    for model, draws in samples.items():
        arrays = dict(sampled_field_counts=draws, observed_field_counts=truth)
        if model == "ldns":
            arrays.update(field_ids=ids, unit_ids=np.arange(1, 100001, dtype=np.int64))
        np.savez_compressed(templates / f"{model}.npz", **arrays)
        normalized = (draws / (units * 5. / 1000) - mean) / sd
        target = (truth / (units * 5. / 1000) - mean) / sd
        expected_energy[model] = {horizon: energy_score(normalized[:, :int(horizon / 5)], target[:int(horizon / 5)])
                                  for horizon in (20., 80., 500.)}
    for seed in (41, 42, 43):
        baseline = data["ldns_forecasts"] / f"seed_{seed}"
        write_json(baseline / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
            scientific_claim_qualified=False, files=[]))
        original_scores = read_rows(paths(parents / f"seed_{seed}", "original")[0])
        for model in ("ldns", "current", "history"):
            cell = baseline if model == "ldns" else corrections / f"seed_{seed}"
            prefix = "forecasts/" if model == "ldns" else f"results/{model}/forecasts/"
            names, artifacts = [], {}
            for raw in original_scores:
                if float(raw["horizon_ms"]) != 20.:
                    continue
                filename = f"{raw['case_id']}_origin_{float(raw['origin_ms']):g}.npz"
                name = prefix + filename
                path = cell / name
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(templates / f"{model}.npz", path)
                names.append(name)
                artifacts[filename] = sha256_file(path)
            bind_files(cell, *names)
            if model != "ldns":
                evaluation_path = cell / prefix / "evaluation.json"
                evaluation = json.loads(evaluation_path.read_text())
                evaluation["artifacts"] = artifacts
                write_json(evaluation_path, evaluation)
        for model in ("original", "current", "history"):
            cell = parents / f"seed_{seed}" if model == "original" else corrections / f"seed_{seed}"
            scores_path = paths(cell, model)[0]
            rows = read_rows(scores_path)
            for row in rows:
                horizon = float(row["horizon_ms"])
                row["observed_spikes_per_unit"] = float(truth[:int(horizon / 5)].sum()) / 100000
                if model != "original":
                    row["energy_score"] = expected_energy[model][horizon]
            write_rows(scores_path, rows)
            bind_files(cell)
            refresh(cell, model)
    table = ldns_root / "analysis/joint_comparison_all_seeds.csv"
    rows = read_rows(table)
    for row in rows:
        horizon = float(row["horizon_ms"])
        row["ldns_energy_score"] = expected_energy["ldns"][horizon]
        row["ldns_observed_spikes_per_unit"] = float(truth[:int(horizon / 5)].sum()) / 100000
    write_rows(table, rows)
    reseal(data)
    return data


def analyze_fixture(data, output):
    return decomposition.analyze(**data, output=output, execution=dict(backend="merlin", run_id="fixture", trial_id="fixture"))


def test_field_components_preserve_mean_variance_identity_and_negative_correction():
    units = np.full(128, 781, dtype=np.int64)
    units[0] += 100000 - units.sum()
    mean, sd = np.linspace(.2, .4, 128), np.linspace(1., 2., 128)
    truth = np.full((100, 128), 2, dtype=np.int64)
    samples = np.stack([truth + (-1 if index % 2 else 1) for index in range(16)])
    for bins in (4, 16, 100):
        result = decomposition.field_components(samples[:, :bins], truth[:bins], mean, sd, units)
        expected = sampled_mean_field_mse(samples[:, :bins], truth[:bins], sd, units, 5.)
        assert result["field_mean_mse_corrected"] == expected
        assert expected < 0 and result["energy_score"] > 0
        assert result["field_mean_mse_corrected"] == pytest.approx(
            result["field_mean_mse_uncorrected"] - result["standardized_generated_variance"] / 16, abs=1e-15)
        assert result["energy_reconstruction"] == pytest.approx(result["energy_score"], abs=1e-15)


def test_complete_frozen_decomposition_preserves_all_conditions(forecast_evidence, tmp_path, monkeypatch):
    original_path = decomposition.DeliveredCell.path
    def without_full_unit_samples(self, name):
        assert "full_unit_samples" not in name and not name.endswith(".pt")
        return original_path(self, name)
    monkeypatch.setattr(decomposition.DeliveredCell, "path", without_full_unit_samples)
    output = tmp_path / "decomposition"
    execution = dict(backend="merlin", run_id="fixture", trial_id="fixture")
    monkeypatch.setattr(decomposition, "cpu_execution_provenance", lambda: execution)
    command = ["forecast-decomposition", "--output", str(output)]
    for name, path in forecast_evidence.items():
        command += ["--" + name.replace("_", "-"), str(path)]
    monkeypatch.setattr(sys, "argv", command)
    decomposition.main()
    report = json.loads((output / "analysis.json").read_text())
    assert report["execution"] == execution
    assert report["status"] == "COMPLETE_DESCRIPTIVE_NATIVE_FORECAST_DECOMPOSITION"
    assert (report["score_rows"], report["block_rows"], report["seed_horizon_rows"]) == (1296, 108, 27)
    rows = read_rows(output / "forecast_decomposition_all_seeds.csv")
    assert len(rows) == 1296
    assert {row["model"] for row in rows} == {"ldns", "current", "history"}
    assert {row["seed"] for row in rows} == {"41", "42", "43"}
    assert len({(row["model"], row["seed"], row["case_id"], row["origin_ms"], row["horizon_ms"]) for row in rows}) == 1296
    assert all(float(row["field_mean_mse_corrected"]) < 0 for row in rows if row["model"] == "current")
    blocks = read_rows(output / "forecast_decomposition_by_block.csv")
    summaries = read_rows(output / "forecast_decomposition_by_seed_horizon.csv")
    assert len(blocks) == 108 and {row["conditions"] for row in blocks} == {"12"}
    assert len(summaries) == 27 and {row["conditions"] for row in summaries} == {"48"}
    assert report["energy_parity"]["verified_rows"] == 1296
    assert report["energy_parity"]["maximum_absolute_difference"] < 1e-12
    assert report["energy_parity"]["maximum_reconstruction_absolute_difference"] < 1e-12
    assert report["exact_source_truth_match"] and report["negative_corrected_estimates_retained"]
    assert not report["training_performed"] and not report["generation_performed"] and not report["full_unit_samples_read"]
    assert not report["scientific_claim_qualified"] and not report["inferential_statistics_performed"]
    for binding in report["consumed"].values():
        assert len(binding["ldns"]["consumed_files"]) == 48
        assert len(binding["corrections"]["consumed_files"]) == 96


@pytest.mark.parametrize("artifact", ["scale", "ldns_delivery", "ldns_array", "native_array"])
def test_decomposition_rejects_changed_manifest_inputs(forecast_evidence, tmp_path, artifact):
    data = forecast_evidence
    filename = "train_block_014_familiar_0_origin_1000.npz"
    path = {"scale": data["field_scale"], "ldns_delivery": data["ldns_forecasts"] / "seed_41/delivery.json",
            "ldns_array": data["ldns_forecasts"] / "seed_41/forecasts" / filename,
            "native_array": data["corrections"] / "seed_41/results/current/forecasts" / filename}[artifact]
    # Replace this link only; all other frozen fixture origins remain intact.
    original = path.read_bytes()
    path.unlink()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="(scale binding differs|forecast delivery differs|input changed)"):
        analyze_fixture(data, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["source", "missing_array_binding", "field_map"])
def test_decomposition_requires_accepted_source_and_complete_array_identity(forecast_evidence, tmp_path, change):
    data = forecast_evidence
    if change == "source":
        path = data["field_comparison"] / "comparison.json"
        report = json.loads(path.read_text())
        report["source"]["dataset_sha256"] = "0" * 64
        write_json(path, report)
        refresh_ldns_field(data["field_comparison"])
    elif change == "missing_array_binding":
        path = data["ldns_comparison"] / "analysis/comparison.json"
        report = json.loads(path.read_text())
        report["bindings"]["41"]["ldns"]["consumed_files"].pop("forecasts/train_block_014_familiar_0_origin_1000.npz")
        write_json(path, report)
        refresh_ldns(data["ldns_comparison"])
    else:
        cell = data["ldns_forecasts"] / "seed_41"
        name = "forecasts/train_block_014_familiar_0_origin_1000.npz"
        path = cell / name
        with np.load(path, allow_pickle=False) as values:
            arrays = {key: values[key].copy() for key in values.files}
        arrays["field_ids"] = arrays["field_ids"][::-1]
        path.unlink()
        np.savez_compressed(path, **arrays)
        bind_files(cell, name)
        reseal(data)
    with pytest.raises(ValueError, match="(source or correction fit differs|accepted score binding|field or unit map differs)"):
        analyze_fixture(data, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["truth", "shape", "energy"])
def test_decomposition_rejects_truth_shape_and_saved_metric_drift(forecast_evidence, tmp_path, change):
    data = forecast_evidence
    cell = data["corrections"] / "seed_41"
    scores_path, evaluation_path, _ = paths(cell, "current")
    if change == "energy":
        rows = read_rows(scores_path)
        rows[0]["energy_score"] = float(rows[0]["energy_score"]) + 1
        write_rows(scores_path, rows)
    else:
        name = "results/current/forecasts/train_block_014_familiar_0_origin_1000.npz"
        path = cell / name
        with np.load(path, allow_pickle=False) as values:
            arrays = {key: values[key].copy() for key in values.files}
        if change == "truth":
            arrays["observed_field_counts"][0, 0] += 1
        else:
            arrays["sampled_field_counts"] = arrays["sampled_field_counts"][:15]
        path.unlink()
        np.savez_compressed(path, **arrays)
        bind_files(cell, name)
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["artifacts"][path.name] = sha256_file(path)
        write_json(evaluation_path, evaluation)
    refresh(cell, "current")
    reseal(data)
    with pytest.raises(ValueError, match="(source truth differ|sixteen-draw integer|saved energy score drift)"):
        analyze_fixture(data, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


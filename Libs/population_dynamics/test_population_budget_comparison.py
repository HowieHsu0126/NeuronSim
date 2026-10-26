"""Compare complete saved recurrent forecasts at registered training ceilings."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from . import population_forecast_analysis as analysis
from .provenance import sha256_file
from .test_population_continuation_cost import continuation_fixture, row, write_json


def write_csv(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def deliver_analysis(root, report):
    write_json(root / "analysis/analysis.json", report)
    names = ("analysis/analysis.json", "analysis/forecast_all_seeds.csv", "analysis/costs_by_origin.csv")
    write_json(root / "delivery.json", dict(status="PASS", exit_code=0, mode="complete", training_seeds=list(analysis.SEEDS),
        scientific_claim_qualified=False, calibration_or_test_accessed=False, files=[row(root, name) for name in names]))


@pytest.fixture
def budget_pair(tmp_path):
    common = dict(representation="recurrent", maximum_epochs=20, base_maximum_epochs=20, encoder_size=16, latent_channels=4)
    config = yaml.safe_load(analysis.CONFIG_PATH.read_text())
    field_metrics, other_metrics = analysis._metric_names(config)
    for seed, parent_attempt in ((41, 0), (42, 1), (43, 1)):
        parent, child, parent_run, child_run, parent_delivery, child_delivery = continuation_fixture(tmp_path, seed, parent_attempt)
        for run, ceiling in ((parent_run, 20), (child_run, 80)):
            run.update(model_family="population", representation="recurrent", parameter_count=520664,
                best_checkpoint_sha256="selected" + str(seed) + str(ceiling), best_epoch=ceiling - 1,
                completed_epochs=ceiling, effective_maximum_epochs=ceiling, stop_reason="epoch_budget",
                peak_process_rss_bytes=1000, hardware=dict(host="fixture", device="cpu"))
        write_json(parent / "training/run.json", parent_run)
        parent_delivery["files"] = [row(parent, record["path"]) for record in parent_delivery["files"]]
        write_json(parent / "delivery.json", parent_delivery)
        child_run["continuation"].update(parent_artifact_sha256={"run.json": sha256_file(parent / "training/run.json")},
            parent_effective_maximum_epochs=20, inherited_early_stopping_patience=4)
        write_json(child / "training/run.json", child_run)
        shutil.copy2(parent / "delivery.json", child / "prerequisites/budget_parent/delivery.json")
        shutil.copy2(parent / "training/run.json", child / "prerequisites/budget_parent/training/run.json")
        child_delivery["optimization_budget_continuation"]["parent_delivery_sha256"] = sha256_file(parent / "delivery.json")
        child_delivery["files"] = [row(child, record["path"]) for record in child_delivery["files"]]
        child_delivery["files"].append(row(child, "training/run.json"))
        write_json(child / "delivery.json", child_delivery)
    roots = []
    for family, ceiling, offset in (("parent", 20, 3.), ("child", 80, 1.)):
        root, campaign = tmp_path / ("analysis_" + family), tmp_path / family
        bindings, deliveries, rows, costs = {}, {}, [], []
        for seed in analysis.SEEDS:
            cell = campaign / f"seed_{seed}"
            run = json.loads((cell / "training/run.json").read_text())
            bindings[f"population_{seed}"] = dict(training_config=run["binding"]["config"],
                training_report_sha256=sha256_file(cell / "training/run.json"), checkpoint_sha256=run["best_checkpoint_sha256"],
                executed_training_budget=dict(completed_epochs=ceiling, best_epoch=ceiling-1, stop_reason="epoch_budget", effective_maximum_epochs=ceiling))
            deliveries[str(seed)] = sha256_file(cell / "delivery.json")
        for seed, block, template, origin in itertools.product(analysis.SEEDS, analysis.BLOCKS, range(4), config["origins_ms"]):
            case = f"train_block_{block:03d}_familiar_{template}"
            cost = dict(seed=seed, block=block, case_id=case, template_id=f"familiar_{template}", origin_ms=origin,
                method="population", generated_horizon_ms=500., inference_seconds=offset,
                full_unit_generation_seconds=offset+1, field_projection_seconds=offset+2, peak_gpu_allocated_bytes=None)
            costs.append(cost)
            for horizon in config["horizons_ms"]:
                values = {key: cost[key] for key in ("seed", "block", "case_id", "template_id", "origin_ms", "method")}
                values.update(horizon_ms=horizon, ensemble_artifact=f"{case}_origin_{origin:g}.npz")
                for metric in field_metrics + other_metrics:
                    if metric in analysis.COST_METRICS: values[metric] = cost[metric]
                    elif metric.startswith("variogram_temporal_lag_") and int(metric.rsplit("_", 1)[1]) >= horizon/5: values[metric] = None
                    elif metric.startswith("refractory_count_violation_"): values[metric] = 0.
                    elif metric == "unit_total_coverage_90": values[metric] = .9
                    elif metric == "unit_bin_mean_mse_corrected": values[metric] = -offset + (block - 14) * .1
                    elif metric == "energy_score": values[metric] = offset if template < 2 else 5.-offset
                    else: values[metric] = offset + template*.1
                rows.append(values)
        write_csv(root / "analysis/forecast_all_seeds.csv", rows)
        write_csv(root / "analysis/costs_by_origin.csv", costs)
        report = dict(status="REPLAYED_DEVELOPMENT", models=["population"], seeds=list(analysis.SEEDS), score_rows=432,
            cost_origin_rows=144, independent_blocks=4, scientific_claim_qualified=False, source_fidelity_qualified=False,
            formal_test_accessed=False, all_unit_metrics_independently_replayed=True, all_observed_field_counts_checked_against_source=True,
            configuration=config, common_training_budget=dict(common, maximum_epochs=ceiling), inputs=dict(forecasts=str(campaign)),
            model_seed_bindings=bindings, campaign_deliveries=deliveries,
            source_binding=dict(dataset_sha256="shared", field_bindings=dict(fixture="shared_map")),
            code_sha256=dict(population_state="shared_numerical_source"), artifacts={name:sha256_file(root / "analysis" / name)
                for name in ("forecast_all_seeds.csv", "costs_by_origin.csv")})
        deliver_analysis(root, report)
        roots.append(root)
    return roots


def test_budget_comparison_real_cli_preserves_mixed_signed_metrics_and_counts_costs_once(budget_pair, tmp_path):
    output = tmp_path / "comparison"
    result = subprocess.run([sys.executable, "-m", "population_dynamics.population_forecast_analysis", "--budget-comparison",
        "--population-analysis", str(budget_pair[1]), "--population-analysis", str(budget_pair[0]), "--output", str(output)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "analysis.json").read_text())
    assert report["comparison_mode"] == "training_budget" and report["executed_training_ceilings"] == [20,80]
    assert report["models"] == ["population_recurrent_budget20", "population_recurrent_budget80"]
    assert report["paired_forecast_conditions"] == 432 and report["score_rows"] == 864 and report["cost_fidelity_origin_rows"] == 288
    assert report["scientific_claim_qualified"] is report["state_selection_enabled"] is False
    paired = read_csv(output / "budget_paired_conditions.csv")
    assert {float(r["continuation_minus_parent"]) for r in paired if r["metric"] == "energy_score"} == {-2., 2.}
    assert all(float(r["parent"]) < 0 and float(r["continuation"]) < 0 for r in paired if r["metric"] == "unit_bin_mean_mse_corrected")
    child = [r for r in read_csv(output / "budget_training.csv") if r["method"] == "population_recurrent_budget80"]
    assert {int(r["seed"]): float(r["training_elapsed_seconds"]) for r in child} == {41:105.,42:125.,43:125.}
    assert all(float(r["incremental_training_elapsed_seconds"]) == 5. for r in child)
    costs = read_csv(output / "budget_cost_differences.csv")
    assert all(r["continuation_minus_parent"] == "" for r in costs if r["metric"] == "origin_total_seconds")
    assert not any("p_value" in r or "q_value" in r for r in paired)
    for name, digest in report["artifacts"].items(): assert sha256_file(output / name) == digest


@pytest.mark.parametrize("failure", ["source", "code", "measurement", "budget", "wrong_parent", "wrong_source", "representation"])
def test_budget_comparison_rejects_changed_bound_protocol_or_parent(budget_pair, tmp_path, failure):
    root = budget_pair[1]
    report = json.loads((root / "analysis/analysis.json").read_text())
    if failure == "source": report["source_binding"]["dataset_sha256"] = "foreign"
    elif failure == "code": report["code_sha256"]["population_state"] = "foreign"
    elif failure == "measurement": report["configuration"]["sampling_seed_base"] += 1
    elif failure == "budget": report["common_training_budget"]["maximum_epochs"] = 79
    else:
        cell = Path(report["inputs"]["forecasts"]) / "seed_41"
        run = json.loads((cell / "training/run.json").read_text())
        if failure == "wrong_parent": run["continuation"]["parent_checkpoint_sha256"] = "f"*64
        elif failure == "wrong_source": run["trainer_sources"] = {"population_state.py":"foreign"}
        else:
            run["representation"] = "fixed"
            run["binding"]["config"]["representation"] = "fixed"
            report["common_training_budget"]["representation"] = "fixed"
            report["model_seed_bindings"]["population_41"]["training_config"]["representation"] = "fixed"
        write_json(cell / "training/run.json", run)
        proof = json.loads((cell / "delivery.json").read_text())
        proof["files"] = [row(cell, record["path"]) for record in proof["files"]]
        write_json(cell / "delivery.json", proof)
        report["campaign_deliveries"]["41"] = sha256_file(cell / "delivery.json")
        report["model_seed_bindings"]["population_41"]["training_report_sha256"] = sha256_file(cell / "training/run.json")
    deliver_analysis(root, report)
    output = tmp_path / ("failed_" + failure)
    with pytest.raises(ValueError): analysis.compare_population_budget_analyses(budget_pair, output)
    assert json.loads((output / "analysis.json").read_text())["status"] == "FAILED"
    assert not (output / "forecast_all_seeds.csv").exists()


def test_default_representation_mode_still_rejects_two_recurrent_analyses(budget_pair, tmp_path):
    with pytest.raises(ValueError, match="one fixed and one recurrent"):
        analysis.compare_population_analyses(budget_pair, tmp_path / "default_rejection")


def test_budget_cli_requires_exactly_two_saved_analyses_before_output(tmp_path):
    output = tmp_path / "invalid"
    result = subprocess.run([sys.executable, "-m", "population_dynamics.population_forecast_analysis", "--budget-comparison",
        "--population-analysis", str(tmp_path / "one"), "--output", str(output)], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and not output.exists()

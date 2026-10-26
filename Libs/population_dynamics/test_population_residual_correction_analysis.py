"""Small receipt/CSV fixtures for complete, nonselective correction comparison."""
import csv
import json
from pathlib import Path
import sys

import pytest

from . import population_residual_correction_analysis as analysis
from .provenance import sha256_file
from .population_residual_refit_provenance import FAMILY_SCHEMA, FIT_SCHEMA, build_refit_receipt, refit_identity

METRICS = (
    "energy_score", "unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
    "unit_total_coverage_90", "unit_total_interval_width_90", "refractory_count_violation_1bin",
    "refractory_count_violation_2bin", "refractory_count_violation_4bin", "inference_seconds",
    "full_unit_generation_seconds", "field_projection_seconds", "peak_gpu_allocated_bytes",
    "variogram_spatial", "variogram_cross_population", "variogram_temporal_lag_1",
    "variogram_temporal_lag_4", "variogram_temporal_lag_16",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paths(cell, model):
    base = cell / ("forecasts" if model == "original" else f"results/{model}/forecasts")
    return base / "scores.csv", base / "evaluation.json", cell / ("replay.json" if model == "original" else f"results/{model}/replay.json")


def refresh(cell, model):
    """Update small-file receipts after an intentional semantic corruption."""
    scores_path, evaluation_path, replay_path = paths(cell, model)
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["scores_sha256"] = sha256_file(scores_path)
    write_json(evaluation_path, evaluation)
    replay = json.loads(replay_path.read_text())
    replay.update(evaluation_sha256=sha256_file(evaluation_path), scores_sha256=sha256_file(scores_path),
                  source_checkpoint_sha256=evaluation["checkpoint_sha256"],
                  dataset_sha256=evaluation["dataset_sha256"], acquisition_design_sha256=evaluation["acquisition_design_sha256"])
    write_json(replay_path, replay)
    delivery = json.loads((cell / "delivery.json").read_text())
    index = {row["path"]: row for row in delivery["files"]}
    for path in (scores_path, evaluation_path, replay_path):
        relative = str(path.relative_to(cell))
        index[relative] = dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))
    delivery["files"] = list(index.values())
    if model != "original":
        delivery["modes"][model].update(evaluation_sha256=sha256_file(evaluation_path), replay_sha256=sha256_file(replay_path))
    write_json(cell / "delivery.json", delivery)


@pytest.fixture
def evidence(tmp_path):
    parents, corrections = tmp_path / "parents", tmp_path / "corrections"
    fit_receipt = dict(status="FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS", job_id="fixture_fit",
        parent_training=False, config=dict(training_seeds=[41, 42, 43], modes=["current", "history"],
                                          alpha_by_mode=dict(current=1., history=1.)))
    config = dict(origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.], joint_samples=16,
                  formal_evaluation=False, finite_ensemble_corrections=True, temporal_lags_bins=[1, 4, 16],
                  refractory_support_windows_bins=[1, 2, 4], field_scale="fitting_block_pooled_standard_deviation_of_per_unit_hz",
                  causal_history="all_native_bins_before_origin", sampling_seed_base=86000,
                  unit_metric_dtype="float32", field_metric_dtype="float64")
    for seed in (41, 42, 43):
        checkpoint = f"{seed:064x}"
        parent_cell, corrected_cell = parents / f"seed_{seed}", corrections / f"seed_{seed}"
        fit_path = corrected_cell / "provenance/fit.json"
        write_json(fit_path, fit_receipt)
        fit_row = dict(path="provenance/fit.json", bytes=fit_path.stat().st_size, sha256=sha256_file(fit_path))
        parent_delivery = dict(status="PASS", exit_code=0, training_seed=seed,
            files=[dict(path="training/best.pt", bytes=10**9, sha256=checkpoint)])
        correction_delivery = dict(status="PASS", exit_code=0, training_seed=seed,
            modes={mode: dict(status="PASS", exit_code=0) for mode in ("current", "history")},
            provenance=dict(parent_checkpoint_sha256=checkpoint,
                            fit_sha256=fit_row["sha256"], fit_job_id=fit_receipt["job_id"]), files=[fit_row])
        write_json(parent_cell / "delivery.json", parent_delivery)
        write_json(corrected_cell / "delivery.json", correction_delivery)
        for model in ("original", "current", "history"):
            cell = parent_cell if model == "original" else corrected_cell
            scores_path, evaluation_path, replay_path = paths(cell, model)
            device = "cuda" if model == "original" else "cpu"
            method = "population" if model == "original" else "population_residual_" + model
            offset = {"original": 0., "current": .2, "history": -.1}[model]
            rows = []
            for block in (14, 15, 16, 17):
                for template in range(4):
                    case = f"train_block_{block:03d}_familiar_{template}"
                    for origin in (1000., 3000., 5000.):
                        for horizon in (20., 80., 500.):
                            row = dict(seed=seed, block=block, case_id=case, template_id=f"familiar_{template}",
                                       origin_ms=origin, horizon_ms=horizon, method=method,
                                       ensemble_artifact=f"{case}_origin_{origin:g}.npz")
                            for index, metric in enumerate(METRICS):
                                value = .05 + index * .001
                                if metric == "energy_score":
                                    value = 1. + .01 * (seed - 41) + .001 * (block - 14) + .0001 * template + horizon / 1000 + origin / 100000 + offset
                                elif metric == "peak_gpu_allocated_bytes":
                                    value = 1024 if device == "cuda" else None
                                elif metric.startswith("variogram_temporal_lag_"):
                                    if int(metric.rsplit("_", 1)[1]) >= horizon / 5:
                                        value = None
                                elif metric.startswith("refractory_count_violation_"):
                                    value = 0.
                                elif metric == "unit_total_coverage_90":
                                    value = .9
                                row[metric] = value
                            rows.append(row)
            write_rows(scores_path, rows)
            evaluation = dict(status="EVALUATED_DEVELOPMENT", model_type="population", training_seed=seed,
                episodes=16, origins=48, score_rows=144, evaluation_blocks=[14, 15, 16, 17],
                generated_unit_count_values=7680000000, checkpoint_sha256=checkpoint,
                dataset_sha256="d"*64, acquisition_design_sha256="e"*64,
                native_source_identity=dict(graph_manifest_sha256="f"*64, dt_ms=.1), configuration=config,
                hardware=dict(device=device, host="synthetic_test_fixture"))
            replay = dict(status="PASS", model_type="population", training_seed=seed,
                verified_origins=48, expected_origins=48, verified_score_rows=144,
                all_forecast_unit_metrics_replayed=True, all_origin_fields_and_latents_exact=True,
                sampled_unit_count_values_consumed=7680000000)
            if model != "original":
                identity = dict(seed=seed, mode=model, alpha=1., parent_checkpoint_sha256=checkpoint,
                                scope="synthetic_reader_fixture")
                evaluation.update(method=method, correction_identity=identity)
                replay.update(method=method, correction_identity=identity)
            write_json(evaluation_path, evaluation)
            write_json(replay_path, replay)
            delivery = json.loads((cell / "delivery.json").read_text())
            # These large arrays/checkpoints intentionally do not exist. A
            # successful reader must consume only the small declared receipts.
            delivery["files"].append(dict(path=str((scores_path.parent / "full_unit_samples.npz").relative_to(cell)),
                                          bytes=10**10, sha256="a"*64))
            write_json(cell / "delivery.json", delivery)
            refresh(cell, model)
    return corrections, parents


def rewrite_fit_manifest(cell, family):
    path = cell / "provenance/fit.json"
    write_json(path, family)
    delivery = json.loads((cell / "delivery.json").read_text())
    index = {row["path"]: row for row in delivery["files"]}
    index["provenance/fit.json"] = dict(path="provenance/fit.json", bytes=path.stat().st_size,
                                      sha256=sha256_file(path))
    delivery["files"] = list(index.values())
    delivery["provenance"]["fit_sha256"] = sha256_file(path)
    delivery["provenance"]["fit_schema"] = FAMILY_SCHEMA
    delivery["provenance"].pop("fit_job_id", None)
    write_json(cell / "delivery.json", delivery)


@pytest.fixture
def native_evidence(evidence, tmp_path):
    corrections, parents = evidence
    refits = tmp_path / "refits"
    for seed in (41, 42, 43):
        for mode in ("current", "history"):
            binding = dict(schema=FIT_SCHEMA, phase="refit", seed=seed, mode=mode,
                config=dict(training_seeds=[41, 42, 43], max_epochs=80, refit_blocks=list(range(14))),
                input_manifest_sha256="a"*64, config_sha256="b"*64,
                sources={"population_residual_memory.py": "c"*64},
                parent_checkpoint_sha256=f"{seed:064x}", selection_sha256="d"*64,
                validation_cases=[])
            write_json(refits / f"seed_{seed}/{mode}/fit.json", dict(schema=FIT_SCHEMA,
                status="FITTED_DEVELOPMENT", phase="refit", binding=binding,
                completed_epochs=3, selected_epoch=3, parent_unchanged=True,
                scientific_claim_qualified=False, predictor_sha256="e"*64,
                run_id="fixture_native_run", trial_id="fixture_native_trial"))
    family = build_refit_receipt(refits)
    for seed in (41, 42, 43):
        cell = corrections / f"seed_{seed}"
        rewrite_fit_manifest(cell, family)
        for mode in ("current", "history"):
            row = next(row for row in family["fits"] if row["seed"] == seed and row["mode"] == mode)
            source = refits / f"seed_{seed}/{mode}/fit.json"
            target = cell / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            delivery = json.loads((cell / "delivery.json").read_text())
            delivery["files"].append(dict(path=row["path"], bytes=target.stat().st_size, sha256=sha256_file(target)))
            write_json(cell / "delivery.json", delivery)
            identity = refit_identity(json.loads(source.read_text()), sha256_file(source))
            _, evaluation_path, replay_path = paths(cell, mode)
            for path in (evaluation_path, replay_path):
                value = json.loads(path.read_text())
                value["correction_identity"] = identity
                write_json(path, value)
            refresh(cell, mode)
    return corrections, parents, refits


def test_native_refits_preserve_complete_comparison_and_actual_fit_provenance(native_evidence, tmp_path):
    corrections, parents, _ = native_evidence
    report = analysis.analyze(corrections, parents, tmp_path / "native_analysis")
    assert report["total_score_rows"] == 1296
    assert report["correction_fit"]["schema"] == FAMILY_SCHEMA
    assert report["scope"]["scientific_claim_qualified"] is False
    for row in report["consumed"]:
        if row["model"] != "original":
            assert row["correction_fit"]["run_id"] == "fixture_native_run"
            assert row["correction_fit"]["trial_id"] == "fixture_native_trial"
            assert "job_id" not in row["correction_fit"]
            assert row["correction_fit"]["path"] == f"provenance/fits/seed_{row['seed']}/{row['model']}/fit.json"


def test_native_refits_reject_mixed_shared_receipts_even_after_rehash(native_evidence, tmp_path):
    corrections, parents, _ = native_evidence
    cell = corrections / "seed_42"
    family = json.loads((cell / "provenance/fit.json").read_text())
    family["different_campaign"] = True
    rewrite_fit_manifest(cell, family)
    with pytest.raises(ValueError, match="same fitted correction manifest"):
        analysis.analyze(corrections, parents, tmp_path / "mixed_native")


def test_native_refits_reject_incomplete_family_after_rehash(native_evidence, tmp_path):
    corrections, parents, _ = native_evidence
    family = json.loads((corrections / "seed_41/provenance/fit.json").read_text())
    family["fits"].pop()
    for seed in (41, 42, 43):
        rewrite_fit_manifest(corrections / f"seed_{seed}", family)
    with pytest.raises(ValueError, match="complete three-seed native refit family"):
        analysis.analyze(corrections, parents, tmp_path / "incomplete_native")


@pytest.mark.parametrize("key,value", [("status", "PARTIAL"), ("phase", "qualify"),
                                      ("parent_unchanged", False), ("selected_epoch", 4)])
def test_native_refit_builder_rejects_unfinished_or_inconsistent_fits(native_evidence, key, value):
    _, _, refits = native_evidence
    path = refits / "seed_43/history/fit.json"
    report = json.loads(path.read_text())
    report[key] = value
    write_json(path, report)
    with pytest.raises(ValueError, match="complete Merlin native observation refit"):
        build_refit_receipt(refits)


def test_native_evaluation_must_use_the_declared_refit(native_evidence, tmp_path):
    corrections, parents, _ = native_evidence
    cell = corrections / "seed_42"
    _, evaluation_path, replay_path = paths(cell, "history")
    for path in (evaluation_path, replay_path):
        value = json.loads(path.read_text())
        value["correction_identity"]["fit_sha256"] = "f"*64
        write_json(path, value)
    refresh(cell, "history")
    with pytest.raises(ValueError, match="evaluation correction differs"):
        analysis.analyze(corrections, parents, tmp_path / "wrong_native_fit")


def test_native_comparison_cli_records_cpu_execution(native_evidence, tmp_path, monkeypatch):
    corrections, parents, _ = native_evidence
    output = tmp_path / "native_cli"
    execution = dict(backend="merlin", run_id="fixture_analysis", trial_id="fixture_analysis_trial")
    monkeypatch.setattr(analysis, "cpu_execution_provenance", lambda: execution)
    monkeypatch.setattr(sys, "argv", ["native-comparison", "--corrections", str(corrections),
                                    "--parent-campaign", str(parents), "--output", str(output)])
    analysis.main()
    assert json.loads((output / "analysis.json").read_text())["execution"] == execution


def test_complete_reader_preserves_every_seed_metric_and_structural_na(evidence, tmp_path):
    corrections, parents = evidence
    output = tmp_path / "analysis"
    report = analysis.analyze(corrections, parents, output)
    assert report["total_score_rows"] == 1296
    assert report["large_arrays_or_checkpoints_read"] is False
    assert len(read_rows(output / "all_model_scores.csv")) == 1296
    means = read_rows(output / "per_seed_horizon_metrics.csv")
    assert len(means) == 486
    assert len(read_rows(output / "paired_source_block_differences.csv")) == 1944
    for row in report["primary_500ms"]:
        # Retain the worse current-only fixture result as well as the better
        # history result; completion is a data contract, not a gain threshold.
        assert row["current_minus_original"] == pytest.approx(.2)
        assert row["history_minus_original"] == pytest.approx(-.1)
        assert row["history_minus_current"] == pytest.approx(-.3)
    unavailable = [row for row in means if row["model"] == "history" and row["metric"] == "variogram_temporal_lag_16"
                   and float(row["horizon_ms"]) == 80.]
    assert len(unavailable) == 3
    assert all(row["mean"] == "" and row["available"] == "0" and row["structural_na"] == "48"
               and row["total"] == "48" for row in unavailable)
    assert {row["hardware"]["device"] for row in report["consumed"]} == {"cpu", "cuda"}
    assert report["scope"]["matched_hardware_speedup_qualified"] is False
    assert report["scope"]["significance_tests_computed"] is False
    assert report["primary_endpoint"]["posterior_proxy_used"] is False
    assert report["correction_fit"]["job_id"] == "fixture_fit"
    corrected = [row for row in report["consumed"] if row["model"] != "original"]
    assert len(corrected) == 6
    assert {row["correction_fit"]["sha256"] for row in corrected} == {report["correction_fit"]["sha256"]}


@pytest.mark.parametrize("failure", ["missing", "failed", "mode_failed"])
def test_incomplete_or_failed_seed_cannot_be_silently_excluded(evidence, tmp_path, failure):
    corrections, parents = evidence
    path = corrections / "seed_43/delivery.json"
    if failure == "missing":
        path.unlink()
    else:
        receipt = json.loads(path.read_text())
        if failure == "failed":
            receipt.update(status="FAILED", exit_code=2)
        else:
            receipt["modes"]["history"].update(status="FAILED", exit_code=2)
        write_json(path, receipt)
    with pytest.raises(ValueError, match="(unavailable|must be PASS)"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_changed_small_score_file_rejects_before_aggregation(evidence, tmp_path):
    corrections, parents = evidence
    path, _, _ = paths(corrections / "seed_42", "history")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="binding changed"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_duplicate_source_horizon_rejects_after_valid_rehash(evidence, tmp_path):
    corrections, parents = evidence
    cell = corrections / "seed_41"
    path, _, _ = paths(cell, "current")
    rows = read_rows(path)
    rows[1]["horizon_ms"] = rows[0]["horizon_ms"]
    write_rows(path, rows)
    refresh(cell, "current")
    with pytest.raises(ValueError, match="paired source/horizon"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("value", ["", "nan", "inf"])
def test_every_primary_energy_row_must_be_finite(evidence, tmp_path, value):
    corrections, parents = evidence
    cell = corrections / "seed_43"
    path, _, _ = paths(cell, "history")
    rows = read_rows(path)
    rows[-1]["energy_score"] = value
    write_rows(path, rows)
    refresh(cell, "history")
    with pytest.raises(ValueError, match="metric: energy_score"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_changed_parent_checkpoint_rejects_even_with_matching_small_receipts(evidence, tmp_path):
    corrections, parents = evidence
    cell = corrections / "seed_42"
    _, path, _ = paths(cell, "current")
    evaluation = json.loads(path.read_text())
    evaluation["checkpoint_sha256"] = "b"*64
    write_json(path, evaluation)
    refresh(cell, "current")
    with pytest.raises(ValueError, match="paired parent source"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_replay_pass_is_required_under_a_pass_delivery(evidence, tmp_path):
    corrections, parents = evidence
    cell = corrections / "seed_41"
    _, _, path = paths(cell, "history")
    replay = json.loads(path.read_text())
    replay["status"] = "MISMATCH"
    write_json(path, replay)
    refresh(cell, "history")
    with pytest.raises(ValueError, match="all-origin replay PASS"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_short_horizon_na_cannot_be_replaced_with_zero(evidence, tmp_path):
    corrections, parents = evidence
    cell = corrections / "seed_41"
    path, _, _ = paths(cell, "current")
    rows = read_rows(path)
    rows[0]["variogram_temporal_lag_16"] = "0"
    write_rows(path, rows)
    refresh(cell, "current")
    with pytest.raises(ValueError, match="structural N/A"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_fit_manifest_must_match_delivery_provenance(evidence, tmp_path):
    corrections, parents = evidence
    path = corrections / "seed_42/delivery.json"
    delivery = json.loads(path.read_text())
    delivery["provenance"]["fit_sha256"] = "0" * 64
    write_json(path, delivery)
    with pytest.raises(ValueError, match="fit receipt differs from delivery provenance"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_different_correction_fits_cannot_be_mixed_across_seeds(evidence, tmp_path):
    corrections, parents = evidence
    cell = corrections / "seed_43"
    fit_path = cell / "provenance/fit.json"
    fit = json.loads(fit_path.read_text())
    fit["config"]["alpha_by_mode"]["history"] = 2.
    write_json(fit_path, fit)
    digest = sha256_file(fit_path)
    delivery_path = cell / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    delivery["provenance"]["fit_sha256"] = digest
    for row in delivery["files"]:
        if row["path"] == "provenance/fit.json":
            row.update(bytes=fit_path.stat().st_size, sha256=digest)
    write_json(delivery_path, delivery)
    with pytest.raises(ValueError, match="same fitted correction manifest"):
        analysis.analyze(corrections, parents, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()



def refresh_eag(report_path, acceptance_path):
    report = json.loads(report_path.read_text())
    table = report_path.parent / "joint_comparison_all_seeds.csv"
    report["artifacts"][table.name] = sha256_file(table)
    write_json(report_path, report)
    acceptance = json.loads(acceptance_path.read_text())
    acceptance["comparisons"]["eag"].update(report_sha256=sha256_file(report_path), all_rows_sha256=sha256_file(table))
    write_json(acceptance_path, acceptance)


@pytest.fixture
def eag_evidence(evidence, tmp_path):
    corrections, parents = evidence
    report_path = tmp_path / "accepted_eag/comparison.json"
    acceptance_path = tmp_path / "acceptance.json"
    metrics = [name for name in METRICS if name not in analysis.COST_METRICS]
    rows, bindings, budgets = [], {}, {}
    for seed in (41, 42, 43):
        parent = parents / f"seed_{seed}"
        scores_path, evaluation_path, _ = paths(parent, "original")
        delivery = json.loads((parent / "delivery.json").read_text())
        checkpoint = next(row["sha256"] for row in delivery["files"] if row["path"] == "training/best.pt")
        bindings[str(seed)] = dict(
            population=dict(delivery_sha256=sha256_file(parent / "delivery.json"), consumed_files={
                "forecasts/evaluation.json": sha256_file(evaluation_path), "forecasts/scores.csv": sha256_file(scores_path),
                "training/best.pt": checkpoint}),
            eag=dict(delivery_sha256=f"{seed + 1000:064x}"))
        budgets[str(seed)] = dict(method="eag", autoencoder_completed_epochs=260, energy_completed_passes=4000,
            observation_law="source_softplus_poisson", native_refractory_qualified=False,
            source_denominators=dict(fit_complete_episodes=56, validation_complete_episodes=16,
                                    fit_origin_windows=168, validation_origin_windows=48, future_bins=100, units=100000))
        for original in read_rows(scores_path):
            row = {key: original[key] for key in ("seed", "block", "case_id", "origin_ms", "horizon_ms")}
            row["candidate_method"] = "population_recurrent_native_observation"
            for metric in metrics:
                population = None if original[metric] == "" else float(original[metric])
                baseline = population
                if population is not None and metric == "energy_score":
                    baseline = population - .15
                elif population is not None and metric == "observed_spikes_per_unit":
                    baseline = population + 1e-8
                row["eag_" + metric] = baseline
                row["population_" + metric] = population
                row["population_minus_eag_" + metric] = population - baseline if population is not None else None
            rows.append(row)
    table = report_path.parent / "joint_comparison_all_seeds.csv"
    write_rows(table, rows)
    report = dict(status="COMPLETE_DESCRIPTIVE_DEVELOPMENT_COMPARISON", matched_rows=432, block_rows=36,
        training_seeds=[41, 42, 43], independent_blocks_per_seed=4, connectivity_instances=1, bindings=bindings,
        same_training_budget=False, same_hardware=False, full_end_to_end_cost_matched=False, speedup_qualified=False,
        inferential_statistics_qualified=False, scientific_claim_qualified=False, baseline_method="eag",
        baseline_training_bindings=budgets, baseline_observation_law="source_softplus_poisson",
        baseline_native_refractory_qualified=False, candidate_method="population_recurrent_native_observation",
        artifacts={table.name: sha256_file(table), "joint_comparison_by_block.csv": "b"*64})
    write_json(report_path, report)
    # Historical storage paths and unused block tables need not exist locally.
    acceptance = dict(status="ACCEPTED_COMPLETE_NATIVE_OBSERVATION_THREE_SEED_COMPARISONS", comparisons=dict(eag=dict(
        report="/historical/accepted/comparison.json", report_sha256=sha256_file(report_path),
        all_rows_sha256=sha256_file(table), block_rows_sha256="b"*64, matched_rows=432, seed_block_horizon_rows=36)))
    write_json(acceptance_path, acceptance)
    return corrections, parents, report_path, acceptance_path


def test_eag_reuse_retains_worse_results_and_observed_precision_differences(eag_evidence, tmp_path):
    corrections, parents, comparison, acceptance = eag_evidence
    report = analysis.analyze(corrections, parents, tmp_path / "with_eag", eag_comparison=comparison, eag_acceptance=acceptance)
    assert report["models"] == ["original", "current", "history"]
    assert report["primary_endpoint"]["models"] == ["original", "current", "history"]
    assert report["artifacts"]["all_model_scores.csv"]["rows"] == 1296
    assert report["artifacts"]["correction_vs_eag_all_seeds.csv"]["rows"] == 864
    assert report["artifacts"]["correction_vs_eag_by_block.csv"]["rows"] == 1008
    for row in report["primary_500ms"]:
        assert row["current_minus_eag"] == pytest.approx(.35)
        assert row["history_minus_eag"] == pytest.approx(.05)
    paired = read_rows(tmp_path / "with_eag/correction_vs_eag_all_seeds.csv")
    assert float(paired[0]["candidate_minus_eag_observed_spikes_per_unit"]) == pytest.approx(-1e-8, abs=1e-12)
    assert paired[0]["eag_variogram_temporal_lag_16"] == ""
    baseline = report["eag_reference"]
    assert baseline["accepted_report_path"] != baseline["comparison_path"]
    assert not baseline["training_repeated"] and not baseline["generation_repeated"] and not baseline["replay_repeated"]
    assert baseline["limitations"]["native_refractory_qualified"] is False
    assert baseline["limitations"]["same_training_budget"] is False
    assert baseline["limitations"]["same_hardware"] is False
    assert baseline["limitations"]["sota_qualified"] is False


@pytest.mark.parametrize("argument", ["eag_comparison", "eag_acceptance"])
def test_eag_options_require_a_pair_before_reading_inputs(tmp_path, argument):
    with pytest.raises(ValueError, match="supplied together"):
        analysis.analyze(tmp_path, tmp_path, tmp_path / "rejected", **{argument: tmp_path / "missing.json"})
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("artifact", ["comparison", "csv"])
def test_eag_accepted_hashes_reject_changed_content(eag_evidence, tmp_path, artifact):
    corrections, parents, comparison, acceptance = eag_evidence
    path = comparison if artifact == "comparison" else comparison.parent / "joint_comparison_all_seeds.csv"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash differs"):
        analysis.analyze(corrections, parents, tmp_path / "rejected", eag_comparison=comparison, eag_acceptance=acceptance)
    assert not (tmp_path / "rejected").exists()


def test_eag_common_parent_delivery_binding_is_required(eag_evidence, tmp_path):
    corrections, parents, comparison, acceptance = eag_evidence
    report = json.loads(comparison.read_text())
    report["bindings"]["43"]["population"]["delivery_sha256"] = "0"*64
    write_json(comparison, report)
    refresh_eag(comparison, acceptance)
    with pytest.raises(ValueError, match="original-parent binding differs"):
        analysis.analyze(corrections, parents, tmp_path / "rejected", eag_comparison=comparison, eag_acceptance=acceptance)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_eag_requires_all_exact_paired_source_horizons(eag_evidence, tmp_path, change):
    corrections, parents, comparison, acceptance = eag_evidence
    path = comparison.parent / "joint_comparison_all_seeds.csv"
    rows = read_rows(path)
    if change == "duplicate":
        rows[-1] = dict(rows[0])
    else:
        rows.pop()
    write_rows(path, rows)
    refresh_eag(comparison, acceptance)
    with pytest.raises(ValueError, match="(432-row|source/horizon pairing)"):
        analysis.analyze(corrections, parents, tmp_path / "rejected", eag_comparison=comparison, eag_acceptance=acceptance)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["nonfinite", "structural_zero"])
def test_eag_metrics_require_finite_or_declared_structural_values(eag_evidence, tmp_path, change):
    corrections, parents, comparison, acceptance = eag_evidence
    path = comparison.parent / "joint_comparison_all_seeds.csv"
    rows = read_rows(path)
    if change == "nonfinite":
        rows[-1]["eag_energy_score"] = "nan"
    else:
        rows[0]["eag_variogram_temporal_lag_16"] = "0"
    write_rows(path, rows)
    refresh_eag(comparison, acceptance)
    with pytest.raises(ValueError, match="(nonfinite metric|structural N/A)"):
        analysis.analyze(corrections, parents, tmp_path / "rejected", eag_comparison=comparison, eag_acceptance=acceptance)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["native_qualification", "source_units"])
def test_eag_observation_law_and_source_protocol_remain_explicit(eag_evidence, tmp_path, change):
    corrections, parents, comparison, acceptance = eag_evidence
    report = json.loads(comparison.read_text())
    if change == "native_qualification":
        report["baseline_native_refractory_qualified"] = True
    else:
        report["baseline_training_bindings"]["42"]["source_denominators"]["units"] = 99999
    write_json(comparison, report)
    refresh_eag(comparison, acceptance)
    with pytest.raises(ValueError, match="(fairness limits|source counts)"):
        analysis.analyze(corrections, parents, tmp_path / "rejected", eag_comparison=comparison, eag_acceptance=acceptance)
    assert not (tmp_path / "rejected").exists()



def refresh_ldns(ldns_root):
    report_path = ldns_root / "analysis/comparison.json"
    table_path = report_path.parent / "joint_comparison_all_seeds.csv"
    report = json.loads(report_path.read_text())
    report["artifacts"][table_path.name] = sha256_file(table_path)
    write_json(report_path, report)
    intake_path = ldns_root / "delivery.json"
    intake = json.loads(intake_path.read_text())
    intake.update(source_delivery_sha256=sha256_file(ldns_root / "source_delivery.json"),
                  source_comparison_sha256=sha256_file(report_path))
    write_json(intake_path, intake)


def refresh_ldns_field(field_root):
    intake_path = field_root / "intake.json"
    intake = json.loads(intake_path.read_text())
    intake["comparison_sha256"] = sha256_file(field_root / "comparison.json")
    write_json(intake_path, intake)


@pytest.fixture
def ldns_evidence(native_evidence, tmp_path):
    corrections, parents, _ = native_evidence
    ldns_root, field_root = tmp_path / "accepted_ldns", tmp_path / "accepted_field"
    cells, source, common_fit = analysis.admit_comparison_cells(corrections, parents)
    rows, bindings, field_bindings = [], {}, {}
    metrics = [name for name in METRICS if name not in analysis.COST_METRICS]
    for seed in (41, 42, 43):
        old_population = dict(delivery_sha256=f"{seed + 1000:064x}", consumed_files={
            "training/run.json": f"{seed + 2000:064x}", "forecasts/fitting_field_scale.npz": "a" * 64})
        bindings[str(seed)] = dict(population=old_population, ldns=dict(delivery_sha256=f"{seed + 3000:064x}"))
        field_bindings[str(seed)] = dict(baseline_population=old_population,
            parent=dict(delivery_sha256=cells[(seed, "original")]["consumed"]["delivery_sha256"], consumed_files={}),
            corrections=dict(delivery_sha256=cells[(seed, "current")]["consumed"]["delivery_sha256"], consumed_files={}))
        assert old_population["delivery_sha256"] != cells[(seed, "original")]["consumed"]["delivery_sha256"]
        for original in cells[(seed, "original")]["rows"].values():
            row = {name: original[name] for name in ("seed", "block", "case_id", "origin_ms", "horizon_ms")}
            row["candidate_method"] = "population_recurrent_predictive_mixture"
            for metric in metrics:
                value = original[metric]
                if value is not None and metric == "energy_score":
                    value -= .15
                elif value is not None and metric == "observed_spikes_per_unit":
                    value += 1e-8
                row["ldns_" + metric] = value
                # Old candidate predictions and differences are deliberately
                # unrelated to the native candidates and must never be reused.
                row["population_" + metric] = 1000. if value is not None else None
                row["population_minus_ldns_" + metric] = 2000. if value is not None else None
            rows.append(row)
    table = ldns_root / "analysis/joint_comparison_all_seeds.csv"
    write_rows(table, rows)
    observed = dict(reference="exact_common_observed_field_count_total_per_100000_units", rtol=5e-6, atol=1e-7,
        tolerance_source="frozen_population_forecast_replay", observed_field_counts_match_exactly=True,
        saved_scalar_differences_retained=True)
    report = dict(status="COMPLETE_DESCRIPTIVE_DEVELOPMENT_COMPARISON", matched_rows=432, block_rows=36,
        training_seeds=[41, 42, 43], independent_blocks_per_seed=4, connectivity_instances=1,
        candidate_method="population_recurrent_predictive_mixture", bindings=bindings,
        same_training_budget=False, same_hardware=False, full_end_to_end_cost_matched=False, speedup_qualified=False,
        inferential_statistics_qualified=False, scientific_claim_qualified=False, observed_total_validation=observed,
        artifacts={table.name: sha256_file(table)})
    write_json(ldns_root / "analysis/comparison.json", report)
    write_json(ldns_root / "source_delivery.json", dict(status="FAILED", exit_code=2, error="'baseline_method'",
        scientific_claim_qualified=False))
    write_json(ldns_root / "source_command.json", ["/historical/python", "-B", "-m",
        "population_dynamics.latent_spike_diffusion_comparison", "--ldns-root", "/historical/ldns",
        "--population-root", "/historical/predictive_mixture", "--output", "/historical/comparison"])
    write_json(ldns_root / "delivery.json", dict(status="PASS", exit_code=0, baseline_method="ldns",
        candidate_method="population_recurrent_predictive_mixture", matched_rows=432, block_rows=36,
        training_seeds=[41, 42, 43], source_artifacts_verified=6, source_seed_deliveries_verified=6,
        source_failed_wrapper_preserved=True, implicit_ldns_schema_preserved=True, training_repeated=False,
        forecast_generation_repeated=False, comparison_repeated=False, scientific_claim_qualified=False))
    refresh_ldns(ldns_root)
    field_artifacts = {"field_comparison_all_seeds.csv": "b" * 64, "field_comparison_by_block.csv": "c" * 64}
    field_report = dict(status="COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_FIELD_COMPARISON", training_seeds=[41, 42, 43],
        score_rows=864, block_rows=72, source_circuits=1, independent_selection_blocks=4,
        source=source, correction_fit=common_fit, bindings=field_bindings, artifacts=field_artifacts,
        execution=dict(backend="merlin", run_id="fixture_field_run", trial_id="fixture_field_trial"))
    field_report.update({name: False for name in ("old_population_predictions_reused", "old_population_costs_reused",
        "training_repeated", "inference_performed", "unit_samples_regenerated", "scientific_claim_qualified",
        "sota_qualified", "inferential_statistics_performed", "full_end_to_end_cost_matched", "speedup_qualified")})
    write_json(field_root / "comparison.json", field_report)
    write_json(field_root / "intake.json", dict(status=field_report["status"], run_id="fixture_field_run",
        trial_id="fixture_field_trial", original_archives_verified=3, field_array_files_verified=294,
        training_repeated=False, generation_repeated=False, scientific_claim_qualified=False, sota_qualified=False,
        artifacts={name: dict(sha256=digest, rows=864 if "all_seeds" in name else 72)
                   for name, digest in field_artifacts.items()}))
    refresh_ldns_field(field_root)
    return corrections, parents, ldns_root, field_root


def run_ldns(evidence, output, **kwargs):
    corrections, parents, ldns_root, field_root = evidence
    return analysis.analyze_ldns(corrections, parents, output, ldns_comparison=ldns_root,
                                field_comparison=field_root, **kwargs)


def test_ldns_reuses_complete_scores_and_source_link_without_original_or_eag_analysis(ldns_evidence, tmp_path, monkeypatch):
    _, _, ldns_root, _ = ldns_evidence
    preserved = {name: (ldns_root / name).read_bytes() for name in
                 ("delivery.json", "source_delivery.json", "source_command.json", "analysis/comparison.json")}
    def forbidden(*args, **kwargs):
        raise AssertionError("completed original/EAG analysis must not be repeated")
    monkeypatch.setattr(analysis, "summarize", forbidden)
    monkeypatch.setattr(analysis, "_accepted_eag", forbidden)
    output = tmp_path / "ldns_only"
    report = run_ldns(ldns_evidence, output)
    assert report["status"] == "COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_LDNS_COMPARISON"
    assert report["models"] == ["current", "history"]
    assert report["score_rows"] == 864 and report["block_metric_rows"] == 1008
    assert report["source_block_horizon_rows"] == 72 and len(report["fidelity_metrics"]) == 14
    assert set(report["artifacts"]) == {
        "correction_vs_ldns_all_seeds.csv", "correction_vs_ldns_by_block.csv", "primary_500ms.csv"}
    assert not (output / "all_model_scores.csv").exists()
    paired = read_rows(output / "correction_vs_ldns_all_seeds.csv")
    blocks = read_rows(output / "correction_vs_ldns_by_block.csv")
    assert len(paired) == 864 and len(blocks) == 1008
    assert {row["seed"] for row in paired} == {"41", "42", "43"}
    assert {row["candidate"] for row in paired} == {"current", "history"}
    assert {row["reference"] for row in paired} == {"ldns"}
    assert float(paired[0]["candidate_minus_ldns_observed_spikes_per_unit"]) == pytest.approx(-1e-8, abs=1e-12)
    assert paired[0]["ldns_variogram_temporal_lag_16"] == ""
    for row in report["primary_500ms"]:
        assert row["current_minus_ldns"] == pytest.approx(.35)
        assert row["history_minus_ldns"] == pytest.approx(.05)
        assert "original_mean" not in row and "eag_mean" not in row
    unavailable = [row for row in blocks if row["metric"] == "variogram_temporal_lag_16"
                   and float(row["horizon_ms"]) == 80.]
    assert len(unavailable) == 24
    assert all(row["paired_available"] == "0" and row["structural_unavailable_pairs"] == "12"
               and row["candidate_minus_ldns"] == "" for row in unavailable)
    reference = report["ldns_reference"]
    assert reference["source_delivery"]["status"] == "FAILED" and reference["source_delivery"]["exit_code"] == 2
    assert reference["limitations"]["same_training_budget"] is False
    assert reference["limitations"]["same_hardware"] is False
    assert reference["limitations"]["sota_qualified"] is False
    assert report["scope"]["scientific_claim_qualified"] is False
    assert report["scope"]["significance_tests_computed"] is False
    assert not report["large_arrays_or_checkpoints_read"]
    assert not report["training_repeated"] and not report["generation_repeated"]
    assert not report["old_population_predictions_reused"] and not report["old_population_costs_reused"]
    assert {name: (ldns_root / name).read_bytes() for name in preserved} == preserved
    assert "files" not in json.loads(preserved["delivery.json"])
    assert "baseline_method" not in json.loads(preserved["analysis/comparison.json"])


@pytest.mark.parametrize("artifact", ["report", "scores", "source_receipt", "field_report"])
def test_ldns_rejects_changed_accepted_artifacts(ldns_evidence, tmp_path, artifact):
    _, _, ldns_root, field_root = ldns_evidence
    path = {"report": ldns_root / "analysis/comparison.json",
            "scores": ldns_root / "analysis/joint_comparison_all_seeds.csv",
            "source_receipt": ldns_root / "source_delivery.json", "field_report": field_root / "comparison.json"}[artifact]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash differs"):
        run_ldns(ldns_evidence, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("binding", ["source", "fit", "native_parent", "native_correction", "old_delivery", "old_run", "old_scale"])
def test_ldns_requires_the_accepted_native_and_old_source_bindings(ldns_evidence, tmp_path, binding):
    _, _, ldns_root, field_root = ldns_evidence
    field_path, ldns_path = field_root / "comparison.json", ldns_root / "analysis/comparison.json"
    field, ldns = json.loads(field_path.read_text()), json.loads(ldns_path.read_text())
    if binding == "source":
        field["source"]["dataset_sha256"] = "0" * 64
    elif binding == "fit":
        field["correction_fit"]["sha256"] = "0" * 64
    elif binding in ("native_parent", "native_correction"):
        method = "parent" if binding == "native_parent" else "corrections"
        field["bindings"]["43"][method]["delivery_sha256"] = "0" * 64
    elif binding == "old_delivery":
        ldns["bindings"]["43"]["population"]["delivery_sha256"] = "0" * 64
    else:
        name = "training/run.json" if binding == "old_run" else "forecasts/fitting_field_scale.npz"
        ldns["bindings"]["43"]["population"]["consumed_files"][name] = "0" * 64
    write_json(field_path, field)
    write_json(ldns_path, ldns)
    refresh_ldns(ldns_root)
    refresh_ldns_field(field_root)
    with pytest.raises(ValueError, match="(source or correction fit differs|native parent or correction delivery differs|old-population)"):
        run_ldns(ldns_evidence, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["duplicate", "missing", "identity"])
def test_ldns_requires_every_exact_source_horizon(ldns_evidence, tmp_path, change):
    ldns_root = ldns_evidence[2]
    path = ldns_root / "analysis/joint_comparison_all_seeds.csv"
    rows = read_rows(path)
    if change == "duplicate":
        rows[-1] = dict(rows[0])
    elif change == "missing":
        rows.pop()
    else:
        rows[-1]["block"] = "99"
    write_rows(path, rows)
    refresh_ldns(ldns_root)
    with pytest.raises(ValueError, match="(432-row|source/horizon pairing|source identity)"):
        run_ldns(ldns_evidence, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["nonfinite", "missing", "structural_zero", "observed_total"])
def test_ldns_rejects_invalid_metrics_without_coercion(ldns_evidence, tmp_path, change):
    ldns_root = ldns_evidence[2]
    path = ldns_root / "analysis/joint_comparison_all_seeds.csv"
    rows = read_rows(path)
    if change == "structural_zero":
        rows[0]["ldns_variogram_temporal_lag_16"] = "0"
    elif change == "observed_total":
        rows[-1]["ldns_observed_spikes_per_unit"] = "1"
    else:
        rows[-1]["ldns_energy_score"] = "nan" if change == "nonfinite" else ""
    write_rows(path, rows)
    refresh_ldns(ldns_root)
    with pytest.raises(ValueError, match="(metric|structural N/A|observed total differs)"):
        run_ldns(ldns_evidence, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("change", ["incomplete_intake", "promoted_wrapper", "changed_tolerance", "incomplete_field"])
def test_ldns_recovery_and_field_acceptance_semantics_are_preserved(ldns_evidence, tmp_path, change):
    _, _, ldns_root, field_root = ldns_evidence
    if change == "incomplete_intake":
        path = ldns_root / "delivery.json"
        value = json.loads(path.read_text())
        value["source_seed_deliveries_verified"] = 5
    elif change == "promoted_wrapper":
        path = ldns_root / "source_delivery.json"
        value = json.loads(path.read_text())
        value.update(status="PASS", exit_code=0)
    elif change == "changed_tolerance":
        path = ldns_root / "analysis/comparison.json"
        value = json.loads(path.read_text())
        value["observed_total_validation"]["rtol"] = .1
    else:
        path = field_root / "intake.json"
        value = json.loads(path.read_text())
        value["status"] = "PARTIAL"
    write_json(path, value)
    refresh_ldns(ldns_root)
    with pytest.raises(ValueError, match="(recovered LDNS intake|failed LDNS wrapper|frozen tolerance|accepted native field comparison)"):
        run_ldns(ldns_evidence, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_ldns_cli_records_cpu_execution(ldns_evidence, tmp_path, monkeypatch):
    corrections, parents, ldns_root, field_root = ldns_evidence
    output = tmp_path / "ldns_cli"
    execution = dict(backend="merlin", run_id="fixture_ldns_run", trial_id="fixture_ldns_trial")
    monkeypatch.setattr(analysis, "cpu_execution_provenance", lambda: execution)
    monkeypatch.setattr(sys, "argv", ["native-comparison", "--corrections", str(corrections),
        "--parent-campaign", str(parents), "--ldns-comparison", str(ldns_root),
        "--field-comparison", str(field_root), "--output", str(output)])
    analysis.main()
    report = json.loads((output / "analysis.json").read_text())
    assert report["execution"] == execution and report["score_rows"] == 864


@pytest.mark.parametrize("arguments", [
    ["--ldns-comparison", "ldns"], ["--field-comparison", "field"],
    ["--ldns-comparison", "ldns", "--field-comparison", "field", "--eag-comparison", "eag", "--eag-acceptance", "acceptance"]])
def test_ldns_cli_requires_paired_exclusive_inputs_before_reading(tmp_path, monkeypatch, arguments):
    output = tmp_path / "rejected"
    monkeypatch.setattr(sys, "argv", ["native-comparison", "--corrections", str(tmp_path),
        "--parent-campaign", str(tmp_path), "--output", str(output), *arguments])
    with pytest.raises(SystemExit) as error:
        analysis.main()
    assert error.value.code == 2
    assert not output.exists()


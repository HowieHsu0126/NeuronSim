"""Descriptive native correction comparisons from complete, replayed deliveries.

Only delivery metadata and score/evaluation/replay files are read. Large count
arrays and parent checkpoints remain covered by their accepted replay receipts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .population_forecast_analysis import COST_METRICS, _delivery_file, _mean, _metric_names
from .provenance import cpu_execution_provenance, freeze_json, sha256_file
from .population_residual_refit_provenance import FAMILY_SCHEMA, SHARED_BINDING_KEYS, refit_identity
from .trajectory_forecast_analysis import _write_rows

SEEDS = (41, 42, 43)
MODELS = ("original", "current", "history")
BLOCKS = (14, 15, 16, 17)
ORIGINS_MS = (1000., 3000., 5000.)
HORIZONS_MS = (20., 80., 500.)
BIN_MS = 5.
METHODS = {"original": "population", "current": "population_residual_current",
           "history": "population_residual_history"}
COMPARISONS = (("current", "original"), ("history", "original"), ("history", "current"))
IDENTITY_COLUMNS = {"seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms", "method", "ensemble_artifact"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _delivery(cell, seed, *, corrected):
    path = cell / "delivery.json"
    _require(path.is_file(), f"complete seed {seed} delivery is unavailable: {cell}")
    delivery = json.loads(path.read_text())
    _require(delivery.get("status") == "PASS" and delivery.get("exit_code") == 0
             and delivery.get("training_seed") == seed, f"seed {seed} delivery must be PASS")
    if corrected:
        _require(set(delivery.get("modes", {})) == {"current", "history"}
                 and all(row.get("status") == "PASS" and row.get("exit_code") == 0
                         for row in delivery["modes"].values()), "both correction modes must be PASS")
    rows = [dict(row, bytes=row.get("bytes", row.get("destination_bytes"))) for row in delivery["files"]]
    index = {row["path"]: row for row in rows}
    _require(len(index) == len(rows), "duplicate delivery manifest path")
    return delivery, index, sha256_file(path)


def _structural_na(metric, horizon, device):
    if metric.startswith("variogram_temporal_lag_"):
        lag = int(metric.rsplit("_", 1)[1])
        if lag >= horizon / BIN_MS:
            return "temporal_lag_not_within_horizon"
    if metric == "peak_gpu_allocated_bytes" and device == "cpu":
        return "cpu_evaluation_has_no_gpu_allocation_metric"
    return ""


def _metric_value(value, metric, horizon, device):
    if _structural_na(metric, horizon, device):
        _require(value == "", "declared structural N/A must remain empty: " + metric)
        return None
    _require(value not in (None, ""), "nonstructural missing metric: " + metric)
    value = float(value)
    _require(math.isfinite(value), "nonfinite metric: " + metric)
    return value


def _score_rows(path, seed, model, config, device):
    field_metrics, remaining_metrics = _metric_names(config)
    metrics = field_metrics + remaining_metrics
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames)
                 and set(reader.fieldnames) == IDENTITY_COLUMNS | set(metrics), "complete unique score columns required")
        source_rows = list(reader)
    expected = {(seed, f"train_block_{block:03d}_familiar_{template}", origin, horizon)
                for block in BLOCKS for template in range(4) for origin in ORIGINS_MS for horizon in HORIZONS_MS}
    _require(len(source_rows) == 144, "complete 144-row score table required")
    rows = {}
    for raw in source_rows:
        _require(None not in raw, "unexpected score CSV fields")
        block, origin, horizon = int(raw["block"]), float(raw["origin_ms"]), float(raw["horizon_ms"])
        key = (int(raw["seed"]), raw["case_id"], origin, horizon)
        _require(key in expected and key not in rows, "duplicate or unexpected paired source/horizon row")
        _require(block in BLOCKS and raw["template_id"] in {f"familiar_{i}" for i in range(4)}
                 and raw["case_id"] == f"train_block_{block:03d}_{raw['template_id']}"
                 and raw["method"] == METHODS[model]
                 and raw["ensemble_artifact"] == f"{raw['case_id']}_origin_{origin:g}.npz", "paired score identity differs")
        row = dict(model=model, seed=seed, block=block, case_id=raw["case_id"], template_id=raw["template_id"],
                   origin_ms=origin, horizon_ms=horizon, method=METHODS[model],
                   ensemble_artifact=raw["ensemble_artifact"], hardware_device=device)
        for metric in metrics:
            row[metric] = _metric_value(raw[metric], metric, horizon, device)
        rows[key] = row
    _require(set(rows) == expected, "complete identical paired source/horizon coverage required")
    return rows, metrics


def _fit_context(cell, delivery, manifest, seed):
    relative = "provenance/fit.json"
    path = _delivery_file(cell, manifest, relative, f"correction fit seed {seed}")
    fit = json.loads(path.read_text())
    digest = manifest[relative]["sha256"]
    provenance = delivery["provenance"]
    _require(fit.get("parent_training") is False and isinstance(fit.get("config"), dict)
             and provenance.get("fit_sha256") == digest,
             "correction fit receipt differs from delivery provenance")
    if fit.get("schema") != FAMILY_SCHEMA:
        _require(fit.get("status") == "FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS"
                 and str(fit.get("job_id")) == str(provenance.get("fit_job_id")),
                 "correction fit receipt differs from delivery provenance")
        consumed = dict(path=relative, sha256=digest, job_id=str(fit["job_id"]))
        return dict(common=dict(sha256=digest, job_id=str(fit["job_id"]), configuration=fit["config"]),
                    consumed={mode: consumed for mode in ("current", "history")}, native_identities=None)
    rows = fit.get("fits", [])
    expected = {(registered_seed, mode) for registered_seed in SEEDS for mode in ("current", "history")}
    _require(fit.get("status") == "COMPLETE_NATIVE_RESIDUAL_REFITS" and fit.get("phase") == "refit"
             and fit.get("training_seeds") == list(SEEDS) and fit.get("modes") == ["current", "history"]
             and len(rows) == 6 and {(row.get("seed"), row.get("mode")) for row in rows} == expected
             and provenance.get("fit_schema") == FAMILY_SCHEMA,
             "complete three-seed native refit family required")
    index = {(row["seed"], row["mode"]): row for row in rows}
    identities, consumed = {}, {}
    for mode in ("current", "history"):
        row = index[(seed, mode)]
        relative_fit = f"provenance/fits/seed_{seed}/{mode}/fit.json"
        _require(row.get("path") == relative_fit, "native refit delivery path differs")
        report_path = _delivery_file(cell, manifest, relative_fit, f"native refit seed {seed} {mode}")
        _require(row.get("sha256") == manifest[relative_fit]["sha256"], "native refit report digest differs")
        report = json.loads(report_path.read_text())
        identity = refit_identity(report, row["sha256"])
        binding = identity["binding"]
        _require(binding["seed"] == seed and binding["mode"] == mode
                 and binding["config"] == fit["config"]
                 and {key: binding[key] for key in SHARED_BINDING_KEYS} == fit.get("shared_binding")
                 and row.get("predictor_sha256") == identity["predictor_sha256"]
                 and row.get("parent_checkpoint_sha256") == binding["parent_checkpoint_sha256"]
                 and row.get("run_id") == report["run_id"] and row.get("trial_id") == report["trial_id"],
                 "native refit differs from the shared fitted campaign")
        identities[mode] = identity
        consumed[mode] = dict(path=relative_fit, sha256=row["sha256"], run_id=report["run_id"],
                              trial_id=report["trial_id"], family_path=relative, family_sha256=digest)
    return dict(common=dict(schema=FAMILY_SCHEMA, sha256=digest, configuration=fit["config"],
                            shared_binding=fit["shared_binding"]),
                consumed=consumed, native_identities=identities)


def _admit(cell, delivery, manifest, delivery_digest, seed, model, parent=None, fit_context=None):
    prefix = "forecasts/" if model == "original" else f"results/{model}/forecasts/"
    names = dict(evaluation=prefix + "evaluation.json", scores=prefix + "scores.csv",
                 replay="replay.json" if model == "original" else f"results/{model}/replay.json")
    paths = {role: _delivery_file(cell, manifest, name, f"{model} seed {seed}") for role, name in names.items()}
    digests = {role: manifest[name]["sha256"] for role, name in names.items()}
    evaluation = json.loads(paths["evaluation"].read_text())
    replay = json.loads(paths["replay"].read_text())
    _require(evaluation.get("status") == "EVALUATED_DEVELOPMENT" and evaluation.get("model_type") == "population"
             and evaluation.get("training_seed") == seed and evaluation.get("episodes") == 16
             and evaluation.get("origins") == 48 and evaluation.get("score_rows") == 144
             and evaluation.get("evaluation_blocks") == list(BLOCKS)
             and evaluation.get("generated_unit_count_values") == 7680000000
             and evaluation.get("scores_sha256") == digests["scores"], "complete native evaluation receipt required")
    _require(replay.get("status") == "PASS" and replay.get("model_type") == "population"
             and replay.get("training_seed") == seed and replay.get("verified_origins") == 48
             and replay.get("expected_origins") == 48 and replay.get("verified_score_rows") == 144
             and replay.get("all_forecast_unit_metrics_replayed") is True
             and replay.get("all_origin_fields_and_latents_exact") is True
             and replay.get("sampled_unit_count_values_consumed") == 7680000000
             and replay.get("evaluation_sha256") == digests["evaluation"]
             and replay.get("scores_sha256") == digests["scores"], "complete independent all-origin replay PASS required")
    for evaluation_key, replay_key in (("checkpoint_sha256", "source_checkpoint_sha256"),
                                      ("dataset_sha256", "dataset_sha256"),
                                      ("acquisition_design_sha256", "acquisition_design_sha256")):
        _require(evaluation[evaluation_key] == replay[replay_key], "evaluation/replay source identity differs")
    config = evaluation["configuration"]
    _require(tuple(config.get("origins_ms", ())) == ORIGINS_MS and tuple(config.get("horizons_ms", ())) == HORIZONS_MS
             and config.get("joint_samples") == 16 and config.get("formal_evaluation") is False
             and config.get("finite_ensemble_corrections") is True
             and config.get("temporal_lags_bins") == [1, 4, 16]
             and config.get("refractory_support_windows_bins") == [1, 2, 4], "registered complete native forecast protocol required")
    if model == "original":
        _require("correction_identity" not in evaluation and "correction_identity" not in replay
                 and evaluation.get("method", "population") == "population"
                 and replay.get("method", "population") == "population", "uncorrected parent receipt required")
        _require("training/best.pt" in manifest
                 and manifest["training/best.pt"]["sha256"] == evaluation["checkpoint_sha256"], "parent checkpoint manifest differs")
    else:
        _require(parent is not None, "accepted original parent required")
        for name in ("checkpoint_sha256", "dataset_sha256", "acquisition_design_sha256", "native_source_identity", "configuration"):
            _require(evaluation[name] == parent["evaluation"][name], "paired parent source or forecast configuration differs: " + name)
        identity = evaluation.get("correction_identity")
        native = fit_context is not None and fit_context["native_identities"] is not None
        coordinates = identity.get("binding", {}) if native and isinstance(identity, dict) else identity
        _require(isinstance(identity, dict) and identity == replay.get("correction_identity")
                 and isinstance(coordinates, dict) and coordinates.get("seed") == seed and coordinates.get("mode") == model
                 and coordinates.get("parent_checkpoint_sha256") == evaluation["checkpoint_sha256"]
                 and evaluation.get("method") == METHODS[model] and replay.get("method") == METHODS[model], "correction identity or method differs")
        if native:
            _require(identity == fit_context["native_identities"][model],
                     "evaluation correction differs from its completed native refit")
        mode_receipt = delivery["modes"][model]
        _require(mode_receipt.get("evaluation_sha256") == digests["evaluation"]
                 and mode_receipt.get("replay_sha256") == digests["replay"]
                 and delivery["provenance"]["parent_checkpoint_sha256"] == evaluation["checkpoint_sha256"], "correction mode delivery binding differs")
    hardware = evaluation["hardware"]
    device = hardware.get("device")
    _require(isinstance(device, str) and (device == "cpu" or device.startswith("cuda")), "recorded CPU/CUDA evaluation hardware required")
    rows, metrics = _score_rows(paths["scores"], seed, model, config, device)
    if parent is not None:
        _require(set(rows) == set(parent["rows"]), "paired source/horizon coverage differs from original")
    consumed = dict(seed=seed, model=model, root=str(cell), delivery_sha256=delivery_digest,
                    artifacts={role: dict(path=names[role], sha256=digests[role]) for role in names},
                    parent_checkpoint_sha256=evaluation["checkpoint_sha256"], hardware=hardware)
    if model != "original":
        consumed["correction_identity"] = evaluation["correction_identity"]
        consumed["correction_fit"] = fit_context["consumed"][model]
    return dict(rows=rows, metrics=metrics, evaluation=evaluation, consumed=consumed, device=device)


def _metric_summary(rows, metric):
    values = [row[metric] for row in rows]
    available = sum(value is not None for value in values)
    return dict(mean=_mean(values), available=available, structural_na=len(values) - available, total=len(values))


def summarize(cells):
    means, differences, primary = [], [], []
    for seed in SEEDS:
        for model in MODELS:
            cell = cells[(seed, model)]
            for horizon in HORIZONS_MS:
                selected = [row for row in cell["rows"].values() if row["horizon_ms"] == horizon]
                for metric in cell["metrics"]:
                    means.append(dict(seed=seed, model=model, method=METHODS[model], horizon_ms=horizon,
                        metric=metric, metric_category="cost" if metric in COST_METRICS else "fidelity",
                        device=cell["device"], structural_na_reason=_structural_na(metric, horizon, cell["device"]),
                        **_metric_summary(selected, metric)))
        for candidate, reference in COMPARISONS:
            left, right = cells[(seed, candidate)], cells[(seed, reference)]
            for block in BLOCKS:
                for horizon in HORIZONS_MS:
                    keys = [key for key, row in left["rows"].items() if row["block"] == block and row["horizon_ms"] == horizon]
                    for metric in left["metrics"]:
                        a, b = [left["rows"][key][metric] for key in keys], [right["rows"][key][metric] for key in keys]
                        paired = [x - y if x is not None and y is not None else None for x, y in zip(a, b)]
                        available = sum(value is not None for value in paired)
                        reasons = sorted({_structural_na(metric, horizon, cell["device"]) for cell in (left, right)} - {""})
                        differences.append(dict(seed=seed, block=block, horizon_ms=horizon, candidate=candidate,
                            reference=reference, metric=metric, metric_category="cost" if metric in COST_METRICS else "fidelity",
                            candidate_device=left["device"], reference_device=right["device"],
                            candidate_mean=_mean(a), reference_mean=_mean(b), candidate_minus_reference=_mean(paired),
                            candidate_available=sum(value is not None for value in a),
                            reference_available=sum(value is not None for value in b), paired_available=available,
                            structural_unavailable_pairs=len(keys) - available, total_pairs=len(keys),
                            structural_na_reason=";".join(reasons)))
        endpoint = {model: _mean([row["energy_score"] for row in cells[(seed, model)]["rows"].values()
                                 if row["horizon_ms"] == 500.]) for model in MODELS}
        primary.append(dict(seed=seed, metric="energy_score", horizon_ms=500., observations_per_model=48,
            **{model + "_mean": value for model, value in endpoint.items()},
            current_minus_original=endpoint["current"] - endpoint["original"],
            history_minus_original=endpoint["history"] - endpoint["original"],
            history_minus_current=endpoint["history"] - endpoint["current"]))
    return means, differences, primary


def _accepted_eag(comparison_path, acceptance_path, cells):
    """Read the accepted paired EAG table; do not restore or rerun the baseline."""
    comparison_path, acceptance_path = Path(comparison_path), Path(acceptance_path)
    acceptance = json.loads(acceptance_path.read_text())
    _require(acceptance.get("status") == "ACCEPTED_COMPLETE_NATIVE_OBSERVATION_THREE_SEED_COMPARISONS",
             "complete native three-seed comparison acceptance required")
    receipt = acceptance["comparisons"]["eag"]
    report_digest = sha256_file(comparison_path)
    _require(report_digest == receipt["report_sha256"], "accepted EAG comparison hash differs")
    report = json.loads(comparison_path.read_text())
    table_path = comparison_path.parent / "joint_comparison_all_seeds.csv"
    table_digest = sha256_file(table_path)
    _require(table_digest == receipt["all_rows_sha256"]
             and table_digest == report["artifacts"][table_path.name]
             and receipt["block_rows_sha256"] == report["artifacts"]["joint_comparison_by_block.csv"],
             "accepted EAG paired-table hash differs")
    _require(report.get("status") == "COMPLETE_DESCRIPTIVE_DEVELOPMENT_COMPARISON"
             and report.get("matched_rows") == receipt.get("matched_rows") == 432
             and report.get("block_rows") == receipt.get("seed_block_horizon_rows") == 36
             and report.get("training_seeds") == list(SEEDS)
             and report.get("independent_blocks_per_seed") == 4 and report.get("connectivity_instances") == 1
             and report.get("baseline_method") == "eag"
             and report.get("candidate_method") == "population_recurrent_native_observation",
             "accepted EAG complete paired development protocol required")
    _require(report.get("baseline_observation_law") == "source_softplus_poisson"
             and report.get("baseline_native_refractory_qualified") is False
             and report.get("same_training_budget") is False and report.get("same_hardware") is False
             and report.get("full_end_to_end_cost_matched") is False
             and report.get("speedup_qualified") is False and report.get("inferential_statistics_qualified") is False
             and report.get("scientific_claim_qualified") is False,
             "accepted EAG observation law and fairness limits required")
    expected, bindings = {}, {}
    _require(set(report["bindings"]) == set(map(str, SEEDS))
             and set(report["baseline_training_bindings"]) == set(map(str, SEEDS)),
             "all three EAG source bindings required")
    for seed in SEEDS:
        original = cells[(seed, "original")]
        population_binding = report["bindings"][str(seed)]["population"]
        consumed = original["consumed"]
        _require(population_binding["delivery_sha256"] == consumed["delivery_sha256"]
                 and population_binding["consumed_files"]["forecasts/evaluation.json"] == consumed["artifacts"]["evaluation"]["sha256"]
                 and population_binding["consumed_files"]["forecasts/scores.csv"] == consumed["artifacts"]["scores"]["sha256"]
                 and population_binding["consumed_files"]["training/best.pt"] == consumed["parent_checkpoint_sha256"],
                 "EAG comparison original-parent binding differs")
        budget = report["baseline_training_bindings"][str(seed)]
        denominators = dict(fit_complete_episodes=56, validation_complete_episodes=16,
                            fit_origin_windows=168, validation_origin_windows=48, future_bins=100, units=100000)
        _require(budget.get("method") == "eag" and budget.get("autoencoder_completed_epochs") == 260
                 and budget.get("energy_completed_passes") == 4000
                 and budget.get("observation_law") == "source_softplus_poisson"
                 and budget.get("native_refractory_qualified") is False
                 and all(budget["source_denominators"].get(key) == value for key, value in denominators.items()),
                 "accepted EAG complete source counts and training budget required")
        expected.update(original["rows"])
        bindings[str(seed)] = dict(population_delivery_sha256=population_binding["delivery_sha256"],
                                  eag_delivery_sha256=report["bindings"][str(seed)]["eag"]["delivery_sha256"])
    metrics = [name for name in cells[(41, "original")]["metrics"] if name not in COST_METRICS]
    _require(len(metrics) == 14, "fourteen matched EAG fidelity metrics required")
    identity_columns = {"seed", "block", "case_id", "origin_ms", "horizon_ms", "candidate_method"}
    columns = identity_columns | {prefix + metric for metric in metrics
                                  for prefix in ("eag_", "population_", "population_minus_eag_")}
    with table_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames)
                 and set(reader.fieldnames) == columns, "complete unique EAG paired metric columns required")
        source_rows = list(reader)
    _require(len(source_rows) == 432, "complete 432-row EAG paired table required")
    rows = {}
    for raw in source_rows:
        _require(None not in raw, "unexpected EAG CSV fields")
        key = (int(raw["seed"]), raw["case_id"], float(raw["origin_ms"]), float(raw["horizon_ms"]))
        _require(key in expected and key not in rows, "duplicate or unexpected EAG source/horizon pairing")
        original = expected[key]
        _require(int(raw["block"]) == original["block"] and raw["candidate_method"] == report["candidate_method"],
                 "EAG paired source identity differs")
        row = {name: original[name] for name in ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")}
        for metric in metrics:
            row[metric] = _metric_value(raw["eag_" + metric], metric, key[-1], "")
        rows[key] = row
    _require(set(rows) == set(expected), "complete identical EAG paired coverage required")
    provenance = dict(acceptance_path=str(acceptance_path), acceptance_sha256=sha256_file(acceptance_path),
        comparison_path=str(comparison_path), accepted_report_path=receipt["report"], report_sha256=report_digest,
        table_path=str(table_path), all_rows_sha256=table_digest, matched_rows=432,
        parent_bindings=bindings, baseline_training_bindings=report["baseline_training_bindings"],
        reused_existing_evidence=True, training_repeated=False, generation_repeated=False, replay_repeated=False,
        limitations=dict(observation_law="source_softplus_poisson", native_refractory_qualified=False,
            same_training_budget=False, same_hardware=False, full_end_to_end_cost_matched=False,
            population_training_information="native event-time information", eag_training_information="binned unit counts",
            observed_scalar_precision="accepted scalar differences retained; no exact cross-model equality imposed",
            fit_episodes=56, validation_episodes=16, source_units=100000, significance_qualified=False,
            sota_qualified=False, matched_cost_or_speedup_qualified=False),
        observed_total_validation=report.get("observed_total_validation"))
    return rows, metrics, provenance


def _accepted_ldns_field_link(field_root, ldns_report, cells, source, common_fit):
    """Reuse the completed field comparison's authenticated old/new source link."""
    field_root = Path(field_root)
    intake_path, report_path = field_root / "intake.json", field_root / "comparison.json"
    intake = json.loads(intake_path.read_text())
    digest = sha256_file(report_path)
    _require(digest == intake.get("comparison_sha256"), "accepted field comparison hash differs")
    report = json.loads(report_path.read_text())
    status = "COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_FIELD_COMPARISON"
    _require(intake.get("status") == report.get("status") == status
             and report.get("training_seeds") == list(SEEDS)
             and report.get("score_rows") == 864 and report.get("block_rows") == 72
             and report.get("source_circuits") == 1 and report.get("independent_selection_blocks") == 4
             and intake.get("original_archives_verified") == 3 and intake.get("field_array_files_verified") == 294,
             "complete accepted native field comparison required")
    for name, count in (("field_comparison_all_seeds.csv", 864), ("field_comparison_by_block.csv", 72)):
        artifact = intake.get("artifacts", {}).get(name, {})
        _require(artifact.get("rows") == count and isinstance(artifact.get("sha256"), str)
                 and artifact["sha256"] == report.get("artifacts", {}).get(name),
                 "accepted field artifact binding differs")
    _require(all(intake.get(name) is False for name in
                 ("training_repeated", "generation_repeated", "scientific_claim_qualified", "sota_qualified"))
             and all(report.get(name) is False for name in
                 ("old_population_predictions_reused", "old_population_costs_reused", "training_repeated",
                  "inference_performed", "unit_samples_regenerated", "scientific_claim_qualified", "sota_qualified",
                  "inferential_statistics_performed", "full_end_to_end_cost_matched", "speedup_qualified")),
             "accepted field development limits required")
    execution = report.get("execution", {})
    _require(execution.get("backend") == "merlin" and bool(execution.get("run_id"))
             and bool(execution.get("trial_id"))
             and execution["run_id"] == intake.get("run_id") and execution["trial_id"] == intake.get("trial_id"),
             "accepted field Merlin execution binding differs")
    _require(report.get("source") == source and report.get("correction_fit") == common_fit,
             "accepted field source or correction fit differs")
    _require(set(report.get("bindings", {})) == set(map(str, SEEDS)), "all three field source bindings required")
    bindings = {}
    for seed in SEEDS:
        binding = report["bindings"][str(seed)]
        parent = cells[(seed, "original")]["consumed"]
        current, history = (cells[(seed, mode)]["consumed"] for mode in ("current", "history"))
        _require(binding["parent"]["delivery_sha256"] == parent["delivery_sha256"]
                 and binding["corrections"]["delivery_sha256"] == current["delivery_sha256"] == history["delivery_sha256"],
                 "accepted field native parent or correction delivery differs")
        old = binding["baseline_population"]
        population = ldns_report["bindings"][str(seed)]["population"]
        shared_files = ("training/run.json", "forecasts/fitting_field_scale.npz")
        _require(old["delivery_sha256"] == population["delivery_sha256"]
                 and all(name in old["consumed_files"] and name in population["consumed_files"]
                         and old["consumed_files"][name] == population["consumed_files"][name] for name in shared_files),
                 "LDNS old-population delivery, training report or fitting scale differs from accepted field source")
        bindings[str(seed)] = dict(native_parent_delivery_sha256=parent["delivery_sha256"],
            native_correction_delivery_sha256=current["delivery_sha256"],
            old_population_delivery_sha256=old["delivery_sha256"],
            old_population_consumed_files={name: old["consumed_files"][name] for name in shared_files},
            ldns_delivery_sha256=ldns_report["bindings"][str(seed)]["ldns"]["delivery_sha256"])
    return dict(intake_path=str(intake_path), intake_sha256=sha256_file(intake_path),
        comparison_path=str(report_path), report_sha256=digest, execution=execution, bindings=bindings,
        source_linkage="completed field comparison of source cohort, observation map, clock and fitting scale")


def _accepted_ldns(ldns_root, field_root, cells, source, common_fit):
    """Read the recovered LDNS baseline scores without changing its failed wrapper."""
    ldns_root = Path(ldns_root)
    intake_path, source_path = ldns_root / "delivery.json", ldns_root / "source_delivery.json"
    command_path = ldns_root / "source_command.json"
    report_path = ldns_root / "analysis/comparison.json"
    table_path = report_path.parent / "joint_comparison_all_seeds.csv"
    intake = json.loads(intake_path.read_text())
    _require(intake.get("status") == "PASS" and intake.get("exit_code") == 0
             and intake.get("baseline_method") == "ldns"
             and intake.get("candidate_method") == "population_recurrent_predictive_mixture"
             and intake.get("training_seeds") == list(SEEDS)
             and intake.get("matched_rows") == 432 and intake.get("block_rows") == 36
             and intake.get("source_artifacts_verified") == 6 and intake.get("source_seed_deliveries_verified") == 6
             and intake.get("source_failed_wrapper_preserved") is True
             and intake.get("implicit_ldns_schema_preserved") is True
             and all(intake.get(name) is False for name in
                 ("training_repeated", "forecast_generation_repeated", "comparison_repeated", "scientific_claim_qualified")),
             "complete recovered LDNS intake required")
    source_digest, report_digest = sha256_file(source_path), sha256_file(report_path)
    _require(source_digest == intake.get("source_delivery_sha256")
             and report_digest == intake.get("source_comparison_sha256"), "accepted LDNS source/report hash differs")
    wrapper = json.loads(source_path.read_text())
    _require(wrapper.get("status") == "FAILED" and wrapper.get("exit_code") == 2
             and wrapper.get("error") == "'baseline_method'" and wrapper.get("scientific_claim_qualified") is False,
             "original failed LDNS wrapper must remain preserved")
    command = json.loads(command_path.read_text())
    _require(isinstance(command, list) and all(isinstance(value, str) for value in command)
             and "-m" in command and command.index("-m") + 1 < len(command)
             and command[command.index("-m") + 1] == "population_dynamics.latent_spike_diffusion_comparison"
             and all(command.count(option) == 1 and command.index(option) + 1 < len(command)
                     for option in ("--ldns-root", "--population-root", "--output")),
             "preserved LDNS source comparison command required")
    report = json.loads(report_path.read_text())
    _require(report.get("status") == "COMPLETE_DESCRIPTIVE_DEVELOPMENT_COMPARISON"
             and report.get("matched_rows") == 432 and report.get("block_rows") == 36
             and report.get("training_seeds") == list(SEEDS)
             and report.get("independent_blocks_per_seed") == 4 and report.get("connectivity_instances") == 1
             and report.get("candidate_method") == intake["candidate_method"] and "baseline_method" not in report
             and set(report.get("bindings", {})) == set(map(str, SEEDS)),
             "accepted complete implicit LDNS development comparison required")
    _require(all(report.get(name) is False for name in
                 ("same_training_budget", "same_hardware", "full_end_to_end_cost_matched", "speedup_qualified",
                  "inferential_statistics_qualified", "scientific_claim_qualified")),
             "accepted LDNS fairness and development limits required")
    observed = report.get("observed_total_validation", {})
    _require(observed.get("reference") == "exact_common_observed_field_count_total_per_100000_units"
             and observed.get("rtol") == 5e-6 and observed.get("atol") == 1e-7
             and observed.get("tolerance_source") == "frozen_population_forecast_replay"
             and observed.get("observed_field_counts_match_exactly") is True
             and observed.get("saved_scalar_differences_retained") is True,
             "accepted LDNS observed-total validation and frozen tolerance required")
    table_digest = sha256_file(table_path)
    _require(table_digest == report.get("artifacts", {}).get(table_path.name), "accepted LDNS paired-table hash differs")
    field_link = _accepted_ldns_field_link(field_root, report, cells, source, common_fit)
    expected = {key: row for seed in SEEDS for key, row in cells[(seed, "original")]["rows"].items()}
    metrics = [name for name in cells[(41, "original")]["metrics"] if name not in COST_METRICS]
    _require(len(metrics) == 14, "fourteen matched LDNS fidelity metrics required")
    identity_columns = {"seed", "block", "case_id", "origin_ms", "horizon_ms", "candidate_method"}
    columns = identity_columns | {prefix + metric for metric in metrics
                                  for prefix in ("ldns_", "population_", "population_minus_ldns_")}
    with table_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames)
                 and set(reader.fieldnames) == columns, "complete unique LDNS paired metric columns required")
        source_rows = list(reader)
    _require(len(source_rows) == 432, "complete 432-row LDNS paired table required")
    rows = {}
    for raw in source_rows:
        _require(None not in raw, "unexpected LDNS CSV fields")
        key = (int(raw["seed"]), raw["case_id"], float(raw["origin_ms"]), float(raw["horizon_ms"]))
        _require(key in expected and key not in rows, "duplicate or unexpected LDNS source/horizon pairing")
        original = expected[key]
        _require(int(raw["block"]) == original["block"] and raw["candidate_method"] == report["candidate_method"],
                 "LDNS paired source identity differs")
        row = {name: original[name] for name in ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")}
        for metric in metrics:
            row[metric] = _metric_value(raw["ldns_" + metric], metric, key[-1], "")
        native_total = original["observed_spikes_per_unit"]
        _require(abs(row["observed_spikes_per_unit"] - native_total)
                 <= observed["atol"] + observed["rtol"] * abs(native_total),
                 "LDNS observed total differs from the admitted native source")
        rows[key] = row
    _require(set(rows) == set(expected), "complete identical LDNS paired coverage required")
    provenance = dict(intake_path=str(intake_path), intake_sha256=sha256_file(intake_path),
        comparison_path=str(report_path), report_sha256=report_digest,
        table_path=str(table_path), all_rows_sha256=table_digest, matched_rows=432,
        source_delivery=dict(path=str(source_path), sha256=source_digest, status=wrapper["status"],
                             exit_code=wrapper["exit_code"], error=wrapper["error"]),
        source_command=dict(path=str(command_path), sha256=sha256_file(command_path), command=command),
        field_source_link=field_link, observed_total_validation=observed,
        reused_existing_evidence=True, training_repeated=False, generation_repeated=False, replay_repeated=False,
        old_population_predictions_reused=False, old_population_costs_reused=False,
        limitations=dict(same_training_budget=False, same_hardware=False, full_end_to_end_cost_matched=False,
            significance_qualified=False, sota_qualified=False, matched_cost_or_speedup_qualified=False,
            observed_scalar_precision="accepted scalar differences retained; no exact cross-model equality imposed"))
    return rows, metrics, provenance


def _baseline_comparisons(cells, baseline_rows, metrics, primary, *, reference):
    paired, blocks = [], []
    for seed in SEEDS:
        for model in ("current", "history"):
            rows = cells[(seed, model)]["rows"]
            for key, candidate in sorted(rows.items()):
                baseline = baseline_rows[key]
                row = {name: candidate[name] for name in ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")}
                row.update(candidate=model, candidate_method=METHODS[model], reference=reference)
                for metric in metrics:
                    a, b = candidate[metric], baseline[metric]
                    row["candidate_" + metric], row[reference + "_" + metric] = a, b
                    row["candidate_minus_" + reference + "_" + metric] = a - b if a is not None and b is not None else None
                paired.append(row)
            for block in BLOCKS:
                for horizon in HORIZONS_MS:
                    keys = [key for key, row in rows.items() if row["block"] == block and row["horizon_ms"] == horizon]
                    for metric in metrics:
                        a, b = [rows[key][metric] for key in keys], [baseline_rows[key][metric] for key in keys]
                        differences = [x - y if x is not None and y is not None else None for x, y in zip(a, b)]
                        available = sum(value is not None for value in differences)
                        blocks.append(dict(seed=seed, block=block, horizon_ms=horizon, candidate=model, reference=reference,
                            metric=metric, candidate_mean=_mean(a),
                            **{reference + "_mean": _mean(b), "candidate_minus_" + reference: _mean(differences)},
                            paired_available=available, structural_unavailable_pairs=len(keys) - available,
                            total_pairs=len(keys), structural_na_reason=_structural_na(metric, horizon, "")))
        endpoint = _mean([row["energy_score"] for key, row in baseline_rows.items() if key[0] == seed and key[-1] == 500.])
        row = next(row for row in primary if row["seed"] == seed)
        row.update({reference + "_mean": endpoint, "current_minus_" + reference: row["current_mean"] - endpoint,
                    "history_minus_" + reference: row["history_mean"] - endpoint})
    return paired, blocks


def admit_comparison_cells(corrections: Path, parent_campaign: Path):
    """Reuse the complete three-seed parent/refit/evaluation/replay admission."""
    corrections, parent_campaign = map(Path, (corrections, parent_campaign))
    cells, reference, common_fit = {}, None, None
    for seed in SEEDS:
        parent_cell, correction_cell = parent_campaign / f"seed_{seed}", corrections / f"seed_{seed}"
        parent_delivery, parent_manifest, parent_digest = _delivery(parent_cell, seed, corrected=False)
        delivery, manifest, delivery_digest = _delivery(correction_cell, seed, corrected=True)
        fit_context = _fit_context(correction_cell, delivery, manifest, seed)
        fit_identity = fit_context["common"]
        _require(common_fit is None or fit_identity == common_fit,
                 "all seeds must use the same fitted correction manifest")
        common_fit = fit_identity
        original = _admit(parent_cell, parent_delivery, parent_manifest, parent_digest, seed, "original")
        identity = {name: original["evaluation"][name] for name in
                    ("dataset_sha256", "acquisition_design_sha256", "native_source_identity", "configuration")}
        _require(reference is None or identity == reference, "training seeds must share one connectivity, source cohort and forecast protocol")
        reference = identity
        cells[(seed, "original")] = original
        for model in ("current", "history"):
            cells[(seed, model)] = _admit(correction_cell, delivery, manifest, delivery_digest, seed, model, original, fit_context)
    return cells, reference, common_fit


def analyze_ldns(corrections: Path, parent_campaign: Path, output: Path, *, ldns_comparison: Path,
                 field_comparison: Path, execution=None):
    """Compare only current/history native forecasts with accepted LDNS scores."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("comparison requires a fresh output directory")
    cells, source, common_fit = admit_comparison_cells(corrections, parent_campaign)
    baseline, metrics, provenance = _accepted_ldns(ldns_comparison, field_comparison, cells, source, common_fit)
    primary = [dict(seed=seed, metric="energy_score", horizon_ms=500., observations_per_model=48,
        **{model + "_mean": _mean([row["energy_score"] for row in cells[(seed, model)]["rows"].values()
                                  if row["horizon_ms"] == 500.]) for model in ("current", "history")}) for seed in SEEDS]
    paired, blocks = _baseline_comparisons(cells, baseline, metrics, primary, reference="ldns")
    tables = {"correction_vs_ldns_all_seeds.csv": paired, "correction_vs_ldns_by_block.csv": blocks,
              "primary_500ms.csv": primary}
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    report = dict(status="COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_LDNS_COMPARISON", training_seeds=list(SEEDS),
        models=["current", "history"], reference="ldns", score_rows=len(paired), block_metric_rows=len(blocks),
        source_block_horizon_rows=72, fidelity_metrics=metrics, score_rows_per_model_seed=144,
        paired_origins_per_model_seed=48,
        primary_endpoint=dict(metric="energy_score", horizon_ms=500., direction="lower_is_better",
                              evidence="native_full_unit_generation", models=["current", "history"], posterior_proxy_used=False),
        primary_500ms=primary, difference_definition="candidate minus reference; no gain eligibility gate",
        scope=dict(coupled_connectivities=1, selection_episodes=16, source_blocks=list(BLOCKS),
                   repeated_training_seeds=3, seeds_are_not_independent_biological_replicates=True,
                   independent_confidence_intervals_computed=False, significance_tests_computed=False,
                   formal_evaluation=False, scientific_claim_qualified=False, sota_qualified=False,
                   matched_hardware_speedup_qualified=False, full_end_to_end_cost_matched=False),
        source=source, correction_fit=common_fit,
        consumed=[cells[(seed, model)]["consumed"] for seed in SEEDS for model in ("current", "history")],
        ldns_reference=provenance, large_arrays_or_checkpoints_read=False,
        old_population_predictions_reused=False, old_population_costs_reused=False,
        training_repeated=False, generation_repeated=False,
        artifact_verification="native delivery/replay receipts and recovered LDNS scores linked by accepted field comparison",
        artifacts={name: dict(sha256=sha256_file(output / name), rows=len(rows)) for name, rows in tables.items()})
    if execution is not None:
        report["execution"] = execution
    freeze_json(output / "analysis.json", report)
    return report


def analyze(corrections: Path, parent_campaign: Path, output: Path, *, eag_comparison=None, eag_acceptance=None,
            execution=None):
    _require((eag_comparison is None) == (eag_acceptance is None), "EAG comparison and acceptance must be supplied together")
    corrections, parent_campaign, output = map(Path, (corrections, parent_campaign, output))
    if output.exists():
        raise FileExistsError("comparison requires a fresh output directory")
    cells, reference, common_fit = admit_comparison_cells(corrections, parent_campaign)
    means, differences, primary = summarize(cells)
    all_rows = [row for seed in SEEDS for model in MODELS for _, row in sorted(cells[(seed, model)]["rows"].items())]
    tables = {"all_model_scores.csv": all_rows, "per_seed_horizon_metrics.csv": means,
              "paired_source_block_differences.csv": differences, "primary_500ms.csv": primary}
    eag_provenance = None
    if eag_comparison is not None:
        eag_rows, eag_metrics, eag_provenance = _accepted_eag(eag_comparison, eag_acceptance, cells)
        eag_paired, eag_blocks = _baseline_comparisons(cells, eag_rows, eag_metrics, primary, reference="eag")
        tables["correction_vs_eag_all_seeds.csv"] = eag_paired
        tables["correction_vs_eag_by_block.csv"] = eag_blocks
    output.mkdir(parents=True)
    for name, rows in tables.items():
        _write_rows(output / name, rows)
    report = dict(status="COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_COMPARISON", training_seeds=list(SEEDS), models=list(MODELS),
        score_rows_per_model_seed=144, paired_origins_per_model_seed=48, total_score_rows=len(all_rows),
        primary_endpoint=dict(metric="energy_score", horizon_ms=500., direction="lower_is_better",
                              evidence="native_full_unit_generation", models=list(MODELS), posterior_proxy_used=False),
        primary_500ms=primary, difference_definition="candidate minus reference; no gain eligibility gate",
        scope=dict(coupled_connectivities=1, selection_episodes=16, source_blocks=list(BLOCKS),
                   repeated_training_seeds=3, seeds_are_not_independent_biological_replicates=True,
                   independent_confidence_intervals_computed=False, significance_tests_computed=False,
                   formal_evaluation=False, scientific_claim_qualified=False, sota_qualified=False,
                   matched_hardware_speedup_qualified=False,
                   hardware="recorded separately for every seed/model; CPU and CUDA costs are not matched-hardware speed evidence"),
        source=reference, correction_fit=common_fit, consumed=[cells[(seed, model)]["consumed"] for seed in SEEDS for model in MODELS],
        large_arrays_or_checkpoints_read=False, artifact_verification="small files through delivery manifests; existing independent replay receipts",
        artifacts={name: dict(sha256=sha256_file(output / name), rows=len(rows)) for name, rows in tables.items()})
    if eag_provenance is not None:
        report["eag_reference"] = eag_provenance
    if execution is not None:
        report["execution"] = execution
    freeze_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corrections", "parent-campaign", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--eag-comparison", type=Path)
    comparison.add_argument("--ldns-comparison", type=Path, help="Recovered LDNS intake directory")
    parser.add_argument("--eag-acceptance", type=Path)
    parser.add_argument("--field-comparison", type=Path, help="Accepted field intake.json and comparison.json directory")
    args = parser.parse_args()
    if (args.eag_comparison is None) != (args.eag_acceptance is None):
        parser.error("--eag-comparison and --eag-acceptance must be supplied together")
    if (args.ldns_comparison is None) != (args.field_comparison is None):
        parser.error("--ldns-comparison and --field-comparison must be supplied together")
    execution = cpu_execution_provenance()
    if args.ldns_comparison is not None:
        report = analyze_ldns(args.corrections, args.parent_campaign, args.output,
                              ldns_comparison=args.ldns_comparison, field_comparison=args.field_comparison, execution=execution)
    else:
        report = analyze(args.corrections, args.parent_campaign, args.output,
                         eag_comparison=args.eag_comparison, eag_acceptance=args.eag_acceptance, execution=execution)
    print(json.dumps(dict(status=report["status"], primary_500ms=report["primary_500ms"])))


if __name__ == "__main__":
    main()

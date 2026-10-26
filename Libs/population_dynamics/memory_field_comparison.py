"""Matched source-field MemNO comparison using existing BRAID/population evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .braid_population_comparison import DeliveredCell, _population_method
from .population_forecast_analysis import _is_prior_refit, _prior_protocol
from .provenance import sha256_file, cpu_execution_provenance
from .population_forecast_analysis import _delivery_file
from .population_residual_correction_analysis import admit_comparison_cells
from .population_residual_refit_provenance import FAMILY_SCHEMA
from .lif_neural_field import sampled_mean_field_mse


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys():
    return {(seed, block, f"train_block_{block:03d}_familiar_{template}", origin, horizon)
            for seed, block, template, origin, horizon in itertools.product(
                (41, 42, 43), range(14, 18), range(4), (1000., 3000., 5000.), (20., 80., 500.))}


def standardized_error(prediction, truth, sd):
    """Source Hz-per-neuron fields with the original fit-only Hz scale."""
    require(prediction.shape == truth.shape and prediction.ndim == 2
            and sd.shape == (prediction.shape[-1],) and np.isfinite(sd).all() and np.all(sd > 0)
            and np.isfinite(prediction).all() and np.isfinite(truth).all()
            and np.all(prediction >= 0) and np.all(truth >= 0), "complete finite source fields and positive scale required")
    return float(np.mean(((prediction.astype(np.float64) - truth.astype(np.float64)) / sd) ** 2))


def compare(memory_campaign, braid_comparison, population_campaign, output):
    memory_campaign, braid_comparison, population_campaign, output = map(Path,
        (memory_campaign, braid_comparison, population_campaign, output))
    require(not output.exists(), "comparison needs a new output")
    prior_delivery = read(braid_comparison / "delivery.json")
    require(prior_delivery["status"] == "PASS" and prior_delivery["exit_code"] == 0,
            "accepted complete BRAID comparison required")
    prior_files = {row["path"]: row for row in prior_delivery["files"]}
    prior_report_path = braid_comparison / "comparison/comparison.json"
    require(sha256_file(prior_report_path) == prior_files["comparison/comparison.json"]["sha256"],
            "BRAID comparison source report changed")
    prior_report = read(prior_report_path)
    table = braid_comparison / "comparison/mean_comparison_all_seeds.csv"
    require(sha256_file(table) == prior_files[str(table.relative_to(braid_comparison))]["sha256"],
            "BRAID comparison table changed")
    with table.open(newline="") as stream:
        previous = list(csv.DictReader(stream))
    indexed = {}
    for row in previous:
        key = (int(row["seed"]), int(row["block"]), row["case_id"], float(row["origin_ms"]), float(row["horizon_ms"]))
        require(key not in indexed, "duplicate prior condition")
        indexed[key] = row
    require(set(indexed) == keys(), "complete432 registered previous conditions required")
    rows, bindings, prior_bindings = [], {}, {}
    for seed in (41, 42, 43):
        memory = DeliveredCell(memory_campaign / f"seed_{seed}", seed)
        require(memory.proof.get("complete_training") is True and memory.proof.get("replay_performed") is True,
                "complete MemNO training and fresh-process replay required")
        run = memory.read("training/run.json")
        evaluation = memory.read("evaluation/evaluation.json")
        replay = memory.read("replay/evaluation.json")
        require(run["status"] == "FITTED_DEVELOPMENT" and run["completed_epochs"] == 300
                and evaluation["status"] == "EVALUATED_DEVELOPMENT" and evaluation["score_rows"] == 144
                and evaluation["origins"] == 48 and evaluation["partial_training_diagnostic"] is False
                and replay["status"] == "REPLAYED_DEVELOPMENT" and replay["score_rows"] == 144
                and replay["binding"] == evaluation["binding"] == run["binding"],
                "complete matched MemNO fit,forecast and replay required")
        population = DeliveredCell(population_campaign / f"seed_{seed}", seed)
        require(prior_report["consumed_deliveries"][str(seed)]["population"]["delivery_sha256"]
                == sha256_file(population.root / "delivery.json"), "prior population delivery differs")
        other = population.read("training/run.json")
        prior_binding = None
        if _is_prior_refit(other):
            initialization = _prior_protocol(other)
            expected_method = _population_method(other, "")
            prior_binding = prior_report.get("population_training_bindings", {}).get(str(seed))
            require(isinstance(prior_binding, dict)
                    and prior_binding.get("method") == expected_method
                    and prior_report.get("candidate_method") == expected_method
                    and prior_binding.get("training_objective") == other["training_objective"]
                    and prior_binding.get("selection_metric") == other["selection_metric"],
                    "BRAID comparison must identify the same prior objective")
            lineage = prior_binding.get("prior_refit_lineage", {})
            require(all(lineage.get(key) == initialization.get(key) for key in
                        ("parent_checkpoint_sha256", "parent_report_sha256", "parent_selected_epoch", "parent_completed_epochs")),
                    "BRAID comparison prior parent lineage differs")
            prior_bindings[str(seed)] = prior_binding
        source, config = run["binding"]["dataset"], run["binding"]["config"]
        require(config["fit_blocks"] == list(range(14)) and config["validation_blocks"] == list(range(14, 18))
                and source["field_units"] == "spikes_per_second_per_neuron"
                and source["bin_seconds"] == .005 and run["field_mean_only"] is True,
                "matched full source field protocol required")
        for name in ("acquisition_design_sha256", "acquisition_ledger_sha256", "dataset_sha256",
                     "episode_table_sha256", "episode_bindings"):
            require(source["cohort"][name] == other["binding"][name], "source cohort differs: " + name)
        for name, value in source["cohort"]["native_source_identity"].items():
            require(other["binding"]["native_source_identity"][name] == value, "native source differs")
        model = other["model_configuration"]
        require(source["source_unit_ids"] == model["unit_ids"]
                and len(model["unit_ids"]) == 100000
                and source["source_field_ids"] == model["field_names"] and model["bin_seconds"] == .005,
                "source unit,field order or clock differs")
        with np.load(population.path("forecasts/fitting_field_scale.npz"), allow_pickle=False) as scale:
            sd, units, ids = [scale[name].copy() for name in ("standard_deviation_hz", "unit_count", "field_ids")]
        require(np.array_equal(ids, np.asarray(source["source_field_ids"]))
                and np.array_equal(units, np.bincount(model["unit_field_index"], minlength=len(ids)))
                and units.sum() == 100000 and np.all(units > 0), "complete source field occupancies differ")
        cached = {}
        for key in sorted(k for k in indexed if k[0] == seed):
            _, block, case, origin, horizon = key
            if (case, origin) not in cached:
                name = f"evaluation/{case}_origin_{int(origin / 5):04d}.npz"
                require(evaluation["artifacts"][Path(name).name] == memory.files[name]["sha256"],
                        "MemNO forecast binding changed")
                with np.load(memory.path(name), allow_pickle=False) as values:
                    prediction, truth = values["predicted_rate_hz"].copy(), values["truth_rate_hz"].copy()
                    require(np.array_equal(values["source_field_ids"], ids)
                            and np.array_equal(values["source_unit_ids"], np.asarray(model["unit_ids"]))
                            and values["bin_seconds"].item() == .005 and values["origin_bin"].item() == origin / 5
                            and prediction.shape == truth.shape == (100, len(ids)), "MemNO forecast map or target bins differ")
                with np.load(population.path(f"forecasts/{case}_origin_{origin:g}.npz"), allow_pickle=False) as values:
                    counts = values["observed_field_counts"].copy()
                # MemNO persisted source rates are float32. This tolerance only
                # covers that representation; source IDs and ticks match exactly.
                require(np.allclose(truth.astype(np.float64) * (units * .005), counts, rtol=2e-7, atol=1e-5),
                        "MemNO and prior complete source truth differ")
                cached[case, origin] = prediction, counts.astype(np.float64) / (units * .005)
            prediction, truth = cached[case, origin]
            bins = int(horizon / 5)
            score = standardized_error(prediction[:bins], truth[:bins], sd)
            prior = indexed[key]
            braid_score = float(prior["braid_field_mean_mse"])
            population_score = float(prior["population_field_mean_mse_corrected"])
            require(np.isfinite(braid_score) and np.isfinite(population_score), "finite prior metric required")
            rows.append(dict(seed=seed, block=block, case_id=case, origin_ms=origin, horizon_ms=horizon,
                **({"candidate_method": prior_binding["method"]} if prior_binding is not None else {}),
                memno_field_mean_mse=score, braid_field_mean_mse=braid_score,
                population_field_mean_mse_corrected=population_score,
                memno_minus_braid_field_mean_mse=score - braid_score,
                population_minus_memno_field_mean_mse=population_score - score))
        bindings[str(seed)] = dict(memory=memory.binding(), population=population.binding())
    groups = defaultdict(list)
    for row in rows:
        groups[row["seed"], row["block"], row["horizon_ms"]].append(row)
    blocks = []
    for (seed, block, horizon), values in sorted(groups.items()):
        require(len(values) == 12, "complete fourtemplates/threeorigins per block required")
        blocks.append(dict(seed=seed, block=block, horizon_ms=horizon, matched_conditions=12,
                          **{name:float(np.mean([row[name] for row in values])) for name in rows[0] if name.endswith("mse")}))
    require(len(rows) == 432 and len(blocks) == 36, "complete432scores/36block groups required")
    require(not prior_bindings or (set(prior_bindings) == {"41", "42", "43"}
            and len({binding["training_objective"] for binding in prior_bindings.values()}) == 1),
            "all three population seeds must use the same prior objective")
    output.mkdir(parents=True)
    for name, values in (("field_comparison_all_seeds.csv", rows), ("field_comparison_by_block.csv", blocks)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    (output / "comparison.json").write_text(json.dumps(dict(status="COMPLETE_DESCRIPTIVE_FIELD_MEAN_COMPARISON",
        score_rows=432, block_rows=36, bindings=bindings, prior_delivery_sha256=sha256_file(braid_comparison / "delivery.json"),
        metric="unweighted_time_field_MSE_of_source_Hz_per_neuron_divided_by_original_fit_standard_deviation_Hz",
        population_metric="unchanged_saved_finite16draw_variance_corrected_mean_MSE",
        **(dict(candidate_method=prior_bindings["41"]["method"], population_training_bindings=prior_bindings)
           if prior_bindings else {}),
        scientific_claim_qualified=False, inference_performed=False, full_unit_speedup_qualified=False,
        source_circuits=1, independent_selection_blocks=4, same_training_budget=False), indent=2) + "\n")
    return dict(score_rows=432, block_rows=36)



def _baseline_table(root, table_name, expected_status):
    root = Path(root)
    delivery = read(root / "delivery.json")
    require(delivery.get("status") == "PASS" and delivery.get("exit_code") == 0, "accepted baseline comparison required")
    index = {row["path"]: row for row in delivery["files"]}
    require(len(index) == len(delivery["files"]), "duplicate baseline comparison path")
    report_path = _delivery_file(root, index, "comparison/comparison.json", "baseline report")
    table_path = _delivery_file(root, index, "comparison/" + table_name, "baseline table")
    report = read(report_path)
    require(report.get("status") == expected_status and report.get("score_rows") == 432,
            "complete baseline field comparison required")
    if "artifacts" in report:
        require(report["artifacts"].get(table_name) == sha256_file(table_path), "baseline report/table binding differs")
    with table_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "unique baseline table columns required")
        records = list(reader)
    indexed = {}
    for row in records:
        key = (int(row["seed"]), int(row["block"]), row["case_id"], float(row["origin_ms"]), float(row["horizon_ms"]))
        require(key not in indexed, "duplicate baseline condition")
        indexed[key] = row
    require(set(indexed) == keys() and len(records) == 432, "all baseline field conditions required")
    return report, indexed, dict(delivery_sha256=sha256_file(root / "delivery.json"),
        report_sha256=sha256_file(report_path), table_sha256=sha256_file(table_path))


def _field_scale(cell, relative, model):
    """Read the four scientific arrays written by the matched evaluator."""
    path = cell.path(relative)
    with np.load(path, allow_pickle=False) as values:
        mean, sd, units, ids = (values[key].copy() for key in
            ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids"))
    require(mean.shape == ids.shape == units.shape == sd.shape == (128,)
            and np.array_equal(ids, np.asarray(model["field_names"]))
            and units.dtype.kind in "iu"
            and np.array_equal(units, np.bincount(model["unit_field_index"], minlength=len(ids)))
            and units.sum() == 100000 and np.all(units > 0)
            and np.isfinite(mean).all() and np.all(mean >= 0)
            and np.isfinite(sd).all() and np.all(sd > 0),
            "complete source fields, occupancies and fit scale required")
    return mean, sd, units, ids, sha256_file(path)


def native_field_scores(samples, truth, sd, units):
    require(samples.shape == (16, 100, 128) and truth.shape == (100, 128)
            and np.isfinite(samples).all() and np.isfinite(truth).all()
            and np.all(samples >= 0) and np.all(truth >= 0) and np.all(truth == np.rint(truth)),
            "complete finite sixteen-draw source field arrays required")
    return {horizon: sampled_mean_field_mse(samples[:, :int(horizon / 5)],
        truth[:int(horizon / 5)], sd, units, 5.)
        for horizon in (20., 80., 500.)}


def compare_native(corrections, parent_campaign, baseline_population_campaign, braid_comparison,
                   memory_comparison, output, *, execution=None):
    """Reuse accepted baseline columns; evaluate only the new saved field samples."""
    output = Path(output)
    require(not output.exists(), "comparison needs a new output")
    cells, source, family = admit_comparison_cells(corrections, parent_campaign)
    require(family.get("schema") == FAMILY_SCHEMA, "native residual refit family required")
    braid_report, braid_rows, braid_proof = _baseline_table(braid_comparison, "mean_comparison_all_seeds.csv",
        "COMPLETE_DESCRIPTIVE_BRAID_POPULATION_MEAN_COMPARISON")
    memory_report, memory_rows, memory_proof = _baseline_table(memory_comparison, "field_comparison_all_seeds.csv",
        "COMPLETE_DESCRIPTIVE_FIELD_MEAN_COMPARISON")
    require(memory_report.get("prior_delivery_sha256") == braid_proof["delivery_sha256"],
            "MemNO reference must bind the accepted BRAID comparison")
    rows, bindings, scale_bindings = [], {}, {}
    source_keys = ("acquisition_design_sha256", "acquisition_ledger_sha256", "dataset_sha256",
                   "episode_table_sha256", "episode_bindings", "native_source_identity")
    map_keys = ("unit_ids", "unit_field_index", "field_names", "minimum_interval_ticks", "bin_seconds", "tick_seconds")
    for seed in (41, 42, 43):
        old = DeliveredCell(Path(baseline_population_campaign) / f"seed_{seed}", seed)
        parent = DeliveredCell(Path(parent_campaign) / f"seed_{seed}", seed)
        corrected = DeliveredCell(Path(corrections) / f"seed_{seed}", seed)
        old_run, parent_run = old.read("training/run.json"), parent.read("training/run.json")
        require(all(run.get("status") == "FITTED_DEVELOPMENT" and run.get("training_seed") == seed
                    for run in (old_run, parent_run)), "complete same-seed parent training reports required")
        old_digest = sha256_file(old.root / "delivery.json")
        b_binding = braid_report["consumed_deliveries"][str(seed)]["population"]
        m_binding = memory_report["bindings"][str(seed)]["population"]
        require(b_binding["delivery_sha256"] == m_binding["delivery_sha256"] == old_digest,
                "baseline original population delivery differs")
        for binding in (b_binding, m_binding):
            require(binding["consumed_files"]["training/run.json"] == sha256_file(old.path("training/run.json")),
                    "baseline original source report differs")
        require(all(old_run["binding"][key] == parent_run["binding"][key] for key in source_keys),
                "old baseline and new candidate source cohort differs")
        old_model, model = old_run["model_configuration"], parent_run["model_configuration"]
        require(all(old_model[key] == model[key] for key in map_keys)
                and model["bin_seconds"] == .005 and model["tick_seconds"] == .0001
                and len(model["unit_ids"]) == 100000, "source observation map or native clock differs")
        original = cells[(seed, "original")]["evaluation"]
        require(original["checkpoint_sha256"] == parent_run["best_checkpoint_sha256"]
                and all(original[key] == parent_run["binding"][key]
                        for key in ("dataset_sha256", "acquisition_design_sha256", "native_source_identity")),
                "new parent training and native comparison source differs")
        mean, sd, units, ids, scale_digest = _field_scale(old, "forecasts/fitting_field_scale.npz", model)
        scale_bindings[str(seed)] = dict(baseline_sha256=scale_digest,
            equality="exact mean_hz, standard_deviation_hz, unit_count and field_ids arrays")
        for binding in (b_binding, m_binding):
            require(binding["consumed_files"]["forecasts/fitting_field_scale.npz"] == scale_digest,
                    "baseline fitting scale binding differs")
        for mode in ("current", "history"):
            prefix = f"results/{mode}/forecasts/"
            candidate_mean, candidate_sd, candidate_units, candidate_ids, current_scale = _field_scale(
                corrected, prefix + "fitting_field_scale.npz", model)
            require(all(np.array_equal(a, b) for a, b in (
                (candidate_mean, mean), (candidate_sd, sd), (candidate_units, units), (candidate_ids, ids))),
                "native correction must use the original fitting scale arrays")
            scale_bindings[str(seed)][mode + "_sha256"] = current_scale
            evaluation = cells[(seed, mode)]["evaluation"]
            # The current writer authenticates scale through delivery, and puts
            # only origin NPZs/full_unit_samples in evaluation.artifacts. Honor
            # an explicit scale artifact binding when a writer supplies one.
            if "fitting_field_scale.npz" in evaluation["artifacts"]:
                require(evaluation["artifacts"]["fitting_field_scale.npz"] == current_scale,
                        "evaluated fitting scale artifact differs")
            for block, template, origin in itertools.product(range(14, 18), range(4), (1000., 3000., 5000.)):
                case = f"train_block_{block:03d}_familiar_{template}"
                relative = prefix + f"{case}_origin_{origin:g}.npz"
                path = corrected.path(relative)
                require(evaluation["artifacts"].get(path.name) == corrected.files[relative]["sha256"],
                        "native field array differs from evaluated artifact")
                with np.load(path, allow_pickle=False) as values:
                    # Origin NPZs carry counts/latents, not a second field map.
                    # Their axes inherit the authenticated evaluator model/scale.
                    scores = native_field_scores(values["sampled_field_counts"], values["observed_field_counts"], sd, units)
                for horizon, score in scores.items():
                    key = (seed, block, case, origin, horizon)
                    b_score = float(braid_rows[key]["braid_field_mean_mse"])
                    m_score = float(memory_rows[key]["memno_field_mean_mse"])
                    require(np.isfinite([score, b_score, m_score]).all()
                            and float(memory_rows[key]["braid_field_mean_mse"]) == b_score,
                            "baseline field score or shared BRAID reference differs")
                    rows.append(dict(seed=seed, block=block, case_id=case, origin_ms=origin, horizon_ms=horizon,
                        candidate=mode, candidate_method=evaluation["method"], candidate_field_mean_mse_corrected=score,
                        braid_field_mean_mse=b_score, memno_field_mean_mse=m_score,
                        candidate_minus_braid_field_mean_mse=score-b_score, candidate_minus_memno_field_mean_mse=score-m_score))
        bindings[str(seed)] = dict(baseline_population=old.binding(), parent=parent.binding(), corrections=corrected.binding())
    groups = defaultdict(list)
    for row in rows:
        groups[row["candidate"], row["seed"], row["block"], row["horizon_ms"]].append(row)
    blocks = []
    for (mode, seed, block, horizon), values in sorted(groups.items()):
        require(len(values) == 12, "complete template/origin block mean required")
        blocks.append(dict(candidate=mode, seed=seed, block=block, horizon_ms=horizon, matched_conditions=12,
            **{name: float(np.mean([row[name] for row in values])) for name in rows[0] if name.endswith("mse") or name.endswith("mse_corrected")}))
    require(len(rows) == 864 and len(blocks) == 72, "both complete native field comparisons required")
    output.mkdir(parents=True)
    for name, values in (("field_comparison_all_seeds.csv", rows), ("field_comparison_by_block.csv", blocks)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    result = dict(status="COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_FIELD_COMPARISON", training_seeds=[41, 42, 43],
        score_rows=864, block_rows=72, source=source, correction_fit=family, bindings=bindings,
        fit_scale_bindings=scale_bindings, baseline_evidence=dict(braid=braid_proof, memno=memory_proof),
        baseline_reports=dict(braid=braid_report, memno=memory_report),
        metric="original fitting-scale standardized source-field mean MSE with unchanged finite16draw correction",
        old_population_predictions_reused=False, old_population_costs_reused=False,
        training_repeated=False, inference_performed=False, unit_samples_regenerated=False,
        scientific_claim_qualified=False, sota_qualified=False, inferential_statistics_performed=False,
        full_end_to_end_cost_matched=False, speedup_qualified=False, source_circuits=1, independent_selection_blocks=4,
        artifacts={name: sha256_file(output / name) for name in ("field_comparison_all_seeds.csv", "field_comparison_by_block.csv")})
    if execution is not None:
        result["execution"] = execution
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("braid-comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("memory-campaign", "population-campaign", "native-corrections", "parent-campaign",
                 "baseline-population-campaign", "memory-comparison"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.native_corrections is not None:
        if args.memory_campaign is not None or args.population_campaign is not None or any(
                value is None for value in (args.parent_campaign, args.baseline_population_campaign, args.memory_comparison)):
            parser.error("native route requires parent-campaign, baseline-population-campaign and memory-comparison; omit legacy campaign arguments")
        result = compare_native(args.native_corrections, args.parent_campaign, args.baseline_population_campaign,
            args.braid_comparison, args.memory_comparison, args.output, execution=cpu_execution_provenance())
    else:
        if args.memory_campaign is None or args.population_campaign is None or any(value is not None for value in (
                args.parent_campaign, args.baseline_population_campaign, args.memory_comparison)):
            parser.error("legacy route requires memory-campaign and population-campaign")
        result = compare(args.memory_campaign, args.braid_comparison, args.population_campaign, args.output)
    print(json.dumps({key: result[key] for key in ("score_rows", "block_rows")}))

if __name__ == "__main__":
    main()

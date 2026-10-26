"""Compare delivered BRAID and population means without new model execution.

The four epoch-selection blocks support descriptive development comparisons.
Training seeds and the sixteen population draws are not independent circuits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .lif_neural_field import sampled_mean_field_mse
from .population_forecast_analysis import (
    validate_evaluation_costs, _measured_origin_columns, _training_attempt_cost,
    _is_prior_refit, _prior_protocol, _prior_parent_evidence, _prior_sampling_budget, PRIOR_METHODS,
)
from .provenance import sha256_file

SEEDS = (41, 42, 43)
ORIGINS = (1000., 3000., 5000.)
HORIZONS = (20., 80., 500.)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text())


def _number(value):
    result = float(value)
    _require(np.isfinite(result), "finite metric or cost required")
    return result


def _key(row):
    return (int(row["block"]), row["case_id"],
            _number(row["origin_ms"]), _number(row["horizon_ms"]))


def _score_table(path, seed, method):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                 "unique score columns required")
        rows = list(reader)
    expected = {(block, f"train_block_{block:03d}_familiar_{template}", origin, horizon)
                for block, template, origin, horizon in itertools.product(
                    range(14, 18), range(4), ORIGINS, HORIZONS)}
    result = {}
    for row in rows:
        key = _key(row)
        _require(int(row["seed"]) == seed and row["method"] == method
                 and key in expected and key not in result,
                 "duplicate, foreign or misidentified score condition")
        result[key] = row
    _require(set(result) == expected and len(rows) == 144,
             "complete 16-episode/48-origin/144-score seed required")
    return result


class DeliveredCell:
    """Check only declared consumed inputs against the successful delivery."""

    def __init__(self, root, seed):
        self.root = Path(root)
        self.proof = _read(self.root / "delivery.json")
        _require(self.proof["status"] == "PASS" and self.proof["exit_code"] == 0
                 and self.proof["training_seed"] == seed
                 and self.proof["scientific_claim_qualified"] is False,
                 "completed development-only seed delivery required")
        self.files = {row["path"]: row for row in self.proof["files"]}
        _require(len(self.files) == len(self.proof["files"]), "duplicate delivery paths")
        self.checked = {}

    def path(self, name):
        _require(name in self.files, "delivery lacks consumed input: " + name)
        path = self.root / name
        if name not in self.checked:
            row = self.files[name]
            digest = sha256_file(path)
            _require(path.stat().st_size == row["bytes"] and digest == row["sha256"],
                     "delivered comparison input changed: " + name)
            self.checked[name] = digest
        return path

    def read(self, name):
        return _read(self.path(name))

    def binding(self):
        return dict(delivery_sha256=sha256_file(self.root / "delivery.json"),
                    consumed_files=self.checked)


def _matching_source(braid, population, braid_run, population_run):
    _require(braid["status"] == population["status"] == "EVALUATED_DEVELOPMENT"
             and braid["training_seed"] == population["training_seed"]
             and braid["score_rows"] == population["score_rows"] == 144
             and braid["origins"] == 48 and population["formal_test_accessed"] is False,
             "complete common development forecasts required")
    binding = braid["binding"]
    _require(binding["origins_ms"] == list(ORIGINS)
             and binding["horizons_ms"] == list(HORIZONS)
             and population["configuration"]["origins_ms"] == list(ORIGINS)
             and population["configuration"]["horizons_ms"] == list(HORIZONS)
             and population["configuration"]["joint_samples"] == 16,
             "matched origin, horizon and sixteen-draw protocol required")
    source = binding["cohort"]
    other = population_run["binding"]
    for name in ("acquisition_design_sha256", "acquisition_ledger_sha256", "dataset_sha256",
                 "episode_table_sha256", "episode_bindings"):
        _require(source[name] == other[name], "source cohort differs: " + name)
    for name, value in source["native_source_identity"].items():
        _require(other["native_source_identity"][name] == value, "native source identity differs")
    _require(source["acquisition_design_sha256"] == population["acquisition_design_sha256"]
             and source["dataset_sha256"] == population["dataset_sha256"],
             "evaluation and training source bindings differ")
    model = population_run["model_configuration"]
    _require(len(binding["source_unit_ids"]) == 100000
             and binding["source_unit_ids"] == model["unit_ids"]
             and binding["source_field_ids"] == model["field_names"]
             and model["bin_seconds"] == .005,
             "complete source unit identities, field identities and clock required")
    _require(braid_run["status"] == population_run["status"] == "FITTED_DEVELOPMENT"
             and braid_run["training_seed"] == population_run["training_seed"] == braid["training_seed"]
             and braid_run["stop_reason"] and population_run["stop_reason"]
             and braid_run["fit_episodes"] == population_run["fitting_episodes"] == 56
             and braid_run["validation_episodes"] == population_run["validation_episodes"] == 16,
             "completed matched episode training required")
    _require(other["config"]["fit_blocks"] == list(range(14))
             and other["config"]["validation_blocks"] == list(range(14, 18)),
             "unchanged whole-block development partition required")


def _population_method(run, ordinary):
    return PRIOR_METHODS[run["training_objective"]] if _is_prior_refit(run) else ordinary


def _population_prior_binding(cell, run):
    """Reuse the selected-parent, native-source and fixed-tensor contract."""
    if not _is_prior_refit(run):
        return None
    checkpoint = torch.load(cell.path("training/best.pt"), map_location="cpu", weights_only=True)
    _require(sha256_file(cell.path("training/best.pt")) == run["best_checkpoint_sha256"],
             "prior comparison selected checkpoint differs from training")
    _, _, lineage = _prior_parent_evidence(cell.root, run, checkpoint)
    return dict(method=_population_method(run, ""), training_objective=run["training_objective"],
                selection_metric=run["selection_metric"], prior_refit_lineage=lineage,
                **_prior_sampling_budget(run))


def _training_cost(cell, run, method):
    if run.get("continuation") is not None or _is_prior_refit(run):
        cost = _training_attempt_cost(cell.root, run)
        result = dict(seed=run["training_seed"], method=method, parameter_count=run["parameter_count"],
                    completed_epochs=run["completed_epochs"], best_epoch=run["best_epoch"],
                    stop_reason=run["stop_reason"],
                    recorded_training_seconds=cost["training_elapsed_seconds"],
                    training_attempts=cost["training_attempts"],
                    training_cost_scope=cost["training_cost_scope"],
                    peak_training_rss_bytes=run["peak_process_rss_bytes"],
                    hardware=json.dumps(run.get("hardware", {}), sort_keys=True))
        if _is_prior_refit(run):
            result.update(method=_population_method(run, method),
                training_objective=run["training_objective"], selection_metric=run["selection_metric"],
                **_prior_sampling_budget(run),
                parent_selected_epoch=run["refit_initialization"]["parent_selected_epoch"],
                parent_completed_epochs=run["refit_initialization"]["parent_completed_epochs"],
                **{key: value for key, value in cost.items()
                   if key not in ("training_elapsed_seconds", "training_attempts", "training_cost_scope")})
        return result
    attempts = run.get("attempt", 0)
    _require(type(attempts) is int and attempts >= 0, "nonnegative fit attempt required")
    seconds = _number(run["elapsed_seconds"])
    for previous in range(attempts):
        record = cell.read(f"training/attempts/attempt_{previous:03d}.json")
        _require(record["attempt"] == previous and record["training_seed"] == run["training_seed"]
                 and record["binding"] == run["binding"], "fit attempt source identity differs")
        seconds += _number(record["elapsed_seconds"])
    _require(seconds >= 0, "nonnegative recorded training cost required")
    return dict(seed=run["training_seed"], method=method, parameter_count=run["parameter_count"],
                completed_epochs=run["completed_epochs"], best_epoch=run["best_epoch"],
                stop_reason=run["stop_reason"], recorded_training_seconds=seconds,
                training_attempts=attempts + 1,
                training_cost_scope="recorded attempt wall times; interrupted unreported tails excluded",
                peak_training_rss_bytes=run["peak_process_rss_bytes"],
                hardware=json.dumps(run.get("hardware", {}), sort_keys=True))


def _block_means(rows):
    """Retain each block; average only its matched templates and origins."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["block"], row["horizon_ms"])].append(row)
    metrics = ("braid_unit_mean_mse", "population_unit_mean_mse_corrected",
               "population_minus_braid_unit_mean_mse", "braid_field_mean_mse",
               "population_field_mean_mse_corrected", "population_minus_braid_field_mean_mse")
    result = []
    for (seed, block, horizon), values in sorted(groups.items()):
        _require(len(values) == 12, "all four templates and three origins per block required")
        result.append(dict(seed=seed, block=block, horizon_ms=horizon, matched_conditions=12,
                           **{name: float(np.mean([row[name] for row in values])) for name in metrics}))
    _require(len(result) == 36, "complete three-seed/four-block/three-horizon means required")
    return result


def _population_cost_receipt(evaluation, run, scores):
    """Reuse measured-cost validation after the two delivered sources match."""
    if not any(name in evaluation for name in ("origin_costs", "episode_costs")):
        return {}
    binding = run["binding"]
    if _is_prior_refit(run):
        _prior_protocol(run)
    else:
        _require(binding["config"].get("training_sequence") == "complete_six_second_episode",
                 "measured costs require the bound complete six-second native protocol")
    _require(binding["native_source_identity"].get("dt_ms") == .1,
             "measured costs require the native event clock")
    # The bound score file preserves evaluator case order. The first case's
    # first configured origin is the full-unit retention point.
    cases = dict.fromkeys((block, case) for block, case, _, _ in scores)
    validation = [dict(block=block, case_id=case) for block, case in cases]
    _require(all(case["case_id"] in binding["episode_bindings"] for case in validation),
             "measured costs require every bound validation episode")
    model = run["model_configuration"]
    rows = []
    for (block, case, origin, horizon), raw in scores.items():
        rows.append(dict(block=block, case_id=case, origin_ms=origin, horizon_ms=horizon,
            **{name: (None if name == "peak_gpu_allocated_bytes" and raw.get(name) in (None, "")
                      else _number(raw[name])) for name in
               ("inference_seconds", "full_unit_generation_seconds", "field_projection_seconds", "peak_gpu_allocated_bytes")}))
    config = evaluation["configuration"]
    _require(config.get("field_metric_dtype") == "float64", "measured field metric arithmetic differs")
    measured = validate_evaluation_costs(evaluation, rows, config, validation=validation,
        unit_count=len(model["unit_ids"]), field_count=len(model["field_names"]),
        bin_ms=model["bin_seconds"] * 1000, source_dt_ms=binding["native_source_identity"]["dt_ms"],
        episode_bin_counts={case["case_id"]: 1200 for case in validation})
    return dict(measured_evaluation_costs=measured)


def _population_origin_cost_columns(receipt, case_id, origin_ms):
    """Preserve legacy nulls; charge new complete episode preparation once."""
    measured = _measured_origin_columns(receipt, case_id, origin_ms)
    columns = {"population_" + name: value for name, value in measured.items()}
    if receipt:
        columns["population_other_origin_costs"] = sum(measured[name] for name in
            ("scoring_seconds", "serialization_seconds", "hash_seconds", "csv_seconds"))
        columns["cost_scope"] = ("BRAID origin complete; population measured origin components; "
            "shared episode preparation charged at first configured origin; evaluator setup and excluded costs separate")
    else:
        columns["population_other_origin_costs"] = None
        columns["cost_scope"] = "BRAID origin complete; population legacy origin loading/scoring/archive unmeasured"
    return columns


def compare(braid_campaign, population_campaign, output):
    """Require all registered seeds before emitting the descriptive comparison."""
    braid_campaign, population_campaign, output = map(Path, (braid_campaign, population_campaign, output))
    if output.exists():
        raise FileExistsError(output)
    rows, costs, training, bindings, evaluation_costs = [], [], [], {}, []
    prior_bindings = {}
    for seed in SEEDS:
        braid = DeliveredCell(braid_campaign / f"seed_{seed}", seed)
        population = DeliveredCell(population_campaign / f"seed_{seed}", seed)
        b_eval = braid.read("forecasts/evaluation.json")
        p_eval = population.read("forecasts/evaluation.json")
        b_run, p_run = braid.read("training/run.json"), population.read("training/run.json")
        b_replay, p_replay = braid.read("forecasts/replay.json"), population.read("replay.json")
        _matching_source(b_eval, p_eval, b_run, p_run)
        candidate = _population_method(p_run, "population_" + p_run["representation"])
        prior_binding = _population_prior_binding(population, p_run)
        if prior_binding is not None:
            prior_bindings[str(seed)] = prior_binding
        for replay, cell in ((b_replay, braid), (p_replay, population)):
            _require(replay["status"] == "PASS" and replay["training_seed"] == seed
                     and replay["verified_origins"] == 48 and replay["verified_score_rows"] == 144
                     and replay["evaluation_sha256"] == sha256_file(cell.path("forecasts/evaluation.json")),
                     "complete independently replayed delivered forecasts required")
        _require(b_replay["all_origin_fields_boundary_and_metrics_exact"] is True
                 and p_replay["all_origin_fields_and_latents_exact"] is True
                 and p_replay["all_forecast_unit_metrics_replayed"] is True,
                 "mean fields and full-unit metrics must have been replayed")
        b_scores = _score_table(braid.path("forecasts/scores.csv"), seed, "braid_mean")
        p_scores = _score_table(population.path("forecasts/scores.csv"), seed, "population")
        _require(b_eval["artifacts"]["scores.csv"] == braid.files["forecasts/scores.csv"]["sha256"]
                 and b_eval["artifacts"]["origin_costs.csv"]
                 == braid.files["forecasts/origin_costs.csv"]["sha256"], "BRAID table binding differs")
        _require(b_eval["binding"]["training_committed_sha256"]
                 == sha256_file(braid.path("training/committed.json"))
                 and p_eval["checkpoint_sha256"] == p_run["best_checkpoint_sha256"],
                 "evaluation training checkpoint binding differs")
        committed = braid.read("training/committed.json")
        _require(committed["binding"]["dataset"] == b_eval["binding"]["cohort"]
                 and committed["binding"]["training_seed"] == seed,
                 "BRAID committed training source differs from evaluated source")
        _require(sha256_file(population.path("forecasts/scores.csv")) == p_eval["scores_sha256"],
                 "population evaluation score binding differs")
        cost_receipt = _population_cost_receipt(p_eval, p_run, p_scores)
        measured = cost_receipt.get("measured_evaluation_costs")
        evaluation_costs.append(dict(seed=seed, method=candidate,
            measured=measured is not None,
            **{name: measured["function_costs"][name] if measured else None for name in
               ("setup_seconds", "elapsed_seconds", "score_hash_seconds", "function_total_seconds", "peak_process_rss_bytes")},
            hardware=json.dumps(measured["hardware"], sort_keys=True) if measured else None,
            cost_scope=json.dumps(measured["cost_scope"], sort_keys=True) if measured else None))
        with np.load(population.path("forecasts/fitting_field_scale.npz"), allow_pickle=False) as scale:
            sd, units, field_ids = (scale[key].copy() for key in
                                   ("standard_deviation_hz", "unit_count", "field_ids"))
        _require(np.array_equal(np.bincount(p_run["model_configuration"]["unit_field_index"],
                                           minlength=len(units)), units)
                 and int(units.sum()) == 100000, "source unit-to-field denominators differ")
        _require(sha256_file(population.path("forecasts/fitting_field_scale.npz"))
                 == b_eval["binding"]["fitting_scale_sha256"], "fit-only field scale differs")
        cached = {}
        for key in sorted(b_scores):
            block, case, origin, horizon = key
            name = f"forecasts/{case}_origin_{origin:g}.npz"
            if name not in cached:
                _require(b_eval["artifacts"][Path(name).name] == braid.files[name]["sha256"]
                         and p_eval["artifacts"][Path(name).name] == population.files[name]["sha256"],
                         "forecast artifact binding differs")
                with np.load(braid.path(name), allow_pickle=False) as values:
                    means, truth = values["expected_field_counts"].copy(), values["observed_field_counts"].copy()
                    _require(np.array_equal(values["field_ids"], field_ids)
                             and np.array_equal(values["unit_count"], units), "BRAID field map differs")
                with np.load(population.path(name), allow_pickle=False) as values:
                    samples = values["sampled_field_counts"].copy()
                    _require(np.array_equal(values["observed_field_counts"], truth), "source truth differs")
                _require(means.shape == truth.shape == (100, len(units))
                         and samples.shape == (16, 100, len(units))
                         and np.isfinite(means).all() and np.all(means >= 0)
                         and np.isfinite(truth).all() and np.all(truth >= 0)
                         and np.all(truth == np.rint(truth)), "complete full-field forecasts required")
                cached[name] = means, truth, samples
            means, truth, samples = cached[name]
            bins = round(horizon / 5.)
            b_field = float(np.mean(((means[:bins] - truth[:bins]) / (units * .005 * sd))**2))
            _require(np.isclose(b_field, _number(b_scores[key]["fitting_standardized_field_rate_mse"]),
                                rtol=1e-12, atol=1e-14), "BRAID saved field MSE differs")
            p_field = sampled_mean_field_mse(samples[:, :bins], truth[:bins].astype(np.int64), sd, units, 5.)
            b_unit = _number(b_scores[key]["unit_bin_mean_mse"])
            p_unit = _number(p_scores[key]["unit_bin_mean_mse_corrected"])
            rows.append(dict(seed=seed, block=block, case_id=case, origin_ms=origin, horizon_ms=horizon,
                **({"candidate_method": candidate} if prior_binding is not None else {}),
                braid_unit_mean_mse=b_unit, population_unit_mean_mse_corrected=p_unit,
                population_minus_braid_unit_mean_mse=p_unit - b_unit,
                braid_field_mean_mse=b_field, population_field_mean_mse_corrected=p_field,
                population_minus_braid_field_mean_mse=p_field - b_field))
        with braid.path("forecasts/origin_costs.csv").open(newline="") as stream:
            b_costs = list(csv.DictReader(stream))
        expected_origins = {(key[0], key[1], key[2]) for key in b_scores}
        seen = set()
        for row in b_costs:
            key = (int(row["block"]), row["case_id"], _number(row["origin_ms"]))
            _require(key in expected_origins and key not in seen, "duplicate or foreign origin cost")
            seen.add(key)
            p_row = p_scores[(*key, 500.)]
            _require(int(row["generated_unit_mean_values"]) == 10000000,
                     "complete 100k-unit/100-bin mean output cost required")
            costs.append(dict(seed=seed, block=key[0], case_id=key[1], origin_ms=key[2],
                braid_mean_values=int(row["generated_unit_mean_values"]), population_count_values=160000000,
                braid_input_loading_seconds=_number(row["input_loading_seconds"]),
                braid_inference_seconds=_number(row["inference_seconds"]),
                braid_full_unit_mean_seconds=_number(row["full_unit_mean_forecast_seconds"]),
                braid_projection_and_scoring_seconds=_number(row["field_projection_and_scoring_seconds"]),
                braid_serialization_and_hash_seconds=_number(row["serialization_and_hash_seconds"]),
                braid_origin_total_seconds=_number(row["origin_total_seconds"]),
                population_inference_seconds=_number(p_row["inference_seconds"]),
                population_full_unit_count_seconds=_number(p_row["full_unit_generation_seconds"]),
                population_projection_seconds=_number(p_row["field_projection_seconds"]),
                output_scope="100k means versus 16 draws of 100k counts; event materialization excluded",
                **_population_origin_cost_columns(cost_receipt, key[1], key[2])))
        _require(seen == expected_origins and len(b_costs) == 48, "complete 48-origin costs required")
        training.extend((_training_cost(braid, b_run, "braid_mean"),
                         _training_cost(population, p_run, candidate)))
        bindings[str(seed)] = dict(braid=braid.binding(), population=population.binding())
    _require(not prior_bindings or (set(prior_bindings) == {str(seed) for seed in SEEDS}
             and len({binding["training_objective"] for binding in prior_bindings.values()}) == 1),
             "all three population seeds must use the same prior objective")
    output.mkdir(parents=True)
    artifacts = {}
    for name, values in (("mean_comparison_all_seeds.csv", rows), ("costs_by_origin.csv", costs),
                         ("mean_comparison_by_block.csv", _block_means(rows)),
                         ("training_costs.csv", training), ("population_evaluation_costs.csv", evaluation_costs)):
        with (output / name).open("w", newline="") as stream:
            columns = list(dict.fromkeys(key for value in values for key in value))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(values)
        artifacts[name] = sha256_file(output / name)
    result = dict(status="COMPLETE_DESCRIPTIVE_BRAID_POPULATION_MEAN_COMPARISON", training_seeds=list(SEEDS),
        score_rows=len(rows), origin_cost_rows=len(costs), independent_blocks=4, source_unit_count=100000,
        descriptive_block_rows=36,
        consumed_deliveries=bindings, artifacts=artifacts, comparison_implementation_sha256=sha256_file(Path(__file__)),
        samples_regenerated=False, training_repeated=False, source_simulation_repeated=False,
        inferential_statistics_performed=False, distribution_comparison_performed=False,
        scientific_claim_qualified=False, speedup_qualified=False,
        unit_metric_arithmetic="BRAID float64 deterministic MSE; population exact integer U-statistic then float32",
        field_metric_arithmetic="float64; population mean error corrected for sixteen-draw sampling variance",
        selection_blocks_are_independent_test=False, full_end_to_end_cost_matched=False)
    if prior_bindings:
        result.update(candidate_method=prior_bindings[str(SEEDS[0])]["method"],
                      population_training_bindings=prior_bindings, inherited_population_training_cost_included=True)
    result["population_measured_evaluation_cost_seeds"] = [row["seed"] for row in evaluation_costs if row["measured"]]
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("braid-campaign", "population-campaign", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.braid_campaign, args.population_campaign, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("consumed_deliveries", "artifacts")}))


if __name__ == "__main__":
    main()

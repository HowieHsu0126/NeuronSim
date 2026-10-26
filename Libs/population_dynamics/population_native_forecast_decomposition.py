"""Describe frozen native and LDNS forecast scores without fitting or generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .braid_population_comparison import DeliveredCell
from .lif_neural_field import sampled_mean_field_mse
from .population_forecast_replay import METRIC_ATOL, METRIC_RTOL
from .population_residual_correction_analysis import (
    BIN_MS, BLOCKS, HORIZONS_MS, METHODS, SEEDS, _accepted_ldns, _require, admit_comparison_cells,
)
from .provenance import cpu_execution_provenance, freeze_json, sha256_file
from .trajectory_forecast_analysis import _write_rows
from .trajectory_scores import energy_score_terms

MODELS = ("ldns", "current", "history")
COMPONENTS = (
    "energy_draw_to_observation", "energy_subtracted_draw_to_draw", "energy_reconstruction", "energy_score",
    "field_mean_mse_uncorrected", "field_mean_mse_corrected", "standardized_generated_variance",
    "saved_energy_score", "energy_minus_saved",
)


def field_components(samples, observed, mean_hz, standard_deviation_hz, unit_count):
    """Use the evaluator's Hz normalization and existing finite-draw mean MSE."""
    corrected = sampled_mean_field_mse(samples, observed, standard_deviation_hz, unit_count, BIN_MS)
    rate_denominator = unit_count * BIN_MS / 1000
    normalized = (samples / rate_denominator - mean_hz) / standard_deviation_hz
    truth = (observed / rate_denominator - mean_hz) / standard_deviation_hz
    terms = energy_score_terms(normalized, truth)
    # This order is the existing sampled_mean_field_mse arithmetic. Variance
    # is unchanged by subtracting the fixed observed field from every draw.
    residual = (samples.astype(np.float64) - observed) / (unit_count * (BIN_MS / 1000) * standard_deviation_hz)
    result = dict(energy_draw_to_observation=terms["draw_to_observation"],
        energy_subtracted_draw_to_draw=terms["subtracted_draw_to_draw"],
        energy_reconstruction=terms["draw_to_observation"] - terms["subtracted_draw_to_draw"],
        energy_score=terms["energy_score"], field_mean_mse_uncorrected=float(np.mean(residual.mean(0) ** 2)),
        field_mean_mse_corrected=corrected, standardized_generated_variance=float(residual.var(0, ddof=1).mean()))
    _require(np.isfinite(list(result.values())).all(), "finite forecast decomposition required")
    return result


def _scale(path, reference):
    digest = sha256_file(path)
    for seed in SEEDS:
        binding = reference["field_source_link"]["bindings"][str(seed)]["old_population_consumed_files"]
        _require(digest == binding["forecasts/fitting_field_scale.npz"], "decomposition fitting scale binding differs")
    with np.load(path, allow_pickle=False) as values:
        mean, sd, units, ids = (values[name].copy() for name in
                              ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids"))
    _require(mean.shape == sd.shape == units.shape == ids.shape == (128,)
             and units.dtype.kind in "iu" and units.sum() == 100000 and np.all(units > 0)
             and len(np.unique(ids)) == 128 and np.isfinite(mean).all() and np.all(mean >= 0)
             and np.isfinite(sd).all() and np.all(sd > 0), "complete positive original fitting scale required")
    return mean, sd, units, ids, digest


def _forecast(cell, relative, expected_digest, *, with_map=False):
    path = cell.path(relative)
    _require(cell.files[relative]["sha256"] == expected_digest, "forecast array differs from accepted score binding")
    names = ["sampled_field_counts", "observed_field_counts"]
    if with_map:
        names += ["field_ids", "unit_ids"]
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in names}
    samples, truth = arrays["sampled_field_counts"], arrays["observed_field_counts"]
    _require(samples.shape == (16, 100, 128) and truth.shape == (100, 128)
             and samples.dtype.kind in "iu" and truth.dtype.kind in "iu"
             and np.all(samples >= 0) and np.all(truth >= 0), "complete sixteen-draw integer field arrays required")
    return arrays


def _aggregate(rows, names, expected_conditions):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in names)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        _require(len(values) == expected_conditions, "complete decomposition aggregation group required")
        result.append(dict(zip(names, key), conditions=len(values),
            **{metric: float(np.mean([row[metric] for row in values])) for metric in COMPONENTS}))
    return result


def analyze(corrections, parent_campaign, ldns_comparison, field_comparison, ldns_forecasts, field_scale, output,
            *, execution=None):
    output, corrections, ldns_forecasts, field_scale = map(Path, (output, corrections, ldns_forecasts, field_scale))
    if output.exists():
        raise FileExistsError("decomposition requires a fresh output directory")
    cells, source, family = admit_comparison_cells(corrections, parent_campaign)
    baseline, _, reference = _accepted_ldns(ldns_comparison, field_comparison, cells, source, family)
    accepted_ldns = json.loads((Path(ldns_comparison) / "analysis/comparison.json").read_text())
    mean, sd, units, field_ids, scale_digest = _scale(field_scale, reference)
    rows, consumed, expected_units = [], {}, None
    for seed in SEEDS:
        native = DeliveredCell(corrections / f"seed_{seed}", seed)
        ldns = DeliveredCell(ldns_forecasts / f"seed_{seed}", seed)
        binding = accepted_ldns["bindings"][str(seed)]["ldns"]
        _require(sha256_file(ldns.root / "delivery.json") == binding["delivery_sha256"], "LDNS forecast delivery differs")
        _require(sha256_file(native.root / "delivery.json") == cells[(seed, "current")]["consumed"]["delivery_sha256"]
                 == cells[(seed, "history")]["consumed"]["delivery_sha256"], "native forecast delivery differs")
        origins = sorted(key for key in baseline if key[0] == seed and key[-1] == HORIZONS_MS[0])
        _require(len(origins) == 48, "complete 48-origin decomposition seed required")
        for _, case, origin, _ in origins:
            filename = f"{case}_origin_{origin:g}.npz"
            relative = "forecasts/" + filename
            saved = _forecast(ldns, relative, binding["consumed_files"].get(relative), with_map=True)
            unit_ids = saved["unit_ids"]
            _require(np.array_equal(saved["field_ids"], field_ids) and unit_ids.shape == (100000,)
                     and unit_ids.dtype.kind in "iu", "LDNS forecast field or unit map differs")
            if expected_units is None:
                _require(len(np.unique(unit_ids)) == 100000, "complete unique source unit identities required")
                expected_units = unit_ids.copy()
            _require(np.array_equal(unit_ids, expected_units), "LDNS forecast unit identities differ across origins or seeds")
            truth = saved["observed_field_counts"]
            for model in MODELS:
                if model == "ldns":
                    arrays, scores, method = saved, baseline, "ldns"
                else:
                    relative = f"results/{model}/forecasts/" + filename
                    cell = cells[(seed, model)]
                    arrays = _forecast(native, relative, cell["evaluation"]["artifacts"].get(filename))
                    _require(np.array_equal(arrays["observed_field_counts"], truth), "native and LDNS source truth differ")
                    scores, method = cell["rows"], METHODS[model]
                for horizon in HORIZONS_MS:
                    key = (seed, case, origin, horizon)
                    bins = int(horizon / BIN_MS)
                    components = field_components(arrays["sampled_field_counts"][:, :bins], truth[:bins], mean, sd, units)
                    reported = scores[key]["energy_score"]
                    difference = components["energy_score"] - reported
                    allowed = METRIC_ATOL + METRIC_RTOL * abs(reported)
                    _require(abs(difference) <= allowed and abs(components["energy_reconstruction"] - reported) <= allowed,
                             f"saved energy score drift: {model} seed {seed} {case} origin {origin:g} horizon {horizon:g}")
                    observed_total = float(truth[:bins].sum(dtype=np.int64)) / 100000
                    reported_total = scores[key]["observed_spikes_per_unit"]
                    _require(abs(observed_total - reported_total) <= METRIC_ATOL + METRIC_RTOL * abs(reported_total),
                             "saved observed total differs from the complete source truth")
                    rows.append(dict(model=model, method=method, seed=seed, block=scores[key]["block"], case_id=case,
                        origin_ms=origin, horizon_ms=horizon, **components,
                        saved_energy_score=reported, energy_minus_saved=difference))
        consumed[str(seed)] = dict(ldns=ldns.binding(), corrections=native.binding())
    expected = {(model, *key) for model in MODELS for key in baseline}
    actual = {(row["model"], row["seed"], row["case_id"], row["origin_ms"], row["horizon_ms"]) for row in rows}
    _require(len(rows) == 1296 and actual == expected, "complete 1296-condition decomposition required")
    blocks = _aggregate(rows, ("model", "seed", "block", "horizon_ms"), 12)
    summaries = _aggregate(rows, ("model", "seed", "horizon_ms"), 48)
    _require(len(blocks) == 108 and len(summaries) == 27, "complete source-block and seed-horizon decomposition required")
    tables = {"forecast_decomposition_all_seeds.csv": rows, "forecast_decomposition_by_block.csv": blocks,
              "forecast_decomposition_by_seed_horizon.csv": summaries}
    output.mkdir(parents=True)
    for name, values in tables.items():
        _write_rows(output / name, values)
    report = dict(status="COMPLETE_DESCRIPTIVE_NATIVE_FORECAST_DECOMPOSITION", training_seeds=list(SEEDS), models=list(MODELS),
        score_rows=len(rows), block_rows=len(blocks), seed_horizon_rows=len(summaries),
        source=source, correction_fit=family, source_blocks=list(BLOCKS), source_circuits=1,
        draws=16, source_units=100000, fields=128, bin_ms=BIN_MS, ldns_reference=reference, consumed=consumed,
        field_scale=dict(path=str(field_scale), sha256=scale_digest),
        normalization="(field counts / (unit_count * bin_ms / 1000) - original fitting mean_hz) / standard_deviation_hz",
        component_definitions=dict(
            energy_draw_to_observation="mean draw-to-observation norm divided by sqrt(time-field coordinates)",
            energy_subtracted_draw_to_draw="sum of unordered distinct-draw norms / (16 * 15 * sqrt(time-field coordinates))",
            energy_reconstruction="energy_draw_to_observation minus energy_subtracted_draw_to_draw",
            field_mean_mse_uncorrected="mean squared standardized ensemble-mean residual",
            field_mean_mse_corrected="existing sampled_mean_field_mse; subtract unbiased draw variance / 16 before averaging",
            standardized_generated_variance="mean unbiased draw variance in original fitting-standardized Hz fields"),
        interpretation="Descriptive score components; not orthogonal causal or error-budget attributions.",
        negative_corrected_estimates_retained=True, exact_source_truth_match=True,
        energy_parity=dict(verified_rows=len(rows), rtol=METRIC_RTOL, atol=METRIC_ATOL,
                           maximum_absolute_difference=max(abs(row["energy_minus_saved"]) for row in rows),
                           maximum_reconstruction_absolute_difference=max(
                               abs(row["energy_reconstruction"] - row["saved_energy_score"]) for row in rows)),
        training_performed=False, generation_performed=False, full_unit_samples_read=False,
        formal_evaluation=False, inferential_statistics_performed=False, scientific_claim_qualified=False, sota_qualified=False,
        artifacts={name: dict(sha256=sha256_file(output / name), rows=len(values)) for name, values in tables.items()})
    if execution is not None:
        report["execution"] = execution
    freeze_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corrections", "parent-campaign", "ldns-comparison", "field-comparison", "ldns-forecasts", "field-scale", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.corrections, args.parent_campaign, args.ldns_comparison, args.field_comparison,
                     args.ldns_forecasts, args.field_scale, args.output, execution=cpu_execution_provenance())
    print(json.dumps({name: report[name] for name in ("status", "score_rows", "block_rows", "seed_horizon_rows")}))


if __name__ == "__main__":
    main()

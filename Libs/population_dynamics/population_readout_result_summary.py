"""Verify a complete native readout diagnostic and describe paired differences.

This reader uses only the standard library. It neither fits parameters nor
generates samples. The cached posterior is a diagnostic reference; hazard
moments do not decompose the renewed refractory process's count error.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys


SEEDS = (41, 42, 43)
MODELS = ("original", "current", "history")
BLOCKS = (14, 15, 16, 17)
TEMPLATES = tuple(range(4))
ORIGINS_MS = (1000, 3000, 5000)
HORIZONS_MS = (20, 80, 500)
BIN_MS, FUTURE_BINS, DRAWS, SOURCE_UNITS = 5, 100, 16, 100000
ARTIFACT_ROWS = {
    "time_moments.csv": 43200,
    "origin_horizon_moments.csv": 1296,
    "seed_horizon_summary.csv": 1512,
    "source_block_summary.csv": 6048,
}
IDENTITY_FIELDS = ("model", "seed", "block", "case_id", "template_id", "origin_ms")
MOMENT_NAMES = (
    "standardized_latent_mean_mse",
    "standardized_latent_ensemble_variance_mean",
    "standardized_latent_lag_one_covariance_mean",
    "predicted_mean_log_hazard_unit_mean",
    "posterior_mean_log_hazard_unit_mean",
    "empirical_log_expected_hazard_unit_mean",
    "posterior_log_expected_hazard_unit_mean",
    "empirical_log_hazard_variance_unit_mean",
    "empirical_jensen_gap_unit_mean",
    "posterior_lognormal_adjustment_unit_mean",
    "mean_displacement_unit_mean",
    "mean_displacement_unit_mean_square",
    "stochastic_gap_displacement_unit_mean",
    "stochastic_gap_displacement_unit_mean_square",
    "total_log_expected_hazard_gap_unit_mean",
    "total_log_expected_hazard_gap_unit_mean_square",
    "decomposition_cross_term_unit_mean",
    "decomposition_closure_max_absolute",
    "decomposition_mean_square_closure",
)
KEY_MOMENT_NAMES = MOMENT_NAMES[:3] + (
    "mean_displacement_unit_mean_square",
    "stochastic_gap_displacement_unit_mean_square",
    "total_log_expected_hazard_gap_unit_mean_square",
    "decomposition_cross_term_unit_mean",
)
NATIVE_NAMES = (
    "energy_score", "variogram_spatial", "variogram_cross_population",
    "variogram_temporal_lag_1", "variogram_temporal_lag_4", "variogram_temporal_lag_16",
    "unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
    "unit_total_coverage_90", "unit_total_interval_width_90",
    "refractory_count_violation_1bin", "refractory_count_violation_2bin", "refractory_count_violation_4bin",
    "inference_seconds", "full_unit_generation_seconds", "field_projection_seconds", "peak_gpu_allocated_bytes",
)
KEY_NATIVE_NAMES = (
    "energy_score", "unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit",
    "observed_spikes_per_unit", "unit_total_coverage_90", "unit_total_interval_width_90",
    "refractory_count_violation_1bin", "refractory_count_violation_2bin", "refractory_count_violation_4bin",
)
PAIRS = (("current", "original"), ("history", "original"), ("history", "current"))
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
NA_VALUES = {"", "NA", "N/A"}
# The diagnostic sums over source units, draws and 512 latent coordinates.
# gamma(n) bounds floating-point accumulation; this is not a fidelity tolerance.
READOUT_OPERATIONS = 2 * SOURCE_UNITS + DRAWS + 512 + FUTURE_BINS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: object, role: str) -> None:
    _require(isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None,
             "invalid SHA-256 identity: " + role)


def _integer(value: str, role: str) -> int:
    number = _number(value, role)
    _require(number is not None and number.is_integer(), "integer required: " + role)
    return int(number)


def _number(value: str, role: str, *, optional: bool = False) -> float | None:
    if optional and value in NA_VALUES:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("finite number required: " + role) from error
    _require(math.isfinite(number), "finite number required: " + role)
    return number


def _close(actual: float, expected: float, terms: list[float] | tuple[float, ...],
           operations: int, role: str) -> None:
    epsilon = sys.float_info.epsilon
    gamma = operations * epsilon / (1 - operations * epsilon)
    bound = gamma * math.fsum(abs(term) for term in terms)
    bound += math.ulp(actual) + math.ulp(expected)
    _require(abs(actual - expected) <= bound, "numerical mismatch: " + role)


def _rows(path: Path, expected_fields: set[str] | None = None):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        _require(fields is not None and len(fields) == len(set(fields)),
                 "missing or duplicate CSV columns: " + path.name)
        if expected_fields is not None:
            _require(set(fields) == expected_fields, "CSV columns differ: " + path.name)
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()),
                     "malformed CSV row: " + path.name)
            yield row


def _identity(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    model = row["model"]
    template_label = row["template_id"]
    _require(template_label in {f"familiar_{template}" for template in TEMPLATES},
             "canonical familiar_N template identity required")
    template = int(template_label[-1])
    seed, block, origin = (
        _integer(row[name], name) for name in ("seed", "block", "origin_ms"))
    _require(model in MODELS and seed in SEEDS and block in BLOCKS
             and template in TEMPLATES and origin in ORIGINS_MS, "unexpected origin identity")
    _require(row["case_id"] == f"train_block_{block:03d}_familiar_{template}",
             "case/template/block identity differs")
    return model, seed, block, template, origin


def _moment_columns() -> list[str]:
    return [("prefix_max_" if name == "decomposition_closure_max_absolute" else "prefix_mean_") + name
            for name in MOMENT_NAMES] + ["endpoint_" + name for name in MOMENT_NAMES]


def _check_moments(values: dict[str, float], role: str) -> None:
    for name in (MOMENT_NAMES[0], MOMENT_NAMES[1], "empirical_log_hazard_variance_unit_mean",
                 "posterior_lognormal_adjustment_unit_mean", "mean_displacement_unit_mean_square",
                 "stochastic_gap_displacement_unit_mean_square",
                 "total_log_expected_hazard_gap_unit_mean_square", "decomposition_closure_max_absolute"):
        _require(values[name] >= 0, "negative nonnegative moment: " + role + "/" + name)
    a = values["mean_displacement_unit_mean"]
    b = values["stochastic_gap_displacement_unit_mean"]
    c = values["total_log_expected_hazard_gap_unit_mean"]
    for actual, terms, label in (
        (a, (values["predicted_mean_log_hazard_unit_mean"],
             -values["posterior_mean_log_hazard_unit_mean"]), "mean displacement"),
        (b, (values["empirical_jensen_gap_unit_mean"],
             -values["posterior_lognormal_adjustment_unit_mean"]), "stochastic displacement"),
        (c, (values["empirical_log_expected_hazard_unit_mean"],
             -values["posterior_log_expected_hazard_unit_mean"]), "total displacement"),
        (c, (a, b), "signed closure"),
    ):
        _close(actual, math.fsum(terms), (*terms, actual), READOUT_OPERATIONS, role + "/" + label)
    squares = (values["mean_displacement_unit_mean_square"],
               values["stochastic_gap_displacement_unit_mean_square"],
               values["decomposition_cross_term_unit_mean"])
    total = values["total_log_expected_hazard_gap_unit_mean_square"]
    _close(total, math.fsum(squares), (*squares, total), READOUT_OPERATIONS,
           role + "/mean-square closure")
    residual = values["decomposition_mean_square_closure"]
    _close(residual, total - math.fsum(squares), (*squares, total), READOUT_OPERATIONS,
           role + "/reported mean-square closure")


def _metadata(analysis: dict) -> None:
    expected = {
        "status": "COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC",
        "training_seeds": list(SEEDS), "models": list(MODELS), "source_blocks": list(BLOCKS),
        "horizon_ms": list(HORIZONS_MS), "origins_per_model_seed": 48,
        "retained_draws": DRAWS, "future_bins": FUTURE_BINS, "source_units": SOURCE_UNITS,
        "coordinate_scale_fitting_blocks": list(range(12)), "native_comparison_reused": True,
    }
    _require(all(analysis.get(key) == value for key, value in expected.items()),
             "complete frozen diagnostic dimensions or source metadata differ")
    for name in ("trajectory_sampling_performed", "parameter_fitting_performed",
                 "native_generation_performed", "new_encoding_performed",
                 "qualification_or_native_scores_recomputed", "significance_tests_computed",
                 "scientific_claim_qualified", "sota_qualified"):
        _require(analysis.get(name) is False, "unexpected diagnostic action or qualification: " + name)
    runtime = analysis.get("runtime", {})
    environment = runtime.get("environment", {})
    _require(runtime.get("execution_backend") == "merlin"
             and isinstance(runtime.get("hostname"), str) and runtime["hostname"]
             and runtime["hostname"].split(".")[0] != "mgmt2"
             and all(isinstance(environment.get(key), str) and environment[key]
                     for key in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")),
             "actual Merlin diagnostic runtime identity required")
    _hash(analysis.get("input_manifest_sha256"), "input manifest")
    consumed = analysis.get("consumed_files", {})
    expected_cells = {f"{seed}/{model}" for seed in SEEDS for model in MODELS}
    _require(set(consumed) == expected_cells, "nine consumed model/seed identities required")
    for seed, model in itertools.product(SEEDS, MODELS):
        prefix = "forecasts/" if model == "original" else f"results/{model}/forecasts/"
        files = {prefix + f"train_block_{block:03d}_familiar_{template}_origin_{origin}.npz"
                 for block, template, origin in itertools.product(BLOCKS, TEMPLATES, ORIGINS_MS)}
        if model == "original":
            files.add("training/best.pt")
            files.update(f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt"
                         for block, template in itertools.product(BLOCKS, TEMPLATES))
        else:
            files.add("provenance/fit.json")
        cell = consumed[f"{seed}/{model}"]
        _require(isinstance(cell, dict) and set(cell) == files,
                 "complete retained trajectories and posterior-cache identities required")
        for name, digest in cell.items():
            _hash(digest, name)
    deliveries = analysis.get("consumed_deliveries", [])
    _require(isinstance(deliveries, list) and len(deliveries) == 9, "nine consumed deliveries required")
    seen = set()
    for row in deliveries:
        key = f"{row.get('seed')}/{row.get('model')}"
        _require(key in expected_cells and key not in seen, "duplicate or unknown consumed delivery")
        seen.add(key)
        _hash(row.get("delivery_sha256"), "consumed delivery")
        _hash(row.get("parent_checkpoint_sha256"), "parent checkpoint")
        _require(row["parent_checkpoint_sha256"] == consumed[f"{row['seed']}/original"]["training/best.pt"],
                 "consumed parent checkpoint identity differs")


def _time_table(directory: Path) -> tuple[dict, float]:
    fields = set(IDENTITY_FIELDS) | {"elapsed_ms", "absolute_bin_end_ms"} | set(MOMENT_NAMES)
    groups = defaultdict(dict)
    count, maximum_residual = 0, 0.0
    for row in _rows(directory / "time_moments.csv", fields):
        key = _identity(row)
        elapsed = _integer(row["elapsed_ms"], "elapsed_ms")
        _require(elapsed % BIN_MS == 0 and BIN_MS <= elapsed <= FUTURE_BINS * BIN_MS,
                 "unexpected retained time bin")
        _require(_integer(row["absolute_bin_end_ms"], "absolute_bin_end_ms") == key[-1] + elapsed,
                 "absolute and relative time identities differ")
        step = elapsed // BIN_MS
        _require(step not in groups[key], "duplicate retained time bin")
        values = {name: _number(row[name], name) for name in MOMENT_NAMES}
        _check_moments(values, "time")
        groups[key][step] = tuple(values[name] for name in MOMENT_NAMES)
        maximum_residual = max(maximum_residual, abs(values["decomposition_mean_square_closure"]))
        count += 1
    expected = set(itertools.product(MODELS, SEEDS, BLOCKS, TEMPLATES, ORIGINS_MS))
    _require(count == ARTIFACT_ROWS["time_moments.csv"] and set(groups) == expected
             and all(set(values) == set(range(1, FUTURE_BINS + 1)) for values in groups.values()),
             "complete retained-trajectory denominator required")
    return groups, maximum_residual


def _origin_table(directory: Path, times: dict) -> tuple[dict, list[str], dict]:
    path = directory / "origin_horizon_moments.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        fields = next(csv.reader(stream))
    native = [name for name in fields if name.startswith("native_")
              and name not in {"native_method", "native_hardware_device", "native_ensemble_artifact"}]
    _require(len(native) == 18 and set(native) == {"native_" + name for name in NATIVE_NAMES},
             "complete native metric columns required")
    metrics = _moment_columns() + native
    expected_fields = set(IDENTITY_FIELDS) | set(metrics) | {
        "horizon_ms", "prefix_bins", "native_method", "native_hardware_device", "native_ensemble_artifact"}
    rows, native_identities = {}, {}
    for row in _rows(path, expected_fields):
        identity = _identity(row)
        horizon = _integer(row["horizon_ms"], "horizon_ms")
        _require(horizon in HORIZONS_MS and _integer(row["prefix_bins"], "prefix_bins") == horizon // BIN_MS,
                 "native horizon/prefix identity differs")
        key = identity + (horizon,)
        _require(key not in rows, "duplicate origin/horizon")
        native_identity = tuple(row[name] for name in (
            "native_method", "native_hardware_device", "native_ensemble_artifact"))
        _require(all(native_identity), "complete native method/device/artifact identity required")
        _require(native_identity[2] == f"train_block_{identity[2]:03d}_familiar_{identity[3]}_origin_{identity[4]}.npz",
                 "native ensemble artifact identity differs")
        if identity in native_identities:
            _require(native_identities[identity] == native_identity, "native identity changed across horizons")
        native_identities[identity] = native_identity
        values = {name: _number(row[name], name, optional=name in native) for name in metrics}
        for prefix in ("prefix_mean_", "endpoint_"):
            moments = {name: values[("prefix_max_" if name == "decomposition_closure_max_absolute"
                                    and prefix == "prefix_mean_" else prefix) + name] for name in MOMENT_NAMES}
            _check_moments(moments, "origin/" + prefix)
        steps = horizon // BIN_MS
        for index, name in enumerate(MOMENT_NAMES):
            selected = [times[identity][step][index] for step in range(1, steps + 1)]
            prefix = "prefix_max_" if name == "decomposition_closure_max_absolute" else "prefix_mean_"
            expected = max(selected) if prefix == "prefix_max_" else math.fsum(selected) / steps
            _close(values[prefix + name], expected, selected, steps + 2, "origin prefix/" + name)
            _close(values["endpoint_" + name], selected[-1], (selected[-1],), 1, "origin endpoint/" + name)
        rows[key] = values
    expected = {identity + (horizon,) for identity in times for horizon in HORIZONS_MS}
    _require(len(rows) == ARTIFACT_ROWS[path.name] and set(rows) == expected,
             "complete origin/horizon denominator required")
    for model in MODELS:
        identities = {value[:2] for key, value in native_identities.items() if key[0] == model}
        _require(len(identities) == 1, "native method or device differs within model")
    return rows, metrics, {model: list(next(value[:2] for key, value in native_identities.items()
                                          if key[0] == model)) for model in MODELS}


def _aggregate_table(directory: Path, origins: dict, metrics: list[str], *, by_block: bool) -> dict:
    name = "source_block_summary.csv" if by_block else "seed_horizon_summary.csv"
    fields = {"model", "seed", "horizon_ms", "metric", "statistic", "value", "available_origins",
              "structural_na_origins", "total_origins", "metric_scope"}
    if by_block:
        fields.add("block")
    groups = defaultdict(list)
    for identity, values in origins.items():
        model, seed, block, _, _, horizon = identity
        group = (model, seed, horizon, block) if by_block else (model, seed, horizon)
        groups[group].append(values)
    result, seen = defaultdict(dict), set()
    for row in _rows(directory / name, fields):
        group = (row["model"], _integer(row["seed"], "seed"), _integer(row["horizon_ms"], "horizon_ms"))
        if by_block:
            group += (_integer(row["block"], "block"),)
        metric = row["metric"]
        _require(group in groups and metric in metrics, "unknown aggregate identity or metric")
        key = group + (metric,)
        _require(key not in seen, "duplicate aggregate metric")
        seen.add(key)
        selected = [values[metric] for values in groups[group] if values[metric] is not None]
        available, total = len(selected), len(groups[group])
        _require(total == (12 if by_block else 48)
                 and _integer(row["total_origins"], "total_origins") == total
                 and _integer(row["available_origins"], "available_origins") == available
                 and _integer(row["structural_na_origins"], "structural_na_origins") == total - available,
                 "aggregate availability or complete denominator differs")
        maximum = metric.endswith("decomposition_closure_max_absolute")
        _require(row["statistic"] == ("maximum" if maximum else "mean"), "aggregate statistic differs")
        scope = ("recorded_native_horizon_metric" if metric.startswith("native_") else
                 "instantaneous_endpoint" if metric.startswith("endpoint_") else "prefix_moment_summary")
        _require(row["metric_scope"] == scope, "aggregate metric scope differs")
        actual = _number(row["value"], name + "/" + metric, optional=True)
        if not selected:
            _require(actual is None, "structural N/A must remain N/A")
        else:
            _require(actual is not None, "available aggregate cannot be N/A")
            expected = max(selected) if maximum else math.fsum(selected) / available
            _close(actual, expected, selected, available + 2, name + "/" + metric)
        result[group][metric] = (actual, available)
    expected = {group + (metric,) for group in groups for metric in metrics}
    _require(len(seen) == ARTIFACT_ROWS[name] and seen == expected,
             "complete aggregate metric denominator required")
    return dict(result)


def _compact_rows(table: dict, keys: list[tuple], metrics: list[str], denominator: int) -> dict:
    rows, partial = [], []
    for row_index, key in enumerate(keys):
        rows.append([*key, *(table[key][metric][0] for metric in metrics)])
        for metric_index, metric in enumerate(metrics):
            available = table[key][metric][1]
            if available not in (0, denominator):
                partial.append([row_index, metric_index, available])
    return {"rows": rows, "total_origins": denominator, "partial_availability": partial}


def _paired(origins: dict, metrics: list[str], *, by_block: bool) -> tuple[dict, list[tuple]]:
    table, keys = {}, []
    for pair_index, (candidate, reference) in enumerate(PAIRS):
        for seed in SEEDS:
            for horizon in ((500,) if by_block else HORIZONS_MS):
                for block in (BLOCKS if by_block else (None,)):
                    key = (pair_index, seed, horizon, block) if by_block else (pair_index, seed, horizon)
                    identities = [(seed, selected_block, template, origin, horizon)
                                  for selected_block, template, origin in itertools.product(
                                      (block,) if by_block else BLOCKS, TEMPLATES, ORIGINS_MS)]
                    values = {}
                    for metric in metrics:
                        deltas = []
                        for identity in identities:
                            left = origins[(candidate,) + identity][metric]
                            right = origins[(reference,) + identity][metric]
                            if left is not None and right is not None:
                                deltas.append(left - right)
                        values[metric] = (math.fsum(deltas) / len(deltas) if deltas else None, len(deltas))
                    table[key] = values
                    keys.append(key)
    return table, keys


def summarize(directory: Path | str) -> dict:
    """Reject incomplete or inconsistent artifacts; return descriptive values.

    analysis.json is the provided receipt. CSV identity is checked against that
    receipt; this function does not independently regenerate simulator outputs.
    """
    directory = Path(directory)
    analysis_path = directory / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    _metadata(analysis)
    artifacts = analysis.get("artifacts", {})
    _require(set(artifacts) == set(ARTIFACT_ROWS), "exactly four diagnostic artifacts required")
    for name, count in ARTIFACT_ROWS.items():
        entry = artifacts[name]
        _require(isinstance(entry, dict) and type(entry.get("rows")) is int and entry["rows"] == count,
                 "artifact row denominator differs: " + name)
        _hash(entry.get("sha256"), name)
        _require(_sha256(directory / name) == entry["sha256"], "artifact SHA-256 differs: " + name)
    times, residual = _time_table(directory)
    origins, metrics, native_identities = _origin_table(directory, times)
    seed_table = _aggregate_table(directory, origins, metrics, by_block=False)
    block_table = _aggregate_table(directory, origins, metrics, by_block=True)
    key_metrics = ["prefix_mean_" + name for name in KEY_MOMENT_NAMES]
    key_metrics += ["native_" + name for name in KEY_NATIVE_NAMES]
    seed_keys = list(itertools.product(MODELS, SEEDS, HORIZONS_MS))
    block_keys = [(model, seed, 500, block) for model, seed, block in itertools.product(MODELS, SEEDS, BLOCKS)]
    paired_seed, paired_seed_keys = _paired(origins, key_metrics, by_block=False)
    paired_block, paired_block_keys = _paired(origins, key_metrics, by_block=True)
    return {
        "status": "VERIFIED_DESCRIPTIVE_NATIVE_READOUT_SUMMARY",
        "analysis_sha256": _sha256(analysis_path),
        "integrity": {"artifact_rows": ARTIFACT_ROWS, "retained_trajectories": len(times),
                      "posterior_caches": 48, "source_units": SOURCE_UNITS, "draws": DRAWS,
                      "future_bins": FUTURE_BINS, "bin_ms": BIN_MS,
                      "aggregate_values_crosschecked": sum(ARTIFACT_ROWS[name] for name in (
                          "seed_horizon_summary.csv", "source_block_summary.csv")),
                      "time_mean_square_closure_max_absolute": residual,
                      "comparison_rule": "IEEE-754 gamma(n) accumulation bounds; no scientific tolerance changed"},
        "runtime": analysis["runtime"], "native_method_device": native_identities,
        "metrics": key_metrics,
        "validated_metrics": metrics,
        "cell_identity_columns": ["model", "seed", "horizon_ms"],
        "block_identity_columns": ["model", "seed", "horizon_ms", "block"],
        "seed_horizon": _compact_rows(seed_table, seed_keys, key_metrics, 48),
        "block_500ms": _compact_rows(block_table, block_keys, key_metrics, 12),
        "pairs_candidate_reference": [list(pair) for pair in PAIRS],
        "paired_identity_columns": ["pair_index", "seed", "horizon_ms"],
        "paired_block_identity_columns": ["pair_index", "seed", "horizon_ms", "block"],
        "paired_seed_horizon_deltas": _compact_rows(paired_seed, paired_seed_keys, key_metrics, 48),
        "paired_block_500ms_deltas": _compact_rows(paired_block, paired_block_keys, key_metrics, 12),
        "availability_encoding": "Null means zero available origins. Other values use all origins unless indexed in partial_availability as [row,metric,available].",
        "delta_definition": "Mean candidate-minus-reference difference over matched available origins. No seed averaging or significance test.",
        "limitations": [
            analysis["q_reference"],
            "Only 16 retained draws; an empirical stochastic gap does not identify physical noise or pure variance error.",
            analysis["hazard_interpretation"],
            analysis["native_metric_windows"],
            "Signed A/B/C differences share the q reference. Squared errors and their cross term retain that reference.",
            "This diagnostic does not qualify scientific claims, SOTA, or a correction's causal effect. No protected evaluation is performed.",
        ],
        "scientific_claim_qualified": False, "significance_tests_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

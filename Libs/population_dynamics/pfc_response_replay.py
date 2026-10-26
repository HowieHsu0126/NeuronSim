"""Independent saved-array replay of PFC paired conditional mean responses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from .pfc_parameter_block_execution import CONDITIONS
from .population_forecast_replay import (
    numpy_unit_metrics, _project_units, METRIC_RTOL, METRIC_ATOL,
)
from .provenance import sha256_file

CHUNK_UNITS = 4096
ARRAY_NAMES = {
    "candidate_generated_counts", "factual_generated_counts",
    "candidate_observed_counts", "factual_observed_counts",
    "candidate_latent_states", "factual_latent_states",
    "candidate_terminal_ready", "factual_terminal_ready",
    "unit_ids", "unit_field_index", "field_names",
}
CONTRAST_METRICS = (
    "unit_mean_contrast_mse_corrected", "field_mean_contrast_mse_hz2_corrected",
    "generated_total_contrast_per_unit", "observed_total_contrast_per_unit",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "PFC_RESPONSE_REPLAY_UNIQUE_CSV_COLUMNS_REQUIRED")
        return list(reader)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def authenticate_outputs(root, report):
    expected = {"arm_scores.csv", "mean_contrasts.csv", "costs.csv"}
    expected |= {f"condition_{index:02d}/{name}" for index in range(10)
                 for name in ("comparison.json", "samples.npz")}
    require(set(report["outputs"]) == expected, "PFC_RESPONSE_REPLAY_COMPLETE_OUTPUT_MANIFEST_REQUIRED")
    for name, binding in report["outputs"].items():
        path = root / name
        require(path.stat().st_size == binding["bytes"] and sha256_file(path) == binding["sha256"],
                "PFC_RESPONSE_REPLAY_OUTPUT_CHANGED:" + name)


def ordered_pair_mse(candidate, factual, candidate_truth, factual_truth, horizon,
                     *, chunk_units=CHUNK_UNITS):
    """Signed integer residual U-statistic over distinct ordered draw pairs."""
    require(candidate.shape == factual.shape and candidate.ndim == 3 and len(candidate) >= 2
            and candidate_truth.shape == factual_truth.shape == candidate.shape[1:]
            and type(horizon) is int and 0 < horizon <= candidate.shape[1],
            "PFC_RESPONSE_REPLAY_COMPLETE_MEAN_CONTRAST_REQUIRED")
    draws, _, units = candidate.shape
    numerator = 0
    for start in range(0, units, chunk_units):
        stop = min(start + chunk_units, units)
        residual = (candidate[:, :horizon, start:stop].astype(np.int64)
                    - factual[:, :horizon, start:stop].astype(np.int64)
                    - candidate_truth[None, :horizon, start:stop].astype(np.int64)
                    + factual_truth[None, :horizon, start:stop].astype(np.int64))
        magnitude = int(np.max(np.abs(residual), initial=0))
        for left in range(draws):
            for right in range(draws):
                if left == right:
                    continue
                if magnitude * magnitude * residual[left].size <= np.iinfo(np.int64).max:
                    numerator += int(np.sum(residual[left] * residual[right], dtype=np.int64))
                else:
                    numerator += sum(int(a) * int(b) for a, b in
                                     zip(residual[left].flat, residual[right].flat))
    return numerator / (draws * (draws - 1) * horizon * units)


def independent_contrasts(candidate, factual, candidate_truth, factual_truth,
                          mapping, fields, sizes, bin_seconds, horizons):
    """Recompute unit and field mean effects without the evaluator's correction."""
    candidate_field, candidate_observed = _project_units(candidate, candidate_truth, mapping, fields)
    factual_field, factual_observed = _project_units(factual, factual_truth, mapping, fields)
    present = sizes > 0
    require(present.any(), "PFC_RESPONSE_REPLAY_DEFINED_FIELD_REQUIRED")
    results = []
    for horizon in horizons:
        unit = ordered_pair_mse(candidate, factual, candidate_truth, factual_truth, horizon)
        # Compute one field's ordered-pair integer numerator at a time. Scaling
        # follows the declared per-field mean-unit Hz observation map.
        field_scores = []
        for field in np.flatnonzero(present):
            field_mse = ordered_pair_mse(
                candidate_field[:, :, field:field + 1], factual_field[:, :, field:field + 1],
                candidate_observed[:, field:field + 1], factual_observed[:, field:field + 1], horizon)
            field_scores.append(field_mse / (float(sizes[field]) * bin_seconds) ** 2)
        difference_sum = sum(int(np.sum(candidate[:, :horizon, start:start + CHUNK_UNITS], dtype=np.int64))
                             - int(np.sum(factual[:, :horizon, start:start + CHUNK_UNITS], dtype=np.int64))
                             for start in range(0, candidate.shape[-1], CHUNK_UNITS))
        observed_sum = (int(np.sum(candidate_truth[:horizon], dtype=np.int64))
                        - int(np.sum(factual_truth[:horizon], dtype=np.int64)))
        results.append(dict(horizon_bins=horizon,
            unit_mean_contrast_mse_corrected=unit,
            field_mean_contrast_mse_hz2_corrected=float(np.mean(field_scores)),
            generated_total_contrast_per_unit=difference_sum / (len(candidate) * candidate.shape[-1]),
            observed_total_contrast_per_unit=observed_sum / candidate.shape[-1]))
    return results


def _compare(actual, expected, label, *, field=False):
    if expected is None:
        require(actual in (None, ""), "PFC_RESPONSE_REPLAY_UNDEFINED_CHANGED:" + label)
        return 0.
    value = float(actual)
    require(np.isfinite(value) and np.isclose(value, expected,
            rtol=0 if field else METRIC_RTOL, atol=1e-8 if field else METRIC_ATOL),
            "PFC_RESPONSE_REPLAY_SCORE_MISMATCH:" + label)
    return abs(value - expected)


def _energy(samples, observed):
    flat = samples.reshape(len(samples), -1)
    truth = observed.reshape(-1)
    target = float(np.mean(np.linalg.norm(flat - truth, axis=1)))
    pair = sum(float(np.linalg.norm(flat[left] - flat[right]))
               for left in range(len(flat)) for right in range(left + 1, len(flat)))
    return (target - pair / (len(flat) * (len(flat) - 1))) / np.sqrt(truth.size)


def _difference(reference, actual, stop_bins=None):
    left = reference[:, :stop_bins] if stop_bins is not None else reference
    right = actual[:, :stop_bins] if stop_bins is not None else actual
    require(left.shape == right.shape, "PFC_RESPONSE_REPLAY_DIAGNOSTIC_SHAPE_CHANGED")
    bins = left.shape[1]
    differences, first = 0, None
    for start in range(0, left.shape[-1], CHUNK_UNITS):
        changed = left[..., start:start + CHUNK_UNITS] != right[..., start:start + CHUNK_UNITS]
        differences += int(np.count_nonzero(changed))
        indices = np.flatnonzero(np.any(changed, axis=(0, 2)))
        if len(indices):
            first = int(indices[0]) if first is None else min(first, int(indices[0]))
    return dict(exact=differences == 0, differences=differences, first_difference_bin=first, bins=bins)


def replay(path, output):
    root, output = Path(path), Path(output)
    evaluation_path = root / "evaluation.json"
    report = read(evaluation_path)
    require(report["status"] == "COMPLETE_TEN_CONDITION_SOURCE_RESPONSE_DEVELOPMENT"
            and (report["conditions"], report["arm_score_rows"], report["mean_contrast_rows"]) == (10, 60, 30)
            and (report["units"], report["fields"]) == (100000, 17)
            and report["scientific_intervention_qualified"] is False
            and report["independent_success_gate_evaluated"] is False,
            "PFC_RESPONSE_REPLAY_COMPLETE_DEVELOPMENT_RESULT_REQUIRED")
    config = report["configuration"]
    require(config["condition_indices"] == list(range(10)) and config["horizons_bins"] == [4, 16, 100]
            and config["joint_samples"] == 16 and config["support_windows"] == [1, 2]
            and config["sampling_stream_seeds"] == dict(initial=41, dynamics=42, observation=43),
            "PFC_RESPONSE_REPLAY_REGISTERED_CONFIGURATION_REQUIRED")
    evaluation_sha = sha256_file(evaluation_path)
    authenticate_outputs(root, report)
    arms, contrasts, costs = table(root / "arm_scores.csv"), table(root / "mean_contrasts.csv"), table(root / "costs.csv")
    require(len(arms) == 60 and len(contrasts) == 30 and len(costs) == 10,
            "PFC_RESPONSE_REPLAY_COMPLETE_TABLES_REQUIRED")
    expected_arms = {(index, arm, horizon) for index in range(10)
                     for arm in ("candidate", "factual") for horizon in config["horizons_bins"]}
    arm_keys = [(int(row["condition_index"]), row["arm"], int(row["horizon_bins"])) for row in arms]
    contrast_keys = [(int(row["condition_index"]), int(row["horizon_bins"])) for row in contrasts]
    require(set(arm_keys) == expected_arms and len(set(arm_keys)) == 60
            and set(contrast_keys) == {(index, horizon) for index in range(10) for horizon in config["horizons_bins"]}
            and len(set(contrast_keys)) == 30, "PFC_RESPONSE_REPLAY_TABLE_GRID_CHANGED")
    arm_index, contrast_index = dict(zip(arm_keys, arms)), dict(zip(contrast_keys, contrasts))
    result = dict(status="REPLAYING_SAVED_PFC_RESPONSE", evaluation_sha256=evaluation_sha,
        checked_conditions=[], consumed_generated_count_values=0, arm_score_rows=0, mean_contrast_rows=0,
        maximum_absolute_metric_difference=0., samples_regenerated=False, source_simulations=0,
        training_updates=0, scientific_intervention_qualified=False, contrast_distribution_qualified=False)
    if output.exists():
        previous = read(output)
        require(previous["evaluation_sha256"] == evaluation_sha,
                "PFC_RESPONSE_REPLAY_EXISTING_IDENTITY_CHANGED")
        if previous["status"] == "PASS_INDEPENDENT_SAVED_PFC_RESPONSE_REPLAY":
            return previous
        result = previous
        if result.get("status") == "FAILED_SAVED_PFC_RESPONSE_REPLAY":
            result.setdefault("preserved_failures", []).append(dict(
                error=result.get("error"), error_type=result.get("error_type"),
                completed_conditions=len(result["checked_conditions"])))
            result.pop("error", None)
            result.pop("error_type", None)
            result["status"] = "REPLAYING_SAVED_PFC_RESPONSE"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, result)
    expected_mapping = np.r_[np.arange(80000, dtype=np.int64) * 16 // 80000,
                             np.full(20000, 16, dtype=np.int64)]
    sizes = np.bincount(expected_mapping, minlength=17)
    intervals = np.r_[np.full(80000, 40, dtype=np.int64), np.full(20000, 20, dtype=np.int64)]
    with tempfile.TemporaryDirectory(prefix="pfc-response-replay-", dir="/tmp/neuronsim") as scratch:
        scratch = Path(scratch)
        reference05 = {}
        try:
            for index in range(10):
                # Only the two diagnostic conditions need their saved arrays
                # again after an interrupted replay; completed score arithmetic
                # is reused without repeating it.
                already = any(row["condition_index"] == index for row in result["checked_conditions"])
                if already and index not in (5, 8):
                    continue
                condition = read(root / f"condition_{index:02d}/comparison.json")
                require(condition["identity"] == report["identity"]
                        and condition["condition_index"] == index and condition["condition_id"] == CONDITIONS[index]
                        and condition["origin_ticks"] == config["origins_ticks_by_condition"][index]
                        and condition["draws"] == 16 and condition["units"] == 100000 and condition["fields"] == 17
                        and condition["samples_sha256"] == report["outputs"][f"condition_{index:02d}/samples.npz"]["sha256"]
                        and condition["future_outcomes_used_only_after_both_generations"] is True,
                        "PFC_RESPONSE_REPLAY_CONDITION_BINDING_CHANGED")
                with np.load(root / f"condition_{index:02d}/samples.npz", allow_pickle=False) as stored:
                    require(set(stored.files) == ARRAY_NAMES, "PFC_RESPONSE_REPLAY_ARRAY_SCHEMA_CHANGED")
                    arrays = {name: stored[name] for name in stored.files}
                require(np.array_equal(arrays["unit_ids"], np.arange(100000))
                        and np.array_equal(arrays["unit_field_index"], expected_mapping)
                        and arrays["field_names"].shape == (17,),
                        "PFC_RESPONSE_REPLAY_SOURCE_UNIT_FIELD_MAP_CHANGED")
                for arm in ("candidate", "factual"):
                    require(arrays[arm + "_generated_counts"].shape == (16, 100, 100000)
                            and arrays[arm + "_generated_counts"].dtype == np.int64
                            and arrays[arm + "_observed_counts"].shape == (100, 100000)
                            and arrays[arm + "_observed_counts"].dtype == np.int64
                            and arrays[arm + "_terminal_ready"].shape == (16, 1, 100000),
                            "PFC_RESPONSE_REPLAY_COMPLETE_FULL_UNIT_ARMS_REQUIRED")
                if index == 0:
                    same = {}
                    for suffix in ("generated_counts", "observed_counts", "latent_states", "terminal_ready"):
                        same[suffix] = np.array_equal(arrays["candidate_" + suffix], arrays["factual_" + suffix])
                    require(all(same.values()), "PFC_RESPONSE_REPLAY_SAME_CONDITION_ARMS_DIFFER")
                    result["same_condition_zero_exact"] = same
                if not already:
                    maximum = 0.
                    for arm in ("candidate", "factual"):
                        values, truth = arrays[arm + "_generated_counts"], arrays[arm + "_observed_counts"]
                        metrics, _, consumed = numpy_unit_metrics(values, truth, intervals, 100,
                                                                   config["horizons_bins"], config["support_windows"])
                        field, field_truth = _project_units(values, truth, expected_mapping, 17)
                        rates, observed = field / (sizes * .005), field_truth / (sizes * .005)
                        for horizon in config["horizons_bins"]:
                            row = arm_index[(index, arm, horizon)]
                            require(row["condition_id"] == CONDITIONS[index] and int(row["origin_ticks"]) == condition["origin_ticks"],
                                    "PFC_RESPONSE_REPLAY_ARM_IDENTITY_CHANGED")
                            source_row = next(row for row in condition[arm + "_scores"] if row["horizon_bins"] == horizon)
                            for key, value in metrics[horizon].items():
                                maximum = max(maximum, _compare(row[key], value, key))
                                maximum = max(maximum, _compare(source_row[key], value, key))
                            energy = _energy(rates[:, :horizon], observed[:horizon])
                            maximum = max(maximum, _compare(row["field_energy_score_hz"], energy, "field energy", field=True))
                            maximum = max(maximum, _compare(source_row["field_energy_score_hz"], energy, "field energy", field=True))
                        result["consumed_generated_count_values"] += consumed
                    effects = independent_contrasts(
                        arrays["candidate_generated_counts"], arrays["factual_generated_counts"],
                        arrays["candidate_observed_counts"], arrays["factual_observed_counts"],
                        expected_mapping, 17, sizes, .005, config["horizons_bins"])
                    for effect in effects:
                        row = contrast_index[(index, effect["horizon_bins"])]
                        source_row = next(row for row in condition["mean_contrasts"]
                                          if row["horizon_bins"] == effect["horizon_bins"])
                        require(row["condition_id"] == CONDITIONS[index]
                                and row["model_coupling"] == "common_explicit_sampling_seeds; conditional_mean_comparison_only"
                                and row["contrast_distribution_qualified"] == "False"
                                and row["scientific_intervention_qualified"] == "False",
                                "PFC_RESPONSE_REPLAY_MEAN_EFFECT_SCOPE_CHANGED")
                        for key in CONTRAST_METRICS:
                            maximum = max(maximum, _compare(row[key], effect[key], key))
                            maximum = max(maximum, _compare(source_row[key], effect[key], key))
                    result["checked_conditions"].append(dict(condition_index=index, condition_id=CONDITIONS[index],
                        arm_score_rows=6, mean_contrast_rows=3, generated_count_values=320000000))
                    result["arm_score_rows"] += 6
                    result["mean_contrast_rows"] += 3
                    result["maximum_absolute_metric_difference"] = max(result["maximum_absolute_metric_difference"], maximum)
                    _write(output, result)
                if index == 5:
                    for name in ("candidate_generated_counts", "candidate_latent_states", "candidate_terminal_ready",
                                 "factual_generated_counts", "factual_latent_states", "factual_terminal_ready",
                                 "candidate_observed_counts", "factual_observed_counts"):
                        target = scratch / (name + ".npy")
                        np.save(target, arrays[name], allow_pickle=False)
                        reference05[name] = target
                elif index == 8:
                    diagnostic = dict(reference_condition="omission_none", candidate_condition="omission_antipodal",
                        origin_tick=2000, first_registered_distractor_tick=8000, known_command_prefix_bins=60,
                        interpretation="Future actuator geometry differs; factual boundary, sampling seeds and commands before tick8000 are shared. Descriptive implementation diagnostic only.")
                    for name in ("candidate_generated_counts", "candidate_latent_states",
                                 "factual_generated_counts", "factual_latent_states"):
                        reference = np.load(reference05[name], mmap_mode="r", allow_pickle=False)
                        diagnostic[name] = _difference(reference, arrays[name], 60)
                        del reference
                    for name in ("candidate_terminal_ready", "factual_terminal_ready",
                                 "candidate_observed_counts", "factual_observed_counts"):
                        reference = np.load(reference05[name], mmap_mode="r", allow_pickle=False)
                        diagnostic[name] = dict(exact=np.array_equal(reference, arrays[name]),
                                                differences=int(np.count_nonzero(reference != arrays[name])))
                        del reference
                    result["source_command_invariance_diagnostic"] = diagnostic
                    _write(output, result)
                del arrays
            require(len(result["checked_conditions"]) == 10 and result["arm_score_rows"] == 60
                    and result["mean_contrast_rows"] == 30
                    and result["consumed_generated_count_values"] == 3200000000,
                    "PFC_RESPONSE_REPLAY_COMPLETE_GENERATED_DENOMINATOR_REQUIRED")
            result.update(status="PASS_INDEPENDENT_SAVED_PFC_RESPONSE_REPLAY",
                declared_outputs_verified=23, declared_output_bytes=sum(row["bytes"] for row in report["outputs"].values()),
                comparison_scope="conditional means only; no joint intervention-law, efficacy, or source-fidelity qualification",
                source_future_arrays_used_for_scoring_only=True)
            _write(output, result)
            return result
        except BaseException as error:
            result.update(status="FAILED_SAVED_PFC_RESPONSE_REPLAY", error=str(error), error_type=type(error).__name__)
            _write(output, result)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.path, args.output)
    print(json.dumps({key: result[key] for key in ("status", "arm_score_rows", "mean_contrast_rows",
                                                 "consumed_generated_count_values")}))


if __name__ == "__main__":
    main()

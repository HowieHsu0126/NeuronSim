"""Matched full-unit BRAID conditional-mean forecasts and independent replay.

The development selection blocks, observed prefixes, actual controls, clocks,
unit identities and field mapping match the population model. No joint sampler
or distribution score is assigned to the deterministic source readout.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np
import tensorflow as tf
import yaml

from .braid import BRAIDMeanPredictor
from .braid_trajectory_training import _binding, _checkpoint_files, validate_config
from .native_trajectory_cohort import TrajectoryCohort
from .provenance import sha256_file

ORIGINS_MS = [1000., 3000., 5000.]
HORIZONS_MS = [20., 80., 500.]


def mean_scores(means, truth, field_index, units, sd, bin_seconds):
    if (means.ndim != 2 or means.shape != truth.shape or means.shape[1] != len(field_index)
            or not np.isfinite(means).all() or np.any(means < 0)
            or not np.isfinite(truth).all() or np.any(truth < 0) or np.any(truth != np.rint(truth))):
        raise ValueError("complete finite identified means and integer truth required")
    index = np.asarray(field_index)
    units, sd = np.asarray(units), np.asarray(sd)
    if (index.dtype.kind not in "iu" or np.any(index < 0) or np.any(index >= len(units))
            or not np.array_equal(np.bincount(index, minlength=len(units)), units)
            or sd.shape != units.shape or not np.isfinite(sd).all() or np.any(sd <= 0)
            or not np.isfinite(bin_seconds) or bin_seconds <= 0):
        raise ValueError("source field projection and positive fitting scale required")
    fields = np.stack([np.bincount(index, weights=row, minlength=len(units)) for row in means])
    actual = np.stack([np.bincount(index, weights=row, minlength=len(units)) for row in truth])
    error = (fields - actual) / (units * bin_seconds * sd)
    metrics = dict(unit_bin_mean_mse=float(np.mean((means.astype(np.float64) - truth)**2)),
                   fitting_standardized_field_rate_mse=float(np.mean(error**2)),
                   predicted_spikes_per_unit=float(means.astype(np.float64).sum(0).mean()),
                   observed_spikes_per_unit=float(truth.astype(np.float64).sum(0).mean()))
    return metrics, fields, actual


def _load(acquisition, fields, training, config_path, fitting_scale, seed):
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    tf.config.threading.set_intra_op_parallelism_threads(config["cpu_threads"])
    tf.config.threading.set_inter_op_parallelism_threads(1)
    cohort = TrajectoryCohort(acquisition, fields, config)
    binding = _binding(cohort, config, seed)
    pointer = json.loads((training / "committed.json").read_text())
    if pointer["binding"] != binding:
        raise ValueError("BRAID evaluation training/data/software binding changed")
    state_path = training / pointer["checkpoint"] / "training.json"
    if sha256_file(state_path) != pointer["training_sha256"]:
        raise ValueError("BRAID committed training record changed")
    state = json.loads(state_path.read_text())
    if not state["stop_reason"] or state["best_epoch"] is None:
        raise ValueError("BRAID evaluation requires completed development training")
    selected = training / state["best_checkpoint"]
    if _checkpoint_files(selected) != state["best_checkpoint_files"]:
        raise ValueError("BRAID selected checkpoint changed")
    model = BRAIDMeanPredictor.load(selected)
    if model.unit_ids != tuple(map(int, cohort.unit_ids)) or model.bin_seconds != cohort.bin_seconds:
        raise ValueError("BRAID selected model unit map or clock changed")
    with np.load(fitting_scale, allow_pickle=False) as scale:
        sd, units, field_ids = (scale[key].copy() for key in ("standard_deviation_hz", "unit_count", "field_ids"))
    if (not np.array_equal(field_ids, cohort.field_ids)
            or not np.array_equal(np.bincount(cohort.unit_field_index, minlength=len(units)), units)):
        raise ValueError("BRAID and population fitting field mapping differs")
    return model, cohort, state, sd, units, field_ids


def _forecast(model, cohort, case, origin_ms, sd, units):
    start = time.perf_counter()
    counts, controls = cohort.window(case)
    loading = time.perf_counter() - start
    boundary = round(origin_ms / (cohort.bin_seconds * 1000))
    horizon = round(max(HORIZONS_MS) / (cohort.bin_seconds * 1000))
    if boundary < 1 or boundary + horizon > len(counts):
        raise ValueError("matched forecast outside complete episode")
    start = time.perf_counter()
    state = model.infer_prefix(
        dict(counts=counts[None, :boundary], stimulus=controls[None, :boundary]),
        dict(unit_ids=model.unit_ids, bin_seconds=model.bin_seconds))
    inference = time.perf_counter() - start
    start = time.perf_counter()
    means = model.forecast(state, controls[None, boundary:boundary + horizon]).numpy()[0]
    generation = time.perf_counter() - start
    truth = counts[boundary:boundary + horizon]
    start = time.perf_counter()
    rows, fields, actual = [], None, None
    for horizon_ms in HORIZONS_MS:
        bins = round(horizon_ms / (cohort.bin_seconds * 1000))
        metrics, current_fields, current_truth = mean_scores(
            means[:bins], truth[:bins], cohort.unit_field_index, units, sd, cohort.bin_seconds)
        rows.append(dict(block=case["block"], case_id=case["case_id"], origin_ms=origin_ms,
                         horizon_ms=horizon_ms, **metrics))
        if bins == horizon:
            fields, actual = current_fields, current_truth
    scoring = time.perf_counter() - start
    return rows, dict(expected_field_counts=fields, observed_field_counts=actual,
                      boundary_state=state.boundary.numpy(), field_ids=cohort.field_ids,
                      unit_count=units), means, truth, dict(input_loading_seconds=loading,
                      inference_seconds=inference, full_unit_mean_forecast_seconds=generation,
                      field_projection_and_scoring_seconds=scoring,
                      generated_unit_mean_values=int(means.size))


def evaluate(acquisition, fields, training, config_path, fitting_scale, output, seed, *, replay=False):
    paths = map(Path, (acquisition, fields, training, config_path, fitting_scale, output))
    acquisition, fields, training, config_path, fitting_scale, output = paths
    if output.exists() and not replay:
        raise FileExistsError("BRAID existing evaluation requires explicit replay")
    model, cohort, state, sd, units, field_ids = _load(acquisition, fields, training, config_path, fitting_scale, seed)
    binding = dict(training_committed_sha256=sha256_file(training / "committed.json"),
                   config_sha256=sha256_file(config_path), fitting_scale_sha256=sha256_file(fitting_scale),
                   cohort=cohort.binding, origins_ms=ORIGINS_MS, horizons_ms=HORIZONS_MS,
                   source_unit_ids=list(map(int, cohort.unit_ids)), source_field_ids=list(map(str, field_ids)),
                   evaluator_sha256=sha256_file(Path(__file__)))
    if replay:
        original = json.loads((output / "evaluation.json").read_text())
        if original["binding"] != binding:
            raise ValueError("BRAID replay source/config/evaluator binding changed")
        for name, digest in original["artifacts"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("BRAID saved forecast changed:" + name)
        with (output / "scores.csv").open() as stream:
            expected = list(csv.DictReader(stream))
    else:
        output.mkdir(parents=True)
    rows, costs, artifacts = [], [], {}
    start = time.perf_counter()
    for case in cohort.validation:
        for origin in ORIGINS_MS:
            origin_start = time.perf_counter()
            current, payload, means, truth, cost = _forecast(model, cohort, case, origin, sd, units)
            for row in current:
                row["seed"] = seed
                row["method"] = "braid_mean"
            name = f"{case['case_id']}_origin_{origin:g}.npz"
            if replay:
                with np.load(output / name, allow_pickle=False) as saved:
                    for key, values in payload.items(): np.testing.assert_array_equal(saved[key], values)
                for actual_row, reference in zip(current, expected[len(rows):len(rows) + 3]):
                    if set(actual_row) != set(reference): raise ValueError("BRAID replay score schema changed")
                    for key, value in actual_row.items():
                        if isinstance(value, str):
                            if reference[key] != value: raise ValueError("BRAID replay identity differs")
                        elif float(reference[key]) != value: raise ValueError("BRAID replay numerical score differs:" + key)
                if not rows:
                    with np.load(output / "full_unit_mean_forecast.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["expected_counts"], means)
                        np.testing.assert_array_equal(saved["observed_counts"], truth)
                        np.testing.assert_array_equal(saved["unit_ids"], cohort.unit_ids)
            else:
                serialization = time.perf_counter()
                np.savez_compressed(output / name, **payload)
                if not rows:
                    np.savez_compressed(output / "full_unit_mean_forecast.npz", expected_counts=means,
                                        observed_counts=truth, unit_ids=cohort.unit_ids)
                    artifacts["full_unit_mean_forecast.npz"] = sha256_file(output / "full_unit_mean_forecast.npz")
                cost["serialization_and_hash_seconds"] = time.perf_counter() - serialization
                artifacts[name] = sha256_file(output / name)
            rows.extend(current)
            costs.append(dict(case_id=case["case_id"], block=case["block"], origin_ms=origin,
                              origin_total_seconds=time.perf_counter() - origin_start,
                              peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, **cost))
    if len(rows) != 144 or len(costs) != 48: raise ValueError("complete 48-origin/144-score denominator required")
    if replay:
        result = dict(status="PASS", training_seed=seed, verified_origins=48, verified_score_rows=144,
                      all_origin_fields_boundary_and_metrics_exact=True, all_unit_mean_values_recomputed=480000000,
                      saved_full_unit_origins_compared=1, evaluation_sha256=sha256_file(output / "evaluation.json"),
                      scientific_claim_qualified=False)
        (output / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    for name, values in [("scores.csv", rows), ("origin_costs.csv", costs)]:
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        artifacts[name] = sha256_file(output / name)
    result = dict(status="EVALUATED_DEVELOPMENT", training_seed=seed, binding=binding, score_rows=len(rows),
                  origins=len(costs), generated_unit_mean_values=480000000,
                  parameter_count=int(model.native.model.count_params()), training_elapsed_seconds=state["elapsed_seconds"],
                  evaluation_elapsed_seconds=time.perf_counter() - start, joint_spike_sampler=False,
                  selection_blocks_reused=True, inferential_statistics_performed=False, scientific_claim_qualified=False,
                  artifacts=artifacts)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "training", "config", "fitting-scale", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", choices=[41, 42, 43], type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.acquisition, args.fields, args.training, args.config, args.fitting_scale,
                      args.output, args.seed, replay=args.replay)
    print(json.dumps({k: v for k, v in result.items() if k not in ("binding", "artifacts")}))


if __name__ == "__main__":
    main()

"""Predict native posterior residuals with whole-fitting-block exclusion.

This diagnoses the existing representation. Posterior coordinates are not physical
innovations, and one-step teacher-observed predictions are not free generation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
import yaml

from .population_prior_residual_diagnostic import cached_residuals, residual_moments, _csv
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file
from .residual_prediction import crossfit_residuals


def causal_inputs(mean, residual, controls):
    """At boundary t use m[t], u[t+1], r[t-1] to predict r[t]."""
    mean, residual, controls = map(np.asarray, (mean, residual, controls))
    if (mean.ndim != 3 or mean.shape[0] < 4
            or residual.shape != (mean.shape[0] - 1, *mean.shape[1:])
            or controls.ndim != 2 or controls.shape[0] != mean.shape[0]
            or any(x.dtype.kind not in "fiu" or not np.isfinite(x).all()
                   for x in (mean, residual, controls))):
        raise ValueError("finite aligned complete episode coordinates and controls required")
    width = mean.shape[1] * mean.shape[2]
    current = mean[1:-1].reshape(-1, width)
    history = residual[:-1].reshape(-1, width)
    return np.concatenate((current, controls[2:], history), axis=1), residual[1:].reshape(-1, width)


def validate_config(config):
    fixed = dict(schema="native_prior_residual_prediction_v1",
        training_seeds=[41, 42, 43], fitting_blocks=list(range(14)),
        excluded_selection_blocks=list(range(14, 18)), templates_per_block=4,
        episode_bins=1200, bin_ms=5, source_units=100000,
        inner_validation="next_outer_block_cyclic",
        feature_sets=["current_state_and_next_control", "plus_previous_residual"],
        template_time_centering=False, bootstrap_stream_seeds=[41, 42, 43],
        decision_scope="fitting_only_diagnostic_for_one_backend_correction",
        memory_decision_requires_improvement_over_current_and_zero=True)
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("registered fitting-only native residual protocol required")
    alphas = np.asarray(config["ridge_alphas"], dtype=float)
    if (alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all()
            or np.any(alphas <= 0) or len(np.unique(alphas)) != len(alphas)):
        raise ValueError("distinct finite positive ridge alphas required")
    if (type(config.get("bootstrap_draws_per_stream")) is not int
            or config["bootstrap_draws_per_stream"] < 1
            or config.get("confidence_level") != .95
            or config.get("cpu_threads") not in (1, 2)):
        raise ValueError("registered bootstrap confidence and CPU execution required")


def _coordinates(campaign, seed, scratch, protocol):
    cell = campaign / f"seed_{seed}"
    delivery = json.loads((cell / "delivery.json").read_text())
    if (delivery.get("status") != "PASS" or delivery.get("exit_code") != 0
            or delivery.get("training_seed") != seed
            or delivery.get("source_units") != protocol["source_units"]
            or delivery.get("frozen_parameters_unchanged") is not True):
        raise ValueError("accepted complete same-seed native prior delivery required")
    files = {row["path"]: row for row in delivery["files"]}
    if len(files) != len(delivery["files"]):
        raise ValueError("duplicate delivery paths")
    consumed = {}
    def stage(name):
        row = files[name]
        target = scratch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cell / name, target)
        if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
            raise ValueError("changed diagnostic input: " + name)
        consumed[name] = row["sha256"]
        return target

    report = json.loads(stage("training/run.json").read_text())
    checkpoint = torch.load(stage("training/best.pt"), map_location="cpu", weights_only=True)
    binding, config = checkpoint["binding"], checkpoint["binding"]["config"]
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    objective_hash = sha256_file(Path(__file__).with_name("population_observation_prior.py"))
    if (binding != report["binding"] or binding["seed"] != seed
            or binding["model_implementation_sha256"] != source_hash
            or report["trainer_sources"]["population_observation_prior.py"] != objective_hash
            or config["observation_objective"] != "fixed_readout_native_event_path_mixture_nll"
            or config["training_seeds"] != protocol["training_seeds"]
            or config["fit_blocks"] != protocol["fitting_blocks"]
            or config["validation_blocks"] != protocol["excluded_selection_blocks"]
            or config["prediction_particles"] != 8 or config["bin_ms"] != protocol["bin_ms"]
            or report["completed_epochs"] != 80
            or checkpoint["model_configuration"] != report["model_configuration"]
            or report["best_checkpoint_sha256"] != consumed["training/best.pt"]
            or checkpoint["next_epoch"] != report["best_epoch"] + 1):
        raise ValueError("native selected checkpoint or fitting protocol differs")
    constructor = checkpoint["model_configuration"]
    if len(constructor["unit_ids"]) != protocol["source_units"] or constructor["representation"] != "recurrent":
        raise ValueError("full-unit recurrent parent required")
    model = ControlledPopulationState(**constructor)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    fields, channels = len(model.field_names), model.latent_channels
    width, current_width = fields * channels, fields * channels + model.input_size
    shape = (len(protocol["fitting_blocks"]), protocol["templates_per_block"], protocol["episode_bins"] - 2)
    X = np.empty((*shape, current_width + width), dtype=np.float64)
    Y = np.empty((*shape, width), dtype=np.float64)
    programs = {}
    for bi, block in enumerate(protocol["fitting_blocks"]):
        for template in range(protocol["templates_per_block"]):
            case = f"train_block_{block:03d}_familiar_{template}"
            path = stage(f"training/posterior_cache/{case}.pt")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            cb = saved["binding"]
            if (cb["case_id"] != case
                    or cb["parent_checkpoint_sha256"] != checkpoint["initialization"]["parent_checkpoint_sha256"]
                    or cb["model_implementation_sha256"] != source_hash
                    or cb["bin_ms"] != protocol["bin_ms"]
                    or any(binding[key] != value for key, value in cb["cohort"].items())):
                raise ValueError("fitting cache identity or lineage differs")
            mean, variance, controls = (saved[key] for key in ("mean", "variance", "controls"))
            if mean.shape != (1, protocol["episode_bins"], fields, channels):
                raise ValueError("complete native episode required")
            if template in programs and not torch.equal(controls, programs[template]):
                raise ValueError("source control programs differ across fitting blocks")
            programs[template] = controls.clone()
            residual, _ = cached_residuals(model, mean, variance, controls)
            X[bi, template], Y[bi, template] = causal_inputs(
                mean[0].double().numpy(), residual, controls[0].double().numpy())
            path.unlink()
    metadata = dict(delivery_sha256=sha256_file(cell / "delivery.json"), files=consumed,
                    selected_epoch=report["best_epoch"], field_names=list(model.field_names),
                    fields=fields, channels=channels, current_width=current_width)
    return X, Y, [json.loads(name)[0] for name in model.field_names], metadata


def paired_decisions(rows, config):
    """Conditional block bootstrap; seeds are averaged within each source block."""
    seeds, blocks = config["training_seeds"], config["fitting_blocks"]
    lookup = {(r["seed"], r["block"], r["predictor"]): r["normalized_mse"] for r in rows}
    if len(lookup) != len(rows) or len(rows) != len(seeds) * len(blocks) * 3:
        raise ValueError("complete unique seed/block/predictor scores required")
    comparisons = {}
    for name, left, right in (("current_vs_zero", "zero", "current"),
                              ("history_vs_current", "current", "history"),
                              ("history_vs_zero", "zero", "history")):
        differences = np.array([[lookup[s, b, left] - lookup[s, b, right] for b in blocks] for s in seeds])
        per_block = differences.mean(axis=0)
        draws = []
        for seed in config["bootstrap_stream_seeds"]:
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(blocks), size=(config["bootstrap_draws_per_stream"], len(blocks)))
            draws.append(per_block[indices].mean(axis=1))
        tail = (1 - config["confidence_level"]) / 2
        low, high = np.quantile(np.concatenate(draws), [tail, 1 - tail])
        per_seed = differences.mean(axis=1)
        comparisons[name] = dict(mean_improvement=float(per_block.mean()),
            conditional_block_bootstrap_interval=[float(low), float(high)],
            per_seed_mean_improvement={str(s): float(v) for s, v in zip(seeds, per_seed)},
            per_block_seed_mean_improvement={str(b): float(v) for b, v in zip(blocks, per_block)},
            directional_support=bool(np.all(per_seed > 0) and low > 0))
    if (comparisons["history_vs_current"]["directional_support"]
            and comparisons["history_vs_zero"]["directional_support"]):
        action = "TEST_CAUSAL_MEMORY_CORRECTION_IN_FREE_GENERATION"
    elif comparisons["current_vs_zero"]["directional_support"]:
        action = "TEST_CURRENT_STATE_MEAN_CORRECTION_IN_FREE_GENERATION"
    else:
        action = "TESTED_LINEAR_FEATURES_DO_NOT_RESOLVE_RESIDUAL_DEPENDENCE"
    return dict(comparisons=comparisons, next_action=action)


def diagnose(campaign, config_path, output):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only admitted Slurm allocation required")
    campaign, config_path, output = map(Path, (campaign, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows, consumed, folds = [], {}, []
    for seed in config["training_seeds"]:
        with tempfile.TemporaryDirectory(prefix=f"neuronsim-residual-prediction-{seed}-") as name:
            X, Y, populations, source = _coordinates(campaign, seed, Path(name), config)
            consumed[str(seed)] = source
            for fold in crossfit_residuals(X, Y, source["current_width"], config["ridge_alphas"]):
                bi, inner = fold["block_index"], fold["inner_block_index"]
                scale = fold["training_target_scale"]
                if scale.shape != (Y.shape[-1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("positive fitting-only target scales required")
                predictions = {"zero": np.zeros_like(Y[bi]), **fold["predictions"]}
                for predictor, prediction in predictions.items():
                    error = Y[bi] - prediction
                    summary, _ = residual_moments(error.reshape(
                        config["templates_per_block"], config["episode_bins"] - 2,
                        source["fields"], source["channels"]), populations)
                    rows.append(dict(seed=seed, block=config["fitting_blocks"][bi],
                        predictor=predictor, inner_block=config["fitting_blocks"][inner],
                        alpha=None if predictor == "zero" else fold["selected_alphas"][predictor],
                        normalized_mse=float(np.square(error / scale).mean()), **summary))
                folds.append(dict(seed=seed, outer_block=config["fitting_blocks"][bi],
                                  inner_block=config["fitting_blocks"][inner],
                                  selected_alphas=fold["selected_alphas"]))
                print(json.dumps(dict(seed=seed, completed_outer_block=config["fitting_blocks"][bi])), flush=True)
            del X, Y, fold, predictions, error
    _csv(output / "prediction_by_block.csv", rows)
    decision = paired_decisions(rows, config)
    result = dict(status="COMPLETE_FITTING_BLOCK_RESIDUAL_PREDICTION", config=config,
        consumed=consumed, folds=folds, **decision, source_units=config["source_units"],
        fitting_episodes_per_seed=56, predicted_positions_per_episode=config["episode_bins"] - 2,
        first_origin_ms=2 * config["bin_ms"], last_origin_ms=(config["episode_bins"] - 1) * config["bin_ms"],
        horizon_ms=config["bin_ms"], score_rows=len(rows),
        target="m[t+1]-f(m[t],u[t+1])",
        current_features="m[t],u[t+1]", history_features="m[t],u[t+1],r[t-1]",
        centering="Fitting-fold feature and target means; no template/time centering.",
        ridge_objective="Mean over outputs of mean-row normalized squared error plus alpha times squared coefficients.",
        uncertainty="Descriptive block bootstrap conditional on fitted cross-validation predictors; seeds averaged within blocks. Does not include refit uncertainty.",
        limitations="One connectivity realization and posterior mean coordinates; one-step observed-history prediction is not free generation or physical innovation identification.",
        fixed_validation_selected_checkpoints=True, native_model_parameters_updated=False,
        fitting_ridge_predictors_trained=True, selection_arrays_accessed=False,
        validation_logs_accessed=False, source_simulation_performed=False,
        encoding_performed=False, generated_unit_observations=False, scientific_claim_qualified=False,
        elapsed_seconds=time.perf_counter()-started, job_id=os.environ["SLURM_JOB_ID"],
        configuration_sha256=sha256_file(config_path),
        software_sources={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "population_prior_residual_prediction.py", "residual_prediction.py",
            "population_prior_residual_diagnostic.py", "population_state.py")},
        artifacts={"prediction_by_block.csv": sha256_file(output / "prediction_by_block.csv")})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.campaign, args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "next_action", "score_rows")}))


if __name__ == "__main__":
    main()

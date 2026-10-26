"""Fitting-block residuals of an accepted native-observation population prior.

Cached posterior means are diagnostic coordinates, not true latent states or
physical process innovations. No forecast is conditioned on these future means.
The checkpoint was already selected by the registered validation procedure;
this command reads fitting caches and fitting optimization logs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file


@torch.no_grad()
def cached_residuals(model, mean, variance, controls):
    """r[t] = m[t+1] - f(m[t], u[t+1]); preserve episode boundaries."""
    if (mean.ndim != 4 or mean.shape[0] != 1 or mean.shape[1] < 3
            or mean.shape[-2:] != (len(model.field_names), model.latent_channels)
            or variance.shape != mean.shape
            or controls.shape != (*mean.shape[:2], model.input_size)
            or any(not torch.isfinite(x).all() for x in (mean, variance, controls))
            or torch.any(variance <= 0)):
        raise ValueError("complete aligned finite posterior means, variances and controls required")
    prediction = model.transition_mean(mean[0, :-1], controls[0, 1:])
    if prediction.shape != mean[0, 1:].shape or not torch.isfinite(prediction).all():
        raise ValueError("finite aligned native transition required")
    residual = (mean[0, 1:] - prediction).double().numpy()
    persistence = (mean[0, 1:] - mean[0, :-1]).double().numpy()
    return residual, float(np.square(persistence).mean())


def leave_block_out_center(values, block):
    """Subtract matched template/time means estimated from all other blocks."""
    if (values.ndim != 5 or values.shape[0] < 2 or type(block) is not int
            or not 0 <= block < values.shape[0]):
        raise ValueError("block/template/time/field/channel residuals and a valid block required")
    others = np.sum(values[np.arange(len(values)) != block], axis=0) / (len(values) - 1)
    return values[block] - others


def _mean_defined(values):
    defined = np.isfinite(values)
    return float(values[defined].mean()) if defined.any() else None


def residual_moments(values, populations):
    """Population covariance and Pearson lag correlation, without cross-episode pairs.

    Input axes are episode/time/source-field/channel. Undefined correlations
    remain NaN in the coordinate artifact and have an explicit denominator.
    """
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 4 or values.shape[1] < 2 or not np.isfinite(values).all()
            or len(populations) != values.shape[2]):
        raise ValueError("finite episode/time/field/channel residuals with source populations required")
    flat = values.reshape(-1, *values.shape[2:])
    mean = flat.mean(0)
    centered = flat - mean
    covariance = np.stack([centered[..., k].T @ centered[..., k] / len(flat)
                           for k in range(values.shape[-1])])
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    denominator = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    correlation = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan),
                            where=denominator > 0)
    left = values[:, :-1].reshape(-1, *values.shape[2:])
    right = values[:, 1:].reshape(-1, *values.shape[2:])
    left, right = left - left.mean(0), right - right.mean(0)
    lag_denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
    lag = np.divide((left * right).sum(0), lag_denominator,
                    out=np.full_like(mean, np.nan), where=lag_denominator > 0)
    upper = np.triu(np.ones((len(populations), len(populations)), dtype=bool), k=1)
    populations = np.asarray(populations)
    masks = dict(all_fields=upper,
                 cross_population=upper & (populations[:, None] != populations[None, :]),
                 within_population=upper & (populations[:, None] == populations[None, :]))
    summary = dict(transitions=len(flat), lag_pairs=len(left), mse=float(np.square(flat).mean()),
                   squared_bias=float(np.square(mean).mean()),
                   mean_coordinate_variance=float(diagonal.mean()),
                   lag1_mean=_mean_defined(lag), lag1_mean_absolute=_mean_defined(np.abs(lag)),
                   lag1_defined_coordinates=int(np.isfinite(lag).sum()),
                   lag1_coordinates=lag.size)
    for label, mask in masks.items():
        cov, corr = covariance[:, mask], correlation[:, mask]
        summary.update({label + "_mean_covariance": _mean_defined(cov),
                        label + "_mean_absolute_covariance": _mean_defined(np.abs(cov)),
                        label + "_mean_correlation": _mean_defined(corr),
                        label + "_mean_absolute_correlation": _mean_defined(np.abs(corr)),
                        label + "_defined_correlations": int(np.isfinite(corr).sum()),
                        label + "_correlation_pairs": corr.size})
    return summary, dict(mean=mean, mse=np.square(flat).mean(0), covariance=covariance,
                         correlation=correlation, lag1_correlation=lag)


def fit_history(rows, config, seed, completed_epochs):
    """Inspect complete committed fitting logs, retaining per-case ESS meaning."""
    cases = {f"train_block_{block:03d}_familiar_{template}"
             for block in config["fit_blocks"] for template in range(4)}
    horizons = [int(h / config["bin_ms"]) for h in config["prior_horizons_ms"]]
    groups = {}
    for row in rows:
        groups.setdefault(int(row["epoch"]), []).append(row)
    if set(groups) != set(range(completed_epochs)):
        raise ValueError("complete committed fitting epochs required")
    result = []
    for epoch, group in sorted(groups.items()):
        if len(group) != len(cases) or {r["case_id"] for r in group} != cases:
            raise ValueError("duplicate, missing or non-fitting case in committed epoch")
        for horizon in horizons:
            nll = np.array([float(r[f"prior_observation_nll_h{horizon}"]) for r in group])
            ess = np.array([float(r[f"prior_observation_particle_ess_h{horizon}"]) for r in group])
            norms = np.array([float(r["gradient_norm"]) for r in group])
            if (not all(np.isfinite(x).all() for x in (nll, ess, norms))
                    or np.any(ess < 1 - 1e-6) or np.any(ess > config["prediction_particles"] + 1e-6)):
                raise ValueError("finite fitting diagnostics and supported ESS required")
            result.append(dict(seed=seed, epoch=epoch, horizon_ms=horizon * config["bin_ms"],
                fitting_cases=len(group), nll_mean=float(nll.mean()), ess_mean=float(ess.mean()),
                ess_min=float(ess.min()), ess_median=float(np.median(ess)), ess_max=float(ess.max()),
                case_mean_ess_within_1e_6_of_one_fraction=float(np.mean(np.abs(ess - 1) <= 1e-6)),
                gradient_norm_mean=float(norms.mean()), gradient_norm_max=float(norms.max())))
    return result


def _csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(campaign, output):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only Slurm allocation on cpu03/gpu or cpu04/cpu required")
    campaign, output = Path(campaign), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    objective_hash = sha256_file(Path(__file__).with_name("population_observation_prior.py"))
    summaries, histories, bindings, seed_summaries = [], [], {}, []
    for seed in (41, 42, 43):
        cell = campaign / f"seed_{seed}"
        delivery = json.loads((cell / "delivery.json").read_text())
        if (delivery.get("status") != "PASS" or delivery.get("exit_code") != 0
                or delivery.get("training_seed") != seed or delivery.get("source_units") != 100000
                or delivery.get("frozen_parameters_unchanged") is not True):
            raise ValueError("accepted complete same-seed native prior delivery required")
        files = {row["path"]: row for row in delivery["files"]}
        if len(files) != len(delivery["files"]):
            raise ValueError("duplicate delivery paths")
        consumed = {}
        with tempfile.TemporaryDirectory(prefix=f"neuronsim-prior-residual-{seed}-") as scratch_name:
            scratch = Path(scratch_name)
            def stage(name):
                row = files[name]
                target = scratch / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cell / name, target)
                if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
                    raise ValueError("delivery input changed: " + name)
                consumed[name] = row["sha256"]
                return target
            report = json.loads(stage("training/run.json").read_text())
            checkpoint = torch.load(stage("training/best.pt"), map_location="cpu", weights_only=True)
            binding = checkpoint["binding"]
            config = binding["config"]
            if (binding != report["binding"] or binding["seed"] != seed
                    or binding["model_implementation_sha256"] != source_hash
                    or report["trainer_sources"]["population_observation_prior.py"] != objective_hash
                    or config["observation_objective"] != "fixed_readout_native_event_path_mixture_nll"
                    or config["training_seeds"] != [41, 42, 43]
                    or config["fit_blocks"] != list(range(14))
                    or config["validation_blocks"] != list(range(14, 18))
                    or config["prediction_particles"] != 8 or config["bin_ms"] != 5
                    or report["completed_epochs"] != 80
                    or checkpoint["model_configuration"] != report["model_configuration"]
                    or report["best_checkpoint_sha256"] != consumed["training/best.pt"]
                    or checkpoint["next_epoch"] != report["best_epoch"] + 1):
                raise ValueError("native checkpoint, source, objective or complete fitting protocol differs")
            constructor = checkpoint["model_configuration"]
            if len(constructor["unit_ids"]) != 100000 or constructor["representation"] != "recurrent":
                raise ValueError("complete 100000-unit native recurrent model required")
            model = ControlledPopulationState(**constructor)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            field_names = constructor["field_names"]
            populations = [json.loads(name)[0] for name in field_names]
            blocks = config["fit_blocks"]
            residuals = np.empty((len(blocks), 4, 1199, len(field_names), model.latent_channels), dtype=np.float64)
            persistence = np.empty((len(blocks), 4))
            posterior_variance = np.zeros((len(blocks), len(field_names), model.latent_channels))
            control_templates = {}
            for bi, block in enumerate(blocks):
                for template in range(4):
                    case = f"train_block_{block:03d}_familiar_{template}"
                    name = f"training/posterior_cache/{case}.pt"
                    path = stage(name)
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    cb = saved["binding"]
                    if (cb["case_id"] != case or cb["parent_checkpoint_sha256"]
                            != checkpoint["initialization"]["parent_checkpoint_sha256"]
                            or cb["model_implementation_sha256"] != source_hash or cb["bin_ms"] != config["bin_ms"]
                            or any(binding[key] != value for key, value in cb["cohort"].items())):
                        raise ValueError("cache case, encoder lineage, native source or clock differs")
                    mean, variance, controls = (saved[key] for key in ("mean", "variance", "controls"))
                    if mean.shape != (1, 1200, len(field_names), model.latent_channels):
                        raise ValueError("complete six-second native fitting cache required")
                    if template in control_templates and not torch.equal(controls, control_templates[template]):
                        raise ValueError("template/time control programs differ across fitting blocks")
                    control_templates[template] = controls.clone()
                    residuals[bi, template], persistence[bi, template] = cached_residuals(model, mean, variance, controls)
                    posterior_variance[bi] += variance[0, 1:].double().mean(0).numpy() / 4
                    path.unlink()
            coordinate_arrays = {}
            per_seed = []
            for mode in ("raw", "other_block_template_time_centered"):
                coordinates = {}
                for bi, block in enumerate(blocks):
                    values = residuals[bi] if mode == "raw" else leave_block_out_center(residuals, bi)
                    summary, detail = residual_moments(values, populations)
                    row = dict(seed=seed, block=block, mode=mode, **summary,
                               persistence_mse=float(persistence[bi].mean()) if mode == "raw" else None)
                    summaries.append(row)
                    per_seed.append(row)
                    for key, value in detail.items():
                        coordinates.setdefault(key, []).append(value)
                coordinate_arrays.update({mode + "_" + key: np.stack(value) for key, value in coordinates.items()})
            np.savez_compressed(output / f"seed_{seed}_coordinates.npz", **coordinate_arrays,
                block_ids=np.asarray(blocks), field_names=np.asarray(field_names),
                source_populations=np.asarray(populations), posterior_variance=posterior_variance,
                process_variance=model.process_log_variance.detach().double().exp().numpy())
            with stage("training/optimization.csv").open() as stream:
                history = fit_history(list(csv.DictReader(stream)), config, seed, report["completed_epochs"])
            histories.extend(history)
            mode_summaries = {}
            for mode in ("raw", "other_block_template_time_centered"):
                rows = [r for r in per_seed if r["mode"] == mode]
                metrics = ("mse", "squared_bias", "mean_coordinate_variance", "lag1_mean", "lag1_mean_absolute",
                           "cross_population_mean_covariance", "cross_population_mean_absolute_covariance",
                           "cross_population_mean_correlation", "cross_population_mean_absolute_correlation")
                mode_summaries[mode] = {key: _mean_defined(np.array([np.nan if r[key] is None else r[key] for r in rows]))
                                        for key in metrics}
            seed_summaries.append(dict(seed=seed, selected_epoch=report["best_epoch"],
                residuals=mode_summaries, last_epoch_fitting_weights=[r for r in history if r["epoch"] == report["best_epoch"]]))
            bindings[str(seed)] = dict(cell=str(cell.resolve()), delivery_sha256=sha256_file(cell / "delivery.json"),
                                      files=consumed)
            del residuals, model, checkpoint
        print(json.dumps(dict(seed=seed, completed_fitting_caches=56)), flush=True)
    _csv(output / "residual_by_block.csv", summaries)
    _csv(output / "fitting_weights_by_epoch.csv", histories)
    names = ["residual_by_block.csv", "fitting_weights_by_epoch.csv",
             *[f"seed_{seed}_coordinates.npz" for seed in (41, 42, 43)]]
    result = dict(status="COMPLETE_FITTING_POSTERIOR_RESIDUAL_DIAGNOSTIC", training_seeds=[41, 42, 43],
        source_units=100000, fitting_cases_per_seed=56, fitting_blocks_per_seed=14,
        transitions_per_case=1199, residual_block_rows=len(summaries), fitting_epoch_horizon_rows=len(histories),
        residual_definition="posterior_mean[t+1] - transition_mean(posterior_mean[t], controls[t+1])",
        centering="Matched template and exact time; mean from the other 13 fitting blocks only.",
        covariance_definition="Population covariance over episode/time within each source block and latent channel.",
        aggregate_weighting="Equal source blocks within each seed; latent coordinates are not pooled across seeds.",
        ess_definition="Per-case mean over three origins of 1/sum(softmax(-joint_event_path_nll)^2); K=8.",
        ess_one_tolerance=1e-6, ess_tolerance_is_acceptance_gate=False,
        interpretation="Posterior-mean residuals are not physical innovations. Nonlinear mean propagation and posterior uncertainty remain confounded. ESS does not measure gradient stability.",
        seed_summaries=seed_summaries, consumed=bindings,
        selection_arrays_accessed=False, validation_logs_accessed=False, source_simulation_performed=False,
        encoding_performed=False, training_performed=False, sampling_performed=False,
        scientific_claim_qualified=False, model_source_sha256=source_hash, objective_source_sha256=objective_hash,
        diagnostic_source_sha256=sha256_file(Path(__file__)), job_id=os.environ["SLURM_JOB_ID"],
        artifacts={name: sha256_file(output / name) for name in names})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.campaign, args.output)
    print(json.dumps({key: result[key] for key in ("status", "residual_block_rows", "fitting_epoch_horizon_rows")}))


if __name__ == "__main__":
    main()

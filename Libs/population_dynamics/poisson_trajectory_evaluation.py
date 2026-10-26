"""Open-loop development scores on complete held-out realization blocks.

Fields always come from generated unit counts. The latent-mean ablation changes
marginal means as well as dependence; it is not a pure covariance intervention.
Count-marginal log scores use one-dimensional Poisson-lognormal quadrature.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml

from .native_count_windows import NativeEpisodeCounts
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .poisson_trajectory_training import partition
from .provenance import sha256_file
from .trajectory_scores import energy_score, field_pairs, variogram_score


def fitting_scale(fields, cases):
    total = squared = count = metadata = None
    for case in cases:
        folder = fields / case["case_id"]
        report = json.loads((folder / "summary.json").read_text())
        if sha256_file(folder / "spatial_fields.npz") != report["artifacts"]["spatial_fields.npz"]:
            raise ValueError("fitting field view changed")
        with np.load(folder / "spatial_fields.npz", allow_pickle=False) as f:
            units = f["total_unit_count"]
            if np.any(units <= 0):
                raise ValueError("explicit handling of empty fields required")
            hz = f["count_sum"] / (units * report["settings"]["bin_ms"] / 1000)
            if total is None:
                total = np.zeros(hz.shape[1]); squared = total.copy(); count = 0
                metadata = dict(unit_count=units.copy(), field_ids=f["field_ids"].copy())
            if not np.array_equal(metadata["field_ids"], f["field_ids"]) or not np.array_equal(metadata["unit_count"], units):
                raise ValueError("cohort field identity differs")
            total += hz.sum(0); squared += (hz * hz).sum(0); count += len(hz)
    mean = total / count
    variance = squared / count - mean * mean
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("positive fitting field variation required for standardization")
    return mean, np.sqrt(variance), metadata


@torch.no_grad()
def poisson_lognormal_scores(y, mu, variance, nodes):
    """Composite per-unit/bin log score, averaged over units at each time.

    Gauss-Hermite integrates each univariate Gaussian log-intensity. This is
    neither a high-dimensional joint likelihood nor an ELBO.
    """
    roots, weights = np.polynomial.hermite.hermgauss(nodes)
    z = torch.as_tensor(roots, device=y.device, dtype=torch.float64)[:, None, None]
    logw = torch.as_tensor(np.log(weights) - .5 * np.log(np.pi), device=y.device, dtype=torch.float64)[:, None, None]
    result = torch.zeros(y.shape[0], dtype=torch.float64, device=y.device)
    for start in range(0, y.shape[1], 4096):
        sl = slice(start, start + 4096)
        obs, center, var = y[:, sl].double(), mu[:, sl].double(), variance[:, sl].double()
        if torch.any(var < 0) or not torch.isfinite(var).all():
            raise ValueError("finite nonnegative marginal log variance required")
        log_rate = center + (2 * var).sqrt() * z
        log_density = obs * log_rate - log_rate.exp() - torch.lgamma(obs + 1)
        logp = torch.logsumexp(logw + log_density, dim=0)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("nonfinite predictive marginal log score")
        result -= logp.sum(-1)
    return (result / y.shape[1]).cpu().numpy()


def evaluate(acquisition, fields, training, config_path, output):
    config = yaml.safe_load(config_path.read_text())
    run = json.loads((training / "run.json").read_text())
    if run["status"] != "FITTED_DEVELOPMENT":
        raise ValueError("completed development fit required before evaluation")
    checkpoint_path = training / "best.pt"
    if sha256_file(checkpoint_path) != run["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    binding = checkpoint["binding"]
    if (sha256_file(acquisition / "design.json") != binding["acquisition_design_sha256"]
            or sha256_file(fields / "dataset.json") != binding["dataset_sha256"]
            or sha256_file(Path(__file__).with_name("poisson_state_space.py")) != binding["model_implementation_sha256"]):
        raise ValueError("evaluation data/model binding differs from training")
    if config["methods"] != ["stochastic", "latent_mean", "independent_marginals"] or config["joint_samples"] < 2:
        raise ValueError("declared paired method comparison and joint ensemble required")
    design = json.loads((acquisition / "design.json").read_text())
    fit, validation = partition(design, binding["config"])
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", configuration=config, training_seed=binding["seed"],
                  checkpoint_sha256=run["best_checkpoint_sha256"], source_fidelity_qualified=False,
                  scientific_claim_qualified=False, formal_test_accessed=False,
                  evaluation_blocks=binding["config"]["validation_blocks"])
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    scale_mean, scale_sd, meta = fitting_scale(fields, fit)
    np.savez(output / "fitting_field_scale.npz", mean_hz=scale_mean, standard_deviation_hz=scale_sd, **meta)
    definition = [json.loads(str(v)) for v in meta["field_ids"]]
    pairs = field_pairs([r[0] for r in definition], [r[2] for r in definition], [r[1] for r in definition])
    native = NativeEpisodeCounts(acquisition / validation[0]["case_id"], fields / validation[0]["case_id"])
    ids = tuple(map(int, native.unit_ids)); model_config = binding["config"]
    model = ControlledPoissonLDS(ids, 1, model_config["latent_size"], model_config["encoder_size"],
                                model_config["bin_ms"] / 1000, initialization_seed=binding["seed"]).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    torch.set_num_threads(model_config["cpu_threads"]); torch.use_deterministic_algorithms(True)
    model.eval(); del checkpoint, native
    horizons = [int(round(ms / model_config["bin_ms"])) for ms in config["horizons_ms"]]
    if any(h * model_config["bin_ms"] != ms for h, ms in zip(horizons, config["horizons_ms"])):
        raise ValueError("aligned forecast horizons required")
    longest, draws = max(horizons), config["joint_samples"]
    rows = []; started = time.perf_counter()
    with torch.no_grad():
        for case_index, case in enumerate(validation):
            data = NativeEpisodeCounts(acquisition / case["case_id"], fields / case["case_id"])
            native_report = json.loads((acquisition / case["case_id"] / "report.json").read_text())
            parameter_path = acquisition / case["case_id"] / "initial_parameters.npz"
            if sha256_file(parameter_path) != native_report["artifacts"]["initial_parameters.npz"]:
                raise ValueError("source refractory parameters changed")
            with np.load(parameter_path, allow_pickle=False) as parameters:
                if not np.array_equal(parameters["unit_ids"], data.unit_ids) or np.any(parameters["t_ref"] <= 0):
                    raise ValueError("valid source per-unit refractory periods required")
                count_bound = torch.as_tensor(np.ceil(model_config["bin_ms"] / parameters["t_ref"]), device="cuda")
            count, stimulus = data.window(0, data.duration_ticks)
            y = torch.as_tensor(count, device="cuda"); u = torch.as_tensor(stimulus, device="cuda")
            unit_field = torch.as_tensor(data.unit_field_index, device="cuda")
            for origin_index, origin in enumerate(config["origins_ms"]):
                boundary = int(round(origin / model_config["bin_ms"]))
                if boundary < 1 or boundary + longest > len(y) or boundary * model_config["bin_ms"] != origin:
                    raise ValueError("complete aligned causal prefix and forecast required")
                t = time.perf_counter()
                state = model.infer_state({"counts": y[None, :boundary], "stimulus": u[None, :boundary]})
                future = u[None, boundary:boundary + longest]
                observed = y[boundary:boundary + longest]
                moments = model.forecast_moments(state, future)
                mu = moments.mean[0] @ model.loading.T + model.log_rate_bias + math.log(model.bin_seconds)
                variance = torch.einsum("nd,tde,ne->tn", model.loading, moments.covariance[0], model.loading)
                observed_field = torch.zeros((longest, len(meta["field_ids"])), dtype=torch.float32, device="cuda")
                observed_field.index_add_(-1, unit_field, observed)
                raw_truth = observed_field.cpu().numpy()
                truth_hz = raw_truth / (meta["unit_count"] * model.bin_seconds)
                truth_scaled = (truth_hz - scale_mean) / scale_sd
                torch.cuda.synchronize(); shared_seconds = time.perf_counter() - t
                seed_base = config["sampling_seed_base"] + binding["seed"] * 100000 + case_index * 100 + origin_index * 3
                marginal_nll = poisson_lognormal_scores(observed, mu, variance, config["quadrature_nodes"])
                for method in config["methods"]:
                    first_row = len(rows)
                    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
                    streams = LatentRandomStreams.from_seeds(seed_base, seed_base + 1, seed_base + 2, device=model.loading.device)
                    if method == "stochastic":
                        sample = model.simulate(state, future, streams, draws)
                        generated = sample.counts[:, 0]
                        latent = sample.latent_states[:, 0].cpu().numpy()
                        del sample
                        expectation = moments.expected_counts[0]
                    elif method == "latent_mean":
                        expectation = mu.exp()
                        generated = torch.poisson(expectation.expand(draws, -1, -1), generator=streams.observation).to(torch.int64)
                        latent = moments.mean[0].cpu().numpy()[None]
                    else:
                        # The same univariate Gaussian log-intensities, sampled
                        # independently per unit/bin, isolate dependence changes.
                        log_intensity = (mu + variance.sqrt() * torch.randn(
                            (draws, *mu.shape), dtype=mu.dtype, device=mu.device,
                            generator=streams.dynamics))
                        intensity = log_intensity.exp()
                        if not torch.isfinite(intensity).all():
                            raise FloatingPointError("nonfinite independent-marginal intensity")
                        sampled = torch.poisson(intensity, generator=streams.observation)
                        if not torch.isfinite(sampled).all() or torch.any(sampled >= torch.iinfo(torch.int64).max):
                            raise FloatingPointError("independent count sample outside int64")
                        generated = sampled.to(torch.int64)
                        expectation = moments.expected_counts[0]
                        latent = np.empty((0,), dtype=np.float32)
                        del log_intensity, intensity, sampled
                    generated_field = torch.zeros((draws, longest, len(meta["field_ids"])), dtype=torch.int64, device="cuda")
                    generated_field.index_add_(-1, unit_field, generated)
                    raw_fields = generated_field.cpu().numpy()
                    sampled_hz = raw_fields / (meta["unit_count"] * model.bin_seconds)
                    sample_scaled = (sampled_hz - scale_mean) / scale_sd
                    torch.cuda.synchronize(); generation_seconds = time.perf_counter() - t
                    if method in ("stochastic", "independent_marginals"):
                        nll_by_bin = marginal_nll
                    else:
                        nll_by_bin = (expectation - observed * mu + torch.lgamma(observed + 1)).mean(-1).cpu().numpy()
                    artifact = f"{case['case_id']}_origin_{origin:g}_{method}.npz"
                    np.savez_compressed(output / artifact, observed_field_counts=raw_truth,
                                        sampled_field_counts=raw_fields, latent_states=latent)
                    for milliseconds, horizon in zip(config["horizons_ms"], horizons):
                        samples = sample_scaled[:, :horizon]; target = truth_scaled[:horizon]
                        total_counts = generated[:, :horizon].sum(1).float()
                        quantiles = torch.quantile(total_counts, torch.tensor([.05, .95], device="cuda"), dim=0)
                        actual_total = observed[:horizon].sum(0)
                        row = dict(seed=binding["seed"], block=case["block"], case_id=case["case_id"],
                                   template_id=case["template_id"], origin_ms=origin, horizon_ms=milliseconds,
                                   method=method, energy_score=energy_score(samples, target),
                                   composite_marginal_nll=float(nll_by_bin[:horizon].mean()),
                                   unit_bin_mean_mse=float((expectation[:horizon] - observed[:horizon]).square().mean()),
                                   expected_spikes_per_unit=float(expectation[:horizon].sum(0).mean()),
                                   unit_total_coverage_90=float(((actual_total >= quantiles[0]) & (actual_total <= quantiles[1])).float().mean()),
                                   unit_total_interval_width_90=float((quantiles[1] - quantiles[0]).mean()),
                                   refractory_count_bound_violation=float((generated[:, :horizon] > count_bound).float().mean()),
                                   shared_inference_seconds=shared_seconds, full_500ms_unit_generation_seconds=generation_seconds,
                                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), ensemble_artifact=artifact)
                        for name, pair in pairs.items():
                            row["variogram_" + name] = variogram_score(samples, target, pair[:, 0], pair[:, 1])
                        for lag in config["temporal_lags_bins"]:
                            row[f"variogram_temporal_lag_{lag}"] = (variogram_score(samples, target, np.arange(target.shape[1]),
                                                                                 np.arange(target.shape[1]), lag)
                                                                     if lag < horizon else None)
                        rows.append(row)
                    score_path = output / "scores.csv"
                    existed = score_path.exists()
                    with score_path.open("a", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        if not existed:
                            writer.writeheader()
                        writer.writerows(rows[first_row:])
                    del generated, generated_field
                print(json.dumps(dict(status="EVALUATING", case=case["case_id"], origin_ms=origin, score_rows=len(rows))), flush=True)
            del y, u
    record.update(status="EVALUATED_DEVELOPMENT", score_rows=len(rows), episodes=len(validation),
                  independent_blocks=len({c["block"] for c in validation}),
                  seconds=time.perf_counter() - started, scores_sha256=sha256_file(output / "scores.csv"),
                  inference="descriptive on epoch-selection blocks; no independent efficacy claim",
                  all_unit_outputs_generated=True, field_outputs_from_unit_counts=True,
                  quadrature_interpretation="univariate composite marginal score, not joint likelihood",
                  latent_mean_interpretation="both dependence and lognormal marginal means change",
                  independent_marginal_interpretation="same unit/bin marginals, independently sampled latent log-intensities")
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "training", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    try:
        print(json.dumps(evaluate(a.acquisition, a.fields, a.training, a.config, a.output)))
    except Exception as error:
        if a.output.is_dir():
            (a.output / "failure.json").write_text(json.dumps(dict(status="FAILED", error=str(error),
                traceback=traceback.format_exc()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()

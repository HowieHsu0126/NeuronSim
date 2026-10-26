"""Fitting-internal stochastic rollout qualification of frozen ridge corrections.

Only correction fitting excludes blocks 12 and 13. The frozen parent previously
trained on all original fitting blocks 0..13 and was selected on blocks 14..17.
This module never loads original selection arrays. Its target is the cached
causal posterior mean, not a native observation or a physical latent truth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from .population_prior_residual_diagnostic import _csv
from .population_prior_residual_prediction import _coordinates, validate_config
from .population_residual_memory import ResidualMemoryPopulationState
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file
from .residual_prediction import fit_residual_predictor

SEEDS = [41, 42, 43]
ALPHAS = [0.0001, 0.01, 1.0, 100.0]
FIT_BLOCKS = list(range(12))
QUALIFICATION_BLOCKS = [12, 13]
ORIGINS_MS = [1000, 3000, 5000]
HORIZONS_MS = [20, 80, 500]
BIN_MS, STEPS, DRAWS, COORDINATES = 5, 100, 16, 512
SOURCE_UNITS, ORIGIN_COUNT = 100000, 24
CANDIDATES = [("parent", "parent", None)] + [
    (f"{mode}_alpha_{format(alpha, 'g').replace('.', 'p')}", mode, alpha)
    for mode in ("current", "history") for alpha in ALPHAS
]


def posterior_coordinate_energy_score(samples, target, scale):
    """Return a trajectory energy score with an unbiased off-diagonal pair term."""
    samples, target, scale = (
        torch.as_tensor(value, dtype=torch.float64) for value in (samples, target, scale)
    )
    if (samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 1 or samples.shape[2] < 1
            or target.shape != samples.shape[1:] or scale.shape != samples.shape[2:]
            or not all(torch.isfinite(value).all() for value in (samples, target, scale))
            or torch.any(scale <= 0)):
        raise ValueError("finite draw/time/coordinate trajectories and positive coordinate scales required")
    normalizer = math.sqrt(samples.shape[1] * samples.shape[2])
    standardized = (samples / scale).flatten(1)
    target_distance = torch.linalg.vector_norm(((samples-target) / scale).flatten(1), dim=1).mean() / normalizer
    # pdist contains each unordered pair once; its mean equals the ordered
    # off-diagonal mean. Half that mean is the unbiased energy-score pair term.
    pair_term = torch.pdist(standardized).mean() / (2 * normalizer)
    score = target_distance - pair_term
    if not all(torch.isfinite(value) for value in (target_distance, pair_term, score)):
        raise ValueError("posterior-coordinate trajectory score is not finite")
    return dict(posterior_coordinate_energy_score=float(score),
                mean_target_distance=float(target_distance), half_mean_offdiagonal_pair_distance=float(pair_term))


def _origin_order():
    return [dict(block=block, template=template, case_id=f"train_block_{block:03d}_familiar_{template}",
                 origin_ms=origin_ms, origin_bin=origin_ms // BIN_MS)
            for block in QUALIFICATION_BLOCKS for template in range(4) for origin_ms in ORIGINS_MS]


def _finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _stage_qualification(campaign, seed, scratch, source, parent):
    cell = campaign / f"seed_{seed}"
    delivery_path = cell / "delivery.json"
    if sha256_file(delivery_path) != source["delivery_sha256"]:
        raise ValueError("source delivery changed after coordinate authentication")
    delivery = json.loads(delivery_path.read_text())
    files = {row["path"]: row for row in delivery["files"]}
    episodes, staged = {}, {}
    for block in QUALIFICATION_BLOCKS:
        for template in range(4):
            case = f"train_block_{block:03d}_familiar_{template}"
            relative = f"training/posterior_cache/{case}.pt"
            target = scratch / "qualification" / f"{case}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cell / relative, target)
            digest = sha256_file(target)
            if (target.stat().st_size != files[relative]["bytes"]
                    or digest != files[relative]["sha256"] or digest != source["files"][relative]):
                raise ValueError("qualification cache differs from consumed source: " + case)
            saved = torch.load(target, map_location="cpu", weights_only=True)
            mean, variance, controls = (parent._tensor(saved[key]) for key in ("mean", "variance", "controls"))
            expected = (1, 1200, len(parent.field_names), parent.latent_channels)
            if (saved["binding"]["case_id"] != case or mean.shape != expected or variance.shape != expected
                    or controls.shape != (1, 1200, parent.input_size)
                    or not all(torch.isfinite(value).all() for value in (mean, variance, controls))
                    or torch.any(variance <= 0)):
                raise ValueError("complete finite qualification posterior and controls required: " + case)
            episodes[case] = dict(mean=mean, variance=variance, controls=controls)
            staged[relative] = digest
    return episodes, staged


def _rollout_origin(parent, correction, episode, origin, coordinate_scale, initial_rng, dynamics_rng):
    start = origin["origin_bin"]
    mean, variance, controls = (episode[key] for key in ("mean", "variance", "controls"))
    boundary_mean, boundary_variance = mean[:, start-1], variance[:, start-1]
    memory = boundary_mean - parent.transition_mean(mean[:, start-2], controls[:, start-1])
    future, target = controls[:, start:start+STEPS], mean[0, start:start+STEPS].reshape(STEPS, COORDINATES)
    shape = (DRAWS, *boundary_mean.shape)
    z = boundary_mean + boundary_variance.sqrt() * torch.randn(
        shape, dtype=boundary_mean.dtype, device=boundary_mean.device, generator=initial_rng)
    h = memory.expand(DRAWS, -1, -1, -1)
    process = parent._field_parameters()[3].exp()
    if not torch.isfinite(process).all() or torch.any(process <= 0):
        raise ValueError("unchanged parent process variance must be finite and positive")
    process_scale = process.sqrt()
    samples = torch.empty((DRAWS, STEPS, COORDINATES), dtype=z.dtype, device=z.device)
    scores, failure, completed_steps = {}, None, 0
    for step in range(STEPS):
        base = parent.transition_mean(z, future[:, step])
        if correction is None:
            z = base + process_scale * torch.randn(shape, dtype=z.dtype, device=z.device, generator=dynamics_rng)
        else:
            z = base + correction.correction(z, future[:, step], h) + process_scale * torch.randn(
                shape, dtype=z.dtype, device=z.device, generator=dynamics_rng)
        h = z - base
        log_hazard = parent.log_hazard(z)
        hazard = log_hazard.exp() * parent.tick_seconds
        invalid = (~torch.isfinite(hazard)) | (hazard <= 0)
        if invalid.any() or not torch.isfinite(z).all():
            failure = dict(stage="native_hazard", step=step+1, elapsed_ms=(step+1)*BIN_MS,
                target_bin=start+step, latent_nonfinite=int((~torch.isfinite(z)).sum()),
                hazard_nonfinite=int((~torch.isfinite(hazard)).sum()),
                hazard_zero_or_negative=int((hazard <= 0).sum()),
                latent_absolute_max=_finite_scalar(z.abs().max()))
            if invalid.any():
                flat = int(invalid.flatten().to(torch.int8).argmax())
                unit = flat % SOURCE_UNITS
                draw = flat // SOURCE_UNITS
                failure.update(first_invalid_draw=draw, first_invalid_batch=0, first_invalid_unit_index=unit,
                    first_invalid_unit_id=int(parent.unit_ids[unit]),
                    first_invalid_log_hazard=_finite_scalar(log_hazard[draw, 0, unit]),
                    first_invalid_hazard=_finite_scalar(hazard[draw, 0, unit]))
            break
        samples[:, step] = z[:, 0].flatten(-2)
        completed_steps = step + 1
        horizon_ms = completed_steps * BIN_MS
        if horizon_ms in HORIZONS_MS:
            try:
                scores[str(horizon_ms)] = posterior_coordinate_energy_score(
                    samples[:, :completed_steps], target[:completed_steps], coordinate_scale)
            except ValueError as error:
                failure = dict(stage="posterior_coordinate_score", step=completed_steps,
                               elapsed_ms=horizon_ms, target_bin=start+step, message=str(error))
                break
    return dict(**origin, status="COMPLETE" if completed_steps == STEPS and failure is None else "FAILED",
                completed_steps=completed_steps, scores=scores, failure=failure)


def _run_candidate(seed, candidate, mode, alpha, parent, correction, episodes, scale,
                   fit_failure, fit_seconds):
    started = time.perf_counter()
    # Each candidate uses the same two streams. Neither is restarted at a new
    # origin: fixed case/template/origin order defines the common random draws.
    initial_rng = torch.Generator(device="cpu").manual_seed(41)
    dynamics_rng = torch.Generator(device="cpu").manual_seed(42)
    first_failure, origins, rows = fit_failure, [], []
    for position, origin in enumerate(_origin_order()):
        if first_failure is not None:
            item = dict(**origin, status="NOT_RUN", completed_steps=0, scores={}, failure=None)
        else:
            item = _rollout_origin(parent, correction, episodes[origin["case_id"]], origin,
                                   scale, initial_rng, dynamics_rng)
            if item["failure"] is not None:
                first_failure = dict(origin_index=position, **origin, **item["failure"])
        origins.append(item)
        for horizon in HORIZONS_MS:
            score = item["scores"].get(str(horizon))
            rows.append(dict(seed=seed, candidate=candidate, mode=mode, alpha=alpha,
                block=origin["block"], template=origin["template"], case_id=origin["case_id"],
                origin_ms=origin["origin_ms"], horizon_ms=horizon, origin_status=item["status"],
                score_status="SCORED" if score is not None else ("NOT_RUN" if item["status"] == "NOT_RUN" else "NOT_REACHED"),
                completed_steps=item["completed_steps"],
                posterior_coordinate_energy_score=None if score is None else score["posterior_coordinate_energy_score"],
                mean_target_distance=None if score is None else score["mean_target_distance"],
                half_mean_offdiagonal_pair_distance=None if score is None else score["half_mean_offdiagonal_pair_distance"]))
    complete_count = sum(item["status"] == "COMPLETE" for item in origins)
    complete = complete_count == ORIGIN_COUNT and first_failure is None
    score_counts = {str(h): sum(row["horizon_ms"] == h and row["score_status"] == "SCORED" for row in rows)
                    for h in HORIZONS_MS}
    means = {str(h): (float(np.mean([row["posterior_coordinate_energy_score"] for row in rows
                                   if row["horizon_ms"] == h])) if complete else None) for h in HORIZONS_MS}
    record = dict(seed=seed, candidate=candidate, mode=mode, alpha=alpha,
        status="COMPLETE" if complete else "FAILED", completed_origins=complete_count,
        attempted_origins=sum(item["status"] != "NOT_RUN" for item in origins),
        not_run_origins=sum(item["status"] == "NOT_RUN" for item in origins),
        first_failure=first_failure, mean_posterior_coordinate_energy_score=means,
        scored_origin_counts=score_counts, origins=origins, fitting_seconds=fit_seconds,
        rollout_seconds=time.perf_counter()-started)
    return record, rows


def aggregate_candidates(records):
    """Select one alpha per mode using only complete coverage in all three seeds."""
    lookup = {(row["seed"], row["candidate"]): row for row in records}
    expected = {(seed, candidate) for seed in SEEDS for candidate, _, _ in CANDIDATES}
    if len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("one record for every seed and candidate is required")
    summaries = []
    for candidate, mode, alpha in CANDIDATES:
        runs = [lookup[seed, candidate] for seed in SEEDS]
        eligible = all(run["status"] == "COMPLETE" and run["completed_origins"] == ORIGIN_COUNT
            and all(run["scored_origin_counts"][str(h)] == ORIGIN_COUNT
                    and run["mean_posterior_coordinate_energy_score"][str(h)] is not None
                    and math.isfinite(run["mean_posterior_coordinate_energy_score"][str(h)])
                    for h in HORIZONS_MS) for run in runs)
        summaries.append(dict(candidate=candidate, mode=mode, alpha=alpha, eligible=eligible,
            per_seed_status={str(run["seed"]): run["status"] for run in runs},
            mean_posterior_coordinate_energy_score={str(h): (float(np.mean([
                run["mean_posterior_coordinate_energy_score"][str(h)] for run in runs])) if eligible else None)
                for h in HORIZONS_MS}))
    parent = summaries[0]
    for row in summaries:
        row["delta_500ms_vs_parent"] = (
            row["mean_posterior_coordinate_energy_score"]["500"] - parent["mean_posterior_coordinate_energy_score"]["500"]
            if row["eligible"] and parent["eligible"] else None)
    selected = {}
    for mode in ("current", "history"):
        eligible = [row for row in summaries if row["mode"] == mode and row["eligible"]]
        if eligible:
            winner = min(eligible, key=lambda row: (row["mean_posterior_coordinate_energy_score"]["500"], row["alpha"]))
            selected[mode] = dict(status="SELECTED", alpha=winner["alpha"], candidate=winner["candidate"],
                mean_500ms_posterior_coordinate_energy_score=winner["mean_posterior_coordinate_energy_score"]["500"],
                delta_500ms_vs_parent=winner["delta_500ms_vs_parent"])
        else:
            selected[mode] = dict(status="NO_ELIGIBLE_CANDIDATE", alpha=None, candidate=None,
                                 mean_500ms_posterior_coordinate_energy_score=None, delta_500ms_vs_parent=None)
    return dict(candidate_summaries=summaries, selected_alphas=selected)


@torch.no_grad()
def qualify(campaign, diagnostic_path, output):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only admitted Slurm allocation required")
    campaign, diagnostic_path, output = map(Path, (campaign, diagnostic_path, output))
    diagnostic = json.loads(diagnostic_path.read_text())
    protocol = diagnostic["config"]
    validate_config(protocol)
    if (diagnostic.get("status") != "COMPLETE_FITTING_BLOCK_RESIDUAL_PREDICTION"
            or protocol["ridge_alphas"] != ALPHAS or protocol["source_units"] != SOURCE_UNITS):
        raise ValueError("accepted original fitting diagnostic and unchanged four-alpha grid required")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    diagnostic_hash = sha256_file(diagnostic_path)
    records, score_rows, sources, scales, artifacts = [], [], {}, {}, {}
    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"residual-rollout-qualification-{seed}-") as directory:
            scratch = Path(directory)
            X, Y, _, source = _coordinates(campaign, seed, scratch, protocol)
            if source != diagnostic["consumed"][str(seed)]:
                raise ValueError("fitting inputs differ from the accepted diagnostic")
            checkpoint_path = scratch / "training/best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            parent = ControlledPopulationState(**checkpoint["model_configuration"])
            parent.load_state_dict(checkpoint["model"], strict=True)
            parent.eval()
            current_width = source["current_width"]
            if (len(parent.unit_ids) != SOURCE_UNITS or len(parent.field_names)*parent.latent_channels != COORDINATES
                    or parent.bin_seconds != BIN_MS/1000 or current_width != COORDINATES+parent.input_size
                    or X.shape != (14, 4, 1198, current_width+COORDINATES)
                    or Y.shape != (14, 4, 1198, COORDINATES)):
                raise ValueError("complete 100000-unit 512-coordinate fitting campaign required")
            episodes, staged = _stage_qualification(campaign, seed, scratch, source, parent)
            sources[str(seed)] = dict(authenticated_coordinates=source, qualification_caches=staged)
            models, fit_failures, fit_seconds, scale = {}, {}, {}, None
            for candidate, mode, alpha in CANDIDATES[1:]:
                fit_start = time.perf_counter()
                print(json.dumps(dict(stage="fit", seed=seed, candidate=candidate)), flush=True)
                try:
                    predictor = fit_residual_predictor(X[:12], Y[:12], current_width if mode == "current" else X.shape[-1], alpha)
                    fitted_scale = predictor.x_scale[:COORDINATES].copy()
                    if scale is None:
                        scale = fitted_scale
                    elif not np.array_equal(scale, fitted_scale):
                        raise ValueError("training-only coordinate scales differ across candidates")
                    models[candidate] = ResidualMemoryPopulationState(parent, predictor, mode,
                        dict(seed=seed, alpha=alpha, fitting_blocks=FIT_BLOCKS, qualification_blocks=QUALIFICATION_BLOCKS,
                             fitting_diagnostic_sha256=diagnostic_hash,
                             parent_checkpoint_sha256=source["files"]["training/best.pt"])).eval()
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    fit_failures[candidate] = dict(stage="fit", error_type=type(error).__name__, message=str(error))
                fit_seconds[candidate] = time.perf_counter()-fit_start
            scales[str(seed)] = None if scale is None else scale.tolist()
            del X, Y
            for candidate, mode, alpha in CANDIDATES:
                failure = fit_failures.get(candidate)
                if scale is None and failure is None:
                    failure = dict(stage="normalization", message="No fitting predictor supplied a training-only coordinate scale")
                record, rows = _run_candidate(seed, candidate, mode, alpha, parent, models.get(candidate),
                    episodes, scale, failure, fit_seconds.get(candidate, 0.0))
                records.append(record)
                score_rows.extend(rows)
                path = output / "candidates" / f"seed_{seed}_{candidate}.json"
                freeze_json(path, record)
                artifacts[str(path.relative_to(output))] = sha256_file(path)
                print(json.dumps({key: record[key] for key in ("seed", "candidate", "status", "completed_origins", "first_failure")}), flush=True)
            if (not all(torch.equal(value, checkpoint["model"][key]) for key, value in parent.state_dict().items())
                    or sha256_file(checkpoint_path) != source["files"]["training/best.pt"]):
                raise ValueError("frozen parent parameters or checkpoint changed during qualification")
            del models, parent, checkpoint, episodes
    _csv(output / "score_rows.csv", score_rows)
    artifacts["score_rows.csv"] = sha256_file(output / "score_rows.csv")
    aggregation = aggregate_candidates(records)
    result = dict(status="COMPLETE_FITTING_INTERNAL_ROLLOUT_QUALIFICATION", training_seeds=SEEDS,
        correction_fitting_blocks=FIT_BLOCKS, correction_qualification_blocks=QUALIFICATION_BLOCKS,
        original_parent_training_blocks=list(range(14)), original_parent_selection_blocks=list(range(14, 18)),
        qualification_is_independent_parent_test=False, original_selection_arrays_accessed=False,
        source_units=SOURCE_UNITS, posterior_coordinates=COORDINATES, ridge_alphas=ALPHAS,
        qualification_episodes_per_seed=8, origins_ms=ORIGINS_MS, origins_per_candidate=ORIGIN_COUNT,
        horizons_ms=HORIZONS_MS, draws=DRAWS, steps=STEPS, bin_ms=BIN_MS,
        candidate_order=[candidate for candidate, _, _ in CANDIDATES], origin_order=_origin_order(),
        noise_stream_seeds=dict(initial=41, dynamics=42),
        stream_policy="Reset once per candidate and seed; advance continuously across the fixed 24-origin order; stop at first invalid origin.",
        original_initial_and_process_variances=True, parent_parameters_and_checkpoint_unchanged=True,
        clipping_or_stabilization_applied=False, native_hazard_guard="exp(log_hazard)*tick_seconds finite and strictly positive at all 100000 units on every generated step",
        observation_sampling_performed=False, generated_native_count_evaluation=False,
        target="Held-out-from-correction-fit cached causal posterior MEAN trajectory; posterior-coordinate proxy only.",
        coordinate_scale="Current-coordinate x_scale from correction fitting blocks0..11 only; no qualification-coordinate normalization.",
        coordinate_scales=scales,
        score_definition="Mean draw-to-target Euclidean trajectory distance minus half the unbiased off-diagonal draw-pair mean distance; each distance divided by sqrt(time_bins*512).",
        alpha_selection="Common alpha per mode among candidates complete at all24origins in all3seeds; smallest mean500ms proxy score; smaller alpha resolves exact ties.",
        aggregation="Mean across24origins within each seed, then equal mean across3training seeds. No significance test or confidence interval.",
        native_fidelity_qualified=False, biological_claim_qualified=False, scientific_claim_qualified=False,
        limitation="Frozen parent already used all14original fitting blocks and original selection blocks14..17. Qualification blocks12/13 only exclude the new correction fit. One connectivity instance and posterior-mean proxy scores do not establish native fidelity.",
        fitting_only_diagnostic_sha256=diagnostic_hash, consumed=sources, runs=records,
        **aggregation, score_rows=len(score_rows), elapsed_seconds=time.perf_counter()-started,
        job_id=os.environ["SLURM_JOB_ID"],
        software_sources={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "population_residual_rollout_qualification.py", "population_residual_memory.py", "residual_prediction.py",
            "population_prior_residual_prediction.py", "population_prior_residual_diagnostic.py", "population_state.py",
            "population_observation_prior.py", "provenance.py")}, artifacts=artifacts)
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "diagnostic", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.campaign, args.diagnostic, args.output)
    print(json.dumps(dict(status=result["status"], selected_alphas=result["selected_alphas"])), flush=True)


if __name__ == "__main__":
    main()

"""Finite-particle native residual NLL gradients on one fixed fitting episode."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import yaml

from . import population_prior_particle_diagnostic as legacy
from .population_prior_particle_diagnostic import BUDGETS, CASE, REPLICATES, _cosine, nested_objective
from .population_residual_observation import configure_correction_parameters, residual_observation_mixture_loss
from .population_residual_observation_fitting import load_batch, load_native_observation_correction, validate_config, write_json
from .population_residual_mean_objective import residual_mean_objective
from .population_residual_observation_training import MEAN_OBJECTIVE
from .population_residual_observation_validation import load_seed
from .population_residual_refit_provenance import build_refit_receipt
from .provenance import freeze_json, sha256_file
from .renewal_trajectory_training import variational_seed

PARAMETERS = ("correction_weight", "correction_bias")
SEEDS = (41, 42, 43)
MODES = ("current", "history")


def probe(model, inputs, origins, horizons, initial_seed, dynamics_seed, particles,
          maximum_particles=32, unit_chunk_size=4096):
    """Reuse the legacy nested draw layout with the unchanged native residual loss."""
    parameters = configure_correction_parameters(model)
    device = model.correction_weight.device
    initial = torch.Generator(device=device).manual_seed(initial_seed)
    dynamics = torch.Generator(device=device).manual_seed(dynamics_seed)
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    objective = partial(residual_observation_mixture_loss, unit_chunk_size=unit_chunk_size,
                        checkpoint_readout=True)
    terms = nested_objective(model, inputs, origins, horizons, initial, dynamics,
                             particles=particles, maximum_particles=maximum_particles,
                             objective=objective)
    gradients = torch.autograd.grad(terms["loss"], parameters)
    if any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("nonfinite native residual likelihood gradient")
    vector = torch.cat([value.detach().double().reshape(-1) for value in gradients]).cpu().numpy()
    norm = float(np.linalg.norm(vector))
    row = dict(status="COMPLETE", particles=particles,
        **{name: float(value.detach()) for name, value in terms.items()},
        gradient_norm=norm, gradient_finite=True, gradient_nonzero=norm > 0,
        seconds=time.perf_counter() - started)
    row.update({name + "_gradient_norm": float(value.detach().double().norm())
                for name, value in zip(PARAMETERS, gradients)})
    return row, vector


def summarize(rows, vectors):
    """The same per-budget variation and paired-budget statistics as the parent probe."""
    result = []
    for budget in BUDGETS:
        selected = [row for row in rows if row["particles"] == budget]
        matrix = np.stack([vectors[budget, replicate] for replicate in REPLICATES])
        result.append(dict(particles=budget,
            loss_mean=float(np.mean([row["loss"] for row in selected])),
            loss_sd=float(np.std([row["loss"] for row in selected], ddof=1)),
            pairwise_gradient_cosines=[_cosine(matrix[i], matrix[j])
                for i in REPLICATES for j in REPLICATES if i < j],
            centered_gradient_rms=float(np.sqrt(np.mean(np.sum((matrix-matrix.mean(0))**2, axis=1)))),
            mean_gradient_norm=float(np.linalg.norm(matrix.mean(0))),
            paired_8_vs_32_gradient_cosines=[_cosine(vectors[8, r], vectors[32, r]) for r in REPLICATES]))
    return result


def unchanged(module, before):
    current = module.state_dict()
    return set(current) == set(before) and all(
        torch.equal(current[name].detach().cpu(), value) for name, value in before.items())


MEAN_CASES = tuple(f"train_block_000_familiar_{index}" for index in range(4))
MEAN_COMPONENTS = ("loss", "native_loss", "field_mean_loss")


def mean_conditions():
    """The four fitting templates retain all three seeds and named replicates."""
    return [dict(seed=seed, mode=mode, case_id=case, replicate=replicate, particles=budget)
            for seed in SEEDS for mode in MODES for case in MEAN_CASES
            for replicate in REPLICATES for budget in BUDGETS]


def mean_probe(model, batch, initial_seed, dynamics_seed, particles,
               maximum_particles=32, unit_chunk_size=4096, mean_unit_chunk_size=32768):
    """Separate gradients of the unchanged unpenalized mean-selection objective."""
    parameters = configure_correction_parameters(model)
    device = model.correction_weight.device
    initial = torch.Generator(device=device).manual_seed(initial_seed)
    dynamics = torch.Generator(device=device).manual_seed(dynamics_seed)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_rng = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    try:
        inputs = tuple(batch[name] for name in
                       ("mean", "variance", "controls", "target_counts", "target_eligible_ticks"))
        objective = partial(residual_mean_objective, unit_chunk_size=unit_chunk_size,
            mean_unit_chunk_size=mean_unit_chunk_size, checkpoint_readout=True,
            initial_ready_in_ticks=batch["initial_ready_in_ticks"],
            field_standard_deviation_hz=batch["field_standard_deviation_hz"])
        terms = nested_objective(model, inputs, batch["origins"], batch["horizons"], initial, dynamics,
            particles=particles, maximum_particles=maximum_particles, objective=objective,
            reconstructs_paths=True)
        gradients = {}
        for index, name in enumerate(MEAN_COMPONENTS):
            gradients[name] = torch.autograd.grad(terms[name], parameters,
                                                  retain_graph=index < len(MEAN_COMPONENTS) - 1)
            if any(not torch.isfinite(value).all() for value in gradients[name]):
                raise FloatingPointError("nonfinite mean-objective component gradient: " + name)
        for total, native, mean in zip(*(gradients[name] for name in MEAN_COMPONENTS)):
            torch.testing.assert_close(total, native + mean)
        vectors = {name: torch.cat([value.detach().double().reshape(-1) for value in values]).cpu().numpy()
                   for name, values in gradients.items()}
        row = dict(status="COMPLETE", particles=particles,
            **{name: float(value.detach()) for name, value in terms.items()},
            component_gradient_norms={name: float(np.linalg.norm(value)) for name, value in vectors.items()},
            native_mean_gradient_cosine=_cosine(vectors["native_loss"], vectors["field_mean_loss"]),
            total_component_sum_verified=True, native_rng_reconstruction_verified=True,
            gradient_finite=True, gradient_nonzero=bool(np.linalg.norm(vectors["loss"]) > 0),
            gradient_norm=float(np.linalg.norm(vectors["loss"])), seconds=time.perf_counter() - started)
        if any(not np.isfinite(row[name]) for name in terms):
            raise FloatingPointError("nonfinite mean-objective component value")
        return row, vectors
    finally:
        if not all(torch.equal(model.state_dict()[name], value) for name, value in before.items()):
            raise RuntimeError("mean-gradient probe changed model state")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("mean-gradient probe populated a persistent parameter gradient")
        if not torch.equal(torch.get_rng_state(), original_rng):
            raise RuntimeError("mean-gradient probe changed the global CPU random stream")


def summarize_components(rows, vectors):
    """Use the existing nested-budget summaries for each loss component."""
    return {name: summarize([dict(row, loss=row[name]) for row in rows],
                            {key: value[name] for key, value in vectors.items()})
            for name in MEAN_COMPONENTS}


def _diagnose_mean(inputs, refits, config_path, output, device, config, family):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    report = dict(status="RUNNING", exit_code=None, run_id=os.environ["RH2_JOB_RUN_ID"],
        trial_id=os.environ["ARNOLD_TRIAL_ID"], host=platform.node(),
        started_utc=datetime.now(timezone.utc).isoformat(), device=str(device),
        training_seeds=list(SEEDS), modes=list(MODES), case_ids=list(MEAN_CASES),
        particle_budgets=list(BUDGETS), replicate_indices=list(REPLICATES),
        expected_objective_evaluations=len(mean_conditions()), source_episode_blocks=[0],
        origins_bins=config["origins"], horizons_bins=config["horizons"],
        source_units=100000, fields=128, channels=4, objective=MEAN_OBJECTIVE,
        parameters=list(PARAMETERS), parameters_updated=False, optimizer_steps=0,
        regularization_included=False, raw_gradient_vectors_retained=False,
        generated_unit_samples=False, validation_outcomes_accessed=False, excluded_blocks_accessed=False,
        nested_draw_layout="maximumK32 per step; K8 is the same first eight paths, including exact renewal-mean reconstruction",
        stream_scope="three diagnostic-only named replicates per seed/case; paired across budgets/modes",
        input_manifest_sha256=family["shared_binding"]["input_manifest_sha256"],
        config_sha256=sha256_file(config_path), refit_receipt=family,
        diagnostic_sha256=sha256_file(Path(__file__)), sampler_sha256=sha256_file(Path(legacy.__file__)),
        scientific_claim_qualified=False, significance_qualified=False, sota_qualified=False,
        records=[], horizon_records=[], cells=[])
    parent = model = checkpoint = None
    loaded_seed = loaded_mode = None
    for seed in SEEDS:
        for mode in MODES:
            for case in MEAN_CASES:
                rows, vectors, before, setup_error = [], {}, None, None
                try:
                    if loaded_seed != seed:
                        parent, pilot, _ = load_seed(inputs, seed)
                        del pilot
                        checkpoint = torch.load(inputs / f"parent/seed_{seed}/training/best.pt",
                                                map_location="cpu", weights_only=True)
                        parent.to(device)
                        loaded_seed, loaded_mode = seed, None
                    if loaded_mode != mode:
                        model = load_native_observation_correction(refits / f"seed_{seed}" / mode,
                            parent, checkpoint["binding"],
                            sha256_file(inputs / f"parent/seed_{seed}/training/best.pt")).to(device).eval()
                        loaded_mode = mode
                    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    batch = load_batch(inputs, seed, case, parent, checkpoint, config, device,
                        field_mean_normalization=model.correction_identity["binding"]["field_mean_normalization"])
                    base = checkpoint["binding"]["config"]["training_variational_seed_base"]
                except Exception as error:
                    setup_error = f"{type(error).__name__}: {error}"
                for replicate in REPLICATES:
                    for budget in BUDGETS:
                        identity = dict(seed=seed, mode=mode, case_id=case, replicate=replicate, particles=budget)
                        try:
                            if setup_error:
                                raise RuntimeError(setup_error)
                            initial = variational_seed(base, seed, "native_mean_particle_initial", case, replicate)
                            dynamics = variational_seed(base, seed, "native_mean_particle_dynamics", case, replicate)
                            identity.update(initial_seed=initial, dynamics_seed=dynamics)
                            row, value = mean_probe(model, batch, initial, dynamics, budget,
                                maximum_particles=max(BUDGETS), unit_chunk_size=config["unit_chunk_size"],
                                mean_unit_chunk_size=config["mean_unit_chunk_size"])
                            vectors[budget, replicate] = value
                            row.update(identity)
                        except Exception as error:
                            row = dict(identity, status="FAILED", error_type=type(error).__name__,
                                       error=str(error), gradient_finite=None, gradient_nonzero=None)
                        rows.append(row); report["records"].append(row)
                        for horizon in config["horizons"]:
                            report["horizon_records"].append(dict(identity, horizon_bins=horizon,
                                status=row["status"], native_nll=row.get(f"native_nll_h{horizon}"),
                                field_mean_mse=row.get(f"field_mean_mse_h{horizon}"),
                                particle_ess=row.get(f"particle_ess_h{horizon}")))
                        write_json(output / "progress.json", report)
                        print(json.dumps({key: row.get(key) for key in
                            ("seed", "mode", "case_id", "replicate", "particles", "status", "loss", "error")}), flush=True)
                state_ok = model is not None and before is not None and unchanged(model, before)
                if before is not None and not state_ok:
                    report["parameters_updated"] = True
                complete = len(vectors) == len(BUDGETS) * len(REPLICATES) and state_ok
                cell = dict(seed=seed, mode=mode, case_id=case, status="COMPLETE" if complete else "INCOMPLETE",
                    exit_code=0 if complete else 2, expected_evaluations=6, complete_evaluations=len(vectors),
                    model_unchanged=state_ok, objective_evaluations=len(rows), records=rows,
                    comparisons=summarize_components(rows, vectors) if complete else {},
                    raw_gradient_vectors_retained=False,
                    input_manifest_sha256=report["input_manifest_sha256"], config_sha256=report["config_sha256"],
                    diagnostic_sha256=report["diagnostic_sha256"], sampler_sha256=report["sampler_sha256"],
                    run_id=report["run_id"], trial_id=report["trial_id"])
                cell_path = output / "cells" / f"seed_{seed}" / mode / case / "analysis.json"
                cell_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(cell_path, cell)
                report["cells"].append(dict(cell, path=str(cell_path.relative_to(output)), sha256=sha256_file(cell_path)))
                del vectors
                write_json(output / "progress.json", report)
    expected = {(r["seed"], r["mode"], r["case_id"], r["replicate"], r["particles"]) for r in mean_conditions()}
    actual = [(r["seed"], r["mode"], r["case_id"], r["replicate"], r["particles"]) for r in report["records"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("complete144-condition mean-gradient denominator differs")
    complete = all(cell["status"] == "COMPLETE" for cell in report["cells"]) and not report["parameters_updated"]
    report.update(status="COMPLETE_NATIVE_RESIDUAL_MEAN_PARTICLE_DIAGNOSTIC" if complete else "PARTIAL_MEAN_PARTICLE_DIAGNOSTIC",
        exit_code=0 if complete else 2, objective_evaluations=len(actual),
        complete_objective_evaluations=sum(row["status"] == "COMPLETE" for row in report["records"]),
        expected_case_groups=24, complete_case_groups=sum(cell["status"] == "COMPLETE" for cell in report["cells"]),
        finished_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output / "analysis.json", report)
    return report


def diagnose(inputs, refits, config_path, output, device):
    if (platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID")
            or not os.environ.get("ARNOLD_TRIAL_ID")):
        raise ValueError("native residual particle diagnosis requires actual Merlin execution")
    inputs, refits, config_path, output = map(Path, (inputs, refits, config_path, output))
    if output.exists():
        raise FileExistsError("fresh particle diagnostic output required")
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    family = build_refit_receipt(refits)
    manifest = json.loads((inputs / "manifest.json").read_text())
    if (manifest.get("scope") != "complete_fitting_and_internal_selection"
            or manifest.get("training_seeds") != list(SEEDS) or manifest.get("blocks") != list(range(14))
            or family["shared_binding"]["input_manifest_sha256"] != sha256_file(inputs / "manifest.json")
            or family["shared_binding"]["config_sha256"] != sha256_file(config_path)
            or family["config"] != config):
        raise ValueError("complete native refits with their original fitting input/configuration required")
    for name, digest in family["shared_binding"]["sources"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError("frozen native fitting source differs: " + name)
    if config.get("objective") == MEAN_OBJECTIVE:
        return _diagnose_mean(inputs, refits, config_path, output, device, config, family)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    report = dict(status="RUNNING", exit_code=None, run_id=os.environ["RH2_JOB_RUN_ID"],
        trial_id=os.environ["ARNOLD_TRIAL_ID"], host=platform.node(),
        started_utc=datetime.now(timezone.utc).isoformat(), device=device, case_id=CASE,
        training_seeds=list(SEEDS), modes=list(MODES), particle_budgets=list(BUDGETS),
        replicate_indices=list(REPLICATES), origins_bins=list(config["origins"]), horizons_bins=list(config["horizons"]),
        source_units=100000, fields=128, channels=4, objective="native_joint_event_path_mixture_nll",
        regularization_included=False, training_prediction_particles=config["prediction_particles"],
        parameters=list(PARAMETERS), parameters_updated=False, generated_unit_samples=False,
        validation_outcomes_accessed=False, excluded_blocks_accessed=False, source_episode_blocks=[0], scientific_claim_qualified=False,
        significance_qualified=False, sota_qualified=False,
        nested_draw_layout="per-step maximum K=32; K8 uses the first eight initial/process draws",
        stream_scope="three named stream replicates per training seed; paired across budgets and modes; not training RNG replay",
        input_manifest_sha256=family["shared_binding"]["input_manifest_sha256"],
        config_sha256=family["shared_binding"]["config_sha256"],
        diagnostic_sha256=sha256_file(Path(__file__)), sampler_sha256=sha256_file(Path(legacy.__file__)),
        refit_receipt=family, records=[], horizon_records=[], cells=[])
    failure = False
    try:
        for seed in SEEDS:
            parent, batch, _ = load_seed(inputs, seed)
            run = json.loads((inputs / f"parent/seed_{seed}/training/run.json").read_text())
            parent.to(device)
            parent_before = {name: value.detach().cpu().clone() for name, value in parent.state_dict().items()}
            origins, horizons = batch["origins"], batch["horizons"]
            if list(origins) != config["origins"] or list(horizons) != config["horizons"]:
                raise ValueError("fixed fitting episode origins/horizons differ")
            arguments = tuple(batch[name].to(device) for name in
                ("mean", "variance", "controls", "target_counts", "target_eligible_ticks"))
            base = run["binding"]["config"]["training_variational_seed_base"]
            for mode in MODES:
                model = load_native_observation_correction(refits / f"seed_{seed}" / mode,
                    parent, run["binding"], run["best_checkpoint_sha256"]).to(device).eval()
                before = {name: getattr(model, name).detach().cpu().clone() for name in PARAMETERS}
                rows, vectors = [], {}
                for replicate in REPLICATES:
                    initial = variational_seed(base, seed, "native_residual_particle_initial", CASE, replicate)
                    dynamics = variational_seed(base, seed, "native_residual_particle_dynamics", CASE, replicate)
                    for budget in BUDGETS:
                        identity = dict(seed=seed, mode=mode, case_id=CASE, replicate=replicate,
                                        particles=budget, initial_seed=initial, dynamics_seed=dynamics)
                        try:
                            row, vector = probe(model, arguments, origins, horizons, initial, dynamics, budget,
                                                maximum_particles=max(BUDGETS), unit_chunk_size=config["unit_chunk_size"])
                            row.update(identity)
                            vectors[budget, replicate] = vector
                        except Exception as error:
                            failure = True
                            row = dict(identity, status="FAILED", error_type=type(error).__name__,
                                       error=str(error), gradient_finite=None, gradient_nonzero=None)
                        rows.append(row)
                        report["records"].append(row)
                        for horizon in horizons:
                            report["horizon_records"].append(dict(identity, horizon_bins=horizon,
                                status=row["status"], native_nll=row.get(f"native_nll_h{horizon}"),
                                particle_ess=row.get(f"particle_ess_h{horizon}"),
                                gradient_norm=row.get("gradient_norm"), gradient_scope="mean NLL over the three horizons"))
                        write_json(output / "progress.json", report)
                        print(json.dumps(row), flush=True)
                parent_ok = unchanged(parent, parent_before)
                correction_ok = all(torch.equal(getattr(model, name).detach().cpu(), before[name]) for name in PARAMETERS)
                if not parent_ok or not correction_ok:
                    report["parameters_updated"] = True
                    raise RuntimeError("diagnostic changed parent or fitted correction")
                cell = dict(seed=seed, mode=mode, parent_unchanged=parent_ok, correction_unchanged=correction_ok,
                            objective_evaluations=len(rows), complete_gradients=len(vectors))
                if len(vectors) == len(BUDGETS) * len(REPLICATES):
                    cell["comparisons"] = summarize(rows, vectors)
                if vectors:
                    ordered = [(budget, replicate) for replicate in REPLICATES for budget in BUDGETS
                               if (budget, replicate) in vectors]
                    path = output / f"seed_{seed}_{mode}_gradients.npz"
                    np.savez_compressed(path, gradients=np.stack([vectors[key] for key in ordered]),
                        particles=np.asarray([key[0] for key in ordered]), replicate=np.asarray([key[1] for key in ordered]),
                        parameter_names=np.asarray(PARAMETERS))
                    cell["gradient_artifact"] = dict(path=path.name, sha256=sha256_file(path))
                report["cells"].append(cell)
                del model, vectors
            del parent, batch, arguments, parent_before
        report.update(status="PARTIAL_DIAGNOSTIC" if failure else "COMPLETE_NATIVE_RESIDUAL_PARTICLE_DIAGNOSTIC",
                      exit_code=2 if failure else 0, objective_evaluations=len(report["records"]))
    except Exception as error:
        report.update(status="FAILED_DIAGNOSTIC", exit_code=2, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "refits", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = diagnose(args.inputs, args.refits, args.config, args.output, args.device)
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()

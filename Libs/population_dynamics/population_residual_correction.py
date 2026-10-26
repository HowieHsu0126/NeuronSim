"""Fit and restore a frozen, fitting-only native population residual correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import yaml

from .population_prior_residual_prediction import _coordinates, validate_config
from .population_residual_memory import ResidualMemoryPopulationState
from . import population_residual_rollout_qualification as rollout_qualification
from .population_state import ControlledPopulationState
from .provenance import canonical_json, freeze_json, sha256_file
from .residual_prediction import RidgeModel, fit_residual_predictor

SCHEMA = "native_population_residual_correction_v1"
ARRAYS = ("x_mean", "x_scale", "y_mean", "y_scale", "coefficients")
SEEDS = [41, 42, 43]
MODES = ["current", "history"]
ALPHA = 0.0001
QUALIFICATION_SELECTION = "complete_fitting_internal_stochastic_rollout_qualification"


def _software():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in (
        "population_residual_correction.py", "population_residual_memory.py", "residual_prediction.py")}


def _source(model):
    return dict(field_names=list(model.field_names), source_units=len(model.unit_ids),
                latent_channels=model.latent_channels, input_size=model.input_size,
                bin_seconds=model.bin_seconds, tick_seconds=model.tick_seconds,
                source_metadata_identity=model.source_metadata_identity)


def _fitting_inputs(fitting, seed, parent_digest):
    if (fitting.get("training_seeds") != SEEDS
            or fitting.get("fitting_blocks") != list(range(14))
            or fitting.get("excluded_selection_blocks") != list(range(14, 18))
            or not isinstance(fitting.get("diagnostic_sha256"), str)
            or len(fitting["diagnostic_sha256"]) != 64):
        raise ValueError("registered fitting-only correction provenance required")
    source = fitting["coordinate_inputs"]
    expected = {"training/run.json", "training/best.pt"}
    expected.update(f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt"
                    for block in range(14) for template in range(4))
    if (set(source["files"]) != expected
            or source["files"]["training/best.pt"] != parent_digest
            or source.get("selected_epoch") != 79
            or seed not in SEEDS):
        raise ValueError("complete same-parent fitting cache identities required")


def _qualification_software():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in (
        "population_residual_rollout_qualification.py", "population_residual_memory.py", "residual_prediction.py",
        "population_prior_residual_prediction.py", "population_prior_residual_diagnostic.py", "population_state.py",
        "population_observation_prior.py", "provenance.py")}


def _qualification_protocol():
    return dict(status="COMPLETE_FITTING_INTERNAL_ROLLOUT_QUALIFICATION", training_seeds=SEEDS,
        correction_fitting_blocks=list(range(12)), correction_qualification_blocks=[12, 13],
        original_parent_training_blocks=list(range(14)), original_parent_selection_blocks=list(range(14, 18)),
        qualification_is_independent_parent_test=False, original_selection_arrays_accessed=False,
        source_units=100000, posterior_coordinates=512, ridge_alphas=rollout_qualification.ALPHAS,
        qualification_episodes_per_seed=8, origins_ms=[1000, 3000, 5000], origins_per_candidate=24,
        horizons_ms=[20, 80, 500], draws=16, steps=100, bin_ms=5,
        candidate_order=[name for name, _, _ in rollout_qualification.CANDIDATES],
        origin_order=rollout_qualification._origin_order(), noise_stream_seeds=dict(initial=41, dynamics=42),
        stream_policy="Reset once per candidate and seed; advance continuously across the fixed 24-origin order; stop at first invalid origin.",
        original_initial_and_process_variances=True, parent_parameters_and_checkpoint_unchanged=True,
        clipping_or_stabilization_applied=False,
        native_hazard_guard="exp(log_hazard)*tick_seconds finite and strictly positive at all 100000 units on every generated step",
        observation_sampling_performed=False, generated_native_count_evaluation=False,
        native_fidelity_qualified=False, biological_claim_qualified=False, scientific_claim_qualified=False)


def _coordinate_digest(coordinates):
    return hashlib.sha256(canonical_json(coordinates).encode("utf-8")).hexdigest()


def _is_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _regularization_selection(selection, *, seed, mode, alpha, fitting):
    if selection is None:
        if alpha != ALPHA:
            raise ValueError("alternate alpha requires authenticated rollout regularization selection")
        return
    if (selection.get("source") != QUALIFICATION_SELECTION or selection.get("mode") != mode
            or selection.get("protocol") != _qualification_protocol()
            or not _is_digest(selection.get("qualification_sha256"))
            or selection.get("diagnostic_sha256") != fitting["diagnostic_sha256"]
            or selection.get("eligible_training_seeds") != SEEDS
            or selection.get("export_fitting_blocks") != list(range(14))
            or selection.get("refit_requires_native_evaluation") is not True
            or selection.get("software_sources") != _qualification_software()):
        raise ValueError("regularization selection protocol or source identity differs")
    digests = selection.get("coordinate_input_sha256_by_seed", {})
    if (set(digests) != {str(value) for value in SEEDS} or not all(_is_digest(value) for value in digests.values())
            or digests[str(seed)] != _coordinate_digest(fitting["coordinate_inputs"])):
        raise ValueError("regularization selection fitting identities differ")
    aggregated = rollout_qualification.aggregate_candidates(selection["runs"])
    chosen = aggregated["selected_alphas"][mode]
    if (isinstance(alpha, (bool, np.bool_)) or alpha not in rollout_qualification.ALPHAS
            or chosen["status"] != "SELECTED" or chosen["alpha"] != alpha
            or selection.get("selected") != chosen):
        raise ValueError("regularization selection does not qualify this mode and alpha across all seeds")


def admit_rollout_qualification(path, diagnostic, diagnostic_sha256):
    """Authenticate existing qualification evidence; do not repeat its scoring."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("complete rollout qualification analysis is required")
    qualification = json.loads(path.read_text())
    protocol = _qualification_protocol()
    if any(qualification.get(key) != value for key, value in protocol.items()):
        raise ValueError("complete registered stochastic rollout qualification required")
    if (qualification.get("fitting_only_diagnostic_sha256") != diagnostic_sha256
            or qualification.get("software_sources") != _qualification_software()
            or qualification.get("score_rows") != 3 * 9 * 24 * 3):
        raise ValueError("qualification diagnostic or software identity differs")
    consumed = qualification["consumed"]
    if set(consumed) != {str(seed) for seed in SEEDS}:
        raise ValueError("all three qualification source identities required")
    for seed in SEEDS:
        original = diagnostic["consumed"][str(seed)]
        caches = {f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt":
                  original["files"][f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt"]
                  for block in (12, 13) for template in range(4)}
        if (consumed[str(seed)]["authenticated_coordinates"] != original
                or consumed[str(seed)]["qualification_caches"] != caches):
            raise ValueError("qualification parent or original fitting cache identities differ")
        scale = np.asarray(qualification["coordinate_scales"][str(seed)], dtype=float)
        if scale.shape != (512,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("finite fitting-only qualification coordinate scales required")
    runs = qualification["runs"]
    aggregated = rollout_qualification.aggregate_candidates(runs)
    if any(qualification.get(key) != value for key, value in aggregated.items()):
        raise ValueError("declared qualification selection differs from complete candidate records")
    if any(aggregated["selected_alphas"][mode]["status"] != "SELECTED" for mode in MODES):
        raise ValueError("no eligible three-seed qualification candidate for a required mode")
    candidate_details = {name: (mode, alpha) for name, mode, alpha in rollout_qualification.CANDIDATES}
    origins = rollout_qualification._origin_order()
    by_artifact = {}
    for run in runs:
        if (candidate_details[run["candidate"]] != (run["mode"], run["alpha"])
                or run["status"] not in ("COMPLETE", "FAILED") or len(run["origins"]) != 24):
            raise ValueError("qualification candidate identity or coverage differs")
        for expected, actual in zip(origins, run["origins"]):
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError("qualification origin order differs")
            if run["status"] == "COMPLETE" and (
                    actual["status"] != "COMPLETE" or actual["completed_steps"] != 100 or actual["failure"] is not None
                    or set(actual["scores"]) != {"20", "80", "500"}
                    or any(not np.isfinite(score["posterior_coordinate_energy_score"]) for score in actual["scores"].values())):
                raise ValueError("complete finite qualification trajectories required")
        by_artifact[f"candidates/seed_{run['seed']}_{run['candidate']}.json"] = run
    artifacts = qualification["artifacts"]
    if set(artifacts) != {"score_rows.csv", *by_artifact}:
        raise ValueError("complete qualification artifact inventory required")
    for name, digest in artifacts.items():
        artifact = path.parent / name
        if not _is_digest(digest) or sha256_file(artifact) != digest:
            raise ValueError("qualification artifact changed: " + name)
        if name in by_artifact and json.loads(artifact.read_text()) != by_artifact[name]:
            raise ValueError("qualification candidate artifact differs from analysis: " + name)
    compact_keys = ("seed", "candidate", "mode", "alpha", "status", "completed_origins",
                    "scored_origin_counts", "mean_posterior_coordinate_energy_score")
    common = dict(source=QUALIFICATION_SELECTION, qualification_sha256=sha256_file(path),
        diagnostic_sha256=diagnostic_sha256, protocol=protocol, eligible_training_seeds=SEEDS,
        coordinate_input_sha256_by_seed={str(seed): _coordinate_digest(diagnostic["consumed"][str(seed)]) for seed in SEEDS},
        runs=[{key: run[key] for key in compact_keys} for run in runs],
        software_sources=qualification["software_sources"], export_fitting_blocks=list(range(14)),
        refit_requires_native_evaluation=True)
    selections = {mode: dict(common, mode=mode, selected=aggregated["selected_alphas"][mode]) for mode in MODES}
    return dict(qualification_sha256=common["qualification_sha256"], protocol=protocol,
                alpha_by_mode={mode: aggregated["selected_alphas"][mode]["alpha"] for mode in MODES},
                regularization_selection_by_mode=selections)


def save_correction(output, predictor, parent, *, seed, mode, parent_checkpoint_sha256, fitting,
                    regularization_selection=None):
    """Export arrays and path-independent identity after authenticated fitting."""
    if mode not in MODES:
        raise ValueError("registered correction mode required")
    _fitting_inputs(fitting, seed, parent_checkpoint_sha256)
    _regularization_selection(regularization_selection, seed=seed, mode=mode, alpha=predictor.alpha, fitting=fitting)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "predictor.npz", **{key: getattr(predictor, key) for key in ARRAYS})
    identity = dict(schema=SCHEMA, seed=seed, mode=mode, alpha=predictor.alpha,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    source=_source(parent), fitting=fitting,
                    predictor_sha256=sha256_file(output / "predictor.npz"),
                    software_sources=_software())
    if regularization_selection is not None:
        identity["regularization_selection"] = json.loads(json.dumps(regularization_selection, allow_nan=False))
    freeze_json(output / "manifest.json", dict(status="FITTED_NATIVE_RESIDUAL_CORRECTION", identity=identity))
    return identity


def load_correction(path, parent, binding, parent_checkpoint_sha256):
    """Authenticate a correction without requiring its original storage path."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    identity = manifest["identity"]
    if (manifest.get("status") != "FITTED_NATIVE_RESIDUAL_CORRECTION"
            or identity.get("schema") != SCHEMA or identity.get("mode") not in MODES
            or identity.get("seed") != binding["seed"]
            or identity.get("parent_checkpoint_sha256") != parent_checkpoint_sha256
            or identity.get("source") != _source(parent)):
        raise ValueError("residual correction parent, seed or source identity differs")
    _fitting_inputs(identity["fitting"], identity["seed"], parent_checkpoint_sha256)
    _regularization_selection(identity.get("regularization_selection"), seed=identity["seed"],
        mode=identity["mode"], alpha=identity["alpha"], fitting=identity["fitting"])
    if identity.get("software_sources") != _software():
        raise ValueError("residual correction implementation changed")
    if sha256_file(path / "predictor.npz") != identity["predictor_sha256"]:
        raise ValueError("residual correction predictor changed")
    with np.load(path / "predictor.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(ARRAYS):
            raise ValueError("complete ridge predictor arrays required")
        predictor = RidgeModel(alpha=identity["alpha"], **{key: saved[key].copy() for key in ARRAYS})
    model = ResidualMemoryPopulationState(parent, predictor, identity["mode"], identity)
    model.eval()
    return model


def validate_fit_config(config, diagnostic, *, qualification_supplied=False):
    expected = dict(schema=SCHEMA, training_seeds=SEEDS, modes=MODES, alpha=ALPHA,
                    fitting_blocks=list(range(14)), excluded_selection_blocks=list(range(14, 18)),
                    source_units=100000, cpu_threads=2,
                    parent_training=False, correction_selection="unanimous_inner_fitting_folds")
    if qualification_supplied:
        del expected["alpha"]
        expected.update(correction_selection=QUALIFICATION_SELECTION, alpha_source="qualification.selected_alphas")
    if config != expected:
        raise ValueError("registered correction fitting configuration and matching qualification argument required")
    validate_config(diagnostic["config"])
    if (diagnostic.get("status") != "COMPLETE_FITTING_BLOCK_RESIDUAL_PREDICTION"
            or diagnostic.get("next_action") != "TEST_CAUSAL_MEMORY_CORRECTION_IN_FREE_GENERATION"
            or diagnostic.get("score_rows") != 126
            or diagnostic.get("selection_arrays_accessed") is not False):
        raise ValueError("accepted fitting-only memory decision required")
    folds = diagnostic["folds"]
    expected_folds = {(seed, block) for seed in SEEDS for block in range(14)}
    if (len(folds) != len(expected_folds)
            or {(row["seed"], row["outer_block"]) for row in folds} != expected_folds
            or any(row["selected_alphas"] != dict(current=ALPHA, history=ALPHA) for row in folds)):
        raise ValueError("one unanimously selected fitting alpha for every seed/block/mode required")


def fit(campaign, diagnostic_path, config_path, output, *, qualification_path=None):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only admitted Slurm allocation required")
    campaign, diagnostic_path, config_path, output = map(Path, (campaign, diagnostic_path, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    validate_fit_config(config, diagnostic, qualification_supplied=qualification_path is not None)
    for name, digest in diagnostic["artifacts"].items():
        if Path(name).name != name or sha256_file(diagnostic_path.parent / name) != digest:
            raise ValueError("accepted residual diagnostic artifact changed")
    for name in ("population_prior_residual_prediction.py", "population_prior_residual_diagnostic.py",
                 "population_state.py"):
        if sha256_file(Path(__file__).with_name(name)) != diagnostic["software_sources"][name]:
            raise ValueError("diagnostic coordinate construction changed: " + name)
    qualification = (None if qualification_path is None else
                     admit_rollout_qualification(qualification_path, diagnostic, sha256_file(diagnostic_path)))
    alpha_by_mode = ({mode: ALPHA for mode in MODES} if qualification is None else qualification["alpha_by_mode"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    records = []
    for seed in SEEDS:
        seed_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"neuronsim-residual-fit-{seed}-") as directory:
            scratch = Path(directory)
            X, Y, _, source = _coordinates(campaign, seed, scratch, diagnostic["config"])
            if source != diagnostic["consumed"][str(seed)]:
                raise ValueError("fitting coordinates differ from accepted diagnostic")
            checkpoint = torch.load(scratch / "training/best.pt", weights_only=True, map_location="cpu")
            parent = ControlledPopulationState(**checkpoint["model_configuration"])
            parent.load_state_dict(checkpoint["model"], strict=True)
            parent.eval()
            if len(parent.unit_ids) != config["source_units"]:
                raise ValueError("complete 100000-unit parent required")
            parent_digest = source["files"]["training/best.pt"]
            fitting = dict(training_seeds=SEEDS, fitting_blocks=config["fitting_blocks"],
                           excluded_selection_blocks=config["excluded_selection_blocks"],
                           diagnostic_sha256=sha256_file(diagnostic_path), coordinate_inputs=source)
            coordinate_seconds = time.perf_counter() - seed_started
            for mode in MODES:
                fit_started = time.perf_counter()
                width = source["current_width"] if mode == "current" else X.shape[-1]
                alpha = alpha_by_mode[mode]
                predictor = fit_residual_predictor(X, Y, width, alpha)
                relative = f"seed_{seed}/{mode}"
                identity = save_correction(output / relative, predictor, parent, seed=seed, mode=mode,
                    parent_checkpoint_sha256=parent_digest, fitting=fitting,
                    regularization_selection=None if qualification is None else qualification["regularization_selection_by_mode"][mode])
                restored = load_correction(output / relative, parent, checkpoint["binding"], parent_digest)
                if restored.correction_identity != identity:
                    raise ValueError("saved correction did not reload with identical identity")
                record = dict(seed=seed, mode=mode, path=relative, alpha=alpha,
                    fitting_rows=int(np.prod(Y.shape[:-1])), feature_width=width, target_width=Y.shape[-1],
                    coordinate_seconds=coordinate_seconds, fit_and_export_seconds=time.perf_counter()-fit_started,
                    manifest_sha256=sha256_file(output / relative / "manifest.json"))
                records.append(record)
                print(json.dumps(record), flush=True)
            del X, Y, parent, checkpoint, restored, predictor
    result = dict(status="FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS", config=config, corrections=records,
        alpha_by_mode=alpha_by_mode,
        diagnostic_sha256=sha256_file(diagnostic_path), config_sha256=sha256_file(config_path),
        software_sources=_software(), job_id=os.environ["SLURM_JOB_ID"],
        node=os.environ["SLURMD_NODENAME"], elapsed_seconds=time.perf_counter()-started,
        parent_training=False, selection_arrays_accessed=False, free_generation_evaluated=False,
        scientific_claim_qualified=False)
    if qualification is not None:
        result.update(qualification_sha256=qualification["qualification_sha256"],
            qualification_protocol=qualification["protocol"], export_fitting_blocks=list(range(14)),
            refit_is_identical_to_qualification_fit=False, native_evaluation_required=True)
    freeze_json(output / "fit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "diagnostic", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    result = fit(args.campaign, args.diagnostic, args.config, args.output, qualification_path=args.qualification)
    print(json.dumps(dict(status=result["status"], corrections=len(result["corrections"]))))


if __name__ == "__main__":
    main()

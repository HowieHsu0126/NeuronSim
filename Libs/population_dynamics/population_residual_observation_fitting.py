"""Fit native-readout residual dynamics with fitting-only epoch selection.

The parent was trained on blocks0--13. Inner selection is development evidence.
This entry point never reads blocks14--17 and does not grant source qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch
import yaml

from .native_renewal_data import NativeRenewalEpisode
from .population_prior_residual_prediction import _coordinates
from .population_residual_memory import ResidualMemoryPopulationState
from .population_residual_observation import configure_correction_parameters, export_affine_predictor
from .population_residual_observation_validation import load_seed
from .population_residual_observation_training import MEAN_OBJECTIVE, NATIVE_OBJECTIVE, run_epochs
from .poisson_trajectory_evaluation import fitting_scale
from .provenance import sha256_file
from .residual_prediction import RidgeModel, fit_residual_predictor


SCHEMA = "native_residual_observation_fit_v1"
ARRAYS = tuple(field.name for field in fields(RidgeModel))
SOURCES = (
    "population_residual_observation_fitting.py", "population_residual_observation_training.py",
    "population_residual_observation.py", "population_residual_observation_validation.py",
    "population_residual_memory.py", "population_prior_residual_prediction.py",
    "population_prior_residual_diagnostic.py", "residual_prediction.py",
    "population_state.py", "refractory_observation.py", "native_renewal_data.py",
    "native_count_windows.py", "population_prior_objective.py",
)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_predictor(path, predictor):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{name: getattr(predictor, name) for name in ARRAYS})
    temporary.replace(path)


def read_predictor(path):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(ARRAYS):
            raise ValueError("native predictor array schema differs")
        return RidgeModel(float(saved["alpha"]), **{
            name: saved[name].copy() for name in ARRAYS if name != "alpha"})


def validate_config(config):
    fixed = dict(schema=SCHEMA, training_seeds=[41, 42, 43],
        fitting_blocks=list(range(12)), selection_blocks=[12, 13],
        refit_blocks=list(range(14)), excluded_blocks=list(range(14, 18)),
        templates_per_block=4, episode_bins=1200, bin_ms=5, source_units=100000,
        origins=[200, 600, 1000], horizons=[4, 16, 100], prediction_particles=8,
        initialization_ridge_alpha=1.0, optimizer="Adam", weight_decay=0.0,
        selection="minimum_complete_unpenalized_validation_loss_earliest_tie",
        stream_protocol="native_residual_observation_named_streams_v1")
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("registered native observation fitting protocol required")
    objective = config.get("objective", NATIVE_OBJECTIVE)
    if objective not in (NATIVE_OBJECTIVE, MEAN_OBJECTIVE):
        raise ValueError("unsupported native residual fitting objective")
    mean_keys = {"field_mean_weight", "mean_unit_chunk_size", "mean_normalization_blocks"}
    if objective == MEAN_OBJECTIVE:
        if (type(config.get("field_mean_weight")) not in (int, float) or config["field_mean_weight"] != 1.0
                or config.get("mean_unit_chunk_size") != 32768
                or config.get("mean_normalization_blocks") != list(range(12))):
            raise ValueError("registered weight-one field mean objective and fitting-only normalization required")
    elif mean_keys & set(config):
        raise ValueError("field mean settings require an explicit mean objective")
    for key in ("learning_rate", "regularization_weight"):
        if type(config.get(key)) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("finite positive " + key + " required")
    for key in ("max_epochs", "patience", "unit_chunk_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("positive integer " + key + " required")
    if config["max_epochs"] > 80:
        raise ValueError("the authorized first fit budget is at most80 epochs")


def cases(blocks):
    return [f"train_block_{block:03d}_familiar_{template}"
            for block in blocks for template in range(4)]


def fit_field_mean_normalization(inputs, config):
    """Reuse the existing field scaler on the declared inner fitting cases."""
    selected = cases(config["mean_normalization_blocks"])
    fields_root = Path(inputs) / "cohort/train_fields"
    summaries = {}
    for case in selected:
        path = fields_root / case / "summary.json"
        summary = json.loads(path.read_text())
        if summary["settings"]["bin_ms"] != config["bin_ms"]:
            raise ValueError("field normalization clock differs from the native bins")
        summaries[case] = sha256_file(path)
    _, sd, metadata = fitting_scale(fields_root, [{"case_id": case} for case in selected])
    return dict(method="poisson_trajectory_evaluation.fitting_scale",
        blocks=list(config["mean_normalization_blocks"]), cases=selected, bin_ms=config["bin_ms"],
        field_ids=list(map(str, metadata["field_ids"])), unit_count=metadata["unit_count"].tolist(),
        standard_deviation_hz=sd.tolist(), source_summaries=summaries)


def validate_field_mean_normalization(normalization, parent, parent_binding, config):
    """Bind the training-only scale to source episodes and the parent's map."""
    if not isinstance(normalization, dict):
        raise ValueError("bound fitting-only field normalization required")
    selected = cases(config["mean_normalization_blocks"])
    if (normalization.get("method") != "poisson_trajectory_evaluation.fitting_scale"
            or normalization.get("blocks") != config["mean_normalization_blocks"]
            or normalization.get("cases") != selected or normalization.get("bin_ms") != config["bin_ms"]
            or normalization.get("field_ids") != list(parent.field_names)
            or set(normalization.get("source_summaries", {})) != set(selected)):
        raise ValueError("field normalization scope or source map differs")
    counts = np.asarray(normalization.get("unit_count"))
    sd = np.asarray(normalization.get("standard_deviation_hz"))
    if (counts.shape != (len(parent.field_names),) or counts.dtype.kind not in "iu"
            or not np.array_equal(counts, np.bincount(parent.unit_field_index.detach().cpu().numpy(),
                                                    minlength=len(parent.field_names)))
            or sd.shape != counts.shape or sd.dtype.kind != "f" or not np.isfinite(sd).all() or np.any(sd <= 0)):
        raise ValueError("positive fixed field scale and exact source unit denominators required")
    expected = {case: parent_binding["episode_bindings"][case]["field_summary_sha256"] for case in selected}
    if normalization["source_summaries"] != expected:
        raise ValueError("field normalization data differs from the parent source")
    return sd


def load_batch(inputs, seed, case, parent, checkpoint, config, device, *, field_mean_normalization=None):
    """Authenticate one original episode and reuse its causal posterior cache."""
    binding = checkpoint["binding"]
    cache = torch.load(inputs / f"parent/seed_{seed}/training/posterior_cache/{case}.pt",
                       map_location="cpu", weights_only=True)
    cb = cache["binding"]
    if (cb["case_id"] != case or cb["bin_ms"] != config["bin_ms"]
            or cb["parent_checkpoint_sha256"] != checkpoint["initialization"]["parent_checkpoint_sha256"]
            or cb["model_implementation_sha256"] != binding["model_implementation_sha256"]
            or any(binding[key] != value for key, value in cb["cohort"].items())):
        raise ValueError("posterior cache lineage differs: " + case)
    source = NativeRenewalEpisode(inputs / f"cohort/train/{case}", inputs / f"cohort/train_fields/{case}",
        expected_source=binding["native_source_identity"], expected_unit_ids=np.asarray(parent.unit_ids),
        expected_unit_field_index=parent.unit_field_index.detach().cpu().numpy())
    record = binding["episode_bindings"][case]
    if (source.binding["episode_report_sha256"] != record["raw_report_sha256"]
            or source.binding["field_summary_sha256"] != record["field_summary_sha256"]
            or tuple(map(str, source.field_ids)) != tuple(parent.field_names)
            or source.duration_ticks != 60000 or source.bin_ticks != 50
            or not np.array_equal(source.minimum_interval_ticks, parent.minimum_interval_ticks.cpu().numpy())):
        raise ValueError("native episode identity, clock or refractory rule differs: " + case)
    mean, variance, controls = (cache[key] for key in ("mean", "variance", "controls"))
    if (mean.shape != (1, 1200, 128, 4) or variance.shape != mean.shape or controls.shape != (1, 1200, 1)
            or any(value.dtype != parent.unit_loading.dtype or not torch.isfinite(value).all()
                   for value in (mean, variance, controls)) or torch.any(variance <= 0)):
        raise ValueError("complete finite causal cache required: " + case)
    pulses = source.pulse_ticks[source.pulse_ticks < source.duration_ticks]
    native = np.bincount(pulses // source.bin_ticks, minlength=1200).astype(np.float32)[:, None]
    if not np.array_equal(controls[0].numpy(), native):
        raise ValueError("cached controls differ from native TC events: " + case)
    windows = [source.renewal_window(origin * 50, (origin + 100) * 50) for origin in config["origins"]]
    for origin, window in zip(config["origins"], windows):
        if window.counts.shape != (100, 100000) or not np.array_equal(window.stimulus, native[origin:origin + 100]):
            raise ValueError("full-unit future label window differs: " + case)
    batch = dict(mean=mean.to(device), variance=variance.to(device), controls=controls.to(device),
        target_counts=torch.from_numpy(np.stack([window.counts for window in windows])).unsqueeze(0).to(device),
        target_eligible_ticks=torch.from_numpy(np.stack([window.eligible_ticks for window in windows])).unsqueeze(0).to(device),
        origins=config["origins"], horizons=config["horizons"])
    if config.get("objective", NATIVE_OBJECTIVE) == MEAN_OBJECTIVE:
        sd = validate_field_mean_normalization(field_mean_normalization, parent, binding, config)
        batch["initial_ready_in_ticks"] = torch.from_numpy(np.stack(
            [window.initial_ready_in_ticks for window in windows])).unsqueeze(0).to(device)
        batch["field_standard_deviation_hz"] = torch.as_tensor(sd, device=device, dtype=torch.float64)
    return batch


def load_native_observation_correction(path, parent, binding, parent_checkpoint_sha256):
    """Admit the new refit schema without asserting old energy-score qualification."""
    path = Path(path)
    report = json.loads((path / "fit.json").read_text())
    identity = report["binding"]
    validate_config(identity["config"])
    if (report.get("schema") != SCHEMA or report.get("status") != "FITTED_DEVELOPMENT"
            or report.get("phase") != "refit"
            or identity["parent_checkpoint_sha256"] != parent_checkpoint_sha256
            or identity["parent_binding_sha256"] != json_digest(binding)
            or identity["seed"] != binding["seed"]
            or identity["fit_cases"] != cases(range(14))
            or identity["validation_cases"] != []
            or type(report.get("completed_epochs")) is not int
            or report["completed_epochs"] != report.get("selected_epoch")
            or not 0 <= report["completed_epochs"] <= identity["config"]["max_epochs"]
            or not identity.get("selection_sha256")
            or report["scientific_claim_qualified"] is not False
            or report["predictor_sha256"] != sha256_file(path / "predictor.npz")):
        raise ValueError("complete authenticated native observation refit required")
    if identity["config"].get("objective", NATIVE_OBJECTIVE) == MEAN_OBJECTIVE:
        validate_field_mean_normalization(identity.get("field_mean_normalization"), parent, binding, identity["config"])
    for name in ("population_residual_memory.py", "population_state.py", "refractory_observation.py"):
        if identity["sources"][name] != sha256_file(Path(__file__).with_name(name)):
            raise ValueError("native correction runtime differs: " + name)
    return ResidualMemoryPopulationState(parent, read_predictor(path / "predictor.npz"), identity["mode"],
        {"schema": SCHEMA, "fit_sha256": sha256_file(path / "fit.json"),
         "predictor_sha256": report["predictor_sha256"], "binding": identity})


def fit(inputs, manifest_path, config_path, output, phase, seeds, modes, device, epoch_limit,
        *, checkpoint_callback=None):
    if platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID"):
        raise RuntimeError("scientific fitting requires actual Merlin execution")
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    if len(set(seeds)) != len(seeds) or not set(seeds) <= {41, 42, 43} or not seeds:
        raise ValueError("use only the registered training seeds")
    if len(set(modes)) != len(modes) or not set(modes) <= {"current", "history"} or not modes:
        raise ValueError("unique current/history modes required")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema"] != "native_residual_observation_inputs_v1"
            or manifest["scope"] != "complete_fitting_and_internal_selection"
            or manifest["blocks"] != list(range(14)) or manifest["training_seeds"] != [41, 42, 43]):
        raise ValueError("full fitting input manifest required")
    if len({row["path"] for row in manifest["files"]}) != len(manifest["files"]):
        raise ValueError("duplicate input paths")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input path escapes staging root")
        path = inputs / relative
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("staged input identity differs: " + str(relative))
    use_mean = config.get("objective", NATIVE_OBJECTIVE) == MEAN_OBJECTIVE
    if use_mean:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    source_names = SOURCES + (("population_residual_mean_objective.py", "native_renewal_expectation.py",
                               "poisson_trajectory_evaluation.py") if use_mean else ())
    sources = {name: sha256_file(Path(__file__).with_name(name)) for name in source_names}
    fitting_normalization = fit_field_mean_normalization(inputs, config) if use_mean and phase == "qualify" else None
    protocol = dict(training_seeds=[41, 42, 43], fitting_blocks=list(range(14)),
        excluded_selection_blocks=list(range(14, 18)), templates_per_block=4,
        episode_bins=1200, bin_ms=5, source_units=100000)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        parent, pilot_batch, _ = load_seed(inputs, seed)
        del pilot_batch
        parent_path = inputs / f"parent/seed_{seed}/training/best.pt"
        checkpoint = torch.load(parent_path, map_location="cpu", weights_only=True)
        coordinates = None
        for mode in modes:
            cell = output / phase / f"seed_{seed}" / mode
            cell.mkdir(parents=True, exist_ok=True)
            selection = None
            if phase == "refit":
                selection_path = output / "qualify" / f"seed_{seed}" / mode / "fit.json"
                selection = json.loads(selection_path.read_text())
                if (selection.get("status") != "FITTED_DEVELOPMENT" or selection["phase"] != "qualify"
                        or selection["binding"]["config_sha256"] != sha256_file(config_path)
                        or selection["binding"]["input_manifest_sha256"] != sha256_file(manifest_path)
                        or selection["binding"]["parent_checkpoint_sha256"] != sha256_file(parent_path)
                        or selection["binding"]["sources"] != sources
                        or selection["binding"]["seed"] != seed or selection["binding"]["mode"] != mode):
                    raise ValueError("same-protocol complete inner selection required")
            fitting = config["fitting_blocks"] if phase == "qualify" else config["refit_blocks"]
            binding = dict(schema=SCHEMA, phase=phase, seed=seed, mode=mode,
                parent_checkpoint_sha256=sha256_file(parent_path),
                parent_binding_sha256=json_digest(checkpoint["binding"]),
                input_manifest_sha256=sha256_file(manifest_path), config_sha256=sha256_file(config_path),
                config=config, sources=sources, fit_cases=cases(fitting),
                validation_cases=cases(config["selection_blocks"]) if phase == "qualify" else [],
                selection_sha256=sha256_file(selection_path) if selection else None,
                runtime=dict(torch=str(torch.__version__), numpy=np.__version__, device_type=device))
            normalization = None
            if use_mean:
                normalization = (fitting_normalization if phase == "qualify"
                                 else selection["binding"].get("field_mean_normalization"))
                validate_field_mean_normalization(normalization, parent, checkpoint["binding"], config)
                binding["field_mean_normalization"] = normalization
            if (cell / "fit.json").exists():
                existing = json.loads((cell / "fit.json").read_text())
                if existing["binding"] != binding or existing["predictor_sha256"] != sha256_file(cell / "predictor.npz"):
                    raise ValueError("existing completed fit identity differs")
                records.append(dict(seed=seed, mode=mode, status="REUSED_COMPLETE"))
                continue
            if (cell / "epoch_checkpoint.pt").exists() and not all(
                    (cell / name).is_file() for name in ("reference.json", "reference.npz")):
                raise ValueError("restore the complete reference pair with the checkpoint")
            if (cell / "reference.json").exists():
                saved = json.loads((cell / "reference.json").read_text())
                if saved["binding"] != binding or saved["sha256"] != sha256_file(cell / "reference.npz"):
                    raise ValueError("saved initialization identity differs")
                reference = read_predictor(cell / "reference.npz")
            else:
                if coordinates is None:
                    with tempfile.TemporaryDirectory(prefix="native-residual-coordinates-") as scratch:
                        X, Y, _, metadata = _coordinates(inputs / "parent", seed, Path(scratch), protocol)
                    coordinates = (X, Y, metadata)
                X, Y, metadata = coordinates
                width = metadata["current_width"] + (metadata["fields"] * metadata["channels"] if mode == "history" else 0)
                reference = fit_residual_predictor(X[fitting], Y[fitting], width, config["initialization_ridge_alpha"])
                save_predictor(cell / "reference.npz", reference)
                write_json(cell / "reference.json", dict(binding=binding, sha256=sha256_file(cell / "reference.npz"),
                    initialization_blocks=fitting, normalization_blocks=fitting))
            parent.to(device)
            model = ResidualMemoryPopulationState(parent, reference, mode, {"schema": SCHEMA, "seed": seed})
            configure_correction_parameters(model)
            frozen = {key: value.detach().clone() for key, value in parent.state_dict().items()}
            loader = lambda case: load_batch(inputs, seed, case, parent, checkpoint, config, device,
                                                   field_mean_normalization=normalization)
            result = run_epochs(model, reference, loader, binding["fit_cases"], binding["validation_cases"],
                config, phase, seed, cell, binding,
                selected_epochs=selection["selected_epoch"] if selection else None, epoch_limit=epoch_limit,
                checkpoint_callback=(lambda path, epoch: checkpoint_callback(cell, path, epoch, False))
                    if checkpoint_callback is not None else None)
            if any(not torch.equal(value, parent.state_dict()[key]) for key, value in frozen.items()):
                raise RuntimeError("frozen parent changed during fitting")
            result.update(schema=SCHEMA, phase=phase, binding=binding, parent_unchanged=True,
                scientific_claim_qualified=False, source_fidelity_qualified=False,
                parent_selection_scope="development_parent_already_fitted_on_blocks0_to13",
                run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ.get("ARNOLD_TRIAL_ID"))
            if result["status"] == "COMPLETE":
                save_predictor(cell / "predictor.npz", export_affine_predictor(model, reference))
                result.update(status="FITTED_DEVELOPMENT", predictor_sha256=sha256_file(cell / "predictor.npz"))
                write_json(cell / "fit.json", result)
            else:
                write_json(cell / "progress.json", result)
            if checkpoint_callback is not None:
                checkpoint_callback(cell, cell / "epoch_checkpoint.pt", result["completed_epochs"], True)
            records.append(dict(seed=seed, mode=mode, status=result["status"],
                                completed_epochs=result["completed_epochs"], selected_epoch=result["selected_epoch"]))
            del model, frozen
            parent.cpu()
        del parent, checkpoint, coordinates
    summary = dict(schema=SCHEMA, phase=phase, records=records,
        scope="requested_seed_mode_cells", requested_seeds=list(seeds), requested_modes=list(modes),
        complete=all(row["status"] in ("FITTED_DEVELOPMENT", "REUSED_COMPLETE") for row in records),
        scientific_claim_qualified=False)
    write_json(output / "analysis.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "manifest", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("qualify", "refit"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--modes", nargs="+", choices=("current", "history"), default=["current", "history"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epoch-limit", type=int)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or (args.epoch_limit is not None and args.epoch_limit < 1):
        raise ValueError("one to8CPU threads and positive optional epoch limit required")
    torch.set_num_threads(args.threads)
    fit(args.inputs, args.manifest, args.config, args.output, args.phase,
        args.seeds, args.modes, args.device, args.epoch_limit)


if __name__ == "__main__":
    main()

"""AFM independent field adaptation on the complete source-bound 56/16 cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
import yaml

from .autoregressive_field_flow import AutoregressiveFieldFlow, SourceCountTransform, IMPLEMENTATION
from .memory_field_cohort import MemoryFieldCohort
from .poisson_trajectory_training import _save_checkpoint
from .provenance import sha256_file


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config, seed):
    fixed = dict(schema="neural_population_autoregressive_field_flow_training_v1",
        training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
        bin_ms=5., dt_ms=.1, duration_ms=6000., expected_units=100000,
        history_bins=200, horizon_bins=1000, expected_poisson_rng_family="per_edge_philox4_32_10",
        context_bins=22, model=dict(hidden=256, layers=4, latent_hidden=16, dense_blocks=5, dropout=.05),
        maximum_steps=20000, batch_size=64, learning_rate=.001, validation_period=100,
        patience_checks=10, validation_draws=1, euler_steps=100, forecast_samples=16,
        forecast_origins_bins=[200, 600, 1000], forecast_bins=100, cpu_threads=2,
        formal_training_completed=False)
    if config != {**fixed, "device": config.get("device")} or seed not in (41, 42, 43):
        raise ValueError("registered complete-cohort AFM protocol and seed41/42/43 required")
    if config["device"] not in ("cpu", "cuda") or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("AFM training requires explicit device and Slurm allocation")


def fitting_transform(fields, occupancy, bin_seconds):
    """Use complete fitting episodes only, with zero-preserving count units."""
    values = np.asarray(fields, dtype=np.float64)
    occupancy = np.asarray(occupancy)
    if (values.ndim != 3 or occupancy.shape != (values.shape[-1],)
            or not np.isfinite(values).all() or np.any(values < 0)
            or np.any(occupancy <= 0) or np.any(occupancy != np.floor(occupancy))):
        raise ValueError("complete finite fitting fields and integer occupancy required")
    transformed = np.sqrt(values * occupancy[None, None] * bin_seconds)
    center = transformed.mean(axis=(0, 1))
    scale = transformed.std(axis=(0, 1))
    # A fit-constant field has no positive fitting variance. Unit scale leaves
    # its exact observations intact; this is not a validation-data fallback.
    return SourceCountTransform(occupancy, bin_seconds, center, np.where(scale > 0, scale, 1.))


def source_binding():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in
        ("autoregressive_field_training.py", "autoregressive_field_flow.py", "memory_field_cohort.py",
         "native_trajectory_cohort.py", "native_count_windows.py", "provenance.py")}


def tensors(cohort, cases, transform, device):
    episodes = [cohort.window(case) for case in cases]
    rates = torch.as_tensor(np.stack([x[0].reshape(1200, -1) for x in episodes]), device=device)
    controls = torch.as_tensor(np.stack([x[1] for x in episodes]), device=device)
    return transform.encode(rates), controls


def pair_batch(fields, controls, episodes, targets, context_bins):
    if (episodes.ndim != 1 or targets.shape != episodes.shape
            or torch.any(targets < context_bins) or torch.any(targets >= fields.shape[1])):
        raise ValueError("complete observed context and in-episode target bins required")
    times = targets[:, None] - context_bins + torch.arange(context_bins, device=fields.device)
    return (fields[episodes[:, None], times], controls[episodes[:, None], times],
            controls[episodes, targets], fields[episodes, targets])


@torch.no_grad()
def validation_loss(model, fields, controls, config, seed):
    model.eval()
    generator = torch.Generator(device=fields.device).manual_seed(seed)
    count = fields.shape[0] * (fields.shape[1] - config["context_bins"])
    total = 0.
    for start in range(0, count, config["batch_size"]):
        indices = torch.arange(start, min(start + config["batch_size"], count), device=fields.device)
        width = fields.shape[1] - config["context_bins"]
        episodes, targets = indices // width, indices % width + config["context_bins"]
        batch = pair_batch(fields, controls, episodes, targets, config["context_bins"])
        loss = model.matching_loss(*batch, generator=generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite full-cohort AFM validation loss")
        total += float(loss) * len(indices)
    return total / count


def train(acquisition, fields, config_path, output, seed, resume=False, stop_after_steps=None):
    acquisition, fields, config_path, output = map(Path, (acquisition, fields, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("positive diagnostic optimizer-step boundary required")
    if output.exists() and not resume:
        raise FileExistsError("existing AFM training requires explicit resume")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    cohort = MemoryFieldCohort(acquisition, fields, config)
    occupancy = np.bincount(cohort.unit_field_index, minlength=len(cohort.field_ids))
    fit_rates = np.stack([cohort.window(case)[0].reshape(1200, -1) for case in cohort.fit])
    transform = fitting_transform(fit_rates, occupancy, cohort.bin_seconds).to(device)
    fit, fit_control = tensors(cohort, cohort.fit, transform, device)
    validation, validation_control = tensors(cohort, cohort.validation, transform, device)
    binding = dict(config=config, training_seed=seed, dataset=cohort.field_binding(),
        source_sha256=source_binding(), torch_version=torch.__version__,
        fit_case_ids=[x["case_id"] for x in cohort.fit], validation_case_ids=[x["case_id"] for x in cohort.validation],
        transform=dict(center=transform.center.cpu().tolist(), scale=transform.scale.cpu().tolist(),
                       occupancy=occupancy.tolist(), bin_seconds=cohort.bin_seconds))
    model = AutoregressiveFieldFlow(len(cohort.field_ids), 1, config["context_bins"], **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator(device=device).manual_seed(seed)
    state = dict(binding=binding, steps=0, best_loss=None, best_step=None, bad_checks=0,
                 history=[], elapsed_seconds=0., stop_reason=None, resumed=resume)
    if resume:
        pointer = json.loads((output / "committed.json").read_text())
        if sha256_file(output / pointer["checkpoint"]) != pointer["sha256"]:
            raise ValueError("AFM committed checkpoint changed")
        saved = torch.load(output / pointer["checkpoint"], map_location=device, weights_only=False)
        if saved["state"]["binding"] != binding:
            raise ValueError("AFM continuation source/config/data changed")
        state = saved["state"]; state["resumed"] = True
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu()); torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
    else:
        output.mkdir(parents=True)
    invocation, start, previous = 0, time.monotonic(), state["elapsed_seconds"]
    try:
        while state["steps"] < config["maximum_steps"] and state["stop_reason"] is None:
            model.train()
            episodes = torch.randint(len(cohort.fit), (config["batch_size"],), device=device, generator=generator)
            targets = torch.randint(config["context_bins"], 1200, (config["batch_size"],), device=device, generator=generator)
            batch = pair_batch(fit, fit_control, episodes, targets, config["context_bins"])
            loss = model.matching_loss(*batch, generator=generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite AFM fitting loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            state["steps"] += 1; invocation += 1
            check = (state["steps"] == 1 or (state["steps"] - 1) % config["validation_period"] == 0
                     or state["steps"] == config["maximum_steps"])
            diagnostic_boundary = stop_after_steps is not None and invocation >= stop_after_steps
            valid = validation_loss(model, validation, validation_control, config, seed) if check else None
            state["history"].append(dict(step=state["steps"], training_loss=float(loss.detach()), validation_loss=valid))
            if valid is not None:
                if state["best_loss"] is None or valid < state["best_loss"]:
                    state.update(best_loss=valid, best_step=state["steps"], bad_checks=0)
                    _save_checkpoint(dict(binding=binding, model=model.state_dict(), transform=transform.state_dict()), output / "best.pt")
                else:
                    state["bad_checks"] += 1
                if state["bad_checks"] > config["patience_checks"]:
                    state["stop_reason"] = "SOURCE_PATIENCE_EXCEEDED"
            if state["steps"] == config["maximum_steps"]:
                state["stop_reason"] = "MAXIMUM_STEPS"
            if check or diagnostic_boundary or state["stop_reason"]:
                state["elapsed_seconds"] = previous + time.monotonic() - start
                name = f"optimizer_step_{state['steps']:05d}.pt"
                payload = dict(state=state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
                _save_checkpoint(payload, output / name)
                write_json(output / "committed.json", dict(checkpoint=name, sha256=sha256_file(output / name)))
                write_json(output / "run.json", dict(status="FITTED_DEVELOPMENT" if state["stop_reason"] else "RUNNING",
                    binding=binding, steps=state["steps"], best_loss=state["best_loss"], best_step=state["best_step"],
                    stop_reason=state["stop_reason"], resumed=state["resumed"], elapsed_seconds=state["elapsed_seconds"],
                    parameter_count=sum(p.numel() for p in model.parameters()), implementation=IMPLEMENTATION,
                    hardware=dict(node=platform.node(), device=str(device), torch_version=torch.__version__),
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    scientific_claim_qualified=False))
                write_json(output / "history.json", state["history"])
            if diagnostic_boundary:
                break
        return json.loads((output / "run.json").read_text())
    except Exception as error:
        write_json(output / ("failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json"),
                   dict(error_type=type(error).__name__, error=str(error), steps=state["steps"], scientific_claim_qualified=False))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.acquisition, args.fields, args.config, args.output, args.seed,
                           args.resume, args.stop_after_steps)))

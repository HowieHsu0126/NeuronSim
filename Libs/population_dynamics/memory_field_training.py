"""Full-episode teacher-forced MemNO field training with committed continuation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import torch
import yaml

from .memory_field_cohort import MemoryFieldCohort
from .memory_neural_operator import MemoryNeuralOperator, IMPLEMENTATION, PAPER_URL
from .poisson_trajectory_training import _save_checkpoint
from .provenance import sha256_file

FIXED_PROTOCOL = dict(schema="neural_population_memory_field_training_v1",
    training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
    bin_ms=5., dt_ms=.1, duration_ms=6000., expected_units=100000,
    history_bins=200, horizon_bins=1000, expected_poisson_rng_family="per_edge_philox4_32_10",
    maximum_epochs=300, batch_size=4, cpu_threads=2, learning_rate=.001,
    learning_rate_half_period=90, gradient_clip_norm=1., weight_decay=0.,
    selection_metric="teacher_forced_fit_scaled_complete_field_mse",
    formal_training_completed=False)


def validate_config(config, seed):
    if config != {**FIXED_PROTOCOL, "device": config.get("device")}:
        raise ValueError("complete registered MemNO development protocol required")
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("training seed41/42/43 required")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("explicit CPU or CUDA device required")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("MemNO compute requires a Slurm allocation")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sources():
    package = Path(__file__).parent
    names = ("memory_field_training.py", "memory_field_cohort.py", "memory_neural_operator.py",
             "native_trajectory_cohort.py", "native_count_windows.py", "provenance.py",
             "poisson_trajectory_training.py")
    result = {name: sha256_file(package / name) for name in names}
    external = package.parent / "external/LDNS"
    for name in ("LICENSE", "source.json", "ldns/networks/s4.py", "ldns/networks/blocks.py",
                 "ldns/networks/count_wrapper.py", "ldns/networks/__init__.py", "ldns/__init__.py"):
        result["external/LDNS/" + name] = sha256_file(external / name)
    return result


def make_model(cohort, device):
    return MemoryNeuralOperator(len(cohort.populations), cohort.grid_shape, 1,
                               cohort.bin_seconds).to(device)


def optimizer_for(model, config):
    # Honor vendored S4 parameter-specific optimizer settings.
    groups = {}
    for parameter in model.parameters():
        settings = dict(lr=config["learning_rate"], weight_decay=config["weight_decay"])
        settings.update(getattr(parameter, "_optim", {}))
        key = tuple(sorted(settings.items()))
        groups.setdefault(key, []).append(parameter)
    return torch.optim.Adam([dict(params=parameters, **dict(key)) for key, parameters in groups.items()])


def teacher_pairs(field, control):
    if field.ndim != 5 or control.ndim != 3 or field.shape[:2] != control.shape[:2] or field.shape[1] < 2:
        raise ValueError("aligned complete batched episode fields and controls required")
    return field[:, :-1], control[:, 1:], field[:, 1:]


def evaluate_epoch(model, cohort, cases, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    rows = []
    batch_size = config["batch_size"]
    scale = torch.as_tensor(cohort.scale, device=model.lifting.weight.device)
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        episodes = [cohort.window(case) for case in batch]
        fields = torch.as_tensor(np.stack([row[0] for row in episodes]), device=scale.device) / scale
        controls = torch.as_tensor(np.stack([row[1] for row in episodes]), device=scale.device)
        observed, inputs, target = teacher_pairs(fields, controls)
        with torch.set_grad_enabled(training):
            prediction = model(observed, inputs)
            loss_per_episode = (prediction - target).square().mean(dim=(1, 2, 3, 4))
            loss = loss_per_episode.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite MemNO complete-episode loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"],
                                               error_if_nonfinite=True)
                optimizer.step()
        rows.extend(dict(case_id=case["case_id"], block=case["block"],
                         loss=float(value)) for case, value in zip(batch, loss_per_episode.detach().cpu()))
    return dict(loss=float(np.mean([row["loss"] for row in rows])), episodes=rows)


def load_committed(output, binding, device):
    pointer = json.loads((output / "committed.json").read_text())
    if pointer["binding"] != binding:
        raise ValueError("MemNO continuation source, config or data changed")
    for name, digest in pointer["files"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("MemNO committed checkpoint changed")
    state = torch.load(output / pointer["checkpoint"], map_location=device, weights_only=False)
    if state["binding"] != binding:
        raise ValueError("MemNO checkpoint binding differs")
    return state


def commit(output, model, optimizer, scheduler, state):
    name = f"checkpoints/epoch_{state['next_epoch']:04d}.pt"
    checkpoint = output / name
    checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        # A checkpoint written before pointer commit is recoverable evidence.
        index = 0
        while checkpoint.with_suffix(f".interrupted_{index}.pt").exists():
            index += 1
        checkpoint.rename(checkpoint.with_suffix(f".interrupted_{index}.pt"))
    if state["best_epoch"] == state["next_epoch"] - 1:
        state["best_checkpoint"] = name
    payload = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if model.lifting.weight.is_cuda else None)
    _save_checkpoint(payload, checkpoint)
    keep = {name, state["best_checkpoint"]} - {None}
    files = {path: sha256_file(output / path) for path in keep}
    write_json(output / "committed.json", dict(binding=state["binding"], checkpoint=name, files=files))
    write_json(output / "epochs.json", state["history"])
    write_json(output / "run.json", dict(status="FITTED_DEVELOPMENT" if state["stop_reason"] else "RUNNING",
        binding=state["binding"], implementation=IMPLEMENTATION, paper_url=PAPER_URL,
        training_seed=state["binding"]["training_seed"], completed_epochs=state["next_epoch"],
        best_epoch=state["best_epoch"], best_loss=state["best_loss"], best_checkpoint=state["best_checkpoint"],
        latest_checkpoint=name, checkpoint_sha256=files, elapsed_seconds=state["elapsed_seconds"],
        training_steps=state["training_steps"], parameter_count=sum(p.numel() for p in model.parameters()),
        fit_episodes=len(state["binding"]["fit_case_ids"]), validation_episodes=len(state["binding"]["validation_case_ids"]),
        complete_episode_bins=1200, source_units=100000, source_field_count=128,
        field_mean_only=True, joint_sampler=False, identified_unit_outputs=False,
        resumed=state["resumed"], stop_reason=state["stop_reason"],
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if model.lifting.weight.is_cuda else 0,
        hardware=dict(node=platform.node(), device=str(model.lifting.weight.device),
                      torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                      slurm_job_id=os.environ.get("SLURM_JOB_ID")),
        scientific_claim_qualified=False, recorded_utc=datetime.now(timezone.utc).isoformat()))
    # Only superseded task-owned complete model copies are removed. Histories,
    # selected models and interrupted checkpoints remain available.
    for path in checkpoint.parent.glob("epoch_*.pt"):
        if ".interrupted_" not in path.name and str(path.relative_to(output)) not in keep:
            path.unlink()


def train(acquisition, fields, config_path, output, seed, resume=False, stop_after_epochs=None):
    acquisition, fields, config_path, output = map(Path, (acquisition, fields, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("positive per-invocation epoch boundary required")
    if output.exists() and not resume:
        raise FileExistsError("MemNO existing output requires explicit continuation")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cohort = MemoryFieldCohort(acquisition, fields, config)
    binding = dict(config=config, training_seed=seed, dataset=cohort.field_binding(), sources=sources(),
        fit_case_ids=[case["case_id"] for case in cohort.fit],
        validation_case_ids=[case["case_id"] for case in cohort.validation],
        rate_scale=cohort.scale.tolist(), torch_version=torch.__version__)
    model = make_model(cohort, device)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["learning_rate_half_period"], gamma=.5)
    if resume:
        payload = load_committed(output, binding, device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        state = {key: value for key, value in payload.items()
                 if key not in ("model", "optimizer", "scheduler", "torch_rng", "cuda_rng")}
        state["resumed"] = True
    else:
        output.mkdir(parents=True)
        state = dict(binding=binding, next_epoch=0, best_epoch=None, best_loss=None, best_checkpoint=None,
                     history=[], elapsed_seconds=0., training_steps=0, resumed=False, stop_reason=None)
        commit(output, model, optimizer, scheduler, state)
    began, previous = time.monotonic(), state["elapsed_seconds"]
    invocation_epochs = 0
    try:
        while state["next_epoch"] < config["maximum_epochs"] and not state["stop_reason"]:
            fitting = evaluate_epoch(model, cohort, cohort.fit, config, optimizer)
            selection = evaluate_epoch(model, cohort, cohort.validation, config)
            epoch = state["next_epoch"]
            state["history"].append(dict(epoch=epoch, training=fitting, validation=selection,
                                         learning_rates=scheduler.get_last_lr()))
            if state["best_loss"] is None or selection["loss"] < state["best_loss"]:
                state.update(best_loss=selection["loss"], best_epoch=epoch)
            scheduler.step()
            state["next_epoch"] += 1
            state["training_steps"] += math.ceil(len(cohort.fit) / config["batch_size"])
            state["elapsed_seconds"] = previous + time.monotonic() - began
            if state["next_epoch"] == config["maximum_epochs"]:
                state["stop_reason"] = "maximum_epochs"
            commit(output, model, optimizer, scheduler, state)
            invocation_epochs += 1
            if stop_after_epochs and invocation_epochs >= stop_after_epochs:
                break
    except BaseException:
        write_json(output / "failure.json", dict(status="FAILED", traceback=traceback.format_exc(),
                    committed_epochs=state["next_epoch"], scientific_claim_qualified=False))
        raise
    return json.loads((output / "run.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args()
    report = train(args.acquisition, args.fields, args.config, args.output, args.seed,
                   args.resume, args.stop_after_epochs)
    print(json.dumps({key: report[key] for key in ("status", "training_seed", "completed_epochs", "best_loss")}))


if __name__ == "__main__":
    main()

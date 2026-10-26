"""Fit prefix-conditioned LDNS on the complete native development cohort.

Autoencoder selection precedes fitting-only latent normalization and diffusion
training. Target windows are encoded separately from observed prefixes. The
source epoch budgets are retained; software completion is not source qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from .latent_spike_diffusion import ControlledLatentSpikeDiffusion, PrefixCondition, SOURCE_REVISION
from .native_trajectory_cohort import TrajectoryCohort
from .poisson_trajectory_training import _save_checkpoint
from .provenance import sha256_file

FIXED_PROTOCOL = dict(
    schema="neural_population_latent_spike_diffusion_training_v1",
    training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
    bin_ms=5., complete_episode_bins=1200, expected_poisson_rng_family="per_edge_philox4_32_10",
    history_bins=200, horizon_bins=1000, forecast_origins_bins=[200, 600, 1000],
    forecast_bins=100, maximum_autoencoder_epochs=260, maximum_diffusion_epochs=2000,
    autoencoder_validation_every=10, diffusion_validation_every=100, validation_draws=4,
    autoencoder_warmup_epochs=10, diffusion_warmup_epochs=50,
    autoencoder_cosine_budget_factor=1.1, diffusion_cosine_budget_factor=1.3,
    learning_rate=.001, weight_decay=.01, mask_probability=.5,
    autoencoder_gradient_limit=2., diffusion_gradient_limit=1.,
    latent_beta=.001, temporal_beta=.2,
    autoencoder_sequence="complete_episode_plus_rotating_separate_prefix_and_future",
    latent_scale="all_fit_origins_separately_encoded_future_windows_population_sd",
    prefix_scale="same_fitting_target_moments",
    validation_randomness="fixed_per_seed_case_origin_draw",
    early_stopping=False, diffusion_ema_decay=.9999,
    diffusion_ema_rule="min_max_decay_and_one_minus_one_plus_step_over_ten_plus_step",
    selection_metric="fixed_mask_ae_loss_then_fixed_noise_ema_diffusion_loss",
    model=dict(latent_channels=16, hidden_channels=256, encoder_blocks=4,
               denoiser_blocks=6, diffusion_steps=1000),
    formal_training_completed=False,
)


def validate_config(config, seed):
    if any(config.get(key) != value for key, value in FIXED_PROTOCOL.items()):
        raise ValueError("frozen complete-cohort causal LDNS protocol required")
    if type(seed) is not int or seed not in config["training_seeds"]:
        raise ValueError("registered training seed41/42/43 required")
    if config.get("device") not in ("cpu", "cuda") or config.get("cpu_threads") not in (1, 2):
        raise ValueError("explicit supported device and bounded CPU threads required")
    if config["device"] == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU training requires CUDA_VISIBLE_DEVICES empty")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _seed(seed, *parts):
    # Validation streams are independent of training masks/noise and case order.
    digest = hashlib.sha256(json.dumps([seed, *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def _sources():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in
            ("latent_spike_diffusion_training.py", "latent_spike_diffusion.py",
             "native_trajectory_cohort.py", "native_count_windows.py", "provenance.py",
             "poisson_trajectory_training.py", "poisson_state_space.py")}


def _model(cohort, config):
    model = ControlledLatentSpikeDiffusion(tuple(map(int, cohort.unit_ids)), input_size=1,
                                          bin_seconds=config["bin_ms"] / 1000,
                                          **config["model"]).to(config["device"])
    # Source DPLR lazily changes C and l_kernel when a longer input first arrives.
    # Materialize the declared maximum lengths before optimizer creation. Later
    # validation and prefix lengths cannot change the parameter representation.
    for block in model.autoencoder.ae_net.encoder:
        block.time_mixer.kernel._setup_C(config["complete_episode_bins"])
    for block in model.denoiser.blocks:
        block.time_mixer.kernel._setup_C(config["forecast_bins"])
    return model


def _window_pairs(counts, controls, config):
    for origin in config["forecast_origins_bins"]:
        stop = origin + config["forecast_bins"]
        if origin < 2 or stop > counts.shape[1] or controls.shape[1] != counts.shape[1]:
            raise ValueError("complete aligned observed-prefix and future windows required")
        yield origin, counts[:, :origin], counts[:, origin:stop], controls[:, origin:stop]


def _ae_sequences(counts, controls, config, epoch, *, validation):
    # Every fit epoch reads the entire episode. Rotating windows train the exact
    # transforms used at all forecast origins without creating extra episodes.
    yield "complete", counts
    pairs = list(_window_pairs(counts, controls, config))
    selected = pairs if validation else [pairs[epoch % len(pairs)]]
    for origin, prefix, target, _ in selected:
        yield f"prefix_{origin}", prefix
        yield f"future_{origin}", target


def _scheduler(optimizer, updates, config, stage):
    warmup = updates * config[stage + "_warmup_epochs"]
    total = updates * config["maximum_" + stage + "_epochs"] * config[stage + "_cosine_budget_factor"]
    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        fraction = (step - warmup) / max(1, total - warmup)
        return max(0., .5 * (1 + math.cos(math.pi * fraction)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _optimizer(model, config, stage):
    module = model.autoencoder if stage == "autoencoder" else model.denoiser
    optimizer = torch.optim.AdamW(module.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    updates = 56 * (3 if stage == "autoencoder" else len(config["forecast_origins_bins"]))
    return optimizer, _scheduler(optimizer, updates, config, stage)


@torch.no_grad()
def _update_ema(ema, model, step, maximum):
    effective = max(0, step - 1)
    decay = 0. if effective == 0 else min(maximum, 1 - (1 + effective) / (10 + effective))
    for name, value in model.denoiser.state_dict().items():
        if value.is_floating_point() or value.is_complex():
            ema[name].sub_((ema[name] - value.detach()) * (1 - decay))
        else:
            ema[name].copy_(value)


def _ae_epoch(model, cohort, config, seed, epoch, rng, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    rows = []
    cases = cohort.fit if training else cohort.validation
    for case in cases:
        counts, controls = cohort.window(case)
        counts = torch.as_tensor(counts, device=config["device"], dtype=torch.float32)[None]
        controls = torch.as_tensor(controls, device=config["device"], dtype=torch.float32)[None]
        terms_by_window = []
        for name, sequence in _ae_sequences(counts, controls, config, epoch, validation=not training):
            draws = 1 if training else config["validation_draws"]
            for draw in range(draws):
                stream = rng if training else torch.Generator(device=config["device"]).manual_seed(
                    _seed(seed, "ae_validation", case["case_id"], name, draw))
                mask = torch.rand(sequence.transpose(1, 2).shape, device=config["device"],
                                  generator=stream) > config["mask_probability"]
                with torch.set_grad_enabled(training):
                    terms = model.autoencoder_loss(sequence, mask=mask,
                        mask_probability=config["mask_probability"], latent_beta=config["latent_beta"],
                        temporal_beta=config["temporal_beta"])
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        terms["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(),
                            config["autoencoder_gradient_limit"], error_if_nonfinite=True)
                        optimizer.step(); scheduler.step()
                values = {key: float(value.detach()) for key, value in terms.items()}
                if not all(math.isfinite(value) for value in values.values()):
                    raise FloatingPointError("nonfinite LDNS autoencoder loss")
                terms_by_window.append(dict(window=name, draw=draw, terms=values))
        rows.append(dict(case_id=case["case_id"], windows=terms_by_window))
    keys = rows[0]["windows"][0]["terms"]
    means = {key: float(np.mean([np.mean([r["terms"][key] for r in row["windows"]])
                                for row in rows])) for key in keys}
    return dict(mean=means, episodes=rows)


@torch.no_grad()
def build_latent_cache(model, cohort, config):
    """Fit moments on all fit targets; keep separate-prefix conditions unmixed."""
    model.eval()
    rows = {"fit": [], "validation": []}
    total = squared = None
    number = 0
    for split, cases in (("fit", cohort.fit), ("validation", cohort.validation)):
        for case in cases:
            counts, controls = cohort.window(case)
            counts = torch.as_tensor(counts, device=config["device"], dtype=torch.float32)[None]
            controls = torch.as_tensor(controls, device=config["device"], dtype=torch.float32)[None]
            for origin, prefix, target, future in _window_pairs(counts, controls, config):
                prefix_latent = model.autoencoder.encode(model._counts(prefix))[:, :, -1:]
                target_latent = model.autoencoder.encode(model._counts(target))
                if not torch.isfinite(prefix_latent).all() or not torch.isfinite(target_latent).all():
                    raise FloatingPointError("nonfinite separately encoded LDNS latent")
                if split == "fit":
                    value = target_latent.double()
                    if total is None:
                        total = value.sum((0, 2), keepdim=True)
                        squared = value.square().sum((0, 2), keepdim=True)
                    else:
                        total += value.sum((0, 2), keepdim=True)
                        squared += value.square().sum((0, 2), keepdim=True)
                    number += value.shape[0] * value.shape[2]
                rows[split].append(dict(case_id=case["case_id"], origin=origin,
                    prefix=prefix_latent.cpu(), target=target_latent.cpu(), stimulus=future.cpu()))
    mean = total / number
    variance = squared / number - mean.square()
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError("positive finite fitting-only LDNS target variation required")
    model.set_latent_scale(mean, variance.sqrt())
    mean, sd = model.latent_mean.cpu(), model.latent_sd.cpu()
    for values in rows.values():
        for row in values:
            row["prefix"] = (row["prefix"] - mean) / sd
            row["target"] = (row["target"] - mean) / sd
    return dict(rows=rows, moments=dict(fit_cases=[c["case_id"] for c in cohort.fit],
        fit_windows=len(rows["fit"]), fit_time_values_per_channel=number,
        mean=mean, population_sd=sd, transform=config["latent_scale"]))


def _diffusion_epoch(model, cache, config, seed, rng, optimizer=None, scheduler=None, ema=None, step=0):
    training = optimizer is not None
    model.denoiser.train(training)
    rows = []
    data = cache["rows"]["fit" if training else "validation"]
    for row in data:
        target = row["target"].to(config["device"])
        stimulus = row["stimulus"].to(config["device"])
        state = PrefixCondition(row["prefix"].to(config["device"]), model.unit_ids, row["origin"])
        terms = []
        for draw in range(1 if training else config["validation_draws"]):
            stream = rng if training else torch.Generator(device=config["device"]).manual_seed(
                _seed(seed, "diffusion_validation", row["case_id"], row["origin"], draw))
            noise = torch.randn(target.shape, device=config["device"], generator=stream)
            timestep = torch.randint(config["model"]["diffusion_steps"], (target.shape[0],),
                                     device=config["device"], generator=stream)
            with torch.set_grad_enabled(training):
                loss = model.diffusion_loss(target, state, stimulus, noise=noise, timestep=timestep)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite LDNS diffusion loss")
                if training:
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(),
                        config["diffusion_gradient_limit"], error_if_nonfinite=True)
                    optimizer.step(); scheduler.step(); step += 1
                    _update_ema(ema, model, step, config["diffusion_ema_decay"])
            terms.append(float(loss.detach()))
        rows.append(dict(case_id=row["case_id"], origin=row["origin"], losses=terms))
    return dict(mean=dict(loss=float(np.mean([np.mean(r["losses"]) for r in rows]))), windows=rows), step


def _commit(output, model, optimizer, scheduler, rng, state, ema):
    previous = json.loads((output / "committed.json").read_text()) if (output / "committed.json").exists() else None
    payload = dict(state=state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), rng=rng.get_state(), global_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if model.betas.is_cuda else None, ema=ema)
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / f"{state['stage']}_{state['next_epoch']:04d}.pt"
    # Each commit has a distinct destination. An interrupted write cannot replace
    # the checkpoint referenced by the previous atomic pointer.
    _save_checkpoint(payload, checkpoint)
    digest = sha256_file(checkpoint)
    _write_json(output / "committed.json", dict(checkpoint=str(checkpoint.relative_to(output)), sha256=digest,
                                               stage=state["stage"], next_epoch=state["next_epoch"]))
    _write_json(output / "epochs.json", state["history"])
    _write_json(output / "run.json", dict(status="FITTED_DEVELOPMENT" if state["stage"] == "complete" else "RUNNING",
        training_seed=state["binding"]["seed"], stage=state["stage"], completed_stage_epochs=state["next_epoch"],
        history_epochs=len(state["history"]), best_epoch=state["best_epoch"], best_loss=state["best_loss"],
        stage_steps=state["steps"], parameter_count=sum(p.numel() for p in model.parameters()),
        latest_checkpoint_sha256=digest, selected_checkpoint_sha256=state["selected_digest"],
        elapsed_seconds=state["elapsed_seconds"], resumed=state["resumed"],
        fit_episodes=56, validation_episodes=16, units=len(model.unit_ids),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_revision=SOURCE_REVISION, scientific_claim_qualified=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))
    if previous is not None and previous["checkpoint"] != str(checkpoint.relative_to(output)):
        # Retain selected stage checkpoints and complete loss history. A previous
        # complete optimizer checkpoint has a verified replacement at this point.
        (output / previous["checkpoint"]).unlink()


def train(acquisition, fields, config_path, output, seed, resume=False, *, stop_after_epochs=None):
    acquisition, fields, config_path, output = map(Path, (acquisition, fields, config_path, output))
    config = yaml.safe_load(config_path.read_text()); validate_config(config, seed)
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("positive per-invocation epoch boundary required")
    if output.exists() and not resume:
        raise FileExistsError("existing LDNS output requires explicit resume")
    torch.set_num_threads(config["cpu_threads"]); torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    cohort = TrajectoryCohort(acquisition, fields, config)
    binding = dict(config=config, seed=seed, cohort=cohort.binding, sources=_sources(),
        source_revision=SOURCE_REVISION, torch_version=torch.__version__)
    model = _model(cohort, config)
    rng = torch.Generator(device=config["device"]).manual_seed(_seed(seed, "training"))
    cache = ema = None
    if resume:
        pointer = json.loads((output / "committed.json").read_text())
        if sha256_file(output / pointer["checkpoint"]) != pointer["sha256"]:
            raise ValueError("LDNS committed checkpoint changed")
        payload = torch.load(output / pointer["checkpoint"], map_location=config["device"], weights_only=False)
        state = payload["state"]
        if state["binding"] != binding or pointer["stage"] != state["stage"] or pointer["next_epoch"] != state["next_epoch"]:
            raise ValueError("LDNS resume source/data/protocol or epoch identity differs")
        model.load_state_dict(payload["model"]); state["resumed"] = True
        rng.set_state(payload["rng"].cpu()); torch.set_rng_state(payload["global_rng"].cpu())
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in payload["cuda_rng"]])
        ema = payload["ema"]
        if state["stage"] == "complete":
            return json.loads((output / "run.json").read_text())
        optimizer, scheduler = _optimizer(model, config, state["stage"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        if state["stage"] == "diffusion":
            if sha256_file(output / "latent_cache.pt") != state["cache_digest"]:
                raise ValueError("LDNS fitting-only latent cache changed")
            cache = torch.load(output / "latent_cache.pt", map_location="cpu", weights_only=False)
    else:
        output.mkdir(parents=True)
        state = dict(binding=binding, stage="autoencoder", next_epoch=0, best_loss=None, best_epoch=None,
                     selected_digest=None, selected_path=None, cache_digest=None, history=[], steps=0, elapsed_seconds=0., resumed=False)
        optimizer, scheduler = _optimizer(model, config, "autoencoder")
        _commit(output, model, optimizer, scheduler, rng, state, ema)
    start, previous, invocation_epochs = time.monotonic(), state["elapsed_seconds"], 0
    try:
        while state["stage"] != "complete":
            stage, epoch = state["stage"], state["next_epoch"]
            maximum = config["maximum_" + stage + "_epochs"]
            if epoch == maximum:
                selected = output / state["selected_path"]
                if sha256_file(selected) != state["selected_digest"]:
                    raise ValueError("LDNS selected stage checkpoint changed")
                saved = torch.load(selected, map_location=config["device"], weights_only=False)
                if saved["binding"] != binding or saved["stage"] != stage or saved["epoch"] != state["best_epoch"]:
                    raise ValueError("LDNS selected stage identity changed")
                model.load_state_dict(saved["model"])
                if stage == "diffusion":
                    state["stage"] = "complete"
                    state["elapsed_seconds"] = previous + time.monotonic() - start
                    _commit(output, model, optimizer, scheduler, rng, state, ema)
                    break
                cache = build_latent_cache(model, cohort, config)
                _save_checkpoint(cache, output / "latent_cache.pt")
                state.update(stage="diffusion", next_epoch=0, best_loss=None, best_epoch=None,
                             selected_digest=None, selected_path=None,
                             cache_digest=sha256_file(output / "latent_cache.pt"), steps=0)
                with torch.no_grad():
                    model.denoiser.conv_out.weight.mul_(.1); model.denoiser.conv_out.bias.mul_(.1)
                ema = {key: value.detach().clone() for key, value in model.denoiser.state_dict().items()}
                optimizer, scheduler = _optimizer(model, config, "diffusion")
                state["elapsed_seconds"] = previous + time.monotonic() - start
                _commit(output, model, optimizer, scheduler, rng, state, ema)
                continue
            if stage == "autoencoder":
                fitting = _ae_epoch(model, cohort, config, seed, epoch, rng, optimizer, scheduler)
                state["steps"] += len(cohort.fit) * 3
            else:
                fitting, state["steps"] = _diffusion_epoch(model, cache, config, seed, rng,
                    optimizer, scheduler, ema, state["steps"])
            validation = None
            if (epoch + 1) % config[stage + "_validation_every"] == 0 or epoch + 1 == maximum:
                if stage == "autoencoder":
                    validation = _ae_epoch(model, cohort, config, seed, epoch, rng)
                else:
                    raw = {key: value.detach().clone() for key, value in model.denoiser.state_dict().items()}
                    try:
                        model.denoiser.load_state_dict(ema)
                        validation, _ = _diffusion_epoch(model, cache, config, seed, rng)
                    finally:
                        model.denoiser.load_state_dict(raw)
                score = validation["mean"]["loss"]
                if state["best_loss"] is None or score < state["best_loss"]:
                    state.update(best_loss=score, best_epoch=epoch)
                    selected_state = dict(model.state_dict())
                    if stage == "diffusion":
                        selected_state.update({"denoiser." + key: value for key, value in ema.items()})
                    path = output / "checkpoints" / f"selected_{stage}_{epoch:04d}.pt"
                    _save_checkpoint(dict(model=selected_state, binding=binding, stage=stage, epoch=epoch,
                                          score=score, cache_digest=state["cache_digest"]), path)
                    state["selected_digest"] = sha256_file(path)
                    state["selected_path"] = str(path.relative_to(output))
            state["history"].append(dict(stage=stage, epoch=epoch, training=fitting, validation=validation))
            state["next_epoch"] = epoch + 1
            state["elapsed_seconds"] = previous + time.monotonic() - start
            _commit(output, model, optimizer, scheduler, rng, state, ema)
            invocation_epochs += 1
            if stop_after_epochs is not None and invocation_epochs >= stop_after_epochs:
                break
        return json.loads((output / "run.json").read_text())
    except Exception as exc:
        name = "failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"
        _write_json(output / name, dict(status="FAILED", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), stage=state["stage"], incomplete_epoch=state["next_epoch"],
            scientific_claim_qualified=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args()
    report = train(args.acquisition, args.fields, args.config, args.output, args.seed, args.resume,
                   stop_after_epochs=args.stop_after_epochs)
    print(json.dumps(report)); return 0


if __name__ == "__main__":
    sys.exit(main())

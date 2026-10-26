"""Complete native-episode BRAID mean forecasting development training.

The source regression RNN and trial-local multi-horizon objective are retained.
The fixed development split is shared with the population model. Selection
loss is not a free-forecast score or a comparison of joint observation laws.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
import tensorflow as tf
import yaml

from .braid import BRAIDMeanPredictor, SOURCE_REVISION
from .native_trajectory_cohort import TrajectoryCohort
from .provenance import sha256_file

FIXED_PROTOCOL = dict(
    schema="neural_population_braid_trajectory_training_v1",
    training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
    bin_ms=5., dt_ms=.1, duration_ms=6000., expected_units=100000,
    history_bins=200, horizon_bins=1000, maximum_epochs=20, early_stopping_patience=4,
    expected_poisson_rng_family="per_edge_philox4_32_10", cpu_threads=2,
    selection_metric="native_trial_local_multi_horizon_poisson_loss",
    formal_training_completed=False,
    model=dict(latent_dim=16, hidden_units=32, steps_ahead=[1, 4, 16],
               learning_rate=.001, gradient_clip_norm=1.),
)


def validate_config(config, seed):
    if any(config.get(k) != v for k, v in FIXED_PROTOCOL.items()):
        raise ValueError("complete source-bound BRAID development protocol required")
    if type(seed) is not int or seed not in config["training_seeds"]:
        raise ValueError("registered training seed41/42/43 required")
    if config.get("device") != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("BRAID requires explicit CPU placement and CUDA_VISIBLE_DEVICES empty")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sources():
    package = Path(__file__).parent
    names = ("braid_trajectory_training.py", "braid.py", "native_trajectory_cohort.py",
             "native_count_windows.py", "nest_gpu_network.py", "network_snapshot.py",
             "nest_gpu_connectivity.py", "nest_reference.py", "mechanistic.py", "provenance.py")
    result = {name: sha256_file(package / name) for name in names}
    native = package.parent / "external/BRAID"
    # These maintained source files are the actual vendored BRAID implementation.
    for path in sorted(native.rglob("*.py")):
        result["external/BRAID/" + str(path.relative_to(native))] = sha256_file(path)
    result["external/BRAID/LICENSE.md"] = sha256_file(native / "LICENSE.md")
    return result


def _binding(cohort, config, seed):
    return dict(config=config, training_seed=seed, source_revision=SOURCE_REVISION,
                tensorflow_version=tf.__version__, dataset=cohort.binding, sources=_sources())


def _losses(model, counts, controls, *, training):
    if training:
        return model.fit_step(counts[None], controls[None])
    steps_before = model.training_steps
    adam_before = int(model.native.model.optimizer.iterations.numpy())
    inputs, labels = model.training_tensors(counts[None], controls[None])
    values = model.native.model.test_on_batch(inputs, labels, return_dict=True)
    result = dict(loss=float(values["loss"]))
    for step in model.steps_ahead:
        result[f"L_pred_h{step}"] = float(values[f"{model.native.rnn.name}_{step}step_loss"])
    if model.training_steps != steps_before or int(model.native.model.optimizer.iterations.numpy()) != adam_before:
        raise RuntimeError("BRAID validation changed the training optimizer")
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError("BRAID nonfinite validation loss")
    return result


def _evaluate(model, cohort, cases, *, training):
    rows = []
    for case in cases:
        counts, controls = cohort.window(case)
        terms = _losses(model, counts, controls, training=training)
        rows.append(dict(case_id=case["case_id"], block=case["block"], terms=terms))
    return dict(mean={key: float(np.mean([r["terms"][key] for r in rows]))
                      for key in rows[0]["terms"]}, episodes=rows)


def _checkpoint_files(path):
    return {name: sha256_file(path / name) for name in
            ("model.json", "model.index", "model.data-00000-of-00001")}


def _restore(output, binding):
    pointer = json.loads((output / "committed.json").read_text())
    if pointer["binding"] != binding:
        raise ValueError("BRAID resume configuration/data/source binding changed")
    path = output / pointer["checkpoint"]
    if _checkpoint_files(path) != pointer["checkpoint_files"]:
        raise ValueError("BRAID committed checkpoint changed")
    state = json.loads((path / "training.json").read_text())
    if sha256_file(path / "training.json") != pointer["training_sha256"] or state["binding"] != binding:
        raise ValueError("BRAID committed epoch state changed")
    model = BRAIDMeanPredictor.load(path)
    if state["best_checkpoint"] is not None:
        best = output / state["best_checkpoint"]
        best_state = json.loads((best / "training.json").read_text())
        if (_checkpoint_files(best) != state["best_checkpoint_files"]
                or best_state["binding"] != binding or best_state["best_epoch"] != state["best_epoch"]):
            raise ValueError("BRAID selected checkpoint binding changed")
    if model.training_steps != state["training_steps"]:
        raise ValueError("BRAID model update count differs from committed epoch")
    return model, state


def _commit(output, model, state):
    root = output / "checkpoints"
    root.mkdir(exist_ok=True)
    epoch = state["next_epoch"] - 1
    path = root / f"epoch_{epoch:03d}" if epoch >= 0 else root / "initial"
    if path.exists():
        # This directory has no committed pointer; retain interrupted output.
        if (output / "committed.json").exists():
            prior = json.loads((output / "committed.json").read_text())
            if prior["checkpoint"] == str(path.relative_to(output)):
                raise ValueError("BRAID would overwrite a committed checkpoint")
        attempt = 0
        while path.with_name(path.name + f"_interrupted_{attempt:03d}").exists():
            attempt += 1
        path.rename(path.with_name(path.name + f"_interrupted_{attempt:03d}"))
    model.save(path)
    if state["best_checkpoint"] == str(path.relative_to(output)):
        state["best_checkpoint_files"] = _checkpoint_files(path)
    _write_json(path / "training.json", state)
    pointer = dict(binding=state["binding"], checkpoint=str(path.relative_to(output)),
                   checkpoint_files=_checkpoint_files(path), training_sha256=sha256_file(path / "training.json"))
    _write_json(output / "committed.json", pointer)
    _write_json(output / "epochs.json", state["history"])
    # Intermediate complete model copies are superseded by latest/best. The
    # complete loss history and interrupted checkpoint directories are retained.
    keep = {pointer["checkpoint"], state["best_checkpoint"]}
    for old in root.iterdir():
        if (not old.is_dir() or "_interrupted_" in old.name
                or str(old.relative_to(output)) in keep or not (old / "training.json").exists()):
            continue
        prior_state = json.loads((old / "training.json").read_text())
        if prior_state["binding"] == state["binding"] and prior_state["next_epoch"] < state["next_epoch"]:
            shutil.rmtree(old)
    _write_json(output / "run.json", dict(
        status="FITTED_DEVELOPMENT" if state["stop_reason"] else "RUNNING",
        training_seed=state["binding"]["training_seed"], completed_epochs=state["next_epoch"],
        best_epoch=state["best_epoch"], best_checkpoint=state["best_checkpoint"],
        latest_checkpoint=pointer["checkpoint"], training_steps=model.training_steps,
        parameter_count=int(model.native.model.count_params()),
        fit_episodes=56, validation_episodes=16, complete_episode_bins=1200, units=100000,
        training_origins_per_episode_per_head=model.block_samples,
        maximum_training_horizon_bins=max(model.steps_ahead),
        common_origin_tail_bins=max(model.steps_ahead),
        elapsed_seconds=state["elapsed_seconds"], peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        resumed=state["resumed"], stop_reason=state["stop_reason"],
        joint_spike_sampler=False, scientific_claim_qualified=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))


def train(acquisition, fields, config_path, output, seed, resume=False, *, stop_after_epochs=None):
    acquisition, fields, config_path, output = map(Path, (acquisition, fields, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("positive per-invocation epoch boundary required")
    if output.exists() and not resume:
        raise FileExistsError("BRAID existing output requires explicit resume")
    if resume and not (output / "committed.json").exists():
        raise ValueError("BRAID resume requires a committed epoch")
    if tf.config.threading.get_intra_op_parallelism_threads() != config["cpu_threads"]:
        tf.config.threading.set_intra_op_parallelism_threads(config["cpu_threads"])
    if tf.config.threading.get_inter_op_parallelism_threads() != 1:
        tf.config.threading.set_inter_op_parallelism_threads(1)
    cohort = TrajectoryCohort(acquisition, fields, config)
    binding = _binding(cohort, config, seed)
    if resume:
        model, state = _restore(output, binding)
        state["resumed"] = True
    else:
        model = BRAIDMeanPredictor(tuple(map(int, cohort.unit_ids)), 1, config["history_bins"],
                                  config["horizon_bins"], config["bin_ms"] / 1000,
                                  config["model"], initialization_seed=seed)
        output.mkdir(parents=True)
        state = dict(binding=binding, next_epoch=0, best_epoch=None, best_checkpoint=None,
                     best_loss=None, stale_epochs=0, history=[], training_steps=0,
                     elapsed_seconds=0., resumed=False, stop_reason=None)
        _commit(output, model, state)
    start, previous = time.monotonic(), state["elapsed_seconds"]
    invocation_epochs = 0
    try:
        while state["next_epoch"] < config["maximum_epochs"] and not state["stop_reason"]:
            epoch = state["next_epoch"]
            fitting = _evaluate(model, cohort, cohort.fit, training=True)
            validation = _evaluate(model, cohort, cohort.validation, training=False)
            score = validation["mean"]["loss"]
            state["history"].append(dict(epoch=epoch, training=fitting, validation=validation))
            if state["best_loss"] is None or score < state["best_loss"]:
                state.update(best_loss=score, best_epoch=epoch,
                             best_checkpoint=f"checkpoints/epoch_{epoch:03d}", stale_epochs=0)
            else:
                state["stale_epochs"] += 1
            state.update(next_epoch=epoch + 1, training_steps=model.training_steps,
                         elapsed_seconds=previous + time.monotonic() - start)
            if state["stale_epochs"] >= config["early_stopping_patience"]:
                state["stop_reason"] = "early_stopping"
            elif state["next_epoch"] == config["maximum_epochs"]:
                state["stop_reason"] = "epoch_budget"
            _commit(output, model, state)
            invocation_epochs += 1
            if stop_after_epochs is not None and invocation_epochs >= stop_after_epochs:
                break
        return json.loads((output / "run.json").read_text())
    except Exception as exc:
        _write_json(output / "failure.json", dict(status="FAILED", error_type=type(exc).__name__,
                    error=str(exc), traceback=traceback.format_exc(), training_seed=seed,
                    incomplete_epoch=state["next_epoch"], last_committed_epoch=state["next_epoch"] - 1,
                    recorded_utc=datetime.now(timezone.utc).isoformat()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.acquisition, args.fields, args.config, args.output, args.seed,
                           args.resume, stop_after_epochs=args.stop_after_epochs)))


if __name__ == "__main__":
    main()

"""Train the existing controlled Poisson LDS on complete native episodes.

This runner uses only the prospectively acquired training cohort. Whole blocks
hold out development validation. ELBO selection is kept distinct from later
open-loop distribution evaluation and source-fidelity qualification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
import yaml

from .native_count_windows import NativeEpisodeCounts
from .native_trajectory_cohort import partition
from .poisson_state_space import ControlledPoissonLDS
from .provenance import sha256_file



def _save_checkpoint(payload, path):
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    temporary.replace(path)


# These are the two immutable campaign-runtime trainer revisions used by the
# original 20-epoch comparison. No other changed trainer revision is accepted.
LEGACY_TRAINER_SOURCES = {
    "poisson_trajectory_training.py": "fe9eb2fe5209d06a0963998742856c0f1accb79f948913328dc445c82531e193",
    "renewal_trajectory_training.py": "247a15956ac5abdbbc7def14b21d87c636199fc87ba97a9769347240bf2d1ef6",
}
CONTINUATION_SCHEMA = "neural_population_trajectory_budget_continuation_v1"
PRIOR_REFIT_SCHEMA = "neural_population_prior_refit_v1"
PRIOR_MIXTURE_SCHEMA = "neural_population_prior_predictive_mixture_v1"
PRIOR_OBSERVATION_SCHEMA = "neural_population_native_observation_prior_v1"
PRIOR_OBJECTIVES = {
    PRIOR_OBSERVATION_SCHEMA: ("fixed_readout_native_event_path_mixture_nll", "validation_native_event_path_mixture_nll"),
    PRIOR_REFIT_SCHEMA: ("fixed_posterior_multistep_prior_kl", "validation_multistep_prior_kl"),
    PRIOR_MIXTURE_SCHEMA: ("fixed_posterior_joint_predictive_mixture_kl", "validation_joint_predictive_mixture_kl"),
}
# Accepted 80-epoch prior runs use these trainer/control-flow revisions.
# Numerical model, observation, loader and prior-objective sources must match.
PRIOR_BUDGET_TRAINER_SOURCES = {
    "poisson_trajectory_training.py": "be4c675d9f6080d5c3120be33b2bd64499f508af5ed9a3a12c381e430b938264",
    "renewal_trajectory_training.py": "10a1f41f84e09c1d79c5ae61bd21b93a80bdbbdc8799db1fe016c9a90402b7dc",
}

# The accepted predictive-mixture producer is immutable run_626306. Only these
# trainer revisions may migrate; the mixture objective and all numerical sources
# must remain byte-identical to that parent.
MIXTURE_BUDGET_TRAINER_SOURCES = {
    "poisson_trajectory_training.py": "fe3204e92e8ad7b06379f3c45fa9a615f1f9294e71c77978e5e5324e9a98075f",
    "renewal_trajectory_training.py": "b00d0fb7e4107a4fdaf7b488a34c2acf001d915fd39eef763cb0cde0779f0323",
}


def validate_prior_budget_binding(parent_binding, binding, parent_sources, sources):
    """Admit exact prior identity or its registered trainer-only migration."""
    config = binding.get("config", {})
    schema = config.get("schema")
    registered_trainers = ({} if schema == PRIOR_OBSERVATION_SCHEMA else
                           MIXTURE_BUDGET_TRAINER_SOURCES if schema == PRIOR_MIXTURE_SCHEMA
                           else PRIOR_BUDGET_TRAINER_SOURCES)
    required = {"poisson_trajectory_training.py", "renewal_trajectory_training.py",
                "population_prior_refit.py", "population_prior_objective.py"}
    if schema == PRIOR_OBSERVATION_SCHEMA:
        required.add("population_observation_prior.py")
    if schema == PRIOR_MIXTURE_SCHEMA:
        required.add("population_predictive_mixture.py")
    if (schema not in PRIOR_OBJECTIVES
            or config.get("observation_objective") != PRIOR_OBJECTIVES[schema][0]
            or not isinstance(parent_sources, dict) or not isinstance(sources, dict)
            or not required <= set(sources)):
        raise ValueError("prior budget continuation requires its registered numerical source set")
    aliases = {"renewal_trajectory_training.py": "trainer_implementation_sha256",
               "poisson_trajectory_training.py": "training_utilities_implementation_sha256"}
    if any(binding.get(key) != sources[name] for name, key in aliases.items()):
        raise ValueError("current prior trainer binding differs from source hashes")
    legacy_sources = {name: registered_trainers.get(name, digest)
                      for name, digest in sources.items()}
    legacy_binding = dict(binding, **{key: registered_trainers.get(name, sources[name])
                                     for name, key in aliases.items()})
    if parent_sources == sources:
        expected = binding
    elif parent_sources == legacy_sources:
        expected = legacy_binding
    else:
        raise ValueError("prior budget numerical or unregistered trainer source revision differs")
    if parent_binding != expected:
        raise ValueError("prior budget config/data/native/numerical source binding differs")
    return {name: dict(parent=parent_sources[name], current=digest)
            for name, digest in sources.items() if parent_sources[name] != digest}


def _execution_options(output, resume, resume_from, total_epochs, parent_checkpoint_sha256, stop_after_epochs):
    if resume and resume_from is not None:
        raise ValueError("ordinary resume and parent continuation are mutually exclusive")
    if resume_from is None and (total_epochs is not None or parent_checkpoint_sha256 is not None):
        raise ValueError("total epochs and parent checksum require explicit resume-from")
    if resume_from is not None:
        if type(total_epochs) is not int or total_epochs < 1:
            raise ValueError("explicit positive prospective total-epochs required")
        parent = Path(resume_from).resolve()
        if output.resolve() == parent or parent in output.resolve().parents:
            raise ValueError("continuation output must not modify the parent run")
        if output.exists():
            raise FileExistsError("continuation requires a new empty output path")
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("positive per-invocation epoch boundary required")


def _trainer_sources(model_type):
    if model_type not in ("poisson", "renewal", "population"):
        raise ValueError("registered trajectory model family required")
    names = ["poisson_trajectory_training.py", "poisson_state_space.py", "native_count_windows.py", "native_trajectory_cohort.py"]
    if model_type == "renewal":
        names += ["renewal_trajectory_training.py", "renewal_state_space.py",
                  "refractory_observation.py", "native_renewal_data.py"]
    if model_type == "population":
        names += ["population_trajectory_training.py", "renewal_trajectory_training.py",
                  "population_state.py", "renewal_state_space.py", "refractory_observation.py", "native_renewal_data.py"]
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _checkpoint_budget(checkpoint, config):
    base = config["maximum_epochs"]
    effective = checkpoint.get("effective_maximum_epochs", base)
    continuation = checkpoint.get("continuation")
    if continuation is not None and not isinstance(continuation, dict):
        raise ValueError("checkpoint continuation budget provenance invalid")
    if (type(effective) is not int or effective < base
            or (continuation is None and effective != base)
            or (continuation is not None and (
                continuation.get("schema") != CONTINUATION_SCHEMA
                or continuation.get("total_epochs") != effective
                or continuation.get("base_maximum_epochs") != base
                or effective <= continuation.get("parent_effective_maximum_epochs", effective)))):
        raise ValueError("checkpoint continuation budget provenance invalid")
    return effective, continuation


def _parent_state(checkpoint, report, history):
    epochs = checkpoint["next_epoch"]
    if (type(epochs) is not int or epochs < 1 or report.get("completed_epochs") != epochs
            or checkpoint["history"] != history or len(history) != epochs
            or [row["epoch"] for row in history] != list(range(epochs))):
        raise ValueError("parent committed history or completed epoch boundary differs")
    for row in history:
        if any(not math.isfinite(value) for section in ("training", "validation") for value in row[section].values()):
            raise ValueError("parent committed history contains nonfinite values")
    losses = [row["validation"]["loss"] for row in history]
    best_epoch = min(range(epochs), key=losses.__getitem__)
    prior_refit = checkpoint["binding"]["config"].get("schema") in PRIOR_OBJECTIVES
    metric = "best_validation_prior_loss" if prior_refit else "best_validation_negative_elbo"
    if (checkpoint["best"] != losses[best_epoch] or checkpoint["stale"] != epochs - best_epoch - 1
            or report.get(metric) != checkpoint["best"]
            or report.get("stale_epochs") != checkpoint["stale"]
            or checkpoint.get("best_epoch", best_epoch) != best_epoch
            or report.get("best_epoch", best_epoch) != best_epoch):
        raise ValueError("parent best/stale state differs from committed validation history")
    return best_epoch


def _prior_cache_artifacts(parent_run, cases):
    """Bind the complete existing cache; continuation must not re-encode cases."""
    if (not cases or any(not isinstance(case, str) or not case for case in cases)
            or len(set(cases)) != len(cases)):
        raise ValueError("complete unique prior cache cases required")
    names = ["posterior_cache/" + case + ".pt" for case in cases]
    if any(not (Path(parent_run) / name).is_file() for name in names):
        raise ValueError("prior continuation requires every bound posterior cache")
    return {name: sha256_file(Path(parent_run) / name) for name in names}


def _prepare_continuation(parent_run, total_epochs, binding, sources, model_type, declared_digest=None,
                          *, prior_cache_cases=None):
    """Validate a terminal parent before creating any child artifact."""
    parent_run = Path(parent_run)
    prior_refit = model_type == "population" and binding["config"].get("schema") in PRIOR_OBJECTIVES
    if not prior_refit and prior_cache_cases is not None:
        raise ValueError("posterior cache continuation requires the registered prior objective")
    report = json.loads((parent_run / "run.json").read_text())
    if report.get("status") not in ("FITTED_DEVELOPMENT", "CHECKPOINTED_DEVELOPMENT"):
        raise ValueError("terminal committed parent run required for continuation")
    latest_path = parent_run / "latest.pt"
    actual_digest = sha256_file(latest_path)
    recorded_digest = report.get("latest_checkpoint_sha256")
    if recorded_digest is None:
        if model_type != "poisson":
            raise ValueError("renewal/population parent requires an existing latest checkpoint receipt")
        if declared_digest is None:
            raise ValueError("legacy Poisson parent requires prospectively registered parent-checkpoint-sha256")
    for digest in (recorded_digest, declared_digest):
        if digest is not None and (type(digest) is not str or len(digest) != 64 or digest != actual_digest):
            raise ValueError("parent latest checkpoint checksum differs")
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
    parent_binding = checkpoint["binding"]
    if report["binding"] != parent_binding or report.get("training_seed") != parent_binding["seed"]:
        raise ValueError("parent report/checkpoint binding differs")
    legacy_binding = dict(binding)
    if model_type == "renewal":
        legacy_binding.update(
            trainer_implementation_sha256=LEGACY_TRAINER_SOURCES["renewal_trajectory_training.py"],
            training_utilities_implementation_sha256=LEGACY_TRAINER_SOURCES["poisson_trajectory_training.py"])
    if not prior_refit and parent_binding != binding and (model_type != "renewal" or parent_binding != legacy_binding):
        raise ValueError("parent config/data/native/numerical source binding differs")
    legacy_sources = ({name: LEGACY_TRAINER_SOURCES.get(name, digest) for name, digest in sources.items()}
                      if model_type in ("poisson", "renewal") else sources)
    parent_sources = checkpoint.get("trainer_sources")
    source_attribution = "checkpoint_recorded_sources"
    if parent_sources is None:
        if model_type == "population":
            raise ValueError("population parent requires checkpoint-recorded trainer source hashes")
        # The original campaign layout keeps the pinned runtime beside model runs.
        runtime = parent_run.parent.parent / "runtime" / "Libs" / "population_dynamics"
        if any(not (runtime / name).is_file() for name in sources):
            raise ValueError("legacy parent requires its pinned sibling runtime source files")
        parent_sources = {name: sha256_file(runtime / name) for name in sources}
        source_attribution = "pinned_sibling_runtime; legacy checkpoint did not record trainer_sources"
        if parent_sources != legacy_sources:
            raise ValueError("legacy parent runtime differs from the explicitly pinned source revision")
    if prior_refit:
        source_migration = validate_prior_budget_binding(parent_binding, binding, parent_sources, sources)
    else:
        if parent_sources not in (sources, legacy_sources):
            raise ValueError("parent numerical or unregistered trainer source revision differs")
        source_migration = {name: dict(parent=parent_sources[name], current=digest)
                            for name, digest in sources.items() if parent_sources[name] != digest}
    if report.get("trainer_sources", None if model_type == "population" else parent_sources) != parent_sources:
        raise ValueError("parent report/checkpoint trainer source hashes differ")
    if model_type == "renewal":
        expected = binding if parent_sources == sources else legacy_binding
        if parent_binding != expected:
            raise ValueError("parent trainer binding and source migration revision differ")
    if prior_refit:
        initialization = checkpoint.get("initialization")
        objective, selection = PRIOR_OBJECTIVES[binding["config"]["schema"]]
        if (report.get("training_objective") != objective
                or report.get("selection_metric") != selection
                or not isinstance(initialization, dict)
                or initialization.get("mode") != "selected_parent_prior_refit"
                or initialization != report.get("refit_initialization")
                or initialization.get("parent_checkpoint_sha256")
                != binding["config"]["parent_checkpoint_sha256"][str(binding["seed"])]
                or report.get("posterior_targets_fixed") is not True
                or report.get("observation_parameters_refitted") is not False):
            raise ValueError("prior objective or original frozen-parent initialization differs")
        if (binding["config"]["schema"] == PRIOR_MIXTURE_SCHEMA and any(
                report.get(key) != binding["config"][key]
                for key in ("prediction_particles", "posterior_target_draws"))):
            raise ValueError("predictive-mixture particle or target-draw settings differ")
        if (binding["config"]["schema"] == PRIOR_OBSERVATION_SCHEMA
                and report.get("prediction_particles") != binding["config"]["prediction_particles"]):
            raise ValueError("native observation mixture particle settings differ")
    history = json.loads((parent_run / "epochs.json").read_text())
    best_epoch = _parent_state(checkpoint, report, history)
    parent_budget, parent_continuation = _checkpoint_budget(checkpoint, binding["config"])
    if (report.get("effective_maximum_epochs", binding["config"]["maximum_epochs"]) != parent_budget
            or report.get("continuation") != parent_continuation):
        raise ValueError("parent run/checkpoint effective budget differs")
    if total_epochs <= max(parent_budget, checkpoint["next_epoch"]):
        raise ValueError("prospective total-epochs must exceed parent completed and registered ceilings")
    best_path = parent_run / "best.pt"
    if sha256_file(best_path) != report["best_checkpoint_sha256"]:
        raise ValueError("parent selected checkpoint checksum differs")
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    if (selected["binding"] != parent_binding or selected["next_epoch"] != best_epoch + 1
            or selected["best"] != checkpoint["best"] or selected["history"] != history[:best_epoch + 1]
            or selected.get("model_configuration") != checkpoint.get("model_configuration")
            or selected.get("trainer_sources", parent_sources) != parent_sources):
        raise ValueError("parent selected checkpoint differs from the committed best epoch")
    if model_type in ("renewal", "population") and ("variational_rng" not in checkpoint or not torch.is_tensor(checkpoint["variational_rng"])):
        raise ValueError("parent renewal/population variational RNG state unavailable")
    if prior_refit:
        if not torch.is_tensor(checkpoint.get("prior_dynamics_rng")):
            raise ValueError("parent prior-dynamics RNG state unavailable")
        if (binding["config"]["schema"] == PRIOR_MIXTURE_SCHEMA
                and not torch.is_tensor(checkpoint.get("prior_target_rng"))):
            raise ValueError("parent predictive-mixture target RNG state unavailable")
        if selected.get("initialization") != checkpoint["initialization"]:
            raise ValueError("selected prior checkpoint original initialization differs")
    artifacts = {name: sha256_file(parent_run / name) for name in
                 ("run.json", "epochs.json", "best.pt", "optimization.csv", "validation.csv")
                 if (parent_run / name).is_file()}
    if prior_refit:
        artifacts.update(_prior_cache_artifacts(parent_run, prior_cache_cases))
    continuation = dict(schema=CONTINUATION_SCHEMA, total_epochs=total_epochs,
        base_maximum_epochs=binding["config"]["maximum_epochs"], parent_effective_maximum_epochs=parent_budget,
        parent_completed_epochs=checkpoint["next_epoch"], parent_best_epoch=best_epoch,
        parent_run=str(parent_run.resolve()), parent_checkpoint_sha256=actual_digest,
        parent_checkpoint_hash_qualification=("existing_parent_latest_receipt" if recorded_digest is not None
                                             else "prospectively_bound_parent_snapshot_not_retrospective_execution_authentication"),
        parent_binding=parent_binding, parent_source_hashes=parent_sources, current_source_hashes=sources,
        parent_source_attribution=source_attribution, parent_artifact_sha256=artifacts,
        source_migration=source_migration,
        inherited_early_stopping_patience=binding["config"]["early_stopping_patience"],
        inherited_stale_epochs=checkpoint["stale"], history_prefix_epochs=len(history),
        training_rng=("explicit_checkpointed_variational_generator" if model_type in ("renewal", "population")
                      else "analytic_objective; epoch_shuffle_reconstructed_from_training_seed_and_absolute_epoch"),
        scientific_claim_qualified=False)
    if prior_refit:
        continuation["prior_dynamics_rng"] = "explicit_checkpointed_generator"
        if binding["config"]["schema"] == PRIOR_MIXTURE_SCHEMA:
            continuation["prior_target_rng"] = "explicit_checkpointed_generator"
    return checkpoint, selected, continuation


def _seed_continuation_output(output, parent_run, selected, binding, sources, continuation):
    output.mkdir(parents=True, exist_ok=False)
    names = ["optimization.csv", "validation.csv"]
    names += sorted(name for name in continuation["parent_artifact_sha256"]
                    if name.startswith("posterior_cache/"))
    for name in names:
        if name in continuation["parent_artifact_sha256"]:
            source = Path(parent_run) / name
            if sha256_file(source) != continuation["parent_artifact_sha256"][name]:
                raise ValueError("parent diagnostic input changed before continuation copy")
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if name.startswith("posterior_cache/") and sha256_file(target) != continuation["parent_artifact_sha256"][name]:
                raise ValueError("copied posterior cache differs from the bound parent")
    payload = dict(selected, binding=binding, trainer_sources=sources, continuation=continuation,
                   effective_maximum_epochs=continuation["total_epochs"])
    _save_checkpoint(payload, output / "best.pt")


def train(acquisition: Path, fields: Path, config_path: Path, output: Path, seed: int, resume=False,
          *, resume_from: Path | None = None, total_epochs: int | None = None,
          parent_checkpoint_sha256: str | None = None, stop_after_epochs: int | None = None):
    _execution_options(output, resume, resume_from, total_epochs, parent_checkpoint_sha256, stop_after_epochs)
    config = yaml.safe_load(config_path.read_text())
    dataset = json.loads((fields / "dataset.json").read_text())
    design = json.loads((acquisition / "design.json").read_text())
    if (dataset["status"] != "BUILT" or dataset["split"] != "train"
            or dataset["design_sha256"] != sha256_file(acquisition / "design.json")
            or dataset["completed_episodes"] != len(design["cases"])
            or seed not in config["training_seeds"]):
        raise ValueError("complete bound training data and registered seed required")
    fit, validation = partition(design, config)
    if config["bin_ms"] != dataset["settings"]["bin_ms"] or config["training_sequence"] != "complete_six_second_episode":
        raise ValueError("configured count exposure/sequence differs from data")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    first = NativeEpisodeCounts(acquisition / fit[0]["case_id"], fields / fit[0]["case_id"])
    unit_ids = tuple(map(int, first.unit_ids))
    if first.duration_ticks * first.dt_ms != 6000.:
        raise ValueError("complete six-second episode required")
    del first
    model = ControlledPoissonLDS(unit_ids, 1, config["latent_size"], config["encoder_size"],
                                config["bin_ms"] / 1000, initialization_seed=seed).to(device)
    initial_decay = config.get("initial_transition_diagonal", .8)
    if type(initial_decay) not in (int, float) or not np.isfinite(initial_decay) or not 0 < initial_decay < 1:
        raise ValueError("initial transition diagonal must lie strictly between zero and one")
    with torch.no_grad():
        model.transition.copy_(initial_decay * torch.eye(model.latent_size, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    binding = dict(config=config, seed=seed, dataset_sha256=sha256_file(fields / "dataset.json"),
                   acquisition_design_sha256=sha256_file(acquisition / "design.json"),
                   model_implementation_sha256=sha256_file(Path(__file__).with_name("poisson_state_space.py")))
    sources = _trainer_sources("poisson")
    checkpoint = selected = continuation = None
    effective_epochs = config["maximum_epochs"]
    if resume_from is not None:
        checkpoint, selected, continuation = _prepare_continuation(
            resume_from, total_epochs, binding, sources, "poisson", parent_checkpoint_sha256)
        effective_epochs = total_epochs
    elif resume:
        checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=True)
        if checkpoint["binding"] != binding:
            raise ValueError("resume training/data/model binding changed")
        if checkpoint.get("trainer_sources", sources) != sources:
            raise ValueError("resume trainer source changed")
        effective_epochs, continuation = _checkpoint_budget(checkpoint, config)
        previous = json.loads((output / "run.json").read_text())
        if (previous["binding"] != binding
                or previous.get("effective_maximum_epochs", config["maximum_epochs"]) != effective_epochs
                or previous.get("continuation") != continuation):
            raise ValueError("resume report/checkpoint binding or registered budget changed")
        if (previous.get("completed_epochs") == checkpoint["next_epoch"]
                and previous.get("latest_checkpoint_sha256") is not None
                and previous["latest_checkpoint_sha256"] != sha256_file(output / "latest.pt")):
            raise ValueError("committed latest checkpoint changed")
    elif output.exists():
        raise FileExistsError("existing training output requires explicit resume")
    next_epoch, best, stale, attempt, history = 0, float("inf"), 0, 0, []
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        next_epoch, best, stale, history = (checkpoint[k] for k in ("next_epoch", "best", "stale", "history"))
        attempt = checkpoint["attempt"] + 1
    if resume_from is not None:
        _seed_continuation_output(output, resume_from, selected, binding, sources, continuation)
    else:
        output.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        if resume_from is not None:
            _save_checkpoint(dict(checkpoint, binding=binding, trainer_sources=sources, continuation=continuation,
                                  effective_maximum_epochs=effective_epochs, attempt=attempt), output / "latest.pt")
    else:
        total = np.zeros(len(unit_ids), dtype=np.float64)
        seconds = 0.
        for case in fit:
            folder = fields / case["case_id"]
            report = json.loads((folder / "summary.json").read_text())
            if sha256_file(folder / "unit_rates.npz") != report["artifacts"]["unit_rates.npz"]:
                raise ValueError("rate initialization source changed")
            with np.load(folder / "unit_rates.npz", allow_pickle=False) as r:
                if not np.array_equal(r["unit_ids"], unit_ids):
                    raise ValueError("fitting unit order mismatch")
                total += r["spike_count"]
            seconds += (report["window_stop_tick"] - report["window_start_tick"]) * report["dt_ms"] / 1000
        with torch.no_grad():
            model.log_rate_bias.copy_(torch.as_tensor(np.log((total + config["initialization_pseudocount"]) / seconds), device=device))
        _save_checkpoint(dict(binding=binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
                             next_epoch=0, best=best, stale=stale, history=history, attempt=0,
                             trainer_sources=sources, effective_maximum_epochs=effective_epochs,
                             continuation=continuation), output / "latest.pt")
    best_epoch = min(range(len(history)), key=lambda i: history[i]["validation"]["loss"]) if history else None
    started = time.perf_counter()
    report = dict(status="RUNNING", binding=binding, training_seed=seed, resumed=resume or resume_from is not None, attempt=attempt,
                  fit_blocks=config["fit_blocks"], validation_blocks=config["validation_blocks"],
                  fitting_episodes=len(fit), validation_episodes=len(validation), units=len(unit_ids),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  torch_version=str(torch.__version__), formal_training_completed=False,
                  source_fidelity_qualified=False, generative_fidelity_qualified=False,
                  completed_epochs=next_epoch, best_epoch=best_epoch, stale_epochs=stale,
                  best_validation_negative_elbo=best if history else None, effective_maximum_epochs=effective_epochs,
                  continuation=continuation, trainer_sources=sources,
                  resume_from=str(resume_from.resolve()) if resume_from is not None else None)
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    terms_names = ["loss", "expected_observation_nll", "initial_kl", "transition_kl"]
    def inputs(case):
        data = NativeEpisodeCounts(acquisition / case["case_id"], fields / case["case_id"])
        if tuple(map(int, data.unit_ids)) != unit_ids:
            raise ValueError("training episode unit order changed")
        count, stimulus = data.window(0, data.duration_ticks)
        return (torch.as_tensor(count, device=device).unsqueeze(0),
                torch.as_tensor(stimulus, device=device).unsqueeze(0))
    try:
        path = output / "optimization.csv"
        exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["attempt", "epoch", "case_id", *terms_names, "gradient_norm", "seconds"])
            if not exists:
                writer.writeheader()
            end_epoch = min(effective_epochs, next_epoch + stop_after_epochs) if stop_after_epochs is not None else effective_epochs
            for epoch in range(next_epoch, end_epoch):
                if stale >= config["early_stopping_patience"]:
                    break
                order = fit.copy()
                random.Random(seed * 100000 + epoch).shuffle(order)
                training = []
                for case in order:
                    start = time.perf_counter()
                    y, u = inputs(case)
                    optimizer.zero_grad(set_to_none=True)
                    terms = model.sequence_loss(y, u)
                    terms["loss"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm_limit"], error_if_nonfinite=True)
                    optimizer.step()
                    numbers = {key: float(value.detach()) for key, value in terms.items()}
                    training.append(numbers)
                    writer.writerow(dict(attempt=attempt, epoch=epoch, case_id=case["case_id"], **numbers,
                                         gradient_norm=float(norm), seconds=time.perf_counter() - start))
                    f.flush()
                    del y, u, terms
                evaluated = []
                with torch.no_grad():
                    for case in validation:
                        y, u = inputs(case)
                        evaluated.append({key: float(value) for key, value in model.sequence_loss(y, u).items()})
                        del y, u
                row = dict(epoch=epoch, training={k: float(np.mean([r[k] for r in training])) for k in terms_names},
                           validation={k: float(np.mean([r[k] for r in evaluated])) for k in terms_names})
                history.append(row)
                value = row["validation"]["loss"]
                improved = value < best
                best, stale = (value, 0) if improved else (best, stale + 1)
                if improved:
                    best_epoch = epoch
                next_epoch = epoch + 1
                checkpoint = dict(binding=binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  next_epoch=next_epoch, best=best, stale=stale, history=history, attempt=attempt,
                                  best_epoch=best_epoch, trainer_sources=sources,
                                  effective_maximum_epochs=effective_epochs, continuation=continuation)
                _save_checkpoint(checkpoint, output / "latest.pt")
                if improved:
                    _save_checkpoint(checkpoint, output / "best.pt")
                report.update(completed_epochs=next_epoch, best_epoch=best_epoch, best_validation_negative_elbo=best,
                              stale_epochs=stale, elapsed_seconds=time.perf_counter() - started)
                (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
                (output / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps(row), flush=True)
        completed = next_epoch >= effective_epochs or stale >= config["early_stopping_patience"]
        report.update(status="FITTED_DEVELOPMENT" if completed else "CHECKPOINTED_DEVELOPMENT",
                      elapsed_seconds=time.perf_counter() - started,
                      stop_reason="validation_patience" if stale >= config["early_stopping_patience"] else
                                  "epoch_budget" if completed else "invocation_epoch_boundary",
                      completed_epochs=next_epoch, best_epoch=best_epoch, stale_epochs=stale,
                      best_validation_negative_elbo=best,
                      latest_checkpoint_sha256=sha256_file(output / "latest.pt"),
                      best_checkpoint_sha256=sha256_file(output / "best.pt"),
                      predictive_evaluation_completed=False)
        (output / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
    except Exception as error:
        report.update(status="FAILED", error=str(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.perf_counter() - started)
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--resume-from", type=Path)
    parser.add_argument("--total-epochs", type=int, help="prospective continuation ceiling; base config is unchanged")
    parser.add_argument("--parent-checkpoint-sha256", help="required prospective latest hash for legacy Poisson parents")
    parser.add_argument("--stop-after-epochs", type=int, help="checkpoint at an invocation boundary without changing its registered ceiling")
    a = parser.parse_args()
    print(json.dumps(train(a.acquisition, a.fields, a.config, a.output, a.seed, a.resume,
                           resume_from=a.resume_from, total_epochs=a.total_epochs,
                           parent_checkpoint_sha256=a.parent_checkpoint_sha256, stop_after_epochs=a.stop_after_epochs)))


if __name__ == "__main__":
    main()

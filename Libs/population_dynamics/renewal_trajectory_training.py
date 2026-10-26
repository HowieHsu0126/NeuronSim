"""Fit a controlled renewal or population model on complete native episodes.

The Mesocircuit protocol retains its complete 1,200-bin, 100,000-unit episodes
and fixed 56/16 development split. PFC and V1 use explicit whole-circuit source
cohorts. Validation uses fixed repeated variational draws; event-time MC ELBO does not establish forecast qualification
and must not be compared numerically with an unordered-count Poisson ELBO.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml

from .native_renewal_data import MINIMUM_INTERVAL_POLICY, NativeRenewalEpisode, load_dataset_index
from .poisson_trajectory_training import (
    _save_checkpoint, partition, _execution_options, _trainer_sources,
    _checkpoint_budget, _prepare_continuation, _seed_continuation_output, _prior_cache_artifacts,
)
from .provenance import sha256_file
from .population_prior_refit import SCHEMAS as PRIOR_REFIT_SCHEMAS, uses_mixture, uses_observation
from .renewal_state_space import ControlledRenewalLDS

SOURCE_COHORT_SCHEMAS = frozenset(("pfc_population_development_training_v1", "v1_population_development_training_v1"))

LOSS_TERMS = ("loss", "sampled_observation_nll", "initial_kl", "transition_kl")
FIXED_BUDGET = dict(
    schema="neural_population_renewal_trajectory_training_v1",
    training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
    latent_size=16, encoder_size=32, bin_ms=5.0, learning_rate=.001,
    maximum_epochs=20, early_stopping_patience=4, initialization_pseudocount=.5,
    gradient_norm_limit=100., initial_transition_diagonal=.98,
    training_sequence="complete_six_second_episode",
    observation_objective="native_event_time_mc_negative_elbo",
    minimum_interval_policy=MINIMUM_INTERVAL_POLICY,
    expected_poisson_rng_family="per_edge_philox4_32_10",
    validation_draws=4, validation_randomness="fixed_per_training_seed_case_and_draw",
    variational_rng="explicit_separate_checkpointed_generator", formal_training_completed=False,
)


POPULATION_FIXED_BUDGET = {key: value for key, value in FIXED_BUDGET.items()
                           if key not in ("latent_size", "initial_transition_diagonal")}
POPULATION_FIXED_BUDGET.update(
    schema="neural_population_state_trajectory_training_v1", model_family="population",
    latent_channels=4, encoder_size=16, representation="recurrent", unit_chunk_size=4096,
)


def validate_config(config: dict, seed: int, *, backend: str = "renewal") -> None:
    if backend == "population" and config.get("schema") in PRIOR_REFIT_SCHEMAS:
        from .population_prior_refit import validate_config as validate_prior
        validate_prior(config, seed, POPULATION_FIXED_BUDGET)
        return
    if backend == "population" and config.get("schema") == "pfc_population_development_training_v1":
        # Frozen Mesocircuit runtimes do not stage this optional source consumer.
        from .pfc_population_training_data import validate_config as validate_pfc
        validate_pfc(config, seed)
        return
    if backend == "population" and config.get("schema") == "v1_population_development_training_v1":
        from .v1_population_training_data import validate_config as validate_v1
        validate_v1(config, seed)
        return
    if backend not in ("renewal", "population"):
        raise ValueError("registered renewal or population training backend required")
    frozen = FIXED_BUDGET if backend == "renewal" else POPULATION_FIXED_BUDGET
    if config.get("model_family", "renewal") != backend:
        raise ValueError("configuration model family differs from the selected training backend")
    if (backend == "population" and config.get("representation") not in ("fixed", "recurrent")):
        raise ValueError("registered fixed or recurrent population representation required")
    if any(config.get(key) != value for key, value in frozen.items()
           if not (backend == "population" and key == "representation")):
        raise ValueError("configuration differs from the frozen complete-episode training protocol")
    if seed not in config["training_seeds"] or type(seed) is not int:
        raise ValueError("registered training seed 41, 42 or 43 required")
    if type(config["cpu_threads"]) is not int or not 1 <= config["cpu_threads"] <= 2:
        raise ValueError("one or two CPU threads required")
    for key in ("training_variational_seed_base", "validation_seed_base"):
        if type(config[key]) is not int or not 0 <= config[key] < 2**63:
            raise ValueError("explicit nonnegative 63-bit variational seed bases required")
    if torch.device(config["device"]).type not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA model placement required")


class TrainingCohort:
    """Validate the complete cohort; retain metadata rather than dense windows."""

    def __init__(self, acquisition: Path, fields: Path, config: dict):
        self.acquisition, self.fields = acquisition, fields
        design = json.loads((acquisition / "design.json").read_text())
        dataset = json.loads((fields / "dataset.json").read_text())
        ledger = json.loads((acquisition / "acquisition.json").read_text())
        self.fit, self.validation = partition(design, config)
        cases = self.fit + self.validation
        if (len(cases) != 72 or len(self.fit) != 56 or len(self.validation) != 16
                or len({c["case_id"] for c in cases}) != 72
                or any({c["template_id"] for c in cases if c["block"] == b}
                       != {f"familiar_{i}" for i in range(4)} for b in range(18))):
            raise ValueError("all 72 unique familiar episodes in the complete 18-block cohort required")
        if (dataset["status"] != "BUILT" or dataset["split"] != "train"
                or dataset["completed_episodes"] != 72 or dataset["independent_blocks"] != 18
                or dataset["cortical_units"] != [100000]
                or dataset["control_delivery_violations"] != 0 or dataset["refractory_violations"] != 0
                or dataset["design_sha256"] != sha256_file(acquisition / "design.json")
                or dataset["episode_table_sha256"] != sha256_file(fields / "episodes.csv")
                or dataset["settings"]["bin_ms"] != 5.0 or dataset["settings"]["start_ms"] != 0.0
                or dataset["settings"]["cells_per_axis"] != 4):
            raise ValueError("complete bound 100000-unit training field dataset required")
        runs = ledger["runs"]
        if (ledger["status"] != "ACQUIRED" or ledger["completed_episodes"] != 72
                or len(runs) != 72 or {r["case_id"] for r in runs} != {c["case_id"] for c in cases}
                or any(r["exit_code"] != 0 for r in runs)):
            raise ValueError("complete successful unique native acquisition ledger required")
        self.records = load_dataset_index(fields)
        rows = list(self.records.values())
        if len(rows) != 72 or set(self.records) != {c["case_id"] for c in cases}:
            raise ValueError("complete unique bound episode table required")
        native_digest = design.get("native_library_sha256")
        if not native_digest or design["config"]["execution"].get("expected_poisson_rng_family") != config["expected_poisson_rng_family"]:
            raise ValueError("prospective native library and Philox family binding required")
        for case in cases:
            self._verify_metadata(case)
            source = json.loads((acquisition / case["case_id"] / "report.json").read_text())
            if (source["status"] != "EXECUTED"
                    or source["native_library_sha256"] != native_digest
                    or source.get("native_poisson_rng_family") != config["expected_poisson_rng_family"]
                    or source["graph_manifest_sha256"] != design["graph_manifest_sha256"]
                    or source["config"] != case["settings"]
                    or source["duration_ms"] != 6000.0
                    or source["dt_ms"] != .1):
                raise ValueError("native episode source, settings or full-duration clock changed")
        first = NativeRenewalEpisode(acquisition / cases[0]["case_id"], fields / cases[0]["case_id"])
        self.unit_ids = first.unit_ids.copy()
        self.unit_field_index, self.field_ids = first.unit_field_index.copy(), first.field_ids.copy()
        self.minimum_interval_ticks = first.minimum_interval_ticks.copy()
        self.source_identity = first.source_identity
        self.dt_ms, self.bin_ms, self.duration_ticks = first.dt_ms, first.bin_ms, first.duration_ticks
        self._check_episode(first)
        self.binding = dict(
            acquisition_design_sha256=sha256_file(acquisition / "design.json"),
            acquisition_ledger_sha256=sha256_file(acquisition / "acquisition.json"),
            dataset_sha256=sha256_file(fields / "dataset.json"),
            episode_table_sha256=dataset["episode_table_sha256"],
            native_source_identity=self.source_identity,
            episode_bindings={r["case_id"]: {k: r[k] for k in ("raw_report_sha256", "field_summary_sha256")}
                              for r in rows},
        )

    def _verify_metadata(self, case: dict) -> None:
        row = self.records[case["case_id"]]
        if (row["split"] != "train" or int(row["block"]) != case["block"]
                or row["template_id"] != case["template_id"] or row["family"] != "trajectory"
                or int(row["cortical_units"]) != 100000
                or row["raw_report_sha256"] != sha256_file(self.acquisition / case["case_id"] / "report.json")
                or row["field_summary_sha256"] != sha256_file(self.fields / case["case_id"] / "summary.json")):
            raise ValueError("bound episode report, field summary or identity changed")

    def _check_episode(self, data: NativeRenewalEpisode) -> None:
        if (len(data.unit_ids) != 100000 or data.bin_ms != 5.0 or data.dt_ms != .1
                or data.duration_ticks * data.dt_ms != 6000.0
                or data.duration_ticks // data.bin_ticks != 1200
                or not np.array_equal(data.minimum_interval_ticks, self.minimum_interval_ticks)
                or not np.array_equal(data.field_ids, self.field_ids)):
            raise ValueError("complete 1200-by-100000 episode with common source support required")

    def window(self, case: dict):
        self._verify_metadata(case)
        data = NativeRenewalEpisode(self.acquisition / case["case_id"], self.fields / case["case_id"],
                                    expected_source=self.source_identity, expected_unit_ids=self.unit_ids,
                                    expected_unit_field_index=self.unit_field_index)
        self._check_episode(data)
        result = data.renewal_window(0, data.duration_ticks)
        if result.counts.shape != (1200, 100000) or result.eligible_ticks.shape != result.counts.shape:
            raise ValueError("complete episode counts and native eligible exposure required")
        return result


def model_device(model, backend: str):
    """Select the concrete model device without coercing an unknown family."""
    if backend == "renewal":
        return model.latent_model.loading.device
    if backend == "population":
        return model.posterior_mean.weight.device
    raise ValueError("registered renewal or population training backend required")


def hazard_bias(model, backend: str):
    if backend == "renewal":
        return model.latent_model.log_rate_bias
    if backend == "population":
        return model.log_hazard_bias
    raise ValueError("registered renewal or population training backend required")


def model_configuration(cohort, config: dict, seed: int, backend: str) -> dict:
    if backend == "population" and config.get("schema") in SOURCE_COHORT_SCHEMAS:
        return cohort.population_configuration(config, seed)
    common = dict(
        unit_ids=list(map(int, cohort.unit_ids)), input_size=1,
        encoder_size=config["encoder_size"], bin_seconds=config["bin_ms"] / 1000,
        initialization_seed=seed, tick_seconds=cohort.dt_ms / 1000,
        minimum_interval_ticks=cohort.minimum_interval_ticks.tolist(),
    )
    if backend == "renewal":
        return dict(common, latent_size=config["latent_size"])
    if backend == "population":
        return dict(common, unit_field_index=list(map(int, cohort.unit_field_index)),
                    field_names=list(map(str, cohort.field_ids)),
                    latent_channels=config["latent_channels"], representation=config["representation"],
                    unit_chunk_size=config["unit_chunk_size"])
    raise ValueError("registered renewal or population training backend required")


def variational_seed(base: int, seed: int, domain: str, case_id: str = "", draw: int = 0) -> int:
    identity = json.dumps([base, seed, domain, case_id, draw], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)


def registered_variational_seed(config, seed, domain):
    """Rotate the three registered seeds across independent training roles."""
    if config.get("random_seed_policy")!="exact_registered_stream_seeds":
        return variational_seed(config["training_variational_seed_base"],seed,domain)
    seeds=(41,42,43)
    if seed not in seeds or domain not in ("training","validation"):
        raise ValueError("REGISTERED_VARIATIONAL_SEED_ROLE_REQUIRED")
    offset=1 if domain=="training" else 2
    return seeds[(seeds.index(seed)+offset)%3]


def _inputs(window, device):
    return tuple(torch.as_tensor(v, device=device).unsqueeze(0)
                 for v in (window.counts, window.stimulus, window.eligible_ticks))


def _numbers(terms: dict, names=LOSS_TERMS) -> dict[str, float]:
    return {key: float(terms[key].detach()) for key in names}


@torch.no_grad()
def evaluate_validation(model, cohort, config, seed, *, on_draw=None):
    """Fixed validation randomness; never receives the training RNG.

    Legacy protocols retain per-case seeds. The registered three-seed protocol
    resets one validation stream before the complete ordered validation sequence.
    """
    model.eval()
    device = model_device(model, config.get("model_family", "renewal"))
    rows = []
    registered=(config.get("random_seed_policy")=="exact_registered_stream_seeds")
    sequence_generator=(torch.Generator(device=device).manual_seed(
        registered_variational_seed(config,seed,"validation")) if registered else None)
    for case in cohort.validation:
        inputs = _inputs(cohort.window(case), device)
        case_model = (cohort.model_for_case(model, case)
                      if config.get("schema") in SOURCE_COHORT_SCHEMAS else model)
        case_model.eval()
        for draw in range(config["validation_draws"]):
            draw_seed=(registered_variational_seed(config,seed,"validation") if registered else
                       variational_seed(config["validation_seed_base"],seed,"validation",case["case_id"],draw))
            generator=(sequence_generator if registered else
                       torch.Generator(device=device).manual_seed(draw_seed))
            started = time.perf_counter()
            values = _numbers(case_model.sequence_loss(*inputs, generator))
            rows.append(values)
            if on_draw is not None:
                on_draw(dict(case_id=case["case_id"], draw=draw, variational_seed=draw_seed,
                             **values, seconds=time.perf_counter() - started))
        del inputs
    return {key: float(np.mean([r[key] for r in rows])) for key in LOSS_TERMS}


def initialize_observation(model, cohort, pseudocount: float, *, backend: str = "renewal") -> dict:
    if backend == "population" and model.metadata is not None:
        return cohort.initialize_observation(model, pseudocount)
    successes = np.zeros(len(cohort.unit_ids), dtype=np.float64)
    eligible = np.zeros_like(successes)
    for case in cohort.fit:
        window = cohort.window(case)
        successes += window.counts.sum(axis=0, dtype=np.float64)
        eligible += window.eligible_ticks.sum(axis=0, dtype=np.float64)
        del window
    probability = (successes + pseudocount) / (eligible + 2 * pseudocount)
    if np.any(probability <= 0) or np.any(probability >= 1):
        raise ValueError("finite strictly interior empirical eligible-tick probability required")
    log_hazard = np.log(-np.log1p(-probability) / model.tick_seconds)
    with torch.no_grad():
        hazard_bias(model, backend).copy_(torch.as_tensor(log_hazard, device=model_device(model, backend)))
    return dict(method="fit_only_eligible_tick_hazard", fitting_episodes=len(cohort.fit),
                pseudocount=pseudocount, total_observed_spikes=float(successes.sum()),
                total_eligible_ticks=float(eligible.sum()))


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hardware(device):
    result = dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                  cuda_runtime=torch.version.cuda, device=str(device), host=platform.node(),
                  cpu=platform.processor(), slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_memory_bytes=props.total_memory,
                      cudnn_version=torch.backends.cudnn.version())
    return result


def train(acquisition: Path, fields: Path, config_path: Path, output: Path, seed: int,
          resume: bool = False, *, stop_after_epochs: int | None = None,
          resume_from: Path | None = None, total_epochs: int | None = None,
          parent_checkpoint_sha256: str | None = None, backend: str = "renewal",
          prior_parent: Path | None = None, prior_cache_source: Path | None = None):
    """Resume only committed epochs; partial-attempt logs remain diagnostic."""
    _execution_options(output, resume, resume_from, total_epochs, parent_checkpoint_sha256, stop_after_epochs)
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed, backend=backend)
    prior_refit = backend == "population" and config.get("schema") in PRIOR_REFIT_SCHEMAS
    if prior_parent is not None and (not prior_refit or resume or resume_from is not None):
        raise ValueError("prior parent initializes only a new registered prior-refit objective")
    if prior_refit and not resume and resume_from is None and prior_parent is None:
        raise ValueError("prior refit requires parent initialization, same-objective resume or explicit budget continuation")
    mixture_refit = prior_refit and uses_mixture(config)
    observation_refit = prior_refit and uses_observation(config)
    if prior_cache_source is not None and (not prior_refit or resume or resume_from is not None):
        raise ValueError("posterior cache source initializes only a new prior refit")
    prior_module = None
    names = LOSS_TERMS
    if prior_refit:
        from . import population_prior_refit as prior_module
        from .population_prior_objective import configure_prior_parameters
        names = prior_module.loss_terms(config)
    pfc = backend == "population" and config.get("schema") == "pfc_population_development_training_v1"
    v1 = backend == "population" and config.get("schema") == "v1_population_development_training_v1"
    source_cohort = pfc or v1
    if pfc:
        from .pfc_population_training_data import PfcPopulationTrainingCohort
        cohort = PfcPopulationTrainingCohort(acquisition, config)
    elif v1:
        from .v1_population_training_data import V1PopulationTrainingCohort
        cohort = V1PopulationTrainingCohort(acquisition, config)
    else:
        cohort = TrainingCohort(acquisition, fields, config)
    implementation_files = dict(model="renewal_state_space.py", observation="refractory_observation.py",
                                loader="native_renewal_data.py", trainer=Path(__file__).name,
                                latent_model="poisson_state_space.py", native_counts="native_count_windows.py",
                                training_utilities="poisson_trajectory_training.py")
    if backend == "population":
        implementation_files.update(model="population_state.py", wrapper="population_trajectory_training.py")
    if v1:
        implementation_files.update(loader="v1_renewal_data.py", cohort="v1_population_training_data.py",
                                    source_cohort="source_population_training.py")
    binding = dict(config=config, seed=seed, **cohort.binding,
                   **{key + "_implementation_sha256": sha256_file(Path(__file__).with_name(name))
                      for key, name in implementation_files.items()})
    sources = _trainer_sources(backend)
    if prior_refit:
        for name in ("population_prior_refit.py", "population_prior_objective.py"):
            sources[name] = sha256_file(Path(__file__).with_name(name))
        if observation_refit:
            sources["population_observation_prior.py"] = sha256_file(Path(__file__).with_name("population_observation_prior.py"))
        if mixture_refit:
            sources["population_predictive_mixture.py"] = sha256_file(Path(__file__).with_name("population_predictive_mixture.py"))
    if pfc:
        for name in ("pfc_population_training_data.py", "pfc_renewal_data.py", "circuit_data.py",
                     "pfc_scale_analysis.py", "pfc_source_dataset.py", "pfc_kernel_parameters.py",
                     "pfc_recurrent_gain_comparison.py", "benchmark_tasks.py"):
            sources[name] = sha256_file(Path(__file__).with_name(name))
        sources["neural_population_pfc_recurrent_gain.yaml"] = sha256_file(
            Path(__file__).resolve().parents[1] / "configs/neural_population_pfc_recurrent_gain.yaml")
    if source_cohort:
        sources["source_population_training.py"] = sha256_file(Path(__file__).with_name("source_population_training.py"))
    if v1:
        for name in ("v1_population_training_data.py", "v1_renewal_data.py", "circuit_data.py",
                     "v1_nest_time_semantics.py", "v1_visual_controls.py"):
            sources[name] = sha256_file(Path(__file__).with_name(name))
    checkpoint = selected_parent = continuation = None
    prior_cache_cases = (tuple(case["case_id"] for case in cohort.fit + cohort.validation)
                         if prior_refit else None)
    effective_epochs = config["maximum_epochs"]
    if resume_from is not None:
        checkpoint, selected_parent, continuation = _prepare_continuation(
            resume_from, total_epochs, binding, sources, backend, parent_checkpoint_sha256,
            prior_cache_cases=prior_cache_cases)
        effective_epochs = total_epochs
    elif resume:
        checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if checkpoint["binding"] != binding:
            raise ValueError("resume config/data/native source/implementation binding changed")
        if checkpoint.get("trainer_sources", None if backend == "population" else sources) != sources:
            raise ValueError("resume trainer source changed")
        effective_epochs, continuation = _checkpoint_budget(checkpoint, config)
        if prior_refit and continuation is not None:
            expected_cache = {name: digest for name, digest in continuation["parent_artifact_sha256"].items()
                              if name.startswith("posterior_cache/")}
            if _prior_cache_artifacts(output, prior_cache_cases) != expected_cache:
                raise ValueError("continued prior posterior cache changed")
    elif output.exists():
        raise FileExistsError("existing training output requires explicit resume")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    constructor = model_configuration(cohort, config, seed, backend)
    if checkpoint is not None and checkpoint["model_configuration"] != constructor:
        raise ValueError("resume unit identity, source field map, native interval or constructor changed")
    if backend == "renewal":
        model_class = ControlledRenewalLDS
    else:
        # Existing renewal runtimes do not stage the optional population backend.
        from .population_state import ControlledPopulationState
        model_class = ControlledPopulationState
    model = model_class(**constructor).to(config["device"])
    device = model_device(model, backend)
    parameters = configure_prior_parameters(model) if prior_refit else tuple(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    generator = torch.Generator(device=device).manual_seed(
        registered_variational_seed(config,seed,"training"))
    prior_generator = (torch.Generator(device=device).manual_seed(variational_seed(
        config["training_variational_seed_base"], seed, "prior_training_dynamics")) if prior_refit else None)
    target_generator = (torch.Generator(device=device).manual_seed(variational_seed(
        config["training_variational_seed_base"], seed, "prior_training_target")) if mixture_refit else None)
    cache_reuse = None
    next_epoch, best, stale, history, best_epoch = 0, float("inf"), 0, [], None
    previous = json.loads((output / "run.json").read_text()) if resume else None
    if previous:
        if (previous["binding"] != binding
                or (backend == "population" and previous.get("trainer_sources") != sources)
                or previous.get("effective_maximum_epochs", config["maximum_epochs"]) != effective_epochs
                or previous.get("continuation") != continuation):
            raise ValueError("resume report/checkpoint binding or registered budget changed")
        # A crash may commit latest before updating the report. Compare hashes
        # only when report and checkpoint refer to the same committed state.
        if (previous["completed_epochs"] == checkpoint["next_epoch"]
                and previous.get("latest_checkpoint_sha256") is not None
                and previous["latest_checkpoint_sha256"] != sha256_file(output / "latest.pt")):
            raise ValueError("committed latest checkpoint changed")
        if (previous.get("best_epoch") == checkpoint["best_epoch"]
                and previous.get("best_checkpoint_sha256") is not None
                and previous["best_checkpoint_sha256"] != sha256_file(output / "best.pt")):
            raise ValueError("committed best checkpoint changed")
    attempt = (checkpoint["attempt"] + 1 if resume_from is not None
               else previous["attempt"] + 1 if previous else 0)
    initialization = None
    if checkpoint is not None:
        if source_cohort:
            for name, expected in model.named_buffers():
                if name not in checkpoint["model"] or not torch.equal(checkpoint["model"][name].to(expected.device), expected):
                    raise ValueError("PFC training checkpoint source buffers changed: " + name)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["variational_rng"].cpu())
        if prior_refit:
            prior_generator.set_state(checkpoint["prior_dynamics_rng"].cpu())
            cache_reuse = checkpoint.get("posterior_cache_reuse")
        if mixture_refit:
            if not torch.is_tensor(checkpoint.get("prior_target_rng")):
                raise ValueError("committed predictive-mixture target RNG state required")
            target_generator.set_state(checkpoint["prior_target_rng"].cpu())
        next_epoch, best, stale, history, best_epoch, initialization = (
            checkpoint[key] for key in ("next_epoch", "best", "stale", "history", "best_epoch", "initialization"))
    if resume_from is not None:
        _seed_continuation_output(output, resume_from, selected_parent, binding, sources, continuation)
    output.mkdir(parents=True, exist_ok=True)
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    if previous and previous["status"] == "RUNNING":
        previous.update(status="INTERRUPTED_WITHOUT_FINAL_REPORT", recovery="rollback_to_committed_epoch")
        _write_json(attempts / f"attempt_{previous['attempt']:03d}.json", previous)
    report = dict(status="RUNNING", binding=binding, model_configuration=constructor,
                  training_seed=seed, attempt=attempt, resumed=resume or resume_from is not None, hardware=_hardware(device),
                  command=list(sys.argv), working_directory=str(Path.cwd()),
                  invocation=dict(acquisition=str(acquisition.resolve()),
                                  fields=str(fields.resolve()) if fields is not None else None,
                                  config=str(config_path.resolve()), output=str(output.resolve()),
                                  seed=seed, resume=resume, stop_after_epochs=stop_after_epochs,
                                  resume_from=str(resume_from.resolve()) if resume_from is not None else None,
                                  total_epochs=total_epochs, parent_checkpoint_sha256=parent_checkpoint_sha256),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  fitting_episodes=len(cohort.fit), validation_episodes=len(cohort.validation),
                  units=len(cohort.unit_ids), parameter_count=sum(p.numel() for p in model.parameters()),
                  observation_likelihood_space="ordered_native_event_times",
                  complete_episode_updates=True, validation_draws=config["validation_draws"],
                  formal_training_completed=False, source_fidelity_qualified=False,
                  generative_fidelity_qualified=False, predictive_evaluation_completed=False,
                  completed_epochs=next_epoch, best_epoch=best_epoch,
                  effective_maximum_epochs=effective_epochs, continuation=continuation, trainer_sources=sources,
                  partial_epoch_updates_are_diagnostic=True, scientific_claim_qualified=False,
                  calibration_outcomes_accessed=False, test_outcomes_accessed=False)
    if backend == "population":
        report.update(model_family="population", representation=config["representation"])
    if prior_refit:
        report.update(training_objective=config["observation_objective"],
            selection_metric=("validation_native_event_path_mixture_nll" if observation_refit else
                              "validation_joint_predictive_mixture_kl" if mixture_refit else "validation_multistep_prior_kl"),
            complete_episode_updates=False,
            complete_source_episode_encoding=True, posterior_targets_fixed=True,
            refit_initialization=initialization, fitting_origin_windows=len(cohort.fit)*len(config["prior_origins_ms"]),
            validation_origin_windows=len(cohort.validation)*len(config["prior_origins_ms"]),
            observation_parameters_refitted=False, prior_parent=str(prior_parent) if prior_parent else None,
            observation_likelihood_evaluated_during_refit=observation_refit)
    if observation_refit:
        report.update(prediction_particles=config["prediction_particles"],
            mixture_scope="joint_all_source_units_and_time_prefix",
            observation_score_normalization="nats_per_identified_unit_and_bin",
            posterior_cache_role="causal_boundary_only", posterior_targets_used=False,
            future_exposure_role="observed_event_path_scoring_label_only",
            future_exposure_used_in_prior_dynamics=False, marginal_likelihood_exact=False)
    if cache_reuse is not None:
        report["posterior_cache_reuse"] = cache_reuse
    if mixture_refit:
        report.update(prediction_particles=config["prediction_particles"],
                      posterior_target_draws=config["posterior_target_draws"],
                      mixture_scope="joint_all_source_fields_and_latent_channels")
    if source_cohort:
        report.update(input_family=config["input_family"], partition_unit="whole_source_circuit",
            fitting_circuits=len(config["fit_circuit_ids"]), validation_circuits=len(config["validation_circuit_ids"]),
            unit_count_scope="canonical_fitting_circuit; each episode retains its complete source map")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    def write_report():
        report.update(elapsed_seconds=time.perf_counter() - started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        _write_json(attempts / f"attempt_{attempt:03d}.json", report)
        _write_json(output / "run.json", report)
    def payload(epoch):
        return dict(binding=binding, model_configuration=constructor,
                    model=model.state_dict(), optimizer=optimizer.state_dict(),
                    variational_rng=generator.get_state(), next_epoch=epoch, history=history,
                    best=best, stale=stale, best_epoch=best_epoch, attempt=attempt, initialization=initialization,
                    trainer_sources=sources, effective_maximum_epochs=effective_epochs, continuation=continuation,
                    **({"prior_dynamics_rng": prior_generator.get_state()} if prior_refit else {}),
                    **({"prior_target_rng": target_generator.get_state()} if mixture_refit else {}),
                    **({"posterior_cache_reuse": cache_reuse} if cache_reuse is not None else {}))
    write_report()
    try:
        if checkpoint is None:
            report["phase"] = "fit_only_hazard_initialization"
            if backend == "renewal":
                with torch.no_grad():
                    model.latent_model.transition.copy_(config["initial_transition_diagonal"] *
                                                         torch.eye(config["latent_size"], device=device))
            # Population decay is an intrinsic constructor constant. Only the
            # empirical observation hazard is initialized from fitting episodes.
            if prior_refit:
                initialization = prior_module.initialize_from_parent(
                    model, cohort, config, seed, prior_parent, constructor)
                report.update(phase="selected_parent_prior_refit", refit_initialization=initialization)
                if prior_cache_source is not None:
                    cache_reuse = prior_module.reuse_posterior_cache(
                        prior_cache_source, output, initialization, seed, prior_cache_cases)
                    report["posterior_cache_reuse"] = cache_reuse
            else:
                initialization = initialize_observation(
                    model, cohort, config["initialization_pseudocount"], backend=backend)
            _save_checkpoint(payload(0), output / "latest.pt")
        elif resume_from is not None:
            _save_checkpoint(payload(next_epoch), output / "latest.pt")
        elif best_epoch is not None:
            # latest is committed before best. Recover an interrupted best copy
            # only when that exact selected epoch is the committed latest state.
            best_path = output / "best.pt"
            selected = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
            if selected is None or selected["next_epoch"] != best_epoch + 1:
                if best_epoch != next_epoch - 1:
                    raise ValueError("selected best checkpoint missing from committed history")
                _save_checkpoint(checkpoint, best_path)
            elif selected["binding"] != binding:
                raise ValueError("selected best checkpoint binding changed")
        prior_data = (prior_module.PosteriorCache(model, cohort, config, output, initialization,
                                                  _save_checkpoint) if prior_refit else None)
        frozen_parameters = ({name: value.detach().cpu().clone() for name, value in model.named_parameters()
                              if not value.requires_grad} if prior_refit else {})
        end_epoch = effective_epochs
        if stop_after_epochs is not None:
            end_epoch = min(end_epoch, next_epoch + stop_after_epochs)
        with (output / "optimization.csv").open("a", newline="") as fit_log, (output / "validation.csv").open("a", newline="") as val_log:
            fit_writer = csv.DictWriter(fit_log, fieldnames=["attempt", "epoch", "case_id", *names, "gradient_norm", "seconds"])
            val_writer = csv.DictWriter(val_log, fieldnames=["attempt", "epoch", "case_id", "draw", "variational_seed",
                *(["prior_dynamics_seed"] if prior_refit else []),
                *(["prior_target_seed"] if mixture_refit else []), *names, "seconds"])
            if fit_log.tell() == 0:
                fit_writer.writeheader()
            if val_log.tell() == 0:
                val_writer.writeheader()
            for epoch in range(next_epoch, end_epoch):
                if stale >= config["early_stopping_patience"]:
                    break
                model.train()
                order = cohort.fit.copy()
                random.Random(seed * 100000 + epoch).shuffle(order)
                trained = []
                for case in order:
                    report.update(phase="optimization", active_epoch=epoch, active_case_id=case["case_id"])
                    update_start = time.perf_counter()
                    inputs = prior_data.inputs(case) if prior_refit else _inputs(cohort.window(case), device)
                    case_model = cohort.model_for_case(model, case) if source_cohort else model
                    case_model.train()
                    optimizer.zero_grad(set_to_none=True)
                    terms = (prior_data.loss(inputs, generator, prior_generator, target_generator) if prior_refit
                             else case_model.sequence_loss(*inputs, generator))
                    terms["loss"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_norm_limit"], error_if_nonfinite=True)
                    optimizer.step()
                    numbers = _numbers(terms, names)
                    trained.append(numbers)
                    fit_writer.writerow(dict(attempt=attempt, epoch=epoch, case_id=case["case_id"],
                                             **numbers, gradient_norm=float(norm), seconds=time.perf_counter() - update_start))
                    fit_log.flush()
                    del inputs, terms, case_model
                report.update(phase="validation", active_epoch=epoch, active_case_id=None)
                def record_draw(row):
                    report["active_case_id"] = row["case_id"]
                    val_writer.writerow(dict(attempt=attempt, epoch=epoch, **row))
                    val_log.flush()
                validation = (prior_data.validation(seed, variational_seed, on_draw=record_draw) if prior_refit
                              else evaluate_validation(model, cohort, config, seed, on_draw=record_draw))
                row = dict(epoch=epoch, training={k: float(np.mean([r[k] for r in trained])) for k in names},
                           validation=validation)
                history.append(row)
                improved = validation["loss"] < best
                best, stale = (validation["loss"], 0) if improved else (best, stale + 1)
                if improved:
                    best_epoch = epoch
                next_epoch = epoch + 1
                report.update(phase="epoch_checkpoint", active_case_id=None)
                committed = payload(next_epoch)
                _save_checkpoint(committed, output / "latest.pt")
                if improved:
                    _save_checkpoint(committed, output / "best.pt")
                report.update(completed_epochs=next_epoch, best_epoch=best_epoch,
                              **({"best_validation_prior_loss": best} if prior_refit else
                                 {"best_validation_negative_elbo": best}), stale_epochs=stale,
                              latest_checkpoint_sha256=sha256_file(output / "latest.pt"),
                              best_checkpoint_sha256=sha256_file(output / "best.pt"))
                _write_json(output / "epochs.json", history)
                write_report()
                print(json.dumps(row, allow_nan=False), flush=True)
        if prior_refit:
            if any(not torch.equal(value.detach().cpu(), frozen_parameters[name])
                   for name, value in model.named_parameters() if name in frozen_parameters):
                raise ValueError("encoder, posterior or observation parameters changed during prior refit")
            report["frozen_parameters_unchanged_in_attempt"] = True
        completed = next_epoch >= effective_epochs or stale >= config["early_stopping_patience"]
        report.update(status="FITTED_DEVELOPMENT" if completed else "CHECKPOINTED_DEVELOPMENT", phase="stopped",
                      stop_reason="validation_patience" if stale >= config["early_stopping_patience"] else
                                  "epoch_budget" if completed else "invocation_epoch_boundary",
                      completed_epochs=next_epoch, best_epoch=best_epoch, stale_epochs=stale,
                      **({"best_validation_prior_loss": best} if prior_refit else
                         {"best_validation_negative_elbo": best}),
                      latest_checkpoint_sha256=sha256_file(output / "latest.pt"),
                      best_checkpoint_sha256=sha256_file(output / "best.pt"))
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                      error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc(),
                      recovery="resume_from_latest_committed_epoch_if_present")
        write_report()
        raise
    _write_json(output / "epochs.json", history)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_report()
    return report


def main(*, backend: str = "renewal"):
    parser = argparse.ArgumentParser(description=f"Fit the {backend} model on complete native episodes.")
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[41, 42, 43], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--resume-from", type=Path)
    mode.add_argument("--prior-parent", type=Path, help="selected training directory for a new prior-refit objective")
    parser.add_argument("--prior-cache-source", type=Path)
    parser.add_argument("--total-epochs", type=int, help="prospective continuation ceiling; base config is unchanged")
    parser.add_argument("--parent-checkpoint-sha256", help="optional additional parent latest checksum; existing receipt must agree")
    parser.add_argument("--stop-after-epochs", type=int,
                        help="checkpoint after this many additional epochs; does not change the registered total ceiling")
    args = parser.parse_args()
    if args.fields is None:
        selected_config = yaml.safe_load(args.config.read_text())
        if not (backend == "population"
                and selected_config.get("schema") in SOURCE_COHORT_SCHEMAS):
            parser.error("--fields is required for the native Mesocircuit cohort")
    options = dict(stop_after_epochs=args.stop_after_epochs, resume_from=args.resume_from,
                   total_epochs=args.total_epochs, parent_checkpoint_sha256=args.parent_checkpoint_sha256)
    if args.prior_parent is not None:
        options["prior_parent"] = args.prior_parent
    if args.prior_cache_source is not None:
        options["prior_cache_source"] = args.prior_cache_source
    if backend != "renewal":
        options["backend"] = backend
    print(json.dumps(train(args.acquisition, args.fields, args.config, args.output, args.seed,
                           args.resume, **options), allow_nan=False))


if __name__ == "__main__":
    main()

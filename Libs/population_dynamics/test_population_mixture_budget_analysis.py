"""Consumer coverage for the registered predictive-mixture 80-to-320 budget."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from . import population_forecast_analysis as analysis
from .poisson_trajectory_training import _trainer_sources
from .provenance import sha256_file
from .test_population_forecast_analysis import (
    _continued_prior_cost_fixture,
    _json,
    _refresh_prior_cell,
    _set_prior_fixture_profile,
)


# Accepted run_626306 receipts. Keep these independent of the admission table.
_ACCEPTED_MIXTURE_TRAINERS = {
    "poisson_trajectory_training.py": "fe3204e92e8ad7b06379f3c45fa9a615f1f9294e71c77978e5e5324e9a98075f",
    "renewal_trajectory_training.py": "b00d0fb7e4107a4fdaf7b488a34c2acf001d915fd39eef763cb0cde0779f0323",
}


def _budget_profile(parent, child, *, mixture, historical=False):
    """Bind tiny fixture tensors to the real budgets and trainer migration."""
    parent_run, parent_checkpoint = _set_prior_fixture_profile(parent, mixture=mixture)
    parent_run["binding"]["config"]["maximum_epochs"] = 80
    sources = _trainer_sources("population")
    names = ["population_prior_refit.py", "population_prior_objective.py"]
    if mixture:
        names.append("population_predictive_mixture.py")
    sources.update({name: sha256_file(Path(analysis.__file__).with_name(name)) for name in names})
    binding = copy.deepcopy(parent_run["binding"])
    binding.update(trainer_implementation_sha256=sources["renewal_trajectory_training.py"],
                   training_utilities_implementation_sha256=sources["poisson_trajectory_training.py"])
    parent_sources = dict(sources, **(_ACCEPTED_MIXTURE_TRAINERS if historical else {}))
    parent_binding = dict(copy.deepcopy(binding),
        trainer_implementation_sha256=parent_sources["renewal_trajectory_training.py"],
        training_utilities_implementation_sha256=parent_sources["poisson_trajectory_training.py"])
    parent_run.update(binding=parent_binding, trainer_sources=parent_sources,
                      completed_epochs=80, best_epoch=79, effective_maximum_epochs=80)
    parent_checkpoint.update(next_epoch=80, trainer_sources=parent_sources, effective_maximum_epochs=80)
    _refresh_prior_cell(parent, parent_run, parent_checkpoint)

    run, checkpoint = _set_prior_fixture_profile(child, mixture=mixture)
    continuation = dict(copy.deepcopy(run["continuation"]),
        base_maximum_epochs=80, total_epochs=320, parent_effective_maximum_epochs=80,
        parent_completed_epochs=80, inherited_early_stopping_patience=12,
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"],
        parent_binding=copy.deepcopy(parent_binding),
        parent_artifact_sha256={"run.json": sha256_file(parent / "training/run.json"),
                               "best.pt": parent_run["best_checkpoint_sha256"]},
        parent_source_hashes=parent_sources, current_source_hashes=sources,
        source_migration={name: dict(parent=parent_sources[name], current=digest)
                          for name, digest in sources.items() if parent_sources[name] != digest},
        prior_dynamics_rng="explicit_checkpointed_generator")
    if mixture:
        continuation["prior_target_rng"] = "explicit_checkpointed_generator"
    run.update(binding=binding, trainer_sources=sources, completed_epochs=320, best_epoch=319,
               effective_maximum_epochs=320, continuation=continuation,
               refit_initialization=copy.deepcopy(parent_run["refit_initialization"]))
    checkpoint.update(next_epoch=320, trainer_sources=sources, effective_maximum_epochs=320,
                      continuation=continuation)
    _json(child / "training/attempts/attempt_002.json",
          dict(run, attempt=2, completed_epochs=160, best_epoch=159, elapsed_seconds=7.))
    for name in ("delivery.json", "training/run.json"):
        shutil.copyfile(parent / name, child / "prerequisites/budget_parent" / name)
    proof = json.loads((child / "delivery.json").read_text())
    proof["optimization_budget_continuation"] = dict(parent=str(parent),
        parent_delivery_sha256=sha256_file(parent / "delivery.json"),
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"],
        parent_completed_epochs=80, total_epochs=320)
    _json(child / "delivery.json", proof)
    _refresh_prior_cell(child, run, checkpoint)
    return run, checkpoint


def _mixture_budget_fixture(root, seed=41):
    child, _ = _continued_prior_cost_fixture(root, seed)
    parent = root / "prior" / f"seed_{seed}"
    run, checkpoint = _budget_profile(parent, child, mixture=True, historical=True)
    return parent, child, run, checkpoint


def _analysis_item(campaign, ceiling):
    """Build the saved-analysis fields consumed by the two pairing functions."""
    bindings = {}
    for seed in analysis.SEEDS:
        cell = campaign / f"seed_{seed}"
        run = json.loads((cell / "training/run.json").read_text())
        checkpoint = torch.load(cell / "training/best.pt", map_location="cpu", weights_only=True)
        bindings[f"population_{seed}"] = dict(training_config=run["binding"]["config"],
            training_report_sha256=sha256_file(cell / "training/run.json"),
            executed_training_budget=dict(base_maximum_epochs=80, effective_maximum_epochs=ceiling),
            prior_refit_lineage=analysis._prior_parent_evidence(cell, run, checkpoint)[2])
    budget = dict(representation="recurrent", maximum_epochs=ceiling, base_maximum_epochs=80,
                  early_stopping_patience=12, training_objective=run["training_objective"],
                  selection_metric=run["selection_metric"],
                  **analysis._prior_config_protocol(run["binding"]["config"]))
    if run["training_objective"] == analysis.PRIOR_MIXTURE_OBJECTIVE:
        budget["mixture_scope"] = analysis.PRIOR_MIXTURE_SCOPE
    # Mirror analyze(..., population_campaign=True), including the helper whose
    # runtime revision differs between the accepted80 and continued320 aggregates.
    code_names = ("population_forecast_analysis.py", "trajectory_forecast_analysis.py",
                  "population_trajectory_evaluation.py", "trajectory_sample_metrics.py",
                  "poisson_state_space.py", "renewal_state_space.py", "refractory_observation.py",
                  "native_renewal_data.py", "poisson_trajectory_evaluation.py",
                  "poisson_trajectory_training.py", "population_state.py", "native_count_windows.py")
    code = {name: sha256_file(Path(analysis.__file__).with_name(name)) for name in code_names}
    code["poisson_trajectory_training.py"] = run["trainer_sources"]["poisson_trajectory_training.py"]
    report = dict(common_training_budget=budget, model_seed_bindings=bindings,
                  inputs=dict(forecasts=str(campaign)), source_binding={"fixture": "same-native-source"},
                  code_sha256=code, configuration={"fixture": "same-forecast-protocol"})
    return report, [], [], [], {}


def test_mixture_continuation_intake_accepts_registered_historical_trainers(tmp_path):
    parent, child, run, checkpoint = _mixture_budget_fixture(tmp_path)
    analysis._campaign_cell(child.parent, 41)
    assert analysis._prior_protocol(run, checkpoint) == run["refit_initialization"]
    lineage = analysis._prior_parent_evidence(child, run, checkpoint)[2]
    assert lineage["parent_checkpoint_sha256"] == run["refit_initialization"]["parent_checkpoint_sha256"]
    parent_run = json.loads((parent / "training/run.json").read_text())
    migration = run["continuation"]["source_migration"]
    assert set(migration) == set(_ACCEPTED_MIXTURE_TRAINERS)
    for name, digest in _ACCEPTED_MIXTURE_TRAINERS.items():
        assert parent_run["trainer_sources"][name] == migration[name]["parent"] == digest
        assert migration[name]["current"] == run["trainer_sources"][name] != digest
    assert run["binding"]["config"]["maximum_epochs"] == 80
    assert run["effective_maximum_epochs"] == 320
    assert run["continuation"]["inherited_early_stopping_patience"] == 12


@pytest.mark.parametrize("failure", ["target_rng", "mixture_source", "unregistered_trainer"])
def test_mixture_continuation_intake_rejects_changed_stream_or_sources(tmp_path, failure):
    _, _, run, checkpoint = _mixture_budget_fixture(tmp_path)
    if failure == "target_rng":
        checkpoint.pop("prior_target_rng")
    elif failure == "mixture_source":
        run["trainer_sources"]["population_predictive_mixture.py"] = "foreign-objective"
    else:
        run["continuation"]["parent_source_hashes"]["renewal_trajectory_training.py"] = "unregistered"
    with pytest.raises(ValueError, match="stream|source|revision"):
        analysis._prior_protocol(run, checkpoint)


def test_mixture_budget_cost_charges_original_refit_and_recovery_once(tmp_path):
    _, child, run, _ = _mixture_budget_fixture(tmp_path)
    cost = analysis._training_attempt_cost(child, run)
    # Original training: 100 + 60 + 20. Mixture80: 4 + 6. Continuation: 7 + 8.
    assert cost["parent_training_elapsed_seconds"] == 190.
    assert cost["incremental_training_elapsed_seconds"] == 15.
    assert cost["training_elapsed_seconds"] == 205.
    assert cost["parent_training_attempts"] == 5
    assert cost["incremental_training_attempts"] == 2
    assert cost["training_attempts"] == 7
    assert cost["full_pipeline_cost_measured"] is False
    assert cost["launcher_staging_seconds"] is None
    assert cost["launcher_archive_seconds"] is None


@pytest.mark.parametrize("failure", ["missing_recovery", "duplicate_recovery"])
def test_mixture_budget_cost_requires_one_bound_recovery_receipt(tmp_path, failure):
    _, child, run, _ = _mixture_budget_fixture(tmp_path)
    proof = json.loads((child / "delivery.json").read_text())
    name = "training/attempts/attempt_002.json"
    row = next(row for row in proof["files"] if row["path"] == name)
    if failure == "missing_recovery":
        proof["files"].remove(row)
    else:
        proof["files"].append(copy.deepcopy(row))
    _json(child / "delivery.json", proof)
    with pytest.raises(ValueError, match="unavailable|duplicate"):
        analysis._training_attempt_cost(child, run)


@pytest.fixture
def mixture_budget_pair(tmp_path):
    for seed in analysis.SEEDS:
        _mixture_budget_fixture(tmp_path, seed)
    return _analysis_item(tmp_path / "prior", 80), _analysis_item(tmp_path / "prior_budget", 320)


def test_mixture_budget_pair_keeps_numerical_code_and_three_seed_lineage(mixture_budget_pair):
    parent, child = mixture_budget_pair
    parent_code, child_code = parent[0]["code_sha256"], child[0]["code_sha256"]
    assert set(parent_code) == set(child_code)
    assert {name for name in parent_code if parent_code[name] != child_code[name]} == {"poisson_trajectory_training.py"}
    assert parent_code["poisson_trajectory_training.py"] == _ACCEPTED_MIXTURE_TRAINERS["poisson_trajectory_training.py"]
    assert child_code["poisson_trajectory_training.py"] == _trainer_sources("population")["poisson_trajectory_training.py"]
    assert analysis._population_budget_pair([child, parent]) == (parent, child)


@pytest.mark.parametrize("failure", ["numerical_code", "unregistered_parent_helper", "unregistered_current_helper",
                                     "added_code_key", "missing_code_key", "particles", "parent_checkpoint"])
def test_mixture_budget_pair_rejects_unmatched_comparison(mixture_budget_pair, tmp_path, failure):
    parent, child = mixture_budget_pair
    if failure == "numerical_code":
        child[0]["code_sha256"]["trajectory_sample_metrics.py"] = "foreign"
    elif failure == "unregistered_parent_helper":
        parent[0]["code_sha256"]["poisson_trajectory_training.py"] = "f" * 64
    elif failure == "unregistered_current_helper":
        child[0]["code_sha256"]["poisson_trajectory_training.py"] = "e" * 64
    elif failure == "added_code_key":
        child[0]["code_sha256"]["unregistered_helper.py"] = "f" * 64
    elif failure == "missing_code_key":
        child[0]["code_sha256"].pop("poisson_trajectory_training.py")
    elif failure == "particles":
        child[0]["model_seed_bindings"]["population_42"]["training_config"]["prediction_particles"] = 9
    else:
        path = tmp_path / "prior_budget/seed_42/training/run.json"
        run = json.loads(path.read_text())
        run["continuation"]["parent_checkpoint_sha256"] = "foreign"
        _json(path, run)
    with pytest.raises(ValueError, match="differs"):
        analysis._population_budget_pair([parent, child])


@pytest.mark.parametrize("seed", analysis.SEEDS)
def test_budget_aggregate_helper_must_match_every_validated_seed_migration(mixture_budget_pair, tmp_path, seed):
    parent, child = mixture_budget_pair
    path = tmp_path / "prior_budget" / f"seed_{seed}" / "training/run.json"
    run = json.loads(path.read_text())
    helper, digest = "poisson_trajectory_training.py", "e" * 64
    run["trainer_sources"][helper] = digest
    run["binding"]["training_utilities_implementation_sha256"] = digest
    lineage = run["continuation"]
    lineage["current_source_hashes"][helper] = digest
    lineage["source_migration"][helper]["current"] = digest
    # This seed still has an admissible historical trainer migration. Its exact
    # current digest disagrees with the aggregate and must therefore be rejected.
    assert analysis.validate_prior_budget_binding(lineage["parent_binding"], run["binding"],
        lineage["parent_source_hashes"], run["trainer_sources"]) == lineage["source_migration"]
    _json(path, run)
    with pytest.raises(ValueError, match="aggregate code differs"):
        analysis._population_budget_pair([parent, child])


def test_non_prior_budget_requires_exact_aggregate_code(tmp_path):
    for seed in analysis.SEEDS:
        _continued_prior_cost_fixture(tmp_path, seed)
    items = []
    for kind, ceiling in (("base", 20), ("parent", 80)):
        campaign, bindings = tmp_path / kind, {}
        for seed in analysis.SEEDS:
            path = campaign / f"seed_{seed}" / "training/run.json"
            run = json.loads(path.read_text())
            bindings[f"population_{seed}"] = dict(training_config=run["binding"]["config"],
                training_report_sha256=sha256_file(path),
                executed_training_budget=dict(effective_maximum_epochs=ceiling))
        report = dict(common_training_budget=dict(representation="recurrent", maximum_epochs=ceiling,
            base_maximum_epochs=20), model_seed_bindings=bindings, inputs=dict(forecasts=str(campaign)),
            source_binding="same", configuration="same", code_sha256={"poisson_trajectory_training.py": "same"})
        items.append((report, [], [], [], {}))
    assert analysis._population_budget_pair(items) == tuple(items)
    items[1][0]["code_sha256"]["poisson_trajectory_training.py"] = "different"
    with pytest.raises(ValueError, match="non-prior aggregate code differs"):
        analysis._population_budget_pair(items)


@pytest.fixture
def continued_objective_pair(tmp_path):
    for seed in analysis.SEEDS:
        child, _ = _continued_prior_cost_fixture(tmp_path, seed)
        parent = tmp_path / "prior" / f"seed_{seed}"
        _budget_profile(parent, child, mixture=False)
        mixture_parent = tmp_path / "mixture" / f"seed_{seed}"
        mixture_child = tmp_path / "mixture_budget" / f"seed_{seed}"
        shutil.copytree(parent, mixture_parent)
        shutil.copytree(child, mixture_child)
        _budget_profile(mixture_parent, mixture_child, mixture=True, historical=True)
    return (_analysis_item(tmp_path / "prior_budget", 320),
            _analysis_item(tmp_path / "mixture_budget", 320))


def test_analytic_and_mixture_continuations_pair_at_same_registered_ceiling(continued_objective_pair):
    analytic, mixture = continued_objective_pair
    assert analysis._population_prior_objective_pair([mixture, analytic]) == (analytic, mixture)
    assert {item[0]["common_training_budget"]["maximum_epochs"]
            for item in continued_objective_pair} == {320}


@pytest.mark.parametrize("scope", ["whole_analysis", "one_seed"])
def test_continued_objective_pair_rejects_unequal_ceilings(continued_objective_pair, scope):
    analytic, mixture = continued_objective_pair
    if scope == "whole_analysis":
        mixture[0]["common_training_budget"]["maximum_epochs"] = 80
        for bound in mixture[0]["model_seed_bindings"].values():
            bound["executed_training_budget"]["effective_maximum_epochs"] = 80
    else:
        mixture[0]["model_seed_bindings"]["population_42"]["executed_training_budget"]["effective_maximum_epochs"] = 80
    with pytest.raises(ValueError, match="budget differs|ceilings differ"):
        analysis._population_prior_objective_pair([analytic, mixture])

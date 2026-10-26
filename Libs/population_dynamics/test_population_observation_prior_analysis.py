"""Admission and saved-result comparisons for the native observation prior."""
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
from .test_population_mixture_budget_analysis import _analysis_item
from .test_population_forecast_analysis import (
    _continued_prior_cost_fixture, _json, _read_rows, _rebind_population_delivery,
    _rebind_prior_analysis, _refresh_prior_cell, _set_prior_fixture_profile,
    delivered_prior_objective_pair, delivered_prior_pair, delivered_representation_pair,
)


_NATIVE = {
    "mixture_scope": "joint_all_source_units_and_time_prefix",
    "observation_score_normalization": "nats_per_identified_unit_and_bin",
    "posterior_cache_role": "causal_boundary_only", "posterior_targets_used": False,
    "future_exposure_role": "observed_event_path_scoring_label_only",
    "future_exposure_used_in_prior_dynamics": False, "marginal_likelihood_exact": False,
}


def _fresh_profile(cell, objective):
    mixed = objective == analysis.PRIOR_MIXTURE_OBJECTIVE
    native = objective == analysis.PRIOR_OBSERVATION_OBJECTIVE
    run, checkpoint = _set_prior_fixture_profile(cell, mixture=mixed)
    config = run["binding"]["config"]
    config["maximum_epochs"] = 80
    sources = _trainer_sources("population")
    names = ["population_prior_refit.py", "population_prior_objective.py"]
    if mixed:
        names.append("population_predictive_mixture.py")
    if native:
        names.append("population_observation_prior.py")
        config.update(schema=analysis.PRIOR_OBSERVATION_SCHEMA,
            observation_objective=objective, prior_selection="fixed_draw_validation_native_event_path_mixture_nll",
            prediction_particles=8)
        config.pop("posterior_target_draws", None)
        run.pop("posterior_target_draws", None)
        checkpoint.pop("prior_target_rng", None)
        run.update(training_objective=objective, selection_metric="validation_native_event_path_mixture_nll",
                   prediction_particles=8, observation_likelihood_evaluated_during_refit=True, **_NATIVE)
        generator = torch.Generator().manual_seed(run["training_seed"])
        for name in ("variational_rng", "prior_dynamics_rng"):
            checkpoint[name] = generator.get_state()
            torch.rand(1, generator=generator)
        for path in (cell / "training/attempts").glob("attempt_*.json"):
            previous = json.loads(path.read_text())
            previous.update(observation_likelihood_evaluated_during_refit=True, **_NATIVE)
            _json(path, previous)
    sources.update({name: sha256_file(Path(analysis.__file__).with_name(name)) for name in names})
    run["binding"].update(trainer_implementation_sha256=sources["renewal_trajectory_training.py"],
        training_utilities_implementation_sha256=sources["poisson_trajectory_training.py"])
    run.update(trainer_sources=sources, completed_epochs=80, best_epoch=79, effective_maximum_epochs=80)
    checkpoint.update(trainer_sources=copy.deepcopy(sources), next_epoch=80, effective_maximum_epochs=80)
    _refresh_prior_cell(cell, run, checkpoint)
    return run, checkpoint


def _native_cell(root, seed=41):
    _continued_prior_cost_fixture(root, seed)
    cell = root / "prior" / f"seed_{seed}"
    run, checkpoint = _fresh_profile(cell, analysis.PRIOR_OBSERVATION_OBJECTIVE)
    return cell, run, checkpoint


def _item(campaign):
    item = _analysis_item(campaign, 80)
    if item[0]["common_training_budget"]["training_objective"] == analysis.PRIOR_OBSERVATION_OBJECTIVE:
        item[0]["common_training_budget"].update(_NATIVE)
        name = analysis.PRIOR_OBSERVATION_MODULE
        item[0]["code_sha256"][name] = sha256_file(Path(analysis.__file__).with_name(name))
    return item


def _pair(root, old_objective):
    for seed in analysis.SEEDS:
        _continued_prior_cost_fixture(root, seed)
        old = root / "prior" / f"seed_{seed}"
        _fresh_profile(old, old_objective)
        native = root / "native" / f"seed_{seed}"
        shutil.copytree(old, native)
        _fresh_profile(native, analysis.PRIOR_OBSERVATION_OBJECTIVE)
    return _item(root / "prior"), _item(root / "native")


def test_native_observation_profile_preserves_frozen_cache_and_unused_targets(tmp_path):
    cell, run, checkpoint = _native_cell(tmp_path)
    assert analysis._prior_protocol(run, checkpoint) == run["refit_initialization"]
    analysis._prior_parent_evidence(cell, run, checkpoint)
    assert run["posterior_targets_fixed"] is True
    assert run["posterior_targets_used"] is False
    budget = analysis._prior_sampling_budget(run)
    assert budget["prediction_particles"] == 8
    assert budget["posterior_target_draws"] is None
    assert budget["posterior_target_integration"] == "not_used"
    assert analysis._training_attempt_cost(cell, run)["training_elapsed_seconds"] == 190.


@pytest.mark.parametrize("failure", ["missing_particles", "particles", "report_particles", "selection",
    "posterior_draws", "future_role", "future_dynamics", "future_posterior", "cache_role",
    "exact_marginal", "source", "dynamics_rng"])
def test_native_observation_profile_rejects_changed_protocol(tmp_path, failure):
    _, run, checkpoint = _native_cell(tmp_path)
    config = run["binding"]["config"]
    if failure == "missing_particles": config.pop("prediction_particles")
    elif failure == "particles": config["prediction_particles"] = 16
    elif failure == "report_particles": run["prediction_particles"] = 4
    elif failure == "selection": run["selection_metric"] = "validation_multistep_prior_kl"
    elif failure == "posterior_draws": config["posterior_target_draws"] = 4
    elif failure == "future_role": run["future_exposure_role"] = "forecast_input"
    elif failure == "future_dynamics": run["future_exposure_used_in_prior_dynamics"] = True
    elif failure == "future_posterior": run["posterior_targets_used"] = True
    elif failure == "cache_role": run["posterior_cache_role"] = "future_posterior_target"
    elif failure == "exact_marginal": run["marginal_likelihood_exact"] = True
    elif failure == "source": run["trainer_sources"].pop(analysis.PRIOR_OBSERVATION_MODULE)
    else: checkpoint.pop("prior_dynamics_rng")
    with pytest.raises(ValueError):
        analysis._prior_protocol(run, checkpoint)


@pytest.mark.parametrize("old_objective", [analysis.PRIOR_REFIT_OBJECTIVE, analysis.PRIOR_MIXTURE_OBJECTIVE])
def test_native_observation_pairs_with_each_original_prior(tmp_path, old_objective):
    old, native = _pair(tmp_path, old_objective)
    assert analysis._population_prior_objective_pair([native, old]) == (old, native)
    assert analysis.PRIOR_OBSERVATION_MODULE in native[0]["code_sha256"]


@pytest.mark.parametrize("failure", ["model", "loader", "score", "missing_key", "unknown_key", "objective_hash"])
def test_native_observation_comparison_rejects_unregistered_sources(tmp_path, failure):
    old, native = _pair(tmp_path, analysis.PRIOR_MIXTURE_OBJECTIVE)
    code = native[0]["code_sha256"]
    if failure == "model": code["poisson_state_space.py"] = "foreign"
    elif failure == "loader": code["native_count_windows.py"] = "foreign"
    elif failure == "score": code["trajectory_sample_metrics.py"] = "foreign"
    elif failure == "missing_key": code.pop("native_renewal_data.py")
    elif failure == "unknown_key": code["unregistered_numerical_source.py"] = "foreign"
    else: code[analysis.PRIOR_OBSERVATION_MODULE] = "f" * 64
    with pytest.raises(ValueError):
        analysis._population_prior_objective_pair([old, native])


@pytest.mark.parametrize("seed", analysis.SEEDS)
def test_native_observation_module_matches_every_training_seed(tmp_path, seed):
    old, native = _pair(tmp_path, analysis.PRIOR_REFIT_OBJECTIVE)
    path = tmp_path / "native" / f"seed_{seed}" / "training/run.json"
    run = json.loads(path.read_text())
    run["trainer_sources"][analysis.PRIOR_OBSERVATION_MODULE] = "f" * 64
    _json(path, run)
    with pytest.raises(ValueError, match="training seed"):
        analysis._population_prior_objective_pair([old, native])


@pytest.mark.parametrize("old_objective", [analysis.PRIOR_REFIT_OBJECTIVE, analysis.PRIOR_MIXTURE_OBJECTIVE])
def test_saved_native_observation_comparison_keeps_labels_and_disclosure(
        delivered_prior_objective_pair, tmp_path, old_objective):
    old_root = delivered_prior_objective_pair[int(old_objective == analysis.PRIOR_MIXTURE_OBJECTIVE)]
    old_report = json.loads((old_root / "analysis/analysis.json").read_text())
    old_campaign = Path(old_report["inputs"]["forecasts"])
    for seed in analysis.SEEDS:
        _fresh_profile(old_campaign / f"seed_{seed}", old_objective)
    native_root, native_campaign = tmp_path / "native_analysis", tmp_path / "native_campaign"
    shutil.copytree(old_root, native_root)
    shutil.copytree(old_campaign, native_campaign)
    for seed in analysis.SEEDS:
        _fresh_profile(native_campaign / f"seed_{seed}", analysis.PRIOR_OBSERVATION_OBJECTIVE)
    for root, campaign, objective in ((old_root, old_campaign, old_objective),
            (native_root, native_campaign, analysis.PRIOR_OBSERVATION_OBJECTIVE)):
        _rebind_prior_analysis(root, campaign, mixture=objective == analysis.PRIOR_MIXTURE_OBJECTIVE)
        report = json.loads((root / "analysis/analysis.json").read_text())
        item = _item(campaign)
        report.update(common_training_budget=item[0]["common_training_budget"], code_sha256=item[0]["code_sha256"])
        _json(root / "analysis/analysis.json", report)
        _rebind_population_delivery(root)
    output = tmp_path / "comparison"
    report = analysis.compare_population_analyses([native_root, old_root], output, prior_objective_comparison=True)
    assert report["models"] == [analysis.PRIOR_METHODS[old_objective], "population_recurrent_native_observation"]
    assert report["training_label_information_match_qualified"] is False
    assert report["scientific_claim_qualified"] is False
    assert report["training_particle_budgets"]["native_observation"]["posterior_target_draws"] is None
    rows = _read_rows(output / "prior_objective_training.csv")
    native_rows = [row for row in rows if row["method"] == "population_recurrent_native_observation"]
    assert len(native_rows) == 3
    assert all(row["posterior_targets_fixed"] == "True" and row["posterior_targets_used"] == "False"
               and row["future_exposure_role"] == _NATIVE["future_exposure_role"] for row in native_rows)
    paired = _read_rows(output / "prior_objective_paired_conditions.csv")
    assert "native_observation_minus_" + analysis.PRIOR_COMPARISON_LABELS[old_objective] in paired[0]

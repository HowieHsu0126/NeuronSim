"""Anchored mean arithmetic, fitting isolation and native source equivalence."""
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from .lfads import LFADSConditionalSimulator, LFADSRandomStreams
from . import lfads_conditional_prior as conditional
from .provenance import sha256_file

CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_anchored_offsets_follow_closed_form_at_episode_native_time():
    curve = np.linspace(-.4, .5, 28, dtype=np.float32).reshape(7, 4)
    phi = np.array([.2, .5, .7, .9], dtype=np.float32)
    actual = conditional.anchored_offsets(curve, phi)
    exact = curve[1:].astype(np.float64) - phi.astype(np.float64)[None] ** np.arange(1, 7)[:, None] * curve[0]
    np.testing.assert_allclose(actual, exact, atol=2e-7, rtol=1e-6)
    np.testing.assert_array_equal(actual[0], curve[1] - phi * curve[0])
    means = np.broadcast_to(np.arange(1200, dtype=np.float32)[None, :, None], (4, 1200, 4)).copy()
    prior = conditional.ConditionalMeanPrior(means)
    np.testing.assert_array_equal(prior.curve("familiar_2", 200, 100)[:, 0], np.arange(199, 300))
    for template, boundary, bins in [("unknown", 200, 100), ("familiar_0", 0, 1),
                                      ("familiar_0", 1199, 2), ("familiar_0", 1.5, 4)]:
        with pytest.raises(ValueError):
            prior.curve(template, boundary, bins)


def test_zero_curve_preserves_signed_zero_bytes_without_another_graph_call():
    future = np.array([[[[-0., 0., -0., 0.], [1., 2., 3., 4.]]]], dtype=np.float32)
    latent = {"future_controller_inputs": future}
    feed = {"source": future}
    model = SimpleNamespace(_full=Mock())
    before = future.tobytes()
    conditional.apply_mean_to_feed(model, feed, latent, np.zeros((3, 4), np.float32))
    assert latent["future_controller_inputs"] is future and future.tobytes() == before
    assert feed["source"] is future
    model._full.session.run.assert_not_called()


@pytest.mark.parametrize("change", ["dtype", "shape", "nonfinite", "boundary_phi"])
def test_invalid_curve_and_parameters_are_rejected(change):
    curve = np.ones((5, 4), np.float32)
    phi = np.full(4, .5, np.float32)
    if change == "dtype": curve = curve.astype(np.float64)
    elif change == "shape": curve = curve[:, :3]
    elif change == "nonfinite": curve[0, 0] = np.nan
    else: phi[0] = 1
    with pytest.raises(ValueError): conditional.anchored_offsets(curve, phi)


@pytest.fixture
def native_model():
    spec = yaml.safe_load((CONFIG_ROOT / "neural_population_dynamics.yaml").read_text())["lfads_baseline_check"]["model"]
    model = LFADSConditionalSimulator([12, 4], 1, 3, 3, .005,
        dict(spec, co_dim=4, controller_input_lag=2), initialization_seed=41, device="cpu")
    counts = np.array([[[0, 2], [1, 0], [2, 1]]], np.int64)
    stimulus = np.linspace(0, .8, 6, dtype=np.float32).reshape(1, 6, 1)
    state = model.infer_state(dict(counts=counts, stimulus=stimulus[:, :3]))
    try: yield model, state, stimulus[:, 3:]
    finally: model.close()


def test_native_zero_mean_matches_all_source_outputs_and_rng_state(native_model):
    model, state, stimulus = native_model
    before = model._full.values()
    sites = list(model._full.randomness.sites)
    left_rng, right_rng = [LFADSRandomStreams.from_seeds(901, 902, 903) for _ in range(2)]
    baseline = model.simulate(state, stimulus, left_rng, 4)
    candidate = conditional.simulate_with_mean(model, state, stimulus, right_rng, 4, np.zeros((4, 4), np.float32))
    assert candidate.base_future_controller_inputs.tobytes() == baseline.future_controller_inputs.tobytes()
    for field in fields(baseline):
        left, right = getattr(baseline, field.name), getattr(candidate, field.name)
        if isinstance(left, np.ndarray):
            assert left.shape == right.shape and left.dtype == right.dtype
            assert left.tobytes() == right.tobytes(), field.name
        else: assert left == right
    for kind in ("initial", "dynamics", "observation"):
        assert getattr(left_rng, kind).bit_generator.state == getattr(right_rng, kind).bit_generator.state
    after = model._full.values()
    for name, value in before.items():
        assert value.tobytes() == after[name].tobytes(), name
    assert model._full.randomness.sites == sites


def test_native_nonzero_mean_preserves_past_boundary_and_same_graph_replay(native_model):
    model, state, stimulus = native_model
    curve = np.linspace(.1, .4, 16, dtype=np.float32).reshape(4, 4)
    rng = lambda: LFADSRandomStreams.from_seeds(901, 902, 903)
    baseline = model.simulate(state, stimulus, rng(), 4)
    first = conditional.simulate_with_mean(model, state, stimulus, rng(), 4, curve)
    second = conditional.simulate_with_mean(model, state, stimulus, rng(), 4, curve)
    for name in ("initial_latent_states", "historical_controller_inputs", "boundary_generator_states"):
        assert getattr(first, name).tobytes() == getattr(baseline, name).tobytes()
    assert not np.array_equal(first.future_controller_inputs, baseline.future_controller_inputs)
    assert first.base_future_controller_inputs.tobytes() == baseline.future_controller_inputs.tobytes()
    for field in fields(first):
        left, right = getattr(first, field.name), getattr(second, field.name)
        if isinstance(left, np.ndarray): assert left.tobytes() == right.tobytes(), field.name


def test_fitting_uses_existing_complete_cache_and_bound_reference(tmp_path):
    from .test_lfads_prior_refit import CacheExecutionTests
    helper = CacheExecutionTests()
    helper.setUp()
    try:
        helper.run_cache()
        cache_path = helper.output / "posterior_cache.json"
        cache = json.loads(cache_path.read_text())
        prior = dict(posterior_cache_sha256=sha256_file(cache_path),
            prior_parameters_sha256=cache["prior_parameters_sha256"], fitted_parameters=8, fitting_episodes=56)
        paired = tmp_path / "paired"; (paired / "analysis").mkdir(parents=True)
        analysis = dict(training_binding=cache["binding"]["training_binding"], arm_bindings={"fitted_prior": dict(
            prior_refit=prior, selected_model_artifacts=cache["binding"]["selected_model_artifacts"])})
        path = paired / "analysis/analysis.json";path.write_text(json.dumps(analysis))
        (paired / "delivery.json").write_text(json.dumps(dict(status="DESCRIBED_REPLAYED_PRIOR_COMPARISON",
            files=[dict(path="analysis/analysis.json", sha256=sha256_file(path))])))
        helper.loader.side_effect = AssertionError("no new neural model may be constructed for mean fitting")
        helper.sampler.side_effect = AssertionError("no new posterior draws may be generated")
        output = tmp_path / "fitted"
        report = conditional.fit_cache_mean(helper.output, paired,
            CONFIG_ROOT / "neural_population_lfads_conditional_prior.yaml", output)
        model, binding = conditional.load_mean_prior(output, prior)
        with np.load(helper.output / cache["episodes"][0]["path"]) as payload:
            expected = payload["controller_paths"].astype(np.float64).mean(axis=0).astype(np.float32)
        np.testing.assert_array_equal(model.means, np.broadcast_to(expected, (4, 1200, 4)))
        assert report["additional_fitted_scalars"] == 19200 and not report["process_noise_identified"]
        assert binding["comparison_protocol"]["primary"]["metric"] == "conditional_empirical_mean_mse_hz2"
        assert not model.means.flags.writeable
        with pytest.raises(ValueError, match="BINDING_CHANGED"):
            conditional.load_mean_prior(output, dict(prior, posterior_cache_sha256="changed"))
        (output / "controller_means.npz").write_bytes(b"changed")
        with pytest.raises(ValueError, match="BINDING_CHANGED"):
            conditional.load_mean_prior(output, prior)
    finally:
        helper.doCleanups()


@pytest.mark.parametrize("seed,budget,selected,completed", [(42, 80, 54, 58), (43, 20, 16, 20)])
def test_matrix_fitting_reuses_complete_cache_with_bound_early_stopped_run(tmp_path, seed, budget, selected, completed):
    from .test_lfads_prior_refit import CacheExecutionTests
    helper = CacheExecutionTests(); helper.setUp()
    try:
        helper.run["binding"] = dict(seed=seed, config=dict(maximum_epochs=20,
            fit_blocks=list(range(14)), validation_blocks=list(range(14, 18))))
        helper.selected.update(next_epoch=selected, training_updates=selected*56)
        helper.model.training_updates = selected*56
        training = dict(status="FITTED_DEVELOPMENT", phase="stopped", training_seed=seed,
            binding=helper.run["binding"], best_epoch=selected-1, completed_epochs=completed,
            effective_maximum_epochs=budget, test_outcomes_accessed=False, calibration_outcomes_accessed=False,
            continuation=None if budget == 20 else dict(schema="neural_population_lfads_budget_continuation_v1",
                total_epochs=80, base_maximum_epochs=20, parent_effective_maximum_epochs=20))
        training_path = helper.training / "run.json"; training_path.write_text(json.dumps(training))
        helper.run_cache()
        cache_path = helper.output / "posterior_cache.json"; cache = json.loads(cache_path.read_text())
        prior = dict(posterior_cache_sha256=sha256_file(cache_path),
            prior_parameters_sha256=cache["prior_parameters_sha256"], fitted_parameters=8,
            fitting_episodes=56, training_seed=seed, selected_epoch=selected)
        paired = tmp_path / "paired"; (paired / "analysis").mkdir(parents=True)
        analysis = dict(training_seed=seed, training_binding=cache["binding"]["training_binding"],
            arm_bindings={"fitted_prior": dict(prior_refit=prior,
                selected_model_artifacts=cache["binding"]["selected_model_artifacts"])})
        path = paired / "analysis/analysis.json"
        def publish_fixture():
            path.write_text(json.dumps(analysis))
            (paired / "delivery.json").write_text(json.dumps(dict(status="DESCRIBED_REPLAYED_PRIOR_COMPARISON",
                files=[dict(path="analysis/analysis.json", sha256=sha256_file(path))])))
        publish_fixture()
        helper.loader.side_effect = AssertionError("mean fitting must reuse the frozen cache")
        helper.sampler.side_effect = AssertionError("mean fitting must not generate posterior draws")
        config = CONFIG_ROOT / "neural_population_lfads_conditional_prior_matrix.yaml"
        output = tmp_path / "mean"
        report = conditional.fit_cache_mean(helper.output, paired, config, output,
            training_seed=seed, network_budget_epochs=budget, training_run=training_path)
        kwargs = dict(training_seed=seed, network_budget_epochs=budget,
                      selected_model_artifacts=cache["binding"]["selected_model_artifacts"])
        model, evidence = conditional.load_mean_prior(output, prior, **kwargs)
        assert report["training_seed"] == seed and report["network_budget_epochs"] == budget
        assert report["training_cell"]["selected_epoch"] == selected
        assert report["training_cell"]["completed_training_epochs"] == completed
        assert not model.means.flags.writeable and evidence["network_budget_epochs"] == budget
        for changed in (dict(kwargs, training_seed=41), dict(kwargs, network_budget_epochs=100-budget),
                        dict(kwargs, selected_model_artifacts={"weights.npz": "different"})):
            with pytest.raises(ValueError, match="BINDING_CHANGED"):
                conditional.load_mean_prior(output, prior, **changed)
        with pytest.raises(ValueError, match="BINDING_CHANGED"):
            conditional.load_mean_prior(output, prior)
        analysis["arm_bindings"]["fitted_prior"]["selected_model_artifacts"] = {"weights.npz": "changed"}
        publish_fixture()
        with pytest.raises(ValueError, match="FITTING_ONLY_CACHE"):
            conditional.fit_cache_mean(helper.output, paired, config, tmp_path / "wrong_weights",
                training_seed=seed, network_budget_epochs=budget, training_run=training_path)
    finally:
        helper.doCleanups()


def training_cell_fixture(tmp_path):
    training_binding = dict(seed=42, config=dict(maximum_epochs=20,
        fit_blocks=list(range(14)), validation_blocks=list(range(14, 18))))
    run = dict(status="FITTED_DEVELOPMENT", phase="stopped", binding=training_binding,
        training_seed=42, effective_maximum_epochs=80, best_epoch=53, completed_epochs=58,
        test_outcomes_accessed=False, calibration_outcomes_accessed=False,
        continuation=dict(schema="neural_population_lfads_budget_continuation_v1",
            total_epochs=80, base_maximum_epochs=20, parent_effective_maximum_epochs=20))
    path = tmp_path / "run.json"; path.write_text(json.dumps(run))
    binding = dict(training_binding=training_binding, training_seed=42, selected_epoch=54,
                   training_run_sha256=sha256_file(path), selected_model_artifacts={"weights.npz": "frozen"})
    return path, run, binding, dict(training_seed=42), dict(training_seed=42, selected_epoch=54)


@pytest.mark.parametrize("change", ["seed", "budget", "checkpoint", "selected_prior", "fit_leakage",
                                      "selection_leakage", "test_access", "calibration_access", "run_hash"])
def test_matrix_training_cell_rejects_cross_cell_and_leakage(tmp_path, change):
    path, run, binding, comparison, prior = training_cell_fixture(tmp_path)
    if change == "seed": comparison["training_seed"] = 43
    elif change == "budget": run["effective_maximum_epochs"] = 20
    elif change == "checkpoint": run["best_epoch"] = 52
    elif change == "selected_prior": prior["selected_epoch"] = 53
    elif change == "fit_leakage": binding["training_binding"]["config"]["fit_blocks"].append(14)
    elif change == "selection_leakage": binding["training_binding"]["config"]["validation_blocks"] = [15, 16, 17]
    elif change == "test_access": run["test_outcomes_accessed"] = True
    elif change == "calibration_access": run["calibration_outcomes_accessed"] = True
    else: run["best_epoch"] = 1
    path.write_text(json.dumps(run))
    if change != "run_hash": binding["training_run_sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        conditional._training_cell(path, binding, comparison, prior, 42, 80)


def test_matrix_protocol_keeps_science_and_requires_explicit_cell():
    path = CONFIG_ROOT / "neural_population_lfads_conditional_prior_matrix.yaml"
    protocol = yaml.safe_load(path.read_text())
    assert conditional._protocol_cell(protocol, path, 43, 80) == (43, 80, True)
    with pytest.raises(ValueError, match="EXPLICIT_REGISTERED"):
        conditional._protocol_cell(protocol, path, None, None)
    protocol["primary"]["horizon_ms"] = 20
    with pytest.raises(ValueError, match="SCIENTIFIC_PROTOCOL_CHANGED"):
        conditional._protocol_cell(protocol, path, 43, 80)
    old_path = CONFIG_ROOT / "neural_population_lfads_conditional_prior.yaml"
    old = yaml.safe_load(old_path.read_text())
    assert conditional._protocol_cell(old, old_path, None, None) == (41, 20, False)
    with pytest.raises(ValueError, match="SINGLE_CELL"):
        conditional._protocol_cell(old, old_path, 42, 20)

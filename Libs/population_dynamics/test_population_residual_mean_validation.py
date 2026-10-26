"""Admission boundaries and the real objective/optimizer path on tiny fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from . import population_residual_mean_validation as validation
from .population_residual_observation_training import MEAN_OBJECTIVE
from .provenance import sha256_file
from .test_population_residual_mean_objective import combined_arguments
from .test_population_residual_observation import predictor


def config():
    return dict(schema="native_residual_observation_fit_v1", training_seeds=[41, 42, 43],
        fitting_blocks=list(range(12)), selection_blocks=[12, 13], refit_blocks=list(range(14)),
        excluded_blocks=list(range(14, 18)), templates_per_block=4, episode_bins=1200, bin_ms=5,
        source_units=100000, origins=[200, 600, 1000], horizons=[4, 16, 100], prediction_particles=8,
        initialization_ridge_alpha=1.0, optimizer="Adam", weight_decay=0.0,
        selection="minimum_complete_unpenalized_validation_loss_earliest_tie",
        stream_protocol="native_residual_observation_named_streams_v1",
        objective=MEAN_OBJECTIVE, field_mean_weight=1.0, mean_unit_chunk_size=32768,
        mean_normalization_blocks=list(range(12)), learning_rate=1e-4,
        regularization_weight=.0001, max_epochs=80, patience=12, unit_chunk_size=4096)


def test_registered_three_origin_protocol_is_accepted():
    validation._registered_config(config())


@pytest.mark.parametrize("name,value", [
    ("origins", [200]), ("prediction_particles", 4), ("unit_chunk_size", 2048),
    ("mean_unit_chunk_size", 4096), ("mean_normalization_blocks", list(range(14))),
    ("learning_rate", .001), ("objective", "native_event_mixture_nll"),
])
def test_protocol_reduction_or_objective_substitution_is_rejected(name, value):
    changed = config()
    changed[name] = value
    with pytest.raises(ValueError):
        validation._registered_config(changed)


@pytest.mark.parametrize("defect", ["scope", "duplicate", "escape", "size", "digest"])
def test_manifest_rejects_changed_scope_or_file_identity(tmp_path, defect):
    source = tmp_path / "source"
    source.write_text("admitted bytes")
    row = dict(path="source", bytes=source.stat().st_size, sha256=sha256_file(source))
    manifest = dict(schema="native_residual_observation_inputs_v1",
        scope="complete_fitting_and_internal_selection", blocks=list(range(14)),
        training_seeds=[41, 42, 43], files=[row])
    if defect == "scope":
        manifest["scope"] = "one_fitting_episode_full_unit_objective_validation"
    elif defect == "duplicate":
        manifest["files"].append(dict(row))
    elif defect == "escape":
        row["path"] = "../source"
    elif defect == "size":
        row["bytes"] += 1
    else:
        row["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validation._verify_input_manifest(tmp_path, path)


@pytest.mark.parametrize("mode,width", [("current", 3), ("history", 7)])
def test_fresh_ridge_receives_only_blocks_zero_through_eleven(monkeypatch, mode, width):
    X = np.broadcast_to(np.arange(14)[:, None, None, None], (14, 1, 2, 7)).copy()
    Y = np.broadcast_to(np.arange(14)[:, None, None, None], (14, 1, 2, 4)).copy()
    seen = {}
    sentinel = object()
    def fit(x, y, selected_width, alpha):
        seen.update(X=x, Y=y, width=selected_width, alpha=alpha)
        return sentinel
    monkeypatch.setattr(validation, "fit_residual_predictor", fit)
    coordinates = (X, Y, dict(current_width=3, fields=2, channels=2))
    assert validation._fresh_reference(coordinates, mode, config()) is sentinel
    np.testing.assert_array_equal(seen["X"], X[:12])
    np.testing.assert_array_equal(seen["Y"], Y[:12])
    assert seen["width"] == width and seen["alpha"] == 1.0


@pytest.mark.parametrize("defect", [None, "term", "key", "rng"])
def test_native_parity_requires_exact_terms_keys_and_public_rng(defect):
    native = dict(loss=torch.tensor(.2), native_nll_h4=torch.tensor(.2), particle_ess_h4=torch.tensor(3.))
    combined = dict(native, native_loss=native["loss"], field_mean_loss=torch.tensor(-.1),
                    field_mean_mse_h4=torch.tensor(-.1))
    combined["loss"] = combined["native_loss"] + combined["field_mean_loss"]
    generators = (torch.Generator().manual_seed(41), torch.Generator().manual_seed(42))
    final = tuple(generator.get_state().clone() for generator in generators)
    if defect == "term":
        combined["native_nll_h4"] = torch.tensor(.21)
    elif defect == "key":
        del combined["particle_ess_h4"]
    elif defect == "rng":
        torch.randn(1, generator=generators[0])
    if defect is None:
        validation._native_parity(native, combined, final, generators, [4])
    else:
        with pytest.raises(ValueError):
            validation._native_parity(native, combined, final, generators, [4])


@pytest.mark.parametrize("seed", [41, 42, 43])
@pytest.mark.parametrize("mode", ["current", "history"])
def test_real_objective_path_preserves_native_terms_and_parent_through_adam(seed, mode):
    args = combined_arguments(mode, seed)
    model = args.pop("model")
    reference = predictor(mode, memory_gain=.15, constant=.01)
    report = dict(optimizer_steps=0)
    validation.exercise_objective(model, reference, args, config(), seed, report)
    assert report["optimizer_steps"] == 1
    assert report["native_terms_exact"] and report["public_rng_exact"]
    assert report["gradient_finite"] and report["gradient_nonzero"]
    assert report["gradient_norm"] > 0 and report["parent_unchanged"]
    assert report["correction_finite"] and report["correction_changed"]
    assert report["post_step_same_random_draws"]
    assert set(report["after_minus_before"]) == set(report["before"]) == set(report["after"])
    assert report["before"]["regularized_loss"] == pytest.approx(
        report["before"]["loss"] + .0001 * report["before"]["ridge_penalty"])
    for name, value in report["before"].items():
        assert report["after_minus_before"][name] == report["after"][name] - value
    for name in ("pure_native_forward_seconds", "forward_seconds", "backward_seconds",
                 "optimizer_seconds", "post_step_forward_seconds"):
        assert report[name] >= 0
    assert report["peak_allocated_bytes"] is None and report["peak_reserved_bytes"] is None


def test_failed_cell_is_persisted_with_its_reason(tmp_path):
    directory = tmp_path / "cell"
    report = validation.measure_cell(None, None, None, config(), 41, "history", directory,
        {"input_manifest_sha256": "declared-input"}, {"device": "cpu"},
        preparation_error="source identity differs")
    saved = json.loads((directory / "analysis.json").read_text())
    assert saved == report
    assert saved["status"] == "FAILED" and saved["exit_code"] == 2
    assert saved["optimizer_steps"] == 0 and "source identity differs" in saved["error"]
    assert saved["scientific_claim_qualified"] is False
    assert saved["production_qualified"] is False
    assert not (directory / "reference.npz").exists()


@pytest.mark.parametrize("host,trial", [("mgmt2", "actual-trial"), ("merlin-worker", None)])
def test_merlin_identity_is_required_before_outputs(monkeypatch, tmp_path, host, trial):
    monkeypatch.setattr(validation.platform, "node", lambda: host)
    monkeypatch.setenv("RH2_JOB_RUN_ID", "actual-run")
    if trial is None:
        monkeypatch.delenv("ARNOLD_TRIAL_ID", raising=False)
    else:
        monkeypatch.setenv("ARNOLD_TRIAL_ID", trial)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Merlin"):
        validation.validate(tmp_path, Path("missing-manifest"), Path("missing-config"), output, "cpu")
    assert not output.exists()

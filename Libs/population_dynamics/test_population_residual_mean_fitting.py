"""Training-only mean normalization, explicit dispatch and resumable selection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import population_residual_observation_fitting as fitting
from . import population_residual_observation_training as training
from .population_residual_observation import configure_correction_parameters
from .provenance import sha256_file
from .test_population_residual_mean_objective import combined_arguments
from .test_population_residual_memory import make_parent, predictor


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/population_residual_mean_fitting.yaml").read_text())


@pytest.mark.parametrize("change", [dict(objective="unknown"), dict(field_mean_weight=0.5),
    dict(field_mean_weight=True), dict(mean_unit_chunk_size=4096), dict(mean_normalization_blocks=list(range(14)))])
def test_mean_configuration_rejects_protocol_drift(change):
    value = config()
    fitting.validate_config(value)
    value.update(change)
    with pytest.raises(ValueError):
        fitting.validate_config(value)


def test_legacy_configuration_remains_explicitly_native():
    value = yaml.safe_load((Path(__file__).parents[1] / "configs/population_residual_observation_fitting.yaml").read_text())
    fitting.validate_config(value)
    assert value.get("objective", training.NATIVE_OBJECTIVE) == training.NATIVE_OBJECTIVE
    value["field_mean_weight"] = 1.
    with pytest.raises(ValueError, match="explicit mean objective"):
        fitting.validate_config(value)


@pytest.fixture
def normalization(tmp_path):
    value = config()
    parent = make_parent()
    units = np.bincount(parent.unit_field_index.numpy(), minlength=len(parent.field_names))
    all_hz, bindings = [], {}
    for index, case in enumerate(fitting.cases(range(12))):
        folder = tmp_path / "cohort/train_fields" / case
        folder.mkdir(parents=True)
        counts = np.arange(3 * len(units)).reshape(3, len(units)) + index + 1
        np.savez(folder / "spatial_fields.npz", count_sum=counts, total_unit_count=units,
                 field_ids=np.asarray(parent.field_names))
        (folder / "summary.json").write_text(json.dumps(dict(settings=dict(bin_ms=5),
            artifacts={"spatial_fields.npz": sha256_file(folder / "spatial_fields.npz")})))
        bindings[case] = {"field_summary_sha256": sha256_file(folder / "summary.json")}
        all_hz.append(counts / (units * .005))
    scale = fitting.fit_field_mean_normalization(tmp_path, value)
    binding = dict(seed=41, episode_bindings=bindings)
    return scale, parent, binding, value, np.concatenate(all_hz), tmp_path


def test_normalization_uses_only_inner_fitting_cases_and_authenticated_units(normalization):
    scale, parent, binding, value, hz, _ = normalization
    sd = fitting.validate_field_mean_normalization(scale, parent, binding, value)
    np.testing.assert_allclose(sd, hz.std(axis=0), rtol=1e-12)
    assert len(scale["source_summaries"]) == 48
    assert not set(scale["cases"]) & set(fitting.cases(range(12, 18)))


@pytest.mark.parametrize("change", ["source", "scope", "map", "units", "scale"])
def test_normalization_rejects_data_scope_or_identity_change(normalization, change):
    scale, parent, binding, value, _, _ = normalization
    scale = copy.deepcopy(scale)
    if change == "source":
        scale["source_summaries"][scale["cases"][0]] = "different"
    elif change == "scope":
        scale["blocks"].append(12)
    elif change == "map":
        scale["field_ids"] = list(reversed(scale["field_ids"]))
    elif change == "units":
        scale["unit_count"][0] += 1
    else:
        scale["standard_deviation_hz"][0] = 0.
    with pytest.raises(ValueError):
        fitting.validate_field_mean_normalization(scale, parent, binding, value)


def test_normalization_rejects_changed_view_and_clock(normalization):
    _, _, _, value, _, root = normalization
    folder = root / "cohort/train_fields" / fitting.cases(range(12))[0]
    report = json.loads((folder / "summary.json").read_text())
    report["settings"]["bin_ms"] = 10
    (folder / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="clock"):
        fitting.fit_field_mean_normalization(root, value)
    report["settings"]["bin_ms"] = 5
    report["artifacts"]["spatial_fields.npz"] = "changed"
    (folder / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="view changed"):
        fitting.fit_field_mean_normalization(root, value)


@pytest.mark.parametrize("seed", [41, 42, 43])
@pytest.mark.parametrize("mode", ["current", "history"])
def test_actual_combined_optimizer_resume_matches_complete_run(tmp_path, seed, mode):
    def run(directory, limit=None):
        batch = combined_arguments(mode, seed, dtype=torch.float32)
        model = batch.pop("model")
        configure_correction_parameters(model)
        frozen = {name: item.clone() for name, item in model.parent.state_dict().items()}
        settings = dict(learning_rate=.0001, regularization_weight=.0001, max_epochs=2,
            patience=2, prediction_particles=8, unit_chunk_size=2, objective=training.MEAN_OBJECTIVE,
            field_mean_weight=1., mean_unit_chunk_size=2)
        report = training.run_epochs(model, predictor(mode, memory_gain=.15, constant=.01),
            lambda case: batch, ["fit"], ["validation"], settings, "qualify", seed,
            directory, {"mode": mode}, epoch_limit=limit)
        assert all(torch.equal(item, model.parent.state_dict()[name]) for name, item in frozen.items())
        return report, model
    full, a = run(tmp_path / "full")
    partial, _ = run(tmp_path / "resume", 1)
    assert partial["status"] == "PARTIAL"
    resumed, b = run(tmp_path / "resume")
    assert full["records"] == resumed["records"]
    for name in training._NAMES:
        assert torch.equal(getattr(a, name), getattr(b, name))
    for epoch in full["records"][1:]:
        row = epoch["training"][0]
        assert row["loss"] == pytest.approx(row["native_loss"] + row["field_mean_loss"])
        assert row["objective"] == pytest.approx(row["loss"] + .0001 * row["penalty"])
        assert row["gradient_norm"] > 0


def test_selection_uses_signed_combined_loss_and_not_native_term(tmp_path):
    model = torch.nn.Module()
    model.correction_weight = torch.nn.Parameter(torch.tensor([.2]))
    model.correction_bias = torch.nn.Parameter(torch.tensor([.1]))
    reference = SimpleNamespace(alpha=1., **{key: torch.ones(1) for key in training._REF_ARRAYS})
    def objective(model, **kwargs):
        assert kwargs["mean_unit_chunk_size"] == 32768
        native = (model.correction_weight + model.correction_bias - 2).square().sum()
        mean = -2 * native - 10
        return dict(loss=native + mean, native_loss=native, field_mean_loss=mean, particle_ess_h4=torch.tensor(8.))
    settings = dict(learning_rate=.01, regularization_weight=.0001, max_epochs=2, patience=2,
        prediction_particles=8, unit_chunk_size=4096, mean_unit_chunk_size=32768,
        objective=training.MEAN_OBJECTIVE, field_mean_weight=1.)
    with patch.object(training, "residual_mean_objective", side_effect=objective), \
            patch.object(training, "residual_observation_mixture_loss", side_effect=AssertionError("wrong objective")), \
            patch.object(training, "standardized_ridge_penalty", return_value=torch.tensor(100.)):
        report = training.run_epochs(model, reference, lambda case: {}, ["fit"], ["validation"],
            settings, "qualify", 41, tmp_path, {})
    assert report["selected_epoch"] == 2
    rows = [item["validation"][0] for item in report["records"]]
    assert rows[-1]["native_loss"] > rows[0]["native_loss"]
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert all(row["field_mean_loss"] < 0 for row in rows)
    assert report["best_validation_loss"] == rows[-1]["loss"]


def test_mean_refit_loader_requires_frozen_training_normalization(normalization):
    scale, parent, binding, settings, _, directory = normalization
    fitting.save_predictor(directory / "predictor.npz", predictor("history"))
    identity = dict(seed=41, mode="history", config=settings,
        parent_checkpoint_sha256="parent", parent_binding_sha256=fitting.json_digest(binding),
        fit_cases=fitting.cases(range(14)), validation_cases=[], selection_sha256="selection",
        field_mean_normalization=scale, sources={name: sha256_file(Path(fitting.__file__).with_name(name)) for name in
        ("population_residual_memory.py", "population_state.py", "refractory_observation.py")})
    report = dict(schema=fitting.SCHEMA, status="FITTED_DEVELOPMENT", phase="refit", binding=identity,
        completed_epochs=1, selected_epoch=1, scientific_claim_qualified=False,
        predictor_sha256=sha256_file(directory / "predictor.npz"))
    (directory / "fit.json").write_text(json.dumps(report))
    loaded = fitting.load_native_observation_correction(directory, parent, binding, "parent")
    assert loaded.correction_identity["binding"]["field_mean_normalization"] == scale
    del identity["field_mean_normalization"]
    (directory / "fit.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="normalization"):
        fitting.load_native_observation_correction(directory, parent, binding, "parent")

"""Native correction admission, serialized memory and independent forecast replay."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import population_forecast_replay as replay_module
from . import population_trajectory_evaluation as evaluation
from . import test_population_trajectory_evaluation as existing
from .population_residual_memory import ResidualMemoryPopulationState
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .residual_prediction import RidgeModel


class LongObservedEpisode(existing.ObservedEpisodeFixture):
    """Keep the registered 20/80/500ms replay horizons on three fixture units."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration_ticks = 600


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    acquisition, fields, training, config_path = existing.MatchedEvaluationTests().fixture(tmp_path)
    case_id = "train_block_001_familiar_0"
    native = LongObservedEpisode(acquisition / case_id, fields / case_id)
    parent = ControlledPopulationState(
        unit_ids=list(map(int, native.unit_ids)), unit_field_index=native.unit_field_index,
        field_names=list(map(str, native.field_ids)), minimum_interval_ticks=native.minimum_interval_ticks,
        tick_seconds=.001, bin_seconds=.005, input_size=1, latent_channels=2,
        encoder_size=3, representation="recurrent", initialization_seed=41, unit_chunk_size=2,
    ).double().eval()
    run = json.loads((training / "run.json").read_text())
    binding = run["binding"]
    binding["config"].update(model_family="population", representation="recurrent", latent_channels=2)
    binding["native_source_identity"] = native.source_identity
    for key, name in (
        ("model_implementation_sha256", "population_state.py"),
        ("observation_implementation_sha256", "refractory_observation.py"),
        ("loader_implementation_sha256", "native_renewal_data.py"),
        ("latent_model_implementation_sha256", "poisson_state_space.py"),
        ("native_counts_implementation_sha256", "native_count_windows.py"),
    ):
        binding[key] = sha256_file(Path(evaluation.__file__).with_name(name))
    torch.save(dict(binding=binding, model=parent.state_dict(), model_configuration=parent.configuration()),
               training / "best.pt")
    run["best_checkpoint_sha256"] = sha256_file(training / "best.pt")
    (training / "run.json").write_text(json.dumps(run))
    settings = yaml.safe_load(config_path.read_text())
    settings.update(origins_ms=[10., 30.], horizons_ms=[20., 80., 500.])
    config_path.write_text(yaml.safe_dump(settings))
    correction_paths = {mode: tmp_path / mode for mode in ("current", "history")}
    for path in correction_paths.values():
        path.mkdir()
    calls = []

    def load_correction(path, loaded_parent, loaded_binding, parent_digest):
        # Only the correction artifact loader is mocked. Parent reconstruction,
        # causal conditioning, sampling, event replay and all metric paths run.
        assert isinstance(loaded_parent, ControlledPopulationState)
        assert loaded_binding == binding
        assert parent_digest == run["best_checkpoint_sha256"]
        mode = Path(path).name
        assert mode in correction_paths
        calls.append(mode)
        d = len(loaded_parent.field_names) * loaded_parent.latent_channels
        p = d + loaded_parent.input_size + (d if mode == "history" else 0)
        coefficients = np.zeros((p, d))
        if mode == "history":
            coefficients[-d:] = .1 * np.eye(d)
        predictor = RidgeModel(.0001, np.zeros(p), np.ones(p), np.full(d, .01), np.ones(d), coefficients)
        identity = dict(fixture="native residual evaluation", mode=mode, alpha=.0001,
                        parent_checkpoint_sha256=parent_digest, seed=loaded_binding["seed"])
        return ResidualMemoryPopulationState(loaded_parent, predictor, mode, identity)

    fake_loader = ModuleType(evaluation.__package__ + ".population_residual_correction")
    fake_loader.load_correction = load_correction
    monkeypatch.setitem(sys.modules, fake_loader.__name__, fake_loader)
    with ExitStack() as stack:
        stack.enter_context(patch.object(evaluation, "NativeRenewalEpisode", LongObservedEpisode))
        stack.enter_context(patch.object(replay_module, "NativeRenewalEpisode", LongObservedEpisode))
        stack.enter_context(patch.object(evaluation, "fitting_scale", existing.MatchedEvaluationTests.scale))
        yield dict(acquisition=acquisition, fields=fields, training=training, config=config_path,
                   corrections=correction_paths, calls=calls, run=run)


def evaluate_case(case, root, mode):
    output = root / ("forecast_" + (mode or "parent"))
    correction = None if mode is None else case["corrections"][mode]
    report = evaluation.evaluate(case["acquisition"], case["fields"], case["training"], case["config"],
                                 output, "population", "cpu", correction_path=correction)
    return output, report


def replay_case(case, forecasts, output, correction):
    return replay_module.replay(case["acquisition"], case["fields"], case["training"], forecasts,
                                output, "cpu", all_origins=True, correction_path=correction)


@pytest.mark.parametrize("mode", ["current", "history"])
def test_corrected_native_outputs_memory_costs_and_event_replay(native_case, tmp_path, mode):
    forecasts, report = evaluate_case(native_case, tmp_path, mode)
    assert report["method"] == "population_residual_" + mode
    assert report["checkpoint_sha256"] == native_case["run"]["best_checkpoint_sha256"]
    assert report["correction_identity"]["mode"] == mode
    assert (report["origins"], report["score_rows"]) == (2, 6)
    assert report["generated_unit_count_values"] == 2 * 4 * 100 * 3
    with (forecasts / "scores.csv").open() as stream:
        assert {row["method"] for row in csv.DictReader(stream)} == {report["method"]}
    for name in report["artifacts"]:
        with np.load(forecasts / name, allow_pickle=False) as arrays:
            assert arrays["initial_memory"].shape == (1, 2, 2)
    for cost in report["origin_costs"]:
        assert cost["full_unit_generation_seconds"] > 0
        assert cost["inference_seconds"] > 0
    receipt = replay_case(native_case, forecasts, tmp_path / "replay.json", native_case["corrections"][mode])
    assert receipt["status"] == "PASS"
    assert receipt["verified_origins"] == 2
    assert receipt["method"] == report["method"]
    assert receipt["correction_identity"] == report["correction_identity"]
    assert receipt["first_origin_exact_replay"]["event_support"] is not None
    assert native_case["calls"] == [mode, mode]


def test_replay_requires_exactly_the_recorded_correction(native_case, tmp_path):
    forecasts, _ = evaluate_case(native_case, tmp_path, "current")
    with pytest.raises(ValueError, match="exactly the recorded correction artifact"):
        replay_case(native_case, forecasts, tmp_path / "missing.json", None)
    with pytest.raises(ValueError, match="correction artifact or method identity differs"):
        replay_case(native_case, forecasts, tmp_path / "changed.json", native_case["corrections"]["history"])
    assert not (tmp_path / "missing.json").exists()
    assert not (tmp_path / "changed.json").exists()


@pytest.mark.parametrize("target", ["full", "last_origin"])
def test_replay_rejects_changed_initial_memory_after_artifact_rehash(native_case, tmp_path, target):
    forecasts, report = evaluate_case(native_case, tmp_path, "history")
    name = "full_unit_samples.npz" if target == "full" else "train_block_001_familiar_0_origin_30.npz"
    path = forecasts / name
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["initial_memory"].flat[-1] += 1
    np.savez_compressed(path, **arrays)
    report["artifacts"][name] = sha256_file(path)
    (forecasts / "evaluation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="(elementwise replay|independent field/latent replay)"):
        replay_case(native_case, forecasts, tmp_path / "rejected.json", native_case["corrections"]["history"])
    assert not (tmp_path / "rejected.json").exists()


def test_uncorrected_native_route_keeps_existing_method_and_array_schema(native_case, tmp_path):
    forecasts, report = evaluate_case(native_case, tmp_path, None)
    assert "correction_identity" not in report and "method" not in report
    with (forecasts / "scores.csv").open() as stream:
        assert {row["method"] for row in csv.DictReader(stream)} == {"population"}
    for name in report["artifacts"]:
        with np.load(forecasts / name, allow_pickle=False) as arrays:
            assert "initial_memory" not in arrays.files
    with pytest.raises(ValueError, match="exactly the recorded correction artifact"):
        replay_case(native_case, forecasts, tmp_path / "unexpected.json", native_case["corrections"]["current"])
    assert replay_case(native_case, forecasts, tmp_path / "parent_replay.json", None)["status"] == "PASS"
    assert native_case["calls"] == []


def test_parent_checkpoint_admission_precedes_correction_loading(native_case, tmp_path):
    checkpoint = native_case["training"] / "best.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="selected checkpoint changed"):
        evaluate_case(native_case, tmp_path, "current")
    assert native_case["calls"] == []
    assert not (tmp_path / "forecast_current").exists()


@pytest.mark.parametrize("schema,model_type", [
    ("native", "poisson"), ("native", "renewal"),
    ("pfc_population_checkpoint_forecast_v1", "population"),
    ("v1_population_checkpoint_forecast_v1", "population"),
])
def test_correction_rejects_other_evaluation_routes_before_source_access(tmp_path, schema, model_type):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(dict(schema=schema)))
    with pytest.raises(ValueError, match="native population evaluation route"):
        evaluation.evaluate(tmp_path, tmp_path, tmp_path, config, tmp_path / "output", model_type,
                            "cpu", correction_path=tmp_path / "correction")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("status,model_type", [
    ("EVALUATED_DEVELOPMENT", "poisson"), ("EVALUATED_DEVELOPMENT", "renewal"),
    ("EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", "population"),
    ("EVALUATED_V1_DEVELOPMENT_CHECKPOINT", "population"),
])
def test_correction_rejects_other_replay_routes_before_source_access(tmp_path, status, model_type):
    (tmp_path / "evaluation.json").write_text(json.dumps(dict(status=status, model_type=model_type)))
    with pytest.raises(ValueError, match="native population replay route"):
        replay_module.replay(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "output.json",
                             "cpu", correction_path=tmp_path / "correction")
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("entry", [evaluation, replay_module])
def test_cost_point_cli_rejects_correction(tmp_path, monkeypatch, capsys, entry):
    args = ["program", "--acquisition", str(tmp_path), "--fields", str(tmp_path),
            "--training", str(tmp_path), "--output", str(tmp_path / "output"),
            "--cost-point", "--correction", str(tmp_path / "correction")]
    args += (["--config", str(tmp_path / "config.yaml")] if entry is evaluation
             else ["--forecasts", str(tmp_path)])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == 2
    assert "not supported for cost-point" in capsys.readouterr().err

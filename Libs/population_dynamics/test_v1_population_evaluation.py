"""V1 original-validation forecasts, causal admission and full-output replay."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import population_trajectory_evaluation as public
from . import population_forecast_replay as replay_public
from . import population_trajectory_training as training
from . import v1_population_evaluation as evaluator
from . import v1_population_training_data as consumer
from .provenance import sha256_file
from .test_v1_population_training_data import configuration, inputs


def fit(root, seed=41):
    spec = inputs(root)
    config = root / "training.yaml"
    config.write_text(yaml.safe_dump(configuration()))
    trained = root / "training"
    training.train(spec, None, config, trained, seed, stop_after_epochs=1)
    values = yaml.safe_load((Path(__file__).resolve().parents[1] /
                            "configs/neural_population_v1_forecast.yaml").read_text())
    values.update(origins_ticks=[10, 20], horizons_bins=[1, 2], support_windows=[1, 2], joint_samples=3)
    forecast = root / "forecast.yaml"
    forecast.write_text(yaml.safe_dump(values))
    return spec, trained, forecast


@pytest.fixture
def fitted(tmp_path):
    return fit(tmp_path)


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_public_v1_forecast_replays_all_origins_and_units(tmp_path, seed):
    spec, trained, config = fit(tmp_path, seed)
    output = tmp_path / "forecasts"
    report = public.evaluate(spec, None, trained, config, output, "population", "cpu")
    assert report["status"] == evaluator.EVALUATION_STATUS
    assert report["training_seed"] == seed
    assert report["episodes"] == 1 and report["score_rows"] == 4 and len(report["origins"]) == 2
    assert report["generated_unit_count_values"] == 36
    assert not report["formal_evaluation"] and not report["scientific_claim_qualified"]
    assert {row["seed_base"] for row in report["origins"]} == {41}
    assert "source_population_evaluation.py" in report["evaluator_sources"]
    receipt = replay_public.replay(spec, None, trained, output, tmp_path / "replay.json", "cpu")
    assert receipt["status"] == "PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY"
    assert len(receipt["origins"]) == 2 and receipt["unit_values_consumed"] == 36
    for row in report["origins"]:
        with np.load(output / row["artifact"], allow_pickle=False) as arrays:
            assert arrays["generated_unit_counts"].shape == (3, 2, 3)
            assert arrays["generated_field_counts"].shape == (3, 2, 2)
            np.testing.assert_array_equal(arrays["unit_ids"], [10, 20, 30])


def test_v1_cli_forecast_and_replay_allow_native_source_without_fields(fitted, tmp_path):
    spec, trained, config = fitted
    output = tmp_path / "cli_forecasts"
    common = ["--acquisition", str(spec), "--training", str(trained), "--device", "cpu"]
    commands = [
        [sys.executable, "-B", "-m", "population_dynamics.population_trajectory_evaluation",
         *common, "--config", str(config), "--output", str(output), "--model-type", "population"],
        [sys.executable, "-B", "-m", "population_dynamics.population_forecast_replay",
         *common, "--forecasts", str(output), "--output", str(tmp_path / "cli_replay.json")],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "cli_replay.json").read_text())["unit_values_consumed"] == 36


@pytest.mark.parametrize("array", ["generated_unit_counts", "unit_ids"])
def test_last_origin_tampering_rejects_even_with_updated_hash(fitted, tmp_path, array):
    spec, trained, config = fitted
    output = tmp_path / "forecasts"
    report = public.evaluate(spec, None, trained, config, output, "population", "cpu")
    name = report["origins"][-1]["artifact"]
    with np.load(output / name, allow_pickle=False) as arrays:
        payload = {key: arrays[key].copy() for key in arrays.files}
    payload[array].flat[-1] += 1
    np.savez_compressed(output / name, **payload)
    report["artifacts"][name] = sha256_file(output / name)
    (output / "evaluation.json").write_text(json.dumps(report))
    receipt = tmp_path / "replay.json"
    with pytest.raises(ValueError, match="ALL_UNIT_OR_SOURCE_MAP"):
        replay_public.replay(spec, None, trained, output, receipt, "cpu")
    assert not receipt.exists()


@pytest.mark.parametrize("change,reason", [
    ("origin", "ORIGIN_DENOMINATOR"), ("score", "COMPLETE_SCORE_DENOMINATOR"),
    ("units", "SOURCE_MAP_SIZE"), ("source", "SOURCE_BINDING"),
])
def test_v1_replay_preserves_complete_source_denominator(fitted, tmp_path, change, reason):
    spec, trained, config = fitted
    output = tmp_path / "forecasts"
    report = public.evaluate(spec, None, trained, config, output, "population", "cpu")
    if change == "origin":
        report["origins"].pop()
    elif change == "score":
        path = output / "scores.csv"
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines + [lines[-1]]))
        report["artifacts"]["scores.csv"] = sha256_file(path)
    elif change == "units":
        report["origins"][-1]["units"] += 1
    else:
        report["origins"][-1]["source_binding"]["block_id"] = "different_block"
    (output / "evaluation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=reason):
        replay_public.replay(spec, None, trained, output, tmp_path / "replay.json", "cpu")
    assert not (tmp_path / "replay.json").exists()


@pytest.mark.parametrize("change", ["missing", "shared"])
def test_v1_replay_binds_shared_implementation_before_loading_weights(fitted, tmp_path, change):
    spec, trained, config = fitted
    output = tmp_path / "forecasts"
    report = public.evaluate(spec, None, trained, config, output, "population", "cpu")
    if change == "missing":
        report.pop("evaluator_sources")
    else:
        report["evaluator_sources"]["source_population_evaluation.py"] = "0" * 64
    (output / "evaluation.json").write_text(json.dumps(report))
    with patch.object(evaluator, "_load", side_effect=AssertionError("weights read")):
        with pytest.raises(ValueError, match="EVALUATOR_OR_STATUS_CHANGED"):
            replay_public.replay(spec, None, trained, output, tmp_path / "replay.json", "cpu")
    assert not (tmp_path / "replay.json").exists()


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_v1_evaluator_requires_original_validation_before_arrays_or_weights(tmp_path, split):
    spec = inputs(tmp_path, validation_split=split)
    trained = tmp_path / "training"; trained.mkdir()
    (trained / "run.json").write_text(json.dumps(dict(input_family="v1_qualified_source_episodes",
        status="CHECKPOINTED_DEVELOPMENT", binding=dict(config=configuration()))))
    with patch.object(consumer, "V1RenewalEpisode", side_effect=AssertionError("source arrays read")), \
         patch.object(evaluator.torch, "load", side_effect=AssertionError("weights read")):
        with pytest.raises(ValueError, match="ORIGINAL_VALIDATION_CIRCUITS_MISSING_OR_MISASSIGNED"):
            evaluator._load(spec, trained, torch.device("cpu"))


def test_v1_evaluator_rejects_changed_checkpoint_source_buffers(fitted):
    spec, trained, _ = fitted
    checkpoint = torch.load(trained / "best.pt", weights_only=True)
    checkpoint["model"]["minimum_interval_ticks"][0] += 1
    torch.save(checkpoint, trained / "best.pt")
    run = json.loads((trained / "run.json").read_text())
    run["best_checkpoint_sha256"] = sha256_file(trained / "best.pt")
    (trained / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="SOURCE_BUFFER_CHANGED"):
        evaluator._load(spec, trained, torch.device("cpu"))


def test_v1_causal_prefix_readiness_and_generation_precede_future_truth(fitted):
    spec, trained, config_path = fitted
    model, cohort, _ = evaluator._load(spec, trained, torch.device("cpu"))
    case = cohort.validation[0]
    data = cohort._episode(case)
    cohort.active_case, cohort.active_episode = case, data
    view = cohort.model_for_case(model, case).eval()
    shared = evaluator.shared
    original_window = data.window
    calls = []
    def prefix_only(start, stop):
        assert (start, stop) == (0, 10)
        calls.append((start, stop))
        return original_window(start, stop)
    with patch.object(data, "window", prefix_only):
        first, controls = shared.prepare_factual_response(view, data, 10, 2)
        data.event_ticks[data.event_ticks >= 10] += 1
        second, again = shared.prepare_factual_response(view, data, 10, 2)
    assert calls == [(0, 10), (0, 10)]
    assert torch.equal(first.mean, second.mean) and torch.equal(first.variance, second.variance)
    np.testing.assert_array_equal(first.ready_in_ticks.cpu().numpy(), [[7, 0, 0]])
    np.testing.assert_array_equal(controls["known_controls"], again["known_controls"])
    assert controls["control_features"] == data.model_control_features
    calls.clear()
    draw = shared._draw
    def observed_window(start, stop):
        calls.append((start, stop))
        return original_window(start, stop)
    def generate(*args, **kwargs):
        calls.append("generate")
        return draw(*args, **kwargs)
    with patch.object(data, "window", observed_window), patch.object(shared, "_draw", generate):
        shared._forecast(model, cohort, case, 10, evaluator._config(config_path), 41,
                         torch.device("cpu"), prepared=(view, data))
    assert calls == [(0, 10), "generate", (10, 20)]


def test_v1_forecast_rejects_misaligned_origin(fitted, tmp_path):
    spec, trained, config = fitted
    values = yaml.safe_load(config.read_text()); values["origins_ticks"] = [11]
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="ALIGNED_PREFIX"):
        public.evaluate(spec, None, trained, config, tmp_path / "invalid", "population", "cpu")


def test_v1_forecast_requires_registered_stream_policy(fitted):
    _, _, config = fitted
    values = yaml.safe_load(config.read_text()); values.pop("random_seed_policy")
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="V1_FORECAST_STREAM_SEEDS"):
        evaluator._config(config)

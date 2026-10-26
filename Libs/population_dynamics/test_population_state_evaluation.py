"""Population streaming identity and complete evaluator-to-replay contracts.

The existing tiny observation fixture replaces only the native input boundary.
Counts, source maps, random streams and forecast/replay code run without stubs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from . import population_forecast_replay as replay_module
from . import population_trajectory_evaluation as evaluator
from .population_state import ControlledPopulationState
from .poisson_state_space import LatentRandomStreams
from .provenance import sha256_file
from .renewal_check import replay_events
from .test_population_forecast_unit_metrics_replay import CompleteOriginFixture, write_rows
from . import test_population_trajectory_evaluation as existing
from .test_native_future_ensemble import fixture as native_ensemble_fixture


def test_cost_point_real_native_loading_full_output_and_registered_scope(native_ensemble_fixture, monkeypatch):
    fixture = native_ensemble_fixture
    monkeypatch.setattr(evaluator, "COST_POINT_UNITS", 32)
    monkeypatch.setattr(evaluator, "COST_POINT_FIELDS", 32)
    monkeypatch.setattr(replay_module, "COST_POINT_UNITS", 32)
    monkeypatch.setattr(replay_module, "COST_POINT_FIELDS", 32)
    root = fixture["graph"].parent
    acquisition, fields, training = [root / name for name in ("acquisition", "fields", "training")]
    for path in (acquisition, fields, training):
        path.mkdir()
    case = evaluator.COST_POINT_CASE
    fixture["reference_episode"].rename(acquisition / case)
    fixture["reference_fields"].rename(fields / case)
    source = json.loads((acquisition / case / "report.json").read_text())
    (acquisition / "design.json").write_text(json.dumps(dict(cases=[dict(case_id=case, split="train", block=14)],
        graph_manifest_sha256=source["graph_manifest_sha256"], native_library_sha256=source["native_library_sha256"])))
    (fields / "dataset.json").write_text(json.dumps(dict(scope="explicit32unit_native_fixture")))
    native = evaluator.NativeRenewalEpisode(acquisition / case, fields / case)
    configuration = dict(unit_ids=list(map(int, native.unit_ids)), unit_field_index=list(map(int, native.unit_field_index)),
        field_names=list(map(str, native.field_ids)), minimum_interval_ticks=list(map(int, native.minimum_interval_ticks)),
        tick_seconds=.0001, bin_seconds=.005, input_size=1, latent_channels=4, encoder_size=16,
        representation="recurrent", initialization_seed=41, unit_chunk_size=16)
    model = ControlledPopulationState(**configuration).double()
    binding = dict(seed=41, config=dict(model_family="population", bin_ms=5., cpu_threads=1),
        acquisition_design_sha256=sha256_file(acquisition / "design.json"), dataset_sha256=sha256_file(fields / "dataset.json"),
        native_source_identity=native.source_identity)
    for key, filename in (("model_implementation_sha256", "population_state.py"),
                          ("observation_implementation_sha256", "refractory_observation.py"),
                          ("loader_implementation_sha256", "native_renewal_data.py"),
                          ("latent_model_implementation_sha256", "poisson_state_space.py"),
                          ("native_counts_implementation_sha256", "native_count_windows.py")):
        binding[key] = sha256_file(Path(evaluator.__file__).with_name(filename))
    torch.save(dict(model=model.state_dict(), model_configuration=configuration, binding=binding), training / "best.pt")
    (training / "run.json").write_text(json.dumps(dict(status="FITTED_DEVELOPMENT", binding=binding,
                                                       best_checkpoint_sha256=sha256_file(training / "best.pt"))))
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_population_cost_point.yaml").read_text())
    config.update(expected_core_units=32, expected_field_count=32)
    config_path = root / "cost_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    (root / config["measurement_config"]).write_text(fixture["config_path"].read_text())
    result = evaluator.evaluate_cost_point(acquisition, fields, training, fixture["fitting_scale_path"], config_path,
                                           root / "cost_output", "cpu")
    assert result["status"] == "EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT"
    assert result["all_unit_outputs_saved"] is True
    assert result["generated_unit_count_values"] == 16 * 100 * 32
    assert result["score_rows"] == 3
    assert result["training_performed"] is False and result["matched_fidelity"] is False
    assert json.loads((root / "cost_output/cost_point.json").read_text()) == result
    delivery = root / "cost_delivery"
    delivery.mkdir()
    (root / "cost_output").rename(delivery / "cost_point")
    (delivery / "forecasts").mkdir()
    (delivery / "forecasts/fitting_field_scale.npz").write_bytes(fixture["fitting_scale_path"].read_bytes())
    proof = dict(status="PASS", exit_code=0, generated_unit_count_values=16 * 100 * 32,
                 binding=dict(checkpoint_sha256=result["checkpoint_sha256"]), process_seconds=2., task_through_archive_seconds=3.)
    proof["files"] = [dict(path=str(p.relative_to(delivery)), bytes=p.stat().st_size, sha256=sha256_file(p))
                      for p in sorted(delivery.rglob("*")) if p.is_file()]
    (delivery / "delivery.json").write_text(json.dumps(proof))
    replay = replay_module.replay_cost_point(acquisition, fields, training, delivery, root / "cost_replay.json", "cpu")
    assert replay["status"] == "PASS" and replay["exact_generated_unit_count_values"] == 16 * 100 * 32
    assert replay["independently_consumed_unit_count_values"] == 16 * 100 * 32
    assert replay["checked_score_cells"] == 42
    # Rebind changed score bytes so rejection exercises numerical verification.
    score_path = delivery / "cost_point/scores.csv"
    with score_path.open(newline="") as stream:
        score_rows = list(csv.DictReader(stream))
    score_rows[0]["unit_total_coverage_90"] = "0.0"
    with score_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader(); writer.writerows(score_rows)
    proof["files"] = [dict(path=str(p.relative_to(delivery)), bytes=p.stat().st_size, sha256=sha256_file(p))
                      for p in sorted(delivery.rglob("*")) if p.is_file() and p.name != "delivery.json"]
    (delivery / "delivery.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="independent cost-point unit metric"):
        replay_module.replay_cost_point(acquisition, fields, training, delivery, root / "changed_cost_replay.json", "cpu")
    assert json.loads((root / "changed_cost_replay.json").read_text())["status"] == "FAILED"


def population_model(representation, *, seed=41):
    model = ControlledPopulationState(
        unit_ids=[11, 17, 29], unit_field_index=[0, 1, 0],
        field_names=['["L23E",0,0]', '["L23I",0,0]'],
        minimum_interval_ticks=[2, 3, 2], tick_seconds=.001, bin_seconds=.005,
        input_size=1, latent_channels=4, encoder_size=16,
        representation=representation, initialization_seed=seed, unit_chunk_size=2).double()
    with torch.no_grad():
        model.log_hazard_bias.copy_(torch.tensor([20., 40., 4000.]).double().log())
    return model


@pytest.mark.parametrize("representation", ["fixed", "recurrent"])
def test_streaming_collector_preserves_counts_events_states_and_all_random_streams(representation):
    model = population_model(representation)
    ready = np.array([[1, 2, 0], [0, 1, 1]], dtype=np.int64)
    history = dict(counts=np.ones((2, 4, 3)), stimulus=np.zeros((2, 4, 1)), ready_in_ticks=ready)
    state = model.infer_state(history)
    future = np.ones((2, 6, 1))
    manual_rng, event_rng, count_rng = [LatentRandomStreams.from_seeds(301, 302, 303) for _ in range(3)]
    global_state = torch.get_rng_state().clone()
    steps = list(model.iter_simulation(state, future, manual_rng, 7, record_events=True))
    assert len(steps) == 6
    expected = (torch.stack([step.counts for step in steps], 2),
                torch.stack([step.latent for step in steps], 2),
                steps[0].initial_latent, steps[-1].ready_in_ticks,
                torch.cat([step.events for step in steps]))
    recorded = evaluator.sample_population_outputs(model, state, future, event_rng, 7, record_events=True)
    count_only = evaluator.sample_population_outputs(model, state, future, count_rng, 7, record_events=False)
    for left, right in zip(expected, recorded):
        assert torch.equal(left, right)
    for left, right in zip(expected[:4], count_only[:4]):
        assert torch.equal(left, right)
    assert count_only[4] is None and len(recorded[4]) > 0
    assert recorded[0].shape == (7, 2, 6, 3)
    assert recorded[1].shape == (7, 2, 6, 2, 4)
    for step in steps:
        assert torch.equal(step.initial_latent, recorded[2])
    for name in ("initial", "dynamics", "observation"):
        states = [getattr(streams, name).get_state() for streams in (manual_rng, event_rng, count_rng)]
        assert torch.equal(states[0], states[1]) and torch.equal(states[0], states[2])
    assert torch.equal(global_state, torch.get_rng_state())
    # The established sparse-event checker accepts one episode at a time.
    for batch in range(2):
        events = recorded[4].numpy()
        events = events[events[:, 1] == batch].copy()
        events[:, 1] = 0
        support, terminal = replay_events(recorded[0][:, batch:batch + 1].numpy(), events,
                                         ready[batch:batch + 1], np.array([2, 3, 2]), 5)
        assert support["refractory_violations"] == 0 and support["boundary_violations"] == 0
        np.testing.assert_array_equal(terminal, recorded[3][:, batch:batch + 1].numpy())


def population_fixture(root, representation, monkeypatch):
    helper = existing.MatchedEvaluationTests()
    acquisition, fields, training, config_path = helper.fixture(root)
    design = json.loads((acquisition / "design.json").read_text())
    case = dict(case_id="train_block_002_familiar_0", block=2, split="train",
                family="trajectory", template_id="familiar_0")
    design["cases"].append(case)
    for folder, name in ((acquisition, "report.json"), (fields, "summary.json")):
        target = folder / case["case_id"]
        target.mkdir()
        (target / name).write_text(json.dumps({"fixture_case": case["case_id"]}))
    (acquisition / "design.json").write_text(json.dumps(design))
    rows = [dict(case_id=row["case_id"], split="train",
                 raw_report_sha256=sha256_file(acquisition / row["case_id"] / "report.json"),
                 field_summary_sha256=sha256_file(fields / row["case_id"] / "summary.json"))
            for row in design["cases"]]
    write_rows(fields / "episodes.csv", rows)
    dataset = json.loads((fields / "dataset.json").read_text())
    dataset.update(completed_episodes=len(rows), episode_table_sha256=sha256_file(fields / "episodes.csv"))
    (fields / "dataset.json").write_text(json.dumps(dataset))
    run = json.loads((training / "run.json").read_text())
    binding = run["binding"]
    binding.update(seed=43, acquisition_design_sha256=sha256_file(acquisition / "design.json"),
                   dataset_sha256=sha256_file(fields / "dataset.json"))
    binding["config"].update(model_family="population", latent_channels=4, encoder_size=16,
                             representation=representation, unit_chunk_size=2, validation_blocks=[1, 2])
    native = CompleteOriginFixture(acquisition / case["case_id"], fields / case["case_id"])
    binding["native_source_identity"] = native.source_identity
    for key, filename in [("model_implementation_sha256", "population_state.py"),
                          ("observation_implementation_sha256", "refractory_observation.py"),
                          ("loader_implementation_sha256", "native_renewal_data.py"),
                          ("latent_model_implementation_sha256", "poisson_state_space.py"),
                          ("native_counts_implementation_sha256", "native_count_windows.py")]:
        binding[key] = sha256_file(Path(__file__).with_name(filename))
    model = population_model(representation, seed=43)
    torch.save(dict(binding=binding, model=model.state_dict(), model_configuration=model.configuration()),
               training / "best.pt")
    run.update(binding=binding, model_family="population", representation=representation,
               best_checkpoint_sha256=sha256_file(training / "best.pt"))
    (training / "run.json").write_text(json.dumps(run))
    settings = yaml.safe_load(config_path.read_text())
    settings.update(origins_ms=[10., 25., 50.], horizons_ms=[20., 80., 500.])
    config_path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(evaluator, "NativeRenewalEpisode", CompleteOriginFixture)
    monkeypatch.setattr(evaluator, "fitting_scale", helper.scale)
    monkeypatch.setattr(replay_module, "NativeRenewalEpisode", CompleteOriginFixture)
    return acquisition, fields, training, config_path


@pytest.mark.parametrize("representation", ["fixed", "recurrent"])
def test_real_population_evaluation_and_all_origin_replay_preserve_every_unit(tmp_path, monkeypatch, representation):
    acquisition, fields, training, config = population_fixture(tmp_path, representation, monkeypatch)
    forecasts = tmp_path / "forecasts"
    result = evaluator.evaluate(acquisition, fields, training, config, forecasts, "population", "cpu")
    assert result["status"] == "EVALUATED_DEVELOPMENT" and result["model_type"] == "population"
    assert result["episodes"] == 2 and result["score_rows"] == 18
    assert not result["likelihood_scores_compared"]
    with np.load(forecasts / "full_unit_samples.npz", allow_pickle=False) as arrays:
        assert arrays["sampled_unit_counts"].shape == (4, 100, 3)
        assert arrays["latent_states"].shape == (4, 100, 2, 4)
        assert arrays["initial_latent_states"].shape == (4, 1, 2, 4)
        assert arrays["terminal_ready_in_ticks"].shape == (4, 1, 3)
        np.testing.assert_array_equal(arrays["unit_ids"], [11, 17, 29])
    def forbidden(*args, **kwargs):
        raise AssertionError("independent replay called the production unit metric scorer")
    monkeypatch.setattr(evaluator, "unit_count_metrics", forbidden)
    called = []
    original = replay_module._draw
    def record_draw(model, model_type, state, future, seed, draws, device):
        called.append(seed)
        return original(model, model_type, state, future, seed, draws, device)
    monkeypatch.setattr(replay_module, "_draw", record_draw)
    report = replay_module.replay(acquisition, fields, training, forecasts, tmp_path / "replay.json",
                                  "cpu", all_origins=True)
    assert report["status"] == "PASS" and report["model_type"] == "population"
    assert report["verified_origins"] == report["expected_origins"] == 6
    assert report["verified_score_rows"] == 18 and report["metric_comparisons"] == 6 * 3 * 8
    assert report["sampled_unit_count_values_consumed"] == 6 * 4 * 100 * 3
    assert report["all_forecast_unit_metrics_replayed"] and report["all_origin_fields_and_latents_exact"]
    assert report["saved_full_unit_origins_compared"] == 1
    assert report["regenerated_origins_without_saved_full_unit_arrays"] == 5
    assert not report["all_origin_unit_counts_compared_to_saved_full_arrays"]
    support = report["first_origin_exact_replay"]["event_support"]
    assert support["refractory_violations"] == 0 and support["boundary_violations"] == 0
    base = result["configuration"]["sampling_seed_base"] + 43 * 100000
    assert called == [base + case * 100 + origin * 3 for case in range(2) for origin in range(3)]
    with (tmp_path / "replay.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 * 3 * 8 and all(row["passed"] == "True" for row in rows)


@pytest.mark.parametrize("changed", ["constructor_map", "state_map", "field_names", "clock", "source", "loader"])
def test_population_checkpoint_map_source_and_clock_rejected_before_any_output(tmp_path, monkeypatch, changed):
    acquisition, fields, training, config = population_fixture(tmp_path, "recurrent", monkeypatch)
    checkpoint = torch.load(training / "best.pt", weights_only=True)
    constructor = checkpoint["model_configuration"]
    if changed == "constructor_map":
        constructor["unit_field_index"] = [1, 0, 1]
    elif changed == "state_map":
        checkpoint["model"]["unit_field_index"] = torch.tensor([1, 0, 1])
    elif changed == "field_names":
        constructor["field_names"][0] = "changed_field"
    elif changed == "clock":
        constructor["tick_seconds"] = .0005
    elif changed == "source":
        checkpoint["binding"]["native_source_identity"]["scope"] = "different_source"
    else:
        checkpoint["binding"]["loader_implementation_sha256"] = "0" * 64
    torch.save(checkpoint, training / "best.pt")
    run = json.loads((training / "run.json").read_text())
    run.update(binding=checkpoint["binding"], best_checkpoint_sha256=sha256_file(training / "best.pt"))
    (training / "run.json").write_text(json.dumps(run))
    def forbidden(*args, **kwargs):
        raise AssertionError("changed checkpoint reached forecast sampling")
    monkeypatch.setattr(evaluator, "_draw", forbidden)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="map changed|implementation changed|source.*changed|clock.*changed|checkpoint.*map|state.*map"):
        evaluator.evaluate(acquisition, fields, training, config, output, "population", "cpu")
    assert not output.exists()

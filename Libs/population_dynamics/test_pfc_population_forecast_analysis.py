"""Delivered score/replay/cost contracts without generating scientific samples."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import pfc_population_forecast_analysis as analysis
from .provenance import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def seal(cell):
    proof = json.loads((cell / "delivery.json").read_text())
    proof["files"] = [dict(path=str(p.relative_to(cell)), bytes=p.stat().st_size,
                           sha256=sha256_file(p)) for p in sorted(cell.rglob("*"))
                      if p.is_file() and p.name != "delivery.json"]
    write(cell / "delivery.json", proof)


@pytest.fixture
def delivered(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "configs/neural_population_pfc_task_training.yaml").read_text())
    forecast = yaml.safe_load((repo / "configs/neural_population_pfc_task_forecast.yaml").read_text())
    prepared = tmp_path / "prepared"
    write(prepared / "preparation.json", {"fixture": "bound input metadata already admitted"})
    input_hash, evaluator_hash = "1" * 64, "2" * 64
    records = [dict(fixture_fit=i) for i in range(80)]
    cases, contracts, source_costs = [], {}, []
    for block in range(3):
        for offset, condition in enumerate(analysis.CONDITIONS):
            episode = f"pfc_working_memory_circuit_074_block_{block:03d}_condition_{offset:02d}"
            case_id = "pfc_e7072637ba0e1076__" + episode
            binding = dict(episode_id=episode, circuit_id="pfc_e7072637ba0e1076",
                           block_id=f"pfc_initial_{block}_external_{block}")
            case = dict(case_id=case_id, episode_id=episode, circuit_id=binding["circuit_id"],
                        circuit_index=74, block_index=block, condition_id=condition,
                        block=binding["block_id"], source_binding=binding)
            cases.append(case)
            records.append(dict(partition="development_validation", **binding))
            contracts[case_id] = dict(block=case["block"], unit_count=100000, field_count=17,
                bin_ms=5., source_dt_ms=.05, episode_bins=1000,
                full_unit_retention="all_origins", episode_inputs_materialized=False)
            source_costs.append(dict(case_id=case_id, source_block=case["block"],
                                     simulator_wall_seconds=10., simulator_cost_scope="complete source"))
    frozen = SimpleNamespace(root=prepared, proof=dict(input_sha256=input_hash), config=config,
        forecast=forecast, records=records, validation=cases, contracts=contracts,
        source_costs=source_costs, checked={},
        files={"Libs/population_dynamics/pfc_population_evaluation.py": {"sha256": evaluator_hash}})
    # Source admission itself is checked against the real frozen preparation in
    # the recorded CLI validation; these fixtures exercise the output consumer.
    monkeypatch.setattr(analysis, "FrozenInputs", lambda path: frozen)
    campaign = tmp_path / "campaign"
    for seed in analysis.SEEDS:
        cell = campaign / f"seed_{seed}"
        run = dict(status="FITTED_DEVELOPMENT", training_seed=seed,
            representation="recurrent",
            input_family="pfc_source_reproduction", fitting_episodes=80, validation_episodes=30,
            binding=dict(config=dict(config, device="cpu", cpu_threads=2, representation="recurrent"),
                         seed=seed, circuit_inputs_sha256=input_hash, complete_episode_bindings=records),
            calibration_outcomes_accessed=False, test_outcomes_accessed=False,
            scientific_claim_qualified=False, completed_epochs=20, best_epoch=19,
            effective_maximum_epochs=20, stop_reason="epoch_budget", trainer_sources={},
            elapsed_seconds=100., peak_process_rss_bytes=1000000, parameter_count=2649)
        checkpoint = cell / "training/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"Opaque model payload; analysis must never open this file.")
        run["best_checkpoint_sha256"] = sha256_file(checkpoint)
        write(cell / "training/run.json", run)
        score_rows, origins, replay_origins, origin_costs, episode_costs, artifacts = [], [], [], [], [], {}
        for case in cases:
            episode_costs.append(dict(case_id=case["case_id"], block=case["block"],
                native_duration_ticks=100000, native_bins=1000, unit_count=100000,
                origins_ms=[v * .05 for v in analysis.ORIGINS], episode_inputs_materialized=False,
                count_dtype=None, stimulus_dtype=None, count_tensor_dtype=None, stimulus_tensor_dtype=None,
                count_tensor_device=None, stimulus_tensor_device=None,
                loading_seconds=.5, window_seconds=0., transfer_seconds=0., episode_preparation_seconds=1.))
            for origin in analysis.ORIGINS:
                artifact = case["case_id"] + f"_origin_{origin}.npz"
                payload = cell / "forecasts" / artifact
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(b"Opaque generated sample fixture; no array loading allowed.")
                artifacts[artifact] = sha256_file(payload)
                origins.append(dict(case_id=case["case_id"], origin_tick=origin,
                    source_binding=case["source_binding"], units=100000, fields=17,
                    source_metadata_identity="same source map " + case["case_id"], artifact=artifact))
                replay_origins.append(dict(case_id=case["case_id"], origin_tick=origin,
                                           unit_values=16 * 100 * 100000))
                timers = dict(inference_seconds=.1, full_unit_generation_seconds=.1,
                              field_projection_seconds=.1, peak_gpu_allocated_bytes=None)
                for horizon in analysis.HORIZONS:
                    metrics = {name: .1 for name in analysis.METRICS}
                    metrics["unit_bin_mean_mse_corrected"] = -.125
                    score_rows.append(dict(case_id=case["case_id"], circuit_id=case["circuit_id"],
                        origin_tick=origin, horizon_bins=horizon, training_seed=seed, **metrics, **timers))
                origin_costs.append(dict(case_id=case["case_id"], block=case["block"], origin_ms=origin * .05,
                    horizons_ms=[v * 5. for v in analysis.HORIZONS], generated_horizon_ms=500.,
                    generated_bins=100, draws=16, unit_count=100000, field_count=17, score_rows=3,
                    generated_unit_count_values=16 * 100 * 100000,
                    generated_unit_count_dtype="torch.int64", generated_unit_count_device="cpu",
                    latent_dtype="torch.float32", latent_device="cpu", field_count_dtype="int64",
                    field_metric_dtype="float64", stored_full_unit_counts=True, **timers,
                    scoring_seconds=.1, serialization_seconds=.1, hash_seconds=.1, csv_seconds=.1,
                    origin_total_seconds=1., peak_process_rss_bytes=1000000))
        scores = cell / "forecasts/scores.csv"
        rows(scores, score_rows)
        artifacts["scores.csv"] = sha256_file(scores)
        scope = {name: "fixture measured boundary" for name in (
            "function_total_seconds", "elapsed_seconds", "setup_seconds", "episode_costs",
            "origin_total_seconds", "horizon_reuse", "peak_process_rss_bytes")}
        scope.update(speedup_qualified=False, excluded_costs=["source execution", "training"])
        evaluation = dict(status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", training_seed=seed,
            configuration=forecast, checkpoint_sha256=run["best_checkpoint_sha256"],
            inputs_sha256=input_hash, original_training_inputs_sha256=input_hash,
            evaluator_sha256=evaluator_hash, scientific_claim_qualified=False,
            formal_evaluation=False, speedup_qualified=False, episodes=30, score_rows=360,
            origins=origins, artifacts=artifacts, origin_costs=origin_costs, episode_costs=episode_costs,
            setup_seconds=1., elapsed_seconds=150., score_hash_seconds=.1,
            function_total_seconds=152., peak_process_rss_bytes=1000000,
            generated_unit_count_values=16 * 100 * 100000 * 120, cost_scope=scope,
            hardware=dict(python="fixture", numpy="fixture", torch="fixture", host="cpu04", device="cpu",
                cpu_threads=1, cpu_interop_threads=1, model_parameter_dtypes=["torch.float32"],
                model_parameter_devices=["cpu"]))
        write(cell / "forecasts/evaluation.json", evaluation)
        write(cell / "replay.json", dict(status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY",
            origins=replay_origins, checkpoint_sha256=run["best_checkpoint_sha256"],
            unit_values_consumed=16 * 100 * 100000 * 120, scientific_claim_qualified=False))
        write(cell / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
            input_family="pfc_source_reproduction", calibration_or_test_accessed=False,
            fit_episodes=80, validation_episodes=30, device="cpu", representation="recurrent",
            scientific_claim_qualified=False, preparation_sha256=sha256_file(prepared / "preparation.json")))
        seal(cell)
    return SimpleNamespace(campaign=campaign, prepared=prepared, frozen=frozen,
                           output=tmp_path / "analysis", cells={s: campaign / f"seed_{s}" for s in analysis.SEEDS})


def test_complete_three_seed_consumer_retains_signed_metrics_and_charges_costs_once(delivered, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("analysis read scientific array"))
    report = analysis.analyze(delivered.campaign, delivered.prepared, delivered.output)
    assert report["score_rows"] == 1080 and report["origin_cost_rows"] == 360
    assert report["source_episode_cost_rows"] == 30
    with (delivered.output / "all_seeds.csv").open() as stream:
        values = list(csv.DictReader(stream))
    assert len(values) == 1080 and all(float(r["unit_bin_mean_mse_corrected"]) == -.125 for r in values)
    with (delivered.output / "costs_by_origin.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    assert sum(bool(r["episode_preparation_seconds"]) for r in costs) == 90
    assert all(not report[key] for key in ("scientific_claim_qualified", "confidence_intervals_computed",
                                         "speedup_qualified", "samples_regenerated", "training_repeated"))


def test_explicit_recovered_seed_cell_preserves_failed_original(delivered):
    original = delivered.cells[41]
    recovered = delivered.campaign / "resource_attempts/seed_41_job_fixture"
    recovered.parent.mkdir()
    original.rename(recovered)
    write(original / "delivery.json", dict(status="RESOURCE_BLOCKED", exit_code=3, training_seed=41,
                                           scientific_claim_qualified=False, files=[]))
    report = analysis.analyze(delivered.campaign, delivered.prepared, delivered.output,
                              seed_cells={41: recovered})
    assert report["consumed_deliveries"]["41"]["path"] == str(recovered)
    assert json.loads((original / "delivery.json").read_text())["status"] == "RESOURCE_BLOCKED"


@pytest.mark.parametrize("change,error", [
    ("failed", "completed development-only"), ("seed", "completed development-only"),
    ("input_binding", "TRAINING_SOURCE_OR_CONFIG"), ("source", "TRAINING_SOURCE_OR_CONFIG"),
    ("budget", "REGISTERED_TRAINING_BUDGET"), ("stop", "REGISTERED_TRAINING_STOP"),
    ("missing_origin", "COMPLETE_REPLAY"), ("duplicate_score", "SCORE_IDENTITY"),
    ("map", "CROSS_SEED_SOURCE_METADATA"), ("checkpoint", "CHECKPOINT_MANIFEST"),
    ("cost", "measured origin total"), ("artifact", "REPLAYED_ARTIFACT_BINDING"),
])
def test_rejects_incomplete_or_changed_delivery_before_writes(delivered, change, error):
    cell = delivered.cells[42]
    file = "delivery.json" if change in ("failed", "seed") else (
        "training/run.json" if change in ("input_binding", "source", "budget", "stop", "checkpoint")
        else "replay.json" if change == "missing_origin" else "forecasts/evaluation.json")
    value = json.loads((cell / file).read_text())
    if change == "failed":
        value["status"] = "RESOURCE_BLOCKED"
    elif change == "seed":
        value["training_seed"] = 43
    elif change == "input_binding":
        value["binding"]["circuit_inputs_sha256"] = "foreign"
    elif change == "source":
        value["binding"]["complete_episode_bindings"][0] = {"foreign": True}
    elif change == "budget":
        value["effective_maximum_epochs"] = 21
    elif change == "stop":
        value["completed_epochs"] = 2
        value["best_epoch"] = 1
    elif change == "checkpoint":
        value["best_checkpoint_sha256"] = "foreign"
    elif change == "missing_origin":
        value["origins"].pop()
    elif change == "map":
        value["origins"][0]["source_metadata_identity"] = "foreign map"
    elif change == "cost":
        value["origin_costs"][0]["origin_total_seconds"] = 0
    elif change == "artifact":
        value["artifacts"][value["origins"][0]["artifact"]] = "foreign"
    else:
        file = "forecasts/scores.csv"
        with (cell / file).open() as stream:
            table = list(csv.DictReader(stream))
        rows(cell / file, table + [table[0]])
        value = json.loads((cell / "forecasts/evaluation.json").read_text())
        value["artifacts"]["scores.csv"] = sha256_file(cell / file)
        file = "forecasts/evaluation.json"
    write(cell / file, value)
    seal(cell)
    with pytest.raises(ValueError, match=error):
        analysis.analyze(delivered.campaign, delivered.prepared, delivered.output)
    assert not delivered.output.exists()


def test_undefined_metric_summary_keeps_missing_count():
    row = dict(seed=41, **{name: -.1 if name == "unit_bin_mean_mse_corrected" else None
                         for name in analysis.METRICS})
    result = analysis.summarize([row], ("seed",))[0]
    assert result["unit_bin_mean_mse_corrected_mean"] == -.1
    assert result["refractory_count_violation_2bin_defined"] == 0
    assert result["refractory_count_violation_2bin_mean"] is None

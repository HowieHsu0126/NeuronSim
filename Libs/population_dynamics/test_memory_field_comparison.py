"""Common field metric units, scales and complete comparison denominator."""
import numpy as np
import pytest

from .memory_field_comparison import keys, standardized_error


def test_rate_metric_matches_source_grouped_count_metric():
    units = np.array([2, 50, 200])
    sd = np.array([1., 2., 4.])
    truth = np.array([[2., 3., 4.], [8., 2., 10.]])
    prediction = truth + np.array([[1., 2., 3.], [0., 1., 4.]])
    grouped_prediction = prediction * units * .005
    grouped_truth = truth * units * .005
    expected = np.mean(((grouped_prediction - grouped_truth) / (units * .005 * sd)) ** 2)
    assert standardized_error(prediction, truth, sd) == pytest.approx(expected, rel=1e-14)


def test_invalid_scales_or_partial_fields_rejected():
    for prediction, truth, sd in ((np.ones((2, 3)), np.ones((2, 2)), np.ones(3)),
            (np.ones((2, 3)), np.ones((2, 3)), np.array([1., 0., 1.])),
            (np.full((2, 3), np.nan), np.ones((2, 3)), np.ones(3))):
        with pytest.raises(ValueError):
            standardized_error(prediction, truth, sd)


def test_complete_registered_condition_denominator():
    registered = keys()
    assert len(registered) == 432
    for seed in (41, 42, 43):
        for block in range(14, 18):
            for horizon in (20., 80., 500.):
                rows = {row for row in registered if row[0] == seed and row[1] == block and row[4] == horizon}
                assert len(rows) == 12
                assert {row[3] for row in rows} == {1000., 3000., 5000.}

@pytest.mark.parametrize("profile", ["ordinary", "analytic_prior", "predictive_mixture"])
def test_memno_dedicated_prior_handoff_preserves_field_scope_and_parent_identity(tmp_path, profile):
    import csv
    import copy
    import itertools
    import json
    from pathlib import Path
    from .braid_population_comparison import compare as compare_braid
    from .memory_field_comparison import compare
    from .provenance import sha256_file
    from .test_braid_population_comparison import fixture, prior_population_fixture, delivery, write_json
    from .population_forecast_analysis import PRIOR_SOURCE_BINDINGS
    fixture(tmp_path)
    population = tmp_path / "population" if profile == "ordinary" else prior_population_fixture(tmp_path)
    if profile == "predictive_mixture":
        from .test_population_forecast_analysis import _set_prior_fixture_profile
        for seed in (41, 42, 43):
            _set_prior_fixture_profile(population / f"seed_{seed}", mixture=True)
    braid = tmp_path / "new_braid_comparison"
    compare_braid(tmp_path / "braid", population, braid / "comparison")

    def deliver_braid():
        files = [dict(path=str(path.relative_to(braid)), bytes=path.stat().st_size, sha256=sha256_file(path))
                 for path in (braid / "comparison").iterdir() if path.is_file()]
        write_json(braid / "delivery.json", dict(status="PASS", exit_code=0, files=files))

    deliver_braid()
    memory = tmp_path / "memory"
    for seed in (41, 42, 43):
        pop = population / f"seed_{seed}"
        run = json.loads((pop / "training/run.json").read_text())
        model = run["model_configuration"]
        source = dict(field_units="spikes_per_second_per_neuron", bin_seconds=.005,
            cohort={key: run["binding"][key] for key in PRIOR_SOURCE_BINDINGS},
            source_unit_ids=model["unit_ids"], source_field_ids=model["field_names"])
        binding = dict(dataset=source, config=dict(fit_blocks=list(range(14)), validation_blocks=list(range(14, 18))))
        cell = memory / f"seed_{seed}"
        write_json(cell / "training/run.json", dict(status="FITTED_DEVELOPMENT", completed_epochs=300,
                                                    binding=binding, field_mean_only=True))
        artifacts = {}
        for block, template, origin in itertools.product(range(14, 18), range(4), (1000., 3000., 5000.)):
            case = f"train_block_{block:03d}_familiar_{template}"
            with np.load(pop / f"forecasts/{case}_origin_{origin:g}.npz", allow_pickle=False) as values:
                truth = (values["observed_field_counts"].astype(np.float64) / 250.).astype(np.float32)
            path = cell / f"evaluation/{case}_origin_{int(origin / 5):04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, predicted_rate_hz=truth + .1, truth_rate_hz=truth,
                source_field_ids=np.asarray(model["field_names"]), source_unit_ids=np.asarray(model["unit_ids"]),
                bin_seconds=.005, origin_bin=int(origin / 5))
            artifacts[path.name] = sha256_file(path)
        write_json(cell / "evaluation/evaluation.json", dict(status="EVALUATED_DEVELOPMENT", score_rows=144,
            origins=48, partial_training_diagnostic=False, binding=binding, artifacts=artifacts))
        write_json(cell / "replay/evaluation.json", dict(status="REPLAYED_DEVELOPMENT", score_rows=144, binding=binding))
        delivery(cell, seed)
        proof = json.loads((cell / "delivery.json").read_text())
        proof.update(complete_training=True, replay_performed=True)
        write_json(cell / "delivery.json", proof)
    output = tmp_path / "memno_prior"
    compare(memory, braid, population, output)
    result = json.loads((output / "comparison.json").read_text())
    assert result["score_rows"] == 432 and result["block_rows"] == 36
    assert result["full_unit_speedup_qualified"] is False and result["inference_performed"] is False
    assert result["scientific_claim_qualified"] is False
    with (output / "field_comparison_all_seeds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 432
    if profile == "ordinary":
        assert "candidate_method" not in result and "population_training_bindings" not in result
        assert all("candidate_method" not in row for row in rows)
        return
    method = ("population_recurrent_predictive_mixture" if profile == "predictive_mixture"
              else "population_recurrent_prior_refit")
    assert result["candidate_method"] == method
    assert all(row["candidate_method"] == method for row in rows)
    assert result["population_training_bindings"]["43"]["prior_refit_lineage"]["parent_completed_epochs"] == 80
    if profile == "predictive_mixture":
        assert all(binding["prediction_particles"] == 8 and binding["posterior_target_draws"] == 4
                   for binding in result["population_training_bindings"].values())
    prior_report_path = braid / "comparison/comparison.json"
    original_report = json.loads(prior_report_path.read_text())
    for failure in ("parent_checkpoint", "unknown_method", "wrong_objective", "wrong_selection", "report_method"):
        prior_report = copy.deepcopy(original_report)
        binding = prior_report["population_training_bindings"]["41"]
        if failure == "parent_checkpoint":
            binding["prior_refit_lineage"]["parent_checkpoint_sha256"] = "foreign"
        elif failure == "unknown_method":
            binding["method"] = "population_recurrent_unregistered"
        elif failure == "wrong_objective":
            binding["training_objective"] = "foreign_objective"
        elif failure == "wrong_selection":
            binding["selection_metric"] = "foreign_selection"
        else:
            prior_report["candidate_method"] = "population_recurrent_unregistered"
        write_json(prior_report_path, prior_report)
        deliver_braid()
        rejected = tmp_path / failure
        with pytest.raises(ValueError, match="prior parent lineage differs|same prior objective"):
            compare(memory, braid, population, rejected)
        assert not rejected.exists()


def test_native_field_scores_use_existing_corrected_metric_without_clipping():
    from .memory_field_comparison import native_field_scores
    from .lif_neural_field import sampled_mean_field_mse
    units = np.full(128, 781)
    units[0] += 100000 - units.sum()
    sd = np.ones(128)
    truth = np.full((100, 128), 3, dtype=np.int64)
    samples = np.stack([truth + (index % 3) for index in range(16)])
    scores = native_field_scores(samples, truth, sd, units)
    for horizon, actual in scores.items():
        n = int(horizon / 5)
        expected = sampled_mean_field_mse(samples[:, :n], truth[:n], sd, units, 5.)
        assert actual == expected
    with pytest.raises(ValueError, match="sixteen-draw"):
        native_field_scores(samples[:8], truth, sd, units)


def test_native_and_legacy_cli_select_distinct_readers(monkeypatch, capsys):
    from . import memory_field_comparison as module
    import sys
    called = []
    monkeypatch.setattr(module, "compare", lambda *args: called.append("legacy") or dict(score_rows=432, block_rows=36))
    monkeypatch.setattr(module, "compare_native", lambda *args, **kwargs: called.append("native") or dict(score_rows=864, block_rows=72))
    monkeypatch.setattr(module, "cpu_execution_provenance", lambda: {"backend": "merlin"})
    monkeypatch.setattr(sys, "argv", ["reader", "--memory-campaign", "memory", "--braid-comparison", "braid",
        "--population-campaign", "population", "--output", "out"])
    module.main()
    monkeypatch.setattr(sys, "argv", ["reader", "--native-corrections", "corrections", "--parent-campaign", "parent",
        "--baseline-population-campaign", "old-parent", "--braid-comparison", "braid", "--memory-comparison", "memory",
        "--output", "out"])
    module.main()
    assert called == ["legacy", "native"]


# Reuse the full native refit/evaluation/replay fixtures. No intake function is mocked.
from .test_population_residual_correction_analysis import (
    evidence, native_evidence, refresh as _refresh_native_receipts,
    write_json as _write_native_json, write_rows as _write_native_rows,
    read_rows as _read_native_rows,
)
from .provenance import sha256_file as _native_sha256


def _bind_native_field_files(cell, *names):
    import json
    proof = json.loads((cell / "delivery.json").read_text())
    index = {row["path"]: row for row in proof["files"]}
    for name in names:
        path = cell / name
        index[name] = dict(path=name, bytes=path.stat().st_size, sha256=_native_sha256(path))
    proof.update(scientific_claim_qualified=False, files=list(index.values()))
    _write_native_json(cell / "delivery.json", proof)


def _seal_native_field_baselines(data):
    """Reseal intentional fixture changes so semantic gates, not stale hashes, decide."""
    import copy
    braid, memory = data["braid"], data["memory"]
    b_report, m_report = data["braid_report"], data["memory_report"]
    b_report["consumed_deliveries"] = {}
    m_report["bindings"] = {}
    for seed in (41, 42, 43):
        cell = data["old_parent"] / f"seed_{seed}"
        binding = dict(delivery_sha256=_native_sha256(cell / "delivery.json"),
            consumed_files={name: _native_sha256(cell / name)
                for name in ("training/run.json", "forecasts/fitting_field_scale.npz")})
        b_report["consumed_deliveries"][str(seed)] = dict(population=copy.deepcopy(binding))
        m_report["bindings"][str(seed)] = dict(population=copy.deepcopy(binding))
    table = "mean_comparison_all_seeds.csv"
    b_report["artifacts"] = {table: _native_sha256(braid / "comparison" / table)}
    _write_native_json(braid / "comparison/comparison.json", b_report)

    def seal(root, table_name):
        names = ("comparison/comparison.json", "comparison/" + table_name)
        _write_native_json(root / "delivery.json", dict(status="PASS", exit_code=0,
            files=[dict(path=name, bytes=(root / name).stat().st_size, sha256=_native_sha256(root / name))
                   for name in names]))

    seal(braid, table)
    m_report["prior_delivery_sha256"] = _native_sha256(braid / "delivery.json")
    _write_native_json(memory / "comparison/comparison.json", m_report)
    seal(memory, "field_comparison_all_seeds.csv")


@pytest.fixture
def native_field_evidence(native_evidence, tmp_path):
    import copy
    import json
    import shutil
    corrections, parents, _ = native_evidence
    data = dict(corrections=corrections, parents=parents, old_parent=tmp_path / "old_parent",
        braid=tmp_path / "braid_comparison", memory=tmp_path / "memory_comparison",
        output=tmp_path / "native_fields")
    units = np.full(128, 781, dtype=np.int64)
    units[0] += 100000 - units.sum()
    ids = np.asarray([f"field_{index:03d}" for index in range(128)])
    mean = np.linspace(.2, .4, 128)
    sd = np.linspace(1., 2., 128)
    model = dict(unit_ids=list(range(100000)),
        unit_field_index=np.repeat(np.arange(128), units).tolist(),
        field_names=ids.tolist(), minimum_interval_ticks=[2] * 100000,
        bin_seconds=.005, tick_seconds=.0001)
    scale = dict(mean_hz=mean, standard_deviation_hz=sd, unit_count=units, field_ids=ids)
    truth = np.ones((100, 128), dtype=np.int64)
    samples = {}
    templates = tmp_path / "field_templates"
    templates.mkdir()
    for mode in ("current", "history"):
        sample = np.full((16, 100, 128), 2, dtype=np.int64)
        if mode == "current":
            sample[::2] = 0
        samples[mode] = sample
        # Actual origin writer keys: there is deliberately no field_ids/unit_count.
        np.savez_compressed(templates / f"{mode}.npz",
            observed_field_counts=truth, sampled_field_counts=sample,
            latent_states=np.zeros((16, 100, 128, 4), dtype=np.float32),
            initial_latent_states=np.zeros((16, 128, 4), dtype=np.float32),
            initial_memory=np.zeros((128, 4), dtype=np.float32))
    for seed in (41, 42, 43):
        parent, old, corrected = (parents / f"seed_{seed}", data["old_parent"] / f"seed_{seed}",
                                  corrections / f"seed_{seed}")
        evaluation = json.loads((parent / "forecasts/evaluation.json").read_text())
        binding = {name: copy.deepcopy(evaluation[name]) for name in
            ("acquisition_design_sha256", "dataset_sha256", "native_source_identity")}
        binding.update(acquisition_ledger_sha256="1" * 64, episode_table_sha256="2" * 64,
            episode_bindings={f"train_block_{block:03d}_familiar_{template}": {}
                for block in range(18) for template in range(4)})
        run = dict(status="FITTED_DEVELOPMENT", training_seed=seed, binding=binding,
            model_configuration=model, best_checkpoint_sha256=evaluation["checkpoint_sha256"])
        _write_native_json(parent / "training/run.json", run)
        _bind_native_field_files(parent, "training/run.json")
        old_run = copy.deepcopy(run)
        old_run["best_checkpoint_sha256"] = f"{seed + 1000:064x}"
        _write_native_json(old / "training/run.json", old_run)
        (old / "forecasts").mkdir()
        np.savez(old / "forecasts/fitting_field_scale.npz", **scale)
        _write_native_json(old / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
            scientific_claim_qualified=False, files=[dict(path="training/best.pt", bytes=10**9,
                sha256=old_run["best_checkpoint_sha256"])]))
        _bind_native_field_files(old, "training/run.json", "forecasts/fitting_field_scale.npz")
        for mode in ("current", "history"):
            base = corrected / f"results/{mode}/forecasts"
            # Different NPZ representation, identical four scientific arrays.
            np.savez_compressed(base / "fitting_field_scale.npz",
                **dict(reversed(list(scale.items()))))
            names = [f"results/{mode}/forecasts/fitting_field_scale.npz"]
            evaluation_path = base / "evaluation.json"
            current = json.loads(evaluation_path.read_text())
            current["artifacts"] = {}
            for _, block, case, origin, _ in sorted(key for key in keys() if key[0] == seed and key[-1] == 20.):
                name = f"{case}_origin_{origin:g}.npz"
                shutil.copyfile(templates / f"{mode}.npz", base / name)
                current["artifacts"][name] = _native_sha256(base / name)
                names.append(f"results/{mode}/forecasts/{name}")
            _write_native_json(evaluation_path, current)
            _bind_native_field_files(corrected, *names)
            _refresh_native_receipts(corrected, mode)
    b_rows, m_rows = [], []
    for seed, block, case, origin, horizon in sorted(keys()):
        identity = dict(seed=seed, block=block, case_id=case, origin_ms=origin, horizon_ms=horizon)
        b_rows.append(dict(identity, braid_field_mean_mse=.5, population_field_mean_mse_corrected=12345.))
        m_rows.append(dict(identity, braid_field_mean_mse=.5, memno_field_mean_mse=.25,
                           population_field_mean_mse_corrected=54321.))
    _write_native_rows(data["braid"] / "comparison/mean_comparison_all_seeds.csv", b_rows)
    _write_native_rows(data["memory"] / "comparison/field_comparison_all_seeds.csv", m_rows)
    data["braid_report"] = dict(status="COMPLETE_DESCRIPTIVE_BRAID_POPULATION_MEAN_COMPARISON",
        score_rows=432, training_seeds=[41, 42, 43], scientific_claim_qualified=False,
        full_end_to_end_cost_matched=False, speedup_qualified=False)
    data["memory_report"] = dict(status="COMPLETE_DESCRIPTIVE_FIELD_MEAN_COMPARISON",
        score_rows=432, block_rows=36, scientific_claim_qualified=False, same_training_budget=False,
        full_unit_speedup_qualified=False, source_circuits=1)
    _seal_native_field_baselines(data)
    data.update(samples=samples, truth=truth, units=units, sd=sd)
    return data


def _compare_native_field_fixture(data):
    from .memory_field_comparison import compare_native
    return compare_native(data["corrections"], data["parents"], data["old_parent"],
        data["braid"], data["memory"], data["output"], execution=dict(backend="merlin", run_id="fixture"))


def test_native_field_complete_864_rows_and_72_blocks_use_new_arrays(native_field_evidence):
    import json
    from .lif_neural_field import sampled_mean_field_mse
    data = native_field_evidence
    report = _compare_native_field_fixture(data)
    assert (report["score_rows"], report["block_rows"]) == (864, 72)
    rows = _read_native_rows(data["output"] / "field_comparison_all_seeds.csv")
    blocks = _read_native_rows(data["output"] / "field_comparison_by_block.csv")
    assert len(rows) == 864 and len(blocks) == 72
    for mode in ("current", "history"):
        selected = [row for row in rows if row["candidate"] == mode]
        assert len(selected) == 432
        assert {(int(row["seed"]), int(row["block"]), row["case_id"], float(row["origin_ms"]),
                 float(row["horizon_ms"])) for row in selected} == keys()
        for row in selected:
            n = int(float(row["horizon_ms"]) / 5)
            expected = sampled_mean_field_mse(data["samples"][mode][:, :n], data["truth"][:n],
                                              data["sd"], data["units"], 5.)
            assert float(row["candidate_field_mean_mse_corrected"]) == pytest.approx(expected, rel=1e-14)
            assert float(row["braid_field_mean_mse"]) == .5
            assert float(row["memno_field_mean_mse"]) == .25
            assert row["candidate_method"] == "population_residual_" + mode
            assert "population_field_mean_mse_corrected" not in row
    assert all(float(row["candidate_field_mean_mse_corrected"]) < 0 for row in rows if row["candidate"] == "current")
    assert all(int(row["matched_conditions"]) == 12 for row in blocks)
    for seed in ("41", "42", "43"):
        scale = report["fit_scale_bindings"][seed]
        assert scale["baseline_sha256"] != scale["current_sha256"]
        old = json.loads((data["old_parent"] / f"seed_{seed}/training/run.json").read_text())
        new = json.loads((data["parents"] / f"seed_{seed}/training/run.json").read_text())
        assert old["best_checkpoint_sha256"] != new["best_checkpoint_sha256"]
    assert report["old_population_predictions_reused"] is False
    assert report["old_population_costs_reused"] is False
    assert report["baseline_reports"]["memno"]["same_training_budget"] is False
    assert report["scientific_claim_qualified"] is False
    assert report["full_end_to_end_cost_matched"] is False
    assert report["speedup_qualified"] is False


@pytest.mark.parametrize("baseline", ["braid", "memory"])
def test_native_field_rejects_changed_baseline_hash(native_field_evidence, baseline):
    data = native_field_evidence
    table = "mean_comparison_all_seeds.csv" if baseline == "braid" else "field_comparison_all_seeds.csv"
    path = data[baseline] / "comparison" / table
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        _compare_native_field_fixture(data)
    assert not data["output"].exists()


@pytest.mark.parametrize("change", ["source", "field_order", "unit_map", "clock", "refractory"])
def test_native_field_rejects_resealed_source_or_map_change(native_field_evidence, change):
    import json
    data = native_field_evidence
    cell = data["old_parent"] / "seed_41"
    path = cell / "training/run.json"
    run = json.loads(path.read_text())
    if change == "source":
        run["binding"]["dataset_sha256"] = "0" * 64
    elif change == "field_order":
        run["model_configuration"]["field_names"].reverse()
    elif change == "unit_map":
        run["model_configuration"]["unit_field_index"][0] = 1
    elif change == "clock":
        run["model_configuration"]["bin_seconds"] = .01
    else:
        run["model_configuration"]["minimum_interval_ticks"][0] += 1
    _write_native_json(path, run)
    _bind_native_field_files(cell, "training/run.json")
    _seal_native_field_baselines(data)
    with pytest.raises(ValueError, match="source cohort|observation map"):
        _compare_native_field_fixture(data)
    assert not data["output"].exists()


@pytest.mark.parametrize("array", ["mean_hz", "standard_deviation_hz", "unit_count", "field_ids"])
def test_native_field_rejects_resealed_scale_change(native_field_evidence, array):
    data = native_field_evidence
    cell = data["corrections"] / "seed_41"
    relative = "results/current/forecasts/fitting_field_scale.npz"
    path = cell / relative
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name].copy() for name in saved.files}
    if array in ("mean_hz", "standard_deviation_hz"):
        values[array][0] = np.nextafter(values[array][0], np.inf)
    elif array == "unit_count":
        values[array][0] -= 1
        values[array][1] += 1
    else:
        values[array] = values[array][::-1]
    np.savez_compressed(path, **values)
    _bind_native_field_files(cell, relative)
    with pytest.raises(ValueError, match="fit scale|fitting scale"):
        _compare_native_field_fixture(data)
    assert not data["output"].exists()


@pytest.mark.parametrize("correct", [True, False])
def test_native_field_honors_explicit_evaluation_scale_binding(native_field_evidence, correct):
    import json
    data = native_field_evidence
    cell = data["corrections"] / "seed_41"
    path = cell / "results/current/forecasts/evaluation.json"
    evaluation = json.loads(path.read_text())
    scale = cell / "results/current/forecasts/fitting_field_scale.npz"
    evaluation["artifacts"]["fitting_field_scale.npz"] = _native_sha256(scale) if correct else "0" * 64
    _write_native_json(path, evaluation)
    _refresh_native_receipts(cell, "current")
    if correct:
        assert _compare_native_field_fixture(data)["score_rows"] == 864
    else:
        with pytest.raises(ValueError, match="evaluated fitting scale"):
            _compare_native_field_fixture(data)
        assert not data["output"].exists()


@pytest.mark.parametrize("missing", ["baseline_condition", "native_condition", "native_artifact", "artifact_binding"])
def test_native_field_rejects_missing_condition_or_artifact(native_field_evidence, missing):
    import json
    data = native_field_evidence
    cell = data["corrections"] / "seed_41"
    if missing == "baseline_condition":
        path = data["braid"] / "comparison/mean_comparison_all_seeds.csv"
        _write_native_rows(path, _read_native_rows(path)[:-1])
        _seal_native_field_baselines(data)
    elif missing == "native_condition":
        path = cell / "results/current/forecasts/scores.csv"
        _write_native_rows(path, _read_native_rows(path)[:-1])
        _refresh_native_receipts(cell, "current")
    elif missing == "native_artifact":
        (cell / "results/current/forecasts/train_block_014_familiar_0_origin_1000.npz").unlink()
    else:
        path = cell / "results/current/forecasts/evaluation.json"
        evaluation = json.loads(path.read_text())
        evaluation["artifacts"].pop("train_block_014_familiar_0_origin_1000.npz")
        _write_native_json(path, evaluation)
        _refresh_native_receipts(cell, "current")
    with pytest.raises((ValueError, FileNotFoundError)):
        _compare_native_field_fixture(data)
    assert not data["output"].exists()


@pytest.mark.parametrize("which", ["samples", "truth"])
def test_native_field_rejects_fractional_count_arrays(which):
    from .memory_field_comparison import native_field_scores
    samples = np.ones((16, 100, 128), dtype=np.int64)
    truth = np.ones((100, 128), dtype=np.int64)
    if which == "samples":
        samples = samples.astype(np.float64)
        samples[0, 0, 0] = 1.5
    else:
        truth = truth.astype(np.float64)
        truth[0, 0] = 1.5
    with pytest.raises(ValueError):
        native_field_scores(samples, truth, np.ones(128), np.full(128, 1))

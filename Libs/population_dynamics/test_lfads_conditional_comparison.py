"""Paired-path, weighted-error and prospective block-routing checks."""
from copy import deepcopy
import random

import numpy as np
import pytest

from .lfads_conditional_comparison import (
    ARMS, BLOCKS, CANDIDATE, GUARDS, ORIGINS, PRIMARY, REFERENCE,
    compare_per_block, conditional_metrics, verify_paired_paths,
)


@pytest.fixture
def origin_rows():
    return [
        dict(arm=arm, block=block, case_id=f"train_block_{block:03d}_familiar_{template}",
             origin_ms=origin, horizon_ms=500,
             **{metric: 3. if arm == CANDIDATE else 4. for metric in PRIMARY},
             **{metric: 1. for metric in GUARDS})
        for arm in ARMS for block in BLOCKS for template in range(4) for origin in ORIGINS
    ]


def test_routing_keeps_arm_names_and_is_order_independent(origin_rows):
    expected = compare_per_block(origin_rows)
    assert expected["route"] == "PASS"
    assert expected["reference_arm"] == REFERENCE
    assert expected["candidate_arm"] == CANDIDATE
    random.Random(41).shuffle(origin_rows)
    assert compare_per_block(origin_rows) == expected


@pytest.mark.parametrize("metric", PRIMARY + GUARDS)
def test_one_failed_block_stops_even_when_all_other_blocks_improve(origin_rows, metric):
    for row in origin_rows:
        if row["arm"] == CANDIDATE and row["block"] == 14:
            row[metric] = 4. if metric in PRIMARY else 2.
    result = compare_per_block(origin_rows)
    assert result["route"] == "STOP"
    assert result["blocks"][0]["metric_pass"][metric] is False
    assert all(block["route"] == "PASS" for block in result["blocks"][1:])


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "nonfinite", "horizon"])
def test_bad_routing_inputs_are_rejected(origin_rows, change):
    if change == "missing": origin_rows.pop()
    elif change == "duplicate": origin_rows.append(origin_rows[0].copy())
    elif change == "identity": origin_rows[0]["case_id"] = "unregistered"
    elif change == "nonfinite": origin_rows[0][PRIMARY[0]] = np.nan
    else: origin_rows[0]["horizon_ms"] = 20
    with pytest.raises(ValueError): compare_per_block(origin_rows)


def test_routing_uses_empirical_error_even_when_mc_correction_reverses_order(origin_rows):
    for row in origin_rows:
        row["conditional_mean_mse_corrected_hz2"] = -100. if row["arm"] == REFERENCE else 10.
    assert compare_per_block(origin_rows)["route"] == "PASS"


def test_draw_average_member_weighting_and_absolute_before_block_average():
    # Field sizes differ. Draw average must precede the square.
    units = np.array([1, 3], np.int64)
    expected = np.array([[[0., 0.], [0., 0.]], [[4., 12.], [0., 0.]]])
    truth = np.zeros((2, 2))
    values, fields = conditional_metrics(expected, expected, truth, units, 1.)
    assert values["conditional_empirical_mean_mse_hz2"] == 2.
    assert values["absolute_global_mean_rate_bias_hz"] == 1.
    unequal = np.array([[[2., 0.]], [[2., 0.]]])
    weighted, _ = conditional_metrics(unequal, unequal, np.zeros((1, 2)), units, 1.)
    assert weighted["conditional_empirical_mean_mse_hz2"] == 1.
    assert weighted["absolute_global_mean_rate_bias_hz"] == .5
    opposite, _ = conditional_metrics(np.zeros_like(unequal), np.zeros_like(unequal),
                                     np.array([[2., 0.]]), units, 1.)
    assert opposite["conditional_signed_time_mean_bias_hz"] == -.5
    assert (weighted["absolute_global_mean_rate_bias_hz"] + opposite["absolute_global_mean_rate_bias_hz"]) / 2 == .5
    assert (weighted["conditional_signed_time_mean_bias_hz"] + opposite["conditional_signed_time_mean_bias_hz"]) / 2 == 0.


def test_pairing_requires_bytes_dtype_shape_and_all_causal_states():
    ref = {name: np.arange(8, dtype=np.float32).reshape(2, 4) for name in (
        "observed_field_counts", "lfads_initial_latent_states", "lfads_historical_controller_inputs",
        "lfads_boundary_generator_states", "lfads_future_controller_inputs")}
    candidate = deepcopy(ref)
    candidate["lfads_base_future_controller_inputs"] = candidate.pop("lfads_future_controller_inputs")
    verify_paired_paths(ref, candidate)
    for name in candidate:
        changed = deepcopy(candidate)
        changed[name][0, 0] = .1
        with pytest.raises(ValueError, match="PAIRED_CAUSAL_PATH_CHANGED"):
            verify_paired_paths(ref, changed)
    changed = deepcopy(candidate)
    changed["lfads_base_future_controller_inputs"] = changed["lfads_base_future_controller_inputs"].astype(np.float64)
    with pytest.raises(ValueError): verify_paired_paths(ref, changed)
    changed = deepcopy(candidate)
    changed["lfads_boundary_generator_states"][0, 0] = -0.
    with pytest.raises(ValueError): verify_paired_paths(ref, changed)


@pytest.mark.parametrize("seed,budget,reuse", [(41, 20, True), (42, 80, False)])
def test_comparison_entry_preserves_all_origins_and_serializes_explicit_arms(tmp_path, monkeypatch, seed, budget, reuse):
    import csv
    import json
    from types import SimpleNamespace
    from pathlib import Path
    import yaml
    from . import lfads_conditional_comparison as module

    config = dict(origins_ms=list(ORIGINS), horizons_ms=[20., 80., 500.])
    protocol = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_lfads_conditional_prior.yaml").read_text())
    units = np.array([1, 3], np.int64)
    scale = dict(unit_count=units, field_ids=np.array(["small", "large"]),
                 mean_hz=np.zeros(2), standard_deviation_hz=np.ones(2))
    selected = 20 if budget == 20 else 54
    prior = dict(fitted_parameters=8, fitting_episodes=56, training_seed=seed, selected_epoch=selected)
    binding = dict(comparison_protocol=protocol, additional_fitted_scalars=19200)
    left = dict(training_binding={"config": {"bin_ms": 5.}}, training_run_sha256="run",
                selected_checkpoint="same", selected_training_sha256="same",
                selected_model_artifacts={"weights.npz": "frozen", "model.json": "frozen_model"},
                selected_epoch=selected-1, effective_training_epoch_budget=budget, native_source_identity={},
                configuration=config, inference_migration=None, prior_refit=prior)
    right = dict(left, conditional_prior=binding)
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for folder in (reference, candidate):
        (folder / "forecasts").mkdir(parents=True)
    indexed, archived = {}, {}
    samples = np.zeros((16, 100, 2))
    truth = np.zeros((100, 2))
    future = np.ones((16, 100, 4), np.float32)
    expected = np.broadcast_to(np.array([.01, .03]), samples.shape).copy()
    for block in BLOCKS:
        for template in range(4):
            case = f"train_block_{block:03d}_familiar_{template}"
            for origin in ORIGINS:
                common = dict(observed_field_counts=truth, sampled_field_counts=samples,
                    lfads_initial_latent_states=np.zeros((16, 2), np.float32),
                    lfads_historical_controller_inputs=np.zeros((16, 2, 4), np.float32),
                    lfads_boundary_generator_states=np.zeros((16, 3), np.float32),
                    lfads_future_generator_states=np.zeros((16, 100, 3), np.float32))
                name = f"{case}_origin_{origin:g}.npz"
                np.savez(reference / "forecasts" / name, **dict(common,
                    lfads_future_generator_states=np.ones((16, 100, 3), np.float32)), lfads_future_controller_inputs=future)
                np.savez(candidate / "forecasts" / name, **common, lfads_future_controller_inputs=future + 1,
                         lfads_base_future_controller_inputs=future)
                for horizon in config["horizons_ms"]:
                    key = (case, origin, horizon)
                    indexed[key] = {metric: "1.0" for metric in GUARDS}
                    bins = int(horizon / 5)
                    archived[key], _ = conditional_metrics(samples[:, :bins], expected[:, :bins] * 2,
                                                           truth[:bins], units, .005)
    def admitted(folder, replay, seed):
        if folder.parent == reference:
            return left, {"evaluation_sha256": "reference"}, scale, indexed
        return right, {"evaluation_sha256": "candidate", "conditional_prior": binding}, scale, indexed
    monkeypatch.setattr(module, "_replayed_forecast", admitted)
    monkeypatch.setattr(module, "_reference_metrics", lambda *a: (archived, {"table_sha256": "reference"}))
    mean_calls = []
    def mean_loader(*args, **kwargs):
        mean_calls.append(kwargs)
        return None, binding
    monkeypatch.setattr(module, "load_mean_prior", mean_loader)
    from unittest.mock import Mock
    readout = SimpleNamespace(expected_field_counts=Mock(side_effect=lambda states, *a:
        (expected * (2 if np.any(states) else 1), None)))
    monkeypatch.setattr(module, "_readout_inputs", lambda *a: (readout, np.array([0, 1, 1, 1])))
    output = tmp_path / "result"
    (tmp_path / "mean_prior.json").write_text(json.dumps(dict(training_seed=seed)))
    result = module.compare(reference, candidate, tmp_path if reuse else None, tmp_path, tmp_path, tmp_path, output)
    assert result["route"] == "PASS" and result["paired_origins"] == 48
    assert result["scientific_claim_qualified"] is False
    with (output / "conditional_by_origin.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 288 and {row["arm"] for row in rows} == set(ARMS)
    saved = json.loads((output / "analysis.json").read_text())
    assert saved["route"] == "PASS" and len(saved["blocks"]) == 4

    assert result["training_seed"] == seed and result["network_budget_epochs"] == budget
    assert result["reference_readout_reconstructed"] is (not reuse)
    assert mean_calls == [dict(training_seed=seed, network_budget_epochs=budget,
                               selected_model_artifacts=left["selected_model_artifacts"])]
    assert readout.expected_field_counts.call_count == (48 if reuse else 96)
    assert (output / "reference_field_errors.csv").exists() is (not reuse)
    for fault in ("seed", "budget", "checkpoint"):
        saved_prior = dict(prior)
        saved_right_budget = right["effective_training_epoch_budget"]
        saved_epoch = left["selected_epoch"]
        if fault == "seed": prior["training_seed"] = 43
        elif fault == "budget": right["effective_training_epoch_budget"] = 100-budget
        else: left["selected_epoch"] = right["selected_epoch"] = selected-2
        with pytest.raises(ValueError, match="REGISTERED_SEED_BUDGET_CHECKPOINT"):
            module.compare(reference, candidate, tmp_path if reuse else None, tmp_path, tmp_path, tmp_path,
                           tmp_path / ("rejected_" + fault))
        prior.clear(); prior.update(saved_prior)
        right["effective_training_epoch_budget"] = saved_right_budget
        left["selected_epoch"] = right["selected_epoch"] = saved_epoch


def test_missing_base_paired_path_is_rejected():
    ref = {name: np.zeros((2, 4), np.float32) for name in (
        "observed_field_counts", "lfads_initial_latent_states", "lfads_historical_controller_inputs",
        "lfads_boundary_generator_states", "lfads_future_controller_inputs")}
    candidate = deepcopy(ref)
    candidate.pop("lfads_future_controller_inputs")
    with pytest.raises((KeyError, ValueError)):
        verify_paired_paths(ref, candidate)


@pytest.mark.parametrize("mode", ["conditional_mean_matrix", "conditional_mean_comparison"])
def test_wrapper_accepts_only_completed_cpu_software(mode):
    from Libs.scripts.run_lfads_conditional_comparison import accepted_software
    receipt = dict(status="ARCHIVED_STATIONARY_PRIOR_SOFTWARE_TESTS", exit_code=0,
                   validation_mode=mode, requested_test_device="cpu")
    assert accepted_software(receipt)
    assert accepted_software(dict(status="PASS", exit_code=0))
    for changed in (dict(receipt, exit_code=1), dict(receipt, requested_test_device="cuda"),
                    dict(receipt, validation_mode="unrelated")):
        assert not accepted_software(changed)


def test_wrapper_reference_layouts_preserve_old_and_stationary_prefixes():
    from Libs.scripts.run_lfads_conditional_comparison import reference_prefix
    assert reference_prefix(dict(status="COMPARED_PRIORS_DEVELOPMENT")) == "fitted_prior"
    assert reference_prefix(dict(status="REPLAYED_STATIONARY_PRIOR_CANDIDATE", exit_code=0)) == "candidate"
    with pytest.raises(ValueError):
        reference_prefix(dict(status="REPLAYED_STATIONARY_PRIOR_CANDIDATE", exit_code=1))


@pytest.mark.parametrize("fault", [None, "parent", "seed", "budget", "weights", "missing_map", "summary_hash"])
def test_wrapper_stages_only_selected_checkpoint_and_field_map(tmp_path, monkeypatch, fault):
    import json
    from Libs.scripts import run_lfads_conditional_comparison as wrapper
    from Libs.scripts import run_lfads_conditional_forecast as forecast
    fit = dict(training_seed=42, network_variant="budget_80", files=[
        dict(path="cache/posterior_cache.json", sha256="cache"),
        dict(path="cache/prior_parameters.npz", sha256="prior")])
    (tmp_path / "delivery.json").write_text(json.dumps(fit))
    reference = dict(reference_delivery_sha256=wrapper.digest(tmp_path / "delivery.json"))
    evaluation = dict(training_seed=42, effective_training_epoch_budget=80,
        selected_checkpoint="checkpoints/epoch_0054_attempt_0000",
        selected_model_artifacts={"model.json": "model", "weights.npz": "weights"},
        prior_refit=dict(posterior_cache_sha256="cache", prior_parameters_sha256="prior"))
    checkpoint = "inputs/training/" + evaluation["selected_checkpoint"] + "/"
    field = "inputs/train_fields/train_block_014_familiar_0/"
    plan = [dict(path=checkpoint+name, sha256=digest, source="unused", bytes=1)
            for name, digest in evaluation["selected_model_artifacts"].items()]
    source_fields = tmp_path / "source_fields"
    source_fields.mkdir()
    map_path = source_fields / "spatial_fields.npz"
    map_path.write_bytes(b"bound-map-fixture")
    summary_path = source_fields / "summary.json"
    summary_path.write_text(json.dumps(dict(artifacts={"spatial_fields.npz": wrapper.digest(map_path)})))
    summary_sha = wrapper.digest(summary_path)
    evaluation["training_binding"] = dict(episode_bindings={
        "train_block_014_familiar_0": dict(field_summary_sha256=summary_sha)})
    # Actual fitting plans omit the unit map but bind its source summary.
    plan += [dict(path=field+"summary.json", sha256=summary_sha,
                  source=str(summary_path), bytes=summary_path.stat().st_size)]
    plan += [dict(path="inputs/train/train_block_014_familiar_0/spikes.npz"),
             dict(path="inputs/prior_refit/train_block_000_familiar_0.npz"),
             dict(path="inputs/training/checkpoints/epoch_0080_attempt_0000/weights.npz")]
    if fault == "parent": reference["reference_delivery_sha256"] = "changed"
    elif fault == "seed": evaluation["training_seed"] = 43
    elif fault == "budget": evaluation["effective_training_epoch_budget"] = 20
    elif fault == "weights": plan[1]["sha256"] = "wrong"
    elif fault == "missing_map": map_path.unlink()
    elif fault == "summary_hash": evaluation["training_binding"]["episode_bindings"]["train_block_014_familiar_0"]["field_summary_sha256"] = "wrong"
    monkeypatch.setattr(forecast, "stationary_input_plan", lambda *a: plan)
    if fault is not None:
        with pytest.raises(ValueError): wrapper.stationary_readout_plan(tmp_path, reference, evaluation)
    else:
        selected = wrapper.stationary_readout_plan(tmp_path, reference, evaluation)
        assert {row["path"] for row in selected} == {checkpoint+"model.json", checkpoint+"weights.npz",
                                                     field+"summary.json", field+"spatial_fields.npz"}
        with pytest.raises(ValueError, match="FIT_RUN_REQUIRED"):
            wrapper.stationary_readout_plan(None, reference, evaluation)


# Sparse authenticated fixtures: no forecasts, latent arrays, source data or models.
def _publish_matrix_fixture(folder, documents):
    import csv
    import json
    from .provenance import sha256_file
    parent, evaluation, report, rows = (documents[key] for key in ("parent", "evaluation", "report", "rows"))
    parent_root = folder / "reference"
    prefix = "fitted_prior" if parent["status"] == "COMPARED_PRIORS_DEVELOPMENT" else "candidate"
    evaluation_path = parent_root / prefix / "forecasts/evaluation.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation))
    def entry(path, root):
        return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256_file(path))
    parent["files"] = [entry(evaluation_path, parent_root)]
    parent_path = parent_root / "delivery.json"; parent_path.write_text(json.dumps(parent))
    report["reference_evaluation_sha256"] = sha256_file(evaluation_path)
    analysis = folder / "analysis"; analysis.mkdir(exist_ok=True)
    table_path = analysis / "paired_block_decisions.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report["artifacts"] = {table_path.name: sha256_file(table_path)}
    report_path = analysis / "analysis.json"; report_path.write_text(json.dumps(report))
    delivery = dict(status=report["status"], exit_code=0, route=report["route"],
        reference_delivery=dict(path=str(parent_path), sha256=sha256_file(parent_path)),
        files=[entry(report_path, folder), entry(table_path, folder)])
    (folder / "delivery.json").write_text(json.dumps(delivery))


def _conditional_matrix_fixture(tmp_path, stop=True):
    import json
    from pathlib import Path
    import yaml
    from . import lfads_conditional_comparison as module
    from .provenance import sha256_file
    protocol_path = Path(__file__).parents[1] / "configs/neural_population_lfads_conditional_prior.yaml"
    baseline = yaml.safe_load(protocol_path.read_text())
    cells, documents, first_binding = [], {}, None
    for budget in (20, 80):
        for seed in (41, 42, 43):
            legacy = (budget, seed) == (20, 41)
            folder = tmp_path / f"budget_{budget}_seed_{seed}"; folder.mkdir()
            protocol = deepcopy(baseline)
            if not legacy:
                for name in ("training_seed", "network_budget_epochs"): protocol.pop(name)
                protocol.update(scope="registered_development_causal_prior_matrix", training_seeds=[41, 42, 43],
                    network_budget_epoch_ceilings=[20, 80], single_cell_protocol_sha256=sha256_file(protocol_path),
                    routing_reference=dict(first_binding, route="PASS", passed_block_metric_checks=24))
            selected = 19 if budget == 20 else 53
            training = dict(seed=seed, config=dict(maximum_epochs=20, fit_blocks=list(range(14)),
                validation_blocks=list(BLOCKS), bin_ms=5., model={"same": True}),
                acquisition_design_sha256="acquisition", acquisition_ledger_sha256="ledger", dataset_sha256="dataset",
                episode_table_sha256="episodes", native_source_identity={"graph": "one"},
                episode_bindings={"complete": "shared"})
            evaluation = dict(status="EVALUATED_DEVELOPMENT", training_seed=seed, training_binding=training,
                effective_training_epoch_budget=budget, selected_epoch=selected, completed_training_epochs=20 if budget == 20 else 58,
                formal_test_accessed=False, native_source_identity={"graph": "one", "dt_ms": .1},
                configuration={"horizons_ms": [20, 80, 500], "joint_samples": 16},
                prior_refit=dict(training_seed=seed, selected_epoch=selected+1, fitted_parameters=8, fitting_episodes=56),
                training_continuation=None if budget == 20 else dict(schema="neural_population_lfads_budget_continuation_v1",
                    total_epochs=80, base_maximum_epochs=20, treatment="optimization_budget_only"))
            parent = dict(status="COMPARED_PRIORS_DEVELOPMENT" if legacy else "REPLAYED_STATIONARY_PRIOR_CANDIDATE",
                training_seed=seed, network_variant="budget_20", network_budget_epochs=budget)
            if not legacy: parent["exit_code"] = 0
            rows, blocks = [], []
            for block in BLOCKS:
                references, candidates, deltas, passed = {}, {}, {}, {}
                for metric in module.METRICS:
                    ref = 10. + block
                    delta = -(seed-40.) if metric in PRIMARY else 0.
                    if stop and (budget, seed, block, metric) == (80, 43, 17, PRIMARY[0]): delta = 12.
                    cand = ref + delta
                    okay = cand < ref if metric in PRIMARY else cand <= ref
                    references[metric], candidates[metric], deltas[metric], passed[metric] = ref, cand, delta, okay
                    rows.append(dict(block=block, horizon_ms=500, metric=metric, reference_arm=REFERENCE,
                        candidate_arm=CANDIDATE, reference=ref, candidate=cand, candidate_minus_reference=delta, passed=okay))
                blocks.append(dict(block=block, reference=references, candidate=candidates, deltas=deltas,
                                   metric_pass=passed, route="PASS" if all(passed.values()) else "STOP"))
            report = dict(status="DESCRIBED_REPLAYED_CONDITIONAL_PRIOR_COMPARISON", training_seed=seed,
                training_seeds=1, independent_blocks=4, connectivity_instances=1, paired_origins=48, score_rows_per_arm=144,
                exact_paired_causal_paths=True, reference_arm=REFERENCE, candidate_arm=CANDIDATE, horizon_ms=500,
                scientific_claim_qualified=False, formal_test_accessed=False, blocks=blocks,
                route="PASS" if all(row["passed"] for row in rows) else "STOP", comparison_protocol=protocol,
                mean_binding=dict(comparison_protocol=protocol, comparison_protocol_sha256=sha256_file(protocol_path)))
            if not legacy: report["network_budget_epochs"] = budget
            values = dict(parent=parent, evaluation=evaluation, report=report, rows=rows)
            _publish_matrix_fixture(folder, values)
            if legacy:
                first_binding = dict(analysis_sha256=sha256_file(folder / "analysis/analysis.json"),
                    delivery_sha256=sha256_file(folder / "delivery.json"))
            cells.append((budget, seed, folder)); documents[budget, seed] = values
    return cells, documents


def test_conditional_matrix_retains_stop_and_averages_seeds_within_four_blocks(tmp_path):
    import csv
    from .lfads_conditional_comparison import aggregate_matrix
    cells, _ = _conditional_matrix_fixture(tmp_path)
    output = tmp_path / "result"
    result = aggregate_matrix(list(reversed(cells)), output)
    assert result["completed_cells"] == 6 and result["training_seeds"] == 3 and result["independent_blocks"] == 4
    assert result["route"] == "STOP" and result["failed_seed_block_metric_checks"] == 1
    assert result["retained_seed_block_metric_decisions"] == 144
    assert result["scientific_claim_qualified"] is False and result["cross_node_runtime_comparison_qualified"] is False
    def read(name):
        with (output / name).open() as stream: return list(csv.DictReader(stream))
    original = read("conditional_matrix_by_seed_block.csv")
    blocks = read("conditional_matrix_by_block.csv")
    summary = read("conditional_matrix_summary.csv")
    assert len(original) == 144 and len(blocks) == 48 and len(summary) == 12
    assert any(float(row["candidate_minus_reference"]) < 0 for row in original)
    stopped = [row for row in original if row["passed"] == "False"]
    assert len(stopped) == 1 and stopped[0]["training_seed"] == "43"
    tail = next(row for row in blocks if row["network_budget_epochs"] == "80" and row["block"] == "17" and row["metric"] == PRIMARY[0])
    assert float(tail["candidate_minus_reference"]) == 3. and tail["failed_seed_checks"] == "1"
    averaged = next(row for row in summary if row["network_budget_epochs"] == "80" and row["metric"] == PRIMARY[0])
    assert float(averaged["candidate_minus_reference"]) == -.75 and averaged["independent_blocks"] == "4"
    assert not any(key in averaged for key in ("p_value", "q_value", "ci_lower", "ci_upper"))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "seed", "budget", "source", "protocol", "decision", "missing_decision", "parent"])
def test_conditional_matrix_rejects_incomplete_or_mixed_bindings(tmp_path, fault):
    from .lfads_conditional_comparison import aggregate_matrix
    cells, documents = _conditional_matrix_fixture(tmp_path)
    folder = cells[-1][2]; last = documents[80, 43]
    if fault == "missing": cells.pop()
    elif fault == "duplicate": cells[-1] = cells[0]
    elif fault == "seed": last["evaluation"]["training_seed"] = 42
    elif fault == "budget": last["evaluation"]["effective_training_epoch_budget"] = 20
    elif fault == "source": last["evaluation"]["training_binding"]["dataset_sha256"] = "different_dataset"
    elif fault == "protocol": last["report"]["comparison_protocol"]["primary"]["horizon_ms"] = 80
    elif fault == "decision": last["rows"][0]["passed"] = not last["rows"][0]["passed"]
    elif fault == "missing_decision": last["rows"].pop()
    if fault not in ("missing", "duplicate"): _publish_matrix_fixture(folder, last)
    if fault == "parent":
        (folder / "reference/delivery.json").write_text("{}")
    with pytest.raises(ValueError): aggregate_matrix(cells, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_conditional_matrix_cli_reads_only_sparse_staged_receipts_and_tables(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from .lfads_conditional_comparison import aggregate_matrix
    cells, _ = _conditional_matrix_fixture(tmp_path)
    command = [sys.executable, "-B", "-m", "Libs.population_dynamics.lfads_conditional_comparison",
               "--output", str(tmp_path / "cli_result")]
    for budget, seed, folder in cells:
        staged_parent = tmp_path / f"staged_parent_{budget}_{seed}"
        shutil.copytree(folder / "reference", staged_parent)
        shutil.rmtree(folder / "reference")
        command.extend(["--matrix-cell", str(budget), str(seed), str(folder),
                        "--matrix-reference", str(budget), str(seed), str(staged_parent / "delivery.json")])
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "cli_result/analysis.json").read_text())
    assert report["completed_cells"] == 6 and report["route"] == "STOP"
    assert all(item["reference_delivery_path"].endswith("reference/delivery.json") for item in report["inputs"])

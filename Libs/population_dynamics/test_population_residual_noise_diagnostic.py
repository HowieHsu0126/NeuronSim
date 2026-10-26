"""Fixed-checkpoint, random-stream and complete-denominator contracts."""
from copy import deepcopy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import population_residual_noise_diagnostic as diagnostic
from .population_residual_mean_objective import residual_mean_objective
from .test_population_residual_mean_objective import combined_arguments


def tiny(mode="history"):
    args = combined_arguments(mode)
    model = args.pop("model")
    return model, args


@pytest.mark.parametrize("mode", ["current", "history"])
def test_factor_one_preserves_native_loss_sampler_paths_and_explicit_rng(mode):
    model, batch = tiny(mode)
    rng, _ = diagnostic.named_streams(41, "case", 0, "cpu")
    with torch.no_grad():
        direct = residual_mean_objective(model, **batch, initial_generator=rng.initial, dynamics_generator=rng.dynamics,
            unit_chunk_size=4096, mean_unit_chunk_size=32768, checkpoint_readout=False)
    paths = []
    original = model.parent.log_hazard
    def capture(path, start=0, stop=None):
        paths.append(path.detach().clone())
        return original(path, start, stop)
    before = diagnostic.snapshot(model)
    global_state = torch.get_rng_state().clone()
    with patch.object(model.parent, "log_hazard", side_effect=capture):
        result = diagnostic.evaluate_variant(model, batch, 41, "case", 0, 1.0)
    assert result["objective"] == {name: float(value) for name, value in direct.items()}
    native = next(path for path in paths if path.ndim == 5)
    generated = [path for path in paths if path.ndim == 4 and path.shape[0] == 8]
    assert torch.equal(native, torch.stack(generated, dim=2))
    assert result["pairing"]["initial_end_sha256"] == diagnostic.tensor_digest(rng.initial.get_state())
    assert result["pairing"]["dynamics_end_sha256"] == diagnostic.tensor_digest(rng.dynamics.get_state())
    assert result["sampled_field_counts"].shape == (8, 6, 4, 2)
    assert result["sampled_field_counts"].dtype == np.int64
    state, controls = diagnostic.boundary_and_controls(model, batch)
    sampled_rng, _ = diagnostic.named_streams(41, "case", 0, "cpu")
    direct_fields = []
    for step in model.iter_simulation(state, controls, sampled_rng, 8, record_events=False):
        direct_fields.append(torch.stack([step.counts[..., model.unit_field_index == field].sum(-1)
                                          for field in range(len(model.field_names))], -1))
    np.testing.assert_array_equal(result["sampled_field_counts"], torch.stack(direct_fields, 2).numpy())
    replay = diagnostic.evaluate_variant(model, batch, 41, "case", 0, 1.0)
    np.testing.assert_array_equal(result["sampled_field_counts"], replay["sampled_field_counts"])
    assert result["pairing"] == replay["pairing"]
    assert diagnostic.unchanged(model, before)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("failure", [False, True])
def test_scale_changes_only_process_parameter_and_restores_bit_exactly(failure):
    model, _ = tiny()
    before = diagnostic.snapshot(model)
    def run():
        with diagnostic.process_variance_scale(model, 1.25):
            state = model.state_dict()
            for name in before:
                if name != "parent.process_log_variance":
                    assert torch.equal(state[name], before[name])
            torch.testing.assert_close(state["parent.process_log_variance"],
                before["parent.process_log_variance"] + math.log(1.25), rtol=0, atol=0)
            if failure:
                raise RuntimeError("injected forward failure")
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            run()
    else:
        run()
    assert diagnostic.unchanged(model, before)


def test_native_process_innovation_standard_deviation_scales_as_square_root():
    model, batch = tiny()
    state, controls = diagnostic.boundary_and_controls(model, batch)
    def first(factor):
        rng, _ = diagnostic.named_streams(41, "case", 0, "cpu")
        with diagnostic.process_variance_scale(model, factor):
            step = next(model.iter_simulation(state, controls, rng, 8, record_events=False))
            base = model.parent.transition_mean(step.initial_latent, controls[:, 0])
            residual = step.latent - base - model.correction(step.initial_latent, controls[:, 0], state.memory.expand_as(step.initial_latent))
            return step.initial_latent, residual
    initial, original = first(1.0)
    changed_initial, changed = first(1.25)
    assert torch.equal(initial, changed_initial)
    torch.testing.assert_close(changed, original * math.sqrt(1.25), rtol=1e-12, atol=1e-14)


def test_objective_failure_restores_full_model():
    model, batch = tiny()
    before = diagnostic.snapshot(model)
    with patch.object(diagnostic, "residual_mean_objective", side_effect=RuntimeError("injected objective failure")):
        with pytest.raises(RuntimeError, match="injected"):
            diagnostic.evaluate_variant(model, batch, 41, "case", 0, .8)
    assert diagnostic.unchanged(model, before)


def test_pairing_is_shared_across_factors_and_modes_but_not_panels():
    signatures = []
    for mode in ("current", "history"):
        model, batch = tiny(mode)
        for factor in (.8, 1., 1.25):
            result = diagnostic.evaluate_variant(model, batch, 41, "case", 0, factor)
            signatures.append(tuple(result["pairing"][key] for key in diagnostic.PAIRING_KEYS))
    assert len(set(signatures)) == 1
    _, left = diagnostic.named_streams(41, "case", 0, "cpu")
    _, right = diagnostic.named_streams(41, "case", 1, "cpu")
    assert set(left.values()).isdisjoint(right.values())


def synthetic_records():
    records = []
    for condition in diagnostic.conditions(diagnostic.PROTOCOL):
        value = condition["variance_multiplier"]
        pairing = dict.fromkeys(diagnostic.PAIRING_KEYS, "same-paired-identity")
        pairing["objective_simulation_latent_rng_equal"] = True
        records.append(dict(**condition, status="COMPLETE", pairing=pairing,
            objective={name: value for name in diagnostic.objective_names(diagnostic.PROTOCOL)},
            sampled_metrics=[dict(origin=origin, horizon=horizon, **dict.fromkeys(diagnostic.SAMPLED_METRICS, value))
                for origin in diagnostic.PROTOCOL["origins"] for horizon in diagnostic.PROTOCOL["horizons"]]))
    return records


def test_full_condition_aggregation_and_paired_deltas():
    result = diagnostic.aggregate_evaluations(synthetic_records(), diagnostic.PROTOCOL)
    assert result["expected_evaluations"] == result["complete_evaluations"] == 144
    assert len(result["objective_rows"]) == 144
    assert len(result["sampled_rows"]) == result["expected_sampled_rows"] == 1296
    assert len(result["objective_means"]) == 36
    assert len(result["sampled_means"]) == 108
    assert result["paired_groups_verified"] == 24
    for row in result["objective_means"]:
        assert row["complete_rows"] == row["expected_rows"] == 4
        assert row["delta_loss"] == pytest.approx(row["variance_multiplier"] - 1)
    for row in result["sampled_means"]:
        assert row["complete_rows"] == row["expected_rows"] == 12
        assert row["delta_energy_score"] == pytest.approx(row["variance_multiplier"] - 1)


def test_failures_remain_in_full_denominator_and_disable_affected_means():
    records = synthetic_records()
    records[1] = {**records[1], "status": "FAILED", "objective": {}, "sampled_metrics": []}
    result = diagnostic.aggregate_evaluations(records, diagnostic.PROTOCOL)
    assert result["complete_evaluations"] == 143
    assert len(result["sampled_rows"]) == 1296
    affected = [r for r in result["objective_means"] if (r["seed"], r["mode"], r["panel"]) == (41, "current", 0)]
    assert all(r["delta_loss"] is None for r in affected)
    baseline = next(r for r in affected if r["variance_multiplier"] == 1.)
    assert baseline["status"] == "INCOMPLETE" and baseline["loss"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "origin", "pairing"])
def test_aggregation_rejects_incomplete_or_unpaired_completed_conditions(change):
    rows = synthetic_records()
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = deepcopy(rows[0])
    elif change == "origin": rows[0]["sampled_metrics"].pop()
    else: rows[0]["pairing"]["dynamics_end_sha256"] = "changed"
    with pytest.raises(ValueError):
        diagnostic.aggregate_evaluations(rows, diagnostic.PROTOCOL)


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_variance_factors_reject_without_mutating_weights(factor):
    model, _ = tiny()
    before = diagnostic.snapshot(model)
    with pytest.raises(ValueError):
        with diagnostic.process_variance_scale(model, factor):
            pass
    assert diagnostic.unchanged(model, before)


@pytest.mark.parametrize("key,value", [("training_seeds", [41, 42, 44]), ("panels", [0]),
    ("variance_multipliers", [.8, 1., 1.2]), ("source_units", 1000), ("templates_per_block", True)])
def test_registered_config_rejects_protocol_changes(key, value):
    config = deepcopy(diagnostic.PROTOCOL)
    config[key] = value
    with pytest.raises(ValueError): diagnostic.validate_config(config)


def test_yaml_matches_registered_protocol():
    path = Path(__file__).parents[1] / "configs/population_residual_noise_diagnostic.yaml"
    assert diagnostic.validate_config(yaml.safe_load(path.read_text())) == diagnostic.PROTOCOL


def test_completed_reuse_verifies_identity_and_field_artifact(tmp_path):
    condition = diagnostic.conditions(diagnostic.PROTOCOL)[0]
    fields = tmp_path / "sampled_fields.npz"
    fields.write_bytes(b"bounded completed field fixture")
    report = dict(**condition, diagnostic_identity_sha256="bound", status="COMPLETE", exit_code=0,
        state_unchanged=True, field_artifact=dict(path=fields.name, bytes=fields.stat().st_size,
                                               sha256=diagnostic.sha256_file(fields)))
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(report))
    assert diagnostic.read_completed(path, condition, "bound") == report
    with pytest.raises(ValueError): diagnostic.read_completed(path, condition, "other")
    fields.write_bytes(b"changed")
    with pytest.raises(ValueError): diagnostic.read_completed(path, condition, "bound")


def test_failed_reuse_preserves_failure_without_fabricating_fields(tmp_path):
    condition = diagnostic.conditions(diagnostic.PROTOCOL)[0]
    report = dict(**condition, diagnostic_identity_sha256="bound", status="FAILED", exit_code=2,
                  error="preserved original failure")
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(report))
    assert diagnostic.read_completed(path, condition, "bound") == report
    assert not (tmp_path / "sampled_fields.npz").exists()

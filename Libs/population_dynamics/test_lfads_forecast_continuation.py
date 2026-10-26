"""Committed-prefix recovery, preserved RNG indices and native call-order checks."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import lfads_trajectory_evaluation as evaluator
from .lfads_forecast_continuation import continuation_default_program
from .provenance import sha256_file
from .test_lfads_trajectory_evaluation import cohort_fixture, call, SmallModel
from .test_lfads_conditional_prior import native_model


def make_parent(fixture, baseline, completed):
    parent = fixture.root / "parent"
    forecasts = parent / "candidate/forecasts"
    forecasts.mkdir(parents=True)
    original = json.loads((baseline / "evaluation.json").read_text())
    record = deepcopy(original)
    record.update(status="RUNNING", completed_origins=completed, score_rows=completed * 3)
    # Model a SIGTERM with one fully written but uncommitted following origin.
    committed_graphs = []
    remaining = completed
    for graph in original["graph_lifetimes"]:
        if not remaining:
            break
        graph = deepcopy(graph)
        graph["completed_cases"] = min(16, remaining)
        committed_graphs.append(graph);remaining -= graph["completed_cases"]
    record["graph_lifetimes"] = committed_graphs
    record.pop("artifacts")
    record.pop("scores_sha256")
    (forecasts / "evaluation.json").write_text(json.dumps(record))
    shutil.copy2(Path(evaluator.__file__), parent / "source_evaluator.py")
    with (baseline / "scores.csv").open(newline="") as stream:
        reader = csv.DictReader(stream);columns, rows = reader.fieldnames, list(reader)
    tail = rows[:(completed + 1) * 3]
    with (forecasts / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns);writer.writeheader();writer.writerows(tail)
    for name in {"fitting_field_scale.npz", "full_unit_samples.npz", *(row["ensemble_artifact"] for row in tail)}:
        shutil.copy2(baseline / name, forecasts / name)
    phase = parent / "phases/forecast/receipt.json";phase.parent.mkdir(parents=True)
    phase.write_text(json.dumps(dict(status="RESOURCE_BLOCKED", command_started=True, exit_code=-15,
        started_utc="2026-09-30T00:00:00+00:00", finished_utc="2026-09-30T00:00:02+00:00")))
    delivery = dict(status="RESOURCE_BLOCKED", exit_code=3, job_id="fixture",
        started_utc="2026-09-30T00:00:00+00:00", completed_utc="2026-09-30T00:00:03+00:00",
        files=[dict(path=str(p.relative_to(parent)), bytes=p.stat().st_size, sha256=sha256_file(p))
               for p in parent.rglob("*") if p.is_file()])
    (parent / "delivery.json").write_text(json.dumps(delivery))
    return parent


@pytest.mark.parametrize("completed", [9, 19])
def test_resume_skips_only_committed_prefix_and_matches_uninterrupted_outputs(cohort_fixture, completed):
    f = cohort_fixture
    call(f, "baseline")
    baseline = f.root / "baseline"
    parent = make_parent(f, baseline, completed)
    parent_digest = sha256_file(parent / "delivery.json")
    SmallModel.samples, SmallModel.histories, SmallModel.graphs = [], [], []
    result = evaluator.evaluate(f.acquisition, f.fields, f.training, f.config, f.root / "continued", "cpu",
                                continue_from=parent)
    assert len(SmallModel.samples) == 48 - completed
    assert len(SmallModel.graphs) == (3 if completed == 9 else 2)
    assert result["completed_origins"] == 48 and result["score_rows"] == 144
    assert sum(row["completed_cases"] for row in result["graph_lifetimes"]) == 48
    assert len(result["graph_lifetimes"]) == 4
    continuation = result["forecast_continuation"]
    assert continuation["inherited_origins"] == completed and continuation["ignored_uncommitted_score_rows"] == 3
    assert continuation["parent_forecast_phase"]["exit_code"] == -15
    assert sha256_file(parent / "delivery.json") == parent_digest
    for name in result["artifacts"]:
        with np.load(baseline / name, allow_pickle=False) as left, np.load(f.root / "continued" / name, allow_pickle=False) as right:
            assert set(left.files) == set(right.files)
            for key in left.files:
                assert left[key].dtype == right[key].dtype
                assert left[key].tobytes() == right[key].tobytes(), (name, key)
    for name, checksum in continuation["inherited_artifacts"].items():
        assert sha256_file(f.root / "continued" / name) == checksum
    with (baseline / "scores.csv").open() as left, (f.root / "continued/scores.csv").open() as right:
        a, b = list(csv.DictReader(left)), list(csv.DictReader(right))
    assert len(a) == len(b) == 144
    for original, resumed in zip(a, b):
        for key in original:
            if not key.endswith(("_seconds", "_bytes")):
                assert original[key] == resumed[key], key


@pytest.mark.parametrize("change", ["payload", "horizon", "checkpoint", "open_parent"])
def test_bad_parent_is_rejected_before_any_new_origin(cohort_fixture, change):
    f = cohort_fixture
    call(f, "baseline")
    parent = make_parent(f, f.root / "baseline", 9)
    root = parent / "candidate/forecasts"
    if change == "payload":
        (root / "train_block_014_familiar_0_origin_1000.npz").write_bytes(b"corrupted")
    elif change == "horizon":
        p = root / "scores.csv";p.write_text(p.read_text().replace(",20.0,", ",21.0,", 1))
        # An internally consistent receipt still cannot admit an invalid horizon.
        d = json.loads((parent / "delivery.json").read_text())
        for row in d["files"]:
            if row["path"] == "candidate/forecasts/scores.csv":
                row.update(bytes=p.stat().st_size, sha256=sha256_file(p))
        (parent / "delivery.json").write_text(json.dumps(d))
    elif change == "checkpoint":
        p = root / "evaluation.json";r = json.loads(p.read_text());r["selected_checkpoint"] = "different";p.write_text(json.dumps(r))
        d = json.loads((parent / "delivery.json").read_text())
        for row in d["files"]:
            if row["path"] == "candidate/forecasts/evaluation.json":
                row.update(bytes=p.stat().st_size, sha256=sha256_file(p))
        (parent / "delivery.json").write_text(json.dumps(d))
    else:
        p = parent / "delivery.json";d=json.loads(p.read_text());d["status"]="RUNNING";p.write_text(json.dumps(d))
    SmallModel.samples = []
    with pytest.raises(ValueError):
        evaluator.evaluate(f.acquisition, f.fields, f.training, f.config, f.root / "rejected", "cpu", continue_from=parent)
    assert not SmallModel.samples


def test_continuation_specialization_retains_both_prior_branches_and_numerical_body():
    original = """def evaluate(x, *, prior_refit=None):
    if prior_refit is not None:
        x += 2
    return x + 1
"""
    changed = original.replace("prior_refit=None):", "prior_refit=None, continue_from=None):")
    changed = changed.replace("    if prior_refit", """    completed_keys = None
    if continue_from is not None:
        completed_keys = verified_prefix(continue_from)
    if prior_refit""")
    assert continuation_default_program(original) == continuation_default_program(changed)
    for old, new in [("x += 2", "x += 3"), ("x + 1", "x + 2"), ("continue_from=None", "continue_from=1")]:
        assert continuation_default_program(original) != continuation_default_program(changed.replace(old, new))


def test_native_generation_has_no_previous_origin_random_or_model_state_dependency(native_model):
    from .lfads import LFADSRandomStreams
    from .lfads_conditional_prior import simulate_with_mean
    from dataclasses import fields
    model, state, stimulus = native_model
    curve = np.linspace(.1, .4, 16, dtype=np.float32).reshape(4, 4)
    streams = lambda: LFADSRandomStreams.from_seeds(901, 902, 903)
    direct = simulate_with_mean(model, state, stimulus, streams(), 4, curve)
    # A skipped earlier origin would have used other streams and controls.
    simulate_with_mean(model, state, stimulus * .3, LFADSRandomStreams.from_seeds(801, 802, 803), 4, curve * -.7)
    model.infer_state(dict(counts=np.full((1, 3, 2), 3, np.int64), stimulus=state.known_input_history * .3))
    fresh_state = model.infer_state(dict(counts=np.array([[[0, 2], [1, 0], [2, 1]]], np.int64),
                                         stimulus=state.known_input_history))
    after_other_origin = simulate_with_mean(model, fresh_state, stimulus, streams(), 4, curve)
    for field in fields(direct):
        left, right = getattr(direct, field.name), getattr(after_other_origin, field.name)
        if isinstance(left, np.ndarray):
            assert left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes(), field.name
        else:
            assert left == right

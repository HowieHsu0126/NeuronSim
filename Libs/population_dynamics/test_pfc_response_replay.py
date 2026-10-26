"""Focused saved-response arithmetic and manifest rejection tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

from . import pfc_response_replay as replay
from .provenance import sha256_file
from .trajectory_scores import energy_score


@pytest.mark.parametrize("horizon", [1, 2])
def test_signed_integer_u_statistic_matches_explicit_ordered_draw_pairs(horizon):
    candidate = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[2, 0], [0, 2]]], dtype=np.int64)
    factual = np.ones_like(candidate)
    candidate_truth = np.array([[0, 0], [1, 0]], dtype=np.int64)
    factual_truth = np.array([[1, 0], [0, 1]], dtype=np.int64)
    residual = candidate[:, :horizon] - factual[:, :horizon] - candidate_truth[None, :horizon] + factual_truth[None, :horizon]
    oracle = sum(float(np.mean(residual[left] * residual[right]))
                 for left in range(3) for right in range(3) if left != right) / 6
    assert replay.ordered_pair_mse(candidate, factual, candidate_truth, factual_truth, horizon,
                                   chunk_units=1) == oracle


def test_negative_effect_correction_and_field_units_are_preserved():
    candidate = np.array([[[0]], [[2]]], dtype=np.int64)
    factual = np.ones_like(candidate)
    truth = np.zeros((1, 1), dtype=np.int64)
    row = replay.independent_contrasts(candidate, factual, truth, truth,
        np.array([0], dtype=np.int64), 1, np.array([1]), .005, [1])[0]
    assert row["unit_mean_contrast_mse_corrected"] == -1.
    assert row["field_mean_contrast_mse_hz2_corrected"] == -40000.
    assert row["generated_total_contrast_per_unit"] == 0.


def test_independent_field_energy_matches_existing_unbiased_score():
    values = np.array([[[0., 2.]], [[1., 1.]], [[3., 0.]]])
    observed = np.array([[1., 2.]])
    assert replay._energy(values, observed) == pytest.approx(energy_score(values, observed), abs=1e-14)


@pytest.mark.parametrize("change", ["changed_bytes", "changed_hash", "missing_condition"])
def test_manifest_corruption_rejects_before_array_loader(tmp_path, monkeypatch, change):
    outputs = {}
    for name in (["arm_scores.csv", "mean_contrasts.csv", "costs.csv"]
                 + [f"condition_{index:02d}/{name}" for index in range(10)
                    for name in ("comparison.json", "samples.npz")]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"small opaque fixture")
        outputs[name] = dict(bytes=path.stat().st_size, sha256=sha256_file(path))
    if change == "changed_bytes":
        (tmp_path / "condition_00/samples.npz").write_bytes(b"changed")
    elif change == "changed_hash":
        outputs["condition_00/samples.npz"]["sha256"] = "0" * 64
    else:
        del outputs["condition_09/samples.npz"]
    monkeypatch.setattr(replay.np, "load", lambda *a, **k: pytest.fail("corruption reached array read"))
    with pytest.raises(ValueError):
        replay.authenticate_outputs(tmp_path, {"outputs": outputs})


def test_difference_report_preserves_all_units_and_first_bin():
    left = np.zeros((2, 4, 5), dtype=np.int64)
    right = left.copy()
    right[1, 2, 4] = 1
    assert replay._difference(left, right, 2)["exact"] is True
    report = replay._difference(left, right, 3)
    assert report["differences"] == 1 and report["first_difference_bin"] == 2


def test_invalid_complete_result_rejects_before_output_or_array_reads(tmp_path, monkeypatch):
    (tmp_path / "evaluation.json").write_text(json.dumps({"status": "PARTIAL"}))
    monkeypatch.setattr(replay.np, "load", lambda *a, **k: pytest.fail("partial result read arrays"))
    output = tmp_path / "replay.json"
    with pytest.raises((ValueError, KeyError)):
        replay.replay(tmp_path, output)
    assert not output.exists()

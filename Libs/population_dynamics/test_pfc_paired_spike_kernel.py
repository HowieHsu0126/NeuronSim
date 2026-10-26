"""Filtration and directed interval checks for the reused paired spike kernel."""
import json
from fractions import Fraction
import sys

import numpy as np
import pytest

from . import pfc_conditional_spike_probability as kernel


@pytest.mark.parametrize("factor", [2, 4])
def test_reversal_first_half_fresh_second_half_already_revealed(factor):
    for residue in range(factor):
        row = kernel.reversal_joint_information(20 * factor + residue, factor)
        assert row["current_coordinates"] == [residue, factor - 1 - residue]
        assert row["both_current_inputs_fresh"] == (residue < factor // 2)
        assert row["individual_history_sigma_fields_equal"] == (residue == 0)


def test_directed_xor_extrema_include_high_probability_corners():
    result = kernel.independent_pair_bounds(10, 30, 60, 90, 100)
    assert result["disagreement_lower_scaled"] == "54"
    assert result["disagreement_upper_scaled"] == "82"
    assert result["neither_lower_scaled"] == "7" and result["both_upper_scaled"] == "27"
    names = ("neither", "original_only", "reversed_only", "both")
    assert sum(int(result[name + "_lower_scaled"]) for name in names) < 100
    assert sum(int(result[name + "_upper_scaled"]) for name in names) > 100


@pytest.mark.parametrize("a,b,xor", [(0, 0, 0), (0, 100, 100), (100, 0, 100), (100, 100, 0)])
def test_exact_always_never_joint_outcomes(a, b, xor):
    result = kernel.independent_pair_bounds(a, a, b, b, 100)
    assert int(result["disagreement_lower_scaled"]) == int(result["disagreement_upper_scaled"]) == xor


def test_rounded_pair_bounds_enclose_exact_nonterminating_source_probabilities():
    bounds = kernel.binomial_tail_bounds(3, 1, 3, 4)
    a, b = bounds[1], bounds[2]
    result = kernel.independent_pair_bounds(int(a["lower_scaled"]), int(a["upper_scaled"]),
                                            int(b["lower_scaled"]), int(b["upper_scaled"]), 10000)
    pa, pb = Fraction(19, 27), Fraction(7, 27)
    truth = pa + pb - 2 * pa * pb
    assert Fraction(int(result["disagreement_lower_scaled"]), 10000) <= truth
    assert truth <= Fraction(int(result["disagreement_upper_scaled"]), 10000)


def test_paired_reader_rejects_future_input_revealed_by_other_history(tmp_path, monkeypatch):
    archive, output = tmp_path / "archive", tmp_path / "output"
    (archive / "analysis").mkdir(parents=True)
    info = dict(refinement_factor=4, coupling=dict(input_order_control=dict(
        name="within_coarse_tick_tree_temporal_reversal", fine_row_order=[3, 2, 1, 0])))
    parent = dict(one_step_kernel_evaluation_complete=True, input_tick=90,
        variants={name: dict(drive_metadata=info) for name in ("original", "reversed")})
    (archive / "analysis/report.json").write_text(json.dumps(parent))
    names = ("analysis/report.json", "analysis/conditional_thresholds.npz", "analysis/binomial_tail_bounds.csv")
    for name in names[1:]:
        (archive / name).write_text("not consumed after conditioning rejection")
    (archive / "delivery.json").write_text(json.dumps(dict(verified=True,
        status="COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL", files=[dict(path=name) for name in names])))
    monkeypatch.setattr(kernel, "sha256_file", lambda path: "test")
    monkeypatch.setattr(kernel.branch, "_bound_file", lambda root, name, record: root / name)
    result = kernel.analyze_pair(archive, "test", output)
    assert result["status"] == "FAILED_PFC_PAIRED_SPIKE_KERNEL"
    assert result["error"]["message"].endswith("JOINT_HISTORY_REVEALS_CURRENT_INPUT")


def test_original_cli_still_dispatches_source_kernel_without_changing_its_arguments(monkeypatch, tmp_path):
    calls = []
    def original(archive, digest, output):
        calls.append((archive, digest, output))
        return dict(status="COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL")
    monkeypatch.setattr(kernel, "analyze", original)
    monkeypatch.setattr(sys, "argv", ["kernel", "--archive", str(tmp_path / "source"),
        "--delivery-sha256", "digest", "--output", str(tmp_path / "out")])
    assert kernel.main() == 0
    assert len(calls) == 1 and calls[0][1] == "digest"

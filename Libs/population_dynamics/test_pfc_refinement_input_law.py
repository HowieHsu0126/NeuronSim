"""Exact mathematical and CLI contracts for the PFC refinement-law analysis."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from population_dynamics.pfc_native_drive import _BINOMIAL_N, _RATE_HZ
from population_dynamics.pfc_refinement_input_law import analyze, routing_law, run


def fraction(record):
    assert type(record["numerator"]) is str
    assert type(record["denominator"]) is str
    assert record["display_float_for_display_only"] is True
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def source_config():
    return dict(excitatory_neurons=8, inhibitory_neurons=2, dt_ms=0.05, duration_ms=0.2)


@pytest.mark.parametrize("probability", [
    _RATE_HZ * (0.05 * 1e-3),
    _RATE_HZ * (0.05 * 1e-3) / 2,
    0.5,
    1.0,
])
def test_exact_category_probability_and_tv_contract(probability):
    law = routing_law(probability)
    actual, ideal = law["actual_categories"], law["ideal_categories"]
    assert sum(actual.values()) == sum(ideal.values()) == 1
    assert all(0 <= value <= 1 for value in (*actual.values(), *ideal.values()))
    tv = sum(abs(actual[name] - ideal[name]) for name in actual) / 2
    assert tv == law["category_total_variation"] == abs(law["rho"])
    assert actual["none"] == ideal["none"]
    assert actual["both"] - ideal["both"] == -law["rho"]
    assert actual["left_only"] - ideal["left_only"] == law["rho"] / 2
    assert actual["right_only"] - ideal["right_only"] == law["rho"] / 2
    assert law["q_rounding_error"] == 0


def test_known_binary64_conditional_probability_discrepancy():
    # q=1/8 and binary64 r approximates 1/7 from below.
    law = routing_law(0.5)
    assert law["q"] == Fraction(1, 8)
    assert law["r_star"] == Fraction(1, 7)
    assert law["r"] == Fraction(2573485501354569, 18014398509481984)
    assert law["rho"] == Fraction(1, 144115188075855872)
    assert law["category_total_variation"] == Fraction(1, 144115188075855872)


def test_full_history_counts_both_populations_and_every_parent_level():
    report = analyze(source_config())
    rows = report["levels"]
    assert report["source_core_units"] == 10
    assert report["coarse_ticks"] == 4
    assert report["final_ticks"] == 16
    assert [row["parent_ticks"] for row in rows] == [4, 8]
    assert [row["child_ticks"] for row in rows] == [8, 16]
    assert [row["parent_dt_ms"] for row in rows] == [0.05, 0.025]
    assert [row["child_dt_ms"] for row in rows] == [0.025, 0.0125]
    expected = _BINOMIAL_N * 10 * (
        4 * abs(fraction(rows[0]["rho"])) + 8 * abs(fraction(rows[1]["rho"]))
    )
    envelope = report["deterministic_parent_envelope"]
    assert fraction(envelope["unclamped_union"]) == expected
    assert fraction(envelope["full_history_tv_bound"]) == min(1, expected)
    assert fraction(report["data_processing"]["observable_law_perturbation_tv_bound"]) == min(1, expected)
    assert fraction(rows[1]["parent_probability"]) == fraction(rows[0]["parent_probability"]) / 2


def test_additional_refinement_level_retains_all_prior_contributions():
    two, three = analyze(source_config()), analyze(source_config(), levels=3)
    assert three["levels"][:2] == two["levels"]
    assert three["levels"][2]["parent_ticks"] == 16
    assert three["final_ticks"] == 32
    previous = fraction(two["deterministic_parent_envelope"]["unclamped_union"])
    extra = fraction(three["levels"][2]["history_union_contribution"])
    assert fraction(three["deterministic_parent_envelope"]["unclamped_union"]) == previous + extra


@pytest.mark.parametrize("probability", [True, False, None, "0.5", 0, -0.1, 1.01,
                                         float("nan"), float("inf"), float("-inf"), 5e-324])
def test_invalid_or_unrepresentable_probability_rejected(probability):
    with pytest.raises(ValueError):
        routing_law(probability)


@pytest.mark.parametrize(("key", "value"), [
    ("excitatory_neurons", True), ("excitatory_neurons", 0),
    ("excitatory_neurons", 3), ("excitatory_neurons", 8.0),
    ("inhibitory_neurons", False), ("inhibitory_neurons", 0),
    ("dt_ms", True), ("dt_ms", 0), ("dt_ms", float("nan")),
    ("dt_ms", float("inf")), ("dt_ms", 0.03),
    ("duration_ms", True), ("duration_ms", 0),
    ("duration_ms", float("nan")), ("duration_ms", float("inf")),
    ("duration_ms", 0.201), ("synapse_execution", "unsupported"),
])
def test_invalid_source_dimensions_or_clock_rejected(key, value):
    config = dict(source_config(), **{key: value})
    with pytest.raises(ValueError):
        analyze(config)


@pytest.mark.parametrize("levels", [True, 0, -1, 1.0, 10000])
def test_invalid_refinement_levels_rejected(levels):
    with pytest.raises(ValueError):
        analyze(source_config(), levels=levels)


def test_missing_source_dimensions_rejected():
    with pytest.raises(ValueError):
        analyze({"dt_ms": 0.05})


def test_scope_and_assumptions_do_not_promote_other_qualifications():
    report = analyze(source_config())
    assert report["comparison_changes_only"] == "binary64_r_to_exact_real_r_star_at_unchanged_binary64_q"
    assert report["deterministic_parent_envelope"]["parent_marginal_or_parent_independence_required"] is False
    assert "ideal independent binomial" in " ".join(report["deterministic_parent_envelope"]["assumptions"])
    assert report["ideal_iid_parent_bound_reported"] is False
    assert report["data_processing"]["coarse_fine_simulator_equality_implied"] is False
    for key in (
        "finite_prng_law_certified", "finite_sampler_law_certified",
        "coarse_fine_neuronal_equality_certified", "source_history_confidence_certified",
        "confirmation_0p01_128_certified", "confirmation_execution_released",
        "confirmation_sampling_design_changed", "numerical_margins_changed",
        "error_budget_allocated", "protocol_amendment_adopted", "production_qualified",
        "scientific_fidelity_qualified", "immutable_archived_source_bundle_transfer_verified",
    ):
        assert report[key] is False
    for key in ("new_episodes", "new_trials", "new_sampling", "source_simulations", "production_episodes"):
        assert report[key] == 0


def test_real_cli_writes_fresh_exact_analysis_with_current_binding(tmp_path):
    config, output = tmp_path / "source.yaml", tmp_path / "analysis.json"
    config.write_text(yaml.safe_dump(source_config()), encoding="utf-8")
    command = [sys.executable, "-B", "-m", "population_dynamics.pfc_refinement_input_law",
               "--config", str(config), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=dict(os.environ))
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert json.loads(result.stdout)["status"] == report["status"]
    assert report["provenance"]["resolved_source_config"] == source_config()
    assert report["provenance"]["immutable_source_bundles_verified"] is False
    hashes = report["provenance"]["current_code_sha256"]
    assert set(hashes) == {"pfc_refinement_input_law.py", "pfc_native_drive.py",
                           "pfc_clock_coupling.py", "pfc_deterministic_replay.py"}
    assert all(len(value) == 64 for value in hashes.values())
    assert fraction(report["levels"][0]["q_rounding_error"]) == 0
    assert fraction(report["deterministic_parent_envelope"]["full_history_tv_bound"]) >= 0
    preserved = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False, env=dict(os.environ))
    assert repeated.returncode != 0
    assert "PFC_REFINEMENT_FRESH_ANALYSIS_OUTPUT_REQUIRED" in repeated.stderr
    assert output.read_bytes() == preserved


def test_nonfinite_config_fails_without_creating_analysis(tmp_path):
    config, output = tmp_path / "invalid.yaml", tmp_path / "analysis.json"
    config.write_text(yaml.safe_dump(dict(source_config(), dt_ms=float("nan"))))
    with pytest.raises(ValueError):
        run(config, output)
    assert not output.exists()


def test_config_provenance_binds_bytes_analyzed_before_live_file_change(tmp_path, monkeypatch):
    config, output = tmp_path / "source.yaml", tmp_path / "analysis.json"
    original_settings = source_config()
    original_bytes = yaml.safe_dump(original_settings).encode("utf-8")
    changed_bytes = yaml.safe_dump(dict(original_settings, excitatory_neurons=16,
                                        duration_ms=0.4)).encode("utf-8")
    config.write_bytes(original_bytes)

    def replace_config_while_analyzing(settings, *, levels=2):
        assert settings == original_settings
        config.write_bytes(changed_bytes)
        return analyze(settings, levels=levels)

    monkeypatch.setattr("population_dynamics.pfc_refinement_input_law.analyze",
                        replace_config_while_analyzing)
    report = run(config, output)
    original_digest = hashlib.sha256(original_bytes).hexdigest()
    changed_digest = hashlib.sha256(changed_bytes).hexdigest()
    assert original_digest != changed_digest
    assert config.read_bytes() == changed_bytes
    assert report["provenance"]["source_config_sha256"] == original_digest
    assert report["provenance"]["resolved_source_config"] == original_settings
    assert report["source_core_units"] == 10
    assert report["coarse_ticks"] == 4
    assert json.loads(output.read_text())["provenance"] == report["provenance"]

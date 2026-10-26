"""Descriptive ratios retain complete denominators and adverse endpoint identities."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from .pfc_coupled_precision_summary import (
    CSV_NAME, FALSE_FLAGS, PARENT_STATUS, main, summarize_comparison,
)

COST_RATIO = 1.5


def endpoint(index, native, coupled, family="field_mean", status="DEFINED_RATIO"):
    undefined = status == "UNDEFINED"
    ratio = coupled / native if status == "DEFINED_RATIO" else ""
    row = {
        "domain_row_index": str(index), "family": family,
        "endpoint_identity": json.dumps({"phase": "cue", "field_index": str(index), "condition": "omission"}),
        "definition_status": "UNDEFINED_NO_WITHIN_PHASE_PAIRS" if undefined else "ESTIMATED_EMPIRICAL_SPREAD",
        "margin": "" if undefined else ".01", "units": "Hz",
        "native_sample_variance": "" if undefined else str(native),
        "coupled_sample_variance": "" if undefined else str(coupled),
        "variance_ratio_status": status, "variance_ratio": str(ratio),
        "variance_simulator_cost_ratio": str(ratio * COST_RATIO) if ratio != "" else "",
        "observed_simulator_cost_ratio": str(COST_RATIO),
        "native_original_independent_blocks": "32", "coupled_original_independent_blocks": "32",
        "native_empirical_status": "UNDEFINED_NO_WITHIN_PHASE_PAIRS" if undefined else
            "EMPIRICAL_ZERO_UNCERTAIN" if native == 0 else "ESTIMATED_EMPIRICAL_SPREAD",
        "coupled_empirical_status": "UNDEFINED_NO_WITHIN_PHASE_PAIRS" if undefined else
            "EMPIRICAL_ZERO_UNCERTAIN" if coupled == 0 else "ESTIMATED_EMPIRICAL_SPREAD",
        "native_cdf_tail_status": "", "coupled_cdf_tail_status": "",
    }
    if family == "count_cdf":
        for prefix, variance in (("native", native), ("coupled", coupled)):
            row[prefix + "_cdf_tail_status"] = (
                "EMPIRICAL_SATURATION_UNCERTAIN" if variance == 0
                else "OBSERVED_CDF_NO_POPULATION_TAIL_GUARANTEE")
    return row


def write_comparison(path: Path, rows, row_count=None):
    path.mkdir()
    with (path / CSV_NAME).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw = (path / CSV_NAME).read_bytes()
    changes = dict(lower=0, higher=0, equal=0)
    for row in rows:
        if row["variance_ratio_status"] != "UNDEFINED":
            a, b = float(row["native_sample_variance"]), float(row["coupled_sample_variance"])
            changes["lower" if b < a else "higher" if b > a else "equal"] += 1
    report = dict(
        status=PARENT_STATUS, row_count=len(rows) if row_count is None else row_count,
        original_independent_blocks=32, independent_circuit_instances=1,
        observed_simulator_cost_ratio=COST_RATIO, variance_change_rows=changes,
        cost_measure="Measured simulator and output-extraction proxy; excludes transfer.",
        interpretation="Descriptive 32-block comparison; no confirmation qualification.",
        outputs={CSV_NAME: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())},
        **{flag: False for flag in FALSE_FLAGS},
    )
    (path / "report.json").write_text(json.dumps(report))
    return path


def test_cli_full_denominator_cost_reversal_and_adverse_rows(tmp_path):
    rows = [
        endpoint(0, 1, .75),  # Raw improvement becomes cost-adjusted worsening.
        endpoint(1, 1, .25),
        endpoint(2, 0, .2, "count_cdf", "NATIVE_ZERO_COUPLED_NONZERO"),
        endpoint(3, 0, 0, "count_cdf", "BOTH_EMPIRICALLY_ZERO_UNCERTAIN"),
        endpoint(4, None, None, "field_lag_covariance", "UNDEFINED"),
        endpoint(5, 1, 1),
        endpoint(6, 0, 0, "paired_response", "BOTH_EMPIRICALLY_ZERO_UNCERTAIN"),
        endpoint(7, 1, 2),
        endpoint(8, 1, 0, "count_cdf"),
    ]
    source = write_comparison(tmp_path / "comparison", rows)
    output = tmp_path / "summary"
    assert main(["--comparison-dir", str(source), "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    total = report["global_summary"]
    assert total["row_count"] == 9
    assert sum(group["row_count"] for group in report["per_family"].values()) == 9
    assert total["variance_ratio_status_counts"] == {
        "DEFINED_RATIO": 5, "NATIVE_ZERO_COUPLED_NONZERO": 1,
        "BOTH_EMPIRICALLY_ZERO_UNCERTAIN": 2, "UNDEFINED": 1,
    }
    assert total["raw_variance_change_rows"] == dict(lower=3, higher=2, equal=3, undefined=1)
    raw, cost = total["variance_ratio"], total["variance_simulator_cost_ratio"]
    assert (raw["lower_than_one"], raw["equal_to_one"], raw["higher_than_one"]) == (3, 1, 1)
    assert (cost["lower_than_one"], cost["equal_to_one"], cost["higher_than_one"]) == (2, 0, 3)
    assert (cost["finite_defined_rows"], cost["excluded_nondefined_rows"]) == (5, 4)
    assert (cost["minimum"], cost["median"], cost["maximum"]) == (0, 1.125, 3)
    assert total["raw_variance_decrease_cost_adjusted_increase_rows"] == 1
    assert total["cost_adjusted_increase_without_raw_increase_rows"] == 2
    assert total["raw_variance_increase_rows"] == 2
    cdf = report["per_family"]["count_cdf"]
    assert sum(item["row_count"] for item in cdf["cdf_tail_status_transitions"]) == 3
    assert any(item["native_status"] == "EMPIRICAL_SATURATION_UNCERTAIN"
               and item["coupled_status"] == "OBSERVED_CDF_NO_POPULATION_TAIL_GUARANTEE"
               and item["row_count"] == 1 for item in cdf["cdf_tail_status_transitions"])
    with (output / "adverse_endpoint_rows.csv").open(newline="") as stream:
        adverse = {int(row["domain_row_index"]): row for row in csv.DictReader(stream)}
    assert set(adverse) == {0, 2, 5, 7}
    for index, row in adverse.items():
        assert all(row[key] == value for key, value in rows[index].items())
    assert adverse[0]["raw_variance_increased"] == "False"
    assert adverse[0]["cost_adjusted_variance_increased"] == "True"
    assert adverse[2]["raw_variance_increased"] == "True"
    assert adverse[2]["cost_adjusted_variance_increased"] == "False"
    with (output / "per_family_summary.csv").open(newline="") as stream:
        family_rows = list(csv.DictReader(stream))
    assert sum(int(row["row_count"]) for row in family_rows) == 9
    assert all(report[flag] is False for flag in FALSE_FLAGS)


def test_nonfinite_defined_ratios_are_reported_but_excluded_from_finite_statistics(tmp_path):
    source = write_comparison(tmp_path / "comparison", [endpoint(0, 5e-324, 1)])
    report = summarize_comparison(source, tmp_path / "summary")
    for name in ("variance_ratio", "variance_simulator_cost_ratio"):
        ratios = report["global_summary"][name]
        assert ratios["finite_defined_rows"] == 0
        assert ratios["nonfinite_defined_rows"] == 1
        assert ratios["higher_than_one"] == 0
        assert ratios["minimum"] is None and ratios["median"] is None and ratios["maximum"] is None
    assert report["global_summary"]["raw_variance_increase_rows"] == 1


@pytest.mark.parametrize("problem,error", [
    ("duplicate", "UNIQUE_DOMAIN_ROW_INDEX_REQUIRED"),
    ("truncated", "COMPLETE_PARENT_DENOMINATOR_REQUIRED"),
    ("tampered", "BOUND_COMPARISON_CSV_CHANGED"),
    ("not_complete", "COMPLETE_FIXED_BANK_COMPARISON_REQUIRED"),
    ("qualification", "DESCRIPTIVE_PARENT_FLAGS_REQUIRED"),
])
def test_invalid_or_incomplete_evidence_cannot_create_summary(tmp_path, problem, error):
    rows = [endpoint(0, 1, .5), endpoint(0 if problem == "duplicate" else 1, 1, .75)]
    source = write_comparison(tmp_path / "comparison", rows, row_count=3 if problem == "truncated" else None)
    if problem == "tampered":
        path = source / CSV_NAME
        path.write_bytes(path.read_bytes().replace(b"omission", b"strong_cue"))
    if problem in ("not_complete", "qualification"):
        path = source / "report.json"
        report = json.loads(path.read_text())
        if problem == "not_complete":
            report["status"] = "RUNNING"
        else:
            report["scientific_fidelity_qualified"] = True
        path.write_text(json.dumps(report))
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match=error):
        summarize_comparison(source, output)
    assert not output.exists()

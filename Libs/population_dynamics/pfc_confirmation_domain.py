"""Complete PFC confirmation endpoint identities without imputing estimates."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .provenance import freeze_json, sha256_file

GROUP_FIELDS = ("left_cohort", "right_cohort", "condition", "phase", "window_ms", "field_index")
NEW_STATUS = "PHYSICAL_THRESHOLD_UNESTIMATED"


def group_key(identity: dict) -> tuple[str, ...]:
    return tuple(str(identity[field]) for field in GROUP_FIELDS)


def complete_domain(domains: list[dict], support: list[dict]) -> tuple[list[dict], dict]:
    """Retain old rows verbatim and add all missing nontrivial integer thresholds.

    Neither a missing threshold nor equal observed CDFs establishes zero
    population discrepancy. New rows therefore have no numerical estimate.
    """
    result = [dict(row) for row in domains]
    identities = set()
    thresholds: dict[tuple[str, ...], set[int]] = {}
    maxima: dict[tuple[str, ...], int] = {}
    for row in domains:
        identity = json.loads(row["endpoint_identity"])
        unique = row["family"], json.dumps(identity, sort_keys=True)
        if unique in identities:
            raise ValueError("DUPLICATE_ENDPOINT_IDENTITY")
        identities.add(unique)
        if row["family"] == "count_cdf":
            key = group_key(identity)
            maximum = int(row["physical_maximum_count"])
            if key in maxima and maxima[key] != maximum:
                raise ValueError("INCONSISTENT_PHYSICAL_COUNT_BOUND")
            maxima[key] = maximum
            thresholds.setdefault(key, set()).add(int(identity["count_threshold"]))
    seen = set()
    added = 0
    for row in support:
        key = group_key(row)
        if key in seen or key not in thresholds:
            raise ValueError("SUPPORT_GROUP_DUPLICATED_OR_MISSING")
        seen.add(key)
        maximum = int(row["physical_maximum_count"])
        if maximum != maxima[key] or maximum < 0:
            raise ValueError("SUPPORT_BOUND_MISMATCH")
        present = thresholds[key]
        if min(present) != int(row["archived_threshold_minimum"]) or max(present) != int(row["archived_threshold_maximum"]):
            raise ValueError("ARCHIVED_THRESHOLD_ENVELOPE_MISMATCH")
        missing = sorted(set(range(maximum)) - present)
        if len(missing) != int(row["unrepresented_nontrivial_thresholds"]):
            raise ValueError("MISSING_THRESHOLD_COUNT_MISMATCH")
        if missing and missing[0] != int(row["first_unrepresented_threshold"]):
            raise ValueError("FIRST_MISSING_THRESHOLD_MISMATCH")
        for threshold in missing:
            identity = {field: row[field] for field in GROUP_FIELDS}
            identity["count_threshold"] = str(threshold)
            result.append({"family": "count_cdf", "endpoint_identity": json.dumps(identity, sort_keys=True),
                           "empirical_status": "NOT_IN_ARCHIVED_COMPARISON", "definition_status": NEW_STATUS,
                           "reason": "physical_support_requires_estimand; point_estimate_and_uncertainty_not_computed",
                           "physical_maximum_count": str(maximum), "estimate": ""})
        added += len(missing)
    if seen != set(thresholds):
        raise ValueError("COUNT_GROUP_MISSING_SUPPORT")
    return result, {"source_endpoint_rows": len(domains), "complete_endpoint_rows": len(result),
                    "new_nontrivial_cdf_thresholds": added, "cdf_support_groups": len(seen),
                    "definition_status_counts": dict(Counter(row["definition_status"] for row in result)),
                    "outside_domain_rule": "count threshold <0 gives CDF0; threshold>=source physical maximum gives CDF1; adjacent-clock contrast0",
                    "new_threshold_estimates_computed": False, "confidence_intervals_computed": False,
                    "numerical_margins_frozen": False, "confirmation_execution_released": False,
                    "scientific_fidelity_qualified": False, "production_episodes": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source_report = args.design_analysis / "report.json"
    source = json.loads(source_report.read_text())
    if source["status"] != "COMPLETE_DESCRIPTIVE_CONFIRMATION_DESIGN_INPUTS":
        raise ValueError("COMPLETED_CALIBRATION_DESIGN_INPUT_REQUIRED")
    tables = {}
    inputs = [{"name": "report.json", "sha256": sha256_file(source_report)}]
    for name in ("endpoint_domains.csv", "count_support.csv"):
        path = args.design_analysis / name
        digest = sha256_file(path)
        if digest != source["outputs"][name]["sha256"]:
            raise ValueError("CALIBRATION_DESIGN_INPUT_HASH_MISMATCH:" + name)
        with path.open(newline="", encoding="utf-8") as stream:
            tables[name] = list(csv.DictReader(stream))
        if len(tables[name]) != source["row_counts"][name]:
            raise ValueError("CALIBRATION_DESIGN_ROW_COUNT_MISMATCH:" + name)
        inputs.append({"name": name, "sha256": digest})
    rows, result = complete_domain(tables["endpoint_domains.csv"], tables["count_support.csv"])
    if result["new_nontrivial_cdf_thresholds"] != source["unrepresented_nontrivial_cdf_thresholds"]:
        raise ValueError("CALIBRATION_DESIGN_MISSING_DOMAIN_TOTAL_MISMATCH")
    args.output.mkdir(parents=True, exist_ok=False)
    table = args.output / "endpoint_domains.csv"
    with table.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result.update(status="COMPLETE_PHYSICAL_ENDPOINT_DOMAIN_ESTIMATES_PARTIAL",
                  source_analysis_at_execution=str(args.design_analysis),
                  inputs=inputs, source_calibration_episodes=source["calibration_episodes"],
                  new_source_simulations=0, output={"path": table.name, "sha256": sha256_file(table), "rows": len(rows)},
                  implementation_sha256=sha256_file(Path(__file__)))
    freeze_json(args.output / "report.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

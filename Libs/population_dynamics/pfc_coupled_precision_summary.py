"""Describe completed fixed-bank precision and its measured cost proxy.

This consumes only the bound comparison CSV. It does not estimate confidence
intervals, qualify coverage or fidelity, or release confirmation experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

PARENT_STATUS = "COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON"
CSV_NAME = "endpoint_precision_comparison.csv"
FALSE_FLAGS = (
    "scientific_fidelity_qualified", "uncertainty_coverage_qualified",
    "confidence_intervals", "confirmation_execution_released",
    "matched_end_to_end_speedup_qualified", "numerical_margins_changed",
)
RATIO_STATUSES = (
    "DEFINED_RATIO", "NATIVE_ZERO_COUPLED_NONZERO",
    "BOTH_EMPIRICALLY_ZERO_UNCERTAIN", "UNDEFINED",
)
STATUS_COLUMNS = (
    "definition_status", "variance_ratio_status", "native_empirical_status",
    "coupled_empirical_status", "native_cdf_tail_status", "coupled_cdf_tail_status",
)
RATIO_COLUMNS = ("variance_ratio", "variance_simulator_cost_ratio")
TRANSITIONS = ("empirical_status", "cdf_tail_status")
REQUIRED_COLUMNS = {
    "domain_row_index", "family", "endpoint_identity",
    "native_sample_variance", "coupled_sample_variance",
    "native_original_independent_blocks", "coupled_original_independent_blocks",
    "observed_simulator_cost_ratio", *STATUS_COLUMNS, *RATIO_COLUMNS,
}
ADVERSE_FLAGS = ("raw_variance_increased", "cost_adjusted_variance_increased")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _direction(value: float) -> str:
    return "lower" if value < 1 else "higher" if value > 1 else "equal"


class _Counts:
    def __init__(self) -> None:
        self.row_count = 0
        self.statuses = {name: Counter() for name in STATUS_COLUMNS}
        self.transitions = {name: Counter() for name in TRANSITIONS}
        self.raw_change = Counter({name: 0 for name in ("lower", "equal", "higher", "undefined")})
        self.ratios: dict[str, list[float]] = {name: [] for name in RATIO_COLUMNS}
        self.nonfinite_ratios = Counter()
        self.adverse_count = 0
        self.cost_only_increases = 0
        self.raw_decrease_cost_increases = 0

    def add(self, row: dict[str, str], raw_change: str, ratios: dict[str, float]) -> None:
        self.row_count += 1
        for name in STATUS_COLUMNS:
            self.statuses[name][row[name]] += 1
        for name in TRANSITIONS:
            self.transitions[name][row["native_" + name], row["coupled_" + name]] += 1
        self.raw_change[raw_change] += 1
        for name, value in ratios.items():
            if math.isfinite(value):
                self.ratios[name].append(value)
            else:
                self.nonfinite_ratios[name] += 1
        cost_increase = (
            "variance_simulator_cost_ratio" in ratios
            and math.isfinite(ratios["variance_simulator_cost_ratio"])
            and ratios["variance_simulator_cost_ratio"] > 1
        )
        self.adverse_count += raw_change == "higher" or cost_increase
        self.cost_only_increases += cost_increase and raw_change != "higher"
        self.raw_decrease_cost_increases += cost_increase and raw_change == "lower"

    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "row_count": self.row_count,
            "raw_variance_change_rows": dict(self.raw_change),
            "adverse_endpoint_rows": self.adverse_count,
            "raw_variance_increase_rows": self.raw_change["higher"],
            "cost_adjusted_increase_without_raw_increase_rows": self.cost_only_increases,
            "raw_variance_decrease_cost_adjusted_increase_rows": self.raw_decrease_cost_increases,
        }
        for name, counts in self.statuses.items():
            result[name + "_counts"] = dict(sorted(counts.items()))
        # Keep zero/undefined categories explicit even when a family has none.
        result["variance_ratio_status_counts"] = {
            status: self.statuses["variance_ratio_status"][status] for status in RATIO_STATUSES
        }
        for name, counts in self.transitions.items():
            result[name + "_transitions"] = [
                dict(native_status=native, coupled_status=coupled, row_count=count)
                for (native, coupled), count in sorted(counts.items())
            ]
        for name, values in self.ratios.items():
            directions = Counter(_direction(value) for value in values)
            result[name] = {
                "finite_defined_rows": len(values),
                "nonfinite_defined_rows": self.nonfinite_ratios[name],
                "excluded_nondefined_rows": self.row_count - self.statuses["variance_ratio_status"]["DEFINED_RATIO"],
                "lower_than_one": directions["lower"],
                "equal_to_one": directions["equal"],
                "higher_than_one": directions["higher"],
                "minimum": min(values) if values else None,
                "median": median(values) if values else None,
                "maximum": max(values) if values else None,
            }
        return result


def _row_values(row: dict[str, str], cost_ratio: float) -> tuple[str, dict[str, float]]:
    status = row["variance_ratio_status"]
    require(status in RATIO_STATUSES, "UNKNOWN_VARIANCE_RATIO_STATUS")
    require(float(row["observed_simulator_cost_ratio"]) == cost_ratio, "ROW_COST_RATIO_CHANGED")
    require(all(int(row[prefix + "_original_independent_blocks"]) == 32
                for prefix in ("native", "coupled")), "ORIGINAL_32_BLOCKS_REQUIRED")
    if status == "UNDEFINED":
        require(all(row[name] == "" for name in (
            "native_sample_variance", "coupled_sample_variance", *RATIO_COLUMNS)),
            "UNDEFINED_ROW_MUST_NOT_SUPPLY_VARIANCE_OR_RATIO")
        return "undefined", {}
    native, coupled = (float(row[name]) for name in ("native_sample_variance", "coupled_sample_variance"))
    require(all(math.isfinite(value) and value >= 0 for value in (native, coupled)),
            "FINITE_NONNEGATIVE_SAMPLE_VARIANCES_REQUIRED")
    change = "lower" if coupled < native else "higher" if coupled > native else "equal"
    if status == "DEFINED_RATIO":
        require(native > 0, "DEFINED_RATIO_REQUIRES_POSITIVE_NATIVE_VARIANCE")
        ratios = {name: float(row[name]) for name in RATIO_COLUMNS}
        require(ratios["variance_ratio"] == coupled / native, "VARIANCE_RATIO_CHANGED")
        require(ratios["variance_simulator_cost_ratio"] == ratios["variance_ratio"] * cost_ratio,
                "VARIANCE_COST_PRODUCT_CHANGED")
        return change, ratios
    require(native == 0 and (coupled == 0 if status == "BOTH_EMPIRICALLY_ZERO_UNCERTAIN" else coupled > 0),
            "ZERO_VARIANCE_CATEGORY_CONTRADICTED")
    require(all(row[name] == "" for name in RATIO_COLUMNS), "ZERO_NATIVE_VARIANCE_HAS_NO_DEFINED_RATIO")
    return change, {}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_comparison(comparison_dir: Path, output: Path) -> dict[str, Any]:
    """Write a descriptive summary only after validating the full bound CSV."""
    report_bytes = (comparison_dir / "report.json").read_bytes()
    parent = json.loads(report_bytes)
    require(parent["status"] == PARENT_STATUS, "COMPLETE_FIXED_BANK_COMPARISON_REQUIRED")
    require(all(parent.get(flag) is False for flag in FALSE_FLAGS), "DESCRIPTIVE_PARENT_FLAGS_REQUIRED")
    require(parent["original_independent_blocks"] == 32 and parent["independent_circuit_instances"] == 1,
            "ORIGINAL_32_BLOCK_ONE_CIRCUIT_BANK_REQUIRED")
    expected_rows = parent["row_count"]
    require(type(expected_rows) is int and expected_rows > 0, "POSITIVE_PARENT_ROW_COUNT_REQUIRED")
    cost_ratio = float(parent["observed_simulator_cost_ratio"])
    require(math.isfinite(cost_ratio) and cost_ratio > 0, "FINITE_POSITIVE_COST_RATIO_REQUIRED")
    binding = parent["outputs"][CSV_NAME]
    csv_path = comparison_dir / CSV_NAME
    digest = hashlib.sha256()
    byte_count = 0
    total, families, indices, adverse = _Counts(), {}, set(), []

    # Binary-line iteration hashes exactly the bytes consumed by csv, once.
    def lines(stream: Any) -> Iterable[str]:
        nonlocal byte_count
        for line in stream:
            digest.update(line)
            byte_count += len(line)
            yield line.decode("utf-8")

    with csv_path.open("rb") as stream:
        reader = csv.DictReader(lines(stream))
        columns = reader.fieldnames
        require(columns is not None and len(columns) == len(set(columns))
                and REQUIRED_COLUMNS.issubset(columns), "COMPLETE_UNIQUE_COMPARISON_COLUMNS_REQUIRED")
        require(not set(ADVERSE_FLAGS).intersection(columns), "ADVERSE_FLAG_COLUMN_COLLISION")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "COMPLETE_CSV_ROW_REQUIRED")
            index = int(row["domain_row_index"])
            require(index >= 0 and index not in indices, "UNIQUE_DOMAIN_ROW_INDEX_REQUIRED")
            indices.add(index)
            require(bool(row["family"]) and isinstance(json.loads(row["endpoint_identity"]), dict),
                    "ENDPOINT_FAMILY_AND_IDENTITY_REQUIRED")
            change, ratios = _row_values(row, cost_ratio)
            total.add(row, change, ratios)
            if row["family"] not in families:
                families[row["family"]] = _Counts()
            families[row["family"]].add(row, change, ratios)
            cost_increase = (
                "variance_simulator_cost_ratio" in ratios
                and math.isfinite(ratios["variance_simulator_cost_ratio"])
                and ratios["variance_simulator_cost_ratio"] > 1
            )
            if change == "higher" or cost_increase:
                adverse.append(dict(row, raw_variance_increased=change == "higher",
                                    cost_adjusted_variance_increased=cost_increase))
    require(total.row_count == expected_rows, "COMPLETE_PARENT_DENOMINATOR_REQUIRED")
    require(byte_count == binding["bytes"] and digest.hexdigest() == binding["sha256"],
            "BOUND_COMPARISON_CSV_CHANGED")
    require({key: total.raw_change[key] for key in ("lower", "higher", "equal")}
            == parent["variance_change_rows"], "PARENT_VARIANCE_COUNTS_DISAGREE")

    family_reports = {name: counts.report() for name, counts in sorted(families.items())}
    result = {
        "status": "COMPLETE_DESCRIPTIVE_COUPLED_PRECISION_SUMMARY",
        "comparison_dir": str(comparison_dir.resolve()),
        "comparison_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "comparison_csv": dict(path=CSV_NAME, bytes=byte_count, sha256=digest.hexdigest()),
        "parent_status": parent["status"],
        "original_independent_blocks": 32,
        "independent_circuit_instances": 1,
        "source_simulations": 0,
        "production_episodes": 0,
        "observed_simulator_cost_ratio": cost_ratio,
        "cost_measure": parent["cost_measure"],
        "parent_interpretation": parent["interpretation"],
        "interpretation": (
            "Descriptive endpoint counts and finite ratios only; exact comparisons to one have no inferential meaning. "
            "Zero empirical variance does not establish zero population uncertainty or qualified fidelity. "
            "Native and coupled share 32 original blocks, not 64; 496 overlapping covariance pairs are not independent samples. "
            "Cost is a simulator/input/full-unit-extraction/current-observation proxy, excluding queue, staging, "
            "serialization, transfer and separate statistics; no matched end-to-end speedup is claimed. "
            "Blank empirical/CDF labels are retained as blank categories. No frozen margins are changed or tested here."
        ),
        "global_summary": total.report(),
        "per_family": family_reports,
        "adverse_row_selection": (
            "Union of raw sample-variance increases (including native zero/coupled nonzero) and finite defined "
            "variance_simulator_cost_ratio > 1; reason flags distinguish these. All original row columns are preserved."
        ),
        "outputs": {
            "per_family_summary": "per_family_summary.csv",
            "adverse_endpoint_rows": "adverse_endpoint_rows.csv",
        },
        **{flag: False for flag in FALSE_FLAGS},
    }
    # All validation precedes output creation, avoiding plausible partial summaries.
    output.mkdir(parents=True, exist_ok=False)
    family_columns = ["family", *next(iter(family_reports.values()))]
    _write_csv(output / "per_family_summary.csv", (
        {"family": name, **{key: json.dumps(value, sort_keys=True, allow_nan=False)
                           if isinstance(value, (dict, list)) else value for key, value in values.items()}}
        for name, values in family_reports.items()), family_columns)
    _write_csv(output / "adverse_endpoint_rows.csv", adverse, [*columns, *ADVERSE_FLAGS])
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_comparison(args.comparison_dir, args.output)
    print(json.dumps({"status": report["status"], "row_count": report["global_summary"]["row_count"],
                      "adverse_endpoint_rows": report["global_summary"]["adverse_endpoint_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

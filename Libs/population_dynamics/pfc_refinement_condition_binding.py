"""Bind all archived PFC control arms and propagate only input-parameter bias.

The compared laws replace binary64 r by exact q/(1-q), with ideal conditional
binomial sampling. Neither finite sampling laws nor source confidence qualify.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import io
import json
from pathlib import Path

import yaml

from .pfc_delay_analysis import PHASES
from .pfc_phase_responses import CONDITIONS, METRICS
from .pfc_refinement_input_law import _rational, mean_parent_envelope
from .pfc_native_drive import _BINOMIAL_N, _RATE_HZ
from .pfc_refinement_source_binding import (
    SOURCE_CLOCKS, SUPPORTED_CODE_HASHES, _Reader, _entries, _exact,
    _integer, _number, _require, _validate_dimensions, _validate_driver,
)
from .pfc_input_pairing import _relative
from .provenance import freeze_json

LABELS = ("numpy_dt_0p05", "numpy_dt_0p025", "numpy_dt_0p0125")


def expectation_bias(width: Fraction, block_tv: Fraction, order: int = 1) -> Fraction:
    """Oscillation times product-law TV; order two requires independent blocks.

    Independent maximal couplings have no mismatch with probability at least
    (1-delta)**order. A bounded observable changes by at most its oscillation
    on the mismatch event. The full U statistic has the kernel expectation;
    its overlapping pair count does not multiply this bias.
    """
    _require(isinstance(width, Fraction) and width >= 0
             and isinstance(block_tv, Fraction) and 0 <= block_tv <= 1
             and type(order) is int and order in (1, 2), "VALID_EXPECTATION_BIAS_INPUTS_REQUIRED")
    return width * (1 - (1 - block_tv) ** order)


def _report(reader: _Reader, path: Path, digest: str) -> dict:
    data = reader.read(path)
    _require(reader.digest(path) == digest, "BOUND_REPORT_DIGEST_CHANGED:" + str(path))
    return json.loads(data)


def _closed(report: dict) -> None:
    for key in ("finite_prng_law_certified", "finite_sampler_law_certified",
                "source_history_confidence_certified", "confirmation_0p01_128_certified",
                "confirmation_execution_released", "scientific_fidelity_qualified",
                "production_qualified", "numerical_margins_changed", "error_budget_allocated"):
        _require(report.get(key) is False, "PRIOR_SCOPE_CHANGED:" + key)


def _case(report: dict, label: str, block: int, condition: str) -> dict:
    _require(report.get("source_execution_complete") is True, "COMPLETE_SOURCE_REPORT_REQUIRED")
    config = report["resolved_config"]
    _validate_dimensions(config, label)
    case = report["selected_case"]
    _require(case.get("case_id") == f"pfc_{block}_{condition}"
             and case.get("source_rng_seed") == block, "SOURCE_CONDITION_IDENTITY_CHANGED")
    _require(case in config["cases"], "DECLARED_SOURCE_CASE_REQUIRED")
    strength = {"omission": 0., "cue": .06, "strong_cue": .5}[condition]
    _require(_number(case["stimulus_strength_nA"]) == Fraction.from_float(strength)
             and _number(case["distractor_strength_nA"]) == 0, "SOURCE_CONTROL_CHANGED")
    ticks = round(5000 / SOURCE_CLOCKS[label])
    _integer(report["completed_duration_ticks"], ticks, "COMPLETE_SOURCE_CLOCK_REQUIRED")
    _integer(report["inputs_used"]["ticks_consumed"], ticks, "COMPLETE_INPUT_CLOCK_REQUIRED")
    for population, units in (("E", 80000), ("I", 20000)):
        _integer(report["inputs_used"][population]["values"], units * ticks,
                 "COMPLETE_SOURCE_POPULATION_INPUTS_REQUIRED")
    return case


def bind_conditions(reader: _Reader, prior: dict, precision: dict, root: Path) -> dict:
    """Supplement accepted strong-cue binding; do not reread neural payloads."""
    _require(prior.get("status") == "COMPLETE_INDEXED_PFC_REFINEMENT_PARAMETER_ARITHMETIC_BINDING"
             and prior.get("parameter_arithmetic_binding_verified") is True,
             "ACCEPTED_PARAMETER_BINDING_REQUIRED")
    _closed(prior)
    records = {(int(row["source_block"]), row["source_label"]): row
               for row in prior["source_bindings"]}
    expected = {(block, label) for block in range(95001, 95033) for label in SOURCE_CLOCKS}
    _require(set(records) == expected and len(prior["source_bindings"]) == 160,
             "ALL_ACCEPTED_ORIGINAL_METADATA_IDENTITIES_REQUIRED")
    native_refs = {}
    for row in precision["settings"]["native_source_reports"]:
        _require(row["cohort"] in LABELS, "KNOWN_NATIVE_REFERENCE_CLOCK_REQUIRED")
        if row["cohort"] != LABELS[0]:
            continue
        key = row["source_rng_seed"], row["case_id"]
        _require(key not in native_refs, "UNIQUE_NATIVE_SOURCE_REFERENCE_REQUIRED")
        native_refs[key] = row
    _require(set(native_refs) == {(block, f"pfc_{block}_{condition}")
                for block in range(95001, 95033) for condition in CONDITIONS},
             "ALL_THREE_NATIVE_CONTROL_ARMS_REQUIRED")
    p0 = Fraction.from_float(_RATE_HZ * (.05 * 1e-3))
    matched, all_records = 0, []
    for block in range(95001, 95033):
        native = {}
        for condition in CONDITIONS:
            ref = native_refs[block, f"pfc_{block}_{condition}"]
            path = root / _relative(ref["path"])
            report = _report(reader, path, ref["report_sha256"])
            _case(report, "native", block, condition)
            _require(report["status"] == "PFC_SOURCE_EXECUTION_COMPLETE", "NATIVE_EXECUTION_STATUS_CHANGED")
            _validate_driver(report["driver"], block, p0)
            native[condition] = report
            if condition == "strong_cue":
                _require(ref["report_sha256"] == records[block, "native"]["report_sha256"],
                         "ACCEPTED_NATIVE_REFERENCE_CHANGED")
            all_records.append(dict(source_block=block, source_label="native", condition=condition,
                                    path=str(path), report_sha256=reader.digest(path)))
        _require(all(report["driver"] == native["strong_cue"]["driver"] for report in native.values()),
                 "CONDITION_NATIVE_INPUT_DECLARATIONS_DIFFER")
        for label in SOURCE_CLOCKS:
            if label == "native":
                continue
            prior_record = records[block, label]
            archive = root / _relative(prior_record["archive"])
            delivery = _report(reader, archive / "delivery.json", prior_record["delivery_sha256"])
            entries = _entries(delivery)
            allowed = {"COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"} if label.endswith("B") else {
                "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT", "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK"}
            _require(delivery.get("verified") is True and delivery.get("status") in allowed,
                "COUPLED_SOURCE_DELIVERY_CHANGED")
            clock = "dt_" + str(SOURCE_CLOCKS[label]).replace(".", "p")
            drives = {}
            for condition in CONDITIONS:
                relative = f"source_cases/{clock}/pfc_{block}_{condition}/report.json"
                _require(relative in entries, "MISSING_DECLARED_CONDITION_REPORT:" + relative)
                path = archive / relative
                report = json.loads(reader.read(path, entries[relative]))
                _case(report, label, block, condition)
                _require(report["status"] == "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"
                         and all(report["code"].get(name) == digest
                                 for name, digest in SUPPORTED_CODE_HASHES.items()),
                         "CONDITION_COUPLING_IMPLEMENTATION_CHANGED")
                drive = report["drive"]
                factor = 2 if label.startswith("half") else 4
                ticks = round(5000 / SOURCE_CLOCKS[label])
                for key, value in (("refinement_factor", factor), ("ticks_generated", ticks),
                                   ("total_ticks", ticks), ("n_e", 80000), ("n_i", 20000)):
                    _integer(drive[key], value, "CONDITION_REFINEMENT_DIMENSIONS_CHANGED")
                _require(drive.get("all_refinement_levels_generated") is True
                         and drive["coupling"]["coarse_source"] == native[condition]["driver"],
                         "CONDITION_COARSE_DRIVER_CHANGED")
                _require(report["coarse_reference"]["report_sha256"]
                         == native_refs[block, f"pfc_{block}_{condition}"]["report_sha256"],
                         "CONDITION_COARSE_REFERENCE_CHANGED")
                if condition == "strong_cue":
                    _require(reader.digest(path) == prior_record["report_sha256"],
                             "ACCEPTED_COUPLED_REPORT_CHANGED")
                drives[condition] = drive
                all_records.append(dict(source_block=block, source_label=label, condition=condition,
                                        path=str(path), report_sha256=reader.digest(path)))
            _require(all(drive == drives["strong_cue"] for drive in drives.values()),
                     "CONDITION_REFINEMENT_INPUT_DECLARATIONS_DIFFER")
            matched += 1
    return dict(source_report_entries=len(all_records), added_condition_report_entries=320,
                condition_names=list(CONDITIONS), source_blocks=32,
                coupled_labels_with_equal_control_input_declarations=matched,
                shared_two_level_trees=prior["shared_two_level_trees"],
                neural_payloads_read=False, input_payload_identity_verified=False,
                condition_input_parameter_binding_verified=True, source_reports=all_records)


def summarize_bias(domains, margins, rates, metric_ranges, block_tv):
    """Retain every frozen row; summarize conditional bias without adopting it."""
    counts, maxima, seen, exceeded = Counter(), {}, set(), Counter()
    rows = 0
    for index, (domain, margin) in enumerate(zip(domains, margins, strict=True)):
        rows += 1
        family, status = domain["family"], domain["definition_status"]
        _require(int(margin["domain_row_index"]) == index and all(
            margin[key] == domain[key] for key in ("family", "endpoint_identity", "definition_status")),
            "FROZEN_DOMAIN_AND_MARGIN_IDENTITIES_DIFFER")
        identity = json.loads(domain["endpoint_identity"])
        _require((identity["left_cohort"], identity["right_cohort"]) in set(zip(LABELS[:-1], LABELS[1:])),
                 "FROZEN_CLOCK_PAIR_CHANGED")
        key = family, domain["endpoint_identity"]
        _require(key not in seen, "DUPLICATE_FROZEN_ENDPOINT"); seen.add(key)
        counts[status] += 1
        if status == "UNDEFINED_NO_WITHIN_PHASE_PAIRS":
            _require(family == "field_lag_covariance" and margin["margin"] == ""
                     and float(identity["lag_ms"]) >= PHASES[identity["phase"]][1] - PHASES[identity["phase"]][0],
                     "UNDEFINED_ENDPOINT_CHANGED")
            continue
        if status == "STRUCTURAL_ZERO_CONTRAST":
            structural = (family == "paired_response" and identity["phase"] == "baseline") or (
                family == "count_cdf" and int(identity["count_threshold"]) >= int(domain["physical_maximum_count"]))
            _require(structural and Fraction(margin["margin"]) == 0, "SOURCE_IDENTITY_CHANGED")
            continue
        _require(status in ("ESTIMATED_EMPIRICAL_SPREAD", "EMPIRICALLY_DEGENERATE_UNCERTAIN"),
                 "KNOWN_FROZEN_STATUS_REQUIRED")
        order = 1
        if family == "count_cdf":
            width = Fraction(2)
        elif family == "field_mean":
            width = 2 * rates[int(identity["field_index"])]
        elif family == "field_lag_covariance":
            leading, trailing = int(identity["leading_field"]), int(identity["trailing_field"])
            diagonal = leading == trailing and float(identity["lag_ms"]) == 0
            width = rates[leading] * rates[trailing] * (1 if diagonal else 2)
            order = 2
        else:
            _require(family in ("source_response", "paired_response"), "KNOWN_ENDPOINT_FAMILY_REQUIRED")
            lo, hi = metric_ranges[identity["metric"]]
            width = (hi - lo) * (4 if family == "paired_response" else 2)
        units = "probability" if family == "count_cdf" else "Hz^2" if order == 2 else "Hz"
        _require(margin["units"] == units, "FROZEN_ENDPOINT_UNITS_CHANGED")
        budget = Fraction(margin["margin"])
        _require(budget > 0, "POSITIVE_FROZEN_MARGIN_REQUIRED")
        bias = expectation_bias(width, block_tv, order)
        ratio = bias / budget
        if ratio > 1:
            exceeded[family] += 1
        if family not in maxima or ratio > maxima[family][0]:
            maxima[family] = ratio, dict(domain_row_index=index, endpoint_identity=identity,
                estimand_order=order, physical_score_or_kernel_oscillation=_rational(width),
                conditional_parameter_bias_upper=_rational(bias), units=units,
                frozen_margin=_rational(budget), conditional_bias_over_frozen_margin=_rational(ratio))
    return dict(domain_rows=rows, definition_status_counts=dict(counts),
                maximum_conditional_bias_ratio_by_family={key: value[1] for key, value in sorted(maxima.items())},
                conditional_parameter_bias_bounds_exceeding_margin_by_family=dict(exceeded),
                undefined_rows_have_no_bias_estimate=True, structural_identity_bias=0,
                bias_subtracted_from_frozen_budget=False, source_confidence_bands_generated=False)


def run(binding_path: Path, binding_digest: str, config_path: Path, root: Path, output: Path) -> dict:
    _require(not output.exists(), "FRESH_CONDITION_BINDING_OUTPUT_REQUIRED")
    reader = _Reader()
    prior = _report(reader, binding_path, binding_digest)
    config = yaml.safe_load(reader.read(config_path))
    _require(config.get("schema") == "pfc_coupled_uncertainty_diagnostic_v1"
             and config.get("confirmation_execution_released") is False
             and config.get("method_adopted_for_confirmation") is False
             and config.get("coverage_qualified") is False and config.get("planning_blocks") == 128,
             "UNCHANGED_DEVELOPMENT_CONFIG_REQUIRED")
    reports, roots = {}, {}
    for name in ("coupled", "domain", "budget", "linear"):
        ref = config["inputs"][name]
        roots[name] = root / _relative(ref["path"])
        reports[name] = _report(reader, roots[name] / "report.json", ref["report_sha256"])
    _require(reports["budget"].get("numerical_margins_frozen") is True
             and reports["budget"]["planning_blocks"] == 128
             and reports["budget"]["settings"]["error_budget"]["count_cdf_absolute_probability"] == .01,
             "UNCHANGED_FROZEN_ERROR_BUDGET_REQUIRED")
    family = bind_conditions(reader, prior, reports["coupled"], root)
    domain_path = roots["domain"] / "endpoint_domains.csv"
    domain_bytes = reader.read(domain_path, reports["domain"]["outputs"][domain_path.name])
    _require(reader.digest(domain_path) == reports["linear"]["domain_table_sha256"],
             "PHYSICAL_SUPPORT_DOMAIN_CHANGED")
    margin_path = roots["budget"] / "frozen_numerical_margins.csv"
    margin_bytes = reader.read(margin_path, reports["budget"]["outputs"][margin_path.name])
    rates = list(map(_number, reports["linear"]["field_rate_upper_bounds_hz"]))
    ranges = {key: tuple(map(_number, value)) for key, value in
              reports["linear"]["per_clock_response_physical_ranges_hz"].items()}
    _require(len(rates) == 17 and all(rate > 0 for rate in rates) and set(ranges) == set(METRICS),
             "FULL_FROZEN_PHYSICAL_SUPPORT_REQUIRED")
    block_tv = _exact(prior["per_block_two_A_B_tree_tv_upper"])
    summary = summarize_bias(csv.DictReader(io.StringIO(domain_bytes.decode("utf-8"))),
                             csv.DictReader(io.StringIO(margin_bytes.decode("utf-8"))), rates, ranges, block_tv)
    _require(summary["domain_rows"] == 84330
             and summary["definition_status_counts"]["UNDEFINED_NO_WITHIN_PHASE_PAIRS"] == 1734,
             "FULL_FROZEN_DOMAIN_REQUIRED")
    moment = mean_parent_envelope(_RATE_HZ * (.05 * 1e-3), trials=_BINOMIAL_N,
                                  units=100000, parent_ticks=100000, levels=2)
    moment_block_tv = min(Fraction(1), 2 * _exact(moment["full_history_tv_upper"]))
    moment["per_condition_two_tree_block_tv_upper"] = _rational(moment_block_tv)
    moment["frozen_endpoint_bias"] = summarize_bias(
        csv.DictReader(io.StringIO(domain_bytes.decode("utf-8"))),
        csv.DictReader(io.StringIO(margin_bytes.decode("utf-8"))), rates, ranges, moment_block_tv)
    moment["native_parent_mean_assumption_adopted_for_confirmation"] = False
    reader.read(Path(__file__))
    input_law_path = Path(__file__).with_name("pfc_refinement_input_law.py")
    reader.read(input_law_path)
    report = dict(status="COMPLETE_THREE_CONDITION_PARAMETER_BINDING_AND_CONDITIONAL_ENDPOINT_BIAS",
        scope="ideal_conditional_sampling_r_parameter_replacement_only",
        comparison_changes_only=prior["comparison_changes_only"], condition_binding=family,
        per_condition_block_input_tv_upper=_rational(block_tv), frozen_endpoint_bias=summary,
        conditional_native_parent_moment_scenario=moment,
        proof=dict(bounded_mean="oscillation(g) * delta",
                   covariance_kernel="oscillation(H) * (2*delta-delta**2)",
                   independent_source_blocks_required_for_order_two=True,
                   independence_proven_by_seed_labels=False,
                   paired_response="sum of the two signed condition-mean biases; no condition independence required",
                   overlapping_pairs_counted_as_independent=False,
                   shared_control_input_law_assumed_for_joint_confidence=False,
                   primitive_input_sampling_ideal=True,
                   physical_support_valid_for_both_compared_laws_required=True,
                   all_other_inputs_and_simulator_maps_unchanged_required=True),
        finite_prng_law_certified=False, finite_sampler_law_certified=False,
        source_history_confidence_certified=False, full_source_runtime_qualified=False,
        coarse_fine_neuronal_equality_certified=False, confirmation_0p01_128_certified=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        production_qualified=False, numerical_margins_changed=False, error_budget_allocated=False,
        new_trials=0, source_simulations=0, new_episodes=0, production_episodes=0,
        provenance=dict(prior_parameter_binding_sha256=binding_digest,
                        explicit_paths_read=len(reader.snapshots), bytes_read=reader.total_bytes,
                        consumed_artifacts=reader.records()))
    report["provenance"]["implementation_sha256"] = reader.digest(Path(__file__))
    report["provenance"]["input_law_implementation_sha256"] = reader.digest(input_law_path)
    freeze_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.binding, args.binding_sha256, args.config, args.project_root, args.output)
    print(json.dumps({key: report[key] for key in ("status", "scope", "source_simulations", "production_episodes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

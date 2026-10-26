"""Compare independent measurement designs on the same archived PFC blocks.

All results are selection-motivated development diagnostics. The identities
assume independent sampling and inner randomization conditional on each fixed
frame. They do not guarantee coverage conditional on the earlier pilot decision.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import yaml

from .pfc_first_passage_history_summary import between_components, bound_json

FIELDS = ("sample_mean", "within_archive_mean_variance_diagnostic",
          "inner_mc_mean_variance_diagnostic", "unit_window_sampling_mean_variance_diagnostic")


def require(condition, message):
    if not condition:
        raise ValueError("PFC_HISTORY_DESIGN_COMPARISON_" + message)


def compare(uniform, stratified):
    require(len(uniform) == len(stratified) > 1, "PAIRED_BLOCK_DENOMINATOR")
    for rows in (uniform, stratified):
        for row in rows:
            require(all(np.isfinite(row[key]) for key in FIELDS)
                    and row[FIELDS[1]] >= 0 and row[FIELDS[2]] >= 0
                    and np.isclose(row[FIELDS[1]], row[FIELDS[2]] + row[FIELDS[3]], rtol=1e-12, atol=0),
                    "MEASUREMENT_VARIANCE_COMPONENTS")
    u = np.array([row["sample_mean"] for row in uniform])
    s = np.array([row["sample_mean"] for row in stratified])
    n = len(u)
    covariance = float(np.sum((u-u.mean())*(s-s.mean()))/(n-1))
    differences, pooled, rows = [], [], []
    for left, right in zip(uniform, stratified):
        difference = {"sample_mean": right["sample_mean"]-left["sample_mean"]}
        difference.update({key: left[key]+right[key] for key in FIELDS[1:]})
        average = {"sample_mean": (right["sample_mean"]+left["sample_mean"])/2}
        average.update({key: difference[key]/4 for key in FIELDS[1:]})
        differences.append(difference); pooled.append(average)
        rows.append(dict(uniform_mean=left["sample_mean"], stratified_mean=right["sample_mean"],
                         design_difference=difference["sample_mean"], equal_weight_mean=average["sample_mean"],
                         difference_measurement_variance=difference[FIELDS[1]],
                         pooled_measurement_variance=average[FIELDS[1]],
                         squared_difference_minus_predicted_noise=difference["sample_mean"]**2-difference[FIELDS[1]]))
    d, m = between_components(differences), between_components(pooled)
    identity_rhs = covariance + d["latent_archived_block_mean_variance_diagnostic"]/4
    identity_residual = m["latent_archived_block_mean_variance_diagnostic"]-identity_rhs
    scale = max(abs(identity_rhs), abs(covariance), d["observed_between_block_variance"], np.finfo(float).tiny)
    require(abs(identity_residual) <= 64*np.finfo(float).eps*scale, "COVARIANCE_IDENTITY")
    baseline = {"uniform": between_components(uniform), "stratified": between_components(stratified)}
    lower = all(m["mean_within_archive_variance_diagnostic"] < value["mean_within_archive_variance_diagnostic"]
                for value in baseline.values())
    return dict(original_blocks=n, cross_design_sample_covariance=covariance,
                design_difference=d, equal_weight_pool=m, original_designs=baseline,
                covariance_identity_residual=identity_residual,
                mean_squared_difference_minus_predicted_noise=float(np.mean([r["squared_difference_minus_predicted_noise"] for r in rows])),
                pooled_reported_measurement_variance_lower_than_both=lower,
                pooling_weights={"uniform": .5, "stratified": .5},
                decision="USE_EQUAL_WEIGHT_POOL_AS_DEVELOPMENT_BASELINE" if lower else "RETAIN_SEPARATE_DEVELOPMENT_BASELINES",
                decision_scope="Uses all valid estimates with fixed weights; does not certify agreement, precision coverage or source qualification",
                negative_covariance_and_variance_diagnostics_preserved=True,
                overlapping_ids_scope="Independent selections and independent inner draws give zero cross-design error covariance conditional on fixed frame values, even when IDs overlap",
                conditioning_scope="Declared randomization-law identities; not conditional guarantees after the favorable pilot decision",
                population_confidence_interval_computed=False, significance_test_computed=False,
                scientific_fidelity_qualified=False, confirmation_execution_released=False,
                production_episodes=0, new_independent_source_blocks=0, new_neural_trajectories=0,
                new_inner_trials=0, third_spike_probability_established=False), rows


def compare_allocation(uniform, stratified, allocation):
    """Compare T with the fixed M=(U+S)/2 baseline without selecting another pool."""
    original, _ = compare(uniform, stratified)
    baseline = [{"sample_mean": (right["sample_mean"] + left["sample_mean"])/2,
                 **{key: (left[key] + right[key])/4 for key in FIELDS[1:]}}
                for left, right in zip(uniform, stratified)]
    comparison, _ = compare(baseline, allocation)
    m, t = comparison["original_designs"]["uniform"], comparison["original_designs"]["stratified"]
    require(m == original["equal_weight_pool"], "FIXED_BASELINE_ARITHMETIC")
    components = {}
    for key in FIELDS[1:]:
        left = float(np.mean([row[key] for row in baseline]))
        right = float(np.mean([row[key] for row in allocation]))
        components[key] = dict(fixed_baseline_M=left, allocation_T=right,
                               allocation_over_baseline=None if left <= 0 else right/left)
    rows = []
    for u, s, left, right in zip(uniform, stratified, baseline, allocation):
        difference = {"sample_mean": right["sample_mean"]-left["sample_mean"],
                      **{key: left[key]+right[key] for key in FIELDS[1:]}}
        row = {label+"_"+key: values[key] for label, values in
               (("uniform_U", u), ("stratified_S", s), ("fixed_baseline_M", left),
                ("allocation_T", right), ("allocation_minus_baseline", difference)) for key in FIELDS}
        row["squared_difference_minus_predicted_noise"] = difference["sample_mean"]**2-difference[FIELDS[1]]
        rows.append(row)
    covariance = comparison["cross_design_sample_covariance"]
    difference = comparison["design_difference"]
    residual = difference["latent_archived_block_mean_variance_diagnostic"]-(
        m["latent_archived_block_mean_variance_diagnostic"]+t["latent_archived_block_mean_variance_diagnostic"]-2*covariance)
    return dict(original_blocks=len(baseline), fixed_baseline_M=m, allocation_T=t,
                allocation_minus_baseline=difference, cross_design_sample_covariance=covariance,
                cross_covariance_designs=["fixed_baseline_M", "allocation_T"],
                difference_covariance_identity_residual=residual, mean_block_variance_components=components,
                mean_squared_difference_minus_predicted_noise=comparison["mean_squared_difference_minus_predicted_noise"],
                fixed_baseline_weights={"uniform": .5, "stratified": .5}, baseline_definition="M=(U+S)/2",
                decision="NO_NEW_BASELINE_SELECTION", new_pool_selected=False,
                ratio_scope="Component ratios require a positive baseline denominator; all signed components remain recorded",
                conditioning_scope="Fresh T randomization and a common archived target; pilot-motivated design choice does not establish conditional coverage",
                negative_covariance_and_variance_diagnostics_preserved=True,
                population_confidence_interval_computed=False, significance_test_computed=False,
                scientific_fidelity_qualified=False, confirmation_execution_released=False,
                production_episodes=0, new_independent_source_blocks=0, new_neural_trajectories=0,
                new_inner_trials=0, third_spike_probability_established=False), rows


def read_accepted_report(project, spec, statuses):
    root = Path(project)/spec["archive"]
    payload = (root/"delivery.json").read_bytes()
    require(hashlib.sha256(payload).hexdigest() == spec["delivery_sha256"], "DELIVERY_CHANGED")
    delivery = json.loads(payload)
    require(delivery["verified"] is True and delivery["status"] in statuses, "ACCEPTED_DESIGN_DELIVERY")
    entries = {row["path"]: row for row in delivery["files"]}
    wrapper = bound_json(root, entries, "report.json")
    report = bound_json(root, entries, "analysis/report.json")
    require(wrapper["status"] == delivery["status"] and wrapper["exit_code"] == 0
            and wrapper.get("error") is None and report.get("error") is None
            and report["population_confidence_interval_computed"] is False
            and report["scientific_fidelity_qualified"] is False
            and report["confirmation_execution_released"] is False, "DESIGN_STATUS_OR_SCOPE")
    return root, entries, report


def read_bound_table(root, entries, relative):
    table = (root/relative).read_bytes()
    require(len(table) == entries[relative]["bytes"]
            and hashlib.sha256(table).hexdigest() == entries[relative]["sha256"], "TABLE_CHANGED")
    rows = list(csv.DictReader(StringIO(table.decode())))
    require([int(row["source_rng_seed"]) for row in rows] == list(range(95001, 95033)), "ALL32_SOURCE_IDENTITIES")
    return rows


def read_design(project, spec, kind, *, include_report=False):
    statuses = {"uniform": ("COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE",),
                "stratified": ("COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_PREPARE",
                               "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE")}[kind]
    root, entries, report = read_accepted_report(project, spec, statuses)
    if kind == "uniform":
        require(report["status"] == "COMPLETE_PFC_PAIRED_HISTORY_DIAGNOSTICS"
                and report["available_original_block_pairs"] == report["required_original_block_pairs"] == 32
                and report["missing_receipts"] == [], "COMPLETE_UNIFORM32")
        relative, prefix, expected_stats = "analysis/paired_block_diagnostics.csv", "", report["statistics"]
    else:
        require(report["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY"
                and report["available_original_blocks"] == report["required_original_blocks"] == 32
                and report["missing_receipt_source_seeds"] == report["unexamined_source_seeds"] == [], "COMPLETE_STRATIFIED32")
        relative, prefix, expected_stats = "analysis/block_diagnostics.csv", "paired_", report["statistics"]["paired"]
    source_rows = read_bound_table(root, entries, relative)
    rows = [{key: float(row[prefix+key]) for key in FIELDS} for row in source_rows]
    computed = between_components(rows)
    require(all(computed[key] == expected_stats[key] for key in (
        "mean", "observed_between_block_variance", "mean_within_archive_variance_diagnostic",
        "latent_archived_block_mean_variance_diagnostic", "empirical_mean_variance_diagnostic",
        "inner_mc_mean_variance_diagnostic", "unit_window_sampling_mean_variance_diagnostic")), "SUMMARY_TABLE_ARITHMETIC")
    return (rows, report) if include_report else rows


def verify_baseline(project, spec, designs, expected, expected_rows):
    root, entries, report = read_accepted_report(project, spec, (
        "COMPLETE_PFC_HISTORY_DESIGN_COMPARISON_PREPARE", "COMPLETE_PFC_HISTORY_DESIGN_COMPARISON_COMPLETE"))
    require(report["status"] == "COMPLETE_PFC_HISTORY_DESIGN_COMPARISON" and report["original_blocks"] == 32
            and report["inputs"] == designs
            and report["decision"] == "USE_EQUAL_WEIGHT_POOL_AS_DEVELOPMENT_BASELINE"
            and all(report[key] == value for key, value in expected.items()), "ACCEPTED_FIXED_BASELINE_CHANGED")
    table = read_bound_table(root, entries, "analysis/block_design_comparison.csv")
    require(all(all(float(actual[key]) == value for key, value in row.items())
                for actual, row in zip(table, expected_rows)), "ACCEPTED_BASELINE_TABLE_CHANGED")
    return report


def cost_records(report):
    # Retain explicitly reported timing fields, including stage/delivery fields
    # if the accepted reader supplied them. Do not infer scheduler elapsed time.
    keys = {"totals", "cost_scope", "pair_analysis_seconds", "wrapper_seconds", "source_stage_seconds",
            "staging_seconds", "delivery_seconds", "scheduler_elapsed_seconds", "end_to_end_seconds"}
    records = {key: value for key, value in report.items() if key in keys}
    measured = any(key in records for key in keys-{"cost_scope"})
    return dict(status="reported" if measured else "unreported", records=records)


def run(project, settings, output):
    project, output = Path(project), Path(output)
    require(settings["schema"] == "pfc_history_design_comparison_v1"
            and settings["required_source_seeds"] == list(range(95001,95033))
            and settings["pooling_weights"] == {"uniform": .5, "stratified": .5}, "FIXED32_DESIGN")
    optional = ("accepted_allocation" in settings, "accepted_baseline_comparison" in settings)
    require(optional[0] == optional[1], "ALLOCATION_AND_ACCEPTED_BASELINE_REQUIRED_TOGETHER")
    if not optional[0]:
        designs = {kind: read_design(project, spec, kind) for kind, spec in settings["accepted_designs"].items()}
        report, rows = compare(designs["uniform"], designs["stratified"])
        report.update(status="COMPLETE_PFC_HISTORY_DESIGN_COMPARISON", inputs=settings["accepted_designs"])
        filename = "block_design_comparison.csv"
    else:
        designs = {kind: read_design(project, spec, kind, include_report=True)
                   for kind, spec in settings["accepted_designs"].items()}
        uniform, u_report = designs["uniform"]
        stratified, s_report = designs["stratified"]
        baseline, baseline_rows = compare(uniform, stratified)
        accepted = verify_baseline(project, settings["accepted_baseline_comparison"], settings["accepted_designs"], baseline, baseline_rows)
        allocation, t_report = read_design(project, settings["accepted_allocation"], "stratified", include_report=True)
        profile = dict(sample_size=640, inner_samples=10000, integration_seed_namespace=4, nominal_inner_trial_budget_per_clock=6400000)
        require(t_report.get("sampling_profile") == profile, "ACCEPTED_ALLOCATION_PROFILE")
        old_profile = dict(sample_size=64, inner_samples=100000, integration_seed_namespace=3, nominal_inner_trial_budget_per_clock=6400000)
        require(s_report.get("sampling_profile", old_profile) == old_profile, "ACCEPTED_STRATIFIED_BASELINE_PROFILE")
        require(t_report["source_census"] == s_report["source_census"], "ALLOCATION_SOURCE_CENSUS_CHANGED")
        require(t_report["uniform_comparison"]["reference"] == s_report["uniform_comparison"]["reference"]
                == settings["accepted_designs"]["uniform"], "ALLOCATION_UNIFORM_REFERENCE_CHANGED")
        report, rows = compare_allocation(uniform, stratified, allocation)
        require(report["fixed_baseline_M"] == accepted["equal_weight_pool"], "FIXED_BASELINE_STATISTICS_CHANGED")
        costs = {label: cost_records(value) for label, value in
                 (("uniform_U", u_report), ("stratified_S", s_report), ("allocation_T", t_report))}
        absent = [label for label in ("uniform_U", "stratified_S") if costs[label]["status"] == "unreported"]
        costs["fixed_baseline_M"] = dict(definition="U+S acquisition costs at the same accounting level",
                                        status="unreported" if absent else "available_as_U_plus_S_components",
                                        components=["uniform_U", "stratified_S"], unreported_components=absent)
        costs.update(matched_runtime_comparison=False,
                     scope="Accepted source cost records are retained. Wrapper timing is not scheduler end-to-end time; unreported staging/delivery overhead remains unreported.")
        report.update(status="COMPLETE_PFC_HISTORY_DESIGN_COMPARISON", comparison_kind="fixed_baseline_M_vs_allocation_T",
                      inputs=settings["accepted_designs"], accepted_allocation=settings["accepted_allocation"],
                      accepted_baseline_comparison=settings["accepted_baseline_comparison"],
                      accepted_baseline_decision=accepted["decision"], sampling_profile=profile,
                      source_census=t_report["source_census"], costs=costs)
        filename = "block_allocation_comparison.csv"
        report["block_allocation_comparison_file"] = filename
    output.mkdir(parents=True, exist_ok=False)
    with (output/filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_rng_seed", *rows[0]])
        writer.writeheader()
        writer.writerows(dict(source_rng_seed=seed, **row) for seed, row in zip(settings["required_source_seeds"], rows))
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "project-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project_root, yaml.safe_load(args.config.read_text()), args.output)
    print(json.dumps({key: result[key] for key in ("status", "cross_design_sample_covariance", "decision")}))


if __name__ == "__main__":
    main()

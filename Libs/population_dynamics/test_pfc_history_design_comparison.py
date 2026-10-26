"""Exact independent measurement-noise checks; no neural simulation."""
from itertools import product
from copy import deepcopy
import csv
import hashlib
from io import StringIO
import json
import numpy as np
import pytest
from .pfc_history_design_comparison import compare, compare_allocation, run, FIELDS
from .pfc_first_passage_history_summary import between_components


def records(means, variances):
    return [dict(sample_mean=float(mean), within_archive_mean_variance_diagnostic=float(variance),
                 inner_mc_mean_variance_diagnostic=float(variance)/4,
                 unit_window_sampling_mean_variance_diagnostic=3*float(variance)/4)
            for mean, variance in zip(means, variances)]


def test_independent_measurement_enumeration_removes_only_measurement_noise():
    latent = np.array([.25, .5, .75])
    au, ass = np.array([.1, .2, .3]), np.array([.2, .1, .4])
    results = []
    for signs in product((-1, 1), repeat=6):
        u = latent + au*np.asarray(signs[:3])
        s = latent + ass*np.asarray(signs[3:])
        result, _ = compare(records(u, au**2), records(s, ass**2))
        results.append(result)
    expected = np.var(latent, ddof=1)
    assert np.mean([r["cross_design_sample_covariance"] for r in results]) == pytest.approx(expected)
    assert np.mean([r["design_difference"]["latent_archived_block_mean_variance_diagnostic"] for r in results]) == pytest.approx(0, abs=1e-15)
    assert np.mean([r["equal_weight_pool"]["latent_archived_block_mean_variance_diagnostic"] for r in results]) == pytest.approx(expected)
    assert all(abs(r["covariance_identity_residual"]) < 1e-15 for r in results)


def test_negative_covariance_and_corrections_are_retained():
    result, rows = compare(records([1,2,3], [1,1,1]), records([3,2,1], [1,1,1]))
    assert result["cross_design_sample_covariance"] == -1
    assert result["equal_weight_pool"]["latent_archived_block_mean_variance_diagnostic"] == -.5
    assert result["design_difference"]["latent_archived_block_mean_variance_diagnostic"] == 2
    assert [row["design_difference"] for row in rows] == [2,0,-2]
    assert result["pooling_weights"] == {"uniform":.5,"stratified":.5}
    assert result["decision"] == "USE_EQUAL_WEIGHT_POOL_AS_DEVELOPMENT_BASELINE"
    assert not result["population_confidence_interval_computed"]


def test_invalid_denominators_or_noise_decomposition_fail():
    with pytest.raises(ValueError, match="PAIRED_BLOCK_DENOMINATOR"):
        compare(records([1], [1]), records([1,2], [1,1]))
    bad = records([1,2], [1,1]);bad[0]["inner_mc_mean_variance_diagnostic"] = 2
    with pytest.raises(ValueError, match="MEASUREMENT_VARIANCE_COMPONENTS"):
        compare(bad, records([1,2], [1,1]))
    bad = records([1,2], [1,1]);bad[0]["sample_mean"] = float("nan")
    with pytest.raises(ValueError, match="MEASUREMENT_VARIANCE_COMPONENTS"):
        compare(bad, records([1,2], [1,1]))


def test_fixed_baseline_allocation_arithmetic_preserves_signed_components():
    uniform = records([1,2,3], [1,1,1])
    stratified = records([3,2,1], [1,1,1])
    allocation = records([3,2,1], [.25,.25,.25])
    for row in allocation:
        row["inner_mc_mean_variance_diagnostic"] = .5
        row["unit_window_sampling_mean_variance_diagnostic"] = -.25
    result, rows = compare_allocation(uniform, stratified, allocation)
    assert result["fixed_baseline_M"]["mean"] == 2.
    assert result["fixed_baseline_M"]["latent_archived_block_mean_variance_diagnostic"] == -.5
    assert result["allocation_T"]["latent_archived_block_mean_variance_diagnostic"] == .75
    assert result["cross_design_sample_covariance"] == 0.
    assert result["allocation_minus_baseline"]["latent_archived_block_mean_variance_diagnostic"] == .25
    components = result["mean_block_variance_components"]
    assert components[FIELDS[1]]["allocation_over_baseline"] == .5
    assert components[FIELDS[3]]["allocation_T"] == -.25
    assert components[FIELDS[3]]["allocation_over_baseline"] == pytest.approx(-2/3)
    assert [row["fixed_baseline_M_sample_mean"] for row in rows] == [2.,2.,2.]
    assert [row["allocation_minus_baseline_sample_mean"] for row in rows] == [1.,0.,-1.]
    assert result["fixed_baseline_weights"] == {"uniform": .5, "stratified": .5}
    assert result["decision"] == "NO_NEW_BASELINE_SELECTION" and result["new_pool_selected"] is False
    assert "equal_weight_pool" not in result and "pooling_weights" not in result
    assert result["population_confidence_interval_computed"] is False


def _archive(root, status, report, table_name, table):
    root.mkdir(parents=True, exist_ok=True)
    content = StringIO()
    writer = csv.DictWriter(content, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(table)
    files = {"report.json": json.dumps(dict(status=status, exit_code=0)).encode(),
             "analysis/report.json": json.dumps(report).encode(), table_name: content.getvalue().encode()}
    entries = []
    for name, payload in files.items():
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        entries.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    payload = json.dumps(dict(status=status, verified=True, files=entries)).encode()
    (root/"delivery.json").write_bytes(payload)
    return dict(archive=root.name, delivery_sha256=hashlib.sha256(payload).hexdigest())


def _reseal_report(root, report):
    payload = json.dumps(report).encode()
    (root/"analysis/report.json").write_bytes(payload)
    delivery = json.loads((root/"delivery.json").read_text())
    entry = next(row for row in delivery["files"] if row["path"] == "analysis/report.json")
    entry.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    encoded = json.dumps(delivery).encode()
    (root/"delivery.json").write_bytes(encoded)
    return dict(archive=root.name, delivery_sha256=hashlib.sha256(encoded).hexdigest())


def test_bound_baseline_profile_and_full32_allocation_requirements(tmp_path):
    seeds = list(range(95001,95033))
    uniform = records(np.arange(32)/32, np.ones(32))
    stratified = records(np.arange(32)/32+.25, np.ones(32))
    allocation = records(np.arange(32)/32+.125+np.where(np.arange(32)%2, -.015625, .015625), np.full(32,.25))
    scope = dict(population_confidence_interval_computed=False, scientific_fidelity_qualified=False,
                 confirmation_execution_released=False)
    u_report = dict(scope, status="COMPLETE_PFC_PAIRED_HISTORY_DIAGNOSTICS", available_original_block_pairs=32,
                    required_original_block_pairs=32, missing_receipts=[], statistics=between_components(uniform))
    u_spec = _archive(tmp_path/"uniform", "COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE", u_report,
                      "analysis/paired_block_diagnostics.csv", [dict(source_rng_seed=seed, **row) for seed,row in zip(seeds,uniform)])
    source = dict(archive="accepted_census", delivery_sha256="c"*64)
    s_report = dict(scope, status="COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY", available_original_blocks=32,
                    required_original_blocks=32, missing_receipt_source_seeds=[], unexamined_source_seeds=[],
                    statistics={"paired": between_components(stratified)}, source_census=source,
                    uniform_comparison={"reference":u_spec}, totals={"coarse":{"inner_trials":3200000}},
                    pair_analysis_seconds=1000., wrapper_seconds=1010., cost_scope="Recorded analysis and wrapper costs")
    s_spec = _archive(tmp_path/"stratified", "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_PREPARE", s_report,
                      "analysis/block_diagnostics.csv", [dict(source_rng_seed=seed, **{"paired_"+key:value for key,value in row.items()})
                                                         for seed,row in zip(seeds,stratified)])
    designs = dict(uniform=u_spec, stratified=s_spec)
    accepted, table = compare(uniform, stratified)
    accepted.update(status="COMPLETE_PFC_HISTORY_DESIGN_COMPARISON", inputs=designs)
    baseline_spec = _archive(tmp_path/"baseline", "COMPLETE_PFC_HISTORY_DESIGN_COMPARISON_PREPARE", accepted,
                             "analysis/block_design_comparison.csv", [dict(source_rng_seed=seed, **row) for seed,row in zip(seeds,table)])
    t_report = deepcopy(s_report)
    t_report.update(statistics={"paired":between_components(allocation)},
                    sampling_profile=dict(sample_size=640, inner_samples=10000, integration_seed_namespace=4,
                                          nominal_inner_trial_budget_per_clock=6400000),
                    pair_analysis_seconds=800., wrapper_seconds=850., delivery_seconds=50.)
    t_spec = _archive(tmp_path/"allocation", "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE", t_report,
                      "analysis/block_diagnostics.csv", [dict(source_rng_seed=seed, **{"paired_"+key:value for key,value in row.items()})
                                                         for seed,row in zip(seeds,allocation)])
    settings = dict(schema="pfc_history_design_comparison_v1", required_source_seeds=seeds,
                    pooling_weights={"uniform":.5,"stratified":.5}, accepted_designs=designs,
                    accepted_allocation=t_spec, accepted_baseline_comparison=baseline_spec)
    result = run(tmp_path, settings, tmp_path/"result")
    assert result["comparison_kind"] == "fixed_baseline_M_vs_allocation_T"
    assert result["fixed_baseline_M"] == accepted["equal_weight_pool"]
    assert result["accepted_baseline_decision"] == "USE_EQUAL_WEIGHT_POOL_AS_DEVELOPMENT_BASELINE"
    assert result["costs"]["fixed_baseline_M"]["status"] == "unreported"
    assert result["costs"]["uniform_U"]["status"] == "unreported"
    assert result["costs"]["allocation_T"]["records"]["delivery_seconds"] == 50.
    assert result["costs"]["matched_runtime_comparison"] is False
    rows = list(csv.DictReader((tmp_path/"result/block_allocation_comparison.csv").open()))
    assert [int(row["source_rng_seed"]) for row in rows] == seeds
    legacy = {key:value for key,value in settings.items() if key not in ("accepted_allocation","accepted_baseline_comparison")}
    assert run(tmp_path, legacy, tmp_path/"legacy")["equal_weight_pool"] == accepted["equal_weight_pool"]
    for mode, message in (("profile", "ACCEPTED_ALLOCATION_PROFILE"), ("census", "ALLOCATION_SOURCE_CENSUS_CHANGED"),
                           ("uniform", "ALLOCATION_UNIFORM_REFERENCE_CHANGED"), ("partial", "COMPLETE_STRATIFIED32")):
        changed = deepcopy(t_report)
        if mode == "profile": changed["sampling_profile"]["integration_seed_namespace"] = 3
        elif mode == "census": changed["source_census"]["delivery_sha256"] = "d"*64
        elif mode == "uniform": changed["uniform_comparison"]["reference"]["archive"] = "different_uniform"
        else: changed["available_original_blocks"] = 31
        modified = dict(settings, accepted_allocation=_reseal_report(tmp_path/"allocation", changed))
        with pytest.raises(ValueError, match=message):
            run(tmp_path, modified, tmp_path/("bad_"+mode))
    settings["accepted_allocation"] = _reseal_report(tmp_path/"allocation", t_report)
    for mode in ("decision", "statistics"):
        changed = deepcopy(accepted)
        if mode == "decision": changed["decision"] = "RETAIN_SEPARATE_DEVELOPMENT_BASELINES"
        else: changed["equal_weight_pool"]["mean"] += 1.
        modified = dict(settings, accepted_baseline_comparison=_reseal_report(tmp_path/"baseline", changed))
        with pytest.raises(ValueError, match="ACCEPTED_FIXED_BASELINE_CHANGED"):
            run(tmp_path, modified, tmp_path/("bad_baseline_"+mode))

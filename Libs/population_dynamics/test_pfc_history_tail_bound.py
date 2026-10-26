"""Meaningful finite-frame concentration and saved-histogram checks."""
from copy import deepcopy
from io import BytesIO
from itertools import combinations, product
import json
import math
from pathlib import Path
import shutil

import yaml
import tarfile

import numpy as np
import pytest

from . import pfc_history_tail_bound as bound
from .test_pfc_first_passage_importance import tail_fixture
from .pfc_stratified_history import allocation


def settings():
    return dict(schema="pfc_history_tail_bound_v1", confidence_level=.95, conditional_error_share=.5,
                family=dict(original_blocks=32, profile_keys=72, clocks_per_pair=2,
                            maximum_sampled_windows=640, positive_strata=15, event_bounds_per_history=2))


def fixture(population=2, budget=2, values=(.2, .4)):
    counts = [8, population] + [0] * 14
    design = allocation(counts, budget)
    n = design[1]["sample_size"]
    frame = dict(allocation=design, nominal_sample_budget=budget, frame_size=sum(counts), sample_size=n,
                 selected_frame_indices=list(range(n)))
    rows = [dict(sample_index=i, frame_index=i, joint_time_bin=1,
                 inclusion_probability=design[1]["inclusion_probability"],
                 second_spike_upper=value, third_spike_enclosing_event_upper=value / 2)
            for i, value in enumerate(values)]
    return frame, rows


def test_census_has_no_sampling_penalty_and_known_zero_stratum_is_retained():
    frame, rows = fixture()
    result = bound.aggregate(frame, rows, settings())
    second = result["event_bounds"]["second_spike"]
    assert second["upper_bound"] == pytest.approx(.2 * .3)
    assert second["strata"][1]["window_sampling_upper_penalty"] == 0
    assert second["strata"][0]["population_size"] == 8
    assert second["strata"][0]["weighted_upper_bound"] == 0
    assert not result["source_history_population_uncertainty_established"]
    assert not result["third_spike_probability_or_bound_qualified"]
    assert not result["family_complete"]


def test_full_family_allocation_includes_both_events_and_preserves_total_error():
    result = bound.error_allocation(settings())
    assert result["maximum_row_event_bounds"] == 5898240
    assert result["maximum_stratum_event_bounds"] == 138240
    error = result["row_failure_probability"] * result["maximum_row_event_bounds"]
    error += result["stratum_failure_probability"] * result["maximum_stratum_event_bounds"]
    assert error == pytest.approx(.05)


def test_non_census_zero_upper_gets_positive_sampling_error():
    frame, rows = fixture(population=10, values=(0., 0.))
    result = bound.aggregate(frame, rows, settings())
    spec = result["event_bounds"]["second_spike"]["strata"][1]
    assert spec["window_sampling_upper_penalty"] == pytest.approx(
        math.sqrt(math.log(1 / result["stratum_failure_probability"]) / 4))
    assert result["event_bounds"]["second_spike"]["upper_bound"] > 0


def test_srswor_hoeffding_upper_controls_exhaustive_finite_population_tail():
    # Independent exact enumeration tests the inequality used by the reducer.
    population = [0.] * 4 + [1.] * 4
    n, delta = 3, .2
    epsilon = math.sqrt(math.log(1 / delta) / (2 * n))
    samples = list(combinations(population, n))
    failures = sum(.5 > sum(sample) / n + epsilon for sample in samples)
    assert failures / len(samples) <= delta


@pytest.mark.parametrize("defect", ["missing", "duplicate", "weight", "nan", "unobserved_stratum", "family"])
def test_rejects_incomplete_or_changed_sampling_contract(defect):
    frame, rows = fixture()
    plan = settings()
    if defect == "missing": rows.pop()
    elif defect == "duplicate": rows[1]["frame_index"] = rows[0]["frame_index"]
    elif defect == "weight": rows[0]["inclusion_probability"] = .5
    elif defect == "nan": rows[0]["second_spike_upper"] = float("nan")
    elif defect == "unobserved_stratum": frame["allocation"][1]["sample_size"] = 0
    else: plan["family"]["original_blocks"] = 1
    with pytest.raises(ValueError):
        bound.aggregate(frame, rows, plan)


def profile_fixture(tmp_path, *, structural=False, corrupt=False):
    profile = tmp_path / "profile"
    profile.mkdir()
    frame, _ = fixture()
    rows = [dict(sample_index=i, frame_index=i, joint_time_bin=1,
                 inclusion_probability=1., unit_index=10 + i,
                 first_spike_tick=None if structural else 2004, window_stop_tick=2100,
                 refractory_ticks=40, observed_count=0 if structural else 1,
                 category="no_first_spike" if structural else "conditional_integration_required") for i in range(2)]
    original = [{key: row[key] for key in ("sample_index", "frame_index", "joint_time_bin", "inclusion_probability")}
                for row in rows]
    integrated = deepcopy(rows)
    for i, row in enumerate(integrated):
        row.update(conditional_estimate=0., inner_monte_carlo_variance=0.)
        if not structural:
            row["integration_report"] = f"sample_{i:03d}/integration/report.json"
    actual = dict(status="COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY", frame=frame, dt_ms=.05,
                  source_rng_seed=95002, source_label="native", profile=dict(field_index=4, phase="cue"),
                  auxiliary_repeat="A", scientific_fidelity_qualified=False,
                  confirmation_execution_released=False, production_episodes=0)
    for name, value in (("report.json", actual), ("selected_frame.json", dict(frame=frame, rows=original)),
                        ("classified_frame.json", rows), ("integrated_frame.json", integrated)):
        (profile / name).write_text(json.dumps(value))
    if not structural:
        with tarfile.open(profile / "integration_details.tar", "w") as archive:
            for i in range(2):
                config, conditional, inner, histogram = tail_fixture()
                conditional["unit_index"] = 10 + i
                inner["stopped_prefix"]["estimate"] = 0.
                if corrupt and i == 1:
                    histogram["stopped_weight_sum"][0] = -1.
                buffer = BytesIO()
                np.savez_compressed(buffer, **histogram)
                members = [("reference/conditional_initial.json", json.dumps(conditional).encode()),
                           ("integration/report.json", json.dumps(inner).encode()),
                           ("integration/weighted_first_hit_histogram.npz", buffer.getvalue())]
                for name, payload in members:
                    member = tarfile.TarInfo(f"sample_{i:03d}/" + name)
                    member.size = len(payload)
                    archive.addfile(member, BytesIO(payload))
    return profile


def test_saved_histograms_reduce_without_new_trials_and_use_family_error(tmp_path):
    profile = profile_fixture(tmp_path)
    result = bound.run(dict(dt_ms=.05), settings(), profile, tmp_path / "result")
    rows = json.loads((tmp_path / "result/row_bounds.json").read_text())
    expected = -4 * math.expm1(-math.log(2 / result["row_failure_probability"]) / 10000)
    assert rows[0]["second_spike_upper"] == pytest.approx(expected, rel=1e-8)
    assert result["new_auxiliary_trials"] == result["new_full_network_episodes"] == 0
    assert result["event_bounds"]["second_spike"]["upper_bound"] == pytest.approx(.2 * expected)


def test_structural_histories_need_no_tar_and_produce_exact_conditional_zero(tmp_path):
    profile = profile_fixture(tmp_path, structural=True)
    result = bound.run(dict(dt_ms=.05), settings(), profile, tmp_path / "result")
    assert all(result["event_bounds"][event]["upper_bound"] == 0 for event in bound.EVENTS)


def test_corrupt_saved_histogram_retains_failure_and_withholds_bound(tmp_path):
    profile = profile_fixture(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="BOUND_WEIGHT_HISTOGRAM"):
        bound.run(dict(dt_ms=.05), settings(), profile, tmp_path / "result")
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["status"] == "FAILED_FINITE_ARCHIVED_HISTORY_TAIL_BOUND"
    assert not (tmp_path / "result/row_bounds.json").exists()


def independent_strata_example():
    populations = ([0., 0., .5, 1.], [0., .25, 1.], [.5])
    sample_sizes = (2, 1, 1)
    total = sum(map(len, populations))
    selections = product(*(combinations(range(len(values)), n)
                           for values, n in zip(populations, sample_sizes)))
    samples = []
    for choice in selections:
        strata = [dict(population_size=len(values), sample_size=n, weight=len(values) / total,
                       sampled_row_upper_mean=sum(values[i] for i in indices) / n)
                  for values, n, indices in zip(populations, sample_sizes, choice)]
        samples.append(strata)
    mean = sum(map(sum, populations[:2])) / total
    return samples, mean


def test_joint_chernoff_controls_every_attainable_exact_lower_tail():
    samples, mean = independent_strata_example()
    assert len(samples) == 18
    values = [bound.stratified_chernoff_upper(strata, .2) for strata in samples]
    cutoffs = sorted({row["sampled_noncensus_weighted_upper_mean"] for row in values})
    c = values[0]["maximum_sample_coefficient"]
    for z in cutoffs:
        if z <= mean:
            probability = sum(row["sampled_noncensus_weighted_upper_mean"] <= z
                              for row in values) / len(values)
            assert probability <= math.exp(-bound.lower_tail_divergence(mean, z) / c) + 1e-15


def test_joint_chernoff_inversion_coverage_and_upper_row_monotonicity():
    samples, mean = independent_strata_example()
    for delta in (.025 / 9216, .01, .05, .2, .5, .8):
        failures = 0
        for strata in samples:
            result = bound.stratified_chernoff_upper(strata, delta)
            failures += result["upper_bound"] < mean + .5 / 8
            inflated = deepcopy(strata)
            for row in inflated:
                row["sampled_row_upper_mean"] = min(1., row["sampled_row_upper_mean"] + .1)
            assert bound.stratified_chernoff_upper(inflated, delta)["upper_bound"] >= result["upper_bound"]
        assert failures / len(samples) <= delta


def test_joint_chernoff_zero_and_all_census_boundaries():
    strata = [dict(population_size=10, sample_size=2, weight=.4, sampled_row_upper_mean=0.),
              dict(population_size=2, sample_size=2, weight=.1, sampled_row_upper_mean=.5)]
    result = bound.stratified_chernoff_upper(strata, .2)
    assert result["upper_bound"] == pytest.approx(.05 + .2 * math.log(5))
    assert bound.stratified_chernoff_upper(strata, .01)["upper_bound"] == pytest.approx(.45)
    strata[0]["sampled_row_upper_mean"] = 1.
    assert bound.stratified_chernoff_upper(strata, .2)["upper_bound"] == pytest.approx(.45)
    census = bound.stratified_chernoff_upper(strata[1:], .2)
    assert census["upper_bound"] == .05
    assert census["window_sampling_upper_penalty"] == 0
    assert bound.stratified_chernoff_upper([], .2)["upper_bound"] == 0


def test_joint_chernoff_method_uses_aggregate_family_error_without_selecting_comparator():
    plan = settings()
    plan["window_sampling_bound"] = "stratified_chernoff"
    errors = bound.error_allocation(plan)
    total = errors["row_failure_probability"] * errors["maximum_row_event_bounds"]
    total += errors["aggregate_failure_probability"] * errors["maximum_aggregate_event_bounds"]
    assert errors["maximum_aggregate_event_bounds"] == 9216
    assert total == pytest.approx(.05)
    frame, rows = fixture(population=1000)
    result = bound.aggregate(frame, rows, plan)
    second = result["event_bounds"]["second_spike"]
    expected = bound.stratified_chernoff_upper(second["strata"], errors["aggregate_failure_probability"])
    assert second["upper_bound"] == expected["upper_bound"]
    assert not result["scientific_fidelity_qualified"]
    assert not result["source_history_population_uncertainty_established"]


def test_shared_stratum_indices_can_violate_joint_chernoff_assumption():
    # Each SRSWOR marginal is uniform, but perfectly shared indices remove
    # independence. For mean=.5,c=.5 and cutoff=0, .5 > exp(-1).
    assert .5 > math.exp(-bound.lower_tail_divergence(.5, 0.) / .5)


def test_lower_tail_divergence_is_stable_at_nearly_equal_mean_and_cutoff():
    from decimal import Decimal, localcontext
    for z in (1e-12, .01, .3):
        mean = math.nextafter(z, math.inf)
        with localcontext() as context:
            context.prec = 80
            m, cutoff = Decimal(mean), Decimal(z)
            expected = float(m - cutoff + cutoff * (cutoff / m).ln())
        assert bound.lower_tail_divergence(mean, z) == pytest.approx(expected, rel=1e-12, abs=0.)
    assert bound.lower_tail_divergence(.5, .5) == 0



def delivered_group_fixture(tmp_path, *, missing_histogram=False):
    """Create real group registry/provenance shapes with structural histories."""
    history = dict(required_source_seeds=list(range(95001, 95033)), auxiliary_repeats=["A", "B"],
        reused_profile_keys=[], profiles=[dict(left_cohort=left, right_cohort=right,
            field_index=index, phase="cue", condition="strong_cue", window_ms=5.0)
            for left, right in (("numpy_dt_0p05", "numpy_dt_0p025"), ("numpy_dt_0p025", "numpy_dt_0p0125"))
            for index in range(18)])
    group = tmp_path / "readback"; group.mkdir()
    settings_path = group / "source_config.yaml"
    settings_path.write_text(yaml.safe_dump(history))
    config = dict(schema="pfc_profile_history_readback_v1", source_config_sha256=bound.sha256_file(settings_path),
        source_rng_seeds=[95002], source_arrays=[dict(archive_prefix="histories/run_fixture",
            source_index_start=5, source_index_stop=9)])
    request = bound.readback_requests(config, history)[0]
    config_path = group / "Libs/configs/neural_population_pfc_profile_history_readback.yaml"
    config_path.parent.mkdir(parents=True); config_path.write_text(yaml.safe_dump(config))
    costs, pairs = {}, {key: dict(key=key, per_clock={}) for key in request["profile_keys"]}
    for clock, spec in request["source_runs"].items():
        label = spec["source_label"]
        dt = .05 if clock == "coarse" else .025
        native = tmp_path / "sources" / label
        native.mkdir(parents=True)
        source_config = dict(dt_ms=dt)
        (native / "case").mkdir()
        (native / "case/config.yaml").write_text(yaml.safe_dump(source_config))
        (native / "case/report.json").write_text(json.dumps(dict(source_execution_complete=True,
                                                               resolved_config=source_config)))
        write_fixture_delivery(native, ["case/config.yaml", "case/report.json"])
        binding = dict(archive=str(native.relative_to(tmp_path)), case_path="case",
            delivery_sha256=bound.sha256_file(native / "delivery.json"),
            report_sha256=bound.sha256_file(native / "case/report.json"))
        source = tmp_path / spec["archive"]; source.mkdir(parents=True)
        (source / "runtime_config.json").write_text(json.dumps(dict(history, source_binding=binding)))
        (source / "analysis").mkdir()
        (source / "analysis/report.json").write_text(json.dumps(dict(status="COMPLETE_PFC_PROFILE_SOURCE_HISTORY",
            source_rng_seed=95002, source_label=label, requested_profile_keys=spec["expected_profile_keys"],
            completed_profile_keys=spec["expected_profile_keys"])))
        names = ["runtime_config.json", "analysis/report.json"]
        for key in spec["expected_profile_keys"]:
            destination = source / "analysis/frames" / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = tmp_path / ("fixture_" + clock + "_" + key.replace("/", "_"))
            staging.mkdir()
            profile = profile_fixture(staging, structural=True)
            shutil.move(profile, destination)
            actual = json.loads((destination / "report.json").read_text())
            repeat = key.split("/", 1)[0]
            selected_profile = next(profile for profile in history["profiles"]
                                    if bound.profile_key(profile, repeat) == key)
            actual.update(dt_ms=dt, source_label=label, profile=selected_profile,
                          auxiliary_repeat=repeat, sample_mean=0., integrated_histories=0, inner_trials=0)
            (destination / "report.json").write_text(json.dumps(actual))
            if missing_histogram and clock == "coarse" and key == request["profile_keys"][0]:
                classified = json.loads((destination / "classified_frame.json").read_text())
                integrated = json.loads((destination / "integrated_frame.json").read_text())
                for row in classified:
                    row.update(first_spike_tick=2004, observed_count=1, category="conditional_integration_required")
                for row in integrated:
                    row.update(first_spike_tick=2004, observed_count=1, category="conditional_integration_required",
                               integration_report=f"sample_{row['sample_index']:03d}/integration/report.json")
                (destination / "classified_frame.json").write_text(json.dumps(classified))
                (destination / "integrated_frame.json").write_text(json.dumps(integrated))
                actual.update(integrated_histories=2, inner_trials=20000)
                (destination / "report.json").write_text(json.dumps(actual))
            names.extend(str(path.relative_to(source)) for path in destination.iterdir())
            if key in pairs:
                pairs[key]["per_clock"][clock] = dict(sample_mean=actual["sample_mean"],
                    integrated_histories=actual["integrated_histories"], inner_trials=actual["inner_trials"], source_binding=binding)
        write_fixture_delivery(source, names, "COMPLETE_PFC_PROFILE_HISTORY_COMPLETE")
        costs[clock] = dict(history_execution_archive=spec["archive"], source_binding=binding)
    report = dict(status="COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK", source_rng_seed=95002,
        readback_group=request["group_name"], readback_group_index=0, full_source_requests=True,
        original_source_blocks=1, paired_profiles=len(pairs), pairs=list(pairs.values()), source_costs=costs,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        new_independent_source_blocks=0, production_episodes=0)
    (group / "analysis").mkdir(); (group / "analysis/report.json").write_text(json.dumps(report))
    write_fixture_delivery(group, ["analysis/report.json", "source_config.yaml",
        "Libs/configs/neural_population_pfc_profile_history_readback.yaml"], "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE")
    return group, request


def write_fixture_delivery(root, names, status=None):
    files = [dict(path=name, bytes=(root/name).stat().st_size, sha256=bound.sha256_file(root/name)) for name in names]
    (root / "delivery.json").write_text(json.dumps(dict(verified=True, status=status, files=files)))


def test_delivered_group_reduces_every_registered_clock_and_preserves_family_scope(tmp_path):
    group, request = delivered_group_fixture(tmp_path)
    result = bound.run_readback_group(settings(), group, tmp_path, tmp_path / "result")
    assert result["group_complete"] is True
    assert result["completed_clock_profiles"] == result["expected_clock_profiles"]
    assert len(result["bounds"]) == result["completed_group_clock_profiles"] == len(request["profile_keys"]) * 2
    assert result["completed_group_event_bounds"] == len(request["profile_keys"]) * 4
    assert result["required_family_clock_profiles"] == 4608
    assert result["required_family_event_bounds"] == result["maximum_aggregate_event_bounds"] == 9216
    assert not result["family_complete"] and not result["scientific_fidelity_qualified"]
    assert result["new_auxiliary_trials"] == result["new_full_network_episodes"] == 0
    assert not result["missing_histograms"] and not result["missing_clock_profiles"]
    for row in result["bounds"]:
        assert row["result"]["event_bounds"]["third_spike_enclosing_event"]["upper_bound"] == 0


def test_delivered_group_missing_histogram_is_explicit_failed_group(tmp_path):
    group, request = delivered_group_fixture(tmp_path, missing_histogram=True)
    with pytest.raises(ValueError, match="INCOMPLETE_GROUP_TAIL_BOUNDS"):
        bound.run_readback_group(settings(), group, tmp_path, tmp_path / "result")
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["status"] == "FAILED_FINITE_ARCHIVED_HISTORY_GROUP_TAIL_BOUND"
    assert not report["group_complete"] and not report["family_complete"]
    assert len(report["missing_histograms"]) == len(report["missing_clock_profiles"]) == 1
    assert report["missing_histograms"][0]["profile_key"] == request["profile_keys"][0]
    assert report["missing_histograms"][0]["clock"] == "coarse"
    assert len(report["completed_clock_profiles"]) == len(request["profile_keys"]) * 2 - 1


def test_delivered_group_rejects_changed_source_binding_and_profile_denominator(tmp_path):
    group, request = delivered_group_fixture(tmp_path)
    path = group / "analysis/report.json"
    report = json.loads(path.read_text())
    report["pairs"].pop()
    path.write_text(json.dumps(report))
    write_fixture_delivery(group, ["analysis/report.json", "source_config.yaml",
        "Libs/configs/neural_population_pfc_profile_history_readback.yaml"], "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE")
    with pytest.raises(ValueError, match="REGISTERED_GROUP_IDENTITY"):
        bound.run_readback_group(settings(), group, tmp_path, tmp_path / "missing-pair")
    assert not json.loads((tmp_path / "missing-pair/report.json").read_text())["group_complete"]
    report["pairs"].append(dict(key=request["profile_keys"][-1]))
    report["source_costs"]["coarse"]["source_binding"]["report_sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    write_fixture_delivery(group, ["analysis/report.json", "source_config.yaml",
        "Libs/configs/neural_population_pfc_profile_history_readback.yaml"], "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE")
    with pytest.raises(ValueError, match="HISTORY_SOURCE_BINDING"):
        bound.run_readback_group(settings(), group, tmp_path, tmp_path / "changed-binding")



def maximum_coefficient_reference(strata, delta):
    census = math.fsum(row["weight"] * row["sampled_row_upper_mean"] for row in strata
                       if row["sample_size"] == row["population_size"] and row["sample_size"] > 0)
    sampled = [row for row in strata if 0 < row["sample_size"] < row["population_size"]]
    if not sampled:
        return census
    mass = math.fsum(row["weight"] for row in sampled)
    z = math.fsum(row["weight"] * row["sampled_row_upper_mean"] for row in sampled)
    c = max(row["weight"] / row["sample_size"] for row in sampled)
    threshold = c * math.log(1 / delta)
    if z == 0:
        return census + min(mass, threshold)
    if bound.lower_tail_divergence(mass, z) <= threshold:
        return census + mass
    lo, hi = z, mass
    for _ in range(100):
        middle = (lo + hi) / 2
        if bound.lower_tail_divergence(middle, z) > threshold:
            hi = middle
        else:
            lo = middle
    return census + hi


def test_capacity_chernoff_matches_equal_coefficients_and_unsaturated_largest_capacity():
    equal = [dict(population_size=4, sample_size=2, weight=.2, sampled_row_upper_mean=.1),
             dict(population_size=6, sample_size=3, weight=.3, sampled_row_upper_mean=.2)]
    large = [dict(population_size=10, sample_size=2, weight=.4, sampled_row_upper_mean=0.),
             dict(population_size=100, sample_size=10, weight=.1, sampled_row_upper_mean=0.)]
    for strata, delta in ((equal, .2), (large, .8)):
        actual = bound.stratified_chernoff_upper(strata, delta)
        assert actual["upper_bound"] == pytest.approx(maximum_coefficient_reference(strata, delta), rel=1e-12)
        assert actual["coefficient_capacity_rule"] == "unknown_stratum_mean_mass_fills_descending_sample_coefficient"
        assert all(row["weight_capacity"] > 0 for row in actual["coefficient_capacities"])


def test_capacity_chernoff_heterogeneous_zero_cutoff_has_exact_limiting_inversion():
    strata = [dict(population_size=2, sample_size=1, weight=.01, sampled_row_upper_mean=0.),
              dict(population_size=200, sample_size=100, weight=.4, sampled_row_upper_mean=0.),
              dict(population_size=2, sample_size=2, weight=.1, sampled_row_upper_mean=.5)]
    result = bound.stratified_chernoff_upper(strata, math.exp(-3))
    # First capacity contributes .01/.01=1 to the limiting exponent. The
    # remaining exponent2 is supplied at coefficient .004, so M=.018.
    assert result["noncensus_probability_upper_bound"] == pytest.approx(.018, rel=1e-10)
    assert result["upper_bound"] == pytest.approx(.068, rel=1e-10)
    assert result["upper_bound"] < maximum_coefficient_reference(strata, math.exp(-3))
    assert result["census_weighted_upper_mean"] == .05


def test_capacity_chernoff_exact_srswor_tail_and_no_weaker_bounds():
    samples, mean = independent_strata_example()
    capacities = sorted([(row["weight"], row["weight"] / row["sample_size"])
                         for row in samples[0] if row["sample_size"] < row["population_size"]],
                        key=lambda pair: pair[1], reverse=True)
    cutoffs = sorted({math.fsum(row["weight"] * row["sampled_row_upper_mean"]
                              for row in strata if row["sample_size"] < row["population_size"])
                      for strata in samples})
    for cutoff in cutoffs:
        if cutoff < mean:
            probability = sum(math.fsum(row["weight"] * row["sampled_row_upper_mean"]
                              for row in strata if row["sample_size"] < row["population_size"]) <= cutoff
                              for strata in samples) / len(samples)
            assert probability <= math.exp(bound._capacity_log_bound(mean, cutoff, capacities)) + 1e-14
    for delta in (.025 / 9216, .01, .05, .2, .5, .8):
        for strata in samples:
            result = bound.stratified_chernoff_upper(strata, delta)
            assert result["upper_bound"] <= maximum_coefficient_reference(strata, delta) + 1e-12


def test_capacity_chernoff_positive_cutoff_monotone_and_caps_do_not_use_observed_values():
    base = [dict(population_size=4, sample_size=1, weight=.02, sampled_row_upper_mean=.1),
            dict(population_size=200, sample_size=100, weight=.4, sampled_row_upper_mean=.1)]
    result = bound.stratified_chernoff_upper(base, .2)
    changed = deepcopy(base); changed[0]["sampled_row_upper_mean"] = .5
    inflated = bound.stratified_chernoff_upper(changed, .2)
    assert inflated["upper_bound"] >= result["upper_bound"]
    assert inflated["coefficient_capacities"] == result["coefficient_capacities"]
    assert bound.stratified_chernoff_upper(base, .1)["upper_bound"] >= result["upper_bound"]


@pytest.mark.parametrize("defect", ["negative_weight", "nonfinite_mean", "weight_mass", "zero_samples", "invalid_count"])
def test_capacity_chernoff_rejects_invalid_support(defect):
    strata = [dict(population_size=4, sample_size=2, weight=.6, sampled_row_upper_mean=.1)]
    if defect == "negative_weight":
        strata[0]["weight"] = -.1
    elif defect == "nonfinite_mean":
        strata[0]["sampled_row_upper_mean"] = float("nan")
    elif defect == "weight_mass":
        strata.append(dict(population_size=4, sample_size=4, weight=.6, sampled_row_upper_mean=0.))
    elif defect == "zero_samples":
        strata[0]["sample_size"] = 0
    else:
        strata[0]["sample_size"] = 5
    with pytest.raises(ValueError, match="CHERNOFF|UNSAMPLED"):
        bound.stratified_chernoff_upper(strata, .2)

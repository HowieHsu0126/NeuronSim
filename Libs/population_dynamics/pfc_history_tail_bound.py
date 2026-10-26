"""Simultaneous upper bounds for fixed archived PFC conditional histories.

The target averages clamped-forcing conditional probabilities over the stored
physical frame. It is not the coupled-source count law or a source population.
Reuse saved importance trials and their stopped first-hit histograms.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from . import pfc_first_passage_importance as importance
from .pfc_history_profile_census import write_json
from .pfc_profile_history_readback import tar_json, readback_requests
from .pfc_profile_history import profile_key
from .provenance import sha256_file
from .pfc_stratified_history import allocation


EVENTS = ("second_spike", "third_spike_enclosing_event")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_HISTORY_TAIL_BOUND_" + message)


def error_allocation(settings: Mapping[str, Any]) -> dict[str, Any]:
    require(settings["schema"] == "pfc_history_tail_bound_v1", "SCHEMA")
    family = settings["family"]
    require(family == dict(original_blocks=32, profile_keys=72, clocks_per_pair=2,
                           maximum_sampled_windows=640, positive_strata=15,
                           event_bounds_per_history=2), "REGISTERED_FAMILY")
    confidence = settings["confidence_level"]
    share = settings["conditional_error_share"]
    require(0 < confidence < 1 and 0 < share < 1, "ERROR_ALLOCATION")
    comparisons = family["original_blocks"] * family["profile_keys"] * family["clocks_per_pair"] * 2
    rows = comparisons * family["maximum_sampled_windows"]
    strata = comparisons * family["positive_strata"]
    method = settings.get("window_sampling_bound", "stratum_hoeffding")
    require(method in ("stratum_hoeffding", "stratified_chernoff"), "SAMPLING_BOUND_METHOD")
    alpha = 1 - confidence
    return dict(confidence_level=confidence, maximum_row_event_bounds=rows,
                window_sampling_bound=method, maximum_aggregate_event_bounds=comparisons,
                aggregate_failure_probability=alpha * (1 - share) / comparisons,
                maximum_stratum_event_bounds=strata,
                row_failure_probability=alpha * share / rows,
                stratum_failure_probability=alpha * (1 - share) / strata,
                conditional_error_share=share)


def lower_tail_divergence(mean: float, cutoff: float) -> float:
    """Return M-z+z log(z/M) without cancellation near z=M."""
    require(math.isfinite(mean) and math.isfinite(cutoff)
            and 0 <= cutoff <= mean, "LOWER_TAIL_ARGUMENTS")
    if cutoff == 0:
        return mean
    ratio = (mean - cutoff) / mean
    if ratio <= .5:
        # (1-r)log(1-r)+r = sum r**k/[k(k-1)]. Terms are
        # positive; 64 terms leave less than 2e-24 relative tail at r<=.5.
        return mean * math.fsum(ratio ** k / (k * (k - 1)) for k in range(2, 66))
    return mean - cutoff + cutoff * (math.log(cutoff) - math.log(mean))


def _capacity_log_bound(mean: float, cutoff: float, capacities: Sequence[tuple[float, float]]) -> float:
    """Upper-bound the lower-tail log probability for unknown stratum means.

    Each pair is (weight capacity, sample coefficient), ordered by decreasing
    coefficient. The maximizing mean allocation fills that order. The final
    calculation uses the LP dual, so an approximate optimizing lambda remains
    a valid Chernoff upper bound rather than an underestimated numerical minimum.
    """
    require(mean > cutoff >= 0 and capacities, "CAPACITY_MEAN_AND_CUTOFF")
    cmax = capacities[0][1]
    remaining, allocated, threshold = mean, [], 0
    for index, (weight, coefficient) in enumerate(capacities):
        amount = min(weight, max(0., remaining))
        if amount > 0:
            allocated.append((amount, coefficient))
            threshold = index
        remaining = max(0., mean - math.fsum(amount for amount, _ in allocated))
        if remaining == 0:
            break
    require(remaining <= 8 * math.ulp(max(1., mean)), "CAPACITY_TOTAL_MASS")
    if cutoff == 0:
        # The lambda->infinity dual is analytic; no numerical optimizer is
        # needed for a zero observed weighted mean.
        values = [-1. / coefficient for _, coefficient in capacities]
        lam_term = 0.
    else:
        ratios = [(amount, coefficient / cmax) for amount, coefficient in allocated]
        require(all(ratio > 0 for _, ratio in ratios), "CAPACITY_COEFFICIENT_RATIO")
        lo, hi = 0., -math.log(cutoff / mean) / min(ratio for _, ratio in ratios)
        require(math.isfinite(hi), "CAPACITY_LAMBDA_RANGE")
        for _ in range(80):
            middle = lo + (hi - lo) / 2
            if middle == lo or middle == hi:
                break
            derivative = math.fsum(amount * math.exp(-middle * ratio) for amount, ratio in ratios)
            if derivative > cutoff:
                lo = middle
            else:
                hi = middle
        scaled_lambda = lo + (hi - lo) / 2
        lam_term = scaled_lambda * cutoff / cmax
        values = [math.expm1(-scaled_lambda * coefficient / cmax) / coefficient
                  for _, coefficient in capacities]
    # For every nu, max_{0<=M_h<=w_h,sum M_h=M} sum M_h*a_h
    # <= nu*M + sum w_h*max(a_h-nu,0). Select the filled-stratum threshold.
    nu = values[threshold]
    terms = [lam_term, nu * mean]
    terms.extend(weight * max(value - nu, 0.) for (weight, _), value in zip(capacities, values))
    absolute_sum = math.fsum(abs(value) for value in terms)
    require(math.isfinite(absolute_sum), "CAPACITY_FINITE_DUAL")
    # Add an upward binary64 guard beside the upper-endpoint root inversion.
    # This allowance assumes ordinary libm accuracy; it is not an interval
    # certificate for transcendental functions. Reference tests check this path.
    allowance = 32 * (len(terms) + 2) * math.ulp(max(1., absolute_sum))
    return min(0., math.nextafter(math.fsum(terms) + allowance, math.inf))


def _maximum_coefficient_upper(mean: float, mass: float, coefficient: float,
                               failure_probability: float) -> float:
    """Closed-form maximum-coefficient envelope, used as the root bracket."""
    threshold = coefficient * math.log(1 / failure_probability)
    if mean == 0:
        return min(mass, math.nextafter(threshold, math.inf))
    if mean == mass or lower_tail_divergence(mass, mean) <= threshold:
        return mass
    lo, hi = mean, mass
    for _ in range(80):
        middle = lo + (hi - lo) / 2
        if middle == lo or middle == hi:
            break
        if lower_tail_divergence(middle, mean) > threshold:
            hi = middle
        else:
            lo = middle
    return min(mass, math.nextafter(hi, math.inf))


def stratified_chernoff_upper(strata: Sequence[Mapping[str, Any]],
                             failure_probability: float) -> dict[str, Any]:
    """Invert the capacity-aware joint Chernoff bound for ideal SRSWOR strata.

    For fixed q_i in [0,1], the stratum MGF is bounded by
    exp[M_h*expm1(-lambda*c_h)/c_h], c_h=w_h/n_h. Independent
    stratum selections allow multiplication. For each possible total mean M,
    maximize over unknown M_h in [0,w_h] by filling decreasing c_h. This
    order maximizes the MGF for every lambda. Optimize lambda and invert M.
    Census means add exactly. Row upper estimates replace sampled q_i by
    monotonicity; union row failure with sampling failure. Caps never use
    observed row means. PRNG implementation does not prove randomization.
    """
    require(math.isfinite(failure_probability) and 0 < failure_probability < 1,
            "CHERNOFF_ERROR_PROBABILITY")
    census, sampled, capacities, weights = [], [], [], []
    for spec in strata:
        population, n = spec["population_size"], spec["sample_size"]
        weight, value = spec["weight"], spec["sampled_row_upper_mean"]
        require(type(population) is int and type(n) is int and 0 <= n <= population
                and type(weight) in (int, float) and type(value) in (int, float)
                and math.isfinite(weight) and 0 <= weight <= 1
                and math.isfinite(value) and 0 <= value <= 1, "CHERNOFF_STRATUM")
        weights.append(weight)
        if n == 0:
            require(value == 0, "UNSAMPLED_NONZERO_MEAN")
        elif n == population:
            census.append(weight * value)
        elif weight > 0:
            sampled.append(weight * value)
            coefficient = weight / n
            require(math.isfinite(coefficient) and coefficient > 0, "CHERNOFF_POSITIVE_SAMPLE_COEFFICIENT")
            capacities.append((weight, coefficient))
    exact = math.fsum(census)
    capacities.sort(key=lambda pair: pair[1], reverse=True)
    mass = math.fsum(weight for weight, _ in capacities)
    require(math.fsum(weights) <= 1 + 8 * math.ulp(1.), "CHERNOFF_TOTAL_WEIGHT_SUPPORT")
    z = min(mass, math.fsum(sampled))
    c = capacities[0][1] if capacities else 0.
    if c == 0:
        upper = 0.
    else:
        # The maximum-coefficient envelope is an analytic safe upper bracket,
        # not a post-outcome choice between statistical methods.
        maximum_upper = _maximum_coefficient_upper(z, mass, c, failure_probability)
        hi = maximum_upper
        largest_capacity = math.fsum(weight for weight, coefficient in capacities if coefficient == c)
        if hi <= largest_capacity or hi == z:
            upper = hi
        else:
            lo = z
            log_delta = math.log(failure_probability)
            for _ in range(80):
                middle = lo + (hi - lo) / 2
                if middle == lo or middle == hi:
                    break
                if _capacity_log_bound(middle, z, capacities) < log_delta:
                    hi = middle
                else:
                    lo = middle
            upper = min(maximum_upper, math.nextafter(hi, math.inf))
    return dict(upper_bound=exact + upper, census_weighted_upper_mean=exact,
                sampled_noncensus_weighted_upper_mean=z, noncensus_mass=mass,
                maximum_sample_coefficient=c, noncensus_probability_upper_bound=upper,
                window_sampling_upper_penalty=upper - z,
                coefficient_capacity_rule="unknown_stratum_mean_mass_fills_descending_sample_coefficient",
                coefficient_capacities=[dict(weight_capacity=weight, sample_coefficient=coefficient)
                                        for weight, coefficient in capacities],
                numerical_inversion="Chernoff dual evaluation with binary64 guard and upper root endpoint",
                transcendental_rounding_certified=False,
                numerical_assumption="ordinary libm accuracy; binary64 guard validated against high-precision references")


def aggregate(frame: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
              settings: Mapping[str, Any]) -> dict[str, Any]:
    """Combine row bounds with the declared fixed-archive sampling bound.

    The legacy per-stratum Hoeffding consumer remains supported. The aggregate
    Chernoff method requires independent stratum selections under ideal SRSWOR.
    Neither method assumes independence across profiles, clocks or blocks.
    """
    errors = error_allocation(settings)
    design = frame["allocation"]
    populations = [spec["population_size"] for spec in design]
    require(len(design) == 16 and design == allocation(populations, frame["nominal_sample_budget"]),
            "SAMPLING_DESIGN")
    require(frame["frame_size"] == sum(populations) and 0 < frame["frame_size"]
            and len(rows) == frame["sample_size"] == sum(spec["sample_size"] for spec in design)
            and frame["nominal_sample_budget"] <= 640, "COMPLETE_FRAME")
    require([row["sample_index"] for row in rows] == list(range(len(rows)))
            and [row["frame_index"] for row in rows] == frame["selected_frame_indices"]
            and len({row["frame_index"] for row in rows}) == len(rows), "PHYSICAL_ROW_IDENTITIES")
    for row in rows:
        h = row["joint_time_bin"]
        require(type(h) is int and 1 <= h <= 15
                and row["inclusion_probability"] == design[h]["inclusion_probability"], "ROW_STRATUM")
        require(all(math.isfinite(row[event + "_upper"]) and 0 <= row[event + "_upper"] <= 1
                    for event in EVENTS), "ROW_BOUND_SUPPORT")
    results = {}
    for event in EVENTS:
        strata = []
        for spec in design:
            h, n, population = spec["stratum"], spec["sample_size"], spec["population_size"]
            values = [row[event + "_upper"] for row in rows if row["joint_time_bin"] == h]
            require(len(values) == n, "MISSING_STRATUM:" + str(h))
            if n == 0:
                require(h == 0 or population == 0, "UNSAMPLED_POSITIVE_STRATUM")
                mean = penalty = upper = 0.
            else:
                mean = math.fsum(values) / n
                penalty = 0. if n == population else math.sqrt(
                    math.log(1 / errors["stratum_failure_probability"]) / (2 * n))
                upper = min(1., mean + penalty)
            strata.append(dict(spec, sampled_row_upper_mean=mean,
                               window_sampling_upper_penalty=penalty,
                               stratum_probability_upper_bound=upper,
                               weighted_upper_bound=spec["weight"] * upper))
        if errors["window_sampling_bound"] == "stratified_chernoff":
            sampling = stratified_chernoff_upper(strata, errors["aggregate_failure_probability"])
            # Legacy fields are a separately allocated historical comparator.
            # Do not take the minimum of the two bounds after seeing outcomes.
            results[event] = dict(sampling, strata=strata,
                                  legacy_hoeffding_comparator=math.fsum(
                                      row["weighted_upper_bound"] for row in strata))
        else:
            results[event] = dict(upper_bound=math.fsum(row["weighted_upper_bound"] for row in strata),
                                  strata=strata)
    return dict(status="COMPLETE_FINITE_ARCHIVED_HISTORY_TAIL_BOUND", **errors,
                frame_size=frame["frame_size"], selected_physical_windows=len(rows),
                event_bounds=results,
                target="Finite physical-frame average of conditional event probabilities given each stored first-spike history and clamped other-unit forcing; no-first histories contribute zero",
                uncertainty_scope="Conditional auxiliary importance trials and SRSWOR within this fixed archive, with Bonferroni allocation over the registered family",
                sampling_assumption="Ideal uniform SRSWOR conditional on the fixed archive; independent selections between strata. Distinct deterministic PRNG seed vectors implement this design but do not prove a randomization law.",
                family_complete=False, source_history_population_uncertainty_established=False,
                coupled_source_conditional_law_qualified=False,
                third_spike_probability_established=False, third_spike_probability_or_bound_qualified=False,
                population_confidence_interval_computed=False, scientific_fidelity_qualified=False,
                confirmation_execution_released=False, new_full_network_episodes=0,
                new_independent_source_blocks=0, new_auxiliary_trials=0, production_episodes=0)


def run(config: Mapping[str, Any], settings: Mapping[str, Any], profile: Path,
        output: Path) -> dict[str, Any]:
    """Reduce one authenticated profile; callers bind the producer's archive."""
    errors = error_allocation(settings)
    actual = json.loads((profile / "report.json").read_text())
    selected = json.loads((profile / "selected_frame.json").read_text())
    classified = json.loads((profile / "classified_frame.json").read_text())
    integrated = json.loads((profile / "integrated_frame.json").read_text())
    frame = actual["frame"]
    require(actual["status"] == "COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY"
            and actual["dt_ms"] == config["dt_ms"] and frame == selected["frame"]
            and len(integrated) == len(classified) == len(selected["rows"]), "PROFILE_IDENTITY")
    require(not actual["scientific_fidelity_qualified"] and not actual["confirmation_execution_released"]
            and actual["production_episodes"] == 0, "UNRELEASED_SCOPE")
    output.mkdir(parents=True, exist_ok=False)
    row_bounds = []
    try:
        eligible = any(row["category"] == "conditional_integration_required" for row in integrated)
        archive = tarfile.open(profile / "integration_details.tar", "r:") if eligible else None
        try:
            for index, (row, before, original) in enumerate(zip(integrated, classified, selected["rows"])):
                require(row["sample_index"] == index and all(row[key] == value for key, value in before.items())
                        and all(row[key] == value for key, value in original.items()), "ROW_IDENTITY")
                category = row["category"]
                if category == "conditional_integration_required":
                    relative = f"sample_{index:03d}"
                    require(row["observed_count"] == 1
                            and row["integration_report"] == relative + "/integration/report.json", "ELIGIBLE_ROW")
                    inner = tar_json(archive, row["integration_report"])
                    conditional = tar_json(archive, relative + "/reference/conditional_initial.json")
                    require(all(conditional[key] == row[key] for key in
                                ("unit_index", "first_spike_tick", "window_stop_tick")), "CONDITIONAL_IDENTITY")
                    require(inner["samples"] == 10000 and inner["stopped_prefix"]["estimate"] == row["conditional_estimate"],
                            "EXISTING_TRIALS_ONLY")
                    histogram_name = relative + "/integration/weighted_first_hit_histogram.npz"
                    with archive.extractfile(histogram_name) as stream, np.load(stream, allow_pickle=False) as data:
                        # The archived interval is unchanged. The derivative uses
                        # the predeclared simultaneous failure allocation.
                        derivative = deepcopy(inner)
                        derivative["stopped_prefix"]["conditional_interval"]["confidence_level"] = 1 - errors["row_failure_probability"]
                        tail = importance.refractory_tail_bound(config, conditional, derivative, data)
                    require(tail["minimum_interspike_ticks"] == row["refractory_ticks"], "SOURCE_REFRACTORY_RULE")
                    second = importance.bounded_mean_interval(row["conditional_estimate"], inner["samples"],
                                                             inner["weight_bound"], 1 - errors["row_failure_probability"])
                    second_upper = min(1., second["upper"])
                    third_upper = tail["third_spike_probability_upper_bound"]
                    enclosing_mean = tail["enclosing_second_spike_estimate"]
                else:
                    require(category in ("no_first_spike", "refractory_covers_remaining_window")
                            and row["conditional_estimate"] == row["inner_monte_carlo_variance"] == 0,
                            "UNRESOLVED_HISTORY")
                    require((category == "no_first_spike" and row["observed_count"] == 0 and row["first_spike_tick"] is None)
                            or (category == "refractory_covers_remaining_window" and row["observed_count"] == 1
                                and row["first_spike_tick"] + row["refractory_ticks"] >= row["window_stop_tick"]), "STRUCTURAL_ZERO")
                    second_upper = third_upper = enclosing_mean = 0.
                row_bounds.append(dict(original, category=category, second_spike_upper=second_upper,
                                       third_spike_enclosing_event_upper=third_upper,
                                       second_spike_estimate=row["conditional_estimate"],
                                       third_spike_enclosing_event_estimate=enclosing_mean))
        finally:
            if archive is not None:
                archive.close()
        result = aggregate(frame, row_bounds, settings)
        result.update(source_rng_seed=actual["source_rng_seed"], source_label=actual["source_label"],
                      profile=actual["profile"], auxiliary_repeat=actual["auxiliary_repeat"],
                      dt_ms=config["dt_ms"], row_bounds_file="row_bounds.json")
        write_json(output / "row_bounds.json", row_bounds)
    except Exception as exc:
        write_json(output / "report.json", dict(status="FAILED_FINITE_ARCHIVED_HISTORY_TAIL_BOUND",
                                                error=dict(type=type(exc).__name__, message=str(exc))))
        raise
    write_json(output / "report.json", result)
    return result


def _delivered_file(archive: Path, delivery: Mapping[str, Any], name: str) -> Path:
    """Authenticate a consumed file against its existing producer delivery."""
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "DELIVERY_RELATIVE_PATH")
    entries = [row for row in delivery["files"] if row["path"] == name]
    require(len(entries) == 1, "UNIQUE_DELIVERY_FILE:" + name)
    path = archive / relative
    require(path.is_file() and path.stat().st_size == entries[0]["bytes"]
            and sha256_file(path) == entries[0]["sha256"], "DELIVERY_FILE_CHANGED:" + name)
    return path


def _delivered_archive(path: Path, status: str) -> dict[str, Any]:
    delivery = json.loads((path / "delivery.json").read_text())
    require(delivery.get("verified") is True and delivery.get("status") == status,
            "COMPLETED_ARCHIVE_REQUIRED:" + str(path))
    return delivery


def _source_config(project_root, binding):
    archive = project_root / binding["archive"]
    path = archive / "delivery.json"
    require(sha256_file(path) == binding["delivery_sha256"], "SOURCE_DELIVERY_BINDING")
    delivery = json.loads(path.read_text())
    case = Path(binding["case_path"])
    require(not case.is_absolute() and ".." not in case.parts, "SOURCE_CASE_RELATIVE_PATH")
    report_path = _delivered_file(archive, delivery, str(case / "report.json"))
    require(sha256_file(report_path) == binding["report_sha256"], "SOURCE_REPORT_BINDING")
    report = json.loads(report_path.read_text())
    config = yaml.safe_load(_delivered_file(archive, delivery, str(case / "config.yaml")).read_text())
    require(report.get("source_execution_complete") is True and report.get("resolved_config") == config,
            "COMPLETE_SOURCE_CONFIG_BINDING")
    return config


def run_readback_group(settings: Mapping[str, Any], readback: Path,
                       project_root: Path, output: Path) -> dict[str, Any]:
    """Reduce every clock/profile in one delivered paired readback group.

    The producer readback has already checked the original importance summaries.
    This consumes only delivery-authenticated stored histograms. Group coverage
    never becomes full-family or source-population qualification.
    """
    errors = error_allocation(settings)
    readback, project_root, output = Path(readback), Path(project_root), Path(output)
    require(not output.exists(), "GROUP_OUTPUT_EXISTS")
    report = dict(status="FAILED_FINITE_ARCHIVED_HISTORY_GROUP_TAIL_BOUND", **errors,
        readback_archive=str(readback), expected_clock_profiles=[], completed_clock_profiles=[],
        missing_clock_profiles=[], missing_histograms=[], failures=[], bounds=[], group_complete=False,
        required_family_clock_profiles=4608, required_family_event_bounds=9216,
        family_complete=False, source_history_population_uncertainty_established=False,
        coupled_source_conditional_law_qualified=False, third_spike_probability_established=False,
        third_spike_probability_or_bound_qualified=False, population_confidence_interval_computed=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        new_full_network_episodes=0, new_independent_source_blocks=0, new_auxiliary_trials=0, production_episodes=0)
    output.mkdir(parents=True)
    try:
        delivery = _delivered_archive(readback, "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE")
        actual = json.loads(_delivered_file(readback, delivery, "analysis/report.json").read_text())
        history_path = _delivered_file(readback, delivery, "source_config.yaml")
        history = yaml.safe_load(history_path.read_text())
        config = yaml.safe_load(_delivered_file(readback, delivery,
            "Libs/configs/neural_population_pfc_profile_history_readback.yaml").read_text())
        require(config["schema"] == "pfc_profile_history_readback_v1"
                and config["source_config_sha256"] == sha256_file(history_path), "FROZEN_HISTORY_CONFIG")
        requests = readback_requests(config, history)
        index = actual.get("readback_group_index")
        require(type(index) is int and 0 <= index < len(requests), "REGISTERED_GROUP_INDEX")
        request = requests[index]
        require(actual.get("status") == "COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK"
                and actual.get("full_source_requests") is True and request["full_source_requests"] is True
                and actual.get("source_rng_seed") == request["source_rng_seed"]
                and actual.get("readback_group") == request["group_name"]
                and actual.get("original_source_blocks") == 1
                and actual.get("paired_profiles") == len(request["profile_keys"])
                and [row["key"] for row in actual["pairs"]] == request["profile_keys"],
                "DELIVERED_REGISTERED_GROUP_IDENTITY")
        require(actual.get("scientific_fidelity_qualified") is False
                and actual.get("confirmation_execution_released") is False
                and actual.get("new_independent_source_blocks") == 0 and actual.get("production_episodes") == 0,
                "GROUP_UNRELEASED_SCOPE")
        expected = [dict(source_rng_seed=request["source_rng_seed"], profile_key=key, clock=clock,
                         source_label=request["source_runs"][clock]["source_label"])
                    for key in request["profile_keys"] for clock in ("coarse", "fine")]
        report.update(expected_clock_profiles=expected, source_rng_seed=request["source_rng_seed"],
            readback_group=request["group_name"], readback_group_index=index,
            readback_delivery_sha256=sha256_file(readback / "delivery.json"),
            history_config_sha256=sha256_file(history_path))
        sources = {}
        for clock in ("coarse", "fine"):
            cost = actual["source_costs"][clock]
            archive = project_root / cost["history_execution_archive"]
            require(cost["history_execution_archive"] == request["source_runs"][clock]["archive"],
                    "HISTORY_EXECUTION_ARCHIVE_BINDING")
            source_delivery = _delivered_archive(archive, "COMPLETE_PFC_PROFILE_HISTORY_COMPLETE")
            source_report = json.loads(_delivered_file(archive, source_delivery, "analysis/report.json").read_text())
            runtime = json.loads(_delivered_file(archive, source_delivery, "runtime_config.json").read_text())
            label = request["source_runs"][clock]["source_label"]
            require(source_report.get("status") == "COMPLETE_PFC_PROFILE_SOURCE_HISTORY"
                    and source_report.get("source_rng_seed") == request["source_rng_seed"]
                    and source_report.get("source_label") == label
                    and source_report.get("requested_profile_keys") == source_report.get("completed_profile_keys")
                    and source_report["completed_profile_keys"] == request["source_runs"][clock]["expected_profile_keys"]
                    and runtime.get("source_binding") == cost["source_binding"]
                    and {key: value for key, value in runtime.items() if key != "source_binding"} == history,
                    "COMPLETE_HISTORY_SOURCE_BINDING")
            sources[clock] = (archive, source_delivery, _source_config(project_root, cost["source_binding"]))
        pair_index = {row["key"]: row for row in actual["pairs"]}
        for identity in expected:
            key, clock = identity["profile_key"], identity["clock"]
            archive, source_delivery, source_config = sources[clock]
            prefix = "analysis/frames/" + key + "/"
            try:
                paths = {name: _delivered_file(archive, source_delivery, prefix + name)
                         for name in ("report.json", "selected_frame.json", "classified_frame.json", "integrated_frame.json")}
                profile_report = json.loads(paths["report.json"].read_text())
                observed = pair_index[key]["per_clock"][clock]
                require(profile_key(profile_report["profile"], profile_report["auxiliary_repeat"]) == key
                        and profile_report["source_rng_seed"] == identity["source_rng_seed"]
                        and profile_report["source_label"] == identity["source_label"]
                        and profile_report["dt_ms"] == source_config["dt_ms"]
                        and all(profile_report[name] == observed[name]
                                for name in ("sample_mean", "integrated_histories", "inner_trials"))
                        and observed["source_binding"] == actual["source_costs"][clock]["source_binding"],
                        "PROFILE_READBACK_BOUND_VALUES")
                integrated = json.loads(paths["integrated_frame.json"].read_text())
                eligible = [row for row in integrated if row["category"] == "conditional_integration_required"]
                if eligible:
                    tar_name = prefix + "integration_details.tar"
                    present = any(row["path"] == tar_name for row in source_delivery["files"])
                    if not present:
                        report["missing_histograms"].append(dict(identity, path=tar_name))
                        raise ValueError("PFC_HISTORY_TAIL_BOUND_MISSING_HISTOGRAM_ARCHIVE:" + tar_name)
                    tar_path = _delivered_file(archive, source_delivery, tar_name)
                    with tarfile.open(tar_path, "r:") as saved:
                        for row in eligible:
                            member = f"sample_{row['sample_index']:03d}/integration/weighted_first_hit_histogram.npz"
                            try:
                                require(saved.getmember(member).isfile(), "REGULAR_HISTOGRAM_MEMBER")
                            except KeyError:
                                report["missing_histograms"].append(dict(identity, path=tar_name, member=member))
                                raise ValueError("PFC_HISTORY_TAIL_BOUND_MISSING_HISTOGRAM_MEMBER:" + member)
                result = run(source_config, settings, paths["report.json"].parent, output / "profiles" / key / clock)
                require(result["profile"] == profile_report["profile"]
                        and result["auxiliary_repeat"] == profile_report["auxiliary_repeat"], "BOUND_PROFILE_IDENTITY")
                report["bounds"].append(dict(identity, result=result))
                report["completed_clock_profiles"].append(identity)
            except Exception as error:
                report["failures"].append(dict(identity, error=dict(type=type(error).__name__, message=str(error))))
        completed = report["completed_clock_profiles"]
        report["missing_clock_profiles"] = [identity for identity in expected if identity not in completed]
        require(not report["failures"] and completed == expected, "INCOMPLETE_GROUP_TAIL_BOUNDS")
        report.update(status="COMPLETE_FINITE_ARCHIVED_HISTORY_GROUP_TAIL_BOUND", group_complete=True,
                      completed_group_clock_profiles=len(completed), completed_group_event_bounds=2 * len(completed))
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        if report["expected_clock_profiles"]:
            report["missing_clock_profiles"] = [identity for identity in report["expected_clock_profiles"]
                                                if identity not in report["completed_clock_profiles"]]
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--readback-group", type=Path)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.readback_group is not None:
        if args.project_root is None or args.source_config is not None or args.profile is not None:
            parser.error("--readback-group requires --project-root and excludes --source-config/--profile")
        result = run_readback_group(settings, args.readback_group, args.project_root, args.output)
        print(json.dumps(dict(status=result["status"], completed_group_clock_profiles=result["completed_group_clock_profiles"],
                              family_complete=result["family_complete"])))
    else:
        if args.source_config is None or args.profile is None or args.project_root is not None:
            parser.error("single profile requires --source-config and --profile")
        result = run(yaml.safe_load(args.source_config.read_text()), settings, args.profile, args.output)
        print(json.dumps(dict(status=result["status"], event_bounds={event: result["event_bounds"][event]["upper_bound"]
                                                                   for event in EVENTS})))


if __name__ == "__main__":
    main()

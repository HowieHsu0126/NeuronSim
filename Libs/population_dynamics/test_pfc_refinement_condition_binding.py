"""Test control-family applicability and exact signed-estimand perturbation."""
from fractions import Fraction
import hashlib
import json

import pytest

from population_dynamics import pfc_refinement_condition_binding as cb
from population_dynamics.pfc_refinement_input_law import mean_parent_envelope, routing_law


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(data)
    return dict(path=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class ControlFixture:
    def __init__(self, root):
        self.root = root
        self.prior = dict(status="COMPLETE_INDEXED_PFC_REFINEMENT_PARAMETER_ARITHMETIC_BINDING",
                          parameter_arithmetic_binding_verified=True, source_bindings=[], shared_two_level_trees=64)
        for key in ("finite_prng_law_certified", "finite_sampler_law_certified", "source_history_confidence_certified",
                    "confirmation_0p01_128_certified", "confirmation_execution_released", "scientific_fidelity_qualified",
                    "production_qualified", "numerical_margins_changed", "error_budget_allocated"):
            self.prior[key] = False
        self.precision = dict(settings=dict(native_source_reports=[]))
        self.reports, self.deliveries, self.statuses = {}, {}, {}
        for block in range(95001, 95033):
            cases = [dict(case_id=f"pfc_{block}_{condition}", source_rng_seed=block,
                          stimulus_strength_nA={"omission": 0., "cue": .06, "strong_cue": .5}[condition],
                          distractor_strength_nA=0.) for condition in cb.CONDITIONS]
            driver = dict(binomial_n=1000, n_e=80000, n_i=20000, total_ticks=100000,
                          ticks_generated=100000, binomial_probability=1.4 * (.05 * 1e-3),
                          external_rate_Hz=1.4, dt_seconds=.05 * 1e-3, random_stream_policy="source_global",
                          seeds=dict(source_rng_seed=block))
            for label, dt in cb.SOURCE_CLOCKS.items():
                archive = root / "archives" / f"{block}_{label[-1] if label != 'native' else 'native'}"
                for case in cases:
                    condition = case["case_id"].split("_", 2)[2]
                    ticks = round(5000 / dt)
                    config = dict(cases=cases, excitatory_neurons=80000, inhibitory_neurons=20000,
                                  excitatory_preference_bins=16, dt_ms=dt, duration_ms=5000.)
                    report = dict(resolved_config=config, selected_case=case, source_execution_complete=True,
                                  completed_duration_ticks=ticks,
                                  inputs_used=dict(ticks_consumed=ticks, E=dict(values=80000 * ticks),
                                                   I=dict(values=20000 * ticks)))
                    if label == "native":
                        path = archive / condition / "report.json"
                        report.update(status="PFC_SOURCE_EXECUTION_COMPLETE", driver=driver)
                    else:
                        clock = "dt_" + str(dt).replace(".", "p")
                        path = archive / "source_cases" / clock / case["case_id"] / "report.json"
                        report.update(status="COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE", code=cb.SUPPORTED_CODE_HASHES,
                            drive=dict(refinement_factor=2 if label.startswith("half") else 4,
                                       ticks_generated=ticks, total_ticks=ticks, n_e=80000, n_i=20000,
                                       all_refinement_levels_generated=True, coupling=dict(coarse_source=driver)),
                            coarse_reference=dict(report_sha256=self.native_digest(block, condition)))
                    self.reports[block, label, condition] = path, report
                    entry = write_json(path, report)
                    if label == "native":
                        self.precision["settings"]["native_source_reports"].append(dict(
                            source_rng_seed=block, case_id=case["case_id"], cohort=cb.LABELS[0],
                            path=str(path.relative_to(root)), report_sha256=entry["sha256"]))
                    else:
                        entry["path"] = str(path.relative_to(archive))
                        self.deliveries.setdefault(archive, {})[entry["path"]] = entry
                    if condition == "strong_cue":
                        self.prior["source_bindings"].append(dict(source_block=block, source_label=label,
                            archive=str(archive.relative_to(root)), report_sha256=entry["sha256"]))
        self.seal()

    def native_digest(self, block, condition):
        return next(row["report_sha256"] for row in self.precision["settings"]["native_source_reports"]
                    if row["case_id"] == f"pfc_{block}_{condition}")

    def seal(self):
        for archive, entries in self.deliveries.items():
            entry = write_json(archive / "delivery.json", dict(
                verified=True, status=self.statuses.get(archive, "COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"
                    if archive.name.endswith("_B") else "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK"),
                files=list(entries.values())))
            for row in self.prior["source_bindings"]:
                if row["archive"] == str(archive.relative_to(self.root)):
                    row["delivery_sha256"] = entry["sha256"]

    def mutate_fine(self, block, label, condition, change):
        path, report = self.reports[block, label, condition]
        change(report)
        entry = write_json(path, report)
        archive = self.root / next(row["archive"] for row in self.prior["source_bindings"]
                                  if row["source_block"] == block and row["source_label"] == label)
        entry["path"] = str(path.relative_to(archive))
        self.deliveries[archive][entry["path"]] = entry
        self.seal()


def test_complete_control_family_metadata(tmp_path):
    fixture = ControlFixture(tmp_path)
    report = cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)
    assert report["source_report_entries"] == 480
    assert report["added_condition_report_entries"] == 320
    assert report["coupled_labels_with_equal_control_input_declarations"] == 128
    assert report["input_payload_identity_verified"] is False
    assert not report["neural_payloads_read"]


@pytest.mark.parametrize("change,error", [
    (lambda r: r["selected_case"].update(stimulus_strength_nA=.5), "SOURCE_CONTROL_CHANGED"),
    (lambda r: r.update(completed_duration_ticks=1), "COMPLETE_SOURCE_CLOCK"),
    (lambda r: r["drive"].update(returned_dtype="wrong"), "INPUT_DECLARATIONS_DIFFER"),
    (lambda r: r.update(code={}), "IMPLEMENTATION_CHANGED"),
    (lambda r: r["coarse_reference"].update(report_sha256="0" * 64), "COARSE_REFERENCE_CHANGED"),
])
def test_authenticated_but_invalid_added_condition_rejected(tmp_path, change, error):
    fixture = ControlFixture(tmp_path)
    fixture.mutate_fine(95001, "half_A", "cue", change)
    with pytest.raises(ValueError, match=error):
        cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)


def test_missing_control_report_rejected(tmp_path):
    fixture = ControlFixture(tmp_path)
    archive = tmp_path / "archives/95001_A"
    fixture.deliveries[archive].pop("source_cases/dt_0p025/pfc_95001_cue/report.json")
    fixture.seal()
    with pytest.raises(ValueError, match="MISSING_DECLARED_CONDITION_REPORT"):
        cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)


def test_prior_scope_cannot_be_promoted(tmp_path):
    fixture = ControlFixture(tmp_path)
    fixture.prior["finite_sampler_law_certified"] = True
    with pytest.raises(ValueError, match="PRIOR_SCOPE_CHANGED"):
        cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)


def test_independent_fine_clock_references_do_not_replace_parent(tmp_path):
    fixture = ControlFixture(tmp_path)
    references = fixture.precision["settings"]["native_source_reports"]
    for label in cb.LABELS[1:]:
        references.extend([dict(row, cohort=label, path="unused/fine_reference.json", report_sha256="0" * 64)
                           for row in references[:96]])
    reader = cb._Reader()
    report = cb.bind_conditions(reader, fixture.prior, fixture.precision, tmp_path)
    assert report["source_report_entries"] == 480
    assert all("unused/fine_reference" not in str(path) for path in reader.snapshots)


def test_duplicate_parent_reference_rejected(tmp_path):
    fixture = ControlFixture(tmp_path)
    references = fixture.precision["settings"]["native_source_reports"]
    references.append(dict(references[0]))
    with pytest.raises(ValueError, match="UNIQUE_NATIVE_SOURCE_REFERENCE"):
        cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)


def test_supported_development_pilot_delivery_retained(tmp_path):
    fixture = ControlFixture(tmp_path)
    fixture.statuses[tmp_path / "archives/95001_A"] = "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT"
    fixture.seal()
    result = cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)
    assert result["source_report_entries"] == 480


def test_pilot_cannot_replace_b_repeat_delivery(tmp_path):
    fixture = ControlFixture(tmp_path)
    fixture.statuses[tmp_path / "archives/95001_B"] = "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT"
    fixture.seal()
    with pytest.raises(ValueError, match="COUPLED_SOURCE_DELIVERY_CHANGED"):
        cb.bind_conditions(cb._Reader(), fixture.prior, fixture.precision, tmp_path)


@pytest.mark.parametrize("delta", [Fraction(0), Fraction(1, 4), Fraction(1)])
def test_product_tv_and_signed_bias(delta):
    assert cb.expectation_bias(Fraction(2), delta) == 2 * delta
    assert cb.expectation_bias(Fraction(2), delta, 2) == 2 * (2 * delta - delta * delta)


def test_exact_covariance_example_requires_order_two():
    # P is zero; Q is Bernoulli(1/4). H(x,y)=(x-y)^2/2.
    delta, width = Fraction(1, 4), Fraction(1, 2)
    q = {0: Fraction(3, 4), 1: Fraction(1, 4)}
    actual = sum(q[x] * q[y] * Fraction((x - y) ** 2, 2) for x in q for y in q)
    assert actual == Fraction(3, 16)
    assert cb.expectation_bias(width, delta) < actual
    assert actual <= cb.expectation_bias(width, delta, 2)


def test_signed_oscillation_bound_attainable():
    # g=-1 at 0, +1 at 1. Its width, rather than its radius, multiplies TV.
    delta = Fraction(1, 4)
    difference = abs(Fraction(-1) - (Fraction(-1) * (1 - delta) + delta))
    assert difference == cb.expectation_bias(Fraction(2), delta)


@pytest.mark.parametrize("width,tv,order", [
    (Fraction(-1), Fraction(1, 4), 1), (Fraction(1), Fraction(2), 1),
    (Fraction(1), Fraction(1, 4), 496), (1., Fraction(1, 4), 1),
])
def test_invalid_bias_inputs_rejected(width, tv, order):
    with pytest.raises(ValueError, match="VALID_EXPECTATION_BIAS_INPUTS"):
        cb.expectation_bias(width, tv, order)


def endpoint(family, status="ESTIMATED_EMPIRICAL_SPREAD", **extra):
    ident = dict(left_cohort=cb.LABELS[0], right_cohort=cb.LABELS[1], condition="cue", phase="cue", **extra)
    if family == "paired_response":
        ident.pop("condition")
    return dict(family=family, definition_status=status, endpoint_identity=json.dumps(ident), physical_maximum_count="3")


def test_full_summary_preserves_units_undefined_and_structural_identities():
    domains = [endpoint("count_cdf", field_index=0, count_threshold=0, window_ms=5),
               endpoint("field_mean", field_index=0),
               endpoint("field_lag_covariance", leading_field=0, trailing_field=0, lag_ms=0),
               endpoint("field_lag_covariance", leading_field=0, trailing_field=1, lag_ms=5),
               endpoint("source_response", metric="vector_x_hz"),
               endpoint("paired_response", metric="vector_x_hz", contrast="cue_minus_omission"),
               endpoint("count_cdf", "STRUCTURAL_ZERO_CONTRAST", field_index=0, count_threshold=3, window_ms=5),
               endpoint("field_lag_covariance", "UNDEFINED_NO_WITHIN_PHASE_PAIRS", leading_field=0, trailing_field=0, lag_ms=1000)]
    margins = [dict(row, domain_row_index=i, margin="" if i == 7 else "0" if i == 6 else ".01",
                    units="probability" if row["family"] == "count_cdf" else "Hz^2" if row["family"] == "field_lag_covariance" else "Hz")
               for i, row in enumerate(domains)]
    result = cb.summarize_bias(domains, margins, [Fraction(600)] * 17,
                              {"vector_x_hz": (Fraction(-600), Fraction(600))}, Fraction(1, 4))
    assert result["domain_rows"] == 8
    assert result["definition_status_counts"]["UNDEFINED_NO_WITHIN_PHASE_PAIRS"] == 1
    worst = result["maximum_conditional_bias_ratio_by_family"]
    assert cb._exact(worst["count_cdf"]["conditional_parameter_bias_upper"]) == Fraction(1, 2)
    assert cb._exact(worst["field_lag_covariance"]["physical_score_or_kernel_oscillation"]) == 720000
    assert cb._exact(worst["paired_response"]["physical_score_or_kernel_oscillation"]) == 4800
    assert not result["source_confidence_bands_generated"]
    assert not result["bias_subtracted_from_frozen_budget"]


def test_summary_rejects_truncated_margin_family():
    with pytest.raises(ValueError, match="shorter"):
        cb.summarize_bias([endpoint("count_cdf")], [], [], {}, Fraction(0))


def test_parent_moment_recurrence_matches_exact_category_mean():
    report = mean_parent_envelope(7e-5, trials=1000, units=100000, parent_ticks=100000, levels=2)
    first, second = report["levels"]
    law = routing_law(7e-5)
    mean = Fraction(1000) * Fraction.from_float(7e-5)
    child_mean = mean * (law["actual_categories"]["both"] + law["actual_categories"]["left_only"])
    assert cb._exact(first["parent_count_mean_upper"]) == mean
    assert cb._exact(second["parent_count_mean_upper"]) == child_mean
    assert child_mean == mean * (1 - law["rho"]) / 2
    assert cb._exact(report["unclamped_union"]) == sum(
        cb._exact(row["first_mismatch_union_contribution"]) for row in report["levels"])
    assert not report["native_parent_mean_verified_for_finite_sampler"]
    assert not report["native_parent_mean_estimated_from_observed_counts"]


def test_parent_moment_against_finite_two_event_enumeration():
    from itertools import product
    law = routing_law(.1)
    p = law["parent_probability"]
    # Include inactive root events explicitly. No pseudorandom draws occur.
    outcomes = [(0, 1 - p), *[(int(name in ("both", "left_only")), p * mass)
                              for name, mass in law["actual_categories"].items()]]
    exact_child = sum((a + b) * pa * pb for (a, pa), (b, pb) in product(outcomes, repeat=2))
    result = mean_parent_envelope(.1, trials=2, units=1, parent_ticks=1, levels=2)
    assert cb._exact(result["levels"][1]["parent_count_mean_upper"]) == exact_child


@pytest.mark.parametrize("bad", [0, -1, 1., True])
def test_invalid_moment_dimensions_rejected(bad):
    with pytest.raises(ValueError, match="POSITIVE_MOMENT_DIMENSIONS"):
        mean_parent_envelope(.1, trials=2, units=bad, parent_ticks=1, levels=2)

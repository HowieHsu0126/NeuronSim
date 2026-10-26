"""Stratum allocation, exact conditional variance, and native history callbacks."""
from copy import deepcopy
from itertools import combinations, product
import json

import numpy as np
import pytest
import yaml

from . import pfc_first_passage_history as history
from .pfc_history_timing_census import census, pair
from .pfc_stratified_history import allocation, paired_components, sampling_callbacks, stratified_components, validate_settings
from .test_pfc_first_passage_history import fixture as history_fixture


def populations(**sizes):
    result = np.zeros(16, dtype=np.int64)
    for h, n in sizes.items():
        result[int(h[1:])] = n
    return result


def paired_census_fixture(multispike=False):
    configs, raws, censuses = {}, {}, {}
    for clock, dt in (("coarse", .05), ("fine", .025)):
        config = dict(excitatory_neurons=256, inhibitory_neurons=8, excitatory_preference_bins=16,
                      duration_ms=5000., dt_ms=dt, weight_scaling_factor=.0256)
        events = []
        for window in range(10):
            for coarse_bin in (-1, 0, 1, 2):
                for fine_bin in (-1, 0, 1, 2):
                    h = (coarse_bin + 1) * 4 + fine_bin + 1
                    time_bin = coarse_bin if clock == "coarse" else fine_bin
                    if time_bin >= 0:
                        first_ms = 102. - time_bin + 5. * window
                        events.append((round(first_ms / dt), 80 + h))
        if multispike and clock == "coarse":
            events.append((2040, 95))
        events.sort()
        ticks = np.asarray([event[0] for event in events], dtype=np.int64)
        units = np.asarray([event[1] for event in events], dtype=np.int64)
        raw = dict(spike_ticks=ticks, spike_unit_indices=units, spike_seconds=ticks * (dt * 1e-3),
                   native_spike_counts=np.bincount(units, minlength=264))
        configs[clock], raws[clock] = config, raw
        censuses[clock] = census(config, raw, field_index=5, phase="cue", window_ms=5., time_edges_ms=[0., 1., 2., 3.])
    report, arrays = pair(*censuses["coarse"], *censuses["fine"])
    return configs, raws, report, arrays


def scored_rows(config, raw, rows):
    result = history.classify(config, history.reference.validate_tape(config, raw), rows)
    for row in result:
        eligible = row["category"] in ("conditional_integration_required", "suppressed_network_reference_required")
        row.update(conditional_estimate=.2 if eligible else 0., inner_monte_carlo_variance=.01 if eligible else 0.)
    return result


def test_allocation_cap_exact_ties_empty_and_singleton_strata():
    common = dict(source_rng_seed=95001, case_id="pfc_95001_strong_cue", expected_units=100000,
                  field_index=5, phase="cue", window_ms=5.)
    for windows, samples, namespace in ((64, 100000, 3), (640, 10000, 4)):
        settings = dict(common, sample_size=windows, importance_sampling={"samples": samples},
                        integration_seed_namespace=namespace,
                        sampling_seed_components=[20261009, 95001, 5, namespace])
        assert validate_settings(settings) is None
        assert windows * samples == 6400000
    for windows, samples, namespace in ((64, 10000, 3), (640, 100000, 4), (128, 50000, 3),
                                         (64, 100000, 4), (640, 10000, 3), (64.0, 100000, 3)):
        invalid = dict(common, sample_size=windows, importance_sampling={"samples": samples},
                       integration_seed_namespace=namespace,
                       sampling_seed_components=[20261009, 95001, 5, namespace])
        with pytest.raises(ValueError, match="REGISTERED_WINDOW_INTEGRATION_BUDGET"):
            validate_settings(invalid)
    with pytest.raises(ValueError, match="SAMPLING_SEED_NAMESPACE"):
        validate_settings(dict(settings, sampling_seed_components=[20261009, 95001, 5, 3]))
    for key, value in (("case_id", "pfc_95002_strong_cue"), ("expected_units", 99999),
                       ("field_index", 4), ("phase", "baseline"), ("window_ms", 10.)):
        with pytest.raises(ValueError, match="FROZEN_FRESH_STRATIFIED_DESIGN"):
            validate_settings(dict(settings, **{key: value}))
    plan = allocation(populations(h0=90, h1=10, h2=10), budget=5)
    assert [row["sample_size"] for row in plan[:3]] == [0, 3, 2]
    assert plan[0]["weight"] == 90 / 110 and plan[0]["inclusion_probability"] == 0.
    assert plan[3]["sampling"] == "empty" and plan[3]["inclusion_probability"] is None
    assert plan[1]["inclusion_probability"] == .3
    plan = allocation(populations(h0=7, h1=1, h2=3))
    assert [row["sample_size"] for row in plan[:3]] == [0, 1, 3]
    assert plan[1]["sampling"] == plan[2]["sampling"] == "census"
    plan = allocation(np.full(16, 100, dtype=np.int64))
    assert sum(row["sample_size"] for row in plan) == 64
    assert [row["sample_size"] for row in plan[1:5]] == [5] * 4
    assert [row["sample_size"] for row in plan[5:]] == [4] * 11
    plan = allocation(populations(h0=50))
    assert sum(row["sample_size"] for row in plan) == 0 and plan[0]["weight"] == 1.
    with pytest.raises(ValueError, match="BUDGET_BELOW_STRATUM_MINIMUM"):
        allocation(populations(h1=10, h2=10), budget=1)
    with pytest.raises(ValueError, match="SIXTEEN_FULL_FRAME_COUNTS"):
        allocation(np.zeros(16, dtype=np.int64))


def test_callbacks_pair_ids_validate_census_and_retain_unresolved_multispikes():
    configs, raws, report, arrays = paired_census_fixture()
    kwargs = dict(field=5, phase="cue", window_ms=5., sample_size=64, seed_components=[20261009, 95001, 5, 3])
    coarse_sampler, coarse_summary = sampling_callbacks(report, arrays, "coarse")
    fine_sampler, fine_summary = sampling_callbacks(report, arrays, "fine")
    coarse, coarse_frame = coarse_sampler(configs["coarse"], **kwargs)
    fine, fine_frame = fine_sampler(configs["fine"], **kwargs)
    assert coarse_frame == fine_frame
    assert len(coarse) == len(fine) == 64
    assert len({row["frame_index"] for row in coarse}) == 64
    assert all(c["frame_index"] == f["frame_index"] and c["unit_index"] == f["unit_index"]
               and c["window_start_ms"] == f["window_start_ms"]
               and c["window_start_tick"] * 2 == f["window_start_tick"] for c, f in zip(coarse, fine))
    assert all(row["joint_time_bin"] > 0 for row in coarse)
    again, frame_again = coarse_sampler(configs["coarse"], **kwargs)
    assert (again, frame_again) == (coarse, coarse_frame)
    changed, _ = coarse_sampler(configs["coarse"], **dict(kwargs, seed_components=[20261009, 95001, 5, 4]))
    assert [row["frame_index"] for row in changed] != coarse_frame["selected_frame_indices"]
    coarse = scored_rows(configs["coarse"], raws["coarse"], coarse)
    fine = scored_rows(configs["fine"], raws["fine"], fine)
    assert coarse_summary(coarse, coarse_frame)["census_classification_matches"]
    assert fine_summary(fine, fine_frame)["census_classification_matches"]
    assert any(c["category"] == "no_first_spike" and f["category"] == "conditional_integration_required"
               for c, f in zip(coarse, fine))
    corrupted = deepcopy(coarse)
    next(row for row in corrupted if row["first_spike_tick"] is not None)["first_spike_tick"] += 1
    with pytest.raises(ValueError, match="CLASSIFICATION_DIFFERS_FROM_CENSUS"):
        coarse_summary(corrupted, coarse_frame)
    with pytest.raises(ValueError, match="SELECTED_FRAME_CHANGED"):
        coarse_summary(coarse[:-1], coarse_frame)
    configs, raws, report, arrays = paired_census_fixture(multispike=True)
    sampler, summary = sampling_callbacks(report, arrays, "coarse")
    selected, frame = sampler(configs["coarse"], **dict(kwargs, sample_size=150))
    rows = scored_rows(configs["coarse"], raws["coarse"], selected)
    assert len(rows) == 150 and any(row["observed_count"] == 2 for row in rows)
    with pytest.raises(ValueError, match="UNRESOLVED_MULTISPIKE_RETAINED"):
        summary(rows, frame)
    with pytest.raises(ValueError, match="SUPPRESSED_NETWORK_REPLAY_REQUIRED_NO_ROWS_DROPPED"):
        history.extract(configs["coarse"], {}, history.reference.validate_tape(configs["coarse"], raws["coarse"]), rows)


def test_exact_1536_case_enumeration_preserves_paired_sampling_covariance():
    # Six SRSWOR subsets, four independent Bernoulli scores per selected ID.
    # Each mean uses two scores 2*z*Bernoulli(1/2), with unbiased inner variance.
    zc, zf = np.asarray([0., .25, .5, .75]), np.asarray([.25, 0., .75, .5])
    sizes = populations(h0=3, h6=4, h10=1)
    means, estimated, coarse_means, fine_means = [], [], [], []
    for subset in combinations(range(4), 2):
        for bits in product((0, 1), repeat=8):
            b = np.asarray(bits).reshape(2, 2, 2)
            selected = np.asarray(subset)
            qc = zc[selected] * b[:, 0, :].sum(axis=1)
            qf = zf[selected] * b[:, 1, :].sum(axis=1)
            vc = zc[selected]**2 * (b[:, 0, 0] - b[:, 0, 1])**2
            vf = zf[selected]**2 * (b[:, 1, 0] - b[:, 1, 1])**2
            labels = [6, 6, 10]
            coarse = stratified_components([*qc, .25], [*vc, 0.], labels, sizes, budget=3)
            fine = stratified_components([*qf, .5], [*vf, 0.], labels, sizes, budget=3)
            contrast = stratified_components([*(qf - qc), .25], [*(vc + vf), 0.], labels,
                                              sizes, budget=3, allow_signed=True)
            means.append(contrast["sample_mean"])
            estimated.append(contrast["within_archive_mean_variance_diagnostic"])
            coarse_means.append(coarse["sample_mean"])
            fine_means.append(fine["sample_mean"])
    assert len(means) == 1536
    assert np.mean(means) == pytest.approx(1 / 32, abs=1e-15)
    assert np.var(means) == pytest.approx(25 / 768, rel=1e-13)
    assert np.mean(estimated) == pytest.approx(25 / 768, rel=1e-13)
    covariance = np.mean((np.asarray(coarse_means) - np.mean(coarse_means))
                         * (np.asarray(fine_means) - np.mean(fine_means)))
    assert covariance == pytest.approx(1 / 256, abs=1e-15)
    assert np.var(means) == pytest.approx(np.var(coarse_means) + np.var(fine_means) - 2 * covariance)


def test_signed_zero_hit_singleton_and_missing_stratum_variance_contracts():
    sizes = populations(h0=6, h15=4)
    zero = stratified_components([0., 0.], [0., 0.], [15, 15], sizes, budget=2)
    assert zero["sample_mean"] == zero["within_archive_mean_variance_diagnostic"] == 0.
    assert zero["sample_denominator"] == 2 and zero["stratum_diagnostics"][0]["weight"] == .6
    signed = stratified_components([-.2, -.2], [.5, .5], [15, 15], sizes, budget=2, allow_signed=True)
    assert signed["sample_mean"] == pytest.approx(-.08)
    assert signed["stratum_diagnostics"][15]["latent_unit_window_variance_diagnostic"] == -.5
    assert signed["within_archive_mean_variance_diagnostic"] == pytest.approx(.02)
    assert signed["inner_mc_mean_variance_diagnostic"] == pytest.approx(.04)
    assert signed["unit_window_sampling_mean_variance_diagnostic"] == pytest.approx(-.02)
    singleton = stratified_components([1.2], [.2], [15], populations(h0=9, h15=1))
    row = singleton["stratum_diagnostics"][15]
    assert singleton["sample_mean"] == .12 and not singleton["importance_estimates_clipped"]
    assert singleton["within_archive_mean_variance_diagnostic"] == pytest.approx(.002)
    assert row["sampled_estimate_variance"] is None and row["latent_unit_window_variance_diagnostic"] is None
    assert row["weighted_sampling_variance"] == 0.
    empty = stratified_components([], [], [], populations(h0=10))
    assert empty["sample_denominator"] == 0 and empty["sample_mean"] == 0.
    with pytest.raises(ValueError, match="MISSING_OR_EXTRA_STRATUM"):
        stratified_components([0.] * 4, [0.] * 4, [1] * 4, populations(h1=4, h2=4), budget=4)


def test_paired_contrast_sign_and_finite_importance_estimates_above_one():
    sizes = populations(h0=6, h15=4)
    frame = dict(sample_size=2, nominal_sample_budget=2, selected_frame_indices=[6, 7], allocation=allocation(sizes, 2))
    coarse, fine = [], []
    for index, (c, f) in enumerate(((1.2, .2), (.8, .4))):
        base = dict(sample_index=index, frame_index=6 + index, unit_index=index, window_start_ms=100.,
                    window_stop_ms=105., joint_time_bin=15, inclusion_probability=.5, observed_count=1,
                    category="conditional_integration_required")
        coarse.append(dict(base, first_spike_tick=2000 + index, conditional_estimate=c, inner_monte_carlo_variance=.01))
        fine.append(dict(base, first_spike_tick=4000 + 3 * index, conditional_estimate=f, inner_monte_carlo_variance=.02))
    stats, pairs = paired_components(coarse, fine, frame)
    assert stats["fine_minus_coarse_count_ge2"] == pytest.approx(-.28)
    assert stats["fine_minus_coarse_cdf_le1"] == pytest.approx(.28)
    assert stats["within_archive_mean_variance_diagnostic"] == pytest.approx(.0084)
    assert pairs[0]["coarse_estimate"] == 1.2
    assert not stats["third_spike_probability_established"]
    changed = deepcopy(fine)
    changed[0]["unit_index"] += 10
    with pytest.raises(ValueError, match="PAIRED_PHYSICAL_IDS"):
        paired_components(coarse, changed, frame)


def test_history_entrypoint_uses_callbacks_on_real_tiny_reference_fixture(tmp_path, monkeypatch):
    config, case, counts_e, counts_i, tape = history_fixture()
    monkeypatch.setitem(history.PHASES, "tiny_fixture", [0., 5.])
    case = dict(case, source_rng_seed=95001)
    raw = dict(spike_ticks=tape.events[:, 0], spike_unit_indices=tape.events[:, 1],
               spike_seconds=tape.events[:, 0] * (config["dt_ms"] * 1e-3), native_spike_counts=tape.unit_counts)
    one = census(config, raw, field_index=0, phase="tiny_fixture", window_ms=5., time_edges_ms=[0., 1., 2., 3.])
    report, arrays = pair(*one, *one)
    sampler, summary = sampling_callbacks(report, arrays, "coarse")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    (source / "report.json").write_text(json.dumps(dict(resolved_config=config, selected_case=case)))
    np.savez_compressed(source / "raw.npz", **raw)
    initial = np.full(40, -60.)
    initial[:2] = -50.000001
    np.save(source / "initial_voltage.npy", initial)

    class FixtureDrive:
        def metadata(self):
            return dict(scope="callback_test_recorded_input_fixture")

    # Keep actual history.run, classification and reference reconstruction. The
    # unrelated native RNG construction and inner IS calculation are isolated.
    monkeypatch.setattr(history, "create_input_drive", lambda *args: (initial.copy(), FixtureDrive()))
    monkeypatch.setattr(history, "prefix_state_arrays", lambda *args: {"fixture_prefix": np.asarray(0)})
    original_extract = history.extract

    def recorded_extract(cfg, selected_case, validated, rows, **kwargs):
        return original_extract(cfg, selected_case, validated, rows, counts_E=counts_e, counts_I=counts_i)

    monkeypatch.setattr(history, "extract", recorded_extract)
    inner_seeds = []
    inner_sample_counts = []

    def integrate(cfg, conditional, forcing, settings, output):
        inner_seeds.append(settings["seed_components"])
        inner_sample_counts.append(settings["samples"])
        return dict(stopped_prefix=dict(estimate=1.2 if conditional["unit_index"] == 0 else .4,
                                        empirical_score_variance=.04), proposal_hits=1)

    monkeypatch.setattr(history.importance, "run", integrate)
    for windows, samples, namespace in ((64, 100000, 3), (640, 10000, 4)):
        inner_seeds.clear()
        inner_sample_counts.clear()
        settings = dict(source_rng_seed=95001, case_id=case["case_id"], dt_ms=.05, expected_units=40,
                        field_index=0, phase="tiny_fixture", window_ms=5., sample_size=windows,
                        sampling_seed_components=[20261009, 95001, 0, namespace], integration_seed_namespace=namespace,
                        source={"role": "test_fixture"}, importance_sampling={"samples": samples})
        output = tmp_path / f"history_namespace_{namespace}"
        actual = history.run(settings, source, output, frame_sampler=sampler, frame_summary=summary)
        assert actual["status"] == "COMPLETE_PFC_FIRST_PASSAGE_HISTORY_PILOT"
        assert actual["sample_denominator"] == actual["integrated_histories"] == 2
        assert actual["nominal_sample_budget"] == windows
        assert actual["sample_mean"] == pytest.approx(.8)
        assert actual["census_classification_matches"]
        assert "empirical_variance_across_sampled_estimates" not in actual
        assert len(inner_seeds) == 2 and all(seed[-2:] == [50, namespace] for seed in inner_seeds)
        assert inner_sample_counts == [samples, samples]
        integrated = json.loads((output / "integrated_frame.json").read_text())
        assert len(integrated) == 2 and max(row["conditional_estimate"] for row in integrated) == 1.2
        assert all(row["inner_monte_carlo_variance"] == pytest.approx(.04 / samples) for row in integrated)

    def reject_summary(rows, frame):
        raise ValueError("CENSUS_BINDING_REJECTED")

    rejected = tmp_path / "rejected_history"
    with pytest.raises(ValueError, match="CENSUS_BINDING_REJECTED"):
        history.run(settings, source, rejected, frame_sampler=sampler, frame_summary=reject_summary)
    failed = json.loads((rejected / "report.json").read_text())
    assert failed["status"] == "FAILED_PFC_FIRST_PASSAGE_HISTORY"
    assert failed["error"]["message"] == "CENSUS_BINDING_REJECTED"

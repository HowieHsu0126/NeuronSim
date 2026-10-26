"""Artifact-reader fixtures; no source simulation or variance enumeration."""
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json

import numpy as np
import pytest
import yaml

from .pfc_history_timing_census import census, pair
from .pfc_stratified_history import sampling_callbacks, paired_components
from . import pfc_stratified_history_summary as reader


SEEDS = list(range(95001, 95033))
PROFILES = ((64, 100000, 3), (640, 10000, 4))
SCOPE = dict(scientific_fidelity_qualified=False, confirmation_execution_released=False,
             production_episodes=0, new_independent_source_blocks=0,
             population_confidence_interval_computed=False, population_history_uncertainty_established=False,
             third_spike_probability_established=False)


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def archive(root, status, files):
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append(dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    delivery = encode(dict(status=status, verified=True, files=entries))
    (root / "delivery.json").write_bytes(delivery)
    return {row["path"]: row for row in entries}, hashlib.sha256(delivery).hexdigest()


def refresh(root):
    delivery = json.loads((root / "delivery.json").read_text())
    for row in delivery["files"]:
        payload = (root / row["path"]).read_bytes()
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (root / "delivery.json").write_bytes(encode(delivery))


def source_registry():
    return {str(seed): {clock: dict(archive=f"source/{seed}/{clock}", case_path="case",
                                    delivery_sha256=("c" if clock == "coarse" else "f") * 64)
                       for clock in ("coarse", "fine")} for seed in SEEDS}


def canonical(census_ref=None, uniform_ref=None, profile=PROFILES[0]):
    sample_size, inner_samples, namespace = profile
    return dict(schema="pfc_stratified_history_v1", required_source_seeds=SEEDS,
                expected_units=100000, field_index=5, phase="cue", window_ms=5., sample_size=sample_size,
                integration_seed_namespace=namespace,
                importance_sampling=dict(samples=inner_samples, batch_size=10000,
                                         proposal_multipliers=[1., 2., 4., 8.], mixture_probabilities=[.25] * 4),
                clocks={"coarse": dict(dt_ms=.05, input_stream="native_numpy_stream"),
                        "fine": dict(dt_ms=.025, input_stream="coupled_binomial_clock_refinement_stream")},
                accepted_census=census_ref or dict(archive="accepted_census", delivery_sha256="a" * 64),
                accepted_uniform_comparison=uniform_ref or dict(archive="uniform", delivery_sha256="b" * 64))


def fixture_entries(producer):
    files = {path: ("fixture identity for " + path).encode() for path in reader.TESTED_FILES}
    files[reader.PRODUCER_CONFIG] = yaml.safe_dump(producer).encode()
    return files


def block_fixture(root, profile=PROFILES[0]):
    sample_size, inner_samples, namespace = profile
    seed = 95001
    registry = source_registry()
    producer = canonical(profile=profile)
    block = deepcopy(producer)
    block.update(source_rng_seed=seed, case_id=f"pfc_{seed}_strong_cue", sampling_seed_components=[20261009, seed, 5, namespace])
    for clock in ("coarse", "fine"):
        block["clocks"][clock]["source"] = registry[str(seed)][clock]
    cases = {}
    for clock, dt, units in (("coarse", .05, [25000, 25001]), ("fine", .025, [25000, 25002])):
        config = dict(excitatory_neurons=80000, inhibitory_neurons=20000, excitatory_preference_bins=16,
                      duration_ms=5000., dt_ms=dt)
        ticks = np.full(2, round(100 / dt), dtype=np.int64)
        ids = np.asarray(units, dtype=np.int64)
        raw = dict(spike_ticks=ticks, spike_unit_indices=ids, native_spike_counts=np.bincount(ids, minlength=100000))
        cases[clock] = census(config, raw, field_index=5, phase="cue", window_ms=5., time_edges_ms=[0., 1., 2., 3.])
    census_report, arrays = pair(*cases["coarse"], *cases["fine"])
    census_report.update(status="COMPLETE_PFC_HISTORY_TIMING_CENSUS", source_rng_seed=seed, case_id=block["case_id"])
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    files = fixture_entries(producer)
    files.update({"stratified_block_config.yaml": yaml.safe_dump(block).encode(),
                  "census/report.json": encode(census_report), "census/frame_assignments.npz": buffer.getvalue()})
    clock_rows, clock_reports = {}, {}
    inner_paths = []
    for clock in ("coarse", "fine"):
        dt = block["clocks"][clock]["dt_ms"]
        meta = census_report["clocks"][clock]
        config = {key: meta[key] for key in ("excitatory_neurons", "inhibitory_neurons", "excitatory_preference_bins", "duration_ms", "dt_ms")}
        sampler, summary = sampling_callbacks(census_report, arrays, clock)
        rows, frame = sampler(config, field=5, phase="cue", window_ms=5., sample_size=sample_size,
                               seed_components=[20261009, seed, 5, namespace])
        files[f"validation/{clock}/selected_frame.json"] = encode(dict(rows=rows, frame=frame))
        for row in rows:
            item = row["frame_index"]
            first = int(arrays[clock + "_first_spike_tick"][item])
            count = int(arrays[clock + "_observed_spike_count"][item])
            row.update(observed_count=count, observed_window_ticks=[] if first < 0 else [first],
                       first_spike_tick=None if first < 0 else first,
                       refractory_ticks=int(arrays[clock + "_refractory_ticks"][item]),
                       category="no_first_spike" if first < 0 else "conditional_integration_required",
                       conditional_estimate=0., inner_monte_carlo_variance=0., proposal_hits=0)
            if count:
                value = (1.2 if row["unit_index"] == 25000 else 0.) if clock == "coarse" else (.4 if row["unit_index"] == 25000 else .2)
                score_variance = .04 if value else 0.
                hits = int(value > 0)
                probability = 1.4 * dt * 1e-3
                inner = dict(SCOPE, status="COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE", samples=inner_samples,
                             reference_archive=registry[str(seed)][clock]["archive"],
                             reference_delivery_sha256=registry[str(seed)][clock]["delivery_sha256"],
                             seed_components=[20261009, seed, 5, item, round(dt * 1000), namespace],
                             batch_size=10000, mixture_probabilities=[.25] * 4, nominal_probability=probability,
                             proposal_probabilities=[probability * multiplier for multiplier in (1., 2., 4., 8.)],
                             stopped_prefix=dict(estimate=value, empirical_score_variance=score_variance),
                             proposal_hits=hits, wall_seconds=.25, sample_frame_row=deepcopy(row))
                relative = f"sample_{row['sample_index']:03d}/integration/report.json"
                path = f"validation/{clock}/" + relative
                files[path] = encode(inner)
                inner_paths.append(path)
                row.update(conditional_estimate=value, inner_monte_carlo_variance=score_variance / inner_samples,
                           proposal_hits=hits, integration_report=relative)
        statistics = summary(rows, frame)
        actual = dict(SCOPE, status="COMPLETE_PFC_FIRST_PASSAGE_HISTORY_PILOT", source_rng_seed=seed, frame=frame,
                      categories=dict(Counter(row["category"] for row in rows)), integrated_histories=2,
                      reconstruction=dict(ticks_reconstructed=6000 if clock == "coarse" else 12000, source_events_consumed=2),
                      reconstruction_seconds=1. if clock == "coarse" else 1.5,
                      wall_seconds=2. if clock == "coarse" else 3., input_stream=block["clocks"][clock]["input_stream"])
        actual.update(statistics)
        if clock == "fine":
            actual.update(paired_physical_frame_equal=True, paired_coarse="../coarse/selected_frame.json")
        clock_rows[clock], clock_reports[clock] = rows, actual
        files[f"validation/{clock}/integrated_frame.json"] = encode(rows)
        files[f"validation/{clock}/report.json"] = encode(actual)
    statistics, _ = paired_components(clock_rows["coarse"], clock_rows["fine"], frame)
    result = dict(SCOPE, status="COMPLETE_PFC_STRATIFIED_HISTORY_PAIR", source_rng_seed=seed, case_id=block["case_id"],
                  statistics=statistics, frame=frame, selected_physical_windows=3, wall_seconds=5.1,
                  per_clock={clock: dict(sample_mean=actual["sample_mean"], integrated_histories=2, inner_trials=2 * inner_samples,
                                         zero_hit_eligible_histories=actual["zero_hit_eligible_histories"],
                                         reconstruction=actual["reconstruction"], reconstruction_seconds=actual["reconstruction_seconds"],
                                         wall_seconds=actual["wall_seconds"])
                             for clock, actual in clock_reports.items()})
    files["validation/report.json"] = encode(result)
    files["report.json"] = encode(dict(SCOPE, status="COMPLETE_PFC_STRATIFIED_HISTORY_COMPLETE", exit_code=0, wall_seconds=6.))
    entries, _ = archive(root, "COMPLETE_PFC_STRATIFIED_HISTORY_COMPLETE", files)
    return dict(pilot_delivery_sha256={}, source_registry=registry), {path: entries[path] for path in reader.TESTED_FILES}, inner_paths


def upstream_project(project, profile=PROFILES[0]):
    uniform_statistics = dict(original_blocks=32, mean=-.125, observed_between_block_variance=0.,
                              mean_within_archive_variance_diagnostic=4., latent_archived_block_mean_variance_diagnostic=-4.,
                              empirical_mean_variance_diagnostic=0., inner_mc_mean_variance_diagnostic=0.,
                              unit_window_sampling_mean_variance_diagnostic=4 / 32)
    uniform = dict(SCOPE, status="COMPLETE_PFC_PAIRED_HISTORY_DIAGNOSTICS", available_original_block_pairs=32,
                   required_original_block_pairs=32, paired_unit_windows=2048, missing_receipts=[],
                   field_index=5, phase="cue", window_ms=5., coarse_dt_ms=.05, fine_dt_ms=.025, statistics=uniform_statistics)
    _, digest = archive(project / "uniform", "COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE", {
        "report.json": encode(dict(status="COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE", exit_code=0)),
        "analysis/report.json": encode(uniform)})
    uniform_ref = dict(archive="uniform", delivery_sha256=digest)
    _, digest = archive(project / "census", "COMPLETE_PFC_HISTORY_TIMING_CENSUS_COMPLETE", {
        "analysis/report.json": encode(dict(status="COMPLETE_FULL_FRAME_CENSUS", completed_original_blocks=32,
                                            required_original_blocks=32, completed_physical_windows=6400000,
                                            required_physical_windows=6400000, failures=[])),
        "source_index.json": encode(dict(source_blocks=source_registry()))})
    census_ref = dict(archive="census", delivery_sha256=digest)
    files = fixture_entries(canonical(census_ref, uniform_ref, profile))
    files["report.json"] = encode(dict(status="COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE", exit_code=0))
    _, digest = archive(project / "preparation", "COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE", files)
    return dict(schema="pfc_stratified_history_summary_v1", required_source_seeds=SEEDS,
                pilot_archives={"95001": "blocks/pilot"}, pilot_delivery_sha256={}, array_archive_prefix="blocks/array_",
                preparation=dict(archive="preparation", delivery_sha256=digest), accepted_uniform_comparison=uniform_ref)


def fake_block(seed, profile=PROFILES[0]):
    clocks = {}
    for clock, mean in (("coarse", .25), ("fine", .125)):
        clocks[clock] = dict(sample_mean=mean, within_archive_mean_variance_diagnostic=1.,
                             inner_mc_mean_variance_diagnostic=.25, unit_window_sampling_mean_variance_diagnostic=.75,
                             integrated_histories=2, inner_trials=2 * profile[1], eligible_zero_estimates=1, zero_hit_histories=1,
                             nonzero_conditional_estimates=1, inner_integration_seconds=.5, reconstruction_seconds=1.,
                             analysis_seconds=2. if clock == "coarse" else 3., reconstruction_ticks=100, source_events_consumed=2)
    return dict(source_rng_seed=seed, archive="fixture", delivery_sha256="d" * 64, clocks=clocks,
                paired=dict(sample_mean=-.125, within_archive_mean_variance_diagnostic=2.,
                            inner_mc_mean_variance_diagnostic=.5, unit_window_sampling_mean_variance_diagnostic=1.5,
                            fine_minus_coarse_cdf_le1=.125), selected_physical_windows=profile[0],
                both_zero_population=160000, both_zero_weight=.8, pair_analysis_seconds=5., wrapper_seconds=6.)


def mark_present(project, settings, seeds):
    for seed in seeds:
        relative = settings["pilot_archives"].get(str(seed), settings["array_archive_prefix"] + str(seed - 95001))
        root = project / relative
        root.mkdir(parents=True)
        (root / "delivery.json").write_text('{"fixture": true}')


def test_read_block_recomputes_census_weights_scores_and_actual_costs(tmp_path):
    for profile in PROFILES:
        settings, entries, _ = block_fixture(tmp_path / str(profile[0]), profile)
        block = reader.read_block(tmp_path / str(profile[0]), 95001, settings, entries)
        assert block["selected_physical_windows"] == 3
        assert block["both_zero_population"] == 199997 and block["both_zero_weight"] == 199997 / 200000
        assert block["clocks"]["coarse"]["sample_mean"] == pytest.approx(6e-6)
        assert block["paired"]["sample_mean"] == pytest.approx(-3e-6)
        assert block["paired"]["fine_minus_coarse_cdf_le1"] == pytest.approx(3e-6)
        assert block["clocks"]["coarse"]["zero_hit_histories"] == 1
        assert block["clocks"]["fine"]["zero_hit_histories"] == 0
        assert block["clocks"]["coarse"]["inner_trials"] == 2 * profile[1]
        assert block["clocks"]["coarse"]["inner_integration_seconds"] == .5
        assert block["pair_analysis_seconds"] == 5.1 and block["wrapper_seconds"] == 6.
        singleton = next(row for row in block["paired"]["stratum_diagnostics"] if row["sample_size"] == 1)
        assert singleton["sampled_estimate_variance"] is None


def test_resealed_code_source_inner_seed_and_score_tampering_rejected(tmp_path):
    for profile in PROFILES:
        root = tmp_path / str(profile[0])
        settings, entries, inner_paths = block_fixture(root, profile)
        path = root / inner_paths[0]
        original = path.read_bytes()
        for key, value, error in (("reference_archive", "different/source", "INNER_SOURCE_SEED_OR_BUDGET"),
                                  ("seed_components", [20261009, 95001, 5, 1, 50, 2], "INNER_SOURCE_SEED_OR_BUDGET")):
            inner = json.loads(original)
            inner[key] = value
            path.write_bytes(encode(inner)); refresh(root)
            with pytest.raises(ValueError, match=error):
                reader.read_block(root, 95001, settings, entries)
        inner = json.loads(original)
        inner["stopped_prefix"]["estimate"] += .1
        path.write_bytes(encode(inner)); refresh(root)
        with pytest.raises(ValueError, match="INNER_SCORE_ARITHMETIC"):
            reader.read_block(root, 95001, settings, entries)
        inner = json.loads(original)
        inner["sample_frame_row"]["first_spike_tick"] += 1
        path.write_bytes(encode(inner)); refresh(root)
        with pytest.raises(ValueError, match="INNER_PREINTEGRATION_IDENTITY"):
            reader.read_block(root, 95001, settings, entries)
        path.write_bytes(original); refresh(root)
        config_path = root / "stratified_block_config.yaml"
        original_config = config_path.read_bytes()
        config = yaml.safe_load(original_config)
        config["clocks"]["coarse"]["source"]["archive"] = "different/source"
        config_path.write_text(yaml.safe_dump(config)); refresh(root)
        with pytest.raises(ValueError, match="FROZEN_CENSUS_SOURCE_CHANGED"):
            reader.read_block(root, 95001, settings, entries)
        config_path.write_bytes(original_config); refresh(root)
        code_path = root / reader.TESTED_FILES[0]
        code_path.write_bytes(code_path.read_bytes() + b"changed"); refresh(root)
        with pytest.raises(ValueError, match="TESTED_IMPLEMENTATION_OR_CONFIG_CHANGED"):
            reader.read_block(root, 95001, settings, entries)


def test_missing_stratum_and_changed_paired_physical_ids_fail(tmp_path):
    for profile in PROFILES:
        root = tmp_path / str(profile[0])
        settings, entries, _ = block_fixture(root, profile)
        path = root / "validation/fine/integrated_frame.json"
        original = path.read_bytes()
        rows = json.loads(original)
        path.write_bytes(encode(rows[:-1])); refresh(root)
        with pytest.raises(ValueError, match="SELECTED_FRAME_CHANGED"):
            reader.read_block(root, 95001, settings, entries)
        rows = json.loads(original)
        rows[0]["unit_index"] += 1
        path.write_bytes(encode(rows)); refresh(root)
        with pytest.raises(ValueError, match="CLASSIFICATION_DIFFERS_FROM_CENSUS"):
            reader.read_block(root, 95001, settings, entries)
        path.write_bytes(original); refresh(root)
        selected_path = root / "validation/fine/selected_frame.json"
        selected = json.loads(selected_path.read_text())
        selected["rows"][0]["frame_index"] += 1
        selected_path.write_bytes(encode(selected)); refresh(root)
        with pytest.raises(ValueError, match="FRESH_RANDOM_SELECTION_CHANGED"):
            reader.read_block(root, 95001, settings, entries)


def test_partial_and_strict_missing_receipts_never_publish_all32_statistics(tmp_path, monkeypatch):
    settings = upstream_project(tmp_path)
    mark_present(tmp_path, settings, [95001])
    monkeypatch.setattr(reader, "read_block", lambda root, seed, settings, preparation_entries: fake_block(seed))
    report = reader.run(tmp_path, settings, tmp_path / "partial", allow_partial=True)
    assert report["status"] == "PARTIAL_PFC_STRATIFIED_HISTORY_SUMMARY"
    assert report["available_original_blocks"] == 1 and report["missing_receipt_source_seeds"] == SEEDS[1:]
    assert report["statistics"] is report["uniform_comparison"] is None
    assert report["missing_receipt_is_terminal_job_evidence"] is False
    assert report["totals"]["coarse"]["inner_trials"] == 200000
    with pytest.raises(ValueError, match="INCOMPLETE32_BLOCK_RECEIPTS"):
        reader.run(tmp_path, settings, tmp_path / "strict")
    assert json.loads((tmp_path / "strict/report.json").read_text())["statistics"] is None
    assert len((tmp_path / "strict/block_diagnostics.csv").read_text().splitlines()) == 2


def test_all32_descriptive_comparison_retains_negative_between_correction(tmp_path, monkeypatch):
    for profile in PROFILES:
        project = tmp_path / str(profile[0])
        settings = upstream_project(project, profile)
        mark_present(project, settings, SEEDS)
        seen = []

        def read(root, seed, settings, preparation_entries):
            seen.append(seed)
            return fake_block(seed, profile)

        monkeypatch.setattr(reader, "read_block", read)
        report = reader.run(project, settings, project / "complete")
        assert seen == SEEDS and report["available_original_blocks"] == 32
        assert report["statistics"]["paired"]["latent_archived_block_mean_variance_diagnostic"] == -2.
        assert report["statistics"]["paired"]["fine_minus_coarse_cdf_le1_mean"] == .125
        assert report["uniform_comparison"]["stratified_over_uniform_variance_ratios"]["mean_within_archive_variance_diagnostic"] == .5
        assert report["uniform_comparison"]["matched_runtime_comparison"] is False
        assert report["totals"]["fine"]["inner_trials"] == 64 * profile[1]
        assert report["selected_physical_windows"] == 32 * profile[0] and report["wrapper_seconds"] == 192.
        assert report["population_confidence_interval_computed"] is False
        assert report["future_stratified_precision_established"] is False

        assert report["sampling_profile"]["sample_size"] == profile[0]
        assert report["sampling_profile"]["inner_samples"] == profile[1]
        assert report["sampling_profile"]["integration_seed_namespace"] == profile[2]
        comparison = report["uniform_comparison"]
        assert comparison["uniform_nominal_window_budget_per_block"] == 64
        assert comparison["stratified_nominal_window_budget_per_block"] == profile[0]
        assert comparison["uniform_nominal_inner_trial_budget_per_clock"] == comparison["stratified_nominal_inner_trial_budget_per_clock"] == 6400000
        assert comparison["nominal_window_budget_per_block"] == (64 if profile[0] == 64 else None)


def test_malformed_delivery_preserves_validated_prefix_and_failure_report(tmp_path, monkeypatch):
    settings = upstream_project(tmp_path)
    mark_present(tmp_path, settings, SEEDS[:2])

    def read(root, seed, settings, preparation_entries):
        if seed == 95002:
            raise ValueError("malformed delivered fixture")
        return fake_block(seed)

    monkeypatch.setattr(reader, "read_block", read)
    with pytest.raises(ValueError, match="malformed delivered fixture"):
        reader.run(tmp_path, settings, tmp_path / "failed", allow_partial=True)
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "FAILED_PFC_STRATIFIED_HISTORY_SUMMARY"
    assert report["failed_source_seed"] == 95002 and report["validated_source_seeds"] == [95001]
    assert report["unexamined_source_seeds"] == SEEDS[2:]
    assert report["statistics"] is report["uniform_comparison"] is None
    assert len((tmp_path / "failed/block_diagnostics.csv").read_text().splitlines()) == 2

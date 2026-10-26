"""Synthetic scalar records test coverage and grouping, not scientific findings."""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import pytest
import yaml

from . import pfc_profile_history_coverage as coverage
from .pfc_first_passage_history_summary import between_components
from .pfc_profile_history import profile_key
from .test_pfc_profile_history import full_settings_fixture


def scalar_statistics(mean):
    return dict(sample_mean=mean, within_archive_mean_variance_diagnostic=2.,
                inner_mc_mean_variance_diagnostic=2.25,
                unit_window_sampling_mean_variance_diagnostic=-.25)


def record_fixture(settings, seeds):
    records = []
    for seed in seeds:
        for repeat in settings["auxiliary_repeats"]:
            for profile in settings["profiles"]:
                key = profile_key(profile, repeat)
                coarse_mean = .125 + (seed-95001)/1024
                difference = .125 if repeat == "A" else .25
                paired = scalar_statistics(difference)
                paired["fine_minus_coarse_cdf_le1"] = -difference
                records.append(dict(source_rng_seed=seed, key=key,
                                    provenance=dict(archive="synthetic_fixture", member=f"{seed}/{key}"),
                                    coarse=scalar_statistics(coarse_mean),
                                    fine=scalar_statistics(coarse_mean+difference), paired=paired))
    return records


@pytest.mark.parametrize("requested", [[95001], [95002, 95001]])
def test_requested_coverage_below32_preserves_records_without_between_block_statistics(requested, monkeypatch):
    settings = full_settings_fixture()
    records = list(reversed(record_fixture(settings, requested)))
    before = deepcopy(records)
    between = Mock(side_effect=AssertionError("Partial scope must not aggregate blocks"))
    monkeypatch.setattr(coverage, "between_components", between)
    report = coverage.aggregate(settings, requested, records)
    assert report["status"] == "COMPLETE_REQUESTED_PFC_PROFILE_HISTORY_COVERAGE"
    assert report["required_original_blocks"] == 32
    assert report["requested_original_blocks"] == report["available_original_blocks"] == len(requested)
    assert report["requested_source_seeds"] == requested
    assert report["expected_pairs"] == report["completed_pairs"] == 72*len(requested)
    assert report["required_registered_pairs"] == 2304
    assert report["full_registered_scope_covered"] is False and report["statistics"] is None
    assert report["records"] == records == before
    between.assert_not_called()


def test_full32_coverage_aggregates_each_key_over32_blocks_and_preserves_negative_diagnostics(monkeypatch):
    settings = full_settings_fixture()
    seeds = settings["required_source_seeds"]
    records = list(reversed(record_fixture(settings, seeds)))
    before = deepcopy(records)
    between = Mock(wraps=between_components)
    monkeypatch.setattr(coverage, "between_components", between)
    report = coverage.aggregate(settings, list(reversed(seeds)), records)
    assert report["full_registered_scope_covered"] is True
    assert report["required_original_blocks"] == report["requested_original_blocks"] == report["available_original_blocks"] == 32
    assert report["expected_pairs"] == report["completed_pairs"] == report["required_registered_pairs"] == 2304
    assert len(report["statistics"]) == 72 and between.call_count == 72*3
    assert all(len(call.args[0]) == 32 for call in between.call_args_list)
    indexed = {(record["source_rng_seed"], record["key"]): record for record in records}
    for key in (report["profile_keys"][0], report["profile_keys"][36]):
        for component in ("coarse", "fine", "paired"):
            expected = between_components([indexed[(seed, key)][component] for seed in seeds])
            assert report["statistics"][key][component] == expected
    a, b = report["profile_keys"][0], report["profile_keys"][36]
    assert report["statistics"][a]["paired"]["mean"] == .125
    assert report["statistics"][b]["paired"]["mean"] == .25
    for summary in report["statistics"].values():
        for component in summary.values():
            assert component["original_blocks"] == 32
            assert component["latent_archived_block_mean_variance_diagnostic"] < 0
            assert component["unit_window_sampling_mean_variance_diagnostic"] < 0
            assert component["negative_variance_diagnostics_preserved"] is True
            assert component["population_confidence_interval_computed"] is False
    assert report["records"] == records == before
    for key in ("population_confidence_interval_computed", "source_history_population_uncertainty_established",
                "third_spike_probability_established", "third_spike_probability_or_bound_qualified",
                "scientific_fidelity_qualified", "confirmation_execution_released"):
        assert report[key] is False
    assert report["production_episodes"] == report["new_independent_source_blocks"] == report["new_full_network_episodes"] == 0


@pytest.mark.parametrize("change", ["duplicate", "missing", "unknown_key", "extra_seed", "illegal_seed", "seed_type",
                                    "missing_field", "missing_cdf", "cdf_sign", "missing_provenance", "bool_statistic",
                                    "string_statistic", "missing_component", "extra_nonfinite"])
def test_coverage_rejects_invalid_record_identity_schema_or_cdf_sign(change):
    settings = full_settings_fixture()
    records = record_fixture(settings, [95001])
    row = records[0]
    if change == "duplicate": records.append(deepcopy(row))
    elif change == "missing": records.pop()
    elif change == "unknown_key": row["key"] = "A/native_half/unknown_frame"
    elif change == "extra_seed": row["source_rng_seed"] = 95002
    elif change == "illegal_seed": row["source_rng_seed"] = 95033
    elif change == "seed_type": row["source_rng_seed"] = "95001"
    elif change == "missing_field": row["coarse"].pop("inner_mc_mean_variance_diagnostic")
    elif change == "missing_cdf": row["paired"].pop("fine_minus_coarse_cdf_le1")
    elif change == "cdf_sign": row["paired"]["fine_minus_coarse_cdf_le1"] = row["paired"]["sample_mean"]
    elif change == "missing_provenance": row.pop("provenance")
    elif change == "bool_statistic": row["coarse"]["sample_mean"] = True
    elif change == "string_statistic": row["coarse"]["sample_mean"] = "0.125"
    elif change == "missing_component": row.pop("paired")
    else: row["paired"]["additional_diagnostic"] = float("nan")
    with pytest.raises(ValueError):
        coverage.aggregate(settings, [95001], records)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_coverage_rejects_nonfinite_required_statistics(value):
    settings = full_settings_fixture()
    records = record_fixture(settings, [95001])
    records[0]["fine"]["within_archive_mean_variance_diagnostic"] = value
    with pytest.raises(ValueError, match="FINITE_NUMERIC_STATISTIC"):
        coverage.aggregate(settings, [95001], records)


@pytest.mark.parametrize("requested", [[], [95001, 95001], [95033], [True], ["95001"]])
def test_coverage_rejects_invalid_requested_source_domain(requested):
    settings = full_settings_fixture()
    with pytest.raises(ValueError, match="REQUESTED_SOURCE_DOMAIN"):
        coverage.aggregate(settings, requested, record_fixture(settings, [95001]))


@pytest.mark.parametrize("change", ["seeds", "profiles", "repeats", "duplicate_profile"])
def test_coverage_rejects_changed_registered_source_profile_or_repeat_domain(change):
    settings = full_settings_fixture()
    records = record_fixture(settings, [95001])
    if change == "seeds": settings["required_source_seeds"] = settings["required_source_seeds"][:-1]
    elif change == "profiles": settings["profiles"] = settings["profiles"][:-1]
    elif change == "repeats": settings["auxiliary_repeats"] = ["A"]
    else: settings["profiles"][-1] = deepcopy(settings["profiles"][0])
    with pytest.raises(ValueError):
        coverage.aggregate(settings, [95001], records)


# These tiny receipts exercise the production copy/binding boundary. They do not
# contain source event tapes or importance samples and never replay a simulator.
def archive_fixture(repo, archive, status, payloads):
    root = repo / archive
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        data = path.read_bytes()
        entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    receipt = dict(verified=True, status=status, job_id=Path(archive).name.removeprefix("run_"), files=entries)
    data = json.dumps(receipt).encode()
    (root / "delivery.json").write_bytes(data)
    return dict(archive=archive, delivery_sha256=hashlib.sha256(data).hexdigest()), receipt


def readback_wrapper(job_id):
    return dict(status=coverage._READBACK_STATUS, exit_code=0, error=None, task="profile_readback",
                mode="complete", job_id=job_id, scientific_fidelity_qualified=False,
                confirmation_execution_released=False, production_episodes=0, new_independent_source_blocks=0)


def readback_pair_fixture(record, labels, costs):
    components = {}
    for component in ("coarse", "fine", "paired"):
        values = dict(record[component], sample_denominator=640, nominal_sample_budget=640,
                      population_confidence_interval_computed=False, importance_estimates_clipped=False,
                      negative_variance_diagnostics_preserved=True, uncertainty_scope="fixture",
                      stratum_diagnostics=[dict(omitted_from_normalized_statistics=True)])
        if component == "paired":
            values.update(third_spike_probability_established=False, scientific_fidelity_qualified=False,
                          confirmation_execution_released=False, production_episodes=0)
        else:
            values.update(source_label=labels[component], source_binding=deepcopy(costs[component]["source_binding"]),
                          integrated_histories=2, inner_trials=20000)
        components[component] = values
    return dict(key=record["key"], statistics=components["paired"],
                per_clock={clock: components[clock] for clock in ("coarse", "fine")},
                paired_selected_physical_windows=640)


def readback_analysis(seed, pairs, costs):
    return dict(status="COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK", source_rng_seed=seed,
                original_source_blocks=1, paired_profiles=len(pairs), pairs=pairs, source_costs=costs,
                third_spike_probability_or_bound_qualified=False,
                source_history_population_uncertainty_established=False, **coverage.BOUNDARY)


def driver_fixture(tmp_path, *, seed_groups=((95001,),), include_groups=True, legacy_change=None,
                   group_change=None, pilot_config_change=False):
    repo, work = tmp_path / "repo", tmp_path / "work"
    repo.mkdir()
    work.mkdir()
    history = full_settings_fixture()
    rows = []
    for seed in history["required_source_seeds"]:
        row = dict(source_rng_seed=seed, archive=f"legacy_block/{seed}", delivery_sha256="a" * 64,
                   selected_physical_windows=640)
        for component in ("coarse", "fine", "paired"):
            row.update({f"{component}_{key}": value for key, value in scalar_statistics(.125).items()})
        row.update(fine_minus_coarse_cdf_le1=-.125, coarse_integrated_histories=2, coarse_inner_trials=20000,
                   fine_integrated_histories=2, fine_inner_trials=20000)
        rows.append(row)
    if legacy_change == "duplicate": rows[-1]["source_rng_seed"] = 95001
    if legacy_change == "windows": rows[0]["selected_physical_windows"] = 639
    if legacy_change == "budget": rows[0]["coarse_inner_trials"] = 19999
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    legacy = dict(status="COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY", required_original_blocks=32,
                  available_original_blocks=32, validated_source_seeds=history["required_source_seeds"],
                  missing_receipt_source_seeds=[], unexamined_source_seeds=[], selected_physical_windows=20480,
                  block_diagnostics_file="block_diagnostics.csv",
                  sampling_profile=dict(sample_size=640, inner_samples=10000,
                                        integration_seed_namespace=3 if legacy_change == "namespace" else 4),
                  significance_test_computed=False, clock_equivalence_qualified=False,
                  third_spike_probability_established=False, cdf_at_count2_contrast_estimated=False,
                  future_stratified_precision_established=False, **coverage.BOUNDARY)
    legacy_reference, _ = archive_fixture(repo, "legacy/run_1", "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE",
                                         {"analysis/report.json": json.dumps(legacy),
                                          "analysis/block_diagnostics.csv": table.getvalue(),
                                          coverage._LEGACY_CONFIG: "schema: fixture\n"})
    history["reused_integration_summary"] = legacy_reference
    source_bindings = {label: dict(archive="accepted_source", delivery_sha256="b" * 64, source_label=label)
                       for label in ("half_A", "quarter_A")}
    pilot_costs = {clock: dict(reconstruction_seconds=1., wall_seconds=2., peak_rss_kib=100,
                              source_reconstruction_calls=1, source_binding=source_bindings[label])
                   for clock, label in (("coarse", "half_A"), ("fine", "quarter_A"))}
    indexed = {row["key"]: row for row in record_fixture(history, [95001])}
    labels = dict(coarse="half_A", fine="quarter_A")
    pairs = [readback_pair_fixture(indexed[key], labels, pilot_costs) for key in history["pilot"]["profile_keys"]]
    pilot_history = deepcopy(history)
    if pilot_config_change: pilot_history["integration_seed_namespace"] = 4
    pilot_reference, _ = archive_fixture(repo, "pilot/run_2", coverage._READBACK_STATUS,
                                        {"report.json": json.dumps(readback_wrapper("2")),
                                         "analysis/report.json": json.dumps(readback_analysis(95001, pairs, pilot_costs)),
                                         "source_config.yaml": yaml.safe_dump(pilot_history)})
    history["accepted_pilot"] = dict(readback=pilot_reference,
                                    source_runs={label: dict(archive="pilot_source/" + label, delivery_sha256="c" * 64)
                                                 for label in labels.values()})
    source_config = yaml.safe_dump(history)
    source_sha = hashlib.sha256(source_config.encode()).hexdigest()
    arrays, foundations, group_paths = [], [], []
    for array_index, seeds in enumerate(seed_groups):
        config = dict(schema="pfc_profile_history_readback_v1", source_config_sha256=source_sha,
                      source_rng_seeds=list(seeds), source_preparation=dict(archive="source_prepare", delivery_sha256="d" * 64),
                      source_arrays=[dict(archive_prefix="source/run_100", source_index_start=0, source_index_stop=159)])
        config_text = yaml.safe_dump(config)
        foundation_reference, foundation_delivery = archive_fixture(
            repo, f"foundation/run_{10 + array_index}", coverage._PREPARATION_STATUS,
            {"source_config.yaml": source_config, coverage._READBACK_CONFIG: config_text})
        foundations.append((foundation_reference, foundation_delivery))
        requests = coverage.readback_requests(config, history)
        prefix = f"readback/run_{20 + array_index}"
        arrays.append(dict(archive_prefix=prefix, task_indices=list(range(len(requests))),
                           preparation_delivery_sha256=foundation_reference["delivery_sha256"]))
        if not include_groups:
            continue
        for request in requests:
            index, seed = request["group_index"], request["source_rng_seed"]
            costs = {}
            for clock, spec in request["source_runs"].items():
                source_index = int(spec["archive"].rsplit("_", 1)[1])
                costs[clock] = dict(history_execution_archive=spec["archive"],
                                    history_execution_job_id=f"100_{source_index}", reconstruction_seconds=1. + source_index,
                                    wall_seconds=2. + source_index, peak_rss_kib=100 + source_index,
                                    source_reconstruction_calls=1,
                                    source_binding=dict(archive=f"accepted_source/{seed}", delivery_sha256="e" * 64,
                                                        source_label=spec["source_label"]))
            indexed = {row["key"]: row for row in record_fixture(history, [seed])}
            labels = {clock: spec["source_label"] for clock, spec in request["source_runs"].items()}
            pairs = [readback_pair_fixture(indexed[key], labels, costs) for key in request["profile_keys"]]
            analysis = readback_analysis(seed, pairs, costs)
            analysis.update(full_source_requests=True, readback_group=request["group_name"], readback_group_index=index)
            if array_index == 0:
                if group_change == "cost_mismatch" and index == 1: costs["coarse"]["wall_seconds"] += 1.
                if group_change == "source_archive" and index == 0: costs["coarse"]["history_execution_archive"] += "_wrong"
                if group_change == "identity" and index == 0: analysis["readback_group"] = "wrong"
                if group_change == "scope" and index == 0: analysis["scientific_fidelity_qualified"] = True
            job_id = f"{20 + array_index}_{index}"
            group_archive = prefix + "_" + str(index)
            archive_fixture(repo, group_archive, coverage._READBACK_STATUS,
                            {"report.json": json.dumps(readback_wrapper(job_id)),
                             "analysis/report.json": json.dumps(analysis), "source_config.yaml": source_config,
                             "preparation_delivery.json": json.dumps(foundation_delivery),
                             coverage._READBACK_CONFIG: config_text})
            group_paths.append(repo / group_archive)
    settings = dict(schema="pfc_profile_history_coverage_v1", source_preparation=foundations[0][0],
                    source_config_sha256=source_sha, requested_source_seeds=[seed for group in seed_groups for seed in group],
                    readback_arrays=arrays)
    copied = []

    def copy(source, relative, expected=None):
        copied.append((Path(source), Path(relative)))
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if expected is not None and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("BOUND_INPUT_CHANGED:" + str(relative))

    return dict(settings=settings, repo=repo, work=work, copy=copy, foundation_delivery=foundations[0][1],
                copied=copied, history=history, group_paths=group_paths,
                foundation_path=repo / foundations[0][0]["archive"], pilot_path=repo / pilot_reference["archive"])


def run_driver(fixture, mode):
    return coverage.run(fixture["settings"], fixture["repo"], fixture["work"], fixture["copy"], mode,
                        fixture["foundation_delivery"])


def test_driver_prepare_reuses_three_pairs_without_opening_missing_group_outputs(tmp_path, monkeypatch):
    fixture = driver_fixture(tmp_path, include_groups=False)
    between = Mock(side_effect=AssertionError("Preparation must not manufacture complete records"))
    monkeypatch.setattr(coverage, "aggregate", between)
    report = run_driver(fixture, "prepare")
    assert report["status"] == "PREPARED_PFC_PROFILE_HISTORY_COVERAGE"
    assert report["reused_legacy_pairs"] == 1 and report["reused_pilot_pairs"] == 2
    assert report["reused_pairs"] == report["completed_pairs"] == 3
    assert report["awaited_pairs"] == len(report["awaited_pair_keys"]) == 69
    assert report["statistics"] is None and report["full_registered_scope_covered"] is False
    assert all(relative.parts[0] == "coverage_inputs" for _, relative in fixture["copied"])
    assert not any("readback" in source.parts for source, _ in fixture["copied"])
    between.assert_not_called()


def test_driver_complete_normalizes_actual_records_and_deduplicates_five_source_costs(tmp_path):
    fixture = driver_fixture(tmp_path)
    report = run_driver(fixture, "complete")
    assert report["status"] == "COMPLETE_REQUESTED_PFC_PROFILE_HISTORY_COVERAGE"
    assert report["completed_pairs"] == 72 and report["awaited_pairs"] == 0 and report["statistics"] is None
    assert report["new_source_execution_count"] == len(report["new_source_costs"]) == 5
    assert len({cost["history_execution_archive"] for cost in report["new_source_costs"]}) == 5
    assert [cost["kind"] for cost in report["reused_cost_references"]] == ["legacy_summary", "accepted_pilot"]
    assert all("history_execution_archive" not in cost
               for cost in report["reused_cost_references"][1]["source_costs"].values())
    assert [row["provenance"]["integration_seed_namespace"] for row in report["records"][:3]] == [4, 5, 5]
    for row in report["records"]:
        assert set(row["coarse"]) == set(row["fine"]) == set(coverage.STAT_FIELDS)
        assert set(row["paired"]) == set(coverage.PAIRED_STAT_FIELDS)
        assert row["paired"]["unit_window_sampling_mean_variance_diagnostic"] < 0
    metadata = report["records"][-1]["provenance"]["component_metadata"]
    assert metadata["coarse"]["source_binding"] and metadata["coarse"]["importance_estimates_clipped"] is False
    assert "stratum_diagnostics" not in metadata["coarse"]
    with (fixture["work"] / "analysis" / report["record_file"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72 and float(rows[-1]["paired_fine_minus_coarse_cdf_le1"]) == -float(rows[-1]["paired_sample_mean"])
    assert all("source_inputs" not in relative.parts and relative.parts[0] == "coverage_inputs"
               for _, relative in fixture["copied"])


@pytest.mark.parametrize("seed_groups", [((95001,), (95002,)), (tuple(range(95001, 95033)),)])
def test_driver_multiple_readback_arrays_and_full32_scope(tmp_path, seed_groups):
    fixture = driver_fixture(tmp_path, seed_groups=seed_groups)
    report = run_driver(fixture, "complete")
    count = sum(map(len, seed_groups))
    assert report["completed_pairs"] == count * 72
    assert report["new_source_execution_count"] == count * 5
    assert report["full_registered_scope_covered"] is (count == 32)
    assert (report["statistics"] is None) is (count != 32)


@pytest.mark.parametrize("target", ["foundation", "pilot", "group"])
def test_driver_rejects_payload_changed_after_manifest_binding(tmp_path, target):
    fixture = driver_fixture(tmp_path)
    path = {"foundation": fixture["foundation_path"] / "source_config.yaml",
            "pilot": fixture["pilot_path"] / "analysis/report.json",
            "group": fixture["group_paths"][0] / "preparation_delivery.json"}[target]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
        run_driver(fixture, "complete")
    report = json.loads((fixture["work"] / "analysis/report.json").read_text())
    assert report["status"] == "FAILED_PFC_PROFILE_HISTORY_COVERAGE" and report["error"]


@pytest.mark.parametrize("change, message", [("duplicate", "LEGACY_UNIQUE32_SOURCE_BLOCKS"),
                                             ("windows", "LEGACY_PHYSICAL_WINDOWS"),
                                             ("budget", "LEGACY_INNER_TRIAL_BUDGET"),
                                             ("namespace", "LEGACY_SAMPLING_DESIGN")])
def test_driver_rejects_legacy_source_or_sampling_changes(tmp_path, change, message):
    fixture = driver_fixture(tmp_path, legacy_change=change)
    with pytest.raises(ValueError, match=message):
        run_driver(fixture, "prepare")


@pytest.mark.parametrize("change, message", [("identity", "READBACK_GROUP_IDENTITY"),
                                             ("source_archive", "SOURCE_COST_EXECUTION_BINDING"),
                                             ("cost_mismatch", "SOURCE_COST_MISMATCH"),
                                             ("scope", "UNRELEASED_SCOPE")])
def test_driver_rejects_group_identity_scope_or_inconsistent_repeated_source_cost(tmp_path, change, message):
    fixture = driver_fixture(tmp_path, group_change=change)
    with pytest.raises(ValueError, match=message):
        run_driver(fixture, "complete")


def test_driver_rejects_pilot_config_with_changed_namespace(tmp_path):
    fixture = driver_fixture(tmp_path, pilot_config_change=True)
    with pytest.raises(ValueError, match="PILOT_SOURCE_CONFIG"):
        run_driver(fixture, "prepare")


@pytest.mark.parametrize("change", [None, "step", "delivery", "group"])
def test_driver_receipt_bound_step_preserves_logical_group_and_rejects_wrong_identity(tmp_path, change):
    fixture = driver_fixture(tmp_path)
    origin = fixture["group_paths"][0]
    delivery = json.loads((origin / "delivery.json").read_text())
    payloads = {entry["path"]: (origin / entry["path"]).read_text() for entry in delivery["files"]}
    wrapper = json.loads(payloads["report.json"])
    wrapper.update(job_id="901", slurm_step_id="7")
    payloads["report.json"] = json.dumps(wrapper)
    if change == "group":
        analysis = json.loads(payloads["analysis/report.json"])
        analysis["readback_group_index"] = 1
        payloads["analysis/report.json"] = json.dumps(analysis)
    reference, _ = archive_fixture(fixture["repo"], "readback/run_901", coverage._READBACK_STATUS, payloads)
    delivery_path = fixture["repo"] / reference["archive"] / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    delivery["job_id"] = "901"
    delivery_path.write_text(json.dumps(delivery))
    reference["delivery_sha256"] = hashlib.sha256(delivery_path.read_bytes()).hexdigest()
    reference.update(job_id="901", slurm_step_id="8" if change == "step" else "7")
    if change == "delivery": reference["delivery_sha256"] = "0" * 64
    fixture["settings"]["readback_arrays"][0]["task_archives"] = {0: reference}
    shutil.rmtree(origin)
    if change:
        expected = {"step": "READBACK_STEP_JOB_IDENTITY", "delivery": "BOUND_INPUT_CHANGED", "group": "READBACK_GROUP_IDENTITY"}[change]
        with pytest.raises(ValueError, match=expected): run_driver(fixture, "complete")
        assert json.loads((fixture["work"] / "analysis/report.json").read_text())["status"] == "FAILED_PFC_PROFILE_HISTORY_COVERAGE"
    else:
        report = run_driver(fixture, "complete")
        assert report["completed_pairs"] == 72 and report["required_registered_pairs"] == 2304
        assert any(row["provenance"]["archive"] == "readback/run_901" for row in report["records"])

"""Profile routing and shared physical-history identity on bounded real fixtures."""
from copy import deepcopy
import hashlib
import json

import yaml

import numpy as np
import pytest

from . import pfc_first_passage_history as history
from . import pfc_profile_history as profiles
from .test_pfc_first_passage_history import fixture as history_fixture
from .test_pfc_history_timing_census import config as census_config, raw as census_raw


def settings_fixture():
    entries = []
    for field, phase in ((4, "cue"), (5, "cue"), (5, "early_delay")):
        for left, right in (("numpy_dt_0p05", "numpy_dt_0p025"),
                            ("numpy_dt_0p025", "numpy_dt_0p0125")):
            row = len(entries)*2
            entries.append(dict(condition="strong_cue", field_index=field, phase=phase,
                                window_ms=5., left_cohort=left, right_cohort=right,
                                endpoints=[dict(row_index=row, count_threshold=1),
                                           dict(row_index=row+1, count_threshold=2)]))
    reused = profiles.profile_key(entries[2], "A")
    pilot = [profiles.profile_key(entries[index], "A") for index in (1, 5)]
    return dict(profiles=entries, auxiliary_repeats=["A", "B"],
                reused_profile_keys=[reused], pilot=dict(profile_keys=pilot),
                phase_codes=dict(cue=1, early_delay=2),
                clock_pair_codes=dict(native_half=0, half_quarter=1),
                auxiliary_repeat_codes=dict(A=0, B=1),
                sampling_seed_epoch=20261010, integration_seed_namespace=5)


def descriptor(request):
    return (request["auxiliary_repeat"], request["pair_name"], request["profile"]["field_index"],
            request["profile"]["phase"], request["clock"])


def test_source_requests_full_maps_five_sources_and_shared_native_without_completed_profile():
    settings = settings_fixture()
    original = deepcopy(settings)
    expected = {
        "native": [("A", "native_half", 4, "cue", "coarse"),
                   ("A", "native_half", 5, "early_delay", "coarse"),
                   ("B", "native_half", 4, "cue", "coarse"),
                   ("B", "native_half", 5, "cue", "coarse"),
                   ("B", "native_half", 5, "early_delay", "coarse")],
        "half_A": [("A", "native_half", 4, "cue", "fine"),
                   ("A", "half_quarter", 4, "cue", "coarse"),
                   ("A", "half_quarter", 5, "cue", "coarse"),
                   ("A", "native_half", 5, "early_delay", "fine"),
                   ("A", "half_quarter", 5, "early_delay", "coarse")],
        "quarter_A": [("A", "half_quarter", 4, "cue", "fine"),
                      ("A", "half_quarter", 5, "cue", "fine"),
                      ("A", "half_quarter", 5, "early_delay", "fine")],
        "half_B": [("B", "native_half", 4, "cue", "fine"),
                   ("B", "half_quarter", 4, "cue", "coarse"),
                   ("B", "native_half", 5, "cue", "fine"),
                   ("B", "half_quarter", 5, "cue", "coarse"),
                   ("B", "native_half", 5, "early_delay", "fine"),
                   ("B", "half_quarter", 5, "early_delay", "coarse")],
        "quarter_B": [("B", "half_quarter", 4, "cue", "fine"),
                      ("B", "half_quarter", 5, "cue", "fine"),
                      ("B", "half_quarter", 5, "early_delay", "fine")],
    }
    by_key = {}
    for label, wanted in expected.items():
        requests = profiles.source_requests(settings, label, pilot=False)
        assert [descriptor(request) for request in requests] == wanted
        assert all(request["key"] not in settings["reused_profile_keys"] for request in requests)
        for request in requests:
            by_key.setdefault(request["key"], []).append(request["clock"])
    assert len(by_key) == 11
    assert all(sorted(clocks) == ["coarse", "fine"] for clocks in by_key.values())
    assert settings == original


def test_source_requests_pilot_selects_only_half_and_quarter_a_for_both_pilot_profiles():
    settings = settings_fixture()
    for label in ("native", "half_A", "quarter_A", "half_B", "quarter_B"):
        requests = profiles.source_requests(settings, label, pilot=True)
        if label in ("half_A", "quarter_A"):
            assert [request["key"] for request in requests] == settings["pilot"]["profile_keys"]
            clock = "coarse" if label == "half_A" else "fine"
            assert [descriptor(request) for request in requests] == [
                ("A", "half_quarter", 4, "cue", clock),
                ("A", "half_quarter", 5, "early_delay", clock)]
        else:
            assert requests == []


def test_source_requests_rejects_unknown_source():
    with pytest.raises(ValueError, match="SOURCE_LABEL"):
        profiles.source_requests(settings_fixture(), "quarter_C", pilot=False)


def test_seed_identity_pairs_clocks_and_separates_phase_pair_and_auxiliary_repeat():
    settings = settings_fixture()

    def identity(label, pair, field, phase):
        request = next(request for request in profiles.source_requests(settings, label, pilot=False)
                       if (request["pair_name"], request["profile"]["field_index"], request["profile"]["phase"])
                       == (pair, field, phase))
        return profiles.seed_identity(settings, 95001, request)

    half = identity("half_A", "half_quarter", 4, "cue")
    assert half == identity("quarter_A", "half_quarter", 4, "cue")
    assert half == ([20261010, 95001, 4, 1, 1, 0, 5], [1, 1, 0])
    assert identity("half_A", "native_half", 4, "cue") == (
        [20261010, 95001, 4, 1, 0, 0, 5], [1, 0, 0])
    assert identity("half_B", "half_quarter", 4, "cue") == (
        [20261010, 95001, 4, 1, 1, 1, 5], [1, 1, 1])
    assert identity("half_A", "half_quarter", 5, "cue") == (
        [20261010, 95001, 5, 1, 1, 0, 5], [1, 1, 0])
    assert identity("half_A", "half_quarter", 5, "early_delay") == (
        [20261010, 95001, 5, 2, 1, 0, 5], [2, 1, 0])


def selected_row(config, unit, start, stop, sample, frame, stratum, probability):
    return dict(sample_index=sample, frame_index=frame, unit_index=unit,
                window_start_ms=float(start), window_stop_ms=float(stop),
                window_start_tick=round(start/config["dt_ms"]),
                window_stop_tick=round(stop/config["dt_ms"]),
                joint_time_bin=stratum, inclusion_probability=probability)


def mapping_fixture():
    config = census_config(dt=.025)
    events = [(round(ms/config["dt_ms"]), unit)
              for ms, unit in ((100., 10), (102., 10), (103., 11), (300., 10))]
    tape = history.reference.validate_tape(config, census_raw(config, events))
    frames = [
        [selected_row(config, 10, 100, 105, 0, 0, 11, .5),
         selected_row(config, 11, 100, 105, 1, 1, 0, .5)],
        [selected_row(config, 11, 100, 105, 0, 1, 0, .25),
         selected_row(config, 10, 105, 110, 1, 2, 0, .25),
         selected_row(config, 10, 100, 105, 2, 0, 14, .25)],
        [selected_row(config, 10, 300, 305, 0, 0, 15, .5),
         selected_row(config, 11, 300, 305, 1, 1, 0, .5)],
    ]
    shared, mappings = profiles.merge_selected_frames(frames)
    classified = history.classify(config, tape, shared)
    return config, frames, shared, mappings, classified


def test_shared_frames_deduplicate_physical_windows_and_restore_each_local_identity():
    _, frames, shared, mappings, classified = mapping_fixture()
    original_frames, original_classified = deepcopy(frames), deepcopy(classified)
    assert len(shared) == 5 and mappings == [[0, 1], [1, 2, 0], [3, 4]]
    assert [row["sample_index"] for row in shared] == list(range(5))
    assert [row["category"] for row in classified] == [
        "suppressed_network_reference_required", "refractory_covers_remaining_window",
        "no_first_spike", "conditional_integration_required", "no_first_spike"]
    for selected, mapping in zip(frames, mappings):
        restored = profiles.local_classified_rows(selected, mapping, classified)
        assert len(restored) == len(selected)
        for local, wanted, shared_index in zip(restored, selected, mapping):
            assert {key: local[key] for key in wanted} == wanted
            assert local["shared_history_index"] == shared_index
            for key in ("category", "observed_count", "observed_window_ticks", "first_spike_tick", "refractory_ticks"):
                assert local[key] == classified[shared_index][key]
    assert frames == original_frames and classified == original_classified


def test_merge_selected_frames_rejects_changed_local_sample_order():
    _, frames, _, _, _ = mapping_fixture()
    frames[0][1]["sample_index"] = 0
    with pytest.raises(ValueError, match="LOCAL_SAMPLE_ORDER"):
        profiles.merge_selected_frames(frames)


@pytest.mark.parametrize("change", ["size", "physical_identity"])
def test_local_classification_rejects_malformed_mapping_or_physical_identity(change):
    _, frames, _, mappings, classified = mapping_fixture()
    selected, mapping = deepcopy(frames[0]), list(mappings[0])
    if change == "size":
        mapping.pop()
        message = "LOCAL_MAPPING_SIZE"
    else:
        selected[0]["window_stop_ms"] += .025
        message = "SHARED_PHYSICAL_WINDOW_CHANGED"
    with pytest.raises(ValueError, match=message):
        profiles.local_classified_rows(selected, mapping, classified)


@pytest.mark.parametrize("bad_index", [-1, "out_of_range", True])
def test_local_classification_rejects_negative_out_of_range_and_boolean_indices(bad_index):
    _, _, _, _, classified = mapping_fixture()
    if bad_index == "out_of_range":
        bad_index = len(classified)
        source_index = 0
    else:
        source_index = bad_index
    selected = [dict(classified[source_index], sample_index=0)]
    with pytest.raises(ValueError, match="SHARED_INDEX_IDENTITY"):
        profiles.local_classified_rows(selected, [bad_index], classified)


def test_local_classification_rejects_classified_shared_index_mismatch():
    _, frames, _, _, classified = mapping_fixture()
    classified[0]["sample_index"] = 17
    with pytest.raises(ValueError, match="SHARED_INDEX_IDENTITY"):
        profiles.local_classified_rows(frames[0][:1], [0], classified)


def test_shared_real_reference_extraction_matches_each_profile_after_local_remapping():
    config, case, counts_e, counts_i, tape = history_fixture()
    frames = [[selected_row(config, 0, 0, 5, 0, 0, 11, .5)],
              [selected_row(config, 1, 0, 5, 0, 1, 14, .25),
               selected_row(config, 0, 0, 5, 1, 0, 15, .25)]]
    shared, mappings = profiles.merge_selected_frames(frames)
    assert mappings == [[0], [1, 0]]
    classified = history.classify(config, tape, shared)
    shared_refs, _ = history.extract(config, case, tape, classified, counts_E=counts_e, counts_I=counts_i)
    assert set(shared_refs) == {0, 1}
    for selected, mapping in zip(frames, mappings):
        local = profiles.local_classified_rows(selected, mapping, classified)
        individual, _ = history.extract(config, case, tape, local, counts_E=counts_e, counts_I=counts_i)
        assert set(individual) == set(range(len(selected)))
        for row, shared_index in zip(local, mapping):
            expected_initial, expected_forcing = individual[row["sample_index"]]
            actual_initial, actual_forcing = shared_refs[shared_index]
            assert actual_initial == expected_initial
            assert actual_forcing.keys() == expected_forcing.keys()
            for key in actual_forcing:
                np.testing.assert_array_equal(actual_forcing[key], expected_forcing[key])



def full_settings_fixture():
    settings = settings_fixture()
    entries = []
    for field in range(6):
        for phase in ("baseline", "cue", "early_delay"):
            for left, right in (("numpy_dt_0p05", "numpy_dt_0p025"), ("numpy_dt_0p025", "numpy_dt_0p0125")):
                index = len(entries)*2
                entries.append(dict(condition="strong_cue", field_index=field, phase=phase, window_ms=5.,
                                    left_cohort=left, right_cohort=right,
                                    endpoints=[dict(row_index=index, count_threshold=1), dict(row_index=index+1, count_threshold=2)]))
    settings.update(schema="pfc_profile_history_v1", profiles=entries, required_source_seeds=list(range(95001, 95033)),
                    expected_units=100000, sample_size=640, importance_sampling=dict(samples=10000), scope="test_fixture")
    settings["pilot"].update(source_rng_seed=95001, source_labels=["half_A", "quarter_A"])
    return settings


def test_optional_source_seed_preserves_all_legacy_requests_without_accepted_pilot():
    settings = full_settings_fixture()
    ordinary = {label: profiles.source_requests(settings, label, pilot=False) for label in profiles.SOURCE_CLOCKS}
    assert sum(map(len, ordinary.values())) == 142
    for seed in (None, 95001, 95002, 95032):
        for label in profiles.SOURCE_CLOCKS:
            assert profiles.source_requests(settings, label, pilot=False, seed=seed) == ordinary[label]


def test_accepted_pilot_excludes_only_its_seed_source_and_profile_in_full_mode():
    original = full_settings_fixture()
    settings = dict(original, accepted_pilot=dict(readback={}, source_runs=dict(half_A={}, quarter_A={})))
    for seed in settings["required_source_seeds"]:
        counts = {}
        for label in profiles.SOURCE_CLOCKS:
            before = profiles.source_requests(original, label, pilot=False)
            after = profiles.source_requests(settings, label, pilot=False, seed=seed)
            counts[label] = len(after)
            if seed == 95001 and label in ("half_A", "quarter_A"):
                assert after == [request for request in before if request["key"] not in settings["pilot"]["profile_keys"]]
                assert len(before) - len(after) == 2
            else:
                assert after == before
            assert profiles.source_requests(settings, label, pilot=False) == before
            assert profiles.source_requests(settings, label, pilot=True, seed=seed) == profiles.source_requests(original, label, pilot=True)
        assert sum(counts.values()) == (138 if seed == 95001 else 142)
        if seed == 95001:
            assert counts == dict(native=35, half_A=33, quarter_A=16, half_B=36, quarter_B=18)


def preparation_fixture(tmp_path, *, accepted=True, change=None):
    """Small hash-bound receipt documents; no simulator data or science scores."""
    repo, work = tmp_path/"repo", tmp_path/"work"
    repo.mkdir(); work.mkdir()
    settings = full_settings_fixture()

    def receipt(relative, status, payloads, *, verified=True):
        origin = repo/relative
        entries = []
        for name, value in payloads.items():
            content = value.encode() if isinstance(value, str) else (json.dumps(value)+"\n").encode()
            destination = origin/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            entries.append(dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        delivery = dict(status=status, verified=verified, files=entries)
        origin.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(delivery)+"\n").encode()
        (origin/"delivery.json").write_bytes(content)
        return dict(archive=relative, delivery_sha256=hashlib.sha256(content).hexdigest()), {entry["path"]: entry for entry in entries}

    sources, source_payloads = {}, {}
    for label, dt in (("half_A", "0p025"), ("quarter_A", "0p0125")):
        metadata = "source_metadata/95001/"+label
        source_payloads[metadata+"/report.json"] = dict(label=label)
        sources[label] = dict(archive="original_source", delivery_sha256="a"*64,
                              raw=dict(path=f"source_cases/dt_{dt}/pfc_95001_strong_cue/raw.npz"),
                              metadata=metadata, refinement_seeds=[95411, 95412])
    source_index = dict(profiles=settings["profiles"],
                        source_blocks={str(seed): dict(sources=deepcopy(sources) if seed == 95001 else {})
                                       for seed in settings["required_source_seeds"]})
    source_payloads["source_index.json"] = source_index
    settings["source_preparation"], source_entries = receipt("source_preparation", "COMPLETE_PFC_HISTORY_PROFILE_CENSUS_PREPARE", source_payloads)
    settings["census_readback"], _ = receipt("census_readback", "COMPLETE_PFC_HISTORY_PROFILE_SUMMARY_PREPARE", {
        "run_index.json": dict(runs=[]), "analysis/report.json": dict(status="COMPLETE_SHARED_PROFILE_CENSUS_READBACK", completed_original_blocks=32, failures=[])})
    settings["reused_integration_summary"], _ = receipt("old_integration", "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE", {})
    source_runs, runtime_bindings, frame_reports = {}, {}, {}
    keys = settings["pilot"]["profile_keys"]
    for label in ("half_A", "quarter_A"):
        source = sources[label]
        binding = dict(archive=source["archive"], delivery_sha256=source["delivery_sha256"],
                       case_path=source["raw"]["path"].rsplit("/", 1)[0],
                       report_sha256=source_entries[source["metadata"]+"/report.json"]["sha256"],
                       refinement_seeds=source["refinement_seeds"])
        if change == "source_binding" and label == "half_A":
            binding["delivery_sha256"] = "b"*64
        runtime_bindings[label] = deepcopy(binding)
        old_settings = deepcopy(settings)
        if change == "old_config" and label == "half_A":
            old_settings["importance_sampling"]["samples"] = 9999
        frames = [dict(key=key, sample_mean=0., integrated_histories=0, inner_trials=0) for key in keys]
        frame_reports[label] = frames
        source_report = dict(status="COMPLETE_PFC_PROFILE_SOURCE_HISTORY", source_rng_seed=95001, source_label=label,
                             pilot=True, requested_profile_keys=list(keys), completed_profile_keys=list(keys), frame_reports=frames)
        if label == "half_A":
            if change == "source_report_incomplete": source_report["status"] = "FAILED_PFC_PROFILE_SOURCE_HISTORY"
            elif change == "source_label": source_report["source_label"] = "quarter_A"
            elif change == "profile_keys": source_report["completed_profile_keys"] = keys[:1]
        source_runs[label], _ = receipt("pilot_"+label, "COMPLETE_PFC_PROFILE_HISTORY_COMPLETE", {
            "report.json": dict(status="COMPLETE_PFC_PROFILE_HISTORY_COMPLETE", exit_code=0, error=None),
            "analysis/report.json": source_report,
            "Libs/configs/neural_population_pfc_profile_history.yaml": yaml.safe_dump(old_settings),
            "runtime_config.json": dict(old_settings, source_binding=binding),
        }, verified=not (change == "source_delivery_unverified" and label == "half_A"))
    pairs = []
    for index, key in enumerate(keys):
        per_clock = {}
        for clock, label in (("coarse", "half_A"), ("fine", "quarter_A")):
            per_clock[clock] = dict(frame_reports[label][index], source_label=label, source_binding=deepcopy(runtime_bindings[label]))
        pairs.append(dict(key=key, per_clock=per_clock))
    readback = dict(status="COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK", source_rng_seed=95001,
                    original_source_blocks=1, paired_profiles=2, pairs=pairs,
                    source_costs={clock: dict(source_binding=deepcopy(runtime_bindings[label]))
                                  for clock, label in (("coarse", "half_A"), ("fine", "quarter_A"))})
    if change == "readback_report_incomplete": readback["status"] = "FAILED_PFC_SHARED_PROFILE_HISTORY_READBACK"
    elif change == "seed": readback["source_rng_seed"] = 95002
    elif change == "readback_binding": readback["pairs"][0]["per_clock"]["coarse"]["source_binding"]["archive"] = "wrong_source"
    readback_ref, _ = receipt("pilot_readback", "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE", {
        "report.json": dict(status="COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE", exit_code=0, error=None),
        "analysis/report.json": readback})
    if change == "delivery_digest": readback_ref["delivery_sha256"] = "0"*64
    if accepted:
        settings["accepted_pilot"] = dict(readback=readback_ref, source_runs=source_runs)
    copied = []

    def copy(source, destination, expected=None):
        content = source.read_bytes()
        if expected is not None and hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("FIXTURE_COPY_HASH_MISMATCH")
        target = work/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied.append(str(destination))

    return settings, repo, work, copy, copied


@pytest.mark.parametrize("accepted", [False, True])
def test_prepare_counts_only_bound_accepted_pilot_reuse_and_preserves_default(tmp_path, accepted):
    settings, repo, work, copy, copied = preparation_fixture(tmp_path, accepted=accepted)
    before = deepcopy(settings)
    report = profiles.prepare_inputs(settings, repo, work, copy)
    assert report["full_source_case_requests"] == dict(native=35, half_A=35, quarter_A=18, half_B=36, quarter_B=18)
    assert report["remaining_clock_frames"] == (4540 if accepted else 4544)
    assert report["remaining_paired_profiles"] == (2270 if accepted else 2272)
    assert report["reused_pilot_clock_frames"] == (4 if accepted else 0)
    for seed, counts in report["remaining_source_case_requests"].items():
        assert sum(counts.values()) == (138 if accepted and seed == "95001" else 142)
    assert settings == before
    if accepted:
        assert len([path for path in copied if path.startswith("accepted_pilot/")]) == 13
        for label in ("half_A", "quarter_A"):
            staged = yaml.safe_load((work/"accepted_pilot"/label/"Libs/configs/neural_population_pfc_profile_history.yaml").read_text())
            assert staged == {key: value for key, value in settings.items() if key != "accepted_pilot"}
    else:
        assert not any(path.startswith("accepted_pilot/") for path in copied)


@pytest.mark.parametrize("change", ["readback_report_incomplete", "source_report_incomplete", "source_delivery_unverified",
                                    "seed", "source_label", "profile_keys", "old_config", "source_binding",
                                    "readback_binding", "delivery_digest"])
def test_prepare_rejects_incomplete_or_mismatched_accepted_pilot(tmp_path, change):
    settings, repo, work, copy, _ = preparation_fixture(tmp_path, change=change)
    with pytest.raises(ValueError):
        profiles.prepare_inputs(settings, repo, work, copy)
    assert not (work/"analysis/report.json").exists()

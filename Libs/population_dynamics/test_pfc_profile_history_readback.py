"""Synthetic archive fixtures for numerical readback; values are not scientific evidence."""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
import tarfile

import numpy as np
import pytest

from . import pfc_first_passage_history as history
from . import pfc_first_passage_importance as importance
from . import pfc_profile_history as producer
from . import pfc_profile_history_readback as reader
from .pfc_history_profile_census import BOUNDARY, write_json
from .pfc_history_timing_census import census, pair
from .pfc_profile_history import seed_identity
from .pfc_stratified_history import sampling_callbacks, paired_components
from .test_pfc_history_timing_census import config as census_config, raw as census_raw
from .test_pfc_profile_history import full_settings_fixture


def add_member(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


def fixture(tmp_path, *, empty=False, pair_kind="half_quarter", repeat="A"):
    cohorts = (("numpy_dt_0p05", "numpy_dt_0p025") if pair_kind == "native_half"
               else ("numpy_dt_0p025", "numpy_dt_0p0125"))
    profile = dict(condition="strong_cue", field_index=4, phase="cue", window_ms=5.,
                   left_cohort=cohorts[0], right_cohort=cohorts[1],
                   endpoints=[dict(row_index=0, count_threshold=1), dict(row_index=1, count_threshold=2)])
    options = dict(samples=10000, batch_size=128, proposal_multipliers=[1., 2.],
                   mixture_probabilities=[.5, .5], confidence_level=.95)
    settings = dict(profiles=[profile], auxiliary_repeats=["A", "B"], required_source_seeds=list(range(95001, 95033)),
                    expected_units=40, sample_size=640, integration_seed_namespace=5,
                    sampling_seed_epoch=20261010, phase_codes=dict(cue=1),
                    clock_pair_codes=dict(native_half=0, half_quarter=1), auxiliary_repeat_codes=dict(A=0, B=1),
                    importance_sampling=options)
    source_binding = dict(archive="fixture/coupled_"+repeat, delivery_sha256=("a" if repeat == "A" else "b")*64,
                          refinement_seeds=[95411, 95412] if repeat == "A" else [99411, 99412])
    bindings = dict(coarse=deepcopy(source_binding), fine=deepcopy(source_binding))
    if pair_kind == "native_half":
        bindings["coarse"] = dict(archive="fixture/native", delivery_sha256="c"*64, refinement_seeds=None)
    labels = ("native", "half_"+repeat) if pair_kind == "native_half" else ("half_"+repeat, "quarter_"+repeat)
    dts = (.05, .025) if pair_kind == "native_half" else (.025, .0125)
    configs, raws, clocks = {}, {}, {}
    for clock, dt in zip(("coarse", "fine"), dts):
        config = census_config(dt=dt)
        events = ((100., 8), (110., 8), (113., 9), (115., 8)) if clock == "coarse" else ((105., 9), (110., 8), (110., 9), (115., 8))
        raw = census_raw(config, [] if empty else [(round(ms/dt), unit) for ms, unit in events])
        configs[clock], raws[clock] = config, raw
        clocks[clock] = census(config, raw, field_index=4, phase="cue", window_ms=5., time_edges_ms=[0., 1., 2., 3.])
    census_report, census_arrays = pair(*clocks["coarse"], *clocks["fine"])
    census_report.update(status="COMPLETE_PFC_HISTORY_PROFILE_FRAME", profile=profile, auxiliary_repeat=repeat, source_rng_seed=95001)
    sampling, context = seed_identity(settings, 95001, dict(profile=profile, auxiliary_repeat=repeat, pair_name=pair_kind))
    integrated_by_clock, frames = {}, {}
    for clock, label in zip(("coarse", "fine"), labels):
        source_binding = bindings[clock]
        directory = tmp_path/clock
        directory.mkdir()
        config = configs[clock]
        sampler, summarize = sampling_callbacks(census_report, census_arrays, clock)
        selected, frame = sampler(config, field=4, phase="cue", window_ms=5., sample_size=640, seed_components=sampling)
        classified = history.classify(config, history.reference.validate_tape(config, raws[clock]), selected)
        for row in classified:
            row["shared_history_index"] = row["sample_index"] + (100 if clock == "coarse" else 200)
        rows = deepcopy(classified)
        members = []
        eligible = 0
        for row in rows:
            row.update(conditional_estimate=0., inner_monte_carlo_variance=0., proposal_hits=0)
            if row["category"] != "conditional_integration_required":
                continue
            eligible += 1
            hits = 0 if clock == "coarse" and row["frame_index"] == 4 else 2
            nominal = importance.native._RATE_HZ * (config["dt_ms"] * 1e-3)
            proposals = nominal*np.asarray(options["proposal_multipliers"])
            bound = 2.
            prefix = importance.summarize_score(float(hits), float(hits), 10000, bound, .95)
            full = importance.summarize_score(.5*hits, .25*hits, 10000, bound, .95)
            inner = dict(status="COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE", samples=10000, batch_size=128,
                         proposal_hits=hits, proposal_no_hits=10000-hits, nominal_probability=nominal,
                         proposal_probabilities=proposals.tolist(), mixture_probabilities=[.5, .5], weight_bound=bound,
                         stopped_prefix=prefix, full_path=full,
                         seed_components=[20261009, 95001, 4, row["frame_index"], round(config["dt_ms"]*1000), 5, *context],
                         reference_archive=source_binding["archive"], reference_delivery_sha256=source_binding["delivery_sha256"],
                         reference_archive_role="accepted_source_provenance_for_extracted_reference",
                         extracted_reference="../reference", sample_frame_row=deepcopy(row), target=reader._TARGET,
                         finite_PRNG_law_certified=False, population_history_uncertainty_established=False,
                         population_first_passage_probability_established=False, third_spike_probability_established=False,
                         scientific_fidelity_qualified=False, confirmation_execution_released=False,
                         new_full_network_episodes=0, new_independent_source_blocks=0, production_episodes=0)
            relative = f"sample_{row['sample_index']:03d}"
            conditional = dict(unit_index=row["unit_index"], population="E", first_spike_tick=row["first_spike_tick"],
                               window_stop_tick=row["window_stop_tick"], fresh_input_included=False)
            members.append((relative+"/integration/report.json", json.dumps(inner).encode()))
            members.append((relative+"/reference/conditional_initial.json", json.dumps(conditional).encode()))
            buffer = BytesIO()
            np.savez_compressed(buffer, tick=np.arange(row["first_spike_tick"]+1, row["window_stop_tick"]))
            members.append((relative+"/reference/reference_window.npz", buffer.getvalue()))
            row.update(conditional_estimate=prefix["estimate"], inner_monte_carlo_variance=prefix["empirical_score_variance"]/10000,
                       proposal_hits=hits, integration_report=relative+"/integration/report.json")
        if members:
            with tarfile.open(directory/"integration_details.tar", "w") as archive:
                for name, payload in members:
                    add_member(archive, name, payload)
        actual = summarize(rows, frame)
        actual.update(status="COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY", source_rng_seed=95001,
                      source_label=label, input_stream="native_numpy_stream" if label == "native" else "coupled_binomial_clock_refinement_stream",
                      dt_ms=config["dt_ms"], source_parent_dt_ms=None if label == "native" else .05,
                      profile=profile, auxiliary_repeat=repeat, clock=clock, frame=frame,
                      integrated_histories=eligible, inner_trials=eligible*10000,
                      categories=dict(Counter(row["category"] for row in rows)), integration_seed_namespace=5,
                      integration_seed_context=context, third_spike_probability_established=False,
                      source_history_population_uncertainty_established=False, **BOUNDARY)
        for name, value in (("report.json", actual), ("selected_frame.json", dict(frame=frame, rows=selected)),
                            ("classified_frame.json", classified), ("integrated_frame.json", rows)):
            write_json(directory/name, value)
        integrated_by_clock[clock], frames[clock] = rows, frame
    arguments = dict(settings=settings, census_report=census_report, census_arrays=census_arrays,
                     coarse_dir=tmp_path/"coarse", fine_dir=tmp_path/"fine", source_bindings=bindings)
    return arguments, integrated_by_clock, frames


def alter_inner(directory, change):
    path = directory/"integration_details.tar"
    with tarfile.open(path, "r:") as archive:
        members = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers()]
    for index, (name, payload) in enumerate(members):
        if name.endswith("/integration/report.json"):
            inner = json.loads(payload)
            change(inner)
            members[index] = (name, json.dumps(inner).encode())
            break
    with tarfile.open(path, "w") as archive:
        for name, payload in members:
            add_member(archive, name, payload)


def test_read_pair_reconstructs_tar_moments_and_matches_paired_statistics(tmp_path):
    arguments, rows, frames = fixture(tmp_path)
    expected, expected_rows = paired_components(rows["coarse"], rows["fine"], frames["coarse"])
    output = tmp_path/"readback"
    result = reader.read_pair(**arguments, output=output)
    assert result["status"] == "COMPLETE_PFC_PROFILE_HISTORY_READBACK"
    assert result["statistics"] == expected and result["paired_selected_physical_windows"] == 5
    assert result["per_clock"]["coarse"]["eligible_zero_estimates"] == 1
    assert result["per_clock"]["coarse"]["structural_zero_histories"] == 2
    assert result["per_clock"]["fine"]["structural_zero_histories"] == 1
    assert result["per_clock"]["coarse"]["moment_reports_recomputed"] == 6
    assert result["per_clock"]["fine"]["moment_reports_recomputed"] == 8
    assert json.loads((output/"paired_frame.json").read_text()) == dict(frame=frames["coarse"], rows=expected_rows)
    assert json.loads((output/"report.json").read_text()) == result
    assert not any(output.glob("sample_*"))
    assert not result["population_confidence_interval_computed"] and not result["scientific_fidelity_qualified"]
    assert not result["third_spike_probability_established"] and not result["confirmation_execution_released"]


@pytest.mark.parametrize("change", ["stopped_sum", "full_squares", "seed", "source", "bound", "snapshot_identity"])
def test_read_pair_rejects_changed_inner_reports_and_writes_failure(tmp_path, change):
    arguments, _, _ = fixture(tmp_path)

    def mutate(inner):
        if change == "stopped_sum":
            inner["stopped_prefix"]["score_sum"] += 1.
        elif change == "full_squares":
            inner["full_path"]["score_square_sum"] += .25
        elif change == "seed":
            inner["seed_components"][-1] = 1
        elif change == "source":
            inner["reference_delivery_sha256"] = "b"*64
        elif change == "bound":
            inner["weight_bound"] = 4.
        else:
            inner["sample_frame_row"]["unit_index"] += 1

    alter_inner(arguments["coarse_dir"], mutate)
    output = tmp_path/"readback"
    with pytest.raises(ValueError):
        reader.read_pair(**arguments, output=output)
    failure = json.loads((output/"report.json").read_text())
    assert failure["status"] == "FAILED_PFC_PROFILE_HISTORY_READBACK" and failure["error"]["type"] == "ValueError"
    assert not (output/"paired_frame.json").exists()


def test_read_pair_ignores_numeric_placeholders_in_preintegration_identity_snapshot(tmp_path):
    arguments, rows, frames = fixture(tmp_path)
    def mutate(inner):
        inner["sample_frame_row"].update(conditional_estimate=99., inner_monte_carlo_variance=-1., proposal_hits=123)
    alter_inner(arguments["coarse_dir"], mutate)
    result = reader.read_pair(**arguments, output=tmp_path/"readback")
    assert result["statistics"] == paired_components(rows["coarse"], rows["fine"], frames["coarse"])[0]


def test_read_pair_rejects_selected_physical_frame_mismatch(tmp_path):
    arguments, _, _ = fixture(tmp_path)
    path = arguments["fine_dir"]/"selected_frame.json"
    selected = json.loads(path.read_text())
    selected["rows"][0]["window_start_ms"] += .0125
    write_json(path, selected)
    with pytest.raises(ValueError, match="SELECTED_FRAME_CHANGED"):
        reader.read_pair(**arguments, output=tmp_path/"readback")
    assert json.loads((tmp_path/"readback/report.json").read_text())["status"] == "FAILED_PFC_PROFILE_HISTORY_READBACK"


def test_read_pair_preserves_empty_frames_without_integration_tar(tmp_path):
    arguments, _, _ = fixture(tmp_path, empty=True)
    assert not (arguments["coarse_dir"]/"integration_details.tar").exists()
    result = reader.read_pair(**arguments, output=tmp_path/"readback")
    assert result["paired_selected_physical_windows"] == 0 and result["statistics"]["sample_mean"] == 0.
    assert all(result["per_clock"][clock]["inner_trials"] == 0 for clock in ("coarse", "fine"))


def test_read_pair_requires_tar_for_eligible_histories(tmp_path):
    arguments, _, _ = fixture(tmp_path)
    (arguments["fine_dir"]/"integration_details.tar").unlink()
    with pytest.raises(FileNotFoundError):
        reader.read_pair(**arguments, output=tmp_path/"readback")
    assert json.loads((tmp_path/"readback/report.json").read_text())["status"] == "FAILED_PFC_PROFILE_HISTORY_READBACK"


@pytest.mark.parametrize("change", ["summary", "repeat", "parent"])
def test_read_pair_rejects_changed_clock_summary_repeat_or_original_parent(tmp_path, change):
    arguments, _, _ = fixture(tmp_path)
    path = arguments["fine_dir"]/"report.json"
    report = json.loads(path.read_text())
    if change == "summary":
        report["sample_mean"] += .1
    elif change == "repeat":
        report["auxiliary_repeat"] = "B"
    else:
        report["source_parent_dt_ms"] = .025
    write_json(path, report)
    with pytest.raises(ValueError):
        reader.read_pair(**arguments, output=tmp_path/"readback")
    assert json.loads((tmp_path/"readback/report.json").read_text())["status"] == "FAILED_PFC_PROFILE_HISTORY_READBACK"


def test_read_pair_rejects_row_paths_outside_expected_tar_member(tmp_path):
    arguments, _, _ = fixture(tmp_path)
    path = arguments["coarse_dir"]/"integrated_frame.json"
    rows = json.loads(path.read_text())
    next(row for row in rows if row["category"] == "conditional_integration_required")["integration_report"] = "../../outside.json"
    write_json(path, rows)
    with pytest.raises(ValueError, match="INTEGRATION_REPORT_PATH"):
        reader.read_pair(**arguments, output=tmp_path/"readback")



def test_actual_packer_roundtrip_preserves_reader_inputs_and_statistics(tmp_path, monkeypatch):
    arguments, rows, frames = fixture(tmp_path)
    expected_statistics = paired_components(rows["coarse"], rows["fine"], frames["coarse"])[0]
    payloads_by_directory, direct_frames, expected_cleanup = {}, {}, set()

    def payloads(path):
        with tarfile.open(path, "r:*") as archive:
            return {member.name: archive.extractfile(member).read()
                    for member in archive.getmembers() if member.isfile()}

    for clock in ("coarse", "fine"):
        directory = arguments[clock+"_dir"]
        archive_path = directory/"integration_details.tar"
        original_payloads = payloads(archive_path)
        payloads_by_directory[directory] = original_payloads
        # These members were created by the local fixture, not supplied externally.
        for name, content in original_payloads.items():
            destination = directory/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        archive_path.unlink()
        expected_cleanup.update(directory/row["integration_report"].split("/")[0]
                                for row in rows[clock] if row["category"] == "conditional_integration_required")
        direct_frames[directory] = {name: (directory/name).read_bytes() for name in
                                    ("selected_frame.json", "classified_frame.json", "integrated_frame.json")}

    removed = []
    original_rmtree = producer.shutil.rmtree

    def checked_rmtree(path, *args, **kwargs):
        path = type(tmp_path)(path)
        assert path in expected_cleanup and path.is_dir()
        # Observe the deletion boundary: every original member must already be
        # recoverable exactly, including samples that have not been deleted yet.
        assert payloads(path.parent/"integration_details.tar") == payloads_by_directory[path.parent]
        removed.append(path)
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(producer.shutil, "rmtree", checked_rmtree)
    for clock in ("coarse", "fine"):
        directory = arguments[clock+"_dir"]
        packed = producer.pack_integration_details(directory, rows[clock])
        assert payloads(directory/"integration_details.tar") == payloads_by_directory[directory]
        assert all((directory/name).read_bytes() == content for name, content in direct_frames[directory].items())
        report = json.loads((directory/"report.json").read_text())
        report.update(packed)
        write_json(directory/"report.json", report)
    assert set(removed) == expected_cleanup and len(removed) == len(expected_cleanup)
    assert all(not directory.exists() for directory in expected_cleanup)
    result = reader.read_pair(**arguments, output=tmp_path/"readback")
    assert result["status"] == "COMPLETE_PFC_PROFILE_HISTORY_READBACK"
    assert result["statistics"] == expected_statistics



def routing_fixture():
    history_settings = full_settings_fixture()
    history_settings["accepted_pilot"] = dict(readback={}, source_runs=dict(half_A={}, quarter_A={}))
    settings = dict(schema="profile_block_readback_fixture", source_rng_seeds=[95001],
                    source_arrays=[dict(archive_prefix="Output/profile_history/complete/run_ARRAY", source_index_start=0, source_index_stop=4)],
                    source_preparation=dict(archive="prepared_history", delivery_sha256="prepared_receipt"),
                    source_config_sha="frozen_history_config", scientific_fidelity_qualified=False,
                    confirmation_execution_released=False, production_episodes=0)
    return settings, history_settings


def test_legacy_readback_request_preserves_fields_and_avoids_new_routing_requirements():
    settings = dict(schema="legacy", source_rng_seed=95001, profile_keys=["A/half_quarter/field_4_cue"],
                    source_runs=dict(coarse=dict(source_label="half_A", archive="old_half"),
                                     fine=dict(source_label="quarter_A", archive="old_quarter")),
                    source_preparation=dict(archive="old_preparation"), source_config_sha="old_config")
    original = deepcopy(settings)
    assert reader.readback_requests(settings, {}) == [dict(original, full_source_requests=False)]
    assert settings == original


def test_block_routing_has_four_ordered_groups_and_complete_remaining_source_keys():
    settings, history_settings = routing_fixture()
    original = deepcopy(settings)
    requests = reader.readback_requests(settings, history_settings)
    names = ["A/native_half", "B/native_half", "A/half_quarter", "B/half_quarter"]
    assert [request["group_name"] for request in requests] == names
    assert [request["group_index"] for request in requests] == list(range(4))
    assert [len(request["profile_keys"]) for request in requests] == [17, 18, 16, 18]
    indices = [(0, 1), (0, 3), (1, 2), (3, 4)]
    source_sizes = dict(native=35, half_A=33, quarter_A=16, half_B=36, quarter_B=18)
    for request, (coarse_index, fine_index) in zip(requests, indices):
        assert request["source_rng_seed"] == 95001 and request["full_source_requests"] is True
        assert request["source_preparation"] == settings["source_preparation"]
        assert request["source_config_sha"] == settings["source_config_sha"]
        assert all(key.startswith(request["group_name"]+"/") for key in request["profile_keys"])
        assert request["scientific_fidelity_qualified"] is False and request["production_episodes"] == 0
        for clock, index in (("coarse", coarse_index), ("fine", fine_index)):
            source = request["source_runs"][clock]
            assert source["archive"] == "Output/profile_history/complete/run_ARRAY_"+str(index)
            assert len(source["expected_profile_keys"]) == source_sizes[source["source_label"]]
    assert not set(history_settings["pilot"]["profile_keys"]) & set(requests[2]["profile_keys"])
    assert "B/half_quarter/field_4_cue" in requests[3]["profile_keys"]
    assert "B/half_quarter/field_5_early_delay" in requests[3]["profile_keys"]
    assert settings == original


def test_block_routing_preserves_target_seed_order_and_global_source_indices_across_ranges():
    settings, history_settings = routing_fixture()
    settings["source_rng_seeds"] = [95002, 95001, 95032]
    settings["source_arrays"] = [dict(archive_prefix="first", source_index_start=0, source_index_stop=4),
                                 dict(archive_prefix="later", source_index_start=5, source_index_stop=159)]
    requests = reader.readback_requests(settings, history_settings)
    assert [request["source_rng_seed"] for request in requests] == [95002]*4+[95001]*4+[95032]*4
    assert [request["group_index"] for request in requests] == list(range(12))
    assert [len(request["profile_keys"]) for request in requests[:4]] == [17, 18, 18, 18]
    assert set(history_settings["pilot"]["profile_keys"]) <= set(requests[2]["profile_keys"])
    assert requests[0]["source_runs"]["coarse"]["archive"] == "later_5"
    assert requests[4]["source_runs"]["coarse"]["archive"] == "first_0"
    assert requests[8]["source_runs"]["coarse"]["archive"] == "later_155"
    assert requests[-1]["source_runs"]["fine"]["archive"] == "later_159"


def test_all32_routing_retains_2270_pairs_and_4540_distinct_clock_frames():
    settings, history_settings = routing_fixture()
    settings["source_rng_seeds"] = list(history_settings["required_source_seeds"])
    settings["source_arrays"][0]["source_index_stop"] = 159
    requests = reader.readback_requests(settings, history_settings)
    assert len(requests) == 128 and sum(len(request["profile_keys"]) for request in requests) == 2270
    sources = {}
    for request in requests:
        for source in request["source_runs"].values():
            identity = (request["source_rng_seed"], source["source_label"])
            if identity in sources:
                assert sources[identity] == source
            sources[identity] = source
    assert len(sources) == 160
    assert sum(len(source["expected_profile_keys"]) for source in sources.values()) == 4540
    for seed in history_settings["required_source_seeds"]:
        assert sum(len(source["expected_profile_keys"]) for (source_seed, _), source in sources.items()
                   if source_seed == seed) == (138 if seed == 95001 else 142)


@pytest.mark.parametrize("change", ["empty_seeds", "duplicate_seeds", "foreign_seed", "seed_type", "empty_arrays", "overlap",
                                    "missing_coverage", "negative_index", "large_index", "reversed_range", "index_type", "history_domain"])
def test_block_routing_rejects_invalid_seed_or_source_index_coverage(change):
    settings, history_settings = routing_fixture()
    source = settings["source_arrays"][0]
    if change == "empty_seeds": settings["source_rng_seeds"] = []
    elif change == "duplicate_seeds": settings["source_rng_seeds"] = [95001, 95001]
    elif change == "foreign_seed": settings["source_rng_seeds"] = [95033]
    elif change == "seed_type": settings["source_rng_seeds"] = ["95001"]
    elif change == "empty_arrays": settings["source_arrays"] = []
    elif change == "overlap": settings["source_arrays"].append(dict(archive_prefix="overlap", source_index_start=4, source_index_stop=9))
    elif change == "missing_coverage": source["source_index_stop"] = 3
    elif change == "negative_index": source["source_index_start"] = -1
    elif change == "large_index": source["source_index_stop"] = 160
    elif change == "reversed_range": source.update(source_index_start=4, source_index_stop=0)
    elif change == "index_type": source["source_index_start"] = True
    else: history_settings["required_source_seeds"] = history_settings["required_source_seeds"][:-1]
    with pytest.raises(ValueError):
        reader.readback_requests(settings, history_settings)


@pytest.mark.parametrize("pair_kind,repeat", [("native_half", "A"), ("native_half", "B"), ("half_quarter", "B")])
def test_actual_read_pair_accepts_native_and_repeat_b_source_bindings(tmp_path, pair_kind, repeat):
    arguments, rows, frames = fixture(tmp_path, pair_kind=pair_kind, repeat=repeat)
    result = reader.read_pair(**arguments, output=tmp_path/"readback")
    assert result["status"] == "COMPLETE_PFC_PROFILE_HISTORY_READBACK"
    assert result["pair_name"] == pair_kind and result["auxiliary_repeat"] == repeat
    assert result["statistics"] == paired_components(rows["coarse"], rows["fine"], frames["coarse"])[0]
    for clock in ("coarse", "fine"):
        report = json.loads((arguments[clock+"_dir"]/"report.json").read_text())
        source = result["per_clock"][clock]
        assert source["source_binding"] == arguments["source_bindings"][clock]
        assert report["integration_seed_context"][-1] == (0 if repeat == "A" else 1)
        if pair_kind == "native_half" and clock == "coarse":
            assert source["source_label"] == "native" and report["source_parent_dt_ms"] is None
            assert report["input_stream"] == "native_numpy_stream"
            assert source["source_binding"]["refinement_seeds"] is None
        else:
            assert source["source_label"].endswith("_"+repeat) and report["source_parent_dt_ms"] == .05
            assert report["input_stream"] == "coupled_binomial_clock_refinement_stream"
    assert result["scientific_fidelity_qualified"] is False and result["confirmation_execution_released"] is False

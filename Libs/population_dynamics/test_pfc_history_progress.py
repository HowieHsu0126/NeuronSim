"""PFC progress consumes bound receipts and preserves scientific and scheduler state."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from population_dynamics import pfc_history_progress as progress
from population_dynamics.provenance import canonical_json, sha256_file


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")
    return sha256_file(path)


def unqualified() -> dict:
    return dict(scientific_fidelity_qualified=False, confirmation_execution_released=False,
                production_episodes=0, new_independent_source_blocks=0)


@pytest.fixture
def receipt_case(tmp_path, monkeypatch):
    root = tmp_path
    keys = ["A/native_half/field_4_cue"]
    original_binding = dict(archive="original/native", delivery_sha256="original_delivery",
                            case_path="source_run", report_sha256="original_report", refinement_seeds=None)
    fine_binding = dict(original_binding, archive="original/half_A", refinement_seeds=[95443, 95444])
    sources = {80: dict(seed=95017, label="native", binding=original_binding, profiles=keys),
               81: dict(seed=95017, label="half_A", binding=fine_binding, profiles=keys,
                        config_sha256="source_config", source_inputs={"source_inputs/source/report.json": "origin_report"})}
    reader = dict(source_rng_seed=95017, group_name="A/native_half", profile_keys=keys,
                  config_sha256="reader_config", source_runs={
                      "coarse": dict(archive=progress.SOURCE_PREFIX + "_80", source_label="native"),
                      "fine": dict(archive=progress.SOURCE_PREFIX + "_81", source_label="half_A")})
    monkeypatch.setattr(progress, "frozen_requests", lambda root, branch, snapshots: (sources, {60: reader}))
    source_archive = root / (progress.SOURCE_PREFIX + "_81")
    source_analysis = dict(unqualified(), status="COMPLETE_PFC_PROFILE_SOURCE_HISTORY", source_rng_seed=95017,
                           source_label="half_A", pilot=False, requested_profile_keys=keys,
                           completed_profile_keys=keys, frame_reports=[dict(key=keys[0])],
                           reconstruction={"ticks_reconstructed": 200000}, reconstruction_seconds=5.0,
                           reconstruction_drive={"scope": "external_input_generator"}, total_inner_trials=100,
                           distinct_selected_physical_windows=1, distinct_integrated_references=1, peak_rss_kib=1000)
    source_report = dict(unqualified(), status="COMPLETE_PFC_PROFILE_HISTORY_COMPLETE", job_id="610289_81",
                         exit_code=0, task="profile_history", mode="complete", inputs=[
                             dict(path=progress.SOURCE_CONFIG, sha256="source_config"),
                             dict(path="source_inputs/source/report.json", sha256="origin_report")])
    source_delivery = dict(status=source_report["status"], job_id="610289_81", verified=True)
    source = dict(job_id="610289_81", native_job_id="610289", archive=str(source_archive),
                  status=source_report["status"], source_rng_seed=95017, source_label="half_A",
                  source_binding=fine_binding, analysis_status=source_analysis["status"], complete_profile_mappings=1,
                  **{key: source_analysis[key] for key in (
                      "reconstruction", "reconstruction_seconds", "reconstruction_drive", "total_inner_trials",
                      "distinct_selected_physical_windows", "distinct_integrated_references", "peak_rss_kib")})
    for name, value, field in (("delivery.json", source_delivery, "delivery_sha256"),
                               ("report.json", source_report, "report_sha256"),
                               ("analysis/report.json", source_analysis, "analysis_sha256")):
        source[field] = write_json(source_archive / name, value)
    source_receipt = dict(unqualified(), status="ACCEPTED_PFC_ORIGINAL_SOURCE_HISTORY", source=source,
                          new_accepted_history_tasks=1, new_accepted_paired_profiles=0)
    source_path = root / progress.BASE_PATH / "source_history_acceptance_610289_81.json"
    write_json(source_path, source_receipt)
    costs = {clock: dict(history_execution_archive=run["archive"], history_execution_job_id="610289_" + index,
                         source_binding=sources[int(index)]["binding"], reconstruction_seconds=5.0,
                         wall_seconds=10.0, peak_rss_kib=1000, source_reconstruction_calls=1)
             for clock, run, index in (("coarse", reader["source_runs"]["coarse"], "80"),
                                       ("fine", reader["source_runs"]["fine"], "81"))}
    reader_archive = root / (progress.READER_PREFIX + "_60")
    reader_analysis = dict(unqualified(), status="COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK",
                           source_rng_seed=95017, readback_group_index=60, readback_group="A/native_half",
                           paired_profiles=1, full_source_requests=True, source_costs=costs,
                           pairs=[dict(key=keys[0])])
    reader_report = dict(unqualified(), status="COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE",
                         job_id="610296_60", exit_code=0, task="profile_readback", mode="complete",
                         inputs=[dict(path=progress.READER_CONFIG, sha256="reader_config")])
    reader_receipt = dict(unqualified(), status="ACCEPTED_PFC_PAIRED_PROFILE_HISTORY_READBACK", job_id="610296_60",
                          readback_group_index=60, archive=str(reader_archive), source_rng_seed=95017,
                          readback_group="A/native_half", paired_profiles=1, new_accepted_paired_profiles=1,
                          paired_profile_keys=keys, source_costs=costs)
    for name, value, field in (("delivery.json", dict(status=reader_report["status"], job_id="610296_60", verified=True),
                                "delivery_sha256"), ("report.json", reader_report, "report_sha256"),
                               ("analysis/report.json", reader_analysis, "analysis_sha256")):
        reader_receipt[field] = write_json(reader_archive / name, value)
    reader_path = root / progress.BASE_PATH / "paired_history_acceptance_610296_60.json"
    write_json(reader_path, reader_receipt)
    baseline = dict(status="ACCEPTED_COMPLETED_ORIGINAL_HISTORIES_AND_RECOVERED_READERS", accepted_sources=76,
                    accepted_pairs=1168, required_original_sources=155, required_pairs=2304,
                    production_episodes=0, scientific_fidelity_qualified=False,
                    sources=[dict(index=index) for index in range(59, 81)],
                    readers=[dict(index=index) for index in range(43, 60)])
    write_json(root / progress.BASELINE_PATH, baseline)
    branch = dict(accepted_history_tasks=76, receipt_qualified_pairs=1168, production_episodes=0,
                  status="ACTIVE_40_TASKS_40_READBACKS_76_HISTORIES_1168_PAIRS_100_NEW_SOURCES_100_NEW_READBACKS_ACCEPTED",
                  source_probability_law_qualified=False, scientific_fidelity_qualified=False,
                  parameter_domain_frozen=False, task_protocol_frozen=False, observation_map_frozen=False,
                  required_pairs=2304, required_original_sources=155, endpoints=84330, undefined_endpoints=1734,
                  development_episodes=140, failures=[dict(job="610296_45", status="FAILED", seconds=18)],
                  parameter_block_execution_20261012={"status": "COMPLETE", "production_episodes": 0})
    state = dict(other_circuit={"scientific_fidelity_qualified": False, "failure": "preserve"})
    cursor = state
    for key in progress.BRANCH_PATH[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[progress.BRANCH_PATH[-1]] = branch
    write_json(root / progress.STATE_PATH, state)
    write_json(root / progress.MIRROR_PATH, branch)
    ledger = b"# exact text and comments stay intact\nexecution:\n  pfc_history_accepted_original_sources: 76\n  pfc_history_accepted_pairs: 1168\n  other: false # preserve\n"
    (root / "state_ledger.yaml").write_bytes(ccfa)
    return dict(root=root, sources=sources, reader=reader, source_path=source_path, source_receipt=source_receipt,
                reader_path=reader_path, reader_receipt=reader_receipt, state=state, branch=branch, ledger=ccfa,
                source_archive=source_archive, reader_archive=reader_archive)


def test_read_only_then_write_preserves_state_and_is_idempotent(receipt_case):
    case = receipt_case
    paths = [case["root"] / name for name in (progress.STATE_PATH, progress.MIRROR_PATH, "state_ledger.yaml")]
    before = [path.read_bytes() for path in paths]
    result = progress.synchronize(case["root"])
    assert result["accepted_original_sources"] == 77 and result["accepted_pairs"] == 1169
    assert result["changed"] and not result["written"] and result["production_episodes"] == 0
    assert [path.read_bytes() for path in paths] == before
    assert progress.synchronize(case["root"], write=True)["written"]
    updated_state = json.loads(paths[0].read_bytes())
    updated = progress.branch_of(updated_state)
    for key, value in case["branch"].items():
        if key not in {"accepted_history_tasks", "receipt_qualified_pairs", "status"}:
            assert updated[key] == value
    assert updated_state["other_circuit"] == case["state"]["other_circuit"]
    assert json.loads(paths[1].read_bytes()) == updated
    assert paths[2].read_bytes() == case["ledger"].replace(b"sources: 76", b"sources: 77").replace(b"pairs: 1168", b"pairs: 1169")
    assert updated["history_receipt_progress"]["source_receipts"][0]["native_job_id"] == "610289"
    written = [path.read_bytes() for path in paths]
    result = progress.synchronize(case["root"], write=True)
    assert not result["changed"] and not result["written"]
    assert [path.read_bytes() for path in paths] == written


@pytest.mark.parametrize("field,value", [("job_id", "613510"), ("source_rng_seed", 95018),
                                          ("source_label", "quarter_B"), ("archive", "/other/archive"),
                                          ("source_binding", {"archive": "other"})])
def test_source_receipt_identity_is_protected(receipt_case, field, value):
    case = receipt_case
    receipt = copy.deepcopy(case["source_receipt"])
    receipt["source"][field] = value
    write_json(case["source_path"], receipt)
    with pytest.raises(ValueError, match="SOURCE_IDENTITY_MISMATCH"):
        progress.synchronize(case["root"])


@pytest.mark.parametrize("name", ["delivery.json", "report.json", "analysis/report.json"])
def test_authenticated_source_metadata_change_is_rejected(receipt_case, name):
    case = receipt_case
    path = case["source_archive"] / name
    value = json.loads(path.read_text())
    value["changed"] = True
    write_json(path, value)
    with pytest.raises(ValueError, match="METADATA_BINDING_CHANGED"):
        progress.synchronize(case["root"])


@pytest.mark.parametrize("field,value", [("job_id", "610296_61"), ("source_rng_seed", 95018),
                                          ("readback_group_index", 61), ("readback_group", "B/native_half")])
def test_reader_identity_is_protected(receipt_case, field, value):
    case = receipt_case
    receipt = copy.deepcopy(case["reader_receipt"])
    receipt[field] = value
    write_json(case["reader_path"], receipt)
    with pytest.raises(ValueError, match="READER_IDENTITY_MISMATCH"):
        progress.synchronize(case["root"])


def test_two_source_binding_change_is_rejected_even_with_updated_metadata_hash(receipt_case):
    case = receipt_case
    receipt = copy.deepcopy(case["reader_receipt"])
    receipt["source_costs"]["fine"]["source_binding"]["report_sha256"] = "other"
    path = case["reader_archive"] / "analysis/report.json"
    analysis = json.loads(path.read_text())
    analysis["source_costs"] = receipt["source_costs"]
    receipt["analysis_sha256"] = write_json(path, analysis)
    write_json(case["reader_path"], receipt)
    with pytest.raises(ValueError, match="READER_SOURCE_BINDING_CHANGED"):
        progress.synchronize(case["root"])


def test_duplicate_profile_is_rejected(receipt_case):
    case = receipt_case
    receipt = copy.deepcopy(case["reader_receipt"])
    receipt["paired_profile_keys"] *= 2
    write_json(case["reader_path"], receipt)
    with pytest.raises(ValueError, match="DUPLICATE_PROFILE"):
        progress.synchronize(case["root"])


@pytest.mark.parametrize("kind", ["source", "reader"])
def test_previously_accepted_receipt_cannot_disappear_or_change(receipt_case, kind):
    case = receipt_case
    progress.synchronize(case["root"], write=True)
    path = case[kind + "_path"]
    original = path.read_bytes()
    value = json.loads(original)
    value["authentication_job"] = "changed"
    write_json(path, value)
    with pytest.raises(ValueError, match="ACCEPTED_RECEIPT_CHANGED"):
        progress.synchronize(case["root"])
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ValueError, match="ACCEPTED_RECEIPT_MISSING|READER_SOURCE_NOT_ACCEPTED"):
        progress.synchronize(case["root"])


def test_stale_ledger_and_mirror_conflict_fail_without_writes(receipt_case):
    case = receipt_case
    ledger_path = case["root"] / "state_ledger.yaml"
    ledger_path.write_bytes(case["ledger"].replace(b"sources: 76", b"sources: 75"))
    before = (case["root"] / progress.STATE_PATH).read_bytes()
    with pytest.raises(ValueError, match="STATE_LEDGER_STALE"):
        progress.synchronize(case["root"], write=True)
    assert (case["root"] / progress.STATE_PATH).read_bytes() == before
    ledger_path.write_bytes(case["ledger"])
    mirror = copy.deepcopy(case["branch"])
    mirror["failures"] = []
    write_json(case["root"] / progress.MIRROR_PATH, mirror)
    with pytest.raises(ValueError, match="MIRROR_DISAGREEMENT"):
        progress.synchronize(case["root"], write=True)
    assert (case["root"] / progress.STATE_PATH).read_bytes() == before


def test_stale_snapshot_is_rejected_before_replacement(receipt_case, monkeypatch):
    case = receipt_case
    original_hash = progress.sha256_file
    source_path = case["source_path"]
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if path == source_path:
            calls += 1
            if calls == 3:
                return "changed"
        return original_hash(path)

    monkeypatch.setattr(progress, "sha256_file", changed_hash)
    paths = [case["root"] / name for name in (progress.STATE_PATH, progress.MIRROR_PATH, "state_ledger.yaml")]
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ValueError, match="CONCURRENT_RECEIPT_CHANGE"):
        progress.synchronize(case["root"], write=True)
    assert [path.read_bytes() for path in paths] == before


def test_scientific_gate_remains_closed(receipt_case):
    case = receipt_case
    receipt = copy.deepcopy(case["source_receipt"])
    receipt["scientific_fidelity_qualified"] = True
    write_json(case["source_path"], receipt)
    with pytest.raises(ValueError, match="SCIENTIFIC_GATE_CHANGED"):
        progress.synchronize(case["root"])

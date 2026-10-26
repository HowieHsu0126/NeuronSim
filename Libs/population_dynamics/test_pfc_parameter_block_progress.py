"""Progress updates preserve accepted identities and detect state conflicts."""
import copy
import json
import os
from pathlib import Path

import pytest

from population_dynamics.pfc_parameter_block_progress import (
    BRANCH_PATH, branch_of, replace_state_pair, synchronize_branch,
)


@pytest.fixture
def sync_case():
    sources = [dict(index=i, receipt=f"source_{i}.json", receipt_sha256=f"hash_{i}",
                    native_job_id=str(1000+i), scheduler_seconds=100+i,
                    verified_files=40, verified_bytes=10000,
                    source_report_sha256=f"report_{i}") for i in range(20)]
    readers = [dict(group=i, receipt=f"reader_{i}.json", receipt_sha256=f"reader_hash_{i}",
                    job_id=f"reader_{i}", scheduler_seconds=200+i,
                    complete_development_episodes=10, verified_files=90,
                    delivery_sha256=f"delivery_{i}") for i in range(2)]
    return dict(branch=dict(complete_new_source_episodes=0, complete_new_readback_episodes=0,
                            accepted_new_source_receipts=[], accepted_new_readback_receipts=[]),
                sources=sources, readers=readers)


def accepted(case):
    return synchronize_branch(case["branch"], case["sources"], case["readers"])


def test_sync_preserves_metadata_and_is_idempotent(sync_case):
    case = copy.deepcopy(sync_case)
    branch = case["branch"]
    branch["scheduler_metadata"] = {"job": "pending", "dependency": "unchanged"}
    branch["failures"] = [{"task": 7, "reason": "preserve"}]
    branch["scientific_fidelity_qualified"] = False
    branch["source_probability_law_qualified"] = False
    branch["parameter_domain_frozen"] = False
    branch["production_episodes"] = 0
    before = copy.deepcopy(case)
    updated = accepted(case)
    assert case == before
    assert updated is not branch
    for key in ("scheduler_metadata", "failures", "scientific_fidelity_qualified", "source_probability_law_qualified",
                "parameter_domain_frozen", "production_episodes"):
        assert updated[key] == branch[key]
    assert updated["complete_new_source_episodes"] == 20
    assert updated["complete_new_readback_episodes"] == 20
    assert synchronize_branch(updated, case["sources"], case["readers"]) == updated


@pytest.mark.parametrize("kind", ["sources", "readers"])
def test_sync_rejects_missing_accepted_receipt(sync_case, kind):
    case = copy.deepcopy(sync_case)
    branch = accepted(case)
    case[kind].pop()
    before = copy.deepcopy(branch)
    with pytest.raises(ValueError, match="ACCEPTED_RECEIPT_MISSING"):
        synchronize_branch(branch, case["sources"], case["readers"])
    assert branch == before


@pytest.mark.parametrize("kind,field", [("sources", "native_job_id"), ("sources", "verified_bytes"),
                                         ("sources", "receipt_sha256"), ("readers", "job_id"),
                                         ("readers", "verified_files"), ("readers", "receipt_sha256")])
@pytest.mark.parametrize("operation", ["change", "remove"])
def test_sync_rejects_changed_accepted_fields(sync_case, kind, field, operation):
    case = copy.deepcopy(sync_case)
    branch = accepted(case)
    record = case[kind][0]
    if operation == "change":
        record[field] = record[field] + "changed" if isinstance(record[field], str) else record[field] + 1
    else:
        del record[field]
    before = copy.deepcopy(branch)
    with pytest.raises(ValueError, match="ACCEPTED_RECEIPT_CHANGED"):
        synchronize_branch(branch, case["sources"], case["readers"])
    assert branch == before


def test_sync_adds_hash_and_preserves_management_accounting(sync_case):
    case = copy.deepcopy(sync_case)
    branch = accepted(case)
    del branch["accepted_new_source_receipts"][0]["receipt_sha256"]
    branch["accepted_new_source_receipts"][0]["scheduler_seconds"] = 123
    branch["accepted_new_readback_receipts"][0]["native_job_id"] = "4321"
    case["sources"][0]["scheduler_seconds"] = None
    case["readers"][0]["scheduler_seconds"] = None
    updated = synchronize_branch(branch, case["sources"], case["readers"])
    assert updated["accepted_new_source_receipts"][0]["receipt_sha256"] == case["sources"][0]["receipt_sha256"]
    assert updated["accepted_new_source_receipts"][0]["scheduler_seconds"] == 123
    assert updated["accepted_new_readback_receipts"][0]["native_job_id"] == "4321"
    assert updated["accepted_new_readback_receipts"][0]["scheduler_seconds"] == 200
    assert "receipt_sha256" not in branch["accepted_new_source_receipts"][0]
    assert synchronize_branch(updated, case["sources"], case["readers"]) == updated


@pytest.mark.parametrize("kind", ["sources", "readers"])
def test_sync_rejects_duplicate_logical_records(sync_case, kind):
    case = copy.deepcopy(sync_case)
    case[kind].append(copy.deepcopy(case[kind][0]))
    with pytest.raises(ValueError, match="DUPLICATE_LOGICAL_RECEIPT"):
        accepted(case)


def test_readback_requires_all_ten_source_indices(sync_case):
    case = copy.deepcopy(sync_case)
    case["sources"] = [row for row in case["sources"] if row["index"] != 9]
    with pytest.raises(ValueError, match="COMPLETE_TEN_SOURCE_GROUP_REQUIRED"):
        accepted(case)


def test_rejects_previous_count_disagreement(sync_case):
    case = copy.deepcopy(sync_case)
    case["branch"]["complete_new_source_episodes"] = 1
    with pytest.raises(ValueError, match="PREVIOUS_COUNT_DISAGREEMENT"):
        accepted(case)


def nested(value):
    state = {"unrelated": {"preserve": True}}
    current = state
    for key in BRANCH_PATH[:-1]:
        current[key] = {}
        current = current[key]
    current[BRANCH_PATH[-1]] = value
    return state


def state_files(tmp_path):
    state, mirror = tmp_path / "state.json", tmp_path / "mirror.json"
    original_state = json.dumps(nested({"version": 1})).encode()
    original_mirror = b'{"version": 1}'
    state.write_bytes(original_state)
    mirror.write_bytes(original_mirror)
    return state, mirror, original_state, original_mirror, nested({"version": 2})


def test_replace_pair_success(tmp_path):
    state, mirror, old_state, old_mirror, updated = state_files(tmp_path)
    replace_state_pair(state, mirror, old_state, old_mirror, updated)
    assert json.loads(state.read_bytes()) == updated
    assert branch_of(json.loads(state.read_bytes())) == json.loads(mirror.read_bytes())


def test_replace_pair_rejects_mirror_disagreement(tmp_path):
    state, mirror, old_state, _, updated = state_files(tmp_path)
    divergent = b'{"version": 3}'
    mirror.write_bytes(divergent)
    with pytest.raises(ValueError, match="MIRROR_DISAGREEMENT"):
        replace_state_pair(state, mirror, old_state, divergent, updated)
    assert state.read_bytes() == old_state
    assert mirror.read_bytes() == divergent


@pytest.mark.parametrize("changed_path", ["state", "mirror"])
def test_replace_pair_rejects_concurrent_change(tmp_path, changed_path):
    state, mirror, old_state, old_mirror, updated = state_files(tmp_path)
    target = state if changed_path == "state" else mirror
    target.write_bytes(b'{"newer_scheduler_metadata": true}')
    before = state.read_bytes(), mirror.read_bytes()
    with pytest.raises(ValueError, match="CONCURRENT_STATE_CHANGE"):
        replace_state_pair(state, mirror, old_state, old_mirror, updated)
    assert (state.read_bytes(), mirror.read_bytes()) == before


def test_second_replace_failure_remains_detectable(tmp_path, monkeypatch):
    state, mirror, old_state, old_mirror, updated = state_files(tmp_path)
    real_replace = os.replace
    replaced = []
    def fail_second(src, dst):
        if Path(dst) in (state, mirror):
            replaced.append(Path(dst))
            if len(replaced) == 2:
                raise OSError("injected second replacement failure")
        return real_replace(src, dst)
    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second replacement failure"):
        replace_state_pair(state, mirror, old_state, old_mirror, updated)
    assert len(replaced) == 2
    before = state.read_bytes(), mirror.read_bytes()
    assert branch_of(json.loads(before[0])) != json.loads(before[1])
    with pytest.raises(ValueError, match="MIRROR_DISAGREEMENT"):
        replace_state_pair(state, mirror, before[0], before[1], updated)
    assert (state.read_bytes(), mirror.read_bytes()) == before


def test_recovery_metadata_is_retained(sync_case):
    case = copy.deepcopy(sync_case)
    branch = accepted(case)
    prior = branch["accepted_new_source_receipts"][7]
    prior.update(actual_slurm_array_task="613942_7", logical_task="613345_7",
                 original_failed_raw_job="613873", original_failed_scheduler_seconds=278)
    updated = synchronize_branch(branch, case["sources"], case["readers"])
    for field in ("actual_slurm_array_task", "logical_task", "original_failed_raw_job",
                  "original_failed_scheduler_seconds"):
        assert updated["accepted_new_source_receipts"][7][field] == prior[field]
    assert synchronize_branch(updated, case["sources"], case["readers"]) == updated


def test_complete_cells_wait_for_summary(sync_case):
    case = copy.deepcopy(sync_case)
    case["sources"] = [dict(case["sources"][0], index=i, receipt=f"source_{i}.json") for i in range(100)]
    case["readers"] = [dict(case["readers"][0], group=i, receipt=f"reader_{i}.json") for i in range(10)]
    updated = accepted(case)
    assert updated["status"].startswith("ACTIVE_")
    updated["descriptive_summary_acceptance"] = dict(complete_development_episodes=100,
                                                    parameter_comparison_rows=1800)
    updated = synchronize_branch(updated, case["sources"], case["readers"])
    assert updated["status"] == "COMPLETE_BOUNDED_PARAMETER_BLOCK_DEVELOPMENT"
    assert synchronize_branch(updated, case["sources"], case["readers"]) == updated


def test_summary_rejects_incomplete_cells(sync_case):
    case = copy.deepcopy(sync_case)
    case["branch"]["descriptive_summary_acceptance"] = dict(complete_development_episodes=100,
                                                            parameter_comparison_rows=1800)
    with pytest.raises(ValueError, match="COMPLETE_SUMMARY_REQUIRES_ALL_CELLS"):
        accepted(case)

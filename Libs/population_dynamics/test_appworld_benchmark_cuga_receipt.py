"""CUGA composite receipt tests use synthetic execution evidence only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from .appworld_benchmark_cuga_receipt import compose_cuga_execution, main
from .appworld_benchmark_scoring import _validate_inputs as validate_scoring_inputs
from .test_appworld_benchmark_scoring import scoring_fixture


def evidence():
    native = {"schema": "appworld_native_tool_session_v1", "scope": "development_tools_only",
              "formal_eligible": False, "split": "train", "task_id": "synthetic_task", "seed": 41,
              "run_root": "/native/attempt", "data_root": "/native/data_root",
              "experiment_name": "development", "appworld_version": "0.2.0.dev0",
              "terminal": True, "status": "closed", "native_world_closed": True, "server_exit_code": 0}
    cleanup = {"child_reaped": True, "exit_code": 0, "termination_signals": [],
               "request_outcome_unresolved": False, "late_response": None}
    owner = {"status": "completed", "rpc_outcome_unresolved": False,
             "worker": {"pid": 2, "exit_code": 0, "reaped": True, "termination_signals": []},
             "relay": {"status": "closed", "rpc_outcome_unresolved": False,
                       "native_cleanup": cleanup, "cleanup_error": None},
             "graph_observation": {"task_id": "synthetic_task", "seed": 41, "status": "graph_ended",
                                   "native_cleanup_settled": True, "native_cleanup": deepcopy(cleanup),
                                   "loop": {"status": "graph_ended", "model_usage": {"complete": True}}}}
    return native, owner


def compose(native, owner):
    return compose_cuga_execution(native, owner, native_session_path="/native/native_session.json",
                                  owner_observation_path="/owner/owner_observation.json")


def test_graph_end_is_unscored_and_uses_native_root_not_cuga_root():
    native, owner = evidence()
    result = compose(native, owner)
    assert result["status"] == "completed" and result["terminal"] is True
    assert result["run_root"] == native["run_root"]
    assert result["native_tgc"] is None and result["scoring_status"] == "not_run"
    assert result["usage_complete"] is False and result["formal_eligible"] is False


@pytest.mark.parametrize("state", ["interrupted", "awaiting_external_tools", "unfinished", "errored"])
def test_known_incomplete_graph_retains_scoreable_agent_failure(state):
    native, owner = evidence()
    owner["graph_observation"]["status"] = state
    owner["status"] = "worker_failed"; owner["worker"]["exit_code"] = 1
    assert compose(native, owner)["status"] == "agent_failure"


def test_native_late_response_does_not_clear_native_unknown():
    native, owner = evidence()
    owner["relay"]["native_cleanup"].update(request_outcome_unresolved=True, late_response={"result": True})
    result = compose(native, owner)
    assert result["status"] == "unknown" and result["native_tgc"] is None


def test_fd_unknown_is_separate_from_a_clean_native_close():
    native, owner = evidence(); owner["relay"]["rpc_outcome_unresolved"] = True
    assert compose(native, owner)["status"] == "unknown"


def test_unreaped_process_is_not_terminal():
    native, owner = evidence(); owner["worker"]["reaped"] = False
    result = compose(native, owner)
    assert result["status"] == "unknown" and result["terminal"] is False


def test_disagreeing_graph_cleanup_stays_unknown():
    native, owner = evidence(); owner["graph_observation"]["native_cleanup"]["late_response"] = {"result": 1}
    assert compose(native, owner)["status"] == "unknown"


def test_initialization_error_is_infrastructure_failure():
    native, owner = evidence(); owner["graph_observation"].update(status="errored", loop={})
    assert compose(native, owner)["status"] == "infrastructure_failure"


def preflight_evidence():
    native, owner = evidence()
    owner.update(status="preflight_failed", failure_stage="preflight",
                 owner_error={"type": "FileNotFoundError", "message": "bubblewrap missing"},
                 graph_observation=None)
    owner["worker"].update(pid=None, spawn_attempted=False, reaped=False, exit_code=None)
    owner["relay"].update(status="cancelled", rpc_outcome_unresolved=True, requests=[])
    return native, owner


def test_unstarted_preflight_is_unscored_infrastructure_failure():
    native, owner = preflight_evidence()
    result = compose(native, owner)
    assert result["status"] == "infrastructure_failure" and result["terminal"] is True
    assert result["execution_status_reason"] == "owner_preflight_failed_before_worker_start"
    assert result["rpc_outcome_unresolved"] is False
    assert result["native_tgc"] is None and result["scoring_status"] == "not_run"
    assert result["usage_complete"] is False


@pytest.mark.parametrize("change", ["spawn", "pid", "request", "cleanup", "native", "cancel"])
def test_preflight_requires_all_no_execution_and_cleanup_proofs(change):
    native, owner = preflight_evidence()
    if change == "spawn":
        owner["worker"]["spawn_attempted"] = True
    elif change == "pid":
        owner["worker"]["pid"] = 2
    elif change == "request":
        owner["relay"]["requests"] = [{"operation": "call"}]
    elif change == "cleanup":
        owner["relay"]["native_cleanup"]["request_outcome_unresolved"] = True
    elif change == "native":
        native["native_world_closed"] = False
    else:
        owner["owner_cancelled"] = True
    assert compose(native, owner)["status"] == "unknown"


def test_historical_missing_spawn_proof_remains_unknown():
    native, owner = preflight_evidence()
    owner["worker"].pop("spawn_attempted")
    assert compose(native, owner)["status"] == "unknown"


@pytest.mark.parametrize("field,value", [("split", "test_normal"), ("seed", True), ("formal_eligible", True)])
def test_formal_or_invalid_identity_is_rejected(field, value):
    native, owner = evidence(); native[field] = value
    with pytest.raises(ValueError): compose(native, owner)


def test_graph_native_identity_mismatch_is_rejected():
    native, owner = evidence(); owner["graph_observation"]["seed"] = 42
    with pytest.raises(ValueError, match="identity differs"): compose(native, owner)


def test_cli_preserves_original_receipts_and_refuses_existing_output(tmp_path):
    native, owner = evidence(); native_path = tmp_path/"native.json"; owner_path = tmp_path/"owner.json"
    native_path.write_text(json.dumps(native)); owner_path.write_text(json.dumps(owner))
    output = tmp_path/"execution.json"; args=["--native-session",str(native_path),"--owner-observation",str(owner_path),"--output",str(output)]
    assert main(args)==0
    result=json.loads(output.read_text())
    assert result["native_session_path"]==str(native_path) and result["owner_observation_path"]==str(owner_path)
    assert json.loads(native_path.read_text())==native and json.loads(owner_path.read_text())==owner
    with pytest.raises(FileExistsError): main(args)


def test_composite_is_accepted_by_existing_native_scorer(scoring_fixture):
    native, owner = evidence()
    for key in ("split", "task_id", "seed", "run_root", "data_root", "experiment_name", "appworld_version"):
        native[key] = scoring_fixture.receipt[key]
    owner["graph_observation"].update(task_id=native["task_id"], seed=native["seed"])
    receipt = compose(native, owner)
    validate_scoring_inputs(receipt, scoring_fixture.manifest)
    owner["worker"]["reaped"] = False
    with pytest.raises(ValueError, match="terminal status"):
        validate_scoring_inputs(compose(native, owner), scoring_fixture.manifest)

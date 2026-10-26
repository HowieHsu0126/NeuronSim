"""Contract checks for paired AppWorld accounting, without model execution."""
from copy import deepcopy
import json
from pathlib import Path

import yaml

import pytest

from .appworld_pair_diagnostic import coverage_evidence, summarize_arm, summarize_verification_reply


def fixture(write_ok=False):
    actions = [
        {"actor": "executor", "kind": "tool", "call_id": "executor_login", "app": "spotify", "api": "login", "arguments": {}},
        {"actor": "executor", "kind": "tool", "call_id": "executor_follow", "app": "spotify", "api": "follow_artist", "arguments": {"artist_id": 14}},
        {"actor": "executor", "kind": "message", "recipient": "planner", "text": "Observed authentication result", "share_results": ["executor_login"]},
    ]
    attempts, generations, transport, events = [], [], [], []
    for index, action in enumerate(actions):
        outcome = ({"ok": True, "result": {"access_token": "synthetic"}} if index == 0
                   else {"ok": write_ok, "result": {"message": "fixture"}} if index == 1
                   else {"delivered_to": "planner", "phase": "execution"})
        response = json.dumps(action)
        receipt = {"response": response}
        ack = {"ok": True, "result": outcome}
        transport.append({"request": {**action, "generation": receipt}, "response": ack})
        events.append({"accepted": True, "request": action, "result": outcome, "http_method": "POST"})
        generations.append({"response": response, "reused_prefix_inference": index == 0,
                            "usage": None if index == 0 else {"total_tokens": 8},
                            "inference_seconds": None if index == 0 else 1.25})
        attempts.append({"attempt_index": index, "actor": action["actor"], "status": "ACCEPTED",
                         "host_state": "DISCOVERY" if index == 0 else "EXECUTE",
                         "submitted": True, "acknowledged": True, "controller_request_index": index,
                         "controller_response": ack})
    host = {"generation_attempts": generations, "host_attempts": attempts,
            "recorded_prefix": {"status": "VERIFIED", "checked_generations": 1,
                                "historical_prefix_token_usage": {"total_tokens": 17},
                                "historical_prefix_inference_seconds": 2.0},
            "host_attempt_count": 3, "host_attempt_limit": 3,
            "allocation": {"planner": 1, "executor": 1}, "allocation_acknowledged": True,
            "verification_calls_used": {"planner": 1, "executor": 1},
            "unknown_native_outcome": False, "outstanding_request": None,
            "cost_accounting": {"provider_token_usage": {"total_tokens": 16}, "provider_inference_seconds": 2.5},
            "error": {"error_code": "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"},
            "host_state": "EXECUTE", "native_evaluation": None}
    report = {"host": host, "controller_process": {"request_outcome_unresolved": False}, "cost": {"host_wall_seconds": 4.0}}
    native = {"recorded_native_prefix": {"status": "PASS"}, "cost_accounting": {"native_wall_seconds_sum": 1.0}}
    return report, native, transport, events, {"state": "FAILED", "exit_code": "1:0"}, 1


def test_native_failure_is_separate_from_accepted_write_and_no_evaluation():
    result = summarize_arm(*fixture())
    assert result["accepted_task_write_requests"] == 1
    assert result["native_write_successes"] == 0
    assert result["native_write_failures"] == 1
    assert result["finish_accepted"] is False
    assert result["native_evaluation_status"] == "NOT_RUN"
    assert result["termination"] == "CAP_FAILURE_RESOLVED"
    assert result["new_inference_cost"]["provider_token_usage"]["total_tokens"] == 16
    assert result["historical_prefix_cost"]["host_tokens"]["total_tokens"] == 17
    assert result["explicit_login_result_shares"][0]["call_id"] == "executor_login"


def test_successful_write_does_not_establish_coverage_or_task_success():
    result = summarize_arm(*fixture(write_ok=True))
    assert result["native_write_successes"] == 1
    assert result["coverage"]["status"] == "NOT_ESTABLISHED"
    assert result["native_evaluation_status"] == "NOT_RUN"
    assert result["scientific_control_transfer_qualified"] is False


def test_unknown_request_outcome_takes_precedence_over_cap_failure():
    args = fixture();args[0]["host"]["unknown_native_outcome"] = True
    result = summarize_arm(*args)
    assert result["termination"] == "UNKNOWN_REQUEST_OUTCOME"
    assert result["native_evaluation_status"] == "UNKNOWN"


def test_acknowledgement_tampering_is_rejected():
    args = list(fixture());args[2] = deepcopy(args[2])
    args[2][1]["response"]["result"]["ok"] = True
    with pytest.raises(ValueError, match="HOST_TRANSPORT_ACKNOWLEDGEMENT_MISMATCH"):
        summarize_arm(*args)


def call(query="", count=1):
    return {"app": "spotify", "api": "search_artists", "native_ok": True,
            "call_id": "planner_search", "arguments": {"query": query, "genre": "classical", "min_follower_count": 22, "page_limit": 5},
            "native_result": [{"artist_id": i, "genre": "classical", "follower_count": 22} for i in range(count)]}


def test_keyword_or_full_page_does_not_establish_exhaustion():
    assert coverage_evidence([call(query="classical")])["status"] == "NOT_ESTABLISHED"
    assert coverage_evidence([call(count=5)])["status"] == "NOT_ESTABLISHED"
    assert coverage_evidence([call()])["status"] == "ESTABLISHED_SINGLE_PAGE_EXHAUSTION"


def test_contradictory_filtered_result_is_rejected():
    value = call();value["native_result"][0]["genre"] = "rock"
    with pytest.raises(ValueError, match="FILTERED_SEARCH_RESULT_CONTRADICTS_TASK_FILTER"):
        coverage_evidence([value])


def test_context_failure_without_response_preserves_attempt_and_missing_metering():
    args = list(fixture());host = args[0]["host"]
    host["error"] = {"error_code": "LOCAL_HOST_GENERATION_FAILED"}
    host["host_attempts"][-1].update(status="GENERATION_FAILED", submitted=False, acknowledged=False,
                                    error_code="LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED")
    host["generation_attempts"][-1] = {"status": "FAILED", "error_code": "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED",
                                        "usage": None, "inference_seconds": None,
                                        "prompt_tokens_before_generation": 66440}
    host["cost_accounting"] = {"provider_token_usage": {"total_tokens": 8}, "provider_inference_seconds": 1.25}
    args[2] = args[2][:-1];args[3] = args[3][:-1]
    result = summarize_arm(*args)
    assert result["termination"] == "CONTEXT_LIMIT_FAILURE_RESOLVED"
    assert result["new_generation_attempts"] == 2
    assert result["new_measured_generations"] == 1
    assert result["new_generation_failures"][0]["prompt_tokens_before_generation"] == 66440
    assert result["explicit_login_result_shares"] == []


def reply_fixture():
    """Extend the existing native/transport fixture with a ten-step synthetic prefix."""
    report, native, _, _, scheduler, _ = fixture()
    protocol = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_verification_reply.yaml").read_text())
    host = report["host"]
    host.update(model_revision=protocol["model_revision"], host_attempt_limit=30,
                verification_message_replies=True, evidence_table_after_prefix=True,
                observation_history_encoding="actor_observation_append_v1",
                verification_reply_slots=1, verification_reply_attempts=1,
                verification_replies_completed=1, pending_verification_reply=None,
                error={"error_code": "FIXTURE_TERMINAL_STOP"})
    host["recorded_prefix"]["checked_generations"] = 10
    host["generation_attempts"], host["host_attempts"] = [], []
    transport, events = [], []
    script = [("DISCOVERY", {"actor": "executor", "kind": "tool", "call_id": "executor_login_" + str(i),
                              "app": "spotify", "api": "login", "arguments": {}}) for i in range(10)]
    script += [
        ("VERIFY", {"actor": "planner", "kind": "tool", "call_id": "planner_verify", "app": "spotify", "api": "read", "arguments": {}}),
        ("VERIFY", {"actor": "executor", "kind": "message", "recipient": "planner", "text": "peer request", "share_results": []}),
        ("VERIFY_REPLY", {"actor": "planner", "kind": "message", "recipient": "executor", "text": "peer reply", "share_results": ["planner_verify"]}),
        ("VERIFY", {"actor": "executor", "kind": "tool", "call_id": "executor_verify", "app": "spotify", "api": "read", "arguments": {}}),
        ("PLANNER_HANDOFF", {"actor": "planner", "kind": "message", "recipient": "executor", "text": "handoff", "share_results": ["planner_verify"]}),
        ("EXECUTE", {"actor": "executor", "kind": "tool", "call_id": "executor_follow", "app": "spotify", "api": "follow_artist", "arguments": {"artist_id": 14}}),
    ]
    for index, (state, action) in enumerate(script):
        if action["kind"] == "message":
            outcome = {"delivered_to": action["recipient"], "phase": "verification" if state in {"VERIFY", "VERIFY_REPLY"} else "execution"}
        else:
            outcome = {"ok": action["api"] != "follow_artist", "result": {"observed": "synthetic"}}
        ack, response = {"ok": True, "result": outcome}, json.dumps(action)
        generation = {"response": response, "model_revision": protocol["model_revision"],
                      "reused_prefix_inference": index < 10, "status": "GENERATED",
                      "usage": None if index < 10 else {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                      "inference_seconds": None if index < 10 else 1.25}
        host["generation_attempts"].append(generation)
        host["host_attempts"].append({"attempt_index": index, "actor": action["actor"], "status": "ACCEPTED",
                                      "host_state": state, "submitted": True, "acknowledged": True,
                                      "controller_request_index": index, "controller_response": ack})
        transport.append({"request": {**action, "generation": {"response": response}}, "response": deepcopy(ack)})
        events.append({"accepted": True, "request": action, "result": outcome,
                       "http_method": "GET" if action.get("api") == "read" else "POST"})
    host["host_attempt_count"] = len(script)
    refresh_reply_cost(host)
    return report, native, transport, events, scheduler, protocol


def refresh_reply_cost(host):
    rows = [row for row in host["generation_attempts"][10:] if row.get("usage") is not None]
    host["cost_accounting"] = {"provider_token_usage": {"total_tokens": sum(row["usage"]["total_tokens"] for row in rows)},
                               "provider_inference_seconds": sum(row["inference_seconds"] for row in rows)}


def reply_terminal_fixture(kind):
    args = list(reply_fixture())
    host = args[0]["host"]
    host["generation_attempts"] = host["generation_attempts"][:13]
    host["host_attempts"] = host["host_attempts"][:13]
    args[2], args[3] = args[2][:13], args[3][:13]
    host.update(host_attempt_count=13, host_state="VERIFY_REPLY", verification_replies_completed=0,
                verification_calls_used={"planner": 1, "executor": 0},
                pending_verification_reply={"sender": "executor", "recipient": "planner", "trigger_attempt_index": 11})
    if kind == "context_refusal":
        host["host_attempts"][-1].update(status="GENERATION_FAILED", submitted=False, acknowledged=False,
                                         error_code="LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED")
        host["generation_attempts"][-1] = {"status": "FAILED", "model_revision": host["model_revision"],
                                           "error_code": "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED",
                                           "usage": None, "inference_seconds": None,
                                           "prompt_tokens_before_generation": 70000}
        host["error"] = {"error_code": "LOCAL_HOST_GENERATION_FAILED"}
        args[2], args[3] = args[2][:-1], args[3][:-1]
    else:
        host.update(unknown_native_outcome=True, outstanding_request={"kind": "message" if kind == "unknown_reply" else "view"},
                    error={"error_code": "LOCAL_HOST_CHANNEL_OUTCOME_UNKNOWN"})
        if kind == "unknown_reply":
            host["host_attempts"][-1].update(status="OUTCOME_UNKNOWN", acknowledged=False)
    refresh_reply_cost(host)
    return args


def test_reply_summary_accounts_actions_costs_and_keeps_native_failure_separate():
    args = reply_fixture()
    before = deepcopy(args[0])
    result = summarize_verification_reply(*args)
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["verification_reply"]["scheduled_slots"] == result["verification_reply"]["attempts"] == 1
    assert result["verification_reply"]["accepted_replies"] == result["verification_reply"]["completed_replies"] == 1
    assert result["verification_reply"]["returned_to_verification_count"] == 1
    assert result["all_verification_complete"] is True
    assert result["native_write_failures"] == 1 and result["native_evaluation_status"] == "NOT_RUN"
    assert result["new_generation_attempts"] == 6
    assert result["actions"][12]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert result["actions"][12]["inference_seconds"] == 1.25
    assert result["scientific_control_transfer_qualified"] is False and result["causal_benefit_qualified"] is False
    assert args[0] == before
    assert "verification_reply" not in summarize_arm(*fixture())


def test_unknown_reply_preserves_pending_and_does_not_count_receipt_as_acceptance():
    result = summarize_verification_reply(*reply_terminal_fixture("unknown_reply"))
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["termination"] == "UNKNOWN_REQUEST_OUTCOME"
    assert result["verification_reply"]["attempts"] == 1 and result["verification_reply"]["accepted_replies"] == 0
    assert result["verification_reply"]["pending"]["trigger_attempt_index"] == 11
    assert result["verification_reply"]["returned_to_verification_count"] == 0
    assert result["all_verification_complete"] is False


def test_accepted_reply_followed_by_unknown_view_preserves_incomplete_host_counters():
    result = summarize_verification_reply(*reply_terminal_fixture("unknown_view"))
    assert result["accounting_complete"] is False
    assert result["termination"] == "UNKNOWN_REQUEST_OUTCOME"
    assert result["host_error"]["error_code"] == "LOCAL_HOST_CHANNEL_OUTCOME_UNKNOWN"
    assert result["verification_reply"]["accepted_replies"] == 1
    assert result["verification_reply"]["completed_replies"] == 0
    assert result["verification_reply"]["pending"]["trigger_attempt_index"] == 11
    assert {row["code"] for row in result["accounting_issues"]} >= {"REPLY_COUNTER_MISMATCH", "PENDING_REPLY_LEDGER_MISMATCH"}


def test_context_refusal_is_retained_without_claiming_executed_token_excess():
    result = summarize_verification_reply(*reply_terminal_fixture("context_refusal"))
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["termination"] == "CONTEXT_LIMIT_FAILURE_RESOLVED"
    assert result["new_generation_failures"][0]["prompt_tokens_before_generation"] == 70000
    assert result["actions"][-1]["usage"] is None
    assert result["new_measured_generations"] == 2
    assert result["verification_reply"]["accepted_replies"] == 0


def test_rejected_reply_consumes_opportunity_and_retains_failure_after_verification_resumes():
    args = list(reply_fixture());host = args[0]["host"]
    host["verification_replies_completed"] = 0
    ack = {"ok": False, "error": "CANNOT_SHARE_UNSEEN_RESULT"}
    host["host_attempts"][12].update(status="CONTROLLER_REJECTED", controller_response=ack)
    args[2][12]["response"] = deepcopy(ack)
    args[3][12]["accepted"] = False
    result = summarize_verification_reply(*args)
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["verification_reply"]["attempts"] == 1 and result["verification_reply"]["accepted_replies"] == 0
    assert result["verification_reply"]["returned_to_verification_count"] == 1
    assert result["new_attempt_status_counts"]["CONTROLLER_REJECTED"] == 1
    assert result["all_verification_complete"] is True


@pytest.mark.parametrize("key,value", [("schema", "wrong"), ("maximum_new_attempts", 21),
                                       ("maximum_total_attempts", 31), ("maximum_new_executions", 2),
                                       ("allocation", {"planner": 2, "executor": 0}),
                                       ("reply_generation_attempts_per_delivered_message", 2)])
def test_reply_protocol_change_is_rejected(key, value):
    args = list(reply_fixture());args[-1][key] = value
    with pytest.raises(ValueError, match="FROZEN_VERIFICATION_REPLY_PROTOCOL_REQUIRED"):
        summarize_verification_reply(*args)


def test_reply_host_model_or_disabled_intervention_cannot_enter_summary():
    args = list(reply_fixture());args[0]["host"]["model_revision"] = "different/model"
    with pytest.raises(ValueError, match="FROZEN_VERIFICATION_REPLY_MODEL_MISMATCH"):
        summarize_verification_reply(*args)
    args = list(reply_fixture());args[0]["host"]["verification_message_replies"] = False
    with pytest.raises(ValueError, match="VERIFICATION_REPLY_HOST_INTERVENTION_NOT_ENABLED"):
        summarize_verification_reply(*args)


@pytest.mark.parametrize("field,value", [("prompt_tokens", 65537), ("completion_tokens", 513)])
def test_executed_generation_token_limits_are_checked_per_attempt(field, value):
    args = list(reply_fixture());host = args[0]["host"]
    usage = host["generation_attempts"][12]["usage"]
    usage[field] = value;usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    refresh_reply_cost(host)
    result = summarize_verification_reply(*args)
    assert result["accounting_complete"] is False
    assert "EXECUTED_GENERATION_TOKEN_LIMIT_EXCEEDED" in {row["code"] for row in result["accounting_issues"]}
    assert result["actions"][12]["usage"][field] == value


def test_attempt_quota_and_reply_counter_mismatches_are_explicit_without_repair():
    args = list(reply_fixture());host = args[0]["host"]
    host.update(host_attempt_limit=31, verification_reply_slots=2, verification_calls_used={"planner": 2, "executor": 1})
    result = summarize_verification_reply(*args)
    assert result["accounting_complete"] is False
    codes = {row["code"] for row in result["accounting_issues"]}
    assert codes >= {"GENERATION_ATTEMPT_BUDGET_MISMATCH", "VERIFICATION_QUOTA_EXCEEDED_OR_INVALID", "REPLY_COUNTER_MISMATCH"}
    assert result["verification_reply"]["scheduled_slots"] == 2
    assert len(result["verification_reply"]["eligible_messages"]) == 1
    assert result["verification_calls_used"]["planner"] == 2
    assert result["all_verification_complete"] is False


def test_historical_action_comparison_stops_before_first_reply_and_records_first_difference():
    args = reply_fixture();reference = deepcopy(args[0])
    reference["host"]["generation_attempts"][12]["response"] = json.dumps({"kind": "different_post_intervention_action"})
    result = summarize_verification_reply(*args, reference_report=reference)
    comparison = result["historical_reference_action_comparison"]
    assert comparison["status"] == "MATCH_BEFORE_FIRST_REPLY" and comparison["compared_new_actions"] == 2
    assert comparison["first_difference"] is None and comparison["causal_benefit_qualified"] is False
    reference["host"]["generation_attempts"][11]["response"] = json.dumps({"kind": "different_trigger_action"})
    result = summarize_verification_reply(*args, reference_report=reference)
    comparison = result["historical_reference_action_comparison"]
    assert comparison["status"] == "PRE_REPLY_ACTION_DIFFERENCE"
    assert comparison["first_difference"]["attempt_index"] == 11
    assert result["accounting_complete"] is True and result["formal_model_comparison_qualified"] is False

"""Descriptive accounting for the single frozen AppWorld task-prefix pair."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file

ARMS = ("original_view", "actor_local_evidence_table")
TARGET_INSTRUCTION = "Follow all the classical artists on Spotify that have at least 22 followers."


def coverage_evidence(calls: list[dict]) -> dict:
    """Check source-supported single-page exhaustion; retain all other evidence.

    Multi-page order is not guaranteed by the installed SQL source. Those
    observations remain available for explicit assessment, never silently pass.
    """
    observations = []
    witnesses = []
    for call in calls:
        if (call.get("app"), call.get("api"), call.get("native_ok")) != ("spotify", "search_artists", True):
            continue
        args, records = call["arguments"], call["native_result"]
        if not isinstance(records, list):
            raise ValueError("SEARCH_RESULT_NOT_A_LIST")
        observed = {key: args.get(key) for key in (
            "query", "genre", "min_follower_count", "max_follower_count",
            "page_index", "page_limit", "sort_by")}
        observed.update({"call_id": call["call_id"], "returned_records": len(records),
                         "rows": [{key: row.get(key) for key in
                                   ("artist_id", "genre", "follower_count")} for row in records]})
        observations.append(observed)
        # Exact public task filter; no answer, database or evaluation access.
        exact_filter = (isinstance(args.get("query", ""), str)
                        and not args.get("query", "").strip()
                        and str(args.get("genre", "")).casefold() == "classical"
                        and args.get("min_follower_count", 0) == 22
                        and args.get("max_follower_count") in (None, sys.maxsize))
        limit, index = args.get("page_limit", 5), args.get("page_index", 0)
        if exact_filter and type(limit) is int and 1 <= limit <= 20 and index == 0 and len(records) < limit:
            if any(str(row.get("genre", "")).casefold() != "classical"
                   or not isinstance(row.get("follower_count"), int)
                   or row["follower_count"] < 22 for row in records):
                raise ValueError("FILTERED_SEARCH_RESULT_CONTRADICTS_TASK_FILTER")
            witnesses.append(call["call_id"])
    return {"status": "ESTABLISHED_SINGLE_PAGE_EXHAUSTION" if witnesses else "NOT_ESTABLISHED",
            "witness_call_ids": witnesses, "observations": observations,
            "rule": "empty query, exact genre/follower task filter, page0 shorter than source page limit",
            "multipage_limit": "requires separate evidence of stable membership and ordering"}


def summarize_arm(report: dict, native: dict, transport: list[dict],
                  events: list[dict], scheduler: dict, prefix_attempts: int, *,
                  allow_partial_inference_measurements: bool = False) -> dict:
    host = report["host"]
    generations, attempts = host["generation_attempts"], host["host_attempts"]
    prefix = host["recorded_prefix"]
    if prefix["status"] != "VERIFIED" or native["recorded_native_prefix"]["status"] != "PASS":
        raise ValueError("VERIFIED_NATIVE_AND_HOST_PREFIX_REQUIRED")
    if prefix["checked_generations"] != prefix_attempts:
        raise ValueError("PREFIX_ATTEMPT_COUNT_MISMATCH")
    if len(attempts) != len(generations) or len(attempts) != host["host_attempt_count"]:
        raise ValueError("ATTEMPT_LEDGER_MISMATCH")
    if host["allocation"] != {"planner": 1, "executor": 1}:
        raise ValueError("FROZEN_ALLOCATION_MISMATCH")
    unresolved = (host["unknown_native_outcome"] or host["outstanding_request"] is not None
                  or report["controller_process"]["request_outcome_unresolved"])
    calls, actions = [], []
    login_ids, login_shares = set(), []
    api_events = {row["request"]["call_id"]: row for row in events
                  if row.get("accepted") is True and row.get("request", {}).get("kind") == "tool"}
    for attempt, generation in zip(attempts, generations):
        index = attempt["attempt_index"]
        action = None
        if "response" not in generation:
            if attempt["status"] != "GENERATION_FAILED" or attempt["submitted"] or generation.get("status") != "FAILED":
                raise ValueError("MISSING_GENERATION_RESPONSE_WITHOUT_FAILURE_RECEIPT")
        else:
            try:
                action = json.loads(generation["response"])
            except json.JSONDecodeError:
                if attempt["submitted"]:
                    raise ValueError("SUBMITTED_MALFORMED_GENERATION")
        if not isinstance(action, dict):
            action = {}
        entry = {"attempt_index": index, "one_based_attempt": index + 1,
                 "is_reused_prefix": index < prefix_attempts, "actor": attempt["actor"],
                 "status": attempt["status"], "host_state": attempt["host_state"],
                 "kind": action.get("kind"), "call_id": action.get("call_id"),
                 "app": action.get("app"), "api": action.get("api"),
                 "recipient": action.get("recipient"), "share_results": action.get("share_results", []),
                 "error": attempt.get("error_code")}
        if attempt["acknowledged"]:
            item = transport[attempt["controller_request_index"]]
            if (item["response"] != attempt["controller_response"]
                    or item["request"]["generation"]["response"] != generation["response"]):
                raise ValueError("HOST_TRANSPORT_ACKNOWLEDGEMENT_MISMATCH")
            entry["error"] = item["response"].get("error")
        if attempt["status"] == "ACCEPTED" and action.get("kind") == "tool":
            event = api_events[action["call_id"]]
            outcome = event["result"]
            if outcome != attempt["controller_response"]["result"]:
                raise ValueError("HOST_NATIVE_OUTCOME_MISMATCH")
            call = {**entry, "http_method": event["http_method"],
                    "arguments": action["arguments"], "native_ok": outcome.get("ok"),
                    "native_result": outcome.get("result"),
                    "native_error": outcome.get("error"),
                    "is_task_write": event["http_method"] != "GET" and action["api"] != "login"}
            calls.append(call)
            if action["api"] == "login" and outcome.get("ok") is True:
                login_ids.add(action["call_id"])
        if attempt["status"] == "ACCEPTED" and action.get("kind") in {"message", "proposal"}:
            for call_id in action.get("share_results", []):
                if call_id in login_ids:
                    login_shares.append({"attempt_index": index, "actor": attempt["actor"],
                                         "recipient": action["recipient"], "call_id": call_id})
        actions.append(entry)
    new_generations = generations[prefix_attempts:]
    if any(g.get("reused_prefix_inference") is not True for g in generations[:prefix_attempts]):
        raise ValueError("PREFIX_COST_PROVENANCE_MISSING")
    if any(g.get("reused_prefix_inference") is True for g in new_generations):
        raise ValueError("NEW_INFERENCE_MISLABELLED_AS_PREFIX")
    measured = [g for g in new_generations if g.get("usage") is not None and g.get("inference_seconds") is not None]
    costs = host["cost_accounting"]
    if allow_partial_inference_measurements:
        # A failed generate() may have elapsed inference time before token counts
        # exist. Preserve its actual metering rather than dropping that interval.
        token_rows = [g for g in new_generations if g.get("usage") is not None]
        time_rows = [g for g in new_generations if g.get("inference_seconds") is not None]
        tokens = sum(g["usage"]["total_tokens"] for g in token_rows) if token_rows else None
        seconds = sum(g["inference_seconds"] for g in time_rows) if time_rows else None
        recorded_tokens = (costs.get("provider_token_usage") or {}).get("total_tokens")
        recorded_seconds = costs.get("provider_inference_seconds")
        if (tokens != recorded_tokens or (seconds is None) != (recorded_seconds is None)
                or (seconds is not None and abs(seconds - recorded_seconds) > 1e-8)):
            raise ValueError("NEW_INFERENCE_TOTAL_MISMATCH")
    elif measured:
        tokens = sum(g["usage"]["total_tokens"] for g in measured)
        seconds = sum(g["inference_seconds"] for g in measured)
        if tokens != costs["provider_token_usage"]["total_tokens"] or abs(seconds - costs["provider_inference_seconds"]) > 1e-8:
            raise ValueError("NEW_INFERENCE_TOTAL_MISMATCH")
    writes = [call for call in calls if call["is_task_write"]]
    error = (host.get("error") or {}).get("error_code")
    if unresolved:
        termination = "UNKNOWN_REQUEST_OUTCOME"
    elif error == "LOCAL_HOST_GENERATION_FAILED" and attempts[-1].get("error_code") == "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED":
        termination = "CONTEXT_LIMIT_FAILURE_RESOLVED"
    elif error == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED" and len(attempts) == host["host_attempt_limit"]:
        termination = "CAP_FAILURE_RESOLVED"
    elif host.get("native_evaluation") is not None:
        termination = "NATIVE_EVALUATION_REPORTED"
    else:
        termination = "OTHER_TERMINATION_REQUIRES_REVIEW"
    evaluation = host.get("native_evaluation")
    new_statuses = Counter(row["status"] for row in actions[prefix_attempts:])
    return {
        "termination": termination, "scheduler": scheduler, "host_error": host.get("error"),
        "host_state": host["host_state"], "unresolved_request_outcome": bool(unresolved),
        "prefix_validation_status": "VERIFIED", "logical_generation_attempts": len(attempts),
        "reused_prefix_generations": prefix_attempts, "new_generation_attempts": len(new_generations),
        "new_attempt_status_counts": dict(new_statuses),
        "new_measured_generations": len(measured),
        "new_generation_failures": [{"attempt_index": a["attempt_index"],
                                     "error_code": g.get("error_code"),
                                     "prompt_tokens_before_generation": g.get("prompt_tokens_before_generation"),
                                     "host_attempt_wall_seconds": g.get("host_attempt_wall_seconds")}
                                    for a, g in zip(attempts[prefix_attempts:], new_generations)
                                    if a["status"] == "GENERATION_FAILED"],
        "new_submitted_generations": sum(a["submitted"] is True for a in attempts[prefix_attempts:]),
        "new_acknowledged_generations": sum(a["acknowledged"] is True for a in attempts[prefix_attempts:]),
        "allocation_acknowledged": host["allocation_acknowledged"],
        "verification_calls_used": host["verification_calls_used"],
        "accepted_proposal_attempt_indices": [a["attempt_index"] for a in actions if a["kind"] == "proposal" and a["status"] == "ACCEPTED"],
        "planner_handoff_completed": any(a["host_state"] == "PLANNER_HANDOFF" and a["status"] == "ACCEPTED" for a in actions),
        "successful_login_call_ids": sorted(login_ids), "explicit_login_result_shares": login_shares,
        "accepted_task_write_requests": len(writes),
        "native_write_successes": sum(c["native_ok"] is True for c in writes),
        "native_write_failures": sum(c["native_ok"] is False for c in writes),
        "native_write_unknowns": sum(type(c["native_ok"]) is not bool for c in writes),
        "task_write_outcomes": [{k: c[k] for k in ("attempt_index", "actor", "call_id", "app", "api", "native_ok", "native_error")} for c in writes],
        "coverage": coverage_evidence(calls),
        "finish_accepted": any(a["kind"] == "finish" and a["status"] == "ACCEPTED" for a in actions),
        "native_evaluation_status": "NOT_RUN" if evaluation is None and not unresolved else "UNKNOWN" if evaluation is None else "REPORTED",
        "native_evaluation": evaluation,
        "historical_prefix_cost": {"host_tokens": prefix["historical_prefix_token_usage"],
                                    "host_inference_seconds": prefix["historical_prefix_inference_seconds"],
                                    "is_new_expenditure": False},
        "new_inference_cost": costs, "new_host_cost": report["cost"],
        "new_native_cost": native["cost_accounting"],
        "model_load_wall_seconds": report.get("model_load_attempt_wall_seconds"),
        "actions": actions, "scientific_control_transfer_qualified": False,
        "formal_model_comparison_qualified": False,
    }


def summarize_verification_reply(report: dict, native: dict, transport: list[dict],
                                 events: list[dict], scheduler: dict, protocol: dict,
                                 reference_report: dict | None = None) -> dict:
    """Account for one exposed-task coordination experiment, retaining failures."""
    frozen = {
        "schema": "appworld_verification_message_reply_development_v1",
        "model_revision": "Qwen/Qwen3-Coder-30B-A3B-Instruct@b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        "prefix_generation_attempts": 10, "maximum_new_attempts": 20,
        "maximum_total_attempts": 30, "maximum_new_executions": 1,
        "allocation": {"planner": 1, "executor": 1},
        "verification_message_replies": True, "reply_generation_attempts_per_delivered_message": 1,
        "lossless_observation_history": True, "evidence_table_after_prefix": True,
        "max_context_tokens": 65536, "max_new_tokens": 512,
    }
    if any(type(protocol.get(key)) is not type(value) or protocol[key] != value for key, value in frozen.items()):
        raise ValueError("FROZEN_VERIFICATION_REPLY_PROTOCOL_REQUIRED")
    if any(type(value) is not int for value in protocol["allocation"].values()):
        raise ValueError("FROZEN_VERIFICATION_REPLY_PROTOCOL_REQUIRED")
    host = report["host"]
    if host.get("model_revision") != protocol["model_revision"]:
        raise ValueError("FROZEN_VERIFICATION_REPLY_MODEL_MISMATCH")
    if (host.get("verification_message_replies") is not True
            or host.get("evidence_table_after_prefix") is not True
            or host.get("observation_history_encoding") != "actor_observation_append_v1"):
        raise ValueError("VERIFICATION_REPLY_HOST_INTERVENTION_NOT_ENABLED")
    result = summarize_arm(report, native, transport, events, scheduler, prefix_attempts=10)
    attempts, generations, actions = host["host_attempts"], host["generation_attempts"], result["actions"]
    issues, eligible, replies = [], [], []
    verification_calls = {"planner": 0, "executor": 0}
    api_events = {row["request"]["call_id"]: row for row in events
                  if row.get("accepted") is True and row.get("request", {}).get("kind") == "tool"}
    expected_pending = None
    unmetered = []
    for index, (entry, attempt, generation) in enumerate(zip(actions, attempts, generations)):
        entry.update(usage=generation.get("usage"), inference_seconds=generation.get("inference_seconds"))
        if attempt["attempt_index"] != index:
            issues.append({"code": "ATTEMPT_INDEX_MISMATCH", "attempt_index": index})
        if generation.get("model_revision") != protocol["model_revision"]:
            issues.append({"code": "GENERATION_MODEL_MISMATCH", "attempt_index": index})
        if index < 10:
            if entry["host_state"] == "VERIFY_REPLY":
                issues.append({"code": "REPLY_BEFORE_VERIFIED_PREFIX", "attempt_index": index})
            continue
        usage = generation.get("usage")
        if usage is None:
            # Refused preparation is an attempted generation, not executed inference.
            if generation.get("status") != "FAILED" or "response" in generation:
                unmetered.append(index)
        elif (not isinstance(usage, dict)
              or any(type(usage.get(key)) is not int or usage[key] < 0
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens"))):
            issues.append({"code": "GENERATION_TOKEN_USAGE_INVALID", "attempt_index": index})
        else:
            if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
                issues.append({"code": "GENERATION_TOKEN_TOTAL_MISMATCH", "attempt_index": index})
            if usage["prompt_tokens"] > protocol["max_context_tokens"] or usage["completion_tokens"] > protocol["max_new_tokens"]:
                issues.append({"code": "EXECUTED_GENERATION_TOKEN_LIMIT_EXCEEDED", "attempt_index": index})
        response = attempt.get("controller_response") or {}
        delivered = response.get("result") or {}
        if entry["host_state"] == "VERIFY" and entry["kind"] == "message" and entry["status"] == "ACCEPTED":
            if (entry["recipient"] in verification_calls and entry["recipient"] != entry["actor"]
                    and isinstance(delivered, dict) and delivered.get("delivered_to") == entry["recipient"]):
                if expected_pending is not None:
                    issues.append({"code": "REPLY_OPPORTUNITY_NOT_CONSUMED", "attempt_index": index})
                expected_pending = {"sender": entry["actor"], "recipient": entry["recipient"], "trigger_attempt_index": index}
                eligible.append(dict(expected_pending))
            else:
                issues.append({"code": "VERIFICATION_MESSAGE_DELIVERY_UNCONFIRMED", "attempt_index": index})
        if entry["host_state"] == "VERIFY_REPLY":
            accepted = entry["status"] == "ACCEPTED"
            next_state = actions[index + 1]["host_state"] if index + 1 < len(actions) else host["host_state"]
            replies.append({"attempt_index": index, "actor": entry["actor"], "status": entry["status"],
                            "trigger_attempt_index": None if expected_pending is None else expected_pending["trigger_attempt_index"],
                            "accepted": accepted, "next_generation_or_terminal_host_state": next_state,
                            "returned_to_verification": next_state == "VERIFY"})
            if expected_pending is None or entry["actor"] != expected_pending["recipient"]:
                issues.append({"code": "REPLY_WITHOUT_MATCHING_OPPORTUNITY", "attempt_index": index})
            if accepted and (expected_pending is None or entry["kind"] != "message"
                             or entry["recipient"] != expected_pending["sender"]
                             or not isinstance(delivered, dict) or delivered.get("delivered_to") != entry["recipient"]):
                issues.append({"code": "ACCEPTED_REPLY_NOT_DELIVERED_TO_SENDER", "attempt_index": index})
            if entry["status"] in {"ACCEPTED", "HOST_REJECTED", "CONTROLLER_REJECTED"}:
                expected_pending = None
                if next_state != "VERIFY":
                    issues.append({"code": "REPLY_DID_NOT_RETURN_TO_VERIFICATION", "attempt_index": index})
        if entry["kind"] == "tool" and entry["status"] == "ACCEPTED" and entry["host_state"] in {"VERIFY", "VERIFY_REPLY"}:
            verification_calls[entry["actor"]] += 1
            if entry["host_state"] == "VERIFY_REPLY" or api_events[entry["call_id"]]["http_method"] != "GET":
                issues.append({"code": "VERIFICATION_TOOL_PERMISSION_VIOLATION", "attempt_index": index})
    if unmetered:
        issues.append({"code": "EXECUTED_GENERATION_TOKEN_USAGE_UNAVAILABLE", "attempt_indices": unmetered})
    if (host["host_attempt_limit"] != 30 or result["logical_generation_attempts"] > 30
            or result["new_generation_attempts"] > 20):
        issues.append({"code": "GENERATION_ATTEMPT_BUDGET_MISMATCH"})
    for actor, count in verification_calls.items():
        recorded = host["verification_calls_used"].get(actor)
        if type(recorded) is not int or not 0 <= recorded <= 1 or count > 1:
            issues.append({"code": "VERIFICATION_QUOTA_EXCEEDED_OR_INVALID", "actor": actor})
        if recorded != count:
            issues.append({"code": "VERIFICATION_CALL_COUNTER_MISMATCH", "actor": actor,
                           "recorded": recorded, "accepted_calls": count})
    accepted_replies = sum(row["accepted"] for row in replies)
    for key, expected in (("verification_reply_slots", len(eligible)),
                          ("verification_reply_attempts", len(replies)),
                          ("verification_replies_completed", accepted_replies)):
        if type(host.get(key)) is not int or host[key] != expected:
            issues.append({"code": "REPLY_COUNTER_MISMATCH", "field": key,
                           "recorded": host.get(key), "ledger_count": expected})
    if host.get("pending_verification_reply") != expected_pending:
        issues.append({"code": "PENDING_REPLY_LEDGER_MISMATCH", "recorded": host.get("pending_verification_reply"),
                       "ledger_pending": expected_pending})
    comparison = {"status": "NOT_SUPPLIED", "reference_role": "historical descriptive context only",
                  "compared_new_actions": 0, "first_difference": None, "causal_benefit_qualified": False}
    if reference_report is not None:
        historical = reference_report.get("host", reference_report)["generation_attempts"]
        stop = next((row["attempt_index"] for row in replies), len(generations))
        comparison.update(status="MATCH_BEFORE_FIRST_REPLY" if replies else "MATCH_WITHOUT_REPLY_OBSERVED",
                          comparison_start_attempt_index=10, comparison_stop_attempt_index_exclusive=stop)
        for index in range(10, stop):
            if index >= len(historical):
                comparison["first_difference"] = {"attempt_index": index, "reason": "REFERENCE_ACTION_MISSING"}
                break
            values = []
            for generation in (generations[index], historical[index]):
                raw = generation.get("response")
                try:
                    values.append(("json", json.loads(raw)))
                except (TypeError, json.JSONDecodeError):
                    values.append(("unparsed_or_missing", raw))
            comparison["compared_new_actions"] += 1
            if values[0] != values[1]:
                comparison["first_difference"] = {"attempt_index": index, "reason": "GENERATED_ACTION_DIFFERS",
                                                   "candidate_response": generations[index].get("response"),
                                                   "reference_response": historical[index].get("response")}
                break
        if comparison["first_difference"] is not None:
            comparison["status"] = "PRE_REPLY_ACTION_DIFFERENCE"
    result.update(
        schema=protocol["schema"],
        status="VERIFICATION_REPLY_ACCOUNTING_REQUIRES_REVIEW" if issues else "COMPLETE_DESCRIPTIVE_VERIFICATION_REPLY_ACCOUNTING",
        accounting_complete=not issues, accounting_issues=issues, new_executions=1,
        model_revision=host["model_revision"], protocol_limits={key: value for key, value in frozen.items() if key != "schema"},
        verification_reply={"enabled": True, "eligible_messages": eligible, "replies": replies,
                            "scheduled_slots": host.get("verification_reply_slots"),
                            "attempts": host.get("verification_reply_attempts"), "accepted_replies": accepted_replies,
                            "completed_replies": host.get("verification_replies_completed"),
                            "pending": host.get("pending_verification_reply"),
                            "returned_to_verification_count": sum(row["returned_to_verification"] for row in replies),
                            "mechanism_exercised": bool(replies)},
        accepted_verification_calls=verification_calls,
        all_verification_complete=(verification_calls == protocol["allocation"]
                                   and host["verification_calls_used"] == protocol["allocation"]
                                   and not result["unresolved_request_outcome"]),
        historical_reference_action_comparison=comparison,
        causal_benefit_qualified=False, scientific_control_transfer_qualified=False,
        formal_model_comparison_qualified=False,
    )
    return result


def summarize_phase_action_decoder(report: dict, native: dict, transport: list[dict],
                                   events: list[dict], scheduler: dict, protocol: dict,
                                   schemas: dict, collaboration_config: dict) -> dict:
    """Read one decoder execution without promoting it to a control comparison."""
    # Older archived analyzers stage only this file and provenance.py. Keep the
    # new schema-builder dependency confined to its separately staged entry point.
    from .appworld_action_decoder import build_phase_action_schema, validate_action_decoder_bounds
    from .appworld_collaboration import content_digest

    bounded = protocol.get("schema") == "appworld_bounded_message_decoder_development_v1"
    bounds = {"message_text_max_length": 256, "max_consecutive_whitespaces": 4} if bounded else None
    if bounded and validate_action_decoder_bounds(protocol.get("action_decoder_bounds")) != bounds:
        raise ValueError("FROZEN_PHASE_ACTION_DECODER_PROTOCOL_REQUIRED")
    frozen = {"schema": ("appworld_bounded_message_decoder_development_v1" if bounded
                         else "appworld_phase_action_decoder_development_v1"),
              "model_revision": "Qwen/Qwen3-Coder-30B-A3B-Instruct@b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
              "historical_reference_job": 597334 if bounded else 596889, "prefix_generation_attempts": 10,
              "maximum_new_attempts": 20, "maximum_total_attempts": 30, "maximum_new_executions": 1,
              "allocation": {"planner": 1, "executor": 1}, "phase_action_decoder": True,
              "verification_message_replies": True, "reply_generation_attempts_per_delivered_message": 1,
              "lossless_observation_history": True, "evidence_table_after_prefix": True,
              "max_context_tokens": 65536, "max_new_tokens": 512}
    if (any(type(protocol.get(key)) is not type(value) or protocol[key] != value for key, value in frozen.items())
            or any(type(value) is not int for value in protocol["allocation"].values())
            or protocol.get("decoder", {}).get("implementation") != "lm-format-enforcer"
            or protocol.get("decoder", {}).get("version") != "0.11.3"):
        raise ValueError("FROZEN_PHASE_ACTION_DECODER_PROTOCOL_REQUIRED")
    host = report["host"]
    if (host.get("model_revision") != frozen["model_revision"]
            or collaboration_config.get("model_revision") != frozen["model_revision"]
            or collaboration_config.get("provider") != "local_transformers"
            or collaboration_config.get("max_actor_turns") != 30
            or collaboration_config.get("verification_budget") != 2):
        raise ValueError("PHASE_ACTION_DECODER_CONFIGURATION_MISMATCH")
    if (host.get("phase_action_decoder") is not True or host.get("verification_message_replies") is not True
            or host.get("evidence_table_after_prefix") is not True
            or host.get("observation_history_encoding") != "actor_observation_append_v1"):
        raise ValueError("PHASE_ACTION_DECODER_HOST_NOT_ENABLED")
    if host.get("action_decoder_bounds") != bounds:
        raise ValueError("PHASE_ACTION_DECODER_BOUNDS_MISMATCH")
    result = summarize_arm(report, native, transport, events, scheduler, 10,
                           allow_partial_inference_measurements=True)
    issues, trace, metadata_rows = [], [], []
    verification_calls = {"planner": 0, "executor": 0}

    def issue(code, index=None, **details):
        issues.append({"code": code, **({"attempt_index": index} if index is not None else {}), **details})

    def nonnegative_number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    for index, (entry, attempt, generation) in enumerate(zip(result["actions"], host["host_attempts"], host["generation_attempts"])):
        entry.update(usage=generation.get("usage"), inference_seconds=generation.get("inference_seconds"))
        if attempt["attempt_index"] != index:
            issue("ATTEMPT_INDEX_MISMATCH", index)
        if generation.get("model_revision") != frozen["model_revision"]:
            issue("GENERATION_MODEL_MISMATCH", index)
        if index < 10:
            if "action_decoder" in generation or "action_schema_sha256" in attempt:
                issue("PREFIX_DECODER_ACTIVITY", index)
            continue
        usage, seconds = generation.get("usage"), generation.get("inference_seconds")
        failed = generation.get("status") == "FAILED" and attempt["status"] == "GENERATION_FAILED"
        if usage is not None:
            if (not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"))):
                issue("GENERATION_TOKEN_USAGE_INVALID", index)
            elif (usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                  or usage["prompt_tokens"] > 65536 or usage["completion_tokens"] > 512):
                issue("GENERATION_TOKEN_BUDGET_OR_TOTAL_MISMATCH", index)
        elif not failed:
            issue("GENERATED_ACTION_TOKEN_USAGE_MISSING", index)
        if seconds is not None and not nonnegative_number(seconds):
            issue("INFERENCE_TIME_INVALID", index)
        elif seconds is None and not failed:
            issue("GENERATED_ACTION_INFERENCE_TIME_MISSING", index)
        expected_digest = None
        try:
            prompt = json.loads(attempt["unencoded_user_prompt"])
            if prompt["actor"] != entry["actor"] or prompt["host_state"] != entry["host_state"]:
                raise ValueError("SAVED_PRIVATE_PROMPT_IDENTITY_MISMATCH")
            expected_schema = build_phase_action_schema(entry["actor"], entry["host_state"], prompt["view"], schemas,
                discovery_call_limit=collaboration_config["discovery_calls_per_actor"],
                reply_recipient=prompt.get("reply_recipient"),
                message_text_max_length=None if bounds is None else bounds["message_text_max_length"])
            expected_digest = content_digest(expected_schema)
            if attempt.get("action_schema_sha256") != expected_digest:
                issue("HOST_SCHEMA_BINDING_MISMATCH", index)
        except (KeyError, TypeError, ValueError) as exc:
            issue("PRIVATE_VIEW_SCHEMA_RECONSTRUCTION_FAILED", index, error_type=type(exc).__name__, error=str(exc))
        metadata = generation.get("action_decoder")
        row = {"attempt_index": index, "actor": entry["actor"], "host_state": entry["host_state"],
               "generation_status": generation.get("status"), "host_attempt_status": attempt["status"],
               "error_code": generation.get("error_code", attempt.get("error_code")),
               "expected_schema_sha256": expected_digest, "metadata": metadata,
               "usage": usage, "inference_seconds": seconds}
        if metadata is None:
            row["decoder_stage"] = ("CONTEXT_REFUSED_BEFORE_DECODER" if failed and generation.get("error_code") == "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED"
                                    else "GENERATION_PREPARATION_FAILED" if failed else "DECODER_RECEIPT_MISSING")
            if not failed or usage is not None or seconds is not None:
                issue("DECODER_RECEIPT_MISSING", index)
        elif not isinstance(metadata, dict):
            row["decoder_stage"] = "DECODER_RECEIPT_INVALID"
            issue("DECODER_RECEIPT_INVALID", index)
        else:
            metadata_rows.append(metadata)
            calls, failures = metadata.get("callback_calls"), metadata.get("callback_failures")
            counts_valid = type(calls) is int and calls >= 0 and type(failures) is int and 0 <= failures <= calls
            if not counts_valid:
                issue("DECODER_CALLBACK_COUNTER_INVALID", index)
            for key in ("callback_wall_seconds", "preparation_wall_seconds"):
                if not nonnegative_number(metadata.get(key)):
                    issue("DECODER_TIME_INVALID", index, field=key)
            try:
                if (not isinstance(metadata.get("schema"), dict)
                        or content_digest(metadata["schema"]) != metadata.get("schema_sha256")
                        or metadata.get("schema_sha256") != expected_digest):
                    issue("DECODER_SCHEMA_BINDING_MISMATCH", index)
            except (TypeError, ValueError):
                issue("DECODER_SCHEMA_BINDING_MISMATCH", index)
            preparation_failed = failed and seconds is None and calls == 0
            if not preparation_failed:
                settings = metadata.get("parser_settings") or {}
                if (metadata.get("implementation") != "lm-format-enforcer" or metadata.get("version") != "0.11.3"
                        or metadata.get("interegular_version") != "0.3.3"
                        or settings.get("max_json_array_length") != 0 or settings.get("force_json_field_order") is not False
                        or type(settings.get("max_consecutive_whitespaces")) is not int
                        or (settings["max_consecutive_whitespaces"] != bounds["max_consecutive_whitespaces"]
                            if bounded else settings["max_consecutive_whitespaces"] <= 512)):
                    issue("DECODER_CONFIGURATION_MISMATCH", index)
            if not failed and (not counts_valid or calls == 0 or failures != 0):
                issue("GENERATED_ACTION_WITHOUT_SUCCESSFUL_CALLBACKS", index)
            row["decoder_stage"] = ("DECODER_PREPARATION_FAILED" if preparation_failed else
                                    "DECODER_CALLBACK_FAILED" if failed and failures else
                                    "MODEL_OR_DECODE_FAILED_AFTER_CALLBACK" if failed and calls else
                                    "MODEL_FAILED_BEFORE_CALLBACK" if failed else "GENERATED_WITH_DECODER")
        entry["decoder_stage"] = row["decoder_stage"]
        trace.append(row)
        if entry["host_state"] == "VERIFY" and entry["kind"] == "tool" and entry["status"] == "ACCEPTED":
            verification_calls[entry["actor"]] += 1
    if (host["host_attempt_limit"] != 30 or result["logical_generation_attempts"] > 30
            or result["new_generation_attempts"] > 20):
        issue("GENERATION_ATTEMPT_BUDGET_MISMATCH")
    for actor, count in verification_calls.items():
        recorded = host["verification_calls_used"].get(actor)
        if type(recorded) is not int or not 0 <= recorded <= 1 or count > 1 or count != recorded:
            issue("VERIFICATION_QUOTA_OR_COUNTER_MISMATCH", actor=actor, recorded=recorded, accepted_calls=count)
    runtime = report.get("actor_runtime", {}).get("action_decoder")
    totals = {"attempt_count": len(metadata_rows)}
    for key in ("callback_calls", "callback_failures", "callback_wall_seconds", "preparation_wall_seconds"):
        values = [row.get(key) for row in metadata_rows]
        totals[key] = sum(values) if all(nonnegative_number(value) for value in values) else None
    totals["schema_sha256s"] = sorted({row["schema_sha256"] for row in metadata_rows if isinstance(row.get("schema_sha256"), str)})
    if metadata_rows and not isinstance(runtime, dict):
        issue("DECODER_RUNTIME_AGGREGATE_MISSING")
    elif isinstance(runtime, dict):
        for key, expected in totals.items():
            actual = runtime.get(key)
            if key.endswith("_seconds"):
                matches = expected is not None and nonnegative_number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8)
            else:
                matches = type(actual) is type(expected) and actual == expected
            if not matches:
                issue("DECODER_RUNTIME_AGGREGATE_MISMATCH", field=key, recorded=actual, ledger_sum=expected)
    result.update(schema=protocol["schema"],
        status="PHASE_ACTION_DECODER_ACCOUNTING_REQUIRES_REVIEW" if issues else "COMPLETE_DESCRIPTIVE_PHASE_ACTION_DECODER_ACCOUNTING",
        accounting_complete=not issues, accounting_issues=issues, decoder_trace=trace,
        decoder_runtime_reported=runtime, decoder_runtime_from_attempts=totals,
        prefix_decoder_activity_absent=not any(row["code"] == "PREFIX_DECODER_ACTIVITY" for row in issues),
        decoder_cost_scope="Preparation and callbacks are included in host wall time; callbacks also overlap inference time. Do not add these intervals again.",
        accepted_verification_calls=verification_calls,
        all_verification_complete=verification_calls == frozen["allocation"] and host["verification_calls_used"] == frozen["allocation"] and not result["unresolved_request_outcome"],
        new_measured_token_generations=sum(row.get("usage") is not None for row in host["generation_attempts"][10:]),
        new_measured_inference_generations=sum(row.get("inference_seconds") is not None for row in host["generation_attempts"][10:]),
        historical_reference={"job_id": frozen["historical_reference_job"], "role": "descriptive context only", "post_prefix_preintervention_comparison_performed": False},
        verification_reply_reported={key: host.get(key) for key in ("verification_reply_slots", "verification_reply_attempts", "verification_replies_completed", "pending_verification_reply")},
        new_executions=1, causal_benefit_qualified=False, scientific_control_transfer_qualified=False,
        formal_model_comparison_qualified=False)
    return result


def analyze_pair(protocol_path: Path, archives: dict[str, Path], scheduler: dict) -> dict:
    protocol = yaml.safe_load(protocol_path.read_text())
    if (protocol["schema"] != "appworld_recorded_prefix_diagnostic_v1"
            or protocol["maximum_task_prefix_pairs"] != 1 or protocol["prefix_attempts"] != 10
            or protocol["remaining_attempts"] != 20 or tuple(protocol["arms"]) != ARMS):
        raise ValueError("FROZEN_SINGLE_PAIR_PROTOCOL_REQUIRED")
    arms, bindings = {}, []
    common_inputs = None
    for arm in ARMS:
        archive = archives[arm]
        if (archive / "inputs/diagnostic_arm.txt").read_text().strip() != arm:
            raise ValueError("ARM_IDENTITY_MISMATCH")
        staged_protocol = archive / "inputs/prefix/protocol.yaml"
        if sha256_file(staged_protocol) != sha256_file(protocol_path):
            raise ValueError("ARM_PROTOCOL_MISMATCH")
        receipt = json.loads((archive / "inputs/copy_out.json").read_text())
        fixed = {row["path"]: row["sha256"] for row in receipt["files"] if row["path"] not in {"diagnostic_arm.txt", "controller_command.json"}}
        if common_inputs is not None and fixed != common_inputs:
            raise ValueError("PAIRED_FROZEN_INPUTS_DIFFER")
        common_inputs = fixed
        source_archive = Path("Output/neural_population_dynamics/agentic_extension/appworld_local_collaboration") / ("run_" + str(receipt["job_id"]))
        selected = {}
        needed = {"gpu_host": ["host/report.json"], "cpu_controller": ["controller/summary.json", "controller/transport.jsonl", "controller/events.json"]}
        for role, names in needed.items():
            delivery = json.loads((archive / role / "copy_out.json").read_text())
            declared = {row["path"]: row for row in delivery["files"]}
            if delivery["verified"] is not True:
                raise ValueError("UNVERIFIED_ARCHIVE_DELIVERY")
            for name in names:
                path = archive / role / name
                digest = sha256_file(path)
                if digest != declared[name]["sha256"]:
                    raise ValueError("DECLARED_INPUT_HASH_MISMATCH:" + str(path))
                bindings.append({"arm": arm, "path": str(source_archive / role / name), "sha256": digest})
                selected[name] = [json.loads(line) for line in path.read_text().splitlines()] if path.suffix == ".jsonl" else json.loads(path.read_text())
        transport = selected["controller/transport.jsonl"]
        if transport[0]["response"]["result"]["task"]["instruction"] != TARGET_INSTRUCTION:
            raise ValueError("PUBLIC_TASK_IDENTITY_MISMATCH")
        host_report = selected["host/report.json"]
        if host_report["host"]["evidence_table_after_prefix"] != (arm == ARMS[1]):
            raise ValueError("OBSERVATION_INTERVENTION_MISMATCH")
        arms[arm] = summarize_arm(host_report, selected["controller/summary.json"], transport,
                                  selected["controller/events.json"], scheduler[arm], 10)
        arms[arm]["archive"] = str(source_archive)
        arms[arm]["job_id"] = str(receipt["job_id"])
    return {"status": "COMPLETE_DESCRIPTIVE_SINGLE_PAIR_ACCOUNTING", "independent_task_prefix_pairs": 1,
            "branch_count": 2, "inference": "dependent descriptive comparison; no significance or allocation-effect claim",
            "protocol_sha256": sha256_file(protocol_path), "inputs": bindings, "arms": arms,
            "new_pair_recorded_tokens": sum(a["new_inference_cost"]["provider_token_usage"]["total_tokens"] for a in arms.values()),
            "new_pair_recorded_inference_seconds": sum(a["new_inference_cost"]["provider_inference_seconds"] for a in arms.values()),
            "new_inference_measurement_scope": "sum of recorded token/time observations; missing attempt measurements remain unreported",
            "all_new_attempts_have_inference_measurements": all(a["new_measured_generations"] == a["new_generation_attempts"] for a in arms.values()),
            "wall_time_accounting": "host and native wall intervals overlap; do not sum them",
            "scientific_control_transfer_qualified": False, "formal_model_comparison_qualified": False}


def analyze_decoder_execution(archive: Path, scheduler: dict, validated_software: Path) -> dict:
    """Bind one terminal decoder execution to its tested code and native receipts."""
    components = scheduler.get("components", [])
    job = archive.name.removeprefix("run_")
    if (not job.isdigit() or scheduler.get("query_node") != "mgmt2" or scheduler.get("exit_code") != 0
            or {str(row["job_id"]) for row in components} != {job, str(int(job)+1)}
            or any(row["state"] in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"} for row in components)):
        raise ValueError("TERMINAL_DECODER_COMPONENT_OBSERVATION_REQUIRED")
    bindings = []
    def load(role, name, *, text=False):
        delivery = json.loads((archive/role/"copy_out.json").read_text())
        entries = {row["path"]:row for row in delivery["files"]}
        if delivery.get("verified") is not True or str(delivery.get("job_id")) != job:
            raise ValueError("VERIFIED_DECODER_ARCHIVE_REQUIRED")
        path = archive/role/name
        entry = entries[name]
        digest = sha256_file(path)
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError("DECLARED_DECODER_INPUT_CHANGED:"+role+"/"+name)
        bindings.append({"path":role+"/"+name,"bytes":entry["bytes"],"sha256":digest})
        return path.read_text() if text else json.loads(path.read_text())
    proof = json.loads((validated_software/"report.json").read_text())
    if proof.get("status") != "SUBMISSION_BINDING_AND_CURRENT_PREFIX_REUSE_VERIFIED" or proof.get("exit_code") != 0:
        raise ValueError("TESTED_DECODER_SOFTWARE_REQUIRED")
    tested = {row["path"]:row["sha256"] for row in proof["inputs"]}
    for name in ["Libs/population_dynamics/appworld_action_decoder.py",
                 "Libs/population_dynamics/appworld_local_actor.py",
                 "Libs/population_dynamics/appworld_local_host.py",
                 "Libs/scripts/run_appworld_local_collaboration.py",
                 "phase_action_decoder_protocol.yaml", "action_decoder_bounds.yaml", "launcher.sbatch"]:
        load("inputs",name,text=True)
        if bindings[-1]["sha256"] != tested[name]:
            raise ValueError("EXECUTED_DECODER_SOFTWARE_DIFFERS_FROM_TESTED:"+name)
    protocol = yaml.safe_load(load("inputs","phase_action_decoder_protocol.yaml",text=True))
    report = load("gpu_host","host/report.json")
    native = load("cpu_controller","controller/summary.json")
    transport = [json.loads(line) for line in load("cpu_controller","controller/transport.jsonl",text=True).splitlines() if line.strip()]
    events = load("cpu_controller","controller/events.json")
    schemas = load("inputs","api_docs.json")
    config = yaml.safe_load(load("inputs","Libs/configs/neural_population_appworld_local_collaboration.yaml",text=True))
    summary = summarize_phase_action_decoder(report,native,transport,events,components,protocol,schemas,config)
    return {"status":summary["status"],"job_id":job,"summary":summary,"inputs":bindings,
            "validated_software":str(validated_software),"scheduler":scheduler,
            "new_model_generations":0,"new_native_executions":0,"scientific_control_transfer_qualified":False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--original-view", type=Path)
    parser.add_argument("--evidence-table", type=Path)
    parser.add_argument("--decoder-execution", type=Path)
    parser.add_argument("--validated-decoder-software", type=Path)
    parser.add_argument("--scheduler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.decoder_execution is not None:
        if args.validated_decoder_software is None or any((args.protocol,args.original_view,args.evidence_table)):
            parser.error("decoder-execution requires validated-decoder-software and excludes pair inputs")
        result = analyze_decoder_execution(args.decoder_execution,json.loads(args.scheduler.read_text()),args.validated_decoder_software)
        args.output.mkdir(parents=True,exist_ok=False)
        freeze_json(args.output/"report.json",result)
        rows = result["summary"]["actions"]
        with (args.output/"actions.csv").open("x",newline="",encoding="utf-8") as stream:
            columns = list(dict.fromkeys(key for row in rows for key in row)) or ["attempt_index"]
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                                 for key,value in row.items()})
        code = 0 if result["summary"]["accounting_complete"] else 1
        freeze_json(args.output/"manifest.json",runtime_manifest({key:str(value) for key,value in vars(args).items()},sys.argv,code))
        print(json.dumps({"status":result["status"],"job_id":result["job_id"],"accounting_complete":code==0}))
        return code
    if args.validated_decoder_software is not None or any(value is None for value in (args.protocol,args.original_view,args.evidence_table)):
        parser.error("pair accounting requires protocol, original-view and evidence-table")
    result = analyze_pair(args.protocol, dict(zip(ARMS, (args.original_view, args.evidence_table))), json.loads(args.scheduler.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    freeze_json(args.output / "report.json", result)
    columns = ["arm", "termination", "host_state", "new_generation_attempts", "accepted_task_write_requests", "native_write_successes", "native_write_failures", "finish_accepted", "native_evaluation_status", "coverage_status", "recorded_new_tokens", "recorded_new_inference_seconds"]
    with (args.output / "arms.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for name, arm in result["arms"].items():
            row = {key: arm[key] for key in columns if key in arm}
            row.update(arm=name, coverage_status=arm["coverage"]["status"], recorded_new_tokens=arm["new_inference_cost"]["provider_token_usage"]["total_tokens"], recorded_new_inference_seconds=arm["new_inference_cost"]["provider_inference_seconds"])
            writer.writerow(row)
    freeze_json(args.output / "manifest.json", runtime_manifest({key: str(value) for key, value in vars(args).items()}, sys.argv, 0))
    print(json.dumps({"status": result["status"], "independent_task_prefix_pairs": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

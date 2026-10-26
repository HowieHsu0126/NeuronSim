"""Host scheduling and failure evidence through the real controller in CPU fixtures."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from .appworld_collaboration import ACTORS, CollaborationCase
from .appworld_local_actor import model_identity
from .appworld_local_host import LocalActorHost


class Backend:
    task = {"instruction": "fixture task without evaluation details"}

    def __init__(self):
        self.calls = []
        self.fail_verification = False

    def call(self, call_id, app, api, arguments, verification):
        self.calls.append({"call_id": call_id, "app": app, "api": api,
                           "arguments": deepcopy(arguments), "verification": verification})
        if verification and self.fail_verification:
            return {"ok": False, "error": "failed verification " + call_id}
        return {"ok": True, "result": {"private_value": "PRIVATE_" + call_id}}

    def evaluate(self):
        return {"success": True, "marker": "TERMINAL_EVALUATION_ONLY"}


class ControllerChannel:
    """Synchronous JSON peer that rejects pipelining and uses CollaborationCase."""

    def __init__(self, case):
        self.case = case
        self.messages = [{"ready": True, "task": deepcopy(case.backend.task), "actors": list(ACTORS)}]
        self.requests = []
        self.pending = False
        self.failure_mode = None
        self.fail_receive = False
        self.terminal_status = "DEVELOPMENT_EXECUTION_COMPLETE"

    def send(self, request):
        assert not self.pending and not self.messages, "Only one request may be outstanding"
        self.pending = True
        self.requests.append(deepcopy(request))
        try:
            response = {"ok": True, "result": self.case.handle(request)}
        except ValueError as exc:
            response = {"ok": False, "error": str(exc)}
        except Exception as exc:
            response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        if request.get("call_id") == "task_write" and self.failure_mode:
            if self.failure_mode == "send":
                raise OSError("write may already have reached controller")
            if self.failure_mode == "receive":
                self.fail_receive = True
            elif self.failure_mode == "opaque":
                response = {"ok": False, "error": "opaque native output decoding failure"}
            elif self.failure_mode == "fatal":
                response = {"ok": False, "error_type": "RuntimeError", "error": "NATIVE_OPERATION_VALUE_ERROR"}
            elif self.failure_mode == "malformed":
                response = {"unexpected_response": True}
        self.messages.append(response)
        if self.case.finished:
            self.messages.append({"terminal": {"status": self.terminal_status}})

    def receive(self):
        if self.fail_receive:
            raise OSError("response connection lost after native write")
        assert self.messages, "Host must not wait for an unsolicited message"
        response = self.messages.pop(0)
        if self.pending:
            self.pending = False
        return deepcopy(response)


class ScriptedActor:
    """Deterministic fixture decisions; preserves separate original text histories."""

    def __init__(self):
        self.config = {
            "model": {"repo_id": "fixture/actor", "revision": "a" * 40, "dtype": "bfloat16",
                      "attention_implementation": "sdpa", "cuda_devices": 2, "max_memory_gib_per_device": 40},
            "generation": {"max_new_tokens": 256, "max_context_tokens": 4096, "do_sample": False},
            "inference_seed": 73003,
        }
        self.histories = {actor: [{"role": "system", "content": actor + " private context"}] for actor in ACTORS}
        self.session_ids = {actor: "context_" + actor for actor in ACTORS}
        self.attempts, self.prompts = [], []
        self.action_schemas = []
        self.discovery_visits = {actor: 0 for actor in ACTORS}
        self.handoff_visits = 0
        self.omit_first_handoff = self.omit_all_handoffs = False
        self.request_planner_reply = True
        self.reply_requested = False
        self.override = None

    @staticmethod
    def message(actor, text, shares=None, kind="message"):
        return {"actor": actor, "kind": kind, "recipient": "executor" if actor == "planner" else "planner",
                "text": text, "share_results": [] if shares is None else shares}

    @staticmethod
    def tool(actor, call_id, api="read"):
        return {"actor": actor, "kind": "tool", "call_id": call_id, "app": "spotify", "api": api, "arguments": {}}

    def action(self, actor, content):
        state, view = content["host_state"], content["view"]
        if state == "DISCOVERY":
            self.discovery_visits[actor] += 1
            if actor == "planner" and self.discovery_visits[actor] >= 2:
                return self.message(actor, "proposed plan", kind="proposal")
            return self.tool(actor, actor + "_private_read")
        if state == "VERIFY":
            return self.tool(actor, "verify_" + actor + "_" + str(view["used_verification_calls"]))
        if state == "PLANNER_HANDOFF":
            self.handoff_visits += 1
            shares = content["required_shared_result_ids"]
            if self.omit_all_handoffs or (self.omit_first_handoff and self.handoff_visits == 1):
                shares = []
            return self.message(actor, "generated verification handoff", shares)
        if state == "VERIFY_REPLY":
            return self.message(actor, "generated verification reply")
        if state == "PLANNER_REPLY":
            return self.message(actor, "generated planner reply")
        if self.request_planner_reply and not self.reply_requested:
            self.reply_requested = True
            return self.message(actor, "please reply using your own view")
        if "task_write" not in view["results"]:
            return self.tool(actor, "task_write", api="write")
        return {"actor": actor, "kind": "finish"}

    def generate(self, actor, user_text, *, action_schema=None):
        self.action_schemas.append(deepcopy(action_schema))
        content = json.loads(user_text)
        self.prompts.append((actor, deepcopy(content)))
        messages = deepcopy(self.histories[actor]) + [{"role": "user", "content": user_text}]
        prompt = json.dumps(messages, ensure_ascii=False)
        record = {"actor": actor, "session_id": self.session_ids[actor], "model_revision": model_identity(self.config),
                  "prompt": prompt, "usage": None, "inference_seconds": None, "status": "PREPARING"}
        self.attempts.append(record)
        action = self.action(actor, content)
        if self.override is not None:
            action = self.override(actor, content, action, len(self.attempts))
        if isinstance(action, Exception):
            record.update(status="FAILED", error_type=type(action).__name__, host_attempt_wall_seconds=0.01)
            if isinstance(action, ValueError):
                record["error_code"] = str(action)
            raise action
        response = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False)
        record.update(response=response, status="GENERATED", inference_seconds=0.01,
                      usage={"prompt_tokens": len(prompt), "completion_tokens": len(response),
                             "total_tokens": len(prompt) + len(response)}, host_attempt_wall_seconds=0.02)
        self.histories[actor] = messages + [{"role": "assistant", "content": response}]
        return {key: record[key] for key in ("session_id", "prompt", "response", "model_revision", "usage", "inference_seconds")}


@pytest.fixture
def setup_host():
    def create(allocation, *, attempt_limit=30, verification_message_replies=False):
        actor = ScriptedActor()
        config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_collaboration.yaml").read_text())
        config.update(provider="local_transformers", model_revision=model_identity(actor.config), max_actor_turns=attempt_limit)
        schemas = {"spotify": {"read": {"method": "GET", "parameters": [], "description": "complete public read documentation"},
                               "write": {"method": "POST", "parameters": []}},
                   "supervisor": {"describe": {"method": "GET", "parameters": [], "description": "unused but still supplied"}}}
        case = CollaborationCase(config, schemas, Backend())
        channel = ControllerChannel(case)
        host = LocalActorHost(actor, channel, config, schemas, allocation,
                              verification_message_replies=verification_message_replies)
        return host, actor, channel, case
    return create


@pytest.mark.parametrize("allocation", [{"planner": 2, "executor": 0}, {"planner": 1, "executor": 1}, {"planner": 0, "executor": 2}])
def test_assigned_allocations_execute_real_controller_phases_and_private_handoff(setup_host, allocation):
    host, actor, channel, case = setup_host(allocation)
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    assert report["allocation_acknowledged"] is True and report["verification_calls_used"] == allocation
    assert [request["allocation"] for request in channel.requests if request["kind"] == "allocate"] == [allocation]
    verification = [call for call in case.backend.calls if call["verification"]]
    assert [call["call_id"].split("_")[1] for call in verification] == ["planner"] * allocation["planner"] + ["executor"] * allocation["executor"]
    planner_ids = [call["call_id"] for call in verification if call["call_id"].startswith("verify_planner")]
    assert report["planner_verification_call_ids"] == planner_ids
    assert set(report["planner_shared_verification_call_ids"]) == set(planner_ids)
    assert set(planner_ids) <= case.visible["executor"]
    assert report["planner_reply_slots"] == 1
    assert [name for name, _ in actor.prompts[:3]] == ["planner", "executor", "planner"]
    handoff_events = [row for row in case.events if row.get("request", {}).get("text") == "generated verification handoff"]
    assert len(handoff_events) == 1 and handoff_events[0]["phase_before"] == "execution"
    for name in ACTORS:
        prompts = [content for actor_name, content in actor.prompts if actor_name == name]
        assert prompts[0]["public_api_schemas"] == case.schemas
        assert all("public_api_schemas" not in content for content in prompts[1:])
        other = "executor" if name == "planner" else "planner"
        assert "PRIVATE_" + other + "_private_read" not in json.dumps(prompts)
    first_execution = next(content for name, content in actor.prompts if name == "executor" and content["host_state"] == "EXECUTE")
    assert set(planner_ids) <= set(first_execution["view"]["results"])
    assert report["native_evaluation"]["marker"] == "TERMINAL_EVALUATION_ONLY"
    assert "TERMINAL_EVALUATION_ONLY" not in json.dumps(actor.histories)
    assert report["terminal"]["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
    assert report["submitted_generation_count"] == report["acknowledged_generation_count"] == report["host_attempt_count"]
    assert all(row["status"] == "ACCEPTED" for row in report["host_attempts"])
    assert report["generation_attempts"] == actor.attempts
    for key in ("scientific_control_transfer_qualified", "generative_transfer_qualified", "formal_model_revision_verified",
                "formal_model_comparison_qualified", "complete_host_actor_native_cost_qualified"):
        assert report[key] is False
    with pytest.raises(RuntimeError, match="LOCAL_HOST_ALREADY_RUN"):
        host.run()


def test_failed_api_attempts_consume_quota_and_are_shared_as_failures(setup_host):
    host, actor, channel, case = setup_host({"planner": 2, "executor": 0})
    case.backend.fail_verification = True
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
    assert report["verification_calls_used"] == {"planner": 2, "executor": 0}
    assert len([call for call in case.backend.calls if call["verification"]]) == 2
    execution = next(content for name, content in actor.prompts if name == "executor" and content["host_state"] == "EXECUTE")
    assert all(execution["view"]["results"][call_id]["ok"] is False for call_id in report["planner_verification_call_ids"])


def test_message_delivery_after_rejected_write_preserves_verification_barrier(setup_host, monkeypatch):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, attempt_limit=6)
    evaluations = []
    original_evaluate = case.backend.evaluate

    def evaluate():
        evaluations.append(True)
        return original_evaluate()

    monkeypatch.setattr(case.backend, "evaluate", evaluate)
    actions = iter([
        actor.tool("executor", "rejected_write", api="write"),
        actor.message("executor", "The requested task is complete."),
        {"actor": "executor", "kind": "finish"},
    ])

    def override(name, content, action, index):
        if index <= 3:
            return action
        assert name == "executor" and content["host_state"] == "VERIFY"
        return next(actions)

    actor.override = override
    report = host.run()
    assert report["status"] == "FAILED"
    assert report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["host_state"] == "VERIFY" and case.phase == "verification"
    assert report["verification_calls_used"] == {"planner": 0, "executor": 0}
    assert report["native_evaluation"] is None and not evaluations and not case.finished
    assert not any(call["api"] == "write" or call["verification"] for call in case.backend.calls)
    assert report["host_attempts"][3]["status"] == "CONTROLLER_REJECTED"
    assert report["host_attempts"][5]["status"] == "CONTROLLER_REJECTED"

    prompts = [content for name, content in actor.prompts if content["host_state"] == "VERIFY"]
    assert len(prompts) == 3
    assert all(content["view"]["used_verification_calls"] == 0 for content in prompts)
    assert all(content["view"]["allocated_verification_calls"] == 2 for content in prompts)
    rejection = prompts[1]["last_feedback"]
    assert rejection["ok"] is False
    acknowledgement = prompts[2]["last_feedback"]
    assert acknowledgement["ok"] is True
    assert acknowledgement["result"] == {"delivered_to": "planner", "phase": "verification"}
    observations = [json.loads(row["content"]) for row in actor.histories["executor"]
                    if row["role"] == "user"]
    assert any(content["last_feedback"] == rejection for content in observations)


@pytest.mark.parametrize("failure", ["malformed", "controller_rejection"])
def test_parse_and_known_controller_rejections_retain_unmodified_generations_and_costs(setup_host, failure):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})

    def override(name, content, action, index):
        if index != 1:
            return action
        if failure == "malformed":
            return "original malformed response without JSON"
        return {**action, "api": "not_a_public_api"}

    actor.override = override
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    first = report["host_attempts"][0]
    assert first["status"] == ("HOST_REJECTED" if failure == "malformed" else "CONTROLLER_REJECTED")
    assert first["submitted"] is (failure != "malformed")
    assert first["acknowledged"] is (failure != "malformed")
    assert report["generation_attempts"][0]["response"] == actor.attempts[0]["response"]
    costs = report["cost_accounting"]
    assert costs["generations_with_token_usage"] == report["host_attempt_count"]
    assert costs["provider_token_usage"]["total_tokens"] == sum(row["usage"]["total_tokens"] for row in actor.attempts)
    assert costs["provider_inference_seconds"] == pytest.approx(0.01 * len(actor.attempts))
    if failure == "malformed":
        assert report["submitted_generation_count"] == report["host_attempt_count"] - 1
    for request in channel.requests:
        if "generation" in request:
            assert {key: value for key, value in request.items() if key != "generation"} == json.loads(request["generation"]["response"])


def test_omitted_handoff_shares_block_execution_until_generated_message_supplies_them(setup_host):
    host, actor, channel, case = setup_host({"planner": 2, "executor": 0})
    actor.omit_first_handoff = True
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
    handoffs = [request for request in channel.requests if request.get("text") == "generated verification handoff"]
    assert len(handoffs) == 2 and handoffs[0]["share_results"] == []
    assert set(handoffs[1]["share_results"]) == set(report["planner_verification_call_ids"])
    states = [row["host_state"] for row in report["host_attempts"]]
    assert states.count("PLANNER_HANDOFF") == 2
    assert states.index("EXECUTE") > max(index for index, state in enumerate(states) if state == "PLANNER_HANDOFF")


def test_missing_handoff_never_becomes_implicit_host_sharing_or_task_execution(setup_host):
    host, actor, channel, case = setup_host({"planner": 2, "executor": 0}, attempt_limit=8)
    actor.omit_all_handoffs = True
    report = host.run()
    assert report["status"] == "FAILED" and report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["host_attempt_count"] == 8
    assert not any(call["api"] == "write" for call in case.backend.calls)
    assert not set(report["planner_verification_call_ids"]) & case.visible["executor"]
    assert report["native_evaluation"] is None


def test_malformed_attempts_exhaust_independent_host_limit_without_controller_turns(setup_host):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1}, attempt_limit=4)
    actor.override = lambda *args: "not JSON"
    report = host.run()
    assert report["status"] == "FAILED" and report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["host_attempt_count"] == report["generation_attempt_count"] == 4
    assert report["submitted_generation_count"] == report["acknowledged_generation_count"] == case.turns == 0
    assert [name for name, _ in actor.prompts] == ["planner", "executor", "planner", "executor"]
    assert not case.backend.calls and report["allocation_attempted"] is False
    assert report["cost_accounting"]["provider_token_usage"] is not None


def test_context_failure_remains_explicit_without_truncation_or_retry(setup_host):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    actor.override = lambda *args: ValueError("LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED")
    report = host.run()
    assert report["status"] == "FAILED" and report["host_attempt_count"] == 1
    assert report["generation_attempts"][0]["error_code"] == "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED"
    assert actor.prompts[0][1]["public_api_schemas"] == case.schemas
    assert report["submitted_generation_count"] == 0 and not case.backend.calls
    assert report["cost_accounting"]["provider_token_usage"] is None
    assert report["cost_accounting"]["provider_inference_seconds"] is None
    assert report["cost_accounting"]["provider_telemetry_status"] == "UNKNOWN"


@pytest.mark.parametrize("mode", ["send", "receive", "opaque", "fatal", "malformed"])
def test_unknown_native_outcome_stops_without_replaying_an_applied_write(setup_host, mode):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    channel.failure_mode = mode
    report = host.run()
    assert report["status"] == "FAILED" and report["unknown_native_outcome"] is True
    assert len([call for call in case.backend.calls if call["call_id"] == "task_write"]) == 1
    assert len([request for request in channel.requests if request.get("call_id") == "task_write"]) == 1
    assert channel.requests[-1]["call_id"] == "task_write"
    last = report["host_attempts"][-1]
    assert last["submission_attempted"] is True
    assert last["submitted"] is (None if mode == "send" else True)
    assert last["acknowledged"] is (mode in {"opaque", "fatal"})
    assert last["status"] in {"OUTCOME_UNKNOWN", "CONTROLLER_FATAL_OR_OPAQUE"}
    assert report["native_evaluation"] is None and report["terminal"] is None
    assert report["cost_accounting"]["generations_with_token_usage"] == report["host_attempt_count"]


def test_host_preflight_rejects_revision_and_allocation_before_channel_use(setup_host):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    with pytest.raises(ValueError, match="LOCAL_HOST_MODEL_IDENTITY_MISMATCH"):
        LocalActorHost(actor, channel, {**host.config, "model_revision": "wrong"}, case.schemas, host.allocation)
    with pytest.raises(ValueError, match="LOCAL_HOST_ALLOCATION_INVALID"):
        LocalActorHost(actor, channel, host.config, case.schemas, {"planner": True, "executor": 1})
    assert not channel.requests and len(channel.messages) == 1


def test_finish_ack_alone_does_not_qualify_completion_or_leak_evaluation(setup_host):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    channel.terminal_status = "FAILED"
    report = host.run()
    assert report["status"] == "FAILED"
    assert report["error"]["error_code"] == "LOCAL_HOST_CONTROLLER_TERMINAL_FAILED"
    assert report["terminal"]["status"] == "FAILED"
    assert report["native_evaluation"]["marker"] == "TERMINAL_EVALUATION_ONLY"
    assert report["host_attempts"][-1]["acknowledged"] is True
    assert channel.requests[-1]["kind"] == "finish"
    assert "TERMINAL_EVALUATION_ONLY" not in json.dumps(actor.histories)


def test_allocation_split_is_hidden_until_after_the_frozen_proposal(setup_host):
    discovery_prompts = []
    for allocation in ({"planner": 2, "executor": 0}, {"planner": 1, "executor": 1}, {"planner": 0, "executor": 2}):
        host, actor, channel, case = setup_host(allocation)
        report = host.run()
        assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
        before = [(name, content) for name, content in actor.prompts if content["host_state"] == "DISCOVERY"]
        assert before
        for name, content in before:
            assert "assigned_verification_calls" not in content
            assert content["view"]["allocated_verification_calls"] is None
            assert content["total_verification_budget"] == 2
        discovery_prompts.append(before)
        after = [(name, content) for name, content in actor.prompts if content["host_state"] != "DISCOVERY"]
        assert all(content["assigned_verification_calls"] == allocation[name] for name, content in after)
        proposal_index = next(index for index, request in enumerate(channel.requests) if request["kind"] == "proposal")
        allocation_index = next(index for index, request in enumerate(channel.requests) if request["kind"] == "allocate")
        assert proposal_index < allocation_index
    assert discovery_prompts[0] == discovery_prompts[1] == discovery_prompts[2]


@pytest.mark.parametrize("sender", ACTORS)
@pytest.mark.parametrize("share", [False, True])
def test_verification_message_schedules_private_peer_reply_without_extra_quota(setup_host, sender, share):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1}, verification_message_replies=True)
    recipient = "executor" if sender == "planner" else "planner"
    sent = False

    def override(name, content, action, index):
        nonlocal sent
        if content["host_state"] == "VERIFY" and name == sender and not sent:
            sent = True
            return actor.message(name, "request verification peer reply")
        if content["host_state"] == "VERIFY_REPLY":
            assert name == recipient and content["reply_recipient"] == sender
            return actor.message(name, "generated verification reply", [name + "_private_read"] if share else [])
        return action

    actor.override = override
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    assert report["verification_reply_slots"] == report["verification_replies_completed"] == 1
    assert report["pending_verification_reply"] is None
    assert report["verification_calls_used"] == {"planner": 1, "executor": 1}
    assert len([call for call in case.backend.calls if call["verification"]]) == 2
    reply_index, (name, reply) = next((i, row) for i, row in enumerate(actor.prompts) if row[1]["host_state"] == "VERIFY_REPLY")
    assert name == recipient
    assert reply["view"]["used_verification_calls"] == int(recipient == "planner")
    assert recipient + "_private_read" in reply["view"]["results"]
    assert sender + "_private_read" not in reply["view"]["results"]
    assert "PRIVATE_" + sender + "_private_read" not in json.dumps(reply)
    next_name, next_prompt = actor.prompts[reply_index + 1]
    assert next_name == sender and next_prompt["host_state"] == "VERIFY"
    assert (recipient + "_private_read" in next_prompt["view"]["results"]) is share
    assert next_prompt["view"]["used_verification_calls"] == 0
    assert report["host_attempt_count"] == len(actor.attempts) == case.turns
    assert "TERMINAL_EVALUATION_ONLY" not in json.dumps(actor.histories)
    reply_event = next(event for event in case.events if event.get("request", {}).get("text") == "generated verification reply")
    assert reply_event["phase_before"] == "verification"


@pytest.mark.parametrize("forbidden", ["read", "write", "proposal", "finish", "wrong_recipient"])
def test_verification_reply_rejects_nonreply_actions_without_native_calls(setup_host, forbidden):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, attempt_limit=5, verification_message_replies=True)

    def override(name, content, action, index):
        if content["host_state"] == "VERIFY":
            return actor.message(name, "please reply")
        if content["host_state"] == "VERIFY_REPLY":
            if forbidden in {"read", "write"}:
                return actor.tool(name, "forbidden_reply_tool", api=forbidden)
            if forbidden == "proposal":
                return actor.message(name, "forbidden proposal", kind="proposal")
            if forbidden == "finish":
                return {"actor": name, "kind": "finish"}
            return {**actor.message(name, "wrong recipient"), "recipient": name}
        return action

    actor.override = override
    report = host.run()
    assert report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["host_attempt_count"] == len(actor.attempts) == 5
    assert report["verification_reply_slots"] == 1 and report["verification_replies_completed"] == 0
    assert report["pending_verification_reply"] is None
    assert report["host_state"] == "VERIFY" and report["verification_reply_attempts"] == 1
    assert report["host_attempts"][-1]["status"] == "HOST_REJECTED"
    assert report["host_attempts"][-1]["error_code"] == "LOCAL_HOST_VERIFICATION_REPLY_MESSAGE_REQUIRED"
    assert report["host_attempts"][-1]["submitted"] is False
    assert case.turns == 4
    assert report["verification_calls_used"] == {"planner": 0, "executor": 0}
    assert not any(call["verification"] or call["api"] == "write" for call in case.backend.calls)
    assert report["native_evaluation"] is None and not case.finished
    assert report["cost_accounting"]["generations_with_token_usage"] == 5


def test_rejected_verification_reply_returns_to_verification_without_retry_or_implicit_sharing(setup_host):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, verification_message_replies=True)
    sent = False
    reply_attempts = 0

    def override(name, content, action, index):
        nonlocal sent, reply_attempts
        if content["host_state"] == "VERIFY" and not sent:
            sent = True
            return actor.message(name, "please reply")
        if content["host_state"] == "VERIFY_REPLY":
            reply_attempts += 1
            if reply_attempts == 1:
                return actor.message(name, "unseen share", ["executor_private_read"])
        return action

    actor.override = override
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    attempts = [row for row in report["host_attempts"] if row["host_state"] == "VERIFY_REPLY"]
    assert [row["status"] for row in attempts] == ["CONTROLLER_REJECTED"]
    assert attempts[0]["controller_response"]["error"] == "CANNOT_SHARE_UNSEEN_RESULT"
    assert report["verification_reply_slots"] == report["verification_reply_attempts"] == 1
    assert report["verification_replies_completed"] == 0 and report["pending_verification_reply"] is None
    reply_index = next(i for i, row in enumerate(actor.prompts) if row[1]["host_state"] == "VERIFY_REPLY")
    assert actor.prompts[reply_index + 1][0] == "executor"
    assert actor.prompts[reply_index + 1][1]["host_state"] == "VERIFY"
    assert report["verification_calls_used"] == {"planner": 0, "executor": 2}
    assert "executor_private_read" not in case.visible["planner"]


@pytest.mark.parametrize("failure_at", ["request", "reply"])
def test_unknown_verification_message_outcome_stops_without_resend(setup_host, monkeypatch, failure_at):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, verification_message_replies=True)
    sent = False

    def override(name, content, action, index):
        nonlocal sent
        if content["host_state"] == "VERIFY" and not sent:
            sent = True
            return actor.message(name, "verification request")
        return action

    actor.override = override
    original_send = channel.send
    failure_text = "verification request" if failure_at == "request" else "generated verification reply"

    def send(request):
        original_send(request)
        if request.get("text") == failure_text:
            channel.fail_receive = True

    monkeypatch.setattr(channel, "send", send)
    report = host.run()
    assert report["status"] == "FAILED" and report["unknown_native_outcome"] is True
    assert report["error"]["error_code"] == "LOCAL_HOST_CHANNEL_OUTCOME_UNKNOWN"
    assert len([request for request in channel.requests if request.get("text") == failure_text]) == 1
    assert channel.requests[-1]["text"] == failure_text
    assert report["verification_reply_slots"] == int(failure_at == "reply")
    assert report["verification_replies_completed"] == 0
    assert (report["pending_verification_reply"] is not None) is (failure_at == "reply")
    assert report["outstanding_request"]["kind"] == "message"
    assert report["verification_calls_used"] == {"planner": 0, "executor": 0}
    assert not any(call["verification"] or call["api"] == "write" for call in case.backend.calls)


def test_verification_reply_pending_when_total_generation_budget_ends(setup_host):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, attempt_limit=4, verification_message_replies=True)
    actor.override = lambda name, content, action, index: actor.message(name, "please reply") if content["host_state"] == "VERIFY" else action
    report = host.run()
    assert report["host_state"] == "VERIFY_REPLY"
    assert report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["host_attempt_count"] == 4 and report["verification_reply_slots"] == 1
    assert report["verification_replies_completed"] == 0
    assert not any(content["host_state"] == "VERIFY_REPLY" for name, content in actor.prompts)
    assert report["pending_verification_reply"]["recipient"] == "planner"


def test_verification_reply_default_retains_prompts_and_existing_execution_reply(setup_host):
    runs = []
    for enabled in [False, True]:
        host, actor, channel, case = setup_host({"planner": 1, "executor": 1}, verification_message_replies=enabled)
        report = host.run()
        assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
        assert report["planner_reply_slots"] == 1
        assert not any(content["host_state"] == "VERIFY_REPLY" for name, content in actor.prompts)
        if enabled:
            assert report["verification_reply_slots"] == report["verification_replies_completed"] == 0
        else:
            assert "verification_message_replies" not in report
            assert "verification_reply_slots" not in report
            assert "pending_verification_reply" not in report
        runs.append((actor.histories, report["host_attempts"]))
    assert runs[0] == runs[1]


def test_verification_reply_flag_requires_boolean_before_channel_use(setup_host):
    host, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    with pytest.raises(ValueError, match="LOCAL_HOST_VERIFICATION_REPLY_FLAG_INVALID"):
        LocalActorHost(actor, channel, host.config, case.schemas, host.allocation, verification_message_replies=1)
    assert not channel.requests


def test_malformed_verification_reply_consumes_one_attempt_then_returns_to_sender(setup_host):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, verification_message_replies=True)
    sent = False

    def override(name, content, action, index):
        nonlocal sent
        if content["host_state"] == "VERIFY" and not sent:
            sent = True
            return actor.message(name, "please reply")
        if content["host_state"] == "VERIFY_REPLY":
            return "unparsed reply retained as generated"
        return action

    actor.override = override
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    assert report["verification_reply_slots"] == report["verification_reply_attempts"] == 1
    assert report["verification_replies_completed"] == 0 and report["pending_verification_reply"] is None
    assert report["verification_calls_used"] == {"planner": 0, "executor": 2}
    attempt = next(row for row in report["host_attempts"] if row["host_state"] == "VERIFY_REPLY")
    assert attempt["status"] == "HOST_REJECTED" and attempt["submitted"] is False
    assert report["generation_attempts"][attempt["producer_attempt_index"]]["response"] == "unparsed reply retained as generated"


def test_controller_rejected_verification_request_does_not_schedule_reply(setup_host):
    host, actor, channel, case = setup_host({"planner": 0, "executor": 2}, verification_message_replies=True)
    sent = False

    def override(name, content, action, index):
        nonlocal sent
        if content["host_state"] == "VERIFY" and not sent:
            sent = True
            return actor.message(name, "unseen observation request", ["planner_private_read"])
        return action

    actor.override = override
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    assert report["verification_reply_slots"] == report["verification_reply_attempts"] == 0
    assert report["verification_replies_completed"] == 0 and report["pending_verification_reply"] is None
    assert report["host_attempts"][3]["status"] == "CONTROLLER_REJECTED"
    assert report["host_attempts"][3]["controller_response"]["error"] == "CANNOT_SHARE_UNSEEN_RESULT"


class VerifiedRoutingPrefix:
    """Only isolate live-branch routing; real replay is tested by the prefix suite."""
    native_checkpoint_sha256 = None

    def __init__(self, verified=True):
        self.verified = verified
        self.replays = 0

    def bind(self, actor, config):
        pass

    def expect_request(self, request, index):
        pass

    def observe_response(self, request, response, index):
        pass

    def check_progress(self, host):
        return False

    def generate(self, actor, name, prompt):
        self.replays += 1
        return actor.generate(name, prompt)

    def report(self):
        return {"status": "VERIFIED" if self.verified else "REPLAYING"}


def test_phase_decoder_live_routing_keeps_default_prompts_and_action_attempt_budget(setup_host):
    runs = []
    for enabled in (False, True):
        original, actor, channel, case = setup_host({"planner": 1, "executor": 1})
        host = LocalActorHost(actor, channel, original.config, case.schemas, original.allocation,
                              recorded_prefix=VerifiedRoutingPrefix(), phase_action_decoder=enabled)
        report = host.run()
        assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
        assert report["host_attempt_count"] == len(actor.action_schemas)
        assert all((item is not None) is enabled for item in actor.action_schemas)
        assert report.get("phase_action_decoder", False) is enabled
        if enabled:
            assert all("action_schema_sha256" in row for row in report["host_attempts"])
        runs.append((actor.prompts, actor.histories, report["host_attempt_count"], report["verification_calls_used"]))
    assert runs[0] == runs[1]


def test_phase_decoder_leaves_unverified_prefix_branch_unmodified(setup_host):
    original, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    prefix = VerifiedRoutingPrefix(verified=False)
    host = LocalActorHost(actor, channel, original.config, case.schemas, original.allocation,
                          recorded_prefix=prefix, phase_action_decoder=True)
    host.state = "DISCOVERY"
    record, request = host._generate("planner", case.view("planner"))
    assert prefix.replays == 1 and actor.action_schemas == [None]
    assert "action_schema_sha256" not in record and request["kind"] == "tool"


def test_phase_decoder_requires_prefix_and_boolean_flag_before_controller_use(setup_host):
    original, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    for value, prefix in ((True, None), (1, VerifiedRoutingPrefix())):
        with pytest.raises(ValueError, match="LOCAL_HOST_ACTION_DECODER_REQUIRES_PREFIX"):
            LocalActorHost(actor, channel, original.config, case.schemas, original.allocation,
                           recorded_prefix=prefix, phase_action_decoder=value)
    assert channel.requests == []

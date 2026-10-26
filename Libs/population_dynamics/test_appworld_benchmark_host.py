"""Generated-action fixtures for causal two-context control; no task/model data."""
from __future__ import annotations

from copy import deepcopy
import json
import random
from types import SimpleNamespace
import time

import pytest

from population_dynamics.appworld_benchmark_collaboration import ActorSnapshot, CALL_ID_DESCRIPTION
from population_dynamics.appworld_reference import NAME
from population_dynamics.appworld_benchmark_host import (
    CollaborationHost, FixedVerificationPolicy, HostBudget, ModelReply,
    UniformVerificationPolicy, full_visible_history, source_api_methods,
    PROMPT_PROJECTION_VERSION, prompt_visible_history,
)
from population_dynamics.appworld_benchmark_runner import _observe_model
from population_dynamics.appworld_benchmark_transport import LocalChatTransport
from population_dynamics.test_appworld_benchmark_collaboration import PersistentFixtureWorld, make_session


def tool(call_id, api, **arguments):
    return {"kind": "tool", "call_id": call_id, "app": "bank", "api": api, "arguments": arguments}


def message(text="Observed evidence", refs=()):
    return {"kind": "message", "text": text, "share_results": list(refs)}


def proposal(refs=()):
    return {"kind": "propose", "text": "Check the observed balance before executing.", "share_results": list(refs)}


class ScriptedGenerator:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def __call__(self, *, actor, messages, max_completion_tokens):
        self.calls.append(deepcopy({"actor": actor, "messages": messages, "limit": max_completion_tokens}))
        expected_actor, action = self.rows.pop(0)
        assert actor == expected_actor
        content = json.dumps(action) if isinstance(action, dict) else action
        return ModelReply({"role": "assistant", "content": content}, 3, True)


def make_host(generator, *, budget=None, policy=None, world=None, verification_budget=2, source_finished=None):
    world, history, session = make_session(verification_budget, world)
    host = CollaborationHost(
        session, generator, budget=budget or HostBudget(20, 100, 10),
        state_encoder=full_visible_history, allocation_policy=policy or FixedVerificationPolicy(1),
        source_finished=source_finished or (lambda: world.balance == 7),
    )
    return world, history, session, host


def test_full_two_context_execution_couples_allocation_observations_messages_and_actions():
    generate = ScriptedGenerator([
        ("planner", tool("initial", "balance")), ("executor", message("Please check the plan")),
        ("planner", proposal(("planner/initial",))), ("planner", tool("pcheck", "balance")),
        ("planner", message(refs=("planner/pcheck",))), ("executor", tool("echeck", "fail")),
        ("executor", message("Verification failed", ("executor/echeck",))),
        ("executor", tool("write", "add", amount=7)),
    ])
    world, _, session, host = make_host(generate)
    result = host.run()
    assert result["status"] == "completed" and result["native_task_completed"] is True
    assert result["model_call_count"] == 8 and result["resolved_completion_tokens"] == 24
    assert world.calls == [("balance",), ("balance",), ("fail",), ("add", 7)]
    assert session.verification_used == {"planner": 1, "executor": 1}
    first_executor = json.loads(generate.calls[1]["messages"][1]["content"])
    assert "planner/initial" not in str(first_executor)
    executor_verification = json.loads(generate.calls[5]["messages"][1]["content"])
    assert "planner/pcheck" in str(executor_verification)
    for export in result["state_exports"]:
        assert "pcheck" not in str(export["representation"])
        assert "echeck" not in str(export["representation"])
        assert not any(event["kind"] == "verification_allocation" for event in export["representation"]["events"])
    assert session.attempts[2]["observation"]["ok"] is False
    with pytest.raises(ValueError, match="repeat"):
        host.run()


@pytest.mark.parametrize("planner_calls,verifier", [(2, "planner"), (0, "executor")])
def test_matched_total_allocation_changes_which_actor_really_calls_the_native_world(planner_calls, verifier):
    generate = ScriptedGenerator([
        ("planner", proposal()), (verifier, tool("check1", "balance")),
        (verifier, tool("check2", "balance")), (verifier, message(refs=(f"{verifier}/check2",))),
        ("executor", tool("write", "add", amount=7)),
    ])
    world, _, session, host = make_host(generate, policy=FixedVerificationPolicy(planner_calls))
    result = host.run()
    verification = [row for row in result["native_attempts"] if row["verification"]]
    assert [row["actor"] for row in verification] == [verifier, verifier]
    assert sum(session.allocation.values()) == sum(session.verification_used.values()) == 2
    assert world.balance == 7


def test_unused_allocated_slots_are_not_reported_as_executed_calls():
    generate = ScriptedGenerator([
        ("planner", proposal()), ("planner", {"kind": "verification_done"}),
        ("planner", message("No additional useful checks")),
        ("executor", tool("write", "add", amount=7)),
    ])
    _, _, session, host = make_host(generate, policy=FixedVerificationPolicy(2))
    result = host.run()
    assert result["status"] == "completed"
    assert session.allocation == {"planner": 2, "executor": 0}
    assert sum(session.verification_used.values()) == 0
    assert len(result["native_attempts"]) == 1


def test_write_during_verification_is_rejected_before_native_mutation_and_counts_model_cost():
    generate = ScriptedGenerator([
        ("planner", proposal()), ("planner", tool("badwrite", "add", amount=99)),
        ("planner", tool("check1", "balance")), ("planner", {"kind": "verification_done"}),
        ("planner", message()), ("executor", tool("write", "add", amount=7)),
    ])
    world, _, session, host = make_host(generate, policy=FixedVerificationPolicy(2))
    result = host.run()
    assert world.calls == [("balance",), ("add", 7)] and world.balance == 7
    assert result["generations"][1]["action_status"] == "rejected"
    assert result["resolved_completion_tokens"] == 18 and session.verification_used["planner"] == 1


@pytest.mark.parametrize("model_repairs_id", [False, True])
def test_invalid_call_id_guidance_reaches_next_request_without_rewriting_or_extra_budget(model_repairs_id):
    next_id = "call_1" if model_repairs_id else "1"
    generate = ScriptedGenerator([
        ("planner", tool("1", "add", amount=99)),
        ("planner", tool(next_id, "add", amount=7)),
    ])
    world, _, session, host = make_host(generate, budget=HostBudget(2, 100, 10))
    result = host.run()
    assert result["model_call_count"] == 2 and result["resolved_completion_tokens"] == 6
    assert result["generations"][0]["action_status"] == "rejected"
    assert result["generations"][0]["action"]["call_id"] == "1"
    assert result["generations"][0]["rejection"] == CALL_ID_DESCRIPTION
    for call in generate.calls:
        instruction = call["messages"][0]["content"]
        assert CALL_ID_DESCRIPTION in instruction
        assert NAME.pattern in instruction and "ASCII" in instruction
        assert "1 to 64" in instruction and '"call_id":"call_1"' in instruction
    next_events = json.loads(generate.calls[1]["messages"][1]["content"])["events"]
    rejections = [event for event in next_events if event["kind"] == "action_rejection"]
    assert len(rejections) == 1 and rejections[0]["payload"]["reason"] == CALL_ID_DESCRIPTION
    if model_repairs_id:
        assert result["status"] == "completed" and world.balance == 7
        assert world.calls == [("add", 7)]
        assert len(session.attempts) == 1 and session.attempts[0]["result_id"] == "planner/call_1"
        assert result["generations"][1]["action_status"] == "accepted"
    else:
        assert result["status"] == "agent_failure" and result["stop_reason"] == "model_budget_exhausted"
        assert world.calls == [] and session.attempts == []
        assert result["generations"][1]["action_status"] == "rejected"


def test_bad_json_exhausts_generation_budget_without_fake_native_actions():
    generate = ScriptedGenerator([("planner", "not json"), ("planner", "[]")])
    world, _, _, host = make_host(generate, budget=HostBudget(2, 100, 10))
    result = host.run()
    assert result["status"] == "agent_failure" and result["stop_reason"] == "model_budget_exhausted"
    assert result["resolved_completion_tokens"] == 6 and world.calls == []
    assert all(row["action_status"] == "rejected" for row in result["generations"])


@pytest.mark.parametrize("path", [["missing"], ["items", 5], ["items", "bad_index"]])
def test_invalid_visible_result_paths_are_action_rejections_before_native_execution(path):
    class ListWorld(PersistentFixtureWorld):
        def read_balance(self):
            return {"balance": self.balance, "items": [1]}
    generate = ScriptedGenerator([
        ("planner", tool("initial", "balance")), ("executor", message()),
        ("planner", tool("bad_reference", "add", amount={"$result": "planner/initial", "path": path})),
        ("planner", tool("valid_write", "add", amount=7)),
    ])
    world, _, session, host = make_host(generate, world=ListWorld())
    result = host.run()
    assert result["status"] == "completed" and world.balance == 7
    assert result["generations"][2]["action_status"] == "rejected"
    assert "invalid visible-result reference path" in result["generations"][2]["rejection"]
    assert len(session.attempts) == 2 and result["resolved_completion_tokens"] == 12


@pytest.mark.parametrize("tokens,resolved,error,expected_tokens,reason", [
    (None, False, None, 0, "model_error_or_unresolved_usage"),
    (4, True, "provider error", 4, "model_error_or_unresolved_usage"),
    (11, True, None, 11, "provider_exceeded_completion_limit"),
])
def test_model_failures_keep_known_cost_and_cannot_execute_their_returned_action(tokens, resolved, error, expected_tokens, reason):
    def generate(**kwargs):
        return ModelReply({"role": "assistant", "content": json.dumps(tool("write", "add", amount=99))}, tokens, resolved, error)
    world, _, _, host = make_host(generate)
    result = host.run()
    assert result["stop_reason"] == reason and result["resolved_completion_tokens"] == expected_tokens
    assert world.calls == []


def test_native_unknown_after_mutation_stops_the_host_without_another_generation():
    class UnresolvedWorld(PersistentFixtureWorld):
        def execute(self, code):
            super().execute(code)
            raise OSError("fixture interrupted return")
    generate = ScriptedGenerator([("planner", tool("write", "add", amount=7))])
    world, _, _, host = make_host(generate, world=UnresolvedWorld())
    result = host.run()
    assert result["status"] == "unknown" and world.balance == 7
    assert len(generate.calls) == 1 and result["native_attempts"][0]["status"] == "unknown"


def test_policy_failure_after_state_delivery_is_not_treated_as_an_action_retry():
    generate = ScriptedGenerator([("planner", proposal())])
    _, _, _, host = make_host(generate, policy=lambda **kwargs: {"planner": 3, "executor": -1})
    with pytest.raises(RuntimeError, match="state export or allocation"):
        host.run()
    assert len(generate.calls) == 1 and len(host.state_exports) == 2


def test_source_completion_reader_error_after_native_write_is_not_a_rejected_action():
    generate = ScriptedGenerator([("planner", tool("write", "add", amount=7))])
    def broken_reader():
        raise ValueError("fixture native completion read failed")
    world, _, _, host = make_host(generate, source_finished=broken_reader)
    with pytest.raises(ValueError, match="completion read"):
        host.run()
    assert world.balance == 7 and len(generate.calls) == 1


def test_stopping_without_a_verification_boundary_is_retained_explicitly():
    generate = ScriptedGenerator([("planner", message()), ("executor", {"kind": "finish"})])
    _, _, _, host = make_host(generate)
    result = host.run()
    assert result["status"] == "completed" and result["stop_reason"] == "executor_stopped"
    assert result["native_task_completed"] is False and result["allocation"] is None


def test_collection_draws_use_the_original_membership_stream_and_no_outcomes():
    for seed in (41, 42, 43):
        generator = random.Random(seed)
        expected = [generator.randrange(3) for _ in range(20)]
        actual = [UniformVerificationPolicy(seed, index)(state_reports={"oracle-looking-string": index}, total_budget=2)["planner"] for index in range(20)]
        assert actual == expected and len(set(actual)) > 1
        assert UniformVerificationPolicy(seed, 4).metadata(2)["assignment_probability"] == 1 / 3
    with pytest.raises(ValueError, match="41, 42 or 43"):
        UniformVerificationPolicy(44, 0)


def test_public_schema_method_is_read_from_the_source_record_not_inferred_from_api_name():
    docs = {"bank": {"get_but_writes": {"app_name": "bank", "api_name": "get_but_writes", "method": "POST"}}}
    assert source_api_methods(docs) == {("bank", "get_but_writes"): "POST"}
    docs["bank"]["get_but_writes"]["app_name"] = "different"
    with pytest.raises(ValueError, match="identity/method"):
        source_api_methods(docs)


def test_actual_elapsed_measurements_survive_frozen_python_timers(monkeypatch):
    # AppWorld freezes task date/time. Emulate the timer replacements used by
    # freezegun, while the OS monotonic clock and a real short wait remain live.
    monkeypatch.setattr(time, "perf_counter", lambda: 123.0)
    monkeypatch.setattr(time, "monotonic", lambda: 123.0)
    class WaitingWorld(PersistentFixtureWorld):
        def execute(self, code):
            time.sleep(0.01)
            return super().execute(code)
    _, _, session = make_session(world=WaitingWorld())
    session.execute("planner", "observed", "bank", "balance", {})
    assert session.attempts[0]["native_wall_seconds"] >= 0.008

    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            time.sleep(0.01)
            return json.dumps({"choices": [{"message": {"role": "assistant", "content": "fixture"}}],
                               "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}).encode()
    transport = LocalChatTransport("http://127.0.0.1:1/v1")
    transport.opener = SimpleNamespace(open=lambda *args, **kwargs: Response())
    class Model:
        model = "fixture"
        def generate(self, messages):
            response = self.lm_call(model=self.model, messages=messages)
            return {"content": response["choices"][0]["message"]["content"],
                    "standardized_usage": SimpleNamespace(dict=lambda: {"tokens": {"input_cache_miss": 5, "input_cache_hit": 0, "output": 3}})}
    model, calls = Model(), []
    _observe_model(model, "planner", transport, calls, [])
    model.generate(messages=[{"role": "user", "content": "synthetic"}])
    assert transport.records[0]["http_wall_seconds"] >= 0.008
    assert calls[0]["transport_wall_seconds"] >= 0.008 and calls[0]["usage_status"] == "resolved"


def _expand_projected_history(projected):
    """Independent reconstruction checks the declared reference meaning."""
    restored = deepcopy(projected)
    assert restored.pop("prompt_projection") == PROMPT_PROJECTION_VERSION
    for position, event in enumerate(restored["events"]):
        payload = event["payload"]
        if event["kind"] == "state_report_delivered":
            events = payload["representation"].get("events")
            if isinstance(events, dict) and set(events) == {"$history_ref"}:
                reference = events["$history_ref"]
                assert reference == {"kind": "prefix", "end_exclusive": position}
                payload["representation"]["events"] = deepcopy(restored["events"][:position])
        elif event["kind"] in {"message_sent", "message_delivered"}:
            for result_id, value in payload["shared_results"].items():
                if not isinstance(value, dict) or set(value) != {"$history_ref"}:
                    continue
                reference = value["$history_ref"]
                assert reference["kind"] == "observation"
                assert 0 <= reference["event_index"] < position
                target = restored["events"][reference["event_index"]]
                for key in reference["path"]:
                    target = target[key]
                payload["shared_results"][result_id] = deepcopy(target)
    return restored


def _json_identity(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def test_prompt_projection_round_trip_preserves_histories_controller_and_visibility():
    _, history, session = make_session(2)
    session.execute("planner", "initial", "bank", "balance", {})
    hidden = history.snapshot("executor").as_model_input()
    assert "planner/initial" not in str(prompt_visible_history(history.snapshot("executor")))
    sent = history.send_message("planner", "executor", "Read this observation", share_results=("planner/initial",))
    history.deliver_message(sent)
    sent = history.send_message("executor", "planner", "Echo", share_results=("planner/initial",))
    history.deliver_message(sent)
    snapshots = {actor: history.snapshot(actor) for actor in ("planner", "executor")}
    for actor, snapshot in snapshots.items():
        history.deliver_state_report(actor, full_visible_history(snapshot))
    controller_before = deepcopy(history.controller_reports())
    session.allocate(FixedVerificationPolicy(1))
    for actor in ("planner", "executor"):
        snapshot = history.snapshot(actor)
        original = snapshot.as_model_input()
        projected = prompt_visible_history(snapshot)
        assert _json_identity(_expand_projected_history(projected)) == _json_identity(original)
        assert _json_identity(snapshot.as_model_input()) == _json_identity(original)
        assert len(_json_identity(projected)) < len(_json_identity(original))
        report = next(event for event in projected["events"] if event["kind"] == "state_report_delivered")
        assert report["payload"]["representation"]["events"]["$history_ref"]["kind"] == "prefix"
    executor = prompt_visible_history(history.snapshot("executor"))
    received = next(event for event in executor["events"] if event["kind"] == "message_delivered")
    assert "$history_ref" not in received["payload"]["shared_results"]["planner/initial"]
    echoed = next(event for event in executor["events"] if event["kind"] == "message_sent")
    reference = echoed["payload"]["shared_results"]["planner/initial"]["$history_ref"]
    assert reference["event_index"] == received["index"]
    assert reference["path"] == ["payload", "shared_results", "planner/initial"]
    assert history.controller_reports() == controller_before
    assert "planner/initial" not in str(hidden)


@pytest.mark.parametrize("original,changed", [
    ({"value": True}, {"value": 1}),
    ({"value": 1}, {"value": 1.0}),
    ({"value": None}, {}),
    ({"raw": "a\n"}, {"raw": "a"}),
    ({"value": {"$history_ref": {"kind": "prefix"}}}, {"value": "different"}),
])
def test_prompt_projection_preserves_changed_observations_with_same_result_id(original, changed):
    events = [
        {"index": 0, "kind": "tool_observation", "payload": {"result_id": "planner/r", "observation": original}},
        {"index": 1, "kind": "message_sent", "payload": {
            "sender": "planner", "recipient": "executor", "text": "A message",
            "shared_results": {"planner/r": changed},
        }},
    ]
    snapshot = ActorSnapshot("planner", tuple(json.dumps(row) for row in events))
    projected = prompt_visible_history(snapshot)
    assert _json_identity(projected["events"]) == _json_identity(events)


def test_prompt_projection_does_not_merge_different_ids_or_use_future_observations():
    observation = {"value": {"$history_ref": {"kind": "prefix"}}, "ok": True}
    events = [
        {"index": 0, "kind": "message_sent", "payload": {"shared_results": {"planner/r": observation}}},
        {"index": 1, "kind": "tool_observation", "payload": {"result_id": "planner/r", "observation": observation}},
        {"index": 2, "kind": "message_sent", "payload": {"shared_results": {"planner/other": observation}}},
        {"index": 3, "kind": "message_sent", "payload": {"shared_results": {"planner/r": observation}}},
    ]
    snapshot = ActorSnapshot("planner", tuple(json.dumps(row) for row in events))
    projected = prompt_visible_history(snapshot)
    assert projected["events"][:3] == events[:3]
    assert projected["events"][3]["payload"]["shared_results"]["planner/r"]["$history_ref"]["event_index"] == 1
    assert _json_identity(_expand_projected_history(projected)) == _json_identity(snapshot.as_model_input())


@pytest.mark.parametrize("change", ["actor", "events", "extra_field", "learned"])
def test_prompt_projection_preserves_noncopy_state_reports(change):
    _, history, _ = make_session(2)
    representation = full_visible_history(history.snapshot("planner"))
    if change == "actor":
        representation["actor"] = "executor"
    elif change == "events":
        representation["events"][0]["payload"]["text"] += " different"
    elif change == "extra_field":
        representation["extra"] = None
    else:
        representation = {"representation": "learned_state", "state": [1, 2, 3]}
    history.deliver_state_report("planner", representation)
    projected = prompt_visible_history(history.snapshot("planner"))
    assert projected["events"][-1]["payload"]["representation"] == representation


def test_host_records_projection_and_keeps_controller_state_and_verification_skip():
    generate = ScriptedGenerator([
        ("planner", proposal()), ("planner", {"kind": "verification_done"}),
        ("planner", message("No further useful check")), ("executor", tool("write", "add", amount=7)),
    ])
    _, history, session, host = make_host(generate, policy=FixedVerificationPolicy(2))
    result = host.run()
    assert result["prompt_projection"] == PROMPT_PROJECTION_VERSION
    assert all(row["prompt_projection"] == PROMPT_PROJECTION_VERSION for row in result["generations"])
    assert session.allocation == {"planner": 2, "executor": 0}
    assert session.verification_used == {"planner": 0, "executor": 0}
    assert result["model_call_count"] == 4
    assert result["resolved_completion_tokens"] == 12
    for call in generate.calls:
        assert PROMPT_PROJECTION_VERSION in call["messages"][0]["content"]
        assert json.loads(call["messages"][1]["content"])["prompt_projection"] == PROMPT_PROJECTION_VERSION
    for actor, representation in history.controller_reports().items():
        assert isinstance(representation["events"], list)
        assert representation["actor"] == actor
    assert "$history_ref" in generate.calls[1]["messages"][1]["content"]

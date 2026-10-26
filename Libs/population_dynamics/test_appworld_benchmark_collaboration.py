"""Synthetic execution/visibility checks, with no AppWorld task or model access."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import traceback

import pytest

from population_dynamics.appworld_benchmark_collaboration import (
    CALL_ID_DESCRIPTION, CollaborationHistory, PersistentToolSession, UnknownNativeOutcome,
)


class PersistentFixtureWorld:
    """Execute the actual generated wrapper against small stateful fixture APIs."""

    def __init__(self):
        self.balance = 0
        self.calls = []
        self.namespace = {"apis": SimpleNamespace(bank=SimpleNamespace(
            add=self.add, balance=self.read_balance, fail=self.fail, mutate_then_fail=self.mutate_then_fail,
        ))}

    def add(self, amount):
        self.calls.append(("add", amount))
        self.balance += amount
        return {"balance": self.balance}

    def read_balance(self):
        self.calls.append(("balance",))
        return {"balance": self.balance}

    def fail(self):
        self.calls.append(("fail",))
        raise ValueError("fixture API failure")

    def mutate_then_fail(self):
        self.balance += 7
        raise RuntimeError("fixture failure after mutation")

    def execute(self, code):
        stream = io.StringIO()
        try:
            with redirect_stdout(stream):
                exec(code, self.namespace)
        except Exception:
            return "Execution failed. Traceback:\n" + traceback.format_exc()
        return stream.getvalue()


def make_session(budget=2, world=None):
    history = CollaborationHistory("Synthetic instruction, with no native task identity.")
    world = world or PersistentFixtureWorld()
    session = PersistentToolSession(world, history, {
        ("bank", "add"): "POST", ("bank", "balance"): "GET", ("bank", "fail"): "GET",
        ("bank", "mutate_then_fail"): "POST",
    }, verification_budget=budget)
    return world, history, session


def reports(history, planner=1, executor=1):
    history.deliver_state_report("planner", {"need": planner})
    history.deliver_state_report("executor", {"need": executor})


def equal_allocation(**kwargs):
    assert kwargs["total_budget"] == 2
    return {"planner": 1, "executor": 1}


def test_actual_generated_wrapper_keeps_one_world_and_no_automatic_api_action():
    world, history, session = make_session()
    assert world.calls == []
    first = session.execute("planner", "credit", "bank", "add", {"amount": 19})
    second = session.execute("executor", "credit", "bank", "add", {"amount": 23})
    assert first["result_id"] != second["result_id"]
    assert second["observation"] == {"ok": True, "result": {"balance": 42}, "raw_output": '{"balance": 42}\n'}
    assert world.calls == [("add", 19), ("add", 23)]
    assert world.balance == 42
    assert all(row["status"] == "returned" and row["native_wall_seconds"] >= 0 for row in session.attempts)


def test_private_history_delivery_and_future_prefix_are_separate():
    _, history, session = make_session()
    before_peer_call = history.snapshot("executor")
    result = session.execute("planner", "read", "bank", "balance", {})
    assert history.snapshot("executor") == before_peer_call
    with pytest.raises(ValueError, match="RESULT_NOT_VISIBLE"):
        history.resolve_arguments("executor", {"amount": {"$result": result["result_id"], "path": ["balance"]}})
    message_id = history.send_message("planner", "executor", "Observed balance.", share_results=(result["result_id"],))
    assert history.snapshot("executor") == before_peer_call
    history.deliver_message(message_id)
    assert history.resolve_arguments("executor", {"amount": {"$result": result["result_id"], "path": ["balance"]}}) == {"amount": 0}
    assert before_peer_call.as_model_input()["events"] == [{
        "index": 0, "kind": "task_instruction", "payload": {"text": "Synthetic instruction, with no native task identity."},
    }]
    received = history.snapshot("executor").as_model_input()["events"]
    assert [event["index"] for event in received] == [0, 1]
    assert received[-1]["kind"] == "message_delivered"
    with pytest.raises(ValueError, match="already delivered"):
        history.deliver_message(message_id)


def test_share_only_visible_results_and_preserve_api_failure_observations():
    world, history, session = make_session()
    with pytest.raises(ValueError, match="unseen"):
        history.send_message("executor", "planner", "Guessed result.", share_results=("planner/secret",))
    observed = session.execute("planner", "failed", "bank", "fail", {})
    assert observed["observation"]["ok"] is False
    assert observed["observation"]["error_type"] == "NativeExecutionError"
    raw = session.attempts[0]["raw_output"]
    assert raw.startswith("Execution failed. Traceback:\n") and "ValueError: fixture API failure" in raw
    assert observed["observation"]["raw_output"] == observed["observation"]["error"] == raw
    message = history.send_message("planner", "executor", "This query failed.", share_results=(observed["result_id"],))
    history.deliver_message(message)
    with pytest.raises(ValueError, match="SUCCESSFUL_RESULT"):
        history.resolve_arguments("executor", {"x": {"$result": observed["result_id"], "path": []}})
    assert world.calls == [("fail",)]
    assert session.attempts[0]["status"] == "returned"


def test_report_policy_can_change_actual_verification_recipient_without_private_history():
    world, history, session = make_session()
    history.record_generated_action("planner", {"private_thought": "must remain local"})
    reports(history, planner=0, executor=2)
    seen = []

    def policy(*, state_reports, total_budget):
        seen.append((state_reports.copy(), total_budget))
        assert "private_thought" not in str(state_reports)
        return {actor: state_reports[actor]["need"] for actor in state_reports}

    assert session.allocate(policy) == {"planner": 0, "executor": 2}
    with pytest.raises(ValueError, match="exhausted"):
        session.verify("planner", "blocked", "bank", "balance", {})
    session.verify("executor", "allowed", "bank", "balance", {})
    session.verify("executor", "failed", "bank", "fail", {})
    assert session.verification_used == {"planner": 0, "executor": 2}
    assert world.calls == [("balance",), ("fail",)]
    assert seen == [({"planner": {"need": 0}, "executor": {"need": 2}}, 2)]
    assert session.attempts[-1]["observation"]["ok"] is False
    with pytest.raises(ValueError, match="exhausted"):
        session.verify("executor", "third", "bank", "balance", {})


@pytest.mark.parametrize("proposal", [
    {"planner": 1}, {"planner": 1, "executor": 2}, {"planner": -1, "executor": 3},
    {"planner": True, "executor": 1}, {"planner": 1.0, "executor": 1},
])
def test_bad_allocation_rejects_before_calls(proposal):
    world, history, session = make_session()
    reports(history)
    with pytest.raises(ValueError, match="exact fixed budget"):
        session.allocate(lambda **kwargs: proposal)
    assert session.allocation is None and world.calls == []


def test_allocation_requires_explicit_both_reports_and_is_fixed_after_choice():
    world, history, session = make_session()
    with pytest.raises(ValueError, match="both actors"):
        session.allocate(equal_allocation)
    reports(history)
    session.allocate(equal_allocation)
    with pytest.raises(ValueError, match="already fixed"):
        session.allocate(equal_allocation)
    with pytest.raises(ValueError, match="already received"):
        history.deliver_state_report("planner", {"need": 2})
    with pytest.raises(ValueError, match="GET"):
        session.verify("planner", "write", "bank", "add", {"amount": 99})
    assert world.calls == [] and session.verification_used == {"planner": 0, "executor": 0}


def test_copies_protect_snapshots_state_reports_controls_and_returned_records():
    world, history, session = make_session()
    state = {"vector": [1.0]}
    history.deliver_state_report("planner", state)
    history.deliver_state_report("executor", {"vector": [2.0]})
    prefix = history.snapshot("planner")
    state["vector"][0] = 999
    prefix.as_model_input()["events"].clear()

    def policy(**kwargs):
        kwargs["state_reports"]["planner"]["vector"][0] = -99
        return {"planner": 1, "executor": 1}

    session.allocate(policy)["planner"] = 99
    result = session.verify("planner", "read", "bank", "balance", {})
    result["observation"]["result"]["balance"] = 999
    session.attempts[0]["observation"]["ok"] = False
    assert history.controller_reports()["planner"] == {"vector": [1.0]}
    assert session.allocation == {"planner": 1, "executor": 1}
    assert session.attempts[0]["observation"] == {"ok": True, "result": {"balance": 0}, "raw_output": '{"balance": 0}\n'}
    assert len(prefix.as_model_input()["events"]) == 2
    assert world.balance == 0


@pytest.mark.parametrize("error_kind", ["exception", "nontext", "interrupt"])
def test_unknown_execution_stops_every_later_native_call_and_preserves_attempt(error_kind):
    class InterruptedWorld(PersistentFixtureWorld):
        def execute(self, code):
            result = super().execute(code)
            if error_kind == "exception":
                raise OSError("fixture connection lost after mutation")
            if error_kind == "interrupt":
                raise KeyboardInterrupt()
            return None

    world, history, session = make_session(world=InterruptedWorld())
    reports(history)
    expected_error = KeyboardInterrupt if error_kind == "interrupt" else UnknownNativeOutcome
    with pytest.raises(expected_error):
        session.execute("executor", "first", "bank", "add", {"amount": 7})
    for call in (
        lambda: session.execute("executor", "second", "bank", "add", {"amount": 7}),
        lambda: session.verify("planner", "third", "bank", "balance", {}),
        lambda: session.allocate(equal_allocation),
    ):
        with pytest.raises(UnknownNativeOutcome):
            call()
    assert world.balance == 7 and world.calls == [("add", 7)]
    assert len(session.attempts) == 1 and session.attempts[0]["status"] == "unknown"
    assert session.attempts[0]["native_wall_seconds"] >= 0


def test_unparsed_native_output_retains_raw_text_without_inventing_success():
    class VerboseWorld(PersistentFixtureWorld):
        def execute(self, code):
            return super().execute(code) + "fixture diagnostic"

    world, history, session = make_session(world=VerboseWorld())
    response = session.execute("planner", "first", "bank", "add", {"amount": 7})
    assert response["observation"]["ok"] is None
    assert response["observation"]["raw_output"] == '{"balance": 7}\nfixture diagnostic'
    assert session.attempts[0]["status"] == "returned" and world.balance == 7
    with pytest.raises(ValueError, match="SUCCESSFUL_RESULT"):
        history.resolve_arguments("planner", {"x": {"$result": response["result_id"], "path": []}})
    session.execute("executor", "next", "bank", "add", {"amount": 1})
    assert world.balance == 8


def test_native_error_keeps_partial_mutation_and_allows_a_subsequent_observation():
    world, history, session = make_session()
    result = session.execute("executor", "write", "bank", "mutate_then_fail", {})
    assert result["observation"]["ok"] is False
    assert "fixture failure after mutation" in result["observation"]["raw_output"]
    assert world.balance == 7
    observed = session.execute("executor", "observe", "bank", "balance", {})
    assert observed["observation"]["result"] == {"balance": 7}
    assert history.snapshot("executor").as_model_input()["events"][-1]["payload"]["observation"]["result"] == {"balance": 7}


@pytest.mark.parametrize("call_id", [
    "1", 1, "", "_call", "call/1", "call 1", "é", "call_1\n", "a" * 65,
])
def test_invalid_call_ids_keep_native_execution_closed_and_explain_the_existing_rule(call_id):
    world, _, session = make_session()
    with pytest.raises(ValueError) as error:
        session.execute("planner", call_id, "bank", "add", {"amount": 99})
    assert str(error.value) == CALL_ID_DESCRIPTION
    assert world.calls == [] and world.balance == 0 and session.attempts == []


@pytest.mark.parametrize("call_id", ["a", "call_1", "call-1", "Z" * 64])
def test_valid_call_ids_preserve_native_execution_and_result_identity(call_id):
    world, _, session = make_session()
    result = session.execute("planner", call_id, "bank", "add", {"amount": 7})
    assert result["result_id"] == f"planner/{call_id}"
    assert result["observation"]["ok"] is True
    assert world.calls == [("add", 7)] and world.balance == 7
    assert len(session.attempts) == 1 and session.attempts[0]["result_id"] == result["result_id"]


def test_duplicate_call_id_cannot_repeat_an_action_but_other_actor_has_distinct_identity():
    world, _, session = make_session()
    session.execute("planner", "same", "bank", "add", {"amount": 1})
    with pytest.raises(ValueError, match="cannot be reused"):
        session.execute("planner", "same", "bank", "add", {"amount": 2})
    session.execute("executor", "same", "bank", "add", {"amount": 3})
    assert world.balance == 4 and len(session.attempts) == 2


def test_arguments_remain_data_when_they_contain_quotes_newlines_and_unicode():
    world, _, session = make_session()
    received = []
    world.namespace["apis"].bank.add = lambda amount: received.append(amount) or amount
    text = "引号'\"\n); raise RuntimeError('must remain data') #"
    result = session.execute("executor", "literal", "bank", "add", {"amount": text})
    assert received == [text] and result["observation"]["result"] == text


@pytest.mark.parametrize("app,api", [("bank-bad", "balance"), ("bank", "class"), ("bank", "read()"), ([], "balance")])
def test_invalid_source_identifiers_never_reach_native_execution(app, api):
    world, _, session = make_session()
    with pytest.raises(ValueError, match="source identifiers"):
        session.execute("planner", "read", app, api, {})
    assert world.calls == [] and session.attempts == []


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), (1, 2), {1: "coerced"}, object()])
def test_non_json_actor_state_is_not_silently_coerced(bad_value):
    _, history, _ = make_session()
    with pytest.raises(ValueError, match="finite JSON"):
        history.deliver_state_report("planner", {"value": bad_value})


def test_zero_verification_budget_is_explicit_and_does_not_block_ordinary_native_actions():
    world, history, session = make_session(budget=0)
    reports(history, planner=0, executor=0)
    session.allocate(lambda **kwargs: {"planner": 0, "executor": 0})
    with pytest.raises(ValueError, match="exhausted"):
        session.verify("planner", "read", "bank", "balance", {})
    session.execute("executor", "write", "bank", "add", {"amount": 3})
    assert world.balance == 3

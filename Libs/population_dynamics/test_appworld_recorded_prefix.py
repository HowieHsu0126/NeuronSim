"""Recorded-prefix fixtures use real host/controller decisions and no inference."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

import pytest
import yaml

from .appworld_collaboration import ACTORS, CollaborationCase, generation_telemetry
from .appworld_local_actor import LocalTransformersActor, model_identity
from .appworld_local_host import LocalActorHost
from .appworld_recorded_prefix import RecordedPrefix, RecordedPrefixError, actor_evidence_table
from .test_appworld_local_host import Backend, ControllerChannel, ScriptedActor


class JsonTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return json.dumps(messages, ensure_ascii=False)


class LoginBackend(Backend):
    def call(self, call_id, app, api, arguments, verification):
        result = super().call(call_id, app, api, arguments, verification)
        if api == "login":
            return {"ok": True, "result": {"access_token": "FIXTURE_TOKEN", "token_type": "Bearer"}}
        return result


class RecordingChannel(ControllerChannel):
    def __init__(self, case):
        super().__init__(case)
        self.transport, self.recording_request = [], None
        self.change_view = False

    def send(self, request):
        self.recording_request = deepcopy(request)
        super().send(request)

    def receive(self):
        response = super().receive()
        if self.recording_request is not None:
            if self.change_view and self.recording_request.get("kind") == "view":
                response["result"]["discovery_calls_used"] += 1
            self.transport.append({"request": self.recording_request, "response": deepcopy(response)})
            self.recording_request = None
        return response


class ReplayThenScriptedActor(ScriptedActor):
    replay_generation = LocalTransformersActor.replay_generation

    def __init__(self):
        super().__init__()
        self.tokenizer = JsonTokenizer()
        self.session_ids = {name: str(uuid.uuid4()) for name in ACTORS}
        self.live_calls = 0

    def generate(self, actor, user_text):
        self.live_calls += 1
        return super().generate(actor, user_text)


def build_case(actor):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_collaboration.yaml").read_text())
    config.update(provider="local_transformers", model_revision=model_identity(actor.config), max_actor_turns=30)
    schemas = {"spotify": {"read": {"method": "GET", "parameters": []},
                            "write": {"method": "POST", "parameters": []},
                            "login": {"method": "POST", "parameters": []}}}
    case = CollaborationCase(config, schemas, LoginBackend())
    return config, schemas, case, RecordingChannel(case)


@pytest.fixture
def archive():
    actor = ScriptedActor()
    message, tool = actor.message, actor.tool
    actions = [message("planner", "observing"), tool("executor", "executor_profile"),
               tool("planner", "planner_search"), tool("executor", "executor_secret"),
               message("planner", "search assertion", ["planner_search"]), "malformed JSON",
               tool("planner", "planner_details"), message("executor", "credentials assertion"),
               message("planner", "details assertion", ["planner_details"]),
               tool("executor", "executor_login_spotify", api="login")]
    actor.override = lambda name, content, action, count: actions[count - 1] if count <= 10 else action
    config, schemas, case, channel = build_case(actor)
    report = LocalActorHost(actor, channel, config, schemas, {"planner": 1, "executor": 1}).run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
    return deepcopy(actor.attempts), deepcopy(report), deepcopy(channel.transport)


def branch(archive, *, table=False, actor=None):
    actor = ReplayThenScriptedActor() if actor is None else actor
    prefix = RecordedPrefix(*archive)
    config, schemas, case, channel = build_case(actor)
    host = LocalActorHost(actor, channel, config, schemas, {"planner": 1, "executor": 1},
                          recorded_prefix=prefix, evidence_table_after_prefix=table)
    return host, actor, case, channel, prefix


def test_ten_attempt_semantic_login_pause_restores_history_and_consumed_budgets(archive):
    host, actor, case, channel, prefix = branch(archive)
    report = host.run(pause_at_prefix_boundary=True)
    assert report["status"] == "PAUSED_AT_VERIFIED_PREFIX", report
    assert actor.live_calls == 0 and len(actor.attempts) == 10
    assert case.turns == 9 and case.discovery_used == {"planner": 2, "executor": 3}
    assert case.verification_used == {"planner": 0, "executor": 0} and case.allocated is None
    assert host._select_actor() == "planner" and host.phase == "discovery"
    assert host.controller_request_count == 28 and len(case.backend.calls) == 5
    assert host.attempts[5]["status"] == "HOST_REJECTED"
    assert actor.histories["executor"][6]["content"] == "malformed JSON"
    assert len(actor.histories["executor"]) == len(actor.histories["planner"]) == 11
    assert "FIXTURE_TOKEN" in json.dumps(case.view("executor"))
    assert "FIXTURE_TOKEN" not in json.dumps(case.view("planner"))
    assert prefix.verified and report["recorded_prefix"]["checked_exchanges"] == 28
    assert set(actor.session_ids.values()).isdisjoint(prefix.source_contexts.values())
    for original, replayed in zip(archive[0], actor.attempts):
        assert replayed["source_generation"] == original
        assert replayed["prompt"] == original["prompt"] and replayed["response"] == original["response"]
        assert replayed["usage"] is None and replayed["inference_seconds"] is None
    for row in channel.transport:
        if "generation" in row["request"]:
            receipt = row["request"]["generation"]
            assert len(receipt) == 6
            assert receipt["session_id"] == actor.session_ids[row["request"]["actor"]]
    assert report["new_generation_attempt_count"] == 0
    assert report["replayed_prefix_generation_count"] == 10
    assert report["cost_accounting"]["provider_token_usage"] is None
    assert report["cost_accounting"]["provider_inference_seconds"] is None
    assert report["cost_accounting"]["provider_telemetry_status"] == "NOT_RUN_RECORDED_PREFIX_ONLY"
    assert report["recorded_prefix"]["historical_prefix_inference_seconds"] == pytest.approx(.1)


def test_prefix_then_live_continuation_keeps_twenty_remaining_attempts(archive):
    host, actor, case, _, _ = branch(archive)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    actor.override = lambda name, content, action, count: actor.message(name, "still observing")
    report = host.run()
    assert actor.live_calls == 20 and report["host_attempt_count"] == 30
    assert report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert report["new_generation_attempt_count"] == 20
    assert report["cost_accounting"]["provider_inference_seconds"] == pytest.approx(.2)
    assert case.turns == 29
    with pytest.raises(RuntimeError, match="LOCAL_HOST_ALREADY_RUN"):
        host.run()


def test_prefix_can_switch_to_live_without_a_pause(archive):
    host, actor, _, _, prefix = branch(archive)
    report = host.run()
    assert prefix.verified and report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    assert actor.live_calls == report["new_generation_attempt_count"] > 0
    assert report["generation_attempt_count"] == actor.live_calls + 10


@pytest.mark.parametrize("mutation", ["native_view", "prompt", "host_feedback"])
def test_any_recorded_replay_mismatch_stops_before_live_generation(archive, mutation):
    inputs = deepcopy(archive)
    if mutation == "prompt":
        inputs[0][2]["prompt"] += "changed"
        inputs[1]["generation_attempts"][2]["prompt"] += "changed"
        for row in inputs[2]:
            if row["request"].get("call_id") == "planner_search":
                row["request"]["generation"]["prompt"] += "changed"
    if mutation == "host_feedback":
        inputs[1]["host_attempts"][5]["error_code"] = "different rejection"
        with pytest.raises(RecordedPrefixError, match="MALFORMED_ATTEMPT_NOT_PRESERVED"):
            RecordedPrefix(*inputs)
        return
    host, actor, _, channel, prefix = branch(inputs)
    channel.change_view = mutation == "native_view"
    report = host.run()
    assert report["status"] == "FAILED" and not prefix.verified
    assert actor.live_calls == 0
    with pytest.raises(RuntimeError, match="LOCAL_HOST_ALREADY_RUN"):
        host.run()


def test_incomplete_or_unsuccessful_login_never_changes_the_cut(archive):
    inputs = deepcopy(archive)
    with pytest.raises(RecordedPrefixError, match="SOURCE_INCOMPLETE"):
        RecordedPrefix(inputs[0][:9], inputs[1], inputs[2])
    row = inputs[2][26]
    row["response"] = {"ok": True, "result": {"ok": False, "error": "authentication failed"}}
    inputs[1]["host_attempts"][9]["controller_response"] = deepcopy(row["response"])
    with pytest.raises(RecordedPrefixError, match="LOGIN_NOT_SUCCESSFUL"):
        RecordedPrefix(*inputs)


def test_verified_pause_cannot_resume_with_changed_private_feedback(archive):
    host, actor, _, _, _ = branch(archive)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    host.feedback["planner"] = {"ok": True, "result": "changed"}
    with pytest.raises(RecordedPrefixError, match="PAUSED_STATE_CHANGED"):
        host.run()
    assert actor.live_calls == 0


def test_tokenizer_only_actor_reconstructs_but_refuses_live_generation(archive):
    config = ScriptedActor().config
    actor = LocalTransformersActor.for_prefix_replay(JsonTokenizer(), config,
                {name: name + " private context" for name in ACTORS})
    host, actor, _, _, _ = branch(archive, actor=actor)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    assert actor.runtime()["live_generation_allowed"] is False
    with pytest.raises(ValueError, match="PREFIX_ONLY_LIVE_GENERATION_FORBIDDEN"):
        actor.generate("planner", "not allowed")
    report = host.run()
    assert report["status"] == "FAILED"
    assert report["host_attempts"][-1]["error_code"] == "LOCAL_ACTOR_PREFIX_ONLY_LIVE_GENERATION_FORBIDDEN"
    assert len(actor.attempts) == 10


def test_post_boundary_table_does_not_change_prefix_or_reveal_peer_native_data(archive):
    host, actor, case, _, _ = branch(archive, table=True)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    assert all("actor_evidence_table" not in row["prompt"] for row in actor.attempts)
    view = case.view("planner")
    table = actor_evidence_table(view, host._own_native_calls("planner", view))
    assert "PRIVATE_executor_secret" not in json.dumps(table)
    assert "FIXTURE_TOKEN" not in json.dumps(table)
    assert all(row["source_actor"] == "planner" and row["api"] == "spotify.read" for row in table["native_results"])
    assert "executor_login_spotify" not in json.dumps(table)
    assert all(row["returned_field_names"] == ["private_value"] for row in table["native_results"])
    assert table["peer_assertions"] == [{"sender": "executor", "text": "credentials assertion", "shared_result_ids": []}]
    report = host.run()
    assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE", report
    first_actor, content = actor.prompts[0]
    assert first_actor == "planner" and content["actor_evidence_table"] == table
    assert content["view"] == view
    assert "assigned_verification_calls" not in content


def test_fresh_branch_ids_and_required_transport_are_enforced(archive):
    actor = ReplayThenScriptedActor()
    actor.session_ids = {name: "context_" + name for name in ACTORS}
    with pytest.raises(RecordedPrefixError, match="FRESH_BRANCH_REQUIRED"):
        branch(archive, actor=actor)
    with pytest.raises(RecordedPrefixError, match="LOGIN_ACK_MISSING"):
        RecordedPrefix(archive[0], archive[1], archive[2][:27])


def test_real_transport_telemetry_is_verified_without_becoming_native_view(archive):
    inputs = deepcopy(archive)
    for row in inputs[2]:
        if "generation" in row["request"]:
            row["provider_telemetry"] = generation_telemetry(row["request"]["generation"])
    host, actor, _, _, prefix = branch(inputs)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    assert actor.live_calls == 0 and prefix.verified
    row = next(row for row in inputs[2] if "provider_telemetry" in row)
    row["provider_telemetry"]["inference_seconds"] += 1.
    with pytest.raises(RecordedPrefixError, match="SOURCE_TELEMETRY_MISMATCH"):
        RecordedPrefix(*inputs)


def test_shared_results_do_not_inherit_unseen_api_or_owner(archive):
    host, _, case, _, _ = branch(archive, table=True)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    view = deepcopy(case.view("planner"))
    view["results"]["shared_without_request_metadata"] = {"ok": True, "result": {"value": 1}}
    table = actor_evidence_table(view, host._own_native_calls("planner", view))
    row = next(r for r in table["native_results"] if r["call_id"] == "shared_without_request_metadata")
    assert row["source_actor"] is None and row["api"] is None


def test_required_native_checkpoint_cannot_be_omitted_by_controller(archive):
    host, actor, _, _, prefix = branch(archive)
    prefix.native_checkpoint_sha256 = 'a'*64
    report = host.run(pause_at_prefix_boundary=True)
    assert report['status']=='FAILED' and report['error']['error_code']=='LOCAL_HOST_READY_INVALID'
    assert actor.live_calls==0 and actor.attempts==[]

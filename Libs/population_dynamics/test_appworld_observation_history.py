"""Lossless actor-local view references; fixtures never load or query a model."""
from copy import deepcopy
import json

import pytest
import yaml

from .appworld_local_host import (LocalActorHost, OBSERVATION_DELTA_FORMAT,
    encode_observation_prompt, reconstruct_observation_prompt)
from .appworld_recorded_prefix import RecordedPrefix
from .test_appworld_local_host import ScriptedActor, setup_host
from .test_appworld_recorded_prefix import archive, build_case, ReplayThenScriptedActor


def serialize(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def prompt(actor="planner"):
    return {"actor": actor, "host_state": "VERIFY", "instruction": "unchanged instruction",
            "view": {"actor": actor, "task": {"instruction": "完整原始任务"}, "phase": "verification",
                     "results": {"seen": {"ok": True, "result": {"payload": "observed " * 300,
                                                                   "values": [True, 1, 1., -0.0, 0.]}}},
                     "inbox": [{"text": "duplicate", "share_results": ["seen"]}] * 2,
                     "allocated_verification_calls": 1, "used_verification_calls": 0,
                     "discovery_calls_used": 2},
            "last_feedback": {"ok": False, "error": "recorded rejection"},
            "public_api_schemas": {"source": {"read": {"method": "GET", "parameters": ["exact_schema"]}}},
            "actor_evidence_table": {"seen": "unchanged auxiliary evidence"},
            "total_verification_budget": 2}


def messages(content):
    return [{"role": "system", "content": "private system"},
            {"role": "user", "content": serialize(content)},
            {"role": "assistant", "content": '{"kind":"message","text":"past action"}'}]


def test_delta_chain_exactly_reconstructs_types_messages_feedback_and_counters():
    original = prompt()
    history = messages(original)
    for turn in range(1, 8):
        current = deepcopy(original)
        current["view"]["results"][f"result_{turn}"] = {"ok": False, "error": "observed failure"}
        current["view"]["inbox"] += [{"text": "duplicate", "share_results": ["seen"]}] * 2
        current["view"]["used_verification_calls"] = turn
        current["last_feedback"] = {"ok": False, "error": str(turn)}
        previous = deepcopy(history)
        raw = serialize(current)
        encoded = encode_observation_prompt(raw, history, "planner")
        assert json.loads(encoded)["observation_history_delta"]["format"] == OBSERVATION_DELTA_FORMAT
        assert reconstruct_observation_prompt(encoded, history, "planner") == raw
        assert history == previous
        assert len(encoded.encode()) < len(raw.encode())
        history += [{"role": "user", "content": encoded},
                    {"role": "assistant", "content": f"unchanged action {turn}"}]
        original = current
    decoded = json.loads(reconstruct_observation_prompt(history[-2]["content"], history[:-2], "planner"))
    assert len(decoded["view"]["inbox"]) == 16
    assert [type(x) for x in decoded["view"]["results"]["seen"]["result"]["values"]] == [bool, int, float, float, float]


@pytest.mark.parametrize("change", ["bool_to_int", "int_to_float", "signed_zero", "deleted_result", "removed_message", "message_order"])
def test_nonimmutable_view_falls_back_to_full_original_prompt(change):
    original = prompt()
    original["view"]["inbox"].append({"text": "distinct", "share_results": []})
    current = deepcopy(original)
    if change == "bool_to_int": current["view"]["results"]["seen"]["result"]["values"][0] = 1
    elif change == "int_to_float": current["view"]["results"]["seen"]["result"]["values"][1] = 1.
    elif change == "signed_zero": current["view"]["results"]["seen"]["result"]["values"][3] = 0.
    elif change == "deleted_result": del current["view"]["results"]["seen"]
    elif change == "removed_message": current["view"]["inbox"].pop()
    else: current["view"]["inbox"].reverse()
    raw = serialize(current)
    assert encode_observation_prompt(raw, messages(original), "planner") == raw


def test_no_prior_view_and_small_views_remain_full():
    content = prompt()
    content["view"].update(results={}, inbox=[])
    raw = serialize(content)
    assert encode_observation_prompt(raw, [], "planner") == raw
    assert encode_observation_prompt(raw, messages(content), "planner") == raw


@pytest.mark.parametrize("index", [True, -1, 2, 3, 100])
def test_reference_must_be_a_strictly_prior_user_message(index):
    original = prompt()
    history = messages(original)
    encoded = json.loads(encode_observation_prompt(serialize(original), history, "planner"))
    encoded["observation_history_delta"]["base_message_index"] = index
    with pytest.raises(ValueError, match="OBSERVATION_REFERENCE_INVALID"):
        reconstruct_observation_prompt(serialize(encoded), history, "planner")


def test_reference_cannot_import_other_actor_or_replace_existing_result():
    original = prompt()
    history = messages(original)
    encoded = encode_observation_prompt(serialize(original), history, "planner")
    foreign = messages(prompt("executor"))
    with pytest.raises(ValueError, match="OBSERVATION_ACTOR_MISMATCH"):
        reconstruct_observation_prompt(encoded, foreign, "planner")
    with pytest.raises(ValueError, match="OBSERVATION_ACTOR_MISMATCH"):
        encode_observation_prompt(serialize(original), foreign, "planner")
    changed = json.loads(encoded)
    changed["observation_history_delta"]["results_added"]["seen"] = {"hidden": "replacement"}
    with pytest.raises(ValueError, match="OBSERVATION_REFERENCE_INVALID"):
        reconstruct_observation_prompt(serialize(changed), history, "planner")


class DecodingActor(ScriptedActor):
    def action(self, actor, content):
        original = reconstruct_observation_prompt(serialize(content), self.histories[actor], actor)
        return super().action(actor, json.loads(original))


class DecodingReplayActor(ReplayThenScriptedActor):
    def action(self, actor, content):
        original = reconstruct_observation_prompt(serialize(content), self.histories[actor], actor)
        return super().action(actor, json.loads(original))


def test_opt_in_preserves_controller_actions_private_views_and_archived_raw_prompts(setup_host):
    runs = []
    for enabled in (False, True):
        seed_host, _, channel, case = setup_host({"planner": 1, "executor": 1})
        actor = DecodingActor()
        original_call = case.backend.call
        def large_observation(*args, _call=original_call, **kwargs):
            result = _call(*args, **kwargs)
            result["result"]["payload"] = "actor-visible fixture " * 300
            return result
        case.backend.call = large_observation
        host = LocalActorHost(actor, channel, seed_host.config, seed_host.schemas, seed_host.allocation,
                              lossless_observation_history=enabled)
        config_before = deepcopy(actor.config)
        report = host.run()
        assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
        assert actor.config == config_before
        assert report["verification_calls_used"] == {"planner": 1, "executor": 1}
        assert report["host_attempt_limit"] == 30
        runs.append((actor, report, channel))
    baseline, encoded = runs
    assert [{k: v for k, v in row.items() if k != "generation"} for row in baseline[2].requests] == [
        {k: v for k, v in row.items() if k != "generation"} for row in encoded[2].requests]
    assert "observation_history_encoding" not in baseline[1]
    assert encoded[1]["observation_history_encoding"] == OBSERVATION_DELTA_FORMAT
    assert any("observation_history_delta" in content for _, content in encoded[0].prompts)
    for attempt, (_, content) in zip(encoded[1]["host_attempts"], baseline[0].prompts):
        assert attempt["unencoded_user_prompt"] == serialize(content)
    for actor in ("planner", "executor"):
        prior = []
        restored = []
        for message in encoded[0].histories[actor]:
            message = deepcopy(message)
            if message["role"] == "user":
                message["content"] = reconstruct_observation_prompt(message["content"], prior, actor)
            restored.append(message)
            prior.append(encoded[0].histories[actor][len(prior)])
        assert restored == baseline[0].histories[actor]
        hidden = "PRIVATE_" + ("executor" if actor == "planner" else "planner") + "_private_read"
        assert hidden not in serialize(restored)
        assert "TERMINAL_EVALUATION_ONLY" not in serialize(restored)


def test_recorded_prefix_stays_byte_exact_and_live_encoding_keeps_twenty_attempts(archive):
    actor = DecodingReplayActor()
    prefix = RecordedPrefix(*archive)
    config, schemas, case, channel = build_case(actor)
    host = LocalActorHost(actor, channel, config, schemas, {"planner": 1, "executor": 1},
                          recorded_prefix=prefix, lossless_observation_history=True)
    assert host.run(pause_at_prefix_boundary=True)["status"] == "PAUSED_AT_VERIFIED_PREFIX"
    assert all("unencoded_user_prompt" not in attempt for attempt in host.attempts)
    for source, replayed in zip(archive[0], actor.attempts):
        assert source["prompt"] == replayed["prompt"] and source["response"] == replayed["response"]
    actor.override = lambda name, content, action, count: actor.message(name, "still observing")
    report = host.run()
    assert report["new_generation_attempt_count"] == 20 and actor.live_calls == 20
    assert report["host_attempt_count"] == report["host_attempt_limit"] == 30
    assert report["error"]["error_code"] == "LOCAL_HOST_ATTEMPT_LIMIT_REACHED"
    assert case.verification_used == {"planner": 0, "executor": 0} and case.allocated is None
    assert all("unencoded_user_prompt" in attempt for attempt in host.attempts[10:])


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_forwards_explicit_opt_in_without_changing_actor_caps(tmp_path, monkeypatch, enabled):
    from Libs.scripts import run_appworld_local_collaboration as entry
    fixture_actor = ScriptedActor()
    config, schemas, _, _ = build_case(fixture_actor)
    inputs = {"actor-config": fixture_actor.config, "collaboration-config": config,
              "allocation": {"planner": 1, "executor": 1}, "api-schemas": schemas,
              "metadata": {}, "controller-command-json": ["fixture-controller"]}
    args = []
    for name, value in inputs.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        args += ["--" + name, str(path)]
    args += ["--model-directory", str(tmp_path), "--output", str(tmp_path / "output")]
    if enabled: args.append("--lossless-observation-history")
    observed = {}
    class FakeActor:
        attempts = []
        def runtime(self): return {"fixture": True}
    class FakeChannel:
        def close_and_wait(self):
            return {"exit_code": 0, "request_outcome_unresolved": False, "termination_signals": []}
    def host(actor, channel, collaboration, supplied_schemas, allocation, **kwargs):
        observed.update(kwargs)
        observed["collaboration"] = collaboration
        raise RuntimeError("STOP_AFTER_FORWARDING")
    monkeypatch.setattr(entry, "verify_snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(entry, "LocalTransformersActor", lambda *args, **kwargs: FakeActor())
    monkeypatch.setattr(entry, "ProcessJsonChannel", lambda *args, **kwargs: FakeChannel())
    monkeypatch.setattr(entry, "LocalActorHost", host)
    assert entry.main(args) == 1
    assert observed["lossless_observation_history"] is enabled
    assert observed["collaboration"] == config
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert report["error"] == "STOP_AFTER_FORWARDING"

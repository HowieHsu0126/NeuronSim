"""Bounded message generation through the executed parser and tokenizer API."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from .appworld_action_decoder import (ActionSchemaDecoder, build_phase_action_schema,
                                     validate_action_decoder_bounds)
from .test_appworld_action_decoder import CharacterTokenizer, PUBLIC_SCHEMAS, check_tokens, schema
from .test_appworld_local_actor import cpu_actor, actor_config
from .test_appworld_local_host import setup_host, VerifiedRoutingPrefix
from .appworld_local_host import LocalActorHost


BOUNDS = {"message_text_max_length": 256, "max_consecutive_whitespaces": 4}


def bounded_schema(actor="executor", state="VERIFY", limit=256):
    view = {"actor": actor, "phase": "discovery" if state == "DISCOVERY" else "verification",
            "allocated_verification_calls": 1, "used_verification_calls": 0,
            "discovery_calls_used": 0, "results": {}, "inbox": []}
    return build_phase_action_schema(actor, state, view, PUBLIC_SCHEMAS,
                                    discovery_call_limit=3, message_text_max_length=limit)


@pytest.mark.parametrize("bad", [None, {}, {**BOUNDS, "extra": 1},
    {**BOUNDS, "message_text_max_length": True}, {**BOUNDS, "max_consecutive_whitespaces": 0}])
def test_invalid_bounds_fail_before_generation(bad):
    with pytest.raises(ValueError, match="BOUNDS_INVALID"):
        validate_action_decoder_bounds(bad)


def test_only_message_text_receives_the_explicit_bound():
    original = schema()
    bounded = bounded_schema()
    expected = deepcopy(original)
    for branch in expected["anyOf"]:
        if branch["properties"]["kind"]["enum"] == ["message"]:
            branch["properties"]["text"]["maxLength"] = 256
    assert bounded == expected
    assert original == schema()


def test_actual_callback_rejects_long_text_and_accepts_unicode_and_field_orders():
    tokenizer = CharacterTokenizer(extra_tokens=[r"\u", r"\u00", r"pre\u"])
    adapter = ActionSchemaDecoder(tokenizer)
    for text in ("", "x" * 32, 'é中🙂 quote" slash\\ newline\n', "x" * 256, "x" * 257):
        action = {"actor": "executor", "kind": "message", "recipient": "planner",
                  "text": text, "share_results": []}
        for reverse in (False, True):
            ordered = dict(reversed(list(action.items()))) if reverse else action
            for escaped in (False, True):
                accepted, metadata = check_tokens((tokenizer, adapter), bounded_schema(),
                                                  json.dumps(ordered, ensure_ascii=escaped))
                assert accepted == (len(text) <= 256)
                assert metadata["callback_failures"] == 0


def saturate_message(tokenizer, adapter, *, actor, kind, text_first=False):
    """Prefer another repeated-letter token until the real mask rejects it."""
    metadata = {}
    state = "DISCOVERY" if kind == "proposal" else "VERIFY"
    callback = adapter.prepare(bounded_schema(actor, state), 512, metadata,
                               max_consecutive_whitespaces=4)
    prompt = tokenizer.encode("Return exactly one JSON action.", add_special_tokens=False)
    sent = list(prompt)

    def emit(text):
        for token in tokenizer.encode(text, add_special_tokens=False):
            assert token in callback(0, torch.tensor(sent, dtype=torch.long)), text
            sent.append(token)

    recipient = "executor" if actor == "planner" else "planner"
    fixed = {"actor": actor, "kind": kind, "recipient": recipient, "share_results": []}
    prefix = '{"text":"' if text_first else json.dumps(fixed)[:-1] + ',"text":"'
    emit(prefix)
    preferred = sorted([(len(text), token) for token, text, _ in adapter.tokenizer_data.regular_tokens
                        if text and set(text) == {"a"}], reverse=True)
    assert preferred
    continuations = 0
    while len(sent) - len(prompt) < 512:
        allowed = callback(0, torch.tensor(sent, dtype=torch.long))
        token = next((token for _, token in preferred if token in allowed), None)
        if token is None:
            break
        sent.append(token)
        continuations += 1
    else:
        raise AssertionError("bounded text still exhausted the 512-token envelope")
    emit('"' + ("," + json.dumps(fixed)[1:] if text_first else "}"))
    assert tokenizer.eos_token_id in callback(0, torch.tensor(sent, dtype=torch.long))
    completion = tokenizer.decode(sent[len(prompt):], skip_special_tokens=True)
    parsed = json.loads(completion)
    assert parsed == {**fixed, "text": "a" * 256}
    assert len(sent) - len(prompt) + 1 <= 512
    assert metadata["callback_failures"] == 0
    return {"actor": actor, "kind": kind, "text_first": text_first,
            "repeated_text_tokens": continuations, "completion_tokens_with_eos": len(sent)-len(prompt)+1,
            "metadata": metadata}


def test_repeated_text_policy_forces_structural_continuation():
    tokenizer = CharacterTokenizer()
    adapter = ActionSchemaDecoder(tokenizer)
    for actor, kind in (("executor", "message"), ("planner", "proposal")):
        for text_first in (False, True):
            saturate_message(tokenizer, adapter, actor=actor, kind=kind, text_first=text_first)


def test_whitespace_limit_and_unbounded_default_use_the_same_parser():
    tokenizer = CharacterTokenizer()
    adapter = ActionSchemaDecoder(tokenizer)
    prompt = tokenizer.encode("prompt", add_special_tokens=False)
    space = tokenizer.encode(" ", add_special_tokens=False)[0]
    for limit in (None, 4):
        metadata = {}
        callback = adapter.prepare(bounded_schema(), 512, metadata, max_consecutive_whitespaces=limit)
        sent = list(prompt)
        for _ in range(4):
            assert space in callback(0, torch.tensor(sent, dtype=torch.long))
            sent.append(space)
        assert (space in callback(0, torch.tensor(sent, dtype=torch.long))) is (limit is None)
        assert metadata["callback_failures"] == 0


def test_actor_keeps_unrepaired_response_and_budget_with_bounded_callback(cpu_actor):
    response = json.dumps({"actor": "executor", "kind": "message", "recipient": "planner",
                           "text": "unchanged model text", "share_results": []})
    cpu_actor.model.responses = [response]
    config = deepcopy(cpu_actor.config)
    receipt = cpu_actor.generate("executor", "private observation", action_schema=bounded_schema(),
                                action_decoder_max_whitespace=4)
    assert receipt["response"] == response == cpu_actor.histories["executor"][-1]["content"]
    assert cpu_actor.config == config
    assert cpu_actor.model.calls[-1]["max_new_tokens"] == config["generation"]["max_new_tokens"]
    assert cpu_actor.attempts[-1]["action_decoder"]["parser_settings"]["max_consecutive_whitespaces"] == 4


def test_host_routes_bounds_after_prefix_without_changing_prompts_or_default(setup_host):
    outputs = []
    for bounds in (None, BOUNDS):
        original, actor, channel, case = setup_host({"planner": 1, "executor": 1})
        options = []
        generate = actor.generate
        def capture(name, prompt, *, action_schema=None, action_decoder_max_whitespace=None):
            options.append(action_decoder_max_whitespace)
            return generate(name, prompt, action_schema=action_schema)
        actor.generate = capture
        host = LocalActorHost(actor, channel, original.config, case.schemas, original.allocation,
            recorded_prefix=VerifiedRoutingPrefix(), phase_action_decoder=True, action_decoder_bounds=bounds)
        report = host.run()
        assert report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
        assert all(value == (None if bounds is None else 4) for value in options)
        assert report.get("action_decoder_bounds") == bounds
        outputs.append((actor.prompts, actor.histories, report["host_attempt_count"]))
    assert outputs[0] == outputs[1]
    original, actor, channel, case = setup_host({"planner": 1, "executor": 1})
    prefix = VerifiedRoutingPrefix(verified=False)
    host = LocalActorHost(actor, channel, original.config, case.schemas, original.allocation,
                         recorded_prefix=prefix, phase_action_decoder=True, action_decoder_bounds=BOUNDS)
    host.state = "DISCOVERY"
    record, request = host._generate("planner", case.view("planner"))
    assert prefix.replays == 1 and actor.action_schemas == [None]
    assert "action_schema_sha256" not in record


@pytest.mark.skipif("APPWORLD_DECODER_TOKENIZER_DIRECTORY" not in os.environ,
                    reason="requires explicitly staged pinned tokenizer assets")
def test_repeated_text_policy_with_pinned_qwen_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(Path(os.environ["APPWORLD_DECODER_TOKENIZER_DIRECTORY"]),
                                               local_files_only=True, trust_remote_code=False)
    adapter = ActionSchemaDecoder(tokenizer)
    cases = [saturate_message(tokenizer, adapter, actor=actor, kind=kind, text_first=first)
             for actor, kind in (("executor", "message"), ("planner", "proposal"))
             for first in (False, True)]
    report = {"status": "PASS_TARGETED_BOUNDED_TEXT_GENERATION", "cases": cases,
              "model_inference_calls": 0, "native_task_calls": 0,
              "global_token_budget_completion_guaranteed": False}
    if "APPWORLD_DECODER_BOUNDS_REPORT" in os.environ:
        Path(os.environ["APPWORLD_DECODER_BOUNDS_REPORT"]).write_text(json.dumps(report, indent=2)+"\n")

"""Exercise the pinned LMFE token callback with small real tokenizer vocabularies."""
from copy import deepcopy
import json

import pytest
import torch
from transformers import PreTrainedTokenizer

from . import appworld_action_decoder as decoder


PUBLIC_SCHEMAS = {
    "spotify": {"read": {"method": "GET"}, "details": {"method": "GET"}, "write": {"method": "POST"}, "login": {"method": "POST"}},
    "other": {"scan": {"method": "GET"}, "read": {"method": "DELETE"}},
}


class CharacterTokenizer(PreTrainedTokenizer):
    """A real Transformers tokenizer API, without loading model or tokenizer assets."""
    def __init__(self, extra_tokens=()):
        self.extra_tokens = sorted(extra_tokens, key=len, reverse=True)
        characters = list(dict.fromkeys(chr(i) for i in range(32, 127))) + ["\n", "\t", "\r", "é", "中", "🙂", *self.extra_tokens]
        self.vocabulary = {"<eos>": 0, **{value: index + 1 for index, value in enumerate(characters)}}
        self.reverse = {index: value for value, index in self.vocabulary.items()}
        super().__init__(eos_token="<eos>", unk_token="<eos>", clean_up_tokenization_spaces=False)

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def get_vocab(self):
        return dict(self.vocabulary)

    def _tokenize(self, text):
        pieces = []
        while text:
            piece = next((token for token in self.extra_tokens if text.startswith(token)), text[0])
            pieces.append(piece)
            text = text[len(piece):]
        return pieces

    def _convert_token_to_id(self, token):
        return self.vocabulary.get(token, 0)

    def _convert_id_to_token(self, index):
        return self.reverse[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


@pytest.fixture
def token_decoder():
    tokenizer = CharacterTokenizer()
    return tokenizer, decoder.ActionSchemaDecoder(tokenizer)


def schema(actor="executor", state="VERIFY", *, used=0, discovery_used=0):
    phase = "discovery" if state == "DISCOVERY" else "verification" if state in {"VERIFY", "VERIFY_REPLY"} else "execution"
    view = {"actor": actor, "phase": phase, "allocated_verification_calls": 1,
            "used_verification_calls": used, "discovery_calls_used": discovery_used,
            "results": {}, "inbox": []}
    return decoder.build_phase_action_schema(actor, state, view, PUBLIC_SCHEMAS, discovery_call_limit=3,
                                              reply_recipient=("planner" if actor == "executor" else "executor") if state == "VERIFY_REPLY" else None)


def tool(actor="executor", app="spotify", api="read", arguments=None):
    return {"actor": actor, "kind": "tool", "call_id": actor + "_chosen_by_model", "app": app, "api": api,
            "arguments": {} if arguments is None else arguments}


def message(actor="executor", kind="message", recipient=None):
    return {"actor": actor, "kind": kind, "recipient": recipient or ("planner" if actor == "executor" else "executor"),
            "text": "model-selected content", "share_results": []}


def check_tokens(token_decoder, action_schema, text, *, max_new_tokens=512):
    tokenizer, adapter = token_decoder
    metadata = {}
    callback = adapter.prepare(action_schema, max_new_tokens, metadata)
    sent = torch.tensor(tokenizer.encode("prompt", add_special_tokens=False), dtype=torch.long)
    for token in tokenizer.encode(text, add_special_tokens=False):
        if token not in callback(0, sent):
            return False, metadata
        sent = torch.cat((sent, torch.tensor([token], dtype=torch.long)))
    return tokenizer.eos_token_id in callback(0, sent), metadata


@pytest.mark.parametrize("actor,state,action,allowed", [
    ("executor", "DISCOVERY", tool(api="login"), True),
    ("executor", "DISCOVERY", tool(api="write"), False),
    ("planner", "DISCOVERY", message("planner", kind="proposal"), True),
    ("executor", "DISCOVERY", message(kind="proposal"), False),
    ("executor", "VERIFY", tool(), True),
    ("executor", "VERIFY", tool(api="details"), True),
    ("planner", "VERIFY", tool("planner"), True),
    ("executor", "VERIFY", tool(api="login"), False),
    ("executor", "VERIFY", tool(api="write"), False),
    ("executor", "VERIFY", {"actor": "executor", "kind": "finish"}, False),
    ("executor", "VERIFY", tool(app="other", api="scan"), True),
    ("executor", "VERIFY", tool(app="spotify", api="scan"), False),
    ("executor", "VERIFY", tool(app="other", api="read"), False),
    ("executor", "EXECUTE", tool(api="write"), True),
    ("executor", "EXECUTE", tool(app="other", api="read"), True),
    ("executor", "EXECUTE", tool(), False),
    ("executor", "EXECUTE", tool(api="login"), False),
    ("executor", "EXECUTE", {"actor": "executor", "kind": "finish"}, True),
    ("planner", "PLANNER_HANDOFF", message("planner"), True),
    ("planner", "PLANNER_HANDOFF", tool("planner"), False),
    ("planner", "PLANNER_REPLY", message("planner"), True),
    ("planner", "VERIFY_REPLY", message("planner"), True),
    ("executor", "VERIFY_REPLY", message(), True),
    ("executor", "VERIFY_REPLY", message(recipient="executor"), False),
    ("executor", "VERIFY_REPLY", tool(), False),
])
def test_real_callback_restricts_existing_phase_kinds_and_source_api_pairs(token_decoder, actor, state, action, allowed):
    accepted, metadata = check_tokens(token_decoder, schema(actor, state), json.dumps(action))
    assert accepted is allowed
    assert metadata["callback_calls"] > 0 and metadata["callback_failures"] == 0


@pytest.mark.parametrize("state,kwargs", [("VERIFY", {"used": 1}), ("DISCOVERY", {"discovery_used": 3})])
def test_exhausted_actor_quota_keeps_messages_without_adding_tool_allowance(token_decoder, state, kwargs):
    restricted = schema(state=state, **kwargs)
    assert check_tokens(token_decoder, restricted, json.dumps(tool()))[0] is False
    assert check_tokens(token_decoder, restricted, json.dumps(message()))[0] is True


def test_arrays_unicode_escapes_field_order_and_argument_meanings_remain_model_choices(token_decoder):
    arguments = {"not_a_declared_parameter": list(range(32)), "unicode": "中🙂", "nested": {"wrong_semantic_value": False, "other": None}}
    action = tool(arguments=arguments)
    # Reverse the schema field order, retain more than 12 consecutive spaces,
    # and require escaped non-ASCII values through the real token callback.
    text = json.dumps(dict(reversed(list(action.items()))), ensure_ascii=True)
    text = " " * 32 + text
    accepted, metadata = check_tokens(token_decoder, schema(), text)
    assert accepted is True
    assert metadata["parser_settings"]["max_json_array_length"] == 0
    assert metadata["parser_settings"]["force_json_field_order"] is False
    assert metadata["parser_settings"]["max_consecutive_whitespaces"] > 512
    assert metadata["callback_calls"] == len(token_decoder[0].encode(text, add_special_tokens=False)) + 1
    assert metadata["schema_sha256"] == decoder.content_digest(schema())
    assert check_tokens(token_decoder, schema(), json.dumps(action, ensure_ascii=False))[0] is True


def test_schema_does_not_read_other_actors_or_even_current_result_contents():
    view = {"actor": "executor", "phase": "verification", "allocated_verification_calls": 1,
            "used_verification_calls": 0, "discovery_calls_used": 2,
            "results": {"private_a": {"secret": "visible only here"}}, "inbox": []}
    before = deepcopy(view)
    first = decoder.build_phase_action_schema("executor", "VERIFY", view, PUBLIC_SCHEMAS, discovery_call_limit=3)
    view["results"] = {"completely_different_id": {"answer": "not a grammar input"}}
    view["inbox"] = [{"text": "other observations"}]
    second = decoder.build_phase_action_schema("executor", "VERIFY", view, PUBLIC_SCHEMAS, discovery_call_limit=3)
    assert first == second
    assert "private_a" not in json.dumps(first) and "visible only here" not in json.dumps(first)
    assert before["results"]["private_a"]["secret"] == "visible only here"


def test_fresh_turn_callback_does_not_retain_another_turns_prefix(token_decoder):
    tokenizer, adapter = token_decoder
    first = adapter.prepare(schema(), 512, {})
    second = adapter.prepare(schema("planner"), 512, {})
    assert first.callback.token_enforcer is not second.callback.token_enforcer
    assert first.callback.token_enforcer.prefix_states == second.callback.token_enforcer.prefix_states == {}
    assert first.callback.token_enforcer.tokenizer_tree is second.callback.token_enforcer.tokenizer_tree


def test_missing_optional_dependency_is_an_explicit_enabled_path_error(monkeypatch):
    def missing(name):
        raise decoder.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(decoder.importlib.metadata, "version", missing)
    with pytest.raises(ValueError, match="LOCAL_ACTOR_ACTION_DECODER_DEPENDENCY_UNAVAILABLE"):
        decoder.ActionSchemaDecoder(CharacterTokenizer())


def test_library_fallback_eos_is_not_accepted_for_an_unfinished_action(token_decoder, monkeypatch):
    tokenizer, adapter = token_decoder
    metadata = {}
    callback = adapter.prepare(schema(), 512, metadata)
    enforcer = callback.callback.token_enforcer
    original = enforcer.get_allowed_tokens

    def premature(tokens):
        result = original(tokens)
        result.allowed_tokens[:] = [tokenizer.eos_token_id]
        return result

    monkeypatch.setattr(enforcer, "get_allowed_tokens", premature)
    with pytest.raises(ValueError, match="LOCAL_ACTOR_ACTION_DECODER_PREMATURE_EOS"):
        callback(0, torch.tensor(tokenizer.encode("prompt", add_special_tokens=False)))
    assert metadata["callback_failures"] == 1


@pytest.mark.parametrize("text,expected", [
    (r'"caf\u00e9"', True),
    (r'"pre\u4e2d"', True),
    (r'"caf\ud83d\ude42"', True),
    (r'"caf\\u0041"', True),
    (r'"caf\q"', False),
    (r'"caf\u00zz"', False),
])
def test_unicode_escape_pieces_use_character_parser_after_freetext_cache(text, expected):
    tokenizer = CharacterTokenizer(extra_tokens=[r"\u", r"\u00", r"pre\u", r"\ud", r"\q", r"\u00zz"])
    adapter = decoder.ActionSchemaDecoder(tokenizer)
    accepted, metadata = check_tokens((tokenizer, adapter), {"type": "string"}, text)
    assert accepted is expected
    assert metadata["escape_branch_traversals"] > 0
    assert metadata["callback_failures"] == 0

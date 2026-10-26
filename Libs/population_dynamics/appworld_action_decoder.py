"""Phase-permitted JSON actions through the pinned LM Format Enforcer adapter.

This restricts action syntax and existing API permissions, not parameter truth,
identifier uniqueness, observation completeness, or task success. No generated
response is repaired. Tokenizer data alone is shared; each turn gets a new parser.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.metadata
import time

from .appworld_collaboration import ACTORS, content_digest, tool_phase_error


LMFE_VERSION = "0.11.3"


def validate_action_decoder_bounds(bounds: dict) -> dict:
    """Validate an explicit experiment setting; historical decoding is unchanged."""
    fields = {"message_text_max_length", "max_consecutive_whitespaces"}
    if (not isinstance(bounds, dict) or set(bounds) != fields
            or any(type(bounds[key]) is not int or bounds[key] <= 0 for key in fields)):
        raise ValueError("LOCAL_ACTOR_ACTION_DECODER_BOUNDS_INVALID")
    return dict(bounds)


def build_phase_action_schema(actor: str, host_state: str, view: dict, schemas: dict,
                              *, discovery_call_limit: int, reply_recipient: str | None = None,
                              message_text_max_length: int | None = None) -> dict:
    """Use only the selected actor's existing view and public API declarations."""
    phases = {"DISCOVERY": "discovery", "VERIFY": "verification", "VERIFY_REPLY": "verification",
              "PLANNER_HANDOFF": "execution", "PLANNER_REPLY": "execution", "EXECUTE": "execution"}
    if actor not in ACTORS or host_state not in phases or view.get("actor") != actor or phases[host_state] != view.get("phase"):
        raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_STATE_MISMATCH")
    peer = "executor" if actor == "planner" else "planner"
    reply_only = host_state in {"VERIFY_REPLY", "PLANNER_HANDOFF", "PLANNER_REPLY"}
    if host_state in {"PLANNER_HANDOFF", "PLANNER_REPLY"} and actor != "planner":
        raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_REPLY_ACTOR_MISMATCH")
    if host_state == "VERIFY_REPLY" and reply_recipient != peer:
        raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_REPLY_RECIPIENT_MISMATCH")
    if host_state == "EXECUTE" and actor != "executor":
        raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_EXECUTOR_REQUIRED")
    if message_text_max_length is not None and (
            type(message_text_max_length) is not int or message_text_max_length <= 0):
        raise ValueError("LOCAL_ACTOR_ACTION_DECODER_TEXT_BOUND_INVALID")

    def envelope(kind: str, extra: dict) -> dict:
        properties = {"actor": {"type": "string", "enum": [actor]},
                      "kind": {"type": "string", "enum": [kind]}, **extra}
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    message = {"recipient": {"type": "string", "enum": [peer]}, "text": {"type": "string"},
               "share_results": {"type": "array", "items": {"type": "string"}}}
    if message_text_max_length is not None:
        message["text"]["maxLength"] = message_text_max_length
    options = [envelope("message", message)]
    if not reply_only:
        if host_state == "DISCOVERY" and actor == "planner":
            options.append(envelope("proposal", message))
        if host_state == "EXECUTE":
            options.append(envelope("finish", {}))
        remaining = True
        if host_state == "DISCOVERY":
            used = view.get("discovery_calls_used")
            if type(used) is not int or used < 0 or type(discovery_call_limit) is not int or discovery_call_limit < 0:
                raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_QUOTA_INVALID")
            remaining = used < discovery_call_limit
        elif host_state == "VERIFY":
            used, allocated = view.get("used_verification_calls"), view.get("allocated_verification_calls")
            if type(used) is not int or type(allocated) is not int or not 0 <= used <= allocated:
                raise ValueError("LOCAL_ACTOR_ACTION_SCHEMA_QUOTA_INVALID")
            remaining = used < allocated
        if remaining:
            for app, apis in sorted(schemas.items()):
                permitted = [api for api, declaration in sorted(apis.items())
                             if tool_phase_error(view["phase"], actor, declaration["method"].upper(), api) is None]
                if permitted:
                    # One branch per app preserves every allowed app/API pair.
                    # Per-API object branches duplicate the free-argument parser
                    # until the model emits app/api, especially with fields reversed.
                    options.append(envelope("tool", {
                        "call_id": {"type": "string"}, "app": {"type": "string", "enum": [app]},
                        "api": {"type": "string", "enum": permitted}, "arguments": {"type": "object"}}))
    return {"anyOf": options}


class _MeasuredCallback:
    def __init__(self, callback, metadata: dict, escape_token_tree):
        self.callback, self.metadata = callback, metadata
        self.escape_token_tree = escape_token_tree

    def __call__(self, batch_id, sent):
        started = time.perf_counter()
        self.metadata["callback_calls"] += 1
        try:
            allowed = self.callback(batch_id, sent)
            enforcer = self.callback.token_enforcer
            state = enforcer.prefix_states[tuple(sent.tolist())]
            shortcut = state.parser.shortcut_key()
            if isinstance(shortcut, tuple) and shortcut[0] == "json_freetext":
                # LMFE's free-text cache rejects incomplete escape tokens as
                # standalone JSON strings, then traverses only quote branches.
                # Reuse its parser traversal on the vocabulary's escape subset;
                # this also handles tokens with ordinary text before the escape.
                additional = []
                enforcer._collect_allowed_tokens(state.parser, self.escape_token_tree, additional, None)
                allowed = allowed + additional
                self.metadata["escape_branch_traversals"] += 1
            eos = enforcer.eos_token_id if isinstance(enforcer.eos_token_id, list) else [enforcer.eos_token_id]
            # LMFE 0.11.3 catches internal failures and offers EOS. Preserve that
            # failure instead of treating an unfinished parser as valid completion.
            if any(token in allowed for token in eos) and not state.parser.can_end():
                raise ValueError("LOCAL_ACTOR_ACTION_DECODER_PREMATURE_EOS")
            return allowed
        except Exception:
            self.metadata["callback_failures"] += 1
            raise
        finally:
            self.metadata["callback_wall_seconds"] += time.perf_counter() - started


class ActionSchemaDecoder:
    """Reuse LMFE's tokenizer adapter; retain a separate grammar for every turn."""
    def __init__(self, tokenizer):
        try:
            version = importlib.metadata.version("lm-format-enforcer")
            if version != LMFE_VERSION:
                raise ValueError("LOCAL_ACTOR_ACTION_DECODER_VERSION_MISMATCH")
            self.lmfe = importlib.import_module("lmformatenforcer")
            self.integration = importlib.import_module("lmformatenforcer.integrations.transformers")
            self.config_class = importlib.import_module("lmformatenforcer.characterlevelparser").CharacterLevelParserConfig
            self.interegular_version = importlib.metadata.version("interegular")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise ValueError("LOCAL_ACTOR_ACTION_DECODER_DEPENDENCY_UNAVAILABLE") from exc
        self.tokenizer_data = self.integration.build_token_enforcer_tokenizer_data(tokenizer)
        tree = self.tokenizer_data.tokenizer_tree
        self.escape_token_tree = type(tree.root)()
        for token_id, text, _ in self.tokenizer_data.regular_tokens:
            if "\\" in text:
                tree._add_token_to_tree(text, token_id, self.escape_token_tree)
        self.max_token_characters = max(1, max(len(text) for _, text, _ in self.tokenizer_data.regular_tokens))

    def prepare(self, schema: dict, max_new_tokens: int, metadata: dict, *,
                max_consecutive_whitespaces: int | None = None):
        if not isinstance(schema, dict) or type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("LOCAL_ACTOR_ACTION_DECODER_INPUT_INVALID")
        if max_consecutive_whitespaces is not None and (
                type(max_consecutive_whitespaces) is not int or max_consecutive_whitespaces <= 0):
            raise ValueError("LOCAL_ACTOR_ACTION_DECODER_WHITESPACE_BOUND_INVALID")
        # No array cap. Whitespace cannot hit this limit within the existing
        # token budget, even if every generated token is entirely whitespace.
        settings = dict(alphabet=self.tokenizer_data.tokenizer_alphabet,
                        max_consecutive_whitespaces=(max_new_tokens * self.max_token_characters + 1
                            if max_consecutive_whitespaces is None else max_consecutive_whitespaces),
                        force_json_field_order=False, max_json_array_length=0)
        config = self.config_class(**settings)
        parser = self.lmfe.JsonSchemaParser(deepcopy(schema), config=config)
        callback = self.integration.build_transformers_prefix_allowed_tokens_fn(self.tokenizer_data, parser)
        # TokenEnforcer.__init__ resets parser.config in this pinned version.
        # Restore the explicit settings before its first token traversal.
        callback.token_enforcer.root_parser.config = config
        metadata.update(implementation="lm-format-enforcer", version=LMFE_VERSION,
                        interegular_version=self.interegular_version,
                        schema_sha256=content_digest(schema), schema=deepcopy(schema),
                        parser_settings={key: value for key, value in settings.items() if key != "alphabet"},
                        callback_calls=0, callback_failures=0, callback_wall_seconds=0.0,
                        escape_branch_traversals=0)
        return _MeasuredCallback(callback, metadata, self.escape_token_tree)

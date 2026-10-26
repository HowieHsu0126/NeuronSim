"""Persistent two-actor execution and causal histories for AppWorld development.

The caller supplies one admitted native world and source-declared API methods.
This module never constructs/restores a world, logs in to an app, evaluates a
task, or selects a learned model. All executed API actions come from the caller.
It reuses the historical visible-result resolver and API-name validation, but
does not use prefix replay, exception rewriting or single-task phase restrictions.

An allocation policy receives only state reports explicitly delivered by both
actors. Actor histories are independent, and messages become visible only on
delivery. The same interfaces support a fitted closure policy and fixed controls;
implementing this execution boundary alone does not implement NeuronSim.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import keyword
import math
import time
from typing import Any, Protocol
import uuid

from .appworld_collaboration import ACTORS, resolve_arguments
from .appworld_reference import NAME


CALL_ID_DESCRIPTION = (
    f"call_id must be a string matching {NAME.pattern}: start with an ASCII letter; "
    "use only ASCII letters, digits, underscores or hyphens; "
    "use 1 to 64 characters total. A valid example is 'call_1'."
)


class NativeWorld(Protocol):
    def execute(self, code: str) -> str:
        """Execute code once in the current native world and return its output."""


class UnknownNativeOutcome(RuntimeError):
    """A native call may have executed; resolve its outcome before continuing."""


_NATIVE_EXECUTION_ERROR_PREFIX = "Execution failed. Traceback:\n"


def _actor(value: str) -> str:
    if value not in ACTORS:
        raise ValueError("actor must be planner or executor")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _json_copy(value: Any) -> Any:
    """Reject coercion of non-JSON values at the model/history boundary."""
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("history content must contain finite JSON values and string keys")


@dataclass(frozen=True)
class ActorSnapshot:
    """An immutable prefix; local indices reveal no undelivered peer events."""

    actor: str
    event_json: tuple[str, ...]

    def as_model_input(self) -> dict[str, Any]:
        return {"actor": self.actor, "events": [json.loads(row) for row in self.event_json]}


class CollaborationHistory:
    """Own actor-visible observations and explicit deliveries for one attempt."""

    def __init__(self, task_instruction: str):
        instruction = _text(task_instruction, "task_instruction")
        self._events: dict[str, list[str]] = {actor: [] for actor in ACTORS}
        self._visible_results: dict[str, dict[str, dict[str, Any]]] = {actor: {} for actor in ACTORS}
        self._pending_messages: dict[str, dict[str, Any]] = {}
        self._reports: dict[str, Any] = {}
        for actor in ACTORS:
            self._append(actor, "task_instruction", {"text": instruction})

    def _append(self, actor: str, kind: str, payload: dict[str, Any]) -> None:
        rows = self._events[_actor(actor)]
        event = {"index": len(rows), "kind": kind, "payload": _json_copy(payload)}
        rows.append(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False))

    def snapshot(self, actor: str) -> ActorSnapshot:
        return ActorSnapshot(_actor(actor), tuple(self._events[actor]))

    def record_public_context(self, context: dict[str, Any]) -> None:
        """Deliver the same source-public task metadata to both actors."""
        if not isinstance(context, dict):
            raise ValueError("public context must be a JSON object")
        context = _json_copy(context)
        for actor in ACTORS:
            self._append(actor, "public_context", context)

    def record_generated_action(self, actor: str, action: dict[str, Any]) -> None:
        if not isinstance(action, dict):
            raise ValueError("generated action must be a JSON object")
        self._append(actor, "generated_action", {"action": action})

    def record_rejection(self, actor: str, reason: str) -> None:
        self._append(actor, "action_rejection", {"reason": _text(reason, "reason")})

    def resolve_arguments(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        visible = self._visible_results[_actor(actor)]
        arguments = _json_copy(arguments)
        try:
            return resolve_arguments(arguments, visible, set(visible))
        except (KeyError, IndexError, TypeError) as error:
            # These errors occur while resolving a model-supplied reference,
            # before any native call. They must remain action rejections.
            raise ValueError(f"invalid visible-result reference path: {type(error).__name__}") from error

    def _record_tool_result(self, actor: str, result_id: str, result: dict[str, Any]) -> None:
        # Only the persistent tool session calls this after native execution.
        if result_id in self._visible_results[_actor(actor)]:
            raise ValueError("tool result already visible")
        observed = _json_copy(result)
        self._visible_results[actor][result_id] = observed
        self._append(actor, "tool_observation", {"result_id": result_id, "observation": observed})

    def send_message(
        self, sender: str, recipient: str, text: str, *, share_results: tuple[str, ...] = (),
    ) -> str:
        _actor(sender)
        _actor(recipient)
        if sender == recipient:
            raise ValueError("a peer message requires distinct sender and recipient")
        message_text = _text(text, "message text")
        if not isinstance(share_results, tuple) or any(not isinstance(ref, str) for ref in share_results):
            raise ValueError("share_results must be a tuple of visible result IDs")
        if len(set(share_results)) != len(share_results):
            raise ValueError("shared result IDs must be unique")
        visible = self._visible_results[sender]
        if any(ref not in visible for ref in share_results):
            raise ValueError("cannot share an unseen tool result")
        # Copy at send time. Delivery cannot accidentally include later results.
        message = {
            "sender": sender, "recipient": recipient, "text": message_text,
            "shared_results": {ref: deepcopy(visible[ref]) for ref in share_results},
        }
        message_id = uuid.uuid4().hex
        self._pending_messages[message_id] = message
        self._append(sender, "message_sent", message)
        return message_id

    def deliver_message(self, message_id: str) -> None:
        if message_id not in self._pending_messages:
            raise ValueError("message is unknown or already delivered")
        message = self._pending_messages[message_id]
        recipient = message["recipient"]
        visible = self._visible_results[recipient]
        for ref, result in message["shared_results"].items():
            if ref in visible and visible[ref] != result:
                raise ValueError("shared result identity conflicts with an existing observation")
        self._append(recipient, "message_delivered", message)
        visible.update(deepcopy(message["shared_results"]))
        del self._pending_messages[message_id]

    def deliver_state_report(self, actor: str, representation: dict[str, Any]) -> None:
        """Explicitly send an actor-computed retained state to the controller.

        The caller computes this state from snapshot(actor), before subsequent
        execution. No automatic peer-history or native-state access is supplied.
        Fitted encoders and their training provenance are the caller's concern.
        """
        _actor(actor)
        if actor in self._reports:
            raise ValueError("this decision boundary already received the actor's state")
        if not isinstance(representation, dict) or not representation:
            raise ValueError("state report must be a nonempty JSON object")
        report = _json_copy(representation)
        self._append(actor, "state_report_delivered", {"representation": report})
        self._reports[actor] = report

    def controller_reports(self) -> dict[str, Any]:
        if set(self._reports) != set(ACTORS):
            raise ValueError("both actors must explicitly deliver state before allocation")
        return deepcopy(self._reports)


class PersistentToolSession:
    """Execute actor-selected APIs once, with one fixed verification allocation.

    Non-verification actions use the same native environment before and after
    allocation. Verification permits only source-declared GET APIs and consumes
    an assigned slot even if the API raises. Generation/rejection/token budgets
    belong to the host. The host decides which phase uses execute or verify;
    actors must not control that classification themselves.
    """

    def __init__(
        self, world: NativeWorld, history: CollaborationHistory,
        api_http_methods: Mapping[tuple[str, str], str], *, verification_budget: int,
    ):
        if type(verification_budget) is not int or verification_budget < 0:
            raise ValueError("verification_budget must be a nonnegative integer")
        if not api_http_methods or any(
            not isinstance(key, tuple) or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
            or method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
            for key, method in api_http_methods.items()
        ):
            raise ValueError("source-declared API HTTP methods are required")
        self._world = world
        self.history = history
        self._api_http_methods = dict(api_http_methods)
        self.verification_budget = verification_budget
        self._allocation: dict[str, int] | None = None
        self._used = {actor: 0 for actor in ACTORS}
        self._attempts: list[dict[str, Any]] = []
        self._unknown = False

    @property
    def attempts(self) -> list[dict[str, Any]]:
        return deepcopy(self._attempts)

    @property
    def allocation(self) -> dict[str, int] | None:
        return deepcopy(self._allocation)

    @property
    def verification_used(self) -> dict[str, int]:
        return dict(self._used)

    def allocate(self, policy: Callable[..., Mapping[str, int]]) -> dict[str, int]:
        if self._unknown:
            raise UnknownNativeOutcome("a preceding native outcome remains unresolved")
        if self._allocation is not None:
            raise ValueError("verification allocation is already fixed for this attempt")
        reports = self.history.controller_reports()
        proposal = policy(state_reports=deepcopy(reports), total_budget=self.verification_budget)
        if (
            not isinstance(proposal, Mapping) or set(proposal) != set(ACTORS)
            or any(type(value) is not int or value < 0 for value in proposal.values())
            or sum(proposal.values()) != self.verification_budget
        ):
            raise ValueError("allocation must divide the exact fixed budget across both actors")
        self._allocation = dict(proposal)
        for actor in ACTORS:
            self.history._append(actor, "verification_allocation", {"assigned_calls": proposal[actor]})
        return dict(self._allocation)

    def execute(self, actor: str, call_id: str, app: str, api: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._call(actor, call_id, app, api, arguments, verification=False)

    def verify(self, actor: str, call_id: str, app: str, api: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._call(actor, call_id, app, api, arguments, verification=True)

    def _call(
        self, actor: str, call_id: str, app: str, api: str, arguments: dict[str, Any], *, verification: bool,
    ) -> dict[str, Any]:
        _actor(actor)
        if self._unknown:
            raise UnknownNativeOutcome("native outcome is unresolved; the call was not repeated")
        if not isinstance(call_id, str) or NAME.fullmatch(call_id) is None:
            raise ValueError(CALL_ID_DESCRIPTION)
        if any(not isinstance(name, str) or not name.isascii() or not name.isidentifier() or keyword.iskeyword(name) for name in (app, api)):
            raise ValueError("app and API must be valid Python source identifiers")
        if (app, api) not in self._api_http_methods:
            raise ValueError("API is absent from the supplied public source schema")
        if verification:
            if self._allocation is None:
                raise ValueError("verification requires an explicit allocation")
            if self._api_http_methods[(app, api)] != "GET":
                raise ValueError("verification requires a source-declared GET API")
            if self._used[actor] >= self._allocation[actor]:
                raise ValueError("actor verification allocation is exhausted")
        result_id = f"{actor}/{call_id}"
        if any(row["result_id"] == result_id for row in self._attempts):
            raise ValueError("an actor call_id cannot be reused within this native attempt")
        resolved = self.history.resolve_arguments(actor, arguments)
        arguments_literal = repr(json.dumps(resolved, ensure_ascii=False, allow_nan=False))
        # Do not catch inside native code: preserve AppWorld's traceback output
        # and state-save path, including writes that happened before an error.
        code = ("import json\n"
                f"print(json.dumps(apis.{app}.{api}(**json.loads({arguments_literal})), ensure_ascii=False))")
        row: dict[str, Any] = {
            "actor": actor, "result_id": result_id, "app": app, "api": api,
            "arguments": resolved, "verification": verification, "status": "pending",
            "raw_output": None, "observation": None,
        }
        self._attempts.append(row)
        self.history._append(actor, "tool_attempt", {key: row[key] for key in ("result_id", "app", "api", "arguments", "verification")})
        if verification:
            self._used[actor] += 1
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        try:
            raw = self._world.execute(code)
            row["raw_output"] = raw
            if not isinstance(raw, str):
                raise TypeError("native output must be text")
        except BaseException as error:
            # A Python/transport error does not establish that the native API
            # was skipped. Stop rather than resend its action.
            self._unknown = True
            row.update(status="unknown", error_type=type(error).__name__, error=str(error))
            self.history._append(actor, "native_outcome_unknown", {"result_id": result_id, "error_type": type(error).__name__, "error": str(error)})
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise UnknownNativeOutcome(f"native outcome unresolved for {result_id}") from error
        finally:
            row["native_wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started
        if raw.startswith(_NATIVE_EXECUTION_ERROR_PREFIX):
            observed = {"ok": False, "error_type": "NativeExecutionError", "error": raw, "raw_output": raw}
        else:
            try:
                result = _json_copy(json.loads(raw))
            except ValueError as error:
                # Native returned text is an observation even if JSON decoding
                # fails. Neither API success nor failure is inferred from that.
                observed = {"ok": None, "raw_output": raw, "parse_error": str(error)}
            else:
                observed = {"ok": True, "result": result, "raw_output": raw}
        row.update(status="returned", observation=observed)
        self.history._record_tool_result(actor, result_id, observed)
        return {"result_id": result_id, "observation": deepcopy(observed)}

"""Bounded local actor scheduling over the existing persistent JSON controller.

The caller owns process startup, shutdown, stdout JSON framing, and artifact
writes. This library never resends a request with an unknown outcome. Actor
receipts remain in the producer ledger; host records only delivery and scheduling
outcomes for those attempts. Scheduling is an engineering execution policy.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Protocol

from .appworld_action_decoder import build_phase_action_schema, validate_action_decoder_bounds
from .appworld_collaboration import ACTORS, content_digest, generation_telemetry, validate_config
from .appworld_local_actor import LocalTransformersActor, model_identity, request_from_receipt
from .appworld_recorded_prefix import RecordedPrefix, RecordedPrefixError, actor_evidence_table


class JsonChannel(Protocol):
    def send(self, request: dict) -> None:
        """Send one JSON request once; failures may leave its delivery unknown."""

    def receive(self) -> dict:
        """Receive one complete JSON object without silently retrying requests."""


# Only controller checks known to precede native execution permit another model
# attempt. Opaque errors and errors marked with error_type stop the host.
RECOVERABLE_REJECTIONS = frozenset({
    "MESSAGE_FIELDS_INVALID", "MESSAGE_CONTENT_REQUIRED", "CANNOT_SHARE_UNSEEN_RESULT",
    "ONE_PLANNER_PROPOSAL_REQUIRED_BEFORE_WRITES", "TOOL_FIELDS_INVALID",
    "UNIQUE_API_IDENTIFIERS_REQUIRED", "API_NOT_IN_PUBLIC_TOOL_SCHEMA",
    "API_ARGUMENT_MAPPING_REQUIRED", "API_ARGUMENT_NAMES_INVALID", "API_IDENTIFIER_INVALID",
    "RESULT_NOT_VISIBLE_TO_ACTOR", "SUCCESSFUL_RESULT_PATH_REQUIRED", "RESULT_PATH_KEY_INVALID",
    "TASK_WRITES_REQUIRE_PROPOSAL_AND_VERIFICATION", "DISCOVERY_CALL_BUDGET_EXHAUSTED",
    "VERIFICATION_REQUIRES_SOURCE_DECLARED_GET", "VERIFICATION_CALL_BUDGET_EXHAUSTED",
    "POST_BUDGET_CALL_NOT_AN_EXECUTOR_TASK_WRITE", "FINISH_REQUIRES_EXECUTOR_AFTER_VERIFICATION",
    "UNKNOWN_ACTOR_REQUEST_KIND",
})
VIEW_FIELDS = {"actor", "task", "phase", "inbox", "results", "allocated_verification_calls",
               "used_verification_calls", "discovery_calls_used"}
ACTION_INSTRUCTION = (
    "Return exactly one JSON action, without markdown or a generation field. "
    "Every action must contain your actor and kind. A tool action has exactly actor, kind, call_id, app, api, "
    "arguments; begin call_id with your own actor name and keep it unique across your own history. "
    "Use a supplied public API with its declared arguments. "
    "A message or proposal has exactly actor, kind, recipient, text, share_results; share_results contains "
    "only result IDs visible to you. The only actor recipients are planner and executor: send messages to your peer, "
    "and send planner proposals to executor. supervisor is an API app, not an actor recipient. "
    "A finish action has exactly actor and kind. Only the planner may propose. Only the executor may finish, "
    "and only in execution after verification. Do not request allocation yourself. "
    "Discovery permits source-declared GET APIs, login and peer messages; the executor cannot propose or finish. "
    "The public supervisor.show_profile and supervisor.show_account_passwords APIs can supply observed identity "
    "and credential facts. API schema examples are placeholders, not observed values; do not guess credentials. "
    "Obtain necessary observations and authentication before the planner proposal: after allocation, "
    "only the assigned extra GET verification attempts are available, "
    "and execution permits executor writes only, with no GETs or login. "
    "Arguments may reference a successful visible result as {\"$result\":\"call_id\",\"path\":[\"field\"]}. "
    "Use only your view and explicitly delivered messages. Native evaluation is unavailable to actors."
)
STATE_INSTRUCTIONS = {
    "DISCOVERY": "Discovery alternates actors. Use permitted GET or login calls, or message your peer. Only the planner may propose, with recipient executor, after necessary observations and authentication. The executor cannot propose; neither actor can finish in discovery. Use observed identity and credential facts, not schema example values.",
    "VERIFY": "Use your remaining assigned verification calls through source-declared GET APIs, or message your peer. Failed API attempts consume calls. Allocation is fixed; do not propose or finish during verification.",
    "VERIFY_REPLY": "Reply to reply_recipient with one generated message using only your own view and explicitly shared results. This reply slot permits only that message, not tools, proposals or finish. It does not grant another verification call.",
    "PLANNER_HANDOFF": "Send generated messages to the executor sharing every required verification result ID, including failed results. This slot permits only messages, not tools, proposals or finish.",
    "EXECUTE": "The executor may execute allowed task writes, message the planner if needed, or finish for terminal evaluation now that verification is complete. No further GETs or login calls are available.",
    "PLANNER_REPLY": "Reply to the executor with one generated message using only your own view. This reply slot permits only messages, not tools, proposals or finish.",
}


OBSERVATION_DELTA_FORMAT = "actor_observation_append_v1"
OBSERVATION_DELTA_RULE = (
    "Reconstruct view.results and view.inbox from base_message_index in your own message history "
    "(zero-based, counting system, user and assistant messages). Resolve that user message's delta "
    "first if present, merge results_added by new ID, and append inbox_added in order including duplicates. "
    "All other current fields are complete."
)


def _observation_json(value) -> str:
    # JSON equality retains bool/int/float and signed-zero distinctions.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def reconstruct_observation_prompt(prompt: str, history: list[dict], actor: str) -> str:
    """Expand only prior same-actor references; return the original canonical prompt.

    ``history`` is the actor's message history strictly before this user prompt.
    Assistant actions, feedback, counters and all non-view fields are untouched.
    """
    content = json.loads(prompt)
    if (not isinstance(content, dict) or content.get("actor") != actor
            or not isinstance(content.get("view"), dict) or content["view"].get("actor") != actor):
        raise ValueError("LOCAL_HOST_OBSERVATION_ACTOR_MISMATCH")
    delta = content.get("observation_history_delta")
    if delta is None:
        return prompt
    fields = {"format", "base_message_index", "results_added", "inbox_added", "reconstruction"}
    if (not isinstance(delta, dict) or set(delta) != fields
            or delta["format"] != OBSERVATION_DELTA_FORMAT
            or delta["reconstruction"] != OBSERVATION_DELTA_RULE
            or type(delta["base_message_index"]) is not int
            or not 0 <= delta["base_message_index"] < len(history)
            or not isinstance(delta["results_added"], dict) or not isinstance(delta["inbox_added"], list)
            or "results" in content["view"] or "inbox" in content["view"]):
        raise ValueError("LOCAL_HOST_OBSERVATION_REFERENCE_INVALID")
    index = delta["base_message_index"]
    if history[index].get("role") != "user":
        raise ValueError("LOCAL_HOST_OBSERVATION_REFERENCE_INVALID")
    base = json.loads(reconstruct_observation_prompt(history[index]["content"], history[:index], actor))["view"]
    if (not isinstance(base.get("results"), dict) or not isinstance(base.get("inbox"), list)
            or base["results"].keys() & delta["results_added"].keys()):
        raise ValueError("LOCAL_HOST_OBSERVATION_REFERENCE_INVALID")
    content["view"]["results"] = {**base["results"], **delta["results_added"]}
    content["view"]["inbox"] = base["inbox"] + delta["inbox_added"]
    del content["observation_history_delta"]
    return _observation_json(content)


def encode_observation_prompt(prompt: str, history: list[dict], actor: str) -> str:
    """Replace confirmed immutable view prefixes only when the result is smaller.

    References never leave the supplied private history. Changed/deleted results
    or non-prefix inbox changes retain the full prompt rather than losing data.
    """
    content = json.loads(reconstruct_observation_prompt(prompt, [], actor))
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") != "user":
            continue
        base = json.loads(reconstruct_observation_prompt(history[index]["content"], history[:index], actor))["view"]
        view = content["view"]
        if (not isinstance(view.get("results"), dict) or not isinstance(view.get("inbox"), list)
                or not isinstance(base.get("results"), dict) or not isinstance(base.get("inbox"), list)
                or not base["results"].keys() <= view["results"].keys()
                or any(_observation_json(value) != _observation_json(view["results"][key])
                       for key, value in base["results"].items())
                or _observation_json(base["inbox"]) != _observation_json(view["inbox"][:len(base["inbox"])])):
            return prompt
        content["observation_history_delta"] = dict(format=OBSERVATION_DELTA_FORMAT,
            base_message_index=index,
            results_added={key: value for key, value in view.pop("results").items() if key not in base["results"]},
            inbox_added=view.pop("inbox")[len(base["inbox"]):], reconstruction=OBSERVATION_DELTA_RULE)
        encoded = _observation_json(content)
        return encoded if len(encoded.encode("utf-8")) < len(prompt.encode("utf-8")) else prompt
    return prompt


class HostFailure(RuntimeError):
    def __init__(self, code: str, *, unknown_outcome: bool = False):
        super().__init__(code)
        self.unknown_outcome = unknown_outcome


class LocalActorHost:
    def __init__(self, actor: LocalTransformersActor, channel: JsonChannel,
                 config: dict, schemas: dict, allocation: dict[str, int], *,
                 recorded_prefix: RecordedPrefix | None = None, evidence_table_after_prefix: bool = False,
                 lossless_observation_history: bool = False, verification_message_replies: bool = False,
                 phase_action_decoder: bool = False, action_decoder_bounds: dict | None = None):
        self.config = deepcopy(validate_config(config))
        if self.config["provider"] != "local_transformers" or model_identity(actor.config) != self.config["model_revision"]:
            raise ValueError("LOCAL_HOST_MODEL_IDENTITY_MISMATCH")
        if (not isinstance(allocation, dict) or set(allocation) != set(ACTORS)
                or any(type(value) is not int or value < 0 for value in allocation.values())
                or sum(allocation.values()) != self.config["verification_budget"]):
            raise ValueError("LOCAL_HOST_ALLOCATION_INVALID")
        if not isinstance(schemas, dict) or not schemas:
            raise ValueError("LOCAL_HOST_PUBLIC_SCHEMAS_REQUIRED")
        if (set(actor.session_ids) != set(ACTORS)
                or any(not isinstance(value, str) or not value for value in actor.session_ids.values())
                or len(set(actor.session_ids.values())) != len(ACTORS)
                or set(actor.histories) != set(ACTORS)
                or actor.histories["planner"] is actor.histories["executor"]):
            raise ValueError("LOCAL_HOST_DISTINCT_CONTEXTS_REQUIRED")
        self.actor, self.channel = actor, channel
        self.schemas, self.allocation = deepcopy(schemas), dict(allocation)
        self.attempt_limit = self.config["max_actor_turns"]
        self.attempts: list[dict] = []
        self._producer_start = len(actor.attempts)
        self._ran = False
        self._pending: dict | None = None
        self.controller_request_count = 0
        self.state, self.phase = "START", None
        self.discovery_actor = "planner"
        self.feedback = {name: None for name in ACTORS}
        self._schemas_provided = {name: False for name in ACTORS}
        self.verification_used = {name: 0 for name in ACTORS}
        self.planner_verification_ids: list[str] = []
        self.planner_shared_ids: set[str] = set()
        self.allocation_attempted = self.allocation_acknowledged = False
        self.planner_reply_slots = 0
        self.native_evaluation = self.terminal = None
        self.status, self.error, self.unknown_outcome = "FAILED", None, False
        if type(evidence_table_after_prefix) is not bool or (evidence_table_after_prefix and recorded_prefix is None):
            raise ValueError("LOCAL_HOST_EVIDENCE_TABLE_REQUIRES_PREFIX")
        if type(lossless_observation_history) is not bool:
            raise ValueError("LOCAL_HOST_OBSERVATION_ENCODING_FLAG_INVALID")
        if type(verification_message_replies) is not bool:
            raise ValueError("LOCAL_HOST_VERIFICATION_REPLY_FLAG_INVALID")
        if type(phase_action_decoder) is not bool or (phase_action_decoder and recorded_prefix is None):
            raise ValueError("LOCAL_HOST_ACTION_DECODER_REQUIRES_PREFIX")
        self.phase_action_decoder = phase_action_decoder
        if action_decoder_bounds is not None and not phase_action_decoder:
            raise ValueError("LOCAL_HOST_ACTION_DECODER_BOUNDS_REQUIRE_DECODER")
        self.action_decoder_bounds = (None if action_decoder_bounds is None
                                      else validate_action_decoder_bounds(action_decoder_bounds))
        self.verification_message_replies = verification_message_replies
        self.verification_reply_slots = self.verification_replies_completed = 0
        self._verification_reply: dict | None = None
        self.lossless_observation_history = lossless_observation_history
        self.recorded_prefix = recorded_prefix
        self.evidence_table_after_prefix = evidence_table_after_prefix
        self._prefix_paused = False
        if recorded_prefix is not None:
            recorded_prefix.bind(actor, self.config)

    def _exchange(self, request: dict, attempt: dict | None = None) -> dict:
        if self._pending is not None:
            raise HostFailure("LOCAL_HOST_REQUEST_ALREADY_OUTSTANDING", unknown_outcome=True)
        index = self.controller_request_count
        if self.recorded_prefix is not None:
            self.recorded_prefix.expect_request(request, index)
        self.controller_request_count += 1
        self._pending = {"request_index": index, "kind": request.get("kind"), "actor": request.get("actor")}
        if attempt is not None:
            attempt.update(controller_request_index=index, submission_attempted=True, submitted=None)
        try:
            self.channel.send(deepcopy(request))
            if attempt is not None:
                attempt["submitted"] = True
            response = self.channel.receive()
        except Exception as exc:
            if attempt is not None:
                attempt.update(status="OUTCOME_UNKNOWN", transport_error_type=type(exc).__name__)
            raise HostFailure("LOCAL_HOST_CHANNEL_OUTCOME_UNKNOWN", unknown_outcome=True) from exc
        if attempt is not None:
            attempt["controller_response"] = deepcopy(response)
        if isinstance(response, dict) and set(response) == {"terminal"}:
            if attempt is not None:
                attempt["status"] = "OUTCOME_UNKNOWN"
            self.terminal = deepcopy(response["terminal"])
            raise HostFailure("LOCAL_HOST_TERMINAL_WITHOUT_REQUEST_ACK", unknown_outcome=True)
        valid = (isinstance(response, dict) and type(response.get("ok")) is bool
                 and ((response["ok"] and set(response) == {"ok", "result"})
                      or (not response["ok"] and "error" in response
                          and set(response) <= {"ok", "error", "error_type"})))
        if not valid:
            if attempt is not None:
                attempt["status"] = "OUTCOME_UNKNOWN"
            raise HostFailure("LOCAL_HOST_CONTROLLER_RESPONSE_INVALID", unknown_outcome=True)
        self._pending = None
        if attempt is not None:
            attempt["acknowledged"] = True
        if self.recorded_prefix is not None:
            self.recorded_prefix.observe_response(request, response, index)
        if not response["ok"]:
            if "error_type" in response or not isinstance(response["error"], str) or response["error"] not in RECOVERABLE_REJECTIONS:
                if attempt is not None:
                    attempt["status"] = "CONTROLLER_FATAL_OR_OPAQUE"
                raise HostFailure("LOCAL_HOST_CONTROLLER_FATAL_OR_OPAQUE", unknown_outcome=True)
        return response

    def _view(self, actor: str) -> dict:
        response = self._exchange({"kind": "view", "actor": actor})
        view = response.get("result")
        if (not response["ok"] or not isinstance(view, dict) or set(view) != VIEW_FIELDS
                or view["actor"] != actor or view["phase"] not in {"discovery", "awaiting_allocation", "verification", "execution"}
                or not isinstance(view["results"], dict) or not isinstance(view["inbox"], list)):
            raise HostFailure("LOCAL_HOST_VIEW_INVALID")
        used = view["used_verification_calls"]
        if type(used) is not int or not self.verification_used[actor] <= used <= self.allocation[actor]:
            raise HostFailure("LOCAL_HOST_VERIFICATION_COUNTER_INVALID")
        if self.allocation_acknowledged and view["allocated_verification_calls"] != self.allocation[actor]:
            raise HostFailure("LOCAL_HOST_ALLOCATION_CHANGED")
        self.verification_used[actor], self.phase = used, view["phase"]
        return deepcopy(view)

    def _select_actor(self) -> str:
        if self.state == "DISCOVERY":
            return self.discovery_actor
        if self.state == "VERIFY":
            return next((name for name in ACTORS if self.verification_used[name] < self.allocation[name]), "planner")
        if self.state == "VERIFY_REPLY":
            return self._verification_reply["recipient"]
        return "planner" if self.state in {"PLANNER_HANDOFF", "PLANNER_REPLY"} else "executor"

    def _own_native_calls(self, actor: str, view: dict) -> dict:
        """Only that actor's accepted actions identify an observed result's API."""
        known = {}
        for attempt in self.attempts:
            if attempt["actor"] != actor or attempt["status"] != "ACCEPTED":
                continue
            generation = self.actor.attempts[attempt["producer_attempt_index"]]
            action = json.loads(generation["response"])
            if action["kind"] == "tool" and action["call_id"] in view["results"]:
                known[action["call_id"]] = {"actor": actor, "api": action["app"] + "." + action["api"]}
        return known

    def _prompt(self, actor: str, view: dict) -> str:
        content = {"host_state": self.state, "actor": actor,
                   "instruction": ACTION_INSTRUCTION + " " + STATE_INSTRUCTIONS[self.state],
                   "view": view, "last_feedback": deepcopy(self.feedback[actor]),
                   "total_verification_budget": self.config["verification_budget"]}
        # The split is a post-proposal intervention. Keep every discovery
        # prompt independent of the assigned allocation until its acknowledgement.
        if self.allocation_acknowledged and view["allocated_verification_calls"] == self.allocation[actor]:
            content["assigned_verification_calls"] = self.allocation[actor]
        if not self._schemas_provided[actor]:
            content["public_api_schemas"] = self.schemas
        if self.state == "VERIFY_REPLY":
            content["reply_recipient"] = self._verification_reply["sender"]
        if self.state == "PLANNER_HANDOFF":
            content["required_shared_result_ids"] = list(self.planner_verification_ids)
            content["remaining_shared_result_ids"] = sorted(set(self.planner_verification_ids) - self.planner_shared_ids)
        if self.evidence_table_after_prefix and self.recorded_prefix.verified:
            content["actor_evidence_table"] = actor_evidence_table(view, self._own_native_calls(actor, view))
        # Full schemas enter each private history once. This remains the raw
        # archive prompt; optional encoding happens only before live generation.
        return json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def _allocate(self) -> None:
        if self.allocation_attempted:
            raise HostFailure("LOCAL_HOST_ALLOCATION_ALREADY_ATTEMPTED")
        self.allocation_attempted = True
        response = self._exchange({"kind": "allocate", "allocation": self.allocation})
        if not response["ok"] or response["result"] != {"phase": "verification", "allocation": self.allocation}:
            raise HostFailure("LOCAL_HOST_ALLOCATION_ACK_INVALID")
        self.allocation_acknowledged = True
        self.state, self.phase = "VERIFY", "verification"

    def _generate(self, actor: str, view: dict) -> tuple[dict, dict | None]:
        record = {"attempt_index": len(self.attempts), "actor": actor, "host_state": self.state,
                  "producer_attempt_index": len(self.actor.attempts), "status": "GENERATING",
                  "submission_attempted": False, "submitted": False, "acknowledged": False}
        self.attempts.append(record)
        try:
            prompt = self._prompt(actor, view)
            if self.lossless_observation_history and (self.recorded_prefix is None or self.recorded_prefix.verified):
                record["unencoded_user_prompt"] = prompt
                prompt = encode_observation_prompt(prompt, self.actor.histories[actor], actor)
            if self.recorded_prefix is not None and not self.recorded_prefix.verified:
                receipt = self.recorded_prefix.generate(self.actor, actor, prompt)
            elif self.phase_action_decoder:
                schema = build_phase_action_schema(actor, self.state, view, self.schemas,
                    discovery_call_limit=self.config["discovery_calls_per_actor"],
                    reply_recipient=self._verification_reply["sender"] if self.state == "VERIFY_REPLY" else None,
                    message_text_max_length=(None if self.action_decoder_bounds is None
                                            else self.action_decoder_bounds["message_text_max_length"]))
                record["action_schema_sha256"] = content_digest(schema)
                options = ({} if self.action_decoder_bounds is None else {
                    "action_decoder_max_whitespace": self.action_decoder_bounds["max_consecutive_whitespaces"]})
                receipt = self.actor.generate(actor, prompt, action_schema=schema, **options)
            else:
                receipt = self.actor.generate(actor, prompt)
            self._schemas_provided[actor] = True
        except Exception as exc:
            record.update(status="GENERATION_FAILED", error_type=type(exc).__name__)
            if isinstance(exc, ValueError) and str(exc).startswith("LOCAL_ACTOR_"):
                record["error_code"] = str(exc)
            if isinstance(exc, RecordedPrefixError):
                record["error_code"] = str(exc)
                raise HostFailure(str(exc)) from exc
            raise HostFailure("LOCAL_HOST_GENERATION_FAILED") from exc
        try:
            request = request_from_receipt(actor, receipt)
            json.dumps(request, allow_nan=False)
        except (TypeError, ValueError) as exc:
            code = str(exc) if str(exc).startswith("LOCAL_ACTOR_") else "LOCAL_HOST_ACTION_NOT_FINITE_JSON"
            record.update(status="HOST_REJECTED", error_code=code)
            self.feedback[actor] = {"ok": False, "error": code}
            return record, None
        if self.state in {"PLANNER_HANDOFF", "PLANNER_REPLY"} and (request["kind"] != "message" or request.get("recipient") != "executor"):
            record.update(status="HOST_REJECTED", error_code="LOCAL_HOST_PLANNER_MESSAGE_REQUIRED")
            self.feedback[actor] = {"ok": False, "error": "LOCAL_HOST_PLANNER_MESSAGE_REQUIRED"}
            return record, None
        if self.state == "VERIFY_REPLY" and (request["kind"] != "message" or request.get("recipient") != self._verification_reply["sender"]):
            code = "LOCAL_HOST_VERIFICATION_REPLY_MESSAGE_REQUIRED"
            record.update(status="HOST_REJECTED", error_code=code)
            self.feedback[actor] = {"ok": False, "error": code}
            return record, None
        return record, request

    def _finish_verification_reply(self, *, delivered: bool) -> None:
        self.verification_replies_completed += int(delivered)
        self._verification_reply = None
        self.state = "VERIFY"

    def _finish(self, response: dict) -> None:
        result = response["result"]
        if not isinstance(result, dict) or result.get("case_finished") is not True or "native_evaluation" not in result:
            raise HostFailure("LOCAL_HOST_FINISH_ACK_INVALID", unknown_outcome=True)
        self.native_evaluation = deepcopy(result["native_evaluation"])
        self.state = "DRAIN_TERMINAL"
        try:
            terminal = self.channel.receive()
        except Exception as exc:
            raise HostFailure("LOCAL_HOST_TERMINAL_UNAVAILABLE") from exc
        if not isinstance(terminal, dict) or set(terminal) != {"terminal"} or not isinstance(terminal["terminal"], dict):
            raise HostFailure("LOCAL_HOST_TERMINAL_INVALID")
        self.terminal = deepcopy(terminal["terminal"])
        if self.terminal.get("status") != "DEVELOPMENT_EXECUTION_COMPLETE":
            raise HostFailure("LOCAL_HOST_CONTROLLER_TERMINAL_FAILED")
        self.status, self.state = "DEVELOPMENT_EXECUTION_COMPLETE", "TERMINAL"

    def run(self, *, pause_at_prefix_boundary: bool = False) -> dict:
        resume = self._ran
        if resume:
            if (not self._prefix_paused or self.status != "PAUSED_AT_VERIFIED_PREFIX"
                    or self.recorded_prefix is None or not self.recorded_prefix.verified
                    or self._pending is not None or self.unknown_outcome or pause_at_prefix_boundary):
                raise RuntimeError("LOCAL_HOST_ALREADY_RUN")
            self.recorded_prefix.verify_resume(self)
            self._prefix_paused = False
            self.status = "FAILED"
        elif pause_at_prefix_boundary and self.recorded_prefix is None:
            raise ValueError("LOCAL_HOST_PREFIX_PAUSE_REQUIRES_PREFIX")
        self._ran = True
        try:
            if not resume:
                ready = self.channel.receive()
                if isinstance(ready, dict) and set(ready) == {"terminal"}:
                    self.terminal = deepcopy(ready["terminal"])
                    raise HostFailure("LOCAL_HOST_CONTROLLER_NOT_READY")
                ready_fields = {"ready", "task", "actors"}
                expected_checkpoint = None if self.recorded_prefix is None else self.recorded_prefix.native_checkpoint_sha256
                if expected_checkpoint is not None:
                    ready_fields.add("native_prefix_checkpoint_sha256")
                if (not isinstance(ready, dict) or set(ready) != ready_fields
                        or ready["ready"] is not True or tuple(ready["actors"]) != ACTORS
                        or (expected_checkpoint is not None and ready.get("native_prefix_checkpoint_sha256") != expected_checkpoint)):
                    raise HostFailure("LOCAL_HOST_READY_INVALID")
                self.state, self.phase = "DISCOVERY", "discovery"
            while self.state != "TERMINAL":
                if (self.recorded_prefix is not None and self.recorded_prefix.check_progress(self)
                        and pause_at_prefix_boundary):
                    self.status = "PAUSED_AT_VERIFIED_PREFIX"
                    self._prefix_paused = True
                    break
                if len(self.attempts) >= self.attempt_limit:
                    raise HostFailure("LOCAL_HOST_ATTEMPT_LIMIT_REACHED")
                actor = self._select_actor()
                view = self._view(actor)
                expected_phase = {"DISCOVERY": "discovery", "VERIFY": "verification", "VERIFY_REPLY": "verification",
                                  "PLANNER_HANDOFF": "execution", "EXECUTE": "execution", "PLANNER_REPLY": "execution"}[self.state]
                if view["phase"] != expected_phase:
                    raise HostFailure("LOCAL_HOST_PHASE_MISMATCH")
                before_used, before_state = self.verification_used[actor], self.state
                record, request = self._generate(actor, view)
                if before_state == "DISCOVERY":
                    self.discovery_actor = "executor" if actor == "planner" else "planner"
                if request is None:
                    if before_state == "VERIFY_REPLY":
                        self._finish_verification_reply(delivered=False)
                    continue
                response = self._exchange(request, record)
                record["status"] = "ACCEPTED" if response["ok"] else "CONTROLLER_REJECTED"
                if not response["ok"]:
                    self.feedback[actor] = deepcopy(response)
                    if before_state == "VERIFY_REPLY":
                        self._finish_verification_reply(delivered=False)
                    continue
                if request["kind"] == "finish":
                    self._finish(response)
                    break
                # Only ordinary actor feedback reaches a subsequent model prompt.
                self.feedback[actor] = deepcopy(response)
                after = self._view(actor)
                if before_state in {"VERIFY", "VERIFY_REPLY"}:
                    expected_increment = int(request["kind"] == "tool")
                    if self.verification_used[actor] != before_used + expected_increment:
                        raise HostFailure("LOCAL_HOST_VERIFICATION_ACK_MISMATCH")
                    if actor == "planner" and expected_increment:
                        if request["call_id"] not in after["results"]:
                            raise HostFailure("LOCAL_HOST_VERIFICATION_RESULT_MISSING")
                        self.planner_verification_ids.append(request["call_id"])
                if request["kind"] == "message" and actor == "planner":
                    self.planner_shared_ids.update(request["share_results"])
                if self.phase == "awaiting_allocation":
                    self._allocate()
                elif before_state == "VERIFY" and self.phase == "execution":
                    if self.verification_used != self.allocation:
                        raise HostFailure("LOCAL_HOST_VERIFICATION_TOTAL_MISMATCH")
                    self.state = "PLANNER_HANDOFF"
                elif (before_state == "VERIFY" and request["kind"] == "message"
                      and self.verification_message_replies
                      and (self.recorded_prefix is None or self.recorded_prefix.verified)
                      and request.get("recipient") in ACTORS and request["recipient"] != actor
                      and isinstance(response["result"], dict)
                      and response["result"].get("delivered_to") == request["recipient"]):
                    self._verification_reply = {"sender": actor, "recipient": request["recipient"],
                                                "trigger_attempt_index": record["attempt_index"]}
                    self.verification_reply_slots += 1
                    self.state = "VERIFY_REPLY"
                elif before_state == "VERIFY_REPLY":
                    # One generation consumes the slot; replies never schedule replies.
                    self._finish_verification_reply(delivered=True)
                elif before_state == "PLANNER_HANDOFF":
                    if set(self.planner_verification_ids) <= self.planner_shared_ids:
                        self.state = "EXECUTE"
                elif before_state == "PLANNER_REPLY":
                    self.state = "EXECUTE"
                elif before_state == "EXECUTE" and request["kind"] == "message":
                    self.planner_reply_slots += 1
                    self.state = "PLANNER_REPLY"
        except Exception as exc:
            self.status = "FAILED"
            self.error = {"error_type": type(exc).__name__, "error_code": str(exc) if isinstance(exc, (HostFailure, RecordedPrefixError)) else "LOCAL_HOST_UNEXPECTED_ERROR"}
            self.unknown_outcome = isinstance(exc, HostFailure) and exc.unknown_outcome
        return self.report()

    def report(self) -> dict:
        generations = deepcopy(self.actor.attempts[self._producer_start:])
        new_generations = [row for row in generations if row.get("reused_prefix_inference") is not True]
        measurements = [generation_telemetry(row) for row in new_generations]
        usage = [row["usage"] for row in measurements if row["usage"] is not None]
        seconds = [row["inference_seconds"] for row in measurements if row["inference_seconds"] is not None]
        new_host_attempts = len(self.attempts) - (len(generations) - len(new_generations))
        complete = bool(new_generations) and len(usage) == len(seconds) == len(new_generations) == new_host_attempts
        return {
            "status": self.status, "error": deepcopy(self.error), "host_state": self.state,
            "provider": "local_transformers", "model_revision": self.config["model_revision"],
            "allocation": dict(self.allocation), "allocation_attempted": self.allocation_attempted,
            "allocation_acknowledged": self.allocation_acknowledged,
            "verification_calls_used": dict(self.verification_used),
            "planner_verification_call_ids": list(self.planner_verification_ids),
            "planner_shared_verification_call_ids": sorted(set(self.planner_verification_ids) & self.planner_shared_ids),
            "planner_reply_slots": self.planner_reply_slots, "host_attempt_limit": self.attempt_limit,
            "host_attempt_count": len(self.attempts), "generation_attempt_count": len(generations),
            "new_generation_attempt_count": len(new_generations),
            "replayed_prefix_generation_count": len(generations) - len(new_generations),
            "recorded_prefix": None if self.recorded_prefix is None else self.recorded_prefix.report(),
            "evidence_table_after_prefix": self.evidence_table_after_prefix,
            **({"observation_history_encoding": OBSERVATION_DELTA_FORMAT} if self.lossless_observation_history else {}),
            **({"phase_action_decoder": True} if self.phase_action_decoder else {}),
            **({"action_decoder_bounds": dict(self.action_decoder_bounds)} if self.action_decoder_bounds else {}),
            **({"verification_message_replies": True,
                "verification_reply_slots": self.verification_reply_slots,
                "verification_reply_attempts": sum(row["host_state"] == "VERIFY_REPLY" for row in self.attempts),
                "verification_replies_completed": self.verification_replies_completed,
                "pending_verification_reply": deepcopy(self._verification_reply)}
               if self.verification_message_replies else {}),
            "submitted_generation_count": sum(row["submitted"] is True for row in self.attempts),
            "acknowledged_generation_count": sum(row["acknowledged"] for row in self.attempts),
            "controller_request_count": self.controller_request_count,
            "outstanding_request": deepcopy(self._pending), "unknown_native_outcome": self.unknown_outcome,
            "host_attempts": deepcopy(self.attempts), "generation_attempts": generations,
            "native_evaluation": deepcopy(self.native_evaluation), "terminal": deepcopy(self.terminal),
            "cost_accounting": {
                "measurement_scope": "new_host_generation_attempts_including_unsubmitted_and_rejected_outputs_excluding_recorded_prefix",
                "provider_telemetry_status": "COMPLETE_FOR_RECORDED_GENERATIONS" if complete else "PARTIAL" if usage or seconds else "NOT_RUN_RECORDED_PREFIX_ONLY" if generations and not new_generations else "UNKNOWN",
                "generations_with_token_usage": len(usage), "generations_with_inference_timing": len(seconds),
                "provider_token_usage": {key: sum(row[key] for row in usage) for key in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else None,
                "provider_inference_seconds": sum(seconds) if seconds else None,
                "complete_host_actor_native_cost_qualified": False,
            },
            "scientific_control_transfer_qualified": False, "generative_transfer_qualified": False,
            "formal_model_revision_verified": False, "formal_model_comparison_qualified": False,
            "complete_host_actor_native_cost_qualified": False,
        }

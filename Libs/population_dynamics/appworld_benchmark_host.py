"""Two independent model contexts controlling one persistent AppWorld session.

This is a development execution host, not a trained NeuronSim method. State
encoders and allocation policies are injected explicitly. The collection
encoder retains each actor's complete visible history; the collection policy
randomizes a fixed budget without task outcomes. Native scoring is absent.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import random
import time
from typing import Any

from .appworld_benchmark_collaboration import (
    CALL_ID_DESCRIPTION, ActorSnapshot, CollaborationHistory, PersistentToolSession, UnknownNativeOutcome,
)
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS
from .appworld_benchmark_transport import ChatContextBudgetExceeded
from .appworld_collaboration import ACTORS


@dataclass(frozen=True)
class ModelReply:
    message: dict[str, Any]
    completion_tokens: int | None
    usage_resolved: bool
    error: str | None = None


@dataclass(frozen=True)
class HostBudget:
    max_model_calls: int
    max_output_tokens: int
    max_completion_tokens: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("host budgets must be explicit positive integers")


class UniformVerificationPolicy:
    """One preassigned draw from seed41/42/43 at the native train-list index.

    Resetting an identical RNG per task would give each task the same allocation.
    Addressing the original ordered membership stream avoids that error without
    introducing another experiment seed. This policy never reads a score.
    """

    def __init__(self, seed: int, task_index: int):
        if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
            raise ValueError("collection seed must be 41, 42 or 43")
        if type(task_index) is not int or task_index < 0:
            raise ValueError("task_index must be its nonnegative native membership index")
        self.seed, self.task_index = seed, task_index

    def __call__(self, *, state_reports: dict[str, Any], total_budget: int) -> dict[str, int]:
        generator = random.Random(self.seed)
        planner = 0
        for _ in range(self.task_index + 1):
            planner = generator.randrange(total_budget + 1)
        return {"planner": planner, "executor": total_budget - planner}

    def metadata(self, total_budget: int) -> dict[str, Any]:
        return {"name": "uniform_verification_collection", "seed": self.seed,
                "draw_index": self.task_index, "assignment_probability": 1 / (total_budget + 1),
                "score_conditioned": False}


class FixedVerificationPolicy:
    def __init__(self, planner_calls: int):
        if type(planner_calls) is not int or planner_calls < 0:
            raise ValueError("planner_calls must be a nonnegative integer")
        self.planner_calls = planner_calls

    def __call__(self, *, state_reports: dict[str, Any], total_budget: int) -> dict[str, int]:
        if self.planner_calls > total_budget:
            raise ValueError("fixed planner allocation exceeds the total budget")
        return {"planner": self.planner_calls, "executor": total_budget - self.planner_calls}

    def metadata(self, total_budget: int) -> dict[str, Any]:
        return {"name": "fixed_verification_control", "planner_calls": self.planner_calls,
                "score_conditioned": False}


def full_visible_history(snapshot: ActorSnapshot) -> dict[str, Any]:
    return {"representation": "full_visible_history", **snapshot.as_model_input()}


PROMPT_PROJECTION_VERSION = "appworld_lossless_prompt_projection_v1"


def _exact_json(value: Any) -> str:
    # Python equality merges True, 1 and 1.0; JSON identity must retain them.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def prompt_visible_history(snapshot: ActorSnapshot) -> dict[str, Any]:
    """Reference exact duplicate content within this actor's existing prefix.

    The execution ledger and controller reports retain their original contents.
    First local observations remain explicit. Every reference points backward.
    """
    original = snapshot.as_model_input()
    projected = deepcopy(original)
    projected["prompt_projection"] = PROMPT_PROJECTION_VERSION
    anchors: dict[str, tuple[str, int, list[str]]] = {}
    for position, event in enumerate(original["events"]):
        payload = event["payload"]
        rendered = projected["events"][position]["payload"]
        if event["kind"] == "state_report_delivered":
            expected = {
                "representation": "full_visible_history",
                "actor": snapshot.actor,
                "events": original["events"][:position],
            }
            if _exact_json(payload["representation"]) == _exact_json(expected):
                rendered["representation"]["events"] = {
                    "$history_ref": {"kind": "prefix", "end_exclusive": position}
                }
        elif event["kind"] == "tool_observation":
            anchors.setdefault(
                payload["result_id"],
                (_exact_json(payload["observation"]), position, ["payload", "observation"]),
            )
        elif event["kind"] in {"message_sent", "message_delivered"}:
            for result_id, observation in payload["shared_results"].items():
                signature = _exact_json(observation)
                prior = anchors.get(result_id)
                if prior is not None and prior[0] == signature:
                    rendered["shared_results"][result_id] = {
                        "$history_ref": {
                            "kind": "observation", "event_index": prior[1], "path": prior[2]
                        }
                    }
                elif event["kind"] == "message_delivered" and prior is None:
                    anchors[result_id] = (
                        signature, position, ["payload", "shared_results", result_id]
                    )
    return projected


def source_api_methods(api_docs: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """Read pinned AppWorld public standard-doc records; never guess from names."""
    if not isinstance(api_docs, Mapping) or not api_docs:
        raise ValueError("source API docs must be a nonempty mapping")
    methods = {}
    for app, app_docs in api_docs.items():
        if not isinstance(app, str) or not isinstance(app_docs, Mapping):
            raise ValueError("source app docs must be named mappings")
        for api, record in app_docs.items():
            if (
                not isinstance(api, str) or not isinstance(record, Mapping)
                or record.get("app_name") != app or record.get("api_name") != api
                or not isinstance(record.get("method"), str)
            ):
                raise ValueError("source API record identity/method is missing or inconsistent")
            methods[(app, api)] = record["method"]
    return methods


_BASE_INSTRUCTIONS = (
    "Solve the user's AppWorld task through the supplied public APIs. You are one of two actors, "
    "planner and executor. Use only your visible events and explicit messages. A peer's unshared "
    "history is unavailable. Source API examples are examples, not observed credentials or IDs. "
    "The host executes each generated tool object in the active AppWorld world and returns its "
    "observation as a visible event. Both actors may call documented public APIs during ordinary "
    "interaction. API discovery belongs to the api_docs app, including show_api_descriptions; "
    "copy application API names and required arguments from the returned source documentation. "
    "Obtain authentication information through documented public APIs and use observed values. "
    "All actions execute in the same persistent world; failed actions may leave earlier changes in place. "
    "Return exactly one JSON object and no markdown. A tool action is "
    "{\"kind\":\"tool\",\"call_id\":\"call_1\",\"app\":\"app_name\",\"api\":\"api_name\",\"arguments\":{}}. "
    f"{CALL_ID_DESCRIPTION} "
    "A message is {\"kind\":\"message\",\"text\":\"message\",\"share_results\":[\"actor/call_id\"]}. "
    "Messages go to your peer. Include the successful visible result IDs needed by the peer in "
    "share_results, including when the message also explains the result in text. These IDs give "
    "the peer access to structured result references. Share only visible result IDs. "
    "Arguments may reference a successful "
    "visible result as {\"$result\":\"actor/call_id\",\"path\":[\"field\"]}. "
    "The planner can request the single extra-verification round with "
    "{\"kind\":\"propose\",\"text\":\"plan and what needs checking\",\"share_results\":[]}. "
    "The executor can stop with {\"kind\":\"finish\"}; this does not call a native completion API. "
    "When the task requires source completion, perform the appropriate public API action yourself. "
)


class CollaborationHost:
    """Schedule generated actions and a single learned/control allocation boundary.

    A planner proposal triggers state export, allocation, GET verification and
    generated reports to the peer. Unused assigned calls are retained as unused.
    The ordinary interaction phase permits both actors to use public APIs.
    Failure to reach the allocation boundary remains visible in the receipt.
    """

    def __init__(
        self, session: PersistentToolSession, generate: Callable[..., ModelReply], *,
        budget: HostBudget, state_encoder: Callable[[ActorSnapshot], dict[str, Any]],
        allocation_policy: Callable[..., Mapping[str, int]],
        source_finished: Callable[[], bool], on_progress: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.session, self.history = session, session.history
        self.generate, self.budget = generate, budget
        self.state_encoder, self.allocation_policy = state_encoder, allocation_policy
        self.source_finished, self.on_progress = source_finished, on_progress
        self.generations: list[dict[str, Any]] = []
        self.state_exports: list[dict[str, Any]] = []
        self.model_calls = self.output_tokens = 0
        self.phase, self.actor = "interact", "planner"
        self.status, self.stop_reason = "running", None
        self.native_task_completed = False
        self._verification_index = 0
        self._started = False

    def report(self) -> dict[str, Any]:
        return deepcopy({
            "status": self.status, "stop_reason": self.stop_reason,
            "phase": self.phase, "next_actor": self.actor,
            "native_task_completed": self.native_task_completed,
            "model_call_count": self.model_calls, "resolved_completion_tokens": self.output_tokens,
            "generations": self.generations, "state_exports": self.state_exports,
            "prompt_projection": PROMPT_PROJECTION_VERSION,
            "allocation": self.session.allocation, "verification_used": self.session.verification_used,
            "native_attempts": self.session.attempts,
            "actor_histories": {actor: self.history.snapshot(actor).as_model_input() for actor in ACTORS},
        })

    def _progress(self) -> None:
        if self.on_progress is not None:
            self.on_progress(self.report())

    def _instructions(self) -> str:
        instruction = _BASE_INSTRUCTIONS + (
            f"This prompt uses {PROMPT_PROJECTION_VERSION}. Event indices are unchanged. "
            "A $history_ref of kind prefix denotes the original local events from index 0 "
            "up to, but excluding, end_exclusive. A reference of kind observation denotes "
            "the exact observation at the indicated earlier local event_index and field path. "
            "References replace duplicate content only. The containing event still records "
            "when a state or message was delivered. References add no observations, tool calls "
            "or permissions. Existing action rules remain unchanged. "
            f"Your actor is {self.actor}. Current phase is {self.phase}. "
        )
        if self.phase == "verify":
            remaining = self.session.allocation[self.actor] - self.session.verification_used[self.actor]
            instruction += (f"You have {remaining} assigned GET verification calls left. "
                            "Use tool for source-declared GET APIs or message your peer. "
                            "If no further check is useful, return {\"kind\":\"verification_done\"}. "
                            "Writes, proposals and finish are unavailable during verification.")
        elif self.phase == "verification_handoff":
            instruction += "Send one message to your peer with the material verification findings and their visible result IDs. Only message is available in this handoff."
        elif self.session.allocation is not None:
            instruction += "The extra-verification allocation has already been used. Continue ordinary task execution; a second proposal is unavailable."
        else:
            instruction += (
                "Resolve information requests with documented tool actions or by sharing results already "
                "visible to you. If a peer repeats a request, identify the missing observation and obtain "
                "or share it. Perform authentication actions that require non-GET APIs in this ordinary "
                "phase. When unresolved information needs extra verification, the planner should generate "
                "kind=propose with the candidate plan and specific source-declared GET checks. "
                "The executor should complete the requested API work and required source completion "
                "actions before generating finish. "
            )
        return instruction

    @staticmethod
    def _parse_action(message: Mapping[str, Any]) -> dict[str, Any]:
        if message.get("tool_calls"):
            raise ValueError("this host requires its declared JSON action protocol")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("model action content must be text")
        action = json.loads(content)
        fields = {
            "tool": {"kind", "call_id", "app", "api", "arguments"},
            "message": {"kind", "text", "share_results"},
            "propose": {"kind", "text", "share_results"},
            "finish": {"kind"}, "verification_done": {"kind"},
        }
        if not isinstance(action, dict) or not isinstance(action.get("kind"), str) or set(action) != fields.get(action["kind"]):
            raise ValueError("action must have exactly its declared fields")
        if action["kind"] in {"message", "propose"} and (
            not isinstance(action["text"], str) or not action["text"].strip()
            or not isinstance(action["share_results"], list)
            or any(not isinstance(ref, str) for ref in action["share_results"])
        ):
            raise ValueError("message requires text and a list of visible result IDs")
        return action

    def _peer_message(self, action: dict[str, Any]) -> None:
        peer = ACTORS[1 - ACTORS.index(self.actor)]
        message_id = self.history.send_message(self.actor, peer, action["text"], share_results=tuple(action["share_results"]))
        self.history.deliver_message(message_id)

    def _next_verifier(self) -> None:
        while self._verification_index < len(ACTORS):
            self.actor = ACTORS[self._verification_index]
            if self.session.allocation[self.actor] > 0:
                self.phase = "verify"
                return
            self._verification_index += 1
        self.phase, self.actor = "execute", "executor"

    def _allocate(self) -> None:
        # Take both immutable prefixes at the same boundary before either state
        # report or any future verification result is appended.
        snapshots = {actor: self.history.snapshot(actor) for actor in ACTORS}
        for actor in ACTORS:
            started = time.clock_gettime(time.CLOCK_MONOTONIC)
            representation = self.state_encoder(snapshots[actor])
            self.history.deliver_state_report(actor, representation)
            self.state_exports.append({
                "actor": actor, "prefix_events": len(snapshots[actor].event_json),
                "wall_seconds": time.clock_gettime(time.CLOCK_MONOTONIC) - started,
                "representation": deepcopy(representation),
            })
        self.session.allocate(self.allocation_policy)
        self._next_verifier()

    def _dispatch(self, action: dict[str, Any]) -> None:
        kind = action["kind"]
        if self.phase == "verification_handoff" and kind != "message":
            raise ValueError("verification handoff requires a peer message")
        if self.phase == "verify" and kind not in {"tool", "message", "verification_done"}:
            raise ValueError("action is unavailable during verification")
        if kind == "tool":
            call = self.session.verify if self.phase == "verify" else self.session.execute
            call(self.actor, action["call_id"], action["app"], action["api"], action["arguments"])
            completed = self.source_finished()
            if type(completed) is not bool:
                raise TypeError("native task_completed must return a bool")
            self.native_task_completed = completed
            if completed:
                self.status, self.stop_reason = "completed", "native_task_completed"
                return
            if self.phase == "verify":
                if self.session.verification_used[self.actor] == self.session.allocation[self.actor]:
                    self.phase = "verification_handoff"
                return
        elif kind == "message":
            self._peer_message(action)
            if self.phase == "verification_handoff":
                self._verification_index += 1
                self._next_verifier()
                return
            if self.phase == "verify":
                return
        elif kind == "propose":
            if self.actor != "planner" or self.session.allocation is not None:
                raise ValueError("only the planner can request the single verification boundary")
            self._peer_message(action)
            try:
                self._allocate()
            except Exception as error:
                # The proposal and state exports may already be delivered.
                # A policy failure is not a reversible model-action rejection.
                raise RuntimeError("state export or allocation policy failed") from error
            return
        elif kind == "verification_done":
            if self.phase != "verify":
                raise ValueError("verification_done requires the verification phase")
            self.phase = "verification_handoff"
            return
        elif kind == "finish":
            if self.actor != "executor":
                raise ValueError("only the executor can finish")
            self.status, self.stop_reason = "completed", "executor_stopped"
            return
        self.actor = ACTORS[1 - ACTORS.index(self.actor)]

    def run(self) -> dict[str, Any]:
        if self._started:
            raise ValueError("a host cannot repeat an attempt")
        self._started = True
        self._progress()
        while self.status == "running":
            remaining = self.budget.max_output_tokens - self.output_tokens
            if self.model_calls >= self.budget.max_model_calls or remaining <= 0:
                self.status, self.stop_reason = "agent_failure", "model_budget_exhausted"
                break
            snapshot = self.history.snapshot(self.actor)
            messages = [
                {"role": "system", "content": self._instructions()},
                {"role": "user", "content": json.dumps(prompt_visible_history(snapshot), ensure_ascii=False, allow_nan=False)},
            ]
            row = {
                "actor": self.actor, "phase": self.phase,
                "prefix_events": len(snapshot.event_json), "status": "pending",
                "prompt_projection": PROMPT_PROJECTION_VERSION,
            }
            self.generations.append(row)
            self.model_calls += 1
            self._progress()
            limit = min(remaining, self.budget.max_completion_tokens)
            try:
                reply = self.generate(actor=self.actor, messages=messages, max_completion_tokens=limit)
            except ChatContextBudgetExceeded as error:
                row.update(
                    status="pre_generation_rejected", error=str(error),
                    completion_tokens=None, usage_resolved=False,
                    generation_started=False, generated_completion_tokens=0,
                    transport_request_id=error.request_id,
                    context_budget_rejection=deepcopy(error.context_budget_rejection),
                )
                self.status, self.stop_reason = "agent_failure", "context_budget_exhausted"
                break
            if not isinstance(reply, ModelReply):
                raise TypeError("generator must return a ModelReply")
            row.update(status="returned", message=deepcopy(reply.message), error=reply.error,
                       usage_resolved=reply.usage_resolved, completion_tokens=reply.completion_tokens)
            self.history.record_generated_action(self.actor, {"model_message": reply.message})
            resolved = reply.usage_resolved is True and type(reply.completion_tokens) is int and reply.completion_tokens >= 0
            if resolved:
                self.output_tokens += reply.completion_tokens
            if reply.error or not resolved:
                self.status, self.stop_reason = "agent_failure", "model_error_or_unresolved_usage"
                break
            if reply.completion_tokens > limit:
                self.status, self.stop_reason = "agent_failure", "provider_exceeded_completion_limit"
                break
            self._progress()
            before_attempts = len(self.session.attempts)
            try:
                action = self._parse_action(reply.message)
                row["action"] = deepcopy(action)
                self._dispatch(action)
            except UnknownNativeOutcome:
                self.status, self.stop_reason = "unknown", "native_outcome_unresolved"
                break
            except ValueError as error:
                if len(self.session.attempts) != before_attempts:
                    raise  # A post-execution failure is not a rejected action.
                row.update(action_status="rejected", rejection=str(error))
                self.history.record_rejection(self.actor, str(error))
            else:
                row["action_status"] = "accepted"
            self._progress()
        self._progress()
        return self.report()

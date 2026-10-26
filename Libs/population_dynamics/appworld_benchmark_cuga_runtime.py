"""Observe the pinned CUGA graph without replacing its planning or execution.

The caller supplies a configured, fresh graph, native tool bindings and public
initial state. This module records callback usage and original graph identities.
Callback coverage still needs comparison with the actual model transport before
these costs qualify for a benchmark. Graph termination does not score a task.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _token_counts(usage: Any) -> dict[str, int] | None:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, Mapping):
        return None
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in fields):
        return None
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        return None
    return {key: usage[key] for key in fields}


class CugaUsageLedger:
    """Retain each LangChain callback and distinguish missing usage from zero."""

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}
        self._started: dict[str, float] = {}

    @property
    def calls(self) -> list[dict[str, Any]]:
        return deepcopy(list(self._calls.values()))

    def start(self, run_id: Any, *, parent_run_id=None, metadata=None,
              tags=None, messages=None, prompts=None) -> None:
        key = str(run_id)
        if key in self._calls:
            raise ValueError("duplicate CUGA model callback run_id")
        row = {
            "run_id": key, "parent_run_id": None if parent_run_id is None else str(parent_run_id),
            "metadata": _copy_json(metadata), "tags": _copy_json(tags),
            "messages": _copy_json(messages), "prompts": _copy_json(prompts),
            "start_observed": True, "status": "pending", "usage_status": "unresolved",
            "raw_usage": None, "llm_output": None, "message_metadata": [],
        }
        self._calls[key] = row
        self._started[key] = time.clock_gettime(time.CLOCK_MONOTONIC)

    def finish(self, run_id: Any, response: Any = None, *, error: BaseException | None = None) -> None:
        key = str(run_id)
        row = self._calls.get(key)
        if row is None:
            self.start(key)
            row = self._calls[key]
            row["start_observed"] = False
            self._started.pop(key)
        if row["status"] != "pending":
            raise ValueError("duplicate CUGA model callback terminal event")
        row["status"] = "errored" if error is not None else "completed"
        if error is not None:
            row.update(error_type=type(error).__name__, error=str(error))
        if key in self._started:
            row["wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - self._started.pop(key)
        if response is not None:
            row["llm_output"] = _copy_json(response.llm_output)
            if isinstance(row["llm_output"], dict):
                row["raw_usage"] = deepcopy(row["llm_output"].get("token_usage"))
            generations = response.generations or []
            row["generation_batch_sizes"] = [len(batch) for batch in generations]
            for batch in generations:
                for generation in batch:
                    message = getattr(generation, "message", None)
                    row["message_metadata"].append({
                        "usage_metadata": _copy_json(getattr(message, "usage_metadata", None)),
                        "response_metadata": _copy_json(getattr(message, "response_metadata", None)),
                    })
        counts = _token_counts(row["raw_usage"])
        row["usage_status"] = "resolved" if counts is not None else "unresolved"
        row["counts"] = counts
        # A single returned message supplies an independent normalized view.
        # Preserve disagreements rather than choosing whichever is favorable.
        if counts is not None and len(row["message_metadata"]) == 1:
            metadata = row["message_metadata"][0]["usage_metadata"]
            if isinstance(metadata, dict):
                expected = {"input_tokens": counts["prompt_tokens"],
                            "output_tokens": counts["completion_tokens"],
                            "total_tokens": counts["total_tokens"]}
                if any(type(metadata.get(name)) is not int or metadata[name] != count
                       for name, count in expected.items()):
                    row["usage_status"] = "conflicting"

    def summary(self) -> dict[str, Any]:
        calls = self.calls
        resolved = bool(calls) and all(row["status"] != "pending" and row["start_observed"]
                                       and row["usage_status"] == "resolved" for row in calls)
        totals = ({key: sum(row["counts"][key] for row in calls)
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                  if resolved else None)
        return {"source": "langchain_model_callbacks", "recorded_calls": len(calls),
                "recorded_usage_resolved": resolved, "totals": totals,
                "transport_coverage_qualified": False}


def make_usage_callback(ledger: CugaUsageLedger):
    # CUGA's LangChain dependency lives in its separate prepared environment.
    from langchain_core.callbacks import AsyncCallbackHandler

    class UsageCallback(AsyncCallbackHandler):
        raise_error = True

        async def on_chat_model_start(self, serialized, messages, *, run_id,
                                      parent_run_id=None, tags=None, metadata=None, **kwargs):
            ledger.start(run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata,
                         messages=[[message.model_dump(mode="json") for message in batch]
                                   for batch in messages])

        async def on_llm_start(self, serialized, prompts, *, run_id,
                               parent_run_id=None, tags=None, metadata=None, **kwargs):
            ledger.start(run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, prompts=prompts)

        async def on_llm_end(self, response, *, run_id, **kwargs):
            ledger.finish(run_id, response)

        async def on_llm_error(self, error, *, run_id, **kwargs):
            ledger.finish(run_id, kwargs.get("response"), error=error)

    return UsageCallback()


class ObservedCugaGraph:
    """Add one callback to the source loop and retain unrenamed graph events."""

    def __init__(self, graph: Any, usage_callback: Any) -> None:
        self.graph = graph
        self.usage_callback = usage_callback
        self.events: list[dict[str, Any]] = []

    def __getattr__(self, name):
        return getattr(self.graph, name)

    async def astream(self, state, *, config, **kwargs):
        observed = dict(config)
        observed["callbacks"] = [*config.get("callbacks", []), self.usage_callback]
        async for event in self.graph.astream(state, config=observed, **kwargs):
            namespace, update = event if isinstance(event, tuple) else ((), event)
            if not isinstance(update, Mapping):
                raise TypeError("CUGA graph update must preserve its source node mapping")
            self.events.append({"namespace": list(namespace), "node_keys": list(update)})
            yield event

    def termination(self, answer: Any, *, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        answer_fields = _copy_json(answer.model_dump(mode="json"))
        pending = list(snapshot.next)
        last_nodes = self.events[-1]["node_keys"] if self.events else []
        interrupted = answer_fields.get("interrupt") is True or "__interrupt__" in last_nodes
        if interrupted:
            status = "interrupted"
        elif answer_fields.get("has_tools") is True:
            status = "awaiting_external_tools"
        elif not self.events or pending or answer_fields.get("end") is not True:
            status = "unfinished"
        else:
            status = "graph_ended"
        # The source display layer renames lite/subgraph events to CodeAgent.
        # Inspect the original namespace and keys before interpreting identity.
        lite_or_supervisor = any(
            name.split(":", 1)[0] in {"CugaLite", "CugaLiteSubgraph", "CugaSupervisor", "CugaSupervisorSubgraph"}
            for event in self.events for name in [*event["namespace"], *event["node_keys"]])
        return {"status": status, "answer": answer_fields, "pending_nodes": pending,
                "graph_events": deepcopy(self.events),
                "lite_or_supervisor_executed": lite_or_supervisor,
                "task_scoring_status": "not_run"}


async def run_observed_cuga_loop(
    loop: Any, state: Any, ledger: CugaUsageLedger, *, observation: dict[str, Any], usage_callback=None,
):
    """Run the source loop once; retain evidence on success, error or cancellation.

    No interrupt is auto-resumed here. An API-only run that asks for external
    browser tools remains interrupted and requires a recorded failure decision.
    """
    if observation:
        raise ValueError("a fresh CUGA loop observation is required")
    callback = make_usage_callback(ledger) if usage_callback is None else usage_callback
    original = loop.graph
    graph = ObservedCugaGraph(original, callback)
    loop.graph = graph
    result = observation
    result.update(status="running", formal_eligible=False)
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    try:
        answer = await loop.run(state=state)
        result.update(graph.termination(answer, thread_id=loop.thread_id))
    except BaseException as error:
        result.update(status="errored", error_type=type(error).__name__, error=str(error))
        # The caller's observation retains evidence when the original error
        # propagates. The owner must not silently score or retry the failed run.
        raise
    finally:
        loop.graph = original
        result.update(graph_events=deepcopy(graph.events), model_calls=ledger.calls,
                      model_usage=ledger.summary(),
                      graph_wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - started)
    return result


def configure_cuga_settings(settings: Any, *, seed: int) -> dict[str, Any]:
    """Keep source role budgets while selecting the admitted full API graph."""
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("CUGA experiment seed must be 41, 42 or 43")
    if settings.get("features.cuga_mode") != "balanced" or settings.get("features.task_decomposition") is not True:
        raise ValueError("load the pinned balanced mode before configuring CUGA")
    overrides = {
        "FEATURES": {"CUGA_MODE": "balanced", "CHAT": False, "SAVE_REUSE": False},
        "ADVANCED_FEATURES": {
            "REGISTRY": False, "LITE_MODE": False, "BENCHMARK": "appworld", "MODE": "api",
            "LANGFUSE_TRACING": False, "E2B_SANDBOX": False, "TRACKER_ENABLED": True,
        },
        "EVOLVE": {"ENABLED": False},
        "SUPERVISOR": {"ENABLED": False},
        "AGENT_SPAWN": {"ENABLED": False},
        "OBSERVABILITY": {"OPENLIT": False},
    }
    settings.update(overrides, merge=True)
    roles = {}
    for role, values in settings.agent.items():
        model = values.get("model")
        if model is None:
            continue
        declared = model.to_dict() if hasattr(model, "to_dict") else dict(model)
        extras = dict(declared.get("extra_params", {}))
        extras.update(seed=seed, max_retries=0)
        settings.set(f"agent.{role}.model.extra_params", extras, merge=False)
        # Streaming changes the provider usage surface; the pinned source
        # default is non-streaming. Preserve that setting explicitly.
        settings.set(f"agent.{role}.model.streaming", False)
        roles[role] = {key: deepcopy(value) for key, value in declared.items()
                       if key not in {"api_key", "base_url", "url"}}
        roles[role].update(extra_params=extras, streaming=False)
    if not roles:
        raise ValueError("the pinned CUGA role settings are absent")
    return {"overrides": overrides, "roles": roles}


def model_client_settings(manager: Any, *, model_name: str, base_url: str, seed: int) -> list[dict[str, Any]]:
    """Inspect the source factory's effective clients after graph construction."""
    records = []
    for client in manager._models.values():
        record = {
            "class": type(client).__name__, "model": getattr(client, "model_name", None),
            "base_url": str(getattr(client, "openai_api_base", "")),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
            "seed": getattr(client, "seed", None), "max_retries": getattr(client, "max_retries", None),
            "streaming": getattr(client, "streaming", None),
        }
        if (record["model"] != model_name or record["base_url"].rstrip("/") != base_url.rstrip("/")
                or type(record["seed"]) is not int or record["seed"] != seed
                or type(record["max_retries"]) is not int or record["max_retries"] != 0
                or record["streaming"] is not False):
            raise ValueError("CUGA effective model client differs from the declared local route")
        if type(record["max_tokens"]) is not int or record["max_tokens"] <= 0:
            raise ValueError("CUGA effective model token limit is absent")
        records.append(record)
    if not records:
        raise ValueError("CUGA graph constructed no observed model clients")
    return records


async def run_bound_cuga_graph(
    runtime: Any, client: Any, *, work_root: str | Path, task_id: str,
    model_name: str, base_url: str, seed: int, observation: dict[str, Any],
    usage_callback=None,
) -> dict[str, Any]:
    """Own one prepared source graph and native client through cleanup.

    The process launcher must configure the pinned CUGA settings before graph
    imports, validate the native/model preparations, and supply a fresh process.
    ``runtime`` contains the actual pinned source classes, configured settings,
    factory manager and import-time SaveReuse constants. This boundary does not
    qualify a launcher, model transport, task receipt or evaluator by itself.
    """
    from .appworld_benchmark_cuga_tools import CugaNativeToolBindings

    if not isinstance(observation, dict) or observation:
        await client.close()
        raise ValueError("a fresh CUGA task observation is required")
    observation.update(status="initializing", formal_eligible=False, task_id=task_id, seed=seed)
    tracker = None
    binding = None
    loop_observation: dict[str, Any] = {}
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    try:
        root = Path(work_root).expanduser().absolute()
        observation["work_root"] = str(root)
        if type(seed) is not int or seed not in (41, 42, 43):
            raise ValueError("CUGA experiment seed must be 41, 42 or 43")
        if not isinstance(task_id, str) or not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
            raise ValueError("CUGA requires one native task identifier")
        if tuple(runtime.save_reuse_constants) != (False, False):
            raise ValueError("CUGA SaveReuse constants must be disabled before graph imports")
        settings = runtime.settings
        expected = {"advanced_features.registry": False, "advanced_features.lite_mode": False,
                    "advanced_features.langfuse_tracing": False, "advanced_features.benchmark": "appworld",
                    "advanced_features.mode": "api", "features.save_reuse": False,
                    "features.chat": False, "features.task_decomposition": True,
                    "advanced_features.e2b_sandbox": False, "supervisor.enabled": False,
                    "agent_spawn.enabled": False, "evolve.enabled": False}
        if any(settings.get(key) != value for key, value in expected.items()):
            raise ValueError("CUGA full API graph and task-isolation settings differ")
        context = _copy_json(client.public_context)
        task_date = context["current_datetime"]
        if not isinstance(task_date, str) or not task_date:
            raise ValueError("the native task datetime is required")
        root.mkdir(parents=True, exist_ok=False)
        trajectories = root / "trajectories"
        (trajectories / "current").mkdir(parents=True)
        tracker = runtime.ActivityTracker()
        tracker.reset(context["instruction"], task_id=task_id)
        tracker.set_base_dir(str(trajectories))
        tracker.experiment_folder = "current"
        tracker.current_date = task_date
        tracker.pi = json.dumps(context["supervisor"], ensure_ascii=False, allow_nan=False)
        binding = CugaNativeToolBindings(
            tracker, api_docs=context["api_docs"], function_calling=context["function_calling"],
            app_descriptions=context["app_descriptions"], executor=client,
            app_definition=runtime.AppDefinition,
        )
        with binding:
            agent = runtime.DynamicAgentGraph(None, langfuse_handler=None)
            await agent.build_graph()
            observation["effective_model_clients"] = model_client_settings(
                runtime.manager, model_name=model_name, base_url=base_url, seed=seed)
            state = runtime.default_state(page=None, observation=None, goal=context["instruction"])
            state.sites = None
            state.current_datetime = task_date
            state.pi = tracker.pi
            thread_id = f"{task_id}-seed{seed}"
            state.thread_id = thread_id
            loop = runtime.AgentLoop(thread_id=thread_id, langfuse_handler=None, graph=agent.graph,
                                     tracker=tracker, env_pointer=None, policy_system=agent.policy_system)
            ledger = CugaUsageLedger()
            await run_observed_cuga_loop(loop, state, ledger, observation=loop_observation,
                                        usage_callback=usage_callback)
            observation["status"] = ("unexpected_graph_variant" if loop_observation["lite_or_supervisor_executed"]
                                     else loop_observation["status"])
    except BaseException as error:
        observation.update(status="errored", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observation["loop"] = loop_observation
        observation["native_tool_calls"] = [] if binding is None else deepcopy(binding.calls)
        try:
            cleanup = await client.close()
            observation["native_cleanup"] = cleanup
            settled = (cleanup.get("child_reaped") is True and cleanup.get("exit_code") == 0
                       and cleanup.get("request_outcome_unresolved") is False
                       and cleanup.get("termination_signals") == [])
            observation["native_cleanup_settled"] = settled
            if not settled and observation["status"] != "errored":
                observation["status"] = "native_execution_unresolved"
        except BaseException as error:
            observation["cleanup_error"] = {"error_type": type(error).__name__, "error": str(error)}
            observation["status"] = "cleanup_failed"
            raise
        finally:
            if tracker is not None:
                tracker.set_tools([])
                tracker.reset("")
                tracker.experiment_folder = None
            observation["bound_graph_wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started
    return observation


def load_cuga_runtime(*, seed: int, model_name: str, base_url: str) -> Any:
    """Construct the pinned source interfaces after configuring a fresh process.

    The launcher owns package/source admission and the task-local CUGA paths.
    Import order is significant because SaveReuse reads settings at import time.
    """
    frozen_modules = (
        "cuga.backend.cuga_graph.graph",
        "cuga.backend.cuga_graph.nodes.answer.final_answer",
        "cuga.backend.cuga_graph.nodes.chat.chat",
    )
    if any(name in sys.modules for name in frozen_modules):
        raise RuntimeError("CUGA graph imports require a fresh task process")
    from cuga.config import base_settings, modes_file_path, settings

    # The pinned loader selects the mode from base_settings. Its later mode
    # table replaces FEATURES and does not retain the cuga_mode selection key.
    source_mode = base_settings.get("features.cuga_mode")
    if source_mode != "balanced" or Path(modes_file_path).name != "balanced.toml":
        raise ValueError("the pinned CUGA loader must select balanced.toml")
    settings.set("features.cuga_mode", source_mode)
    declared = configure_cuga_settings(settings, seed=seed)
    declared["source_mode"] = {"name": source_mode, "path": str(modes_file_path)}
    from cuga.backend.llm import models

    manager = models.LLMManager()
    manager.clear_pre_instantiated_model()
    manager._models.clear()
    os.environ["CUGA_LOCAL_API_KEY"] = "local-model-placeholder"
    models.set_current_llm_override({
        "platform": "openai", "model": model_name, "base_url": base_url,
        "api_key": "CUGA_LOCAL_API_KEY",
    })
    from cuga.backend.activity_tracker.tracker import ActivityTracker
    from cuga.backend.cuga_graph.graph import DynamicAgentGraph
    from cuga.backend.cuga_graph.nodes.answer import final_answer
    from cuga.backend.cuga_graph.nodes.chat import chat
    from cuga.backend.cuga_graph.state.agent_state import default_state
    from cuga.backend.cuga_graph.utils.agent_loop import AgentLoop
    from cuga.backend.tools_env.registry.utils.types import AppDefinition

    return SimpleNamespace(
        settings=settings, manager=manager, declared_settings=declared,
        save_reuse_constants=(final_answer.ENABLE_SAVE_REUSE, chat.ENABLE_SAVE_REUSE),
        ActivityTracker=ActivityTracker, DynamicAgentGraph=DynamicAgentGraph,
        AgentLoop=AgentLoop, default_state=default_state, AppDefinition=AppDefinition,
    )

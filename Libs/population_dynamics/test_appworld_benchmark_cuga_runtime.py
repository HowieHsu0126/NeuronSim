"""Callback and graph-observation tests; no CUGA model or native task runs."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from population_dynamics.appworld_benchmark_cuga_runtime import CugaUsageLedger, run_observed_cuga_loop


def response(usage=None, metadata=None):
    message = SimpleNamespace(usage_metadata=metadata, response_metadata={"model_name": "fixture"})
    return SimpleNamespace(llm_output={"token_usage": usage, "model_name": "fixture"},
                           generations=[[SimpleNamespace(message=message)]])


def test_usage_preserves_raw_details_and_separate_calls():
    ledger = CugaUsageLedger()
    raw = {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11,
           "prompt_tokens_details": {"cached_tokens": 2}}
    ledger.start("first", parent_run_id="parent", metadata={"langgraph_node": "APIPlannerAgent"})
    ledger.finish("first", response(raw, {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}))
    ledger.start("second")
    ledger.finish("second", response({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
    raw["total_tokens"] = 1000
    assert ledger.calls[0]["raw_usage"]["total_tokens"] == 11
    assert ledger.calls[0]["raw_usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    assert ledger.calls[0]["parent_run_id"] == "parent"
    assert ledger.calls[0]["wall_seconds"] >= 0
    assert ledger.summary()["totals"] == {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert ledger.summary()["transport_coverage_qualified"] is False


@pytest.mark.parametrize("raw", [None, {}, {"total_tokens": 3},
    {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3},
    {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 1},
    {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4}])
def test_missing_or_invalid_usage_cannot_be_counted_as_zero(raw):
    ledger = CugaUsageLedger()
    ledger.start("call")
    ledger.finish("call", response(raw, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}))
    assert ledger.calls[0]["raw_usage"] == raw
    assert ledger.calls[0]["usage_status"] == "unresolved"
    assert ledger.summary()["totals"] is None


def test_conflicting_normalized_usage_is_retained():
    ledger = CugaUsageLedger()
    ledger.start("call")
    ledger.finish("call", response({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                                   {"input_tokens": 1, "output_tokens": 3, "total_tokens": 4}))
    assert ledger.calls[0]["usage_status"] == "conflicting"
    assert ledger.summary()["totals"] is None


def test_failure_pending_and_missing_start_keep_accounting_unresolved():
    ledger = CugaUsageLedger()
    assert ledger.summary()["totals"] is None
    ledger.start("failed")
    ledger.finish("failed", error=TimeoutError("model completion unknown"))
    ledger.start("pending")
    ledger.finish("orphan", response({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}))
    calls = ledger.calls
    assert calls[0]["status"] == "errored" and calls[0]["error_type"] == "TimeoutError"
    assert calls[1]["status"] == "pending"
    assert calls[2]["start_observed"] is False
    assert ledger.summary()["totals"] is None
    with pytest.raises(ValueError, match="duplicate"):
        ledger.start("pending")
    with pytest.raises(ValueError, match="duplicate"):
        ledger.finish("failed")


class Graph:
    def __init__(self, events, pending=(), failure=None):
        self.events = events
        self.pending = pending
        self.failure = failure
        self.requests = []

    async def astream(self, state, **kwargs):
        self.requests.append((state, kwargs))
        for event in self.events:
            yield event
        if self.failure is not None:
            raise self.failure

    def get_state(self, config):
        assert config == {"configurable": {"thread_id": "synthetic-thread"}}
        return SimpleNamespace(next=self.pending)


class Loop:
    def __init__(self, graph, answer):
        self.graph = graph
        self.thread_id = "synthetic-thread"
        self.answer = answer
        self.callback = object()
        self.config = {"recursion_limit": 135, "callbacks": [self.callback],
                       "configurable": {"thread_id": self.thread_id, "policy_system": object()}}
        self.seen = []
        self.runs = 0

    async def run(self, state):
        self.runs += 1
        async for event in self.graph.astream(state, config=self.config, stream_mode="updates", subgraphs=True):
            self.seen.append(event)
        return SimpleNamespace(model_dump=lambda **kwargs: deepcopy(self.answer))


@pytest.mark.parametrize("events,pending,answer,expected", [
    ([((), {"FinalAnswerAgent": {}})], (), {"end": True, "interrupt": False, "has_tools": False}, "graph_ended"),
    ([((), {"__interrupt__": {}})], (), {"end": True, "interrupt": True, "has_tools": False}, "interrupted"),
    ([((), {"ActionAgent": {}})], (), {"end": False, "has_tools": True}, "awaiting_external_tools"),
    ([((), {"CodeAgent": {}})], ("PlanControllerAgent",), {"end": True, "has_tools": False}, "unfinished"),
    ([], (), {"end": True, "has_tools": False}, "unfinished"),
])
def test_source_stream_and_termination_are_preserved(events, pending, answer, expected):
    graph = Graph(events, pending)
    loop = Loop(graph, answer)
    state, callback, observation = object(), object(), {}
    result = asyncio.run(run_observed_cuga_loop(loop, state, CugaUsageLedger(),
                         observation=observation, usage_callback=callback))
    assert result is observation and result["status"] == expected
    assert result["task_scoring_status"] == "not_run"
    assert loop.graph is graph and loop.runs == 1
    assert loop.seen == events
    submitted_state, options = graph.requests[0]
    assert submitted_state is state
    assert options["config"]["recursion_limit"] == 135
    assert options["config"]["configurable"] is loop.config["configurable"]
    assert options["config"]["callbacks"] == [loop.callback, callback]
    assert loop.config["callbacks"] == [loop.callback]
    assert options["stream_mode"] == "updates" and options["subgraphs"] is True


def test_original_lite_namespace_survives_display_name_collision():
    graph = Graph([(("CugaLiteSubgraph:instance",), {"CodeAgent": {}})])
    loop = Loop(graph, {"end": True, "has_tools": False})
    result = asyncio.run(run_observed_cuga_loop(loop, {}, CugaUsageLedger(),
                         observation={}, usage_callback=object()))
    assert result["lite_or_supervisor_executed"] is True
    assert result["graph_events"] == [{"namespace": ["CugaLiteSubgraph:instance"], "node_keys": ["CodeAgent"]}]


@pytest.mark.parametrize("error", [RuntimeError("graph failed"), asyncio.CancelledError()])
def test_error_or_cancellation_preserves_observation_without_retry(error):
    graph = Graph([((), {"APIPlannerAgent": {}})], failure=error)
    loop = Loop(graph, {})
    observation = {}
    with pytest.raises(type(error)) as raised:
        asyncio.run(run_observed_cuga_loop(loop, {}, CugaUsageLedger(),
                    observation=observation, usage_callback=object()))
    assert raised.value is error
    assert loop.graph is graph and loop.runs == 1
    assert observation["status"] == "errored"
    assert observation["graph_events"] == [{"namespace": [], "node_keys": ["APIPlannerAgent"]}]
    assert observation["model_usage"]["totals"] is None


class Settings:
    def __init__(self):
        self.values = {"features.cuga_mode": "balanced", "features.task_decomposition": True}
        self.agent = {
            "planner": {"model": {"temperature": .1, "max_tokens": 16000, "extra_params": {"top_p": .9}}},
            "action": {"model": {"temperature": .1, "max_tokens": 400}},
        }

    def update(self, values, merge):
        assert merge is True
        for section, fields in values.items():
            for key, value in fields.items():
                self.values[f"{section}.{key}".lower()] = value

    def set(self, key, value, **kwargs):
        parts = key.split(".")
        self.agent[parts[1]]["model"][parts[-1]] = deepcopy(value)

    def get(self, key):
        return self.values.get(key)


def test_source_role_settings_keep_different_budgets_and_existing_sampling():
    from population_dynamics.appworld_benchmark_cuga_runtime import configure_cuga_settings
    settings = Settings()
    record = configure_cuga_settings(settings, seed=42)
    assert settings.agent["planner"]["model"]["max_tokens"] == 16000
    assert settings.agent["action"]["model"]["max_tokens"] == 400
    assert record["roles"]["planner"]["extra_params"] == {"top_p": .9, "seed": 42, "max_retries": 0}
    assert record["roles"]["action"]["extra_params"] == {"seed": 42, "max_retries": 0}
    assert settings.get("advanced_features.registry") is False
    assert settings.get("evolve.enabled") is False
    settings.values["features.task_decomposition"] = False
    with pytest.raises(ValueError, match="balanced mode"):
        configure_cuga_settings(settings, seed=42)


def prepared_runtime():
    from population_dynamics.appworld_benchmark_cuga_runtime import configure_cuga_settings
    settings = Settings()
    configure_cuga_settings(settings, seed=41)

    class Tracker:
        def __init__(self):
            self.tools = {}
            self.resets = []

        def reset(self, intent, task_id="default"):
            self.resets.append((intent, task_id))
            self.pi = None
            self.current_date = None
            self.task_id = task_id

        def set_base_dir(self, value):
            self.base_dir = value

        def set_tools(self, tools):
            self.tools = {}
            for tool in tools:
                self.tools.setdefault(tool.metadata["server_name"], []).append(tool)

    tracker = Tracker()
    model = SimpleNamespace(model_name="fixture-model", openai_api_base="http://127.0.0.1:1234/v1",
                            temperature=.1, max_tokens=16000, seed=41, max_retries=0, streaming=False)
    runtime = SimpleNamespace(settings=settings, manager=SimpleNamespace(_models={"planner": model}),
                              save_reuse_constants=(False, False), ActivityTracker=lambda: tracker,
                              AppDefinition=SimpleNamespace, tracker=tracker, builds=0, runs=0,
                              failure=None, run_failure=None, captured_state=None)

    class BoundGraph(Graph):
        def __init__(self, *args, **kwargs):
            super().__init__([((), {"FinalAnswerAgent": {}})])
            assert "sample" in tracker.tools
            self.graph = self
            self.policy_system = object()
            runtime.builds += 1

        async def build_graph(self):
            if runtime.failure is not None:
                raise runtime.failure

        def get_state(self, config):
            assert config["configurable"]["thread_id"] == "fixture_1-seed41"
            return SimpleNamespace(next=())

    class BoundLoop(Loop):
        def __init__(self, **kwargs):
            super().__init__(kwargs["graph"], {"end": True, "interrupt": False, "has_tools": False})
            self.thread_id = kwargs["thread_id"]
            self.config["configurable"]["thread_id"] = self.thread_id
            assert kwargs["env_pointer"] is None

        async def run(self, state):
            runtime.runs += 1
            runtime.captured_state = deepcopy(vars(state))
            runtime.captured_date = tracker.current_date
            runtime.captured_pi = tracker.pi
            result = await tracker.tools["sample"][0].ainvoke({"value": "token unchanged"})
            assert result.model_dump() == {"value": "token unchanged"}
            if runtime.run_failure is not None:
                raise runtime.run_failure
            return await super().run(state)

    runtime.DynamicAgentGraph = BoundGraph
    runtime.AgentLoop = BoundLoop
    runtime.default_state = lambda **kwargs: SimpleNamespace(input=kwargs["goal"])
    return runtime


class NativeClient:
    def __init__(self):
        self.closed = 0
        self.calls = []
        self.public_context = {
            "instruction": "synthetic task only", "current_datetime": "2023-05-17T08:15:00+05:45",
            "supervisor": {"first_name": "Synthetic", "email": "synthetic@example.invalid"},
            "app_descriptions": {"sample": "Synthetic app"},
            "api_docs": {"sample": {"echo": {
                "app_name": "sample", "api_name": "echo", "path": "/echo", "method": "POST",
                "description": "Return input", "response_schemas": {},
            }}},
            "function_calling": [{"type": "function", "function": {"name": "sample__echo", "parameters": {
                "type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"],
            }}}],
        }
        self.cleanup = {"child_reaped": True, "exit_code": 0, "request_outcome_unresolved": False,
                        "termination_signals": []}

    async def __call__(self, app, api, arguments):
        self.calls.append((app, api, deepcopy(arguments)))
        return deepcopy(arguments)

    async def close(self):
        self.closed += 1
        return deepcopy(self.cleanup)


def run_bound(runtime, client, root, observation):
    from population_dynamics.appworld_benchmark_cuga_runtime import run_bound_cuga_graph
    return asyncio.run(run_bound_cuga_graph(runtime, client, work_root=root, task_id="fixture_1",
                       model_name="fixture-model", base_url="http://127.0.0.1:1234/v1", seed=41,
                       observation=observation, usage_callback=object()))


def test_bound_graph_shares_native_date_and_releases_task_bindings(tmp_path):
    runtime, client, observation = prepared_runtime(), NativeClient(), {}
    result = run_bound(runtime, client, tmp_path / "attempt", observation)
    assert result is observation and result["status"] == "graph_ended"
    assert result["formal_eligible"] is False
    assert runtime.builds == runtime.runs == client.closed == 1
    assert client.calls == [("sample", "echo", {"value": "token unchanged"})]
    assert runtime.captured_date == runtime.captured_state["current_datetime"] == client.public_context["current_datetime"]
    assert runtime.captured_pi == runtime.captured_state["pi"]
    assert runtime.tracker.tools == {} and runtime.tracker.experiment_folder is None
    assert runtime.tracker.current_date is None and runtime.tracker.pi is None
    assert "get_tools_by_server" not in vars(runtime.tracker)
    assert (tmp_path / "attempt/trajectories/current").is_dir()
    assert result["native_cleanup_settled"] is True
    assert result["loop"]["model_usage"]["transport_coverage_qualified"] is False


@pytest.mark.parametrize("stage", ["constants", "settings", "build", "model_identity"])
def test_bound_graph_failure_always_reaps_client_and_clears_current_tools(tmp_path, stage):
    runtime, client, observation = prepared_runtime(), NativeClient(), {}
    if stage == "constants":
        runtime.save_reuse_constants = (True, False)
    elif stage == "settings":
        runtime.settings.values["advanced_features.registry"] = True
    elif stage == "build":
        runtime.failure = RuntimeError("graph construction failed")
    else:
        runtime.manager._models["planner"].max_retries = 2
    with pytest.raises((ValueError, RuntimeError)):
        run_bound(runtime, client, tmp_path / "attempt", observation)
    assert runtime.runs == 0 and client.closed == 1
    assert runtime.tracker.tools == {}
    assert "get_tools_by_server" not in vars(runtime.tracker)
    assert observation["status"] == "errored"


def test_bound_graph_does_not_replace_existing_evidence(tmp_path):
    runtime, client = prepared_runtime(), NativeClient()
    observation = {"status": "preserved"}
    with pytest.raises(ValueError, match="fresh"):
        run_bound(runtime, client, tmp_path / "attempt", observation)
    assert observation == {"status": "preserved"} and client.closed == 1
    assert runtime.builds == 0


def test_native_unresolved_cleanup_prevents_complete_graph_status(tmp_path):
    runtime, client = prepared_runtime(), NativeClient()
    client.cleanup["request_outcome_unresolved"] = True
    result = run_bound(runtime, client, tmp_path / "attempt", {})
    assert result["status"] == "native_execution_unresolved"
    assert result["loop"]["status"] == "graph_ended"


@pytest.mark.parametrize("error", [RuntimeError("failed after a tool action"), asyncio.CancelledError()])
def test_bound_graph_runtime_error_keeps_one_action_and_owns_cleanup(tmp_path, error):
    runtime, client, observation = prepared_runtime(), NativeClient(), {}
    runtime.run_failure = error
    with pytest.raises(type(error)) as raised:
        run_bound(runtime, client, tmp_path / "attempt", observation)
    assert raised.value is error
    assert len(client.calls) == runtime.runs == client.closed == 1
    assert observation["status"] == observation["loop"]["status"] == "errored"
    assert observation["native_tool_calls"][0]["status"] == "returned"
    assert runtime.tracker.tools == {} and runtime.tracker.experiment_folder is None


def test_bound_graph_preserves_existing_work_directory(tmp_path):
    runtime, client, observation = prepared_runtime(), NativeClient(), {}
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "retained.txt"
    marker.write_text("original evidence")
    with pytest.raises(FileExistsError):
        run_bound(runtime, client, root, observation)
    assert marker.read_text() == "original evidence"
    assert runtime.builds == 0 and client.closed == 1

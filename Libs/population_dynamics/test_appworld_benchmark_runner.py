"""Mock-native development boundary tests; no AppWorld benchmark evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from population_dynamics import appworld_benchmark_runner as runner
from population_dynamics.appworld_benchmark_scoring import _validate_inputs as validate_scorer_inputs
from population_dynamics.appworld_benchmark_transport import LocalChatTransport
from population_dynamics.test_appworld_benchmark_transport import context_exhausted, server


RATES = {"input_cache_miss": 0.0, "input_cache_hit": 0.0, "input_cache_write": 0.0, "output": 0.0}


def model_config():
    return {"name": "synthetic-model", "cost_per_token": dict(RATES)}


def agent_config(method="official_react_code"):
    config = {"model_config": model_config(), "max_steps": 2}
    if method == "official_full_code":
        config.update(code_prompt_file_path="source-code-prompt", retrial_prompt_file_path="source-retrial-prompt", demo_task_ids=["traincase_2"])
    else:
        config["prompt_file_path"] = "source-prompt"
    if method in {"official_full_code", "official_function_calling"}:
        config["api_predictor_config"] = {
            "model_config": model_config(), "prompt_file_path": "source-api-prompt",
            "demo_task_ids": ["traincase_3"], "mode": "predicted",
        }
    return config


class FakeUsage:
    def __init__(self, raw):
        raw = raw if isinstance(raw, dict) else {}
        prompt = raw.get("prompt_tokens", 0)
        completion = raw.get("completion_tokens", 0)
        if raw.get("total_tokens", 0) != prompt + completion:
            prompt = completion = 0  # Simulate native corrupted-usage fallback.
        self.value = {
            "tokens": {"input_cache_miss": prompt, "input_cache_hit": 0, "input_cache_write": 0, "output": completion},
            "cost": dict(RATES), "cost_per_token": dict(RATES),
        }

    def dict(self):
        return deepcopy(self.value)


class FakeModel:
    def __init__(self, config):
        self.model = config["name"]
        self.config = config
        self.lm_call = self.uninjected

    def uninjected(self, **kwargs):
        raise AssertionError("every model consumer must receive the supplied transport")

    def generate(self, messages, tools=None):
        response = self.lm_call(model=self.model, messages=messages, tools=tools, seed=self.config["seed"])
        output = deepcopy(response["choices"][0]["message"])
        output["standardized_usage"] = FakeUsage(response.get("usage"))
        if response.get("error"):
            output["error"] = response["error"]
        return output


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / "native-appworld"
    data_root = source_root / "data"
    (data_root / "datasets").mkdir(parents=True)
    (data_root / "tasks").mkdir()
    manifest = {"schema": "appworld_split_membership_v1", "appworld_version": "fixture-only", "data_root": str(source_root), "splits": {}}
    for split, prefix in (("train", "traincase"), ("dev", "devcase"), ("test_normal", "normalcase"), ("test_challenge", "challengecase")):
        task_ids = [f"{prefix}_{variant}" for variant in (1, 2, 3)]
        payload = (task_ids[0] + ":tagged\n" + "\n".join(task_ids[1:]) + "\n").encode()
        membership_file = data_root / "datasets" / f"{split}.txt"
        membership_file.write_bytes(payload)
        manifest["splits"][split] = {
            "membership_file": str(membership_file), "membership_sha256": hashlib.sha256(payload).hexdigest(),
            "task_ids": task_ids, "task_count": 3, "scenario_count": 1,
        }
    config = {
        "schema": "appworld_benchmark_v1", "seeds": [41, 42, 43], "formal": {"released": False},
        "splits": {"development": ["train", "dev"]},
        "methods": {method: {"status": "synthetic-test"} for method in runner._AGENT_CLASSES},
    }
    load_calls, solve_calls, agents, worlds = [], [], [], []
    loaded_ids = {split: deepcopy(entry["task_ids"]) for split, entry in manifest["splits"].items()}

    class FakePathStore:
        def __init__(self):
            self.root = str(tmp_path / "original-root")

        def update_root(self, root):
            self.root = root
            os.environ["APPWORLD_ROOT"] = root

        def reload(self):
            self.root = os.environ.get("APPWORLD_ROOT", str(tmp_path / "original-root"))

    class FakeAppWorld:
        init_defaults = SimpleNamespace(load_ground_truth=True, include_direct_functions=False)

        @staticmethod
        def evaluate(*args, **kwargs):
            raise AssertionError("execution process must not access an evaluator")

    def load_task_ids(**kwargs):
        assert set(kwargs) == {"dataset_name"}  # No difficulty or ground-truth filter.
        assert Path(os.environ["APPWORLD_ROOT"], "data").resolve() == data_root
        load_calls.append(kwargs)
        return deepcopy(loaded_ids[kwargs["dataset_name"]])

    native = SimpleNamespace(AppWorld=FakeAppWorld, path_store=FakePathStore(), load_task_ids=load_task_ids, version="fixture-only")

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.language_model = FakeModel(kwargs["model_config"])
            if "api_predictor_config" in kwargs:
                self.api_predictor = SimpleNamespace(language_model=FakeModel(kwargs["api_predictor_config"]["model_config"]))
            agents.append(self)
            FakeAppWorld.init_defaults.include_direct_functions = True

        def solve_tasks(self, task_ids, **kwargs):
            solve_calls.append((task_ids, kwargs))
            assert self.kwargs["appworld_config"]["load_ground_truth"] is False
            FakeAppWorld.init_defaults.load_ground_truth = False
            world = {"task_id": task_ids[0], "observations": []}
            worlds.append(world)
            if hasattr(self, "api_predictor"):
                self.api_predictor.language_model.generate(messages=[{"role": "user", "content": "mock API prediction"}])
            for step in range(2):
                output = self.language_model.generate(messages=[{"role": "user", "content": f"mock step {step}"}])
                world["observations"].append(output["content"])
                if output.get("error"):
                    break

    monkeypatch.setattr(runner, "_check_execution_location", lambda root: None)
    monkeypatch.setattr(runner, "_load_native", lambda: native)
    monkeypatch.setattr(runner, "_load_agent_class", lambda method: FakeAgent)
    monkeypatch.setenv("APPWORLD_ROOT", native.path_store.root)
    monkeypatch.setenv("APPWORLD_CACHE", str(tmp_path / "original-cache"))
    transport_calls = []

    def transport(**kwargs):
        transport_calls.append(deepcopy(kwargs))
        return {"choices": [{"message": {"role": "assistant", "content": "mock generated code"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}

    return SimpleNamespace(
        config=config, manifest=manifest, native=native, root=tmp_path / "attempt",
        load_calls=load_calls, solve_calls=solve_calls, agents=agents, worlds=worlds,
        loaded_ids=loaded_ids, transport=transport, transport_calls=transport_calls,
    )


def execute(fixture, method="official_react_code", **overrides):
    args = {
        "split": "dev", "task_id": "devcase_1", "method": method, "seed": 41,
        "agent_config": agent_config(method), "run_root": fixture.root, "lm_call": fixture.transport,
    }
    args.update(overrides)
    return runner.run_development_attempt(fixture.config, fixture.manifest, **args)


def test_delegates_one_persistent_native_solve_without_scoring(native_fixture):
    fixture = native_fixture
    original_defaults = deepcopy(vars(fixture.native.AppWorld.init_defaults))
    original_environment = {key: os.environ.get(key) for key in ("APPWORLD_ROOT", "APPWORLD_CACHE")}
    receipt = execute(fixture)
    assert fixture.solve_calls == [(["devcase_1"], {"experiment_name": "development", "num_processes": 1, "process_index": 0})]
    assert len(fixture.worlds) == 1 and len(fixture.worlds[0]["observations"]) == 2
    assert receipt["status"] == "completed" and receipt["official_solve_returned"]
    assert receipt["terminal"] and receipt["scope"] == "development_only"
    assert receipt["native_tgc"] is None and receipt["scoring_status"] == "not_run"
    assert not receipt["formal_eligible"]
    assert receipt["lm_call_count"] == 2 and receipt["usage_complete"]
    assert all(call["official_usage"]["tokens"]["output"] == 2 for call in receipt["model_calls"])
    assert receipt["monetary_cost_status"] == "caller_supplied_rates_not_verified"
    assert (fixture.root / "data").resolve() == Path(fixture.manifest["data_root"]) / "data"
    assert len(fixture.load_calls) == 4
    assert json.loads((fixture.root / "execution_receipt.json").read_text()) == receipt
    assert vars(fixture.native.AppWorld.init_defaults) == original_defaults
    assert {key: os.environ.get(key) for key in original_environment} == original_environment
    kwargs = fixture.agents[0].kwargs
    assert kwargs["prompt_file_path"] == "source-prompt"
    assert kwargs["model_config"]["use_cache"] is False and kwargs["model_config"]["max_retries"] == 1
    assert kwargs["appworld_config"]["load_ground_truth"] is False
    assert all(call["seed"] == 41 for call in fixture.transport_calls)


def test_full_code_injects_both_native_consumers_and_preserves_demos(native_fixture):
    receipt = execute(native_fixture, method="official_full_code")
    kwargs = native_fixture.agents[0].kwargs
    assert kwargs["demo_task_ids"] == ["traincase_2"]
    assert kwargs["api_predictor_config"]["demo_task_ids"] == ["traincase_3"]
    assert kwargs["code_prompt_file_path"] == "source-code-prompt"
    assert kwargs["api_predictor_config"]["model_config"]["use_cache"] is False
    assert kwargs["api_predictor_config"]["model_config"]["max_retries"] == 1
    assert [call["consumer"] for call in receipt["model_calls"]] == ["api_predictor", "primary", "primary"]
    assert receipt["lm_call_count"] == 3 and receipt["usage_complete"]


@pytest.mark.parametrize("released", [False, True])
def test_formal_execution_stays_closed_before_native_import(native_fixture, released, monkeypatch):
    native_fixture.config["formal"]["released"] = released
    monkeypatch.setattr(runner, "_load_native", lambda: pytest.fail("formal rejection must precede native imports"))
    with pytest.raises(ValueError, match="FORMAL_RELEASE_CLOSED|DEVELOPMENT_ONLY_BOUNDARY"):
        execute(native_fixture, split="test_challenge", task_id="challengecase_1")
    assert not native_fixture.root.exists()


@pytest.mark.parametrize("seed", [40, 44, True, 41.0])
def test_rejects_unapproved_seed(native_fixture, seed):
    with pytest.raises(ValueError, match="attempt seed"):
        execute(native_fixture, seed=seed)
    assert not native_fixture.root.exists()


@pytest.mark.parametrize("location", ["world", "predictor"])
def test_prohibits_target_ground_truth(native_fixture, location):
    kwargs = agent_config("official_full_code")
    if location == "world":
        kwargs["appworld_config"] = {"load_ground_truth": True}
    else:
        kwargs["api_predictor_config"]["mode"] = "ground_truth"
    with pytest.raises(ValueError, match="ground_truth"):
        execute(native_fixture, method="official_full_code", agent_config=kwargs)
    assert not native_fixture.root.exists()


@pytest.mark.parametrize("location", ["full_code", "predictor"])
@pytest.mark.parametrize("demo_id", ["devcase_2", "challengecase_1", "missing_1"])
def test_demo_membership_is_train_only(native_fixture, location, demo_id):
    kwargs = agent_config("official_full_code")
    destination = kwargs if location == "full_code" else kwargs["api_predictor_config"]
    destination["demo_task_ids"] = [demo_id]
    with pytest.raises(ValueError, match="resolve to train"):
        execute(native_fixture, method="official_full_code", agent_config=kwargs)
    assert not native_fixture.agents


def test_train_target_cannot_be_its_own_ground_truth_demo(native_fixture):
    kwargs = agent_config("official_full_code")
    kwargs["demo_task_ids"] = ["traincase_1"]
    with pytest.raises(ValueError, match="exclude the target"):
        execute(native_fixture, method="official_full_code", split="train", task_id="traincase_1", agent_config=kwargs)


def test_missing_target_membership_rejected(native_fixture):
    with pytest.raises(ValueError, match="absent"):
        execute(native_fixture, task_id="missing_1")
    assert not native_fixture.load_calls


@pytest.mark.parametrize("defect", ["file_bytes", "native_membership", "version"])
def test_native_admission_failure_preserves_terminal_receipt(native_fixture, defect):
    fixture = native_fixture
    if defect == "file_bytes":
        Path(fixture.manifest["splits"]["train"]["membership_file"]).write_text("changed native bytes")
    elif defect == "native_membership":
        fixture.loaded_ids["dev"] = list(reversed(fixture.loaded_ids["dev"]))
    else:
        fixture.native.version = "different-version"
    with pytest.raises(runner.DevelopmentExecutionError) as error:
        execute(fixture)
    receipt = json.loads(error.value.receipt_path.read_text())
    assert receipt["terminal"] and receipt["status"] == "infrastructure_failure"
    assert receipt["stage"] == "source_admission" and not fixture.solve_calls
    assert receipt["native_tgc"] is None


def test_function_calling_rejects_plain_text_capability(native_fixture):
    with pytest.raises(ValueError, match="validated tool-call transport"):
        execute(native_fixture, method="official_function_calling")
    with pytest.raises(ValueError, match="validation reference"):
        execute(native_fixture, method="official_function_calling", transport_capabilities=runner.TransportCapabilities(tool_calls_validated=True))


def test_function_calling_requires_and_records_explicit_validation(native_fixture):
    receipt = execute(
        native_fixture, method="official_function_calling",
        transport_capabilities=runner.TransportCapabilities(tool_calls_validated=True, validation_reference="synthetic-mock-capability-record"),
    )
    assert receipt["tool_transport_validation_reference"] == "synthetic-mock-capability-record"
    assert receipt["lm_call_count"] == 3 and receipt["native_tgc"] is None


def test_transport_exception_retains_failure_and_unknown_usage(native_fixture):
    def transport(**kwargs):
        raise RuntimeError("synthetic transport failure")
    with pytest.raises(runner.DevelopmentExecutionError) as error:
        execute(native_fixture, lm_call=transport)
    receipt = json.loads(error.value.receipt_path.read_text())
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"]
    assert receipt["lm_call_count"] == 1 and not receipt["usage_complete"]
    assert receipt["model_calls"][0]["transport_status"] == "failed"
    assert receipt["native_tgc"] is None and not receipt["official_solve_returned"]


def test_native_agent_failure_is_retained_without_scoring(native_fixture):
    def transport(**kwargs):
        response = native_fixture.transport(**kwargs)
        response["error"] = "synthetic model failure"
        return response
    receipt = execute(native_fixture, lm_call=transport)
    assert receipt["status"] == "agent_failure" and receipt["official_solve_returned"]
    assert receipt["agent_failures"] == ["synthetic model failure"]
    assert receipt["lm_call_count"] == 1 and receipt["usage_complete"]
    assert receipt["native_tgc"] is None


@pytest.mark.parametrize("raw_usage", [None, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 100}])
def test_missing_or_corrupt_usage_cannot_be_measured_zero(native_fixture, raw_usage):
    def transport(**kwargs):
        response = native_fixture.transport(**kwargs)
        response["usage"] = raw_usage
        return response
    receipt = execute(native_fixture, lm_call=transport)
    assert receipt["status"] == "completed" and not receipt["usage_complete"]
    assert all(call["usage_status"] == "unresolved" for call in receipt["model_calls"])
    assert receipt["model_calls"][0]["raw_usage"] == raw_usage


def test_existing_or_unresolved_run_is_never_reused(native_fixture):
    native_fixture.root.mkdir()
    marker = native_fixture.root / "execution_receipt.json"
    marker.write_text('{"status":"running"}')
    with pytest.raises(FileExistsError, match="unresolved"):
        execute(native_fixture)
    assert marker.read_text() == '{"status":"running"}' and not native_fixture.load_calls


def test_native_cost_rates_must_be_explicit(native_fixture):
    kwargs = agent_config()
    kwargs["model_config"].pop("cost_per_token")
    with pytest.raises(ValueError, match="cost_per_token"):
        execute(native_fixture, agent_config=kwargs)
    assert not native_fixture.root.exists()


def test_inconsistent_raw_totals_remain_unresolved_with_inflated_native_output():
    raw_usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 100}
    standardized_usage = {
        "tokens": {"input_cache_miss": 5, "input_cache_hit": 0, "output": 95},
    }
    assert not runner._usage_is_resolved(raw_usage, standardized_usage)


def test_initial_receipt_preserves_invocation_and_resolved_settings_before_model_calls(native_fixture, monkeypatch):
    fixture = native_fixture
    metadata = {"source": {"revision": "synthetic-revision"}, "model": {"name": "synthetic-model"}}
    expected_metadata = deepcopy(metadata)
    agent_class = runner._load_agent_class("official_react_code")
    original_solve = agent_class.solve_tasks
    inspected = []

    def inspect_initial_receipt(self, task_ids, **kwargs):
        initial = json.loads((fixture.root / "execution_receipt.json").read_text())
        assert initial["status"] == "running" and initial["terminal"] is False
        assert initial["model_calls"] == [] and fixture.transport_calls == []
        assert initial["invocation_metadata"] == expected_metadata
        resolved = initial["resolved_agent_config"]
        assert resolved["model_config"]["seed"] == 41
        assert resolved["model_config"]["use_cache"] is False
        assert resolved["model_config"]["max_retries"] == 1
        assert resolved["appworld_config"]["load_ground_truth"] is False
        inspected.append(initial)
        metadata["source"]["revision"] = "changed-after-persistence"
        self.kwargs["model_config"]["max_retries"] = 99
        return original_solve(self, task_ids, **kwargs)

    monkeypatch.setattr(agent_class, "solve_tasks", inspect_initial_receipt)
    receipt = execute(fixture, invocation_metadata=metadata)
    assert len(inspected) == 1
    assert receipt["invocation_metadata"] == expected_metadata
    assert receipt["resolved_agent_config"]["model_config"]["max_retries"] == 1


@pytest.mark.parametrize(
    "reasoning_fields",
    [
        {}, {"reasoning_content": None}, {"reasoning_content": " keep whitespace "},
        {"reasoning_content": 7}, {"reasoning_content": False},
        {"reasoning_content": ["invalid"]}, {"reasoning_content": {"invalid": True}},
    ],
    ids=["absent", "null", "text", "integer", "boolean", "list", "mapping"],
)
def test_observer_preserves_raw_response_and_omits_only_null_reasoning(reasoning_fields):
    response = {
        "id": "synthetic-response",
        "choices": [{"message": {
            "role": "assistant", "content": "print(1)",
            "tool_calls": [{"id": "synthetic-call", "function": {"name": "example", "arguments": "{}"}}],
            "error": "synthetic returned error",
            "unknown_field": {"nested": ["preserved"]},
            **deepcopy(reasoning_fields),
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "unknown_response_field": {"nested": ["preserved"]},
    }
    expected_raw = deepcopy(response)
    calls, failures, native_outputs = [], [], []
    model = SimpleNamespace(model="synthetic-model")

    def native_generate():
        returned = model.lm_call(model=model.model)
        # The pinned native method adds usage to the original message object.
        output = returned["choices"][0]["message"]
        output["standardized_usage"] = FakeUsage(returned["usage"])
        native_outputs.append(output)
        return output

    model.generate = native_generate
    runner._observe_model(model, "primary", lambda **kwargs: response, calls, failures)
    output = model.generate()
    native_output = native_outputs[0]
    omitted_null = "reasoning_content" in reasoning_fields and reasoning_fields["reasoning_content"] is None
    assert len(calls) == 1
    assert calls[0]["raw_response"] == expected_raw
    assert calls[0]["raw_usage"] == expected_raw["usage"]
    assert calls[0]["usage_status"] == "resolved"
    assert failures == ["synthetic returned error"]
    if omitted_null:
        assert output is not native_output
        assert "reasoning_content" not in output
        assert native_output["reasoning_content"] is None
        assert calls[0]["response_normalizations"] == [{
            "stage": "native_generate_output",
            "field": "reasoning_content", "operation": "omit_null",
        }]
    else:
        assert output is native_output
        assert calls[0]["response_normalizations"] == []
    expected_keys = set(native_output) - ({"reasoning_content"} if omitted_null else set())
    assert set(output) == expected_keys
    assert all(output[key] is native_output[key] for key in expected_keys)
    response["usage"]["prompt_tokens"] = 99
    response["choices"][0]["message"]["unknown_field"]["nested"].append("native mutation")
    response["unknown_response_field"]["nested"].append("native mutation")
    assert calls[0]["raw_response"] == expected_raw
    assert calls[0]["raw_usage"] == expected_raw["usage"]


def test_native_generate_failure_retains_raw_response_before_processing(native_fixture):
    response = {
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "unknown_response_field": {"retained": True},
    }
    with pytest.raises(runner.DevelopmentExecutionError) as error:
        execute(native_fixture, lm_call=lambda **kwargs: response)
    receipt = json.loads(error.value.receipt_path.read_text())
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"]
    assert receipt["stage"] == "native_solve_tasks"
    assert receipt["lm_call_count"] == 1 and not receipt["usage_complete"]
    call = receipt["model_calls"][0]
    assert call["transport_status"] == "returned"
    assert call["raw_response"] == response and call["raw_usage"] == response["usage"]
    assert call["official_usage"] is None and call["response_normalizations"] == []
    assert receipt["error_traceback"].startswith("Traceback (most recent call last):")
    assert "in generate" in receipt["error_traceback"]
    assert "IndexError: list index out of range" in receipt["error_traceback"]
    assert receipt["native_tgc"] is None and receipt["scoring_status"] == "not_run"


def test_native_parser_failure_retains_raw_response_and_full_traceback(native_fixture, monkeypatch):
    response = native_fixture.transport()
    response["choices"][0]["message"]["reasoning_content"] = 7
    expected_raw = deepcopy(response)
    agent_class = runner._load_agent_class("official_react_code")

    def parse_native_response(self, task_ids, **kwargs):
        output = self.language_model.generate(messages=[{"role": "user", "content": "mock parser"}])
        output.get("reasoning_content", "").strip()

    monkeypatch.setattr(agent_class, "solve_tasks", parse_native_response)
    with pytest.raises(runner.DevelopmentExecutionError) as error:
        execute(native_fixture, lm_call=lambda **kwargs: response)
    receipt = json.loads(error.value.receipt_path.read_text())
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"]
    assert receipt["stage"] == "native_solve_tasks" and not receipt["official_solve_returned"]
    assert receipt["lm_call_count"] == 1 and receipt["usage_complete"]
    call = receipt["model_calls"][0]
    assert call["raw_response"] == expected_raw
    assert call["response_normalizations"] == []
    assert receipt["error_type"] == "AttributeError"
    assert receipt["error_traceback"].startswith("Traceback (most recent call last):")
    assert "in run_development_attempt" in receipt["error_traceback"]
    assert "in parse_native_response" in receipt["error_traceback"]
    assert "AttributeError: 'int' object has no attribute 'strip'" in receipt["error_traceback"]
    assert receipt["native_tgc"] is None and receipt["scoring_status"] == "not_run"


def test_official_no_room_terminal_retains_prior_observation_and_unknown_native_usage(native_fixture):
    f = native_fixture
    first = {"choices": [{"message": {"role": "assistant", "content": "retained native observation"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
    with server(responses=[(200, first), (400, context_exhausted())]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        receipt = execute(f, lm_call=client)
    assert receipt["status"] == "agent_failure" and receipt["stop_reason"] == "context_budget_exhausted"
    assert receipt["terminal"] is True and receipt["official_solve_returned"] is False
    assert receipt["usage_complete"] is False and receipt["lm_call_count"] == len(seen) == 2
    assert f.worlds[0]["observations"] == ["retained native observation"]
    first_call, rejected = receipt["model_calls"]
    assert first_call["raw_response"] == first and first_call["usage_status"] == "resolved"
    assert rejected["transport_status"] == "pre_generation_rejected"
    assert rejected["raw_response"] is rejected["raw_usage"] is rejected["official_usage"] is None
    assert rejected["usage_status"] == "unresolved"
    assert rejected["generation_started"] is False and rejected["generated_completion_tokens"] == 0
    assert rejected["transport_request_id"] == client.records[-1]["request_id"]
    validate_scorer_inputs(receipt, f.manifest)

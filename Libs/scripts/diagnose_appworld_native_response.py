"""Inspect pinned native response normalization using synthetic completions only.

Execute on Merlin CPU after restoring the admitted native runtime. This does not
load AppWorld tasks, call a model endpoint, or produce benchmark evidence.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from population_dynamics.appworld_benchmark_runner import _load_agent_class, _observe_model


def main() -> int:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    output.mkdir(exist_ok=False)
    report = {"schema": "appworld_native_response_diagnostic_v1", "synthetic": True,
              "native_tasks_executed": 0, "model_generations": 0, "formal_eligible": False,
              "cases": [], "status": "RUNNING"}
    try:
        agent_class = _load_agent_class("official_react_code")
        agent_module = inspect.getmodule(agent_class)
        report["agent_class"] = str(agent_class)
        report["agent_signature"] = str(inspect.signature(agent_class))
        (output / "react_code_agent.py.txt").write_text(inspect.getsource(agent_module))
        model_module = importlib.import_module("appworld_agents.code.simplified.language_model")
        (output / "language_model.py.txt").write_text(inspect.getsource(model_module))
        # Constructor-only fixture values. The callable below supplies every response.
        os.environ["OPENAI_API_KEY"] = "synthetic-fixture-no-credential"
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
        agent = agent_class(**inputs["resolved_agent_config"])
        model = agent.language_model
        model_module = inspect.getmodule(type(model))
        (output / "language_model.py.txt").write_text(inspect.getsource(model_module))
        report["model_class"] = str(type(model))
        report["generate_signature"] = str(inspect.signature(model.generate))
        fixture = {"id": "synthetic-diagnostic", "object": "chat.completion", "created": 0,
                   "model": inputs["resolved_agent_config"]["model_config"]["name"],
                   "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None,
                                "message": {"role": "assistant", "content": "```python\nprint(1)\n```"}}],
                   "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}
        for observed, reasoning_variant in [(observed, variant) for observed in (False, True) for variant in ("absent", "null", "text")]:
            case_fixture = deepcopy(fixture)
            if reasoning_variant != "absent":
                case_fixture["choices"][0]["message"]["reasoning_content"] = None if reasoning_variant == "null" else "synthetic reasoning"
            case = {"observer_enabled": observed, "reasoning_variant": reasoning_variant,
                    "transport_calls": [], "fixture": case_fixture}
            report["cases"].append(case)
            try:
                fresh_agent = agent_class(**inputs["resolved_agent_config"])
                fresh_model = fresh_agent.language_model
                def synthetic_call(**kwargs):
                    case["transport_calls"].append(deepcopy(kwargs))
                    return deepcopy(case_fixture)
                calls, failures = [], []
                if observed:
                    _observe_model(fresh_model, "primary", synthetic_call, calls, failures)
                else:
                    fresh_model.lm_call = synthetic_call
                result = fresh_model.generate([{"role": "user", "content": "Synthetic normalization diagnostic."}])
                case.update(status="RETURNED", output=result, observer_records=calls, agent_failures=failures)
                # Invoke the real ReAct parser without a Task or AppWorld instance.
                fresh_agent.messages = [{"role": "user", "content": "Synthetic normalization diagnostic."}]
                fresh_agent.step_number = 1
                fresh_agent.logger = SimpleNamespace(show_message=lambda **kwargs: None)
                try:
                    execution_inputs, usage, status = fresh_agent.next_execution_inputs_usage_and_status([])
                    case["native_step"] = {"status": "RETURNED", "code": [item.content for item in execution_inputs],
                                           "usage": usage, "agent_status": str(status)}
                except Exception as error:
                    case["native_step"] = {"status": "FAILED", "error_type": type(error).__name__,
                                           "error": str(error), "traceback": traceback.format_exc()}

            except Exception as error:
                case.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if inputs.get("verify_null_reasoning_fix"):
            checks = []
            for case in report["cases"]:
                step = case.get("native_step", {})
                original_null_failure = not case["observer_enabled"] and case["reasoning_variant"] == "null"
                passed = (step.get("status") == "FAILED" and step.get("error_type") == "AttributeError"
                          and "strip" in step.get("error", "")) if original_null_failure else (
                          step.get("status") == "RETURNED" and step.get("code") == ["print(1)"])
                if case["observer_enabled"]:
                    records = case.get("observer_records", [])
                    passed = passed and len(records) == 2 and all(
                        row.get("raw_response") == case["fixture"] and row.get("usage_status") == "resolved"
                        for row in records)
                    if case["reasoning_variant"] == "null":
                        passed = passed and "reasoning_content" not in case.get("output", {}) and all(
                            row.get("response_normalizations") == [{"stage": "native_generate_output", "field": "reasoning_content", "operation": "omit_null"}]
                            for row in records)
                checks.append({"observer_enabled": case["observer_enabled"],
                               "reasoning_variant": case["reasoning_variant"], "passed": bool(passed)})
            report["verification"] = checks
            if len(checks) != 6 or not all(row["passed"] for row in checks):
                raise ValueError("Native null-reasoning compatibility verification failed")
        report["status"] = "DIAGNOSTIC_COMPLETED"
    except Exception as error:
        report.update(status="DIAGNOSTIC_FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, default=lambda value: value.dict() if hasattr(value, "dict") else str(value)) + "\n")
    print(json.dumps({"status": report["status"], "case_statuses": [row["status"] for row in report["cases"]]}), flush=True)
    return 0 if report["status"] == "DIAGNOSTIC_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

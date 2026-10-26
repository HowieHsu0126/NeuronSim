"""Synthetic HTTP-interface tests; no model or AppWorld task is executed."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from population_dynamics.appworld_benchmark_model_probe import probe_model_transport
from population_dynamics.appworld_benchmark_transport import ChatTransportError


@pytest.fixture
def config():
    return {
        "seeds": [41, 42, 43],
        "model": {
            "repo_id": "synthetic/model", "revision": "synthetic-revision",
            "tool_call_parser": "qwen3_coder", "serving_backend": "vllm", "serving_version": "fixture-version",
        },
        "development_execution": {"model": {"max_completion_tokens": 512, "temperature": 0.7, "top_p": 0.8}},
    }


def phase_of(payload):
    if payload["messages"][-1]["role"] == "tool":
        return "tool_result_continuation"
    return "automatic_tool_call" if "tools" in payload else "plain_chat"


class FakeTransport:
    def __init__(self, mutate=None, failure=None):
        self.mutate = mutate
        self.failure = failure
        self.requests = []
        self.records = []

    def __call__(self, **payload):
        self.requests.append(deepcopy(payload))
        seed, phase = payload["seed"], phase_of(payload)
        request_id = f"synthetic-request-{len(self.requests)}"
        record = {"request_id": request_id, "attempts": 1, "status": "pending", "completion_known": False, "usage": None}
        self.records.append(record)
        if self.failure == (seed, phase):
            record.update(status="http_error", status_code=503)
            raise ChatTransportError("synthetic HTTP failure", request_id, 503)
        message = {"role": "assistant", "content": f"Arbitrary nonempty wording for {phase} seed {seed}."}
        finish_reason = "stop"
        if phase == "automatic_tool_call":
            message = {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call-{seed}", "type": "function", "function": {"name": "readiness__add", "arguments": '{"a":19,"b":23}'}}],
            }
            finish_reason = "tool_calls"
        response = {
            "object": "chat.completion", "model": payload["model"],
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if self.mutate is not None:
            self.mutate(seed, phase, response)
        record.update(status="completed", completion_known=True, usage=deepcopy(response.get("usage")))
        return response


def rows(result):
    return [probe for seed in result["per_seed"] for probe in seed["probes"]]


def test_complete_three_seed_sequence_uses_auto_tools_and_isolated_histories(config):
    transport = FakeTransport()
    result = probe_model_transport(config, transport=transport)
    assert result["schema"] == "appworld_model_transport_probe_v1" and result["status"] == "PASS"
    assert result["seeds"] == [41, 42, 43]
    assert result["model_identity"] == {"repo_id": "synthetic/model", "revision": "synthetic-revision"}
    assert result["tool_call_parser"] == "qwen3_coder" and result["serving_backend"] == "vllm"
    assert result["serving_version"] == "fixture-version"
    assert result["appworld_tasks_executed"] == 0 and result["native_task_qualification"] == "NOT_EVALUATED"
    assert result["request_count"] == result["expected_request_count"] == 9 and result["not_run_count"] == 0
    assert [(request["seed"], phase_of(request)) for request in transport.requests] == [
        (seed, phase) for seed in (41, 42, 43)
        for phase in ("plain_chat", "automatic_tool_call", "tool_result_continuation")
    ]
    for index, seed in enumerate((41, 42, 43)):
        plain, tool, continuation = transport.requests[index * 3:index * 3 + 3]
        assert plain["stream"] is False and plain["n"] == 1 and "tools" not in plain
        assert len(plain["messages"]) == len(tool["messages"]) == 1
        assert tool["tool_choice"] == continuation["tool_choice"] == "auto"
        assert tool["tools"][0]["function"]["name"] == "readiness__add"
        assistant = result["per_seed"][index]["probes"][1]["response"]["choices"][0]["message"]
        assert continuation["messages"][0] == tool["messages"][0]
        assert continuation["messages"][1] == assistant
        assert continuation["messages"][2] == {"role": "tool", "tool_call_id": f"call-{seed}", "name": "readiness__add", "content": "42"}
        assert continuation["messages"][2]["tool_call_id"] == assistant["tool_calls"][0]["id"]
    for request, row in zip(transport.requests, rows(result)):
        assert row["request"] == request and row["raw_usage"] == row["response"]["usage"]
        assert request["model"] == config["model"]["repo_id"]
        assert request["max_completion_tokens"] == 512 and request["temperature"] == 0.7
    assert result["transport_records"] == transport.records
    transport.records[0]["status"] = "changed-after-return"
    assert result["transport_records"][0]["status"] == "completed"


def test_reused_call_ids_across_independent_seed_conversations_are_valid(config):
    def mutate(seed, phase, response):
        if phase == "automatic_tool_call":
            response["choices"][0]["message"]["tool_calls"][0]["id"] = "same-id-in-independent-conversations"
    transport = FakeTransport(mutate=mutate)
    result = probe_model_transport(config, transport=transport)
    assert result["status"] == "PASS"
    for request in transport.requests:
        if phase_of(request) == "tool_result_continuation":
            assert request["messages"][-1]["tool_call_id"] == request["messages"][-2]["tool_calls"][0]["id"]


@pytest.mark.parametrize("defect", ["missing", "negative", "boolean", "float", "wrong_total"])
def test_malformed_raw_usage_fails_without_coercion_or_retries(config, defect):
    def mutate(seed, phase, response):
        if (seed, phase) != (41, "plain_chat"):
            return
        if defect == "missing":
            response.pop("usage")
        elif defect == "negative":
            response["usage"]["prompt_tokens"] = -1
        elif defect == "boolean":
            response["usage"]["completion_tokens"] = True
        elif defect == "float":
            response["usage"]["prompt_tokens"] = 10.0
        else:
            response["usage"]["total_tokens"] = 99
    transport = FakeTransport(mutate=mutate)
    result = probe_model_transport(config, transport=transport)
    assert result["status"] == "FAIL" and result["request_count"] == 9
    assert result["per_seed"][0]["probes"][0]["status"] == "FAIL"
    assert [seed["status"] for seed in result["per_seed"]] == ["FAIL", "PASS", "PASS"]
    assert all(record["attempts"] == 1 for record in result["transport_records"])


@pytest.mark.parametrize("defect", ["empty_id", "non_string_id", "wrong_name", "wrong_type", "missing_call", "multiple_calls", "wrong_arguments", "float_argument", "string_argument", "extra_argument", "duplicate_argument", "invalid_json"])
def test_invalid_tool_calls_skip_continuation_and_preserve_raw_output(config, defect):
    def mutate(seed, phase, response):
        if (seed, phase) != (42, "automatic_tool_call"):
            return
        message = response["choices"][0]["message"]
        call = message["tool_calls"][0]
        if defect == "empty_id":
            call["id"] = "   "
        elif defect == "non_string_id":
            call["id"] = 42
        elif defect == "wrong_name":
            call["function"]["name"] = "readiness__subtract"
        elif defect == "wrong_type":
            call["type"] = "other"
        elif defect == "missing_call":
            message["tool_calls"] = []
        elif defect == "multiple_calls":
            message["tool_calls"].append(deepcopy(call))
        else:
            arguments = {
                "wrong_arguments": '{"a":18,"b":24}', "float_argument": '{"a":19.0,"b":23}',
                "string_argument": '{"a":"19","b":23}', "extra_argument": '{"a":19,"b":23,"c":0}',
                "duplicate_argument": '{"a":0,"a":19,"b":23}', "invalid_json": "{invalid-json}",
            }
            call["function"]["arguments"] = arguments[defect]
    transport = FakeTransport(mutate=mutate)
    result = probe_model_transport(config, transport=transport)
    assert result["status"] == "FAIL" and result["request_count"] == 8 and result["not_run_count"] == 1
    tool, continuation = result["per_seed"][1]["probes"][1:]
    assert tool["status"] == "FAIL" and tool["response"] is not None
    assert "validated_tool_call" not in tool
    assert continuation["status"] == "NOT_RUN" and continuation["response"] is None and continuation["request"] is None
    assert not any(request["seed"] == 42 and phase_of(request) == "tool_result_continuation" for request in transport.requests)
    assert result["per_seed"][2]["status"] == "PASS"


@pytest.mark.parametrize("phase", ["plain_chat", "automatic_tool_call", "tool_result_continuation"])
def test_http_failure_is_retained_and_other_seeds_continue(config, phase):
    transport = FakeTransport(failure=(41, phase))
    result = probe_model_transport(config, transport=transport)
    assert result["status"] == "FAIL"
    failed = next(probe for probe in result["per_seed"][0]["probes"] if probe["phase"] == phase)
    assert failed["status"] == "FAIL" and failed["error_type"] == "ChatTransportError"
    assert failed["response"] is None and failed["raw_usage"] is None
    assert result["request_count"] == (8 if phase == "automatic_tool_call" else 9)
    assert [seed["status"] for seed in result["per_seed"][1:]] == ["PASS", "PASS"]
    assert len({(request["seed"], phase_of(request)) for request in transport.requests}) == len(transport.requests)
    assert any(record["status"] == "http_error" for record in result["transport_records"])


@pytest.mark.parametrize("defect", ["streaming", "bad_finish_reason", "wrong_role", "empty_text", "unexpected_tool"])
def test_plain_and_continuation_require_complete_assistant_text(config, defect):
    def mutate(seed, phase, response):
        if (seed, phase) != (43, "tool_result_continuation"):
            return
        choice = response["choices"][0]
        if defect == "streaming":
            response["object"] = "chat.completion.chunk"
            choice["delta"] = {"content": "partial"}
        elif defect == "bad_finish_reason":
            choice["finish_reason"] = "length"
        elif defect == "wrong_role":
            choice["message"]["role"] = "tool"
        elif defect == "empty_text":
            choice["message"]["content"] = " "
        else:
            choice["message"]["tool_calls"] = [{"id": "unexpected"}]
    result = probe_model_transport(config, transport=FakeTransport(mutate=mutate))
    assert result["status"] == "FAIL" and result["per_seed"][2]["probes"][2]["status"] == "FAIL"


def test_tool_request_finish_reason_must_match_a_parsed_call(config):
    def mutate(seed, phase, response):
        if (seed, phase) == (41, "automatic_tool_call"):
            response["choices"][0]["finish_reason"] = "stop"
    result = probe_model_transport(config, transport=FakeTransport(mutate=mutate))
    assert result["status"] == "FAIL" and result["per_seed"][0]["probes"][2]["status"] == "NOT_RUN"


def test_prior_transport_records_remain_identified_without_becoming_probe_requests(config):
    transport = FakeTransport()
    transport.records.append({"request_id": "earlier-request", "status": "unresolved"})
    result = probe_model_transport(config, transport=transport)
    assert result["status"] == "PASS" and result["transport_record_start_index"] == 1
    assert len(result["transport_records"]) == 10 and result["request_count"] == 9
    assert rows(result)[0]["transport_record_indices"] == [1]


@pytest.mark.parametrize("seeds", [[41, 42], [41, 42, 44], [41.0, 42, 43], [43, 42, 41]])
def test_seed_contract_is_checked_before_any_request(config, seeds):
    config["seeds"] = seeds
    transport = FakeTransport()
    with pytest.raises(ValueError, match="exactly"):
        probe_model_transport(config, transport=transport)
    assert transport.requests == []

"""Neutral local chat/tool-parser qualification; no AppWorld tasks or scoring.

The launcher supplies and records real model, source and runtime provenance.
This probe tests the configured HTTP interface and preserves raw evidence. A
PASS establishes no native-task success, benchmark score, or scientific claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from typing import Any

from .appworld_benchmark_protocol import EXPERIMENT_SEEDS
from .appworld_benchmark_transport import LocalChatTransport


_TOOL_NAME = "readiness__add"
_EXPECTED_ARGUMENTS = {"a": 19, "b": 23}
_TOOL = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": "Return the sum of two integer inputs.",
        "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False,
        },
    },
}


def _probe_settings(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if config.get("seeds") != list(EXPERIMENT_SEEDS) or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("probe seeds must be exactly [41, 42, 43]")
    model = config.get("model", {})
    for field in ("repo_id", "revision", "tool_call_parser", "serving_backend", "serving_version"):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(f"explicit model.{field} is required for probe provenance")
    generation = config.get("development_execution", {}).get("model", {})
    max_tokens = generation.get("max_completion_tokens")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("positive development max_completion_tokens is required")
    for field in ("temperature", "top_p"):
        value = generation.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"finite development {field} is required")
    if generation["temperature"] < 0 or not 0 < generation["top_p"] <= 1:
        raise ValueError("temperature must be nonnegative and top_p must be in (0, 1]")
    settings = {
        "model": model["repo_id"], "stream": False, "n": 1,
        "max_completion_tokens": max_tokens,
        "temperature": generation["temperature"], "top_p": generation["top_p"],
    }
    return settings, {field: model[field] for field in ("repo_id", "revision", "tool_call_parser", "serving_backend", "serving_version")}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate function argument key: {key}")
        result[key] = value
    return result


def _validate_response(response: Any, phase: str) -> dict[str, Any] | None:
    if not isinstance(response, Mapping):
        raise ValueError("response must be a non-stream chat-completion object")
    if response.get("object", "chat.completion") != "chat.completion" or response.get("error"):
        raise ValueError("response must be a completed non-stream chat response without an error")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("response must contain exactly one completion choice")
    choice = choices[0]
    if "delta" in choice or not isinstance(choice.get("message"), Mapping):
        raise ValueError("a complete message is required; streaming deltas are prohibited")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise ValueError("completion message role must be assistant")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ValueError("assistant content must be text or null")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("raw usage is missing")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise ValueError(f"raw usage {field} must be a nonnegative integer")
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        raise ValueError("raw usage total_tokens does not equal prompt_tokens + completion_tokens")
    tool_calls = message.get("tool_calls")
    if phase != "automatic_tool_call":
        if choice.get("finish_reason") != "stop":
            raise ValueError("plain and continuation responses must finish with stop")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("plain and continuation responses require nonempty assistant text")
        if tool_calls is not None and not (isinstance(tool_calls, list) and len(tool_calls) == 0):
            raise ValueError("plain and continuation responses must not contain tool calls")
        return None
    if choice.get("finish_reason") != "tool_calls":
        raise ValueError("automatic tool response must finish with tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("automatic tool response must contain exactly one parsed function call")
    call = tool_calls[0]
    if not isinstance(call, Mapping) or call.get("type") != "function":
        raise ValueError("parsed call type must be function")
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("tool-call id must be a nonempty string")
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != _TOOL_NAME:
        raise ValueError(f"parsed function name must be {_TOOL_NAME}")
    if not isinstance(function.get("arguments"), str):
        raise ValueError("parsed function arguments must be a JSON string")
    arguments = json.loads(function["arguments"], object_pairs_hook=_unique_object)
    if not isinstance(arguments, dict) or set(arguments) != set(_EXPECTED_ARGUMENTS):
        raise ValueError("function arguments must contain exactly a and b")
    if any(type(arguments[key]) is not int or arguments[key] != value for key, value in _EXPECTED_ARGUMENTS.items()):
        raise ValueError("function arguments must be the integers a=19 and b=23")
    return {"id": call_id, "name": function["name"], "arguments": arguments}


def _request(
    transport: LocalChatTransport, seed: int, phase: str, payload: dict[str, Any],
) -> dict[str, Any]:
    start_index = len(transport.records)
    record: dict[str, Any] = {
        "seed": seed, "phase": phase, "status": "FAIL", "request": deepcopy(payload),
        "response": None, "raw_usage": None, "transport_record_indices": [],
    }
    try:
        response = transport(**deepcopy(payload))
        record["response"] = deepcopy(response)
        if isinstance(response, Mapping):
            record["raw_usage"] = deepcopy(response.get("usage"))
        tool_call = _validate_response(response, phase)
        if tool_call is not None:
            record["validated_tool_call"] = tool_call
        record["status"] = "PASS"
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        record["transport_record_indices"] = list(range(start_index, len(transport.records)))
    return record


def probe_model_transport(config: Mapping[str, Any], *, transport: LocalChatTransport) -> dict[str, Any]:
    """Run all three neutral probes for each prescribed seed without retries.

    If automatic tool-call validation fails, its continuation is NOT_RUN. Later
    seeds still receive fresh neutral requests. Each request/response and raw
    usage remain in per_seed[].probes; transport_records retains the supplied
    client's complete record list with the probe's starting index identified.
    """
    settings, model = _probe_settings(config)
    if not callable(transport) or not isinstance(getattr(transport, "records", None), list):
        raise ValueError("transport must be a callable with LocalChatTransport-compatible records")
    started_at = datetime.now(timezone.utc).isoformat()
    transport_start = len(transport.records)
    per_seed: list[dict[str, Any]] = []
    for seed in EXPERIMENT_SEEDS:
        common = {**settings, "seed": seed}
        plain_payload = {**common, "messages": [{"role": "user", "content": "Reply with a short greeting."}]}
        plain = _request(transport, seed, "plain_chat", plain_payload)
        tool_payload = {
            **common,
            "messages": [{"role": "user", "content": "Use the readiness__add tool once to add the integers 19 and 23."}],
            "tools": [deepcopy(_TOOL)], "tool_choice": "auto",
        }
        tool = _request(transport, seed, "automatic_tool_call", tool_payload)
        if tool["status"] == "PASS":
            call = tool["validated_tool_call"]
            assistant = deepcopy(tool["response"]["choices"][0]["message"])
            continuation_payload = {
                **common, "tools": [deepcopy(_TOOL)], "tool_choice": "auto",
                "messages": [
                    *deepcopy(tool_payload["messages"]), assistant,
                    {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": str(call["arguments"]["a"] + call["arguments"]["b"])},
                ],
            }
            continuation = _request(transport, seed, "tool_result_continuation", continuation_payload)
        else:
            continuation = {
                "seed": seed, "phase": "tool_result_continuation", "status": "NOT_RUN",
                "reason": "automatic_tool_call_failed", "request": None, "response": None,
                "raw_usage": None, "transport_record_indices": [],
            }
        probes = [plain, tool, continuation]
        per_seed.append({"seed": seed, "status": "PASS" if all(probe["status"] == "PASS" for probe in probes) else "FAIL", "probes": probes})
    records = [probe for seed_result in per_seed for probe in seed_result["probes"]]
    return {
        "schema": "appworld_model_transport_probe_v1", "status": "PASS" if all(row["status"] == "PASS" for row in per_seed) else "FAIL",
        "seeds": list(EXPERIMENT_SEEDS), "model_identity": {"repo_id": model["repo_id"], "revision": model["revision"]},
        "model_identity_source": "launcher_configuration", "tool_call_parser": model["tool_call_parser"],
        "serving_backend": model["serving_backend"], "serving_version": model["serving_version"],
        "qualification_scope": "neutral_chat_and_automatic_tool_call_interface",
        "appworld_tasks_executed": 0, "native_task_qualification": "NOT_EVALUATED",
        "generation_settings": settings, "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
        "per_seed": per_seed, "expected_request_count": len(EXPERIMENT_SEEDS) * 3,
        "request_count": sum(probe["status"] != "NOT_RUN" for probe in records),
        "not_run_count": sum(probe["status"] == "NOT_RUN" for probe in records),
        "transport_record_start_index": transport_start, "transport_records": deepcopy(transport.records),
    }

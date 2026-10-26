"""Wire-contract tests; these do not qualify a real Qwen tool-call parser."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import urllib.error
from threading import Thread

import pytest

from population_dynamics.appworld_benchmark_transport import (
    ChatContextBudgetExceeded, ChatTransportError, LocalChatTransport,
)


@contextmanager
def server(*, status=200, body=None, extra_declared_bytes=0, responses=None):
    requests = []
    if body is None:
        body = {"choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, json.loads(content), self.headers.get("X-Request-Id")))
            response_status, response_body = (
                responses[min(len(requests) - 1, len(responses) - 1)]
                if responses else (status, body)
            )
            encoded = json.dumps(response_body, ensure_ascii=False).encode("utf-8")
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded) + extra_declared_bytes))
            self.send_header("X-Request-Id", "server-receipt")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            pass

    endpoint = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=endpoint.serve_forever, daemon=False)
    thread.start()
    try:
        yield f"http://127.0.0.1:{endpoint.server_port}/v1", requests
    finally:
        endpoint.shutdown()
        endpoint.server_close()
        thread.join()


def request():
    return {"model": "pinned-local", "messages": [{"role": "user", "content": "retain 中文 and spaces"}],
            "seed": 41, "max_tokens": 128, "temperature": 0.6,
            "client_name": "openai", "api_type": "chat_completions"}


def test_exact_wire_payload_and_measured_usage(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    with server() as (url, seen):
        client = LocalChatTransport(url)
        result = client(**request())
    assert len(seen) == 1
    assert seen[0][0] == "/v1/chat/completions"
    expected = request()
    del expected["client_name"]
    del expected["api_type"]
    assert seen[0][1] == expected
    assert seen[0][2] == client.records[0]["request_id"]
    assert result["usage"]["total_tokens"] == 5
    assert client.records[0]["usage"] == result["usage"]
    assert client.records[0]["completion_known"] is True
    assert client.records[0]["server_request_id"] == "server-receipt"


def test_no_retry_on_server_failure():
    with server(status=500, body={"error": "temporary failure"}) as (url, seen):
        client = LocalChatTransport(url)
        with pytest.raises(ChatTransportError) as raised:
            client(**request())
    assert len(seen) == 1
    assert raised.value.completion_known is False
    assert raised.value.status_code == 500
    assert client.records[0]["attempts"] == 1
    assert client.records[0]["usage"] is None


def test_missing_usage_is_preserved():
    body = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
    with server(body=body) as (url, _):
        client = LocalChatTransport(url)
        result = client(**request())
    assert "usage" not in result
    assert client.records[0]["usage"] is None


def test_tool_calls_remain_exact_and_do_not_qualify_parser():
    call = {"id": "call_1", "type": "function",
            "function": {"name": "fixture__echo", "arguments": '{"text":"中文"}'}}
    body = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
    with server(body=body) as (url, seen):
        client = LocalChatTransport(url)
        payload = request()
        payload["tools"] = [{"type": "function", "function": {"name": "fixture__echo"}}]
        result = client(**payload)
    assert result["choices"][0]["message"]["tool_calls"] == [call]
    assert seen[0][1]["tools"] == payload["tools"]
    assert not hasattr(client, "tool_calls_validated")


@pytest.mark.parametrize("extra", [
    {"stream": True}, {"n": 2}, {"max_retries": 3}, {"api_type": "responses"},
    {"api_key": "unused"}, {"base_url": "http://other.invalid/v1"},
])
def test_unsupported_requests_do_not_reach_server(extra):
    with server() as (url, seen):
        client = LocalChatTransport(url)
        with pytest.raises(ValueError):
            client(**{**request(), **extra})
    assert seen == []
    assert client.records == []


def test_malformed_server_object_is_unresolved():
    with server(body={"choices": []}) as (url, seen):
        client = LocalChatTransport(url)
        with pytest.raises(ChatTransportError):
            client(**request())
    assert len(seen) == 1
    assert client.records[0]["completion_known"] is False
    assert client.records[0]["usage"] is None


def test_truncated_http_response_remains_unresolved_without_retry():
    with server(extra_declared_bytes=10) as (url, seen):
        client = LocalChatTransport(url)
        with pytest.raises(ChatTransportError):
            client(**request())
    assert len(seen) == 1
    assert client.records[0]["status"] == "unresolved"
    assert client.records[0]["completion_known"] is False
    assert client.records[0]["usage"] is None


def context_overflow(*, requested=8192, context=32768, prompt=26113):
    """Synthetic body using the observed vLLM 0.11.2 validation message."""
    return {"error": {"message": (
        f"'max_tokens' or 'max_completion_tokens' is too large: {requested}. "
        f"This model's maximum context length is {context} tokens and your "
        f"request has {prompt} input tokens ({requested} > {context} - {prompt})."
    )}}


def completion():
    return {"choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 26113, "completion_tokens": 2, "total_tokens": 26115}}


@pytest.mark.parametrize("parameter", ["max_tokens", "max_completion_tokens"])
def test_verified_overflow_reissues_once_with_exact_history_and_smaller_cap(parameter):
    payload = request()
    payload.pop("max_tokens")
    payload[parameter] = 8192
    original = deepcopy(payload)
    with server(responses=[(400, context_overflow()), (200, completion())]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        result = client(**payload)
    assert payload == original and len(seen) == len(client.records) == 2
    expected = deepcopy(seen[0][1])
    expected[parameter] = 6655
    assert seen[1][1] == expected
    assert seen[1][1]["messages"] == payload["messages"]
    assert seen[1][1]["seed"] == 41
    first, second = client.records
    assert first["status"] == "pre_generation_rejected"
    assert first["generation_started"] is False and first["generated_completion_tokens"] == 0
    assert first["usage"] is None
    assert json.loads(first["error_response_body"]) == context_overflow()
    assert first["request_id"] != second["request_id"]
    assert second["corrects_request_id"] == first["request_id"]
    assert first["logical_request_id"] == second["logical_request_id"] == first["request_id"]
    assert first["attempts"] == second["attempts"] == 1
    assert first[parameter] == 8192 and second[parameter] == 6655
    assert second["usage"] == result["usage"] == completion()["usage"]
    assert first["context_budget_correction"] == second["context_budget_correction"] == {
        "source": "vllm_0.11.2_pre_generation_context_rejection",
        "parameter": parameter, "context_window_tokens": 32768, "prompt_tokens": 26113,
        "requested_completion_tokens": 8192, "adjusted_completion_tokens": 6655,
    }


def test_one_remaining_context_token_is_allowed():
    payload = {**request(), "max_tokens": 8192}
    with server(responses=[(400, context_overflow(prompt=32767)),
                           (200, {"choices": [{"message": {"role": "assistant", "content": "x"}}]})]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        client(**payload)
    assert len(seen) == 2 and seen[1][1]["max_tokens"] == 1
    assert client.records[1]["usage"] is None


@pytest.mark.parametrize("body,window,payload_update", [
    (context_overflow(), None, {}),
    (context_overflow(), 65536, {}),
    (context_overflow(requested=8191), 32768, {}),
    (context_overflow(context=65536), 32768, {}),
    (context_overflow(prompt=0), 32768, {}),
    (context_overflow(prompt=32768), 32768, {}),
    (context_overflow(prompt=32769), 32768, {}),
    (context_overflow(), 32768, {"max_completion_tokens": 8192}),
    (context_overflow(), 32768, {"max_tokens": True}),
    ({"error": {"message": "unverified no-room rejection"}}, 32768, {}),
    ({"error": context_overflow()["error"]["message"]}, 32768, {}),
    ("not a structured error", 32768, {}),
])
def test_unverified_or_nonpositive_context_failure_has_no_reissue_or_invented_usage(
    body, window, payload_update,
):
    payload = {**request(), "max_tokens": 8192, **payload_update}
    with server(status=400, body=body) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=window)
        with pytest.raises(ChatTransportError) as raised:
            client(**payload)
    assert len(seen) == len(client.records) == 1
    assert raised.value.completion_known is False
    assert client.records[0]["status"] == "http_error" and client.records[0]["usage"] is None
    assert "generation_started" not in client.records[0]
    assert "context_budget_correction" not in client.records[0]


def test_second_context_rejection_is_not_reissued():
    payload = {**request(), "max_tokens": 8192}
    with server(responses=[
        (400, context_overflow()), (400, context_overflow(requested=6655, prompt=26114)),
    ]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatTransportError) as raised:
            client(**payload)
    assert len(seen) == len(client.records) == 2
    assert raised.value.request_id == client.records[1]["request_id"]
    assert client.records[1]["status"] == "http_error" and client.records[1]["usage"] is None


def test_server_failure_does_not_gain_a_retry_when_context_correction_is_enabled():
    with server(status=500, body=context_overflow()) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatTransportError):
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == 1 and client.records[0]["usage"] is None


def test_incomplete_error_body_does_not_qualify_context_correction():
    with server(status=400, body=context_overflow(), extra_declared_bytes=10) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatTransportError):
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == 1
    assert client.records[0]["status"] == "unresolved" and client.records[0]["usage"] is None


@pytest.mark.parametrize("failure", [TimeoutError("synthetic timeout"),
                                      urllib.error.URLError("synthetic connection loss")])
def test_ambiguous_transport_failure_has_no_reissue(monkeypatch, failure):
    client = LocalChatTransport("http://127.0.0.1:1/v1", context_window_tokens=32768)
    sent = []

    def fail(request, timeout):
        sent.append(request)
        raise failure

    monkeypatch.setattr(client.opener, "open", fail)
    with pytest.raises(ChatTransportError):
        client(**{**request(), "max_tokens": 8192})
    assert len(sent) == len(client.records) == 1
    assert client.records[0]["status"] == "unresolved"
    assert client.records[0]["completion_known"] is False and client.records[0]["usage"] is None


@pytest.mark.parametrize("window", [0, -1, True, 32768.0, "32768"])
def test_invalid_configured_context_is_rejected(window):
    with pytest.raises(ValueError, match="positive integer context"):
        LocalChatTransport("http://127.0.0.1:1/v1", context_window_tokens=window)


def context_exhausted(*, context=32768, prompt=32768):
    """Synthetic body using the pinned vLLM 0.11.2 no-room message."""
    return {"error": {"message": (
        f"This model's maximum context length is {context} tokens. "
        f"However, your request has {prompt} input tokens. "
        "Please reduce the length of the input messages."
    ), "type": "BadRequestError", "code": 400}}


@pytest.mark.parametrize("prompt", [32768, 32769])
def test_pinned_no_room_rejection_resolves_no_generation_without_usage_or_reissue(prompt):
    body = context_exhausted(prompt=prompt)
    with server(status=400, body=body) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatContextBudgetExceeded) as raised:
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == len(client.records) == 1
    record = client.records[0]
    assert record["status"] == "pre_generation_rejected"
    assert record["generation_started"] is False and record["generated_completion_tokens"] == 0
    assert record["usage"] is None and json.loads(record["error_response_body"]) == body
    assert raised.value.generation_started is False and raised.value.generated_completion_tokens == 0
    assert raised.value.context_budget_rejection == record["context_budget_rejection"] == {
        "source": "vllm_0.11.2_pre_generation_context_rejection",
        "reason": "context_window_exhausted", "context_window_tokens": 32768, "prompt_tokens": prompt,
    }


@pytest.mark.parametrize("body,window,status", [
    (context_exhausted(), None, 400),
    (context_exhausted(), 65536, 400),
    (context_exhausted(context=65536, prompt=65536), 32768, 400),
    (context_exhausted(prompt=32767), 32768, 400),
    (context_exhausted(), 32768, 500),
    ({"error": {"message": "unverified no-room rejection"}}, 32768, 400),
])
def test_unverified_no_room_response_keeps_unknown_generation(body, window, status):
    with server(status=status, body=body) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=window)
        with pytest.raises(ChatTransportError) as raised:
            client(**request())
    assert not isinstance(raised.value, ChatContextBudgetExceeded)
    assert len(seen) == 1 and client.records[0]["usage"] is None
    assert "generation_started" not in client.records[0]


def test_no_room_after_cap_correction_is_recorded_without_a_third_request():
    with server(responses=[(400, context_overflow()), (400, context_exhausted())]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatContextBudgetExceeded) as raised:
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == len(client.records) == 2
    assert client.records[1]["status"] == "pre_generation_rejected"
    assert client.records[1]["generated_completion_tokens"] == 0
    assert client.records[1]["request_id"] == raised.value.request_id
    assert client.records[1]["corrects_request_id"] == client.records[0]["request_id"]


def test_observed_vllm_wrapper_body_corrects_the_completion_cap_once():
    body = {"error": {
        "message": (
            "'max_tokens' or 'max_completion_tokens' is too large: 8192. "
            "This model's maximum context length is 32768 tokens and your request "
            "has 26113 input tokens (8192 > 32768 - 26113). None"
        ),
        "type": "BadRequestError", "param": None, "code": 400,
    }}
    payload = {**request(), "max_tokens": None, "max_completion_tokens": 8192}
    before = deepcopy(payload)
    with server(responses=[(400, body), (200, completion())]) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        result = client(**payload)
    assert payload == before and len(seen) == len(client.records) == 2
    assert seen[1][1] == {**seen[0][1], "max_completion_tokens": 6655}
    assert json.loads(client.records[0]["error_response_body"]) == body
    assert client.records[0]["status"] == "pre_generation_rejected"
    assert client.records[1]["corrects_request_id"] == client.records[0]["request_id"]
    assert result == completion()


@pytest.mark.parametrize("message_kind", ["overflow", "no_room"])
@pytest.mark.parametrize("suffix", [" None ", " None None", " other"])
def test_wrapper_suffix_must_be_the_exact_observed_literal(message_kind, suffix):
    body = context_overflow() if message_kind == "overflow" else context_exhausted()
    body["error"]["message"] += suffix
    with server(status=400, body=body) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatTransportError):
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == len(client.records) == 1
    assert client.records[0]["status"] == "http_error"
    assert client.records[0]["usage"] is None


def test_pinned_no_room_wrapper_body_resolves_no_generation_without_reissue():
    body = context_exhausted()
    body["error"]["message"] += " None"
    body["error"]["param"] = None
    with server(status=400, body=body) as (url, seen):
        client = LocalChatTransport(url, context_window_tokens=32768)
        with pytest.raises(ChatContextBudgetExceeded) as raised:
            client(**{**request(), "max_tokens": 8192})
    assert len(seen) == len(client.records) == 1
    assert raised.value.generation_started is False
    assert raised.value.generated_completion_tokens == 0
    assert client.records[0]["status"] == "pre_generation_rejected"
    assert client.records[0]["usage"] is None
    assert json.loads(client.records[0]["error_response_body"]) == body

"""HTTP transport for a local AppWorld model server.

Only an exact, verified vLLM pre-generation context rejection permits one request
with a reduced completion cap. Ambiguous failures are never retried. Raw usage
is never inferred from missing fields. Tool-parser admission is separate.
"""
from __future__ import annotations

from copy import deepcopy
import http.client
import json
import re
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import uuid


# Both templates are from vLLM 0.11.2's _validate_input, before generation.
# serving_chat adds the exact ' None' suffix when the exception has no cause.
_VLLM_COMPLETION_OVERFLOW = re.compile(
    r"'max_tokens' or 'max_completion_tokens' is too large: (?P<requested>[0-9]+)\. "
    r"This model's maximum context length is (?P<context>[0-9]+) tokens and your "
    r"request has (?P<prompt>[0-9]+) input tokens "
    r"\((?P=requested) > (?P=context) - (?P=prompt)\)\.(?: None)?"
)

_VLLM_CONTEXT_EXHAUSTED = re.compile(
    r"This model's maximum context length is (?P<context>[0-9]+) tokens\. "
    r"However, your request has (?P<prompt>[0-9]+) input tokens\. "
    r"Please reduce the length of the input messages\.(?: None)?"
)


class ChatTransportError(RuntimeError):
    """A request failed; completion and usage can remain unknown."""

    def __init__(
        self, message: str, request_id: str, status_code: int | None = None, *,
        response_body: Any = None,
    ):
        super().__init__(message)
        self.request_id = request_id
        self.status_code = status_code
        self.completion_known = False
        self.response_body = deepcopy(response_body)


class ChatContextBudgetExceeded(ChatTransportError):
    """Pinned server validation proved that generation did not start."""

    def __init__(self, error: ChatTransportError, rejection: dict[str, Any]):
        super().__init__(
            "local model input exhausted the configured context window",
            error.request_id, error.status_code, response_body=error.response_body,
        )
        self.context_budget_rejection = deepcopy(rejection)
        self.generation_started = False
        self.generated_completion_tokens = 0


class LocalChatTransport:
    """Single-flight, non-streaming client for an admitted local model endpoint."""

    def __init__(
        self, base_url: str, *, timeout_seconds: float = 600.0,
        context_window_tokens: int | None = None,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
        ):
            raise ValueError("a model endpoint without credentials/query/fragment is required")
        if timeout_seconds <= 0:
            raise ValueError("positive transport timeout required")
        if context_window_tokens is not None and (
            type(context_window_tokens) is not int or context_window_tokens <= 0
        ):
            raise ValueError("an explicit positive integer context window is required")
        self.context_window_tokens = context_window_tokens
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # The endpoint is explicit. Ambient internet proxies must not intercept
        # private task messages sent to a local Slurm model server.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.records: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        payload = deepcopy(kwargs)
        payload.pop("client_name", None)
        api_type = payload.pop("api_type", "chat_completions")
        if api_type != "chat_completions":
            raise ValueError("only chat_completions are supported")
        if payload.get("stream", False) is not False:
            raise ValueError("streaming completions are not supported")
        if payload.get("n", 1) != 1:
            raise ValueError("the benchmark transport requires one returned completion")
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ValueError("an explicit served model name is required")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise ValueError("a nonempty native message history is required")
        # Internal retry controls must be applied by the caller, not forwarded
        # to an OpenAI-compatible server as unknown generation parameters.
        if any(key in payload for key in ("max_retries", "retry_after_n_seconds")):
            raise ValueError("retry controls are not generation parameters")
        for key in ("api_key", "base_url", "api_base"):
            if key in payload:
                raise ValueError("endpoint and credentials belong outside the generation payload")
        logical_request_id = "appworld-" + uuid.uuid4().hex
        try:
            return self._post(payload, logical_request_id=logical_request_id)
        except ChatTransportError as error:
            correction = self._context_correction(payload, error)
            if correction is None:
                raise
            rejected = self.records[-1]
            rejected.update(
                status="pre_generation_rejected", generation_started=False,
                generated_completion_tokens=0, context_budget_correction=deepcopy(correction),
            )
            adjusted = deepcopy(payload)
            adjusted[correction["parameter"]] = correction["adjusted_completion_tokens"]
            # _post has no recovery loop. A failed correction is never reissued.
            return self._post(
                adjusted, logical_request_id=logical_request_id,
                corrects_request_id=error.request_id, correction=correction,
            )

    def _context_correction(
        self, payload: dict[str, Any], error: ChatTransportError,
    ) -> dict[str, Any] | None:
        if self.context_window_tokens is None or error.status_code != 400:
            return None
        body = error.response_body
        detail = body.get("error") if isinstance(body, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else None
        if not isinstance(message, str):
            return None
        match = _VLLM_COMPLETION_OVERFLOW.fullmatch(message)
        if match is None:
            return None
        limits = [(key, payload[key]) for key in ("max_tokens", "max_completion_tokens")
                  if payload.get(key) is not None]
        if len(limits) != 1:
            return None
        parameter, requested = limits[0]
        if type(requested) is not int or requested <= 0:
            return None
        context, prompt = int(match["context"]), int(match["prompt"])
        if context != self.context_window_tokens or int(match["requested"]) != requested:
            return None
        remaining = context - prompt
        # Only the separate pinned no-room signature resolves an exhausted
        # prompt. An inconsistent overflow message cannot qualify a correction.
        if not 0 < remaining < requested:
            return None
        return {
            "source": "vllm_0.11.2_pre_generation_context_rejection",
            "parameter": parameter, "context_window_tokens": context,
            "prompt_tokens": prompt, "requested_completion_tokens": requested,
            "adjusted_completion_tokens": remaining,
        }

    def _context_exhaustion(self, error: ChatTransportError) -> dict[str, Any] | None:
        if self.context_window_tokens is None or error.status_code != 400:
            return None
        body = error.response_body
        detail = body.get("error") if isinstance(body, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else None
        if not isinstance(message, str):
            return None
        match = _VLLM_CONTEXT_EXHAUSTED.fullmatch(message)
        if match is None:
            return None
        context, prompt = int(match["context"]), int(match["prompt"])
        if context != self.context_window_tokens or prompt < context:
            return None
        return {
            "source": "vllm_0.11.2_pre_generation_context_rejection",
            "reason": "context_window_exhausted",
            "context_window_tokens": context, "prompt_tokens": prompt,
        }

    def _post(
        self, payload: dict[str, Any], *, logical_request_id: str,
        corrects_request_id: str | None = None, correction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_id = ("appworld-" + uuid.uuid4().hex) if corrects_request_id else logical_request_id
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
            method="POST",
        )
        record: dict[str, Any] = {
            "request_id": request_id, "status": "pending", "attempts": 1,
            "completion_known": False, "usage": None,
            "logical_request_id": logical_request_id,
            "max_tokens": payload.get("max_tokens"),
            "max_completion_tokens": payload.get("max_completion_tokens"),
        }
        if corrects_request_id is not None:
            record.update(corrects_request_id=corrects_request_id,
                          context_budget_correction=deepcopy(correction))
        self.records.append(record)
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                record["status_code"] = response.status
                record["server_request_id"] = response.headers.get("X-Request-Id")
            result = json.loads(raw)
            if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
                raise ValueError("server did not return a chat-completion object")
            if len(result["choices"]) != 1:
                raise ValueError("server did not return exactly one completion")
            record.update(status="completed", completion_known=True, usage=deepcopy(result.get("usage")))
            return result
        except urllib.error.HTTPError as error:
            record.update(status="http_error", status_code=error.code,
                          server_request_id=error.headers.get("X-Request-Id"))
            try:
                raw_error = error.read()
            except (http.client.HTTPException, TimeoutError, OSError, ValueError) as read_error:
                record.update(status="unresolved", error_type=type(read_error).__name__,
                              error=str(read_error))
                raise ChatTransportError(
                    f"local model error response unresolved: {read_error}", request_id, error.code,
                ) from read_error
            finally:
                error.close()
            record["error_response_body"] = raw_error.decode("utf-8", errors="replace")
            try:
                error_body = json.loads(raw_error)
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_body = None
            transport_error = ChatTransportError(
                f"local model server returned HTTP {error.code}", request_id, error.code,
                response_body=error_body,
            )
            rejection = self._context_exhaustion(transport_error)
            if rejection is not None:
                record.update(
                    status="pre_generation_rejected", generation_started=False,
                    generated_completion_tokens=0, context_budget_rejection=deepcopy(rejection),
                )
                raise ChatContextBudgetExceeded(transport_error, rejection) from error
            raise transport_error from error
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as error:
            record.update(status="unresolved", error_type=type(error).__name__, error=str(error))
            raise ChatTransportError(f"local model completion unresolved: {error}", request_id) from error
        finally:
            record["http_wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started

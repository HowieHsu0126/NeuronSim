"""Dedicated Unix-FD channel between isolated CUGA and its native-tool owner.

The owner supplies an already started NativeToolClient. This module does not
start native processes, retry requests, add request timeouts, or score tasks.
relay_native_tools owns the parent socket and native client cleanup.
NativeToolProxy.connect owns the inherited child FD. stdout and stderr remain
ordinary logs.

The process owner must cancel and await the relay if the isolated process exits
while a native call is outstanding. Cancellation reaches NativeToolClient,
which retains its pending exchange for cleanup.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import socket
from typing import Any

from .appworld_benchmark_native_tools import (
    NativeToolExecutionError,
    UnknownNativeOutcome,
    _json_copy,
    _strict_json,
)


class _ProtocolError(ValueError):
    pass


class _JsonPeer:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self.buffer = bytearray()
        self._search_from = 0

    @classmethod
    async def from_socket(cls, sock: socket.socket) -> _JsonPeer:
        sock.setblocking(False)
        try:
            reader, writer = await asyncio.open_connection(sock=sock)
        except BaseException:
            sock.close()
            raise
        return cls(reader, writer)

    async def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(
            _json_copy(message), ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"
        self.writer.write(payload)
        await self.writer.drain()

    async def receive(self) -> dict[str, Any]:
        # Scan each incoming byte once, without readline's default 64 KiB limit.
        while True:
            boundary = self.buffer.find(b"\n", self._search_from)
            if boundary >= 0:
                line = bytes(self.buffer[:boundary])
                del self.buffer[:boundary + 1]
                self._search_from = 0
                value = _strict_json(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise _ProtocolError("native proxy frame must be an object")
                return value
            self._search_from = len(self.buffer)
            chunk = await self.reader.read(65536)
            if not chunk:
                raise EOFError("native proxy peer closed before a complete frame")
            self.buffer.extend(chunk)

    def abort(self) -> None:
        # No pending output is retried or flushed after an uncertain exchange.
        self.writer.transport.abort()


class NativeToolProxy:
    """Child-side duck-typed executor for run_bound_cuga_graph."""

    def __init__(self, peer: _JsonPeer):
        self._peer = peer
        self._lock = asyncio.Lock()
        self._next_id = 1
        self._closed = False
        self._broken = False
        self._close_task: asyncio.Task | None = None
        self.public_context: dict[str, Any] = {}

    @classmethod
    async def connect(cls, fd: int) -> NativeToolProxy:
        """Take ownership of the inherited connected Unix socket FD."""
        if type(fd) is not int or fd < 0:
            raise ValueError("native proxy requires a nonnegative inherited FD")
        sock = socket.socket(fileno=fd)
        proxy = cls(await _JsonPeer.from_socket(sock))
        try:
            response = await proxy._exchange("context")
            if (
                set(response) != {"id", "status", "public_context"}
                or response["status"] != "context"
                or not isinstance(response["public_context"], dict)
            ):
                proxy._fail("parent did not return public context")
            proxy.public_context = _json_copy(response["public_context"])
        except BaseException:
            proxy._broken = True
            proxy._peer.abort()
            raise
        return proxy

    def _fail(self, message: str):
        self._broken = True
        self._peer.abort()
        raise UnknownNativeOutcome(message)

    async def _exchange(self, operation: str, **fields: Any) -> dict[str, Any]:
        if self._broken or (self._closed and operation != "close"):
            raise UnknownNativeOutcome("native proxy is closed or unresolved")
        request = _json_copy({
            "id": self._next_id, "operation": operation, **fields,
        })
        self._next_id += 1
        try:
            await self._peer.send(request)
            response = await self._peer.receive()
            if (
                type(response.get("id")) is not int
                or response["id"] != request["id"]
                or not isinstance(response.get("status"), str)
            ):
                raise _ProtocolError("native proxy response identity is invalid")
        except BaseException as error:
            self._broken = True
            self._peer.abort()
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            raise UnknownNativeOutcome(
                "native proxy response is unresolved; request was not repeated"
            ) from error
        return response

    async def __call__(self, app: str, api: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments, dict):
            raise TypeError("native tool arguments must be a dictionary")
        arguments = _json_copy(arguments)
        async with self._lock:
            response = await self._exchange(
                "call", app=app, api=api, arguments=arguments
            )
            status = response["status"]
            if status == "returned" and set(response) == {"id", "status", "result"}:
                return _json_copy(response["result"])
            if status in {"native_error", "rejected", "unknown", "protocol_error"}:
                error = response.get("error")
                if not isinstance(error, str):
                    self._fail("native proxy error response lacks text")
                if status == "native_error":
                    raise NativeToolExecutionError(error)
                if status == "rejected":
                    error_class = {
                        "TypeError": TypeError, "ValueError": ValueError,
                    }.get(response.get("error_type"))
                    if error_class is None:
                        self._fail("native proxy rejection has an unsupported error_type")
                    raise error_class(error)
                self._fail(error)
            self._fail("native proxy returned an unsupported response")

    async def _close_once(self) -> dict[str, Any]:
        async with self._lock:
            try:
                if self._broken:
                    raise UnknownNativeOutcome(
                        "proxy channel is unresolved; owner retains native cleanup"
                    )
                response = await self._exchange("close")
                if (
                    response["status"] == "closed"
                    and set(response) == {"id", "status", "cleanup"}
                    and isinstance(response["cleanup"], dict)
                ):
                    return _json_copy(response["cleanup"])
                if response["status"] == "cleanup_error":
                    raise RuntimeError(
                        "parent native cleanup failed: "
                        + str(response.get("error_type", "Error"))
                        + ": " + str(response.get("error", ""))
                    )
                self._fail("parent did not return its native cleanup receipt")
            finally:
                self._peer.abort()

    async def close(self) -> dict[str, Any]:
        """Request cleanup once; repeat callers receive the same result/error."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_once())
        return deepcopy(await asyncio.shield(self._close_task))


def _validate_request(
    request: dict[str, Any], expected_id: int, context_sent: bool,
) -> str:
    if type(request.get("id")) is not int or request["id"] != expected_id:
        raise _ProtocolError("proxy request IDs must increase once from 1")
    operation = request.get("operation")
    fields = {
        "context": {"id", "operation"},
        "call": {"id", "operation", "app", "api", "arguments"},
        "close": {"id", "operation"},
    }
    if not isinstance(operation, str) or set(request) != fields.get(operation):
        raise _ProtocolError("unsupported proxy operation or fields")
    if operation == "context" and context_sent:
        raise _ProtocolError("public context was already sent")
    if operation != "context" and not context_sent:
        raise _ProtocolError("public context must be requested first")
    return operation


async def relay_native_tools(
    sock: socket.socket, client: Any, *, observation: dict[str, Any],
) -> dict[str, Any]:
    """Own the parent socket and exactly one native-client cleanup operation.

    The native client supplies the only request timeout. Known tool errors are
    recoverable responses. EOF/protocol errors stop the relay. Owner cancellation
    propagates after cleanup; it is never converted into a retry.

    native_cleanup is the actual client.close() result. reply_sent means only
    that writer.drain completed. It does not claim that the peer received a
    response. Only status=closed records completion of the close exchange on
    this side. Any other terminal channel state remains RPC-unresolved.
    """
    if not isinstance(observation, dict) or observation:
        sock.close()
        await client.close()
        raise ValueError("a fresh relay observation is required")

    observation.update(
        status="starting", requests=[], rpc_outcome_unresolved=False,
        native_cleanup=None, cleanup_error=None, owner_cancelled=False,
    )
    peer: _JsonPeer | None = None
    cleanup_task: asyncio.Task | None = None
    current: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    failure: BaseException | None = None

    async def close_native_once() -> dict[str, Any]:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(client.close())
        while True:
            try:
                result = await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.cancelled():
                    raise
                observation["owner_cancelled"] = True
                if cleanup_task.done():
                    # A cancellation can race with a successful close. Keep the
                    # completed receipt before propagating owner cancellation.
                    result = cleanup_task.result()
                    break
        if not isinstance(result, dict):
            raise TypeError("native cleanup must return a dictionary")
        observation["native_cleanup"] = _json_copy(result)
        return observation["native_cleanup"]

    try:
        peer = await _JsonPeer.from_socket(sock)
        context = _json_copy(client.public_context)
        if not isinstance(context, dict):
            raise TypeError("native client has no public context")
        observation["status"] = "serving"
        expected_id = 1
        context_sent = False

        while True:
            current = None
            request = None
            try:
                request = await peer.receive()
            except (ValueError, TypeError) as error:
                raise _ProtocolError(str(error)) from error
            operation = _validate_request(request, expected_id, context_sent)
            expected_id += 1
            current = {
                "id": request["id"], "operation": operation,
                "request": deepcopy(request),
                "status": "received", "reply_sent": False,
            }
            observation["requests"].append(current)

            if operation == "context":
                response = {
                    "id": request["id"], "status": "context",
                    "public_context": context,
                }
            elif operation == "call":
                current["status"] = "forwarded"
                try:
                    result = await client(
                        request["app"], request["api"], request["arguments"]
                    )
                    response = {
                        "id": request["id"], "status": "returned",
                        "result": _json_copy(result),
                    }
                except NativeToolExecutionError as error:
                    response = {
                        "id": request["id"], "status": "native_error",
                        "error_type": type(error).__name__, "error": str(error),
                    }
                except UnknownNativeOutcome as error:
                    observation["rpc_outcome_unresolved"] = True
                    response = {
                        "id": request["id"], "status": "unknown",
                        "error_type": type(error).__name__, "error": str(error),
                    }
                except (ValueError, TypeError) as error:
                    response = {
                        "id": request["id"], "status": "rejected",
                        "error_type": type(error).__name__, "error": str(error),
                    }
            else:
                try:
                    cleanup = await close_native_once()
                except BaseException as error:
                    observation["cleanup_error"] = {
                        "error_type": type(error).__name__, "error": str(error),
                    }
                    if isinstance(error, Exception):
                        await peer.send({
                            "id": request["id"], "status": "cleanup_error",
                            "error_type": type(error).__name__, "error": str(error),
                        })
                    raise
                response = {
                    "id": request["id"], "status": "closed", "cleanup": cleanup,
                }

            await peer.send(response)
            current.update(status=response["status"], reply_sent=True)
            if "error" in response:
                current.update(
                    error=response["error"], error_type=response.get("error_type")
                )
            if operation == "context":
                context_sent = True
            if response["status"] == "unknown":
                observation["status"] = "unknown"
                break
            if operation == "close":
                observation["status"] = "closed"
                break

    except asyncio.CancelledError as error:
        observation.update(status="cancelled", owner_cancelled=True)
        failure = error
    except EOFError as error:
        observation.update(status="eof", error=str(error))
    except _ProtocolError as error:
        observation.update(
            status="protocol_error", error=str(error),
            invalid_request=deepcopy(request),
        )
        if peer is not None:
            try:
                await peer.send({
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "status": "protocol_error", "error": str(error),
                })
            except asyncio.CancelledError as cancelled:
                observation["owner_cancelled"] = True
                failure = cancelled
            except Exception as delivery_error:
                observation["protocol_error_delivery"] = {
                    "error_type": type(delivery_error).__name__,
                    "error": str(delivery_error),
                }
    except BaseException as error:
        observation.update(
            status="errored", error_type=type(error).__name__, error=str(error),
        )
        failure = error
    finally:
        try:
            await close_native_once()
        except BaseException as error:
            observation.update(
                status="cleanup_failed",
                cleanup_error={
                    "error_type": type(error).__name__, "error": str(error),
                },
            )
            if failure is None:
                failure = error
        finally:
            if peer is not None:
                peer.abort()
            else:
                sock.close()
        if observation["owner_cancelled"] and observation["status"] != "cleanup_failed":
            observation["status"] = "cancelled"
        if observation["status"] != "closed":
            observation["rpc_outcome_unresolved"] = True

    if observation["owner_cancelled"]:
        raise asyncio.CancelledError
    if failure is not None:
        raise failure
    return observation

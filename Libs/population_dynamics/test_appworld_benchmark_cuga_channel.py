"""Real socketpair tests; no model, AppWorld task or subprocess is started."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import socket

import pytest

from .appworld_benchmark_cuga_channel import (
    NativeToolProxy,
    _JsonPeer,
    relay_native_tools,
)
from .appworld_benchmark_native_tools import (
    NativeToolExecutionError,
    UnknownNativeOutcome,
)


class FakeNativeClient:
    def __init__(self):
        self.public_context = {
            "instruction": "synthetic public task",
            "large": "文\n" * 100000,
        }
        self.calls = []
        self.close_calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.delay_close = False
        self.close_error = None
        self.cleanup = {
            "pid": 123,
            "child_reaped": True,
            "exit_code": 0,
            "termination_signals": [],
            "request_outcome_unresolved": False,
            "late_response": None,
            "pending_error": None,
        }

    async def __call__(self, app, api, arguments):
        self.calls.append((app, api, deepcopy(arguments)))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if api == "native_failure":
                raise NativeToolExecutionError(
                    "Execution failed. Traceback:\nsynthetic API failure"
                )
            if api == "reject":
                raise ValueError("synthetic rejected arguments")
            if api == "type_error":
                raise TypeError("synthetic argument type failure")
            if api == "unknown":
                self.cleanup["request_outcome_unresolved"] = True
                self.cleanup["exit_code"] = 1
                raise UnknownNativeOutcome("synthetic unresolved native return")
            if api == "hold":
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cleanup.update(
                        exit_code=1,
                        request_outcome_unresolved=True,
                        late_response={
                            "id": 1, "status": "returned", "result": False,
                        },
                    )
                    raise
            await asyncio.sleep(0)
            return deepcopy(arguments)
        finally:
            self.active_calls -= 1

    async def close(self):
        self.close_calls += 1
        self.close_started.set()
        if self.delay_close:
            await self.close_release.wait()
        await asyncio.sleep(0)
        if self.close_error is not None:
            raise self.close_error
        return deepcopy(self.cleanup)


async def connect_pair(client):
    parent, child = socket.socketpair()
    observation = {}
    relay = asyncio.create_task(
        relay_native_tools(parent, client, observation=observation)
    )
    proxy = await NativeToolProxy.connect(child.detach())
    return proxy, relay, observation


def test_real_socketpair_preserves_large_public_context_and_json_types(capsys):
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        assert proxy.public_context == client.public_context
        value = {
            "null": None,
            "false": False,
            "zero": 0,
            "integer": 2**80,
            "negative": -17,
            "float": 1.25,
            "unicode": "神经\n\"quoted\"\\path",
            "nested": [{"value": []}, True],
            "large": "x" * 200000,
        }
        print("ordinary stdout remains a log")
        assert await proxy("app", "echo", value) == value
        first, second = await asyncio.gather(proxy.close(), proxy.close())
        assert first == second == client.cleanup
        first["exit_code"] = 99
        assert (await proxy.close())["exit_code"] == 0
        assert await relay is observation
        assert observation["status"] == "closed"
        assert observation["rpc_outcome_unresolved"] is False
        assert all(row["reply_sent"] for row in observation["requests"])
        assert all("reply_delivered" not in row for row in observation["requests"])
        assert client.close_calls == 1
        assert len(client.calls) == 1

    asyncio.run(run())
    assert capsys.readouterr().out == "ordinary stdout remains a log\n"


def test_large_frame_search_work_is_linear_across_small_reads():
    class CountedBuffer(bytearray):
        scanned = 0

        def find(self, value, start=0, end=None):
            stop = len(self) if end is None else end
            self.scanned += max(0, stop - start)
            return super().find(value, start, stop)

        def __contains__(self, value):
            self.scanned += len(self)
            return super().__contains__(value)

    class ChunkReader:
        def __init__(self, payload):
            self.payload = payload
            self.offset = 0

        async def read(self, size):
            chunk = self.payload[self.offset:self.offset + min(size, 512)]
            self.offset += len(chunk)
            return chunk

    async def run():
        value = {"large": "x" * (64 * 1024)}
        payload = json.dumps(value).encode() + b"\n"
        peer = _JsonPeer(ChunkReader(payload), None)
        peer.buffer = CountedBuffer()
        assert await peer.receive() == value
        assert peer.buffer.scanned <= 2 * len(payload)

    asyncio.run(run())


def test_known_native_value_and_type_errors_allow_resolved_followup():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        with pytest.raises(NativeToolExecutionError, match="synthetic API failure"):
            await proxy("app", "native_failure", {})
        with pytest.raises(ValueError, match="synthetic rejected arguments"):
            await proxy("app", "reject", {})
        with pytest.raises(TypeError, match="synthetic argument type failure"):
            await proxy("app", "type_error", {})
        assert await proxy("app", "echo", {"value": False}) == {"value": False}
        await proxy.close()
        await relay
        assert [row["status"] for row in observation["requests"]] == [
            "context", "native_error", "rejected", "rejected", "returned", "closed",
        ]
        rejections = [
            row for row in observation["requests"] if row["status"] == "rejected"
        ]
        assert [row["error_type"] for row in rejections] == ["ValueError", "TypeError"]
        assert observation["rpc_outcome_unresolved"] is False
        assert client.close_calls == 1

    asyncio.run(run())


def test_concurrent_calls_are_serialized_without_frame_mixing():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        results = await asyncio.gather(*[
            proxy("app", "echo", {"index": index}) for index in range(12)
        ])
        assert results == [{"index": index} for index in range(12)]
        assert client.max_active_calls == 1
        await proxy.close()
        await relay
        assert [row["id"] for row in observation["requests"]] == list(range(1, 15))
        assert client.close_calls == 1

    asyncio.run(run())


def test_close_waits_for_existing_call_and_runs_only_once():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        call = asyncio.create_task(proxy("app", "hold", {"value": 7}))
        await client.started.wait()
        close = asyncio.create_task(proxy.close())
        await asyncio.sleep(0)
        client.release.set()
        assert await call == {"value": 7}
        assert await close == client.cleanup
        assert await proxy.close() == client.cleanup
        await relay
        assert observation["status"] == "closed"
        assert client.close_calls == 1

    asyncio.run(run())


def test_protocol_error_executes_no_native_call_and_still_cleans_up():
    async def run():
        client = FakeNativeClient()
        parent, child = socket.socketpair()
        child.setblocking(False)
        observation = {}
        relay = asyncio.create_task(
            relay_native_tools(parent, client, observation=observation)
        )
        reader, writer = await asyncio.open_connection(sock=child)
        writer.write(b'{"id":2,"operation":"context"}\n')
        await writer.drain()
        response = json.loads(await reader.readline())
        assert response["status"] == "protocol_error"
        assert await relay is observation
        assert observation["status"] == "protocol_error"
        assert observation["rpc_outcome_unresolved"] is True
        assert client.calls == []
        assert client.close_calls == 1
        writer.close()
        await writer.wait_closed()

    asyncio.run(run())


def test_eof_before_first_request_closes_owned_native_client():
    async def run():
        client = FakeNativeClient()
        parent, child = socket.socketpair()
        observation = {}
        relay = asyncio.create_task(
            relay_native_tools(parent, client, observation=observation)
        )
        child.close()
        await relay
        assert observation["status"] == "eof"
        assert observation["rpc_outcome_unresolved"] is True
        assert observation["native_cleanup"] == client.cleanup
        assert client.calls == []
        assert client.close_calls == 1

    asyncio.run(run())


def test_eof_after_call_reply_sent_does_not_invent_a_close_exchange():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        assert await proxy("app", "echo", {"value": 8}) == {"value": 8}
        proxy._peer.abort()
        await relay
        call = observation["requests"][-1]
        assert call["operation"] == "call" and call["reply_sent"] is True
        assert observation["status"] == "eof"
        assert observation["rpc_outcome_unresolved"] is True
        assert observation["native_cleanup"] == client.cleanup
        assert client.close_calls == 1
        assert len(client.calls) == 1

    asyncio.run(run())


def test_unknown_native_return_blocks_followup_without_retry():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        with pytest.raises(UnknownNativeOutcome, match="synthetic unresolved"):
            await proxy("app", "unknown", {})
        with pytest.raises(UnknownNativeOutcome):
            await proxy("app", "echo", {"must_not_execute": True})
        with pytest.raises(UnknownNativeOutcome):
            await proxy.close()
        await relay
        assert len(client.calls) == 1
        assert client.close_calls == 1
        assert observation["status"] == "unknown"
        assert observation["rpc_outcome_unresolved"] is True
        assert observation["native_cleanup"]["request_outcome_unresolved"] is True

    asyncio.run(run())


def test_owner_cancellation_reaches_native_call_and_preserves_late_cleanup():
    async def run():
        client = FakeNativeClient()
        proxy, relay, observation = await connect_pair(client)
        call = asyncio.create_task(proxy("app", "hold", {"mutation": 1}))
        await client.started.wait()
        relay.cancel()
        with pytest.raises(asyncio.CancelledError):
            await relay
        with pytest.raises(UnknownNativeOutcome):
            await call
        with pytest.raises(UnknownNativeOutcome):
            await proxy.close()
        with pytest.raises(UnknownNativeOutcome):
            await proxy.close()
        assert len(client.calls) == 1
        assert client.close_calls == 1
        assert observation["owner_cancelled"] is True
        assert observation["rpc_outcome_unresolved"] is True
        cleanup = observation["native_cleanup"]
        assert cleanup["request_outcome_unresolved"] is True
        assert cleanup["late_response"]["result"] is False

    asyncio.run(run())


def test_owner_cancellation_waits_for_in_progress_cleanup():
    async def run():
        client = FakeNativeClient()
        client.delay_close = True
        proxy, relay, observation = await connect_pair(client)
        close = asyncio.create_task(proxy.close())
        await client.close_started.wait()
        relay.cancel()
        await asyncio.sleep(0)
        assert not relay.done()
        client.close_release.set()
        assert await close == client.cleanup
        with pytest.raises(asyncio.CancelledError):
            await relay
        assert observation["native_cleanup"] == client.cleanup
        assert observation["owner_cancelled"] is True
        assert observation["rpc_outcome_unresolved"] is True
        assert client.close_calls == 1

    asyncio.run(run())


def test_cancellation_racing_with_finished_cleanup_retains_actual_receipt():
    class CancelAtCloseCompletion(FakeNativeClient):
        relay = None

        async def close(self):
            self.close_calls += 1
            result = deepcopy(self.cleanup)
            asyncio.get_running_loop().call_soon(self.relay.cancel)
            return result

    async def run():
        client = CancelAtCloseCompletion()
        proxy, relay, observation = await connect_pair(client)
        client.relay = relay
        assert await proxy.close() == client.cleanup
        with pytest.raises(asyncio.CancelledError):
            await relay
        assert observation["native_cleanup"] == client.cleanup
        assert observation["owner_cancelled"] is True
        assert observation["status"] == "cancelled"
        assert observation["rpc_outcome_unresolved"] is True
        assert client.close_calls == 1

    asyncio.run(run())


def test_unsettled_cleanup_is_returned_unchanged_not_promoted():
    async def run():
        client = FakeNativeClient()
        client.cleanup.update(
            exit_code=1,
            request_outcome_unresolved=True,
            late_response={"id": 7, "status": "returned", "result": None},
        )
        proxy, relay, observation = await connect_pair(client)
        assert await proxy.close() == client.cleanup
        assert await proxy.close() == client.cleanup
        await relay
        assert observation["native_cleanup"] == client.cleanup
        assert client.close_calls == 1

    asyncio.run(run())


def test_cleanup_exception_is_reported_without_second_close():
    async def run():
        client = FakeNativeClient()
        client.close_error = RuntimeError("synthetic cleanup failure")
        proxy, relay, observation = await connect_pair(client)
        with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
            await proxy.close()
        with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
            await relay
        with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
            await proxy.close()
        assert client.close_calls == 1
        assert observation["status"] == "cleanup_failed"
        assert observation["cleanup_error"] == {
            "error_type": "RuntimeError", "error": "synthetic cleanup failure",
        }

    asyncio.run(run())

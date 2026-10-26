"""Native tool bridge contracts with synthetic worlds and real child processes.

No test loads an AppWorld task, runs a model, evaluates a benchmark task, or
claims CUGA graph admission. The child fixture only exercises process ownership
and the persistent native-tool transport boundary.
"""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from population_dynamics import appworld_benchmark_native_tools as native


_API_DOCS = {"sample": {name: {} for name in ("echo", "record", "increment")}}
_NATIVE_ERROR = "Execution failed. Traceback:\nRuntimeError: synthetic native failure"


class _World:
    def __init__(self):
        self.codes = []
        self.arguments = []
        self.count = 0
        self.apis = SimpleNamespace(sample=SimpleNamespace(
            echo=self.echo, record=self.record, increment=self.increment,
        ))

    def echo(self, value):
        self.arguments.append({"value": deepcopy(value)})
        return value

    def record(self, **arguments):
        self.arguments.append(deepcopy(arguments))
        return arguments

    def increment(self, amount=1):
        self.arguments.append({"amount": amount})
        self.count += amount
        return {"count": self.count}

    def execute(self, code):
        self.codes.append(code)
        output = StringIO()
        with redirect_stdout(output):
            exec(code, {"apis": self.apis})
        return output.getvalue()


class _OutputWorld:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.codes = []

    def execute(self, code):
        self.codes.append(code)
        result = self.outputs.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_source_quoting_and_token_parameters_reach_one_native_call_unchanged(tmp_path):
    world = _World()
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    marker = tmp_path / "injected"
    payload = (
        "\"); __import__('pathlib').Path(" + repr(str(marker))
        + ").write_text('executed'); #\n"
        + "quotes='\"', backslash=\\, unicode=你好"
    )
    arguments = {
        "payload": payload,
        "access_token": "literal-token",
        "file_system_access_token": "literal-file-token",
        "nested": {
            "access_token": ["true", "null", None, False],
            "reference": {"$ref": "previous_call.result"},
            "template": "{{history.previous.access_token}}",
        },
    }
    expected = deepcopy(arguments)
    assert dispatcher.dispatch("sample", "record", arguments) == expected
    assert world.arguments == [expected] and len(world.codes) == 1
    assert not marker.exists()
    attempt = dispatcher.attempts[0]
    assert attempt["arguments"] == expected and attempt["status"] == "returned"
    assert attempt["raw_output"] == json.dumps(expected, ensure_ascii=False) + "\n"
    arguments["nested"]["access_token"].append("later mutation")
    attempt["arguments"]["nested"]["access_token"].append("receipt mutation")
    assert dispatcher.attempts[0]["arguments"] == expected


def test_one_world_preserves_mutations_across_dispatches():
    world = _World()
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    assert dispatcher.dispatch("sample", "increment", {"amount": 2}) == {"count": 2}
    assert dispatcher.dispatch("sample", "increment", {"amount": 3}) == {"count": 5}
    assert world.count == 5 and len(world.codes) == 2
    assert [row["index"] for row in dispatcher.attempts] == [0, 1]


def test_api_error_dictionary_remains_returned_data():
    value = {"error": "invalid access token", "message": "request rejected", "success": False}
    world = _OutputWorld(json.dumps(value) + "\n")
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    assert dispatcher.dispatch("sample", "echo", {"value": value}) == value
    assert not dispatcher.blocked
    assert dispatcher.attempts[0]["status"] == "returned"


def test_native_execution_error_is_recorded_and_allows_the_next_call():
    world = _OutputWorld(_NATIVE_ERROR, "null\n")
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    with pytest.raises(native.NativeToolExecutionError, match="synthetic native failure"):
        dispatcher.dispatch("sample", "echo", {"value": "first"})
    assert dispatcher.attempts[0]["raw_output"] == _NATIVE_ERROR
    assert dispatcher.attempts[0]["status"] == "native_error"
    assert not dispatcher.blocked
    assert dispatcher.dispatch("sample", "echo", {"value": None}) is None
    assert len(world.codes) == 2


@pytest.mark.parametrize("raw", [
    None, b"null", {}, "", "not JSON", "{}\n{}", '{"value": 1, "value": 2}',
    '{"nested": {"value": 1, "value": 2}}', "NaN", "Infinity", "{}\ntrailing text",
])
def test_nontext_malformed_or_ambiguous_native_output_blocks_all_later_calls(raw):
    world = _OutputWorld(raw, "true")
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    with pytest.raises(native.UnknownNativeOutcome):
        dispatcher.dispatch("sample", "echo", {"value": "first"})
    assert dispatcher.blocked and len(world.codes) == 1
    attempt = dispatcher.attempts[0]
    assert attempt["status"] == "unknown" and attempt["native_wall_seconds"] >= 0
    if isinstance(raw, str):
        assert attempt["raw_output"] == raw
    with pytest.raises(native.UnknownNativeOutcome):
        dispatcher.dispatch("sample", "echo", {"value": "second"})
    assert len(world.codes) == len(dispatcher.attempts) == 1


def test_transport_exception_after_possible_execution_is_never_retried():
    world = _OutputWorld(TimeoutError("native transport ended after send"), "true")
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    with pytest.raises(native.UnknownNativeOutcome) as error:
        dispatcher.dispatch("sample", "increment", {"amount": 1})
    assert isinstance(error.value.__cause__, TimeoutError)
    assert dispatcher.blocked
    assert dispatcher.attempts[0]["error_type"] == "TimeoutError"
    with pytest.raises(native.UnknownNativeOutcome):
        dispatcher.dispatch("sample", "increment", {"amount": 1})
    assert len(world.codes) == 1


@pytest.mark.parametrize("app,api,arguments", [
    ("missing", "echo", {}),
    ("sample", "echo; injected()", {}),
    ("sample", "echo", []),
])
def test_rejected_local_inputs_do_not_execute_or_block_the_world(app, api, arguments):
    world = _World()
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    with pytest.raises((ValueError, TypeError)):
        dispatcher.dispatch(app, api, arguments)
    assert world.codes == [] and dispatcher.attempts == [] and not dispatcher.blocked
    assert dispatcher.dispatch("sample", "echo", {"value": True}) is True


def test_native_duration_uses_the_os_monotonic_clock(monkeypatch):
    readings = iter((10.0, 10.375))
    clock_ids = []

    def clock(clock_id):
        clock_ids.append(clock_id)
        return next(readings)

    monkeypatch.setattr(native.time, "clock_gettime", clock)
    dispatcher = native.NativeToolDispatcher(_OutputWorld("false\n"), _API_DOCS)
    assert dispatcher.dispatch("sample", "echo", {"value": False}) is False
    assert clock_ids == [native.time.CLOCK_MONOTONIC, native.time.CLOCK_MONOTONIC]
    assert dispatcher.attempts[0]["native_wall_seconds"] == 0.375


def _request(request_id, *, api="echo", arguments=None):
    return {
        "id": request_id, "operation": "call", "app": "sample", "api": api,
        "arguments": {"value": None} if arguments is None else arguments,
    }


def _serve(world, requests):
    dispatcher = native.NativeToolDispatcher(world, _API_DOCS)
    reader = StringIO("".join(json.dumps(row) + "\n" for row in requests))
    writer = StringIO()
    context = {"instruction": "synthetic public instruction", "api_docs": _API_DOCS}
    exit_code = native.serve_native_tools(dispatcher, context, reader, writer)
    return exit_code, [json.loads(line) for line in writer.getvalue().splitlines()], dispatcher


def test_server_sends_public_readiness_and_distinguishes_rejection_native_error_and_return():
    world = _OutputWorld(_NATIVE_ERROR, '{"value": "null"}\n')
    requests = [
        _request(1, api="missing"), _request(2), _request(3),
        {"id": 4, "operation": "close"},
    ]
    exit_code, responses, dispatcher = _serve(world, requests)
    assert exit_code == 0
    assert responses[0] == {
        "status": "ready",
        "public_context": {"instruction": "synthetic public instruction", "api_docs": _API_DOCS},
    }
    assert [row["status"] for row in responses[1:]] == ["rejected", "native_error", "returned", "closed"]
    assert [row["id"] for row in responses[1:]] == [1, 2, 3, 4]
    assert responses[3]["result"] == {"value": "null"}
    assert responses[3]["raw_output"] == '{"value": "null"}\n'
    assert len(world.codes) == len(dispatcher.attempts) == 2


def test_server_unknown_outcome_stops_before_the_next_queued_call():
    world = _OutputWorld("unframed output", "true")
    exit_code, responses, dispatcher = _serve(world, [_request(1), _request(2)])
    assert exit_code == 1 and responses[-1]["status"] == "unknown"
    assert len(responses) == 2 and len(world.codes) == 1 and dispatcher.blocked


@pytest.mark.parametrize("request_message", [
    _request(2), _request(True), {**_request(1), "operation": "retry"},
    {**_request(1), "unrecognized": True}, ["not", "an", "object"],
])
def test_server_protocol_errors_do_not_execute_native_calls(request_message):
    world = _World()
    exit_code, responses, dispatcher = _serve(world, [request_message])
    assert exit_code == 1 and responses[-1]["status"] == "protocol_error"
    assert world.codes == [] and dispatcher.attempts == []


def test_server_eof_is_aborted_without_inventing_a_close():
    exit_code, responses, dispatcher = _serve(_World(), [])
    assert exit_code == 1 and [row["status"] for row in responses] == ["ready"]
    assert dispatcher.attempts == []


_CHILD_PROGRAM = r'''
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from population_dynamics.appworld_benchmark_native_tools import NativeToolDispatcher, serve_native_tools

mode, journal_name = sys.argv[1:]
journal = Path(journal_name)
marker = journal.with_suffix(".mutated")
release = journal.with_suffix(".release")

def record(value):
    with journal.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()

if mode in {"eof", "malformed", "wrong_id"}:
    print(json.dumps({"status": "ready", "public_context": {"synthetic": True}}), flush=True)
    request = json.loads(sys.stdin.readline())
    record(request)
    if mode == "malformed":
        print("{malformed response", flush=True)
    elif mode == "wrong_id":
        print(json.dumps({"id": request["id"] + 1, "status": "returned", "result": True}), flush=True)
    sys.exit(0)

class World:
    def __init__(self):
        self.count = 0
        self.apis = SimpleNamespace(sample=SimpleNamespace(
            echo=self.echo, mutate=self.mutate, native_error=self.native_error,
        ))

    def echo(self, value):
        return value

    def mutate(self, token, wait_for_release=False):
        self.count += 1
        marker.write_text(str(self.count))
        if wait_for_release:
            while not release.exists():
                time.sleep(0.005)
        return {"count": self.count, "token": token}

    def native_error(self):
        raise RuntimeError("synthetic native failure")

    def execute(self, code):
        record({"code": code})
        output = StringIO()
        try:
            with redirect_stdout(output):
                exec(code, {"apis": self.apis})
        except RuntimeError as error:
            return "Execution failed. Traceback:\n" + str(error)
        return output.getvalue()

docs = {"sample": {name: {} for name in ("echo", "mutate", "native_error")}}
context = {"synthetic": True, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
if mode == "large":
    context["public_catalog_padding"] = "x" * 70000
sys.exit(serve_native_tools(NativeToolDispatcher(World(), docs), context, sys.stdin, sys.stdout))
'''


async def _start_client(tmp_path, mode="normal"):
    journal = tmp_path / "calls.jsonl"
    client = await native.NativeToolClient.start(
        [sys.executable, "-u", "-c", _CHILD_PROGRAM, mode, str(journal)],
        tmp_path / "child.stderr",
        timeout_seconds=10,
        shutdown_timeout_seconds=3,
    )
    return client, journal


def _journal_rows(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []


def _assert_reaped(client, cleanup):
    assert cleanup["child_reaped"] and client.process.returncode is not None
    assert cleanup["exit_code"] == client.process.returncode
    with pytest.raises(ProcessLookupError):
        os.kill(client.process.pid, 0)


def test_real_child_roundtrip_preserves_json_types_and_persistent_mutations(tmp_path):
    async def run():
        client, journal = await _start_client(tmp_path)
        values = [
            None, True, False, 0, 1.25, "null", "true", '{"value": 1}',
            [], [1, None, "null"], {"nested": {"access_token": "literal", "$ref": "opaque"}},
        ]
        try:
            assert client.public_context == {"synthetic": True, "visible_devices": ""}
            for value in values:
                result = await client("sample", "echo", {"value": value})
                assert result == value and type(result) is type(value)
            assert await client("sample", "mutate", {"token": "first"}) == {"count": 1, "token": "first"}
            assert await client("sample", "mutate", {"token": "second"}) == {"count": 2, "token": "second"}
        finally:
            cleanup = await client.close()
        _assert_reaped(client, cleanup)
        assert cleanup["exit_code"] == 0 and cleanup["termination_signals"] == []
        assert not cleanup["request_outcome_unresolved"]
        assert len(_journal_rows(journal)) == len(values) + 2
        assert [row["id"] for row in client.calls] == list(range(1, len(values) + 3))
        assert await client.close() == cleanup

    asyncio.run(run())


def test_real_child_large_public_context_and_result_are_not_truncated(tmp_path):
    async def run():
        client, journal = await _start_client(tmp_path, "large")
        value = {"text": "你好" * 40000, "literal": "null"}
        try:
            assert len(client.public_context["public_catalog_padding"]) == 70000
            assert await client("sample", "echo", {"value": value}) == value
        finally:
            cleanup = await client.close()
        _assert_reaped(client, cleanup)
        assert cleanup["exit_code"] == 0 and len(_journal_rows(journal)) == 1

    asyncio.run(run())


def test_real_child_rejected_and_native_error_calls_do_not_block_resolved_followup(tmp_path):
    async def run():
        client, journal = await _start_client(tmp_path)
        try:
            with pytest.raises(ValueError, match="absent"):
                await client("sample", "missing", {})
            with pytest.raises(native.NativeToolExecutionError, match="synthetic native failure"):
                await client("sample", "native_error", {})
            value = {"error": "API failure as data", "success": False}
            assert await client("sample", "echo", {"value": value}) == value
        finally:
            cleanup = await client.close()
        _assert_reaped(client, cleanup)
        assert cleanup["exit_code"] == 0 and not cleanup["request_outcome_unresolved"]
        assert [row["status"] for row in client.calls] == ["rejected", "native_error", "returned"]
        assert len(_journal_rows(journal)) == 2

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["eof", "malformed", "wrong_id"])
def test_real_child_unresolved_response_blocks_later_calls_and_reaps_without_retry(tmp_path, mode):
    async def run():
        client, journal = await _start_client(tmp_path, mode)
        try:
            with pytest.raises(native.UnknownNativeOutcome):
                await client("sample", "mutate", {"token": "first"})
            with pytest.raises(native.UnknownNativeOutcome):
                await client("sample", "mutate", {"token": "second"})
        finally:
            cleanup = await client.close()
        _assert_reaped(client, cleanup)
        assert cleanup["request_outcome_unresolved"]
        assert len(_journal_rows(journal)) == len(client.calls) == 1
        assert client.calls[0]["status"] == "unknown"

    asyncio.run(run())


@pytest.mark.parametrize("timeout_source", ["caller", "client"])
def test_pending_mutation_after_timeout_is_not_retried_and_late_result_is_retained(tmp_path, timeout_source):
    async def run():
        client, journal = await _start_client(tmp_path)
        marker, release = journal.with_suffix(".mutated"), journal.with_suffix(".release")
        if timeout_source == "client":
            client.timeout_seconds = 0.2

        async def wait_for_mutation():
            while not marker.exists():
                await asyncio.sleep(0.005)

        pending = asyncio.create_task(client(
            "sample", "mutate", {"token": "one mutation", "wait_for_release": True},
        ))
        try:
            await asyncio.wait_for(wait_for_mutation(), 5)
            assert marker.read_text() == "1"
            if timeout_source == "caller":
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(pending, 0.01)
            else:
                with pytest.raises(native.UnknownNativeOutcome):
                    await pending
            with pytest.raises(native.UnknownNativeOutcome):
                await client("sample", "mutate", {"token": "must not execute"})
            assert len(_journal_rows(journal)) == 1
        finally:
            release.write_text("release owned pending action")
            cleanup = await client.close()
            await asyncio.gather(pending, return_exceptions=True)
        _assert_reaped(client, cleanup)
        assert cleanup["request_outcome_unresolved"] and cleanup["pending_error"] is None
        assert cleanup["late_response"]["id"] == 1
        assert cleanup["late_response"]["status"] == "returned"
        assert cleanup["late_response"]["result"] == {"count": 1, "token": "one mutation"}
        assert cleanup["termination_signals"] == []
        assert len(_journal_rows(journal)) == len(client.calls) == 1
        assert client.calls[0]["status"] == "unknown"
        assert await client.close() == cleanup

    asyncio.run(run())


@pytest.mark.parametrize("split", ["test_normal", "test_challenge"])
def test_native_server_rejects_protected_test_splits_before_loading_any_task(tmp_path, monkeypatch, split):
    monkeypatch.setattr(native, "_load_native", lambda: pytest.fail("native task loading is forbidden"))
    config = {
        "schema": "appworld_benchmark_v1",
        "splits": {"development": ["train", "dev"]},
        "formal": {"released": True},
    }
    manifest = {"schema": "appworld_split_membership_v1"}
    with pytest.raises(ValueError, match="restricted to train/dev"):
        native.run_native_tool_server(
            config, manifest, split=split, task_id="synthetic_protected_1", seed=41,
            source_root=tmp_path, run_root=tmp_path / "must_not_exist",
            reader=StringIO(), writer=StringIO(),
        )
    assert not (tmp_path / "must_not_exist").exists()


@pytest.mark.parametrize("task_datetime", [
    datetime.fromisoformat("2023-05-17T08:15:00"),
    datetime.fromisoformat("2023-05-17T08:15:00.123456+05:45"),
])
def test_public_context_reads_only_explicit_public_task_fields(task_datetime):
    accesses = []
    function_schema = [{"type": "function", "function": {"name": "sample__echo"}}]

    class Docs(dict):
        def function_calling(self):
            accesses.append("api_docs.function_calling")
            return deepcopy(function_schema)

    docs = Docs({"sample": {"echo": {"description": "synthetic API", "parameters": []}}})
    public = {
        "instruction": "synthetic task only",
        "datetime": task_datetime,
        "supervisor": {
            "first_name": "Synthetic", "last_name": "User",
            "email": "synthetic@example.invalid", "phone_number": "000",
            "password": "must not appear",
        },
        "app_descriptions": {"sample": "synthetic application"},
        "api_docs": docs,
    }

    class PublicOnlyTask:
        def __getattribute__(self, name):
            if name not in public:
                raise AssertionError(f"forbidden task field: {name}")
            accesses.append(name)
            return public[name]

    result = native.public_task_context(PublicOnlyTask())
    assert set(result) == {
        "instruction", "supervisor", "current_datetime", "app_descriptions", "api_docs", "function_calling",
    }
    assert result["current_datetime"] == task_datetime.isoformat()
    assert result["supervisor"] == {key: public["supervisor"][key] for key in (
        "first_name", "last_name", "email", "phone_number",
    )}
    assert result["function_calling"] == function_schema
    assert set(accesses) == set(public) | {"api_docs.function_calling"}
    result["api_docs"]["sample"]["echo"]["description"] = "changed copy"
    assert docs["sample"]["echo"]["description"] == "synthetic API"



@pytest.mark.parametrize("date_fields", [{}, {"datetime": None}, {"datetime": "2023-05-17"}])
def test_public_context_requires_native_task_datetime_without_host_clock_fallback(date_fields):
    task = SimpleNamespace(
        instruction="synthetic task only",
        supervisor=dict(first_name="Synthetic", last_name="User", email="synthetic@example.invalid", phone_number="000"),
        api_docs={}, app_descriptions={}, **date_fields,
    )
    with pytest.raises(AttributeError):
        native.public_task_context(task)

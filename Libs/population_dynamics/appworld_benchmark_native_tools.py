"""Persistent native AppWorld tools across the CUGA environment boundary.

The server owns one admitted train/dev world. The async client supplies the
external-tool callable; it does not authenticate, retry or score. Cancellation
leaves the outstanding exchange owned until close reaps the child. A session
receipt is not a scored agent attempt or evidence of full CUGA graph admission.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import json
import keyword
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, TextIO

from .appworld_benchmark_collaboration import (
    UnknownNativeOutcome, _NATIVE_EXECUTION_ERROR_PREFIX, _json_copy,
)
from .appworld_benchmark_development import _admit_source, load_benchmark_config
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, validate_split_manifest
from .appworld_benchmark_runner import (
    _check_execution_location, _load_native, _verify_memberships, _write_receipt,
)


def _strict_json(text: str) -> Any:
    def object_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON object key: " + key)
            result[key] = value
        return result
    return _json_copy(json.loads(text, object_pairs_hook=object_pairs))


class NativeToolExecutionError(RuntimeError):
    """AppWorld returned its native execution traceback after executing once."""


class NativeToolDispatcher:
    """Dispatch raw arguments without the collaboration result-reference syntax."""

    def __init__(self, world: Any, api_docs: Mapping[str, Mapping[str, Any]]):
        if not callable(getattr(world, "execute", None)):
            raise TypeError("an admitted persistent native world is required")
        self.world = world
        self.identities = set()
        for app, apis in api_docs.items():
            if not isinstance(apis, Mapping):
                raise ValueError("public API documents must be grouped by app")
            for api in apis:
                for name in (app, api):
                    if (not isinstance(name, str) or not name.isascii()
                            or not name.isidentifier() or keyword.iskeyword(name)):
                        raise ValueError("public app/API names must be Python identifiers")
                self.identities.add((app, api))
        if not self.identities:
            raise ValueError("a nonempty public native API catalog is required")
        self.blocked = False
        self._attempts: list[dict[str, Any]] = []

    @property
    def attempts(self) -> list[dict[str, Any]]:
        return deepcopy(self._attempts)

    def dispatch(self, app: str, api: str, arguments: dict[str, Any]) -> Any:
        if self.blocked:
            raise UnknownNativeOutcome("a preceding native result remains unresolved")
        if not isinstance(app, str) or not isinstance(api, str) or (app, api) not in self.identities:
            raise ValueError("API is absent from the current public native catalog")
        if not isinstance(arguments, dict):
            raise TypeError("native tool arguments must be a dictionary")
        arguments = _json_copy(arguments)
        literal = repr(json.dumps(arguments, ensure_ascii=False, allow_nan=False))
        code = ("import json\n"
                f"print(json.dumps(apis.{app}.{api}(**json.loads({literal})), ensure_ascii=False))")
        row = {"index": len(self._attempts), "app": app, "api": api, "arguments": arguments,
               "status": "executing", "raw_output": None}
        self._attempts.append(row)
        start = time.clock_gettime(time.CLOCK_MONOTONIC)
        try:
            raw = self.world.execute(code)
            row["raw_output"] = raw if isinstance(raw, str) else None
            if not isinstance(raw, str):
                raise UnknownNativeOutcome("native execute returned a non-text output")
            if raw.startswith(_NATIVE_EXECUTION_ERROR_PREFIX):
                row["status"] = "native_error"
                raise NativeToolExecutionError(raw)
            try:
                result = _strict_json(raw)
            except (ValueError, TypeError) as error:
                raise UnknownNativeOutcome("native return value could not be decoded without coercion") from error
            row.update(status="returned", returned_type=type(result).__name__)
            return result
        except NativeToolExecutionError:
            raise
        except BaseException as error:
            self.blocked = True
            row.update(status="unknown", error_type=type(error).__name__, error=str(error))
            if isinstance(error, (KeyboardInterrupt, SystemExit, UnknownNativeOutcome)):
                raise
            raise UnknownNativeOutcome("native operation result is unresolved") from error
        finally:
            row["native_wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - start


def public_task_context(task: Any) -> dict[str, Any]:
    """Read the explicit public fields; never traverse the Task object."""
    supervisor = task.supervisor
    fields = ("first_name", "last_name", "email", "phone_number")
    user = {key: supervisor[key] if isinstance(supervisor, Mapping) else getattr(supervisor, key)
            for key in fields}
    docs = task.api_docs
    return _json_copy({
        "instruction": task.instruction, "supervisor": user,
        "current_datetime": task.datetime.isoformat(),
        "app_descriptions": deepcopy(task.app_descriptions),
        "api_docs": {app: {api: deepcopy(dict(doc)) for api, doc in apis.items()}
                     for app, apis in docs.items()},
        "function_calling": deepcopy(docs.function_calling()),
    })


def serve_native_tools(
    dispatcher: NativeToolDispatcher, public_context: Mapping[str, Any],
    reader: TextIO, writer: TextIO, *, on_progress=None,
) -> int:
    """Serve sequential calls in the current world; EOF is an aborted session."""
    def emit(message):
        writer.write(json.dumps(_json_copy(message), ensure_ascii=False, allow_nan=False) + "\n")
        writer.flush()

    emit({"status": "ready", "public_context": _json_copy(dict(public_context))})
    expected_id = 1
    for line in reader:
        request = None
        try:
            request = _strict_json(line)
            if not isinstance(request, dict) or type(request.get("id")) is not int or request["id"] != expected_id:
                raise ValueError("native request IDs must increase once from 1")
            operation = request.get("operation")
            fields = {"id", "operation"} if operation == "close" else {"id", "operation", "app", "api", "arguments"}
            if operation not in {"call", "close"} or set(request) != fields:
                raise ValueError("unknown native operation or request fields")
        except (ValueError, TypeError) as error:
            emit({"id": request.get("id") if isinstance(request, dict) else None,
                  "status": "protocol_error", "error": str(error)})
            return 1
        if operation == "close":
            emit({"id": expected_id, "status": "closed"})
            return 0
        try:
            result = dispatcher.dispatch(request["app"], request["api"], request["arguments"])
            response = {"id": expected_id, "status": "returned", "result": result,
                        "raw_output": dispatcher.attempts[-1]["raw_output"]}
        except NativeToolExecutionError as error:
            response = {"id": expected_id, "status": "native_error", "error": str(error)}
        except UnknownNativeOutcome as error:
            response = {"id": expected_id, "status": "unknown", "error": str(error)}
        except (ValueError, TypeError) as error:
            response = {"id": expected_id, "status": "rejected", "error": str(error)}
        finally:
            if on_progress is not None:
                on_progress(dispatcher.attempts)
        emit(response)
        if response["status"] == "unknown":
            return 1
        expected_id += 1
    return 1


class NativeToolClient:
    """One async exchange at a time with an owned native child process."""

    def __init__(self, process, stderr_stream, timeout_seconds, shutdown_timeout_seconds):
        self.process = process
        self._stderr = stderr_stream
        self.timeout_seconds = timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.public_context: dict[str, Any] | None = None
        self.calls: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._buffer = bytearray()
        self._next_id = 1
        self._pending: asyncio.Task | None = None
        self._uncertain = False
        self._closed = False
        self._cleanup: dict[str, Any] | None = None

    @classmethod
    async def start(
        cls, command: Sequence[str], stderr_path: str | Path, *,
        timeout_seconds: float = 600, shutdown_timeout_seconds: float = 30,
        env: Mapping[str, str] | None = None,
    ) -> NativeToolClient:
        if (isinstance(command, (str, bytes)) or not command
                or any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command)):
            raise ValueError("an explicit native child command is required")
        for limit in (timeout_seconds, shutdown_timeout_seconds):
            if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
                raise ValueError("native communication timeouts must be positive and finite")
        environment = dict(os.environ if env is None else env)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        stream = Path(stderr_path).open("x", encoding="utf-8")
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=stream, start_new_session=True, env=environment,
            )
        except BaseException:
            stream.close()
            raise
        client = cls(process, stream, timeout_seconds, shutdown_timeout_seconds)
        try:
            ready = await asyncio.wait_for(client._read_message(), timeout_seconds)
            if ready.get("status") != "ready" or not isinstance(ready.get("public_context"), dict):
                raise UnknownNativeOutcome("native child did not return public task readiness")
            client.public_context = ready["public_context"]
        except BaseException:
            client._uncertain = True
            await client.close()
            raise
        return client

    async def _read_message(self) -> dict[str, Any]:
        # read() avoids StreamReader.readline's default 64 KiB limit, which can
        # truncate complete official API catalogs and ordinary tool results.
        while b"\n" not in self._buffer:
            chunk = await self.process.stdout.read(65536)
            if not chunk:
                raise EOFError("native child stdout closed before a complete response")
            self._buffer.extend(chunk)
        line, _, tail = self._buffer.partition(b"\n")
        self._buffer = bytearray(tail)
        response = _strict_json(line.decode("utf-8"))
        if not isinstance(response, dict):
            raise ValueError("native response must be an object")
        return response

    async def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        self.process.stdin.write((json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        await self.process.stdin.drain()
        response = await self._read_message()
        if type(response.get("id")) is not int or response["id"] != request["id"]:
            raise UnknownNativeOutcome("native response identity differs from the outstanding request")
        return response

    async def __call__(self, app: str, api: str, arguments: dict[str, Any]) -> Any:
        if not isinstance(arguments, dict):
            raise TypeError("native tool arguments must be a dictionary")
        arguments = _json_copy(arguments)
        async with self._lock:
            if self._closed or self._uncertain:
                raise UnknownNativeOutcome("native client is closed or has an unresolved exchange")
            request = {"id": self._next_id, "operation": "call", "app": app, "api": api, "arguments": arguments}
            _json_copy(request)  # Check the complete request before any bytes are sent.
            self._next_id += 1
            row = {"id": request["id"], "app": app, "api": api, "arguments": arguments, "status": "pending"}
            self.calls.append(row)
            start = time.clock_gettime(time.CLOCK_MONOTONIC)
            self._pending = asyncio.create_task(self._exchange(request))
            try:
                response = await asyncio.wait_for(asyncio.shield(self._pending), self.timeout_seconds)
            except BaseException as error:
                self._uncertain = True
                row.update(status="unknown", error_type=type(error).__name__, error=str(error))
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                raise UnknownNativeOutcome("native response is unresolved; action was not repeated") from error
            finally:
                row["wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - start
            self._pending = None
            row["response"] = deepcopy(response)
            row["status"] = response.get("status")
            if response.get("status") == "returned" and "result" in response:
                return _json_copy(response["result"])
            if response.get("status") == "native_error":
                raise NativeToolExecutionError(response.get("error", "native execution failed"))
            if response.get("status") == "rejected":
                raise ValueError(response.get("error", "native request was rejected"))
            self._uncertain = True
            raise UnknownNativeOutcome(response.get("error", "native response has no resolved return value"))

    async def close(self) -> dict[str, Any]:
        """Reap the child, retaining a late response after timeout/cancellation."""
        if self._cleanup is not None:
            return deepcopy(self._cleanup)
        self._closed = True
        pending_response = None
        pending_error = None
        signals = []
        try:
            if self._pending is not None:
                try:
                    pending_response = await asyncio.wait_for(
                        asyncio.shield(self._pending), self.shutdown_timeout_seconds)
                except (Exception, asyncio.CancelledError) as error:
                    self._uncertain = True
                    pending_error = {"error_type": type(error).__name__, "error": str(error)}
            if (not self._uncertain and self.public_context is not None
                    and self.process.returncode is None):
                try:
                    response = await asyncio.wait_for(self._exchange(
                        {"id": self._next_id, "operation": "close"}), self.shutdown_timeout_seconds)
                    if response.get("status") != "closed":
                        raise ValueError("native close was not acknowledged")
                except Exception as error:
                    self._uncertain = True
                    pending_error = {"error_type": type(error).__name__, "error": str(error)}
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), self.shutdown_timeout_seconds)
            except asyncio.TimeoutError:
                signals.append("SIGTERM")
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.process.wait(), self.shutdown_timeout_seconds)
                except asyncio.TimeoutError:
                    signals.append("SIGKILL")
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await asyncio.wait_for(self.process.wait(), self.shutdown_timeout_seconds)
            if self._pending is not None:
                # After child reaping, consume the owned read task's terminal
                # result. Never spawn a replacement operation from this result.
                results = await asyncio.wait_for(
                    asyncio.gather(self._pending, return_exceptions=True), self.shutdown_timeout_seconds)
                if pending_response is None and isinstance(results[0], dict):
                    pending_response = results[0]
            self._cleanup = {
                "pid": self.process.pid, "child_reaped": self.process.returncode is not None,
                "exit_code": self.process.returncode, "termination_signals": signals,
                "request_outcome_unresolved": self._uncertain,
                "late_response": pending_response, "pending_error": pending_error,
            }
            return deepcopy(self._cleanup)
        finally:
            self._stderr.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, error_type, error, traceback):
        await self.close()


def run_native_tool_server(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, task_id: str,
    seed: int, source_root: str | Path, run_root: str | Path,
    reader: TextIO, writer: TextIO,
) -> dict[str, Any]:
    """Own a fresh development world, without loading any agent or evaluator."""
    if config.get("schema") != "appworld_benchmark_v1" or manifest.get("schema") != "appworld_split_membership_v1":
        raise ValueError("canonical benchmark config and native split manifest are required")
    if split not in {"train", "dev"} or split not in config.get("splits", {}).get("development", ()):
        raise ValueError("native tool server is restricted to train/dev")
    if tuple(config.get("seeds", ())) != EXPERIMENT_SEEDS or any(type(s) is not int for s in config["seeds"]):
        raise ValueError("canonical experiment seeds must be exactly [41, 42, 43]")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("native attempt seed must be 41, 42 or 43")
    if task_id not in validate_split_manifest(manifest).get(split, ()):
        raise ValueError("native task is absent from admitted development membership")
    development = config["development_execution"]
    if development.get("status") != "provisional":
        raise ValueError("this native tool server requires provisional development settings")
    kwargs = deepcopy(dict(development["appworld_config"]))
    if (kwargs.get("raise_on_extra_parameters") is not True
            or kwargs.get("load_ground_truth", False) is not False
            or kwargs.get("ground_truth_mode") is not None
            or any(value is not None for key, value in kwargs.items() if key.startswith("remote_"))):
        raise ValueError("native extra-parameter checks, ground-truth exclusion and local execution are required")
    if "random_seed" in kwargs and kwargs["random_seed"] != seed:
        raise ValueError("native random_seed conflicts with the attempt seed")
    kwargs.update(load_ground_truth=False, random_seed=seed)
    source = _admit_source(config, Path(source_root).resolve(strict=True))
    root = Path(run_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("a native tool session must use a fresh run root")
    _check_execution_location(root)
    data_root = Path(manifest["data_root"]).resolve(strict=True)
    data = (data_root / "data").resolve(strict=True)
    if not data.is_dir():
        raise ValueError("native data_root/data is absent")
    root.mkdir()
    receipt_path = root / "native_session.json"
    report = {
        "schema": "appworld_native_tool_session_v1", "scope": "development_tools_only",
        "formal_eligible": False, "split": split, "task_id": task_id, "seed": seed,
        "run_root": str(root), "data_root": str(data_root), "experiment_name": "development",
        "appworld_version": manifest["appworld_version"], "source": source, "native_kwargs": kwargs,
        "status": "initializing", "terminal": False, "attempts": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "native_world_closed": False,
        "agent_run_qualified": False, "scoring_status": "not_run",
    }
    _write_receipt(receipt_path, report)
    saved = {key: os.environ.get(key) for key in ("APPWORLD_ROOT", "APPWORLD_CACHE")}
    native = None
    start = time.clock_gettime(time.CLOCK_MONOTONIC)
    try:
        (root / "data").symlink_to(data, target_is_directory=True)
        os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / ".cache"))
        native = _load_native()
        native.path_store.update_root(str(root))
        native.path_store.reload()
        _verify_memberships(native, manifest, data)
        if native.AppWorld.id_to_time_freezer:
            raise ValueError("another native world is active")
        with native.AppWorld(task_id=task_id, experiment_name="development", **kwargs) as world:
            dispatcher = NativeToolDispatcher(world, world.task.api_docs)
            context = public_task_context(world.task)
            report.update(status="serving", membership_verified=True)
            _write_receipt(receipt_path, report)

            def progress(attempts):
                report["attempts"] = attempts
                _write_receipt(receipt_path, report)

            result = serve_native_tools(dispatcher, context, reader, writer, on_progress=progress)
            report.update(status="closed" if result == 0 else "aborted", server_exit_code=result)
        report["native_world_closed"] = True
    except BaseException as error:
        report.update(status="unknown", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if native is not None:
            native.path_store.reload()
        # reload() can modify environment variables, so restore the caller again.
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report.update(terminal=True, ended_at=datetime.now(timezone.utc).isoformat(),
                      native_session_wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - start)
        _write_receipt(receipt_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    args = parser.parse_args(argv)
    protocol_output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            report = run_native_tool_server(
                load_benchmark_config(args.config), _strict_json(args.split_manifest.read_text()),
                split=args.split, task_id=args.task_id, seed=args.seed, source_root=args.source_root,
                run_root=args.run_root, reader=sys.stdin, writer=protocol_output,
            )
        return 0 if report["status"] == "closed" else 1
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the complete CUGA graph behind a filesystem and process boundary.

The parent owns native AppWorld execution. Only a dedicated socket descriptor,
public source/runtime mounts, and one fresh writable sandbox enter the worker.
A completed owner receipt describes execution closure, not AppWorld task success.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata as metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from population_dynamics.appworld_benchmark_cuga_channel import (
    NativeToolProxy,
    relay_native_tools,
)

PROCESS_TERMINATE_GRACE_SECONDS = 10.0
PROCESS_KILL_REAP_GRACE_SECONDS = 10.0
WORKER_CONFIG_PATH = Path("/work/worker_config.json")
GRAPH_OBSERVATION_PATH = Path("/work/graph_observation.json")
REQUIRED_PACKAGES = (
    "cuga",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "langgraph",
    "fastmcp",
    "torch",
    "browsergym-core",
)


@dataclass(frozen=True)
class _WorkerLaunch:
    command: tuple[str, ...]
    environment: dict[str, str]
    stdout_path: Path
    stderr_path: Path
    graph_observation_path: Path
    observation_path: Path


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _read_json(path: Path) -> Any:
    # The worker controls its output directory. Do not follow a worker-created
    # symlink into native data when the outside owner consumes that output.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Worker JSON evidence must be a regular file")
        return json.load(handle, parse_constant=_reject_json_constant)


def _local_model_url(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("base_url must name a local HTTP endpoint")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
    ):
        raise ValueError("base_url must be a local HTTP endpoint without credentials")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    else:
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("base_url must use a loopback address")
        except ValueError as error:
            raise ValueError("base_url must use localhost or a loopback address") from error
    port = parsed.port  # Validate its spelling and range.
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def _package_spec(expected: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(expected, Mapping):
        raise TypeError("expected_packages must map distribution names to versions")
    result: dict[str, str] = {}
    for name, version in expected.items():
        if not isinstance(name, str) or not isinstance(version, str):
            raise TypeError("Package names and versions must be strings")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", canonical) or not version.strip():
            raise ValueError("Package names and versions must be nonempty identities")
        if canonical in result:
            raise ValueError(f"Duplicate package identity: {canonical}")
        result[canonical] = version
    missing = sorted(set(REQUIRED_PACKAGES) - result.keys())
    if missing:
        raise ValueError(f"Missing required package identities: {', '.join(missing)}")
    if result["cuga"] != "0.3.2":
        raise ValueError("The qualified CUGA distribution identity must be 0.3.2")
    return result


def _verify_expected_packages(
    expected: Mapping[str, str], observation: dict[str, Any]
) -> None:
    versions = _package_spec(expected)
    observation.update(status="checking", expected=versions, actual={}, mismatches=[])
    for name, wanted in versions.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        observation["actual"][name] = actual
        if actual != wanted:
            observation["mismatches"].append(name)
    if observation["mismatches"]:
        observation["status"] = "failed"
        raise RuntimeError(
            "Installed package identities differ: "
            + ", ".join(observation["mismatches"])
        )
    observation["status"] = "passed"


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _check_mounts(
    sources: Sequence[Path], protected_roots: Sequence[Path]
) -> None:
    """Reject both ancestors and descendants of a protected path."""
    protected = tuple(Path(path).resolve(strict=False) for path in protected_roots)
    for candidate in sources:
        source = Path(candidate).resolve(strict=False)
        for protected_root in protected:
            if _overlaps(source, protected_root):
                raise ValueError(
                    f"Mount source overlaps a protected path: {source} / {protected_root}"
                )


def _directory(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _python_runtime(
    cuga_python: str | os.PathLike[str],
    python_base_prefix: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    executable = Path(os.path.abspath(cuga_python))
    if executable.parent.name != "bin":
        raise ValueError("cuga_python must be an explicit interpreter in a prefix/bin")
    prefix = _directory(executable.parent.parent, "Python prefix")
    executable = prefix / "bin" / executable.name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Python interpreter is not executable: {executable}")
    base_prefix = _directory(python_base_prefix, "Python base prefix")
    if prefix == Path("/") or base_prefix == Path("/"):
        raise ValueError("A runtime prefix cannot expose the host filesystem root")
    return executable, prefix, base_prefix


def _python_runtime_alias(
    executable: Path, prefix: Path, base_prefix: Path,
) -> Path | None:
    """Retain an image Python alias without exposing any additional host files."""
    current = executable
    seen: set[Path] = set()
    while current.is_symlink():
        if current in seen:
            raise ValueError("Python interpreter symlink cycle")
        seen.add(current)
        link = Path(os.readlink(current))
        current = Path(os.path.abspath(link if link.is_absolute() else current.parent / link))
        if current.is_relative_to(prefix) or current.is_relative_to(base_prefix):
            continue
        alias = current.parent.parent
        if (
            current.parent.name != "bin"
            or alias.parent != base_prefix.parent
            or not alias.is_symlink()
            or alias.resolve(strict=True) != base_prefix
            or current.resolve(strict=True)
            != (base_prefix / "bin" / current.name).resolve(strict=True)
        ):
            raise ValueError("Python interpreter link leaves the admitted runtime")
        return alias
    if not (current.is_relative_to(prefix) or current.is_relative_to(base_prefix)):
        raise ValueError("Python interpreter leaves the admitted runtime")
    return None


def _system_mounts() -> list[tuple[Path, str]]:
    candidates = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc/ld.so.cache")
    return [(Path(path), path) for path in candidates if Path(path).exists()]


def _bubblewrap_command(
    *,
    executable: Path,
    prefix: Path,
    base_prefix: Path,
    application_lib: Path,
    sandbox: Path,
    owner_root: Path,
    protected_roots: Sequence[Path],
    worker_arguments: Sequence[str],
) -> tuple[str, ...]:
    bubblewrap = shutil.which("bwrap", path="/usr/bin:/bin")
    if bubblewrap is None:
        raise FileNotFoundError("bubblewrap is not available in /usr/bin or /bin")
    mounts = _system_mounts()
    mounts.extend(((base_prefix, str(base_prefix)), (prefix, str(prefix))))
    mounts.append((application_lib, "/app"))
    _check_mounts(
        [source for source, _ in mounts],
        [*protected_roots, owner_root],
    )
    _check_mounts([sandbox], protected_roots)
    command = [
        bubblewrap,
        "--die-with-parent",
        "--unshare-all",
        "--share-net",
        "--cap-drop",
        "ALL",
        "--tmpfs", "/tmp",
    ]
    seen: set[tuple[str, str]] = set()
    for source, destination in mounts:
        identity = (str(source), destination)
        if identity not in seen:
            command.extend(("--ro-bind", str(source), destination))
            seen.add(identity)
    alias = _python_runtime_alias(executable, prefix, base_prefix)
    if alias is not None:
        command.extend(("--symlink", str(base_prefix), str(alias)))
    command.extend(
        (
            "--bind", str(sandbox), "/work",
            # Keep procfs absent: a host /proc bind would expose owner processes.
            "--dir", "/proc",
            "--dev", "/dev",
            "--chdir", "/work",
            "--",
            str(executable),
            *worker_arguments,
        )
    )
    return tuple(command)


def _worker_environment(
    *, prefix: Path, base_prefix: Path, base_url: str
) -> dict[str, str]:
    # No inherited environment, HOME/AGENT_HOME replacement, proxies or secrets.
    return {
        "PATH": f"{prefix}/bin:{base_prefix}/bin:/usr/bin:/bin",
        "PYTHONPATH": "/app",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "CUGA_LOGGING_DIR": "/work/logs",
        "CUGA_DBS_DIR": "/work/dbs",
        "FASTEMBED_CACHE_PATH": "/work/cache/fastembed",
        "XDG_CACHE_HOME": "/work/cache",
        "HF_HOME": "/work/cache/huggingface",
        "TMPDIR": "/work/tmp",
        "OPENAI_API_KEY": "local-model",
        "OPENAI_BASE_URL": base_url,
    }


def _validate_task(
    *, task_id: str, seed: int, model_name: str, base_url: str
) -> str:
    if not isinstance(task_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id
    ):
        raise ValueError("task_id must be one source task identifier")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (41, 42, 43):
        raise ValueError("seed must be one of 41, 42, 43")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be nonempty")
    return _local_model_url(base_url)


def _validate_timeout(timeout_seconds: float | None) -> None:
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number or None")


def _prepare_cuga_launch(
    channel_fd: int,
    *,
    cuga_python: str | os.PathLike[str],
    python_base_prefix: str | os.PathLike[str],
    application_lib: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    task_id: str,
    seed: int,
    model_name: str,
    base_url: str,
    expected_packages: Mapping[str, str],
    protected_roots: Sequence[str | os.PathLike[str]],
    observation: dict[str, Any],
) -> _WorkerLaunch:
    local_url = _validate_task(
        task_id=task_id, seed=seed, model_name=model_name, base_url=base_url
    )
    packages = _package_spec(expected_packages)
    executable, prefix, base_prefix = _python_runtime(cuga_python, python_base_prefix)
    source_lib = _directory(application_lib, "Application library")
    raw_root = Path(output_root)
    if raw_root.is_symlink():
        raise ValueError("output_root cannot be a symlink")
    root = raw_root.resolve(strict=False)
    if root.exists():
        raise FileExistsError(f"output_root must be fresh: {root}")
    sandbox = root / "sandbox"
    protected = tuple(Path(path).resolve(strict=False) for path in protected_roots)
    if not protected:
        raise ValueError("Explicit native data/task/scoring protected_roots are required")
    command = _bubblewrap_command(
        executable=executable,
        prefix=prefix,
        base_prefix=base_prefix,
        application_lib=source_lib,
        sandbox=sandbox,
        owner_root=root,
        protected_roots=protected,
        worker_arguments=(
            "-m", "population_dynamics.appworld_benchmark_cuga_process",
            "worker", "--channel-fd", str(channel_fd),
        ),
    )
    # Validate the complete mount plan before creating any output directory.
    root.mkdir(parents=True, exist_ok=False)
    sandbox.mkdir()
    for directory in ("logs", "dbs", "cache/fastembed", "cache/huggingface", "tmp"):
        (sandbox / directory).mkdir(parents=True, exist_ok=True)
    runtime = {
        "executable": str(executable),
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
    }
    _write_json(
        sandbox / WORKER_CONFIG_PATH.name,
        {
            "task_id": task_id,
            "seed": seed,
            "model_name": model_name,
            "base_url": local_url,
            "expected_packages": packages,
            "python_runtime": runtime,
        },
    )
    launch = _WorkerLaunch(
        command=command,
        environment=_worker_environment(
            prefix=prefix, base_prefix=base_prefix, base_url=local_url
        ),
        stdout_path=root / "worker.stdout.log",
        stderr_path=root / "worker.stderr.log",
        graph_observation_path=sandbox / GRAPH_OBSERVATION_PATH.name,
        observation_path=root / "process_observation.json",
    )
    observation["python_runtime"] = runtime
    observation["paths"] = {
        "output_root": str(root),
        "sandbox": str(sandbox),
        "stdout": str(launch.stdout_path),
        "stderr": str(launch.stderr_path),
        "graph_observation": str(launch.graph_observation_path),
        "process_observation": str(launch.observation_path),
    }
    return launch


async def _reap_worker(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    observation: dict[str, Any],
) -> None:
    for sent_signal, grace in (
        (signal.SIGTERM, PROCESS_TERMINATE_GRACE_SECONDS),
        (signal.SIGKILL, PROCESS_KILL_REAP_GRACE_SECONDS),
    ):
        if process.returncode is not None:
            break
        try:
            os.killpg(process.pid, sent_signal)
        except ProcessLookupError:
            pass
        except OSError as error:
            observation.setdefault("signal_errors", []).append(_error_record(error))
        else:
            observation["termination_signals"].append(sent_signal.name)
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=grace)
        except asyncio.TimeoutError:
            continue
        else:
            break
    if process.returncode is not None:
        observation["exit_code"] = await wait_task
        observation["reaped"] = True
    else:
        observation["reap_error"] = "Worker remained unreaped after TERM and KILL grace"
        wait_task.cancel()
        try:
            await wait_task
        except asyncio.CancelledError:
            pass


def _native_cleanup_settled(cleanup: Any) -> bool:
    return (
        isinstance(cleanup, dict)
        and cleanup.get("child_reaped") is True
        and cleanup.get("exit_code") == 0
        and cleanup.get("request_outcome_unresolved") is False
        and cleanup.get("termination_signals") == []
    )


async def _run_isolated_worker(
    native_client: Any,
    *,
    prepare_launch: Callable[[int], _WorkerLaunch],
    observation: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Own launch, relay and process cleanup for one isolated worker command.

    The launch builder runs once after relay ownership is established. Retaining
    the spawn task prevents cancellation from discarding an unknown child PID.
    """
    if observation:
        raise ValueError("observation must be empty for a fresh execution")
    started_monotonic = time.monotonic()
    worker = {"pid": None, "spawn_attempted": False, "exit_code": None,
              "reaped": False, "termination_signals": []}
    relay: dict[str, Any] = {}
    observation.update(
        status="running", stage="preflight", worker=worker, relay=relay,
        graph_observation=None, rpc_outcome_unresolved=True,
    )
    parent_socket, child_socket = socket.socketpair()
    started = asyncio.Event()

    async def run_relay() -> dict[str, Any]:
        started.set()
        return await relay_native_tools(parent_socket, native_client, observation=relay)

    relay_task = asyncio.create_task(run_relay())
    spawn_task: asyncio.Task[asyncio.subprocess.Process] | None = None
    process: asyncio.subprocess.Process | None = None
    wait_task: asyncio.Task[int] | None = None
    launch: _WorkerLaunch | None = None
    logs: list[Any] = []
    cancelled = False
    timed_out = False

    async def monitor() -> None:
        assert wait_task is not None
        done, _ = await asyncio.wait(
            (wait_task, relay_task), return_when=asyncio.FIRST_COMPLETED
        )
        if relay_task in done:
            # A clean channel close can precede final observation writes.
            if (
                not relay_task.cancelled()
                and relay_task.exception() is None
                and relay.get("status") == "closed"
            ):
                await asyncio.shield(wait_task)
            else:
                # EOF can precede bubblewrap's child-exit status propagation.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(wait_task),
                        timeout=PROCESS_TERMINATE_GRACE_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
            return
        if not relay_task.done():
            relay_task.cancel()

    async def settle() -> None:
        nonlocal process, wait_task
        await started.wait()
        if not relay_task.done():
            relay_task.cancel()
        try:
            if spawn_task is not None and process is None:
                try:
                    process = await spawn_task
                except Exception as error:
                    observation.setdefault("owner_error", _error_record(error))
            # Keep the inherited descriptor open until spawn has a known outcome.
            child_socket.close()
            if process is not None:
                worker["pid"] = process.pid
                if wait_task is None:
                    wait_task = asyncio.create_task(process.wait())
                await _reap_worker(process, wait_task, worker)
        finally:
            child_socket.close()
            try:
                await relay_task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                observation["relay_error"] = _error_record(error)
            finally:
                parent_socket.close()

    try:
        _validate_timeout(timeout_seconds)
        launch = prepare_launch(child_socket.fileno())
        observation["command"] = list(launch.command)
        observation["stage"] = "starting"
        logs.append(launch.stdout_path.open("wb"))
        logs.append(launch.stderr_path.open("wb"))
        worker["spawn_attempted"] = True
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *launch.command,
                env=launch.environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=logs[0],
                stderr=logs[1],
                pass_fds=(child_socket.fileno(),),
                start_new_session=True,
            )
        )
        process = await asyncio.shield(spawn_task)
        worker["pid"] = process.pid
        child_socket.close()
        wait_task = asyncio.create_task(process.wait())
        observation["stage"] = "running"
        await asyncio.wait_for(monitor(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        observation["timeout_seconds"] = timeout_seconds
    except asyncio.CancelledError:
        cancelled = True
        observation["owner_cancelled"] = True
    except Exception as error:
        observation["owner_error"] = _error_record(error)
    finally:
        observation["failure_stage"] = observation["stage"]
        observation["stage"] = "closing"
        cleanup_task = asyncio.create_task(settle())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancelled = True
                observation["owner_cancelled"] = True
            except Exception:
                break
        try:
            cleanup_task.result()
        except Exception as error:
            observation["cleanup_error"] = _error_record(error)
        for handle in logs:
            handle.close()

    if launch is not None:
        try:
            graph = _read_json(launch.graph_observation_path)
            if not isinstance(graph, dict):
                raise TypeError("Worker graph observation must be a JSON object")
            observation["graph_observation"] = graph
        except (OSError, ValueError, TypeError) as error:
            observation["graph_observation_error"] = _error_record(error)
    # An aborted channel is not an uncertain tool call when no worker could
    # have used it and the native owner confirms that no request is pending.
    preflight_failed = (
        observation["failure_stage"] == "preflight"
        and spawn_task is None and worker["spawn_attempted"] is False
        and worker["pid"] is None and "owner_error" in observation
        and relay.get("requests") == []
        and _native_cleanup_settled(relay.get("native_cleanup"))
        and not relay.get("cleanup_error")
        and "cleanup_error" not in observation and "relay_error" not in observation
        and not cancelled and not timed_out
    )
    observation["rpc_outcome_unresolved"] = (
        False if preflight_failed else relay.get("rpc_outcome_unresolved") is not False
    )
    settled = preflight_failed or (
        worker["reaped"] is True
        and relay.get("status") == "closed"
        and not observation["rpc_outcome_unresolved"]
        and _native_cleanup_settled(relay.get("native_cleanup"))
        and not relay.get("cleanup_error")
        and "cleanup_error" not in observation
        and "relay_error" not in observation
    )
    if preflight_failed:
        observation["status"] = "preflight_failed"
    elif not settled:
        observation["status"] = "unknown"
    elif cancelled:
        observation["status"] = "cancelled"
    elif timed_out:
        observation["status"] = "timed_out"
    elif worker["exit_code"] != 0 or "owner_error" in observation:
        observation["status"] = "worker_failed"
    elif observation["graph_observation"] is None:
        observation["status"] = "worker_failed"
    else:
        observation["status"] = "completed"
    observation["stage"] = "settled" if settled else "unresolved"
    observation["wall_seconds"] = time.monotonic() - started_monotonic
    if launch is not None:
        try:
            _write_json(launch.observation_path, observation)
        except (OSError, ValueError, TypeError) as error:
            observation["observation_write_error"] = _error_record(error)
            observation["status"] = "unknown"
            observation["stage"] = "unresolved"
    if cancelled:
        raise asyncio.CancelledError
    return observation


async def run_isolated_cuga_graph(
    native_client: Any,
    *,
    cuga_python: str | os.PathLike[str],
    python_base_prefix: str | os.PathLike[str],
    application_lib: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    task_id: str,
    seed: int,
    model_name: str,
    base_url: str,
    expected_packages: Mapping[str, str],
    protected_roots: Sequence[str | os.PathLike[str]],
    observation: dict[str, Any],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one source CUGA graph; the caller separately performs native scoring.

    output_root must not exist. timeout_seconds, when supplied, bounds worker
    execution after launch. No per-request timeout or retry is introduced.
    Cancellation propagates only after process and native cleanup are observed.
    """
    def prepare(channel_fd: int) -> _WorkerLaunch:
        return _prepare_cuga_launch(
            channel_fd,
            cuga_python=cuga_python,
            python_base_prefix=python_base_prefix,
            application_lib=application_lib,
            output_root=output_root,
            task_id=task_id,
            seed=seed,
            model_name=model_name,
            base_url=base_url,
            expected_packages=expected_packages,
            protected_roots=protected_roots,
            observation=observation,
        )

    return await _run_isolated_worker(
        native_client,
        prepare_launch=prepare,
        observation=observation,
        timeout_seconds=timeout_seconds,
    )


def _verify_worker_runtime(
    expected: Mapping[str, str], observation: dict[str, Any]
) -> None:
    actual = {
        "executable": str(Path(sys.executable).absolute()),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
    }
    if not isinstance(expected, Mapping) or set(expected) != set(actual):
        raise ValueError("python_runtime must identify executable, prefix and base_prefix")
    observation.update(expected=dict(expected), actual=actual)
    if any(actual[key] != expected[key] for key in actual):
        observation["status"] = "failed"
        raise RuntimeError("Worker interpreter identity differs from the mounted runtime")
    observation["status"] = "passed"


async def _worker_main(channel_fd: int) -> int:
    observation: dict[str, Any] = {"status": "initializing"}
    proxy: NativeToolProxy | None = None
    exit_code = 0
    descriptor_transferred = False
    try:
        if channel_fd < 3 or not stat.S_ISSOCK(os.fstat(channel_fd).st_mode):
            raise ValueError("channel_fd must be a dedicated inherited socket")
        descriptor_transferred = True
        proxy = await NativeToolProxy.connect(channel_fd)
        config = _read_json(WORKER_CONFIG_PATH)
        if not isinstance(config, dict) or set(config) != {
            "task_id", "seed", "model_name", "base_url",
            "expected_packages", "python_runtime",
        }:
            raise ValueError("Worker configuration has missing or unsupported fields")
        local_url = _validate_task(
            task_id=config["task_id"],
            seed=config["seed"],
            model_name=config["model_name"],
            base_url=config["base_url"],
        )
        _verify_worker_runtime(
            config["python_runtime"], observation.setdefault("worker_runtime_identity", {})
        )
        _verify_expected_packages(
            config["expected_packages"],
            observation.setdefault("worker_package_identity", {}),
        )
        # Import CUGA-facing code only after distribution identities are verified.
        from population_dynamics.appworld_benchmark_cuga_runtime import (
            load_cuga_runtime,
            run_bound_cuga_graph,
        )

        runtime = load_cuga_runtime(
            seed=config["seed"], model_name=config["model_name"], base_url=local_url
        )
        # Runtime identity evidence already occupies the worker observation.
        # The task adapter owns a fresh graph observation and work directory.
        graph_observation: dict[str, Any] = {}
        try:
            await run_bound_cuga_graph(
                runtime,
                proxy,
                work_root=WORKER_CONFIG_PATH.parent / "graph",
                task_id=config["task_id"],
                model_name=config["model_name"],
                base_url=local_url,
                seed=config["seed"],
                observation=graph_observation,
            )
        finally:
            observation.update(graph_observation)
    except BaseException as error:
        exit_code = 1
        observation["worker_exception"] = _error_record(error)
        if observation["status"] == "initializing":
            observation["status"] = "initialization_failed"
    finally:
        if proxy is not None:
            try:
                # Proxy close is idempotent: run_bound may have closed it already.
                await proxy.close()
            except BaseException as error:
                exit_code = 1
                observation["worker_cleanup_error"] = _error_record(error)
        elif not descriptor_transferred and channel_fd >= 3:
            try:
                if stat.S_ISSOCK(os.fstat(channel_fd).st_mode):
                    os.close(channel_fd)
            except OSError:
                pass
        try:
            _write_json(GRAPH_OBSERVATION_PATH, observation)
        except (OSError, ValueError, TypeError) as error:
            print(f"Cannot preserve worker graph observation: {error}", file=sys.stderr)
            exit_code = 1
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker", help="Run the isolated source CUGA graph")
    worker.add_argument("--channel-fd", type=int, required=True)
    args = parser.parse_args(argv)
    if args.channel_fd < 3:
        parser.error("--channel-fd must identify a dedicated socket descriptor (>=3)")
    return asyncio.run(_worker_main(args.channel_fd))


if __name__ == "__main__":
    raise SystemExit(main())

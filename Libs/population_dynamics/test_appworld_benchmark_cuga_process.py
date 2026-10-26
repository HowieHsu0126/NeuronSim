"""Exercise the CUGA process owner through real bubblewrap/socketpair workers.

These tests require a node on which bubblewrap user namespaces are available.
They do not load a model, score AppWorld tasks, or qualify the CUGA source graph.
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from population_dynamics import appworld_benchmark_cuga_process as process


def _clean_native_cleanup() -> dict:
    return {
        "child_reaped": True,
        "exit_code": 0,
        "request_outcome_unresolved": False,
        "termination_signals": [],
    }


class _NativeClient:
    public_context = {"task_id": "fixture_task", "instruction": "Public fixture task"}

    def __init__(self, *, delay_cleanup: bool = False) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.close_count = 0
        self.cancelled_calls = 0
        self.call_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.release_call = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        if not delay_cleanup:
            self.release_cleanup.set()
        self.cleanup = _clean_native_cleanup()

    async def __call__(self, app: str, api: str, arguments: dict) -> dict:
        self.calls.append((app, api, arguments))
        self.call_started.set()
        if api == "block":
            try:
                await self.release_call.wait()
            except asyncio.CancelledError:
                self.cancelled_calls += 1
                self.cleanup["request_outcome_unresolved"] = True
                raise
        return {"arguments": arguments}

    async def close(self) -> dict:
        self.close_count += 1
        self.close_started.set()
        await self.release_cleanup.wait()
        return self.cleanup


class PreflightLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def run_preflight(self, native):
        observation = {}

        def missing_bubblewrap(_channel_fd):
            raise FileNotFoundError("bubblewrap is not available in /usr/bin or /bin")

        with mock.patch.object(process.asyncio, "create_subprocess_exec") as spawn:
            await process._run_isolated_worker(
                native, prepare_launch=missing_bubblewrap, observation=observation)
        spawn.assert_not_called()
        self.assertEqual(native.close_count, 1)
        self.assertEqual(native.calls, [])
        self.assertIsNone(observation["worker"]["pid"])
        self.assertFalse(observation["worker"]["spawn_attempted"])
        return observation

    async def test_missing_bubblewrap_has_known_preflight_failure_without_worker(self):
        observation = await self.run_preflight(_NativeClient())
        self.assertEqual(observation["status"], "preflight_failed")
        self.assertEqual(observation["stage"], "settled")
        self.assertFalse(observation["rpc_outcome_unresolved"])
        self.assertEqual(observation["relay"]["requests"], [])

    async def test_pending_native_cleanup_is_not_promoted_to_preflight_terminal(self):
        native = _NativeClient()
        native.cleanup["request_outcome_unresolved"] = True
        observation = await self.run_preflight(native)
        self.assertEqual(observation["status"], "unknown")
        self.assertTrue(observation["rpc_outcome_unresolved"])


class PythonAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cuga_alias_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "python-3.13.12"
        self.prefix = self.root / "venv"
        for directory in (self.base / "bin", self.prefix / "bin"):
            directory.mkdir(parents=True)
        self.binary = self.base / "bin" / "python3.13"
        self.binary.write_text("fixture executable")
        self.binary.chmod(0o755)
        self.alias = self.root / "python-3.13"
        self.alias.symlink_to(self.base, target_is_directory=True)
        self.executable = self.prefix / "bin" / "python"

    def test_image_alias_keeps_original_venv_executable(self) -> None:
        self.executable.symlink_to(self.alias / "bin" / "python3.13")
        owner = self.root / "owner"
        sandbox = owner / "sandbox"
        sandbox.mkdir(parents=True)
        with mock.patch.object(process.shutil, "which", return_value="/usr/bin/bwrap"), \
                mock.patch.object(process, "_system_mounts", return_value=[]):
            command = process._bubblewrap_command(
                executable=self.executable, prefix=self.prefix, base_prefix=self.base,
                application_lib=self.root / "public_lib", sandbox=sandbox,
                owner_root=owner, protected_roots=[self.root / "private"],
                worker_arguments=["-c", "pass"],
            )
        self.assertLess(command.index("--tmpfs"), command.index("--ro-bind"))
        index = command.index("--symlink")
        self.assertEqual(command[index:index + 3], ("--symlink", str(self.base), str(self.alias)))
        self.assertEqual(command[command.index("--") + 1], str(self.executable))
        self.assertNotIn(str(self.root), command)

    def test_relative_interpreter_chain_retains_alias(self) -> None:
        self.executable.symlink_to(self.alias / "bin" / "python3.13")
        other = self.prefix / "bin" / "python3"
        other.symlink_to("./python")
        self.assertEqual(process._python_runtime_alias(other, self.prefix, self.base), self.alias)

    def test_canonical_base_needs_no_alias(self) -> None:
        self.executable.symlink_to(self.binary)
        self.assertIsNone(process._python_runtime_alias(self.executable, self.prefix, self.base))

    def test_alias_to_other_runtime_is_rejected(self) -> None:
        other = self.root / "unadmitted"
        (other / "bin").mkdir(parents=True)
        (other / "bin" / "python3.13").write_text("other")
        self.alias.unlink()
        self.alias.symlink_to(other, target_is_directory=True)
        self.executable.symlink_to(self.alias / "bin" / "python3.13")
        with self.assertRaisesRegex(ValueError, "leaves the admitted runtime"):
            process._python_runtime_alias(self.executable, self.prefix, self.base)

    def test_interpreter_link_cycle_is_rejected(self) -> None:
        self.executable.symlink_to("python3")
        (self.prefix / "bin" / "python3").symlink_to("python")
        with self.assertRaisesRegex(ValueError, "symlink cycle"):
            process._python_runtime_alias(self.executable, self.prefix, self.base)


class InputValidationTests(unittest.TestCase):
    def test_worker_output_cannot_follow_a_host_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private.json"
            private.write_text('{"protected_fixture": true}')
            output = root / "worker_output.json"
            output.symlink_to(private)
            with self.assertRaises(OSError):
                process._read_json(output)
            self.assertEqual(private.read_text(), '{"protected_fixture": true}')

    def test_worker_output_fifo_is_rejected_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "worker_output.json"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError, msg="A FIFO is not terminal JSON evidence"):
                process._read_json(fifo)

    def test_model_url_accepts_only_local_endpoints(self) -> None:
        self.assertEqual(
            process._local_model_url("http://localhost:8123/v1"),
            "http://127.0.0.1:8123/v1",
        )
        self.assertEqual(
            process._local_model_url("http://[::1]:8123/v1"),
            "http://[::1]:8123/v1",
        )
        for value in (
            "https://example.org/v1",
            "http://0.0.0.0:8123",
            "http://127.0.0.1.example.org",
            "http://name:secret@127.0.0.1",
            "file:///work/model",
            "http://127.0.0.1/v1?token=secret",
            "http://127.0.0.1:99999",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                process._local_model_url(value)

    def test_protected_mounts_reject_containment_and_symlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "native"
            protected.mkdir()
            child = protected / "tasks"
            child.mkdir()
            alias = root / "alias"
            alias.symlink_to(protected, target_is_directory=True)
            for source in (root, protected, child, alias):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    process._check_mounts([source], [protected])
            public = root / "public"
            public.mkdir()
            process._check_mounts([public], [protected])

    def test_environment_has_no_ambient_secret_proxy_or_home(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"NEURONSIM_TEST_SECRET": "fixture-only", "HTTPS_PROXY": "fixture-proxy"},
        ):
            environment = process._worker_environment(
                prefix=Path("/runtime"), base_prefix=Path("/python"),
                base_url="http://127.0.0.1:8123/v1",
            )
        for name in ("NEURONSIM_TEST_SECRET", "HTTPS_PROXY", "HOME", "AGENT_HOME"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["PYTHONPATH"], "/app")
        self.assertEqual(environment["CUGA_LOGGING_DIR"], "/work/logs")
        self.assertEqual(environment["CUGA_DBS_DIR"], "/work/dbs")
        self.assertNotIn("CUGA_PACKAGE_ROOT", environment)

    def test_package_identities_compare_every_declared_distribution(self) -> None:
        expected = {name: "fixture-version" for name in process.REQUIRED_PACKAGES}
        expected["cuga"] = "0.3.2"
        expected["extra-distribution"] = "7.1"
        observation: dict = {}
        with mock.patch.object(
            process.metadata, "version", side_effect=lambda name: expected[name]
        ) as lookup:
            process._verify_expected_packages(expected, observation)
        self.assertEqual(observation["status"], "passed")
        self.assertEqual(lookup.call_count, len(expected))
        self.assertEqual(observation["actual"], expected)

        def mismatched(name: str) -> str:
            if name == "extra-distribution":
                raise process.metadata.PackageNotFoundError(name)
            return "mismatch" if name == "torch" else expected[name]

        with mock.patch.object(process.metadata, "version", side_effect=mismatched):
            with self.assertRaises(RuntimeError):
                process._verify_expected_packages(expected, observation)
        self.assertEqual(observation["status"], "failed")
        self.assertEqual(set(observation["mismatches"]), {"torch", "extra-distribution"})
        self.assertIsNone(observation["actual"]["extra-distribution"])

    def test_incomplete_or_wrong_cuga_identity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            process._package_spec({"cuga": "0.3.2"})
        expected = {name: "fixture-version" for name in process.REQUIRED_PACKAGES}
        expected["cuga"] = "0.3.3"
        with self.assertRaises(ValueError):
            process._package_spec(expected)

    def test_cli_rejects_log_descriptors_and_invalid_arguments(self) -> None:
        for arguments in (
            [],
            ["worker"],
            ["worker", "--channel-fd", "1"],
            ["worker", "--channel-fd", "not-an-integer"],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    process.main(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_seed_task_and_timeout_validation(self) -> None:
        for seed in (0, 40, 44, True, "41"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                process._validate_task(
                    task_id="fixture_task", seed=seed, model_name="local-model",
                    base_url="http://127.0.0.1:8123/v1",
                )
        with self.assertRaises(ValueError):
            process._validate_task(
                task_id="../private", seed=41, model_name="local-model",
                base_url="http://127.0.0.1:8123/v1",
            )
        for timeout in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                process._validate_timeout(timeout)
        process._validate_timeout(None)


@unittest.skipUnless(
    shutil.which("bwrap", path="/usr/bin:/bin"),
    "Real bubblewrap is required for process-boundary tests",
)
class BubblewrapOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cuga_process_test_")
        self.root = Path(self.temporary.name)
        self.protected = self.root / "native"
        self.protected.mkdir()
        self.protected_file = self.protected / "task_and_scoring_secret.txt"
        self.protected_file.write_text("fixture private data", encoding="utf-8")
        self.application_lib = Path(__file__).resolve().parents[1]
        self.native = _NativeClient()

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture_launch(
        self, body: str, *, name: str = "owner", source_worker: bool = False,
        config: dict | None = None,
    ):
        owner = self.root / name
        sandbox = owner / "sandbox"
        sandbox.mkdir(parents=True)
        private = owner / "owner_private.txt"
        private.write_text("owner-only fixture", encoding="utf-8")
        executable, prefix, base_prefix = process._python_runtime(
            sys.executable, sys.base_prefix
        )
        for directory in ("logs", "dbs", "cache", "tmp"):
            (sandbox / directory).mkdir()
        if not source_worker:
            source = (
                "import asyncio, json, os, sys\n"
                "from pathlib import Path\n"
                "from population_dynamics.appworld_benchmark_cuga_channel "
                "import NativeToolProxy\n"
                "async def run():\n"
                "    proxy = await NativeToolProxy.connect(int(sys.argv[1]))\n"
                + textwrap.indent(textwrap.dedent(body).strip() + "\n", "    ")
                + "asyncio.run(run())\n"
            )
            (sandbox / "fixture_worker.py").write_text(source, encoding="utf-8")
        if config is not None:
            (sandbox / "worker_config.json").write_text(json.dumps(config), encoding="utf-8")

        def prepare(channel_fd: int) -> process._WorkerLaunch:
            arguments = (
                (
                    "-m", "population_dynamics.appworld_benchmark_cuga_process",
                    "worker", "--channel-fd", str(channel_fd),
                )
                if source_worker else
                ("/work/fixture_worker.py", str(channel_fd))
            )
            command = process._bubblewrap_command(
                executable=executable,
                prefix=prefix,
                base_prefix=base_prefix,
                application_lib=self.application_lib,
                sandbox=sandbox,
                owner_root=owner,
                protected_roots=[self.protected],
                worker_arguments=arguments,
            )
            return process._WorkerLaunch(
                command=command,
                environment=process._worker_environment(
                    prefix=prefix, base_prefix=base_prefix,
                    base_url="http://127.0.0.1:8123/v1",
                ),
                stdout_path=owner / "stdout.log",
                stderr_path=owner / "stderr.log",
                graph_observation_path=sandbox / "graph_observation.json",
                observation_path=owner / "owner_observation.json",
            )

        return prepare, owner, private

    async def test_actual_socket_worker_closes_once_and_cannot_read_private_paths(self) -> None:
        body = f"""
        answer = await proxy("fixture", "echo", {{"x": 7}})
        assert answer == {{"arguments": {{"x": 7}}}}
        assert proxy.public_context["task_id"] == "fixture_task"
        assert list(Path("/proc").iterdir()) == []
        assert not Path({str(self.protected_file)!r}).exists()
        assert not Path({str(self.root / "owner" / "owner_private.txt")!r}).exists()
        assert "NEURONSIM_TEST_SECRET" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        assert "HOME" not in os.environ
        print("ordinary stdout log")
        print("ordinary stderr log", file=sys.stderr)
        cleanup = await proxy.close()
        Path("/work/graph_observation.json").write_text(
            json.dumps({{"status": "graph_ended", "cleanup": cleanup}}))
        """
        prepare, owner, _ = self._fixture_launch(body)
        observation: dict = {}
        with mock.patch.dict(
            os.environ,
            {"NEURONSIM_TEST_SECRET": "fixture-only", "HTTPS_PROXY": "fixture-proxy"},
        ):
            returned = await process._run_isolated_worker(
                self.native, prepare_launch=prepare, observation=observation
            )
        self.assertIs(returned, observation)
        self.assertEqual(
            observation["status"], "completed",
            {**observation, "worker_stderr": (owner / "stderr.log").read_text()},
        )
        self.assertEqual(observation["relay"]["status"], "closed")
        self.assertFalse(observation["rpc_outcome_unresolved"])
        self.assertEqual(observation["graph_observation"]["status"], "graph_ended")
        self.assertTrue(observation["worker"]["reaped"])
        self.assertEqual(observation["worker"]["exit_code"], 0)
        self.assertEqual(observation["worker"]["termination_signals"], [])
        self.assertEqual(self.native.close_count, 1)
        self.assertEqual(len(self.native.calls), 1)
        self.assertIn("ordinary stdout log", (owner / "stdout.log").read_text())
        self.assertIn("ordinary stderr log", (owner / "stderr.log").read_text())
        saved = json.loads((owner / "owner_observation.json").read_text())
        self.assertEqual(saved["relay"]["native_cleanup"], self.native.cleanup)

    async def test_worker_exit_after_call_without_close_remains_unknown(self) -> None:
        prepare, _, _ = self._fixture_launch("""
        await proxy("fixture", "echo", {"before_exit": True})
        os._exit(7)
        """)
        observation: dict = {}
        await process._run_isolated_worker(
            self.native, prepare_launch=prepare, observation=observation
        )
        self.assertEqual(observation["status"], "unknown")
        self.assertTrue(observation["worker"]["reaped"])
        self.assertEqual(observation["worker"]["exit_code"], 7)
        self.assertTrue(observation["rpc_outcome_unresolved"])
        self.assertEqual(self.native.close_count, 1)
        self.assertEqual(len(self.native.calls), 1)
        self.assertIsNone(observation["graph_observation"])

    async def test_cancel_during_native_call_waits_for_native_cleanup_and_reaps_worker(self) -> None:
        prepare, _, _ = self._fixture_launch("""
        await proxy("fixture", "block", {"in_flight": True})
        """)
        observation: dict = {}
        owner = asyncio.create_task(
            process._run_isolated_worker(
                self.native, prepare_launch=prepare, observation=observation
            )
        )
        await asyncio.wait_for(self.native.call_started.wait(), timeout=15)
        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner
        self.assertEqual(observation["status"], "unknown")
        self.assertTrue(observation["owner_cancelled"])
        self.assertTrue(observation["worker"]["reaped"])
        self.assertTrue(observation["rpc_outcome_unresolved"])
        self.assertTrue(observation["relay"]["native_cleanup"]["request_outcome_unresolved"])
        self.assertEqual(self.native.close_count, 1)
        self.assertEqual(self.native.cancelled_calls, 1)

    async def test_cancel_during_cleanup_preserves_actual_cleanup_once(self) -> None:
        self.native = _NativeClient(delay_cleanup=True)
        prepare, _, _ = self._fixture_launch("""
        await proxy.close()
        Path("/work/graph_observation.json").write_text(
            json.dumps({"status": "graph_ended"}))
        """)
        observation: dict = {}
        owner = asyncio.create_task(
            process._run_isolated_worker(
                self.native, prepare_launch=prepare, observation=observation
            )
        )
        await asyncio.wait_for(self.native.close_started.wait(), timeout=15)
        owner.cancel()
        await asyncio.sleep(0)
        self.native.release_cleanup.set()
        with self.assertRaises(asyncio.CancelledError):
            await owner
        self.assertTrue(observation["worker"]["reaped"])
        self.assertEqual(self.native.close_count, 1)
        self.assertEqual(observation["relay"]["native_cleanup"], self.native.cleanup)
        self.assertTrue(observation["owner_cancelled"])

    async def test_explicit_timeout_reaps_worker_without_retry(self) -> None:
        prepare, _, _ = self._fixture_launch("""
        await proxy("fixture", "block", {})
        """)
        observation: dict = {}
        await process._run_isolated_worker(
            self.native, prepare_launch=prepare, observation=observation,
            timeout_seconds=0.5,
        )
        self.assertEqual(observation["status"], "unknown")
        self.assertEqual(observation["timeout_seconds"], 0.5)
        self.assertTrue(observation["worker"]["reaped"])
        self.assertEqual(self.native.close_count, 1)
        self.assertLessEqual(len(self.native.calls), 1)

    async def test_source_worker_package_mismatch_exits_nonzero_with_closed_relay(self) -> None:
        expected = {name: "fixture-unmatched-version" for name in process.REQUIRED_PACKAGES}
        expected["cuga"] = "0.3.2"
        observation: dict = {}
        await process.run_isolated_cuga_graph(
            self.native,
            cuga_python=sys.executable,
            python_base_prefix=sys.base_prefix,
            application_lib=self.application_lib,
            output_root=self.root / "source_owner",
            task_id="fixture_task",
            seed=41,
            model_name="local-fixture",
            base_url="http://127.0.0.1:8123/v1",
            expected_packages=expected,
            protected_roots=[self.protected],
            observation=observation,
        )
        self.assertEqual(observation["status"], "worker_failed", observation)
        self.assertEqual(observation["relay"]["status"], "closed")
        self.assertFalse(observation["rpc_outcome_unresolved"])
        self.assertNotEqual(observation["worker"]["exit_code"], 0)
        graph = observation["graph_observation"]
        self.assertEqual(graph["status"], "initialization_failed")
        self.assertEqual(graph["worker_package_identity"]["status"], "failed")
        self.assertEqual(self.native.close_count, 1)
        self.assertEqual(self.native.calls, [])

    async def test_source_worker_invalid_config_closes_channel_and_reports_initialization(self) -> None:
        prepare, _, _ = self._fixture_launch(
            "", source_worker=True, config={"unsupported": "field"}
        )
        observation: dict = {}
        await process._run_isolated_worker(
            self.native, prepare_launch=prepare, observation=observation
        )
        self.assertEqual(observation["status"], "worker_failed", observation)
        self.assertEqual(observation["relay"]["status"], "closed")
        self.assertEqual(
            observation["graph_observation"]["worker_exception"]["type"], "ValueError"
        )
        self.assertEqual(self.native.close_count, 1)

    async def test_launch_failure_keeps_native_cleanup_and_never_retries_spawn(self) -> None:
        prepare, owner, _ = self._fixture_launch("await proxy.close()")

        def absent_executable(channel_fd: int) -> process._WorkerLaunch:
            launch = prepare(channel_fd)
            return process._WorkerLaunch(
                command=("/definitely/absent/cuga-worker",),
                environment=launch.environment,
                stdout_path=launch.stdout_path,
                stderr_path=launch.stderr_path,
                graph_observation_path=launch.graph_observation_path,
                observation_path=launch.observation_path,
            )

        observation: dict = {}
        with mock.patch.object(
            process.asyncio, "create_subprocess_exec",
            wraps=asyncio.create_subprocess_exec,
        ) as spawn:
            await process._run_isolated_worker(
                self.native, prepare_launch=absent_executable, observation=observation
            )
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(observation["status"], "unknown")
        self.assertIsNone(observation["worker"]["pid"])
        self.assertEqual(observation["owner_error"]["type"], "FileNotFoundError")
        self.assertEqual(observation["relay"]["native_cleanup"], self.native.cleanup)
        self.assertEqual(self.native.close_count, 1)
        self.assertTrue((owner / "owner_observation.json").is_file())

    async def test_protected_source_mount_rejection_precedes_output_creation(self) -> None:
        expected = {name: "fixture-version" for name in process.REQUIRED_PACKAGES}
        expected["cuga"] = "0.3.2"
        output = self.root / "rejected_owner"
        with self.assertRaises(ValueError):
            process._prepare_cuga_launch(
                19,
                cuga_python=sys.executable,
                python_base_prefix=sys.base_prefix,
                application_lib=self.application_lib,
                output_root=output,
                task_id="fixture_task",
                seed=41,
                model_name="local-fixture",
                base_url="http://127.0.0.1:8123/v1",
                expected_packages=expected,
                protected_roots=[self.application_lib / "private_native_data"],
                observation={},
            )
        self.assertFalse(output.exists())


class WorkerAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def _exercise_worker(self, *, fail_in_graph: bool) -> dict:
        from population_dynamics import appworld_benchmark_cuga_runtime as runtime
        import socket

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = root / "work"
            sandbox.mkdir()
            config_path = sandbox / "worker_config.json"
            observation_path = sandbox / "graph_observation.json"
            config_path.write_text(json.dumps({
                "task_id": "fixture_task", "seed": 41, "model_name": "local-fixture",
                "base_url": "http://127.0.0.1:8123/v1", "expected_packages": {},
                "python_runtime": {
                    "executable": str(Path(sys.executable).absolute()),
                    "prefix": str(Path(sys.prefix).resolve()),
                    "base_prefix": str(Path(sys.base_prefix).resolve()),
                },
            }))
            native = _NativeClient()
            seen = []

            async def bound_graph(_runtime, proxy, **kwargs):
                self.assertIs(proxy, native)
                self.assertEqual(kwargs["observation"], {})
                self.assertEqual(kwargs["work_root"], sandbox / "graph")
                self.assertFalse(kwargs["work_root"].exists())
                kwargs["work_root"].mkdir()
                seen.append(kwargs["work_root"])
                kwargs["observation"].update(
                    status="errored" if fail_in_graph else "graph_ended",
                    task_id=kwargs["task_id"], seed=kwargs["seed"],
                    loop={"status": "errored" if fail_in_graph else "graph_ended"},
                )
                if fail_in_graph:
                    raise RuntimeError("synthetic graph execution error")

            def verify_packages(_expected, observation):
                observation.update(status="passed")

            left, right = socket.socketpair()
            try:
                with mock.patch.object(process, "WORKER_CONFIG_PATH", config_path), \
                        mock.patch.object(process, "GRAPH_OBSERVATION_PATH", observation_path), \
                        mock.patch.object(process, "_verify_expected_packages", side_effect=verify_packages), \
                        mock.patch.object(process.NativeToolProxy, "connect", new=mock.AsyncMock(return_value=native)), \
                        mock.patch.object(runtime, "load_cuga_runtime", return_value=object()), \
                        mock.patch.object(runtime, "run_bound_cuga_graph", new=bound_graph):
                    exit_code = await process._worker_main(right.fileno())
            finally:
                left.close()
                right.close()
            self.assertEqual(exit_code, 1 if fail_in_graph else 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(native.close_count, 1)
            saved = json.loads(observation_path.read_text())
            self.assertEqual(saved["worker_runtime_identity"]["status"], "passed")
            self.assertEqual(saved["worker_package_identity"]["status"], "passed")
            self.assertEqual(saved["task_id"], "fixture_task")
            self.assertEqual(saved["seed"], 41)
            return saved

    async def test_source_worker_supplies_fresh_graph_state_and_directory(self):
        saved = await self._exercise_worker(fail_in_graph=False)
        self.assertEqual(saved["status"], "graph_ended")
        self.assertNotIn("worker_exception", saved)

    async def test_source_worker_preserves_graph_failure_and_identity_receipts(self):
        saved = await self._exercise_worker(fail_in_graph=True)
        self.assertEqual(saved["status"], "errored")
        self.assertEqual(saved["worker_exception"]["message"], "synthetic graph execution error")


if __name__ == "__main__":
    unittest.main()

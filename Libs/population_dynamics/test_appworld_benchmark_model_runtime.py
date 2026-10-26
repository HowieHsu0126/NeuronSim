"""Mocked server lifecycle tests; no weights, GPUs, network or processes run."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from population_dynamics import appworld_benchmark_model_runtime as runtime


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "neural_population_appworld_benchmark.yaml"
    config = runtime.load_benchmark_config(config_path)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    source_metadata = tmp_path / "source_metadata.json"
    source_metadata.write_text(json.dumps({"id": config["model"]["repo_id"], "sha": config["model"]["revision"]}))
    snapshot_config = tmp_path / "snapshot.yaml"
    snapshot_config.write_text(yaml.safe_dump({
        "model": {key: config["model"][key] for key in ("repo_id", "revision")},
        "inference_seed": 42,
    }))
    packages = {"vllm": config["model"]["serving_version"], "torch": "synthetic-torch"}
    package_receipt = {
        "status": "MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED",
        "python": str(Path(sys.prefix) / "bin" / "python"), "packages": packages,
    }
    runtime_receipt = tmp_path / "environment.json"
    runtime_receipt.write_text(json.dumps(package_receipt))
    state = SimpleNamespace(
        config=config, model_directory=model_directory, source_metadata=source_metadata,
        snapshot_config=snapshot_config, runtime_receipt=runtime_receipt, output=tmp_path / "qualification",
        packages=deepcopy(packages), gpu_count=config["model_qualification"]["tensor_parallel_size"], gpu_available=True,
        resource_status="PASS", snapshot_error=None, probe_status="PASS", probe_error=None,
        startup_error=None, exit_after_probe=None, disappear_before_term=False, forced_kill=False,
        interrupt_probe=False, observed_interrupt=None,
        events=[], snapshot_calls=[], package_queries=[], gpu_queries=[], launch_calls=[], transports=[], kill_calls=[], wait_calls=[],
        server=None,
        transport_record={"request_id": "synthetic-request", "attempts": 1, "status": "completed", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
    )

    def original_term_handler(signum, frame):
        return None

    state.original_term_handler = original_term_handler
    state.current_term_handler = original_term_handler
    state.signal_changes = []

    def getsignal(signum):
        assert signum == signal.SIGTERM
        return state.current_term_handler

    def set_signal(signum, handler):
        assert signum == signal.SIGTERM
        previous = state.current_term_handler
        state.current_term_handler = handler
        state.signal_changes.append(handler)
        return previous

    fake_signal = SimpleNamespace(SIGTERM=signal.SIGTERM, SIGKILL=signal.SIGKILL, getsignal=getsignal, signal=set_signal)
    monkeypatch.setattr(runtime, "signal", fake_signal)

    def package_version(name):
        state.package_queries.append(name)
        return state.packages[name]

    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(metadata=SimpleNamespace(version=package_version)))
    torch = ModuleType("torch")
    torch.__version__ = "synthetic-torch-runtime"
    torch.version = SimpleNamespace(cuda="synthetic-cuda-runtime")

    def device_count():
        state.gpu_queries.append("device_count")
        return state.gpu_count

    torch.cuda = SimpleNamespace(
        device_count=device_count, is_available=lambda: state.gpu_available,
        get_device_name=lambda index: f"synthetic-device-{index}",
        get_device_properties=lambda index: SimpleNamespace(total_memory=24 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("SLURM_JOB_GPUS", "3,5")
    # Unit fixtures use a valid scheduler shape without launching a job.
    for name in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "cpu04")
    monkeypatch.setattr(runtime, "_check_execution_location", lambda path: None)
    monkeypatch.setattr(runtime, "_free_port", lambda: 18001)

    def resource_probe(output, required_memory_gib):
        state.events.append("resource_admission")
        result = {"status": state.resource_status, "required_memory_gib": required_memory_gib}
        output.write_text(json.dumps(result))
        return result

    def verify_snapshot(directory, metadata, source_config):
        state.events.append("snapshot_verification")
        state.snapshot_calls.append((directory, deepcopy(metadata), deepcopy(source_config)))
        assert directory == model_directory
        assert source_config["inference_seed"] == 41
        if metadata["id"] != config["model"]["repo_id"] or metadata["sha"] != config["model"]["revision"]:
            raise ValueError("LOCAL_ACTOR_METADATA_PIN_MISMATCH")
        if state.snapshot_error is not None:
            raise state.snapshot_error
        return {
            "model_revision": config["model"]["repo_id"] + "@" + config["model"]["revision"],
            "all_published_file_identities_verified": True, "files": {}, "bytes": 0,
            "synthetic_test_double": True,
        }

    monkeypatch.setattr(runtime, "probe_resources", resource_probe)
    monkeypatch.setattr(runtime, "verify_snapshot", verify_snapshot)

    class FakeServer:
        pid = 81231

        def __init__(self):
            self.returncode = None
            self.kill_received = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            state.wait_calls.append(timeout)
            if state.forced_kill and not self.kill_received:
                raise subprocess.TimeoutExpired("synthetic-owned-server", timeout)
            assert self.returncode is not None
            return self.returncode

    def popen(command, **kwargs):
        state.events.append("server_start")
        state.launch_calls.append((list(command), kwargs))
        assert kwargs["start_new_session"] is True
        state.server = FakeServer()
        return state.server

    def killpg(pid, signum):
        assert state.server is not None and pid == state.server.pid
        state.kill_calls.append((pid, signum))
        if signum == signal.SIGTERM:
            if state.disappear_before_term:
                state.server.returncode = 23
                raise ProcessLookupError("synthetic server disappeared between poll and SIGTERM")
            if state.server.returncode is None and not state.forced_kill:
                state.server.returncode = -int(signal.SIGTERM)
        elif signum == signal.SIGKILL:
            state.server.kill_received = True
            state.server.returncode = -int(signal.SIGKILL)
        else:
            raise AssertionError("unexpected process signal")

    monkeypatch.setattr(runtime, "subprocess", SimpleNamespace(Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(runtime, "os", SimpleNamespace(environ=os.environ, killpg=killpg))

    def wait_for_model(server, endpoint, model_name, timeout_seconds):
        state.events.append("server_readiness")
        assert server is state.server and model_name == config["model"]["repo_id"]
        if state.startup_error is not None:
            server.returncode = 17
            raise state.startup_error
        return {"data": [{"id": model_name}]}

    class FakeTransport:
        def __init__(self, endpoint, *, timeout_seconds):
            self.endpoint = endpoint
            self.timeout_seconds = timeout_seconds
            self.records = []
            state.transports.append(self)

    def probe_model_transport(probe_config, *, transport):
        state.events.append("neutral_probe")
        assert probe_config == config
        transport.records.append(deepcopy(state.transport_record))
        if state.interrupt_probe:
            try:
                state.current_term_handler(signal.SIGTERM, None)
            except BaseException as error:
                state.observed_interrupt = type(error)
                raise
        if state.probe_error is not None:
            raise state.probe_error
        if state.exit_after_probe is not None:
            state.server.returncode = state.exit_after_probe
        return {
            "schema": "appworld_model_transport_probe_v1", "status": state.probe_status,
            "seeds": [41, 42, 43], "appworld_tasks_executed": 0,
            "native_task_qualification": "NOT_EVALUATED", "transport_records": deepcopy(transport.records),
        }

    monkeypatch.setattr(runtime, "wait_for_model", wait_for_model)
    monkeypatch.setattr(runtime, "LocalChatTransport", FakeTransport)
    monkeypatch.setattr(runtime, "probe_model_transport", probe_model_transport)
    return state


def qualify(fixture):
    report = runtime.qualify_model_runtime(
        fixture.config, model_directory=fixture.model_directory, source_metadata=fixture.source_metadata,
        snapshot_config=fixture.snapshot_config, runtime_receipt=fixture.runtime_receipt, output=fixture.output,
    )
    assert fixture.current_term_handler is fixture.original_term_handler
    assert fixture.signal_changes[-1] is fixture.original_term_handler
    assert json.loads((fixture.output / "qualification.json").read_text()) == report
    return report


def test_success_owns_server_cleanup_and_remains_neutral(runtime_fixture):
    fixture = runtime_fixture
    report = qualify(fixture)
    assert report["status"] == "PASS" and report["tool_calls_validated"] is True
    assert report["scope"] == "neutral_model_transport_only"
    assert report["appworld_tasks_executed"] == 0
    assert report["native_task_qualified"] is False and report["formal_eligible"] is False
    assert report["seeds"] == [41, 42, 43]
    assert fixture.events == ["resource_admission", "snapshot_verification", "server_start", "server_readiness", "neutral_probe"]
    assert len(fixture.launch_calls) == 1
    command, kwargs = fixture.launch_calls[0]
    assert command[command.index("--model") + 1] == str(fixture.model_directory)
    assert kwargs["start_new_session"] and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1" and kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM)]
    cleanup = report["server_cleanup"]
    assert cleanup["process_group_term_sent"] and cleanup["server_parent_reaped"]
    assert cleanup["returncode_before_termination"] is None and not cleanup["forced_kill"]
    assert json.loads((fixture.output / "transport_records.json").read_text()) == [fixture.transport_record]
    invocation = json.loads((fixture.output / "invocation.json").read_text())
    assert invocation["status"] == "RUNNING" and invocation["config"] == fixture.config


def test_shared_server_context_stays_live_across_operations_without_repeating_probe(runtime_fixture):
    fixture = runtime_fixture
    fixture.output.mkdir()
    report = {"status": "RUNNING"}
    with runtime.owned_model_server(
        fixture.config, model_directory=fixture.model_directory, source_metadata=fixture.source_metadata,
        snapshot_config=fixture.snapshot_config, runtime_receipt=fixture.runtime_receipt,
        output=fixture.output, report=report,
    ) as endpoint:
        assert endpoint == "http://127.0.0.1:18001/v1"
        assert fixture.server.poll() is None
        fixture.events.append("actor_operation")
        assert fixture.server.poll() is None
        fixture.events.append("independent_score_operation")
        report["status"] = "PASS"
    assert fixture.events == ["resource_admission", "snapshot_verification", "server_start", "server_readiness",
                              "actor_operation", "independent_score_operation"]
    assert not fixture.transports and len(fixture.launch_calls) == 1
    assert report["status"] == "PASS" and report["server_cleanup"]["server_parent_reaped"]
    assert fixture.current_term_handler is fixture.original_term_handler
    assert not (fixture.output / "transport_probe.json").exists()


def test_shared_server_body_failure_propagates_after_owned_cleanup(runtime_fixture):
    fixture = runtime_fixture
    fixture.output.mkdir()
    report = {"status": "RUNNING"}
    with pytest.raises(RuntimeError, match="caller operation failed"):
        with runtime.owned_model_server(
            fixture.config, model_directory=fixture.model_directory, source_metadata=fixture.source_metadata,
            snapshot_config=fixture.snapshot_config, runtime_receipt=fixture.runtime_receipt,
            output=fixture.output, report=report,
        ):
            raise RuntimeError("caller operation failed")
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM)]
    assert report["server_cleanup"]["server_parent_reaped"]
    assert fixture.current_term_handler is fixture.original_term_handler


@pytest.mark.parametrize("defect", ["snapshot_identity", "metadata_identity", "snapshot_hash"])
def test_source_identity_or_hash_rejection_never_starts_server(runtime_fixture, defect):
    fixture = runtime_fixture
    if defect == "snapshot_identity":
        source = yaml.safe_load(fixture.snapshot_config.read_text())
        source["model"]["revision"] = "different-source-revision"
        fixture.snapshot_config.write_text(yaml.safe_dump(source))
    elif defect == "metadata_identity":
        fixture.source_metadata.write_text(json.dumps({"id": "other/model", "sha": fixture.config["model"]["revision"]}))
    else:
        fixture.snapshot_error = ValueError("LOCAL_ACTOR_FILE_HASH_MISMATCH:synthetic.safetensors")
    report = qualify(fixture)
    assert report["status"] == "FAIL" and not report["tool_calls_validated"]
    assert report["stage"] == "source_snapshot_verification"
    assert not fixture.launch_calls and not fixture.kill_calls and not fixture.transports
    if defect == "snapshot_identity":
        assert not fixture.snapshot_calls
    else:
        assert len(fixture.snapshot_calls) == 1


@pytest.mark.parametrize("defect", ["package_version", "gpu_count", "gpu_unavailable"])
def test_runtime_package_or_visible_gpu_mismatch_prevents_launch(runtime_fixture, defect):
    fixture = runtime_fixture
    if defect == "package_version":
        fixture.packages["vllm"] = "unprepared-package-version"
    elif defect == "gpu_count":
        fixture.gpu_count += 1
    else:
        fixture.gpu_available = False
    report = qualify(fixture)
    assert report["status"] == "FAIL" and report["stage"] == "runtime_identity"
    assert not fixture.snapshot_calls and not fixture.launch_calls and not fixture.kill_calls
    if defect == "package_version":
        assert not fixture.gpu_queries


def test_resource_rejection_prevents_runtime_and_model_work(runtime_fixture):
    fixture = runtime_fixture
    fixture.resource_status = "BLOCKED"
    report = qualify(fixture)
    assert report["status"] == "FAIL" and report["stage"] == "resource_admission"
    assert not fixture.package_queries and not fixture.snapshot_calls and not fixture.launch_calls


@pytest.mark.parametrize("failure_kind", ["reported_failure", "exception"])
def test_probe_failure_stops_owned_server_and_retains_transport(runtime_fixture, failure_kind):
    fixture = runtime_fixture
    fixture.transport_record.update(status="unresolved", usage=None)
    if failure_kind == "reported_failure":
        fixture.probe_status = "FAIL"
    else:
        fixture.probe_error = RuntimeError("synthetic probe exception")
    report = qualify(fixture)
    assert report["status"] == "FAIL" and not report["tool_calls_validated"]
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM)]
    assert report["server_cleanup"]["server_parent_reaped"]
    assert json.loads((fixture.output / "transport_records.json").read_text()) == [fixture.transport_record]
    if failure_kind == "reported_failure":
        assert report["transport_probe_status"] == "FAIL"
        assert json.loads((fixture.output / "transport_probe.json").read_text())["status"] == "FAIL"
    else:
        assert report["error_type"] == "RuntimeError"
        assert not (fixture.output / "transport_probe.json").exists()


def test_sigterm_interrupt_cleans_server_preserves_records_and_restores_handler(runtime_fixture):
    fixture = runtime_fixture
    fixture.interrupt_probe = True
    fixture.transport_record.update(status="unresolved", usage=None)
    report = qualify(fixture)
    assert fixture.observed_interrupt is KeyboardInterrupt
    assert report["status"] == "FAIL" and report["error_type"] == "KeyboardInterrupt"
    assert not report["tool_calls_validated"]
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM)]
    assert report["server_cleanup"]["server_parent_reaped"]
    assert fixture.current_term_handler is fixture.original_term_handler
    assert json.loads((fixture.output / "transport_records.json").read_text()) == [fixture.transport_record]


def test_unexpected_server_exit_after_successful_probe_cannot_qualify(runtime_fixture):
    fixture = runtime_fixture
    fixture.exit_after_probe = 19
    report = qualify(fixture)
    assert report["transport_probe_status"] == "PASS"
    assert report["status"] == "FAIL" and not report["tool_calls_validated"]
    assert report["server_cleanup"]["returncode_before_termination"] == 19
    assert report["server_cleanup"]["server_parent_reaped"]


def test_server_disappearing_between_poll_and_sigterm_cannot_qualify(runtime_fixture):
    fixture = runtime_fixture
    fixture.disappear_before_term = True
    report = qualify(fixture)
    cleanup = report["server_cleanup"]
    assert cleanup["returncode_before_termination"] is None
    assert cleanup["process_group_term_sent"] is False and cleanup["returncode"] == 23
    assert report["transport_probe_status"] == "PASS"
    assert report["status"] == "FAIL" and not report["tool_calls_validated"]


def test_forced_termination_invalidates_an_otherwise_successful_probe(runtime_fixture):
    fixture = runtime_fixture
    fixture.forced_kill = True
    report = qualify(fixture)
    assert report["transport_probe_status"] == "PASS"
    assert report["status"] == "FAIL" and not report["tool_calls_validated"]
    assert report["server_cleanup"]["forced_kill"] and report["server_cleanup"]["server_parent_reaped"]
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM), (fixture.server.pid, signal.SIGKILL)]
    assert fixture.wait_calls == [30, 10]


def test_startup_exit_preserves_failure_and_reaps_owned_server(runtime_fixture):
    fixture = runtime_fixture
    fixture.startup_error = RuntimeError("synthetic server exited before readiness")
    report = qualify(fixture)
    assert report["status"] == "FAIL" and report["stage"] == "server_startup"
    assert not fixture.transports and "transport_probe_status" not in report
    assert fixture.kill_calls == [(fixture.server.pid, signal.SIGTERM)]
    assert report["server_cleanup"]["server_parent_reaped"]


def test_merlin_qualification_retains_real_identity_fields(runtime_fixture, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.delenv("SLURM_JOB_GPUS")
    monkeypatch.setenv("RH2_JOB_RUN_ID", "synthetic-merlin-run")
    monkeypatch.setenv("ARNOLD_TRIAL_ID", "synthetic-merlin-trial")
    monkeypatch.setenv("RH2_CHECKPOINT_HDFS", "hdfs://fixture/checkpoints/synthetic-merlin-run")
    report = qualify(runtime_fixture)
    assert report["status"] == "PASS"
    assert report["execution_backend"] == "merlin"
    assert report["run_id"] == "synthetic-merlin-run"
    assert report["trial_id"] == "synthetic-merlin-trial"
    assert "slurm_job_id" not in report
    assert "slurm_job_gpus" not in report["runtime"]
    assert report["runtime"]["run_id"] == report["run_id"]

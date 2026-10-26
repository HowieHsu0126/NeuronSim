
"""Pending source-output transport; no simulator execution."""
import importlib.util
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock
import pytest


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    source = Path(os.environ["NRNSIM_PFC_RUNTIME_TEST_SOURCE"])
    spec = importlib.util.spec_from_file_location("pfc_source_persistence_test_runtime", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefix = "hdfs://haruna/checkpoints/native-test-allocation"
    for key, value in {
        "RH2_JOB_RUN_ID": "native-test-allocation", "ARNOLD_TRIAL_ID": "native-test-trial",
        "ARNOLD_MONITOR_ROLE": "head", "ARNOLD_APPLIED_GPU_NUM": "0",
        "CUDA_VISIBLE_DEVICES": "", "RH2_CHECKPOINT_HDFS": prefix,
    }.items():
        monkeypatch.setenv(key, value)
    path = tmp_path / "result.tar.gz"
    path.write_bytes(b"immutable source bundle test fixture")
    sleep = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    return module, path, prefix + "/pfc/task_2641/result.tar.gz", sleep


def result(code, error=""):
    return subprocess.CompletedProcess([], code, stdout="", stderr=error)


def test_successful_upload_runs_once(runtime, monkeypatch):
    module, path, destination, sleep = runtime
    run = Mock(return_value=result(0))
    monkeypatch.setattr(module.subprocess, "run", run)
    module.persist_output(path, destination)
    assert run.call_count == 1 and run.call_args.args[0] == ["hdfs", "dfs", "-put", str(path), destination]
    sleep.assert_not_called()


def test_observed_tls_failure_retries_same_owned_bytes(runtime, monkeypatch):
    module, path, destination, sleep = runtime
    before = path.read_bytes()
    run = Mock(side_effect=[result(255, "javax.net.ssl.SSLHandshakeException"), result(0)])
    monkeypatch.setattr(module.subprocess, "run", run)
    module.persist_output(path, destination)
    assert [call.args[0] for call in run.call_args_list] == [
        ["hdfs", "dfs", "-put", str(path), destination],
        ["hdfs", "dfs", "-put", "-f", str(path), destination],
    ]
    assert path.read_bytes() == before
    sleep.assert_called_once_with(1)


def test_permanent_transport_error_is_not_retried(runtime, monkeypatch):
    module, path, destination, sleep = runtime
    run = Mock(return_value=result(1, "Permission denied"))
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        module.persist_output(path, destination)
    assert error.value.returncode == 1 and run.call_count == 1
    sleep.assert_not_called()


def test_transient_retry_is_bounded(runtime, monkeypatch):
    module, path, destination, sleep = runtime
    run = Mock(return_value=result(255, "java.io.EOFException"))
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as error:
        module.persist_output(path, destination)
    assert error.value.returncode == 255 and run.call_count == 3
    assert [call.args[0][3] for call in run.call_args_list] == [str(path), "-f", "-f"]
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_original_allocation_cannot_be_overwritten(runtime, monkeypatch):
    module, path, _, sleep = runtime
    run = Mock()
    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError, match="OUTSIDE_ALLOCATION"):
        module.persist_output(path, "hdfs://haruna/checkpoints/6de397acc866aadd/pfc/task_2641/result.tar.gz")
    run.assert_not_called()
    sleep.assert_not_called()


def test_explicit_progress_overwrite_keeps_same_destination(runtime, monkeypatch):
    module, path, destination, sleep = runtime
    run = Mock(return_value=result(0))
    monkeypatch.setattr(module.subprocess, "run", run)
    module.persist_output(path, destination, overwrite=True)
    assert run.call_args.args[0] == ["hdfs", "dfs", "-put", "-f", str(path), destination]
    sleep.assert_not_called()

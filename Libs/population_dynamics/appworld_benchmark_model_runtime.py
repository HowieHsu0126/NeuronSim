"""Own one local vLLM server, qualify its transport, and stop the server.

This allocated diagnostic reads no AppWorld task. It verifies the existing
model snapshot with the established intake function. It does not instantiate
the historical Transformers actor or reuse its generation protocol.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import yaml

from .appworld_benchmark_development import load_benchmark_config
from .appworld_benchmark_execution import execution_identity, probe_resources
from .appworld_benchmark_model_probe import probe_model_transport
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS
from .appworld_benchmark_runner import _check_execution_location
from .appworld_benchmark_transport import LocalChatTransport
from .appworld_local_actor import verify_snapshot
from .provenance import freeze_json


def build_server_command(config: dict[str, Any], model_directory: Path, port: int) -> list[str]:
    model, settings = config["model"], config["model_qualification"]
    if tuple(config.get("seeds", ())) != EXPERIMENT_SEEDS or any(type(seed) is not int for seed in config.get("seeds", ())):
        raise ValueError("model qualification requires exactly seeds 41, 42, 43")
    if model["serving_backend"] != "vllm" or model["tool_call_parser"] != "qwen3_coder":
        raise ValueError("only the declared native vLLM Qwen parser is supported")
    if settings["dtype"] != "bfloat16" or settings["enable_prefix_caching"] is not False:
        raise ValueError("qualification requires bfloat16 and explicit disabled prefix caching")
    for key in ("tensor_parallel_size", "max_model_len", "max_num_seqs"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"positive integer model setting required: {key}")
    if not 0 < settings["gpu_memory_utilization"] < 1:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if settings["enforce_eager"] is not True:
        raise ValueError("this initial qualification uses the declared eager execution")
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_directory), "--served-model-name", model["repo_id"],
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(settings["tensor_parallel_size"]),
        "--distributed-executor-backend", "mp", "--dtype", settings["dtype"],
        "--max-model-len", str(settings["max_model_len"]),
        "--max-num-seqs", str(settings["max_num_seqs"]),
        "--gpu-memory-utilization", str(settings["gpu_memory_utilization"]),
        "--seed", str(EXPERIMENT_SEEDS[0]), "--enable-auto-tool-choice",
        "--tool-call-parser", model["tool_call_parser"], "--generation-config", "vllm",
        "--no-enable-prefix-caching", "--enforce-eager", "--disable-log-requests",
    ]


def inspect_model_runtime(config: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != "MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED":
        raise ValueError("a completed model-package preparation receipt is required")
    if Path(sys.prefix).resolve() != Path(receipt["python"]).parent.parent.resolve():
        raise ValueError("qualification must run in the prepared model environment")
    packages = {name: importlib.metadata.version(name) for name in receipt["packages"]}
    if packages != receipt["packages"] or packages.get("vllm") != config["model"]["serving_version"]:
        raise ValueError("installed packages differ from the pinned preparation receipt")
    # CUDA inspection belongs inside the admitted GPU job, never module import.
    import torch
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count != config["model_qualification"]["tensor_parallel_size"]:
        raise ValueError("scheduler-visible GPUs differ from tensor parallel size")
    devices = [{"logical_index": index, "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
               for index in range(count)]
    return {"packages": packages, "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda, "devices": devices,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **execution_identity(),
            **({"slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS")}
               if os.environ.get("SLURM_JOB_ID") else {})}


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_model(server: subprocess.Popen, base_url: str, model_name: str, timeout_seconds: float) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness with code {server.returncode}")
        try:
            with opener.open(base_url + "/models", timeout=5) as response:
                result = json.load(response)
            models = result.get("data", [])
            if any(row.get("id") == model_name for row in models):
                return result
            raise ValueError("local endpoint does not expose the declared served model")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            # Read-only readiness polling never resubmits a generation.
            time.sleep(1)
    raise TimeoutError("vLLM did not become ready within the configured startup window")


def stop_owned_process(process: subprocess.Popen) -> dict:
    """Terminate an owned process group and reap its parent."""
    result = {"pid": process.pid, "process_group_term_sent": False, "forced_kill": False,
              "returncode_before_termination": process.poll()}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        result["process_group_term_sent"] = True
    except ProcessLookupError:
        pass
    try:
        result["returncode"] = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        result["forced_kill"] = True
        result["returncode"] = process.wait(timeout=10)
    result["parent_reaped"] = True
    return result


def stop_server(server: subprocess.Popen) -> dict:
    result = stop_owned_process(server)
    result["server_parent_reaped"] = result.pop("parent_reaped")
    return result


@contextmanager
def owned_model_server(
    config: dict[str, Any], *, model_directory: Path, source_metadata: Path,
    snapshot_config: Path, runtime_receipt: Path, output: Path, report: dict[str, Any],
):
    """Admit, start and reap one server without making a generation request.

    The caller owns a fresh existing output directory and its final receipt.
    This context adds the established runtime/snapshot/server fields to report.
    It sets report.status=FAIL on cleanup failure; callers must retain that
    failure when they decide their final status. Body exceptions propagate after
    owned cleanup and SIGTERM-handler restoration.
    """
    _check_execution_location(output)
    _check_execution_location(model_directory / "snapshot_verification")
    if not output.is_dir() or any((output / name).exists() for name in ("server.log", "resource_admission.json")):
        raise ValueError("server output must be a fresh caller-owned directory")
    server = None
    launch_started = None
    previous_term_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("owned model server received SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        settings = config["model_qualification"]
        report["stage"] = "resource_admission"
        resource_result = probe_resources(output / "resource_admission.json", settings["required_host_memory_gib"])
        if resource_result["status"] != "PASS":
            raise ValueError("node-local resource admission did not pass")
        report["stage"] = "runtime_identity"
        environment = json.loads(runtime_receipt.read_text())
        report["runtime"] = inspect_model_runtime(config, environment)
        report["stage"] = "source_snapshot_verification"
        source_config = yaml.safe_load(snapshot_config.read_text())
        if any(source_config["model"][key] != config["model"][key] for key in ("repo_id", "revision")):
            raise ValueError("source snapshot configuration differs from benchmark model identity")
        # The existing intake helper validates this unused field; no historical
        # actor or historical generation protocol is invoked.
        source_config["inference_seed"] = EXPERIMENT_SEEDS[0]
        phase_started = time.perf_counter()
        report["snapshot_verification"] = verify_snapshot(
            model_directory, json.loads(source_metadata.read_text()), source_config,
        )
        report["source_verification_wall_seconds"] = time.perf_counter() - phase_started
        report["stage"] = "server_startup"
        port = _free_port()
        report["endpoint"] = f"http://127.0.0.1:{port}/v1"
        command = build_server_command(config, model_directory, port)
        report["server_command"] = command
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        # Keep scheduler visibility and isolate this server's process group.
        launch_started = time.perf_counter()
        with (output / "server.log").open("x") as log:
            server = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report["server_pid"] = server.pid
            report["models_response"] = wait_for_model(
                server, report["endpoint"], config["model"]["repo_id"], settings["startup_timeout_seconds"],
            )
            report["server_startup_wall_seconds"] = time.perf_counter() - launch_started
            report["stage"] = "server_ready"
            yield report["endpoint"]
    finally:
        if server is not None:
            try:
                report["server_cleanup"] = stop_server(server)
                if report["server_cleanup"]["forced_kill"]:
                    report.update(status="FAIL", cleanup_error="server required forced termination")
                elif (report["server_cleanup"]["returncode_before_termination"] is not None
                      or not report["server_cleanup"]["process_group_term_sent"]):
                    report.update(status="FAIL", cleanup_error="server exited before owned cleanup")
            except Exception as error:
                report.update(status="FAIL", cleanup_error_type=type(error).__name__, cleanup_error=str(error))
        if launch_started is not None:
            report["server_launch_to_cleanup_wall_seconds"] = time.perf_counter() - launch_started
        signal.signal(signal.SIGTERM, previous_term_handler)


def qualify_model_runtime(
    config: dict[str, Any], *, model_directory: Path, source_metadata: Path,
    snapshot_config: Path, runtime_receipt: Path, output: Path,
) -> dict[str, Any]:
    _check_execution_location(output)
    _check_execution_location(model_directory / "snapshot_verification")
    if output.exists() or output.is_symlink():
        raise FileExistsError("an existing model qualification cannot be overwritten")
    output.mkdir()
    report: dict[str, Any] = {
        "schema": "appworld_model_runtime_qualification_v1", "status": "RUNNING",
        "scope": "neutral_model_transport_only", "formal_eligible": False,
        "appworld_tasks_executed": 0, "native_task_qualified": False,
        "model_identity": deepcopy(config["model"]), "seeds": list(EXPERIMENT_SEEDS),
        "config": deepcopy(config), "model_directory": str(model_directory),
        "runtime_receipt": str(runtime_receipt), "source_metadata": str(source_metadata),
        "snapshot_config": str(snapshot_config), "started_at": datetime.now(timezone.utc).isoformat(),
        **execution_identity(),
        "stage": "resource_admission", "tool_calls_validated": False,
    }
    freeze_json(output / "invocation.json", report)
    transport = None
    started = time.perf_counter()
    try:
        with owned_model_server(
            config, model_directory=model_directory, source_metadata=source_metadata,
            snapshot_config=snapshot_config, runtime_receipt=runtime_receipt, output=output, report=report,
        ):
            report["stage"] = "neutral_transport_probe"
            transport = LocalChatTransport(report["endpoint"], timeout_seconds=config["model_qualification"]["http_timeout_seconds"])
            result = probe_model_transport(config, transport=transport)
            freeze_json(output / "transport_probe.json", result)
            report["transport_probe_status"] = result["status"]
            report["tool_calls_validated"] = result["status"] == "PASS"
            report["status"] = "PASS" if report["tool_calls_validated"] else "FAIL"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        if transport is not None:
            freeze_json(output / "transport_records.json", transport.records)
        if report["status"] != "PASS":
            report["tool_calls_validated"] = False
        report["wall_seconds"] = time.perf_counter() - started
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "qualification.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--snapshot-config", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify_model_runtime(
        load_benchmark_config(args.config), model_directory=args.model_directory.resolve(),
        source_metadata=args.source_metadata, snapshot_config=args.snapshot_config,
        runtime_receipt=args.runtime_receipt, output=args.output.absolute(),
    )
    print(json.dumps({key: report.get(key) for key in ("status", "stage", "tool_calls_validated", "appworld_tasks_executed")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

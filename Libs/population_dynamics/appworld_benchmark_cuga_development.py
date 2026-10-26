"""Run one development CUGA attempt with owned native execution and scoring."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

from .appworld_benchmark_cuga_process import (
    _check_mounts,
    _native_cleanup_settled,
    _package_spec,
    run_isolated_cuga_graph,
)
from .appworld_benchmark_cuga_receipt import CUGA_METHOD, compose_cuga_execution
from .appworld_benchmark_execution import execution_identity
from .appworld_benchmark_development import load_benchmark_config
from .appworld_benchmark_model_runtime import owned_model_server
from .appworld_benchmark_native_tools import NativeToolClient
from .appworld_benchmark_pipeline import run_child, validate_preparations
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS
from .appworld_benchmark_runner import _check_execution_location, _write_receipt
from .appworld_benchmark_scoring import _validate_inputs as validate_scoring_inputs
from .provenance import freeze_json

CUGA_SOURCE_REVISION = "cbbdf40f8cb99a56a417b3957139a103e7096890"
CUGA_PREPARATION_STATUS = "CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _executable(value: str | os.PathLike[str]) -> Path:
    # Preserve the virtual environment path instead of resolving its symlink.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Prepared interpreter is unavailable: {path}")
    return path


def validate_cuga_preparation(
    environment: Mapping[str, Any],
) -> tuple[Path, dict[str, str]]:
    if environment.get("status") != CUGA_PREPARATION_STATUS:
        raise ValueError("CUGA runtime preparation has not been admitted")
    if environment.get("source_revision") != CUGA_SOURCE_REVISION:
        raise ValueError("CUGA source revision differs from the admitted source")
    hostname = environment.get("hostname")
    if not isinstance(hostname, str) or hostname.split(".")[0] != socket.gethostname().split(".")[0]:
        raise ValueError("Prepared CUGA runtime belongs to a different compute node")
    packages = _package_spec(environment.get("packages"))
    interpreter = environment.get("python")
    if not isinstance(interpreter, str) or not interpreter:
        raise ValueError("CUGA preparation lacks its prepared interpreter")
    return _executable(interpreter), packages


def native_tool_command(
    native_python: Path, *, config_path: Path, split_manifest: Path,
    source_root: Path, run_root: Path, split: str, task_id: str, seed: int,
) -> list[str]:
    return [
        str(native_python), "-m", "population_dynamics.appworld_benchmark_native_tools",
        "--config", str(config_path), "--split-manifest", str(split_manifest),
        "--source-root", str(source_root), "--run-root", str(run_root),
        "--split", split, "--task-id", task_id, "--seed", str(seed),
    ]


async def _run_native_cuga(
    *, native_command: list[str], stderr_path: Path, environment: dict[str, str],
    native_record: dict[str, Any], owner: dict[str, Any], cuga_arguments: dict[str, Any],
    progress,
) -> None:
    native_record.update(command=native_command, stderr=str(stderr_path), status="starting")
    progress()
    # start() closes its own client when the initial ready response fails.
    try:
        client = await NativeToolClient.start(native_command, stderr_path, env=environment)
    except BaseException:
        native_record["status"] = "startup_failed"
        progress()
        raise
    try:
        native_record["status"] = "ready"
        progress()
        await run_isolated_cuga_graph(client, observation=owner, **cuga_arguments)
    finally:
        # An owner failure before relay creation has not transferred cleanup.
        # Once the relay exists, its recorded cleanup remains authoritative.
        if not owner.get("relay"):
            try:
                native_record["cleanup_before_relay"] = await client.close()
            except BaseException as error:
                native_record.update(
                    status="cleanup_unresolved", cleanup_error_type=type(error).__name__,
                    cleanup_error=str(error),
                )
                progress()
                raise
        cleanup = owner.get("relay", {}).get("native_cleanup", native_record.get("cleanup_before_relay"))
        native_record["status"] = "closed" if _native_cleanup_settled(cleanup) else "cleanup_unresolved"
        progress()


def run_cuga_development_pipeline(
    *, config_path: Path, native_environment: Path, split_manifest: Path,
    model_qualification: Path, model_directory: Path, source_metadata: Path,
    snapshot_config: Path, runtime_receipt: Path, cuga_environment: Path,
    python_base_prefix: Path, public_application_lib: Path, repository_root: Path, output: Path,
    split: str, task_id: str, seed: int, command: Sequence[str] | None = None,
) -> dict[str, Any]:
    paths = {
        "config_path": config_path, "native_environment": native_environment,
        "split_manifest": split_manifest, "model_qualification": model_qualification,
        "model_directory": model_directory, "source_metadata": source_metadata,
        "snapshot_config": snapshot_config, "runtime_receipt": runtime_receipt,
        "cuga_environment": cuga_environment, "python_base_prefix": python_base_prefix,
        "public_application_lib": public_application_lib, "repository_root": repository_root,
    }
    paths = {name: Path(path).expanduser().resolve(strict=True) for name, path in paths.items()}
    config = load_benchmark_config(paths["config_path"])
    if split not in {"train", "dev"}:
        raise ValueError("CUGA development execution admits only train/dev")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("seed must be 41, 42 or 43")
    native = _read_object(paths["native_environment"])
    manifest = _read_object(paths["split_manifest"])
    qualification = _read_object(paths["model_qualification"])
    model_runtime = _read_object(paths["runtime_receipt"])
    cuga = _read_object(paths["cuga_environment"])
    validate_preparations(config, native, manifest, qualification, model_runtime)
    cuga_python, expected_packages = validate_cuga_preparation(cuga)
    if task_id not in manifest["splits"][split]["task_ids"]:
        raise ValueError("Requested task is absent from the admitted development membership")
    native_python = _executable(native["native_python"])
    source_root = Path(native["source_checkout"]).expanduser().resolve(strict=True)
    data_root = Path(native["data_root"]).expanduser().resolve(strict=True)
    if not source_root.is_dir() or not data_root.is_dir():
        raise ValueError("Prepared native source and data roots must be directories")
    if not all(paths[name].is_dir() for name in ("python_base_prefix", "public_application_lib", "repository_root")):
        raise ValueError("Python base prefix, staged application library and repository root must be directories")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("An existing CUGA pipeline attempt cannot be overwritten or repeated")
    _check_execution_location(output)
    _check_execution_location(data_root / "pipeline_location_check")
    _check_execution_location(paths["public_application_lib"] / "pipeline_location_check")
    native_root = output / "native_run"
    cuga_root = output / "cuga_process"
    execution_path = output / "execution_receipt.json"
    score_path = output / "development_score.json"
    native_session_path = native_root / "native_session.json"
    owner_path = cuga_root / "process_observation.json"
    protected_roots = [paths["repository_root"], data_root, native_root, source_root, score_path]
    _check_mounts([paths["public_application_lib"]], protected_roots)
    output.mkdir()
    server_output = output / "model_server"
    server_output.mkdir()
    report: dict[str, Any] = {
        "schema": "appworld_cuga_development_pipeline_v1", "scope": "one_development_attempt",
        "status": "RUNNING", "formal_eligible": False, "neuronsim_method": False,
        "agent": "cuga", "method": CUGA_METHOD, "split": split, "task_id": task_id, "seed": seed,
        "native_tgc": None, "scoring_status": "not_run", "execution_status": None,
        "stage": "model_server", "model_server": {"status": "RUNNING"},
        "native_client": {"status": "not_started"}, "cuga_owner": {}, "processes": {},
        "native_preparation": deepcopy(native), "cuga_preparation": deepcopy(cuga),
        "config": deepcopy(config), "input_paths": {key: str(value) for key, value in paths.items()},
        "native_session": str(native_session_path), "owner_observation": str(owner_path),
        "execution_receipt": str(execution_path), "scoring_outcome": str(score_path),
        "protected_roots": [str(path) for path in protected_roots],
        **execution_identity(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": list(command) if command is not None else None,
        "neutral_probe_repeated": False,
    }
    freeze_json(output / "invocation.json", report)
    receipt_path = output / "pipeline.json"
    progress = lambda: _write_receipt(receipt_path, report)
    progress()
    environment = dict(
        os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1", OPENAI_API_KEY="local-appworld-placeholder",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    started = time.monotonic()
    failure: BaseException | None = None
    try:
        with owned_model_server(
            config, model_directory=paths["model_directory"],
            source_metadata=paths["source_metadata"], snapshot_config=paths["snapshot_config"],
            runtime_receipt=paths["runtime_receipt"], output=server_output,
            report=report["model_server"],
        ) as endpoint:
            report["stage"] = "native_and_cuga"
            launch = native_tool_command(
                native_python, config_path=paths["config_path"],
                split_manifest=paths["split_manifest"], source_root=source_root,
                run_root=native_root, split=split, task_id=task_id, seed=seed,
            )
            asyncio.run(_run_native_cuga(
                native_command=launch, stderr_path=output / "native.stderr.log",
                environment=environment, native_record=report["native_client"],
                owner=report["cuga_owner"],
                cuga_arguments={
                    "cuga_python": cuga_python, "python_base_prefix": paths["python_base_prefix"],
                    "application_lib": paths["public_application_lib"], "output_root": cuga_root,
                    "task_id": task_id, "seed": seed, "model_name": config["model"]["repo_id"],
                    "base_url": endpoint, "expected_packages": expected_packages,
                    "protected_roots": protected_roots,
                },
                progress=progress,
            ))
            report["stage"] = "execution_receipt"
            native_session = _read_object(native_session_path)
            execution = compose_cuga_execution(
                native_session, report["cuga_owner"],
                native_session_path=str(native_session_path), owner_observation_path=str(owner_path),
            )
            expected = {"task_id": task_id, "split": split, "seed": seed, "method": CUGA_METHOD}
            if (
                any(execution.get(key) != value for key, value in expected.items())
                or Path(execution["run_root"]).resolve() != native_root.resolve()
                or Path(execution["data_root"]).resolve() != data_root
                or execution["appworld_version"] != manifest["appworld_version"]
            ):
                raise ValueError("CUGA/native execution identity differs from the requested attempt")
            freeze_json(execution_path, execution)
            report.update(execution_status=execution["status"], usage_complete=execution["usage_complete"])
            if (
                execution["status"] == "unknown" or execution["terminal"] is not True
                or execution["rpc_outcome_unresolved"] is not False
            ):
                report.update(status="UNKNOWN_NATIVE_EXECUTION", stage="scoring_withheld",
                              scoring_withheld_reason="execution_outcome_unresolved")
            else:
                validate_scoring_inputs(execution, manifest)
                report["stage"] = "scorer"
                report["processes"]["scorer"] = {}
                run_child(
                    [str(native_python), "-m", "population_dynamics.appworld_benchmark_scoring",
                     "--execution-receipt", str(execution_path),
                     "--split-manifest", str(paths["split_manifest"]), "--output", str(score_path)],
                    output=output, name="scorer", environment=environment,
                    record=report["processes"]["scorer"], progress=progress,
                )
                outcome = _read_object(score_path)
                if (
                    outcome.get("schema") != "appworld_development_score_v1"
                    or outcome.get("formal_eligible") is not False
                    or any(outcome.get(key) != value for key, value in expected.items())
                ):
                    raise ValueError("Scoring output differs from the requested CUGA attempt")
                report["outcome"] = outcome
                report["scoring_status"] = outcome.get("scoring_status")
                if (
                    report["processes"]["scorer"]["exit_code"] == 0
                    and outcome.get("scoring_status") == "completed"
                    and type(outcome.get("native_tgc")) is bool
                ):
                    report["native_tgc"] = outcome["native_tgc"]
            if type(report["native_tgc"]) is bool:
                report["status"] = "SCORED_DEVELOPMENT_ATTEMPT"
            elif report["status"] != "UNKNOWN_NATIVE_EXECUTION":
                report["status"] = "INCOMPLETE_DEVELOPMENT_ATTEMPT"
    except BaseException as error:
        failure = error
        unknown = (
            isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError))
            or report["cuga_owner"].get("status") == "unknown"
            or report["native_client"].get("status") == "cleanup_unresolved"
        )
        report.update(status="UNKNOWN" if unknown else "FAILED",
                      error_type=type(error).__name__, error=str(error))
    finally:
        server = report["model_server"]
        if server["status"] != "FAIL":
            server["status"] = (
                "STOPPED" if server.get("server_cleanup", {}).get("server_parent_reaped")
                else "NOT_STARTED" if "server_pid" not in server else "UNKNOWN"
            )
        if server["status"] in {"FAIL", "UNKNOWN"}:
            report["status"] = "FAILED_RUNTIME_CLEANUP"
        report.update(ended_at=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.monotonic() - started)
        progress()
    if isinstance(failure, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
        raise failure
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    for name in (
        "native-environment", "split-manifest", "model-qualification", "model-directory",
        "source-metadata", "snapshot-config", "runtime-receipt", "cuga-environment",
        "python-base-prefix", "public-application-lib", "repository-root", "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    args = parser.parse_args(argv)
    invocation = list(sys.argv) if argv is None else list(argv)
    result = run_cuga_development_pipeline(**vars(args), command=invocation)
    print(json.dumps({"status": result["status"], "native_tgc": result["native_tgc"]}))
    return 0 if result["status"] == "SCORED_DEVELOPMENT_ATTEMPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

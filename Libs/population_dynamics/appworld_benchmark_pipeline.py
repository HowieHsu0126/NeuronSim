"""One real development attempt: owned model server, actor, separate scorer.

Run this owner in the prepared model environment. Native actor/scorer commands
use the separately prepared AppWorld interpreter on the same compute node.
This entry point never repeats the neutral model probe or retries a task.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

from .appworld_benchmark_development import load_benchmark_config, validate_model_qualification
from .appworld_benchmark_execution import execution_identity
from .appworld_benchmark_model_runtime import owned_model_server, stop_owned_process
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, validate_split_manifest
from .appworld_benchmark_runner import _check_execution_location, _write_receipt
from .appworld_benchmark_scoring import _validate_inputs as validate_scoring_inputs
from .provenance import freeze_json


_AGENTS = ("official_react_code", "official_full_code", "official_function_calling", "collaboration_fixed", "collaboration_uniform")


def select_development_task(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, index: int,
) -> dict[str, Any]:
    """Select by native membership order, excluding configured prompt demos.

    This helper reads membership metadata only. It does not open task content,
    execution artifacts or scores. Its recorded index refers to the eligible
    list after excluding all configured official demonstration task IDs.
    """
    if split not in {"train", "dev"} or type(index) is not int or index < 0:
        raise ValueError("selection requires a train/dev split and a nonnegative integer index")
    validate_split_manifest(manifest)
    development = config["development_execution"]
    demo_ids = set(development["api_predictor"]["demo_task_ids"])
    for settings in development["methods"].values():
        demo_ids.update(settings.get("demo_task_ids", []))
    members = list(manifest["splits"][split]["task_ids"])
    eligible = [task_id for task_id in members if task_id not in demo_ids]
    if index >= len(eligible):
        raise ValueError("selection index exceeds the non-demo development membership")
    task_id = eligible[index]
    return {"selection_rule": "native_split_order_excluding_configured_prompt_demos", "split": split,
            "task_id": task_id, "eligible_index": index, "native_membership_index": members.index(task_id),
            "excluded_demo_task_ids": sorted(demo_ids.intersection(members)),
            "eligible_task_count": len(eligible), "split_membership_sha256": manifest["splits"][split]["membership_sha256"],
            "appworld_version": manifest["appworld_version"], "task_content_read": False, "scores_read": False}


def validate_preparations(
    config: Mapping[str, Any], native: Mapping[str, Any], manifest: Mapping[str, Any],
    qualification: Mapping[str, Any], model_runtime: Mapping[str, Any],
) -> None:
    """Consume the existing package, split and neutral qualification contracts."""
    if config.get("seeds") != list(EXPERIMENT_SEEDS) or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("canonical seeds must be exactly 41, 42, 43")
    if native.get("status") != "PREPARED_RUNTIME_AND_SPLIT_METADATA":
        raise ValueError("native runtime preparation has not completed")
    if native.get("source_revision") != config["source"]["inspected_revision"]:
        raise ValueError("prepared native source differs from the canonical revision")
    if native.get("requested_data_version") != config["source"]["candidate_data_version"]:
        raise ValueError("prepared data release request differs from the canonical release")
    if native.get("packages", {}).get("appworld") != manifest.get("appworld_version") or manifest.get("appworld_version") != config["source"]["candidate_package_version"]:
        raise ValueError("native preparation, split manifest and canonical package versions differ")
    if Path(native["data_root"]).resolve() != Path(manifest["data_root"]).resolve():
        raise ValueError("prepared data root differs from the split manifest")
    if native.get("hostname", "").split(".")[0] != socket.gethostname().split(".")[0]:
        raise ValueError("native environment and model server must be on the same compute node")
    if manifest.get("schema") != "appworld_split_membership_v1":
        raise ValueError("an admitted native split manifest is required")
    validate_split_manifest(manifest)
    validate_model_qualification(config, qualification)
    qualified_packages = qualification["runtime"]["packages"]
    if (model_runtime.get("status") != "MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED"
            or model_runtime.get("packages") != qualified_packages):
        raise ValueError("model packages differ from completed qualification")
    # owned_model_server independently compares installed package versions with
    # this receipt before launch. A relocated environment may reuse the
    # qualification, but a different dependency stack cannot reuse its PASS.


def actor_command(
    native_python: Path, *, config_path: Path, split_manifest: Path, source_root: Path,
    attempt_root: Path, endpoint: str, agent: str, split: str, task_id: str, seed: int,
    verification_budget: int | None, planner_calls: int | None, max_model_calls: int | None,
    model_qualification: Path | None = None,
) -> list[str]:
    common = ["--config", str(config_path), "--split-manifest", str(split_manifest),
              "--source-root", str(source_root), "--run-root", str(attempt_root),
              "--model-url", endpoint, "--split", split, "--task-id", task_id, "--seed", str(seed)]
    if agent.startswith("official_"):
        if agent == "official_function_calling":
            if model_qualification is None:
                raise ValueError("FunctionCalling requires the model qualification path")
            common += ["--model-qualification", str(model_qualification)]
        return [str(native_python), "-m", "population_dynamics.appworld_benchmark_development", *common, "--method", agent]
    policy = "fixed" if agent == "collaboration_fixed" else "uniform"
    arguments = [str(native_python), "-m", "population_dynamics.appworld_benchmark_collaboration_development",
                 *common, "--policy", policy, "--verification-budget", str(verification_budget),
                 "--max-model-calls", str(max_model_calls)]
    if policy == "fixed":
        arguments += ["--planner-calls", str(planner_calls)]
    return arguments


def run_child(command: list[str], *, output: Path, name: str, environment: dict[str, str], record: dict[str, Any], progress) -> None:
    """Run once, close logs and reap the actor/scorer before its consumer starts."""
    record.update(command=command, status="starting", exit_code=None,
                  stdout=str(output / f"{name}.stdout.log"), stderr=str(output / f"{name}.stderr.log"))
    process = None
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    try:
        with Path(record["stdout"]).open("x") as stdout, Path(record["stderr"]).open("x") as stderr:
            process = subprocess.Popen(command, cwd=output, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(pid=process.pid, status="running")
            progress()
            record.update(exit_code=process.wait(), status="exited")
    except BaseException as error:
        record.update(status="interrupted_or_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if process is not None:
                try:
                    record["cleanup"] = stop_owned_process(process)
                except BaseException as error:
                    record.update(cleanup_error_type=type(error).__name__, cleanup_error=str(error))
                    raise
        finally:
            record["wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started
            progress()


def _expected_method(agent: str, verification_budget: int | None, planner_calls: int | None) -> str:
    if agent.startswith("official_"):
        return agent
    policy = "fixed" if agent == "collaboration_fixed" else "uniform"
    return (f"development_full_history_{policy}_verification_budget_{verification_budget}"
            + (f"_planner_{planner_calls}" if policy == "fixed" else ""))


def run_development_pipeline(
    *, config_path: Path, native_environment: Path, split_manifest: Path,
    model_qualification: Path, model_directory: Path, source_metadata: Path,
    snapshot_config: Path, runtime_receipt: Path, output: Path,
    agent: str, split: str, task_id: str, seed: int,
    verification_budget: int | None = None, planner_calls: int | None = None,
    max_model_calls: int | None = None, command: Sequence[str] | None = None,
) -> dict[str, Any]:
    # Child working directories differ from the owner's directory. Resolve
    # inputs once so relative CLI paths cannot silently select different files.
    config_path = config_path.expanduser().resolve(strict=True)
    native_environment = native_environment.expanduser().resolve(strict=True)
    split_manifest = split_manifest.expanduser().resolve(strict=True)
    model_qualification = model_qualification.expanduser().resolve(strict=True)
    model_directory = model_directory.expanduser().resolve(strict=True)
    source_metadata = source_metadata.expanduser().resolve(strict=True)
    snapshot_config = snapshot_config.expanduser().resolve(strict=True)
    runtime_receipt = runtime_receipt.expanduser().resolve(strict=True)
    config = load_benchmark_config(config_path)
    if agent not in _AGENTS or split not in {"train", "dev"}:
        raise ValueError("only the declared development agents and train/dev splits are available")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("seed must be 41, 42 or 43")
    if agent.startswith("official_"):
        if any(value is not None for value in (verification_budget, planner_calls, max_model_calls)):
            raise ValueError("official agents use their canonical settings, without collaboration overrides")
    else:
        if type(verification_budget) is not int or verification_budget < 0 or type(max_model_calls) is not int or max_model_calls <= 0:
            raise ValueError("collaboration requires explicit verification and model-call budgets")
        if agent == "collaboration_uniform" and (split != "train" or planner_calls is not None):
            raise ValueError("uniform collection is train-only and cannot specify a fixed quota")
        if agent == "collaboration_fixed" and (type(planner_calls) is not int or not 0 <= planner_calls <= verification_budget):
            raise ValueError("fixed control requires a planner quota within its total budget")
    native = json.loads(native_environment.read_text())
    manifest = json.loads(split_manifest.read_text())
    qualification = json.loads(model_qualification.read_text())
    model_runtime = json.loads(runtime_receipt.read_text())
    validate_preparations(config, native, manifest, qualification, model_runtime)
    if task_id not in manifest["splits"][split]["task_ids"]:
        raise ValueError("requested task is absent from the admitted development membership")
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("an existing pipeline attempt cannot be overwritten or repeated")
    _check_execution_location(output)
    _check_execution_location(Path(native["data_root"]) / "pipeline_location_check")
    native_python = Path(native["native_python"])
    if not native_python.is_file() or not os.access(native_python, os.X_OK):
        raise ValueError("prepared native interpreter is unavailable on this node")
    output.mkdir()
    server_output = output / "model_server"
    server_output.mkdir()
    attempt = output / "attempt"
    execution_path, score_path = attempt / "execution_receipt.json", attempt / "development_score.json"
    method = _expected_method(agent, verification_budget, planner_calls)
    report: dict[str, Any] = {
        "schema": "appworld_native_development_pipeline_v1", "scope": "one_development_attempt",
        "status": "RUNNING", "formal_eligible": False, "neuronsim_method": False,
        "agent": agent, "method": method, "split": split, "task_id": task_id, "seed": seed,
        "native_tgc": None, "stage": "model_server", "model_server": {"status": "RUNNING"},
        "native_preparation": deepcopy(native), "model_qualification": str(model_qualification),
        "split_manifest": str(split_manifest), "config": deepcopy(config),
        "source_metadata": str(source_metadata), "model_directory": str(model_directory),
        "runtime_receipt": str(runtime_receipt), "snapshot_config": str(snapshot_config),
        "execution_receipt": str(execution_path), "scoring_outcome": str(score_path),
        "verification_budget": verification_budget, "planner_calls": planner_calls, "max_model_calls": max_model_calls,
        "processes": {}, "started_at": datetime.now(timezone.utc).isoformat(),
        **execution_identity(),
        "command": list(command) if command is not None else None,
        "neutral_probe_repeated": False,
    }
    freeze_json(output / "invocation.json", report)
    receipt_path = output / "pipeline.json"
    progress = lambda: _write_receipt(receipt_path, report)
    progress()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       OPENAI_API_KEY="local-appworld-placeholder")
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    failure: BaseException | None = None
    try:
        with owned_model_server(
            config, model_directory=model_directory, source_metadata=source_metadata,
            snapshot_config=snapshot_config, runtime_receipt=runtime_receipt,
            output=server_output, report=report["model_server"],
        ) as endpoint:
            report["stage"] = "actor"
            report["processes"]["actor"] = {}
            run_child(actor_command(
                native_python, config_path=config_path, split_manifest=split_manifest,
                source_root=Path(native["source_checkout"]), attempt_root=attempt, endpoint=endpoint,
                agent=agent, split=split, task_id=task_id, seed=seed, verification_budget=verification_budget,
                planner_calls=planner_calls, max_model_calls=max_model_calls,
                model_qualification=model_qualification,
            ), output=output, name="actor", environment=environment, record=report["processes"]["actor"], progress=progress)
            execution = json.loads(execution_path.read_text())
            validate_scoring_inputs(execution, manifest)
            expected = {"task_id": task_id, "split": split, "seed": seed, "method": method}
            if any(execution.get(key) != value for key, value in expected.items()) or Path(execution["run_root"]).absolute() != attempt:
                raise ValueError("actor receipt differs from the requested attempt identity")
            report.update(execution_status=execution["status"], model_call_count=execution.get("lm_call_count"),
                          usage_complete=execution["usage_complete"], stage="scorer")
            report["processes"]["scorer"] = {}
            run_child([str(native_python), "-m", "population_dynamics.appworld_benchmark_scoring",
                       "--execution-receipt", str(execution_path), "--split-manifest", str(split_manifest),
                       "--output", str(score_path)],
                      output=output, name="scorer", environment=environment, record=report["processes"]["scorer"], progress=progress)
            outcome = json.loads(score_path.read_text())
            if outcome.get("schema") != "appworld_development_score_v1" or any(outcome.get(key) != value for key, value in expected.items()):
                raise ValueError("scoring output differs from the requested attempt identity")
            report["outcome"] = outcome
            if outcome.get("scoring_status") == "completed" and type(outcome.get("native_tgc")) is bool:
                report["native_tgc"] = outcome["native_tgc"]
            log_root = attempt / "experiments/outputs" / execution["experiment_name"] / "tasks" / task_id
            api_log = log_root / "logs/api_calls.jsonl"
            report["native_api_log"] = str(api_log) if api_log.is_file() else None
            if api_log.is_file():
                with api_log.open() as stream:
                    report["native_api_log_rows"] = sum(bool(line.strip()) for line in stream)
            report["native_versions"] = {name: (log_root / "version" / f"{name}.txt").read_text().strip()
                                         for name in ("code", "data") if (log_root / "version" / f"{name}.txt").is_file()}
        if report["model_server"].get("status") == "FAIL":
            report["status"] = "FAILED_RUNTIME_CLEANUP"
        elif report["processes"]["scorer"]["exit_code"] == 0 and type(report["native_tgc"]) is bool:
            report["status"] = "SCORED_DEVELOPMENT_ATTEMPT"
        else:
            report["status"] = "INCOMPLETE_DEVELOPMENT_ATTEMPT"
    except BaseException as error:
        failure = error
        report.update(status="UNKNOWN" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                      error_type=type(error).__name__, error=str(error))
    finally:
        server = report["model_server"]
        if server["status"] != "FAIL":
            server["status"] = ("STOPPED" if server.get("server_cleanup", {}).get("server_parent_reaped")
                                else "NOT_STARTED" if "server_pid" not in server else "UNKNOWN")
        report.update(ended_at=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - started)
        progress()
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "native-environment", "split-manifest", "model-qualification", "model-directory", "source-metadata", "snapshot-config", "runtime-receipt", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--agent", choices=_AGENTS, required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, choices=EXPERIMENT_SEEDS, required=True)
    parser.add_argument("--verification-budget", type=int)
    parser.add_argument("--planner-calls", type=int)
    parser.add_argument("--max-model-calls", type=int)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        values = vars(args)
        values["config_path"] = values.pop("config")
        report = run_development_pipeline(**values, command=[sys.executable, "-m", __spec__.name, *arguments])
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: report.get(key) for key in ("status", "agent", "task_id", "seed", "native_tgc")}))
    return 0 if report["status"] == "SCORED_DEVELOPMENT_ATTEMPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

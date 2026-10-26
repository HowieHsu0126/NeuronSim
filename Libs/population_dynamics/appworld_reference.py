"""Native AppWorld trace execution and isolated evaluation.

This runner is a reference/replay interface, not an agent or a learned simulator.
Each CLI invocation owns one fresh native world. Named native checkpoints only
save databases; fresh-process prefix replay is available by repeating the trace.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import yaml

from .provenance import freeze_json, runtime_manifest, sha256_file

APPWORLD_VERSION = "0.1.3.post1"
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


def validate_config(config: Any) -> dict[str, Any]:
    fields = {"expected_appworld_version", "run_class", "dataset", "task_id", "random_seed", "limits", "steps"}
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError("APPWORLD_CONFIG_FIELDS_INVALID")
    if config["expected_appworld_version"] != APPWORLD_VERSION:
        raise ValueError("APPWORLD_VERSION_NOT_VALIDATED")
    if config["run_class"] != "engineering_diagnostic" or config["dataset"] not in {"train", "dev"}:
        raise ValueError("APPWORLD_DIAGNOSTIC_REQUIRES_TRAIN_OR_DEV")
    if not isinstance(config["task_id"], str) or re.fullmatch(r"[a-f0-9]+_[1-9][0-9]*", config["task_id"]) is None:
        raise ValueError("APPWORLD_TASK_ID_INVALID")
    seed = config["random_seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("APPWORLD_NATIVE_RANDOM_SEED_INVALID")
    limits = config["limits"]
    if not isinstance(limits, dict) or set(limits) != {"max_interactions", "max_api_calls_per_interaction", "timeout_seconds"}:
        raise ValueError("APPWORLD_LIMIT_FIELDS_INVALID")
    for name, value in limits.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"APPWORLD_POSITIVE_INTEGER_LIMIT_REQUIRED:{name}")
    steps = config["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 128:
        raise ValueError("APPWORLD_BOUNDED_NONEMPTY_TRACE_REQUIRED")
    labels, checkpoints = set(), set()
    for step in steps:
        if not isinstance(step, dict) or "label" not in step:
            raise ValueError("APPWORLD_STEP_MAPPING_REQUIRED")
        label = step["label"]
        if not isinstance(label, str) or NAME.fullmatch(label) is None or label in labels:
            raise ValueError("APPWORLD_UNIQUE_STEP_LABEL_REQUIRED")
        labels.add(label)
        operations = set(step) - {"label"}
        if len(operations) != 1 or not operations <= {"code", "save_state", "load_state"}:
            raise ValueError("APPWORLD_EXACTLY_ONE_STEP_OPERATION_REQUIRED")
        operation = next(iter(operations)); value = step[operation]
        if operation == "code":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("APPWORLD_NONEMPTY_CODE_REQUIRED")
        else:
            if not isinstance(value, str) or NAME.fullmatch(value) is None:
                raise ValueError("APPWORLD_CHECKPOINT_NAME_INVALID")
            if operation == "save_state":
                if value in checkpoints:
                    raise ValueError("APPWORLD_CHECKPOINT_OVERWRITE_REJECTED")
                checkpoints.add(value)
            elif value not in checkpoints:
                raise ValueError("APPWORLD_CHECKPOINT_MUST_EXIST_IN_PREFIX")
    # Only execute() persists the canonical terminal databases consumed by the
    # native evaluator. A final checkpoint load would leave stale terminal data.
    if "code" not in steps[-1]:
        raise ValueError("APPWORLD_TRACE_MUST_END_WITH_CODE_TO_PERSIST_TERMINAL_STATE")
    return config


def load_config(path: Path) -> dict[str, Any]:
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _native_root(output: Path, data_root: Path) -> Path:
    # Native AppWorld deletes its experiment directory during initialization.
    # A new owning output prevents existing experiment data from being exposed.
    output.mkdir(parents=True, exist_ok=False)
    native = output / "native"
    native.mkdir()
    (native / "data").symlink_to(data_root / "data", target_is_directory=True)
    return native


def _activate(native: Path, cache: Path) -> None:
    os.environ["APPWORLD_ROOT"] = str(native)
    os.environ["APPWORLD_CACHE"] = str(cache)
    os.environ["TMPDIR"] = str(cache / "tmp")
    (cache / "tmp").mkdir(parents=True, exist_ok=True)


def _reject_active_world() -> None:
    # Check before changing environment/root paths or acquiring cleanup ownership.
    # The CLI is process-isolated; this also protects accidental in-process use.
    module = sys.modules.get("appworld")
    world_class = getattr(module, "AppWorld", None)
    if world_class is not None and world_class.id_to_time_freezer:
        raise ValueError("APPWORLD_REQUIRES_ISOLATED_PROCESS_WITH_NO_ACTIVE_WORLD")


def run_reference(config: dict[str, Any], data_root: Path, output: Path,
                  command: list[str]) -> tuple[dict[str, Any], int]:
    config = validate_config(config)
    _reject_active_world()
    data_root, output = data_root.resolve(), output.resolve()
    if not (data_root / "data/datasets" / (config["dataset"] + ".txt")).is_file():
        raise ValueError("APPWORLD_DATASET_INPUT_MISSING")
    native = _native_root(output, data_root)
    _activate(native, output / ".cache")
    started = time.monotonic()
    summary: dict[str, Any] = {"status": "FAILED", "scope": config["run_class"],
        "task_id": config["task_id"], "dataset": config["dataset"], "steps_completed": 0,
        "native_experiment_name": "reference", "data_root": str(data_root),
        "agent_execution_performed": False, "task_solution_qualified": False,
        "complete_state_restore_qualified": False, "generative_transfer_qualified": False}
    close_all = None
    code = 1
    try:
        installed = importlib.metadata.version("appworld")
        if installed != config["expected_appworld_version"]:
            raise ValueError("APPWORLD_INSTALLED_VERSION_MISMATCH")
        # Optional runtime dependency; metadata/config commands must stay light.
        from appworld import AppWorld, update_root
        from appworld.task import load_task_ids
        if AppWorld.id_to_time_freezer:
            raise ValueError("APPWORLD_REQUIRES_ISOLATED_PROCESS_WITH_NO_ACTIVE_WORLD")
        close_all = AppWorld.close_all
        update_root(str(native))
        if config["task_id"] not in load_task_ids(config["dataset"]):
            raise ValueError("APPWORLD_TASK_NOT_IN_DECLARED_PARTITION")
        # load_state() closes the old clock internally. Native context-manager
        # exit would stop that stale clock again; the public close_all() owns
        # cleanup of the currently registered clocks/APIs exactly once below.
        world = AppWorld(task_id=config["task_id"], experiment_name="reference",
                         load_ground_truth=False, random_seed=config["random_seed"],
                         **config["limits"])
        if world.task.ground_truth is not None:
            raise ValueError("APPWORLD_GROUND_TRUTH_EXPOSED_TO_RUNTIME")
        summary.update(appworld_version=installed, instruction=world.task.instruction,
            generator_id=world.task.generator_id, db_version=world.task.db_version,
            ground_truth_loaded=False, task_datetime=world.task.datetime.isoformat())
        with (output / "trace.jsonl").open("x", encoding="utf-8") as log:
            for index, step in enumerate(config["steps"]):
                operation = next(key for key in step if key != "label")
                row = {"step_index": index, "label": step["label"], "operation": operation,
                       "num_interactions_before": world.num_interactions,
                       "environment_io_entries_before": len(world.environment_io)}
                try:
                    if operation == "code":
                        row["code"] = step["code"]
                        row["output"] = world.execute(step["code"])
                        row["native_execution_succeeded"] = not row["output"].startswith("Execution failed.")
                        if not row["native_execution_succeeded"]:
                            raise RuntimeError("APPWORLD_NATIVE_STEP_FAILED:" + step["label"])
                    elif operation == "save_state":
                        row["checkpoint_id"] = world.save_state(step[operation])
                    else:
                        row["checkpoint_id"] = step[operation]
                        world.load_state(step[operation])
                        if world.task.ground_truth is not None:
                            raise ValueError("APPWORLD_GROUND_TRUTH_EXPOSED_AFTER_RESTORE")
                    row["operation_succeeded"] = True
                except Exception as exc:
                    row.update(operation_succeeded=False, error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    row.update(num_interactions_after=world.num_interactions,
                               environment_io_entries_after=len(world.environment_io))
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                summary["steps_completed"] += 1
        terminal = native / "experiments/outputs/reference/tasks" / config["task_id"] / "dbs"
        if not terminal.is_dir() or not any(terminal.iterdir()):
            raise RuntimeError("APPWORLD_TERMINAL_DATABASES_MISSING")
        summary["terminal_databases_persisted"] = True
        summary["status"] = "PARTIAL"
        summary["native_trace_executed"] = True
        code = 2
    except Exception as exc:
        summary.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        if close_all is not None:
            try:
                close_all()
            except Exception as exc:
                summary.update(status="FAILED", cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
                code = 1
        summary["wall_seconds_including_runtime_import"] = time.monotonic() - started
        freeze_json(output / "summary.json", summary)
        artifacts = {"summary.json": sha256_file(output / "summary.json")}
        if (output / "trace.jsonl").is_file():
            artifacts["trace.jsonl"] = sha256_file(output / "trace.jsonl")
        # Native delta databases/model hashes are consumed by evaluation and replay.
        native_output = native / "experiments/outputs/reference/tasks" / config["task_id"]
        if native_output.exists():
            artifacts.update({str(path.relative_to(output)): sha256_file(path)
                              for path in native_output.rglob("*") if path.is_file()})
        freeze_json(output / "manifest.json", {"schema": "appworld_native_trace_v1",
            "source_partition_sha256": sha256_file(data_root / "data/datasets" / (config["dataset"] + ".txt")),
            "artifacts": artifacts, **runtime_manifest(config, command, code)})
    return summary, code


def evaluate_reference(run: Path, output: Path, command: list[str]) -> tuple[dict[str, Any], int]:
    """Score persisted terminal databases in a separate process, using native tests."""
    _reject_active_world()
    run, output = run.resolve(), output.resolve()
    if output.is_relative_to(run):
        raise ValueError("APPWORLD_EVALUATION_MUST_BE_SEPARATE_FROM_RUNTIME_OUTPUT")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = validate_config(manifest["resolved_config"])
    if manifest["exit_code"] != 2:
        raise ValueError("APPWORLD_CANNOT_EVALUATE_FAILED_REFERENCE_RUN")
    partition = run / "native/data/datasets" / (config["dataset"] + ".txt")
    if not partition.is_file() or sha256_file(partition) != manifest["source_partition_sha256"]:
        raise ValueError("APPWORLD_SOURCE_PARTITION_CHANGED_SINCE_EXECUTION")
    for name, expected in manifest["artifacts"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha256_file(path) != expected:
            raise ValueError("APPWORLD_REFERENCE_ARTIFACT_MISMATCH:" + name)
    output.mkdir(parents=True, exist_ok=False)
    _activate(run / "native", output / ".cache")
    if importlib.metadata.version("appworld") != config["expected_appworld_version"]:
        raise ValueError("APPWORLD_INSTALLED_VERSION_MISMATCH")
    from appworld import AppWorld, update_root
    from appworld.evaluator import evaluate_task
    if AppWorld.id_to_time_freezer:
        raise ValueError("APPWORLD_EVALUATOR_REQUIRES_FRESH_PROCESS")
    update_root(str(run / "native"))
    report = {"status": "FAILED", "task_id": config["task_id"],
              "agent_execution_performed": False, "generative_transfer_qualified": False,
              "isolated_evaluator": True}
    code = 1
    try:
        tracker = evaluate_task(task_id=config["task_id"], experiment_name="reference",
                                suppress_errors=True, save_report=False)
        # Native failed assertions are negative task outcomes, not evaluator
        # crashes. Preserve every failed requirement and its original traceback.
        report.update(status="PARTIAL", native_evaluation=tracker.to_dict(stats_only=False))
        code = 2
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            AppWorld.close_all()
        except Exception as exc:
            report.update(status="FAILED", cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
            code = 1
        freeze_json(output / "evaluation.json", report)
        freeze_json(output / "manifest.json", {"schema": "appworld_native_evaluation_v1",
            "reference_manifest_sha256": sha256_file(run / "manifest.json"),
            "artifacts": {"evaluation.json": sha256_file(output / "evaluation.json")},
            **runtime_manifest(config, command, code)})
    return report, code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            report, code = run_reference(load_config(args.config), args.data_root, args.output, sys.argv)
        else:
            report, code = evaluate_reference(args.run, args.output, sys.argv)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

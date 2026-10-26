"""Separate-process native scoring for development AppWorld attempts.

Invoke this module in a fresh Slurm process after the execution process exits.
It never imports language-model or agent modules, or closes a live AppWorld to make scoring
possible. The official evaluator retains its open-CachedDBHandler rejection.
Only train/dev receipts are accepted. Singleton aggregate SGC is not reported.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, TERMINAL_STATUSES, validate_split_manifest
from .appworld_benchmark_runner import _check_execution_location, _load_native, _verify_memberships
from .provenance import canonical_json


_SCOREABLE_STATUSES = {"completed", "agent_failure"}


def _load_evaluator() -> Callable[..., dict[str, Any]]:
    return importlib.import_module("appworld.evaluator").evaluate_tasks


def _validate_inputs(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "appworld_development_execution_v1":
        raise ValueError("a development execution receipt is required")
    if receipt.get("scope") != "development_only" or receipt.get("formal_eligible") is not False:
        raise ValueError("formal scoring is unavailable in this development scorer")
    if receipt.get("split") not in {"train", "dev"}:
        raise ValueError("formal/test splits are prohibited in the development scorer")
    if receipt.get("terminal") is not True or receipt.get("status") not in TERMINAL_STATUSES:
        raise ValueError("execution must have an explicit terminal status before scoring")
    if type(receipt.get("seed")) is not int or receipt["seed"] not in EXPERIMENT_SEEDS:
        raise ValueError("attempt seed must be exactly one of 41, 42, 43")
    if not isinstance(receipt.get("method"), str) or not receipt["method"].strip():
        raise ValueError("execution receipt must identify its method")
    if type(receipt.get("usage_complete")) is not bool:
        raise ValueError("execution receipt must preserve a boolean usage_complete flag")
    if receipt.get("native_tgc") is not None or receipt.get("scoring_status") != "not_run":
        raise ValueError("execution receipt must be unscored and must not contain native_tgc")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "appworld_split_membership_v1":
        raise ValueError("an admitted appworld_split_membership_v1 manifest is required")
    memberships = validate_split_manifest(manifest)
    split = receipt["split"]
    if split not in memberships or receipt.get("task_id") not in memberships[split]:
        raise ValueError("receipt task does not match its admitted split membership")
    for field in ("run_root", "data_root", "experiment_name", "appworld_version"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise ValueError(f"execution receipt is missing {field}")
    experiment = Path(receipt["experiment_name"])
    if experiment.is_absolute() or ".." in experiment.parts or str(experiment) == ".":
        raise ValueError("experiment_name must stay within the native experiment output directory")


def _validate_native_row(result: Any, task_id: str) -> dict[str, Any]:
    if not isinstance(result, Mapping) or not isinstance(result.get("individual"), Mapping):
        raise ValueError("native evaluation is missing individual task results")
    if set(result["individual"]) != {task_id}:
        raise ValueError("native evaluation must contain exactly the requested task ID")
    row = result["individual"][task_id]
    if not isinstance(row, Mapping) or type(row.get("success")) is not bool:
        raise ValueError("native task success must be a bool")
    num_tests = row.get("num_tests")
    if type(num_tests) is not int or num_tests <= 0:
        raise ValueError("native task num_tests must be a positive integer")
    passes, failures = row.get("passes"), row.get("failures")
    if not isinstance(passes, list) or not isinstance(failures, list):
        raise ValueError("native evaluation must include its passes and failures lists")
    if len(passes) + len(failures) > num_tests:
        raise ValueError("native test counts exceed num_tests")
    # Native TestTracker.success is pass_count == num_tests. A negative result
    # can contain fewer observed checks; do not invent failures for the rest.
    if row["success"] != (len(passes) == num_tests):
        raise ValueError("native success is inconsistent with its passes count")
    return deepcopy(dict(row))


def score_development_attempt(
    execution_receipt: Mapping[str, Any], manifest: Mapping[str, Any], output_path: str | Path, *,
    execution_receipt_path: str | Path | None = None,
    split_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write one new canonical outcome without changing the execution receipt.

    Call from the separate scoring CLI for real evaluation. An existing output
    file is refused before native access. Evaluator/admission errors produce an
    evaluator_failure outcome with native_tgc=None. A native False remains a
    valid scored negative. Other terminal execution statuses are retained as
    unscored outcomes without importing or calling native scoring code.
    """
    _validate_inputs(execution_receipt, manifest)
    receipt = execution_receipt
    root = Path(receipt["run_root"]).expanduser().absolute()
    evaluation_directory = root / "experiments" / "outputs" / receipt["experiment_name"] / "tasks" / receipt["task_id"] / "evaluation"
    report_path = evaluation_directory / "report.md"
    version_path = evaluation_directory / "version.txt"
    output = Path(output_path).expanduser().absolute()
    if output.resolve() in {report_path.resolve(), version_path.resolve()}:
        raise ValueError("score output must be separate from native report files")
    output.parent.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, Any] = {
        "schema": "appworld_development_score_v1", "scope": "development_only", "formal_eligible": False,
        "split": receipt["split"], "task_id": receipt["task_id"], "method": receipt["method"], "seed": receipt["seed"],
        "status": receipt["status"], "execution_status": receipt["status"], "native_tgc": None,
        "scoring_status": "not_run", "usage_complete": receipt["usage_complete"],
        "evaluator_called": False, "membership_verified": False, "native_task_result": None,
        "native_evaluation_report": None, "native_evaluation_version_file": None,
        "provenance": {
            "appworld_version": manifest["appworld_version"], "run_root": str(root),
            "data_root": manifest["data_root"], "experiment_name": receipt["experiment_name"],
            "execution_receipt_path": str(execution_receipt_path) if execution_receipt_path is not None else None,
            "split_manifest_path": str(split_manifest_path) if split_manifest_path is not None else None,
            "membership_file": manifest["splits"][receipt["split"]].get("membership_file"),
            "membership_sha256": manifest["splits"][receipt["split"]]["membership_sha256"],
        },
    }
    # Reserve this new output before any evaluator side effects. An interrupted
    # scoring process leaves an existing, unresolved file that cannot be reused.
    with output.open("x", encoding="utf-8") as stream:
        if receipt["status"] not in _SCOREABLE_STATUSES:
            outcome["unscored_reason"] = "execution_status_not_scoreable"
        else:
            saved_environment = {name: os.environ.get(name) for name in ("APPWORLD_ROOT", "APPWORLD_CACHE")}
            native = None
            try:
                _check_execution_location(root)
                if not root.is_dir() or root.is_symlink():
                    raise ValueError("execution run_root must remain its original directory")
                source_root = Path(manifest["data_root"]).resolve(strict=True)
                data_directory = (source_root / "data").resolve(strict=True)
                if Path(receipt["data_root"]).resolve(strict=True) != source_root:
                    raise ValueError("execution receipt data_root differs from admitted manifest")
                data_link = root / "data"
                if not data_link.is_symlink() or data_link.resolve(strict=True) != data_directory:
                    raise ValueError("execution data symlink differs from the admitted native data directory")
                if receipt["appworld_version"] != manifest["appworld_version"]:
                    raise ValueError("execution version differs from admitted manifest")
                if not evaluation_directory.resolve().is_relative_to(root.resolve()):
                    raise ValueError("native evaluation output must remain within run_root")
                for path in (report_path, version_path):
                    if path.exists() or path.is_symlink():
                        raise FileExistsError(f"existing native evaluation output cannot be overwritten: {path}")
                os.environ["APPWORLD_ROOT"] = str(root)
                os.environ["APPWORLD_CACHE"] = str(root / ".cache")
                native = _load_native()
                native.path_store.update_root(str(root))
                native.path_store.reload()
                _verify_memberships(native, manifest, data_directory)
                outcome["membership_verified"] = True
                evaluate_tasks = _load_evaluator()
                outcome["evaluator_called"] = True
                result = evaluate_tasks(
                    task_ids=[receipt["task_id"]], experiment_name=receipt["experiment_name"],
                    suppress_errors=True, include_details=True, save_reports=True,
                )
                row = _validate_native_row(result, receipt["task_id"])
                if not report_path.is_file() or not version_path.is_file():
                    raise ValueError("native evaluator did not retain its requested report and version files")
                outcome.update(native_tgc=row["success"], native_task_result=row, scoring_status="completed")
            except Exception as error:
                outcome.update(status="evaluator_failure", native_tgc=None, scoring_status="evaluator_failure", error_type=type(error).__name__, error=str(error))
            finally:
                for name, value in saved_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                if native is not None:
                    native.path_store.reload()
                for name, value in saved_environment.items():
                    if value is None:
                        os.environ.pop(name, None)
                if report_path.is_file():
                    outcome["native_evaluation_report"] = str(report_path)
                if version_path.is_file():
                    outcome["native_evaluation_version_file"] = str(version_path)
        outcome["recorded_at"] = datetime.now(timezone.utc).isoformat()
        stream.write(canonical_json(outcome))
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.execution_receipt.read_text(encoding="utf-8"))
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        outcome = score_development_attempt(
            receipt, manifest, args.output,
            execution_receipt_path=args.execution_receipt, split_manifest_path=args.split_manifest,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"development scoring rejected: {error}", file=sys.stderr)
        return 2
    print(f"{outcome['status']}: native_tgc={outcome['native_tgc']}; output={args.output}")
    return 0 if outcome["scoring_status"] == "completed" and type(outcome["native_tgc"]) is bool else 1


if __name__ == "__main__":
    raise SystemExit(main())

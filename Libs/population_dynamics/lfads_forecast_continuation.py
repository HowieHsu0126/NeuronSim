"""Admit a committed LFADS forecast prefix from a closed resource-stopped attempt."""
from __future__ import annotations

import ast
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from .provenance import sha256_file


class _WithoutContinuation(ast.NodeTransformer):
    """Specialize the optional continuation path to None, leaving other paths intact."""

    def visit_FunctionDef(self, node):
        if node.name not in {"evaluate", "main"}:
            return node
        for index in reversed(range(len(node.args.kwonlyargs))):
            if (node.args.kwonlyargs[index].arg == "continue_from"
                    and isinstance(node.args.kw_defaults[index], ast.Constant)
                    and node.args.kw_defaults[index].value is None):
                del node.args.kwonlyargs[index]
                del node.args.kw_defaults[index]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "completed_keys"
                and isinstance(node.value, ast.Constant) and node.value.value is None):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id in {"continue_from", "completed_keys"} and not node.orelse
                and len(test.ops) == len(test.comparators) == 1
                and isinstance(test.ops[0], ast.IsNot)
                and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "parser"
                and call.func.attr == "add_argument" and call.args
                and isinstance(call.args[0], ast.Constant) and call.args[0].value == "--continue-from"):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [key for key in node.keywords if not (
            key.arg == "continue_from" and isinstance(key.value, ast.Attribute)
            and isinstance(key.value.value, ast.Name) and key.value.value.id == "args"
            and key.value.attr == "continue_from")]
        return self.generic_visit(node)


def continuation_default_program(source):
    return ast.dump(_WithoutContinuation().visit(ast.parse(source)), include_attributes=False)


def prepare_continuation(parent, output, record, cases, evaluator_source):
    """Copy only committed origins; independent full replay is still mandatory."""
    parent, output, evaluator_source = map(Path, (parent, output, evaluator_source))
    delivery_path = parent / "delivery.json"
    delivery = json.loads(delivery_path.read_text())
    if delivery["status"] != "RESOURCE_BLOCKED" or delivery["exit_code"] != 3:
        raise ValueError("CLOSED_RESOURCE_STOPPED_FORECAST_PARENT_REQUIRED")
    entries = {row["path"]: row for row in delivery["files"]}
    if len(entries) != len(delivery["files"]):
        raise ValueError("UNIQUE_PARENT_DELIVERY_PATHS_REQUIRED")

    def checked(relative):
        source = parent / relative
        entry = entries[relative]
        if source.is_symlink() or source.stat().st_size != entry["bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError("PARENT_FORECAST_ARTIFACT_CHANGED:" + relative)
        return source

    prefix = "candidate/forecasts/"
    old_path = checked(prefix + "evaluation.json")
    old = json.loads(old_path.read_text())
    committed = old["completed_origins"]
    config = record["configuration"]
    registered = [(case, origin) for origin in config["origins_ms"] for case in cases]
    if (type(committed) is not int or not 1 <= committed <= len(registered)
            or old["score_rows"] != committed * len(config["horizons_ms"])
            or old["status"] not in {"RUNNING", "FAILED", "EVALUATED_DEVELOPMENT"}):
        raise ValueError("COMMITTED_FORECAST_PREFIX_REQUIRED")
    metadata = ("model_type", "configuration", "training_seed", "training_binding", "training_run_sha256",
        "selected_checkpoint", "selected_training_sha256", "selected_model_artifacts", "selected_epoch",
        "native_source_identity", "inference_migration", "unit_metrics_implementation_sha256",
        "scientific_claim_qualified", "formal_test_accessed", "calibration_outcomes_accessed")
    if any(old[key] != record[key] for key in metadata) or any(
            old.get(key) != record.get(key) for key in ("prior_refit", "conditional_prior", "model_variant")):
        raise ValueError("FORECAST_CONTINUATION_SCIENTIFIC_BINDING_CHANGED")
    source_path = parent / "source_evaluator.py"
    if (sha256_file(source_path) != old["evaluator_implementation_sha256"]
            or continuation_default_program(source_path.read_text()) !=
                continuation_default_program(evaluator_source.read_text())):
        raise ValueError("FORECAST_CONTINUATION_NUMERICAL_PROGRAM_CHANGED")
    score_path = checked(prefix + "scores.csv")
    with score_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    count = old["score_rows"]
    if columns is None or len(columns) != len(set(columns)) or len(rows) < count:
        raise ValueError("COMMITTED_FORECAST_SCORE_ROWS_REQUIRED")
    inherited = rows[:count]
    expected_rows = [(case, origin, horizon) for case, origin in registered[:committed]
                     for horizon in config["horizons_ms"]]
    names = ["fitting_field_scale.npz", "full_unit_samples.npz"]
    for row, (case, origin, horizon) in zip(inherited, expected_rows):
        expected_name = f"{case['case_id']}_origin_{origin:g}.npz"
        if (None in row or any(value is None for value in row.values())
                or row["case_id"] != case["case_id"] or row["template_id"] != case["template_id"]
                or int(row["block"]) != case["block"] or int(row["seed"]) != record["training_seed"]
                or float(row["origin_ms"]) != origin or float(row["horizon_ms"]) != horizon
                or row["method"] != "lfads" or row["ensemble_artifact"] != expected_name):
            raise ValueError("COMMITTED_FORECAST_PREFIX_IDENTITY_CHANGED")
        if expected_name not in names:
            names.append(expected_name)
    sources = {name: checked(prefix + name) for name in names}
    with np.load(sources["fitting_field_scale.npz"], allow_pickle=False) as previous, np.load(
            output / "fitting_field_scale.npz", allow_pickle=False) as current:
        if (set(previous.files) != set(current.files) or any(
                previous[key].dtype != current[key].dtype or not np.array_equal(previous[key], current[key])
                for key in previous.files)):
            raise ValueError("FORECAST_CONTINUATION_OBSERVATION_SCALE_CHANGED")
    phase_path = checked("phases/forecast/receipt.json")
    phase = json.loads(phase_path.read_text())
    if phase["status"] not in {"RESOURCE_BLOCKED", "PASS"} or not phase["command_started"]:
        raise ValueError("CLOSED_PARENT_FORECAST_PHASE_REQUIRED")
    artifacts = {}
    for name, source in sources.items():
        shutil.copy2(source, output / name)
        artifacts[name] = sha256_file(output / name)
        if artifacts[name] != entries[prefix + name]["sha256"]:
            raise ValueError("COPIED_FORECAST_PREFIX_CHANGED:" + name)
    with (output / "scores.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(inherited)
    # Commit inherited counters only after all copied files and rows are valid.
    # An interrupted copy remains uncommitted; the original parent stays intact.
    record.update(completed_origins=committed, score_rows=count,
        graph_lifetimes=copy.deepcopy(old["graph_lifetimes"]),
        forecast_continuation=dict(parent_job_id=delivery["job_id"],
            parent_delivery_sha256=sha256_file(delivery_path), parent_evaluation_sha256=sha256_file(old_path),
            parent_evaluator_sha256=old["evaluator_implementation_sha256"],
            current_evaluator_sha256=sha256_file(evaluator_source), numerical_program_without_continuation_equal=True,
            inherited_origins=committed, inherited_score_rows=count, ignored_uncommitted_score_rows=len(rows)-count,
            inherited_artifacts=artifacts.copy(), parent_forecast_phase=phase,
            parent_attempt_started_utc=delivery.get("started_utc"), parent_attempt_completed_utc=delivery.get("completed_utc"),
            previous_continuation=old.get("forecast_continuation"),
            random_streams="Original canonical case_index and origin_index; each origin recreates its three streams",
            cost_scope="Current elapsed_seconds is this attempt only; parent phase/attempt times include abandoned work",
            full_48_origin_replay_required=True))
    return {(case["case_id"], origin) for case, origin in registered[:committed]}, artifacts, count

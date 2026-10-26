"""Build causal train examples from retained AppWorld collaboration receipts.

The builder reads no native task data and calls no model or evaluator. A caller
supplies an explicit planned roster, the admitted split manifest, and retained
execution/score/input-admission records. Missing and unsettled attempts remain
in coverage. Only settled, actually allocated boundaries produce examples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from .appworld_benchmark_host import UniformVerificationPolicy
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, TERMINAL_STATUSES, validate_split_manifest
from .appworld_benchmark_scoring import _validate_inputs as validate_scoring_inputs, _validate_native_row


_ACTORS = ("planner", "executor")
_KEY_FIELDS = ("split", "task_id", "method", "seed")
_SETTLED = {"completed", "agent_failure"}
_SOURCE_MODULES = (
    "appworld_benchmark_protocol.py", "appworld_benchmark_collaboration.py",
    "appworld_benchmark_host.py", "appworld_benchmark_collaboration_development.py",
    "appworld_benchmark_scoring.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("APPWORLD_CLOSURE_DATA: " + message)


def _sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             name + " must be an explicit SHA256")
    return value


def _key(row: Mapping[str, Any]) -> tuple:
    _require(isinstance(row, Mapping), "attempt identity must be an object")
    _require(row.get("split") == "train", "only train attempts are accepted")
    _require(type(row.get("seed")) is int and row["seed"] in EXPERIMENT_SEEDS,
             "seed must be 41, 42 or 43")
    _require(all(isinstance(row.get(name), str) and row[name].strip()
                 for name in ("task_id", "method")), "task and method identity required")
    return tuple(row[name] for name in _KEY_FIELDS)


def _policy(receipt: Mapping[str, Any], train_ids: list[str]) -> tuple[dict, float | None]:
    budget, policy = receipt.get("verification_budget"), receipt.get("allocation_policy")
    _require(type(budget) is int and budget >= 0, "verification budget required")
    _require(isinstance(policy, dict) and policy.get("score_conditioned") is False,
             "explicit outcome-blind collection/control policy required")
    probability = policy.get("assignment_probability")
    if probability is not None:
        _require(type(probability) in (int, float) and math.isfinite(probability)
                 and 0 < probability <= 1, "invalid recorded assignment probability")
    name = policy.get("name")
    if name == "uniform_verification_collection":
        native_index = train_ids.index(receipt["task_id"])
        _require(policy.get("seed") == receipt["seed"]
                 and type(policy.get("draw_index")) is int
                 and policy["draw_index"] == native_index,
                 "uniform draw must retain the original native train index")
        if probability is not None:
            _require(probability == 1 / (budget + 1), "uniform probability differs from budget")
        expected = UniformVerificationPolicy(receipt["seed"], native_index)(
            state_reports={}, total_budget=budget)
        method = f"development_full_history_uniform_verification_budget_{budget}"
    elif name == "fixed_verification_control":
        planner = policy.get("planner_calls")
        _require(type(planner) is int and 0 <= planner <= budget, "invalid fixed planner quota")
        expected = {"planner": planner, "executor": budget - planner}
        method = f"development_full_history_fixed_verification_budget_{budget}_planner_{planner}"
    else:
        raise ValueError("APPWORLD_CLOSURE_DATA: unsupported collection/control policy")
    _require(receipt["method"] == method, "method and allocation policy differ")
    return expected, probability


def _binding(receipt: Mapping[str, Any], admission: Mapping[str, Any],
             manifest: Mapping[str, Any], manifest_identity: Mapping[str, Any]) -> dict:
    _require(isinstance(admission, Mapping)
             and admission.get("status") == "RESTORED_INPUT_IDENTITIES_VERIFIED"
             and admission.get("formal_task_contents_inspected") is False,
             "verified input-admission identity required")
    split = admission.get("split_manifest", {})
    _require(isinstance(split, Mapping), "split-manifest binding must be an object")
    _require(all(split.get(field) == manifest_identity.get(field) for field in ("bytes", "sha256")),
             "input admission and consumed split-manifest bytes differ")
    source, model = receipt.get("source", {}), receipt.get("model_identity", {})
    _require(isinstance(source, Mapping) and isinstance(model, Mapping), "source/model identity required")
    for name, record in (("source", source), ("model", model)):
        _require(isinstance(record.get("revision"), str)
                 and re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is not None,
                 name + " requires an exact revision")
    _require(isinstance(source.get("repository"), str) and bool(source["repository"])
             and isinstance(model.get("repo_id"), str) and bool(model["repo_id"]),
             "named source repository and model required")
    _require(model.get("language_model_training") is False, "frozen backbone identity required")
    source_manifest = _sha(admission.get("source_archive_manifest_sha256"), "source archive")
    archive_manifest = source.get("archive_manifest", {})
    _require(isinstance(archive_manifest, Mapping), "source archive binding must be an object")
    _require(archive_manifest.get("sha256") == source_manifest,
             "execution and admitted source archive differ")
    _require(admission.get("model_revision") == model["revision"], "admitted model revision differs")
    _require(receipt.get("appworld_version") == manifest.get("appworld_version")
             and receipt.get("data_root") == manifest.get("data_root"),
             "execution and split-manifest version/data binding differ")
    hashes = admission.get("source_sha256", {})
    _require(isinstance(hashes, Mapping), "protocol source hashes required")
    protocol = {name: _sha(hashes.get("Libs/population_dynamics/" + name), name)
                for name in _SOURCE_MODULES}
    _require(isinstance(admission.get("data_archive"), Mapping), "data archive binding required")
    _require(receipt.get("state_encoder") == "full_visible_history_collection_not_learned"
             and receipt.get("neuronsim_method") is False,
             "this builder accepts the existing full-history collection/control protocol")
    return {
        "appworld_version": receipt["appworld_version"],
        "appworld_source": {"repository": source["repository"], "revision": source["revision"],
                           "archive_manifest_sha256": source_manifest},
        "model": {"repo_id": model["repo_id"], "revision": model["revision"]},
        "split_manifest": dict(manifest_identity),
        "data_archive_sha256": _sha(admission.get("data_archive", {}).get("sha256"), "data archive"),
        "protocol_source_sha256": protocol, "state_encoder": receipt["state_encoder"],
    }


def _score(receipt: Mapping[str, Any], score: Mapping[str, Any] | None,
           manifest: Mapping[str, Any]) -> dict:
    if score is None:
        return {"scoring_status": "absent", "native_tgc": None}
    _require(isinstance(score, Mapping) and score.get("schema") == "appworld_development_score_v1"
             and score.get("scope") == "development_only" and score.get("formal_eligible") is False,
             "canonical development scoring receipt required")
    _require(_key(score) == _key(receipt) and receipt.get("terminal") is True,
             "score cannot belong to another or unfinished attempt")
    _require(score.get("execution_status") == receipt["status"]
             and score.get("usage_complete") == receipt["usage_complete"],
             "score and execution status/usage differ")
    provenance = score.get("provenance", {})
    _require(isinstance(provenance, Mapping), "score provenance must be an object")
    for name in ("appworld_version", "data_root", "run_root", "experiment_name"):
        _require(provenance.get(name) == receipt.get(name), "score provenance differs: " + name)
    _require(provenance.get("membership_sha256") == manifest["splits"]["train"]["membership_sha256"],
             "score membership identity differs")
    status, success = score.get("scoring_status"), score.get("native_tgc")
    if status == "completed":
        _require(receipt["status"] in _SETTLED and score.get("status") == receipt["status"]
                 and score.get("membership_verified") is True and score.get("evaluator_called") is True
                 and type(success) is bool, "completed native score is not qualified")
        native = _validate_native_row({"individual": {receipt["task_id"]: score.get("native_task_result")}},
                                     receipt["task_id"])
        _require(native["success"] == success, "native score and native task result differ")
        _require(all(isinstance(score.get(name), str) and score[name]
                     for name in ("native_evaluation_report", "native_evaluation_version_file")),
                 "completed score lacks report/version provenance")
    else:
        _require(status in {"not_run", "evaluator_failure"} and success is None
                 and score.get("native_task_result") is None,
                 "unscored outcome must retain an unknown native target")
        _require(score.get("status") == ("evaluator_failure" if status == "evaluator_failure"
                                        else receipt["status"]), "scoring failure status differs")
    return {"scoring_status": status, "native_tgc": success,
            "execution_status": score["execution_status"], "status": score["status"],
            "native_evaluation_report": score.get("native_evaluation_report"),
            "native_evaluation_version_file": score.get("native_evaluation_version_file"),
            "provenance": deepcopy(provenance)}


def _history_events(host: Mapping[str, Any]) -> dict[str, list[dict]]:
    histories = host.get("actor_histories")
    _require(isinstance(histories, dict) and set(histories) == set(_ACTORS),
             "both actor histories are required at a settled boundary")
    events = {}
    for actor in _ACTORS:
        history = histories[actor]
        _require(isinstance(history, dict) and history.get("actor") == actor
                 and isinstance(history.get("events"), list), "actor history identity differs")
        events[actor] = history["events"]
        for index, event in enumerate(events[actor]):
            _require(isinstance(event, dict) and type(event.get("index")) is int
                     and event["index"] == index and isinstance(event.get("kind"), str)
                     and isinstance(event.get("payload"), dict), "history indices/payloads are inconsistent")
    return events


def _example(receipt: Mapping[str, Any], expected_allocation: dict, probability: float | None,
             score: dict) -> dict | None:
    host = receipt["host"]
    _require("allocation" in host, "settled host must explicitly record allocation")
    allocation = host["allocation"]
    exports = host.get("state_exports")
    _require(isinstance(host.get("native_attempts"), list), "settled host lacks native attempt ledger")
    used = host.get("verification_used")
    _require(isinstance(used, dict) and set(used) == set(_ACTORS)
             and all(type(value) is int and value >= 0 for value in used.values()),
             "verification usage must contain explicit integer counts")
    _require(isinstance(exports, list), "settled host lacks state export records")
    if allocation is None:
        _require(not exports, "settled allocation-null attempt contains a partial state boundary")
        _require(host.get("verification_used") == {actor: 0 for actor in _ACTORS}
                 and not any(row.get("verification") is True for row in host.get("native_attempts", [])),
                 "allocation-null attempt contains verification execution")
        return None
    _require(isinstance(allocation, dict) and set(allocation) == set(_ACTORS)
             and all(type(allocation[actor]) is int and allocation[actor] >= 0 for actor in _ACTORS)
             and allocation == expected_allocation, "actual allocation differs from its recorded policy")
    _require(len(exports) == 2 and all(isinstance(row, dict) for row in exports)
             and {row.get("actor") for row in exports} == set(_ACTORS),
             "allocated boundary requires exactly two state exports")
    events = _history_events(host)
    prefixes, future_ids, before_ids = {}, set(), set()
    for export in exports:
        actor, count = export["actor"], export.get("prefix_events")
        history = events[actor]
        _require(type(count) is int and 0 < count <= len(history) - 2,
                 "state export prefix count is incomplete")
        prefix = history[:count]
        _require(not any(event["kind"] in {"state_report_delivered", "verification_allocation"}
                         or (event["kind"] == "tool_attempt" and event["payload"].get("verification") is True)
                         for event in prefix), "prefix crosses a prior state/allocation boundary")
        expected_state = {"representation": "full_visible_history", "actor": actor, "events": prefix}
        _require(export.get("representation") == expected_state, "exported state contains a different/future prefix")
        _require(history[count]["kind"] == "state_report_delivered"
                 and history[count]["payload"] == {"representation": expected_state},
                 "state export is not bound to its delivered actor prefix")
        _require(history[count + 1]["kind"] == "verification_allocation"
                 and history[count + 1]["payload"] == {"assigned_calls": allocation[actor]},
                 "state report is not followed by the actual allocation")
        prefixes[actor] = {"actor": actor, "events": deepcopy(prefix)}
        for index, event in enumerate(history):
            if event["kind"] == "tool_attempt":
                result_id = event["payload"].get("result_id")
                _require(isinstance(result_id, str) and result_id.startswith(actor + "/"),
                         "tool result actor identity differs")
                target = before_ids if index < count else future_ids
                _require(result_id not in before_ids | future_ids, "duplicate tool-attempt identity")
                target.add(result_id)
    attempts = host.get("native_attempts")
    _require(isinstance(attempts, list), "native attempt ledger required")
    rows = {}
    verification_counts = {actor: 0 for actor in _ACTORS}
    future = []
    for row in attempts:
        _require(isinstance(row, dict) and row.get("actor") in _ACTORS
                 and isinstance(row.get("result_id"), str) and type(row.get("verification")) is bool,
                 "native attempt identity is malformed")
        result_id = row["result_id"]
        _require(result_id not in rows, "duplicate native result identity")
        rows[result_id] = row
        actor = row["actor"]
        if row["verification"]:
            _require(result_id in future_ids, "verification occurred before allocation")
            verification_counts[actor] += 1
        matching = [event["payload"] for event in events[actor]
                    if event["kind"] == "tool_attempt" and event["payload"].get("result_id") == result_id]
        _require(len(matching) == 1 and all(matching[0].get(name) == row.get(name)
                 for name in ("app", "api", "arguments", "verification")), "native/history call differs")
        _require(row.get("status") == "returned", "settled example contains unresolved native execution")
        observation = row.get("observation")
        _require(isinstance(observation, dict) and "ok" in observation and
                 (type(observation["ok"]) is bool or observation["ok"] is None),
                 "native observation classification missing")
        observed = [event["payload"].get("observation") for event in events[actor]
                    if event["kind"] == "tool_observation" and event["payload"].get("result_id") == result_id]
        _require(observed == [observation], "native/history observation differs")
        if result_id in future_ids:
            kind = ("native_execution_error" if observation["ok"] is False else
                    "unparsed_native_return" if observation["ok"] is None else "json_native_return")
            future.append({"actor": actor, "result_id": result_id, "app": row["app"], "api": row["api"],
                           "verification": row["verification"], "return_kind": kind,
                           "observation": deepcopy(observation), "native_wall_seconds": row.get("native_wall_seconds")})
    _require(set(rows) == before_ids | future_ids, "native ledger and actor histories differ")
    _require(host.get("verification_used") == verification_counts
             and all(verification_counts[actor] <= allocation[actor] for actor in _ACTORS),
             "verification usage differs from the actual native ledger")
    return {"identity": {name: receipt[name] for name in _KEY_FIELDS},
            "features": {"actor_prefixes": prefixes, "verification_budget": receipt["verification_budget"],
                         "host_budget": deepcopy(receipt["host_budget"])},
            "treatment": {"allocation": deepcopy(allocation), "assignment_probability": probability,
                          "policy": deepcopy(receipt["allocation_policy"])},
            "labels": {"future_native_observations": future, "official_score": deepcopy(score)},
            "official_target_available": type(score["native_tgc"]) is bool}


def build_closure_dataset(manifest: Mapping[str, Any], *, manifest_identity: Mapping[str, Any],
                          plan: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]) -> dict:
    """Preserve the planned denominator and construct only causal train examples.

    Each attempt supplies execution_receipt, optional score_receipt, and the
    existing input_admission record. The plan is a list of split/task/method/seed
    identities. An absent execution stays missing; no native data is opened.
    """
    # Reject non-finite values and non-JSON inputs before emitting any dataset.
    json.dumps([manifest, manifest_identity, plan, attempts], allow_nan=False)
    _require(isinstance(manifest, Mapping) and manifest.get("schema") == "appworld_split_membership_v1",
             "admitted split manifest required")
    _require(isinstance(attempts, (list, tuple)), "attempt records must be a sequence")
    _require(isinstance(manifest_identity, Mapping), "consumed manifest identity required")
    memberships = validate_split_manifest(manifest)
    _sha(manifest_identity.get("sha256"), "consumed split manifest")
    _require(type(manifest_identity.get("bytes")) is int and manifest_identity["bytes"] > 0,
             "consumed split-manifest byte count required")
    train_ids = list(manifest["splits"]["train"]["task_ids"])
    _require(len(train_ids) == len(set(train_ids)), "duplicate native train membership")
    _require(isinstance(plan, (list, tuple)) and plan, "explicit nonempty planned roster required")
    planned = {}
    for row in plan:
        key = _key(row)
        _require(row["task_id"] in memberships["train"], "planned task is not in admitted train membership")
        _require(key not in planned, "duplicate planned attempt")
        planned[key] = {name: row[name] for name in _KEY_FIELDS}
    supplied = {}
    for record in attempts:
        _require(isinstance(record, Mapping) and isinstance(record.get("execution_receipt"), Mapping),
                 "each supplied attempt requires its execution receipt")
        key = _key(record["execution_receipt"])
        _require(key in planned and key not in supplied, "unplanned or duplicate execution receipt")
        supplied[key] = record
    rows, examples, bindings = [], [], []
    for key, identity in planned.items():
        if key not in supplied:
            rows.append({**identity, "coverage_status": "missing", "decision_reached": None,
                         "official_score": {"scoring_status": "absent", "native_tgc": None}})
            continue
        record, receipt = supplied[key], supplied[key]["execution_receipt"]
        _require(receipt.get("schema") == "appworld_development_execution_v1"
                 and receipt.get("scope") == "development_only" and receipt.get("formal_eligible") is False,
                 "canonical development execution required")
        _require(type(receipt.get("terminal")) is bool and receipt.get("native_tgc") is None
                 and receipt.get("scoring_status") == "not_run", "execution/score boundary differs")
        _require(receipt.get("target_load_ground_truth") is False
                 and receipt.get("native_kwargs", {}).get("load_ground_truth") is False,
                 "ground-truth-free collection required")
        if receipt["terminal"]:
            validate_scoring_inputs(receipt, manifest)
        else:
            _require(receipt.get("status") in {"running", "unknown"}, "unfinished execution status differs")
            _require(record.get("score_receipt") is None, "unfinished attempt cannot have an official score")
        binding = _binding(receipt, record.get("input_admission"), manifest, manifest_identity)
        bindings.append(binding)
        _require(binding == bindings[0], "mixed data/model/source/protocol identities")
        expected_allocation, probability = _policy(receipt, train_ids)
        score = _score(receipt, record.get("score_receipt"), manifest)
        status = receipt["status"]
        coverage = "unfinished" if not receipt["terminal"] else ("settled" if status in _SETTLED else status)
        _require(not receipt["terminal"] or status in TERMINAL_STATUSES, "unknown execution status vocabulary")
        host = receipt.get("host")
        _require(host is None or isinstance(host, Mapping), "host receipt is malformed")
        if host is not None and host.get("allocation") is not None:
            allocation = host["allocation"]
            _require(isinstance(allocation, dict) and set(allocation) == set(_ACTORS)
                     and all(type(allocation[actor]) is int for actor in _ACTORS)
                     and allocation == expected_allocation, "recorded allocation differs from policy")
        if coverage == "settled":
            _require(isinstance(host, Mapping) and host.get("status") == status
                     and receipt.get("membership_verified") is True,
                     "settled collaboration lacks matching admitted host evidence")
        elif host is not None and receipt["terminal"]:
            _require(host.get("status") == status, "terminal host and execution status differ")
        row = {**identity, "coverage_status": coverage, "execution_status": status,
               "terminal": receipt["terminal"], "usage_complete": receipt.get("usage_complete"),
               "decision_reached": None if host is None else host.get("allocation") is not None,
               "allocation": None if host is None else deepcopy(host.get("allocation")),
               "allocation_policy": deepcopy(receipt["allocation_policy"]),
               "recorded_assignment_probability": probability,
               "native_membership_index": train_ids.index(receipt["task_id"]),
               "host_status": None if host is None else host.get("status"),
               "host_stop_reason": None if host is None else host.get("stop_reason"),
               "host_native_task_completed": None if host is None else host.get("native_task_completed"),
               "official_score": score, "binding": binding,
               "cost": {name: deepcopy(receipt.get(name)) for name in
                        ("lm_call_count", "execution_wall_seconds", "measured_monetary_cost", "monetary_accounting")},
               "input_files": deepcopy(record.get("input_files", {}))}
        if coverage == "settled":
            example = _example(receipt, expected_allocation, probability, score)
            if example is not None:
                examples.append(example)
        rows.append(row)
    counts = Counter(row["coverage_status"] for row in rows)
    scored = [row for row in rows if type(row["official_score"]["native_tgc"]) is bool]
    return {"schema": "appworld_closure_dataset_v1", "scope": "train_only",
            "fitted_model": False, "formal_eligible": False,
            "split_manifest": dict(manifest_identity), "attempts": rows, "decision_examples": examples,
            "coverage": {"planned_attempts": len(planned), "supplied_attempts": len(supplied),
                         "by_coverage_status": dict(sorted(counts.items())),
                         "by_execution_status": dict(sorted(Counter(
                             row.get("execution_status", "missing") for row in rows).items())),
                         "allocated_attempts": sum(row["decision_reached"] is True for row in rows),
                         "allocation_null_attempts": sum(row["decision_reached"] is False for row in rows),
                         "settled_allocation_null_attempts": sum(row["coverage_status"] == "settled"
                             and row["decision_reached"] is False for row in rows),
                         "official_scored_attempts": len(scored),
                         "official_successes": sum(row["official_score"]["native_tgc"] for row in scored),
                         "decision_examples": len(examples),
                         "examples_with_official_target": sum(row["official_target_available"] for row in examples),
                         "planned_execution_complete": all(row["coverage_status"] == "settled" for row in rows),
                         "planned_scoring_complete": len(scored) == len(planned)}}


def _read_json(path: Path) -> tuple[Any, dict]:
    payload = path.read_bytes()
    value = json.loads(payload)
    json.dumps(value, allow_nan=False)
    return value, {"path": str(path.absolute()), "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="JSON list of split/task_id/method/seed objects")
    parser.add_argument("--attempts", type=Path, required=True,
                        help="JSON list of execution_receipt, score_receipt (or null), input_admission paths")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _require(not args.output.exists(), "output already exists")
        manifest, manifest_file = _read_json(args.split_manifest)
        plan, plan_file = _read_json(args.plan)
        descriptors, descriptor_file = _read_json(args.attempts)
        _require(isinstance(descriptors, list), "attempt descriptors must be a list")
        records = []
        for descriptor in descriptors:
            _require(isinstance(descriptor, dict)
                     and set(descriptor) == {"execution_receipt", "score_receipt", "input_admission"},
                     "attempt descriptor fields differ")
            record, identities = {}, {}
            for name, filename in descriptor.items():
                if filename is None:
                    _require(name == "score_receipt", "only the scoring receipt can be absent")
                    record[name] = None
                    continue
                _require(isinstance(filename, str) and filename, "receipt path must be nonempty text")
                path = Path(filename)
                if not path.is_absolute():
                    path = args.attempts.parent / path
                record[name], identities[name] = _read_json(path)
            record["input_files"] = identities
            records.append(record)
        result = build_closure_dataset(
            manifest, manifest_identity={name: manifest_file[name] for name in ("bytes", "sha256")},
            plan=plan, attempts=records)
        result["input_files"] = {"split_manifest": manifest_file, "plan": plan_file,
                                 "attempt_descriptors": descriptor_file}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"closure data rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "coverage": result["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

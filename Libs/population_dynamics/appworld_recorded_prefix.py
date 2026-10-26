"""Replay the recorded successful-authentication prefix through the existing host.

Runtime replay performs no native calls, inference, or hidden-state imports.
The preparation entry point freezes only the already observed prefix evidence.
The host supplies each real exchange; the original ten-attempt source prefix is
checked before any new generation. Fresh context IDs are the sole receipt change.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from .appworld_collaboration import (ACTORS, content_digest, database_journal_signature, generation_telemetry, validate_prefix_checkpoint)
from .provenance import freeze_json, sha256_file


PREFIX_ATTEMPTS = 10
LOGIN_ACTION = {"actor": "executor", "kind": "tool", "call_id": "executor_login_spotify",
                "app": "spotify", "api": "login"}
LOGIN_BOUNDARY = {
    "host_attempt_count": 10, "controller_actor_turns": 9,
    "host_attempt_limit": 30, "host_state": "DISCOVERY", "phase": "discovery",
    "next_actor": "planner", "discovery_calls_used": {"planner": 2, "executor": 3},
    "verification_calls_used": {"planner": 0, "executor": 0},
    "allocation_acknowledged": False,
}
RECEIPT_FIELDS = ("session_id", "prompt", "response", "model_revision", "usage", "inference_seconds")


class RecordedPrefixError(ValueError):
    """Recorded history or current replay differs before live continuation."""


class RecordedPrefix:
    def __init__(self, generation_attempts: list[dict], source_host_report: dict,
                 transport_records: list[dict], expected_boundary: dict | None = None, *, native_checkpoint_sha256: str | None = None):
        self.native_checkpoint_sha256 = native_checkpoint_sha256
        self.expected_boundary = deepcopy(LOGIN_BOUNDARY if expected_boundary is None else expected_boundary)
        if self.expected_boundary != LOGIN_BOUNDARY:
            raise RecordedPrefixError("RECORDED_PREFIX_LOGIN_BOUNDARY_INVALID")
        source = source_host_report.get("host", source_host_report)
        if (not isinstance(generation_attempts, list) or len(generation_attempts) < PREFIX_ATTEMPTS
                or not isinstance(source.get("host_attempts"), list)
                or len(source["host_attempts"]) < PREFIX_ATTEMPTS
                or source.get("host_attempt_limit") != LOGIN_BOUNDARY["host_attempt_limit"]):
            raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_INCOMPLETE")
        self.generations = deepcopy(generation_attempts[:PREFIX_ATTEMPTS])
        self.host_attempts = deepcopy(source["host_attempts"][:PREFIX_ATTEMPTS])
        if source.get("generation_attempts", generation_attempts)[:PREFIX_ATTEMPTS] != self.generations:
            raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_GENERATIONS_DIFFER")
        self.source_contexts = {}
        for index, (generation, attempt) in enumerate(zip(self.generations, self.host_attempts)):
            actor, session = generation.get("actor"), generation.get("session_id")
            if (actor not in ACTORS or not isinstance(session, str) or not session
                    or generation.get("status") != "GENERATED"
                    or any(key not in generation for key in RECEIPT_FIELDS)
                    or generation_telemetry(generation)["errors"]
                    or attempt.get("actor") != actor or attempt.get("attempt_index") != index
                    or attempt.get("producer_attempt_index") != index
                    or attempt.get("host_state") != "DISCOVERY"
                    or generation.get("model_revision") != source.get("model_revision")):
                raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_ATTEMPT_INVALID")
            if actor in self.source_contexts and self.source_contexts[actor] != session:
                raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_CONTEXT_CHANGED")
            self.source_contexts[actor] = session
        if len(set(self.source_contexts.values())) != len(ACTORS):
            raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_CONTEXTS_NOT_DISTINCT")
        malformed = self.host_attempts[5]
        try:
            json.loads(self.generations[5]["response"])
        except ValueError:
            pass
        else:
            raise RecordedPrefixError("RECORDED_PREFIX_SIXTH_ATTEMPT_NOT_MALFORMED")
        if (malformed.get("status") != "HOST_REJECTED"
                or malformed.get("error_code") != "LOCAL_ACTOR_RESPONSE_NOT_JSON"
                or malformed.get("submitted") is not False or malformed.get("acknowledged") is not False):
            raise RecordedPrefixError("RECORDED_PREFIX_MALFORMED_ATTEMPT_NOT_PRESERVED")
        last = self.host_attempts[-1]
        last_index = last.get("controller_request_index")
        if (type(last_index) is not int or last.get("status") != "ACCEPTED"
                or last.get("acknowledged") is not True
                or not isinstance(transport_records, list) or len(transport_records) < last_index + 2):
            raise RecordedPrefixError("RECORDED_PREFIX_LOGIN_ACK_MISSING")
        self.transport = []
        for row in transport_records[:last_index + 2]:
            if (not isinstance(row, dict) or not {"request", "response"} <= row.keys()
                    or set(row) - {"request", "response", "provider_telemetry"}):
                raise RecordedPrefixError("RECORDED_PREFIX_TRANSPORT_ROW_INVALID")
            if ("provider_telemetry" in row and row["provider_telemetry"]
                    != generation_telemetry(row["request"].get("generation"))):
                raise RecordedPrefixError("RECORDED_PREFIX_SOURCE_TELEMETRY_MISMATCH")
            self.transport.append(deepcopy({key: row[key] for key in ("request", "response")}))
        login = json.loads(self.generations[-1]["response"])
        if any(login.get(key) != value for key, value in LOGIN_ACTION.items()):
            raise RecordedPrefixError("RECORDED_PREFIX_LAST_ACTION_NOT_EXECUTOR_LOGIN")
        result = self.transport[last_index].get("response", {})
        native = result.get("result", {})
        if (result.get("ok") is not True or native.get("ok") is not True
                or not isinstance(native.get("result"), dict)
                or not isinstance(native["result"].get("access_token"), str)
                or not native["result"]["access_token"]):
            raise RecordedPrefixError("RECORDED_PREFIX_LOGIN_NOT_SUCCESSFUL")
        if self.transport[-1].get("request") != {"kind": "view", "actor": "executor"}:
            raise RecordedPrefixError("RECORDED_PREFIX_POST_LOGIN_VIEW_MISSING")
        for generation, attempt in zip(self.generations, self.host_attempts):
            if attempt.get("submitted") is not True:
                continue
            index = attempt.get("controller_request_index")
            row = self.transport[index]
            request = {**json.loads(generation["response"]),
                       "generation": {key: deepcopy(generation[key]) for key in RECEIPT_FIELDS}}
            if row != {"request": request, "response": attempt.get("controller_response")}:
                raise RecordedPrefixError("RECORDED_PREFIX_TRANSPORT_SOURCE_MISMATCH")
        self.branch_contexts = None
        self.exchange_count = self.checked_attempts = 0
        self.views = {}
        self.verified = False
        self._resume_state = None

    def bind(self, actor, config: dict) -> None:
        if (self.branch_contexts is not None or actor.attempts
                or any(len(actor.histories[name]) != 1 or actor.histories[name][0].get("role") != "system" for name in ACTORS)
                or set(actor.session_ids.values()) & set(self.source_contexts.values())
                or config["max_actor_turns"] != self.expected_boundary["host_attempt_limit"]
                or config["discovery_calls_per_actor"] != 8 or config["verification_budget"] != 2
                or any(row["model_revision"] != config["model_revision"] for row in self.generations)):
            raise RecordedPrefixError("RECORDED_PREFIX_FRESH_BRANCH_REQUIRED")
        self.branch_contexts = dict(actor.session_ids)

    def generate(self, actor, name: str, prompt: str) -> dict:
        index = len(actor.attempts)
        if self.verified or index >= PREFIX_ATTEMPTS or self.generations[index]["actor"] != name:
            raise RecordedPrefixError("RECORDED_PREFIX_ACTOR_ORDER_MISMATCH")
        try:
            return actor.replay_generation(name, prompt, self.generations[index])
        except ValueError as exc:
            raise RecordedPrefixError(str(exc)) from exc

    def expect_request(self, request: dict, index: int) -> None:
        if self.verified:
            return
        if index != self.exchange_count or index >= len(self.transport):
            raise RecordedPrefixError("RECORDED_PREFIX_EXCHANGE_ORDER_MISMATCH")
        expected = deepcopy(self.transport[index]["request"])
        if "generation" in expected:
            expected["generation"]["session_id"] = self.branch_contexts[expected["actor"]]
        if request != expected:
            raise RecordedPrefixError("RECORDED_PREFIX_REQUEST_MISMATCH")

    def observe_response(self, request: dict, response: dict, index: int) -> None:
        if self.verified:
            return
        if response != self.transport[index]["response"]:
            raise RecordedPrefixError("RECORDED_PREFIX_NATIVE_RESPONSE_MISMATCH")
        if request.get("kind") == "view":
            self.views[request["actor"]] = deepcopy(response["result"])
        self.exchange_count += 1

    @staticmethod
    def _state(host) -> dict:
        return {"config": deepcopy(host.config), "schemas": deepcopy(host.schemas),
                "actor_config": deepcopy(host.actor.config), "allocation": dict(host.allocation),
                "attempt_limit": host.attempt_limit, "evidence_table_after_prefix": host.evidence_table_after_prefix,
                "host_state": host.state, "phase": host.phase, "next_actor": host._select_actor(),
                "feedback": deepcopy(host.feedback), "histories": deepcopy(host.actor.histories),
                "contexts": dict(host.actor.session_ids), "attempts": deepcopy(host.attempts),
                "producer_attempts": deepcopy(host.actor.attempts),
                "request_count": host.controller_request_count, "pending": deepcopy(host._pending),
                "schemas_provided": dict(host._schemas_provided),
                "verification_used": dict(host.verification_used),
                "allocation_attempted": host.allocation_attempted,
                "allocation_acknowledged": host.allocation_acknowledged,
                "planner_verification_ids": list(host.planner_verification_ids),
                "planner_shared_ids": sorted(host.planner_shared_ids), "reply_slots": host.planner_reply_slots}

    def check_progress(self, host) -> bool:
        if self.verified:
            return False
        for index in range(self.checked_attempts, len(host.attempts)):
            if index >= PREFIX_ATTEMPTS or host.attempts[index] != self.host_attempts[index]:
                raise RecordedPrefixError("RECORDED_PREFIX_HOST_OUTCOME_MISMATCH")
        self.checked_attempts = len(host.attempts)
        if self.checked_attempts != PREFIX_ATTEMPTS:
            return False
        observed = {"host_attempt_count": len(host.attempts),
                    "controller_actor_turns": sum(row["submitted"] is True for row in host.attempts),
                    "host_attempt_limit": host.attempt_limit, "host_state": host.state,
                    "phase": host.phase, "next_actor": host._select_actor(),
                    "discovery_calls_used": {name: self.views.get(name, {}).get("discovery_calls_used") for name in ACTORS},
                    "verification_calls_used": dict(host.verification_used),
                    "allocation_acknowledged": host.allocation_acknowledged}
        if (observed != self.expected_boundary or self.exchange_count != len(self.transport)
                or host.controller_request_count != len(self.transport) or host._pending is not None
                or host.allocation_attempted or host.unknown_outcome
                or any(self.views[name]["allocated_verification_calls"] is not None for name in ACTORS)):
            raise RecordedPrefixError("RECORDED_PREFIX_BOUNDARY_MISMATCH")
        self.verified = True
        self._resume_state = self._state(host)
        return True

    def verify_resume(self, host) -> None:
        if not self.verified or self._resume_state != self._state(host):
            raise RecordedPrefixError("RECORDED_PREFIX_PAUSED_STATE_CHANGED")

    def report(self) -> dict:
        return {"status": "VERIFIED" if self.verified else "UNVERIFIED",
                "expected_boundary": deepcopy(self.expected_boundary),
                "source_actor_contexts": dict(self.source_contexts),
                "branch_actor_contexts": deepcopy(self.branch_contexts),
                "substituted_receipt_fields": ["session_id"],
                "checked_generations": self.checked_attempts, "checked_exchanges": self.exchange_count,
                "reused_controller_generation_request_indices": [row["controller_request_index"] for row in self.host_attempts if row["submitted"] is True],
                "historical_prefix_inference_seconds": sum(row["inference_seconds"] for row in self.generations),
                "historical_prefix_token_usage": {key: sum(row["usage"][key] for row in self.generations)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                "historical_inference_is_new_measurement": False,
                "required_native_checkpoint_sha256": self.native_checkpoint_sha256}


def actor_evidence_table(view: dict, own_native_calls: dict | None = None) -> dict:
    """Project only already-visible results; peer text remains an assertion."""
    rows = []
    own_native_calls = {} if own_native_calls is None else own_native_calls
    for call_id, value in sorted(view["results"].items()):
        payload = value.get("result") if isinstance(value, dict) else None
        fields = sorted(payload) if isinstance(payload, dict) else sorted({key for item in payload if isinstance(item, dict) for key in item}) if isinstance(payload, list) else []
        known = own_native_calls.get(call_id, {})
        rows.append({"call_id": call_id, "source_actor": known.get("actor"), "api": known.get("api"),
                     "native_status": "ok" if value.get("ok") is True else "error" if value.get("ok") is False else "unknown",
                     "error_type": value.get("error_type"), "returned_field_names": fields,
                     "source_reference": ["view", "results", call_id]})
    return {"native_results": rows,
            "peer_assertions": [{"sender": row["actor"], "text": row["text"],
                                 "shared_result_ids": deepcopy(row["share_results"])} for row in view["inbox"]]}


def prepare_prefix_protocol(source: Path, output: Path) -> dict:
    """Freeze the observed login prefix; subsequent source actions are excluded."""
    if output.exists():
        raise FileExistsError(output)
    inventories, bindings = {}, []
    for part in ("gpu_host", "cpu_controller", "inputs"):
        manifest = json.loads((source / part / "copy_out.json").read_text())
        if manifest.get("verified") is not True or str(manifest.get("job_id")) != "590688":
            raise ValueError("RECORDED_PREFIX_SOURCE_DELIVERY_INVALID")
        inventories[part] = {row["path"]: row for row in manifest["files"]}
    def bound(part, name):
        entry = inventories[part][name]
        path = source / part / name
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("RECORDED_PREFIX_SOURCE_ARTIFACT_CHANGED:" + name)
        bindings.append({"archive": str(source), "component": part, **entry})
        return path
    generations = json.loads(bound("gpu_host", "host/generation_attempts.json").read_text())[:PREFIX_ATTEMPTS]
    whole_host = json.loads(bound("gpu_host", "host/report.json").read_text())["host"]
    host = {key: whole_host[key] for key in ("host_attempt_limit", "model_revision")}
    host.update(host_attempts=whole_host["host_attempts"][:PREFIX_ATTEMPTS], generation_attempts=generations)
    transport = [json.loads(line) for line in bound("cpu_controller", "controller/transport.jsonl").read_text().splitlines()][:28]
    prefix = RecordedPrefix(generations, host, transport)
    config_name = "Libs/configs/neural_population_appworld_local_collaboration.yaml"
    config = yaml.safe_load(bound("inputs", config_name).read_text())
    native_config = yaml.safe_load(bound("cpu_controller", "controller/native_replay/call_005.yaml").read_text())
    if ({k:v for k,v in native_config.items() if k != "steps"}
            != {k:v for k,v in config["native"].items() if k != "steps"}):
        raise ValueError("RECORDED_PREFIX_NATIVE_CONFIGURATION_CHANGED")
    injected = {"label": "record_boundary", "save_state": "before_next_call"}
    if native_config["steps"].count(injected) != 1:
        raise ValueError("RECORDED_PREFIX_BOUNDARY_SAVE_IDENTITY_CHANGED")
    steps = [step for step in native_config["steps"] if step != injected]
    trace_name = "controller/native_replay/call_005/trace.jsonl"
    trace = [json.loads(line) for line in bound("cpu_controller", trace_name).read_text().splitlines()]
    observations = [row["output"] for row in trace if row["operation"] == "code"]
    if len(steps) != 6 or len(observations) != 6:
        raise ValueError("RECORDED_PREFIX_SIX_NATIVE_STEPS_REQUIRED")
    state_name = "controller/native_replay/call_005/native/experiments/outputs/reference/tasks/07b42fd_1/dbs"
    names = sorted(name for name in inventories["cpu_controller"]
                   if name.startswith(state_name + "/") and Path(name).suffix in {".jsonl", ".json"})
    if (state_name + "/model_hashes.json" not in names
            or not any(name.endswith(".jsonl") for name in names)):
        raise ValueError("RECORDED_PREFIX_COMPLETE_SOURCE_JOURNALS_REQUIRED")
    actual = {state_name + "/" + path.name for path in (source / "cpu_controller" / state_name).iterdir() if path.is_file()}
    if actual != set(names):
        raise ValueError("RECORDED_PREFIX_SOURCE_STATE_INVENTORY_DIFFERS")
    for name in names:
        bound("cpu_controller", name)
    journals = database_journal_signature(source / "cpu_controller" / state_name)
    checkpoint = {"schema": "appworld_recorded_native_prefix_v1", "task_id": config["native"]["task_id"],
        "native_configuration_sha256": content_digest({k:v for k,v in config["native"].items() if k != "steps"}),
        "native_step_count": 6, "steps_sha256": content_digest(steps),
        "observations_sha256": content_digest(observations), "database_journals": journals,
        "prefix_transport_requests": len(prefix.transport),
        "reused_generation_request_indices": [r["controller_request_index"] for r in host["host_attempts"] if r["submitted"] is True],
        "source_artifacts": bindings}
    validate_prefix_checkpoint(checkpoint, config["native"])
    output.mkdir(parents=True, exist_ok=False)
    freeze_json(output / "source_generations.json", generations)
    freeze_json(output / "source_host_report.json", host)
    with (output / "source_transport.jsonl").open("x") as stream:
        for row in transport:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    freeze_json(output / "native_checkpoint.json", checkpoint)
    inputs = {key: {"path": name, "sha256": sha256_file(output / name)} for key, name in
              (("generations", "source_generations.json"), ("host_report", "source_host_report.json"),
               ("transport", "source_transport.jsonl"), ("native_checkpoint", "native_checkpoint.json"))}
    settings = {"schema": "appworld_recorded_prefix_diagnostic_v1", "source_case_job": 590688,
        "prefix_attempts": 10, "remaining_attempts": 20, "allocation": {"planner":1,"executor":1},
        "maximum_task_prefix_pairs": 1, "arms": ["original_view", "actor_local_evidence_table"],
        "inputs": inputs,
        "comparison_scope": "one dependent task-prefix pair with matched attempt and verification limits; consumed tokens and wall costs measured separately",
        "endpoints": ["accepted_native_task_writes_and_outcomes", "generated_sharing_of_observed_authentication",
                      "source_supported_target_coverage", "finish_and_native_evaluation"],
        "stop_rule": "one pair at finish or original 30-attempt cap; no tuning or automatic retry",
        "scientific_control_transfer_qualified": False}
    (output / "protocol.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    return {"status": "PREPARED_RECORDED_PREFIX_DIAGNOSTIC", "source_case_job": 590688,
        "prefix_generations": len(generations), "prefix_transport_requests": len(transport),
        "native_steps": len(steps), "database_journal_files": len(journals["files"]),
        "native_model_hash_file_is_only_mutation_counters": True,
        "source_bindings_verified": len(bindings), "protocol_sha256": sha256_file(output / "protocol.yaml"),
        "native_checkpoint_sha256": content_digest(checkpoint), "source_simulation_or_inference_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_prefix_protocol(args.source_archive, args.output)
    freeze_json(args.output / "preparation.json", report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Two-agent AppWorld development execution through typed, recorded requests.

Model contexts are supplied by the host, not simulated by this module. Each tool
call reuses the native reference CLI in a fresh process; replay overhead is
recorded separately from the agent's call budget. No agent sees native evaluation
or model hashes. This development runner does not qualify a scientific claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import select
import subprocess
import sys
import time
from typing import Any

import yaml

from .appworld_reference import NAME, validate_config as validate_native
from .provenance import canonical_json, freeze_json, runtime_manifest, sha256_file

ACTORS = ("planner", "executor")


def validate_config(config: dict) -> dict:
    required = {"native", "provider", "max_actor_turns", "discovery_calls_per_actor",
                "verification_budget", "idle_timeout_seconds"}
    provider = config.get("provider")
    expected = required | ({"model_revision"} if provider == "local_transformers" else set())
    if set(config) != expected or provider not in {"host_collaboration", "fixture", "local_transformers"}:
        raise ValueError("COLLABORATION_CONFIG_INVALID")
    if provider == "local_transformers" and (not isinstance(config["model_revision"], str)
                                               or not config["model_revision"].strip()):
        raise ValueError("LOCAL_MODEL_REVISION_REQUIRED")
    validate_native(config["native"])
    for name in required - {"native", "provider"}:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError("COLLABORATION_POSITIVE_LIMIT_REQUIRED:" + name)
    if config["max_actor_turns"] > 48 or config["verification_budget"] > 8:
        raise ValueError("COLLABORATION_DEVELOPMENT_BOUNDS_EXCEEDED")
    return config


def generation_telemetry(receipt: Any) -> dict:
    """Validate independently observable cost components, including rejected turns."""
    receipt = receipt if isinstance(receipt, dict) else {}
    usage, seconds, errors = None, None, []
    candidate = receipt.get("usage")
    if "usage" not in receipt:
        errors.append("GENERATION_TOKEN_USAGE_MISSING")
    elif (not isinstance(candidate, dict)
          or set(candidate) != {"prompt_tokens", "completion_tokens", "total_tokens"}
          or any(type(value) is not int or value < 0 for value in candidate.values())
          or candidate["total_tokens"] != candidate["prompt_tokens"] + candidate["completion_tokens"]):
        errors.append("GENERATION_TOKEN_USAGE_INVALID")
    else:
        usage = dict(candidate)
    candidate = receipt.get("inference_seconds")
    if "inference_seconds" not in receipt:
        errors.append("GENERATION_INFERENCE_SECONDS_MISSING")
    else:
        try:
            valid_seconds = type(candidate) in {int, float} and math.isfinite(candidate) and candidate >= 0
        except OverflowError:
            valid_seconds = False
        if valid_seconds:
            seconds = candidate
        else:
            errors.append("GENERATION_INFERENCE_SECONDS_INVALID")
    status = "COMPLETE" if not errors else "PARTIAL" if usage is not None or seconds is not None else "UNKNOWN"
    return {"status": status, "usage": usage, "inference_seconds": seconds, "errors": errors}


def resolve_arguments(value: Any, results: dict, available: set[str]) -> Any:
    if isinstance(value, dict):
        if "$result" in value:
            if set(value) != {"$result", "path"} or value["$result"] not in available:
                raise ValueError("RESULT_NOT_VISIBLE_TO_ACTOR")
            result = deepcopy(results[value["$result"]])
            if not result["ok"] or not isinstance(value["path"], list):
                raise ValueError("SUCCESSFUL_RESULT_PATH_REQUIRED")
            result = result["result"]
            for key in value["path"]:
                if type(key) not in {str, int}:
                    raise ValueError("RESULT_PATH_KEY_INVALID")
                result = result[key]
            return result
        return {key: resolve_arguments(item, results, available) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(item, results, available) for item in value]
    return value


def api_step(call_id: str, app: str, api: str, arguments: dict) -> dict:
    if any(NAME.fullmatch(name) is None for name in (call_id, app, api)):
        raise ValueError("API_IDENTIFIER_INVALID")
    encoded = repr(json.dumps(arguments, allow_nan=False))
    # API errors are observations for the model. They remain explicit failures
    # in the event ledger and still consume a tool attempt. Interpreter/process
    # failures remain failures of the native reference CLI.
    code = ("try:\n"
            f"    collab_response = apis.{app}.{api}(**json.loads({encoded}))\n"
            "except Exception as collab_error:\n"
            "    print(json.dumps({'ok': False, 'error_type': type(collab_error).__name__, 'error': str(collab_error)}, sort_keys=True))\n"
            "else:\n"
            "    print(json.dumps({'ok': True, 'result': collab_response}, sort_keys=True))\n")
    return {"label": call_id, "code": code}


def check_prefix_replay(previous_observations: list, observations: list,
                        previous_hashes: dict, boundary_hashes: dict) -> None:
    if observations != previous_observations:
        raise RuntimeError("NATIVE_PREFIX_VISIBLE_REPLAY_MISMATCH")
    if boundary_hashes != previous_hashes:
        raise RuntimeError("NATIVE_PREFIX_DATABASE_REPLAY_MISMATCH")


def database_journal_signature(directory: Path) -> dict:
    """Bind saved SQL changes; native model_hashes are mutation counters only.

    Identical journals represent identical source changes against the same task
    base. This does not claim a Python/RNG checkpoint or a full database dump.
    """
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise RuntimeError("NATIVE_DATABASE_CHANGE_JOURNALS_MISSING")
    return {"format": "appworld_sql_change_journals_v1",
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths},
            "model_mutation_counters": json.loads((directory / "model_hashes.json").read_text())}


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_prefix_checkpoint(checkpoint: dict, native: dict) -> dict:
    expected = {"schema", "task_id", "native_configuration_sha256", "native_step_count",
                "steps_sha256", "observations_sha256", "database_journals",
                "prefix_transport_requests", "reused_generation_request_indices", "source_artifacts"}
    if (not isinstance(checkpoint, dict) or set(checkpoint) != expected
            or checkpoint["schema"] != "appworld_recorded_native_prefix_v1"
            or checkpoint["task_id"] != native["task_id"]
            or checkpoint["native_configuration_sha256"] != content_digest({k:v for k,v in native.items() if k != "steps"})
            or checkpoint["native_step_count"] != 6 or checkpoint["prefix_transport_requests"] != 28
            or not isinstance(checkpoint["database_journals"], dict)
            or checkpoint["database_journals"].get("format") != "appworld_sql_change_journals_v1"
            or not checkpoint["database_journals"].get("files")):
        raise ValueError("RECORDED_NATIVE_PREFIX_CONTRACT_INVALID")
    indices = checkpoint["reused_generation_request_indices"]
    if (not isinstance(indices, list) or len(indices) != 9 or indices != sorted(set(indices))
            or any(type(i) is not int or not 0 <= i < 28 for i in indices)):
        raise ValueError("RECORDED_NATIVE_PREFIX_REQUEST_INDICES_INVALID")
    return deepcopy(checkpoint)


class PrefixReplay:
    """Native task state comes from identical-prefix replay, not Python pickling."""

    def __init__(self, native: dict, data_root: Path, output: Path,
                 costs: list[dict] | None = None, *, prefix_checkpoint: dict | None = None):
        self.native, self.data_root, self.output = deepcopy(native), data_root, output
        self.steps = [{"label": "initial_task", "code": "import json\nprint(json.dumps(apis.supervisor.show_active_task(), sort_keys=True))"}]
        self.observations: list[str] = []
        self.last_run: Path | None = None
        self.model_hashes: dict | None = None
        self.prefix_checkpoint = None if prefix_checkpoint is None else validate_prefix_checkpoint(prefix_checkpoint, self.native)
        self.checkpoint_verification = {"status": "NOT_REQUESTED" if prefix_checkpoint is None else "NOT_REACHED"}
        # The server owns the ledger so failed initialization retains its cost.
        self.costs = [] if costs is None else costs
        self.output.mkdir(parents=True, exist_ok=False)
        self.task = self._run(self.steps, read_only=False)

    def _command(self, name: str, args: list[str], expected: int = 2) -> None:
        started = time.monotonic()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        row = {"command": args, "exit_code": None}
        try:
            with (self.output / (name + ".log")).open("x", encoding="utf-8") as log:
                result = subprocess.run([sys.executable, "-m", "Libs.population_dynamics.appworld_reference", *args],
                                        stdout=log, stderr=subprocess.STDOUT)
            row["exit_code"] = result.returncode
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            # Native children run serially; these are CPU seconds, not wall time.
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row.update(wall_seconds=time.monotonic() - started,
                       cpu_user_seconds=after.ru_utime - before.ru_utime,
                       cpu_system_seconds=after.ru_stime - before.ru_stime)
            self.costs.append(row)
        if result.returncode != expected:
            raise RuntimeError("NATIVE_REFERENCE_FAILED:" + name)

    @staticmethod
    def _hashes(run: Path, task: str, suffix: str) -> dict:
        return database_journal_signature(run / "native/experiments/outputs/reference/tasks" / task / suffix)

    def _run(self, steps: list[dict], read_only: bool) -> Any:
        index = len(steps) - 1
        name = f"call_{index:03d}"
        run = self.output / name
        config = deepcopy(self.native)
        config["steps"] = deepcopy(steps)
        if self.last_run is not None:
            config["steps"].insert(-1, {"label": "record_boundary", "save_state": "before_next_call"})
        path = self.output / (name + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        self._command(name, ["run", "--config", str(path), "--data-root", str(self.data_root), "--output", str(run)])
        rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
        observations = [row["output"] for row in rows if row["operation"] == "code"]
        if self.last_run is not None:
            boundary = self._hashes(run, config["task_id"], "checkpoints/before_next_call")
            check_prefix_replay(self.observations, observations[:-1], self.model_hashes, boundary)
        hashes = self._hashes(run, config["task_id"], "dbs")
        if read_only and hashes != self.model_hashes:
            raise RuntimeError("VERIFICATION_GET_CHANGED_TASK_DATABASE")
        if self.prefix_checkpoint is not None and len(steps) == self.prefix_checkpoint["native_step_count"]:
            observed = {"steps_sha256": content_digest(steps), "observations_sha256": content_digest(observations),
                        "database_journals": hashes}
            self.checkpoint_verification = {"status": "FAILED", "native_step_count": len(steps),
                "observed": observed, "evidence_scope": "source SQL changes against the same task base; not a Python/RNG checkpoint"}
            if any(observed[key] != self.prefix_checkpoint[key] for key in observed):
                raise RuntimeError("RECORDED_NATIVE_PREFIX_CHECKPOINT_MISMATCH")
            self.checkpoint_verification["status"] = "PASS"
        if (self.prefix_checkpoint is not None and len(steps) > self.prefix_checkpoint["native_step_count"]
                and self.checkpoint_verification["status"] != "PASS"):
            raise RuntimeError("RECORDED_NATIVE_PREFIX_NOT_VERIFIED")
        self.last_run, self.model_hashes, self.observations = run, hashes, observations
        return json.loads(observations[-1])

    def call(self, call_id: str, app: str, api: str, arguments: dict, verification: bool) -> dict:
        step = api_step(call_id, app, api, arguments)
        result = self._run([*self.steps, step], read_only=verification)
        self.steps.append(step)
        return result

    def evaluate(self) -> dict:
        assert self.last_run is not None
        output = self.output / "terminal_evaluation"
        self._command("terminal_evaluation", ["evaluate", "--run", str(self.last_run), "--output", str(output)])
        return json.loads((output / "evaluation.json").read_text())["native_evaluation"]


def tool_phase_error(phase: str, actor: str, method: str, api: str) -> str | None:
    """The controller's existing phase/method permissions, independent of arguments."""
    if phase == "awaiting_allocation":
        return "CONTROL_ALLOCATION_REQUIRED"
    if phase == "discovery" and method != "GET" and api != "login":
        return "TASK_WRITES_REQUIRE_PROPOSAL_AND_VERIFICATION"
    if phase == "verification" and method != "GET":
        return "VERIFICATION_REQUIRES_SOURCE_DECLARED_GET"
    if phase == "execution" and (actor != "executor" or method == "GET" or api == "login"):
        return "POST_BUDGET_CALL_NOT_AN_EXECUTOR_TASK_WRITE"
    return None


class CollaborationCase:
    """Keep recipient visibility and query allowances separate from native state."""

    def __init__(self, config: dict, schemas: dict, backend: PrefixReplay):
        self.config, self.schemas, self.backend = validate_config(config), schemas, backend
        self.phase = "discovery"
        self.results: dict[str, dict] = {}
        self.visible = {actor: set() for actor in ACTORS}
        self.inbox: dict[str, list] = {actor: [] for actor in ACTORS}
        self.discovery_used = {actor: 0 for actor in ACTORS}
        self.allocated: dict[str, int] | None = None
        self.verification_used = {actor: 0 for actor in ACTORS}
        self.turns = 0
        self.identities: dict[str, str] = {}
        self.events: list[dict] = []
        self.finished = False

    def view(self, actor: str) -> dict:
        if actor not in ACTORS:
            raise ValueError("UNKNOWN_ACTOR")
        return {"actor": actor, "task": self.backend.task, "phase": self.phase,
                "inbox": deepcopy(self.inbox[actor]),
                "results": {key: deepcopy(self.results[key]) for key in sorted(self.visible[actor])},
                "allocated_verification_calls": None if self.allocated is None else self.allocated[actor],
                "used_verification_calls": self.verification_used[actor],
                "discovery_calls_used": self.discovery_used[actor]}

    def _receipt(self, request: dict) -> dict:
        receipt = request.get("generation")
        local = self.config["provider"] == "local_transformers"
        required = {"session_id", "prompt", "response", "model_revision"}
        if local:
            required |= {"usage", "inference_seconds"}
        if not isinstance(receipt, dict) or set(receipt) != required:
            raise ValueError("EXACT_GENERATION_RECEIPT_REQUIRED")
        if not all(isinstance(receipt[key], str) and receipt[key] for key in ("session_id", "prompt", "response")):
            raise ValueError("GENERATION_RECEIPT_EMPTY")
        if local:
            if receipt["model_revision"] != self.config["model_revision"]:
                raise ValueError("LOCAL_MODEL_REVISION_MISMATCH")
            telemetry = generation_telemetry(receipt)
            if telemetry["errors"]:
                raise ValueError(telemetry["errors"][0])
        payload = {key: value for key, value in request.items() if key != "generation"}
        if json.loads(receipt["response"]) != payload:
            raise ValueError("ACTOR_RESPONSE_WAS_MODIFIED")
        actor = request["actor"]
        if actor in self.identities and self.identities[actor] != receipt["session_id"]:
            raise ValueError("ACTOR_CONTEXT_ID_CHANGED")
        if any(other != actor and session == receipt["session_id"] for other, session in self.identities.items()):
            raise ValueError("ACTORS_REQUIRE_DISTINCT_CONTEXTS")
        self.identities[actor] = receipt["session_id"]
        return payload

    def handle(self, request: dict) -> dict:
        if self.finished:
            raise ValueError("CASE_ALREADY_FINISHED")
        if request.get("kind") == "view" and set(request) == {"kind", "actor"}:
            return self.view(request["actor"])
        if request.get("kind") == "allocate":
            if set(request) != {"kind", "allocation"} or self.phase != "awaiting_allocation":
                raise ValueError("ALLOCATION_REQUIRES_FROZEN_PROPOSAL_BOUNDARY")
            allocation = request["allocation"]
            if not isinstance(allocation, dict) or set(allocation) != set(ACTORS):
                raise ValueError("ALLOCATION_ACTORS_INVALID")
            if any(type(n) is not int or n < 0 for n in allocation.values()) or sum(allocation.values()) != self.config["verification_budget"]:
                raise ValueError("ALLOCATION_TOTAL_MUST_MATCH_FROZEN_BUDGET")
            self.allocated = dict(allocation)
            self.phase = "verification"
            self.events.append(deepcopy(request))
            return {"phase": self.phase, "allocation": self.allocated}
        actor = request.get("actor")
        if actor not in ACTORS:
            raise ValueError("UNKNOWN_ACTOR")
        if self.turns >= self.config["max_actor_turns"]:
            raise ValueError("ACTOR_TURN_LIMIT_REACHED")
        payload = self._receipt(request)
        self.turns += 1
        event = {"index": len(self.events), "request": deepcopy(request), "phase_before": self.phase}
        self.events.append(event)
        try:
            if payload["kind"] in {"message", "proposal"}:
                allowed = {"actor", "kind", "recipient", "text", "share_results"}
                if set(payload) != allowed or payload["recipient"] not in ACTORS or payload["recipient"] == actor:
                    raise ValueError("MESSAGE_FIELDS_INVALID")
                if not isinstance(payload["text"], str) or not payload["text"].strip() or not isinstance(payload["share_results"], list):
                    raise ValueError("MESSAGE_CONTENT_REQUIRED")
                if any(key not in self.visible[actor] for key in payload["share_results"]):
                    raise ValueError("CANNOT_SHARE_UNSEEN_RESULT")
                if payload["kind"] == "proposal":
                    if actor != "planner" or self.phase != "discovery":
                        raise ValueError("ONE_PLANNER_PROPOSAL_REQUIRED_BEFORE_WRITES")
                    self.phase = "awaiting_allocation"
                self.visible[payload["recipient"]].update(payload["share_results"])
                self.inbox[payload["recipient"]].append(deepcopy(payload))
                result = {"delivered_to": payload["recipient"], "phase": self.phase}
            elif payload["kind"] == "tool":
                if set(payload) != {"actor", "kind", "call_id", "app", "api", "arguments"}:
                    raise ValueError("TOOL_FIELDS_INVALID")
                call_id, app, api = payload["call_id"], payload["app"], payload["api"]
                if (not all(isinstance(x, str) and NAME.fullmatch(x) for x in (call_id, app, api))
                        or call_id in self.results or call_id in {"initial_task", "record_boundary"}):
                    raise ValueError("UNIQUE_API_IDENTIFIERS_REQUIRED")
                if app not in self.schemas or api not in self.schemas[app]:
                    raise ValueError("API_NOT_IN_PUBLIC_TOOL_SCHEMA")
                schema = self.schemas[app][api]
                method = schema["method"].upper()
                if not isinstance(payload["arguments"], dict):
                    raise ValueError("API_ARGUMENT_MAPPING_REQUIRED")
                parameters = {p["name"]: p for p in schema["parameters"]}
                if set(payload["arguments"]) - set(parameters) or any(p["required"] and name not in payload["arguments"] for name, p in parameters.items()):
                    raise ValueError("API_ARGUMENT_NAMES_INVALID")
                args = resolve_arguments(payload["arguments"], self.results, self.visible[actor])
                verification = self.phase == "verification"
                phase_error = tool_phase_error(self.phase, actor, method, api)
                if phase_error is not None:
                    raise ValueError(phase_error)
                if self.phase == "discovery":
                    if self.discovery_used[actor] >= self.config["discovery_calls_per_actor"]:
                        raise ValueError("DISCOVERY_CALL_BUDGET_EXHAUSTED")
                    self.discovery_used[actor] += 1
                elif verification:
                    assert self.allocated is not None
                    if self.verification_used[actor] >= self.allocated[actor]:
                        raise ValueError("VERIFICATION_CALL_BUDGET_EXHAUSTED")
                    # Charge the attempt before native execution, including API errors.
                    self.verification_used[actor] += 1
                event.update(http_method=method, verification=verification)
                try:
                    result = self.backend.call(call_id, app, api, args, verification)
                except ValueError as exc:
                    raise RuntimeError("NATIVE_OPERATION_VALUE_ERROR:" + str(exc)) from exc
                self.results[call_id] = result
                self.visible[actor].add(call_id)
                if verification and self.verification_used == self.allocated:
                    self.phase = "execution"
            elif payload["kind"] == "finish":
                if set(payload) != {"actor", "kind"} or actor != "executor" or self.phase != "execution":
                    raise ValueError("FINISH_REQUIRES_EXECUTOR_AFTER_VERIFICATION")
                try:
                    evaluation = self.backend.evaluate()
                except ValueError as exc:
                    raise RuntimeError("NATIVE_OPERATION_VALUE_ERROR:" + str(exc)) from exc
                self.finished = True
                result = {"case_finished": True, "native_evaluation": evaluation}
            else:
                raise ValueError("UNKNOWN_ACTOR_REQUEST_KIND")
            event.update(accepted=True, result=deepcopy(result), phase_after=self.phase)
            return result
        except Exception as exc:
            event.update(accepted=False, error_type=type(exc).__name__, error=str(exc))
            raise


def serve(config: dict, schemas: dict, data_root: Path, output: Path, command: list[str], *, prefix_checkpoint: dict | None = None) -> int:
    started = time.monotonic()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    request_wait_seconds = 0.0
    native_costs: list[dict] = []
    generation_count = inference_count = usage_count = 0
    request_count = reused_generation_count = 0
    reused_inference_seconds = 0.0
    reused_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    inference_seconds = 0.0
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    output.mkdir(parents=True, exist_ok=False)
    backend = None
    case = None
    status, code, error = "FAILED", 1, None
    try:
        validate_config(config)
        checkpoint = None if prefix_checkpoint is None else validate_prefix_checkpoint(prefix_checkpoint, config["native"])
        reused_indices = set() if checkpoint is None else set(checkpoint["reused_generation_request_indices"])
        if checkpoint is None:
            backend = PrefixReplay(config["native"], data_root, output / "native_replay", native_costs)
        else:
            backend = PrefixReplay(config["native"], data_root, output / "native_replay", native_costs, prefix_checkpoint=checkpoint)
        case = CollaborationCase(config, schemas, backend)
        ready = {"ready": True, "task": backend.task, "actors": ACTORS}
        if checkpoint is not None:
            ready["native_prefix_checkpoint_sha256"] = content_digest(checkpoint)
        print(json.dumps(ready), flush=True)
        with (output / "transport.jsonl").open("x", encoding="utf-8") as transport:
            while not case.finished:
                wait_started = time.monotonic()
                try:
                    if not select.select([sys.stdin], [], [], config["idle_timeout_seconds"])[0]:
                        raise TimeoutError("MODEL_HOST_INPUT_TIMEOUT")
                    line = sys.stdin.readline()
                finally:
                    # Includes model host coordination and transport. It is not
                    # an inference timer and is already inside session wall time.
                    request_wait_seconds += time.monotonic() - wait_started
                if not line:
                    raise EOFError("MODEL_HOST_CLOSED_BEFORE_FINISH")
                request = json.loads(line)
                request_index = request_count
                request_count += 1
                telemetry = None
                reused = request_index in reused_indices
                if (config["provider"] == "local_transformers" and isinstance(request, dict)
                        and ("generation" in request or request.get("kind") not in {"view", "allocate"})):
                    telemetry = generation_telemetry(request.get("generation"))
                    if reused:
                        reused_generation_count += 1
                        if telemetry["errors"]:
                            raise ValueError("RECORDED_NATIVE_PREFIX_TELEMETRY_INVALID")
                        reused_inference_seconds += telemetry["inference_seconds"]
                        for name, value in telemetry["usage"].items():
                            reused_token_usage[name] += value
                    else:
                        generation_count += 1
                        if telemetry["inference_seconds"] is not None:
                            inference_count += 1
                            inference_seconds += telemetry["inference_seconds"]
                        if telemetry["usage"] is not None:
                            usage_count += 1
                            for name, value in telemetry["usage"].items():
                                token_usage[name] += value
                request_error = None
                try:
                    response = {"ok": True, "result": case.handle(request)}
                except ValueError as exc:
                    response = {"ok": False, "error": str(exc)}
                except Exception as exc:
                    response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
                    request_error = exc
                row = {"request": request, "response": response}
                if telemetry is not None:
                    row["provider_telemetry"] = telemetry
                    if prefix_checkpoint is not None:
                        row["provider_telemetry_is_reused"] = reused
                try:
                    encoded = json.dumps(row, allow_nan=False)
                except ValueError:
                    # Nonfinite input is rejected but must remain inspectable in
                    # valid JSON. Preserve the exact source text as a string.
                    row.update(request=None, request_raw_json=line)
                    encoded = json.dumps(row, allow_nan=False)
                transport.write(encoded + "\n")
                transport.flush()
                print(json.dumps(response, allow_nan=False), flush=True)
                if request_error is not None:
                    raise request_error
        status, code = "DEVELOPMENT_EXECUTION_COMPLETE", 2
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
    finally:
        cpu_ended = resource.getrusage(resource.RUSAGE_SELF)
        provider_complete = generation_count > 0 and inference_count == usage_count == generation_count
        provider_status = ("COMPLETE_FOR_OBSERVED_GENERATIONS" if provider_complete else
                           "PARTIAL" if inference_count or usage_count else "UNKNOWN")
        costs = {
            "schema_version": 1,
            "status": ("PARTIAL_HOST_ACTOR_NATIVE_COST_UNQUALIFIED" if provider_complete else
                       "PARTIAL_MISSING_OR_INVALID_PROVIDER_TELEMETRY" if generation_count else
                       "PARTIAL_MISSING_PROVIDER_TELEMETRY"),
            "measurement_scope": "serve_entry_through_terminal_execution_before_artifact_writes",
            "server_session_wall_seconds": time.monotonic() - started,
            "request_wait_wall_seconds": request_wait_seconds,
            "server_process_cpu_user_seconds": cpu_ended.ru_utime - cpu_started.ru_utime,
            "server_process_cpu_system_seconds": cpu_ended.ru_stime - cpu_started.ru_stime,
            "native_command_count": len(native_costs),
            "native_wall_seconds_sum": sum(row["wall_seconds"] for row in native_costs),
            "native_cpu_user_seconds_sum": sum(row["cpu_user_seconds"] for row in native_costs),
            "native_cpu_system_seconds_sum": sum(row["cpu_system_seconds"] for row in native_costs),
            "provider_measurement_scope": "host_reported_valid_components_of_received_generation_requests_only",
            "provider_measurement_excludes": ["model_loading", "context_setup", "host_coordination", "unreported_generation_attempts"],
            "provider_telemetry_status": provider_status,
            "provider_generation_request_count": generation_count,
            "provider_inference_observation_count": inference_count,
            "provider_token_usage_observation_count": usage_count,
            "provider_inference_seconds": inference_seconds if inference_count else None,
            "provider_token_usage": token_usage if usage_count else None,
            "complete_host_actor_native_cost_qualified": False,
        }
        if prefix_checkpoint is not None:
            costs.update(provider_measurement_scope="new received generations excluding recorded prefix; every native replay measured anew",
                recorded_prefix_generation_requests=reused_generation_count,
                historical_prefix_inference_seconds=reused_inference_seconds,
                historical_prefix_token_usage=reused_token_usage,
                historical_inference_is_new_measurement=False)
            if not generation_count and reused_generation_count:
                costs["provider_telemetry_status"] = "NOT_RUN_RECORDED_PREFIX_ONLY"
        summary = {"status": status, "error": error, "scientific_control_transfer_qualified": False,
                   "generative_transfer_qualified": False, "formal_model_revision_verified": False,
                   "provider": config["provider"],
                   "agent_execution_semantics": "host_supplied_distinct_contexts",
                   "agent_execution_performed": bool(case and config["provider"] in {"host_collaboration", "local_transformers"} and len(case.identities) == 2),
                   "actor_contexts": {} if case is None else case.identities,
                   "allocation": None if case is None else case.allocated,
                   "verification_calls_used": {} if case is None else case.verification_used,
                   "cost_accounting": costs,
                   "recorded_native_prefix": None if prefix_checkpoint is None else getattr(
                       backend, "checkpoint_verification", {"status": "NOT_REACHED"}),
                   "database_replay_evidence": "saved SQL change journals against the same task base; mutation counters are auxiliary"}
        freeze_json(output / "summary.json", summary)
        freeze_json(output / "events.json", [] if case is None else case.events)
        freeze_json(output / "replay_costs.json", native_costs)
        freeze_json(output / "manifest.json", {**runtime_manifest(config, command, code),
            "artifacts": {name: sha256_file(output / name) for name in ("summary.json", "events.json", "replay_costs.json")}})
        print(json.dumps({"terminal": summary}, allow_nan=False), flush=True)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--api-schemas", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path)
    args = parser.parse_args(argv)
    config = validate_config(yaml.safe_load(args.config.read_text()))
    schemas = json.loads(args.api_schemas.read_text())
    checkpoint = None if args.prefix_checkpoint is None else json.loads(args.prefix_checkpoint.read_text())
    return serve(config, schemas, args.data_root.resolve(), args.output.resolve(), sys.argv, prefix_checkpoint=checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())

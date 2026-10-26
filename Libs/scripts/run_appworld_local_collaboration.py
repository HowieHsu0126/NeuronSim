"""Run the existing AppWorld collaboration controller with pinned local actors.

The launcher supplies the CPU controller command. This entry point owns GPU
model loading, a single outstanding JSON request, and host-side cost records.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import select
import subprocess
import sys
import time

import yaml

from Libs.population_dynamics.appworld_action_decoder import validate_action_decoder_bounds
from Libs.population_dynamics.appworld_collaboration import content_digest, validate_prefix_checkpoint, validate_config as validate_collaboration
from Libs.population_dynamics.appworld_local_actor import (
    ACTORS, LocalTransformersActor, model_identity, validate_config as validate_actor, verify_snapshot,
)
from Libs.population_dynamics.appworld_local_host import LocalActorHost
from Libs.population_dynamics.appworld_recorded_prefix import RecordedPrefix
from Libs.population_dynamics.provenance import freeze_json, runtime_manifest, sha256_file


class ProcessJsonChannel:
    """Line JSON over a child process; an uncertain write is never retried."""

    def __init__(self, command: list[str], stderr_path: Path, timeout_seconds: float):
        if (not isinstance(command, list) or not command
                or any(not isinstance(value, str) or not value or "\0" in value for value in command)):
            raise ValueError("CONTROLLER_COMMAND_INVALID")
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("CONTROLLER_RESPONSE_TIMEOUT_INVALID")
        self.timeout_seconds = timeout_seconds
        self.pending = False
        self.last_pending_request = None
        self.sent_requests = self.received_messages = 0
        self.invalid_stdout_line = None
        self._buffer = bytearray()
        self.stderr_stream = stderr_path.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.stderr_stream, bufsize=0, start_new_session=True)
        except Exception:
            self.stderr_stream.close()
            raise

    def send(self, request: dict) -> None:
        if self.pending:
            raise RuntimeError("CONTROLLER_REQUEST_ALREADY_PENDING")
        payload = (json.dumps(request, allow_nan=False) + "\n").encode("utf-8")
        self.pending = True
        self.last_pending_request = request
        assert self.process.stdin is not None
        written = 0
        while written < len(payload):
            written += os.write(self.process.stdin.fileno(), payload[written:])
        self.sent_requests += 1

    def receive(self) -> dict:
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.timeout_seconds
        while b"\n" not in self._buffer:
            remaining = max(0, deadline - time.monotonic())
            if not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("CONTROLLER_RESPONSE_UNRESOLVED")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise EOFError("CONTROLLER_STDOUT_CLOSED")
            self._buffer.extend(chunk)
        line, _, tail = self._buffer.partition(b"\n")
        self._buffer = bytearray(tail)
        try:
            message = json.loads(line.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("CONTROLLER_MESSAGE_NOT_OBJECT")
        except ValueError:
            self.invalid_stdout_line = {"bytes_hex": line.hex()}
            raise
        self.received_messages += 1
        self.pending = False
        self.last_pending_request = None
        return message

    def close_and_wait(self) -> dict:
        """Close the case once; late output is evidence, never a retry trigger."""
        signals = []
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.stdin = None
        try:
            try:
                remaining, _ = self.process.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                signals.append("SIGTERM")
                self.process.terminate()
                try:
                    remaining, _ = self.process.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    signals.append("SIGKILL")
                    self.process.kill()
                    remaining, _ = self.process.communicate()
            return {
                "exit_code": self.process.returncode, "termination_signals": signals,
                "sent_requests": self.sent_requests, "received_messages": self.received_messages,
                "request_outcome_unresolved": self.pending,
                "last_pending_request": self.last_pending_request,
                "invalid_stdout_line": self.invalid_stdout_line,
                "late_stdout_bytes_hex": (bytes(self._buffer) + remaining).hex(),
            }
        finally:
            self.stderr_stream.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-config", type=Path, required=True)
    parser.add_argument("--collaboration-config", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--api-schemas", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--controller-command-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-config", type=Path)
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--evidence-table-after-prefix", action="store_true")
    parser.add_argument("--lossless-observation-history", action="store_true",
                        help="Encode repeated actor-local views by lossless references after any verified prefix; retain raw prompts.")
    parser.add_argument("--verification-message-replies", action="store_true",
                        help="After a verified prefix, give each delivered verification message one peer reply attempt within the existing budget.")
    parser.add_argument("--phase-action-decoder", action="store_true",
                        help="After the verified prefix, constrain action syntax to existing phase and public API permissions.")
    parser.add_argument("--action-decoder-bounds", type=Path,
                        help="Explicit YAML text/whitespace bounds for a separate decoder experiment.")
    args = parser.parse_args(argv)
    if (args.prefix_only or args.evidence_table_after_prefix or args.phase_action_decoder) and args.prefix_config is None:
        parser.error("prefix-only, evidence-table and phase-action-decoder modes require --prefix-config")
    if args.action_decoder_bounds is not None and not args.phase_action_decoder:
        parser.error("action-decoder-bounds requires --phase-action-decoder")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    actor = channel = None
    actor_config = collaboration = None
    prefix = None
    code = 1
    report = {
        "status": "FAILED", "stage": "configuration",
        "scientific_control_transfer_qualified": False,
        "formal_model_comparison_qualified": False,
        "complete_host_actor_native_cost_qualified": False,
    }
    try:
        bounds = (None if args.action_decoder_bounds is None else
                  validate_action_decoder_bounds(yaml.safe_load(args.action_decoder_bounds.read_text())))
        if bounds is not None:
            report["action_decoder_bounds"] = {"configuration": str(args.action_decoder_bounds),
                "sha256": sha256_file(args.action_decoder_bounds), "values": bounds}
        actor_config = validate_actor(yaml.safe_load(args.actor_config.read_text()))
        collaboration = validate_collaboration(yaml.safe_load(args.collaboration_config.read_text()))
        if (collaboration["provider"] != "local_transformers"
                or collaboration["model_revision"] != model_identity(actor_config)):
            raise ValueError("LOCAL_ACTOR_CONTROLLER_REVISION_MISMATCH")
        allocation = yaml.safe_load(args.allocation.read_text())
        if (not isinstance(allocation, dict) or set(allocation) != set(ACTORS)
                or any(type(value) is not int or value < 0 for value in allocation.values())
                or sum(allocation.values()) != collaboration["verification_budget"]):
            raise ValueError("LOCAL_ACTOR_ALLOCATION_INVALID")
        schemas = json.loads(args.api_schemas.read_text())
        if not isinstance(schemas, dict) or not schemas:
            raise ValueError("LOCAL_HOST_PUBLIC_SCHEMAS_REQUIRED")
        command = json.loads(args.controller_command_json.read_text())
        if (not isinstance(command, list) or not command
                or any(not isinstance(value, str) or not value or "\0" in value for value in command)):
            raise ValueError("CONTROLLER_COMMAND_INVALID")
        metadata = json.loads(args.metadata.read_text())
        if args.prefix_config is not None:
            settings = yaml.safe_load(args.prefix_config.read_text())
            if (settings.get("schema") != "appworld_recorded_prefix_diagnostic_v1"
                    or settings.get("source_case_job") != 590688 or settings.get("prefix_attempts") != 10
                    or settings.get("remaining_attempts") != 20 or settings.get("allocation") != allocation):
                raise ValueError("LOCAL_PREFIX_PROTOCOL_INVALID")
            loaded = {}
            for name in ("generations", "host_report", "transport", "native_checkpoint"):
                binding = settings["inputs"][name]
                path = args.prefix_config.parent / binding["path"]
                if sha256_file(path) != binding["sha256"]:
                    raise ValueError("LOCAL_PREFIX_SOURCE_BINDING_CHANGED:" + name)
                loaded[name] = ([json.loads(line) for line in path.read_text().splitlines()]
                                if name == "transport" else json.loads(path.read_text()))
            checkpoint = validate_prefix_checkpoint(loaded["native_checkpoint"], collaboration["native"])
            prefix = RecordedPrefix(loaded["generations"], loaded["host_report"], loaded["transport"],
                                    native_checkpoint_sha256=content_digest(checkpoint))
            report["recorded_prefix_protocol"] = {"sha256": sha256_file(args.prefix_config),
                "source_case_job": settings["source_case_job"], "prefix_attempts": 10,
                "remaining_attempts": 20, "prefix_only": args.prefix_only,
                "evidence_table_after_prefix": args.evidence_table_after_prefix}
        report["stage"] = "file_verification"
        report["snapshot_verification"] = verify_snapshot(args.model_directory, metadata, actor_config, tokenizer_only=args.prefix_only)
        report["stage"] = "model_loading"
        load_started = time.perf_counter()
        try:
            prompts = {role: ("You are the " + role + " in a typed tool collaboration. "
                              "Use only your permitted observations and return one JSON action without markdown.")
                       for role in ACTORS}
            if args.prefix_only:
                tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(str(args.model_directory), local_files_only=True, trust_remote_code=False)
                actor = LocalTransformersActor.for_prefix_replay(tokenizer, actor_config, prompts)
            else:
                actor = LocalTransformersActor(args.model_directory, actor_config, prompts)
        finally:
            report["model_load_attempt_wall_seconds"] = time.perf_counter() - load_started
        # Prefix replay can execute up to max_interactions bounded native calls.
        # This response bound includes their declared timeout envelope and the
        # existing idle allowance; a timeout remains an unresolved operation.
        limits = collaboration["native"]["limits"]
        response_timeout = limits["max_interactions"] * limits["timeout_seconds"] + collaboration["idle_timeout_seconds"]
        report["controller_response_timeout_seconds"] = response_timeout
        report["controller_command"] = command
        channel = ProcessJsonChannel(command, output / "controller_stderr.log", response_timeout)
        report["stage"] = "collaboration"
        report["host"] = LocalActorHost(actor, channel, collaboration, schemas, allocation,
            recorded_prefix=prefix, evidence_table_after_prefix=args.evidence_table_after_prefix,
            lossless_observation_history=args.lossless_observation_history,
            verification_message_replies=args.verification_message_replies,
            phase_action_decoder=args.phase_action_decoder, action_decoder_bounds=bounds).run(
                pause_at_prefix_boundary=args.prefix_only)
        report["status"] = report["host"]["status"]
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
    finally:
        if channel is not None:
            try:
                report["controller_process"] = channel.close_and_wait()
            except Exception as exc:
                report["controller_close_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
                report["status"] = "FAILED"
        if (report["status"] == "DEVELOPMENT_EXECUTION_COMPLETE"
                and report.get("controller_process", {}).get("exit_code") == 2
                and not report["controller_process"]["request_outcome_unresolved"]
                and not report["controller_process"]["termination_signals"]):
            code = 2
        elif (args.prefix_only and report["status"] == "PAUSED_AT_VERIFIED_PREFIX"
              and report.get("host", {}).get("recorded_prefix", {}).get("status") == "VERIFIED"
              and report.get("host", {}).get("new_generation_attempt_count") == 0
              and report.get("controller_process", {}).get("exit_code") == 1
              and not report["controller_process"]["request_outcome_unresolved"]
              and not report["controller_process"]["termination_signals"]):
            try:
                late = bytes.fromhex(report["controller_process"]["late_stdout_bytes_hex"]).decode("utf-8")
                terminal = [json.loads(line)["terminal"] for line in late.splitlines() if line.strip()]
                valid = (len(terminal) == 1 and isinstance(terminal[0], dict)
                    and isinstance(terminal[0].get("error"), dict)
                    and terminal[0]["error"].get("error") == "MODEL_HOST_CLOSED_BEFORE_FINISH"
                    and isinstance(terminal[0].get("recorded_native_prefix"), dict)
                    and terminal[0]["recorded_native_prefix"].get("status") == "PASS")
            except (ValueError, KeyError, TypeError) as exc:
                report["prefix_validation_terminal_error"] = {"error_type":type(exc).__name__,"error":str(exc)}
                valid = False
            if valid:
                code = 0
                report["status"] = "RECORDED_PREFIX_NATIVE_VALIDATION_COMPLETE"
                report["prefix_validation_controller_terminal"] = terminal[0]
            else:
                report["status"] = "FAILED"
                report["prefix_validation_failure"] = "RECORDED_NATIVE_PREFIX_TERMINAL_NOT_VERIFIED"
        else:
            report["status"] = "FAILED"
        if actor is not None:
            try:
                report["actor_runtime"] = actor.runtime()
            except Exception as exc:
                report["actor_runtime_error_type"] = type(exc).__name__
                report["status"], code = "FAILED", 1
        attempts = [] if actor is None else actor.attempts
        freeze_json(output / "generation_attempts.json", attempts)
        ended = resource.getrusage(resource.RUSAGE_SELF)
        report["cost"] = {
            "measurement_scope": "host entry through controller shutdown, before final artifact writes",
            "host_wall_seconds": time.perf_counter() - started,
            "host_cpu_user_seconds": ended.ru_utime - cpu_started.ru_utime,
            "host_cpu_system_seconds": ended.ru_stime - cpu_started.ru_stime,
            "process_peak_rss_bytes": ended.ru_maxrss * 1024,
            "note": "Native CPU costs and GPU allocation costs have separate scopes; overlapping wall times are not summed.",
        }
        report["package_versions"] = {}
        for package in ("torch", "transformers", "huggingface-hub", "accelerate"):
            try:
                report["package_versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["package_versions"][package] = None
        report["exit_code"] = code
        report["input_sha256"] = {
            name: sha256_file(getattr(args, name))
            for name in ("actor_config", "collaboration_config", "allocation", "api_schemas", "metadata", "controller_command_json")
            if getattr(args, name).is_file()
        }
        if args.prefix_config is not None and args.prefix_config.is_file():
            report["input_sha256"]["prefix_config"] = sha256_file(args.prefix_config)
        freeze_json(output / "report.json", report)
        freeze_json(output / "manifest.json", {
            **runtime_manifest({"actor": actor_config, "collaboration": collaboration},
                               sys.argv if argv is None else argv, code),
            "artifacts": {name: sha256_file(output / name) for name in ("report.json", "generation_attempts.json")},
        })
        print(json.dumps({"status": report["status"], "exit_code": code,
                          "generation_attempts": len(attempts)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Exercise pinned local model loading and two isolated neutral JSON contexts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import yaml

from Libs.population_dynamics.appworld_collaboration import generation_telemetry
from Libs.population_dynamics.appworld_local_actor import (
    ACTORS, LocalTransformersActor, download_snapshot, model_identity,
    request_from_receipt, snapshot_entries, validate_config, verify_snapshot,
)
from Libs.population_dynamics.provenance import freeze_json, runtime_manifest, sha256_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicitly permit pinned public model intake")
    parser.add_argument("--intake-only", action="store_true", help="Verify source files on CPU without loading the model")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    model = None
    config = None
    code = 1
    report = {
        "status": "PREPARING", "stage": "configuration", "appworld_tasks_executed": 0,
        "scientific_control_transfer_qualified": False,
        "formal_model_comparison_qualified": False,
        "complete_host_actor_native_cost_qualified": False,
        "readiness_scope": ("source file intake and integrity verification only; no model inference" if args.intake_only
                            else "two neutral JSON generations; no AppWorld observations or execution"),
        "intake_only": args.intake_only,
        "download_requested": args.download, "readiness_results": [],
        "inference_seed_is_training_seed": False,
    }
    try:
        config = validate_config(yaml.safe_load(args.config.read_text()))
        metadata = json.loads(args.metadata.read_text())
        report["model_revision"] = model_identity(config)
        report["source_metadata_sha256"] = sha256_file(args.metadata)
        entries = snapshot_entries(metadata, config)
        report["source_declared_file_bytes"] = sum(row["size"] for row in entries.values())
        freeze_json(output / "resolved_config.json", config)
        if args.download:
            report["stage"] = "download"
            intake_started = time.perf_counter()
            try:
                download_snapshot(args.model_directory, metadata, config)
            finally:
                report["download_wall_seconds"] = time.perf_counter() - intake_started
        report["stage"] = "file_verification"
        verify_started = time.perf_counter()
        report["snapshot_verification"] = verify_snapshot(args.model_directory, metadata, config)
        report["file_verification_wall_seconds"] = time.perf_counter() - verify_started
        if args.intake_only:
            report.update(status="LOCAL_MODEL_FILES_VERIFIED", stage="file_verification_complete")
            code = 0
            return code
        report["stage"] = "model_loading"
        system_prompts = {
            actor: ("You are the " + actor + " in a typed collaboration runtime. "
                    "Return exactly one JSON action object, without markdown or commentary.")
            for actor in ACTORS
        }
        load_started = time.perf_counter()
        try:
            model = LocalTransformersActor(args.model_directory, config, system_prompts)
        finally:
            report["model_load_attempt_wall_seconds"] = time.perf_counter() - load_started
        report["stage"] = "neutral_generations"
        for actor in ACTORS:
            recipient = next(other for other in ACTORS if other != actor)
            expected = {"actor": actor, "kind": "message", "recipient": recipient,
                        "text": "readiness_" + actor, "share_results": []}
            user_text = ("Return exactly one JSON object identical to the object below. "
                         "Preserve every key and value; do not add markdown or commentary.\n"
                         + json.dumps(expected))
            receipt = model.generate(actor, user_text)
            row = {"actor": actor, "accepted": False}
            try:
                request = request_from_receipt(actor, receipt)
                if {k: v for k, v in request.items() if k != "generation"} != expected:
                    raise ValueError("LOCAL_ACTOR_NEUTRAL_ACTION_MISMATCH")
                if generation_telemetry(receipt)["status"] != "COMPLETE":
                    raise ValueError("LOCAL_ACTOR_TELEMETRY_INCOMPLETE")
                row["accepted"] = True
            except ValueError as exc:
                row["error_code"] = str(exc)
            report["readiness_results"].append(row)
        report["status"] = ("LOCAL_MODEL_INFERENCE_READY"
                            if all(row["accepted"] for row in report["readiness_results"])
                            else "LOCAL_MODEL_JSON_READINESS_FAILED")
        code = 0 if report["status"] == "LOCAL_MODEL_INFERENCE_READY" else 1
    except Exception as exc:
        report.update(status="LOCAL_MODEL_READINESS_FAILED", error_type=type(exc).__name__)
        if isinstance(exc, ValueError) and str(exc).startswith("LOCAL_ACTOR_"):
            report["error_code"] = str(exc)
        # Public download errors can contain signed CDN query strings. Record
        # error classes/status without persisting those transient credentials.
        response = getattr(exc, "response", None)
        report["http_status"] = getattr(response, "status_code", None)
        report["error_errno"] = getattr(exc, "errno", None)
    finally:
        attempts = [] if model is None else model.attempts
        if model is not None:
            try:
                report["runtime"] = model.runtime()
            except Exception as exc:
                report["runtime_error_type"] = type(exc).__name__
                report["status"] = "LOCAL_MODEL_READINESS_FAILED"
                code = 1
        report["generation_attempts"] = len(attempts)
        report["successful_generations"] = sum(row["status"] == "GENERATED" for row in attempts)
        report["generations_with_token_usage"] = sum(row["usage"] is not None for row in attempts)
        report["observed_token_usage"] = {
            key: sum(row["usage"][key] for row in attempts if row["usage"] is not None)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        } if any(row["usage"] is not None for row in attempts) else None
        report["observed_inference_seconds"] = (
            sum(row["inference_seconds"] for row in attempts if row["inference_seconds"] is not None)
            if any(row["inference_seconds"] is not None for row in attempts) else None)
        ended = resource.getrusage(resource.RUSAGE_SELF)
        report["cost"] = {
            "scope": "runner entry through terminal computation before artifact writes",
            "host_wall_seconds": time.perf_counter() - started,
            "host_cpu_user_seconds": ended.ru_utime - cpu_started.ru_utime,
            "host_cpu_system_seconds": ended.ru_stime - cpu_started.ru_stime,
            "process_peak_rss_bytes": ended.ru_maxrss * 1024,
        }
        report["package_versions"] = {}
        for package in ("torch", "transformers", "huggingface-hub", "accelerate"):
            try:
                report["package_versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["package_versions"][package] = None
        report["exit_code"] = code
        freeze_json(output / "generation_attempts.json", attempts)
        freeze_json(output / "report.json", report)
        freeze_json(output / "manifest.json", {
            **runtime_manifest(config or {}, sys.argv if argv is None else argv, code),
            "artifacts": {name: sha256_file(output / name)
                          for name in ("report.json", "generation_attempts.json")},
            "source_metadata_sha256": report.get("source_metadata_sha256"),
        })
        print(json.dumps({"status": report["status"], "stage": report["stage"],
                          "exit_code": code, "generation_attempts": len(attempts)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure lossless prompt projection on retained train execution boundaries.

No world, model inference, scorer, or test task is opened. Original token counts
must match the recorded server counts before projection measurements qualify.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any

from transformers import AutoTokenizer
from population_dynamics import appworld_benchmark_host as current_host

ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
          "chat_template.jinja", "merges.txt", "config.json")
REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"


def exact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def decode(row: dict) -> bytes:
    raw = base64.b64decode(row["content_base64"], validate=True)
    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Declared content identity differs")
    return raw


def expand_projection(projected: dict) -> dict:
    result = deepcopy(projected)
    result.pop("prompt_projection")
    events = result["events"]
    for index, event in enumerate(events):
        payload = event["payload"]
        if event["kind"] == "state_report_delivered":
            value = payload["representation"].get("events")
            if isinstance(value, dict) and set(value) == {"$history_ref"}:
                ref = value["$history_ref"]
                if ref != {"kind": "prefix", "end_exclusive": index}:
                    raise ValueError("Projection prefix reference differs")
                payload["representation"]["events"] = deepcopy(events[:index])
        elif event["kind"] in {"message_sent", "message_delivered"}:
            for result_id, value in payload["shared_results"].items():
                if not isinstance(value, dict) or set(value) != {"$history_ref"}:
                    continue
                ref = value["$history_ref"]
                anchor = ref["event_index"]
                if ref["kind"] != "observation" or type(anchor) is not int or not 0 <= anchor < index:
                    raise ValueError("Projection observation reference is not backward")
                path = ref["path"]
                if path not in (["payload", "observation"], ["payload", "shared_results", result_id]):
                    raise ValueError("Unexpected projection observation path")
                original = events[anchor]
                for key in path:
                    original = original[key]
                payload["shared_results"][result_id] = deepcopy(original)
    return result


def prefix_session(receipt: dict, actor: str, events: list[dict]) -> SimpleNamespace:
    allocations = [row["payload"]["assigned_calls"] for row in events
                   if row["kind"] == "verification_allocation"]
    if len(allocations) > 1:
        raise ValueError("Repeated allocation in retained prefix")
    allocation = None if not allocations else {actor: allocations[0]}
    observed_ids = {event["payload"]["result_id"] for event in events
                    if event["kind"] == "tool_observation"}
    used = sum(
        attempt["verification"] is True and attempt["result_id"] in observed_ids
        for attempt in receipt["host"]["native_attempts"] if attempt["actor"] == actor
    )
    if allocation is not None and used > allocation[actor]:
        raise ValueError("Observed verification consumption exceeds allocation")
    return SimpleNamespace(allocation=allocation, verification_used={actor: used})


def original_count(call: dict, generation: dict) -> tuple[int | None, str]:
    if generation["status"] == "pre_generation_rejected":
        rejection = call.get("context_budget_rejection")
        if rejection != generation.get("context_budget_rejection"):
            raise ValueError("Generation and model rejection proofs differ")
        return (rejection or {}).get("prompt_tokens"), "pre_generation_rejection"
    raw = call.get("raw_usage") or {}
    response = (call.get("raw_response") or {}).get("usage") or {}
    count = raw.get("prompt_tokens")
    if response and response.get("prompt_tokens") != count:
        raise ValueError("Provider usage records disagree")
    return count, "raw_usage"


def measure(source: dict, returned: dict, tokenizer_dir: Path) -> dict:
    if importlib.metadata.version("transformers") != "4.57.0":
        raise ValueError("Pinned transformers 4.57.0 required")
    if source["model_inputs"]["model_revision"] != REVISION:
        raise ValueError("Pinned tokenizer revision differs")
    for name in ASSETS:
        row = source["model_inputs"]["files"][name]
        raw = (tokenizer_dir / name).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Tokenizer asset differs: " + name)

    manifest = json.loads(decode(returned["files"]["output_manifest.json"]))
    if (manifest["run_id"], manifest["trial_id"]) != (returned["source_run_id"], returned["source_trial_id"]):
        raise ValueError("Returned manifest execution identity differs")
    def read_member(name: str) -> dict:
        row = returned["files"][name]
        declared = manifest["files"][name.removeprefix("artifacts/")]
        if {key: row[key] for key in ("bytes", "sha256")} != declared:
            raise ValueError("Selected member differs from source manifest")
        return json.loads(decode(row))

    admission = read_member("artifacts/input_admission.json")
    for name, sha in admission["source_sha256"].items():
        if name in source["files"] and source["files"][name]["sha256"] != sha:
            raise ValueError("Original source input and execution admission differ: " + name)
    paths = sorted(name for name in returned["files"]
                   if name.startswith("artifacts/development/collaboration_uniform/train/")
                   and name.endswith("/execution_receipt.json"))
    expected = sum(row["agent"] == "collaboration_uniform" for row in source["development_attempts"])
    if not paths or len(paths) != expected:
        raise ValueError("Complete original uniform roster required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), use_fast=True, local_files_only=True, trust_remote_code=False,
    )
    def count(messages: list[dict]) -> int:
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    rows = []
    attempts = []
    with tempfile.TemporaryDirectory(prefix="original-appworld-source-") as temporary:
        root = Path(temporary)
        for name, record in source["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "Libs":
                raise ValueError("Unsafe original source path")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(decode(record))
        package = "_appworld_original_prompt_source"
        module = ModuleType(package)
        module.__path__ = [str(root / "Libs/population_dynamics")]
        sys.modules[package] = module
        old_host = importlib.import_module(package + ".appworld_benchmark_host")
        for path in paths:
            receipt = read_member(path)
            if (receipt["split"] != "train" or receipt["seed"] not in (41, 42, 43)
                    or receipt["formal_eligible"] is not False or receipt["terminal"] is not True):
                raise ValueError("Only settled development train receipts are admissible")
            host = receipt["host"]
            generations, calls = host["generations"], receipt["model_calls"]
            if len(generations) != len(calls):
                raise ValueError("Generation and model-call ledgers differ")
            score_path = path.rsplit("/", 1)[0] + "/development_score.json"
            # Collaboration scoring is beside the attempt directory in this pipeline.
            if score_path not in returned["files"]:
                score_path = path.rsplit("/attempt/", 1)[0] + "/development_score.json"
            score = read_member(score_path)
            if score["scoring_status"] != "completed" or type(score["native_tgc"]) is not bool:
                raise ValueError("Independent completed score required")
            attempts.append({"task_id": receipt["task_id"], "seed": receipt["seed"],
                             "status": receipt["status"], "native_tgc": score["native_tgc"],
                             "generations": len(generations), "member": path})
            for actor, history in host["actor_histories"].items():
                used = prefix_session(receipt, actor, history["events"]).verification_used[actor]
                if used != host["verification_used"][actor]:
                    raise ValueError("Final verification ledger cannot be reconstructed from observed native returns")
            for index, (generation, call) in enumerate(zip(generations, calls)):
                actor = generation["actor"]
                if call.get("consumer", actor) != actor:
                    raise ValueError("Model call belongs to another actor")
                prefix = generation["prefix_events"]
                events = host["actor_histories"][actor]["events"][:prefix]
                if len(events) != prefix:
                    raise ValueError("Generation prefix exceeds original history")
                event_json = tuple(exact(event) for event in events)
                snapshot = current_host.ActorSnapshot(actor, event_json)
                old_snapshot = old_host.ActorSnapshot(actor, event_json)
                state = SimpleNamespace(actor=actor, phase=generation["phase"],
                                        session=prefix_session(receipt, actor, events))
                original_input = old_snapshot.as_model_input()
                projected = current_host.prompt_visible_history(snapshot)
                lossless = exact(expand_projection(projected)) == exact(original_input)
                old_input = (old_host.prompt_visible_history(old_snapshot)
                             if hasattr(old_host, "prompt_visible_history") else original_input)
                messages = [
                    {"role": "system", "content": old_host.CollaborationHost._instructions(state)},
                    {"role": "user", "content": json.dumps(old_input, ensure_ascii=False, allow_nan=False)},
                ]
                projected_messages = [
                    {"role": "system", "content": current_host.CollaborationHost._instructions(state)},
                    {"role": "user", "content": json.dumps(projected, ensure_ascii=False, allow_nan=False)},
                ]
                recorded, evidence = original_count(call, generation)
                original_tokens, projected_tokens = count(messages), count(projected_messages)
                matched = type(recorded) is int and recorded == original_tokens
                window = receipt["transport_context_window_tokens"]
                full_output = receipt["host_budget"]["max_completion_tokens"]
                rows.append({
                    "task_id": receipt["task_id"], "seed": receipt["seed"],
                    "generation": index, "actor": actor, "phase": generation["phase"],
                    "prefix_events": prefix, "recorded_prompt_tokens": recorded,
                    "recorded_source": evidence, "reconstructed_prompt_tokens": original_tokens,
                    "original_count_exact": matched, "lossless_roundtrip": lossless,
                    "projected_prompt_tokens": projected_tokens,
                    "projection_measurement_qualified": matched and lossless,
                    "context_window_tokens": window, "full_output_allowance": full_output,
                    "original_full_output_margin": window - original_tokens - full_output,
                    "projected_full_output_margin": window - projected_tokens - full_output,
                    "prompt_tokens_saved": original_tokens - projected_tokens,
                })
    qualified = all(row["projection_measurement_qualified"] for row in rows)
    return {
        "schema": "appworld_prompt_token_diagnostic_v1",
        "status": "EXACT_ORIGINAL_COUNTS_AND_LOSSLESS_PROJECTION" if qualified else "UNVERIFIED_RECONSTRUCTION",
        "source_run_id": returned["source_run_id"], "source_trial_id": returned["source_trial_id"],
        "model_revision": REVISION, "transformers_version": importlib.metadata.version("transformers"),
        "prompt_projection": current_host.PROMPT_PROJECTION_VERSION,
        "tokenizer_class": type(tokenizer).__name__, "add_generation_prompt": True,
        "tokenizer_asset_identity": {name: source["model_inputs"]["files"][name] for name in ASSETS},
        "original_host_sha256": source["files"]["Libs/population_dynamics/appworld_benchmark_host.py"]["sha256"],
        "current_host_sha256": hashlib.sha256(Path(current_host.__file__).read_bytes()).hexdigest(),
        "attempts": attempts, "rows": rows,
        "summary": {
            "attempts": len(attempts), "generations": len(rows),
            "exact_original_counts": sum(row["original_count_exact"] for row in rows),
            "lossless_roundtrips": sum(row["lossless_roundtrip"] for row in rows),
            "max_original_prompt_tokens": max(row["reconstructed_prompt_tokens"] for row in rows),
            "max_projected_prompt_tokens": max(row["projected_prompt_tokens"] for row in rows),
            "original_full_output_overflows": sum(row["original_full_output_margin"] < 0 for row in rows),
            "projected_full_output_overflows": sum(row["projected_full_output_margin"] < 0 for row in rows),
        },
        "native_tasks_executed": 0, "model_generations": 0, "formal_eligible": False,
        "interpretation": "Prompt representation diagnostic on the same retained prefixes; task performance is untested.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--returned-files", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = measure(json.loads(args.source_inputs.read_text()),
                     json.loads(args.returned_files.read_text()), args.tokenizer_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], **report["summary"]}), flush=True)
    return 0 if report["status"] == "EXACT_ORIGINAL_COUNTS_AND_LOSSLESS_PROJECTION" else 1


if __name__ == "__main__":
    raise SystemExit(main())

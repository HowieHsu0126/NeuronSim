"""Inspect saved truncations with the executed tokenizer/mask; never call a model.

Only decoded responses were archived. The diagnostic therefore checks a
text-equivalent re-tokenization, not the original generated token sequence.
Completion witnesses stay in memory and are never sent to the native task.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer

from Libs.population_dynamics.appworld_action_decoder import ActionSchemaDecoder
from Libs.population_dynamics.appworld_collaboration import content_digest


def message_suffix(text: str, schema: dict) -> str | None:
    """Find a syntax-only witness for an open top-level message text field."""
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg != "Unterminated string starting at" or not re.search(
                r'"text"\s*:\s*$', text[:error.pos]):
            return None
    else:
        return None
    try:
        value = json.loads(text + '"}')
    except json.JSONDecodeError:
        # Do not guess how to finish a partial escape or a nested structure.
        return None
    if value.get("kind") not in {"message", "proposal"}:
        return None
    branches = [branch for branch in schema["anyOf"]
                if branch["properties"]["kind"].get("enum") == [value["kind"]]]
    if len(branches) != 1:
        return None
    branch = branches[0]
    suffix = '"'
    for key in branch["required"]:
        if key in value:
            continue
        declaration = branch["properties"][key]
        if key == "share_results":
            item = []
        elif len(declaration.get("enum", [])) == 1:
            item = declaration["enum"][0]
        else:
            return None
        suffix += "," + json.dumps(key) + ":" + json.dumps(item)
    return suffix + "}"


def diagnose(host_report: Path, tokenizer_directory: Path, output: Path) -> dict:
    source = json.loads(host_report.read_text())["host"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, local_files_only=True,
                                             trust_remote_code=False)
    adapter = ActionSchemaDecoder(tokenizer)
    cases = []
    report = {"status": "RUNNING", "cases": cases, "model_inference_calls": 0,
              "native_task_calls": 0, "original_token_ids_available": False,
              "scope": "Text-equivalent re-tokenized prefixes; no logits or original token path.",
              "completion_witnesses_executed": False}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for attempt in source["host_attempts"]:
        if attempt["attempt_index"] < source["replayed_prefix_generation_count"] or attempt["status"] != "HOST_REJECTED":
            continue
        generation = source["generation_attempts"][attempt["producer_attempt_index"]]
        if not generation.get("reached_token_cap"):
            continue
        started = time.perf_counter()
        response = generation["response"]
        schema = generation["action_decoder"]["schema"]
        assert content_digest(schema) == generation["action_decoder"]["schema_sha256"]
        prompt_ids = tokenizer.encode(generation["prompt"], add_special_tokens=False)
        assert len(prompt_ids) == generation["prompt_tokens_before_generation"]
        generated_ids = tokenizer.encode(response, add_special_tokens=False)
        assert tokenizer.decode(generated_ids, skip_special_tokens=True) == response
        cap = generation["usage"]["completion_tokens"]
        metadata = {}
        callback = adapter.prepare(schema, cap, metadata)
        case = {"attempt_index": attempt["attempt_index"], "actor": generation["actor"],
                "recorded_completion_tokens": cap, "retokenized_tokens": len(generated_ids),
                "schema_sha256": content_digest(schema), "checkpoints": [],
                "retokenized_path_rejection": None, "status": "RUNNING"}
        cases.append(case)
        checkpoints = {64, 128, 256, 384, 448, len(generated_ids)}
        sent_ids = list(prompt_ids)
        for position in range(len(generated_ids) + 1):
            sent = torch.tensor(sent_ids, dtype=torch.long)
            allowed = callback(0, sent)
            if position in checkpoints:
                prefix_text = tokenizer.decode(generated_ids[:position], skip_special_tokens=True)
                suffix = message_suffix(prefix_text, schema) if response.startswith(prefix_text) else None
                check = {"retokenized_prefix_tokens": position,
                         "text_is_saved_response_prefix": response.startswith(prefix_text),
                         "syntax_witness_constructible": suffix is not None}
                if suffix is not None:
                    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                    witness_ids = list(sent_ids)
                    rejection = None
                    for offset, token in enumerate(suffix_ids):
                        if token not in callback(0, torch.tensor(witness_ids, dtype=torch.long)):
                            rejection = offset
                            break
                        witness_ids.append(token)
                    complete = rejection is None and tokenizer.eos_token_id in callback(
                        0, torch.tensor(witness_ids, dtype=torch.long))
                    check.update(suffix_tokens=len(suffix_ids), rejection_suffix_offset=rejection,
                                 complete_json_and_eos_allowed=complete,
                                 within_recorded_token_cap=position + len(suffix_ids) + 1 <= cap)
                case["checkpoints"].append(check)
            if position == len(generated_ids):
                break
            token = generated_ids[position]
            if token not in allowed:
                case["retokenized_path_rejection"] = {"token_offset": position, "token_id": token}
                break
            sent_ids.append(token)
        case.update(status="COMPLETE", callback_calls=metadata["callback_calls"],
                    callback_failures=metadata["callback_failures"],
                    callback_wall_seconds=metadata["callback_wall_seconds"],
                    wall_seconds=time.perf_counter() - started)
        save()
        del callback
        gc.collect()
    expected = sum(a["attempt_index"] >= source["replayed_prefix_generation_count"]
                   and a["status"] == "HOST_REJECTED"
                   and source["generation_attempts"][a["producer_attempt_index"]].get("reached_token_cap", False)
                   for a in source["host_attempts"])
    assert len(cases) == expected and expected > 0
    report.update(status="COMPLETE_TEXT_EQUIVALENT_TERMINATION_DIAGNOSTIC",
                  truncated_host_rejections=expected,
                  retokenized_paths_admitted=sum(c["retokenized_path_rejection"] is None for c in cases),
                  cases_with_budget_feasible_end=sum(any(
                      p.get("complete_json_and_eos_allowed") and p.get("within_recorded_token_cap")
                      for p in c["checkpoints"]) for c in cases),
                  callback_failures=sum(c["callback_failures"] for c in cases))
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-report", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.host_report, args.tokenizer, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))

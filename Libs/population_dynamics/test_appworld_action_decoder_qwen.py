"""Integration check for the pinned Qwen tokenizer, without model inference."""
import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from .appworld_action_decoder import ActionSchemaDecoder, build_phase_action_schema


@pytest.mark.skipif("APPWORLD_DECODER_TOKENIZER_DIRECTORY" not in os.environ,
                    reason="requires explicitly staged pinned tokenizer assets")
def test_pinned_qwen_callback_with_real_public_schemas_and_chat_prefix():
    directory = Path(os.environ["APPWORLD_DECODER_TOKENIZER_DIRECTORY"])
    report_path = Path(os.environ["APPWORLD_DECODER_TOKENIZER_REPORT"])
    started = time.perf_counter()
    progress = {"status": "RUNNING", "stage": "TOKENIZER_LOAD", "cases": []}
    def save_progress(stage):
        progress.update(stage=stage, wall_seconds=time.perf_counter()-started)
        report_path.write_text(json.dumps(progress, indent=2) + "\n")
    save_progress("TOKENIZER_LOAD")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    progress["tokenizer_load_wall_seconds"] = time.perf_counter()-started
    save_progress("DECODER_TOKENIZER_DATA_BUILD")
    prepared = time.perf_counter()
    adapter = ActionSchemaDecoder(tokenizer)
    progress["decoder_tokenizer_data_wall_seconds"] = time.perf_counter()-prepared
    public = json.loads(Path(os.environ["APPWORLD_DECODER_API_SCHEMAS"]).read_text())
    view = {"actor": "executor", "phase": "verification", "allocated_verification_calls": 1,
            "used_verification_calls": 0, "discovery_calls_used": 0}
    action_schema = build_phase_action_schema("executor", "VERIFY", view, public, discovery_call_limit=3)
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "Return one JSON action."},
        {"role": "user", "content": json.dumps({"actor": "executor", "view": view})}],
        tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cases = []
    progress["cases"] = cases

    def check(label, text, expected):
        save_progress(label)
        metadata = {}
        callback = adapter.prepare(action_schema, 512, metadata)
        sent = torch.tensor(prompt_ids, dtype=torch.long)
        accepted = True
        generated_ids = tokenizer.encode(text, add_special_tokens=False)
        for token in generated_ids:
            if token not in callback(0, sent):
                accepted = False
                break
            sent = torch.cat((sent, torch.tensor([token], dtype=torch.long)))
        if accepted:
            accepted = tokenizer.eos_token_id in callback(0, sent)
        cases.append({"label": label, "accepted": accepted, "expected": expected,
                      "tokens": len(generated_ids), "callback_calls": metadata["callback_calls"],
                      "callback_failures": metadata["callback_failures"],
                      "callback_wall_seconds": metadata["callback_wall_seconds"],
                      "schema_sha256": metadata["schema_sha256"]})
        save_progress(label + "_COMPLETE")
        assert accepted is expected, label
        assert metadata["callback_failures"] == 0

    reads = [(app, api) for app, apis in public.items() for api, definition in apis.items()
             if definition["method"].upper() == "GET"]
    writes = [(app, api) for app, apis in public.items() for api, definition in apis.items()
              if definition["method"].upper() != "GET"]
    assert reads and writes
    arguments = {"unicode": "中🙂café", "escaped": 'quotes" slash\\ newline\n',
                 "array": list(range(32)), "nested": {"values": [None, True, False, -1.25e-3]}}
    app, api = reads[0]
    action = {"actor": "executor", "kind": "tool", "call_id": "model_chosen", "app": app,
              "api": api, "arguments": arguments}
    for escaped in (False, True):
        for reverse in (False, True):
            ordered = dict(reversed(list(action.items()))) if reverse else action
            check(f"read_unicode_{escaped}_reverse_{reverse}",
                  " " * 32 + json.dumps(ordered, ensure_ascii=escaped), True)
    for app, api in (reads[-1], reads[len(reads)//2]):
        check("other_public_GET_" + app + "." + api, json.dumps({**action, "app": app, "api": api}), True)
    app, api = writes[0]
    check("write_forbidden_in_verification", json.dumps({**action, "app": app, "api": api}), False)
    check("unknown_app_api_pair", json.dumps({**action, "api": "__undeclared_API__"}), False)
    check("finish_forbidden_in_verification", json.dumps({"actor": "executor", "kind": "finish"}), False)
    check("wrong_actor", json.dumps({**action, "actor": "planner"}), False)
    check("message_remains_a_model_choice", json.dumps({"actor": "executor", "kind": "message",
          "recipient": "planner", "text": "中🙂", "share_results": [f"chosen_{i}" for i in range(32)]},
          ensure_ascii=False), True)
    generation = json.loads((directory / "generation_config.json").read_text())
    generation_eos = generation["eos_token_id"]
    generation_eos = generation_eos if isinstance(generation_eos, list) else [generation_eos]
    # Other model stop IDs must remain masked while the JSON parser is incomplete.
    callback = adapter.prepare(action_schema, 512, {})
    initial_allowed = callback(0, torch.tensor(prompt_ids, dtype=torch.long))
    assert tokenizer.eos_token_id in generation_eos
    assert not set(generation_eos).intersection(initial_allowed)
    report = {"status": "PASS", "model_inference_calls": 0, "native_task_calls": 0,
              "tokenizer_class": type(tokenizer).__name__, "tokenizer_eos": tokenizer.eos_token_id,
              "model_generation_eos": generation_eos, "prompt_tokens": len(prompt_ids),
              "public_apps": len(public), "cases": cases,
              "tokenizer_load_wall_seconds": progress["tokenizer_load_wall_seconds"],
              "decoder_tokenizer_data_wall_seconds": progress["decoder_tokenizer_data_wall_seconds"],
              "wall_seconds": time.perf_counter()-started,
              "limitation": "Detects premature EOS; does not classify internal fallback after a complete JSON action."}
    report_path.write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.skipif("APPWORLD_DECODER_TOKENIZER_DIRECTORY" not in os.environ,
                    reason="requires explicitly staged pinned tokenizer assets")
def test_pinned_qwen_unicode_escape_shortcut_regression():
    """Reproduce the failed escape token without repeating full public-schema cases."""
    tokenizer = AutoTokenizer.from_pretrained(Path(os.environ["APPWORLD_DECODER_TOKENIZER_DIRECTORY"]),
                                               local_files_only=True, trust_remote_code=False)
    adapter = ActionSchemaDecoder(tokenizer)
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": "Return one JSON action."},
        {"role": "user", "content": "Check JSON representation only."}], tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    cases = []
    for text, expected in [(json.dumps("café", ensure_ascii=True), True),
                           (json.dumps("prefix中🙂café", ensure_ascii=True), True),
                           (json.dumps("caf\\u0041", ensure_ascii=True), True),
                           (r'"caf\q"', False), (r'"caf\u00zz"', False)]:
        metadata = {};callback = adapter.prepare({"type": "string"}, 512, metadata)
        sent = torch.tensor(prefix, dtype=torch.long)
        rejected = None
        for index, token in enumerate(tokenizer.encode(text, add_special_tokens=False)):
            if token not in callback(0, sent):
                rejected = {"index": index, "token": token, "decoded_token": tokenizer.decode([token])}
                break
            sent = torch.cat((sent, torch.tensor([token], dtype=torch.long)))
        accepted = rejected is None and tokenizer.eos_token_id in callback(0, sent)
        cases.append({"text": text, "expected": expected, "accepted": accepted,
                      "rejection": rejected, "metadata": metadata})
        assert accepted is expected, cases[-1]
        assert metadata["callback_failures"] == 0
    if "APPWORLD_DECODER_UNICODE_REPORT" in os.environ:
        Path(os.environ["APPWORLD_DECODER_UNICODE_REPORT"]).write_text(json.dumps({
            "status": "PASS_TARGETED_UNICODE_ESCAPE_REGRESSION", "model_inference_calls": 0,
            "native_task_calls": 0, "cases": cases}, ensure_ascii=False, indent=2) + "\n")

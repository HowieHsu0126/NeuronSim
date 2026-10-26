"""Decoder accounting with existing native/transport fixtures; no model inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from .appworld_action_decoder import build_phase_action_schema
from .appworld_collaboration import content_digest
from .appworld_pair_diagnostic import summarize_arm, summarize_phase_action_decoder, analyze_decoder_execution
from .test_appworld_pair_diagnostic import reply_fixture, reply_terminal_fixture


def decoder_fixture(partial=None):
    old = list(reply_fixture() if partial is None else reply_terminal_fixture("context_refusal"))
    report, native, transport, events, scheduler, _ = old
    protocol = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_phase_action_decoder.yaml").read_text())
    config = {"provider": "local_transformers", "model_revision": protocol["model_revision"],
              "max_actor_turns": 30, "verification_budget": 2, "discovery_calls_per_actor": 3}
    schemas = {"spotify": {"read": {"method": "GET"}, "login": {"method": "POST"}, "follow_artist": {"method": "POST"}}}
    host = report["host"];host["phase_action_decoder"] = True
    for index, (attempt, generation) in enumerate(zip(host["host_attempts"], host["generation_attempts"])):
        if index < 10:
            continue
        state, actor = attempt["host_state"], attempt["actor"]
        phase = "verification" if state in {"VERIFY", "VERIFY_REPLY"} else "execution"
        view = {"actor": actor, "phase": phase, "allocated_verification_calls": 1,
                "used_verification_calls": int(index in {12, 14, 15}), "discovery_calls_used": 0,
                "results": {"visible_fixture": {"ok": True}}, "inbox": []}
        prompt = {"actor": actor, "host_state": state, "view": view}
        if state == "VERIFY_REPLY":prompt["reply_recipient"] = "executor"
        expected = build_phase_action_schema(actor, state, view, schemas, discovery_call_limit=3,
                                               reply_recipient=prompt.get("reply_recipient"))
        attempt.update(unencoded_user_prompt=json.dumps(prompt), action_schema_sha256=content_digest(expected))
        if generation["status"] == "FAILED" and partial == "context":
            continue
        metadata = {"schema": expected, "schema_sha256": content_digest(expected),
                    "implementation": "lm-format-enforcer", "version": "0.11.3", "interegular_version": "0.3.3",
                    "parser_settings": {"max_json_array_length": 0, "force_json_field_order": False, "max_consecutive_whitespaces": 513},
                    "callback_calls": 3, "callback_failures": 0, "callback_wall_seconds": 0.125,
                    "preparation_wall_seconds": 0.25}
        generation["action_decoder"] = metadata
        if generation["status"] == "FAILED":
            generation.pop("prompt_tokens_before_generation", None)
            if partial == "preparation":
                metadata.pop("implementation");metadata.pop("version");metadata.pop("interegular_version");metadata.pop("parser_settings")
                metadata.update(callback_calls=0, callback_wall_seconds=0.0)
                generation["error_code"] = "LOCAL_ACTOR_ACTION_DECODER_DEPENDENCY_UNAVAILABLE"
            else:
                generation.update(inference_seconds=0.75, error_code="LOCAL_ACTOR_ACTION_DECODER_PREMATURE_EOS" if partial == "callback" else "MODEL_RUNTIME_FAILURE")
                metadata.update(callback_calls=2, callback_failures=int(partial == "callback"))
            attempt["error_code"] = generation["error_code"]
    refresh_costs(report)
    return report, native, transport, events, scheduler, protocol, schemas, config


def refresh_costs(report):
    host = report["host"]
    generations = host["generation_attempts"][10:]
    token_rows = [row for row in generations if row.get("usage") is not None]
    time_rows = [row for row in generations if row.get("inference_seconds") is not None]
    host["cost_accounting"] = {"provider_token_usage": {"total_tokens": sum(row["usage"]["total_tokens"] for row in token_rows)} if token_rows else None,
                               "provider_inference_seconds": sum(row["inference_seconds"] for row in time_rows) if time_rows else None}
    rows = [row["action_decoder"] for row in generations if isinstance(row.get("action_decoder"), dict)]
    runtime = {"attempt_count": len(rows), "schema_sha256s": sorted({row["schema_sha256"] for row in rows})}
    runtime.update({key: sum(row[key] for row in rows) for key in ("callback_calls", "callback_failures", "callback_wall_seconds", "preparation_wall_seconds")})
    report["actor_runtime"] = {"action_decoder": runtime} if rows else {}


def test_bounded_message_protocol_binds_the_new_schema_and_whitespace_without_relabeling_history():
    args = list(decoder_fixture())
    report, native, transport, events, scheduler, old, schemas, config = args
    protocol = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_bounded_message_decoder.yaml").read_text())
    args[5] = protocol
    report["host"]["action_decoder_bounds"] = protocol["action_decoder_bounds"]
    for attempt, generation in zip(report["host"]["host_attempts"][10:], report["host"]["generation_attempts"][10:]):
        prompt = json.loads(attempt["unencoded_user_prompt"])
        expected = build_phase_action_schema(prompt["actor"], prompt["host_state"], prompt["view"], schemas,
            discovery_call_limit=3, reply_recipient=prompt.get("reply_recipient"), message_text_max_length=256)
        attempt["action_schema_sha256"] = content_digest(expected)
        metadata = generation["action_decoder"]
        metadata.update(schema=expected, schema_sha256=content_digest(expected))
        metadata["parser_settings"]["max_consecutive_whitespaces"] = 4
    refresh_costs(report)
    result = summarize_phase_action_decoder(*args)
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["historical_reference"]["job_id"] == 597334
    assert result["native_write_failures"] == 1 and result["causal_benefit_qualified"] is False
    with pytest.raises(ValueError, match="BOUNDS_MISMATCH"):
        legacy = list(args); legacy[5] = old
        summarize_phase_action_decoder(*legacy)
    report["host"]["generation_attempts"][10]["action_decoder"]["parser_settings"]["max_consecutive_whitespaces"] = 513
    result = summarize_phase_action_decoder(*args)
    assert not result["accounting_complete"]
    assert "DECODER_CONFIGURATION_MISMATCH" in {item["code"] for item in result["accounting_issues"]}


def test_decoder_archive_entry_checks_receipts_and_executed_software(tmp_path):
    from .provenance import sha256_file
    report,native,transport,events,scheduler,protocol,schemas,config = decoder_fixture()
    protocol["schema"] = "appworld_bounded_message_decoder_development_v1"
    protocol["historical_reference_job"] = 597334
    protocol["action_decoder_bounds"] = {"message_text_max_length":256,"max_consecutive_whitespaces":4}
    report["host"]["action_decoder_bounds"] = protocol["action_decoder_bounds"]
    for attempt,generation in zip(report["host"]["host_attempts"][10:],report["host"]["generation_attempts"][10:]):
        prompt = json.loads(attempt["unencoded_user_prompt"])
        expected = build_phase_action_schema(prompt["actor"],prompt["host_state"],prompt["view"],schemas,
            discovery_call_limit=3,reply_recipient=prompt.get("reply_recipient"),message_text_max_length=256)
        attempt["action_schema_sha256"] = content_digest(expected)
        generation["action_decoder"].update(schema=expected,schema_sha256=content_digest(expected))
        generation["action_decoder"]["parser_settings"]["max_consecutive_whitespaces"] = 4
    refresh_costs(report)
    archive = tmp_path/"run_900";software = tmp_path/"software";software.mkdir()
    entries = {};tested = []
    def write(role,name,value):
        path = archive/role/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value if isinstance(value,str) else json.dumps(value))
        row = {"path":name,"bytes":path.stat().st_size,"sha256":sha256_file(path)}
        entries.setdefault(role,[]).append(row)
        if role == "inputs":tested.append(row)
    for name in ["Libs/population_dynamics/appworld_action_decoder.py","Libs/population_dynamics/appworld_local_actor.py",
                 "Libs/population_dynamics/appworld_local_host.py","Libs/scripts/run_appworld_local_collaboration.py","launcher.sbatch"]:
        write("inputs",name,"fixture tested source: "+name)
    write("inputs","phase_action_decoder_protocol.yaml",yaml.safe_dump(protocol))
    write("inputs","action_decoder_bounds.yaml",yaml.safe_dump(protocol["action_decoder_bounds"]))
    write("inputs","api_docs.json",schemas)
    write("inputs","Libs/configs/neural_population_appworld_local_collaboration.yaml",yaml.safe_dump(config))
    write("gpu_host","host/report.json",report)
    write("cpu_controller","controller/summary.json",native)
    write("cpu_controller","controller/transport.jsonl","\n".join(json.dumps(row) for row in transport)+"\n")
    write("cpu_controller","controller/events.json",events)
    def receipts():
        for role,rows in entries.items():
            (archive/role/"copy_out.json").write_text(json.dumps({"verified":True,"job_id":"900","files":rows}))
    receipts()
    (software/"report.json").write_text(json.dumps({"status":"SUBMISSION_BINDING_AND_CURRENT_PREFIX_REUSE_VERIFIED","exit_code":0,"inputs":tested}))
    observation = {"query_node":"mgmt2","exit_code":0,"components":[{"job_id":"900","state":"FAILED"},{"job_id":"901","state":"FAILED"}]}
    result = analyze_decoder_execution(archive,observation,software)
    assert result["summary"]["accounting_complete"] and result["new_model_generations"] == 0
    assert result["summary"]["native_write_failures"] == 1
    import subprocess
    import sys
    scheduler_path = tmp_path/"scheduler.json"
    scheduler_path.write_text(json.dumps(observation))
    output = tmp_path/"readback"
    command = [sys.executable,"-B","-m","Libs.population_dynamics.appworld_pair_diagnostic",
               "--decoder-execution",str(archive),"--validated-decoder-software",str(software),
               "--scheduler",str(scheduler_path),"--output",str(output)]
    child = subprocess.run(command,capture_output=True,text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads((output/"report.json").read_text())["summary"]["accounting_complete"]
    assert (output/"actions.csv").is_file() and json.loads((output/"manifest.json").read_text())["exit_code"] == 0
    observation["components"][0]["state"] = "RUNNING"
    with pytest.raises(ValueError,match="TERMINAL_DECODER"):
        analyze_decoder_execution(archive,observation,software)
    observation["components"][0]["state"] = "FAILED"
    path = archive/"inputs/launcher.sbatch";path.write_text("changed source")
    with pytest.raises(ValueError,match="DECLARED_DECODER_INPUT_CHANGED"):
        analyze_decoder_execution(archive,observation,software)
    entry = next(row for row in entries["inputs"] if row["path"] == "launcher.sbatch")
    entry.update(bytes=path.stat().st_size,sha256=sha256_file(path));receipts()
    # The independent tested binding must still reject a rewritten receipt.
    tested_proof = json.loads((software/"report.json").read_text())
    old_entry = next(row for row in tested_proof["inputs"] if row["path"] == "launcher.sbatch")
    old_entry["sha256"] = "0" * 64
    (software/"report.json").write_text(json.dumps(tested_proof))
    with pytest.raises(ValueError,match="EXECUTED_DECODER_SOFTWARE_DIFFERS"):
        analyze_decoder_execution(archive,observation,software)
    failed_output = tmp_path/"invalid_readback"
    child = subprocess.run([*command[:-1],str(failed_output)],capture_output=True,text=True)
    assert child.returncode != 0 and not failed_output.exists()


def test_phase_action_decoder_accounting_keeps_native_failure_and_original_input():
    args = decoder_fixture();before = deepcopy(args[0])
    result = summarize_phase_action_decoder(*args)
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["native_write_failures"] == 1 and result["native_evaluation_status"] == "NOT_RUN"
    assert result["decoder_runtime_from_attempts"]["callback_calls"] == 18
    assert result["prefix_decoder_activity_absent"] is True
    assert result["all_verification_complete"] is True
    assert result["historical_reference"]["job_id"] == 596889
    assert result["historical_reference"]["post_prefix_preintervention_comparison_performed"] is False
    assert result["causal_benefit_qualified"] is False
    assert args[0] == before


@pytest.mark.parametrize("partial,stage", [("context", "CONTEXT_REFUSED_BEFORE_DECODER"),
                                          ("preparation", "DECODER_PREPARATION_FAILED"),
                                          ("model", "MODEL_OR_DECODE_FAILED_AFTER_CALLBACK"),
                                          ("callback", "DECODER_CALLBACK_FAILED")])
def test_phase_action_decoder_failures_preserve_partial_metering(partial, stage):
    args = decoder_fixture(partial)
    result = summarize_phase_action_decoder(*args)
    assert result["accounting_complete"] is True, result["accounting_issues"]
    assert result["decoder_trace"][-1]["decoder_stage"] == stage
    assert result["decoder_trace"][-1]["usage"] is None
    assert result["all_verification_complete"] is False
    expected_seconds = 3.25 if partial in {"model", "callback"} else 2.5
    assert result["new_inference_cost"]["provider_inference_seconds"] == expected_seconds
    assert result["new_measured_token_generations"] == 2
    assert result["new_measured_inference_generations"] == (3 if partial in {"model", "callback"} else 2)
    if partial in {"model", "callback"}:
        # The old default is deliberately unchanged; only the new entry opts in.
        with pytest.raises(ValueError, match="NEW_INFERENCE_TOTAL_MISMATCH"):
            summarize_arm(*args[:5], 10)


@pytest.mark.parametrize("damage,code", [("prefix", "PREFIX_DECODER_ACTIVITY"),
                                        ("zero_callbacks", "GENERATED_ACTION_WITHOUT_SUCCESSFUL_CALLBACKS"),
                                        ("runtime", "DECODER_RUNTIME_AGGREGATE_MISMATCH"),
                                        ("view", "PRIVATE_VIEW_SCHEMA_RECONSTRUCTION_FAILED"),
                                        ("schema", "DECODER_SCHEMA_BINDING_MISMATCH")])
def test_phase_action_decoder_trace_damage_is_explicit(damage, code):
    args = decoder_fixture();host = args[0]["host"]
    if damage == "prefix":host["generation_attempts"][0]["action_decoder"] = {"callback_calls": 1}
    elif damage == "zero_callbacks":
        host["generation_attempts"][10]["action_decoder"]["callback_calls"] = 0
        refresh_costs(args[0])
    elif damage == "runtime":args[0]["actor_runtime"]["action_decoder"]["callback_wall_seconds"] += 0.5
    elif damage == "view":host["host_attempts"][10]["unencoded_user_prompt"] = json.dumps({"actor": "executor", "host_state": "VERIFY", "view": {}})
    else:
        metadata = host["generation_attempts"][10]["action_decoder"]
        metadata["schema"] = {"type": "object"};metadata["schema_sha256"] = content_digest(metadata["schema"])
        host["host_attempts"][10]["action_schema_sha256"] = metadata["schema_sha256"]
        refresh_costs(args[0])
    result = summarize_phase_action_decoder(*args)
    assert result["accounting_complete"] is False
    assert code in {row["code"] for row in result["accounting_issues"]}
    assert result["native_write_failures"] == 1


def test_phase_action_decoder_protocol_and_model_changes_are_rejected():
    args = list(decoder_fixture());args[5]["maximum_new_attempts"] = 21
    with pytest.raises(ValueError, match="FROZEN_PHASE_ACTION_DECODER_PROTOCOL_REQUIRED"):
        summarize_phase_action_decoder(*args)
    args = list(decoder_fixture());args[0]["host"]["model_revision"] = "another/model"
    with pytest.raises(ValueError, match="PHASE_ACTION_DECODER_CONFIGURATION_MISMATCH"):
        summarize_phase_action_decoder(*args)

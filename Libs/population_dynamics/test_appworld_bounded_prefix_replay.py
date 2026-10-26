"""Replay accepted native transport, without repeating native or model calls."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from .appworld_collaboration import content_digest, validate_prefix_checkpoint
from .appworld_local_actor import LocalTransformersActor
from .appworld_local_host import LocalActorHost
from .appworld_recorded_prefix import RecordedPrefix
from .provenance import sha256_file
from transformers import AutoTokenizer


@pytest.mark.skipif("APPWORLD_BOUNDED_PREFIX_SOURCE" not in os.environ,
                    reason="requires explicit accepted native transport")
def test_current_host_and_tokenizer_reuse_the_complete_native_prefix(tmp_path):
    source = Path(os.environ["APPWORLD_BOUNDED_PREFIX_SOURCE"])
    accepted = json.loads((source/"report.json").read_text())
    assert accepted["status"] == "RECORDED_PREFIX_NATIVE_VALIDATION_COMPLETE" and accepted["exit_code"] == 0
    declaration = {r["path"]:r for r in json.loads((source/"delivery.json").read_text())["files"]}
    paths = ["prefix/protocol.yaml", "prefix/source_generations.json", "prefix/source_host_report.json",
             "prefix/source_transport.jsonl", "prefix/native_checkpoint.json", "controller/transport.jsonl",
             "Libs/configs/neural_population_appworld_tool_actor.yaml",
             "Libs/configs/neural_population_appworld_local_collaboration.yaml", "api_docs.json",
             "allocation.yaml", "host/report.json"]
    for name in paths:
        assert sha256_file(source/name) == declaration[name]["sha256"]
    protocol = yaml.safe_load((source/"prefix/protocol.yaml").read_text())
    for binding in protocol["inputs"].values():
        assert sha256_file(source/"prefix"/binding["path"]) == binding["sha256"]
    config = yaml.safe_load((source/paths[7]).read_text())
    actor_config = yaml.safe_load((source/paths[6]).read_text())
    checkpoint = validate_prefix_checkpoint(json.loads((source/"prefix/native_checkpoint.json").read_text()), config["native"])
    generations = json.loads((source/"prefix/source_generations.json").read_text())
    source_report = json.loads((source/"prefix/source_host_report.json").read_text())
    original_transport = [json.loads(x) for x in (source/"prefix/source_transport.jsonl").read_text().splitlines()]
    native_transport = [json.loads(x) for x in (source/"controller/transport.jsonl").read_text().splitlines()]
    class RetainedChannel:
        index = 0
        pending = None
        ready = True
        def send(self, request):
            assert self.pending is None
            expected = deepcopy(native_transport[self.index]["request"])
            if "generation" in expected:
                expected["generation"]["session_id"] = actor.session_ids[expected["actor"]]
            assert request == expected
            self.pending = deepcopy(native_transport[self.index]["response"])
            self.index += 1
        def receive(self):
            if self.ready:
                self.ready = False
                return {"ready":True, "task":deepcopy(native_transport[0]["response"]["result"]["task"]),
                        "actors":["planner","executor"], "native_prefix_checkpoint_sha256":content_digest(checkpoint)}
            assert self.pending is not None
            response,self.pending = self.pending,None
            return response
    tokenizer = AutoTokenizer.from_pretrained(Path(os.environ["APPWORLD_DECODER_TOKENIZER_DIRECTORY"]),
                                               local_files_only=True, trust_remote_code=False)
    prompts = {role:"You are the " + role + " in a typed tool collaboration. Use only your permitted observations and return one JSON action without markdown."
               for role in ("planner","executor")}
    actor = LocalTransformersActor.for_prefix_replay(tokenizer, actor_config, prompts)
    prefix = RecordedPrefix(generations, source_report, original_transport,
                            native_checkpoint_sha256=content_digest(checkpoint))
    channel = RetainedChannel()
    host = LocalActorHost(actor, channel, config, json.loads((source/"api_docs.json").read_text()),
        yaml.safe_load((source/"allocation.yaml").read_text()), recorded_prefix=prefix,
        evidence_table_after_prefix=True,lossless_observation_history=True,verification_message_replies=True,
        phase_action_decoder=True, action_decoder_bounds={"message_text_max_length":256,"max_consecutive_whitespaces":4})
    report = host.run(pause_at_prefix_boundary=True)
    assert report["status"] == "PAUSED_AT_VERIFIED_PREFIX", report.get("error")
    assert report["recorded_prefix"]["status"] == "VERIFIED"
    assert report["new_generation_attempt_count"] == 0 and report["host_attempt_count"] == 10
    assert channel.pending is None and channel.index == len(native_transport)
    assert all("action_decoder" not in g for g in actor.attempts)
    assert all(g["prompt"] == original["prompt"] and g["response"] == original["response"]
               for g,original in zip(actor.attempts,generations))
    output = {"status":"PASS_RETAINED_NATIVE_TRANSPORT_AND_CURRENT_PRIVATE_PREFIX",
              "new_model_generations":0,"new_native_executions":0,"transport_exchanges":channel.index,
              "accepted_native_validation":str(source),"native_proof_reused":True,"host_prefix":report["recorded_prefix"]}
    if "APPWORLD_BOUNDED_PREFIX_REPORT" in os.environ:
        Path(os.environ["APPWORLD_BOUNDED_PREFIX_REPORT"]).write_text(json.dumps(output,indent=2)+"\n")

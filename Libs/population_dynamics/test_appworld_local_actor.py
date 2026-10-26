"""Pinned input integrity and local actor contracts without model downloads or CUDA."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from . import appworld_local_actor as local_actor
from Libs.scripts import run_appworld_local_model_readiness as readiness


@pytest.fixture
def actor_config():
    return {
        "model": {"repo_id": "fixture/actor", "revision": "a" * 40,
                  "dtype": "bfloat16", "attention_implementation": "sdpa",
                  "cuda_devices": 2, "max_memory_gib_per_device": 40},
        "generation": {"max_new_tokens": 256, "max_context_tokens": 4096, "do_sample": False},
        "inference_seed": 73003,
    }


def published_file(name, payload, *, lfs=False):
    row = {"rfilename": name, "size": len(payload)}
    if lfs:
        row["lfs"] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    else:
        row["blobId"] = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    return row


@pytest.fixture
def snapshot(tmp_path, actor_config):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    payloads = {
        "config.json": b'{"architectures":["FixtureModel"]}',
        "tokenizer.json": b'{"version":"fixture"}',
        "tokenizer_config.json": b'{"fixture":true}',
        "model.safetensors.index.json": json.dumps({"weight_map": {"layer.a": shards[0], "layer.b": shards[1]}}).encode(),
        shards[0]: b"fixture shard one; identity testing only",
        shards[1]: b"fixture shard two; identity testing only",
    }
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
    metadata = {"id": actor_config["model"]["repo_id"], "sha": actor_config["model"]["revision"],
                "siblings": [published_file(name, payload, lfs=name.endswith(".safetensors"))
                             for name, payload in payloads.items()]}
    return directory, metadata, payloads


def test_published_git_and_lfs_identities_cover_every_loaded_file(snapshot, actor_config):
    directory, metadata, payloads = snapshot
    result = local_actor.verify_snapshot(directory, metadata, actor_config)
    assert result["model_revision"] == "fixture/actor@" + "a" * 40
    assert result["all_published_file_identities_verified"] is True
    assert set(result["files"]) == set(payloads)
    assert result["bytes"] == sum(map(len, payloads.values()))
    for name, payload in payloads.items():
        assert result["files"][name]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert result["files"][name]["source_identity"] == (
            "published_lfs_sha256" if name.endswith(".safetensors") else "published_git_blob_sha1")


@pytest.mark.parametrize("field,value", [("sha", "b" * 40), ("id", "fixture/substitute")])
def test_snapshot_rejects_wrong_source_identity_before_file_reads(snapshot, actor_config, monkeypatch, field, value):
    directory, metadata, _ = snapshot
    metadata[field] = value

    def forbidden_hash(*args):
        raise AssertionError("Wrong pin must be rejected before reading model files")

    monkeypatch.setattr(local_actor, "sha256_file", forbidden_hash)
    with pytest.raises(ValueError, match="LOCAL_ACTOR_METADATA_PIN_MISMATCH"):
        local_actor.verify_snapshot(directory, metadata, actor_config)


@pytest.mark.parametrize("filename", ["config.json", "model-00001-of-00002.safetensors"])
def test_same_size_corruption_rejects_git_blob_and_lfs_inputs(snapshot, actor_config, filename):
    directory, metadata, payloads = snapshot
    original = payloads[filename]
    (directory / filename).write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="LOCAL_ACTOR_FILE_HASH_MISMATCH"):
        local_actor.verify_snapshot(directory, metadata, actor_config)


def test_missing_published_shard_is_rejected(snapshot, actor_config):
    directory, metadata, _ = snapshot
    (directory / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="LOCAL_ACTOR_FILE_SIZE_MISMATCH"):
        local_actor.verify_snapshot(directory, metadata, actor_config)


@pytest.mark.parametrize("mapping", [
    {"layer.a": "model-00001-of-00002.safetensors"},
    {"layer.a": "model-00001-of-00002.safetensors", "layer.b": "unpublished.safetensors"},
])
def test_hash_valid_index_must_match_the_complete_published_shard_set(snapshot, actor_config, mapping):
    directory, metadata, _ = snapshot
    name = "model.safetensors.index.json"
    payload = json.dumps({"weight_map": mapping}).encode()
    (directory / name).write_bytes(payload)
    metadata["siblings"] = [published_file(name, payload) if row["rfilename"] == name else row
                            for row in metadata["siblings"]]
    with pytest.raises(ValueError, match="LOCAL_ACTOR_WEIGHT_INDEX_MISMATCH"):
        local_actor.verify_snapshot(directory, metadata, actor_config)


def test_explicit_null_lfs_keeps_the_accepted_git_blob_verification_path(snapshot, actor_config):
    directory, metadata, _ = snapshot
    row = next(row for row in metadata["siblings"] if row["rfilename"] == "config.json")
    row["lfs"] = None
    assert local_actor.snapshot_entries(metadata, actor_config)["config.json"]["blobId"] == row["blobId"]
    result = local_actor.verify_snapshot(directory, metadata, actor_config)
    assert result["files"]["config.json"]["source_identity"] == "published_git_blob_sha1"


class TokenBatch(dict):
    def to(self, device):
        assert device.type == "cpu", "The fixture must never move a tensor to CUDA"
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class FixtureTokenizer:
    def __init__(self, torch):
        self.torch = torch
        self.template_calls = []
        self.prompts = []
        self.decode_error = None

    @staticmethod
    def render(messages):
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":")) + "\n<assistant>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.template_calls.append(deepcopy(messages))
        return self.render(messages)

    def __call__(self, prompt, *, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        self.prompts.append(prompt)
        ids = self.torch.tensor([[ord(char) for char in prompt]], dtype=self.torch.long, device="cpu")
        return TokenBatch(input_ids=ids, attention_mask=self.torch.ones_like(ids))

    eos_token_id = 0
    all_special_ids = [0]

    def __len__(self):
        return 128

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, tokens, *, skip_special_tokens=False):
        if hasattr(tokens, "device"):
            assert tokens.device.type == "cpu"
            tokens = tokens.tolist()
        if self.decode_error is not None:
            raise self.decode_error
        return "".join(chr(value) for value in tokens if value != 0)


class FixtureModel:
    def __init__(self, torch):
        self.torch = torch
        self.responses = []
        self.calls = []
        self.error = None
        self.weight = torch.zeros(1, device="cpu")
        self.hf_device_map = {"embedding": 0, "layers": 1}

    def eval(self):
        return self

    def parameters(self):
        # Only the constructor's allocation check sees this pretend parameter.
        # Embeddings, input tensors, and every generated tensor remain on CPU.
        return iter([SimpleNamespace(device=SimpleNamespace(type="cuda"), is_meta=False, numel=lambda: 17)])

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def generate(self, **kwargs):
        assert kwargs["input_ids"].device.type == "cpu"
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        response = self.responses.pop(0)
        output = self.torch.tensor([[*[ord(char) for char in response], 0]], dtype=self.torch.long, device="cpu")
        if "prefix_allowed_tokens_fn" in kwargs:
            sent = kwargs["input_ids"][0]
            for token in output[0]:
                assert int(token) in kwargs["prefix_allowed_tokens_fn"](0, sent)
                sent = self.torch.cat((sent, token.reshape(1)))
        return self.torch.cat((kwargs["input_ids"], output), dim=1)


@pytest.fixture
def cpu_actor(actor_config, tmp_path, monkeypatch):
    """Exercise the real constructor/generate path with fake allocation and CPU tensors."""
    torch = pytest.importorskip("torch")
    tokenizer = FixtureTokenizer(torch)
    model = FixtureModel(torch)
    loader_calls = {}
    cuda = SimpleNamespace(
        device_count=lambda: 2, is_available=lambda: True,
        get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 1024**3),
        device=lambda i: nullcontext(), is_bf16_supported=lambda: True,
        reset_peak_memory_stats=lambda i: None, synchronize=lambda i: None,
        get_device_name=lambda i: "CPU fixture allocation",
        max_memory_allocated=lambda i: 0, max_memory_reserved=lambda i: 0,
    )
    torch_runtime = SimpleNamespace(
        cuda=cuda, bfloat16=torch.bfloat16, set_num_threads=lambda n: None,
        manual_seed=lambda seed: None, inference_mode=torch.inference_mode,
        __version__=torch.__version__, version=SimpleNamespace(cuda=None),
    )

    def load_tokenizer(*args, **kwargs):
        loader_calls["tokenizer"] = (args, kwargs)
        return tokenizer

    def load_model(*args, **kwargs):
        loader_calls["model"] = (args, kwargs)
        return model

    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                                   AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
    original_import = local_actor.importlib.import_module

    def fixture_import(name, *args, **kwargs):
        if name == "torch":
            return torch_runtime
        if name == "transformers":
            return transformers
        return original_import(name, *args, **kwargs)

    clock = itertools.count()
    monkeypatch.setattr(local_actor.time, "perf_counter", lambda: next(clock) * 0.125)
    with monkeypatch.context() as constructor_patch:
        constructor_patch.setattr(local_actor.importlib, "import_module", fixture_import)
        actor = local_actor.LocalTransformersActor(tmp_path / "fake_model", actor_config,
                                                  {"planner": "planner private system", "executor": "executor private system"})
    actor.fixture_loader_calls = loader_calls
    return actor


def message_text(actor, text):
    return json.dumps({"actor": actor, "kind": "message", "recipient": "executor" if actor == "planner" else "planner",
                       "text": text, "share_results": []}, ensure_ascii=False, indent=2) + "\n"


def test_two_contexts_preserve_private_histories_original_text_and_actual_usage(cpu_actor):
    responses = [message_text("planner", "planner-only response"),
                 message_text("executor", "executor-only response"),
                 message_text("planner", "second planner response")]
    cpu_actor.model.responses = list(responses)
    receipts = [cpu_actor.generate("planner", "planner-only observation"),
                cpu_actor.generate("executor", "executor-only observation"),
                cpu_actor.generate("planner", "second planner observation")]
    assert receipts[0]["session_id"] == receipts[2]["session_id"]
    assert receipts[0]["session_id"] != receipts[1]["session_id"]
    assert cpu_actor.histories["planner"] is not cpu_actor.histories["executor"]
    assert "planner-only" not in receipts[1]["prompt"]
    assert "executor-only" not in receipts[2]["prompt"]
    assert cpu_actor.tokenizer.template_calls[2][2] == {"role": "assistant", "content": responses[0]}
    assert len(cpu_actor.tokenizer.template_calls[1]) == 2
    for index, receipt in enumerate(receipts):
        prompt = cpu_actor.tokenizer.prompts[index]
        assert receipt["prompt"] == prompt == FixtureTokenizer.render(cpu_actor.tokenizer.template_calls[index])
        assert receipt["response"] == responses[index]
        assert receipt["usage"] == {"prompt_tokens": len(prompt), "completion_tokens": len(responses[index]) + 1,
                                    "total_tokens": len(prompt) + len(responses[index]) + 1}
        assert receipt["inference_seconds"] > 0
        assert receipt["model_revision"] == local_actor.model_identity(cpu_actor.config)
        assert cpu_actor.attempts[index]["response"] == responses[index]
        assert cpu_actor.attempts[index]["status"] == "GENERATED"
        assert "past_key_values" not in cpu_actor.model.calls[index]
        assert cpu_actor.model.calls[index]["do_sample"] is False
    for _, kwargs in cpu_actor.fixture_loader_calls.values():
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    assert cpu_actor.fixture_loader_calls["model"][1]["max_memory"]["cpu"] == 0


def test_context_overflow_is_a_recorded_failure_without_generation_or_history_loss(cpu_actor):
    before = deepcopy(cpu_actor.histories)
    cpu_actor.config["generation"]["max_context_tokens"] = 257
    with pytest.raises(ValueError, match="LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED"):
        cpu_actor.generate("planner", "observation exceeds remaining context capacity")
    assert not cpu_actor.model.calls and cpu_actor.histories == before
    assert len(cpu_actor.attempts) == 1
    record = cpu_actor.attempts[0]
    assert record["status"] == "FAILED" and record["error_code"] == "LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED"
    assert record["prompt_tokens_before_generation"] > 1
    assert record["usage"] is None and record["inference_seconds"] is None
    assert record["host_attempt_wall_seconds"] > 0


def test_failed_model_generation_keeps_elapsed_cost_and_previous_successful_history(cpu_actor):
    cpu_actor.model.responses = [message_text("planner", "first response")]
    first = cpu_actor.generate("planner", "first observation")
    before = deepcopy(cpu_actor.histories)
    cpu_actor.model.error = RuntimeError("fixture inference failure")
    with pytest.raises(RuntimeError, match="fixture inference failure"):
        cpu_actor.generate("planner", "failed observation")
    assert cpu_actor.histories == before and len(cpu_actor.attempts) == 2
    success, failure = cpu_actor.attempts
    assert success["status"] == "GENERATED" and success["usage"] == first["usage"]
    assert failure["status"] == "FAILED" and failure["error_type"] == "RuntimeError"
    assert failure["inference_seconds"] > 0 and failure["host_attempt_wall_seconds"] >= failure["inference_seconds"]
    assert failure["usage"] is None and failure["prompt_tokens_before_generation"] > 0
    assert "failed observation" in failure["prompt"]


def test_decode_failure_retains_generated_token_counts_and_timing(cpu_actor):
    response = message_text("planner", "decode failure")
    cpu_actor.model.responses = [response]
    cpu_actor.tokenizer.decode_error = ValueError("fixture decode failure")
    before = deepcopy(cpu_actor.histories)
    with pytest.raises(ValueError, match="fixture decode failure"):
        cpu_actor.generate("planner", "observation")
    record = cpu_actor.attempts[0]
    assert record["status"] == "FAILED" and record["error_type"] == "ValueError"
    assert record["usage"]["completion_tokens"] == len(response) + 1
    assert record["usage"]["total_tokens"] == record["usage"]["prompt_tokens"] + len(response) + 1
    assert record["inference_seconds"] > 0 and cpu_actor.histories == before


def test_action_envelope_attaches_a_copy_without_rewriting_model_output(cpu_actor):
    response = '  {"actor":"planner","kind":"tool","call_id":"read_one","app":"spotify","api":"read","arguments":{"query":"original value"},"extra":"preserve for controller validation"}  \n'
    cpu_actor.model.responses = [response]
    receipt = cpu_actor.generate("planner", "Produce an action")
    original = deepcopy(receipt)
    request = local_actor.request_from_receipt("planner", receipt)
    assert {key: value for key, value in request.items() if key != "generation"} == json.loads(response)
    assert request["generation"] == original and request["generation"]["response"] == response
    receipt["usage"]["completion_tokens"] = -1
    assert request["generation"]["usage"] == original["usage"]


@pytest.mark.parametrize("response,error", [
    ("not JSON", "LOCAL_ACTOR_RESPONSE_NOT_JSON"),
    ('{"actor":"executor","kind":"finish"}', "LOCAL_ACTOR_ACTION_ENVELOPE_INVALID"),
    ('{"actor":"planner","kind":"allocate"}', "LOCAL_ACTOR_ACTION_ENVELOPE_INVALID"),
    ('{"actor":"planner","kind":"message","generation":{}}', "LOCAL_ACTOR_ACTION_ENVELOPE_INVALID"),
    ("[]", "LOCAL_ACTOR_ACTION_ENVELOPE_INVALID"),
])
def test_rejected_model_output_keeps_its_original_attempt_and_observed_cost(cpu_actor, response, error):
    cpu_actor.model.responses = [response]
    receipt = cpu_actor.generate("planner", "Produce an action")
    with pytest.raises(ValueError, match=error):
        local_actor.request_from_receipt("planner", receipt)
    assert cpu_actor.attempts[0]["status"] == "GENERATED"
    assert cpu_actor.attempts[0]["response"] == response
    assert cpu_actor.attempts[0]["usage"] == receipt["usage"]
    assert cpu_actor.attempts[0]["inference_seconds"] > 0


def readiness_arguments(snapshot, actor_config, tmp_path, monkeypatch, cpu_actor):
    directory, metadata, _ = snapshot
    config_path, metadata_path = tmp_path / "actor.yaml", tmp_path / "metadata.json"
    config_path.write_text(yaml.safe_dump(actor_config))
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / "readiness"
    monkeypatch.setattr(readiness, "LocalTransformersActor", lambda *args: cpu_actor)
    monkeypatch.setattr(readiness.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(readiness, "runtime_manifest", lambda config, command, code: {"exit_code": code})

    def forbidden_download(*args):
        raise AssertionError("CPU readiness fixtures must not download a model")

    monkeypatch.setattr(readiness, "download_snapshot", forbidden_download)
    return ["--config", str(config_path), "--metadata", str(metadata_path), "--model-directory", str(directory),
            "--output", str(output)], output


def test_readiness_aggregates_two_real_fixture_generations_without_qualifying_appworld(snapshot, actor_config,
                                                                                      tmp_path, monkeypatch, cpu_actor):
    cpu_actor.model.responses = [message_text(actor, "readiness_" + actor) for actor in local_actor.ACTORS]
    args, output = readiness_arguments(snapshot, actor_config, tmp_path, monkeypatch, cpu_actor)
    assert readiness.main(args) == 0
    report = json.loads((output / "report.json").read_text())
    attempts = json.loads((output / "generation_attempts.json").read_text())
    assert report["status"] == "LOCAL_MODEL_INFERENCE_READY"
    assert report["generation_attempts"] == report["successful_generations"] == report["generations_with_token_usage"] == 2
    assert report["observed_token_usage"] == {key: sum(row["usage"][key] for row in attempts)
                                             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    assert report["observed_inference_seconds"] == sum(row["inference_seconds"] for row in attempts)
    assert len({row["session_id"] for row in attempts}) == 2
    assert report["appworld_tasks_executed"] == 0 and report["download_requested"] is False
    assert report["scientific_control_transfer_qualified"] is False
    assert report["formal_model_comparison_qualified"] is False
    assert report["complete_host_actor_native_cost_qualified"] is False


def test_runtime_inspection_failure_cannot_discard_failed_generation_evidence(snapshot, actor_config,
                                                                             tmp_path, monkeypatch, cpu_actor):
    cpu_actor.model.error = RuntimeError("fixture inference failure")

    def failed_runtime():
        raise ValueError("fixture runtime inspection failure")

    monkeypatch.setattr(cpu_actor, "runtime", failed_runtime)
    args, output = readiness_arguments(snapshot, actor_config, tmp_path, monkeypatch, cpu_actor)
    assert readiness.main(args) == 1
    report = json.loads((output / "report.json").read_text())
    attempts = json.loads((output / "generation_attempts.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "LOCAL_MODEL_READINESS_FAILED"
    assert report["error_type"] == "RuntimeError" and report["runtime_error_type"] == "ValueError"
    assert report["generation_attempts"] == 1 and report["successful_generations"] == 0
    assert report["observed_token_usage"] is None
    assert report["observed_inference_seconds"] == attempts[0]["inference_seconds"] > 0
    assert attempts[0]["status"] == "FAILED" and attempts[0]["usage"] is None
    assert manifest["artifacts"]["generation_attempts.json"] == hashlib.sha256((output / "generation_attempts.json").read_bytes()).hexdigest()
    assert manifest["artifacts"]["report.json"] == hashlib.sha256((output / "report.json").read_bytes()).hexdigest()


def test_intake_only_verifies_files_without_importing_or_loading_model(snapshot, actor_config, tmp_path, monkeypatch):
    directory, metadata, _payloads = snapshot
    config_path = tmp_path / "config.yaml"
    metadata_path = tmp_path / "metadata.json"
    output = tmp_path / "intake"
    config_path.write_text(yaml.safe_dump(actor_config))
    metadata_path.write_text(json.dumps(metadata))
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU intake must not load a model")
    monkeypatch.setattr(readiness, "LocalTransformersActor", forbidden)
    code = readiness.main(["--config", str(config_path), "--metadata", str(metadata_path),
                           "--model-directory", str(directory), "--output", str(output), "--intake-only"])
    report = json.loads((output / "report.json").read_text())
    assert code == 0 and report["status"] == "LOCAL_MODEL_FILES_VERIFIED"
    assert report["snapshot_verification"]["all_published_file_identities_verified"] is True
    assert report["generation_attempts"] == 0 and report["observed_inference_seconds"] is None
    assert report["formal_model_comparison_qualified"] is False


def test_tokenizer_only_scope_never_claims_or_relaxes_weight_verification(snapshot, actor_config):
    directory, metadata, _ = snapshot
    for name in local_actor.snapshot_entries(metadata, actor_config):
        if name.endswith('.safetensors'):
            (directory/name).unlink()
    result = local_actor.verify_snapshot(directory,metadata,actor_config,tokenizer_only=True)
    assert result['verification_scope']=='published_non_weight_files_only'
    assert result['model_weights_verified'] is False and result['all_published_file_identities_verified'] is False
    assert all(not name.endswith('.safetensors') for name in result['files'])
    with pytest.raises(ValueError,match='FILE_SIZE_MISMATCH'):
        local_actor.verify_snapshot(directory,metadata,actor_config)


def test_optional_action_decoder_uses_actual_callback_without_changing_prompt_receipt_or_budget(cpu_actor):
    from .test_appworld_action_decoder import schema, tool
    response = json.dumps(tool("executor", arguments={"model_selected": [1, 2, None]}), indent=1) + "\n"
    cpu_actor.model.responses = [response]
    before = deepcopy(cpu_actor.config)
    receipt = cpu_actor.generate("executor", "unchanged private observation", action_schema=schema())
    assert set(receipt) == {"session_id", "prompt", "response", "model_revision", "usage", "inference_seconds"}
    assert receipt["response"] == response == cpu_actor.histories["executor"][-1]["content"]
    assert receipt["prompt"] == FixtureTokenizer.render(cpu_actor.tokenizer.template_calls[-1])
    assert "model_selected" not in receipt["prompt"]
    options = cpu_actor.model.calls[-1]
    assert options["max_new_tokens"] == before["generation"]["max_new_tokens"]
    assert options["do_sample"] is False and options["use_cache"] is True
    assert cpu_actor.config == before
    metadata = cpu_actor.attempts[-1]["action_decoder"]
    assert metadata["schema_sha256"] == local_actor.content_digest(schema())
    assert metadata["callback_calls"] == receipt["usage"]["completion_tokens"]
    assert metadata["callback_wall_seconds"] > 0 and metadata["preparation_wall_seconds"] > 0
    assert cpu_actor.runtime()["action_decoder"]["callback_calls"] == metadata["callback_calls"]


def test_default_generation_does_not_require_or_construct_optional_decoder(cpu_actor, monkeypatch):
    def forbidden(*args):
        raise AssertionError("disabled decoder must not initialize")
    monkeypatch.setattr(local_actor, "ActionSchemaDecoder", forbidden)
    cpu_actor.model.responses = [message_text("planner", "unchanged default")]
    cpu_actor.generate("planner", "default observation")
    assert "prefix_allowed_tokens_fn" not in cpu_actor.model.calls[-1]
    assert "action_decoder" not in cpu_actor.attempts[-1]
    assert "action_decoder" not in cpu_actor.runtime()


def test_context_refusal_precedes_decoder_preparation_and_never_increases_budget(cpu_actor, monkeypatch):
    from .test_appworld_action_decoder import schema
    def forbidden(*args):
        raise AssertionError("context refusal cannot prepare the decoder")
    monkeypatch.setattr(local_actor, "ActionSchemaDecoder", forbidden)
    cpu_actor.config["generation"]["max_context_tokens"] = 257
    with pytest.raises(ValueError, match="LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED"):
        cpu_actor.generate("executor", "observation", action_schema=schema())
    assert not cpu_actor.model.calls and cpu_actor.attempts[-1]["usage"] is None
    assert "action_decoder" not in cpu_actor.attempts[-1]

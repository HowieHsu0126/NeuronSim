"""Pinned local actor inference for the existing AppWorld JSON transport.

The model owns separate planner/executor histories. Readiness uses neutral
messages only; it does not execute AppWorld or qualify an allocation comparison.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import re
import time
import uuid

from .appworld_action_decoder import ActionSchemaDecoder
from .appworld_collaboration import content_digest
from .provenance import sha256_file

ACTORS = ("planner", "executor")
PIN_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
REQUIRED_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json",
                  "model.safetensors.index.json"}
OPTIONAL_FILES = {"generation_config.json", "special_tokens_map.json",
                  "vocab.json", "merges.txt", "chat_template.jinja"}


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != {"model", "generation", "inference_seed"}:
        raise ValueError("LOCAL_ACTOR_CONFIG_FIELDS_INVALID")
    model = config["model"]
    if (not isinstance(model, dict) or set(model) !=
            {"repo_id", "revision", "dtype", "attention_implementation",
             "cuda_devices", "max_memory_gib_per_device"}):
        raise ValueError("LOCAL_ACTOR_MODEL_FIELDS_INVALID")
    if (not isinstance(model["repo_id"], str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repo_id"]) is None
            or not isinstance(model["revision"], str) or PIN_PATTERN.fullmatch(model["revision"]) is None):
        raise ValueError("LOCAL_ACTOR_PIN_REQUIRED")
    if model["dtype"] != "bfloat16" or model["attention_implementation"] != "sdpa":
        raise ValueError("LOCAL_ACTOR_RUNTIME_NOT_VALIDATED")
    if any(type(model[k]) is not int or model[k] <= 0
           for k in ("cuda_devices", "max_memory_gib_per_device")):
        raise ValueError("LOCAL_ACTOR_DEVICE_BOUNDS_INVALID")
    generation = config["generation"]
    if (not isinstance(generation, dict) or set(generation) !=
            {"max_new_tokens", "max_context_tokens", "do_sample"}
            or generation["do_sample"] is not False):
        raise ValueError("LOCAL_ACTOR_GENERATION_FIELDS_INVALID")
    if any(type(generation[k]) is not int or generation[k] <= 0
           for k in ("max_new_tokens", "max_context_tokens")):
        raise ValueError("LOCAL_ACTOR_TOKEN_BOUNDS_INVALID")
    if generation["max_new_tokens"] >= generation["max_context_tokens"]:
        raise ValueError("LOCAL_ACTOR_CONTEXT_BUDGET_INVALID")
    if type(config["inference_seed"]) is not int or not 0 <= config["inference_seed"] < 2**32:
        raise ValueError("LOCAL_ACTOR_INFERENCE_SEED_INVALID")
    return config


def model_identity(config: dict) -> str:
    model = validate_config(config)["model"]
    return model["repo_id"] + "@" + model["revision"]


def snapshot_entries(metadata: dict, config: dict) -> dict[str, dict]:
    model = validate_config(config)["model"]
    if metadata.get("id") != model["repo_id"] or metadata.get("sha") != model["revision"]:
        raise ValueError("LOCAL_ACTOR_METADATA_PIN_MISMATCH")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("LOCAL_ACTOR_FILE_METADATA_MISSING")
    entries = {}
    for row in siblings:
        if not isinstance(row, dict) or not isinstance(row.get("rfilename"), str):
            raise ValueError("LOCAL_ACTOR_FILE_METADATA_INVALID")
        name = row["rfilename"]
        if name in entries:
            raise ValueError("LOCAL_ACTOR_DUPLICATE_FILENAME")
        if name in REQUIRED_FILES | OPTIONAL_FILES or name.endswith(".safetensors"):
            if Path(name).name != name or name.startswith("."):
                raise ValueError("LOCAL_ACTOR_UNSAFE_FILENAME")
            entries[name] = row
    if not REQUIRED_FILES <= entries.keys() or not any(n.endswith(".safetensors") for n in entries):
        raise ValueError("LOCAL_ACTOR_REQUIRED_FILES_MISSING")
    for row in entries.values():
        if type(row.get("size")) is not int or row["size"] <= 0:
            raise ValueError("LOCAL_ACTOR_FILE_SIZE_MISSING")
        lfs = row.get("lfs")
        if lfs is not None:
            if (not isinstance(lfs, dict) or not isinstance(lfs.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", lfs["sha256"]) is None
                    or lfs.get("size") != row["size"]):
                raise ValueError("LOCAL_ACTOR_LFS_IDENTITY_INVALID")
        elif not isinstance(row.get("blobId"), str) or PIN_PATTERN.fullmatch(row["blobId"]) is None:
            raise ValueError("LOCAL_ACTOR_GIT_BLOB_IDENTITY_MISSING")
    return entries


def verify_snapshot(directory: Path, metadata: dict, config: dict, *, tokenizer_only: bool = False) -> dict:
    """Reject substituted or incomplete inputs before any model is loaded."""
    entries = snapshot_entries(metadata, config)
    verified = {}
    for name, row in entries.items():
        if tokenizer_only and name.endswith(".safetensors"):
            continue
        path = directory / name
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise ValueError("LOCAL_ACTOR_FILE_SIZE_MISMATCH:" + name)
        sha = sha256_file(path)
        if row.get("lfs") is not None:
            if sha != row["lfs"]["sha256"]:
                raise ValueError("LOCAL_ACTOR_FILE_HASH_MISMATCH:" + name)
            identity_kind = "published_lfs_sha256"
        else:
            payload = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
            if blob != row["blobId"]:
                raise ValueError("LOCAL_ACTOR_FILE_HASH_MISMATCH:" + name)
            identity_kind = "published_git_blob_sha1"
        verified[name] = {"bytes": row["size"], "sha256": sha, "source_identity": identity_kind}
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map")
    if (not isinstance(mapping, dict) or not mapping
            or set(mapping.values()) != {name for name in entries if name.endswith(".safetensors")}):
        raise ValueError("LOCAL_ACTOR_WEIGHT_INDEX_MISMATCH")
    result = {"model_revision": model_identity(config), "files": verified,
            "bytes": sum(row["bytes"] for row in verified.values()),
            "all_published_file_identities_verified": not tokenizer_only}
    if tokenizer_only:
        result.update(verification_scope="published_non_weight_files_only", model_weights_verified=False)
    return result


def download_snapshot(directory: Path, metadata: dict, config: dict) -> None:
    """Explicit intake only; this is never called implicitly by inference."""
    entries = snapshot_entries(metadata, config)
    hub = importlib.import_module("huggingface_hub")
    model = config["model"]
    hub.snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                          local_dir=str(directory), allow_patterns=sorted(entries),
                          max_workers=1, token=False)


def request_from_receipt(actor: str, receipt: dict) -> dict:
    """Attach original text and measured costs without repairing the action."""
    try:
        action = json.loads(receipt["response"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LOCAL_ACTOR_RESPONSE_NOT_JSON") from exc
    if (not isinstance(action, dict) or action.get("actor") != actor
            or action.get("kind") not in {"tool", "message", "proposal", "finish"}
            or "generation" in action):
        raise ValueError("LOCAL_ACTOR_ACTION_ENVELOPE_INVALID")
    return {**action, "generation": deepcopy(receipt)}


class LocalTransformersActor:
    """One pinned model, two private conversation histories, no shared KV cache."""

    def __init__(self, directory: Path, config: dict, system_prompts: dict[str, str]):
        self.config = deepcopy(validate_config(config))
        if set(system_prompts) != set(ACTORS) or any(not isinstance(v, str) or not v for v in system_prompts.values()):
            raise ValueError("LOCAL_ACTOR_SYSTEM_PROMPTS_REQUIRED")
        # Import only for actual GPU inference; CPU intake/contract tests do not
        # need to import a model runtime.
        self.torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        torch = self.torch
        model_config = self.config["model"]
        count = torch.cuda.device_count()
        if not torch.cuda.is_available() or count != model_config["cuda_devices"]:
            raise ValueError("LOCAL_ACTOR_CUDA_ALLOCATION_MISMATCH")
        self.devices = list(range(count))
        memory_bytes = model_config["max_memory_gib_per_device"] * 1024**3
        if any(torch.cuda.get_device_properties(i).total_memory <= memory_bytes for i in self.devices):
            raise ValueError("LOCAL_ACTOR_DEVICE_MEMORY_BOUND_INVALID")
        torch.set_num_threads(1)
        torch.manual_seed(self.config["inference_seed"])
        for i in self.devices:
            with torch.cuda.device(i):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("LOCAL_ACTOR_BFLOAT16_UNSUPPORTED")
            torch.cuda.reset_peak_memory_stats(i)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False)
        started = time.perf_counter()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation=model_config["attention_implementation"],
            device_map="auto", max_memory={**{i: memory_bytes for i in self.devices}, "cpu": 0})
        self.model.eval()
        self._synchronize()
        self.model_load_seconds = time.perf_counter() - started
        if any(p.device.type != "cuda" or p.is_meta for p in self.model.parameters()):
            raise ValueError("LOCAL_ACTOR_CPU_OR_DISK_OFFLOAD_REJECTED")
        self.device_map = {key: str(value) for key, value in self.model.hf_device_map.items()}
        self.histories = {actor: [{"role": "system", "content": system_prompts[actor]}] for actor in ACTORS}
        self.session_ids = {actor: str(uuid.uuid4()) for actor in ACTORS}
        self.attempts: list[dict] = []
        self._action_schema_decoder = None

    @classmethod
    def for_prefix_replay(cls, tokenizer, config: dict, system_prompts: dict[str, str]):
        """Tokenizer-only reconstruction for CPU validation; cannot infer live actions."""
        if set(system_prompts) != set(ACTORS) or any(not isinstance(v, str) or not v for v in system_prompts.values()):
            raise ValueError("LOCAL_ACTOR_SYSTEM_PROMPTS_REQUIRED")
        actor = cls.__new__(cls)
        actor.config = deepcopy(validate_config(config))
        actor.tokenizer = tokenizer
        actor.histories = {name: [{"role": "system", "content": system_prompts[name]}] for name in ACTORS}
        actor.session_ids = {name: str(uuid.uuid4()) for name in ACTORS}
        actor.attempts = []
        actor._prefix_only = True
        return actor

    def replay_generation(self, actor: str, user_text: str, original: dict) -> dict:
        """Rebuild one private turn, preserving historical costs as provenance only.

        The existing controller receipt contract requires historical usage/time;
        these travel unchanged in the replayed receipt. The producer ledger marks
        them separately and records no newly measured inference for this turn.
        """
        if (actor not in ACTORS or not isinstance(user_text, str) or not user_text
                or original.get("actor") != actor or original.get("status") != "GENERATED"
                or original.get("model_revision") != model_identity(self.config)
                or not isinstance(original.get("response"), str)):
            raise ValueError("LOCAL_ACTOR_RECORDED_GENERATION_INVALID")
        messages = deepcopy(self.histories[actor]) + [{"role": "user", "content": user_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt != original.get("prompt"):
            raise ValueError("LOCAL_ACTOR_RECORDED_PROMPT_MISMATCH")
        record = {"actor": actor, "session_id": self.session_ids[actor],
                  "model_revision": model_identity(self.config), "prompt": prompt,
                  "response": original["response"], "status": "REPLAYED_PREFIX",
                  "usage": None, "inference_seconds": None, "reused_prefix_inference": True,
                  "source_generation": deepcopy(original)}
        self.attempts.append(record)
        self.histories[actor] = messages + [{"role": "assistant", "content": original["response"]}]
        receipt = {key: deepcopy(original[key]) for key in
                   ("session_id", "prompt", "response", "model_revision", "usage", "inference_seconds")}
        receipt["session_id"] = self.session_ids[actor]
        return receipt

    def _synchronize(self) -> None:
        for device in self.devices:
            self.torch.cuda.synchronize(device)

    def generate(self, actor: str, user_text: str, *, action_schema: dict | None = None,
                 action_decoder_max_whitespace: int | None = None) -> dict:
        if getattr(self, "_prefix_only", False):
            raise ValueError("LOCAL_ACTOR_PREFIX_ONLY_LIVE_GENERATION_FORBIDDEN")
        if actor not in ACTORS or not isinstance(user_text, str) or not user_text:
            raise ValueError("LOCAL_ACTOR_INPUT_INVALID")
        if action_decoder_max_whitespace is not None and action_schema is None:
            raise ValueError("LOCAL_ACTOR_ACTION_DECODER_WHITESPACE_REQUIRES_SCHEMA")
        messages = deepcopy(self.histories[actor]) + [{"role": "user", "content": user_text}]
        record = {"actor": actor, "session_id": self.session_ids[actor],
                  "model_revision": model_identity(self.config), "status": "PREPARING",
                  "usage": None, "inference_seconds": None}
        self.attempts.append(record)
        attempt_started = time.perf_counter()
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            record["prompt"] = prompt
            tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            prompt_tokens = int(tokens["input_ids"].shape[-1])
            record["prompt_tokens_before_generation"] = prompt_tokens
            generation = self.config["generation"]
            if prompt_tokens + generation["max_new_tokens"] > generation["max_context_tokens"]:
                raise ValueError("LOCAL_ACTOR_CONTEXT_BUDGET_EXCEEDED")
            generation_options = {}
            if action_schema is not None:
                metadata = {"schema_sha256": content_digest(action_schema), "schema": deepcopy(action_schema),
                            "callback_calls": 0, "callback_failures": 0, "callback_wall_seconds": 0.0}
                record["action_decoder"] = metadata
                prepared = time.perf_counter()
                try:
                    if self._action_schema_decoder is None:
                        self._action_schema_decoder = ActionSchemaDecoder(self.tokenizer)
                    generation_options["prefix_allowed_tokens_fn"] = self._action_schema_decoder.prepare(
                        action_schema, generation["max_new_tokens"], metadata,
                        max_consecutive_whitespaces=action_decoder_max_whitespace)
                finally:
                    metadata["preparation_wall_seconds"] = time.perf_counter() - prepared
            tokens = tokens.to(self.model.get_input_embeddings().weight.device)
            self._synchronize()
            generation_started = time.perf_counter()
            try:
                with self.torch.inference_mode():
                    output = self.model.generate(**tokens, max_new_tokens=generation["max_new_tokens"],
                                                 do_sample=False, use_cache=True, **generation_options)
                self._synchronize()
            finally:
                record["inference_seconds"] = time.perf_counter() - generation_started
            generated = output[0, prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            record["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                               "total_tokens": prompt_tokens + completion_tokens}
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            record.update(response=response, status="GENERATED",
                          reached_token_cap=completion_tokens == generation["max_new_tokens"])
            self.histories[actor] = messages + [{"role": "assistant", "content": response}]
            return {key: record[key] for key in
                    ("session_id", "prompt", "response", "model_revision", "usage", "inference_seconds")}
        except Exception as exc:
            record.update(status="FAILED", error_type=type(exc).__name__)
            if isinstance(exc, ValueError) and str(exc).startswith("LOCAL_ACTOR_"):
                record["error_code"] = str(exc)
            raise
        finally:
            record["host_attempt_wall_seconds"] = time.perf_counter() - attempt_started

    def runtime(self) -> dict:
        if getattr(self, "_prefix_only", False):
            return {"mode": "tokenizer_only_recorded_prefix", "live_generation_allowed": False,
                    "actor_contexts": dict(self.session_ids), "context_semantics": "separate_message_histories",
                    "kv_cache_shared_between_generations": False}
        decoder_attempts = [row["action_decoder"] for row in self.attempts if "action_decoder" in row]
        decoder_runtime = ({"action_decoder": {
            "attempt_count": len(decoder_attempts),
            "callback_calls": sum(row["callback_calls"] for row in decoder_attempts),
            "callback_failures": sum(row["callback_failures"] for row in decoder_attempts),
            "callback_wall_seconds": sum(row["callback_wall_seconds"] for row in decoder_attempts),
            "preparation_wall_seconds": sum(row.get("preparation_wall_seconds", 0.0) for row in decoder_attempts),
            "schema_sha256s": sorted({row["schema_sha256"] for row in decoder_attempts}),
        }} if decoder_attempts else {})
        return {"model_load_seconds": self.model_load_seconds, "device_map": self.device_map,
                **decoder_runtime,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "torch_version": self.torch.__version__, "cuda_version": self.torch.version.cuda,
                "devices": [{"index": i, "name": self.torch.cuda.get_device_name(i),
                             "total_memory_bytes": self.torch.cuda.get_device_properties(i).total_memory,
                             "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(i),
                             "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(i)}
                            for i in self.devices],
                "actor_contexts": dict(self.session_ids), "context_semantics": "separate_message_histories",
                "kv_cache_shared_between_generations": False}

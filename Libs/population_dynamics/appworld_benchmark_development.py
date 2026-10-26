"""One real development attempt with canonical settings and official prompts.

This actor CLI does not score outcomes. Invoke the separate native scoring CLI
only after this process exits. FunctionCalling requires a completed neutral
model qualification receipt for the declared model and serving settings.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any

import yaml

from .appworld_benchmark_runner import (
    DevelopmentExecutionError,
    TransportCapabilities,
    _prepare_agent_config,
    run_development_attempt,
)
from .appworld_benchmark_transport import LocalChatTransport
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS
from .provenance import freeze_json


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "Libs/configs/neural_population_appworld_benchmark.yaml"
_METHODS = ("official_react_code", "official_full_code", "official_function_calling")
_RATE_FIELDS = {"input_cache_miss", "input_cache_hit", "input_cache_write", "output"}
_PROMPT_FIELDS = ("prompt_file_path", "code_prompt_file_path", "retrial_prompt_file_path", "demo_messages_file_path")


@dataclass(frozen=True)
class PreparedOfficialAgent:
    agent_config: dict[str, Any]
    source_provenance: dict[str, Any]


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    # Population intake load_config requires a different campaign/cohort schema.
    # Reuse the project's YAML dependency without applying those intake gates.
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "appworld_benchmark_v1":
        raise ValueError("canonical appworld_benchmark_v1 YAML mapping required")
    return value


def validate_model_qualification(config: Mapping[str, Any], qualification: Mapping[str, Any]) -> None:
    """Validate the neutral qualification contract shared by owner and actor.

    The pipeline owner also checks its actual server packages against these
    qualified packages. The native actor consumes only the resulting receipt.
    """
    if (qualification.get("schema") != "appworld_model_runtime_qualification_v1"
            or qualification.get("scope") != "neutral_model_transport_only"
            or qualification.get("status") != "PASS" or qualification.get("tool_calls_validated") is not True
            or qualification.get("transport_probe_status") != "PASS"):
        raise ValueError("real neutral model qualification has not passed")
    if qualification.get("seeds") != list(EXPERIMENT_SEEDS) or any(type(seed) is not int for seed in qualification["seeds"]):
        raise ValueError("model qualification must retain exactly seeds 41, 42, 43")
    fields = ("repo_id", "revision", "serving_backend", "serving_version", "tool_call_parser")
    if any(qualification.get("model_identity", {}).get(key) != config["model"][key] for key in fields):
        raise ValueError("model identity differs from completed qualification")
    if qualification.get("config", {}).get("model_qualification") != config["model_qualification"]:
        raise ValueError("serving settings differ from completed qualification")
    packages = qualification.get("runtime", {}).get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("completed qualification has no model package identity")
    cleanup = qualification.get("server_cleanup", {})
    if (cleanup.get("server_parent_reaped") is not True or cleanup.get("forced_kill") is not False
            or cleanup.get("process_group_term_sent") is not True or cleanup.get("returncode_before_termination") is not None):
        raise ValueError("qualification did not finish owned-server cleanup")


def _git_output(source_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True, capture_output=True,
    ).stdout



def _archive_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate source archive manifest key: {key}")
        result[key] = value
    return result


def _read_source_archive(
    binding: Any, source_root: Path, repository: str, revision: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise ValueError("source.archive_manifest must provide path and sha256")
    declared_path, digest = binding.get("path"), binding.get("sha256")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("source.archive_manifest.path must be an explicit file path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("source.archive_manifest.sha256 must be an exact SHA-256")
    manifest_path = Path(declared_path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = source_root / manifest_path
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("source archive manifest must be a regular non-symlink file")
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("source archive manifest byte hash differs from its binding")
    manifest = json.loads(payload, object_pairs_hook=_archive_json_object)
    if not isinstance(manifest, dict) or manifest.get("schema") != "appworld_source_archive_manifest_v1":
        raise ValueError("appworld_source_archive_manifest_v1 is required")
    if manifest.get("repository") != repository or manifest.get("revision") != revision:
        raise ValueError("source archive manifest repository or revision differs from the inspected source")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("source archive manifest requires a nonempty files mapping")
    return manifest_path.resolve(strict=True), manifest


def _verified_archive_file(source_root: Path, relative: Any, record: Any) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or "\x00" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("archive source paths must be canonical relative paths without traversal")
    if (not isinstance(record, Mapping) or type(record.get("bytes")) is not int
            or record["bytes"] < 0 or not isinstance(record.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise ValueError(f"malformed archive file record: {relative}")
    current = source_root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError(f"archive source file is missing: {relative}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"archive source path contains a symlink: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ValueError(f"archive source parent is not a directory: {relative}")
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise ValueError(f"archive source entry is not a regular file: {relative}")
    payload = current.read_bytes()
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"archive source file differs from bound bytes: {relative}")
    return current


def _admit_source(config: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    source = config.get("source", {})
    if source.get("repository") != "StonyBrookNLP/appworld":
        raise ValueError("the declared source repository must be StonyBrookNLP/appworld")
    revision = source.get("inspected_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source.inspected_revision must be an exact Git commit")
    if "archive_manifest" in source:
        manifest_path, manifest = _read_source_archive(
            source["archive_manifest"], source_root, source["repository"], revision,
        )
        for relative, record in manifest["files"].items():
            _verified_archive_file(source_root, relative, record)
        return {
            "repository": source["repository"], "revision": revision, "root": str(source_root),
            "admission_method": "source_archive_manifest",
            "archive_manifest": {
                "path": str(manifest_path), "sha256": source["archive_manifest"]["sha256"],
                "schema": manifest["schema"], "file_count": len(manifest["files"]),
            },
            "prompts": [],
        }
    top_level = Path(_git_output(source_root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top_level != source_root:
        raise ValueError("source_root must be the root of the admitted AppWorld checkout")
    actual_revision = _git_output(source_root, "rev-parse", "HEAD").decode().strip()
    if actual_revision != revision:
        raise ValueError("AppWorld source revision differs from source.inspected_revision")
    origin = _git_output(source_root, "remote", "get-url", "origin").decode().strip()
    allowed_origins = {
        "https://github.com/StonyBrookNLP/appworld", "https://github.com/StonyBrookNLP/appworld.git",
        "git@github.com:StonyBrookNLP/appworld.git", "ssh://git@github.com/StonyBrookNLP/appworld.git",
    }
    if origin not in allowed_origins:
        raise ValueError("source origin is not the declared official AppWorld repository")
    return {"repository": source["repository"], "revision": actual_revision, "root": str(source_root), "origin": origin, "prompts": []}


def _resolve_prompt(source_root: Path, relative: Any, source: dict[str, Any]) -> str:
    if not isinstance(relative, str):
        raise ValueError("official prompt paths must be explicit source-relative strings")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("experiments", "prompts"):
        raise ValueError("prompt must originate under the admitted experiments/prompts directory")
    if "archive_manifest" in source:
        _, manifest = _read_source_archive(
            source["archive_manifest"], source_root, source["repository"], source["revision"],
        )
        relative_path = path.as_posix()
        if relative_path not in manifest["files"]:
            raise ValueError(f"official prompt is not listed in the bound source archive: {relative_path}")
        record = manifest["files"][relative_path]
        prompt = _verified_archive_file(source_root, relative_path, record)
        source["prompts"].append({
            "relative_path": relative_path, "path": str(prompt),
            "bytes": record["bytes"], "sha256": record["sha256"],
            "matches_admitted_archive_bytes": True,
        })
        return str(prompt)
    prompt = (source_root / path).resolve(strict=True)
    if not prompt.is_relative_to(source_root) or not prompt.is_file():
        raise ValueError("prompt must remain a file inside the admitted source checkout")
    relative_path = path.as_posix()
    _git_output(source_root, "ls-files", "--error-unmatch", "--", relative_path)
    committed = _git_output(source_root, "show", f"{source['revision']}:{relative_path}")
    if prompt.read_bytes() != committed:
        raise ValueError(f"tracked official prompt was modified: {relative_path}")
    source["prompts"].append({"relative_path": relative_path, "path": str(prompt), "matches_admitted_git_bytes": True})
    return str(prompt)


def prepare_official_agent(
    config: Mapping[str, Any], method: str, source_root: str | Path,
) -> PreparedOfficialAgent:
    """Build native kwargs from YAML and verify only the listed source prompts."""
    if config.get("schema") != "appworld_benchmark_v1" or method not in _METHODS:
        raise ValueError("a canonical benchmark config and supported official method are required")
    development = config.get("development_execution")
    if not isinstance(development, Mapping) or development.get("status") != "provisional":
        raise ValueError("provisional development_execution configuration required")
    accounting = development.get("monetary_accounting", {})
    rates = accounting.get("native_bookkeeping_cost_per_token")
    if accounting.get("status") != "unpriced_local_inference" or not isinstance(rates, Mapping) or set(rates) != _RATE_FIELDS:
        raise ValueError("explicit unpriced local-inference bookkeeping rates required")
    if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate != 0 for rate in rates.values()):
        raise ValueError("unpriced local inference uses zero bookkeeping rates, not a measured dollar cost")
    model = deepcopy(dict(development["model"]))
    if model.get("client_name") != "openai" or model.get("api_type") != "chat_completions":
        raise ValueError("the local transport requires the declared OpenAI chat-completion interface")
    model["name"] = config["model"]["repo_id"]
    model["cost_per_token"] = deepcopy(dict(rates))
    source_root = Path(source_root).expanduser().resolve(strict=True)
    source = _admit_source(config, source_root)
    kwargs = deepcopy(dict(development["methods"][method]))
    model_overrides = kwargs.pop("model_overrides", {})
    if model_overrides and (method != "official_function_calling" or model_overrides != {"tool_choice": "auto"}):
        raise ValueError("only the source FunctionCalling tool_choice=auto model override is supported")
    primary_model = model | model_overrides
    appworld_config = deepcopy(dict(development["appworld_config"])) | kwargs.pop("appworld_config", {})
    if appworld_config.get("raise_on_extra_parameters") is not True:
        raise ValueError("native raise_on_extra_parameters must remain enabled")
    kwargs.update(
        model_config=primary_model, appworld_config=appworld_config,
        usage_tracker_config=deepcopy(dict(development["usage_tracker_config"])),
    )
    for field in _PROMPT_FIELDS:
        if kwargs.get(field) is not None:
            kwargs[field] = _resolve_prompt(source_root, kwargs[field], source)
    if method in {"official_full_code", "official_function_calling"}:
        predictor = deepcopy(dict(development["api_predictor"]))
        predictor["model_config"] = deepcopy(model)
        predictor["prompt_file_path"] = _resolve_prompt(source_root, predictor["prompt_file_path"], source)
        kwargs["api_predictor_config"] = predictor
    return PreparedOfficialAgent(agent_config=kwargs, source_provenance=source)


def run_configured_development_attempt(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, task_id: str,
    method: str, seed: int, source_root: str | Path, run_root: str | Path,
    model_url: str, timeout_seconds: float | None = None,
    config_path: str | Path | None = None, manifest_path: str | Path | None = None,
    model_qualification: str | Path | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run an unscored development attempt and retain invocation/transport data."""
    if split not in {"train", "dev"}:
        raise ValueError("test/formal splits remain closed in the development CLI")
    root = Path(run_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"existing or unresolved run root cannot be reused: {root}")
    capabilities = TransportCapabilities()
    if model_qualification is not None:
        qualification_path = Path(model_qualification).expanduser().resolve(strict=True)
        validate_model_qualification(config, json.loads(qualification_path.read_text()))
        capabilities = TransportCapabilities(tool_calls_validated=True, validation_reference=str(qualification_path))
    elif method == "official_function_calling":
        raise ValueError("FunctionCalling requires a passed real-model qualification receipt")
    prepared = prepare_official_agent(config, method, source_root)
    # Reuse the same metadata/seed/demo gate as the native runner. The resulting
    # kwargs include its enforced settings and are recorded before invocation.
    kwargs = _prepare_agent_config(
        config, manifest, prepared.agent_config, split, task_id, method, seed, capabilities,
    )
    development = config["development_execution"]
    timeout = development["transport"]["timeout_seconds"] if timeout_seconds is None else timeout_seconds
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("transport timeout must be a positive finite number")
    context_window_tokens = (
        config["model_qualification"]["max_model_len"]
        if config["model"].get("serving_backend") == "vllm"
        and config["model"].get("serving_version") == "0.11.2" else None
    )
    transport = LocalChatTransport(
        model_url, timeout_seconds=timeout, context_window_tokens=context_window_tokens,
    )
    invocation_path = root / "development_invocation.json"
    invocation: dict[str, Any] = {
        "schema": "appworld_development_invocation_v1", "scope": "development_only", "formal_eligible": False,
        "split": split, "task_id": task_id, "method": method, "seed": seed,
        "status": "running", "execution_status": None, "source": prepared.source_provenance,
        "model_identity": deepcopy(config["model"]), "agent_config": deepcopy(kwargs),
        "model_url": transport.base_url, "transport_timeout_seconds": timeout,
        "transport_context_window_tokens": context_window_tokens,
        "monetary_accounting": deepcopy(development["monetary_accounting"]), "measured_monetary_cost": None,
        "budget_adaptations": deepcopy(development["budget_adaptations"]),
        "configuration_status": "provisional_not_formally_frozen",
        "config_path": str(config_path) if config_path is not None else None,
        "tool_transport_validation_reference": capabilities.validation_reference,
        "split_manifest_path": str(manifest_path) if manifest_path is not None else None,
        "command": list(command) if command is not None else None,
        "execution_receipt_path": str(root / "execution_receipt.json"),
        "started_at": datetime.now(timezone.utc).isoformat(), "transport_records": [],
    }
    retain_invocation = False
    try:
        receipt = run_development_attempt(
            config, manifest, split=split, task_id=task_id, method=method, seed=seed,
            agent_config=kwargs, run_root=root, lm_call=transport,
            transport_capabilities=capabilities,
            invocation_metadata={
                key: deepcopy(value) for key, value in invocation.items()
                if key not in {"status", "execution_status", "transport_records"}
            },
        )
        retain_invocation = True
        invocation.update(status="execution_returned", execution_status=receipt["status"])
    except DevelopmentExecutionError as error:
        retain_invocation = True
        invocation.update(status="execution_failed", error_type=type(error).__name__, error=str(error), execution_receipt_path=str(error.receipt_path))
        raise
    except BaseException as error:
        # A transport record proves the native boundary reached model execution.
        # Do not write into an unrelated directory from a rejected/reused run.
        retain_invocation = bool(transport.records)
        invocation.update(status="execution_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if retain_invocation:
            invocation["transport_records"] = deepcopy(transport.records)
            invocation["ended_at"] = datetime.now(timezone.utc).isoformat()
            if invocation_path.exists() or invocation_path.is_symlink():
                raise FileExistsError(f"invocation receipt cannot be overwritten: {invocation_path}")
            freeze_json(invocation_path, invocation)
    return invocation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--split", required=True, help="train or dev only")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--method", required=True, choices=_METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=(41, 42, 43))
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--model-qualification", type=Path)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        config = load_benchmark_config(args.config)
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        source_root = args.source_root or Path(config["development_execution"]["source_root"])
        if not source_root.is_absolute():
            source_root = _PROJECT_ROOT / source_root
        invocation = run_configured_development_attempt(
            config, manifest, split=args.split, task_id=args.task_id, method=args.method, seed=args.seed,
            source_root=source_root, run_root=args.run_root, model_url=args.model_url, timeout_seconds=args.timeout_seconds,
            config_path=args.config, manifest_path=args.split_manifest,
            model_qualification=args.model_qualification,
            command=[sys.executable, "-m", "population_dynamics.appworld_benchmark_development", *arguments],
        )
    except DevelopmentExecutionError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"development invocation rejected: {error}", file=sys.stderr)
        return 2
    print(f"{invocation['execution_status']}; invocation receipt: {Path(args.run_root) / 'development_invocation.json'}")
    return 0 if invocation["execution_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

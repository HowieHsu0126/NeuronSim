"""Development-only execution of the pinned official simplified AppWorld agents.

Run one attempt per allocated process on node-local scratch. This boundary admits
split identities and delegates to native ``Agent.solve_tasks``. It does not
load a target Task itself, score a task, or establish benchmark eligibility.
The manifest data_root is the AppWorld root; native data live in data_root/data.
Formal execution remains unavailable even when a caller changes released=True;
source, method and budget freezes require the subsequent formal runner.

``lm_call`` is a Python callable returning an OpenAI chat-completion dictionary.
The callable receives native LanguageModel arguments and must not cache or retry
ambiguous requests. A proven pre-generation rejection may receive one recorded
context-cap correction; each returned completion retains its actual usage. Preserve native
LanguageModel.generate and its Usage objects; observe transport and returned
usage so missing/corrupt counts cannot become measured zero-cost evidence.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
from types import SimpleNamespace
from typing import Any

from .appworld_benchmark_execution import execution_identity
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, validate_split_manifest
from .appworld_benchmark_transport import ChatContextBudgetExceeded
from .provenance import canonical_json, sha256_file


_AGENT_CLASSES = {
    "official_react_code": ("react_code_agent", "SimplifiedReActCodeAgent"),
    "official_full_code": ("full_code_agent", "SimplifiedFullCodeAgent"),
    "official_function_calling": ("function_calling_agent", "SimplifiedFunctionCallingAgent"),
}
_OFFICIAL_SPLITS = {"train", "dev", "test_normal", "test_challenge"}
_COST_FIELDS = {"input_cache_miss", "input_cache_hit", "input_cache_write", "output"}
_LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "zfs", "overlay"}


@dataclass(frozen=True)
class TransportCapabilities:
    tool_calls_validated: bool = False
    validation_reference: str | None = None


class DevelopmentExecutionError(RuntimeError):
    """Execution failed after a fresh run root was created; inspect its receipt."""

    def __init__(self, message: str, receipt_path: Path):
        super().__init__(f"{message}; execution receipt: {receipt_path}")
        self.receipt_path = receipt_path


def _load_native() -> SimpleNamespace:
    # Keep native imports behind the execution boundary, after local paths exist.
    appworld = importlib.import_module("appworld")
    task_module = importlib.import_module("appworld.task")
    path_module = importlib.import_module("appworld.common.path_store")
    return SimpleNamespace(
        AppWorld=appworld.AppWorld,
        load_task_ids=task_module.load_task_ids,
        path_store=path_module.path_store,
        version=importlib.metadata.version("appworld"),
    )


def _load_agent_class(method: str) -> type:
    module, class_name = _AGENT_CLASSES[method]
    return getattr(importlib.import_module(f"appworld_agents.code.simplified.{module}"), class_name)


def _check_execution_location(run_root: Path) -> None:
    execution_identity(require_allocation=True)
    if not run_root.parent.is_dir():
        raise ValueError("run_root parent must be an existing node-local scratch directory")
    result = subprocess.run(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(run_root.parent)],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip() not in _LOCAL_FILESYSTEMS:
        raise ValueError(f"run_root is not on an admitted node-local filesystem: {result.stdout.strip()}")


def _model_config(value: Any, seed: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each native model consumer requires model_config")
    model = deepcopy(dict(value))
    rates = model.get("cost_per_token")
    if not isinstance(rates, Mapping) or set(rates) != _COST_FIELDS:
        raise ValueError("explicit cost_per_token requires all four native rate fields")
    for rate in rates.values():
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
            raise ValueError("cost_per_token rates must be explicit finite nonnegative numbers")
    if "seed" in model and model["seed"] != seed:
        raise ValueError("native model seed conflicts with the attempt seed")
    if model.get("api_type", "chat_completions") != "chat_completions":
        raise ValueError("the supplied callable contract requires chat_completions")
    model.update(use_cache=False, max_retries=1, retry_after_n_seconds=0, seed=seed)
    return model


def _demo_ids(value: Any, train_ids: set[str], target_id: str, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an explicit list of admitted train task IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate task IDs")
    if target_id in value or any(task_id not in train_ids for task_id in value):
        raise ValueError(f"{label} must resolve to train and must exclude the target task")


def _prepare_agent_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any], agent_config: Mapping[str, Any],
    split: str, task_id: str, method: str, seed: int, capabilities: TransportCapabilities,
) -> dict[str, Any]:
    if manifest.get("schema") != "appworld_split_membership_v1":
        raise ValueError("an admitted appworld_split_membership_v1 manifest is required")
    memberships = validate_split_manifest(manifest)
    if set(memberships).difference(_OFFICIAL_SPLITS):
        raise ValueError("only original AppWorld benchmark split names are supported")
    if tuple(config.get("seeds", ())) != EXPERIMENT_SEEDS or any(type(s) is not int for s in config.get("seeds", ())):
        raise ValueError("canonical config seeds must be exactly [41, 42, 43]")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("attempt seed must be exactly one of 41, 42, 43")
    if split not in {"train", "dev"}:
        reason = "FORMAL_RELEASE_CLOSED" if not config.get("formal", {}).get("released", False) else "DEVELOPMENT_ONLY_BOUNDARY"
        raise ValueError(f"{reason}: formal task execution is unavailable in this runner")
    if split not in config.get("splits", {}).get("development", ()):
        raise ValueError("split is not declared for development in the canonical config")
    if split not in memberships or task_id not in memberships[split]:
        raise ValueError("target task is absent from the admitted development membership")
    if method not in _AGENT_CLASSES or method not in config.get("methods", {}):
        raise ValueError("method is not a registered supported official simplified agent")
    if method == "official_function_calling" and (
        capabilities.tool_calls_validated is not True
        or not isinstance(capabilities.validation_reference, str)
        or not capabilities.validation_reference.strip()
    ):
        raise ValueError("FunctionCalling requires validated tool-call transport and a validation reference")
    if not isinstance(agent_config, Mapping):
        raise ValueError("agent_config must contain explicit native constructor arguments")
    kwargs = deepcopy(dict(agent_config))
    appworld_config = kwargs.get("appworld_config", {})
    if not isinstance(appworld_config, Mapping):
        raise ValueError("appworld_config must be a mapping")
    appworld_config = dict(appworld_config)
    if appworld_config.get("load_ground_truth", False) is not False:
        raise ValueError("target load_ground_truth must be False")
    if appworld_config.get("ground_truth_mode") is not None:
        raise ValueError("target ground_truth_mode is unavailable")
    if any(value is not None for key, value in appworld_config.items() if key.startswith("remote_")):
        raise ValueError("development execution uses a local native environment")
    if "random_seed" in appworld_config and appworld_config["random_seed"] != seed:
        raise ValueError("AppWorld random_seed conflicts with the attempt seed")
    appworld_config.update(load_ground_truth=False, random_seed=seed)
    kwargs["appworld_config"] = appworld_config
    kwargs["model_config"] = _model_config(kwargs.get("model_config"), seed)
    if kwargs.get("skip_if_finished", False) is not False:
        raise ValueError("skip_if_finished is unavailable for a fresh attempt")
    kwargs.update(skip_if_finished=False, log_lm_calls=True)
    train_ids = set(memberships.get("train", ()))
    if method == "official_full_code":
        _demo_ids(kwargs.get("demo_task_ids"), train_ids, task_id, "FullCode demo_task_ids")
    if method in {"official_full_code", "official_function_calling"}:
        predictor = kwargs.get("api_predictor_config")
        if not isinstance(predictor, Mapping):
            raise ValueError("this native method requires api_predictor_config")
        predictor = deepcopy(dict(predictor))
        if predictor.get("mode", "predicted") == "ground_truth":
            raise ValueError("ground_truth API predictor mode is prohibited")
        _demo_ids(predictor.get("demo_task_ids"), train_ids, task_id, "API predictor demo_task_ids")
        if predictor.get("mode", "predicted") == "predicted" and not predictor["demo_task_ids"]:
            raise ValueError("native predicted API mode requires at least one train demo")
        predictor["model_config"] = _model_config(predictor.get("model_config"), seed)
        kwargs["api_predictor_config"] = predictor
    return kwargs


def _verify_memberships(native: SimpleNamespace, manifest: Mapping[str, Any], data_root: Path) -> None:
    if native.version != manifest["appworld_version"]:
        raise ValueError("native AppWorld package version differs from admitted manifest")
    for split, entry in manifest["splits"].items():
        expected_file = (data_root / "datasets" / f"{split}.txt").resolve(strict=True)
        declared_file = Path(entry["membership_file"]).resolve(strict=True)
        if declared_file != expected_file or sha256_file(declared_file) != entry["membership_sha256"]:
            raise ValueError(f"native split-file identity mismatch: {split}")
        # Do not pass difficulty: that filter reads ground-truth metadata.
        task_ids = native.load_task_ids(dataset_name=split)
        if task_ids != entry["task_ids"]:
            raise ValueError(f"official loader membership differs from admitted IDs: {split}")


def _usage_is_resolved(raw: Any, standardized: Any) -> bool:
    if not isinstance(raw, Mapping) or not isinstance(standardized, Mapping):
        return False
    if any(type(raw.get(field)) is not int or raw[field] < 0 for field in ("prompt_tokens", "completion_tokens", "total_tokens")):
        return False
    tokens = standardized.get("tokens", {})
    if not isinstance(tokens, Mapping):
        return False
    fields = ("input_cache_miss", "input_cache_hit", "output")
    if any(type(tokens.get(field)) not in (int, float) or not math.isfinite(tokens[field]) or tokens[field] < 0 for field in fields):
        return False
    input_tokens = tokens["input_cache_miss"] + tokens["input_cache_hit"]
    return (
        raw["prompt_tokens"] + raw["completion_tokens"] == raw["total_tokens"]
        and input_tokens == raw["prompt_tokens"]
        and tokens["output"] == raw["completion_tokens"]
        and input_tokens + tokens["output"] == raw["total_tokens"]
    )


def _observe_model(
    model: Any, consumer: str, lm_call: Callable[..., dict[str, Any]],
    calls: list[dict[str, Any]], agent_failures: list[str],
) -> None:
    original_generate = model.generate

    def observed_call(**kwargs: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "consumer": consumer, "model": getattr(model, "model", None),
            "raw_response": None, "response_normalizations": [],
            "raw_usage": None, "official_usage": None, "usage_status": "unresolved",
        }
        calls.append(record)
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        try:
            response = lm_call(**kwargs)
            record["transport_status"] = "returned"
            record["raw_response"] = deepcopy(response)
            if isinstance(response, Mapping):
                record["raw_usage"] = deepcopy(response.get("usage"))
            return response
        except BaseException as error:
            record.update(transport_status="failed", error_type=type(error).__name__, error=str(error))
            if isinstance(error, ChatContextBudgetExceeded):
                record.update(
                    transport_status="pre_generation_rejected",
                    generation_started=False, generated_completion_tokens=0,
                    transport_request_id=error.request_id,
                    context_budget_rejection=deepcopy(error.context_budget_rejection),
                )
            raise
        finally:
            record["transport_wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started

    def observed_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        first_call = len(calls)
        output = original_generate(*args, **kwargs)
        new_calls = calls[first_call:]
        if "reasoning_content" in output and output["reasoning_content"] is None:
            # Native ReAct strips this optional field; an explicit null is absent.
            output = output.copy()
            del output["reasoning_content"]
            for record in new_calls:
                record["response_normalizations"].append({
                    "stage": "native_generate_output",
                    "field": "reasoning_content", "operation": "omit_null",
                })
        if output.get("error"):
            agent_failures.append(str(output["error"]))
        usage = output.get("standardized_usage")
        if len(new_calls) == 1 and usage is not None:
            standardized = usage.dict()
            new_calls[0]["official_usage"] = deepcopy(standardized)
            if _usage_is_resolved(new_calls[0]["raw_usage"], standardized):
                new_calls[0]["usage_status"] = "resolved"
        return output

    model.lm_call = observed_call
    model.generate = observed_generate


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".pending.json")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(receipt))
    os.replace(temporary, path)


def run_development_attempt(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, task_id: str,
    method: str, seed: int, agent_config: Mapping[str, Any], run_root: str | Path,
    lm_call: Callable[..., dict[str, Any]],
    transport_capabilities: TransportCapabilities = TransportCapabilities(),
    invocation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one fresh native attempt and return its unscored terminal receipt.

    Admission failures before root creation raise ValueError/FileExistsError.
    Failures after root creation retain a terminal receipt and raise
    DevelopmentExecutionError. Interrupted processes leave evidence for recovery;
    an existing root is never reused, resumed, removed, or overwritten.
    Supplied monetary rates remain unverified accounting inputs. Native token
    usage and transport timing are recorded separately from monetary claims.
    The initial receipt records independent copies of invocation_metadata and
    resolved agent_config before native source or model initialization.
    """
    if not callable(lm_call):
        raise ValueError("lm_call must be a Python callable")
    if invocation_metadata is not None and not isinstance(invocation_metadata, Mapping):
        raise ValueError("invocation_metadata must be a mapping when supplied")
    kwargs = _prepare_agent_config(
        config, manifest, agent_config, split, task_id, method, seed, transport_capabilities,
    )
    root = Path(run_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"existing or unresolved run_root cannot be reused: {root}")
    _check_execution_location(root)
    source_root = Path(manifest["data_root"]).resolve(strict=True)
    data_root = (source_root / "data").resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("manifest data_root/data must be the native data directory")
    root.mkdir()
    receipt_path = root / "execution_receipt.json"
    calls: list[dict[str, Any]] = []
    agent_failures: list[str] = []
    receipt: dict[str, Any] = {
        "schema": "appworld_development_execution_v1", "scope": "development_only",
        **execution_identity(),
        "formal_eligible": False, "native_tgc": None, "scoring_status": "not_run",
        "split": split, "task_id": task_id, "method": method, "seed": seed,
        "appworld_version": manifest["appworld_version"], "run_root": str(root),
        "data_root": str(source_root), "native_data_directory": str(data_root),
        "experiment_name": "development",
        "status": "running", "terminal": False, "stage": "source_admission",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_calls": calls, "agent_failures": agent_failures,
        "resolved_agent_config": deepcopy(kwargs),
        "invocation_metadata": deepcopy(dict(invocation_metadata)) if invocation_metadata is not None else None,
        "monetary_cost_status": "caller_supplied_rates_not_verified",
        "tool_transport_validation_reference": transport_capabilities.validation_reference,
        "cache_enabled": False, "native_lm_max_retries": 1,
        "target_load_ground_truth": False, "official_solve_returned": False,
    }
    _write_receipt(receipt_path, receipt)
    saved_environment = {name: os.environ.get(name) for name in ("APPWORLD_ROOT", "APPWORLD_CACHE")}
    native = None
    saved_defaults = None
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    failure: BaseException | None = None
    try:
        (root / "data").symlink_to(data_root, target_is_directory=True)
        os.environ["APPWORLD_ROOT"] = str(root)
        os.environ["APPWORLD_CACHE"] = str(root / ".cache")
        native = _load_native()
        native.path_store.update_root(str(root))
        native.path_store.reload()
        saved_defaults = deepcopy(native.AppWorld.init_defaults)
        _verify_memberships(native, manifest, data_root)
        receipt["membership_verified"] = True
        receipt["stage"] = "agent_initialization"
        agent = _load_agent_class(method)(**kwargs)
        _observe_model(agent.language_model, "primary", lm_call, calls, agent_failures)
        if method in {"official_full_code", "official_function_calling"}:
            _observe_model(agent.api_predictor.language_model, "api_predictor", lm_call, calls, agent_failures)
        receipt["stage"] = "native_solve_tasks"
        _write_receipt(receipt_path, receipt)
        agent.solve_tasks([task_id], experiment_name="development", num_processes=1, process_index=0)
        receipt["official_solve_returned"] = True
        receipt["status"] = "agent_failure" if agent_failures else "completed"
    except ChatContextBudgetExceeded as error:
        agent_failures.append(str(error))
        receipt.update(
            status="agent_failure", stop_reason="context_budget_exhausted",
            error_type=type(error).__name__, error=str(error),
        )
    except BaseException as error:
        failure = error
        receipt.update(
            status="unknown" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "infrastructure_failure",
            error_type=type(error).__name__, error=str(error),
            error_traceback=traceback.format_exc(),
        )
    finally:
        if native is not None and saved_defaults is not None:
            native.AppWorld.init_defaults = saved_defaults
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if native is not None:
            native.path_store.reload()
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
        receipt.update(
            terminal=True, ended_at=datetime.now(timezone.utc).isoformat(),
            execution_wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - started,
            lm_call_count=len(calls), usage_complete=all(call["usage_status"] == "resolved" for call in calls),
        )
        _write_receipt(receipt_path, receipt)
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    if failure is not None:
        raise DevelopmentExecutionError(f"native development execution failed: {failure}", receipt_path) from failure
    return receipt

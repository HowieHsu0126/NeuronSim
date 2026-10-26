"""Run one unscored, two-actor development attempt in a native AppWorld.

The CLI exposes collection and fixed-control policies only. Neither is NeuronSim.
Use a fresh Slurm process and node-local run root. Invoke the existing separate
scoring CLI after this process exits. Formal task execution remains unavailable.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .appworld_benchmark_collaboration import CollaborationHistory, PersistentToolSession
from .appworld_benchmark_development import (
    _admit_source, _git_output, _read_source_archive, _verified_archive_file,
    load_benchmark_config,
)
from .appworld_benchmark_host import (
    CollaborationHost, FixedVerificationPolicy, HostBudget, ModelReply,
    UniformVerificationPolicy, full_visible_history, source_api_methods,
)
from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, validate_split_manifest
from .appworld_benchmark_runner import (
    DevelopmentExecutionError, _check_execution_location, _load_native,
    _model_config, _observe_model, _verify_memberships, _write_receipt,
)
from .appworld_benchmark_transport import LocalChatTransport


_MODEL_MODULE = "appworld_agents.code.simplified.language_model"
_MODEL_SOURCE = "experiments/code/simplified/language_model.py"
_SUPERVISOR_FIELDS = ("first_name", "last_name", "email", "phone_number")


def _language_model_class(source_root: Path, source: dict[str, Any]) -> type:
    model_class = importlib.import_module(_MODEL_MODULE).LanguageModel
    filename = inspect.getsourcefile(model_class)
    if filename is None:
        raise ValueError("installed native LanguageModel has no inspectable source file")
    path = Path(filename).resolve(strict=True)
    provenance = {"module": _MODEL_MODULE, "file": str(path), "git_path": _MODEL_SOURCE}
    if "archive_manifest" in source:
        _, manifest = _read_source_archive(
            source["archive_manifest"], source_root, source["repository"], source["revision"],
        )
        if _MODEL_SOURCE not in manifest["files"]:
            raise ValueError("native LanguageModel is not listed in the bound source archive")
        record = manifest["files"][_MODEL_SOURCE]
        archived = _verified_archive_file(source_root, _MODEL_SOURCE, record)
        expected = archived.read_bytes()
        provenance["matches_admitted_archive_bytes"] = True
    else:
        expected = _git_output(source_root, "show", f"{source['revision']}:{_MODEL_SOURCE}")
        provenance["matches_admitted_git_bytes"] = True
    if path.read_bytes() != expected:
        raise ValueError("installed LanguageModel differs from the pinned official source")
    source["language_model"] = provenance
    return model_class


def _public_context(task: Any) -> dict[str, Any]:
    # Read precisely the metadata exposed by the official simplified agents.
    # Do not traverse Task.__dict__, which can contain a ground-truth object.
    supervisor = task.supervisor
    main_user = {
        field: supervisor[field] if isinstance(supervisor, Mapping) else getattr(supervisor, field)
        for field in _SUPERVISOR_FIELDS
    }
    docs = task.api_docs
    if "api_docs" not in docs or "supervisor" not in docs:
        raise ValueError("native public documentation and supervisor apps are required")
    bootstrap_docs = {app: {api: deepcopy(dict(record)) for api, record in docs[app].items()}
                      for app in ("api_docs", "supervisor")}
    return {"main_user": main_user, "app_descriptions": deepcopy(task.app_descriptions),
            "bootstrap_api_docs": bootstrap_docs}


def _model_generator(models: Mapping[str, Any], calls: list[dict[str, Any]]):
    def generate(*, actor: str, messages: list[dict[str, Any]], max_completion_tokens: int) -> ModelReply:
        start = len(calls)
        reply = models[actor].generate(
            messages=messages, max_completion_tokens=max_completion_tokens, max_tokens=None,
        )
        if not isinstance(reply, Mapping):
            raise TypeError("native LanguageModel must return an assistant-message mapping")
        new_calls = calls[start:]
        resolved = len(new_calls) == 1 and new_calls[0]["usage_status"] == "resolved"
        tokens = new_calls[0]["raw_usage"]["completion_tokens"] if resolved else None
        message = {key: deepcopy(value) for key, value in reply.items() if key not in {"standardized_usage", "error"}}
        return ModelReply(message, tokens, resolved, reply.get("error"))
    return generate


def run_collaboration_development_attempt(
    config: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, task_id: str,
    seed: int, policy: str, verification_budget: int, max_model_calls: int,
    source_root: str | Path, run_root: str | Path, model_url: str,
    planner_calls: int | None = None, command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run collection/fixed control, preserving a scorer-compatible receipt."""
    if config.get("schema") != "appworld_benchmark_v1":
        raise ValueError("canonical appworld_benchmark_v1 configuration required")
    if manifest.get("schema") != "appworld_split_membership_v1":
        raise ValueError("an admitted native split manifest is required")
    if split not in {"train", "dev"} or split not in config.get("splits", {}).get("development", ()):
        raise ValueError("test/formal splits are unavailable in this development runner")
    if tuple(config.get("seeds", ())) != EXPERIMENT_SEEDS or any(type(value) is not int for value in config["seeds"]):
        raise ValueError("canonical seeds must be exactly [41, 42, 43]")
    if type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("seed must be 41, 42 or 43")
    memberships = validate_split_manifest(manifest)
    if task_id not in memberships.get(split, ()):
        raise ValueError("target task is absent from admitted development membership")
    if type(verification_budget) is not int or verification_budget < 0:
        raise ValueError("verification budget must be a nonnegative integer")
    if policy == "uniform":
        if split != "train" or planner_calls is not None:
            raise ValueError("uniform collection is train-only and does not accept a fixed allocation")
        index = manifest["splits"]["train"]["task_ids"].index(task_id)
        allocation_policy = UniformVerificationPolicy(seed, index)
    elif policy == "fixed":
        allocation_policy = FixedVerificationPolicy(planner_calls)
        if planner_calls > verification_budget:
            raise ValueError("fixed planner allocation exceeds total verification budget")
    else:
        raise ValueError("only explicit uniform collection or fixed control is implemented")
    development = config["development_execution"]
    if development.get("status") != "provisional":
        raise ValueError("this runner requires provisional development settings")
    rates = development["monetary_accounting"]["native_bookkeeping_cost_per_token"]
    if development["monetary_accounting"].get("status") != "unpriced_local_inference" or any(value != 0 for value in rates.values()):
        raise ValueError("explicit unpriced local bookkeeping rates required")
    model_kwargs = _model_config({**deepcopy(development["model"]), "name": config["model"]["repo_id"],
                                 "cost_per_token": deepcopy(rates)}, seed)
    budget = HostBudget(max_model_calls, development["usage_tracker_config"]["max_output_tokens_overall"],
                        model_kwargs["max_completion_tokens"])
    source_root = Path(source_root).resolve(strict=True)
    source = _admit_source(config, source_root)
    root = Path(run_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"existing or unresolved run root cannot be reused: {root}")
    _check_execution_location(root)
    data_root = Path(manifest["data_root"]).resolve(strict=True)
    data = (data_root / "data").resolve(strict=True)
    if not data.is_dir():
        raise ValueError("manifest data_root/data must be a native data directory")
    context_window_tokens = (
        config["model_qualification"]["max_model_len"]
        if config["model"].get("serving_backend") == "vllm"
        and config["model"].get("serving_version") == "0.11.2" else None
    )
    transport = LocalChatTransport(
        model_url, timeout_seconds=development["transport"]["timeout_seconds"],
        context_window_tokens=context_window_tokens,
    )
    native_kwargs = deepcopy(dict(development["appworld_config"]))
    if native_kwargs.get("raise_on_extra_parameters") is not True:
        raise ValueError("native raise_on_extra_parameters must remain enabled")
    if native_kwargs.get("load_ground_truth", False) is not False or native_kwargs.get("ground_truth_mode") is not None:
        raise ValueError("target ground-truth access is prohibited")
    if any(value is not None for key, value in native_kwargs.items() if key.startswith("remote_")):
        raise ValueError("this development runner requires a local native world")
    if "random_seed" in native_kwargs and native_kwargs["random_seed"] != seed:
        raise ValueError("native random_seed conflicts with the attempt seed")
    native_kwargs.update(load_ground_truth=False, random_seed=seed)
    root.mkdir()
    receipt_path = root / "execution_receipt.json"
    calls: list[dict[str, Any]] = []
    failures: list[str] = []
    receipt: dict[str, Any] = {
        "schema": "appworld_development_execution_v1", "scope": "development_only", "formal_eligible": False,
        "split": split, "task_id": task_id, "seed": seed,
        "method": (f"development_full_history_{policy}_verification_budget_{verification_budget}"
                   + (f"_planner_{planner_calls}" if policy == "fixed" else "")),
        "neuronsim_method": False,
        "run_root": str(root), "data_root": str(data_root), "native_data_directory": str(data),
        "experiment_name": "development",
        "appworld_version": manifest["appworld_version"], "status": "running", "terminal": False,
        "native_tgc": None, "scoring_status": "not_run", "target_load_ground_truth": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "model_calls": calls, "agent_failures": failures,
        "source": source, "model_identity": deepcopy(config["model"]), "model_kwargs": model_kwargs,
        "native_kwargs": native_kwargs, "host_budget": vars(budget), "verification_budget": verification_budget,
        "allocation_policy": allocation_policy.metadata(verification_budget),
        "state_encoder": "full_visible_history_collection_not_learned",
        "monetary_accounting": deepcopy(development["monetary_accounting"]), "measured_monetary_cost": None,
        "transport_records": transport.records, "model_url": transport.base_url,
        "transport_context_window_tokens": context_window_tokens,
        "command": list(command) if command is not None else None, "host": None, "stage": "native_admission",
    }
    _write_receipt(receipt_path, receipt)
    saved_environment = {name: os.environ.get(name) for name in ("APPWORLD_ROOT", "APPWORLD_CACHE", "OPENAI_API_KEY")}
    native = host = None
    failure: BaseException | None = None
    started = time.clock_gettime(time.CLOCK_MONOTONIC)
    try:
        (root / "data").symlink_to(data, target_is_directory=True)
        os.environ.update(APPWORLD_ROOT=str(root), APPWORLD_CACHE=str(root / ".cache"),
                          OPENAI_API_KEY="local-appworld-placeholder")
        native = _load_native()
        native.path_store.update_root(str(root))
        native.path_store.reload()
        _verify_memberships(native, manifest, data)
        receipt["membership_verified"] = True
        if native.AppWorld.id_to_time_freezer:
            raise ValueError("another native world is active; use a fresh process for this attempt")
        model_class = _language_model_class(source_root, source)
        models = {actor: model_class(**deepcopy(model_kwargs)) for actor in ("planner", "executor")}
        for actor, model in models.items():
            _observe_model(model, actor, transport, calls, failures)
        receipt["stage"] = "native_world_initialization"
        _write_receipt(receipt_path, receipt)
        with native.AppWorld(task_id=task_id, experiment_name="development", **native_kwargs) as world:
            api_methods = source_api_methods(world.task.api_docs)
            history = CollaborationHistory(world.task.instruction)
            history.record_public_context(_public_context(world.task))
            session = PersistentToolSession(world, history, api_methods, verification_budget=verification_budget)

            def progress(report):
                receipt.update(stage="native_collaboration", host=report)
                _write_receipt(receipt_path, receipt)

            host = CollaborationHost(
                session, _model_generator(models, calls), budget=budget,
                state_encoder=full_visible_history, allocation_policy=allocation_policy,
                source_finished=world.task_completed, on_progress=progress,
            )
            result = host.run()
            receipt.update(host=result, status=result["status"], native_host_returned=True)
    except BaseException as error:
        failure = error
        receipt.update(status="unknown" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "infrastructure_failure",
                       error_type=type(error).__name__, error=str(error))
        if host is not None:
            host.status, host.stop_reason = receipt["status"], "execution_exception"
            receipt["host"] = host.report()
    finally:
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
            else:
                os.environ[name] = value
        receipt.update(terminal=True, ended_at=datetime.now(timezone.utc).isoformat(),
                       execution_wall_seconds=time.clock_gettime(time.CLOCK_MONOTONIC) - started,
                       lm_call_count=len(calls), usage_complete=all(row["usage_status"] == "resolved" for row in calls))
        _write_receipt(receipt_path, receipt)
    if isinstance(failure, (KeyboardInterrupt, SystemExit)):
        raise failure
    if failure is not None:
        raise DevelopmentExecutionError(f"native collaboration failed: {failure}", receipt_path) from failure
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", choices=EXPERIMENT_SEEDS, type=int, required=True)
    parser.add_argument("--policy", choices=("uniform", "fixed"), required=True)
    parser.add_argument("--verification-budget", type=int, required=True)
    parser.add_argument("--planner-calls", type=int)
    parser.add_argument("--max-model-calls", type=int, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        config = load_benchmark_config(args.config)
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        receipt = run_collaboration_development_attempt(
            config, manifest, split=args.split, task_id=args.task_id, seed=args.seed, policy=args.policy,
            verification_budget=args.verification_budget, planner_calls=args.planner_calls,
            max_model_calls=args.max_model_calls, source_root=args.source_root, run_root=args.run_root,
            model_url=args.model_url, command=[sys.executable, "-m", __spec__.name, *arguments],
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": receipt["status"], "execution_receipt": str(args.run_root / "execution_receipt.json")}))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

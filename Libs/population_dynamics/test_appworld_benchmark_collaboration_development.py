"""Native-boundary fixtures for the collection CLI; no real AppWorld evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from population_dynamics import appworld_benchmark_collaboration_development as runner
from population_dynamics import appworld_benchmark_scoring as scoring
from population_dynamics.appworld_benchmark_runner import DevelopmentExecutionError
from population_dynamics.appworld_benchmark_transport import LocalChatTransport
from population_dynamics.appworld_benchmark_scoring import _validate_inputs as validate_scorer_inputs
from population_dynamics.test_appworld_benchmark_collaboration import PersistentFixtureWorld
from population_dynamics.test_appworld_benchmark_host import _expand_projected_history, message, proposal, tool
from population_dynamics.test_appworld_benchmark_transport import context_exhausted, context_overflow, server


@pytest.fixture
def language_model_source(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    archived = source_root / runner._MODEL_SOURCE
    archived.parent.mkdir(parents=True)
    payload = b"class LanguageModel:\n    pass\n"
    archived.write_bytes(payload)
    installed = tmp_path / "installed" / "language_model.py"
    installed.parent.mkdir()
    installed.write_bytes(payload)
    module = ModuleType("_appworld_language_model_source_fixture")
    module.__file__ = str(installed)
    exec(compile(payload, str(installed), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original_import = runner.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == runner._MODEL_MODULE:
            return module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(runner.importlib, "import_module", import_module)
    monkeypatch.setattr(runner, "_git_output", lambda *args: pytest.fail("archive path invoked Git"))
    manifest_path = tmp_path / "source_archive_manifest.json"
    manifest = {
        "schema": "appworld_source_archive_manifest_v1",
        "repository": "StonyBrookNLP/appworld",
        "revision": "a" * 40,
        "files": {runner._MODEL_SOURCE: {
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        }},
    }
    config = {"source": {
        "repository": manifest["repository"], "inspected_revision": manifest["revision"],
    }}

    def bind():
        manifest_path.write_text(json.dumps(manifest))
        config["source"]["archive_manifest"] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        return runner._admit_source(config, source_root)

    return SimpleNamespace(
        root=source_root, archived=archived, installed=installed, payload=payload,
        manifest=manifest, manifest_path=manifest_path, bind=bind,
        source=bind(), model_class=module.LanguageModel,
    )


def test_archive_language_model_admission_preserves_pinned_bytes_without_git(language_model_source):
    f = language_model_source
    assert not (f.root / ".git").exists()
    assert runner._language_model_class(f.root, f.source) is f.model_class
    provenance = f.source["language_model"]
    assert provenance["matches_admitted_archive_bytes"] is True
    assert "matches_admitted_git_bytes" not in provenance
    assert f.source["archive_manifest"]["path"] == str(f.manifest_path)
    assert provenance["file"] == str(f.installed.resolve())
    assert provenance["git_path"] == runner._MODEL_SOURCE


def test_archive_language_model_rejects_modified_installed_source(language_model_source):
    f = language_model_source
    f.installed.write_bytes(f.payload + b"# changed installed source\n")
    with pytest.raises(ValueError, match="differs from the pinned official source"):
        runner._language_model_class(f.root, f.source)
    assert "language_model" not in f.source


def test_archive_language_model_rechecks_source_bytes_after_admission(language_model_source):
    f = language_model_source
    f.archived.write_bytes(f.payload + b"# changed archived source\n")
    with pytest.raises(ValueError, match="differs from bound bytes"):
        runner._language_model_class(f.root, f.source)
    assert "language_model" not in f.source


def test_archive_language_model_rechecks_manifest_binding(language_model_source):
    f = language_model_source
    f.manifest_path.write_text(f.manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest byte hash differs"):
        runner._language_model_class(f.root, f.source)
    assert "language_model" not in f.source


def test_archive_language_model_requires_its_own_manifest_entry(language_model_source):
    f = language_model_source
    other = f.root / "experiments/README.md"
    other.write_bytes(b"fixture")
    f.manifest["files"] = {"experiments/README.md": {
        "bytes": 7, "sha256": hashlib.sha256(b"fixture").hexdigest(),
    }}
    source = f.bind()
    with pytest.raises(ValueError, match="not listed in the bound source archive"):
        runner._language_model_class(f.root, source)
    assert "language_model" not in source


@pytest.mark.parametrize("modified", [False, True])
def test_git_language_model_admission_retains_git_show_and_installed_byte_check(
    language_model_source, monkeypatch, modified,
):
    f = language_model_source
    source = {
        "repository": f.source["repository"], "revision": f.source["revision"],
        "root": str(f.root), "prompts": [],
    }
    calls = []

    def git_output(*arguments):
        calls.append(arguments)
        return f.payload

    monkeypatch.setattr(runner, "_git_output", git_output)
    if modified:
        f.installed.write_bytes(f.payload + b"# changed installed source\n")
        with pytest.raises(ValueError, match="differs from the pinned official source"):
            runner._language_model_class(f.root, source)
        assert "language_model" not in source
    else:
        assert runner._language_model_class(f.root, source) is f.model_class
        assert source["language_model"] == {
            "module": runner._MODEL_MODULE, "file": str(f.installed.resolve()),
            "git_path": runner._MODEL_SOURCE, "matches_admitted_git_bytes": True,
        }
    assert calls == [(f.root, "show", f"{source['revision']}:{runner._MODEL_SOURCE}")]


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    data_root = tmp_path / "native"
    (data_root / "data/datasets").mkdir(parents=True)
    manifest = {"schema": "appworld_split_membership_v1", "appworld_version": "synthetic-fixture", "data_root": str(data_root), "splits": {}}
    for split, prefix in (("train", "traincase"), ("dev", "devcase"), ("test_normal", "normalcase"), ("test_challenge", "challengecase")):
        task_ids = [f"{prefix}_{variant}" for variant in (1, 2, 3)]
        payload = "\n".join(task_ids).encode()
        path = data_root / "data/datasets" / f"{split}.txt"
        path.write_bytes(payload)
        manifest["splits"][split] = {"task_ids": task_ids, "task_count": 3, "scenario_count": 1,
                                     "membership_file": str(path), "membership_sha256": hashlib.sha256(payload).hexdigest()}
    rates = {name: 0.0 for name in ("input_cache_miss", "input_cache_hit", "input_cache_write", "output")}
    config = {
        "schema": "appworld_benchmark_v1", "seeds": [41, 42, 43], "formal": {"released": False},
        "splits": {"development": ["train", "dev"]}, "model": {"repo_id": "fixture-model", "revision": "fixture-revision"},
        "development_execution": {
            "status": "provisional", "model": {"client_name": "openai", "api_type": "chat_completions", "max_completion_tokens": 10},
            "monetary_accounting": {"status": "unpriced_local_inference", "native_bookkeeping_cost_per_token": rates},
            "usage_tracker_config": {"max_output_tokens_overall": 100},
            "transport": {"timeout_seconds": 30}, "appworld_config": {"raise_on_extra_parameters": True},
        },
    }
    actions = [proposal(), tool("pcheck", "balance"), message(refs=("planner/pcheck",)),
               tool("echeck", "balance"), message(refs=("executor/echeck",)), tool("write", "add", amount=7)]
    worlds, models, request_payloads, loader_calls = [], [], [], []

    def api_record(app, api, method):
        return {"app_name": app, "api_name": api, "method": method, "path": "/fixture",
                "description": "fixture public API", "parameters": [], "response_schemas": {}}
    docs = {app: {api: api_record(app, api, method) for api, method in entries}
            for app, entries in {
                "bank": [("balance", "GET"), ("add", "POST"), ("fail", "GET")],
                "api_docs": [("show_app_descriptions", "GET")],
                "supervisor": [("complete_task", "POST")],
            }.items()}

    class PublicTask:
        instruction = "Synthetic fixture: add seven."
        app_descriptions = {"bank": "Fixture bank"}
        api_docs = docs
        supervisor = {"first_name": "Synthetic", "last_name": "User", "email": "fixture@example.invalid", "phone_number": None}
        @property
        def ground_truth(self):
            raise AssertionError("actor process read ground truth")

    class World(PersistentFixtureWorld):
        id_to_time_freezer = {}
        def __init__(self, task_id, experiment_name, **kwargs):
            assert kwargs["load_ground_truth"] is False
            assert kwargs["random_seed"] == 41
            assert task_id == "traincase_1" and experiment_name == "development"
            assert Path(os.environ["APPWORLD_ROOT"], "data").resolve() == data_root / "data"
            super().__init__()
            self.task, self.kwargs, self.closed = PublicTask(), kwargs, False
            worlds.append(self)
        def __enter__(self):
            self.id_to_time_freezer["own"] = object()
            return self
        def __exit__(self, *args):
            self.id_to_time_freezer.pop("own")
            self.closed = True
        def task_completed(self):
            return self.balance == 7
        def evaluate(self, *args, **kwargs):
            raise AssertionError("execution process called native scoring")

    class PathStore:
        def update_root(self, root):
            self.root = root
            os.environ["APPWORLD_ROOT"] = root
        def reload(self):
            self.root = os.environ.get("APPWORLD_ROOT", str(tmp_path / "original"))
    def load_ids(**kwargs):
        assert set(kwargs) == {"dataset_name"}
        loader_calls.append(kwargs)
        return list(manifest["splits"][kwargs["dataset_name"]]["task_ids"])
    native = SimpleNamespace(AppWorld=World, version=manifest["appworld_version"], path_store=PathStore(), load_task_ids=load_ids)

    class Model:
        def __init__(self, **kwargs):
            self.kwargs, self.model = kwargs, kwargs["name"]
            assert kwargs["seed"] == 41 and kwargs["use_cache"] is False and kwargs["max_retries"] == 1
            assert os.environ["OPENAI_API_KEY"] == "local-appworld-placeholder"
            models.append(self)
        def generate(self, messages, **kwargs):
            reply = self.lm_call(model=self.model, messages=messages, seed=self.kwargs["seed"], **kwargs)
            usage = reply["usage"]
            return {**reply["choices"][0]["message"], "standardized_usage": SimpleNamespace(dict=lambda: {
                "tokens": {"input_cache_miss": usage["prompt_tokens"], "input_cache_hit": 0, "output": usage["completion_tokens"]},
            })}

    class Transport:
        def __init__(self, base_url, **kwargs):
            self.base_url, self.records = base_url, []
        def __call__(self, **kwargs):
            request_payloads.append(deepcopy(kwargs))
            assert "api_key" not in kwargs
            assert kwargs["max_tokens"] is None and kwargs["max_completion_tokens"] <= 10
            action = actions.pop(0)
            usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            self.records.append({"usage": usage, "status": "completed"})
            return {"choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}], "usage": usage}

    monkeypatch.setattr(runner, "_check_execution_location", lambda root: None)
    monkeypatch.setattr(runner, "_admit_source", lambda *args: {"revision": "fixture", "root": str(source)})
    monkeypatch.setattr(runner, "_language_model_class", lambda *args: Model)
    monkeypatch.setattr(runner, "_load_native", lambda: native)
    monkeypatch.setattr(runner, "LocalChatTransport", Transport)
    monkeypatch.setenv("APPWORLD_ROOT", str(tmp_path / "original"))
    monkeypatch.setenv("APPWORLD_CACHE", str(tmp_path / "original-cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-original-placeholder")
    return SimpleNamespace(config=config, manifest=manifest, root=tmp_path / "attempt", source=source,
                           native=native, worlds=worlds, models=models, payloads=request_payloads,
                           loader_calls=loader_calls, actions=actions, tmp=tmp_path)


def execute(fixture, **overrides):
    arguments = {"split": "train", "task_id": "traincase_1", "seed": 41, "policy": "fixed", "planner_calls": 1,
                 "verification_budget": 2, "max_model_calls": 20, "source_root": fixture.source,
                 "run_root": fixture.root, "model_url": "http://127.0.0.1:1/v1"}
    arguments.update(overrides)
    return runner.run_collaboration_development_attempt(fixture.config, fixture.manifest, **arguments)


def test_native_boundary_creates_one_world_two_models_and_a_scorer_compatible_receipt(native_fixture):
    f = native_fixture
    receipt = execute(f)
    assert receipt["status"] == "completed" and receipt["terminal"] is True
    assert receipt["native_tgc"] is None and receipt["scoring_status"] == "not_run"
    assert receipt["neuronsim_method"] is False and receipt["formal_eligible"] is False
    admitted_data = str((Path(f.manifest["data_root"]) / "data").resolve())
    assert receipt["native_data_directory"] == admitted_data
    assert os.readlink(f.root / "data") == admitted_data
    assert len(f.worlds) == 1 and f.worlds[0].closed and f.worlds[0].balance == 7
    assert len(f.models) == 2 and f.models[0] is not f.models[1]
    assert [call["consumer"] for call in receipt["model_calls"]] == ["planner", "planner", "planner", "executor", "executor", "executor"]
    assert all(call["usage_status"] == "resolved" for call in receipt["model_calls"])
    assert receipt["lm_call_count"] == 6 and receipt["usage_complete"] is True
    validate_scorer_inputs(receipt, f.manifest)
    assert os.environ["OPENAI_API_KEY"] == "fixture-original-placeholder"
    assert os.environ["APPWORLD_ROOT"] == str(f.tmp / "original")
    first = json.loads(f.payloads[0]["messages"][1]["content"])["events"]
    context = next(event["payload"] for event in first if event["kind"] == "public_context")
    assert set(context["bootstrap_api_docs"]) == {"api_docs", "supervisor"}
    assert set(context["main_user"]) == {"first_name", "last_name", "email", "phone_number"}
    assert len(f.loader_calls) == 4
    saved = json.loads((f.root / "execution_receipt.json").read_text())
    assert saved == receipt


@pytest.mark.parametrize("split,task_id,policy", [
    ("test_normal", "normalcase_1", "fixed"), ("test_challenge", "challengecase_1", "fixed"),
    ("dev", "devcase_1", "uniform"), ("train", "challengecase_1", "fixed"),
])
def test_closed_memberships_and_collection_scope_reject_before_native_access(native_fixture, split, task_id, policy):
    f = native_fixture
    f.config["formal"]["released"] = True
    with pytest.raises(ValueError):
        execute(f, split=split, task_id=task_id, policy=policy, planner_calls=None if policy == "uniform" else 1)
    assert f.worlds == [] and f.models == [] and not f.root.exists()


@pytest.mark.parametrize("policy", ["fixed", "uniform"])
def test_initial_and_failed_receipts_bind_the_actual_native_input_link(native_fixture, monkeypatch, policy):
    f = native_fixture
    admitted_data = str((Path(f.manifest["data_root"]) / "data").resolve())
    initial = []

    def fail_after_data_link_creation():
        record = json.loads((f.root / "execution_receipt.json").read_text())
        assert record["native_data_directory"] == admitted_data
        assert record["run_root"] == str(f.root)
        assert record["data_root"] == f.manifest["data_root"]
        assert record["terminal"] is False
        assert os.readlink(f.root / "data") == admitted_data
        initial.append(record)
        raise RuntimeError("synthetic native initialization failure")

    monkeypatch.setattr(runner, "_load_native", fail_after_data_link_creation)
    with pytest.raises(DevelopmentExecutionError, match="synthetic native initialization failure"):
        execute(f, policy=policy, planner_calls=None if policy == "uniform" else 1)
    receipt = json.loads((f.root / "execution_receipt.json").read_text())
    assert len(initial) == 1
    assert receipt["native_data_directory"] == admitted_data
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"] is True
    assert receipt["native_tgc"] is None and receipt["scoring_status"] == "not_run"


def test_existing_attempt_is_not_reused(native_fixture):
    f = native_fixture
    f.root.mkdir()
    sentinel = f.root / "retained"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError):
        execute(f)
    assert sentinel.read_text() == "existing data" and f.worlds == []


def test_another_active_world_is_not_closed_or_replaced(native_fixture):
    f = native_fixture
    f.native.AppWorld.id_to_time_freezer["other"] = object()
    with pytest.raises(DevelopmentExecutionError, match="another native world"):
        execute(f)
    assert f.worlds == [] and f.models == [] and "other" in f.native.AppWorld.id_to_time_freezer
    receipt = json.loads((f.root / "execution_receipt.json").read_text())
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"] is True
    assert os.environ["OPENAI_API_KEY"] == "fixture-original-placeholder"


def test_actual_cli_records_command_and_produces_the_separate_scorer_input(native_fixture, capsys):
    f = native_fixture
    config_path, manifest_path = f.tmp / "config.yaml", f.tmp / "splits.json"
    config_path.write_text(yaml.safe_dump(f.config))
    manifest_path.write_text(json.dumps(f.manifest))
    arguments = ["--config", str(config_path), "--split-manifest", str(manifest_path),
                 "--source-root", str(f.source), "--run-root", str(f.root),
                 "--model-url", "http://127.0.0.1:1/v1", "--split", "train", "--task-id", "traincase_1",
                 "--seed", "41", "--policy", "fixed", "--planner-calls", "1", "--verification-budget", "2",
                 "--max-model-calls", "20"]
    assert runner.main(arguments) == 0
    receipt = json.loads((f.root / "execution_receipt.json").read_text())
    assert receipt["command"][3:] == arguments
    validate_scorer_inputs(receipt, f.manifest)
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_model_budget_failure_stays_scorable_with_known_cost_and_no_oracle(native_fixture):
    f = native_fixture
    f.actions[:] = [{"kind": "finish"}]
    receipt = execute(f, max_model_calls=1)
    assert receipt["status"] == "agent_failure" and receipt["lm_call_count"] == 1
    assert receipt["host"]["resolved_completion_tokens"] == 3 and receipt["native_tgc"] is None
    assert f.worlds[0].calls == [] and f.worlds[0].closed
    validate_scorer_inputs(receipt, f.manifest)


@pytest.mark.parametrize("quota,verifier", [(0, "executor"), (2, "planner")])
def test_fixed_control_identity_retains_total_budget_and_allocation_in_scorer_input(native_fixture, quota, verifier):
    f = native_fixture
    f.actions[:] = [proposal(), tool("check1", "balance"), tool("check2", "balance"),
                   message(refs=(f"{verifier}/check2",)), tool("write", "add", amount=7)]
    receipt = execute(f, planner_calls=quota)
    assert receipt["status"] == "completed"
    assert receipt["method"] == f"development_full_history_fixed_verification_budget_2_planner_{quota}"
    validate_scorer_inputs(receipt, f.manifest)


def enable_pinned_context_correction(fixture):
    fixture.config["model"].update(serving_backend="vllm", serving_version="0.11.2")
    fixture.config["model_qualification"] = {"max_model_len": 32768}


def test_context_correction_after_allocation_keeps_one_logical_call_and_scorable_receipt(
    native_fixture, monkeypatch,
):
    f = native_fixture
    enable_pinned_context_correction(f)
    responses = []
    for index, action in enumerate(f.actions):
        if index == 1:
            responses.append((400, context_overflow(requested=10, prompt=32763)))
        responses.append((200, {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}],
            "usage": {"prompt_tokens": 32763, "completion_tokens": 3, "total_tokens": 32766},
        }))
    monkeypatch.setattr(runner, "LocalChatTransport", LocalChatTransport)
    before = deepcopy(f.config)
    with server(responses=responses) as (url, seen):
        receipt = execute(f, model_url=url)
    assert f.config == before
    assert receipt["status"] == "completed" and receipt["transport_context_window_tokens"] == 32768
    assert receipt["lm_call_count"] == receipt["host"]["model_call_count"] == 6
    assert receipt["usage_complete"] is True
    assert receipt["host"]["resolved_completion_tokens"] == 18
    assert len(seen) == len(receipt["transport_records"]) == 7
    original, adjusted = seen[1][1], seen[2][1]
    assert original["max_completion_tokens"] == 10 and adjusted["max_completion_tokens"] == 5
    assert adjusted == {**original, "max_completion_tokens": 5}
    assert adjusted["max_tokens"] is None and adjusted["seed"] == 41
    assert len(receipt["host"]["state_exports"]) == 2
    assert receipt["host"]["allocation"] == {"planner": 1, "executor": 1}
    assert receipt["host"]["verification_used"] == {"planner": 1, "executor": 1}
    assert f.worlds[0].closed and f.worlds[0].balance == 7
    validate_scorer_inputs(receipt, f.manifest)
    assert json.loads((f.root / "execution_receipt.json").read_text()) == receipt


@pytest.mark.parametrize("prompt", [32768, 32769])
def test_nonpositive_overflow_signature_stays_infrastructure_failure_without_synthetic_usage(
    native_fixture, monkeypatch, prompt,
):
    f = native_fixture
    enable_pinned_context_correction(f)
    monkeypatch.setattr(runner, "LocalChatTransport", LocalChatTransport)
    with server(status=400, body=context_overflow(requested=10, prompt=prompt)) as (url, seen):
        with pytest.raises(DevelopmentExecutionError):
            execute(f, model_url=url)
    receipt = json.loads((f.root / "execution_receipt.json").read_text())
    assert receipt["status"] == "infrastructure_failure" and receipt["terminal"] is True
    assert receipt["usage_complete"] is False and receipt["lm_call_count"] == 1
    assert receipt["model_calls"][0]["raw_usage"] is None
    assert receipt["model_calls"][0]["usage_status"] == "unresolved"
    assert len(seen) == len(receipt["transport_records"]) == 1
    assert receipt["transport_records"][0]["usage"] is None
    assert "generation_started" not in receipt["transport_records"][0]
    assert f.worlds[0].closed and f.worlds[0].calls == []


def test_verified_no_room_after_thirteen_calls_preserves_state_and_can_be_scored(
    native_fixture, monkeypatch,
):
    f = native_fixture
    enable_pinned_context_correction(f)
    f.actions[:] = [
        proposal(), tool("pcheck", "balance"), message(refs=("planner/pcheck",)),
        tool("echeck", "balance"), message(refs=("executor/echeck",)),
        tool("write", "add", amount=3),
        *[message() for _ in range(7)],
    ]
    responses = [(200, {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(action)}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }) for action in f.actions]
    responses.append((400, context_exhausted()))
    monkeypatch.setattr(runner, "LocalChatTransport", LocalChatTransport)
    with server(responses=responses) as (url, seen):
        receipt = execute(f, model_url=url)
    host = receipt["host"]
    assert receipt["status"] == host["status"] == "agent_failure"
    assert host["stop_reason"] == "context_budget_exhausted"
    assert receipt["native_host_returned"] is True and receipt["terminal"] is True
    assert receipt["lm_call_count"] == host["model_call_count"] == len(seen) == 14
    assert len(receipt["transport_records"]) == 14
    assert receipt["usage_complete"] is False and host["resolved_completion_tokens"] == 39
    assert len(host["state_exports"]) == 2
    assert host["verification_used"] == {"planner": 1, "executor": 1}
    assert f.worlds[0].closed and f.worlds[0].balance == 3
    assert [row["status"] for row in host["generations"][:-1]] == ["returned"] * 13
    assert all(row["usage_status"] == "resolved" for row in receipt["model_calls"][:-1])
    final = receipt["model_calls"][-1]
    generation = host["generations"][-1]
    assert final["transport_status"] == generation["status"] == "pre_generation_rejected"
    assert final["raw_response"] is final["raw_usage"] is final["official_usage"] is None
    assert final["usage_status"] == "unresolved"
    assert final["generation_started"] is generation["generation_started"] is False
    assert final["generated_completion_tokens"] == generation["generated_completion_tokens"] == 0
    assert final["transport_request_id"] == generation["transport_request_id"] == receipt["transport_records"][-1]["request_id"]
    assert "message" not in generation and "action" not in generation
    snapshot = json.loads(seen[-1][1]["messages"][1]["content"])
    assert host["actor_histories"][host["next_actor"]] == _expand_projected_history(snapshot)
    validate_scorer_inputs(receipt, f.manifest)

    # Exercise unchanged scoring admission with a synthetic independent evaluator.
    # A valid negative score does not turn missing native usage into measured cost.
    evaluation_calls = []

    def evaluate(**kwargs):
        assert f.worlds[0].closed
        evaluation_calls.append(kwargs)
        directory = f.root / "experiments/outputs/development/tasks/traincase_1/evaluation"
        directory.mkdir(parents=True)
        (directory / "report.md").write_text("Synthetic negative evaluation fixture.")
        (directory / "version.txt").write_text(f.manifest["appworld_version"])
        return {"individual": {"traincase_1": {
            "success": False, "num_tests": 1, "passes": [], "failures": ["synthetic unmet target"],
        }}}

    monkeypatch.setattr(scoring, "_check_execution_location", lambda root: None)
    monkeypatch.setattr(scoring, "_load_native", lambda: f.native)
    monkeypatch.setattr(scoring, "_load_evaluator", lambda: evaluate)
    before = deepcopy(receipt)
    outcome = scoring.score_development_attempt(
        receipt, f.manifest, f.root / "development_score.json",
        execution_receipt_path=f.root / "execution_receipt.json",
    )
    assert receipt == before and json.loads((f.root / "execution_receipt.json").read_text()) == before
    assert len(evaluation_calls) == 1 and outcome["evaluator_called"] is True
    assert outcome["scoring_status"] == "completed" and outcome["native_tgc"] is False
    assert outcome["execution_status"] == "agent_failure" and outcome["usage_complete"] is False

"""Mocked development CLI tests; synthetic Git/transport records are not evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from population_dynamics import appworld_benchmark_development as development
from population_dynamics.appworld_benchmark_runner import DevelopmentExecutionError
from population_dynamics.appworld_benchmark_transport import LocalChatTransport


@pytest.fixture
def development_fixture(tmp_path, monkeypatch):
    config = development.load_benchmark_config(development._DEFAULT_CONFIG)
    config = deepcopy(config)
    source_root = tmp_path / "appworld-source"
    source_root.mkdir()
    config["development_execution"]["source_root"] = str(source_root)
    prompt_paths = {config["development_execution"]["api_predictor"]["prompt_file_path"]}
    for method_config in config["development_execution"]["methods"].values():
        for field in development._PROMPT_FIELDS:
            if method_config.get(field) is not None:
                prompt_paths.add(method_config[field])
    tracked = {}
    for relative in prompt_paths:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"Synthetic source prompt fixture: {relative}\n".encode()
        path.write_bytes(payload)
        tracked[relative] = payload
    manifest = {
        "schema": "appworld_split_membership_v1", "appworld_version": "fixture-version",
        "data_root": str(tmp_path / "native-data-root"), "splits": {},
    }
    for split, prefixes in (("train", ("82e2fac", "29caf6f", "d0b1f43")), ("dev", ("devcase",)), ("test_challenge", ("challengecase",))):
        ids = [f"{prefix}_{variant}" for prefix in prefixes for variant in (1, 2, 3)]
        manifest["splits"][split] = {
            "task_ids": ids, "task_count": len(ids), "scenario_count": len(prefixes),
            "membership_sha256": "a" * 64,
            "membership_file": str(tmp_path / "native-data-root" / "data" / "datasets" / f"{split}.txt"),
        }
    config_path, manifest_path = tmp_path / "config.yaml", tmp_path / "manifest.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest_path.write_text(json.dumps(manifest))
    state = SimpleNamespace(
        config=config, manifest=manifest, config_path=config_path, manifest_path=manifest_path,
        source_root=source_root, root=tmp_path / "attempt", tracked=tracked,
        revision=config["source"]["inspected_revision"], origin="https://github.com/StonyBrookNLP/appworld.git",
        git_calls=[], runs=[], execution_failure=False,
        transport_record={"request_id": "synthetic-request", "attempts": 1, "status": "completed", "completion_known": True, "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )

    def git_output(root, *arguments):
        assert root == source_root
        state.git_calls.append(arguments)
        if arguments == ("rev-parse", "--show-toplevel"):
            return (str(source_root) + "\n").encode()
        if arguments == ("rev-parse", "HEAD"):
            return (state.revision + "\n").encode()
        if arguments == ("remote", "get-url", "origin"):
            return (state.origin + "\n").encode()
        if arguments[:3] == ("ls-files", "--error-unmatch", "--"):
            relative = arguments[-1]
            if relative not in state.tracked:
                raise subprocess.CalledProcessError(1, ["git", *arguments])
            return (relative + "\n").encode()
        if arguments[0] == "show":
            _, relative = arguments[1].split(":", 1)
            return state.tracked[relative]
        raise AssertionError(f"unexpected broad Git operation: {arguments}")

    def run_boundary(config_argument, manifest_argument, **kwargs):
        state.runs.append(kwargs)
        assert config_argument["schema"] == "appworld_benchmark_v1"
        assert manifest_argument == manifest
        root = Path(kwargs["run_root"])
        root.mkdir()
        receipt = {
            "status": "infrastructure_failure" if state.execution_failure else "completed",
            "run_root": str(root), "native_tgc": None, "scoring_status": "not_run",
            "resolved_agent_config": deepcopy(kwargs["agent_config"]),
            "invocation_metadata": deepcopy(kwargs["invocation_metadata"]),
        }
        receipt_path = root / "execution_receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        kwargs["lm_call"].records.append(deepcopy(state.transport_record))
        if state.execution_failure:
            raise DevelopmentExecutionError("synthetic native execution failure", receipt_path)
        return receipt

    monkeypatch.setattr(development, "_git_output", git_output)
    monkeypatch.setattr(development, "run_development_attempt", run_boundary)
    return state


def cli_arguments(fixture, *, method="official_react_code", split="dev", task_id="devcase_1", seed=41):
    return [
        "--config", str(fixture.config_path), "--split-manifest", str(fixture.manifest_path),
        "--source-root", str(fixture.source_root), "--run-root", str(fixture.root),
        "--model-url", "http://127.0.0.1:8000/v1", "--split", split, "--task-id", task_id,
        "--method", method, "--seed", str(seed),
    ]


def rewrite_config(fixture):
    fixture.config_path.write_text(yaml.safe_dump(fixture.config, sort_keys=False))


def qualification_fixture(config):
    return {
        'schema': 'appworld_model_runtime_qualification_v1', 'scope': 'neutral_model_transport_only',
        'status': 'PASS', 'tool_calls_validated': True, 'transport_probe_status': 'PASS',
        'seeds': [41, 42, 43], 'model_identity': deepcopy(config['model']),
        'config': {'model_qualification': deepcopy(config['model_qualification'])},
        'runtime': {'packages': {'vllm': config['model']['serving_version'], 'torch': 'synthetic-torch'}},
        'server_cleanup': {'server_parent_reaped': True, 'forced_kill': False,
                           'process_group_term_sent': True, 'returncode_before_termination': None},
    }


def test_function_calling_cli_consumes_qualified_transport_and_retains_reference(development_fixture):
    fixture = development_fixture
    path = fixture.root.parent/'qualification.json'
    path.write_text(json.dumps(qualification_fixture(fixture.config)))
    arguments = cli_arguments(fixture, method='official_function_calling', seed=43)
    assert development.main(arguments + ['--model-qualification', str(path)]) == 0
    call = fixture.runs[0]
    assert call['method'] == 'official_function_calling'
    assert call['transport_capabilities'].tool_calls_validated is True
    assert call['transport_capabilities'].validation_reference == str(path)
    assert call['agent_config']['model_config']['seed'] == 43
    assert call['agent_config']['api_predictor_config']['model_config']['seed'] == 43
    invocation = json.loads((fixture.root/'development_invocation.json').read_text())
    assert invocation['tool_transport_validation_reference'] == str(path)
    assert invocation['formal_eligible'] is False


@pytest.mark.parametrize('defect', ['failed','model_revision','serving_settings','cleanup','missing_packages','seeds'])
def test_function_calling_invalid_qualification_rejects_before_source_or_native_access(development_fixture, defect):
    fixture = development_fixture
    receipt = qualification_fixture(fixture.config)
    if defect == 'failed': receipt['status'] = 'FAIL'
    elif defect == 'model_revision': receipt['model_identity']['revision'] = 'different'
    elif defect == 'serving_settings': receipt['config']['model_qualification']['max_model_len'] += 1
    elif defect == 'cleanup': receipt['server_cleanup']['forced_kill'] = True
    elif defect == 'missing_packages': receipt['runtime'] = {}
    else: receipt['seeds'] = [41,42,44]
    path = fixture.root.parent/'qualification.json'
    path.write_text(json.dumps(receipt))
    assert development.main(cli_arguments(fixture, method='official_function_calling') + ['--model-qualification', str(path)]) != 0
    assert not fixture.runs and not fixture.git_calls and not fixture.root.exists()


def test_factory_uses_canonical_react_defaults_and_source_prompt(development_fixture):
    fixture = development_fixture
    prepared = development.prepare_official_agent(fixture.config, "official_react_code", fixture.source_root)
    kwargs = prepared.agent_config
    assert kwargs["max_steps"] == 50 and kwargs["ignore_multiple_calls"] is True
    assert kwargs["max_prompt_length"] is None and kwargs["max_output_length"] is None
    assert kwargs["prompt_file_path"] == str(fixture.source_root / "experiments/prompts/react_code_agent/instructions.txt")
    assert kwargs["appworld_config"]["raise_on_extra_parameters"] is True
    assert kwargs["model_config"]["name"] == fixture.config["model"]["repo_id"]
    assert kwargs["model_config"]["max_completion_tokens"] == 8192
    assert kwargs["model_config"]["temperature"] == 0.7 and kwargs["model_config"]["top_p"] == 0.8
    assert kwargs["usage_tracker_config"]["max_output_tokens_per_task"] == 100000
    assert kwargs["usage_tracker_config"]["max_output_tokens_overall"] == 100000
    assert len(prepared.source_provenance["prompts"]) == 1
    assert prepared.source_provenance["prompts"][0]["matches_admitted_git_bytes"]


def test_fullcode_preserves_source_algorithms_and_same_model_consumers(development_fixture):
    fixture = development_fixture
    kwargs = development.prepare_official_agent(fixture.config, "official_full_code", fixture.source_root).agent_config
    assert kwargs["max_steps"] == 20 and kwargs["max_num_retrials"] == 5
    assert kwargs["compress_api_docs"] and kwargs["remove_code_demo_comments"]
    assert kwargs["demo_task_ids"] == ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
    predictor = kwargs["api_predictor_config"]
    assert predictor["demo_task_ids"] == kwargs["demo_task_ids"]
    assert predictor["model_config"] == kwargs["model_config"]
    assert predictor["mode"] == "predicted" and predictor["max_predicted_apis"] == 20
    assert kwargs["usage_tracker_config"]["max_cost_overall"] == 1000
    assert kwargs["usage_tracker_config"]["max_cost_per_task"] == 10


def test_function_calling_factory_supports_source_zero_shot(development_fixture):
    fixture = development_fixture
    kwargs = development.prepare_official_agent(fixture.config, "official_function_calling", fixture.source_root).agent_config
    assert kwargs["prompt_file_path"].endswith("function_calling_agent/instructions.txt")
    assert kwargs["demo_messages_file_path"] is None and kwargs["max_steps"] == 50
    assert kwargs["model_config"]["tool_choice"] == "auto"
    assert "tool_choice" not in kwargs["api_predictor_config"]["model_config"]
    assert kwargs["appworld_config"]["include_direct_functions"] is True
    assert kwargs["appworld_config"]["direct_function_separator"] == "__"


def test_real_cli_routes_arguments_and_retains_effective_settings(development_fixture):
    fixture = development_fixture
    arguments = cli_arguments(fixture, method="official_full_code", seed=43) + ["--timeout-seconds", "123"]
    assert development.main(arguments) == 0
    invocation = json.loads((fixture.root / "development_invocation.json").read_text())
    call = fixture.runs[0]
    assert call["split"] == "dev" and call["task_id"] == "devcase_1"
    assert call["method"] == "official_full_code" and call["seed"] == 43
    assert isinstance(call["lm_call"], LocalChatTransport)
    assert call["lm_call"].base_url == "http://127.0.0.1:8000/v1"
    assert call["lm_call"].timeout_seconds == 123
    assert call["lm_call"].context_window_tokens == fixture.config["model_qualification"]["max_model_len"] == 32768
    assert invocation["transport_context_window_tokens"] == 32768
    for model in (call["agent_config"]["model_config"], call["agent_config"]["api_predictor_config"]["model_config"]):
        assert model["seed"] == 43 and model["use_cache"] is False and model["max_retries"] == 1
        assert model["max_completion_tokens"] == 8192
    assert call["agent_config"]["appworld_config"]["load_ground_truth"] is False
    assert invocation["agent_config"] == call["agent_config"]
    assert call["invocation_metadata"]["source"] == invocation["source"]
    assert call["invocation_metadata"]["model_identity"] == invocation["model_identity"]
    assert call["invocation_metadata"]["agent_config"] == call["agent_config"]
    assert "transport_records" not in call["invocation_metadata"]
    native_receipt = json.loads((fixture.root / "execution_receipt.json").read_text())
    assert native_receipt["invocation_metadata"] == call["invocation_metadata"]
    assert native_receipt["resolved_agent_config"] == call["agent_config"]
    assert invocation["transport_records"] == [fixture.transport_record]
    assert invocation["monetary_accounting"]["status"] == "unpriced_local_inference"
    assert invocation["measured_monetary_cost"] is None and not invocation["formal_eligible"]
    assert "one completion" in invocation["budget_adaptations"][0]["limitation"]
    assert invocation["command"][-len(arguments):] == arguments
    assert json.loads((fixture.root / "execution_receipt.json").read_text())["native_tgc"] is None


def test_cli_uses_configured_timeout_by_default(development_fixture):
    assert development.main(cli_arguments(development_fixture)) == 0
    assert development_fixture.runs[0]["lm_call"].timeout_seconds == 600


def test_execution_failure_retains_transport_and_invocation_settings(development_fixture):
    fixture = development_fixture
    fixture.execution_failure = True
    fixture.transport_record.update(status="unresolved", completion_known=False, usage=None)
    assert development.main(cli_arguments(fixture, seed=42)) == 1
    invocation = json.loads((fixture.root / "development_invocation.json").read_text())
    assert invocation["status"] == "execution_failed" and invocation["seed"] == 42
    assert invocation["transport_records"] == [fixture.transport_record]
    assert invocation["agent_config"]["model_config"]["seed"] == 42
    native_receipt = json.loads((fixture.root / "execution_receipt.json").read_text())
    assert native_receipt["status"] == "infrastructure_failure" and native_receipt["native_tgc"] is None


def test_test_split_is_rejected_before_source_or_native_access(development_fixture):
    fixture = development_fixture
    assert development.main(cli_arguments(fixture, split="test_challenge", task_id="challengecase_1")) != 0
    assert not fixture.git_calls and not fixture.runs and not fixture.root.exists()


def test_target_demo_is_rejected_before_native_execution(development_fixture):
    fixture = development_fixture
    assert development.main(cli_arguments(fixture, method="official_full_code", split="train", task_id="82e2fac_1")) != 0
    assert not fixture.runs and not fixture.root.exists()


def test_function_calling_cli_requires_a_passed_qualification_receipt(development_fixture):
    fixture = development_fixture
    assert development.main(cli_arguments(fixture, method="official_function_calling")) != 0
    assert not fixture.git_calls and not fixture.runs


def test_existing_run_is_preserved_before_source_access(development_fixture):
    fixture = development_fixture
    fixture.root.mkdir()
    marker = fixture.root / "execution_receipt.json"
    marker.write_bytes(b"preserved existing execution")
    assert development.main(cli_arguments(fixture)) != 0
    assert marker.read_bytes() == b"preserved existing execution"
    assert not fixture.git_calls and not fixture.runs
    assert not (fixture.root / "development_invocation.json").exists()


def test_incompatible_yaml_schema_is_rejected(development_fixture):
    fixture = development_fixture
    fixture.config_path.write_text("schema: wrong_schema\n")
    assert development.main(cli_arguments(fixture)) != 0
    assert not fixture.git_calls and not fixture.runs


@pytest.mark.parametrize("defect", ["revision", "origin", "modified_prompt", "untracked_prompt", "outside_source"])
def test_source_rejection_precedes_native_execution(development_fixture, defect):
    fixture = development_fixture
    relative = fixture.config["development_execution"]["methods"]["official_react_code"]["prompt_file_path"]
    if defect == "revision":
        fixture.revision = "b" * 40
    elif defect == "origin":
        fixture.origin = "https://github.com/other/project.git"
    elif defect == "modified_prompt":
        (fixture.source_root / relative).write_text("modified prompt")
    elif defect == "untracked_prompt":
        fixture.tracked.pop(relative)
    elif defect == "outside_source":
        fixture.config["development_execution"]["methods"]["official_react_code"]["prompt_file_path"] = "../other_prompt.txt"
        rewrite_config(fixture)
    assert development.main(cli_arguments(fixture)) != 0
    assert not fixture.runs and not fixture.root.exists()


def test_unrelated_prompt_edit_does_not_trigger_a_broad_source_check(development_fixture):
    fixture = development_fixture
    relative = fixture.config["development_execution"]["methods"]["official_full_code"]["code_prompt_file_path"]
    (fixture.source_root / relative).write_text("unrelated method edit")
    prepared = development.prepare_official_agent(fixture.config, "official_react_code", fixture.source_root)
    assert len(prepared.source_provenance["prompts"]) == 1
    assert all(not (call[0] == "show" and call[1].endswith(relative)) for call in fixture.git_calls)


def test_unpriced_bookkeeping_cannot_silently_become_a_dollar_rate(development_fixture):
    fixture = development_fixture
    fixture.config["development_execution"]["monetary_accounting"]["native_bookkeeping_cost_per_token"]["output"] = 0.01
    rewrite_config(fixture)
    assert development.main(cli_arguments(fixture)) != 0
    assert not fixture.runs


def test_model_override_cannot_change_the_shared_backbone(development_fixture):
    fixture = development_fixture
    fixture.config["development_execution"]["methods"]["official_full_code"]["model_overrides"] = {"name": "other-model"}
    with pytest.raises(ValueError, match="model override"):
        development.prepare_official_agent(fixture.config, "official_full_code", fixture.source_root)


def test_credentialed_transport_url_is_rejected(development_fixture, capsys):
    fixture = development_fixture
    arguments = cli_arguments(fixture)
    arguments[arguments.index("--model-url") + 1] = "http://user:private-secret@127.0.0.1:8000/v1"
    assert development.main(arguments) != 0
    assert not fixture.runs and "private-secret" not in capsys.readouterr().err


def bind_archive_fixture(fixture):
    payload = (json.dumps(fixture.archive_document, sort_keys=True) + "\n").encode()
    fixture.archive_path.write_bytes(payload)
    fixture.config["source"]["archive_manifest"] = {
        "path": str(fixture.archive_path), "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.fixture
def archive_fixture(development_fixture):
    fixture = development_fixture
    fixture.archive_extra = "src/appworld/archive_fixture.py"
    extra = fixture.source_root / fixture.archive_extra
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"# synthetic archived source fixture\n")
    contents = {**fixture.tracked, fixture.archive_extra: extra.read_bytes()}
    fixture.archive_document = {
        "schema": "appworld_source_archive_manifest_v1",
        "repository": "StonyBrookNLP/appworld", "revision": fixture.revision,
        "files": {
            relative: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for relative, payload in contents.items()
        },
    }
    fixture.archive_path = fixture.source_root.parent / "source-archive-manifest.json"
    bind_archive_fixture(fixture)
    return fixture


def test_archive_cli_uses_bound_files_without_git_and_keeps_compact_provenance(archive_fixture):
    fixture = archive_fixture
    rewrite_config(fixture)
    assert development.main(cli_arguments(fixture)) == 0
    assert fixture.git_calls == []
    source = fixture.runs[0]["invocation_metadata"]["source"]
    assert source["revision"] == fixture.revision
    assert source["admission_method"] == "source_archive_manifest"
    assert source["archive_manifest"]["file_count"] == len(fixture.archive_document["files"])
    assert source["archive_manifest"]["sha256"] == fixture.config["source"]["archive_manifest"]["sha256"]
    assert "files" not in source and "files" not in source["archive_manifest"]
    assert len(source["prompts"]) == 1
    assert source["prompts"][0]["matches_admitted_archive_bytes"] is True
    assert "matches_admitted_git_bytes" not in source["prompts"][0]


@pytest.mark.parametrize("target", ["prompt", "other", "missing"])
def test_archive_rejects_changed_or_missing_listed_files(archive_fixture, target):
    fixture = archive_fixture
    relative = (fixture.config["development_execution"]["methods"]["official_react_code"]["prompt_file_path"]
                if target == "prompt" else fixture.archive_extra)
    path = fixture.source_root / relative
    if target == "missing":
        path.unlink()
    else:
        path.write_bytes(b"x" * len(path.read_bytes()))
    with pytest.raises(ValueError, match="archive source file"):
        development.prepare_official_agent(fixture.config, "official_react_code", fixture.source_root)
    assert fixture.git_calls == []


@pytest.mark.parametrize("defect", ["hash", "bytes", "revision", "repository", "schema", "empty"])
def test_archive_rejects_invalid_manifest_binding_and_identity(archive_fixture, defect):
    fixture = archive_fixture
    if defect == "hash":
        fixture.config["source"]["archive_manifest"]["sha256"] = "0" * 64
    elif defect == "bytes":
        fixture.archive_path.write_bytes(fixture.archive_path.read_bytes() + b"\n")
    else:
        if defect == "revision":
            fixture.archive_document["revision"] = "0" * 40
        elif defect == "repository":
            fixture.archive_document["repository"] = "example/other"
        elif defect == "schema":
            fixture.archive_document["schema"] = "other"
        else:
            fixture.archive_document["files"] = {}
        bind_archive_fixture(fixture)
    with pytest.raises(ValueError):
        development.prepare_official_agent(fixture.config, "official_react_code", fixture.source_root)
    assert fixture.git_calls == []


def test_archive_prompt_must_be_listed_even_when_present(archive_fixture):
    fixture = archive_fixture
    relative = fixture.config["development_execution"]["methods"]["official_react_code"]["prompt_file_path"]
    del fixture.archive_document["files"][relative]
    bind_archive_fixture(fixture)
    with pytest.raises(ValueError, match="not listed"):
        development.prepare_official_agent(fixture.config, "official_react_code", fixture.source_root)


def test_archive_prompt_is_rechecked_after_source_admission(archive_fixture):
    fixture = archive_fixture
    source = development._admit_source(fixture.config, fixture.source_root)
    relative = fixture.config["development_execution"]["methods"]["official_react_code"]["prompt_file_path"]
    path = fixture.source_root / relative
    path.write_bytes(b"x" * len(path.read_bytes()))
    with pytest.raises(ValueError, match="differs from bound bytes"):
        development._resolve_prompt(fixture.source_root, relative, source)


@pytest.mark.parametrize("relative", ["../escape.py", "/escape.py", "src/../escape.py",
                                      "src//escape.py", "src/./escape.py", r"src\escape.py", ""])
def test_archive_rejects_unsafe_file_paths(archive_fixture, relative):
    fixture = archive_fixture
    fixture.archive_document["files"][relative] = fixture.archive_document["files"][fixture.archive_extra]
    bind_archive_fixture(fixture)
    with pytest.raises(ValueError, match="relative paths"):
        development._admit_source(fixture.config, fixture.source_root)


@pytest.mark.parametrize("kind", ["file", "parent"])
def test_archive_rejects_symlinks_inside_source_root(archive_fixture, kind):
    fixture = archive_fixture
    record = fixture.archive_document["files"][fixture.archive_extra]
    if kind == "file":
        link = fixture.source_root / "linked.py"
        link.symlink_to(fixture.source_root / fixture.archive_extra)
        relative = "linked.py"
    else:
        link = fixture.source_root / "linked"
        link.symlink_to((fixture.source_root / fixture.archive_extra).parent, target_is_directory=True)
        relative = "linked/archive_fixture.py"
    fixture.archive_document["files"][relative] = record
    bind_archive_fixture(fixture)
    with pytest.raises(ValueError, match="symlink"):
        development._admit_source(fixture.config, fixture.source_root)


@pytest.mark.parametrize("record", [
    {}, [], {"bytes": True, "sha256": "a" * 64}, {"bytes": -1, "sha256": "a" * 64},
    {"bytes": "1", "sha256": "a" * 64}, {"bytes": 1, "sha256": "bad"},
])
def test_archive_rejects_malformed_file_records(archive_fixture, record):
    fixture = archive_fixture
    fixture.archive_document["files"][fixture.archive_extra] = record
    bind_archive_fixture(fixture)
    with pytest.raises(ValueError, match="malformed archive file record"):
        development._admit_source(fixture.config, fixture.source_root)


def test_archive_binding_is_explicit_and_never_falls_back_to_git(archive_fixture):
    fixture = archive_fixture
    fixture.config["source"]["archive_manifest"] = None
    with pytest.raises(ValueError, match="archive_manifest"):
        development._admit_source(fixture.config, fixture.source_root)
    assert fixture.git_calls == []

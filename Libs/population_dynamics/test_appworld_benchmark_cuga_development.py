"""CUGA pipeline ownership and scoring tests using synthetic receipts only."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import sys
from unittest import mock

import pytest

from population_dynamics import appworld_benchmark_cuga_development as cuga_pipeline
from population_dynamics import appworld_benchmark_pipeline as existing_pipeline
from population_dynamics.test_appworld_benchmark_pipeline import pipeline_fixture


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cuga_fixture(pipeline_fixture, monkeypatch, tmp_path):
    state = pipeline_fixture
    Path(state["native"]["source_checkout"]).mkdir()
    public_lib = tmp_path / "staged_public_application"
    public_lib.mkdir()
    repository_root = tmp_path / "private_repository"
    repository_root.mkdir()
    cuga_environment = {
        "status": cuga_pipeline.CUGA_PREPARATION_STATUS,
        "source_revision": cuga_pipeline.CUGA_SOURCE_REVISION,
        "hostname": socket.gethostname(), "python": sys.executable,
        "packages": {
            name: "synthetic-version" for name in (
                "cuga", "langchain", "langchain-core", "langchain-openai",
                "langgraph", "fastmcp", "torch", "browsergym-core",
            )
        },
    }
    cuga_environment["packages"]["cuga"] = "0.3.2"
    environment_path = tmp_path / "cuga_environment.json"
    _write(environment_path, cuga_environment)
    state["paths"].update(
        cuga_environment=environment_path, python_base_prefix=Path(sys.base_prefix),
        public_application_lib=public_lib, repository_root=repository_root,
    )
    state.update(
        cuga_environment=cuga_environment, native_tgc=False, graph_status="graph_ended",
        native_unknown=False, startup_failure=False, owner_setup_failure=False,
        cleanup_error=False, wrong_native_task=False, wrong_graph_task=False, wrong_score_identity=False,
        evaluator_failure=False, clients=[], events=[], scorer_calls=[],
    )
    monkeypatch.setattr(
        cuga_pipeline, "owned_model_server", existing_pipeline.owned_model_server
    )
    monkeypatch.setattr(cuga_pipeline, "_check_execution_location", lambda path: None)

    class Native:
        def __init__(self, run_root, task_id, split, seed):
            self.run_root = run_root
            self.close_count = 0
            self.session = {
                "schema": "appworld_native_tool_session_v1",
                "scope": "development_tools_only", "formal_eligible": False,
                "split": split, "task_id": task_id, "seed": seed,
                "run_root": str(run_root), "data_root": state["native"]["data_root"],
                "experiment_name": "synthetic_cuga",
                "appworld_version": state["manifest"]["appworld_version"],
                "terminal": False, "status": "running", "native_world_closed": False,
                "server_exit_code": None,
            }

        async def close(self):
            self.close_count += 1
            state["events"].append("native_closed")
            if state["cleanup_error"]:
                raise RuntimeError("synthetic native cleanup could not be confirmed")
            unknown = state["native_unknown"]
            self.session.update(
                terminal=not unknown, status="unresolved" if unknown else "closed",
                native_world_closed=not unknown, server_exit_code=None if unknown else 0,
            )
            if state["wrong_native_task"]:
                self.session["task_id"] = "traincase_2"
            _write(self.run_root / "native_session.json", self.session)
            return {
                "child_reaped": not unknown, "exit_code": None if unknown else 0,
                "termination_signals": [], "request_outcome_unresolved": unknown,
            }

    async def start(command, stderr_path, *, env=None, **kwargs):
        assert state["server_alive"]
        assert command[0] == sys.executable
        assert command[1:3] == ["-m", "population_dynamics.appworld_benchmark_native_tools"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert Path(env["PYTHONPATH"]).is_absolute()
        assert all(Path(command[command.index(flag) + 1]).is_absolute() for flag in (
            "--config", "--split-manifest", "--source-root", "--run-root"
        ))
        state["native_commands"].append(list(command))
        state["events"].append("native_start")
        run_root = Path(command[command.index("--run-root") + 1])
        assert run_root.parent.is_dir() and not run_root.exists()
        run_root.mkdir()
        with Path(stderr_path).open("x", encoding="utf-8"):
            pass
        client = Native(
            run_root, command[command.index("--task-id") + 1],
            command[command.index("--split") + 1], int(command[command.index("--seed") + 1]),
        )
        state["clients"].append(client)
        if state["startup_failure"]:
            await client.close()
            raise RuntimeError("synthetic ready-response failure after owned cleanup")
        return client

    async def owner(client, *, observation, **arguments):
        assert state["server_alive"] and client.close_count == 0
        if state["owner_setup_failure"]:
            observation.update(
                status="running", stage="preflight", relay={},
                worker={}, graph_observation=None, rpc_outcome_unresolved=True,
            )
            raise OSError("synthetic failure before relay creation")
        root = arguments["output_root"]
        assert not root.exists()
        root.mkdir()
        protected = {Path(path).resolve() for path in arguments["protected_roots"]}
        assert repository_root.resolve() in protected
        assert Path(state["native"]["data_root"]).resolve() in protected
        assert Path(state["native"]["source_checkout"]).resolve() in protected
        assert client.run_root.resolve() in protected
        assert (state["paths"]["output"] / "development_score.json").resolve() in protected
        assert not any(path == root.resolve() or path in root.resolve().parents for path in protected)
        assert arguments["application_lib"] == public_lib
        assert arguments["cuga_python"] == Path(sys.executable).absolute()
        assert arguments["expected_packages"] == cuga_environment["packages"]
        cleanup = await client.close()
        graph = {
            "task_id": "traincase_2" if state["wrong_graph_task"] else arguments["task_id"],
            "seed": arguments["seed"], "status": state["graph_status"],
            "native_cleanup_settled": not state["native_unknown"],
            "native_cleanup": deepcopy(cleanup), "loop": {"synthetic_fixture": True},
        }
        observation.update(
            status="unknown" if state["native_unknown"] else (
                "worker_failed" if state["graph_status"] == "errored" else "completed"
            ),
            stage="unresolved" if state["native_unknown"] else "settled",
            worker={
                "pid": 123, "exit_code": 1 if state["graph_status"] == "errored" else 0,
                "reaped": True, "termination_signals": [],
            },
            relay={
                "status": "owner_cancelled" if state["native_unknown"] else "closed",
                "rpc_outcome_unresolved": state["native_unknown"],
                "native_cleanup": cleanup, "cleanup_error": None,
            },
            rpc_outcome_unresolved=state["native_unknown"], graph_observation=graph,
        )
        _write(root / "process_observation.json", observation)
        state["events"].append("owner_closed")
        return observation

    def scorer(command, *, output, name, environment, record, progress):
        assert state["server_alive"] and name == "scorer"
        assert command[:3] == [
            sys.executable, "-m", "population_dynamics.appworld_benchmark_scoring"
        ]
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert state["events"][-1] == "owner_closed"
        assert all(client.close_count == 1 for client in state["clients"])
        execution_path = Path(command[command.index("--execution-receipt") + 1])
        execution = json.loads(execution_path.read_text())
        assert Path(execution["run_root"]) == state["clients"][0].run_root
        assert execution["status"] != "unknown"
        score_path = Path(command[command.index("--output") + 1])
        outcome = {
            "schema": "appworld_development_score_v1", "scope": "development_only",
            "formal_eligible": False,
            **{key: execution[key] for key in ("split", "task_id", "seed", "method", "status")},
            "scoring_status": "completed", "native_tgc": state["native_tgc"],
            "evaluator_called": True, "synthetic_fixture": True,
        }
        if state["wrong_score_identity"]:
            outcome["seed"] = 42
        if state["evaluator_failure"]:
            outcome.update(scoring_status="evaluator_failure", native_tgc=None)
        _write(score_path, outcome)
        record.update(
            command=list(command), status="exited",
            exit_code=1 if state["evaluator_failure"] else 0,
        )
        state["scorer_calls"].append(list(command))
        state["events"].append("scorer")
        progress()

    state["start_mock"] = mock.AsyncMock(side_effect=start)
    state["owner_mock"] = mock.AsyncMock(side_effect=owner)
    state["scorer_mock"] = mock.Mock(side_effect=scorer)
    monkeypatch.setattr(cuga_pipeline.NativeToolClient, "start", state["start_mock"])
    monkeypatch.setattr(cuga_pipeline, "run_isolated_cuga_graph", state["owner_mock"])
    monkeypatch.setattr(cuga_pipeline, "run_child", state["scorer_mock"])
    return state


def _run(state, **overrides):
    arguments = dict(state["paths"], split="train", task_id="traincase_1", seed=41)
    arguments.update(overrides)
    return cuga_pipeline.run_cuga_development_pipeline(**arguments)


@pytest.mark.parametrize("native_tgc", [False, True])
def test_completed_attempt_preserves_negative_and_positive_native_scores(cuga_fixture, native_tgc):
    state = cuga_fixture
    state["native_tgc"] = native_tgc
    report = _run(state)
    assert report["status"] == "SCORED_DEVELOPMENT_ATTEMPT"
    assert report["native_tgc"] is native_tgc
    assert report["formal_eligible"] is False and report["usage_complete"] is False
    assert state["server_events"] == ["start", "stop"]
    assert state["events"] == ["native_start", "native_closed", "owner_closed", "scorer"]
    assert state["clients"][0].close_count == 1 and state["scorer_mock"].call_count == 1
    execution = json.loads(Path(report["execution_receipt"]).read_text())
    assert execution["run_root"] == str(state["paths"]["output"] / "native_run")
    assert execution["method"] == "cuga_full_graph"
    assert execution["cuga_usage_transport_coverage_qualified"] is False
    assert Path(report["owner_observation"]).is_file()


def test_graph_error_with_closed_native_state_is_independently_scored(cuga_fixture):
    state = cuga_fixture
    state["graph_status"] = "errored"
    report = _run(state)
    assert report["execution_status"] == "agent_failure"
    assert report["cuga_owner"]["worker"]["exit_code"] == 1
    assert report["status"] == "SCORED_DEVELOPMENT_ATTEMPT"
    assert report["native_tgc"] is False and state["scorer_mock"].call_count == 1


def test_unknown_native_outcome_withholds_scoring(cuga_fixture):
    state = cuga_fixture
    state["native_unknown"] = True
    report = _run(state)
    assert report["status"] == "UNKNOWN_NATIVE_EXECUTION"
    assert report["execution_status"] == "unknown"
    assert report["native_tgc"] is None and report["scoring_status"] == "not_run"
    assert state["scorer_mock"].call_count == 0
    assert not Path(report["scoring_outcome"]).exists()
    assert state["clients"][0].close_count == 1
    assert state["server_events"] == ["start", "stop"]


@pytest.mark.parametrize("field,value", [
    ("status", "NOT_ADMITTED"),
    ("source_revision", "0" * 40),
    ("hostname", "different-node-for-cuga-fixture.invalid"),
    ("python", "/not/an/installed/interpreter"),
])
def test_cuga_preparation_rejects_wrong_admission_identity(cuga_fixture, field, value):
    state = cuga_fixture
    state["cuga_environment"][field] = value
    _write(state["paths"]["cuga_environment"], state["cuga_environment"])
    with pytest.raises(ValueError):
        _run(state)
    assert state["server_events"] == [] and state["start_mock"].call_count == 0
    assert not state["paths"]["output"].exists()


@pytest.mark.parametrize("wrong_package", ["cuga", "missing_torch"])
def test_cuga_preparation_rejects_wrong_or_missing_packages(cuga_fixture, wrong_package):
    state = cuga_fixture
    if wrong_package == "cuga":
        state["cuga_environment"]["packages"]["cuga"] = "0.3.3"
    else:
        state["cuga_environment"]["packages"].pop("torch")
    _write(state["paths"]["cuga_environment"], state["cuga_environment"])
    with pytest.raises(ValueError):
        _run(state)
    assert state["server_events"] == [] and state["start_mock"].call_count == 0


def test_native_source_revision_remains_bound_to_canonical_preparation(cuga_fixture):
    state = cuga_fixture
    state["native"]["source_revision"] = "0" * 40
    _write(state["paths"]["native_environment"], state["native"])
    with pytest.raises(ValueError):
        _run(state)
    assert state["server_events"] == [] and state["start_mock"].call_count == 0


@pytest.mark.parametrize("overrides", [
    {"task_id": "outside_admitted_membership"},
    {"split": "test_normal", "task_id": "normalcase_1"},
    {"seed": 44},
])
def test_development_task_identity_is_required_before_launch(cuga_fixture, overrides):
    state = cuga_fixture
    with pytest.raises(ValueError):
        _run(state, **overrides)
    assert state["server_events"] == [] and state["start_mock"].call_count == 0


@pytest.mark.parametrize("identity_fault", [
    "wrong_native_task", "wrong_graph_task", "wrong_score_identity",
])
def test_generated_identity_mismatch_cannot_be_reported_as_completed(cuga_fixture, identity_fault):
    state = cuga_fixture
    state[identity_fault] = True
    report = _run(state)
    assert report["status"] == "FAILED" and report["native_tgc"] is None
    assert report["error_type"] == "ValueError"
    assert state["clients"][0].close_count == 1
    assert state["server_events"] == ["start", "stop"]
    if identity_fault != "wrong_score_identity":
        assert state["scorer_mock"].call_count == 0


def test_native_startup_failure_uses_start_cleanup_once(cuga_fixture):
    state = cuga_fixture
    state["startup_failure"] = True
    report = _run(state)
    assert report["status"] == "FAILED"
    assert report["native_client"]["status"] == "startup_failed"
    assert state["clients"][0].close_count == 1
    assert state["owner_mock"].call_count == 0 and state["scorer_mock"].call_count == 0
    assert state["server_events"] == ["start", "stop"]


def test_owner_failure_before_relay_creation_closes_the_returned_native_client(cuga_fixture):
    state = cuga_fixture
    state["owner_setup_failure"] = True
    report = _run(state)
    assert report["status"] == "FAILED"
    assert report["native_client"]["cleanup_before_relay"]["child_reaped"] is True
    assert state["clients"][0].close_count == 1
    assert state["scorer_mock"].call_count == 0 and state["server_events"] == ["start", "stop"]


def test_unconfirmed_pre_relay_cleanup_remains_unknown(cuga_fixture):
    state = cuga_fixture
    state["owner_setup_failure"] = True
    state["cleanup_error"] = True
    report = _run(state)
    assert report["status"] == "UNKNOWN"
    assert report["native_client"]["status"] == "cleanup_unresolved"
    assert report["native_client"]["cleanup_error_type"] == "RuntimeError"
    assert state["clients"][0].close_count == 1
    assert state["scorer_mock"].call_count == 0 and state["server_events"] == ["start", "stop"]


def test_model_server_startup_failure_does_not_start_native_execution(cuga_fixture):
    state = cuga_fixture
    with mock.patch.object(
        cuga_pipeline, "owned_model_server", side_effect=RuntimeError("synthetic model startup failure")
    ):
        report = _run(state)
    assert report["status"] == "FAILED" and report["native_tgc"] is None
    assert report["model_server"]["status"] == "NOT_STARTED"
    assert state["start_mock"].call_count == 0 and state["owner_mock"].call_count == 0
    assert state["scorer_mock"].call_count == 0


def test_model_server_cleanup_failure_overrides_scored_completion(cuga_fixture):
    state = cuga_fixture
    state["cleanup_failure"] = True
    report = _run(state)
    assert report["status"] == "FAILED_RUNTIME_CLEANUP"
    assert report["model_server"]["status"] == "FAIL"
    assert report["native_tgc"] is False
    assert state["scorer_mock"].call_count == 1 and state["server_events"] == ["start", "stop"]


def test_evaluator_failure_is_preserved_without_a_native_tgc(cuga_fixture):
    state = cuga_fixture
    state["evaluator_failure"] = True
    report = _run(state)
    assert report["status"] == "INCOMPLETE_DEVELOPMENT_ATTEMPT"
    assert report["scoring_status"] == "evaluator_failure"
    assert report["native_tgc"] is None and report["processes"]["scorer"]["exit_code"] == 1


def test_existing_output_and_unstaged_repository_library_are_rejected(cuga_fixture):
    state = cuga_fixture
    with pytest.raises(ValueError):
        _run(state, public_application_lib=state["paths"]["repository_root"])
    assert state["server_events"] == [] and not state["paths"]["output"].exists()
    state["paths"]["output"].mkdir()
    marker = state["paths"]["output"] / "keep.txt"
    marker.write_text("existing evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(state)
    assert marker.read_text() == "existing evidence" and state["server_events"] == []


def test_cli_passes_all_explicit_inputs_and_returns_success_for_a_scored_negative(cuga_fixture):
    state = cuga_fixture
    arguments = []
    for name, path in state["paths"].items():
        flag = "--config" if name == "config_path" else "--" + name.replace("_", "-")
        arguments.extend([flag, str(path)])
    arguments.extend(["--split", "train", "--task-id", "traincase_1", "--seed", "41"])
    with mock.patch.object(
        cuga_pipeline, "run_cuga_development_pipeline",
        return_value={"status": "SCORED_DEVELOPMENT_ATTEMPT", "native_tgc": False},
    ) as run:
        assert cuga_pipeline.main(arguments) == 0
    for name, path in state["paths"].items():
        assert run.call_args.kwargs[name] == path
    assert run.call_args.kwargs["seed"] == 41
    assert run.call_args.kwargs["command"] == arguments

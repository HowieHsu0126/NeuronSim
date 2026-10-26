"""Recipient information boundaries and actual verification-call accounting."""
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from .appworld_collaboration import CollaborationCase, resolve_arguments, api_step, check_prefix_replay
from . import appworld_collaboration as collaboration


class Backend:
    task = {"instruction": "fixture task"}

    def __init__(self):
        self.calls = []
        self.fail = False

    def call(self, *args):
        self.calls.append(args)
        return {"ok": False, "error": "ordinary API failure"} if self.fail else {"ok": True, "result": {"value": len(self.calls)}}

    def evaluate(self):
        return {"success": False, "num_tests": 1, "passes": [], "failures": [{"requirement": "fixture"}]}


@pytest.fixture
def case():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_appworld_collaboration.yaml").read_text())
    config["provider"] = "fixture"
    schemas = {"spotify": {"read": {"method": "GET", "parameters": []},
                           "write": {"method": "POST", "parameters": []}}}
    return CollaborationCase(config, schemas, Backend())


def generated(actor, kind, **kwargs):
    request = {"actor": actor, "kind": kind, **kwargs}
    return {**request, "generation": {"session_id": "fixture_" + actor, "prompt": "fixture prompt",
                                       "response": json.dumps(request), "model_revision": None}}


def tool(actor, call_id, api="read"):
    return generated(actor, "tool", call_id=call_id, app="spotify", api=api, arguments={})


def proposal(case):
    return case.handle(generated("planner", "proposal", recipient="executor", text="fixture proposal", share_results=[]))


def test_results_are_private_until_explicit_message(case):
    case.handle(tool("planner", "source_read"))
    assert not case.view("executor")["results"]
    with pytest.raises(ValueError, match="NOT_VISIBLE"):
        resolve_arguments({"$result": "source_read", "path": ["value"]}, case.results, case.visible["executor"])
    case.handle(generated("planner", "message", recipient="executor", text="inspect this result", share_results=["source_read"]))
    assert resolve_arguments({"$result": "source_read", "path": ["value"]}, case.results, case.visible["executor"]) == 1
    assert "native_evaluation" not in case.view("executor")
    assert "model_hashes" not in case.view("executor")


@pytest.mark.parametrize("allocation", [{"planner": 2, "executor": 0}, {"planner": 0, "executor": 2}])
def test_equal_total_allocations_change_which_actor_can_query(case, allocation):
    proposal(case)
    case.handle({"kind": "allocate", "allocation": allocation})
    funded = next(actor for actor, value in allocation.items() if value)
    empty = next(actor for actor, value in allocation.items() if not value)
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        case.handle(tool(empty, "unfunded"))
    for index in range(2):
        case.handle(tool(funded, "verification_" + str(index)))
    assert case.verification_used == allocation and len(case.backend.calls) == 2
    assert case.phase == "execution"
    with pytest.raises(ValueError, match="NOT_AN_EXECUTOR_TASK_WRITE"):
        case.handle(tool(funded, "hidden_extra_read"))


def test_writes_cannot_spend_verification_budget_or_execute_early(case):
    with pytest.raises(ValueError, match="WRITES_REQUIRE"):
        case.handle(tool("executor", "early_write", "write"))
    proposal(case)
    case.handle({"kind": "allocate", "allocation": {"planner": 1, "executor": 1}})
    with pytest.raises(ValueError, match="SOURCE_DECLARED_GET"):
        case.handle(tool("executor", "verification_write", "write"))
    assert not case.backend.calls and case.verification_used == {"planner": 0, "executor": 0}


def test_failed_get_and_pagination_attempts_still_consume_separate_calls(case):
    proposal(case)
    case.handle({"kind": "allocate", "allocation": {"planner": 2, "executor": 0}})
    case.backend.fail = True
    assert case.handle(tool("planner", "page_one"))["ok"] is False
    assert case.verification_used["planner"] == 1
    case.handle(tool("planner", "page_two"))
    assert case.verification_used["planner"] == 2
    with pytest.raises(ValueError):
        case.handle(tool("planner", "free_retry"))
    assert len(case.backend.calls) == 2


def test_hidden_or_arbitrary_python_access_is_not_a_tool(case):
    request = generated("planner", "tool", call_id="hidden", app="ground_truth", api="solution", arguments={})
    with pytest.raises(ValueError, match="PUBLIC_TOOL_SCHEMA"):
        case.handle(request)
    request = generated("planner", "code", code="open('ground_truth.py').read()")
    with pytest.raises(ValueError, match="UNKNOWN_ACTOR_REQUEST"):
        case.handle(request)
    assert not case.backend.calls


def test_modified_model_response_and_reused_context_are_rejected(case):
    request = tool("planner", "original")
    changed = deepcopy(request)
    changed["api"] = "write"
    with pytest.raises(ValueError, match="MODIFIED"):
        case.handle(changed)
    case.handle(request)
    changed = tool("executor", "other")
    changed["generation"]["session_id"] = "fixture_planner"
    with pytest.raises(ValueError, match="DISTINCT_CONTEXTS"):
        case.handle(changed)


def test_no_allocation_after_observing_verification_results(case):
    proposal(case)
    with pytest.raises(ValueError, match="CONTROL_ALLOCATION_REQUIRED"):
        case.handle(tool("planner", "before_allocation"))
    case.handle({"kind": "allocate", "allocation": {"planner": 1, "executor": 1}})
    case.handle(tool("planner", "first"))
    with pytest.raises(ValueError, match="FROZEN_PROPOSAL"):
        case.handle({"kind": "allocate", "allocation": {"planner": 2, "executor": 0}})


def test_generated_code_preserves_string_data_without_interpretation():
    value = "quote' newline\n__import__('os').system('false')"
    step = api_step("sample", "spotify", "read", {"query": value})
    compile(step["code"], "fixture", "exec")
    assert "apis.spotify.read" in step["code"]


def test_prefix_identity_needs_both_observations_and_hidden_database_hashes():
    check_prefix_replay(["response"], ["response"], {"state": "same"}, {"state": "same"})
    with pytest.raises(RuntimeError, match="DATABASE_REPLAY_MISMATCH"):
        check_prefix_replay(["response"], ["response"], {"state": "same"}, {"state": "changed"})
    with pytest.raises(RuntimeError, match="VISIBLE_REPLAY_MISMATCH"):
        check_prefix_replay(["response"], ["changed"], {"state": "same"}, {"state": "same"})


@pytest.mark.parametrize("returncode", [2, 1])
def test_native_command_cost_includes_nonzero_exit(tmp_path, monkeypatch, returncode):
    backend = collaboration.PrefixReplay.__new__(collaboration.PrefixReplay)
    backend.output, backend.costs = tmp_path, []
    clocks = iter([10.0, 14.0])
    usage = iter([SimpleNamespace(ru_utime=3.0, ru_stime=1.0),
                  SimpleNamespace(ru_utime=4.5, ru_stime=1.25)])
    monkeypatch.setattr(collaboration.time, "monotonic", lambda: next(clocks))
    monkeypatch.setattr(collaboration.resource, "getrusage", lambda who: next(usage))
    monkeypatch.setattr(collaboration.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode))
    if returncode == 2:
        backend._command("native", ["run"])
    else:
        with pytest.raises(RuntimeError, match="NATIVE_REFERENCE_FAILED"):
            backend._command("native", ["run"])
    assert backend.costs == [{"command": ["run"], "exit_code": returncode,
                              "wall_seconds": 4.0, "cpu_user_seconds": 1.5,
                              "cpu_system_seconds": 0.25}]


def test_native_launch_error_retains_cost_and_propagates(tmp_path, monkeypatch):
    backend = collaboration.PrefixReplay.__new__(collaboration.PrefixReplay)
    backend.output, backend.costs = tmp_path, []

    def fail(*args, **kwargs):
        raise OSError("fixture launch error")

    monkeypatch.setattr(collaboration.subprocess, "run", fail)
    with pytest.raises(OSError, match="fixture launch error"):
        backend._command("native", ["run"])
    assert backend.costs[0]["exit_code"] is None
    assert backend.costs[0]["error_type"] == "OSError"
    assert backend.costs[0]["wall_seconds"] >= 0


def test_server_wait_and_nested_native_cost_are_not_added_to_wall_time(case, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(collaboration.time, "monotonic", lambda: clock[0])

    class TimedBackend(Backend):
        def __init__(self, native, data_root, output, costs):
            super().__init__()
            self.costs = costs
            self.add_cost(2.0)

        def add_cost(self, seconds):
            clock[0] += seconds
            self.costs.append({"command": ["fixture"], "exit_code": 2,
                               "wall_seconds": seconds, "cpu_user_seconds": seconds / 2,
                               "cpu_system_seconds": 0.25})

        def evaluate(self):
            self.add_cost(3.0)
            return super().evaluate()

    class ReadyCase:
        def __init__(self, config, schemas, backend):
            self.backend, self.finished = backend, False
            self.identities, self.events = {}, []
            self.allocated, self.verification_used = None, {}

        def handle(self, request):
            self.backend.evaluate()
            self.finished = True
            return {"case_finished": True}

    class Input:
        def readline(self):
            clock[0] += 1.0
            return '{}\n'

    def ready(*args):
        clock[0] += 4.0
        return ([True], [], [])

    monkeypatch.setattr(collaboration, "PrefixReplay", TimedBackend)
    monkeypatch.setattr(collaboration, "CollaborationCase", ReadyCase)
    monkeypatch.setattr(collaboration.select, "select", ready)
    monkeypatch.setattr(collaboration.sys, "stdin", Input())
    output = tmp_path / "case"
    assert collaboration.serve(case.config, {}, tmp_path, output, ["fixture"]) == 2
    summary = json.loads((output / "summary.json").read_text())
    costs = summary["cost_accounting"]
    assert costs["server_session_wall_seconds"] == 10.0
    assert costs["request_wait_wall_seconds"] == 5.0
    assert costs["native_wall_seconds_sum"] == 5.0
    assert costs["native_cpu_user_seconds_sum"] == 2.5
    assert costs["native_command_count"] == 2
    assert costs["provider_token_usage"] is None
    assert costs["provider_inference_seconds"] is None
    assert costs["complete_host_actor_native_cost_qualified"] is False
    assert summary["formal_model_revision_verified"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["artifacts"]["summary.json"] == collaboration.sha256_file(output / "summary.json")


def test_failed_backend_initialization_still_archives_native_cost(case, tmp_path, monkeypatch):
    def failed_backend(native, data_root, output, costs):
        costs.append({"command": ["run"], "exit_code": 1, "wall_seconds": 2.0,
                      "cpu_user_seconds": 0.5, "cpu_system_seconds": 0.25})
        raise RuntimeError("fixture initialization failed")

    monkeypatch.setattr(collaboration, "PrefixReplay", failed_backend)
    output = tmp_path / "failed"
    assert collaboration.serve(case.config, {}, tmp_path, output, ["fixture"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    rows = json.loads((output / "replay_costs.json").read_text())
    assert summary["error"]["error"] == "fixture initialization failed"
    assert rows[0]["exit_code"] == 1
    assert summary["cost_accounting"]["native_command_count"] == 1
    assert summary["cost_accounting"]["native_wall_seconds_sum"] == 2.0


LOCAL_REVISION = "fixture/model@exact-source-revision"


@pytest.fixture
def local_case(case):
    config = {**deepcopy(case.config), "provider": "local_transformers", "model_revision": LOCAL_REVISION}
    return CollaborationCase(config, case.schemas, Backend())


def local_receipt(request, **changes):
    request = deepcopy(request)
    request["generation"].update(session_id="local_" + request["actor"], model_revision=LOCAL_REVISION,
                                 usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                                 inference_seconds=0.25)
    request["generation"].update(changes)
    return request


def serve_requests(case, requests, tmp_path, monkeypatch):
    monkeypatch.setattr(collaboration, "PrefixReplay", lambda *args: case.backend)
    monkeypatch.setattr(collaboration.select, "select", lambda *args: ([True], [], []))
    monkeypatch.setattr(collaboration.sys, "stdin", StringIO("".join(json.dumps(row) + "\n" for row in requests)))
    output = tmp_path / "served"
    code = collaboration.serve(case.config, case.schemas, tmp_path, output, ["fixture"])
    summary = json.loads((output / "summary.json").read_text())
    transport = [json.loads(line) for line in (output / "transport.jsonl").read_text().splitlines()]
    return code, summary, transport


def test_local_provider_roundtrip_aggregates_generation_costs(local_case, tmp_path, monkeypatch):
    requests = [
        local_receipt(generated("planner", "proposal", recipient="executor", text="check before write", share_results=[])),
        {"kind": "allocate", "allocation": {"planner": 1, "executor": 1}},
        local_receipt(tool("planner", "planner_check")),
        local_receipt(tool("executor", "executor_check")),
        local_receipt(tool("executor", "task_write", "write")),
        local_receipt(generated("executor", "finish")),
    ]
    code, summary, transport = serve_requests(local_case, requests, tmp_path, monkeypatch)
    assert code == 2 and len(local_case.backend.calls) == 3
    assert summary["provider"] == "local_transformers"
    assert summary["agent_execution_performed"] is True
    assert summary["agent_execution_semantics"] == "host_supplied_distinct_contexts"
    assert summary["formal_model_revision_verified"] is False
    costs = summary["cost_accounting"]
    assert costs["provider_inference_seconds"] == 1.25
    assert costs["provider_token_usage"] == {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    assert costs["provider_generation_request_count"] == costs["provider_inference_observation_count"] == costs["provider_token_usage_observation_count"] == 5
    assert costs["provider_telemetry_status"] == "COMPLETE_FOR_OBSERVED_GENERATIONS"
    assert costs["status"] == "PARTIAL_HOST_ACTOR_NATIVE_COST_UNQUALIFIED"
    assert costs["complete_host_actor_native_cost_qualified"] is False
    assert "unreported_generation_attempts" in costs["provider_measurement_excludes"]
    assert len(transport) == 6 and transport[0]["request"] == requests[0]
    assert transport[0]["provider_telemetry"]["status"] == "COMPLETE"
    assert "provider_telemetry" not in transport[1]


def test_local_revision_rejection_precedes_native_calls_and_context_changes(local_case):
    with pytest.raises(ValueError, match="LOCAL_MODEL_REVISION_MISMATCH"):
        local_case.handle(local_receipt(tool("planner", "wrong_revision"), model_revision=LOCAL_REVISION + "-other"))
    assert not local_case.backend.calls and not local_case.identities and local_case.turns == 0


@pytest.mark.parametrize("changes", [
    {"usage": None},
    {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    {"usage": {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2}},
    {"usage": {"prompt_tokens": 1, "completion_tokens": -1, "total_tokens": 0}},
    {"usage": {"prompt_tokens": 1.0, "completion_tokens": 1, "total_tokens": 2}},
    {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3}},
    {"inference_seconds": float("nan")},
    {"inference_seconds": float("inf")},
    {"inference_seconds": float("-inf")},
    {"inference_seconds": True},
    {"inference_seconds": "0.25"},
    {"inference_seconds": -0.25},
])
def test_local_malformed_generation_telemetry_cannot_execute(local_case, changes):
    with pytest.raises(ValueError, match="GENERATION_.*INVALID"):
        local_case.handle(local_receipt(tool("planner", "invalid_telemetry"), **changes))
    assert not local_case.backend.calls and not local_case.identities and local_case.turns == 0


def test_local_receipt_still_requires_unmodified_response_and_stable_distinct_contexts(local_case):
    original = local_receipt(tool("planner", "source"))
    changed = deepcopy(original)
    changed["api"] = "write"
    with pytest.raises(ValueError, match="MODIFIED"):
        local_case.handle(changed)
    local_case.handle(original)
    with pytest.raises(ValueError, match="CONTEXT_ID_CHANGED"):
        local_case.handle(local_receipt(tool("planner", "changed_context"), session_id="another_context"))
    with pytest.raises(ValueError, match="DISTINCT_CONTEXTS"):
        local_case.handle(local_receipt(tool("executor", "shared_context"), session_id="local_planner"))
    assert len(local_case.backend.calls) == 1


@pytest.mark.parametrize("failure", ["api", "request", "revision", "native"])
def test_rejected_or_failed_requests_retain_observed_cost(local_case, tmp_path, monkeypatch, failure):
    request = local_receipt(tool("planner", "failure"))
    if failure == "api":
        local_case.backend.fail = True
    elif failure == "request":
        request = local_receipt(generated("planner", "unsupported_kind"))
    elif failure == "revision":
        request["generation"]["model_revision"] = "wrong_revision"
    else:
        def fail(*args):
            raise RuntimeError("native failure")
        monkeypatch.setattr(local_case.backend, "call", fail)
    code, summary, transport = serve_requests(local_case, [request], tmp_path, monkeypatch)
    assert code == 1
    costs = summary["cost_accounting"]
    assert costs["provider_inference_seconds"] == 0.25
    assert costs["provider_token_usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert costs["provider_generation_request_count"] == 1
    assert costs["provider_telemetry_status"] == "COMPLETE_FOR_OBSERVED_GENERATIONS"
    assert costs["complete_host_actor_native_cost_qualified"] is False
    assert len(transport) == 1 and transport[0]["request"] == request
    assert transport[0]["provider_telemetry"]["status"] == "COMPLETE"
    if failure == "api":
        assert transport[0]["response"]["result"]["ok"] is False
    else:
        assert transport[0]["response"]["ok"] is False
    if failure == "native":
        assert summary["error"] == {"error_type": "RuntimeError", "error": "native failure"}


@pytest.mark.parametrize("changes,drop,has_usage,has_time", [
    ({"inference_seconds": float("nan")}, None, True, False),
    ({"usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}}, None, False, True),
    ({}, "usage", False, True),
    ({"usage": None, "inference_seconds": False}, None, False, False),
])
def test_invalid_receipts_report_partial_or_unknown_costs(local_case, tmp_path, monkeypatch,
                                                          changes, drop, has_usage, has_time):
    request = local_receipt(tool("planner", "bad_cost"), **changes)
    if drop:
        del request["generation"][drop]
    code, summary, transport = serve_requests(local_case, [request], tmp_path, monkeypatch)
    assert code == 1 and not local_case.backend.calls
    costs = summary["cost_accounting"]
    assert costs["provider_generation_request_count"] == 1
    assert costs["provider_token_usage_observation_count"] == int(has_usage)
    assert costs["provider_inference_observation_count"] == int(has_time)
    assert costs["provider_telemetry_status"] == ("PARTIAL" if has_usage or has_time else "UNKNOWN")
    assert costs["status"] == "PARTIAL_MISSING_OR_INVALID_PROVIDER_TELEMETRY"
    assert costs["provider_token_usage"] == ({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12} if has_usage else None)
    assert costs["provider_inference_seconds"] == (0.25 if has_time else None)
    assert costs["complete_host_actor_native_cost_qualified"] is False
    assert len(transport) == 1 and transport[0]["response"]["ok"] is False
    if "inference_seconds" in changes and isinstance(changes["inference_seconds"], float):
        assert transport[0]["request"] is None
        assert "NaN" in transport[0]["request_raw_json"]
    assert transport[0]["provider_telemetry"]["errors"]


@pytest.mark.parametrize("provider", ["fixture", "host_collaboration"])
def test_legacy_provider_keeps_exact_four_key_receipts_and_six_key_config(case, provider):
    case.config["provider"] = provider
    assert collaboration.validate_config(case.config) == case.config
    request = tool("planner", "legacy")
    assert request["generation"]["model_revision"] is None
    assert case.handle(request)["ok"] is True
    with pytest.raises(ValueError, match="EXACT_GENERATION_RECEIPT_REQUIRED"):
        case.handle(local_receipt(tool("executor", "extra_fields")))
    with pytest.raises(ValueError, match="COLLABORATION_CONFIG_INVALID"):
        collaboration.validate_config({**case.config, "model_revision": LOCAL_REVISION})
    assert len(case.backend.calls) == 1


def test_local_config_requires_one_nonempty_exact_revision_field(case):
    config = {**case.config, "provider": "local_transformers"}
    with pytest.raises(ValueError, match="COLLABORATION_CONFIG_INVALID"):
        collaboration.validate_config(config)
    for revision in (None, "", " ", False):
        with pytest.raises(ValueError, match="LOCAL_MODEL_REVISION_REQUIRED"):
            collaboration.validate_config({**config, "model_revision": revision})
    assert collaboration.validate_config({**config, "model_revision": LOCAL_REVISION})["model_revision"] == LOCAL_REVISION


@pytest.mark.parametrize("operation", ["call", "evaluate"])
def test_native_value_error_terminates_without_consuming_next_request(local_case, tmp_path, monkeypatch, operation):
    def fail(*args):
        raise ValueError("malformed native JSON")
    if operation == "evaluate":
        monkeypatch.setattr(local_case.backend, "evaluate", fail)
        requests = [
            local_receipt(generated("planner", "proposal", recipient="executor", text="verify first", share_results=[])),
            {"kind": "allocate", "allocation": {"planner": 1, "executor": 1}},
            local_receipt(tool("planner", "planner_verification")),
            local_receipt(tool("executor", "executor_verification")),
            local_receipt(generated("executor", "finish")),
        ]
    else:
        monkeypatch.setattr(local_case.backend, "call", fail)
        requests = [local_receipt(tool("planner", "native_parse_failure"))]
    expected_requests = len(requests)
    expected_generations = sum("generation" in request for request in requests)
    requests.append(local_receipt(tool("planner", "must_not_execute")))
    code, summary, transport = serve_requests(local_case, requests, tmp_path, monkeypatch)
    assert code == 1 and len(transport) == expected_requests
    assert summary["error"] == {"error_type": "RuntimeError", "error": "NATIVE_OPERATION_VALUE_ERROR:malformed native JSON"}
    assert transport[-1]["response"]["error_type"] == "RuntimeError"
    assert summary["cost_accounting"]["provider_generation_request_count"] == expected_generations
    assert summary["cost_accounting"]["provider_inference_seconds"] == 0.25 * expected_generations
    assert not any(call[0] == "must_not_execute" for call in local_case.backend.calls)


def test_empty_native_mutation_counters_do_not_hide_different_sql_changes(tmp_path):
    state = tmp_path / 'state'; state.mkdir()
    (state / 'model_hashes.json').write_text('{}')
    (state / 'spotify.jsonl').write_text('["UPDATE fixture SET value = ?", [1], false]\n')
    before = collaboration.database_journal_signature(state)
    (state / 'spotify.jsonl').write_text('["UPDATE fixture SET value = ?", [2], false]\n')
    after = collaboration.database_journal_signature(state)
    assert before['model_mutation_counters'] == after['model_mutation_counters'] == {}
    with pytest.raises(RuntimeError, match='DATABASE_REPLAY_MISMATCH'):
        check_prefix_replay(['same observation'], ['same observation'], before, after)
    with pytest.raises(RuntimeError, match='CHANGE_JOURNALS_MISSING'):
        collaboration.database_journal_signature(tmp_path / 'missing')


@pytest.mark.parametrize('mismatch', [None, 'observations_sha256', 'database_journals'])
def test_recorded_native_checkpoint_is_checked_before_accepting_call(case, tmp_path, monkeypatch, mismatch):
    backend = collaboration.PrefixReplay.__new__(collaboration.PrefixReplay)
    backend.native = deepcopy(case.config['native'])
    backend.output = tmp_path
    backend.data_root = tmp_path
    backend.last_run = backend.model_hashes = None
    backend.observations = []
    steps = [{'label': str(i), 'code': 'print fixture'} for i in range(6)]
    observations = [json.dumps({'fixture_index': i}) for i in range(6)]
    run = tmp_path / 'call_005'
    state = run / 'native/experiments/outputs/reference/tasks' / backend.native['task_id'] / 'dbs'
    state.mkdir(parents=True)
    (state/'model_hashes.json').write_text('{}')
    (state/'spotify.jsonl').write_text('["UPDATE fixture SET value = ?", [1], false]\n')
    (run/'trace.jsonl').write_text(''.join(json.dumps({'operation':'code','output':value})+'\n' for value in observations))
    checkpoint = dict(schema='appworld_recorded_native_prefix_v1', task_id=backend.native['task_id'],
        native_configuration_sha256=collaboration.content_digest({k:v for k,v in backend.native.items() if k!='steps'}),
        native_step_count=6, steps_sha256=collaboration.content_digest(steps),
        observations_sha256=collaboration.content_digest(observations),
        database_journals=collaboration.database_journal_signature(state), prefix_transport_requests=28,
        reused_generation_request_indices=[1,4,7,10,13,17,20,23,26], source_artifacts=['fixture'])
    backend.prefix_checkpoint = collaboration.validate_prefix_checkpoint(checkpoint, backend.native)
    backend.checkpoint_verification = {'status':'NOT_REACHED'}
    if mismatch=='observations_sha256':
        backend.prefix_checkpoint[mismatch]='0'*64
    elif mismatch=='database_journals':
        (state/'spotify.jsonl').write_text('["UPDATE fixture SET value = ?", [2], false]\n')
    monkeypatch.setattr(backend,'_command',lambda *args:None)
    if mismatch is None:
        assert backend._run(steps,False)=={'fixture_index':5}
        assert backend.checkpoint_verification['status']=='PASS'
        assert backend.last_run==run
    else:
        with pytest.raises(RuntimeError,match='RECORDED_NATIVE_PREFIX_CHECKPOINT_MISMATCH'):
            backend._run(steps,False)
        assert backend.checkpoint_verification['status']=='FAILED' and backend.last_run is None


def test_prefix_constructor_accepts_checkpoint_and_starts_unverified(case, tmp_path, monkeypatch):
    native = case.config['native']
    checkpoint = dict(schema='appworld_recorded_native_prefix_v1', task_id=native['task_id'],
        native_configuration_sha256=collaboration.content_digest({k:v for k,v in native.items() if k!='steps'}),
        native_step_count=6, steps_sha256='0'*64, observations_sha256='0'*64,
        database_journals={'format':'appworld_sql_change_journals_v1','files':{'fixture.jsonl':{}},'model_mutation_counters':{}},
        prefix_transport_requests=28, reused_generation_request_indices=[1,4,7,10,13,17,20,23,26], source_artifacts=['fixture'])
    monkeypatch.setattr(collaboration.PrefixReplay,'_run',lambda self,steps,read_only:{'instruction':'fixture task'})
    backend = collaboration.PrefixReplay(native,tmp_path,tmp_path/'native',prefix_checkpoint=checkpoint)
    assert backend.prefix_checkpoint==checkpoint and backend.checkpoint_verification=={'status':'NOT_REACHED'}


def test_prefix_only_controller_close_preserves_native_checkpoint_evidence(case, tmp_path, monkeypatch):
    native = case.config['native']
    checkpoint = dict(schema='appworld_recorded_native_prefix_v1', task_id=native['task_id'],
        native_configuration_sha256=collaboration.content_digest({k:v for k,v in native.items() if k!='steps'}),
        native_step_count=6, steps_sha256='0'*64, observations_sha256='0'*64,
        database_journals={'format':'appworld_sql_change_journals_v1','files':{'fixture.jsonl':{}},'model_mutation_counters':{}},
        prefix_transport_requests=28,reused_generation_request_indices=[1,4,7,10,13,17,20,23,26],source_artifacts=['fixture'])
    case.backend.checkpoint_verification={'status':'PASS','native_step_count':6}
    monkeypatch.setattr(collaboration,'PrefixReplay',lambda *args,**kwargs:case.backend)
    monkeypatch.setattr(collaboration.sys,'stdin',StringIO(''))
    monkeypatch.setattr(collaboration.select,'select',lambda *args:([collaboration.sys.stdin],[],[]))
    output=tmp_path/'prefix_closed'
    assert collaboration.serve(case.config,case.schemas,tmp_path,output,['fixture'],prefix_checkpoint=checkpoint)==1
    summary=json.loads((output/'summary.json').read_text())
    assert summary['error']['error']=='MODEL_HOST_CLOSED_BEFORE_FINISH'
    assert summary['recorded_native_prefix']==case.backend.checkpoint_verification
    assert summary['scientific_control_transfer_qualified'] is False

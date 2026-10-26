"""Exact metadata coverage and delegated snapshot reuse; no numerical kernels."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from . import pfc_merlin_observation_completion as completion

COLLECTOR_URI = "hdfs://haruna/checkpoints/20261022/" + completion.COLLECTOR_RUN + "/pfc_downstream/progress.json"


def progress_for(indices=None):
    indices = completion.ORIGINAL_IDS if indices is None else set(indices)
    restored = sorted(indices - completion.BASELINE)
    snapshots = [] if not restored else [dict(indices=restored, bundle_uri=COLLECTOR_URI[:-len('/progress.json')]
        + f"/readout_snapshots/readouts_{restored[0]:04d}_{restored[-1]:04d}/result.tar.gz",
        bundle_bytes=1, bundle_sha256="a" * 64, run_id=completion.COLLECTOR_RUN,
        trial_id=completion.COLLECTOR_TRIAL, source_simulations=0, production_episodes=0)]
    return dict(status="WAITING_COMPLETE3000_VERIFIED_READOUTS", run_id=completion.COLLECTOR_RUN,
        trial_id=completion.COLLECTOR_TRIAL, required=3000, production_episodes=0,
        collected=len(indices), collected_indices=sorted(indices), readout_snapshots=snapshots)


def test_exact_original2999_snapshot_coverage():
    progress = progress_for()
    indices, snapshots = completion.validate_collector(progress, COLLECTOR_URI, [])
    assert len(indices) == 2999 and 1906 not in indices
    assert set(snapshots[0]["indices"]) | completion.BASELINE | {1906} == set(range(3000))


@pytest.mark.parametrize("change", ["duplicate", "missing", "foreign", "recovery", "sha", "source_owner", "source_scope", "uri"])
def test_invalid_snapshot_metadata_is_rejected(change):
    progress = progress_for()
    snapshot = progress["readout_snapshots"][0]
    if change == "duplicate":
        snapshot["indices"].append(snapshot["indices"][0])
    elif change == "missing":
        snapshot["indices"].remove(36)
    elif change == "foreign":
        progress["collected_indices"].append(3000)
    elif change == "recovery":
        snapshot["indices"].append(1906)
    elif change == "sha":
        snapshot["bundle_sha256"] = "invalid"
    elif change == "source_owner":
        snapshot["trial_id"] = "foreign-trial"
    elif change == "source_scope":
        snapshot["source_simulations"] = 1
    else:
        snapshot["bundle_uri"] = "hdfs://foreign/result.tar.gz"
    with pytest.raises(ValueError):
        completion.validate_collector(progress, COLLECTOR_URI, [])


def test_previously_observed_immutable_snapshot_cannot_change():
    progress = progress_for(completion.BASELINE | {35, 36})
    previous = copy.deepcopy(progress["readout_snapshots"])
    progress["readout_snapshots"][0]["bundle_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="IMMUTABLE_SNAPSHOT_HISTORY_CHANGED"):
        completion.validate_collector(progress, COLLECTOR_URI, previous)


def test_runtime_source_binding_rejects_other_code_before_import(tmp_path):
    path = tmp_path / "runtime.py"
    path.write_text('raise AssertionError("must not execute")')
    binding = dict(path=str(path), bytes=path.stat().st_size, sha256=completion.sha(path))
    with pytest.raises(ValueError, match="QUALIFIED_RUNTIME_SOURCE_REQUIRED"):
        completion.load_runtime(binding, tmp_path)
    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="LOCAL_INPUT_BINDING_CHANGED"):
        completion.load_runtime(binding, tmp_path)


class MockRuntime:
    def __init__(self, domain):
        self.DOMAIN = domain
        self.calls = []
    def sha(self, path):
        return hashlib.sha256(str(path).encode()).hexdigest()
    def restore_snapshot(self, snapshot):
        self.calls.append(("restore", copy.deepcopy(snapshot)))
        self.unpack_readouts(snapshot["bundle_uri"], snapshot["bundle_sha256"])
        return list(snapshot["indices"])
    def unpack_readouts(self, bundle, digest):
        self.calls.append(("unpack", bundle, digest))
    def persist_readouts(self, indices, plan, entry):
        self.calls.append(("persist", list(indices), entry(1906)))
        return dict(indices=[1906], run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                    source_simulations=0, production_episodes=0)
    def hdfs_json(self, uri):
        self.calls.append(("observe", uri))
        return progress_for()
    def publish_json(self, name, value):
        self.calls.append(("publish_json", name, value["status"]))
    def publish_results(self, roots, code):
        self.calls.append(("publish_results", list(roots), code))
    def simulate(self, *args):
        pytest.fail("Source simulation is forbidden")
    def generate_readouts(self, *args):
        pytest.fail("Readout generation is forbidden")


def completion_fixture(tmp_path, monkeypatch):
    runtime = MockRuntime(tmp_path / "domain")
    prepared = runtime.DOMAIN / "prepared"
    prepared.mkdir(parents=True)
    rows = []
    for index in range(3000):
        circuit, offset = divmod(index, 30)
        block, condition = 6 + offset // 10, offset % 10
        rows.append(dict(task_index=index, identity=dict(circuit_index=circuit, block_index=block,
            condition_index=condition, episode_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{condition:02d}")))
    (prepared / "execution.json").write_text(json.dumps(dict(rows=rows)))
    accepted = dict(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=1906, identity=rows[1906]["identity"],
        run_id=completion.SOURCE_RUN, trial_id=completion.SOURCE_TRIAL, source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scientific_parameter_domain_qualified=False, scaling_qualified=False,
        source_bundle_uri="hdfs://haruna/checkpoints/20261021/" + completion.SOURCE_RUN + "/pfc/task_1906/result.tar.gz",
        source_bundle_sha256=completion.SOURCE_BUNDLE_SHA256, observation_entry=completion.observation_entry(runtime,1906))
    accepted_path = tmp_path / "accepted1906.json"
    accepted_path.write_text(json.dumps(accepted))
    work, summary_output, dynamics_output = tmp_path / "completion", tmp_path / "summary", tmp_path / "dynamics"
    request = dict(schema=completion.SCHEMA, runtime=dict(path="staged-runtime.py", bytes=16600, sha256=completion.RUNTIME_SHA256),
        accepted_1906=dict(path=str(accepted_path), bytes=accepted_path.stat().st_size, sha256=completion.sha(accepted_path)),
        preparation_sha256=completion.sha(prepared / "execution.json"), collector_progress_uri=COLLECTOR_URI,
        work=str(work), summary_output=str(summary_output), dynamics_output=str(dynamics_output),
        publication_roots=[str(summary_output),str(dynamics_output)],
        consumer_sha256={"pfc_parameter_domain_summary.py":completion.sha(completion.summary.__file__),
                         "provenance.py":completion.sha(completion.provenance.__file__), "pfc_parameter_task_dynamics.py":"c"*64},
        summary_command=["python3","-B","-m","population_dynamics.pfc_parameter_domain_summary","--prepared",str(prepared),
                         "--observations",str(work / "complete_observations.json"),"--output",str(summary_output)],
        dynamics_command=["python3","-B","unchanged_dynamics.py",str(runtime.DOMAIN)])
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(completion,"load_runtime",lambda binding,base:runtime)
    monkeypatch.setattr(completion,"validate_plan",lambda plan:None)
    loaded = []
    monkeypatch.setattr(completion.summary,"load_observation",lambda entry,row:loaded.append(row["task_index"]))
    monkeypatch.setattr(completion.provenance,"freeze_json",lambda path,value:path.write_text(json.dumps(value)))
    commands=[]
    monkeypatch.setattr(completion.subprocess,"run",lambda argv,**kwargs:(commands.append(argv) or SimpleNamespace(returncode=0)))
    monkeypatch.setenv("RH2_JOB_RUN_ID","new-completion-run")
    monkeypatch.setenv("ARNOLD_TRIAL_ID","new-completion-trial")
    monkeypatch.setenv("ARNOLD_APPLIED_GPU_NUM","0")
    return request_path, request, runtime, loaded, commands


def test_completion_restores_only_existing_data_and_publishes_exact_manifest(tmp_path, monkeypatch):
    path, request, runtime, loaded, commands = completion_fixture(tmp_path,monkeypatch)
    state = completion.run(path)
    restores = [call[1] for call in runtime.calls if call[0]=="restore"]
    assert len(restores)==2 and restores[0]["indices"]==[1906]
    assert restores[0]["bundle_uri"].endswith("/"+completion.SOURCE_RUN+"/pfc/task_1906/result.tar.gz")
    assert restores[0]["bundle_bytes"]==86010908 and restores[0]["bundle_sha256"]==completion.SOURCE_BUNDLE_SHA256
    labels=[call[0] for call in runtime.calls]
    assert labels.index("persist") < labels.index("observe")
    assert loaded==[0,1906]  # aggregate owns the unchanged full3000 numerical checks
    manifest_path=Path(request["work"])/"complete_observations.json"
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest)==3000
    assert "pfc_working_memory_circuit_063_block_007_condition_06" in manifest
    publication=[call for call in runtime.calls if call[0]=="publish_results"][-1]
    assert manifest_path in publication[1] and Path(request["work"])/"completion_report.json" in publication[1]
    assert Path(request["work"])/"consumer.log" in publication[1] and publication[2]==0
    assert commands==[request["summary_command"],request["dynamics_command"]]
    assert state["collected"]==3000 and state["source_execution_reruns"]==state["readout_execution_reruns"]==0
    assert all(state[field] is False for field in completion.QUALIFICATION_FLAGS)


def test_publication_bundle_cannot_replace_accepted_original1906_source(tmp_path,monkeypatch):
    path,request,runtime,_,_=completion_fixture(tmp_path,monkeypatch)
    accepted_path=Path(request["accepted_1906"]["path"])
    accepted=json.loads(accepted_path.read_text())
    accepted["source_bundle_uri"]="hdfs://haruna/checkpoints/publication/pfc_publication/task_1906/result.tar.gz"
    accepted_path.write_text(json.dumps(accepted))
    request["accepted_1906"].update(bytes=accepted_path.stat().st_size,sha256=completion.sha(accepted_path))
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match="ACCEPTED1906_ORIGINAL_BUNDLE_REQUIRED"):
        completion.run(path)
    assert not any(call[0] in ("restore","persist","observe") for call in runtime.calls)


class SupplementalMockRuntime(MockRuntime):
    def persist_readouts(self, indices, plan, entry):
        index = indices[0]
        self.calls.append(("persist", list(indices), entry(index)))
        return dict(indices=list(indices), bundle_bytes=1032594, bundle_sha256="a" * 64,
            bundle_uri="hdfs://haruna/checkpoints/" + os.environ["RH2_JOB_RUN_ID"]
                + f"/pfc_downstream/readout_snapshots/readouts_{index:04d}_{index:04d}/result.tar.gz",
            run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
            source_simulations=0, production_episodes=0)


def local_binding(path, **fields):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=completion.sha(path), **fields)


def supplemental_fixture(tmp_path, monkeypatch):
    path, request, old_runtime, loaded, commands = completion_fixture(tmp_path, monkeypatch)
    runtime = SupplementalMockRuntime(old_runtime.DOMAIN)
    monkeypatch.setattr(completion, "load_runtime", lambda binding,base:runtime)
    rows = json.loads((runtime.DOMAIN / "prepared/execution.json").read_bytes())["rows"]
    accepted1906 = json.loads(Path(request["accepted_1906"]["path"]).read_bytes())
    reused = dict(indices=[1906], bundle_bytes=1032594, bundle_sha256="a" * 64,
        bundle_uri="hdfs://haruna/checkpoints/" + completion.PREVIOUS_COMPLETION_RUN
            + "/pfc_downstream/readout_snapshots/readouts_1906_1906/result.tar.gz",
        run_id=completion.PREVIOUS_COMPLETION_RUN, trial_id=completion.PREVIOUS_COMPLETION_TRIAL,
        source_simulations=0, production_episodes=0)
    prior = dict(schema=completion.SCHEMA, status="WAITING_ORIGINAL2999_SNAPSHOTS",
        execution=dict(backend="merlin", run_id=completion.PREVIOUS_COMPLETION_RUN,
                       trial_id=completion.PREVIOUS_COMPLETION_TRIAL, allocated_gpus=0),
        collector_execution=dict(run_id=completion.COLLECTOR_RUN, trial_id=completion.COLLECTOR_TRIAL),
        required_episodes=3000, runtime_sha256=completion.RUNTIME_SHA256,
        preparation_sha256=request["preparation_sha256"], consumer_sha256=request["consumer_sha256"],
        accepted_1906_metadata_sha256=request["accepted_1906"]["sha256"],
        source_1906_snapshot=completion.accepted_source_snapshot(accepted1906, rows[1906], runtime),
        recovered_1906_snapshot=reused, source_simulations=0, production_episodes=0,
        source_execution_reruns=0, readout_execution_reruns=0, failures=[],
        **{field:False for field in completion.QUALIFICATION_FLAGS})
    prior_path = tmp_path / "prior_completion.json"
    prior_path.write_text(json.dumps(dict(uri="hdfs://haruna/checkpoints/" + completion.PREVIOUS_COMPLETION_RUN
        + "/pfc_downstream/observation_completion_progress.json", progress=prior)))
    request["prior_completion_progress"] = local_binding(prior_path)
    sources = []
    for index in sorted(completion.SUPPLEMENTAL_SOURCE_INDICES):
        checkpoint = "hdfs://haruna/checkpoints/" + completion.SOURCE_RUN + f"/pfc/task_{index:04d}"
        receipt = dict(status="COMPLETE", exit_code=0, persisted=True, index=index,
            run_id=completion.SOURCE_RUN, trial_id=completion.SOURCE_TRIAL, xie_job=f"original-{index}",
            checkpoint_path=checkpoint, result_bytes=86010908, result_sha256="b" * 64,
            publication_recovery=dict(source_bundle_uri=checkpoint + "/result.tar.gz", run_id="publisher-run"))
        accepted = dict(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=index, identity=rows[index]["identity"],
            run_id=completion.SOURCE_RUN, trial_id=completion.SOURCE_TRIAL, xie_job=receipt["xie_job"],
            source_simulations=0, production_episodes=0, source_qualification_granted=False,
            scientific_parameter_domain_qualified=False, scaling_qualified=False, remote_receipt=receipt,
            source_bundle_uri=checkpoint + "/result.tar.gz", source_bundle_sha256="b" * 64,
            receipt_uri=f"hdfs://haruna/publisher/pfc_publication/task_{index:04d}/receipt.json",
            validation_execution=dict(backend="merlin", run_id="intake-run", trial_id="intake-trial",
                                      allocated_gpus=0, numpy_version="2.4.2"),
            observation_entry=completion.observation_entry(runtime, index))
        accepted_path = tmp_path / f"accepted_{index}.json"
        accepted_path.write_text(json.dumps(accepted))
        sources.append(local_binding(accepted_path, index=index))
    request["accepted_sources"] = sources
    path.write_text(json.dumps(request))
    return path, request, runtime, loaded, commands


def test_prepare_supplements_reuses1906_and_publishes_ten_single_task_snapshots(tmp_path, monkeypatch):
    path, request, runtime, loaded, commands = supplemental_fixture(tmp_path, monkeypatch)
    before = {binding["path"]:Path(binding["path"]).read_bytes()
              for binding in [request["accepted_1906"], request["prior_completion_progress"], *request["accepted_sources"]]}
    state = completion.prepare_supplements(path)
    restores = [call[1] for call in runtime.calls if call[0] == "restore"]
    assert len(restores) == 11 and restores[0] == state["reused_1906_snapshot"]
    assert completion.SOURCE_RUN not in restores[0]["bundle_uri"]
    assert restores[0]["run_id"] == completion.PREVIOUS_COMPLETION_RUN
    assert [snapshot["indices"][0] for snapshot in restores[1:]] == sorted(completion.SUPPLEMENTAL_SOURCE_INDICES)
    persists = [call for call in runtime.calls if call[0] == "persist"]
    assert [call[1] for call in persists] == [[index] for index in sorted(completion.SUPPLEMENTAL_SOURCE_INDICES)]
    assert all(call[1] != [1906] for call in persists)
    assert commands == loaded == []
    assert not any(call[0] == "observe" for call in runtime.calls)
    assert not Path(request["summary_output"]).exists() and not Path(request["dynamics_output"]).exists()
    assert state["status"] == "SUPPLEMENTAL_READOUTS_READY" and state["prepared_episodes"] == 10
    assert state["closure_verified"] is False and state["summary_executions"] == state["dynamics_executions"] == 0
    assert all(state[field] is False for field in completion.QUALIFICATION_FLAGS)
    assert state["takeover_inputs"]["consumer_start_authorized"] is False
    assert len(state["takeover_inputs"]["readout_snapshots"]) == 11
    for source in state["supplements"]:
        assert source["source_execution"]["run_id"] == completion.SOURCE_RUN
        assert source["source_bundle_snapshot"]["bundle_bytes"] == 86010908
        assert source["readout_snapshot"]["run_id"] == os.environ["RH2_JOB_RUN_ID"]
    assert all(Path(path).read_bytes() == value for path,value in before.items())


@pytest.mark.parametrize("change", ["identity", "bundle", "source_trial", "source_job", "qualification", "receipt", "validation"])
def test_prepare_rejects_invalid_source_bindings_before_restoring(tmp_path, monkeypatch, change):
    path, request, runtime, _, commands = supplemental_fixture(tmp_path, monkeypatch)
    binding = request["accepted_sources"][0]
    accepted_path = Path(binding["path"])
    accepted = json.loads(accepted_path.read_bytes())
    if change == "identity":
        accepted["identity"]["block_index"] += 1
    elif change == "bundle":
        accepted["source_bundle_sha256"] = "c" * 64
    elif change == "source_trial":
        accepted["trial_id"] = "foreign-trial"
    elif change == "source_job":
        accepted["xie_job"] = "foreign-job"
    elif change == "qualification":
        accepted["scaling_qualified"] = True
    elif change == "receipt":
        accepted["remote_receipt"]["persisted"] = False
    else:
        accepted["validation_execution"]["numpy_version"] = "different"
    accepted_path.write_text(json.dumps(accepted))
    binding.update(local_binding(accepted_path))
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        completion.prepare_supplements(path)
    assert commands == [] and not any(call[0] in ("restore", "persist") for call in runtime.calls)


@pytest.mark.parametrize("change", ["owner", "runtime", "acceptance_sha", "source_snapshot", "snapshot_owner", "snapshot_uri", "scope"])
def test_prepare_rejects_invalid_prior1906_provenance(tmp_path, monkeypatch, change):
    path, request, runtime, _, commands = supplemental_fixture(tmp_path, monkeypatch)
    prior_path = Path(request["prior_completion_progress"]["path"])
    wrapper = json.loads(prior_path.read_bytes())
    prior = wrapper["progress"]
    if change == "owner":
        prior["execution"]["trial_id"] = "foreign-trial"
    elif change == "runtime":
        prior["runtime_sha256"] = "b" * 64
    elif change == "acceptance_sha":
        prior["accepted_1906_metadata_sha256"] = "b" * 64
    elif change == "source_snapshot":
        prior["source_1906_snapshot"]["bundle_sha256"] = "b" * 64
    elif change == "snapshot_owner":
        prior["recovered_1906_snapshot"]["trial_id"] = "foreign-trial"
    elif change == "snapshot_uri":
        prior["recovered_1906_snapshot"]["bundle_uri"] = "hdfs://foreign/result.tar.gz"
    else:
        prior["source_execution_reruns"] = 1
    prior_path.write_text(json.dumps(wrapper))
    request["prior_completion_progress"] = local_binding(prior_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):
        completion.prepare_supplements(path)
    assert commands == [] and not any(call[0] in ("restore", "persist") for call in runtime.calls)


def test_prepare_rejects_duplicate_source_index(tmp_path, monkeypatch):
    path, request, runtime, _, _ = supplemental_fixture(tmp_path, monkeypatch)
    request["accepted_sources"][-1] = request["accepted_sources"][0]
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="OBSERVATION_INDICES_DUPLICATE"):
        completion.prepare_supplements(path)
    assert not any(call[0] in ("restore", "persist") for call in runtime.calls)


@pytest.mark.parametrize("index", [1906, 843])
def test_prepare_rejects_restored_observation_report_conflict(tmp_path, monkeypatch, index):
    path, request, runtime, _, commands = supplemental_fixture(tmp_path, monkeypatch)
    original_restore = runtime.restore_snapshot
    original_sha = runtime.sha
    def restore(snapshot):
        result = original_restore(snapshot)
        if snapshot["indices"] == [index]:
            runtime.sha = lambda path:("c" * 64 if f"task_{index:04d}" in str(path) else original_sha(path))
        return result
    monkeypatch.setattr(runtime, "restore_snapshot", restore)
    with pytest.raises(ValueError, match="REPORT"):
        completion.prepare_supplements(path)
    assert commands == []
    assert not any(call[0] == "persist" and call[1] == [index] for call in runtime.calls)


def test_prepare_cli_selects_only_preparation_and_default_cli_keeps_run(monkeypatch):
    calls = []
    monkeypatch.setattr(completion, "prepare_supplements", lambda path:calls.append(("prepare",path)))
    monkeypatch.setattr(completion, "run", lambda path:calls.append(("run",path)))
    completion.main(["--request", "request.json", "--prepare-supplements"])
    completion.main(["--request", "request.json"])
    assert calls == [("prepare",Path("request.json")), ("run",Path("request.json"))]



def prepared_handoff_fixture(tmp_path,monkeypatch,overlap=()):
    path,request,runtime,loaded,commands=supplemental_fixture(tmp_path,monkeypatch)
    consumer_work=request["work"]
    request["work"]=str(tmp_path/"preparation_work");path.write_text(json.dumps(request))
    prepared=completion.prepare_supplements(path)
    prepared_path=Path(request["work"])/"supplemental_readout_preparation.json"
    request["prepared_supplements"]=local_binding(prepared_path)
    previous_path=tmp_path/"previous_execution.json"
    previous_path.write_text(json.dumps(dict(run_id=completion.PREVIOUS_COMPLETION_RUN,
        trial_id=completion.PREVIOUS_COMPLETION_TRIAL,platform_status="failed",terminal_confirmed=True,final_exit_code=1)))
    request["previous_execution"]=local_binding(previous_path)
    request["work"]=consumer_work
    path.write_text(json.dumps(request))
    runtime.calls.clear();loaded.clear();commands.clear()
    collected=(completion.ORIGINAL_IDS-completion.SUPPLEMENTAL_SOURCE_INDICES)|set(overlap)
    monkeypatch.setattr(runtime,"hdfs_json",lambda uri:progress_for(collected))
    return path,request,runtime,loaded,commands,prepared


@pytest.mark.parametrize("change",["running","unknown","success","unconfirmed","missing_status","missing_exit","zero_exit","bool_exit","owner"])
def test_prepared_consumer_requires_confirmed_previous_failure(tmp_path,monkeypatch,change):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch)
    previous_path=Path(request["previous_execution"]["path"]);previous=json.loads(previous_path.read_bytes())
    if change in ("running","unknown","success"):previous["platform_status"]=change
    elif change=="unconfirmed":previous["terminal_confirmed"]=False
    elif change=="missing_status":previous.pop("platform_status")
    elif change=="missing_exit":previous.pop("final_exit_code")
    elif change=="zero_exit":previous["final_exit_code"]=0
    elif change=="bool_exit":previous["final_exit_code"]=True
    else:previous["trial_id"]="foreign-trial"
    previous_path.write_text(json.dumps(previous));request["previous_execution"]=local_binding(previous_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match="CONFIRMED_FAILURE_REQUIRED"):completion.run(path)
    assert commands==[] and runtime.calls==[] and not Path(request["work"]).exists()


def test_confirmed_failure_handoff_uses_union_and_immutable_snapshots(tmp_path,monkeypatch):
    path,request,runtime,loaded,commands,prepared=prepared_handoff_fixture(tmp_path,monkeypatch)
    state=completion.run(path)
    restores=[call[1] for call in runtime.calls if call[0]=="restore"]
    assert restores[0]["run_id"]==completion.COLLECTOR_RUN
    assert len(restores)==12
    assert all(call[0]!="persist" for call in runtime.calls)
    assert not any("/"+completion.SOURCE_RUN+"/pfc/task_" in snapshot["bundle_uri"] for snapshot in restores)
    assert commands==[request["summary_command"],request["dynamics_command"]] and loaded==[0]
    assert state["collected"]==state["available_count"]==3000
    assert state["observed_collector_count"]==2989
    assert state["source_overlap_indices"]==[]
    assert state["reused_1906_snapshot"]==prepared["reused_1906_snapshot"]
    manifest=json.loads((Path(request["work"])/"complete_observations.json").read_bytes())
    assert len(manifest)==3000 and all(state[field] is False for field in completion.QUALIFICATION_FLAGS)


def test_original_supplement_overlap_is_verified_and_not_restored_twice(tmp_path,monkeypatch):
    overlap={843,917}
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch,overlap)
    state=completion.run(path)
    restores=[call[1] for call in runtime.calls if call[0]=="restore"]
    assert len(restores)==10
    assert state["source_overlap_indices"]==sorted(overlap)
    assert not overlap & {snapshot["indices"][0] for snapshot in restores[1:]}
    assert state["collected"]==3000 and state["observed_collector_count"]==2991
    assert len(commands)==2


@pytest.mark.parametrize("report_name",["development5ms/report.json","observation1ms/report.json"])
def test_original_overlap_report_conflict_blocks_consumers(tmp_path,monkeypatch,report_name):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch,{843})
    original_restore=runtime.restore_snapshot;original_sha=runtime.sha
    def restore(snapshot):
        result=original_restore(snapshot)
        if snapshot["run_id"]==completion.COLLECTOR_RUN:
            runtime.sha=lambda p:("c"*64 if "task_0843" in str(p) and str(p).endswith(report_name) else original_sha(p))
        return result
    monkeypatch.setattr(runtime,"restore_snapshot",restore)
    with pytest.raises(ValueError,match="ORIGINAL_SUPPLEMENTAL_REPORT_CONFLICT"):completion.run(path)
    assert commands==[] and not (Path(request["work"])/"complete_observations.json").exists()


def test_missing_original_id_keeps_union_closed_without_consumers(tmp_path,monkeypatch):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch)
    missing=completion.ORIGINAL_IDS-completion.SUPPLEMENTAL_SOURCE_INDICES-{2641}
    monkeypatch.setattr(runtime,"hdfs_json",lambda uri:progress_for(missing))
    def stop_wait(seconds):raise RuntimeError("test stops pending collection")
    monkeypatch.setattr(completion.time,"sleep",stop_wait)
    with pytest.raises(RuntimeError,match="pending collection"):completion.run(path)
    assert commands==[] and not any(call[0]=="restore" for call in runtime.calls)
    report=json.loads((Path(request["work"])/"completion_report.json").read_bytes())
    assert report["available_count"]==2999
    assert not (Path(request["work"])/"complete_observations.json").exists()


def test_preparation_report_paths_cannot_replace_explicit_staged_sources(tmp_path,monkeypatch):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch)
    report_path=Path(request["prepared_supplements"]["path"]);report=json.loads(report_path.read_bytes())
    for source in report["supplements"]:source["acceptance_binding"]["path"]="/not-staged/producer-path.json"
    report["accepted_1906_binding"]["path"]="/not-staged/accepted1906.json"
    report["prior_completion_progress"]["path"]="/not-staged/prior-progress.json"
    report_path.write_text(json.dumps(report));request["prepared_supplements"]=local_binding(report_path)
    path.write_text(json.dumps(request))
    assert completion.run(path)["collected"]==3000 and len(commands)==2


def test_missing_explicit_source_staging_is_rejected(tmp_path,monkeypatch):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch)
    request.pop("accepted_sources");path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match="STAGED_SUPPLEMENTAL_ACCEPTANCES_REQUIRED"):completion.run(path)
    assert commands==[] and not any(call[0]=="restore" for call in runtime.calls)


@pytest.mark.parametrize("change",["runtime","plan","consumer","source","acceptance_hash","single_id"])
def test_changed_prepared_report_bindings_are_rejected(tmp_path,monkeypatch,change):
    path,request,runtime,_,commands,_=prepared_handoff_fixture(tmp_path,monkeypatch)
    report_path=Path(request["prepared_supplements"]["path"]);report=json.loads(report_path.read_bytes())
    if change=="runtime":report["runtime_sha256"]="c"*64
    elif change=="plan":report["preparation_sha256"]="c"*64
    elif change=="consumer":report["consumer_sha256"]["provenance.py"]="c"*64
    elif change=="source":report["supplements"][0]["source_bundle_snapshot"]["bundle_sha256"]="c"*64
    elif change=="acceptance_hash":report["supplements"][0]["acceptance_binding"]["sha256"]="c"*64
    else:report["supplements"][0]["readout_snapshot"]["indices"]=[843,917]
    report_path.write_text(json.dumps(report));request["prepared_supplements"]=local_binding(report_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError):completion.run(path)
    assert commands==[] and not any(call[0]=="restore" for call in runtime.calls)


def source_readout_fixture(tmp_path, monkeypatch, index=2979, publication_kind="recovery"):
    monkeypatch.setattr(completion, "restore_native_source_readout", lambda source,runtime:
        runtime.restore_snapshot(source.get("readout_snapshot", source["source_bundle_snapshot"])))
    path, previous, old_runtime, loaded, commands = completion_fixture(tmp_path, monkeypatch)
    runtime = SupplementalMockRuntime(old_runtime.DOMAIN)
    monkeypatch.setattr(completion, "load_runtime", lambda binding,base:runtime)
    monkeypatch.setenv("ARNOLD_MONITOR_ROLE", "head")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    prepared = runtime.DOMAIN / "prepared/execution.json"
    plan = json.loads(prepared.read_bytes())
    rows = plan["rows"]
    rows[index]["config_sha256"] = hashlib.sha256(f"recipe-{index}".encode()).hexdigest()
    prepared.write_text(json.dumps(plan))
    previous["preparation_sha256"] = completion.sha(prepared)
    source_execution = dict(run_id=completion.SOURCE_RUN, trial_id=completion.SOURCE_TRIAL, xie_job=f"original-{index}")
    if publication_kind == "native_source":
        source_execution.update(run_id="new-source-run", trial_id="new-source-trial")
        publisher = {key:source_execution[key] for key in ("run_id", "trial_id")}
    else:
        publisher = dict(run_id="3604660682913335", trial_id="439239523")
    prefix = "hdfs://haruna/checkpoints/20261025/" + publisher["run_id"]
    original = "hdfs://haruna/checkpoints/20261025/" + source_execution["run_id"] + f"/pfc/task_{index:04d}"
    checkpoint = original if publication_kind == "native_source" else prefix + f"/pfc_publication/task_{index:04d}"
    receipt = dict(source_execution, status="COMPLETE", exit_code=0, persisted=True, index=index,
        checkpoint_path=checkpoint, result_bytes=86010908, result_sha256="b" * 64)
    if publication_kind == "recovery":
        receipt["publication_recovery"] = dict(publisher, checkpoint_path=prefix, source_checkpoint_path=original,
            source_bundle_uri=original + "/result.tar.gz", published_bundle_uri=checkpoint + "/result.tar.gz",
            allocated_gpus=0, source_simulations=0, scientific_admission_granted=False)
    validation = dict(backend="merlin", run_id="intake-run", trial_id="intake-trial",
                      allocated_gpus=0, numpy_version="2.4.2")
    accepted = dict(source_execution, status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=index,
        identity=rows[index]["identity"], config_sha256=rows[index]["config_sha256"],
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scientific_parameter_domain_qualified=False, scaling_qualified=False, remote_receipt=receipt,
        source_bundle_uri=original + "/result.tar.gz", source_bundle_sha256="b" * 64,
        receipt_uri=checkpoint + "/receipt.json", validation_execution=validation,
        observation_entry=completion.observation_entry(runtime, index))
    accepted_path = tmp_path / f"accepted_{index}.json"
    accepted_path.write_text(json.dumps(accepted))
    binding = local_binding(accepted_path, index=index)
    intake_checkpoint = "hdfs://haruna/checkpoints/20261025/intake-run/pfc_scientific_intake"
    intake = dict(schema="pfc_merlin_scientific_intake_v1", status="PASS", failures=[], execution=validation,
        checkpoint_path=intake_checkpoint, requested_task_indices=[index], new_accepted_episode_count=1,
        completed=[dict(index=index, acceptance_uri=intake_checkpoint + f"/task_{index:04d}_acceptance.json",
                        acceptance_sha256=binding["sha256"])], source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scientific_parameter_domain_qualified=False, scaling_qualified=False)
    intake_path = tmp_path / "scientific_intake.json"
    intake_path.write_text(json.dumps(intake))
    request = dict(schema=completion.SOURCE_READOUT_SCHEMA, index=index, runtime=previous["runtime"],
        accepted_source=binding, scientific_intake=local_binding(intake_path), source_execution=source_execution,
        publication_execution=publisher, preparation_sha256=previous["preparation_sha256"],
        consumer_sha256=previous["consumer_sha256"], work=previous["work"])
    if publication_kind != "recovery": request["publication_kind"] = publication_kind
    path.write_text(json.dumps(request))
    return path, request, runtime, loaded, commands

@pytest.mark.parametrize("index", [2979, 2717])
def test_single_source_publishes_existing_bound_readout_without_consumers(tmp_path, monkeypatch, index):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch, index)
    before = {request[key]["path"]:Path(request[key]["path"]).read_bytes()
              for key in ("accepted_source", "scientific_intake")}
    state = completion.publish_source_readout(path)
    restores = [call[1] for call in runtime.calls if call[0] == "restore"]
    assert len(restores) == 1 and restores[0]["indices"] == [index]
    assert restores[0]["bundle_uri"].endswith("/" + completion.SOURCE_RUN + f"/pfc/task_{index:04d}/result.tar.gz")
    assert [call[1] for call in runtime.calls if call[0] == "persist"] == [[index]]
    assert not any(call[0] == "observe" for call in runtime.calls) and loaded == commands == []
    assert state["status"] == "SOURCE_READOUT_READY" and state["published_episodes"] == 1
    assert state["published_indices"] == state["readout_snapshot"]["indices"] == [index]
    assert state["source"]["source_execution"] == request["source_execution"]
    assert state["publication_execution"] == request["publication_execution"]
    assert state["readout_snapshot"]["run_id"] == os.environ["RH2_JOB_RUN_ID"]
    assert len({state["source"]["source_execution"]["run_id"], state["publication_execution"]["run_id"],
                state["execution"]["run_id"]}) == 3
    assert state["required_episodes"] == 3000 and state["closure_verified"] is False
    assert state["consumer_start_authorized"] is False and all(state[field] is False for field in completion.QUALIFICATION_FLAGS)
    assert all(state[field] == 0 for field in ("source_simulations", "production_episodes", "source_execution_reruns",
        "readout_execution_reruns", "summary_executions", "dynamics_executions"))
    assert all(Path(filename).read_bytes() == data for filename,data in before.items())
    assert runtime.calls[-1][0] == "publish_results" and runtime.calls[-1][2] == 0


@pytest.mark.parametrize("change", ["request_index", "bool_index", "source_identity", "source_execution",
    "publisher_run", "publisher_trial", "publisher_alias", "publisher_receipt_path", "publisher_bundle_path", "intake_status",
    "intake_record_hash", "intake_execution", "intake_requested", "intake_scope", "missing_intake"])
def test_single_source_rejects_changed_identity_or_intake_before_restore(tmp_path, monkeypatch, change):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch)
    accepted_path = Path(request["accepted_source"]["path"])
    accepted = json.loads(accepted_path.read_bytes())
    intake_path = Path(request["scientific_intake"]["path"])
    intake = json.loads(intake_path.read_bytes())
    if change == "request_index": request["index"] = 2978
    elif change == "bool_index": request["index"] = True
    elif change == "source_identity": accepted["identity"]["condition_index"] += 1
    elif change == "source_execution": request["source_execution"]["xie_job"] = "foreign-job"
    elif change == "publisher_run": accepted["remote_receipt"]["publication_recovery"]["run_id"] = "foreign-run"
    elif change == "publisher_trial": accepted["remote_receipt"]["publication_recovery"]["trial_id"] = "foreign-trial"
    elif change == "publisher_alias": request["publication_execution"]["run_id"] = completion.SOURCE_RUN
    elif change == "publisher_receipt_path": accepted["receipt_uri"] = "hdfs://foreign/receipt.json"
    elif change == "publisher_bundle_path": accepted["remote_receipt"]["publication_recovery"]["published_bundle_uri"] = accepted["source_bundle_uri"]
    elif change == "intake_status": intake["status"] = "FAILED"
    elif change == "intake_record_hash": intake["completed"][0]["acceptance_sha256"] = "c" * 64
    elif change == "intake_execution": intake["execution"]["trial_id"] = "foreign-trial"
    elif change == "intake_requested": intake["requested_task_indices"] = [2978]
    elif change == "intake_scope": intake["source_qualification_granted"] = True
    else: request.pop("scientific_intake")
    accepted_path.write_text(json.dumps(accepted))
    request["accepted_source"].update(local_binding(accepted_path))
    intake_path.write_text(json.dumps(intake))
    if "scientific_intake" in request: request["scientific_intake"] = local_binding(intake_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError): completion.publish_source_readout(path)
    assert commands == loaded == [] and not any(call[0] in ("restore", "persist") for call in runtime.calls)


@pytest.mark.parametrize("report_name", ["development5ms/report.json", "observation1ms/report.json"])
def test_single_source_rejects_each_restored_report_conflict_before_persist(tmp_path, monkeypatch, report_name):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch)
    original_restore, original_sha = runtime.restore_snapshot, runtime.sha
    def restore(snapshot):
        restored = original_restore(snapshot)
        runtime.sha = lambda p:("c" * 64 if str(p).endswith(report_name) else original_sha(p))
        return restored
    monkeypatch.setattr(runtime, "restore_snapshot", restore)
    with pytest.raises(ValueError, match="RESTORED_REPORT_CHANGED"): completion.publish_source_readout(path)
    assert commands == loaded == [] and not any(call[0] == "persist" for call in runtime.calls)
    state = json.loads((Path(request["work"]) / "source_readout_publication.json").read_bytes())
    assert state["status"] == "FAILED_SOURCE_READOUT_PUBLICATION" and state["published_episodes"] == 0


@pytest.mark.parametrize("change", ["restore_id", "snapshot_ids", "snapshot_trial"])
def test_single_source_enforces_single_id_restore_and_publication(tmp_path, monkeypatch, change):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch)
    if change == "restore_id":
        monkeypatch.setattr(runtime, "restore_snapshot", lambda snapshot:[2978])
    else:
        original_persist = runtime.persist_readouts
        def persist(indices, plan, entry):
            snapshot = original_persist(indices, plan, entry)
            if change == "snapshot_ids": snapshot["indices"] = [2979, 2978]
            else: snapshot["trial_id"] = "foreign-trial"
            return snapshot
        monkeypatch.setattr(runtime, "persist_readouts", persist)
    with pytest.raises(ValueError): completion.publish_source_readout(path)
    assert commands == loaded == []
    state = json.loads((Path(request["work"]) / "source_readout_publication.json").read_bytes())
    assert state["status"] == "FAILED_SOURCE_READOUT_PUBLICATION" and state["published_episodes"] == 0


def test_single_source_preserves_complete_plan_and_legacy_fixed_set(tmp_path, monkeypatch):
    path, request, runtime, _, commands = source_readout_fixture(tmp_path, monkeypatch)
    prepared = runtime.DOMAIN / "prepared/execution.json"
    plan = json.loads(prepared.read_bytes())
    with pytest.raises(ValueError, match="SUPPLEMENTAL_SOURCE_INDEX_INVALID"):
        completion.supplemental_source(request["accepted_source"], path.parent, plan, runtime)
    plan["rows"].pop()
    prepared.write_text(json.dumps(plan))
    request["preparation_sha256"] = completion.sha(prepared)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="COMPLETE3000_PLAN_REQUIRED"): completion.publish_source_readout(path)
    assert commands == [] and not any(call[0] in ("restore", "persist") for call in runtime.calls)


@pytest.mark.parametrize("field,value", [("ARNOLD_APPLIED_GPU_NUM", "1"), ("ARNOLD_MONITOR_ROLE", "worker"),
                                        ("CUDA_VISIBLE_DEVICES", "0")])
def test_single_source_requires_cpu_head_allocation(tmp_path, monkeypatch, field, value):
    path, request, runtime, _, _ = source_readout_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match="MERLIN_CPU_SOURCE_READOUT_REQUIRED"): completion.publish_source_readout(path)
    assert runtime.calls == [] and not Path(request["work"]).exists()


def test_single_source_cli_is_exclusive_and_does_not_start_completion(monkeypatch):
    calls = []
    monkeypatch.setattr(completion, "publish_source_readout", lambda path:calls.append(("publish",path)))
    monkeypatch.setattr(completion, "prepare_supplements", lambda path:calls.append(("prepare",path)))
    monkeypatch.setattr(completion, "run", lambda path:calls.append(("run",path)))
    completion.main(["--request", "request.json", "--publish-source-readout"])
    assert calls == [("publish",Path("request.json"))]
    with pytest.raises(SystemExit):
        completion.main(["--request", "request.json", "--publish-source-readout", "--prepare-supplements"])
    assert calls == [("publish",Path("request.json"))]


@pytest.mark.parametrize("index", [2641, 2717])
def test_new_source_native_publication_uses_real_bound_identity(tmp_path, monkeypatch, index):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch, index, "native_source")
    before = {request[key]["path"]:Path(request[key]["path"]).read_bytes()
              for key in ("accepted_source", "scientific_intake")}
    state = completion.publish_source_readout(path)
    assert state["status"] == "SOURCE_READOUT_READY" and state["publication_kind"] == "native_source"
    assert state["source"]["source_execution"] == request["source_execution"]
    assert state["publication_execution"] == {key:request["source_execution"][key] for key in ("run_id", "trial_id")}
    assert state["source"]["source_execution"]["run_id"] != completion.SOURCE_RUN
    restores = [call[1] for call in runtime.calls if call[0] == "restore"]
    assert len(restores) == 1 and restores[0]["indices"] == [index]
    assert restores[0]["run_id"] == "new-source-run" and restores[0]["trial_id"] == "new-source-trial"
    assert "/new-source-run/pfc/" in restores[0]["bundle_uri"]
    assert [call[1] for call in runtime.calls if call[0] == "persist"] == [[index]]
    assert state["readout_snapshot"]["run_id"] not in {completion.SOURCE_RUN, "new-source-run"}
    assert state["published_indices"] == [index] and state["published_episodes"] == 1
    assert commands == loaded == [] and not any(call[0] == "observe" for call in runtime.calls)
    assert all(state[field] == 0 for field in ("source_simulations", "production_episodes",
        "source_execution_reruns", "readout_execution_reruns", "summary_executions", "dynamics_executions"))
    assert state["closure_verified"] is state["consumer_start_authorized"] is False
    assert all(state[field] is False for field in completion.QUALIFICATION_FLAGS)
    assert all(Path(filename).read_bytes() == data for filename,data in before.items())


@pytest.mark.parametrize("change", ["request_old_source", "request_trial", "acceptance_run", "receipt_run",
    "receipt_trial", "source_bundle", "checkpoint", "publisher", "recovery_metadata", "config", "validation_gpu", "missing_mode"])
def test_native_publication_rejects_source_receipt_and_path_conflicts(tmp_path, monkeypatch, change):
    path, request, runtime, loaded, commands = source_readout_fixture(tmp_path, monkeypatch, 2641, "native_source")
    accepted_path = Path(request["accepted_source"]["path"])
    accepted = json.loads(accepted_path.read_bytes())
    if change == "request_old_source": request["source_execution"]["run_id"] = completion.SOURCE_RUN
    elif change == "request_trial": request["source_execution"]["trial_id"] = "foreign-trial"
    elif change == "acceptance_run": accepted["run_id"] = completion.SOURCE_RUN
    elif change == "receipt_run": accepted["remote_receipt"]["run_id"] = completion.SOURCE_RUN
    elif change == "receipt_trial": accepted["remote_receipt"]["trial_id"] = "foreign-trial"
    elif change == "source_bundle": accepted["source_bundle_uri"] = "hdfs://foreign/result.tar.gz"
    elif change == "checkpoint": accepted["remote_receipt"]["checkpoint_path"] = "hdfs://foreign/pfc/task_2641"
    elif change == "publisher": request["publication_execution"]["run_id"] = "foreign-publisher"
    elif change == "recovery_metadata": accepted["remote_receipt"]["publication_recovery"] = {}
    elif change == "config": accepted["config_sha256"] = "c" * 64
    elif change == "validation_gpu": accepted["validation_execution"]["allocated_gpus"] = 1
    else: request.pop("publication_kind")
    accepted_path.write_text(json.dumps(accepted))
    request["accepted_source"].update(local_binding(accepted_path))
    intake_path = Path(request["scientific_intake"]["path"])
    intake = json.loads(intake_path.read_bytes())
    intake["completed"][0]["acceptance_sha256"] = request["accepted_source"]["sha256"]
    intake_path.write_text(json.dumps(intake))
    request["scientific_intake"] = local_binding(intake_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError): completion.publish_source_readout(path)
    assert commands == loaded == [] and not any(call[0] in ("restore", "persist") for call in runtime.calls)


@pytest.mark.parametrize("publication_kind", ["recovery", "native_source"])
def test_recovery_receipt_cannot_be_relabelled_as_native(tmp_path, monkeypatch, publication_kind):
    path, request, runtime, _, _ = source_readout_fixture(tmp_path, monkeypatch)
    request["publication_kind"] = publication_kind
    path.write_text(json.dumps(request))
    if publication_kind == "recovery":
        state = completion.publish_source_readout(path)
        assert state["status"] == "SOURCE_READOUT_READY" and state["publication_kind"] == "recovery"
    else:
        with pytest.raises(ValueError, match="NATIVE_SOURCE_PUBLISHER_CHANGED"): completion.publish_source_readout(path)
        assert not any(call[0] in ("restore", "persist") for call in runtime.calls)


def test_legacy_supplemental_source_keeps_original_execution_gate(tmp_path, monkeypatch):
    path, request, runtime, _, _ = source_readout_fixture(tmp_path, monkeypatch, 843, "native_source")
    plan = json.loads((runtime.DOMAIN / "prepared/execution.json").read_bytes())
    with pytest.raises(ValueError, match="SUPPLEMENTAL_SOURCE_ACCEPTANCE_CHANGED"):
        completion.supplemental_source(request["accepted_source"], path.parent, plan, runtime)
    assert not any(call[0] in ("restore", "persist") for call in runtime.calls)



def published_handoff_fixture(tmp_path, monkeypatch, overlap=()):
    path, request, runtime, loaded, commands, prepared = prepared_handoff_fixture(tmp_path, monkeypatch)
    indices = (2641, 2717, 2979)
    preparation = runtime.DOMAIN / "prepared/execution.json"
    plan = json.loads(preparation.read_bytes())
    for index in indices:
        plan["rows"][index]["config_sha256"] = hashlib.sha256(f"recipe-{index}".encode()).hexdigest()
    preparation.write_text(json.dumps(plan))
    request["preparation_sha256"] = completion.sha(preparation)
    prior_path = Path(request["prior_completion_progress"]["path"])
    prior = json.loads(prior_path.read_bytes())
    prior["progress"]["preparation_sha256"] = request["preparation_sha256"]
    prior_path.write_text(json.dumps(prior))
    request["prior_completion_progress"] = local_binding(prior_path)
    prepared_path = Path(request["prepared_supplements"]["path"])
    prepared = json.loads(prepared_path.read_bytes())
    prepared["preparation_sha256"] = request["preparation_sha256"]
    prepared["prior_completion_progress"].update(request["prior_completion_progress"])
    prepared_path.write_text(json.dumps(prepared))
    request["prepared_supplements"] = local_binding(prepared_path)
    publications = []
    for index in indices:
        directory = tmp_path / f"single_{index}"
        single_path, single, _, _, _ = source_readout_fixture(directory, monkeypatch, index,
            "recovery" if index == 2979 else "native_source")
        accepted_path = Path(single["accepted_source"]["path"])
        accepted = json.loads(accepted_path.read_bytes())
        if index != 2979:
            source = dict(run_id="ee70a852bcea8ed6", trial_id="439275320",
                          xie_job="643653_641" if index == 2641 else "643653_717")
            checkpoint = f"hdfs://haruna/checkpoints/20261025/ee70a852bcea8ed6/pfc/task_{index:04d}"
            accepted.update(source)
            accepted["remote_receipt"].update(source, checkpoint_path=checkpoint)
            accepted.update(receipt_uri=checkpoint+"/receipt.json", source_bundle_uri=checkpoint+"/result.tar.gz")
            single["source_execution"] = source
            single["publication_execution"] = {key:source[key] for key in ("run_id", "trial_id")}
        accepted["observation_entry"] = completion.observation_entry(runtime, index)
        accepted_path.write_text(json.dumps(accepted))
        single["accepted_source"].update(local_binding(accepted_path))
        intake_path = Path(single["scientific_intake"]["path"])
        intake = json.loads(intake_path.read_bytes())
        intake["completed"][0]["acceptance_sha256"] = single["accepted_source"]["sha256"]
        intake_path.write_text(json.dumps(intake))
        single.update(runtime=request["runtime"], preparation_sha256=request["preparation_sha256"],
                      consumer_sha256=request["consumer_sha256"], scientific_intake=local_binding(intake_path),
                      work=str(directory / "publication"))
        single_path.write_text(json.dumps(single))
        monkeypatch.setattr(completion, "load_runtime", lambda binding,base:runtime)
        monkeypatch.setenv("RH2_JOB_RUN_ID", f"publication-{index}")
        monkeypatch.setenv("ARNOLD_TRIAL_ID", f"publication-trial-{index}")
        completion.publish_source_readout(single_path)
        publication_path = Path(single["work"]) / "source_readout_publication.json"
        publications.append(dict(source_readout_request=local_binding(single_path),
                                 source_readout_publication=local_binding(publication_path)))
    request["source_readout_publications"] = publications
    path.write_text(json.dumps(request))
    monkeypatch.setenv("RH2_JOB_RUN_ID", "new-completion-run")
    monkeypatch.setenv("ARNOLD_TRIAL_ID", "new-completion-trial")
    monkeypatch.setattr(completion, "load_runtime", lambda binding,base:runtime)
    monkeypatch.setattr(completion.summary, "load_observation", lambda entry,row:loaded.append(row["task_index"]))
    monkeypatch.setattr(completion.subprocess, "run", lambda argv,**kwargs:(commands.append(argv) or SimpleNamespace(returncode=0)))
    collected = (completion.ORIGINAL_IDS - completion.SUPPLEMENTAL_SOURCE_INDICES - set(indices)) | set(overlap)
    monkeypatch.setattr(runtime, "hdfs_json", lambda uri:progress_for(collected))
    runtime.calls.clear(); loaded.clear(); commands.clear()
    return path, request, runtime, loaded, commands, plan


def test_published_handoff_binds_two_native_and_one_recovery_source(tmp_path, monkeypatch):
    path, request, runtime, _, _, plan = published_handoff_fixture(tmp_path, monkeypatch)
    sources = completion.published_source_handoff(request, path.parent, plan, runtime)
    assert set(sources) == {2641, 2717, 2979}
    for index in (2641, 2717):
        assert sources[index]["source_execution"]["run_id"] == "ee70a852bcea8ed6"
        assert sources[index]["source_execution"]["trial_id"] == "439275320"
    assert sources[2979]["source_execution"]["run_id"] == completion.SOURCE_RUN
    assert sources[2979]["publication_recovery"]["run_id"] == "3604660682913335"
    assert all(source["readout_snapshot"]["indices"] == [index] for index,source in sources.items())
    assert runtime.calls == []


@pytest.mark.parametrize("change", ["request_binding", "report_binding", "plan", "request_source", "report_status",
    "reported_source", "reported_intake", "intake_status", "source_config", "snapshot", "publisher", "scope"])
def test_published_handoff_rejects_changed_request_report_and_source_proof(tmp_path, monkeypatch, change):
    path, request, runtime, _, _, plan = published_handoff_fixture(tmp_path, monkeypatch)
    binding = request["source_readout_publications"][0]
    single_path = Path(binding["source_readout_request"]["path"])
    report_path = Path(binding["source_readout_publication"]["path"])
    single = json.loads(single_path.read_bytes()); published = json.loads(report_path.read_bytes())
    if change == "request_binding": binding["source_readout_request"]["sha256"] = "c" * 64
    elif change == "report_binding": binding["source_readout_publication"]["sha256"] = "c" * 64
    elif change == "plan": single["preparation_sha256"] = "c" * 64
    elif change == "request_source": single["source_execution"]["trial_id"] = "foreign-trial"
    elif change == "report_status": published["status"] = "FAILED_SOURCE_READOUT_PUBLICATION"
    elif change == "reported_source": published["source"]["source_execution"]["run_id"] = "foreign-run"
    elif change == "reported_intake": published["scientific_intake_binding"]["sha256"] = "c" * 64
    elif change == "intake_status":
        intake_path = Path(single["scientific_intake"]["path"])
        intake = json.loads(intake_path.read_bytes()); intake["status"] = "FAILED"
        intake_path.write_text(json.dumps(intake)); single["scientific_intake"] = local_binding(intake_path)
    elif change == "source_config":
        accepted_path = Path(single["accepted_source"]["path"])
        accepted = json.loads(accepted_path.read_bytes()); accepted["config_sha256"] = "c" * 64
        accepted_path.write_text(json.dumps(accepted)); single["accepted_source"].update(local_binding(accepted_path))
        intake_path = Path(single["scientific_intake"]["path"])
        intake = json.loads(intake_path.read_bytes())
        intake["completed"][0]["acceptance_sha256"] = single["accepted_source"]["sha256"]
        intake_path.write_text(json.dumps(intake)); single["scientific_intake"] = local_binding(intake_path)
    elif change == "snapshot": published["readout_snapshot"]["trial_id"] = "foreign-trial"
    elif change == "publisher": published["publication_execution"]["run_id"] = "foreign-run"
    else: published["scientific_claim_qualified"] = True
    if change not in ("request_binding", "report_binding"):
        single_path.write_text(json.dumps(single)); binding["source_readout_request"] = local_binding(single_path)
        report_path.write_text(json.dumps(published)); binding["source_readout_publication"] = local_binding(report_path)
    with pytest.raises(ValueError): completion.published_source_handoff(request, path.parent, plan, runtime)
    assert runtime.calls == []


def test_published_handoff_rejects_duplicate_index(tmp_path, monkeypatch):
    path, request, runtime, _, _, plan = published_handoff_fixture(tmp_path, monkeypatch)
    request["source_readout_publications"].append(request["source_readout_publications"][0])
    with pytest.raises(ValueError, match="DUPLICATE_OR_RESERVED_INDEX"):
        completion.published_source_handoff(request, path.parent, plan, runtime)
    assert runtime.calls == []


@pytest.mark.parametrize("overlap", [(), (2641, 2979)])
def test_completion_reuses_published_snapshots_for_exact3000_without_duplicate_restore(tmp_path, monkeypatch, overlap):
    path, request, runtime, loaded, commands, _ = published_handoff_fixture(tmp_path, monkeypatch, overlap)
    state = completion.run(path)
    restores = [call[1] for call in runtime.calls if call[0] == "restore"]
    restored_single = [snapshot["indices"][0] for snapshot in restores if len(snapshot["indices"]) == 1
                       and snapshot["run_id"].startswith("publication-")]
    assert restored_single == sorted({2641, 2717, 2979} - set(overlap))
    assert state["restored_source_readout_publication_indices"] == restored_single
    assert set(overlap) <= set(state["source_overlap_indices"])
    assert state["collected"] == state["available_count"] == 3000
    assert not any(call[0] == "persist" for call in runtime.calls)
    assert loaded == [0] and commands == [request["summary_command"], request["dynamics_command"]]
    manifest = json.loads((Path(request["work"]) / "complete_observations.json").read_bytes())
    assert len(manifest) == 3000 and all(state[field] is False for field in completion.QUALIFICATION_FLAGS)
    assert state["source_execution_reruns"] == state["readout_execution_reruns"] == 0


@pytest.mark.parametrize("report_name", ["development5ms/report.json", "observation1ms/report.json"])
def test_published_original_overlap_checks_both_reports_before_consumers(tmp_path, monkeypatch, report_name):
    path, request, runtime, _, commands, _ = published_handoff_fixture(tmp_path, monkeypatch, (2641,))
    original_restore, original_sha = runtime.restore_snapshot, runtime.sha
    def restore(snapshot):
        restored = original_restore(snapshot)
        if snapshot["run_id"] == completion.COLLECTOR_RUN:
            runtime.sha = lambda p:("c" * 64 if "task_2641" in str(p) and str(p).endswith(report_name) else original_sha(p))
        return restored
    monkeypatch.setattr(runtime, "restore_snapshot", restore)
    with pytest.raises(ValueError, match="ORIGINAL_PUBLISHED_SOURCE_REPORT_CONFLICT"):
        completion.run(path)
    assert commands == [] and not (Path(request["work"]) / "complete_observations.json").exists()


@pytest.mark.parametrize("change", ["running", "unknown", "zero_exit"])
def test_publication_field_keeps_failed_nonzero_predecessor_gate(tmp_path, monkeypatch, change):
    path, request, runtime, _, commands, _ = published_handoff_fixture(tmp_path, monkeypatch)
    previous_path = Path(request["previous_execution"]["path"])
    previous = json.loads(previous_path.read_bytes())
    if change == "zero_exit": previous["final_exit_code"] = 0
    else: previous["platform_status"] = change
    previous_path.write_text(json.dumps(previous)); request["previous_execution"] = local_binding(previous_path)
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="CONFIRMED_FAILURE_REQUIRED"): completion.run(path)
    assert commands == [] and runtime.calls == [] and not Path(request["work"]).exists()



class LegacyNativeArchiveRuntime:
    def __init__(self, domain, observation_names):
        self.DOMAIN = domain
        self.observation_names = set(observation_names)
        self.unpack_calls = []

    def unpack_readouts(self, bundle, digest):
        self.unpack_calls.append((str(bundle), digest))
        with tarfile.open(bundle, "r:gz") as archive:
            for member in archive:
                # Reproduce the old selector: observations survive, metadata at
                # the isolated native source root is omitted.
                if member.name in self.observation_names:
                    target = Path("/" + member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.extractfile(member).read()
                    if target.exists():
                        assert target.read_bytes() == data
                    else:
                        target.write_bytes(data)


def native_report_archive_fixture(tmp_path, monkeypatch, published=False, change=None, existing=False):
    index = 2641
    domain = tmp_path / "domain"
    source_archive = domain / "source_publication_loss_recovery_20261025/task_2641/run_merlin_439275320_pfc_2641"
    observation = domain / "observations/task_2641"
    source_names = ("delivery.json", "source_run/report.json", "qualification/delivery.json",
                    "qualification/comparison/qualification.json")
    reports = {source_archive / name:json.dumps(dict(source_report=name, value="preserved")).encode()
               for name in source_names}
    development = json.dumps(dict(source_archive=str(source_archive if change != "development_root" else tmp_path/"foreign"))).encode()
    observations = {
        observation / "development5ms/report.json":development,
        observation / "development5ms/phase_responses.csv":b"phase,value\n0,1\n",
        observation / "development5ms/temporal_responses.csv":b"tick,value\n0,1\n",
        observation / "development5ms/unit_population_statistics.npz":b"small observation array bytes",
        observation / "observation1ms/report.json":b'{"observation_bin_ms":1}',
        observation / "observation1ms/observation.npz":b"small full-unit observation bytes",
    }
    payloads = {str(path).lstrip("/"):data for path,data in {**reports, **observations}.items()}
    manifest = {name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payloads.items()}
    delivery_name = str(source_archive / "delivery.json").lstrip("/")
    if change == "manifest_hash": manifest[delivery_name]["sha256"] = "c" * 64
    elif change == "manifest_set": manifest.pop(delivery_name)
    elif change == "missing_report": payloads.pop(delivery_name); manifest.pop(delivery_name)
    elif change == "unsafe_member":
        payloads["../foreign.json"] = b"unsafe"
        manifest["../foreign.json"] = dict(bytes=6,sha256=hashlib.sha256(b"unsafe").hexdigest())
    archive_path = tmp_path / "native_input.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name,data in {**payloads,"merlin_return_manifest.json":json.dumps(manifest).encode()}.items():
            member = tarfile.TarInfo(name); member.size = len(data)
            archive.addfile(member,io.BytesIO(data))
    raw = archive_path.read_bytes()
    source_uri = "hdfs://haruna/checkpoints/ee70a852bcea8ed6/pfc/task_2641/result.tar.gz"
    readout_uri = "hdfs://haruna/checkpoints/publication-run/pfc_downstream/readout_snapshots/readouts_2641_2641/result.tar.gz"
    selected_uri = readout_uri if published else source_uri
    selected = dict(indices=[index],bundle_uri=selected_uri,bundle_bytes=len(raw),bundle_sha256=hashlib.sha256(raw).hexdigest())
    if change == "bytes": selected["bundle_bytes"] += 1
    elif change == "sha": selected["bundle_sha256"] = "c" * 64
    accepted = dict(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE",index=index,
        source_archive=str(source_archive if change != "accepted_root" else tmp_path/"outside_domain"),
        observation_entry=dict(path=str(observation),development_report_sha256=hashlib.sha256(development).hexdigest(),
                               observation_report_sha256=hashlib.sha256(observations[observation/"observation1ms/report.json"]).hexdigest()))
    accepted_path = tmp_path / "accepted_native.json"; accepted_path.write_text(json.dumps(accepted))
    source = dict(index=index,acceptance_binding=local_binding(accepted_path),source_bundle_snapshot=selected)
    if published:
        source["readout_snapshot"] = selected
        source["source_bundle_snapshot"] = dict(indices=[index],bundle_uri=source_uri,bundle_bytes=999999,bundle_sha256="f"*64)
    runtime = LegacyNativeArchiveRuntime(domain, [str(path).lstrip("/") for path in observations])
    downloads = []
    def download(argv, **kwargs):
        assert argv[:3] == ["hdfs", "dfs", "-get"] and argv[3] == selected_uri
        downloads.append(list(argv))
        Path(argv[4]).write_bytes(raw)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(completion.subprocess,"run",download)
    if existing:
        for path,data in reports.items():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    if change == "existing_conflict":
        target = source_archive / "delivery.json"
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b"protected existing report")
    return source,runtime,reports,observations,downloads


@pytest.mark.parametrize("published",[False,True])
@pytest.mark.parametrize("existing",[False,True])
def test_native_helper_restores_four_reports_from_one_verified_input(tmp_path,monkeypatch,published,existing):
    source,runtime,reports,observations,downloads=native_report_archive_fixture(tmp_path,monkeypatch,published,existing=existing)
    assert completion.restore_native_source_readout(source,runtime)==[2641]
    assert len(downloads)==len(runtime.unpack_calls)==1
    selected=source.get("readout_snapshot",source["source_bundle_snapshot"])
    assert downloads[0][3]==selected["bundle_uri"] and runtime.unpack_calls[0][1]==selected["bundle_sha256"]
    assert all(path.read_bytes()==data for path,data in reports.items())
    assert all(path.read_bytes()==data for path,data in observations.items())


@pytest.mark.parametrize("published",[False,True])
@pytest.mark.parametrize("change",["bytes","sha","manifest_hash","manifest_set","missing_report",
                                   "unsafe_member","accepted_root","development_root","existing_conflict"])
def test_native_helper_rejects_transport_manifest_root_and_existing_conflicts(tmp_path,monkeypatch,published,change):
    source,runtime,reports,_,downloads=native_report_archive_fixture(tmp_path,monkeypatch,published,change)
    with pytest.raises(ValueError):completion.restore_native_source_readout(source,runtime)
    assert len(downloads)==(0 if change=="accepted_root" else 1)
    if change in ("bytes","sha","accepted_root"):
        assert runtime.unpack_calls==[]
    if change=="existing_conflict":
        target=next(path for path in reports if path.name=="delivery.json")
        assert target.read_bytes()==b"protected existing report"



def authorized_stop_document(run_id, trial_id, exit_code=None):
    return dict(run_id=run_id, trial_id=trial_id, platform_status="stopped", terminal_confirmed=True,
        final_exit_code=exit_code,
        user_authorization=dict(source="user_explicit_reply", date="2026-10-26",
            allowed_run_ids=["a5e61c466d9c82dc", "28bf3f86a34814e0"], actions=["safe_stop", "unfinished_takeover"]),
        stop_operation=dict(run_id=run_id, trial_id=trial_id, action="stop", accepted=True, response_base_code=0,
                            observed_utc="2026-10-26T06:37:00+00:00"),
        terminal_observation=dict(run_id=run_id, trial_id=trial_id, platform_status="stopped", terminal_confirmed=True,
                                  observed_utc="2026-10-26T06:38:01.076810+00:00"))


@pytest.mark.parametrize("exit_code", [None, 0, 143])
def test_explicit_authorized_stop_preserves_actual_exit_and_a5_identity(tmp_path, exit_code):
    document = authorized_stop_document(completion.PREVIOUS_COMPLETION_RUN, completion.PREVIOUS_COMPLETION_TRIAL, exit_code)
    path = tmp_path / "stopped.json"; path.write_text(json.dumps(document))
    result = completion.validate_previous_execution(local_binding(path), tmp_path)
    assert result["platform_status"] == "stopped" and result["termination_kind"] == "user_authorized_stop"
    assert result["final_exit_code"] is None if exit_code is None else result["final_exit_code"] == exit_code
    assert result["user_authorization"] == document["user_authorization"]


@pytest.mark.parametrize("change", ["running", "unknown", "unconfirmed", "owner", "trial", "no_authorization",
    "goal_activation", "date", "scope", "actions", "accepted", "response_bool", "response_error", "operation_owner",
    "operation_trial", "terminal_owner", "terminal_trial", "terminal_live", "terminal_unconfirmed", "time", "exit_bool"])
def test_authorized_stop_rejects_missing_or_mismatched_operational_proof(tmp_path, change):
    document = authorized_stop_document(completion.PREVIOUS_COMPLETION_RUN, completion.PREVIOUS_COMPLETION_TRIAL)
    if change in ("running", "unknown"): document["platform_status"] = change
    elif change == "unconfirmed": document["terminal_confirmed"] = False
    elif change == "owner": document["run_id"] = "other-run"
    elif change == "trial": document["trial_id"] = "other-trial"
    elif change == "no_authorization": document.pop("user_authorization")
    elif change == "goal_activation": document["user_authorization"]["source"] = "goal_activation"
    elif change == "date": document["user_authorization"]["date"] = "2026-10-25"
    elif change == "scope": document["user_authorization"]["allowed_run_ids"][1] = "other-run"
    elif change == "actions": document["user_authorization"]["actions"] = ["safe_stop"]
    elif change == "accepted": document["stop_operation"]["accepted"] = False
    elif change == "response_bool": document["stop_operation"]["response_base_code"] = False
    elif change == "response_error": document["stop_operation"]["response_base_code"] = 1
    elif change == "operation_owner": document["stop_operation"]["run_id"] = "other-run"
    elif change == "operation_trial": document["stop_operation"]["trial_id"] = "other-trial"
    elif change == "terminal_owner": document["terminal_observation"]["run_id"] = "other-run"
    elif change == "terminal_trial": document["terminal_observation"]["trial_id"] = "other-trial"
    elif change == "terminal_live": document["terminal_observation"]["platform_status"] = "running"
    elif change == "terminal_unconfirmed": document["terminal_observation"]["terminal_confirmed"] = False
    elif change == "time": document["terminal_observation"]["observed_utc"] = "not-a-time"
    else: document["final_exit_code"] = True
    path = tmp_path / "stopped.json"; path.write_text(json.dumps(document))
    with pytest.raises(ValueError): completion.validate_previous_execution(local_binding(path), tmp_path)


def test_authorized_stop_allows_exact3000_handoff_without_changing_saved_progress(tmp_path, monkeypatch):
    path, request, runtime, loaded, commands, _ = prepared_handoff_fixture(tmp_path, monkeypatch)
    previous_path = Path(request["previous_execution"]["path"])
    previous_path.write_text(json.dumps(authorized_stop_document(completion.PREVIOUS_COMPLETION_RUN, completion.PREVIOUS_COMPLETION_TRIAL)))
    request["previous_execution"] = local_binding(previous_path); path.write_text(json.dumps(request))
    prior_bytes = Path(request["prior_completion_progress"]["path"]).read_bytes()
    state = completion.run(path)
    assert state["collected"] == 3000 and state["previous_execution"]["platform_status"] == "stopped"
    assert state["previous_execution"]["final_exit_code"] is None
    assert Path(request["prior_completion_progress"]["path"]).read_bytes() == prior_bytes
    assert loaded == [0] and commands == [request["summary_command"], request["dynamics_command"]]
    assert not any(call[0] == "persist" for call in runtime.calls)

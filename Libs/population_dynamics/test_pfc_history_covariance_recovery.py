"""Recovery boundaries and cache transport; no covariance kernel is run."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from . import pfc_history_covariance_recovery as recovery


def plan_fixture():
    old = sorted(recovery.intake.REQUIRED_GROUPS - recovery.SUPPLEMENTAL_KEYS)[:766]
    snapshot = dict(schema="pfc_complete_domain_history_covariance_v1",
        execution=dict(run_id="413de649f13ddf8f", trial_id="437784903"),
        request_sha256=recovery.ANALYSIS_SHA256, required_groups=1000, completed_groups=766,
        completed_group_keys=old, completed=[dict(circuit_index=key // 30, condition_index=key % 30,
            report_sha256="a" * 64) for key in old], failures=[], source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scaling_qualified=False, uncertainty_coverage_qualified=False)
    analysis = dict(schema=snapshot["schema"], required_groups=1000, missing_native_event_indices=list(range(35)))
    return analysis, snapshot


def test_exact218_targets_exclude_all766_and16_previous_groups():
    analysis, snapshot = plan_fixture()
    targets, old, records = recovery.recovery_targets(analysis, snapshot)
    assert len(targets) == 218 and len(old) == 782 and len(records) == 766
    assert targets.isdisjoint(snapshot["completed_group_keys"])
    assert targets.isdisjoint(recovery.SUPPLEMENTAL_KEYS)
    assert targets | old == recovery.intake.REQUIRED_GROUPS


@pytest.mark.parametrize("change", ["duplicate", "supplemental_overlap", "count", "source_request"])
def test_changed_original_group_allocation_is_rejected(change):
    analysis, snapshot = plan_fixture()
    if change == "duplicate":
        snapshot["completed_group_keys"][-1] = snapshot["completed_group_keys"][0]
    elif change == "supplemental_overlap":
        snapshot["completed_group_keys"][0] = 0
        snapshot["completed"][0] = dict(circuit_index=0, condition_index=0, report_sha256="a" * 64)
    elif change == "count":
        snapshot["completed_groups"] = 765
    else:
        snapshot["request_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        recovery.recovery_targets(analysis, snapshot)


def test_changed_bound_input_fails_before_consumption(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"schema":"original"}')
    binding = recovery.descriptor(path)
    path.write_text('{"schema":"changed"}')
    with pytest.raises(ValueError, match="ORIGINAL_FILE_BINDING_CHANGED"):
        recovery.bound_json(binding, tmp_path)


def response(code, stdout=b"", stderr=b""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr, args=["hdfs", "dfs", "-cat", "hdfs://x/input"])


def test_only_nonzero_cat_reads_retry_and_preserve_stderr(monkeypatch):
    results = iter([response(255, stderr=b"TLS read failure"), response(255, stderr=b"second read failure"), response(0, b'{"ready":true}')])
    calls, sleeps, failures = [], [], []
    def run(*args, **kwargs):
        calls.append(args)
        return next(results)
    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    assert recovery.read_json_cat("hdfs://x/input", failures) == {"ready": True}
    assert len(calls) == 3 and sleeps == [5, 5]
    assert [row["exit_code"] for row in failures] == [255, 255]
    assert failures[0]["stderr"] == "TLS read failure"


def test_exhausted_cat_has_three_attempts_exit_and_stderr(monkeypatch):
    calls, failures = [], []
    def run(*args, **kwargs):
        calls.append(args)
        return response(255, stderr=b"original stderr")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)
    with pytest.raises(recovery.HDFSReadError) as caught:
        recovery.read_json_cat("hdfs://x/input", failures)
    assert len(calls) == len(failures) == len(caught.value.attempts) == 3
    details = recovery.failure_details(caught.value)
    assert details["exit_code"] == 255 and details["stderr"] == "original stderr"


def test_successful_cat_with_invalid_json_is_not_retried(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(args)
        return response(0, b"invalid json")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: pytest.fail("Semantic errors must not retry"))
    with pytest.raises(json.JSONDecodeError):
        recovery.read_json_cat("hdfs://x/input", [])
    assert len(calls) == 1


def checkpoint_fixture(tmp_path, extra=None):
    collector = dict(run_id="collector-run", trial_id="collector-trial")
    execution = dict(backend="merlin", **collector)
    files = {"cache/group_0001/report.json": b'{"original":"report"}',
             "cache/group_0001/matched_history_variance1ms.csv": b"original signed variance strings",
             "cache/group_0001/history_pair_unit_bin_coincidences.csv": b"original pairs"}
    rows = {name:dict(path=name, bytes=len(data), sha256=recovery.digest(data)) for name,data in files.items()}
    entry = dict(group_key=1, report=rows["cache/group_0001/report.json"], tables={
        recovery.intake.VARIANCE_NAME: rows["cache/group_0001/" + recovery.intake.VARIANCE_NAME],
        recovery.intake.PAIR_NAME: rows["cache/group_0001/" + recovery.intake.PAIR_NAME]})
    progress = dict(execution=execution, checkpoint_sequence=1, returned_groups=1, returned_group_keys=[1],
        failures=[], source_failures=[], source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scientific_parameter_domain_qualified=False,
        scaling_qualified=False, uncertainty_coverage_qualified=False, source_ci_qualified=False)
    controls = {"config.json": dict(schema=recovery.COLLECTOR_SCHEMA, main_run_id="413de649f13ddf8f",
                    main_trial_id="437784903", main_request_sha256=recovery.ANALYSIS_SHA256),
                "input_manifest.json": dict(schema=recovery.intake.SCHEMA, groups=[entry]),
                "collection_progress.json": progress, "source_progress.json": dict(original=True)}
    archive_path = tmp_path / "cache.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name,data in {**files, **{name:json.dumps(value).encode() for name,value in controls.items()}}.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
    data = archive_path.read_bytes()
    receipt = dict(schema=recovery.COLLECTOR_SCHEMA, execution=execution, persisted=True,
        checkpoint_sequence=1, source_simulations=0, production_episodes=0, source_qualification_granted=False,
        bundle_bytes=len(data), bundle_sha256=recovery.digest(data), added_group_keys=[1], returned_groups=1)
    return archive_path, receipt, collector, entry, files


def restore(archive, receipt, collector, destination, previous=None, sequence=0):
    return recovery.restore_delta(archive, receipt, destination, previous or {}, sequence, {1}, {}, collector)


def test_exact_collector_delta_restores_original_bound_files(tmp_path):
    archive, receipt, collector, entry, files = checkpoint_fixture(tmp_path)
    destination = tmp_path / "restored"
    restored = restore(archive, receipt, collector, destination)
    assert restored == {1: entry}
    for name,data in files.items():
        assert (destination / name).read_bytes() == data
    assert not (destination / "config.json").exists()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "unexpected"])
def test_unsafe_or_undeclared_checkpoint_members_are_rejected_before_write(tmp_path, kind):
    name = "../escape" if kind == "traversal" else "undeclared"
    member = tarfile.TarInfo(name)
    member.size = 1
    if kind in ("symlink", "hardlink"):
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "../escape"
        member.size = 0
    archive, receipt, collector, _, _ = checkpoint_fixture(tmp_path, member)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="PATH_INVALID|REGULAR_UNIQUE|UNDECLARED_MEMBER"):
        restore(archive, receipt, collector, destination)
    assert not destination.exists() and not (tmp_path / "escape").exists()


def test_delta_never_overwrites_different_original_input(tmp_path):
    archive, receipt, collector, _, files = checkpoint_fixture(tmp_path)
    destination = tmp_path / "restored"
    target = destination / next(iter(files))
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different original input")
    with pytest.raises(ValueError, match="CHECKPOINT_EXISTING_INPUT_CHANGED"):
        restore(archive, receipt, collector, destination)
    assert target.read_bytes() == b"different original input"


@pytest.mark.parametrize("change", ["bundle_hash", "sequence", "execution", "added_keys"])
def test_delta_receipt_binding_or_chain_changes_are_rejected(tmp_path, change):
    archive, receipt, collector, _, _ = checkpoint_fixture(tmp_path)
    if change == "bundle_hash":
        receipt["bundle_sha256"] = "0" * 64
    elif change == "sequence":
        receipt["checkpoint_sequence"] = 2
    elif change == "execution":
        receipt["execution"]["run_id"] = "foreign"
    else:
        receipt["added_group_keys"] = []
    with pytest.raises(ValueError, match="BUNDLE_CHANGED|CHAIN_CHANGED|DELTA_DENOMINATOR_CHANGED"):
        restore(archive, receipt, collector, tmp_path / "restored")


def test_merge_reuses782_and218_without_mutating_inputs():
    analysis, snapshot = plan_fixture()
    targets, old_keys, _ = recovery.recovery_targets(analysis, snapshot)
    old = {key:dict(group_key=key, source="cached-original") for key in old_keys}
    new = {key:dict(group_key=key, source="new-recovery") for key in targets}
    before = copy.deepcopy(old)
    result = recovery.merge_manifest(old, new, targets)
    assert len(result["groups"]) == 1000
    assert [row["group_key"] for row in result["groups"]] == sorted(recovery.intake.REQUIRED_GROUPS)
    assert old == before
    new.pop(min(targets))
    with pytest.raises(ValueError, match="EXACT782_PLUS218_MERGE_REQUIRED"):
        recovery.merge_manifest(old, new, targets)



def source_acceptance_fixture(tmp_path, indices=(843,853,863)):
    analysis = dict(source_origins={}, identities={})
    payloads = {}
    for position,index in enumerate(indices):
        checkpoint = f"hdfs://haruna/source-run/pfc/task_{index:04d}"
        analysis["source_origins"][str(index)] = dict(checkpoint_path=checkpoint, receipt_uri=checkpoint+"/receipt.json",
            source_run="source-run", source_trial="source-trial")
        analysis["identities"][str(index)] = dict(circuit_index=28,condition_index=3,block_index=6+position,episode_id=f"ep-{index}")
        receipt = dict(index=index,run_id="source-run",trial_id="source-trial",xie_job=f"original-{index}",
            persisted=True,exit_code=0,status="COMPLETE",checkpoint_path=checkpoint,result_bytes=123,result_sha256="b"*64,
            publication_recovery=dict(source_bundle_uri=checkpoint+"/result.tar.gz"))
        payloads[checkpoint+"/receipt.json"] = receipt
        payloads[checkpoint+"/manifest.json"] = {f"original/task_{index:04d}/source_run/raw.npz":dict(bytes=2,sha256="a"*64)}
    index = indices[0]
    receipt = payloads[analysis["source_origins"][str(index)]["receipt_uri"]]
    rescue_uri = f"hdfs://haruna/rescue/task_{index:04d}/receipt.json"
    payloads[rescue_uri] = copy.deepcopy(receipt)
    accepted = dict(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE",index=index,identity=analysis["identities"][str(index)],
        run_id="source-run",trial_id="source-trial",xie_job=receipt["xie_job"],remote_receipt=receipt,
        receipt_uri=rescue_uri,source_bundle_uri=receipt["checkpoint_path"]+"/result.tar.gz",source_bundle_sha256="b"*64,
        source_simulations=0,production_episodes=0,source_qualification_granted=False,
        scientific_parameter_domain_qualified=False,scaling_qualified=False,
        validation_execution=dict(backend="merlin",allocated_gpus=0,numpy_version="2.4.2",run_id="intake",trial_id="trial"))
    path = tmp_path/"accepted.json"; path.write_text(json.dumps(accepted))
    return analysis, dict(recovery.descriptor(path),index=index), payloads


def test_bound_source_override_uses_isolated_receipt_and_original_bundle(tmp_path, monkeypatch):
    analysis,binding,payloads = source_acceptance_fixture(tmp_path)
    overrides = recovery.load_source_overrides([binding],tmp_path,analysis)
    calls=[]
    monkeypatch.setattr(recovery,"read_cat",lambda uri,failures:(calls.append(uri) or json.dumps(payloads[uri]).encode()))
    metadata=tmp_path/"metadata";metadata.mkdir()
    group = recovery.prepare_group(analysis,843,metadata,[],overrides)
    task=group["tasks"][0];origin=analysis["source_origins"]["843"]
    assert task["receipt_uri"]==overrides[843]["receipt_uri"]!=origin["receipt_uri"]
    assert task["bundle_uri"]==origin["checkpoint_path"]+"/result.tar.gz"
    assert task["checkpoint_path"]==origin["checkpoint_path"] and task["source_run"]==origin["source_run"]
    assert origin["checkpoint_path"]+"/manifest.json" in calls
    assert origin["receipt_uri"] not in calls


@pytest.mark.parametrize("change",["identity","bundle","scope","receipt"])
def test_source_override_conflicts_are_rejected(tmp_path,change):
    analysis,binding,_ = source_acceptance_fixture(tmp_path)
    path=Path(binding["path"]);value=json.loads(path.read_bytes())
    if change=="identity":value["identity"]["block_index"]+=1
    elif change=="bundle":value["source_bundle_sha256"]="c"*64
    elif change=="scope":value["production_episodes"]=1
    else:value["remote_receipt"]["xie_job"]="other"
    path.write_text(json.dumps(value));binding.update(recovery.descriptor(path))
    with pytest.raises(ValueError):recovery.load_source_overrides([binding],tmp_path,analysis)


def test_override_receipt_changed_after_intake_is_rejected(tmp_path,monkeypatch):
    analysis,binding,payloads=source_acceptance_fixture(tmp_path)
    overrides=recovery.load_source_overrides([binding],tmp_path,analysis)
    payloads[overrides[843]["receipt_uri"]]["result_sha256"]="c"*64
    monkeypatch.setattr(recovery,"read_cat",lambda uri,failures:json.dumps(payloads[uri]).encode())
    metadata=tmp_path/"metadata";metadata.mkdir()
    with pytest.raises(ValueError,match="SOURCE_OVERRIDE_RECEIPT_CONFLICT"):
        recovery.prepare_group(analysis,843,metadata,[],overrides)


def test_readiness_union_requires_three_histories_and_excludes_completed():
    excluded=recovery.intake.REQUIRED_GROUPS-{843}
    assert recovery.ready_recovery_groups([853],{843},excluded,{843})==[]
    assert recovery.ready_recovery_groups([853,863],{843},excluded,{843})==[843]
    assert recovery.ready_recovery_groups([853,863],{843},excluded|{843},{843})==[]


def recovery_delta_fixture(tmp_path, mutate=None, extra=None, key=None):
    analysis,original=plan_fixture();targets,_,_=recovery.recovery_targets(analysis,original)
    key=min(targets) if key is None else key
    assert key in targets
    source=dict(run_id="resume-run",trial_id="resume-trial",checkpoint_uri="hdfs://haruna/resume-run/pfc_history_covariance_recovery")
    execution=dict(backend="merlin",run_id=source["run_id"],trial_id=source["trial_id"])
    files={f"group_{key:04d}/report.json":b'{"original":"report"}',
        f"group_{key:04d}/"+recovery.intake.VARIANCE_NAME:b"signed original values",
        f"group_{key:04d}/"+recovery.intake.PAIR_NAME:b"original pairs"}
    rows={name:dict(path=name,bytes=len(data),sha256=recovery.digest(data)) for name,data in files.items()}
    entry=dict(group_key=key,report=rows[f"group_{key:04d}/report.json"],
        tables={name:rows[f"group_{key:04d}/"+name] for name in recovery.TABLES},report_uri="hdfs://haruna/original-report")
    tasks=[];analysis.update(source_origins={},identities={})
    for position,index in enumerate([key,key+10,key+20]):
        identity=dict(circuit_index=key//30,condition_index=key%30,block_index=6+position,episode_id=f"ep-{index}")
        origin=dict(source_run="source-run",source_trial="source-trial",checkpoint_path=f"hdfs://haruna/source/pfc/task_{index:04d}",
                    receipt_uri=f"hdfs://haruna/source/pfc/task_{index:04d}/receipt.json")
        analysis["identities"][str(index)]=identity;analysis["source_origins"][str(index)]=origin
        tasks.append(dict(origin,index=index,identity=identity,receipt_sha256="a"*64,manifest_sha256="a"*64))
    group_request=dict(schema=recovery.intake.SOURCE_SCHEMA,circuit_index=key//30,condition_index=key%30,
                       task_indices=[key,key+10,key+20],tasks=tasks)
    progress=dict(schema=recovery.SCHEMA,execution=execution,checkpoint_sequence=1,core_sha256=recovery.CORE_SHA256,
        original_analysis_sha256=recovery.ANALYSIS_SHA256,original_snapshot_sha256=recovery.SNAPSHOT_SHA256,
        target_group_keys=sorted(targets),required_recovery_groups=218,required_full_groups=1000,
        original_completed_groups=766,completed_supplemental_groups=16,completed_recovery_groups=1,completed_group_keys=[key],
        completed=[dict(group_key=key,circuit_index=key//30,condition_index=key%30,report_uri=entry["report_uri"],
                        report_sha256=entry["report"]["sha256"])],failures=[],source_simulations=0,production_episodes=0,
        source_qualification_granted=False,scientific_parameter_domain_qualified=False,scaling_qualified=False,
        uncertainty_coverage_qualified=False,source_ci_qualified=False)
    controls={"progress.json":progress,"recovery_inputs.json":dict(schema=recovery.intake.SCHEMA,groups=[entry]),
              f"group_requests/group_{key:04d}.json":group_request}
    if mutate is not None:mutate(progress,entry,files,group_request)
    path=tmp_path/"recovery_cache.tar.gz"
    with tarfile.open(path,"w:gz") as archive:
        for name,data in {**files,**{name:json.dumps(value).encode() for name,value in controls.items()}}.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
        if extra is not None:archive.addfile(extra,io.BytesIO(b"x") if extra.isfile() else None)
    receipt=dict(schema=recovery.SCHEMA,execution=execution,persisted=True,checkpoint_sequence=1,
        source_simulations=0,production_episodes=0,source_qualification_granted=False,completed_recovery_groups=1,
        bundle_uri=source["checkpoint_uri"]+"/batch_0001/cache.tar.gz",bundle_bytes=path.stat().st_size,
        bundle_sha256=recovery.digest(path.read_bytes()),added_group_keys=[key])
    return path,receipt,source,analysis,targets,entry,files


def test_recovery_schema_restores_bound_outputs_without_core_execution(tmp_path,monkeypatch):
    path,receipt,source,analysis,targets,entry,files=recovery_delta_fixture(tmp_path)
    monkeypatch.setattr(recovery.core,"run",lambda *args:pytest.fail("Completed covariance must not rerun"))
    destination=tmp_path/"restored"
    result=recovery.restore_recovery_delta(path,receipt,destination,{},0,targets,source,analysis)
    assert result=={entry["group_key"]:entry}
    assert all((destination/name).read_bytes()==data for name,data in files.items())
    assert (destination/f"group_requests/group_{entry['group_key']:04d}.json").exists()
    assert not (destination/"progress.json").exists()
    assert recovery.ready_recovery_groups(list(range(3000)),[],recovery.intake.REQUIRED_GROUPS,targets)==[]


@pytest.mark.parametrize("change",["core","analysis","snapshot","denominator","source","table_hash","identity"])
def test_recovery_checkpoint_changed_contract_is_rejected_before_write(tmp_path,change):
    def mutate(progress,entry,files,request):
        if change in ("core","analysis","snapshot"):
            field={"core":"core_sha256","analysis":"original_analysis_sha256","snapshot":"original_snapshot_sha256"}[change]
            progress[field]="c"*64
        elif change=="denominator":progress["required_full_groups"]=999
        elif change=="source":progress["execution"]["trial_id"]="other"
        elif change=="table_hash":entry["tables"][recovery.intake.VARIANCE_NAME]["sha256"]="c"*64
        else:request["tasks"][0]["identity"]=dict(request["tasks"][0]["identity"],episode_id="other")
    path,receipt,source,analysis,targets,_,_=recovery_delta_fixture(tmp_path,mutate)
    destination=tmp_path/"restored"
    with pytest.raises(ValueError):recovery.restore_recovery_delta(path,receipt,destination,{},0,targets,source,analysis)
    assert not destination.exists()


@pytest.mark.parametrize("change",["hash","sequence","added_keys","uri"])
def test_recovery_checkpoint_receipt_conflicts_are_rejected(tmp_path,change):
    path,receipt,source,analysis,targets,_,_=recovery_delta_fixture(tmp_path)
    if change=="hash":receipt["bundle_sha256"]="c"*64
    elif change=="sequence":receipt["checkpoint_sequence"]=2
    elif change=="added_keys":receipt["added_group_keys"]=[]
    else:receipt["bundle_uri"]="hdfs://foreign/cache.tar.gz"
    with pytest.raises(ValueError):recovery.restore_recovery_delta(path,receipt,tmp_path/"restored",{},0,targets,source,analysis)


def test_bound_resume_interface_reports_original_execution_and_completed_groups(tmp_path):
    path,receipt,source,analysis,targets,entry,_=recovery_delta_fixture(tmp_path)
    receipt_path=tmp_path/"receipt.json";receipt_path.write_text(json.dumps(receipt))
    request=dict(resume_execution=source,resume_checkpoints=[dict(receipt=recovery.descriptor(receipt_path),archive=recovery.descriptor(path))])
    destination=tmp_path/"restored"
    groups,provenance=recovery.resume_recovery(request,tmp_path,destination,targets,analysis,{})
    assert groups=={entry["group_key"]:entry}
    assert provenance[0]["completed_groups"]==1 and provenance[0]["original_execution"]["run_id"]==source["run_id"]
    assert recovery.resume_recovery({},tmp_path,tmp_path/"unused",targets,analysis,{})==({},[])



def test_resumed_compute_requires_bound_original_terminal(tmp_path):
    source=dict(run_id="old-run",trial_id="old-trial")
    request=dict(resume_execution=source,resume_checkpoints=[{}])
    current=dict(run_id="new-run",trial_id="new-trial")
    with pytest.raises(ValueError,match="RESUME_SOURCE_TERMINAL_REQUIRED"):
        recovery.validate_resume_terminal(request,tmp_path,current)
    terminal=dict(schema=recovery.SCHEMA,execution=dict(backend="merlin",**source),
        original_snapshot_sha256=recovery.SNAPSHOT_SHA256,source_simulations=0,production_episodes=0,
        source_qualification_granted=False,status="RECOVERING_REMAINING_GROUPS",failures=[])
    path=tmp_path/"terminal.json";path.write_text(json.dumps(terminal))
    request["resume_source_terminal"]=recovery.descriptor(path)
    with pytest.raises(ValueError,match="STILL_RUNNING_OR_UNRESOLVED"):
        recovery.validate_resume_terminal(request,tmp_path,current)
    terminal.update(status="FAILED_COVARIANCE_RECOVERY",failures=[dict(message="original read failed")],
                    core_sha256=recovery.CORE_SHA256,original_analysis_sha256=recovery.ANALYSIS_SHA256)
    path.write_text(json.dumps(terminal));request["resume_source_terminal"]=recovery.descriptor(path)
    result=recovery.validate_resume_terminal(request,tmp_path,current)
    assert result["status"]=="FAILED_COVARIANCE_RECOVERY"
    with pytest.raises(ValueError,match="SEPARATE_RESUME_EXECUTION_REQUIRED"):
        recovery.validate_resume_terminal(request,tmp_path,source)



def test_normalized_absolute_binding_paths_are_used_by_source_and_resume(tmp_path):
    source_analysis,binding,_=source_acceptance_fixture(tmp_path)
    binding["path"]=Path(binding["path"]).name
    overrides=recovery.load_source_overrides([binding],tmp_path,source_analysis)
    canonical=overrides[binding["index"]]["acceptance_binding"]
    assert canonical["path"]==str((tmp_path/binding["path"]).resolve())
    assert canonical["bytes"]==binding["bytes"] and canonical["sha256"]==binding["sha256"]
    path,receipt,source,analysis,targets,entry,_=recovery_delta_fixture(tmp_path)
    receipt_path=tmp_path/"relative_receipt.json";receipt_path.write_text(json.dumps(receipt))
    receipt_binding=recovery.descriptor(receipt_path);receipt_binding["path"]=receipt_path.name
    archive_binding=recovery.descriptor(path);archive_binding["path"]=path.name
    request=dict(resume_execution=source,resume_checkpoints=[dict(receipt=receipt_binding,archive=archive_binding)])
    groups,provenance=recovery.resume_recovery(request,tmp_path,tmp_path/"restored_relative",targets,analysis,{})
    assert groups=={entry["group_key"]:entry}
    assert provenance[0]["receipt"]["path"]==str(receipt_path.resolve())
    assert provenance[0]["archive"]["path"]==str(path.resolve())


@pytest.mark.parametrize("change",[None,"running","unknown","unconfirmed","zero","bool","owner"])
def test_resume_uses_real_failed_operations_execution_record(tmp_path,change):
    source=dict(run_id="28bf3f86a34814e0",trial_id="438662058")
    value=dict(**source,platform_status="failed",terminal_confirmed=True,final_exit_code=1)
    if change in ("running","unknown"):value["platform_status"]=change
    elif change=="unconfirmed":value["terminal_confirmed"]=False
    elif change=="zero":value["final_exit_code"]=0
    elif change=="bool":value["final_exit_code"]=True
    elif change=="owner":value["trial_id"]="other"
    path=tmp_path/"execution.json";path.write_text(json.dumps(value))
    request=dict(resume_execution=source,resume_checkpoints=[{}],resume_source_terminal=recovery.descriptor(path))
    current=dict(run_id="new-run",trial_id="new-trial")
    if change is None:
        result=recovery.validate_resume_terminal(request,tmp_path,current)
        assert result["evidence_kind"]=="operations_execution" and result["status"]=="failed"
        assert result["binding"]["path"]==str(path.resolve())
    else:
        with pytest.raises(ValueError):recovery.validate_resume_terminal(request,tmp_path,current)


def test_successful_original1000_intake_cannot_resume_partial_cache(tmp_path):
    source=dict(run_id="original-run",trial_id="original-trial")
    terminal=dict(schema=recovery.SCHEMA,status="COMPLETE1000_RECOVERED_COVARIANCE_INTAKE",
        execution=dict(backend="merlin",**source),original_snapshot_sha256=recovery.SNAPSHOT_SHA256,
        source_simulations=0,production_episodes=0,source_qualification_granted=False,
        persisted=True,exit_code=0,complete_groups=1000,failures=[])
    path=tmp_path/"completed_receipt.json";path.write_text(json.dumps(terminal))
    request=dict(resume_execution=source,resume_checkpoints=[{}],resume_source_terminal=recovery.descriptor(path))
    with pytest.raises(ValueError,match="REUSE_FINAL_INTAKE"):
        recovery.validate_resume_terminal(request,tmp_path,dict(run_id="new-run"))



def source_replacement_fixture(tmp_path, monkeypatch, index=2641):
    from . import pfc_merlin_observation_completion as observations
    job, recipe_sha = recovery.SOURCE_REPLACEMENT_RECIPES[index]
    circuit, condition = index // 30, index % 10
    key = circuit * 30 + condition
    analysis = dict(source_origins={}, identities={})
    payloads = {}
    for position, current in enumerate((key, key+10, key+20)):
        block = 6 + position
        identity = dict(circuit_index=circuit, condition_index=condition, block_index=block,
            scenario_id="pfc_working_memory", circuit_id=f"pfc_working_memory_circuit_{circuit:03d}",
            block_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}",
            episode_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{condition:02d}",
            split="calibration" if circuit < 90 else "test", sample_unit="complete_mechanistic_episode",
            connectivity_seed=200000000+circuit, initial_state_seed=210000000+circuit*1000+block,
            background_seed=220000000+circuit*1000+block, stimulus_seed=230000000+circuit*1000+block)
        checkpoint = f"hdfs://haruna/checkpoints/20261021/6de397acc866aadd/pfc/task_{current:04d}"
        original_job = job if current == index else f"original-{current}"
        origin = dict(source_run="6de397acc866aadd", source_trial="435906314", checkpoint_path=checkpoint,
                      receipt_uri=checkpoint+"/receipt.json", xie_job=original_job)
        analysis["source_origins"][str(current)] = origin
        analysis["identities"][str(current)] = identity
        payloads[origin["receipt_uri"]] = dict(index=current, run_id=origin["source_run"], trial_id=origin["source_trial"],
            status="COMPLETE", persisted=True, exit_code=0, xie_job=original_job, checkpoint_path=checkpoint,
            result_bytes=123, result_sha256="b"*64)
        payloads[checkpoint+"/manifest.json"] = {f"original/task_{current:04d}/source_run/raw.npz":dict(bytes=2,sha256="a"*64)}
    runtime = SimpleNamespace(DOMAIN=tmp_path/"domain")
    plan = dict(raw_dt_ms=.0125, duration_ms=5000., coupled_core_units=100000, rows=[{} for _ in range(3000)])
    plan["rows"][index] = dict(identity=copy.deepcopy(analysis["identities"][str(index)]), config_sha256=recipe_sha)
    monkeypatch.setattr(observations, "load_runtime", lambda binding,base:runtime)
    monkeypatch.setattr(observations, "qualified_readout_plan", lambda request,runtime:(plan,{}))
    source = dict(run_id="0123456789abcdef", trial_id="439999111", xie_job=job)
    checkpoint = f"hdfs://haruna/checkpoints/20261025/{source['run_id']}/pfc/task_{index:04d}"
    receipt = dict(source, index=index, status="COMPLETE", persisted=True, exit_code=0,
                   checkpoint_path=checkpoint, result_bytes=456, result_sha256="d"*64)
    validation = dict(backend="merlin", run_id="fedcba9876543210", trial_id="439999222",
                      allocated_gpus=0, numpy_version="2.4.2")
    accepted = dict(source, status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=index,
        identity=copy.deepcopy(analysis["identities"][str(index)]), config_sha256=recipe_sha,
        remote_receipt=receipt, receipt_uri=checkpoint+"/receipt.json", source_bundle_uri=checkpoint+"/result.tar.gz",
        source_bundle_sha256="d"*64, validation_execution=validation,
        observation_entry=dict(path=str(runtime.DOMAIN/"observations"/f"task_{index:04d}"),
            development_report_sha256="a"*64, observation_report_sha256="b"*64),
        numerical_source={name:dict(returned_sha256=value,frozen_sha256=value,frozen_path="/producer/"+name)
                          for name,value in recovery.FROZEN_SOURCE_NUMERICAL_SHA256.items()},
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scientific_parameter_domain_qualified=False, scaling_qualified=False)
    accepted_path = tmp_path/"replacement_acceptance.json";accepted_path.write_text(json.dumps(accepted))
    acceptance_binding = dict(recovery.descriptor(accepted_path),index=index)
    intake_checkpoint = "hdfs://haruna/checkpoints/20261025/fedcba9876543210/pfc_scientific_intake"
    intake = dict(schema="pfc_merlin_scientific_intake_v1",status="PASS",failures=[],execution=validation,
        checkpoint_path=intake_checkpoint, requested_task_indices=[index],new_accepted_episode_count=1,
        completed=[dict(index=index,acceptance_uri=intake_checkpoint+f"/task_{index:04d}_acceptance.json",
                        acceptance_sha256=acceptance_binding["sha256"])],source_simulations=0,production_episodes=0,
        source_qualification_granted=False,scientific_parameter_domain_qualified=False,scaling_qualified=False)
    intake_path = tmp_path/"replacement_intake.json";intake_path.write_text(json.dumps(intake))
    request = dict(schema=observations.SOURCE_READOUT_SCHEMA,index=index,publication_kind="native_source",
        accepted_source=acceptance_binding,scientific_intake=recovery.descriptor(intake_path),source_execution=source,
        publication_execution={name:source[name] for name in ("run_id","trial_id")},runtime={},
        preparation_sha256="a"*64,consumer_sha256={},work=str(tmp_path/"unused_publication_work"))
    request_path = tmp_path/"replacement_request.json";request_path.write_text(json.dumps(request))
    record = dict(index=index,original_origin=copy.deepcopy(analysis["source_origins"][str(index)]),
                  source_readout_request=recovery.descriptor(request_path))
    payloads[accepted["receipt_uri"]] = receipt
    payloads[checkpoint+"/manifest.json"] = {f"isolated/task_{index:04d}/run_merlin_439999111_pfc_{index:04d}/source_run/raw.npz":dict(bytes=2,sha256="a"*64)}
    return analysis, record, payloads, plan, accepted_path, intake_path, request_path


@pytest.mark.parametrize("index",[2641,2717])
def test_real_native_source_replacement_retains_original_origin_and_frozen_inputs(tmp_path,monkeypatch,index):
    analysis,record,payloads,_,_,_,_=source_replacement_fixture(tmp_path,monkeypatch,index)
    before=copy.deepcopy(analysis)
    overrides=recovery.load_source_replacements([record],tmp_path,analysis)
    override=overrides[index]
    assert override["source_run"]=="0123456789abcdef" and override["source_trial"]=="439999111"
    assert override["original_origin"]==before["source_origins"][str(index)]
    assert override["config_sha256"]==recovery.SOURCE_REPLACEMENT_RECIPES[index][1]
    assert override["numerical_source_sha256"]==recovery.FROZEN_SOURCE_NUMERICAL_SHA256
    calls=[]
    monkeypatch.setattr(recovery,"read_cat",lambda uri,failures:(calls.append(uri) or json.dumps(payloads[uri]).encode()))
    metadata=tmp_path/"metadata";metadata.mkdir()
    key=index//30*30+index%10
    request=recovery.prepare_group(analysis,key,metadata,[],overrides)
    task=next(task for task in request["tasks"] if task["index"]==index)
    assert task["source_run"]==override["source_run"] and task["source_trial"]==override["source_trial"]
    assert task["original_source_origin"]==before["source_origins"][str(index)]
    assert task["checkpoint_path"]==override["effective_origin"]["checkpoint_path"]
    assert task["source_replacement_binding"]["sha256"]==record["source_readout_request"]["sha256"]
    assert before["source_origins"][str(index)]["checkpoint_path"]+"/manifest.json" not in calls
    assert override["effective_origin"]["checkpoint_path"]+"/manifest.json" in calls
    assert analysis==before


@pytest.mark.parametrize("change",["index","origin","identity","config","recipe","clock","units","stream",
                                    "numerical","source","receipt","not_pass","missing_intake","old_source","job"])
def test_source_replacement_requires_bound_qualified_same_dataset(tmp_path,monkeypatch,change):
    analysis,record,_,plan,accepted_path,intake_path,request_path=source_replacement_fixture(tmp_path,monkeypatch)
    accepted=json.loads(accepted_path.read_bytes());request=json.loads(request_path.read_bytes());intake=json.loads(intake_path.read_bytes())
    if change=="index":record["index"]=2979
    elif change=="origin":record["original_origin"]["source_trial"]="foreign"
    elif change=="identity":accepted["identity"]["block_index"]+=1
    elif change=="config":accepted["config_sha256"]="c"*64
    elif change=="recipe":plan["rows"][2641]["config_sha256"]="c"*64
    elif change=="clock":plan["raw_dt_ms"]=.025
    elif change=="units":plan["coupled_core_units"]=99999
    elif change=="stream":plan["rows"][2641]["identity"]["initial_state_seed"]+=1
    elif change=="numerical":accepted["numerical_source"]["pfc_native_drive.py"]["returned_sha256"]="c"*64
    elif change=="source":accepted["run_id"]="6de397acc866aadd"
    elif change=="receipt":accepted["remote_receipt"]["trial_id"]="435906314"
    elif change=="not_pass":intake["status"]="FAILED"
    elif change=="missing_intake":request.pop("scientific_intake")
    elif change=="old_source":
        origin=analysis["source_origins"]["2641"]
        for current in (accepted,accepted["remote_receipt"],request["source_execution"]):
            current.update(run_id=origin["source_run"],trial_id=origin["source_trial"])
        accepted["remote_receipt"]["checkpoint_path"]=origin["checkpoint_path"]
        accepted.update(receipt_uri=origin["receipt_uri"],source_bundle_uri=origin["checkpoint_path"]+"/result.tar.gz")
        request["publication_execution"]={name:origin["source_"+name.removesuffix("_id")] for name in ("run_id","trial_id")}
    else:
        for current in (accepted,accepted["remote_receipt"],request["source_execution"]):current["xie_job"]="other-job"
    accepted_path.write_text(json.dumps(accepted));request["accepted_source"].update(recovery.descriptor(accepted_path))
    intake["completed"][0]["acceptance_sha256"]=request["accepted_source"]["sha256"]
    intake_path.write_text(json.dumps(intake))
    if "scientific_intake" in request:request["scientific_intake"]=recovery.descriptor(intake_path)
    request_path.write_text(json.dumps(request));record["source_readout_request"]=recovery.descriptor(request_path)
    before=copy.deepcopy(analysis)
    with pytest.raises(ValueError):recovery.load_source_replacements([record],tmp_path,analysis)
    assert analysis==before and not (tmp_path/"unused_publication_work").exists()


def replacement_cache_fixture(tmp_path,monkeypatch,index=2641,original_source=False,mutate_request=None):
    analysis,record,payloads,_,_,_,_=source_replacement_fixture(tmp_path,monkeypatch,index)
    overrides=recovery.load_source_replacements([record],tmp_path,analysis)
    monkeypatch.setattr(recovery,"read_cat",lambda uri,failures:json.dumps(payloads[uri]).encode())
    metadata=tmp_path/"metadata";metadata.mkdir()
    key=index//30*30+index%10
    effective=recovery.prepare_group(analysis,key,metadata,[],{} if original_source else overrides)
    def mutate(progress,entry,files,request):
        request.update(copy.deepcopy(effective))
        if mutate_request is not None:mutate_request(request)
    path,receipt,source,_,targets,entry,files=recovery_delta_fixture(tmp_path,mutate,key=key)
    return path,receipt,source,analysis,targets,entry,files,overrides


@pytest.mark.parametrize("original_source",[False,True])
def test_replacement_checkpoint_reuses_completed_cache_without_changing_original_outputs(tmp_path,monkeypatch,original_source):
    path,receipt,source,analysis,targets,entry,files,overrides=replacement_cache_fixture(tmp_path,monkeypatch,original_source=original_source)
    monkeypatch.setattr(recovery.core,"run",lambda *args:pytest.fail("Completed cache must not rerun"))
    destination=tmp_path/"restored"
    result=recovery.restore_recovery_delta(path,receipt,destination,{},0,targets,source,analysis,overrides)
    assert result=={entry["group_key"]:entry}
    assert all((destination/name).read_bytes()==data for name,data in files.items())
    request=json.loads((destination/f"group_requests/group_{entry['group_key']:04d}.json").read_bytes())
    task=request["tasks"][0]
    assert task["source_run"]==("6de397acc866aadd" if original_source else "0123456789abcdef")
    excluded=(recovery.intake.REQUIRED_GROUPS-targets)|{entry["group_key"]}
    assert entry["group_key"] not in recovery.ready_recovery_groups(list(range(3000)),overrides,excluded,targets)


@pytest.mark.parametrize("change",["missing_proof","binding","original_origin","source","config"])
def test_replacement_checkpoint_requires_same_validated_source_proof(tmp_path,monkeypatch,change):
    def mutate(request):
        task=request["tasks"][0]
        if change=="binding":task["source_replacement_binding"]["sha256"]="c"*64
        elif change=="original_origin":task["original_source_origin"]["source_trial"]="foreign"
        elif change=="source":task["source_trial"]="435906314"
        elif change=="config":task["config_sha256"]="c"*64
    path,receipt,source,analysis,targets,_,_,overrides=replacement_cache_fixture(tmp_path,monkeypatch,mutate_request=mutate)
    if change=="missing_proof":overrides={}
    destination=tmp_path/"restored"
    with pytest.raises(ValueError):recovery.restore_recovery_delta(path,receipt,destination,{},0,targets,source,analysis,overrides)
    assert not destination.exists()



def covariance_authorized_stop_document(exit_code=None):
    run_id, trial_id = "28bf3f86a34814e0", "438662058"
    return dict(run_id=run_id, trial_id=trial_id, platform_status="stopped", terminal_confirmed=True, final_exit_code=exit_code,
        user_authorization=dict(source="user_explicit_reply", date="2026-10-26",
            allowed_run_ids=["a5e61c466d9c82dc", "28bf3f86a34814e0"], actions=["safe_stop", "unfinished_takeover"]),
        stop_operation=dict(run_id=run_id, trial_id=trial_id, action="stop", accepted=True, response_base_code=0,
                            observed_utc="2026-10-26T06:40:00+00:00"),
        terminal_observation=dict(run_id=run_id, trial_id=trial_id, platform_status="stopped", terminal_confirmed=True,
                                  observed_utc="2026-10-26T06:41:30.809216+00:00"))


@pytest.mark.parametrize("exit_code", [None, 0, 143])
def test_covariance_authorized_stop_admits_resume_and_preserves_exit(tmp_path, exit_code):
    document = covariance_authorized_stop_document(exit_code)
    path = tmp_path / "stopped.json"; path.write_text(json.dumps(document))
    request = dict(resume_execution=dict(run_id=document["run_id"], trial_id=document["trial_id"]),
        resume_checkpoints=[{}], resume_source_terminal=recovery.descriptor(path))
    result = recovery.validate_resume_terminal(request, tmp_path, dict(run_id="successor", trial_id="successor-trial"))
    assert result["status"] == "stopped" and result["evidence_kind"] == "authorized_operational_stop"
    assert result["original_execution"]["final_exit_code"] is None if exit_code is None else result["original_execution"]["final_exit_code"] == exit_code
    assert result["termination_kind"] == "user_authorized_stop"


@pytest.mark.parametrize("change", ["running", "unknown", "unauthorized", "scope", "operation", "terminal", "unconfirmed", "exit_bool", "same_successor"])
def test_covariance_authorized_stop_rejects_live_unknown_or_unbound_predecessor(tmp_path, change):
    document = covariance_authorized_stop_document()
    if change in ("running", "unknown"): document["platform_status"] = change
    elif change == "unauthorized": document.pop("user_authorization")
    elif change == "scope": document["user_authorization"]["allowed_run_ids"][0] = "other-run"
    elif change == "operation": document["stop_operation"]["trial_id"] = "other-trial"
    elif change == "terminal": document["terminal_observation"]["run_id"] = "other-run"
    elif change == "unconfirmed": document["terminal_observation"]["terminal_confirmed"] = False
    elif change == "exit_bool": document["final_exit_code"] = False
    path = tmp_path / "stopped.json"; path.write_text(json.dumps(document))
    source = dict(run_id="28bf3f86a34814e0", trial_id="438662058")
    request = dict(resume_execution=source, resume_checkpoints=[{}], resume_source_terminal=recovery.descriptor(path))
    successor = source if change == "same_successor" else dict(run_id="successor")
    with pytest.raises(ValueError): recovery.validate_resume_terminal(request, tmp_path, successor)


def test_authorized_stop_cannot_recompute_known_complete1000_result(tmp_path):
    document = covariance_authorized_stop_document(0); document["complete_groups"] = 1000
    path = tmp_path / "stopped.json"; path.write_text(json.dumps(document))
    request = dict(resume_execution=dict(run_id=document["run_id"], trial_id=document["trial_id"]),
        resume_checkpoints=[{}], resume_source_terminal=recovery.descriptor(path))
    with pytest.raises(ValueError, match="REUSE_FINAL_INTAKE"):
        recovery.validate_resume_terminal(request, tmp_path, dict(run_id="successor"))

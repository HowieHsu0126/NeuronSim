"""Frozen-cohort admission, native seeds and export recovery on disposable fixtures.

Synthetic qualification receipts are engineering fixtures. They do not freeze
repository scientific protocols or contribute any mechanistic episodes.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from . import benchmark_production as production
from .benchmark_dataset import read_episode
from .benchmark_tasks import v1_visual_config, ca3_commands, condition_for_episode
from . import ca3_native_observation as observer
from .circuit_campaign import campaign_config, freeze_identities
from .circuit_data import CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits, SCHEMA
from .provenance import sha256_file
from .v1_visual_controls import make_commands, CONTROL_FEATURES

CONFIG = Path(__file__).resolve().parents[1]/"configs/neural_population_dynamics.yaml"


def fixture_configuration(root, scenario):
    root.mkdir(exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text())
    specification = config["simulation_campaign"]["scenarios"][scenario]
    for name in ("parameter_domain", "task_protocol", "observation_map"):
        specification[name]["frozen"] = True
    specification["observation_map"]["scientific_bin_ticks"] = 1
    path = root/"campaign.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    proof = dict(status="PASS", scenario_id=scenario, coupled_core_neurons=230924 if scenario=="v1_visual" else 168106,
                 source_revision="synthetic-production-test-source", evidence=["disposable engineering fixture only"],
                 checks={name: True for name in ("source_reproduction", "numerical_integrity", "driver_statistics",
                                                 "control_delivery", "full_unit_recording", "coupling", "scaling_fidelity")})
    if scenario == "v1_visual":
        proof["source_recipes"] = dict(
            construction_config=yaml.safe_load((CONFIG.parent/"neural_population_v1_circuit_development.yaml").read_text()),
            visual_config=yaml.safe_load((CONFIG.parent/"neural_population_v1_constructed_visual_development.yaml").read_text()))
    qualification = root/"qualification.json"
    qualification.write_text(json.dumps(proof))
    table = root/"identities.csv"
    freeze_identities(campaign_config(path), scenario, qualification, table)
    return path, qualification, table


@pytest.fixture(scope="module", params=production.LEGACY_SCENARIOS)
def cohort(request, tmp_path_factory):
    path, qualification, table = fixture_configuration(tmp_path_factory.mktemp(request.param), request.param)
    return production.ProductionCohort.load(path, request.param, qualification, table)


def source_fixture(root, cohort, identity, *, source_document_padding=0):
    root.mkdir(parents=True)
    n = 230924 if cohort.scenario=="v1_visual" else 168106
    units = CircuitUnits("synthetic-native-circuit", np.arange(n), np.full(n,"E" if cohort.scenario=="v1_visual" else "CA3_Pyramidal"), np.zeros(n,dtype=np.int64),
                         ("E" if cohort.scenario=="v1_visual" else "CA3_Pyramidal",), "synthetic source map", cohort.qualification["source_revision"],
                         circuit_provenance=dict(connectivity_identity="synthetic-realized-graph",parameter_identity="synthetic-parameters"))
    record = dict(source_revision=units.source_revision,circuit_provenance=units.circuit_provenance,
                  complete_unit_recording=True,simulation_completed=True,core_unit_count=n)
    admission = dict(scenario_id=cohort.scenario,cohort_identity=identity,
                     source_qualification_sha256=cohort.qualification_sha256,
                     identity_table_sha256=cohort.identity_table_sha256,
                     construction_config=dict(benchmark_circuit_candidate=cohort.declaration(identity)))
    if cohort.scenario=="v1_visual":
        settings = cohort.v1_visual_config(identity)
        admission["construction_config"] = cohort.v1_construction_config(identity)
        documents = {}
        for role in ("recurrent", "inputs"):
            path = root/(role+".json")
            path.write_text(json.dumps(dict(original_config=admission["construction_config"], original_config_sha256="fixture-config",
                                           synthetic_construction_detail="x"*source_document_padding)))
            documents[role] = production.source_document(path)
        admission["source_documents"] = documents
        constructor_sha, inputs_sha = documents["recurrent"]["sha256"], documents["inputs"]["sha256"]
        controls = make_commands(settings["stimulus"])
        seeds = dict(native_kernel_rng_seed=143202461,lgn_starting_noise_seed=identity["stimulus_seed"],
                     background_starting_noise_seed=identity["background_seed"])
        record.update(recorded_until_tick=12000,expected_duration_ticks=12000,input_mode="constructed_visual_inputs",
                      initial_vm_identity="source-reset",native_report_sha256="native-report",
                      upstream_bindings=dict(spike_inputs=dict(config=settings),
                          recurrent=dict(files={"primary/report.json":dict(sha256=constructor_sha)}),
                          inputs=dict(files={"primary/report.json":dict(sha256=inputs_sha)})),
                      visual_commands=dict(source_command_representation_verified=True,config=settings))
        admission.update(native_report_sha256="native-report",construction_report_sha256=constructor_sha,input_report_sha256=inputs_sha)
        ep = CircuitEpisode("synthetic-native-episode",units.circuit_id,"synthetic-native-block","development",12000,.25,
            np.array([2000,2004,12000]),np.array([0,1,0]),controls["control_ticks"],controls["control_values"],
            CONTROL_FEATURES,seeds=seeds,source_record=record,privileged=dict(terminal_voltage=np.zeros(n)),task_outcome="failed_synthetic_target")
    else:
        seeds=dict(circuit_seed=identity["connectivity_seed"],uniform_seed=identity["background_seed"],
                   engine_seed=identity["initial_state_seed"],cue_seed=identity["stimulus_seed"])
        production.prepare_circuit(cohort, cohort.qualification_path.parent/"campaign.yaml",
                                   identity["circuit_index"], root/"prepared")
        prepared = root/"prepared/circuit.yaml"
        admission["construction_config"] = json.loads(json.dumps(yaml.safe_load(prepared.read_text())))
        formation = dict(realization_circuit_binding=dict(source_realization_config_sha256=sha256_file(prepared),
            circuit_id=identity["circuit_id"], circuit_seed=identity["connectivity_seed"], fresh_native_graph_constructed=True))
        formation_path = root/"formation.json"; formation_path.write_text(json.dumps(formation))
        admission.update(formation_report_sha256=sha256_file(formation_path),source_config_sha256=sha256_file(prepared),
            source_documents=dict(prepared_config=production.source_document(prepared), formation=production.source_document(formation_path)))
        (root/"native_run").mkdir()
        pattern=np.arange(2200,dtype=np.int32)%550
        pattern_path=root/"native_run/trainPat.csv"; pattern_path.write_text("\n".join(map(str,pattern))+"\n")
        formed=dict(formation_report_sha256=admission["formation_report_sha256"], normalized_input_sha256="synthetic-realized-graph",
                    pattern_sha256=sha256_file(pattern_path), populations=dict(CA3_Pyramidal=n))
        condition=condition_for_episode(cohort.campaign["scenarios"][cohort.scenario]["task_protocol"],cohort.scenario,identity)
        commands=ca3_commands(pattern,0,condition,identity["stimulus_seed"])
        np.savez_compressed(root/"ordered_cue_commands.npz",**commands)
        targets=np.unique(commands["unit_ids"])
        values=np.zeros((1001,len(targets)+3),dtype=np.float32)
        if len(targets):
            values[commands["time_ms"],np.searchsorted(targets,commands["unit_ids"])]=1
        record.update(recorded_until_tick=1001,expected_duration_ticks=1001, condition=condition, formation_binding=formed)
        ep = CircuitEpisode("synthetic-native-episode",units.circuit_id,"synthetic-native-block","development",1001,1.,
            np.array([0,520,1000]),np.array([0,1,0]),np.arange(1001),values,
            tuple(dict(name="fixture_current_"+str(i),unit="dimensionless") for i in range(values.shape[1])),
            seeds=seeds,source_record=record,privileged=dict(native_current=np.array([10000.])),task_outcome="failed_synthetic_target")
        report=dict(status="COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE", actual_split="development", duration_ms=1001,
            complete_episodes=1, units=n, events=3, condition=condition, candidate_identity=identity, formed_circuit=formed,
            unit_millisecond_counts_compared=n*1001, native_terminal_footer_verified=True, all_unit_event_counter_equality=True,
            independent_current_replay=dict(exact_float32_bits_equal=True))
        (root/"report.json").write_text(json.dumps(report))
        with (root/"native_observation.bin").open("wb") as stream:
            name=b"CA3_Pyramidal"
            stream.write(observer.HEADER.pack(observer.MAGIC,1,n))
            stream.write(observer.GROUP.pack(0,0,n,0,0,0,n,0,len(name))+name)

    with CircuitArchiveWriter(root/"dataset",dict(schema=SCHEMA,scope="source_reproduction",scenario_id=cohort.scenario,
                                       source_revision=units.source_revision,sample_unit="complete_mechanistic_episode")) as writer:
        writer.register_circuit(units)
        writer.add(ep)
    return CircuitDataset(root/"dataset",splits=("development",)),ep,admission


def test_complete_frozen_cohort_preserves_all_splits(cohort):
    assert len(cohort.identities)==100000
    assert cohort.identity(0,0,0)["split"]=="train"
    assert cohort.identity(70,0,0)["split"]=="validation"
    assert cohort.identity(80,0,0)["split"]=="calibration"
    assert cohort.identity(90,0,0)["split"]=="test"
    assert cohort.identity(99,99,9)["initial_state_seed"] not in (41,42,43)


@pytest.mark.parametrize("index",[0,70,80,90])
def test_writer_roundtrip_native_seed_roles_and_committed_resume(cohort,tmp_path,index):
    identity=cohort.identity(index,1,9)
    source,native,admission=source_fixture(tmp_path/"native",cohort,identity)
    original_catalog=sha256_file(source.root/"dataset.json")
    result=production.write_episode(cohort,identity,source,tmp_path/"production",admission=admission)
    assert result["status"]=="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
    actual=CircuitDataset(tmp_path/"production",splits=(identity["split"],),allow_test=identity["split"]=="test")
    ep=read_episode(actual,actual.episodes[identity["episode_id"]])
    assert ep.seeds=={k:identity[k] for k in production.COHORT_SEEDS}
    assert ep.source_record["native_seed_streams"]==native.seeds
    assert ep.source_record["native_identity"]["episode_id"]==native.episode_id
    assert ep.task_outcome=="failed_synthetic_target"
    if cohort.scenario=="ca3_pattern_completion":
        assert "not a microscopic" in ep.source_record["cohort_seed_semantics"]["initial_state_seed"]
    else:
        assert ep.source_record["cohort_seed_semantics"]["initial_state_seed"].startswith("unused;")
    np.testing.assert_array_equal(ep.spike_ticks,native.spike_ticks)
    np.testing.assert_array_equal(ep.control_values,native.control_values)
    assert sha256_file(source.root/"dataset.json")==original_catalog
    reused=production.completed_episode(cohort,identity,tmp_path/"production",resume=True)
    assert reused["status"]=="REUSED_COMMITTED_PRODUCTION_EPISODE" and reused["new_source_simulations"]==0
    assert actual.catalog["complete_episodes"]==1
    with pytest.raises(ValueError,match="EXPLICIT_RESUME"):
        production.completed_episode(cohort,identity,tmp_path/"production",resume=False)


def test_write_rejects_missing_or_rebound_admission_before_creating_output(cohort,tmp_path):
    identity=cohort.identity(0,1,0)
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity)
    invalid=deepcopy(admission)
    invalid["construction_config"]["benchmark_circuit_candidate"]["construction_scope"]="development"
    for proof,message in [({},"SOURCE_ADMISSION_REQUIRED"),(invalid,"FRESH_DECLARED_CONSTRUCTION_REQUIRED")]:
        with pytest.raises(ValueError,match=message):
            production.write_episode(cohort,identity,source,tmp_path/"rejected",admission=proof)
        assert not (tmp_path/"rejected").exists()


def test_qualification_changed_after_loading_is_rejected_before_writes(tmp_path):
    path,q,table=fixture_configuration(tmp_path,"v1_visual")
    loaded=production.ProductionCohort.load(path,"v1_visual",q,table)
    q.write_text(q.read_text()+" ")
    with pytest.raises(ValueError,match="QUALIFICATION_OR_COHORT_CHANGED"):
        loaded.writer(tmp_path/"rejected",loaded.qualification["source_revision"])
    assert not (tmp_path/"rejected").exists()


def test_cli_unfrozen_campaign_rejects_before_reading_table_or_writing(cohort,tmp_path):
    document=yaml.safe_load((cohort.qualification_path.parent/"campaign.yaml").read_text())
    document["simulation_campaign"]["scenarios"][cohort.scenario]["task_protocol"]["frozen"]=False
    config=tmp_path/"unfrozen_campaign.yaml";config.write_text(yaml.safe_dump(document,sort_keys=False))
    args=["--config",str(config),"--scenario",cohort.scenario,"--qualification",str(cohort.qualification_path),
          "--identity-table",str(tmp_path/"missing.csv"),"--circuit-index","0","prepare-circuit",
          "--base-config",str(CONFIG),"--output",str(tmp_path/"rejected")]
    with pytest.raises(ValueError,match="NOT_QUALIFIED.*task_protocol"):
        production.main(args)
    assert not (tmp_path/"rejected").exists()


def test_prepare_declares_production_without_model_or_source_execution(cohort,tmp_path):
    base=CONFIG.parent/"neural_population_v1_circuit_development.yaml" if cohort.scenario=="v1_visual" else cohort.qualification_path.parent/"campaign.yaml"
    result=production.prepare_circuit(cohort,base,70,tmp_path/"prepared")
    config=yaml.safe_load((tmp_path/"prepared/circuit.yaml").read_text())
    assert result["production_episodes"]==0 and result["source_simulations"]==0
    assert config["benchmark_circuit_candidate"]==cohort.declaration(cohort.identity(70,0,0))
    expected_seed=cohort.identity(70,0,0)["connectivity_seed"]
    if cohort.scenario=="v1_visual":
        assert config["builder_seed"]==expected_seed and config["glif_assignment_seed"]==expected_seed+1000000
    else:
        assert config["ca3_source_realization"]["circuit_seed"]==expected_seed
        assert config["ca3_source_realization"]["split"]=="development"


def test_committed_resume_skips_source_read_and_simulation(cohort,tmp_path,monkeypatch):
    identity=cohort.identity(0,1,0)
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity)
    output=tmp_path/"production"
    production.write_episode(cohort,identity,source,output,admission=admission)
    monkeypatch.setenv("SLURM_JOB_ID","synthetic")
    if cohort.scenario=="ca3_pattern_completion":
        monkeypatch.setenv("SLURM_JOB_GPUS","0")
        native=Mock(side_effect=AssertionError("completed source must not rerun"))
        from . import ca3_benchmark
        monkeypatch.setattr(ca3_benchmark,"run",native)
        result=production.run_ca3(cohort,identity,Path("unavailable-config"),Path("unavailable-runtime"),
            Path("unavailable-formation"),Path("unavailable-analysis"),output,Path("unavailable-current-replay"),resume=True)
        native.assert_not_called()
    else:
        monkeypatch.setenv("SLURMD_NODENAME","cpu03")
        monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
        reader=Mock(side_effect=AssertionError("completed source must not reread"))
        monkeypatch.setattr(production,"bound_json",reader)
        result=production.export_v1(cohort,identity,*[Path("missing") for _ in range(4)],output,resume=True)
        reader.assert_not_called()
    assert result["status"]=="REUSED_COMMITTED_PRODUCTION_EPISODE"


def test_retained_source_attempt_is_reconcilable_without_catalog(cohort,tmp_path):
    source=tmp_path/"production/source_execution/retained";source.mkdir(parents=True)
    (source/"report.json").write_text(json.dumps(dict(status="FAILED")))
    assert production.completed_episode(cohort,cohort.identity(0,0,0),source.parents[1],resume=True) is None
    (source.parents[1]/"unknown-side-effect").write_text("uncertain")
    with pytest.raises(ValueError,match="UNRESOLVED_EXECUTION_OUTPUT"):
        production.completed_episode(cohort,cohort.identity(0,0,0),source.parents[1],resume=True)


def test_authenticated_document_rejects_copied_config_relabel(cohort,tmp_path):
    identity=cohort.identity(0,1,0)
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity)
    invalid=deepcopy(admission)
    if cohort.scenario=="v1_visual":
        invalid["construction_config"]["edge_chunk_size"]+=1
    else:
        invalid["construction_config"]["simulation_campaign"]["sampling"]["circuit_instances"]-=1
    with pytest.raises(ValueError,match="AUTHENTICATED_CONSTRUCTION_CONFIG_DIFFERS"):
        production.write_episode(cohort,identity,source,tmp_path/"rejected",admission=invalid)
    assert not (tmp_path/"rejected").exists()
    invalid=deepcopy(admission)
    role="recurrent" if cohort.scenario=="v1_visual" else "formation"
    invalid["source_documents"][role]["payload"]+=" "
    with pytest.raises(ValueError,match="SOURCE_DOCUMENT_DIFFERS"):
        production.write_episode(cohort,identity,source,tmp_path/"rejected",admission=invalid)
    assert not (tmp_path/"rejected").exists()


def test_v1_driver_recipe_rejects_changed_background_rate(cohort,tmp_path):
    if cohort.scenario!="v1_visual":
        pytest.skip("V1-only driver check")
    identity=cohort.identity(0,1,0)
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity)
    changed=deepcopy(cohort.v1_visual_config(identity))
    changed["background"]["rate_hz"]+=1
    settings=cohort.v1_construction_config(identity)
    recurrent=dict(original_config=settings,original_config_sha256="fixture-config",
        rng=dict(builder_seed=identity["connectivity_seed"],glif_assignment_seed=identity["connectivity_seed"]+1000000))
    inputs=dict(original_config=settings,original_config_sha256="fixture-config",rng=dict(lgn_selection_python_seed=identity["connectivity_seed"]))
    native=dict(original_config_sha256="fixture-config",bindings=dict(spike_inputs=dict(config=changed)))
    with pytest.raises(ValueError,match="ACTUAL_CONTROL_OR_NOISE_DIFFERS"):
        production.bind_v1(cohort,identity,recurrent,inputs,native,dict(config=changed),dict(config=changed))


def test_ca3_retained_complete_source_uses_existing_reader_before_commit(cohort,tmp_path,monkeypatch):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only retained execution check")
    identity=cohort.identity(0,1,0)
    output=tmp_path/"production"
    retained=output/"source_execution"/identity["episode_id"]/"result"
    _,_,admission=source_fixture(retained,cohort,identity)
    prepared=retained/"prepared/circuit.yaml"
    formation=tmp_path/"formation";formation.mkdir()
    (formation/"native_report.json").write_text(admission["source_documents"]["formation"]["payload"])
    formed=json.loads((retained/"report.json").read_text())["formed_circuit"]
    monkeypatch.setattr(production.ca3_benchmark,"retrieval_realization",lambda *_:None)
    monkeypatch.setattr(production,"qualify_input",lambda *a,**k:formed)
    native=Mock(side_effect=AssertionError("retained native execution must not rerun"))
    monkeypatch.setattr(production.ca3_benchmark,"run",native)
    trace=Mock(return_value=dict(synthetic_trace_validation=True))
    monkeypatch.setattr(production,"verify_retained_ca3_trace",trace)
    result=production.run_ca3(cohort,identity,prepared,Path("unused-runtime"),formation,Path("unused-analysis"),output,Path("unused-replay"),resume=True)
    assert result["status"]=="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
    native.assert_not_called()
    trace.assert_called_once()
    assert (retained/"report.json").exists()


def test_ca3_retained_changed_commands_reject_before_commit(cohort,tmp_path,monkeypatch):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only retained execution check")
    identity=cohort.identity(0,1,0)
    output=tmp_path/"production";retained=output/"source_execution"/identity["episode_id"]/"result"
    _,_,admission=source_fixture(retained,cohort,identity)
    formation=tmp_path/"formation";formation.mkdir()
    (formation/"native_report.json").write_text(admission["source_documents"]["formation"]["payload"])
    formed=json.loads((retained/"report.json").read_text())["formed_circuit"]
    with np.load(retained/"ordered_cue_commands.npz") as saved:
        commands={k:saved[k] for k in saved.files}
    commands["time_ms"]=commands["time_ms"]+1
    np.savez_compressed(retained/"ordered_cue_commands.npz",**commands)
    monkeypatch.setattr(production.ca3_benchmark,"retrieval_realization",lambda *_:None)
    monkeypatch.setattr(production,"qualify_input",lambda *a,**k:formed)
    native=Mock(side_effect=AssertionError("retained native execution must not rerun"))
    monkeypatch.setattr(production.ca3_benchmark,"run",native)
    # condition 0 has no cue assignments; corrupt its saved key set instead.
    np.savez_compressed(retained/"ordered_cue_commands.npz",**commands,invalid_key=np.array([1]))
    with pytest.raises(ValueError,match="ORDERED_COMMANDS_DIFFER"):
        production.run_ca3(cohort,identity,retained/"prepared/circuit.yaml",Path("unused"),formation,Path("unused"),output,Path("unused"),resume=True)
    native.assert_not_called()
    assert not (output/"dataset.json").exists()


def test_ca3_fresh_attempt_uses_local_scratch_and_retains_result(cohort,tmp_path,monkeypatch):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only scratch execution check")
    identity=cohort.identity(0,1,0)
    fixture_root=tmp_path/"fixture"
    _,_,admission=source_fixture(fixture_root,cohort,identity)
    formation=tmp_path/"formation";formation.mkdir()
    (formation/"native_report.json").write_text(admission["source_documents"]["formation"]["payload"])
    formed=json.loads((fixture_root/"report.json").read_text())["formed_circuit"]
    monkeypatch.setattr(production.ca3_benchmark,"retrieval_realization",lambda *_:None)
    monkeypatch.setattr(production,"qualify_input",lambda *a,**k:formed)
    monkeypatch.setenv("SLURMD_NODENAME","gpu03")
    monkeypatch.setenv("SLURM_JOB_ID","synthetic-filesystem-test")
    monkeypatch.setenv("SLURM_JOB_GPUS","0")
    observed=[]
    def native_stub(packaged,formation,analysis,config,attempt,**kwargs):
        import shutil
        assert kwargs["production_identity"] == identity
        assert attempt.is_relative_to(Path("/tmp")) and not attempt.exists()
        observed.append(attempt)
        shutil.copytree(fixture_root,attempt)
        return json.loads((attempt/"report.json").read_text())
    monkeypatch.setattr(production.ca3_benchmark,"run",native_stub)
    output=tmp_path/"production"
    result=production.run_ca3(cohort,identity,fixture_root/"prepared/circuit.yaml",Path("unused"),formation,Path("unused"),output,Path("unused"))
    assert result["status"]=="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
    assert len(observed)==1 and not observed[0].parent.exists()
    assert (output/"source_execution"/identity["episode_id"]/"result/report.json").exists()


def test_ca3_failed_local_attempt_is_retained_and_resume_refuses_rerun(cohort,tmp_path,monkeypatch):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only scratch failure check")
    identity=cohort.identity(0,1,0)
    fixture_root=tmp_path/"fixture"
    _,_,admission=source_fixture(fixture_root,cohort,identity)
    formation=tmp_path/"formation";formation.mkdir()
    (formation/"native_report.json").write_text(admission["source_documents"]["formation"]["payload"])
    formed=json.loads((fixture_root/"report.json").read_text())["formed_circuit"]
    monkeypatch.setattr(production.ca3_benchmark,"retrieval_realization",lambda *_:None)
    monkeypatch.setattr(production,"qualify_input",lambda *a,**k:formed)
    monkeypatch.setenv("SLURMD_NODENAME","gpu03");monkeypatch.setenv("SLURM_JOB_ID","synthetic")
    monkeypatch.setenv("SLURM_JOB_GPUS","0")
    observed=[]
    def native_stub(packaged,formation,analysis,config,attempt,**kwargs):
        observed.append(attempt);attempt.mkdir()
        (attempt/"report.json").write_text(json.dumps(dict(status="FAILED_CA3_BENCHMARK_DEVELOPMENT",error="synthetic native failure")))
        raise RuntimeError("synthetic native failure")
    native=Mock(side_effect=native_stub);monkeypatch.setattr(production.ca3_benchmark,"run",native)
    output=tmp_path/"production"
    args=(cohort,identity,fixture_root/"prepared/circuit.yaml",Path("unused"),formation,Path("unused"),output,Path("unused"))
    with pytest.raises(RuntimeError,match="synthetic native failure"):
        production.run_ca3(*args)
    retained=output/"source_execution"/identity["episode_id"]
    assert (retained/"execution_error.json").exists() and (retained/"result/report.json").exists()
    assert not observed[0].parent.exists() and not (output/"dataset.json").exists()
    with pytest.raises(ValueError,match="COMPLETE_RECONCILED_EPISODE_REQUIRED"):
        production.run_ca3(*args,resume=True)
    assert native.call_count==1


def test_ca3_writer_rejects_different_condition_with_matching_seeds(cohort,tmp_path):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only source condition check")
    identity=cohort.identity(0,1,0)
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity)
    wrong=cohort.identity(0,1,1)
    # Condition identity varies within one noise block; stream seeds are equal.
    rebound=deepcopy(admission);rebound["cohort_identity"]=wrong
    with pytest.raises(ValueError,match="SOURCE_CONDITION_DIFFERS"):
        production.write_episode(cohort,wrong,source,tmp_path/"rejected",admission=rebound)
    assert not (tmp_path/"rejected").exists()


def test_large_authenticated_construction_reports_roundtrip(cohort,tmp_path):
    if cohort.scenario!="v1_visual":
        pytest.skip("V1-only large construction document check")
    identity=cohort.identity(0,1,0)
    # Actual construction reports are over 200 kB. Use synthetic padding to
    # exercise archive metadata storage without relabeling any real data.
    source,_,admission=source_fixture(tmp_path/"native",cohort,identity,source_document_padding=256000)
    result=production.write_episode(cohort,identity,source,tmp_path/"production",admission=admission)
    assert result["status"]=="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
    assert production.completed_episode(cohort,identity,tmp_path/"production",resume=True)["new_source_simulations"]==0


def test_retained_ca3_trace_rejects_truncated_native_record(cohort,tmp_path):
    if cohort.scenario!="ca3_pattern_completion":
        pytest.skip("CA3-only retained native trace check")
    identity=cohort.identity(0,1,0)
    source_fixture(tmp_path/"native",cohort,identity)
    bound=production.bind_ca3_episode(tmp_path/"native")
    engine=tmp_path/"host_current_replay";engine.write_text("synthetic engine identity only")
    engine.with_name("host_current_replay_build.json").write_text(json.dumps(dict(status="BUILT_AND_VALIDATED",binary_sha256=sha256_file(engine))))
    with pytest.raises(ValueError):
        production.verify_retained_ca3_trace(bound,engine,None)

"""Full 150-cell cohort validation with small arrays and explicit clocks."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import pfc_reference_quality_summary as summary
from .provenance import sha256_file


def write_json(path,value):
    path.write_text(json.dumps(value))


def attempt_report(root, job, **changes):
    directory = root / ("run_" + job)
    directory.mkdir(parents=True)
    report = dict(job_id=job, task_index=14, node="cpu05", status="FAILED",
                  exit_code=1, error="original failure", source_qualification_granted=False,
                  production_episodes=0)
    report.update(changes)
    write_json(directory / "report.json", report)
    return directory


def test_startup_failure_retains_original_error_and_source_attempt_is_not_duplicated(tmp_path):
    executions, readouts = tmp_path / "executions", tmp_path / "readouts"
    failed = attempt_report(executions, "101", error="SOURCE_MODULE_CHANGED")
    represented = executions / "run_102"
    represented.mkdir()  # A recovered scheduler timeout has no outer report.
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    startups, readers = summary.auxiliary_attempt_failures(executions, readouts, expected, {"102"})
    assert readers == [] and len(startups) == 1
    assert startups[0]["job_id"] == "101" and startups[0]["error"] == "SOURCE_MODULE_CHANGED"
    assert startups[0]["episode_id"] == "episode14" and startups[0]["exit_code"] == 1
    assert startups[0]["report_sha256"] == sha256_file(failed / "report.json")


def test_multiple_failed_readouts_retain_producer_and_exclude_success(tmp_path):
    executions, readouts = tmp_path / "executions", tmp_path / "readouts"
    attempt_report(executions, "101", status="COMPLETE_SOURCE_READOUT_FAILED")
    attempt_report(readouts, "102", producer_job="101", error="RESOURCE_BLOCKED")
    attempt_report(readouts, "103", producer_job="101", error="READOUT_DATA_CHANGED")
    attempt_report(readouts, "104", producer_job="101",
                   status="PASS_COMPLETE_REFERENCE_QUALITY_READOUT", exit_code=0, error=None)
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    startups, readers = summary.auxiliary_attempt_failures(executions, readouts, expected, {"101"})
    assert startups == []
    assert [(row["job_id"], row["producer_job"], row["error"]) for row in readers] == [
        ("102", "101", "RESOURCE_BLOCKED"), ("103", "101", "READOUT_DATA_CHANGED")]


@pytest.mark.parametrize("readout", [False, True])
def test_unfinished_auxiliary_attempt_rejects_missing_terminal_report(tmp_path, readout):
    executions, readouts = tmp_path / "executions", tmp_path / "readouts"
    ((readouts if readout else executions) / "run_101").mkdir(parents=True)
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    with pytest.raises(ValueError, match="ATTEMPT_TERMINAL_REPORT_REQUIRED"):
        summary.auxiliary_attempt_failures(executions, readouts, expected, set())


@pytest.mark.parametrize("changes,match", [
    ({"job_id": "foreign"}, "ATTEMPT_IDENTITY_OR_SCOPE"),
    ({"task_index": 15}, "ATTEMPT_IDENTITY_OR_SCOPE"),
    ({"source_qualification_granted": True}, "ATTEMPT_IDENTITY_OR_SCOPE"),
    ({"producer_job": "foreign"}, "READOUT_PRODUCER_NOT_BOUND"),
    ({"status": "RUNNING"}, "READOUT_TERMINAL_FAILURE"),
    ({"status": "PASS_COMPLETE_REFERENCE_QUALITY_READOUT", "exit_code": 1}, "SUCCESSFUL_READOUT_EXIT"),
])
def test_auxiliary_readout_identity_and_terminal_status_reject_conflicts(tmp_path, changes, match):
    readouts = tmp_path / "readouts"
    values = dict(producer_job="101")
    values.update(changes)
    attempt_report(readouts, "102", **values)
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    with pytest.raises(ValueError, match=match):
        summary.auxiliary_attempt_failures(tmp_path / "executions", readouts, expected, {"101"})


def test_source_success_without_batch_is_not_a_startup_failure(tmp_path):
    executions = tmp_path / "executions"
    attempt_report(executions, "101", status="COMPLETE_SOURCE_TASK", exit_code=0, error=None)
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    with pytest.raises(ValueError, match="STARTUP_FAILURE_WITHOUT_SOURCE_BATCH_REQUIRED"):
        summary.auxiliary_attempt_failures(executions, tmp_path / "readouts", expected, set())


@pytest.mark.parametrize("complete", [True, False])
def test_interrupted_readout_requires_existing_completed_source_timeout(tmp_path, complete):
    executions, readouts = tmp_path / "executions", tmp_path / "readouts"
    (readouts / "run_101").mkdir(parents=True)
    attempt_report(readouts, "102", producer_job="101",
                   status="PASS_COMPLETE_REFERENCE_QUALITY_READOUT", exit_code=0, error=None)
    outcome = dict(raw_job_id="101", state="TIMEOUT", node="cpu05",
                   source_execution_complete=complete,
                   native_source_step=dict(state="COMPLETED", exit_code="0:0"))
    expected = dict(task_index=14, identity=dict(episode_id="episode14"))
    kwargs = dict(scheduler_timeouts={"101": dict(path="terminal.json", sha256="bound", terminal=outcome)})
    if not complete:
        with pytest.raises(ValueError, match="INTERRUPTED_READOUT_TIMEOUT_BINDING_REQUIRED"):
            summary.auxiliary_attempt_failures(executions, readouts, expected, {"101"}, **kwargs)
    else:
        startups, readers = summary.auxiliary_attempt_failures(executions, readouts, expected, {"101"}, **kwargs)
        assert startups == [] and len(readers) == 1
        assert readers[0]["status"] == "READOUT_INTERRUPTED_BY_SCHEDULER_TIMEOUT"
        assert readers[0]["scheduler_terminal"]["terminal"] == outcome


def source_metadata_fixture(source, report, row, config):
    """Supply complete source metadata; arrays keep the existing private math scale."""
    identity=row["identity"]
    driver=dict(seeds=dict(initial_rng_seed=identity["initial_state_seed"],external_noise_seed=identity["background_seed"]),
                random_stream_policy="independent_initial_external",packed_draw_order="tick_major_I_then_E",
                n_e=80000,n_i=20000,dt_seconds=.0125*.001,ticks_generated=400000,total_ticks=400000,
                source_entry_sha256="source",numpy_version="fixture",external_event_totals=dict(E=identity["block_index"],I=2),
                binomial_n=1000,binomial_probability=1.4*(.0125*.001),external_rate_Hz=1.4)
    initial=hashlib.sha256((identity["block_id"]+"initial").encode()).hexdigest()
    final=hashlib.sha256((identity["block_id"]+"final").encode()).hexdigest()
    report.update(initial_voltage=dict(file_sha256=initial,units="mV",declared_mean=-55.,declared_variance=100./12.,
                                       observed_mean=-55.,observed_variance=(100000-1)/100000*100./12.),
                  external_input=dict(driver=driver,final_rng_sha256=final),
                  source_checks=dict(complete_source_input_consumption=dict(passed=True)),source_cost=dict(wall_seconds=1.0))
    code=dict(executor_sha256="executor",drive_sha256="drive",kernel_parameterization_sha256="kernel")
    qualification=dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",backend="numpy",validated_dt_ms=[.0125],
        validated_kernel_parameters=[summary.resolve_kernel_parameters(config)],input_replay_status="PASS_NATIVE_NUMPY_DRIVE_REPLAY",
        input_replay_passed_comparisons=24,source_entry_sha256="source",numpy_version="fixture",
        executor_sha256="executor",drive_sha256="drive",kernel_parameterization_sha256="kernel")
    qpath=source/"qualification/comparison/qualification.json";qpath.parent.mkdir(parents=True);write_json(qpath,qualification)
    qentry=dict(path="comparison/qualification.json",bytes=qpath.stat().st_size,sha256=sha256_file(qpath))
    qdelivery=source/"qualification/delivery.json";write_json(qdelivery,dict(files=[qentry]))
    native=dict(status="PFC_SOURCE_EXECUTION_COMPLETE",source_execution_complete=True,case_id=identity["episode_id"],
        config_sha256=row["config_sha256"],resolved_config=config,validations=report["source_checks"],
        simulator_cost=report["source_cost"],driver=driver,source_entry_sha256="source",backend="numpy",code=code,
        circuit_provenance=summary.circuit_provenance(config,"source"),
        backend_qualification=dict(qualification=qualification,receipt_sha256=qentry["sha256"],delivery_sha256=sha256_file(qdelivery)),
        outputs={"initial_voltage.npy":dict(path="initial_voltage.npy",bytes=800128,sha256=initial),
                 "final_rng.npz":dict(path="final_rng.npz",bytes=3811,sha256=final)})
    native_path=source/"source_run/report.json";native_path.parent.mkdir();write_json(native_path,native)
    files=[dict(path=str(path.relative_to(source)),bytes=path.stat().st_size,sha256=sha256_file(path))
           for path in (qpath,qdelivery,native_path)]
    files.extend(dict(value,path="source_run/"+name) for name,value in native["outputs"].items())
    delivery=source/"delivery.json";value=json.loads(delivery.read_text());value["files"]=files;write_json(delivery,value)
    report["source_delivery_sha256"]=sha256_file(delivery)


@pytest.fixture
def cohort(tmp_path):
    preparation=tmp_path/"prepared";preparation.mkdir()
    settings=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/neural_population_pfc_reference_source_qualification.yaml").read_text())
    rows=[];mapping={}
    for circuit in summary.CIRCUITS:
        for block in summary.BLOCKS:
            for k,condition in enumerate(summary.CONDITIONS):
                episode=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{k:02d}"
                split="train" if circuit<70 else "validation" if circuit<80 else "calibration"
                identity=dict(scenario_id="pfc_working_memory",circuit_id=f"pfc_working_memory_circuit_{circuit:03d}",circuit_index=circuit,
                    episode_id=episode,block_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}",block_index=block,condition_index=k,
                    split=split,connectivity_seed=200000000+circuit,initial_state_seed=210000000+circuit*1000+block,
                    background_seed=220000000+circuit*1000+block,stimulus_seed=230000000+circuit*1000+block,sample_unit="complete_mechanistic_episode")
                # Full contract scale is kept in the recipe; private math checks use 16E/4I arrays.
                config=dict(benchmark_episode_candidate=dict(identity=identity,condition_id=condition,production_eligible=False),
                    dt_ms=.0125,duration_ms=5000.,excitatory_neurons=80000,inhibitory_neurons=20000,weight_scaling_factor=.0256)
                configpath=preparation/(episode+".yaml");configpath.write_text(yaml.safe_dump(config))
                row=dict(identity=identity,condition=dict(condition_id=condition),task_index=len(rows),config_path=configpath.name,
                    config_sha256=sha256_file(configpath),qualification="bound")
                rows.append(row)
                directory=tmp_path/episode;directory.mkdir();mapping[episode]=str(directory)
                source=directory/"source";source.mkdir();write_json(source/"delivery.json",dict(status="ARCHIVED_PFC_SOURCE_EXECUTION",case_id=episode,exit_code=0,source_execution_complete=True,result_status="PFC_SOURCE_EXECUTION_COMPLETE",scientific_fidelity_qualified=False,production_episodes=0,files=[]))
                dense=np.zeros((1000,20),dtype=np.int64)
                # Independent blocks differ by one count; cue and distractor interact.
                cue=k<5;distractor=k%5
                count=block+(2 if cue else 0)+distractor+(3*distractor if cue else 0)
                dense[0,0]=count;dense[1,16]=2*count
                unit_counts=dense.sum(axis=0,keepdims=True)
                fields=np.zeros((1000,17),dtype=np.int64);fields[:,0]=dense[:,0];fields[:,16]=dense[:,16]
                arrays=dict(phase_unit_counts=unit_counts,temporal_unit_mean_counts=dense.mean(axis=0),
                    temporal_unit_variance_counts=dense.var(axis=0,ddof=1),population_field_counts=fields,
                    phase_edges_ticks=np.array([0,400000],dtype=np.int64))
                np.savez(directory/"unit_population_statistics.npz",**arrays)
                rate=count/16/5
                phases=[dict(circuit_index=circuit,block_index=block,episode_id=episode,condition_id=condition,phase_index=0,
                    start_tick=0,stop_tick=400000,start_ms=0.,stop_ms=5000.,excitatory_spikes=count,inhibitory_spikes=count*2,
                    **dict({name:rate for name in summary.RATE_FIELDS},inhibitory_rate_hz=count*2/4/5,all_unit_rate_hz=count*3/20/5),direction_defined=False,decoded_angle_deg=None,
                    circular_cue_error_deg=None,direction_missing_reason="silent_or_resultant_below_existing_roundoff_rule")]
                summary.write_rows(directory/"phase_responses.csv",phases)
                temporal=[]
                for index in range(1000):
                    e=count if index==0 else 0
                    temporal.append(dict(circuit_index=circuit,block_index=block,episode_id=episode,
                        condition_id=condition,start_tick=index*400,stop_tick=(index+1)*400,
                        excitatory_spikes=e,inhibitory_spikes=2*count if index==1 else 0,
                        vector_x_spikes=float(e),vector_y_spikes=0.,
                        direction_defined=e>0,decoded_angle_deg=0. if e else None))
                summary.write_rows(directory/"temporal_responses.csv",temporal)
                report=dict(status="PASS_REFERENCE_SOURCE_QUALITY_READOUT",identity=identity,condition=row["condition"],
                    core_units=20,duration_ticks=400000,dt_ms=.0125,temporal_bins=1000,phase_count=1,events=count*3+1,terminal_tick_events=1,
                    config_sha256=row["config_sha256"],source_archive=str(source),source_delivery_sha256=sha256_file(source/"delivery.json"),
                    source_qualification_granted=False,scientific_claim_qualified=False,production_episodes=0,
                    outputs={name:dict(path=name,sha256=sha256_file(directory/name)) for name in ("phase_responses.csv","temporal_responses.csv","unit_population_statistics.npz")})
                source_metadata_fixture(source,report,row,config)
                write_json(directory/"report.json",report)
    plan=dict(schema="pfc_reference_quality_execution_v1",scope="development",production_episodes=0,source_quality_qualified=False,
        settings=settings,rows=rows,campaign_sha256="campaign",settings_sha256="settings",parameter_design_sha256="parameters")
    write_json(preparation/"execution.json",plan)
    manifest=tmp_path/"readouts.json";write_json(manifest,mapping)
    return preparation,manifest,mapping,plan


def test_complete_150_cell_aggregation_has_paired_effects_and_full_undefined_denominator(cohort,tmp_path):
    preparation,manifest,mapping,plan=cohort
    assert len(summary.validate_plan(preparation)["rows"])==150
    assert len(summary.readout_mapping(manifest))==150
    report=summary._aggregate(preparation,manifest,tmp_path/"summary",core_units=20,n_e=16)
    assert report["complete_episodes"]==150 and len(report["bindings"])==150
    assert len(report["realized_paired_blocks"])==15
    with (tmp_path/"summary/initial_input_statistics.csv").open() as stream: inputs=list(csv.DictReader(stream))
    assert len(inputs)==15 and len({row["block_id"] for row in inputs})==15
    assert all(row["independent_input_blocks"]=="1" and row["paired_conditions"]=="10" for row in inputs)
    with (tmp_path/"summary/external_input_statistics.csv").open() as stream: external=list(csv.DictReader(stream))
    assert len(external)==30 and {row["population"] for row in external}=={"E","I"}
    assert not report["source_qualification_granted"] and report["production_episodes"]==0
    with (tmp_path/"summary/paired_contrast_summary.csv").open() as stream: rows=list(csv.DictReader(stream))
    selected=[r for r in rows if r["contrast"]=="distractor_difference_of_differences" and r["condition_id"]=="cue_aligned"]
    assert selected
    for row in selected:
        expected=3*2/4/5 if row["metric"]=="inhibitory_rate_hz" else 3*3/20/5 if row["metric"]=="all_unit_rate_hz" else 3/16/5
        assert float(row["mean"])==pytest.approx(expected)
        assert float(row["sample_sd"])==pytest.approx(0,abs=1e-16)
        assert row["total_blocks"]==row["defined_blocks"]=="3"
    with (tmp_path/"summary/phase_block_summary.csv").open() as stream: phases=list(csv.DictReader(stream))
    angles=[r for r in phases if r["metric"]=="absolute_circular_cue_error_deg"]
    assert len(angles)==50 and all(r["undefined_blocks"]=="3" and r["mean"]==r["sample_sd"]=="" for r in angles)
    with np.load(tmp_path/"summary/unit_variance_statistics.npz") as arrays:
        values=arrays["circuit_023_condition_00_between_block_phase_unit_variance_counts2"]
        assert values.shape==(1,20) and values[0,0]==1 and values[0,16]==4
    assert "no confidence intervals" in report["uncertainty"]
    with (tmp_path/"summary/lagged_angular_displacement.csv").open() as stream: angular=list(csv.DictReader(stream))
    assert len(angular)==450 and {r["lag_ms"] for r in angular}=={"5.0","10.0","100.0"}
    assert all(r["mean_signed_displacement_deg"]=="" and r["undefined_pairs"]==r["candidate_pairs"] for r in angular)


def test_covariance_axes_distinguish_temporal_from_matched_block_variance():
    x=np.array([[[0.,0.],[2.,4.]],[[1.,3.],[3.,7.]],[[2.,6.],[4.,10.]]])
    temporal,between=summary.covariance_axes(x)
    np.testing.assert_allclose(temporal,[[2,4],[4,8]])
    np.testing.assert_allclose(between,[[1,3],[3,9]])
    for groups in ([0],[1],[0,1]):
        np.testing.assert_allclose(temporal[np.ix_(groups,groups)].sum(),x[:,:,groups].sum(axis=2).var(axis=1,ddof=1).mean())
        np.testing.assert_allclose(between[np.ix_(groups,groups)].sum(),x[:,:,groups].sum(axis=2).var(axis=0,ddof=1).mean())


def test_public_entry_point_rejects_small_unit_readout(cohort,tmp_path):
    preparation,manifest,_,_=cohort
    with pytest.raises(ValueError,match="UNIT_OR_CLOCK_MISMATCH"):summary.aggregate(preparation,manifest,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_missing_and_duplicate_cells_withhold_all_outputs(cohort,tmp_path):
    preparation,manifest,mapping,plan=cohort
    first=plan["rows"][0]["identity"]["episode_id"]
    removed=dict(mapping);removed.pop(first);write_json(manifest,removed)
    with pytest.raises(ValueError,match="INCOMPLETE_COHORT"):summary._aggregate(preparation,manifest,tmp_path/"missing",core_units=20,n_e=16)
    assert not (tmp_path/"missing").exists()
    write_json(manifest,list(mapping.values())+[mapping[first]])
    with pytest.raises(ValueError,match="DUPLICATE_READOUT_DIRECTORY"):summary.readout_mapping(manifest)


@pytest.mark.parametrize("field,value,match",[("dt_ms",.025,"UNIT_OR_CLOCK"),("core_units",19,"UNIT_OR_CLOCK"),
    ("status","FAILED","FAILED_OR_STALE"),("config_sha256","changed","FAILED_OR_STALE")])
def test_failed_stale_or_clock_mismatch_rejection(cohort,field,value,match):
    _,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]]);p=directory/"report.json"
    report=json.loads(p.read_text());report[field]=value;write_json(p,report)
    with pytest.raises(ValueError,match=match):summary.load_cell(directory,row,core_units=20)


def test_altered_bound_array_delivery_and_recipe_values_are_rejected(cohort):
    preparation,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]])
    path=directory/"unit_population_statistics.npz"
    original=path.read_bytes();path.write_bytes(original+b"altered")
    with pytest.raises(ValueError,match="BOUND_READOUT_CHANGED"):summary.load_cell(directory,row,core_units=20)
    path.write_bytes(original)
    delivery=directory/"source/delivery.json";delivery.write_text("{}")
    with pytest.raises(ValueError,match="SOURCE_DELIVERY_CHANGED"):summary.load_cell(directory,row,core_units=20)
    recipe=preparation/row["config_path"];recipe.write_text(recipe.read_text()+"\n")
    with pytest.raises(ValueError,match="BOUND_RECIPE_CHANGED"):summary.validate_plan(preparation)


def test_cli_incomplete_cohort_returns_nonzero_with_explicit_status(cohort,tmp_path,capsys):
    preparation,manifest,mapping,plan=cohort
    mapping.pop(plan["rows"][0]["identity"]["episode_id"]);write_json(manifest,mapping)
    with pytest.raises(SystemExit) as error:
        summary.main(["--preparation",str(preparation),"--readouts",str(manifest),"--output",str(tmp_path/"out")])
    assert error.value.code==1
    report=json.loads(capsys.readouterr().out)
    assert report["authoritative_aggregate_withheld"] and not (tmp_path/"out").exists()


def test_descriptive_does_not_drop_one_undefined_block():
    result=summary.descriptive([1,None,3])
    assert result==dict(total_blocks=3,defined_blocks=2,undefined_blocks=1,mean=None,sample_sd=None)


def test_rehashed_altered_phase_rate_does_not_override_count_semantics(cohort):
    _,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]])
    path=directory/"phase_responses.csv"
    with path.open() as stream: rows=list(csv.DictReader(stream))
    rows[0]["excitatory_rate_hz"]="99"
    path.unlink();summary.write_rows(path,rows)
    p=directory/"report.json";report=json.loads(p.read_text())
    report["outputs"][path.name]["sha256"]=sha256_file(path);write_json(p,report)
    with pytest.raises(AssertionError):summary.load_cell(directory,row,core_units=20)


def test_rehashed_failed_delivery_does_not_override_source_completion(cohort):
    _,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]])
    delivery=directory/"source/delivery.json";value=json.loads(delivery.read_text())
    value["source_execution_complete"]=False;write_json(delivery,value)
    p=directory/"report.json";report=json.loads(p.read_text());report["source_delivery_sha256"]=sha256_file(delivery);write_json(p,report)
    with pytest.raises(ValueError,match="INCOMPLETE_OR_WRONG_SOURCE_DELIVERY"):summary.load_cell(directory,row,core_units=20)


@pytest.mark.parametrize("field",["source_checks","source_cost","external_input","initial_voltage"])
def test_copied_source_metadata_must_match_authenticated_native_report(cohort,field):
    _,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]])
    path=directory/"report.json";report=json.loads(path.read_text())
    if field=="source_checks":report[field]["complete_source_input_consumption"]["passed"]=False
    elif field=="source_cost":report[field]["wall_seconds"]=2.
    elif field=="external_input":report[field]["driver"]["seeds"]["external_noise_seed"]+=1
    else:report[field]["file_sha256"]="changed"
    write_json(path,report)
    with pytest.raises(ValueError,match="SOURCE_CHECKS_OR_COST|INPUT_METADATA|REALIZED_INPUT_BINDING"):
        summary.load_cell(directory,row,core_units=20)


def test_altered_native_metadata_cannot_be_read_without_its_delivery_binding(cohort):
    _,_,mapping,plan=cohort
    row=plan["rows"][0];directory=Path(mapping[row["identity"]["episode_id"]])
    path=directory/"source/source_run/report.json";native=json.loads(path.read_text());native["driver"]["ticks_generated"]=1
    write_json(path,native)
    with pytest.raises(ValueError,match="SOURCE_METADATA_CHANGED"):
        summary.load_cell(directory,row,core_units=20)


@pytest.mark.parametrize("changed",["initial","rng","totals"])
def test_paired_contrasts_require_realized_input_pairing(cohort,changed):
    _,_,mapping,plan=cohort
    rows=plan["rows"][:10]
    reports=[json.loads((Path(mapping[row["identity"]["episode_id"]])/"report.json").read_text()) for row in rows]
    assert len(summary.paired_block_binding(rows,reports)["episode_ids"])==10
    if changed=="initial":reports[-1]["initial_voltage"]["file_sha256"]="other"
    elif changed=="rng":reports[-1]["external_input"]["final_rng_sha256"]="other"
    else:reports[-1]["external_input"]["driver"]["external_event_totals"]["E"]+=1
    with pytest.raises(ValueError,match="REALIZED_CONDITION_PAIRING_CHANGED"):
        summary.paired_block_binding(rows,reports)


def test_matched_time_unit_statistics_distinguishes_shared_and_single_unit_fluctuations():
    dense = np.zeros((3,1000,20),dtype=np.int64)
    dense[:,0,0] = [1,2,3]
    dense[:,0,1] = [1,2,3]
    dense[:,1,16] = [1,2,3]
    events = []
    fields = np.zeros((3,1000,17),dtype=np.int64)
    mapping = np.r_[np.arange(16),np.full(4,16)]
    for block in range(3):
        bins,ids = np.nonzero(dense[block])
        repeated_ids = np.repeat(ids,dense[block,bins,ids])
        repeated_bins = np.repeat(bins,dense[block,bins,ids])
        order = np.argsort(repeated_ids,kind="stable")
        events.append((repeated_ids[order],repeated_bins[order]))
        for unit in range(20):fields[block,:,mapping[unit]] += dense[block,:,unit]
    result = summary.matched_time_unit_statistics(events,dense.sum(axis=1)[:,None,:],fields,core_units=20,n_e=16)
    expected = dense.var(axis=0,ddof=1).mean(axis=0)
    np.testing.assert_array_equal(result["unit_variance"],expected)
    e = result["population"]["excitatory"]
    assert e["between_block_matched_time_diagonal_contribution_counts2"] == pytest.approx(2*.001/16**2)
    assert e["between_block_matched_time_ordered_offdiagonal_contribution_counts2"] == pytest.approx(2*.001/16**2)
    assert result["population"]["inhibitory"]["between_block_matched_time_ordered_offdiagonal_contribution_counts2"] == pytest.approx(0.)


def input_report():
    """Exact theoretical means allow independent checks of normalization."""
    return dict(identity=dict(circuit_index=23,block_index=3,block_id="circuit23_block3",split="train",
                initial_state_seed=210023003,background_seed=220023003),
                source_archive="bound_source",source_delivery_sha256="source_delivery",
                initial_voltage=dict(units="mV",declared_mean=-55.,declared_variance=100./12.,
                    observed_mean=-55.,observed_variance=99999/100000*100./12.,file_sha256="initial"),
                external_input=dict(final_rng_sha256="rng",driver=dict(n_e=80000,n_i=20000,total_ticks=400000,
                    ticks_generated=400000,seeds=dict(initial_rng_seed=210023003,external_noise_seed=220023003),
                    binomial_n=1000,binomial_probability=1.4*(.0125*.001),dt_seconds=.0125*.001,
                    external_rate_Hz=1.4,external_event_totals=dict(E=560000000,I=140000000))))


def test_input_sampler_moments_and_per_afferent_rate_use_one_block():
    initial,external=summary.input_statistics(input_report())
    assert initial["units"]==100000 and initial["independent_input_blocks"]==1
    assert initial["standardized_mean_deviation"]==0
    assert initial["ideal_expected_saved_variance_mV2"]==pytest.approx(99999/100000*100./12.)
    for row,trials,expected in zip(external,(32000000000000,8000000000000),(560000000,140000000)):
        assert row["ideal_bernoulli_trials"]==trials and row["binomial_draws"]==trials//1000
        assert row["ideal_expected_total_events"]==pytest.approx(expected)
        assert row["ideal_total_event_variance"]==pytest.approx(expected*(1-1.75e-5))
        assert row["standardized_total_deviation"]==pytest.approx(0,abs=1e-10)
        assert row["observed_rate_hz_per_afferent"]==pytest.approx(1.4)
        assert row["duration_seconds"]==5. and row["independent_input_blocks"]==1
        assert not {"p_value","confidence_interval","accepted","source_qualification_granted"}&row.keys()


def test_input_diagnostics_keep_unfavorable_standardized_deviations():
    report=input_report()
    report["initial_voltage"]["observed_mean"]=-54.9
    report["external_input"]["driver"]["external_event_totals"]["E"]+=1000000
    initial,external=summary.input_statistics(report)
    assert initial["standardized_mean_deviation"]==pytest.approx(.1/math.sqrt((100./12.)/100000))
    e=external[0]
    assert e["standardized_total_deviation"]==pytest.approx(1000000/math.sqrt(560000000*(1-1.75e-5)))
    assert len(external)==2


@pytest.mark.parametrize("changed",["units","variance","initial_mean","external_count","seed"])
def test_invalid_input_statistics_fail_explicitly(changed):
    report=input_report()
    if changed=="units":report["initial_voltage"]["units"]="V"
    elif changed=="variance":report["initial_voltage"]["observed_variance"]=-1
    elif changed=="initial_mean":report["initial_voltage"]["observed_mean"]=float("nan")
    elif changed=="external_count":report["external_input"]["driver"]["external_event_totals"]["E"]=-1
    else:report["external_input"]["driver"]["seeds"]["external_noise_seed"]+=1
    with pytest.raises(ValueError,match="INITIAL_MOMENT_METADATA|EXTERNAL_TOTAL_SUPPORT|INPUT_STATISTICS_IDENTITY"):
        summary.input_statistics(report)

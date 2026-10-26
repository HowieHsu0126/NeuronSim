"""Auxiliary nesting, covariance estimands, and source identity regressions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from . import pfc_refinement_repeat_comparison as repeat
from .pfc_coupled_precision import summarize_endpoints
from .test_pfc_coupled_precision import margin
from .pfc_confirmation_uncertainty import UNDEFINED


def endpoint(index=0, family="source_response"):
    keys = dict(condition="cue", metric="all_unit_rate_hz")
    if family == "field_lag_covariance":
        keys = dict(condition="cue",leading_field=0,trailing_field=0,lag_ms=0)
    r=margin(family,index=index,**keys)
    return r


def summary_rows(margins, old_scores, new_scores):
    old, new = [], []
    for r,a,b in zip(margins,old_scores,new_scores):
        kind = "covariance_jackknife_pseudovalue" if r["family"] == "field_lag_covariance" else "paired_block_linear_difference"
        old.append(dict(r,coupled_signed_estimate=str(a.mean()),coupled_score_kind=kind))
        new.append(dict(r,signed_estimate=b.mean(),score_kind=kind))
    return old,new


def test_cancelling_auxiliary_changes_are_visible_and_keep32_blocks():
    a=np.tile([-1.,1.],16)[None,:]
    b=-a
    old,new=summary_rows([endpoint()],a,b)
    rows,ids,midpoints,deltas=repeat.compare_scores(old,new,a,b)
    assert rows[0]["estimate_change"] == 0
    assert rows[0]["observed_block_mean_absolute_change"] == 2
    assert rows[0]["observed_block_rms_change"] == 2
    assert rows[0]["original_independent_blocks"] == 32
    assert rows[0]["auxiliary_realizations"] == 2
    np.testing.assert_array_equal(midpoints,np.zeros((1,32)))
    np.testing.assert_array_equal(deltas,b-a)
    np.testing.assert_array_equal(ids,[0])


def test_covariance_points_remain_separate_and_do_not_create_block_scatter():
    fields={l:{c:np.zeros((32,1000,17)) for c in repeat.CONDITIONS} for l in repeat.LABELS}
    responses={l:np.zeros((3,32,5,8)) for l in repeat.LABELS}
    cdfs={l:np.ones((3,32,5,3,17,1)) for l in repeat.LABELS}
    a=np.tile([0.,2.],16)
    fields[repeat.LABELS[0]]["cue"][:,20:60,0]=a[:,None]
    m=[endpoint(family="field_lag_covariance")]
    oldrows,oldscore=summarize_endpoints(m,fields,responses,cdfs,np.array([0]),repeat.SEEDS)
    fields[repeat.LABELS[0]]["cue"][:,20:60,0]=(2-a)[:,None]
    newrows,newscore=summarize_endpoints(m,fields,responses,cdfs,np.array([0]),repeat.SEEDS)
    old=[dict(m[0],coupled_signed_estimate=str(oldrows[0]["signed_estimate"]),
              coupled_score_kind=oldrows[0]["score_kind"])]
    rows,ids,midpoints,deltas=repeat.compare_scores(old,newrows,oldscore,newscore)
    assert rows[0]["original_estimate"] == pytest.approx(32/31)
    assert rows[0]["repeated_estimate"] == pytest.approx(32/31)
    assert rows[0]["observed_block_rms_change"] == ""
    assert ids.size == midpoints.size == deltas.size == 0
    # Averaging the two fields instead would make every block constant and erase covariance.
    fields[repeat.LABELS[0]]["cue"][:,20:60,0]=1
    wrong,_=summarize_endpoints(m,fields,responses,cdfs,np.array([0]),repeat.SEEDS)
    assert wrong[0]["signed_estimate"] == 0


def test_undefined_scores_remain_undefined():
    m=margin("field_lag_covariance",status=UNDEFINED,condition="cue",phase="baseline",
             lag_ms=100,leading_field=0,trailing_field=0)
    a=np.full((1,32),np.nan)
    rows,*_=repeat.compare_scores([dict(m,coupled_signed_estimate="")],[dict(m,signed_estimate="")],a,a)
    assert rows[0]["original_estimate"] == rows[0]["repeated_estimate"] == ""
    assert rows[0]["auxiliary_scatter_kind"] == "UNDEFINED"


@pytest.mark.parametrize("defect",["duplicate_blocks","bad_point","changed_margin","changed_kind"])
def test_endpoint_mismatch_rejected(defect):
    a=np.arange(32,dtype=float)[None,:];b=a+1
    old,new=summary_rows([endpoint()],a,b)
    if defect=="duplicate_blocks": a=np.tile(a,(1,2));b=np.tile(b,(1,2))
    if defect=="bad_point": new[0]["signed_estimate"]+=1
    if defect=="changed_margin": new[0]["margin"]="1"
    if defect=="changed_kind": new[0]["score_kind"]="covariance_jackknife_pseudovalue"
    with pytest.raises(ValueError):repeat.compare_scores(old,new,a,b)


def wrapper_fixture():
    entry=dict(job="599226_0",source_rng_seed=95001)
    d=dict(status="COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK",job_id="599226_0")
    w=dict(d,exit_code=0,tests_only=False,selected_source_rng_seed="95001",expected_new_development_episodes=6,
        original_independent_blocks=0,new_original_independent_blocks=0,reused_original_independent_blocks=1,
        auxiliary_repeat_index=1,confirmation_execution_released=False,
        commands=[dict(phase=p,exit_code=0) for p in ("tests_reused","source_input_validation","source_execution")])
    return d,w,entry


def test_successful_wrapper_and_failed_source_rejection():
    d,w,e=wrapper_fixture()
    repeat.check_wrapper(d,w,e)
    w["commands"][-1]["exit_code"]=1
    with pytest.raises(ValueError,match="SUCCESSFUL_SOURCE_PHASE"):repeat.check_wrapper(d,w,e)


@pytest.mark.parametrize("key,value",[("original_independent_blocks",1),("auxiliary_repeat_index",0),
                                    ("job_id","599179_0"),("tests_only",True)])
def test_wrapper_cannot_promote_old_or_failed_scope(key,value):
    d,w,e=wrapper_fixture();w[key]=value
    with pytest.raises(ValueError,match="COMPLETE_BOUND_REPEAT_WRAPPER"):repeat.check_wrapper(d,w,e)


def test_actual_registration_and_case_identity():
    root=Path(__file__).resolve().parents[2]
    config=root/"Libs/configs/neural_population_pfc_refinement_repeat_comparison.yaml"
    settings,blocks=repeat.configuration(config,root)
    b=blocks[0]
    r=dict(scope=repeat.REFINEMENT_REPEAT_SCOPE,settings_sha256=settings["protocol"]["sha256"],
        source_rng_seed=95001,condition="cue",refinement_seeds=b["refinement_seeds"],
        original_refinement_seeds=b["original_refinement_seeds"],coarse_reference=b["coarse_references"]["cue"],
        code=settings["source_engine_sha256"],auxiliary_repeat_index=1,original_independent_blocks=0,
        new_original_independent_blocks=0,reused_original_independent_blocks=1,refinement_seeds_independent_units=False,
        production_episodes=0,confirmation_execution_released=False,scientific_fidelity_qualified=False)
    repeat.check_case(r,b,settings)
    for key,value in [("refinement_seeds",[95411,95412]),("original_refinement_seeds",[99411,99412]),
                      ("source_rng_seed",95002),("coarse_reference",{}),("code",{}),
                      ("scope","development_paired_clock_pilot")]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError,match="REPEAT_SOURCE_IDENTITY"):repeat.check_case(bad,b,settings)


def test_incomplete_or_reordered_readout_index_rejected():
    settings={"_sha256":"abc"}
    for seeds in (repeat.SEEDS[:-1],repeat.SEEDS[::-1],repeat.SEEDS[:-1]+[repeat.SEEDS[0]]):
        index=dict(schema="pfc_refinement_repeat_readout_index_v1",settings_sha256="abc",
                   blocks=[dict(source_rng_seed=s,path=str(s)) for s in seeds])
        with pytest.raises(ValueError,match="COMPLETE_UNIQUE_32_BLOCK"):
            repeat.assemble_readouts(Path("."),index,settings,None,None,None,None)


def make_reader_fixtures(root):
    plan=dict(schema="pfc_refinement_repeat_readout_plan_v1",settings_sha256="fixture-settings",blocks=[])
    for i,seed in enumerate(repeat.SEEDS):
        archive=root/str(seed);(archive/"analysis").mkdir(parents=True)
        job="fixture_"+str(i)
        wrapper=dict(status="COMPLETE_PFC_REFINEMENT_REPEAT_BLOCK",job_id=job,exit_code=0,mode="block",
                     commands=[dict(phase="analysis",exit_code=0)])
        report=dict(status=repeat.BLOCK_STATUS,source_rng_seed=seed,source_case_count=6,
                    settings_sha256="fixture-settings")
        files=[]
        for name,content in (("report.json",wrapper),("analysis/report.json",report)):
            path=archive/name;path.write_text(json.dumps(content))
            files.append(dict(path=name,bytes=path.stat().st_size,sha256=repeat.sha256_file(path)))
        (archive/"delivery.json").write_text(json.dumps(dict(status=wrapper["status"],job_id=job,files=files)))
        plan["blocks"].append(dict(source_rng_seed=seed,archive=str(seed),job=job))
    return plan


def test_complete_readout_plan_resolves_bound_report_identities(tmp_path):
    plan=make_reader_fixtures(tmp_path)
    index=repeat.index_from_plan(tmp_path,plan,"fixture-settings")
    assert len(index["blocks"]) == 32
    assert index["schema"] == "pfc_refinement_repeat_readout_index_v1"
    assert index["blocks"][0]["report_sha256"] == repeat.sha256_file(tmp_path/"95001/analysis/report.json")
    assert index["blocks"][-1]["path"] == "95032/analysis"


@pytest.mark.parametrize("defect",["missing","tampered","failed","wrong_job","wrong_seed"])
def test_partial_or_wrong_readout_plan_never_becomes_complete_index(tmp_path,defect):
    plan=make_reader_fixtures(tmp_path);first=tmp_path/"95001"
    if defect=="missing": (first/"delivery.json").unlink()
    elif defect=="tampered": (first/"analysis/report.json").write_text("{}")
    elif defect=="failed":
        d=json.loads((first/"delivery.json").read_text());d["status"]="FAILED"
        (first/"delivery.json").write_text(json.dumps(d))
    elif defect=="wrong_job": plan["blocks"][0]["job"]="other_job"
    else: plan["blocks"][0]["source_rng_seed"]=95002
    with pytest.raises((ValueError,FileNotFoundError)):
        repeat.index_from_plan(tmp_path,plan,"fixture-settings")

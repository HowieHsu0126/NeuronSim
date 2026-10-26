"""Existing source pairing, causal generation order and signed mean effects."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .benchmark_tasks import pfc_working_memory_config, task_protocol
from .pfc_compiled_response import _program
from .pfc_response_evaluation import source_pairing, mean_contrast_scores, evaluate_source_response
from .pfc_source_execution import _case_config
from .pfc_population_evaluation import source_controls
from .provenance import sha256_file
from .test_pfc_compiled_response import compiled
from .test_pfc_factual_response import fitted_view
from .test_pfc_population_evaluation import fitted


def pairing_files(tmp_path, factual, candidate, *, first=48000):
    sources = []
    for index, data in enumerate((factual, candidate)):
        if index and data.episode_id == factual.episode_id:
            continue
        identity = data.source_config["benchmark_episode_candidate"]["identity"]
        sources.append(dict(episode_id=data.episode_id,
            condition_id=data.source_config["benchmark_episode_candidate"]["condition_id"],
            source_archive=f"/fixture/source_{index}",
            delivery_sha256=data.binding["source_delivery_sha256"],
            complete_units=len(data.unit_ids), event_count=len(data.event_ticks),
            circuit_index=identity["circuit_index"], block_index=identity["block_index"]))
    while len(sources) < 10:
        sources.append(dict(episode_id=f"unused_{len(sources)}", condition_id=f"unused_{len(sources)}",
                            source_archive="/fixture/unused", delivery_sha256="f"*64,
                            complete_units=len(factual.unit_ids), event_count=0))
    pairing = ([] if candidate.episode_id == factual.episode_id else
        [dict(condition_id=sources[1]["condition_id"],reference_condition=sources[0]["condition_id"],
              first_changed_current_tick=first,initial_and_final_rng_exact=True)])
    identity = factual.source_config["benchmark_episode_candidate"]["identity"]
    circuit = dict(circuit_index=identity["circuit_index"], block_index=identity["block_index"],
                   source_bindings=[dict(row, events=row["event_count"]) for row in sources], pairing=pairing)
    archive=tmp_path/"reader";archive.mkdir()
    report=archive/"report.json"
    report.write_text(json.dumps(dict(status="PASS_PFC_TEN_CONDITION_SOURCE_READBACK",exit_code=0,
                                     circuits=[circuit])))
    (archive/"delivery.json").write_text(json.dumps(dict(status="fixture",files=[])))
    receipt=dict(status="ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK",archive=str(archive),
        delivery_sha256=sha256_file(archive/"delivery.json"),report_sha256=sha256_file(report),
        complete_development_episodes=10,sources=sources,pairing=pairing,
        circuit_index=identity["circuit_index"],block_index=identity["block_index"])
    path=tmp_path/"acceptance.json";path.write_text(json.dumps(receipt))
    return path


def alternative(compiled, index):
    identity=deepcopy(compiled.source_config["benchmark_episode_candidate"]["identity"])
    identity["condition_index"]=index
    identity["episode_id"]=identity["episode_id"][:-2]+f"{index:02d}"
    config=pfc_working_memory_config(compiled.source_config,task_protocol("pfc_working_memory"),identity)
    binding=dict(compiled.binding,episode_id=identity["episode_id"],source_delivery_sha256="e"*64)
    case,_,dimensions=_case_config(config,identity["episode_id"])
    program=_program(config,case,dimensions)
    from .circuit_data import _control_bins
    def window(start,stop):
        counts=compiled.window(start,stop)[0]
        controls=_control_bins(program["starts"],program["values"],"piecewise_constant",
                               start,stop,compiled.bin_ticks)
        return counts,controls
    return SimpleNamespace(source_config=config,binding=binding,episode_id=identity["episode_id"],
        unit_ids=compiled.unit_ids,unit_field_index=compiled.unit_field_index,field_names=compiled.field_names,
        minimum_interval_ticks=compiled.minimum_interval_ticks,dt_ms=compiled.dt_ms,
        bin_ticks=compiled.bin_ticks,duration_ticks=compiled.duration_ticks,
        event_ticks=compiled.event_ticks,window=window)


def test_same_source_pair_binding_reuses_receipt(compiled,tmp_path):
    receipt=pairing_files(tmp_path,compiled,compiled)
    result=source_pairing(receipt,compiled,compiled,8000)
    assert result["first_changed_current_tick"]==compiled.duration_ticks
    assert not result["source_future_arrays_reverified"]


@pytest.mark.parametrize("change,reason",[
    ("report","RECEIPT_CHANGED"),("delivery","RECEIPT_CHANGED"),
    ("source","ADMITTED_SOURCE_AND_PAIRING_DIFFER"),("map","SAME_NATIVE_MAP"),
    ("prefix","ORIGIN_EXCEEDS_VERIFIED_NATIVE_PREFIX"),("direction","PREVIOUSLY_VERIFIED_REFERENCE_PAIR"),
])
def test_invalid_existing_pairing_rejects(compiled,tmp_path,change,reason):
    candidate=alternative(compiled,1)
    receipt=pairing_files(tmp_path,compiled,candidate,first=8000)
    if change in ("report","delivery"):
        path=tmp_path/"reader"/(change+".json")
        path.write_text(path.read_text()+" ")
    elif change=="source":
        candidate.binding["source_delivery_sha256"]="a"*64
    elif change=="map":
        candidate.unit_field_index=candidate.unit_field_index[::-1]
    elif change=="prefix":
        with pytest.raises(ValueError,match=reason):
            source_pairing(receipt,compiled,candidate,8100)
        return
    elif change=="direction":
        with pytest.raises(ValueError,match=reason):
            source_pairing(receipt,candidate,compiled,8000)
        return
    with pytest.raises(ValueError,match=reason):
        source_pairing(receipt,compiled,candidate,8000)


def test_signed_effect_mse_matches_ordered_draw_pairs():
    candidate=np.array([[[1,0],[0,1]],[[0,1],[1,0]],[[2,0],[0,2]]],dtype=np.int64)
    factual=np.ones_like(candidate)
    candidate_truth=np.array([[0,0],[1,0]],dtype=np.int64)
    factual_truth=np.array([[1,0],[0,1]],dtype=np.int64)
    mapping=np.array([0,1],dtype=np.int64)
    rows=mean_contrast_scores(candidate,factual,candidate_truth,factual_truth,mapping,2,
                             np.array([1,1]),.005,[1,2])
    difference=candidate-factual
    actual=candidate_truth-factual_truth
    for row in rows:
        horizon=row["horizon_bins"]
        residual=difference[:,:horizon]-actual[None,:horizon]
        exact=sum(np.mean(residual[i]*residual[j]) for i in range(3) for j in range(3) if i!=j)/6
        assert np.isclose(row["unit_mean_contrast_mse_corrected"],exact,atol=1e-7)
        assert np.isclose(row["field_mean_contrast_mse_hz2_corrected"],exact/(.005**2),atol=1e-7)
        assert not row["contrast_distribution_qualified"]


def test_negative_finite_draw_mean_effect_is_retained():
    candidate=np.array([[[0]],[[2]]],dtype=np.int64)
    factual=np.ones_like(candidate)
    zero=np.zeros((1,1),dtype=np.int64)
    row=mean_contrast_scores(candidate,factual,zero,zero,np.array([0]),1,np.array([1]),.005,[1])[0]
    assert row["unit_mean_contrast_mse_corrected"]==-1.
    assert row["field_mean_contrast_mse_hz2_corrected"]==-40000.


def test_same_case_full_evaluator_has_zero_effect(compiled,tmp_path):
    receipt=pairing_files(tmp_path,compiled,compiled)
    cfg=dict(horizons_bins=[1,2],joint_samples=2,support_windows=[1,2])
    rng=torch.get_rng_state().clone()
    report,arrays=evaluate_source_response(fitted_view(compiled),compiled,compiled,8000,cfg,receipt,41)
    np.testing.assert_array_equal(arrays["candidate_generated_counts"],arrays["factual_generated_counts"])
    assert all(row["unit_mean_contrast_mse_corrected"]==0 for row in report["mean_contrasts"])
    assert torch.equal(rng,torch.get_rng_state())
    assert report["future_outcomes_used_only_after_both_generations"]


def test_candidate_future_is_read_after_both_generations(compiled,tmp_path):
    from . import pfc_response_evaluation as evaluator
    candidate=alternative(compiled,1)
    receipt=pairing_files(tmp_path,compiled,candidate,first=8000)
    cfg=dict(horizons_bins=[1,2],joint_samples=2,support_windows=[1,2])
    original_draw=evaluator._draw
    drawn=[]
    def draw(*args,**kwargs):
        drawn.append(args[0].source_metadata_identity)
        return original_draw(*args,**kwargs)
    original_window=candidate.window
    def scoring_only(start,stop):
        assert len(drawn)==2
        assert start==8000 and stop==8200
        return original_window(start,stop)
    with patch.object(evaluator,"_draw",draw),patch.object(candidate,"window",scoring_only):
        report,arrays=evaluate_source_response(fitted_view(compiled),compiled,candidate,8000,cfg,receipt,41)
    assert len(drawn)==2
    assert arrays["candidate_generated_counts"].shape==(2,2,20)
    assert report["pairing"]["candidate_condition"]=="cue_aligned"
    assert not report["independent_success_gate_evaluated"]


@pytest.mark.parametrize("seed", [0,42,43,101,True])
def test_other_sampling_seed_assignments_reject_before_source_pairing(compiled,tmp_path,seed):
    cfg=dict(horizons_bins=[1,2],joint_samples=2,support_windows=[1,2])
    with patch("Libs.population_dynamics.pfc_response_evaluation.source_pairing",
               side_effect=AssertionError("UNAUTHORIZED_SEED_READ")):
        with pytest.raises(ValueError,match="SEEDS_41_42_43"):
            evaluate_source_response(fitted_view(compiled),compiled,compiled,8000,cfg,
                                     tmp_path/"missing.json",seed)

def test_registered_response_config_parses_real_yaml():
    from .pfc_response_evaluation import _response_config
    path=Path(__file__).resolve().parents[1]/"configs/neural_population_pfc_response_development.yaml"
    config=_response_config(path)
    assert config["condition_indices"]==list(range(10))
    assert config["sampling_stream_seeds"]==dict(initial=41,dynamics=42,observation=43)


def test_resource_checkpoint_loader_keeps_default_closed(tmp_path):
    from .pfc_population_evaluation import _load
    training=tmp_path/"training";training.mkdir()
    run=dict(status="RUNNING",input_family="pfc_source_reproduction",training_seed=41,completed_epochs=1)
    (training/"run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError,match="COMMITTED_PFC_DEVELOPMENT"):
        _load(tmp_path/"inputs.json",training,torch.device("cpu"))


def test_resource_checkpoint_loader_rejects_changed_delivery_binding(tmp_path):
    from .pfc_population_evaluation import _load
    training=tmp_path/"training";training.mkdir()
    run=dict(status="RUNNING",input_family="pfc_source_reproduction",training_seed=41,completed_epochs=1)
    (training/"run.json").write_text(json.dumps(run))
    (training/"best.pt").write_bytes(b"fixture")
    receipt=dict(status="RESOURCE_BLOCKED",exit_code=3,training_seed=41,
        input_family="pfc_source_reproduction",scientific_claim_qualified=False,
        files=[dict(path="training/run.json",bytes=(training/"run.json").stat().st_size,sha256="0"*64)])
    p=tmp_path/"delivery.json";p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="COMMITTED_RESOURCE_ARTIFACT_CHANGED"):
        _load(tmp_path/"inputs.json",training,torch.device("cpu"),resource_delivery=p)


def test_terminal_resource_checkpoint_uses_all_original_loader_checks(fitted,tmp_path):
    from .pfc_population_evaluation import _load
    spec,training,config=fitted
    path=training/"run.json"
    run=json.loads(path.read_text());run["status"]="RUNNING";path.write_text(json.dumps(run))
    rows=[dict(path="training/"+name,bytes=(training/name).stat().st_size,
               sha256=sha256_file(training/name)) for name in ("run.json","best.pt")]
    receipt=dict(status="RESOURCE_BLOCKED",exit_code=3,training_seed=41,
        input_family="pfc_source_reproduction",scientific_claim_qualified=False,files=rows)
    p=tmp_path/"delivery.json";p.write_text(json.dumps(receipt))
    model,cohort,report=_load(spec,training,torch.device("cpu"),resource_delivery=p)
    assert report["completed_epochs"]==1 and report["status"]=="RUNNING"
    assert model.unit_ids==tuple(report["model_configuration"]["unit_ids"])
    checkpoint=torch.load(training/"best.pt",weights_only=True)
    source=next(iter(checkpoint["trainer_sources"]))
    checkpoint["trainer_sources"][source]="f"*64
    torch.save(checkpoint,training/"best.pt")
    report["best_checkpoint_sha256"]=sha256_file(training/"best.pt")
    path.write_text(json.dumps(report))
    receipt["files"]=rows=[dict(path="training/"+name,bytes=(training/name).stat().st_size,
                              sha256=sha256_file(training/name)) for name in ("run.json","best.pt")]
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match="FITTED_IMPLEMENTATION_CHANGED"):
        _load(spec,training,torch.device("cpu"),resource_delivery=p)

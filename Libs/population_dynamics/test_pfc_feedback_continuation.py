"""Paired feedback continuation retains its original failed-window boundary."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from . import pfc_deterministic_replay as source
from .pfc_coupled_drive import create_coupled_drive
from .pfc_first_passage_suppression import continue_feedback
from .provenance import sha256_file


@pytest.mark.parametrize("lag", [1, 2])
def test_saved_feedback_diagnostic_uses_terminal_states_and_preserves_failure(tmp_path, lag):
    config=dict(dt_ms=.025 if lag == 1 else .0125,duration_ms=5.,excitatory_neurons=32,inhibitory_neurons=8,
        excitatory_preference_bins=16,weight_scaling_factor=64.,stimulus_start_ms=0.,
        stimulus_duration_ms=0.,stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_strength_nA=0.,
        distractor_start_ms=0.,distractor_duration_ms=0.,distractor_center_deg=90.,distractor_width_deg=40.)
    case=dict(case_id="feedback_fixture",source_rng_seed=95001,
        stimulus_strength_nA=0.,distractor_strength_nA=0.)
    initial,drive=create_coupled_drive(config,case,.05,[95411,95412])
    start=80
    actual=source.simulate(config,case,initial,drive=drive,stop_tick=start)
    raw=actual["raw"]
    terminal={k:v.copy() for k,v in actual["terminal_state"].items()}
    # Create a causal terminal fixture for one absent focal output at start-1.
    # The voltage difference is internal; synaptic feedback starts with E x.
    terminal["E__x"][0]+=1.
    terminal["E__lastspike"][0]=(start-lag)*config["dt_ms"]*.001
    if lag == 2:
        # One already executed post-hit synaptic step is present in this saved
        # terminal. Projection still used the state from before that step.
        c = source._constants(config, "E")
        q = {key: value.copy() for key, value in actual["terminal_state"].items()
             if key.startswith("E__")}
        state = {key[3:]: value for key, value in q.items()}
        baseline = {key: value.copy() for key, value in state.items()}
        state["x"][0] += 1.
        source._rk2(baseline,c,(start-1)*config["dt_ms"]*.001,config["dt_ms"]*.001,source._Arithmetic("numpy"),"E")
        source._rk2(state,c,(start-1)*config["dt_ms"]*.001,config["dt_ms"]*.001,source._Arithmetic("numpy"),"E")
        for key in ("s_NMDA", "x"):
            terminal["E__"+key] = state[key]
            actual["terminal_state"]["E__"+key] = baseline[key]
    rng=source.snapshot_drive_state(drive)
    fixture=tmp_path/"original";fixture.mkdir()
    files=[]
    def save_npz(name,data):
        p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**data)
    def save_json(name,data):
        p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data)+"\n")
    save_json("source/report.json",dict(resolved_config=config,selected_case=case))
    (fixture/"source/config.yaml").write_text(yaml.safe_dump(config))
    np.save(fixture/"source/initial_voltage.npy",initial)
    native_metadata=drive.metadata()
    inputs=actual["inputs_used"]
    inputs["input_mode"]="coupled_binomial_clock_refinement_stream_with_saved_focal_counterfactual_inputs"
    for name,state in (("witness",terminal),("suppression",actual["terminal_state"])):
        base="validation/half_A/"+name+"/"
        save_npz(base+"source_terminal.npz",state)
        save_npz(base+"source_raw.npz",raw)
        save_npz(base+"native_prefix_rng.npz",rng)
    save_json("validation/half_A/witness/report.json",dict(
        status="COMPLETE_PFC_FIRST_PASSAGE_WITNESS",prefix_rng_equal=True,
        prefix_ticks=start,unit_index=0,window_start_tick=60,inputs_used=inputs,
        replacement=dict(original_native_metadata=native_metadata)))
    save_json("validation/half_A/suppression/report.json",dict(
        status="FAILED_PFC_FIRST_PASSAGE_SUPPRESSION",replayed_prefix_ticks=start,
        event_boundary=dict(first_suppressed_crossing_tick=start-lag),
        error=dict(message="PFC_FIRST_PASSAGE_SUPPRESSION_NO_OBSERVABLE_RECURRENT_FEEDBACK_EFFECT")))
    parent_inputs=[]
    for name in ("config.yaml","report.json","initial_voltage.npy"):
        p=fixture/"source"/name
        parent_inputs.append(dict(path="source_inputs/half_A/source/"+name,source=str(p),sha256=sha256_file(p)))
    source_settings = dict(refinement_seeds=[95411,95412])
    if lag == 1:
        save_json("report.json",dict(status="FAILED_PFC_FIRST_PASSAGE_FINE_SUPPRESSION_EXECUTE",inputs=parent_inputs))
    else:
        # The quarter-case runner records execution.json and binds its source
        # through the frozen source delivery instead of parent input rows.
        source_files = [dict(path=Path(item["source"]).name,
                             bytes=Path(item["source"]).stat().st_size,
                             sha256=item["sha256"]) for item in parent_inputs]
        save_json("source/delivery.json",dict(files=source_files))
        source_settings.update(archive=str(fixture/"source"),case_path=".",
                               delivery_sha256=sha256_file(fixture/"source/delivery.json"))
        save_json("execution.json",dict(status="FAILED_PFC_QUARTER_SUPPRESSION",
                                       selected_case="half_A",exit_code=1))
    for p in fixture.rglob("*"):
        if p.is_file():files.append(dict(path=str(p.relative_to(fixture)),bytes=p.stat().st_size,sha256=sha256_file(p)))
    save_json("delivery.json",dict(files=files))
    settings=dict(expected_units=40,cases=[dict(name="half_A",dt_ms=config["dt_ms"],case_id=case["case_id"],
        source_rng_seed=95001,source=source_settings)])
    report=continue_feedback(settings,fixture,tmp_path/"result")
    assert report["status"]=="COMPLETE_PFC_DELAYED_FEEDBACK_DIAGNOSTIC"
    assert not report["original_probability_gate_passed"]
    assert not report["original_probability_window_changed"]
    assert report["original_prefix_simulations"]==0
    assert [x["nonfocal_nmda_values_changed"] for x in report["transitions"]]==([0,8] if lag == 1 else [8,8])
    assert report["production_episodes"]==0
    config_path=tmp_path/"settings.yaml"
    config_path.write_text(yaml.safe_dump(dict(settings,schema="pfc_fine_clock_suppression_v1")))
    command=[sys.executable,"-B","-m","Libs.population_dynamics.pfc_first_passage_suppression",
        "--config",str(config_path),"--output",str(tmp_path/"cli_result"),
        "--continue-feedback-from",str(fixture),"--case-name","half_A"]
    child=subprocess.run(command,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
    assert child.returncode==0,child.stderr
    assert json.loads(child.stdout)["status"]=="COMPLETE_PFC_DELAYED_FEEDBACK_DIAGNOSTIC"
    # A saved state changed after archival must be rejected before continuation.
    (fixture/"validation/half_A/witness/source_terminal.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="SAVED_FEEDBACK_FILE_CHANGED"):
        continue_feedback(settings,fixture,tmp_path/"corrupt_result")
    assert json.loads((tmp_path/"corrupt_result/report.json").read_text())["status"]=="FAILED_PFC_FEEDBACK_CONTINUATION"

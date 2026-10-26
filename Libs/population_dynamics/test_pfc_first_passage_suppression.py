"""Unchanged source arithmetic and complete-network suppression semantics."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from . import pfc_deterministic_replay as source
from . import pfc_first_passage_reference as reference
from . import pfc_native_drive as native
from .pfc_first_passage_probability import first_passage_ticks
from .pfc_first_passage_suppression import replay_reference, verify_feedback_reconstruction


def fixture(dt=.05, multispike=True):
    config=dict(dt_ms=dt,duration_ms=5.,excitatory_neurons=32,inhibitory_neurons=8,
        excitatory_preference_bins=16,weight_scaling_factor=64.,stimulus_start_ms=0.,
        stimulus_duration_ms=0.,stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_strength_nA=0.,
        distractor_start_ms=0.,distractor_duration_ms=0.,distractor_center_deg=90.,distractor_width_deg=40.)
    case=dict(case_id="suppressed_reference_fixture",source_rng_seed=95001,
              stimulus_strength_nA=0.,distractor_strength_nA=0.)
    initial=np.full(40,-60.);initial[:2]=-50.000001
    ticks=round(5/dt);e=np.zeros((ticks,32),dtype=np.uint16);e[0,:2]=20;e[1,:2]=3
    i=np.full((ticks,8),10,dtype=np.uint16)
    if multispike:e[round(2.25/dt):,0]=50
    return config,case,initial,e,i


@pytest.fixture
def oracle():
    default=Path(__file__).resolve().parents[2]/"Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/first_passage_fine_history/prepare/run_603250/Libs/population_dynamics/pfc_deterministic_replay.py"
    path=Path(os.environ.get("NEURONSIM_SUPPRESSION_BASELINE",str(default)))
    spec=importlib.util.spec_from_file_location("Libs.population_dynamics._suppression_oracle",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dt",[.05,.025,.0125])
def test_default_source_is_bitwise_equal_to_frozen_source_before_hook(dt,oracle):
    config,case,initial,e,i=fixture(dt)
    old=oracle.simulate(config,case,initial,e,i)
    actual=source.simulate(config,case,initial,e,i)
    for family in ("raw","terminal_state"):
        assert old[family].keys()==actual[family].keys()
        for key in old[family]:np.testing.assert_array_equal(actual[family][key],old[family][key])
    assert actual["inputs_used"]==old["inputs_used"] and "suppressed_reference" not in actual


@pytest.mark.parametrize("dt",[.05,.025,.0125])
@pytest.mark.parametrize("kernel",[{}, dict(sigma_weight_profile=30., Jpos_excit2excit=1.7)])
def test_actual_multispike_replay_retains_first_hit_and_recomputes_feedback(dt,kernel):
    config,case,initial,e,i=fixture(dt)
    config.update(kernel)
    plain=source.simulate(config,case,initial,e,i)
    tape=reference.validate_tape(config,plain["raw"])
    focal=tape.events[tape.events[:,1]==0,0];assert len(focal)>=2
    tau,hit=map(int,focal[:2]);stop=len(e)
    selection=reference.FirstSpikeWindow(0,tau,0,stop,round(2/dt))
    actual,checks=replay_reference(config,case,initial,tape,selection,counts_E=e,counts_I=i)
    result=actual["suppressed_reference"]
    assert checks["first_suppressed_crossing_tick"]==hit
    assert checks["observed_focal_window_ticks"]==focal.tolist()
    assert checks["retained_focal_window_ticks"]==[tau]
    predicted=first_passage_ticks(config,result["conditional_initial"],result["forcing"],e[tau:stop-1,0][None,:])
    assert int(predicted[0])==hit
    # The missing focal output changes recurrent NMDA drive. Final voltages
    # can coincide because a nonfocal unit is refractory or reset at the end.
    assert not np.array_equal(actual["terminal_state"]["I__s_NMDA_total"],
                              plain["terminal_state"]["I__s_NMDA_total"])
    rebuilt=reference.reconstruct(config,case,reference.validate_tape(config,actual["raw"]),
        stop_tick=stop,counts_E=e,counts_I=i,selection=selection)
    assert rebuilt["conditional_initial"]==result["conditional_initial"]
    for key,value in rebuilt["terminal_state"].items():
        np.testing.assert_array_equal(value,actual["terminal_state"][key])
    verified = verify_feedback_reconstruction(config,case,actual,plain["terminal_state"],selection,
        counts_E=e,counts_I=i)
    assert verified["nonfocal_recurrent_values_changed"] > 0
    with pytest.raises(ValueError, match="NO_OBSERVABLE_RECURRENT_FEEDBACK_EFFECT"):
        verify_feedback_reconstruction(config,case,actual,actual["terminal_state"],selection,
            counts_E=e,counts_I=i)
    assert actual["inputs_used"]==plain["inputs_used"]


def test_no_second_spike_keeps_all_original_events_and_states():
    config,case,initial,e,i=fixture(multispike=False)
    plain=source.simulate(config,case,initial,e,i);tape=reference.validate_tape(config,plain["raw"])
    focal=tape.events[tape.events[:,1]==0,0];assert len(focal)==1
    actual,checks=replay_reference(config,case,initial,tape,
        reference.FirstSpikeWindow(0,int(focal[0]),0,len(e),40),counts_E=e,counts_I=i)
    for family in ("raw","terminal_state"):
        for key in plain[family]:np.testing.assert_array_equal(actual[family][key],plain[family][key])
    assert not checks["original_window_multispike"]


def test_focal_future_input_changes_do_not_change_suppressed_recurrent_forcing():
    config,case,initial,e,i=fixture()
    plain=source.simulate(config,case,initial,e,i)
    tau=int(plain["raw"]["spike_ticks"][plain["raw"]["spike_unit_indices"]==0][0])
    intervention=source.FocalSpikeSuppression(0,tau,len(e))
    a=source.simulate(config,case,initial,e,i,focal_spike_suppression=intervention)
    altered=e.copy();altered[tau:,0]=0
    b=source.simulate(config,case,initial,altered,i,focal_spike_suppression=intervention)
    assert a["suppressed_reference"]["conditional_initial"]==b["suppressed_reference"]["conditional_initial"]
    for key,value in a["suppressed_reference"]["forcing"].items():
        np.testing.assert_array_equal(value,b["suppressed_reference"]["forcing"][key])
    for key in ("spike_ticks","spike_unit_indices","native_spike_counts"):
        np.testing.assert_array_equal(a["raw"][key],b["raw"][key])
    for key,value in a["terminal_state"].items():
        keep=np.arange(len(value))!=0 if key in ("E__v","E__s_AMPA") else slice(None)
        np.testing.assert_array_equal(value[keep],b["terminal_state"][key][keep])
    assert a["terminal_state"]["E__s_AMPA"][0]!=b["terminal_state"]["E__s_AMPA"][0]


def test_invalid_suppression_refuses_before_random_draw_and_requires_first_spike():
    config,case,initial,e,i=fixture();_,drive=native.create_drive(config,case)
    for spec in (source.FocalSpikeSuppression(32,1,100),source.FocalSpikeSuppression(0,-1,100),
                 source.FocalSpikeSuppression(0,2,101),source.FocalSpikeSuppression(True,2,100),{}):
        with pytest.raises(ValueError,match="PFC_REFERENCE_SUPPRESSION"):
            source.simulate(config,case,initial,drive=drive,focal_spike_suppression=spec)
        assert drive.ticks==0
    with pytest.raises(ValueError,match="FIRST_SPIKE_NOT_RETAINED"):
        source.simulate(config,case,initial,e,i,focal_spike_suppression=source.FocalSpikeSuppression(0,0,100))


@pytest.mark.parametrize("dt", [.025, .0125])
def test_fine_override_can_reconstruct_complete_suppressed_reference(dt):
    from .pfc_first_passage_witness import FocalOverrideDrive, bound_prefix_state
    from . import pfc_coupled_drive as coupled
    config, case, initial, _, _ = fixture(dt)
    _, a = coupled.create_coupled_drive(config,case,.05,[95411,95412])
    _, b = coupled.create_coupled_drive(config,case,.05,[95411,95412])
    native_counts = b.take(b.total_ticks)
    wrapped = FocalOverrideDrive(a,unit_index=0,start_tick=1,stop_tick=5,counts=native_counts["E"][1:5,0])
    streamed = source.simulate(config,case,initial,drive=wrapped)
    _, c = coupled.create_coupled_drive(config,case,.05,[95411,95412])
    check = FocalOverrideDrive(c,unit_index=0,start_tick=1,stop_tick=5,counts=native_counts["E"][1:5,0])
    rebuilt = reference.reconstruct(config,case,reference.validate_tape(config,streamed["raw"]),
                                    stop_tick=c.total_ticks,drive=check)
    for key,value in rebuilt["terminal_state"].items():
        np.testing.assert_array_equal(value,streamed["terminal_state"][key])
    kind = "coupled_binomial_clock_refinement_stream"
    for key,value in bound_prefix_state(a,kind).items():
        np.testing.assert_array_equal(value,bound_prefix_state(c,kind)[key])


def test_remaining_fine_case_selection_preserves_full_denominator(tmp_path):
    from .pfc_first_passage_suppression import validate_fine_histories
    settings = dict(cases=[dict(name="half_A", dt_ms=.025),
                           dict(name="quarter_A", dt_ms=.0125)])
    for names in ([], ["quarter_A", "quarter_A"], ["unregistered"]):
        with pytest.raises(ValueError, match="REGISTERED_FINE_CASE_SELECTION"):
            validate_fine_histories(settings, tmp_path, tmp_path/str(len(list(tmp_path.iterdir()))), case_names=names)
    with pytest.raises(FileNotFoundError):
        validate_fine_histories(settings, tmp_path, tmp_path/"remaining", case_names=["quarter_A"])
    import json
    report = json.loads((tmp_path/"remaining/report.json").read_text())
    assert report["registered_case_names"] == ["half_A", "quarter_A"]
    assert report["selected_case_names"] == ["quarter_A"]
    assert report["unexecuted_case_names"] == ["half_A"]
    assert report["status"] == "FAILED_PFC_FINE_CLOCK_SUPPRESSION"
    assert report["production_episodes"] == 0

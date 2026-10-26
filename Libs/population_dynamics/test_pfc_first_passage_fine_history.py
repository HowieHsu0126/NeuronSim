"""Three-stream coupled prefix accounting and fine-clock reference integration."""
import numpy as np
import pytest
from . import pfc_first_passage_history as history
from . import pfc_first_passage_reference as reference
from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native

KIND = "coupled_binomial_clock_refinement_stream"


def fixture(dt):
    config = dict(dt_ms=dt,duration_ms=5.,excitatory_neurons=32,inhibitory_neurons=8,
        excitatory_preference_bins=16,weight_scaling_factor=64.,stimulus_start_ms=0.,
        stimulus_duration_ms=5.,stimulus_center_deg=180.,stimulus_width_deg=360.,stimulus_strength_nA=1.,
        distractor_start_ms=0.,distractor_duration_ms=0.,distractor_center_deg=90.,distractor_width_deg=40.)
    case = dict(case_id="coupled_history_fixture",source_rng_seed=95001,
                stimulus_strength_nA=1.,distractor_strength_nA=0.)
    report = dict(resolved_config=config,selected_case=case,coarse_dt_ms=.05,refinement_seeds=[95411,95412])
    return config,case,report


@pytest.mark.parametrize("dt",[.025,.0125])
def test_prefix_chunking_rng_and_buffer_match_without_mutation(dt):
    config,case,report=fixture(dt)
    for stop in (8,9,10,11,12):
        _,a=history.create_input_drive(config,case,report,KIND)
        _,b=history.create_input_drive(config,case,report,KIND)
        chunks=[a.take(n) for n in (1,2,stop-3)]; expected=b.take(stop)
        for key in ("E","I"):np.testing.assert_array_equal(np.concatenate([v[key] for v in chunks]),expected[key])
        left,right=history.prefix_state_arrays(a,KIND),history.prefix_state_arrays(b,KIND)
        assert left.keys()==right.keys()
        for key in left:np.testing.assert_array_equal(left[key],right[key])
        offset=stop%a.factor
        assert int(left["buffer_offset"])==offset
        assert left["buffered_parent_children"].shape==(a.factor if offset else 0,40)
        left["buffered_parent_children"][:]=1000
        again=history.prefix_state_arrays(a,KIND)
        for key in right:np.testing.assert_array_equal(again[key],right[key])
        x,y=a.take(7),b.take(7)
        for key in x:np.testing.assert_array_equal(x[key],y[key])


def test_native_prefix_retains_native_rng_identity():
    config,case,_=fixture(.05)
    _,drive=history.create_input_drive(config,case,{},"native_numpy_stream")
    drive.take(9)
    result=history.prefix_state_arrays(drive,"native_numpy_stream")
    for key,value in native._state_arrays(drive.get_state()).items():np.testing.assert_array_equal(result[key],value)
    assert int(result["fine_ticks"])==9 and "buffered_parent_children" not in result


@pytest.mark.parametrize("dt",[.025,.0125])
def test_complete_prefix_equals_authoritative_three_rng_export(dt):
    config,case,report=fixture(dt)
    _,drive=history.create_input_drive(config,case,report,KIND)
    drive.take(drive.total_ticks)
    observed=history.prefix_state_arrays(drive,KIND)
    for key,value in drive.final_state_arrays().items():np.testing.assert_array_equal(observed[key],value)
    assert observed["buffered_parent_children"].shape==(0,40) and int(observed["buffer_offset"])==0


@pytest.mark.parametrize("dt",[.025,.0125])
def test_coupled_history_reference_matches_individual_reconstruction(dt):
    config,case,report=fixture(dt)
    initial,drive=history.create_input_drive(config,case,report,KIND)
    initial[0]=-50.000001  # Explicit bounded fixture; not a native initial draw claim.
    simulation=source.simulate(config,case,initial,drive=drive)
    tape=reference.validate_tape(config,simulation["raw"])
    ticks=round(config["duration_ms"]/dt)
    rows=history.classify(config,tape,[dict(sample_index=0,unit_index=0,window_start_tick=0,window_stop_tick=ticks)])
    assert rows[0]["category"]=="conditional_integration_required"
    _,batched=history.create_input_drive(config,case,report,KIND)
    actual,_=history.extract(config,case,tape,rows,drive=batched)
    _,individual=history.create_input_drive(config,case,report,KIND)
    row=rows[0]
    selection=reference.FirstSpikeWindow(0,row["first_spike_tick"],0,ticks,row["refractory_ticks"])
    frames=[]
    def snapshot(stage,tick,state):
        if stage=="pre_rk2" and selection.first_spike_tick<tick<ticks:
            frames.append({key:state["E__"+key][0] for key in ("s_GABA","s_NMDA_total","I_stim","s_NMDA","x")})
    expected=reference.reconstruct(config,case,tape,stop_tick=ticks,drive=individual,selection=selection,on_snapshot=snapshot)
    conditional,forcing=actual[0]
    assert conditional==expected["conditional_initial"]
    for key in frames[0]:np.testing.assert_array_equal(forcing[key],[f[key] for f in frames])
    a,b=history.prefix_state_arrays(batched,KIND),history.prefix_state_arrays(individual,KIND)
    for key in a:np.testing.assert_array_equal(a[key],b[key])

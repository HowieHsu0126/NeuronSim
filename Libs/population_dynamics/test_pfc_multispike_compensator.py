"""Source-parity, conditioning schedule, and bounded-input integration tests."""
from itertools import product

import numpy as np
import pytest

from . import pfc_multispike_compensator as multispike
from . import pfc_deterministic_replay as executor


def settings():
    return dict(window_ms=5.,physical_maximum_count=3,fields=[3,4,5,6,7],
                decimal_scale_digits=40,factorial_numeric_allowance=dict(numerator=1,denominator=100000000))


def config():
    return dict(dt_ms=.05,duration_ms=20.,excitatory_neurons=32,inhibitory_neurons=8,
        excitatory_preference_bins=16,weight_scaling_factor=64.,stimulus_start_ms=0.,stimulus_duration_ms=20.,
        stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_strength_nA=.5,
        distractor_start_ms=10.,distractor_duration_ms=5.,distractor_center_deg=90.,distractor_width_deg=40.)


def test_tail_cutoff_preserves_nonzero_omitted_probability():
    law=multispike.truncated_law(1000,1.4*.05e-3,100,3,40,dict(numerator=1,denominator=100000000))
    assert law["cutoff"]<1000 and int(law["omitted_tail_upper_scaled"])>0
    assert int(law["worst_case_per_unit_window_factorial_width_scaled"])*100000000<=int(law["scale"])
    lo,hi,omitted=multispike.integrate_histogram(np.zeros(law["cutoff"]+1,dtype=np.int64),1000,law)
    assert lo==0 and hi>0 and omitted==hi
    lo,hi,_=multispike.integrate_histogram(np.full(law["cutoff"]+1,1000,dtype=np.int64),1000,law)
    assert lo<=2000*int(law["scale"])==hi


def test_nonmonotone_input_response_is_integrated_without_threshold_search():
    law=multispike.truncated_law(2,.5,1,2,20,dict(numerator=1,denominator=100000000))
    assert law["cutoff"]==2
    lo,hi,omitted=multispike.integrate_histogram(np.array([3,0,3]),3,law)
    assert lo==hi==3*int(law["scale"]) and omitted==0


def test_factorial_compensator_is_unbiased_but_not_a_pathwise_bound_or_variance_guarantee():
    law=multispike.truncated_law(1,.5,4,2,20,dict(numerator=1,denominator=100000000))
    compensated=[];observed=[]
    for first,second in product((0,1),repeat=2):
        count=first;weights=np.zeros(2,dtype=np.int64);total=0
        weights+=count*np.array([0,1]);total+=count;count+=second
        for _ in range(2):
            event=int(count<2)
            weights+=count*event;total+=count;count+=event
        lo,hi,_=multispike.integrate_histogram(weights,total,law)
        assert lo==hi
        compensated.append(lo/int(law["scale"]));observed.append(count*(count-1))
    assert observed==[2,2,2,2] and compensated==[2,2,3,1]
    assert np.mean(compensated)==np.mean(observed)
    assert np.var(compensated)>.0==np.var(observed)


def test_window_boundary_discards_previous_window_weight():
    observer=multispike.MultispikeObserver(config(),settings(),phases={"fixture":(0.,20.)})
    observer.observed_ticks=observer.window_ticks;observer.window_id=0;observer.counts[:]=1
    # No other locals are needed: a new target-tick window has zero preceding count.
    observer.pre_rk2(dict(tick=observer.window_ticks))
    assert observer.pending is None and not observer.eligible_weights.any()
    assert all(row["factorial_compensator_upper"] is None for row in observer.results())


def test_observer_preserves_full_source_arrays_and_tracks_per_unit_factorial_counts():
    cfg=config();case=dict(case_id="fixture",stimulus_strength_nA=.5,distractor_strength_nA=0.)
    initial=np.linspace(-59.,-51.,40,dtype=np.float64)
    ticks=round(cfg["duration_ms"]/cfg["dt_ms"])
    # Deliberately rare but source-supported input history exercises multiple spikes
    # and actual inputs outside the retained probability-enumeration range.
    counts_e=np.full((ticks,32),40,dtype=np.uint16);counts_i=np.zeros((ticks,8),dtype=np.uint16)
    plain=executor.simulate(cfg,case,initial,counts_e,counts_i)
    observer=multispike.MultispikeObserver(cfg,settings(),phases={"fixture":(0.,20.)})
    with observer.observing():
        observed=executor.simulate(cfg,case,initial,counts_e,counts_i)
    for group in ("raw","terminal_state"):
        assert observed[group].keys()==plain[group].keys()
        for key in plain[group]:np.testing.assert_array_equal(observed[group][key],plain[group][key])
    assert observer.observed_ticks==ticks and observer.parity_units>0 and observer.actual_above_cutoff>0
    raw=plain["raw"];count=np.zeros((ticks//observer.window_ticks,32),dtype=np.int64)
    mask=raw["spike_unit_indices"]<32
    np.add.at(count,(raw["spike_ticks"][mask]//observer.window_ticks,raw["spike_unit_indices"][mask]),1)
    assert count.max()<=3 and count.max()>1
    for fi,field in enumerate(observer.fields):
        ids=observer.units[observer.field_slot==fi]
        assert observer.observed_factorial[0,fi]==np.sum(count[:,ids]*(count[:,ids]-1))
    rows=observer.results()
    assert all(r["source_path_complete"] and r["observed_complete_windows"]==4 for r in rows)
    assert all(r["conditional_input_interval_width"]<=1e-8 for r in rows)


def test_input_response_weight_cannot_exceed_predictable_total():
    law=multispike.truncated_law(1,.5,1,1,20,dict(numerator=1,denominator=100000000))
    with pytest.raises(ValueError,match="WEIGHTED_RESPONSE_HISTOGRAM"):
        multispike.integrate_histogram(np.array([0,2]),1,law)

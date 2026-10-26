"""First-crossing oracle, input timing, and Monte Carlo denominator checks."""
from math import comb

import numpy as np
import pytest

from . import pfc_deterministic_replay as source
from . import pfc_first_passage_reference as reference
from . import pfc_first_passage_probability as probability


def fixture(dt):
    config = dict(dt_ms=dt, duration_ms=5., excitatory_neurons=32,inhibitory_neurons=8,
        excitatory_preference_bins=16,weight_scaling_factor=64.,stimulus_start_ms=0.,stimulus_duration_ms=0.,
        stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_strength_nA=0.,
        distractor_start_ms=0.,distractor_duration_ms=0.,distractor_center_deg=90.,distractor_width_deg=40.)
    case = dict(case_id='first_passage_oracle',stimulus_strength_nA=0.,distractor_strength_nA=0.)
    initial = np.full(40,-60.,dtype=np.float64); initial[0] = -50.000001
    ticks = round(5./dt)
    e = np.zeros((ticks,32),dtype=np.uint16); e[0,0]=20; e[1,0]=3
    i = np.full((ticks,8),10,dtype=np.uint16)
    original = source.simulate(config,case,initial,e,i)
    tape = reference.validate_tape(config,original['raw'])
    selected = reference.select_first_spike_window(config,tape,eligible_units=np.array([0]),
        phase_start_tick=0,phase_stop_tick=ticks)
    reference.require_observed_reference(tape,selected)
    rows = []
    def capture(stage,tick,state):
        if stage=='pre_rk2' and selected.first_spike_tick < tick < ticks:
            rows.append(dict(tick=tick,time_seconds=tick*dt*1e-3,
                **{key:state['E__'+key][0] for key in ('s_GABA','s_NMDA_total','I_stim','s_NMDA','x')}))
    reconstructed = reference.reconstruct(config,case,tape,stop_tick=ticks,counts_E=e,counts_I=i,
                                           selection=selected,on_snapshot=capture)
    forcing = {key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    # Match the source's multiplication order exactly, including for nonzero tau.
    forcing['time_seconds'] = forcing['tick'] * (dt*1e-3)
    return config,case,initial,e,i,original,selected,reconstructed['conditional_initial'],forcing


@pytest.mark.parametrize('dt',[.05,.025])
def test_first_crossing_matches_full_coupled_source(dt):
    config,case,initial,e,i,original,selected,conditional,forcing=fixture(dt)
    slots=len(forcing['tick'])
    alternatives=np.array([np.zeros(slots),np.full(slots,40),
        np.where(np.arange(slots)<slots//2,0,40),np.arange(slots)%7],dtype=np.uint16)
    actual=probability.first_passage_ticks(config,conditional,forcing,alternatives)
    expected=[]
    for draws in alternatives:
        inputs=e.copy()
        inputs[selected.first_spike_tick:selected.window_stop_tick-1,selected.unit_index]=draws
        replay=source.simulate(config,case,initial,inputs,i)
        mask=(replay['raw']['spike_unit_indices']==selected.unit_index)&(replay['raw']['spike_ticks']>selected.first_spike_tick)
        ticks=replay['raw']['spike_ticks'][mask]
        expected.append(int(ticks[0]) if len(ticks) else -1)
    np.testing.assert_array_equal(actual,expected)
    assert np.any(actual>=0) and np.any(actual<0)
    assert np.any(original['raw']['spike_unit_indices']>=32)  # Nontrivial inhibitory feedback.
    assert np.all(actual[actual>=0]>=selected.first_spike_tick+selected.refractory_ticks)


def test_future_focal_state_is_rejected():
    config,_,_,_,_,_,_,initial,forcing=fixture(.05)
    forcing['s_AMPA']=np.zeros(len(forcing['tick']))
    with pytest.raises(ValueError,match='FOCAL_FUTURE_STATE_LEAKAGE'):
        probability.first_passage_ticks(config,initial,forcing,np.zeros((1,len(forcing['tick'])),dtype=np.uint16))


def test_binomial_interval_retains_zero_hit_uncertainty_and_coverage():
    interval=probability.binomial_interval(0,1000000,.95)
    assert interval['lower']==0 and abs(interval['upper']-(1-.025**(1/1000000)))<1e-12
    n=10
    intervals=[probability.binomial_interval(k,n,.95) for k in range(n+1)]
    for p in (.001,.01,.1,.3,.5,.9,.99,.999):
        coverage=sum(comb(n,k)*p**k*(1-p)**(n-k) for k,bounds in enumerate(intervals)
                     if bounds['lower']<=p<=bounds['upper'])
        assert coverage>=.95-1e-12
    with pytest.raises(ValueError,match='BERNOULLI_DENOMINATOR'):
        probability.binomial_interval(11,10,.95)


def test_persistent_sampling_stream_is_chunk_invariant():
    config,_,_,_,_,_,_,initial,forcing=fixture(.05)
    kwargs=dict(samples=32,seed_components=[20261009,95001,28022,2004],confidence=.95)
    a,ha=probability.monte_carlo(config,initial,forcing,batch_size=8,**kwargs)
    b,hb=probability.monte_carlo(config,initial,forcing,batch_size=16,**kwargs)
    assert a['input_draw_sha256']==b['input_draw_sha256']
    assert a['final_rng_state']==b['final_rng_state']
    np.testing.assert_array_equal(ha,hb)
    assert int(ha.sum())==a['hits'] and a['hits']+a['no_hits']==32
    assert a['interval']['upper']>0


def test_numerical_failure_is_preserved_not_counted_as_no_hit(monkeypatch,tmp_path):
    config,_,_,_,_,_,_,initial,forcing=fixture(.05)
    def fail(*args,**kwargs):
        raise ValueError('NONFINITE_ACTIVE_TRANSITION')
    monkeypatch.setattr(probability,'first_passage_ticks',fail)
    with pytest.raises(ValueError,match='NONFINITE_ACTIVE_TRANSITION'):
        probability.monte_carlo(config,initial,forcing,samples=8,batch_size=8,
            seed_components=[20261009,95001,28022,2004],failure_output=tmp_path)
    with np.load(tmp_path/'failed_batch_0.npz',allow_pickle=False) as data:
        assert data['counts'].shape==(8,len(forcing['tick'])) and int(data['batch_start'])==0

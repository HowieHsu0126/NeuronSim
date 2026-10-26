"""Exact finite-path importance identities and bounded-real-score coverage."""
from itertools import product
from math import comb,exp,log

import numpy as np
import pytest

from . import pfc_first_passage_importance as importance


def mass(counts,p,n=2):
    return np.prod([comb(n,int(x))*p**int(x)*(1-p)**(n-int(x)) for x in counts])


def test_stopped_marginal_mixture_recovers_exact_hit_probability_and_reduces_full_weight_variance():
    p=.1;proposals=np.array([.1,.2,.4,.8]);mixture=np.full(4,.25)
    target=0.;weighted=0.;full_mean=0.;stopped_second=0.;full_second=0.;normalization=0.
    for sequence in product(range(3),repeat=3):
        counts=np.asarray(sequence,dtype=np.int64)
        q=sum(alpha*mass(counts,pj) for alpha,pj in zip(mixture,proposals));normalization+=q
        crossings=np.flatnonzero(np.cumsum(counts)>=2)
        if not len(crossings):continue
        r=int(crossings[0])+1
        prefix=float(np.exp(importance.log_likelihood_ratio(np.array([counts[:r].sum()]),np.array([2*r]),p,proposals,mixture))[0])
        full=float(np.exp(importance.log_likelihood_ratio(np.array([counts.sum()]),np.array([6]),p,proposals,mixture))[0])
        expected_prefix=mass(counts[:r],p)/sum(alpha*mass(counts[:r],pj) for alpha,pj in zip(mixture,proposals))
        assert abs(prefix-expected_prefix)<1e-13 and 0<prefix<=4 and 0<full<=4
        target+=mass(counts,p);weighted+=q*prefix;full_mean+=q*full
        stopped_second+=q*prefix**2;full_second+=q*full**2
    assert abs(normalization-1)<1e-14
    assert abs(weighted-target)<1e-14 and abs(full_mean-target)<1e-14
    assert stopped_second-target**2<=full_second-target**2+1e-14


def test_bounded_mean_interval_covers_non_bernoulli_scores_and_keeps_zero_uncertainty():
    bounds=importance.bounded_mean_interval(0.,1000000,4.,.95)
    assert bounds['lower']==0 and abs(bounds['upper']-4*(-np.expm1(-log(40)/1000000)))<1e-15
    values=[0.,.3,1.];probabilities=[.5,.25,.25];mean=sum(x*p for x,p in zip(values,probabilities))
    coverage=0.
    for indices in product(range(3),repeat=4):
        sample_mean=sum(values[j] for j in indices)/4
        interval=importance.bounded_mean_interval(sample_mean,4,1.,.95)
        if interval['lower']<=mean<=interval['upper']:
            coverage+=np.prod([probabilities[j] for j in indices])
    assert coverage>=.95-1e-14


def test_extreme_prefix_counts_remain_finite_and_obey_defensive_bound():
    p=7e-5;proposals=p*np.array([1.,2.,4.,8.]);mixture=np.full(4,.25)
    successes=np.array([0,1,20,60,1000,95000],dtype=np.int64)
    trials=np.full(len(successes),95000,dtype=np.int64)
    logs=importance.log_likelihood_ratio(successes,trials,p,proposals,mixture)
    assert np.isfinite(logs).all() and np.all(logs<=log(4)+1e-14)
    assert logs[-1]<-1000  # Kept in log form; an underflowed positive event score must be rejected.


def tail_fixture(dt_ms=.05, first_tick=2004, stop_tick=2100):
    config = {'dt_ms': dt_ms}
    conditional = dict(population='E', first_spike_tick=first_tick, window_stop_tick=stop_tick,
        timing='first_spike_reset_with_preinput_ampa', fresh_input_included=False,
        state=dict(v=-.06, lastspike=first_tick*(dt_ms*1e-3), not_refractory=False))
    probability = importance.native._RATE_HZ * (dt_ms*1e-3)
    report = dict(status='COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE', samples=10000,
        target='second_focal_spike_before_window_end_given_first_spike_past_and_other_unit_future_innovations',
        nominal_probability=probability, proposal_probabilities=[probability*x for x in (1,2,4,8)],
        mixture_probabilities=[.25]*4, weight_bound=4., proposal_hits=0, proposal_no_hits=10000,
        stopped_prefix=dict(score_sum=0., conditional_interval=dict(confidence_level=.95)))
    ticks = np.arange(first_tick+1, stop_tick)
    histogram = dict(tick=ticks, proposal_hits=np.zeros((4,len(ticks)),dtype=np.int64),
                     stopped_weight_sum=np.zeros(len(ticks),dtype=np.float64))
    return config, conditional, report, histogram


def test_refractory_tail_uses_inclusive_cutoff_and_does_not_treat_enclosing_mean_as_third_probability():
    config, conditional, report, histogram = tail_fixture()
    for tick, weight in [(2044,.2),(2059,.3),(2060,.4)]:
        index = tick-int(histogram['tick'][0])
        histogram['proposal_hits'][0,index] = 1
        histogram['stopped_weight_sum'][index] = weight
    report.update(proposal_hits=3, proposal_no_hits=9997)
    report['stopped_prefix']['score_sum'] = .9
    result = importance.refractory_tail_bound(config,conditional,report,histogram)
    assert result['minimum_interspike_ticks'] == 40
    assert result['second_spike_cutoff_inclusive'] == 2059
    assert result['enclosing_second_spike_estimate'] == .5/10000
    assert result['third_spike_probability_upper_bound'] > .5/10000
    assert result['third_spike_probability_or_bound_qualified'] is False
    assert result['population_history_uncertainty_established'] is False


@pytest.mark.parametrize('dt_ms,r',[(.05,40),(.025,80),(.0125,160)])
def test_refined_source_refractory_boundary_and_eligible_zero_uncertainty(dt_ms,r):
    first = 10000
    config,conditional,report,histogram = tail_fixture(dt_ms,first,first+2*r)
    exact = importance.refractory_tail_bound(config,conditional,report,histogram)
    assert exact['minimum_interspike_ticks'] == r
    assert exact['structural_zero'] and exact['third_spike_probability_upper_bound'] == 0
    config,conditional,report,histogram = tail_fixture(dt_ms,first,first+2*r+1)
    eligible = importance.refractory_tail_bound(config,conditional,report,histogram)
    assert not eligible['structural_zero']
    assert eligible['third_spike_probability_upper_bound'] > 0
    # The pinned floating-clock implementation permits the next threshold at r,
    # and forbids r-1, including later locations in the complete 5-second source.
    dt = dt_ms*1e-3
    arithmetic = importance.source._Arithmetic('numpy')
    for tau in (first,round(4900/dt_ms)):
        assert int(arithmetic.integer_steps((tau+r-1)*dt-tau*dt,dt)) < r
        assert int(arithmetic.integer_steps((tau+r)*dt-tau*dt,dt)) >= r


@pytest.mark.parametrize('defect',['clock','negative_weight','total','early_hit','wrong_law'])
def test_refractory_tail_rejects_invalid_archived_inputs(defect):
    config,conditional,report,histogram = tail_fixture()
    if defect == 'clock': histogram['tick'][0] += 1
    elif defect == 'negative_weight': histogram['stopped_weight_sum'][0] = -1.
    elif defect == 'total': report['stopped_prefix']['score_sum'] = .1
    elif defect == 'wrong_law': report['nominal_probability'] *= 2
    else:
        histogram['proposal_hits'][0,0] = 1
        histogram['stopped_weight_sum'][0] = .1
        report.update(proposal_hits=1,proposal_no_hits=9999)
        report['stopped_prefix']['score_sum'] = .1
    with pytest.raises(ValueError):
        importance.refractory_tail_bound(config,conditional,report,histogram)


def test_truncated_stopping_weights_recover_exact_enclosing_event_probability():
    p=.1; proposals=np.array([.1,.2,.4,.8]); mixture=np.full(4,.25)
    target=weighted=0.
    for sequence in product(range(3),repeat=3):
        counts=np.asarray(sequence,dtype=np.int64)
        crossings=np.flatnonzero(np.cumsum(counts)>=2)
        if not len(crossings) or crossings[0] > 1: continue
        length=int(crossings[0])+1
        q=sum(alpha*mass(counts,pj) for alpha,pj in zip(mixture,proposals))
        weight=exp(float(importance.log_likelihood_ratio(np.array([counts[:length].sum()]),
            np.array([2*length]),p,proposals,mixture)[0]))
        target+=mass(counts,p)
        weighted+=q*weight
    assert abs(target-weighted)<1e-14
    # Exhaustive discrete spike trains verify the enclosing event, including
    # a hit on b-1 and the first impossible third-spike time b.
    for t1,t2,t3 in product(range(9),repeat=3):
        if t1+2 <= t2 and t2+2 <= t3 and t3 < 8:
            assert t2 <= 8-1-2

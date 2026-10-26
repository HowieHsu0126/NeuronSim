"""Conditional first-hit importance sampling with marginal stopped-path weights.

A native component bounds every weight.  This changes the proposal, not the
source input law.  All uncertainty reported here conditions on one reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import yaml

from .pfc_first_passage_probability import first_passage_ticks
from . import pfc_native_drive as native
from . import pfc_deterministic_replay as source


def require(value, message):
    if not value:
        raise ValueError('PFC_FIRST_PASSAGE_IMPORTANCE_' + message)


def log_likelihood_ratio(successes, trials, probability, proposals, mixture):
    """Marginal P/Q for a prefix, with its binomial coefficients cancelled."""
    successes, trials = np.asarray(successes), np.asarray(trials)
    proposals, mixture = np.asarray(proposals,dtype=np.float64), np.asarray(mixture,dtype=np.float64)
    require(successes.shape == trials.shape and successes.dtype.kind in 'iu' and trials.dtype.kind in 'iu'
            and np.all((successes >= 0) & (successes <= trials)), 'PREFIX_COUNTS')
    require(proposals.ndim == mixture.ndim == 1 and proposals.shape == mixture.shape
            and np.all((proposals > 0) & (proposals < 1)) and np.all(mixture > 0)
            and float(mixture.sum()) == 1. and np.any(proposals == probability), 'DEFENSIVE_MIXTURE')
    log_density = (successes[...,None] * np.log(proposals/probability)
        + (trials-successes)[...,None] * (np.log1p(-proposals)-np.log1p(-probability))
        + np.log(mixture))
    return -logsumexp(log_density, axis=-1)


def bounded_mean_interval(mean, samples, bound, confidence=.95):
    """Two-sided Chernoff inversion for IID variables in [0,bound].

    For Y in [0,1], convexity gives E exp(tY) <= 1-mu+mu exp(t).
    Hence each tail is bounded by exp(-n KL(observed_mean || mu)).
    Invert the two tails at (1-confidence)/2. This is not a binomial
    Clopper-Pearson interval and accepts noninteger weighted scores.
    """
    require(0 <= mean <= bound and samples > 0 and bound > 0 and 0 < confidence < 1, 'BOUNDED_MEAN_INPUT')
    a = mean/bound
    c = math.log(2/(1-confidence))/samples
    if a == 0:
        low, high = 0., -math.expm1(-c)
    elif a == 1:
        low, high = math.exp(-c), 1.
    else:
        def residual(mu):
            return a*(math.log(a)-math.log(mu)) + (1-a)*(math.log1p(-a)-math.log1p(-mu)) - c
        tiny, almost_one = np.nextafter(0.,1.), np.nextafter(1.,0.)
        low = (0. if residual(tiny) <= 0 else
               brentq(residual,tiny,a,xtol=1e-15,rtol=8*np.finfo(float).eps))
        high = (1. if residual(almost_one) <= 0 else
                brentq(residual,a,almost_one,xtol=1e-15,rtol=8*np.finfo(float).eps))
    return dict(lower=low*bound,upper=high*bound,confidence_level=confidence,
                method='two_sided_bounded_mean_Chernoff_inversion',score_bound=bound,
                scope='conditional_IID_auxiliary_weighted_scores')


def summarize_score(total, squares, samples, bound, confidence):
    mean = total/samples
    numerator = squares-total*mean
    tolerance = 32*np.finfo(float).eps*max(squares,total*mean)
    require(numerator >= -tolerance, 'NEGATIVE_EMPIRICAL_VARIANCE')
    variance = max(0.,numerator)/(samples-1)
    interval = bounded_mean_interval(mean,samples,bound,confidence)
    # The target is a probability even though an individual score may exceed 1.
    interval['upper'] = min(1.,interval['upper'])
    return dict(estimate=mean,empirical_score_variance=variance,monte_carlo_standard_error=math.sqrt(variance/samples),
                conditional_interval=interval,score_sum=total,score_square_sum=squares)


def refractory_tail_bound(config, conditional, report, histogram):
    """Bound a third spike using the archived second-spike stopping times.

    With minimum interspike spacing r and an exclusive window end b, three
    spikes imply T2 <= b-1-r. The stopped P/Q weights estimate that enclosing
    event without simulating feedback after T2. Only conditional auxiliary
    uncertainty is covered; source-history and simultaneous coverage remain
    outside this calculation. Callers must authenticate all input artifacts.
    """
    require(config['dt_ms'] in (.05, .025, .0125), 'BOUND_SOURCE_CLOCK')
    dt = config['dt_ms'] * 1e-3
    constants = source._constants(config, 'E')
    tau, stop = conditional['first_spike_tick'], conditional['window_stop_tick']
    require(type(tau) is int and type(stop) is int and 0 <= tau < stop
            and conditional['population'] == 'E'
            and conditional['timing'] == 'first_spike_reset_with_preinput_ampa'
            and conditional['fresh_input_included'] is False
            and conditional['state']['lastspike'] == tau * dt
            and conditional['state']['not_refractory'] is False
            and conditional['state']['v'] == constants['reset'], 'BOUND_FIRST_SPIKE_RESET')
    refractory_ticks = int(np.asarray((constants['refractory'] + 1e-3 * dt) / dt, dtype=np.int64))
    ticks = np.asarray(histogram['tick'])
    require(ticks.dtype.kind in 'iu' and np.array_equal(ticks, np.arange(tau + 1, stop)),
            'BOUND_HISTOGRAM_CLOCK')
    require(report['status'] == 'COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE'
            and report['target'] == 'second_focal_spike_before_window_end_given_first_spike_past_and_other_unit_future_innovations',
            'BOUND_SECOND_SPIKE_TARGET')
    samples = report['samples']
    require(type(samples) is int and samples > 0, 'BOUND_SAMPLE_COUNT')
    probability = native._RATE_HZ * dt
    proposals = np.asarray(report['proposal_probabilities'], dtype=np.float64)
    mixture = np.asarray(report['mixture_probabilities'], dtype=np.float64)
    log_likelihood_ratio(np.array([0]), np.array([0]), probability, proposals, mixture)
    bound = 1 / float(mixture[proposals == probability].sum())
    require(report['nominal_probability'] == probability and report['weight_bound'] == bound,
            'BOUND_SOURCE_PROPOSAL')
    hits = np.asarray(histogram['proposal_hits'])
    require(hits.dtype.kind in 'iu' and hits.shape == (len(proposals), len(ticks))
            and np.all(hits >= 0) and int(hits.sum()) == report['proposal_hits']
            and report['proposal_no_hits'] == samples - int(hits.sum()) >= 0,
            'BOUND_HIT_DENOMINATOR')
    weights = np.asarray(histogram['stopped_weight_sum'])
    require(weights.dtype == np.float64 and weights.shape == ticks.shape
            and np.isfinite(weights).all() and np.all(weights >= 0), 'BOUND_WEIGHT_HISTOGRAM')
    totals = hits.sum(axis=0)
    require(np.all(weights[totals == 0] == 0)
            and np.all(weights <= np.nextafter(totals * bound, np.inf)), 'BOUND_WEIGHT_SUPPORT')
    total = math.fsum(weights.tolist())
    require(math.isclose(total, report['stopped_prefix']['score_sum'], rel_tol=1e-12, abs_tol=0),
            'BOUND_WEIGHT_TOTAL')
    require(np.all(totals[ticks < tau + refractory_ticks] == 0), 'BOUND_REFRACTORY_SUPPORT')
    cutoff = stop - 1 - refractory_ticks
    score_sum = math.fsum(weights[ticks <= cutoff].tolist())
    estimate = score_sum / samples
    confidence = report['stopped_prefix']['conditional_interval']['confidence_level']
    structural_zero = tau + 2 * refractory_ticks >= stop
    interval = bounded_mean_interval(estimate, samples, bound, confidence)
    upper = 0. if structural_zero else min(1., interval['upper'])
    return dict(status='COMPLETE_CONDITIONAL_REFRACTORY_THIRD_SPIKE_BOUND',
        first_spike_tick=tau, window_stop_tick=stop, dt_ms=config['dt_ms'],
        minimum_interspike_ticks=refractory_ticks, second_spike_cutoff_inclusive=cutoff,
        structural_zero=structural_zero, enclosing_second_spike_score_sum=score_sum,
        enclosing_second_spike_estimate=estimate, samples=samples,
        enclosing_event_conditional_interval=interval, third_spike_probability_upper_bound=upper,
        bound_basis='minimum_interspike_spacing' if structural_zero else 'refractory_truncated_stopped_weights',
        scope='One authenticated source history under nominal IID focal innovations; no source-population or simultaneous coverage',
        post_second_spike_feedback_simulated=False, finite_PRNG_law_certified=False,
        population_history_uncertainty_established=False, third_spike_probability_established=False,
        third_spike_probability_or_bound_qualified=False, scientific_fidelity_qualified=False,
        confirmation_execution_released=False, new_full_network_episodes=0,
        new_independent_source_blocks=0, production_episodes=0)


def run(config, conditional, forcing, settings, output):
    samples, batch_size = settings['samples'], settings['batch_size']
    # Preserve the same multiplication grouping used by the accepted nominal sampler.
    p = native._RATE_HZ * (config['dt_ms'] * 1e-3)
    proposals = p*np.asarray(settings['proposal_multipliers'],dtype=np.float64)
    mixture = np.asarray(settings['mixture_probabilities'],dtype=np.float64)
    log_likelihood_ratio(np.array([0]),np.array([0]),p,proposals,mixture)
    bound = 1/float(mixture[proposals == p].sum())
    rng = np.random.RandomState(np.asarray(settings['seed_components'],dtype=np.uint32))
    components = rng.choice(len(proposals),size=samples,p=mixture)
    slots = len(forcing['tick']); tau=conditional['first_spike_tick']
    prefix_hist=np.zeros(slots);full_hist=np.zeros(slots);hit_hist=np.zeros((len(proposals),slots),dtype=np.int64)
    sums=[];squares=[];full_sums=[];full_squares=[];products=[];witnesses=[]
    input_digest=hashlib.sha256();maximum_weight=0.;minimum_log_weight=0.;hits=0;begin=time.perf_counter()
    for start in range(0,samples,batch_size):
        labels=components[start:min(start+batch_size,samples)]
        draws=rng.binomial(native._BINOMIAL_N,proposals[labels,None],size=(len(labels),slots)).astype(np.uint16)
        input_digest.update(memoryview(draws))
        try:
            first=first_passage_ticks(config,conditional,forcing,draws)
            rows=np.flatnonzero(first>=0);prefix_lengths=first[rows]-tau
            cumulative=np.cumsum(draws[rows],axis=1,dtype=np.int64)
            prefix_successes=cumulative[np.arange(len(rows)),prefix_lengths-1]
            total_successes=cumulative[:,-1]
            log_prefix=log_likelihood_ratio(prefix_successes,prefix_lengths*native._BINOMIAL_N,p,proposals,mixture)
            log_full=log_likelihood_ratio(total_successes,np.full(len(rows),slots*native._BINOMIAL_N,dtype=np.int64),p,proposals,mixture)
            weight=np.exp(log_prefix);full_weight=np.exp(log_full)
            require(np.all(np.isfinite(weight)) and np.all(np.isfinite(full_weight))
                    and np.all(weight>0) and np.all(full_weight>0), 'NONFINITE_OR_UNDERFLOWED_POSITIVE_WEIGHT')
            require(np.all(weight<=np.nextafter(bound,np.inf)) and np.all(full_weight<=np.nextafter(bound,np.inf)), 'DEFENSIVE_WEIGHT_BOUND')
        except Exception:
            np.savez_compressed(output/('failed_batch_'+str(start)+'.npz'),counts=draws,components=labels,batch_start=np.array(start))
            raise
        positions=first[rows]-int(forcing['tick'][0])
        prefix_hist += np.bincount(positions,weights=weight,minlength=slots)
        full_hist += np.bincount(positions,weights=full_weight,minlength=slots)
        for label in range(len(proposals)):
            hit_hist[label] += np.bincount(positions[labels[rows]==label],minlength=slots)
        sums.append(float(weight.sum()));squares.append(float(np.dot(weight,weight)))
        full_sums.append(float(full_weight.sum()));full_squares.append(float(np.dot(full_weight,full_weight)))
        products.append(float(np.dot(weight,full_weight)));hits+=len(rows)
        if len(rows):
            maximum_weight=max(maximum_weight,float(weight.max()),float(full_weight.max()))
            minimum_log_weight=min(minimum_log_weight,float(log_prefix.min()),float(log_full.min()))
        for j in range(min(len(rows),settings['maximum_witnesses']-len(witnesses))):
            witnesses.append(dict(sample_index=start+int(rows[j]),component=int(labels[rows[j]]),
                first_hit_tick=int(first[rows[j]]),prefix_input_count=int(prefix_successes[j]),
                total_input_count=int(total_successes[j]),log_prefix_weight=float(log_prefix[j]),
                log_full_weight=float(log_full[j]),counts=draws[rows[j]].copy()))
        progress=dict(completed=start+len(labels),proposal_hits=hits,prefix_score_sum=math.fsum(sums),wall_seconds=time.perf_counter()-begin)
        (output/'sampling_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress),flush=True)
    total,second=math.fsum(sums),math.fsum(squares)
    full_total,full_second=math.fsum(full_sums),math.fsum(full_squares)
    require(int(hit_hist.sum())==hits and abs(float(prefix_hist.sum())-total)<=1e-12*max(abs(total),np.finfo(float).tiny)
            and abs(float(full_hist.sum())-full_total)<=1e-12*max(abs(full_total),np.finfo(float).tiny), 'WEIGHTED_HISTOGRAM_SUM')
    state=rng.get_state()
    result=dict(status='COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE',samples=samples,proposal_hits=hits,
        proposal_no_hits=samples-hits,component_samples=np.bincount(components,minlength=len(proposals)).tolist(),
        component_hits=hit_hist.sum(axis=1).tolist(),nominal_probability=p,proposal_probabilities=proposals.tolist(),
        proposal_probability_ratios=[list(float(value).as_integer_ratio()) for value in proposals],
        mixture_probabilities=mixture.tolist(),weight_bound=bound,maximum_observed_weight=maximum_weight,
        minimum_observed_log_weight=minimum_log_weight,
        stopped_prefix=summarize_score(total,second,samples,bound,settings['confidence_level']),
        full_path=summarize_score(full_total,full_second,samples,bound,settings['confidence_level']),
        paired_score_cross_sum=math.fsum(products),seed_components=settings['seed_components'],batch_size=batch_size,
        draw_layout='one_component_per_complete_sample; all_labels_then_sample_major_input_batches',
        input_draw_sha256=input_digest.hexdigest(),component_draw_sha256=hashlib.sha256(memoryview(components)).hexdigest(),
        final_rng_state=dict(algorithm=state[0],key=state[1].tolist(),position=int(state[2]),has_gauss=int(state[3]),cached_gaussian=float(state[4])),
        witness_selection='first_hits_in_original_sample_order_for_source_validation_only',witness_count=len(witnesses),
        numpy_version=np.__version__,wall_seconds=time.perf_counter()-begin,reference_archive=settings['reference']['archive'],
        reference_delivery_sha256=settings['reference']['delivery_sha256'],
        target='second_focal_spike_before_window_end_given_first_spike_past_and_other_unit_future_innovations',
        finite_PRNG_law_certified=False,population_history_uncertainty_established=False,
        population_first_passage_probability_established=False,third_spike_probability_established=False,
        new_full_network_episodes=0,new_independent_source_blocks=0,confirmation_execution_released=False,
        scientific_fidelity_qualified=False,production_episodes=0)
    np.savez_compressed(output/'weighted_first_hit_histogram.npz',tick=forcing['tick'],proposal_hits=hit_hist,
                        stopped_weight_sum=prefix_hist,full_weight_sum=full_hist)
    if witnesses:
        np.savez_compressed(output/'first_hit_witnesses.npz',
            counts=np.stack([row['counts'] for row in witnesses]),
            **{key:np.asarray([row[key] for row in witnesses]) for key in witnesses[0] if key!='counts'})
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-config','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--refractory-bound-from',type=Path,
                        help='Read an existing importance result; generate no new trajectories.')
    args=parser.parse_args()
    if args.refractory_bound_from is not None:
        if args.config is not None:
            parser.error('--config is only used for new importance sampling')
        report=json.loads((args.refractory_bound_from/'report.json').read_text())
        conditional=json.loads((args.reference/'conditional_initial.json').read_text())
        config=yaml.safe_load(args.source_config.read_text())
        with np.load(args.refractory_bound_from/'weighted_first_hit_histogram.npz',allow_pickle=False) as data:
            result=refractory_tail_bound(config,conditional,report,data)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    if args.config is None:
        parser.error('--config is required for new importance sampling')
    settings=yaml.safe_load(args.config.read_text())
    require(settings['samples']==1000000 and settings['batch_size']==10000
            and settings['proposal_multipliers']==[1.,2.,4.,8.] and settings['mixture_probabilities']==[.25]*4
            and settings['seed_components']==[20261009,95001,28022,2004,1]
            and settings['confidence_level']==.95 and settings['maximum_witnesses']==8, 'FROZEN_PROPOSAL_PLAN')
    config=yaml.safe_load(args.source_config.read_text())
    conditional=json.loads((args.reference/'conditional_initial.json').read_text())
    require(conditional['unit_index']==28022 and conditional['first_spike_tick']==2004
            and conditional['window_stop_tick']==2100 and config['dt_ms']==.05, 'REGISTERED_REFERENCE')
    with np.load(args.reference/'reference_window.npz',allow_pickle=False) as data:
        forcing={key:data[key] for key in data.files}
    args.output.mkdir(parents=True,exist_ok=False)
    result=run(config,conditional,forcing,settings,args.output)
    print(json.dumps({key:result[key] for key in ('status','samples','proposal_hits','stopped_prefix','full_path')}),flush=True)


if __name__=='__main__':
    main()

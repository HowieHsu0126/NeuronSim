"""Whole-block empirical sampling precision for the fixed PFC calibration cohort.

These conditional bootstrap diagnostics are not confidence intervals. Covariance
resamples are centered at their exact conditional expectation, (B-1)/B times
the sample covariance contrast. Linear field/CDF means are centered at the
original contrast. Zero empirical spread remains an undefined standardized
endpoint, never evidence of zero population uncertainty.
"""
from __future__ import annotations

import numpy as np

from .pfc_delay_analysis import PHASES, require
from .pfc_phase_responses import METRICS as RESPONSE_METRICS, CONTRASTS as RESPONSE_CONTRASTS
from .source_variant_analysis import (
    covariance_from_gram, paired_bootstrap_multiplicities, time_inner_product_gram,
)

CONDITIONS = ('omission', 'cue', 'strong_cue')
SCHEMA = 'pfc_numerical_precision_v1'
STATISTICS = ('estimate', 'bootstrap_mean', 'analytic_bootstrap_mean',
              'bootstrap_bias_estimate', 'analytic_bootstrap_bias',
              'bootstrap_standard_error', 'bootstrap_mean_monte_carlo_se')


def validate_settings(settings, cohorts):
    require(settings['schema'] == SCHEMA
            and settings['scope'] == 'development_calibration'
            and settings['inference'] == 'sampling_precision_only'
            and settings['independent_unit'] == 'joint_initial_background_rng_block',
            'development-only whole-block precision settings required')
    seeds = settings['joint_rng_seeds']
    require(isinstance(seeds, list) and len(seeds) >= 3
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)
            and seeds == sorted(set(seeds)), 'complete ordered precision cohort required')
    require(type(settings['bootstrap_replicates']) is int and settings['bootstrap_replicates'] >= 2
            and type(settings['bootstrap_seed']) is int and 0 <= settings['bootstrap_seed'] < 2**32,
            'explicit bootstrap count and computational seed required')
    require(len(cohorts) == 3 and {c['config']['dt_ms'] for c in cohorts.values()} == {.05,.025,.0125},
            'three declared calibration clocks required')
    for cohort in cohorts.values():
        require(cohort['config'].get('source_replication') == dict(
                    purpose='numerical_calibration',
                    independent_unit='joint_initial_background_rng_block', joint_rng_seeds=seeds)
                and cohort['report']['joint_rng_blocks'] == seeds
                and cohort['report']['completed_source_cases'] == 3*len(seeds)
                and cohort['report']['registered_source_cases'] == 3*len(seeds),
                'complete declared numerical-calibration blocks and all controls required')
    return seeds


def resampled_lag_covariance(rates, lag, multiplicities):
    """Reuse the verified covariance engine for directed phase-contained lags."""
    x = np.asarray(rates, dtype=float)
    require(x.ndim == 3 and min(x.shape) > 0 and np.isfinite(x).all()
            and type(lag) is int and lag >= 0, 'finite rates and nonnegative integer lag required')
    if lag >= x.shape[1]:
        return None
    paired = x if lag == 0 else np.concatenate((x[:,:-lag],x[:,lag:]),axis=2)
    values = covariance_from_gram(time_inner_product_gram(paired),multiplicities)
    return values if lag == 0 else values[:,:x.shape[2],x.shape[2]:]


class PrecisionSummary:
    """Accumulate endpoint summaries and a shared-draw maximum without a large draw table."""
    def __init__(self, replicates):
        require(type(replicates) is int and replicates >= 2, 'at least two bootstrap draws required')
        self.replicates = replicates
        self.maximum = np.zeros(replicates)
        self.defined = 0
        self.variable = 0
        self.degenerate = 0
        self.undefined = 0

    def add(self, point, draws, analytic_center, identities):
        point, draws, center = map(lambda x: np.asarray(x,dtype=float), (point,draws,analytic_center))
        require(point.ndim == center.ndim == 1 and point.shape == center.shape
                and draws.shape == (self.replicates,len(point)) and len(identities) == len(point)
                and np.isfinite(point).all() and np.isfinite(center).all() and np.isfinite(draws).all(),
                'all finite endpoint estimates and every bootstrap draw required')
        constant = np.all(draws == draws[0],axis=0)
        mean = draws.mean(axis=0)
        spread = draws.std(axis=0,ddof=1)
        # Exact constant resamples can acquire roundoff variance from mean subtraction.
        mean[constant], spread[constant] = draws[0,constant], 0.
        variable = ~constant
        require(np.all(spread[variable] > 0) and np.isfinite(spread).all(),
                'nonconstant bootstrap spread must remain finite and positive')
        if variable.any():
            z = np.abs((draws[:,variable]-center[variable])/spread[variable])
            self.maximum = np.maximum(self.maximum,z.max(axis=1))
        self.defined += len(point)
        self.variable += int(variable.sum())
        self.degenerate += int(constant.sum())
        return [dict(identity, estimate=float(point[i]),bootstrap_mean=float(mean[i]),
            analytic_bootstrap_mean=float(center[i]),bootstrap_bias_estimate=float(mean[i]-point[i]),
            analytic_bootstrap_bias=float(center[i]-point[i]),bootstrap_standard_error=float(spread[i]),
            bootstrap_mean_monte_carlo_se=float(spread[i]/np.sqrt(self.replicates)),
            status='EMPIRICALLY_DEGENERATE' if constant[i] else 'ESTIMATED_EMPIRICAL_SPREAD')
            for i,identity in enumerate(identities)]

    def add_undefined(self, identities):
        self.undefined += len(identities)
        return [dict(identity,**{key:None for key in STATISTICS},status='UNDEFINED_PHASE_CONTAINED_LAG')
                for identity in identities]

    def finish(self):
        maximum = self.maximum if self.variable else np.full(self.replicates,np.nan)
        return maximum, dict(requested_endpoints=self.defined+self.undefined,
            finite_endpoints=self.defined,positive_spread_endpoints=self.variable,
            empirically_degenerate_endpoints=self.degenerate,undefined_lag_endpoints=self.undefined,
            maximum_status='DEFINED_OVER_POSITIVE_SPREAD_ENDPOINTS_ONLY' if self.variable else 'UNDEFINED_NO_POSITIVE_SPREAD',
            maximum_excludes_endpoints=self.degenerate+self.undefined,
            maximum_is_simultaneous_confidence_statement=False,
            bootstrap_standard_error_ddof=1,
            zero_empirical_spread_implies_zero_population_uncertainty=False)


def analyze(cohorts, cdf_payload, covariances, lags_ms, settings):
    seeds = validate_settings(settings,cohorts)
    labels = sorted(cohorts,key=lambda label:cohorts[label]['config']['dt_ms'],reverse=True)
    pairs = list(zip(labels[:-1],labels[1:]))
    require(cdf_payload is not None and np.array_equal(cdf_payload['source_rng_seeds'],seeds),
            'complete block-level count CDFs required for precision analysis')
    multiplicities = paired_bootstrap_multiplicities(len(seeds),settings['bootstrap_replicates'],settings['bootstrap_seed'])
    weights = multiplicities/len(seeds)
    summary = PrecisionSummary(len(weights))
    mean_rows, covariance_rows, cdf_rows = [],[],[]
    fields = len(next(iter(cohorts.values()))['sizes'])
    for ci,condition in enumerate(CONDITIONS):
        for pi,(phase,(start,stop)) in enumerate(PHASES.items()):
            rates = {label:cohort['fields'][condition+'_field_counts'][:,round(start/5):round(stop/5)]
                     /(cohort['sizes']*.005) for label,cohort in cohorts.items()}
            for left,right in pairs:
                difference = rates[left].mean(axis=1)-rates[right].mean(axis=1)
                point = difference.mean(axis=0)
                identities = [dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                                   field_index=i,units='Hz') for i in range(fields)]
                mean_rows.extend(summary.add(point,weights@difference,point,identities))
            for lag_ms in lags_ms:
                # Only neighboring arrays are retained at once, with identical resample multiplicities.
                previous = resampled_lag_covariance(rates[labels[0]],round(lag_ms/5),multiplicities)
                for left,right in pairs:
                    following = resampled_lag_covariance(rates[right],round(lag_ms/5),multiplicities)
                    identities = [dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                        lag_ms=lag_ms,leading_field=i,trailing_field=j,units='Hz^2')
                        for i in range(fields) for j in range(fields)]
                    if previous is None or following is None:
                        require(previous is None and following is None,'matched phase lag domains required')
                        covariance_rows.extend(summary.add_undefined(identities))
                    else:
                        point = (covariances[left,condition,phase,lag_ms]-covariances[right,condition,phase,lag_ms]).ravel()
                        draws = (previous-following).reshape(len(weights),-1)
                        covariance_rows.extend(summary.add(point,draws,(len(seeds)-1)/len(seeds)*point,identities))
                    previous = following
            for left,right in pairs:
                a,b = cdf_payload[left+'_cdf'][ci,:,pi],cdf_payload[right+'_cdf'][ci,:,pi]
                for wi,width in enumerate(cdf_payload['window_ms']):
                    difference = (a[:,wi]-b[:,wi]).reshape(len(seeds),-1)
                    point = difference.mean(axis=0)
                    identities = [dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                        window_ms=float(width),field_index=i,count_threshold=int(threshold),units='probability')
                        for i in range(fields) for threshold in cdf_payload['count_support']]
                    cdf_rows.extend(summary.add(point,weights@difference,point,identities))
    response_rows,paired_rows=[],[]
    responses=[cohorts[label].get('phase_responses') for label in labels]
    if any(value is not None for value in responses):
        require(all(value is not None for value in responses),'complete source response clocks required')
        for left,right in pairs:
            a,b=cohorts[left]['phase_responses']['values'],cohorts[right]['phase_responses']['values']
            for ci,condition in enumerate(CONDITIONS):
                for pi,phase in enumerate(PHASES):
                    difference=a[ci,:,pi]-b[ci,:,pi]
                    point=difference.mean(axis=0)
                    identities=[dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                        metric=metric,units='Hz') for metric in RESPONSE_METRICS]
                    response_rows.extend(summary.add(point,weights@difference,point,identities))
            for name,positive,negative in RESPONSE_CONTRASTS:
                i,j=CONDITIONS.index(positive),CONDITIONS.index(negative)
                for pi,phase in enumerate(PHASES):
                    difference=(a[i,:,pi]-a[j,:,pi])-(b[i,:,pi]-b[j,:,pi])
                    point=difference.mean(axis=0)
                    identities=[dict(left_cohort=left,right_cohort=right,contrast=name,phase=phase,
                        metric=metric,units='Hz') for metric in RESPONSE_METRICS]
                    paired_rows.extend(summary.add(point,weights@difference,point,identities))
    maximum,diagnostic = summary.finish()
    report = dict(status='COMPLETE_CALIBRATION_SAMPLING_PRECISION',settings=settings,
        independent_joint_blocks=len(seeds),joint_rng_seeds=seeds,source_episodes_reused=9*len(seeds),
        bootstrap_replicates=len(weights),bootstrap_seed=settings['bootstrap_seed'],
        all_clocks_conditions_and_endpoints_share_resample_indices=True,
        covariance_bootstrap_center='(B-1)/B times signed sample covariance contrast',
        linear_bootstrap_center='signed empirical block-mean contrast',
        families=['field_mean_rate','signed_field_lag_covariance','marginal_unit_window_count_cdf'],
        unmeasured_by_this_precision_analysis=['individual_unit_joint_distribution','circular_resultant_and_control_response','other_circuit_instances'],
        diagnostics=diagnostic,confidence_intervals=False,p_values=False,q_values=False,
        confirmation_execution_released=False,numerical_equivalence_qualified=False,
        scientific_fidelity_qualified=False,production_episodes=0,
        interpretation='Conditional empirical whole-block bootstrap sampling spread, with all endpoints retained. It does not select error margins or confirmation replication, establish a confidence band, or qualify unobserved count tails or full joint activity.')
    tables={'field_mean_sampling_precision.csv':mean_rows,
            'field_lag_covariance_sampling_precision.csv':covariance_rows,
            'count_cdf_sampling_precision.csv':cdf_rows}
    if response_rows:
        tables['source_response_sampling_precision.csv']=response_rows
        tables['paired_response_sampling_precision.csv']=paired_rows
        report['families']+=['source_angle_vector_and_region_response','paired_source_control_response']
        report['unmeasured_by_this_precision_analysis'].remove('circular_resultant_and_control_response')
        report['direction_undefined_blocks_retained']=True
        report['source_angle_vector_definition']='sum(E count * cos/sin(source preference)) / (N_E*phase_seconds)'
    payload=dict(bootstrap_multiplicities=multiplicities,source_rng_seeds=np.asarray(seeds),
                 max_standardized_empirical_fluctuation=maximum)
    return tables,payload,report

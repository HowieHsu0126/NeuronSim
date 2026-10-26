"""Assess complete PFC uncertainty using joint bootstrap and primitive outer bounds.

Calibration-only: the bootstrap coverage approximation and data-selected routing
are not confirmation evidence. Constant primitives retain their physical ranges.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import shutil
from pathlib import Path
import platform

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS
from .pfc_delay_analysis import PHASES
from .provenance import freeze_json, sha256_file

IDENTITY = {
    'field_mean': ('left_cohort','right_cohort','condition','phase','field_index'),
    'field_lag_covariance': ('left_cohort','right_cohort','condition','phase','lag_ms','leading_field','trailing_field'),
    'count_cdf': ('left_cohort','right_cohort','condition','phase','window_ms','field_index','count_threshold'),
    'source_response': ('left_cohort','right_cohort','condition','phase','metric'),
    'paired_response': ('left_cohort','right_cohort','contrast','phase','metric'),
}
PRECISION = {family: family+'_sampling_precision.csv' for family in IDENTITY}
STRUCTURAL = 'STRUCTURAL_ZERO_CONTRAST'
UNDEFINED = 'UNDEFINED_NO_WITHIN_PHASE_PAIRS'
EMPIRICAL = 'EMPIRICALLY_DEGENERATE_UNCERTAIN'


def identity_key(family, row):
    numeric = {'field_index','lag_ms','leading_field','trailing_field','window_ms','count_threshold'}
    return (family, *(float(row[key]) if key in numeric else row[key] for key in IDENTITY[family]))


def bootstrap_primitives(values, multiplicities, previous_maximum, chunk_size=256):
    """Extend a maximum draw by draw, keeping empirical constants unestimated."""
    x = np.asarray(values, dtype=float)
    m = np.asarray(multiplicities)
    require(x.ndim == 2 and np.isfinite(x).all() and m.ndim == 2 and m.shape[1] == x.shape[1]
            and m.dtype.kind in 'iu' and np.all(m >= 0) and np.all(m.sum(1) == x.shape[1]),
            'COMPLETE_SHARED_WHOLE_BLOCK_RESAMPLES_REQUIRED')
    maximum = np.asarray(previous_maximum, dtype=float).copy()
    require(maximum.shape == (len(m),) and np.isfinite(maximum).all() and np.all(maximum >= 0),
            'ONE_PRIOR_MAXIMUM_PER_IDENTICAL_DRAW_REQUIRED')
    constant = np.all(x == x[:, :1], axis=1)
    point, spread, influence = x.mean(1), np.zeros(len(x)), np.full(len(x), np.nan)
    point[constant] = x[constant, 0]
    weights = m/x.shape[1]
    for first in range(0, len(x), chunk_size):
        indices = np.arange(first, min(first+chunk_size, len(x)))
        indices = indices[~constant[indices]]
        if not len(indices):
            continue
        centered = x[indices]-point[indices, None]
        draws = weights@centered.T
        sd = draws.std(0, ddof=1)
        require(np.all(sd > 0), 'NONCONSTANT_PRIMITIVE_BOOTSTRAP_SPREAD_REQUIRED')
        spread[indices] = sd
        maximum = np.maximum(maximum, np.max(np.abs(draws)/sd, axis=1))
        squared = centered*centered
        influence[indices] = squared.max(1)/squared.sum(1)
    return point, spread, maximum, influence


def monotone_cdf_envelope(support, lower, upper, thresholds, physical_maximum):
    """Propagate simultaneous anchors, including only source-proven saturation."""
    support, thresholds = np.asarray(support), np.asarray(thresholds)
    lo, hi = np.asarray(lower, dtype=float).copy(), np.asarray(upper, dtype=float).copy()
    require(support.ndim == 1 and len(support) == len(lo) == len(hi)
            and np.all(np.diff(support) > 0) and np.all((lo >= 0) & (lo <= hi) & (hi <= 1)),
            'ORDERED_VALID_CDF_ANCHORS_REQUIRED')
    lo = np.maximum.accumulate(lo)
    hi = np.minimum.accumulate(hi[::-1])[::-1]
    require(np.all(lo <= hi), 'CROSSING_MONOTONE_CDF_ANCHORS')
    left = np.searchsorted(support, thresholds, side='right')-1
    right = np.searchsorted(support, thresholds, side='left')
    out_lo, out_hi = np.zeros(len(thresholds)), np.ones(len(thresholds))
    valid = left >= 0; out_lo[valid] = lo[left[valid]]
    valid = right < len(support); out_hi[valid] = hi[right[valid]]
    below = thresholds < 0; saturated = thresholds >= physical_maximum
    out_lo[below] = out_hi[below] = 0.
    out_lo[saturated] = out_hi[saturated] = 1.
    require(np.all(out_lo <= out_hi), 'CROSSING_CDF_ENVELOPE')
    return out_lo, out_hi


def signed_interval(lower, upper, signs):
    lo, hi, signs = map(np.asarray, (lower, upper, signs))
    require(lo.shape == hi.shape == signs.shape and np.all(lo <= hi)
            and np.all(np.isin(signs, [-1, 1])), 'SIGNED_PRIMITIVE_INTERVALS_REQUIRED')
    return float(np.where(signs > 0, lo, -hi).sum()), float(np.where(signs > 0, hi, -lo).sum())


def intersect_intervals(first, second):
    lo, hi = max(first[0], second[0]), min(first[1], second[1])
    require(lo <= hi, 'CROSSING_SIMULTANEOUS_CHANNELS')
    return lo, hi


def covariance_jackknife(kernels, pair_indices, blocks):
    """Delete-one pseudovalues of the unbiased order-two U statistic."""
    h = np.asarray(kernels, dtype=float)
    pairs = np.asarray(pair_indices)
    require(blocks >= 3 and h.ndim == 2 and h.shape[1] == blocks*(blocks-1)//2
            and pairs.shape == (h.shape[1], 2) and np.isfinite(h).all(), 'COMPLETE_COVARIANCE_PAIR_KERNELS_REQUIRED')
    require({tuple(pair) for pair in pairs.tolist()} == {(i,j) for i in range(blocks) for j in range(i+1,blocks)},
            'EACH_UNORDERED_BLOCK_PAIR_ONCE_REQUIRED')
    incidence = np.zeros((len(pairs), blocks))
    incidence[np.arange(len(pairs)), pairs[:,0]] = 1.
    incidence[np.arange(len(pairs)), pairs[:,1]] = 1.
    point = h.mean(1)
    g = h@incidence/(blocks-1)
    projection = (blocks-1)/(blocks-2)*(g-point[:,None])
    pseudovalues = point[:,None]+2*projection
    se = pseudovalues.std(1, ddof=1)/np.sqrt(blocks)
    squares = projection*projection
    denominator = squares.sum(1)
    influence = np.divide(squares.max(1), denominator, out=np.full(len(h), np.nan), where=denominator > 0)
    residual = h-point[:,None]-projection[:,pairs[:,0]]-projection[:,pairs[:,1]]
    energy = ((h-point[:,None])**2).sum(1)
    residual_fraction = np.divide((residual*residual).sum(1), energy, out=np.full(len(h), np.nan), where=energy > 0)
    return pseudovalues, se, influence, residual_fraction


def write_rows(path, rows):
    require(bool(rows), 'NONEMPTY_OUTPUT_TABLE_REQUIRED')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_npz(path, report):
    require(sha256_file(path) == report['outputs'][path.name]['sha256'], 'BOUND_INPUT_CHANGED:'+path.name)
    with np.load(path, allow_pickle=False) as stored:
        return {name:stored[name] for name in stored.files}


def prepare_primitives(cdf, responses, labels, physical, response_ranges):
    values, identities, lookup, lower, upper = [], [], {}, [], []
    windows, support = cdf['window_ms'].tolist(), cdf['count_support'].tolist()
    fields = len(cdf['field_unit_counts'])
    def add(key, value, lo, hi, identity):
        require(key not in lookup, 'DUPLICATE_PRIMITIVE_IDENTITY')
        lookup[key] = len(values); values.append(value); identities.append(identity); lower.append(lo); upper.append(hi)
    for label in labels:
        x = cdf[label+'_cdf']; y = responses[label+'_values']
        for ci,condition in enumerate(CONDITIONS):
            for pi,phase in enumerate(PHASES):
                for wi,width in enumerate(windows):
                    for field in range(fields):
                        maximum = physical[float(width), field]
                        for ki,threshold in enumerate(support):
                            identity = dict(family='primitive_cdf',cohort=label,condition=condition,phase=phase,
                                            window_ms=width,field_index=field,count_threshold=threshold)
                            lo = 1. if threshold >= maximum else 0.
                            add(('cdf',label,condition,phase,float(width),field,threshold),x[ci,:,pi,wi,field,ki],lo,1.,identity)
                for mi,metric in enumerate(METRICS):
                    lo,hi = response_ranges[metric]
                    add(('response',label,condition,phase,metric),y[ci,:,pi,mi],lo,hi,
                        dict(family='primitive_response',cohort=label,condition=condition,phase=phase,metric=metric))
    return np.asarray(values), identities, lookup, np.asarray(lower), np.asarray(upper)


def binomial_probability_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact binomial upper bound by CDF inversion, for a fixed n."""
    require(type(successes) is int and type(trials) is int and 0 <= successes <= trials
            and trials > 0 and 0 < alpha < 1, 'VALID_FIXED_BINOMIAL_DESIGN_REQUIRED')
    if successes == trials:
        return 1.
    if successes == 0:
        return -math.expm1(math.log(alpha)/trials)
    lower,upper = 0.,1.
    for _ in range(64):
        probability = (lower+upper)/2
        cumulative = math.fsum(math.comb(trials,k)*probability**k*(1-probability)**(trials-k)
                               for k in range(successes+1))
        if cumulative > alpha:
            lower = probability
        else:
            upper = probability
    return upper


def frozen_tail_departures(values, frozen_centers):
    """Count any frozen-primitive departure once per whole independent block."""
    x,c = np.asarray(values),np.asarray(frozen_centers)
    require(x.ndim == 2 and c.shape == (x.shape[0],) and x.shape[1] > 0
            and np.isfinite(x).all() and np.isfinite(c).all(), 'FROZEN_PRIMITIVE_BY_BLOCK_ARRAY_REQUIRED')
    return np.any(x != c[:,None],axis=0)


def construct_bands(domains, precision, primitive, lookup, support, physical, critical, tail_probability_bound=None):
    point, se, physical_lo, physical_hi = primitive
    lo, hi = physical_lo.copy(), physical_hi.copy()
    variable = se > 0
    if tail_probability_bound is not None:
        require(0 <= tail_probability_bound <= 1, 'VALID_JOINT_TAIL_PROBABILITY_REQUIRED')
        selected = ~variable & (physical_lo < physical_hi)
        lo[selected] = point[selected]-tail_probability_bound*(point[selected]-physical_lo[selected])
        hi[selected] = point[selected]+tail_probability_bound*(physical_hi[selected]-point[selected])
    lo[variable] = np.maximum(lo[variable], point[variable]-critical*se[variable])
    hi[variable] = np.minimum(hi[variable], point[variable]+critical*se[variable])
    require(np.all(lo <= hi), 'CROSSING_PRIMITIVE_BANDS')
    cache, rows = {}, []
    for index,row in enumerate(domains):
        family, status = row['family'], row['definition_status']
        ident = json.loads(row['endpoint_identity'])
        out = dict(domain_row_index=index, family=family, endpoint_identity=row['endpoint_identity'], definition_status=status,
                   estimate=row['estimate'], lower='', upper='', channel='undefined', direct_bootstrap_standard_error='')
        if status == UNDEFINED:
            rows.append(out); continue
        if status == STRUCTURAL:
            out.update(lower=0., upper=0., channel='source_identity'); rows.append(out); continue
        direct = precision.get(identity_key(family,ident))
        band, channels = None, []
        if direct is not None and direct['status'] == 'ESTIMATED_EMPIRICAL_SPREAD':
            center, scale = float(direct['estimate']), float(direct['bootstrap_standard_error'])
            require(scale > 0, 'POSITIVE_DIRECT_STANDARD_ERROR_REQUIRED')
            band = (center-critical*scale, center+critical*scale); channels.append('direct_contrast')
            out['direct_bootstrap_standard_error'] = scale
        if family == 'count_cdf':
            condition,phase,width,field,threshold = (ident['condition'],ident['phase'],float(ident['window_ms']),int(ident['field_index']),int(ident['count_threshold']))
            intervals = []
            for label in [ident['left_cohort'],ident['right_cohort']]:
                key = (label,condition,phase,width,field)
                if key not in cache:
                    indices = [lookup[('cdf',*key,k)] for k in support]
                    thresholds = np.arange(physical[width,field]+1)
                    cache[key] = monotone_cdf_envelope(support,lo[indices],hi[indices],thresholds,physical[width,field])
                a,b = cache[key]
                intervals.append((a[threshold],b[threshold]))
            outer = signed_interval([v[0] for v in intervals],[v[1] for v in intervals],[1,-1])
            band = outer if band is None else intersect_intervals(band,outer); channels.append('monotone_primitive_cdf')
        elif family in ('source_response','paired_response'):
            names = [ident['condition']] if family == 'source_response' else list({name:(pos,neg) for name,pos,neg in CONTRASTS}[ident['contrast']])
            indices = [lookup[('response',label,condition,ident['phase'],ident['metric'])]
                       for label in [ident['left_cohort'],ident['right_cohort']] for condition in names]
            signs = [1,-1] if len(names) == 1 else [1,-1,-1,1]
            outer = signed_interval(lo[indices],hi[indices],signs)
            band = outer if band is None else intersect_intervals(band,outer); channels.append('primitive_response')
        require(band is not None, 'UNSUPPORTED_DEFINED_ENDPOINT:'+str(index))
        out.update(lower=float(band[0]),upper=float(band[1]),channel='+'.join(channels)); rows.append(out)
    return rows, lo, hi


def run(config_path, project_root, output):
    config = yaml.safe_load(config_path.read_text())
    require(config['schema'] == 'pfc_confirmation_uncertainty_diagnostic_v1' and config['scope'] == 'development_calibration'
            and config['confirmation_execution_released'] is False and config['method_adopted_for_confirmation'] is False,
            'CALIBRATION_DIAGNOSTIC_ONLY_REQUIRED')
    if 'reuse_diagnostic' in config:
        return refine_tail_bounds(config_path, project_root, output, config)
    roots = {name:project_root/item['path'] for name,item in config['inputs'].items()}
    reports = {name:bound_report(roots[name],item['report_sha256']) for name,item in config['inputs'].items()}
    comparison, domain = reports['comparison'], reports['domain']
    require(comparison['status'] == 'COMPLETE_DESCRIPTIVE_FLUCTUATION_COMPARISON'
            and domain['status'] == 'COMPLETE_CALIBRATION_BLOCK_CDF_SCORES'
            and reports['covariance']['status'] == 'COMPLETE_CALIBRATION_COVARIANCE_PAIR_SCORES'
            and reports['linear']['status'] == 'COMPLETE_CALIBRATION_LINEAR_BLOCK_SCORES', 'COMPLETE_SCORE_RECOVERY_REQUIRED')
    domain_path = roots['domain']/'endpoint_domains.csv'
    require(sha256_file(domain_path) == domain['outputs'][domain_path.name]['sha256'], 'DOMAIN_BINDING_CHANGED')
    domains = read_rows(domain_path)
    require(len(domains) == domain['domain_rows'] == 84330, 'COMPLETE_DOMAIN_REQUIRED')
    for name in ('covariance','linear'):
        require(reports[name]['domain_table_sha256'] == sha256_file(domain_path), 'SCORE_DOMAIN_BINDING_CHANGED')
    cdf = load_npz(roots['comparison']/'block_count_cdfs.npz', comparison)
    responses = load_npz(roots['comparison']/'block_source_responses.npz', comparison)
    bootstrap = load_npz(roots['comparison']/'calibration_bootstrap.npz', comparison)
    covariance = load_npz(roots['covariance']/'paired_block_covariance_kernels.npz', reports['covariance'])
    seeds,labels = config['joint_rng_seeds'],config['cohorts']
    for data in (cdf,responses,bootstrap,covariance):
        require(np.array_equal(data['source_rng_seeds'],seeds), 'SHARED_BLOCK_IDENTITY_CHANGED')
    for data in (cdf,responses):
        require(np.array_equal(data['condition_names'],CONDITIONS) and np.array_equal(data['phase_names'],list(PHASES)),
                'SHARED_CONDITION_OR_PHASE_AXIS_CHANGED')
    require(np.array_equal(responses['metric_names'],METRICS), 'SOURCE_RESPONSE_METRICS_CHANGED')
    multiplicities = bootstrap['bootstrap_multiplicities']; previous = bootstrap['max_standardized_empirical_fluctuation']
    require(multiplicities.shape == (5000,len(seeds)) and comparison['calibration_precision']['diagnostics']['positive_spread_endpoints'] == 37044,
            'REGISTERED_COMPLETE_BOOTSTRAP_REQUIRED')
    precision = {}
    for family,name in PRECISION.items():
        path = roots['comparison']/name
        require(sha256_file(path) == comparison['outputs'][name]['sha256'], 'PRECISION_BINDING_CHANGED:'+name)
        for row in read_rows(path):
            key = identity_key(family,row); require(key not in precision,'DUPLICATE_PRECISION_IDENTITY'); precision[key] = row
    require(sum(row['status'] == 'ESTIMATED_EMPIRICAL_SPREAD' for row in precision.values()) == 37044,
            'EXISTING_MAXIMUM_ENDPOINT_FAMILY_CHANGED')
    physical = {}
    for row in domains:
        if row['family'] == 'count_cdf':
            ident = json.loads(row['endpoint_identity']); key = float(ident['window_ms']),int(ident['field_index'])
            maximum = int(row['physical_maximum_count'])
            require(key not in physical or physical[key] == maximum,'INCONSISTENT_PHYSICAL_COUNT_SUPPORT'); physical[key] = maximum
    values,identities,lookup,physical_lo,physical_hi = prepare_primitives(cdf,responses,labels,physical,reports['linear']['per_clock_response_physical_ranges_hz'])
    require(np.all(values >= physical_lo[:,None]) and np.all(values <= physical_hi[:,None]), 'PRIMITIVE_PHYSICAL_SUPPORT_CONTRADICTION')
    point,spread,maximum,influence = bootstrap_primitives(values,multiplicities,previous,config['endpoint_chunk_size'])
    alpha = float(config['nominal_alpha']); require(0 < alpha < 1,'VALID_NOMINAL_ALPHA_REQUIRED')
    critical = float(np.quantile(maximum,1-alpha,method='higher'))
    old_critical = float(np.quantile(previous,1-alpha,method='higher'))
    primitive = point,spread,physical_lo,physical_hi
    rows,lo,hi = construct_bands(domains,precision,primitive,lookup,cdf['count_support'].tolist(),physical,critical)
    n = len(seeds); planned = int(config['planning_blocks'])
    require(n <= planned <= 128,'FIXED_RESERVATION_PLANNING_LIMIT_REQUIRED')
    projected,_,_ = construct_bands(domains,precision,primitive,lookup,cdf['count_support'].tolist(),physical,critical*np.sqrt(n/planned))
    for row,projection in zip(rows,projected):
        row['planning_lower_root_n_at_'+str(planned)] = projection['lower']
        row['planning_upper_root_n_at_'+str(planned)] = projection['upper']
    primitive_rows = [dict(primitive_identity=json.dumps(ident,sort_keys=True),estimate=float(point[i]),bootstrap_standard_error=float(spread[i]),
                           physical_lower=float(physical_lo[i]),physical_upper=float(physical_hi[i]),lower=float(lo[i]),upper=float(hi[i]),
                           largest_block_squared_influence_fraction='' if np.isnan(influence[i]) else float(influence[i]),
                           status='source_identity' if physical_lo[i] == physical_hi[i] else ('nondegenerate_bootstrap_anchor' if spread[i] > 0 else 'physical_range_fallback'))
                      for i,ident in enumerate(identities)]
    h,pairs = covariance['kernels'],covariance['pair_block_indices']
    cov_rows = []; diagnostic_limit = float(config['single_block_influence_fraction_flag'])
    for first in range(0,len(h),config['endpoint_chunk_size']):
        last = min(first+config['endpoint_chunk_size'],len(h))
        _,jackknife_se,block_influence,residual = covariance_jackknife(h[first:last],pairs,n)
        for j,index in enumerate(covariance['domain_row_indices'][first:last]):
            row = domains[index]; identity = json.loads(row['endpoint_identity'])
            bootstrap_se = float(precision[identity_key(row['family'],identity)]['bootstrap_standard_error'])
            cov_rows.append(dict(domain_row_index=int(index),endpoint_identity=row['endpoint_identity'],
                bootstrap_standard_error=bootstrap_se,jackknife_standard_error=float(jackknife_se[j]),
                jackknife_to_bootstrap_se_ratio=float(jackknife_se[j]/bootstrap_se),
                largest_block_squared_influence_fraction='' if np.isnan(block_influence[j]) else float(block_influence[j]),
                pair_residual_energy_fraction='' if np.isnan(residual[j]) else float(residual[j]),
                influence_flag=bool(block_influence[j] >= diagnostic_limit),first_order_degenerate=bool(jackknife_se[j] == 0)))
    # Verify reuse of covariance resampling conventions on fixed spaced endpoints.
    selected = np.linspace(0,len(h)-1,8,dtype=int)
    pair_weights = 2*multiplicities[:,pairs[:,0]]*multiplicities[:,pairs[:,1]]/(n*(n-1))
    sampled = pair_weights@h[selected].T
    reuse_error = 0.
    for j,index in enumerate(selected):
        expected = cov_rows[index]['bootstrap_standard_error']; actual = sampled[:,j].std(ddof=1)
        error = abs(float(actual)-expected); reuse_error = max(reuse_error,error)
        require(error <= 1e-11*max(1.,abs(expected)), 'COVARIANCE_BOOTSTRAP_CONVENTION_CHANGED')
    summary = {}
    for family in IDENTITY:
        subset = [row for row in rows if row['family'] == family and row['lower'] != '']
        widths = np.asarray([row['upper']-row['lower'] for row in subset])
        future = np.asarray([row['planning_upper_root_n_at_'+str(planned)]-row['planning_lower_root_n_at_'+str(planned)] for row in subset])
        summary[family] = dict(defined_endpoints=len(subset),width_quantiles_0_50_95_100=np.quantile(widths,[0,.5,.95,1]).tolist(),
                              planning_width_quantiles_0_50_95_100=np.quantile(future,[0,.5,.95,1]).tolist())
    degenerate = [row for row in rows if row['definition_status'] == EMPIRICAL]
    require(all(row['lower'] < row['upper'] for row in degenerate), 'EMPIRICAL_DEGENERACY_CANNOT_BECOME_ZERO_UNCERTAINTY')
    output.mkdir(parents=True,exist_ok=False)
    tables = {'endpoint_bands.csv':rows,'primitive_bands.csv':primitive_rows,'covariance_influence.csv':cov_rows}
    for name,table in tables.items():write_rows(output/name,table)
    path = output/'joint_bootstrap_maximum.npz'
    np.savez_compressed(path,prior_maximum=previous,joint_maximum=maximum,source_rng_seeds=np.asarray(seeds))
    outputs = {name:dict(bytes=(output/name).stat().st_size,sha256=sha256_file(output/name)) for name in [*tables,path.name]}
    report = dict(status='COMPLETE_CALIBRATION_UNCERTAINTY_DIAGNOSTIC',settings=config,settings_sha256=sha256_file(config_path),
        domain_rows=len(rows),defined_endpoints=sum(row['lower'] != '' for row in rows),undefined_endpoints=sum(row['lower'] == '' for row in rows),
        channels=dict(Counter(row['channel'] for row in rows)),primitive_endpoints=len(values),
        primitive_status_counts=dict(Counter(row['status'] for row in primitive_rows)),
        empirical_degenerate_endpoints_with_nonzero_width=len(degenerate),joint_critical_value=critical,prior_critical_value=old_critical,
        prior_maximum_reused_without_recomputing_all_covariances=True,prior_bootstrap_sha256=comparison['outputs']['calibration_bootstrap.npz']['sha256'],
        covariance_reuse_probe_endpoints=len(selected),maximum_covariance_bootstrap_se_roundtrip_error=reuse_error,
        original_independent_blocks=n,bootstrap_replicates=len(multiplicities),covariance_overlapping_pairs=len(pairs),
        primitive_single_block_influence_flags=int(np.count_nonzero(influence >= diagnostic_limit)),
        covariance_single_block_influence_flags=sum(row['influence_flag'] for row in cov_rows),
        covariance_first_order_degenerate=sum(row['first_order_degenerate'] for row in cov_rows),
        covariance_jackknife_to_bootstrap_se_ratio_quantiles=np.quantile([row['jackknife_to_bootstrap_se_ratio'] for row in cov_rows],[0,.5,.95,1]).tolist(),
        family_widths=summary,outputs=outputs,python_version=platform.python_version(),numpy_version=np.__version__,
        inference='approximate_joint_whole_block_bootstrap_with_primitive_outer_bounds',nominal_coverage=1-alpha,
        coverage_qualified=False,finite_sample_distribution_free_guarantee=False,
        assumptions='Independent identically distributed whole blocks and adequate joint bootstrap approximation for every selected nondegenerate contrast/primitive. Positive empirical spread and influence diagnostics do not prove regularity or coverage.',
        routing='Calibration-selected candidate channels only. Freeze routing and fallback rules before confirmation; use fresh confirmation multiplicities and one combined critical value.',
        planning_projection='Root-n width projection with calibration centers and critical value held fixed; not an observed future interval, power guarantee or selected confirmation sample size.',
        observed_saturation_is_source_identity=False,empirical_degeneracy_is_zero_population_uncertainty=False,
        influences_are_exclusions=False,numerical_margins_frozen=False,confirmation_replication_selected=False,
        method_adopted_for_confirmation=False,confirmation_execution_released=False,scientific_fidelity_qualified=False,
        new_source_simulations=0,raw_spike_data_reread=False,production_episodes=0,implementation_sha256=sha256_file(Path(__file__)))
    freeze_json(output/'report.json',report)
    return report


def refine_tail_bounds(config_path, project_root, output, config):
    """Change only bound propagation; reuse completed bootstrap and influences."""
    source = config['reuse_diagnostic']; root = project_root/source['path']
    prior = bound_report(root,source['report_sha256'])
    require(prior['status'] == 'COMPLETE_CALIBRATION_UNCERTAINTY_DIAGNOSTIC'
            and all(prior['settings'][key] == config[key] for key in ('inputs','cohorts','joint_rng_seeds')),
            'UNCHANGED_COMPLETED_PRIMITIVE_DIAGNOSTIC_REQUIRED')
    prior_tables = {}
    for name in ('endpoint_bands.csv','primitive_bands.csv','covariance_influence.csv'):
        path = root/name
        require(sha256_file(path) == prior['outputs'][name]['sha256'],'PRIOR_DIAGNOSTIC_TABLE_CHANGED:'+name)
        if name != 'covariance_influence.csv':
            prior_tables[name] = read_rows(path)
    maxima = load_npz(root/'joint_bootstrap_maximum.npz',prior)
    require(np.array_equal(maxima['source_rng_seeds'],config['joint_rng_seeds']), 'REUSED_BOOTSTRAP_BLOCK_ORDER_CHANGED')
    domain_root = project_root/config['inputs']['domain']['path']
    domain = bound_report(domain_root,config['inputs']['domain']['report_sha256'])
    path = domain_root/'endpoint_domains.csv'
    require(sha256_file(path) == domain['outputs'][path.name]['sha256'],'SOURCE_DOMAIN_CHANGED')
    domains = read_rows(path)
    require(len(domains) == prior['domain_rows'] == len(prior_tables['endpoint_bands.csv']),'FULL_PRIOR_DOMAIN_REQUIRED')
    precision,physical = {},{}
    for index,(row,old) in enumerate(zip(domains,prior_tables['endpoint_bands.csv'])):
        require(int(old['domain_row_index']) == index and all(row[key] == old[key]
                for key in ('family','endpoint_identity','definition_status','estimate')), 'PRIOR_ENDPOINT_IDENTITY_CHANGED')
        ident = json.loads(row['endpoint_identity'])
        if old['direct_bootstrap_standard_error'] != '':
            precision[identity_key(row['family'],ident)] = dict(status='ESTIMATED_EMPIRICAL_SPREAD',
                estimate=old['estimate'],bootstrap_standard_error=old['direct_bootstrap_standard_error'])
        if row['family'] == 'count_cdf':
            key = float(ident['window_ms']),int(ident['field_index']); maximum = int(row['physical_maximum_count'])
            require(key not in physical or physical[key] == maximum,'FROZEN_PHYSICAL_SUPPORT_REQUIRED'); physical[key] = maximum
    primitive_rows = prior_tables['primitive_bands.csv']; lookup = {}; support = set()
    for i,row in enumerate(primitive_rows):
        ident = json.loads(row['primitive_identity'])
        if ident['family'] == 'primitive_cdf':
            threshold = int(ident['count_threshold']); support.add(threshold)
            key = ('cdf',ident['cohort'],ident['condition'],ident['phase'],float(ident['window_ms']),int(ident['field_index']),threshold)
        else:
            require(ident['family'] == 'primitive_response','KNOWN_PRIMITIVE_FAMILY_REQUIRED')
            key = ('response',ident['cohort'],ident['condition'],ident['phase'],ident['metric'])
        require(key not in lookup,'UNIQUE_PRIOR_PRIMITIVE_REQUIRED'); lookup[key] = i
    support = sorted(support)
    primitive = tuple(np.asarray([float(row[key]) for row in primitive_rows])
                      for key in ('estimate','bootstrap_standard_error','physical_lower','physical_upper'))
    point,spread,physical_lo,physical_hi = primitive
    selected = (spread == 0) & (physical_lo < physical_hi)
    require(all((row['status'] == 'physical_range_fallback') == bool(selected[i]) for i,row in enumerate(primitive_rows)),
            'CALIBRATION_CONSTANT_ROUTE_CHANGED')
    alpha = float(config['nominal_alpha']); tail_alpha = float(config['joint_tail_alpha']); bootstrap_alpha = alpha-tail_alpha
    require(0 < tail_alpha < alpha < 1,'POSITIVE_BOOTSTRAP_AND_TAIL_ERROR_BUDGETS_REQUIRED')
    critical = float(np.quantile(maxima['joint_maximum'],1-bootstrap_alpha,method='higher'))
    n,planned = len(config['joint_rng_seeds']),int(config['planning_blocks'])
    require(n <= planned <= 128,'UNCHANGED_CONFIRMATION_RESERVATION_REQUIRED')
    # Zero departures follow from calibration selection, not independent validation.
    tail = binomial_probability_upper(0,n,tail_alpha)
    future_tail = binomial_probability_upper(0,planned,tail_alpha)
    rows,lo,hi = construct_bands(domains,precision,primitive,lookup,support,physical,critical,tail)
    projected,_,_ = construct_bands(domains,precision,primitive,lookup,support,physical,critical*np.sqrt(n/planned),future_tail)
    summary,maximum_endpoints = {},{}
    for row,projection in zip(rows,projected):
        row['planning_lower_root_n_and_zero_tail_at_'+str(planned)] = projection['lower']
        row['planning_upper_root_n_and_zero_tail_at_'+str(planned)] = projection['upper']
    for family in IDENTITY:
        subset = [row for row in rows if row['family'] == family and row['lower'] != '']
        widths = np.asarray([row['upper']-row['lower'] for row in subset])
        future = np.asarray([row['planning_upper_root_n_and_zero_tail_at_'+str(planned)]-row['planning_lower_root_n_and_zero_tail_at_'+str(planned)] for row in subset])
        summary[family] = dict(defined_endpoints=len(subset),width_quantiles_0_50_95_100=np.quantile(widths,[0,.5,.95,1]).tolist(),
                              planning_width_quantiles_0_50_95_100=np.quantile(future,[0,.5,.95,1]).tolist())
        maximum_endpoints[family] = sorted(subset,key=lambda row:max(abs(row['lower']),abs(row['upper'])),reverse=True)[:3]
    for i,row in enumerate(primitive_rows):
        row['lower'],row['upper'] = float(lo[i]),float(hi[i])
        row['calibration_constant_tail_candidate'] = bool(selected[i])
        row['joint_tail_probability_upper'] = tail if selected[i] else ''
        if selected[i]:row['status'] = 'joint_event_tail_bound_candidate'
    degenerate = [row for row in rows if row['definition_status'] == EMPIRICAL]
    require(all(row['lower'] < row['upper'] for row in degenerate),'NONZERO_EMPIRICAL_DEGENERACY_UNCERTAINTY_REQUIRED')
    output.mkdir(parents=True,exist_ok=False)
    write_rows(output/'endpoint_bands.csv',rows);write_rows(output/'primitive_bands.csv',primitive_rows)
    for name in ('covariance_influence.csv','joint_bootstrap_maximum.npz'):
        shutil.copy2(root/name,output/name)
        require(sha256_file(output/name) == prior['outputs'][name]['sha256'],'REUSED_OUTPUT_COPY_CHANGED')
    outputs = {name:dict(bytes=(output/name).stat().st_size,sha256=sha256_file(output/name)) for name in prior['outputs']}
    report = dict(prior)
    report.update(status='COMPLETE_CALIBRATION_JOINT_TAIL_UNCERTAINTY_DIAGNOSTIC',settings=config,
        settings_sha256=sha256_file(config_path),implementation_sha256=sha256_file(Path(__file__)),outputs=outputs,
        joint_critical_value=critical,prior_critical_value=prior['joint_critical_value'],bootstrap_alpha=bootstrap_alpha,joint_tail_alpha=tail_alpha,
        joint_tail_candidate_primitives=int(selected.sum()),joint_tail_calibration_departures_by_selection=0,
        joint_tail_probability_upper_calibration=tail,joint_tail_probability_upper_zero_departure_planning=future_tail,
        joint_tail_is_one_event_per_independent_block=True,joint_tail_primitives_are_independent_replicates=False,
        joint_tail_proof='For the fixed selected primitive set, A is any departure from its frozen constants. E[X] lies in [c-p(c-l),c+p(u-c)] simultaneously for every selected primitive if p=P(A) is bounded. A fixed-n one-sided binomial upper bound controls this one joint event.',
        joint_tail_requires_route_frozen_before_confirmation=True,
        calibration_tail_bound_has_confirmatory_coverage=False,
        primitive_status_counts=dict(Counter(row['status'] for row in primitive_rows)),family_widths=summary,largest_absolute_endpoint_bands=maximum_endpoints,
        bootstrap_draws_recomputed=False,covariance_influence_recomputed=False,reused_diagnostic_report_sha256=source['report_sha256'],
        inference='approximate_joint_bootstrap_plus_fixed_joint_departure_event',
        interpretation='Calibration-selected approximate bootstrap plus one joint tail event; no independent coverage evidence, equivalence decision or confirmation release.',
        planning_projection='Nondegenerate widths use root-n scaling of calibration centers/critical value; tail component assumes zero departures at128. This is a planning scenario, not future evidence, power guarantee or selected n.',
        python_version=platform.python_version(),numpy_version=np.__version__)
    freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=run(args.config,args.project_root,args.output)
    print(json.dumps({key:r[key] for key in ('status','defined_endpoints','joint_critical_value','empirical_degenerate_endpoints_with_nonzero_width')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

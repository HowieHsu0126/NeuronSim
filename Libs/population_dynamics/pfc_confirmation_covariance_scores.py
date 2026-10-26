"""Recover directed PFC covariance pair kernels from existing field readouts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_delay_analysis import PHASES, read
from .pfc_scale_analysis import _source_manifest, _source_bound_file
from .provenance import freeze_json, sha256_file
from .source_variant_analysis import time_inner_product_gram


def round_robin_pairings(blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Partition all unordered pairs into deterministic disjoint matchings."""
    require(type(blocks) is int and blocks >= 2, 'AT_LEAST_TWO_BLOCKS_REQUIRED')
    pairs = list(combinations(range(blocks), 2))
    lookup = {pair: index for index, pair in enumerate(pairs)}
    vertices = list(range(blocks))
    if blocks % 2:
        vertices.append(None)
    matchings = []
    for _ in range(len(vertices)-1):
        matching = []
        for k in range(len(vertices)//2):
            a, b = vertices[k], vertices[-1-k]
            if a is not None and b is not None:
                matching.append(lookup[tuple(sorted((a,b)))])
        matchings.append(matching)
        vertices = [vertices[0], vertices[-1], *vertices[1:-1]]
    return np.asarray(pairs,dtype=np.int64), np.asarray(matchings,dtype=np.int64)


def lag_pair_kernels(rates: np.ndarray, lag: int, pairs: np.ndarray) -> np.ndarray | None:
    """H_abij=mean_time((R_ai-R_bi)*(R_aj_lag-R_bj_lag))/2.

    Shared across-block centering cancels in these differences. For positive
    lags, select a directed quadrant only after making the full augmented Gram.
    """
    x = np.asarray(rates,dtype=np.float64)
    require(x.ndim == 3 and len(x) >= 2 and x.shape[1] > 0 and x.shape[2] > 0
            and np.isfinite(x).all() and type(lag) is int and lag >= 0,
            'FINITE_BLOCK_TIME_FIELD_RATES_AND_NONNEGATIVE_LAG_REQUIRED')
    if lag >= x.shape[1]:
        return None
    require(pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype.kind in 'iu'
            and np.all((pairs >= 0) & (pairs < len(x))) and np.all(pairs[:,0] < pairs[:,1]),
            'VALID_UNORDERED_BLOCK_PAIRS_REQUIRED')
    fields = x.shape[2]
    augmented = x if lag == 0 else np.concatenate((x[:,:-lag],x[:,lag:]),axis=2)
    gram = time_inner_product_gram(augmented)
    if lag:
        gram = gram[:,:,:fields,fields:]
    a,b = pairs.T
    return (gram[a,a]+gram[b,b]-gram[a,b]-gram[b,a])/2


def difference_bounds(left: np.ndarray, right: np.ndarray, lag: int):
    """Physical kernel bounds, not observed extrema or scientific margins."""
    a,b = np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    require(a.ndim == b.ndim == 1 and a.shape == b.shape and len(a) > 0
            and np.isfinite(a).all() and np.isfinite(b).all() and np.all(a > 0) and np.all(b > 0)
            and type(lag) is int and lag >= 0, 'POSITIVE_PHYSICAL_RATE_BOUNDS_REQUIRED')
    radius = (np.outer(a,a)+np.outer(b,b))/2
    lower,upper = -radius,radius.copy()
    if lag == 0:
        indices = np.arange(len(a))
        lower[indices,indices] = -b*b/2
        upper[indices,indices] = a*a/2
    return lower,upper


def field_cohort(root: Path, expected: dict, seeds: list[int], bin_ms: float) -> dict:
    require(sha256_file(root/'delivery.json') == expected['delivery_sha256'], 'FIELD_SOURCE_DELIVERY_CHANGED')
    manifest = _source_manifest(read(root/'delivery.json'))
    report = read(_source_bound_file(root,manifest,'analysis/report.json'))
    config_path = _source_bound_file(root,manifest,'config.yaml')
    config = yaml.safe_load(config_path.read_text())
    require(sha256_file(config_path) == report['config_sha256']
            and report['status'] == 'COMPLETE_SOURCE_FLUCTUATION_CALIBRATION'
            and report['completed_source_cases'] == report['registered_source_cases'] == 3*len(seeds)
            and report['joint_rng_blocks'] == seeds and report['phases_ms'] == PHASES
            and report['dt_ms'] == config['dt_ms'] == expected['dt_ms']
            and config['readout_bin_ms'] == report['bin_ms'] == bin_ms
            and report['coupled_core_units'] == expected['coupled_units'] == 100000,
            'COMPLETE_FROZEN_FIELD_SOURCE_REQUIRED')
    path = _source_bound_file(root,manifest,'analysis/field_fluctuations.npz')
    require(sha256_file(path) == report['outputs'][path.name]['sha256'], 'FIELD_TRAJECTORY_BINDING_CHANGED')
    with np.load(path,allow_pickle=False) as stored:
        sizes = stored['field_unit_counts']
        require(sizes.dtype.kind in 'iu' and np.array_equal(sizes,expected['field_unit_counts'])
                and np.array_equal(stored['source_rng_seeds'],seeds), 'SOURCE_FIELD_OR_BLOCK_AXIS_CHANGED')
        rates = {}
        for condition in ('omission','cue','strong_cue'):
            counts = stored[condition+'_field_counts']
            require(counts.shape == (len(seeds),1000,len(sizes)) and counts.dtype.kind in 'iu'
                    and np.all(counts >= 0), 'COMPLETE_INTEGER_FIELD_TRAJECTORY_REQUIRED')
            rates[condition] = counts.astype(float)/(sizes*(bin_ms/1000))
    return dict(rates=rates,sizes=sizes,config=config,source_field_sha256=sha256_file(path))


def recover_covariances(domains, cohorts, labels, seeds, bin_ms, lags):
    fields = len(cohorts[labels[0]]['sizes'])
    physical = {}
    for row in domains:
        if row['family'] != 'count_cdf':
            continue
        identity = json.loads(row['endpoint_identity'])
        if float(identity['window_ms']) != bin_ms:
            continue
        field,count = int(identity['field_index']),int(row['physical_maximum_count'])
        require(field not in physical or physical[field] == count, 'INCONSISTENT_SOURCE_BIN_COUNT_BOUND')
        physical[field] = count
    require(set(physical) == set(range(fields)), 'PHYSICAL_BOUNDS_FOR_EVERY_FIELD_REQUIRED')
    rate_bounds = np.asarray([physical[i]/(bin_ms/1000) for i in range(fields)])
    for cohort in cohorts.values():
        for value in cohort['rates'].values():
            require(np.all((value >= 0) & (value <= rate_bounds)), 'SOURCE_FIELD_RATE_EXCEEDS_REFRACTORY_BOUND')
    pairs,matchings = round_robin_pairings(len(seeds))
    clock_pairs = set(zip(labels[:-1],labels[1:]))
    groups = defaultdict(list)
    valid,undefined = [],[]
    seen = set()
    for row_index,row in enumerate(domains):
        if row['family'] != 'field_lag_covariance':
            continue
        ident = json.loads(row['endpoint_identity'])
        left,right = ident['left_cohort'],ident['right_cohort']
        condition,phase,lag_ms = ident['condition'],ident['phase'],float(ident['lag_ms'])
        i,j = int(ident['leading_field']),int(ident['trailing_field'])
        key = (left,right,condition,phase,lag_ms,i,j)
        require(key not in seen and (left,right) in clock_pairs and lag_ms in lags
                and 0 <= i < fields and 0 <= j < fields, 'FROZEN_UNIQUE_COVARIANCE_ENDPOINT_REQUIRED')
        seen.add(key)
        start,stop = PHASES[phase]
        if lag_ms >= stop-start:
            require(row['definition_status'] == 'UNDEFINED_NO_WITHIN_PHASE_PAIRS' and row['estimate'] == '',
                    'UNDEFINED_PHASE_LAG_MUST_REMAIN_UNESTIMATED')
            undefined.append(row_index)
            continue
        slot = len(valid);valid.append(row_index)
        groups[(condition,phase,lag_ms)].append((slot,left,right,i,j,float(row['estimate'])))
    expected = len(clock_pairs)*3*len(PHASES)*len(lags)*fields*fields
    require(len(seen) == expected, 'FULL_COVARIANCE_ENDPOINT_FAMILY_REQUIRED')
    matrix = np.empty((len(valid),len(pairs)),dtype=np.float64)
    lower = np.empty(len(valid));upper = np.empty(len(valid))
    max_error,max_roundoff = 0.,0.
    for (condition,phase,lag_ms),rows in groups.items():
        start,stop = PHASES[phase]
        lag = round(lag_ms/bin_ms)
        require(lag*bin_ms == lag_ms, 'INTEGER_READOUT_LAG_REQUIRED')
        first,last = round(start/bin_ms),round(stop/bin_ms)
        kernels = {label:lag_pair_kernels(cohorts[label]['rates'][condition][:,first:last],lag,pairs) for label in labels}
        bounds_lo,bounds_hi = difference_bounds(rate_bounds,rate_bounds,lag)
        for left,right in clock_pairs:
            subset = [row for row in rows if row[1:3] == (left,right)]
            slots=np.asarray([row[0] for row in subset]);ii=np.asarray([row[3] for row in subset]);jj=np.asarray([row[4] for row in subset])
            values = (kernels[left]-kernels[right])[:,ii,jj].T
            lo,hi = bounds_lo[ii,jj],bounds_hi[ii,jj]
            require(np.isfinite(values).all() and np.all(values >= lo[:,None]) and np.all(values <= hi[:,None]),
                    'PAIRED_COVARIANCE_KERNEL_OUTSIDE_PHYSICAL_BOUND')
            point=values.mean(axis=1);reference=np.asarray([row[5] for row in subset])
            # Arithmetic roundtrip allowance only; scales with operations and Gram-derived values.
            roundoff=128*np.finfo(float).eps*len(pairs)*np.maximum(1.,np.max(np.abs(kernels[left][:,ii,jj]),axis=0)+np.max(np.abs(kernels[right][:,ii,jj]),axis=0))
            error=np.abs(point-reference)
            require(np.all(error <= roundoff), 'UNBIASED_COVARIANCE_POINT_ROUNDTRIP_FAILED:'+str((condition,phase,lag_ms,left,right)))
            matrix[slots]=values;lower[slots]=lo;upper[slots]=hi
            max_error=max(max_error,float(error.max()));max_roundoff=max(max_roundoff,float(roundoff.max()))
    return dict(kernels=matrix,domain_row_indices=np.asarray(valid,dtype=np.int64),
        undefined_domain_row_indices=np.asarray(undefined,dtype=np.int64),
        pair_block_indices=pairs,matching_pair_indices=matchings,
        source_rng_seeds=np.asarray(seeds,dtype=np.int64),physical_lower_bound=lower,physical_upper_bound=upper,
        point_estimates=matrix.mean(axis=1)),dict(
        defined_covariance_endpoints=len(valid),undefined_covariance_endpoints=len(undefined),
        original_independent_blocks=len(seeds),overlapping_unordered_pairs=len(pairs),
        disjoint_matchings=len(matchings),independent_pairs_within_each_matching=matchings.shape[1],
        covariance_orientation='leading_field_at_t_to_trailing_field_at_t_plus_lag',units='Hz^2',
        field_rate_upper_bounds_hz=rate_bounds.tolist(),maximum_archived_point_roundtrip_error=max_error,
        maximum_arithmetic_roundtrip_allowance=max_roundoff,
        roundtrip_allowance_is_scientific_margin=False,
        overlapping_pairs_or_matchings_are_independent_replicates=False)


def run(config_path,project_root,output):
    config=yaml.safe_load(config_path.read_text())
    require(config['schema']=='pfc_covariance_pair_scores_v1' and config['scope']=='development_calibration'
            and config['independent_unit']=='joint_initial_background_rng_block'
            and config['confirmation_execution_released'] is False, 'CALIBRATION_ONLY_COVARIANCE_RECOVERY_REQUIRED')
    comparison=bound_report(project_root/config['comparison_archive'],config['comparison_report_sha256'])
    domain_root=project_root/config['domain_archive'];domain=bound_report(domain_root,config['domain_report_sha256'])
    require(domain['status']=='COMPLETE_CALIBRATION_BLOCK_CDF_SCORES', 'COMPLETE_PHYSICAL_DOMAIN_WITH_BLOCK_SCORES_REQUIRED')
    table=domain_root/'endpoint_domains.csv'
    require(sha256_file(table)==domain['outputs'][table.name]['sha256'], 'PHYSICAL_DOMAIN_TABLE_CHANGED')
    domains=read_rows(table);seeds=config['joint_rng_seeds'];labels=list(config['field_archives'])
    require(seeds==sorted(set(seeds)) and len(seeds)>=2
            and all(comparison['inputs'][label]['source_rng_blocks']==seeds for label in labels), 'FROZEN_CALIBRATION_BLOCKS_REQUIRED')
    bin_ms=comparison['settings']['bin_ms'];lags=comparison['settings']['lag_ms']
    cohorts={label:field_cohort(project_root/path,comparison['inputs'][label],seeds,bin_ms) for label,path in config['field_archives'].items()}
    payload,checks=recover_covariances(domains,cohorts,labels,seeds,bin_ms,lags)
    output.mkdir(parents=True,exist_ok=False);artifact=output/'paired_block_covariance_kernels.npz';np.savez_compressed(artifact,**payload)
    report=dict(status='COMPLETE_CALIBRATION_COVARIANCE_PAIR_SCORES',**checks,
        settings=config,settings_sha256=sha256_file(config_path),python_version=platform.python_version(),numpy_version=np.__version__,
        domain_rows=len(domains),domain_table_sha256=sha256_file(table),
        domain_row_indices_semantics='zero-based row of bound source endpoint_domains.csv; header excluded',
        source_field_sha256={label:cohort['source_field_sha256'] for label,cohort in cohorts.items()},
        pair_kernel='(D_aa+D_bb-D_ab-D_ba)/2; block indices swapped only; full directed lag retained',
        interpretation='Unbiased source-clock covariance contrast and physical score ranges; no confidence intervals or qualification. Same seed labels are not a common physical noise path.',
        new_source_simulations=0,raw_spike_data_reread=False,confidence_intervals=False,numerical_margins_frozen=False,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,production_episodes=0,
        outputs={artifact.name:dict(bytes=artifact.stat().st_size,sha256=sha256_file(artifact))},
        implementation_sha256=sha256_file(Path(__file__)),gram_implementation_sha256=sha256_file(Path(__file__).with_name('source_variant_analysis.py')))
    freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=run(args.config,args.project_root,args.output)
    print(json.dumps({key:r[key] for key in ('status','defined_covariance_endpoints','undefined_covariance_endpoints','original_independent_blocks','overlapping_unordered_pairs')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

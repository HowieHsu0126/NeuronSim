"""Full-count driver dependence diagnostics, not a source qualification decision.

StreamingCountTrace accepts complete target-axis chunks and retains only complete
population sums and a fixed target sample. It never simulates or resamples a driver.
HAC intervals are asymptotic and require stationary, weakly dependent observations;
reported bandwidth and record-span checks do not prove those assumptions. No
non-rejection or interval is converted into equivalence without a separately frozen
contamination budget and a production-matched protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np

SCHEMA = 'poisson_count_dependence_v1'


@dataclass(frozen=True)
class CountTrace:
    arrival_ticks: np.ndarray
    sampled_counts: np.ndarray
    aggregate_counts: np.ndarray
    sampled_ids: np.ndarray
    sampled_groups: np.ndarray
    sampled_means: np.ndarray
    group_labels: tuple[str, ...]
    aggregate_means: np.ndarray
    group_target_counts: tuple[int, ...]
    complete_target_count: int


def select_targets(groups: np.ndarray, per_group: int = 8) -> np.ndarray:
    """Structural selection, independent of counts; producer may supply other IDs."""
    if type(per_group) is not int or per_group < 2:
        raise ValueError('at least two structural target samples per group required')
    selected = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if len(indices) <= per_group:
            chosen = indices
        else:
            if per_group < 4:
                positions = np.linspace(0, len(indices)-1, per_group, dtype=int)
            else:
                positions = sorted({0, 1, len(indices)-2, len(indices)-1,
                    *map(int, np.linspace(2, len(indices)-3, per_group-4))})
            chosen = indices[positions]
        selected.extend(map(int, chosen))
    return np.asarray(selected, dtype=np.int64)


class StreamingCountTrace:
    """Reduce full-target count chunks to O(time * (sample + populations)) storage.

    Target order and native arrival ticks are fixed by the producer. This does not
    make sample-target checks equivalent to testing every target pair. Population
    sums include every supplied target; production-scale qualification separately
    requires that these identities/counts match the production source mapping.
    """
    def __init__(self, target_ids, generator_index, lambda_per_target,
                 sampled_indices: Sequence[int] | None = None):
        ids, groups, means = map(np.asarray, (target_ids, generator_index, lambda_per_target))
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids)
                or groups.shape != ids.shape or groups.dtype.kind not in 'iu'
                or np.any(groups < 0) or means.shape != ids.shape
                or not np.isfinite(means).all() or np.any(means < 0)):
            raise ValueError('unique target IDs, group IDs and nonnegative finite means required')
        selected = select_targets(groups) if sampled_indices is None else np.asarray(sampled_indices)
        if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in 'iu'
                or np.any(selected < 0) or np.any(selected >= len(ids))
                or len(np.unique(selected)) != len(selected)):
            raise ValueError('unique in-range fixed sampled indices required')
        self.ids, self.groups, self.means = ids.copy(), groups.copy(), means.astype(float)
        self.selected = selected.astype(np.int64)
        self.labels = np.unique(groups)
        self.indices = [np.flatnonzero(groups == label) for label in self.labels]
        self.expected_tick = None
        self.tick_parts, self.sample_parts, self.aggregate_parts = [], [], []

    def update(self, counts, arrival_ticks) -> None:
        x, ticks = np.asarray(counts), np.asarray(arrival_ticks)
        if (x.ndim != 2 or x.shape[1] != len(self.ids) or x.dtype.kind not in 'iu'
                or np.any(x < 0) or np.any(x > np.iinfo(np.int32).max)
                or ticks.shape != (len(x),) or ticks.dtype.kind not in 'iu'
                or not len(ticks) or np.any(ticks < 0) or np.any(np.diff(ticks.astype(np.int64)) != 1)):
            raise ValueError('complete integer target chunks and consecutive arrival ticks required')
        if self.expected_tick is not None and int(ticks[0]) != self.expected_tick:
            raise ValueError('arrival gap, overlap or reordering')
        if len(self.ids) * int(x.max()) > np.iinfo(np.int64).max:
            raise ValueError('population count sum would overflow int64')
        group_counts = np.column_stack([x[:, index].sum(axis=1, dtype=np.int64) for index in self.indices])
        aggregate = np.column_stack((group_counts, group_counts.sum(axis=1, dtype=np.int64)))
        self.tick_parts.append(ticks.astype(np.int64, copy=True))
        self.sample_parts.append(x[:, self.selected].astype(np.int32, copy=True))
        self.aggregate_parts.append(aggregate)
        self.expected_tick = int(ticks[-1]) + 1

    def finish(self) -> CountTrace:
        if not self.tick_parts:
            raise ValueError('no complete counts supplied')
        group_means = np.asarray([self.means[index].sum() for index in self.indices])
        return CountTrace(np.concatenate(self.tick_parts), np.concatenate(self.sample_parts),
            np.concatenate(self.aggregate_parts), self.ids[self.selected], self.groups[self.selected],
            self.means[self.selected], tuple([f'generator_{int(g)}' for g in self.labels] + ['all_targets']),
            np.r_[group_means, group_means.sum()],
            tuple([len(index) for index in self.indices] + [len(self.ids)]), len(self.ids))


def hac_interval(center: float, influence, bandwidth: int, *, confidence: float = .95,
                 minimum_record_spans: float = 20.) -> dict:
    """Bartlett/Newey-West interval for a moment functional via its influence.

    The confidence interval is approximate, not an exact finite-record bound.
    n/(bandwidth+1) is a record-length diagnostic, not an effective sample size.
    A requested bandwidth is never reduced to manufacture an interval.
    """
    x = np.asarray(influence, dtype=float)
    if (x.ndim != 1 or not np.isfinite(x).all() or not math.isfinite(center)
            or type(bandwidth) is not int or bandwidth < 0
            or not 0 < confidence < 1 or minimum_record_spans <= 0):
        raise ValueError('invalid HAC series or declared settings')
    n = len(x)
    result = dict(method='Bartlett_HAC_delta_normal_approximation', confidence_level=confidence,
        observations=n, bandwidth=bandwidth, record_spans=n/(bandwidth+1),
        minimum_record_spans=minimum_record_spans, interval=None, standard_error=None)
    if n < 2 or bandwidth >= n or result['record_spans'] < minimum_record_spans:
        return {**result, 'status': 'INSUFFICIENT_RECORD_FOR_DECLARED_HAC'}
    x = x-x.mean()
    size = 1 << (2*n-1).bit_length()
    spectrum = np.fft.rfft(x, n=size)
    covariance = np.fft.irfft(spectrum*spectrum.conjugate(), n=size)[:bandwidth+1]/n
    weights = 1-np.arange(1, bandwidth+1)/(bandwidth+1)
    long_run = float(covariance[0]+2*np.dot(weights, covariance[1:]))
    rounding = 64*np.finfo(float).eps*max(1., float(covariance[0]))*(bandwidth+1)
    if long_run < -rounding:
        raise FloatingPointError('negative Bartlett long-run variance beyond rounding')
    standard_error = math.sqrt(max(0., long_run)/n)
    radius = NormalDist().inv_cdf((1+confidence)/2)*standard_error
    return {**result, 'status': 'APPROXIMATE_INTERVAL', 'standard_error': standard_error,
            'interval': [float(center-radius), float(center+radius)]}


def covariance_diagnostic(left, right, left_mean: float, right_mean: float, *, lag: int = 0,
                          bandwidth: int = 500, confidence: float = .95,
                          minimum_record_spans: float = 20.) -> dict:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
            or type(lag) is not int or lag < 0 or left_mean < 0 or right_mean < 0
            or not math.isfinite(left_mean+right_mean)):
        raise ValueError('aligned finite count traces, valid lag and means required')
    if lag:
        x, y = x[:-lag], y[lag:]
    n = len(x)
    if n < 2 or left_mean == 0 or right_mean == 0:
        return dict(status='NOT_ESTIMABLE', observations=n, lag_ticks=lag,
                    covariance=None, covariance_over_poisson_sd=None, uncertainty=None)
    centered_x, centered_y = x-x.mean(), y-y.mean()
    product = centered_x*centered_y
    covariance = float(product.sum()/(n-1))
    scale = math.sqrt(left_mean*right_mean)
    normalized = covariance/scale
    influence = (product-product.mean())*(n/(n-1))/scale
    uncertainty = hac_interval(normalized, influence, bandwidth, confidence=confidence,
                               minimum_record_spans=minimum_record_spans)
    sd_product = math.sqrt(float(np.dot(centered_x, centered_x)*np.dot(centered_y, centered_y)))
    pearson = float(np.dot(centered_x, centered_y)/sd_product) if sd_product else None
    return dict(status='ESTIMATED', observations=n, lag_ticks=lag, covariance=covariance,
        covariance_over_poisson_sd=normalized, pearson_correlation=pearson, uncertainty=uncertainty)


def complete_window_sums(values, ticks, width: int):
    x, ticks = np.asarray(values), np.asarray(ticks)
    if type(width) is not int or width < 1 or not len(ticks):
        raise ValueError('positive native window width and nonempty trace required')
    leading = int((-int(ticks[0])) % width)
    available = max(0, len(ticks)-leading)
    n = available//width
    used = n*width
    result = x[leading:leading+used].reshape(n, width, x.shape[1]).sum(axis=1, dtype=np.int64)
    return result, dict(native_window_ticks=width, windows=n,
        discarded_leading_ticks=min(leading,len(ticks)), discarded_trailing_ticks=available-used,
        first_complete_tick=int(ticks[0])+leading if n else None)


def _variance_diagnostic(values, expected: float, bandwidth: int, confidence: float, spans: float):
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2 or expected == 0:
        return dict(status='NOT_ESTIMABLE', expected_count=expected,
                    observed_mean=float(x.mean()) if n else None, variance_excess_fraction=None,
                    uncertainty=None, zero_rate_events=int(x.sum()) if expected == 0 else None)
    centered = x-x.mean()
    squared = centered**2
    variance = float(squared.sum()/(n-1))
    excess = variance/expected-1
    influence = (squared-squared.mean())*(n/(n-1))/expected
    return dict(status='ESTIMATED', expected_count=expected,
        observed_mean=float(x.mean()), mean_relative_error=float(x.mean()/expected-1),
        sample_variance=variance, variance_excess_fraction=excess,
        uncertainty=hac_interval(excess,influence,bandwidth,confidence=confidence,
                                 minimum_record_spans=spans),
        mean_uncertainty=hac_interval(float(x.mean()/expected-1),centered/expected,bandwidth,
            confidence=confidence,minimum_record_spans=spans))


def _pairs(groups, maximum):
    possible = list(combinations(range(len(groups)), 2))
    if len(possible) <= maximum:
        return possible
    mandatory = []
    first = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        first.append(int(indices[0]))
        if len(indices) > 1:
            mandatory.append((int(indices[0]),int(indices[1])))
    mandatory.extend(tuple(sorted(pair)) for pair in combinations(first,2))
    if len(mandatory) > maximum:
        raise ValueError('pair budget cannot retain generator-stratified sentinel pairs')
    result = set(mandatory)
    for index in np.linspace(0,len(possible)-1,maximum,dtype=int):
        result.add(possible[int(index)])
        if len(result) == maximum:
            break
    return sorted(result)


def analyze_traces(trace: CountTrace, *, window_ticks=(1,50,200,800,5000),
                   temporal_lags=(1,2,5,10,50,100,500), hac_bandwidth_ticks=500,
                   alpha=.05, minimum_record_spans=20., maximum_pairs=256) -> dict:
    windows = tuple(window_ticks); lags = tuple(temporal_lags)
    if (not windows or len(set(windows)) != len(windows) or any(type(w) is not int or w < 1 for w in windows)
            or not lags or len(set(lags)) != len(lags) or any(type(l) is not int or l < 1 for l in lags)
            or type(hac_bandwidth_ticks) is not int or hac_bandwidth_ticks < max(lags)
            or not 0 < alpha < 1 or type(maximum_pairs) is not int or maximum_pairs < 1):
        raise ValueError('distinct positive windows/lags and bandwidth covering declared lags required')
    family = 2*len(windows)*len(trace.group_labels)
    report = dict(schema=SCHEMA, status='DIAGNOSTIC_ONLY', scientific_reference_qualified=False,
        equivalence_decision='NOT_EVALUATED_NO_FROZEN_CONTAMINATION_MARGIN',
        interpretation='HAC intervals are asymptotic; non-rejection is not independence or equivalence.',
        coverage=dict(complete_targets=trace.complete_target_count,
            group_labels=trace.group_labels,group_target_counts=trace.group_target_counts,
            sampled_target_ids=trace.sampled_ids.tolist(),
            native_arrival_ticks=[int(trace.arrival_ticks[0]),int(trace.arrival_ticks[-1])]),
        settings=dict(window_ticks=windows,temporal_lags=lags,hac_bandwidth_ticks=hac_bandwidth_ticks,
            alpha=alpha,minimum_record_spans=minimum_record_spans,maximum_pairs=maximum_pairs,
            primary_mean_variance_interval_family_size=family,
            primary_interval_correction='Bonferroni over all declared groups, windows, means and variances',
            covariance_intervals='pointwise exploratory, no independence decision'),
        aggregate_variance=[],target_pairs=[],cross_generator_pairs=[],temporal=[])
    for width in windows:
        totals, alignment = complete_window_sums(trace.aggregate_counts,trace.arrival_ticks,width)
        bandwidth = math.ceil(hac_bandwidth_ticks/width)
        for index,label in enumerate(trace.group_labels):
            estimate = _variance_diagnostic(totals[:,index],width*trace.aggregate_means[index],bandwidth,
                1-alpha/family,minimum_record_spans)
            report['aggregate_variance'].append(dict(group=label,**alignment,**estimate))
    for left,right in _pairs(trace.sampled_groups,maximum_pairs):
        estimate = covariance_diagnostic(trace.sampled_counts[:,left],trace.sampled_counts[:,right],
            float(trace.sampled_means[left]),float(trace.sampled_means[right]),bandwidth=hac_bandwidth_ticks,
            confidence=1-alpha,minimum_record_spans=minimum_record_spans)
        report['target_pairs'].append(dict(left_target=int(trace.sampled_ids[left]),right_target=int(trace.sampled_ids[right]),
            same_generator=bool(trace.sampled_groups[left]==trace.sampled_groups[right]),**estimate))
    # The all-target aggregate overlaps each group, so it is excluded from zero-covariance pairs.
    for left,right in combinations(range(len(trace.group_labels)-1),2):
        for lag in (0,*lags):
            directions = ((left,right),) if lag == 0 else ((left,right),(right,left))
            for source,target in directions:
                estimate = covariance_diagnostic(trace.aggregate_counts[:,source],trace.aggregate_counts[:,target],
                    float(trace.aggregate_means[source]),float(trace.aggregate_means[target]),lag=lag,
                    bandwidth=hac_bandwidth_ticks,confidence=1-alpha,minimum_record_spans=minimum_record_spans)
                report['cross_generator_pairs'].append(dict(left_group=trace.group_labels[source],
                    right_group=trace.group_labels[target],**estimate))
    series = [(f'target_{int(uid)}',trace.sampled_counts[:,i],trace.sampled_means[i])
              for i,uid in enumerate(trace.sampled_ids)]
    series += [(label,trace.aggregate_counts[:,i],trace.aggregate_means[i])
               for i,label in enumerate(trace.group_labels)]
    for label,values,mean in series:
        for lag in lags:
            estimate = covariance_diagnostic(values,values,float(mean),float(mean),lag=lag,
                bandwidth=hac_bandwidth_ticks,confidence=1-alpha,minimum_record_spans=minimum_record_spans)
            report['temporal'].append(dict(series=label,**estimate))
    return report



def independent_record_variance(records: Sequence[CountTrace], record_ids: Sequence[str], *,
        window_ticks=(1,50,200,800,5000), alpha=.05, minimum_records=20) -> dict:
    """Phase-aligned variance across independent full records, with seed jackknife.

    At each window position, variance is estimated across records, then averaged
    over positions. Entire records are the independent units; time dependence is
    unrestricted within a record. Intervals remain asymptotic across records.
    The estimator does not mistake a repeated time sample for a new realization.
    Record IDs must identify independently initialized random streams, not replays.
    """
    r=len(records)
    if (r<2 or len(record_ids)!=r or len(set(record_ids))!=r or not 0<alpha<1
            or type(minimum_records) is not int or minimum_records<3
            or not window_ticks or len(set(window_ticks))!=len(window_ticks)):
        raise ValueError('independent unique record IDs and valid declared settings required')
    base=records[0]
    for record in records[1:]:
        if (record.group_labels!=base.group_labels or record.group_target_counts!=base.group_target_counts
                or record.complete_target_count!=base.complete_target_count
                or not np.array_equal(record.arrival_ticks,base.arrival_ticks)
                or not np.array_equal(record.aggregate_means,base.aggregate_means)):
            raise ValueError('phase-aligned records with identical source groups and means required')
    family=2*len(window_ticks)*len(base.group_labels)
    confidence=1-alpha/family
    z=NormalDist().inv_cdf((1+confidence)/2)
    result=dict(schema=SCHEMA,method='phase_aligned_independent_record_variance_seed_jackknife',
        independent_records=r,record_ids=list(record_ids),minimum_records=minimum_records,
        confidence_level=confidence,interval_family_size=family,scientific_reference_qualified=False,
        equivalence_decision='NOT_EVALUATED_NO_FROZEN_CONTAMINATION_MARGIN',aggregate_variance=[],
        interpretation='Time bins are not independent units; jackknife intervals are approximate across independent records.')
    for width in window_ticks:
        binned=[complete_window_sums(record.aggregate_counts,record.arrival_ticks,width)[0] for record in records]
        n=len(binned[0])
        for index,label in enumerate(base.group_labels):
            expected=float(width*base.aggregate_means[index])
            row=dict(group=label,native_window_ticks=width,windows_per_record=n,expected_count=expected,
                variance_interval=None,mean_interval=None,standard_error=None)
            if n==0 or expected==0:
                result['aggregate_variance'].append({**row,'status':'NOT_ESTIMABLE'})
                continue
            values=np.stack([value[:,index] for value in binned]).astype(float)
            normalized=(values-expected)/math.sqrt(expected)
            deviation=normalized-normalized.mean(axis=0)
            squares=deviation**2;ss=squares.sum(axis=0)
            estimate=float(np.mean(ss/(r-1))-1)
            per_record_mean=values.mean(axis=1)/expected-1
            mean_error=float(per_record_mean.mean())
            row.update(variance_excess_fraction=estimate,mean_relative_error=mean_error,
                       status='INSUFFICIENT_INDEPENDENT_RECORDS')
            if r>=minimum_records:
                leave_one=np.mean((ss[None,:]-r/(r-1)*squares)/(r-2),axis=1)-1
                se=math.sqrt(float((r-1)/r*np.sum((leave_one-leave_one.mean())**2)))
                mean_se=float(per_record_mean.std(ddof=1)/math.sqrt(r))
                row.update(status='APPROXIMATE_INDEPENDENT_RECORD_INTERVAL',standard_error=se,
                    variance_interval=[estimate-z*se,estimate+z*se],
                    mean_interval=[mean_error-z*mean_se,mean_error+z*mean_se])
            result['aggregate_variance'].append(row)
    return result


def analyze_legacy(directory: Path, *, chunk_rows=2048, **settings) -> dict:
    metadata = json.loads((directory/'acquisition.json').read_text())
    if metadata.get('status') != 'EXECUTED':
        raise ValueError('executed uniquely decoded arrival-count acquisition required')
    support = metadata['config'].get('expected_arrival_support')
    if support is None:
        raise ValueError('explicit arrival support required; onset is not inferred')
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('positive chunk row count required')
    with np.load(directory/'counts.npz',allow_pickle=False) as archive:
        counts, ticks = archive['counts'], archive['arrival_ticks']
        if (counts.ndim != 2 or counts.dtype.kind not in 'iu' or np.any(counts < 0)
                or ticks.ndim != 1 or len(ticks) != len(counts) or ticks.dtype.kind not in 'iu'
                or not len(ticks) or np.any(np.diff(ticks.astype(np.int64)) != 1)):
            raise ValueError('valid integer counts and complete consecutive arrival clock required')
        inside=(ticks>=support['first_tick'])&(ticks<=support['last_tick'])
        if not inside.any():
            raise ValueError('record has no overlap with declared arrival support')
        outside_events = sum(int(v) for v in counts[~inside].flat)
        stream=StreamingCountTrace(archive['target_ids'],archive['generator_index'],archive['lambda_per_target'])
        indices=np.flatnonzero(inside)
        for start in range(0,len(indices),chunk_rows):
            rows=indices[start:start+chunk_rows]
            stream.update(counts[rows],ticks[rows])
    report=analyze_traces(stream.finish(),**settings)
    report.update(source_acquisition=str(directory),config=metadata['config'],
        dt_ms=metadata['config']['dt_ms'],outside_support_event_count=outside_events,
        legacy_scientific_reference_qualified=metadata.get('scientific_reference_qualified',False))
    previous=directory/'analysis.json'
    if previous.exists():
        report['legacy_law_test_outcome']=json.loads(previous.read_text()).get('law_test_outcome')
    if outside_events:
        report['status']='SUPPORT_VIOLATION_PRESERVED'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--window-ticks',type=int,nargs='+',default=[1,50,200,800,5000])
    parser.add_argument('--temporal-lags',type=int,nargs='+',default=[1,2,5,10,50,100,500])
    parser.add_argument('--hac-bandwidth-ticks',type=int,default=500)
    parser.add_argument('--minimum-record-spans',type=float,default=20.)
    parser.add_argument('--maximum-pairs',type=int,default=256)
    args=parser.parse_args()
    report=analyze_legacy(args.input,window_ticks=tuple(args.window_ticks),temporal_lags=tuple(args.temporal_lags),
        hac_bandwidth_ticks=args.hac_bandwidth_ticks,minimum_record_spans=args.minimum_record_spans,
        maximum_pairs=args.maximum_pairs)
    print(json.dumps(report,allow_nan=False))
    return int(report['status']=='SUPPORT_VIOLATION_PRESERVED')


if __name__=='__main__':
    raise SystemExit(main())

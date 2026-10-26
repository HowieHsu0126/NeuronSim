"""Empirical PFC clock precision on unchanged frozen endpoint identities.

Scores retain original independent blocks. Covariance scores are delete-one
U-statistic pseudovalues, not independent pair kernels. No intervals, population
zero claims, numerical qualification or confirmation decisions are produced.
"""
from __future__ import annotations

from collections import defaultdict
import json

import numpy as np

from .pfc_confirmation_covariance_scores import lag_pair_kernels, round_robin_pairings
from .pfc_confirmation_linear_scores import linear_block_score
from .pfc_confirmation_uncertainty import (
    IDENTITY, STRUCTURAL, UNDEFINED, EMPIRICAL, covariance_jackknife, identity_key,
)
from .pfc_delay_analysis import PHASES, require
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS

LABELS = ('numpy_dt_0p05', 'numpy_dt_0p025', 'numpy_dt_0p0125')
WINDOWS_MS = (5, 50, 100)
BIN_MS = 5.


def _integer(value, name, maximum=None):
    number = float(value)
    require(np.isfinite(number) and number >= 0 and number.is_integer()
            and (maximum is None or number < maximum), 'INVALID_' + name)
    return int(number)


def _inputs(fields, responses, cdfs, support, seeds):
    block_ids = np.asarray(seeds)
    require(block_ids.ndim == 1 and block_ids.dtype.kind in 'iu' and len(block_ids) >= 3
            and np.all(np.diff(block_ids) > 0), 'ORDERED_DISTINCT_ORIGINAL_BLOCKS_REQUIRED')
    support = np.asarray(support)
    require(support.ndim == 1 and support.dtype.kind in 'iu' and len(support) > 0
            and support[0] == 0 and np.all(np.diff(support) > 0),
            'ORDERED_INTEGER_OBSERVED_SUPPORT_FROM_ZERO_REQUIRED')
    require(all(set(data) == set(LABELS) for data in (fields, responses, cdfs)),
            'ALL_THREE_CANONICAL_CLOCKS_REQUIRED')
    blocks = len(block_ids)
    means = {}
    for label in LABELS:
        require(set(fields[label]) == set(CONDITIONS), 'ALL_SOURCE_CONDITIONS_REQUIRED')
        means[label] = {}
        for condition in CONDITIONS:
            values = np.asarray(fields[label][condition])
            require(values.shape == (blocks, 1000, 17) and np.isfinite(values).all()
                    and np.all(values >= 0), 'COMPLETE_FINITE_FIELD_RATES_REQUIRED')
            for phase, (start, stop) in PHASES.items():
                means[label][condition, phase] = values[:, round(start/BIN_MS):round(stop/BIN_MS)].mean(1)
        values = np.asarray(responses[label])
        require(values.shape == (3, blocks, 5, 8) and np.isfinite(values).all(),
                'COMPLETE_FINITE_SOURCE_RESPONSES_REQUIRED')
        values = np.asarray(cdfs[label])
        require(values.shape == (3, blocks, 5, 3, 17, len(support))
                and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
                and np.all(np.diff(values, axis=-1) >= 0) and np.all(values[..., -1] == 1),
                'COMPLETE_MONOTONE_OBSERVED_CDF_MASS_REQUIRED')
    return means, support, blocks


def _empirical_summary(row, score, point, kind, influence=None, residual=None):
    constant = bool(np.all(score == score[0]))
    variance = 0. if constant else float(score.var(ddof=1))
    if influence is None:
        deviations = score - score.mean()
        energy = float(np.dot(deviations, deviations))
        influence = float(np.max(deviations*deviations)/energy) if energy > 0 else None
    row.update(
        signed_estimate=float(point), score_kind=kind, sample_variance=variance,
        sample_variance_kind=('jackknife_pseudovalue_variance' if kind == 'covariance_jackknife_pseudovalue'
                              else 'paired_block_difference_variance'),
        standard_error=float(np.sqrt(variance/len(score))),
        empirical_status=(STRUCTURAL if row['definition_status'] == STRUCTURAL else
                          EMPIRICAL if constant else 'ESTIMATED_EMPIRICAL_SPREAD'),
        empirical_constant=constant, nonzero_score_blocks=int(np.count_nonzero(score)),
        first_order_degenerate=constant if kind == 'covariance_jackknife_pseudovalue' else '',
        maximum_block_influence_fraction=(float(influence) if influence is not None and np.isfinite(influence) else ''),
        pair_residual_energy_fraction=(float(residual) if residual is not None and np.isfinite(residual) else ''),
    )


def summarize_endpoints(margins, fields, responses, cdfs, support, seeds):
    """Return frozen-order descriptive rows and (endpoint, original-block) scores.

    Frozen empirical labels describe the archived domain and remain unchanged;
    current empirical status is reported separately. At or beyond the observed
    maximum, a CDF equals one empirically; only a verified source physical bound
    establishes a structural support identity.
    Only a frozen, truly undefined within-phase lag receives NaN score entries.
    """
    responses = {label: np.asarray(values, dtype=np.float64) for label, values in responses.items()}
    cdfs = {label: np.asarray(values, dtype=np.float64) for label, values in cdfs.items()}
    means, support, blocks = _inputs(fields, responses, cdfs, support, seeds)
    rows = [dict(row) for row in margins]
    scores = np.full((len(rows), blocks), np.nan)
    pairs, _ = round_robin_pairings(blocks)
    groups = defaultdict(list)
    seen_keys, seen_indices = set(), set()
    clock_pairs = set(zip(LABELS[:-1], LABELS[1:]))
    for slot, row in enumerate(rows):
        family = row['family']
        require(family in IDENTITY, 'KNOWN_FROZEN_ENDPOINT_FAMILY_REQUIRED')
        ident = json.loads(row['endpoint_identity'])
        require(set(ident) == set(IDENTITY[family]), 'FROZEN_ENDPOINT_IDENTITY_FIELDS_REQUIRED')
        index = _integer(row['domain_row_index'], 'DOMAIN_ROW_INDEX')
        key = identity_key(family, ident)
        require(index not in seen_indices and key not in seen_keys, 'UNIQUE_FROZEN_ENDPOINT_REQUIRED')
        seen_indices.add(index); seen_keys.add(key)
        left, right, phase = ident['left_cohort'], ident['right_cohort'], ident['phase']
        require((left, right) in clock_pairs and phase in PHASES, 'CANONICAL_CLOCK_PAIR_AND_PHASE_REQUIRED')
        pi = list(PHASES).index(phase)
        if family == 'paired_response':
            require(ident['contrast'] in {item[0] for item in CONTRASTS}, 'KNOWN_PAIRED_CONTRAST_REQUIRED')
        else:
            require(ident['condition'] in CONDITIONS, 'KNOWN_SOURCE_CONDITION_REQUIRED')
        expected_units = 'probability' if family == 'count_cdf' else 'Hz^2' if family == 'field_lag_covariance' else 'Hz'
        require(row['units'] == expected_units, 'FROZEN_ENDPOINT_UNITS_CHANGED')
        if family in ('source_response', 'paired_response'):
            require(ident['metric'] in METRICS, 'KNOWN_SOURCE_METRIC_REQUIRED')
        if family in ('field_mean', 'count_cdf'):
            _integer(ident['field_index'], 'FIELD_INDEX', 17)
        status = row['definition_status']
        row.update(
            signed_estimate='', score_kind='', sample_variance='', sample_variance_kind='',
            standard_error='', empirical_status='', empirical_constant='', nonzero_score_blocks='',
            first_order_degenerate='', maximum_block_influence_fraction='', pair_residual_energy_fraction='',
            original_independent_blocks=blocks, overlapping_pair_kernels=0,
            cdf_tail_status='', cdf_empirically_saturated_both_clocks='',
        )
        if family == 'field_lag_covariance':
            i = _integer(ident['leading_field'], 'LEADING_FIELD', 17)
            j = _integer(ident['trailing_field'], 'TRAILING_FIELD', 17)
            lag_ms = float(ident['lag_ms'])
            lag = _integer(lag_ms/BIN_MS, 'READOUT_LAG')
            start, stop = PHASES[phase]
            undefined = lag >= round((stop-start)/BIN_MS)
            require((status == UNDEFINED) == undefined, 'FROZEN_UNDEFINED_LAG_CONTRADICTED')
            if undefined:
                require(row['margin'] == '', 'UNDEFINED_MARGIN_MUST_REMAIN_EMPTY')
                row.update(score_kind=UNDEFINED, empirical_status=UNDEFINED)
                continue
            require(status != STRUCTURAL, 'STRUCTURAL_STATUS_REQUIRES_BASELINE_PAIRED_RESPONSE')
            groups[ident['condition'], phase, lag].append((slot, left, right, i, j))
        else:
            require(status != UNDEFINED, 'UNDEFINED_STATUS_REQUIRES_MISSING_WITHIN_PHASE_LAG')
        structural = family == 'paired_response' and phase == 'baseline'
        if family == 'count_cdf':
            require('physical_maximum_count' in row, 'VERIFIED_PHYSICAL_COUNT_MAXIMUM_REQUIRED')
            maximum = _integer(row['physical_maximum_count'], 'PHYSICAL_MAXIMUM_COUNT')
            threshold = _integer(ident['count_threshold'], 'COUNT_THRESHOLD')
            structural = threshold >= maximum
        require((status == STRUCTURAL) == structural, 'STRUCTURAL_STATUS_REQUIRES_SOURCE_IDENTITY')
        margin = float(row['margin'])
        require(np.isfinite(margin) and (margin == 0 if structural else margin > 0),
                'FROZEN_NUMERICAL_MARGIN_REQUIRED')
        if family == 'field_lag_covariance':
            continue
        if family == 'count_cdf':
            window = _integer(ident['window_ms'], 'COUNT_WINDOW')
            require(window in WINDOWS_MS, 'FROZEN_COUNT_WINDOW_REQUIRED')
            ci = CONDITIONS.index(ident['condition'])
            wi, fi = WINDOWS_MS.index(window), int(ident['field_index'])
            ki = int(np.searchsorted(support, threshold, side='right')-1)
            a, b = cdfs[left][ci, :, pi, wi, fi, ki], cdfs[right][ci, :, pi, wi, fi, ki]
            delta = a-b
            saturated = bool(np.all(a == 1) and np.all(b == 1))
            if structural:
                require(saturated and np.all(delta == 0), 'SOURCE_PHYSICAL_SUPPORT_CDF_CONTRADICTED')
            row.update(
                cdf_tail_status=('SOURCE_PHYSICAL_SUPPORT_IDENTITY' if structural
                                 else 'AT_OR_BEYOND_OBSERVED_MAXIMUM_UNCERTAIN' if threshold >= support[-1]
                                 else 'EMPIRICAL_SATURATION_UNCERTAIN' if saturated
                                 else 'OBSERVED_CDF_NO_POPULATION_TAIL_GUARANTEE'),
                cdf_empirically_saturated_both_clocks=saturated,
            )
            kind = 'paired_block_cdf_difference'
        else:
            delta = linear_block_score(family, ident, means, responses)
            if structural:
                pos, neg = next((CONDITIONS.index(a), CONDITIONS.index(b))
                                for name, a, b in CONTRASTS if name == ident['contrast'])
                mi = METRICS.index(ident['metric'])
                require(np.all(delta == 0) and all(np.all(
                    responses[label][pos, :, pi, mi] == responses[label][neg, :, pi, mi])
                    for label in (left, right)), 'PRE_CUE_STRUCTURAL_ZERO_CONTRADICTED')
            kind = 'paired_block_linear_difference'
        require(delta.shape == (blocks,) and np.isfinite(delta).all(), 'FINITE_ORIGINAL_BLOCK_SCORES_REQUIRED')
        scores[slot] = delta
        _empirical_summary(row, delta, delta.mean(), kind)

    # Only one condition/phase/lag's pair kernels are resident at once.
    for (condition, phase, lag), group in groups.items():
        start, stop = PHASES[phase]
        kernels = {label: lag_pair_kernels(
            fields[label][condition][:, round(start/BIN_MS):round(stop/BIN_MS)], lag, pairs)
            for label in LABELS}
        for left, right in zip(LABELS[:-1], LABELS[1:]):
            selected = [item for item in group if item[1:3] == (left, right)]
            if not selected:
                continue
            slots = [item[0] for item in selected]
            ii, jj = [item[3] for item in selected], [item[4] for item in selected]
            h = (kernels[left][:, ii, jj]-kernels[right][:, ii, jj]).T
            pseudovalues, _, influence, residual = covariance_jackknife(h, pairs, blocks)
            require(np.isfinite(pseudovalues).all(), 'FINITE_COVARIANCE_PSEUDOVALUES_REQUIRED')
            scores[slots] = pseudovalues
            for k, slot in enumerate(slots):
                rows[slot]['overlapping_pair_kernels'] = len(pairs)
                _empirical_summary(rows[slot], pseudovalues[k], h[k].mean(),
                                   'covariance_jackknife_pseudovalue', influence[k], residual[k])
    require(all(np.isnan(scores[i]).all() if row['definition_status'] == UNDEFINED
                else np.isfinite(scores[i]).all() for i, row in enumerate(rows)),
            'ONLY_UNDEFINED_LAGS_MAY_LACK_SCORES')
    return rows, scores

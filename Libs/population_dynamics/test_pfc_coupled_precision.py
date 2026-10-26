"""Frozen endpoint identity, directed block jackknife and complete CDF support."""
import json
from itertools import combinations

import numpy as np
import pytest

from .pfc_coupled_precision import LABELS, summarize_endpoints
from .pfc_confirmation_uncertainty import STRUCTURAL, UNDEFINED, EMPIRICAL
from .pfc_phase_responses import CONDITIONS


@pytest.fixture
def inputs():
    n = 4
    fields = {label: {condition: np.zeros((n, 1000, 17)) for condition in CONDITIONS}
              for label in LABELS}
    responses = {label: np.zeros((3, n, 5, 8)) for label in LABELS}
    cdfs = {label: np.broadcast_to([.25, .75, 1.], (3, n, 5, 3, 17, 3)).copy()
            for label in LABELS}
    return fields, responses, cdfs, np.array([0, 2, 5]), [95001, 95002, 95003, 95004]


def margin(family, index=0, status='ESTIMATED_EMPIRICAL_SPREAD', physical_maximum_count=100, **identity):
    ident = dict(left_cohort=LABELS[0], right_cohort=LABELS[1], phase='cue')
    ident.update(identity)
    row = dict(domain_row_index=str(index), family=family, endpoint_identity=json.dumps(ident),
               definition_status=status, margin='' if status == UNDEFINED else '0' if status == STRUCTURAL else '.01',
               units='probability' if family == 'count_cdf' else 'Hz^2' if family == 'field_lag_covariance' else 'Hz')
    if family == 'count_cdf' and physical_maximum_count is not None:
        row['physical_maximum_count'] = str(physical_maximum_count)
    return row


def test_directed_covariance_matches_independent_delete_one_block_calculation(inputs):
    fields, responses, cdfs, support, seeds = inputs
    rng = np.random.default_rng(702)
    for label in LABELS[:2]:
        fields[label]['cue'][:, 20:60, :2] = rng.uniform(0, 5, (4, 40, 2))
    margins = [margin('field_lag_covariance', index=19+i, condition='cue', lag_ms=5,
                      leading_field=a, trailing_field=b) for i, (a, b) in enumerate([(0, 1), (1, 0)])]
    rows, scores = summarize_endpoints(margins, *inputs)
    pairs = list(combinations(range(4), 2))
    points = []
    for ri, (i, j) in enumerate([(0, 1), (1, 0)]):
        h = []
        for a, b in pairs:
            values = []
            for label in LABELS[:2]:
                x = fields[label]['cue'][:, 20:60]
                values.append(np.mean((x[a, :-1, i]-x[b, :-1, i]) *
                                      (x[a, 1:, j]-x[b, 1:, j]))/2)
            h.append(values[0]-values[1])
        h = np.array(h); point = h.mean(); points.append(point)
        deleted = [h[[block not in pair for pair in pairs]].mean() for block in range(4)]
        expected = 4*point-3*np.array(deleted)
        np.testing.assert_allclose(scores[ri], expected, rtol=1e-12, atol=1e-13)
        assert rows[ri]['signed_estimate'] == pytest.approx(point)
        assert rows[ri]['sample_variance'] == pytest.approx(expected.var(ddof=1))
        assert rows[ri]['standard_error'] == pytest.approx(expected.std(ddof=1)/2)
        assert not np.isclose(rows[ri]['standard_error'], h.std(ddof=1)/np.sqrt(6))
        assert rows[ri]['overlapping_pair_kernels'] == 6
        assert rows[ri]['original_independent_blocks'] == 4
        assert rows[ri]['sample_variance_kind'] == 'jackknife_pseudovalue_variance'
    assert not np.isclose(*points)
    for original, result in zip(margins, rows):
        assert all(result[key] == value for key, value in original.items())


def test_signed_source_and_paired_responses_and_phase_means(inputs):
    fields, responses, *_ = inputs
    responses[LABELS[0]][1, :, 1, 6] = [-2, 1, 4, 8]
    responses[LABELS[0]][0, :, 1, 6] = -1
    responses[LABELS[1]][1, :, 1, 6] = 2
    responses[LABELS[1]][0, :, 1, 6] = [0, 1, 2, 3]
    fields[LABELS[0]]['cue'][:] = 500
    fields[LABELS[0]]['cue'][:, 20:60, 2] = np.array([1., 2., 4., 8.])[:, None]
    margins = [
        margin('paired_response', 7, contrast='cue_minus_omission', metric='vector_x_hz'),
        margin('source_response', 2, condition='cue', metric='vector_x_hz'),
        margin('field_mean', 11, condition='cue', field_index=2),
    ]
    rows, scores = summarize_endpoints(margins, *inputs)
    np.testing.assert_array_equal(scores, [[-3, 1, 5, 10], [-4, -1, 2, 6], [1, 2, 4, 8]])
    assert [row['domain_row_index'] for row in rows] == ['7', '2', '11']
    assert rows[0]['signed_estimate'] == 3.25
    assert rows[0]['nonzero_score_blocks'] == 4


def test_cdf_zero_mass_gaps_and_beyond_support_are_right_continuous(inputs):
    fields, responses, cdfs, support, seeds = inputs
    a = np.array([[0, .2, 1], [.1, .6, 1], [0, .5, 1], [.3, .8, 1]])
    b = np.array([[0, .4, 1], [.3, .3, 1], [0, .7, 1], [.1, .7, 1]])
    cdfs[LABELS[0]][1, :, 1, 0, 0] = a
    cdfs[LABELS[1]][1, :, 1, 0, 0] = b
    thresholds = [0, 1, 2, 4, 5, 20]
    margins = [margin('count_cdf', i, EMPIRICAL, condition='cue', window_ms=5,
                      field_index=0, count_threshold=threshold) for i, threshold in enumerate(thresholds)]
    rows, scores = summarize_endpoints(margins, *inputs)
    np.testing.assert_allclose(scores, (a-b)[:, [0, 0, 1, 1, 2, 2]].T)
    assert all(row['definition_status'] == EMPIRICAL for row in rows)
    assert rows[0]['empirical_status'] == 'ESTIMATED_EMPIRICAL_SPREAD'
    assert rows[-1]['empirical_status'] == EMPIRICAL
    assert rows[-1]['empirical_constant'] and rows[-1]['standard_error'] == 0
    assert rows[-1]['cdf_tail_status'] == 'AT_OR_BEYOND_OBSERVED_MAXIMUM_UNCERTAIN'
    assert rows[-1]['cdf_empirically_saturated_both_clocks']


def test_frozen_undefined_and_structural_zero_preserve_their_distinct_meaning(inputs):
    margins = [
        margin('field_lag_covariance', 91, UNDEFINED, condition='cue', phase='baseline',
               lag_ms=100, leading_field=0, trailing_field=1),
        margin('paired_response', 12, STRUCTURAL, contrast='cue_minus_omission',
               phase='baseline', metric='all_unit_rate_hz'),
        margin('field_lag_covariance', 17, condition='cue', phase='baseline',
               lag_ms=5, leading_field=0, trailing_field=1),
    ]
    rows, scores = summarize_endpoints(margins, *inputs)
    assert np.isnan(scores[0]).all() and np.isfinite(scores[1:]).all()
    assert rows[0]['signed_estimate'] == rows[0]['sample_variance'] == ''
    assert rows[1]['definition_status'] == rows[1]['empirical_status'] == STRUCTURAL
    assert rows[1]['signed_estimate'] == rows[1]['sample_variance'] == 0
    assert rows[2]['first_order_degenerate'] is True
    assert rows[2]['empirical_status'] == EMPIRICAL
    assert rows[2]['maximum_block_influence_fraction'] == ''
    assert rows[2]['pair_residual_energy_fraction'] == ''


@pytest.mark.parametrize('status,lag', [(UNDEFINED, 5), ('ESTIMATED_EMPIRICAL_SPREAD', 100)])
def test_frozen_undefined_status_cannot_contradict_available_time_pairs(inputs, status, lag):
    row = margin('field_lag_covariance', status=status, condition='cue', phase='baseline',
                 lag_ms=lag, leading_field=0, trailing_field=1)
    with pytest.raises(ValueError, match='FROZEN_UNDEFINED_LAG_CONTRADICTED'):
        summarize_endpoints([row], *inputs)


def test_equal_clock_contrasts_do_not_hide_broken_pre_cue_identity(inputs):
    responses = inputs[1]
    for label in LABELS[:2]:
        responses[label][1, :, 0, 0] = 2  # Difference of clock contrasts is zero, but source identity is false.
    row = margin('paired_response', status=STRUCTURAL, contrast='cue_minus_omission',
                 phase='baseline', metric='all_unit_rate_hz')
    with pytest.raises(ValueError, match='PRE_CUE_STRUCTURAL_ZERO_CONTRADICTED'):
        summarize_endpoints([row], *inputs)


@pytest.mark.parametrize('invalid', ['nonfinite', 'decreasing', 'missing_mass'])
def test_invalid_complete_cdf_is_rejected(inputs, invalid):
    values = inputs[2][LABELS[2]]
    if invalid == 'nonfinite':
        values[0, 0, 0, 0, 0, 0] = np.nan
    elif invalid == 'decreasing':
        values[0, 0, 0, 0, 0] = [.8, .7, 1]
    else:
        values[0, 0, 0, 0, 0, -1] = .9
    with pytest.raises(ValueError, match='COMPLETE_MONOTONE_OBSERVED_CDF_MASS_REQUIRED'):
        summarize_endpoints([], *inputs)


def test_covariance_first_order_degeneracy_retains_nonzero_pair_residual(inputs):
    # Equal block influence can coexist with nonconstant overlapping kernels.
    inputs[0][LABELS[0]]['cue'][:, 20:60, 0] = np.array([2., 2., 0., 0.])[:, None]
    row = margin('field_lag_covariance', condition='cue', lag_ms=0,
                 leading_field=0, trailing_field=0)
    rows, scores = summarize_endpoints([row], *inputs)
    np.testing.assert_allclose(scores[0], np.full(4, 4/3))
    assert rows[0]['signed_estimate'] == pytest.approx(4/3)
    assert rows[0]['first_order_degenerate'] is True
    assert rows[0]['sample_variance'] == rows[0]['standard_error'] == 0
    assert rows[0]['pair_residual_energy_fraction'] == pytest.approx(1.)
    assert rows[0]['empirical_status'] == EMPIRICAL
    assert rows[0]['nonzero_score_blocks'] == 4


def test_physical_support_cdf_identities_preserve_frozen_structural_status(inputs):
    for label in LABELS[:2]:
        inputs[2][label][1, :, 1, 0, 0, 1:] = 1
    margins = [margin('count_cdf', i, STRUCTURAL, physical_maximum_count=2,
                      condition='cue', window_ms=5, field_index=0, count_threshold=threshold)
               for i, threshold in enumerate([2, 4])]
    rows, scores = summarize_endpoints(margins, *inputs)
    np.testing.assert_array_equal(scores, np.zeros((2, 4)))
    for original, row in zip(margins, rows):
        assert all(row[key] == value for key, value in original.items())
        assert row['empirical_status'] == STRUCTURAL
        assert row['cdf_tail_status'] == 'SOURCE_PHYSICAL_SUPPORT_IDENTITY'
        assert row['cdf_empirically_saturated_both_clocks']
        assert row['signed_estimate'] == row['sample_variance'] == row['standard_error'] == 0


@pytest.mark.parametrize('left_cdf,right_cdf', [(.75, .75), (.8, .75)])
def test_physical_support_cdf_rejects_mass_beyond_bound_even_if_clock_delta_zero(inputs, left_cdf, right_cdf):
    inputs[2][LABELS[0]][1, :, 1, 0, 0, 1] = left_cdf
    inputs[2][LABELS[1]][1, :, 1, 0, 0, 1] = right_cdf
    row = margin('count_cdf', status=STRUCTURAL, physical_maximum_count=2,
                 condition='cue', window_ms=5, field_index=0, count_threshold=2)
    with pytest.raises(ValueError, match='SOURCE_PHYSICAL_SUPPORT_CDF_CONTRADICTED'):
        summarize_endpoints([row], *inputs)


@pytest.mark.parametrize('status,threshold', [(STRUCTURAL, 1), (EMPIRICAL, 2)])
def test_structural_cdf_status_must_follow_verified_physical_bound(inputs, status, threshold):
    row = margin('count_cdf', status=status, physical_maximum_count=2,
                 condition='cue', window_ms=5, field_index=0, count_threshold=threshold)
    with pytest.raises(ValueError, match='STRUCTURAL_STATUS_REQUIRES_SOURCE_IDENTITY'):
        summarize_endpoints([row], *inputs)


@pytest.mark.parametrize('bound', [None, 2.5])
def test_count_cdf_requires_verified_integer_physical_maximum(inputs, bound):
    row = margin('count_cdf', physical_maximum_count=bound,
                 condition='cue', window_ms=5, field_index=0, count_threshold=0)
    with pytest.raises(ValueError, match='PHYSICAL_'):
        summarize_endpoints([row], *inputs)

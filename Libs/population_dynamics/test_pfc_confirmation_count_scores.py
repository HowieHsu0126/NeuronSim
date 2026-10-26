"""Full-grid empirical CDF recovery tests retain block-level uncertainty."""
import json
import numpy as np
import pytest
from .pfc_confirmation_count_scores import recover_scores
from .pfc_confirmation_domain import NEW_STATUS


def fixture():
    data = dict(count_support=np.array([0,2,4]), source_rng_seeds=np.array([10,11]),
                condition_names=np.array(['cue']), phase_names=np.array(['delay']),
                window_ms=np.array([10.]), field_unit_counts=np.array([2]))
    data['left_cdf'] = np.array([[.5,.75,1.],[.25,.5,1.]]).reshape(1,2,1,1,1,3)
    data['right_cdf'] = np.array([[.5,.5,1.],[.25,.5,1.]]).reshape(1,2,1,1,1,3)
    base = dict(left_cohort='left',right_cohort='right',condition='cue',phase='delay',window_ms='10.0',field_index='0')
    rows = []
    for threshold in [0,1,3,5]:
        rows.append(dict(family='count_cdf',endpoint_identity=json.dumps(base | {'count_threshold':str(threshold)}),
                         empirical_status='test',definition_status=NEW_STATUS if threshold else 'EMPIRICALLY_DEGENERATE_UNCERTAIN',
                         reason='test',physical_maximum_count='6',estimate='' if threshold else '0.0'))
    return rows,data,['left','right'],[10,11]


def test_gapped_support_recovers_scores_and_keeps_empirical_zero_uncertain():
    rows,matrix,indices,checks = recover_scores(*fixture())
    np.testing.assert_array_equal(indices,[0,1,2,3])
    np.testing.assert_array_equal(matrix,[[0,0],[0,0],[.25,0],[0,0]])
    assert rows[2]['estimate'] == '0.125'
    assert rows[2]['definition_status'] == 'ESTIMATED_EMPIRICAL_SPREAD'
    assert rows[3]['definition_status'] == 'EMPIRICALLY_DEGENERATE_UNCERTAIN'
    assert checks['recovered_previously_unestimated_thresholds'] == 3
    assert checks['independent_joint_rng_blocks'] == 2


def test_incomplete_terminal_mass_cannot_be_extended_as_one():
    args = fixture();args[1]['left_cdf'][0,0,0,0,0,-1] = .95
    with pytest.raises(ValueError,match='CDF_TERMINAL_MASS_NOT_COMPLETE'):
        recover_scores(*args)


def test_shuffled_block_ids_are_rejected():
    args = fixture();args[1]['source_rng_seeds'] = np.array([11,10])
    with pytest.raises(ValueError,match='INDEPENDENT_BLOCK_IDENTITY_MISMATCH'):
        recover_scores(*args)


def test_archived_estimate_disagreement_is_preserved_as_failure():
    args = fixture();args[0][0]['estimate'] = '0.1'
    with pytest.raises(ValueError,match='ARCHIVED_CDF_POINT_ESTIMATE_MISMATCH'):
        recover_scores(*args)


def test_source_refractory_bound_contradiction_is_rejected():
    args = fixture();args[0][2]['physical_maximum_count'] = '3'
    with pytest.raises(ValueError,match='SOURCE_REFRACTORY_CDF_BOUND_CONTRADICTED'):
        recover_scores(*args)

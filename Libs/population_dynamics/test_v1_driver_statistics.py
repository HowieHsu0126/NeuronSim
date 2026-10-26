"""Protect source count-law and independent invocation reporting."""
import numpy as np
import pytest
from .v1_driver_statistics import source_count_moments,seed_collision_summary,count_diagnostics,block_seed_reuse,envelope_count_summary,finite_seed_count_summary,finite_seed_count_lookup,finite_gray_joint_count_summary


def test_source_primary_is_bernoulli_and_unused_terminal_rate_selects_branch():
    clock=np.array([0.,.001,.002])
    rates=np.array([[10.,20.,0.],[10.,20.,200.]])
    result=source_count_moments(clock,rates)
    np.testing.assert_allclose(result['mean'],[.03,.03])
    np.testing.assert_allclose(result['variance'],[.0099+.0196,.03])
    assert result['poisson'].tolist()==[False,True]


def test_actual_seed_collisions_preserve_pairs_and_multiplicity():
    result=seed_collision_summary(np.array([3,3,3,5,5,9]))
    assert result==dict(draws=6,unique_seeds=3,repeated_draws=3,colliding_pairs=4,largest_multiplicity=3)
    with pytest.raises(ValueError,match='SEED_DOMAIN'):
        seed_collision_summary(np.array([10000]))


def test_whole_invocation_statistics_keep_silent_units_and_no_equivalence_claim():
    counts=np.array([[0,1],[0,2],[0,3],[0,4]])
    moments=dict(mean=np.array([0.,2.]),variance=np.array([0.,2.]),poisson=np.array([False,True]))
    result=count_diagnostics(counts,moments)
    assert result[0]['mean_count_interval_95']==[0.,0.] and result[0]['variance_ratio'] is None
    assert result[1]['noise_blocks']==4 and result[1]['mean_count']==2.5
    assert all(row['scientific_equivalence']=='NOT_EVALUATED_NO_FROZEN_TOLERANCE' for row in result)


def test_seed_reuse_separates_unit_block_pairs_without_rejecting_independence():
    seeds=np.array([[1,1],[1,2],[3,2]])
    counts=np.array([[4,8],[4,5],[6,5]])
    result=block_seed_reuse(seeds,counts,np.array([10,20]),np.array([4,7,9]))
    assert result['colliding_pairs']==4
    assert result['same_unit_cross_block_seed_pairs']==2
    assert result['cross_unit_seed_pairs']==2
    assert result['same_block_cross_unit_seed_pairs']==1
    assert result['cross_block_seed_pairs']==3
    assert result['per_source_unit'][0]['repeated_seed_groups']==[
        dict(primary_seed=1,noise_block_indices=[4,7],invocation_count=2,count=4)]
    assert result['distributional_independence_tested'] is False
    assert result['event_train_equality_tested'] is False


def test_same_unit_seed_count_mismatch_is_not_hidden():
    with pytest.raises(ValueError,match='SAME_UNIT_SEED_COUNT_REPLAY_DIFFERS'):
        block_seed_reuse(np.array([[1],[1]]),np.array([[4],[5]]),
                         np.array([10]),np.array([4,7]))


def test_seed_reuse_requires_distinct_block_and_unit_identities():
    with pytest.raises(ValueError,match='BOUND_SEED_COUNT_MATRIX'):
        block_seed_reuse(np.array([[1],[2]]),np.array([[0],[0]]),
                         np.array([10]),np.array([4,4]))


def test_condition_envelope_retains_all_conditions_and_joint_descriptions():
    counts=np.zeros((3,10,2),dtype=np.int64)
    counts[:,0,0]=[1,2,3]
    counts[:,1,1]=[2,4,6]
    expected=np.zeros(counts.shape,dtype=np.float64)
    result=envelope_count_summary(counts,expected)
    assert result['conditions']==10 and result['source_units']==2
    assert result['total_count_mean_by_condition'][:2]==[2.,4.]
    assert result['descriptive_total_count_covariance'][0][1]==2.
    assert result['joint_distribution_qualified'] is False
    assert len(result['descriptive_count_error_covariance'])==10


def test_condition_envelope_rejects_missing_conditions():
    with pytest.raises(ValueError,match='TEN_CONDITION_COUNT_ENVELOPE'):
        envelope_count_summary(np.zeros((3,9,2),dtype=np.int64),
                               np.zeros((3,9,2),dtype=np.float64))


def test_finite_seed_moments_and_source_class_lookup_are_exact():
    table=np.vstack([np.arange(10000,dtype=np.int64)%2,np.arange(10000,dtype=np.int64)%4])
    summary=finite_seed_count_summary(table)
    assert summary['mean_count']==[.5,1.5]
    assert summary['count_variance']==[.25,1.25]
    assert summary['all_seeds_enumerated'] and not summary['joint_noise_law_qualified']
    counts=finite_seed_count_lookup(table,np.array([1,0,1]),np.array([[3,2,4],[0,1,2]]))
    np.testing.assert_array_equal(counts,[[3,0,0],[0,1,2]])


def test_finite_seed_lookup_rejects_partial_seed_domain_and_invalid_identity():
    with pytest.raises(ValueError,match='COMPLETE_FINITE_SEED_COUNT_TABLE'):
        finite_seed_count_summary(np.zeros((2,9999),dtype=np.int64))
    with pytest.raises(ValueError,match='LOOKUP_IDENTITIES'):
        finite_seed_count_lookup(np.zeros((2,10000),dtype=np.int64),np.array([2]),np.array([[0]]))


def test_joint_count_reference_separates_independent_seeds_from_shared_paths():
    table = np.vstack([np.arange(10000) % 2, 2 * (np.arange(10000) % 2)])
    classes = np.array([0, 0, 1, 1])
    seeds = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
    independent = finite_gray_joint_count_summary(
        table, classes, finite_seed_count_lookup(table, classes, seeds))
    reference = independent['independent_primary_seed_reference']
    assert reference['total_count_variance'] == 2.5
    np.testing.assert_allclose(reference['source_model_total_covariance'], [[.5, 0], [0, 2]])
    np.testing.assert_allclose(independent['descriptive_mean_distinct_unit_count_covariance'], 0, atol=1e-14)
    pair = independent['conditional_pair_references']
    np.testing.assert_allclose(pair['equal_primary_seed_covariance'], [[.25, .5], [.5, 1]])
    np.testing.assert_allclose(
        np.asarray(pair['equal_primary_seed_covariance']) / 10000
        + np.asarray(pair['distinct_primary_seed_covariance']) * 9999 / 10000,
        0, atol=1e-14)
    shared = finite_gray_joint_count_summary(
        table, classes, np.array([[0, 0, 0, 0], [1, 1, 2, 2]] * 2))
    assert shared['observed_total_count_variance'] == 12
    assert shared['observed_to_independent_total_variance_ratio'] == 4.8
    assert shared['distinct_unit_ordered_pairs'] == [[2, 4], [4, 2]]
    assert shared['distributional_independence_tested'] is False
    assert shared['scientific_equivalence'] == 'NOT_EVALUATED_NO_FROZEN_TOLERANCE'


def test_joint_count_reference_preserves_silent_units_and_single_unit_classes():
    result = finite_gray_joint_count_summary(
        np.zeros((1, 10000), dtype=np.int64), np.array([0]), np.zeros((3, 1), dtype=np.int64))
    assert result['full_lgn_units'] == 1
    assert result['observed_to_independent_total_variance_ratio'] is None
    assert result['descriptive_mean_distinct_unit_count_covariance'] == [[None]]
    with pytest.raises(ValueError, match='JOINT_COUNT_IDENTITIES'):
        finite_gray_joint_count_summary(np.zeros((2, 10000), dtype=np.int64),
                                       np.array([0]), np.zeros((3, 1), dtype=np.int64))

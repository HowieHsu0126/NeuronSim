"""Direct scores and complete-invocation bootstrap, without scientific claims."""
import numpy as np
import pytest
from .v1_driver_joint_precision import comparison_scores,joint_bootstrap,require_experiment_seeds,finite_population_counts,replay_starting_seeds


def test_joint_population_scores_preserve_complete_invocations():
    values=np.array([[[0,1],[2,1]],[[2,1],[0,1]],[[1,1],[1,1]]],dtype=np.int64)
    means=np.array([[1.,1.],[1.,1.]])
    lags=np.array([[.5,0.],[.5,0.]])
    result=comparison_scores(values,means,lags,np.array([1,1]),np.array([1.,1.]),np.array([0,1]))
    np.testing.assert_array_equal(result['mean_scores'],[[-1,1],[1,-1],[0,0]])
    np.testing.assert_array_equal(result['total_covariance_scores'][0],[[1,-1],[-1,1]])
    np.testing.assert_array_equal(result['pooled_second_moment_scores'],[0,0,0])
    np.testing.assert_array_equal(result['temporal_covariance_reference'][0],np.eye(2))
    np.testing.assert_array_equal(result['temporal_covariance_scores'][0,0],[[1,-1],[-1,1]])


def test_joint_bootstrap_keeps_exact_cross_population_relations():
    scores=np.arange(10,dtype=float)[:,None]*np.array([1.,-2.])[None,:]
    result=joint_bootstrap(scores,draws=1000,seed=41)
    np.testing.assert_allclose(result['estimate'],[4.5,-9])
    assert result['lower'][1] == pytest.approx(-2*result['upper'][0])
    assert result['upper'][1] == pytest.approx(-2*result['lower'][0])
    assert result['unit']=='complete starting-seed invocation'
    assert np.array_equal(joint_bootstrap(scores,draws=1000,seed=41)['lower'],result['lower'])


def test_constant_score_retains_defined_zero_width():
    result=joint_bootstrap(np.zeros((100,2,2)),draws=1000,seed=42)
    assert not result['estimate'].any() and not result['lower'].any() and not result['upper'].any()


def test_zero_variance_class_requires_separate_defined_score():
    with pytest.raises(ValueError,match='BOUND_ARRAYS'):
        comparison_scores(np.zeros((3,2,2),dtype=np.int64),np.zeros((2,2)),np.ones((2,1)),np.array([1,1]),np.array([0.,1.]),np.array([0]))


def test_experiment_seeds_are_exactly_the_authorized_three():
    assert require_experiment_seeds(dict(bootstrap_seeds=[41,42,43])) == [41,42,43]
    for seeds in ([41],[41,42,44],[43,42,41],[41,42,True]):
        with pytest.raises(ValueError,match='EXPERIMENT_SEEDS'):
            require_experiment_seeds(dict(bootstrap_seeds=seeds))
    with pytest.raises(ValueError,match='SINGLE_EXPERIMENT'):
        require_experiment_seeds(dict(bootstrap_seeds=[41,42,43],bootstrap_seed=41))
    with pytest.raises(ValueError,match='BOOTSTRAP_INPUT'):
        joint_bootstrap(np.zeros((100,2)),draws=1000,seed=97013)


def test_finite_population_counts_preserve_duplicate_seeds_and_integer_sums():
    table=np.array([[0,1,0],[2,0,1],[0,0,0],[1,1,1]],dtype=np.int32)
    seeds=np.array([[1,1,0],[3,2,3]],dtype=np.int32)
    result=finite_population_counts(table,seeds)
    np.testing.assert_array_equal(result,table[seeds].sum(axis=1,dtype=np.int64))
    assert result.dtype==np.int64
    np.testing.assert_array_equal(result,[[4,1,2],[2,2,2]])
    with pytest.raises(ValueError,match='FINITE_COUNT_INPUT'):
        finite_population_counts(table,np.array([[4]]))


def test_new_development_seed_batch_is_bounded_and_does_not_reuse_history():
    np.testing.assert_array_equal(replay_starting_seeds(dict(first_starting_noise_seed=900000000,invocations=3),np.arange(100)),
                                  [900000000,900000001,900000002])
    for selection in (dict(first_starting_noise_seed=99,invocations=3),dict(first_starting_noise_seed=2**32-1,invocations=3),
                      dict(first_starting_noise_seed=True,invocations=3),dict(first_starting_noise_seed=100,invocations=1)):
        with pytest.raises(ValueError,match='REPLAY_SEEDS_OVERLAP|REPLAY_SEED_RANGE'):
            replay_starting_seeds(selection,np.arange(100))

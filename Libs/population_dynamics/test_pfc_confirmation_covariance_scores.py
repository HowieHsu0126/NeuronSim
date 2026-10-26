"""Directed covariance, U-statistic normalization and independent pair partitions."""
import numpy as np
import pytest
from .pfc_confirmation_covariance_scores import round_robin_pairings,lag_pair_kernels,difference_bounds


@pytest.mark.parametrize('n',[2,3,4,5,32])
def test_round_robin_partitions_all_pairs_without_reusing_a_block_within_matching(n):
    pairs,matchings=round_robin_pairings(n)
    assert pairs.shape==(n*(n-1)//2,2)
    assert matchings.shape==(n if n%2 else n-1,n//2)
    np.testing.assert_array_equal(np.sort(matchings.ravel()),np.arange(len(pairs)))
    assert np.all(pairs[:,0]<pairs[:,1])
    for matching in matchings:
        vertices=pairs[matching].ravel()
        assert len(set(vertices.tolist()))==len(vertices)


def test_directed_lag_is_not_symmetrized_or_transposed_in_block_swap():
    x=np.array([[[1.,0.],[0.,1.]],[[0.,0.],[0.,0.]]])
    pairs,_=round_robin_pairings(2)
    h=lag_pair_kernels(x,1,pairs)[0]
    np.testing.assert_array_equal(h,[[0.,.5],[0.,0.]])
    assert h[0,1]!=h[1,0]


def test_positive_lag_diagonal_can_be_negative():
    x=np.array([[[2.],[0.]],[[0.],[2.]]])
    pairs,_=round_robin_pairings(2)
    assert lag_pair_kernels(x,1,pairs)[0,0,0]==-2.


@pytest.mark.parametrize('lag',[0,1,3])
def test_all_pair_mean_is_direct_unbiased_lag_covariance(lag):
    x=np.random.default_rng(1201).normal(size=(5,7,2))
    pairs,_=round_robin_pairings(5)
    leading=x if lag==0 else x[:,:-lag]
    trailing=x if lag==0 else x[:,lag:]
    expected=np.einsum('bti,btj->ij',leading-leading.mean(0),trailing-trailing.mean(0))/((len(x)-1)*leading.shape[1])
    actual=lag_pair_kernels(x,lag,pairs).mean(0)
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)
    common=np.arange(7.)[:,None]+np.array([1.,-1.])
    np.testing.assert_allclose(lag_pair_kernels(x+common,lag,pairs).mean(0),actual,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(lag_pair_kernels(x[[3,1,4,0,2]],lag,pairs).mean(0),actual,rtol=1e-13,atol=1e-14)


def test_physical_diagonal_bound_is_sharper_only_at_zero_lag():
    left,right=np.array([1.,2.]),np.array([3.,4.])
    lo,hi=difference_bounds(left,right,0)
    np.testing.assert_array_equal(np.diag(lo),[-4.5,-8.])
    np.testing.assert_array_equal(np.diag(hi),[.5,2.])
    assert (lo[0,1],hi[0,1])==(-7.,7.)
    lo,hi=difference_bounds(left,right,1)
    np.testing.assert_array_equal(np.diag(lo),[-5.,-10.])
    np.testing.assert_array_equal(np.diag(hi),[5.,10.])


def test_absent_within_phase_pairs_are_undefined():
    pairs,_=round_robin_pairings(2)
    assert lag_pair_kernels(np.ones((2,2,1)),2,pairs) is None

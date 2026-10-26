"""Verify source-order vectors, shared background and caller RNG lifetime."""
import random
import numpy as np
import pytest
from .v1_driver_seed_contract import expected_bernoulli_seed_vector, verify_seed_plan


def plan(ids,seed):
    n=len(ids)
    return dict(population=np.zeros(n,dtype=np.int64),source_id=np.asarray(ids),source_counts=np.zeros(n,dtype=np.int64),
        source_event_unit=np.array([],dtype=np.int64),source_event_time_ms=np.array([],dtype=float),
        source_primary_seed=expected_bernoulli_seed_vector(seed,n),source_poisson_seed=np.full(n,-1,dtype=np.int64),source_sampler_code=np.zeros(n,dtype=np.int8))


def test_full_source_order_vector_and_rng_preservation():
    before=np.random.get_state();python_before=random.getstate()
    value=plan(np.arange(17400),130002001)
    assert verify_seed_plan(value,130002001,branch='bernoulli')['source_units']==17400
    assert np.array_equal(np.random.get_state()[1],before[1]) and np.random.get_state()[2:]==before[2:]
    assert random.getstate()==python_before


def test_changed_vector_rejects_even_with_valid_seed_range():
    value=plan([31,7,99],12345);value['source_primary_seed'][1]=(value['source_primary_seed'][1]+1)%10000
    with pytest.raises(ValueError,match='ACTUAL_VECTOR_DIFFERS'):verify_seed_plan(value,12345,branch='bernoulli')


def test_single_background_consumes_primary_and_poisson_seeds():
    seed=120002001;rng=np.random.RandomState(seed);value=plan([0],seed)
    value['source_primary_seed'][0]=rng.randint(10000);value['source_poisson_seed'][0]=rng.randint(10000);value['source_sampler_code'][0]=1
    assert verify_seed_plan(value,seed,branch='single_shared_background_poisson')['seed_vector_exact']
    value['source_poisson_seed'][0]+=1
    with pytest.raises(ValueError,match='ACTUAL_VECTOR_DIFFERS'):verify_seed_plan(value,seed,branch='single_shared_background_poisson')


def test_multiple_background_units_require_another_contract():
    with pytest.raises(ValueError,match='ACTUAL_VECTOR_DIFFERS'):
        verify_seed_plan(plan([0,1],123),123,branch='single_shared_background_poisson')

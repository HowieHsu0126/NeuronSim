import numpy as np
import pytest

from Libs.population_dynamics.native_driver_trace import decode_isolated_arrivals
from Libs.population_dynamics.nest_gpu_poisson import _decode


def test_single_addition_matches_strict_decay_inversion():
    rng=np.random.default_rng(208)
    decay=np.float32(np.exp(-.1/.5))
    arrivals=rng.poisson(2.,(200,500)).astype(np.int64)
    pre=np.zeros((201,500),np.float32);post=np.empty((200,500),np.float32)
    for t in range(200):
        post[t]=(pre[t].astype(np.float64)+arrivals[t]).astype(np.float32)
        pre[t+1]=decay*post[t]
    fast,_=decode_isolated_arrivals(pre[:-1],post)
    strict,_=_decode(pre.astype(np.float64),np.full(500,decay,dtype=np.float64))
    np.testing.assert_array_equal(fast,arrivals)
    np.testing.assert_array_equal(strict,fast)


def test_noninteger_or_large_input_is_not_rounded_into_validity():
    with pytest.raises(ValueError,match="no unique integer"):
        decode_isolated_arrivals(np.array([0.],np.float32),np.array([.25],np.float32))
    with pytest.raises(ValueError,match="below"):
        decode_isolated_arrivals(np.array([0.],np.float32),np.array([2**20],np.float32))

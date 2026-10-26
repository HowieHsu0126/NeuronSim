"""Ensure batched source connections retain identities and associations."""
import numpy as np
import pytest

from .v1_nest_network import edge_batches


def group():
    return {'source':np.array([2,0,2,1]),'target':np.array([0,2,0,1]),
            'params':{'receptor_type':4,'weight':np.array([2.,-3.,0.,7.]),'delay':np.array([1.7,.9,1.,3.8])}}


def test_nonunique_unsorted_edges_preserve_weight_delay_and_port_across_batches():
    raw=group(); mapping=np.array([31,11,21])
    batches=list(edge_batches([raw],mapping,mapping,3))
    assert [len(row['source']) for row in batches]==[3,1]
    assert np.concatenate([row['source'] for row in batches]).tolist()==[21,31,21,11]
    assert np.concatenate([row['target'] for row in batches]).tolist()==[31,21,31,11]
    np.testing.assert_array_equal(np.concatenate([row['weight'] for row in batches]),raw['params']['weight'])
    np.testing.assert_array_equal(np.concatenate([row['delay'] for row in batches]),raw['params']['delay'])
    assert all(row['receptor_type']==4 for row in batches)


def test_scalar_delay_broadcast_preserves_zero_weights_and_source_arrays():
    raw=group();raw['params']['delay']=1.7;original=raw['params']['weight'].copy()
    rows=list(edge_batches([raw],np.array([9,7,8]),np.array([1,3,2]),2))
    assert np.concatenate([row['delay'] for row in rows]).tolist()==[1.7]*4
    rows[0]['weight'][0]=999
    np.testing.assert_array_equal(raw['params']['weight'],original)


@pytest.mark.parametrize('change',['bad_id','float_id','bad_weight_shape','nonfinite_weight','zero_delay','negative_delay','bad_port'])
def test_invalid_raw_connections_fail_before_native_use(change):
    raw=group()
    if change=='bad_id':raw['source'][0]=3
    elif change=='float_id':raw['source']=raw['source'].astype(float)
    elif change=='bad_weight_shape':raw['params']['weight']=np.array([1.])
    elif change=='nonfinite_weight':raw['params']['weight'][0]=np.nan
    elif change=='zero_delay':raw['params']['delay'][0]=0.
    elif change=='negative_delay':raw['params']['delay'][0]=-1.
    else:raw['params']['receptor_type']=0
    with pytest.raises(ValueError):list(edge_batches([raw],np.array([1,2,3]),np.array([1,2,3]),3))

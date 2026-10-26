"""Exported clock arrays preserve the existing scientific reader interfaces."""
import numpy as np
import pytest

from .pfc_coupled_precision_comparison import export_arrays, SEEDS, SIZES, LABELS
from .pfc_confirmation_linear_scores import validate_responses, source_metric_bounds
from .pfc_delay_analysis import PHASES
from .pfc_phase_responses import CONDITIONS, METRICS


@pytest.fixture
def exported(tmp_path):
    n = len(SEEDS)
    fields = {label: {condition: np.broadcast_to(
        (100*(li+1)+ci+np.arange(17)/100)[None,None,:], (n,1000,17))
        for ci,condition in enumerate(CONDITIONS)} for li,label in enumerate(LABELS)}
    base = np.arange(3*n*5*8, dtype=float).reshape(3,n,5,8)%19/10
    base[:,:,0,:] = np.arange(n)[None,:,None]/100+np.arange(8)[None,None,:]/1000
    responses = {label: base*(1+li/10) for li,label in enumerate(LABELS)}
    support = np.array([0,2,5], dtype=np.int64)
    cdfs = {label: np.broadcast_to(np.array([.2,.8,1.]),(3,n,5,3,17,3)) for label in LABELS}
    scores = np.vstack([np.arange(n)/100, np.full(n,np.nan)])
    indices = [7,413]
    export_arrays(tmp_path,fields,responses,cdfs,support,scores,indices)
    return tmp_path,fields,responses,cdfs,support,scores,indices


def test_exported_responses_are_accepted_by_existing_scientific_reader(exported):
    root,_,responses,_,_,_,_=exported
    with np.load(root/'coupled_responses.npz',allow_pickle=False) as stored:
        data={key:stored[key] for key in stored.files}
    bounds=source_metric_bounds(np.r_[np.full(16,600.),1000.],SIZES)
    actual=validate_responses(data,list(LABELS),SEEDS,bounds)
    for label in LABELS:np.testing.assert_array_equal(actual[label],responses[label])
    assert data['condition_names'].tolist()==list(CONDITIONS)
    assert data['phase_names'].tolist()==list(PHASES)
    assert data['metric_names'].tolist()==list(METRICS)
    assert data['units'].item()=='Hz'
    np.testing.assert_array_equal(data['phase_start_ms'],[x[0] for x in PHASES.values()])
    np.testing.assert_array_equal(data['phase_stop_ms'],[x[1] for x in PHASES.values()])


def test_export_retains_original_row_map_time_fields_and_complete_cdf_support(exported):
    root,fields,_,cdfs,support,scores,indices=exported
    with np.load(root/'coupled_block_scores.npz',allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['source_rng_seeds'],SEEDS)
        np.testing.assert_array_equal(stored['domain_row_indices'],indices)
        np.testing.assert_array_equal(stored['scores'],scores)
    with np.load(root/'coupled_field_rates.npz',allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['field_indices'],np.arange(17))
        np.testing.assert_array_equal(stored['field_unit_counts'],SIZES)
        np.testing.assert_array_equal(stored['time_start_ms'],np.arange(1000)*5.)
        assert stored['bin_ms'].item()==5. and stored['units'].item()=='Hz'
        for label,level in fields.items():
            for condition,values in level.items():
                np.testing.assert_array_equal(stored[label+'_'+condition],values)
    with np.load(root/'coupled_count_cdfs.npz',allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['count_support'],support)
        assert stored['units'].item()=='probability'
        for label,values in cdfs.items():np.testing.assert_array_equal(stored[label+'_cdf'],values)

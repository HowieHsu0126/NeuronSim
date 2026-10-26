"""Small serialized graphs distinguish complete coupling from duplicated components."""
import json
from pathlib import Path

import numpy as np
import pytest

from . import carlsim_checkpoint_graph as graph
from .test_carlsim_normalization import checkpoint
from .provenance import sha256_file


def row(pre,post,weight=1.):
    pg,pu=(0,pre) if pre<2 else (1,pre-2)
    qg,qu=(0,post) if post<2 else (1,post-2)
    return (pg,qg,pu,qu,7,weight,4.,1)


def test_cli_uses_group_local_ids_and_retains_all_nodes(tmp_path,capsys):
    source=tmp_path/'cycle.dat';checkpoint(source,[row(i,(i+1)%5) for i in range(5)])
    output=tmp_path/'result'
    assert graph.main(['--source',str(source),'--output',str(output),'--expected-sha256',sha256_file(source),'--chunk-records','2'])==0
    capsys.readouterr();report=json.loads((output/'report.json').read_text())
    assert report['components']['strong']==dict(components=1,largest_neurons=5,sizes=[5])
    assert report['components']['weak']==dict(components=1,largest_neurons=5,sizes=[5])
    assert report['nonzero_interunit_group_matrix']==[[1,1],[1,2]]
    assert report['distinct_directed_interunit_pairs']==5
    assert not report['source_scale_qualified'] and report['production_episodes']==0
    with np.load(output/'unit_components.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['unit_ids'],np.arange(5))
        assert len(np.unique(data['strong_component']))==1


def test_two_components_and_an_isolate_are_not_promoted_by_header_size(tmp_path):
    source=tmp_path/'split.dat';checkpoint(source,[row(0,1),row(1,0),row(2,3),row(3,2)])
    report=graph.analyze(source,tmp_path/'result')
    assert sorted(report['components']['strong']['sizes'])==[1,2,2]
    assert report['components']['strong']['largest_neurons']==2
    assert report['isolated_neurons']==1
    assert not report['all_units_in_one_strong_component']


def test_zero_links_self_loops_and_signed_parallel_synapses_have_explicit_semantics(tmp_path):
    source=tmp_path/'parallel.dat'
    rows=[row(0,1,1.),row(0,1,-1.)]*256+[row(1,0),row(1,2,0.),row(2,2)]
    checkpoint(source,rows)
    report=graph.analyze(source,tmp_path/'result',chunk_records=3)
    assert report['synapse_records']==515 and report['nonzero_interunit_synapses']==513
    assert report['negative_weight_records']==256 and report['zero_weight_records']==1 and report['self_loop_records']==1
    assert report['distinct_directed_interunit_pairs']==2
    assert sorted(report['components']['strong']['sizes'])==[1,1,1,2]
    assert report['isolated_neurons']==3


def test_chunk_boundaries_preserve_components_and_matrices(tmp_path):
    source=tmp_path/'cycle.dat';checkpoint(source,[row(i,(i+1)%5) for i in range(5)]+[row(0,2),row(4,1,-2.)])
    left=graph.analyze(source,tmp_path/'one',chunk_records=1)
    right=graph.analyze(source,tmp_path/'all',chunk_records=100)
    for name in ['source_sha256','components','nonzero_interunit_group_matrix','nonzero_interunit_half_matrix','distinct_directed_interunit_pairs']:
        assert left[name]==right[name]


@pytest.mark.parametrize('bad',[(1,1,3,0,7,1.,2.,1),(1,1,0,0,7,float('nan'),2.,1)])
def test_invalid_endpoints_or_weights_reject_the_whole_graph(tmp_path,bad):
    source=tmp_path/'bad.dat';checkpoint(source,[bad])
    output=tmp_path/'result'
    with pytest.raises(ValueError):graph.analyze(source,output)
    assert json.loads((output/'report.json').read_text())['status']=='FAILED'
    assert not (output/'unit_components.npz').exists()


def test_declared_hash_and_coupled_population_identity_are_enforced(tmp_path):
    source=tmp_path/'tiny.dat';checkpoint(source,[row(0,1)])
    with pytest.raises(ValueError,match='SOURCE_HASH_CHANGED'):
        graph.analyze(source,tmp_path/'hash',expected_sha256='0'*64)
    with pytest.raises(ValueError,match='COUPLED_SOURCE_IDENTITY_REQUIRED'):
        graph.analyze(source,tmp_path/'coupled',coupled_extension=True)
    assert not (tmp_path/'coupled').exists()

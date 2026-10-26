"""Whole-group counts retain multiplicity and explicit identity metadata."""
import hashlib
import json
from unittest.mock import patch
import numpy as np
import pytest
from .v1_edge_layout import describe_groups, inspect_layout


def test_preserves_explicit_ids_empty_groups_and_repeated_edge_counts():
    raw={'nodes':[], 'edges':[
        dict(source=np.array([0,1,0]),target=np.array([2,2,3]),edge_type_id=27,
             params=dict(weight=np.array([1.,2.,3.]),delay=1.5,receptor_type=1,source_rule_id=9)),
        dict(source=np.array([],dtype=int),target=np.array([],dtype=int),label='empty',params={})]}
    meta={'node_type_id':np.array([10,10,11,12])}
    types={10:{'pop_name':'source'},11:{'pop_name':'target'},12:{'pop_name':'target'}}
    report=describe_groups(raw,meta,types)
    assert report['raw_tuples']==3 and report['empty_groups']==1
    assert report['extra_group_key_counts']=={'edge_type_id':1,'label':1}
    assert report['groups'][0]['extra_fields']['edge_type_id']['value']==27
    assert report['groups'][0]['parameter_fields']['source_rule_id']['value']==9
    assert report['groups'][0]['target_populations']=={'target':3}
    assert report['groups'][0]['target_node_types']==[
        {'node_type_id':11,'tuple_occurrences':2},{'node_type_id':12,'tuple_occurrences':1}]
    assert report['groups_with_multiple_target_node_types']==1
    assert not report['group_order_to_sonata_type_qualified']
    assert np.array_equal(raw['edges'][0]['target'],[2,2,3])


def test_invalid_endpoint_is_not_an_unclassified_type():
    raw={'edges':[dict(source=np.array([0]),target=np.array([2]),params={})]}
    with pytest.raises(ValueError,match='ENDPOINT'):
        describe_groups(raw,{'node_type_id':np.array([10])},{10:{'pop_name':'p'}})


def test_changed_source_fails_before_unpickle(tmp_path):
    paths={key:tmp_path/key for key in ['graph','nodes','types']}
    for path in paths.values():path.write_bytes(b'fixture')
    archive=tmp_path/'baseline';(archive/'weights').mkdir(parents=True)
    reference=archive/'weights/report.json'
    reference.write_text(json.dumps(dict(status='SOURCE_WEIGHT_RULE_DIFFERS_DIAGNOSTIC',
        input_sha256={key:'changed' for key in paths})))
    (archive/'delivery.json').write_text(json.dumps(dict(files=[dict(path='weights/report.json',
        bytes=reference.stat().st_size,sha256=hashlib.sha256(reference.read_bytes()).hexdigest())])))
    with patch('population_dynamics.v1_edge_layout.pickle.load') as load:
        with pytest.raises(ValueError,match='CHANGED_BEFORE_UNPICKLE'):
            inspect_layout(**paths,comparison_archive=archive,output=tmp_path/'output')
        load.assert_not_called()
    assert json.loads((tmp_path/'output/report.json').read_text())['exit_code']==1

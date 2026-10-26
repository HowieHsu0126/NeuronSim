"""Check ID-based alignment, source group mapping, and missing declarations."""
import h5py
import numpy as np
import pytest

from population_dynamics.v1_node_metadata_comparison import compare_metadata


def nodes(path, *, ids=(10, 20), indices=(1, 0), types=(1, 2), y=True):
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('nodes/v1')
        for name, values in dict(node_id=ids, node_group_id=(0, 0),
                                 node_group_index=indices, node_type_id=types).items():
            group[name] = np.asarray(values, dtype=np.int64)
        for name in ('x', 'z', 'tuning_angle') + (('y',) if y else ()):
            group.create_dataset('0/'+name, data=np.array([4., 3.]))
    return path


def table():
    return {1:dict(node_type_id='1',pop_name='A',model_template='old'),
            2:dict(node_type_id='2',pop_name='B',model_template='old')}


def test_aligns_ids_and_maps_group_indices(tmp_path):
    old = nodes(tmp_path/'old.h5')
    current = nodes(tmp_path/'current.h5', ids=(20, 10), indices=(0, 1), types=(2, 1))
    result = compare_metadata(old,current,table(),table(),witnesses=(10,))
    assert result['nodes'] == 2 and not result['id_mapping']['row_order_equal']
    assert all(v['different_nodes'] == 0 for v in result['fields'].values())
    assert result['witnesses'][0]['legacy']['x']['value'] == 3.


def test_reports_missing_fields_and_same_id_changed_meaning(tmp_path):
    old = nodes(tmp_path/'old.h5', y=False)
    current = nodes(tmp_path/'current.h5')
    new_types=table()
    new_types[1]['model_template']='new'
    new_types[1]['layer']='L5'
    result=compare_metadata(old,current,table(),new_types)
    assert result['fields']['y']['legacy_missing'] == 2
    assert result['fields']['y']['compared_nodes'] == 0
    assert result['fields']['y']['presence_mismatches'] == 2
    assert result['fields']['node_type_id']['different_nodes'] == 0
    assert result['node_declaration_difference_counts']['model_template'] == 1
    assert result['node_declaration_difference_counts']['layer'] == 1
    assert len(result['type_tables']['same_id_declaration_differences']) == 1


def test_counts_type_transitions_and_coordinate_changes(tmp_path):
    old = nodes(tmp_path/'old.h5')
    current = nodes(tmp_path/'current.h5', types=(2, 2))
    with h5py.File(current,'r+') as handle:
        handle['nodes/v1/0/x'][1]=3.25
    result=compare_metadata(old,current,table(),table(),witnesses=(10,))
    assert result['fields']['node_type_id']['different_nodes'] == 1
    assert result['fields']['x']['different_nodes'] == 1
    assert result['fields']['x']['finite_max_absolute_difference'] == .25
    assert sum(row['nodes'] for row in result['node_type_transitions']) == 2
    assert result['witnesses'][0]['current']['node_type_id'] == 2


def test_nonfinite_and_signed_zero_are_visible(tmp_path):
    old=nodes(tmp_path/'old.h5');current=nodes(tmp_path/'current.h5')
    for path,zero in ((old,-0.),(current,0.)):
        with h5py.File(path,'r+') as handle:handle['nodes/v1/0/x'][...]=[np.nan,zero]
    result=compare_metadata(old,current,table(),table())
    assert result['fields']['x']['both_nan'] == 1
    assert result['fields']['x']['legacy_nonfinite'] == 1
    assert result['fields']['x']['signed_zero_mismatches'] == 1


@pytest.mark.parametrize('bad_ids,message', [((10,10),'DUPLICATE_NODE_ID'),((10,30),'ID_SETS_NOT_COMPARABLE')])
def test_rejects_uncomparable_identities(tmp_path,bad_ids,message):
    old=nodes(tmp_path/'old.h5');current=nodes(tmp_path/'current.h5',ids=bad_ids)
    with pytest.raises(ValueError,match=message):compare_metadata(old,current,table(),table())


def test_retains_verified_prefix_and_resumes_without_reread(tmp_path, monkeypatch):
    import json
    from population_dynamics import v1_node_metadata_comparison as module
    source=dict(etag='same-source',expected_bytes=12,parent_index_sha256='index')
    payload=b'abcdefghijkl'
    starts=[]
    class Reader:
        fail_once=True
        def __init__(self,*args,**kwargs):
            self.position=0;self.body=0;self.requested=0;self.failed=None
        def seek(self,offset):self.position=offset
        def read(self,length):
            starts.append(self.position);self.requested+=length
            if Reader.fail_once and self.position==4:
                Reader.fail_once=False;self.failed='OSError'
                raise OSError('truncated')
            data=payload[self.position:self.position+length]
            self.position+=len(data);self.body+=len(data)
            return data
        def counters(self):return dict(response_body_bytes=self.body,requested_bytes=self.requested,redirect_body_bytes=0)
        def close(self):pass
    monkeypatch.setattr(module,'RangeFile',Reader)
    first=tmp_path/'first';first.mkdir();report={}
    with pytest.raises(OSError,match='truncated'):
        module.download_legacy(source,'unused',first,report,range_bytes=4)
    assert (first/'legacy_v1_nodes.h5.part').read_bytes()==payload[:4]
    assert report['verified_prefix_bytes']==4
    assert report['recovery']['attempt_body_upper_bound_bytes']==8
    (first/'report.json').write_text(json.dumps(report))
    second=tmp_path/'second';second.mkdir();next_report={}
    final=module.download_legacy(source,'unused',second,next_report,resume=first,range_bytes=2)
    assert final.read_bytes()==payload
    assert starts==[0,4,4,6,8,10]
    assert next_report['resumed_prefix_bytes']==4 and next_report['verified_prefix_bytes']==12
    assert next_report['recovery']['response_body_upper_bound_bytes']==16
    assert next_report['legacy_asset']['complete']


@pytest.mark.parametrize('defect', ['source','prefix'])
def test_resume_rejects_changed_source_or_prefix_before_network(tmp_path,monkeypatch,defect):
    import hashlib,json
    from population_dynamics import v1_node_metadata_comparison as module
    old=tmp_path/'old';old.mkdir();part=old/'legacy_v1_nodes.h5.part';part.write_bytes(b'abcd')
    source=dict(etag='same',expected_bytes=8,parent_index_sha256='index')
    digest=hashlib.sha256(b'abcd').hexdigest()
    report=dict(source_identity=source,verified_prefix_bytes=4,verified_prefix_sha256=digest,
        legacy_asset=dict(path=part.name,bytes=4,sha256=digest),
        recovery=dict(attempt=1,response_body_upper_bound_bytes=4))
    (old/'report.json').write_text(json.dumps(report))
    if defect=='source':source=dict(source,etag='different')
    else:part.write_bytes(b'abce')
    def forbidden(*args,**kwargs):raise AssertionError('network must not start')
    monkeypatch.setattr(module,'RangeFile',forbidden)
    new=tmp_path/'new';new.mkdir()
    with pytest.raises(ValueError,match='SOURCE_IDENTITY_DIFFERS|VERIFIED_METADATA_PREFIX_REQUIRED'):
        module.download_legacy(source,'unused',new,{},resume=old,range_bytes=4)

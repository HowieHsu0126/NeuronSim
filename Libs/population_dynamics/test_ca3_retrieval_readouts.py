"""Sparse fixtures exercise source identities and raw-event readouts, not biology."""
import csv
import json
import struct

import numpy as np
import pytest

from . import ca3_retrieval_readouts as readouts
from .carlsim_records import (GROUP_HEADER,NETWORK_HEADER,NETWORK_SIGNATURE,
    SPIKE_HEADER,SPIKE_SIGNATURE,validate_retrieval)
from .provenance import sha256_file

PYR_EVENTS=[(1000,688),(520,138),(500,138),(519,138),(520,138),(539,138),(540,138),(499,138),
            (520,413),(539,688),(520,137),(730,413),(749,413),(750,413),(769,413),(770,413),
            (960,688),(979,688),(980,688),(999,688),(1000,688)]


def _json(path,value):
    path.write_text(json.dumps(value))


def _events(path,units,rows):
    path.write_bytes(SPIKE_HEADER.pack(SPIKE_SIGNATURE,.2,units,1,1)+
                     b''.join(struct.pack('<ii',*row) for row in rows))


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    root=tmp_path/'runtime';run=root/'native_run';(run/'results').mkdir(parents=True)
    (root/'bin').mkdir();(root/'bin/ca3_retrieval').write_bytes(b'synthetic fixture binary')
    _json(root/'manifest.json',{'fixture':True})
    # Only the pattern hash is replaced in this structural fixture. Production
    # pins the actual author's bytes; all native population sizes remain intact.
    pattern='\n'.join(str(i) for _ in range(4) for i in range(275))+'\n'
    (root/'trainPat.csv').write_text(pattern);(run/'trainPat.csv').write_text(pattern)
    pattern_sha=sha256_file(root/'trainPat.csv');monkeypatch.setattr(readouts,'PATTERN_SHA256',pattern_sha)
    layout={};cursor=0
    for name in sorted(readouts.POPULATIONS):
        units=readouts.POPULATIONS[name];layout[name]=(cursor,cursor+units);cursor+=units
    order=list(reversed(sorted(layout)))
    payload=NETWORK_HEADER.pack(NETWORK_SIGNATURE,.3,45.849998474121094,1.,cursor,len(order))
    for name in order:
        start,stop=layout[name]
        payload+=GROUP_HEADER.pack(start,stop-1,stop-start,1,1,name.encode())
        rows=PYR_EVENTS if name==readouts.PYRAMIDAL else [(0,0),(1000,stop-start-1)]
        _events(run/'results'/('spk_'+name+'.dat'),stop-start,rows)
    (run/readouts.INPUT_NETWORK).write_bytes(payload)
    proofs=[]
    for path,preserved,path_only,original,staged in [
        (readouts.MAIN_PATH,False,True,readouts.AUTHOR_MAIN_SHA256,readouts.OVERLAY_SHA256),
        ('generateCONFIGStateSTP.h',True,False,readouts.CONFIG_SHA256,readouts.CONFIG_SHA256),
        ('generateSETUPStateSTP.h',True,False,readouts.SETUP_SHA256,readouts.SETUP_SHA256)]:
        proofs.append(dict(path=path,source_bytes_preserved=preserved,source_path_binding_only=path_only,
                           source_sha256=original,staged_sha256=staged))
    build=dict(status='NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING',
        scope='native_source_retrieval',expected_duration_ms=1001,author_revision=readouts.AUTHOR_REVISION,
        framework_revision=readouts.SOURCE_REVISION,core_neurons=84053,populations=readouts.POPULATIONS,
        pattern_sha256=pattern_sha,pattern_integer_count=1100,
        overlay_source_sha256=readouts.OVERLAY_SHA256,original_author_main_sha256=readouts.AUTHOR_MAIN_SHA256,
        retrieval_program_changed_only_in_input_path=True,source_file_byte_checks=proofs,
        binary_sha256=sha256_file(root/'bin/ca3_retrieval'),
        retrieval_input_contract=dict(effective_input_binding=readouts.INPUT_NETWORK,source_path_binding_only=True,
                                     source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True))
    _json(root/'build_report.json',build)
    native=dict(build,status='NATIVE_RETRIEVAL_EXIT_ZERO',exit_code=0,native_run_executed=True,
                retrieval_executed=True,normalized_input_qualified=True,
                normalized_input_sha256=sha256_file(run/readouts.INPUT_NETWORK),
                runtime_manifest_sha256=sha256_file(root/'manifest.json'))
    _json(root/'native_report.json',native)
    validate_retrieval(root,root/'observations')
    return root,layout[readouts.PYRAMIDAL][0]


def test_memberships_are_asymmetric_source_subsets():
    masks=readouts.source_masks(830)
    np.testing.assert_array_equal(masks['target_never_boosted'].sum(1),[137]*3)
    np.testing.assert_array_equal(masks['other_never_boosted'].sum(1),[274]*3)
    np.testing.assert_array_equal(masks['eligible_for_boost'].sum(1),[138]*3)
    assert masks['eligible_for_boost'][0,137] and not masks['target_never_boosted'][0,137]
    assert masks['target_never_boosted'][0,138] and masks['target_never_boosted'][0,274]
    assert not any(value[:,825:].any() for value in masks.values())
    with pytest.raises(ValueError):readouts.source_masks(824)


def test_half_open_boundaries_duplicates_and_block_order():
    rows=np.asarray(PYR_EVENTS,dtype=np.int32)
    results=[]
    for ordered,chunk in [(rows,1),(rows,4),(rows[::-1],3)]:
        result=readouts.count_pyramidal_windows((ordered[i:i+chunk] for i in range(0,len(rows),chunk)),830)
        results.append(result)
        assert result['window_counts'][0,0,138]==2
        assert result['window_counts'][0,1,138]==3
        assert result['window_counts'][2,1,688]==2
        assert result['full_counts'][688]==7 and result['per_ms'][1000]==2
        assert result['records']==21
    for key in ('full_counts','per_ms','window_counts'):
        np.testing.assert_array_equal(results[0][key],results[1][key])
        np.testing.assert_array_equal(results[0][key],results[2][key])
    for wrong in [[(1001,0)],[(0,830)],[(0,-1)]]:
        with pytest.raises(ValueError):readouts.count_pyramidal_windows([np.asarray(wrong)],830)


def test_bound_cli_path_saves_global_identities_counts_and_all_cues(runtime,tmp_path):
    root,start=runtime;output=tmp_path/'readouts'
    report=readouts.analyze(root,output)
    assert report['cues_retained']==3 and report['full_record_recounted_units']==84053
    assert report['full_record_unit_counts_match'] and not report['pattern_completion_qualified']
    assert not report['actual_current_trace_verified'] and not report['terminal_native_clock_independently_verified']
    with np.load(output/'pyramidal_window_counts.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['pyramidal_global_unit_ids'],np.arange(start,start+75377))
        np.testing.assert_array_equal(saved['pyramidal_local_unit_ids'],np.arange(75377))
        assert saved['window_counts'].shape==(3,2,75377)
        assert saved['window_counts'][0,1,138]==3 and saved['full_record_unit_counts'][688]==7
        assert saved['eligible_for_boost_mask'][0].sum()==138
        assert saved['target_never_boosted_mask'][0].sum()==137
        assert saved['other_never_boosted_mask'][0].sum()==274
    with (output/'descriptive_readouts.csv').open() as f:rows=list(csv.DictReader(f))
    assert [row['cue_id'] for row in rows]==['1','2','3']
    first=rows[0]
    assert int(first['target_baseline_event_count'])==2 and int(first['target_cue_event_count'])==3
    assert int(first['other_cue_event_count'])==2 and int(first['eligible_cue_event_count'])==1
    assert float(first['target_cue_rate_hz'])==3/(137*.020)
    assert float(first['other_cue_rate_hz'])==2/(274*.020)
    assert rows[2]['following_source_background_ms']=='1'
    for name,digest in report['artifacts'].items():assert sha256_file(output/name)==digest
    before=sha256_file(output/'pyramidal_window_counts.npz')
    with pytest.raises(FileExistsError):readouts.analyze(root,output)
    assert sha256_file(output/'pyramidal_window_counts.npz')==before


@pytest.mark.parametrize('mutation',['native_status','pattern','raw_events','unit_ids','unit_totals','per_ms','config_proof'])
def test_wrong_or_mutated_inputs_are_rejected(runtime,tmp_path,mutation):
    root,start=runtime
    if mutation=='native_status':
        path=root/'native_report.json';native=json.loads(path.read_text());native['status']='RUNNING_NATIVE_RETRIEVAL';_json(path,native)
        p=root/'observations/recording.json';recording=json.loads(p.read_text());recording['native_report_sha256']=sha256_file(path);_json(p,recording)
    elif mutation=='pattern':
        (root/'native_run/trainPat.csv').write_text('0\n')
    elif mutation=='raw_events':
        p=root/'native_run/results/spk_CA3_Pyramidal.dat';p.write_bytes(p.read_bytes()+struct.pack('<ii',520,138))
    elif mutation=='config_proof':
        p=root/'build_report.json';build=json.loads(p.read_text());build['source_file_byte_checks'][1]['staged_sha256']='0'*64;_json(p,build)
    else:
        p=root/'observations/population_counts.npz'
        with np.load(p,allow_pickle=False) as saved:values={key:saved[key] for key in saved.files}
        if mutation=='unit_ids':values['unit_ids'][[0,1]]=values['unit_ids'][[1,0]]
        elif mutation=='unit_totals':
            values['unit_counts'][start+138]+=1;values['unit_counts'][start+137]-=1
        else:
            column=list(values['group_names']).index(readouts.PYRAMIDAL)
            values['population_counts_per_ms'][500,column]-=1;values['population_counts_per_ms'][501,column]+=1
        np.savez_compressed(p,**values)
        q=root/'observations/recording.json';recording=json.loads(q.read_text());recording['observation_sha256']=sha256_file(p);_json(q,recording)
    output=tmp_path/'rejected'
    with pytest.raises(ValueError):readouts.analyze(root,output)
    assert not output.exists()

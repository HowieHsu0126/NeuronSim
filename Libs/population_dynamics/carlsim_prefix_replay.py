"""Replay native CARLsim events and saved-network bytes, excluding wall time.

The saved network still is not a full-state restart. Byte equality checks the
realized saved payload; it does not validate its synaptic semantics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from .carlsim_records import network_header, _sha


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _same_bytes(a, b, *, ignore_execution_time=False):
    hashes=[hashlib.sha256(),hashlib.sha256()]
    equal=True; compared=0; offset=0
    with a.open('rb') as left,b.open('rb') as right:
        while True:
            x,y=left.read(1048576),right.read(1048576)
            if not x and not y:break
            hashes[0].update(x);hashes[1].update(y)
            if ignore_execution_time and offset==0:
                # Header is <ifffii: bytes12..15 contain elapsed wall seconds.
                x,y=x[:12]+x[16:],y[:12]+y[16:]
            equal &= x==y;compared+=max(len(x),len(y));offset+=1048576
    return equal,compared,[h.hexdigest() for h in hashes]


def _independent_counts(root, metadata, recording):
    n,duration=metadata['units'],recording['duration_ms']
    counts=np.zeros(n,dtype=np.int64)
    fields=np.zeros((duration,len(metadata['groups'])),dtype=np.int64)
    events=0
    for group in metadata['groups']:
        path=root/'native_run/results'/('spk_'+group.name+'.dat')
        with path.open('rb') as stream:
            header=stream.read(20)
            if len(header)!=20:raise ValueError('truncated event header')
            while True:
                block=stream.read(65536)
                if not block:break
                if len(block)%8:raise ValueError('truncated event pair')
                for timestamp,local in struct.iter_unpack('<ii',block):
                    if not 0<=timestamp<duration or not 0<=local<group.units:
                        raise ValueError('native event identity or time invalid')
                    counts[group.start+local]+=1
                    fields[timestamp,group.group_id]+=1
                    events+=1
    with np.load(root/'observations/population_counts.npz',allow_pickle=False) as saved:
        if (not np.array_equal(saved['unit_ids'],np.arange(n))
            or not np.array_equal(saved['unit_counts'],counts)
            or not np.array_equal(saved['population_counts_per_ms'],fields)):
            raise ValueError('independent event count projection differs')
    if events!=recording['records']:raise ValueError('recorded event denominator differs')
    return dict(events=events,unit_count_values=n,field_count_values=int(fields.size))


def compare(original, repeated, output):
    original,repeated,output=map(Path,(original,repeated,output))
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();reports=[];recordings=[];headers=[]
    for root in (original,repeated):
        report=_read(root/'native_report.json')
        recording=_read(root/'observations/recording.json')
        if (report['status']!='NATIVE_PREFIX_EXIT_ZERO' or report['exit_code']!=0
            or recording['status']!='DECODED_NATIVE_PREFIX_FILES'
            or recording['native_report_sha256']!=_sha(root/'native_report.json')
            or recording['observation_sha256']!=_sha(root/'observations/population_counts.npz')):
            raise ValueError('complete bound native execution and recording required')
        for row in recording['files']:
            if _sha(root/row['relative_path'])!=row['sha256']:
                raise ValueError('native spike file changed')
        reports.append(report);recordings.append(recording)
        headers.append(network_header(root/'native_run/network5p.dat'))
    if reports[1].get('independent_repeat_of_report_sha256')!=_sha(original/'native_report.json'):
        raise ValueError('repeat does not identify original completed native run')
    for key in ['framework_revision','author_revision','reused_binary_sha256','core_neurons','populations','expected_prefix_ms']:
        if reports[0][key]!=reports[1][key]:raise ValueError('native source changed: '+key)
    for key in ['units','groups','simulation_seconds','payload_offset']:
        if headers[0][key]!=headers[1][key]:raise ValueError('native header identity changed: '+key)
    checks=[]
    names=[('saved_network','native_run/network5p.dat',True)]
    names += [(g.name,'native_run/results/spk_'+g.name+'.dat',False) for g in headers[0]['groups']]
    for name,relative,ignore in names:
        equal,compared,digests=_same_bytes(original/relative,repeated/relative,ignore_execution_time=ignore)
        if ignore and digests!=[r['checkpoint_sha256'] for r in reports]:
            raise ValueError('saved network differs from native report')
        checks.append(dict(name=name,relative_path=relative,equal=equal,bytes_compared=compared,
                           original_sha256=digests[0],repeated_sha256=digests[1],
                           ignored_byte_range=[12,16] if ignore else None))
    counts=[_independent_counts(root,header,recording) for root,header,recording in zip((original,repeated),headers,recordings)]
    passed=all(row['equal'] for row in checks) and counts[0]==counts[1]
    result=dict(status='PASS_NATIVE_PREFIX_REPLAY' if passed else 'MISMATCH_NATIVE_PREFIX_REPLAY',
        original_report_sha256=_sha(original/'native_report.json'),repeated_report_sha256=_sha(repeated/'native_report.json'),
        units=headers[0]['units'],duration_ms=recordings[0]['duration_ms'],checks=checks,
        independently_reconstructed_counts=counts,elapsed_seconds=time.perf_counter()-started,
        implementation_sha256=_sha(Path(__file__)),source_event_file_order_preserved=True,
        ignored_difference='saved-network elapsed wall-time float only; all simulation metadata and payload compared',
        native_internal_spike_counter_verified=False,synaptic_payload_semantics_validated=False,
        full_state_restart_available=False,complete_formation_retrieval=False,
        source_fidelity_qualified=False,full_scale_qualified=False,production_episodes=0)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if not passed:raise ValueError('native prefix repeat differs; mismatch artifact retained')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['original','repeated','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(compare(a.original,a.repeated,a.output),indent=2))


if __name__=='__main__':main()

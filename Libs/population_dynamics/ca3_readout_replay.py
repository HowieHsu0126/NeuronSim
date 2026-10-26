"""Independently replay CA3 cue readouts with direct binary reads and Counters."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import struct

import numpy as np

from .provenance import sha256_file
from .carlsim_records import validate_coupled_identity


def replay(runtime, readouts, output, *, coupled_extension=False, source_realization_config=None):
    runtime, readouts, output = map(Path, (runtime, readouts, output))
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((readouts/'report.json').read_text())
    if report['status'] != ('DESCRIBED_COUPLED_EXTENSION_CA3_RETRIEVAL_EVENTS' if coupled_extension
                            else 'DESCRIBED_SOURCE_SCHEDULED_CA3_RETRIEVAL_EVENTS'):
        raise ValueError('completed source readouts required')
    if source_realization_config is not None or 'source_realization' in report:
        from .ca3_realization import bind_runtime

        if not coupled_extension:
            raise ValueError('CA3_REALIZATION_COUPLED_READOUT_REPLAY_REQUIRED')
        native = json.loads((runtime/'native_report.json').read_text())
        bind_runtime(runtime, native, source_realization_config)
        if (report.get('source_realization') != native['source_realization']
                or report.get('realization_circuit_binding') != native.get('realization_circuit_binding')):
            raise ValueError('CA3_REALIZATION_READOUT_NATIVE_IDENTITY_DIFFERS')
    scale = 2 if coupled_extension else 1
    if coupled_extension:
        native = json.loads((runtime/'native_report.json').read_text())
        validate_coupled_identity(native)
        if any(report.get(k) != native[k] for k in ('coupled_extension','coupled_extension_config')):
            raise ValueError('coupled readout and native identity differ')
    for name,digest in report['artifacts'].items():
        if Path(name).name != name or sha256_file(readouts/name) != digest:
            raise ValueError('readout artifact changed')
    for name,digest in report['input_sha256'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha256_file(runtime/relative) != digest:
            raise ValueError('source readout input changed')
    input_name = 'network65p_assem550_divnorm.dat' if coupled_extension else 'network65p_assem275_divnorm.dat'
    with (runtime/'native_run'/input_name).open('rb') as stream:
        signature,version,seconds,_,units,group_count=struct.unpack('<ifffii',stream.read(24))
        if signature != 294338571 or version != float(np.float32(.3)) or units != 84053*scale or group_count != 8:
            raise ValueError('source network header changed')
        groups=[]
        for group_id in range(group_count):
            start,end,nx,ny,nz,raw=struct.unpack('<iiiii100s',stream.read(120))
            name=raw.split(b'\0',1)[0].decode()
            groups.append((name,start,end+1))
    name,start,stop=next(g for g in groups if g[0]=='CA3_Pyramidal')
    if stop-start != 75377*scale or seconds != float(np.float32(45.85)):
        raise ValueError('source Pyramidal identity changed')
    path=runtime/'native_run/results/spk_CA3_Pyramidal.dat'
    bound=next(r for r in report['checked_event_files'] if r['group']==name)
    if sha256_file(path) != bound['sha256']:
        raise ValueError('source event bytes changed')
    with path.open('rb') as stream:
        signature,version,nx,ny,nz=struct.unpack('<ifiii',stream.read(20))
        raw=stream.read()
    if signature != 206661989 or version != float(np.float32(.2)) or nx*ny*nz != stop-start or len(raw)%8:
        raise ValueError('raw event format changed')
    events=list(struct.iter_unpack('<ii',raw))
    if any(t<0 or t>=1001 or u<0 or u>=stop-start for t,u in events):
        raise ValueError('raw event bounds changed')
    full=Counter(u for t,u in events)
    starts=((500,520),(730,750),(960,980));stops=((520,540),(750,770),(980,1000))
    counts=[[Counter(u for t,u in events if lo<=t<hi) for lo,hi in zip(a,b)] for a,b in zip(starts,stops)]
    with np.load(readouts/'pyramidal_window_counts.npz',allow_pickle=False) as saved:
        arrays={k:saved[k] for k in saved.files}
    comparisons=0
    def exact(key, expected):
        nonlocal comparisons
        value=np.asarray(expected)
        if not np.array_equal(arrays[key],value):
            raise ValueError('independent array replay failed: '+key)
        comparisons += value.size
    exact('pyramidal_global_unit_ids',range(start,stop))
    exact('pyramidal_local_unit_ids',range(stop-start))
    exact('cue_ids',[1,2,3]);exact('window_names',['baseline','cue'])
    exact('window_start_ms',starts);exact('window_stop_ms',stops)
    exact('scheduled_train_pattern_indices',[int(x) for x in (runtime/'trainPat.csv').read_text().split()])
    exact('full_record_unit_counts',[full[u] for u in range(stop-start)])
    exact('window_counts',[[[c[u] for u in range(stop-start)] for c in pair] for pair in counts])
    masks={'target_never_boosted_mask':[],'other_never_boosted_mask':[],'eligible_for_boost_mask':[]}
    with (readouts/'descriptive_readouts.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=3:
        raise ValueError('all three cue rows required')
    scalars=0
    assembly, cutoff = 275*scale, 138*scale
    for cue in range(3):
        begin=assembly*cue
        target=set(range(begin+cutoff,begin+assembly))
        other=set().union(*(set(range(assembly*j+cutoff,assembly*j+assembly)) for j in range(3) if j!=cue))
        eligible=set(range(begin,begin+cutoff))
        expected=dict(cue_id=cue+1,assembly_local_start=begin,assembly_global_start=start+begin,
            baseline_start_ms=starts[cue][0],baseline_stop_ms=stops[cue][0],
            cue_start_ms=starts[cue][1],cue_stop_ms=stops[cue][1],following_source_background_ms=(210,210,1)[cue])
        for prefix,members,mask in [('target',target,'target_never_boosted_mask'),
                                   ('other',other,'other_never_boosted_mask'),
                                   ('eligible',eligible,'eligible_for_boost_mask')]:
            masks[mask].append([u in members for u in range(stop-start)])
            n=len(members);a,b=counts[cue]
            na,nb=sum(a[u] for u in members),sum(b[u] for u in members)
            aa,ab=sum(a[u]>0 for u in members),sum(b[u]>0 for u in members)
            values=dict(units=n,baseline_event_count=na,cue_event_count=nb,
                baseline_rate_hz=float(Fraction(na*1000,n*20)),cue_rate_hz=float(Fraction(nb*1000,n*20)),
                baseline_active_units=aa,cue_active_units=ab,baseline_spiking_fraction=aa/n,cue_spiking_fraction=ab/n,
                delta_rate_hz=float(Fraction((nb-na)*1000,n*20)),delta_spiking_fraction=(ab-aa)/n)
            expected.update({prefix+'_'+key:value for key,value in values.items()})
        for measure in ('rate_hz','spiking_fraction'):
            for phase in ('baseline','cue','delta'):
                key=phase+'_'+measure;expected['target_minus_other_'+key]=expected['target_'+key]-expected['other_'+key]
        if set(rows[cue])!=set(expected):
            raise ValueError('readout scalar schema changed')
        for key,value in expected.items():
            actual=float(rows[cue][key]);allowed=1e-12+1e-12*abs(value)
            if not np.isfinite(actual) or abs(actual-value)>allowed:
                raise ValueError('independent scalar replay failed: '+key)
            scalars+=1
    for key,value in masks.items():exact(key,value)
    result=dict(status=('PASS_COUPLED_EXTENSION_RAW_EVENT_READOUT_REPLAY' if coupled_extension
                        else 'PASS_INDEPENDENT_RAW_EVENT_READOUT_REPLAY'),
        source_report_sha256=sha256_file(readouts/'report.json'),raw_events_sha256=bound['sha256'],
        implementation_sha256=sha256_file(Path(__file__)),pyramidal_events=len(events),
        array_values_compared=comparisons,window_count_values_compared=3*2*(stop-start),
        scalar_values_compared=scalars,all_three_cues_retained=True,
        method='struct.iter_unpack raw events; Python Counter/set membership; rational count-rate calculations',
        scalar_tolerance=dict(rtol=1e-12,atol=1e-12),scientific_claim_qualified=False,
        scope='independent recomputation of one source execution; no new source run or biological replicate')
    if coupled_extension:
        result.update(coupled_extension=report['coupled_extension'],
                      coupled_extension_config=report['coupled_extension_config'])
    if source_realization_config is not None:
        result['source_realization'] = report['source_realization']
        result['realization_circuit_binding'] = report['realization_circuit_binding']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','readouts','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--coupled-extension',action='store_true')
    parser.add_argument('--source-realization-config', type=Path)
    args=parser.parse_args();print(json.dumps(replay(args.runtime,args.readouts,args.output,coupled_extension=args.coupled_extension,
                                                   source_realization_config=args.source_realization_config),indent=2))


if __name__=='__main__':main()

"""Replay a spatial source snapshot across MPI ranks under verified fixed drive.

The recorded background program is a numerical diagnostic, never a predictive
input. This runner qualifies transport separately from native Poisson semantics.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .nest_gpu_connectivity import aligned_batches, raw_connect
from .nest_gpu_network import STATE_FIELDS, prepare, ticks
from .network_snapshot import load_initial_network
from .provenance import sha256_file


def partition(arrays, plan, distributed):
    """Cut the source periodic square at x=0, retaining every source identity."""
    ids = arrays['unit_ids']
    ranks = (arrays['positions_mm'][:, 0] >= 0).astype(int) if distributed else np.zeros(len(ids), dtype=int)
    owners = dict(zip(map(int, ids), map(int, ranks)))
    maps, groups = [], []
    for rank in range(2 if distributed else 1):
        cortex = np.asarray([u for u in plan['cortex'] if owners[int(u)] == rank], dtype=np.int64)
        tc = np.asarray([u for u in plan['tc'] if owners[int(u)] == rank], dtype=np.int64)
        if not len(cortex):
            raise ValueError('each partition must contain cortical neurons')
        ordered = np.concatenate([cortex, tc])
        maps.append({int(u): i for i, u in enumerate(ordered)})
        groups.append((cortex, tc))
    return owners, maps, groups


def load_drive(directory, snapshot, metadata, plan):
    report = json.loads((directory/'drive.json').read_text())
    if (report.get('schema') != 'mesocircuit_background_arrivals_v1' or report.get('status') != 'PASS'
            or report.get('counts_decoding_validated') is not True
            or report.get('spike_capture_complete_against_native_counters') is not True
            or report.get('snapshot_manifest_sha256') != sha256_file(snapshot/'manifest.json')
            or report.get('counts_sha256') != sha256_file(directory/'counts.npz')):
        raise ValueError('verified complete background from this source snapshot required')
    with np.load(directory/'counts.npz', allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    n = int(round(metadata['duration_ms']/metadata['dt_ms']))
    counts = data['counts']
    if (not np.array_equal(data['unit_ids'], plan['cortex'])
            or not np.array_equal(data['arrival_ticks'], np.arange(n))
            or float(data['dt_ms']) != metadata['dt_ms'] or counts.shape != (n,len(plan['cortex']))
            or counts.dtype.kind not in 'iu' or np.any(counts < 0)
            or np.any(counts > 2**24)):
        raise ValueError('background identity, clock, count shape or exact FP32 multiplicity invalid')
    return data


def background_program(data, background_edges, cortex, dt):
    edges = background_edges.copy(); edges.sort(order='target')
    if len(edges) != len(cortex) or not np.array_equal(edges['target'], cortex):
        raise ValueError('one source background edge per cortical target required')
    programs = {}
    for index, uid in enumerate(cortex):
        arrival = np.flatnonzero(data['counts'][:, index])
        delay = int(ticks([edges['delay'][index]], dt)[0])
        emission = arrival-delay
        if np.any(emission < 1):
            raise ValueError('background emission must be strictly positive')
        programs[int(uid)] = dict(emission_ticks=emission,
            multiplicities=data['counts'][arrival,index], weight=float(edges['weight'][index]), delay_ticks=delay)
    return programs


def unique_source_batches(rows):
    """Keep native RemoteConnect source lists unique while retaining all edges.

    The pinned remote mapping kernels initialize only the first sorted occurrence
    of a source ID. Repeated source IDs therefore require separate native calls.
    A neuron's multiple synapses remain distinct, including repeated targets.
    """
    if not len(rows):
        return
    order = np.argsort(rows['source'], kind='stable')
    _, starts, counts = np.unique(rows['source'][order], return_index=True, return_counts=True)
    for occurrence in range(int(counts.max())):
        yield rows[order[starts[counts > occurrence] + occurrence]]


def execute(snapshot, drive, output, schedule, distributed, omit_cross, batch_size):
    if batch_size < 1:
        raise ValueError('positive edge batch size required')
    if distributed:
        local_rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if len(visible)!=2 or local_rank not in (0,1):
            raise ValueError('two scheduler-visible devices and two local MPI ranks required')
        os.environ['CUDA_VISIBLE_DEVICES']=visible[local_rank]
    ng=importlib.import_module('nestgpu')
    if distributed: ng.ConnectMpiInit()
    rank=ng.HostId() if distributed else 0
    hosts=ng.HostNum() if distributed else 1
    output.mkdir(parents=True,exist_ok=True)
    path=output/f'rank_{rank}.json'
    if path.exists():
        raise ValueError('fresh rank output required')
    start=time.perf_counter()
    report=dict(status='RUNNING',rank=rank,hosts=hosts,distributed=distributed,
        omit_cross=omit_cross,schedule=schedule,edge_batch_size=batch_size,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        native_library_sha256=sha256_file(Path(os.environ['NESTGPU_LIB'])),
        snapshot_manifest_sha256=sha256_file(snapshot/'manifest.json'),
        drive_report_sha256=sha256_file(drive/'drive.json'),
        scientific_reference_qualified=False,native_poisson_qualified=False,checks={})
    try:
        if distributed and hosts!=2: raise ValueError('exactly two MPI ranks required')
        metadata, arrays=load_initial_network(snapshot)
        plan=prepare(metadata,arrays,schedule)
        data=load_drive(drive,snapshot,metadata,plan)
        dt=metadata['dt_ms']; duration=metadata['duration_ms']
        owners,maps,groups=partition(arrays,plan,distributed)
        local_cortex,local_tc=groups[rank]
        source_indices=np.searchsorted(plan['cortex'],local_cortex)
        ng.SetRandomSeed(81201);ng.SetTimeResolution(dt);ng.SetKernelStatus('verbosity_level',0)
        cells=ng.Create('iaf_psc_exp',len(local_cortex))
        for name,values in plan['mapped_parameters'].items():
            ng.SetStatus(cells,name,{'array':values[source_indices].tolist()})
        initial={name:np.asarray(ng.GetNeuronStatus(cells,name)).reshape(-1) for name in plan['mapped_parameters']}
        report['checks']['initial_parameters_exact_fp32']=all(np.array_equal(initial[k],v[source_indices].astype(np.float32))
            for k,v in plan['mapped_parameters'].items())
        tc_nodes=ng.Create('spike_generator',len(local_tc)) if len(local_tc) else None
        for uid in local_tc:
            ng.SetStatus([maps[rank][int(uid)]],{'spike_times':((plan['tc_schedules'][int(uid)]-1)*dt).tolist()})
        capacity=math.ceil(duration/np.min(plan['mapped_parameters']['t_ref']))+2
        ng.ActivateSpikeCount(cells);ng.ActivateRecSpikeTimes(cells,capacity)
        if tc_nodes is not None:
            ng.ActivateSpikeCount(tc_nodes);ng.ActivateRecSpikeTimes(tc_nodes,
                max(len(v) for v in plan['tc_schedules'].values())+2)
        bg_ids=[d['id'] for d in plan['background']]
        bg_edges=plan['edges'][np.isin(plan['edges']['source'],bg_ids)]
        programs=background_program(data,bg_edges,plan['cortex'],dt)
        bg_nodes=ng.Create('spike_generator',len(local_cortex))
        ng.ActivateSpikeCount(bg_nodes)
        ng.ActivateRecSpikeTimes(bg_nodes,max(len(p['emission_ticks']) for p in programs.values())+2)
        program_checks=[]
        for j,uid in enumerate(local_cortex):
            p=programs[int(uid)]
            ng.SetStatus([int(bg_nodes[j])],{'spike_times':((p['emission_ticks']-1)*dt).tolist(),
                'spike_gen_mul':p['multiplicities'].astype(float).tolist()})
            read=ng.GetStatus([int(bg_nodes[j])])[0]
            program_checks.append(np.array_equal(np.asarray(read['spike_gen_mul']),p['multiplicities']))
        report['checks']['background_multiplicities_readback_exact']=all(program_checks)
        counts=[0]*hosts;incoming=[];cross_edges=0
        edges=plan['edges'][~np.isin(plan['edges']['source'],bg_ids)]
        # All ranks traverse the same ordered declarations. Counts belong to the
        # receiving rank; native array blocks are target-local, not source-local.
        for sr in range(hosts):
            for tr in range(hosts):
                take=np.asarray([owners[int(s)]==sr and owners[int(t)]==tr for s,t in zip(edges['source'],edges['target'])])
                subset=edges[take]
                if sr!=tr:
                    cross_edges+=len(subset)
                    if omit_cross: continue
                for port in (0,1):
                    rows=subset[(subset['weight']<0)==bool(port)]
                    for group in unique_source_batches(rows) if sr!=tr else [rows]:
                        for lo,hi in aligned_batches(counts[tr],len(group),maximum_batch_size=batch_size):
                            batch=group[lo:hi]
                            source=[maps[sr][int(u)] for u in batch['source']]
                            target=[maps[tr][int(u)] for u in batch['target']]
                            weight=batch['weight'].astype(np.float32);delay=batch['delay'].astype(np.float32)
                            if sr==tr:
                                if rank==tr: raw_connect(ng,source,target,weight.tolist(),delay.tolist(),port)
                            else:
                                ng.RemoteConnect(sr,source,tr,target,{'rule':'one_to_one'},
                                    {'weight_array':weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     'delay_array':delay.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),'receptor':port})
                            if rank==tr:
                                incoming.extend((int(t),int(s),port,int(ticks([d],dt)[0]),float(np.float32(w)))
                                    for s,t,w,d in zip(batch['source'],batch['target'],batch['weight'],batch['delay']))
                        counts[tr]+=len(group)
        for tr,(cortex_ids,tc_ids) in enumerate(groups):
            weights=[programs[int(u)]['weight'] for u in cortex_ids]
            delays=[programs[int(u)]['delay_ticks']*dt for u in cortex_ids]
            for lo,hi in aligned_batches(counts[tr],len(cortex_ids),maximum_batch_size=batch_size):
                if rank==tr:
                    raw_connect(ng,[len(cortex_ids)+len(tc_ids)+j for j in range(lo,hi)],
                        [maps[tr][int(u)] for u in cortex_ids[lo:hi]],weights[lo:hi],delays[lo:hi],0)
            if rank==tr:
                incoming.extend((int(u),-int(u),0,programs[int(u)]['delay_ticks'],float(np.float32(programs[int(u)]['weight'])))
                    for u in cortex_ids)
            counts[tr]+=len(cortex_ids)
        rec=ng.CreateRecord('',[k for _ in local_cortex for k in STATE_FIELDS],
            [maps[rank][int(u)] for u in local_cortex for _ in STATE_FIELDS],[0]*(len(local_cortex)*len(STATE_FIELDS)))
        ng.Calibrate()
        native_ids=[]
        for begin in range(0,len(local_cortex),32):
            native_ids.extend(ng.GetConnections(target=cells[begin:min(begin+32,len(local_cortex))]).ToList())
        report['checks']['native_incoming_connection_count_exact']=(
            len(native_ids)==counts[rank] and len(set(native_ids))==len(native_ids))
        report['native_enumerated_connections']=len(native_ids)
        raw=ng.GetConnectionStatus(native_ids)
        expected=sorted((maps[rank][t],port,delay,weight) for t,s,port,delay,weight in incoming)
        actual=sorted((e['target'],e['port'],int(ticks([e['delay']],dt,recorded=True)[0]),e['weight']) for e in raw)
        report['checks']['incoming_target_attributes_exact']=actual==expected
        report.update(cortical_units=len(local_cortex),tc_units=len(local_tc),incoming_edges=counts[rank],
            original_cross_edges=cross_edges,native_edges=raw,
            source_identity_readback='raw remote image IDs retained; target multiset alone is not source-ID certification',
            source_unit_count=len(arrays['unit_ids']),background_events=int(data['counts'].sum()))
        if not all(report['checks'].values()):raise ValueError('native import validation failed')
        path.write_text(json.dumps(report,indent=2)+'\n')
        ng.Simulate(duration)
        cell_trains=list(ng.GetRecSpikeTimes(cells))
        tc_trains=list(ng.GetRecSpikeTimes(tc_nodes)) if tc_nodes is not None else []
        bg_trains=list(ng.GetRecSpikeTimes(bg_nodes))
        trains=cell_trains+tc_trains;unit_ids=np.concatenate([local_cortex,local_tc])
        native_counts=list(np.asarray(ng.GetNeuronStatus(cells,'spike_count')).reshape(-1))
        if tc_nodes is not None:native_counts+=list(np.asarray(ng.GetNeuronStatus(tc_nodes,'spike_count')).reshape(-1))
        report['checks']['spike_counters_exact']=np.array_equal(native_counts,[len(t) for t in trains])
        traffic=0;active_sources=set()
        local_trains={int(u):ticks(t,dt,recorded=True) for u,t in zip(local_cortex,cell_trains)}
        if distributed and not omit_cross:
            for edge in edges:
                source,target=int(edge['source']),int(edge['target'])
                if source in local_trains and owners[target]!=rank:
                    delay=int(ticks([edge['delay']],dt)[0])
                    number=int(np.searchsorted(local_trains[source],round(duration/dt)-delay,side='left'))
                    traffic+=number
                    if number:active_sources.add(source)
        report['expected_in_window_cortical_cross_arrivals']=traffic
        report['active_cortical_cross_sources']=len(active_sources)
        report['checks']['background_emission_ticks_exact']=all(np.array_equal(ticks(t,dt,recorded=True),programs[int(u)]['emission_ticks'])
            for u,t in zip(local_cortex,bg_trains))
        report['checks']['background_counters_exact']=np.array_equal(
            np.asarray(ng.GetNeuronStatus(bg_nodes,'spike_count')).reshape(-1),[len(t) for t in bg_trains])
        report['checks']['tc_emission_ticks_exact']=all(np.array_equal(ticks(t,dt,recorded=True),
            plan['tc_schedules'][int(u)][plan['tc_schedules'][int(u)]*dt<=duration]) for u,t in zip(local_tc,tc_trains))
        event_ids=np.concatenate([np.full(len(t),u,dtype=np.int64) for u,t in zip(unit_ids,trains)])
        raw_times=np.concatenate([np.asarray(t,dtype=float) for t in trains]);et=ticks(raw_times,dt,recorded=True)
        order=np.lexsort((event_ids,et));record=np.asarray(ng.GetRecordData(rec),dtype=float)
        clock=ticks(record[:,0],dt,recorded=True)
        report['checks']['complete_finite_state_clock']=np.array_equal(clock,np.arange(round(duration/dt)+1)) and bool(np.isfinite(record).all())
        np.savez_compressed(output/f'rank_{rank}.npz',unit_ids=local_cortex,recorded_unit_ids=unit_ids,
            event_ids=event_ids[order],event_ticks=et[order],raw_times_ms=raw_times[order],ticks=clock,
            states=record[:,1:].reshape(len(clock),len(local_cortex),len(STATE_FIELDS)),
            native_counts=np.asarray(native_counts,dtype=np.int64),fields=np.asarray(STATE_FIELDS))
        report.update(status='EXECUTED' if all(report['checks'].values()) else 'FAILED',spikes=len(event_ids),
            cortical_spikes=sum(map(len,cell_trains)),arrays_sha256=sha256_file(output/f'rank_{rank}.npz'))
    except Exception as error:
        report.update(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report['implementation_sha256']=sha256_file(Path(__file__))
    report['seconds']=time.perf_counter()-start
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ['rank','status','spikes','seconds','error']}),flush=True)
    if distributed:
        if report['status']!='EXECUTED':os._exit(1)
        ng.MpiFinalize()
    return int(report['status']!='EXECUTED')


def compare(reference, distributed, output):
    paths=[reference/'rank_0.json',distributed/'rank_0.json',distributed/'rank_1.json']
    reports=[json.loads(p.read_text()) for p in paths];arrays=[]
    for path,report in zip(paths,reports):
        npz=path.with_suffix('.npz')
        if report['status']!='EXECUTED' or sha256_file(npz)!=report['arrays_sha256']:
            raise ValueError('executed immutable rank arrays required')
        with np.load(npz,allow_pickle=False) as f:arrays.append({k:f[k] for k in f.files})
    ref=arrays[0];parts=arrays[1:]
    ids=np.concatenate([a['unit_ids'] for a in parts]);unit_order=np.argsort(ids)
    events=np.concatenate([a['event_ids'] for a in parts]);ets=np.concatenate([a['event_ticks'] for a in parts]);order=np.lexsort((events,ets))
    states=np.concatenate([a['states'] for a in parts],axis=1)[:,unit_order]
    delta=np.max(np.abs(states-ref['states']),axis=(0,1))
    checks=dict(reference_single_rank=reports[0]['hosts']==1 and not reports[0]['distributed'],
        coupled_conditions=all(not r['omit_cross'] for r in reports),
        cortical_cross_traffic_both_directions=all(r.get('expected_in_window_cortical_cross_arrivals',0)>0 for r in reports[1:]),
        native_edge_counts_verified=all(r['checks'].get('native_incoming_connection_count_exact',False) for r in reports),
        incoming_edge_counts_equal=sum(r['incoming_edges'] for r in reports[1:])==reports[0]['incoming_edges'],
        all_recorded_units_exact=np.array_equal(np.sort(np.concatenate([a['recorded_unit_ids'] for a in parts])),np.sort(ref['recorded_unit_ids'])),
        source_snapshot_same=len({r['snapshot_manifest_sha256'] for r in reports})==1,
        fixed_background_same=len({r['drive_report_sha256'] for r in reports})==1,
        schedule_same=len({r['schedule'] for r in reports})==1,
        native_library_same=len({r['native_library_sha256'] for r in reports})==1,
        two_devices=len({r['cuda_visible_devices'] for r in reports[1:]})==2,
        two_ranks=all(r['hosts']==2 and r['rank']==i for i,r in enumerate(reports[1:])),
        all_cortical_units_exact=np.array_equal(ids[unit_order],ref['unit_ids']),
        spike_ticks_and_identities_exact=np.array_equal(events[order],ref['event_ids']) and np.array_equal(ets[order],ref['event_ticks']),
        state_clocks_exact=all(np.array_equal(a['ticks'],ref['ticks']) for a in parts),
        voltage_within_existing_bound=bool(delta[0]<=.0005),
        currents_within_existing_bound=bool(np.max(delta[1:3])<=.005),refractory_exact=bool(delta[3]==0))
    result=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks,
        compared_cortical_units=len(ids),compared_state_values=int(ref['states'].size),reference_spikes=len(ref['event_ids']),
        state_max_absolute_delta=dict(zip(STATE_FIELDS,delta.tolist())),
        cortical_cross_arrivals_by_rank=[r['expected_in_window_cortical_cross_arrivals'] for r in reports[1:]],
        source_identity_scope='whole-network dynamics and incoming target attributes; raw remote-image IDs are not decoded',
        native_poisson_qualified=False,scientific_reference_qualified=False)
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--drive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--schedule',choices=('initial','effective'),default='effective')
    parser.add_argument('--distributed',action='store_true')
    parser.add_argument('--omit-cross',action='store_true')
    parser.add_argument('--edge-batch-size',type=int,default=4096)
    args=parser.parse_args()
    raise SystemExit(execute(args.snapshot,args.drive,args.output,args.schedule,args.distributed,args.omit_cross,args.edge_batch_size))


if __name__=='__main__':main()

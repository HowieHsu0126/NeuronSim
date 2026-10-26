"""Native regression for remote source identity with repeated presynaptic IDs.

Equal weights/delays and distinct emission patterns isolate identity routing.
Cold and primed remote maps exercise both native map insertion paths.
"""
from __future__ import annotations
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from .nest_gpu_network import ticks
from .nest_gpu_mpi_network import unique_source_batches
from .provenance import sha256_file

DT=.1
DURATION=8.
WEIGHT=1024.
DELAY=1.
SOURCE_TICKS={0:np.array([10,30]),1:np.array([20,40])}
EDGE_DTYPE=np.dtype([('source','i8'),('target','i8')])


def execute(output, mode, primed):
    distributed=mode!='reference'
    if distributed:
        rank=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']);visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if len(visible)!=2 or rank not in (0,1):raise ValueError('exactly two allocated devices required')
        os.environ['CUDA_VISIBLE_DEVICES']=visible[rank]
    ng=importlib.import_module('nestgpu')
    if distributed:ng.ConnectMpiInit()
    rank=ng.HostId() if distributed else 0
    output.mkdir(parents=True,exist_ok=True);path=output/f'rank_{rank}.json'
    if path.exists():raise ValueError('fresh rank output required')
    report=dict(status='RUNNING',mode=mode,primed=primed,rank=rank,hosts=ng.HostNum() if distributed else 1,
        native_library_sha256=sha256_file(Path(os.environ['NESTGPU_LIB'])),scientific_reference_qualified=False)
    try:
        ng.SetTimeResolution(DT);ng.SetKernelStatus('verbosity_level',0)
        target_host=not distributed or rank==1
        cells=ng.Create('iaf_psc_exp',6 if target_host else 1)
        ng.SetStatus(cells,dict(C_m=250.,tau_m=10.,E_L=-65.,t_ref=2.,I_e=0.,V_m_rel=0.,Theta_rel=15.,
            V_reset_rel=0.,tau_ex=.5,tau_in=.5,I_syn_ex=0.,I_syn_in=0.,den_delay=0.))
        source_host=not distributed or rank==0
        generators=ng.Create('spike_generator',2) if source_host else None
        if source_host:
            for i in (0,1):ng.SetStatus([int(generators[i])],{'spike_times':((SOURCE_TICKS[i]-1)*DT).tolist()})
            ng.ActivateSpikeCount(generators);ng.ActivateRecSpikeTimes(generators,4)
        def connect(rows):
            source=[int(s)+ (1 if distributed else 6) for s in rows['source']]
            target=list(map(int,rows['target']))
            syn={'weight_array':(ctypes.c_float*len(rows))(*([WEIGHT]*len(rows))),
                 'delay_array':(ctypes.c_float*len(rows))(*([DELAY]*len(rows))),'receptor':0}
            if distributed:ng.RemoteConnect(0,source,1,target,{'rule':'one_to_one'},syn)
            else:ng.Connect(source,target,{'rule':'one_to_one'},syn)
        if primed:connect(np.array([(0,4),(1,5)],dtype=EDGE_DTYPE))
        edges=np.array([(0,0),(1,1),(0,2),(1,3)],dtype=EDGE_DTYPE)
        for rows in unique_source_batches(edges) if mode=='unique' else [edges]:connect(rows)
        rec=ng.CreateRecord('',['I_syn_ex']*6,list(range(6)),[0]*6) if target_host else None
        ng.Calibrate()
        propagator=np.asarray(ng.GetNeuronStatus(cells,'P11ex')).reshape(-1) if target_host else None
        ng.Simulate(DURATION)
        if source_host:
            trains=ng.GetRecSpikeTimes(generators)
            report['source_ticks']=[ticks(t,DT,recorded=True).tolist() for t in trains]
            report['source_ticks_exact']=all(np.array_equal(ticks(t,DT,recorded=True),SOURCE_TICKS[i]) for i,t in enumerate(trains))
            report['source_counters_exact']=np.array_equal(np.asarray(ng.GetNeuronStatus(generators,'spike_count')).reshape(-1),[2,2])
        if target_host:
            raw=np.asarray(ng.GetRecordData(rec));clock=ticks(raw[:,0],DT,recorded=True);currents=raw[:,1:]
            # Arrivals enter the port before iaf_psc_exp decays that current.
            impulse=currents[1:]/propagator[None,:]-currents[:-1]
            expected=np.zeros_like(impulse)
            for source,target in [*edges.tolist(),*([(0,4),(1,5)] if primed else [])]:
                expected[SOURCE_TICKS[source]+int(round(DELAY/DT)),target]+=WEIGHT
            error=np.abs(impulse-expected)
            report.update(current_impulse_max_error_pa=float(error.max()),routing_matches_declared_sources=bool(error.max()<=.005),
                          state_clock_complete=np.array_equal(clock,np.arange(round(DURATION/DT)+1)))
            np.savez_compressed(output/'target_currents.npz',ticks=clock,currents=currents,propagator=propagator,
                observed_impulses=impulse,expected_impulses=expected)
            report['target_arrays_sha256']=sha256_file(output/'target_currents.npz')
        report['status']='EXECUTED'
    except Exception as error:
        report.update(status='FAILED',error=str(error),traceback=traceback.format_exc())
    report['implementation_sha256']=sha256_file(Path(__file__))
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if distributed:
        if report['status']!='EXECUTED':os._exit(1)
        ng.MpiFinalize()
    return int(report['status']!='EXECUTED')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['reference','bulk','unique'],required=True);p.add_argument('--primed',action='store_true')
    a=p.parse_args();raise SystemExit(execute(a.output,a.mode,a.primed))


if __name__=='__main__':main()

"""Deterministic bidirectional coupling probe for the pinned MPI runtime.

Eight neurons occupy two logical partitions. Native arrays, counters and edge
readbacks are exported before comparison with a single-device execution.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .nest_gpu_network import STATE_FIELDS, ticks
from .provenance import sha256_file

DT_MS = .1
DURATION_MS = 20.
# Semantic IDs: cortex0..7, pulse generators8/9. Every weight differs so native
# readback can identify target/delay/receptor independently of remote image IDs.
EDGES = [
    (8,0,21001.,.5), (9,4,21002.,.7),
    (0,1,13001.,.6), (4,5,13002.,.8),
    (0,4,14001.,1.), (0,4,1001.,1.2),
    (4,0,14002.,1.3), (4,0,1002.,1.5),
    (1,6,15001.,1.1), (5,2,15002.,.9),
    (2,3,16001.,.4), (6,7,16002.,.6),
    (3,4,-5001.,.8), (7,0,-5002.,1.4),
]
PULSES = {8:[1.,5.,9.], 9:[2.,6.,10.]}


def owner(uid):
    return (uid // 4) if uid < 8 else uid-8


def local_id(uid):
    return uid % 4 if uid < 8 else 4


def execute(output: Path, distributed: bool, omit_cross: bool):
    if distributed:
        local_rank=int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        visible=os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible)!=2 or local_rank not in (0,1):
            raise ValueError("two scheduler-visible devices and two local MPI ranks required")
        os.environ["CUDA_VISIBLE_DEVICES"]=visible[local_rank]
    ng=importlib.import_module("nestgpu")
    if distributed:ng.ConnectMpiInit()
    rank=ng.HostId() if distributed else 0
    hosts=ng.HostNum() if distributed else 1
    output.mkdir(parents=True,exist_ok=True)
    report=dict(status="RUNNING",rank=rank,hosts=hosts,distributed=distributed,omit_cross=omit_cross,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        native_library_sha256=sha256_file(Path(os.environ["NESTGPU_LIB"])),
        scientific_reference_qualified=False,started=time.time())
    path=output/f"rank_{rank}.json"
    try:
        if distributed and hosts != 2:raise ValueError("probe requires exactly two MPI ranks")
        ng.SetTimeResolution(DT_MS);ng.SetRandomSeed(81201);ng.SetKernelStatus("verbosity_level",0)
        cells=ng.Create("iaf_psc_exp",4 if distributed else 8)
        pulse=ng.Create("spike_generator",1 if distributed else 2)
        semantic=np.arange(rank*4,(rank+1)*4) if distributed else np.arange(8)
        mapped=dict(C_m=np.full(len(semantic),250.),tau_m=np.full(len(semantic),10.),
                    E_L=np.full(len(semantic),-65.),t_ref=np.full(len(semantic),2.),I_e=np.zeros(len(semantic)),
                    V_m_rel=semantic.astype(float),Theta_rel=np.full(len(semantic),15.),
                    V_reset_rel=np.zeros(len(semantic)),tau_ex=np.full(len(semantic),.5),
                    tau_in=np.full(len(semantic),.5),I_syn_ex=np.zeros(len(semantic)),
                    I_syn_in=np.zeros(len(semantic)),den_delay=np.zeros(len(semantic)))
        for name,values in mapped.items():ng.SetStatus(cells,name,{"array":values.tolist()})
        for logical in ([rank] if distributed else [0,1]):
            native=int(pulse[0] if distributed else pulse[logical])
            ng.SetStatus([native],{"spike_times":(np.asarray(PULSES[8+logical])-DT_MS).tolist()})
        ng.ActivateSpikeCount(cells);ng.ActivateRecSpikeTimes(cells,20)
        ng.ActivateSpikeCount(pulse);ng.ActivateRecSpikeTimes(pulse,10)
        incoming=[]
        for source,target,weight,delay in EDGES:
            cross=owner(source)!=owner(target)
            if omit_cross and cross:continue
            receptor=int(weight<0)
            syn=dict(weight_array=(ctypes.c_float*1)(weight),delay_array=(ctypes.c_float*1)(delay),receptor=receptor)
            if distributed:
                if cross:
                    ng.RemoteConnect(owner(source),[local_id(source)],owner(target),[local_id(target)],
                                     {"rule":"one_to_one"},syn)
                elif owner(source)==rank:
                    ng.Connect([local_id(source)],[local_id(target)],{"rule":"one_to_one"},syn)
                if owner(target)==rank:incoming.append((source,target,weight,delay))
            else:
                ng.Connect([source],[target],{"rule":"one_to_one"},syn)
                incoming.append((source,target,weight,delay))
        rec=ng.CreateRecord("",[k for _ in semantic for k in STATE_FIELDS],
            [int(cells[i]) for i in range(len(semantic)) for _ in STATE_FIELDS],[0]*(len(semantic)*len(STATE_FIELDS)))
        ng.Calibrate()
        raw=ng.GetConnectionStatus(list(range(len(incoming))))
        expected=sorted((local_id(t) if distributed else t,int(w<0),int(round(d/DT_MS)),float(np.float32(w)))
                        for s,t,w,d in incoming)
        actual=sorted((e["target"],e["port"],int(ticks([e["delay"]],DT_MS,recorded=True)[0]),e["weight"]) for e in raw)
        report.update(native_edges=raw,expected_target_attributes=expected,
            target_attribute_multiset_exact=actual==expected,created_incoming_edges=len(incoming),
            source_id_readback="raw native source/image IDs retained; weight tags identify expected edges but do not prove remote source identity")
        if not report["target_attribute_multiset_exact"]:raise ValueError("native incoming attributes mismatch")
        path.write_text(json.dumps(report,indent=2))
        ng.Simulate(DURATION_MS)
        trains=ng.GetRecSpikeTimes(cells);pulse_trains=ng.GetRecSpikeTimes(pulse)
        counts=np.asarray(ng.GetNeuronStatus(cells,"spike_count")).reshape(-1).astype(np.int64)
        report["counters_exact"]=np.array_equal(counts,[len(t) for t in trains])
        ids=np.concatenate([np.full(len(t),u,dtype=np.int64) for u,t in zip(semantic,trains)])
        times=np.concatenate([np.asarray(t,dtype=float) for t in trains]);event_ticks=ticks(times,DT_MS,recorded=True)
        order=np.lexsort((ids,event_ticks));states=np.asarray(ng.GetRecordData(rec),dtype=float)
        np.savez(output/f"rank_{rank}.npz",unit_ids=semantic,event_ids=ids[order],event_ticks=event_ticks[order],
            raw_times_ms=times[order],ticks=ticks(states[:,0],DT_MS,recorded=True),
            states=states[:,1:].reshape(len(states),len(semantic),len(STATE_FIELDS)),native_counts=counts,
            fields=np.asarray(STATE_FIELDS))
        report.update(status="EXECUTED" if report["counters_exact"] else "FAILED",spikes=len(ids),
                      pulse_trains=pulse_trains,seconds=time.time()-report["started"])
    except Exception as error:
        report.update(status="FAILED",error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report.get(k) for k in ("rank","status","spikes","error")}),flush=True)
    if distributed:
        if report["status"]!="EXECUTED":
            # A peer may be inside a collective Simulate. A nonzero rank exit
            # lets the launcher stop the group instead of hanging in Finalize.
            os._exit(1)
        ng.MpiFinalize()
    return int(report["status"]!="EXECUTED")


def compare(reference: Path, distributed: Path, negative: Path, output: Path):
    source=json.loads((reference/"rank_0.json").read_text())
    ranks=[json.loads((distributed/f"rank_{r}.json").read_text()) for r in (0,1)]
    controls=[json.loads((negative/f"rank_{r}.json").read_text()) for r in (0,1)]
    checks={"all_executed":all(r["status"]=="EXECUTED" for r in [source,*ranks,*controls]),
            "two_distinct_devices":len({r["cuda_visible_devices"] for r in ranks})==2,
            "two_real_mpi_ranks":all(r["hosts"]==2 and r["rank"]==i for i,r in enumerate(ranks))}
    with np.load(reference/"rank_0.npz") as x: ref={k:x[k] for k in x.files}
    arrays=[];control_arrays=[]
    for rank in (0,1):
        with np.load(distributed/f"rank_{rank}.npz") as x:arrays.append({k:x[k] for k in x.files})
        with np.load(negative/f"rank_{rank}.npz") as x:control_arrays.append({k:x[k] for k in x.files})
    ids=np.concatenate([a["unit_ids"] for a in arrays]);events=np.concatenate([a["event_ids"] for a in arrays])
    event_ticks=np.concatenate([a["event_ticks"] for a in arrays]);order=np.lexsort((events,event_ticks))
    checks["unit_ids_exact"]=np.array_equal(ids,ref["unit_ids"])
    checks["all_spikes_exact"]=np.array_equal(events[order],ref["event_ids"]) and np.array_equal(event_ticks[order],ref["event_ticks"])
    checks["state_clock_exact"]=all(np.array_equal(a["ticks"],ref["ticks"]) for a in arrays)
    states=np.concatenate([a["states"] for a in arrays],axis=1)
    maxima=np.max(np.abs(states-ref["states"]),axis=(0,1))
    checks["voltage_within_existing_bound"]=bool(maxima[0] <= .0005)
    checks["currents_within_existing_bound"]=bool(max(maxima[1:3]) <= .005)
    checks["refractory_exact"]=bool(maxima[3]==0)
    for rank,(a,c) in enumerate(zip(arrays,control_arrays)):
        recipient=2+rank*4
        checks[f"rank_{rank}_cross_only_recipient_fires"]=bool(np.any(a["event_ids"]==recipient))
        checks[f"rank_{rank}_omission_silences_recipient"]=bool(not np.any(c["event_ids"]==recipient))
        checks[f"rank_{rank}_cross_inhibition_arrives"]=bool(np.any(a["states"][:,:,2]<0))
        checks[f"rank_{rank}_omission_removes_inhibition"]=bool(np.all(c["states"][:,:,2]==0))
    for uid in sorted({s for s,t,w,d in EDGES if owner(s)!=owner(t)}):
        checks[f"cross_source_{uid}_emits"]=bool(np.any(ref["event_ids"]==uid))
    result=dict(status="PASS" if all(checks.values()) else "FAILED",checks=checks,
        state_max_absolute_delta=dict(zip(STATE_FIELDS,maxima.tolist())),
        compared_state_values=int(ref["states"].size),compared_units=len(ids),
        source_library_sha256=source["native_library_sha256"],
        distributed_library_sha256=ranks[0]["native_library_sha256"],
        scope="deterministic eight-neuron two-partition circuit, not large-scale or Poisson qualification",
        scientific_reference_qualified=False)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--distributed",action="store_true")
    parser.add_argument("--omit-cross",action="store_true")
    args=parser.parse_args()
    raise SystemExit(execute(args.output,args.distributed,args.omit_cross))


if __name__=="__main__":main()

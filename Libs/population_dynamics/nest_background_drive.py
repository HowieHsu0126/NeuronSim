"""Recover source-indexed background arrivals from complete native GPU traces.

The observed PSC is separated into source-known recurrent/TC arrivals and an
integer number of background arrivals. Outward floating-point intervals must
identify a unique nonnegative integer; no unconditional rounding is permitted.
This is a bounded reference diagnostic, not a population generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from .network_snapshot import load_initial_network
from .provenance import sha256_file

SCHEMA = "mesocircuit_background_arrivals_v1"
F32_U = 2.0 ** -24
F32_TINY = 2.0 ** -126
F64_U = 2.0 ** -53


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def arrival_intervals(currents, decay, known, event_counts):
    """Bound unrecorded positive input after known same-port arrivals.

    The pinned kernel decays a FP32 current once per step; all arrivals are
    positive after choosing the sign of the receptor. N known event products,
    their accumulation, one unknown background product and the current update
    need at most N+5 rounding stages. This deliberately bounds FP64 accumulation
    by the weaker FP32 error model. All additions are same-sign, so the bound
    applies to their sum. The independently summed known inputs use FP64 bounds.
    """
    currents, decay, known = (np.asarray(x, dtype=float) for x in (currents, decay, known))
    event_counts = np.asarray(event_counts)
    if (currents.ndim != 2 or len(currents) < 2 or known.shape != currents[:-1].shape
            or event_counts.shape != known.shape or decay.shape != (currents.shape[1],)):
        raise ValueError("incompatible current, propagator or arrival shapes")
    if (not np.isfinite(currents).all() or np.any(currents < 0)
            or not np.isfinite(known).all() or np.any(known < 0)
            or not np.isfinite(decay).all() or np.any(decay < F32_TINY) or np.any(decay > 1)
            or event_counts.dtype.kind not in "iu" or np.any(event_counts < 0)
            or not np.array_equal(currents,currents.astype(np.float32).astype(float))
            or not np.array_equal(decay,decay.astype(np.float32).astype(float))):
        raise ValueError("finite nonnegative FP32 states, normal propagators and integer arrival counts required")
    previous, current = currents[:-1], currents[1:]
    accumulated_lo = np.maximum(0., _down(_down(current-F32_TINY)/_up(decay*(1+F32_U))))
    accumulated_hi = _up(_up(current+F32_TINY)/_down(decay*(1-F32_U)))
    operations = event_counts.astype(float)+5.
    nu = _up(operations*F32_U)
    gamma = _up(nu/_down(1-nu))
    if np.any(gamma >= 1): raise ValueError("current error bound is not invertible")
    additive = _up(operations*F32_TINY/_down(1-nu))
    total_lo = _down(_down(accumulated_lo-additive)/_up(1+gamma))
    total_hi = _up(_up(accumulated_hi+additive)/_down(1-gamma))
    nu64 = _up((event_counts.astype(float)+1)*F64_U)
    gamma64 = _up(nu64/_down(1-nu64))
    known_lo = _down(known/_up(1+gamma64))
    known_hi = _up(known/_down(1-gamma64))
    lower = _down(_down(total_lo-previous)-known_hi)
    upper = _up(_up(total_hi-previous)-known_lo)
    return lower, upper


def decode_background(currents, decay, known, event_counts, weights):
    lower, upper = arrival_intervals(currents,decay,known,event_counts)
    weights = np.asarray(weights,dtype=float)
    if (weights.shape != (currents.shape[1],) or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.array_equal(weights,weights.astype(np.float32).astype(float))):
        raise ValueError("one finite positive native FP32 background weight per target required")
    count_lo, count_hi = _down(lower/weights), _up(upper/weights)
    first, last = np.maximum(0.,np.ceil(count_lo)), np.floor(count_hi)
    valid = (first==last) & (first < 2**63)
    if not np.all(valid):
        examples=[{"tick":int(t),"column":int(j),"lower":float(count_lo[t,j]),
                   "upper":float(count_hi[t,j])} for t,j in np.argwhere(~valid)[:8]]
        raise ValueError("background count is noninteger or ambiguous: "+str(examples))
    return first.astype(np.int64), {"max_count_interval_width":float(np.max(count_hi-count_lo)),
                                    "max_known_arrivals_per_bin":int(event_counts.max())}


def known_arrivals(edges, spike_ids, spike_ticks, unit_ids, length, dt, *, weight_precision="float32"):
    """Accumulate the entire recorded source spike history through signed edges."""
    if weight_precision not in ("float32", "float64"):
        raise ValueError("unknown runtime synaptic weight precision")
    result = np.zeros((length,len(unit_ids),2),dtype=float)
    numbers = np.zeros(result.shape,dtype=np.int64)
    if len(spike_ids) != len(spike_ticks) or np.any(spike_ticks < 0):
        raise ValueError("invalid source spike history")
    pairs=np.column_stack((spike_ids,spike_ticks))
    if len(pairs) != len(np.unique(pairs,axis=0)):
        raise ValueError("source unit emitted more than once in one tick")
    ordered=edges.copy();ordered.sort(order=["source","target","delay","weight"])
    sources,starts,counts=np.unique(ordered["source"],return_index=True,return_counts=True)
    for uid,start,count in zip(sources,starts,counts):
        times=spike_ticks[spike_ids==uid]
        if not len(times):continue
        outgoing=ordered[start:start+count]
        delay=np.rint(outgoing["delay"]/dt).astype(np.int64)
        if np.any(np.abs(outgoing["delay"]-delay*dt)>1e-9):raise ValueError("off-grid source edge")
        target=np.searchsorted(unit_ids,outgoing["target"])
        if np.any(target>=len(unit_ids)) or not np.array_equal(unit_ids[target],outgoing["target"]):
            raise ValueError("source edge target not in recorded cortex")
        weight=outgoing["weight"].astype(weight_precision).astype(float)
        port=(weight<0).astype(int)
        # One spike at a time avoids a spike-count by outdegree temporary matrix.
        for t in times:
            arrivals=t+delay;keep=(arrivals>=0)&(arrivals<length)
            np.add.at(result,(arrivals[keep],target[keep],port[keep]),np.abs(weight[keep]))
            np.add.at(numbers,(arrivals[keep],target[keep],port[keep]),1)
    return result,numbers


def validate_spike_counts(unit_ids, evidence, spike_ids):
    order=np.argsort(evidence["unit_ids"])
    if not np.array_equal(evidence["unit_ids"][order],unit_ids):
        raise ValueError("native spike counters do not cover source identities")
    indices=np.searchsorted(unit_ids,spike_ids)
    if np.any(indices>=len(unit_ids)) or not np.array_equal(unit_ids[indices],spike_ids):
        raise ValueError("unknown spike identity in recording")
    actual=np.bincount(indices,minlength=len(unit_ids))
    if (not np.array_equal(actual,evidence["native_counts"][order])
            or not np.array_equal(actual,evidence["recorded_counts"][order])):
        raise ValueError("spike recording is incomplete against native counters")


def extract(snapshot,run,output):
    output.mkdir(parents=True,exist_ok=False)
    report={"schema":SCHEMA,"status":"FAILED","counts_decoding_validated":False,
            "scientific_reference_qualified":False,"source_run":str(run.resolve()),
            "snapshot_manifest_sha256":sha256_file(snapshot/"manifest.json")}
    try:
        metadata,arrays=load_initial_network(snapshot)
        capture=json.loads((run/"report.json").read_text())
        if capture["status"]!="EXECUTED" or capture["snapshot_manifest_sha256"]!=report["snapshot_manifest_sha256"]:
            raise ValueError("capture is not an executed replay of this snapshot")
        if not capture["checks"].get("spike_count_recording_complete",False):
            raise ValueError("native spike-count completeness evidence is required")
        for name in ["states.npz","spikes.npz","propagators.npz","spike_counts.npz"]:
            if sha256_file(run/name)!=capture["artifacts"][name]:raise ValueError("capture artifact changed: "+name)
        state=np.load(run/"states.npz",allow_pickle=False)
        spike=np.load(run/"spikes.npz",allow_pickle=False)
        prop=np.load(run/"propagators.npz",allow_pickle=False)
        counter=np.load(run/"spike_counts.npz",allow_pickle=False)
        validate_spike_counts(arrays["unit_ids"],counter,spike["spike_unit_ids"])
        ids=arrays["unit_ids"][arrays["populations"]!="TC"]
        dt=float(metadata["dt_ms"]);length=int(round(metadata["duration_ms"]/dt))
        if (not np.array_equal(state["unit_ids"],ids) or not np.array_equal(prop["unit_ids"],ids)
                or not np.array_equal(state["ticks"],np.arange(length+1))
                or state["states"].shape[:2]!=(length+1,len(ids))):
            raise ValueError("complete aligned cortical state and propagator records required")
        if not np.isin(spike["spike_unit_ids"],arrays["unit_ids"]).all():raise ValueError("unknown recorded spike source")
        field=list(state["fields"]);pf=list(prop["fields"])
        edges=arrays["edges"]
        bg_ids=[d["id"] for d in metadata["devices"] if d["model"]=="poisson_generator"]
        bg=edges[np.isin(edges["source"],bg_ids)].copy();bg.sort(order="target")
        if len(bg)!=len(ids) or not np.array_equal(bg["target"],ids):
            raise ValueError("exactly one source background edge per cortical target required")
        neural=edges[np.isin(edges["source"],arrays["unit_ids"]) & np.isin(edges["target"],ids)]
        arrivals,numbers=known_arrivals(neural,spike["spike_unit_ids"],spike["spike_ticks"],ids,length,dt)
        ex=state["states"][:,:,field.index("I_syn_ex")]
        inh=-state["states"][:,:,field.index("I_syn_in")]
        p11ex=prop["values"][:,pf.index("P11ex")].reshape(-1)
        p11in=prop["values"][:,pf.index("P11in")].reshape(-1)
        low,high=arrival_intervals(inh,p11in,arrivals[:,:,1],numbers[:,:,1])
        if np.any(low>0) or np.any(high<0):
            invalid=(low>0)|(high<0);examples=[{"tick":int(t),"unit_id":int(ids[j]),
                "lower_pa":float(low[t,j]),"upper_pa":float(high[t,j])} for t,j in np.argwhere(invalid)[:8]]
            raise ValueError("inhibitory events disagree with native current: "+str(examples))
        weights=bg["weight"].astype(np.float32).astype(float)
        counts,diagnostics=decode_background(ex,p11ex,arrivals[:,:,0],numbers[:,:,0],weights)
        delay=np.rint(bg["delay"]/dt).astype(np.int64)
        if np.any((counts>0)&(np.arange(length)[:,None]-delay[None,:]<=0)):
            raise ValueError("decoded background contains unrepresentable nonpositive source emission")
        with (output/"counts.npz").open("xb") as stream:
            np.savez_compressed(stream,unit_ids=ids,arrival_ticks=np.arange(length,dtype=np.int64),
                                counts=counts,dt_ms=dt)
        report.update(status="PASS",counts_decoding_validated=True,counts_sha256=sha256_file(output/"counts.npz"),
                      dt_ms=dt,simulation_duration_ms=metadata["duration_ms"],last_complete_arrival_tick=length-1,
                      cortical_unit_count=len(ids),background_event_count=int(counts.sum()),
                      background_max_multiplicity=int(counts.max()),
                      known_inhibitory_arrivals_verified=True,spike_capture_complete_against_native_counters=True,normal_native_propagators_verified=True,max_inhibitory_interval_width_pa=float(np.max(high-low)),
                      interval_model="one native FP32 decay; N+5 conservative FP32 arrival stages; FP64 known-sum interval",
                      source_report_sha256=sha256_file(run/"report.json"),schedule=capture["schedule"],
                      diagnostics=diagnostics)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    (output/"drive.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="traceback"}))
    return int(report["status"]!="PASS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(extract(args.snapshot,args.run,args.output))


if __name__=="__main__":main()

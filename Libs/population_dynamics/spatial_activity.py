"""Descriptive cortical activity and spatial fields from complete native events.

Within-episode pulse differences are not independent repeats or causal state
findings. TC emissions supply timing markers and are excluded from cortex data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from .nest_gpu_network import ticks
from .provenance import sha256_file
from .spatial_field import PeriodicPopulationGrid


@dataclass(frozen=True)
class Activity:
    bin_ms: float = 1.
    cells_per_axis: int = 4
    start_ms: float | None = None
    pulse_window_ms: float = 10.


def analyze(graph: Path, episode: Path, output: Path, settings: Activity):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(status="FAILED", scope="descriptive_within_one_graph_initial_state_noise_episode",
                  scientific_reference_qualified=False, neuroscience_finding_qualified=False,
                  settings=asdict(settings), source_episode=str(episode.resolve()))
    try:
        m=json.loads((graph/"manifest.json").read_text())
        r=json.loads((episode/"report.json").read_text())
        if r["status"]!="EXECUTED" or not r["checks"]["spike_count_recording_complete"]:
            raise ValueError("complete executed native recording required")
        if r["graph_manifest_sha256"]!=sha256_file(graph/"manifest.json"):
            raise ValueError("episode and geometry graph mismatch")
        for name in ("spikes.npz","spike_counts.npz"):
            if sha256_file(episode/name)!=r["artifacts"][name]:
                raise ValueError("native event/count artifact mismatch")
        with np.load(episode/"spikes.npz",allow_pickle=False) as data:
            ids,labels,xy=(data[k] for k in ("unit_ids","populations","positions_mm"))
            senders,event_ticks=(data[k] for k in ("spike_unit_ids","spike_ticks"))
        with np.load(episode/"spike_counts.npz",allow_pickle=False) as data:
            count_ids,native_counts,recorded_counts=(data[k] for k in ("unit_ids","native_counts","recorded_counts"))
        if (len(np.unique(ids))!=len(ids) or event_ticks.shape!=senders.shape
                or event_ticks.dtype.kind not in "iu" or senders.dtype.kind not in "iu"
                or not np.isfinite(xy).all() or not np.array_equal(native_counts,recorded_counts)):
            raise ValueError("invalid event/count identity or completeness")
        ordered=np.argsort(ids);sorted_ids=ids[ordered]
        positions=np.searchsorted(sorted_ids,senders)
        if np.any(positions>=len(ids)) or not np.array_equal(sorted_ids[positions],senders):
            raise ValueError("unknown native event unit")
        positions=ordered[positions]
        event_counts=np.bincount(positions,minlength=len(ids))
        lookup={int(u):i for i,u in enumerate(ids)}
        if set(map(int,count_ids))!=set(lookup) or not np.array_equal(
                event_counts[[lookup[int(u)] for u in count_ids]],recorded_counts):
            raise ValueError("saved events disagree with independent native counters")
        dt=float(r["dt_ms"])
        duration=int(ticks([r["duration_ms"]],dt)[0])
        if np.any(event_ticks<1) or np.any(event_ticks>duration):
            raise ValueError("events outside actual simulated support")
        onset=m["source_config"]["simulation"]["t_presim"] if settings.start_ms is None else settings.start_ms
        left,bin_width,pulse_width=map(int,ticks([onset,settings.bin_ms,settings.pulse_window_ms],dt))
        if bin_width<1 or pulse_width<1 or not 0<=left<duration or (duration-left)%bin_width:
            raise ValueError("positive aligned complete analysis bins required")
        edges=np.arange(left,duration+1,bin_width,dtype=np.int64)
        cortical=labels!="TC";event_cortical=cortical[positions]
        observer=PeriodicPopulationGrid.from_positions(ids[cortical],labels[cortical],xy[cortical],
            extent_mm=m["resolved_source_network"]["extent"],cells_per_axis=settings.cells_per_axis,
            definition="Source cortical soma coordinates in mm; source populations and periodic square cells; TC excluded")
        readout=observer.project_events(senders[event_cortical],event_ticks[event_cortical],
            observed_unit_ids=ids[cortical],bin_edges_ticks=edges,dt_seconds=dt/1000)
        inside=(event_ticks>=left)&(event_ticks<duration)
        if int(readout.field.count_sum.sum())!=int(np.sum(inside&event_cortical)):
            raise ValueError("cortical field event count mismatch")
        f=readout.field
        np.savez_compressed(output/"spatial_fields.npz",source_unit_ids=ids[cortical],
            source_populations=labels[cortical],unit_field_index=observer.mapping.group_index,
            field_ids=f.field_ids,field_populations=observer.population_labels,centers_mm=observer.centers_mm,
            total_unit_count=f.total_unit_count,count_sum=f.count_sum,observed_unit_count=f.observed_unit_count,
            complete=f.complete,observed_rate_hz=f.observed_rate_hz,density_hz_per_mm2=readout.density_hz_per_mm2,
            bin_edges_ticks=edges,dt_ms=dt,cell_area_mm2=observer.cell_area_mm2)
        unit_counts=np.bincount(positions[inside],minlength=len(ids))
        seconds=(duration-left)*dt/1000
        np.savez(output/"unit_rates.npz",unit_ids=ids[cortical],population=labels[cortical],
                 spike_count=unit_counts[cortical],rate_hz=unit_counts[cortical]/seconds)
        population=[]
        for name in m["resolved_source_network"]["populations"][:-1]:
            mask=labels==name;c=unit_counts[mask]
            population.append(dict(population=name,units=int(mask.sum()),spikes=int(c.sum()),
                mean_rate_hz=float(c.sum()/seconds/mask.sum()),active_units=int(np.sum(c>0)),
                minimum_unit_rate_hz=float(c.min()/seconds),maximum_unit_rate_hz=float(c.max()/seconds)))
        # Describe only equally observed early windows around actual TC emissions.
        pulse_records=[]
        for pulse in np.unique(event_ticks[~event_cortical]):
            pulse=int(pulse)
            if pulse-pulse_width<left or pulse+pulse_width>duration:
                pulse_records.append(dict(tc_emission_tick=pulse,status="INCOMPLETE_EQUAL_WINDOW"));continue
            item=dict(tc_emission_tick=pulse,status="DESCRIPTIVE",populations=[])
            for name in m["resolved_source_network"]["populations"][:-1]:
                match=labels[positions]==name;denom=int(np.sum(labels==name))*pulse_width*dt/1000
                before=int(np.sum(match&(event_ticks>=pulse-pulse_width)&(event_ticks<pulse)))
                after=int(np.sum(match&(event_ticks>=pulse)&(event_ticks<pulse+pulse_width)))
                item["populations"].append(dict(population=name,pre_spikes=before,post_spikes=after,
                    pre_rate_hz=before/denom,post_rate_hz=after/denom))
            pulse_records.append(item)
        event_order=np.lexsort((event_ticks,senders));sorted_senders=senders[event_order];sorted_ticks=event_ticks[event_order]
        same=np.diff(sorted_senders)==0
        cortical_pairs=event_cortical[event_order][1:]&same
        minimum_isi=int(np.min(np.diff(sorted_ticks)[cortical_pairs])) if cortical_pairs.any() else None
        refractory=int(ticks([m["resolved_source_network"]["neuron_params"]["t_ref"]],dt)[0])
        violations=int(np.sum(np.diff(sorted_ticks)[cortical_pairs]<refractory))
        report.update(status="DESCRIBED",cortical_units=int(cortical.sum()),tc_units=int((~cortical).sum()),
            source_events=len(senders),cortical_events_in_window=int(np.sum(inside&event_cortical)),
            excluded_final_right_boundary_events=int(np.sum(event_ticks==duration)),window_start_tick=left,
            window_stop_tick=duration,dt_ms=dt,populations=population,pulse_responses=pulse_records,
            cortical_minimum_isi_ticks=minimum_isi,cortical_refractory_violations=violations,
            caveat="One realization; equilibration, elapsed time and pulse history are confounded. No stationarity, equivalence or state-dependent causal conclusion.",
            event_count_reconciliation=True,field_count_reconciliation=True,
            episode_report_sha256=sha256_file(episode/"report.json"))
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report["seconds"]=time.perf_counter()-start
    report["artifacts"]={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
    (output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--episode",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.graph,args.episode,args.output,Activity(**yaml.safe_load(args.config.read_text())))
    print(json.dumps({k:report.get(k) for k in ("status","error","cortical_units","cortical_events_in_window","seconds")}))
    raise SystemExit(int(report["status"]!="DESCRIBED"))


if __name__=="__main__":main()

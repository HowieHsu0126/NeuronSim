"""Compare native CPU/GPU networks using matched recorded background arrivals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
import yaml
from .network_snapshot import load_initial_network
from .nest_background_drive import known_arrivals
from .provenance import sha256_file


def compare(snapshot,drive,gpu,cpu,protocol,output):
    output.mkdir(parents=True,exist_ok=False)
    report={"status":"FAILED","scientific_reference_qualified":False,
            "gpu":str(gpu.resolve()),"cpu":str(cpu.resolve()),"checks":{},
            "numerical_protocol":protocol,"bounds_role":"existing numerical engineering bounds; not population-field fidelity calibration"}
    try:
        metadata,arrays=load_initial_network(snapshot)
        gr=json.loads((gpu/"report.json").read_text());cr=json.loads((cpu/"report.json").read_text())
        dr=json.loads((drive/"drive.json").read_text())
        digest=sha256_file(snapshot/"manifest.json")
        if (gr["status"]!="EXECUTED" or cr["status"]!="EXECUTED"
                or gr["snapshot_manifest_sha256"]!=digest or cr["snapshot_manifest_sha256"]!=digest
                or not dr["counts_decoding_validated"] or dr["snapshot_manifest_sha256"]!=digest
                or sha256_file(drive/"counts.npz")!=cr["counts_sha256"]
                or cr["counts_sha256"]!=dr["counts_sha256"]
                or gr["schedule"]!=cr["schedule"] or gr["schedule"]!=dr["schedule"]
                or gr["gpu_source_revision"]!=protocol["source_revision"]):
            raise ValueError("paired source, schedule or drive identity mismatch")
        for root,r in [(gpu,gr),(cpu,cr)]:
            for name in ["spikes.npz","states.npz","parameter_readback.json"]:
                if sha256_file(root/name)!=r["artifacts"][name]:raise ValueError("native artifact changed: "+str(root/name))
        gs=np.load(gpu/"spikes.npz",allow_pickle=False);cs=np.load(cpu/"spikes.npz",allow_pickle=False)
        g=np.load(gpu/"states.npz",allow_pickle=False);c=np.load(cpu/"states.npz",allow_pickle=False)
        d=np.load(drive/"counts.npz",allow_pickle=False)
        ids=arrays["unit_ids"][arrays["populations"]!="TC"]
        for x in [gs,cs]:
            for field in ["unit_ids","populations","positions_mm"]:
                if not np.array_equal(x[field],arrays[field]):raise ValueError("native source identity changed")
        if not np.array_equal(g["unit_ids"],ids) or not np.array_equal(c["unit_ids"],ids):
            raise ValueError("cortical state identity changed")
        report["clock_errors_ms"]={}
        for backend,state_archive,spike_archive in [("gpu",g,gs),("cpu",c,cs)]:
            for kind,raw,tick_values in [("state",state_archive["raw_times_ms"],state_archive["ticks"]),
                                         ("spike",spike_archive["raw_spike_times_ms"],spike_archive["spike_ticks"])]:
                maximum=float(np.max(np.abs(raw-tick_values*metadata["dt_ms"]))) if len(raw) else 0.
                report["clock_errors_ms"][backend+"_"+kind]=maximum
                report["checks"][backend+"_"+kind+"_existing_clock_bound"]=maximum<=protocol["time_atol_ms"]
        report["quantization"]=cr["quantization"]
        report["schedule"]=cr["schedule"]
        report["spikes"]={}
        for name,units in [("cortical",ids),("thalamic",arrays["unit_ids"][arrays["populations"]=="TC"])]:
            gm=np.isin(gs["spike_unit_ids"],units);cm=np.isin(cs["spike_unit_ids"],units)
            ga=np.column_stack((gs["spike_unit_ids"][gm],gs["spike_ticks"][gm]))
            ca=np.column_stack((cs["spike_unit_ids"][cm],cs["spike_ticks"][cm]))
            exact=np.array_equal(ga,ca)
            report["checks"][name+"_spike_ticks_exact"]=exact
            report["spikes"][name]={"gpu_count":len(ga),"cpu_count":len(ca),"exact":exact,
                 "gpu_only_events":len(set(map(tuple,ga))-set(map(tuple,ca))),
                 "cpu_only_events":len(set(map(tuple,ca))-set(map(tuple,ga)))}
        dt=float(metadata["dt_ms"]);length=len(d["arrival_ticks"])
        common=np.intersect1d(g["ticks"],c["ticks"])
        common=common[common<=dr["last_complete_arrival_tick"]]
        if not len(common):raise ValueError("no complete shared state interval")
        gi=np.searchsorted(g["ticks"],common);ci=np.searchsorted(c["ticks"],common)
        gf=list(g["fields"]);cf=list(c["fields"])
        if np.any(np.diff(common)!=1):raise ValueError("gaps in shared state recording")
        # CPU post-arrival PSCs are translated using CPU's own emitted spikes,
        # its native weight precision and the independently fixed background.
        edges=arrays["edges"]
        neural=edges[np.isin(edges["source"],arrays["unit_ids"]) & np.isin(edges["target"],ids)]
        precision="float32" if cr["quantization"]=="gpu_fp32_inputs" else "float64"
        incoming,_=known_arrivals(neural,cs["spike_unit_ids"],cs["spike_ticks"],ids,length,dt,weight_precision=precision)
        bg_ids=[r["id"] for r in metadata["devices"] if r["model"]=="poisson_generator"]
        bg=edges[np.isin(edges["source"],bg_ids)].copy();bg.sort(order="target")
        if not np.array_equal(bg["target"],ids) or not np.array_equal(d["unit_ids"],ids):raise ValueError("background target order changed")
        weights=bg["weight"].astype(precision).astype(float)
        incoming[:,:,0]+=d["counts"]*weights
        actual_cpu=c["states"][ci].copy()
        actual_cpu[:,:,cf.index("I_syn_ex")]-=incoming[common,:,0]
        actual_cpu[:,:,cf.index("I_syn_in")]+=incoming[common,:,1]
        gr_params=json.loads((gpu/"parameter_readback.json").read_text())
        rest=np.asarray(gr_params["E_L"],dtype=float).reshape(1,-1)
        gpu_values={"V_m":g["states"][gi,:,gf.index("V_m_rel")]+rest,
                    "I_syn_ex":g["states"][gi,:,gf.index("I_syn_ex")],
                    "I_syn_in":g["states"][gi,:,gf.index("I_syn_in")]}
        report["state_comparison"]={"common_first_tick":int(common[0]),"common_last_tick":int(common[-1]),
            "common_tick_count":len(common),"cortical_unit_count":len(ids),
            "cpu_available_ticks":[int(c["ticks"][0]),int(c["ticks"][-1])],
            "gpu_available_ticks":[int(g["ticks"][0]),int(g["ticks"][-1])],
            "cpu_phase":"raw post-arrival PSC minus actual CPU arrivals","gpu_phase":"raw pre-arrival PSC","fields":{}}
        for field,values in gpu_values.items():
            truth=actual_cpu[:,:,cf.index(field)];error=values-truth
            bound=protocol["voltage_atol_mv"] if field=="V_m" else protocol["current_atol_pa"]
            worst=np.unravel_index(np.argmax(np.abs(error)),error.shape)
            report["checks"][field+"_existing_numerical_bound"]=bool(np.all(np.abs(error)<=bound))
            report["state_comparison"]["fields"][field]={"max_absolute_error":float(np.max(np.abs(error))),
                "rmse":float(np.sqrt(np.mean(error**2))),"mean_absolute_error":float(np.mean(np.abs(error))),
                "bound":bound,"over_bound_samples":int(np.count_nonzero(np.abs(error)>bound)),
                "sample_count":error.size,"worst_tick":int(common[worst[0]]),"worst_unit_id":int(ids[worst[1]])}
        report["status"]="PASS" if all(report["checks"].values()) else "FAILED"
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    (output/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="traceback"}))
    return int(report["status"]!="PASS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["snapshot","drive","gpu","cpu","output"]:parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if (cfg.get("campaign")!="neural_population_dynamics" or cfg.get("run_class")!="engineering_diagnostic"
            or cfg.get("exact_cortical_spike_ticks") is not True
            or cfg.get("exact_thalamic_spike_ticks") is not True
            or cfg.get("state_scope")!="every common recorded native tick with complete input"
            or cfg.get("scientific_reference_qualified") is not False):
        raise ValueError("unrecognized paired-network comparison protocol")
    protocol_path=Path(cfg["numerical_protocol"])
    protocol=yaml.safe_load(protocol_path.read_text())
    protocol["source_path"]=str(protocol_path);protocol["sha256"]=sha256_file(protocol_path)
    raise SystemExit(compare(args.snapshot,args.drive,args.gpu,args.cpu,protocol,args.output))


if __name__=="__main__":main()

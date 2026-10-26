"""Replay an exported Mesocircuit network in one fresh NEST GPU episode.

This is a reference-qualification runner, not a qualified truth generator.
Deterministic TC relays are folded into their source-defined emission schedules.
Native Poisson delay indices are translated explicitly; neuronal edges are intact.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from .network_snapshot import load_initial_network, NEURON_PARAMETERS
from .nest_gpu_connectivity import ExplicitConnector
from .provenance import sha256_file

SOURCE_REVISION = "f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5"
GPU_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
# Poisson Update uses tick-1 but direct delivery looks up tick-delay+1.
# Adding two native delay ticks restores arrival at emission+source delay.
POISSON_DELAY_OFFSET_TICKS = 2
EDGE_READBACK_DTYPE = np.dtype([("target", "<i8"), ("source", "<i8"),
                              ("port", "<i4"), ("delay_ticks", "<i8"), ("weight", "<f4")])
STATE_FIELDS = ("V_m_rel", "I_syn_ex", "I_syn_in", "refractory_step")
PROPAGATOR_FIELDS = ("P11ex", "P11in", "P20", "P21ex", "P21in", "P22")


def ticks(values, dt, *, recorded=False):
    values = np.asarray(values, dtype=float)
    if not math.isfinite(dt) or dt <= 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative grid times required")
    result = np.rint(values / dt).astype(np.int64)
    radius = np.full(values.shape, 1e-9)
    if recorded:
        radius = (0.5 * np.abs(np.spacing(values.astype(np.float32))).astype(float)
                  + result * 0.5 * float(np.spacing(np.float32(dt)))
                  + 4 * np.finfo(float).eps * np.maximum(values, dt))
    if np.any(radius >= dt / 2) or np.any(np.abs(values - result * dt) > radius):
        raise ValueError("time does not uniquely identify a native tick")
    return result


def prepare(metadata, arrays, schedule):
    if metadata["source_revision"] != SOURCE_REVISION:
        raise ValueError("unreviewed Mesocircuit source revision")
    if metadata["cortical_neuron_model"] != "iaf_psc_exp" or metadata["thalamic_neuron_model"] != "parrot_neuron":
        raise ValueError("unsupported source neuron model")
    if schedule not in ("initial", "effective"):
        raise ValueError("unknown source pulse schedule")
    dt = metadata["dt_ms"]
    ticks([metadata["duration_ms"]], dt)
    ids, labels, edges = arrays["unit_ids"], arrays["populations"], arrays["edges"]
    if not np.isfinite(arrays["positions_mm"]).all() or not np.isfinite(arrays["cortical_parameters"]).all():
        raise ValueError("nonfinite source geometry or parameters")
    if not np.isfinite(edges["weight"]).all() or np.any(edges["delay"] <= 0):
        raise ValueError("invalid source connections")
    ticks(edges["delay"], dt)
    cortex, tc = ids[labels != "TC"], ids[labels == "TC"]
    devices = {int(d["id"]): d for d in metadata["devices"]}
    if len(devices) != len(metadata["devices"]) or set(devices).intersection(ids):
        raise ValueError("source identity collision")
    if not np.isin(edges["source"], [*ids, *devices]).all() or not np.isin(edges["target"], ids).all():
        raise ValueError("unknown source connection identity")
    params = dict(zip(NEURON_PARAMETERS, arrays["cortical_parameters"].T))
    for k in ("C_m", "tau_m", "tau_syn_ex", "tau_syn_in"):
        if np.any(params[k] <= 0): raise ValueError("nonpositive neuronal time/scale")
    ticks(params["t_ref"], dt)  # Off-grid refractory cases remain unqualified.
    mapped = {k: params[k] for k in ("C_m", "tau_m", "E_L", "I_e", "t_ref")}
    mapped.update(V_m_rel=params["V_m"]-params["E_L"], Theta_rel=params["V_th"]-params["E_L"],
                  V_reset_rel=params["V_reset"]-params["E_L"], tau_ex=params["tau_syn_ex"],
                  tau_in=params["tau_syn_in"], I_syn_ex=np.zeros(len(cortex)),
                  I_syn_in=np.zeros(len(cortex)), den_delay=np.zeros(len(cortex)))
    background = []
    pulse = {}
    for uid, device in devices.items():
        p = device["parameters"]
        if device["model"] == "poisson_generator":
            stop = math.inf if p["stop"] == "positive_infinity" else float(p["stop"])
            if p["origin"] != 0 or p["start"] != 0 or math.isnan(stop) or stop <= metadata["duration_ms"]:
                raise ValueError("only continuous default background is in this qualification scope")
            if not math.isfinite(p["rate"]) or p["rate"] < 0: raise ValueError("invalid source background rate")
            if not np.isin(edges["target"][edges["source"] == uid], cortex).all():
                raise ValueError("background must directly target cortex")
            background.append(device)
        elif device["model"] == "spike_generator":
            initial = ticks(p["initial_spike_times_ms"], dt)
            effective = ticks(p["effective_spike_times_ms"], dt)
            control = metadata["config"].get("pulse_control")
            if control is None:
                if not np.array_equal(initial,effective): raise ValueError("changed pulses lack a source decision boundary")
            else:
                boundary = int(ticks([control["decision_time_ms"]],dt)[0])
                if not np.array_equal(initial[initial <= boundary],effective[effective <= boundary]):
                    raise ValueError("future intervention changed the source history")
            pulse[uid] = initial if schedule == "initial" else effective
            if np.any(pulse[uid] < 1) or np.any(np.diff(pulse[uid]) <= 0):
                raise ValueError("ordered positive pulse times required")
            if not np.isin(edges["target"][edges["source"] == uid], tc).all():
                raise ValueError("pulse folding requires exclusively TC targets")
        else:
            raise ValueError("unsupported source input device: "+device["model"])
    relays = edges[np.isin(edges["target"], tc)]
    schedules = {}
    for uid in tc:
        incoming = relays[relays["target"] == uid]
        if len(incoming) > 1 or any(int(e["source"]) not in pulse for e in incoming):
            raise ValueError("TC folding requires at most one external deterministic input per relay")
        schedules[int(uid)] = (pulse[int(incoming[0]["source"])] + int(ticks([incoming[0]["delay"]], dt)[0])
                               if len(incoming) else np.empty(0, dtype=np.int64))
    return {"cortex": cortex, "tc": tc, "mapped_parameters": mapped,
            "background": background, "tc_schedules": schedules,
            "edges": edges[~np.isin(edges["target"], tc)], "folded_relay_edges": len(relays)}


def run(snapshot, output, schedule="effective", record_state=False):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "FAILED", "schedule": schedule, "snapshot": str(snapshot.resolve()),
              "snapshot_manifest_sha256": sha256_file(snapshot/"manifest.json"),
              "scientific_reference_qualified": False, "gpu_source_revision": GPU_REVISION,
              "execution": "one continuous preloaded episode", "record_all_cortical_states": record_state,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "checks": {}}
    try:
        metadata, arrays = load_initial_network(snapshot)
        plan = prepare(metadata, arrays, schedule)
        h, duration = metadata["dt_ms"], metadata["duration_ms"]
        ng = importlib.import_module("nestgpu")
        report["native_library_sha256"] = sha256_file(Path(os.environ["NESTGPU_LIB"]))
        seed = metadata["config"]["random_streams"]["background_drive"]
        ng.SetRandomSeed(seed); ng.SetTimeResolution(h); ng.SetKernelStatus("verbosity_level", 0)
        cortex = ng.Create("iaf_psc_exp", len(plan["cortex"]))
        id_map = {int(uid): int(cortex[i]) for i, uid in enumerate(plan["cortex"])}
        for name, values in plan["mapped_parameters"].items():
            ng.SetStatus(cortex, name, {"array": values.tolist()})
        actual = ng.GetStatus(cortex)
        parameter_readback = {k: [row[k] for row in actual] for k in plan["mapped_parameters"]}
        report["checks"]["initial_parameters_exact_fp32"] = all(
            np.array_equal(np.asarray(parameter_readback[k]).reshape(-1), v.astype(np.float32).astype(float))
            for k, v in plan["mapped_parameters"].items())
        (output/"parameter_readback.json").write_text(json.dumps(parameter_readback, indent=2)+"\n")
        maximum_spikes = int(math.ceil(duration / max(h, float(np.min(plan["mapped_parameters"]["t_ref"]))))) + 2
        ng.ActivateSpikeCount(cortex)
        ng.ActivateRecSpikeTimes(cortex, maximum_spikes)
        tc_groups = []
        for uid in plan["tc"]:
            group = ng.Create("spike_generator", 1); uid = int(uid)
            times = plan["tc_schedules"][uid]
            ng.SetStatus(group, {"spike_times": ((times-1)*h).tolist()})
            ng.ActivateSpikeCount(group)
            ng.ActivateRecSpikeTimes(group, len(times)+2)
            id_map[uid] = int(group[0]); tc_groups.append(group)
        background_ids = set()
        for device in plan["background"]:
            group = ng.Create("poisson_generator", 1)
            ng.SetStatus(group, {"rate": device["parameters"]["rate"]})
            id_map[device["id"]] = int(group[0]); background_ids.add(device["id"])
        connector = ExplicitConnector(ng)
        edges = plan["edges"]
        expected = np.empty(len(edges), dtype=EDGE_READBACK_DTYPE)
        for begin in range(0, len(edges), 100000):
            batch = edges[begin:begin+100000]
            source = [id_map[int(u)] for u in batch["source"]]
            target = [id_map[int(u)] for u in batch["target"]]
            ports = (batch["weight"] < 0).astype(int)
            delay_ticks = ticks(batch["delay"], h) + np.isin(batch["source"], list(background_ids))*POISSON_DELAY_OFFSET_TICKS
            connector.connect(source, target, batch["weight"].tolist(), (delay_ticks*h).tolist(), ports.tolist())
            expected[begin:begin+len(batch)] = list(zip(batch["target"],batch["source"],ports,delay_ticks,batch["weight"]))
        state_record = None
        if record_state:
            state_record = ng.CreateRecord("", [k for _ in plan["cortex"] for k in STATE_FIELDS],
                                           [id_map[int(uid)] for uid in plan["cortex"] for _ in STATE_FIELDS],
                                           [0]*(len(plan["cortex"])*len(STATE_FIELDS)))
        ng.Calibrate()
        calibrated = ng.GetStatus(cortex)
        np.savez_compressed(output/"propagators.npz", unit_ids=plan["cortex"],
                            fields=np.asarray(PROPAGATOR_FIELDS),
                            values=np.asarray([[row[k] for k in PROPAGATOR_FIELDS]
                                               for row in calibrated], dtype=float))
        inverse = {v:k for k,v in id_map.items()}
        native = ng.GetConnectionStatus(ng.GetConnections())
        readback = np.empty(len(native), dtype=EDGE_READBACK_DTYPE)
        for i,e in enumerate(native):
            readback[i] = (inverse[e["target"]], inverse[e["source"]], e["port"],
                           int(ticks([e["delay"]], h, recorded=True)[0]), e["weight"])
        expected.sort(order=list(EDGE_READBACK_DTYPE.names));readback.sort(order=list(EDGE_READBACK_DTYPE.names))
        np.save(output/"edge_readback.npy", readback, allow_pickle=False)
        report["checks"]["exact_translated_edge_multiset"] = np.array_equal(expected, readback)
        report.update(cortical_neurons=len(plan["cortex"]), thalamic_relays=len(plan["tc"]),
                      native_connections=connector.connection_count, source_connections=len(arrays["edges"]),
                      folded_relay_edges=plan["folded_relay_edges"], background_seed=seed,
                      poisson_native_delay_offset_ticks=POISSON_DELAY_OFFSET_TICKS)
        if not all(report["checks"].values()): raise ValueError("native network readback failed")
        simulation_start = time.perf_counter();ng.Simulate(duration)
        report["simulation_seconds"] = time.perf_counter()-simulation_start
        trains = list(ng.GetRecSpikeTimes(cortex)) + [ng.GetRecSpikeTimes(g)[0] for g in tc_groups]
        units = np.concatenate([plan["cortex"], plan["tc"]])
        native_counts = np.concatenate([
            np.asarray([row["spike_count"] for row in ng.GetStatus(cortex)],dtype=np.int64).reshape(-1),
            np.asarray([ng.GetStatus(group)[0]["spike_count"] for group in tc_groups],dtype=np.int64).reshape(-1)])
        recorded_counts = np.asarray([len(train) for train in trains],dtype=np.int64)
        np.savez_compressed(output/"spike_counts.npz",unit_ids=units,native_counts=native_counts,
                            recorded_counts=recorded_counts,
                            capacities=np.asarray([maximum_spikes]*len(plan["cortex"])+
                                [len(plan["tc_schedules"][int(uid)])+2 for uid in plan["tc"]]))
        report["checks"]["spike_count_recording_complete"] = bool(np.array_equal(native_counts,recorded_counts))
        spike_ids = np.concatenate([np.full(len(v), u, dtype=np.int64) for u,v in zip(units,trains)])
        raw_times = np.concatenate([np.asarray(v, dtype=float) for v in trains])
        spike_ticks = ticks(raw_times, h, recorded=True)
        order = np.lexsort((spike_ids, spike_ticks))
        np.savez_compressed(output/"spikes.npz", unit_ids=arrays["unit_ids"], populations=arrays["populations"],
                            positions_mm=arrays["positions_mm"], spike_unit_ids=spike_ids[order],
                            raw_spike_times_ms=raw_times[order], spike_ticks=spike_ticks[order], dt_ms=h)
        report["checks"]["tc_emissions_exact"] = all(
            np.array_equal(ticks(train,h,recorded=True), plan["tc_schedules"][int(uid)][plan["tc_schedules"][int(uid)]*h <= duration])
            for uid,train in zip(plan["tc"],trains[len(plan["cortex"]):]))
        if state_record is not None:
            raw = np.asarray(ng.GetRecordData(state_record), dtype=float)
            state_ticks = ticks(raw[:,0],h,recorded=True)
            if not np.array_equal(state_ticks, np.arange(int(round(duration/h))+1)):
                raise ValueError("incomplete state recording clock")
            np.savez_compressed(output/"states.npz", ticks=state_ticks, raw_times_ms=raw[:,0],
                                unit_ids=plan["cortex"], fields=np.asarray(STATE_FIELDS),
                                states=raw[:,1:].reshape(len(raw),len(plan["cortex"]),len(STATE_FIELDS)))
        report["status"] = "EXECUTED" if all(report["checks"].values()) else "FAILED"
        report["spike_count"] = len(raw_times)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    report.update(total_seconds=time.perf_counter()-started,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    report["artifacts"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["artifacts","traceback"]}))
    return int(report["status"] != "EXECUTED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=("initial","effective"), default="effective")
    parser.add_argument("--record-state", action="store_true")
    args=parser.parse_args()
    raise SystemExit(run(args.snapshot,args.output,args.schedule,args.record_state))


if __name__ == "__main__":
    main()

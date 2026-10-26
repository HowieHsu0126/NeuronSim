"""Stream production-sized native Poisson driver counts with selected PSC replay.

This isolates the driver: no recurrent edges, no TC events, unit excitatory
weights, and targets that cannot spike. It does not qualify neuronal dynamics.
The cortical group, TC group, eight source groups and target order match the
spatial runtime. Full target counts are reduced only after unique FP32 decoding.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import yaml

from .nest_gpu_poisson import _clock_ticks, _decode, native_rng_identity
from .poisson_count_dependence import StreamingCountTrace
from .provenance import sha256_file


def decode_isolated_arrivals(before, after):
    """Invert one integer input added to an already rounded FP32 current.

    The isolated source has exactly one +1-weight background edge per target.
    sendDirectSpikeKernel converts its integer count exactly below 2**20;
    GetSpikes/GetInputSpikes adds that integer in double and rounds once to
    FP32. Half an outward ULP of the result therefore contains the addition
    error. FTZ-sized values receive a conservative minimum-normal allowance.
    This shortcut does not apply to recurrent/multiple/non-unit-weight inputs.
    """
    before, after = np.asarray(before), np.asarray(after)
    if (before.dtype != np.float32 or after.dtype != np.float32 or before.shape != after.shape
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any(before < 0) or np.any(after < 0) or np.any(after >= 2**20)):
        raise ValueError("isolated finite nonnegative FP32 current below 2**20 required")
    difference = after.astype(np.float64)-before.astype(np.float64)
    radius = np.abs(np.spacing(after)).astype(np.float64)*.5 + np.finfo(np.float32).tiny
    radius *= 1 + 8*np.finfo(np.float64).eps
    candidate = np.rint(difference)
    if np.any(radius >= .5) or np.any(np.abs(candidate-difference)>radius) or np.any(candidate < 0):
        raise ValueError("no unique integer inside isolated-arrival rounding interval")
    return candidate.astype(np.int64), float((2*radius).max())


class NativeScalarReadout:
    """Copy and free the caller-owned malloc array returned by the pinned C API.

    Source: BaseNeuron::GetScalVar -> NESTGPU::GetNeuronVar -> C wrapper.
    No pointer is retained by the native object. Only scalar variables use this
    path; array/group parameters have different ownership and are not accepted.
    """
    def __init__(self, ng, group):
        self.ng, self.group = ng, group
        self.free = ctypes.CDLL(None).free
        self.free.argtypes = [ctypes.c_void_p]; self.free.restype = None

    def read(self, name):
        if not self.ng.IsNeuronScalVar(self.group.i0, name):
            raise ValueError("owned scalar-variable readout required")
        pointer = self.ng.NESTGPU_GetNeuronVar(self.group.i0, self.group.n, name.encode())
        try:
            if not pointer or self.ng.GetErrorCode() != 0:
                raise RuntimeError(str(self.ng.GetErrorMessage()))
            return np.ctypeslib.as_array(pointer, shape=(self.group.n,)).copy()
        finally:
            if pointer: self.free(ctypes.cast(pointer, ctypes.c_void_p))


def acquire(graph_manifest: Path, config_path: Path, expected_library_sha256: str, output: Path):
    cfg = yaml.safe_load(config_path.read_text())
    manifest = json.loads(graph_manifest.read_text()); net = manifest["resolved_source_network"]
    library_path = Path(os.environ["NESTGPU_LIB"])
    if sha256_file(library_path) != expected_library_sha256:
        raise ValueError("native library differs from the qualified engineering runtime")
    rng_identity = native_rng_identity(library_path)
    if cfg.get("expected_poisson_rng_family",rng_identity) != rng_identity:
        raise ValueError("native Poisson RNG family differs from the declared variant")
    sizes = np.asarray(net["num_neurons"][:8], dtype=int)
    rates = float(net["bg_rate"]) * np.asarray(net["ext_indegrees"], dtype=float)
    dt = float(manifest["resolved_simulation"]["sim_resolution"])
    duration = float(cfg["duration_ms"]); n_steps = round(duration/dt)
    if duration <= 0 or abs(n_steps*dt-duration) > 1e-10 or cfg["chunk_ticks"] < 1:
        raise ValueError("positive aligned duration and chunk width required")
    decoder = cfg.get("decoder", "outward_full_interval")
    if decoder not in ("outward_full_interval", "isolated_single_addition"):
        raise ValueError("unknown source count decoder")
    if sizes.sum() != manifest["cortical_neurons"] or len(sizes) != 8 or rates.shape != (8,):
        raise ValueError("complete eight-population source mapping required")
    physical_delay = round(float(net["delay_poisson"])/dt)
    if not np.isclose(physical_delay*dt, net["delay_poisson"], rtol=0, atol=1e-10):
        raise ValueError("source background delay is not aligned")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", configuration=cfg, graph_manifest_sha256=sha256_file(graph_manifest),
                  native_library_sha256=expected_library_sha256, native_poisson_rng_family=rng_identity,
                  scientific_reference_qualified=False,
                  scope="isolated_production_population_driver", target_count=int(sizes.sum()),
                  population_names=net["populations"][:8], population_sizes=sizes.tolist(),
                  rates_hz=rates.tolist(), dt_ms=dt, physical_delay_ticks=physical_delay,
                  backend_delay_ticks=physical_delay+2, expected_arrival_support=[physical_delay+1,n_steps-1],
                  arrival_onset_rule="first source update at tick1 plus physical delay; backend delay minus1",
                  native_group_order="all cortical units; TC units; eight individual background generators",
                  omitted="recurrent/TC connections and TC events; targets cannot fire; all weights are +1",
                  source_neural_fidelity_qualified=False, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    started = time.perf_counter()
    (output/"acquisition.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        import nestgpu as ng
        ng.SetTimeResolution(dt); ng.SetRandomSeed(int(cfg["seed"])); ng.SetKernelStatus("verbosity_level", 0)
        cortex = ng.Create("iaf_psc_exp", int(sizes.sum()))
        ng.SetStatus(cortex, {"tau_m":10., "C_m":250., "E_L":-65., "V_m_rel":0., "V_reset_rel":0.,
                             "Theta_rel":1e6, "I_e":0., "t_ref":0., "den_delay":0.,
                             "tau_ex":float(cfg["tau_syn_ms"]), "tau_in":float(cfg["tau_syn_ms"]),
                             "I_syn_ex":0., "I_syn_in":0.})
        ng.ActivateSpikeCount(cortex)
        ng.Create("spike_generator", int(net["num_neurons"][8]))
        generators = [ng.Create("poisson_generator", 1) for _ in sizes]
        offset = 0
        for group, count, rate in zip(generators, sizes, rates):
            ng.SetStatus(group, {"rate":float(rate)})
            target = ng.NodeSeq(cortex.i0+offset, int(count))
            ng.Connect(group, target, {"rule":"all_to_all"}, {"weight":1.,"delay":(physical_delay+2)*dt,"receptor":0})
            offset += int(count)
        ids = np.arange(cortex.i0,cortex.i0+cortex.n,dtype=np.int64)
        labels = np.repeat(np.arange(8),sizes)
        means = (np.float32(dt)*rates.astype(np.float32)/np.float32(1000)).astype(np.float32)
        trace = StreamingCountTrace(ids,labels,means[labels])
        selected = trace.selected
        recorder = ng.CreateRecord("", ["I_syn_ex"]*len(selected), ids[selected].tolist(), [0]*len(selected))
        ng.Calibrate()
        decay = np.asarray(ng.GetNeuronStatus(cortex,"P11ex"),dtype=np.float32)
        if np.any(decay < np.finfo(np.float32).tiny) or np.any(decay > 1):
            raise ValueError("normal positive PSC decay required")
        readout = NativeScalarReadout(ng,cortex)
        library = ctypes.CDLL(str(library_path))
        start, step = library.NESTGPU_StartSimulation, library.NESTGPU_SimulationStep
        for function in (start,step): function.argtypes=[]; function.restype=ctypes.c_int
        def checked(function):
            if function()!=0 or ng.GetErrorCode()!=0: raise RuntimeError(str(ng.GetErrorMessage()))
        ng.SetSimTime(duration); checked(start)
        # pre[k] = fl32(P11ex * post[k-1]) is the actual neuronal update
        # before arrivals at k. The selected native recorder independently
        # checks this phase identity at every recorded tick.
        previous = readout.read("I_syn_ex")*decay
        predicted_selected_pre = np.zeros((n_steps+1,len(selected)),np.float32)
        predicted_selected_pre[1] = previous[selected]
        buffer = np.empty((int(cfg["chunk_ticks"])+1,len(ids)),np.float64)
        after_buffer = np.empty((int(cfg["chunk_ticks"]),len(ids)),np.float32)
        buffer[0] = previous; used = 0; first_tick=1
        max_interval = max_residual = 0.; total_decoded=0
        for tick in range(1,n_steps+1):
            checked(step)
            current = readout.read("I_syn_ex")
            if tick == n_steps: break
            pre_next = current*decay
            predicted_selected_pre[tick+1] = pre_next[selected]
            after_buffer[used] = current
            used += 1; buffer[used] = pre_next
            if used == cfg["chunk_ticks"] or tick == n_steps-1:
                if decoder == "outward_full_interval":
                    counts, diagnostic = _decode(buffer[:used+1],decay.astype(np.float64))
                else:
                    counts, width = decode_isolated_arrivals(buffer[:used].astype(np.float32),after_buffer[:used])
                    diagnostic = {"max_interval_width":width,"max_reconstruction_residual":float(
                        np.max(np.abs(after_buffer[:used].astype(np.float64)-buffer[:used]-counts)))}
                    if first_tick == 1 or tick == n_steps-1:
                        strict, _ = _decode(buffer[:used+1],decay.astype(np.float64))
                        if not np.array_equal(strict,counts): raise ValueError("strict and isolated count decoders differ")
                trace.update(counts,np.arange(first_tick,first_tick+used,dtype=np.int64))
                total_decoded += counts.size
                max_interval=max(max_interval,diagnostic["max_interval_width"])
                max_residual=max(max_residual,diagnostic["max_reconstruction_residual"])
                first_tick += used; buffer[0] = buffer[used]; used=0
            if tick%10000==0:
                print(json.dumps(dict(tick=tick,total_ticks=n_steps,seconds=time.perf_counter()-started)),flush=True)
        result = trace.finish()
        raw = np.asarray(ng.GetRecordData(recorder),dtype=np.float64)
        ticks = _clock_ticks(raw[:,0],dt)
        if not np.array_equal(ticks,np.arange(n_steps+1)) or not np.array_equal(raw[:,1:],predicted_selected_pre):
            raise ValueError("selected native recorder disagrees with full-target pre-arrival reconstruction")
        selected_counts, _ = _decode(raw[:,1:],decay[selected].astype(np.float64))
        if not np.array_equal(selected_counts[1:],result.sampled_counts):
            raise ValueError("selected native record count replay differs")
        if np.any(result.aggregate_counts[result.arrival_ticks < physical_delay+1]):
            raise ValueError("events outside source-declared arrival support")
        spike_counts=np.asarray(ng.GetNeuronStatus(cortex,"spike_count")).reshape(-1)
        if np.any(spike_counts): raise ValueError("isolation targets fired")
        np.savez_compressed(output/"trace.npz",arrival_ticks=result.arrival_ticks,
                            sampled_counts=result.sampled_counts,aggregate_counts=result.aggregate_counts,
                            sampled_ids=result.sampled_ids,sampled_groups=result.sampled_groups,sampled_means=result.sampled_means,
                            aggregate_means=result.aggregate_means,group_target_counts=result.group_target_counts,
                            group_labels=result.group_labels,complete_target_count=result.complete_target_count,
                            selected_native_pre_currents=raw[:,1:].astype(np.float32),record_ticks=ticks,
                            selected_decay=decay[selected],target_ids=ids,generator_index=labels,
                            lambda_per_target=means[labels],terminal_current=current)
        report.update(status="ACQUIRED_ENGINEERING",decoded_unit_bins=total_decoded,
                      selected_native_pre_values_checked=int(raw[:,1:].size),selected_count_values_checked=int(selected_counts[1:].size),
                      maximum_decoding_interval_width=max_interval,maximum_count_reconstruction_residual=max_residual,
                      target_spikes=0,seconds=time.perf_counter()-started,
                      peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      trace_sha256=sha256_file(output/"trace.npz"))
    except Exception as error:
        report.update(status="FAILED",error=repr(error),seconds=time.perf_counter()-started)
        raise
    finally:
        (output/"acquisition.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-manifest",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--expected-library-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(acquire(args.graph_manifest,args.config,args.expected_library_sha256,args.output)))


if __name__=="__main__":
    main()

"""Stream a source-law spatial graph into one continuous native GPU episode.

Graph shards remain bounded; native readback covers complete adjacencies of
preselected sources, never the Cartesian product of all neurons. Execution and
import checks are engineering evidence, not source-dynamics qualification.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import yaml

from .mesocircuit_continuation import prepare_pulse_protocol
from .nest_gpu_connectivity import ExplicitConnector
from .nest_gpu_network import (EDGE_READBACK_DTYPE, GPU_REVISION,
    POISSON_DELAY_OFFSET_TICKS, STATE_FIELDS, ticks)
from .nest_gpu_poisson import native_rng_identity
from .nest_gpu_live_control import (future_poisson_reset_tick, replace_masked_tc_future,
                                   reset_uniform_poisson_future)
from .provenance import sha256_file
from .spatial_connectivity import EDGE_DTYPE, SCHEMA, SOURCE_REVISION, native_delay_ticks, create_layout


@dataclass(frozen=True)
class Execution:
    edge_batch_size: int = 100000
    readback_sources_per_population: int = 3
    state_units_per_population: int = 2
    schedule: str = "effective"
    duration_ms: float | None = None
    background_seed: int | None = None
    initial_state_seed: int | None = None
    source_rng_library: str | None = None
    pulse_control: dict | None = None
    state_observation_ticks: list[int] | None = None
    expected_poisson_rng_family: str | None = None
    live_pulse_control: bool = False
    future_background_noise: dict | None = None

    def validate(self):
        for key in ("edge_batch_size", "readback_sources_per_population"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"positive integer {key} required")
        if type(self.state_units_per_population) is not int or self.state_units_per_population < 0:
            raise ValueError("nonnegative integer state sample size required")
        if type(self.live_pulse_control) is not bool:
            raise ValueError("boolean live pulse control required")
        if self.live_pulse_control and self.schedule != "effective":
            raise ValueError("live pulse control requires the effective schedule")
        if self.schedule not in ("initial", "effective"):
            raise ValueError("unknown pulse schedule")
        if self.duration_ms is not None and (isinstance(self.duration_ms, bool)
                or not math.isfinite(self.duration_ms) or self.duration_ms <= 0):
            raise ValueError("positive finite episode duration required")

        for key in ("background_seed", "initial_state_seed"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or not 0 < value < 2**31):
                raise ValueError(f"positive int31 {key} required")
        if self.initial_state_seed is not None and not self.source_rng_library:
            raise ValueError("source RNG library required for source-law initial-state resampling")
        if self.pulse_control is not None and not isinstance(self.pulse_control, dict):
            raise ValueError("source pulse-control dictionary required")
        if self.expected_poisson_rng_family not in (None,"per_edge_xorwow","per_edge_philox4_32_10"):
            raise ValueError("unknown declared native Poisson RNG family")
        if self.state_observation_ticks is not None:
            values = self.state_observation_ticks
            if (not isinstance(values, list) or not values
                    or any(type(t) is not int or t < 1 for t in values)
                    or any(a >= b for a,b in zip(values, values[1:]))):
                raise ValueError("ordered distinct positive observation ticks required")
        if self.future_background_noise is not None:
            noise = self.future_background_noise
            if (not isinstance(noise, dict) or set(noise) != {"boundary_time_ms", "seeds_per_population"}
                    or isinstance(noise["boundary_time_ms"], bool)
                    or not math.isfinite(noise["boundary_time_ms"]) or noise["boundary_time_ms"] <= 0
                    or not isinstance(noise["seeds_per_population"], list)
                    or len(noise["seeds_per_population"]) != 8
                    or any(type(seed) is not int or not 0 < seed < 2**31 for seed in noise["seeds_per_population"])
                    or len(set(noise["seeds_per_population"])) != 8):
                raise ValueError("future background noise requires a positive boundary and eight distinct int31 seeds")


def selected_ids(ids, labels, names, count):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([values[np.linspace(0, len(values)-1, min(count, len(values)), dtype=int)]
                           for name in names for values in [ids[labels == name]]])


def prepare_graph(manifest, layout, settings):
    settings.validate()
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "GENERATED"
            or manifest.get("source_revision") != SOURCE_REVISION
            or manifest.get("external_device_edges_included") is not False):
        raise ValueError("complete reviewed recurrent/TC source graph required")
    net, config = manifest["resolved_source_network"], manifest["source_config"]
    sim = dict(manifest.get("resolved_simulation", config["simulation"]))
    dt = float(sim["sim_resolution"])
    if settings.duration_ms is not None:
        if settings.duration_ms <= sim["t_presim"]:
            raise ValueError("episode must include source presimulation and recorded interval")
        sim["t_sim"] = settings.duration_ms - sim["t_presim"]
    duration = float(sim["t_presim"] + sim["t_sim"])
    ticks([duration], dt)
    if settings.future_background_noise is not None:
        if not net["poisson_input"]:
            raise ValueError("future background noise requires source Poisson input")
        boundary = int(ticks([settings.future_background_noise["boundary_time_ms"]], dt)[0])
        delay = int(native_delay_ticks([net["delay_poisson"]], dt)[0]) + POISSON_DELAY_OFFSET_TICKS
        if future_poisson_reset_tick(boundary, delay) >= int(ticks([duration], dt)[0]):
            raise ValueError("future background noise reset must precede episode completion")
    if net["neuron_model"] != "iaf_psc_exp" or net["dc_input"]:
        raise ValueError("only source iaf_psc_exp without additional DC devices supported")
    if net["thalamic_input"] and net["thalamic_input_type"] != "pulses":
        raise ValueError("TC folding requires deterministic external pulses without recurrent TC input")
    ids, labels, xy = (layout[k] for k in ("unit_ids", "populations", "positions_mm"))
    names = net["populations"]
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or labels.shape != ids.shape or xy.shape != (len(ids), 2)
            or not np.isfinite(xy).all() or len(names) != 9 or names[-1] != "TC"
            or not np.array_equal([np.sum(labels == name) for name in names], net["num_neurons"])
            or len(ids) != sum(net["num_neurons"])):
        raise ValueError("source unit identities, populations or positions invalid")
    extent = float(net["extent"])
    if np.any(np.abs(xy) > extent / 2):
        raise ValueError("source uniform layout must lie inside its periodic square")
    cortex, tc = ids[labels != "TC"], ids[labels == "TC"]
    vm = layout["cortical_initial_vm_mv"]
    if vm.shape != cortex.shape or not np.isfinite(vm).all():
        raise ValueError("finite unmodified cortical initial voltages required")
    p = net["neuron_params"]
    for key in ("C_m", "tau_m", "tau_syn_ex", "tau_syn_in", "t_ref"):
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError("invalid source neuronal constants")
    ticks([p["t_ref"]], dt)
    full = lambda value: np.full(len(cortex), value, dtype=float)
    mapped = {k: full(p[k]) for k in ("C_m", "tau_m", "E_L", "t_ref")}
    currents = np.empty(len(cortex))
    for i, name in enumerate(names[:-1]):
        currents[labels[labels != "TC"] == name] = net["DC_amp"][i]
    mapped.update(I_e=currents, V_m_rel=vm-p["E_L"], Theta_rel=full(p["V_th"]-p["E_L"]),
                  V_reset_rel=full(p["V_reset"]-p["E_L"]), tau_ex=full(p["tau_syn_ex"]),
                  tau_in=full(p["tau_syn_in"]), I_syn_ex=full(0), I_syn_in=full(0), den_delay=full(0))
    if not all(np.isfinite(v).all() for v in mapped.values()):
        raise ValueError("nonfinite mapped neuronal parameters")
    arrivals = np.empty(0, dtype=np.int64)
    stimulated = np.zeros(len(tc), dtype=bool)
    protocol = None
    if net["thalamic_input"]:
        radius = float(net["th_radius"])
        if not 0 < radius <= extent / 2:
            raise ValueError("only central source circular pulse masks up to half extent supported")
        protocol = prepare_pulse_protocol(net, sim,
            config.get("pulse_control") if settings.pulse_control is None else settings.pulse_control)
        delay = int(native_delay_ticks([net["th_delay_pulse_generator"]], dt)[0])
        if protocol is None:
            emission = np.arange(net["th_pulse_start"], duration, net["th_interval"]) - net["th_delay_pulse_generator"]
            emission = ticks(emission, dt)
        else:
            emission = np.asarray(protocol.initial_ticks if settings.schedule == "initial" or settings.live_pulse_control else protocol.effective_ticks,
                                  dtype=np.int64)
        if np.any(emission < 1) or np.any(np.diff(emission) <= 0):
            raise ValueError("positive ordered source pulse emissions required")
        arrivals = emission + delay
        stimulated = np.sum(xy[labels == "TC"]**2, axis=1) <= radius**2
    if settings.live_pulse_control and (protocol is None or protocol.decision_tick is None):
        raise ValueError("live pulse control requires a source protocol and decision tick")
    rates = float(net["bg_rate"]) * np.asarray(net["ext_indegrees"], dtype=float)
    if rates.shape != (8,) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("invalid source background rates")
    return dict(cortex=cortex, tc=tc, mapped=mapped, dt=dt, duration=duration,
                tc_schedules={int(uid): arrivals.copy() if active else np.empty(0, dtype=np.int64)
                              for uid, active in zip(tc, stimulated)}, rates=rates,
                readback=selected_ids(ids, labels, names, settings.readback_sources_per_population),
                state_ids=selected_ids(ids, labels, names[:-1], settings.state_units_per_population),
                pulse_protocol=protocol, stimulated_tc_mask=stimulated)


def translated_batch(edges, dt):
    if edges.dtype != EDGE_DTYPE or edges.ndim != 1:
        raise ValueError("invalid source edge dtype or shape")
    if not np.isfinite(edges["weight"]).all() or np.any(edges["source"] == edges["target"]):
        raise ValueError("invalid source edge weight or autapse")
    result = np.empty(len(edges), dtype=EDGE_READBACK_DTYPE)
    result["source"], result["target"] = edges["source"], edges["target"]
    result["port"] = edges["weight"] < 0
    result["weight"] = edges["weight"]
    if not np.isfinite(result["weight"]).all():
        raise ValueError("weight overflows native precision")
    result["delay_ticks"] = native_delay_ticks(edges["raw_delay"], dt)
    return result


def connect_batch(connector, rows, mapping, dt):
    source_ids, native_ids = mapping
    def native(values):
        take = np.searchsorted(source_ids, values)
        if np.any(take >= len(source_ids)) or not np.array_equal(source_ids[take], values):
            raise ValueError("unknown connection unit ID")
        return native_ids[take]
    connector.connect_arrays(native(rows["source"]), native(rows["target"]), rows["weight"],
                             rows["delay_ticks"]*dt, rows["port"])


def _read(ng, nodes, key):
    return np.asarray(ng.GetNeuronStatus(nodes, key)).reshape(-1)


def observe_continuous_state(ng, nodes, unit_ids, observation_ticks, dt, duration, output,
                             boundary_tick=None, boundary_action=None, *, noise_reset_tick=None,
                             noise_reset_action=None):
    """Use the public C step API once-started, matching the native Simulate loop.

    Each observation follows arrival accumulation at its tick. Reads do not
    restore, modify or advance state. This avoids the rejected repeated-Simulate
    path, which resets the native iteration index.
    """
    n_steps = int(ticks([duration],dt)[0])
    native_steps = int(np.rint(np.float32(duration)/np.float32(dt)))
    if n_steps != native_steps or (observation_ticks and observation_ticks[-1] > n_steps):
        raise ValueError("observations must fit the same native simulation clock")
    if boundary_tick is not None and (type(boundary_tick) is not int or not 1 <= boundary_tick < n_steps
                                      or boundary_action is None):
        raise ValueError("valid live boundary and action required")
    if noise_reset_tick is not None and (type(noise_reset_tick) is not int
            or not 1 <= noise_reset_tick < n_steps or noise_reset_action is None):
        raise ValueError("valid future noise reset tick and action required")
    library = ctypes.CDLL(os.environ["NESTGPU_LIB"])
    start = library.NESTGPU_StartSimulation; step = library.NESTGPU_SimulationStep
    for function in [start, step]:
        function.argtypes = []; function.restype = ctypes.c_int
    def checked(function):
        result = function()
        if result != 0 or ng.GetErrorCode() != 0:
            raise RuntimeError("native continuous step failed: "+str(ng.GetErrorMessage()))
    propagators = {k:_read(ng,nodes,k) for k in ("P11ex","P11in")} if observation_ticks else {}
    if any(not np.isfinite(v).all() or np.any(v < np.finfo(np.float32).tiny) or np.any(v > 1)
           for v in propagators.values()):
        raise ValueError("normal finite observation propagators required")
    requested = set(observation_ticks); records=[]
    boundary_receipt=None
    noise_receipt=None
    ng.SetSimTime(duration); checked(start)
    for tick in range(1,n_steps+1):
        checked(step)
        if tick == boundary_tick:
            boundary_receipt=boundary_action()
        if tick in requested:
            state={k:_read(ng,nodes,k) for k in STATE_FIELDS}
            if any(v.shape != unit_ids.shape or not np.isfinite(v).all() for v in state.values()):
                raise ValueError("invalid full cortical observation")
            filename=f"state_observation_{tick:09d}.npz"
            np.savez_compressed(output/filename,unit_ids=unit_ids,**state,**propagators)
            records.append(dict(tick=tick,time_ms=tick*dt,path=filename,sha256=sha256_file(output/filename)))
        if tick == noise_reset_tick:
            noise_receipt=noise_reset_action()
    receipt=dict(schema="mesocircuit_continuous_state_observations_v1",dt_ms=dt,
        native_phase="post-arrival after recorded neuronal update",duration_ms=duration,
        cortical_units=len(unit_ids),observations=records,
        execution="one StartSimulation followed by exactly the native number of SimulationStep calls",
        native_library_sha256=sha256_file(Path(os.environ["NESTGPU_LIB"])),full_hidden_state=False,
        live_pulse_control=boundary_receipt, future_background_noise=noise_receipt)
    (output/"state_observations.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def run(graph: Path, output: Path, settings: Execution):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(status="RUNNING", graph=str(graph.resolve()), config=asdict(settings),
                  scientific_reference_qualified=False, gpu_source_revision=GPU_REVISION,
                  execution="one continuous preloaded episode", slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  resumed=False, checks={})
    def checkpoint(phase):
        report.update(phase=phase, elapsed_seconds=time.perf_counter()-start,
                      peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: report[k] for k in ("phase", "elapsed_seconds", "peak_process_rss_mib")}), flush=True)
    try:
        manifest = json.loads((graph/"manifest.json").read_text())
        report["graph_manifest_sha256"] = sha256_file(graph/"manifest.json")
        if sha256_file(graph/"layout.npz") != manifest["layout_sha256"]:
            raise ValueError("source layout checksum mismatch")
        with np.load(graph/"layout.npz", allow_pickle=False) as f:
            layout = {k: f[k] for k in f.files}
        settings.validate()
        if settings.initial_state_seed is not None:
            library = Path(settings.source_rng_library)
            if sha256_file(library) != manifest["library_sha256"]:
                raise ValueError("initial-state RNG library differs from source graph generator")
            state_config = copy.deepcopy(manifest["source_config"])
            state_config["random_streams"]["initial_state"] = settings.initial_state_seed
            sampled = create_layout(library, manifest["resolved_source_network"], state_config)
            if any(not np.array_equal(sampled[k], layout[k]) for k in ("unit_ids", "populations", "positions_mm")):
                raise ValueError("initial-state resampling changed source layout")
            layout["cortical_initial_vm_mv"] = sampled["cortical_initial_vm_mv"]
            report["initial_state_seed"] = settings.initial_state_seed
            report["initial_state_rng_library_sha256"] = sha256_file(library)
        plan = prepare_graph(manifest, layout, settings)
        if (len({p["path"] for p in manifest["parts"]}) != len(manifest["parts"])
                or sum(p["edges"] for p in manifest["parts"]) != manifest["edges"]):
            raise ValueError("graph shard declaration/count mismatch")
        net, config = manifest["resolved_source_network"], manifest["source_config"]
        dt, duration = plan["dt"], plan["duration"]
        report.update(cortical_neurons=len(plan["cortex"]), thalamic_relays=len(plan["tc"]),
                      recurrent_tc_edges=manifest["edges"], duration_ms=duration, dt_ms=dt,
                      background_seed=(config["random_streams"]["background_drive"]
                                       if settings.background_seed is None else settings.background_seed),
                      poisson_native_delay_offset_ticks=POISSON_DELAY_OFFSET_TICKS,
                      readback_source_unit_ids=plan["readback"].tolist(),
                      state_unit_ids=plan["state_ids"].tolist())
        ng = importlib.import_module("nestgpu")
        report["native_library_sha256"] = sha256_file(Path(os.environ["NESTGPU_LIB"]))
        report["native_poisson_rng_family"] = native_rng_identity(Path(os.environ["NESTGPU_LIB"]))
        if (settings.expected_poisson_rng_family is not None
                and report["native_poisson_rng_family"] != settings.expected_poisson_rng_family):
            raise ValueError("native Poisson RNG family differs from configured source variant")
        ng.SetRandomSeed(report["background_seed"])
        ng.SetTimeResolution(dt); ng.SetKernelStatus("verbosity_level", 0)
        cortex = ng.Create("iaf_psc_exp", len(plan["cortex"]))
        mapping = {int(uid): int(cortex[i]) for i, uid in enumerate(plan["cortex"])}
        readback = {}
        for name, values in plan["mapped"].items():
            ng.SetStatus(cortex, name, {"array": values.tolist()})
            readback[name] = _read(ng, cortex, name)
        report["checks"]["initial_parameters_exact_fp32"] = all(
            np.array_equal(readback[k], v.astype(np.float32)) for k, v in plan["mapped"].items())
        np.savez(output/"initial_parameters.npz", unit_ids=plan["cortex"], **readback)
        if not report["checks"]["initial_parameters_exact_fp32"]:
            raise ValueError("initial parameter readback mismatch")
        capacity = math.ceil(duration / max(dt, float(np.min(plan["mapped"]["t_ref"])))) + 2
        ng.ActivateSpikeCount(cortex); ng.ActivateRecSpikeTimes(cortex, capacity)
        # One native group preserves individual generators without exhausting
        # NEST GPU's node-group table at large TC population sizes.
        tc_nodes = ng.Create("spike_generator", len(plan["tc"]))
        tc_capacity = max(len(v) for v in plan["tc_schedules"].values()) + 2
        if settings.live_pulse_control:
            tc_capacity=max(tc_capacity,len(plan["pulse_protocol"].effective_ticks)+2)
        ng.ActivateSpikeCount(tc_nodes); ng.ActivateRecSpikeTimes(tc_nodes, tc_capacity)
        for i, uid in enumerate(plan["tc"]):
            uid = int(uid)
            arrivals = plan["tc_schedules"][uid]
            ng.SetStatus([int(tc_nodes[i])], {"spike_times": ((arrivals-1)*dt).tolist()})
            mapping[uid] = int(tc_nodes[i])
        # Device IDs occupy a distinct source namespace after all recorded units.
        backgrounds = []
        if net["poisson_input"]:
            for i in range(8):
                group = ng.Create("poisson_generator", 1)
                ng.SetStatus(group, {"rate": float(plan["rates"][i])})
                uid = int(layout["unit_ids"].max()) + 1 + i
                mapping[uid] = int(group[0]); backgrounds.append(uid)
        np.savez(output/"unit_mapping.npz", source_ids=np.asarray(list(mapping)),
                 native_ids=np.asarray(list(mapping.values())))
        connector = ExplicitConnector(ng, maximum_batch_size=settings.edge_batch_size)
        source_ids = np.asarray(sorted(mapping), dtype=np.int64)
        array_mapping = (source_ids, np.asarray([mapping[int(u)] for u in source_ids], dtype=np.int32))
        sample = []
        labels_by_id = {int(u): str(p) for u, p in zip(layout["unit_ids"], layout["populations"])}
        imported = 0
        checkpoint("importing")
        import_start = time.perf_counter()
        for part_index, part in enumerate(manifest["parts"]):
            path = graph/part["path"]
            if path.parent.resolve() != graph.resolve() or sha256_file(path) != part["sha256"]:
                raise ValueError("source shard path/checksum mismatch")
            edges = np.load(path, mmap_mode="r", allow_pickle=False)
            if edges.dtype != EDGE_DTYPE or edges.shape != (part["edges"],):
                raise ValueError("source shard row count/type mismatch")
            if part["target_population"] == "TC":
                raise ValueError("TC folding cannot accept recurrent TC inputs")
            for begin in range(0, len(edges), settings.edge_batch_size):
                batch = edges[begin:begin+settings.edge_batch_size]
                if (any(labels_by_id.get(int(u)) != part["source_population"] for u in np.unique(batch["source"]))
                        or any(labels_by_id.get(int(u)) != part["target_population"] for u in np.unique(batch["target"]))):
                    raise ValueError("source shard population/identity mismatch")
                rows = translated_batch(batch, dt)
                connect_batch(connector, rows, array_mapping, dt)
                sample.append(rows[np.isin(rows["source"], plan["readback"])].copy())
                imported += len(rows)
            del edges
            if (part_index+1) % 50 == 0 or part_index+1 == len(manifest["parts"]):
                report.update(imported_shards=part_index+1, imported_edges=imported)
                checkpoint("importing")
        report["import_seconds"] = time.perf_counter()-import_start
        for i, uid in enumerate(backgrounds):
            targets = layout["unit_ids"][layout["populations"] == net["populations"][i]]
            rows = np.empty(len(targets), dtype=EDGE_READBACK_DTYPE)
            rows["source"], rows["target"], rows["port"] = uid, targets, 0
            rows["weight"] = net["weight_ext"]
            if not np.isfinite(rows["weight"]).all() or np.any(rows["weight"] < 0):
                raise ValueError("invalid source excitatory background weight")
            rows["delay_ticks"] = native_delay_ticks([net["delay_poisson"]], dt)[0] + POISSON_DELAY_OFFSET_TICKS
            connect_batch(connector, rows, array_mapping, dt); sample.append(rows)
        report.update(native_connections=connector.connection_count, background_edges=connector.connection_count-imported)
        if imported != manifest["edges"]:
            raise ValueError("incomplete source import")
        state_record = None
        if len(plan["state_ids"]):
            state_record = ng.CreateRecord("", [k for _ in plan["state_ids"] for k in STATE_FIELDS],
                [mapping[int(u)] for u in plan["state_ids"] for _ in STATE_FIELDS], [0]*(len(plan["state_ids"])*len(STATE_FIELDS)))
        checkpoint("calibrating")
        calibration_start = time.perf_counter(); ng.Calibrate()
        report["calibration_seconds"] = time.perf_counter()-calibration_start
        checkpoint("native_readback")
        # Pinned getConnections allocates source_count * target_count buffers.
        # This query samples source adjacency, including all background sources.
        native = ng.GetConnectionStatus(ng.GetConnections(
            source=[mapping[int(u)] for u in plan["readback"]]+[mapping[u] for u in backgrounds], target=cortex))
        inverse = {v:k for k,v in mapping.items()}
        actual = np.empty(len(native), dtype=EDGE_READBACK_DTYPE)
        for i, edge in enumerate(native):
            actual[i] = (inverse[edge["target"]], inverse[edge["source"]], edge["port"],
                         int(ticks([edge["delay"]], dt, recorded=True)[0]), edge["weight"])
        expected = np.concatenate(sample)
        expected.sort(order=list(EDGE_READBACK_DTYPE.names)); actual.sort(order=list(EDGE_READBACK_DTYPE.names))
        report["checks"]["sampled_complete_adjacencies_exact"] = np.array_equal(actual, expected)
        report["readback_edges"] = len(actual)
        report["readback_is_full_graph"] = len(actual) == connector.connection_count
        np.save(output/"expected_adjacencies.npy", expected, allow_pickle=False)
        np.save(output/"native_adjacencies.npy", actual, allow_pickle=False)
        if not report["checks"]["sampled_complete_adjacencies_exact"]:
            raise ValueError("native adjacency mismatch")
        del native, sample, expected, actual
        checkpoint("simulating")
        simulation_start = time.perf_counter()
        if (settings.state_observation_ticks is None and not settings.live_pulse_control
                and settings.future_background_noise is None):
            ng.Simulate(duration)
        else:
            protocol=plan["pulse_protocol"]
            action=(lambda:replace_masked_tc_future(ng,tc_nodes,plan["stimulated_tc_mask"].tolist(),protocol)) if settings.live_pulse_control else None
            noise_tick=noise_action=None
            if settings.future_background_noise is not None:
                noise=settings.future_background_noise
                noise_boundary=int(ticks([noise["boundary_time_ms"]],dt)[0])
                noise_delay=int(native_delay_ticks([net["delay_poisson"]],dt)[0])+POISSON_DELAY_OFFSET_TICKS
                noise_tick=future_poisson_reset_tick(noise_boundary,noise_delay)
                def noise_action():
                    receipts=[]
                    for index,uid in enumerate(backgrounds):
                        targets=layout["unit_ids"][layout["populations"]==net["populations"][index]]
                        native_targets=[mapping[int(target)] for target in targets]
                        receipts.append(reset_uniform_poisson_future(ng,mapping[uid],native_targets,
                            noise_boundary,noise_delay,noise_tick,noise["seeds_per_population"][index],dt))
                    return receipts
            observation = observe_continuous_state(ng,cortex,plan["cortex"],settings.state_observation_ticks or [],
                dt,duration,output,protocol.decision_tick if settings.live_pulse_control else None,action,
                noise_reset_tick=noise_tick,noise_reset_action=noise_action)
            report["state_observation_ticks"] = [r["tick"] for r in observation["observations"]]
            if settings.live_pulse_control:
                report["execution"]="one continuous episode with future-only native device control"
                report["pulse_control"]=dict(protocol.record(),boundary_phase="after the continuous native decision-tick step; before device schedule mutation")
                report["native_live_control"]=observation["live_pulse_control"]
                effective=np.asarray(protocol.effective_ticks,dtype=np.int64)+protocol.delay_ticks
                for uid,active in zip(plan["tc"],plan["stimulated_tc_mask"]):
                    plan["tc_schedules"][int(uid)]=effective.copy() if active else np.empty(0,dtype=np.int64)
            if settings.future_background_noise is not None:
                report["future_background_noise"]=observation["future_background_noise"]
        report["simulation_seconds"] = time.perf_counter()-simulation_start
        checkpoint("recording")
        # Read once; the native call drains recording buffers.
        trains = list(ng.GetRecSpikeTimes(cortex)) + list(ng.GetRecSpikeTimes(tc_nodes))
        units = np.concatenate([plan["cortex"], plan["tc"]])
        native_counts = np.concatenate([_read(ng, cortex, "spike_count"),
                                        _read(ng, tc_nodes, "spike_count")]).astype(np.int64)
        counts = np.asarray([len(train) for train in trains], dtype=np.int64)
        np.savez(output/"spike_counts.npz", unit_ids=units, native_counts=native_counts, recorded_counts=counts)
        report["checks"]["spike_count_recording_complete"] = np.array_equal(native_counts, counts)
        report["checks"]["tc_emissions_exact"] = all(np.array_equal(ticks(train, dt, recorded=True),
            plan["tc_schedules"][int(uid)][plan["tc_schedules"][int(uid)]*dt <= duration])
            for uid, train in zip(plan["tc"], trains[len(plan["cortex"]):]))
        event_ids = np.repeat(units, counts)
        raw_times = np.concatenate([np.asarray(train, dtype=float) for train in trains])
        event_ticks = ticks(raw_times, dt, recorded=True)
        order = np.lexsort((event_ids, event_ticks))
        np.savez_compressed(output/"spikes.npz", unit_ids=layout["unit_ids"], populations=layout["populations"],
            positions_mm=layout["positions_mm"], spike_unit_ids=event_ids[order],
            raw_spike_times_ms=raw_times[order], spike_ticks=event_ticks[order], dt_ms=dt)
        if state_record is not None:
            raw = np.asarray(ng.GetRecordData(state_record), dtype=float)
            clock = ticks(raw[:,0], dt, recorded=True)
            if not np.array_equal(clock, np.arange(int(round(duration/dt))+1)) or not np.isfinite(raw).all():
                raise ValueError("invalid sampled-state recording")
            np.savez_compressed(output/"states.npz", ticks=clock, raw_times_ms=raw[:,0],
                unit_ids=plan["state_ids"], fields=np.asarray(STATE_FIELDS),
                states=raw[:,1:].reshape(len(raw), len(plan["state_ids"]), len(STATE_FIELDS)))
        terminal = {k: _read(ng, cortex, k) for k in STATE_FIELDS}
        report["checks"]["terminal_states_finite"] = all(np.isfinite(v).all() for v in terminal.values())
        np.savez(output/"terminal_states.npz", unit_ids=plan["cortex"], **terminal)
        report.update(spike_count=len(event_ids), cortical_spike_count=int(counts[:len(plan["cortex"])].sum()),
                      status="EXECUTED" if all(report["checks"].values()) else "FAILED")
    except Exception as error:
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report["implementation_sha256"] = sha256_file(Path(__file__))
    report["artifacts"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "report.json"}
    checkpoint("finished")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Execution(**yaml.safe_load(args.config.read_text()))
    report = run(args.graph, args.output, settings)
    print(json.dumps({k: report.get(k) for k in ("status", "checks", "error", "elapsed_seconds")}))
    raise SystemExit(int(report["status"] != "EXECUTED"))


if __name__ == "__main__":
    main()

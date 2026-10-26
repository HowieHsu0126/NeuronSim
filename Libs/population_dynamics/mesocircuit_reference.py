"""Pinned spatial Mesocircuit reference with native thalamic and DC controls.

Native dynamics/parameters remain upstream. Runtime compatibility is confined to
this source module. Operational runs do not qualify full-scale scientific claims.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from .mesocircuit_source import load_config, verify_source, derive_parameters
from .mesocircuit_normalization import normalization_record
from .mesocircuit_continuation import (prepare_pulse_protocol, replace_future_pulses,
                                      configure_update_interval, check_update_segments)
from .nest_reference import fingerprint_connections, _set_rng_phase
from .network_snapshot import export_initial_network
from .provenance import freeze_json, sha256_file, runtime_manifest


class _SourceNumpyConstants:
    """Preserve the upstream meanings of removed NumPy constant aliases."""
    Inf = np.inf
    NINF = -np.inf

    def __getattr__(self, name):
        return getattr(np, name)


class _SingleRankSpatialBinding:
    """Keep the source layer metadata when every node belongs to this rank.

    NEST 3.10 GetLocalNodeCollection loses spatial metadata in the inspected
    upstream writer. Returning the original collection is equivalent only for
    one rank; this binding must never be used as a distributed fallback.
    """
    def __init__(self, native, mpi):
        if native.NumProcesses() != 1 or mpi.COMM_WORLD.size != 1:
            raise ValueError("MESOCIRCUIT_SINGLE_RANK_BINDING_REQUIRES_ONE_RANK")
        object.__setattr__(self, "_native", native)

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __setattr__(self, name, value):
        setattr(self._native, name, value)

    def GetLocalNodeCollection(self, nodes):
        if self._native.NumProcesses() != 1:
            raise ValueError("MESOCIRCUIT_RANK_COUNT_CHANGED")
        return nodes


def load_native_network(source: Path):
    # NEST and MPI are optional runtime dependencies; metadata commands stay light.
    import nest
    from mpi4py import MPI
    source = source.resolve()
    prior = sys.modules.get("mesocircuit")
    if prior is not None and Path(prior.__file__).resolve() != source / "mesocircuit/__init__.py":
        raise ValueError("DIFFERENT_MESOCIRCUIT_SOURCE_ALREADY_IMPORTED")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("mesocircuit.simulation.network")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != source / "mesocircuit/simulation/network.py":
        raise ValueError("MESOCIRCUIT_NATIVE_MODULE_PATH_MISMATCH")
    module.np = _SourceNumpyConstants()
    module.nest = _SingleRankSpatialBinding(nest, MPI)
    return module, nest


@contextmanager
def _fresh_native_directory(output: Path):
    # The upstream constructor wipes CWD/raw_data. Only a newly created output
    # may be its working directory, so existing experiment files cannot be wiped.
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw_data").mkdir()
    previous = Path.cwd()
    os.chdir(output)
    try:
        yield
    finally:
        os.chdir(previous)


def _native_controls(network, nest, net: dict[str, Any], sim: dict[str, Any], pulse_protocol=None):
    report: dict[str, Any] = {"thalamic_input": net["thalamic_input"], "dc_input": net["dc_input"]}
    arrays: dict[str, np.ndarray] = {}
    if net["thalamic_input"] and net["thalamic_input_type"] == "pulses":
        generator = network.spike_pulse_input_th
        edges = nest.GetConnections(source=generator, target=network.pops[-1])
        fields = edges.get(["source", "target", "weight", "delay"])
        arrays.update({"pulse_" + key: np.asarray(value) for key, value in fields.items()})
        arrays["pulse_emission_ms"] = np.asarray(generator.spike_times, dtype=float)
        arrivals = np.arange(net["th_pulse_start"], sim["t_presim"] + sim["t_sim"], net["th_interval"])
        expected = arrivals - net["th_delay_pulse_generator"]
        if pulse_protocol is not None:
            expected = pulse_protocol.milliseconds(pulse_protocol.initial_ticks)
            arrivals = expected + net["th_delay_pulse_generator"]
        if not np.allclose(arrays["pulse_emission_ms"], expected, rtol=0, atol=1e-10):
            raise ValueError("NATIVE_PULSE_SCHEDULE_READBACK_MISMATCH")
        if not np.allclose(arrays["pulse_delay"], net["th_delay_pulse_generator"], rtol=0, atol=1e-10):
            raise ValueError("NATIVE_PULSE_DELAY_READBACK_MISMATCH")
        arrays["expected_tc_arrivals_ms"] = arrivals
        report.update(thalamic_type="pulses", target_count=len(fields["target"]),
                      radius_mm=net["th_rel_radius"] * net["extent"], center_mm=[0., 0.],
                      pulse_arrivals_ms=arrivals.tolist(), pulse_emission_ms=expected.tolist())
    elif net["thalamic_input"]:
        readback = network.poisson_input_th.get(["rate", "start", "stop"])
        expected = {"rate": net["th_rate"], "start": net["th_start"], "stop": net["th_start"] + net["th_duration"]}
        for key in expected:
            if not np.allclose(readback[key], expected[key], rtol=0, atol=1e-10):
                raise ValueError("NATIVE_THALAMIC_POISSON_READBACK_MISMATCH")
        report.update(thalamic_type="poisson", poisson=expected)
    if net["dc_input"]:
        readback = network.dc_stim_input.get(["amplitude", "start", "stop"])
        expected_amplitude = net["dc_amp"] * net["K_ext_" + net["base_model"]]
        for key, expected in (("amplitude", expected_amplitude), ("start", net["dc_start"]), ("stop", net["dc_start"] + net["dc_dur"])):
            if not np.allclose(readback[key], expected, rtol=0, atol=1e-10):
                raise ValueError("NATIVE_DC_DEVICE_READBACK_MISMATCH")
        report["dc_device"] = {"amplitude_pa": expected_amplitude.tolist(), "start_ms": net["dc_start"], "stop_ms": net["dc_start"] + net["dc_dur"],
                               "semantics": "source dc_amp multiplied by unscaled source K_ext; device parameters, not membrane-current validation"}
    return report, arrays


def _independent_initial_voltages(network, net, seed):
    """Sample the native cortical Gaussian laws with a separate PCG64 stream."""
    rng = np.random.Generator(np.random.PCG64(seed))
    kind = net["V0_type"]
    mean = net["neuron_params"]["V0_mean"][kind]
    std = net["neuron_params"]["V0_std"][kind]
    expected = []
    for i, pop in enumerate(network.pops[:-1]):
        mu = mean if kind == "original" else mean[i]
        sigma = std if kind == "original" else std[i]
        values = rng.normal(mu, sigma, len(pop))
        pop.set(V_m=values)
        if not np.array_equal(np.asarray(pop.get("V_m")), values):
            raise ValueError("INDEPENDENT_INITIAL_VOLTAGE_READBACK_MISMATCH")
        expected.append(values)
    return np.concatenate(expected)


def run_reference(config: dict[str, Any], source: Path, output: Path,
                  command: list[str]) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    source, output = source.resolve(), output.resolve()
    provenance = verify_source(source)
    net, sim = derive_parameters(source, config)
    pulse_protocol = prepare_pulse_protocol(net, sim, config.get("pulse_control"))
    normalization_policy = config.get("connectivity_normalization", "native_source")
    normalization = normalization_record(net, normalization_policy)
    native, nest = load_native_network(source)
    if nest.__version__ != config["expected_nest_version"]:
        raise ValueError("MESOCIRCUIT_NEST_VERSION_MISMATCH")
    timings = {}
    with _fresh_native_directory(output):
        circuit = SimpleNamespace(data_dir_circuit=str(output), net_dict=net, sim_dict=sim)
        network = native.Network(circuit, config["local_num_threads"])
        update_policy = config.get("kernel_update_interval", "native")
        kernel_update = configure_update_interval(nest, net, sim, update_policy)
        streams = config.get("random_streams")
        rng_phases = [{"phase": "spatial_layout" if streams else "native_single_stream",
                       "seed": int(nest.rng_seed), "native_time_ms": float(nest.biological_time)}]
        tick = time.perf_counter(); network.create(); timings["create_seconds"] = time.perf_counter() - tick
        ids = np.concatenate([np.asarray(pop.tolist(), dtype=np.int64) for pop in network.pops])
        labels = np.concatenate([np.full(len(pop), str(label)) for pop, label in zip(network.pops, net["populations"])])
        xy = np.concatenate([np.asarray(nest.GetPosition(pop), dtype=np.float64) for pop in network.pops])
        initial_voltage = (_independent_initial_voltages(network, net, streams["initial_state"])
                           if streams else np.concatenate([np.asarray(pop.get("V_m"), dtype=float)
                                                           for pop in network.pops[:-1]]))
        if streams:
            rng_phases.append({"phase": "initial_state", "seed": streams["initial_state"],
                               "engine": "numpy.PCG64", "scope": "native cortical Gaussian V_m only"})
            rng_phases.append(_set_rng_phase(nest, streams["connectivity"], "connectivity"))
        if (len(ids) != sum(net["num_neurons"]) or len(np.unique(ids)) != len(ids)
                or xy.shape != (len(ids), 2) or not np.isfinite(xy).all()
                or np.any(np.abs(xy) > net["extent"] / 2)):
            raise ValueError("NATIVE_SPATIAL_IDENTITY_OR_EXTENT_MISMATCH")
        tick = time.perf_counter(); network.connect(); timings["connect_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter(); connectivity = fingerprint_connections(nest, ids, 32); timings["connectivity_readback_seconds"] = time.perf_counter() - tick
        try:
            check_update_segments(nest, sim, pulse_protocol, kernel_update)
        except Exception as exc:
            kernel_update.update(runtime_status="REJECTED_BEFORE_TIME_ADVANCE", native_time_ms=float(nest.biological_time), error=str(exc))
            freeze_json(Path("kernel_update.json"), kernel_update)
            raise
        kernel_update["physical_input_delay_min_ms"] = connectivity["delay_min_ms"]
        kernel_update["physical_input_delay_max_ms"] = connectivity["delay_max_ms"]
        freeze_json(Path("kernel_update.json"), kernel_update)
        if pulse_protocol is not None and pulse_protocol.initial_override:
            network.spike_pulse_input_th.set(spike_times=pulse_protocol.milliseconds(pulse_protocol.initial_ticks).tolist())
        controls, control_arrays = _native_controls(network, nest, net, sim, pulse_protocol)
        connected_voltage = np.concatenate([np.asarray(pop.get("V_m"), dtype=float) for pop in network.pops[:-1]])
        if not np.array_equal(connected_voltage, initial_voltage):
            raise ValueError("CONNECTIVITY_CREATION_CHANGED_INITIAL_VOLTAGES")
        if streams:
            rng_phases.append(_set_rng_phase(nest, streams["background_drive"], "background_drive"))
        snapshot_files = []
        if config.get("export_initial_network", False):
            snapshot_files = export_initial_network(network, nest, net, sim, config, provenance,
                ids, labels, xy, connectivity, pulse_protocol, Path("initial_network"))
        tick = time.perf_counter(); network.presimulate(sim["t_presim"]); timings["presimulation_seconds"] = time.perf_counter() - tick
        extra_files = ["kernel_update.json", *snapshot_files]
        protocol_record = None
        remaining_ms = sim["t_sim"]
        if pulse_protocol is not None:
            protocol_record = pulse_protocol.record()
            if pulse_protocol.decision_tick is not None:
                boundary = pulse_protocol.decision_time_ms
                remaining_ms = sim["t_presim"] + sim["t_sim"] - boundary
                prefix_ms = boundary - sim["t_presim"]
                tick = time.perf_counter()
                if prefix_ms > 0:
                    nest.data_prefix = "sim_"
                    nest.Simulate(prefix_ms)
                timings["predecision_simulation_seconds"] = time.perf_counter() - tick
                cortical_ids = ids[labels != "TC"]
                boundary_voltage = np.concatenate([np.asarray(pop.get("V_m"), dtype=float) for pop in network.pops[:-1]])
                if boundary_voltage.shape != cortical_ids.shape or not np.isfinite(boundary_voltage).all():
                    raise ValueError("INVALID_NATIVE_BOUNDARY_VOLTAGES")
                captured = [nest.GetStatus(recorder)[0]["events"] for recorder in network.spike_recorders]
                captured_senders = np.concatenate([np.asarray(row["senders"], dtype=np.int64) for row in captured])
                captured_times = np.concatenate([np.asarray(row["times"], dtype=np.float64) for row in captured])
                captured_order = np.lexsort((captured_senders, captured_times))
                captured_senders, captured_times = captured_senders[captured_order], captured_times[captured_order]
                np.savez_compressed("boundary_state.npz", cortical_unit_ids=cortical_ids,
                    cortical_voltage_mv=boundary_voltage, captured_spike_senders=captured_senders,
                    captured_spike_times_s=captured_times / 1000,
                    capture_time_ms=np.asarray(float(nest.biological_time)))
                extra_files.append("boundary_state.npz")
                protocol_record["state_readout"] = "cortical V_m only; privileged diagnostic, not an observed model input"
                protocol_record["unexported_state"] = ["synaptic currents", "refractory counters", "pending event queues", "RNG internal states"]
                protocol_record["captured_observation_namespaces"] = "all recorded units including TC; cortical state IDs are separate"
                protocol_record["captured_timestamp_at_boundary"] = "retained in captured readout; excluded from strict observed history"
                try:
                    protocol_record.update(replace_future_pulses(network.spike_pulse_input_th, nest, pulse_protocol))
                except Exception as exc:
                    protocol_record.update(runtime_status="FAILED", error_type=type(exc).__name__, error=str(exc))
                    freeze_json(Path("pulse_protocol.json"), protocol_record)
                    raise
            effective = pulse_protocol.milliseconds(pulse_protocol.effective_ticks)
            control_arrays["initial_pulse_emission_ms"] = control_arrays["pulse_emission_ms"].copy()
            control_arrays["pulse_emission_ms"] = effective
            control_arrays["expected_tc_arrivals_ms"] = effective + net["th_delay_pulse_generator"]
            controls.update(pulse_emission_ms=effective.tolist(),
                pulse_arrivals_ms=control_arrays["expected_tc_arrivals_ms"].tolist())
            extra_files.append("pulse_protocol.json")
        tick = time.perf_counter(); network.simulate(remaining_ms); timings["simulation_and_native_export_seconds"] = time.perf_counter() - tick
        events = [nest.GetStatus(recorder)[0]["events"] for recorder in network.spike_recorders]
        senders = np.concatenate([np.asarray(row["senders"], dtype=np.int64) for row in events])
        times_ms = np.concatenate([np.asarray(row["times"], dtype=np.float64) for row in events])
        order = np.lexsort((senders, times_ms)); senders, times_ms = senders[order], times_ms[order]
        if not np.isfinite(times_ms).all() or not np.isin(senders, ids).all() or np.any(times_ms <= 0) or np.any(times_ms > sim["t_presim"] + sim["t_sim"]):
            raise ValueError("NATIVE_SPIKE_IDENTITY_OR_TIME_MISMATCH")
        if protocol_record is not None:
            if pulse_protocol.decision_tick is not None:
                event_ticks = np.rint(times_ms / pulse_protocol.resolution_ms)
                captured_ticks = np.rint(captured_times / pulse_protocol.resolution_ms)
                before = event_ticks < pulse_protocol.decision_tick
                captured_before = captured_ticks < pulse_protocol.decision_tick
                protocol_record["captured_history_matches_completed_run"] = bool(
                    np.array_equal(captured_senders[captured_before], senders[before])
                    and np.array_equal(captured_times[captured_before], times_ms[before]))
                protocol_record["captured_spike_count"] = len(captured_times)
                protocol_record["strict_history_spike_count"] = int(captured_before.sum())
                if np.any(captured_ticks > pulse_protocol.decision_tick):
                    raise ValueError("CAPTURED_HISTORY_CONTAINS_FUTURE_EVENTS")
            protocol_record["runtime_status"] = "EXECUTED_AWAITING_INDEPENDENT_REPLAY"
            freeze_json(Path("pulse_protocol.json"), protocol_record)
        if controls.get("thalamic_type") == "pulses":
            target_ids = control_arrays["pulse_target"]
            for uid in ids[labels == "TC"]:
                expected = control_arrays["expected_tc_arrivals_ms"] if uid in target_ids else np.empty(0)
                actual = times_ms[senders == uid]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-10):
                    raise ValueError("THALAMIC_PULSE_DELIVERY_REPLAY_MISMATCH")
            controls["tc_pulse_delivery_replayed"] = True
        np.savez_compressed("spikes.npz", unit_ids=ids, populations=labels, positions_mm=xy,
                            spike_senders=senders, spike_times_s=times_ms / 1000,
                            initial_cortical_voltage_mv=initial_voltage)
        np.savez_compressed("controls.npz", **control_arrays)
        freeze_json(Path("controls.json"), controls)
        freeze_json(Path("connectivity_parameters.json"), normalization)
        summary = {"status": "PARTIAL", "scope": config["run_class"], "source_revision": provenance["revision"],
                   "unit_count": len(ids), "cortical_unit_count": int(np.count_nonzero(labels != "TC")),
                   "thalamic_unit_count": int(np.count_nonzero(labels == "TC")), "spike_count": len(times_ms),
                   "spatial_coordinates": "source-generated 2D positions in mm", "boundary": "periodic square",
                   "extent_mm": net["extent"], "base_model": net["base_model"], "connect_method": net["connect_method"],
                   "source_population_counts": net["num_neurons"].tolist(),
                   "analysis_start_s": sim["t_presim"] / 1000, "simulation_stop_s": (sim["t_presim"] + sim["t_sim"]) / 1000,
                   "presimulation_spikes_retained": True, "native_rng_seed": sim["rng_seed"],
                   "rng_mode": "adapter_separate_phases" if streams else "native_single_stream",
                   "rng_phases": rng_phases,
                   "random_streams_independently_separated": streams is not None,
                   "native_single_stream_numerical_sequence_preserved": streams is None and normalization_policy == "native_source" and pulse_protocol is None and update_policy == "native",
                   "kernel_update_interval": update_policy,
                   "kernel_update_record": "kernel_update.json",
                   "connectivity_normalization": normalization_policy,
                   "native_source_probabilities_preserved": normalization_policy == "native_source",
                   "initial_state_readout": "cortical membrane voltages only; not a complete post-presimulation microscopic state",
                   "connectivity": connectivity, "stage_costs": timings,
                   "runtime_scope": "run_reference including source verification, derivation and native import; full CLI time belongs to the external execution receipt",
                   "runtime_seconds": time.perf_counter() - started, "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "runtime_compatibility": {"numpy_constants": ["NINF=-inf", "Inf=+inf"], "local_node_collection": "original spatial collection at one rank only", "source_files_modified": False, "distributed_runtime_qualified": False},
                   "native_nest_version": nest.__version__, "scientific_reference_qualified": False,
                   "large_scale_fidelity_qualified": False, "inverse_intervention_qualified": False}
        if protocol_record is not None:
            summary["pulse_protocol"] = "pulse_protocol.json"
            summary["continuation_runtime_executed"] = pulse_protocol.decision_tick is not None
        freeze_json(Path("summary.json"), summary)
        files = ["spikes.npz", "controls.npz", "controls.json", "connectivity_parameters.json", "summary.json", "raw_data/positions.h5", "raw_data/spike_recorder.h5", "raw_data/" + sim["fname_nodeids"]] + extra_files
        freeze_json(Path("manifest.json"), {"schema": "mesocircuit_spatial_reference_v1", "source_manifest_sha256": sha256_file(source / "source.json"),
                    "artifacts": {name: sha256_file(Path(name)) for name in files}, **runtime_manifest(config, command, 2)})
    return summary, 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        summary, code = run_reference(config, Path(config["source_dir"]), args.output, sys.argv)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

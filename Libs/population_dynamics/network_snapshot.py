"""Lossless initial-network snapshots for cross-backend source qualification.

The snapshot is taken before time advances. It keeps native IDs, geometry,
signed multiedges, device semantics, and actual neuronal parameters. NPY arrays
support bounded memory access without introducing a GPU-side HDF5 dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .nest_reference import EDGE_DTYPE
from .provenance import sha256_file

NEURON_PARAMETERS = ("C_m", "tau_m", "tau_syn_ex", "tau_syn_in", "E_L", "V_th",
                     "V_reset", "t_ref", "I_e", "V_m")
SCHEMA = "mesocircuit_native_initial_network_v1"


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value == math.inf:
            return "positive_infinity"
        if value == -math.inf:
            return "negative_infinity"
        raise ValueError("NaN source metadata cannot be exported")
    return value


def _devices(nodes, model, names):
    ids = nodes.tolist()
    values = nodes.get(list(names))
    columns = {key: np.atleast_1d(values[key]) for key in names}
    if any(v.shape != (len(ids),) for v in columns.values()):
        raise ValueError("native device parameter cardinality mismatch")
    return [{"id": int(uid), "model": model,
             "parameters": {key: _plain(values[i]) for key, values in columns.items()}}
            for i, uid in enumerate(ids)]


def export_initial_network(network, nest, net, sim, config, provenance, ids, labels,
                           positions, connectivity, pulse_protocol, output: Path) -> list[str]:
    if float(nest.biological_time) != 0.:
        raise ValueError("initial network must be exported before time advances")
    output.mkdir(parents=True, exist_ok=False)
    ids = np.asarray(ids, dtype=np.int64)
    labels = np.asarray(labels)
    cortex = labels != "TC"
    values = []
    for pop in network.pops[:-1]:
        fields = pop.get(list(NEURON_PARAMETERS))
        columns = [np.atleast_1d(fields[name]).astype(float) for name in NEURON_PARAMETERS]
        if any(column.shape != (len(pop),) for column in columns):
            raise ValueError("native cortical parameter cardinality mismatch")
        values.append(np.column_stack(columns))
    parameters = np.concatenate(values)
    if parameters.shape != (int(cortex.sum()), len(NEURON_PARAMETERS)):
        raise ValueError("native cortical parameter/ID mismatch")
    devices = []
    if net["poisson_input"]:
        devices.extend(_devices(network.poisson_bg_input, "poisson_generator",
                                ("rate", "origin", "start", "stop")))
    if net["thalamic_input"]:
        if net["thalamic_input_type"] == "pulses":
            pulse = network.spike_pulse_input_th
            effective = (pulse_protocol.milliseconds(pulse_protocol.effective_ticks).tolist()
                         if pulse_protocol is not None else list(pulse.spike_times))
            devices.append({"id": int(pulse.tolist()[0]), "model": "spike_generator",
                            "parameters": {"initial_spike_times_ms": list(pulse.spike_times),
                                           "effective_spike_times_ms": effective}})
        else:
            devices.extend(_devices(network.poisson_input_th, "poisson_generator",
                                    ("rate", "origin", "start", "stop")))
    if net["dc_input"]:
        devices.extend(_devices(network.dc_stim_input, "dc_generator",
                                ("amplitude", "origin", "start", "stop")))
    valid_sources = np.asarray([*ids, *(r["id"] for r in devices)], dtype=np.int64)
    if len(np.unique(valid_sources)) != len(valid_sources):
        raise ValueError("unit/device identity collision")
    count = int(connectivity["connection_count"])
    edges = np.lib.format.open_memmap(output / "edges.npy", mode="w+", dtype=EDGE_DTYPE,
                                     shape=(count,))
    digest, cursor = hashlib.sha256(), 0
    batch = connectivity["target_batch_size"]
    for begin in range(0, len(ids), batch):
        conn = nest.GetConnections(target=nest.NodeCollection(ids[begin:begin+batch].tolist()),
                                   synapse_model="static_synapse")
        fields = conn.get(["source", "target", "weight", "delay"])
        chunk = np.empty(len(fields["source"]), dtype=EDGE_DTYPE)
        for name in chunk.dtype.names:
            chunk[name] = fields[name]
        if (not np.isfinite(chunk["weight"]).all() or not np.isfinite(chunk["delay"]).all()
                or not np.isin(chunk["source"], valid_sources).all()
                or not np.isin(chunk["target"], ids).all()):
            raise ValueError("unknown input device or invalid native connection")
        chunk.sort(order=["target", "source", "weight", "delay"])
        if cursor + len(chunk) > count:
            raise ValueError("native connections changed while exporting")
        edges[cursor:cursor+len(chunk)] = chunk
        digest.update(chunk.tobytes())
        cursor += len(chunk)
    edges.flush()
    del edges
    if cursor != count or digest.hexdigest() != connectivity["sha256"]:
        raise ValueError("snapshot does not reproduce the native connectivity fingerprint")
    arrays = {"unit_ids.npy": ids, "populations.npy": labels,
              "positions_mm.npy": np.asarray(positions, dtype=float),
              "cortical_parameters.npy": parameters}
    for name, value in arrays.items():
        np.save(output/name, value, allow_pickle=False)
    metadata = {
        "schema": SCHEMA, "source_revision": provenance["revision"],
        "native_time_ms": 0., "dt_ms": sim["sim_resolution"],
        "duration_ms": sim["t_presim"] + sim["t_sim"],
        "analysis_start_ms": sim["t_presim"], "extent_mm": net["extent"],
        "boundary": "periodic_square", "cortical_neuron_model": net["neuron_model"],
        "thalamic_neuron_model": "parrot_neuron",
        "parameter_columns": list(NEURON_PARAMETERS), "devices": devices,
        "connectivity": connectivity, "config": config,
        "resolved_source_network": net,
        "control_execution": "effective preloaded schedule is a separate replay of native future-only control",
        "scientific_reference_qualified": False,
    }
    (output/"metadata.json").write_text(json.dumps(_plain(metadata), indent=2, allow_nan=False)+"\n")
    names = ["edges.npy", *arrays, "metadata.json"]
    manifest = {"schema": SCHEMA, "artifacts": {name: sha256_file(output/name) for name in names}}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return [str(Path(output.name)/name) for name in [*names, "manifest.json"]]


def load_initial_network(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((path/"manifest.json").read_text())
    names = {"edges.npy", "unit_ids.npy", "populations.npy", "positions_mm.npy",
             "cortical_parameters.npy", "metadata.json"}
    if manifest.get("schema") != SCHEMA or set(manifest.get("artifacts", {})) != names:
        raise ValueError("initial-network manifest schema mismatch")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(path/name) != digest:
            raise ValueError("initial-network snapshot changed: "+name)
    metadata = json.loads((path/"metadata.json").read_text())
    if metadata["schema"] != SCHEMA or metadata["native_time_ms"] != 0:
        raise ValueError("not a native initial-network snapshot")
    arrays = {name[:-4]: np.load(path/name, mmap_mode="r", allow_pickle=False)
              for name in names if name.endswith(".npy")}
    ids = arrays["unit_ids"]
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("sorted unique native unit IDs required")
    if (arrays["populations"].shape != ids.shape
            or arrays["positions_mm"].shape != (len(ids), 2)
            or arrays["cortical_parameters"].shape !=
               (int(np.count_nonzero(arrays["populations"] != "TC")), len(NEURON_PARAMETERS))
            or arrays["edges"].dtype != EDGE_DTYPE
            or len(arrays["edges"]) != metadata["connectivity"]["connection_count"]):
        raise ValueError("initial-network array schema mismatch")
    if metadata["parameter_columns"] != list(NEURON_PARAMETERS):
        raise ValueError("unknown native parameter column order")
    return metadata, arrays

"""Original NEST3 replay with explicit per-target background multiplicities.

Source neural and deterministic input edges are retained. Each original
background edge is represented by a target-specific spike_generator whose
integer multiplicities encode the supplied arrival-count realization.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .network_snapshot import load_initial_network, NEURON_PARAMETERS
from .provenance import sha256_file

SOURCE_REVISION = "f59ada6b3c3efb1fa18f96c638dd1741ebfb77e5"
NEST_VERSION = "3.10.0"
DRIVE_SCHEMA = "mesocircuit_background_arrivals_v1"
QUANTIZATIONS = ("source_float64", "gpu_fp32_inputs")
STATE_FIELDS = ("V_m", "I_syn_ex", "I_syn_in")


def _ticks(values, dt):
    values = np.asarray(values, dtype=np.float64)
    if not math.isfinite(dt) or dt <= 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative native grid times required")
    result = np.rint(values / dt)
    if np.any(result >= 2**63):
        raise ValueError("native tick integer overflow")
    radius = 4 * np.finfo(np.float64).eps * np.maximum(np.maximum(values, result * dt), dt)
    if np.any(radius >= dt / 2) or np.any(np.abs(values - result * dt) > radius):
        raise ValueError("time is not on the native grid")
    return result.astype(np.int64)


def quantize_parameters(values, mode):
    if mode not in QUANTIZATIONS:
        raise ValueError("unknown quantization mode")
    original = {name: np.asarray(values[name], dtype=np.float64) for name in NEURON_PARAMETERS}
    if mode == "source_float64":
        return {name: value.copy() for name, value in original.items()}
    result = {name: value.astype(np.float32).astype(np.float64) for name, value in original.items()}
    # Match GPU-relative inputs, not separately rounded absolute voltages.
    for name in ("V_m", "V_th", "V_reset"):
        relative = (original[name] - original["E_L"]).astype(np.float32).astype(np.float64)
        result[name] = result["E_L"] + relative
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("parameter quantization overflow")
    return result


def load_drive(path, snapshot, metadata, cortical_ids):
    path = Path(path)
    receipt = json.loads((path / "drive.json").read_text())
    if receipt.get("schema") != DRIVE_SCHEMA or receipt.get("counts_decoding_validated") is not True:
        raise ValueError("validated background-arrival drive required")
    if receipt.get("snapshot_manifest_sha256") != sha256_file(Path(snapshot) / "manifest.json"):
        raise ValueError("drive belongs to a different initial-network snapshot")
    if receipt.get("counts_sha256") != sha256_file(path / "counts.npz"):
        raise ValueError("background count artifact hash mismatch")
    if receipt.get("dt_ms") != metadata["dt_ms"] or receipt.get("simulation_duration_ms") != metadata["duration_ms"]:
        raise ValueError("drive clock/duration differs from snapshot")
    with np.load(path / "counts.npz", allow_pickle=False) as archive:
        required = {"unit_ids", "arrival_ticks", "counts", "dt_ms"}
        if not required <= set(archive.files):
            raise ValueError("incomplete background count arrays")
        data = {key: archive[key].copy() for key in required}
    h = float(metadata["dt_ms"])
    duration_ticks = int(_ticks([metadata["duration_ms"]], h)[0])
    ids, arrival, counts = data["unit_ids"], data["arrival_ticks"], data["counts"]
    if data["dt_ms"].shape != () or float(data["dt_ms"]) != h:
        raise ValueError("array clock differs from snapshot")
    if ids.dtype.kind not in "iu" or not np.array_equal(ids, cortical_ids):
        raise ValueError("drive unit IDs must equal sorted cortical source IDs")
    if arrival.dtype.kind not in "iu" or not np.array_equal(arrival, np.arange(duration_ticks)):
        raise ValueError("drive must cover consecutive arrival ticks from zero")
    if counts.dtype.kind not in "iu" or counts.shape != (duration_ticks, len(cortical_ids)) or np.any(counts < 0):
        raise ValueError("nonnegative integer count matrix with exact target shape required")
    last = receipt.get("last_complete_arrival_tick")
    if isinstance(last, bool) or not isinstance(last, int) or last != duration_ticks - 1:
        raise ValueError("background arrival coverage is incomplete")
    return receipt, data


def prepare(snapshot, drive, schedule="effective", quantization="source_float64"):
    snapshot = Path(snapshot)
    metadata, arrays = load_initial_network(snapshot)
    if metadata["source_revision"] != SOURCE_REVISION:
        raise ValueError("unreviewed native source revision")
    if metadata["cortical_neuron_model"] != "iaf_psc_exp" or metadata["thalamic_neuron_model"] != "parrot_neuron":
        raise ValueError("unsupported native neuron model")
    if schedule not in ("initial", "effective"):
        raise ValueError("unknown pulse schedule")
    if quantization not in QUANTIZATIONS:
        raise ValueError("unknown quantization mode")
    h = float(metadata["dt_ms"])
    if int(_ticks([metadata["duration_ms"]], h)[0]) < 2:
        raise ValueError("at least two native simulation ticks required")
    ids, labels, edges = arrays["unit_ids"], arrays["populations"], arrays["edges"]
    cortical = ids[labels != "TC"]
    thalamic = ids[labels == "TC"]
    if not len(cortical) or not np.isfinite(arrays["positions_mm"]).all() or not np.isfinite(arrays["cortical_parameters"]).all():
        raise ValueError("invalid cortical geometry or initial parameters")
    devices = {int(device["id"]): device for device in metadata["devices"]}
    if len(devices) != len(metadata["devices"]) or set(devices).intersection(ids):
        raise ValueError("source/device identity collision")
    if not np.isin(edges["source"], [*ids, *devices]).all() or not np.isin(edges["target"], ids).all():
        raise ValueError("unknown source edge identity")
    if not np.isfinite(edges["weight"]).all() or not np.isfinite(edges["delay"]).all() or np.any(edges["delay"] <= 0):
        raise ValueError("invalid source edge values")
    delays = _ticks(edges["delay"], h)
    if np.any(delays < 1):
        raise ValueError("source delay shorter than one tick")
    pulse, background = {}, set()
    for uid, device in devices.items():
        p = device["parameters"]
        if device["model"] == "poisson_generator":
            stop = math.inf if p["stop"] == "positive_infinity" else float(p["stop"])
            if p["origin"] != 0 or p["start"] != 0 or math.isnan(stop) or stop <= metadata["duration_ms"]:
                raise ValueError("only continuous source background is supported")
            if not math.isfinite(p["rate"]) or p["rate"] < 0:
                raise ValueError("invalid original background rate")
            background.add(uid)
        elif device["model"] == "spike_generator":
            initial = _ticks(p["initial_spike_times_ms"], h)
            effective = _ticks(p["effective_spike_times_ms"], h)
            for train in (initial, effective):
                if np.any(train < 1) or np.any(np.diff(train) <= 0):
                    raise ValueError("ordered positive deterministic emissions required")
            control = metadata["config"].get("pulse_control")
            if control is None:
                if not np.array_equal(initial, effective):
                    raise ValueError("changed pulse history lacks a source decision")
            else:
                boundary = int(_ticks([control["decision_time_ms"]], h)[0])
                if not np.array_equal(initial[initial <= boundary], effective[effective <= boundary]):
                    raise ValueError("future pulse control changed the source history")
                if "initial_emission_ms" in control and not np.array_equal(initial, _ticks(control["initial_emission_ms"], h)):
                    raise ValueError("initial pulse control metadata mismatch")
                if "future_emission_ms" in control:
                    future = _ticks(control["future_emission_ms"], h)
                    if np.any(future <= boundary) or np.any(np.diff(future) <= 0):
                        raise ValueError("future pulse emissions do not follow the decision")
                    if not np.array_equal(effective, np.concatenate([initial[initial <= boundary], future])):
                        raise ValueError("effective pulse control metadata mismatch")
            pulse[uid] = initial if schedule == "initial" else effective
        else:
            raise ValueError("unsupported original source device: " + device["model"])
    background_mask = np.isin(edges["source"], list(background))
    background_edges = edges[background_mask]
    if len(background_edges) != len(cortical) or not np.array_equal(np.sort(background_edges["target"]), cortical):
        raise ValueError("exactly one original background edge per cortical target required")
    background_edges = background_edges[np.argsort(background_edges["target"], kind="stable")]
    receipt, data = load_drive(drive, snapshot, metadata, cortical)
    background_program = []
    for column, edge in enumerate(background_edges):
        selected = np.flatnonzero(data["counts"][:, column] > 0)
        emission = data["arrival_ticks"][selected].astype(np.int64) - int(_ticks([edge["delay"]], h)[0])
        if np.any(emission <= 0):
            raise ValueError("positive background count requires unknown prior or zero-time emission history")
        background_program.append({
            "source_id": int(edge["source"]), "target_id": int(edge["target"]),
            "emission_ticks": emission,
            "arrival_ticks": data["arrival_ticks"][selected].astype(np.int64),
            "multiplicities": [int(value) for value in data["counts"][selected, column]],
            "weight": float(np.float32(edge["weight"])) if quantization == "gpu_fp32_inputs" else float(edge["weight"]),
            "delay": float(edge["delay"]),
        })
    params = dict(zip(NEURON_PARAMETERS, arrays["cortical_parameters"].T))
    for key in ("C_m", "tau_m", "tau_syn_ex", "tau_syn_in"):
        if np.any(params[key] <= 0):
            raise ValueError("nonpositive source membrane/synaptic parameter")
    if np.any(params["t_ref"] < 0):
        raise ValueError("negative refractory period")
    parameters = quantize_parameters(params, quantization)
    weights = edges["weight"].copy()
    if quantization == "gpu_fp32_inputs":
        weights = weights.astype(np.float32).astype(np.float64)
    if not np.isfinite(weights).all():
        raise ValueError("weight quantization overflow")
    expected_edges = edges.copy()
    expected_edges["weight"] = weights
    return {
        "metadata": metadata, "arrays": arrays, "cortical_ids": cortical,
        "thalamic_ids": thalamic, "parameters": parameters, "pulse_schedules": pulse,
        "background_program": background_program, "background_mask": background_mask,
        "expected_edges": expected_edges, "drive_receipt": receipt,
    }


def run(snapshot, drive, output, schedule="effective", quantization="source_float64"):
    snapshot, drive, output = Path(snapshot), Path(drive), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "FAILED", "schedule": schedule, "quantization": quantization,
        "snapshot": str(snapshot.resolve()), "drive": str(drive.resolve()),
        "scientific_reference_qualified": False,
        "state_recording_phase": "native CPU post-arrival PSCs",
        "execution": "one continuous native NEST3 episode with explicit background multiplicities",
        "TC_representation": "original parrot neurons and original deterministic input edges",
        "accumulation_caveat": "GPU can round weight*n to FP32; native CPU multiplication/accumulation remains double. Integer multiplicities are not modified.",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "checks": {},
    }
    try:
        plan = prepare(snapshot, drive, schedule, quantization)
        metadata, arrays = plan["metadata"], plan["arrays"]
        report.update(snapshot_manifest_sha256=sha256_file(snapshot / "manifest.json"),
                      counts_sha256=sha256_file(drive / "counts.npz"),
                      last_complete_arrival_tick=plan["drive_receipt"]["last_complete_arrival_tick"])
        h, duration = float(metadata["dt_ms"]), float(metadata["duration_ms"])
        nest = importlib.import_module("nest")
        if nest.__version__ != NEST_VERSION:
            raise ValueError("exact NEST3.10.0 runtime required")
        defaults = nest.GetDefaults("spike_generator")
        if "spike_multiplicities" not in defaults:
            raise ValueError("native spike_generator multiplicities unavailable")
        nest.ResetKernel()
        seed = int(metadata["config"]["random_streams"]["background_drive"])
        nest.SetKernelStatus({
            "resolution": h, "local_num_threads": 1, "rng_seed": seed,
            "min_delay": h, "max_delay": max(h, float(np.max(arrays["edges"]["delay"]))),
        })
        report["kernel_recording_profile"] = "min_delay equals resolution for step-resolved native recording"
        parameters = plan["parameters"]
        cortex = nest.Create("iaf_psc_exp", len(plan["cortical_ids"]), params=[
            {key: float(value[index]) for key, value in parameters.items()}
            for index in range(len(plan["cortical_ids"]))
        ])
        tc = nest.Create("parrot_neuron", len(plan["thalamic_ids"])) if len(plan["thalamic_ids"]) else nest.NodeCollection([])
        id_map = dict(zip(plan["cortical_ids"].tolist(), cortex.tolist()))
        id_map.update(zip(plan["thalamic_ids"].tolist(), tc.tolist()))
        inverse_neural = {int(native): int(source) for source, native in id_map.items()}
        parameter_readback = cortex.get(list(NEURON_PARAMETERS))
        parameter_readback = {key: np.atleast_1d(value).astype(float).tolist() for key, value in parameter_readback.items()}
        report["checks"]["initial_parameters_exact"] = all(
            np.array_equal(np.asarray(parameter_readback[key]), parameters[key]) for key in parameters
        )
        (output / "parameter_readback.json").write_text(json.dumps(parameter_readback, indent=2) + "\n")
        source_inverse = dict(inverse_neural)
        for source_id, emission in plan["pulse_schedules"].items():
            node = nest.Create("spike_generator", params={"spike_times": (emission * h).tolist()})
            id_map[source_id] = int(node.tolist()[0])
            source_inverse[int(node.tolist()[0])] = source_id

        background_native = {}
        input_rows = []
        for item in plan["background_program"]:
            node = nest.Create("spike_generator", params={
                "spike_times": (item["emission_ticks"] * h).tolist(),
                "spike_multiplicities": item["multiplicities"],
            })
            native_id = int(node.tolist()[0])
            if not np.array_equal(np.asarray(node.get("spike_multiplicities")).reshape(-1), np.asarray(item["multiplicities"])):
                raise ValueError("native multiplicity readback changed integer counts")
            if not np.array_equal(_ticks(np.asarray(node.get("spike_times")).reshape(-1), h), item["emission_ticks"]):
                raise ValueError("native background emission readback changed timing")
            background_native[item["target_id"]] = native_id
            source_inverse[native_id] = item["source_id"]
            for emission, arrival, count in zip(item["emission_ticks"], item["arrival_ticks"], item["multiplicities"]):
                input_rows.append((item["source_id"], item["target_id"], int(emission), int(arrival), count))
        expected = plan["expected_edges"]
        source_ids = np.asarray([
            background_native[int(edge["target"])] if is_background else id_map[int(edge["source"])]
            for edge, is_background in zip(expected, plan["background_mask"])
        ], dtype=np.int64)
        target_ids = np.asarray([id_map[int(uid)] for uid in expected["target"]], dtype=np.int64)
        for begin in range(0, len(expected), 65536):
            end = min(begin + 65536, len(expected))
            nest.Connect(
                source_ids[begin:end], target_ids[begin:end], conn_spec="one_to_one",
                syn_spec={"synapse_model": "static_synapse",
                          "weight": expected["weight"][begin:end].copy(),
                          "delay": expected["delay"][begin:end].copy()},
            )
        neural = nest.NodeCollection(sorted(inverse_neural))
        connections = nest.GetConnections(target=neural, synapse_model="static_synapse").get(["source", "target", "weight", "delay"])
        readback = np.empty(len(np.atleast_1d(connections["source"])), dtype=expected.dtype)
        for index, (src, dst, weight, delay) in enumerate(zip(
            np.atleast_1d(connections["source"]), np.atleast_1d(connections["target"]),
            np.atleast_1d(connections["weight"]), np.atleast_1d(connections["delay"])
        )):
            readback[index] = (inverse_neural[int(dst)], source_inverse[int(src)], weight, delay)
        order = list(expected.dtype.names)
        readback.sort(order=order)
        expected_sorted = np.sort(expected, order=order)
        report["checks"]["all_source_edges_exact"] = np.array_equal(readback, expected_sorted)
        np.save(output / "edge_readback.npy", readback, allow_pickle=False)
        event_dtype = np.dtype([("source_device_id", "<i8"), ("target_unit_id", "<i8"),
                               ("emission_tick", "<i8"), ("arrival_tick", "<i8"), ("multiplicity", "<u8")])
        np.save(output / "background_program.npy", np.asarray(input_rows, dtype=event_dtype), allow_pickle=False)
        if not all(report["checks"].values()):
            raise ValueError("native initial parameter/edge readback failed")

        meter = nest.Create("multimeter", params={"interval": h, "record_from": list(STATE_FIELDS)})
        recorder = nest.Create("spike_recorder")
        nest.Connect(meter, cortex, syn_spec={"delay": h})
        nest.Connect(neural, recorder, syn_spec={"delay": h})
        simulation_start = time.perf_counter()
        nest.Simulate(duration)
        report["simulation_seconds"] = time.perf_counter() - simulation_start
        spike_events = recorder.get("events")
        raw_times = np.asarray(spike_events["times"], dtype=float)
        spike_ticks = _ticks(raw_times, h)
        spike_units = np.asarray([inverse_neural[int(uid)] for uid in spike_events["senders"]], dtype=np.int64)
        spike_order = np.lexsort((spike_units, spike_ticks))
        np.savez_compressed(
            output / "spikes.npz", unit_ids=arrays["unit_ids"], populations=arrays["populations"],
            positions_mm=arrays["positions_mm"], spike_unit_ids=spike_units[spike_order],
            raw_spike_times_ms=raw_times[spike_order], spike_ticks=spike_ticks[spike_order], dt_ms=h,
        )
        state_events = meter.get("events")
        raw_state_times = np.asarray(state_events["times"], dtype=float)
        all_ticks = _ticks(raw_state_times, h)
        state_ticks = np.unique(all_ticks)
        if not len(state_ticks) or state_ticks[0] < 1 or np.any(np.diff(state_ticks) != 1) or state_ticks[-1] > int(_ticks([duration], h)[0]):
            raise ValueError("incomplete or invalid native state recording interval")
        target_index = {native: index for index, native in enumerate(cortex.tolist())}
        columns = np.asarray([target_index[int(uid)] for uid in state_events["senders"]], dtype=np.int64)
        flat = (all_ticks - state_ticks[0]) * len(cortex) + columns
        if len(flat) != len(state_ticks) * len(cortex) or len(np.unique(flat)) != len(flat):
            raise ValueError("missing/duplicate cortical state samples")
        states = np.empty((len(state_ticks) * len(cortex), len(STATE_FIELDS)))
        for column, field in enumerate(STATE_FIELDS):
            states[flat, column] = np.asarray(state_events[field], dtype=float)
        if not np.isfinite(states).all():
            raise ValueError("nonfinite native state")
        time_by_tick = np.empty(len(state_ticks))
        time_by_tick[all_ticks - state_ticks[0]] = raw_state_times
        np.savez_compressed(
            output / "states.npz", ticks=state_ticks, raw_times_ms=time_by_tick,
            unit_ids=plan["cortical_ids"], fields=np.asarray(STATE_FIELDS),
            states=states.reshape(len(state_ticks), len(cortex), len(STATE_FIELDS)),
            recording_phase=np.asarray("CPU post-arrival PSCs"),
            background_complete=state_ticks <= plan["drive_receipt"]["last_complete_arrival_tick"],
        )
        report.update(
            status="EXECUTED", nest_version=nest.__version__, cortical_neurons=len(cortex),
            thalamic_relays=len(tc), source_connections=len(expected),
            target_specific_background_generators=len(plan["background_program"]),
            positive_background_entries=len(input_rows),
            background_count_total=sum(sum(item["multiplicities"]) for item in plan["background_program"]),
            spike_count=len(spike_ticks), first_state_tick=int(state_ticks[0]),
            last_state_tick=int(state_ticks[-1]), state_sample_count=len(state_ticks),
            background_multiplicity_encoding="native spike_generator.spike_multiplicities",
        )
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report["total_seconds"] = time.perf_counter() - started
    report["artifacts"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("traceback", "artifacts")}))
    return int(report["status"] != "EXECUTED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=("initial", "effective"), default="effective")
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default="source_float64")
    args = parser.parse_args()
    raise SystemExit(run(args.snapshot, args.drive, args.output, args.schedule, args.quantization))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-2.0-or-later
"""CUDA execution of source-derived GLIF cells and delayed network events."""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
from .provenance import sha256_file
from .v1_glif_component_reference import SIGNALS, require

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
BUFFER = ctypes.POINTER(ctypes.c_char)


def integer32(values: np.ndarray) -> np.ndarray:
    require(values.dtype.kind in "iu" and np.all(values >= 0)
            and np.all(values <= np.iinfo(np.int32).max), "GLIF_CUDA_INTEGER_DOMAIN")
    return np.ascontiguousarray(values, dtype=np.int32)


def device_library(library: Path):
    library = library.resolve(strict=True)
    native = ctypes.CDLL(str(library))
    native.glif_device_info.argtypes = [I32, BUFFER, ctypes.c_int, BUFFER, ctypes.c_int]
    native.glif_device_info.restype = ctypes.c_int
    info = np.zeros(4, dtype=np.int32)
    name, error = ctypes.create_string_buffer(256), ctypes.create_string_buffer(2048)
    require(native.glif_device_info(info, name, len(name), error, len(error)) == 0,
            "GLIF_CUDA_DEVICE:" + error.value.decode())
    details = dict(device=name.value.decode(), compute_capability=info[:2].tolist(),
                   cuda_runtime_version=int(info[2]), cuda_driver_version=int(info[3]),
                   library=str(library), library_sha256=sha256_file(library), precision="float64")
    return native, details


def parameter_arrays(plan, weights=None):
    weights = np.zeros(plan.count) if weights is None else weights
    scalar = np.ascontiguousarray(np.stack([
        plan.e_l, plan.threshold, plan.reset, plan.p33, plan.p30,
        plan.asc_amplitudes[:, 0], plan.asc_amplitudes[:, 1],
        plan.asc_step_decay[:, 0], plan.asc_step_decay[:, 1],
        plan.asc_mean_coefficient[:, 0], plan.asc_mean_coefficient[:, 1],
        plan.asc_refractory_decay[:, 0], plan.asc_refractory_decay[:, 1], weights]))
    synapse = np.ascontiguousarray(np.stack([
        plan.p11.T, plan.p21.T, plan.p31.T, plan.p32.T, plan.psc_initial.T]))
    state = np.zeros((5+2*plan.receptors, plan.count), dtype=np.float64)
    state[0] = plan.initial_voltage
    state[1:3] = plan.initial_asc.T
    state[3] = plan.initial_asc.sum(axis=1)
    return scalar, synapse, state


def metrics_report(metrics, details):
    details.update(kernel_elapsed_ms=float(metrics[0]), allocation_and_upload_ms=float(metrics[1]),
                   download_ms=float(metrics[2]), library_call_ms=float(metrics[3]),
                   device_allocation_bytes=int(metrics[4]))
    return details


def output_states(plan, terminal, fired):
    require(np.all((fired == 0) | (fired == 1)), "GLIF_CUDA_SPIKE_FLAGS_INVALID")
    rows, steps = np.nonzero(fired)
    return dict(initial_voltage_mV=plan.initial_voltage+plan.e_l, initial_ASC_pA=plan.initial_asc.copy(),
                terminal_voltage_mV=terminal[0]+plan.e_l, terminal_ASC_pA=terminal[1:3].T.copy(),
                spike_native_ids=plan.native_ids[rows], spike_times_ms=(steps+1)*plan.dt)


def emulate_cuda(plan, library: Path) -> tuple[dict[str, np.ndarray], dict]:
    native, details = device_library(library)
    scalar, synapse, state = parameter_arrays(plan, plan.weights)
    integer = integer32(np.stack([plan.refractory_counts, plan.input_index, plan.input_receptor]))
    drive = integer32(plan.drive)
    signals = np.empty((len(SIGNALS), plan.count, plan.steps-1), dtype=np.float64)
    terminal = np.empty_like(state)
    fired = np.empty((plan.count, plan.steps), dtype=np.uint8)
    metrics, error = np.empty(5, dtype=np.float64), ctypes.create_string_buffer(2048)
    native.glif_component_run.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
        F64, F64, I32, I32, F64, F64, F64, U8, F64, BUFFER, ctypes.c_int]
    native.glif_component_run.restype = ctypes.c_int
    status = native.glif_component_run(plan.count, plan.receptors, plan.steps, len(drive), plan.dt,
        scalar, synapse, integer, drive, state, signals, terminal, fired, metrics, error, len(error))
    require(status == 0, "GLIF_CUDA_EXECUTION:" + error.value.decode())
    result = output_states(plan, terminal, fired)
    result.update({"signal_"+key: signals[i] for i, key in enumerate(SIGNALS)})
    details.update(kernel_launches=plan.steps, neuron_cases=plan.count, recurrent_connectivity=False)
    return result, metrics_report(metrics, details)


def emulate_network_cuda(plan, edges: dict, external: dict, library: Path):
    native, details = device_library(library)
    scalar, synapse, state = parameter_arrays(plan)
    integer = integer32(np.stack([plan.refractory_counts, np.zeros(plan.count, dtype=np.int64),
                                  np.zeros(plan.count, dtype=np.int64)]))
    target, receptor, delay = (integer32(edges[key]) for key in ("target", "receptor", "delay"))
    event_sources = integer32(external["source"])
    offsets = np.ascontiguousarray(edges["offsets"], dtype=np.int64)
    event_offsets = np.ascontiguousarray(external["offsets"], dtype=np.int64)
    weights = np.ascontiguousarray(edges["weight"], dtype=np.float64)
    slots = int(delay.max()) + 1
    terminal = np.empty_like(state)
    fired = np.empty((plan.count, plan.steps), dtype=np.uint8)
    metrics, error = np.empty(5, dtype=np.float64), ctypes.create_string_buffer(2048)
    native.glif_network_run.argtypes = [ctypes.c_int]*7 + [
        ctypes.c_double, F64, F64, I32, F64, I64, I32, I32, I32, F64, I64, I32,
        F64, U8, F64, BUFFER, ctypes.c_int]
    native.glif_network_run.restype = ctypes.c_int
    status = native.glif_network_run(plan.count, plan.receptors, plan.steps, len(offsets)-1,
        len(target), len(event_sources), slots, plan.dt, scalar, synapse, integer, state,
        offsets, target, receptor, delay, weights, event_offsets, event_sources,
        terminal, fired, metrics, error, len(error))
    require(status == 0, "GLIF_CUDA_NETWORK_EXECUTION:" + error.value.decode())
    result = output_states(plan, terminal, fired)
    result["spike_count"] = fired.sum(axis=1, dtype=np.int64)
    details.update(neurons=plan.count, recurrent_connectivity=True, connection_rows=len(target),
                   external_events=len(event_sources), delay_ring_slots=slots,
                   kernel_launches=3*plan.steps, floating_point_reduction="CUDA float64 atomicAdd")
    return result, metrics_report(metrics, details)

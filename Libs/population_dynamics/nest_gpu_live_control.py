"""Replace folded TC schedules during one continuously stepped native run."""
from __future__ import annotations

import ctypes
import os

import numpy as np

from .mesocircuit_continuation import PulseProtocol


def future_poisson_reset_tick(boundary_tick: int, native_delay_ticks: int) -> int:
    """Last arrival from an emission at the observed boundary.

    The pinned direct Poisson kernel reads the rate written at zero-based
    iteration ``arrival_tick - native_delay_ticks + 1``. That iteration's
    biological emission tick is one greater. The spatial source therefore
    adds two ticks to its requested physical background delay.
    """
    if (type(boundary_tick) is not int or boundary_tick < 1
            or type(native_delay_ticks) is not int or native_delay_ticks < 2):
        raise ValueError("positive boundary and native Poisson delay of at least two ticks required")
    return boundary_tick + native_delay_ticks - 2


def reset_uniform_poisson_future(ng, source_node, target_nodes, boundary_tick,
                                native_delay_ticks, reset_tick, seed, resolution_ms):
    """Reset one source's future RNG after its committed arrivals.

    All outgoing delays must equal the declared delay. Heterogeneous delays
    require a different per-edge switching implementation and are rejected.
    The native call changes RNG states only. It retains rates and event queues.
    """
    expected_tick = future_poisson_reset_tick(boundary_tick, native_delay_ticks)
    if (type(reset_tick) is not int or reset_tick != expected_tick
            or type(seed) is not int or not 0 < seed < 2**31
            or type(source_node) is not int or source_node < 0
            or not np.isfinite(resolution_ms) or resolution_ms <= 0
            or len(target_nodes) < 1):
        raise ValueError("declared reset tick, positive int31 seed and connected native source required")
    connections = ng.GetConnectionStatus(ng.GetConnections(source=[source_node]))
    delays = np.asarray([row["delay"] for row in connections], dtype=np.float64)
    targets = [int(target_nodes[index]) for index in range(len(target_nodes))]
    if (ng.GetErrorCode() != 0 or delays.shape != (len(target_nodes),)
            or sorted(int(row["target"]) for row in connections) != sorted(targets)
            or any(int(row["source"]) != source_node for row in connections)
            or not np.isfinite(delays).all()
            or not np.array_equal(np.rint(delays / resolution_ms),
                                  np.full(len(target_nodes), native_delay_ticks))
            or not np.allclose(delays, native_delay_ticks * resolution_ms,
                               rtol=1e-6, atol=resolution_ms * 1e-6)):
        raise ValueError("one uniform-delay outgoing background edge per target required")
    library = ctypes.CDLL(os.environ["NESTGPU_LIB"])
    function = library.NeuronSim_ResetFuturePoissonRng
    function.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_longlong]
    function.restype = ctypes.c_int
    result = function(source_node, seed, native_delay_ticks, reset_tick)
    if result != 0 or ng.GetErrorCode() != 0:
        raise RuntimeError("native future Poisson RNG reset failed: " + str(ng.GetErrorMessage()))
    return dict(source_node=source_node, boundary_tick=boundary_tick, reset_tick=reset_tick,
                native_delay_ticks=native_delay_ticks, future_seed=seed,
                outgoing_edges_verified=len(connections),
                committed_emissions_through_tick=boundary_tick,
                last_committed_arrival_tick=expected_tick,
                first_resampled_arrival_tick=expected_tick + 1,
                state_and_queues_reinitialized=False, scientific_claim_qualified=False)


def remaining_tc_arrivals(protocol: PulseProtocol) -> np.ndarray:
    """Retain pending arrivals of committed emissions and future emissions."""
    if protocol.decision_tick is None:
        raise ValueError("native control requires an explicit decision tick")
    return np.asarray([t + protocol.delay_ticks for t in protocol.effective_ticks
                       if t + protocol.delay_ticks > protocol.decision_tick], dtype=np.int64)


def replace_masked_tc_future(ng, native_nodes, active_mask, protocol: PulseProtocol):
    """Update only source-mask device schedules, without clearing event queues.

    The local native interface checks the actual tick and generator type before
    writes. It verifies device times, multiplicities and the consumed index.
    Empty future emissions cancel uncommitted pulses; committed arrivals remain.
    """
    if protocol.decision_tick is None or protocol.decision_tick < 1:
        raise ValueError("positive native decision tick required")
    node_count = len(native_nodes)
    if node_count != len(active_mask) or any(type(v) not in (bool, np.bool_) for v in active_mask):
        raise ValueError("one source-mask boolean per native TC generator required")
    receipt = dict(decision_tick=protocol.decision_tick,
                   future_schedule_replaced=protocol.future_ticks is not None,
                   committed_emission_ticks=list(protocol.committed_ticks),
                   remaining_tc_arrival_ticks=remaining_tc_arrivals(protocol).tolist(),
                   scientific_intervention_qualified=False)
    # Calibration leaves the host schedule intact. Compare its decoded native
    # indices before any device mutation, including silent mask nodes.
    initial_arrivals = np.asarray(protocol.initial_ticks, dtype=np.int64) + protocol.delay_ticks
    # The pinned NodeSeq raises ValueError at its end. Use its declared length
    # instead of Python sequence iteration, which requires IndexError to stop.
    for index in range(node_count):
        node, active = native_nodes[index], active_mask[index]
        actual = np.asarray(ng.GetNeuronStatus([int(node)], "spike_times")).reshape(-1)
        if ng.GetErrorCode() != 0 or not np.isfinite(actual).all():
            raise ValueError("initial TC schedule readback failed")
        expected = initial_arrivals if active else np.empty(0, dtype=np.int64)
        decoded = np.floor(actual.astype(np.float32) / np.float32(protocol.resolution_ms) + .5).astype(np.int64) + 1
        if not np.array_equal(decoded, expected):
            raise ValueError("initial TC schedule differs from the source protocol")
    receipt.update(initial_tc_schedule_verified=True,
                   boundary_phase="after the continuous native decision-tick step; before device schedule mutation")
    if protocol.future_ticks is None:
        return receipt
    remaining = remaining_tc_arrivals(protocol)
    if np.any(remaining > np.iinfo(np.int32).max):
        raise ValueError("native spike generator tick overflows int32")
    library = ctypes.CDLL(os.environ["NESTGPU_LIB"])
    function = library.NeuronSim_ReplaceFutureSpikeGenerator
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_longlong]
    function.restype = ctypes.c_int
    updated = 0
    for index in range(node_count):
        node, active = native_nodes[index], active_mask[index]
        if not active:
            continue
        schedule = np.ascontiguousarray(remaining, dtype=np.int32)
        result = function(int(node), len(schedule), schedule.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                          protocol.decision_tick)
        if result != 0 or ng.GetErrorCode() != 0:
            raise RuntimeError("native future schedule update failed: " + str(ng.GetErrorMessage()))
        expected = ((remaining - 1) * np.float32(protocol.resolution_ms)).astype(np.float32)
        actual = np.asarray(ng.GetNeuronStatus([int(node)], "spike_times")).reshape(-1)
        if not np.array_equal(actual, expected):
            raise ValueError("native future schedule host readback mismatch")
        updated += 1
    receipt.update(updated_generators=updated, device_schedule_readback_verified=True)
    return receipt

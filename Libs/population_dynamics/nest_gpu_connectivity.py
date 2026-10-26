"""Explicit NEST GPU edge import with source-defined connection-block boundaries."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import resource
import time
from typing import Any, Iterator, Sequence

import numpy as np

# Pinned connect.h constructor. OneToOne resets array pointers at each native
# block; keeping each call inside a block preserves every supplied attribute.
NATIVE_CONNECTION_BLOCK_SIZE = 16 * 1024 * 1024


def aligned_batches(start: int, count: int, block_size: int = NATIVE_CONNECTION_BLOCK_SIZE,
                    maximum_batch_size: int = 1_000_000) -> Iterator[tuple[int, int]]:
    if any(type(x) is not int for x in (start, count, block_size, maximum_batch_size)):
        raise ValueError("integer connection offsets and capacities required")
    if start < 0 or count < 0 or block_size < 1 or maximum_batch_size < 1:
        raise ValueError("invalid connection offsets or capacities")
    offset = 0
    while offset < count:
        room = block_size - (start + offset) % block_size
        length = min(count - offset, room, maximum_batch_size)
        yield offset, offset + length
        offset += length


def _validate_edges(source, target, weight, delay, port):
    count = len(source)
    if any(len(values) != count for values in (target, weight, delay, port)):
        raise ValueError("one source, target, weight, delay and receptor per edge required")
    if any(type(x) is not int or x < 0 for values in (source, target, port) for x in values):
        raise ValueError("nonnegative native integer IDs/receptors required")
    if any(not math.isfinite(float(x)) for x in weight):
        raise ValueError("finite signed weights required")
    if any(not math.isfinite(float(x)) or x <= 0 for x in delay):
        raise ValueError("finite positive delays required")


def raw_connect(ng, source, target, weight, delay, port: int) -> None:
    """Use native ctypes arrays whose lifetime covers the synchronous CUDA copy."""
    weights = (ctypes.c_float * len(weight))(*weight)
    delays = (ctypes.c_float * len(delay))(*delay)
    ng.Connect(list(source), list(target), {"rule": "one_to_one"},
               {"weight_array": weights, "delay_array": delays, "receptor": port})


def raw_connect_arrays(ng, source, target, weight, delay, port: int) -> None:
    """Same pinned native calls as Connect, with contiguous NumPy-owned buffers.

    The synchronous native call copies all arrays before these owners leave
    scope. This removes per-edge Python-object conversion, not any source edges.
    """
    source = np.ascontiguousarray(source, dtype=np.int32)
    target = np.ascontiguousarray(target, dtype=np.int32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    delay = np.ascontiguousarray(delay, dtype=np.float32)
    ng.ConnSpecInit(); ng.SynSpecInit()
    ng.SetConnSpecParam("rule", ng.conn_rule_name.index("one_to_one"))
    ng.SetSynSpecIntParam("receptor", int(port))
    ng.SetSynSpecFloatPtParam("weight_array", weight.ctypes.data_as(ctypes.c_void_p))
    ng.SetSynSpecFloatPtParam("delay_array", delay.ctypes.data_as(ctypes.c_void_p))
    ng.NESTGPU_ConnectGroupGroup(source.ctypes.data_as(ctypes.c_void_p), len(source),
                               target.ctypes.data_as(ctypes.c_void_p), len(target))
    if ng.GetErrorCode() != 0:
        raise ValueError(ng.GetErrorMessage())


@dataclass
class ExplicitConnector:
    """Own all connection creation for one fresh kernel until calibration.

    The caller supplies the number of previously created native connections, if
    any, and must not create unaccounted connections behind this object's back.
    No deduplication or edge-sign rewriting is performed.
    """
    backend: Any
    connection_count: int = 0
    maximum_batch_size: int = 1_000_000

    def connect(self, source: Sequence[int], target: Sequence[int],
                weight: Sequence[float], delay: Sequence[float],
                port: Sequence[int]) -> dict[str, Any]:
        _validate_edges(source, target, weight, delay, port)
        count_before, calls = self.connection_count, 0
        # Native receptor is scalar per call. Stable grouping preserves the
        # complete edge multiset, including duplicate pairs with distinct delays.
        for receptor in sorted(set(port)):
            indices = [i for i, p in enumerate(port) if p == receptor]
            for left, right in aligned_batches(self.connection_count, len(indices),
                                               maximum_batch_size=self.maximum_batch_size):
                take = indices[left:right]
                raw_connect(self.backend, [source[i] for i in take],
                            [target[i] for i in take], [weight[i] for i in take],
                            [delay[i] for i in take], receptor)
                calls += 1
            self.connection_count += len(indices)
        return {"edges_added": self.connection_count - count_before,
                "native_connect_calls": calls, "connection_count": self.connection_count}


    def connect_arrays(self, source, target, weight, delay, port) -> dict[str, Any]:
        """Vectorized equivalent of connect, retaining receptor/block ordering."""
        source, target, weight, delay, port = map(np.asarray, (source, target, weight, delay, port))
        if source.ndim != 1 or any(v.shape != source.shape for v in (target, weight, delay, port)):
            raise ValueError("one array entry per connection required")
        for values in (source, target, port):
            if (values.dtype.kind not in "iu" or np.any(values < 0)
                    or np.any(values > np.iinfo(np.int32).max)):
                raise ValueError("native int32 nonnegative IDs/receptors required")
        for values in (weight, delay):
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError("finite numeric native attributes required")
            if np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValueError("native attribute overflows float32")
        if np.any(delay <= 0):
            raise ValueError("positive native delays required")
        count_before, calls = self.connection_count, 0
        for receptor in np.unique(port):
            indices = np.flatnonzero(port == receptor)
            for left, right in aligned_batches(self.connection_count, len(indices),
                                                maximum_batch_size=self.maximum_batch_size):
                take = indices[left:right]
                raw_connect_arrays(self.backend, source[take], target[take], weight[take],
                                   delay[take], int(receptor))
                calls += 1
            self.connection_count += len(indices)
        return {"edges_added": self.connection_count-count_before,
                "native_connect_calls": calls, "connection_count": self.connection_count}


def _row(row: dict[str, Any], dt: float) -> tuple:
    return (row["source"], row["target"], row["port"],
            round(row["delay"] / dt), row["weight"])


def probe(mode: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"mode": mode, "status": "FAILED", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_reference_qualified": False, "neuron_count_scale_claim": False}
    try:
        ng = importlib.import_module("nestgpu")
        ng.SetTimeResolution(.1)
        ng.SetRandomSeed(74001)
        ng.SetKernelStatus("verbosity_level", 0)
        prefilling = mode.startswith("block_")
        count = 0
        if prefilling:
            # Exactly two slots remain. This is an edge-storage stress fixture,
            # not a population-fidelity experiment.
            dummy = ng.Create("iaf_psc_exp", 4096)
            ng.Connect(dummy, dummy[0:4095], {"rule": "all_to_all"},
                       {"weight": 1., "delay": .2, "receptor": 0})
            ng.Connect(dummy[0:4094], dummy[0:4094], {"rule": "one_to_one"},
                       {"weight": 1., "delay": .2, "receptor": 0})
            count = 4096 * 4095 + 4094
            assert count == NATIVE_CONNECTION_BLOCK_SIZE - 2
        cells = ng.Create("iaf_psc_exp", 4)
        sources = ng.Create("spike_generator", 2)
        ng.SetStatus(sources, {"spike_times": [.9, 2.9]})
        src = [sources[0], sources[0], sources[1], sources[1], sources[0], sources[1]]
        dst = [cells[0], cells[0], cells[1], cells[2], cells[3], cells[3]]
        weights = [1., 2., 3., 4., 5., 6.]
        delays = [.3, .4, .5, .6, .7, .8]
        ports = [0] * 6
        if mode == "small_signed":
            weights[1] = -2.;weights[3] = -4.;ports[1] = ports[3] = 1
        if mode == "block_native":
            raw_connect(ng, src, dst, weights, delays, 0)
            report["native_connect_calls"] = 1
        else:
            connector = ExplicitConnector(ng, count)
            report.update(connector.connect(src, dst, weights, delays, ports))
        ng.Calibrate()
        actual = ng.GetConnectionStatus(ng.GetConnections(source=sources, target=cells))
        expected = [{"source": s, "target": t, "port": p,
                     "weight": ctypes.c_float(w).value, "delay": d}
                    for s, t, w, d, p in zip(src, dst, weights, delays, ports)]
        observed_rows = sorted(_row(r, .1) for r in actual)
        expected_rows = sorted(_row(r, .1) for r in expected)
        report.update(prefill_connections=count, actual_edges=actual,
                      expected_edges=expected, exact_edge_multiset=observed_rows == expected_rows,
                      status="PASS" if observed_rows == expected_rows else "FAILED",
                      total_native_nodes=(4096 if prefilling else 0) + 6)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    report.update(seconds=time.perf_counter()-start,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("small_signed", "block_native", "block_aware"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(probe(args.mode, args.output))


if __name__ == "__main__":
    main()

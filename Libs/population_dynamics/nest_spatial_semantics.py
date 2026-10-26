"""Bounded native NEST 3.10 delay and periodic-connectivity diagnostics.

Uses at most 18 live neurons and never calls Simulate. Inputs, exact float hex
values, all native edge readbacks, source headers, and failures are retained.
These are engineering semantics checks, not scientific reference qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

import nest
import numpy as np


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reset(dt=0.1):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt, "local_num_threads": 1, "rng_seed": 830101})


def read_edges():
    connections = nest.GetConnections()
    if len(connections) == 0:
        return []
    values = connections.get(["source", "target", "weight", "delay"])
    values = {key: np.atleast_1d(value) for key, value in values.items()}
    return [{"source": int(s), "target": int(t), "weight": float(w), "delay_ms": float(d)}
            for s, t, w, d in zip(values["source"], values["target"], values["weight"], values["delay"])]


def round_half_away(value):
    magnitude = abs(value)
    integral = math.floor(magnitude)
    return math.copysign(integral + int(magnitude - integral >= 0.5), value)


def displacement(source, target, length):
    result = []
    for s, t in zip(source, target):
        d = float(t) - float(s)
        d -= length * round_half_away(d * (1.0 / length))
        result.append(d)
    return result


def native_ball_predicate(vector, radius):
    # mask_impl.h:146-167: coordinate rejection, L1 early acceptance, sqrt.
    absolute = [abs(x) for x in vector]
    if any(x > radius for x in absolute):
        return False
    if sum(absolute) <= radius:
        return True
    return math.sqrt(sum(x * x for x in vector)) <= radius


def native_round_function():
    library = ctypes.CDLL(str(Path(nest.__file__).parent / "nestkernel_api.so"))
    for symbol in ("_Z8ld_roundd", "_ZN4nest8ld_roundEd"):
        try:
            function = getattr(library, symbol)
        except AttributeError:
            continue
        function.argtypes = [ctypes.c_double]
        function.restype = ctypes.c_long
        return function, symbol
    return None, None


def delay_probe(output):
    native_round, symbol = native_round_function()
    rows = []
    mismatches = []
    kernel_profiles = []
    for dt in (0.1, 0.025):
        reset(dt)
        nodes = nest.Create("parrot_neuron", 2)
        kernel_profiles.append({k: nest.GetKernelStatus()[k] for k in
                                ("resolution", "tics_per_ms", "ms_per_tic", "tics_per_step")})
        cases = []
        # Declare a fixed grid of half-step neighbors, microtic offsets and
        # arbitrary distance-derived delays before inspecting any readback.
        for tick in (0, 1, 2, 3, 4, 7, 15, 63, 127):
            midpoint = (tick + 0.5) * dt
            for direction, value in (("minus_ulp", np.nextafter(midpoint, -np.inf)),
                                     ("exact", midpoint),
                                     ("plus_ulp", np.nextafter(midpoint, np.inf))):
                cases.append((f"half_{tick}_{direction}", float(value)))
            for offset in (-0.00051, -0.00049, -0.0001, 0.0001, 0.00049, 0.00051):
                cases.append((f"half_{tick}_offset_{offset}", midpoint + offset))
        for distance in (0.0, math.sqrt(2.0) / 10, 0.123456789, 0.3, 0.49999999999999994,
                         0.5, math.sqrt(0.3**2 + 0.4**2), 1.1600000000000001):
            cases.append((f"linear_distance_{distance.hex()}", 0.5 + distance / 0.3))
        for index, (label, value) in enumerate(cases, 1):
            scaled = value * (1.0 / dt)
            # The header documents nearest integer with midpoint upward.
            documented = math.floor(scaled) + int(scaled - math.floor(scaled) >= 0.5)
            native_prediction = int(native_round(scaled)) if native_round is not None else documented
            row = {"resolution_ms": dt, "label": label, "input_delay_ms": value,
                   "input_delay_hex": value.hex(), "scaled_steps": scaled,
                   "scaled_steps_hex": scaled.hex(), "documented_nearest_up_steps": documented,
                   "native_ld_round_steps": native_prediction, "weight_marker": index}
            before = len(nest.GetConnections())
            try:
                nest.Connect(nodes[0], nodes[1], syn_spec={"weight": float(index), "delay": value})
                edge = [e for e in read_edges() if e["weight"] == float(index)]
                require(len(edge) == 1, "delay marker did not identify exactly one edge")
                row.update(accepted=True, readback=edge[0],
                           readback_delay_hex=edge[0]["delay_ms"].hex())
                expected_ms = native_prediction * dt
                if native_prediction < 1 or edge[0]["delay_ms"] != expected_ms:
                    mismatches.append({"case": row, "expected_readback_ms": expected_ms})
            except Exception as error:
                row.update(accepted=False, error_type=type(error).__name__, error=str(error))
                if native_prediction >= 1 or len(nest.GetConnections()) != before:
                    mismatches.append({"case": row, "unexpected_rejection_or_side_effect": True})
            rows.append(row)
    write_json(output / "delay_readbacks.json", rows)
    write_json(output / "delay_mismatches.json", mismatches)
    require(not mismatches, "native delay readback differs from source-backed conversion")
    return {"status": "PASS", "cases": len(rows), "accepted": sum(r["accepted"] for r in rows),
            "native_round_symbol": symbol, "kernel_profiles": kernel_profiles,
            "documented_vs_native_round_differences": [r for r in rows if
                r["documented_nearest_up_steps"] != r["native_ld_round_steps"]],
            "source_rule": "nest_time.h:524-541: ld_round(ms*STEPS_PER_MS), distinct from ms_stamp-to-Time; numerics.h:103-123 documents midpoint-up rounding",
            "intermediate_tic_quantization_used": False}


def spatial_probe(output):
    diagonal = 0.3
    points = [
        (0., 0.), (-0.5, 0.), (np.nextafter(0.5, -np.inf), 0.),
        (0.49, 0.49), (-0.49, -0.49), (0.49, 0.), (-0.49, 0.),
        (diagonal, 0.4), (np.nextafter(diagonal, -np.inf), 0.4),
        (np.nextafter(diagonal, np.inf), 0.4),
        (-diagonal, -0.4), (0.25, 0.25), (-0.25, -0.25),
        (0., -0.5), (0., np.nextafter(0.5, -np.inf)), (0.1, -0.2),
    ]
    points = [[float(x), float(y)] for x, y in points]
    write_json(output / "spatial_inputs.json", {"extent_mm": 1., "radius_mm": 0.5,
        "positions_mm": points, "position_hex": [[v.hex() for v in p] for p in points],
        "diagonal_cases": [7, 8, 9, 10], "units": len(points)})
    cases = []
    mismatches = []
    default_pairs = None
    for flag in (None, False, True):
        for radius in (0.5, float(np.nextafter(0.5, np.inf))):
            reset()
            nodes = nest.Create("parrot_neuron", positions=nest.spatial.free(
                pos=points, extent=[1., 1.], edge_wrap=True))
            ids = list(map(int, nodes.tolist()))
            actual_positions = [list(map(float, p)) for p in nest.GetPosition(nodes)]
            spec = {"rule": "pairwise_bernoulli", "p": 1.0,
                    "allow_autapses": False, "allow_multapses": True,
                    "mask": {"circular": {"radius": radius}}}
            if flag is not None:
                spec["allow_oversized_mask"] = flag
            entry = {"allow_oversized_mask": flag, "radius_mm": radius,
                     "radius_hex": radius.hex(), "unit_ids": ids, "actual_positions_mm": actual_positions}
            try:
                nest.Connect(nodes, nodes, conn_spec=spec,
                             syn_spec={"weight": nest.spatial.distance, "delay": 0.1})
                edges = read_edges()
                entry.update(accepted=True, edges=edges)
            except Exception as error:
                entry.update(accepted=False, error_type=type(error).__name__, error=str(error), edges=read_edges())
            if radius > 0.5:
                expected_accept = flag is True
                if entry["accepted"] != expected_accept:
                    mismatches.append({"case": entry, "oversize_default_or_override_differs": True})
                cases.append(entry)
                continue
            if not entry["accepted"]:
                mismatches.append({"case": entry, "half_extent_mask_rejected": True})
                cases.append(entry)
                continue
            expected = {}
            for i, source in enumerate(ids):
                for j, target in enumerate(ids):
                    if source == target:
                        continue
                    vector = displacement(actual_positions[i], actual_positions[j], 1.0)
                    if native_ball_predicate(vector, radius):
                        expected[(source, target)] = math.sqrt(sum(x*x for x in vector))
            counts = Counter((e["source"], e["target"]) for e in edges)
            observed = set(counts)
            wanted = set(expected)
            missing = sorted(wanted - observed)
            extra = sorted(observed - wanted)
            duplicate = [list(pair) + [n] for pair, n in counts.items() if n != 1]
            distance_errors = []
            for edge in edges:
                pair = (edge["source"], edge["target"])
                if pair in expected:
                    err = abs(edge["weight"] - expected[pair])
                    bound = 4 * math.ulp(max(expected[pair], 1e-300))
                    if err > bound:
                        distance_errors.append({"edge": edge, "expected_distance": expected[pair],
                                                "absolute_error": err, "four_ulp_bound": bound})
            entry.update(expected_pairs=[list(p) for p in sorted(expected)], missing_pairs=missing,
                         extra_pairs=extra, duplicate_pairs=duplicate, distance_errors=distance_errors)
            if missing or extra or duplicate or distance_errors:
                mismatches.append({"case": entry, "pair_or_distance_mismatch": True})
            if flag is None:
                default_pairs = observed
                # Three distinct repeated calls must retain three synapses.
                for repetition in range(2):
                    nest.Connect(nodes, nodes, conn_spec=spec,
                                 syn_spec={"weight": nest.spatial.distance, "delay": 0.1})
                repeated_edges = read_edges()
                repeated = Counter((e["source"], e["target"]) for e in repeated_edges)
                entry["three_call_edges"] = repeated_edges
                entry["three_call_multiplicities_exact"] = set(repeated) == wanted and all(n == 3 for n in repeated.values())
                if not entry["three_call_multiplicities_exact"]:
                    mismatches.append({"case": entry, "repeated_call_multiplicity_differs": True})
            elif observed != default_pairs:
                mismatches.append({"case": entry, "oversize_flag_changed_half_extent_pairs": True})
            cases.append(entry)
    write_json(output / "spatial_readbacks.json", cases)
    write_json(output / "spatial_mismatches.json", mismatches)
    require(not mismatches, "native spatial source predicate or repeated-pair semantics unresolved")
    return {"status": "PASS", "units": len(points), "cases": len(cases),
            "half_extent_single_call_edges": len(default_pairs), "repeat_connect": 3,
            "allow_oversized_mask_default": False, "boundary": "native axis/L1/sqrt predicate is inclusive",
            "source": "layer_impl.h:69-75,412-425; mask_impl.h:146-167; connection_creator_impl.h:92-111"}


def weight_probe(output):
    rows = []
    for sign in (1., -1.):
        reset()
        nodes = nest.Create("parrot_neuron", 2)
        weight = nest.math.redraw(nest.random.normal(mean=sign, std=0.75),
                                  min=0. if sign > 0 else -math.inf,
                                  max=math.inf if sign > 0 else 0.)
        nest.Connect(nodes[0], nodes[1], conn_spec={"rule": "fixed_total_number", "N": 512,
                     "allow_autapses": False, "allow_multapses": True},
                     syn_spec={"weight": weight, "delay": 0.1})
        edges = read_edges()
        values = [edge["weight"] for edge in edges]
        rows.append({"mean": sign, "std": 0.75, "edges": edges})
        require(len(edges) == 512 and all(math.isfinite(v) and sign*v > 0 for v in values),
                "redraw produced nonfinite, zero, or wrong-sign native weights")
    write_json(output / "weight_readbacks.json", rows)
    return {"status": "PASS", "positive_samples": 512, "negative_samples": 512,
            "scope": "finite sign-support check only; no distributional qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    nest.set_verbosity("M_ERROR")
    report = {"schema": "native_nest_spatial_semantics_v1", "status": "FAILED",
              "scientific_reference_qualified": False, "nest_version": nest.__version__,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
              "script_sha256": sha256(Path(__file__)), "python": sys.version,
              "maximum_live_neurons": 16, "simulate_called": False, "checks": {}}
    require(nest.__version__ == "3.10.0", "NEST3.10.0 required")
    include = Path(nest.__file__).parent / "include/nest"
    names = ["nest_time.h", "numerics.h", "layer_impl.h", "mask_impl.h", "connection_creator_impl.h", "position.h"]
    source_dir = output / "native_source"
    source_dir.mkdir()
    for name in names:
        source = include / name
        if source.exists():
            shutil.copy2(source, source_dir / name)
    report["native_source_sha256"] = {p.name: sha256(p) for p in source_dir.iterdir()}
    report["native_library_sha256"] = sha256(Path(nest.__file__).parent / "nestkernel_api.so")
    for name, probe in (("delay", delay_probe), ("spatial", spatial_probe), ("weight", weight_probe)):
        try:
            report["checks"][name] = probe(output)
        except Exception as error:
            report["checks"][name] = {"status": "FAILED", "error_type": type(error).__name__,
                                      "error": str(error), "traceback": traceback.format_exc()}
        print(json.dumps({"check": name, **report["checks"][name]}, allow_nan=False), flush=True)
    report["status"] = "PASS" if all(c["status"] == "PASS" for c in report["checks"].values()) else "FAILED"
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

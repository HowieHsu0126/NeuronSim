"""Diagnose the pinned source's exponential connection rule at fixed positions.

Exact pair enumeration is batched and restricted to an engineering network.
Native edges include multiplicity. Conditional moments are descriptive, without
an uncalibrated statistical pass threshold or a source normalization correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .mesocircuit_reference import (
    _fresh_native_directory, _independent_initial_voltages, load_native_network)
from .mesocircuit_source import derive_parameters, load_config, verify_source
from .mesocircuit_normalization import normalization_record
from .nest_reference import _set_rng_phase, fingerprint_connections
from .provenance import freeze_json, runtime_manifest, sha256_file


def conditional_moments(source_ids, source_xy, target_ids, target_xy, *,
                        extent_mm, beta_mm, radius_mm, p0, repeats):
    """Mean and variance of synapse count, conditional on all given coordinates."""
    if (not np.isfinite([extent_mm, beta_mm, radius_mm, p0]).all()
            or extent_mm <= 0 or beta_mm <= 0 or not 0 < radius_mm <= extent_mm / 2
            or not 0 <= p0 <= 1 or type(repeats) is not int or repeats < 1):
        raise ValueError("EXPONENTIAL_CONNECTIVITY_PARAMETERS_INVALID")
    source_ids, target_ids = np.asarray(source_ids), np.asarray(target_ids)
    source_xy, target_xy = np.asarray(source_xy), np.asarray(target_xy)
    if (source_ids.ndim != 1 or target_ids.ndim != 1
            or source_xy.shape != (len(source_ids), 2)
            or target_xy.shape != (len(target_ids), 2)
            or not np.isfinite(source_xy).all() or not np.isfinite(target_xy).all()
            or np.any(np.abs(source_xy) > extent_mm / 2)
            or np.any(np.abs(target_xy) > extent_mm / 2)):
        raise ValueError("CONNECTIVITY_IDENTITIES_OR_POSITIONS_INVALID")
    delta = np.abs(target_xy[:, None, :] - source_xy[None, :, :])
    delta = np.minimum(delta, extent_mm - delta)
    distance = np.sqrt(np.square(delta).sum(axis=-1))
    possible = (distance <= radius_mm) & (target_ids[:, None] != source_ids[None, :])
    probability = np.where(possible, p0 * np.exp(-distance / beta_mm), 0.)
    return (repeats * probability.sum(axis=1),
            repeats * (probability * (1 - probability)).sum(axis=1))


def uniform_kernel_integral(extent_mm, beta_mm, radius_mm):
    """Area-normalized integral of exp(-r/beta) over the declared circular mask."""
    f = radius_mm / beta_mm
    return 2 * np.pi * beta_mm**2 / extent_mm**2 * (1 - np.exp(-f) * (1 + f))


def run_connectivity_check(config: dict, source: Path, output: Path,
                           command: list[str], target_batch_size=32):
    if config["run_class"] != "engineering_diagnostic":
        raise ValueError("EXACT_CONNECTIVITY_CHECK_REQUIRES_ENGINEERING_CONFIG")
    if type(target_batch_size) is not int or target_batch_size < 1:
        raise ValueError("CONNECTIVITY_POSITIVE_BATCH_SIZE_REQUIRED")
    source, output = source.resolve(), output.resolve()
    provenance = verify_source(source)
    net, sim = derive_parameters(source, config)
    normalization_policy = config.get("connectivity_normalization", "native_source")
    normalization = normalization_record(net, normalization_policy)
    if net["connect_method"] != "distr_indegree_exp":
        raise ValueError("EXPONENTIAL_BERNOULLI_SOURCE_REQUIRED")
    native, nest = load_native_network(source)
    if nest.__version__ != config["expected_nest_version"]:
        raise ValueError("MESOCIRCUIT_NEST_VERSION_MISMATCH")
    with _fresh_native_directory(output):
        network = native.Network(SimpleNamespace(data_dir_circuit=str(output),
            net_dict=net, sim_dict=sim), config["local_num_threads"])
        network.create()
        ids = np.concatenate([np.asarray(pop.tolist(), dtype=np.int64) for pop in network.pops])
        labels = np.concatenate([np.full(len(pop), str(label))
                                 for pop, label in zip(network.pops, net["populations"])])
        xy = np.concatenate([np.asarray(nest.GetPosition(pop), dtype=float) for pop in network.pops])
        streams = config.get("random_streams")
        if streams:
            _independent_initial_voltages(network, net, streams["initial_state"])
            _set_rng_phase(nest, streams["connectivity"], "connectivity")
        network.connect()
        fingerprint = fingerprint_connections(nest, ids, target_batch_size)
        raw = {key: [] for key in ("source", "target", "weight", "delay")}
        for start in range(0, len(ids), target_batch_size):
            connections = nest.GetConnections(
                target=nest.NodeCollection(ids[start:start + target_batch_size].tolist()),
                synapse_model="static_synapse")
            fields = connections.get(list(raw))
            for key in raw:
                raw[key].append(np.asarray(fields[key]))
        edges = {key: np.concatenate(value).astype(np.int64 if key in ("source", "target") else float)
                 for key, value in raw.items()}
        cortical_ids = ids[labels != "TC"]
        source_index = {int(uid): i for i, uid in enumerate(ids)}
        population_index = {str(label): i for i, label in enumerate(net["populations"])}
        groups = np.array([population_index[label] for label in labels])
        observed = np.zeros((len(cortical_ids), len(net["populations"])), dtype=np.int64)
        target_index = {int(uid): i for i, uid in enumerate(cortical_ids)}
        external = 0
        invalid = 0
        for src, tgt in zip(edges["source"], edges["target"]):
            if int(src) not in source_index:
                external += 1
                continue
            si, ti = source_index[int(src)], source_index[int(tgt)]
            if int(tgt) not in target_index:
                raise ValueError("NATIVE_NEURON_EDGE_TARGETS_THALAMUS")
            a, b = groups[ti], groups[si]
            observed[target_index[int(tgt)], b] += 1
            d = np.abs(xy[si] - xy[ti]); d = np.minimum(d, net["extent"] - d)
            if src == tgt or np.linalg.norm(d) > net["mask_radius"][a, b] or net["p0"][a, b] == 0:
                invalid += 1
        if invalid:
            raise ValueError("NATIVE_EDGES_VIOLATE_SOURCE_CONNECTIVITY_SUPPORT")
        means, variances = np.zeros_like(observed, dtype=float), np.zeros_like(observed, dtype=float)
        rows = []
        for a, target_label in enumerate(net["populations"][:-1]):
            ti = np.flatnonzero(labels == target_label)
            oi = np.array([target_index[int(uid)] for uid in ids[ti]])
            for b, source_label in enumerate(net["populations"]):
                si = np.flatnonzero(labels == source_label)
                for start in range(0, len(ti), target_batch_size):
                    ix = ti[start:start + target_batch_size]
                    mean, variance = conditional_moments(
                        ids[si], xy[si], ids[ix], xy[ix],
                        extent_mm=float(net["extent"]), beta_mm=float(net["beta"][a, b]),
                        radius_mm=float(net["mask_radius"][a, b]), p0=float(net["p0"][a, b]),
                        repeats=int(net["repeat_connect"][a, b]))
                    means[oi[start:start + len(ix)], b] = mean
                    variances[oi[start:start + len(ix)], b] = variance
                m, v = means[oi, b].sum(), variances[oi, b].sum()
                actual = int(observed[oi, b].sum())
                uniform = len(ti) * (len(si) - int(a == b)) * float(net["p0_raw"][a, b]) * uniform_kernel_integral(
                    float(net["extent"]), float(net["beta"][a, b]), float(net["mask_radius"][a, b]))
                rows.append({
                    "target_population": str(target_label), "source_population": str(source_label),
                    "nominal_num_synapses": int(net["num_synapses"][a, b]),
                    "actual_synapses": actual, "conditional_mean": float(m), "conditional_variance": float(v),
                    "descriptive_standardized_residual": float((actual-m)/np.sqrt(v)) if v > 0 else None,
                    "uniform_layout_mean_with_self_exclusion": float(uniform),
                    "p0_raw": float(net["p0_raw"][a, b]), "p0": float(net["p0"][a, b]),
                    "repeats": int(net["repeat_connect"][a, b]),
                    "beta_mm": float(net["beta"][a, b]), "mask_radius_mm": float(net["mask_radius"][a, b])})
        np.savez_compressed("connectivity_arrays.npz", unit_ids=ids, populations=labels,
            positions_mm=xy, cortical_ids=cortical_ids, observed_indegrees=observed,
            conditional_mean=means, conditional_variance=variances, **edges)
        freeze_json(Path("population_pairs.json"), rows)
        freeze_json(Path("connectivity_parameters.json"), normalization)
        total_mean, total_variance = float(means.sum()), float(variances.sum())
        summary = {
            "status": "PARTIAL", "scope": "native_connectivity_engineering_diagnostic",
            "source_revision": provenance["revision"], "unit_count": len(ids), "extent_mm": float(net["extent"]),
            "native_connection_fingerprint": fingerprint, "all_incoming_static_synapses": len(edges["source"]),
            "external_input_synapses": external, "neuronal_synapses": int(observed.sum()),
            "nominal_neuronal_synapses": int(np.sum(net["num_synapses"])),
            "conditional_mean_neuronal_synapses": total_mean,
            "conditional_variance_neuronal_synapses": total_variance,
            "descriptive_standardized_residual": (int(observed.sum()) - total_mean) / np.sqrt(total_variance),
            "uniform_layout_mean_with_self_exclusion": sum(row["uniform_layout_mean_with_self_exclusion"] for row in rows),
            "autapses_or_out_of_mask_edges": invalid,
            "target_batch_size": target_batch_size, "statistical_connectivity_qualified": False,
            "source_normalization_corrected": normalization_policy == "kernel_integral",
            "connectivity_normalization": normalization_policy,
            "source_code_modified": False, "scientific_reference_qualified": False,
            "interpretation": "Nominal parameters, conditional Bernoulli moments and realized synapse counts are separate quantities. No statistical PASS from one draw."}
        freeze_json(Path("summary.json"), summary)
        freeze_json(Path("manifest.json"), {
            "schema": "mesocircuit_connectivity_check_v1",
            "source_manifest_sha256": sha256_file(source / "source.json"),
            "artifacts": {name: sha256_file(Path(name)) for name in (
                "connectivity_arrays.npz", "population_pairs.json", "connectivity_parameters.json", "summary.json")},
            **runtime_manifest(config, command, 2)})
    return summary, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        summary, code = run_connectivity_check(
            config, Path(config["source_dir"]), args.output, sys.argv, args.target_batch_size)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

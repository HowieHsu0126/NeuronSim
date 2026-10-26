"""Paired whole-block neural effects of a declared Poisson source variant.

Covariances remove the across-block mean separately at every matched time, then
average over time. They are not conditional-on-history or fixed-noise covariances.
The pooled ensemble count Fano is mean_time(Var_block(count)) /
mean_time(Mean_block(count)). No equivalence or qualification decision is made.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import yaml

from .nest_gpu_network import ticks
from .nest_gpu_spatial import Execution
from .provenance import sha256_file
from .pulse_history_analysis import _verified_events
from .source_variant_cohort import SCHEMA
from .spatial_field import PeriodicPopulationGrid
from .trajectory_data import verify_controls


def require_acquired(ledger: dict, case_ids: list[str]) -> None:
    """A partial or failed acquisition cannot enter a complete-cohort analysis."""
    rows = ledger.get("runs", [])
    identities = [row["case_id"] for row in rows]
    if (ledger.get("status") != "ACQUIRED"
            or ledger.get("completed_episodes") != len(case_ids)
            or ledger.get("expected_episodes") != len(case_ids)
            or len(identities) != len(set(identities))
            or set(identities) != set(case_ids)
            or any(row.get("exit_code") != 0 for row in rows)):
        raise ValueError("complete ACQUIRED cohort with every declared case required")


def validate_cases(declaration: dict, original: dict, reference_design: dict,
                   config: dict) -> tuple[list[dict], dict[str, dict], dict]:
    if (declaration["schema"] != SCHEMA or config != declaration["config"]
            or config["split"] != "train" or config["template_id"] != "familiar_0"
            or config["formal_training_seeds"] != [41, 42, 43]
            or declaration["common_noise_across_rng_families"] is not False
            or declaration["common_initial_states"] is not True):
        raise ValueError("unchanged training-only source-variant declaration required")
    cases = sorted(declaration["cases"], key=lambda row: row["block"])
    selected = [c for c in original["cases"] if c["split"] == "train"
                and c["template_id"] == config["template_id"]]
    references = {c["case_id"]: c for c in reference_design["cases"]}
    wanted = {c["case_id"]: c for c in selected}
    n = config["expected_blocks"]
    if (n < 3 or len(cases) != n or len(wanted) != n
            or len({c["case_id"] for c in cases}) != n
            or len({c["block"] for c in cases}) != n
            or len({c["initial_state_seed"] for c in cases}) != n
            or len({c["background_seed"] for c in cases}) != n
            or set(wanted) != {c["case_id"] for c in cases}):
        raise ValueError("one case for every distinct declared initial/background block required")
    for candidate in cases:
        identity = candidate["case_id"]
        if Path(identity).name != identity or identity not in references:
            raise ValueError("invalid or missing reference case identity")
        reference = references[identity]
        if reference != wanted[identity]:
            raise ValueError("reference acquisition differs from original prospective case")
        proposed = copy.deepcopy(reference)
        proposed["settings"]["expected_poisson_rng_family"] = config["candidate_rng_family"]
        if candidate != proposed:
            raise ValueError("candidate changes more than the declared Poisson source family")
    templates = declaration["templates"]
    original_templates = [t for t in original["templates"] if t["template_id"] == config["template_id"]]
    if (len(templates) != 1 or templates != original_templates
            or len(templates[0]["motifs"]) != 9
            or {m["position"] for m in templates[0]["motifs"]} != set(range(9))):
        raise ValueError("all nine unchanged familiar-template motifs required")
    if not set(config["previously_observed_candidate_cases"]) <= set(wanted):
        raise ValueError("undeclared previously observed candidate")
    return cases, references, templates[0]


def verify_runtime_pair(reference: dict, candidate: dict, reference_case: dict,
                        candidate_case: dict, declaration: dict) -> dict:
    """Require equal equations/settings/initial RNG binding apart from source RNG."""
    cfg = declaration["config"]
    for name, report, case, library, family in (
            ("reference", reference, reference_case, declaration["reference_native_library_sha256"],
             cfg["reference_rng_family"]),
            ("candidate", candidate, candidate_case, declaration["native_library_sha256"],
             cfg["candidate_rng_family"])):
        if (report["status"] != "EXECUTED" or not report["checks"]
                or not all(report["checks"].values())
                or report["graph_manifest_sha256"] != declaration["graph_manifest_sha256"]
                or report["initial_state_rng_library_sha256"] != declaration["source_rng_sha256"]
                or report["native_library_sha256"] != library
                or report["initial_state_seed"] != case["initial_state_seed"]
                or report["background_seed"] != case["background_seed"]
                or asdict(Execution(**report["config"])) != asdict(Execution(**case["settings"]))):
            raise ValueError("episode runtime, control, initial state or library identity differs")
        actual_family = report.get("native_poisson_rng_family")
        if ((name == "candidate" and actual_family != family)
                or (actual_family is not None and actual_family != family)):
            raise ValueError("reported Poisson source family differs")
    keys = ("dt_ms", "duration_ms", "gpu_source_revision", "cortical_neurons",
            "thalamic_relays", "recurrent_tc_edges", "background_edges",
            "native_connections", "poisson_native_delay_offset_ticks")
    if any(reference[key] != candidate[key] for key in keys):
        raise ValueError("paired episodes differ in non-RNG numerical/source settings")
    return {name: {"native_library_sha256": report["native_library_sha256"],
                   "rng_family": cfg[name + "_rng_family"],
                   "rng_family_evidence": ("runtime_export" if "native_poisson_rng_family" in report
                                          else "pinned_reference_library_and_frozen_declaration"),
                   "implementation_sha256": report.get("implementation_sha256"),
                   "initial_state_rng_library_sha256": report["initial_state_rng_library_sha256"]}
            for name, report in (("reference", reference), ("candidate", candidate))}


def paired_bootstrap_multiplicities(blocks: int, replicates: int, seed: int) -> np.ndarray:
    if type(blocks) is not int or blocks < 2 or type(replicates) is not int or replicates < 1:
        raise ValueError("at least two blocks and positive bootstrap count required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, blocks, size=(replicates, blocks))
    result = np.zeros((replicates, blocks), dtype=np.int64)
    np.add.at(result, (np.repeat(np.arange(replicates), blocks), indices.ravel()), 1)
    return result


def time_inner_product_gram(values: np.ndarray) -> np.ndarray:
    """Return D[b,c,p,q] after subtracting each matched-time across-block mean."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 1 or not np.isfinite(x).all():
        raise ValueError("finite block by matched-time by population array required")
    centered = x - x.mean(axis=0, keepdims=True)
    blocks, times, populations = x.shape
    flat = centered.transpose(0, 2, 1).reshape(blocks * populations, times)
    return (flat @ flat.T / times).reshape(blocks, populations, blocks, populations).transpose(0, 2, 1, 3)


def covariance_from_gram(gram: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    """Exact mean matched-time sample covariance for whole-record resamples."""
    d = np.asarray(gram, dtype=np.float64)
    counts = np.asarray(multiplicities)
    if (d.ndim != 4 or d.shape[0] < 2 or d.shape[0] != d.shape[1]
            or d.shape[2] != d.shape[3] or not np.isfinite(d).all()
            or counts.ndim != 2 or counts.shape[1] != d.shape[0]
            or counts.dtype.kind not in "iu" or np.any(counts < 0)
            or not np.all(counts.sum(1) == d.shape[0])):
        raise ValueError("valid time Gram and complete block-bootstrap multiplicities required")
    blocks = d.shape[0]
    weights = counts.astype(float) / blocks
    diagonal = d[np.arange(blocks), np.arange(blocks)]
    first = np.einsum("rb,bpq->rpq", weights, diagonal)
    outer = (weights[:, :, None] * weights[:, None, :]).reshape(len(weights), blocks * blocks)
    second = (outer @ d.reshape(blocks * blocks, -1)).reshape(first.shape)
    result = blocks / (blocks - 1) * (first - second)
    result = (result + result.transpose(0, 2, 1)) / 2
    diagonal_result = np.diagonal(result, axis1=1, axis2=2)
    tolerance = 128 * np.finfo(float).eps * np.maximum(1., np.max(np.abs(first), axis=(1, 2)))
    if np.any(diagonal_result < -tolerance[:, None]):
        raise FloatingPointError("negative covariance diagonal beyond roundoff")
    index = np.arange(d.shape[2])
    result[:, index, index] = np.maximum(0., diagonal_result)
    return result


def pooled_ensemble_fano(covariance: np.ndarray, mean_counts: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean_counts, dtype=float)
    variance = np.diagonal(np.asarray(covariance), axis1=-2, axis2=-1)
    if mean.shape != variance.shape or np.any(mean < 0):
        raise ValueError("nonnegative aligned count means required")
    output = np.full(mean.shape, np.nan)
    np.divide(variance, mean, out=output, where=mean > 0)
    return output


def difference_interval(reference: float, candidate: float, draws: np.ndarray,
                        level: float) -> dict:
    x = np.asarray(draws, dtype=float)
    if x.ndim != 1 or not len(x) or not 0 < level < 1:
        raise ValueError("nonempty difference replicates and valid interval level required")
    invalid = int((~np.isfinite(x)).sum())
    finite_point = np.isfinite(reference) and np.isfinite(candidate)
    result = {"reference": float(reference) if np.isfinite(reference) else None,
              "candidate": float(candidate) if np.isfinite(candidate) else None,
              "difference": float(candidate - reference) if finite_point else None,
              "ci_low": None, "ci_high": None, "bootstrap_replicates": len(x),
              "invalid_bootstrap_replicates": invalid}
    if not finite_point:
        return {**result, "interval_status": "UNDEFINED_POINT_ESTIMAND"}
    if invalid:
        return {**result, "interval_status": "UNDEFINED_BOOTSTRAP_REPLICATES"}
    lower, upper = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return {**result, "ci_low": float(lower), "ci_high": float(upper),
            "interval_status": "DEGENERATE_INTERVAL" if lower == upper else "POINTWISE_PERCENTILE_INTERVAL"}


def aggregate_aligned(counts: np.ndarray, edges: np.ndarray, start_tick: int,
                      stop_tick: int, width_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    x, edges = np.asarray(counts), np.asarray(edges)
    if (x.ndim != 3 or x.dtype.kind not in "iu" or np.any(x < 0)
            or edges.ndim != 1 or edges.dtype.kind not in "iu" or len(edges) != x.shape[1] + 1
            or len(edges) < 2 or np.any(np.diff(edges) <= 0)
            or not np.all(np.diff(edges) == edges[1] - edges[0])):
        raise ValueError("complete integer count trajectories and uniform edges required")
    base = int(edges[1] - edges[0])
    if (width_ticks < base or width_ticks % base or start_tick not in edges or stop_tick not in edges
            or start_tick >= stop_tick or (stop_tick - start_tick) % width_ticks):
        raise ValueError("analysis window must align exactly with complete bins")
    left = int(np.searchsorted(edges, start_tick))
    right = int(np.searchsorted(edges, stop_tick))
    factor = width_ticks // base
    result = x[:, left:right].reshape(len(x), (right - left) // factor, factor, x.shape[2]).sum(2)
    return result, np.arange(start_tick, stop_tick + 1, width_ticks, dtype=np.int64)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _array_identity(left: Path, right: Path) -> dict:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("paired initial-state/mapping array names differ")
        for name in a.files:
            if a[name].dtype != b[name].dtype or not np.array_equal(a[name], b[name]):
                raise ValueError("paired initial-state/mapping values differ: " + name)
        return {name: True for name in a.files}


def extract_pair(reference_path: Path, candidate_path: Path, reference_field: Path,
                 reference_case: dict, candidate_case: dict, declaration: dict,
                 graph_meta: dict, layout: dict, template: dict, observer: PeriodicPopulationGrid) -> dict:
    """Verify one completed pair and project all native events without a unit-time matrix."""
    reports = [_json(path / "report.json") for path in (reference_path, candidate_path)]
    runtime = verify_runtime_pair(*reports, reference_case, candidate_case, declaration)
    initial = {name: _array_identity(reference_path / name, candidate_path / name)
               for name in ("initial_parameters.npz", "unit_mapping.npz")}
    for name in ("expected_adjacencies.npy", "native_adjacencies.npy"):
        arrays = [np.load(path / name, allow_pickle=False) for path in (reference_path, candidate_path)]
        if not np.array_equal(*arrays):
            raise ValueError("paired adjacency readbacks differ")
        initial[name] = True
    field_summary = _json(reference_field / "summary.json")
    if (field_summary["status"] != "DESCRIBED" or field_summary["cortical_refractory_violations"]
            or field_summary["settings"] != {"bin_ms": 5., "cells_per_axis": 4,
                                             "start_ms": 0., "pulse_window_ms": 10.}
            or field_summary["episode_report_sha256"] != sha256_file(reference_path / "report.json")):
        raise ValueError("reference field view differs from the declared observer")
    for name, digest in field_summary["artifacts"].items():
        if sha256_file(reference_field / name) != digest:
            raise ValueError("reference field artifact changed")
    populations = list(graph_meta["resolved_source_network"]["populations"][:-1])
    if len(populations) != 8 or len(observer.mapping.field_ids) != 128:
        raise ValueError("source eight-population four-by-four observer required")
    units = np.asarray([(layout["populations"] == p).sum() for p in populations], dtype=np.int64)
    if np.any(units <= 0) or units.sum() != 100000:
        raise ValueError("complete100000-cortical-unit population map required")
    full_edges = np.arange(0, 60000 + 1, 50, dtype=np.int64)
    # Native dt is bound before using the fixed five-millisecond field grid.
    if reports[0]["dt_ms"] != .1 or reports[0]["duration_ms"] != 6000.:
        raise ValueError("declared0.1ms6000ms native episode required")
    motifs = sorted(template["motifs"], key=lambda row: row["position"])
    focal_ticks = ticks([m["focal_arrival_ms"] for m in motifs], .1)
    window_ticks = int(ticks([declaration["config"]["response_window_ms"]], .1)[0])
    if np.any(focal_ticks < 10000) or np.any(focal_ticks + window_ticks > 60000):
        raise ValueError("motif response window outside declared analysis support")
    population_ids = np.full(len(layout["unit_ids"]), -1, dtype=np.int64)
    for index, population in enumerate(populations):
        population_ids[layout["populations"] == population] = index
    outputs = []
    for kind, path, case in (("reference", reference_path, reference_case),
                              ("candidate", candidate_path, candidate_case)):
        report, events, indices = _verified_events(path, case, declaration["graph_manifest_sha256"])
        for name, digest in report["artifacts"].items():
            if sha256_file(path / name) != digest:
                raise ValueError("episode artifact changed: " + name)
        for name in ("unit_ids", "populations", "positions_mm"):
            if not np.array_equal(events[name], layout[name]):
                raise ValueError("event/layout identity mismatch: " + name)
        controls = verify_controls(events, indices, case, graph_meta, report["dt_ms"])
        if controls["tc_arrival_ticks"] != ticks(template["pulse_arrivals_ms"], .1).tolist():
            raise ValueError("controls differ from the declared stimulus template")
        if np.any(events["spike_ticks"] < 1) or np.any(events["spike_ticks"] > 60000):
            raise ValueError("native events outside the declared simulated support")
        cortical = events["populations"][indices] != "TC"
        times = events["spike_ticks"][cortical]
        event_population = population_ids[indices[cortical]]
        projected = observer.project_events(events["spike_unit_ids"][cortical], times,
            observed_unit_ids=observer.mapping.unit_ids, bin_edges_ticks=full_edges, dt_seconds=.0001)
        field_counts = projected.field.count_sum
        inside = (times >= 0) & (times < 60000)
        counts = np.bincount((times[inside] // 50) * 8 + event_population[inside],
                            minlength=1200 * 8).reshape(1200, 8)
        grouped = np.column_stack([field_counts[:, observer.population_labels == p].sum(1)
                                   for p in populations])
        if not np.array_equal(counts, grouped) or not projected.field.complete.all():
            raise ValueError("full-population and complete spatial field counts disagree")
        responses = np.asarray([np.bincount(event_population[(times >= start) & (times < start + window_ticks)],
                                            minlength=8) for start in focal_ticks], dtype=np.int64)
        if kind == "reference":
            with np.load(reference_field / "spatial_fields.npz", allow_pickle=False) as existing:
                pairs = ((existing["count_sum"], field_counts),
                         (existing["bin_edges_ticks"], full_edges),
                         (existing["field_ids"], observer.mapping.field_ids),
                         (existing["source_unit_ids"], observer.mapping.unit_ids),
                         (existing["unit_field_index"], observer.mapping.group_index),
                         (existing["total_unit_count"], projected.field.total_unit_count))
                if not all(np.array_equal(a, b) for a, b in pairs):
                    raise ValueError("raw-event projection differs from canonical reference fields")
        outputs.append({"population_counts": counts, "field_counts": field_counts,
                        "motif_counts": responses, "controls": controls,
                        "raw_report_sha256": sha256_file(path / "report.json"),
                        "artifacts": report["artifacts"], "raw_episode": str(path.resolve()),
                        "recorded_cortical_events": int(len(times)),
                        "right_boundary_cortical_events": int(np.sum(times == 60000))})
    if outputs[0]["controls"] != outputs[1]["controls"]:
        raise ValueError("paired source controls differ")
    return {"reference": outputs[0], "candidate": outputs[1], "runtime": runtime,
            "initial_state_and_mapping": initial, "population_units": units,
            "populations": populations, "bin_edges_ticks": full_edges,
            "motif_focal_ticks": focal_ticks}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(config_path: Path, design_path: Path, original_design_path: Path,
            reference: Path, reference_fields: Path, candidate: Path,
            graph: Path, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    declaration, original = _json(design_path), _json(original_design_path)
    reference_design = _json(reference / "design.json")
    cases, references, template = validate_cases(declaration, original, reference_design, config)
    require_acquired(_json(reference / "acquisition.json"), [c["case_id"] for c in reference_design["cases"]])
    require_acquired(_json(candidate / "acquisition.json"), [c["case_id"] for c in cases])
    if (_json(candidate / "design.json") != declaration
            or sha256_file(original_design_path) != declaration["original_design_sha256"]
            or reference_design["prospective_design_sha256"] != declaration["original_design_sha256"]
            or reference.resolve() != Path(declaration["reference_acquisition"]).resolve()
            or sha256_file(graph / "manifest.json") != declaration["graph_manifest_sha256"]):
        raise ValueError("prospective design/acquisition/graph binding changed")
    binding = _json(reference_fields / "binding.json")
    if (binding["split"] != "train" or binding["design_sha256"] != sha256_file(reference / "design.json")
            or binding["graph_manifest_sha256"] != declaration["graph_manifest_sha256"]):
        raise ValueError("reference field binding differs")
    meta = _json(graph / "manifest.json")
    if sha256_file(graph / "layout.npz") != meta["layout_sha256"]:
        raise ValueError("source layout changed")
    with np.load(graph / "layout.npz", allow_pickle=False) as data:
        layout = {k: data[k] for k in ("unit_ids", "populations", "positions_mm")}
    cortical = layout["populations"] != "TC"
    observer = PeriodicPopulationGrid.from_positions(layout["unit_ids"][cortical],
        layout["populations"][cortical], layout["positions_mm"][cortical],
        extent_mm=meta["resolved_source_network"]["extent"], cells_per_axis=4,
        definition="Source cortical soma coordinates; same fixed source population/spatial count observer")
    count_parts = {name: [] for name in ("reference", "candidate")}
    field_parts = {name: [] for name in count_parts}
    response_parts = {name: [] for name in count_parts}
    evidence = []
    for case in cases:
        identity = case["case_id"]
        pair = extract_pair(reference / identity, candidate / identity, reference_fields / identity,
                            references[identity], case, declaration, meta, layout, template, observer)
        for source in count_parts:
            for key, target in (("population_counts", count_parts), ("field_counts", field_parts),
                                ("motif_counts", response_parts)):
                target[source].append(pair[source].pop(key))
        evidence.append({"case_id": identity, "block": case["block"],
                         "initial_state_seed": case["initial_state_seed"],
                         "background_seed": case["background_seed"],
                         "reference": pair["reference"], "candidate": pair["candidate"],
                         "runtime": pair["runtime"], "initial_state_and_mapping": pair["initial_state_and_mapping"]})
        print(json.dumps({"status": "EXTRACTING_COMPLETE_COHORT", "completed_pairs": len(evidence),
                          "expected_pairs": len(cases), "last_case": identity}), flush=True)
    counts = {name: np.stack(value) for name, value in count_parts.items()}
    fields = {name: np.stack(value) for name, value in field_parts.items()}
    responses = {name: np.stack(value) for name, value in response_parts.items()}
    units, populations, edges = pair["population_units"], pair["populations"], pair["bin_edges_ticks"]
    n = len(cases)
    multiplicities = paired_bootstrap_multiplicities(n, config["bootstrap_replicates"], config["bootstrap_seed"])
    weights = multiplicities / n
    uniform = np.ones((1, n), dtype=np.int64)
    start_tick, stop_tick = map(int, ticks([config["analysis_start_ms"], config["analysis_stop_ms"]], .1))
    rows, rates_by_block, motif_by_block = [], [], []
    covariance_arrays, metric_arrays = {}, {}
    def add_row(metric, population, bin_ms, motif, units_text, reference_value, candidate_value, draws, bins):
        metadata = {"metric": metric, "population": population, "bin_width_ms": bin_ms,
                    "motif_position": motif["position"] if motif else None,
                    "focal_arrival_ms": motif["focal_arrival_ms"] if motif else None,
                    "local_conditioning": motif["local_conditioning"] if motif else None,
                    "focal_present": motif["focal_present"] if motif else None,
                    "units": units_text, "independent_blocks": n, "matched_time_bins": bins,
                    "interval_level": config["interval_level"]}
        rows.append({**metadata, **difference_interval(reference_value, candidate_value, draws, config["interval_level"])})
    window_counts = {source: aggregate_aligned(x, edges, start_tick, stop_tick, 50)[0]
                     for source, x in counts.items()}
    seconds = (stop_tick - start_tick) * .0001
    block_rates = {name: x.sum(1) / (units * seconds) for name, x in window_counts.items()}
    for p, population in enumerate(populations):
        delta = weights @ (block_rates["candidate"][:, p] - block_rates["reference"][:, p])
        add_row("mean_rate_hz", population, None, None, "Hz",
                block_rates["reference"][:, p].mean(), block_rates["candidate"][:, p].mean(), delta, window_counts["reference"].shape[1])
        for b, case in enumerate(cases):
            for source in counts:
                rates_by_block.append({"source": source, "case_id": case["case_id"], "block": case["block"],
                    "population": population, "units": int(units[p]), "mean_rate_hz": float(block_rates[source][b, p]),
                    "window_count": int(window_counts[source][b, :, p].sum()), "window_seconds": seconds})
    for width in config["bin_widths_ms"]:
        width_ticks = int(ticks([width], .1)[0])
        point, boot, fanos, boot_fanos = {}, {}, {}, {}
        key = str(int(width)) + "ms"
        for source in counts:
            selected, selected_edges = aggregate_aligned(counts[source], edges, start_tick, stop_tick, width_ticks)
            gram = time_inner_product_gram(selected)
            count_covariance = covariance_from_gram(gram, uniform)[0]
            bootstrap_covariance = covariance_from_gram(gram, multiplicities)
            mean_per_block = selected.mean(1)
            point[source] = count_covariance / np.outer(units * width / 1000, units * width / 1000)
            boot[source] = bootstrap_covariance / np.outer(units * width / 1000, units * width / 1000)
            fanos[source] = pooled_ensemble_fano(count_covariance, mean_per_block.mean(0))
            boot_fanos[source] = pooled_ensemble_fano(bootstrap_covariance, weights @ mean_per_block)
            covariance_arrays[source + "_rate_covariance_hz2_" + key] = point[source]
            covariance_arrays[source + "_mean_count_" + key] = mean_per_block.mean(0)
            covariance_arrays[source + "_count_covariance_" + key] = count_covariance
            metric_arrays[source + "_counts_" + key] = selected
            metric_arrays["bin_edges_" + key] = selected_edges
        for p, population in enumerate(populations):
            add_row("ensemble_count_fano", population, width, None, "count_variance/count_mean",
                    fanos["reference"][p], fanos["candidate"][p],
                    boot_fanos["candidate"][:, p] - boot_fanos["reference"][:, p], len(selected_edges) - 1)
            add_row("ensemble_rate_variance_hz2", population, width, None, "Hz2",
                    point["reference"][p, p], point["candidate"][p, p],
                    boot["candidate"][:, p, p] - boot["reference"][:, p, p], len(selected_edges) - 1)
        off = ~np.eye(len(populations), dtype=bool)
        add_row("mean_offdiagonal_ensemble_rate_covariance_hz2", "all_populations", width, None, "Hz2",
                point["reference"][off].mean(), point["candidate"][off].mean(),
                boot["candidate"][:, off].mean(1) - boot["reference"][:, off].mean(1), len(selected_edges) - 1)
        covariance_arrays["candidate_minus_reference_rate_covariance_hz2_" + key] = point["candidate"] - point["reference"]
    for m, motif in enumerate(sorted(template["motifs"], key=lambda row: row["position"])):
        for p, population in enumerate(populations):
            per_source = {source: x[:, m, p] / units[p] for source, x in responses.items()}
            add_row("motif_response_counts_per_neuron", population, config["response_window_ms"], motif,
                    "counts/neuron", per_source["reference"].mean(), per_source["candidate"].mean(),
                    weights @ (per_source["candidate"] - per_source["reference"]), 1)
            for b, case in enumerate(cases):
                for source in responses:
                    motif_by_block.append({"source": source, "case_id": case["case_id"], "block": case["block"],
                        "population": population, "motif_position": motif["position"],
                        "focal_arrival_ms": motif["focal_arrival_ms"], "local_conditioning": motif["local_conditioning"],
                        "focal_present": motif["focal_present"], "window_ms": config["response_window_ms"],
                        "count": int(responses[source][b, m, p]), "neurons": int(units[p]),
                        "counts_per_neuron": float(per_source[source][b])})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "count_trajectories.npz",
        reference_population_counts=counts["reference"], candidate_population_counts=counts["candidate"],
        reference_spatial_counts=fields["reference"], candidate_spatial_counts=fields["candidate"],
        reference_motif_counts=responses["reference"], candidate_motif_counts=responses["candidate"],
        population_units=units, populations=np.asarray(populations), field_ids=observer.mapping.field_ids,
        field_populations=observer.population_labels, field_total_unit_count=np.bincount(observer.mapping.group_index, minlength=128),
        source_unit_ids=observer.mapping.unit_ids, unit_field_index=observer.mapping.group_index,
        bin_edges_ticks=edges, dt_ms=.1, motif_focal_ticks=pair["motif_focal_ticks"],
        case_ids=np.asarray([c["case_id"] for c in cases]), blocks=np.asarray([c["block"] for c in cases]),
        initial_state_seeds=np.asarray([c["initial_state_seed"] for c in cases]),
        background_seeds=np.asarray([c["background_seed"] for c in cases]),
        bootstrap_multiplicities=multiplicities, **metric_arrays)
    np.savez_compressed(output / "ensemble_covariances.npz", populations=np.asarray(populations), **covariance_arrays)
    _write_csv(output / "summary.csv", rows)
    _write_csv(output / "mean_rates_by_block.csv", rates_by_block)
    _write_csv(output / "motif_responses_by_block.csv", motif_by_block)
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    result = {"status": "ANALYZED_DEVELOPMENT_SOURCE_VARIANTS", "independent_blocks": n,
        "reference_episodes": n, "candidate_episodes": n, "cortical_neurons": int(units.sum()),
        "populations": populations, "spatial_fields": 128, "motifs": template["motifs"],
        "analysis_start_ms": config["analysis_start_ms"], "analysis_stop_ms": config["analysis_stop_ms"],
        "bin_widths_ms": config["bin_widths_ms"], "summary_rows": len(rows),
        "bootstrap_replicates": len(multiplicities), "bootstrap_seed": config["bootstrap_seed"],
        "interval_level": config["interval_level"], "interval_scope": "pointwise paired whole-block percentile intervals",
        "fano_definition": "mean_time sampleVar_blocks(count) / mean_time mean_blocks(count)",
        "covariance_definition": "mean_time sampleCov_blocks(per-neuron population rates at each matched time)",
        "offdiagonal_definition": "mean of56 ordered off-diagonal entries of the full8x8 ensemble rate covariance",
        "covariance_is_conditional_on_history_or_noise": False,
        "motif_response_definition": "native-tick count/neuron in[focal_TC_arrival,focal_TC_arrival+20ms); includes omitted focal conditions",
        "motif_responses_are_matched_causal_pulse_effects": False,
        "common_initial_states": True, "common_noise_across_rng_families": False,
        "independence_boundary": "one initial-state/background-seed block; both sources paired within block; one graph and template",
        "variance_components_separately_identified": False,
        "source_change_scope": "native per-edge Poisson RNG family; graph, numerical equations, initial state and controls held fixed",
        "source_change_validation": "separate exact source-stream and deterministic neuronal regression evidence; no neural equivalence claim",
        "previously_observed_candidate_cases": config["previously_observed_candidate_cases"],
        "previous_observation_boundary": "one candidate case was observed before the declared remaining-cohort acquisition",
        "raw_events_retained": True, "all_count_trajectories_retained": True,
        "raw_data_access": "selected familiar_0 training cases only; no calibration/test outcomes accessed",
        "p_values_computed": False, "equivalence_decision": "NOT_EVALUATED", "scientific_reference_qualified": False,
        "neuroscience_finding_qualified": False, "formal_training_completed": False, "formal_training_seeds": [41, 42, 43],
        "independent_statistics_replay": "PENDING",
        "source_identities": {"reference_library_sha256": declaration["reference_native_library_sha256"],
            "candidate_library_sha256": declaration["native_library_sha256"],
            "reference_rng_family": config["reference_rng_family"], "candidate_rng_family": config["candidate_rng_family"],
            "initial_state_rng_library_sha256": declaration["source_rng_sha256"],
            "graph_manifest_sha256": declaration["graph_manifest_sha256"]},
        "input_hashes": {str(path): sha256_file(path) for path in (config_path, design_path, original_design_path,
            reference / "design.json", reference / "acquisition.json", candidate / "design.json",
            candidate / "acquisition.json", reference_fields / "binding.json", graph / "manifest.json", graph / "layout.npz")},
        "implementation_sha256": sha256_file(Path(__file__)),
        "summary": rows,
        "artifacts": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "design", "original-design", "reference", "reference-fields",
                 "candidate", "graph", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.design, args.original_design, args.reference,
                     args.reference_fields, args.candidate, args.graph, args.output)
    print(json.dumps({k: result[k] for k in ("status", "independent_blocks", "summary_rows", "independent_statistics_replay")}))


if __name__ == "__main__":
    main()

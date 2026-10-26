"""Summarize complete3000 PFC development observations without a scientific gate.

Conditions share each joint history. Three histories per parameter circuit
supply descriptions only. Native unit identities and signed covariances stay
visible; time bins and units do not increase independent history counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from . import pfc_reference_quality_summary as quality
from .pfc_parameter_domain_quality import validate_plan, BLOCKS, EPISODES, SPLITS
from .provenance import cpu_execution_provenance, freeze_json, sha256_file

SCHEMA = "pfc_parameter_domain_summary_v1"
OBSERVATION_BIN_TICKS = 80
OBSERVATION_BIN_SECONDS = .001


def observation_mapping(path: Path, plan: dict) -> dict[str, dict]:
    value = quality.read_json(path)
    expected = {row["identity"]["episode_id"] for row in plan["rows"]}
    quality.require(set(value) == expected, "PFC_DOMAIN_SUMMARY_COMPLETE3000_MANIFEST_REQUIRED")
    seen = set()
    result = {}
    for episode, entry in value.items():
        quality.require(isinstance(entry, dict) and set(entry) == {
            "path", "development_report_sha256", "observation_report_sha256"},
            "PFC_DOMAIN_SUMMARY_BOUND_OBSERVATION_ENTRY_REQUIRED")
        directory = Path(entry["path"])
        if not directory.is_absolute():
            directory = path.parent / directory
        directory = directory.resolve()
        quality.require(directory not in seen, "PFC_DOMAIN_SUMMARY_DUPLICATE_DIRECTORY")
        seen.add(directory)
        result[episode] = dict(entry, path=str(directory))
    return result


def load_observation(entry: dict, expected: dict) -> tuple[dict, list[dict], dict, dict, dict]:
    root = Path(entry["path"])
    directory = root / "development5ms"
    observation = root / "observation1ms"
    quality.require(sha256_file(directory / "report.json") == entry["development_report_sha256"]
        and sha256_file(observation / "report.json") == entry["observation_report_sha256"],
        "PFC_DOMAIN_SUMMARY_BOUND_REPORT_CHANGED")
    report, phases, old = quality.load_cell(directory, expected)
    for population, refractory_ticks in (("E", 160), ("I", 80)):
        check = report["source_checks"].get("refractory_spacing_" + population, {})
        quality.require(check.get("passed") is True and check.get("violation_count") == 0
            and check.get("minimum_required_ticks") == refractory_ticks,
            "PFC_DOMAIN_SUMMARY_BINARY1MS_REFRACTORY_REQUIRED")
    frozen = quality.read_json(observation / "report.json")
    quality.require(frozen.get("status") == "DESCRIBED_FROZEN1MS_SOURCE_OBSERVATION"
        and frozen.get("identity") == expected["identity"] and frozen.get("condition") == expected["condition"]
        and frozen.get("config_sha256") == expected["config_sha256"]
        and frozen.get("development_readout_report_sha256") == entry["development_report_sha256"]
        and frozen.get("source_delivery_sha256") == report["source_delivery_sha256"]
        and frozen.get("source_dataset_catalog_sha256") == report["output_dataset_catalog_sha256"]
        and frozen.get("source_archive") == report["source_archive"]
        and frozen.get("core_units") == 100000 and frozen.get("bins") == 5000
        and frozen.get("population_fields") == 17 and frozen.get("bin_ticks") == 80
        and frozen.get("bin_ms") == 1. and frozen.get("native_event_tick_ms") == .0125
        and frozen.get("native_event_ticks_preserved") is True
        and frozen.get("all_unit_integer_reaggregation_exact") is True
        and frozen.get("existing5ms_counts_exact") is True
        and frozen.get("source_qualification_granted") is False
        and frozen.get("scientific_claim_qualified") is False and frozen.get("production_episodes") == 0,
        "PFC_DOMAIN_SUMMARY_FROZEN_OBSERVATION_CHANGED")
    current = frozen.get("observation_map", {})
    quality.require(current.get("frozen") is True
        and current.get("unit_order") == "all_E_then_all_I_ascending_source_indices"
        and current.get("populations") == ["E", "I"]
        and current.get("temporal_resolution", {}).get("scientific_bin_ticks") == 80
        and current.get("temporal_resolution", {}).get("scientific_bin_ms") == 1.,
        "PFC_DOMAIN_SUMMARY_OBSERVATION_MAP_CHANGED")
    path = observation / "observation.npz"
    quality.require(sha256_file(path) == frozen["output_sha256"], "PFC_DOMAIN_SUMMARY_OBSERVATION_ARRAY_CHANGED")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    fields = arrays["population_field_counts"]
    units = arrays["unit_counts"]
    terminal = arrays["terminal_unit_counts"]
    quality.require(fields.shape == (5000,17) and units.shape == terminal.shape == (100000,)
        and fields.dtype.kind in "iu" and units.dtype.kind in "iu" and terminal.dtype.kind in "iu"
        and np.all(fields >= 0) and np.all(units >= 0) and np.all(terminal >= 0)
        and int(fields.sum()) == int(units.sum()) == frozen["retained_events"]
        and int(terminal.sum()) == frozen["terminal_events"]
        and frozen["retained_events"] + frozen["terminal_events"] == report["events"]
        and frozen["terminal_events"] == report["terminal_tick_events"],
        "PFC_DOMAIN_SUMMARY_EVENT_OR_UNIT_DENOMINATOR_CHANGED")
    np.testing.assert_array_equal(fields.reshape(1000,5,17).sum(axis=1), old["population_field_counts"])
    np.testing.assert_array_equal(units, old["phase_unit_counts"].sum(axis=0))
    quality.require(all(row["boundary_aligned"] is True for row in frozen["source_control_transitions"]),
        "PFC_DOMAIN_SUMMARY_UNALIGNED_CONTROL_TRANSITION")
    for name in ("vector_x","vector_y"):
        quality.require(arrays[name].shape == (5000,) and np.isfinite(arrays[name]).all(),
            "PFC_DOMAIN_SUMMARY_FINITE_VECTOR_TRAJECTORY_REQUIRED")
    return report, phases, old, frozen, arrays


def phase_fluctuations(fields: np.ndarray, phase_counts: np.ndarray, phase_edges: np.ndarray,
                       field_sizes: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Separate temporal and matched-history covariance at the frozen1ms bin.

    The source refractory period permits at most one event per unit in1ms.
    Thus sum_t X_i(t)^2 equals its phase count. This supplies the temporal
    single-unit diagonal without constructing a time by unit matrix. Matching
    counts from different histories is still needed for the history diagonal;
    that quantity cannot be inferred from phase totals.
    """
    fields, phase_counts, phase_edges, field_sizes = map(np.asarray,
        (fields, phase_counts, phase_edges, field_sizes))
    quality.require(fields.ndim == 3 and fields.shape[0] == len(BLOCKS)
        and fields.shape[2] == len(field_sizes) and fields.dtype.kind in "iu"
        and np.all(fields >= 0) and field_sizes.dtype.kind in "iu"
        and len(field_sizes) == 17 and np.all(field_sizes > 0),
        "PFC_DOMAIN_PHASE_COMPLETE_INTEGER_FIELDS_REQUIRED")
    quality.require(phase_edges.ndim == 1 and phase_edges.dtype.kind in "iu"
        and phase_edges[0] == 0 and phase_edges[-1] == fields.shape[1] * OBSERVATION_BIN_TICKS
        and np.all(np.diff(phase_edges) > OBSERVATION_BIN_TICKS)
        and np.all(phase_edges % OBSERVATION_BIN_TICKS == 0),
        "PFC_DOMAIN_PHASE_BOUNDARY_ALIGNED1MS_REQUIRED")
    units = int(field_sizes.sum())
    quality.require(phase_counts.shape == (len(BLOCKS), len(phase_edges) - 1, units)
        and phase_counts.dtype.kind in "iu" and np.all(phase_counts >= 0),
        "PFC_DOMAIN_PHASE_COMPLETE_UNIT_COUNTS_REQUIRED")
    mapping = np.repeat(np.arange(17), field_sizes)
    populations = (("excitatory", range(16), 0, int(field_sizes[:16].sum())),
        ("inhibitory", [16], int(field_sizes[:16].sum()), units),
        ("all_units", range(17), 0, units))
    covariance_rows, population_rows = [], []
    for phase, (first, stop) in enumerate(zip(phase_edges[:-1], phase_edges[1:])):
        selected = fields[:, int(first) // OBSERVATION_BIN_TICKS:int(stop) // OBSERVATION_BIN_TICKS]
        bins = selected.shape[1]
        counts = phase_counts[:, phase]
        quality.require(np.all(counts <= bins) and np.all(selected <= field_sizes),
            "PFC_DOMAIN_PHASE_BINARY1MS_COUNT_BOUND_VIOLATION")
        for block in range(len(BLOCKS)):
            np.testing.assert_array_equal(selected[block].sum(axis=0),
                np.bincount(mapping, weights=counts[block], minlength=17))
        temporal, between = quality.covariance_axes(selected)
        for left in range(17):
            for right in range(17):
                scale = float(field_sizes[left]) * float(field_sizes[right]) * OBSERVATION_BIN_SECONDS**2
                covariance_rows.append(dict(phase_index=phase, start_tick=int(first), stop_tick=int(stop),
                    leading_field=left, trailing_field=right, blocks=len(BLOCKS), time_bins=bins,
                    bin_ticks=OBSERVATION_BIN_TICKS, bin_ms=1.,
                    temporal_covariance_counts2=float(temporal[left, right]),
                    between_block_matched_time_covariance_counts2=float(between[left, right]),
                    temporal_covariance_rate_Hz2=float(temporal[left, right]) / scale,
                    between_block_matched_time_covariance_rate_Hz2=float(between[left, right]) / scale))
        for name, groups, lo, hi in populations:
            population_units = hi - lo
            values = selected[:, :, list(groups)].sum(axis=2) / population_units
            history_variance = float(values.var(axis=0, ddof=1).mean())
            for block, block_index in enumerate(BLOCKS):
                unit_totals = counts[block, lo:hi].astype(np.float64)
                # Exact binary-count second moment; ddof1 matches temporal covariance.
                diagonal = float((unit_totals * (1. - unit_totals / bins)).sum()
                    / ((bins - 1) * population_units**2))
                variance = float(values[block].var(ddof=1))
                population_rows.append(dict(phase_index=phase, start_tick=int(first), stop_tick=int(stop),
                    block_index=block_index, population=name, units=population_units, time_bins=bins,
                    bin_ticks=OBSERVATION_BIN_TICKS, bin_ms=1.,
                    mean_rate_Hz=float(values[block].mean()) / OBSERVATION_BIN_SECONDS,
                    temporal_population_mean_variance_rate_Hz2=variance / OBSERVATION_BIN_SECONDS**2,
                    temporal_diagonal_contribution_rate_Hz2=diagonal / OBSERVATION_BIN_SECONDS**2,
                    temporal_ordered_offdiagonal_contribution_rate_Hz2=(variance - diagonal) / OBSERVATION_BIN_SECONDS**2,
                    between_block_matched_time_population_mean_variance_rate_Hz2=history_variance / OBSERVATION_BIN_SECONDS**2,
                    between_block_matched_time_diagonal_contribution_rate_Hz2=None,
                    between_block_matched_time_ordered_offdiagonal_contribution_rate_Hz2=None,
                    history_diagonal_status="UNAVAILABLE_WITHOUT_MATCHED1MS_UNIT_COUNTS",
                    temporal_diagonal_method="binary1ms_counts_from_validated_source_refractory_and_phase_totals"))
    return covariance_rows, population_rows


def summarize_circuit(circuit: int, loaded: dict[int, list[tuple]]) -> tuple[list, list, list, list, dict]:
    phase_edges = loaded[0][0][2]["phase_edges_ticks"]
    values = np.empty((3,len(phase_edges)-1,len(quality.RATE_FIELDS),10))
    phase_rows, paired_rows, contrast_rows, covariance_rows = [], [], [], []
    unit_arrays = {}
    for condition_index, condition in enumerate(quality.CONDITIONS):
        cells = loaded[condition_index]
        reports, phases, old, observations, arrays = zip(*cells)
        for a in old:
            quality.require(np.array_equal(a["phase_edges_ticks"],phase_edges), "PFC_DOMAIN_SUMMARY_PHASE_CLOCK_CHANGED")
        for phase in range(len(phase_edges)-1):
            for field_index, metric in enumerate(quality.RATE_FIELDS):
                samples = [float(rows[phase][metric]) for rows in phases]
                values[:,phase,field_index,condition_index] = samples
                phase_rows.append(dict(circuit_index=circuit,split=reports[0]["identity"]["split"],
                    condition_id=condition,phase_index=phase,metric=metric,start_tick=int(phase_edges[phase]),
                    stop_tick=int(phase_edges[phase+1]),**quality.descriptive(samples)))
            angles = [abs(float(rows[phase]["circular_cue_error_deg"])) if rows[phase]["circular_cue_error_deg"] else None for rows in phases]
            phase_rows.append(dict(circuit_index=circuit,split=reports[0]["identity"]["split"],
                condition_id=condition,phase_index=phase,metric="absolute_circular_cue_error_deg",
                start_tick=int(phase_edges[phase]),stop_tick=int(phase_edges[phase+1]),**quality.descriptive(angles)))
        fields = np.stack([a["population_field_counts"] for a in arrays])
        temporal, between = quality.covariance_axes(fields)
        sizes = np.r_[np.full(16,5000),20000]
        rates = fields / sizes / .001
        temporal_rates, between_rates = quality.covariance_axes(rates)
        for left in range(17):
            for right in range(17):
                covariance_rows.append(dict(circuit_index=circuit,condition_id=condition,leading_field=left,
                    trailing_field=right,blocks=3,time_bins=5000,bin_ticks=80,bin_ms=1.,
                    temporal_covariance_counts2=float(temporal[left,right]),
                    between_block_matched_time_covariance_counts2=float(between[left,right]),
                    temporal_covariance_rate_Hz2=float(temporal_rates[left,right]),
                    between_block_matched_time_covariance_rate_Hz2=float(between_rates[left,right])))
        key = f"condition_{condition_index:02d}"
        unit_arrays[key+"_population_field_counts1ms"] = fields
        unit_arrays[key+"_vector_x1ms"] = np.stack([a["vector_x"] for a in arrays])
        unit_arrays[key+"_vector_y1ms"] = np.stack([a["vector_y"] for a in arrays])
        phase_units = np.stack([a["phase_unit_counts"] for a in old])
        unit_arrays[key+"_phase_unit_mean_counts"] = phase_units.mean(axis=0)
        unit_arrays[key+"_between_block_phase_unit_variance_counts2"] = phase_units.var(axis=0,ddof=1)
        unit_arrays[key+"_mean_retained_unit_counts"] = np.stack([a["unit_counts"] for a in arrays]).mean(axis=0)
    cue, interaction = quality.paired_contrasts(values)
    for label,data,conditions in (("cue_minus_omission",cue,quality.CONDITIONS[:5]),
            ("distractor_difference_of_differences",interaction,quality.CONDITIONS[1:5])):
        for phase in range(data.shape[1]):
            for f,metric in enumerate(quality.RATE_FIELDS):
                for k,condition in enumerate(conditions):
                    for b,block in enumerate(BLOCKS):
                        paired_rows.append(dict(circuit_index=circuit,block_index=block,phase_index=phase,
                            contrast=label,condition_id=condition,metric=metric,value=float(data[b,phase,f,k])))
                    contrast_rows.append(dict(circuit_index=circuit,phase_index=phase,contrast=label,
                        condition_id=condition,metric=metric,**quality.descriptive(data[:,phase,f,k])))
    return phase_rows, paired_rows, contrast_rows, covariance_rows, unit_arrays


def summarize_phase_fluctuations(circuit: int, loaded: dict[int, list[tuple]]) -> tuple[list[dict], list[dict]]:
    covariance_rows, population_rows = [], []
    for condition_index, condition in enumerate(quality.CONDITIONS):
        cells = loaded[condition_index]
        fields = np.stack([cell[4]["population_field_counts"] for cell in cells])
        phase_counts = np.stack([cell[2]["phase_unit_counts"] for cell in cells])
        edges = cells[0][2]["phase_edges_ticks"]
        sizes = np.r_[np.full(16, 5000, dtype=np.int64), 20000]
        covariance, population = phase_fluctuations(fields, phase_counts, edges, sizes)
        identity = dict(circuit_index=circuit, condition_id=condition, split=cells[0][0]["identity"]["split"])
        covariance_rows.extend(dict(identity, **row) for row in covariance)
        population_rows.extend(dict(identity, **row) for row in population)
    return covariance_rows, population_rows


def aggregate(preparation: Path, observations: Path, output: Path) -> dict:
    plan = quality.read_json(preparation / "execution.json")
    validate_plan(plan)
    mapping = observation_mapping(observations,plan)
    quality.require(not output.exists(), "PFC_DOMAIN_SUMMARY_OUTPUT_EXISTS")
    execution = cpu_execution_provenance()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pfc-domain-summary-",dir="/tmp") as temp:
        work = Path(temp)
        phase_rows, paired_rows, contrast_rows, covariance_rows, costs, bindings, pairings = [],[],[],[],[],[],[]
        phase_covariance_rows, phase_population_rows = [], []
        expected = {row["identity"]["episode_id"]: row for row in plan["rows"]}
        observation_map = None
        for circuit in range(100):
            loaded = {}
            for k in range(10):
                loaded[k] = []
                for b in BLOCKS:
                    episode = f"pfc_working_memory_circuit_{circuit:03d}_block_{b:03d}_condition_{k:02d}"
                    row = expected[episode]
                    config = quality.relative_file(preparation,row["config_path"])
                    quality.require(sha256_file(config) == row["config_sha256"], "PFC_DOMAIN_SUMMARY_RECIPE_CHANGED")
                    began = time.perf_counter()
                    cell = load_observation(mapping[episode],row)
                    loaded[k].append(cell)
                    report,_,_,frozen,_ = cell
                    if observation_map is None:
                        observation_map = frozen["observation_map"]
                    quality.require(frozen["observation_map"] == observation_map,
                        "PFC_DOMAIN_SUMMARY_COHORT_OBSERVATION_MAP_CHANGED")
                    bindings.append(dict(episode_id=episode,**mapping[episode],source_archive=report["source_archive"],
                        source_delivery_sha256=report["source_delivery_sha256"],config_sha256=row["config_sha256"]))
                    costs.append(dict(episode_id=episode,source_cost=report.get("source_cost"),
                        source_cost_status="recorded" if report.get("source_cost") is not None else "unavailable",
                        acquired_readout_cost=frozen.get("authenticated_read_and_joint_aggregation_seconds"),
                        acquired_readout_cost_status="recorded" if frozen.get("authenticated_read_and_joint_aggregation_seconds") is not None else "unavailable_in_existing_readout_report",
                        summary_authentication_and_loading_seconds=time.perf_counter()-began))
            circuit_pairings = []
            for b_index,b in enumerate(BLOCKS):
                rows = [expected[f"pfc_working_memory_circuit_{circuit:03d}_block_{b:03d}_condition_{k:02d}"] for k in range(10)]
                reports = [loaded[k][b_index][0] for k in range(10)]
                circuit_pairings.append(quality.paired_block_binding(rows,reports))
            quality.require(len({r["initial_voltage_sha256"] for r in circuit_pairings}) == 3
                and len({r["final_rng_sha256"] for r in circuit_pairings}) == 3,
                "PFC_DOMAIN_SUMMARY_REUSED_HISTORY_INPUTS")
            pairings.extend(circuit_pairings)
            p,a,c,v,u = summarize_circuit(circuit,loaded)
            phase_covariance, phase_population = summarize_phase_fluctuations(circuit, loaded)
            phase_covariance_rows.extend(phase_covariance)
            phase_population_rows.extend(phase_population)
            phase_rows.extend(p);paired_rows.extend(a);contrast_rows.extend(c);covariance_rows.extend(v)
            u.update(time_start_ticks=np.arange(5000,dtype=np.int64)*80,
                time_stop_ticks=np.arange(1,5001,dtype=np.int64)*80,
                source_unit_ids=np.arange(100000,dtype=np.int64),field_unit_counts=np.r_[np.full(16,5000),20000],
                block_indices=np.asarray(BLOCKS,dtype=np.int64))
            np.savez_compressed(work/f"circuit_{circuit:03d}_joint_activity.npz",**u)
        quality.write_rows(work/"phase_block_summary.csv",phase_rows)
        quality.write_rows(work/"paired_block_contrasts.csv",paired_rows)
        quality.write_rows(work/"paired_contrast_summary.csv",contrast_rows)
        quality.write_rows(work/"population_field_covariance1ms.csv",covariance_rows)
        quality.write_rows(work/"phase_field_covariance1ms.csv", phase_covariance_rows)
        quality.write_rows(work/"phase_population_variance1ms.csv", phase_population_rows)
        freeze_json(work/"source_and_readout_costs.json",dict(episodes=costs,summary_elapsed_seconds=time.perf_counter()-start,
            source_and_readout_costs_are_separate=True,unavailable_costs_are_not_zero=True))
        files = {p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in work.iterdir() if p.is_file()}
        report = dict(schema=SCHEMA,status="COMPLETE3000_DESCRIPTIVE_PARAMETER_DOMAIN_COHORT",
            execution=execution,
            complete_development_episodes=EPISODES,parameter_circuits=100,blocks=list(BLOCKS),conditions=list(quality.CONDITIONS),
            circuit_split_counts=SPLITS,core_units=100000,dt_ms=.0125,duration_ticks=400000,
            observation_bin_ticks=80,observation_bin_ms=1.,observation_map=observation_map,
            historical_phase_tables_preserved=True,
            preparation_sha256=sha256_file(preparation/"execution.json"),observation_manifest_sha256=sha256_file(observations),
            bindings=bindings,realized_paired_blocks=pairings,outputs=files,
            independent_unit="joint initial/external history nested in parameter circuit; conditions paired",
            temporal_covariance_scope="within_episode_time",between_block_covariance_scope="matched_time_across3histories",
            phase_covariance_scope="source_boundary_aligned1ms_bins_within_each_of8phases",
            temporal_unit_diagonal_method="binary1ms_second_moment_from_source_refractory_and_complete_phase_counts",
            matched_history_unit_diagonal_status="UNAVAILABLE_WITHOUT_MATCHED1MS_UNIT_COUNTS",
            grid_scope="complete synthetic100parameter circuit grid; not biological population frequencies",
            uncertainty="descriptive mean/sampleSD only; no confidence intervals,pvalues or scaling/source equivalence",
            source_qualification_granted=False,scientific_parameter_domain_qualified=False,scientific_claim_qualified=False,
            uncertainty_coverage_qualified=False,scaling_qualified=False,production_episodes=0,new_source_simulations=0)
        freeze_json(work/"report.json",report)
        shutil.copytree(work,output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report=aggregate(args.prepared,args.observations,args.output)
    print(json.dumps({key:report[key] for key in ("status","complete_development_episodes","parameter_circuits","production_episodes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

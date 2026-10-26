"""Screen count-CDF gaps with empirical first/two-moment envelopes.

Moment values are held at their existing calibration estimates. These envelopes
bound the corresponding empirical distributions, not unknown source moments.
No bootstrap, source draws, confidence interval, or confirmation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, require
from .pfc_delay_analysis import PHASES
from .provenance import freeze_json, sha256_file

ARITHMETIC_TOLERANCE = 1e-11  # serialization check, never the scientific margin


def count_moments(cdf, support, maximum):
    x, support = np.asarray(cdf), np.asarray(support)
    require(x.ndim == 2 and x.shape[1] == len(support) and x.shape[0] > 0
            and support.ndim == 1 and support.dtype.kind in "iu"
            and np.all(support >= 0) and np.all(np.diff(support) > 0), "MOMENT_CDF_AXES")
    require(np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
            and np.all(np.diff(x, axis=1) >= 0) and np.all(x[:, -1] == 1), "MOMENT_COMPLETE_CDF")
    mass = np.diff(np.column_stack((np.zeros(len(x)), x)), axis=1)
    require(np.all(mass[:, support > maximum] == 0), "MOMENT_SOURCE_SUPPORT")
    return mass @ support, mass @ (support.astype(np.float64) ** 2)


def moment_interval(mean, second, threshold, maximum, *, use_second):
    """Distribution bounds given exact moments, with integer tail N>=k+1."""
    require(maximum >= 0 and int(maximum) == maximum and int(threshold) == threshold
            and math.isfinite(mean) and math.isfinite(second)
            and -ARITHMETIC_TOLERANCE <= mean <= maximum + ARITHMETIC_TOLERANCE,
            "MOMENT_RANGE")
    variance = second - mean * mean
    require(variance >= -ARITHMETIC_TOLERANCE and second <= maximum * mean + ARITHMETIC_TOLERANCE,
            "MOMENT_SECOND_RANGE")
    variance = max(variance, 0.)
    if threshold < 0:
        return 0., 0.
    if threshold >= maximum:
        return 1., 1.
    # 1[N>=k+1] <= N/(k+1); 1[N<=k] <= (M-N)/(M-k).
    lo = max(0., 1. - mean / (threshold + 1))
    hi = min(1., (maximum - mean) / (maximum - threshold))
    if use_second:
        # One-sided variance inequalities at the actual integer event boundary.
        if threshold + 1 > mean:
            gap = threshold + 1 - mean
            lo = max(lo, 1. - variance / (variance + gap * gap))
        if threshold < mean:
            gap = mean - threshold
            hi = min(hi, variance / (variance + gap * gap))
    require(lo <= hi + ARITHMETIC_TOLERANCE, "MOMENT_CROSSING_BOUNDS")
    return max(0., lo), min(1., hi)


def difference_interval(left, right):
    return left[0] - right[1], left[1] - right[0]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bound_inputs(root, config):
    files, reports = {}, {}
    for key, ref in config["inputs"].items():
        base = root / ref["path"]
        reports[key] = bound_report(base, ref["report_sha256"])
        files[key] = {}
        for name in ref["files"]:
            path, entry = base / name, reports[key]["outputs"][name]
            require(path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"], "MOMENT_INPUT_BINDING:" + name)
            files[key][name] = path
    return files, reports


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def source_groups(cdf, rates, labels, physical):
    expected_seeds = np.arange(95001, 95033)
    require(np.array_equal(cdf["source_rng_seeds"], expected_seeds)
            and np.array_equal(rates["source_rng_seeds"], expected_seeds), "MOMENT_INDEPENDENT_BLOCKS")
    require(cdf["condition_names"].tolist() == rates["condition_names"].tolist()
            and np.array_equal(cdf["field_indices"], rates["field_indices"])
            and np.array_equal(cdf["field_unit_counts"], rates["field_unit_counts"])
            and int(cdf["field_unit_counts"].sum()) == 100000
            and str(cdf["units"]) == "probability" and str(rates["units"]) == "Hz", "MOMENT_OBSERVATION_MAP")
    require(cdf["phase_names"].tolist() == list(PHASES) and float(rates["bin_ms"]) == 5.
            and np.array_equal(rates["time_start_ms"], np.arange(1000) * 5.), "MOMENT_PHASE_CLOCK")
    support, windows = cdf["count_support"], cdf["window_ms"].tolist()
    groups, maximum_error = {}, 0.
    for label in labels:
        values = cdf[label + "_cdf"]
        require(values.shape == (3,32,5,len(windows),17,len(support)), "MOMENT_CDF_SHAPE")
        for ci, condition in enumerate(cdf["condition_names"].tolist()):
            full_rates = rates[label + "_" + condition]
            require(full_rates.shape == (32,1000,17), "MOMENT_FIELD_RATE_SHAPE")
            for pi, (phase, (start, stop)) in enumerate(PHASES.items()):
                phase_rates = full_rates[:, round(start / 5.):round(stop / 5.)].mean(axis=1)
                for wi, width in enumerate(windows):
                    require((stop - start) % width == 0, "MOMENT_COMPLETE_WINDOWS")
                    for field in range(17):
                        maximum = physical[float(width), field]
                        x = values[ci, :, pi, wi, field]
                        first, second = count_moments(x, support, maximum)
                        expected = phase_rates[:, field] * width / 1000.
                        error = float(np.max(np.abs(first - expected)))
                        maximum_error = max(maximum_error, error)
                        require(error <= ARITHMETIC_TOLERANCE, "MOMENT_FIELD_RATE_IDENTITY")
                        mean, raw_second = float(first.mean()), float(second.mean())
                        key = (label, condition, phase, float(width), field)
                        groups[key] = dict(first=first, second=second, mean=mean, raw_second=raw_second,
                            cdf=x.mean(axis=0), support=support, maximum=maximum,
                            first_moment_block_constant=bool(np.all(first == first[0])),
                            second_moment_block_constant=bool(np.all(second == second[0])))
    return groups, maximum_error


def run(config_path, project_root, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="FAILED_PFC_CDF_MOMENT_ENVELOPE", method_adopted=False,
        uncertainty_coverage_qualified=False, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, production_episodes=0, source_simulations=0,
        new_bootstrap_draws=0, new_independent_blocks=0, population_moment_intervals_computed=False)
    try:
        cfg = yaml.safe_load(config_path.read_text())
        require(cfg["schema"] == "pfc_cdf_moment_envelope_v1" and cfg["planning_blocks"] == 128
                and cfg["method_adopted"] is False and cfg["confirmation_execution_released"] is False, "MOMENT_SCREEN_ONLY")
        files, reports = bound_inputs(project_root, cfg)
        require(reports["prior_uncertainty"]["frozen_margin_sha256"] == cfg["frozen_margin_sha256"], "MOMENT_FROZEN_MARGIN")
        domains = read_csv(files["domain"]["endpoint_domains.csv"])
        planning = read_csv(files["prior_uncertainty"]["calibration_budget_assessment.csv"])
        require(len(domains) == len(planning) == cfg["endpoint_rows"] == 84330, "MOMENT_FULL_DOMAIN")
        physical = {}
        for index, (row, plan) in enumerate(zip(domains, planning)):
            ident = json.loads(row["endpoint_identity"])
            require(row["family"] == plan["family"] and ident == json.loads(plan["endpoint_identity"])
                    and int(plan["domain_row_index"]) == index, "MOMENT_ENDPOINT_ALIGNMENT")
            if row["family"] == "count_cdf":
                key = float(ident["window_ms"]), int(ident["field_index"])
                maximum = int(row["physical_maximum_count"])
                require(physical.get(key, maximum) == maximum, "MOMENT_SHARED_PHYSICAL_SUPPORT")
                physical[key] = maximum
        all_rows, moment_rows, identities, first_arrays, second_arrays, summaries = [], [], [], [], [], {}
        metadata = None
        for realization, source in (("A", "original"), ("B", "repeated")):
            cdf = load_npz(files[source]["coupled_count_cdfs.npz"])
            rates = load_npz(files[source]["coupled_field_rates.npz"])
            axis = {key: cdf[key] for key in ("source_rng_seeds", "condition_names", "phase_names", "window_ms", "field_unit_counts", "field_indices")}
            if metadata is not None:
                require(all(np.array_equal(axis[k], metadata[k]) for k in axis), "MOMENT_A_B_AXIS_IDENTITY")
                require(np.array_equal(cdf[cfg["cohorts"][0]+"_cdf"], coarse), "MOMENT_SHARED_COARSE_CDF")
            else:
                metadata, coarse = axis, cdf[cfg["cohorts"][0]+"_cdf"].copy()
            groups, error = source_groups(cdf, rates, cfg["cohorts"], physical)
            counts, unresolved = Counter(), []
            for key, group in groups.items():
                identity = dict(realization=realization, cohort=key[0], condition=key[1], phase=key[2], window_ms=key[3], field_index=key[4])
                identities.append(json.dumps(identity, sort_keys=True)); first_arrays.append(group["first"]); second_arrays.append(group["second"])
                moment_rows.append(dict(**identity, mean_count=group["mean"], second_raw_count_moment=group["raw_second"],
                    mixture_count_variance=group["raw_second"]-group["mean"]**2,
                    first_moment_block_constant=group["first_moment_block_constant"],
                    second_moment_block_constant=group["second_moment_block_constant"]))
            for index, (row, plan) in enumerate(zip(domains, planning)):
                if row["family"] != "count_cdf":
                    continue
                ident = json.loads(row["endpoint_identity"])
                coords = (ident["condition"],ident["phase"],float(ident["window_ms"]),int(ident["field_index"]))
                left, right = (groups[(ident[side], *coords)] for side in ("left_cohort", "right_cohort"))
                threshold = int(ident["count_threshold"])
                margin = float(plan["margin"])
                structural = threshold >= left["maximum"]
                require((margin == 0) == structural, "MOMENT_STRUCTURAL_MARGIN")
                estimates=[]
                for group in (left,right):
                    ki = int(np.searchsorted(group["support"], threshold, side="right"))-1
                    estimates.append(0. if ki < 0 else float(group["cdf"][ki]))
                empirical_difference=estimates[0]-estimates[1]
                bounds={}
                for name, use_second in (("first",False),("two",True)):
                    single=[moment_interval(g["mean"],g["raw_second"],threshold,g["maximum"],use_second=use_second) for g in (left,right)]
                    lo,hi=difference_interval(*single)
                    require(lo-ARITHMETIC_TOLERANCE <= empirical_difference <= hi+ARITHMETIC_TOLERANCE, "MOMENT_EMPIRICAL_ENCLOSURE")
                    bounds[name]=(lo,hi)
                gap=plan["planning_status"]=="PLANNING_BAND_EXCEEDS_BUDGET"
                first_fits=max(abs(x) for x in bounds["first"])<=margin
                two_fits=max(abs(x) for x in bounds["two"])<=margin
                counts.update(cdf_rows=1, source_identities=int(structural), first_moment_fit=int(first_fits and not structural),
                    two_moment_fit=int(two_fits and not structural), original_planning_gaps=int(gap),
                    original_gaps_first_fit=int(gap and first_fits), original_gaps_two_fit=int(gap and two_fits))
                record=dict(realization=realization,domain_row_index=index,endpoint_identity=row["endpoint_identity"],
                    physical_maximum_count=left["maximum"],margin=margin,original_planning_status=plan["planning_status"],
                    empirical_difference=empirical_difference,first_moment_lower=bounds["first"][0],first_moment_upper=bounds["first"][1],
                    two_moment_lower=bounds["two"][0],two_moment_upper=bounds["two"][1],first_moment_envelope_fits=first_fits,
                    two_moment_envelope_fits=two_fits,left_mean=left["mean"],right_mean=right["mean"],
                    left_second_raw=left["raw_second"],right_second_raw=right["raw_second"],
                    interpretation="EMPIRICAL_MOMENTS_HELD_FIXED_NOT_A_CONFIDENCE_INTERVAL")
                all_rows.append(record)
                if gap and not two_fits:
                    unresolved.append(record)
            require(counts["cdf_rows"]==cfg["cdf_rows"]==48660, "MOMENT_COMPLETE_CDF_DOMAIN")
            summaries[realization]=dict(counts=counts,maximum_count_mean_rate_identity_error=error,
                unresolved_original_gap_rows=[r["domain_row_index"] for r in unresolved],
                first_moment_constant_groups=sum(g["first_moment_block_constant"] for g in groups.values()),
                second_moment_constant_groups=sum(g["second_moment_block_constant"] for g in groups.values()))
        write_csv(output/"cdf_moment_envelopes.csv",all_rows)
        write_csv(output/"primitive_count_moments.csv",moment_rows)
        write_csv(output/"original_gap_moment_envelopes.csv",[r for r in all_rows if r["original_planning_status"]=="PLANNING_BAND_EXCEEDS_BUDGET"])
        np.savez_compressed(output/"primitive_count_moment_blocks.npz",identities=np.array(identities),
            first=np.stack(first_arrays),second_raw=np.stack(second_arrays),source_rng_seeds=metadata["source_rng_seeds"])
        report.update(status="COMPLETE_PFC_CDF_MOMENT_ENVELOPE_SCREEN",settings=cfg,settings_sha256=sha256_file(config_path),
            summaries=summaries,original_independent_blocks=32,auxiliary_realizations=2,retained_domain_rows=84330,
            source_moment_groups=len(moment_rows),cdf_realization_rows=len(all_rows),
            interpretation="Optimistic screen only: population moments are not known or bounded. Even a fitted envelope does not establish CDF coverage. Persistent failures here show this particular point-moment envelope is insufficient; no all-method impossibility follows.",
            implementation_sha256=sha256_file(Path(__file__)),
            outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file()})
    except Exception as exc:
        report["error"]=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report["wall_seconds"]=time.perf_counter()-started
        freeze_json(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run(args.config,args.project_root,args.output)
    print(json.dumps(dict(status=report["status"],summaries=report["summaries"])))


if __name__=="__main__":
    main()

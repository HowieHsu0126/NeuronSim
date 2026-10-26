"""Recompute LFADS budget differences from the six archived score tables.

Uses standard-library arithmetic, not the aggregation implementation. Previously
qualified count generation is referenced, not repeated. Selection-block results
remain descriptive and this check grants no scientific fidelity qualification.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from .provenance import freeze_json, sha256_file

BUDGETS = ("original_20", "continued_80_ceiling")
SEEDS = (41, 42, 43)
IDENTITY = ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")
METRICS = ("energy_score", "variogram_spatial", "variogram_cross_population",
           "variogram_temporal_lag_1", "variogram_temporal_lag_4", "variogram_temporal_lag_16",
           "unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
           "unit_total_coverage_90", "unit_total_interval_width_90",
           "refractory_count_violation_1bin", "refractory_count_violation_2bin", "refractory_count_violation_4bin")


def number(value):
    if value == "": return None
    result = float(value)
    if not math.isfinite(result): raise ValueError("nonfinite metric")
    return result


def equal(actual, expected):
    value = number(actual)
    if expected is None:
        if value is not None: raise ValueError("undefined metric became defined")
    elif value is None or not math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"aggregation disagreement: {value} != {expected}")


def replay(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", source_archive=str(root), score_rows_checked=0,
                  pair_rows_checked=0, block_rows_checked=0, summary_rows_checked=0,
                  seed_rows_checked=0, consumed_files=[], scientific_claim_qualified=False,
                  implementation_sha256=sha256_file(Path(__file__)))
    try:
        delivery_path=root/"delivery.json"
        delivery=json.loads(delivery_path.read_text())
        report["delivery_sha256"]=sha256_file(delivery_path)
        if delivery["status"]!="ARCHIVED_REPLAYED_BUDGET_COMPARISON": raise ValueError("complete archive required")
        declared={r["path"]:r for r in delivery["files"]}
        if len(declared)!=len(delivery["files"]): raise ValueError("duplicate archive path")
        def read(name):
            path=Path(name)
            if path.is_absolute() or ".." in path.parts: raise ValueError("unsafe source path")
            entry=declared[name]
            if sha256_file(root/path)!=entry["sha256"]: raise ValueError("archive changed: "+name)
            report["consumed_files"].append(dict(path=name,sha256=entry["sha256"]))
            return (root/path).read_text()
        def table(name): return list(csv.DictReader(read(name).splitlines()))
        analysis=json.loads(read("analysis/analysis.json"))
        if (analysis["status"]!="DESCRIBED_REPLAYED_BUDGET_COMPARISON" or analysis["training_seeds"]!=list(SEEDS)
                or analysis["score_rows"]!=864 or analysis["forecast_origins"]!=288
                or analysis["scientific_claim_qualified"] or analysis["formal_test_accessed"]):
            raise ValueError("analysis scope changed")
        raw={}
        for budget in BUDGETS:
            for seed in SEEDS:
                prefix=f"forecasts/{budget}/seed_{seed}"
                receipt=json.loads(read(prefix+"_replay.json"))
                ev=json.loads(read(prefix+"_forecast/evaluation.json"))
                name=prefix+"_forecast/scores.csv"
                rows=table(name)
                if (receipt["status"]!="PASS" or receipt["verified_origins"]!=48 or len(rows)!=144
                        or receipt["scores_sha256"]!=declared[name]["sha256"]
                        or receipt["evaluation_sha256"]!=declared[prefix+"_forecast/evaluation.json"]["sha256"]):
                    raise ValueError("incomplete bound source score table")
                for row in rows:
                    if row["seed"]!=str(seed) or row["method"]!="lfads": raise ValueError("score identity changed")
                    key=(budget,*[row[k] for k in IDENTITY])
                    if key in raw: raise ValueError("duplicate source score")
                    raw[key]=row
        combined=table("analysis/forecast_all_seeds.csv")
        seen=set()
        for row in combined:
            key=(row["budget"],*[row[k] for k in IDENTITY])
            if key in seen: raise ValueError("duplicate aggregate score")
            seen.add(key)
            for metric in METRICS: equal(row[metric],number(raw[key][metric]))
            report["score_rows_checked"]+=1
        if seen!=set(raw): raise ValueError("aggregate score denominator changed")
        pairs={}; block_values=defaultdict(list); seed_values=defaultdict(list)
        for key,original in raw.items():
            if key[0]!=BUDGETS[0]: continue
            continued=raw[(BUDGETS[1],*key[1:])]
            for metric in METRICS:
                a,b=number(original[metric]),number(continued[metric])
                if (a is None)!=(b is None): raise ValueError("unmatched metric support")
                delta=None if a is None else b-a
                identity=(*key[1:],metric);pairs[identity]=(a,b,delta)
                block_values[(key[1],key[2],key[6],metric)].append((a,b,delta))
                if delta is not None: seed_values[(key[1],key[6],metric)].append(delta)
        seen=set()
        for row in table("analysis/paired_differences_by_case.csv"):
            key=(*[row[k] for k in IDENTITY],row["metric"])
            if key in seen: raise ValueError("duplicate paired case")
            seen.add(key)
            for name,value in zip(("original","continued","continued_minus_original"),pairs[key]): equal(row[name],value)
            report["pair_rows_checked"]+=1
        if seen!=set(pairs): raise ValueError("paired case denominator changed")
        seen=set()
        for row in table("analysis/paired_differences_by_block.csv"):
            key=tuple(row[k] for k in ("seed","block","horizon_ms","metric"));values=block_values[key]
            if key in seen: raise ValueError("duplicate paired block")
            seen.add(key);defined=[v for v in values if v[0] is not None]
            if int(row["registered_matched_cases"])!=len(values) or int(row["defined_matched_cases"])!=len(defined):
                raise ValueError("block denominator changed")
            for i,name in enumerate(("original","continued","continued_minus_original")):
                equal(row[name],fmean(v[i] for v in defined) if defined else None)
            report["block_rows_checked"]+=1
        if seen!=set(block_values): raise ValueError("block coverage changed")
        expected_summaries={(key[2],key[3]) for key in block_values};seen=set()
        for row in table("analysis/paired_summary.csv"):
            key=(row["horizon_ms"],row["metric"])
            if key in seen: raise ValueError("duplicate summary")
            seen.add(key)
            values=[fmean(seed_values[(str(seed),*key)]) for seed in SEEDS if seed_values[(str(seed),*key)]]
            if values and len(values)!=3: raise ValueError("seed denominator changed")
            for name,value in zip(("mean_continued_minus_original","minimum_seed_mean","maximum_seed_mean"),
                                  (fmean(values),min(values),max(values)) if values else (None,None,None)):
                equal(row[name],value)
            if row["training_seeds"]!="3" or row["epoch_selection_blocks"]!="4" or row["inferential_intervals"]!="False":
                raise ValueError("summary inference scope changed")
            report["summary_rows_checked"]+=1
        if seen!=expected_summaries: raise ValueError("summary coverage changed")
        expected_seed={(k[0],k[1],k[6]) for k in raw};seen=set()
        for row in table("analysis/forecast_by_seed.csv"):
            key=(row["budget"],row["seed"],row["horizon_ms"])
            if key in seen: raise ValueError("duplicate seed summary")
            seen.add(key)
            members=[v for k,v in raw.items() if (k[0],k[1],k[6])==key]
            if len(members)!=48 or row["registered_case_origins"]!="48":raise ValueError("seed origin denominator changed")
            for metric in METRICS:
                values=[number(v[metric]) for v in members if v[metric]!=""]
                equal(row[metric],fmean(values) if values else None)
            report["seed_rows_checked"]+=1
        if seen!=expected_seed: raise ValueError("seed summary coverage changed")
        report["status"]="PASS_INDEPENDENT_BUDGET_AGGREGATION_REPLAY"
    except Exception as error:
        report["error"]=dict(type=type(error).__name__,message=str(error))
    freeze_json(output/"replay.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=replay(args.archive,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!="consumed_files"},indent=2))
    return 0 if report["status"].startswith("PASS_") else 1

if __name__=="__main__":sys.exit(main())

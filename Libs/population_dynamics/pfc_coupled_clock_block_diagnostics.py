"""Describe block heterogeneity in the completed Jpos2.3 clock bank.

Consumes existing readouts and summaries only. Exact zero is a numerical value,
not a biological response threshold; no interval, test or qualification is made.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics

CLOCKS = ("numpy_dt_0p05", "numpy_dt_0p025", "numpy_dt_0p0125")
PAIRS = ((CLOCKS[0], CLOCKS[1]), (CLOCKS[1], CLOCKS[2]), (CLOCKS[0], CLOCKS[2]))
PHASES = ("baseline", "cue", "early_delay", "middle_delay", "late_delay")
METRICS = ("all_unit_rate_hz", "excitatory_rate_hz", "inhibitory_rate_hz",
           "cue_region_rate_hz", "noncue_region_rate_hz", "cue_selectivity_hz",
           "vector_x_hz", "vector_y_hz")
CONTRASTS = ("cue_minus_omission", "strong_cue_minus_omission", "strong_cue_minus_cue")
STARTS = {"reference_2560": 93001, "expanded_100k": 95001}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Archive:
    def __init__(self, path, bindings, expected=None):
        self.path = Path(path)
        self.bindings = bindings
        data = (self.path / "delivery.json").read_bytes()
        self.sha256 = digest(data)
        require(expected is None or self.sha256 == expected, "upstream delivery changed")
        self.delivery = json.loads(data)
        require(self.delivery.get("exit_code", 0) == 0 and
                self.delivery.get("verified", True), "unsuccessful delivery")
        self.files = {r["path"]: r for r in self.delivery["files"]}
        require(len(self.files) == len(self.delivery["files"]), "duplicate delivered path")
        bindings[str(self.path / "delivery.json")] = self.sha256

    def read(self, name):
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe input path")
        data = (self.path / name).read_bytes()
        entry = self.files[name]
        require(len(data) == entry["bytes"] and digest(data) == entry["sha256"], "input changed: " + name)
        self.bindings[str(self.path / name)] = entry["sha256"]
        return data

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8"))))


def pivot_block(rows, cohort, seed):
    expected = set(itertools.product(PAIRS, CONTRASTS, PHASES, METRICS))
    seen, values = set(), defaultdict(dict)
    for r in rows:
        pair = (r["left_clock"], r["right_clock"])
        key = (pair, r["contrast"], r["phase"], r["metric"])
        require(key in expected and key not in seen, "missing or duplicate endpoint domain")
        require(r["cohort"] == cohort and int(r["source_rng_seed"]) == seed and
                r["units"] == "Hz" and r["evidence_kind"] == "archived_source_execution",
                "block identity or units changed")
        seen.add(key)
        a, b, delta = (float(r[k]) for k in ("left_value", "right_value", "left_minus_right"))
        require(all(math.isfinite(x) for x in (a, b, delta)) and
                math.isclose(a-b, delta, rel_tol=1e-10, abs_tol=1e-9), "invalid signed response")
        v = values[key[1:]]
        for clock, value in zip(pair, (a, b)):
            require(clock not in v or v[clock] == value, "clock value differs between pair records")
            v[clock] = value
    require(seen == expected, "incomplete block response domain")
    return [dict(cohort=cohort, source_rng_seed=seed,
                 partition="pilot3" if seed < STARTS[cohort]+3 else "remaining29",
                 contrast=k[0], phase=k[1], metric=k[2], units="Hz", **v)
            for k, v in sorted(values.items())]


def moments(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values), "invalid block sample")
    return dict(mean=statistics.mean(values), block_sd=statistics.stdev(values),
                minimum=min(values), maximum=max(values))


def response_diagnostics(rows):
    groups = defaultdict(list)
    for row in rows:
        base = (row["cohort"], row["contrast"], row["phase"], row["metric"])
        groups[(row["partition"], *base)].append(row)
        groups[("full32_descriptive", *base)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        require(len(group) == {"pilot3": 3, "remaining29": 29, "full32_descriptive": 32}[key[0]],
                "incorrect partition denominator")
        for left, right in PAIRS:
            changes = [r[left]-r[right] for r in group]
            maximum = max(abs(x) for x in changes)
            result.append(dict(zip(("partition", "cohort", "contrast", "phase", "metric"), key),
                left_clock=left, right_clock=right, blocks=len(group),
                exact_zero_left=sum(r[left] == 0 for r in group),
                exact_zero_right=sum(r[right] == 0 for r in group),
                exact_zero_membership_changed=sum((r[left] == 0) != (r[right] == 0) for r in group),
                opposite_signed_values=sum(r[left]*r[right] < 0 for r in group),
                mean_signed_change=statistics.mean(changes),
                mean_absolute_change=statistics.mean(abs(x) for x in changes),
                maximum_absolute_change=maximum,
                maximum_change_source_rng_seeds=";".join(str(r["source_rng_seed"]) for r,d in zip(group,changes) if abs(d)==maximum),
                **{"change_"+k:v for k,v in moments(changes).items()}))
    return result


def summarize_units(rows):
    groups = defaultdict(list)
    keys = ("cohort", "left_clock", "right_clock", "condition", "phase", "population")
    metrics = ("phase_count_changed_fraction", "mean_signed_phase_count", "mean_absolute_phase_count",
               "mean_unit_bin_absolute_count", "root_mean_square_unit_bin_count")
    for r in rows:
        for partition in (r["partition"], "full32_descriptive"):
            groups[(partition, *(r[k] for k in keys))].append(r)
    result=[]
    for key, group in sorted(groups.items()):
        require(len(group)=={"pilot3":3,"remaining29":29,"full32_descriptive":32}[key[0]] and
                len({r["source_rng_seed"] for r in group})==len(group), "unit block denominator")
        out=dict(zip(("partition", *keys),key),blocks=len(group))
        for metric in metrics:
            out.update({metric+"_"+k:v for k,v in moments([float(r[metric]) for r in group]).items()})
        result.append(out)
    return result


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(summary, readout, pilot_summary, pilot_readout, output):
    output=Path(output);require(not output.exists(), "output must be new")
    bindings={}
    s=Archive(summary,bindings);sr=s.json("summary/report.json")
    require(sr["status"]=="COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_SUMMARY" and
            sr["complete_source_episodes"]==576 and sr["original_independent_blocks_per_size"]==32,
            "complete 576-episode summary required")
    require(all(sr[k] is False for k in ("confidence_intervals","p_values","scientific_fidelity_qualified",
                "confirmation_execution_released","cross_size_pairing")), "descriptive evidence required")
    ps=Archive(pilot_summary,bindings,sr["reused_pilot_summary"]["delivery_sha256"])
    pr=ps.json("summary/report.json")
    readers=[(Archive(pilot_readout,bindings,pr["source_delivery_sha256"]),range(0,3)),
             (Archive(readout,bindings,sr["source_delivery_sha256"]),range(3,32))]
    all_rows=[]
    for reader, offsets in readers:
        report=reader.json("analysis/report.json")
        require(report["complete_source_episodes_consumed"] == 18*len(offsets), "source denominator")
        for cohort,start in STARTS.items():
            for offset in offsets:
                seed=start+offset
                all_rows.extend(pivot_block(reader.csv(f"analysis/{cohort}/{seed}/paired_response_comparison.csv"),cohort,seed))
    response=response_diagnostics(all_rows)
    units=summarize_units(s.csv("summary/unit_differences_by_block.csv"))
    # Check the independently joined block means against the accepted pooled summary.
    accepted=s.csv("summary/paired_response_comparison_summary.csv")
    index={(r["partition"],r["cohort"],r["contrast"],r["phase"],r["metric"],r["left_clock"],r["right_clock"]):r for r in accepted}
    for r in response:
        key=tuple(r[k] for k in ("partition","cohort","contrast","phase","metric","left_clock","right_clock"))
        a=index[key]
        require(math.isclose(float(a["left_minus_right_mean"]),r["mean_signed_change"],rel_tol=1e-10,abs_tol=1e-9), "joined and pooled means differ")
    require(len(index)==len(response), "pooled endpoint denominator differs")
    output.mkdir(parents=True)
    for name,rows in (("response_by_block.csv",all_rows),("response_clock_diagnostics.csv",response),("unit_clock_diagnostics.csv",units)):
        write_csv(output/name,rows)
    report=dict(status="COMPLETE_DESCRIPTIVE_BLOCK_CLOCK_DIAGNOSTICS",job_id=os.environ.get("SLURM_JOB_ID"),
        finished_utc=datetime.now(timezone.utc).isoformat(),implementation_sha256=digest(Path(__file__).read_bytes()),
        input_sha256=bindings,complete_source_episodes_reused=576,independent_blocks_per_size=32,
        response_block_rows=len(all_rows),response_diagnostic_rows=len(response),unit_diagnostic_rows=len(units),
        accepted_summary_mean_checks=len(response),source_simulations=0,raw_source_arrays_read=0,
        confidence_intervals=False,p_values=False,scientific_fidelity_qualified=False,
        confirmation_execution_released=False,production_episodes=0,
        interpretation="All registered blocks, controls, phases and scalar metrics retained. Exact-zero membership is a numerical descriptor, not activity prevalence or memory success. Block SD and changes are descriptive. Coupled binomial marginal laws differ across clocks; no pure integration-error or convergence claim.",
        outputs={name:dict(bytes=(output/name).stat().st_size,sha256=digest((output/name).read_bytes())) for name in
            ("response_by_block.csv","response_clock_diagnostics.csv","unit_clock_diagnostics.csv")})
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def cdf_block_maxima(rows):
    keys=("cohort","source_rng_seed","left_clock","right_clock","condition","phase")
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,group in sorted(groups.items()):
        support={(r["window_ms"],r["field_index"]) for r in group}
        require(len(group)==len(support)==51,"complete 3-window/17-field CDF support required")
        values=[float(r["maximum_absolute_cdf_difference"]) for r in group]
        require(all(math.isfinite(v) and 0<=v<=1 for v in values),"invalid CDF difference")
        maximum=max(values);winners=[r for r,v in zip(group,values) if v==maximum]
        require(len({r["partition"] for r in group})==1,"block partition differs")
        result.append(dict(zip(keys,key),partition=group[0]["partition"],maximum_absolute_cdf_difference=maximum,
            maximizing_window_field=";".join(r["window_ms"]+":"+r["field_index"] for r in winners),
            maximizing_saved_count_threshold=";".join(r["first_maximizing_count_threshold"] for r in winners)))
    require(len(result)==64*3*3*5,"all64 blocks/3clock pairs/3conditions/5phases required")
    return result


def cdf_only(summary,output):
    output=Path(output);require(not output.exists(),"output must be new")
    bindings={};archive=Archive(summary,bindings);source=archive.json("summary/report.json")
    require(source["complete_source_episodes"]==576 and source["cdf_supremum_rows"]==146880,
            "complete registered CDF summary required")
    blocks=cdf_block_maxima(archive.csv("summary/count_cdf_suprema_by_block.csv"))
    groups=defaultdict(list)
    keys=("cohort","left_clock","right_clock","condition","phase")
    for r in blocks:
        for partition in (r["partition"],"full32_descriptive"):
            groups[(partition,*(r[k] for k in keys))].append(r)
    summaries=[]
    for key,group in sorted(groups.items()):
        require(len(group)=={"pilot3":3,"remaining29":29,"full32_descriptive":32}[key[0]] and
                len({r["source_rng_seed"] for r in group})==len(group),"CDF block denominator differs")
        values=[r["maximum_absolute_cdf_difference"] for r in group];maximum=max(values)
        summaries.append(dict(zip(("partition",*keys),key),blocks=len(group),**moments(values),
            maximum_source_rng_seeds=";".join(r["source_rng_seed"] for r,v in zip(group,values) if v==maximum)))
    output.mkdir(parents=True)
    write_csv(output/"cdf_maxima_by_block.csv",blocks);write_csv(output/"cdf_maxima_summary.csv",summaries)
    report=dict(status="COMPLETE_DESCRIPTIVE_CDF_CLOCK_DIAGNOSTICS",source_simulations=0,
        raw_source_arrays_read=0,source_cdf_rows=146880,block_maxima_rows=len(blocks),summary_rows=len(summaries),
        complete_source_episodes_reused=576,independent_blocks_per_size=32,
        input_sha256=bindings,implementation_sha256=digest(Path(__file__).read_bytes()),
        confidence_intervals=False,p_values=False,scientific_fidelity_qualified=False,confirmation_execution_released=False,
        production_episodes=0,interpretation="Per-block maximum over all saved windows/fields/count support, followed by descriptive block moments. These are empirical distribution differences, not tests or numerical error bounds. All controls and partitions retained.")
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("summary","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("readout","pilot-summary","pilot-readout"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--cdf-only",action="store_true")
    args=parser.parse_args()
    if args.cdf_only:
        report=cdf_only(args.summary,args.output)
    else:
        require(all((args.readout,args.pilot_summary,args.pilot_readout)),"complete readout paths required")
        report=run(args.summary,args.readout,args.pilot_summary,args.pilot_readout,args.output)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)}))


if __name__=="__main__":
    main()

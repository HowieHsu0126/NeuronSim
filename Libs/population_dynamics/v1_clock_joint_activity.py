"""Compare all-unit temporal statistics in already accepted V1 clock pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from . import benchmark_activity_statistics as activity
from . import v1_clock_comparison as clock
from .provenance import freeze_json, sha256_file

WINDOWS_MS = (("full", 0, 3000), ("pre_command", 0, 501), ("command_response", 501, 3000))
LAGS_MS = (0, 1, 10, 100)


def physical_windows(state: dict, events: dict, populations: np.ndarray, dt_ms: float):
    """Apply the established diagnostic windows at either native clock."""
    if dt_ms not in (clock.REFERENCE_DT_MS, clock.FINE_DT_MS):
        raise ValueError("JOINT_CLOCK_NATIVE_RESOLUTION_NOT_SUPPORTED")
    ticks = np.rint(events["time_ms"] / dt_ms).astype(np.int64)
    if not np.array_equal(ticks*dt_ms, events["time_ms"]):
        raise ValueError("JOINT_CLOCK_EVENT_NOT_ON_NATIVE_CLOCK")
    if len(populations) != len(state["source_id"]):
        raise ValueError("JOINT_CLOCK_COMPLETE_UNIT_MAP_REQUIRED")
    result = {}
    for name, start, stop in WINDOWS_MS:
        summary, arrays = activity.event_statistics(ticks, events["unit_index"], populations,
            duration_ticks=int(3000/dt_ms), start_tick=int(start/dt_ms), stop_tick=int(stop/dt_ms),
            bin_ticks=int(1/dt_ms), lag_bins=LAGS_MS)
        result[name] = (summary, arrays)
    return result


def accepted_comparison(case: dict, job: dict) -> tuple[dict, str]:
    if case["identity"] != job["identity"]:
        raise ValueError("JOINT_CLOCK_CASE_IDENTITY_DIFFERS")
    directory = Path(job["comparison_output"])
    delivery = json.loads((directory/"delivery.json").read_text())
    report_path = directory/"analysis/report.json"
    digest = sha256_file(report_path)
    if (delivery.get("exit_code") != 0 or delivery.get("report_sha256") != digest
            or str(delivery.get("job_id")) != str(job["comparison_job"])
            or delivery.get("reference_archive") != case["reference"]
            or delivery.get("fine_archive") != job["native_archive"]):
        raise ValueError("JOINT_CLOCK_ACCEPTED_COMPARISON_REQUIRED")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "V1_CLOCK_COMPARISON_COMPLETE_DEVELOPMENT" or report.get("mode") != "full"
            or report.get("source_bindings_and_requested_inputs_exact") is not True
            or report.get("physical_parameters_and_initial_states_exact") is not True
            or report.get("reference_resolution_ms") != .25 or report.get("fine_resolution_ms") != .125
            or report["all_units"]["units"] != 230924
            or report["inputs"]["reference"]["consumed"]["delivery.json"]["sha256"] != case["delivery_sha256"]
            or str(report["inputs"]["fine"]["job_id"]) != str(job["native_job"])):
        raise ValueError("JOINT_CLOCK_VALIDATED_FULL_PAIR_REQUIRED")
    return report, digest


def analyze_pair(case: dict, job: dict, output: Path) -> dict:
    comparison, comparison_sha = accepted_comparison(case, job)
    implementation = {name: sha256_file(path) for name, path in (
        ("adapter", Path(__file__)), ("activity_statistics", Path(activity.__file__)),
        ("clock_alignment", Path(clock.__file__)))}
    if output.exists():
        report = json.loads((output/"report.json").read_text())
        if (report.get("status") != "COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PAIR"
                or report.get("identity") != case["identity"] or report.get("comparison_sha256") != comparison_sha
                or report.get("implementation") != implementation):
            raise ValueError("JOINT_CLOCK_COMPLETED_REUSE_BINDING_DIFFERS")
        for name, record in report["artifacts"].items():
            path = output/name
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError("JOINT_CLOCK_REUSED_ARTIFACT_CHANGED")
        return report
    output.mkdir(parents=True)
    result = dict(status="FAILED_V1_CLOCK_JOINT_ACTIVITY_PAIR", identity=case["identity"],
        comparison_sha256=comparison_sha, implementation=implementation, scientific_qualification=False,
        production_episodes=0, new_source_simulations=0, diagnostic_bin_ms=1.0, lag_ms=list(LAGS_MS),
        scientific_bins_frozen=False, windows=[], artifacts={},
        interpretation="Within-episode temporal descriptors. They do not establish independent trials, noise covariance, stationarity, a full joint law, equivalence or convergence.")
    try:
        with tempfile.TemporaryDirectory(prefix="neuronsim-joint-clock-inputs-") as temporary:
            work = Path(temporary)
            states, events, populations, descriptions = {}, {}, {}, {}
            for label, archive, dt in (("reference", Path(case["reference"]), .25), ("fine", Path(job["native_archive"]), .125)):
                consumed = comparison["inputs"][label]["consumed"]
                if sha256_file(archive/"delivery.json") != consumed["delivery.json"]["sha256"]:
                    raise ValueError("JOINT_CLOCK_NATIVE_DELIVERY_CHANGED")
                for name in ("primary/native/cortical_state.npz", "primary/native/cortical_spikes.npz", clock.SOURCE_TYPE_CSV):
                    target=work/label/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(archive/name,target)
                    if target.stat().st_size != consumed[name]["bytes"] or sha256_file(target) != consumed[name]["sha256"]:
                        raise ValueError("JOINT_CLOCK_NATIVE_ARRAY_CHANGED:" + name)
                state=clock.align_state(clock.load_arrays(work/label/"primary/native/cortical_state.npz"))
                event=clock.align_spikes(clock.load_arrays(work/label/"primary/native/cortical_spikes.npz"),state,dt)
                names,codes=clock.source_populations(work/label/clock.SOURCE_TYPE_CSV,state["node_type_id"])
                if len(state["source_id"]) != 230924 or len(names) != 21:
                    raise ValueError("JOINT_CLOCK_ALL_UNITS_AND_POPULATIONS_REQUIRED")
                states[label],events[label],populations[label]=state,event,names[codes]
                descriptions[label]=physical_windows(state,event,names[codes],dt)
            for key in ("source_id", "node_type_id"):
                if not np.array_equal(states["reference"][key],states["fine"][key]):
                    raise ValueError("JOINT_CLOCK_UNIT_IDENTITY_DIFFERS")
            if not np.array_equal(populations["reference"],populations["fine"]):
                raise ValueError("JOINT_CLOCK_POPULATION_IDENTITY_DIFFERS")
            for name, start, stop in WINDOWS_MS:
                reference,a=descriptions["reference"][name];fine,b=descriptions["fine"][name]
                for key in ("population_index", "population_names", "population_unit_counts", "lag_bins"):
                    if not np.array_equal(a[key],b[key]):raise ValueError("JOINT_CLOCK_WINDOW_INDICES_DIFFER")
                arrays=dict(source_id=states["reference"]["source_id"],population=populations["reference"],
                    population_names=a["population_names"],population_index=a["population_index"],
                    population_unit_counts=a["population_unit_counts"],lag_ms=np.asarray(LAGS_MS))
                for label, values in (("reference",a),("fine",b)):
                    arrays.update({label+"_"+key:values[key] for key in ("unit_totals","unit_mean_count",
                        "unit_count_autocovariance","population_counts","total_counts","population_count_covariance")})
                path=output/(name+".npz");np.savez_compressed(path,**arrays)
                result["artifacts"][path.name]=dict(bytes=path.stat().st_size,sha256=sha256_file(path))
                delta=b["unit_count_autocovariance"]-a["unit_count_autocovariance"]
                pop_delta=b["population_count_covariance"]-a["population_count_covariance"]
                result["windows"].append(dict(window=name,start_ms=start,stop_ms_exclusive=stop,
                    reference=reference,fine=fine,
                    unit_autocovariance_difference=[dict(lag_ms=lag,mean=float(delta[i].mean()),
                        mean_absolute=float(np.abs(delta[i]).mean()),maximum_absolute=float(np.abs(delta[i]).max())) for i,lag in enumerate(LAGS_MS)],
                    population_covariance_difference=[dict(lag_ms=lag,maximum_absolute=float(np.abs(pop_delta[i]).max())) for i,lag in enumerate(LAGS_MS)]))
            result["terminal_events"]={label:int(np.count_nonzero(value["time_ms"]==3000.)) for label,value in events.items()}
            result.update(status="COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PAIR",units=230924,populations=21)
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        freeze_json(output/"report.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",type=Path,required=True);parser.add_argument("--submissions",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--episode-id")
    args=parser.parse_args(argv)
    plan=json.loads(args.cases.read_text());submitted=json.loads(args.submissions.read_text())
    cases={row["identity"]["episode_id"]:row for row in plan["cases"]}
    jobs={row["identity"]["episode_id"]:row for row in submitted["cases"]}
    if (not cases or len(cases)!=len(plan["cases"]) or len(jobs)!=len(submitted["cases"])
            or cases.keys()!=jobs.keys() or plan["full_units"]!=230924):
        raise ValueError("JOINT_CLOCK_PANEL_IDENTITIES_DIFFER")
    selected=[args.episode_id] if args.episode_id else list(cases)
    reports=[]
    for key in selected:
        report=analyze_pair(cases[key],jobs[key],args.output/"cases"/key)
        reports.append(dict(episode_id=key,status=report["status"],path=str(Path("cases")/key/"report.json")))
        print(json.dumps(dict(episode_id=key,status=report["status"])),flush=True)
    if args.episode_id is None:
        freeze_json(args.output/"report.json",dict(status="COMPLETE_V1_CLOCK_JOINT_ACTIVITY_PANEL",pairs=len(reports),
            planned_pairs=len(cases),reports=reports,diagnostic_bin_ms=1.0,lag_ms=list(LAGS_MS),
            cases_sha256=sha256_file(args.cases),submissions_sha256=sha256_file(args.submissions),
            scientific_qualification=False,production_episodes=0,new_source_simulations=0))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

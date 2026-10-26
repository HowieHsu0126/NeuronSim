"""Complete-path per-unit factorial-moment compensator for the native PFC law.

The readout integrates the next fresh input at each observed past. It does not
integrate all possible histories, certify the finite PRNG, or imply variance
reduction. Source trajectories and their RNG stream remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as executor
from . import pfc_native_drive as native
from .pfc_conditional_spike_probability import binomial_tail_bounds
from .pfc_delay_analysis import PHASES
from .pfc_fluctuation_analysis import unit_mapping
from .pfc_threshold_branch_probe import STATE_FIELDS, E_FIELDS
from .provenance import freeze_json, sha256_file

ANCHORS = dict(pre_rk2='_rk2(e, constants["E"], t, dt, arithmetic, "E")',
               pre_threshold='spikes = {}',
               pre_input='i["s_AMPA"][:] += drive["I"][tick % chunk_rows]')


def require(value, message):
    if not value:
        raise ValueError("PFC_MULTISPIKE_" + message)


def observer_anchors():
    lines, start = inspect.getsourcelines(executor.simulate)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    result = {}
    for name, statement in ANCHORS.items():
        target = ast.dump(ast.parse(statement).body[0], include_attributes=False)
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)
                   and ast.dump(n, include_attributes=False) == target]
        require(len(matches) == 1, "SOURCE_ANCHOR:" + name)
        result[name] = dict(line=start + matches[0].lineno - 1, statement=statement)
    require(result["pre_rk2"]["line"] < result["pre_threshold"]["line"] < result["pre_input"]["line"], "SOURCE_SCHEDULE")
    return result


def truncated_law(n, probability, window_ticks, maximum, digits, allowance):
    numerator, denominator = float(probability).as_integer_ratio()
    tails = binomial_tail_bounds(n, numerator, denominator, digits)
    scale = 10**digits
    low = [max(0, int(tails[k]["lower_scaled"]) - int(tails[k+1]["upper_scaled"])) for k in range(n+1)]
    high = [int(tails[k]["upper_scaled"]) - int(tails[k+1]["lower_scaled"]) for k in range(n+1)]
    require(allowance["numerator"] > 0 and allowance["denominator"] > 0, "NUMERIC_ALLOWANCE")
    budget = scale * allowance["numerator"] // allowance["denominator"]
    accumulated_width = 0
    for cutoff in range(n+1):
        accumulated_width += high[cutoff] - low[cutoff]
        width = 2 * maximum * window_ticks * (accumulated_width + int(tails[cutoff+1]["upper_scaled"]))
        if width <= budget:
            break
    else:
        raise ValueError("PFC_MULTISPIKE_DECIMAL_PRECISION_INSUFFICIENT")
    return dict(binomial_n=n, probability=float(probability), probability_numerator=numerator,
        probability_denominator=denominator, decimal_scale_digits=digits, scale=str(scale), cutoff=cutoff,
        mass_lower_scaled=[str(v) for v in low[:cutoff+1]], mass_upper_scaled=[str(v) for v in high[:cutoff+1]],
        omitted_tail_lower_scaled=tails[cutoff+1]["lower_scaled"], omitted_tail_upper_scaled=tails[cutoff+1]["upper_scaled"],
        worst_case_per_unit_window_factorial_width_scaled=str(width), numeric_allowance=allowance,
        omitted_mass_renormalized=False, monotonicity_assumed=False)


def integrate_histogram(success_weights, total_weight, law):
    values = np.asarray(success_weights)
    require(values.ndim == 1 and len(values) == law["cutoff"] + 1 and values.dtype.kind in "iu"
            and np.all(values >= 0) and np.all(values <= total_weight), "WEIGHTED_RESPONSE_HISTOGRAM")
    scale = int(law["scale"])
    lower = sum(int(w)*int(p) for w,p in zip(values,law["mass_lower_scaled"]))
    upper = sum(int(w)*int(p) for w,p in zip(values,law["mass_upper_scaled"]))
    omitted = int(total_weight)*int(law["omitted_tail_upper_scaled"])
    upper = min(int(total_weight)*scale, upper + omitted)
    require(0 <= lower <= upper, "CONDITIONAL_SUM_BOUNDS")
    return 2*lower, 2*upper, 2*omitted


def candidate_step(state, base, counts, constants, t, dt):
    """Execute unchanged RK2 for every supplied count, without a monotone shortcut."""
    counts = np.asarray(counts, dtype=np.float64)
    grid = counts.ndim == 2
    require(counts.ndim in (1,2), "CANDIDATE_COUNT_AXES")
    shadow = {k: (np.asarray(v)[:,None].copy() if grid else np.asarray(v).copy()) for k,v in state.items()}
    shadow["s_AMPA"] = np.asarray(base)[:,None] + counts if grid else np.asarray(base) + counts
    executor._rk2(shadow, constants, t, dt, executor._Arithmetic("numpy"), "E")
    require(np.isfinite(shadow["v"]).all(), "NONFINITE_COUNTERFACTUAL_VOLTAGE")
    decisions = (shadow["v"] > constants["threshold"]) & shadow["not_refractory"]
    return decisions, shadow


class MultispikeObserver:
    def __init__(self, config, settings, *, phases=None):
        self.config, self.settings = config, settings
        self.phases = PHASES if phases is None else phases
        self.dt = config["dt_ms"]*1e-3
        self.ticks = round(config["duration_ms"]/config["dt_ms"])
        self.window_ticks = round(settings["window_ms"]/config["dt_ms"])
        require(self.window_ticks > 0 and self.window_ticks*config["dt_ms"] == settings["window_ms"], "COMPLETE_CLOCK_WINDOWS")
        mapping, sizes = unit_mapping(config["excitatory_neurons"],config["inhibitory_neurons"],config["excitatory_preference_bins"])
        self.fields = np.asarray(settings["fields"],dtype=np.int64)
        require(len(self.fields)>0 and np.all(np.diff(self.fields)>0)
                and np.all((0<=self.fields)&(self.fields<config["excitatory_preference_bins"])), "E_FIELDS_REQUIRED")
        self.units = np.flatnonzero(np.isin(mapping[:config["excitatory_neurons"]],self.fields))
        self.field_slot = np.searchsorted(self.fields,mapping[self.units])
        self.sizes = sizes[self.fields]
        self.lookup = np.full(config["excitatory_neurons"],-1,dtype=np.int64);self.lookup[self.units]=np.arange(len(self.units))
        self.phase = np.full(self.ticks,-1,dtype=np.int8)
        for pi, (name,(start,stop)) in enumerate(self.phases.items()):
            require(0<=start<stop<=config["duration_ms"] and (stop-start)%settings["window_ms"]==0, "COMPLETE_PHASE_WINDOWS")
            self.phase[round(start/config["dt_ms"]):round(stop/config["dt_ms"])]=pi
        probability = native._RATE_HZ*self.dt
        self.law = truncated_law(native._BINOMIAL_N,probability,self.window_ticks,settings["physical_maximum_count"],
                                 settings["decimal_scale_digits"],settings["factorial_numeric_allowance"])
        shape=(len(self.phases),len(self.fields))
        self.success_weights=np.zeros((*shape,self.law["cutoff"]+1),dtype=np.int64)
        self.eligible_weights=np.zeros(shape,dtype=np.int64)
        self.observed_factorial=np.zeros(shape,dtype=np.int64)
        self.counts=np.zeros(len(self.units),dtype=np.int64)
        self.base=self.actual=None
        self.window_id=-1;self.observed_ticks=0;self.pending=None
        self.parity_units=0;self.nonmonotone_candidate_states=0;self.actual_above_cutoff=0
        self.capture_seconds=0.;self.failure_snapshot={}
        self.anchors=observer_anchors()
        self.arithmetic=executor._Arithmetic("numpy")

    def pre_rk2(self, local):
        tick=int(local["tick"])
        require(tick==self.observed_ticks, "SOURCE_TICK_SEQUENCE")
        self.pending=None
        if tick==0 or tick//self.window_ticks!=self.window_id or self.phase[tick]<0:
            return
        e=local["e"];constants=local["constants"]["E"]
        refractory_ticks=int(np.asarray((constants["refractory"]+1e-3*self.dt)/self.dt,dtype=np.int64))
        available=self.arithmetic.integer_steps(local["t"]-e["lastspike"][self.units],self.dt)>=refractory_ticks
        slots=np.flatnonzero((self.counts>0)&available)
        if not len(slots):return
        units=self.units[slots]
        state={k:e[k][units].copy() for k in STATE_FIELDS+E_FIELDS}
        require(np.array_equal(state["s_AMPA"],self.base[slots]+self.actual[slots]), "PRE_INPUT_AMPA_RECONSTRUCTION")
        decisions, grid=candidate_step(state,self.base[slots],np.arange(self.law["cutoff"]+1,dtype=float)[None,:],constants,local["t"],self.dt)
        actual_decisions, actual=candidate_step(state,self.base[slots],self.actual[slots],constants,local["t"],self.dt)
        inside=self.actual[slots]<=self.law["cutoff"]
        idx=np.flatnonzero(inside);ks=self.actual[slots][inside].astype(np.int64)
        require(np.array_equal(grid["v"][idx,ks],actual["v"][inside])
                and np.array_equal(decisions[idx,ks],actual_decisions[inside]), "GRID_ACTUAL_TRANSITION_PARITY")
        self.actual_above_cutoff+=int(np.count_nonzero(~inside))
        self.nonmonotone_candidate_states+=int(np.count_nonzero(np.any(np.diff(decisions.astype(np.int8),axis=1)<0,axis=1)))
        pi=int(self.phase[tick]);weights=self.counts[slots]
        for fi in range(len(self.fields)):
            mask=self.field_slot[slots]==fi
            self.eligible_weights[pi,fi]+=weights[mask].sum()
            self.success_weights[pi,fi]+=weights[mask]@decisions[mask].astype(np.int64)
        self.pending=(units,actual,actual_decisions)

    def pre_threshold(self,local):
        if self.pending is None:return
        units,expected,decisions=self.pending
        e=local["e"]
        keys=("v","s_AMPA","s_GABA","not_refractory")
        agree={k:np.array_equal(expected[k],e[k][units]) for k in keys}
        actual_decisions=(e["v"][units]>local["constants"]["E"]["threshold"])&e["not_refractory"][units]
        if not all(agree.values()) or not np.array_equal(decisions,actual_decisions):
            self.failure_snapshot=dict(unit=units,tick=np.array(local["tick"]),
                **{"expected_"+k:expected[k] for k in keys},**{"actual_"+k:e[k][units].copy() for k in keys})
            raise ValueError("PFC_MULTISPIKE_FULL_SOURCE_TRANSITION_PARITY:"+str(agree))
        self.parity_units+=len(units)

    def pre_input(self,local):
        tick=int(local["tick"]);window=tick//self.window_ticks
        if window!=self.window_id:
            self.counts.fill(0);self.window_id=window
        slots=self.lookup[local["spikes"]["E"]];slots=slots[slots>=0]
        pi=int(self.phase[tick])
        if pi>=0:
            np.add.at(self.observed_factorial[pi],self.field_slot[slots],2*self.counts[slots])
        self.counts[slots]+=1
        require(np.all(self.counts<=self.settings["physical_maximum_count"]), "SOURCE_WINDOW_COUNT_SUPPORT")
        self.base=local["e"]["s_AMPA"][self.units].copy()
        self.actual=local["drive"]["E"][tick%local["chunk_rows"],self.units].copy()
        self.observed_ticks=tick+1

    @contextmanager
    def observing(self):
        previous=sys.gettrace();target=executor.simulate.__code__
        stages={value["line"]:name for name,value in self.anchors.items()}
        def trace(frame,event,arg):
            if frame.f_code is not target:return None
            if event=="line" and frame.f_lineno in stages:
                started=time.perf_counter()
                getattr(self,stages[frame.f_lineno])(frame.f_locals)
                self.capture_seconds+=time.perf_counter()-started
            return trace
        try:
            sys.settrace(trace);yield self
        finally:sys.settrace(previous)

    def results(self):
        rows=[];scale=int(self.law["scale"]);complete=self.observed_ticks==self.ticks
        for pi,(phase,(start,stop)) in enumerate(self.phases.items()):
            windows=round((stop-start)/self.settings["window_ms"])
            for fi,field in enumerate(self.fields):
                denominator=int(self.sizes[fi])*windows
                lo,hi,omitted=integrate_histogram(self.success_weights[pi,fi],int(self.eligible_weights[pi,fi]),self.law)
                rows.append(dict(phase=phase,field_index=int(field),field_units=int(self.sizes[fi]),declared_windows=windows,source_path_complete=complete,
                    observed_complete_windows=max(0,min(self.observed_ticks,round(stop/self.config["dt_ms"]))-round(start/self.config["dt_ms"]))//self.window_ticks,
                    denominator=denominator,eligible_previous_count_weight=int(self.eligible_weights[pi,fi]),
                    observed_factorial_sum=int(self.observed_factorial[pi,fi]),
                    compensator_sum_lower_scaled=str(lo),compensator_sum_upper_scaled=str(hi),omitted_contribution_upper_scaled=str(omitted),
                    factorial_compensator_lower=lo/(scale*denominator) if complete else None,
                    factorial_compensator_upper=hi/(scale*denominator) if complete else None,
                    conditional_input_interval_width=(hi-lo)/(scale*denominator) if complete else None))
        return rows


def source_files(root,settings,names):
    base=root/settings["source"]["archive"]
    require(sha256_file(base/"delivery.json")==settings["source"]["delivery_sha256"], "SOURCE_DELIVERY_BINDING")
    delivery=json.loads((base/"delivery.json").read_text())
    require(delivery["status"]=="ARCHIVED_PFC_SOURCE_EXECUTION" and delivery["source_execution_complete"] is True
            and delivery["exit_code"]==0, "SOURCE_COMPLETION")
    manifest={e["path"]:e for e in delivery["files"]};result={}
    for name in names:
        path=base/"source_run"/name;entry=manifest["source_run/"+name]
        require(path.stat().st_size==entry["bytes"] and sha256_file(path)==entry["sha256"], "SOURCE_FILE_BINDING:"+name)
        result[name]=path
    return result,manifest


def prepare(config_path,root,output):
    settings=yaml.safe_load(config_path.read_text())
    require(settings["schema"]=="pfc_multispike_compensator_v1" and settings["case_id"]=="pfc_95001_strong_cue"
            and settings["source_rng_seed"]==95001 and settings["expected_units"]==100000
            and settings["duration_ms"]==5000. and settings["dt_ms"]==.05
            and settings["window_ms"]==5. and settings["physical_maximum_count"]==3
            and settings["fields"]==[3,4,5,6,7] and settings["phase_names"]==list(PHASES)
            and settings["confirmation_execution_released"] is False and settings["production_episodes"]==0, "FROZEN_READOUT_SCOPE")
    files,manifest=source_files(root,settings,("report.json","config.yaml","initial_voltage.npy"))
    require(sha256_file(files["report.json"])==settings["source"]["report_sha256"], "SOURCE_REPORT_BINDING")
    report=json.loads(files["report.json"].read_text());config=yaml.safe_load(files["config.yaml"].read_text());case=report["selected_case"]
    require(report["status"]=="PFC_SOURCE_EXECUTION_COMPLETE" and config==report["resolved_config"]
            and case["case_id"]==settings["case_id"] and case["source_rng_seed"]==95001
            and case["stimulus_strength_nA"]==.5 and case["distractor_strength_nA"]==0., "SOURCE_CASE")
    require(config["dt_ms"]==settings["dt_ms"] and config["duration_ms"]==settings["duration_ms"]
            and config["excitatory_neurons"]==80000 and config["inhibitory_neurons"]==20000, "SOURCE_DIMENSIONS")
    bindings={}
    for module in (executor,native):
        relative="Libs/population_dynamics/"+Path(module.__file__).name
        digest=sha256_file(Path(module.__file__))
        require(digest==manifest[relative]["sha256"], "UNCHANGED_SOURCE_IMPLEMENTATION:"+relative)
        bindings[relative]=digest
    initial,drive=native.create_drive(config,case)
    require(np.array_equal(initial,np.load(files["initial_voltage.npy"],allow_pickle=False)), "SOURCE_INITIAL_VOLTAGE")
    observer=MultispikeObserver(config,settings)
    require(drive.probability==observer.law["probability"], "SOURCE_PROBABILITY")
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(status="PREPARED_PFC_MULTISPIKE_READOUT",settings=settings,settings_sha256=sha256_file(config_path),
        full_config=config,selected_case=case,source_code_bindings=bindings,observer_anchors=observer.anchors,
        law=observer.law,source_input_manifest={name:manifest["source_run/"+name] for name in
            ("report.json","config.yaml","initial_voltage.npy","raw.npz","terminal_state.npz","final_rng.npz")},
        implementation_sha256=sha256_file(Path(__file__)),source_replays=0,new_independent_blocks=0,
        confirmation_execution_released=False,production_episodes=0)
    freeze_json(output/"plan.json",plan);return plan


def execute(prepared,root,output):
    output.mkdir(parents=True,exist_ok=False);plan=json.loads((prepared/"plan.json").read_text())
    require(plan["status"]=="PREPARED_PFC_MULTISPIKE_READOUT" and plan["implementation_sha256"]==sha256_file(Path(__file__))
            and plan["observer_anchors"]==observer_anchors(), "PREPARED_IMPLEMENTATION_CHANGED")
    started=time.perf_counter();observer=None;simulation=None
    report=dict(status="FAILED_PFC_MULTISPIKE_READOUT",source_replays=1,new_full_episodes=0,new_independent_blocks=0,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,production_episodes=0,
        full_bank_readout_complete=False,source_probability_coverage_qualified=False,variance_reduction_established=False,
        finite_PRNG_law_certified=False,conditioning="own_path_past_before_the_fresh_input; no_seed_state_future_rows_or_other_clock_history")
    try:
        settings,config,case=plan["settings"],plan["full_config"],plan["selected_case"]
        files,_=source_files(root,settings,tuple(plan["source_input_manifest"]))
        for relative,digest in plan["source_code_bindings"].items():
            require(sha256_file(root/relative)==digest,"EXECUTED_SOURCE_CHANGED")
        initial,drive=native.create_drive(config,case)
        require(np.array_equal(initial,np.load(files["initial_voltage.npy"],allow_pickle=False)),"REPLAY_INITIAL_VOLTAGE")
        observer=MultispikeObserver(config,settings)
        with observer.observing():
            simulation=executor.simulate(config,case,initial,drive=drive,backend="numpy",observe_applied_current=False)
        checks={}
        with np.load(files["raw.npz"],allow_pickle=False) as old:
            for key,value in simulation["raw"].items():checks["raw/"+key]=np.array_equal(value,old[key])
        with np.load(files["terminal_state.npz"],allow_pickle=False) as old:
            require(set(old.files)==set(simulation["terminal_state"]),"TERMINAL_KEYS")
            for key,value in simulation["terminal_state"].items():checks["terminal/"+key]=np.array_equal(value,old[key])
        state=native._state_arrays(drive.get_state())
        with np.load(files["final_rng.npz"],allow_pickle=False) as old:
            require(set(old.files)==set(state),"RNG_KEYS")
            for key,value in state.items():checks["rng/"+key]=np.array_equal(value,old[key])
        require(all(checks.values()),"REPLAY_ARRAY_PARITY:"+str([k for k,v in checks.items() if not v]))
        require(observer.observed_ticks==observer.ticks,"COMPLETE_OBSERVER_PATH")
        report.update(status="COMPLETE_PFC_MULTISPIKE_READOUT",all_source_arrays_bitwise_equal=True,source_array_checks=checks,
            observed_source_blocks=1,available_original_blocks=32,source_case_id=case["case_id"],source_rng_seed=case["source_rng_seed"],
            source_units=config["excitatory_neurons"]+config["inhibitory_neurons"],source_duration_ms=config["duration_ms"],
            hazard_units=len(observer.units),observed_ticks=observer.observed_ticks,conditional_transition_parity_units=observer.parity_units,
            nonmonotone_retained_candidate_states=observer.nonmonotone_candidate_states,actual_inputs_above_cutoff=observer.actual_above_cutoff,
            observer_seconds=observer.capture_seconds,simulation_cost=simulation["cost"],law=observer.law,
            prepared_plan_sha256=sha256_file(prepared/"plan.json"),source_binding=settings["source"],
            interpretation="Per-unit factorial compensator on one complete existing source path, with directed conditional-input bounds. Equality is in expectation over histories, not a pathwise bound on realized counts. No full-history confidence or variance-reduction claim.")
    except Exception as exc:
        report["error"]=dict(type=type(exc).__name__,message=str(exc))
        if simulation is not None:
            np.savez_compressed(output/"failed_replay_raw.npz",**simulation["raw"])
            np.savez_compressed(output/"failed_replay_terminal.npz",**simulation["terminal_state"])
        raise
    finally:
        if observer is not None:
            rows=observer.results()
            with (output/"factorial_compensator.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            np.savez_compressed(output/"weighted_input_responses.npz",success_weights=observer.success_weights,
                eligible_weights=observer.eligible_weights,observed_factorial=observer.observed_factorial,
                field_indices=observer.fields,field_units=observer.sizes,phase_names=np.array(list(observer.phases)),
                input_counts=np.arange(observer.law["cutoff"]+1))
            if observer.failure_snapshot:np.savez(output/"failed_transition.npz",**observer.failure_snapshot)
            report["partial_observed_ticks"]=observer.observed_ticks
        report["wall_seconds"]=time.perf_counter()-started
        report["outputs"]={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file()}
        freeze_json(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","execute"))
    parser.add_argument("--project-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path);parser.add_argument("--prepared",type=Path)
    args=parser.parse_args()
    result=prepare(args.config,args.project_root,args.output) if args.stage=="prepare" else execute(args.prepared,args.project_root,args.output)
    print(json.dumps(dict(status=result["status"],output=str(args.output))))


if __name__=="__main__":main()

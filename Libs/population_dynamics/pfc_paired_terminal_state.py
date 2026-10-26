"""Compare complete weak-cue/omission terminal states from existing PFC episodes.

A zero late readout is not automatically a zero hidden state or a population
identity. This diagnostic does not use new simulation or raw spike arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as executor
from .pfc_confirmation_count_scores import bound_report, require
from .pfc_confirmation_uncertainty import write_rows
from .pfc_scale_analysis import _source_manifest, _source_bound_file
from .pfc_source_execution import _STATE_FIELDS, _IDENTITY_FIELDS
from .provenance import freeze_json, sha256_file

DYNAMIC_FIELDS = {'E':('v','s_AMPA','s_GABA','s_NMDA','x','I_stim'), 'I':('v','s_AMPA','s_GABA')}
STATE_UNITS = {'v':'V','I_stim':'A','lastspike':'s','not_refractory':'boolean',
               'spike_counts':'spikes','remaining_refractory_ticks':'ticks'}


def difference_summary(left, right):
    a,b = np.asarray(left),np.asarray(right)
    require(a.shape == b.shape and a.ndim == 1 and a.dtype == b.dtype
            and a.dtype.kind in 'fiub' and np.isfinite(a).all() and np.isfinite(b).all(),
            'MATCHING_FINITE_TERMINAL_ARRAYS_REQUIRED')
    delta = a.astype(np.float64)-b.astype(np.float64)
    absolute = np.abs(delta)
    peak=float(absolute.max(initial=0))
    rms=0. if peak==0 else peak*float(np.sqrt(np.mean((delta/peak)**2)))
    return dict(values=len(a),different_values=int(np.count_nonzero(a != b)),
        exact_equal=bool(np.array_equal(a,b)),maximum_absolute_difference=float(absolute.max(initial=0)),
        l1_difference=float(absolute.sum()),root_mean_square_difference=rms,
        left_minimum=float(a.min()),left_maximum=float(a.max()),right_minimum=float(b.min()),right_maximum=float(b.max()),
        left_l1=float(np.abs(a.astype(float)).sum()),right_l1=float(np.abs(b.astype(float)).sum()))


def validate_terminal(state, ne, ni):
    expected={population+'__'+name for population in ('E','I') for name in (*_STATE_FIELDS[population],*_IDENTITY_FIELDS)}
    require(set(state)==expected,'COMPLETE_TERMINAL_STATE_SCHEMA_REQUIRED')
    for population,n,offset in [('E',ne,0),('I',ni,ne)]:
        for name in (*_STATE_FIELDS[population],*_IDENTITY_FIELDS):
            x=state[population+'__'+name]
            require(x.shape==(n,) and x.dtype.kind in 'fiub' and np.isfinite(x).all(),'COMPLETE_FINITE_UNIT_STATE_REQUIRED')
        require(np.array_equal(state[population+'__source_indices'],np.arange(n))
                and np.array_equal(state[population+'__archive_indices'],np.arange(n)+offset),'TERMINAL_UNIT_MAPPING_CHANGED')
        require(state[population+'__not_refractory'].dtype.kind=='b'
                and state[population+'__spike_counts'].dtype.kind in 'iu'
                and np.all(state[population+'__spike_counts']>=0),'STATE_FLAG_OR_COUNTER_TYPE_CHANGED')


def remaining_refractory(state, population, config):
    dt=config['dt_ms']/1000; ticks=round(config['duration_ms']/config['dt_ms'])
    time_next=ticks*dt
    constants=executor._constants(config,population)
    refractory_ticks=int(np.asarray((constants['refractory']+1e-3*dt)/dt,dtype=np.int64))
    elapsed=executor._Arithmetic('numpy').integer_steps(time_next-state[population+'__lastspike'],dt)
    return np.maximum(0,refractory_ticks-elapsed)


def compare_terminal_states(left,right,config):
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons']
    validate_terminal(left,ne,ni);validate_terminal(right,ne,ni)
    rows=[];dynamic_equal=True;voltage_equal=True
    for population in ('E','I'):
        for name in (*_STATE_FIELDS[population],'spike_counts'):
            key=population+'__'+name;summary=difference_summary(left[key],right[key])
            role=('dynamic' if name in DYNAMIC_FIELDS[population] else
                  ('cumulative_counter' if name=='spike_counts' else 'cached_or_historical'))
            rows.append(dict(population=population,variable=name,units=STATE_UNITS.get(name,'dimensionless'),role=role,**summary))
            if role=='dynamic':dynamic_equal &= summary['exact_equal']
            if name=='v':voltage_equal &= summary['exact_equal']
        a,b=remaining_refractory(left,population,config),remaining_refractory(right,population,config)
        summary=difference_summary(a,b);dynamic_equal &= summary['exact_equal']
        rows.append(dict(population=population,variable='remaining_refractory_ticks',units='ticks',role='dynamic',**summary))
    kernel=executor._Arithmetic('numpy').rfft(executor._kernel(ne,config))
    projections=[]
    for state in (left,right):
        value=state['E__s_NMDA']
        projections.append((np.fft.irfft(kernel*np.fft.rfft(value)),np.asarray([np.cumsum(value,dtype=np.float64)[-1]])))
    projected_equal=True
    for i,population in enumerate(('E','I')):
        summary=difference_summary(projections[0][i],projections[1][i]);projected_equal &= summary['exact_equal']
        rows.append(dict(population=population,variable='next_s_NMDA_total',units='dimensionless',role='recomputed_input',**summary))
    return rows,dict(dynamic_state_exactly_equal=bool(dynamic_equal),voltage_exactly_equal=bool(voltage_equal),
        next_nmda_inputs_exactly_equal=bool(projected_equal),
        private_nmda_state_exactly_equal=bool(np.array_equal(left['E__s_NMDA'],right['E__s_NMDA'])),
        cumulative_spike_counters_equal=bool(all(np.array_equal(left[p+'__spike_counts'],right[p+'__spike_counts']) for p in ('E','I'))))


def stage_case(origin, destination, binding, cohort, expected_seed, condition):
    raw=(origin/'delivery.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==binding['delivery_sha256'],'SOURCE_DELIVERY_CHANGED')
    delivery=json.loads(raw);manifest=_source_manifest(delivery)
    case_id=f'pfc_{expected_seed}_{condition}'
    require(delivery['status']=='ARCHIVED_PFC_SOURCE_EXECUTION' and delivery['result_status']=='PFC_SOURCE_EXECUTION_COMPLETE'
            and delivery['source_execution_complete'] is True and delivery['case_id']==case_id,'COMPLETE_SOURCE_CASE_REQUIRED')
    destination.mkdir(parents=True,exist_ok=False);files=[]
    for name in ('report.json','terminal_state.npz','initial_voltage.npy','final_rng.npz'):
        key='source_run/'+name;row=manifest[key];target=destination/name
        shutil.copy2(origin/key,target)
        require(target.stat().st_size==row['bytes'] and sha256_file(target)==row['sha256'],'STAGED_TERMINAL_INPUT_CHANGED:'+name)
        files.append(dict(path=key,bytes=row['bytes'],sha256=row['sha256']))
    report=json.loads((destination/'report.json').read_text())
    require(sha256_file(destination/'report.json')==binding['report_sha256'] and report['case_id']==case_id
            and report['status']=='PFC_SOURCE_EXECUTION_COMPLETE' and report['source_execution_complete'] is True
            and report['backend']=='numpy' and report['seeds']=={'source_rng_seed':expected_seed}
            and report['resolved_config']==cohort['config'] and report['config_sha256']==cohort['report']['config_sha256']
            and report['completed_duration_ticks']==round(5000/cohort['config']['dt_ms'])
            and report['code']['executor_sha256']==sha256_file(Path(executor.__file__))
            and report['code']['runner_sha256']==binding['source_runner_sha256']
            and all(check['passed'] is True for check in report['validations'].values()),'SOURCE_CASE_BINDING_OR_VALIDATION_CHANGED')
    require(report['selected_case']['stimulus_strength_nA']==(0.06 if condition=='cue' else 0.)
            and report['selected_case']['distractor_strength_nA']==0.,'FROZEN_WEAK_OMISSION_CONTROL_REQUIRED')
    for name in ('terminal_state.npz','initial_voltage.npy','final_rng.npz'):
        require(sha256_file(destination/name)==report['outputs'][name]['sha256'],'SOURCE_OUTPUT_REPORT_BINDING_CHANGED')
    with np.load(destination/'terminal_state.npz',allow_pickle=False) as stored:state={name:stored[name] for name in stored.files}
    with np.load(destination/'final_rng.npz',allow_pickle=False) as stored:rng={name:stored[name] for name in stored.files}
    initial=np.load(destination/'initial_voltage.npy',allow_pickle=False)
    return state,initial,rng,report,dict(archive=str(origin),case_id=case_id,delivery_sha256=binding['delivery_sha256'],
                                     report_sha256=binding['report_sha256'],files=files)


def run(config_path, project_root, output, scratch_parent):
    settings=yaml.safe_load(config_path.read_text())
    require(settings['schema']=='pfc_paired_terminal_state_v1' and settings['scope']=='development_calibration'
            and settings['conditions']==['cue','omission'] and settings['confirmation_execution_released'] is False,
            'BOUNDED_CALIBRATION_TERMINAL_DIAGNOSTIC_REQUIRED')
    comparison=bound_report(project_root/settings['comparison_archive'],settings['comparison_report_sha256'])
    seeds=settings['joint_rng_seeds'];require(seeds==list(range(95001,95033)),'COMPLETE_EXISTING_32_BLOCK_CALIBRATION_REQUIRED')
    cohorts={}
    for label,path in settings['field_archives'].items():
        root=project_root/path;expected=comparison['inputs'][label]
        require(sha256_file(root/'delivery.json')==expected['delivery_sha256'],'SOURCE_COHORT_DELIVERY_CHANGED')
        manifest=_source_manifest(json.loads((root/'delivery.json').read_text()))
        report=json.loads(_source_bound_file(root,manifest,'analysis/report.json').read_text())
        config_path_source=_source_bound_file(root,manifest,'config.yaml');config=yaml.safe_load(config_path_source.read_text())
        origins=json.loads(_source_bound_file(root,manifest,'source_archives.json').read_text())
        require(report['status']=='COMPLETE_SOURCE_FLUCTUATION_CALIBRATION' and report['joint_rng_blocks']==seeds
                and report['completed_source_cases']==report['registered_source_cases']==len(origins)==96
                and report['coupled_core_units']==100000 and config['duration_ms']==5000.
                and sha256_file(config_path_source)==report['config_sha256'],'COMPLETE_FROZEN_COHORT_REQUIRED')
        bindings={}
        require(len(report['source_bindings'])==len(origins),'SOURCE_INDEX_LENGTH_CHANGED')
        for origin,binding in zip(origins,report['source_bindings']):
            case_id=binding['case_id'];require(case_id not in bindings,'DUPLICATE_SOURCE_CASE')
            bindings[case_id]=(Path(origin),binding)
        cohorts[label]=dict(config=config,report=report,bindings=bindings)
    output.mkdir(parents=True,exist_ok=False);scratch_parent.mkdir(parents=True,exist_ok=True)
    state_rows,pair_rows,inputs=[],[],[];started=time.perf_counter()
    report=dict(status='RUNNING',settings=settings,settings_sha256=sha256_file(config_path),new_source_simulations=0,
        raw_spike_data_reread=False,scientific_fidelity_qualified=False,confirmation_execution_released=False,
        full_time_series_state_available=False,full_terminal_state_available=True,production_episodes=0)
    try:
        for label,cohort in cohorts.items():
            for seed in seeds:
                with tempfile.TemporaryDirectory(prefix=f'{label}_{seed}_',dir=scratch_parent) as directory:
                    states,initials,rngs,source_reports=[],[],[],[]
                    for condition in settings['conditions']:
                        case_id=f'pfc_{seed}_{condition}';origin,binding=cohort['bindings'][case_id]
                        state,initial,rng,source_report,used=stage_case(origin,Path(directory)/condition,binding,cohort,seed,condition)
                        states.append(state);initials.append(initial);rngs.append(rng);source_reports.append(source_report);inputs.append(used)
                    require(np.array_equal(initials[0],initials[1]),'PAIRED_INITIAL_VOLTAGES_DIFFER')
                    require(set(rngs[0])==set(rngs[1]) and all(np.array_equal(rngs[0][key],rngs[1][key]) for key in rngs[0]),'PAIRED_FINAL_RNG_STATES_DIFFER')
                    require(source_reports[0]['driver']==source_reports[1]['driver'],'PAIRED_DRIVER_METADATA_DIFFER')
                    rows,checks=compare_terminal_states(states[0],states[1],cohort['config'])
                    identity=dict(cohort=label,dt_ms=cohort['config']['dt_ms'],source_rng_seed=seed,contrast='cue_minus_omission')
                    state_rows.extend(dict(identity,**row) for row in rows)
                    pair_rows.append(dict(identity,initial_voltages_equal=True,final_rng_states_equal=True,**checks))
                    print(json.dumps(dict(identity,**checks)),flush=True)
        report.update(status='COMPLETE_PAIRED_TERMINAL_STATE_DIAGNOSTIC')
    except Exception as exc:
        report.update(status='FAILED_PAIRED_TERMINAL_STATE_DIAGNOSTIC',failure=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        outputs={}
        for name,rows in [('state_differences.csv',state_rows),('pair_summary.csv',pair_rows)]:
            if rows:
                path=output/name;write_rows(path,rows);outputs[name]=dict(bytes=path.stat().st_size,sha256=sha256_file(path))
        freeze_json(output/'source_inputs.json',inputs);outputs['source_inputs.json']=dict(bytes=(output/'source_inputs.json').stat().st_size,sha256=sha256_file(output/'source_inputs.json'))
        field_summary={}
        for row in state_rows:
            key=row['population']+'__'+row['variable']
            if key not in field_summary:field_summary[key]=dict(units=row['units'],pairs=0,exact_equal_pairs=0,maximum_absolute_difference=0.,maximum_l1_difference=0.)
            item=field_summary[key];item['pairs']+=1;item['exact_equal_pairs']+=int(row['exact_equal'])
            item['maximum_absolute_difference']=max(item['maximum_absolute_difference'],row['maximum_absolute_difference'])
            item['maximum_l1_difference']=max(item['maximum_l1_difference'],row['l1_difference'])
        report.update(completed_pairs=len(pair_rows),expected_pairs=3*len(seeds),existing_source_episodes_consumed=len(inputs),
            original_independent_blocks=len(seeds),distinct_clock_levels=len(cohorts),units_per_episode=100000,
            staged_source_bytes=sum(row['bytes'] for case in inputs for row in case['files']),
            aggregate_checks={key:sum(int(row[key]) for row in pair_rows) for key in ('dynamic_state_exactly_equal','voltage_exactly_equal','next_nmda_inputs_exactly_equal','private_nmda_state_exactly_equal','cumulative_spike_counters_equal')},
            per_state_field=field_summary,outputs=outputs,elapsed_seconds=time.perf_counter()-started,
            implementation_sha256=sha256_file(Path(__file__)),executor_sha256=sha256_file(Path(executor.__file__)),
            interpretation='Observed terminal-state comparison only. Equal late response counts or equal recomputed NMDA inputs do not imply equality of private synaptic states. Neither equality nor small observed differences proves a population zero, a future-time contraction bound, or equivalence in uninspected blocks.',
            refractory_state='next source tick uses archived lastspike and the pinned executor timestep rule; expired timestamp differences are reported separately',
            state_time='after final source tick; next NMDA inputs are recomputed using that terminal state, not reused from the pre-update cache')
        freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scratch-parent',type=Path,required=True)
    args=parser.parse_args();r=run(args.config,args.project_root,args.output,args.scratch_parent)
    print(json.dumps({key:r[key] for key in ('status','completed_pairs','aggregate_checks')}));return 0


if __name__=='__main__':
    raise SystemExit(main())

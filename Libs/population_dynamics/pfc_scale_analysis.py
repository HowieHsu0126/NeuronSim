"""Replay PFC source readouts at the reference size or a registered streamed size."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import h5py
import numpy as np
import yaml

from .circuit_data import CircuitDataset, SCHEMA
from . import pfc_reference_replay, pfc_source_comparison
from .pfc_delay_analysis import (PHASES, CONDITIONS, SOURCE_REVISION, SOURCE_ENTRY_SHA256,
    READOUT_RTOL, READOUT_ATOL, _check_readout, read, require, write_json, write_rows)
from .pfc_response_analysis import summarize_counts
from .provenance import canonical_json, sha256_file
from . import pfc_kernel_parameters as kernel_parameters

SOURCE_SEEDS = (93001,93002,93003)
N_E, N_I, DURATION_TICKS, BIN_TICKS = 2048,512,100000,100
CIRCUIT = 'pfc_source_reference'
RAW_COLUMNS = {'spike_ticks','spike_unit_indices','spike_seconds','native_spike_counts','terminal_voltage_mV'}


def geometry(n_e, center=120., width=30.):
    middle=round(n_e/360*center);half=round(n_e/360*width/2)
    ids={(middle+offset)%n_e for offset in range(-half,half+1)}
    require(0<len(ids)<n_e,'proper source cue-preference subset required')
    return ids,dict(cue_units=len(ids),cue_fraction_e=len(ids)/n_e,
                    effective_cue_center_deg=middle*360/n_e)


def config_contract(design, native_config, factorized_config):
    require(design['status']=='REGISTERED_BEFORE_SOURCE_EXECUTION'
            and design['scope']=='bounded_source_scaling_diagnostic'
            and design['source_rng_blocks']==list(SOURCE_SEEDS)
            and design['training_seeds']==[41,42,43] and design['cases']==9
            and design['duration_ms']==5000. and design['units']==N_E+N_I
            and design['phases_ms']==PHASES and design['relative_tolerance']==0.
            and design['terminal_voltage_mV_max_absolute_error']==1e-10
            and design['required_event_and_integer_arrays']=='exact'
            and design['production_episodes']==0
            and design['full_scale_qualified'] is False and design['scientific_fidelity_qualified'] is False,
            'registered reference-size source design required')
    expected=dict(scope='source_reproduction',expected_source_revision=SOURCE_REVISION,
        excitatory_neurons=N_E,inhibitory_neurons=N_I,weight_scaling_factor=1.,
        dt_ms=.05,duration_ms=5000.,stimulus_center_deg=120.,stimulus_width_deg=30.,
        stimulus_start_ms=100.,stimulus_duration_ms=200.,distractor_center_deg=90.,
        distractor_width_deg=40.,distractor_start_ms=400.,distractor_duration_ms=50.,
        excitatory_preference_bins=16,readout_bin_ms=5.,state_monitored_per_population=8)
    require(all(native_config[k]==v for k,v in expected.items())
            and native_config.get('synapse_execution','native_source')=='native_source'
            and factorized_config==dict(native_config,synapse_execution='factorized_uniform_projections'),
            'native/factorized physical parameters, clock or controls differ')
    registered={f'pfc_{seed}_{condition}':(seed,condition) for seed in SOURCE_SEEDS for condition in CONDITIONS}
    require(len(native_config['cases'])==9 and {r['case_id'] for r in native_config['cases']}==set(registered),
            'complete registered configuration cases required')
    for row in native_config['cases']:
        seed,condition=registered[row['case_id']]
        require(row['source_rng_seed']==seed and row['stimulus_strength_nA']==CONDITIONS[condition]
                and row['distractor_strength_nA']==0.,'configuration case/RNG/current changed')
    return registered


def preflight(paths, design, configs, *, comparison_implementation: Path | None = None,
              replay_implementation: Path | None = None):
    """Inspect receipts before arrays; archive callers bind their recorded code files.

    Defaults require the current implementations. Historical archive readers supply
    the implementation files they already verified against the archive manifest.
    No archived code is executed.
    """
    comparison_implementation = (Path(pfc_source_comparison.__file__) if comparison_implementation is None
                                 else Path(comparison_implementation))
    replay_implementation = (Path(pfc_reference_replay.__file__) if replay_implementation is None
                             else Path(replay_implementation))
    statuses=[];loaded={}
    for role,path in paths.items():
        try:
            item=read(path);loaded[role]=item
            statuses.append(dict(role=role,path=str(path),status=item.get('status','MISSING_STATUS')))
        except FileNotFoundError:
            statuses.append(dict(role=role,path=str(path),status='MISSING'))
    expected=dict(native='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',
        factorized='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',fresh='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS',
        comparison='PASS_SOURCE_UNIFORM_PROJECTION_EQUIVALENCE',replay='PASS')
    require(all(row['status']==expected[row['role']] for row in statuses),
            'all native/factorized/fresh producers and both comparison receipts must complete: '+json.dumps(statuses))
    reports={key:loaded[key] for key in ('native','factorized','fresh')}
    registered=config_contract(design,*configs)
    for role,report in reports.items():
        config=configs[0] if role=='native' else configs[1]
        execution='native_source' if role=='native' else 'factorized_uniform_projections'
        require(report['resolved_config']==config and report['source_revision']==SOURCE_REVISION
                and report['source_entry_sha256']==SOURCE_ENTRY_SHA256 and report['scope']=='source_reproduction'
                and report['synapse_execution']==execution and report['independent_source_rng_blocks']==3
                and report['full_scale_qualified'] is False and report['scientific_fidelity_qualified'] is False
                and len(report['cases'])==9 and {r['case_id'] for r in report['cases']}==set(registered),
                'complete source/config/case execution binding required')
        for row in report['cases']:
            seed,condition=registered[row['case_id']]
            require(row['core_units']==N_E+N_I and row['source_rng_seed']==seed
                    and row['stimulus_strength_nA']==CONDITIONS[condition] and row['distractor_strength_nA']==0.,
                    'native case control, unit count or source RNG changed')
    for key in ('brian2_version','numpy_version'):
        require(len({r[key] for r in reports.values()})==1,'native simulator environment differs')
    comparison,replay=loaded['comparison'],loaded['replay']
    require(comparison['candidate_report_sha256']==sha256_file(paths['factorized'])
            and len(comparison['reference_bindings'])==1
            and comparison['reference_bindings'][0]['report_sha256']==sha256_file(paths['native'])
            and Path(comparison['reference_bindings'][0]['path']).is_absolute()
            and comparison['source_revision']==SOURCE_REVISION
            and comparison['source_entry_sha256']==SOURCE_ENTRY_SHA256
            and comparison['execution_implementation_sha256']==reports['factorized']['execution_implementation_sha256']
            and comparison['comparison_implementation_sha256']==sha256_file(comparison_implementation)
            and comparison['full_scale_qualified'] is False and comparison['scientific_fidelity_qualified'] is False,
            'source comparison is not bound to this complete native/factorized pair')
    require(reports['factorized']['execution_implementation_sha256']==reports['fresh']['execution_implementation_sha256']
            and replay['original_report_sha256']==sha256_file(paths['factorized'])
            and replay['replayed_report_sha256']==sha256_file(paths['fresh'])
            and replay['source_revision']==SOURCE_REVISION and replay['complete_episodes']==9
            and replay['core_units']==N_E+N_I and replay['refractory_violations']==0
            and replay['comparison_implementation_sha256']==sha256_file(replay_implementation)
            and replay['replay_runs_count_as_additional_samples'] is False
            and replay['full_scale_qualified'] is False and replay['scientific_fidelity_qualified'] is False,
            'complete fresh factorized replay binding required')
    require(len(comparison['cases'])==9 and {r['case_id'] for r in comparison['cases']}==set(registered),
            'source comparison must retain all nine cases')
    maximum=[]
    for row in comparison['cases']:
        seed,condition=registered[row['case_id']]
        require(row['source_rng_seed']==seed and row['stimulus_strength_nA']==CONDITIONS[condition]
                and row['spike_arrays_exact'] is True and np.isfinite(row['max_voltage_mV_difference'])
                and 0<=row['max_voltage_mV_difference']<=1e-10,
                'per-case source equality or frozen voltage tolerance failed')
        maximum.append(row['max_voltage_mV_difference'])
    require(comparison['max_voltage_mV_difference']==max(maximum),'source comparison maximum differs')
    return loaded,registered,statuses


def raw_values(root,case,*,unit_count=N_E+N_I,dt_ms=.05,duration_ticks=DURATION_TICKS):
    require(type(unit_count) is int and unit_count > 0, 'positive native unit count required')
    require(type(dt_ms) in (int,float) and np.isfinite(dt_ms) and dt_ms > 0
            and type(duration_ticks) is int and duration_ticks > 0,
            'positive explicit native clock and duration required')
    require(Path(case['raw_file']).name==case['raw_file'],'native raw path escaped its root')
    path=root/case['raw_file']
    require(sha256_file(path)==case['raw_sha256'],'native raw artifact changed')
    with np.load(path,allow_pickle=False) as stored:
        require(set(stored.files)==RAW_COLUMNS,'complete native raw schema required')
        data={name:stored[name] for name in stored.files}
    ticks,ids=data['spike_ticks'],data['spike_unit_indices']
    require(ticks.ndim==ids.ndim==1 and ticks.shape==ids.shape and ticks.dtype.kind in 'iu'
            and ids.dtype.kind in 'iu' and (ticks>=0).all() and (ticks<=duration_ticks).all()
            and (np.diff(ticks)>=0).all() and (ids>=0).all() and (ids<unit_count).all()
            and len(ticks)==case['native_events'],'native event clock/unit support changed')
    require(data['spike_seconds'].shape==ticks.shape and np.isfinite(data['spike_seconds']).all()
            and np.allclose(data['spike_seconds']*1000/dt_ms,ticks,rtol=0,atol=1e-8),
            'native seconds/ticks differ')
    require(data['native_spike_counts'].shape==(unit_count,)
            and data['native_spike_counts'].dtype.kind in 'iu'
            and np.array_equal(np.bincount(ids,minlength=unit_count),data['native_spike_counts'])
            and data['terminal_voltage_mV'].shape==(unit_count,)
            and np.isfinite(data['terminal_voltage_mV']).all(),'complete native unit counts/state required')
    return data


def dataset_binding(root,report,config,config_sha,registered):
    dataset=CircuitDataset(root/'dataset',splits=('development',))
    catalog=dataset.catalog
    rows=[row for shard in catalog['shards'] for row in shard['episodes']]
    require(len(rows)==9 and catalog['complete_episodes']==9 and len(dataset.episodes)==9
            and {r['episode_id'] for r in rows}==set(registered) and all(r['split']=='development' for r in rows)
            and set(catalog['circuits'])=={CIRCUIT},'complete development-only native archive required')
    binding=catalog['binding']
    require(binding['schema']==SCHEMA and binding['scope']=='source_reproduction'
            and binding['scenario_id']=='pfc_working_memory' and binding['source_revision']==SOURCE_REVISION
            and binding['source_entry_sha256']==SOURCE_ENTRY_SHA256
            and binding['resolved_source_config']==config and binding['config_sha256']==config_sha
            and binding['synapse_execution']=='native_source','native archive/config identity changed')
    identity=kernel_parameters.circuit_provenance(config, SOURCE_ENTRY_SHA256)
    circuit=catalog['circuits'][CIRCUIT]
    require(circuit['units']==N_E+N_I and circuit['provenance']==identity,'realized source circuit identity changed')
    with h5py.File(dataset._verified(circuit['path'],circuit['sha256']),'r') as units:
        require(np.array_equal(units['unit_ids'][:],np.arange(N_E+N_I))
                and np.array_equal(units['populations'].asstr()[:],np.array(['E']*N_E+['I']*N_I))
                and np.array_equal(units['field_index'][:],np.r_[np.arange(N_E)*16//N_E,np.full(N_I,16)])
                and tuple(units['field_names'].asstr()[:])==tuple(
                    [f'E_preference_{i*360/16:g}_{(i+1)*360/16:g}_deg' for i in range(16)]+['I_population'])
                and units.attrs['source_revision']==SOURCE_REVISION
                and units.attrs['coordinate_semantics']=='E angle=360*source_index/N_E degrees; I has no preference coordinate'
                and units.attrs['coordinate_units']=='not_available' and 'coordinates' not in units
                and json.loads(units.attrs['circuit_provenance_json'])==identity,
                'complete source unit/preference map changed')
    return dataset,identity


def archive_window(dataset,case,raw,seed,condition,identity):
    row=dataset.episodes[case['case_id']]
    require(row['duration_ticks']==DURATION_TICKS and row['dt_ms']==.05
            and row['circuit_id']==CIRCUIT and row['block_id']==f'pfc_source_block_{seed}',
            'native archive episode timing or block changed')
    shard=dataset.shards[row['shard']]
    expected_controls=np.zeros((5,2));expected_controls[1,0]=CONDITIONS[condition]
    with h5py.File(dataset._verified(shard['path'],shard['sha256']),'r') as file:
        group=file[case['case_id']];metadata=json.loads(group.attrs['metadata_json'])
        require(np.array_equal(group['spike_ticks'][:],raw['spike_ticks'])
                and np.array_equal(group['spike_unit_indices'][:],raw['spike_unit_indices'])
                and np.array_equal(group['control_ticks'][:],[0,2000,6000,8000,9000])
                and np.array_equal(group['control_values'][:],expected_controls),
                'native archive events or prescribed control program changed')
        record=metadata['source_record']
        require(all(metadata[k]==row[k] for k in ('episode_id','circuit_id','block_id','split','duration_ticks','dt_ms'))
                and metadata['seeds']=={'source_rng_seed':seed} and metadata['control_kind']=='piecewise_constant'
                and record['complete_unit_recording'] is True and record['simulation_completed'] is True
                and record['recorded_until_tick']==record['expected_duration_ticks']==DURATION_TICKS
                and record['core_unit_count']==N_E+N_I and record['source_revision']==SOURCE_REVISION
                and record['circuit_provenance']==identity and record['raw_sha256']==case['raw_sha256']
                and record['native_source_sha256']==SOURCE_ENTRY_SHA256
                and record['initial_and_background_streams']=='source_global_rng_not_separated',
                'complete native episode/stream identity required')
    window=dataset.window(case['case_id'],boundary_tick=50000,history_ticks=50000,horizon_ticks=50000,bin_ticks=BIN_TICKS)
    dense=np.concatenate([window['history'],window['targets']])
    controls=np.zeros((1000,2));controls[20:60,0]=CONDITIONS[condition]
    expected_features=[dict(name='cue_current',unit='nA',population='E',center_degree=120.,width_degree=30.),
                       dict(name='distractor_current',unit='nA',population='E',center_degree=90.,width_degree=40.)]
    require(dense.shape==(1000,N_E+N_I) and np.array_equal(window['unit_ids'],np.arange(N_E+N_I))
            and window['bin_seconds']==.005 and np.array_equal(window['known_controls'],controls)
            and window['control_features']==expected_features and window['observation_mask'].all(),
            'complete native archive observation/control mapping required')
    return dense


def scalar_readout(events,n_e,n_i,duration_ms):
    """Independent raw-event Counter and math.fsum reference at either source size."""
    counts=Counter(int(unit) for unit in events)
    cue,profile=geometry(n_e)
    e=sum(n for unit,n in counts.items() if unit<n_e)
    i=sum(n for unit,n in counts.items() if unit>=n_e)
    x=math.fsum(n*math.cos(2*math.pi*unit/n_e) for unit,n in counts.items() if unit<n_e)
    y=math.fsum(n*math.sin(2*math.pi*unit/n_e) for unit,n in counts.items() if unit<n_e)
    length=math.hypot(x,y);defined=e>0 and length>64*math.ulp(1.)*e
    angle=math.degrees(math.atan2(y,x))%360 if defined else None
    seconds=duration_ms/1000
    inside=sum(n for unit,n in counts.items() if unit in cue)
    in_rate=inside/len(cue)/seconds;out_rate=(e-inside)/(n_e-len(cue))/seconds
    return dict(excitatory_spikes=e,inhibitory_spikes=i,excitatory_rate_hz=e/n_e/seconds,
        inhibitory_rate_hz=i/n_i/seconds,cue_units=len(cue),cue_region_rate_hz=in_rate,
        noncue_region_rate_hz=out_rate,cue_selectivity_hz=in_rate-out_rate,
        vector_x_spikes=x,vector_y_spikes=y,resultant_length=length/e if e else None,
        decoded_angle_deg=angle,circular_cue_error_deg=(angle-120+180)%360-180 if defined else None,
        direction_defined=defined,all_unit_rate_hz=(e+i)/(n_e+n_i)/seconds)


def paired_effects(rows,seeds):
    result=[]
    metrics=('all_unit_rate_hz','excitatory_rate_hz','inhibitory_rate_hz','cue_selectivity_hz',
             'cue_region_rate_hz','noncue_region_rate_hz','resultant_length')
    for seed in seeds:
        for phase in PHASES:
            members={r['condition']:r for r in rows if r['source_rng_seed']==seed and r['phase']==phase}
            for left,right in (('cue','omission'),('strong_cue','omission'),('strong_cue','cue')):
                a,b=members[left],members[right]
                values={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in metrics}
                values['decoded_angle_change_deg']=None if not (a['direction_defined'] and b['direction_defined']) else (
                    a['decoded_angle_deg']-b['decoded_angle_deg']+180)%360-180
                result.append(dict(source_rng_seed=seed,phase=phase,contrast=f'{left}_minus_{right}',**values))
    require(all(all(row[k] in (None,0.) for k in metrics) for row in result if row['phase']=='baseline'),
            'paired pre-cue source responses differ')
    return result


def descriptive_cross_size(reference,rows,config):
    """Combine per-case rows with explicit independent cohorts, never pair sizes."""
    reference=Path(reference);report=read(reference/'analysis.json');replay=read(reference/'readout_replay.json')
    require(report['status']=='DESCRIBED_REPLAYED_DELAY_DIAGNOSTIC' and report['units']==1280
            and report['source_rng_blocks']==3 and report['complete_source_episodes']==9
            and report['phases_ms']==PHASES and report['full_unit_500ms_prefixes_verified']==9
            and report['source_revision']==SOURCE_REVISION and report['production_episodes']==0
            and all(report[k] is False for k in ('confidence_intervals','p_values','q_values','full_scale_qualified'))
            and report['phase_unit_count_values_verified']==57600
            and report['working_memory_qualified'] is False and report['scientific_fidelity_qualified'] is False
            and report['neuroscientific_claim_qualified'] is False
            and replay['status']=='PASS' and replay['complete_episodes']==9 and replay['source_rng_blocks']==3
            and replay['phase_unit_count_values_checked']==57600
            and replay['design_sha256']==report['design_sha256']
            and replay['analysis_artifacts']==report['artifacts']
            and replay['original_report_sha256']==report['original_report_sha256']
            and sha256_file(reference/'readout_replay.json')==report['readout_replay_sha256'],
            'verified descriptive 1280-unit source readout provenance required')
    require(report['configuration']['excitatory_neurons']==1024
            and report['configuration']['inhibitory_neurons']==256
            and report['configuration']['weight_scaling_factor']==2.
            and all(report['configuration'][key]==config[key] for key in pfc_source_comparison.PHYSICAL_FIELDS
                    if key not in ('excitatory_neurons','inhibitory_neurons','weight_scaling_factor')),
            'registered 1280-unit source comparator required')
    for name,digest in report['artifacts'].items():
        require(Path(name).name==name and sha256_file(reference/name)==digest,'cross-size reference artifact changed')
    with (reference/'responses.csv').open(newline='') as stream:
        old=list(csv.DictReader(stream))
    expected={(seed,condition,phase) for seed in (91001,91002,91003) for condition in CONDITIONS for phase in PHASES}
    require(len(old)==45 and {(int(r['source_rng_seed']),r['condition'],r['phase']) for r in old}==expected,
            'all independent old-size cases required')
    keys=('excitatory_rate_hz','inhibitory_rate_hz','cue_selectivity_hz','resultant_length','circular_cue_error_deg')
    result=[]
    for total,n_e,n_i,weight,cohort in ((1280,1024,256,2.,old),(2560,N_E,N_I,1.,rows)):
        _,profile=geometry(n_e)
        for row in cohort:
            count=profile['cue_units'] if CONDITIONS[row['condition']]>0 else 0
            require(int(row['cue_units'])==profile['cue_units'],'source-rounded cue subset differs')
            parsed={key:None if row[key] in ('',None) else float(row[key]) for key in keys}
            require(all(v is None or np.isfinite(v) for v in parsed.values()),'finite cross-size readouts required')
            result.append(dict(units=total,excitatory_neurons=n_e,inhibitory_neurons=n_i,weight_scaling_factor=weight,
                case_id=row['case_id'],source_rng_seed=int(row['source_rng_seed']),condition=row['condition'],
                phase=row['phase'],**profile,stimulated_e_units=count,stimulated_e_fraction=count/n_e,
                **parsed))
    return result,dict(reference_analysis_sha256=sha256_file(reference/'analysis.json'),
        reference_readout_replay_sha256=report['readout_replay_sha256'],paired_across_sizes=False,
        interpretation='independent source blocks; unit count, normalized synaptic weights and source-rounded cue fraction change together; not a pure N effect or increased anatomical extent')


def analyze(native,factorized,fresh,comparison_path,replay_path,design_path,native_config_path,factorized_config_path,
            output,*,reference_analysis=None):
    native,factorized,fresh,comparison_path,replay_path,design_path,native_config_path,factorized_config_path,output=map(
        Path,(native,factorized,fresh,comparison_path,replay_path,design_path,native_config_path,factorized_config_path,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    result=dict(status='CHECKING_INPUTS',full_scale_qualified=False,scientific_fidelity_qualified=False,
        working_memory_qualified=False,neuroscientific_claim_qualified=False,production_episodes=0,
        formal_training_completed=False,confidence_intervals=False,p_values=False,q_values=False)
    try:
        design=read(design_path)
        configs=[yaml.safe_load(p.read_text()) for p in (native_config_path,factorized_config_path)]
        paths=dict(native=native/'report.json',factorized=factorized/'report.json',fresh=fresh/'report.json',
                   comparison=comparison_path,replay=replay_path)
        inputs,registered,statuses=preflight(paths,design,configs)
        require(inputs['comparison']['design_sha256']==sha256_file(design_path),'source comparison design changed')
        write_rows(output/'input_status.csv',statuses)
        reports={k:inputs[k] for k in ('native','factorized','fresh')}
        indexed={role:{r['case_id']:r for r in report['cases']} for role,report in reports.items()}
        compared={r['case_id']:r for r in inputs['comparison']['cases']}
        dataset,identity=dataset_binding(native,reports['native'],configs[0],sha256_file(native_config_path),registered)
        rows,independent,phase_counts=[],[],{}
        coverage=unit_checks=scalar_checks=0
        _,profile=geometry(N_E)
        for case_id,(seed,condition) in registered.items():
            case=indexed['native'][case_id]
            for role,root in (('factorized',factorized),('fresh',fresh)):
                other=indexed[role][case_id]
                require(other['native_events']==case['native_events']
                        and Path(other['raw_file']).name==other['raw_file']
                        and sha256_file(root/other['raw_file'])==other['raw_sha256'],
                        'source comparison/replay artifact or event coverage changed')
            raw=raw_values(native,case)
            values=sum(array.size for array in raw.values())
            require(values==compared[case_id]['values'],'computed native array coverage differs from source comparison')
            coverage+=values
            dense=archive_window(dataset,case,raw,seed,condition,identity)
            ticks,ids=raw['spike_ticks'],raw['spike_unit_indices']
            require(np.array_equal(np.bincount(ids[ticks<DURATION_TICKS],minlength=N_E+N_I),dense.sum(0)),
                    'complete native/archive totals differ')
            for phase,(start,stop) in PHASES.items():
                selected=(ticks>=round(start/.05))&(ticks<round(stop/.05))
                count=np.bincount(ids[selected],minlength=N_E+N_I)
                require(np.array_equal(count,dense[round(start/5):round(stop/5)].sum(0)),'native/archive phase counts differ')
                key=f'{case_id}_{phase}';phase_counts[key]=count
                summary=summarize_counts(count,N_E,120.,30.,stop-start)
                summary['all_unit_rate_hz']=int(count.sum())/(N_E+N_I)/((stop-start)/1000)
                events=[int(unit) for tick,unit in zip(ticks,ids) if start/.05<=int(tick)<stop/.05]
                counted=Counter(events)
                require(all(int(count[i])==counted[i] for i in range(N_E+N_I)),'independent all-unit phase count mismatch')
                unit_checks+=N_E+N_I
                reference=scalar_readout(events,N_E,N_I,stop-start)
                scalar_checks+=_check_readout(summary,reference)
                stimulated=profile['cue_units'] if CONDITIONS[condition]>0 else 0
                info=dict(case_id=case_id,source_rng_seed=seed,condition=condition,phase=phase,
                    start_ms=start,stop_ms=stop,units=N_E+N_I,excitatory_neurons=N_E,inhibitory_neurons=N_I,
                    stimulus_strength_nA=CONDITIONS[condition],cue_fraction_e=profile['cue_fraction_e'],
                    effective_cue_center_deg=profile['effective_cue_center_deg'],
                    stimulated_e_units=stimulated,stimulated_e_fraction=stimulated/N_E)
                rows.append(dict(info,**summary));independent.append(dict(info,**reference))
        require(coverage==inputs['comparison']['array_values']==inputs['replay']['fresh_process_array_values_checked'],
                'full computed array coverage differs from required comparisons')
        effects=paired_effects(rows,SOURCE_SEEDS)
        scalar_index={(r['source_rng_seed'],r['condition'],r['phase']):r for r in independent}
        for row in effects:
            left,right=row['contrast'].split('_minus_')
            a=scalar_index[row['source_rng_seed'],left,row['phase']]
            b=scalar_index[row['source_rng_seed'],right,row['phase']]
            expected={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in row
                if k not in ('source_rng_seed','phase','contrast','decoded_angle_change_deg')}
            expected['decoded_angle_change_deg']=None if not (a['direction_defined'] and b['direction_defined']) else (
                a['decoded_angle_deg']-b['decoded_angle_deg']+180)%360-180
            scalar_checks+=_check_readout(row,expected)
        tables={'responses.csv':rows,'paired_effects.csv':effects}
        cross_info=None
        if reference_analysis is not None:
            tables['cross_size_responses.csv'],cross_info=descriptive_cross_size(reference_analysis,rows,configs[0])
        for name,table in tables.items():write_rows(output/name,table)
        np.savez_compressed(output/'phase_unit_counts.npz',**phase_counts)
        replay=dict(status='PASS',complete_episodes=9,units=N_E+N_I,source_rng_blocks=3,
            phase_unit_count_values_checked=unit_checks,scalar_readout_values_checked=scalar_checks,
            source_array_values_bound=coverage,rtol=READOUT_RTOL,atol=READOUT_ATOL,
            method='native raw-event Counter and scalar math.fsum checked against archive/vector readouts',
            native_report_sha256=sha256_file(paths['native']),
            artifacts={name:sha256_file(output/name) for name in (*tables,'phase_unit_counts.npz')},
            implementation_sha256=sha256_file(Path(__file__)),scientific_fidelity_qualified=False)
        write_json(output/'readout_replay.json',replay)
        result.update(status='DESCRIBED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC',units=N_E+N_I,
            source_rng_blocks=3,source_rng_seeds=list(SOURCE_SEEDS),complete_source_episodes=9,
            training_seeds=[41,42,43],configuration=configs[0],phases_ms=PHASES,
            source_revision=SOURCE_REVISION,source_entry_sha256=SOURCE_ENTRY_SHA256,
            native_report_sha256=sha256_file(paths['native']),source_comparison_sha256=sha256_file(comparison_path),
            fresh_replay_sha256=sha256_file(replay_path),design_sha256=sha256_file(design_path),
            computed_source_array_values=coverage,phase_unit_count_values_verified=unit_checks,
            scalar_readout_values_verified=scalar_checks,source_cue_geometry=profile,
            coordinate_semantics='fixed360-degree cue preference domain; not measured cortical positions or increased anatomy',
            source_rng_scope='three independent joint initial/background source blocks; not training seeds or biological replicates',
            interpretation='native phase readouts; phase gaps do not certify uninterrupted memory; 2560 is not100k qualification',
            cross_size_description=cross_info,readout_replay_sha256=sha256_file(output/'readout_replay.json'),
            artifacts=replay['artifacts'],implementation_sha256=sha256_file(Path(__file__)))
        write_json(output/'analysis.json',result)
        return result
    except BaseException as error:
        incomplete=isinstance(error,FileNotFoundError) or 'both comparison receipts must complete' in str(error)
        result.update(status='BLOCKED_INCOMPLETE_INPUTS' if incomplete else 'FAILED',
                      error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        write_json(output/'analysis.json',result)
        raise



_SOURCE_EXECUTION_CHECKS = {
    'full_raw_schema', 'full_terminal_schema', 'numeric_array_types', 'canonical_event_schema',
    'full_unit_output_shapes', 'finite_terminal_state', 'finite_raw_outputs', 'legal_unit_ids',
    'legal_spike_ticks', 'exact_spike_seconds', 'canonical_tick_then_unit_order',
    'full_unit_identity_mapping', 'terminal_voltage_matches_raw', 'complete_source_input_consumption',
    'full_event_count_equality', 'terminal_spike_count_equality', 'refractory_spacing_E',
    'refractory_spacing_I', 'terminal_lastspike_matches_events', 'terminal_refractory_state_matches_events',
}
_SOURCE_OUTPUTS = {'config.yaml', 'request.json', 'raw.npz', 'terminal_state.npz', 'units.npz',
                   'unit_metadata.json', 'initial_voltage.npy', 'final_rng.npz'}
_SOURCE_CURRENT_OUTPUTS = {'applied_current_observation.npz', 'applied_current_metadata.json'}
_SOURCE_STIMULUS_PERIOD_SECONDS = .001
# These two archived wrappers differ only in the explicit source-clock
# admission check and its call argument. At .05 ms both use the same qualified
# numerical engine/drive. Keep actual wrapper hashes on every source binding.
_SOURCE_CLOCK_ADMISSION_WRAPPERS = frozenset({
    'c0c4a09a7c2126b7391b13ee11fd6652180267d04ce3bf128ab4279ca2788f5f',
    '9685bc655eb13098060a74db3cd9b076baa01ab821614df8b69302374c30e5d8',
})
_SOURCE_CODE_FILES = {'runner_sha256': 'pfc_source_execution.py',
                      'executor_sha256': 'pfc_deterministic_replay.py', 'drive_sha256': 'pfc_native_drive.py'}
_SOURCE_KERNEL_CODE_FILES = dict(_SOURCE_CODE_FILES, kernel_parameterization_sha256='pfc_kernel_parameters.py')
# Preserved default-only executables, independently bound in archived deliveries.
# This archive compatibility is not permission to run a changed executable.
_LEGACY_DEFAULT_KERNEL_EXECUTORS = frozenset({
    'a7ec32dbf04b26424e9047a40621c469b50561541567679355a97eea2bbe34bc',
    'bce09608051ea67235865c07c8d4b0058c3d0cddf35f7bddaa72bf0b0804ec1a',
    '8e07afc152410d6c2c6197067e8738ffbd8c385b824012bdf84f68827f483410',
})
_SOURCE_RESPONSE_FIELDS = (
    'case_id', 'source_rng_seed', 'condition', 'phase', 'start_ms', 'stop_ms', 'units',
    'excitatory_neurons', 'inhibitory_neurons', 'stimulus_strength_nA', 'cue_fraction_e',
    'effective_cue_center_deg', 'stimulated_e_units', 'stimulated_e_fraction',
    'excitatory_spikes', 'inhibitory_spikes', 'excitatory_rate_hz', 'inhibitory_rate_hz',
    'cue_units', 'cue_region_rate_hz', 'noncue_region_rate_hz', 'cue_selectivity_hz',
    'vector_x_spikes', 'vector_y_spikes', 'resultant_length', 'decoded_angle_deg',
    'circular_cue_error_deg', 'direction_defined', 'all_unit_rate_hz')


def observed_source_registration(config_path):
    """Admit observed archives under their original registered source purpose.

    The numerical-calibration registrar stays restricted to its own protocol.
    The existing recurrent-gain registrar binds the exact declared development
    tuple, circuit size, conditions, seeds and configuration bytes.
    """
    config = yaml.safe_load(Path(config_path).read_text())
    if config.get("source_replication", {}).get("purpose") != "source_parameter_domain_development":
        return _source_registration(config_path)
    from .pfc_recurrent_gain_comparison import register_source_config
    protocol_path = Path(__file__).resolve().parents[1] / "configs/neural_population_pfc_recurrent_gain.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    size = config["excitatory_neurons"] + config["inhibitory_neurons"]
    label = {2560: "reference_2560", 100000: "expanded_100k"}.get(size)
    if label is None:
        raise ValueError("REGISTERED_OBSERVED_GAIN_CIRCUIT_SIZE_REQUIRED")
    return register_source_config(config_path, protocol["cohorts"][label], "candidate")


def _source_registration(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    expected = dict(scope='source_reproduction', expected_source_revision=SOURCE_REVISION,
                    synapse_execution='factorized_uniform_projections', duration_ms=5000.,
                    stimulus_center_deg=120., stimulus_width_deg=30., stimulus_start_ms=100.,
                    stimulus_duration_ms=200.)
    require(isinstance(config, dict) and all(config.get(k) == v for k, v in expected.items()),
            'registered five-phase source configuration required')
    n_e, n_i = config['excitatory_neurons'], config['inhibitory_neurons']
    kernel_parameters.resolve_kernel_parameters(config)
    require(type(n_e) is int and type(n_i) is int and n_e > 0 and n_i > 0,
            'positive complete source population sizes required')
    dt_ms, scale = config['dt_ms'], config['weight_scaling_factor']
    require(type(dt_ms) in (int, float) and np.isfinite(dt_ms) and dt_ms > 0
            and type(scale) in (int, float) and np.isfinite(scale) and scale > 0
            and np.isclose(n_e * scale, 2048., rtol=0, atol=1e-10)
            and np.isclose(n_i * scale, 512., rtol=0, atol=1e-10),
            'source clock and original size normalization required')
    for boundary in {v for interval in PHASES.values() for v in interval}:
        require(np.isclose(round(boundary / dt_ms) * dt_ms, boundary, rtol=0, atol=1e-9),
                'registered phase boundaries must lie on the source clock')
    cases = config.get('cases')
    replication = config.get('source_replication')
    if replication is None:
        block_count = 3
        declared_seeds = None
    else:
        require(isinstance(replication, dict)
                and replication.get('purpose') == 'numerical_calibration'
                and replication.get('independent_unit') == 'joint_initial_background_rng_block',
                'explicit numerical calibration replication required')
        declared_seeds = replication.get('joint_rng_seeds')
        require(isinstance(declared_seeds, list) and len(declared_seeds) >= 3
                and all(type(seed) is int and 0 <= seed < 2**32 for seed in declared_seeds)
                and declared_seeds == sorted(set(declared_seeds)),
                'ordered unique declared joint RNG blocks required')
        block_count = len(declared_seeds)
    case_count = block_count * len(CONDITIONS)
    require(isinstance(cases, list) and len(cases) == case_count
            and all(isinstance(c, dict) for c in cases),
            'canonical config must retain all nine source cases' if replication is None
            else 'all declared calibration blocks and controls required')
    seeds = sorted({row.get('source_rng_seed') for row in cases
                    if type(row.get('source_rng_seed')) is int})
    require(len(seeds) == block_count and all(0 <= seed < 2**32 for seed in seeds)
            and (declared_seeds is None or seeds == declared_seeds),
            'registered joint source RNG blocks differ from declaration')
    registered = {f'pfc_{seed}_{condition}': (seed, condition)
                  for seed in seeds for condition in CONDITIONS}
    require(len({row.get('case_id') for row in cases}) == case_count
            and {row.get('case_id') for row in cases} == set(registered),
            'complete unique source case identities required')
    for row in cases:
        seed, condition = registered[row['case_id']]
        require(row.get('source_rng_seed') == seed and row.get('stimulus_strength_nA') == CONDITIONS[condition]
                and row.get('distractor_strength_nA') == 0., 'source RNG identity or registered current changed')
    geometry(n_e)
    dimensions = dict(n_e=n_e, n_i=n_i, coupled_core_units=n_e+n_i, dt_seconds=dt_ms/1000,
                      duration_ticks=round(config['duration_ms']/dt_ms), duration_seconds=config['duration_ms']/1000,
                      weight_scaling_factor=scale, N_E_times_scale=n_e*scale, N_I_times_scale=n_i*scale)
    return config, registered, seeds, dimensions


def _source_manifest(delivery):
    rows = delivery.get('files')
    require(isinstance(rows, list), 'source delivery file inventory missing')
    result = {}
    for row in rows:
        name = row.get('path') if isinstance(row, dict) else None
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and '..' not in Path(name).parts and str(Path(name)) == name and name not in result,
                'source delivery contains an invalid or duplicate file identity')
        result[name] = row
    return result


def _source_bound_file(archive, manifest, name, expected=None):
    require(name in manifest, 'consumed source file absent from delivery: '+name)
    row = manifest[name]
    path = archive / name
    require(type(row.get('bytes')) is int and row['bytes'] >= 0
            and path.stat().st_size == row['bytes'] and sha256_file(path) == row.get('sha256'),
            'source delivery file size or hash mismatch: '+name)
    if expected is not None:
        require(expected.get('path') == Path(name).name and expected.get('sha256') == row['sha256']
                and expected.get('bytes') == row['bytes'], 'source report/output binding mismatch: '+name)
    return path


def _source_header(source_run, config, registered, config_sha):
    source_run = Path(source_run)
    source_run = source_run if source_run.name == 'source_run' else source_run / 'source_run'
    archive = source_run.parent
    status = dict(path=str(source_run), case_id='', status='MISSING_DELIVERY', report_status='',
                  report_sha256='', delivery_sha256='', error='')
    # A delivery is written only after archival. Never inspect a running producer's report or arrays.
    if not (archive/'delivery.json').is_file():
        return status, None
    delivery = read(archive/'delivery.json')
    require(delivery.get('status') == 'ARCHIVED_PFC_SOURCE_EXECUTION'
            and delivery.get('case_id') in registered, 'registered terminal source delivery required')
    status.update(case_id=delivery['case_id'], delivery_sha256=sha256_file(archive/'delivery.json'))
    manifest = _source_manifest(delivery)
    report_path = _source_bound_file(archive, manifest, 'source_run/report.json')
    report = read(report_path)
    terminal = {'PFC_SOURCE_EXECUTION_COMPLETE', 'PFC_SOURCE_EXECUTION_FAILED',
                'PFC_SOURCE_EXECUTION_VALIDATION_FAILED'}
    require(report.get('status') in terminal and report['status'] == delivery.get('result_status')
            and report.get('case_id') == status['case_id']
            and report.get('source_execution_complete') is delivery.get('source_execution_complete')
            and delivery.get('scientific_fidelity_qualified') is False and delivery.get('production_episodes') == 0
            and report.get('scientific_fidelity_qualified') is False and report.get('production_episodes') == 0,
            'source terminal status, identity or diagnostic scope mismatch')
    status.update(report_status=report['status'], report_sha256=manifest['source_run/report.json']['sha256'])
    case = next(row for row in config['cases'] if row['case_id'] == status['case_id'])
    if 'resolved_config' in report:
        require(report['resolved_config'] == config, 'source report differs from canonical configuration')
    if 'selected_case' in report:
        require(report['selected_case'] == case, 'selected source case identity changed')
    if 'config_sha256' in report:
        require(report['config_sha256'] == config_sha, 'source canonical configuration bytes changed')
    complete = report['status'] == 'PFC_SOURCE_EXECUTION_COMPLETE'
    require(report.get('source_execution_complete') is complete, 'source completion flag disagrees with terminal status')
    status['status'] = 'COMPLETE' if complete else 'FAILED_SOURCE_EXECUTION'
    return status, dict(archive=archive, source_run=source_run, manifest=manifest, report=report,
                        binding=dict({k:status[k] for k in ('path', 'case_id', 'report_sha256', 'delivery_sha256')},
                            source_runner_sha256=report.get('code', {}).get('runner_sha256')))


def _source_applied_current(paths, metadata, report, config, case, dimensions, terminal):
    """Check stored currents against source assignments without replacing observations."""
    declared = set(paths) & _SOURCE_CURRENT_OUTPUTS
    marked = (any(key.startswith('applied_current_') for key in metadata)
              or any(key.startswith('applied_current_') for key in report.get('validations', {}))
              or 'applied_current_capture_seconds' in report.get('costs', {}))
    if not declared:
        require(not marked, 'declared applied current observation is missing')
        return dict(status='UNAVAILABLE_LEGACY_ARCHIVE', verified=False)
    require(declared == _SOURCE_CURRENT_OUTPUTS
            and metadata.get('applied_current_observation_file') == 'applied_current_observation.npz'
            and metadata.get('applied_current_metadata_file') == 'applied_current_metadata.json',
            'applied current file links or inventory mismatch')
    n_e, dt, ticks = dimensions['n_e'], dimensions['dt_seconds'], dimensions['duration_ticks']
    stride = round(_SOURCE_STIMULUS_PERIOD_SECONDS / dt)
    current_metadata = read(paths['applied_current_metadata.json'])
    expected = dict(schema='pfc_applied_current_segments_v1', variable='I_stim', population='E', units='A',
        observed_unit_count=n_e, dt_seconds=dt, duration_ticks=ticks,
        stimulus_operation_stride_ticks=stride, timing='after_source_stimulus_operation_before_rk2',
        capture_rule='initial_source_operation_and_cue_or_distractor_active_state_transitions',
        source_indices='all_E_source_indices_ascending',
        archive_indices='E_population_archive_indices_in_E_then_I_unit_order',
        segment_bounds='start_inclusive_stop_exclusive',
        segment_interval_semantics='held by source current-update semantics; not repeated measurements',
        coverage='full E I_stim vectors at segment starts; excludes inhibitory, synaptic and total membrane currents')
    require(all(current_metadata.get(k) == value for k, value in expected.items()),
            'applied current units, identities or source-clock metadata mismatch')
    with np.load(paths['applied_current_observation.npz'], allow_pickle=False) as stored:
        arrays = {key:stored[key] for key in stored.files}
    dtypes = dict(source_unit_indices=np.int64, archive_unit_indices=np.int64,
        segment_start_ticks=np.int64, segment_stop_ticks=np.int64,
        cue_active=np.bool_, distractor_active=np.bool_, current_A=np.float64)
    require(set(arrays) == set(dtypes) and all(arrays[k].dtype == v for k,v in dtypes.items()),
            'applied current array schema or dtype mismatch')
    starts, stops, values = (arrays[k] for k in ('segment_start_ticks','segment_stop_ticks','current_A'))
    require(starts.ndim == 1 and len(starts) > 0 and stops.shape == starts.shape
            and arrays['cue_active'].shape == starts.shape and arrays['distractor_active'].shape == starts.shape
            and values.shape == (len(starts), n_e) and np.isfinite(values).all()
            and np.array_equal(arrays['source_unit_indices'], np.arange(n_e))
            and np.array_equal(arrays['archive_unit_indices'], np.arange(n_e)),
            'applied current full-E identity, shape or finite-value mismatch')
    program = []
    for prefix, default_center in (('stimulus',180.), ('distractor',90.)):
        start = config.get(prefix+'_start_ms', 0.) * 1e-3
        stop = start + config.get(prefix+'_duration_ms', 0.) * 1e-3
        center = round(n_e / 360. * config.get(prefix+'_center_deg', default_center))
        half = round(n_e / 360. * config.get(prefix+'_width_deg', 40.) / 2)
        indices = np.asarray([i % n_e for i in range(center-half, center+half+1)])
        amplitude = case.get(prefix+'_strength_nA', config.get(prefix+'_strength_nA', .07 if prefix=='stimulus' else 0.)) * 1e-9
        program.append((start, stop, indices, amplitude))
    expected_starts, states = [], []
    for tick in range(0, ticks, stride):
        # Preserve source floating-point operation times, including boundary rounding.
        time_seconds = (tick // stride) * _SOURCE_STIMULUS_PERIOD_SECONDS
        active = tuple(start <= time_seconds < stop for start,stop,_,_ in program)
        if not states or active != states[-1]:
            expected_starts.append(tick); states.append(active)
    require(np.array_equal(starts, expected_starts)
            and np.array_equal(stops, expected_starts[1:] + [ticks])
            and np.array_equal(arrays['cue_active'], [state[0] for state in states])
            and np.array_equal(arrays['distractor_active'], [state[1] for state in states]),
            'applied current source-clock segment coverage mismatch')
    expected_current = np.zeros(n_e, dtype=np.float64)
    for index, (cue_on, distractor_on) in enumerate(states):
        if cue_on:
            expected_current[program[0][2]] = program[0][3]
        else:
            expected_current.fill(0.)
        if distractor_on:
            expected_current[program[1][2]] = program[1][3]
        require(np.array_equal(values[index], expected_current),
                'applied current observed values differ from source control assignments')
    require(np.array_equal(values[-1], terminal['E__I_stim']),
            'applied current terminal state mismatch')
    return dict(status='VERIFIED_ACTUAL_CURRENT', verified=True, units='A', observed_E_units=n_e,
        segments=len(starts), observed_scalar_values=int(values.size),
        source_intervals=[dict(start_tick=int(start), stop_tick=int(stop),
            cue_active=bool(arrays['cue_active'][i]), distractor_active=bool(arrays['distractor_active'][i]),
            nonzero_E_units=int(np.count_nonzero(values[i])),
            minimum_current_A=float(values[i].min()), maximum_current_A=float(values[i].max()))
            for i,(start,stop) in enumerate(zip(starts,stops))],
        observation_sha256=sha256_file(paths['applied_current_observation.npz']),
        metadata_sha256=sha256_file(paths['applied_current_metadata.json']))


def _source_comparison_code(code, dt_ms):
    comparison = dict(code)
    if dt_ms == .05 and code['runner_sha256'] in _SOURCE_CLOCK_ADMISSION_WRAPPERS:
        # There is no single common wrapper hash. This declares the reviewed
        # comparison class; source_bindings retain which wrapper actually ran.
        comparison['runner_sha256'] = None
        comparison['reviewed_runner_compatibility'] = dict(
            dt_ms=.05, allowed_runner_sha256=sorted(_SOURCE_CLOCK_ADMISSION_WRAPPERS),
            difference='explicit source-clock admission only')
    return comparison


def _source_kernel_binding(config, code, receipt, report, request):
    kernel = kernel_parameters.resolve_kernel_parameters(config)
    provenance = kernel_parameters.circuit_provenance(config, SOURCE_ENTRY_SHA256)
    helper = code.get('kernel_parameterization_sha256')
    if helper is None:
        require(code.get('executor_sha256') in _LEGACY_DEFAULT_KERNEL_EXECUTORS
                and kernel == kernel_parameters.DEFAULT_KERNEL_PARAMETERS
                and 'validated_kernel_parameters' not in receipt
                and 'kernel_parameterization_sha256' not in receipt
                and all('resolved_kernel_parameters' not in row and 'circuit_provenance' not in row
                        for row in (report, request)),
                'legacy source archive supports only the verified default kernel')
        return dict(status='VERIFIED_LEGACY_DEFAULT_ONLY', resolved_kernel_parameters=kernel,
                    circuit_provenance=provenance)
    kernel_parameters.require_kernel_coverage(receipt, config, helper)
    require(all(row.get('resolved_kernel_parameters') == kernel
                and row.get('circuit_provenance') == provenance for row in (report, request)),
            'source kernel report/request identity mismatch')
    return dict(status='EXPLICIT_NATIVE_KERNEL_COVERAGE', resolved_kernel_parameters=kernel,
                circuit_provenance=provenance, kernel_parameterization_sha256=helper)


def _source_execution_metadata(header, config, registered, dimensions, config_sha):
    archive, manifest, report = header['archive'], header['manifest'], header['report']
    case_id = report['case_id']
    seed, condition = registered[case_id]
    case = next(row for row in config['cases'] if row['case_id'] == case_id)
    require(report.get('resolved_config') == config and report.get('selected_case') == case
            and report.get('config_sha256') == config_sha
            and report.get('registered_case_ids') == [row['case_id'] for row in config['cases']]
            and report.get('registered_case_count') == len(registered) and report.get('selected_case_count') == 1
            and report.get('completed_case_count') == 1
            and all(report.get('dimensions', {}).get(k) == v for k, v in dimensions.items())
            and report.get('source_entry_sha256') == SOURCE_ENTRY_SHA256
            and report.get('working_memory_qualified') is False
            and report.get('fidelity_at_requested_scale_qualified') is False and report.get('learned_simulator') is False,
            'complete source execution/configuration binding required')
    validations = report.get('validations', {})
    require(_SOURCE_EXECUTION_CHECKS <= set(validations)
            and all(row.get('passed') is True for row in validations.values()),
            'all source execution checks must pass')


def _source_values(header, config, registered, dimensions, config_sha):
    archive, manifest, report = header['archive'], header['manifest'], header['report']
    case_id = report['case_id']
    seed, condition = registered[case_id]
    case = next(row for row in config['cases'] if row['case_id'] == case_id)
    _source_execution_metadata(header,config,registered,dimensions,config_sha)
    outputs = report.get('outputs', {})
    require(set(outputs) in (_SOURCE_OUTPUTS, _SOURCE_OUTPUTS | _SOURCE_CURRENT_OUTPUTS),
            'complete source output inventory required')
    paths = {name:_source_bound_file(archive, manifest, 'source_run/'+name, outputs[name])
             for name in sorted(outputs)}
    require(sha256_file(paths['config.yaml']) == config_sha, 'archived canonical source config hash changed')
    request = read(paths['request.json'])
    require(request.get('status') == 'PFC_SOURCE_EXECUTION_REQUESTED'
            and all(request.get(k) == report[k] for k in ('case_id', 'selected_case', 'registered_case_ids',
                    'registered_case_count', 'config_sha256', 'backend', 'dimensions', 'code', 'backend_qualification')),
            'archived request and completed source execution differ')
    code = report.get('code', {})
    code_files = (_SOURCE_KERNEL_CODE_FILES if 'kernel_parameterization_sha256' in code else _SOURCE_CODE_FILES)
    require(set(code) == set(code_files), 'source producer code identities missing')
    for key, name in code_files.items():
        path = _source_bound_file(archive, manifest, 'Libs/population_dynamics/'+name)
        require(sha256_file(path) == code[key], 'archived source producer code changed: '+name)
    qualification = report['backend_qualification']
    receipt_path = _source_bound_file(archive, manifest, 'qualification/comparison/qualification.json')
    delivery_path = _source_bound_file(archive, manifest, 'qualification/delivery.json')
    receipt, delivery = read(receipt_path), read(delivery_path)
    expected = dict(status='PASS_NATIVE_STREAM_SOURCE_EXECUTION', source_entry_sha256=SOURCE_ENTRY_SHA256,
                    backend=report['backend'], executor_sha256=code['executor_sha256'], drive_sha256=code['drive_sha256'],
                    input_replay_status='PASS_NATIVE_NUMPY_DRIVE_REPLAY', input_replay_passed_comparisons=24,
                    global_cases_passed=2, global_arrays_passed=10, scientific_fidelity_qualified=False,
                    full_scale_qualified=False, production_episodes=0)
    require(qualification.get('receipt_file') == 'comparison/qualification.json'
            and qualification.get('receipt_sha256') == sha256_file(receipt_path)
            and qualification.get('delivery_sha256') == sha256_file(delivery_path)
            and qualification.get('qualification') == receipt
            and all(receipt.get(k) == v for k, v in expected.items())
            and delivery.get('status') == 'ARCHIVED_PFC_STREAM_VALIDATION'
            and delivery.get('result_status') == 'PASS_NATIVE_STREAM_SOURCE_EXECUTION',
            'archived backend qualification does not bind this source executor')
    clocks = receipt.get('validated_dt_ms')
    if clocks is None and receipt.get('global_reference_report_sha256') == (
            '99f1f6e8f93499dc0fa6310ade1ecc6cb6364514050542d9e6034d8b69b9814a'):
        clocks = [0.05]
    require(isinstance(clocks, list) and len(clocks) == 1
            and type(clocks[0]) in (int, float) and np.isfinite(clocks[0]) and clocks[0] > 0
            and config['dt_ms'] == clocks[0], 'source clock not covered by native qualification')
    kernel_binding = _source_kernel_binding(config, code, receipt, report, request)
    header['binding']['kernel_parameters'] = kernel_binding
    _source_bound_file(archive/'qualification', _source_manifest(delivery), 'comparison/qualification.json')
    n_e, n_i, total = dimensions['n_e'], dimensions['n_i'], dimensions['coupled_core_units']
    dt, duration_ticks = dimensions['dt_seconds'], dimensions['duration_ticks']
    driver = report.get('driver', {})
    policy = config.get('random_stream_policy', 'source_global')
    if policy == 'source_global':
        require(not any(name in item for item in (config, case)
                        for name in ('initial_rng_seed', 'external_noise_seed')),
                'global source archive cannot carry independent stream aliases')
        seeds = {'source_rng_seed': seed}
    elif policy == 'independent_initial_external':
        from .pfc_input_pairing import _resolve_cases

        resolved = next(row for row in _resolve_cases(config) if row['case_id'] == case_id)
        require(resolved['external_noise_seed'] == seed, 'registered external source seed differs')
        seeds = {name: resolved[name] for name in ('initial_rng_seed', 'external_noise_seed')}
    else:
        raise ValueError('unsupported source archive random stream policy')
    expected_driver = dict(n_e=n_e, n_i=n_i, dt_seconds=dt, total_ticks=duration_ticks,
        ticks_generated=duration_ticks, seeds=seeds, random_stream_policy=policy,
        source_entry_sha256=SOURCE_ENTRY_SHA256, rng='numpy.random.RandomState(MT19937)',
        initial_draw_order='I_then_E', initial_voltage_archive_order='E_then_I',
        packed_draw_order='tick_major_I_then_E', binomial_n=1000, binomial_probability=1.4*dt,
        external_rate_Hz=1.4, returned_dtype='uint16')
    require(report.get('seeds') == seeds and report.get('random_stream_policy') == policy
            and all(driver.get(k) == v for k, v in expected_driver.items())
            and driver.get('numpy_version') == receipt.get('numpy_version')
            and report.get('completed_duration_ticks') == duration_ticks,
            'source RNG block, input stream or clock mismatch')
    used, totals = report.get('inputs_used', {}), report.get('source_external_event_totals', {})
    require(used.get('input_mode') == 'native_numpy_stream' and used.get('ticks_consumed') == duration_ticks
            and driver.get('external_event_totals') == totals
            and all(used.get(pop, {}).get('values') == size*duration_ticks
                    and type(totals.get(pop)) is int and totals[pop] >= 0
                    and used.get(pop, {}).get('events') == totals[pop] for pop, size in (('E', n_e), ('I', n_i))),
            'complete source input consumption mismatch')
    loaded = {}
    for name in ('raw.npz', 'terminal_state.npz', 'units.npz', 'final_rng.npz'):
        with np.load(paths[name], allow_pickle=False) as stored:
            loaded[name] = {key:stored[key] for key in stored.files}
    raw, terminal, units = (loaded[name] for name in ('raw.npz', 'terminal_state.npz', 'units.npz'))
    require(set(raw) == RAW_COLUMNS, 'complete source raw schema required')
    ids, ticks, seconds = (raw[name] for name in ('spike_unit_indices', 'spike_ticks', 'spike_seconds'))
    require(ids.ndim == ticks.ndim == seconds.ndim == 1 and ids.shape == ticks.shape == seconds.shape
            and ids.dtype == ticks.dtype == np.int64 and seconds.dtype == np.float64
            and len(ids) == report.get('neural_event_count') and ((ids >= 0) & (ids < total)).all()
            and ((ticks >= 0) & (ticks < duration_ticks)).all()
            and np.array_equal(seconds, ticks*dt)
            and not np.any(ticks[1:] < ticks[:-1])
            and not np.any((ticks[1:] == ticks[:-1]) & (ids[1:] < ids[:-1])),
            'source event identity, order or clock mismatch')
    counts = np.bincount(ids, minlength=total)
    require(raw['native_spike_counts'].dtype == np.int64 and raw['native_spike_counts'].shape == (total,)
            and np.array_equal(counts, raw['native_spike_counts'])
            and raw['terminal_voltage_mV'].shape == (total,) and np.isfinite(raw['terminal_voltage_mV']).all(),
            'source full-unit event counts or voltage mismatch')
    state_fields = {'E': ('v','s_AMPA','s_GABA','s_NMDA_total','s_NMDA','x','I_stim','lastspike','not_refractory'),
                    'I': ('v','s_AMPA','s_GABA','s_NMDA_total','lastspike','not_refractory')}
    require(set(terminal) == {pop+'__'+name for pop in ('E','I')
            for name in (*state_fields[pop], 'source_indices','archive_indices','spike_counts')},
            'source terminal all-unit schema mismatch')
    for pop, size, offset in (('E', n_e, 0), ('I', n_i, n_e)):
        require(all(value.shape == (size,) and value.dtype.kind in 'biuf' and np.isfinite(value).all()
                    for name, value in terminal.items() if name.startswith(pop+'__'))
                and np.array_equal(terminal[pop+'__source_indices'], np.arange(size))
                and np.array_equal(terminal[pop+'__archive_indices'], np.arange(size)+offset)
                and np.array_equal(terminal[pop+'__spike_counts'], counts[offset:offset+size]),
                'source terminal full-unit identity/count mapping mismatch')
    require(np.array_equal(raw['terminal_voltage_mV'], np.r_[terminal['E__v'], terminal['I__v']]/.001),
            'source terminal voltage units mismatch')
    expected_units = dict(unit_ids=np.arange(total, dtype=np.int64),
        source_unit_indices=np.r_[np.arange(n_e, dtype=np.int64), np.arange(n_i, dtype=np.int64)],
        population_index=np.r_[np.zeros(n_e, dtype=np.uint8), np.ones(n_i, dtype=np.uint8)],
        excitatory_preference_unit_ids=np.arange(n_e, dtype=np.int64),
        excitatory_preferred_angle_deg=360.*np.arange(n_e, dtype=np.float64)/n_e)
    require(set(units) == set(expected_units) and all(units[k].dtype == v.dtype
            and np.array_equal(units[k], v) for k,v in expected_units.items()),
            'source full-unit identity or preference mapping mismatch')
    metadata = read(paths['unit_metadata.json'])
    require(metadata.get('unit_file') == 'units.npz' and metadata.get('unit_count') == total
            and metadata.get('population_index_labels') == {'0':'E','1':'I'}
            and metadata.get('inhibitory_preference_coordinate') is None
            and metadata.get('anatomical_soma_coordinates_provided') is False
            and metadata.get('initial_voltage_file') == 'initial_voltage.npy'
            and metadata.get('initial_voltage_units') == 'mV'
            and metadata.get('terminal_state_units') == dict(v='V', I_stim='A', lastspike='s',
                   synaptic_state_variables='dimensionless', not_refractory='boolean')
            and metadata.get('coupling', {}).get('normalization') == config['weight_scaling_factor']
            and metadata.get('coupling', {}).get('representation') == 'implicit source rules; no independent network tiling',
            'source unit/observation metadata mismatch')
    if 'kernel_parameterization_sha256' in code:
        require(metadata.get('coupling', {}).get('resolved_kernel_parameters') == kernel_binding['resolved_kernel_parameters']
                and metadata.get('coupling', {}).get('circuit_provenance') == kernel_binding['circuit_provenance'],
                'source unit metadata kernel identity mismatch')
    header['binding']['applied_current_observation'] = _source_applied_current(
        paths, metadata, report, config, case, dimensions, terminal)
    initial = np.load(paths['initial_voltage.npy'], allow_pickle=False)
    require(initial.shape == (total,) and initial.dtype == np.float64 and np.isfinite(initial).all()
            and loaded['final_rng.npz'], 'complete source initial state and final RNG required')
    require(all(value.dtype.kind in 'biufUS' for value in loaded['final_rng.npz'].values()),
            'source final RNG contains unsupported arrays')
    common = dict(code=_source_comparison_code(code, config['dt_ms']), backend=report['backend'],
        qualification_sha256=qualification['receipt_sha256'],
        qualification_delivery_sha256=qualification['delivery_sha256'],
        driver={k:v for k,v in driver.items() if k not in ('seeds','external_event_totals')})
    prefix = ticks < round(config['stimulus_start_ms']/config['dt_ms'])
    paired = dict(prefix_ticks=ticks[prefix], prefix_ids=ids[prefix], initial=initial,
                  final_rng=loaded['final_rng.npz'], inputs_used=used)
    return raw, common, paired


def _source_phase_rows(raw, case_id, registered, config, dimensions):
    n_e, n_i, total = dimensions['n_e'], dimensions['n_i'], dimensions['coupled_core_units']
    seed, condition = registered[case_id]
    ticks, ids = raw['spike_ticks'], raw['spike_unit_indices']
    intervals = [(name, round(start/config['dt_ms']), round(stop/config['dt_ms']))
                 for name,(start,stop) in PHASES.items()]
    starts = [start for _,start,_ in intervals]
    independent = {phase:Counter() for phase in PHASES}
    # Separate scalar event routing and counting from the NumPy masks/bincount path.
    for tick, unit in zip(ticks, ids):
        tick = int(tick)
        index = bisect_right(starts, tick)-1
        if index >= 0 and tick < intervals[index][2]:
            independent[intervals[index][0]][int(unit)] += 1
    rows, scalar_rows, vectors = [], [], {}
    scalar_checks = 0
    _, profile = geometry(n_e)
    for phase,(start,stop) in PHASES.items():
        selected = (ticks >= round(start/config['dt_ms'])) & (ticks < round(stop/config['dt_ms']))
        counts = np.bincount(ids[selected], minlength=total)
        require(all(int(counts[unit]) == independent[phase][unit] for unit in range(total)),
                'independent all-unit source phase counts differ')
        vectors[case_id+'_'+phase] = counts
        summary = summarize_counts(counts, n_e, 120., 30., stop-start)
        summary['all_unit_rate_hz'] = int(counts.sum())/total/((stop-start)/1000)
        scalar = scalar_readout(independent[phase].elements(), n_e, n_i, stop-start)
        scalar_checks += _check_readout(summary, scalar)
        stimulated = profile['cue_units'] if CONDITIONS[condition] > 0 else 0
        info = dict(case_id=case_id, source_rng_seed=seed, condition=condition, phase=phase,
            start_ms=start, stop_ms=stop, units=total, excitatory_neurons=n_e, inhibitory_neurons=n_i,
            stimulus_strength_nA=CONDITIONS[condition], cue_fraction_e=profile['cue_fraction_e'],
            effective_cue_center_deg=profile['effective_cue_center_deg'], stimulated_e_units=stimulated,
            stimulated_e_fraction=stimulated/n_e)
        rows.append(dict(info, **summary)); scalar_rows.append(dict(info, **scalar))
    return rows, scalar_rows, vectors, scalar_checks


def _source_pairing(paired, registered, seeds):
    checks = []
    for seed in seeds:
        for left,right in (('cue','omission'), ('strong_cue','omission'), ('strong_cue','cue')):
            a,b = paired[f'pfc_{seed}_{left}'], paired[f'pfc_{seed}_{right}']
            checks.append(dict(source_rng_seed=seed, left=left, right=right,
                exact_pre_cue_events=np.array_equal(a['prefix_ticks'], b['prefix_ticks'])
                    and np.array_equal(a['prefix_ids'], b['prefix_ids']),
                left_prefix_events=len(a['prefix_ticks']), right_prefix_events=len(b['prefix_ticks']),
                exact_initial_voltage=np.array_equal(a['initial'], b['initial']),
                exact_final_source_rng=set(a['final_rng']) == set(b['final_rng']) and all(
                    np.array_equal(a['final_rng'][k], b['final_rng'][k]) for k in a['final_rng']),
                same_source_input_consumption=a['inputs_used'] == b['inputs_used']))
    return checks


def _source_coverage(output, registered, statuses, described):
    rows = []
    for case_id,(seed,condition) in registered.items():
        attempts = [row for row in statuses if row['case_id'] == case_id]
        complete = sum(row['status'] in ('COMPLETE','DESCRIBED_COMPLETE') for row in attempts)
        failed = sum(row['status'] == 'FAILED_SOURCE_EXECUTION' for row in attempts)
        rejected = any(row['status'] == 'REJECTED' for row in attempts)
        state = ('DUPLICATE_COMPLETE' if complete > 1 else 'REJECTED' if rejected else
                 'DESCRIBED_COMPLETE' if case_id in described else 'AVAILABLE_COMPLETE' if complete else
                 'FAILED_SOURCE_EXECUTION' if failed else 'MISSING')
        rows.append(dict(case_id=case_id, source_rng_seed=seed, condition=condition, status=state,
                         input_attempts=len(attempts), completed_inputs=complete, failed_inputs=failed,
                         described=case_id in described, paths=json.dumps([row['path'] for row in attempts])))
    with (output/'case_coverage.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fields=('path','case_id','status','report_status','report_sha256','delivery_sha256','error')
    with (output/'input_status.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(statuses)
    return rows


def analyze_source_runs(source_runs, config_path, output):
    """Describe completed streamed source archives against the full registered denominator.

    This is a readout replay, not a second source simulation or large-scale fidelity
    qualification. Missing deliveries are recorded without reading unfinished runs.
    """
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = dict(status='CHECKING_SOURCE_INPUTS', full_scale_qualified=False, scientific_fidelity_qualified=False,
        working_memory_qualified=False, neuroscientific_claim_qualified=False, production_episodes=0,
        formal_training_completed=False, confidence_intervals=False, p_values=False, q_values=False,
        paired_effects_available=False, complete_source_episodes=0, described_source_cases=0)
    statuses, described, registered = [], set(), {}
    try:
        config, registered, seeds, dimensions = _source_registration(config_path)
        config_sha = sha256_file(config_path)
        result.update(registered_source_cases=len(registered), registered_source_rng_blocks=len(seeds),
                      source_rng_seeds=seeds, configuration=config, config_sha256=config_sha,
                      units=dimensions['coupled_core_units'], phases_ms=PHASES)
        headers = {}
        for source_run in source_runs:
            status = dict(path=str(source_run), case_id='', status='REJECTED', report_status='',
                          report_sha256='', delivery_sha256='', error='')
            try:
                status, header = _source_header(source_run, config, registered, config_sha)
                statuses.append(status)
                if header is not None and status['status'] == 'COMPLETE':
                    headers.setdefault(status['case_id'], []).append((header, status))
            except (ValueError, OSError, KeyError, TypeError) as error:
                status['error'] = str(error); status['status'] = 'REJECTED'; statuses.append(status)
        coverage = _source_coverage(output, registered, statuses, described)
        require(not any(row['status'] == 'REJECTED' for row in statuses),
                'source input provenance rejected; see input_status.csv')
        require(not any(row['status'] == 'DUPLICATE_COMPLETE' for row in coverage),
                'duplicate completed source cases must not be selected or counted twice')
        rows, scalar_rows, vectors, paired, bindings = [], [], {}, {}, []
        scalar_checks = 0
        common = None
        for case_id in registered:
            if case_id not in headers:
                continue
            header, status = headers[case_id][0]
            try:
                raw, binding, paired[case_id] = _source_values(header, config, registered, dimensions, config_sha)
                if common is None:
                    common = binding
                require(binding == common, 'source runs differ in code, backend qualification or input clock')
                current, scalar, counts, checks = _source_phase_rows(raw, case_id, registered, config, dimensions)
            except Exception as error:
                status['status'] = 'REJECTED'; status['error'] = str(error)
                raise
            rows.extend(current); scalar_rows.extend(scalar); vectors.update(counts); scalar_checks += checks
            described.add(case_id); status['status'] = 'DESCRIBED_COMPLETE'; bindings.append(header['binding'])
        complete = len(described) == len(registered)
        tables = ['responses.csv', 'phase_unit_counts.npz', 'case_coverage.csv', 'input_status.csv']
        with (output/'responses.csv').open('x', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=_SOURCE_RESPONSE_FIELDS); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(output/'phase_unit_counts.npz', **vectors)
        if complete:
            checks = _source_pairing(paired, registered, seeds)
            write_rows(output/'source_pairing_checks.csv', checks)
            tables.append('source_pairing_checks.csv')
            require(all(all(row[key] is True for key in ('exact_pre_cue_events', 'exact_initial_voltage',
                        'exact_final_source_rng', 'same_source_input_consumption')) for row in checks),
                    'complete source cases lack equal pre-cue events or matched source RNG blocks')
            effects = paired_effects(rows, seeds)
            independent_effects = paired_effects(scalar_rows, seeds)
            for row, reference in zip(effects, independent_effects):
                scalar_checks += _check_readout(row, {k:v for k,v in reference.items()
                    if k not in ('source_rng_seed','phase','contrast')})
            write_rows(output/'paired_effects.csv', effects); tables.append('paired_effects.csv')
            result.update(paired_effects_available=True, paired_contrasts=len(effects),
                          source_prefix_and_rng_pairs_verified=len(checks))
        _source_coverage(output, registered, statuses, described)
        artifacts = {name:sha256_file(output/name) for name in tables}
        count_values = len(vectors)*dimensions['coupled_core_units']
        replay = dict(status='PASS' if described else 'NOT_RUN_NO_COMPLETE_CASES',
            complete_episodes=len(described), registered_source_cases=len(registered),
            units=dimensions['coupled_core_units'], phase_unit_count_values_checked=count_values,
            scalar_readout_values_checked=scalar_checks, config_sha256=config_sha, source_bindings=bindings,
            method='all raw events independently routed to phases with Counter; scalar math.fsum versus NumPy readouts',
            rtol=READOUT_RTOL, atol=READOUT_ATOL, artifacts=artifacts,
            implementation_sha256=sha256_file(Path(__file__)), scientific_fidelity_qualified=False,
            replay_runs_count_as_additional_samples=False)
        write_json(output/'readout_replay.json', replay)
        result.update(status='DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC' if complete else
                          'PARTIAL_SOURCE_EXECUTION_DIAGNOSTIC' if described else 'BLOCKED_INCOMPLETE_INPUTS',
            complete_source_episodes=len(described), described_source_cases=len(described),
            missing_source_cases=[case_id for case_id in registered if case_id not in described],
            failed_source_attempts=sum(row['status']=='FAILED_SOURCE_EXECUTION' for row in statuses),
            unresolved_input_paths=[row['path'] for row in statuses if row['status']=='MISSING_DELIVERY'],
            phase_unit_count_values_verified=count_values, scalar_readout_values_verified=scalar_checks,
            source_bindings=bindings, common_execution_binding=common, source_revision=SOURCE_REVISION,
            source_entry_sha256=SOURCE_ENTRY_SHA256, source_cue_geometry=geometry(dimensions['n_e'])[1],
            source_rng_scope='joint initial/background source blocks; not training seeds or biological replicates',
            interpretation='source-backend phase descriptions; phase gaps do not establish uninterrupted memory; no large-scale fidelity qualification',
            coordinate_semantics='360 * source E index / N_E degrees; I has no preference angle or soma coordinates',
            readout_replay_sha256=sha256_file(output/'readout_replay.json'), artifacts=artifacts,
            implementation_sha256=sha256_file(Path(__file__)))
        write_json(output/'analysis.json', result)
        return result
    except Exception as error:
        if registered:
            _source_coverage(output, registered, statuses, described)
        result.update(status='FAILED', error=str(error), error_type=type(error).__name__,
                      described_source_cases=len(described), complete_source_episodes=len(described),
                      traceback=traceback.format_exc())
        write_json(output/'analysis.json', result)
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    legacy=('native','factorized','factorized-replay','source-comparison','fresh-replay','design',
            'native-config','factorized-config')
    for name in legacy:
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-analysis',type=Path)
    parser.add_argument('--config',type=Path,help='Canonical registered-cohort config for streamed source-run readouts')
    parser.add_argument('--require-complete',action='store_true',
                        help='Return failure for partial registered source cohorts while preserving their diagnostic artifacts')
    parser.add_argument('--source-run',type=Path,action='append',default=[],
                        help='Completed delivery root or source_run directory; repeat for each archived attempt')
    a=parser.parse_args(argv)
    if a.require_complete and a.config is None:
        parser.error('--require-complete requires --config for a registered source cohort')
    if a.config is not None or a.source_run:
        if a.config is None:
            parser.error('--source-run requires --config to preserve the registered denominator')
        if any(getattr(a,name.replace('-','_')) is not None for name in legacy) or a.reference_analysis is not None:
            parser.error('streamed source-run inputs cannot be combined with historical native/factorized inputs')
        result=analyze_source_runs(a.source_run,a.config,a.output)
    else:
        missing=[name for name in legacy if getattr(a,name.replace('-','_')) is None]
        if missing:
            parser.error('historical analysis requires '+', '.join('--'+name for name in missing))
        result=analyze(a.native,a.factorized,a.factorized_replay,a.source_comparison,a.fresh_replay,a.design,
                       a.native_config,a.factorized_config,a.output,reference_analysis=a.reference_analysis)
    print(json.dumps(result,indent=2,allow_nan=False))
    incomplete = result['status']=='BLOCKED_INCOMPLETE_INPUTS'
    if a.require_complete:
        incomplete = (result['status']!='DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC'
                      or result['complete_source_episodes']!=result['registered_source_cases']
                      or bool(result['missing_source_cases']) or bool(result['unresolved_input_paths']))
    return 1 if incomplete else 0


if __name__=='__main__':
    raise SystemExit(main())

"""Describe replayed native PFC delay responses without a working-memory claim.

The 5 s episodes extend the same three joint source RNG blocks. Their entire
500 ms spike prefixes must match the archived weak/omission/strong episodes.
"""
from __future__ import annotations

import argparse
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
from . import pfc_reference_replay
from .pfc_response_analysis import summarize_counts
from .provenance import canonical_json, sha256_file

SEEDS = (91001, 91002, 91003)
CONDITIONS = {'cue': .06, 'omission': 0., 'strong_cue': .5}
PHASES = {'baseline': [0.,100.], 'cue': [100.,300.],
          'early_delay': [300.,500.], 'middle_delay': [1000.,2000.],
          'late_delay': [4000.,5000.]}
CIRCUIT = 'pfc_source_reference'
SOURCE_REVISION = 'd7c4f07fa83a052829e40790fd1594955fe1c067'
SOURCE_ENTRY_SHA256 = '416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720'
READOUT_RTOL, READOUT_ATOL = 1e-10, 1e-9
PARAMETERS = ('excitatory_neurons','inhibitory_neurons','weight_scaling_factor','dt_ms',
    'stimulus_center_deg','stimulus_width_deg','stimulus_start_ms','stimulus_duration_ms',
    'distractor_center_deg','distractor_width_deg','distractor_start_ms','distractor_duration_ms',
    'state_monitored_per_population','excitatory_preference_bins','readout_bin_ms')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_rows(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance(config, source_hash):
    intrinsic={k:config[k] for k in ('excitatory_neurons','inhibitory_neurons','weight_scaling_factor','dt_ms')}
    topology=dict(excitatory_kernel='source_circulant_gaussian', inhibitory_self_connections=False,
        ei_and_ie='source_all_to_all', neurons=[1024,256], source_entry_sha256=source_hash)
    digest=lambda value:hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return dict(parameter_identity=digest(dict(intrinsic,source_entry_sha256=source_hash)),
                connectivity_identity=digest(topology))


def registered_cases(report, conditions):
    expected={f'pfc_{seed}_{condition}':(seed,condition) for seed in SEEDS for condition in conditions}
    rows=report['cases']
    require(len(rows)==len(expected) and {r['case_id'] for r in rows}==set(expected),
            'complete registered source case set required')
    for row in rows:
        seed,condition=expected[row['case_id']]
        require(row['source_rng_seed']==seed and row['stimulus_strength_nA']==CONDITIONS[condition]
                and row['distractor_strength_nA']==0. and row['core_units']==1280,
                'source condition, control strength or joint RNG identity changed')
    return {r['case_id']:r for r in rows},expected


def check_design(design, config, config_path):
    require(design['status']=='REGISTERED_BEFORE_EXECUTION' and design['source_rng_blocks']==3
            and design['source_rng_seeds']==list(SEEDS) and design['source_episodes']==9
            and design['duration_ms']==5000. and design['units']==1280
            and design['cue_strengths_nA']==[0.,.06,.5] and design['cue_window_ms']==[100.,300.]
            and design['prefix_reference_ms']==500. and design['phases_ms']==PHASES
            and design['config_sha256']==sha256_file(config_path),
            'frozen source delay design or config changed')
    expected=dict(scope='source_reproduction',expected_source_revision=SOURCE_REVISION,
        excitatory_neurons=1024,inhibitory_neurons=256,weight_scaling_factor=2.,dt_ms=.05,
        duration_ms=5000.,stimulus_center_deg=120.,stimulus_width_deg=30.,
        stimulus_start_ms=100.,stimulus_duration_ms=200.,readout_bin_ms=5.,
        excitatory_preference_bins=16)
    require(all(config[k]==v for k,v in expected.items()),'registered source parameters, clock or timing changed')
    require(design['source_fidelity_qualified'] is False and design['scientific_claim_qualified'] is False
            and design['production_episodes']==0,'source diagnostic scope changed')


def check_source(report, config, duration):
    require(report['status']=='NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS'
            and report['scope']=='source_reproduction' and report['source_revision']==SOURCE_REVISION
            and report['source_entry_sha256']==SOURCE_ENTRY_SHA256
            and report['resolved_config']['duration_ms']==duration
            and report['independent_source_rng_blocks']==3
            and report['full_scale_qualified'] is False and report['scientific_fidelity_qualified'] is False
            and all(report['resolved_config'][k]==config[k] for k in PARAMETERS),
            'complete matched native source parameters required')


def raw_events(root, case, duration_ticks):
    relative=Path(case['raw_file'])
    require(relative.name==case['raw_file'], 'native raw path must be local to its episode root')
    path=root/relative
    require(sha256_file(path)==case['raw_sha256'],'native raw artifact changed')
    with np.load(path,allow_pickle=False) as saved:
        arrays={k:saved[k] for k in saved.files}
    require(set(arrays)=={'spike_ticks','spike_unit_indices','spike_seconds','native_spike_counts','terminal_voltage_mV'},
            'complete native event/count/terminal schema required')
    ticks,ids=arrays['spike_ticks'],arrays['spike_unit_indices']
    require(ticks.ndim==ids.ndim==1 and ticks.shape==ids.shape and ticks.dtype.kind in 'iu'
            and ids.dtype.kind in 'iu' and (ticks>=0).all() and (ticks<=duration_ticks).all()
            and (ids>=0).all() and (ids<1280).all() and (np.diff(ticks)>=0).all()
            and len(ticks)==case['native_events'],'native event clock or unit support changed')
    seconds=arrays['spike_seconds']
    require(seconds.shape==ticks.shape and np.isfinite(seconds).all()
            and np.allclose(seconds*1000/.05,ticks,rtol=0,atol=1e-8),
            'native seconds and ticks differ')
    require(arrays['native_spike_counts'].shape==(1280,)
            and arrays['native_spike_counts'].dtype.kind in 'iu'
            and np.array_equal(np.bincount(ids,minlength=1280),arrays['native_spike_counts'])
            and arrays['terminal_voltage_mV'].shape==(1280,)
            and np.isfinite(arrays['terminal_voltage_mV']).all(),'all-unit native counts or state invalid')
    return arrays


def unit_identity(dataset, config, report):
    require(set(dataset.catalog['circuits'])=={CIRCUIT},'one complete source circuit required')
    circuit=dataset.catalog['circuits'][CIRCUIT]
    expected=provenance(config,report['source_entry_sha256'])
    require(circuit['units']==1280 and circuit['provenance']==expected,'source circuit parameter/topology identity changed')
    with h5py.File(dataset._verified(circuit['path'],circuit['sha256']),'r') as stored:
        require(np.array_equal(stored['unit_ids'][:],np.arange(1280))
                and np.array_equal(stored['populations'].asstr()[:],np.array(['E']*1024+['I']*256))
                and np.array_equal(stored['field_index'][:],np.r_[np.arange(1024)*16//1024,np.full(256,16)])
                and tuple(stored['field_names'].asstr()[:])==tuple(
                    [f'E_preference_{i*360/16:g}_{(i+1)*360/16:g}_deg' for i in range(16)]+['I_population'])
                and stored.attrs['source_revision']==SOURCE_REVISION
                and stored.attrs['coordinate_semantics']=='E angle=360*source_index/N_E degrees; I has no preference coordinate'
                and stored.attrs['coordinate_units']=='not_available' and 'coordinates' not in stored
                and json.loads(stored.attrs['circuit_provenance_json'])==expected,
                'complete source unit/population/preference identity changed')
    return expected


def archive_counts(root, report, config, cases, native):
    dataset=CircuitDataset(root/'dataset',splits=('development',))
    rows=[row for shard in dataset.catalog['shards'] for row in shard['episodes']]
    require(len(rows)==9 and dataset.catalog['complete_episodes']==9
            and len(dataset.episodes)==9 and {r['episode_id'] for r in rows}==set(cases)
            and all(r['split']=='development' for r in rows),'complete development-only archive required')
    binding=dataset.catalog['binding']
    require(binding['schema']==SCHEMA and binding['scope']=='source_reproduction'
            and binding['scenario_id']=='pfc_working_memory' and binding['source_revision']==SOURCE_REVISION
            and binding['source_entry_sha256']==report['source_entry_sha256']
            and binding['resolved_source_config']==config,'archive source/config binding changed')
    identity=unit_identity(dataset,config,report)
    controls=np.zeros((1000,2)); result={}
    for case_id,(seed,condition) in cases.items():
        row=dataset.episodes[case_id]
        require(row['duration_ticks']==100000 and row['dt_ms']==.05
                and row['circuit_id']==CIRCUIT and row['block_id']==f'pfc_source_block_{seed}',
                'episode timing, source RNG block or circuit identity changed')
        shard=dataset.shards[row['shard']]
        with h5py.File(dataset._verified(shard['path'],shard['sha256']),'r') as stored:
            group=stored[case_id];metadata=json.loads(group.attrs['metadata_json'])
            control_ticks=np.rint(np.array(sorted({0.,config['stimulus_start_ms'],
                config['stimulus_start_ms']+config['stimulus_duration_ms'],config['distractor_start_ms'],
                config['distractor_start_ms']+config['distractor_duration_ms']}))/.05).astype(np.int64)
            control_values=np.zeros((len(control_ticks),2))
            control_values[(control_ticks>=2000)&(control_ticks<6000),0]=CONDITIONS[condition]
            require(np.array_equal(group['spike_ticks'][:],native[case_id]['spike_ticks'])
                    and np.array_equal(group['spike_unit_indices'][:],native[case_id]['spike_unit_indices'])
                    and np.array_equal(group['control_ticks'][:],control_ticks)
                    and np.array_equal(group['control_values'][:],control_values),
                    'archive native events or exact control change points differ')
            require(all(metadata[k]==row[k] for k in ('episode_id','circuit_id','block_id','split','duration_ticks','dt_ms'))
                    and metadata['seeds']=={'source_rng_seed':seed} and metadata['control_kind']=='piecewise_constant',
                    'episode metadata/RNG/control semantics changed')
            record=metadata['source_record']
            require(record['complete_unit_recording'] is True and record['simulation_completed'] is True
                    and record['recorded_until_tick']==record['expected_duration_ticks']==100000
                    and record['core_unit_count']==1280 and record['source_revision']==SOURCE_REVISION
                    and record['circuit_provenance']==identity and record['native_source_sha256']==SOURCE_ENTRY_SHA256
                    and record['raw_sha256']==report_case(report,case_id)['raw_sha256']
                    and record['initial_and_background_streams']=='source_global_rng_not_separated',
                    'complete source episode receipt changed')
        window=dataset.window(case_id,boundary_tick=50000,history_ticks=50000,horizon_ticks=50000,bin_ticks=100)
        dense=np.concatenate([window['history'],window['targets']])
        controls[:]=0;controls[20:60,0]=CONDITIONS[condition]
        features=({'name':'cue_current','unit':'nA','population':'E','center_degree':120.,'width_degree':30.},
            {'name':'distractor_current','unit':'nA','population':'E',
             'center_degree':config['distractor_center_deg'],'width_degree':config['distractor_width_deg']})
        require(dense.shape==(1000,1280) and np.array_equal(window['unit_ids'],np.arange(1280))
                and window['bin_seconds']==.005 and np.array_equal(window['known_controls'],controls)
                and window['control_features']==list(features) and window['observation_mask'].all(),
                'full-unit archive or prescribed current timing differs')
        result[case_id]=dense
    return result


def report_case(report,case_id):
    return next(row for row in report['cases'] if row['case_id']==case_id)


def reference(root, config, conditions, expected_status):
    delivery=read(root/'delivery.json')
    require(delivery['status']==expected_status and delivery['units']==1280
            and delivery['duration_ms']==500 and delivery['source_rng_blocks']==3
            and delivery['working_memory_qualified'] is False and delivery['neuroscientific_claim_qualified'] is False
            and delivery['production_episodes']==0,'completed source reference delivery required')
    inventory={row['path']:row for row in delivery['files']}
    required=['original/report.json','original/dataset/dataset.json']
    report=read(root/'original/report.json')
    check_source(report,config,500.)
    cases,expected=registered_cases(report,conditions)
    required.extend('original/'+row['raw_file'] for row in cases.values())
    dataset=CircuitDataset(root/'original/dataset',splits=('development',))
    required.extend('original/dataset/'+row['path'] for row in dataset.catalog['circuits'].values())
    for name in required:
        require(name in inventory and sha256_file(root/name)==inventory[name]['sha256'],
                'consumed reference artifact differs from archived delivery: '+name)
    unit_identity(dataset,config,report)
    return report,cases,expected


def prefix_check(current, prior):
    a=current['spike_ticks']<10000; b=prior['spike_ticks']<10000
    require(np.array_equal(current['spike_ticks'][a],prior['spike_ticks'][b])
            and np.array_equal(current['spike_unit_indices'][a],prior['spike_unit_indices'][b]),
            '500ms full-unit native spike prefix changed')
    return int(a.sum())


def independent_summary(events, duration_ms):
    """Raw-event scalar reference: no NumPy projection/dot or producer helper."""
    counts=Counter(int(unit) for unit in events)
    ne,ni=1024,256
    center,half=round(ne/360*120),round(ne/360*30/2)
    cue={(center+offset)%ne for offset in range(-half,half+1)}
    e=sum(value for unit,value in counts.items() if unit<ne)
    i=sum(value for unit,value in counts.items() if unit>=ne)
    x=math.fsum(value*math.cos(2*math.pi*unit/ne) for unit,value in counts.items() if unit<ne)
    y=math.fsum(value*math.sin(2*math.pi*unit/ne) for unit,value in counts.items() if unit<ne)
    length=math.hypot(x,y);defined=e>0 and length>64*math.ulp(1.)*e
    angle=math.degrees(math.atan2(y,x))%360 if defined else None
    seconds=duration_ms/1000
    inside=sum(value for unit,value in counts.items() if unit in cue)
    rate_in=inside/len(cue)/seconds;rate_out=(e-inside)/(ne-len(cue))/seconds
    return dict(excitatory_spikes=e,inhibitory_spikes=i,excitatory_rate_hz=e/ne/seconds,
        inhibitory_rate_hz=i/ni/seconds,cue_units=len(cue),cue_region_rate_hz=rate_in,
        noncue_region_rate_hz=rate_out,cue_selectivity_hz=rate_in-rate_out,
        vector_x_spikes=x,vector_y_spikes=y,resultant_length=length/e if e else None,
        decoded_angle_deg=angle,circular_cue_error_deg=(angle-120+180)%360-180 if defined else None,
        direction_defined=defined,all_unit_rate_hz=(e+i)/(ne+ni)/seconds)


def _check_readout(actual, expected):
    checks=0
    for name,value in expected.items():
        observed=actual[name]
        if value is None:
            require(observed is None or observed=='','undefined circular quantity must remain missing')
        elif isinstance(value,(bool,np.bool_)):
            require(observed in (value,str(value)),'circular direction validity changed')
        else:
            observed=float(observed)
            difference=abs((observed-value+180)%360-180) if name in (
                'decoded_angle_deg','circular_cue_error_deg','decoded_angle_change_deg') else abs(observed-value)
            require(np.isfinite(observed) and difference<=READOUT_ATOL+READOUT_RTOL*abs(value),
                    'independent raw-event readout mismatch: '+name)
        checks+=1
    return checks


def replay_readouts(original, analysis_dir, output):
    original,analysis_dir,output=map(Path,(original,analysis_dir,output))
    require(not output.exists(),'readout replay output already exists')
    record=read(analysis_dir/'analysis.json')
    require(record['original_report_sha256']==sha256_file(original/'report.json'),'readout source report changed')
    for name,digest in record['artifacts'].items():
        require(sha256_file(analysis_dir/name)==digest,'readout artifact changed: '+name)
    report=read(original/'report.json');cases={r['case_id']:r for r in report['cases']}
    with (analysis_dir/'responses.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    require(len(rows)==45 and {(r['case_id'],r['phase']) for r in rows}=={
        (case,phase) for case in cases for phase in PHASES},'complete phase readout denominator required')
    checks=count_checks=0
    independently_scored={}
    with np.load(analysis_dir/'phase_unit_counts.npz',allow_pickle=False) as phase_counts:
        require(set(phase_counts.files)=={f'{case}_{phase}' for case in cases for phase in PHASES},
                'complete per-phase all-unit arrays required')
        for case_id,case in cases.items():
            raw=raw_events(original,case,100000)
            for row in (r for r in rows if r['case_id']==case_id):
                start,stop=PHASES[row['phase']]
                require(float(row['start_ms'])==start and float(row['stop_ms'])==stop,
                        'phase exposure changed')
                # Count individual raw events directly, independently of the archive bins.
                events=[int(unit) for tick,unit in zip(raw['spike_ticks'],raw['spike_unit_indices'])
                        if start/.05<=int(tick)<stop/.05]
                counts=Counter(events)
                expected=np.array([counts[unit] for unit in range(1280)],dtype=np.int64)
                require(np.array_equal(phase_counts[f'{case_id}_{row["phase"]}'],expected),
                        'independent raw-event phase counts differ')
                independent=independent_summary(events,stop-start)
                independently_scored[case_id,row['phase']]=independent
                count_checks+=1280;checks+=_check_readout(row,independent)
    with (analysis_dir/'paired_effects.csv').open(newline='') as stream:
        paired=list(csv.DictReader(stream))
    labels={'cue_minus_omission':('cue','omission'),
            'strong_cue_minus_omission':('strong_cue','omission'),'strong_cue_minus_cue':('strong_cue','cue')}
    require(len(paired)==45 and {(int(r['source_rng_seed']),r['phase'],r['contrast']) for r in paired}=={
        (seed,phase,label) for seed in SEEDS for phase in PHASES for label in labels},
        'complete paired effects required')
    for row in paired:
        left,right=labels[row['contrast']];seed=int(row['source_rng_seed']);phase=row['phase']
        a,b=independently_scored[f'pfc_{seed}_{left}',phase],independently_scored[f'pfc_{seed}_{right}',phase]
        expected={key:None if a[key] is None or b[key] is None else a[key]-b[key]
                  for key in row if key not in ('source_rng_seed','phase','contrast','decoded_angle_change_deg')}
        expected['decoded_angle_change_deg']=None if not (a['direction_defined'] and b['direction_defined']) else (
            a['decoded_angle_deg']-b['decoded_angle_deg']+180)%360-180
        checks+=_check_readout(row,expected)
    result=dict(status='PASS',phase_unit_count_values_checked=count_checks,
        scalar_readout_values_checked=checks,complete_episodes=9,source_rng_blocks=3,
        producer_sha256=record['implementation_sha256'],analysis_artifacts=record['artifacts'],
        design_sha256=record['design_sha256'],
        original_report_sha256=record['original_report_sha256'],
        rtol=READOUT_RTOL,atol=READOUT_ATOL,method='raw event Counter and scalar math.fsum circular moments',
        scientific_fidelity_qualified=False,working_memory_qualified=False,
        implementation_sha256=sha256_file(Path(__file__)))
    write_json(output,result)
    return result


def contrasts(rows):
    metrics=('all_unit_rate_hz','excitatory_rate_hz','inhibitory_rate_hz','cue_selectivity_hz',
             'cue_region_rate_hz','noncue_region_rate_hz','resultant_length')
    result=[]
    for seed in SEEDS:
        for phase in PHASES:
            selected={r['condition']:r for r in rows if r['source_rng_seed']==seed and r['phase']==phase}
            for left,right in (('cue','omission'),('strong_cue','omission'),('strong_cue','cue')):
                a,b=selected[left],selected[right]
                delta={m:None if a[m] is None or b[m] is None else a[m]-b[m] for m in metrics}
                angle=None if not (a['direction_defined'] and b['direction_defined']) else (
                    a['decoded_angle_deg']-b['decoded_angle_deg']+180)%360-180
                result.append(dict(source_rng_seed=seed,phase=phase,contrast=f'{left}_minus_{right}',
                    decoded_angle_change_deg=angle,**delta))
    require(all(all(row[m] in (None,0.) for m in metrics) for row in result if row['phase']=='baseline'),
            'matched pre-cue readouts differ')
    return result


def analyze(original, replay, replay_receipt, design_path, config_path, weak_reference, strong_reference, output):
    original,replay,replay_receipt,design_path,config_path,weak_reference,strong_reference,output=map(
        Path,(original,replay,replay_receipt,design_path,config_path,weak_reference,strong_reference,output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result=dict(status='CHECKING_INPUTS',working_memory_qualified=False,full_scale_qualified=False,
        scientific_fidelity_qualified=False,neuroscientific_claim_qualified=False,production_episodes=0,
        confidence_intervals=False,p_values=False,q_values=False)
    try:
        design=read(design_path);config=yaml.safe_load(config_path.read_text())
        check_design(design,config,config_path)
        report,fresh,receipt=read(original/'report.json'),read(replay/'report.json'),read(replay_receipt)
        check_source(report,config,5000.);check_source(fresh,config,5000.)
        require(report['resolved_config']==fresh['resolved_config']==config,
                'executed source config differs from the registered file')
        require(receipt['status']=='PASS' and receipt['complete_episodes']==9 and receipt['core_units']==1280
                and receipt['refractory_violations']==0
                and receipt['original_report_sha256']==sha256_file(original/'report.json')
                and receipt['replayed_report_sha256']==sha256_file(replay/'report.json')
                and receipt['source_revision']==SOURCE_REVISION
                and receipt['comparison_implementation_sha256']==sha256_file(Path(pfc_reference_replay.__file__))
                and receipt['replay_runs_count_as_additional_samples'] is False,
                'all nine fresh native array replays must PASS before analysis')
        cases,identities=registered_cases(report,CONDITIONS)
        fresh_cases,_=registered_cases(fresh,CONDITIONS)
        weak,weak_cases,_=reference(weak_reference,config,('cue','omission'),'ARCHIVED_REPLAYED_SOURCE_DIAGNOSTIC')
        strong,strong_cases,_=reference(strong_reference,config,('strong_cue',),'ARCHIVED_REPLAYED_CUE_STRENGTH_DIAGNOSTIC')
        for prior in (weak,strong,fresh):
            require(all(prior[k]==report[k] for k in ('source_revision','source_entry_sha256','brian2_version','numpy_version')),
                    'source implementation or simulator environment changed')
        raw={case:raw_events(original,row,100000) for case,row in cases.items()}
        dense=archive_counts(original,report,config,identities,raw)
        rows,counts,prefixes=[],{},[]
        for case_id,(seed,condition) in identities.items():
            fresh_path=replay/fresh_cases[case_id]['raw_file']
            require(sha256_file(fresh_path)==fresh_cases[case_id]['raw_sha256'],
                    'fresh native replay artifact changed after its PASS receipt')
            prior_root=strong_reference if condition=='strong_cue' else weak_reference
            prior_case=(strong_cases if condition=='strong_cue' else weak_cases)[case_id]
            prior=raw_events(prior_root/'original',prior_case,10000)
            events=prefix_check(raw[case_id],prior)
            prefixes.append(dict(case_id=case_id,source_rng_seed=seed,condition=condition,
                stop_ms=500.,events_checked=events,unit_count=1280,prefix_equal=True,
                reference_raw_sha256=prior_case['raw_sha256'],extended_raw_sha256=cases[case_id]['raw_sha256']))
            ticks,ids=raw[case_id]['spike_ticks'],raw[case_id]['spike_unit_indices']
            require(np.array_equal(np.bincount(ids[ticks<100000],minlength=1280),dense[case_id].sum(0)),
                    'native full-episode and archive totals differ')
            for phase,(start,stop) in PHASES.items():
                selected=(ticks>=round(start/.05))&(ticks<round(stop/.05))
                count=np.bincount(ids[selected],minlength=1280)
                require(np.array_equal(count,dense[case_id][round(start/5):round(stop/5)].sum(0)),
                        'native versus archive phase counts differ')
                counts[f'{case_id}_{phase}']=count
                summary=summarize_counts(count,1024,120.,30.,stop-start)
                summary['all_unit_rate_hz']=int(count.sum())/1280/((stop-start)/1000)
                rows.append(dict(case_id=case_id,source_rng_seed=seed,condition=condition,
                    phase=phase,start_ms=start,stop_ms=stop,**summary))
        tables={'responses.csv':rows,'paired_effects.csv':contrasts(rows),'prefix_comparisons.csv':prefixes}
        for name,table in tables.items():write_rows(output/name,table)
        np.savez_compressed(output/'phase_unit_counts.npz',**counts)
        result.update(status='READOUT_REPLAY_PENDING',complete_source_episodes=9,source_rng_blocks=3,
            units=1280,duration_ms=5000.,source_revision=SOURCE_REVISION,configuration=config,
            source_rng_scope='three joint initial/background source RNG blocks; duration extension and replay add no independent blocks',
            original_report_sha256=sha256_file(original/'report.json'),
            fresh_native_replay_sha256=sha256_file(replay_receipt),design_sha256=sha256_file(design_path),
            weak_delivery_sha256=sha256_file(weak_reference/'delivery.json'),
            strong_delivery_sha256=sha256_file(strong_reference/'delivery.json'),
            phases_ms=PHASES,full_unit_500ms_prefixes_verified=9,
            interpretation='descriptive phase amplitude/direction and paired source-model effects; sparse phases do not certify uninterrupted persistence across all 4.7s',
            implementation_sha256=sha256_file(Path(__file__)),
            response_implementation_sha256=sha256_file(Path(__file__).with_name('pfc_response_analysis.py')),
            artifacts={name:sha256_file(output/name) for name in (*tables,'phase_unit_counts.npz')})
        write_json(output/'analysis.json',result)
        checked=replay_readouts(original,output,output/'readout_replay.json')
        result.update(status='DESCRIBED_REPLAYED_DELAY_DIAGNOSTIC',
            readout_replay_sha256=sha256_file(output/'readout_replay.json'),
            phase_unit_count_values_verified=checked['phase_unit_count_values_checked'],
            scalar_readout_values_verified=checked['scalar_readout_values_checked'])
        write_json(output/'analysis.json',result)
        return result
    except BaseException as error:
        result.update(status='BLOCKED_INCOMPLETE_INPUTS' if isinstance(error,FileNotFoundError) else 'FAILED',
                      error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        write_json(output/'analysis.json',result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','replay','replay-receipt','design','config','weak-reference','strong-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    result=analyze(a.original,a.replay,a.replay_receipt,a.design,a.config,a.weak_reference,a.strong_reference,a.output)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()


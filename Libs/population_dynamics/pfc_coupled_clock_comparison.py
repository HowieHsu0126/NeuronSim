"""Compare complete coupled PFC pilot episodes with the existing clock sample.

One shared calibration block supplies descriptive path and response differences;
this is not an estimate of ensemble variance, convergence or confirmation power.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_coupled_clock_execution as coupled_execution
from .pfc_coupled_clock_execution import case_grid, case_key
from .pfc_temporal_readouts import temporal_rows
from .pfc_delay_analysis import PHASES, _check_readout, write_rows
from .pfc_fluctuation_analysis import unit_mapping, prepared_events, unit_chunks
from . import pfc_count_histograms
from .pfc_phase_responses import CONDITIONS, METRICS, CONTRASTS
from .pfc_response_analysis import summarize_counts
from .pfc_scale_analysis import _source_manifest, _source_bound_file, _source_pairing
from .provenance import freeze_json, sha256_file

LABELS=('numpy_dt_0p05','numpy_dt_0p025','numpy_dt_0p0125')
DT=(.05,.025,.0125)
ADJACENT_CLOCK_PAIRS=tuple(zip(LABELS[:-1],LABELS[1:]))
ALL_CLOCK_PAIRS=ADJACENT_CLOCK_PAIRS+((LABELS[0],LABELS[2]),)
GAIN_ANALYSIS_SCHEMA="pfc_recurrent_gain_coupled_clock_comparison_v1"
GAIN_INDEX_SCHEMA="pfc_recurrent_gain_coupled_clock_source_index_v1"
GAIN_REPLICATES_ANALYSIS_SCHEMA="pfc_recurrent_gain_coupled_clock_replicates_comparison_v1"
GAIN_ANALYSIS_REGISTRATIONS = {
    GAIN_ANALYSIS_SCHEMA: ('development_recurrent_gain_coupled_clock_comparison',
                           'pfc_coupled_clock_recurrent_gain_v1'),
    GAIN_REPLICATES_ANALYSIS_SCHEMA: ('development_recurrent_gain_coupled_clock_replicates_comparison',
                                      'pfc_coupled_clock_recurrent_gain_replicates_v1'),
}


def require(condition,message):
    if not condition:raise ValueError(message)


def discrepancy_rows(fields,clock_pairs=ADJACENT_CLOCK_PAIRS):
    rows=[]
    for scheme,levels in fields.items():
        for left,right in clock_pairs:
            for condition in CONDITIONS:
                a,b=levels[left][condition],levels[right][condition]
                require(a.shape==b.shape==(1000,17) and np.isfinite(a).all() and np.isfinite(b).all(),
                        'COMPLETE_FINITE_5MS_FIELDS_REQUIRED')
                for phase,(start,stop) in PHASES.items():
                    difference=a[round(start/5):round(stop/5)]-b[round(start/5):round(stop/5)]
                    for field in range(17):
                        delta=difference[:,field]
                        rows.append(dict(scheme=scheme,left_cohort=left,right_cohort=right,
                            condition=condition,phase=phase,field_index=field,units='Hz',
                            matched_time_bins=len(delta),signed_mean_hz=float(delta.mean()),
                            mean_absolute_hz=float(np.abs(delta).mean()),
                            rms_hz=float(np.sqrt(np.mean(delta*delta))),
                            maximum_absolute_hz=float(np.abs(delta).max())))
    return rows


def rms_comparison(rows):
    keys=('left_cohort','right_cohort','condition','phase','field_index')
    indexed={}
    for row in rows:
        key=tuple(row[k] for k in keys);pair=indexed.setdefault(key,{})
        require(row['scheme'] not in pair,'DUPLICATE_PATH_COMPARISON_ROW')
        pair[row['scheme']]=row
    result=[]
    for key,pair in indexed.items():
        require(set(pair)=={'native_same_seed','coupled_inputs'},'BOTH_CLOCK_COUPLING_SCHEMES_REQUIRED')
        old,new=(float(pair[scheme]['rms_hz']) for scheme in ('native_same_seed','coupled_inputs'))
        ratio=new/old if old else None
        status=('DEFINED_RATIO' if old else ('BOTH_ZERO' if new==0 else 'NATIVE_ZERO_COUPLED_NONZERO'))
        result.append(dict(zip(keys,key),native_same_seed_rms_hz=old,coupled_rms_hz=new,
                           coupled_minus_native_rms_hz=new-old,coupled_over_native_rms=ratio,ratio_status=status))
    return result


def response_rows(values,clock_pairs=ADJACENT_CLOCK_PAIRS):
    source,paired=[],[]
    for scheme,levels in values.items():
        for left,right in clock_pairs:
            a,b=levels[left],levels[right]
            require(a.shape==b.shape==(3,5,8) and np.isfinite(a).all() and np.isfinite(b).all(),
                    'COMPLETE_FINITE_PHASE_RESPONSES_REQUIRED')
            for ci,condition in enumerate(CONDITIONS):
                for pi,phase in enumerate(PHASES):
                    for mi,metric in enumerate(METRICS):
                        source.append(dict(scheme=scheme,left_cohort=left,right_cohort=right,
                            condition=condition,phase=phase,metric=metric,units='Hz',
                            left_value=float(a[ci,pi,mi]),right_value=float(b[ci,pi,mi]),
                            left_minus_right=float(a[ci,pi,mi]-b[ci,pi,mi])))
            for contrast,positive,negative in CONTRASTS:
                i,j=CONDITIONS.index(positive),CONDITIONS.index(negative)
                x,y=a[i]-a[j],b[i]-b[j]
                for pi,phase in enumerate(PHASES):
                    for mi,metric in enumerate(METRICS):
                        paired.append(dict(scheme=scheme,left_cohort=left,right_cohort=right,
                            contrast=contrast,phase=phase,metric=metric,units='Hz',
                            left_value=float(x[pi,mi]),right_value=float(y[pi,mi]),
                            left_minus_right=float(x[pi,mi]-y[pi,mi])))
    return source,paired



def accumulate_case_histograms(raw, config, condition_index, histograms):
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons']
    mapping,sizes=unit_mapping(ne,ni,config['excitatory_preference_bins'])
    bins=round(config['duration_ms']/5);bin_ticks=round(5/config['dt_ms'])
    events=prepared_events(raw,ne+ni,bin_ticks,bins)
    reconstructed=np.zeros((bins,len(sizes)),dtype=np.int64)
    for start,stop,counts in unit_chunks([events],ne+ni,bins):
        fields=mapping[start:stop]
        for field in np.unique(fields):reconstructed[:,field]+=counts[0][:,fields==field].sum(axis=1)
        for pi,(_, (left,right)) in enumerate(PHASES.items()):
            selected=counts[:,round(left/5):round(right/5)]
            for wi,width in enumerate((5,50,100)):
                pfc_count_histograms.accumulate(histograms,selected,fields,width//5,
                    condition=condition_index,phase=pi,window=wi)
    return reconstructed


def cdf_on_support(values, old_support, new_support):
    old,new=np.asarray(old_support),np.asarray(new_support)
    require(old.ndim==new.ndim==1 and len(old)>0 and len(new)>0 and old[0]==new[0]==0
            and np.all(np.diff(old)>0) and np.all(np.diff(new)>0)
            and values.shape[-1]==len(old) and np.all(values[...,-1]==1.),
            'COMPLETE_OBSERVED_CDF_SUPPORT_REQUIRED')
    indices=np.searchsorted(old,new,side='right')-1
    return np.take(values,np.minimum(indices,len(old)-1),axis=-1)


def count_cdf_rows(cdfs,support,clock_pairs=ADJACENT_CLOCK_PAIRS):
    rows=[]
    for scheme,levels in cdfs.items():
        for left,right in clock_pairs:
            a,b=levels[left],levels[right]
            require(a.shape==b.shape==(3,5,3,17,len(support)), 'COMPLETE_PILOT_CDF_AXES_REQUIRED')
            for ci,condition in enumerate(CONDITIONS):
                for pi,phase in enumerate(PHASES):
                    for wi,window in enumerate((5,50,100)):
                        for field in range(17):
                            for ki,threshold in enumerate(support):
                                x,y=float(a[ci,pi,wi,field,ki]),float(b[ci,pi,wi,field,ki])
                                rows.append(dict(scheme=scheme,left_cohort=left,right_cohort=right,
                                    condition=condition,phase=phase,window_ms=window,field_index=field,
                                    count_threshold=int(threshold),units='probability',left_cdf=x,
                                    right_cdf=y,left_minus_right=x-y))
    return rows

def pilot_case(root,manifest,relative,seed,condition,dt,sizes,histograms,condition_index):
    def bound(name):return _source_bound_file(root,manifest,str(relative/name))
    report=json.loads(bound('report.json').read_text())
    case=f'pfc_{seed}_{condition}'
    require(report['status']=='COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE'
            and report['source_execution_complete'] is True and report['case_id']==case
            and report['source_rng_seed']==seed and report['condition']==condition
            and report['dt_ms']==dt and report['coupled_core_units']==100000
            and report['duration_ms']==5000. and report['coarse_final_rng_matches_archived_episode'] is True
            and all(report['coarse_current_agreement'].values())
            and all(row['passed'] is True for row in report['validations'].values())
            and report['inputs_used']['input_mode']=='coupled_binomial_clock_refinement_stream',
            'COMPLETE_VALIDATED_COUPLED_CASE_REQUIRED')
    for name,row in report['outputs'].items():
        require(sha256_file(bound(name))==row['sha256'],'COUPLED_CASE_OUTPUT_BINDING_CHANGED:'+name)
    with np.load(bound('field_observations.npz'),allow_pickle=False) as stored:
        require(np.array_equal(stored['field_unit_counts'],sizes)
                and np.array_equal(stored['field_indices'],np.arange(17))
                and np.array_equal(stored['time_start_ms'],np.arange(1000)*5.),'PILOT_FIELD_AXIS_CHANGED')
        counts=stored['field_counts'];rates=stored['rates_hz']
    require(counts.shape==(1000,17) and counts.dtype.kind in 'iu' and np.all(counts>=0)
            and np.array_equal(rates,counts/(sizes[None,:]*.005)),'PILOT_FIELD_UNITS_OR_COUNTS_CHANGED')
    with bound('phase_responses.csv').open(newline='') as stream:phase_rows=list(csv.DictReader(stream))
    require(len(phase_rows)==5 and {row['phase'] for row in phase_rows}==set(PHASES),'COMPLETE_UNIQUE_SOURCE_PHASES_REQUIRED')
    indexed={row['phase']:row for row in phase_rows}
    mapping,_=unit_mapping(80000,20000,16);values=np.empty((5,8));scalar_checks=0
    with np.load(bound('phase_unit_counts.npz'),allow_pickle=False) as stored:
        require(set(stored.files)=={case+'_'+phase for phase in PHASES},'COMPLETE_UNIT_PHASE_VECTORS_REQUIRED')
        for pi,(phase,(start,stop)) in enumerate(PHASES.items()):
            row=indexed[phase];units=stored[case+'_'+phase]
            require(row['case_id']==case and row['condition']==condition and int(row['source_rng_seed'])==seed
                    and units.shape==(100000,) and units.dtype.kind in 'iu' and np.all(units>=0),
                    'PILOT_PHASE_IDENTITY_CHANGED')
            grouped=np.zeros(17,dtype=np.int64);np.add.at(grouped,mapping,units)
            require(np.array_equal(grouped,counts[round(start/5):round(stop/5)].sum(axis=0)),
                    'PILOT_UNIT_COUNTS_AND_FIELDS_DISAGREE')
            summary=summarize_counts(units,80000,120.,30.,stop-start)
            summary['all_unit_rate_hz']=float(units.sum())/(100000*(stop-start)/1000)
            scalar_checks+=_check_readout(row,summary)
            summary['vector_x_hz']=summary['vector_x_spikes']/(80000*(stop-start)/1000)
            summary['vector_y_hz']=summary['vector_y_spikes']/(80000*(stop-start)/1000)
            values[pi]=[summary[metric] for metric in METRICS]
    with np.load(bound('raw.npz'),allow_pickle=False) as raw:
        ticks=raw['spike_ticks'];ids=raw['spike_unit_indices'];mask=ticks<round(100/dt)
        prefix_ticks=ticks[mask];prefix_ids=ids[mask]
    rebuilt=accumulate_case_histograms({'spike_ticks':ticks,'spike_unit_indices':ids},
        report['resolved_config'],condition_index,histograms)
    require(np.array_equal(rebuilt,counts),'PILOT_HISTOGRAM_AND_DIRECT_FIELDS_DISAGREE')
    with np.load(bound('final_rng.npz'),allow_pickle=False) as stored:rng={key:stored[key] for key in stored.files}
    paired=dict(initial=np.load(bound('initial_voltage.npy'),allow_pickle=False),final_rng=rng,
                prefix_ticks=prefix_ticks,prefix_ids=prefix_ids,inputs_used=report['inputs_used'])
    return rates,values,paired,dict(case_id=case,dt_ms=dt,report_sha256=sha256_file(bound('report.json')),
                                  scalar_values_verified=scalar_checks,cost=report['cost'],wall_seconds=report['wall_seconds'])


def run_legacy(config_path,project_root,output):
    settings=yaml.safe_load(config_path.read_text())
    require(settings.get('schema')=='pfc_coupled_clock_comparison_v1'
            and settings.get('scope')=='development_single_block_comparison', 'DESCRIPTIVE_COUPLED_COMPARISON_SCOPE_REQUIRED')
    pilot=project_root/settings['pilot_archive'];pilot_delivery=json.loads((pilot/'delivery.json').read_text())
    require(pilot_delivery['status']=='COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT'
            and str(pilot_delivery['job_id'])==str(settings['pilot_job']), 'COMPLETE_BOUND_PILOT_REQUIRED')
    manifest=_source_manifest(pilot_delivery)
    settings_path=_source_bound_file(pilot,manifest,'Libs/configs/neural_population_pfc_coupled_clock_pilot.yaml')
    pilot_settings=yaml.safe_load(settings_path.read_text());grid=case_grid(pilot_settings);seed=pilot_settings['source_rng_seed']
    calibration=project_root/settings['calibration_archive']
    require(sha256_file(calibration/'analysis/report.json')==settings['calibration_report_sha256'], 'CALIBRATION_COMPARISON_REPORT_CHANGED')
    calibration_report=json.loads((calibration/'analysis/report.json').read_text())
    cal_manifest=_source_manifest(json.loads((calibration/'delivery.json').read_text()))
    response_path=_source_bound_file(calibration,cal_manifest,'analysis/block_source_responses.npz')
    require(sha256_file(response_path)==calibration_report['outputs']['block_source_responses.npz']['sha256'],
            'CALIBRATION_RESPONSE_OUTPUT_CHANGED')
    native_fields={};native_values={};cohort_inputs=[];expected_sizes=np.r_[np.full(16,5000),20000]
    with np.load(response_path,allow_pickle=False) as stored:
        require(stored['condition_names'].tolist()==list(CONDITIONS)
                and stored['phase_names'].tolist()==list(PHASES)
                and stored['metric_names'].tolist()==list(METRICS)
                and stored['source_rng_seeds'].tolist()==list(range(95001,95033)), 'CALIBRATION_RESPONSE_AXES_CHANGED')
        bi=stored['source_rng_seeds'].tolist().index(seed)
        for label in LABELS:native_values[label]=stored[label+'_values'][:,bi].copy()
    for label,dt in zip(LABELS,DT):
        root=project_root/settings['calibration_field_archives'][label]
        expected=calibration_report['inputs'][label]
        require(sha256_file(root/'delivery.json')==expected['delivery_sha256'],'CALIBRATION_FIELD_DELIVERY_CHANGED')
        fm=_source_manifest(json.loads((root/'delivery.json').read_text()))
        report_path=_source_bound_file(root,fm,'analysis/report.json')
        report=json.loads(report_path.read_text())
        require(report['status']=='COMPLETE_SOURCE_FLUCTUATION_CALIBRATION'
                and report['dt_ms']==dt and report['joint_rng_blocks']==list(range(95001,95033))
                and report['coupled_core_units']==100000 and report['bin_ms']==5.,'COMPLETE_CANONICAL_FIELD_COHORT_REQUIRED')
        path=_source_bound_file(root,fm,'analysis/field_fluctuations.npz')
        with np.load(path,allow_pickle=False) as stored:
            require(np.array_equal(stored['field_unit_counts'],expected_sizes)
                    and stored['source_rng_seeds'].tolist()==list(range(95001,95033)), 'CANONICAL_FIELD_AXES_CHANGED')
            native_fields[label]={condition:stored[condition+'_field_counts'][bi]/(expected_sizes[None,:]*.005)
                                  for condition in CONDITIONS}
        cohort_inputs.append(dict(label=label,delivery_sha256=expected['delivery_sha256'],field_sha256=sha256_file(path)))
    coupled_fields={LABELS[0]:native_fields[LABELS[0]]}
    coupled_values={LABELS[0]:native_values[LABELS[0]]}
    case_inputs=[];pair_checks=[];histograms={label:{} for label in LABELS[1:]}
    for label,dt in zip(LABELS[1:],DT[1:]):
        coupled_fields[label]={};coupled_values[label]=np.empty((3,5,8));pairs={};registered={}
        for ci,condition in enumerate(CONDITIONS):
            relative=Path('source_cases')/case_key(dt,condition,seed)
            fields,values,paired,info=pilot_case(pilot,manifest,relative,seed,condition,dt,expected_sizes,histograms[label],ci)
            coupled_fields[label][condition]=fields;coupled_values[label][ci]=values
            case=f'pfc_{seed}_{condition}';pairs[case]=paired;registered[case]=(seed,condition);case_inputs.append(info)
        checks=_source_pairing(pairs,registered,[seed])
        flags=('exact_pre_cue_events','exact_initial_voltage','exact_final_source_rng','same_source_input_consumption')
        require(all(all(row[key] for key in flags) for row in checks),'COUPLED_CONDITION_INPUT_PAIRING_FAILED')
        pair_checks.extend(dict(cohort=label,**row) for row in checks)
    output.mkdir(parents=True,exist_ok=False)
    histogram_metadata={};coordinates={};observations={}
    for label in LABELS[1:]:
        path=output/(label+'_count_histograms.npz')
        histogram_metadata[label]=pfc_count_histograms.write(path,histograms[label],conditions=CONDITIONS,
            seeds=[seed],phases=PHASES,windows_ms=[5,50,100],field_sizes=expected_sizes)
        with np.load(path,allow_pickle=False) as stored:
            coordinates[label]=stored['coordinates'];observations[label]=stored['observations']
    count_path=_source_bound_file(calibration,cal_manifest,'analysis/block_count_cdfs.npz')
    require(sha256_file(count_path)==calibration_report['outputs']['block_count_cdfs.npz']['sha256'],
            'CALIBRATION_COUNT_CDF_OUTPUT_CHANGED')
    with np.load(count_path,allow_pickle=False) as stored:
        require(stored['condition_names'].tolist()==list(CONDITIONS)
                and stored['phase_names'].tolist()==list(PHASES)
                and stored['source_rng_seeds'].tolist()==list(range(95001,95033))
                and stored['window_ms'].tolist()==[5,50,100]
                and np.array_equal(stored['field_unit_counts'],expected_sizes),'CALIBRATION_COUNT_CDF_AXES_CHANGED')
        old_support=stored['count_support']
        support=np.arange(max(int(old_support[-1]),*(int(x[:,5].max()) for x in coordinates.values()))+1,dtype=np.int64)
        native_cdfs={label:cdf_on_support(stored[label+'_cdf'][:,bi],old_support,support) for label in LABELS}
    coupled_cdfs={LABELS[0]:native_cdfs[LABELS[0]]}
    for label in LABELS[1:]:
        coupled_cdfs[label]=pfc_count_histograms.cdf(coordinates[label],observations[label],support,(3,1,5,3,17))[:,0]
    count_rows=count_cdf_rows({'native_same_seed':native_cdfs,'coupled_inputs':coupled_cdfs},support)
    np.savez(output/'pilot_count_cdfs.npz',count_support=support,
        **{scheme+'_'+label:values for scheme,levels in [('native',native_cdfs),('coupled',coupled_cdfs)] for label,values in levels.items()})
    field_rows=discrepancy_rows({'native_same_seed':native_fields,'coupled_inputs':coupled_fields})
    comparison_rows=rms_comparison(field_rows)
    source_rows,paired_rows=response_rows({'native_same_seed':native_values,'coupled_inputs':coupled_values})
    tables={'field_clock_discrepancies.csv':field_rows,'pathwise_rms_comparison.csv':comparison_rows,
            'source_response_comparison.csv':source_rows,'paired_response_comparison.csv':paired_rows,
            'shared_input_checks.csv':pair_checks,'count_cdf_comparison.csv':count_rows}
    for name,rows in tables.items():write_rows(output/name,rows)
    arrays={scheme+'_'+label:values for scheme,levels in [('native',native_values),('coupled',coupled_values)] for label,values in levels.items()}
    np.savez(output/'phase_responses.npz',**arrays)
    counts={key:sum(row['coupled_minus_native_rms_hz']<0 if key=='lower' else
                    row['coupled_minus_native_rms_hz']>0 if key=='higher' else
                    row['coupled_minus_native_rms_hz']==0 for row in comparison_rows) for key in ('lower','higher','equal')}
    report=dict(status='COMPLETE_DESCRIPTIVE_COUPLED_CLOCK_COMPARISON',settings=settings,
        settings_sha256=sha256_file(config_path),pilot_delivery_sha256=sha256_file(pilot/'delivery.json'),
        calibration_report_sha256=settings['calibration_report_sha256'],cohort_inputs=cohort_inputs,pilot_cases=case_inputs,
        selected_source_rng_seed=seed,original_independent_blocks=1,complete_new_mechanistic_episodes=6,
        existing_native_reference_episodes_selected=9,shared_input_checks=len(pair_checks),
        source_simulations=0,production_episodes=0,confirmation_execution_released=False,
        row_counts={name:len(rows) for name,rows in tables.items()},pathwise_rms_change_rows=counts,
        count_histogram_metadata=histogram_metadata,empirical_count_support=support.tolist(),
        unit_count_histogram_implementation_sha256=sha256_file(Path(pfc_count_histograms.__file__)),
        scientific_fidelity_qualified=False,ensemble_variance_reduction_established=False,confidence_intervals=False,
        numerical_margins_changed=False,confirmation_sample_size_selected=False,
        interpretation='All conditions and signed effects retained. Pathwise discrepancy between two clock realizations is not weak-error bias. '
        'One shared calibration block cannot establish variance reduction, coverage, working memory, numerical equivalence or confirmation power. '
        'Existing native fine seeded trajectories and new coupled trajectories are different realizations; neither is discarded. '
        'All-unit5/50/100ms count CDFs retain complete unit/window denominators and observed support without clipping; empirical saturation is not a population support proof.',
        implementation_sha256=sha256_file(Path(__file__)),
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file()})
    freeze_json(output/'report.json',report)
    return report



def analysis_registration(settings, project_root):
    registration = GAIN_ANALYSIS_REGISTRATIONS.get(settings.get('schema'))
    require(registration is not None
            and settings.get('scope') == registration[0]
            and settings.get('confirmation_execution_released') is False
            and settings.get('clock_pairs') == [[.05, .025], [.025, .0125], [.05, .0125]],
            'REGISTERED_DESCRIPTIVE_RECURRENT_GAIN_COMPARISON_REQUIRED')
    protocol_path = project_root/settings['protocol_config']
    require(sha256_file(protocol_path) == settings['protocol_sha256'], 'RECURRENT_GAIN_PROTOCOL_CHANGED')
    protocol = yaml.safe_load(protocol_path.read_text())
    require(protocol.get('schema') == registration[1], 'REGISTERED_RECURRENT_GAIN_PROTOCOL_REQUIRED')
    cohorts = (coupled_execution.GAIN_COHORTS if protocol['schema'] == coupled_execution.GAIN_SCHEMA
               else coupled_execution.GAIN_REPLICATES_COHORTS)
    blocks = [coupled_execution.select_settings(protocol, seed, cohort)
              for cohort, (_, _, seeds, _) in cohorts.items() for seed in seeds]
    return protocol, blocks


def indexed_source_archives(index_path, settings, blocks, project_root):
    index_message = ('COMPLETE_SIX_BLOCK_SOURCE_INDEX_REQUIRED' if len(blocks) == 6
                     else 'COMPLETE_58_BLOCK_SOURCE_INDEX_REQUIRED')
    require(index_path is not None, index_message)
    index = json.loads(index_path.read_text())
    expected = [(block['cohort'], block['source_rng_seed']) for block in blocks]
    rows = index.get('blocks', [])
    require(index.get('schema') == GAIN_INDEX_SCHEMA
            and index.get('protocol_sha256') == settings['protocol_sha256']
            and [(row.get('cohort'), row.get('source_rng_seed')) for row in rows] == expected,
            index_message)
    archives = []
    for row in rows:
        require(isinstance(row.get('archive'), str) and row['archive']
                and isinstance(row.get('delivery_sha256'), str) and len(row['delivery_sha256']) == 64,
                'COMPLETED_SOURCE_ARCHIVE_AND_DELIVERY_HASH_REQUIRED')
        root = project_root/row['archive']
        require(sha256_file(root/'delivery.json') == row['delivery_sha256'], 'COUPLED_SOURCE_DELIVERY_CHANGED')
        delivery = json.loads((root/'delivery.json').read_text())
        require(delivery.get('status') == 'COMPLETE_COUPLED_CLOCK_RECURRENT_GAIN_BLOCK',
                'COMPLETE_RECURRENT_GAIN_SOURCE_BLOCK_REQUIRED')
        manifest = _source_manifest(delivery)
        report = json.loads(_source_bound_file(root, manifest, 'report.json').read_text())
        require(report.get('status') == delivery['status'] and report.get('exit_code') == 0
                and report.get('tests_only') is False and report.get('selected_cohort') == row['cohort']
                and str(report.get('selected_source_rng_seed')) == str(row['source_rng_seed'])
                and report.get('expected_new_development_episodes') == 6
                and report.get('confirmation_execution_released') is False,
                'COUPLED_SOURCE_BLOCK_IDENTITY_CHANGED')
        archives.append((root, manifest, row))
    return index, archives


def bound_case_outputs(root, manifest, relative, report):
    paths = {}
    for name, record in report['outputs'].items():
        require(Path(name).name == name, 'SOURCE_OUTPUT_FILENAME_REQUIRED')
        path = _source_bound_file(root, manifest, str(relative/name))
        require(path.stat().st_size == record['bytes'] and sha256_file(path) == record['sha256'],
                'SOURCE_OUTPUT_BINDING_CHANGED:'+name)
        paths[name] = path
    required = set(coupled_execution.REFERENCE_FILES)-{'report.json'}
    require(required | {'raw.npz', 'terminal_state.npz'} <= set(paths), 'COMPLETE_SOURCE_OUTPUTS_REQUIRED')
    paths['report.json'] = _source_bound_file(root, manifest, str(relative/'report.json'))
    return paths


def inspect_case_readouts(paths, config, case, *, delivered_readouts):
    """Read source events and independently verify existing descriptive products."""
    ne, ni = config['excitatory_neurons'], config['inhibitory_neurons']; total = ne+ni
    require(config['duration_ms'] == 5000. and config['readout_bin_ms'] == 5.
            and config['excitatory_preference_bins'] == 16, 'COMPLETE_REGISTERED_OBSERVATION_MAP_REQUIRED')
    with np.load(paths['raw.npz'], allow_pickle=False) as stored:
        raw = {name: stored[name] for name in ('spike_ticks', 'spike_unit_indices')}
    # temporal_rows validates ordering, all unit identities, and the complete time domain.
    temporal = temporal_rows(raw['spike_ticks'], raw['spike_unit_indices'], n_e=ne, n_i=ni,
                            dt_ms=config['dt_ms'], bin_ms=5., duration_ms=5000.)
    fields = coupled_execution.field_observations(raw, config)
    values = np.empty((5, len(METRICS))); phase_counts = np.empty((5, total), dtype=np.int64)
    summaries = []; checks = 0
    phase_rows = {}
    if delivered_readouts:
        required = {'field_observations.npz', 'phase_responses.csv', 'phase_unit_counts.npz', 'temporal_readouts.csv'}
        require(required <= set(paths), 'COMPLETE_COUPLED_READOUT_PRODUCTS_REQUIRED')
        with np.load(paths['field_observations.npz'], allow_pickle=False) as stored:
            require(set(stored.files) == set(fields) and all(np.array_equal(stored[key], value)
                    for key, value in fields.items()), 'SOURCE_FIELD_READOUT_CHANGED')
        with paths['temporal_readouts.csv'].open(newline='') as stream: rows = list(csv.DictReader(stream))
        require(len(rows) == len(temporal), 'COMPLETE_TEMPORAL_READOUT_BINS_REQUIRED')
        for actual, expected in zip(rows, temporal):
            require(set(actual) == set(expected), 'TEMPORAL_READOUT_SCHEMA_CHANGED')
            checks += _check_readout(actual, expected)
        with paths['phase_responses.csv'].open(newline='') as stream: rows = list(csv.DictReader(stream))
        require(len(rows) == len(PHASES) and {row['phase'] for row in rows} == set(PHASES),
                'COMPLETE_UNIQUE_SOURCE_PHASES_REQUIRED')
        phase_rows = {row['phase']: row for row in rows}
    for pi, (phase, (start, stop)) in enumerate(PHASES.items()):
        selected = ((raw['spike_ticks'] >= round(start/config['dt_ms']))
                    & (raw['spike_ticks'] < round(stop/config['dt_ms'])))
        counts = np.bincount(raw['spike_unit_indices'][selected], minlength=total)
        phase_counts[pi] = counts
        summary = summarize_counts(counts, ne, config['stimulus_center_deg'], config['stimulus_width_deg'], stop-start)
        summary['all_unit_rate_hz'] = float(counts.sum())/(total*(stop-start)/1000)
        if delivered_readouts:
            row = phase_rows[phase]
            require(row['case_id'] == case['case_id'] and int(row['source_rng_seed']) == case['source_rng_seed']
                    and int(row['units']) == total and int(row['excitatory_neurons']) == ne
                    and int(row['inhibitory_neurons']) == ni
                    and case['case_id'] == f"pfc_{case['source_rng_seed']}_{row['condition']}"
                    and float(row['start_ms']) == start and float(row['stop_ms']) == stop,
                    'SOURCE_PHASE_IDENTITY_OR_DENOMINATOR_CHANGED')
            checks += _check_readout(row, summary)
        summary['vector_x_hz'] = summary['vector_x_spikes']/(ne*(stop-start)/1000)
        summary['vector_y_hz'] = summary['vector_y_spikes']/(ne*(stop-start)/1000)
        values[pi] = [summary[metric] for metric in METRICS]
        summaries.append(dict(phase=phase, start_ms=start, stop_ms=stop, **summary))
    if delivered_readouts:
        with np.load(paths['phase_unit_counts.npz'], allow_pickle=False) as stored:
            require(set(stored.files) == {case['case_id']+'_'+phase for phase in PHASES},
                    'COMPLETE_UNIT_PHASE_VECTORS_REQUIRED')
            for pi, phase in enumerate(PHASES):
                require(np.array_equal(stored[case['case_id']+'_'+phase], phase_counts[pi]),
                        'SOURCE_PHASE_UNIT_COUNTS_CHANGED')
    with np.load(paths['final_rng.npz'], allow_pickle=False) as stored: rng = {key: stored[key] for key in stored.files}
    initial = np.load(paths['initial_voltage.npy'], allow_pickle=False)
    require(initial.shape == (total,) and np.isfinite(initial).all(), 'FULL_SOURCE_INITIAL_STATE_REQUIRED')
    before_cue = raw['spike_ticks'] < round(config['stimulus_start_ms']/config['dt_ms'])
    paired = dict(initial=initial, final_rng=rng, prefix_ticks=raw['spike_ticks'][before_cue],
                  prefix_ids=raw['spike_unit_indices'][before_cue])
    return dict(raw=raw, fields=fields, values=values, phase_counts=phase_counts,
                phase_rows=summaries, paired=paired, semantic_values_verified=checks)


def read_coarse_case(block, project_root, condition):
    config, case, report, _ = coupled_execution.load_coarse_inputs(block, project_root, condition)
    root = project_root/block['coarse_references'][condition]['archive']
    manifest = _source_manifest(json.loads((root/'delivery.json').read_text()))
    paths = bound_case_outputs(root, manifest, Path('source_run'), report)
    result = inspect_case_readouts(paths, config, case, delivered_readouts=False)
    result.update(config=config, case=case, report=report, paths=paths)
    result['paired']['inputs_used'] = report['inputs_used']
    return result


def read_fine_case(block, project_root, root, manifest, dt_ms, condition, coarse, protocol_sha256):
    relative = Path('source_cases')/case_key(dt_ms, condition, block['source_rng_seed'])
    report_path = _source_bound_file(root, manifest, str(relative/'report.json'))
    report = json.loads(report_path.read_text()); config = dict(coarse['config'], dt_ms=dt_ms)
    require(report.get('status') == 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE'
            and report.get('source_execution_complete') is True and report.get('scope') == block['scope']
            and report.get('cohort') == block['cohort'] and report.get('source_rng_seed') == block['source_rng_seed']
            and report.get('refinement_seeds') == block['refinement_seeds']
            and report.get('kernel_parameters') == block['kernel_parameters']
            and report.get('case_id') == coarse['case']['case_id'] and report.get('condition') == condition
            and report.get('dt_ms') == dt_ms and report.get('coarse_dt_ms') == .05
            and report.get('resolved_config') == config and report.get('selected_case') == coarse['case']
            and report.get('coarse_reference') == block['coarse_references'][condition]
            and report.get('coupled_core_units') == config['excitatory_neurons']+config['inhibitory_neurons']
            and report.get('duration_ms') == 5000. and report.get('settings_sha256') == protocol_sha256
            and report.get('numpy_version') == np.__version__
            and report.get('confirmation_execution_released') is False,
            'REGISTERED_COUPLED_GAIN_CASE_BINDING_REQUIRED')
    require(report.get('coarse_final_rng_matches_archived_episode') is True
            and bool(report.get('coarse_current_agreement')) and all(report['coarse_current_agreement'].values())
            and bool(report.get('validations')) and all(row['passed'] is True for row in report['validations'].values())
            and report['inputs_used']['input_mode'] == 'coupled_binomial_clock_refinement_stream',
            'COMPLETE_VALIDATED_COUPLED_CASE_REQUIRED')
    qualification = coupled_execution.qualify_fine_clock(block, project_root, dt_ms, config)
    delivered = report.get('fine_clock_qualification', {})
    require(all(delivered.get(key) == qualification[key] for key in
                ('receipt_sha256', 'delivery_sha256', 'qualification')), 'FINE_CLOCK_QUALIFICATION_BINDING_CHANGED')
    for module in (coupled_execution.executor, coupled_execution.native_drive, coupled_execution.native_execution,
                   coupled_execution.pfc_clock_coupling, coupled_execution.pfc_coupled_drive):
        require(report['code'].get(module.__name__.split('.')[-1]) == sha256_file(Path(module.__file__)),
                'COUPLED_SOURCE_IMPLEMENTATION_CHANGED')
    require(report.get('runner_sha256') == sha256_file(Path(coupled_execution.__file__)),
            'COUPLED_EXECUTION_IMPLEMENTATION_CHANGED')
    paths = bound_case_outputs(root, manifest, relative, report)
    result = inspect_case_readouts(paths, config, coarse['case'], delivered_readouts=True)
    require(np.array_equal(result['paired']['initial'], coarse['paired']['initial']), 'COUPLED_COARSE_INITIAL_STATE_CHANGED')
    rng = result['paired']['final_rng']; expected = coarse['paired']['final_rng']
    require({key.removeprefix('coarse__') for key in rng if key.startswith('coarse__')} == set(expected)
            and all(np.array_equal(rng['coarse__'+key], value) for key, value in expected.items()),
            'COUPLED_COARSE_FINAL_RNG_CHANGED')
    for name in ('units.npz', 'unit_metadata.json'):
        require(sha256_file(paths[name]) == sha256_file(coarse['paths'][name]), 'COUPLED_SOURCE_UNIT_MAP_CHANGED')
    with np.load(paths['applied_current_observation.npz'], allow_pickle=False) as stored:
        actual_current = {key: stored[key] for key in stored.files}
    with np.load(coarse['paths']['applied_current_observation.npz'], allow_pickle=False) as stored:
        coarse_current = {key: stored[key] for key in stored.files}
    require(all(coupled_execution.current_agreement(actual_current, coarse_current, dt_ms, .05).values()),
            'COUPLED_PHYSICAL_CONTROL_CHANGED')
    result.update(config=config, case=coarse['case'], report=report, paths=paths)
    result['paired']['inputs_used'] = report['inputs_used']
    return result


def joint_unit_statistics(cases, condition_index, histograms):
    """One joint chunk traversal: all units, clocks, phases and count windows."""
    config = cases[LABELS[0]]['config']; ne = config['excitatory_neurons']; ni = config['inhibitory_neurons']
    mapping, sizes = unit_mapping(ne, ni, 16); total = ne+ni; bins = 1000
    events = [prepared_events(cases[label]['raw'], total, round(5/dt), bins) for label, dt in zip(LABELS, DT)]
    phase_difference = np.empty((3, 5, total), dtype=np.int64)
    bin_difference = np.empty((3, 5, 3, total), dtype=np.float64)
    reconstructed = np.zeros((3, bins, 17), dtype=np.int64)
    for start, stop, counts in unit_chunks(events, total, bins):
        fields = mapping[start:stop]
        for field in np.unique(fields): reconstructed[:, :, field] += counts[:, :, fields == field].sum(axis=2)
        for pi, (_, (left, right)) in enumerate(PHASES.items()):
            selected = counts[:, round(left/5):round(right/5)].astype(np.int64, copy=False)
            for li, label in enumerate(LABELS):
                require(np.array_equal(selected[li].sum(axis=0), cases[label]['phase_counts'][pi, start:stop]),
                        'STREAMED_PHASE_COUNTS_CHANGED')
                for wi, width in enumerate((5, 50, 100)):
                    pfc_count_histograms.accumulate(histograms[label], selected[li:li+1], fields, width//5,
                                                   condition=condition_index, phase=pi, window=wi)
            for pair, (left_label, right_label) in enumerate(ALL_CLOCK_PAIRS):
                difference = selected[LABELS.index(left_label)]-selected[LABELS.index(right_label)]
                phase_difference[pair, pi, start:stop] = difference.sum(axis=0)
                bin_difference[pair, pi, :, start:stop] = np.stack((difference.mean(axis=0),
                    np.abs(difference).mean(axis=0), np.sqrt(np.square(difference, dtype=np.float64).mean(axis=0))))
    for li, label in enumerate(LABELS):
        require(np.array_equal(reconstructed[li], cases[label]['fields']['field_counts']),
                'STREAMED_UNIT_COUNTS_AND_SOURCE_FIELDS_DISAGREE')
    return phase_difference, bin_difference


def write_gain_block(output, block, cases_by_condition, *, evidence_kind):
    """Consume one size/seed block; no pairing or inference across sizes."""
    output.mkdir(parents=True, exist_ok=False)
    cohort, seed = block['cohort'], block['source_rng_seed']
    ne, ni = block['excitatory_neurons'], block['inhibitory_neurons']; total = ne+ni
    _, sizes = unit_mapping(ne, ni, 16)
    fields = {label: {} for label in LABELS}; values = {label: np.empty((3, 5, len(METRICS))) for label in LABELS}
    histograms = {label: {} for label in LABELS}; paired = {label: {} for label in LABELS}
    phase_delta = np.empty((3, 3, 5, total), dtype=np.int64)
    bin_delta = np.empty((3, 3, 5, 3, total), dtype=np.float64)
    phase_rows = []; inputs = []; checks = []
    for ci, condition in enumerate(CONDITIONS):
        cases = cases_by_condition(condition)
        require(set(cases) == set(LABELS), 'ALL_THREE_REGISTERED_CLOCKS_REQUIRED')
        pd, bd = joint_unit_statistics(cases, ci, histograms)
        phase_delta[:, ci] = pd; bin_delta[:, ci] = bd
        for label, case in cases.items():
            fields[label][condition] = case['fields']['rates_hz']; values[label][ci] = case['values']
            paired[label][case['case']['case_id']] = case['paired']
            phase_rows.extend(dict(cohort=cohort, source_rng_seed=seed, clock=label, condition=condition, **row)
                              for row in case['phase_rows'])
            inputs.append(dict(cohort=cohort, source_rng_seed=seed, clock=label, condition=condition,
                report_sha256=sha256_file(case['paths']['report.json']), semantic_values_verified=case['semantic_values_verified']))
    for label in LABELS:
        rows = _source_pairing(paired[label], {f'pfc_{seed}_{condition}': (seed, condition) for condition in CONDITIONS}, [seed])
        flags = ('exact_pre_cue_events', 'exact_initial_voltage', 'exact_final_source_rng', 'same_source_input_consumption')
        require(all(all(row[key] for key in flags) for row in rows), 'COUPLED_CONDITION_INPUT_PAIRING_FAILED')
        checks.extend(dict(cohort=cohort, clock=label, **row) for row in rows)
    identity = dict(cohort_names=np.asarray([cohort]), source_rng_seeds=np.asarray([seed], dtype=np.int64),
                    evidence_kind=np.asarray(evidence_kind))
    axes = dict(clock_names=np.asarray(LABELS), clock_dt_ms=np.asarray(DT),
        clock_pairs=np.asarray(ALL_CLOCK_PAIRS), condition_names=np.asarray(CONDITIONS),
        phase_names=np.asarray(list(PHASES)), phase_bounds_ms=np.asarray(list(PHASES.values())))
    np.savez_compressed(output/'unit_count_clock_differences.npz', **identity, **axes,
        source_unit_ids=np.arange(total), population_indices=np.r_[np.zeros(ne, dtype=np.int8), np.ones(ni, dtype=np.int8)],
        phase_count_left_minus_right=phase_delta[None, None], bin_count_differences=bin_delta[None, None],
        phase_count_axes=np.asarray(['cohort','source_rng_seed','clock_pair','condition','phase','unit']),
        bin_count_axes=np.asarray(['cohort','source_rng_seed','clock_pair','condition','phase','metric','unit']),
        bin_metric_names=np.asarray(['signed_mean_count','mean_absolute_count','rms_count']),
        phase_count_units=np.asarray('spikes per unit per phase'), bin_count_units=np.asarray('spikes per unit per 5 ms bin'),
        bin_ms=np.asarray(5.), matched_phase_bins=np.asarray([round((stop-start)/5) for start,stop in PHASES.values()]))
    coordinates = {}; observations = {}; denominators = {}; histogram_metadata = {}
    for label in LABELS:
        path = output/(label+'_count_histograms.npz')
        histogram_metadata[label] = pfc_count_histograms.write(path, histograms[label], conditions=CONDITIONS,
            seeds=[seed], phases=PHASES, windows_ms=[5,50,100], field_sizes=sizes)
        with np.load(path, allow_pickle=False) as stored: payload = {key: stored[key] for key in stored.files}
        coordinates[label] = payload['coordinates']; observations[label] = payload['observations']
        denominators[label] = payload['expected_observations'][:, 0]
        # Keep the existing histogram format and its seed axis; add its explicit cohort/clock identity.
        np.savez_compressed(path, **payload, cohort_names=identity['cohort_names'], cohort_index=np.asarray(0),
                            clock_name=np.asarray(label), evidence_kind=np.asarray(evidence_kind))
    support = np.arange(max(int(rows[:,5].max()) for rows in coordinates.values())+1, dtype=np.int64)
    cdfs = {label: pfc_count_histograms.cdf(coordinates[label], observations[label], support, (3,1,5,3,17))[:,0]
            for label in LABELS}
    require(all(np.all(value[..., -1] == 1.) for value in cdfs.values()), 'COMPLETE_OBSERVED_CDF_SUPPORT_REQUIRED')
    np.savez_compressed(output/'source_count_cdfs.npz', **identity, **axes, count_support=support,
        window_ms=np.asarray([5,50,100]), field_indices=np.arange(17), field_unit_counts=sizes,
        cdf=np.stack([cdfs[label] for label in LABELS])[None,None],
        cdf_left_minus_right=np.stack([cdfs[left]-cdfs[right] for left,right in ALL_CLOCK_PAIRS])[None,None],
        expected_observations=np.stack([denominators[label] for label in LABELS])[None,None],
        cdf_axes=np.asarray(['cohort','source_rng_seed','clock','condition','phase','window','field','count_threshold']),
        difference_axes=np.asarray(['cohort','source_rng_seed','clock_pair','condition','phase','window','field','count_threshold']),
        denominator_axes=np.asarray(['cohort','source_rng_seed','clock','condition','phase','window','field']))
    np.savez_compressed(output/'source_phase_responses.npz', **identity, **axes, metric_names=np.asarray(METRICS),
        values=np.stack([values[label] for label in LABELS])[None,None],
        value_axes=np.asarray(['cohort','source_rng_seed','clock','condition','phase','metric']), units=np.asarray('Hz'))
    source_rows, contrast_rows = response_rows({'coupled_inputs': values}, ALL_CLOCK_PAIRS)
    tables = {'field_clock_discrepancies.csv': discrepancy_rows({'coupled_inputs': fields}, ALL_CLOCK_PAIRS),
        'source_response_comparison.csv': source_rows, 'paired_response_comparison.csv': contrast_rows,
        'count_cdf_comparison.csv': count_cdf_rows({'coupled_inputs': cdfs}, support, ALL_CLOCK_PAIRS)}
    for rows in tables.values():
        for row in rows:
            row.update(cohort=cohort, source_rng_seed=seed,
                       left_clock=row.pop('left_cohort'), right_clock=row.pop('right_cohort'))
    tables.update(shared_input_checks=checks, semantic_phase_readouts=phase_rows)
    for name, rows in tables.items():
        for row in rows: row['evidence_kind'] = evidence_kind
        write_rows(output/(name if name.endswith('.csv') else name+'.csv'), rows)
    return dict(cohort=cohort, source_rng_seed=seed, cases=inputs, row_counts={name: len(rows) for name,rows in tables.items()},
                count_histogram_metadata=histogram_metadata, count_support=support.tolist())


def run_recurrent_gain(config_path, project_root, output, source_index):
    settings = yaml.safe_load(config_path.read_text()); _, blocks = analysis_registration(settings, project_root)
    index, archives = indexed_source_archives(source_index, settings, blocks, project_root)
    output.mkdir(parents=True, exist_ok=False); results = []
    evidence_kind = index.get('evidence_kind', 'archived_source_execution')
    for block, (root, manifest, _) in zip(blocks, archives):
        def cases_for_condition(condition):
            coarse = read_coarse_case(block, project_root, condition)
            cases = {LABELS[0]: coarse}
            for label, dt in zip(LABELS[1:], DT[1:]):
                cases[label] = read_fine_case(block, project_root, root, manifest, dt, condition, coarse, settings['protocol_sha256'])
            return cases
        results.append(write_gain_block(output/block['cohort']/str(block['source_rng_seed']), block,
                                        cases_for_condition, evidence_kind=evidence_kind))
    rows = {}
    for result in results:
        for name, count in result['row_counts'].items(): rows[name] = rows.get(name,0)+count
    report = dict(status='COMPLETE_DESCRIPTIVE_RECURRENT_GAIN_CLOCK_COMPARISON',
        scope=settings['scope'], settings_sha256=sha256_file(config_path), protocol_sha256=settings['protocol_sha256'],
        source_index_sha256=sha256_file(source_index), source_archives=index['blocks'], source_evidence_kind=evidence_kind,
        blocks=results, row_counts=rows, complete_source_episodes_consumed=9*len(blocks), coarse_episodes_reused=3*len(blocks),
        coupled_fine_episodes_consumed=6*len(blocks), original_independent_blocks_per_size=len(blocks)//2, source_simulations=0,
        confirmation_execution_released=False, confidence_intervals=False, scientific_fidelity_qualified=False,
        production_episodes=0, cross_size_pairing=False,
        interpretation='Descriptive clock sensitivity under coupled source-input laws. All controls, fields, units, phases and complete empirical count support retained; no CI, p value, threshold, confirmation or numerical-equivalence claim.',
        implementation_sha256=sha256_file(Path(__file__)),
        outputs={str(path.relative_to(output)):dict(bytes=path.stat().st_size,sha256=sha256_file(path))
                 for path in output.rglob('*') if path.is_file()})
    if evidence_kind == 'synthetic_unit_fixture':
        report.update(status='COMPLETE_SYNTHETIC_RECURRENT_GAIN_READOUT_VALIDATION', scientific_result=False,
                      complete_source_episodes_consumed=0, coarse_episodes_reused=0, coupled_fine_episodes_consumed=0,
                      synthetic_cases_consumed=9*len(blocks), original_independent_blocks_per_size=0)
    freeze_json(output/'report.json', report)
    return report


def run(config_path, project_root, output, source_index=None):
    schema = yaml.safe_load(config_path.read_text()).get('schema')
    if schema in GAIN_ANALYSIS_REGISTRATIONS:
        return run_recurrent_gain(config_path, project_root, output, source_index)
    require(source_index is None, 'SOURCE_INDEX_REQUIRES_RECURRENT_GAIN_ANALYSIS')
    return run_legacy(config_path, project_root, output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-index',type=Path);args=parser.parse_args()
    report=run(args.config,args.project_root,args.output,args.source_index)
    print(json.dumps({key:report[key] for key in ('status','row_counts','pathwise_rms_change_rows') if key in report}));return 0


if __name__=='__main__':raise SystemExit(main())

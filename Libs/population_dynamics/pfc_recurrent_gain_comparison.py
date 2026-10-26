"""Descriptive paired effects of the source-supported local recurrent kernel gain.

Runs only inside the qualified parameterization overlay. Separate source RNG
blocks pair parameters within a size; sizes remain independent. No inference or
parameter-domain qualification is performed.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import yaml
from . import pfc_scale_analysis as source, pfc_count_histograms as histograms
from .pfc_delay_analysis import PHASES, require, write_json, write_rows
from .pfc_fluctuation_analysis import (
    unit_mapping, prepared_events, unit_chunks, unit_moments, field_covariance, population_summary,
)
from .pfc_response_analysis import summarize_counts
from .provenance import sha256_file

CONDITIONS = ('omission', 'cue', 'strong_cue')
LABELS = ('reference_2560', 'expanded_100k')
PARAMETERS = ('default', 'candidate')
WINDOWS = (5., 50., 100.)
METRICS = ('excitatory_rate_hz','inhibitory_rate_hz','all_unit_rate_hz','cue_region_rate_hz',
           'noncue_region_rate_hz','cue_selectivity_hz','vector_x_hz','vector_y_hz')
CONTRASTS = (('cue_minus_omission',1,0),('strong_cue_minus_omission',2,0),('strong_cue_minus_cue',2,1))
PHYSICS = ('scope','expected_source_revision','excitatory_neurons','inhibitory_neurons','weight_scaling_factor',
           'dt_ms','duration_ms','stimulus_center_deg','stimulus_width_deg','stimulus_strength_nA',
           'stimulus_start_ms','stimulus_duration_ms','distractor_center_deg','distractor_width_deg',
           'distractor_start_ms','distractor_duration_ms','state_monitored_per_population',
           'excitatory_preference_bins','readout_bin_ms','synapse_execution')


def validate_input_pair(before, after):
    require(hasattr(source, 'kernel_parameters'), 'QUALIFIED_ISOLATED_PARAMETERIZATION_OVERLAY_REQUIRED')
    left, right = before['resolved_config'], after['resolved_config']
    require(all(left[key] == right[key] for key in PHYSICS)
            and before['selected_case'] == after['selected_case']
            and before['case_id'] == after['case_id']
            and before['dimensions'] == after['dimensions'], 'MATCHED_PHYSICAL_CASE_AND_SIZE_REQUIRED')
    require(source.kernel_parameters.resolve_kernel_parameters(left) == dict(sigma_weight_profile=20.,Jpos_excit2excit=1.6)
            and source.kernel_parameters.resolve_kernel_parameters(right) == dict(sigma_weight_profile=20.,Jpos_excit2excit=2.3),
            'ONLY_REGISTERED_LOCAL_KERNEL_GAIN_CHANGE_REQUIRED')
    require(before['driver'] == after['driver'] and before['inputs_used'] == after['inputs_used']
            and before['code']['drive_sha256'] == after['code']['drive_sha256'],
            'EXACT_INITIAL_BACKGROUND_DRIVER_AND_CONSUMPTION_PAIR_REQUIRED')
    return dict(case_id=before['case_id'],source_rng_seed=before['selected_case']['source_rng_seed'],
                paired_initial_background_input_law=True,dimensions=before['dimensions'])


def selected_blocks(available, requested):
    available, requested = list(available), list(requested)
    require(len(available) == len(set(available)) and len(requested) == len(set(requested)) and len(requested)>=2
            and set(requested) <= set(available), 'EXACT_UNIQUE_SOURCE_BLOCKS_REQUIRED')
    return np.asarray([available.index(seed) for seed in requested],dtype=np.int64)


def select_histogram(data, seeds):
    require(tuple(data['condition_names']) == CONDITIONS and tuple(data['phase_names']) == tuple(PHASES)
            and np.array_equal(data['window_ms'],WINDOWS), 'UNCHANGED_COMPLETE_HISTOGRAM_AXES_REQUIRED')
    selected = selected_blocks(data['source_rng_seeds'],seeds)
    lookup = {int(old):new for new,old in enumerate(selected)}
    result = {}
    for coordinate, number in zip(data['coordinates'],data['observations']):
        if int(coordinate[1]) in lookup:
            key = list(map(int,coordinate));key[1] = lookup[key[1]]
            require(tuple(key) not in result, 'DUPLICATE_HISTOGRAM_COORDINATE')
            result[tuple(key)] = int(number)
    require(result, 'EMPTY_SELECTED_HISTOGRAM')
    return result


def describe(values):
    values = np.asarray(values,dtype=float)
    require(values.ndim==1 and len(values)>=2, 'AT_LEAST_TWO_BLOCK_DESCRIPTIVE_DENOMINATOR_REQUIRED')
    complete = bool(np.isfinite(values).all())
    return dict(blocks=len(values),defined_blocks=int(np.isfinite(values).sum()),
                mean=float(values.mean()) if complete else None,
                block_sd=float(values.std(ddof=1)) if complete else None)


def independent_effect(left, right, left_seeds, right_seeds):
    require(len(left_seeds)==len(set(left_seeds))==len(left) and len(right_seeds)==len(set(right_seeds))==len(right),
            'EXACT_PER_SIZE_BLOCK_DENOMINATORS_REQUIRED')
    require(set(left_seeds).isdisjoint(right_seeds), 'CROSS_SIZE_BLOCKS_MUST_REMAIN_INDEPENDENT')
    a,b = describe(left),describe(right)
    return dict(reference_mean_effect=a['mean'],reference_block_sd=a['block_sd'],
                expanded_mean_effect=b['mean'],expanded_block_sd=b['block_sd'],
                expanded_minus_reference_mean_effect=None if a['mean'] is None or b['mean'] is None else b['mean']-a['mean'],
                reference_blocks=len(left_seeds),expanded_blocks=len(right_seeds),cross_size_blocks_paired=False)


def response_arrays(counts, config, seeds):
    require(len(seeds)>=2 and len(seeds)==len(set(seeds)), 'UNIQUE_RESPONSE_SOURCE_BLOCKS_REQUIRED')
    shape=(len(CONDITIONS),len(seeds),len(PHASES));values=np.empty((*shape,len(METRICS)))
    direction=np.zeros(shape,dtype=bool);angles=np.full(shape,np.nan);resultant=np.full(shape,np.nan);rows=[]
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons']
    for ci,condition in enumerate(CONDITIONS):
        for bi,seed in enumerate(seeds):
            case=f'pfc_{seed}_{condition}'
            for pi,(phase,(start,stop)) in enumerate(PHASES.items()):
                count=counts[case+'_'+phase]
                require(count.shape==(ne+ni,) and count.dtype.kind in 'iu' and np.all(count>=0), 'COMPLETE_ALL_UNIT_PHASE_COUNTS_REQUIRED')
                result=summarize_counts(count,ne,config['stimulus_center_deg'],config['stimulus_width_deg'],stop-start)
                result['all_unit_rate_hz']=float(count.sum())/((ne+ni)*(stop-start)/1000)
                result['vector_x_hz']=result['vector_x_spikes']/(ne*(stop-start)/1000)
                result['vector_y_hz']=result['vector_y_spikes']/(ne*(stop-start)/1000)
                values[ci,bi,pi]=[result[name] for name in METRICS]
                direction[ci,bi,pi]=result['direction_defined']
                if result['decoded_angle_deg'] is not None:angles[ci,bi,pi]=result['decoded_angle_deg']
                if result['resultant_length'] is not None:resultant[ci,bi,pi]=result['resultant_length']
                rows.append(dict(case_id=case,source_rng_seed=seed,condition=condition,phase=phase,**result))
    return dict(values=values,direction_defined=direction,decoded_angle_deg=angles,resultant_length=resultant),rows


def _npz(path):
    with np.load(path,allow_pickle=False) as stored:return {key:stored[key] for key in stored.files}


def register_source_config(config_path, cohort, parameter):
    """Register this development comparison without changing archived purpose.

    The numerical-calibration registrar remains unchanged. Default archives can
    contain a larger registered cohort, while the source protocol fixes the exact
    three selected blocks; candidate archives must contain exactly those blocks.
    """
    config=yaml.safe_load(Path(config_path).read_text())
    require(parameter in PARAMETERS and isinstance(config,dict), 'REGISTERED_GAIN_PARAMETER_REQUIRED')
    fixed=dict(scope='source_reproduction',expected_source_revision=source.SOURCE_REVISION,
        synapse_execution='factorized_uniform_projections',dt_ms=.05,duration_ms=5000.,
        stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_start_ms=100.,stimulus_duration_ms=200.,
        readout_bin_ms=5.,excitatory_preference_bins=16)
    require(all(config.get(key)==value for key,value in fixed.items()), 'REGISTERED_GAIN_SOURCE_PHYSICS_REQUIRED')
    ne,ni,scale=config['excitatory_neurons'],config['inhibitory_neurons'],config['weight_scaling_factor']
    require(type(ne)is int and type(ni)is int and ne>0 and ni>0 and ne+ni==cohort['coupled_core_units']
            and type(scale)in (int,float) and np.isfinite(scale) and scale>0
            and np.isclose(ne*scale,2048.,rtol=0,atol=1e-10)
            and np.isclose(ni*scale,512.,rtol=0,atol=1e-10), 'REGISTERED_GAIN_SOURCE_SIZE_NORMALIZATION_REQUIRED')
    expected_kernel=dict(sigma_weight_profile=20.,Jpos_excit2excit=1.6 if parameter=='default'else 2.3)
    require(source.kernel_parameters.resolve_kernel_parameters(config)==expected_kernel, 'REGISTERED_GAIN_KERNEL_REQUIRED')
    cases=config.get('cases')
    require(isinstance(cases,list)and cases and all(isinstance(row,dict)for row in cases), 'REGISTERED_GAIN_CASES_REQUIRED')
    seeds=sorted({row.get('source_rng_seed')for row in cases if type(row.get('source_rng_seed'))is int})
    require(len(seeds)>=3 and all(0<=seed<2**32 for seed in seeds), 'REGISTERED_GAIN_SOURCE_SEEDS_REQUIRED')
    registered={f'pfc_{seed}_{condition}':(seed,condition)for seed in seeds for condition in CONDITIONS}
    require(len(cases)==len(registered)and {row.get('case_id')for row in cases}==set(registered), 'COMPLETE_REGISTERED_GAIN_CONTROL_ENVELOPE_REQUIRED')
    for row in cases:
        seed,condition=registered[row['case_id']]
        require(row['source_rng_seed']==seed and row.get('stimulus_strength_nA')==source.CONDITIONS[condition]
                and row.get('distractor_strength_nA')==0., 'REGISTERED_GAIN_CONDITION_CURRENT_REQUIRED')
    selected={row['case_id']for row in cohort['default_sources']}
    require(len(cohort['source_rng_seeds'])==3 and len(selected)==9
            and selected=={f'pfc_{seed}_{condition}'for seed in cohort['source_rng_seeds']for condition in CONDITIONS}
            and selected<=set(registered), 'EXACT_MAIN_PROTOCOL_THREE_BLOCK_CASE_SET_REQUIRED')
    replication=config.get('source_replication')
    if parameter=='candidate':
        require(sha256_file(config_path)==cohort['config_sha256'] and set(registered)==selected
                and seeds==cohort['source_rng_seeds'], 'EXACT_FROZEN_GAIN_SOURCE_CONFIG_REQUIRED')
    if replication is None:
        require(parameter=='default'and seeds==cohort['source_rng_seeds'], 'LEGACY_DEFAULT_THREE_BLOCK_REGISTRATION_REQUIRED')
    else:
        require(isinstance(replication,dict)
                and replication.get('purpose')==('source_parameter_domain_development'if parameter=='candidate'else'numerical_calibration')
                and replication.get('independent_unit')=='joint_initial_background_rng_block'
                and replication.get('joint_rng_seeds')==seeds, 'SOURCE_PURPOSE_AND_FULL_BLOCK_DECLARATION_REQUIRED')
    source.geometry(ne)
    dimensions=dict(n_e=ne,n_i=ni,coupled_core_units=ne+ni,dt_seconds=.05/1000,duration_ticks=100000,
                    duration_seconds=5.,weight_scaling_factor=scale,N_E_times_scale=ne*scale,N_I_times_scale=ni*scale)
    return config,registered,seeds,dimensions


def prepare_headers(index, label, protocol):
    cohorts={}
    for parameter in PARAMETERS:
        headers={};registration=None
        for entry in index['sources'][label][parameter]:
            archive=Path(entry['path']);config_path=archive/'source_run/config.yaml'
            config,registered,seeds,dimensions=register_source_config(config_path,protocol['cohorts'][label],parameter)
            status,header=source._source_header(archive,config,registered,sha256_file(config_path))
            require(header is not None and status['status']=='COMPLETE' and status['case_id']==entry['case_id']
                    and status['report_sha256']==entry['report_sha256'] and status['delivery_sha256']==entry['delivery_sha256'],
                    'BOUND_COMPLETE_SOURCE_CASE_REQUIRED')
            require(status['case_id'] not in headers,'DUPLICATE_SOURCE_CASE')
            if registration is not None:require(registration[0]==config,'SAME_PARAMETER_COHORT_CONFIG_REQUIRED')
            registration=(config,registered,seeds,dimensions,sha256_file(config_path));headers[status['case_id']]=header
        require(set(headers)=={row['case_id']for row in protocol['cohorts'][label]['default_sources']}, 'ALL_EXACT_PROTOCOL_GAIN_CASE_HEADERS_REQUIRED')
        cohorts[parameter]=dict(headers=headers,registration=registration)
    require(set(cohorts['default']['headers'])==set(cohorts['candidate']['headers']), 'PAIRED_PARAMETER_CASE_SETS_REQUIRED')
    checks=[validate_input_pair(cohorts['default']['headers'][case]['report'],cohorts['candidate']['headers'][case]['report'])
            for case in cohorts['default']['headers']]
    return cohorts,checks


def validate_native_cache_bridge(bridge, fluctuation, phase, headers):
    expected=set(headers)
    require(len(expected)==9 and bridge.get('status')=='PASS_STREAMED_SOURCE_EQUIVALENCE'
            and bridge.get('passed_cases')==bridge.get('registered_cases')==9
            and all(row['status']=='PASS' and row['spike_arrays_exact'] for row in bridge['cases']),
            'COMPLETE_NATIVE_NUMPY_SPIKE_PARITY_REQUIRED')
    source_bindings={row['case_id']:row for row in bridge['source_bindings']}
    native_bindings={row['case_id']:row for row in fluctuation['source_bindings']}
    require(set(source_bindings)==set(native_bindings)==expected, 'NATIVE_CACHE_BRIDGE_CASES_CHANGED')
    for case in expected:
        a,b=source_bindings[case],native_bindings[case]
        require(all(a[key]==headers[case]['binding'][key] for key in ('report_sha256','delivery_sha256'))
                and a['native_raw_sha256']==b['raw_sha256']
                and b['native_report_sha256']==phase['native_report_sha256'],
                'NATIVE_MOMENT_CACHE_SOURCE_PARITY_CHANGED')


def analyze_cohort(cohort,label,parameter,seeds,cache,output):
    # JSON input indexes serialize paths as strings; normalize at this boundary.
    cache = {name: Path(path) for name, path in cache.items()}
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    config,registered,_,dimensions,config_sha=cohort['registration'];total=dimensions['coupled_core_units']
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons'];mapping,sizes=unit_mapping(ne,ni,16)
    require(len(cohort['headers'])==9 and set(cohort['headers'])=={f'pfc_{seed}_{c}'for seed in seeds for c in CONDITIONS},'COMPLETE_THREE_BLOCK_CONTROL_ENVELOPE_REQUIRED')
    cached_fields=cached_units=cached_hist=None;phase_counts={}
    if parameter=='default':
        cached_fields=_npz(cache['fields']);indices=selected_blocks(cached_fields['source_rng_seeds'],seeds)
        cached_fields={k:(v[indices] if k.endswith('_field_counts') else v)for k,v in cached_fields.items()}
        with np.load(cache['phase_counts'],allow_pickle=False) as stored:
            for case in cohort['headers']:
                for phase in PHASES:phase_counts[case+'_'+phase]=stored[case+'_'+phase]
        if label=='reference_2560':
            proof={key:json.loads(Path(cache[key]).read_text()) for key in ('bridge','fluctuation_report','phase_report')}
            validate_native_cache_bridge(proof['bridge'],proof['fluctuation_report'],proof['phase_report'],cohort['headers'])
            require(sha256_file(cache['moments'])==proof['fluctuation_report']['outputs']['unit_moments.npz']['sha256']
                    and sha256_file(cache['fields'])==proof['fluctuation_report']['outputs']['field_fluctuations.npz']['sha256']
                    and sha256_file(cache['phase_counts'])==proof['phase_report']['artifacts']['phase_unit_counts.npz'], 'NATIVE_CACHED_OUTPUT_BINDING_CHANGED')
            require(np.array_equal(_npz(cache['fields'])['source_rng_seeds'],seeds),'NATIVE_MOMENTS_REQUIRE_EXACT_THREE_BLOCKS')
            cached_units=_npz(cache['moments'])
        else:cached_hist=select_histogram(_npz(cache['histograms']),seeds)
    histogram={} if cached_hist is None else cached_hist
    moments={};fields={};summaries=[];bindings=[];pair_arrays={};field_rates=np.empty((3,3,5,17))
    for ci,condition in enumerate(CONDITIONS):
        events=[];direct=[]
        for seed in seeds:
            case=f'pfc_{seed}_{condition}';header=cohort['headers'][case]
            raw,common,pair=source._source_values(header,config,registered,dimensions,config_sha)
            pair_arrays[case]=pair
            if parameter=='candidate':
                _,_,vectors,_=source._source_phase_rows(raw,case,registered,config,dimensions);phase_counts.update(vectors)
            else:
                for phase,(start,stop)in PHASES.items():
                    chosen=(raw['spike_ticks']>=round(start/config['dt_ms']))&(raw['spike_ticks']<round(stop/config['dt_ms']))
                    require(np.array_equal(np.bincount(raw['spike_unit_indices'][chosen],minlength=total),phase_counts[case+'_'+phase]),'CACHED_DEFAULT_PHASE_COUNTS_CHANGED')
            events.append(prepared_events(raw,total,100,1000));ids,bins=events[-1]
            direct.append(np.bincount(bins*17+mapping[ids],minlength=1000*17).reshape(1000,17))
            bindings.append(header['binding']);del raw
        group_counts=np.stack(direct)
        if cached_fields is not None:require(np.array_equal(group_counts,cached_fields[condition+'_field_counts']),'SELECTED_DEFAULT_FIELD_TRAJECTORIES_CHANGED')
        means=np.empty((5,total));variances=np.empty_like(means);counted_fields=np.zeros_like(group_counts)
        if cached_units is not None:
            means[:]=cached_units[condition+'_mean_counts'];variances[:]=cached_units[condition+'_variance_counts']
        for left,right,counts in unit_chunks(events,total,1000):
            for group in np.unique(mapping[left:right]):counted_fields[:,:,group]+=counts[:,:,mapping[left:right]==group].sum(2)
            for pi,(phase,(start,stop))in enumerate(PHASES.items()):
                selected=counts[:,round(start/5):round(stop/5)]
                if cached_units is None:means[pi,left:right],variances[pi,left:right]=unit_moments(selected)
                if cached_hist is None:
                    for wi,width in enumerate(WINDOWS):histograms.accumulate(histogram,selected,mapping[left:right],round(width/5),condition=ci,phase=pi,window=wi)
                for bi,seed in enumerate(seeds):require(np.array_equal(selected[bi].sum(0),phase_counts[f'pfc_{seed}_{condition}_{phase}'][left:right]),'UNIT_BIN_AND_PHASE_COUNTS_DISAGREE')
        require(np.array_equal(counted_fields,group_counts),'ALL_UNIT_FIELD_AGGREGATION_DISAGREES')
        moments[condition+'_mean_counts']=means;moments[condition+'_variance_counts']=variances
        fields[condition+'_field_counts']=group_counts;covariances=[]
        for pi,(phase,(start,stop))in enumerate(PHASES.items()):
            current=group_counts[:,round(start/5):round(stop/5)]
            cov=cached_fields[condition+'_field_covariance_counts'][pi] if cached_units is not None else field_covariance(current)
            covariances.append(cov);field_rates[ci,:,pi]=current.sum(1)/sizes/((stop-start)/1000)
            for population,groups in [('E',list(range(16))),('I',[16]),('all',list(range(17)))]:
                summaries.append(dict(condition=condition,phase=phase,population=population,joint_rng_blocks=3,
                    **population_summary(means[pi],variances[pi],current,cov,mapping,groups)))
        fields[condition+'_field_covariance_counts']=np.asarray(covariances)
        del events,counts
    condition_pairing=source._source_pairing(pair_arrays,registered,seeds)
    require(all(all(row[key] for key in ('exact_pre_cue_events','exact_initial_voltage','exact_final_source_rng','same_source_input_consumption'))for row in condition_pairing),'SOURCE_CONDITION_INPUT_PAIRING_FAILED')
    np.savez_compressed(output/'phase_unit_counts.npz',**phase_counts)
    np.savez_compressed(output/'unit_moments.npz',unit_ids=np.arange(total),unit_field_indices=mapping,**moments)
    np.savez_compressed(output/'field_fluctuations.npz',source_rng_seeds=np.asarray(seeds),field_unit_counts=sizes,**fields)
    histograms.write(output/'unit_count_histograms.npz',histogram,conditions=CONDITIONS,seeds=seeds,phases=PHASES,windows_ms=WINDOWS,field_sizes=sizes)
    write_rows(output/'population_variance.csv',summaries)
    responses,rows=response_arrays(phase_counts,config,seeds);write_rows(output/'responses.csv',rows)
    kernel=source.kernel_parameters.source_kernel(ne,config);parameters=source.kernel_parameters.resolve_kernel_parameters(config)
    report=dict(status='COMPLETE_DESCRIPTIVE_GAIN_COHORT',cohort=label,parameter=parameter,source_rng_seeds=seeds,
        cases=9,units=total,dt_ms=.05,duration_ms=5000.,source_bindings=bindings,condition_pairing=condition_pairing,
        kernel_parameters=parameters,Jneg=source.kernel_parameters._normalization(parameters['sigma_weight_profile'],parameters['Jpos_excit2excit']),
        actual_discrete_kernel_mean=float(kernel.mean()),kernel_peak=float(kernel.max()),kernel_minimum=float(kernel.min()),weight_scaling_factor=config['weight_scaling_factor'],
        kernel_interpretation='Local E-E peak changes; source-derived Jneg maintains continuum mean normalization. Global conductance scaling is unchanged.',
        source_rounded_cue_geometry=source.geometry(ne)[1],native_cached_moments_reused=cached_units is not None,
        cached_default_fields_and_phases_reused=parameter=='default',expanded_histograms_selected_by_block=cached_hist is not None,
        unit_variance_denominator_blocks=3,source_simulations=0,production_episodes=0,scientific_parameter_domain_qualified=False,
        confidence_intervals=False,p_values=False,outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p))for p in output.iterdir()if p.is_file()})
    write_json(output/'report.json',report)
    return dict(responses=responses,field_rates=field_rates,histogram=histogram,moments=moments,variance=summaries,report=report,pairs=pair_arrays)


def compare_results(results,seeds,output):
    payload=dict(condition_names=np.asarray(CONDITIONS),phase_names=np.asarray(list(PHASES)),metric_names=np.asarray(METRICS),window_ms=np.asarray(WINDOWS))
    response_effects={};field_effects={};contrast_effects={};rows=[];angles=[];cross=[];field_rows=[];field_cross=[];variance_rows=[];unit_differences={}
    for label in LABELS:
        before,after=[results[label][parameter]for parameter in PARAMETERS]
        require(all(item['responses']['values'].shape==(len(CONDITIONS),len(seeds[label]),len(PHASES),len(METRICS))
                    and item['field_rates'].shape==(len(CONDITIONS),len(seeds[label]),len(PHASES),17)
                    for item in (before,after)), 'PAIRED_RESULT_BLOCK_AXES_REQUIRED')
        payload[label+'_source_rng_seeds']=np.asarray(seeds[label])
        for parameter,item in zip(PARAMETERS,[before,after]):
            for key,value in item['responses'].items():payload[label+'_'+parameter+'_'+key]=value
            payload[label+'_'+parameter+'_field_rates_hz']=item['field_rates']
        delta=after['responses']['values']-before['responses']['values'];response_effects[label]=delta
        field_delta=after['field_rates']-before['field_rates'];field_effects[label]=field_delta
        payload[label+'_paired_parameter_effects_hz']=delta
        payload[label+'_paired_field_effects_hz']=field_delta
        contrasts=np.asarray([delta[pos]-delta[neg]for _,pos,neg in CONTRASTS]);contrast_effects[label]=contrasts
        payload[label+'_parameter_effects_on_condition_contrasts_hz']=contrasts
        both=before['responses']['direction_defined']&after['responses']['direction_defined']
        angle=np.where(both,(after['responses']['decoded_angle_deg']-before['responses']['decoded_angle_deg']+180)%360-180,np.nan)
        payload[label+'_paired_angle_defined']=both;payload[label+'_paired_angle_change_deg']=angle
        for ci,condition in enumerate(CONDITIONS):
            for pi,phase in enumerate(PHASES):
                for mi,metric in enumerate(METRICS):rows.append(dict(cohort=label,condition=condition,phase=phase,metric=metric,units='Hz',**describe(delta[ci,:,pi,mi])))
                for field in range(17):field_rows.append(dict(cohort=label,condition=condition,phase=phase,field_index=field,population='I' if field==16 else 'E',units='Hz',**describe(field_delta[ci,:,pi,field])))
                for bi,seed in enumerate(seeds[label]):angles.append(dict(cohort=label,source_rng_seed=seed,condition=condition,phase=phase,
                    default_direction_defined=bool(before['responses']['direction_defined'][ci,bi,pi]),candidate_direction_defined=bool(after['responses']['direction_defined'][ci,bi,pi]),
                    paired_angle_change_deg=float(angle[ci,bi,pi])if both[ci,bi,pi]else None))
        for name in before['moments']:unit_differences[label+'_'+name]=after['moments'][name]-before['moments'][name]
        for a,b in zip(before['variance'],after['variance']):
            require(all(a[key]==b[key]for key in ('condition','phase','population','units','joint_rng_blocks')),'VARIANCE_COMPONENT_AXES_CHANGED')
            for metric in a:
                if metric in ('condition','phase','population','units','joint_rng_blocks'):continue
                variance_rows.append(dict(cohort=label,condition=a['condition'],phase=a['phase'],population=a['population'],metric=metric,
                    default=a[metric],candidate=b[metric],candidate_minus_default=None if a[metric]is None or b[metric]is None else b[metric]-a[metric],joint_rng_blocks=a['joint_rng_blocks']))
    for ci,condition in enumerate(CONDITIONS):
        for pi,phase in enumerate(PHASES):
            for mi,metric in enumerate(METRICS):cross.append(dict(condition=condition,phase=phase,metric=metric,units='Hz',
                **independent_effect(response_effects[LABELS[0]][ci,:,pi,mi],response_effects[LABELS[1]][ci,:,pi,mi],seeds[LABELS[0]],seeds[LABELS[1]])))
            for field in range(17):field_cross.append(dict(condition=condition,phase=phase,field_index=field,units='Hz',
                **independent_effect(field_effects[LABELS[0]][ci,:,pi,field],field_effects[LABELS[1]][ci,:,pi,field],seeds[LABELS[0]],seeds[LABELS[1]])))
    contrast_rows=[];contrast_cross=[]
    for ci,(name,_,_)in enumerate(CONTRASTS):
        for pi,phase in enumerate(PHASES):
            for mi,metric in enumerate(METRICS):
                for label in LABELS:contrast_rows.append(dict(cohort=label,contrast=name,phase=phase,metric=metric,units='Hz',**describe(contrast_effects[label][ci,:,pi,mi])))
                contrast_cross.append(dict(contrast=name,phase=phase,metric=metric,units='Hz',**independent_effect(contrast_effects[LABELS[0]][ci,:,pi,mi],contrast_effects[LABELS[1]][ci,:,pi,mi],seeds[LABELS[0]],seeds[LABELS[1]])))
    np.savez_compressed(output/'response_and_field_effects.npz',**payload)
    np.savez_compressed(output/'paired_all_unit_moment_differences.npz',**unit_differences)
    for name,table in [('paired_response_effects',rows),('cross_size_response_effects',cross),('paired_angle_changes',angles),
        ('paired_field_effects',field_rows),('cross_size_field_effects',field_cross),('variance_component_changes',variance_rows),
        ('paired_condition_contrast_effects',contrast_rows),('cross_size_condition_contrast_effects',contrast_cross)]:write_rows(output/(name+'.csv'),table)
    threshold=np.arange(max(max(key[5]for key in results[l][p]['histogram'])for l in LABELS for p in PARAMETERS)+1,dtype=np.int64)
    cdf_payload=dict(count_thresholds=threshold,condition_names=np.asarray(CONDITIONS),phase_names=np.asarray(list(PHASES)),window_ms=np.asarray(WINDOWS))
    cdf_effect={};cdf_rows=[];cdf_cross=[]
    for label in LABELS:
        for parameter in PARAMETERS:
            table=results[label][parameter]['histogram'];keys=sorted(table)
            cdf_payload[label+'_'+parameter]=histograms.cdf(np.asarray(keys,dtype=np.int64),np.asarray([table[k]for k in keys],dtype=np.int64),threshold,(len(CONDITIONS),len(seeds[label]),len(PHASES),len(WINDOWS),17))
        cdf_effect[label]=cdf_payload[label+'_candidate']-cdf_payload[label+'_default'];cdf_payload[label+'_paired_parameter_effect']=cdf_effect[label]
        cdf_payload[label+'_source_rng_seeds']=np.asarray(seeds[label])
    for ci,condition in enumerate(CONDITIONS):
        for pi,phase in enumerate(PHASES):
            for wi,width in enumerate(WINDOWS):
                for field in range(17):
                    for ki,k in enumerate(threshold):
                        identity=dict(condition=condition,phase=phase,window_ms=width,field_index=field,count_threshold=int(k))
                        for label in LABELS:cdf_rows.append(dict(cohort=label,**identity,**describe(cdf_effect[label][ci,:,pi,wi,field,ki])))
                        cdf_cross.append(dict(**identity,**independent_effect(cdf_effect[LABELS[0]][ci,:,pi,wi,field,ki],cdf_effect[LABELS[1]][ci,:,pi,wi,field,ki],seeds[LABELS[0]],seeds[LABELS[1]])))
    np.savez_compressed(output/'complete_empirical_cdf_effects.npz',**cdf_payload)
    write_rows(output/'paired_cdf_effects.csv',cdf_rows);write_rows(output/'cross_size_cdf_effects.csv',cdf_cross)
    return dict(response_rows=len(rows),field_rows=len(field_rows),cdf_rows=len(cdf_rows),complete_observed_integer_count_thresholds=threshold.tolist())


def run(config_path,input_index,output):
    settings=yaml.safe_load(Path(config_path).read_text())
    require(settings.get('schema')=='pfc_recurrent_gain_comparison_v1' and settings.get('scope')=='descriptive_parameter_development'
            and all(settings.get(k)is False for k in ('scientific_parameter_domain_qualified','confidence_intervals','p_values','confirmation_execution_released')),
            'DESCRIPTIVE_UNQUALIFIED_PARAMETER_COMPARISON_ONLY')
    protocol_path=Path(config_path).resolve().parents[2]/settings['source_protocol']
    require(sha256_file(protocol_path)==settings['source_protocol_sha256'], 'BOUND_SOURCE_PROTOCOL_REQUIRED')
    protocol=yaml.safe_load(protocol_path.read_text())
    index=json.loads(Path(input_index).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    results={};checks=[];seeds={};status='FAILED'
    try:
        for label in LABELS:
            cohorts,paired=prepare_headers(index,label,protocol);checks.extend(dict(cohort=label,**row)for row in paired)
            seeds[label]=settings['source_rng_seeds'][label]
            require(seeds[label]==([93001,93002,93003]if label==LABELS[0]else[95001,95002,95003]),'FROZEN_INDEPENDENT_SIZE_BLOCKS_REQUIRED')
            results[label]={}
            for parameter in PARAMETERS:
                results[label][parameter]=analyze_cohort(cohorts[parameter],label,parameter,seeds[label],index['cache'][label],output/(label+'_'+parameter))
                print(json.dumps(dict(status='COHORT_READOUT_COMPLETE',cohort=label,parameter=parameter)),flush=True)
            for case in cohorts['default']['headers']:
                a,b=[results[label][p]['pairs'][case]for p in PARAMETERS]
                require(np.array_equal(a['initial'],b['initial'])and set(a['final_rng'])==set(b['final_rng'])and all(np.array_equal(a['final_rng'][k],b['final_rng'][k])for k in a['final_rng']), 'ACTUAL_CROSS_PARAMETER_INITIAL_AND_FINAL_RNG_MISMATCH')
        counts=compare_results(results,seeds,output);status='COMPLETE_DESCRIPTIVE_PAIRED_RECURRENT_GAIN_COMPARISON'
        report=dict(status=status,settings=settings,settings_sha256=sha256_file(config_path),input_pairing=checks,
            source_rng_seeds=seeds,independent_blocks_per_size=3,within_size_parameters_paired=True,cross_size_blocks_paired=False,
            scientific_parameter_domain_qualified=False,confirmation_execution_released=False,confidence_intervals=False,p_values=False,
            source_simulations=0,production_episodes=0,comparison=counts,wall_seconds=time.monotonic()-started,
            interpretation='Descriptive local recurrent-kernel parameter effects at two independent sizes. Signed changes, controls, undefined angles and all source units are retained. Persistence does not establish bistability; these results do not qualify production, numerical clocks or uncertainty.',
            implementation_sha256=sha256_file(Path(__file__)),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p))for p in output.iterdir()if p.is_file()})
    except Exception as error:
        write_json(output/'report.json',dict(status=status,error=str(error),error_type=type(error).__name__,input_pairing=checks,source_simulations=0,scientific_parameter_domain_qualified=False));raise
    write_json(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps({'status':run(args.config,args.inputs,args.output)['status']}));return 0

if __name__=='__main__':raise SystemExit(main())

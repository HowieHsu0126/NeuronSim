"""Fixed source-block replication; pilot3, new29 and full32 remain separate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import yaml
from . import pfc_recurrent_gain_comparison as gain
from .pfc_delay_analysis import require, write_json, write_rows
from .provenance import sha256_file

source=gain.source
PARTITIONS={'new29':np.arange(3,32,dtype=np.int64),'full32':np.arange(32,dtype=np.int64)}


def source_registration(config_path,entry,protocol):
    """Admit frozen mixed pilot/new/full configs without relabeling purposes."""
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text())
    require(sha256_file(config_path)==entry['config_sha256'],'EXACT_REPLICATION_SOURCE_CONFIG_REQUIRED')
    size,parameter=entry['size'],entry['parameter']
    require(size in (2560,100000)and parameter in gain.PARAMETERS and entry['origin']in('new','existing'),'REGISTERED_REPLICATION_SOURCE_ROLE_REQUIRED')
    full=protocol['reference_full_rng_seeds'if size==2560 else'expanded_full_rng_seeds']
    if entry['origin']=='new':
        family=('reference'if size==2560 else'expanded')+'_'+parameter
        declaration=protocol['cohorts'][family];seeds=declaration['new_source_rng_seeds'];purpose='source_parameter_domain_development'
        require(entry['config_sha256']==declaration['config_sha256'],'FROZEN_NEW_BLOCK_CONFIG_REQUIRED')
    elif size==100000 and parameter=='default':seeds=full;purpose='numerical_calibration'
    else:seeds=full[:3];purpose=None if parameter=='default'else'source_parameter_domain_development'
    fixed=dict(scope='source_reproduction',expected_source_revision=source.SOURCE_REVISION,synapse_execution='factorized_uniform_projections',
        dt_ms=.05,duration_ms=5000.,stimulus_center_deg=120.,stimulus_width_deg=30.,stimulus_start_ms=100.,stimulus_duration_ms=200.,readout_bin_ms=5.,excitatory_preference_bins=16)
    require(all(config.get(k)==v for k,v in fixed.items()),'UNCHANGED_REPLICATION_SOURCE_PHYSICS_REQUIRED')
    ne,ni,scale=config['excitatory_neurons'],config['inhibitory_neurons'],config['weight_scaling_factor']
    require(type(ne)is int and type(ni)is int and ne>0 and ni>0 and ne+ni==size and type(scale)in(int,float)and np.isfinite(scale)and scale>0
            and np.isclose(ne*scale,2048.,rtol=0,atol=1e-10)and np.isclose(ni*scale,512.,rtol=0,atol=1e-10),'UNCHANGED_REPLICATION_POPULATION_NORMALIZATION_REQUIRED')
    require(source.kernel_parameters.resolve_kernel_parameters(config)==dict(sigma_weight_profile=20.,Jpos_excit2excit=1.6 if parameter=='default'else 2.3),'EXACT_REPLICATION_KERNEL_REQUIRED')
    registered={f'pfc_{seed}_{condition}':(seed,condition)for seed in seeds for condition in gain.CONDITIONS}
    cases=config['cases'];require(len(cases)==len(registered)and {r['case_id']for r in cases}==set(registered),'FULL_SOURCE_CONFIG_CASE_SET_REQUIRED')
    for row in cases:
        seed,condition=registered[row['case_id']]
        require(row['source_rng_seed']==seed and row['stimulus_strength_nA']==source.CONDITIONS[condition]and row['distractor_strength_nA']==0.,'SOURCE_CONTROL_OR_BLOCK_CHANGED')
    replication=config.get('source_replication')
    if purpose is None:require(replication is None,'PILOT_DEFAULT_SOURCE_PURPOSE_CHANGED')
    else:require(isinstance(replication,dict)and replication.get('purpose')==purpose and replication.get('independent_unit')=='joint_initial_background_rng_block'
                 and replication.get('joint_rng_seeds')==seeds,'ORIGINAL_REPLICATION_PURPOSE_AND_BLOCKS_REQUIRED')
    require(entry['case_id']in registered and registered[entry['case_id']][0]==entry['source_rng_seed'],'REGISTERED_REPLICATION_CASE_REQUIRED')
    dimensions=dict(n_e=ne,n_i=ni,coupled_core_units=size,dt_seconds=.05/1000,duration_ticks=100000,duration_seconds=5.,weight_scaling_factor=scale,N_E_times_scale=ne*scale,N_I_times_scale=ni*scale)
    return config,registered,seeds,dimensions,entry['config_sha256']


def validate_bank(entries,existing,protocol):
    frozen={(r['size'],r['parameter'],r['case_id']):r for r in existing}
    require(len(existing)==len(frozen)==123,'EXACT_PREPARED_EXISTING_SOURCE_BANK_REQUIRED')
    actual={};new=0
    for entry in entries:
        key=entry['size'],entry['parameter'],entry['case_id'];require(key not in actual,'DUPLICATE_REPLICATION_SOURCE_CASE');actual[key]=entry
        if key in frozen:require(entry['origin']=='existing'and all(entry[k]==frozen[key][k]for k in ('report_sha256','delivery_sha256','config_sha256','source_rng_seed')),'PREPARED_SOURCE_BINDING_CHANGED')
        else:require(entry['origin']=='new','UNKNOWN_UNDECLARED_SOURCE_ORIGIN');new+=1
    expected={(size,p,f'pfc_{seed}_{c}')for size,bank in [(2560,protocol['reference_full_rng_seeds']),(100000,protocol['expanded_full_rng_seeds'])]for p in gain.PARAMETERS for seed in bank for c in gain.CONDITIONS}
    require(len(actual)==384 and set(actual)==expected and new==261 and set(frozen)<=set(actual),'COMPLETE_FIXED_384_CASE_REPLICATION_BANK_REQUIRED')
    return actual


def prepare_headers(entries,existing,protocol):
    bank=validate_bank(entries,existing,protocol);headers={};registrations={}
    for key,entry in bank.items():
        archive=Path(entry['path']);registration=source_registration(archive/'source_run/config.yaml',entry,protocol)
        config,registered,_,_,sha=registration;status,header=source._source_header(archive,config,registered,sha)
        require(header is not None and status['status']=='COMPLETE'and status['case_id']==entry['case_id']and all(status[k]==entry[k]for k in ('report_sha256','delivery_sha256')),'COMPLETE_BOUND_REPLICATION_SOURCE_HEADER_REQUIRED')
        headers[key]=header;registrations[key]=registration
    paired=[]
    for size in (2560,100000):
        for case in sorted(key[2]for key in headers if key[:2]==(size,'default')):
            paired.append(dict(size=size,**gain.validate_input_pair(headers[size,'default',case]['report'],headers[size,'candidate',case]['report'])))
    return headers,registrations,paired


def bin_counts(raw,total_units,*,total_bins=1000,bin_ticks=100):
    ticks,ids=raw['spike_ticks'],raw['spike_unit_indices']
    require(ticks.dtype.kind in 'iu'and ids.dtype.kind in 'iu'and ticks.shape==ids.shape and ticks.ndim==1 and np.all((ticks>=0)&(ticks<total_bins*bin_ticks))and np.all((ids>=0)&(ids<total_units)),'EXACT_LEGAL_SOURCE_EVENT_BINS_REQUIRED')
    counts=np.bincount((ticks//bin_ticks)*total_units+ids,minlength=total_bins*total_units).reshape(total_bins,total_units)
    require(int(counts.max(initial=0))<=np.iinfo(np.uint16).max,'TEMPORARY_COUNT_STORAGE_OVERFLOW')
    return counts.astype(np.uint16)


def matrix_chunks(matrices,chunk_units=1024):
    require(len(matrices)>=2 and chunk_units>0 and all(m.shape==matrices[0].shape and m.dtype==np.uint16 for m in matrices),'ALIGNED_TEMPORARY_BLOCK_COUNT_MATRICES_REQUIRED')
    for left in range(0,matrices[0].shape[1],chunk_units):
        right=min(left+chunk_units,matrices[0].shape[1])
        yield left,right,np.stack([m[:,left:right]for m in matrices]).astype(np.int64)


def partition_moments(counts,*,full_cached=False):
    require(counts.ndim==3 and len(counts)==32 and counts.dtype==np.int64,'EXACT_32_BLOCK_WIDENED_MOMENT_INPUT_REQUIRED')
    result={'new29':gain.unit_moments(counts[3:])}
    if not full_cached:result['full32']=gain.unit_moments(counts)
    return result


def subset_histogram(table,indices):
    reverse={int(old):new for new,old in enumerate(indices)}
    require(len(reverse)==len(indices)and len(indices)>=2,'UNIQUE_HISTOGRAM_PARTITION_BLOCKS_REQUIRED')
    return {(key[0],reverse[key[1]],*key[2:]):value for key,value in table.items()if key[1]in reverse}


def load_pilot(root,label,parameter):
    folder=Path(root)/(label+'_'+parameter);report=json.loads((folder/'report.json').read_text())
    require(report['status']=='COMPLETE_DESCRIPTIVE_GAIN_COHORT'and report['unit_variance_denominator_blocks']==3,'COMPLETE_THREE_BLOCK_PILOT_CACHE_REQUIRED')
    for name,binding in report['outputs'].items():
        path=folder/name;require(path.stat().st_size==binding['bytes']and sha256_file(path)==binding['sha256'],'PILOT_CACHE_OUTPUT_CHANGED')
    return dict(report=report,phase_counts=gain._npz(folder/'phase_unit_counts.npz'),fields=gain._npz(folder/'field_fluctuations.npz'),moments=gain._npz(folder/'unit_moments.npz'),
                histogram=gain.select_histogram(gain._npz(folder/'unit_count_histograms.npz'),report['source_rng_seeds']))


def analyze_bank_cohort(headers,registrations,label,parameter,seeds,pilot,expanded_cache,output,scratch):
    size=2560 if label=='reference_2560' else 100000
    first=(size,parameter,f'pfc_{seeds[0]}_{gain.CONDITIONS[0]}');config=registrations[first][0]
    ne,ni=config['excitatory_neurons'],config['inhibitory_neurons'];mapping,sizes=gain.unit_mapping(ne,ni,16)
    require(np.array_equal(pilot['fields']['source_rng_seeds'],seeds[:3]),'EXACT_EXPOSED_PILOT_BLOCK_ORDER_REQUIRED')
    phase_counts=dict(pilot['phase_counts']);histogram=dict(pilot['histogram']);cached_fields=pilot['fields'];full_cache=None
    for binding in pilot['report']['source_bindings']:
        key=(size,parameter,binding['case_id']);require(all(headers[key]['binding'][name]==binding[name]for name in ('report_sha256','delivery_sha256')),'PILOT_SOURCE_BINDING_CHANGED')
    if label=='expanded_100k'and parameter=='default':
        full_cache=gain._npz(expanded_cache['moments']);cached_fields=gain._npz(expanded_cache['fields'])
        require(np.array_equal(cached_fields['source_rng_seeds'],seeds),'EXACT_32_BLOCK_DEFAULT_CACHE_REQUIRED')
        phase_counts=gain._npz(expanded_cache['phase_counts']);histogram=gain.select_histogram(gain._npz(expanded_cache['histograms']),seeds)
    full_moments={};new_moments={};fields={};pairs={};source_bindings=[];decoded_cases=0
    for ci,condition in enumerate(gain.CONDITIONS):
        with tempfile.TemporaryDirectory(prefix=f'{label}-{parameter}-{condition}-',dir=scratch)as directory:
            matrices=[]
            for seed in seeds:
                case=f'pfc_{seed}_{condition}';key=(size,parameter,case);header=headers[key]
                local_config,registered,_,dimensions,config_sha=registrations[key]
                raw,_,pair=source._source_values(header,local_config,registered,dimensions,config_sha);pairs[case]=pair
                matrix=bin_counts(raw,size);require(np.array_equal(matrix.sum(0,dtype=np.int64),raw['native_spike_counts']),'TEMPORARY_FULL_UNIT_EVENT_TOTALS_CHANGED')
                for phase,(start,stop)in gain.PHASES.items():
                    value=matrix[round(start/5):round(stop/5)].sum(0,dtype=np.int64);name=case+'_'+phase
                    if name in phase_counts:require(np.array_equal(value,phase_counts[name]),'CACHED_SOURCE_PHASE_COUNTS_CHANGED')
                    else:phase_counts[name]=value
                path=Path(directory)/(case+'.npy');np.save(path,matrix,allow_pickle=False);matrices.append(np.load(path,mmap_mode='r',allow_pickle=False))
                decoded_cases+=1;source_bindings.append(header['binding']);del raw,matrix
            counted_fields=np.zeros((32,1000,17),dtype=np.int64);new_hist={}
            means={name:np.empty((5,size))for name in PARTITIONS};variances={name:np.empty((5,size))for name in PARTITIONS}
            if full_cache is not None:
                means['full32'][:]=full_cache[condition+'_mean_counts'];variances['full32'][:]=full_cache[condition+'_variance_counts']
            for left,right,counts in matrix_chunks(matrices):
                for field in np.unique(mapping[left:right]):counted_fields[:,:,field]+=counts[:,:,mapping[left:right]==field].sum(2,dtype=np.int64)
                for pi,(phase,(start,stop))in enumerate(gain.PHASES.items()):
                    selected=counts[:,round(start/5):round(stop/5)]
                    for scope,(mean,variance)in partition_moments(selected,full_cached=full_cache is not None).items():means[scope][pi,left:right]=mean;variances[scope][pi,left:right]=variance
                    if full_cache is None:
                        for wi,width in enumerate(gain.WINDOWS):gain.histograms.accumulate(new_hist,selected[3:],mapping[left:right],round(width/5),condition=ci,phase=pi,window=wi)
                    for bi,seed in enumerate(seeds):require(np.array_equal(selected[bi].sum(0,dtype=np.int64),phase_counts[f'pfc_{seed}_{condition}_{phase}'][left:right]),'CHUNK_AND_SOURCE_PHASE_COUNTS_DISAGREE')
            cached=cached_fields[condition+'_field_counts'];require(np.array_equal(counted_fields[:len(cached)],cached),'CACHED_BLOCK_FIELD_TRAJECTORIES_CHANGED')
            fields[condition+'_field_counts']=counted_fields
            full_moments[condition+'_mean_counts']=means['full32'];full_moments[condition+'_variance_counts']=variances['full32']
            new_moments[condition+'_mean_counts']=means['new29'];new_moments[condition+'_variance_counts']=variances['new29']
            for key,value in new_hist.items():histogram[(key[0],key[1]+3,*key[2:])]=value
            del counts,matrices
        print(json.dumps(dict(status='CONDITION_COUNTS_COMPLETE',cohort=label,parameter=parameter,condition=condition,raw_cases_decoded=32)),flush=True)
    checks=source._source_pairing(pairs,{case:None for case in pairs},seeds)
    require(all(all(row[name]for name in ('exact_pre_cue_events','exact_initial_voltage','exact_final_source_rng','same_source_input_consumption'))for row in checks),'FULL_BANK_CONDITION_PAIRING_FAILED')
    results={}
    for scope,indices in PARTITIONS.items():
        target=Path(output)/scope/(label+'_'+parameter);target.mkdir(parents=True,exist_ok=False)
        chosen=[seeds[int(i)]for i in indices];moments=new_moments if scope=='new29'else full_moments;selected_fields={};summaries=[]
        selected_counts={f'pfc_{seed}_{c}_{phase}':phase_counts[f'pfc_{seed}_{c}_{phase}']for seed in chosen for c in gain.CONDITIONS for phase in gain.PHASES}
        field_rates=np.empty((3,len(indices),5,17))
        for ci,condition in enumerate(gain.CONDITIONS):
            group=fields[condition+'_field_counts'][indices];selected_fields[condition+'_field_counts']=group;covariances=[]
            for pi,(phase,(start,stop))in enumerate(gain.PHASES.items()):
                current=group[:,round(start/5):round(stop/5)]
                cov=cached_fields[condition+'_field_covariance_counts'][pi]if full_cache is not None and scope=='full32'else gain.field_covariance(current)
                covariances.append(cov);field_rates[ci,:,pi]=current.sum(1,dtype=np.int64)/sizes/((stop-start)/1000)
                for population,groups in [('E',list(range(16))),('I',[16]),('all',list(range(17)))]:
                    summaries.append(dict(condition=condition,phase=phase,population=population,joint_rng_blocks=len(indices),**gain.population_summary(moments[condition+'_mean_counts'][pi],moments[condition+'_variance_counts'][pi],current,cov,mapping,groups)))
            selected_fields[condition+'_field_covariance_counts']=np.asarray(covariances)
        hist=subset_histogram(histogram,indices);responses,rows=gain.response_arrays(selected_counts,config,chosen)
        np.savez_compressed(target/'phase_unit_counts.npz',**selected_counts);np.savez_compressed(target/'unit_moments.npz',unit_ids=np.arange(size),unit_field_indices=mapping,**moments)
        np.savez_compressed(target/'field_fluctuations.npz',source_rng_seeds=np.asarray(chosen),field_unit_counts=sizes,**selected_fields)
        gain.histograms.write(target/'unit_count_histograms.npz',hist,conditions=gain.CONDITIONS,seeds=chosen,phases=gain.PHASES,windows_ms=gain.WINDOWS,field_sizes=sizes)
        write_rows(target/'responses.csv',rows);write_rows(target/'population_variance.csv',summaries)
        wanted={f'pfc_{seed}_{c}'for seed in chosen for c in gain.CONDITIONS}
        report=dict(status='COMPLETE_DESCRIPTIVE_REPLICATE_COHORT',partition=scope,cohort=label,parameter=parameter,source_rng_seeds=chosen,cases=3*len(chosen),units=size,
            unit_variance_denominator_blocks=len(chosen),source_bindings=[r for r in source_bindings if r['case_id']in wanted],condition_pairing=[r for r in checks if r['source_rng_seed']in chosen],
            raw_decode_cases_in_shared_pass=decoded_cases,full32_default_moments_reused=full_cache is not None and scope=='full32',pilot_readouts_reused=True,
            covariance_estimator='center across blocks at each matched time bin; denominator n-1; average within phase',temporary_count_storage='checked uint16; int64 reductions and float64 sample covariance',
            scientific_parameter_domain_qualified=False,confirmation_execution_released=False,production_episodes=0,confidence_intervals=False,p_values=False,activation_threshold_introduced=False,
            outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p))for p in target.iterdir()if p.is_file()})
        write_json(target/'report.json',report);results[scope]=dict(responses=responses,field_rates=field_rates,histogram=hist,moments=moments,variance=summaries,report=report)
    return results,pairs


def run(config_path,index_path,output):
    config_path=Path(config_path);settings=yaml.safe_load(config_path.read_text())
    require(settings.get('schema')=='pfc_recurrent_gain_replicate_comparison_v1'and settings.get('scope')=='descriptive_fixed_replication'
            and all(settings.get(k)is False for k in ('scientific_parameter_domain_qualified','confirmation_execution_released','confidence_intervals','p_values','activation_threshold_introduced')),'DESCRIPTIVE_FIXED_REPLICATION_ONLY')
    root=config_path.resolve().parents[2];protocol_path=root/settings['source_protocol']
    require(sha256_file(protocol_path)==settings['source_protocol_sha256'],'FROZEN_REPLICATION_PROTOCOL_REQUIRED')
    protocol=yaml.safe_load(protocol_path.read_text());index=json.loads(Path(index_path).read_text())
    require(protocol['full_blocks_per_size']==32 and protocol['new_blocks_per_size']==29 and protocol['new_source_cases']==261,'UNCHANGED_FIXED_REPLICATION_BUDGET_REQUIRED')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic();completed=[]
    try:
        pilot=Path(index['pilot_root']);require(sha256_file(pilot/'report.json')==protocol['pilot_readout_sha256'],'EXACT_COMPLETED_PILOT_READOUT_REQUIRED')
        existing=json.loads(Path(index['existing_sources']).read_text());headers,registrations,input_checks=prepare_headers(index['sources'],existing,protocol)
        shutil.copytree(pilot,output/'pilot3');results={scope:{label:{} for label in gain.LABELS}for scope in PARTITIONS};all_pairs={}
        seeds={'reference_2560':protocol['reference_full_rng_seeds'],'expanded_100k':protocol['expanded_full_rng_seeds']};scratch=Path(index['scratch']);scratch.mkdir(parents=True,exist_ok=True)
        for label in gain.LABELS:
            for parameter in gain.PARAMETERS:
                cached=load_pilot(pilot,label,parameter);result,pairs=analyze_bank_cohort(headers,registrations,label,parameter,seeds[label],cached,index['expanded_cache'],output,scratch)
                all_pairs[label,parameter]=pairs
                for scope in PARTITIONS:results[scope][label][parameter]=result[scope]
                completed.append(dict(cohort=label,parameter=parameter,source_cases=96))
            for case in all_pairs[label,'default']:
                a,b=all_pairs[label,'default'][case],all_pairs[label,'candidate'][case]
                require(np.array_equal(a['initial'],b['initial'])and set(a['final_rng'])==set(b['final_rng'])and all(np.array_equal(a['final_rng'][k],b['final_rng'][k])for k in a['final_rng']),'ACTUAL_REPLICATE_PARAMETER_INPUT_PAIR_FAILED')
            del all_pairs[label,'default'],all_pairs[label,'candidate']
        comparisons={}
        for scope,indices in PARTITIONS.items():
            selected={label:[seeds[label][int(i)]for i in indices]for label in gain.LABELS};comparisons[scope]=gain.compare_results(results[scope],selected,output/scope)
        report=dict(status='COMPLETE_DESCRIPTIVE_FIXED_GAIN_REPLICATION',settings=settings,completed=completed,source_cases=384,reused_source_cases=123,new_source_cases=261,source_simulations=0,
            raw_case_decodes=384,condition_group_chunk_traversals=12,partitions=dict(pilot3='unchanged exposed pilot artifacts reused',new29='new independent blocks separate',full32='pilot and new blocks combined descriptively'),
            pilot_report_sha256=protocol['pilot_readout_sha256'],input_pairing=input_checks,comparisons=comparisons,
            scientific_parameter_domain_qualified=False,confirmation_execution_released=False,production_episodes=0,confidence_intervals=False,p_values=False,activation_threshold_introduced=False,
            training_seeds_unchanged=[41,42,43],confirmation_seed_bank_untouched=[96001,96128],
            interpretation='Continuous descriptive replication; independent sizes and paired parameters/conditions. Exposed pilot prompted replication. No significance, activation, bistability, clock equivalence or domain qualification.',
            wall_seconds=time.monotonic()-started,implementation_sha256=sha256_file(Path(__file__)))
    except Exception as error:
        write_json(output/'report.json',dict(status='FAILED_FIXED_REPLICATION_READOUT',error=str(error),completed=completed,source_simulations=0,scientific_parameter_domain_qualified=False));raise
    write_json(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps({'status':run(args.config,args.inputs,args.output)['status']}));return 0

if __name__=='__main__':raise SystemExit(main())

"""Descriptive matched-clock PFC size comparison with independent source blocks.

Each size retains its own source units, field masses and seed axis. Conditions
are paired only within a size. No cross-size neuron or stochastic-path pairing
and no numerical-equivalence or biological-replicate claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_count_histograms, pfc_phase_responses
from .pfc_delay_analysis import PHASES, require, write_json, write_rows
from .pfc_fluctuation_comparison import CONDITIONS, PHYSICAL_KEYS, METRICS, lag_covariance, load_cohort
from .provenance import sha256_file
from .pfc_scale_analysis import geometry

LABELS = ('reference_2560', 'expanded_100k')


def validate_cohorts(cohorts, settings):
    require(tuple(cohorts) == LABELS, 'EXPLICIT_TWO_SIZE_COHORTS_REQUIRED')
    reference = cohorts[LABELS[0]]
    ref_binding = reference['report']['common_execution_binding']
    seed_sets = []
    for label, cohort in cohorts.items():
        config, report = cohort['config'], cohort['report']; expected = settings['cohorts'][label]
        require(report['joint_rng_blocks'] == expected['source_rng_seeds']
                and report['coupled_core_units'] == expected['coupled_units']
                and config['dt_ms'] == settings['dt_ms'] == .0125
                and config['readout_bin_ms'] == settings['bin_ms'] == 5.
                and config['duration_ms'] == 5000. and config['excitatory_preference_bins'] == 16
                and config['excitatory_neurons']*config['weight_scaling_factor'] == 2048.
                and config['inhibitory_neurons']*config['weight_scaling_factor'] == 512.,
                'EXACT_REGISTERED_SIZE_CLOCK_AND_BLOCKS_REQUIRED')
        require(all(config[key] == reference['config'][key] for key in PHYSICAL_KEYS)
                and config['synapse_execution'] == reference['config']['synapse_execution'] == 'factorized_uniform_projections',
                'MATCHED_SOURCE_PHYSICS_REQUIRED')
        binding = report['common_execution_binding']
        require(binding['backend'] == ref_binding['backend'] == 'numpy'
                and binding['code'] == ref_binding['code']
                and binding['qualification_sha256'] == ref_binding['qualification_sha256']
                and binding['qualification_delivery_sha256'] == ref_binding['qualification_delivery_sha256'],
                'MATCHED_QUALIFIED_NUMPY_SOURCE_IMPLEMENTATION_REQUIRED')
        driver, ref_driver = binding['driver'], ref_binding['driver']
        require(driver['numpy_version'] == '2.4.2'
                and all(driver[key] == ref_driver[key] for key in ('source_entry_sha256','rng','random_stream_policy','binomial_n',
                    'binomial_probability','dt_seconds','external_rate_Hz','initial_draw_order','packed_draw_order','total_ticks')),
                'MATCHED_SOURCE_INPUT_LAW_REQUIRED')
        require(driver['n_e'] == config['excitatory_neurons'] and driver['n_i'] == config['inhibitory_neurons']
                and cohort['phase_responses'] is not None and cohort['count_histograms'] is not None,
                'COMPLETE_OWN_SIZE_RESPONSES_AND_CDFS_REQUIRED')
        seed_sets.append(set(report['joint_rng_blocks']))
    require(not seed_sets[0].intersection(seed_sets[1]), 'INDEPENDENT_SIZE_COHORT_SEEDS_REQUIRED')


def summary_difference(a, b):
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    require(a.ndim == b.ndim == 1 and len(a) >= 2 and len(b) >= 2
            and np.isfinite(a).all() and np.isfinite(b).all(), 'SEPARATE_FINITE_WITHIN_SIZE_BLOCK_VALUES_REQUIRED')
    return dict(left_blocks=len(a), right_blocks=len(b), left_mean=float(a.mean()), right_mean=float(b.mean()),
                left_block_sd=float(a.std(ddof=1)), right_block_sd=float(b.std(ddof=1)),
                left_minus_right=float(a.mean()-b.mean()), cross_size_pairing=False)


def compare_cdfs(cohorts, windows):
    require(windows == [5,50,100], 'REGISTERED_ALL_UNIT_COUNT_WINDOWS_REQUIRED')
    hist = {label: cohort['count_histograms'] for label, cohort in cohorts.items()}
    require(all(np.array_equal(item['windows_ms'],windows) for item in hist.values()), 'COMPLETE_PER_SIZE_COUNT_WINDOWS_REQUIRED')
    maximum = max(int(item['coordinates'][:,5].max()) for item in hist.values())
    support = np.arange(maximum+1, dtype=np.int64)
    cdfs = {label:pfc_count_histograms.cdf(item['coordinates'],item['observations'],support,item['shape']) for label,item in hist.items()}
    payload = dict(count_support=support,condition_names=np.asarray(CONDITIONS),phase_names=np.asarray(list(PHASES)),
                   window_ms=np.asarray(windows),field_indices=np.arange(17),units=np.asarray('probability'))
    for label, cohort in cohorts.items():
        seeds = cohort['report']['joint_rng_blocks']; sizes = cohort['sizes']
        denominators = np.asarray([[round((stop-start)/width)*sizes for width in windows] for start,stop in PHASES.values()], dtype=np.int64)
        payload.update({label+'_cdf':cdfs[label],label+'_source_rng_seeds':np.asarray(seeds,dtype=np.int64),
                        label+'_field_unit_counts':sizes,label+'_unit_window_denominators':denominators})
    left, right = LABELS; rows=[]; distances=[]
    for ci,condition in enumerate(CONDITIONS):
        for pi,phase in enumerate(PHASES):
            for wi,width in enumerate(windows):
                for fi in range(17):
                    context=dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,window_ms=width,field_index=fi,
                        left_field_units=int(cohorts[left]['sizes'][fi]),right_field_units=int(cohorts[right]['sizes'][fi]),
                        left_unit_window_denominator=int(payload[left+'_unit_window_denominators'][pi,wi,fi]),
                        right_unit_window_denominator=int(payload[right+'_unit_window_denominators'][pi,wi,fi]))
                    for ki,threshold in enumerate(support):
                        rows.append(dict(context,count_threshold=int(threshold),**summary_difference(cdfs[left][ci,:,pi,wi,fi,ki],cdfs[right][ci,:,pi,wi,fi,ki])))
                    delta=cdfs[left][ci,:,pi,wi,fi].mean(0)-cdfs[right][ci,:,pi,wi,fi].mean(0)
                    distances.append(dict(context,max_abs_empirical_mean_cdf_difference=float(np.max(np.abs(delta))),status='DESCRIPTIVE_ONLY'))
    return rows,distances,payload


def compare_responses(cohorts):
    payload=dict(condition_names=np.asarray(CONDITIONS),phase_names=np.asarray(list(PHASES)),metric_names=np.asarray(pfc_phase_responses.METRICS),
                 contrast_names=np.asarray([name for name,_,_ in pfc_phase_responses.CONTRASTS]),units=np.asarray('Hz'))
    for label,cohort in cohorts.items():
        for name in ('values','direction_defined','decoded_angle_deg','resultant_length'):
            payload[label+'_'+name]=cohort['phase_responses'][name]
        values=cohort['phase_responses']['values']
        payload[label+'_source_rng_seeds']=np.asarray(cohort['report']['joint_rng_blocks'],dtype=np.int64)
        payload[label+'_paired_condition_values']=np.asarray([values[CONDITIONS.index(pos)]-values[CONDITIONS.index(neg)]
                                                           for _,pos,neg in pfc_phase_responses.CONTRASTS])
    left,right=LABELS;rows=[];paired=[]
    for ci,condition in enumerate(CONDITIONS):
        for pi,phase in enumerate(PHASES):
            for mi,metric in enumerate(pfc_phase_responses.METRICS):
                rows.append(dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,metric=metric,units='Hz',
                    **summary_difference(payload[left+'_values'][ci,:,pi,mi],payload[right+'_values'][ci,:,pi,mi])))
    for ci,(contrast,_,_) in enumerate(pfc_phase_responses.CONTRASTS):
        for pi,phase in enumerate(PHASES):
            for mi,metric in enumerate(pfc_phase_responses.METRICS):
                paired.append(dict(left_cohort=left,right_cohort=right,contrast=contrast,phase=phase,metric=metric,units='Hz',
                    within_size_conditions_paired=True,**summary_difference(payload[left+'_paired_condition_values'][ci,:,pi,mi],
                                                                           payload[right+'_paired_condition_values'][ci,:,pi,mi])))
    return rows,paired,payload


def compare(reference_archive, expanded_archive, config_path, output):
    settings=yaml.safe_load(Path(config_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result=dict(status='RUNNING',new_source_simulations=0,production_episodes=0,scientific_fidelity_qualified=False,
                working_memory_qualified=False,finite_size_scaling_qualified=False,confidence_intervals=False,p_values=False,q_values=False,
                numerical_equivalence_qualified=False,confirmation_execution_released=False,cross_size_neuron_pairing=False,cross_size_noise_pairing=False,
                within_size_conditions_paired=True,independent_circuits_per_size=1)
    try:
        require(settings['schema']=='pfc_independent_matched_clock_size_comparison_v1' and settings['scope']=='development_calibration'
                and settings['inference']=='descriptive_only' and settings['lag_ms']==[0,5,10,100]
                and settings['count_windows_ms']==[5,50,100] and settings['cross_size_pairing'] is False,
                'EXPLICIT_INDEPENDENT_SIZE_DESCRIPTION_REQUIRED')
        cohorts={LABELS[0]:load_cohort(reference_archive,registered_size=2560),LABELS[1]:load_cohort(expanded_archive,registered_size=100000)}
        validate_cohorts(cohorts,settings)
        covariance_payload=dict(field_indices=np.arange(17),units=np.asarray('Hz^2'));rates_payload=dict(units=np.asarray('Hz'),time_start_ms=np.arange(1000)*5.)
        covariances={};field_rows=[];cov_rows=[];population_rows=[]
        left,right=LABELS
        for label,cohort in cohorts.items():
            scale=cohort['sizes']*.005
            for payload in (covariance_payload,rates_payload):
                payload[label+'_source_rng_seeds']=np.asarray(cohort['report']['joint_rng_blocks'],dtype=np.int64)
                payload[label+'_field_unit_counts']=cohort['sizes']
            for condition in CONDITIONS:
                rates=cohort['fields'][condition+'_field_counts']/scale
                rates_payload[label+'_'+condition]=rates
                for pi,(phase,(begin,end)) in enumerate(PHASES.items()):
                    part=rates[:,round(begin/5):round(end/5)]
                    for lag_ms in settings['lag_ms']:
                        covariance=lag_covariance(part,round(lag_ms/5));covariances[label,condition,phase,lag_ms]=covariance
                        if covariance is not None:covariance_payload[f'{label}_{condition}_{phase}_lag_{lag_ms}ms']=covariance
                        if lag_ms==0:
                            original=cohort['fields'][condition+'_field_covariance_counts'][pi]/np.outer(scale,scale)
                            require(np.allclose(covariance,original,rtol=1e-10,atol=1e-12),'SOURCE_ZERO_LAG_COVARIANCE_ROUNDTRIP_REQUIRED')
        for condition in CONDITIONS:
            for phase,(begin,end) in PHASES.items():
                values=[rates_payload[label+'_'+condition][:,round(begin/5):round(end/5)].mean(1) for label in LABELS]
                for field in range(17):
                    field_rows.append(dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,field_index=field,units='Hz',
                        left_field_units=int(cohorts[left]['sizes'][field]),right_field_units=int(cohorts[right]['sizes'][field]),
                        **summary_difference(values[0][:,field],values[1][:,field])))
                for lag_ms in settings['lag_ms']:
                    a,b=(covariances[label,condition,phase,lag_ms]for label in LABELS)
                    require((a is None)==(b is None),'MATCHED_WITHIN_PHASE_LAG_DOMAIN_REQUIRED')
                    for i in range(17):
                        for j in range(17):
                            cov_rows.append(dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,lag_ms=lag_ms,
                                leading_field=i,trailing_field=j,units='Hz^2',left_value=None if a is None else float(a[i,j]),
                                right_value=None if b is None else float(b[i,j]),left_minus_right=None if a is None else float(a[i,j]-b[i,j]),
                                status='UNDEFINED_PHASE_CONTAINED_LAG' if a is None else 'DEFINED'))
        for key,left_values in cohorts[left]['rows'].items():
            condition,phase,population=key
            for metric in METRICS:
                a,b=left_values[metric],cohorts[right]['rows'][key][metric]
                population_rows.append(dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,population=population,
                    metric=metric,left_value=a,right_value=b,left_minus_right=None if a is None or b is None else a-b,
                    status='UNDEFINED' if a is None or b is None else 'DEFINED'))
        cdf_rows,cdf_distances,cdf_payload=compare_cdfs(cohorts,settings['count_windows_ms'])
        responses,paired,response_payload=compare_responses(cohorts)
        tables={'field_mean_size_differences.csv':field_rows,'field_covariance_size_differences.csv':cov_rows,
                'population_size_differences.csv':population_rows,'count_cdf_size_differences.csv':cdf_rows,
                'count_cdf_descriptive_distances.csv':cdf_distances,'source_response_size_differences.csv':responses,
                'paired_response_size_differences.csv':paired}
        for name,rows in tables.items():write_rows(output/name,rows)
        arrays={'field_rates.npz':rates_payload,'lagged_field_covariance_hz2.npz':covariance_payload,
                'block_count_cdfs.npz':cdf_payload,'block_source_responses.npz':response_payload}
        for name,payload in arrays.items():np.savez_compressed(output/name,**payload)
        result.update(status='COMPLETE_DESCRIPTIVE_INDEPENDENT_MATCHED_CLOCK_SIZE_COMPARISON',settings=settings,settings_sha256=sha256_file(config_path),
            matched_dt_ms=.0125,source_episodes_reused=sum(c['report']['completed_source_cases']for c in cohorts.values()),
            inputs={label:dict(archive=str(c['root']),delivery_sha256=c['delivery_sha256'],coupled_units=c['report']['coupled_core_units'],
                source_rng_seeds=c['report']['joint_rng_blocks'],field_unit_counts=c['sizes'].tolist(),source_cue_geometry=geometry(c['config']['excitatory_neurons'])[1],
                source_response_binding=c['phase_responses']['metadata'],common_execution_binding=c['report']['common_execution_binding'])for label,c in cohorts.items()},
            source_unit_observation_mapping='own source E indices map to16 preference fields plusone I field; no units are paired across sizes',
            cdf_weighting='equal-weight means over whole independent blocks; per-size complete unit/window masses are retained separately',
            covariance_estimator='across-block matched-time sample covariance within each size, averaged over complete phase-contained time pairs',
            count_mass_checks={label:c['count_histograms']['mass_checks']for label,c in cohorts.items()},
            count_spike_total_checks={label:c['count_histograms']['spike_checks']for label,c in cohorts.items()},
            row_counts={name:len(rows)for name,rows in tables.items()},undefined_covariance_rows=sum(row['status']!='DEFINED'for row in cov_rows),
            interpretation='Descriptive differences between independently sampled size cohorts at matched source clock, backend and physics. Unit outputs and controls remain complete; size equivalence, working-memory qualification and biological generalization require separate evidence. The existing33 clock-budget gaps are unchanged.',
            implementation_sha256=sha256_file(Path(__file__)),wall_seconds=time.monotonic()-start,
            outputs={name:dict(bytes=(output/name).stat().st_size,sha256=sha256_file(output/name))for name in [*tables,*arrays]})
    except Exception as error:
        result.update(status='FAILED',error=str(error),error_type=type(error).__name__);write_json(output/'report.json',result);raise
    write_json(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reference-archive',type=Path,required=True);parser.add_argument('--expanded-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=compare(args.reference_archive,args.expanded_archive,args.config,args.output)
    print(json.dumps({key:result[key]for key in ['status','source_episodes_reused','row_counts']}));return 0


if __name__=='__main__':raise SystemExit(main())

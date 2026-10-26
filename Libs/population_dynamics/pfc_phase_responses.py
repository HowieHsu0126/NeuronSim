"""Source-angle phase responses and paired controls from existing PFC readouts."""
from __future__ import annotations

import csv
import numpy as np

from .pfc_delay_analysis import PHASES, require, _check_readout
from .pfc_paired_source_difference import load_reference
from .pfc_response_analysis import summarize_counts
from .pfc_scale_analysis import _source_bound_file
from .provenance import sha256_file

CONDITIONS = ('omission','cue','strong_cue')
METRICS = ('all_unit_rate_hz','excitatory_rate_hz','inhibitory_rate_hz',
           'cue_region_rate_hz','noncue_region_rate_hz','cue_selectivity_hz',
           'vector_x_hz','vector_y_hz')
CONTRASTS = (('cue_minus_omission','cue','omission'),
             ('strong_cue_minus_omission','strong_cue','omission'),
             ('strong_cue_minus_cue','strong_cue','cue'))
REFERENCE_FILES = ('delivery.json','analysis/analysis.json','analysis/readout_replay.json',
                   'analysis/responses.csv','analysis/phase_unit_counts.npz')


def load(cohort, manifest):
    root,config,upstream = cohort['root'],cohort['config'],cohort['report']
    if not any(name.startswith('reference/') for name in manifest):
        return None
    for name in REFERENCE_FILES:
        _source_bound_file(root,manifest,'reference/'+name)
    reference = root/'reference'
    require(sha256_file(reference/'delivery.json') == upstream['reference_delivery_sha256'],
            'phase reference delivery differs from verified fluctuation input')
    registered={row['case_id']:row for row in config['cases']}
    report,bindings,_,count_path = load_reference(reference,upstream['config_sha256'],registered)
    seeds=upstream['joint_rng_blocks'];ne,ni=config['excitatory_neurons'],config['inhibitory_neurons']
    require(all(all(bindings[row['case_id']][key] == row[key]
                    for key in ('report_sha256','delivery_sha256','source_runner_sha256'))
                for row in upstream['source_bindings'])
            and report['common_execution_binding'] == upstream['common_execution_binding']
            and report['configuration'] == config and report['source_rng_seeds'] == seeds
            and report['phases_ms'] == PHASES and report['units'] == ne+ni
            and report['paired_effects_available'] is True
            and report['source_prefix_and_rng_pairs_verified'] == 3*len(seeds)
            and report['paired_contrasts'] == 3*len(seeds)*len(PHASES),
            'complete matching source response and input-pair evidence required')
    with (reference/'analysis/responses.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    expected={(case,phase) for case in registered for phase in PHASES}
    require(len(rows)==len(expected) and {(r['case_id'],r['phase']) for r in rows}==expected,
            'complete unique case/phase response rows required')
    indexed={(r['case_id'],r['phase']):r for r in rows}
    shape=(3,len(seeds),len(PHASES))
    values=np.empty((*shape,len(METRICS)))
    direction=np.zeros(shape,dtype=bool)
    angles=np.full(shape,np.nan)
    resultant=np.empty(shape)
    checks=0;baselines={}
    mapping=cohort['units']['unit_field_indices']
    with np.load(count_path,allow_pickle=False) as stored:
        require(set(stored.files)=={case+'_'+phase for case,phase in expected},
                'complete source unit/phase vectors required')
        for ci,condition in enumerate(CONDITIONS):
            for bi,seed in enumerate(seeds):
                case=f'pfc_{seed}_{condition}'
                for pi,(phase,(start,stop)) in enumerate(PHASES.items()):
                    row=indexed[case,phase]
                    require(row['condition']==condition and int(row['source_rng_seed'])==seed
                            and float(row['start_ms'])==start and float(row['stop_ms'])==stop
                            and int(row['units'])==ne+ni and int(row['excitatory_neurons'])==ne
                            and int(row['inhibitory_neurons'])==ni
                            and float(row['stimulus_strength_nA'])==registered[case]['stimulus_strength_nA'],
                            'source response row identity changed')
                    counts=stored[case+'_'+phase]
                    require(counts.shape==(ne+ni,) and counts.dtype.kind in 'iu' and np.all(counts>=0),
                            'complete nonnegative source unit counts required')
                    grouped=np.zeros(len(cohort['sizes']),dtype=np.int64)
                    np.add.at(grouped,mapping,counts)
                    truth=cohort['fields'][condition+'_field_counts'][bi,round(start/5):round(stop/5)].sum(axis=0)
                    require(np.array_equal(grouped,truth),'phase source counts disagree with verified field totals')
                    if phase=='baseline':
                        if bi in baselines:require(np.array_equal(baselines[bi],counts),'paired baseline unit counts changed')
                        else:baselines[bi]=counts.copy()
                    value=summarize_counts(counts,ne,config['stimulus_center_deg'],config['stimulus_width_deg'],stop-start)
                    value['all_unit_rate_hz']=float(counts.sum())/((ne+ni)*(stop-start)/1000)
                    value['direction_defined']=bool(value['direction_defined'])
                    checks+=_check_readout(row,value)
                    scale=ne*(stop-start)/1000
                    value['vector_x_hz']=value['vector_x_spikes']/scale
                    value['vector_y_hz']=value['vector_y_spikes']/scale
                    values[ci,bi,pi]=[value[name] for name in METRICS]
                    direction[ci,bi,pi]=value['direction_defined']
                    if value['decoded_angle_deg'] is not None:angles[ci,bi,pi]=value['decoded_angle_deg']
                    resultant[ci,bi,pi]=value['resultant_length']
    require(np.isfinite(values).all(),'finite source response components required')
    return dict(values=values,direction_defined=direction,decoded_angle_deg=angles,resultant_length=resultant,
        metadata=dict(source_delivery_sha256=upstream['reference_delivery_sha256'],scalar_values_verified=checks,
            complete_case_phase_rows=len(rows),complete_unit_phase_counts=(ne+ni)*len(rows),
            source_prefix_and_rng_pairs_verified=report['source_prefix_and_rng_pairs_verified'],
            vector_definition='sum(E phase spike counts * cos/sin(2*pi*source_E_index/N_E)) / (N_E*phase_seconds)',
            vector_units='Hz',cue_membership='source-rounded center/half-width with inclusive wrapped endpoints',
            all_blocks_retained_regardless_of_direction=True))


def compare(cohorts):
    require(all(c.get('phase_responses') is not None for c in cohorts.values()),
            'source phase responses required at all three clocks')
    labels=sorted(cohorts,key=lambda name:cohorts[name]['config']['dt_ms'],reverse=True)
    seeds=cohorts[labels[0]]['report']['joint_rng_blocks']
    rows,paired=[] ,[]
    for left,right in zip(labels[:-1],labels[1:]):
        a,b=cohorts[left]['phase_responses']['values'],cohorts[right]['phase_responses']['values']
        for ci,condition in enumerate(CONDITIONS):
            for pi,phase in enumerate(PHASES):
                for mi,metric in enumerate(METRICS):
                    rows.append(dict(left_cohort=left,right_cohort=right,condition=condition,phase=phase,
                        metric=metric,units='Hz',joint_rng_blocks=len(seeds),
                        left_mean=float(a[ci,:,pi,mi].mean()),right_mean=float(b[ci,:,pi,mi].mean()),
                        left_minus_right=float((a[ci,:,pi,mi]-b[ci,:,pi,mi]).mean())))
        for name,positive,negative in CONTRASTS:
            i,j=CONDITIONS.index(positive),CONDITIONS.index(negative)
            first,second=a[i]-a[j],b[i]-b[j]
            for pi,phase in enumerate(PHASES):
                for mi,metric in enumerate(METRICS):
                    paired.append(dict(left_cohort=left,right_cohort=right,contrast=name,phase=phase,
                        metric=metric,units='Hz',joint_rng_blocks=len(seeds),
                        left_mean=float(first[:,pi,mi].mean()),right_mean=float(second[:,pi,mi].mean()),
                        left_minus_right=float((first[:,pi,mi]-second[:,pi,mi]).mean())))
    payload=dict(condition_names=np.asarray(CONDITIONS),source_rng_seeds=np.asarray(seeds),
                 phase_names=np.asarray(list(PHASES)),metric_names=np.asarray(METRICS))
    for label,cohort in cohorts.items():
        for name in ('values','direction_defined','decoded_angle_deg','resultant_length'):
            payload[label+'_'+name]=cohort['phase_responses'][name]
    return {'source_response_clock_differences.csv':rows,'paired_response_clock_differences.csv':paired},payload,dict(
        status='COMPLETE_SOURCE_PHASE_RESPONSE_COMPARISON',source_episodes_reused=9*len(seeds),
        joint_rng_blocks=seeds,metrics=list(METRICS),contrasts=[x[0] for x in CONTRASTS],
        source_coordinates='true source E-unit preferences; no field-center approximation',
        direction_undefined_is_not_exclusion=True,
        input_checks={name:c['phase_responses']['metadata'] for name,c in cohorts.items()},
        independent_unit='joint_initial_background_rng_block',confidence_intervals=False,
        scientific_fidelity_qualified=False,interpretation='Descriptive signed clock differences and differences of paired source-control responses. Every source block, including undefined directions, is retained.')

"""Known-truth falsification stress of one nonconstant whole-block CDF channel.

This finite synthetic law preserves each block's mean count. It tests the
approximate bootstrap channel, not neuronal simulation fidelity or biological
rare-event frequency. Tail routing is fixed before Monte Carlo repetitions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .pfc_cdf_coupling_bounds import augment_cdf_bands, integer_cdf_coupling
from .pfc_confirmation_uncertainty import (
    STRUCTURAL, binomial_probability_upper, bootstrap_primitives, construct_bands,
    frozen_tail_departures, identity_key,
)
from .pfc_confirmation_count_scores import bound_report, require
from .pfc_numerical_precision import PrecisionSummary
from .provenance import freeze_json, sha256_file
from .source_variant_analysis import paired_bootstrap_multiplicities

DENOMINATOR = 100000
REFERENCE_COUNT_MASSES = (82000, 16000, 2000)
MEAN_COUNT_NUMERATOR = 20000
MAX_PERTURBATION = REFERENCE_COUNT_MASSES[1]//2
SUPPORT = (0, 1, 2, 3)
MODES = ('fresh_group_critical', 'calibration_floor_sensitivity')
FLOOR = 5.015885232391703
LAWS = (
    dict(name='regular_inside', d_numerators=[960,980], probability_numerators=[1,1], probability_denominator=2, rare_index=None),
    dict(name='rare_mean_outside', d_numerators=[960,980,5000], probability_numerators=[495,495,10], probability_denominator=1000, rare_index=2),
    dict(name='near_margin_inside', d_numerators=[850,1050], probability_numerators=[1,1], probability_denominator=2, rare_index=None),
    dict(name='near_margin_outside', d_numerators=[950,1150], probability_numerators=[1,1], probability_denominator=2, rare_index=None),
)


def law_truth(law):
    counts=law['d_numerators'];probabilities=law['probability_numerators'];denominator=law['probability_denominator']
    require(len(counts)==len(probabilities) and sum(probabilities)==denominator
            and all(type(n)is int and 0<=n<=MAX_PERTURBATION for n in counts)
            and all(type(n)is int and n>0 for n in probabilities), 'VALID_COMPLETE_FINITE_BLOCK_LAW_REQUIRED')
    expectation=sum((Fraction(x*p,DENOMINATOR*denominator) for x,p in zip(counts,probabilities)),Fraction())
    rare=Fraction(0)if law['rare_index']is None else Fraction(probabilities[law['rare_index']],denominator)
    return dict(mean_d=float(expectation),exact_mean_d=str(expectation),cdf_contrasts=[float(expectation),-float(expectation),0.,0.],
                expected_count=MEAN_COUNT_NUMERATOR/DENOMINATOR, exact_expected_count=str(Fraction(MEAN_COUNT_NUMERATOR,DENOMINATOR)), rare_probability=float(rare))


def sample_law(law,law_index,repetition,blocks,data_seed=97101,bootstrap_seed=97102):
    data_sequence=np.random.SeedSequence([data_seed,law_index,repetition])
    bootstrap_child=int(np.random.SeedSequence([bootstrap_seed,law_index,repetition]).generate_state(1,dtype=np.uint32)[0])
    rng=np.random.default_rng(data_sequence)
    probabilities=np.asarray(law['probability_numerators'],dtype=float)/law['probability_denominator']
    categories=rng.choice(len(probabilities),size=blocks,p=probabilities)
    values=np.asarray(law['d_numerators'],dtype=np.int64)[categories]
    rare_count=0 if law['rare_index']is None else int(np.count_nonzero(categories==law['rare_index']))
    return values,bootstrap_child,rare_count,np.bincount(categories,minlength=len(probabilities)).tolist()


def cdf_primitives(d_numerators):
    d=np.asarray(d_numerators)
    require(d.ndim==1 and len(d)>=2 and d.dtype.kind in 'iu' and np.all((d>=0)&(d<=MAX_PERTURBATION)), 'EXACT_VALID_INTEGER_CDF_PERTURBATIONS_REQUIRED')
    reference=np.repeat(np.array([REFERENCE_COUNT_MASSES],dtype=np.int64),len(d),axis=0)
    perturbed=reference+np.column_stack((d,-2*d,d))
    require(np.all(perturbed>=0)and np.all(perturbed.sum(1)==DENOMINATOR)
            and np.all(perturbed@np.array([0,1,2])==MEAN_COUNT_NUMERATOR), 'EXACT_COUNT_MASS_AND_MEAN_PRESERVATION_REQUIRED')
    left=np.column_stack((np.cumsum(perturbed,axis=1)/DENOMINATOR,np.ones(len(d))))
    right=np.column_stack((np.cumsum(reference,axis=1)/DENOMINATOR,np.ones(len(d))))
    differences=(left-right).T
    k,numerators,roundtrip=integer_cdf_coupling(differences,DENOMINATOR)
    require(np.array_equal(numerators,d), 'KNOWN_K_EQUALS_ABSOLUTE_D_REQUIRED')
    values=np.vstack((left.T,right.T));lookup={}
    for li,label in enumerate(('perturbed','reference')):
        for ki,threshold in enumerate(SUPPORT):lookup['cdf',label,'cue','late_delay',5.,0,threshold]=li*4+ki
    physical_lo=np.tile([0.,0.,1.,1.],2);physical_hi=np.ones(8)
    # The known reference CDF anchors, selected by protocol before any repetition.
    frozen_tail_indices=np.array([4,5]);frozen_tail_centers=np.array([.82,.98])
    departures=frozen_tail_departures(values[frozen_tail_indices],frozen_tail_centers)
    require(not np.any(departures), 'FIXED_REFERENCE_TAIL_ROUTE_CHANGED')
    return values,differences,k,lookup,physical_lo,physical_hi,int(departures.sum()),roundtrip


def analyze_sample(d_numerators,multiplicities,truth,floor=FLOOR,margin=.01):
    values,differences,k,lookup,physical_lo,physical_hi,departures,roundtrip=cdf_primitives(d_numerators)
    n=len(d_numerators);identities=[];domains=[]
    for threshold in SUPPORT:
        identity=dict(left_cohort='perturbed',right_cohort='reference',condition='cue',phase='late_delay',window_ms=5,field_index=0,count_threshold=threshold)
        identities.append(identity);domains.append(dict(family='count_cdf',endpoint_identity=json.dumps(identity,sort_keys=True),
            definition_status='ESTIMATED_EMPIRICAL_SPREAD'if threshold<2 else STRUCTURAL,estimate=float(differences[threshold].mean())))
    if np.all(d_numerators==d_numerators[0]):
        return [dict(mode=mode,status='UNRESOLVED_EMPIRICAL_CONSTANT_VARIABLE_LAW',simultaneous_covered=None,within_margin=None,
                     erroneous_within_margin=None,group_disjoint_from_margin=None)for mode in MODES]
    precision_summary=PrecisionSummary(len(multiplicities));point=differences.mean(1)
    direct=precision_summary.add(point,(multiplicities/n)@differences.T,point,identities)
    precision={identity_key('count_cdf',identity):row for identity,row in zip(identities,direct)}
    p,se,maximum,influence=bootstrap_primitives(values,multiplicities,precision_summary.maximum,256)
    require(np.all(se[[0,1]]>0)and np.all(se[[2,3,4,5,6,7]]==0), 'FROZEN_NONCONSTANT_AND_CONSTANT_PRIMITIVE_ROUTES_REQUIRED')
    kp,kse,maximum,kinfluence=bootstrap_primitives(k[None,:],multiplicities,maximum,256)
    require(kse[0]>0,'NONCONSTANT_K_MEAN_STANDARD_ERROR_REQUIRED')
    fresh=float(np.quantile(maximum,.975,method='higher'));tail=binomial_probability_upper(departures,n,.025)
    records=[]
    for mode,critical in zip(MODES,(fresh,max(fresh,floor))):
        legacy,_,_=construct_bands(domains,precision,(p,se,physical_lo,physical_hi),lookup,list(SUPPORT),{(5.,0):2},critical,tail)
        bands,upper=augment_cdf_bands(legacy,np.zeros(4,dtype=np.int64),kp,kse,critical)
        lo=np.asarray([row['lower']for row in bands]);hi=np.asarray([row['upper']for row in bands]);known=np.asarray(truth['cdf_contrasts'])
        covered=bool(np.all((lo<=known)&(known<=hi)))
        within=bool(np.all(lo>=-margin)and np.all(hi<=margin))
        disjoint=bool(np.any((lo>margin)|(hi<-margin)))
        records.append(dict(mode=mode,status='DEFINED',simultaneous_covered=covered,within_margin=within,
            erroneous_within_margin=bool(within and max(abs(x)for x in truth['cdf_contrasts'])>margin),group_disjoint_from_margin=disjoint,
            fresh_critical=fresh,applied_critical=critical,mean_k=float(kp[0]),k_mean_standard_error=float(kse[0]),k_upper=float(upper[0]),
            direct_mean_d=float(point[0]),direct_mean_standard_error=float(direct[0]['bootstrap_standard_error']),
            largest_block_squared_influence_fraction=float(kinfluence[0]),single_block_influence_flag=bool(kinfluence[0]>=.25),
            frozen_tail_departures=departures,frozen_tail_upper=tail,
            lower_by_threshold=json.dumps(lo.tolist()),upper_by_threshold=json.dumps(hi.tolist()),
            maximum_probability_count_roundtrip_error=roundtrip['maximum_probability_roundtrip_error']))
    return records


def exact_binomial_interval(successes,trials,alpha=.05):
    require(type(successes)is int and type(trials)is int and 0<=successes<=trials and trials>0, 'VALID_MONTE_CARLO_RATE_COUNTS_REQUIRED')
    lower=0. if successes==0 else 1-binomial_probability_upper(trials-successes,trials,alpha/2)
    upper=1. if successes==trials else binomial_probability_upper(successes,trials,alpha/2)
    return lower,upper


def summarize(records,laws):
    rows=[]
    for law in laws:
        truth=law_truth(law)
        for mode in MODES:
            all_rows=[row for row in records if row['law']==law['name']and row['mode']==mode]
            strata=[('all',all_rows)]
            if law['rare_index']is not None:
                strata += [('no_rare_observed',[row for row in all_rows if row['rare_blocks']==0]),('rare_observed',[row for row in all_rows if row['rare_blocks']>0])]
            for stratum,selected in strata:
                n=len(selected)
                if n==0:
                    rows.append(dict(law=law['name'],mode=mode,stratum=stratum,metric='NO_REPETITIONS',repetitions=0,defined_repetitions=0,
                        successes='',rate='',mc_lower_95='',mc_upper_95='',known_mean_d=truth['mean_d']));continue
                defined=sum(row['status']=='DEFINED'for row in selected)
                for metric in ['simultaneous_covered','within_margin','erroneous_within_margin','group_disjoint_from_margin']:
                    count=sum(row.get(metric)is True for row in selected);lo,hi=exact_binomial_interval(count,n)
                    rows.append(dict(law=law['name'],mode=mode,stratum=stratum,metric=metric,repetitions=n,defined_repetitions=defined,
                        successes=count,rate=count/n,mc_lower_95=lo,mc_upper_95=hi,known_mean_d=truth['mean_d']))
    return rows


def _write_rows(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x',newline='',encoding='utf-8')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(config_path,project_root,output):
    config=yaml.safe_load(Path(config_path).read_text())
    require(config.get('schema')=='pfc_nonconstant_uncertainty_stress_v1'and config.get('scope')=='known_truth_falsification'
            and config.get('method_adopted')is False and config.get('coverage_qualified')is False,
            'BOUNDED_SYNTHETIC_FALSIFICATION_ONLY_REQUIRED')
    require(config['blocks']==128 and config['bootstrap_replicates']==5000 and config['repetitions_per_law']==500
            and config['data_generator_seed']==97101 and config['bootstrap_generator_seed']==97102
            and config['nominal_alpha']==.05 and config['bootstrap_alpha']==.025 and config['tail_alpha']==.025
            and config['cdf_margin']==.01 and config['laws']==list(LAWS)
            and config['reference_count_masses']==list(REFERENCE_COUNT_MASSES), 'FIXED_STRESS_PROTOCOL_REQUIRED')
    ref=config['calibration_critical_floor'];prior=bound_report(Path(project_root)/ref['path'],ref['report_sha256'])
    require(prior['joint_critical_value']==FLOOR and prior['bootstrap_alpha']==.025 and prior['joint_tail_alpha']==.025
            and prior['coverage_qualified']is False, 'BOUND_CALIBRATION_FLOOR_REQUIRED')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[];samples=[];sample_ids=[]
    try:
        for li,law in enumerate(LAWS):
            truth=law_truth(law)
            for repetition in range(500):
                d,bootstrap_seed,rare_count,categories=sample_law(law,li,repetition,128)
                samples.append(d);sample_ids.append((li,repetition))
                m=paired_bootstrap_multiplicities(128,5000,bootstrap_seed)
                result=analyze_sample(d,m,truth)
                for row in result:
                    records.append(dict(law=law['name'],repetition=repetition,original_blocks=128,bootstrap_seed=bootstrap_seed,
                        rare_blocks=rare_count,category_counts=json.dumps(categories),known_mean_d=truth['mean_d'],**row))
            print(json.dumps(dict(status='LAW_COMPLETED',law=law['name'],repetitions=500)),flush=True)
        summary=summarize(records,LAWS)
        _write_rows(output/'per_replicate.csv',records);_write_rows(output/'per_law_summary.csv',summary)
        np.savez_compressed(output/'synthetic_block_samples.npz',d_numerators=np.asarray(samples,dtype=np.int64),
            law_and_repetition=np.asarray(sample_ids,dtype=np.int64),count_denominator=np.asarray(DENOMINATOR), reference_count_masses=np.asarray(REFERENCE_COUNT_MASSES),
            law_names=np.asarray([law['name']for law in LAWS]),count_support=np.asarray(SUPPORT))
        rare_rows=[row for row in records if row['law']=='rare_mean_outside'and row['mode']==MODES[0]]
        no_rare=sum(row['rare_blocks']==0 for row in rare_rows);no_rare_ci=exact_binomial_interval(no_rare,len(rare_rows))
        report=dict(status='COMPLETE_KNOWN_TRUTH_NONCONSTANT_CDF_STRESS',settings=config,settings_sha256=sha256_file(config_path),
            known_truth={law['name']:law_truth(law)for law in LAWS},fixed_tail_route='reference CDF0/1 at known centers .82/.98; terminal CDF1 is structural; perturbed/K anchors designated nonconstant before MC',
            calibration_critical_floor=FLOOR,floor_is_future_full_family_critical=False,
            total_independent_synthetic_repetitions=len(samples),reported_mode_rows=len(records),status_counts=dict(Counter(row['status']for row in records)),
            rare_no_event_probability_analytic=(1-law_truth(LAWS[1])['rare_probability'])**128,rare_no_event_repetitions=no_rare,rare_no_event_rate=no_rare/len(rare_rows),
            rare_no_event_mc_interval_95=list(no_rare_ci),mc_summary=summary,
            unresolved_rates='Unresolved repetitions remain in the total denominator and do not count as covered or within-margin successes.',
            confidence_interval_scope='Exact binomial Monte Carlo rate intervals over independent synthetic repetitions; not coverage qualification of the tested channel',
            source_simulations=0,production_episodes=0,method_adopted=False,coverage_qualified=False,scientific_fidelity_qualified=False,
            confirmation_execution_released=False,full_endpoint_family_coverage_qualified=False,
            interpretation='One legal bounded count-CDF group with nonconstant block variation and a 40 Hz equivalent fixed mean. This near-boundary synthetic failure probe is not fitted to actual PFC. Failures falsify unrestricted use of the tested approximate channel; they do not show actual PFC rare events, actual CDF contrasts outside the margin, or a full-family false-qualification rate.',
            implementation_sha256=sha256_file(Path(__file__)),wall_seconds=time.monotonic()-start,
            outputs={path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path))for path in output.iterdir()if path.is_file()})
    except Exception as error:
        if records:_write_rows(output/'partial_per_replicate.csv',records)
        if samples:
            np.savez_compressed(output/'partial_synthetic_block_samples.npz',d_numerators=np.asarray(samples,dtype=np.int64),
                law_and_repetition=np.asarray(sample_ids,dtype=np.int64),count_denominator=np.asarray(DENOMINATOR),
                reference_count_masses=np.asarray(REFERENCE_COUNT_MASSES))
        freeze_json(output/'report.json',dict(status='FAILED_KNOWN_TRUTH_STRESS',error=str(error),completed_mode_rows=len(records),
            settings=config,source_simulations=0,method_adopted=False,coverage_qualified=False,scientific_fidelity_qualified=False))
        raise
    freeze_json(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.config,args.project_root,args.output)
    print(json.dumps({key:report[key]for key in ['status','total_independent_synthetic_repetitions','status_counts','rare_no_event_rate']}));return 0


if __name__=='__main__':raise SystemExit(main())

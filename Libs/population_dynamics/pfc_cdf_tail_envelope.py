"""Evaluate a clipped-mean plus joint-departure upper certificate for CDF K.

The interval is finite-sample for IID assessed blocks when its clipping domain
is frozen independently. Source calibration projections are not assessed data.
The reserved non-CDF error budget is not itself a qualified inference channel.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import yaml
from .pfc_confirmation_count_scores import bound_report, require
from .pfc_confirmation_uncertainty import binomial_probability_upper
from .pfc_uncertainty_stress import LAWS, law_truth, exact_binomial_interval
from .provenance import freeze_json, sha256_file


def upper_certificate(values, lower, upper, alpha_core=.0125, alpha_tail=.0125):
    x=np.asarray(values,dtype=float)
    a,b=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    require(x.ndim==2 and x.shape[0]>0 and x.shape[1]>0 and a.shape==b.shape==(x.shape[0],), 'COMPLETE_FIXED_GROUP_BLOCK_DOMAIN_REQUIRED')
    require(np.isfinite(x).all() and np.isfinite(a).all() and np.isfinite(b).all()
            and np.all((0<=x)&(x<=1)) and np.all((0<=a)&(a<=b)&(b<=1)), 'VALID_PHYSICAL_K_AND_FIXED_CLIP_RANGES_REQUIRED')
    require(0<alpha_core<1 and 0<alpha_tail<1 and alpha_core+alpha_tail<1, 'VALID_ERROR_ALLOCATION_REQUIRED')
    groups,n=x.shape
    departures=np.any((x<a[:,None])|(x>b[:,None]),axis=0)
    p_upper=binomial_probability_upper(int(departures.sum()),n,alpha_tail)
    clipped_mean=np.clip(x,a[:,None],b[:,None]).mean(1)
    radius=(b-a)*math.sqrt(math.log(groups/alpha_core)/(2*n))
    tail=p_upper*(1-b)
    return dict(upper=np.minimum(1.,clipped_mean+radius+tail),clipped_mean=clipped_mean,
                core_radius=radius,tail_allowance=tail,joint_departures=int(departures.sum()),
                joint_tail_probability_upper=p_upper,groups=groups,blocks=n)


def write_rows(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf-8')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def verified_input(root,reference,name):
    base=root/reference['path'];report=bound_report(base,reference['report_sha256'])
    path=base/name;binding=report['outputs'][name]
    require(path.stat().st_size==binding['bytes'] and sha256_file(path)==binding['sha256'], 'INPUT_BINDING_CHANGED:'+name)
    return path,report


def run(config_path,project_root,output):
    cfg=yaml.safe_load(Path(config_path).read_text())
    require(cfg['schema']=='pfc_cdf_tail_certificate_sensitivity_v1' and cfg['scope']=='development_method_sensitivity'
            and cfg['planning_blocks']==128 and cfg['cdf_margin']==.01
            and cfg['alpha_core']==.0125 and cfg['alpha_joint_tail']==.0125 and cfg['reserved_non_cdf_alpha']==.025
            and cfg['confirmation_execution_released']is False and cfg['method_adopted']is False, 'FIXED_UNQUALIFIED_CANDIDATE_REQUIRED')
    root=Path(project_root);output=Path(output)
    stress_path,stress_report=verified_input(root,cfg['inputs']['stress'],'synthetic_block_samples.npz')
    source_path,source_report=verified_input(root,cfg['inputs']['source_coupling'],'cdf_group_block_coupling_scores.npz')
    require(stress_report['total_independent_synthetic_repetitions']==2000 and source_report['cdf_groups']==1530,
            'COMPLETE_PREVIOUS_EXPERIMENTS_REQUIRED')
    output.mkdir(parents=True,exist_ok=False)
    with np.load(stress_path,allow_pickle=False)as data:
        samples=data['d_numerators'];ids=data['law_and_repetition'];den=int(data['count_denominator'])
        require(samples.shape==(2000,128) and ids.shape==(2000,2) and den==100000
                and data['law_names'].tolist()==[law['name']for law in LAWS], 'COMPLETE_FIXED_STRESS_SAMPLE_BANK_REQUIRED')
    # Protocol constants; never computed from an assessed Monte Carlo sample.
    fixed_ranges=[(.0096,.0098),(.0096,.0098),(.0085,.0105),(.0095,.0115)]
    records=[]
    for values,(li,repetition)in zip(samples,ids):
        li=int(li);a,b=fixed_ranges[li];truth=law_truth(LAWS[li])['mean_d']
        bound=upper_certificate(values[None,:]/den,[a],[b],cfg['alpha_core'],cfg['alpha_joint_tail'])
        value=float(bound['upper'][0]);inside=value<=cfg['cdf_margin']
        records.append(dict(law=LAWS[li]['name'],repetition=int(repetition),known_mean_k=truth,clip_lower=a,clip_upper=b,
            upper_mean_k=value,joint_departures=bound['joint_departures'],joint_tail_probability_upper=bound['joint_tail_probability_upper'],
            mean_k_covered=bool(truth<=value),cdf_group_within_margin=bool(inside),erroneous_within_margin=bool(inside and truth>.01)))
    require(len({(row['law'],row['repetition'])for row in records})==2000, 'UNIQUE_REUSED_MC_REPETITIONS_REQUIRED')
    summary=[]
    for law in LAWS:
        subset=[row for row in records if row['law']==law['name']];require(len(subset)==500, 'ALL_STRESS_LAWS_RETAINED_REQUIRED')
        for metric in ['mean_k_covered','cdf_group_within_margin','erroneous_within_margin']:
            count=sum(row[metric]for row in subset);lo,hi=exact_binomial_interval(count,500)
            summary.append(dict(law=law['name'],metric=metric,count=count,repetitions=500,rate=count/500,mc_lower_95=lo,mc_upper_95=hi))
    with np.load(source_path,allow_pickle=False)as data:
        scores=data['scores'];seeds=data['source_rng_seeds'];identities=data['group_identities']
        require(scores.shape==(1530,32) and np.array_equal(seeds,np.arange(95001,95033))
                and np.all((0<=scores)&(scores<=1)) and len(identities)==1530, 'FULL_ORIGINAL_CALIBRATION_K_DOMAIN_REQUIRED')
    # Same-data extrema only DEFINE a candidate frozen domain. No same-data CI.
    lower=scores.min(1);upper=scores.max(1);point=scores.mean(1);n=cfg['planning_blocks']
    p0=binomial_probability_upper(0,n,cfg['alpha_joint_tail'])
    core=(upper-lower)*math.sqrt(math.log(len(scores)/cfg['alpha_core'])/(2*n))
    tail=p0*(1-upper);projected=np.minimum(1.,point+core+tail)
    planning=[dict(group_index=i,group_identity=str(identities[i]),calibration_mean_k=float(point[i]),
        calibration_clip_lower=float(lower[i]),calibration_clip_upper=float(upper[i]),
        zero_departure_probability_upper_at_128=p0,tail_allowance_at_128=float(tail[i]),core_radius_at_128=float(core[i]),
        calibration_mean_held_fixed_upper_at_128=float(projected[i]),tail_allowance_alone_exceeds_margin=bool(tail[i]>.01),
        certificate_planning_within_margin=bool(projected[i]<=.01),status='PLANNING_ONLY_NO_CONFIRMATION_DATA')for i in range(1530)]
    write_rows(output/'reused_stress_certificates.csv',records);write_rows(output/'stress_certificate_summary.csv',summary)
    write_rows(output/'source_cdf_certificate_planning.csv',planning)
    report=dict(status='COMPLETE_CDF_TAIL_CERTIFICATE_SENSITIVITY',settings=cfg,settings_sha256=sha256_file(config_path),
        reused_synthetic_repetitions=2000,new_monte_carlo_samples=0,new_bootstrap_draws=0,new_source_simulations=0,
        source_cdf_groups=1530,source_calibration_blocks=32,planning_blocks=128,zero_departure_probability_upper_at_128=p0,
        groups_tail_allowance_alone_exceeds_margin=int(np.sum(tail>.01)),groups_planning_within_margin=int(np.sum(projected<=.01)),
        tail_allowance_range=[float(tail.min()),float(tail.max())],stress_summary=summary,
        mathematical_scope='For fixed independent-of-assessment ranges and IID whole blocks, K<=clip(K,a,b)+(1-b)1_A. One-sided Hoeffding union bounds cover all clipped means; one joint-event Clopper-Pearson upper bound covers P(A).',
        cdf_family_nominal_coverage=1-cfg['alpha_core']-cfg['alpha_joint_tail'],
        reserved_non_cdf_budget_is_qualified=False,full_family_coverage_qualified=False,
        source_planning_is_confidence_evidence=False,
        planning_interpretation='Calibration extrema define this candidate domain; held-fixed calibration means and zero future departures are optimistic projections, not observed future intervals. A tail term above the margin blocks this particular certificate for n<=128 under these ranges and allocation, not every possible method or the simulator.',
        stress_interpretation='The fixed synthetic ranges come from the declared regular-law support, not from each assessed saved sample. Saved data are reused to diagnose this new conservative construction, not to qualify actual PFC coverage.',
        method_adopted=False,confirmation_execution_released=False,scientific_fidelity_qualified=False,production_episodes=0,
        implementation_sha256=sha256_file(Path(__file__)),
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p))for p in output.iterdir()if p.is_file()})
    freeze_json(output/'report.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.config,args.project_root,args.output)
    print(json.dumps({k:result[k]for k in ['status','groups_tail_allowance_alone_exceeds_margin','groups_planning_within_margin']}))
    return 0


if __name__=='__main__':raise SystemExit(main())

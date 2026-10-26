"""Freeze PFC numerical error budgets and assess existing calibration planning.

Budgets are normative accuracy requirements, not estimates of acceptable error
from clock differences. A planning interval does not qualify numerical fidelity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import platform

import numpy as np
import yaml

from .pfc_confirmation_count_scores import bound_report, read_rows, require
from .pfc_delay_analysis import PHASES
from .pfc_phase_responses import CONDITIONS, CONTRASTS, METRICS
from .pfc_response_analysis import summarize_counts
from .pfc_scale_analysis import _source_manifest, _source_bound_file
from .provenance import freeze_json, sha256_file

STRUCTURAL = 'STRUCTURAL_ZERO_CONTRAST'
UNDEFINED = 'UNDEFINED_NO_WITHIN_PHASE_PAIRS'


def response_spike_sensitivities(config: dict) -> tuple[dict, dict]:
    """Largest effect of one source spike; paired arms concatenate signed weights."""
    ne,ni = config['excitatory_neurons'],config['inhibitory_neurons']
    require(type(ne) is int and type(ni) is int and ne > 0 and ni > 0, 'POSITIVE_SOURCE_POPULATIONS_REQUIRED')
    readout = summarize_counts(np.zeros(ne+ni,dtype=np.int64),ne,config['stimulus_center_deg'],config['stimulus_width_deg'],1000.)
    cue,noncue = readout['cue_units'],ne-readout['cue_units']
    angles = 2*np.pi*np.arange(ne)/ne
    coefficients = dict(all_unit_rate_hz=1/(ne+ni),excitatory_rate_hz=1/ne,inhibitory_rate_hz=1/ni,
        cue_region_rate_hz=1/cue,noncue_region_rate_hz=1/noncue,cue_selectivity_hz=max(1/cue,1/noncue),
        vector_x_hz=float(np.max(np.abs(np.cos(angles))))/ne,
        vector_y_hz=float(np.max(np.abs(np.sin(angles))))/ne)
    values = {(phase,metric):coefficients[metric]/((stop-start)/1000)
              for phase,(start,stop) in PHASES.items() for metric in METRICS}
    return values,dict(excitatory_units=ne,inhibitory_units=ni,cue_units=cue,noncue_units=noncue,
        phase_metric_event_sensitivities_hz={phase:{metric:values[phase,metric] for metric in METRICS} for phase in PHASES},
        definition='maximum absolute coefficient of one spike in the source linear observable, not a vector lattice step; paired arms use concatenated signed weights, not one event per arm')


def response_budget(mean, block_sd, event_sensitivity, policy):
    require(np.isfinite([mean,block_sd,event_sensitivity]).all() and block_sd >= 0 and event_sensitivity > 0,
            'FINITE_SOURCE_RESPONSE_REFERENCE_REQUIRED')
    values = dict(signal=policy['response_signal_fraction']*abs(mean),
                  block_fluctuation=policy['response_block_sd_fraction']*block_sd,
                  single_spike=policy['response_single_spike_allowance']*event_sensitivity)
    margin = max(values.values())
    return margin,values,'+'.join(key for key,value in values.items() if value == margin)


def reference_maps(field_rows, response_rows, paired_rows, config, reference_cohort, blocks):
    fields = config['excitatory_preference_bins']+1
    ne,ni = config['excitatory_neurons'],config['inhibitory_neurons']
    require(ne % (fields-1) == 0, 'SOURCE_EQUAL_PREFERENCE_FIELDS_REQUIRED')
    sizes = [ne//(fields-1)]*(fields-1)+[ni]
    field_map,source_map,paired_map = {},{},{}
    for row in field_rows:
        if row['cohort'] != reference_cohort:continue
        key = row['condition'],row['phase'],int(row['field_index'])
        require(key not in field_map and int(row['joint_rng_blocks']) == blocks and 0 <= key[2] < fields,
                'UNIQUE_COMPLETE_FIELD_REFERENCE_REQUIRED')
        require(int(row['units_in_field']) == sizes[key[2]] and row['population'] == ('E' if key[2] < fields-1 else 'I'),
                'SOURCE_FIELD_MAP_CHANGED')
        sd,q = float(row['matched_time_marginal_sd_hz']),float(row['single_event_bin_rate_quantum_hz'])
        require(math.isfinite(sd) and sd >= 0 and q == 1/(sizes[key[2]]*config['readout_bin_ms']/1000),
                'SOURCE_REFERENCE_SCALE_OR_EVENT_FLOOR_CHANGED')
        field_map[key] = dict(sd=sd,floor=q,scale=max(sd,q))
    require(set(field_map) == {(condition,phase,field) for condition in CONDITIONS for phase in PHASES for field in range(fields)},
            'COMPLETE_CONDITION_PHASE_FIELD_REFERENCE_REQUIRED')
    for rows,mapping,identity in [(response_rows,source_map,'condition'),(paired_rows,paired_map,'contrast')]:
        for row in rows:
            if row['cohort'] != reference_cohort:continue
            key = row[identity],row['phase'],row['metric']
            require(key not in mapping and int(row['joint_rng_blocks']) == blocks and row['units'] == 'Hz'
                    and float(row['dt_ms']) == config['dt_ms'] and int(row['coupled_units']) == ne+ni,
                    'SOURCE_RESPONSE_REFERENCE_IDENTITY_CHANGED')
            require([float(row['start_ms']),float(row['stop_ms'])] == PHASES[key[1]],'SOURCE_PHASE_EXPOSURE_CHANGED')
            mean,sd = float(row['mean']),float(row['block_sd'])
            require(math.isfinite(mean) and math.isfinite(sd) and sd >= 0,'FINITE_RESPONSE_SCALE_REQUIRED')
            mapping[key] = dict(mean=mean,sd=sd)
        names = CONDITIONS if identity == 'condition' else [item[0] for item in CONTRASTS]
        require(set(mapping) == {(name,phase,metric) for name in names for phase in PHASES for metric in METRICS},
                'COMPLETE_RESPONSE_REFERENCE_REQUIRED')
    return field_map,source_map,paired_map


def build_margins(domains, references, sensitivities, policy):
    fields,sources,paired = references
    rows = []
    for index,domain in enumerate(domains):
        family,status = domain['family'],domain['definition_status']
        ident = json.loads(domain['endpoint_identity'])
        row = dict(domain_row_index=index,family=family,endpoint_identity=domain['endpoint_identity'],definition_status=status,
            margin='',units='probability' if family == 'count_cdf' else ('Hz^2' if family == 'field_lag_covariance' else 'Hz'),
            rule='',reference_mean='',reference_block_sd='',reference_field_scale_leading_hz='',reference_field_scale_trailing_hz='',
            reference_field_floor_leading_hz='',reference_field_floor_trailing_hz='',signal_allowance='',fluctuation_allowance='',single_spike_allowance='',
            dominant_allowance='',noise_allowance_exceeds_reference_signal='')
        if status == UNDEFINED:
            row['rule']='undefined_no_within_phase_pairs';rows.append(row);continue
        if status == STRUCTURAL:
            row.update(margin=0.,rule='source_proven_identity',dominant_allowance='identity');rows.append(row);continue
        if family == 'count_cdf':
            row.update(margin=policy['count_cdf_absolute_probability'],rule='absolute_count_threshold_probability',dominant_allowance='probability_budget')
        elif family in ('field_mean','field_lag_covariance'):
            a = fields[ident['condition'],ident['phase'],int(ident['field_index'] if family == 'field_mean' else ident['leading_field'])]
            row.update(reference_field_scale_leading_hz=a['scale'],reference_field_floor_leading_hz=a['floor'])
            if family == 'field_mean':
                margin = policy['field_mean_scale_fraction']*a['scale']
                row.update(margin=margin,fluctuation_allowance=margin,rule='fixed_matched_time_field_scale',dominant_allowance='field_fluctuation')
            else:
                b = fields[ident['condition'],ident['phase'],int(ident['trailing_field'])]
                margin = policy['covariance_entry_scale_fraction']*a['scale']*b['scale']
                row.update(margin=margin,fluctuation_allowance=margin,reference_field_scale_trailing_hz=b['scale'],reference_field_floor_trailing_hz=b['floor'],
                           rule='entrywise_product_of_fixed_marginal_scales',dominant_allowance='field_fluctuation')
        else:
            require(family in ('source_response','paired_response'),'KNOWN_ENDPOINT_FAMILY_REQUIRED')
            ref = sources[ident['condition'],ident['phase'],ident['metric']] if family == 'source_response' else paired[ident['contrast'],ident['phase'],ident['metric']]
            margin,parts,dominant = response_budget(ref['mean'],ref['sd'],sensitivities[ident['phase'],ident['metric']],policy)
            row.update(margin=margin,rule='max_own_signal_own_block_scatter_one_source_spike',reference_mean=ref['mean'],reference_block_sd=ref['sd'],
                signal_allowance=parts['signal'],fluctuation_allowance=parts['block_fluctuation'],single_spike_allowance=parts['single_spike'],
                dominant_allowance=dominant,noise_allowance_exceeds_reference_signal=parts['block_fluctuation'] > abs(ref['mean']))
        require(math.isfinite(row['margin']) and row['margin'] > 0,'POSITIVE_NONSTRUCTURAL_MARGIN_REQUIRED')
        rows.append(row)
    return rows


def assess_planning(margins, bands, planning_blocks):
    require(len(margins) == len(bands),'COMPLETE_MARGIN_AND_BAND_DOMAIN_REQUIRED')
    rows,summary = [],defaultdict(Counter)
    lower_key = 'planning_lower_root_n_and_zero_tail_at_'+str(planning_blocks)
    upper_key = 'planning_upper_root_n_and_zero_tail_at_'+str(planning_blocks)
    for index,(margin,band) in enumerate(zip(margins,bands)):
        require(int(margin['domain_row_index']) == int(band['domain_row_index']) == index
                and all(margin[key] == band[key] for key in ('family','endpoint_identity','definition_status')),
                'MARGIN_AND_UNCERTAINTY_IDENTITIES_DIFFER')
        row = dict(domain_row_index=index,family=margin['family'],endpoint_identity=margin['endpoint_identity'],definition_status=margin['definition_status'],
            margin=margin['margin'],calibration_estimate=band['estimate'],planning_lower=band[lower_key],planning_upper=band[upper_key],
            planning_status='',calibration_point_outside_margin='',planning_absolute_bound_over_margin='',
            calibration_interval_relation='',planning_interval_relation='')
        ident = json.loads(margin['endpoint_identity']); group = (ident['left_cohort'],ident['right_cohort'],margin['family'])
        if margin['definition_status'] == UNDEFINED:
            require(margin['margin'] == '' and band[lower_key] == band[upper_key] == '','UNDEFINED_ENDPOINT_MUST_REMAIN_UNESTIMATED')
            row['planning_status']='UNDEFINED'
        elif margin['definition_status'] == STRUCTURAL:
            require(float(margin['margin']) == float(band[lower_key]) == float(band[upper_key]) == 0.,'STRUCTURAL_IDENTITY_CHANGED')
            row['planning_status']='SOURCE_IDENTITY';row['calibration_point_outside_margin']=False
        else:
            delta,lo,hi = float(margin['margin']),float(band[lower_key]),float(band[upper_key])
            require(delta > 0 and math.isfinite(lo) and math.isfinite(hi) and lo <= hi,'FINITE_DEFINED_PLANNING_BAND_REQUIRED')
            absolute = max(abs(lo),abs(hi))
            cal_lo,cal_hi = float(band['lower']),float(band['upper'])
            require(math.isfinite(cal_lo) and math.isfinite(cal_hi) and cal_lo <= cal_hi,'VALID_CALIBRATION_BAND_REQUIRED')
            def relation(a,b):
                if a >= -delta and b <= delta:return 'WITHIN_BUDGET'
                if b < -delta or a > delta:return 'DISJOINT_FROM_BUDGET'
                return 'OVERLAPS_BUDGET'
            row.update(calibration_interval_relation=relation(cal_lo,cal_hi),planning_interval_relation=relation(lo,hi))
            row.update(planning_status='PLANNING_BAND_WITHIN_BUDGET' if absolute <= delta else 'PLANNING_BAND_EXCEEDS_BUDGET',
                calibration_point_outside_margin=abs(float(band['estimate'])) > delta,planning_absolute_bound_over_margin=absolute/delta)
        summary[group][row['planning_status']]+=1;rows.append(row)
    return rows,[dict(left_cohort=key[0],right_cohort=key[1],family=key[2],counts=dict(value)) for key,value in sorted(summary.items())]


def write_rows(path, rows):
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(config_path, project_root, output):
    settings = yaml.safe_load(config_path.read_text())
    require(settings['schema'] == 'pfc_numerical_error_budget_v1' and settings['scope'] == 'prospective_numerical_error_budget'
            and settings['policy_status'] == 'frozen_definition_before_feasibility'
            and settings['confirmation_execution_released'] is False,'PROSPECTIVE_FROZEN_ERROR_BUDGET_REQUIRED')
    roots = {key:project_root/value['path'] for key,value in settings['inputs'].items()}
    reports = {key:bound_report(roots[key],value['report_sha256']) for key,value in settings['inputs'].items()}
    require(reports['design']['status'] == 'COMPLETE_DESCRIPTIVE_CONFIRMATION_DESIGN_INPUTS'
            and reports['uncertainty']['status'] == 'COMPLETE_CALIBRATION_JOINT_TAIL_UNCERTAINTY_DIAGNOSTIC'
            and reports['domain']['status'] == 'COMPLETE_CALIBRATION_BLOCK_CDF_SCORES','COMPLETE_BOUND_CALIBRATION_INPUTS_REQUIRED')
    tables = {}
    for key,names in [('design',['marginal_field_scales.csv','response_scales.csv','paired_response_scales.csv']),
                      ('domain',['endpoint_domains.csv']),('uncertainty',['endpoint_bands.csv'])]:
        for name in names:
            path = roots[key]/name
            require(sha256_file(path) == reports[key]['outputs'][name]['sha256'],'BOUND_CALIBRATION_TABLE_CHANGED:'+name)
            tables[name] = read_rows(path)
    require(len(tables['endpoint_domains.csv']) == reports['domain']['domain_rows'] == reports['uncertainty']['domain_rows'] == 84330,
            'FULL_FROZEN_ENDPOINT_DOMAIN_REQUIRED')
    source_root = project_root/settings['source_archive']
    require(sha256_file(source_root/'delivery.json') == reports['comparison']['inputs'][settings['reference_cohort']]['delivery_sha256'],
            'SOURCE_DELIVERY_BINDING_CHANGED')
    source_manifest = _source_manifest(json.loads((source_root/'delivery.json').read_text()))
    source_report = json.loads(_source_bound_file(source_root,source_manifest,'analysis/report.json').read_text())
    source_path = _source_bound_file(source_root,source_manifest,'config.yaml')
    require(sha256_file(source_path) == source_report['config_sha256'],'SOURCE_OBSERVATION_CONFIG_CHANGED')
    source_config = yaml.safe_load(source_path.read_text())
    reference,seeds = settings['reference_cohort'],settings['joint_rng_seeds']
    require(reports['design']['settings']['fluctuation_archives'][reference] == settings['source_archive']
            and source_report['status'] == 'COMPLETE_SOURCE_FLUCTUATION_CALIBRATION'
            and source_report['joint_rng_blocks'] == reports['design']['independent_blocks'][reference] == seeds
            and source_config['dt_ms'] == settings['reference_dt_ms'] == .0125
            and source_report['phases_ms'] == PHASES and source_report['coupled_core_units'] == 100000,
            'FINEST_COMPLETE_SOURCE_REFERENCE_REQUIRED')
    references = reference_maps(tables['marginal_field_scales.csv'],tables['response_scales.csv'],tables['paired_response_scales.csv'],source_config,reference,len(seeds))
    sensitivity,sensitivity_report = response_spike_sensitivities(source_config);policy = settings['error_budget']
    require(policy == dict(field_mean_scale_fraction=.10,covariance_entry_scale_fraction=.10,count_cdf_absolute_probability=.01,
                           response_signal_fraction=.01,response_block_sd_fraction=.10,response_single_spike_allowance=1.),
            'DECLARED_ERROR_BUDGET_POLICY_CHANGED')
    margins = build_margins(tables['endpoint_domains.csv'],references,sensitivity,policy)
    output.mkdir(parents=True,exist_ok=False)
    margin_path = output/'frozen_numerical_margins.csv';write_rows(margin_path,margins)
    planning_blocks = int(settings['maximum_confirmation_blocks']);require(planning_blocks == 128,'UNCHANGED_CONFIRMATION_RESERVATION_REQUIRED')
    planning,summary = assess_planning(read_rows(margin_path),tables['endpoint_bands.csv'],planning_blocks)
    plan_path=output/'calibration_budget_assessment.csv';write_rows(plan_path,planning)
    counts = Counter(row['planning_status'] for row in planning)
    exceeded = [row for row in planning if row['planning_status'] == 'PLANNING_BAND_EXCEEDS_BUDGET']
    weak_noise_dominated = [row for row in margins if row['family'] == 'paired_response' and row['noise_allowance_exceeds_reference_signal'] is True]
    tail_alpha,epsilon = reports['uncertainty']['joint_tail_alpha'],policy['count_cdf_absolute_probability']
    minimum_tail_n = math.ceil(math.log(tail_alpha)/math.log1p(-epsilon))
    best_tail_at_limit = -math.expm1(math.log(tail_alpha)/planning_blocks)
    tail_limited = [row for row in exceeded if row['family'] == 'count_cdf'
                    and math.isclose(max(abs(float(row['planning_lower'])),abs(float(row['planning_upper']))),best_tail_at_limit,rel_tol=1e-10,abs_tol=1e-14)]
    report = dict(status='FROZEN_NUMERICAL_BUDGET_CONFIRMATION_PLAN_UNRESOLVED' if exceeded else 'FROZEN_NUMERICAL_BUDGET_PLANNING_PRECISION_SUFFICIENT_COVERAGE_UNQUALIFIED',
        settings=settings,settings_sha256=sha256_file(config_path),source_config_sha256=sha256_file(source_path),
        reference_cohort=reference,reference_independent_blocks=len(seeds),margin_rows=len(margins),
        finite_margin_rows=sum(row['margin'] != '' for row in margins),undefined_margin_rows=sum(row['margin'] == '' for row in margins),
        source_observation_sensitivity=sensitivity_report,numerical_margins_frozen=True,
        budgets_are_normative_design_choices=True,budgets_inferred_from_clock_differences=False,
        covariance_budget_scope='entrywise signed lag covariance normalized by fixed full-phase matched-time marginal scales; no operator-norm or correlation-error claim',
        field_scale_floor='one field spike per5ms readout bin; fixed before confirmation; reference SD is not a standard error',
        response_budget_scope='Each condition and each paired contrast uses its own reference signal and whole-block SD. One source spike allowance is outside the percentages. Near-zero vectors do not imply angle accuracy.',
        paired_response_noise_allowance_exceeds_reference_signal=len(weak_noise_dominated),
        weak_response_interpretation='Noise-relative accuracy can permit error larger than a tiny mean effect. This does not preserve that effect sign or relative magnitude.',
        planning_blocks=planning_blocks,planning_status_counts=dict(counts),planning_by_clock_pair_and_family=summary,
        calibration_points_outside_budget=sum(row['calibration_point_outside_margin'] is True for row in planning),
        largest_planning_exceedances=sorted(exceeded,key=lambda row:row['planning_absolute_bound_over_margin'],reverse=True)[:12],
        tail_channel_zero_departure_minimum_n_for_one_percentage_point=minimum_tail_n,
        tail_channel_best_case_bound_at_reserved_limit=best_tail_at_limit,tail_channel_limited_planning_endpoints=len(tail_limited),
        tail_channel_scope='Only the candidate pooled-departure channel, with its frozen alpha and zero-departure scenario. This is not an impossibility proof for all inferential methods or evidence of simulator inaccuracy.',
        uncertainty_method_adopted=False,uncertainty_coverage_qualified=False,confirmation_n_selected=None,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,new_source_simulations=0,raw_spike_data_reread=False,production_episodes=0,
        interpretation='Frozen error requirements and calibration planning assessment only. Point estimates and projected intervals are not independent equivalence tests. Do not widen margins or automatically add blocks after inadequate planning precision.',
        python_version=platform.python_version(),numpy_version=np.__version__,implementation_sha256=sha256_file(Path(__file__)),
        outputs={path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path)) for path in (margin_path,plan_path)})
    freeze_json(output/'report.json',report)
    return report


def diagnose_archived_budget(config_path, project_root, output, budget_archive, report_sha256):
    """Separate calibration conflict from width without recomputing frozen margins."""
    root=project_root/budget_archive
    prior=bound_report(root,report_sha256)
    require(prior['numerical_margins_frozen'] is True and sha256_file(config_path) == prior['settings_sha256'],
            'FROZEN_POLICY_MUST_REMAIN_UNCHANGED')
    margin_path=root/'frozen_numerical_margins.csv'
    require(sha256_file(margin_path) == prior['outputs'][margin_path.name]['sha256'],'FROZEN_MARGIN_TABLE_CHANGED')
    reference=prior['settings']['inputs']['uncertainty'];band_root=project_root/reference['path']
    band_report=bound_report(band_root,reference['report_sha256']);band_path=band_root/'endpoint_bands.csv'
    require(sha256_file(band_path) == band_report['outputs'][band_path.name]['sha256'],'CALIBRATION_BANDS_CHANGED')
    rows,summary=assess_planning(read_rows(margin_path),read_rows(band_path),prior['planning_blocks'])
    counts=Counter(row['planning_status'] for row in rows)
    require(dict(counts) == prior['planning_status_counts'],'PREVIOUS_PLANNING_ASSESSMENT_CHANGED')
    relations={name:Counter(row[name] for row in rows if row[name]) for name in ('calibration_interval_relation','planning_interval_relation')}
    by_group=defaultdict(lambda:dict(calibration=Counter(),planning=Counter()))
    for row in rows:
        ident=json.loads(row['endpoint_identity']);key=ident['left_cohort'],ident['right_cohort'],row['family']
        if row['calibration_interval_relation']:by_group[key]['calibration'][row['calibration_interval_relation']]+=1
        if row['planning_interval_relation']:by_group[key]['planning'][row['planning_interval_relation']]+=1
    disjoint=[row for row in rows if row['calibration_interval_relation']=='DISJOINT_FROM_BUDGET']
    output.mkdir(parents=True,exist_ok=False);path=output/'calibration_budget_diagnosis.csv';write_rows(path,rows)
    report=dict(prior)
    report.update(status='COMPLETE_FROZEN_BUDGET_CAUSE_DIAGNOSIS',frozen_budget_report_sha256=report_sha256,
        frozen_margin_archive=str(budget_archive),frozen_margin_sha256=prior['outputs'][margin_path.name]['sha256'],
        frozen_margins_recomputed=False,bootstrap_or_covariance_recomputed=False,
        calibration_interval_relations=dict(relations['calibration_interval_relation']),planning_interval_relations=dict(relations['planning_interval_relation']),
        interval_relations_by_clock_pair_and_family=[dict(left_cohort=key[0],right_cohort=key[1],family=key[2],
            calibration=dict(value['calibration']),planning=dict(value['planning'])) for key,value in sorted(by_group.items())],
        exploratory_calibration_disjoint_endpoints=disjoint,
        diagnosis_scope='Exploratory calibration interval geometry against frozen budgets. Disjoint calibration bands suggest clock-sensitive quantities under the approximate procedure, not independently confirmed source error. Overlapping bands leave both equivalence and non-equivalence unresolved.',
        confirmation_n_selected=None,confirmation_execution_released=False,scientific_fidelity_qualified=False,
        implementation_sha256=sha256_file(Path(__file__)),
        outputs={path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path))})
    freeze_json(output/'report.json',report)
    return report


def source_variance_plan(margins, precision, family_size, blocks=128):
    """Optimistic scalar Bernstein constraints, not source variance certificates.

    The scenario grants all alpha=.05 and the whole endpoint margin to source
    uncertainty, with zero bias, zero increment term and zero inner error.
    Every actual requirement can be tighter. A covariance pseudovalue variance
    cannot supply the ordinary-block variance used in this scalar formula.
    """
    require(type(family_size) is int and family_size > 0 and blocks == 128,
            'FROZEN_SOURCE_VARIANCE_PLANNING_DOMAIN_REQUIRED')
    logarithm = math.log(2*family_size/.05)
    for index,(margin,stat) in enumerate(zip(margins,precision,strict=True)):
        require(int(margin['domain_row_index']) == int(stat['domain_row_index']) == index
                and all(margin[key] == stat[key] for key in ('family','endpoint_identity','definition_status','margin','units')),
                'FROZEN_SOURCE_PRECISION_IDENTITIES_DIFFER')
        family,status = margin['family'],margin['definition_status']
        row = dict(domain_row_index=index,family=family,endpoint_identity=margin['endpoint_identity'],
            definition_status=status,margin=margin['margin'],units=margin['units'],
            variance_units={'probability':'probability^2','Hz':'Hz^2','Hz^2':'Hz^4'}[margin['units']],
            source_sampling_target='declared_ideal_joint_source_history',
            original_source_blocks=stat['coupled_original_independent_blocks'],
            planning_source_blocks=blocks,nonstructural_defined_family_size=family_size,
            source_alpha_upper_scenario=.05,source_score_kind=stat['coupled_score_kind'],
            calibration_signed_estimate=stat['coupled_signed_estimate'],
            calibration_sample_variance=stat['coupled_sample_variance'],
            calibration_sample_variance_kind=stat['coupled_sample_variance_kind'],
            current_empirical_status=stat['coupled_empirical_status'],
            scalar_source_variance_ceiling='',scalar_increment_ceiling_if_zero_variance='',
            sample_variance_over_optimistic_ceiling='',calibration_point_outside_margin='',
            planning_status='',population_variance_certified=False)
        require(int(stat['coupled_original_independent_blocks']) == 32,
                'ORIGINAL_SOURCE_BLOCK_COUNT_REQUIRED')
        if status == UNDEFINED:
            require(margin['margin'] == '' and stat['coupled_signed_estimate'] == '',
                    'UNDEFINED_SOURCE_ENDPOINT_CHANGED')
            row['planning_status']='UNDEFINED_NO_WITHIN_PHASE_PAIRS'
        elif status == STRUCTURAL:
            require(float(margin['margin']) == float(stat['coupled_signed_estimate']) == 0,
                    'STRUCTURAL_SOURCE_IDENTITY_CHANGED')
            row['planning_status']='SOURCE_IDENTITY'
        elif family == 'field_lag_covariance':
            require(stat['coupled_score_kind'] == 'covariance_jackknife_pseudovalue'
                    and stat['coupled_sample_variance_kind'] == 'jackknife_pseudovalue_variance',
                    'COVARIANCE_SCORE_SEMANTICS_CHANGED')
            row['planning_status']='U_STATISTIC_REQUIRES_ITS_SOURCE_CONCENTRATION_METHOD'
        else:
            require(family in ('field_mean','count_cdf','source_response','paired_response')
                    and stat['coupled_score_kind'] in ('paired_block_linear_difference','paired_block_cdf_difference')
                    and stat['coupled_sample_variance_kind'] == 'paired_block_difference_variance',
                    'ORDINARY_SOURCE_BLOCK_SCORE_REQUIRED')
            delta,var,point = float(margin['margin']),float(stat['coupled_sample_variance']),float(stat['coupled_signed_estimate'])
            require(delta > 0 and np.isfinite([delta,var,point]).all() and var >= 0,
                    'FINITE_SOURCE_CALIBRATION_VALUES_REQUIRED')
            ceiling = blocks*delta*delta/(2*logarithm)
            row.update(scalar_source_variance_ceiling=ceiling,
                       scalar_increment_ceiling_if_zero_variance=3*ceiling/delta,
                       sample_variance_over_optimistic_ceiling=var/ceiling,
                       calibration_point_outside_margin=abs(point)>delta,
                       planning_status='CALIBRATION_ONLY_NO_SOURCE_VARIANCE_CERTIFICATE')
        yield row


def auxiliary_variance_point_plan(row, auxiliary, identity_tolerance):
    """Reuse saved A/B moments; a negative between estimate remains unresolved."""
    fields=dict(auxiliary_planning_status='NOT_AVAILABLE',auxiliary_realizations='',
        auxiliary_midpoint_estimate='',auxiliary_midpoint_sample_variance='',
        auxiliary_within_variance_estimate='',between_root_variance_estimate='',
        between_root_estimate_over_optimistic_ceiling='',
        point_auxiliary_realizations_for_variance_ceiling='')
    if auxiliary is None:
        return fields
    require(int(auxiliary['domain_row_index'])==row['domain_row_index'] and all(
        auxiliary[key]==row[key] for key in ('family','endpoint_identity','definition_status','margin','units','variance_units')),
        'AUXILIARY_SOURCE_ENDPOINT_IDENTITY_CHANGED')
    require(int(auxiliary['original_independent_blocks'])==32 and int(auxiliary['auxiliary_realizations'])==2,
            'ORIGINAL_TWO_AUXILIARY_REALIZATIONS_REQUIRED')
    require(math.isclose(float(auxiliary['original_estimate']),float(row['calibration_signed_estimate']),
                         rel_tol=identity_tolerance['rtol'],abs_tol=identity_tolerance['atol']),
            'AUXILIARY_ORIGINAL_SOURCE_SCORE_CHANGED')
    fields.update(auxiliary_realizations=2,auxiliary_midpoint_estimate=auxiliary['estimate_midpoint'],
        auxiliary_midpoint_sample_variance=auxiliary['midpoint_sample_variance'],
        auxiliary_within_variance_estimate=auxiliary['auxiliary_within_variance_estimate'],
        between_root_variance_estimate=auxiliary['between_block_variance_estimate'])
    within,between,midpoint=map(float,(auxiliary['auxiliary_within_variance_estimate'],
        auxiliary['between_block_variance_estimate'],auxiliary['midpoint_sample_variance']))
    require(np.isfinite([within,between,midpoint]).all() and within>=0 and midpoint>=0,
            'FINITE_UNCLIPPED_AUXILIARY_MOMENTS_REQUIRED')
    require(math.isclose(midpoint,between+within/2,rel_tol=identity_tolerance['rtol'],abs_tol=identity_tolerance['atol']),
            'AUXILIARY_VARIANCE_DECOMPOSITION_CHANGED')
    if row['scalar_source_variance_ceiling']=='':
        fields['auxiliary_planning_status']='SOURCE_IDENTITY_NO_VARIANCE_PLANNING'
    elif between<0:
        fields['auxiliary_planning_status']='NEGATIVE_BETWEEN_ESTIMATE_UNRESOLVED'
    else:
        ceiling=Fraction(str(row['scalar_source_variance_ceiling']))
        b,w=Fraction(auxiliary['between_block_variance_estimate']),Fraction(auxiliary['auxiliary_within_variance_estimate'])
        fields['between_root_estimate_over_optimistic_ceiling']=between/float(ceiling)
        if b>ceiling or (b==ceiling and w>0):
            fields['auxiliary_planning_status']='NO_FINITE_AUXILIARY_COUNT_IN_POINT_SCENARIO'
        elif b==0 and w==0:
            fields['auxiliary_planning_status']='OBSERVED_ZERO_COMPONENTS_UNCERTAIN'
        else:
            count=2 if w==0 else max(2,math.ceil(w/(ceiling-b)))
            fields.update(auxiliary_planning_status='FINITE_COUNT_IN_POINT_SCENARIO_ONLY',
                          point_auxiliary_realizations_for_variance_ceiling=count)
    return fields


def diagnose_source_variance(config_path, project_root, output, budget_archive, budget_sha256,
                             precision_archive, precision_sha256, auxiliary_archive=None, auxiliary_sha256=None):
    """Consume bound completed scores; preserve all frozen scientific requirements."""
    require(not output.exists(),'FRESH_SOURCE_VARIANCE_PLANNING_OUTPUT_REQUIRED')
    config_bytes = config_path.read_bytes()
    budget_root,precision_root = project_root/budget_archive,project_root/precision_archive
    budget,precision = bound_report(budget_root,budget_sha256),bound_report(precision_root,precision_sha256)
    require(hashlib.sha256(config_bytes).hexdigest() == budget['settings_sha256']
            and yaml.safe_load(config_bytes) == budget['settings']
            and budget['numerical_margins_frozen'] is True and budget['planning_blocks'] == 128,
            'UNCHANGED_FROZEN_SOURCE_BUDGET_REQUIRED')
    require(precision['status'] == 'COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON'
            and precision['original_independent_blocks'] == 32,'COMPLETE_COUPLED_SOURCE_PRECISION_REQUIRED')
    inputs = {}
    for role,root,report,name in [('margins',budget_root,budget,'frozen_numerical_margins.csv'),
                                  ('precision',precision_root,precision,'endpoint_precision_comparison.csv')]:
        path=root/name;data=path.read_bytes();expected=report['outputs'][name]
        require(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],
                'BOUND_SOURCE_PLANNING_TABLE_CHANGED:'+name)
        inputs[role]=dict(data=data,path=str(path),bytes=len(data),sha256=expected['sha256'])
    auxiliary_report=None
    if auxiliary_archive is not None:
        auxiliary_root=project_root/auxiliary_archive
        auxiliary_report=bound_report(auxiliary_root,auxiliary_sha256)
        require(auxiliary_report['status']=='COMPLETE_DESCRIPTIVE_PFC_AUXILIARY_VARIANCE_DIAGNOSTIC'
                and auxiliary_report['row_count']==49650 and auxiliary_report['original_independent_blocks']==32
                and auxiliary_report['auxiliary_realizations']==2,'COMPLETE_ARCHIVED_AUXILIARY_MOMENTS_REQUIRED')
        name='auxiliary_variance_endpoints.csv';path=auxiliary_root/name;data=path.read_bytes()
        expected=auxiliary_report['outputs'][name]
        require(len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'],
                'BOUND_AUXILIARY_SOURCE_TABLE_CHANGED')
        inputs['auxiliary']=dict(data=data,path=str(path),bytes=len(data),sha256=expected['sha256'])
    def rows(role):
        return csv.DictReader(io.TextIOWrapper(io.BytesIO(inputs[role]['data']),encoding='utf-8',newline=''))
    definitions=Counter(row['definition_status'] for row in rows('margins'))
    require(sum(definitions.values())==84330 and definitions[UNDEFINED]==1734 and definitions[STRUCTURAL]==6618,
            'FULL_FROZEN_SOURCE_ENDPOINT_DOMAIN_REQUIRED')
    family_size=sum(count for status,count in definitions.items() if status not in (UNDEFINED,STRUCTURAL))
    output.mkdir(parents=True,exist_ok=False)
    path=output/'source_variance_planning.csv';counts=Counter();maxima={};focus=[];auxiliary_counts=Counter()
    auxiliary_rows=rows('auxiliary') if auxiliary_report else iter(())
    current_auxiliary=next(auxiliary_rows,None);matched_auxiliary=0
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=None
        for row in source_variance_plan(rows('margins'),rows('precision'),family_size):
            auxiliary=None
            if current_auxiliary is not None:
                require(int(current_auxiliary['domain_row_index'])>=row['domain_row_index'],
                        'ORDERED_AUXILIARY_SOURCE_TABLE_REQUIRED')
                if int(current_auxiliary['domain_row_index'])==row['domain_row_index']:
                    auxiliary=current_auxiliary;matched_auxiliary+=1
                    current_auxiliary=next(auxiliary_rows,None)
            row.update(auxiliary_variance_point_plan(row,auxiliary,
                auxiliary_report['numeric_identity_check'] if auxiliary_report else None))
            auxiliary_counts[row['auxiliary_planning_status']]+=1
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);counts[row['planning_status']]+=1
            if row['sample_variance_over_optimistic_ceiling'] != '':
                family=row['family'];ratio=row['sample_variance_over_optimistic_ceiling']
                if family not in maxima or ratio>maxima[family]['sample_variance_over_optimistic_ceiling']:maxima[family]=row
            if row['domain_row_index'] in (59934,60054):focus.append(row)
    require(sum(counts.values())==84330 and len(focus)==2 and current_auxiliary is None
            and (auxiliary_report is None or matched_auxiliary==49650),
            'COMPLETE_SOURCE_VARIANCE_PLANNING_REQUIRED')
    report=dict(status='COMPLETE_IDEAL_SOURCE_VARIANCE_PLANNING_NO_CERTIFICATE',
        scope='calibration_planning_for_declared_ideal_joint_source_law',domain_rows=84330,
        definition_status_counts=dict(definitions),planning_status_counts=dict(counts),
        reused_auxiliary_moment_rows=matched_auxiliary,auxiliary_planning_status_counts=dict(auxiliary_counts),
        auxiliary_moments_recomputed=False,negative_between_estimates_clipped=False,
        original_source_blocks=32,maximum_confirmation_blocks=128,source_alpha_upper_scenario=.05,
        source_alpha_allocated=False,nonstructural_defined_family_size=family_size,
        mathematical_condition='v_e+r_e*delta_e/3 <= n*delta_e**2/(2*log(2*M/alpha_H))',
        scenario=dict(source_bias=0,martingale_increment=0,inner_error=0,
                      all_existing_nominal_error_granted_to_source=.05),
        maximum_calibration_sample_variance_ratios=maxima,registered_weak_cue_focus=focus,
        sample_variances_are_population_certificates=False,planning_arithmetic='binary64 libm log; not a probability certificate',
        covariance_pseudovalues_treated_as_independent=False,
        nested_variance_scenario='between_root+within/R, conditional on equal-law independent auxiliary draws; estimates are not bounds',
        fixed_history_between_interpretation='Estimated variance of conditional means over the 32 fixed roots; population interpretation additionally requires the prescribed independent representative root law',
        finite_prng_certificate_required_for_active_ideal_route=False,
        uncertainty_method_adopted=False,source_history_confidence_certified=False,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,
        numerical_margins_recomputed=False,numerical_margins_changed=False,error_budget_allocated=False,
        new_source_simulations=0,new_trials=0,new_episodes=0,production_episodes=0,
        provenance=dict(config_path=str(config_path),config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            frozen_budget_report_sha256=budget_sha256,coupled_precision_report_sha256=precision_sha256,
            auxiliary_variance_report_sha256=auxiliary_sha256,
            inputs={role:{key:value for key,value in item.items() if key!='data'} for role,item in inputs.items()},
            implementation_sha256=sha256_file(Path(__file__))),
        outputs={path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path))})
    freeze_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-budget-archive',type=Path)
    parser.add_argument('--frozen-budget-report-sha256')
    parser.add_argument('--source-precision-archive',type=Path)
    parser.add_argument('--source-precision-report-sha256')
    parser.add_argument('--auxiliary-variance-archive',type=Path)
    parser.add_argument('--auxiliary-variance-report-sha256')
    args=parser.parse_args()
    require(bool(args.frozen_budget_archive) == bool(args.frozen_budget_report_sha256),'BOTH_FROZEN_BUDGET_BINDINGS_REQUIRED')
    require(bool(args.source_precision_archive)==bool(args.source_precision_report_sha256),
            'BOTH_SOURCE_PRECISION_BINDINGS_REQUIRED')
    require(bool(args.auxiliary_variance_archive)==bool(args.auxiliary_variance_report_sha256)
            and (args.auxiliary_variance_archive is None or args.source_precision_archive is not None),
            'AUXILIARY_MOMENTS_REQUIRE_SOURCE_PLANNING_AND_BOTH_BINDINGS')
    if args.source_precision_archive:
        require(args.frozen_budget_archive is not None,'FROZEN_BUDGET_REQUIRED_FOR_SOURCE_PLANNING')
        r=diagnose_source_variance(args.config,args.project_root,args.output,args.frozen_budget_archive,
            args.frozen_budget_report_sha256,args.source_precision_archive,args.source_precision_report_sha256,
            args.auxiliary_variance_archive,args.auxiliary_variance_report_sha256)
        print(json.dumps({key:r[key] for key in ('status','domain_rows','planning_status_counts','source_history_confidence_certified')}))
    else:
        r=(diagnose_archived_budget(args.config,args.project_root,args.output,args.frozen_budget_archive,args.frozen_budget_report_sha256)
           if args.frozen_budget_archive else run(args.config,args.project_root,args.output))
        print(json.dumps({key:r[key] for key in ('status','margin_rows','planning_status_counts','confirmation_n_selected')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

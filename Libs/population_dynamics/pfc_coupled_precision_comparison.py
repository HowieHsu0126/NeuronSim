"""Compare full fixed-bank clock precision without changing numerical margins."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_coupled_precision as precision
from . import pfc_count_histograms
from .pfc_coupled_clock_comparison import DT, LABELS, cdf_on_support, pilot_case
from .pfc_coupled_clock_execution import case_key
from .pfc_delay_analysis import PHASES, write_rows, require
from .pfc_phase_responses import CONDITIONS, METRICS
from .pfc_scale_analysis import _source_bound_file, _source_manifest, _source_pairing
from .provenance import freeze_json, sha256_file

SEEDS = list(range(95001, 95033))
SIZES = np.r_[np.full(16, 5000), 20000]
UNDEFINED = 'UNDEFINED_NO_WITHIN_PHASE_PAIRS'


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def bound_analysis(root, reference):
    path = root/reference['path']
    require(sha256_file(path/'report.json') == reference['report_sha256'], 'BOUND_ANALYSIS_REPORT_CHANGED')
    return path, json.loads((path/'report.json').read_text())


def output_file(root, report, name):
    path = root/name
    require(sha256_file(path) == report['outputs'][name]['sha256'], 'BOUND_ANALYSIS_OUTPUT_CHANGED:'+name)
    return path


def load_inputs(root, settings):
    budget, budget_report = bound_analysis(root, settings['budget'])
    margins = read_rows(output_file(budget, budget_report, 'frozen_numerical_margins.csv'))
    domain, domain_report = bound_analysis(root, settings['domain'])
    domains = read_rows(output_file(domain, domain_report, 'endpoint_domains.csv'))
    require(len(margins) == len(domains) == 84330, 'COMPLETE_FROZEN_ENDPOINT_DOMAIN_REQUIRED')
    for index, (margin, row) in enumerate(zip(margins, domains)):
        require(int(margin['domain_row_index']) == index and all(margin[key] == row[key]
                for key in ('family', 'endpoint_identity', 'definition_status')), 'FROZEN_ENDPOINT_ROW_CHANGED')
        margin['physical_maximum_count'] = row['physical_maximum_count'] if row['family'] == 'count_cdf' else ''
    comparison, report = bound_analysis(root, settings['native_comparison'])
    with np.load(output_file(comparison, report, 'block_source_responses.npz'), allow_pickle=False) as stored:
        for key, expected in [('condition_names', CONDITIONS), ('phase_names', tuple(PHASES)),
                              ('metric_names', METRICS), ('source_rng_seeds', SEEDS)]:
            require(np.array_equal(stored[key], expected), 'NATIVE_RESPONSE_AXIS_CHANGED:'+key)
        responses = {label: stored[label+'_values'].copy() for label in LABELS}
    with np.load(output_file(comparison, report, 'block_count_cdfs.npz'), allow_pickle=False) as stored:
        for key, expected in [('condition_names', CONDITIONS), ('phase_names', tuple(PHASES)),
                              ('source_rng_seeds', SEEDS), ('window_ms', [5, 50, 100]), ('field_unit_counts', SIZES)]:
            require(np.array_equal(stored[key], expected), 'NATIVE_COUNT_AXIS_CHANGED:'+key)
        support = stored['count_support'].copy()
        cdfs = {label: stored[label+'_cdf'].copy() for label in LABELS}
    fields, input_bindings = {}, []
    for label, dt in zip(LABELS, DT):
        path = root/settings['native_fields'][label]
        require(sha256_file(path/'delivery.json') == report['inputs'][label]['delivery_sha256'],
                'NATIVE_FIELD_DELIVERY_CHANGED')
        manifest = _source_manifest(json.loads((path/'delivery.json').read_text()))
        source_report = json.loads(_source_bound_file(path, manifest, 'analysis/report.json').read_text())
        require(source_report['joint_rng_blocks'] == SEEDS and source_report['dt_ms'] == dt
                and source_report['coupled_core_units'] == 100000, 'CANONICAL_NATIVE_FIELD_BANK_REQUIRED')
        file = _source_bound_file(path, manifest, 'analysis/field_fluctuations.npz')
        with np.load(file, allow_pickle=False) as stored:
            require(np.array_equal(stored['field_unit_counts'], SIZES)
                    and np.array_equal(stored['source_rng_seeds'], SEEDS), 'NATIVE_FIELD_AXIS_CHANGED')
            fields[label] = {condition: stored[condition+'_field_counts']/(SIZES[None, None, :]*.005)
                             for condition in CONDITIONS}
        input_bindings.append(dict(cohort=label,field_sha256=sha256_file(file)))
    return margins, domains, fields, responses, cdfs, support, input_bindings


def native_baseline(root, settings, output):
    margins, domains, fields, responses, cdfs, support, bindings = load_inputs(root, settings)
    rows, scores = precision.summarize_endpoints(margins, fields, responses, cdfs, support, SEEDS)
    _, covariance = bound_analysis(root, settings['native_covariance'])
    covariance_roundoff = covariance['maximum_arithmetic_roundtrip_allowance']
    maximum_error = 0.
    for row, previous in zip(rows, domains):
        if row['definition_status'] == UNDEFINED:
            require(row['signed_estimate'] == '' and previous['estimate'] == '', 'UNDEFINED_NATIVE_ESTIMATE_CHANGED')
            continue
        error = abs(row['signed_estimate']-float(previous['estimate']))
        allowance = covariance_roundoff if row['family'] == 'field_lag_covariance' else 1e-12
        require(error <= allowance, 'NATIVE_POINT_ROUNDTRIP_FAILED:'+row['domain_row_index'])
        maximum_error = max(maximum_error, error)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/'endpoint_statistics.csv', rows)
    np.savez_compressed(output/'block_scores.npz', source_rng_seeds=np.asarray(SEEDS),
                        domain_row_indices=np.arange(len(rows)), scores=scores)
    return dict(status='COMPLETE_NATIVE_PRECISION_BASELINE',row_count=len(rows),
        maximum_native_point_roundtrip_error=maximum_error,
        covariance_arithmetic_roundtrip_allowance=covariance_roundoff,
        allowance_is_scientific_margin=False, native_field_inputs=bindings,
        empirical_status_counts=dict(Counter(row['empirical_status'] for row in rows)),
        core_implementation_sha256=sha256_file(Path(precision.__file__)))


def calibrated_arrays(root, settings, native_fields, output):
    pilot_path, pilot_report = bound_analysis(root, settings['pilot_comparison'])
    require(pilot_report['selected_source_rng_seed'] == 95001
            and pilot_report['shared_input_checks'] == 6, 'COMPLETE_REUSED_FIRST_BLOCK_COMPARISON_REQUIRED')
    responses = {LABELS[0]: None, **{label: np.empty((3,32,5,8)) for label in LABELS[1:]}}
    fields = {LABELS[0]: native_fields[LABELS[0]], **{label: {c: np.empty((32,1000,17)) for c in CONDITIONS}
              for label in LABELS[1:]}}
    cdf_blocks = {label: [] for label in LABELS[1:]}
    supports, cases, pairing = [], [], []
    entries = settings['source_blocks']
    require([row['source_rng_seed'] for row in entries] == SEEDS, 'COMPLETE_CANONICAL_COUPLED_BANK_REQUIRED')
    for bi, entry in enumerate(entries):
        seed = entry['source_rng_seed']; archive = root/entry['archive']
        delivery = json.loads((archive/'delivery.json').read_text())
        expected = 'COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT' if bi == 0 else 'COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK'
        require(delivery['status'] == expected and str(delivery['job_id']) == str(entry['job']), 'INCOMPLETE_COUPLED_SOURCE_BLOCK')
        manifest = _source_manifest(delivery)
        if bi == 0:
            require(sha256_file(archive/'delivery.json') == pilot_report['pilot_delivery_sha256'], 'FIRST_BLOCK_DELIVERY_CHANGED')
            with np.load(output_file(pilot_path, pilot_report, 'phase_responses.npz'), allow_pickle=False) as stored:
                for label in LABELS[1:]: responses[label][:, 0] = stored['coupled_'+label]
            with np.load(output_file(pilot_path, pilot_report, 'pilot_count_cdfs.npz'), allow_pickle=False) as stored:
                supports.append(stored['count_support'].copy())
                for label in LABELS[1:]: cdf_blocks[label].append(stored['coupled_'+label].copy())
            for label, dt in zip(LABELS[1:], DT[1:]):
                for condition in CONDITIONS:
                    relative = 'source_cases/'+case_key(dt,condition,seed)+'/field_observations.npz'
                    with np.load(_source_bound_file(archive,manifest,relative), allow_pickle=False) as stored:
                        require(np.array_equal(stored['field_unit_counts'],SIZES), 'FIRST_BLOCK_FIELD_AXIS_CHANGED')
                        fields[label][condition][0] = stored['rates_hz']
            cases.extend(dict(source_rng_seed=seed,reused_comparison=True,**row) for row in pilot_report['pilot_cases'])
            continue
        block_support = []
        block_cdfs = {}
        for label,dt in zip(LABELS[1:],DT[1:]):
            histogram = {}; pairs = {}; registered = {}
            for ci,condition in enumerate(CONDITIONS):
                relative = Path('source_cases')/case_key(dt,condition,seed)
                rates,values,pair,info = pilot_case(archive,manifest,relative,seed,condition,dt,SIZES,histogram,ci)
                source = json.loads(_source_bound_file(archive,manifest,str(relative/'report.json')).read_text())
                require(source['refinement_seeds'] == [95411+2*(seed-95001),95412+2*(seed-95001)],
                        'REGISTERED_REFINEMENT_STREAMS_CHANGED')
                fields[label][condition][bi] = rates; responses[label][ci,bi] = values
                case = f'pfc_{seed}_{condition}'; pairs[case] = pair; registered[case] = (seed,condition)
                cases.append(dict(source_rng_seed=seed,reused_comparison=False,**info))
            checks = _source_pairing(pairs,registered,[seed])
            flags = ('exact_pre_cue_events','exact_initial_voltage','exact_final_source_rng','same_source_input_consumption')
            require(all(all(row[key] for key in flags) for row in checks),'COUPLED_CONDITION_PAIRING_FAILED')
            pairing.extend(dict(cohort=label,**row) for row in checks)
            path = output/'count_histograms'/f'{seed}_{label}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            pfc_count_histograms.write(path,histogram,conditions=CONDITIONS,seeds=[seed],phases=PHASES,
                windows_ms=[5,50,100],field_sizes=SIZES)
            with np.load(path,allow_pickle=False) as stored:
                coordinates,observations=stored['coordinates'],stored['observations']
            support=np.arange(int(coordinates[:,5].max())+1,dtype=np.int64)
            block_support.append(support)
            block_cdfs[label]=(pfc_count_histograms.cdf(coordinates,observations,support,(3,1,5,3,17))[:,0],support)
        support=np.arange(max(x[-1] for x in block_support)+1,dtype=np.int64);supports.append(support)
        for label in LABELS[1:]:cdf_blocks[label].append(cdf_on_support(*block_cdfs[label],support))
        print(json.dumps(dict(status='COUPLED_BLOCK_READOUT_COMPLETE',source_rng_seed=seed)),flush=True)
    support=np.arange(max(x[-1] for x in supports)+1,dtype=np.int64)
    cdfs={label:np.stack([cdf_on_support(x,s,support) for x,s in zip(cdf_blocks[label],supports)],axis=1)
          for label in LABELS[1:]}
    write_rows(output/'shared_input_checks.csv',pairing)
    return fields,responses,cdfs,support,cases



def source_costs(root, settings, cases):
    native = {seed: 0. for seed in SEEDS}; coarse = dict(native); seen = set()
    for entry in settings['native_source_reports']:
        key = entry['cohort'], entry['case_id']
        require(key not in seen, 'DUPLICATE_NATIVE_COST_CASE');seen.add(key)
        path = root/entry['path']
        require(sha256_file(path) == entry['report_sha256'], 'NATIVE_COST_REPORT_CHANGED')
        report = json.loads(path.read_text());seed = entry['source_rng_seed']
        require(report['case_id'] == entry['case_id'] and report['seeds']['source_rng_seed'] == seed
                and report['source_execution_complete'] is True, 'COMPLETE_BOUND_NATIVE_COST_CASE_REQUIRED')
        cost = report['simulator_cost']['wall_seconds']
        require(np.isfinite(cost) and cost > 0, 'MEASURED_SIMULATOR_WALL_TIME_REQUIRED')
        native[seed] += cost
        if entry['cohort'] == LABELS[0]:coarse[seed] += cost
    require(seen == {(label, f'pfc_{seed}_{condition}') for label in LABELS for seed in SEEDS for condition in CONDITIONS},
            'COMPLETE_NATIVE_COST_GRID_REQUIRED')
    fine = {seed:0. for seed in SEEDS};seen=set()
    for case in cases:
        seed=case['source_rng_seed'];key=seed,case['dt_ms'],case['case_id'];cost=case['cost']['wall_seconds']
        require(key not in seen and np.isfinite(cost) and cost > 0,
                'UNIQUE_MEASURED_COUPLED_COST_CASE_REQUIRED');seen.add(key)
        fine[seed] += cost
    require(seen == {(seed,dt,f'pfc_{seed}_{condition}') for seed in SEEDS for dt in DT[1:] for condition in CONDITIONS},
            'COMPLETE_COUPLED_COST_GRID_REQUIRED')
    rows=[dict(source_rng_seed=seed,native_simulator_wall_seconds=native[seed],
               coupled_simulator_wall_seconds=coarse[seed]+fine[seed],
               reused_coarse_simulator_wall_seconds=coarse[seed],coupled_fine_simulator_wall_seconds=fine[seed]) for seed in SEEDS]
    ratio=sum(row['coupled_simulator_wall_seconds'] for row in rows)/sum(native.values())
    return rows,ratio



def export_arrays(output, fields, responses, cdfs, support, scores, domain_indices):
    """Retain the named axes required by existing response/score consumers."""
    seed_axis = np.asarray(SEEDS, dtype=np.int64)
    condition_axis = np.asarray(CONDITIONS)
    phase_axis = np.asarray(list(PHASES))
    field_axis = np.arange(len(SIZES), dtype=np.int64)
    np.savez_compressed(output/'coupled_block_scores.npz', source_rng_seeds=seed_axis,
        domain_row_indices=np.asarray(domain_indices, dtype=np.int64), scores=scores)
    np.savez_compressed(output/'coupled_field_rates.npz', source_rng_seeds=seed_axis,
        field_unit_counts=SIZES, field_indices=field_axis, condition_names=condition_axis,
        time_start_ms=np.arange(1000)*precision.BIN_MS, bin_ms=precision.BIN_MS, units='Hz',
        **{label+'_'+condition:values for label,level in fields.items() for condition,values in level.items()})
    np.savez_compressed(output/'coupled_responses.npz', source_rng_seeds=seed_axis,
        condition_names=condition_axis, phase_names=phase_axis, metric_names=np.asarray(METRICS),
        phase_start_ms=np.asarray([start for start,stop in PHASES.values()]),
        phase_stop_ms=np.asarray([stop for start,stop in PHASES.values()]), units='Hz',
        **{label+'_values':values for label,values in responses.items()})
    np.savez_compressed(output/'coupled_count_cdfs.npz', source_rng_seeds=seed_axis,
        count_support=support, condition_names=condition_axis, phase_names=phase_axis,
        window_ms=[5,50,100], field_unit_counts=SIZES, field_indices=field_axis, units='probability',
        **{label+'_cdf':values for label,values in cdfs.items()})


def coupled_comparison(root, settings, output):
    margins, _, native_fields, native_responses, native_cdfs, native_support, _ = load_inputs(root, settings)
    baseline, baseline_report = bound_analysis(root, settings['native_precision_baseline'])
    require(baseline_report['status']=='COMPLETE_NATIVE_PRECISION_BASELINE'
            and baseline_report['core_implementation_sha256']==sha256_file(Path(precision.__file__)),
            'MATCHING_VERIFIED_NATIVE_STATISTICS_REQUIRED')
    native_rows=read_rows(output_file(baseline,baseline_report,'endpoint_statistics.csv'))
    output.mkdir(parents=True,exist_ok=False)
    fields,responses,cdfs,support,cases=calibrated_arrays(root,settings,native_fields,output)
    cost_rows,cost_ratio=source_costs(root,settings,cases)
    write_rows(output/'simulator_cost_comparison.csv',cost_rows)
    common=np.arange(max(native_support[-1],support[-1])+1,dtype=np.int64)
    for label in LABELS[1:]:cdfs[label]=cdf_on_support(cdfs[label],support,common)
    cdfs[LABELS[0]]=cdf_on_support(native_cdfs[LABELS[0]],native_support,common)
    responses[LABELS[0]]=native_responses[LABELS[0]]
    rows,scores=precision.summarize_endpoints(margins,fields,responses,cdfs,common,SEEDS)
    require(len(rows)==len(native_rows),'COMPLETE_MATCHED_ENDPOINTS_REQUIRED')
    comparisons=[];changes=Counter()
    extra=[key for key in rows[0] if key not in margins[0]]
    for new,old,margin in zip(rows,native_rows,margins):
        require(all(str(new[key])==old[key] for key in ('domain_row_index','family','endpoint_identity','definition_status','margin')),
                'NATIVE_COUPLED_ENDPOINT_IDENTITY_CHANGED')
        row=dict(margin)
        for prefix,values in [('native',old),('coupled',new)]:
            row.update({prefix+'_'+key:values[key] for key in extra})
        if new['definition_status']==UNDEFINED:
            row.update(variance_ratio='',variance_ratio_status='UNDEFINED',coupled_minus_native_estimate='')
        else:
            a,b=float(old['sample_variance']),new['sample_variance']
            status='DEFINED_RATIO' if a else 'BOTH_EMPIRICALLY_ZERO_UNCERTAIN' if b==0 else 'NATIVE_ZERO_COUPLED_NONZERO'
            row.update(variance_ratio=b/a if a else '',variance_ratio_status=status,
                coupled_minus_native_estimate=new['signed_estimate']-float(old['signed_estimate']))
            changes['lower' if b<a else 'higher' if b>a else 'equal']+=1
        row['observed_simulator_cost_ratio']=cost_ratio
        row['variance_simulator_cost_ratio']=row['variance_ratio']*cost_ratio if row['variance_ratio']!='' else ''
        comparisons.append(row)
    write_rows(output/'endpoint_precision_comparison.csv',comparisons)
    export_arrays(output, fields, responses, cdfs, common, scores,
                  [int(row['domain_row_index']) for row in rows])
    freeze_json(output/'case_costs.json',dict(cases=cases,source_wall_seconds_sum=sum(x['wall_seconds'] for x in cases),
        cost_scope='192fine-case wall times including input construction, source outputs and checks; native coarse and transfer costs excluded here. Not a matched end-to-end speedup.'))
    return dict(status='COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON',row_count=len(rows),
        variance_change_rows=dict(changes),observed_simulator_cost_ratio=cost_ratio,
        cost_measure=settings['cost_measure'],matched_end_to_end_speedup_qualified=False,complete_new_fine_episodes=186,reused_first_block_fine_episodes=6,
        reused_native_coarse_episodes=96,native_statistics_reused_from=settings['native_precision_baseline'],
        source_case_count=len(cases),core_implementation_sha256=sha256_file(Path(precision.__file__)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['native','coupled'],required=True)
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text())
    require(settings['schema']=='pfc_coupled_precision_comparison_v1' and settings['source_rng_seeds']==SEEDS
            and settings['confirmation_execution_released'] is False,'FIXED_CALIBRATION_COMPARISON_REQUIRED')
    started=time.perf_counter()
    report=(native_baseline if args.mode=='native' else coupled_comparison)(args.project_root,settings,args.output)
    report.update(source_simulations=0,production_episodes=0,original_independent_blocks=32,
        independent_circuit_instances=1,confirmation_execution_released=False,scientific_fidelity_qualified=False,
        phase_boundaries_ms=PHASES,
        delay_boundary_semantics='Registered nominal300ms boundary; early_delay is the frozen300-500ms observation window, not an assertion that delivered current is zero throughout. Actual current segments are retained in each bound source archive.',
        uncertainty_coverage_qualified=False,confidence_intervals=False,numerical_margins_changed=False,
        wall_seconds=time.perf_counter()-started,settings=settings,settings_sha256=sha256_file(args.config),
        interpretation='Descriptive signed estimates and block-based precision. Native and coupled share32blocks, not64. '
        'Zero empirical variance is not zero population uncertainty. Covariance scores are32delete-one pseudovalues, not496independent pairs. '
        'Complete-domain confirmation coverage and matched end-to-end cost remain separate unresolved gates.',
        outputs={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=sha256_file(p))
                 for p in args.output.rglob('*') if p.is_file()})
    freeze_json(args.output/'report.json',report)
    print(json.dumps({key:report[key] for key in ('status','row_count','wall_seconds')}));return 0


if __name__=='__main__':raise SystemExit(main())

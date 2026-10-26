"""Compare replayed LFADS causal forecasts before and after budget-only training.

Reuses the existing per-budget LFADS ensemble/covariance analysis. The four
blocks also selected epochs; all outputs are descriptive, without inferential CI/p/q.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import traceback

import numpy as np

from . import lfads_forecast_analysis as forecast
from .lfads_trajectory_evaluation import BUDGET_TRAINER_SHA256, MATCHED_PROTOCOL
from .lfads_trajectory_training import (FIXED_PROTOCOL, LEGACY_TRAINER_SHA256,
    QUALIFIED_MODEL_SHA256, _history_digest, _source_transition, _verified_run)
from .population_forecast_replay import METRIC_ATOL, METRIC_RTOL, UNIT_METRICS, _score_rows
from .provenance import sha256_file
from .trajectory_forecast_analysis import _write_rows

SEEDS = (41, 42, 43)
BLOCKS = (14, 15, 16, 17)
BUDGETS = ('original_20', 'continued_80_ceiling')
EXPECTED_UNITS = 100000
TABLES = ('forecast_all_seeds.csv', 'forecast_by_block.csv', 'forecast_summary.csv',
          'covariance_diagnostic.csv', 'covariance_summary.csv', 'costs_by_origin.csv')
CASES = tuple(dict(case_id=f'train_block_{b:03d}_familiar_{t}',block=b,template_id=f'familiar_{t}')
              for b in BLOCKS for t in range(4))
METRICS = ('energy_score', 'variogram_spatial', 'variogram_cross_population',
           'variogram_temporal_lag_1', 'variogram_temporal_lag_4', 'variogram_temporal_lag_16',
           *UNIT_METRICS, 'refractory_count_violation_1bin', 'refractory_count_violation_2bin',
           'refractory_count_violation_4bin')


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _safe_child(root, relative):
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'training map must stay inside its declared root')
    return root/p


def training_locations(original_root, continued_root, campaign_path):
    campaign=read(campaign_path)
    require(campaign['training_seeds']==list(SEEDS) and campaign['total_epoch_ceiling']==80
            and campaign['base_epoch_ceiling']==20 and set(campaign['outputs'])=={str(s) for s in SEEDS},
            'registered three-seed20-to80 budget campaign required')
    return {BUDGETS[0]:{s:original_root/f'seed_{s}' for s in SEEDS},
            BUDGETS[1]:{s:_safe_child(continued_root,campaign['outputs'][str(s)]) for s in SEEDS}}


def inventory(roots, locations):
    rows=[]
    for budget,seed in product(BUDGETS,SEEDS):
        files=dict(training=locations[budget][seed]/'run.json',
            evaluation=roots[budget]/f'seed_{seed}_forecast/evaluation.json',
            replay=roots[budget]/f'seed_{seed}_replay.json')
        row=dict(budget=budget,seed=seed)
        for role,path in files.items():
            row[role+'_path']=str(path.resolve())
            try:
                report=read(path)
                row[role+'_status']=report.get('status','MISSING_STATUS')
                row[role+'_error']=report.get('error','')
            except (FileNotFoundError,json.JSONDecodeError) as error:
                row[role+'_status']='MISSING_OR_INVALID'
                row[role+'_error']=str(error)
        row['ready']=(row['training_status']=='FITTED_DEVELOPMENT'
                      and row['evaluation_status']=='EVALUATED_DEVELOPMENT' and row['replay_status']=='PASS')
        rows.append(row)
    return rows


def _model_config(config, seed):
    require(seed in SEEDS and all(config.get(k)==v for k,v in FIXED_PROTOCOL.items()),
            'unchanged full-capacity model, source rates and frozen20epoch base config required')
    digest=hashlib.sha256(json.dumps(config['model'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(digest==QUALIFIED_MODEL_SHA256,'qualified LFADS model configuration changed')


def _training(training, evaluation, seed):
    run,latest_dir,latest,selected_dir,selected=_verified_run(training,56)
    binding=run['binding'];_model_config(binding['config'],seed)
    require(binding['seed']==seed and run['training_seed']==seed
            and evaluation['training_binding']==binding
            and evaluation['training_run_sha256']==sha256_file(training/'run.json')
            and evaluation['selected_checkpoint']==run['best_checkpoint']
            and evaluation['selected_training_sha256']==sha256_file(selected_dir/'training.json')
            and evaluation['selected_model_artifacts']==selected['model_artifacts']
            and evaluation['completed_training_epochs']==run['completed_epochs']
            and evaluation['selected_epoch']==run['best_epoch']
            and evaluation['training_stop_reason']==run['stop_reason'],
            'evaluated training, selected checkpoint or executed budget changed')
    effective=run.get('effective_maximum_epochs',20)
    require(evaluation.get('effective_training_epoch_budget',20)==effective
            and evaluation.get('training_continuation')==run.get('continuation'),
            'evaluation continuation metadata or effective budget differs')
    for name in ('formal_training_completed','generative_fidelity_qualified','convergence_established'):
        require(run.get(name) is False,'training receipt must retain unqualified scientific scope')
    return dict(run=run,latest=latest,selected=selected,latest_dir=latest_dir,selected_dir=selected_dir,
                training_root=training,effective=effective)


def _row_metrics(row):
    values={}
    bins=int(float(row['horizon_ms'])/5)
    for metric in METRICS:
        unavailable=metric.startswith('variogram_temporal_lag_') and int(metric.rsplit('_',1)[1])>=bins
        if unavailable:
            require(row[metric]=='','unavailable temporal lag must remain explicitly empty')
            values[metric]=None
        else:
            value=float(row[metric]);require(np.isfinite(value),'nonfinite required forecast metric')
            # Finite-ensemble U-statistic squared-error estimates can be negative.
            if metric!='unit_bin_mean_mse_corrected' and not metric.startswith('variogram_'):
                require(value>=0,'negative required forecast score/count')
            if metric=='unit_total_coverage_90' or metric.startswith('refractory_count_violation_'):
                require(value<=1,'forecast fraction outside0..1')
            values[metric]=value
    return values


def _verify_replay_csv(path, report, score_rows, seed):
    require(report['metric_tolerance']==dict(rtol=METRIC_RTOL,atol=METRIC_ATOL)
            and report['field_tolerance']==dict(rtol=1e-10,atol=1e-12),'frozen replay tolerances changed')
    csv_path=path.with_suffix('.csv')
    require(report['csv_artifact']==csv_path.name and report['csv_sha256']==sha256_file(csv_path),
            'independent replay comparison CSV changed')
    expected=set(product((c['case_id'] for c in CASES),MATCHED_PROTOCOL['origins_ms'],
                         MATCHED_PROTOCOL['horizons_ms'],(*UNIT_METRICS,
                         'refractory_count_violation_1bin','refractory_count_violation_2bin','refractory_count_violation_4bin')))
    checked=set()
    indices={c['case_id']:i for i,c in enumerate(CASES)}
    for row in csv_rows(csv_path):
        key=(row['case_id'],float(row['origin_ms']),float(row['horizon_ms']),row['metric'])
        require(key in expected and key not in checked,'duplicate or unexpected replay comparison row')
        checked.add(key)
        case_index=indices[key[0]];origin_index=MATCHED_PROTOCOL['origins_ms'].index(key[1])
        base=86000+seed*100000+case_index*100+origin_index*3
        reference='exact_saved_full_unit_array' if case_index==origin_index==0 else 'regenerated_counts_no_saved_full_unit_array'
        require(row['model']=='lfads' and int(row['training_seed'])==seed and row['passed']=='True'
                and int(row['block'])==CASES[case_index]['block']
                and row['template_id']==CASES[case_index]['template_id']
                and [int(row[k]) for k in ('initial_seed','dynamics_seed','observation_seed')]==[base,base+1,base+2]
                and row['unit_count_reference']==reference,
                'replay failed or case-specific initial/dynamics/observation randomness differs')
        actual=float(row['evaluator_float32']);recomputed=float(row['numpy_float64'])
        delta=float(row['absolute_difference']);allowed=float(row['allowed_difference'])
        require(all(np.isfinite([actual,recomputed,delta,allowed]))
                and actual==score_rows[key[:3]][key[3]]
                and np.isclose(delta,abs(actual-recomputed),rtol=1e-12,atol=1e-15)
                and np.isclose(allowed,METRIC_ATOL+METRIC_RTOL*abs(actual),rtol=1e-12,atol=1e-15)
                and delta<=allowed,'replay values or recorded tolerance decision changed')
    require(checked==expected,'complete1152-unit-metric replay comparisons required')


def load_forecast(root, training, seed):
    folder=root/f'seed_{seed}_forecast';evaluation=read(folder/'evaluation.json')
    replay_path=root/f'seed_{seed}_replay.json';replay=read(replay_path)
    require(evaluation['configuration']==MATCHED_PROTOCOL and evaluation['training_seed']==seed
            and evaluation['model_type']=='lfads' and evaluation['evaluation_blocks']==list(BLOCKS)
            and evaluation['independent_blocks']==4 and evaluation['episodes']==16
            and evaluation['completed_origins']==48 and evaluation['score_rows']==144
            and evaluation['all_unit_outputs_generated'] is True and evaluation['field_outputs_from_unit_counts'] is True
            and evaluation['likelihood_scores_compared'] is False,
            'complete matched48-origin144-row LFADS forecast required')
    for key in ('scientific_claim_qualified','formal_test_accessed','calibration_outcomes_accessed','convergence_established'):
        require(evaluation[key] is False,'descriptive selection-block scope changed')
    require(replay['status']=='PASS' and replay['training_seed']==seed
            and replay['verified_origins']==48 and replay['verified_score_rows']==144
            and replay['metric_comparisons']==1152 and replay['field_scores_recomputed']==720
            and replay['all_forecast_unit_metrics_replayed'] is True
            and replay['saved_full_unit_origins_compared']==1
            and replay['regenerated_origins_without_saved_full_unit_arrays']==47
            and replay['all_origin_unit_counts_compared_to_saved_full_arrays'] is False
            and replay['scientific_fidelity_qualified'] is False
            and replay['evaluation_sha256']==sha256_file(folder/'evaluation.json')
            and replay['scores_sha256']==evaluation['scores_sha256']==sha256_file(folder/'scores.csv'),
            'complete source-bound PASS replay required')
    origin_rows=replay['origins']
    origin_keys={(r['case_id'],float(r['origin_ms'])) for r in origin_rows}
    require(len(origin_rows)==48 and origin_keys==set(product((c['case_id'] for c in CASES),
            MATCHED_PROTOCOL['origins_ms'])) and all(r['exact_field_and_lfads_paths'] is True for r in origin_rows),
            'complete independently replayed origin identities required')
    training_state=_training(training,evaluation,seed)
    binding=evaluation['training_binding'];source=evaluation['native_source_identity']
    require(set(source)=={'gpu_source_revision','native_library_sha256','native_poisson_rng_family','graph_manifest_sha256','dt_ms'}
            and source['dt_ms']==.1 and source['native_poisson_rng_family']=='per_edge_philox4_32_10'
            and all(source.get(k)==v for k,v in binding['native_source_identity'].items())
            and replay['dataset_sha256']==binding['dataset_sha256']
            and evaluation['dataset_sha256']==binding['dataset_sha256']
            and evaluation['acquisition_design_sha256']==binding['acquisition_design_sha256'],
            'full evaluated native identity or its training subset changed')
    for name,digest in evaluation['artifacts'].items():
        require(Path(name).name==name and sha256_file(folder/name)==digest,
                'replayed unit/field/latent artifact changed: '+name)
    with np.load(folder/'fitting_field_scale.npz',allow_pickle=False) as saved:
        scale={k:saved[k] for k in saved.files}
    require(np.isfinite(scale['mean_hz']).all() and np.isfinite(scale['standard_deviation_hz']).all()
            and (scale['standard_deviation_hz']>0).all() and int(scale['unit_count'].sum())==EXPECTED_UNITS,
            'complete finite fitting-only field scale required')
    with np.load(folder/'full_unit_samples.npz',allow_pickle=False) as saved:
        unit_ids=saved['unit_ids']
    require(unit_ids.shape==(EXPECTED_UNITS,) and unit_ids.dtype.kind in 'iu'
            and (np.diff(unit_ids)>0).all(),'full identified unit map changed')
    record=dict(evaluation,artifacts={k:v for k,v in evaluation['artifacts'].items() if k!='fitting_field_scale.npz'})
    raw=_score_rows(folder,record,CASES)
    parsed={}
    for key,row in raw.items():
        parsed[key]=dict(seed=seed,block=int(row['block']),case_id=row['case_id'],template_id=row['template_id'],
                         origin_ms=float(row['origin_ms']),horizon_ms=float(row['horizon_ms']),**_row_metrics(row))
    _verify_replay_csv(replay_path,replay,parsed,seed)
    return dict(training_state,evaluation=evaluation,replay=replay,rows=parsed,scale=scale,unit_ids=unit_ids,
                folder=folder,replay_path=replay_path,source=source)


def _migration_pair(original, continued):
    old,new=original['evaluation']['inference_migration'],continued['evaluation']['inference_migration']
    require(old is not None and new is not None and old['receipt']==new['receipt']
            and old['receipt_sha256']==new['receipt_sha256'],'numerical inference migration evidence differs across budgets')
    receipt=old['receipt'];base=original['run']['binding']
    require(receipt['status']=='QUALIFIED_INFERENCE_ONLY'
            and receipt['from_adapter_sha256']==base['sources']['lfads.py']
            and receipt['unchanged_sources']=={k:v for k,v in base['sources'].items() if k!='lfads.py'},
            'original numerical inference migration does not match parent sources')
    bridge=new.get('trainer_bridge',{})
    require(bridge.get('checkpoint_scope')=='continued_child'
            and bridge.get('legacy_trainer_sha256')==LEGACY_TRAINER_SHA256
            and bridge.get('budget_trainer_sha256')==BUDGET_TRAINER_SHA256
            and bridge.get('saved_trainer_sha256')==BUDGET_TRAINER_SHA256
            and bridge.get('inference_trainer_sha256')==BUDGET_TRAINER_SHA256
            and bridge.get('verified_continuation_sha256')==continued['run']['continuation_sha256'],
            'continued inference requires its pinned trainer metadata bridge and verified lineage')


def pair_bindings(original, continued):
    left,right=original['run'],continued['run']
    require(original['effective']==20 and left['completed_epochs']==20 and left.get('continuation') is None
            and left['binding']['sources']['lfads_trajectory_training.py']==LEGACY_TRAINER_SHA256,
            'immutable original20epoch parent required')
    lineage=right.get('continuation')
    require(continued['effective']==80 and isinstance(lineage,dict) and lineage['total_epochs']==80
            and lineage['base_maximum_epochs']==20 and lineage['parent_effective_maximum_epochs']==20
            and lineage['parent_completed_epochs']==20 and lineage['inherited_early_stopping_patience']==4
            and lineage['parent_binding']==left['binding']
            and lineage['parent_checkpoint_sha256']==left['latest_training_sha256']
            and lineage['parent_artifact_sha256']['run.json']==sha256_file(original['training_root']/'run.json')
            and lineage['parent_history_sha256']==_history_digest(original['latest']['history'])
            and lineage['parent_best_training_sha256']==sha256_file(original['selected_dir']/'training.json')
            and lineage['parent_best_model_artifacts']==original['selected']['model_artifacts']
            and lineage['parent_latest_model_artifacts']==original['latest']['model_artifacts']
            and right['binding']['sources']['lfads_trajectory_training.py']==BUDGET_TRAINER_SHA256
            and lineage['source_migration']==_source_transition(left['binding'],right['binding']),
            'exact parent checkpoints/history/model/optimizer and frozen80epoch/patience4 lineage required')
    require(original['source']==continued['source'] and original['scale'].keys()==continued['scale'].keys()
            and all(np.array_equal(original['scale'][k],continued['scale'][k]) for k in original['scale'])
            and np.array_equal(original['unit_ids'],continued['unit_ids']),
            'matched full source, unit identities, field map and fitting scale required')
    require(original['evaluation']['unit_metrics_implementation_sha256']==continued['evaluation']['unit_metrics_implementation_sha256'],
            'numerical unit-metric implementation changed')
    require(original['rows'].keys()==continued['rows'].keys(),'matched actual case/origin/horizon grid required')
    for key,row in original['rows'].items():
        require(row['observed_spikes_per_unit']==continued['rows'][key]['observed_spikes_per_unit'],
                'native observation changes between paired budget rows')
    _migration_pair(original,continued)
    inherited=right['best_epoch']<20
    same=original['selected']['model_artifacts']==continued['selected']['model_artifacts']
    if inherited:
        require(same and right['best_epoch']==left['best_epoch'],'inherited best differs from the parent selected model')
    return dict(selected_checkpoint_inherited=inherited,selected_model_same_as_original=same,
                selected_weights_same_as_original=(original['selected']['model_artifacts']['weights.npz']
                                                  ==continued['selected']['model_artifacts']['weights.npz']))


def aggregate(root, cache, destination, records):
    cache=Path(cache) if cache is not None else root/'analysis'
    reused=(cache/'analysis.json').exists()
    if not reused:
        cache=destination
        forecast.analyze(root,cache)
    report=read(cache/'analysis.json')
    require(report['status']=='DESCRIBED_REPLAYED_DEVELOPMENT' and report['training_seeds']==list(SEEDS)
            and report['score_rows']==432 and report['forecast_origins']==144 and report['independent_blocks']==4
            and report['connectivity_instances']==1 and report['configuration']==MATCHED_PROTOCOL
            and report['scientific_claim_qualified'] is False and report['formal_test_accessed'] is False
            and report['convergence_established'] is False and report['all_unit_metrics_independently_replayed'] is True
            and report['source_binding']=={k:records[SEEDS[0]]['run']['binding'][k] for k in
                ('acquisition_design_sha256','dataset_sha256','native_source_identity')}
            and report['implementation_sha256']==sha256_file(Path(forecast.__file__)),
            'source-bound existing LFADS aggregate implementation and scope required')
    for seed,record in records.items():
        bound=report['model_seed_bindings'][str(seed)];ev=record['evaluation']
        require(bound['evaluation_sha256']==sha256_file(record['folder']/'evaluation.json')
                and bound['replay_sha256']==sha256_file(record['replay_path'])
                and bound['selected_model_artifacts']==ev['selected_model_artifacts']
                and bound['completed_training_epochs']==ev['completed_training_epochs']
                and bound['selected_epoch']==ev['selected_epoch'] and bound['artifacts']==ev['artifacts']
            and bound['training_stop_reason']==ev['training_stop_reason']
            and bound['training_config']==ev['training_binding']['config'],
                'cached analysis no longer matches its replayed forecast/checkpoint')
    tables={}
    for name in TABLES:
        require(sha256_file(cache/name)==report['artifacts'][name],'cached analysis table changed: '+name)
        tables[name]=csv_rows(cache/name)
    for row in tables['costs_by_origin.csv']:
        for name,value in row.items():
            if (name.endswith('_seconds') or name.endswith('_bytes')) and value!='':
                require(np.isfinite(float(value)) and float(value)>=0,'invalid cached origin cost')
    expected={(seed,*key) for seed,r in records.items() for key in r['rows']}
    seen=set()
    for row in tables['forecast_all_seeds.csv']:
        key=(int(row['seed']),row['case_id'],float(row['origin_ms']),float(row['horizon_ms']))
        require(key in expected and key not in seen,'cached forecast row identity changed')
        seen.add(key);actual=records[key[0]]['rows'][key[1:]]
        for metric in METRICS:
            value=None if row[metric]=='' else float(row[metric])
            require(value==actual[metric],'cached forecast metric differs from replayed CSV')
    require(seen==expected,'cached forecast rows incomplete')
    return cache,report,tables,reused


def differences(records):
    cases,blocks,summaries=[],[],[]
    for seed in SEEDS:
        for key,left in records[BUDGETS[0]][seed]['rows'].items():
            right=records[BUDGETS[1]][seed]['rows'][key]
            for metric in METRICS:
                a,b=left[metric],right[metric]
                require((a is None)==(b is None),'paired metric availability differs')
                cases.append({k:left[k] for k in ('seed','block','case_id','template_id','origin_ms','horizon_ms')} |
                    dict(metric=metric,original=a,continued=b,continued_minus_original=None if a is None else b-a))
    groups=defaultdict(list)
    for row in cases:groups[(row['seed'],row['block'],row['horizon_ms'],row['metric'])].append(row)
    for (seed,block,horizon,metric),members in sorted(groups.items()):
        require(len(members)==12,'paired block must retain every template/origin')
        blocks.append(dict(seed=seed,block=block,horizon_ms=horizon,metric=metric,registered_matched_cases=12,
            defined_matched_cases=sum(r['original'] is not None for r in members),
            **{k:forecast._mean([r[k] for r in members]) for k in ('original','continued','continued_minus_original')}))
    for horizon,metric in product(MATCHED_PROTOCOL['horizons_ms'],METRICS):
        values=[forecast._mean([r['continued_minus_original'] for r in blocks
                               if (r['seed'],r['horizon_ms'],r['metric'])==(s,horizon,metric)]) for s in SEEDS]
        mean=forecast._mean(values)
        summaries.append(dict(horizon_ms=horizon,metric=metric,mean_continued_minus_original=mean,
            minimum_seed_mean=None if mean is None else min(values),maximum_seed_mean=None if mean is None else max(values),
            training_seeds=3,epoch_selection_blocks=4,inferential_intervals=False))
    return cases,blocks,summaries


def covariance_differences(tables):
    key=lambda r:(int(r['seed']),r['template_id'],float(r['origin_ms']),float(r['horizon_ms']),r['statistic'])
    indexed={}
    for budget in BUDGETS:
        data=tables[budget]['covariance_diagnostic.csv'];index={key(r):r for r in data}
        require(len(index)==len(data),'duplicate covariance row')
        indexed[budget]=index
    require(indexed[BUDGETS[0]].keys()==indexed[BUDGETS[1]].keys(),'covariance settings differ')
    result=[]
    for identity,a in indexed[BUDGETS[0]].items():
        b=indexed[BUDGETS[1]][identity]
        require(all(a[k]==b[k] for k in ('independent_blocks','pairs','phases','observed_covariance_mean')),
                'native covariance, pair set or phase denominator differs')
        result.append(dict(seed=identity[0],template_id=identity[1],origin_ms=identity[2],horizon_ms=identity[3],
            statistic=identity[4],**{k+'_continued_minus_original':float(b[k])-float(a[k]) for k in
            ('predicted_covariance_mean','covariance_mse','covariance_mae')}))
    return result


def training_tables(records, selected):
    budgets,curves,attempts=[],[],[]
    summed={}
    for budget,seed in product(BUDGETS,SEEDS):
        record=records[budget][seed];run=record['run']
        current=[]
        count=run.get('attempt')
        if type(count) is int:
            for attempt in range(count+1):
                path=record['training_root']/'attempts'/f'attempt_{attempt:04d}.json'
                info=read(path) if path.exists() else {}
                seconds=info.get('elapsed_seconds')
                if seconds is not None:
                    require(np.isfinite(seconds) and seconds>=0,'invalid reported attempt time')
                row=dict(budget=budget,seed=seed,attempt=attempt,status=info.get('status','MISSING'),
                         reported_attempt_seconds=seconds,error=info.get('error',''),receipt_present=bool(info))
                attempts.append(row);current.append(row)
        else:
            attempts.append(dict(budget=budget,seed=seed,attempt=None,status='UNRECORDED',
                                 reported_attempt_seconds=None,error='',receipt_present=False))
        complete=bool(current) and all(r['reported_attempt_seconds'] is not None for r in current)
        summed[budget,seed]=sum(r['reported_attempt_seconds'] for r in current) if complete else None
        last_seconds=run.get('elapsed_seconds')
        if last_seconds is not None:
            require(np.isfinite(last_seconds) and last_seconds>=0,'invalid final attempt time')
        inherited=budget==BUDGETS[1]
        info=selected[seed] if inherited else dict(selected_checkpoint_inherited=False,selected_model_same_as_original=True,
                                                 selected_weights_same_as_original=True)
        budgets.append(dict(budget=budget,seed=seed,base_epoch_ceiling=20,effective_epoch_ceiling=record['effective'],
            completed_epochs=run['completed_epochs'],selected_epoch_zero_based=run['best_epoch'],
            training_updates=run['training_updates'],stale_epochs=run['stale_epochs'],stop_reason=run['stop_reason'],
            **info,reported_stage_attempt_seconds=summed[budget,seed],
            last_attempt_reported_seconds=run.get('elapsed_seconds'),training_run_sha256=sha256_file(record['training_root']/'run.json')))
        for row in record['latest']['history']:
            for partition in ('training','validation'):
                for metric,value in row[partition].items():
                    require(np.isfinite(value),'nonfinite posterior training curve')
                    curves.append(dict(budget=budget,seed=seed,epoch=row['epoch'],partition=partition,metric=metric,value=value,
                                       inherited_parent_epoch=inherited and row['epoch']<20,objective_scope='full_response_posterior'))
    for row in budgets:
        a=summed[BUDGETS[0],row['seed']];b=summed[row['budget'],row['seed']]
        row['reported_parent_plus_stage_seconds']=(b if row['budget']==BUDGETS[0]
            else None if a is None or b is None else a+b)
        row['complete_end_to_end_training_cost_measured']=False
    return budgets,curves,attempts


def evaluation_costs(records):
    result=[]
    for budget,seed in product(BUDGETS,SEEDS):
        ev=records[budget][seed]['evaluation'];graphs=ev['graph_lifetimes']
        require(len(graphs)==3 and sorted(g['origin_ms'] for g in graphs)==MATCHED_PROTOCOL['origins_ms']
                and all(g['completed_cases']==16 for g in graphs),'three complete origin graphs required for cost accounting')
        for seconds in (ev['elapsed_seconds'],ev['verification_and_scale_seconds'],
                        *(g['setup_seconds'] for g in graphs)):
            require(isinstance(seconds,(int,float)) and np.isfinite(seconds) and seconds>=0,
                    'finite nonnegative evaluation and unique graph costs required')
        peak=ev['peak_tensorflow_gpu_allocated_bytes']
        require(peak is None or (type(peak) is int and peak>=0),'invalid measured TensorFlow peak')
        result.append(dict(budget=budget,seed=seed,evaluation_elapsed_seconds=ev['elapsed_seconds'],
            verification_and_scale_seconds=ev['verification_and_scale_seconds'],
            unique_graph_setup_seconds=sum(g['setup_seconds'] for g in graphs),
            peak_tensorflow_gpu_allocated_bytes=ev['peak_tensorflow_gpu_allocated_bytes'],
            gpu_memory_measurement=ev['gpu_memory_measurement'],
            graph_execution_devices=json.dumps([g['execution_device'] for g in graphs],sort_keys=True)))
    return result


def analyze(original, continued, original_training, continued_training, campaign, output,
            *, original_analysis=None, continued_analysis=None):
    paths=map(Path,(original,continued,original_training,continued_training,campaign,output))
    original,continued,original_training,continued_training,campaign,output=paths
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='CHECKING_INPUTS',scientific_claim_qualified=False,convergence_established=False,
                formal_test_accessed=False,confidence_intervals=False,p_values=False,q_values=False)
    try:
        roots=dict(zip(BUDGETS,(original,continued)))
        locations=training_locations(original_training,continued_training,campaign)
        status_rows=inventory(roots,locations);_write_rows(output/'input_status.csv',status_rows)
        if not all(row['ready'] for row in status_rows):
            result.update(status='BLOCKED_INCOMPLETE_INPUTS',ready_seed_budget_pairs=sum(r['ready'] for r in status_rows),
                          required_seed_budget_pairs=6,input_status=status_rows)
            write_json(output/'analysis.json',result)
            raise ValueError('all three seeds at both budgets require fitted runs and complete PASS replays')
        records={b:{s:load_forecast(roots[b],locations[b][s],s) for s in SEEDS} for b in BUDGETS}
        selected={s:pair_bindings(records[BUDGETS[0]][s],records[BUDGETS[1]][s]) for s in SEEDS}
        common=records[BUDGETS[0]][SEEDS[0]]
        for budget,seed in product(BUDGETS,SEEDS):
            item=records[budget][seed]
            require(item['source']==common['source'] and item['scale'].keys()==common['scale'].keys()
                    and all(np.array_equal(item['scale'][k],common['scale'][k]) for k in common['scale'])
                    and np.array_equal(item['unit_ids'],common['unit_ids']),
                    'cohort/source/observation map differs across seed/budget')
        aggregate_info={};tables={}
        for budget,cache in zip(BUDGETS,(original_analysis,continued_analysis)):
            path,report,data,reused=aggregate(roots[budget],cache,output/(budget+'_analysis'),records[budget])
            tables[budget]=data;aggregate_info[budget]=dict(path=str(path.resolve()),reused=reused,
                analysis_sha256=sha256_file(path/'analysis.json'),consumed_tables={n:report['artifacts'][n] for n in TABLES})
        case_pairs,block_pairs,paired_summary=differences(records)
        train_budget,curves,attempts=training_tables(records,selected)
        outputs={'paired_differences_by_case.csv':case_pairs,'paired_differences_by_block.csv':block_pairs,
                 'paired_summary.csv':paired_summary,'paired_covariance.csv':covariance_differences(tables),
                 'training_budget.csv':train_budget,'posterior_objective_curves.csv':curves,'training_attempts.csv':attempts,
                 'evaluation_costs_by_seed.csv':evaluation_costs(records)}
        for name in TABLES:
            outputs[name]=[dict(budget=b,**row) for b in BUDGETS for row in tables[b][name]]
        by_seed=[]
        for budget,seed,horizon in product(BUDGETS,SEEDS,MATCHED_PROTOCOL['horizons_ms']):
            members=[r for r in records[budget][seed]['rows'].values() if r['horizon_ms']==horizon]
            by_seed.append(dict(budget=budget,seed=seed,horizon_ms=horizon,registered_case_origins=48,
                                **{m:forecast._mean([r[m] for r in members]) for m in METRICS}))
        outputs['forecast_by_seed.csv']=by_seed
        for name,data in outputs.items():_write_rows(output/name,data)
        result.update(status='DESCRIBED_REPLAYED_BUDGET_COMPARISON',training_seeds=list(SEEDS),budgets=list(BUDGETS),
            score_rows=864,forecast_origins=288,epoch_selection_blocks=4,connectivity_instances=1,units=EXPECTED_UNITS,
            complete_seed_budget_pairs=6,actual_training_budgets_matched=False,
            inherited_selected_checkpoints={str(s):selected[s] for s in SEEDS},aggregate_sources=aggregate_info,
            budget_campaign_sha256=sha256_file(campaign),configuration=MATCHED_PROTOCOL,
            full_native_source_identity=common['source'],implementation_sha256=sha256_file(Path(__file__)),
            randomness_scope='same declared initial/dynamics/observation seed schedule; rate-dependent Poisson consumption need not align underlying draws',
            covariance_scope='existing LFADS ensemble-based total covariance across four selection blocks; no identified conditional covariance claim',
            objective_scope='posterior training/selection curves are separate from causal forecast performance',
            cost_scope='actual epoch/stop budgets differ; stage attempt clocks exclude preflight/import and external qualification; missing clocks remain unavailable; graph setup counted once per graph',
            cache_scope='reuse validates consumed table hashes and exact evaluation/replay/checkpoint bindings; unused cached covariance arrays are not reread',
            unit_replay_scope='six saved first-origin full-unit comparisons;282other origins use regenerated unit metrics and exact saved fields/latent paths',
            artifacts={n:sha256_file(output/n) for n in ['input_status.csv',*outputs]})
        write_json(output/'analysis.json',result)
        return result
    except BaseException as error:
        if result['status']!='BLOCKED_INCOMPLETE_INPUTS':
            result.update(status='FAILED',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
            write_json(output/'analysis.json',result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','continued','original-training','continued-training','campaign','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--original-analysis',type=Path)
    parser.add_argument('--continued-analysis',type=Path)
    args=parser.parse_args()
    report=analyze(args.original,args.continued,args.original_training,args.continued_training,args.campaign,args.output,
                   original_analysis=args.original_analysis,continued_analysis=args.continued_analysis)
    print(json.dumps({k:report[k] for k in ('status','score_rows','forecast_origins','scientific_claim_qualified')}))


if __name__=='__main__':main()

